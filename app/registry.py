"""Model store: a disk-backed bundle directory plus a small LRU in-memory cache,
with atomic writes, two-lock concurrency, and safe zip import/export.

Bundle (de)serialization and skops load-safety live in ``model_io``; this module
orchestrates *where* and *when* those run (cache, disk locks, atomic publish).
"""

from __future__ import annotations

import json
import os
import shutil
import threading
from collections import OrderedDict
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO

from . import model_archive, model_report
from .classifier import ClassifierModel
from .model_io import CARD_FILE, MANIFEST_FILE, UnsafeModelError, _read_bundle, _write_bundle
from .settings import get_settings

# Suffix for the untouched copy scripts/prune_bundle_labels.py keeps before it
# repairs a bundle; that script imports this constant, so the two cannot drift.
_BACKUP_SUFFIX = ".prebackup"


# `Registry.list` shadows the builtin inside the class body, so any annotation after
# that method has to name the builtin through this alias.
_Rows = list[dict]


class Registry:
    """Disk-backed model store with a bounded LRU memory cache (low RAM)."""

    def __init__(self, models_dir: str | Path, max_in_memory: int = 2) -> None:
        self.dir = Path(models_dir)
        self.max = max(1, max_in_memory)
        self._cache: OrderedDict[str, ClassifierModel] = OrderedDict()
        # Two locks. `_lock` guards only the in-memory LRU dict (fast, no I/O);
        # `_disk_lock` serialises every bundle publish/read/delete so a reader can
        # never observe a bundle mid-rmtree/mid-replace (a disk TOCTOU). Ordering
        # is one-directional: save(), the get() miss path and delete() acquire
        # `_lock` while holding `_disk_lock` (so disk state and cache state change
        # atomically together — the cache can never outlive the disk), and NO path
        # acquires `_disk_lock` while holding `_lock` — no ordering cycle/deadlock.
        self._lock = threading.Lock()
        self._disk_lock = threading.Lock()

    def _path(self, name: str) -> Path:
        return self.dir / name

    def exists(self, name: str) -> bool:
        return (self._path(name) / "config.json").exists()

    def list(self) -> list[str]:
        if not self.dir.exists():
            return []
        # Hidden ".name.tmp" dirs are in-progress/crashed writes, never models.
        # "<name>.prebackup" copies (kept by scripts/prune_bundle_labels.py before it
        # repairs a bundle) are valid bundles but not models: serving one would hand
        # out exactly the pre-repair weights the repair existed to remove.
        return sorted(
            p.name for p in self.dir.iterdir()
            if not p.name.startswith(".")
            and not p.name.endswith(_BACKUP_SUFFIX)
            and (p / "config.json").exists()
        )

    def sweep_stale_tmp(self) -> int:
        """Delete hidden ``.*.tmp`` staging left by a crashed/killed save or export.

        A save leaves a directory (the atomic rename never ran); an export leaves a
        file (the download never finished). Both are invisible to list()/exists()
        already, but they leak disk until the same name is retrained. Called on
        startup; returns how many were removed."""
        if not self.dir.exists():
            return 0
        removed = 0
        with self._disk_lock:
            for path in self.dir.iterdir():
                if not (path.name.startswith(".") and path.name.endswith(".tmp")):
                    continue
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    # An export staged here whose download never completed.
                    path.unlink(missing_ok=True)
                if not path.exists():  # ignore_errors can leave it; count only real removals
                    removed += 1
        return removed

    def _tmp_path(self, name: str) -> Path:
        """Hidden staging dir for atomic writes (safe_name rejects leading dots,
        so a real model can never collide with it)."""
        return self.dir / f".{name}.tmp"

    def _evict(self) -> None:
        while len(self._cache) > self.max:
            self._cache.popitem(last=False)

    def in_memory_count(self) -> int:
        """Number of models currently resident in the LRU cache (RAM signal)."""
        with self._lock:
            return len(self._cache)

    def save(
        self,
        name: str,
        model: ClassifierModel,
        metadata: dict,
        on_step: Callable[[str], None] = lambda _msg: None,
        *,
        overwrite: bool = False,
    ) -> None:
        """Write the bundle atomically: stage in a hidden tmp dir, then rename.

        A crash mid-write can only leave a hidden ``.name.tmp`` dir behind
        (invisible to list()/exists()) — never a half-readable model.

        An existing bundle is replaced only with ``overwrite=True`` (the
        label-pruning repair script); otherwise ``FileExistsError``, checked
        before staging and again under the disk lock. /train refuses existing
        names at SUBMIT time, which does not cover a model imported while the
        run waited in the queue — trusting it let a queued run destroy that
        import. Replacing is not atomic (rmtree, then rename).

        ``on_step`` is called with a short description before each sub-step; the
        training job routes it into progress updates so even a very slow save
        keeps emitting a liveness heartbeat.
        """
        if not overwrite and self.exists(name):
            raise FileExistsError(name)
        tmp = self._tmp_path(name)
        # Stage the (possibly multi-minute) skops dump WITHOUT the disk lock: the
        # tmp dir is uniquely named and invisible to readers (list()/exists() skip
        # dot-dirs), so a concurrent read of ANOTHER model is not blocked by it.
        # Only one training runs at a time, so no concurrent save writes this tmp.
        if tmp.exists():
            shutil.rmtree(tmp)  # leftover from a previous crash
        _write_bundle(tmp, model, metadata, on_step)
        with self._disk_lock:
            target = self._path(name)
            if target.exists():
                if not overwrite:  # appeared while we staged
                    shutil.rmtree(tmp, ignore_errors=True)
                    raise FileExistsError(name)
                shutil.rmtree(target)
            # Stepping again after the dumps marks them finished — otherwise a
            # stall here would be indistinguishable from one inside the last dump.
            on_step("Publishing bundle (atomic rename)")
            os.replace(tmp, target)
            # Publish to the cache while STILL holding _disk_lock: a concurrent
            # delete() takes _disk_lock for its rmtree, so it can no longer land
            # between the rename and the cache insert and leave the model cached
            # but absent on disk. (Nesting is one-directional — no path takes
            # _disk_lock while holding _lock — so there is no ordering cycle.)
            with self._lock:
                self._cache[name] = model
                self._cache.move_to_end(name)
                self._evict()

    def load_fresh(self, name: str) -> tuple[ClassifierModel, dict]:
        with self._disk_lock:
            if not self.exists(name):
                raise FileNotFoundError(name)
            return _read_bundle(self._path(name))

    def get(self, name: str) -> ClassifierModel:
        """Return a cached model (LRU), loading from disk on miss."""
        with self._lock:
            if name in self._cache:
                self._cache.move_to_end(name)
                return self._cache[name]
        # Miss path: read AND publish to the cache while holding _disk_lock, so a
        # concurrent delete() (which pops the cache, then rmtrees under _disk_lock)
        # can never land between our read and our insert — otherwise a deleted
        # model would stay cached and a same-name re-import would serve the OLD
        # weights. Nesting _lock inside _disk_lock is the documented legal order
        # (save() does the same); the reverse never happens.
        with self._disk_lock:
            if not self.exists(name):
                raise FileNotFoundError(name)
            model, _ = _read_bundle(self._path(name))
            with self._lock:
                self._cache[name] = model
                self._cache.move_to_end(name)
                self._evict()
        return model

    def _documents(self, name: str) -> tuple[dict, dict]:
        """Both of a bundle's JSON documents, under ONE lock hold.

        The two reports below are answers about the same bundle at the same moment;
        reading the config once per report let a concurrent delete or overwrite land
        between the halves of one answer.
        """
        with self._disk_lock:
            if not self.exists(name):
                raise FileNotFoundError(name)
            return model_report.read_documents(self._path(name))

    def info(self, name: str) -> dict:
        """Cheap metadata read (config + metrics) without loading the head."""
        return model_report.describe(name, *self._documents(name))

    def label_diagnostics(self, name: str) -> _Rows:
        """Per label: its F1, its row count and the threshold serving applies, weakest first."""
        return model_report.label_diagnostics(*self._documents(name))

    def update_info(self, name: str, info: dict) -> dict:
        """Replace the author-supplied ``info`` block of an existing bundle.

        Documentation is presentation-only data — nothing in the serving path reads
        it, and the cached model object does not carry it — so correcting an author
        name or a license must not cost a retrain. Written the same way the share
        store writes: tmp file plus rename, under the disk lock, so a crash mid-write
        cannot leave a metrics.json that no longer parses.

        An empty mapping removes the block. Returns what is now stored.
        """
        def edit(metadata: dict) -> None:
            if info:
                metadata["info"] = info
            else:
                metadata.pop("info", None)

        self._edit_metadata(name, edit)
        return info

    def append_evaluation(self, name: str, record: dict) -> None:
        """Add one evaluation result to the bundle, beside its training metrics.

        Beside, never over: the training metrics describe the run that produced the
        model and are the bundle's own account of itself. Appended rather than replaced
        because a model is scored on several datasets over its life, and keeping only
        the newest would throw away exactly the comparison this exists for.
        """
        def edit(metadata: dict) -> None:
            metadata.setdefault("evaluations", []).append(record)

        self._edit_metadata(name, edit)

    def _edit_metadata(self, name: str, edit: Callable[[dict], None]) -> None:
        """Read metrics.json, apply ``edit`` to it, write it back atomically.

        Tmp file plus rename under the disk lock, the same discipline the bundle publish
        and the share store use: a crash mid-write cannot leave a metrics.json that no
        longer parses. Shared by every editor of that document so there is one place
        where that guarantee lives.
        """
        with self._disk_lock:
            if not self.exists(name):
                raise FileNotFoundError(name)
            path = self._path(name) / "metrics.json"
            metadata = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            edit(metadata)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, path)

    def delete(self, name: str) -> None:
        # Pop the cache INSIDE the disk-lock section, AFTER the rmtree: popping
        # first (outside) let a concurrent cold get()/save() insert the model
        # again while our rmtree waited for the lock — cache outliving disk.
        # With this ordering, any insert either happens before us (we remove it)
        # or after we release (its exists() check then fails -> no insert).
        with self._disk_lock:
            if not self.exists(name):
                raise FileNotFoundError(name)
            shutil.rmtree(self._path(name))
            with self._lock:
                # Every spelling, not just this one: on a case-insensitive
                # filesystem get("M") loads the bundle "m" and caches it under
                # "M", and that key kept serving the deleted weights -- after a
                # retrain of "m", the OLD model. On a case-sensitive one this may
                # also drop a different model's entry, which only costs a reload.
                key = name.casefold()
                for cached in [k for k in self._cache if k.casefold() == key]:
                    del self._cache[cached]

    def export_to(self, name: str, target: BinaryIO) -> None:
        """Write the bundle's archive into ``target`` (card + manifest added by ``model_archive``).

        The transport artifacts are never read from disk: they are regenerated per
        export, so a stale copy could otherwise ship beside the fresh one.

        What gets packed is the same allowlist ``model_archive.unpack`` enforces on the
        way back in, minus those two. A denylist could not hold: ``update_info`` stages
        a ``metrics.json.tmp`` next to the file it replaces, so a crash in that window
        leaves one behind — and we would have produced an archive we then refuse.

        Compression now happens under the disk lock, where before only the read did.
        That is a few seconds of extra hold on a 50 MB bundle, and it is the price of
        the members being read one at a time from files that must still exist: the
        alternative, holding open handles outside the lock, would make ``delete``
        fail outright on Windows.
        """
        packable = model_archive.ALLOWED_MEMBERS - {MANIFEST_FILE, CARD_FILE}
        with self._disk_lock:
            if not self.exists(name):
                raise FileNotFoundError(name)
            sources = {
                file.name: file
                for file in sorted(self._path(name).iterdir())
                if file.is_file() and file.name in packable
            }
            model_archive.pack_into(target, name, sources)

    def import_zip(self, name: str, data: bytes) -> dict:
        """Validate an uploaded archive (``model_archive.unpack``) and install it."""
        if self.exists(name):
            raise FileExistsError(name)
        payloads = model_archive.unpack(data)
        # Stage + validate in a hidden tmp dir; publish only complete bundles.
        # Under the disk lock so a concurrent load/save/delete can never observe
        # the tmp dir or the exists()->replace window mid-flight.
        with self._disk_lock:
            tmp = self._tmp_path(name)
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir(parents=True)
            try:
                for member, payload in payloads.items():
                    (tmp / member).write_bytes(payload)
                _read_bundle(tmp)  # validates config + skops safety; raises if unsafe
            except Exception as exc:
                shutil.rmtree(tmp, ignore_errors=True)
                if isinstance(exc, KeyError):
                    # Malformed config = invalid input (400), not a server fault.
                    # Archive-level damage is already mapped by model_archive.unpack,
                    # which runs before this block; what can still fail here is
                    # writing the members and _read_bundle's own validation.
                    raise UnsafeModelError(f"Invalid model bundle: {exc!r}") from exc
                raise
            try:
                os.replace(tmp, self._path(name))
            except OSError:
                shutil.rmtree(tmp, ignore_errors=True)
                if self.exists(name):
                    # Lost a same-name import race; report it as the usual conflict.
                    raise FileExistsError(name) from None
                raise
        return self.info(name)  # outside the disk lock: info() re-acquires it sequentially


@lru_cache
def get_registry() -> Registry:
    settings = get_settings()
    return Registry(settings.models_dir, settings.effective_max_models_in_memory())

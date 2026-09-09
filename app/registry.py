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

from . import model_archive
from .classifier import ClassifierModel
from .data import label_vocabulary
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
        """Delete hidden ``.name.tmp`` staging dirs orphaned by a crashed/killed
        save (the atomic rename never ran, so list()/exists() already ignore
        them, but they leak disk until the same name is retrained). Called on
        startup; returns how many were removed."""
        if not self.dir.exists():
            return 0
        removed = 0
        with self._disk_lock:
            for path in self.dir.iterdir():
                if path.is_dir() and path.name.startswith(".") and path.name.endswith(".tmp"):
                    shutil.rmtree(path, ignore_errors=True)
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
    ) -> None:
        """Write the bundle atomically: stage in a hidden tmp dir, then rename.

        A crash mid-write can only leave a hidden ``.name.tmp`` dir behind
        (invisible to list()/exists()) — never a half-readable model. Overwriting
        an existing name is NOT atomic (rmtree, then rename); acceptable because
        /train refuses existing names — revisit if direct overwrite callers appear.

        ``on_step`` is called with a short description before each sub-step; the
        training job routes it into progress updates so even a very slow save
        keeps emitting a liveness heartbeat.
        """
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

    def info(self, name: str) -> dict:
        """Cheap metadata read (config + metrics) without loading the head."""
        with self._disk_lock:
            if not self.exists(name):
                raise FileNotFoundError(name)
            config = json.loads((self._path(name) / "config.json").read_text(encoding="utf-8"))
            metrics_path = self._path(name) / "metrics.json"
            metadata = (
                json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
            )
        classes = config.get("classes") or []
        config.pop("uri_to_label", None)  # potentially large; not needed for info
        # Derived, never stored: it follows from the labels, so it stays correct for
        # bundles trained before this existed and cannot be typed in wrong.
        vocabulary = label_vocabulary(classes)
        return {"name": name, **config, "label_vocabulary": vocabulary, "metadata": metadata}

    def label_diagnostics(self, name: str) -> _Rows:
        """Per label: its F1, how many rows carry it, and the threshold serving applies.

        Weakest first — the end anyone reviewing a model looks at. A model's headline
        F1 says how good it is on average; this says *where* it is weak, which is what
        decides whether a given answer deserves a second look.

        ``f1`` and ``support`` are ``None`` for bundles trained before they were
        recorded. ``threshold`` is ``None`` for binary/multiclass, where serving picks
        the argmax and never reads a threshold — reporting one would describe a rule
        the model does not apply.
        """
        info = self.info(name)
        metrics = (info.get("metadata") or {}).get("metrics") or {}
        scores = metrics.get("per_label_f1") or {}
        support = (info.get("metadata") or {}).get("per_label_support") or {}
        thresholds = info.get("per_label_thresholds") or {}
        names = self._uri_to_label(name)
        single_label = info.get("task_type") in ("binary", "multiclass")

        entries = [
            {
                "uri": uri,
                "label": names.get(uri, uri),
                "f1": scores.get(uri),
                "support": support.get(uri),
                "threshold": None if single_label else thresholds.get(uri, info.get("global_threshold")),
            }
            for uri in info.get("classes") or []
        ]
        # Unscored labels last: unknown is not the same as weak.
        return sorted(entries, key=lambda e: (e["f1"] is None, e["f1"] or 0.0))

    def _uri_to_label(self, name: str) -> dict:
        """The display-name map, which ``info()`` drops because it can be large."""
        with self._disk_lock:
            config = json.loads((self._path(name) / "config.json").read_text(encoding="utf-8"))
        return config.get("uri_to_label") or {}

    def update_info(self, name: str, info: dict) -> dict:
        """Replace the author-supplied ``info`` block of an existing bundle.

        Documentation is presentation-only data — nothing in the serving path reads
        it, and the cached model object does not carry it — so correcting an author
        name or a license must not cost a retrain. Written the same way the share
        store writes: tmp file plus rename, under the disk lock, so a crash mid-write
        cannot leave a metrics.json that no longer parses.

        An empty mapping removes the block. Returns what is now stored.
        """
        with self._disk_lock:
            if not self.exists(name):
                raise FileNotFoundError(name)
            path = self._path(name) / "metrics.json"
            metadata = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            if info:
                metadata["info"] = info
            else:
                metadata.pop("info", None)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, path)
        return info

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
                self._cache.pop(name, None)

    def export_zip(self, name: str) -> bytes:
        """Read a bundle from disk and pack it (card + manifest added by ``model_archive``).

        The transport artifacts are never read from disk: they are regenerated per
        export, so a stale copy could otherwise ship beside the fresh one.
        """
        with self._disk_lock:
            if not self.exists(name):
                raise FileNotFoundError(name)
            members = {
                file.name: file.read_bytes()
                for file in sorted(self._path(name).iterdir())
                if file.is_file() and file.name not in (MANIFEST_FILE, CARD_FILE)
            }
        return model_archive.pack(name, members)

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
                    # BadZipFile no longer reaches here: the archive is fully read
                    # and validated by model_archive.unpack before this point.
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

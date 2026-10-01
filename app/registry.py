"""Model store: a disk-backed bundle directory plus a small LRU in-memory cache,
with atomic writes, two-lock concurrency, and safe zip import/export.

Bundle (de)serialization and skops load-safety live in ``model_io``; this module
orchestrates *where* and *when* those run (cache, disk locks, atomic publish).

Past the project's ~300-line guide and kept whole (audit 2026-09-27, M-2). The archive
methods (``stage_export``, ``export_to``, ``import_zip``, ``import_archive``) are the obvious
seam, but they publish through the same ``_disk_lock`` and the same stage/publish pair as
``save``. Moving them out would hand that lock to a second module, and the invariant that
keeps a half-written bundle unpublishable would then live in two places.
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import shutil
import tempfile
import threading
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO

from . import durability, model_archive, model_report
from .classifier import ClassifierModel
from .manifest import CARD_FILE, MANIFEST_FILE
from .model_io import (
    UnsafeModelError,
    _read_bundle,
    _write_bundle,
    check_loadable,
)
from .settings import get_settings
from .staged_archives import StagedArchives

logger = logging.getLogger("api_v3.registry")

# How many evaluation records a bundle keeps, newest last: enough to compare a model across
# datasets, which is their point -- appended for good, they grew the document every model
# detail reads and every export packs without end (audit 2026-09-30, R15).
MAX_EVALUATIONS = 50

# Suffix for the untouched copy scripts/prune_bundle_labels.py keeps before it
# repairs a bundle; that script imports this constant, so the two cannot drift.
BACKUP_SUFFIX = ".prebackup"


# A bundle renamed aside while `publish` replaces it; see `sweep_stale_tmp`.
_REPLACED = re.compile(r"^\.(?P<name>.+)\.replaced-[0-9a-f]+\.tmp$")


def is_backup_name(name: str) -> bool:
    """Is ``name`` a repair backup's -- never a model, to any route?

    The label repair keeps the untouched bundle as ``<name>.prebackup``; serving it hands out
    exactly the weights the repair removed. The listing hid such names, every other route
    served them, and a model trained or imported under one was invisible and live at once
    (audit 2026-09-30, R15).
    """
    return name.endswith(BACKUP_SUFFIX)


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
        # Taken only while `_disk_lock` is held, never the other way round (see stage_export).
        self._staged = StagedArchives(self.dir)
        # Names an import holds from before its upload until it is installed (`importing`).
        self._importing: set[str] = set()
        # What each cached model was loaded from (`_stamp`), checked on every get().
        self._stamps: dict[str, tuple[int, int] | None] = {}

    def _path(self, name: str) -> Path:
        return self.dir / name

    def exists(self, name: str) -> bool:
        return not is_backup_name(name) and (self._path(name) / "config.json").exists()

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
            and not is_backup_name(p.name)
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
            for path in list(self.dir.iterdir()):
                if not (path.name.startswith(".") and path.name.endswith(".tmp")):
                    continue
                replaced = _REPLACED.match(path.name)
                if replaced and path.is_dir() and not self._path(replaced["name"]).exists():
                    # A replace killed between its two renames (`publish`): the old bundle
                    # goes back instead of out with the rest (audit 2026-09-30, W04).
                    os.replace(path, self._path(replaced["name"]))
                    logger.warning("Restored model %r from a replace that was cut short",
                                   replaced["name"])
                    continue
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    # An export staged here whose download never completed.
                    path.unlink(missing_ok=True)
                if not path.exists():  # ignore_errors can leave it; count only real removals
                    removed += 1
        return removed

    def new_staging(self, name: str) -> Path:
        """A fresh hidden directory to write one bundle of ``name`` into -- its own, whatever
        else is staging the same name.

        One per operation (audit 2026-09-30, R01): a training and an import of one name shared
        `.{name}.tmp`, and each removed what it found there as a crash leftover, so a failing
        import took a training's finished bundle with it and two that overlapped published a
        mix. A leading dot (which ``safe_name`` refuses, so no model collides) and a `.tmp`
        end: invisible to list()/exists(), and the startup sweep's if nothing publishes it.
        """
        self.dir.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=f".{name}.", suffix=".tmp", dir=self.dir))

    def _evict(self) -> None:
        while len(self._cache) > self.max:
            evicted, _ = self._cache.popitem(last=False)
            self._stamps.pop(evicted, None)

    def _stamp(self, name: str) -> tuple[int, int] | None:
        """What says a bundle on disk changed: its config.json's mtime and size, or None when
        it is gone. Every rewrite of a bundle rewrites config.json -- a publish, the label
        repairs -- and the repairs run beside the server, which kept serving the model it had
        cached until a restart (audit 2026-09-30, R15). One stat per get()."""
        try:
            stat = (self._path(name) / "config.json").stat()
        except OSError:
            return None
        return (stat.st_mtime_ns, stat.st_size)

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
        import. Replacing renames the old bundle aside first (see ``publish``).

        ``on_step`` is called with a short description before each sub-step; the
        training job routes it into progress updates so even a very slow save
        keeps emitting a liveness heartbeat.
        """
        staged = self.stage(name, model, metadata, on_step, overwrite=overwrite)
        self.publish(name, staged, model=model, overwrite=overwrite, on_step=on_step)

    def stage(
        self,
        name: str,
        model: ClassifierModel,
        metadata: dict,
        on_step: Callable[[str], None] = lambda _msg: None,
        *,
        overwrite: bool = False,
        into: Path | None = None,
    ) -> Path:
        """Write the bundle into a hidden staging dir — invisible until :meth:`publish` — and
        return it: ``into`` (one :meth:`new_staging` made), or a new one.

        Separate from publishing so a training in a CHILD process can stage while the
        API process publishes under its own disk lock: a second Registry in the child
        would bring a second lock, and bypass the one this registry serialises on. The
        parent makes the directory and hands it over, so it can discard it whatever
        becomes of the child.
        """
        if not overwrite and self.exists(name):
            raise FileExistsError(name)
        tmp = into if into is not None else self.new_staging(name)
        # Stage the (possibly multi-minute) skops dump WITHOUT the disk lock: the
        # tmp dir is this operation's own and invisible to readers (list()/exists()
        # skip dot-dirs), so nothing else reads or writes it meanwhile.
        _write_bundle(tmp, model, metadata, on_step)
        try:
            # Before anything can publish it: a bundle the loader would refuse must not
            # become a model (audit 2026-09-30, T01).
            check_loadable(tmp)
        except UnsafeModelError:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        # On disk before a rename can expose it, and here rather than in publish: outside
        # the disk lock, and in the child process that wrote it (audit 2026-09-30, R13).
        durability.sync_tree(tmp)
        return tmp

    def publish(
        self,
        name: str,
        staged: Path,
        *,
        model: ClassifierModel | None = None,
        overwrite: bool = False,
        on_step: Callable[[str], None] = lambda _msg: None,
    ) -> None:
        """Move the bundle staged in ``staged`` into place atomically, under the disk lock.

        ``model`` goes straight into the cache when this process has it; a bundle
        staged by another process is loaded on first use instead.

        :raises FileExistsError: if the name was taken meanwhile (and ``overwrite`` is
            off); the staged bundle is removed rather than left behind.
        """
        with self._disk_lock:
            target = self._path(name)
            aside: Path | None = None
            if target.exists():
                if not overwrite:  # appeared while we staged
                    shutil.rmtree(staged, ignore_errors=True)
                    raise FileExistsError(name)
                # Renamed aside, not removed: rmtree-then-rename left a window with no model,
                # and an error or a kill inside it lost the model for good -- the next start
                # swept the staged copy too (audit 2026-09-30, W04). An error between the two
                # renames puts the old bundle back below; a kill, `sweep_stale_tmp` at the
                # next start.
                aside = self.dir / f".{name}.replaced-{secrets.token_hex(4)}.tmp"
                os.replace(target, aside)
            # Stepping again after the dumps marks them finished — otherwise a
            # stall here would be indistinguishable from one inside the last dump.
            on_step("Publishing bundle (atomic rename)")
            try:
                os.replace(staged, target)
            except BaseException:
                if aside is not None:
                    os.replace(aside, target)
                shutil.rmtree(staged, ignore_errors=True)
                raise
            durability.sync_dir(self.dir)  # the rename itself (R13)
            if aside is not None:
                shutil.rmtree(aside, ignore_errors=True)
            if model is None:
                return
            # Publish to the cache while STILL holding _disk_lock: a concurrent
            # delete() takes _disk_lock for its rmtree, so it can no longer land
            # between the rename and the cache insert and leave the model cached
            # but absent on disk. (Nesting is one-directional — no path takes
            # _disk_lock while holding _lock — so there is no ordering cycle.)
            stamp = self._stamp(name)
            with self._lock:
                self._cache[name] = model
                self._stamps[name] = stamp
                self._cache.move_to_end(name)
                self._evict()

    @contextmanager
    def importing(self, name: str) -> Iterator[None]:
        """Hold ``name`` for one import, from before its upload until it is installed or failed.

        An upload takes minutes, and nothing else could see it coming: a training of the same
        name was accepted meanwhile, and whichever published second failed after all its work
        (audit 2026-09-30, R01). A second import of the name is refused at once
        (``FileExistsError``); ``/train`` asks :meth:`is_importing`. Under ``_lock`` -- a set,
        no I/O -- so the lock order (``_disk_lock`` before ``_lock``) is untouched.
        """
        with self._lock:
            if name in self._importing:
                raise FileExistsError(name)
            self._importing.add(name)
        try:
            yield
        finally:
            with self._lock:
                self._importing.discard(name)

    def is_importing(self, name: str) -> bool:
        with self._lock:
            return name in self._importing

    def discard_staged(self, staged: Path) -> None:
        """Remove a staged bundle that will not be published (a stopped or killed run)."""
        shutil.rmtree(staged, ignore_errors=True)

    def load_fresh(self, name: str) -> tuple[ClassifierModel, dict]:
        with self._disk_lock:
            if not self.exists(name):
                raise FileNotFoundError(name)
            return _read_bundle(self._path(name))

    def get(self, name: str) -> ClassifierModel:
        """Return a cached model (LRU), loading from disk on a miss -- or when the bundle on
        disk changed since it was cached (``_stamp``)."""
        stamp = self._stamp(name)
        with self._lock:
            if name in self._cache and stamp is not None and self._stamps.get(name) == stamp:
                self._cache.move_to_end(name)
                return self._cache[name]
        # Miss path: read AND publish to the cache while holding _disk_lock, so a
        # concurrent delete() (which pops the cache, then rmtrees under _disk_lock)
        # can never land between our read and our insert — otherwise a deleted
        # model would stay cached and a same-name re-import would serve the OLD
        # weights. Nesting _lock inside _disk_lock is the documented legal order
        # (save() does the same); the reverse never happens.
        with self._disk_lock:
            # Again, now that the disk lock is ours: loads take turns on it, and a request
            # that waited behind the one loading this model finds it here instead of reading
            # it once more -- eight waiting requests loaded one model eight times (audit
            # 2026-09-30, R04).
            stamp = self._stamp(name)  # under the lock: what is loaded is what is stamped
            with self._lock:
                if name in self._cache and stamp is not None and self._stamps.get(name) == stamp:
                    self._cache.move_to_end(name)
                    return self._cache[name]
            if not self.exists(name):
                with self._lock:  # removed by hand: it must not keep answering from memory
                    self._cache.pop(name, None)
                    self._stamps.pop(name, None)
                raise FileNotFoundError(name)
            model, _ = _read_bundle(self._path(name))
            with self._lock:
                self._cache[name] = model
                self._stamps[name] = stamp
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
            evaluations = metadata.setdefault("evaluations", [])
            evaluations.append(record)
            # The newest MAX_EVALUATIONS (R15): every one ever made grew the bundle without end.
            del evaluations[:-MAX_EVALUATIONS]

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
            durability.sync_file(tmp)  # R13: never an empty metrics.json after a power cut
            os.replace(tmp, path)
            durability.sync_dir(path.parent)

    def delete(self, name: str) -> None:
        # Pop the cache INSIDE the disk-lock section, AFTER the rmtree: popping
        # first (outside) let a concurrent cold get()/save() insert the model
        # again while our rmtree waited for the lock — cache outliving disk.
        # With this ordering, any insert either happens before us (we remove it)
        # or after we release (its exists() check then fails -> no insert).
        with self._disk_lock:
            if not self.exists(name):
                raise FileNotFoundError(name)
            # A rename first, which frees the name in one step, then the removal. As one
            # rmtree, a delete cut short -- a file another process holds, an I/O error -- left
            # part of the bundle under the name: no model, but in the way of the next import
            # (audit 2026-09-30, R11). Hidden and ending in .tmp, a leftover is the startup
            # sweep's to remove.
            doomed = self.dir / f".{name}.deleted-{secrets.token_hex(4)}.tmp"
            os.replace(self._path(name), doomed)
            try:
                shutil.rmtree(doomed)
            except OSError as exc:
                logger.warning("Deleted model %r, but could not remove all of it yet (%s); "
                               "the next start does.", name, exc)
            with self._lock:
                # Every spelling, not just this one: on a case-insensitive
                # filesystem get("M") loads the bundle "m" and caches it under
                # "M", and that key kept serving the deleted weights -- after a
                # retrain of "m", the OLD model. On a case-sensitive one this may
                # also drop a different model's entry, which only costs a reload.
                key = name.casefold()
                for cached in [k for k in self._cache if k.casefold() == key]:
                    del self._cache[cached]
                    self._stamps.pop(cached, None)

    def stage_export(self, name: str) -> tuple[Path, Callable[[], None]]:
        """The bundle's archive in a staging file beside the bundles, and its release.

        Here rather than in the route (audit ARC-1) because *where* the staging file goes
        is this class's business: on a container ``/tmp`` is often tmpfs, i.e. RAM, which
        would give back exactly what streaming to a file removes — and the hidden
        ``.*.tmp`` name is the one ``sweep`` already cleans, so a download that dies
        mid-flight leaks nothing permanently.

        Downloads of one bundle state share one file (``staged_archives``; each used to pack
        its own, audit 2026-09-30 S02). The caller calls the release once the body is sent;
        the last one deletes the file. Raises ``FileNotFoundError`` for an unknown name.
        """
        with self._disk_lock:
            sources = self._packable(name)
            # What the archive is packed from, down to each member's size and mtime: after an
            # `update_info` or a re-publish, a download gets a fresh archive, not the old one.
            state = (name, tuple((member, stat.st_size, stat.st_mtime_ns)
                                 for member, stat in ((m, p.stat()) for m, p in sources.items())))
            return self._staged.acquire(state, lambda stream: model_archive.pack_into(stream, name, sources))

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
        with self._disk_lock:
            model_archive.pack_into(target, name, self._packable(name))

    def _packable(self, name: str) -> dict[str, Path]:
        """The members an export packs, by name. The caller holds ``_disk_lock``."""
        if not self.exists(name):
            raise FileNotFoundError(name)
        packable = model_archive.ALLOWED_MEMBERS - {MANIFEST_FILE, CARD_FILE}
        return {
            file.name: file
            for file in sorted(self._path(name).iterdir())
            if file.is_file() and file.name in packable
        }

    def import_zip(self, name: str, data: bytes) -> dict:
        """Install a model from archive bytes already in memory.

        A convenience over :meth:`import_archive`, which is the real implementation: the
        bytes are spooled to a staging file and installed from there, so there is ONE
        validate-stage-publish path rather than two that have to keep agreeing. Callers that
        receive an upload should stream it to a file and use `import_archive` directly —
        holding the whole archive is the peak API-5 is about.
        """
        self.dir.mkdir(parents=True, exist_ok=True)
        handle, staged = tempfile.mkstemp(prefix=".import-", suffix=".zip.tmp", dir=self.dir)
        try:
            with os.fdopen(handle, "wb") as sink:
                sink.write(data)
            return self.import_archive(name, Path(staged))
        finally:
            Path(staged).unlink(missing_ok=True)

    def import_archive(self, name: str, archive_path: Path) -> dict:
        """Validate an archive on disk (``model_archive.unpack_into``) and install it."""
        if self.exists(name):
            raise FileExistsError(name)
        # Stage + validate in a hidden tmp dir; publish only complete bundles.
        # Under the disk lock so a concurrent load/save/delete can never observe
        # the tmp dir or the exists()->replace window mid-flight.
        with self._disk_lock:
            tmp = self.new_staging(name)
            try:
                # Writes the members straight into the staging dir, one at a time, and
                # removes them all again if any check fails.
                model_archive.unpack_into(archive_path, tmp)
                _read_bundle(tmp)  # validates config + skops safety; raises if unsafe
                durability.sync_tree(tmp)  # on disk before the rename exposes it (R13)
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
            durability.sync_dir(self.dir)
        return self.info(name)  # outside the disk lock: info() re-acquires it sequentially


@lru_cache
def get_registry() -> Registry:
    settings = get_settings()
    return Registry(settings.models_dir, settings.effective_max_models_in_memory())

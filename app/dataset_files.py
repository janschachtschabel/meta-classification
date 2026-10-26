"""How a new dataset file comes into existence in the data directory.

Spooled to a staging file nobody can name, then linked into place -- so a dataset exists the
moment it is complete, and a name that exists is refused rather than replaced. Shared by the
upload (`POST /datasets/import`) and by everything else that writes a new dataset (the
corrections' "save as dataset"); a route module cannot import another, so it lives here.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def staging_path(data_dir: Path, name: str) -> Path:
    """A staging file for one new dataset, beside the target and nameable by no route.

    Spool-then-rename is what makes the rename the moment a dataset exists, so a failure
    part-way leaves something no route can ask for rather than a truncated CSV that lists,
    inspects and trains as if it were complete. Two properties carry that:

    * a **leading dot**, which ``safe_name`` rejects, and a ``.part`` suffix, which the
      listing's CSV glob does not match and the startup sweep does clean;
    * a **unique** name. It used to be ``<name>.part``, i.e. the same file for every request
      importing one name — so two concurrent imports interleaved their bytes into whichever
      one reached the rename (audit SEC-12).
    """
    handle, staged = tempfile.mkstemp(prefix=".import-", suffix=".part", dir=data_dir)
    os.close(handle)
    return Path(staged)


def publish_new(staging: Path, target: Path) -> None:
    """Move the staged file into place, or refuse because the name was taken.

    ``os.replace`` is atomic but silently OVERWRITES, and the 409 check runs before the
    upload — so a name created during a long upload was replaced by it anyway. ``os.link``
    fails with ``FileExistsError`` instead, atomically, which is the answer the routes
    already give for a name that exists (audit SEC-12).

    :raises FileExistsError: the target appeared while the file was being written.
    """
    try:
        os.link(staging, target)
    except FileExistsError:
        staging.unlink(missing_ok=True)
        raise
    except OSError:
        # No hard links here (a filesystem or platform that refuses them). Fall back to the
        # atomic-but-overwriting rename after one more check: the race window shrinks from
        # the whole upload to these two statements, which is the best this can do without
        # links.
        if target.exists():
            staging.unlink(missing_ok=True)
            raise FileExistsError(target) from None
        try:
            os.replace(staging, target)
        except BaseException:
            staging.unlink(missing_ok=True)
            raise
        return
    staging.unlink(missing_ok=True)

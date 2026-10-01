"""The scripts that write inside a live model bundle (audit T-5).

`patch_bundle_labels.py` rewrites `config.json` — the document `model_io` trusts for class
order and per-label thresholds — and had no test and no type check, while every writer in
`app/` goes through a tmp-then-rename. 22 of 24 scripts are untested; these two are the ones
where being untested costs a bundle rather than a benchmark number.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from patch_bundle_labels import patch, write_config_atomically  # noqa: E402


def _bundle(tmp_path: Path, uri_to_label: dict[str, str] | None = None) -> Path:
    bundle = tmp_path / "model"
    bundle.mkdir()
    (bundle / "config.json").write_text(
        json.dumps({
            "classes": ["uri:a", "uri:b"],
            "task_type": "multilabel",
            "per_label_thresholds": {"uri:a": 0.4},
            "uri_to_label": {"uri:a": "Alpha"} if uri_to_label is None else uri_to_label,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return bundle


def _config(bundle: Path) -> dict:
    return json.loads((bundle / "config.json").read_text(encoding="utf-8"))


# --- the write itself ----------------------------------------------------------------------


def test_a_failed_write_leaves_the_original_config_intact(tmp_path, monkeypatch):
    """The reason this is atomic: a plain `write_text` that dies half-way (a full disk, a
    kill) truncates the config.json inside a bundle that may be serving, and the registry
    then refuses to load the model at all."""
    import patch_bundle_labels

    bundle = _bundle(tmp_path)
    before = (bundle / "config.json").read_text(encoding="utf-8")

    def _die(_src, _dst):
        raise OSError("no space left on device")

    monkeypatch.setattr(patch_bundle_labels.os, "replace", _die)
    with pytest.raises(OSError):
        write_config_atomically(bundle / "config.json", {"classes": ["replaced"]})

    assert (bundle / "config.json").read_text(encoding="utf-8") == before
    assert not list(bundle.glob("*.tmp")), "the staging file was left behind"


def test_a_successful_write_replaces_the_config_in_one_step(tmp_path):
    bundle = _bundle(tmp_path)

    write_config_atomically(bundle / "config.json", {"classes": ["uri:a"], "new": True})

    assert _config(bundle) == {"classes": ["uri:a"], "new": True}
    assert not list(bundle.glob("*.tmp"))


def test_the_written_file_keeps_the_readable_form_model_io_writes(tmp_path):
    """`ensure_ascii=False, indent=2`, the same as `model_io._write_bundle`: a repair should
    not turn a reviewable file into one line of escapes."""
    bundle = _bundle(tmp_path)

    write_config_atomically(bundle / "config.json", {"label": "Ökologie"})
    text = (bundle / "config.json").read_text(encoding="utf-8")

    assert "Ökologie" in text, "non-ASCII was escaped"
    assert "\n  " in text, "the indentation is gone"


# --- what the script is for ----------------------------------------------------------------


def test_a_dry_run_changes_nothing(tmp_path):
    """The script defaults to a dry run; that default is only worth anything if it holds."""
    bundle = _bundle(tmp_path)
    before = (bundle / "config.json").read_text(encoding="utf-8")

    corrected, added, nameless = patch(bundle, {"uri:a": "Alpha korrigiert", "uri:b": "Beta"},
                                       apply=False)

    assert (corrected, added) == (1, 1)
    assert (bundle / "config.json").read_text(encoding="utf-8") == before
    assert nameless == 0


def test_applying_it_adds_and_corrects_names_and_touches_nothing_else(tmp_path):
    """The two things `model_io` reads from this file — class order and thresholds — must come
    out byte-identical. The script asserts that itself after writing; this proves the
    assertion is reached and true."""
    bundle = _bundle(tmp_path)

    patch(bundle, {"uri:a": "Alpha korrigiert", "uri:b": "Beta"}, apply=True)

    after = _config(bundle)
    assert after["uri_to_label"] == {"uri:a": "Alpha korrigiert", "uri:b": "Beta"}
    assert after["classes"] == ["uri:a", "uri:b"], "class ORDER decides the head's columns"
    assert after["per_label_thresholds"] == {"uri:a": 0.4}


def test_the_first_original_is_kept_as_the_backup_not_the_previous_patch(tmp_path):
    """Patching twice must still leave the pre-repair file recoverable."""
    bundle = _bundle(tmp_path)

    patch(bundle, {"uri:a": "erste Korrektur"}, apply=True)
    patch(bundle, {"uri:a": "zweite Korrektur"}, apply=True)

    backup = json.loads((bundle / "config.json.bak").read_text(encoding="utf-8"))
    assert backup["uri_to_label"] == {"uri:a": "Alpha"}, "the backup is the previous patch"
    assert _config(bundle)["uri_to_label"]["uri:a"] == "zweite Korrektur"



def test_the_label_repair_rewrites_models_and_nothing_else(tmp_path):
    """W04 (audit 2026-09-30): every directory holding a config.json was patched -- a training's
    staging directory, a delete's tombstone, and the `.prebackup` copy the repairs keep so the
    untouched original survives. Run as the script runs."""
    import subprocess

    models = tmp_path / "models"
    config = {"classes": ["u:a"], "uri_to_label": {"u:a": "alt"}, "per_label_thresholds": {"u:a": 0.5}}
    for directory in ("m", ".m.0a1b2c3d.tmp", "m.prebackup", ".m.deleted-0a1b2c3d.tmp"):
        (models / directory).mkdir(parents=True)
        (models / directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    names = tmp_path / "label_names.json"
    names.write_text('{"u:a": "neu"}', encoding="utf-8")

    subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "patch_bundle_labels.py"),  # noqa: S603
                    "--names", str(names), "--models-dir", str(models), "--apply"],
                   check=True, capture_output=True, text=True, timeout=120)

    changed = {d.name for d in models.iterdir()
               if json.loads((d / "config.json").read_text(encoding="utf-8"))["uri_to_label"]["u:a"] == "neu"}
    assert changed == {"m"}

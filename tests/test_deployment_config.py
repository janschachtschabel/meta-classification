"""The deployment descriptors are inside a gate (audit OPS-4…6, OPS-10…12, DEP-5).

Nothing in the suite read `docker-compose.yml`, the Helm chart or the workflows, which is
how a compose file came to document a memory budget it never established, how `.dockerignore`
came to disagree with `.gitignore` about 57 MB of exported bundles, and how the gate that
guards a *release* came to be weaker than the one that guards a push.

These assert the invariant, not the value: a limit must exist and be readable, the two ignore
files must agree, and the release gate must run every step the push gate runs — so a future
edit that re-opens one of these gaps fails here instead of in production.
"""

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CHART = ROOT / "deploy" / "helm" / "classification-api"


def _compose() -> dict:
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))


def _service() -> dict:
    return _compose()["services"]["classification-api"]


# --- OPS-5: the documented memory budget must actually exist ------------------------------


def test_compose_establishes_the_memory_limit_its_comment_promises():
    """`APIV3_TRAIN_MEMORY_MB: auto` reads the container's cgroup limit. With no limit set,
    `memory_limit_bytes()` returns None and the cap is simply off — so the comment described
    protection the quickstart did not have."""
    service = _service()

    limit = service.get("mem_limit") or (
        service.get("deploy", {}).get("resources", {}).get("limits", {}).get("memory")
    )
    assert limit, (
        "docker-compose.yml sets APIV3_TRAIN_MEMORY_MB=auto, which sizes the head-fit "
        "threads from the container's memory limit — but declares no limit, so the cap is off"
    )


# --- OPS-6: logs are bounded like the data volume is --------------------------------------


def test_compose_rotates_its_logs():
    """A `restart: unless-stopped` service at INFO writes forever. The data volume is
    carefully bounded; the logs beside it were not."""
    options = _service().get("logging", {}).get("options", {})

    assert options.get("max-size"), "no max-size: an unbounded log on a long-lived service"
    assert options.get("max-file"), "no max-file: rotation without a retention count"


# --- OPS-10: the two ignore files must agree ----------------------------------------------


def _ignored_dirs(path: Path) -> set[str]:
    """Directory entries from an ignore file, normalised to a bare name."""
    names = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        # `**/__pycache__` and `__pycache__/` name the same directory.
        names.add(line.removeprefix("**/").removeprefix("/").rstrip("/"))
    return names


def test_every_gitignored_directory_that_exists_is_out_of_the_build_context():
    """Too generated or too large for git means too large for the build context too.
    `fertige-modelle/` was 57 MB of a 64 MB context, uploaded to the daemon on every build."""
    git_dirs = {
        name
        for name in _ignored_dirs(ROOT / ".gitignore")
        if "*" not in name and (ROOT / name).is_dir()
    }
    docker_dirs = _ignored_dirs(ROOT / ".dockerignore")

    assert git_dirs <= docker_dirs, (
        f"gitignored but shipped into the build context: {sorted(git_dirs - docker_dirs)}"
    )


# --- OPS-11: two pushes must not race for a mutable tag -----------------------------------


def _workflow(name: str) -> dict:
    return yaml.safe_load((ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("filename", ["ci.yml", "docker.yml"])
def test_every_workflow_serialises_runs_per_ref(filename):
    """Without a concurrency group, two quick pushes to main race and the mutable `main`
    image tag ends up pointing at whichever build finished last — not the newer commit."""
    assert "concurrency" in _workflow(filename), (
        f"{filename} has no concurrency group, so two pushes to the same ref race"
    )


# --- OPS-12: the release gate is not the weaker gate ---------------------------------------


def _step_names(job: dict) -> set[str]:
    return {step["name"] for step in job.get("steps", []) if "name" in step}


def test_the_release_gate_runs_every_check_the_push_gate_runs():
    """`docker.yml` gates tag releases; `ci.yml` does not run on tags at all. Any check only
    `ci.yml` ran was therefore skipped for exactly the builds that ship."""
    push_gate = _step_names(_workflow("ci.yml")["jobs"]["api"])
    release_gate = _step_names(_workflow("docker.yml")["jobs"]["test"])

    assert push_gate <= release_gate, (
        f"only the push gate runs: {sorted(push_gate - release_gate)} — a tag release is "
        "gated by the weaker of the two workflows"
    )


def test_the_push_gate_also_runs_on_tags():
    """Belt and braces for the check above: if the two gates ever diverge again, at least
    both workflows see a release."""
    # PyYAML parses the bare key `on` as the boolean True.
    triggers = _workflow("ci.yml")[True]

    assert triggers["push"].get("tags"), "ci.yml does not trigger on tags"


# --- B01 (audit 2026-09-30): the GitLab suite can pass the chart's render gate -------------


def _gitlab() -> dict:
    return yaml.safe_load((ROOT / ".gitlab-ci.yml").read_text(encoding="utf-8"))


def test_the_gitlab_suite_job_brings_the_helm_its_render_tests_demand():
    """`tests/test_helm_chart.py` refuses to skip wherever `CI` is set, and GitLab always
    sets it. The job running the suite had no helm, so eight render tests failed in every
    pipeline, and the build and deploy stages behind them never ran -- not even for the
    `v4.0.1` tag. The job brings its own helm, verified against the release checksum like
    everything else this pipeline pulls by hand."""
    suite_jobs = [
        job for job in _gitlab().values()
        if isinstance(job, dict) and "pytest tests" in " ".join(job.get("script", []))
    ]
    assert suite_jobs, "no GitLab job runs the test suite"
    for job in suite_jobs:
        setup = " ".join(job.get("before_script", []))
        assert "get.helm.sh" in setup, (
            "the GitLab suite job installs no helm, so the chart's render tests fail under CI"
        )
        assert "sha256sum -c" in setup, "the helm the suite job installs is not checksum-verified"


# --- DEP-5: licences are an invariant, not a one-time assessment ---------------------------


def test_the_audit_job_gates_on_licences():
    """The 40-package tree was all-permissive when audited by hand. That is an assessment of
    a moment; a transitive GPL arriving in a routine bump is what a gate catches."""
    audit = _workflow("ci.yml")["jobs"]["audit"]
    script = " ".join(step.get("run", "") for step in audit["steps"])

    assert "pip-licenses" in script or "licensecheck" in script, (
        "no licence gate in the audit job: an incompatible licence can arrive transitively"
    )


# --- OPS-4: the API key can come from a secret the chart does not own ----------------------


def test_the_chart_accepts_an_externally_managed_secret():
    """Keys as Helm values land in the release secret, in any values file used to install and
    — via the `--set` route NOTES.txt recommends — in shell history and CI logs. Rotation
    then means a chart upgrade. An `existingSecret` is the hook sealed-secrets, External
    Secrets and Vault all need."""
    values = yaml.safe_load((CHART / "values.yaml").read_text(encoding="utf-8"))

    assert "existingSecret" in values["config"]["auth"], (
        "config.auth.existingSecret is missing: no path for sealed-secrets/ESO/Vault"
    )


def test_both_secret_wiring_points_name_the_external_secret():
    """A wiring floor, not the guarantee: both the Secret template and the StatefulSet read
    `existingSecret` at all. That the chart then renders no Secret of its own and the pod
    follows is proven by rendering, in `tests/test_helm_chart.py`. This half survives when
    helm is not installed."""
    secret = (CHART / "templates" / "secret-env.yaml").read_text(encoding="utf-8")
    statefulset = (CHART / "templates" / "statefulset.yaml").read_text(encoding="utf-8")

    assert "existingSecret" in secret, "secret-env.yaml still always renders its own Secret"
    assert "existingSecret" in statefulset, (
        "statefulset.yaml does not read config.auth.existingSecret, so the pod still mounts "
        "the chart's own Secret"
    )



# --- R07: one worker, whatever the environment says ----------------------------------------------


def _image_command() -> list[str]:
    """The image's CMD, in its exec form."""
    import json

    line = next(line for line in (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines()
                if line.startswith("CMD ["))
    return json.loads(line[len("CMD "):])


def test_the_image_runs_one_worker_even_when_the_environment_asks_for_more(monkeypatch):
    """R07 (audit 2026-09-30): uvicorn takes its worker count from WEB_CONCURRENCY unless the
    command names one, and the training job, model cache, rate limits and share links live in
    ONE process -- with several workers, 9 of 20 share links created in one were unknown to
    the next. Read through uvicorn's own rule, as the image starts it."""
    import uvicorn

    command = _image_command()
    assert command[:2] == ["uvicorn", "app.main:app"]
    workers = int(command[command.index("--workers") + 1]) if "--workers" in command else None
    monkeypatch.setenv("WEB_CONCURRENCY", "4")

    assert uvicorn.Config("app.main:app", workers=workers).workers == 1


# --- B02/B03: what the GitLab pipeline pushes is what the charts reference ----------------------

needs_sh = pytest.mark.skipif(shutil.which("sh") is None, reason="no POSIX shell to expand in")


def _expanded(words: list[str], env: dict[str, str]) -> list[str]:
    """Each word as GitLab's shell expands it -- `${CI_COMMIT_TAG#v}` included, which a
    string comparison cannot judge."""
    script = "\n".join(f"printf '%s\\n' {word}" for word in words)
    done = subprocess.run(["sh", "-c", script], capture_output=True, text=True, timeout=30,  # noqa: S603, S607
                          env={**os.environ, **env}, check=True)
    return done.stdout.splitlines()


def _pushed(job: str, env: dict[str, str]) -> set[str]:
    """The image references a GitLab job pushes, for one set of CI variables."""
    refs = [line.split("docker image push", 1)[1].strip()
            for line in _gitlab()[job]["script"] if line.startswith("docker image push")]
    return set(_expanded(refs, env))


_REGISTRY = {"DOCKER_REGISTRY": "registry.example", "DOCKER_IMAGE_PATH": "wlo/classification-api"}


@needs_sh
def test_a_release_pushes_the_image_tag_the_chart_names():
    """B02 (audit 2026-09-30): the chart names its image by `appVersion` -- 4.0.1 -- and the
    GitLab tag build pushed only the git tag, v4.0.1. Installed from the repository as its own
    README says, the chart pulled an image no pipeline had pushed to that registry."""
    app_version = yaml.safe_load((CHART / "Chart.yaml").read_text(encoding="utf-8"))["appVersion"]
    tag = f"v{app_version}"

    pushed = _pushed("build and push (tags)", {**_REGISTRY, "CI_COMMIT_TAG": tag, "CI_COMMIT_REF_NAME": tag})

    assert f"registry.example/wlo/classification-api:{app_version}" in pushed, pushed
    assert f"registry.example/wlo/classification-api:{tag}" in pushed, "the git tag itself is still pushed"


@needs_sh
def test_every_main_pipeline_rolls_out_the_image_it_built():
    """B03 (audit 2026-09-30): the branch chart named the image `:main` and carried the same
    version every time, so `helm upgrade` changed nothing in the pod spec, the pod never
    rolled, IfNotPresent kept the cached image, and `helm rollback` restored the same `:main`.
    The chart now names the immutable sha- tag the branch build pushes, under a version that
    differs per pipeline."""
    job = _gitlab()["build and push helm chart"]

    def versions(sha: str, iid: str) -> tuple[str, str]:
        env = {"CI_COMMIT_REF_SLUG": "main", "CI_COMMIT_SHORT_SHA": sha, "CI_PIPELINE_IID": iid}
        chart, app = _expanded([f'"{job["variables"][k]}"' for k in ("CHART_VERSION", "APP_VERSION")], env)
        return chart, app

    first_chart, first_app = versions("1a2b3c4", "41")
    second_chart, second_app = versions("5d6e7f8", "42")
    built = _pushed("build and push (branches)",
                    {**_REGISTRY, "CI_COMMIT_REF_SLUG": "main", "CI_COMMIT_SHORT_SHA": "1a2b3c4"})

    assert f"registry.example/wlo/classification-api:{first_app}" in built, (
        f"the chart names :{first_app}, which the branch build does not push")
    assert first_app != "main", "the chart still names the mutable branch tag"
    assert first_app != second_app and first_chart != second_chart, (
        "two pipelines produce the same chart, so an upgrade does not roll the pod")


# --- B04: compose hands the container what .env says ------------------------------------------

_PATHS = ("APIV3_DATA_DIR", "APIV3_MODELS_DIR", "APIV3_SHARE_LINKS_FILE",
          "APIV3_FEEDBACK_FILE", "APIV3_JOB_HISTORY_FILE")


def test_compose_reads_every_setting_from_env_and_keeps_the_volume_paths():
    """B04 (audit 2026-09-30): compose passed on a fixed list of variables, so fifteen settings
    `.env.example` documents -- the rate limits, the upload cap, the UI switch -- and
    `FORWARDED_ALLOW_IPS` never reached the container. `.env` is now the container's
    environment; the volume paths stay pinned over it, so a `.env` written for a local run
    cannot point the container inside its own filesystem. Optional, so keys exported in the
    shell instead still work."""
    service = _service()
    entries = service.get("env_file") or []
    env_files = [entry if isinstance(entry, dict) else {"path": entry} for entry in
                 (entries if isinstance(entries, list) else [entries])]

    assert any(e["path"] == ".env" and e.get("required") is False for e in env_files), env_files
    environment = service["environment"]
    assert all(environment.get(name, "").startswith("/data/") for name in _PATHS), (
        "a volume path is no longer pinned over .env")


@pytest.mark.skipif(shutil.which("docker") is None, reason="no docker CLI to render compose with")
def test_a_setting_in_env_reaches_the_container(tmp_path):
    """The same, as compose itself renders it -- from a copy with a placeholder `.env`, so a
    real one beside the repository's compose file is never read."""
    (tmp_path / "docker-compose.yml").write_bytes((ROOT / "docker-compose.yml").read_bytes())
    (tmp_path / ".env").write_text(
        "APIV3_API_KEY_ADMIN=render-test\nAPIV3_API_KEY_READONLY=render-test\n"
        "FORWARDED_ALLOW_IPS=10.42.0.0/16\nAPIV3_MAX_UPLOAD_MB=99\nAPIV3_DATA_DIR=./local-data\n",
        encoding="utf-8")
    clean = {k: v for k, v in os.environ.items() if not k.startswith(("APIV3_", "FORWARDED_"))}

    done = subprocess.run(["docker", "compose", "config", "--format", "json"], cwd=tmp_path,  # noqa: S603, S607
                          capture_output=True, text=True, timeout=60, env=clean, check=True)

    environment = json.loads(done.stdout)["services"]["classification-api"]["environment"]
    assert environment.get("FORWARDED_ALLOW_IPS") == "10.42.0.0/16"
    assert environment.get("APIV3_MAX_UPLOAD_MB") == "99"
    assert environment.get("APIV3_DATA_DIR") == "/data/datasets", "the volume path lost to .env"


# --- B05: the licence gate catches a GPL licence as the metadata spells it ----------------------


def _licence_step() -> str:
    [step] = [s for s in _workflow("ci.yml")["jobs"]["audit"]["steps"]
              if "piplicenses" in s.get("run", "")]
    return step["run"]


def _gate(tmp_path: Path, classifier: str) -> subprocess.CompletedProcess[str]:
    """The audit job's own pip-licenses command, run against one installed distribution."""
    pytest.importorskip("piplicenses", reason="pip-licenses (requirements-dev.txt) is not installed")
    [command] = [line for line in _licence_step().splitlines() if "-m piplicenses" in line]
    args = shlex.split(command.strip())[3:]  # after "python -m piplicenses"
    site = tmp_path / "site"
    (site / "licence_probe-1.0.dist-info").mkdir(parents=True)
    (site / "licence_probe-1.0.dist-info" / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: licence-probe\nVersion: 1.0\nLicense: see classifier\n"
        f"Classifier: {classifier}\n", encoding="utf-8")
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-m", "piplicenses", *args, "--packages", "licence-probe"],
        capture_output=True, text=True, timeout=120, check=False,
        env={**os.environ, "PYTHONPATH": str(site)})


def test_the_licence_gate_fails_on_a_gpl_package_as_its_metadata_spells_it(tmp_path):
    """B05 (audit 2026-09-30): `--fail-on "GPL;AGPL;LGPL"` compared whole licence names, and
    no package calls its licence "GPL": Unidecode's metadata says "GNU General Public
    License v2 or later (GPLv2+)", and the gate passed it. Run here exactly as the audit job
    runs it, against a package that declares that classifier."""
    gate = _gate(tmp_path, "License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)")

    assert gate.returncode != 0, f"a GPLv2+ package passed the licence gate:\n{gate.stdout}"


def test_the_licence_gate_passes_a_permissive_package(tmp_path):
    """The other half: a gate that fails on everything gates nothing."""
    gate = _gate(tmp_path, "License :: OSI Approved :: MIT License")

    assert gate.returncode == 0, gate.stdout + gate.stderr


def test_the_licence_gate_reads_the_tree_the_image_installs():
    """It installed `requirements.lock` -- 17 direct pins, resolved fresh -- so it judged the
    tree the next recompile would get, not the one the image ships. And its own tool came
    unpinned; it is pinned with the other dev tools now."""
    step = _licence_step()
    dev_pin = next(line.strip() for line in (ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
                   .splitlines() if line.lower().startswith("pip-licenses=="))

    assert "--require-hashes -r requirements-hashes.lock" in step, step
    assert dev_pin in step, f"the audit job does not install {dev_pin}"

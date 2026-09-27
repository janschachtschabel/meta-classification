"""The deployment descriptors are inside a gate (audit OPS-4…6, OPS-10…12, DEP-5).

Nothing in the suite read `docker-compose.yml`, the Helm chart or the workflows, which is
how a compose file came to document a memory budget it never established, how `.dockerignore`
came to disagree with `.gitignore` about 57 MB of exported bundles, and how the gate that
guards a *release* came to be weaker than the one that guards a push.

These assert the invariant, not the value: a limit must exist and be readable, the two ignore
files must agree, and the release gate must run every step the push gate runs — so a future
edit that re-opens one of these gaps fails here instead of in production.
"""

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


def test_the_chart_does_not_template_a_key_it_was_not_given():
    """With `existingSecret` set, the chart must neither require the plaintext values nor
    render a Secret of its own — otherwise the external secret is decoration."""
    secret = (CHART / "templates" / "secret-env.yaml").read_text(encoding="utf-8")
    statefulset = (CHART / "templates" / "statefulset.yaml").read_text(encoding="utf-8")

    assert "existingSecret" in secret, "secret-env.yaml still always renders its own Secret"
    assert "existingSecret" in statefulset, (
        "statefulset.yaml does not read config.auth.existingSecret, so the pod still mounts "
        "the chart's own Secret"
    )

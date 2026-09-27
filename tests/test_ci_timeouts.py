"""Every CI job declares a timeout (audit T-3).

This suite is full of thread rendezvous with deliberately generous waits, so a deadlock does
not fail — it hangs. GitHub's default job timeout is **360 minutes**, and GitLab's project
default is 60, so an undeclared timeout turns a bug that should surface in minutes into a runner
held for hours and a pipeline whose failure mode is "still running".

Asserted here rather than left to review because nothing else in the suite reads the CI files,
which is how all thirteen jobs came to have none.
"""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]

# GitLab top-level keys that configure the pipeline instead of defining a job.
_GITLAB_NOT_A_JOB = {
    "stages", "variables", "default", "include", "workflow", "image", "services",
    "before_script", "after_script", "cache", "pages",
}


def _github_jobs() -> list[tuple[str, str, dict]]:
    found = []
    for workflow in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
        data = yaml.safe_load(workflow.read_text(encoding="utf-8"))
        for name, job in (data.get("jobs") or {}).items():
            found.append((workflow.name, name, job))
    return found


def _gitlab_jobs() -> list[tuple[str, str, dict]]:
    path = ROOT / ".gitlab-ci.yml"
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        (path.name, name, body)
        for name, body in data.items()
        # A leading dot marks a template GitLab never runs on its own.
        if isinstance(body, dict) and name not in _GITLAB_NOT_A_JOB and not name.startswith(".")
    ]


def test_there_are_ci_jobs_to_check():
    """A guard that silently finds nothing to guard is worse than none."""
    assert len(_github_jobs()) >= 5
    assert len(_gitlab_jobs()) >= 5


@pytest.mark.parametrize("workflow, job", [(w, j) for w, j, _ in _github_jobs()])
def test_every_github_job_has_a_timeout(workflow, job):
    body = next(b for w, n, b in _github_jobs() if (w, n) == (workflow, job))

    assert "timeout-minutes" in body, (
        f"{workflow}:{job} has no timeout-minutes, so it inherits GitHub's 360-minute default"
    )
    assert 1 <= body["timeout-minutes"] <= 60, (
        f"{workflow}:{job} timeout is {body['timeout-minutes']} minutes — long enough to stop "
        "being a timeout"
    )


@pytest.mark.parametrize("job", [j for _, j, _ in _gitlab_jobs()])
def test_every_gitlab_job_has_a_timeout(job):
    body = next(b for _, n, b in _gitlab_jobs() if n == job)

    assert "timeout" in body, f".gitlab-ci.yml:{job} has no timeout"

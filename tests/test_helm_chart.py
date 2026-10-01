"""The Helm chart is checked by RENDERING it (audit SEC-10, OPS-4).

Both guards were pinned by substring — `"fail" in template`, `"existingSecret" in secret`.
A substring survives an inverted condition: the word is present either way, so those tests
stay green while the logic is wrong, and the audit rows they back said "verified" on the
strength of a grep. Only the template engine answers what a given set of values produces.

GitHub's ubuntu runners ship helm, and GitLab's suite job installs a checksum-pinned one
(`.gitlab-ci.yml`, pinned by `tests/test_deployment_config.py`), so this is a real gate in both
CIs; elsewhere it skips and says so. The substring tests stay where they are — they cost
nothing and still hold without helm.
"""

import os
import re
import shutil
import subprocess
from itertools import chain
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CHART = ROOT / "deploy" / "helm" / "classification-api"

HELM = shutil.which("helm")

# Skipping is for a maintainer's machine, never for CI: a silent skip there would leave these
# guarantees exactly where the audit found them — asserted in a document, never executed.
pytestmark = pytest.mark.skipif(
    HELM is None and not os.environ.get("CI"),
    reason="helm is not on PATH, so the chart's render guarantees are unverified in this run",
)

# Placeholders, never a real key: under test is the wiring, not the credential.
KEYS = ("config.auth.adminKey=render-test", "config.auth.readonlyKey=render-test")
# Silences the TLS guard, so a secret-path render fails for its own reason and not OPS-4's.
INSECURE = "ingress.allowInsecure=true"
# Names the proxy uvicorn may believe, so a render with the (default) ingress fails for its
# own reason and not S04's.
PROXIED = "config.limits.forwardedAllowIps=10.42.0.0/16"


def _helm(*overrides: str) -> subprocess.CompletedProcess[str]:
    assert HELM is not None, "helm is gone from the CI image, so the chart is no longer gated"
    args = chain.from_iterable(("--set", override) for override in overrides)
    return subprocess.run(
        [HELM, "template", "release", str(CHART), *args],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _render(*overrides: str) -> list[dict[str, Any]]:
    """The objects a release with these values actually produces."""
    result = _helm(*overrides)
    assert result.returncode == 0, f"render failed: {result.stderr}"
    return [document for document in yaml.safe_load_all(result.stdout) if document]


def _refused(*overrides: str) -> str:
    """The message a release with these values is refused with."""
    result = _helm(*overrides)
    assert result.returncode != 0, f"expected a refusal, got:\n{result.stdout[:400]}"
    return result.stderr


def _of_kind(objects: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [obj for obj in objects if obj.get("kind") == kind]


def _mounted_secret(objects: list[dict[str, Any]]) -> str:
    """The name of the Secret the pod reads its keys from."""
    [statefulset] = _of_kind(objects, "StatefulSet")
    [container] = statefulset["spec"]["template"]["spec"]["containers"]
    names = [entry["secretRef"]["name"] for entry in container["envFrom"] if "secretRef" in entry]
    assert len(names) == 1, f"the pod reads {len(names)} secrets, so which one wins is luck"
    return names[0]


# --- SEC-10 / OPS-4: where the API keys come from -----------------------------------------


def test_an_external_secret_replaces_the_charts_own_and_the_pod_reads_it():
    """With `existingSecret` set the chart must render no Secret at all. Two candidates for
    one envFrom is worse than none: nothing says which the pod ends up reading."""
    objects = _render("config.auth.existingSecret=classify-api-keys", INSECURE, PROXIED)

    assert _of_kind(objects, "Secret") == [], "the chart still renders a Secret beside the external one"
    assert _mounted_secret(objects) == "classify-api-keys"


def test_nothing_on_the_pod_is_derived_from_the_keys():
    """S12 (audit 2026-09-30): a `checksum/secret-env` pod annotation carried a SHA-256 of the
    rendered Secret, readable by anyone allowed to read pods -- and with the rest of that
    manifest public, a weak key could be confirmed offline against it. Two releases that
    differ only in their keys render the same StatefulSet."""
    first = _render("config.auth.adminKey=render-test-a", "config.auth.readonlyKey=render-test-b",
                    INSECURE, PROXIED)
    second = _render("config.auth.adminKey=render-test-c", "config.auth.readonlyKey=render-test-d",
                     INSECURE, PROXIED)

    assert _of_kind(first, "StatefulSet") == _of_kind(second, "StatefulSet")


def test_without_an_external_secret_the_chart_renders_and_mounts_its_own():
    objects = _render(*KEYS, INSECURE, PROXIED)

    [secret] = _of_kind(objects, "Secret")
    assert set(secret["data"]) == {"APIV3_API_KEY_ADMIN", "APIV3_API_KEY_READONLY"}
    assert _mounted_secret(objects) == secret["metadata"]["name"]


def test_a_missing_key_refuses_the_release_and_says_what_to_do_instead():
    """The refusal is the only text the operator sees — the render aborts here, so NOTES.txt
    is never printed. A message naming only `adminKey` sends them to `--set adminKey=`, which
    puts the key in their shell history, the CI log and the release secret."""
    message = _refused(INSECURE, PROXIED)

    assert "config.auth.adminKey" in message
    assert "existingSecret" in message, (
        "the refusal does not mention the Secret-you-manage path, and nothing else can: "
        "this very error prevents NOTES.txt from ever being shown"
    )


# --- SEC-10: an ingress must not publish the API key in cleartext -------------------------


def test_an_ingress_without_tls_refuses_the_release():
    message = _refused(*KEYS, PROXIED)

    assert "ingress.tls" in message
    assert "X-API-Key" in message, "the refusal does not say WHAT leaks, so it reads as pedantry"


def test_the_documented_escape_hatch_renders_an_ingress_without_tls():
    """`allowInsecure` is the only way to say "TLS terminates above me" — if it did not work,
    a mesh or cloud-load-balancer deployment could not use the chart at all."""
    [ingress] = _of_kind(_render(*KEYS, INSECURE, PROXIED), "Ingress")

    assert "tls" not in ingress["spec"]


def test_tls_values_reach_the_rendered_ingress():
    objects = _render(
        *KEYS,
        PROXIED,
        "ingress.tls[0].secretName=classify-tls",
        "ingress.tls[0].hosts[0]=classify.example.de",
    )

    [ingress] = _of_kind(objects, "Ingress")
    assert ingress["spec"]["tls"] == [
        {"secretName": "classify-tls", "hosts": ["classify.example.de"]}
    ]


# --- S04: whom uvicorn believes about a client's address -------------------------------------


def test_an_ingress_with_the_rate_limiter_and_no_trusted_proxy_refuses_the_release():
    """S04 (audit 2026-09-30): behind the ingress every request reaches the pod from the
    controller, so without forwardedAllowIps all clients share ONE rate-limit bucket and one
    busy client throttles everyone. values.yaml asked for the setting; nothing enforced it."""
    message = _refused(*KEYS, INSECURE)

    assert "config.limits.forwardedAllowIps" in message
    assert "rateLimitEnabled=false" in message, "and the way out for whoever cannot name the proxy"


def test_trusting_every_address_refuses_the_release():
    """With "*" any client names its own address in X-Forwarded-For: its own rate-limit
    bucket, or someone else's to exhaust."""
    message = _refused(*KEYS, INSECURE, "config.limits.forwardedAllowIps=*")

    assert "config.limits.forwardedAllowIps" in message


@pytest.mark.parametrize("values", [
    (PROXIED,), ("config.limits.rateLimitEnabled=false",), ("ingress.enabled=false",),
])
def test_a_named_proxy_no_limiter_or_no_ingress_renders(values):
    _render(*KEYS, INSECURE, *values)


@pytest.mark.parametrize("grace, wait", [("60", "50"), ("5", "0")])
def test_the_shutdown_wait_stays_inside_the_pods_grace_period(grace, wait):
    """R03 (audit 2026-09-30): the app waits that long for a running training before recording
    it as interrupted -- longer than the grace, and the kubelet's kill comes first."""
    [configmap] = _of_kind(_render(*KEYS, INSECURE, PROXIED, f"terminationGracePeriod={grace}"),
                           "ConfigMap")

    assert configmap["data"]["APIV3_SHUTDOWN_WAIT_SECONDS"] == wait


def test_the_trusted_proxy_reaches_uvicorn():
    [configmap] = _of_kind(_render(*KEYS, INSECURE, PROXIED), "ConfigMap")

    assert configmap["data"]["FORWARDED_ALLOW_IPS"] == "10.42.0.0/16"


def _documented_commands() -> list[list[str]]:
    """The `--set` values of every helm command the chart README tells a reader to run."""
    readme = (CHART / "README.md").read_text(encoding="utf-8")
    commands = []
    for block in re.findall(r"```bash\n(.*?)```", readme, re.DOTALL):
        for line in re.sub(r"\\\s*\n\s*", " ", block).splitlines():
            if not re.match(r"\s*helm\s+(install|upgrade|template)\b", line):
                continue
            if "-f " in line or "--values" in line:
                continue  # needs a values file this test has no business inventing
            commands.append(re.findall(r"--set\s+(\S+)", line))
    return commands


def test_every_install_command_the_readme_documents_actually_renders():
    """A reader copies the first command they see. The TLS guard was added to the chart and the
    README was re-read rather than run, so its recommended install had been refused ever since."""
    commands = _documented_commands()
    assert commands, "no helm command found in the README — the extraction has rotted"

    for overrides in commands:
        result = _helm(*overrides)
        assert result.returncode == 0, (
            f"the README documents a command that does not render: --set {' --set '.join(overrides)}"
            f"\n{result.stderr}"
        )


def test_the_tls_guard_leaves_a_release_without_an_ingress_alone():
    """The guard sits inside `if ingress.enabled` — outside it, turning the ingress off would
    be impossible, which is the port-forward path the chart's own NOTES describe."""
    objects = _render(*KEYS, "ingress.enabled=false")

    assert _of_kind(objects, "Ingress") == []

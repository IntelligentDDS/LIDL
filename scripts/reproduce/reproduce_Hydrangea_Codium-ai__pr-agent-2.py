import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INSTANCE_ID = "Codium-ai__pr-agent-2"
ORIGINAL = ROOT / "datasets/Hydrangea/repo" / INSTANCE_ID
WORKSPACE = ROOT / "datasets/Hydrangea/environments/Codium-ai__pr-agent-41166dc2/workspaces" / INSTANCE_ID
ARTIFACTS = ROOT / "datasets/Hydrangea/fix_repo" / INSTANCE_ID
PYTHON = ROOT / "datasets/Hydrangea/environments/Codium-ai__pr-agent-41166dc2/.venv/bin/python"
SECRET_MARKERS = ("api_key", "auth", "credential", "password", "secret", "token")
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    re.compile(r"\bya29\.[0-9A-Za-z_-]+\b"),
)


def run(command, cwd, phase):
    started_at = datetime.now(timezone.utc)
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    with (ARTIFACTS / "run.log").open("a", encoding="utf-8") as file:
        file.write(
            f"\n{'=' * 80}\nphase: {phase}\nstarted_at: {started_at.isoformat()}\n"
            f"finished_at: {datetime.now(timezone.utc).isoformat()}\ncwd: {cwd}\n"
            f"command: {subprocess.list2cmdline(command)}\nreturncode: {result.returncode}\n"
            f"stdout:\n{result.stdout}stderr:\n{result.stderr}"
        )
    return result


def initialize_log():
    records = json.loads((ROOT / "datasets/Hydrangea/hydrangea_defect.json").read_text(encoding="utf-8"))
    defect = next(record for record in records if record["instance_id"] == INSTANCE_ID)
    environment = {
        key: "<redacted>" if any(marker in key.lower() for marker in SECRET_MARKERS) else value
        for key, value in os.environ.items()
    }
    (ARTIFACTS / "run.log").write_text(
        "defect:\n" + json.dumps(defect, ensure_ascii=False, indent=2)
        + "\ninvestigation:\n"
        + "- The original diff budget uses the model's advertised window instead of the deployment limit.\n"
        + "- Upstream PR #437 added max_model_tokens and get_max_tokens for this case.\n"
        + "environment:\n" + json.dumps(environment, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_local():
    tests = ARTIFACTS / "tests/test.py"
    original = run([str(PYTHON), "-m", "pytest", "-q", str(tests)], ORIGINAL, "original-fail-to-pass")
    if original.returncode == 0:
        raise SystemExit("Original source unexpectedly passed FAIL_TO_PASS tests")
    dry_run = run(["patch", "--dry-run", "--batch", "-p1", "-i", str(ARTIFACTS / "patch.diff")], ORIGINAL, "patch-dry-run")
    patched = run([str(PYTHON), "-m", "pytest", "-q", str(tests)], WORKSPACE, "patched-tests")
    compile_result = run(
        [str(PYTHON), "-m", "py_compile", str(tests), str(WORKSPACE / "pr_agent/algo/pr_processing.py"), str(WORKSPACE / "pr_agent/algo/utils.py"), str(Path(__file__))],
        ROOT,
        "compile-python",
    )
    if dry_run.returncode or patched.returncode or compile_result.returncode:
        raise SystemExit("Local validation failed")


def validate_docker():
    for apply_patch, expected in (("0", 1), ("1", 0)):
        tag = f"lidl-{INSTANCE_ID.lower()}-{apply_patch}"
        build = run(["docker", "build", "--build-arg", f"APPLY_PATCH={apply_patch}", "-t", tag, "-f", str(ARTIFACTS / "Dockerfile"), str(ROOT)], ROOT, f"docker-build-{apply_patch}")
        if build.returncode:
            raise SystemExit("Docker build failed")
        result = run(["docker", "run", "--rm", tag], ROOT, f"docker-run-{apply_patch}")
        if (result.returncode == 0) != (expected == 0):
            raise SystemExit(f"Unexpected Docker result for APPLY_PATCH={apply_patch}")


def scan_secrets():
    paths = [ARTIFACTS / "patch.diff", ARTIFACTS / "tests/test.py", ARTIFACTS / "Dockerfile", ARTIFACTS / "run.log", Path(__file__)]
    hits = [(path, pattern.pattern) for path in paths for pattern in SECRET_PATTERNS if pattern.search(path.read_text(encoding="utf-8"))]
    with (ARTIFACTS / "run.log").open("a", encoding="utf-8") as file:
        file.write(f"\nsecret_hits: {len(hits)}\npatch_bytes: {(ARTIFACTS / 'patch.diff').stat().st_size}\n")
    if hits:
        raise SystemExit("Secret scan failed")


def main():
    parser = argparse.ArgumentParser(description=f"Reproduce {INSTANCE_ID}.")
    parser.add_argument("--runtime", choices=["local", "docker", "all"], default="local")
    args = parser.parse_args()
    initialize_log()
    if args.runtime in {"local", "all"}:
        validate_local()
    if args.runtime in {"docker", "all"}:
        validate_docker()
    scan_secrets()


if __name__ == "__main__":
    main()

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INSTANCE_ID = "Abonia1__Context-Based-LLMChatbot-0"
BASE_COMMIT = "f21f7e157157d235e0b9ddd2fc997a7d33096da1"
ENVIRONMENT = ROOT / "datasets/Hydrangea/environments/Abonia1__Context-Based-LLMChatbot-f21f7e15"
ORIGINAL_REPO = ROOT / "datasets/Hydrangea/repo" / INSTANCE_ID
WORKSPACE = ENVIRONMENT / "workspaces" / INSTANCE_ID
ARTIFACTS = ROOT / "datasets/Hydrangea/fix_repo" / INSTANCE_ID
DEFECTS = ROOT / "datasets/Hydrangea/hydrangea_defect.json"
PYTHON = ENVIRONMENT / ".venv/bin/python"
SECRET_MARKERS = ("api_key", "auth", "credential", "password", "secret", "token")


def safe_environment() -> dict[str, str]:
    return {
        key: "<redacted>"
        if any(marker in key.lower() for marker in SECRET_MARKERS)
        else value
        for key, value in os.environ.items()
    }


def run(command: list[str], cwd: Path, phase: str) -> subprocess.CompletedProcess:
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


def require(result: subprocess.CompletedProcess, expected: int, phase: str) -> None:
    if result.returncode != expected:
        raise SystemExit(f"{phase} returned {result.returncode}, expected {expected}")


def record_investigation() -> None:
    record = next(
        item
        for item in json.loads(DEFECTS.read_text(encoding="utf-8"))
        if item["instance_id"] == INSTANCE_ID
    )
    environment = "\n".join(
        f"{key}={value}" for key, value in sorted(safe_environment().items())
    )
    now = datetime.now(timezone.utc).isoformat()
    with (ARTIFACTS / "run.log").open("a", encoding="utf-8") as file:
        file.write(
            f"phase: heading-segmentation-patch\nstarted_at: {now}\nfinished_at: {now}\n"
            f"instance_id: {INSTANCE_ID}\nbase_commit: {BASE_COMMIT}\nstatus: reproducible_fixed\n"
            f"complete_record:\n{json.dumps(record, ensure_ascii=False, indent=2)}\n"
            "production_path:\n"
            "- chat.py sends every loaded document directly to CharacterTextSplitter.\n"
            "root_cause:\n"
            "- Character-length splitting has no level-three heading boundary, so a short heading section can merge into the next segment.\n"
            "patch_contract:\n"
            "- Split each document at level-three headings first, retain each heading with its body, then apply the existing character splitter.\n"
            "validation_oracle:\n"
            "- Three production-contract tests fail on immutable source and pass after patch.diff.\n"
            f"environment:\n{environment}\n"
        )


def prepare_workspace() -> None:
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    shutil.copytree(ORIGINAL_REPO, WORKSPACE, symlinks=True)
    require(
        run(["patch", "--batch", "-p1", "-i", str(ARTIFACTS / "patch.diff")], WORKSPACE, "apply-patch-isolated-workspace"),
        0,
        "apply-patch-isolated-workspace",
    )


def run_local() -> None:
    tests = str(ARTIFACTS / "tests/test.py")
    require(run([str(PYTHON), "-m", "pytest", "-q", tests], ORIGINAL_REPO, "local-original-expected-failure"), 1, "local-original-expected-failure")
    require(run(["patch", "--batch", "--dry-run", "-p1", "-i", str(ARTIFACTS / "patch.diff")], ORIGINAL_REPO, "patch-dry-run"), 0, "patch-dry-run")
    prepare_workspace()
    require(run([str(PYTHON), "-m", "pytest", "-q", tests], WORKSPACE, "local-patched"), 0, "local-patched")


def run_docker() -> None:
    for apply_patch, expected in (("0", 1), ("1", 0)):
        tag = f"lidl-{INSTANCE_ID.lower()}-{apply_patch}"
        require(run(["docker", "build", "--build-arg", f"APPLY_PATCH={apply_patch}", "-t", tag, "-f", str(ARTIFACTS / "Dockerfile"), str(ROOT)], ROOT, f"docker-build-apply-patch-{apply_patch}"), 0, f"docker-build-apply-patch-{apply_patch}")
        require(run(["docker", "run", "--rm", tag], ROOT, f"docker-run-apply-patch-{apply_patch}"), expected, f"docker-run-apply-patch-{apply_patch}")


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Reproduce {INSTANCE_ID}.")
    parser.add_argument("--runtime", choices=["local", "docker", "all"], default="local")
    args = parser.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "run.log").write_text("", encoding="utf-8")
    record_investigation()
    if args.runtime in ("local", "all"):
        run_local()
    if args.runtime in ("docker", "all"):
        run_docker()


if __name__ == "__main__":
    main()

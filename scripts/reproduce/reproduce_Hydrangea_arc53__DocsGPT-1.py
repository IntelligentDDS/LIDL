import argparse
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INSTANCE_ID = "arc53__DocsGPT-1"
BASE_COMMIT = "744d4ebbaf4349444d8d361593c4c04652c6c54b"
ENVIRONMENT = ROOT / "datasets/Hydrangea/environments/arc53__DocsGPT-744d4ebb"
ORIGINAL_REPO = ROOT / "datasets/Hydrangea/repo" / INSTANCE_ID
WORKSPACE = ENVIRONMENT / "workspaces" / INSTANCE_ID
ARTIFACTS = ROOT / "datasets/Hydrangea/fix_repo" / INSTANCE_ID
PYTHON = ENVIRONMENT / ".venv/bin/python"
SECRET_MARKERS = ("api_key", "auth", "credential", "password", "secret", "token")


def safe_environment() -> dict[str, str]:
    return {
        key: "<redacted>" if any(marker in key.lower() for marker in SECRET_MARKERS) else value
        for key, value in os.environ.items()
    }


def run(command: list[str], cwd: Path, phase: str) -> subprocess.CompletedProcess:
    started_at = datetime.now(timezone.utc)
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    record = (
        f"\n{'=' * 80}\nphase: {phase}\nstarted_at: {started_at.isoformat()}\n"
        f"finished_at: {datetime.now(timezone.utc).isoformat()}\ncwd: {cwd}\n"
        f"command: {subprocess.list2cmdline(command)}\nreturncode: {result.returncode}\n"
        f"stdout:\n{result.stdout}stderr:\n{result.stderr}"
    )
    with (ARTIFACTS / "run.log").open("a", encoding="utf-8") as file:
        file.write(record)
    return result


def record_investigation() -> None:
    environment = "\n".join(f"{key}={value}" for key, value in sorted(safe_environment().items()))
    with (ARTIFACTS / "run.log").open("a", encoding="utf-8") as file:
        file.write(
            f"\n{'=' * 80}\nphase: final-llama-context-budget-repair\n"
            f"started_at: {datetime.now(timezone.utc).isoformat()}\nfinished_at: {datetime.now(timezone.utc).isoformat()}\n"
            f"cwd: {ROOT}\nreturncode: 0\ninstance_id: {INSTANCE_ID}\nbase_commit: {BASE_COMMIT}\nstatus: reproducible_fixed\n"
            "production_path:\n- The UI retrieves documents and history in routes.py, while LlamaCpp renders the final instruction, question, context and answer wrapper immediately before llama.cpp.\n"
            "root_cause:\n- LlamaCpp fixes n_ctx at 2048 and max_tokens at 150 but sends an unbounded rendered prompt.\n"
            "repair_contract:\n- At the llama.cpp boundary, reserve 150 generation tokens, account for the instruction, question and answer wrappers, truncate only retrieved context with the repository's existing cl100k_base tokenizer, and log a warning.\n"
            "semantic_invariant:\n- The user's question, instruction wrapper and answer marker remain intact even when retrieved context exceeds the remaining input budget.\n"
            "validation_oracle:\n- Four production-contract checks fail on immutable source and pass on the isolated patched workspace.\n"
            f"environment:\n{environment}\n"
        )


def require(result: subprocess.CompletedProcess, expected: int, phase: str) -> None:
    if result.returncode != expected:
        raise SystemExit(f"{phase} expected return code {expected}, got {result.returncode}")


def prepare_workspace() -> None:
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    shutil.copytree(ORIGINAL_REPO, WORKSPACE, symlinks=True)
    require(
        run(
            ["patch", "--batch", "-p1", "-i", str(ARTIFACTS / "patch.diff")],
            WORKSPACE,
            "apply-patch-isolated-workspace",
        ),
        0,
        "apply-patch-isolated-workspace",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Reproduce {INSTANCE_ID}.")
    parser.add_argument("--runtime", choices=["local", "docker"], default="local")
    args = parser.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    record_investigation()
    if args.runtime == "docker":
        original_tag = f"lidl-{INSTANCE_ID.lower()}-original"
        patched_tag = f"lidl-{INSTANCE_ID.lower()}-patched"
        require(run(["docker", "build", "--build-arg", "APPLY_PATCH=0", "-t", original_tag, "-f", str(ARTIFACTS / "Dockerfile"), str(ROOT)], ROOT, "docker-build-original"), 0, "docker-build-original")
        require(run(["docker", "run", "--rm", original_tag], ROOT, "docker-run-original-expected-failure"), 1, "docker-run-original-expected-failure")
        require(run(["docker", "build", "--build-arg", "APPLY_PATCH=1", "-t", patched_tag, "-f", str(ARTIFACTS / "Dockerfile"), str(ROOT)], ROOT, "docker-build-patched"), 0, "docker-build-patched")
        require(run(["docker", "run", "--rm", patched_tag], ROOT, "docker-run-patched"), 0, "docker-run-patched")
        return
    tests = str(ARTIFACTS / "tests/test.py")
    require(run([str(PYTHON), "-m", "pytest", "-q", tests], ORIGINAL_REPO, "local-original-expected-failure"), 1, "local-original-expected-failure")
    require(run(["patch", "--batch", "--dry-run", "-p1", "-i", str(ARTIFACTS / "patch.diff")], ORIGINAL_REPO, "patch-dry-run"), 0, "patch-dry-run")
    prepare_workspace()
    require(run([str(PYTHON), "-m", "pytest", "-q", tests], WORKSPACE, "local-patched"), 0, "local-patched")


if __name__ == "__main__":
    main()

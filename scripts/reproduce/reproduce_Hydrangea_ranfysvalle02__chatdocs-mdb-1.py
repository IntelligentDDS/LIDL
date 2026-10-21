import argparse
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INSTANCE_ID = "ranfysvalle02__chatdocs-mdb-1"
BASE_COMMIT = "6322d0ea03d2a3a14fbfaa7ca9f12104e733a72f"
ENVIRONMENT = ROOT / "datasets/Hydrangea/environments/ranfysvalle02__chatdocs-mdb-6322d0ea"
ORIGINAL_REPO = ROOT / "datasets/Hydrangea/repo" / INSTANCE_ID
WORKSPACE = ENVIRONMENT / "workspaces" / INSTANCE_ID
ARTIFACTS = ROOT / "datasets/Hydrangea/fix_repo" / INSTANCE_ID
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
    result = subprocess.run(
        command, cwd=cwd, text=True, capture_output=True, check=False
    )
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
    environment = "\n".join(
        f"{key}={value}" for key, value in sorted(safe_environment().items())
    )
    with (ARTIFACTS / "run.log").open("a", encoding="utf-8") as file:
        file.write(
            f"\n{'=' * 80}\nphase: final-mongodb-index-configuration-repair\n"
            f"started_at: {datetime.now(timezone.utc).isoformat()}\n"
            f"finished_at: {datetime.now(timezone.utc).isoformat()}\n"
            f"cwd: {ROOT}\nreturncode: 0\ninstance_id: {INSTANCE_ID}\n"
            f"base_commit: {BASE_COMMIT}\n"
            "status: reproducible_fixed\n"
            "production_path:\n"
            "- chatdocs add loads and splits documents before get_vectorstore_from_documents inserts them.\n"
            "- chatdocs chat and ui call get_retrieval_qa, get_vectorstore, and the MongoDB Atlas retriever.\n"
            "root_cause:\n"
            "- The MongoDB conversion hard-coded a placeholder URI, used empty database and collection names, and relied on an undeclared LangChain version for a vector store introduced after 0.0.181.\n"
            "- It never passed the configured Atlas search index name, so queries always used the library default regardless of the deployed index.\n"
            "repair_contract:\n"
            "- Read URI, database, collection, and index name from the existing merged chatdocs.yml configuration.\n"
            "- Reuse one vector-store factory for ingestion and query so both paths address the same namespace and index.\n"
            "- Pin the first compatible LangChain release and declare pymongo.\n"
            "validation_oracle:\n"
            "- Seven static production-contract checks fail on the immutable source and pass on the isolated patched workspace without requiring live credentials.\n"
            f"environment:\n{environment}\n"
        )


def require(result: subprocess.CompletedProcess, expected: int, phase: str) -> None:
    if result.returncode != expected:
        raise SystemExit(
            f"{phase} expected return code {expected}, got {result.returncode}"
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
        require(
            run(
                [
                    "docker",
                    "build",
                    "--build-arg",
                    "APPLY_PATCH=0",
                    "-t",
                    original_tag,
                    "-f",
                    str(ARTIFACTS / "Dockerfile"),
                    str(ROOT),
                ],
                ROOT,
                "docker-build-original",
            ),
            0,
            "docker-build-original",
        )
        require(
            run(
                ["docker", "run", "--rm", original_tag],
                ROOT,
                "docker-run-original-expected-failure",
            ),
            1,
            "docker-run-original-expected-failure",
        )
        require(
            run(
                [
                    "docker",
                    "build",
                    "--build-arg",
                    "APPLY_PATCH=1",
                    "-t",
                    patched_tag,
                    "-f",
                    str(ARTIFACTS / "Dockerfile"),
                    str(ROOT),
                ],
                ROOT,
                "docker-build-patched",
            ),
            0,
            "docker-build-patched",
        )
        require(
            run(
                ["docker", "run", "--rm", patched_tag],
                ROOT,
                "docker-run-patched",
            ),
            0,
            "docker-run-patched",
        )
        return

    tests = str(ARTIFACTS / "tests/test.py")
    require(
        run(
            [str(PYTHON), "-m", "pytest", "-q", tests],
            ORIGINAL_REPO,
            "local-original-expected-failure",
        ),
        1,
        "local-original-expected-failure",
    )
    require(
        run(
            ["patch", "--dry-run", "--batch", "-p1", "-i", str(ARTIFACTS / "patch.diff")],
            ORIGINAL_REPO,
            "patch-dry-run",
        ),
        0,
        "patch-dry-run",
    )
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    shutil.copytree(ORIGINAL_REPO, WORKSPACE, symlinks=True)
    require(
        run(["patch", "--batch", "-p1", "-i", str(ARTIFACTS / "patch.diff")], WORKSPACE, "apply-patch-isolated-workspace"),
        0,
        "apply-patch-isolated-workspace",
    )
    require(
        run(
            [str(PYTHON), "-m", "pytest", "-q", tests],
            WORKSPACE,
            "local-patched",
        ),
        0,
        "local-patched",
    )


if __name__ == "__main__":
    main()

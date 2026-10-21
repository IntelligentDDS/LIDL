import argparse
import ast
import hashlib
import json
import re
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger


ROOT = Path(__file__).resolve().parents[2]
RUNNER_DIR = ROOT / "scripts/reproduce"
FIX_REPO = ROOT / "datasets/Hydrangea/fix_repo"
ENVIRONMENTS = ROOT / "datasets/Hydrangea/environments"
INSTALL_LOG = ENVIRONMENTS / "install.log"


def evaluate_path(node: ast.expr, values: dict[str, Path | str]) -> Path | str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name) and node.id == "ROOT":
        return ROOT
    if isinstance(node, ast.Name) and node.id in values:
        return values[node.id]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left = evaluate_path(node.left, values)
        right = evaluate_path(node.right, values)
        return Path(left) / str(right)
    raise ValueError(f"Unsupported path expression: {ast.unparse(node)}")


def read_runner_environment(runner: Path) -> tuple[str, Path]:
    values: dict[str, Path | str] = {}
    tree = ast.parse(runner.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in {
            "INSTANCE_ID",
            "ENVIRONMENT",
            "PYTHON",
        }:
            continue
        values[target.id] = evaluate_path(node.value, values)

    instance_id = values.get("INSTANCE_ID")
    environment = values.get("ENVIRONMENT")
    python = values.get("PYTHON")
    if not isinstance(instance_id, str):
        raise ValueError(f"Missing INSTANCE_ID in {runner}")
    if environment is None and isinstance(python, Path):
        environment = python.parents[2]
    if not isinstance(environment, Path):
        raise ValueError(f"Missing environment path in {runner}")
    if ENVIRONMENTS not in environment.parents:
        raise ValueError(f"Environment is outside {ENVIRONMENTS}: {environment}")
    return instance_id, environment


def read_python_dependencies(instance_id: str) -> list[str]:
    dockerfile = FIX_REPO / instance_id / "Dockerfile"
    text = dockerfile.read_text(encoding="utf-8")
    logical_lines = re.sub(r"\\\s*\n", " ", text).splitlines()
    dependencies = []
    for line in logical_lines:
        if "pip install" not in line:
            continue
        arguments = line.split("pip install", 1)[1].split("&&", 1)[0]
        for argument in shlex.split(arguments):
            if argument.startswith("-"):
                continue
            dependencies.append(argument)
    test_source = (FIX_REPO / instance_id / "tests/test.py").read_text(encoding="utf-8")
    required_modules = {"numpy": "numpy==1.26.4", "yaml": "PyYAML==6.0.2", "tabulate": "tabulate==0.9.0"}
    requirements = ROOT / "datasets/Hydrangea/repo" / instance_id / "requirements.txt"
    if requirements.is_file() and "tabulate==0.9.0" in requirements.read_text() and "tabulate" not in {dependency_name(item) for item in dependencies}:
        dependencies.append("tabulate==0.9.0")
    declared = {dependency_name(item) for item in dependencies}
    for module, specification in required_modules.items():
        if re.search(rf"^\s*(?:from {module}(?:\.| import)|import {module}\b)", test_source, re.MULTILINE) and dependency_name(specification) not in declared:
            logger.warning(f"Adding missing test dependency {specification} for {instance_id}")
            dependencies.append(specification)
    if not dependencies:
        raise ValueError(f"No Python dependencies found in {dockerfile}")
    return dependencies


def dependency_name(specification: str) -> str:
    return re.split(r"[<>=!~\[]", specification, maxsplit=1)[0].lower()


def collect_environments(
    selected_instances: set[str] | None,
) -> list[tuple[Path, list[str], list[str]]]:
    grouped: dict[Path, dict[str, object]] = {}
    known_instances = set()
    for runner in sorted(RUNNER_DIR.glob("reproduce_Hydrangea_*.py")):
        instance_id, environment = read_runner_environment(runner)
        known_instances.add(instance_id)
        if selected_instances and instance_id not in selected_instances:
            continue
        group = grouped.setdefault(environment, {"instances": [], "dependencies": {}})
        instances = group["instances"]
        dependencies = group["dependencies"]
        assert isinstance(instances, list)
        assert isinstance(dependencies, dict)
        instances.append(instance_id)
        for specification in read_python_dependencies(instance_id):
            name = dependency_name(specification)
            previous = dependencies.get(name)
            if previous and previous != specification:
                raise ValueError(
                    f"Conflicting dependencies in {environment.name}: "
                    f"{previous} and {specification}"
                )
            dependencies[name] = specification

    unknown = (selected_instances or set()) - known_instances
    if unknown:
        raise ValueError(f"Unknown instance IDs: {sorted(unknown)}")

    result = []
    for environment, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        instances = group["instances"]
        dependencies = group["dependencies"]
        assert isinstance(instances, list)
        assert isinstance(dependencies, dict)
        result.append((environment, sorted(instances), sorted(dependencies.values())))
    return result


def append_command_log(
    command: list[str], result: subprocess.CompletedProcess[str], environment: Path
) -> None:
    INSTALL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with INSTALL_LOG.open("a", encoding="utf-8") as file:
        file.write(
            f"\n{'=' * 80}\n"
            f"timestamp: {datetime.now(timezone.utc).isoformat()}\n"
            f"environment: {environment}\n"
            f"command: {subprocess.list2cmdline(command)}\n"
            f"returncode: {result.returncode}\n"
            f"stdout:\n{result.stdout}"
            f"stderr:\n{result.stderr}"
        )


def run(command: list[str], environment: Path) -> None:
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    append_command_log(command, result, environment)
    if result.returncode != 0:
        logger.error(
            f"Command failed for {environment.name}: "
            f"{subprocess.list2cmdline(command)}"
        )
        raise RuntimeError(f"Environment installation failed: {environment.name}")


def environment_digest(instances: list[str], dependencies: list[str], python: str) -> str:
    payload = json.dumps(
        {"instances": instances, "dependencies": dependencies, "python": python},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prepare_test_resources(environment, instances):
    if not set(instances) & {"AgentJ-WR__Private-GPT-Flask-0", "REFRIED-BoolEANS__privatgpt-gui-0"}:
        return
    data_path = environment / "nltk_data"
    code = (
        "import nltk; "
        f"nltk.data.path.insert(0, {str(data_path)!r}); "
        f"assert nltk.download('punkt_tab', download_dir={str(data_path)!r}, raise_on_error=True); "
        "nltk.data.find('tokenizers/punkt_tab/english/'); "
        f"assert nltk.download('averaged_perceptron_tagger_eng', download_dir={str(data_path)!r}, raise_on_error=True); "
        "nltk.data.find('taggers/averaged_perceptron_tagger_eng/'); "
        "nltk.pos_tag(nltk.word_tokenize('Verify the document parser.'))"
    )
    run([str(environment / ".venv/bin/python"), "-c", code], environment)


def install_environment(
    environment: Path,
    instances: list[str],
    dependencies: list[str],
    python: str,
    recreate: bool,
) -> str:
    virtualenv = environment / ".venv"
    marker = environment / ".reproduce-environment.json"
    digest = environment_digest(instances, dependencies, python)
    if marker.exists() and virtualenv.joinpath("bin/python").exists() and not recreate:
        state = json.loads(marker.read_text(encoding="utf-8"))
        if state.get("digest") == digest:
            logger.info(
                f"Skipping ready environment {environment.name}: "
                f"instances={instances[:2]}"
            )
            prepare_test_resources(environment, instances)
            return "skipped"

    if recreate and virtualenv.exists():
        logger.warning(f"Recreating environment {environment.name}")
        shutil.rmtree(virtualenv)

    environment.mkdir(parents=True, exist_ok=True)
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv is not available on PATH")
    if not virtualenv.joinpath("bin/python").exists():
        run([uv, "venv", "--python", python, str(virtualenv)], environment)
    run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(virtualenv / "bin/python"),
            *dependencies,
        ],
        environment,
    )
    prepare_test_resources(environment, instances)
    marker.write_text(
        json.dumps(
            {
                "digest": digest,
                "python": python,
                "instances": instances,
                "dependencies": dependencies,
                "installed_at": datetime.now(timezone.utc).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    logger.info(
        f"Installed environment {environment.name}: "
        f"instances={instances[:2]}, dependencies={dependencies[:2]}"
    )
    return "installed"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install shared Python environments for Hydrangea reproduction."
    )
    parser.add_argument(
        "--instance-id",
        action="append",
        help="Install only the environment for this instance; repeat for multiple instances.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Install only the first N shared environments.",
    )
    parser.add_argument(
        "--python",
        default="3.10",
        help="Python version or executable passed to uv venv.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate selected .venv directories.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List selected environments and dependencies without installing.",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    environments = collect_environments(set(args.instance_id or []) or None)
    if args.limit is not None:
        environments = environments[: args.limit]
    logger.info(
        f"Selected {len(environments)} shared environments: "
        f"{[environment.name for environment, _, _ in environments[:2]]}"
    )
    if args.dry_run:
        for environment, instances, dependencies in environments:
            logger.info(
                f"Would install {environment.name}: "
                f"instances={instances[:2]}, dependencies={dependencies[:2]}"
            )
        return

    installed = 0
    skipped = 0
    for environment, instances, dependencies in environments:
        try:
            status = install_environment(
                environment,
                instances,
                dependencies,
                args.python,
                args.recreate,
            )
        except Exception as error:
            logger.error(f"Failed to install {environment.name}: {error}")
            raise
        installed += status == "installed"
        skipped += status == "skipped"
    logger.info(f"Environment installation complete: installed={installed}, skipped={skipped}")


if __name__ == "__main__":
    main()

import argparse
import hashlib
import json
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
TAGS = ["agixt_1369", "crewai_1753", "autogen_3361", "autogen_4785", "autogen_5007", "camel_1145", "camel_1614", "chatdev_413", "chatdev_465", "metagpt_1313"]
HARNESS_FILES = {"reproduce.py": "/opt/reproduce.py", "run_test_entrypoint.sh": "/usr/local/bin/run_test_entrypoint.sh"}


def save_json(path, data):
    temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def run(command, directory, timeout=3600, required=True):
    started = datetime.now(timezone.utc).isoformat()
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        record = {"command": command, "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
                  "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except subprocess.TimeoutExpired as error:
        logger.error(f"Image command timed out: {command}: {error}")
        record = {"command": command, "started_at": started, "timeout": timeout,
                  "stdout": error.stdout.decode(errors="replace") if isinstance(error.stdout, bytes) else error.stdout,
                  "stderr": error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else error.stderr}
        with (directory / "commands.jsonl").open("a") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        raise
    with (directory / "commands.jsonl").open("a") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
    if required and result.returncode:
        raise RuntimeError(f"Image command failed: {command}; see {directory / 'commands.jsonl'}")
    return result


def inspect_image(reference, directory):
    result = run(["docker", "image", "inspect", reference], directory, required=False)
    return json.loads(result.stdout)[0] if result.returncode == 0 else None


def prepare(tag, args):
    directory = args.output_dir / tag
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    target = f"{args.image_prefix}:{tag}-amd64"
    if manifest_path.is_file() and not args.rebuild:
        manifest = json.loads(manifest_path.read_text())
        current = inspect_image(target, directory)
        if manifest.get("status") == "ready" and current and current["Id"] == manifest["image_id"]:
            logger.info(f"Verified image already prepared: {tag}")
            return tag, target
    source = f"alfin06/agentissue-bench:{tag}"
    source_file = directory / "source_image.json"
    source_metadata = json.loads(source_file.read_text()) if source_file.is_file() else None
    if source_metadata:
        reference = source_metadata.get("RepoDigests", [source_metadata["Id"]])[0]
    else:
        reference = source
    original = inspect_image(reference, directory)
    if original is None:
        command = ["docker", "pull"]
        if tag in {"crewai_1753", "agixt_1369"}:
            command += ["--platform", "linux/amd64"]
        run(command + [reference], directory)
        original = inspect_image(reference, directory)
    if original is None:
        raise ValueError(f"Cannot inspect source image: {reference}")
    save_json(source_file, original)
    if tag in {"crewai_1753", "agixt_1369"}:
        if original["Architecture"] != "amd64":
            raise ValueError(f"Expected AMD64 base for {tag}, got {original['Architecture']}")
        base = original["Id"]
    else:
        base = args.base_image
        if inspect_image(base, directory) is None:
            run(["docker", "pull", "--platform", "linux/amd64", base], directory)
        base = inspect_image(base, directory)["Id"]
        container = f"lidl-image-extract-{tag}-{uuid.uuid4().hex[:8]}"
        run(["docker", "create", "--name", container, original["Id"]], directory)
        try:
            for name, source_path in HARNESS_FILES.items():
                run(["docker", "cp", f"{container}:{source_path}", str(directory / name)], directory)
        finally:
            run(["docker", "rm", container], directory)
    if tag == "crewai_1753":
        run(["docker", "tag", base, target], directory)
        dockerfile = None
    elif tag == "agixt_1369":
        dockerfile = f"FROM {base}\nRUN apt-get update && apt-get install -y --no-install-recommends patch && rm -rf /var/lib/apt/lists/*\n"
    else:
        environment = dict(value.split("=", 1) for value in original["Config"]["Env"])
        dockerfile = (
            f"FROM {base}\nWORKDIR /app\n"
            "RUN apt-get update && apt-get install -y --no-install-recommends git build-essential cmake gcc g++ curl patch && rm -rf /var/lib/apt/lists/*\n"
            f"ENV BUGGY_COMMIT={environment['BUGGY_COMMIT']} FIXED_COMMIT={environment['FIXED_COMMIT']} SKIP_LLAMA_CPP_PYTHON=1\n"
            "COPY reproduce.py /opt/reproduce.py\nCOPY run_test_entrypoint.sh /usr/local/bin/run_test_entrypoint.sh\n"
            "RUN chmod +x /usr/local/bin/run_test_entrypoint.sh\n"
            'ENTRYPOINT ["/usr/local/bin/run_test_entrypoint.sh"]\nCMD ["help"]\n'
        )
    if dockerfile:
        (directory / "Dockerfile").write_text(dockerfile)
        run(["docker", "build", "--platform", "linux/amd64", "-t", target, str(directory)], directory)
    image = inspect_image(target, directory)
    if not image or image["Architecture"] != "amd64":
        raise ValueError(f"Built image architecture verification failed: {tag}")
    run(["docker", "run", "--rm", "--entrypoint", "bash", target, "-c",
         "command -v patch && python --version && test -x /usr/local/bin/run_test_entrypoint.sh"], directory, timeout=120)
    hashes = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
              for name in [*HARNESS_FILES, "Dockerfile"] if (directory / name).is_file()}
    save_json(manifest_path, {"status": "ready", "source_image_id": original["Id"], "source_reference": reference,
                              "base_image_id": base, "image": target, "image_id": image["Id"], "architecture": image["Architecture"],
                              "file_sha256": hashes, "validation": "startup_and_tools_only", "prepared_at": datetime.now(timezone.utc).isoformat()})
    logger.info(f"Prepared {tag}: {target}; benchmark tests remain to be evaluated")
    return tag, target


def main():
    parser = argparse.ArgumentParser(description="Prepare native repair benchmark images without changing tests or repair outputs")
    parser.add_argument("--image", action="append", choices=TAGS)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--base-image", default="python:3.10.18-slim-trixie")
    parser.add_argument("--image-prefix", default="lidl-agentissue-repro")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/repair_image_preparation")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    if args.max_workers < 1:
        parser.error("--max-workers must be positive")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mapping_path = args.output_dir / "image_map.json"
    mapping = json.loads(mapping_path.read_text()) if mapping_path.exists() else {}
    failures = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(prepare, tag, args): tag for tag in dict.fromkeys(args.image or TAGS)}
        for future in as_completed(futures):
            tag = futures[future]
            try:
                name, image = future.result()
                mapping[name] = image
                save_json(mapping_path, mapping)
            except Exception as error:
                logger.error(f"Image preparation failed for {tag}: {error}")
                failures[tag] = str(error)
    save_json(args.output_dir / "failures.json", failures)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())

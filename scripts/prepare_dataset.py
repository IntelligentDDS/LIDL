import argparse
import csv
import hashlib
import io
import json
import logging
import re
import shutil
import stat
import zipfile
from contextlib import ExitStack
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)
DATASETS = {"Hydrangea": "hydrangea_defect.json", "AgentIssue-Bench": "agent_issue.json"}


def save(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Refusing to replace different existing file: {path}")
        return
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def safe_path(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe asset path: {relative}")
    destination = root / path
    if not destination.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Asset escapes destination: {relative}")
    return destination


def run(command):
    subprocess.run(command, check=True, timeout=900)


def inspect_zip(archive, rows):
    members = {}
    for info in archive.infolist():
        name = info.filename
        if "\\" in name or name.startswith("/") or ".." in Path(name).parts:
            raise ValueError(f"Unsafe ZIP entry: {name}")
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode))):
            raise ValueError(f"Unsupported ZIP link or special file: {name}; package regular source files")
        normalized = Path(name).as_posix()
        if normalized in members:
            raise ValueError(f"Duplicate ZIP entry: {name}")
        members[normalized] = info
    for dataset, filename in DATASETS.items():
        path = f"datasets/{dataset}/{filename}"
        if path not in members:
            raise ValueError(f"ZIP must preserve datasets/ prefix and include {path}")
        records = json.loads(archive.read(members[path]))
        indexed = {item["instance_id"]: item for item in records}
        if len(indexed) != len(records):
            raise ValueError(f"Duplicate instances in ZIP metadata: {path}")
        for row in rows:
            if row["dataset"] == dataset and indexed.get(row["instance_id"]) != json.loads(row["instance_json"]):
                raise ValueError(f"ZIP metadata differs from CSV: {dataset}/{row['instance_id']}")
    return members


def extract_zip_repository(archive, members, row, destination):
    prefix = f"datasets/{row['dataset']}/repo/{row['instance_id']}/"
    selected = [(name[len(prefix):], info) for name, info in members.items() if name.startswith(prefix) and not info.is_dir()]
    if not selected:
        raise ValueError(f"ZIP has no source files for {prefix}")
    for relative, info in selected:
        target = safe_path(destination, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as source, target.open("wb") as output:
            shutil.copyfileobj(source, output)
        permissions = (info.external_attr >> 16) & 0o777
        if permissions:
            target.chmod(permissions)


def prepare_repository(row, root, local_repos, archive=None, members=None, zip_hash=None):
    dataset, iid, commit = row["dataset"], row["instance_id"], row["base_commit"]
    destination = root / "datasets" / dataset / "repo" / iid
    marker = root / "datasets/.prepare" / dataset / f"{iid}.json"
    identity = {"repository_url": row["repository_url"], "base_commit": commit}
    if archive is not None:
        identity["zip_sha256"] = zip_hash
    if destination.exists():
        if marker.is_file() and json.loads(marker.read_text()) == identity:
            logger.info("Repository already prepared: %s", iid)
            return
        raise ValueError(f"Existing repository has no matching checkpoint: {destination}")
    if archive is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".prepare-", dir=destination.parent) as temporary:
            checkout = Path(temporary) / "repo"
            checkout.mkdir()
            extract_zip_repository(archive, members, row, checkout)
            save(marker, json.dumps(identity, sort_keys=True).encode())
            checkout.rename(destination)
        logger.info("Prepared ZIP snapshot: %s @ %s (commit from matching metadata)", iid, commit)
        return
    cache = root / "datasets/.prepare/git" / hashlib.sha256(row["repository_url"].encode()).hexdigest()
    if not cache.exists():
        run(["git", "init", "--bare", str(cache)])
    available = subprocess.run(["git", "--git-dir", str(cache), "cat-file", "-e", f"{commit}^{{commit}}"], capture_output=True)
    if available.returncode:
        source = str(local_repos / dataset / "repo" / iid) if local_repos else row["repository_url"]
        run(["git", "--git-dir", str(cache), "fetch", "--depth=1", source, commit])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=destination.parent) as temporary:
        temporary = Path(temporary)
        archive = temporary / "repository.tar"
        run(["git", "--git-dir", str(cache), "archive", "--format=tar", f"--output={archive}", commit])
        checkout = temporary / "repo"
        checkout.mkdir()
        with tarfile.open(archive) as tar:
            tar.extractall(checkout, filter="data")
        save(marker, json.dumps(identity, sort_keys=True).encode())
        checkout.rename(destination)
    logger.info("Prepared repository: %s @ %s", iid, commit)


def main():
    parser = argparse.ArgumentParser(description="Restore LIDL dataset files and exact-commit repositories from one CSV")
    parser.add_argument("--csv", type=Path, default=Path(__file__).resolve().parents[1] / "datasets/lidl_dataset.csv")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--limit", type=int, help="Validate first N instances per dataset in a separate destination")
    parser.add_argument("--metadata-only", action="store_true", help="Restore metadata and assets without downloading repositories")
    sources = parser.add_mutually_exclusive_group()
    sources.add_argument("--local-repos", type=Path, help="Fetch commits from existing local Git repositories instead of GitHub")
    sources.add_argument("--repo-zip", type=Path, help="Import downloaded ZIP containing datasets/<dataset>/repo and dataset JSON files; no Git/network required")
    parser.add_argument("--zip-sha256", help="Expected SHA-256 supplied by the ZIP publisher")
    parser.add_argument("--install-environments", action="store_true", help="Run the embedded Hydrangea installer; requires uv and loguru")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.zip_sha256 and (not args.repo_zip or not re.fullmatch(r"[0-9a-fA-F]{64}", args.zip_sha256)):
        parser.error("--zip-sha256 requires --repo-zip and a 64-digit hexadecimal checksum")
    if args.metadata_only and (args.repo_zip or args.install_environments):
        parser.error("--metadata-only cannot be combined with ZIP import or environment installation")
    if not hasattr(tarfile, "data_filter") and not args.repo_zip:
        parser.error("Use Python 3.11.8+ with safe tar extraction support")
    csv.field_size_limit(sys.maxsize)
    with args.csv.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    keys = set()
    for row in rows:
        key = (row["dataset"], row["instance_id"])
        if row["dataset"] not in DATASETS or key in keys:
            raise ValueError(f"Invalid or duplicate instance: {key}")
        keys.add(key)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", row["instance_id"]) or not re.fullmatch(r"[0-9a-f]{40}", row["base_commit"]):
            raise ValueError(f"Invalid instance ID or commit: {key}")
        if not re.fullmatch(r"https://github.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?", row["repository_url"]):
            raise ValueError(f"Invalid repository URL: {key}")
        instance = json.loads(row["instance_json"])
        if instance["instance_id"] != row["instance_id"] or instance["base_commit"] != row["base_commit"]:
            raise ValueError(f"Inconsistent metadata: {key}")
    selected = []
    for dataset in DATASETS:
        group = [r for r in rows if r["dataset"] == dataset]
        if args.limit is not None:
            logger.warning("Validation restricts %s to first %s instances", dataset, args.limit)
            group = group[:args.limit]
        selected.extend(group)
    logger.info("Selected %s instances; first two: %s", len(selected), [(r['dataset'], r['instance_id']) for r in selected[:2]])
    root = args.root.resolve()
    with ExitStack() as stack:
        archive, members, zip_hash = None, None, None
        if args.repo_zip:
            source = stack.enter_context(args.repo_zip.open("rb"))
            zip_hash = hashlib.file_digest(source, "sha256").hexdigest()
            if args.zip_sha256 and zip_hash != args.zip_sha256.lower():
                raise ValueError("ZIP SHA-256 does not match publisher checksum")
            source.seek(0)
            archive = stack.enter_context(zipfile.ZipFile(source))
            members = inspect_zip(archive, rows)
            logger.info("ZIP SHA-256: %s; source bytes are trusted from this archive, not verified against GitHub", zip_hash)
        for row in selected:
            assets = json.loads(row["assets_json"])
            for relative, content in assets.items():
                save(safe_path(root, relative), content.encode("utf-8"))
            if not args.metadata_only:
                prepare_repository(row, root, args.local_repos.resolve() if args.local_repos else None, archive, members, zip_hash)
    for dataset, filename in DATASETS.items():
        instances = [json.loads(r["instance_json"]) for r in selected if r["dataset"] == dataset]
        save(root / "datasets" / dataset / filename, (json.dumps(instances, ensure_ascii=False, indent=2) + "\n").encode())
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=["benchmark", "instance_id", "defect_category"])
    writer.writeheader()
    writer.writerows({"benchmark": r["dataset"], "instance_id": r["instance_id"], "defect_category": r["defect_category"]} for r in selected)
    save(root / "scripts/classification_agreed.csv", buffer.getvalue().encode())
    if args.install_environments:
        if args.metadata_only:
            raise ValueError("Environment installation requires prepared repositories")
        command = [sys.executable, str(root / "scripts/reproduce/install_Hydrangea_environments.py")]
        for r in selected:
            if r["dataset"] == "Hydrangea":
                command.extend(["--instance-id", r["instance_id"]])
        run(command)
    logger.info("Finished: %s; AgentIssue test environments still require upstream benchmark Docker images", root)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        main()
    except Exception as error:
        logger.error("Dataset preparation failed: %s", error)
        raise SystemExit(1)

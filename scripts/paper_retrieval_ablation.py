import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import monotonic

from run_main_single import build_analyzer_evidence, normalize_line_rankings, normalize_relation_edges, select_covered_lines

import networkx as nx
import pandas as pd
from loguru import logger

from codekg_retrieval import SOURCE_EXTENSIONS, build_entity_index, retrieve_files
from paper_results import (
    DATASETS, parse_dataset_patch, functions_for_lines, functions_for_ranked_lines,
    files_for_ranked_lines, limit_ranked_lines, normalize_path, rank_metrics,
)

ROOT = Path(__file__).resolve().parents[1]
METRICS = [f"{level} {metric}" for level in ("File", "Function", "Line") for metric in ("Top-1", "Top-3", "Top-5", "MAP", "MRR")]
VARIANTS = ["retrieval_full", "retrieval_without_E", "retrieval_without_I", "retrieval_without_R", "without_V", "full"]
ADDED_VARIANTS = ["retrieval_without_TEXTFILE", "retrieval_without_BM25", "retrieval_without_PageRank"]
TABLE_VARIANTS = [*VARIANTS[1:4], *ADDED_VARIANTS, "without_V", "full"]
TABLE_NAMES = {
    "retrieval_without_E": "w/o (V & Re)",
    "retrieval_without_I": "w/o (V & Ri)",
    "retrieval_without_R": "w/o (V & Rr)",
    "retrieval_without_TEXTFILE": "w/o (V & TEXTFILE)",
    "retrieval_without_BM25": "w/o (V & BM25)",
    "retrieval_without_PageRank": "w/o (V & PageRank)",
    "without_V": "w/o V",
    "full": "LIDL",
}


def evaluate(lines, repo, instance):
    locations = limit_ranked_lines([(normalize_path(item["path"], instance["instance_id"]), item["line"]) for item in lines])
    files, line_map = parse_dataset_patch(instance)
    gold_lines = [(path, line) for path, numbers in line_map.items() for line in numbers]
    gold_functions = functions_for_lines(repo, line_map)
    predictions = [files_for_ranked_lines(locations), functions_for_ranked_lines(repo, locations), locations]
    metrics = {}
    for level, gold, predicted in zip(("File", "Function", "Line"), (files, gold_functions, gold_lines), predictions):
        result = rank_metrics(gold, predicted, (1, 3, 5))
        for cutoff in (1, 3, 5):
            metrics[f"{level} Top-{cutoff}"] = result["hits"][cutoff] if result else 0.0
        metrics[f"{level} MAP"] = result["map"] if result else 0.0
        metrics[f"{level} MRR"] = result["mrr"] if result else 0.0
    return metrics


def write_json_atomic(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, separators=(",", ":"))
        file.write("\n")
    temporary.replace(path)


def compact_artifact(details):
    compact = dict(details)
    if "line_rankings" in details and "relation_edges" in details:
        statements, lines = normalize_line_rankings(details["line_rankings"])
        fields = {"statement_start", "statement_end", "statement_kind", "statement_code", "hypothesis_coverages", "hypothesis_matches"}
        by_id = {item["statement_id"]: item for item in statements}
        for original, line in zip(details["line_rankings"], lines):
            line["absent_statement_fields"] = sorted(fields - original.keys())
            statement = by_id[line["statement_id"]]
            if any(statement.get(key) != original.get(key) for key in fields):
                raise ValueError("Statement normalization would merge different payloads")
        compact["statements"] = statements
        compact["line_rankings"] = lines
    if "relation_edges" in details:
        fields, edges = normalize_relation_edges(details["relation_edges"])
        if any(set(item) != set(fields) for item in details["relation_edges"]):
            logger.warning("Keeping original relation dictionaries to preserve extra or missing fields")
        else:
            compact["relation_edge_fields"] = fields
            compact["relation_edges"] = edges
    compact["storage_schema"] = "lossless_normalized_v1"
    return compact


def shortlist_case(dataset, instance, model, output_dir, source_dir, resume=False):
    iid = instance["instance_id"]
    prior = source_dir / "cases" / dataset["name"] / iid
    source_rows = json.loads((prior / "metrics.json").read_text())
    if {row["variant"] for row in source_rows} != set(VARIANTS + ADDED_VARIANTS):
        raise ValueError(f"Incomplete retrieval inputs: {prior}")
    source_rows.sort(key=lambda row: (VARIANTS + ADDED_VARIANTS).index(row["variant"]))
    destination = output_dir / "cases" / dataset["name"] / iid
    destination.mkdir(parents=True, exist_ok=True)
    input_paths = [prior / "metrics.json", prior / "fingerprint.json", Path(__file__),
                   ROOT / "scripts/run_main_single.py", ROOT / "scripts/paper_results.py"]
    fingerprint = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in input_paths}
    fingerprint["instance"] = hashlib.sha256(json.dumps(instance, sort_keys=True).encode()).hexdigest()
    source_fingerprint = json.loads((prior / "fingerprint.json").read_text())
    for path in (ROOT / "scripts/run_main_single.py", ROOT / "scripts/paper_results.py", ROOT / "scripts/codekg_retrieval.py"):
        if source_fingerprint.get(str(path)) != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f"Retrieval source code version differs: {path}")
    checkpoint = destination / "checkpoint.json"
    records = []
    if resume and checkpoint.is_file():
        saved = json.loads(checkpoint.read_text())
        if saved["fingerprint"] == fingerprint:
            records = saved["records"]
    completed = {row["variant"] for row in records}
    repo = dataset["repo_base"] / iid
    for source_row in source_rows:
        variant = source_row["variant"]
        if variant in completed:
            continue
        started = monotonic()
        if variant == "full":
            record = dict(source_row)
        elif variant == "without_V":
            baseline = next(row for row in records if row["variant"] == "retrieval_full")
            record = {**baseline, "variant": variant}
        else:
            artifact = Path(source_row["artifact"])
            data = json.loads(artifact.read_text())
            lines = data["line_rankings"]
            if "statements" in data:
                statements = {item["statement_id"]: item for item in data["statements"]}
                lines = [{**{k: v for k, v in statements[item["statement_id"]].items()
                              if k != "statement_id" and k not in item.get("absent_statement_fields", [])},
                          **item} for item in lines]
            before = evaluate(lines, repo, instance)
            if any(abs(before[key] - source_row[key]) > 1e-10 for key in METRICS):
                raise ValueError(f"Restored raw metrics differ: {iid}/{variant}")
            ordered_paths = [item["path"] for item in data["candidate_context"]]
            # The production selector does not access code_graph; all required context is in the rankings.
            shortlist = select_covered_lines(lines, ordered_paths, None, 100)
            selected = {(item["path"], item["line"]) for item in shortlist}
            ranked_anchors = [item for item in lines if (item["path"], item["line"]) in selected]
            if len(ranked_anchors) != len(selected):
                raise ValueError(f"Duplicate or missing shortlist anchors: {iid}/{variant}")
            path = destination / f"{variant}.json"
            write_json_atomic(path, {
                "source_artifact": str(artifact), "statement_budget": 100,
                "prediction_rule": "Retain selected statement anchors in original retrieval order; evaluate up to 10 unique anchor lines, without range expansion or LLM reranking.",
                "shortlist": shortlist,
                "ranked_locations": [{"path": item["path"], "line": item["line"]} for item in ranked_anchors],
            })
            record = {**source_row, **evaluate(ranked_anchors, repo, instance), "artifact": str(path)}
            del data, lines, ranked_anchors, shortlist
        records.append(record)
        write_json_atomic(checkpoint, {"fingerprint": fingerprint, "records": records})
        logger.info(f"Completed shortlist variant {dataset['name']}/{iid}/{variant}: seconds={monotonic() - started:.2f}")
    write_json_atomic(destination / "metrics.json", records)
    logger.info(f"Completed shortlist case {dataset['name']}/{iid}; first two: {records[:2]}")
    return records


def process_case(dataset, instance, model, output_dir, variants=VARIANTS, resume=False, reuse_dir=None):
    iid = instance["instance_id"]
    source = ROOT / "results/results_main" / dataset["name"] / model / iid
    destination = output_dir / "cases" / dataset["name"] / iid
    destination.mkdir(parents=True, exist_ok=True)
    fingerprint = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [source / name for name in ("codekg.json", "annotation.json", "analyzer.json", "retrieval.json", "output.json")]
        + [Path(__file__), ROOT / "scripts/codekg_retrieval.py", ROOT / "scripts/paper_results.py", ROOT / "scripts/run_main_single.py"]
    }
    fingerprint["instance"] = hashlib.sha256(json.dumps(instance, sort_keys=True).encode()).hexdigest()
    metrics_path = destination / "metrics.json"
    fingerprint_path = destination / "fingerprint.json"
    records = []
    if resume and metrics_path.is_file() and fingerprint_path.is_file():
        previous_fingerprint = json.loads(fingerprint_path.read_text())
        if previous_fingerprint == fingerprint:
            records = [row for row in json.loads(metrics_path.read_text())
                       if row["variant"] in variants and Path(row["artifact"]).is_file()]
            if [row["variant"] for row in records] == list(variants):
                logger.info(f"Resuming verified offline case {dataset['name']}/{iid}")
                return records
    if reuse_dir is not None and not records:
        prior = reuse_dir / "cases" / dataset["name"] / iid
        prior_fingerprint = json.loads((prior / "fingerprint.json").read_text())
        if prior_fingerprint.get(str(Path(__file__))) != "849c273c62dd73a5e03fd294e059c523c4c334efbf838c0b36b98e6a5aba53b4":
            raise ValueError(f"Unverified source version for unchanged variants: {prior}")
        source_keys = set(fingerprint) - {str(Path(__file__))}
        if any(prior_fingerprint.get(key) != fingerprint[key] for key in source_keys):
            raise ValueError(f"Unchanged-variant source fingerprint mismatch: {dataset['name']}/{iid}")
        records = [row for row in json.loads((prior / "metrics.json").read_text())
                   if row["variant"] in variants and row["variant"] != "retrieval_without_BM25"]
        if len(records) != len(variants) - 1 or any(not Path(row["artifact"]).is_file() for row in records):
            raise ValueError(f"Missing unchanged variants: {dataset['name']}/{iid}")
        logger.info(f"Reusing {len(records)} unchanged variants from {prior}")
    write_json_atomic(metrics_path, records)
    write_json_atomic(fingerprint_path, fingerprint)
    completed_variants = {row["variant"] for row in records}
    graph_data = json.loads((source / "codekg.json").read_text())
    graph = nx.MultiDiGraph()
    for node in graph_data["nodes"]:
        graph.add_node(node["id"], **{k: v for k, v in node.items() if k != "id"})
    for edge in graph_data["edges"]:
        graph.add_edge(edge["source"], edge["target"], key=edge.get("key"), **{k: v for k, v in edge.items() if k not in {"source", "target", "key"}})
    annotation = json.loads((source / "annotation.json").read_text())
    for node, attributes in graph.nodes(data=True):
        attributes["llm_role_annotations"] = annotation["validated_annotations"].get(node, [])
        attributes["llm_roles"] = list(dict.fromkeys(item["role"] for item in attributes["llm_role_annotations"]))
    saved_analyzer = json.loads((source / "analyzer.json").read_text())
    analyzer = build_analyzer_evidence(graph, instance["problem_statement"], annotation, saved_analyzer["raw_result"])
    for field in ("E", "I", "I_file_evidence", "R", "fault_hypotheses"):
        if analyzer[field] != saved_analyzer[field]:
            raise ValueError(f"Rebuilt full analyzer differs from saved evidence: {dataset['name']}/{iid}/{field}")
    saved = json.loads((source / "retrieval.json").read_text())
    saved_full = saved.get("variants", {}).get("full", saved)
    config = saved_full.get("configuration", saved.get("common", {}).get("configuration", {}))
    index = build_entity_index(graph)
    repo = dataset["repo_base"] / iid
    for variant in variants:
        if variant in completed_variants:
            logger.info(f"Resuming verified variant {dataset['name']}/{iid}/{variant}")
            continue
        started = monotonic()
        artifact = destination / f"{variant}.json"
        if variant == "without_V":
            lines = saved_full.get("line_rankings", [])
            details = {"source": str(source / "retrieval.json"), "line_rankings": lines}
        elif variant == "full":
            output = json.loads((source / "output.json").read_text())
            lines = output.get("ranked_locations", [])
            details = {"source": str(source / "output.json"), "output": output}
        else:
            channel = variant.rsplit("_", 1)[-1] if "without" in variant else None
            variant_graph, variant_index, variant_analyzer = graph, index, analyzer
            removed = set()
            if channel in {"TEXTFILE", "R"}:
                removed = {node for node, data in graph.nodes(data=True) if data.get("type") == "textfile"} if channel == "TEXTFILE" else set()
                variant_graph = graph.copy()
                variant_graph.remove_nodes_from(removed)
                kept_annotations = {} if channel == "R" else {
                    path: values for path, values in annotation["validated_annotations"].items() if path in variant_graph
                }
                patterns = {role: [] for role in annotation["pattern_updates"]}
                for node, attributes in variant_graph.nodes(data=True):
                    values = kept_annotations.get(node, [])
                    attributes["llm_role_annotations"] = values
                    attributes["llm_roles"] = list(dict.fromkeys(item["role"] for item in values))
                for values in kept_annotations.values():
                    for item in values:
                        keywords = patterns.setdefault(item["role"], [])
                        keywords.extend(word for word in item["keywords"] if word not in keywords)
                variant_annotation = {"validated_annotations": kept_annotations, "pattern_updates": patterns}
                variant_analyzer = build_analyzer_evidence(variant_graph, instance["problem_statement"], variant_annotation, analyzer["raw_result"])
                variant_index = build_entity_index(variant_graph)
                if channel == "R" and any(data.get("llm_role_annotations") or data.get("llm_roles") for _, data in variant_graph.nodes(data=True)):
                    raise ValueError(f"Annotation leaked into R ablation: {iid}")
            elif channel == "BM25":
                # Remove the independent issue retrieval channel, not BM25 used inside symptom inference.
                variant_index = None
            rankings = {
                "E": [] if channel == "E" else variant_analyzer["E"],
                "I": [] if channel == "I" else variant_analyzer["I"],
                "R": [] if channel == "R" else variant_analyzer["R"]["matched_files"],
                "I_evidence": {} if channel == "I" else variant_analyzer["I_file_evidence"],
            }
            details = retrieve_files(
                variant_graph, instance["problem_statement"],
                seed_count=config.get("seed_count", 30),
                candidate_count=config.get("candidate_count", 20),
                entity_budget=config.get("entity_budget", 40),
                entity_index=variant_index, use_direct=channel != "E", use_inference=True,
                use_roles=channel != "R", protected_file_paths=rankings["E"],
                analyzer_rankings=rankings,
                fault_hypotheses=[] if channel == "I" else analyzer.get("fault_hypotheses", []),
                use_bm25=channel != "BM25", use_pagerank=channel != "PageRank",
            )
            details["analyzer_rankings"] = rankings
            details["ablation_scope"] = "Deterministic retrieval replay with fixed raw LLM outputs and rebuilt affected evidence"
            lines = details["line_rankings"]
            if channel == "TEXTFILE":
                paths = {item["path"] for item in details["file_rankings"] + lines + details["candidate_context"]}
                if paths & removed or any(data.get("type") == "textfile" for _, data in variant_graph.nodes(data=True)):
                    raise ValueError(f"TEXTFILE leaked into retrieval: {iid}")
                details["removed_textfile_nodes"] = sorted(removed)
            if channel in {"BM25", "PageRank"}:
                if rankings["I"] != analyzer["I"] or rankings["I_evidence"] != analyzer["I_file_evidence"] or rankings["R"] != analyzer["R"]["matched_files"]:
                    raise ValueError(f"Independent retrieval ablation changed I/R evidence: {iid}/{channel}")
                source_name = "BM25" if channel == "BM25" else "CodeKG"
                if any(source_name in item["source_ranks"] for item in details["file_rankings"]):
                    raise ValueError(f"Disabled fusion source remains: {iid}/{channel}")
                score_field = "lexical_score" if channel == "BM25" else "symbol_graph_score"
                if any(item[score_field] != 0 for item in details["entity_rankings"]):
                    raise ValueError(f"Disabled entity score remains: {iid}/{channel}")
        metrics = evaluate(lines, repo, instance)
        computed = monotonic()
        write_json_atomic(artifact, compact_artifact(details))
        records.append({"dataset": dataset["name"], "instance_id": iid, "model": model, "variant": variant, **metrics, "artifact": str(artifact)})
        write_json_atomic(metrics_path, records)
        logger.info(f"Completed variant {dataset['name']}/{iid}/{variant}: compute_seconds={computed - started:.2f}, storage_seconds={monotonic() - computed:.2f}, artifact_bytes={artifact.stat().st_size}")
    records.sort(key=lambda row: list(variants).index(row["variant"]))
    write_json_atomic(metrics_path, records)
    logger.info(f"Completed offline ablation {dataset['name']}/{iid}; first two: {records[:2]}")
    return records


def write_table(summary, destination, model):
    names = {
        "retrieval_full": "Retrieval (all evidence)",
        "retrieval_without_E": r"Retrieval w/o $\mathcal{E}$",
        "retrieval_without_I": r"Retrieval w/o $\mathcal{I}$",
        "retrieval_without_R": r"Retrieval w/o $\mathcal{R}$",
        "without_V": r"LIDL w/o $\mathcal{V}$ (saved Retrieval)",
        "full": "LIDL (saved final output)",
    }
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Offline retrieval-evidence ablation and saved validation-stage comparison on GPT-5.6-Terra. No new LLM calls are made. Retrieval ablations are compared against recomputed full Retrieval, not final LIDL. Dashes indicate that variant-specific LLM costs were not measured.}",
             r"\label{tab:ablation}", r"\setlength{\tabcolsep}{2.5pt}", r"\renewcommand{\arraystretch}{1.12}",
             r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{ll*{5}{c}c*{5}{c}c*{5}{c}c*{3}{c}}", r"\toprule",
             r"Model & Method & \multicolumn{5}{c}{File} & & \multicolumn{5}{c}{Function} & & \multicolumn{5}{c}{Line} & & \multicolumn{3}{c}{Cost} \\",
             r"\cline{3-7}\cline{9-13}\cline{15-19}\cline{21-23}",
             r" & & Top-1 & Top-3 & Top-5 & MAP & MRR & & Top-1 & Top-3 & Top-5 & MAP & MRR & & Top-1 & Top-3 & Top-5 & MAP & MRR & & In (k) & Out (k) & \$ \\"]
    for group, variants in [("Retrieval evidence (offline recomputation)", VARIANTS[:4]), ("Validation stages (saved outputs)", VARIANTS[4:])]:
        lines.extend([r"\midrule", rf"\multicolumn{{23}}{{l}}{{{group}}} \\"])
        subset = summary[summary["variant"].isin(variants)]
        best = {column: subset[column].max() for column in METRICS}
        for variant in variants:
            row = subset[subset["variant"] == variant].iloc[0]
            values = []
            for column in METRICS:
                text = f"{row[column]:.4f}"
                values.append(rf"\textbf{{{text}}}" if abs(row[column] - best[column]) < 1e-10 else text)
            cells = [model, names[variant], *values[:5], "", *values[5:10], "", *values[10:15], "", "--", "--", "--"]
            lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}%", "}", r"\end{table*}"])
    (destination / "ablation.tex").write_text("\n".join(lines) + "\n")


def write_extended_table(summary, destination):
    columns = [f"{level} {metric}" for level in ("File", "Function", "Line") for metric in ("Top-1", "Top-3", "MAP")]
    selected = summary.set_index("variant").loc[TABLE_VARIANTS]
    rows = []
    markdown = ["| Method | " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * 10) + " |"]
    latex = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Ablation study of LIDL components. $R_e$: direct extraction. $R_i$: symptom-based inference. $R_r$: annotation-based retrieval. V: validator. Backbone LLM: GPT-5.6-Terra.}",
             r"\label{tab:ablation}", r"\begin{tabular}{l*{9}{c}}", r"\toprule",
             r"Method & \multicolumn{3}{c}{File} & \multicolumn{3}{c}{Function} & \multicolumn{3}{c}{Line} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
             " & " + " & ".join(["Top-1", "Top-3", "MAP"] * 3) + r" \\", r"\midrule"]
    for variant, row in selected.iterrows():
        values = [f"{row[column]:.4f}" for column in columns]
        name = TABLE_NAMES[variant]
        rows.append({"Method": name, **{column: row[column] for column in columns}})
        markdown.append("| " + " | ".join([name, *values]) + " |")
        latex_name = name.replace("&", r"\&")
        for label in ("Re", "Ri", "Rr"):
            latex_name = latex_name.replace(label, "$R_" + label[-1] + "$")
        latex.append(" & ".join([latex_name, *values]) + r" \\")
    latex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    pd.DataFrame(rows).to_csv(destination / "ablation.csv", index=False)
    (destination / "ablation.md").write_text("\n".join(markdown) + "\n")
    (destination / "ablation.tex").write_text("\n".join(latex) + "\n")


def final_code_filter(args):
    destination = args.output_dir.resolve() / "final_code_filter" / args.model
    if args.limit is not None:
        destination = destination / f"validation_limit{args.limit}"
    destination.mkdir(parents=True, exist_ok=True)
    protocol = {
        "scope": "Post-validation output filtering, not a rerun without non-code files.",
        "classification": "Keep paths whose lowercase suffix is in codekg_retrieval.SOURCE_EXTENSIONS; implementation-level code/text classification.",
        "source_extensions": sorted(SOURCE_EXTENSIONS), "model": args.model,
        "prediction_rule": "Normalize and deduplicate original final predictions, cap at 10, then remove text-classified locations without refill; retain ordering and all instances and ground truth.",
        "limit": args.limit,
    }
    write_json_atomic(destination / "protocol.json", protocol)
    rows, affected = [], []
    for dataset in DATASETS:
        instances = json.loads(dataset["dataset_json"].read_text())
        if args.limit is not None:
            logger.warning(f"Validating first {args.limit} real cases of {dataset['name']}")
            instances = instances[:args.limit]
        for instance in instances:
            iid = instance["instance_id"]
            source = ROOT / "results/results_main" / dataset["name"] / args.model / iid / "output.json"
            case = destination / "cases" / dataset["name"] / iid
            case.mkdir(parents=True, exist_ok=True)
            fingerprint = {"output": hashlib.sha256(source.read_bytes()).hexdigest(),
                           "instance": hashlib.sha256(json.dumps(instance, sort_keys=True).encode()).hexdigest(),
                           "script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           "metrics": hashlib.sha256((ROOT / "scripts/paper_results.py").read_bytes()).hexdigest(),
                           "extensions": sorted(SOURCE_EXTENSIONS)}
            checkpoint = case / "metrics.json"
            result = json.loads(checkpoint.read_text()) if args.resume and checkpoint.is_file() else None
            if result is None or result.get("fingerprint") != fingerprint:
                output = json.loads(source.read_text())
                locations = limit_ranked_lines([(normalize_path(item["path"], iid), item["line"]) for item in output["ranked_locations"]])
                full = [{"path": path, "line": line} for path, line in locations]
                filtered = [item for item in full if Path(item["path"]).suffix.lower() in SOURCE_EXTENSIONS]
                removed = [item for item in full if Path(item["path"]).suffix.lower() not in SOURCE_EXTENSIONS]
                records = [{"dataset": dataset["name"], "instance_id": iid, "model": args.model, "variant": variant,
                            **evaluate(predictions, dataset["repo_base"] / iid, instance)}
                           for variant, predictions in (("full", full), ("final_code_only", filtered))]
                result = {"fingerprint": fingerprint, "source": str(source), "full_predictions": full,
                          "filtered_predictions": filtered, "removed_predictions": removed, "records": records}
                write_json_atomic(checkpoint, result)
            rows.extend(result["records"])
            if result["removed_predictions"]:
                affected.append({"dataset": dataset["name"], "instance_id": iid, "removed_predictions": result["removed_predictions"]})
            if len(rows) <= 4:
                logger.info(f"Final output filtering real case {dataset['name']}/{iid}: {result}")
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["model", "variant"], sort=False)[METRICS].mean().reset_index()
    summary["instances"] = frame.groupby(["model", "variant"], sort=False).size().to_numpy()
    if args.limit is None:
        original = pd.read_csv(ROOT / "results/paper_results/ablation_retrieval_raw.csv")
        for row in rows:
            if row["variant"] != "full":
                continue
            prior = original[(original["dataset"] == row["dataset"]) & (original["instance_id"] == row["instance_id"]) & (original["model"] == row["model"]) & (original["variant"] == "full")]
            if len(prior) != 1 or any(abs(prior.iloc[0][metric] - row[metric]) > 1e-10 for metric in METRICS):
                raise ValueError(f"Full baseline mismatch: {row['dataset']}/{row['instance_id']}")
    frame.to_csv(destination / "instances.csv", index=False)
    summary.to_csv(destination / "summary.csv", index=False)
    write_json_atomic(destination / "affected_instances.json", affected)
    names = {"full": "LIDL (Full)", "final_code_only": "LIDL with post-validation code-only filtering"}
    output = destination if args.limit is not None else args.output_dir.resolve()
    markdown = ["# Post-validation Code-only Filtering", "", protocol["scope"], "", protocol["prediction_rule"], "",
                "| Method | " + " | ".join(METRICS) + " |", "| " + " | ".join(["---"] * (len(METRICS) + 1)) + " |"]
    for row in summary.to_dict("records"):
        markdown.append("| " + " | ".join([names[row["variant"]], *[f"{row[key]:.4f}" for key in METRICS]]) + " |")
    markdown.extend(["", f"Affected instances: {len(affected)}/{len(frame) // 2}; removed prediction lines: {sum(len(item['removed_predictions']) for item in affected)}.", "",
                     "Classification follows SOURCE_EXTENSIONS; all original ground truth is retained. This does not measure removal of non-code evidence from retrieval or validation.", ""])
    (output / "ablation_final_code_filter.md").write_text("\n".join(markdown))
    summary.to_csv(output / "ablation_final_code_filter.csv", index=False)
    logger.info(f"Final filtering complete: {summary.to_dict('records')}; affected={len(affected)}; output={output}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Offline E/I/R retrieval ablations and saved V comparison; no LLM calls")
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/paper_results")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--extended", action="store_true", help="Add TEXTFILE, BM25 and PageRank rows to ablation.md/csv/tex while preserving the original rows")
    parser.add_argument("--reuse-unchanged", type=Path, help="Reuse verified rebuilt-evidence variants except BM25 from this run directory")
    parser.add_argument("--shortlist-from", type=Path, help="Apply the production deterministic shortlist to saved retrieval variants without recomputing retrieval")
    parser.add_argument("--final-code-only", action="store_true", help="Evaluate code-only filtering of saved final predictions, without rerunning any agent")
    args = parser.parse_args()
    if args.final_code_only:
        if args.extended or args.shortlist_from or args.reuse_unchanged:
            parser.error("--final-code-only cannot combine with retrieval ablation modes")
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        return final_code_filter(args)
    if args.shortlist_from and (not args.extended or args.reuse_unchanged):
        parser.error("--shortlist-from requires --extended and cannot combine with --reuse-unchanged")
    if args.reuse_unchanged and not args.extended:
        parser.error("--reuse-unchanged requires --extended")
    if args.max_workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("Workers and limit must be positive")
    destination = args.output_dir.resolve()
    if args.extended:
        destination = destination / "ablation_runs" / args.model / ("post_shortlist" if args.shortlist_from else "independent_channels")
    if args.limit is not None:
        destination = destination / f"validation_limit{args.limit}"
    destination.mkdir(parents=True, exist_ok=True)
    variants = [*VARIANTS, *ADDED_VARIANTS] if args.extended else VARIANTS
    protocol = {
        "model": args.model, "variants": variants, "limit": args.limit,
        "scope": "Offline ablation with fixed raw LLM outputs, retained per-file annotations and rebuilt deterministic evidence; no new LLM calls or annotation-budget reallocation.",
        "TEXTFILE": "Remove textfile nodes, edges and annotations; rebuild pattern updates from retained annotations, E/I/R, corpus support, lexical index and graph propagation.",
        "BM25": "Disable independent issue-to-entity lexical retrieval, its seeds, fusion and entity scores; preserve symptom-to-file BM25 within I, I evidence, R, hypotheses and line matching.",
        "R": "Clear annotations, roles and pattern updates before indexing; disable R ranks, role seeds and scores.",
        "PageRank": "Disable both PageRank computations and all resulting file/entity scores; keep graph structure, direct and annotation evidence.",
    }
    if args.shortlist_from:
        protocol.update({"source_run": str(args.shortlist_from.resolve()), "stage": "after deterministic Analyzer shortlist, before LLM validation",
                         "statement_budget": 100, "prediction_rule": "Original retrieval order restricted to selected representative anchor lines, up to 10 unique predictions; no statement-range credit or LLM reranking; full LIDL unchanged.",
                         "prompt_feasibility": "Validator-specific token exclusions are not applied without V."})
    (destination / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    jobs = []
    rows = []
    for dataset in DATASETS:
        instances = json.loads(dataset["dataset_json"].read_text())
        if args.limit is not None:
            logger.warning(f"Validating first {args.limit} instances from {dataset['name']}")
            instances = instances[:args.limit]
        for instance in instances:
            jobs.append((dataset, instance))
    failures = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        if args.shortlist_from:
            futures = {executor.submit(shortlist_case, ds, inst, args.model, destination, args.shortlist_from.resolve(), args.resume): (ds, inst) for ds, inst in jobs}
        else:
            futures = {executor.submit(process_case, ds, inst, args.model, destination, variants, args.resume, args.reuse_unchanged): (ds, inst) for ds, inst in jobs}
        for future in as_completed(futures):
            ds, inst = futures[future]
            try:
                rows.extend(future.result())
            except Exception as error:
                logger.error(f"Offline computation failed {ds['name']}/{inst['instance_id']}: {error}")
                failures.append({"dataset": ds['name'], "instance_id": inst['instance_id'], "error": str(error)})
    (destination / "failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2) + "\n")
    if failures:
        logger.error("Incomplete offline calculation; no aggregate table produced, resume after resolving errors")
        return 1
    frame = pd.DataFrame(rows)
    frame.to_csv(destination / "ablation_retrieval_raw.csv", index=False)
    summary = frame.groupby(["model", "variant"], sort=False)[METRICS].mean().reset_index()
    summary["instances"] = frame.groupby(["model", "variant"], sort=False).size().to_numpy()
    summary.to_csv(destination / "ablation_retrieval_summary.csv", index=False)
    if args.extended:
        original = pd.read_csv(ROOT / "results/paper_results/ablation_retrieval_raw.csv")
        for row in rows:
            preserved = {"full"} if args.shortlist_from else {"retrieval_full", "without_V", "full"}
            if row["variant"] not in preserved:
                continue
            baseline = original[(original["dataset"] == row["dataset"]) & (original["instance_id"] == row["instance_id"]) & (original["model"] == row["model"]) & (original["variant"] == row["variant"])]
            if len(baseline) != 1 or any(abs(float(baseline.iloc[0][column]) - row[column]) > 1e-10 for column in METRICS):
                raise ValueError(f"Original ablation changed: {row['dataset']}/{row['instance_id']}/{row['variant']}")
        table_dir = destination if args.limit is not None else args.output_dir.resolve()
        write_extended_table(summary, table_dir)
    else:
        write_table(summary, destination, args.model)
    logger.info(f"Offline calculation complete: {summary.to_dict('records')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

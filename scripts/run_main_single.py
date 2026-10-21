import argparse
import hashlib
import json
import os
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import litellm
from dotenv import load_dotenv
from loguru import logger
from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator
from rank_bm25 import BM25Okapi

from codekg_retrieval import (
    FILE_TYPES,
    ROLE_PATTERNS,
    build_complete_codekg,
    build_entity_index,
    extract_direct_evidence,
    graph_to_dict,
    owning_file,
    repository_fingerprint,
    retrieve_files,
    tokenize,
)
from utils import atomic_json, call_llm, log_case, trace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = PROJECT_ROOT / "results" / "results_main" / "Hydrangea"
ANNOTATION_ROLES = tuple(ROLE_PATTERNS)
FULL_EVIDENCE = {"use_direct": True, "use_inference": True, "use_roles": True}
I_FIELDS = (
    "related_symbols",
    "faulty_behaviors",
    "retrieval_keywords",
    "failure_stages",
    "symptoms",
)
ANNOTATION_SYSTEM_PROMPT = """You are the semantic Annotation component of an LLM defect-localization system.
Your task is to classify every supplied repository file into zero or more roles: LLM_PROMPT, LLM_CALL, LLM_CONFIG, LLM_TOOL, and LLM_MEMORY.

# Annotation Rules
- Do classify each file independently from its supplied source.
- Do return concise semantic descriptions and exact literal keywords copied from the file for every classification.
- Do allow one file to have multiple roles.
- Do omit files unrelated to the five LLM workflow roles.
- Do not invent, normalize, or paraphrase keywords.
- Do return only one valid JSON object matching the requested schema."""

ANALYZER_SYSTEM_PROMPT = """You are the Defect Analyzer of an LLM application repository.
Your task is to infer defect evidence from the supplied issue without guessing repository file paths.

# Analysis Rules
- Do infer failure stages, observable symptoms, likely faulty behaviors, related symbols, and retrieval keywords grounded in the issue.
- Do generate concise fault hypotheses containing faulty_operation, involved_symbol, wrong_value, and expected_constraint.
- Do leave a field empty when the issue does not support it.
- Do predict relevant roles only from LLM_PROMPT, LLM_CALL, LLM_CONFIG, LLM_TOOL, and LLM_MEMORY.
- Do treat predicted roles as supplementary evidence rather than proof.
- Do not invent file paths, symbols, values, constraints, or behavior absent from the issue.
- Do return only one valid JSON object matching the requested schema."""

JOINT_VALIDATOR_SYSTEM_PROMPT = """You are the Context-aware Validator of an LLM defect-localization system.
Your task is to assign every supplied candidate file a counterfactual score and rank every supplied Statement anchor by root-cause likelihood.

# Counterfactual Validation Rules
- Do ask for each file: if its faulty operation were correctly repaired while all other code remained unchanged, would the reported defect disappear?
- Do assign every supplied file an integer score from 1 to 10, where 8-10 means a likely root cause, 6-7 means a contributing or propagating file, and 1-5 means a symptom or weakly related file.
- Do infer the smallest plausible correction only to judge causal responsibility; do not generate, apply, or execute a patch.
- Do rank every supplied Statement anchor ID exactly once from most to least suspicious.
- Do rank root-cause decisions and transformations above propagation calls, exception sites, definitions, documentation, tests, and nearby context.
- Do keep file and Statement decisions hierarchical; rank a Statement in a lower-scored file above Statements in a higher-scored file only when its supplied evidence gives stronger root-cause support.
- Do treat a hypothesis match as strong only when exact symbol evidence is combined with operation overlap or substantial contextual coverage.
- Do use only the issue, fault hypotheses, retrieval sources, semantic annotations, confidence, source context, and CodeKG relations supplied in the input.
- Do treat traces and graph relations as evidence rather than proof.
- Do not invent files, Statements, evidence, or IDs.
- Do return exactly one valid JSON object shaped as {"scored_files":{"F1":10,"F2":1},"ranked_locations":["L2","L1"]}, containing every supplied file ID and Statement ID exactly once, with no reasons, objects, Markdown, or additional fields."""

FINAL_VALIDATOR_SYSTEM_PROMPT = """You are the final exact-line Validator of an LLM defect-localization system.
Your task is to rank the physical source lines whose correction would most directly resolve the reported issue.

# Exact-Line Validation Rules
- Do read every complete source file supplied in the batch.
- Do ask for each candidate line: if this line were correctly repaired while all other code remained unchanged, would the reported defect disappear?
- Do use the issue, fault hypotheses, complete source, and file rank to judge causal responsibility.
- Do rank root-cause decisions and transformations above propagation calls, exception sites, definitions, documentation, tests, and nearby context.
- Do prefer different root-cause Statements; use multiple lines from one Statement only when each line is independently plausible.
- Do return exactly ten different physical lines unless fewer than ten non-empty source lines are supplied.
- Do use only supplied file IDs and existing physical line numbers.
- Do not invent a file or line, and do not generate, apply, or execute a patch.
- Do return exactly one valid JSON object shaped as {"ranked_locations":[{"file_id":"F1","line":42}]}, with no Markdown or additional fields."""

FINAL_MERGER_SYSTEM_PROMPT = """You are the global exact-line Validator of an LLM defect-localization system.
Your task is to merge and rank the supplied batch-level line candidates by root-cause likelihood.

# Global Ranking Rules
- Do ask for each candidate: if this exact line were correctly repaired while all other code remained unchanged, would the reported defect disappear?
- Do use the issue, source context, file rank, and batch rank as evidence.
- Do rank root causes above propagation, symptoms, definitions, documentation, tests, and nearby context.
- Do return exactly ten unique supplied candidate IDs unless fewer than ten candidates are supplied.
- Do not invent IDs, and do not generate, apply, or execute a patch.
- Do return exactly one valid JSON object shaped as {"ranked_location_ids":["C2","C1"]}, with no Markdown or additional fields."""







class AnnotationItem(BaseModel):
    file_id: str = Field(validation_alias=AliasChoices("file_id", "id"))
    descriptions: list[str] = Field(default_factory=list, validation_alias=AliasChoices("descriptions", "description"))
    keywords: list[str] = Field(default_factory=list)

    @field_validator("descriptions", "keywords", mode="before")
    @classmethod
    def normalize_text_lists(cls, value):
        if isinstance(value, str):
            return [value]
        return value


class AnnotationResult(BaseModel):
    LLM_PROMPT: list[AnnotationItem] = Field(default_factory=list)
    LLM_CALL: list[AnnotationItem] = Field(default_factory=list)
    LLM_CONFIG: list[AnnotationItem] = Field(default_factory=list)
    LLM_TOOL: list[AnnotationItem] = Field(default_factory=list)
    LLM_MEMORY: list[AnnotationItem] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_file_annotations(cls, value):
        if isinstance(value, list):
            file_items = value
        elif isinstance(value, dict):
            wrapped_items = next(
                (
                    value.get(key)
                    for key in ("files", "annotations", "classifications")
                    if isinstance(value.get(key), list)
                ),
                None,
            )
            if wrapped_items is not None:
                file_items = wrapped_items
            elif not set(value).intersection(ANNOTATION_ROLES):
                file_items = [
                    {
                        "file_id": file_id,
                        **(
                            annotations
                            if isinstance(annotations, dict) and "roles" in annotations
                            else {"roles": annotations}
                        ),
                    }
                    for file_id, annotations in value.items()
                ]
            else:
                return value
        else:
            return value
        normalized = {role: [] for role in ANNOTATION_ROLES}
        for file_item in file_items:
            if not isinstance(file_item, dict):
                continue
            file_id = file_item.get("file_id") or file_item.get("id")
            raw_roles = file_item.get("roles", file_item.get("classifications", []))
            descriptions = file_item.get("descriptions", file_item.get("description", {}))
            keywords = file_item.get("keywords", {})
            if isinstance(raw_roles, dict):
                raw_roles = [
                    {"role": role, **(annotation if isinstance(annotation, dict) else {})}
                    for role, annotation in raw_roles.items()
                ]
            elif isinstance(raw_roles, list) and all(isinstance(role, str) for role in raw_roles):
                raw_roles = [
                    {
                        "role": role,
                        "description": descriptions.get(role, []) if isinstance(descriptions, dict) else descriptions,
                        "keywords": keywords.get(role, []) if isinstance(keywords, dict) else keywords,
                    }
                    for role in raw_roles
                ]
            for annotation in raw_roles:
                if not isinstance(annotation, dict):
                    continue
                role = annotation.get("role")
                if role not in normalized or not file_id:
                    continue
                normalized[role].append(
                    {
                        "file_id": file_id,
                        "descriptions": annotation.get("descriptions", annotation.get("description", [])),
                        "keywords": annotation.get("keywords", []),
                    }
                )
        return normalized


class FaultHypothesis(BaseModel):
    faulty_operation: str = ""
    involved_symbol: str = ""
    wrong_value: str = ""
    expected_constraint: str = ""


class AnalyzerResult(BaseModel):
    failure_stages: list[str] = Field(default_factory=list, validation_alias=AliasChoices("failure_stages", "failure_stage"))
    symptoms: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "symptoms",
            "observable_symptoms",
            "observable_symptom",
            "symptom_analysis",
            "description",
        ),
    )
    faulty_behaviors: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "faulty_behaviors",
            "faulty_behavior",
            "likely_faulty_behaviors",
            "likely_faulty_behavior",
        ),
    )
    related_symbols: list[str] = Field(default_factory=list)
    retrieval_keywords: list[str] = Field(default_factory=list)
    predicted_roles: list[str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("predicted_roles", "LLM_tags", "roles"),
    )
    fault_hypotheses: list[FaultHypothesis] = Field(
        default_factory=list,
        validation_alias=AliasChoices("fault_hypotheses", "fault_hypothesis"),
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_defect_evidence(cls, value):
        if isinstance(value, dict) and isinstance(value.get("defect_evidence"), dict):
            return value["defect_evidence"]
        return value

    @field_validator("fault_hypotheses", mode="before")
    @classmethod
    def normalize_fault_hypotheses(cls, value):
        if isinstance(value, dict):
            return [value]
        return value

    @field_validator(
        "failure_stages",
        "symptoms",
        "faulty_behaviors",
        "related_symbols",
        "retrieval_keywords",
        "predicted_roles",
        mode="before",
    )
    @classmethod
    def normalize_string_lists(cls, values):
        if isinstance(values, str):
            return [value.strip() for value in re.split(r"[,;\n]+", values) if value.strip()]
        if isinstance(values, dict):
            return list(values)
        if isinstance(values, list):
            normalized = []
            for value in values:
                if isinstance(value, str):
                    normalized.append(value)
                elif isinstance(value, dict):
                    normalized.append(
                        str(value.get("role") or value.get("name") or value.get("value") or "")
                    )
                else:
                    normalized.append(str(value))
            return [value for value in normalized if value]
        return values


class RankedFile(BaseModel):
    file_id: str = Field(validation_alias=AliasChoices("file_id", "id"))
    score: int = Field(validation_alias=AliasChoices("score", "counterfactual_score"), ge=1, le=10)
    behavior_match: int = Field(default=0, ge=0, le=10)
    symbol_match: int = Field(default=0, ge=0, le=10)
    execution_path_support: int = Field(default=0, ge=0, le=10)
    counter_evidence: int = Field(default=0, ge=0, le=10)
    reason: str = ""










class RankedLocation(BaseModel):
    location_id: str = Field(validation_alias=AliasChoices("location_id", "id"))
    score: int | None = Field(default=None, ge=0, le=10)
    roles: list[str] = Field(default_factory=list, validation_alias=AliasChoices("roles", "llm_roles"))
    reason: str = ""




class JointValidatorResult(BaseModel):
    scored_files: list[RankedFile]
    ranked_locations: list[RankedLocation]

    @field_validator("scored_files", mode="before")
    @classmethod
    def normalize_scored_files(cls, values):
        if isinstance(values, dict):
            return [
                {"file_id": file_id, "score": score}
                for file_id, score in values.items()
            ]
        return values

    @field_validator("ranked_locations", mode="before")
    @classmethod
    def normalize_ranked_locations(cls, values):
        return [{"location_id": value} if isinstance(value, str) else value for value in values]


class FinalLocation(BaseModel):
    file_id: str = Field(validation_alias=AliasChoices("file_id", "id"))
    line: int = Field(ge=1)


class FinalValidatorResult(BaseModel):
    ranked_locations: list[FinalLocation] = Field(default_factory=list)


class FinalMergerResult(BaseModel):
    ranked_location_ids: list[str] = Field(default_factory=list)


def selected_instances(dataset, instance_id=None, limit=None):
    if instance_id:
        selected = [item for item in dataset if item.get("instance_id") == instance_id]
        if not selected:
            raise ValueError(f"Instance not found: {instance_id}")
        return selected
    return dataset if limit is None else dataset[:limit]


def instance_fingerprint(instance):
    content = json.dumps(instance, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def run_configuration(instance, args, repo_fingerprint):
    return {
        "instance_id": instance["instance_id"],
        "instance_sha256": instance_fingerprint(instance),
        "repo_path": str(args.repo_base / instance["instance_id"]),
        "repository_sha256": repo_fingerprint,
        "model": args.model,
        "cache_base": str(args.cache_base),
        "recorded_evidence_channels": ["E", "I", "R", "CodeKG", "V"],
        "retrieval_variants": ["full"],
        "validator_variants": ["full"],
        "posthoc_statistics_source": "full retrieval, complete Joint Validator Statement rankings, and exact-line Final Validator",
        "seed_count": args.seed_count,
        "candidate_count": args.candidate_count,
        "entity_budget": args.entity_budget,
        "line_token_budget_ratio": args.line_token_budget_ratio,
        "lexical_weight": args.lexical_weight,
        "top_k": args.top_k,
        "max_output_tokens": args.max_output_tokens,
        "max_input_tokens": args.max_input_tokens,
        "validator_pipeline": "all_joint_statements_first_occurrence_top_10_files_to_complete_file_exact_line_batches_then_global_merge",
        "analyzer_pipeline": "direct_extraction_plus_llm_behavior_inference_then_bm25_codekg_retrieval_plus_llm_annotation_retrieval",
        "annotation_pipeline": "regex_seed_plus_one_hop_bfs_plus_bm25_plus_llm_annotation",
        "annotation_seed_count": args.annotation_seed_count,
        "annotation_expanded_count": args.annotation_expanded_count,
        "line_chunking": "hypothesis_by_file_tree_sitter_statement_coverage_then_mmr_and_offline_line_mapping",
    }


def completed(case_path, configuration):
    output_path = case_path / "output.json"
    input_path = case_path / "input.json"
    if not output_path.is_file() or not input_path.is_file():
        return False
    try:
        output = json.loads(output_path.read_text(encoding="utf-8"))
        previous_configuration = json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        logger.warning(f"Invalid completed instance files in {case_path}: {error}")
        return False
    return (
        previous_configuration == configuration
        and output.get("status") == "completed"
        and bool(output.get("ranked_locations"))
    )






def normalize_relation_edges(relation_edges):
    fields = [
        "source",
        "target",
        "relation",
        "direction",
        "confidence",
        "evidence_level",
        "weight",
        "provenance",
    ]
    return fields, [[item.get(field) for field in fields] for item in relation_edges]


def normalize_line_rankings(line_rankings):
    statements = []
    statement_ids = {}
    normalized_lines = []
    statement_fields = {
        "statement_start",
        "statement_end",
        "statement_kind",
        "statement_code",
        "hypothesis_coverages",
        "hypothesis_matches",
    }
    for item in line_rankings:
        statement = {key: item.get(key) for key in statement_fields}
        statement_key = (
            item["path"],
            item.get("statement_start", item["line"]),
            item.get("statement_end", item["line"]),
            item.get("statement_kind", "line"),
        )
        statement_id = statement_ids.get(statement_key)
        if statement_id is None:
            statement_id = f"S{len(statements) + 1}"
            statement_ids[statement_key] = statement_id
            statements.append({"statement_id": statement_id, **statement})
        normalized_lines.append(
            {
                **{key: value for key, value in item.items() if key not in statement_fields},
                "statement_id": statement_id,
            }
        )
    return statements, normalized_lines


def retrieval_artifact(retrieval_variants):
    full = retrieval_variants["full"]
    common_configuration = {
        key: value
        for key, value in full["configuration"].items()
        if key not in {"evidence_channels", "lexical_weight", "graph_weight"}
    }
    relation_edge_fields, relation_edges = normalize_relation_edges(full["relation_edges"])
    common = {
        "configuration": common_configuration,
        "entity_rankings": [
            {key: value for key, value in item.items() if key != "symbol_graph_score"}
            for item in full["entity_rankings"]
        ],
        "direct_evidence": full["direct_evidence"],
        "role_evidence": full["role_evidence"],
        "relation_edge_fields": relation_edge_fields,
        "relation_edges": relation_edges,
    }
    variants = {}
    for name, retrieval in retrieval_variants.items():
        statements, line_rankings = normalize_line_rankings(retrieval["line_rankings"])
        variants[name] = {
            "configuration": {
                "evidence_channels": retrieval["configuration"]["evidence_channels"],
                "lexical_weight": retrieval["configuration"]["lexical_weight"],
                "graph_weight": retrieval["configuration"]["graph_weight"],
            },
            "entity_graph_scores": [
                {
                    "entity": item["entity"],
                    "symbol_graph_score": item["symbol_graph_score"],
                }
                for item in retrieval["entity_rankings"]
            ],
            "seeds": retrieval["seeds"],
            "file_rankings": retrieval["file_rankings"],
            "statements": statements,
            "line_rankings": line_rankings,
        }
    return {
        "schema_version": 2,
        "normalization": {
            "common.relation_edges": "rows matching common.relation_edge_fields",
            "line_rankings.statement_id": "variants.<name>.statements",
        },
        "common": common,
        "variants": variants,
    }


def file_nodes(code_graph):
    return {
        node: data
        for node, data in code_graph.nodes(data=True)
        if data.get("type") in FILE_TYPES
    }


def split_prompt_batches(items, messages_for_items, model, token_budget):
    batches = []
    current = []
    for item in items:
        tentative = [*current, item]
        tokens = litellm.token_counter(model=model, messages=messages_for_items(tentative))
        if tokens <= token_budget:
            current = tentative
            continue
        if current:
            batches.append(current)
            current = [item]
            tokens = litellm.token_counter(model=model, messages=messages_for_items(current))
        if tokens > token_budget:
            raise ValueError(f"One complete repository item requires {tokens} tokens and exceeds {token_budget}")
    if current:
        batches.append(current)
    return batches


def annotation_messages(items):
    return [
        {"role": "system", "content": ANNOTATION_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps({"files": items}, ensure_ascii=False)},
    ]


def regex_annotation_seeds(code_graph, top_k):
    scored = []
    compiled = {
        role: [re.compile(pattern, re.IGNORECASE) for pattern in definition["patterns"].values()]
        for role, definition in ROLE_PATTERNS.items()
    }
    total_patterns = sum(len(patterns) for patterns in compiled.values())
    for path, data in file_nodes(code_graph).items():
        code = data.get("code", "")
        matches = [pattern.findall(code) for patterns in compiled.values() for pattern in patterns]
        matched_types = sum(bool(items) for items in matches)
        match_count = sum(len(items) for items in matches)
        if not match_count:
            continue
        coverage = matched_types / total_patterns
        density = match_count / max(len(code) / 1000, 0.001)
        scored.append(
            {
                "path": path,
                "score": 0.7 * coverage + min(density / 10, 0.3),
                "coverage": coverage,
                "density": density,
                "match_count": match_count,
            }
        )
    scored.sort(key=lambda item: (-item["score"], item["path"]))
    return scored, [item["path"] for item in scored[:top_k]]


def expand_annotation_files(code_graph, seeds, hops=1):
    expanded = set(seeds)
    frontier = set(seeds)
    details = []
    for hop in range(1, hops + 1):
        neighbors = set()
        for node in frontier:
            for source, target in code_graph.in_edges(node):
                owner = owning_file(source, code_graph.nodes[source])
                if owner and owner not in expanded:
                    neighbors.add(owner)
            for source, target in code_graph.out_edges(node):
                owner = owning_file(target, code_graph.nodes[target])
                if owner and owner not in expanded:
                    neighbors.add(owner)
        details.append({"hop": hop, "files": sorted(neighbors)})
        expanded.update(neighbors)
        frontier = neighbors
        if not frontier:
            break
    return expanded, details


def rank_annotation_expansion(code_graph, expanded, seeds, top_k):
    paths = sorted(expanded - set(seeds))
    if not paths:
        return [], []
    documents = [tokenize(code_graph.nodes[path].get("code", "")) for path in paths]
    query = []
    for definition in ROLE_PATTERNS.values():
        for pattern in definition["patterns"].values():
            query.extend(tokenize(pattern))
    scores = BM25Okapi(documents).get_scores(query)
    ranked = [
        {"path": path, "bm25_score": float(score)}
        for path, score in sorted(zip(paths, scores), key=lambda item: (-item[1], item[0]))
    ]
    return ranked, [item["path"] for item in ranked[:top_k]]


def annotate_repository(code_graph, args, case_path, cache_path=None):
    for _, data in code_graph.nodes(data=True):
        data["llm_role_annotations"] = []
        data["llm_roles"] = []
    if cache_path and cache_path.is_file():
        artifact = json.loads(cache_path.read_text(encoding="utf-8"))
        cached_annotations = artifact.get("validated_annotations", {})
        if cached_annotations and any(cached_annotations.values()):
            for path, annotations in cached_annotations.items():
                if path not in code_graph:
                    continue
                code_graph.nodes[path]["llm_role_annotations"] = annotations
                code_graph.nodes[path]["llm_roles"] = list(dict.fromkeys(item["role"] for item in annotations))
            artifact["cache_hit"] = True
            atomic_json(case_path / "annotation.json", artifact)
            logger.info(f"Annotation cache hit: {cache_path}")
            return artifact
        logger.warning(f"Ignoring empty Annotation cache: {cache_path}")
    regex_results, seeds = regex_annotation_seeds(code_graph, args.annotation_seed_count)
    expanded, expansion_details = expand_annotation_files(code_graph, seeds)
    expansion_rankings, selected_expansion = rank_annotation_expansion(
        code_graph, expanded, seeds, args.annotation_expanded_count
    )
    selected_paths = list(dict.fromkeys([*seeds, *selected_expansion]))
    items = []
    paths_by_id = {}

    annotation_token_limit = int(args.max_input_tokens * 0.9)

    def add_annotation_chunks(path, text, file_index, chunk_index=1):
        file_id = f"A{file_index}P{chunk_index}"
        item = {"file_id": file_id, "path": path, "text": text}
        tokens = litellm.token_counter(model=args.model, messages=annotation_messages([item]))
        if tokens <= annotation_token_limit:
            items.append(item)
            paths_by_id[file_id] = path
            return
        if len(text) < 2:
            raise ValueError(f"One source character in {path} exceeds the annotation token limit")
        middle = len(text) // 2
        add_annotation_chunks(path, text[:middle], file_index, chunk_index * 2)
        add_annotation_chunks(path, text[middle:], file_index, chunk_index * 2 + 1)

    for file_index, path in enumerate(selected_paths, 1):
        add_annotation_chunks(path, code_graph.nodes[path].get("code", ""), file_index)
    batches = split_prompt_batches(
        items,
        annotation_messages,
        args.model,
        annotation_token_limit,
    ) if items else []
    raw_results = []
    annotations_by_path = {path: [] for path in selected_paths}
    pattern_updates = {role: [] for role in ANNOTATION_ROLES}
    for batch_index, batch in enumerate(batches, 1):
        messages = annotation_messages(batch)
        prompt_tokens = litellm.token_counter(model=args.model, messages=messages)
        result = call_llm(
            messages,
            args.model,
            args.openai_base_url,
            args.openai_api_key,
            pydantic_object=AnnotationResult,
            log_file=case_path / "calls_llm_annotation.jsonl",
            log_metadata={"batch_index": batch_index, "batch_count": len(batches), "prompt_tokens": prompt_tokens},
            temperature=0,
            max_tokens=args.max_output_tokens,
            response_format={"type": "json_object"},
        )
        raw_results.append(result)
        valid_ids = {item["file_id"] for item in batch}
        for role in ANNOTATION_ROLES:
            for annotation in result.get(role, []):
                file_id = annotation["file_id"]
                if file_id not in valid_ids or file_id not in paths_by_id:
                    continue
                path = paths_by_id[file_id]
                code = code_graph.nodes[path].get("code", "")
                keywords = sorted(
                    {
                        keyword
                        for keyword in annotation.get("keywords", [])
                        if keyword and keyword in code
                    },
                    key=lambda value: (len(value), value),
                )
                keywords = [
                    keyword
                    for keyword in keywords
                    if not any(keyword != other and keyword.startswith(other) for other in keywords)
                ]
                record = {
                    "role": role,
                    "description": annotation.get("descriptions", []),
                    "pattern": "llm_semantic_annotation",
                    "matched_evidence": keywords,
                    "keywords": keywords,
                    "provenance": "llm_semantic_annotation",
                }
                annotations_by_path[path].append(record)
                pattern_updates[role].extend(keyword for keyword in keywords if keyword not in pattern_updates[role])
    for path, annotations in annotations_by_path.items():
        code_graph.nodes[path]["llm_role_annotations"] = annotations
        code_graph.nodes[path]["llm_roles"] = list(dict.fromkeys(item["role"] for item in annotations))
    artifact = {
        "regex_rankings": regex_results,
        "seed_files": seeds,
        "bfs_expansion": expansion_details,
        "expanded_bm25_rankings": expansion_rankings,
        "selected_files": selected_paths,
        "annotation_chunks": [
            {"file_id": item["file_id"], "path": item["path"], "character_count": len(item["text"])}
            for item in items
        ],
        "raw_results": raw_results,
        "validated_annotations": annotations_by_path,
        "pattern_updates": pattern_updates,
        "pattern_update_scope": "instance_artifact_only",
        "cache_hit": False,
    }
    atomic_json(case_path / "annotation.json", artifact)
    if cache_path:
        atomic_json(cache_path, artifact)
    return artifact



def analyzer_messages(issue):
    return [
        {"role": "system", "content": ANALYZER_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps({"issue": issue, "allowed_roles": ANNOTATION_ROLES}, ensure_ascii=False)},
    ]


def retrieve_behavior_files(code_graph, behavior_fields, issue, use_bm25=True):
    if not use_bm25:
        return [], []
    entities = []
    documents = []
    document_texts = []
    for node, data in code_graph.nodes(data=True):
        if data.get("type") not in {"file", "textfile", "class", "function", "attribute"}:
            continue
        text = "\n".join((node, data.get("name", ""), data.get("code", "")))
        entities.append((node, data))
        document_texts.append(text.lower())
        documents.append(tokenize(text))
    if not entities:
        return [], []
    raw_field_scores = {}
    field_support = {}
    issue_text = issue.lower()
    repository_text = "\n".join(document_texts)
    for field in I_FIELDS:
        values = behavior_fields.get(field, [])
        query_tokens = tokenize(" ".join(values))
        if not query_tokens:
            continue
        scores = BM25Okapi(documents).get_scores(query_tokens)
        raw_field_scores[field] = [float(score) for score in scores]
        field_support[field] = [
            {
                "term": value,
                "issue_supported": value.lower() in issue_text,
                "repository_supported": value.lower() in repository_text,
            }
            for value in values
        ]
    rankings = []
    for index, (node, data) in enumerate(entities):
        field_raw = {field: scores[index] for field, scores in raw_field_scores.items()}
        maximum_raw_score = max(field_raw.values(), default=0.0)
        if maximum_raw_score <= 0:
            continue
        rankings.append(
            {
                "entity": node,
                "file": owning_file(node, data),
                "entity_type": data.get("type"),
                "maximum_raw_bm25_score": maximum_raw_score,
                "raw_field_scores": field_raw,
                "matched_term_count": sum(token in set(documents[index]) for values in behavior_fields.values() for token in tokenize(" ".join(values))),
                "exact_symbol_match": any(value.lower() in {node.lower(), data.get("name", "").lower()} for value in behavior_fields.get("related_symbols", [])),
                "issue_supported_behavior": any(item["issue_supported"] for item in field_support.get("faulty_behaviors", [])),
                "repository_supported_behavior": any(item["repository_supported"] for item in field_support.get("faulty_behaviors", [])),
                "field_support": field_support,
            }
        )
    rankings.sort(
        key=lambda item: (
            -item["exact_symbol_match"],
            -item["issue_supported_behavior"],
            -item["repository_supported_behavior"],
            -item["matched_term_count"],
            -item["maximum_raw_bm25_score"],
            item["entity"],
        )
    )
    files = list(dict.fromkeys(item["file"] for item in rankings if item.get("file")))
    return rankings, files


def retrieve_instance_patterns(code_graph, pattern_updates):
    matches = []
    for path, data in sorted(file_nodes(code_graph).items()):
        code = data.get("code", "")
        roles = []
        keywords = []
        for role, values in pattern_updates.items():
            matched = [value for value in values if value and value in code]
            if matched:
                roles.append(role)
                keywords.extend(matched)
        if roles:
            matches.append({"path": path, "roles": roles, "keywords": list(dict.fromkeys(keywords))})
    return matches


def analyze_defect(code_graph, issue, annotation_artifact, args, case_path):
    messages = analyzer_messages(issue)
    prompt_tokens = litellm.token_counter(model=args.model, messages=messages)
    result = call_llm(
        messages,
        args.model,
        args.openai_base_url,
        args.openai_api_key,
        pydantic_object=AnalyzerResult,
        log_file=case_path / "calls_llm_analyzer.jsonl",
        log_metadata={"prompt_tokens": prompt_tokens},
        temperature=0,
        max_tokens=args.max_output_tokens,
        response_format={"type": "json_object"},
    )
    artifact = build_analyzer_evidence(code_graph, issue, annotation_artifact, result)
    atomic_json(case_path / "analyzer.json", artifact)
    return artifact


def build_analyzer_evidence(code_graph, issue, annotation_artifact, result, use_bm25=True):
    result = dict(result)
    if not result.get("symptoms"):
        result["symptoms"] = [issue]
        logger.warning("Analyzer returned no symptoms; using the full issue as symptom evidence")
    if not result.get("faulty_behaviors"):
        result["faulty_behaviors"] = [
            hypothesis["faulty_operation"]
            for hypothesis in result.get("fault_hypotheses", [])
            if hypothesis.get("faulty_operation")
        ]
        if not result["faulty_behaviors"]:
            result["faulty_behaviors"] = [issue]
        logger.warning("Analyzer returned no faulty behaviors; using available issue evidence")
    known_paths = set(file_nodes(code_graph))
    behavior_fields = {field: result.get(field, []) for field in I_FIELDS}
    behavior_query = " ".join(value for values in behavior_fields.values() for value in values)
    behavior_rankings, all_inferred_files = retrieve_behavior_files(code_graph, behavior_fields, issue, use_bm25=use_bm25)
    inference_by_file = {}
    for item in behavior_rankings:
        path = item.get("file")
        if not path:
            continue
        evidence = inference_by_file.setdefault(
            path,
            {
                "exact_symbol_match": False,
                "issue_supported_behavior": False,
                "repository_supported_behavior": False,
                "matched_term_count": 0,
                "maximum_raw_bm25_score": 0.0,
            },
        )
        evidence["exact_symbol_match"] |= item["exact_symbol_match"]
        evidence["issue_supported_behavior"] |= item["issue_supported_behavior"]
        evidence["repository_supported_behavior"] |= item["repository_supported_behavior"]
        evidence["matched_term_count"] = max(evidence["matched_term_count"], item["matched_term_count"])
        evidence["maximum_raw_bm25_score"] = max(
            evidence["maximum_raw_bm25_score"], item["maximum_raw_bm25_score"]
        )
    inferred_files = [
        path
        for path, evidence in sorted(
            inference_by_file.items(),
            key=lambda pair: (
                -pair[1]["exact_symbol_match"],
                -pair[1]["issue_supported_behavior"],
                -pair[1]["repository_supported_behavior"],
                -pair[1]["matched_term_count"],
                -pair[1]["maximum_raw_bm25_score"],
                len(Path(pair[0]).parts),
                pair[0],
            ),
        )
        if evidence["exact_symbol_match"]
        or evidence["issue_supported_behavior"]
        or evidence["repository_supported_behavior"]
        or evidence["matched_term_count"] >= 2
    ]
    shallow_inferred_files = sorted(
        (
            (path, evidence)
            for path, evidence in inference_by_file.items()
            if len(Path(path).parts) <= 2 and evidence["maximum_raw_bm25_score"] > 0
        ),
        key=lambda pair: (-pair[1]["maximum_raw_bm25_score"], pair[0]),
    )[:20]
    inferred_files = list(dict.fromkeys([*(path for path, _ in shallow_inferred_files), *inferred_files]))
    file_inference_scores = {
        path: evidence["maximum_raw_bm25_score"]
        for path, evidence in inference_by_file.items()
    }
    predicted_roles = [role for role in result.get("predicted_roles", []) if role in ANNOTATION_ROLES]
    direct_files = list(
        dict.fromkeys(
            item["file"]
            for item in extract_direct_evidence(code_graph, issue)
            if item.get("file") in known_paths
        )
    )
    instance_pattern_matches = retrieve_instance_patterns(code_graph, annotation_artifact["pattern_updates"])
    issue_terms = set(tokenize(issue))
    role_rankings = []
    pattern_match_by_path = {item["path"]: item for item in instance_pattern_matches}
    for path, data in sorted(file_nodes(code_graph).items()):
        roles = set(data.get("llm_roles", [])).intersection(predicted_roles)
        pattern_match = pattern_match_by_path.get(path, {})
        roles.update(set(pattern_match.get("roles", [])).intersection(predicted_roles))
        if not roles:
            continue
        annotation_terms = set()
        for annotation in data.get("llm_role_annotations", []):
            annotation_terms.update(tokenize(" ".join([*annotation.get("description", []), *annotation.get("keywords", [])])))
        annotation_terms.update(tokenize(" ".join(pattern_match.get("keywords", []))))
        role_rankings.append(
            {
                "path": path,
                "roles": sorted(roles),
                "issue_overlap": len(issue_terms.intersection(annotation_terms)),
                "matched_keywords": pattern_match.get("keywords", []),
            }
        )
    role_rankings.sort(key=lambda item: (-item["issue_overlap"], -len(item["matched_keywords"]), item["path"]))
    positive_role_files = [item["path"] for item in role_rankings if item["issue_overlap"] > 0]
    role_files = positive_role_files or ([role_rankings[0]["path"]] if role_rankings else [])
    channels_by_file = {}
    for path in known_paths:
        channels = []
        if path in direct_files:
            channels.append("E")
        if path in inferred_files:
            channels.append("I")
        if path in role_files:
            channels.append("R")
        if channels:
            channels_by_file[path] = channels
    artifact = {
        "failure_stages": result.get("failure_stages", []),
        "symptoms": result.get("symptoms", []),
        "faulty_behaviors": result.get("faulty_behaviors", []),
        "related_symbols": result.get("related_symbols", []),
        "retrieval_keywords": result.get("retrieval_keywords", []),
        "fault_hypotheses": result.get("fault_hypotheses", []),
        "behavior_query": behavior_query,
        "E": direct_files,
        "I": inferred_files,
        "I_all_ranked_files": all_inferred_files,
        "I_evidence_order": ["exact_symbol", "issue_supported_behavior", "repository_supported_behavior", "matched_terms", "raw_bm25"],
        "I_file_evidence": inference_by_file,
        "I_entity_rankings": behavior_rankings,
        "R": {"predicted_roles": predicted_roles, "matched_files": role_files, "rankings": role_rankings, "selection": "positive_issue_annotation_overlap_else_best_supplement", "instance_pattern_matches": instance_pattern_matches},
        "channels_by_file": channels_by_file,
        "confidence_by_file": {path: len(channels) for path, channels in channels_by_file.items()},
        "raw_result": result,
    }
    return artifact


def enrich_retrieval_with_analyzer(retrieval, analyzer, settings):
    enabled = {
        "E": settings["use_direct"],
        "I": settings["use_inference"],
        "R": settings["use_roles"],
    }
    context_by_path = {item["path"]: item for item in retrieval["candidate_context"]}
    file_ranking_by_path = {item["path"]: item for item in retrieval["file_rankings"]}
    ordered_evidence_paths = list(dict.fromkeys([
        *analyzer["E"],
        *analyzer["I"],
        *analyzer["R"]["matched_files"],
    ]))
    enabled_evidence_paths = [
        path
        for path in ordered_evidence_paths
        if any(enabled[channel] for channel in analyzer["channels_by_file"].get(path, []))
    ]
    missing = [path for path in enabled_evidence_paths if path not in context_by_path]
    if missing:
        logger.warning(
            f"Analyzer evidence budget excluded {len(missing)} files; first two: {missing[:2]}; full rankings remain in analyzer.json"
        )
    ordered_paths = list(context_by_path)
    contexts = []
    for path in ordered_paths:
        item = context_by_path[path]
        channels = [
            channel for channel in analyzer["channels_by_file"].get(path, []) if enabled[channel]
        ]
        analyzer_confidence = len(channels)
        contexts.append({**item, "evidence_channels": channels, "analyzer_confidence": analyzer_confidence})
    retrieval["candidate_context"] = contexts
    retrieval["file_rankings"] = [file_ranking_by_path[path] for path in ordered_paths if path in file_ranking_by_path]
    retrieval["analyzer_evidence"] = {
        "enabled_channels": enabled,
        "channels_by_file": {item["path"]: item["evidence_channels"] for item in contexts if item["evidence_channels"]},
    }


def build_local_execution_subgraph(code_graph, path):
    file_entities = {
        node
        for node, data in code_graph.nodes(data=True)
        if owning_file(node, data) == path
    }
    nodes = set(file_entities)
    edges = []
    for source, target, data in code_graph.edges(data=True):
        relation = data.get("type")
        if relation not in {"imports", "calls", "extends"}:
            continue
        source_local = source in file_entities
        target_local = target in file_entities
        if not source_local and not target_local:
            continue
        nodes.update((source, target))
        edges.append(
            {
                "source": source,
                "target": target,
                "type": relation,
                "direction": "outgoing" if source_local else "incoming",
                "confidence": data.get("confidence", 1.0),
                "resolution": data.get("resolution", "exact"),
            }
        )
    return {
        "nodes": [
            {
                "id": node,
                "type": code_graph.nodes[node].get("type"),
                "file": owning_file(node, code_graph.nodes[node]),
                "signature": code_graph.nodes[node].get("code", "").splitlines()[:1],
            }
            for node in sorted(nodes)
        ],
        "edges": sorted(edges, key=lambda item: (item["type"], item["source"], item["target"])),
    }


def attach_local_subgraphs(code_graph, retrieval):
    for candidate in retrieval["candidate_context"]:
        candidate["local_execution_subgraph"] = build_local_execution_subgraph(code_graph, candidate["path"])












def select_covered_lines(line_rankings, ordered_paths, code_graph, limit):
    started = time.monotonic()
    rank_by_location = {
        (item["path"], item["line"]): rank
        for rank, item in enumerate(line_rankings, 1)
    }
    item_by_location = {
        (item["path"], item["line"]): item
        for item in line_rankings
    }
    file_rank = {path: rank for rank, path in enumerate(ordered_paths, 1)}
    statement_groups = {}
    for item in line_rankings:
        if item["path"] not in file_rank:
            continue
        statement_start = item.get("statement_start", item["line"])
        statement_end = item.get("statement_end", item["line"])
        if statement_end - statement_start + 1 > 12:
            statement_start += ((item["line"] - statement_start) // 6) * 6
            statement_end = min(statement_start + 5, statement_end)
        key = (item["path"], statement_start, statement_end)
        statement_groups.setdefault(key, []).append(item)

    statements = []
    for key, items in statement_groups.items():
        representative = min(
            items,
            key=lambda item: rank_by_location[(item["path"], item["line"])],
        )
        statements.append(
            {
                **representative,
                "statement_start": key[1],
                "statement_end": key[2],
                "statement_lines": sorted(item["line"] for item in items),
                "hypothesis_coverages": [
                    max(
                        (item.get("hypothesis_coverages", [])[index] for item in items),
                        default=0.0,
                    )
                    for index in range(
                        max((len(item.get("hypothesis_coverages", [])) for item in items), default=0)
                    )
                ],
                "hypothesis_matches": [
                    {
                        "coverage": max(
                            (item.get("hypothesis_matches", [])[index]["coverage"] for item in items),
                            default=0.0,
                        ),
                        "symbol_match": any(
                            item.get("hypothesis_matches", [])[index]["symbol_match"] for item in items
                        ),
                        "operation_overlap": max(
                            (item.get("hypothesis_matches", [])[index]["operation_overlap"] for item in items),
                            default=0,
                        ),
                    }
                    for index in range(
                        max((len(item.get("hypothesis_matches", [])) for item in items), default=0)
                    )
                ],
            }
        )
    statements.sort(
        key=lambda item: rank_by_location[(item["path"], item["line"])]
    )
    grouped_seconds = time.monotonic() - started
    anchor_limit = min(limit, 100)
    selected = []
    selected_keys = set()
    selected_files = set()
    selected_blocks = set()
    selected_types = set()
    selected_sources = set()
    term_started = time.monotonic()
    terms_by_code = {}
    term_cache = {}
    for item in statements:
        code = item.get("statement_code", item["code"])
        terms = terms_by_code.get(code)
        if terms is None:
            terms = set(tokenize(code))
            terms_by_code[code] = terms
        term_cache[(item["path"], item["statement_start"], item["statement_end"])] = terms
    term_seconds = time.monotonic() - term_started
    max_rank = max(rank_by_location.values(), default=1)
    max_file_rank = max(file_rank.values(), default=1)

    def statement_key(item):
        return item["path"], item["statement_start"], item["statement_end"]

    def add(item, selection_reason):
        key = statement_key(item)
        if key in selected_keys or len(selected) >= anchor_limit:
            return
        selected.append({**item, "selection_reason": selection_reason})
        selected_keys.add(key)
        selected_files.add(item["path"])
        selected_blocks.add((item["path"], item.get("block")))
        selected_types.add(item.get("anchor_type"))
        selected_sources.update(item.get("retrieval_sources", []))

    hypothesis_count = max(
        (len(item.get("hypothesis_matches", [])) for item in statements),
        default=0,
    )
    high_evidence_paths = ordered_paths[: min(10, len(ordered_paths))]

    # Balance exact evidence, Retrieval coverage, and MMR within 100 anchors.
    exact_limit = min(15, anchor_limit)
    for index in range(hypothesis_count):
        for path in high_evidence_paths:
            if len(selected) >= exact_limit:
                break
            candidates = [
                item
                for item in statements
                if item["path"] == path
                and statement_key(item) not in selected_keys
                and item["hypothesis_matches"][index]["symbol_match"]
                and (
                    item["hypothesis_matches"][index]["operation_overlap"] > 0
                    or item["hypothesis_matches"][index]["coverage"] >= 0.2
                )
            ]
            if candidates:
                add(
                    max(
                        candidates,
                        key=lambda item: (
                            item["hypothesis_matches"][index]["operation_overlap"],
                            item["hypothesis_matches"][index]["coverage"],
                            -rank_by_location[(item["path"], item["line"])],
                        ),
                    ),
                    f"hypothesis_{index + 1}_file_{file_rank[path]}_strong",
                )

    retrieval_limit = min(anchor_limit, len(selected) + 45)
    file_leaders = [
        next((item for item in statements if item["path"] == path), None)
        for path in ordered_paths
    ]
    for item in [*file_leaders, *statements]:
        if len(selected) >= retrieval_limit:
            break
        if item is not None:
            add(item, "retrieval_coverage")

    def pair_redundancy(item, chosen):
        terms = term_cache[statement_key(item)]
        chosen_terms = term_cache[statement_key(chosen)]
        intersection_size = len(terms & chosen_terms)
        union_size = len(terms) + len(chosen_terms) - intersection_size
        lexical = intersection_size / union_size if union_size else 0.0
        structural = 0.2 if item["path"] == chosen["path"] else 0.0
        if item.get("block") == chosen.get("block"):
            structural += 0.3
        return min(1.0, lexical + structural)

    remaining = [item for item in statements if statement_key(item) not in selected_keys]
    selection_started = time.monotonic()
    redundancy_by_key = {
        statement_key(item): max(
            (pair_redundancy(item, chosen) for chosen in selected),
            default=0.0,
        )
        for item in remaining
    }
    while remaining and len(selected) < anchor_limit:
        best = max(
            remaining,
            key=lambda item: (
                0.38 * (1.0 - (rank_by_location[(item["path"], item["line"])] - 1) / max_rank)
                + 0.16 * (1.0 - (file_rank[item["path"]] - 1) / max_file_rank)
                + 0.16 * min(item.get("hypothesis_coverage", 0.0), 1.0)
                + 0.12 * float(item.get("hypothesis_symbol_match", False))
                + 0.05 * (item["path"] not in selected_files)
                + 0.03 * ((item["path"], item.get("block")) not in selected_blocks)
                + 0.01 * (item.get("anchor_type") not in selected_types)
                + 0.01 * bool(set(item.get("retrieval_sources", [])) - selected_sources)
                - 0.30 * redundancy_by_key[statement_key(item)],
                -rank_by_location[(item["path"], item["line"])],
            ),
        )
        remaining.remove(best)
        add(best, "mmr")
        for item in remaining:
            key = statement_key(item)
            redundancy_by_key[key] = max(
                redundancy_by_key[key],
                pair_redundancy(item, best),
            )

    selection_seconds = time.monotonic() - selection_started
    logger.info(
        f"Statement shortlist selected {len(selected)}/{len(statements)} anchors: "
        f"group_seconds={grouped_seconds:.2f}, "
        f"term_seconds={term_seconds:.2f}, "
        f"selection_seconds={selection_seconds:.2f}, "
        f"total_seconds={time.monotonic() - started:.2f}"
    )
    output = []
    for anchor in selected:
        context_window = []
        statement_lines = anchor["statement_lines"]
        context_line_numbers = list(dict.fromkeys([
            statement_lines[0],
            anchor["line"],
            statement_lines[-1],
        ]))
        for line_number in context_line_numbers:
            context = item_by_location.get((anchor["path"], line_number))
            if context is not None:
                context_window.append([line_number, context["code"]])
        output.append(
            {
                **anchor,
                "candidate_role": "statement_anchor",
                "anchor_line": anchor["line"],
                "context_window": context_window,
            }
        )
    return output

def validate_files_and_lines_once(problem_statement, retrieval, code_graph, args, case_path):
    retrieval_variants = {"full": retrieval}
    candidates_by_path = {}
    variant_paths = {}
    for variant, retrieval in retrieval_variants.items():
        paths = []
        for candidate in retrieval["candidate_context"]:
            candidates_by_path.setdefault(candidate["path"], candidate)
            paths.append(candidate["path"])
        variant_paths[variant] = set(paths)
    ordered_paths = sorted(
        candidates_by_path,
        key=lambda path: min(
            (
                index
                for retrieval in retrieval_variants.values()
                for index, candidate in enumerate(retrieval["candidate_context"])
                if candidate["path"] == path
            ),
            default=len(candidates_by_path),
        ),
    )
    file_ids = {path: f"F{index}" for index, path in enumerate(ordered_paths, 1)}
    lines_by_location = {}
    variant_line_ids = {}
    for variant, retrieval in retrieval_variants.items():
        locations = []
        for item in retrieval["line_rankings"]:
            key = (item["path"], item["line"])
            if key not in lines_by_location:
                lines_by_location[key] = {
                    "location_id": f"L{len(lines_by_location) + 1}",
                    "path": item["path"],
                    "line": item["line"],
                    "code": item["code"],
                    "block": item.get("block"),
                    "block_kind": item.get("block_kind"),
                    "retrieval_sources": item.get("retrieval_sources", []),
                    "candidate_role": item.get("candidate_role", "anchor"),
                    "anchor_line": item.get("anchor_line", item["line"]),
                    "sources": item["sources"],
                    "anchor_type": item.get("anchor_type", "context"),
                    "statement_start": item.get("statement_start", item["line"]),
                    "statement_end": item.get("statement_end", item["line"]),
                    "statement_kind": item.get("statement_kind", "line"),
                    "hypothesis_matches": item.get("hypothesis_matches", []),
                    "raw_rank": item["raw_rank"],
                }
            locations.append(lines_by_location[key]["location_id"])
        variant_line_ids[variant] = locations
    candidate_files = []
    shortlist_limit = max(args.top_k * 10, 100)
    covered_lines = select_covered_lines(
        retrieval["line_rankings"], ordered_paths, code_graph, shortlist_limit
    )
    retained_lines = [
        {
            **lines_by_location[(item["path"], item["line"])],
            "candidate_role": item.get("candidate_role", "statement_anchor"),
            "anchor_line": item.get("anchor_line", item["line"]),
            "statement_start": item.get("statement_start", item["line"]),
            "statement_end": item.get("statement_end", item["line"]),
            "statement_lines": list(
                range(
                    item.get("statement_start", item["line"]),
                    item.get("statement_end", item["line"]) + 1,
                )
            ),
            "context_window": item.get("context_window", [[item["line"], item["code"]]]),
            "hypothesis_matches": item.get("hypothesis_matches", []),
            "selection_reason": item.get("selection_reason", "mmr"),
        }
        for item in covered_lines
    ]
    joint_token_limit = int(args.max_input_tokens * 0.9)
    prompt_infeasible = [
        item
        for item in retained_lines
        if litellm.token_counter(
            model=args.model,
            text=json.dumps(item["context_window"], ensure_ascii=False),
        ) > joint_token_limit
    ]
    if prompt_infeasible:
        logger.warning(
            "Joint Validator excluded prompt-infeasible Statements while retaining them in artifacts: "
            f"{[(item['path'], item['statement_start'], item['statement_end']) for item in prompt_infeasible]}"
        )
        infeasible_keys = {
            (item["path"], item["statement_start"], item["statement_end"])
            for item in prompt_infeasible
        }
        retained_lines = [
            item
            for item in retained_lines
            if (item["path"], item["statement_start"], item["statement_end"]) not in infeasible_keys
        ]
    if not retained_lines:
        raise ValueError("Joint Validator has no prompt-feasible Statement anchors")
    for index, item in enumerate(retained_lines, 1):
        item["source_location_id"] = item["location_id"]
        item["location_id"] = f"L{index}"
    retained_id_by_location = {
        (item["path"], item["line"]): item["location_id"]
        for item in retained_lines
    }
    for variant, retrieval in retrieval_variants.items():
        variant_line_ids[variant] = [
            retained_id_by_location[(item["path"], item["line"])]
            for item in retrieval["line_rankings"]
            if (item["path"], item["line"]) in retained_id_by_location
        ]
    retained_line_ids = {item["location_id"] for item in retained_lines}
    if len(retained_line_ids) < len(lines_by_location):
        logger.warning(
            f"Joint Validator deterministic shortlist retained {len(retained_line_ids)}/{len(lines_by_location)} lines; full candidates remain in artifacts"
        )
    for path in ordered_paths:
        candidate = candidates_by_path[path]
        candidate_files.append(
            {
                "file_id": file_ids[path],
                "path": path,
                "annotations": candidate.get("llm_roles", []),
                "retrieval_sources": candidate.get("retrieval_sources", []),
                "entities": [
                    [entity["name"].rsplit(":", 1)[-1], entity["type"]]
                    for entity in candidate.get("relevant_entities", [])[:2]
                ],
                "relations": [
                    [
                        str(edge.get("source", "")).rsplit(":", 1)[-1],
                        edge.get("type"),
                        str(edge.get("target", "")).rsplit(":", 1)[-1],
                    ]
                    for edge in candidate.get("local_execution_subgraph", {}).get("edges", [])
                    if edge.get("resolution", "exact") != "ambiguous"
                ][:3],
            }
        )
    compact_lines = [
        [
            item["location_id"],
            file_ids[item["path"]],
            item.get("block_kind"),
            item.get("anchor_type", "context"),
            [item["statement_start"], item["statement_end"]],
            item.get("context_window", [[item["line"], item["code"]]]),
            item.get("retrieval_sources", []),
            [
                [
                    index + 1,
                    int(match.get("symbol_match", False)),
                    match.get("operation_overlap", 0),
                    round(match.get("coverage", 0.0), 3),
                ]
                for index, match in enumerate(item.get("hypothesis_matches", []))
                if match.get("symbol_match")
                or match.get("operation_overlap", 0) > 0
                or match.get("coverage", 0.0) > 0
            ],
            item.get("selection_reason", "mmr"),
            item["raw_rank"],
        ]
        for item in retained_lines
    ]
    payload = {
        "prompt_schema": {
            "candidate_line": [
                "location_id",
                "file_id",
                "block_kind",
                "anchor_type",
                "statement_range",
                "statement_lines",
                "retrieval_sources",
                "hypothesis_matches_as_[hypothesis_id,symbol_match,operation_overlap,coverage]",
                "selection_reason",
                "raw_rank",
            ]
        },
        "issue": problem_statement,
        "fault_hypotheses": retrieval.get("fault_hypotheses", []),
        "candidate_files": candidate_files,
        "candidate_lines": compact_lines,
    }
    messages = [
        {"role": "system", "content": JOINT_VALIDATOR_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    prompt_tokens = litellm.token_counter(model=args.model, messages=messages)
    if prompt_tokens > joint_token_limit:
        oversized = max(
            retained_lines,
            key=lambda item: litellm.token_counter(
                model=args.model,
                text=json.dumps(item.get("context_window", []), ensure_ascii=False),
            ),
        )
        oversized_tokens = litellm.token_counter(
            model=args.model,
            text=json.dumps(oversized.get("context_window", []), ensure_ascii=False),
        )
        raise ValueError(
            f"Joint Validator input has {prompt_tokens} tokens and exceeds the safe limit {joint_token_limit}; "
            f"largest Statement context is {oversized_tokens} tokens at "
            f"{oversized['path']}:{oversized['statement_start']}-{oversized['statement_end']}"
        )
    result = call_llm(
        messages,
        args.model,
        args.openai_base_url,
        args.openai_api_key,
        pydantic_object=JointValidatorResult,
        log_file=case_path / "calls_llm_joint_validator.jsonl",
        log_metadata={
            "prompt_tokens": prompt_tokens,
            "candidate_file_count": len(candidate_files),
            "candidate_line_count": len(lines_by_location),
            "shortlist_line_count": len(retained_lines),
        },
        temperature=0,
        max_tokens=args.max_output_tokens,
        response_format={"type": "json_object"},
    )
    returned_line_ids = [item["location_id"] for item in result["ranked_locations"]]
    valid_returned_ids = list(
        dict.fromkeys(location_id for location_id in returned_line_ids if location_id in retained_line_ids)
    )
    missing_line_ids = retained_line_ids.difference(valid_returned_ids)
    if missing_line_ids:
        logger.warning(
            f"Joint Validator omitted {len(missing_line_ids)} anchor IDs; appended them by Retrieval rank"
        )
        valid_returned_ids.extend(
            item["location_id"]
            for item in retained_lines
            if item["location_id"] in missing_line_ids
        )
    result["ranked_locations"] = [
        {"location_id": location_id}
        for location_id in valid_returned_ids
    ]
    paths_by_id = {identifier: path for path, identifier in file_ids.items()}
    valid_scores = {
        item["file_id"]: item
        for item in result["scored_files"]
        if item["file_id"] in paths_by_id
    }
    if set(valid_scores) != set(paths_by_id):
        missing = sorted(set(paths_by_id) - set(valid_scores))
        raise ValueError(f"Joint Validator omitted candidate file IDs: {missing}")
    locations_by_id = {
        item["location_id"]: item for item in retained_lines
    }
    validator_results = {}
    for variant, paths in variant_paths.items():
        enabled_file_ids = {file_ids[path] for path in paths}
        ranked_files = sorted(
            (valid_scores[file_id] for file_id in enabled_file_ids),
            key=lambda item: (-item["score"], paths_by_id[item["file_id"]]),
        )
        allowed_line_ids = set(variant_line_ids[variant]).intersection(retained_line_ids)
        ranked_statements = []
        invalid_ids = []
        seen_statements = set()
        for item in result.get("ranked_locations", []):
            location_id = item["location_id"]
            if location_id not in allowed_line_ids or location_id not in locations_by_id:
                invalid_ids.append(location_id)
                continue
            location = locations_by_id[location_id]
            key = (
                location["path"],
                location["statement_start"],
                location["statement_end"],
            )
            if key in seen_statements:
                continue
            seen_statements.add(key)
            ranked_statements.append(
                {
                    **location,
                    "score": item.get("score"),
                    "roles": item.get("roles", []),
                    "reason": item.get("reason", ""),
                }
            )
        ranked_locations = []
        seen_locations = set()
        line_type_priority = {
            "assignment": 0,
            "condition": 1,
            "call": 2,
            "return_or_raise": 3,
            "import": 4,
            "signature": 5,
            "context": 6,
        }
        for statement_rank, statement in enumerate(ranked_statements, 1):
            line_order = sorted(
                statement["statement_lines"],
                key=lambda line: (
                    line_type_priority.get(
                        lines_by_location.get((statement["path"], line), {}).get(
                            "anchor_type", "context"
                        ),
                        6,
                    ),
                    abs(line - statement["line"]),
                    line,
                ),
            )
            for line_number in line_order:
                key = (statement["path"], line_number)
                if key in seen_locations:
                    continue
                seen_locations.add(key)
                source = lines_by_location.get(key, {})
                ranked_locations.append(
                    {
                        **statement,
                        "line": line_number,
                        "code": source.get("code", statement["code"]),
                        "statement_rank": statement_rank,
                    }
                )
        file_details = [
            {
                "file_id": item["file_id"],
                "path": paths_by_id[item["file_id"]],
                "score": item["score"],
                "reason": item.get("reason", ""),
            }
            for item in ranked_files
        ]
        selected_blocks = []
        for location in ranked_locations:
            containing = [
                (start, end, kind, name)
                for start, end, kind, name in structural_ranges(code_graph, location["path"])
                if start <= location["line"] <= end
            ]
            if containing:
                start, end, kind, name = min(containing, key=lambda value: value[1] - value[0])
                block = {
                    "path": location["path"],
                    "start_line": start,
                    "end_line": end,
                    "kind": kind,
                    "name": name,
                }
                if block not in selected_blocks:
                    selected_blocks.append(block)
        atomic_json(
            case_path / f"structure_rankings_{variant}.json",
            {"selected_blocks": selected_blocks, "ranked_fragments": selected_blocks},
        )
        atomic_json(
            case_path / f"line_candidates_{variant}.json",
            {
                "variant": variant,
                "all_candidates": [
                    lines_by_location[(item["path"], item["line"])]
                    for item in retrieval_variants[variant]["line_rankings"]
                ],
                "batch_shortlist": retained_lines,
                "validator_statements": ranked_statements,
                "offline_expanded_lines": ranked_locations,
            },
        )
        evidence_channels = {"E", "I", "R"}
        posthoc_rankings = {
            f"without_{channel}": [
                item
                for item in ranked_locations
                if (set(item.get("retrieval_sources", [])) & evidence_channels) != {channel}
            ][: args.top_k]
            for channel in evidence_channels
        }
        posthoc_rankings["without_V"] = [
            lines_by_location[(item["path"], item["line"])]
            for item in retrieval["line_rankings"][: args.top_k]
        ]
        validator_results[variant] = {
            "status": "completed" if ranked_locations else "failed",
            "file_validator_rankings": file_details,
            "adaptive_file_ranking": {
                "counterfactual_scores": file_details,
                "policy": "single_joint_counterfactual_validation",
            },
            "invalid_prediction_file_ids": [],
            "ranked_locations": ranked_locations[: args.top_k],
            "complete_ranked_locations": ranked_locations,
            "complete_ranked_statements": ranked_statements,
            "posthoc_rankings": posthoc_rankings,
            "ranked_files": list(dict.fromkeys(item["path"] for item in ranked_locations[: args.top_k])),
            "invalid_prediction_location_ids": invalid_ids,
            "line_validator_batches": [{"joint_validator": True, "prompt_tokens": prompt_tokens}],
            "prompt_omissions": [],
            "error": None if ranked_locations else "Joint Validator produced no valid candidate line",
        }
    return validator_results, prompt_tokens


def final_validator_messages(problem_statement, fault_hypotheses, source_items):
    return [
        {"role": "system", "content": FINAL_VALIDATOR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "issue": problem_statement,
                    "fault_hypotheses": fault_hypotheses,
                    "source_files": source_items,
                },
                ensure_ascii=False,
            ),
        },
    ]


def numbered_source(lines, start_line):
    return "\n".join(
        f"{line_number}: {line}"
        for line_number, line in enumerate(lines, start_line)
    )


def split_final_source_item(item, problem_statement, fault_hypotheses, model, token_budget):
    messages = final_validator_messages(problem_statement, fault_hypotheses, [item])
    prompt_tokens = litellm.token_counter(model=model, messages=messages)
    if prompt_tokens <= token_budget:
        return [item]
    if item["start_line"] == item["end_line"]:
        raise ValueError(
            f"One complete source line in {item['path']} requires {prompt_tokens} tokens and exceeds {token_budget}"
        )
    source_lines = item["source"].splitlines()
    middle_offset = len(source_lines) // 2
    if middle_offset < 1:
        raise ValueError(f"Cannot split oversized source item: {item['path']}")
    fragments = []
    for fragment_index, fragment_lines in enumerate(
        (source_lines[:middle_offset], source_lines[middle_offset:]),
        1,
    ):
        start_line = item["start_line"] if fragment_index == 1 else item["start_line"] + middle_offset
        end_line = start_line + len(fragment_lines) - 1
        fragment = {
            **item,
            "source_id": f"{item['source_id']}.P{fragment_index}",
            "start_line": start_line,
            "end_line": end_line,
            "source": "\n".join(fragment_lines),
            "complete_file": False,
        }
        fragments.extend(
            split_final_source_item(
                fragment,
                problem_statement,
                fault_hypotheses,
                model,
                token_budget,
            )
        )
    return fragments


def deterministic_final_candidates(
    selected_paths,
    source_lines_by_path,
    ranked_statements,
    existing_candidates,
    top_k,
):
    candidates = list(existing_candidates)
    seen_locations = {(item["path"], item["line"]) for item in candidates}

    def add(path, line, source):
        if path not in source_lines_by_path or not 1 <= line <= len(source_lines_by_path[path]):
            return
        key = (path, line)
        if key in seen_locations:
            return
        seen_locations.add(key)
        lines = source_lines_by_path[path]
        candidates.append(
            {
                "candidate_id": f"C{len(candidates) + 1}",
                "path": path,
                "line": line,
                "code": lines[line - 1],
                "context": [
                    [context_line, lines[context_line - 1]]
                    for context_line in range(
                        max(1, line - 2),
                        min(len(lines), line + 2) + 1,
                    )
                ],
                "batch_index": source.get("batch_index", 0),
                "batch_rank": source.get("batch_rank", len(candidates) + 1),
                "file_rank": selected_paths.index(path) + 1,
                "fallback_source": source.get("fallback_source"),
            }
        )

    for statement_rank, statement in enumerate(ranked_statements, 1):
        add(
            statement["path"],
            statement.get("anchor_line", statement["line"]),
            {
                "batch_rank": statement_rank,
                "fallback_source": "joint_statement",
            },
        )
        if len(candidates) >= top_k:
            return candidates
    for path in selected_paths:
        for line, code in enumerate(source_lines_by_path[path], 1):
            if not code.strip():
                continue
            add(
                path,
                line,
                {
                    "batch_rank": line,
                    "fallback_source": "source_file_order",
                },
            )
            if len(candidates) >= top_k:
                return candidates
    return candidates


def merge_final_candidates(problem_statement, candidates, args, case_path):
    candidate_payload = [
        {
            "candidate_id": item["candidate_id"],
            "path": item["path"],
            "line": item["line"],
            "code": item["code"],
            "context": item["context"],
            "batch_rank": item["batch_rank"],
        }
        for item in candidates
    ]
    messages = [
        {"role": "system", "content": FINAL_MERGER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {"issue": problem_statement, "candidate_lines": candidate_payload},
                ensure_ascii=False,
            ),
        },
    ]
    prompt_tokens = litellm.token_counter(model=args.model, messages=messages)
    token_budget = int(args.max_input_tokens * args.line_token_budget_ratio)
    if prompt_tokens > token_budget:
        raise ValueError(
            f"Final Validator merger requires {prompt_tokens} tokens and exceeds {token_budget}"
        )
    result = call_llm(
        messages,
        args.model,
        args.openai_base_url,
        args.openai_api_key,
        pydantic_object=FinalMergerResult,
        log_file=case_path / "calls_llm_final_validator_merge.jsonl",
        log_metadata={
            "prompt_tokens": prompt_tokens,
            "candidate_count": len(candidates),
        },
        temperature=0,
        max_tokens=args.max_output_tokens,
        response_format={"type": "json_object"},
    )
    candidates_by_id = {item["candidate_id"]: item for item in candidates}
    returned_ids = list(
        dict.fromkeys(
            candidate_id
            for candidate_id in result["ranked_location_ids"]
            if candidate_id in candidates_by_id
        )
    )
    if len(returned_ids) < min(args.top_k, len(candidates)):
        logger.warning(
            f"Final Validator merger returned {len(returned_ids)}/{min(args.top_k, len(candidates))} required lines; appending batch-ranked candidates"
        )
        returned_ids.extend(
            item["candidate_id"]
            for item in candidates
            if item["candidate_id"] not in returned_ids
        )
    return [candidates_by_id[candidate_id] for candidate_id in returned_ids[: args.top_k]], prompt_tokens


def validate_final_lines(
    problem_statement,
    fault_hypotheses,
    ranked_statements,
    code_graph,
    args,
    case_path,
):
    selected_paths = list(dict.fromkeys(item["path"] for item in ranked_statements))[:10]
    selected_statements = [
        item for item in ranked_statements if item["path"] in selected_paths
    ]
    if not selected_paths:
        raise ValueError("Joint Validator produced no Statement paths for Final Validator")
    file_ids = {path: f"F{index}" for index, path in enumerate(selected_paths, 1)}
    paths_by_file_id = {identifier: path for path, identifier in file_ids.items()}
    token_budget = int(args.max_input_tokens * args.line_token_budget_ratio)
    source_items = []
    for file_rank, path in enumerate(selected_paths, 1):
        source_lines = code_graph.nodes[path].get("code", "").splitlines()
        if not source_lines:
            logger.warning(f"Final Validator candidate file has no source lines: {path}")
            continue
        item = {
            "source_id": file_ids[path],
            "file_id": file_ids[path],
            "file_rank": file_rank,
            "path": path,
            "start_line": 1,
            "end_line": len(source_lines),
            "complete_file": True,
            "source": numbered_source(source_lines, 1),
        }
        fragments = split_final_source_item(
            item,
            problem_statement,
            fault_hypotheses,
            args.model,
            token_budget,
        )
        if len(fragments) > 1:
            logger.warning(
                f"Final Validator split oversized complete file {path} into {len(fragments)} exhaustive fragments"
            )
        source_items.extend(fragments)
    if not source_items:
        raise ValueError("Final Validator has no non-empty source files")
    batches = split_prompt_batches(
        source_items,
        lambda items: final_validator_messages(
            problem_statement,
            fault_hypotheses,
            items,
        ),
        args.model,
        token_budget,
    )
    candidates = []
    seen_locations = set()
    batch_details = []
    source_lines_by_path = {
        path: code_graph.nodes[path].get("code", "").splitlines()
        for path in selected_paths
    }
    for batch_index, batch in enumerate(batches, 1):
        messages = final_validator_messages(problem_statement, fault_hypotheses, batch)
        prompt_tokens = litellm.token_counter(model=args.model, messages=messages)
        result = call_llm(
            messages,
            args.model,
            args.openai_base_url,
            args.openai_api_key,
            pydantic_object=FinalValidatorResult,
            log_file=case_path / "calls_llm_final_validator.jsonl",
            log_metadata={
                "batch_index": batch_index,
                "batch_count": len(batches),
                "prompt_tokens": prompt_tokens,
                "source_items": [item["source_id"] for item in batch],
            },
            temperature=0,
            max_tokens=args.max_output_tokens,
            response_format={"type": "json_object"},
        )
        ranges_by_file_id = {}
        for item in batch:
            ranges_by_file_id.setdefault(item["file_id"], []).append(
                (item["start_line"], item["end_line"])
            )
        invalid_locations = []
        accepted_locations = []
        for batch_rank, location in enumerate(result["ranked_locations"], 1):
            file_id = location["file_id"]
            line = location["line"]
            path = paths_by_file_id.get(file_id)
            if (
                path is None
                or not any(start <= line <= end for start, end in ranges_by_file_id.get(file_id, []))
                or line > len(source_lines_by_path[path])
            ):
                invalid_locations.append({"file_id": file_id, "line": line})
                continue
            key = (path, line)
            if key in seen_locations:
                continue
            seen_locations.add(key)
            lines = source_lines_by_path[path]
            candidate = {
                "candidate_id": f"C{len(candidates) + 1}",
                "path": path,
                "line": line,
                "code": lines[line - 1],
                "context": [
                    [context_line, lines[context_line - 1]]
                    for context_line in range(max(1, line - 2), min(len(lines), line + 2) + 1)
                ],
                "batch_index": batch_index,
                "batch_rank": batch_rank,
                "file_rank": selected_paths.index(path) + 1,
            }
            candidates.append(candidate)
            accepted_locations.append(candidate)
        if len(accepted_locations) < min(args.top_k, sum(len(item["source"].splitlines()) for item in batch)):
            logger.warning(
                f"Final Validator batch {batch_index} returned only {len(accepted_locations)} valid lines; deterministic candidates will fill missing ranks"
            )
        if invalid_locations:
            logger.warning(
                f"Final Validator batch {batch_index} returned invalid locations: {invalid_locations}"
            )
        batch_details.append(
            {
                "batch_index": batch_index,
                "prompt_tokens": prompt_tokens,
                "source_items": [item["source_id"] for item in batch],
                "raw_ranked_locations": result["ranked_locations"],
                "invalid_locations": invalid_locations,
                "accepted_locations": accepted_locations,
            }
        )
    candidates = deterministic_final_candidates(
        selected_paths,
        source_lines_by_path,
        selected_statements,
        candidates,
        args.top_k,
    )
    if not candidates:
        raise ValueError("Final Validator and deterministic fallback produced no valid exact-line candidates")
    if len(batches) == 1:
        final_candidates = candidates[: args.top_k]
        merger_tokens = 0
    else:
        final_candidates, merger_tokens = merge_final_candidates(
            problem_statement,
            candidates,
            args,
            case_path,
        )
    final_locations = [
        {
            "path": item["path"],
            "line": item["line"],
            "code": item["code"],
            "candidate_id": item["candidate_id"],
            "batch_index": item["batch_index"],
            "batch_rank": item["batch_rank"],
            "file_rank": item["file_rank"],
        }
        for item in final_candidates
    ]
    artifact = {
        "statement_limit": len(ranked_statements),
        "selected_statements": selected_statements,
        "selected_files": selected_paths,
        "source_items": source_items,
        "token_budget": token_budget,
        "batches": batch_details,
        "batch_candidates": candidates,
        "final_locations": final_locations,
        "merger_prompt_tokens": merger_tokens,
    }
    atomic_json(case_path / "final_validator.json", artifact)
    logger.info(
        f"Final Validator selected {len(selected_paths)} files from {len(selected_statements)} Statements: "
        f"{selected_paths[:2]}, batches={len(batches)}, final={[(item['path'], item['line']) for item in final_locations[:2]]}"
    )
    return final_locations, artifact


def structural_ranges(code_graph, path):
    code = code_graph.nodes[path].get("code", "")
    line_count = len(code.splitlines())
    if line_count == 0:
        return []
    definitions = sorted(
        (
            data.get("start_line", 1),
            data.get("end_line", data.get("start_line", 1)),
            data.get("type", "module"),
            node,
        )
        for node, data in code_graph.nodes(data=True)
        if data.get("file_path") == path and data.get("type") in {"function", "class"}
    )
    top_level = []
    for start, end, kind, node in definitions:
        if any(parent_start <= start and end <= parent_end for parent_start, parent_end, _, _ in top_level):
            continue
        top_level.append((start, end, kind, node))
    ranges = []
    cursor = 1
    for start, end, kind, node in top_level:
        if cursor < start:
            ranges.append((cursor, start - 1, "module", f"{path}:module:{cursor}-{start - 1}"))
        ranges.append((start, end, kind, node))
        cursor = max(cursor, end + 1)
    if cursor <= line_count:
        ranges.append((cursor, line_count, "module", f"{path}:module:{cursor}-{line_count}"))
    return ranges or [(1, line_count, "module", f"{path}:module:1-{line_count}")]




















def run_instance(instance, args):
    instance_id = instance["instance_id"]
    case_path = args.output_base / instance_id
    repo_path = args.repo_base / instance_id
    if not repo_path.is_dir():
        logger.error(f"Repository not found: {repo_path}")
        return False
    repo_sha256 = repository_fingerprint(repo_path)
    configuration = run_configuration(instance, args, repo_sha256)
    if completed(case_path, configuration) and not args.overwrite:
        logger.info(f"Completed instance skipped: {instance_id}")
        return True
    if case_path.exists():
        shutil.rmtree(case_path)
    case_path.mkdir(parents=True)
    atomic_json(case_path / "input_raw.json", instance)
    atomic_json(case_path / "input.json", configuration)
    trace(case_path, "pipeline", "started", instance_id=instance_id)
    log_case(case_path, "INFO", f"Started {instance_id}")
    started = time.monotonic()

    try:
        repo_path = args.repo_base / instance_id
        if not repo_path.is_dir():
            raise FileNotFoundError(f"Repository not found: {repo_path}")

        graph_started = time.monotonic()
        cache_path = args.cache_base / f"{instance['repo'].replace('/', '__')}-{instance['base_commit']}.pickle"
        code_graph, entity_index, cache_hit = build_complete_codekg(
            repo_path,
            case_path / ".graph_stage",
            cache_path,
        )
        annotation_started = time.monotonic()
        annotation_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.model)
        annotation_cache_path = args.cache_base / (
            f"{instance['repo'].replace('/', '__')}-{instance['base_commit']}-"
            f"{annotation_model}-annotation-cache.json"
        )
        annotation_artifact = annotate_repository(code_graph, args, case_path, annotation_cache_path)
        entity_index = build_entity_index(code_graph)
        analyzer_artifact = analyze_defect(
            code_graph,
            instance["problem_statement"],
            annotation_artifact,
            args,
            case_path,
        )
        annotation_seconds = time.monotonic() - annotation_started
        atomic_json(case_path / "codekg.json", graph_to_dict(code_graph))
        graph_seconds = time.monotonic() - graph_started
        trace(
            case_path,
            "graph",
            "completed",
            nodes=code_graph.number_of_nodes(),
            edges=code_graph.number_of_edges(),
            cache_hit=cache_hit,
            cache_path=str(cache_path),
            seconds=graph_seconds,
        )
        log_case(
            case_path,
            "INFO",
            f"Graph complete: nodes={code_graph.number_of_nodes()}, edges={code_graph.number_of_edges()}, cache_hit={cache_hit}",
        )

        retrieval_started = time.monotonic()
        direct_file_paths = analyzer_artifact["E"]
        retrieval = retrieve_files(
            code_graph,
            instance["problem_statement"],
            seed_count=args.seed_count,
            candidate_count=args.candidate_count,
            lexical_weight=args.lexical_weight,
            entity_budget=args.entity_budget,
            entity_index=entity_index,
            protected_file_paths=direct_file_paths,
            fault_hypotheses=analyzer_artifact["fault_hypotheses"],
            analyzer_rankings={
                "E": analyzer_artifact["E"],
                "I": analyzer_artifact["I"],
                "R": analyzer_artifact["R"]["matched_files"],
                "I_evidence": analyzer_artifact["I_file_evidence"],
            },
            **FULL_EVIDENCE,
        )
        enrich_retrieval_with_analyzer(retrieval, analyzer_artifact, FULL_EVIDENCE)
        retrieval["fault_hypotheses"] = analyzer_artifact["fault_hypotheses"]
        attach_local_subgraphs(code_graph, retrieval)
        retrieval_variants = {"full": retrieval}
        retrieval_variant_seconds = {"full": time.monotonic() - retrieval_started}
        retrieval_output_started = time.monotonic()
        atomic_json(case_path / "retrieval.json", retrieval_artifact(retrieval_variants))
        logger.info(
            f"Retrieval artifact saved for {instance_id}: "
            f"bytes={(case_path / 'retrieval.json').stat().st_size}, "
            f"seconds={time.monotonic() - retrieval_output_started:.2f}"
        )
        gold_files = {
            str(path).replace("\\", "/").removeprefix(f"{instance_id}/")
            for path in instance.get("patch", {})
        } if isinstance(instance.get("patch"), dict) else set()
        retrieval_recall_at_20 = {}
        for variant, retrieval in retrieval_variants.items():
            top_20 = {item["path"] for item in retrieval["file_rankings"][:20]}
            retrieval_recall_at_20[variant] = (
                len(gold_files.intersection(top_20)) / len(gold_files) if gold_files else None
            )
        atomic_json(case_path / "retrieval_recall_at_20.json", {"gold_files": sorted(gold_files), "recall_at_20": retrieval_recall_at_20})
        retrieval_seconds = time.monotonic() - retrieval_started
        trace(
            case_path,
            "retrieval",
            "completed",
            variants=list(retrieval_variants),
            seconds=retrieval_seconds,
        )

        validator_started = time.monotonic()
        validator_results, joint_validator_tokens = validate_files_and_lines_once(
            instance["problem_statement"],
            retrieval,
            code_graph,
            args,
            case_path,
        )
        joint_validator_seconds = time.monotonic() - validator_started
        atomic_json(case_path / "validator_results.json", validator_results)
        logger.info(
            f"Joint Validator complete for {instance_id}: "
            f"{[(item['path'], item['line']) for item in validator_results['full']['ranked_locations'][:2]]}, "
            f"prompt_tokens={joint_validator_tokens}"
        )
        final_validator_started = time.monotonic()
        final_locations, final_validator_artifact = validate_final_lines(
            instance["problem_statement"],
            retrieval.get("fault_hypotheses", []),
            validator_results["full"]["complete_ranked_statements"],
            code_graph,
            args,
            case_path,
        )
        final_validator_seconds = time.monotonic() - final_validator_started
        validator_seconds = time.monotonic() - validator_started
        validator_variant_seconds = {"full": validator_seconds}
        for result in validator_results.values():
            result["joint_ranked_locations"] = result["ranked_locations"]
            result["ranked_locations"] = final_locations
            result["complete_ranked_locations"] = final_locations
            result["ranked_files"] = list(
                dict.fromkeys(item["path"] for item in final_locations)
            )
            result["final_validator"] = {
                "statement_limit": final_validator_artifact["statement_limit"],
                "selected_files": final_validator_artifact["selected_files"],
                "batch_count": len(final_validator_artifact["batches"]),
                "token_budget": final_validator_artifact["token_budget"],
                "merger_prompt_tokens": final_validator_artifact["merger_prompt_tokens"],
            }
            result["seconds"] = validator_seconds
        atomic_json(case_path / "validator_results.json", validator_results)
        if not final_locations:
            raise ValueError("Final Validator produced no valid line predictions")
        final_files = list(dict.fromkeys(item["path"] for item in final_locations))
        gold_lines = set()
        if isinstance(instance.get("patch"), dict):
            hunk_pattern = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
            for raw_path, patch in instance["patch"].items():
                path = str(raw_path).replace("\\", "/").removeprefix(f"{instance_id}/")
                old_line = None
                changed = []
                context = []
                for text in patch.get("before", []):
                    match = hunk_pattern.match(text)
                    if match:
                        old_line = int(match.group(1))
                    elif old_line is not None and text.startswith("-") and not text.startswith("---"):
                        changed.append(old_line)
                        old_line += 1
                    elif old_line is not None and text.startswith("+") and not text.startswith("+++"):
                        continue
                    elif old_line is not None:
                        context.append(old_line)
                        old_line += 1
                gold_lines.update((path, line) for line in (changed or context))
        stage_diagnostics = {"gold_files": sorted(gold_files), "gold_lines": sorted(gold_lines), "variants": {}}
        for variant, retrieval in retrieval_variants.items():
            result = validator_results[variant]
            structure_path = case_path / f"structure_rankings_{variant}.json"
            line_path = case_path / f"line_candidates_{variant}.json"
            structure = json.loads(structure_path.read_text(encoding="utf-8")) if structure_path.is_file() else {}
            line_candidates = json.loads(line_path.read_text(encoding="utf-8")) if line_path.is_file() else {}
            all_line_locations = {(item["path"], item["line"]) for item in line_candidates.get("all_candidates", [])}
            shortlist_line_locations = {
                (item["path"], line)
                for item in line_candidates.get("batch_shortlist", [])
                for line in range(
                    item.get("statement_start", item["line"]),
                    item.get("statement_end", item["line"]) + 1,
                )
            }
            final_line_locations = {(item["path"], item["line"]) for item in result["ranked_locations"]}
            selected_ranges = [
                (item["path"], item["start_line"], item["end_line"])
                for item in structure.get("selected_blocks", [])
            ]
            function_gold = {
                location
                for location in gold_lines
                if any(path == location[0] and start <= location[1] <= end for path, start, end in selected_ranges)
            }
            stage_diagnostics["variants"][variant] = {
                "E_file_recall": len(gold_files.intersection(analyzer_artifact["E"])) / len(gold_files) if gold_files else None,
                "I_file_recall": len(gold_files.intersection(analyzer_artifact["I"])) / len(gold_files) if gold_files else None,
                "EIR_file_recall": len(gold_files.intersection(analyzer_artifact["channels_by_file"])) / len(gold_files) if gold_files else None,
                "retrieval_top_20_file_recall": retrieval_recall_at_20[variant],
                "validator_top_10_file_recall": len(gold_files.intersection(item["path"] for item in result["file_validator_rankings"][:10])) / len(gold_files) if gold_files else None,
                "function_candidate_file_recall": len(gold_files.intersection(item["path"] for item in structure.get("ranked_fragments", []))) / len(gold_files) if gold_files else None,
                "function_selected_file_recall": len(gold_files.intersection(item["path"] for item in structure.get("selected_blocks", []))) / len(gold_files) if gold_files else None,
                "line_candidate_file_recall": len(gold_files.intersection(item["path"] for item in line_candidates.get("all_candidates", []))) / len(gold_files) if gold_files else None,
                "line_shortlist_file_recall": len(gold_files.intersection(item["path"] for item in line_candidates.get("batch_shortlist", []))) / len(gold_files) if gold_files else None,
                "final_file_recall": len(gold_files.intersection(item["path"] for item in result["ranked_locations"])) / len(gold_files) if gold_files else None,
                "gold_line_in_selected_function": len(function_gold) / len(gold_lines) if gold_lines else None,
                "line_candidate_recall": len(gold_lines.intersection(all_line_locations)) / len(gold_lines) if gold_lines else None,
                "line_shortlist_recall": len(gold_lines.intersection(shortlist_line_locations)) / len(gold_lines) if gold_lines else None,
                "final_line_recall": len(gold_lines.intersection(final_line_locations)) / len(gold_lines) if gold_lines else None,
                "gold_line_stage": {
                    f"{path}:{line}": {
                        "selected_function": (path, line) in function_gold,
                        "generated": (path, line) in all_line_locations,
                        "shortlisted": (path, line) in shortlist_line_locations,
                        "final": (path, line) in final_line_locations,
                    }
                    for path, line in sorted(gold_lines)
                },
            }
        atomic_json(case_path / "stage_diagnostics.json", stage_diagnostics)

        output = {
            "instance_id": instance_id,
            "status": "completed",
            "ranked_locations": final_locations,
            "ranked_files": final_files,
            "ranking_details": final_locations,
            "invalid_prediction_location_ids": validator_results["full"]["invalid_prediction_location_ids"],
            "artifacts": {
                "input_raw": "input_raw.json",
                "input": "input.json",
                "codekg": "codekg.json",
                "annotation": "annotation.json",
                "analyzer": "analyzer.json",
                "retrieval": "retrieval.json",
                "retrieval_recall_at_20": "retrieval_recall_at_20.json",
                "stage_diagnostics": "stage_diagnostics.json",
                "validator_results": "validator_results.json",
                "llm_annotation_calls_raw": "calls_llm_annotation.jsonl",
                "llm_analyzer_calls_raw": "calls_llm_analyzer.jsonl",
                "llm_joint_validator_calls_raw": "calls_llm_joint_validator.jsonl",
                "final_validator": "final_validator.json",
                "llm_final_validator_calls_raw": "calls_llm_final_validator.jsonl",
                "llm_final_validator_merge_calls_raw": "calls_llm_final_validator_merge.jsonl",
                "trace": "trace.jsonl",
                "run_log": "run.log",
            },
            "timing": {
                "graph_seconds": graph_seconds,
                "graph_cache_hit": cache_hit,
                "annotation_analyzer_seconds": annotation_seconds,
                "annotation_selected_file_count": len(annotation_artifact["selected_files"]),
                "retrieval_seconds": retrieval_seconds,
                "retrieval_variant_seconds": retrieval_variant_seconds,
                "validator_seconds": validator_seconds,
                "joint_validator_seconds": joint_validator_seconds,
                "final_validator_seconds": final_validator_seconds,
                "validator_variant_seconds": validator_variant_seconds,
                "total_seconds": time.monotonic() - started,
            },
        }
        atomic_json(case_path / "output.json", output)
        trace(
            case_path,
            "pipeline",
            "completed",
            ranked_locations=[
                {"path": item["path"], "line": item["line"]} for item in output["ranked_locations"]
            ],
            ranked_files=output["ranked_files"],
        )
        log_case(
            case_path,
            "INFO",
            f"Completed {instance_id}: {[(item['path'], item['line']) for item in final_locations]}",
        )
        logger.info(
            f"Completed {instance_id}: {[(item['path'], item['line']) for item in final_locations[:2]]}"
        )
        return True
    except Exception as error:
        output = {
            "instance_id": instance_id,
            "status": "failed",
            "ranked_locations": [],
            "ranked_files": [],
            "error": str(error),
            "error_type": type(error).__name__,
            "total_seconds": time.monotonic() - started,
        }
        atomic_json(case_path / "output.json", output)
        trace(case_path, "pipeline", "failed", error=str(error), error_type=type(error).__name__)
        log_case(case_path, "ERROR", f"Failed {instance_id}: {error}")
        logger.error(f"Failed {instance_id}: {error}")
        return False


def main():
    load_dotenv(PROJECT_ROOT / "scripts" / ".env")
    parser = argparse.ArgumentParser(description="Run Issue-guided CodeKG localization")
    parser.add_argument("--dataset-json", type=Path, default=PROJECT_ROOT / "datasets/Hydrangea/hydrangea_defect.json")
    parser.add_argument("--repo-base", type=Path, default=PROJECT_ROOT / "datasets/Hydrangea/repo")
    parser.add_argument("--output-base", type=Path, default=DEFAULT_RESULTS / os.getenv("LLM_MODEL", "model"))
    parser.add_argument("--cache-base", type=Path, default=PROJECT_ROOT / "results" / "results_main" / "._cache")
    parser.add_argument("--instance-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--model", default=os.getenv("LLM_MODEL"))
    parser.add_argument("--openai-base-url", default=os.getenv("LLM_BASE_URL"))
    parser.add_argument("--openai-api-key", default=os.getenv("LLM_API_KEY"))
    parser.add_argument("--annotation-seed-count", type=int, default=5)
    parser.add_argument("--annotation-expanded-count", type=int, default=5)
    parser.add_argument("--seed-count", type=int, default=30)
    parser.add_argument("--candidate-count", type=int, default=20)
    parser.add_argument("--entity-budget", type=int, default=40)
    parser.add_argument("--line-token-budget-ratio", type=float, default=0.9)
    parser.add_argument("--lexical-weight", type=float, default=0.65)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-output-tokens", type=int, default=8192)
    parser.add_argument("--max-input-tokens", type=int, default=180000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.max_workers < 1:
        parser.error("--max-workers must be at least 1")
    if (
        args.annotation_seed_count < 1
        or args.annotation_expanded_count < 1
        or args.seed_count < 1
        or args.candidate_count < 1
        or args.entity_budget < 1
        or not 0 < args.line_token_budget_ratio <= 1
        or not 0 <= args.lexical_weight <= 1
    ):
        parser.error("Invalid retrieval configuration")
    if not 1 <= args.top_k <= 10:
        parser.error("--top-k must be between 1 and 10")
    if args.max_input_tokens <= args.max_output_tokens:
        parser.error("--max-input-tokens must exceed --max-output-tokens")
    if not args.model or not args.openai_base_url or not args.openai_api_key:
        parser.error("Missing model, base URL, or API key")

    args.dataset_json = args.dataset_json.expanduser().resolve()
    args.repo_base = args.repo_base.expanduser().resolve()
    args.output_base = args.output_base.expanduser().resolve()
    args.cache_base = args.cache_base.expanduser().resolve()
    dataset = json.loads(args.dataset_json.read_text(encoding="utf-8"))
    instances = selected_instances(dataset, args.instance_id, args.limit)
    if not instances:
        parser.error("No instances selected")
    logger.info(f"Selected {len(instances)} instances with {min(args.max_workers, len(instances))} workers")

    success_count = 0
    with ProcessPoolExecutor(max_workers=min(args.max_workers, len(instances))) as executor:
        futures = {executor.submit(run_instance, instance, args): instance["instance_id"] for instance in instances}
        for future in as_completed(futures):
            instance_id = futures[future]
            try:
                if future.result():
                    success_count += 1
            except Exception as error:
                logger.error(f"Worker crashed for {instance_id}: {error}")
            logger.info(f"Progress: {success_count}/{len(instances)} successful")
    return 0 if success_count == len(instances) else 1


if __name__ == "__main__":
    raise SystemExit(main())

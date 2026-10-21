import fcntl
import hashlib
import os
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
from rank_bm25 import BM25Okapi

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAGENT_ROOT = PROJECT_ROOT / "compare_code" / "LocAgent"
if str(LOCAGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(LOCAGENT_ROOT))

from tree_sitter_language_pack import get_parser

EXCLUDED_DIRECTORIES = {
    ".git",
    ".eggs",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "__pypackages__",
    "build",
    "dist",
    "env",
    "external",
    "node_modules",
    "site-packages",
    "third_party",
    "vendor",
    "venv",
}
TEXT_EXTENSIONS = {
    ".c",
    ".cc",
    ".cfg",
    ".conf",
    ".cpp",
    ".cs",
    ".css",
    ".go",
    ".h",
    ".hpp",
    ".html",
    ".ini",
    ".java",
    ".jinja",
    ".jinja2",
    ".js",
    ".json",
    ".jsx",
    ".kt",
    ".md",
    ".php",
    ".plist",
    ".py",
    ".rb",
    ".rs",
    ".sh",
    ".sql",
    ".swift",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".vue",
    ".xml",
    ".yaml",
    ".yml",
}
SOURCE_EXTENSIONS = {
    ".c", ".cc", ".cpp", ".cs", ".go", ".h", ".hpp", ".java", ".js",
    ".jsx", ".kt", ".php", ".py", ".rb", ".rs", ".sh", ".swift", ".ts",
    ".tsx", ".vue",
}
REPOSITORY_NODE = "__repo__"
RELATION_WEIGHTS = {
    "contains": 0.4,
    "imports": 0.8,
    "calls": 1.0,
    "extends": 0.7,
}
REVERSE_FACTORS = {
    "contains": 1.0,
    "imports": 0.5,
    "calls": 0.7,
    "extends": 0.7,
}
SEARCHABLE_TYPES = {"file", "textfile", "class", "function", "attribute"}
FILE_TYPES = {"file", "textfile"}
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "when",
    "with",
}
DIRECT_EVIDENCE_WEIGHT = 2.0
PAGERANK_ALPHA = 0.85
SYMBOL_AGGREGATION_COUNT = 3
GRAPH_CACHE_VERSION = 8
TREE_SITTER_PARSERS = {}
TREE_SITTER_FAILURES = {}
TREE_SITTER_LANGUAGES = {
    ".c": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cs": "c_sharp",
    ".go": "go",
    ".h": "c",
    ".hpp": "cpp",
    ".java": "java",
    ".js": "javascript",
    ".jsx": "javascript",
    ".kt": "kotlin",
    ".php": "php",
    ".py": "python",
    ".rb": "ruby",
    ".rs": "rust",
    ".sh": "bash",
    ".swift": "swift",
    ".ts": "typescript",
    ".tsx": "tsx",
}
TREE_SITTER_DEFINITIONS = {
    "python": {"classes": {"class_definition"}, "functions": {"function_definition"}},
    "javascript": {"classes": {"class_declaration"}, "functions": {"function_declaration", "method_definition"}},
    "typescript": {"classes": {"class_declaration"}, "functions": {"function_declaration", "method_definition"}},
    "tsx": {"classes": {"class_declaration"}, "functions": {"function_declaration", "method_definition"}},
    "java": {"classes": {"class_declaration", "interface_declaration", "enum_declaration"}, "functions": {"method_declaration", "constructor_declaration"}},
    "kotlin": {"classes": {"class_declaration", "object_declaration"}, "functions": {"function_declaration"}},
    "go": {"classes": {"type_declaration"}, "functions": {"function_declaration", "method_declaration"}},
    "c": {"classes": {"struct_specifier", "enum_specifier"}, "functions": {"function_definition"}},
    "cpp": {"classes": {"class_specifier", "struct_specifier", "enum_specifier"}, "functions": {"function_definition"}},
    "c_sharp": {"classes": {"class_declaration", "interface_declaration", "struct_declaration"}, "functions": {"method_declaration", "constructor_declaration"}},
    "php": {"classes": {"class_declaration", "interface_declaration", "trait_declaration"}, "functions": {"function_definition", "method_declaration"}},
    "ruby": {"classes": {"class", "module"}, "functions": {"method", "singleton_method"}},
    "rust": {"classes": {"struct_item", "enum_item", "trait_item"}, "functions": {"function_item"}},
    "swift": {"classes": {"class_declaration", "struct_declaration", "protocol_declaration"}, "functions": {"function_declaration", "init_declaration"}},
    "bash": {"classes": set(), "functions": {"function_definition"}},
}
ROLE_PATTERNS = {
    "LLM_PROMPT": {
        "description": "prompts, instructions, messages, and chat templates",
        "patterns": {
            "prompt_identifier": r"\b(?:system_prompt|user_prompt|prompt_template|prompt|instruction|messages?)\b",
            "message_role": r"[\"']role[\"']\s*:\s*[\"'](?:system|user|assistant)[\"']",
        },
    },
    "LLM_CALL": {
        "description": "model inference and completion calls",
        "patterns": {
            "completion_call": r"\b(?:chat\.)?completions?\.create\b|\b(?:completion|acompletion)\s*\(",
            "model_inference": r"\b(?:generate_content|generate|invoke|ainvoke|predict)\s*\(",
        },
    },
    "LLM_CONFIG": {
        "description": "model selection, endpoints, credentials, and generation parameters",
        "patterns": {
            "model_setting": r"\b(?:model_name|model_id|model|temperature|top_p|max_tokens|max_output_tokens)\b",
            "endpoint_setting": r"\b(?:api_key|base_url|api_base|endpoint|deployment_name)\b",
        },
    },
    "LLM_TOOL": {
        "description": "tool definitions, tool selection, and function calling",
        "patterns": {
            "tool_identifier": r"\b(?:tool_choice|tool_calls?|function_call|available_tools|tools)\b",
            "tool_decorator": r"@(?:tool|function_tool)\b",
        },
    },
    "LLM_MEMORY": {
        "description": "conversation history, checkpoints, and persistent model memory",
        "patterns": {
            "memory_identifier": r"\b(?:chat_history|conversation_history|message_history|memory|memories|scratchpad)\b",
            "checkpoint_identifier": r"\b(?:checkpoint|checkpointer|thread_id)\b",
        },
    },
}
ROLE_QUERY_PATTERNS = {
    "LLM_PROMPT": {
        "prompt_language": r"\b(?:prompt|instruction|message|template|system message|user message)\b",
        "answer_behavior": r"\b(?:answer|respond|response|contradict|truth|hallucinat|character|language model)\w*\b",
    },
    "LLM_CALL": {
        "inference_language": r"\b(?:completion|generate|generation|inference|invoke|model call|llm call)\w*\b",
        "call_failure": r"\b(?:response|stream|timeout|retry|rate limit)\w*\b",
    },
    "LLM_CONFIG": {
        "model_configuration": r"\b(?:model|temperature|token|endpoint|configuration|config|api key|base url|environment|\.env)\w*\b",
        "provider_setting": r"\b(?:provider|deployment|credential|setting|settings)\w*\b",
    },
    "LLM_TOOL": {
        "tool_language": r"\b(?:tool|function call|function calling|schema|plugin|extension)\w*\b",
        "tool_failure": r"\b(?:argument|parameter|keyword argument|kwargs?)\w*\b",
    },
    "LLM_MEMORY": {
        "memory_language": r"\b(?:memory|history|conversation|checkpoint|context|thread|session|previous message)\w*\b",
        "memory_failure": r"\b(?:forget|remember|retain|persistent)\w*\b",
    },
}


def is_excluded(relative_path):
    return any(part in EXCLUDED_DIRECTORIES for part in relative_path.parts)


def repository_text_files(repo_path):
    files = []
    for path in sorted(repo_path.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative_path = path.relative_to(repo_path)
        if is_excluded(relative_path) or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        files.append((path, relative_path.as_posix()))
    return files


def read_repository_text(path):
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        print(f"Invalid UTF-8 replaced in {path}: {error}", file=sys.stderr)
        return path.read_text(encoding="utf-8", errors="replace")


def symbol_name(node):
    return node.split(":", 1)[-1].rsplit(".", 1)[-1]


def annotate_relation_confidence(graph):
    resolutions = defaultdict(set)
    for source, target, data in graph.edges(data=True):
        relation = data.get("type")
        if relation in {"calls", "extends"}:
            resolutions[(source, relation, symbol_name(target))].add(target)

    for source, target, _, data in graph.edges(keys=True, data=True):
        relation = data.get("type")
        ambiguity = len(resolutions.get((source, relation, symbol_name(target)), {target}))
        data["provenance"] = data.get("provenance", "tree_sitter")
        data["ambiguity"] = max(data.get("ambiguity", 1), ambiguity)
        data["confidence"] = min(data.get("confidence", 1.0), 1.0 / ambiguity)
        if data["ambiguity"] == 1:
            data["resolution"] = "exact"
            data["evidence_level"] = "exact"
        elif data.get("receiver") or data["ambiguity"] == 2:
            data["resolution"] = "constrained"
            data["evidence_level"] = "constrained"
        else:
            data["resolution"] = "ambiguous"
            data["evidence_level"] = "ambiguous"


def repository_fingerprint(repo_path):
    files = repository_text_files(repo_path)
    digest = hashlib.sha256()
    for source_path, relative_path in files:
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def repository_signature(files):
    signature = []
    for source_path, relative_path in files:
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        signature.append(
            {
                "path": relative_path,
                "size": source_path.stat().st_size,
                "sha256": digest,
            }
        )
    return signature


def tree_sitter_parse(code, path):
    language_name = TREE_SITTER_LANGUAGES.get(path.suffix.lower())
    if not language_name:
        return None, None
    if language_name in TREE_SITTER_FAILURES:
        return None, language_name
    try:
        parser = TREE_SITTER_PARSERS.get(language_name)
        if parser is None:
            parser = get_parser(language_name)
            TREE_SITTER_PARSERS[language_name] = parser
        return parser.parse(code.encode("utf-8")), language_name
    except Exception as error:
        TREE_SITTER_FAILURES[language_name] = str(error)
        print(
            f"Tree-sitter parser unavailable for {language_name}; files use file_only parsing: {error}",
            file=sys.stderr,
        )
        return None, language_name


def role_annotations(code):
    annotations = []
    for role, definition in ROLE_PATTERNS.items():
        for pattern_name, pattern in definition["patterns"].items():
            matches = list(re.finditer(pattern, code, flags=re.IGNORECASE))
            if not matches:
                continue
            annotations.append(
                {
                    "role": role,
                    "pattern": pattern_name,
                    "matched_evidence": list(dict.fromkeys(match.group(0) for match in matches)),
                    "match_count": len(matches),
                    "provenance": "fixed_pattern_library",
                }
            )
    return annotations


def annotate_llm_roles(graph):
    for _, data in graph.nodes(data=True):
        if data.get("type") not in SEARCHABLE_TYPES:
            continue
        annotations = role_annotations(data.get("code", ""))
        data["llm_role_annotations"] = annotations
        data["llm_roles"] = list(dict.fromkeys(item["role"] for item in annotations))


def walk_tree(root):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.children))


def node_text(node, source):
    return source[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def nearest_ancestor(node, types):
    parent = node.parent
    while parent:
        if parent.type in types:
            return parent
        parent = parent.parent
    return None


def node_name(node, source):
    name = node.child_by_field_name("name")
    if name:
        return node_text(name, source)
    declarator = node.child_by_field_name("declarator")
    search_root = declarator or node
    identifiers = [
        child for child in walk_tree(search_root)
        if child.type in {"identifier", "field_identifier", "type_identifier", "property_identifier", "constant"}
    ]
    return node_text(identifiers[-1], source) if identifiers else f"anonymous_{node.start_point.row + 1}"


def declaration_identifier(node, source):
    left = node.child_by_field_name("left") or node.child_by_field_name("name") or node.child_by_field_name("declarator")
    if not left:
        return None
    identifiers = [
        child for child in walk_tree(left)
        if child.type in {"identifier", "field_identifier", "property_identifier", "constant"}
    ]
    return node_text(identifiers[-1], source) if identifiers else None


def extract_import_modules(root, language_name, source):
    modules = []
    import_types = {
        "import_statement", "import_from_statement", "import_declaration", "use_declaration",
        "using_directive", "require", "namespace_use_declaration",
    }
    for node in walk_tree(root):
        if node.type not in import_types:
            continue
        text = node_text(node, source)
        if language_name == "python":
            match = re.match(r"(?:from\s+([\w.]+)\s+)?import\s+([\w.]+)", text)
            if match:
                modules.append(match.group(1) or match.group(2))
        else:
            quoted = re.findall(r"[\"']([^\"']+)[\"']", text)
            modules.extend(quoted or re.findall(r"(?:import|use|using|require)\s+([\w.:/\\-]+)", text))
    return modules


def extract_call_reference(node, source):
    function = node.child_by_field_name("function") or node.child_by_field_name("name")
    if not function:
        return None, None
    identifiers = [
        node_text(child, source)
        for child in walk_tree(function)
        if child.type in {"identifier", "field_identifier", "property_identifier", "constant"}
    ]
    if not identifiers:
        return None, None
    return identifiers[-1], identifiers[-2] if len(identifiers) > 1 else None


def extract_parent_names(class_node, source):
    names = []
    for field in ("superclasses", "superclass", "interfaces", "base_clause", "superclass_clause", "delegation_specifiers"):
        parent = class_node.child_by_field_name(field)
        if not parent:
            continue
        for child in walk_tree(parent):
            if child.type in {"identifier", "type_identifier", "constant"}:
                value = node_text(child, source)
                if value not in names:
                    names.append(value)
    return names


def resolve_module_path(relative_path, module, file_paths):
    normalized = module.replace("::", "/").replace(".", "/").lstrip("/")
    parent = Path(relative_path).parent
    candidates = []
    for base in (Path("."), parent):
        path = (base / normalized).as_posix().lstrip("./")
        candidates.extend([path, f"{path}.py", f"{path}.js", f"{path}.ts", f"{path}.tsx", f"{path}/__init__.py", f"{path}/index.js", f"{path}/index.ts"])
    return next((candidate for candidate in candidates if candidate in file_paths), None)


def add_tree_sitter_structure(graph, relative_path, code, tree, language_name):
    source = code.encode("utf-8")
    definitions = TREE_SITTER_DEFINITIONS.get(language_name, {"classes": set(), "functions": set()})
    class_nodes = {}
    function_nodes = {}
    definition_types = definitions["classes"] | definitions["functions"]
    for syntax_node in walk_tree(tree.root_node):
        if syntax_node.type not in definition_types:
            continue
        name = node_name(syntax_node, source)
        parent_class = nearest_ancestor(syntax_node, definitions["classes"])
        if syntax_node.type in definitions["classes"]:
            identifier = f"{relative_path}:{name}"
            owner = relative_path
            node_type = "class"
            class_nodes[syntax_node.id] = identifier
        else:
            owner = class_nodes.get(parent_class.id, relative_path) if parent_class else relative_path
            identifier = f"{owner}.{name}" if parent_class else f"{relative_path}:{name}"
            node_type = "function"
            function_nodes[syntax_node.id] = identifier
        graph.add_node(identifier, type=node_type, name=name, file_path=relative_path, code=node_text(syntax_node, source), start_line=syntax_node.start_point.row + 1, end_line=syntax_node.end_point.row + 1, parser="tree_sitter", language=language_name)
        graph.add_edge(owner, identifier, type="contains")
    assignment_types = {"assignment", "annotated_assignment", "variable_declarator", "const_item", "static_item", "field_declaration"}
    for syntax_node in walk_tree(tree.root_node):
        if syntax_node.type not in assignment_types or nearest_ancestor(syntax_node, definitions["functions"]):
            continue
        name = declaration_identifier(syntax_node, source)
        if not name:
            continue
        parent_class = nearest_ancestor(syntax_node, definitions["classes"])
        owner = class_nodes.get(parent_class.id, relative_path) if parent_class else relative_path
        identifier = f"{owner}:@{name}"
        if identifier in graph:
            continue
        graph.add_node(identifier, type="attribute", name=name, file_path=relative_path, code=node_text(syntax_node, source), start_line=syntax_node.start_point.row + 1, end_line=syntax_node.end_point.row + 1, parser="tree_sitter", language=language_name)
        graph.add_edge(owner, identifier, type="contains")
    return class_nodes, function_nodes


def add_package(graph, package_path):
    if package_path in graph:
        return
    parent_path = Path(package_path).parent.as_posix()
    parent = REPOSITORY_NODE if parent_path == "." else parent_path
    if parent != REPOSITORY_NODE:
        add_package(graph, parent)
    graph.add_node(package_path, type="package", name=package_path)
    graph.add_edge(parent, package_path, type="contains")


def build_codekg(files, stage_path):
    graph = nx.MultiDiGraph()
    graph.add_node(REPOSITORY_NODE, type="repo", name="repository", parser="tree_sitter")
    file_paths = {relative_path for _, relative_path in files}
    parsed = {}
    class_index = defaultdict(list)
    function_index = defaultdict(list)
    for source_path, relative_path in files:
        code = read_repository_text(source_path)
        tree, language_name = tree_sitter_parse(code, source_path)
        node_type = "file" if source_path.suffix.lower() in SOURCE_EXTENSIONS else "textfile"
        graph.add_node(relative_path, type=node_type, name=source_path.name, file_path=relative_path, code=code, language=language_name or source_path.suffix.lower().lstrip("."), parse_status="parsed" if tree else "file_only", parser="tree_sitter" if tree else "text")
        parent_path = Path(relative_path).parent.as_posix()
        parent = REPOSITORY_NODE if parent_path == "." else parent_path
        if parent != REPOSITORY_NODE:
            add_package(graph, parent)
        graph.add_edge(parent, relative_path, type="contains", provenance="tree_sitter")
        if not tree:
            continue
        class_nodes, function_nodes = add_tree_sitter_structure(graph, relative_path, code, tree, language_name)
        for identifier in class_nodes.values():
            class_index[graph.nodes[identifier]["name"]].append(identifier)
        for identifier in function_nodes.values():
            function_index[graph.nodes[identifier]["name"]].append(identifier)
        parsed[relative_path] = (tree, language_name, code, class_nodes, function_nodes)
    for relative_path, (tree, language_name, code, class_nodes, function_nodes) in parsed.items():
        source = code.encode("utf-8")
        imported_files = []
        for module in extract_import_modules(tree.root_node, language_name, source):
            target = resolve_module_path(relative_path, module, file_paths)
            if target:
                imported_files.append(target)
                graph.add_edge(relative_path, target, type="imports", provenance="tree_sitter", confidence=1.0, resolution="exact")
        definitions = TREE_SITTER_DEFINITIONS[language_name]
        for syntax_node in walk_tree(tree.root_node):
            if syntax_node.type in definitions["classes"]:
                source_id = class_nodes.get(syntax_node.id)
                for parent_name in extract_parent_names(syntax_node, source):
                    targets = class_index.get(parent_name, [])
                    imported = [target for target in targets if graph.nodes[target].get("file_path") in imported_files]
                    candidates = imported or targets
                    for target_id in candidates:
                        graph.add_edge(source_id, target_id, type="extends", provenance="tree_sitter", confidence=1.0 / len(candidates), ambiguity=len(candidates), resolution="exact" if len(candidates) == 1 else "ambiguous")
            if syntax_node.type not in {"call", "call_expression", "invocation_expression", "method_invocation", "function_call", "command"}:
                continue
            called_name, receiver = extract_call_reference(syntax_node, source)
            if not called_name:
                continue
            owner_syntax = nearest_ancestor(syntax_node, definitions["functions"])
            source_id = function_nodes.get(owner_syntax.id) if owner_syntax else relative_path
            targets = function_index.get(called_name, [])
            same_file = [target for target in targets if graph.nodes[target].get("file_path") == relative_path]
            imported = [target for target in targets if graph.nodes[target].get("file_path") in imported_files]
            receiver_targets = [target for target in targets if receiver and receiver.lower() in target.lower()]
            candidates = same_file or receiver_targets or imported or targets
            for target_id in candidates:
                graph.add_edge(source_id, target_id, type="calls", provenance="tree_sitter", receiver=receiver, confidence=1.0 / len(candidates), ambiguity=len(candidates), resolution="exact" if len(candidates) == 1 else "ambiguous")
    annotate_relation_confidence(graph)
    annotate_llm_roles(graph)
    return graph


def build_complete_codekg(repo_path, stage_path, cache_path=None):
    files = repository_text_files(repo_path)
    if cache_path is None:
        graph = build_codekg(files, stage_path)
        return graph, build_entity_index(graph), False

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    signature = repository_signature(files)
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            if cache_path.is_file():
                try:
                    with cache_path.open("rb") as file:
                        cached = pickle.load(file)
                    if (
                        cached.get("version") == GRAPH_CACHE_VERSION
                        and cached.get("signature") == signature
                        and "entity_index" in cached
                    ):
                        return cached["graph"], cached["entity_index"], True
                except Exception as error:
                    print(f"Invalid CodeKG cache {cache_path}: {error}", file=sys.stderr)
            graph = build_codekg(files, stage_path)
            entity_index = build_entity_index(graph)
            temporary_path = cache_path.with_suffix(cache_path.suffix + f".{os.getpid()}.tmp")
            try:
                with temporary_path.open("wb") as file:
                    pickle.dump(
                        {
                            "version": GRAPH_CACHE_VERSION,
                            "signature": signature,
                            "graph": graph,
                            "entity_index": entity_index,
                        },
                        file,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                os.replace(temporary_path, cache_path)
            finally:
                temporary_path.unlink(missing_ok=True)
            return graph, entity_index, False
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def graph_to_dict(graph):
    return {
        "directed": graph.is_directed(),
        "multigraph": graph.is_multigraph(),
        "nodes": [{"id": node, **data} for node, data in graph.nodes(data=True)],
        "edges": [
            {"source": source, "target": target, "key": key, **data}
            for source, target, key, data in graph.edges(keys=True, data=True)
        ],
    }


def owning_file(node, data):
    if data.get("type") in FILE_TYPES:
        return node
    if data.get("type") in SEARCHABLE_TYPES - FILE_TYPES:
        return data.get("file_path") or (node.split(":", 1)[0] if ":" in node else None)
    return None


def add_symbol_edge(graph, source, target, evidence):
    current = graph.get_edge_data(source, target, {"weight": 0.0, "evidence": []})
    graph.add_edge(
        source,
        target,
        weight=current["weight"] + evidence["weight"],
        evidence=current["evidence"] + [evidence],
    )


def build_symbol_graph(code_graph, use_confidence=True, allow_ambiguous=True):
    graph = nx.DiGraph()
    graph.add_nodes_from(
        node for node, data in code_graph.nodes(data=True) if data.get("type") in SEARCHABLE_TYPES
    )
    relation_edges = []
    for source, target, data in code_graph.edges(data=True):
        relation = data.get("type")
        if relation not in RELATION_WEIGHTS or source not in graph or target not in graph:
            continue
        confidence = data.get("confidence", 1.0) if use_confidence else 1.0
        evidence_level = data.get("evidence_level", data.get("resolution", "exact"))
        if evidence_level == "ambiguous" and not allow_ambiguous:
            continue
        weight = RELATION_WEIGHTS[relation] * confidence
        forward = {
            "relation": relation,
            "direction": "forward",
            "confidence": confidence,
            "evidence_level": evidence_level,
            "weight": weight,
            "provenance": data.get("provenance", "tree_sitter"),
            "source": source,
            "target": target,
        }
        reverse = {
            **forward,
            "direction": "reverse",
            "weight": weight * REVERSE_FACTORS[relation],
            "source": target,
            "target": source,
        }
        add_symbol_edge(graph, source, target, forward)
        add_symbol_edge(graph, target, source, reverse)
        relation_edges.append(forward)
    return graph, relation_edges


def tokenize(text):
    expanded = []
    token = []
    for character in text:
        if character.isalnum() or character == "_":
            if token and character.isupper() and token[-1].islower():
                expanded.append("".join(token).lower())
                token = [character]
            else:
                token.append(character)
        elif token:
            expanded.append("".join(token).lower())
            token = []
    if token:
        expanded.append("".join(token).lower())
    return [item for item in expanded if len(item) > 1 and item not in STOPWORDS]


def build_entity_index(code_graph):
    entities = []
    documents = []
    for node, data in code_graph.nodes(data=True):
        if data.get("type") not in SEARCHABLE_TYPES:
            continue
        annotations = data.get("llm_role_annotations", [])
        role_text = "\n".join(
            " ".join(
                (
                    item["role"],
                    ROLE_PATTERNS[item["role"]]["description"],
                    item["pattern"],
                    *item["matched_evidence"],
                )
            )
            for item in annotations
        )
        text = "\n".join((node, data.get("type", ""), data.get("code", ""), role_text))
        entities.append((node, data))
        documents.append(tokenize(text))
    if not entities:
        raise ValueError("CodeKG contains no searchable entities")
    return {
        "entities": entities,
        "documents": documents,
        "bm25": BM25Okapi(documents),
    }


def rank_entities(code_graph, issue, entity_index=None):
    entity_index = entity_index or build_entity_index(code_graph)
    entities = entity_index["entities"]
    documents = entity_index["documents"]
    query = tokenize(issue)
    if not query:
        raise ValueError("Issue contains no searchable terms")
    bm25_scores = entity_index["bm25"].get_scores(query)
    query_terms = set(query)
    rankings = []
    for (node, data), document, bm25_score in zip(entities, documents, bm25_scores):
        overlap = len(query_terms.intersection(document)) / len(query_terms)
        score = max(float(bm25_score), 0.0) + overlap
        rankings.append(
            {
                "entity": node,
                "entity_type": data.get("type"),
                "file": owning_file(node, data),
                "bm25_score": float(bm25_score),
                "query_overlap": overlap,
                "lexical_score": score,
            }
        )
    rankings.sort(key=lambda item: (-item["lexical_score"], item["entity"]))
    return rankings


def exact_identifier_match(text, value):
    return bool(re.search(rf"(?<![A-Za-z0-9_]){re.escape(value)}(?![A-Za-z0-9_])", text))


def exact_path_match(text, value):
    return bool(re.search(rf"(?<![A-Za-z0-9_./-]){re.escape(value)}(?![A-Za-z0-9_./-])", text))


def explicit_symbol_match(text, value):
    escaped = re.escape(value)
    patterns = (
        rf"(?<![A-Za-z0-9_]){escaped}\s*\(",
        rf"\b(?:in|function|method|class)\s+[`'\"]?{escaped}(?![A-Za-z0-9_])",
        rf"[`'\"]{escaped}[`'\"]",
    )
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def extract_direct_evidence(code_graph, issue):
    normalized_issue = issue.replace("\\", "/")
    basename_nodes = defaultdict(list)
    qualified_symbol_nodes = defaultdict(list)
    short_symbol_nodes = defaultdict(list)
    for node, data in code_graph.nodes(data=True):
        if data.get("type") in FILE_TYPES:
            basename_nodes[Path(node).name].append(node)
        elif data.get("type") in {"class", "function", "attribute"}:
            qualified = node.split(":", 1)[-1]
            qualified_symbol_nodes[qualified].append(node)
            short_symbol_nodes[symbol_name(node)].append(node)

    evidence = []
    for node, data in code_graph.nodes(data=True):
        node_type = data.get("type")
        matches = []
        if node_type in FILE_TYPES:
            if exact_path_match(normalized_issue, node):
                matches.append(("repository_path", node))
            basename = Path(node).name
            if len(basename_nodes[basename]) == 1 and exact_identifier_match(normalized_issue, basename):
                matches.append(("unique_filename", basename))
        elif node_type in {"class", "function", "attribute"}:
            qualified = node.split(":", 1)[-1]
            short = symbol_name(node)
            if len(qualified_symbol_nodes[qualified]) == 1 and explicit_symbol_match(issue, qualified):
                matches.append(("qualified_symbol", qualified))
            elif len(short_symbol_nodes[short]) == 1 and explicit_symbol_match(issue, short):
                matches.append(("unique_symbol", short))
        for evidence_type, matched_text in matches:
            evidence.append(
                {
                    "entity": node,
                    "entity_type": node_type,
                    "file": owning_file(node, data),
                    "evidence_type": evidence_type,
                    "matched_text": matched_text,
                    "weight": DIRECT_EVIDENCE_WEIGHT,
                }
            )
    evidence.sort(key=lambda item: (item["entity"], item["evidence_type"]))
    return evidence


def normalize_scores(scores):
    maximum = max(scores.values(), default=0.0)
    if maximum <= 0:
        return {key: 0.0 for key in scores}
    return {key: value / maximum for key, value in scores.items()}


def extract_role_evidence(code_graph, issue):
    matched_roles = {}
    for role, patterns in ROLE_QUERY_PATTERNS.items():
        matches = []
        for pattern_name, pattern in patterns.items():
            matched_text = list(
                dict.fromkeys(match.group(0) for match in re.finditer(pattern, issue, flags=re.IGNORECASE))
            )
            if matched_text:
                matches.append({"pattern": pattern_name, "matched_text": matched_text})
        if matches:
            matched_roles[role] = matches
    evidence = []
    for node, data in code_graph.nodes(data=True):
        for annotation in data.get("llm_role_annotations", []):
            role = annotation["role"]
            if role not in matched_roles:
                continue
            evidence.append(
                {
                    "entity": node,
                    "entity_type": data.get("type"),
                    "file": owning_file(node, data),
                    "role": role,
                    "issue_evidence": matched_roles[role],
                    "pattern": annotation["pattern"],
                    "matched_evidence": annotation["matched_evidence"],
                    "score": float(sum(len(item["matched_text"]) for item in matched_roles[role])),
                    "provenance": "fixed_pattern_library",
                }
            )
    evidence.sort(key=lambda item: (item["entity"], item["role"], item["pattern"]))
    return evidence


def build_personalization(
    entity_rankings,
    direct_evidence,
    role_evidence,
    seed_count,
    use_direct=True,
    use_inference=True,
    use_roles=True,
):
    lexical = normalize_scores({item["entity"]: item["lexical_score"] for item in entity_rankings})
    direct_raw = defaultdict(float)
    for item in direct_evidence:
        direct_raw[item["entity"]] = max(direct_raw[item["entity"]], item["weight"])
    role_raw = defaultdict(float)
    for item in role_evidence:
        role_raw[item["entity"]] += item["score"]
    direct = normalize_scores(direct_raw)
    roles = normalize_scores(role_raw)

    seed_entities = set()
    if use_inference:
        for item in entity_rankings:
            if item["lexical_score"] <= 0:
                continue
            seed_entities.add(item["entity"])
            if len(seed_entities) == seed_count:
                break
    if use_direct:
        seed_entities.update(direct)
    if use_roles:
        ranked_roles = sorted(roles.items(), key=lambda item: (-item[1], item[0]))
        seed_entities.update(entity for entity, _ in ranked_roles[:seed_count])
    personalization = {
        entity: (
            (2.0 * direct.get(entity, 0.0) if use_direct else 0.0)
            + (lexical.get(entity, 0.0) if use_inference else 0.0)
            + (roles.get(entity, 0.0) if use_roles else 0.0)
        )
        for entity in seed_entities
    }
    personalization = {entity: score for entity, score in personalization.items() if score > 0}
    if not personalization:
        return {}, []
    file_by_entity = {item["entity"]: item["file"] for item in entity_rankings}
    seeds = [
        {
            "entity": entity,
            "file": file_by_entity[entity],
            "direct_evidence_score": direct.get(entity, 0.0) if use_direct else 0.0,
            "inference_score": lexical.get(entity, 0.0) if use_inference else 0.0,
            "role_evidence_score": roles.get(entity, 0.0) if use_roles else 0.0,
            "personalization_score": score,
        }
        for entity, score in personalization.items()
    ]
    seeds.sort(key=lambda item: (-item["personalization_score"], item["entity"]))
    return personalization, seeds


def pagerank(graph, personalization):
    if not personalization:
        return {node: 0.0 for node in graph.nodes}
    if graph.number_of_edges() > 0:
        return nx.pagerank(
            graph,
            alpha=PAGERANK_ALPHA,
            personalization=personalization,
            weight="weight",
        )
    total = sum(personalization.values())
    return {node: personalization.get(node, 0.0) / total for node in graph.nodes}


def aggregate_symbol_scores(code_graph, symbol_scores):
    scores_by_file = defaultdict(list)
    for node, score in symbol_scores.items():
        file_path = owning_file(node, code_graph.nodes[node])
        if file_path:
            scores_by_file[file_path].append(score)
    aggregated = {}
    for file_path, scores in scores_by_file.items():
        top_scores = sorted(scores, reverse=True)[:SYMBOL_AGGREGATION_COUNT]
        aggregated[file_path] = 0.5 * top_scores[0] + 0.5 * sum(top_scores) / len(top_scores)
    return aggregated


def combine_file_scores(
    file_nodes,
    lexical_scores,
    graph_scores,
    lexical_weight,
    analyzer_rankings=None,
    code_graph=None,
    issue="",
):
    normalized_lexical = normalize_scores(lexical_scores)
    normalized_graph = normalize_scores(graph_scores)
    analyzer_rankings = analyzer_rankings or {}
    ranked_sources = {
        "I": analyzer_rankings.get("I", []),
        "R": analyzer_rankings.get("R", []),
        "BM25": sorted(file_nodes, key=lambda path: (-normalized_lexical.get(path, 0.0), path)),
        "CodeKG": sorted(file_nodes, key=lambda path: (-normalized_graph.get(path, 0.0), path)),
    }
    weights = {"I": 2.0, "R": 1.0, "BM25": 2.0, "CodeKG": 2.0}
    ranks = {
        source: {path: rank for rank, path in enumerate(paths, 1)}
        for source, paths in ranked_sources.items()
    }
    direct_files = set(analyzer_rankings.get("E", []))
    inference_evidence = analyzer_rankings.get("I_evidence", {})
    issue_lower = issue.lower()
    text_artifact_relevant = bool(
        re.search(
            r"\b(?:documentation|readme|prompt|template|config(?:uration)?|yaml|json|toml|environment variable)\b",
            issue_lower,
        )
    )
    rankings = []
    for file_path in sorted(file_nodes):
        lexical_score = normalized_lexical.get(file_path, 0.0)
        graph_score = normalized_graph.get(file_path, 0.0)
        source_ranks = {
            source: source_ranks[file_path]
            for source, source_ranks in ranks.items()
            if file_path in source_ranks
            and (source not in {"BM25", "CodeKG"} or (lexical_score if source == "BM25" else graph_score) > 0)
        }
        retrieval_sources = list(source_ranks)
        if file_path in direct_files:
            retrieval_sources.insert(0, "E")
        rrf_score = sum(weights[source] / (60 + rank) for source, rank in source_ranks.items())
        if file_path in direct_files:
            rrf_score += 4.0 / 61
        evidence = inference_evidence.get(file_path, {})
        inference_quality = (
            1.0
            if evidence.get("exact_symbol_match")
            else 0.75
            if evidence.get("issue_supported_behavior")
            else 0.5
            if evidence.get("matched_term_count", 0) >= 2
            else 0.2
        )
        if "I" in source_ranks:
            rrf_score += inference_quality * 2.0 / (60 + source_ranks["I"])
        suffix = Path(file_path).suffix.lower()
        artifact_weight = (
            0.25
            if suffix in {".md", ".rst", ".txt"}
            and file_path not in direct_files
            and not text_artifact_relevant
            else 1.0
        )
        duplicate_weight = 1.0
        fused_score = rrf_score * artifact_weight
        rankings.append(
            {
                "path": file_path,
                "lexical_score": lexical_score,
                "graph_score": graph_score,
                "hybrid_score": fused_score,
                "rrf_score": rrf_score,
                "artifact_weight": artifact_weight,
                "duplicate_weight": duplicate_weight,
                "retrieval_sources": retrieval_sources,
                "source_ranks": source_ranks,
                "inference_quality": inference_quality,
            }
        )
    rankings.sort(
        key=lambda item: (
            item["path"] not in direct_files,
            -item["hybrid_score"],
            item["path"],
        )
    )
    return rankings


def score_entities(entity_rankings, symbol_scores, use_direct, use_inference, use_roles):
    lexical = normalize_scores(
        {item["entity"]: item["lexical_score"] for item in entity_rankings}
    )
    graph = normalize_scores(symbol_scores)
    direct = normalize_scores(
        {item["entity"]: item.get("direct_evidence_score", 0.0) for item in entity_rankings}
    )
    roles = normalize_scores(
        {item["entity"]: item.get("role_evidence_score", 0.0) for item in entity_rankings}
    )
    scored = []
    for item in entity_rankings:
        components = {
            "lexical": lexical.get(item["entity"], 0.0) if use_inference else 0.0,
            "graph": graph.get(item["entity"], 0.0),
            "direct": direct.get(item["entity"], 0.0) if use_direct else 0.0,
            "role": roles.get(item["entity"], 0.0) if use_roles else 0.0,
        }
        score = (
            0.45 * components["lexical"]
            + 0.25 * components["graph"]
            + 0.20 * components["direct"]
            + 0.10 * components["role"]
        )
        scored.append({**item, "entity_score": score, "entity_score_components": components})
    return scored


def remove_redundant_file_entities(code_graph, entities):
    structured_files = {
        owning_file(node, data)
        for node, data in code_graph.nodes(data=True)
        if data.get("type") in {"class", "function", "attribute"}
    }
    return [
        item
        for item in entities
        if item["entity_type"] not in FILE_TYPES or item["file"] not in structured_files
    ]


def select_global_entities(
    code_graph,
    entity_rankings,
    symbol_scores,
    file_rankings,
    candidate_count,
    entity_budget,
    use_direct,
    use_inference,
    use_roles,
):
    file_rank_by_path = {
        item["path"]: rank for rank, item in enumerate(file_rankings[:candidate_count], 1)
    }
    eligible = [item for item in entity_rankings if item["file"] in file_rank_by_path]
    eligible = remove_redundant_file_entities(code_graph, eligible)
    eligible = score_entities(eligible, symbol_scores, use_direct, use_inference, use_roles)
    eligible.sort(
        key=lambda item: (
            -item["entity_score"],
            file_rank_by_path[item["file"]],
            item["entity"],
        )
    )

    selected = []
    selected_names = set()
    for path in file_rank_by_path:
        best = next((item for item in eligible if item["file"] == path), None)
        if best is None:
            continue
        selected.append(best)
        selected_names.add(best["entity"])
        if len(selected) == entity_budget:
            return selected
    for item in eligible:
        if item["entity"] in selected_names:
            continue
        selected.append(item)
        selected_names.add(item["entity"])
        if len(selected) == entity_budget:
            break
    selected.sort(
        key=lambda item: (
            file_rank_by_path[item["file"]],
            -item["entity_score"],
            item["entity"],
        )
    )
    return selected


def select_code_snippets(code, start_line, query_terms, role_annotations):
    lines = code.splitlines()
    if not lines:
        return []
    role_terms = set()
    for annotation in role_annotations:
        for evidence in annotation.get("matched_evidence", []):
            role_terms.update(tokenize(evidence))
    centers = []
    for index, line in enumerate(lines):
        line_terms = set(tokenize(line))
        query_overlap = len(query_terms.intersection(line_terms))
        role_overlap = len(role_terms.intersection(line_terms))
        stripped = line.lstrip()
        signature = index == 0 or stripped.startswith(("def ", "async def ", "class "))
        operation = bool(
            re.search(
                r"\b(?:return|raise|await|yield|if|elif|for|while)\b|"
                r"\b[A-Za-z_][A-Za-z0-9_.]*\s*\(|"
                r"\b[A-Za-z_][A-Za-z0-9_]*\s*=",
                stripped,
            )
        )
        if not (signature or query_overlap or role_overlap or operation):
            continue
        selection = (
            "query_match"
            if query_overlap
            else "role_match"
            if role_overlap
            else "signature"
            if signature
            else "operation"
        )
        score = 4.0 * query_overlap + 2.0 * role_overlap + 1.0 * signature + 0.25 * operation
        centers.append((index, selection, score))
    snippets = []
    for center, selection, center_score in centers:
        for index in range(max(0, center - 1), min(len(lines), center + 2)):
            if not lines[index].strip():
                continue
            snippets.append(
                {
                    "start_line": start_line + index,
                    "end_line": start_line + index,
                    "code": lines[index],
                    "selection": selection if index == center else "context",
                    "local_score": center_score if index == center else center_score * 0.5,
                }
            )
    return snippets


def hypothesis_terms(hypotheses):
    terms = set()
    symbols = set()
    hypothesis_evidence = []
    for hypothesis in hypotheses or []:
        operation_terms = set(tokenize(str(hypothesis.get("faulty_operation", ""))))
        contextual_terms = set(
            tokenize(
                " ".join(
                    str(hypothesis.get(field, ""))
                    for field in ("wrong_value", "expected_constraint")
                )
            )
        )
        hypothesis_terms = operation_terms | contextual_terms
        terms.update(hypothesis_terms)
        symbol = str(hypothesis.get("involved_symbol", "")).strip().lower()
        symbol_values = set()
        if symbol:
            symbol_values.update((symbol, symbol.rsplit(".", 1)[-1]))
            symbols.update(symbol_values)
        hypothesis_evidence.append(
            {
                "terms": hypothesis_terms,
                "operation_terms": operation_terms,
                "symbols": symbol_values,
            }
        )
    return terms, symbols, hypothesis_evidence


def statement_has_symbol(statement_code, symbol):
    symbol = symbol.strip().lower()
    if not symbol:
        return False
    candidates = [symbol]
    terminal = symbol.rsplit(".", 1)[-1]
    if terminal != symbol and not terminal.startswith("__"):
        candidates.append(terminal)
    lowered = statement_code.lower()
    return any(
        re.search(rf"(?<![a-z0-9_]){re.escape(candidate)}(?![a-z0-9_])", lowered)
        for candidate in candidates
    )


def statement_ranges(code, path):
    tree, _ = tree_sitter_parse(code, Path(path))
    if tree is None:
        return []
    ranges = []
    for node in walk_tree(tree.root_node):
        node_type = node.type
        is_statement = (
            node_type.endswith(("_statement", "_declaration"))
            or node_type in {
                "assignment",
                "augmented_assignment",
                "call",
                "expression_statement",
                "lexical_declaration",
                "variable_declaration",
            }
        )
        if not node.is_named or not is_statement:
            continue
        start = node.start_point.row + 1
        end = node.end_point.row + 1
        ranges.append((start, end, node_type))
    return ranges


def build_line_rankings(code_graph, candidate_context, query_terms, hypotheses=None):
    locations = []
    grounded_terms, grounded_symbols, hypothesis_evidence = hypothesis_terms(hypotheses)
    for file_rank, candidate in enumerate(candidate_context, 1):
        path = candidate["path"]
        code_text = code_graph.nodes[path].get("code", "")
        lines = code_text.splitlines()
        statements = statement_ranges(code_text, path)
        definitions = [
            (
                data.get("start_line", 1),
                data.get("end_line", data.get("start_line", 1)),
                data.get("type"),
                node,
            )
            for node, data in code_graph.nodes(data=True)
            if data.get("file_path") == path and data.get("type") in {"function", "class"}
        ]
        entity_scores = {
            entity["name"]: entity["entity_score"]
            for entity in candidate["relevant_entities"]
        }
        block_by_line = {}
        for block in definitions:
            for line_number in range(block[0], block[1] + 1):
                current = block_by_line.get(line_number)
                if current is None or block[1] - block[0] < current[1] - current[0]:
                    block_by_line[line_number] = block
        statement_by_line = {}
        for statement in statements:
            for line_number in range(statement[0], statement[1] + 1):
                current = statement_by_line.get(line_number)
                if current is None or (statement[1] - statement[0], statement[0]) < (
                    current[1] - current[0],
                    current[0],
                ):
                    statement_by_line[line_number] = statement
        import_parenthesis_depth = 0
        statement_cache = {}
        for line_number, code in enumerate(lines, 1):
            if not code.strip():
                continue
            block = block_by_line.get(line_number)
            block_name = block[3] if block else f"{path}:module"
            block_kind = block[2] if block else "module"
            statement = statement_by_line.get(
                line_number,
                (line_number, line_number, "line"),
            )
            statement_start, statement_end, statement_kind = statement
            statement_key = (statement_start, statement_end, statement_kind)
            statement_data = statement_cache.get(statement_key)
            if statement_data is None:
                statement_code = "\n".join(lines[statement_start - 1:statement_end])
                statement_terms = set(tokenize(statement_code))
                hypothesis_matches = [
                    {
                        "coverage": len(evidence["terms"].intersection(statement_terms)) / len(evidence["terms"])
                        if evidence["terms"]
                        else 0.0,
                        "symbol_match": any(
                            statement_has_symbol(statement_code, symbol)
                            for symbol in evidence["symbols"]
                        ),
                        "operation_overlap": len(evidence["operation_terms"].intersection(statement_terms)),
                    }
                    for evidence in hypothesis_evidence
                ]
                statement_data = {
                    "code": statement_code,
                    "terms": statement_terms,
                    "hypothesis_matches": hypothesis_matches,
                    "hypothesis_coverages": [match["coverage"] for match in hypothesis_matches],
                    "symbol_match": any(
                        statement_has_symbol(statement_code, symbol)
                        for symbol in grounded_symbols
                    ),
                }
                statement_cache[statement_key] = statement_data
            statement_code = statement_data["code"]
            statement_terms = statement_data["terms"]
            hypothesis_matches = statement_data["hypothesis_matches"]
            hypothesis_coverages = statement_data["hypothesis_coverages"]
            symbol_match = statement_data["symbol_match"]
            overlap_count = len(query_terms.intersection(statement_terms))
            hypothesis_overlap = len(grounded_terms.intersection(statement_terms))
            hypothesis_coverage = max(hypothesis_coverages, default=0.0)
            stripped = code.lstrip()
            signature = stripped.startswith(("def ", "async def ", "class "))
            import_continuation = import_parenthesis_depth > 0
            starts_import = bool(
                re.match(r"(?:from\s+\S+\s+import|import\s+)", stripped)
            )
            anchor_type = (
                "import"
                if starts_import or import_continuation
                else "condition"
                if re.search(r"\b(?:if|elif|while|for)\b", stripped)
                else "return_or_raise"
                if re.search(r"\b(?:return|raise|yield)\b", stripped)
                else "assignment"
                if re.search(r"(?<![=!<>])=(?!=)", stripped)
                else "call"
                if re.search(r"\b[A-Za-z_][A-Za-z0-9_.]*\s*\(", stripped)
                else "semantic_comment"
                if stripped.startswith("#") and (hypothesis_overlap or overlap_count)
                else "signature"
                if signature
                else "context"
            )
            operation = anchor_type not in {"signature", "context"}
            import_parenthesis_depth += code.count("(") - code.count(")") if starts_import or import_continuation else 0
            import_parenthesis_depth = max(import_parenthesis_depth, 0)
            local_score = (
                6.0 * symbol_match
                + 4.0 * hypothesis_coverage
                + 1.0 * hypothesis_overlap
                + 1.0 * overlap_count
                + 1.0 * operation
                + 0.25 * signature
            )
            entity_score = entity_scores.get(block_name, 0.0)
            retrieval_sources = list(candidate.get("retrieval_sources", []))
            locations.append(
                {
                    "path": path,
                    "line": line_number,
                    "code": code,
                    "block": block_name,
                    "block_kind": block_kind,
                    "statement_start": statement_start,
                    "statement_end": statement_end,
                    "statement_kind": statement_kind,
                    "statement_code": statement_code,
                    "file_rank": file_rank,
                    "file_hybrid_score": candidate["hybrid_score"],
                    "entity_score": entity_score,
                    "line_local_score": local_score,
                    "line_query_overlap": overlap_count / len(query_terms) if query_terms else 0.0,
                    "hypothesis_overlap": hypothesis_overlap,
                    "hypothesis_coverage": hypothesis_coverage,
                    "hypothesis_coverages": hypothesis_coverages,
                    "hypothesis_matches": hypothesis_matches,
                    "hypothesis_symbol_match": symbol_match,
                    "anchor_type": anchor_type,
                    "retrieval_sources": retrieval_sources,
                    "sources": [
                        {
                            "block": block_name,
                            "block_kind": block_kind,
                            "selection": "hypothesis_match" if hypothesis_overlap or symbol_match else "query_match" if overlap_count else "signature" if signature else "operation" if operation else "block_context",
                            "anchor_type": anchor_type,
                            "local_score": local_score,
                        }
                    ],
                }
            )
    locations.sort(
        key=lambda item: (
            -(0.35 * item["file_hybrid_score"] + 0.20 * item["entity_score"] + 0.45 * min(item["line_local_score"], 20.0) / 20.0),
            -item["hypothesis_symbol_match"],
            -item["hypothesis_overlap"],
            -item["line_query_overlap"],
            item["file_rank"],
            item["path"],
            item["line"],
        )
    )
    for rank, item in enumerate(locations, 1):
        item["id"] = f"L{rank}"
        item["raw_rank"] = rank
        item["raw_ranking_rule"] = (
            "descending_weighted_file_block_line_score, "
            "descending_line_query_overlap, file_rank, path, line"
        )
    return locations


def relation_paths(symbol_graph, seeds, candidate_entities, max_paths=2, cutoff=3):
    paths = []
    for seed in seeds[:10]:
        source = seed["entity"]
        reachable = nx.single_source_shortest_path(symbol_graph, source, cutoff=cutoff)
        for target in candidate_entities:
            nodes = reachable.get(target)
            if not nodes or len(nodes) < 2:
                continue
            edges = []
            for left, right in zip(nodes, nodes[1:]):
                choices = symbol_graph[left][right]["evidence"]
                edges.append(max(choices, key=lambda item: item["weight"]))
            paths.append(
                {
                    "nodes": nodes,
                    "edges": edges,
                    "hop_count": len(edges),
                    "minimum_confidence": min(edge["confidence"] for edge in edges),
                }
            )
    paths.sort(
        key=lambda item: (
            item["hop_count"],
            -item["minimum_confidence"],
            item["nodes"],
        )
    )
    unique = []
    seen = set()
    for path in paths:
        key = tuple(path["nodes"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
        if len(unique) == max_paths:
            break
    return unique


def build_candidate_context(
    code_graph,
    issue,
    candidates,
    selected_entities,
    symbol_scores,
    symbol_graph,
    seeds,
    direct_evidence,
    role_evidence,
    use_inference,
    use_roles,
):
    query_terms = set(tokenize(issue)) if use_inference else set()
    direct_by_file = defaultdict(list)
    for item in direct_evidence:
        direct_by_file[item["file"]].append(item)
    roles_by_file = defaultdict(list)
    for item in role_evidence:
        roles_by_file[item["file"]].append(item)

    entities_by_file = defaultdict(list)
    for entity in selected_entities:
        entities_by_file[entity["file"]].append(entity)

    contexts = []
    for candidate in candidates:
        path = candidate["path"]
        file_data = code_graph.nodes[path]
        relevant = entities_by_file[path]
        relevant_entities = []
        for entity in relevant:
            data = code_graph.nodes[entity["entity"]]
            code = data.get("code", "")
            signature = ""
            if entity["entity_type"] in {"class", "function"} and code:
                signature = code.splitlines()[0]
            relevant_entities.append(
                {
                    "name": entity["entity"],
                    "type": entity["entity_type"],
                    "signature": signature,
                    "signature_source": "definition_line" if signature else None,
                    "start_line": data.get("start_line", 1),
                    "end_line": data.get("end_line"),
                    "lexical_score": entity["lexical_score"],
                    "graph_score": symbol_scores.get(entity["entity"], 0.0),
                    "entity_score": entity["entity_score"],
                    "entity_score_components": entity["entity_score_components"],
                    "role_annotations": data.get("llm_role_annotations", []) if use_roles else [],
                    "snippets": select_code_snippets(
                        code,
                        data.get("start_line", 1),
                        query_terms,
                        data.get("llm_role_annotations", []) if use_roles else [],
                    ),
                }
            )
        target_entities = [item["entity"] for item in relevant]
        contexts.append(
            {
                **candidate,
                "parse_status": file_data.get("parse_status", "parsed"),
                "direct_evidence": direct_by_file[path],
                "role_evidence": roles_by_file[path],
                "llm_roles": file_data.get("llm_roles", []) if use_roles else [],
                "relevant_entities": relevant_entities,
                "relation_paths": relation_paths(symbol_graph, seeds, target_entities),
            }
        )
    return contexts


def retrieve_files(
    code_graph,
    issue,
    seed_count=30,
    candidate_count=20,
    lexical_weight=0.65,
    entity_budget=40,
    entity_index=None,
    use_direct=True,
    use_inference=True,
    use_roles=True,
    protected_file_paths=None,
    analyzer_rankings=None,
    fault_hypotheses=None,
    use_bm25=True,
    use_pagerank=True,
):
    if use_bm25:
        entity_rankings = rank_entities(code_graph, issue, entity_index)
    else:
        entity_rankings = [
            {"entity": node, "entity_type": data["type"], "file": owning_file(node, data),
             "bm25_score": 0.0, "query_overlap": 0.0, "lexical_score": 0.0}
            for node, data in sorted(code_graph.nodes(data=True))
            if data.get("type") in SEARCHABLE_TYPES
        ]
    direct_evidence = extract_direct_evidence(code_graph, issue)
    role_evidence = extract_role_evidence(code_graph, issue)
    personalization, seeds = build_personalization(
        entity_rankings,
        direct_evidence,
        role_evidence,
        seed_count,
        use_direct=use_direct,
        use_inference=use_inference and use_bm25,
        use_roles=use_roles,
    )

    lexical_scores = defaultdict(float)
    if use_inference:
        for item in entity_rankings:
            if item["file"]:
                lexical_scores[item["file"]] = max(
                    lexical_scores[item["file"]], item["lexical_score"]
                )
    file_nodes = {
        node for node, data in code_graph.nodes(data=True) if data.get("type") in FILE_TYPES
    }
    expansion_graph, _ = build_symbol_graph(code_graph, use_confidence=True, allow_ambiguous=False)
    symbol_graph, relation_edges = build_symbol_graph(code_graph, use_confidence=True, allow_ambiguous=True)
    expansion_scores = pagerank(expansion_graph, personalization) if use_pagerank else {}
    symbol_scores = pagerank(symbol_graph, personalization) if use_pagerank else {}
    symbol_file_scores = aggregate_symbol_scores(code_graph, expansion_scores)
    effective_lexical_weight = lexical_weight if use_inference else 0.0
    file_rankings = combine_file_scores(
        file_nodes,
        lexical_scores,
        symbol_file_scores,
        effective_lexical_weight,
        analyzer_rankings=analyzer_rankings,
        code_graph=code_graph,
        issue=issue,
    )

    direct_scores = defaultdict(float)
    for item in direct_evidence:
        direct_scores[item["entity"]] = max(direct_scores[item["entity"]], item["weight"])
    role_scores = defaultdict(float)
    for item in role_evidence:
        role_scores[item["entity"]] += item["score"]
    for item in entity_rankings:
        item["direct_evidence_score"] = direct_scores.get(item["entity"], 0.0)
        item["role_evidence_score"] = role_scores.get(item["entity"], 0.0)
        item["symbol_graph_score"] = symbol_scores.get(item["entity"], 0.0)

    enabled_direct = direct_evidence if use_direct else []
    enabled_roles = role_evidence if use_roles else []
    context_rankings = entity_rankings
    if not use_inference:
        context_rankings = [
            {**item, "bm25_score": 0.0, "query_overlap": 0.0, "lexical_score": 0.0}
            for item in entity_rankings
        ]
    protected_files = list(
        dict.fromkeys(
            path for path in (protected_file_paths or []) if path in file_nodes
        )
    )
    protected_file_set = set(protected_files)
    rankings_by_path = {item["path"]: item for item in file_rankings}
    paths_by_digest = defaultdict(list)
    for item in file_rankings:
        path = item["path"]
        digest = hashlib.sha256(
            code_graph.nodes[path].get("code", "").encode("utf-8")
        ).hexdigest()
        paths_by_digest[digest].append(path)
    issue_path_terms = set(tokenize(issue))
    canonical_by_digest = {}
    for digest, paths in paths_by_digest.items():
        protected_aliases = [path for path in paths if path in protected_file_set]
        canonical_by_digest[digest] = min(
            protected_aliases or paths,
            key=lambda path: (
                len(Path(path).parts),
                -len(issue_path_terms.intersection(tokenize(path))),
                -len(rankings_by_path[path].get("retrieval_sources", [])),
                -rankings_by_path[path].get("graph_score", 0.0),
                len(path),
                path,
            ),
        )
    canonical_rankings = []
    seen_canonical = set()
    for item in file_rankings:
        path = item["path"]
        digest = hashlib.sha256(
            code_graph.nodes[path].get("code", "").encode("utf-8")
        ).hexdigest()
        canonical_path = canonical_by_digest[digest]
        if canonical_path in seen_canonical:
            continue
        seen_canonical.add(canonical_path)
        aliases = paths_by_digest[digest]
        alias_items = [rankings_by_path[alias] for alias in aliases]
        canonical = dict(rankings_by_path[canonical_path])
        canonical["hybrid_score"] = max(alias["hybrid_score"] for alias in alias_items)
        canonical["rrf_score"] = max(alias["rrf_score"] for alias in alias_items)
        canonical["lexical_score"] = max(alias["lexical_score"] for alias in alias_items)
        canonical["graph_score"] = max(alias["graph_score"] for alias in alias_items)
        canonical["inference_quality"] = max(alias["inference_quality"] for alias in alias_items)
        canonical["retrieval_sources"] = list(
            dict.fromkeys(
                source
                for alias in alias_items
                for source in alias.get("retrieval_sources", [])
            )
        )
        canonical["duplicate_aliases"] = aliases
        canonical_rankings.append(canonical)
    canonical_rankings.sort(
        key=lambda item: (
            item["path"] not in protected_file_set,
            -item["hybrid_score"],
            item["path"],
        )
    )
    candidate_paths = [
        item["path"]
        for item in canonical_rankings[: max(candidate_count, len(protected_files))]
    ]
    rankings_by_path = {item["path"]: item for item in canonical_rankings}
    candidates = [rankings_by_path[path] for path in candidate_paths]
    selected_entities = select_global_entities(
        code_graph,
        context_rankings,
        symbol_scores,
        candidates,
        len(candidates),
        max(entity_budget, len(candidates)),
        use_direct,
        use_inference,
        use_roles,
    )
    candidate_context = build_candidate_context(
        code_graph,
        issue,
        candidates,
        selected_entities,
        symbol_scores,
        symbol_graph,
        seeds,
        enabled_direct,
        enabled_roles,
        use_inference,
        use_roles,
    )
    line_rankings = build_line_rankings(
        code_graph,
        candidate_context,
        set(tokenize(issue)) if use_inference else set(),
        hypotheses=fault_hypotheses,
    )
    return {
        "configuration": {
            "evidence_channels": {"E": use_direct, "I": use_inference, "R": use_roles},
            "use_bm25": use_bm25,
            "use_pagerank": use_pagerank,
            "seed_count": seed_count,
            "candidate_count": candidate_count,
            "entity_budget": entity_budget,
            "entity_selection": "one_per_candidate_file_then_global_competition",
            "line_selection": "evidence_lines_with_adjacent_context_ranked_for_token_budget",
            "lexical_weight": lexical_weight if use_inference else 0.0,
            "graph_weight": 1 - lexical_weight if use_inference else 1.0,
            "pagerank_alpha": PAGERANK_ALPHA,
            "direct_evidence_weight": DIRECT_EVIDENCE_WEIGHT,
            "symbol_aggregation": "0.5 * max + 0.5 * mean(top_3)",
            "relation_weights": RELATION_WEIGHTS,
            "reverse_factors": REVERSE_FACTORS,
            "role_pattern_library": ROLE_PATTERNS,
            "role_query_patterns": ROLE_QUERY_PATTERNS,
        },
        "entity_rankings": entity_rankings,
        "direct_evidence": direct_evidence,
        "role_evidence": role_evidence,
        "seeds": seeds,
        "relation_edges": relation_edges,
        "file_rankings": file_rankings,
        "line_rankings": line_rankings,
        "candidate_context": candidate_context,
    }

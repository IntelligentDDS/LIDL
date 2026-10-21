import os
import ast


def create_structure(directory_path):
    """Create the structure of the repository directory by parsing Python files.
    :param directory_path: Path to the repository directory.
    :return: A dictionary representing the structure.
    """
    structure = {}

    for root, _, files in os.walk(directory_path):
        repo_name = os.path.basename(directory_path)
        relative_root = os.path.relpath(root, directory_path)
        if relative_root == ".":
            relative_root = repo_name
        curr_struct = structure
        for part in relative_root.split(os.sep):
            if part not in curr_struct:
                curr_struct[part] = {}
            curr_struct = curr_struct[part]
        for file_name in files:
            if file_name.endswith(".py"):
                file_path = os.path.join(root, file_name)
                class_info, function_names, file_lines = parse_python_file(file_path)
                curr_struct[file_name] = {
                    "classes": class_info,
                    "functions": function_names,
                    "text": file_lines,
                }
            else:
                curr_struct[file_name] = {}

    return structure


def parse_python_file(file_path, file_content=None):
    """Parse a Python file to extract class and function definitions with their line numbers.
    :param file_path: Path to the Python file.
    :return: Class names, function names, and file contents
    """
    if file_content is None:
        try:
            with open(file_path, "r") as file:
                file_content = file.read()
                parsed_data = ast.parse(file_content)
        except Exception as e:  # Catch all types of exceptions
            print(f"Error in file {file_path}: {e}")
            return [], [], ""
    else:
        try:
            parsed_data = ast.parse(file_content)
        except Exception as e:  # Catch all types of exceptions
            print(f"Error in file {file_path}: {e}")
            return [], [], ""

    class_info = []
    function_names = []
    class_methods = set()

    for node in ast.walk(parsed_data):
        if isinstance(node, ast.ClassDef):
            methods = []
            for n in node.body:
                if isinstance(n, ast.FunctionDef):
                    methods.append(
                        {
                            "name": n.name,
                            "start_line": n.lineno,
                            "end_line": n.end_lineno,
                            "text": file_content.splitlines()[n.lineno - 1 : n.end_lineno],
                        }
                    )
                    class_methods.add(n.name)
            class_info.append(
                {
                    "name": node.name,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "text": file_content.splitlines()[node.lineno - 1 : node.end_lineno],
                    "methods": methods,
                }
            )
        elif isinstance(node, ast.FunctionDef) and not isinstance(node, ast.AsyncFunctionDef):
            if node.name not in class_methods:
                function_names.append(
                    {
                        "name": node.name,
                        "start_line": node.lineno,
                        "end_line": node.end_lineno,
                        "text": file_content.splitlines()[node.lineno - 1 : node.end_lineno],
                    }
                )

    return class_info, function_names, file_content.splitlines()


def construct_graph_context(found_related_locs, code_graph, graph_tags, structure, max_context_items=10):
    """Construct graph context string from found locations."""
    if not found_related_locs or not code_graph:
        return ""

    context_items = []
    seen_items = set()

    # extract all function/class names from found_related_locs
    target_names = []
    for file_name, locs in found_related_locs.items():
        if isinstance(locs, list):
            for loc in locs:
                if "function:" in loc:
                    func_name = loc.split("function:")[1].strip()
                    target_names.append(func_name)
                elif "class:" in loc:
                    class_name = loc.split("class:")[1].strip()
                    target_names.append(class_name)

    # get dependencies for each target
    for name in target_names[: max_context_items // 2]:  # limit number of items
        if name not in code_graph.nodes():
            continue

        # get nodes that call this function/class (incoming edges)
        callers = list(code_graph.predecessors(name))
        if callers and name not in seen_items:
            seen_items.add(name)
            context_items.append(f"- {name} is called by: {', '.join(callers[:5])}")

        # get nodes that this function/class calls (outgoing edges)
        callees = list(code_graph.successors(name))
        if callees and name not in seen_items:
            seen_items.add(name)
            context_items.append(f"- {name} calls: {', '.join(callees[:5])}")

        # get node information
        if name in code_graph.nodes():
            node_data = code_graph.nodes[name]
            if "info" in node_data and node_data["info"] != "none":
                info = node_data["info"]
                # truncate long information
                if len(info) > 200:
                    info = info[:200] + "..."
                context_items.append(f"- {name}: {info}")

    if not context_items:
        return ""

    # format output
    context = "The following functions/classes are related based on the code dependency graph:\n"
    context += "\n".join(context_items[:max_context_items])

    return context


def get_function_dependencies(func_name, code_graph, graph_tags):
    """Get functions that call the given function."""
    dependencies = []

    # Search in graph tags for references
    for tag in graph_tags:
        if tag.get("name") == func_name and tag.get("kind") == "ref":
            # This is a reference to our function
            file_path = tag.get("rel_fname", "")
            line = tag.get("line", -1)

            # Find containing function at this location
            for other_tag in graph_tags:
                if other_tag.get("rel_fname") == file_path and other_tag.get("kind") == "def" and other_tag.get("category") == "function":
                    if line >= other_tag.get("line", [0])[0] and line <= other_tag.get("line", [0])[-1]:
                        caller = f"{other_tag.get('name')} ({file_path})"
                        if caller not in dependencies:
                            dependencies.append(caller)

    return dependencies


# def get_class_dependencies(class_name, code_graph, graph_tags):
#     """Get locations where the class is instantiated or inherited."""
#     dependencies = []

#     # Search graph for class usage
#     if class_name in code_graph.nodes():
#         # Get incoming edges (who references this class)
#         for predecessor in code_graph.predecessors(class_name):
#             # Find file location for predecessor
#             for tag in graph_tags:
#                 if tag.get("name") == predecessor and tag.get("kind") == "def":
#                     location = f"{predecessor} ({tag.get('rel_fname', 'unknown')})"
#                     if location not in dependencies:
#                         dependencies.append(location)

#     return dependencies

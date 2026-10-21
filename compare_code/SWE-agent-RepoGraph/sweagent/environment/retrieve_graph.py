# retrieve code graph scripts

import pickle
import sys
import json


def main(func_name):
    with open('/root/graph.pkl', 'rb') as f:
        G = pickle.load(f)
    with open('/root/tags.json', 'r') as f:
        tags = f.readlines()
    tags = [json.loads(tag) for tag in tags]

    if func_name not in G:
        print("None")
        return

    try:
        successors = list(G.successors(func_name))
        predecessors = list(G.predecessors(func_name))
        tags2names = {}
        for tag in tags:
            tags2names.setdefault(tag["name"], []).append(tag)
        returned_files = []
        for item in successors + [func_name] + predecessors:
            for tag in tags2names[item]:
                if "test" in tag["fname"]:
                    continue
                returned_files.append({
                    "fname": tag["fname"],
                    "line": tag["line"],
                    "name": tag["name"],
                    "kind": tag["kind"],
                    "category": tag["category"],
                    "info": tag["info"],
                })
        print(returned_files)
    except (KeyError, AttributeError):
        print("None")

if __name__ == '__main__':
    func_name = sys.argv[1]
    main(func_name)
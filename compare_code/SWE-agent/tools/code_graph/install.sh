#!/usr/bin/env bash

# Get the bundle directory path
bundle_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Install python dependencies for code_graph tool
pip install networkx==3.4.2

# Add the bin directory to PATH so code_graph command can be found
export PATH="$bundle_dir/bin:$PATH"

# Make sure the code_graph script is executable
chmod +x "$bundle_dir/bin/code_graph"

# Write environment variables if the function exists
if command -v _write_env &> /dev/null; then
    _write_env "CODE_GRAPH_PATH" "$bundle_dir/bin"
fi

echo "code_graph tool installed successfully"
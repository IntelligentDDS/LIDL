#!/bin/bash

# Make search_repo executable
chmod +x /root/tools/repo_graph/bin/search_repo

# Add to PATH if not already there
if ! echo $PATH | grep -q "/root/tools/repo_graph/bin"; then
    echo 'export PATH=$PATH:/root/tools/repo_graph/bin' >> ~/.bashrc
    export PATH=$PATH:/root/tools/repo_graph/bin
fi

echo "repo_graph tool installed"
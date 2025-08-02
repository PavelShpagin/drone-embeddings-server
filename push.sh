#!/bin/bash

# This script automates adding, committing, and pushing all changes.
# It requires a single argument: the commit message.

# --- Check for commit message ---
if [ -z "$1" ]; then
    echo "Error: No commit message provided."
    echo "Usage: ./push.sh \"Your commit message\""
    exit 1
fi

git add .
git commit -m "$1"
git push

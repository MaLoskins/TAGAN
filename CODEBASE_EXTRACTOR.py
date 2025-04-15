#!/usr/bin/env python3
"""
Script to generate a codebase tree representation and collect code files.

Usage:
    python CODEBASE_EXTRACT.py <directory> [comma-separated-file-names]

The script will:
1. Traverse the provided directory and subdirectories.
2. Ignore certain folders (e.g. __pycache__, node_modules, venv, .venv, etc.)
   and files (e.g. package-lock.json, yarn.lock, or files with extensions like .csv, .db, .parquet)
   that are typically not part of the codebase.
3. Skip any file larger than 300 KB.
4. Optionally, if a comma-separated list of file names is provided as a second argument,
   only those files will be added to the output and shown in the tree.
5. Generate a tree diagram of the directory structure (only including allowed files and directories leading to them).
6. Create an output file called "output.md" in the provided directory that contains:
   - The tree diagram (wrapped in triple backticks)
   - The contents of each accepted code file (each with its own header and wrapped in triple backticks)
"""

import os
import sys


# Define sets of directories and files to ignore.
IGNORED_DIRS = {"__pycache__", "documentation", "tests", "node_modules", "venv", ".venv", "data", "dist", "build", ".git", "workspace"}
IGNORED_FILES = {"package-lock.json", "yarn.lock", "requirements.txt", "test*"}  # adjust as needed

# Define file extensions that are considered non-code (commonly data or binary files)
IGNORED_EXTENSIONS = {".csv", ".md", ".db", ".parquet", ".sqlite", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".zip", ".tar", ".gz"}

# Maximum file size to include (in bytes)
MAX_FILE_SIZE = 300 * 1024  # 300 KB

# Global set for allowed file names. If empty, all accepted files are included.
SPECIFIED_FILES = set()

#SPECIFIED_FILES.update(["docker-compose.yml", "Dockerfile", "deploy.sh", "wait-for-neo4j.sh", "wait-for-dependencies.sh"])

def should_ignore_dir(dirname):
    """
    Returns True if the directory name should be ignored.
    """
    return dirname in IGNORED_DIRS

def should_ignore_file(filename, filepath):
    """
    Returns True if the file should be ignored based on name, extension, file size,
    or if a specified file list is provided and the file is not in it.
    """
    base = os.path.basename(filename)
    if base in IGNORED_FILES:
        return True

    # Ignore by file extension.
    ext = os.path.splitext(filename)[1].lower()
    if ext in IGNORED_EXTENSIONS:
        return True

    # Check file size. If file size cannot be determined, assume it is acceptable.
    try:
        if os.path.getsize(filepath) > MAX_FILE_SIZE:
            return True
    except Exception as e:
        # In case of error, simply do not ignore the file based on size.
        pass

    # If SPECIFIED_FILES is not empty, only include files that are in the allowed list.
    if SPECIFIED_FILES and base not in SPECIFIED_FILES:
        return True

    return False

def build_tree(current_path, prefix=""):
    """
    Recursively builds the tree representation of the directory structure,
    but only includes files that pass the ignore filters (including the allowed files if specified).
    
    Returns:
        has_allowed: True if the current directory subtree contains any allowed files.
        lines: List of strings representing the tree diagram for the subtree.
        code_files: List of file paths that have been accepted in this subtree.
    """
    lines = []
    allowed_in_subtree = False
    code_files_local = []

    try:
        items = sorted(os.listdir(current_path))
    except Exception as e:
        # Skip directories that cannot be accessed.
        return False, [], []

    # Filter out directories that need to be ignored.
    filtered_items = []
    for item in items:
        item_path = os.path.join(current_path, item)
        if os.path.isdir(item_path) and should_ignore_dir(item):
            continue
        filtered_items.append(item)

    for index, item in enumerate(filtered_items):
        item_path = os.path.join(current_path, item)
        connector = "├── " if index != len(filtered_items) - 1 else "└── "

        if os.path.isdir(item_path):
            # Recursively build the tree for the subdirectory.
            has_allowed, subtree_lines, subtree_files = build_tree(item_path, prefix + ("    " if index == len(filtered_items) - 1 else "│   "))
            if has_allowed:
                lines.append(prefix + connector + item + "/")
                lines.extend(subtree_lines)
                allowed_in_subtree = True
                code_files_local.extend(subtree_files)
        else:
            # It's a file.
            if should_ignore_file(item, item_path):
                continue
            lines.append(prefix + connector + item)
            allowed_in_subtree = True
            code_files_local.append(item_path)

    return allowed_in_subtree, lines, code_files_local

def generate_tree(start_path):
    """
    Generate a tree representation of the directory structure starting at start_path.
    
    Returns:
        tree_str: A string with the tree diagram.
        code_files: A list of file paths that have been accepted (not ignored) and match allowed list if specified.
    """
    # Build the tree starting from the provided directory.
    _, lines, code_files = build_tree(start_path)
    root = os.path.abspath(start_path)
    tree_lines = [root] + lines
    tree_str = "\n".join(tree_lines)
    return tree_str, code_files

def main():
    global SPECIFIED_FILES

    if len(sys.argv) < 2:
        print("Usage: python CODEBASE_EXTRACT.py <directory> [comma-separated-file-names]")
        sys.exit(1)
    
    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print("Error: The provided path is not a directory.")
        sys.exit(1)
    
    # If a comma-separated list of file names is provided, update SPECIFIED_FILES.
    if len(sys.argv) >= 3:
        allowed_list = [fname.strip() for fname in sys.argv[2].split(",") if fname.strip()]
        SPECIFIED_FILES = set(allowed_list)
        print("Filtering to only include files with names in:", SPECIFIED_FILES)

    tree_diagram, code_files = generate_tree(directory)
    
    output_content = []
    output_content.append("# CODEBASE\n")
    output_content.append("## Directory Tree:\n")
    output_content.append("### " + os.path.abspath(directory) + "\n")
    output_content.append("```\n" + tree_diagram + "\n```\n")
    output_content.append("## Code Files\n")
    
    for file_path in code_files:
        rel_path = os.path.relpath(file_path, directory)
        full_path = os.path.join(os.path.abspath(directory), rel_path)
        output_content.append("\n### " + full_path + "\n")
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                code = f.read()
        except Exception as e:
            code = "Error reading file: " + str(e)
        output_content.append("```\n" + code + "\n```\n")
    
    output_file_path = os.path.join(directory, "output.md")
    try:
        with open(output_file_path, "w", encoding="utf-8") as out_file:
            out_file.write("\n".join(output_content))
        print("Output successfully saved to:", output_file_path)
    except Exception as e:
        print("Failed to write output file:", str(e))

if __name__ == "__main__":
    main()

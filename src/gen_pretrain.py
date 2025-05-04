import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm

# Define directories to skip at the module level for clarity
DIRS_TO_SKIP = {
    # Specific
    'ThirdParty', 'Extras', 'thirdparty',
    # Generic
    '.git', 'node_modules', 'bin', 'obj', 'Build', 'build', 'Intermediate', 'DerivedDataCache'
}

def is_path_in_skipped_dir(path: Path, skip_dirs: set) -> bool:
    """Checks if any part of the path is in the set of directories to skip."""
    return any(part in skip_dirs for part in path.parts)

def find_source_files(code_dir, extensions, skip_dirs):
    """
    Recursively finds all files with specified extensions in a directory,
    excluding specified directories.

    Args:
        code_dir (str): The root directory to search in.
        extensions (list): A list of file extensions to look for.
        skip_dirs (set): A set of directory names to exclude.

    Returns:
        list: A list of Path objects for the found files.
    """
    source_files_found = []
    root_path = Path(code_dir)
    print(f"Searching in: {root_path}")
    print(f"Skipping directories: {', '.join(skip_dirs)}")

    all_files = []
    for ext in extensions:
        # Ensure extension starts with a dot
        if not ext.startswith('.'):
            ext = '.' + ext
        # Use rglob to find all matching files initially
        all_files.extend(root_path.rglob(f'*{ext}'))

    # Filter out files in skipped directories
    for file_path in all_files:
        # Get the path relative to the root to check parts against skip_dirs
        try:
            relative_path = file_path.relative_to(root_path)
            if not is_path_in_skipped_dir(relative_path, skip_dirs):
                source_files_found.append(file_path)
        except ValueError:
            # This can happen if file_path is not under root_path, though unlikely with rglob
            # Or if file_path IS root_path (e.g., searching '.')
            # For simplicity, we'll just include it if it doesn't raise error and isn't skipped
            if not is_path_in_skipped_dir(file_path, skip_dirs):
                 source_files_found.append(file_path)


    return source_files_found

def generate_pretrain_dataset(code_dir, output_file):
    """
    Generates a pretraining dataset in JSON Lines format from C++ and C# source code files,
    excluding specified directories.

    Args:
        code_dir (str): The directory containing the source code files.
        output_file (str): The path to the output JSON Lines file.
    """
    # Hardcoded extensions
    extensions_to_find = ['.cpp', '.h', '.hpp', '.cc', '.cxx', '.hxx', '.cs']
    print(f"Searching for files with extensions: {', '.join(extensions_to_find)}...")
    # Pass the DIRS_TO_SKIP set to find_source_files
    source_files = find_source_files(code_dir, extensions_to_find, DIRS_TO_SKIP)
    total_files = len(source_files)
    print(f"Found {total_files} files after filtering. Generating dataset...")

    if total_files == 0:
        print("No files found to process. Exiting.")
        return # Exit early if no files

    root_path = Path(code_dir)
    output_path = Path(output_file)

    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed_files = 0
    try:
        with open(output_path, 'w', encoding='utf-8') as outfile:
            # Wrap the loop with tqdm for progress bar
            for file_path in tqdm(source_files, desc="Processing files", unit="file"):
                try:
                    # Use the already validated file_path which is absolute
                    relative_path = file_path.relative_to(root_path).as_posix() # Use POSIX paths for consistency
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as infile:
                        content = infile.read()
                        record = {
                            "file_path": relative_path,
                            "content": content
                        }
                        outfile.write(json.dumps(record, ensure_ascii=False) + '\n')
                        processed_files += 1
                except Exception as e:
                    # Log the error but continue processing other files
                    print(f"\nError processing file {file_path}: {e}") # Add newline to avoid messing up tqdm bar
    except IOError as e:
        print(f"Error opening or writing to output file {output_path}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print(f"\nSuccessfully generated pretraining dataset: {output_path}") # Add newline
    print(f"Processed {processed_files}/{total_files} files.") # Show processed vs total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pretraining dataset from C++ and C# source code files, excluding specified directories.")
    parser.add_argument("code_dir", help="Directory containing the source code files.")
    parser.add_argument("-o", "--output", default="./cpp_cs_dataset/dataset.pretrain.jsonl", help="Output JSON Lines file path (default: ./cpp_cs_dataset/dataset.pretrain.jsonl)")

    args = parser.parse_args()

    code_dir_path = Path(args.code_dir)
    if not code_dir_path.is_dir():
        print(f"Error: Directory not found: {args.code_dir}")
    else:
        # Pass the absolute path to the function
        generate_pretrain_dataset(str(code_dir_path.resolve()), args.output)

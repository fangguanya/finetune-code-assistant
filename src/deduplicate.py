import os
from pathlib import Path
import shutil # For potentially safer file replacement

def deduplicate_jsonl(filepath: Path):
    """
    Removes duplicate lines from a JSONL file in place.
    Keeps the first occurrence of each unique line.
    """
    if not filepath.is_file():
        print(f"Error: File not found - {filepath}")
        return

    temp_filepath = filepath.with_suffix(filepath.suffix + '.tmp')
    seen_lines = set()
    original_line_count = 0
    deduplicated_line_count = 0

    print(f"Deduplicating {filepath}...")
    try:
        with open(filepath, 'r', encoding='utf-8') as infile, \
             open(temp_filepath, 'w', encoding='utf-8') as outfile:
            for line in infile:
                original_line_count += 1
                # Strip whitespace for comparison, but write original line
                stripped_line = line.strip()
                if stripped_line not in seen_lines:
                    seen_lines.add(stripped_line)
                    outfile.write(line) # Write the original line including newline
                    deduplicated_line_count += 1
                # Optional: Progress indicator for large files
                # if original_line_count % 100000 == 0:
                #    print(f"Processed {original_line_count} lines...", end='\r')

        print(f"\nFinished deduplicating {filepath}.")
        print(f"Original lines: {original_line_count}")
        print(f"Unique lines: {deduplicated_line_count}")
        print(f"Removed {original_line_count - deduplicated_line_count} duplicate lines.")

        # Replace the original file with the temporary deduplicated file
        # os.replace(temp_filepath, filepath) # Simple replace
        shutil.move(str(temp_filepath), str(filepath)) # shutil.move might be more robust across filesystems/drives
        print(f"Successfully updated {filepath}.")

    except Exception as e:
        print(f"Error during deduplication of {filepath}: {e}")
        # Clean up temporary file if it exists on error
        if temp_filepath.exists():
            try:
                os.remove(temp_filepath)
            except OSError:
                print(f"Warning: Could not remove temporary file {temp_filepath}")
    finally:
        # Ensure temp file is removed if replacement failed silently for some reason
        if temp_filepath.exists():
             try:
                 os.remove(temp_filepath)
                 print(f"Cleaned up temporary file {temp_filepath}")
             except OSError:
                  print(f"Warning: Could not remove temporary file {temp_filepath}")


# --- How to use it ---
output_dir = Path("./cpp_cs_dataset") # Make sure this matches your OUTPUT_DIR
train_file = output_dir / "dataset.train.jsonl"
test_file = output_dir / "dataset.test.jsonl"

deduplicate_jsonl(train_file)
deduplicate_jsonl(test_file)

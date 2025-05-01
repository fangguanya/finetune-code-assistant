import asyncio
import json
import os
import random
import sys
import logging # <-- Import logging module
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

# Tree-sitter imports
# Ensure tree-sitter is installed: pip install tree-sitter
try:
    from tree_sitter import Language, Parser
except ImportError:
    # Cannot use logging here as it's not configured yet
    print("Fatal: 'tree-sitter' library not found.", file=sys.stderr)
    print("Please install it using: pip install tree-sitter", file=sys.stderr)
    sys.exit(1)

# --- 配置常量 ---
# 定义 C++ 和 C# 的文件扩展名
CPP_EXTENSIONS = {'.cpp', '.h', '.hpp', '.cc', '.cxx', '.hxx'}
CSHARP_EXTENSIONS = {'.cs'}
TARGET_EXTENSIONS = CPP_EXTENSIONS.union(CSHARP_EXTENSIONS)

# 节点大小范围（以字节为单位，tree-sitter 使用字节偏移）
MIN_NODE_SPAN = 20  # 最小代码块字节数
MAX_NODE_SPAN_RATIO = 0.1 # 最大代码块占文件大小比例
MAX_NODE_SPAN_ABS = 5000 # 最大代码块绝对字节数
MIN_MAX_NODE_SPAN = 100 # max_node_span 的下限

# 输出配置
MAX_CHUNK_LEN = 8000 # prefix + middle + suffix 的最大长度
TRAIN_TEST_SPLIT_RATIO = 0.98 # 训练集比例
OUTPUT_DIR = Path("./cpp_cs_dataset") # 输出目录
TRAIN_FILENAME = "dataset.train.jsonl"
TEST_FILENAME = "dataset.test.jsonl"
FILES_LOG_FILENAME = "files.log" # Log processed files
IGNORES_LOG_FILENAME = "ignores.log" # Log skipped directories
ERROR_LOG_FILENAME = "error.log" # Log errors and warnings

# --- Tree-sitter Language Loading ---
# 此脚本现在 *期望* 语言库文件已经存在于指定路径
CPP_LANGUAGE: Optional[Language] = None
CSHARP_LANGUAGE: Optional[Language] = None
# Determine library extension based on OS
LIB_EXTENSION = '.dll' if sys.platform == 'win32' else '.so'
BUILD_LIB_PATH = Path('build') / f'languages{LIB_EXTENSION}'
VENDOR_DIR = Path('vendor') # Directory to clone grammars into

# Function to locate or build the language library
def load_or_build_languages():
    """Loads or builds the Tree-sitter language libraries.
    NOTE: Automatic building requires py-tree-sitter < 0.22
    Logs warnings/errors using the configured logging system.
    """
    global CPP_LANGUAGE, CSHARP_LANGUAGE

    cpp_grammar_path = VENDOR_DIR / 'tree-sitter-cpp'
    csharp_grammar_path = VENDOR_DIR / 'tree-sitter-c-sharp' # Corrected name

    BUILD_LIB_PATH.parent.mkdir(exist_ok=True)

    needs_build = not BUILD_LIB_PATH.exists()
    if not needs_build:
        try:
            lib_mtime = BUILD_LIB_PATH.stat().st_mtime
            cpp_mtime = cpp_grammar_path.stat().st_mtime if cpp_grammar_path.exists() else -1
            cs_mtime = csharp_grammar_path.stat().st_mtime if csharp_grammar_path.exists() else -1
            if cpp_mtime > lib_mtime or cs_mtime > lib_mtime:
                needs_build = True
        except FileNotFoundError:
            needs_build = True
        except Exception as e:
             logging.warning(f"Could not check modification times ({e}). Assuming build not needed if library exists.", exc_info=True)


    if needs_build:
        logging.info("Attempting to build Tree-sitter languages (requires py-tree-sitter < 0.22)...")

        clone_commands = []
        if not cpp_grammar_path.is_dir():
            logging.warning(f"Directory not found: {cpp_grammar_path}. Will attempt to clone.")
            VENDOR_DIR.mkdir(parents=True, exist_ok=True)
            clone_commands.append(f"git clone https://github.com/tree-sitter/tree-sitter-cpp {cpp_grammar_path}")
        if not csharp_grammar_path.is_dir():
            logging.warning(f"Directory not found: {csharp_grammar_path}. Will attempt to clone.")
            VENDOR_DIR.mkdir(parents=True, exist_ok=True)
            clone_commands.append(f"git clone https://github.com/tree-sitter/tree-sitter-c-sharp {csharp_grammar_path}")

        if clone_commands:
             logging.error("Required grammar repositories missing.")
             logging.error("Please run the following commands in your terminal:")
             for cmd in clone_commands:
                 logging.error(f"  {cmd}")
             logging.error("\nThen re-run this script.")
             sys.exit(1)

        try:
            if not hasattr(Language, 'build_library'):
                raise AttributeError("Language.build_library not found. You might need py-tree-sitter < 0.22 to build automatically.")

            Language.build_library(
                str(BUILD_LIB_PATH),
                [str(cpp_grammar_path), str(csharp_grammar_path)]
            )
            logging.info(f"Languages built successfully to '{BUILD_LIB_PATH}'.")
        except AttributeError as attr_err:
            logging.error(f"{attr_err}")
            logging.error("Automatic building failed. This script version requires Language.build_library.")
            logging.error("Please either install an older version of the library (pip install \"py-tree-sitter<0.22\")")
            logging.error(f"OR manually compile the library and place it at: {BUILD_LIB_PATH}")
            sys.exit(1)
        except Exception as build_e:
            logging.error(f"Failed to build Tree-sitter languages: {build_e}", exc_info=True)
            logging.error("Troubleshooting steps:")
            logging.error("1. Ensure you have a C/C++ compiler installed.")
            logging.error("2. Make sure the compiler is in your system's PATH.")
            logging.error(f"3. Verify grammar repos exist: {cpp_grammar_path.resolve()}, {csharp_grammar_path.resolve()}")
            logging.error("4. Try deleting the 'build' directory and running again.")
            sys.exit(1)

    try:
        CPP_LANGUAGE = Language(str(BUILD_LIB_PATH), 'cpp')
        try:
             CSHARP_LANGUAGE = Language(str(BUILD_LIB_PATH), 'c_sharp')
        except ValueError:
             logging.warning("Could not load C# language with symbol 'c_sharp', trying 'csharp'...")
             CSHARP_LANGUAGE = Language(str(BUILD_LIB_PATH), 'csharp')

        logging.info(f"Tree-sitter languages loaded successfully from {BUILD_LIB_PATH}.")
    except Exception as load_e:
        logging.error(f"Failed to load Tree-sitter languages from '{BUILD_LIB_PATH}': {load_e}", exc_info=True)
        logging.error("This might indicate an issue with the build process or the compiled library itself.")
        logging.error(f"Ensure the file exists and is a valid library for your system: {BUILD_LIB_PATH.resolve()}")
        sys.exit(1)

# --- Tree-sitter 解析和节点提取 ---
# (CPP_CRITICAL_NODE_TYPES, CSHARP_CRITICAL_NODE_TYPES remain the same)
CPP_CRITICAL_NODE_TYPES = {
    'function_definition',
    'class_specifier', 'struct_specifier', 'enum_specifier', 'union_specifier',
    'namespace_definition',
    'template_declaration',
    'if_statement', 'for_statement', 'while_statement', 'do_statement', 'switch_statement',
    'compound_statement', # C/C++ block statements {}
    'try_statement', 'catch_clause',
    'declaration', # 可能过于宽泛，但可以包含变量、类型定义等
    'expression_statement', # 包含函数调用、赋值等
}
CSHARP_CRITICAL_NODE_TYPES = {
    'method_declaration',
    'class_declaration', 'struct_declaration', 'interface_declaration', 'enum_declaration', 'record_declaration', 'record_struct_declaration',
    'namespace_declaration',
    'if_statement', 'for_statement', 'foreach_statement', 'while_statement', 'do_statement', 'switch_statement',
    'block', # C# block statements {}
    'try_statement', 'catch_clause', 'finally_clause',
    'property_declaration', 'event_declaration', 'indexer_declaration',
    'constructor_declaration', 'destructor_declaration',
    'local_function_statement',
    'using_statement',
    'lock_statement',
    'expression_statement',
}


def get_critical_nodes_from_tree(node, file_content_bytes: bytes, critical_types: set, min_span: int, max_span: int) -> List[Dict[str, int]]:
    """递归遍历 Tree-sitter 树并提取关键节点"""
    nodes = []
    if node.start_byte is None or node.end_byte is None or node.start_byte >= node.end_byte:
        return nodes

    node_span = node.end_byte - node.start_byte
    is_critical = node.type in critical_types
    is_within_span = min_span <= node_span <= max_span

    if is_critical and is_within_span:
        nodes.append({'start': node.start_byte, 'end': node.end_byte})
    elif node.type == 'ERROR':
         logging.debug(f"Encountered ERROR node type at bytes {node.start_byte}-{node.end_byte}, stopping recursion here.")
         return nodes # Stop recursion into error nodes

    for child in node.children:
        if child.start_byte >= node.start_byte and child.end_byte <= node.end_byte and child != node:
             nodes.extend(get_critical_nodes_from_tree(child, file_content_bytes, critical_types, min_span, max_span))
    return nodes


def extract_critical_blocks_cpp_cs(file_path: Path, file_contents: str, language: Language, critical_types: set) -> List[Dict[str, int]]:
    """使用 Tree-sitter 解析 C++/C# 文件并提取关键块"""
    if not language: # Safety check
        logging.error(f"Language object not available for parsing file: {file_path}")
        return []
    parser = Parser()
    parser.set_language(language)

    file_content_bytes = file_contents.encode('utf-8', errors='ignore') # Tree-sitter 使用 bytes
    if not file_content_bytes:
        logging.warning(f"File content is empty after encoding (or originally empty): {file_path}")
        return []

    try:
        tree = parser.parse(file_content_bytes)
        if tree is None:
             logging.warning(f"Tree-sitter parsing timed out for file: {file_path}")
             return []
    except Exception as e:
        logging.error(f"Tree-sitter parsing error for file {file_path}: {e}", exc_info=True)
        return [] # 解析失败返回空

    root_node = tree.root_node
    if not root_node or root_node.has_error():
        logging.warning(f"Parsing resulted in errors or empty tree for file: {file_path}")
        # Optionally log specific errors if needed: traverse for ERROR nodes

    file_len = len(file_content_bytes)
    max_node_span_calc = max(MIN_MAX_NODE_SPAN, int(file_len * MAX_NODE_SPAN_RATIO))
    max_node_span = min(max_node_span_calc, MAX_NODE_SPAN_ABS)

    critical_nodes = get_critical_nodes_from_tree(root_node, file_content_bytes, critical_types, MIN_NODE_SPAN, max_node_span)

    unique_nodes = []
    seen_spans = set()
    for node in critical_nodes:
        span_key = (node['start'], node['end'])
        if span_key not in seen_spans:
            unique_nodes.append(node)
            seen_spans.add(span_key)

    unique_nodes.sort(key=lambda x: (x['start'], x['end']))
    return unique_nodes


# --- 数据集生成逻辑 ---

def determine_split() -> str:
    """根据比例决定数据集划分"""
    return 'train' if random.random() < TRAIN_TEST_SPLIT_RATIO else 'test'

async def process_file(target_file_path: Path, root_dir: Path, train_set_writer, test_set_writer) -> Tuple[int, int]:
    """处理单个 C++/C# 文件"""
    successful_samples = 0
    file_ext = target_file_path.suffix.lower()
    language: Optional[Language] = None
    critical_types: set = set()
    lang_name = "unknown"

    # Log the file being processed (INFO level goes to files.log by default config)
    logging.info(f"{target_file_path.as_posix()}")

    if file_ext in CPP_EXTENSIONS: language, critical_types, lang_name = CPP_LANGUAGE, CPP_CRITICAL_NODE_TYPES, "cpp"
    elif file_ext in CSHARP_EXTENSIONS: language, critical_types, lang_name = CSHARP_LANGUAGE, CSHARP_CRITICAL_NODE_TYPES, "csharp"
    else: return 0, 0 # Not a target file type

    if not language:
         logging.error(f"Skipping {target_file_path}, language object not available (should not happen if loaded).")
         return 0, 1 # Failed file

    try:
        MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024 # 10 MB
        file_size = await asyncio.to_thread(os.path.getsize, target_file_path)
        if file_size == 0:
            logging.warning(f"Skipping empty file: {target_file_path}")
            return 0, 0 # Not an error, just skip
        if file_size > MAX_FILE_SIZE_BYTES:
             logging.warning(f"Skipping {target_file_path}, file size ({file_size} bytes) exceeds limit ({MAX_FILE_SIZE_BYTES} bytes).")
             return 0, 1 # Treat large files as skipped/failed for this process

        file_contents = await asyncio.to_thread(target_file_path.read_text, encoding='utf-8', errors='ignore')
        if not file_contents:
             logging.warning(f"File content is empty after reading (or reading with errors ignored): {target_file_path}")
             return 0, 0 # Skip if empty after read

        nodes = await asyncio.to_thread(extract_critical_blocks_cpp_cs, target_file_path, file_contents, language, critical_types)
        if not nodes:
            # No critical nodes found or parsing failed (already logged in extract func)
            return 0, 0 # 0 samples, 0 failed file

        relative_path = target_file_path.relative_to(root_dir).as_posix()
        file_contents_bytes = file_contents.encode('utf-8', errors='ignore')
        file_len_bytes = len(file_contents_bytes)

        for node in nodes:
            start, end = node['start'], node['end']
            if not (0 <= start < end <= file_len_bytes):
                 logging.warning(f"Skipping invalid node span {start}-{end} (file len: {file_len_bytes}) in {target_file_path}")
                 continue

            middle_len = end - start
            if middle_len <= 0: continue # Should be caught by above, but safety check
            remaining = MAX_CHUNK_LEN - middle_len
            if remaining < 0:
                logging.debug(f"Node span {start}-{end} (len {middle_len}) exceeds MAX_CHUNK_LEN {MAX_CHUNK_LEN} in {target_file_path}. Skipping sample.")
                continue

            half_remaining = remaining // 2
            prefix_start = max(0, start - half_remaining)
            suffix_start = end
            suffix_end = min(file_len_bytes, suffix_start + half_remaining)

            current_prefix_len = start - prefix_start
            current_suffix_len = suffix_end - suffix_start
            current_total_len = current_prefix_len + middle_len + current_suffix_len

            if current_total_len > MAX_CHUNK_LEN:
                excess = current_total_len - MAX_CHUNK_LEN
                cut_suffix = min(excess // 2 + excess % 2, current_suffix_len)
                suffix_end -= cut_suffix
                excess -= cut_suffix # Corrected excess calculation

                if excess > 0:
                     cut_prefix = min(excess, current_prefix_len)
                     prefix_start += cut_prefix

            prefix_bytes = file_contents_bytes[prefix_start:start]
            middle_bytes = file_contents_bytes[start:end]
            suffix_bytes = file_contents_bytes[suffix_start:suffix_end]

            prefix = prefix_bytes.decode('utf-8', errors='replace')
            middle = middle_bytes.decode('utf-8', errors='replace')
            suffix = suffix_bytes.decode('utf-8', errors='replace')

            dataset_split = determine_split()
            jsonl_data = {'filePath': relative_path, 'language': lang_name, 'prefix': prefix, 'middle': middle, 'suffix': suffix}
            try:
                 jsonl_string = json.dumps(jsonl_data, ensure_ascii=False)
            except TypeError as json_err:
                 # Log error with file path and node details for context
                 logging.error(f"Could not serialize data to JSON for node {start}-{end} in {target_file_path}: {json_err}. Data: {jsonl_data}", exc_info=True)
                 continue # Skip this sample

            writer = train_set_writer if dataset_split == 'train' else test_set_writer
            await asyncio.to_thread(writer.write, jsonl_string + '\n')
            successful_samples += 1
        return successful_samples, 0

    except FileNotFoundError:
        logging.error(f"File not found: {target_file_path}")
        return 0, 1
    except UnicodeDecodeError as ude:
        logging.error(f"Could not decode file {target_file_path} as utf-8: {ude}", exc_info=False) # Keep ude brief
        return 0, 1
    except OSError as os_err:
         logging.error(f"OS Error processing file {target_file_path}: {os_err}", exc_info=True)
         return 0, 1
    except Exception as e:
        # Catch any other unexpected error during processing of this file
        logging.exception(f"Unexpected Error processing file {target_file_path}: {e}") # Use logging.exception for traceback
        return 0, 1 # Mark as failed file

# --- Logging Setup ---
def setup_logging(files_log_path: Path, error_log_path: Path):
    """Configures logging for INFO to files.log and WARNING/ERROR to error.log."""
    log_level = logging.INFO # Set base level

    # Root logger configuration
    logger = logging.getLogger()
    logger.setLevel(log_level)
    # Remove existing handlers to avoid duplicates if script is run multiple times in same process
    if logger.hasHandlers():
        logger.handlers.clear()

    # Handler for INFO messages (processed file paths) to files.log (overwrite)
    info_formatter = logging.Formatter('%(message)s')
    info_handler = logging.FileHandler(files_log_path, mode='w', encoding='utf-8')
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(info_formatter)
    # Filter to only allow INFO level, preventing warnings/errors from going here
    info_handler.addFilter(lambda record: record.levelno == logging.INFO)
    logger.addHandler(info_handler)

    # Handler for WARNING and ERROR messages to error.log (overwrite)
    error_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    error_handler = logging.FileHandler(error_log_path, mode='w', encoding='utf-8')
    error_handler.setLevel(logging.WARNING) # Catch WARNING and above
    error_handler.setFormatter(error_formatter)
    logger.addHandler(error_handler)

    # (Optional) Handler for console output (show INFO and above)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler = logging.StreamHandler(sys.stderr) # Log info/warn/error to stderr
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    # logger.addHandler(console_handler) # Uncomment to enable console logging via logging module


# --- Main Execution ---
async def main(root_dir_str: str):
    """主函数"""
    root_dir = Path(root_dir_str).resolve()
    if not root_dir.is_dir():
        # Logging not set up yet, print to stderr
        print(f"Fatal: Provided path '{root_dir_str}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Configure Logging ---
    files_log_path = OUTPUT_DIR / FILES_LOG_FILENAME
    ignores_log_path = OUTPUT_DIR / IGNORES_LOG_FILENAME
    error_log_path = OUTPUT_DIR / ERROR_LOG_FILENAME
    setup_logging(files_log_path, error_log_path)
    # -------------------------

    # --- Load Languages (after logging setup) ---
    try:
        load_or_build_languages()
        if not CPP_LANGUAGE or not CSHARP_LANGUAGE:
             logging.critical("Languages were not loaded correctly after attempt. Exiting.")
             sys.exit(1)
    except SystemExit: # Catch sys.exit calls from load_or_build
        raise # Re-raise to exit script
    except Exception as load_exc:
        logging.critical(f"Unhandled exception during language loading: {load_exc}", exc_info=True)
        sys.exit(1)
    # ------------------------------------------

    train_file_path = OUTPUT_DIR / TRAIN_FILENAME
    test_file_path = OUTPUT_DIR / TEST_FILENAME

    # --- File Discovery ---
    logging.info(f"Scanning for {', '.join(TARGET_EXTENSIONS)} files in {root_dir}...")
    target_files = []

    def find_files_sync():
        """Synchronous function to find files using os.walk, skipping specific directories."""
        count = 0
        dirs_to_skip_specific = {'ThirdParty', 'Extras', 'thirdparty'}
        dirs_to_skip_generic = {'.git', 'node_modules', 'bin', 'obj', 'Build', 'build', 'Intermediate', 'DerivedDataCache'}
        try:
             # Open ignores log in write/overwrite mode ('w')
             with open(ignores_log_path, 'w', encoding='utf-8') as ignores_log_file:
                 ignores_log_file.write(f"# Skipped directories during scan of: {root_dir}\n")
                 for dirpath, dirnames, filenames in os.walk(root_dir, topdown=True):
                     current_dir_path = Path(dirpath)
                     dirs_to_remove = set()

                     # Check and log directories to be skipped
                     original_dirnames = list(dirnames) # Copy for iteration while modifying
                     for d in original_dirnames:
                         if d in dirs_to_skip_specific or d in dirs_to_skip_generic or d.startswith('.'):
                             full_skip_path = current_dir_path / d
                             ignores_log_file.write(f"{full_skip_path.as_posix()}\n") # Log path
                             # Use list.remove() on the original dirnames list being modified by os.walk
                             try:
                                 dirnames.remove(d)
                             except ValueError: # Should not happen if iterating a copy, but safety
                                 pass

                     # Process files in the current directory
                     for filename in filenames:
                         file_path = current_dir_path / filename
                         if file_path.suffix.lower() in TARGET_EXTENSIONS:
                             target_files.append(file_path)
                             count += 1
                             # Keep console print for progress as logging might be buffered/slow for this
                             if count % 1000 == 0:
                                 print(f"Found {count} files...", end='\r', file=sys.stdout) # Print progress to stdout
        except OSError as e:
             # Log error related to ignores.log
             logging.error(f"Error writing to ignores log file '{ignores_log_path}': {e}. Ignore logging stopped.", exc_info=True)
             # Continue scan without ignore logging if file fails mid-way? Or abort?
             # For simplicity, log the error and continue the scan without further ignore logging.
             for dirpath, dirnames, filenames in os.walk(root_dir, topdown=True):
                 # Simplified pruning without logging
                 dirnames[:] = [d for d in dirnames if not (d in dirs_to_skip_specific or d in dirs_to_skip_generic or d.startswith('.'))]
                 for filename in filenames:
                      file_path = Path(dirpath) / filename
                      if file_path.suffix.lower() in TARGET_EXTENSIONS:
                          target_files.append(file_path)
                          count += 1
                          if count % 1000 == 0: print(f"Found {count} files...", end='\r', file=sys.stdout)
        finally:
             print(file=sys.stdout) # Newline after scan finish / progress indicator on stdout

    await asyncio.to_thread(find_files_sync)

    if not target_files:
        logging.warning("No target files found in the specified directory.")
        sys.exit(0)

    logging.info(f"Found {len(target_files)} target files. Starting processing...")
    print(f"Found {len(target_files)} target files. Starting processing...") # Also print to console

    total_files = len(target_files)
    total_samples = 0
    successful_files = 0
    failed_files = 0

    # --- File Processing ---
    train_set_writer = None
    test_set_writer = None
    try:
        train_set_writer = open(train_file_path, 'a', encoding='utf-8') # Keep append for dataset
        test_set_writer = open(test_file_path, 'a', encoding='utf-8')  # Keep append for dataset

        tasks = [process_file(file_path, root_dir, train_set_writer, test_set_writer) for file_path in target_files]

        import time
        start_time = time.time()
        results_iterable = None
        try:
             from tqdm.asyncio import tqdm_asyncio
             results_iterable = tqdm_asyncio(asyncio.as_completed(tasks), total=total_files, desc="Processing files", unit="file", file=sys.stdout) # Ensure tqdm writes to stdout
        except ImportError:
             results_iterable = asyncio.as_completed(tasks)
             print("Processing (install tqdm for progress bar: pip install tqdm)...", file=sys.stdout)

        for future in results_iterable:
             try:
                 samples, failed_flag = await future
                 total_samples += samples
                 if failed_flag: failed_files += 1
                 else: successful_files += 1
             except Exception as task_exc:
                 logging.exception(f"Error processing future result: {task_exc}") # Log exceptions from await future
                 failed_files += 1

    except OSError as e:
        logging.error(f"Error opening dataset output files '{train_file_path}' or '{test_file_path}': {e}", exc_info=True)
    except Exception as main_exc:
         logging.exception(f"An unexpected error occurred during main processing loop: {main_exc}")
    finally:
        if train_set_writer: await asyncio.to_thread(train_set_writer.close)
        if test_set_writer: await asyncio.to_thread(test_set_writer.close)

    end_time = time.time()
    duration = end_time - start_time

    # --- Summary Output (Print to console, not log files) ---
    print("\n--- Processing Summary ---")
    print(f"Total files scanned initially: {total_files}") # Clarify this count
    files_attempted = successful_files + failed_files
    print(f"Files attempted processing: {files_attempted}")
    print(f"Files processed successfully (may include 0 samples): {successful_files}")
    print(f"Files failed or skipped (errors, too large, etc.): {failed_files}")
    print(f"Total samples generated: {total_samples}")
    print(f"Train samples written to: {train_file_path.resolve()}")
    print(f"Test samples written to: {test_file_path.resolve()}")
    print(f"Processed file list written to: {files_log_path.resolve()}")
    print(f"Skipped directory list written to: {ignores_log_path.resolve()}")
    print(f"Errors and warnings written to: {error_log_path.resolve()}") # <-- Inform about error log
    print(f"Total processing time: {duration:.2f} seconds")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <path_to_cpp_cs_project_root>")
        sys.exit(1)

    project_root_dir = sys.argv[1]
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Wrap main execution in try/except to catch errors before logging is fully set up
    # or if logging setup itself fails.
    try:
        asyncio.run(main(project_root_dir))
    except Exception as e:
        # Basic fallback if logging failed or error occurred before setup
        print(f"Critical error during script execution: {e}", file=sys.stderr)
        # Optionally print traceback
        # import traceback
        # traceback.print_exc()
        sys.exit(1)
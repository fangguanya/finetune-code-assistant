# -*- coding: utf-8 -*-
# 首先，需要安装 tree-sitter 和相应的语言库
# pip install tree-sitter==0.21.3
# 注意：此脚本现在期望 C++ 和 C# 的 tree-sitter 语言库 (.so 或 .dll)
# 已被编译并放置在指定的路径下 (例如 'build/languages.so|dll')。
# 它不再尝试自动编译它们。

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
    print("Error: 'tree-sitter' library not found.", file=sys.stderr)
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

# --- Tree-sitter Language Loading ---
# 此脚本现在 *期望* 语言库文件已经存在于指定路径
CPP_LANGUAGE: Optional[Language] = None
CSHARP_LANGUAGE: Optional[Language] = None
# Determine library extension based on OS
LIB_EXTENSION = '.dll' if sys.platform == 'win32' else '.so'
# Path where the script expects the pre-compiled shared library
# 用户需要确保这个文件是通过其他方式编译并放置在这里的
EXPECTED_LIB_PATH = Path('build') / f'languages{LIB_EXTENSION}'

# Attempt to load the pre-compiled library
try:
    if not EXPECTED_LIB_PATH.exists():
        raise FileNotFoundError(f"Expected language library not found at: {EXPECTED_LIB_PATH}")
    CPP_LANGUAGE = Language(str(EXPECTED_LIB_PATH), 'cpp')
    CSHARP_LANGUAGE = Language(str(EXPECTED_LIB_PATH), 'csharp')
    print(f"Tree-sitter languages loaded successfully from {EXPECTED_LIB_PATH}.")
except Exception as load_e:
    print(f"\nError: Failed to load Tree-sitter languages from '{EXPECTED_LIB_PATH}': {load_e}", file=sys.stderr)
    print("\nThis script requires a pre-compiled Tree-sitter library containing C++ and C# grammars.", file=sys.stderr)
    print("Please ensure the following steps are completed:", file=sys.stderr)
    print("1. You have a C/C++ compiler installed.", file=sys.stderr)
    print("2. You have cloned the grammar repositories:", file=sys.stderr)
    print("   git clone https://github.com/tree-sitter/tree-sitter-cpp vendor/tree-sitter-cpp")
    print("   git clone https://github.com/tree-sitter/tree-sitter-c-sharp vendor/tree-sitter-c-sharp")
    print(f"3. You have manually built the shared library.")
    print(f"   IMPORTANT: `Language.build_library` was removed in py-tree-sitter >= 0.22.", file=sys.stderr)
    print(f"   You MUST use an older version (e.g., 0.21.x) to build the library:", file=sys.stderr)
    print(f"     pip install \"py-tree-sitter<0.22\"", file=sys.stderr)
    print(f"   Then run the build command:", file=sys.stderr)
    print(f"     python -c \"from tree_sitter import Language; Language.build_library('{EXPECTED_LIB_PATH}', ['vendor/tree-sitter-cpp', 'vendor/tree-sitter-c-sharp'])\"", file=sys.stderr)
    print(f"4. The resulting library file ('{EXPECTED_LIB_PATH.name}') exists at the expected path: '{EXPECTED_LIB_PATH.resolve()}'", file=sys.stderr)
    print(f"5. You can upgrade py-tree-sitter back to the latest version after building the library if needed.", file=sys.stderr)
    sys.exit(1)


# --- Tree-sitter 解析和节点提取 ---

# 定义我们认为"关键"的 C++ 和 C# 节点类型
# 这需要根据 tree-sitter-cpp 和 tree-sitter-c-sharp 的 grammar 来确定
# 这是一个示例列表，可能需要调整
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
    # Add more as needed based on grammar inspection
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
    # 'local_declaration_statement', # 可能太细粒度
    # Add more as needed based on grammar inspection
}

def get_critical_nodes_from_tree(node, file_content_bytes: bytes, critical_types: set, min_span: int, max_span: int) -> List[Dict[str, int]]:
    """递归遍历 Tree-sitter 树并提取关键节点"""
    nodes = []
    # Skip processing if node location is invalid (can happen with errors)
    if node.start_byte is None or node.end_byte is None or node.start_byte >= node.end_byte:
        return nodes

    node_span = node.end_byte - node.start_byte

    # 检查当前节点是否是关键类型且大小合适
    is_critical = node.type in critical_types
    is_within_span = min_span <= node_span <= max_span

    if is_critical and is_within_span:
        # Optional: Check if the node content is mainly whitespace
        # node_text = file_content_bytes[node.start_byte:node.end_byte].strip()
        # if node_text:
        nodes.append({'start': node.start_byte, 'end': node.end_byte})
    elif node.type == 'ERROR':
         # Stop recursion into error nodes to avoid potentially large/invalid subtrees
         return nodes

    # Decide whether to recurse based on whether the parent was added
    # If parent was added, we might not need its direct children unless they are also significant blocks.
    # If parent was NOT added (e.g., too small, too large, or not critical type), definitely check children.
    # This simple recursion checks all children regardless. Adjust if needed.
    for child in node.children:
        # Avoid infinite recursion in case of grammar errors leading to cycles (unlikely but possible)
        if child.start_byte >= node.start_byte and child.end_byte <= node.end_byte and child != node:
             nodes.extend(get_critical_nodes_from_tree(child, file_content_bytes, critical_types, min_span, max_span))

    return nodes


def extract_critical_blocks_cpp_cs(file_contents: str, language: Language, critical_types: set) -> List[Dict[str, int]]:
    """使用 Tree-sitter 解析 C++/C# 文件并提取关键块"""
    if not language: # Safety check
        print("Error: Language not loaded for parsing.", file=sys.stderr)
        return []
    parser = Parser()
    parser.set_language(language)

    file_content_bytes = file_contents.encode('utf-8') # Tree-sitter 使用 bytes
    try:
        # Add a timeout to parsing large/complex files to prevent hangs
        # Adjust timeout value as needed (in microseconds)
        TIMEOUT_MICROS = 10_000_000 # 10 seconds
        tree = parser.parse(file_content_bytes, timeout_micros=TIMEOUT_MICROS)
        if tree is None:
             # print(f"Tree-sitter parsing timed out.", file=sys.stderr)
             return []
    except Exception as e:
        # print(f"Tree-sitter parsing error: {e}", file=sys.stderr)
        return [] # 解析失败返回空

    root_node = tree.root_node

    # 动态计算 max_node_span
    file_len = len(file_content_bytes)
    max_node_span_calc = max(MIN_MAX_NODE_SPAN, int(file_len * MAX_NODE_SPAN_RATIO))
    max_node_span = min(max_node_span_calc, MAX_NODE_SPAN_ABS) # 应用绝对上限

    # Start traversal from the root
    critical_nodes = get_critical_nodes_from_tree(root_node, file_content_bytes, critical_types, MIN_NODE_SPAN, max_node_span)

    # 去重和排序
    unique_nodes = []
    seen_spans = set()
    for node in critical_nodes:
        span_key = (node['start'], node['end'])
        if span_key not in seen_spans:
            unique_nodes.append(node)
            seen_spans.add(span_key)

    # Sort by start byte, then by end byte (for nested cases)
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

    # 确定语言和关键节点类型
    language: Optional[Language] = None
    critical_types: set = set()
    lang_name = "unknown"

    # --- Log the file being processed ---
    logging.info(f"{target_file_path}")
    # ------------------------------------

    if file_ext in CPP_EXTENSIONS:
        language = CPP_LANGUAGE
        critical_types = CPP_CRITICAL_NODE_TYPES
        lang_name = "cpp"
    elif file_ext in CSHARP_EXTENSIONS:
        language = CSHARP_LANGUAGE
        critical_types = CSHARP_CRITICAL_NODE_TYPES
        lang_name = "csharp"
    else:
        return 0, 0 # 不是目标文件类型，不算失败

    if not language: # Ensure language was loaded
         # This check should ideally be redundant due to the global load, but good safety.
         print(f"Warning: Skipping {target_file_path}, language object not available.", file=sys.stderr)
         return 0, 1 # Count as failure if language is missing

    try:
        # Limit file size read to prevent memory issues with huge files
        MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024 # 10 MB limit (adjust as needed)
        file_size = await asyncio.to_thread(os.path.getsize, target_file_path)
        if file_size == 0:
            # print(f"Skipping empty file: {target_file_path}")
            return 0, 0 # Skip empty files, not an error
        if file_size > MAX_FILE_SIZE_BYTES:
             # print(f"Warning: Skipping {target_file_path}, file size ({file_size} bytes) exceeds limit ({MAX_FILE_SIZE_BYTES} bytes).", file=sys.stderr)
             return 0, 1 # Treat as failure / skip

        # Read file content (in thread)
        file_contents = await asyncio.to_thread(target_file_path.read_text, encoding='utf-8', errors='ignore')

        # Parse and extract nodes (run in thread pool as it can be CPU intensive)
        nodes = await asyncio.to_thread(extract_critical_blocks_cpp_cs, file_contents, language, critical_types)

        if not nodes:
            # Parsing might be okay, just no relevant nodes found or file was empty after ignore errors
            return 0, 0 # 0 samples, 0 failed file

        relative_path = target_file_path.relative_to(root_dir).as_posix()
        # Encode once for consistent byte slicing
        file_contents_bytes = file_contents.encode('utf-8', errors='ignore')
        file_len_bytes = len(file_contents_bytes)

        for node in nodes:
            start, end = node['start'], node['end']
            # Basic validation of node boundaries against actual byte length
            if not (0 <= start < end <= file_len_bytes):
                 # print(f"Warning: Skipping invalid node span {start}-{end} (file len: {file_len_bytes}) in {target_file_path}")
                 continue

            middle_len = end - start
            # Ensure middle_len calculation is sound
            if middle_len <= 0:
                 continue

            remaining = MAX_CHUNK_LEN - middle_len

            if remaining < 0:
                # Middle itself is too long, skip this sample
                # print(f"Warning: Node span {start}-{end} (len {middle_len}) exceeds MAX_CHUNK_LEN {MAX_CHUNK_LEN} in {target_file_path}. Skipping sample.")
                continue

            # Calculate context window size
            half_remaining = remaining // 2

            # Determine prefix and suffix boundaries, clamping to valid byte range
            prefix_start = max(0, start - half_remaining)
            # Suffix ends *at* the byte offset 'end', context starts from there
            suffix_start = end
            suffix_end = min(file_len_bytes, suffix_start + half_remaining)

            # Ensure total length constraint (adjust boundaries if needed)
            # Note: This logic prioritizes keeping 'middle' intact.
            current_prefix_len = start - prefix_start
            current_suffix_len = suffix_end - suffix_start
            current_total_len = current_prefix_len + middle_len + current_suffix_len

            if current_total_len > MAX_CHUNK_LEN:
                excess = current_total_len - MAX_CHUNK_LEN
                # Prioritize shrinking suffix, then prefix
                cut_suffix = min(excess // 2 + excess % 2, current_suffix_len)
                suffix_end -= cut_suffix
                excess -= (current_suffix_len - (suffix_end - suffix_start)) # Recalc excess based on actual cut

                if excess > 0:
                     cut_prefix = min(excess, current_prefix_len)
                     prefix_start += cut_prefix


            # Slice bytes and decode to string
            prefix_bytes = file_contents_bytes[prefix_start:start]
            middle_bytes = file_contents_bytes[start:end]
            # Suffix slice: from suffix_start up to adjusted suffix_end
            suffix_bytes = file_contents_bytes[suffix_start:suffix_end]

            # Decode with error handling
            prefix = prefix_bytes.decode('utf-8', errors='replace') # Replace errors instead of ignore?
            middle = middle_bytes.decode('utf-8', errors='replace')
            suffix = suffix_bytes.decode('utf-8', errors='replace')

            dataset_split = determine_split()
            jsonl_data = {
                'filePath': relative_path,
                'language': lang_name,
                'prefix': prefix,
                'middle': middle,
                'suffix': suffix,
                # Optional: Add byte spans for verification
                # 'span_bytes': [start, end],
                # 'context_bytes': [prefix_start, suffix_end]
            }
            # Ensure the data is JSON serializable
            try:
                 jsonl_string = json.dumps(jsonl_data, ensure_ascii=False)
            except TypeError as json_err:
                 print(f"Error: Could not serialize data to JSON for node {start}-{end} in {target_file_path}: {json_err}", file=sys.stderr)
                 print(f"Data: {jsonl_data}", file=sys.stderr)
                 continue # Skip this sample

            writer = train_set_writer if dataset_split == 'train' else test_set_writer
            # Run file writing in a thread
            await asyncio.to_thread(writer.write, jsonl_string + '\n')
            successful_samples += 1

        return successful_samples, 0 # samples, failed_flag (0 means success)

    except FileNotFoundError:
        # This specific file not found, log and mark as failed
        print(f"Error: File not found {target_file_path}", file=sys.stderr)
        return 0, 1
    except UnicodeDecodeError as ude:
        # Error reading the file content itself
        # print(f"Error: Could not decode file {target_file_path} as utf-8: {ude}", file=sys.stderr)
        return 0, 1
    except OSError as os_err: # Catch OS errors like permission denied during getsize/read
         print(f"OS Error processing file {target_file_path}: {os_err}", file=sys.stderr)
         return 0, 1
    except Exception as e:
        # Catch any other unexpected error during processing of this file
        print(f"Unexpected Error processing file {target_file_path}: {type(e).__name__} - {e}", file=sys.stderr)
        # import traceback # Uncomment for full debugging traceback
        # traceback.print_exc()
        return 0, 1 # Mark as failed file

# --- Main Execution ---
async def main(root_dir_str: str):
    """主函数"""
    root_dir = Path(root_dir_str).resolve()
    if not root_dir.is_dir():
        print(f"Error: Provided path '{root_dir_str}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    # --- Configure Logging ---
    log_file = OUTPUT_DIR / 'files.log'
    # Ensure output dir exists for the log file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s', # Only log the message itself (file path)
        filename=log_file,
        filemode='a' # Append mode
    )
    # -------------------------

    # Ensure Tree-sitter languages are loaded (already handled globally)
    # Global variables CPP_LANGUAGE and CSHARP_LANGUAGE are used directly
    if not CPP_LANGUAGE or not CSHARP_LANGUAGE:
         # This should not happen if the initial loading succeeded
        print("Error: Languages were not loaded correctly.", file=sys.stderr)
        sys.exit(1)


    # Create output directory (redundant check, already done for logging)
    # OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_file_path = OUTPUT_DIR / TRAIN_FILENAME
    test_file_path = OUTPUT_DIR / TEST_FILENAME

    # --- File Discovery ---
    print(f"Scanning for {', '.join(TARGET_EXTENSIONS)} files in {root_dir}...")
    target_files = []
    # Use os.walk for potentially faster/more robust file discovery than rglob on huge trees
    def find_files_sync():
        """Synchronous function to find files using os.walk, skipping specific directories."""
        count = 0
        # topdown=True allows us to modify dirnames to prune traversal
        for dirpath, dirnames, filenames in os.walk(root_dir, topdown=True):
             # Exclude specific directory names (case-sensitive)
             # Modify dirnames in-place to prevent descending into them
             if 'ThirdParty' in dirnames:
                 dirnames.remove('ThirdParty')

             # Also skip common hidden/build directories for efficiency
             # Note: Modifying dirnames affects further traversal; use list comprehension for current level check
             dirs_to_skip = {'.git', 'node_modules', 'bin', 'obj', 'Build', 'build', 'Intermediate', 'DerivedDataCache'}
             dirnames[:] = [d for d in dirnames if d not in dirs_to_skip and not d.startswith('.')] # Filter dirnames for next level

             # Process files in the current directory
             for filename in filenames:
                 # Check if the file has one of the target extensions
                 file_path = Path(dirpath) / filename
                 if file_path.suffix.lower() in TARGET_EXTENSIONS:
                     target_files.append(file_path)
                     count += 1
                     # Optional: Provide progress feedback during scanning
                     if count % 1000 == 0:
                         print(f"Found {count} files...", end='\\r')
        print() # Newline after scan finish

    # Run synchronous file discovery in a thread
    await asyncio.to_thread(find_files_sync)

    if not target_files:
        print("No target files found in the specified directory.", file=sys.stderr)
        sys.exit(0)

    print(f"Found {len(target_files)} target files. Starting processing...")

    total_files = len(target_files)
    total_samples = 0
    successful_files = 0
    failed_files = 0

    # --- File Processing ---
    # Use 'a' mode (append) for output files, ensure they are opened correctly
    train_set_writer = None
    test_set_writer = None
    try:
        # Open files in the main thread before starting async tasks
        train_set_writer = open(train_file_path, 'a', encoding='utf-8')
        test_set_writer = open(test_file_path, 'a', encoding='utf-8')

        # Concurrently process files
        tasks = [process_file(file_path, root_dir, train_set_writer, test_set_writer) for file_path in target_files]

        processed_count = 0
        import time
        start_time = time.time()

        # Use tqdm for progress bar if available
        results_iterable = None
        try:
             from tqdm.asyncio import tqdm_asyncio
             # Wrap asyncio.as_completed with tqdm for progress
             results_iterable = tqdm_asyncio(asyncio.as_completed(tasks), total=total_files, desc="Processing files", unit="file")
        except ImportError:
             results_iterable = asyncio.as_completed(tasks)
             print("Processing (install tqdm for progress bar: pip install tqdm)...")

        # Process results as they complete
        for future in results_iterable:
             try:
                 # Await the result of the completed task
                 samples, failed_flag = await future
                 total_samples += samples
                 if failed_flag:
                     failed_files += 1
                 else:
                     # Count as successful if no critical error occurred for this file
                     successful_files += 1
             except Exception as task_exc:
                 # Catch errors from the task runner itself (should be rare if process_file handles errors)
                 print(f"Error processing future: {task_exc}", file=sys.stderr)
                 failed_files += 1 # Count as failed if the future itself errored
             finally:
                 processed_count += 1
                 # Manual progress update if tqdm is not used
                 # if not results_iterable.__class__.__name__.startswith('tqdm') and processed_count % 100 == 0:
                 #    elapsed = time.time() - start_time
                 #    rate = processed_count / elapsed if elapsed > 0 else 0
                 #    print(f"Processed {processed_count}/{total_files} files ({rate:.2f} files/s)...", end='\r')

        # Final newline if manual progress was printed
        # if not results_iterable.__class__.__name__.startswith('tqdm'):
        #     print()


    except OSError as e:
        print(f"Error opening output files '{train_file_path}' or '{test_file_path}': {e}", file=sys.stderr)
        # Ensure cleanup happens even if opening fails
    except Exception as main_exc:
         print(f"An unexpected error occurred during main processing: {main_exc}", file=sys.stderr)
         # import traceback
         # traceback.print_exc()

    finally:
        # Ensure files are closed reliably
        if train_set_writer:
             await asyncio.to_thread(train_set_writer.close)
        if test_set_writer:
            await asyncio.to_thread(test_set_writer.close)

    end_time = time.time()
    duration = end_time - start_time

    # --- Summary Output ---
    print("\n--- Processing Summary ---")
    print(f"Total files found: {total_files}")
    # Recalculate attempted based on successful + failed counts
    files_attempted = successful_files + failed_files
    print(f"Files attempted processing: {files_attempted}")
    print(f"Files processed successfully (may include 0 samples): {successful_files}")
    print(f"Files failed during processing (read/parse errors): {failed_files}")
    skipped_files = total_files - files_attempted # Files skipped due to size limit or other pre-checks
    # print(f"Files skipped (e.g., too large): {skipped_files}") # Optional detail
    print(f"Total samples generated: {total_samples}")
    print(f"Train samples written to: {train_file_path.resolve()}")
    print(f"Test samples written to: {test_file_path.resolve()}")
    print(f"Processed file list written to: {log_file.resolve()}") # <-- Inform user about log file
    print(f"Total processing time: {duration:.2f} seconds")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <path_to_cpp_cs_project_root>")
        sys.exit(1)

    project_root_dir = sys.argv[1]
    # Set appropriate asyncio event loop policy for Windows
    if sys.platform == "win32":
        # Necessary for subprocesses and other async features on Windows
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main(project_root_dir))
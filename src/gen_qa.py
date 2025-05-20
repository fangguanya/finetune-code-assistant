# 生成对应C++&C#代码的SFT数据集
import os
import re
import argparse
import json
import time # Import the time module

# NEW: Import for libclang
try:
    from clang.cindex import Index, TranslationUnit, CursorKind, Config
    LIBCLANG_AVAILABLE = True
    # You might need to point to the libclang library file explicitly
    # Config.set_library_file('/path/to/your/libclang.so_or_dll')
except ImportError:
    LIBCLANG_AVAILABLE = False
    print("Warning: libclang not found. C++ AST parsing will be skipped. "
          "Please install libclang and the Python bindings (pip install libclang).")

# C++ 和 C# 的文件扩展名
CPP_EXTENSIONS = ['.cpp', '.h', '.hpp', '.c', '.cc']
CSHARP_EXTENSIONS = ['.cs']

# 复杂度量化相关的关键字 (可以根据需要扩展)
COMPLEXITY_KEYWORDS = [
    'if', 'else if', 'else', 'for', 'while', 'do', 'switch', 'case',
    'try', 'catch', 'finally', 'throw', 'goto',
    # C# specific
    'foreach', 'yield', 'async', 'await'
]

# NEW: Keywords for C++ cyclomatic complexity (approximated)
CPP_CYCLOMATIC_KEYWORDS = [
    CursorKind.IF_STMT, CursorKind.FOR_STMT, CursorKind.WHILE_STMT, CursorKind.DO_STMT,
    CursorKind.CASE_STMT, CursorKind.DEFAULT_STMT, CursorKind.CXX_TRY_STMT, # try block itself
    # Conditional operator (?:) is harder to catch directly without deeper expr analysis
    # Logical AND (&&) and OR (||) in conditions also add to complexity
]

def find_source_files(directory):
    """扫描指定目录下的C++和C#源文件"""
    source_files = {'cpp': [], 'csharp': []}
    for root, _, files in os.walk(directory):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            file_path = os.path.join(root, file)
            if ext in CPP_EXTENSIONS:
                source_files['cpp'].append(file_path)
            elif ext in CSHARP_EXTENSIONS:
                source_files['csharp'].append(file_path)
    return source_files

def calculate_complexity(code_block, language="unknown", ast_node=None):
    """
    计算代码块的复杂度指标。
    如果提供了AST节点 (主要用于C++)，则可以计算更精确的指标。
    """
    lines = code_block.strip().split('\n')
    lines_of_code = len(lines)
    
    csharp_keyword_count = 0 # For C# or C++ fallback
    cyclomatic_complexity = 1 # Default for a straight path of execution

    num_variable_declarations_in_scope = 0
    num_comment_blocks = 0
    total_comment_characters = 0
    complexity_score = 0 # Initialize

    if language == "cpp" and ast_node and LIBCLANG_AVAILABLE:
        # Calculate cyclomatic complexity using AST
        current_cyclomatic_complexity = 1 # Renamed to avoid conflict with outer scope variable
        def find_complexity_points(cursor):
            nonlocal current_cyclomatic_complexity
            try:
                kind = cursor.kind # Access kind here to catch potential ValueError early
                if kind in CPP_CYCLOMATIC_KEYWORDS:
                    current_cyclomatic_complexity += 1
                
                if kind in [CursorKind.IF_STMT, CursorKind.WHILE_STMT, CursorKind.FOR_STMT]:
                    for child in cursor.get_children():
                        tokens = [token.spelling for token in child.get_tokens()]
                        current_cyclomatic_complexity += tokens.count('&&')
                        current_cyclomatic_complexity += tokens.count('||')
                        break
            except ValueError as e:
                # If kind is unknown, libclang might raise ValueError. Silently skip this node for complexity.
                # A more advanced solution might log this if verbose mode is on.
                # print(f"Warning: Encountered an unknown CursorKind for node '{cursor.spelling}'. Details: {e}")
                pass # Silently ignore for now to allow script to continue

            for child in cursor.get_children():
                find_complexity_points(child)
        find_complexity_points(ast_node)
        cyclomatic_complexity = current_cyclomatic_complexity # Assign to outer scope var
        
        # --- Count local variable declarations --- 
        if ast_node.kind in [CursorKind.FUNCTION_DECL, CursorKind.CXX_METHOD, CursorKind.CONSTRUCTOR, CursorKind.DESTRUCTOR]:
            for child_cursor in ast_node.get_children():
                if child_cursor.kind == CursorKind.COMPOUND_STMT:
                    body_stack = list(child_cursor.get_children())
                    visited_in_body = set()
                    while body_stack:
                        stmt_node = body_stack.pop()
                        if stmt_node.hash in visited_in_body: continue
                        visited_in_body.add(stmt_node.hash)
                        if stmt_node.kind == CursorKind.VAR_DECL:
                            num_variable_declarations_in_scope += 1
                        if stmt_node.kind == CursorKind.COMPOUND_STMT:
                            body_stack.extend(list(stmt_node.get_children()))
                    break
        
        # --- Count comments using regex on the code_block for C++ ---
        if code_block:
            comment_matches = re.findall(r"//.*|/\*[\s\S]*?\*/", code_block) # Corrected regex for //
            num_comment_blocks = len(comment_matches)
            for comment_text in comment_matches:
                total_comment_characters += len(comment_text)

        complexity_score = (
            lines_of_code * 0.15 +
            cyclomatic_complexity * 1.5 +
            num_variable_declarations_in_scope * 0.4 +
            num_comment_blocks * 0.25 +
            total_comment_characters * 0.005
        )
        csharp_keyword_count = 0 # Ensure this is 0 for C++ AST path as it's not relevant

    else: # Fallback for C# or if C++ AST is not available
        code_lower = code_block.lower()
        for keyword in COMPLEXITY_KEYWORDS:
            csharp_keyword_count += len(re.findall(r'\b' + re.escape(keyword) + r'\b', code_lower))
        
        # Approximate cyclomatic for C# using keyword count for the cyclomatic_complexity variable
        cyclomatic_complexity = csharp_keyword_count + 1 

        if code_block:
            comment_matches = re.findall(r"//.*|/\*[\s\S]*?\*/", code_block) # Corrected regex for //
            num_comment_blocks = len(comment_matches)
            for comment_text in comment_matches:
                total_comment_characters += len(comment_text)

        complexity_score = (
            lines_of_code * 0.4 +
            csharp_keyword_count * 1.0 +
            num_comment_blocks * 0.25 +
            total_comment_characters * 0.005
        )
        # num_variable_declarations_in_scope remains 0 (or N/A in output) for C#
    
    return {
        "lines_of_code": lines_of_code,
        "control_flow_statements": csharp_keyword_count,
        "cyclomatic_complexity": f"{cyclomatic_complexity:.0f}" if (language == "cpp" and ast_node and LIBCLANG_AVAILABLE) else f"Approx: {cyclomatic_complexity:.0f}",
        "num_variable_declarations": num_variable_declarations_in_scope if (language == "cpp" and ast_node and LIBCLANG_AVAILABLE) else "N/A",
        "num_comment_blocks": num_comment_blocks,
        "total_comment_characters": total_comment_characters,
        "calculated_complexity_score": complexity_score
    }

def extract_cpp_elements(file_path, compiler_args=None):
    """
    从C++文件中使用 libclang提取函数/方法和类/结构体。
    """
    if not LIBCLANG_AVAILABLE:
        print(f"Skipping C++ AST parsing for {file_path} as libclang is not available.")
        # Fallback to regex or return empty (for this example, return empty)
        return []

    elements = []
    if compiler_args is None:
        compiler_args = ['-std=c++20'] # Default, can be overridden

    try:
        index = Index.create()
        # TU_SKIP_FUNCTION_BODIES can speed up parsing if we only need declarations,
        # but we need bodies for complexity and full code.
        # TU_DETAILED_PREPROCESSING_RECORD for macro expansions if needed (adds overhead).
        tu = index.parse(file_path, args=compiler_args, 
                         options=0) # Use 0 for default parsing (includes function bodies)

        if not tu:
            print(f"Error: Unable to parse C++ file {file_path} with libclang.")
            return elements
        
        # Check for parsing errors
        has_errors = False
        for diag in tu.diagnostics:
            if diag.severity >= diag.Error: # Error or Fatal
                # print(f"Clang Error/Warning in {file_path}: {diag.spelling} at L{diag.location.line}:C{diag.location.column}")
                has_errors = True # We can still try to extract but be aware
        # if has_errors:
            # print(f"Note: {file_path} has parsing errors/warnings, AST-based extraction might be incomplete.")


        # Helper to get raw code block from extent
        def get_code_from_extent(extent):
            try:
                with open(extent.start.file.name, 'r', encoding='utf-8', errors='ignore') as f_content:
                    file_content_lines = f_content.readlines()
                
                start_line = extent.start.line -1
                end_line = extent.end.line -1
                start_col = extent.start.column -1
                end_col = extent.end.column -1

                if start_line == end_line:
                    return file_content_lines[start_line][start_col:end_col]
                
                block = [file_content_lines[start_line][start_col:]]
                for i in range(start_line + 1, end_line):
                    block.append(file_content_lines[i])
                if end_line > start_line : # ensure end_line is valid before access
                    block.append(file_content_lines[end_line][:end_col])
                return "".join(block)
            except Exception as e:
                # print(f"Could not extract code for node: {e}")
                return "/* Could not extract code */"


        for cursor in tu.cursor.get_children():
            # We are interested in elements defined in the main file, not in includes
            if cursor.location.file and cursor.location.file.name == file_path:
                element_type = None
                name = cursor.spelling
                code_block = ""
                ast_node_ref = cursor # Keep a reference to the AST node

                if cursor.kind in [CursorKind.FUNCTION_DECL, CursorKind.CXX_METHOD]:
                    # Ensure it's a definition, not just a declaration
                    if cursor.is_definition():
                        element_type = "function/method"
                        # For functions/methods, extent usually covers the whole definition
                        code_block = get_code_from_extent(cursor.extent)
                
                elif cursor.kind in [CursorKind.CLASS_DECL, CursorKind.STRUCT_DECL]:
                     if cursor.is_definition():
                        element_type = "class" if cursor.kind == CursorKind.CLASS_DECL else "struct"
                        code_block = get_code_from_extent(cursor.extent)
                
                # Add other kinds if needed (e.g., ENUM_DECL, VAR_DECL for global vars)

                if element_type and name and code_block.strip():
                    elements.append({
                        "file_path": file_path,
                        "type": element_type,
                        "name": name,
                        "code_block": code_block.strip(),
                        "language": "cpp",
                        "ast_node": ast_node_ref # Store AST node for later use (e.g. complexity)
                    })
                    # Recursively process children of classes/structs/namespaces if needed
                    # For now, this top-level extraction should get primary functions and classes.
                    # If you need methods inside classes, you'd iterate cursor.get_children() for class cursors.
                    if element_type in ["class", "struct"]:
                        for child_cursor in cursor.get_children():
                            if child_cursor.location.file and child_cursor.location.file.name == file_path:
                                child_element_type = None
                                child_name = child_cursor.spelling
                                child_code_block = ""
                                child_ast_node_ref = child_cursor

                                if child_cursor.kind == CursorKind.CXX_METHOD and child_cursor.is_definition():
                                    child_element_type = "method"
                                    child_code_block = get_code_from_extent(child_cursor.extent)
                                elif child_cursor.kind == CursorKind.CONSTRUCTOR and child_cursor.is_definition():
                                    child_element_type = "constructor"
                                    child_code_block = get_code_from_extent(child_cursor.extent)
                                elif child_cursor.kind == CursorKind.DESTRUCTOR and child_cursor.is_definition():
                                    child_element_type = "destructor"
                                    child_code_block = get_code_from_extent(child_cursor.extent)
                                # Could add field decls, nested classes, etc.
                                
                                if child_element_type and child_name and child_code_block.strip():
                                    elements.append({
                                        "file_path": file_path,
                                        "type": child_element_type,
                                        "name": f"{name}::{child_name}", # Qualified name
                                        "code_block": child_code_block.strip(),
                                        "language": "cpp",
                                        "ast_node": child_ast_node_ref
                                    })


    except Exception as e:
        print(f"Error processing C++ file {file_path} with libclang: {e}")
    
    return elements

def extract_csharp_elements(file_path):
    """
    从C#文件中提取方法和类/结构体/接口。
    这是一个简化的提取器，可能无法处理所有C#复杂语法。
    """
    elements = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading C# file {file_path}: {e}")
        return elements

    # 匹配方法 (简化版, 考虑 public/private/protected/internal, static, async, return type, name, params, body)
    # This regex is simplified and might need refinement for all C# features (e.g., generic constraints, expression-bodied members).
    # ((?:public|private|protected|internal|static|async|virtual|override|sealed|unsafe)\\s+)* --- Modifiers
    # ([\\w<>\\|\\[\\]\\|,\\?\\s]+?) --- Return type (can include generics, arrays, nullable)
    # \\s+ --- Separator
    # ([\\w<>]+) --- Method name (can include generics)
    # \\s* --- Optional whitespace
    # \\(([^)]*)\\) --- Parameters
    # \\s* --- Optional whitespace
    # (?:where\\s+[\\w\\s:,<>()]+)? --- Optional generic constraints
    # \\s* --- Optional whitespace
    # \\{((?:[^{}]|\\{[^{}]*\\})*)\\} --- Body (simple brace matching)
    method_pattern = re.compile(
        r'^\\s*' # Optional leading whitespace
        r'((?:(?:public|private|protected|internal|static|async|virtual|override|sealed|unsafe)\\s+)*)?' # Modifiers
        r'([\\w<>\\|\\[\\]\\|,\\?\\s]+?)' # Return type
        r'\\s+'
        r'([\\w<>]+)' # Method name
        r'\\s*'
        r'\\(([^)]*)\\)' # Parameters
        r'\\s*(?:where\\s+[\\w\\s:,<>()]+)?' # Optional generic constraints
        r'\\s*\\{((?:[^{}]|\\{[^{}]*\\})*)\\}' # Body
        , re.MULTILINE
    )
    for match in method_pattern.finditer(content):
         # Avoid capturing control structures or other blocks if the "return type" looks like a keyword
        if match.group(2).strip() in ['if', 'for', 'while', 'switch', 'try', 'catch', 'lock', 'using']:
            continue
        code_block = match.group(0)
        elements.append({
            "file_path": file_path,
            "type": "method",
            "name": match.group(3),
            "code_block": code_block,
            "language": "csharp"
        })

    # 匹配类/结构体/接口 (简化版)
    # ((?:public|private|protected|internal|static|abstract|sealed)\\s+)* --- Modifiers
    # (class|struct|interface) --- Type
    # \\s+ --- Separator
    # ([\\w<>]+) --- Name (can include generics)
    # \\s*(?:[:\\s\\w<>,]+)? --- Optional inheritance/implementation
    # \\s* --- Optional whitespace
    # (?:where\\s+[\\w\\s:,<>()]+)? --- Optional generic constraints
    # \\s* --- Optional whitespace
    # \\{((?:[^{}]|\\{[^{}]*\\})*)\\} --- Body
    class_struct_interface_pattern = re.compile(
        r'^\\s*' # Optional leading whitespace
        r'((?:(?:public|private|protected|internal|static|abstract|sealed)\\s+)*)?' # Modifiers
        r'(class|struct|interface)\\s+([\\w<>]+)' # Type and Name
        r'\\s*(?:[:\\s\\w<>,]+)?' # Optional inheritance / implementation
        r'\\s*(?:where\\s+[\\w\\s:,<>()]+)?' # Optional generic constraints
        r'\\s*\\{((?:[^{}]|\\{[^{}]*\\})*)\\}' # Body
        , re.MULTILINE
    )
    for match in class_struct_interface_pattern.finditer(content):
        code_block = match.group(0)
        elements.append({
            "file_path": file_path,
            "type": match.group(2),
            "name": match.group(3),
            "code_block": code_block,
            "language": "csharp"
        })
        
    return elements

def generate_sft_data(elements, min_complexity_score):
    """根据复杂度阈值生成SFT数据"""
    sft_records = []
    for element in elements:
        # Pass the ast_node if available (for C++)
        ast_node = element.get("ast_node") 
        metrics = calculate_complexity(element['code_block'], element['language'], ast_node=ast_node)
        
        # Clean up ast_node from the element dict before serializing to JSON, as it's not serializable
        element_for_output = element.copy()
        if "ast_node" in element_for_output:
            del element_for_output["ast_node"]

        if metrics['calculated_complexity_score'] >= min_complexity_score:
            # instruction = (
            #     f"请详细分析以下 {element['language']} 代码 ({element['type']} '{element['name']}') 的功能、核心逻辑、实现方式，"
            #     f"并评估其可读性、可维护性和潜在的改进点。\\n"
            #     f"代码来源: {element['file_path']}"
            # )
            # output_data = {
            #     "file_path": element['file_path'],
            #     "element_type": element['type'],
            #     "element_name": element['name'],
            #     "language": element['language'],
            #     "code_block": element['code_block'],
            #     "metrics": metrics,
            #     "analysis_prompt": (
            #         "1. **功能概述**: (请描述该代码块的主要功能和用途)\\n"
            #         "2. **核心逻辑分析**: (请逐步解释代码块的关键步骤和业务逻辑)\\n"
            #         "3. **实现方式评估**: (请评论其实现方式的优点和缺点)\\n"
            #         "4. **可读性与可维护性**: (请评估代码的可读性和可维护性，例如命名、注释、结构等)\\n"
            #         "5. **复杂度评估**: (根据指标和实际代码，给出综合的复杂度判断)\\n"
            #         "   - 代码行数: " + str(metrics['lines_of_code']) + "\\n"
            #         "   - 控制流语句数 (基于关键字): " + str(metrics['control_flow_statements']) + "\\n"
            #         "   - 圈复杂度 (C++/AST): " + str(metrics.get('cyclomatic_complexity', 'N/A')) + "\\n"
            #         "   - 计算复杂度得分: " + f"{metrics['calculated_complexity_score']:.2f}" + "\\n"
            #         "6. **潜在改进点**: (请提出具体的改进建议，例如重构、优化、增加测试等)\\n"
            #         "7. **上下文依赖**: (此代码块是否严重依赖外部模块、类或特定数据结构？请简要说明)"
            #     )
            # }
            # sft_records.append({
            #     "instruction": instruction,
            #     "output": json.dumps(output_data, ensure_ascii=False, indent=2) # Store output as a JSON string
            # })
            sft_records.append({
                "file_path": element['file_path'],
                "type": element['type'],
                "name": element['name'],
                "language": element['language'],
                "code_block": element['code_block'],
                "metrics": metrics,
                
            })
    return sft_records

def main():
    parser = argparse.ArgumentParser(description="Scan C++/C# source files, analyze complexity, and generate SFT data.")
    parser.add_argument("source_directory", type=str, help="The root directory of the source code to scan.")
    parser.add_argument("-o", "--output_file", type=str, default="sft_dataset.json", help="The output file for SFT data (JSON Lines format).")
    parser.add_argument("-m", "--min_complexity", type=float, default=5.0, help="Minimum complexity score for a code element to be included in the SFT dataset.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output.")
    parser.add_argument("--cpp_compiler_args", type=str, help="Comma-separated compiler arguments for C++ parsing (e.g., '-std=c++17,-Iinclude').")

    args = parser.parse_args()

    if not os.path.isdir(args.source_directory):
        print(f"Error: Source directory '{args.source_directory}' not found.")
        return

    if args.verbose:
        print(f"Scanning directory: {args.source_directory}")
        print(f"Minimum complexity score: {args.min_complexity}")
        print(f"Output file: {args.output_file}")

    source_files = find_source_files(args.source_directory)
    
    if args.verbose:
        print(f"Found {len(source_files['cpp'])} C++ files and {len(source_files['csharp'])} C# files.")

    if args.cpp_compiler_args:
        cpp_args = args.cpp_compiler_args.split(',')
        if args.verbose:
            print(f"Using C++ compiler arguments: {cpp_args}")
    else:
        cpp_args = ['-x', 'c++', '-Xclang', '-ast-dump', '-fsyntax-only', '-std=c++20']

    # --- Progress Bar Logic with ETA ---
    total_files_to_process = len(source_files['cpp']) + len(source_files['csharp'])
    processed_files_count = 0
    start_time = time.time() # Record start time

    def format_time(seconds):
        """Formats seconds into HH:MM:SS or MM:SS string."""
        if seconds < 0: seconds = 0
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"

    def print_progress(processed, total, file_name="", eta_seconds=None):
        percentage = (processed / total) * 100 if total > 0 else 0
        bar_length = 30 # Reduced bar length to make space for ETA
        filled_length = int(bar_length * processed // total) if total > 0 else 0
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        eta_str = ""
        if eta_seconds is not None and processed > 0 and processed < total:
            eta_str = f" ETA: {format_time(eta_seconds)}"
        elif processed == total:
            eta_str = f" Took: {format_time(time.time() - start_time)}"

        status_line = f'Progress: |{bar}| {percentage:.2f}% ({processed}/{total}){eta_str} Processing: {file_name[:40]:<40}'
        print(f'\r{status_line}', end='')
        if processed == total:
            print() # Newline at the end

    if total_files_to_process == 0:
        if args.verbose:
            print("No source files found to process.")
        # Create an empty JSON array if no files
        try:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                f.write("[]")
            print(f"SFT dataset successfully saved to {args.output_file} (empty as no files were processed).")
        except IOError as e:
            print(f"Error writing empty output file {args.output_file}: {e}")
        return # Exit early if no files
    # --- End Progress Bar Logic ---

    # --- Streaming JSON Output Logic ---
    first_record_written_to_file = False
    try:
        with open(args.output_file, 'w', encoding='utf-8') as outfile:
            outfile.write("[") # Start of the JSON array

            for lang, files in source_files.items():
                extractor = None
                if lang == 'cpp':
                    if LIBCLANG_AVAILABLE:
                        if args.verbose:
                            print("Using libclang (AST-based) for C++ files.")
                        extractor = lambda f: extract_cpp_elements(f, compiler_args=cpp_args)
                    else:
                        if args.verbose:
                            print("Warning: libclang not found, C++ processing will be skipped.")
                        # Increment count for skipped files for progress bar accuracy
                        processed_files_count += len(files) 
                        # Calculate ETA even when skipping
                        elapsed_time = time.time() - start_time
                        avg_time_per_file = elapsed_time / processed_files_count if processed_files_count > 0 else 0
                        eta = (total_files_to_process - processed_files_count) * avg_time_per_file if avg_time_per_file > 0 else None
                        print_progress(processed_files_count, total_files_to_process, "Skipping C++ files", eta)
                        continue 
                elif lang == 'csharp':
                    if args.verbose:
                        print("Using regex-based extraction for C# files.")
                    extractor = extract_csharp_elements
                
                if not extractor:
                    # Should not happen if lang is cpp or csharp, but as a safeguard
                    processed_files_count += len(files)
                    elapsed_time = time.time() - start_time
                    avg_time_per_file = elapsed_time / processed_files_count if processed_files_count > 0 else 0
                    eta = (total_files_to_process - processed_files_count) * avg_time_per_file if avg_time_per_file > 0 else None
                    print_progress(processed_files_count, total_files_to_process, f"Skipping {lang} files (no extractor)", eta)
                    continue

                for file_path in files:
                    current_file_name = os.path.basename(file_path)
                    
                    # Calculate ETA before printing progress
                    eta_value = None
                    if processed_files_count > 0: # Avoid division by zero and calculate ETA after at least one file
                        elapsed_time = time.time() - start_time
                        avg_time_per_file = elapsed_time / processed_files_count
                        remaining_files = total_files_to_process - processed_files_count
                        eta_value = remaining_files * avg_time_per_file

                    if args.verbose:
                        print(f"\nProcessing {lang.upper()} file: {file_path}") 
                        print_progress(processed_files_count, total_files_to_process, current_file_name, eta_value)
                    else:
                        print_progress(processed_files_count, total_files_to_process, current_file_name, eta_value)
                    
                    file_elements = extractor(file_path)
                    
                    if file_elements:
                        # Generate SFT data only for elements from the current file
                        sft_data_for_file = generate_sft_data(file_elements, args.min_complexity)
                        
                        for record in sft_data_for_file:
                            if first_record_written_to_file:
                                outfile.write(",\n") # Add comma and newline for readability
                            else:
                                outfile.write("\n") # Add newline for first record for readability
                            
                            json.dump(record, outfile, ensure_ascii=False, indent=2 if args.verbose else None)
                            first_record_written_to_file = True
                    
                    processed_files_count += 1
                    
                    # Recalculate ETA after processing for more accuracy if needed, or use previous for display
                    # For simplicity, we use the ETA calculated before processing for the current line update.
                    # More sophisticated ETA could re-calculate here.
                    final_eta_value = None
                    if processed_files_count > 0:
                        elapsed_time_after = time.time() - start_time
                        avg_time_per_file_after = elapsed_time_after / processed_files_count
                        remaining_files_after = total_files_to_process - processed_files_count
                        final_eta_value = remaining_files_after * avg_time_per_file_after

                    if args.verbose:
                        print_progress(processed_files_count, total_files_to_process, current_file_name, final_eta_value) 
                    else:
                        print_progress(processed_files_count, total_files_to_process, current_file_name, final_eta_value)

            if first_record_written_to_file: # if any record was written, add a final newline before closing bracket
                 outfile.write("\n")
            outfile.write("]") # End of the JSON array
        
        print(f"\nSFT dataset successfully saved to {args.output_file}")

    except IOError as e:
        print(f"\nError during file processing or writing output file {args.output_file}: {e}")
    # --- End Streaming JSON Output Logic ---

    # Removed old block processing and writing logic
    # all_elements = [] ...
    # sft_data = generate_sft_data(all_elements, args.min_complexity) ...
    # try: with open(args.output_file, 'w', encoding='utf-8') as f: json.dump(sft_data, f) ...

if __name__ == "__main__":
    main()



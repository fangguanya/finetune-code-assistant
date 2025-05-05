import asyncio
import json
import os
import random
import sys
import re # Needed for comment cleaning
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Set, TextIO

# Tree-sitter imports
try:
    from tree_sitter import Language, Parser, Node, Tree
except ImportError:
    print("Fatal: 'tree-sitter' library not found.", file=sys.stderr)
    print("Please install it using: pip install tree-sitter", file=sys.stderr)
    sys.exit(1)

# --- 配置常量 ---
CPP_EXTENSIONS = {'.cpp', '.h', '.hpp', '.cc', '.cxx', '.hxx'}
CSHARP_EXTENSIONS = {'.cs'}
TARGET_EXTENSIONS = CPP_EXTENSIONS.union(CSHARP_EXTENSIONS)

# 输出配置
DEFAULT_OUTPUT_DIR = Path("./unrealdoc_output") # 默认输出目录
FILES_LOG_FILENAME = "processed_files.log" # Log processed files
IGNORES_LOG_FILENAME = "skipped_dirs.log" # Log skipped directories
ERROR_LOG_FILENAME = "errors.log" # Log errors and warnings
DOC_SUFFIX = ".dox.md" # 文档文件的后缀

# --- Tree-sitter Language Loading (与 gen.py 类似) ---
CPP_LANGUAGE: Optional[Language] = None
CSHARP_LANGUAGE: Optional[Language] = None
LIB_EXTENSION = '.dll' if sys.platform == 'win32' else '.so'
BUILD_LIB_PATH = Path('build') / f'languages{LIB_EXTENSION}'
VENDOR_DIR = Path('vendor')

def load_or_build_languages():
    """Loads or builds the Tree-sitter language libraries. (Copied/adapted from gen.py)"""
    global CPP_LANGUAGE, CSHARP_LANGUAGE

    cpp_grammar_path = VENDOR_DIR / 'tree-sitter-cpp'
    csharp_grammar_path = VENDOR_DIR / 'tree-sitter-c-sharp'

    BUILD_LIB_PATH.parent.mkdir(exist_ok=True)
    needs_build = not BUILD_LIB_PATH.exists()
    if not needs_build:
        try:
            lib_mtime = BUILD_LIB_PATH.stat().st_mtime
            cpp_mtime = cpp_grammar_path.stat().st_mtime if cpp_grammar_path.exists() else -1
            cs_mtime = csharp_grammar_path.stat().st_mtime if csharp_grammar_path.exists() else -1
            if cpp_mtime > lib_mtime or cs_mtime > lib_mtime:
                needs_build = True
        except FileNotFoundError: needs_build = True
        except Exception as e: logging.warning(f"Could not check modification times ({e}).", exc_info=True)

    if needs_build:
        logging.info("Attempting to build Tree-sitter languages (requires py-tree-sitter < 0.22)...")
        clone_commands = []
        if not cpp_grammar_path.is_dir():
            logging.warning(f"Dir not found: {cpp_grammar_path}. Will try to clone.")
            VENDOR_DIR.mkdir(parents=True, exist_ok=True)
            clone_commands.append(f"git clone https://github.com/tree-sitter/tree-sitter-cpp {cpp_grammar_path}")
        if not csharp_grammar_path.is_dir():
            logging.warning(f"Dir not found: {csharp_grammar_path}. Will try to clone.")
            VENDOR_DIR.mkdir(parents=True, exist_ok=True)
            clone_commands.append(f"git clone https://github.com/tree-sitter/tree-sitter-c-sharp {csharp_grammar_path}")

        if clone_commands:
             logging.error("Required grammar repositories missing.")
             logging.error("Please run:")
             for cmd in clone_commands: logging.error(f"  {cmd}")
             logging.error("Then re-run.")
             sys.exit(1)

        try:
            if not hasattr(Language, 'build_library'):
                 raise AttributeError("Language.build_library not found. Need py-tree-sitter < 0.22")
            Language.build_library(str(BUILD_LIB_PATH), [str(cpp_grammar_path), str(csharp_grammar_path)])
            logging.info(f"Languages built successfully to '{BUILD_LIB_PATH}'.")
        except AttributeError as attr_err:
            logging.error(f"{attr_err}")
            logging.error("Automatic building failed. Need Language.build_library.")
            logging.error("Install older lib (pip install \"py-tree-sitter<0.22\")")
            logging.error(f"OR manually compile and place lib at: {BUILD_LIB_PATH}")
            sys.exit(1)
        except Exception as build_e:
            logging.error(f"Failed to build Tree-sitter languages: {build_e}", exc_info=True)
            logging.error("Ensure C/C++ compiler installed & in PATH.")
            logging.error(f"Verify grammar repos: {cpp_grammar_path.resolve()}, {csharp_grammar_path.resolve()}")
            sys.exit(1)

    try:
        CPP_LANGUAGE = Language(str(BUILD_LIB_PATH), 'cpp')
        try: CSHARP_LANGUAGE = Language(str(BUILD_LIB_PATH), 'c_sharp')
        except ValueError:
             logging.warning("Could not load C# language with symbol 'c_sharp', trying 'csharp'...")
             CSHARP_LANGUAGE = Language(str(BUILD_LIB_PATH), 'csharp')
        logging.info(f"Tree-sitter languages loaded successfully from {BUILD_LIB_PATH}.")
    except Exception as load_e:
        logging.error(f"Failed to load Tree-sitter languages from '{BUILD_LIB_PATH}': {load_e}", exc_info=True)
        logging.error(f"Ensure file exists and is valid: {BUILD_LIB_PATH.resolve()}")
        sys.exit(1)

# --- Tree-sitter 解析和节点提取 ---

def get_text(node: Node, content_bytes: bytes) -> str:
    """Safely extract text from a node."""
    if node is None or node.start_byte is None or node.end_byte is None:
        return ""
    # 使用 'replace' 来处理潜在的解码错误，尽管前面有检查，这里再加一层保险
    return content_bytes[node.start_byte:node.end_byte].decode('utf-8', errors='replace')

def clean_comment(raw_comment: str) -> str:
    """Cleans raw comment text, removing comment markers and leading/trailing whitespace."""
    lines = raw_comment.strip().splitlines()
    cleaned_lines = []
    in_multi_line = False

    if not lines:
        return ""

    # Detect and strip /* */ style
    if lines[0].strip().startswith("/*"):
        lines[0] = lines[0].split("/*", 1)[-1]
        # Handle single-line block comment like /* comment */
        if lines[0].strip().endswith("*/"):
            lines[0] = lines[0].rsplit("*/", 1)[0]
            in_multi_line = False # It was a single line block
        elif len(lines) > 1 and lines[-1].strip().endswith("*/"):
             lines[-1] = lines[-1].rsplit("*/", 1)[0]
             in_multi_line = True
        elif len(lines) == 1: # Single line block wasn't closed properly?
             in_multi_line = False
        else:
             in_multi_line = True # Assume leading '*' might exist

    for line in lines:
        line = line.strip()
        # Strip // style
        if line.startswith("//"):
            line = line[2:].strip()
        # Strip leading * commonly found in /* */ blocks, only if it looks like a block comment
        elif in_multi_line and line.startswith("*"):
            line = line[1:].strip()

        # Append the cleaned line if it's not empty or if it's an intentional blank line within the comment
        # Keep blank lines for potential paragraph separation
        cleaned_lines.append(line)


    # Remove leading/trailing empty lines that result from cleaning markers
    start_index = 0
    while start_index < len(cleaned_lines) and not cleaned_lines[start_index].strip():
        start_index += 1

    end_index = len(cleaned_lines) - 1
    while end_index >= start_index and not cleaned_lines[end_index].strip():
        end_index -= 1

    return "\n".join(cleaned_lines[start_index:end_index+1])

def find_leading_comment(node: Optional[Node], content_bytes: bytes) -> Optional[str]:
    """Finds the immediately preceding comment for a given node.
    Uses previous_sibling iteration for compatibility with older tree-sitter versions.
    """
    if not node:
        return None

    comment_text: Optional[str] = None
    max_lines_gap = 2 # Maximum blank lines allowed between comment and node
    target_start_line = node.start_point[0]

    current_sibling = node.previous_sibling
    while current_sibling:
        # Calculate the line difference between the node's start and the sibling's end
        line_diff = target_start_line - current_sibling.end_point[0]

        # Stop searching if the gap becomes too large
        if line_diff > max_lines_gap:
            logging.debug(f"Stopping comment search for node at line {target_start_line + 1}: previous sibling at line {current_sibling.end_point[0] + 1} is too far (gap > {max_lines_gap}).")
            break

        # Check if this sibling is the comment we are looking for
        if current_sibling.is_named and current_sibling.type == 'comment':
            raw_comment = get_text(current_sibling, content_bytes)
            comment_text = clean_comment(raw_comment)
            logging.debug(f"Found comment for node at line {target_start_line + 1} ending at line {current_sibling.end_point[0] + 1}.")
            # Found the closest preceding comment, stop searching
            break
        # If the sibling is named but not a comment, stop searching
        # Assumes comments must be directly before the element, not separated by other code elements.
        elif current_sibling.is_named:
             logging.debug(f"Stopping comment search for node at line {target_start_line + 1}: encountered non-comment named node {current_sibling.type} at line {current_sibling.start_point[0] + 1}.")
             break

        # Move to the previous sibling
        current_sibling = current_sibling.previous_sibling

    # If loop finished without finding comment, comment_text remains None
    return comment_text

def find_child_by_type(node: Optional[Node], type_name: str) -> Optional[Node]:
    """Find the first direct child node of a specific type."""
    if not node: return None
    for child in node.children:
        if child.type == type_name:
            return child
    return None

def find_descendant_by_type(node: Optional[Node], type_name: str) -> Optional[Node]:
    """Find the first descendant node of a specific type."""
    if not node: return None
    queue = list(node.children)
    while queue:
        current = queue.pop(0)
        if current.type == type_name:
            return current
        queue.extend(current.children)
    return None

def extract_documentation_elements_cpp(tree: Tree, content_bytes: bytes) -> List[Dict[str, Any]]:
    """Extracts elements and their preceding comments from a C++ tree."""
    elements = []
    root_node = tree.root_node

    # Adjusted query to capture declaration/definition nodes more directly
    query_str = """
    (function_definition) @function
    (declaration type: (_) declarator: (function_declarator)) @function_declaration ; Free function declarations like `void func();`
    (class_specifier) @class
    (struct_specifier) @struct
    (enum_specifier) @enum
    ; Capture declarations that are likely variables (simplified: has type and init_declarator/identifier)
    ; Exclude function parameters and template parameters by checking parent context implicitly/heuristically
    (declaration
      type: (_) @type_node
      declarator: [(identifier) @id_node (init_declarator declarator: (identifier) @id_node)] @decl_node
      (#not-match? @decl_node "(parameter_declaration|template_parameter_list|template_declaration)")) @variable_declaration
    (namespace_definition) @namespace
    """
    # Note: The @variable_declaration query is complex and might need refinement based on real-world code.
    # It tries to capture simple variable declarations but exclude things like function parameters.
    try:
        query = CPP_LANGUAGE.query(query_str)
        captures = query.captures(root_node)
    except Exception as e:
        logging.error(f"Error creating or executing C++ tree query: {e}", exc_info=True)
        return elements

    processed_nodes = set() # Avoid duplicates if node matches multiple patterns

    for node, capture_name in captures:
        # Use the main captured node (e.g., function_definition, class_specifier)
        element_node = node
        if element_node in processed_nodes: continue

        start_line = element_node.start_point[0] + 1
        element = {'type': capture_name, 'node': element_node, 'start_line': start_line, 'name': '?', 'details': {}, 'comment': None}

        try:
            # Attempt to find the most relevant name and refine type
            name_node = None
            element_type = capture_name # Initial type

            if capture_name == 'function' or capture_name == 'function_declaration':
                element_type = 'function' # Normalize
                declarator = find_child_by_type(element_node, 'function_declarator')
                if declarator:
                    # Try qualified identifier first (e.g., MyClass::MyFunc)
                    name_node = find_descendant_by_type(declarator, 'qualified_identifier')
                    # Fallback to simple identifier or operator name etc.
                    if not name_node: name_node = find_child_by_type(declarator, 'identifier')
                    if not name_node: name_node = find_child_by_type(declarator, 'operator_name')
                    if not name_node: name_node = find_child_by_type(declarator, 'destructor_name') # Check within declarator too
                    if name_node: element['name'] = get_text(name_node, content_bytes)
                    element['signature'] = get_text(declarator, content_bytes)
                    # Extract return type if possible (simplified)
                    type_node = find_child_by_type(element_node, 'type_identifier') or \
                                find_child_by_type(element_node, 'primitive_type') or \
                                element_node.child_by_field_name('type') # Some grammars use 'type' field
                    if type_node: element['return_type'] = get_text(type_node, content_bytes)
                # Handle constructors separately if needed (often lack explicit return type node)
                elif find_descendant_by_type(element_node, 'destructor_name'):
                    element_type = 'destructor'
                    name_node = find_descendant_by_type(element_node, 'destructor_name')
                    if name_node: element['name'] = get_text(name_node, content_bytes)
                else: # Could be constructor (often name matches class), operator overload etc.
                    # Try finding identifier if declarator logic failed (e.g. constructor)
                    id_node = find_child_by_type(element_node, 'identifier')
                    # This might be too general, needs context (e.g. parent is class_specifier)
                    # Basic name guess for constructors: find class name if parent is class_specifier
                    if element_node.parent and element_node.parent.type in ('class_specifier', 'struct_specifier'):
                        class_name_node = find_child_by_type(element_node.parent, 'type_identifier')
                        func_id_node = find_child_by_type(element_node, 'identifier') # Name inside function def
                        if class_name_node and func_id_node and get_text(class_name_node, content_bytes) == get_text(func_id_node, content_bytes):
                            element_type = 'constructor'
                            element['name'] = get_text(func_id_node, content_bytes)
                            name_node = func_id_node # Mark name as found

                    if not name_node: # If still no name, log and skip
                        logging.debug(f"Skipping potentially complex function-like construct without clear name/declarator at line {start_line}")
                        continue
            elif capture_name in ['class', 'struct', 'enum']:
                name_node = find_child_by_type(element_node, 'type_identifier')
                if name_node: element['name'] = get_text(name_node, content_bytes)
                element_type = capture_name # Use 'class', 'struct', 'enum'
            elif capture_name == 'namespace':
                 name_node = element_node.child_by_field_name('name') # Use field name if available
                 if not name_node: name_node = find_child_by_type(element_node, 'identifier') # Fallback
                 if name_node: element['name'] = get_text(name_node, content_bytes)
                 element_type = 'namespace'
            elif capture_name == 'variable_declaration':
                # The query already captured @id_node and @type_node implicitly via structure, need to find them
                # Find the specific node captured as 'id_node' for the name
                decl_node = element_node.child_by_field_name('declarator') # The declarator part of the match
                id_cap_node = None
                if decl_node:
                    if decl_node.type == 'identifier':
                        id_cap_node = decl_node
                    elif decl_node.type == 'init_declarator':
                        id_cap_node = decl_node.child_by_field_name('declarator') # Usually an identifier here

                type_cap_node = element_node.child_by_field_name('type')

                if id_cap_node and id_cap_node.type == 'identifier': # Ensure it's an identifier
                    name_node = id_cap_node
                    element['name'] = get_text(name_node, content_bytes)
                    element_type = 'variable' # Refine type
                    if type_cap_node:
                        element['details']['variable_type'] = get_text(type_cap_node, content_bytes)
                else:
                     logging.debug(f"Skipping declaration at line {start_line} - couldn't extract identifier. Node: {element_node.text.decode()[:50]}")
                     continue # Skip if no identifier found

            # --- Find Comment ---
            if element['name'] != '?': # Only proceed if we have a name
                 # Use the element_node (the declaration/definition node) to find the comment
                 element['comment'] = find_leading_comment(element_node, content_bytes)
                 element['type'] = element_type # Assign refined type
                 elements.append(element)
                 processed_nodes.add(element_node) # Mark as processed

        except Exception as e:
             # Use element_node for line number reporting if possible
             report_line = element_node.start_point[0] + 1 if element_node else start_line
             logging.warning(f"Error processing C++ captured node {capture_name} near line {report_line}: {e}", exc_info=True)

    elements.sort(key=lambda x: x['start_line'])
    return elements


def extract_documentation_elements_csharp(tree: Tree, content_bytes: bytes) -> List[Dict[str, Any]]:
    """Extracts elements and their preceding comments from a C# tree."""
    elements = []
    root_node = tree.root_node

    # Using captures that point to the declaration node itself
    query_str = """
    (method_declaration) @method
    (class_declaration) @class
    (struct_declaration) @struct
    (interface_declaration) @interface
    (enum_declaration) @enum
    (property_declaration) @property
    (field_declaration) @field
    (constructor_declaration) @constructor
    (destructor_declaration) @destructor
    (event_declaration) @event
    (delegate_declaration) @delegate
    (indexer_declaration) @indexer
    ; TODO: Add more as needed (e.g., operators)
    """
    try:
        query = CSHARP_LANGUAGE.query(query_str)
        captures = query.captures(root_node)
    except Exception as e:
        logging.error(f"Error creating or executing C# tree query: {e}", exc_info=True)
        return elements

    processed_nodes = set()
    for node, capture_name in captures:
         element_node = node # The capture directly points to the declaration
         if element_node in processed_nodes: continue
         processed_nodes.add(element_node)

         start_line = element_node.start_point[0] + 1
         element_type = capture_name # e.g., 'method', 'class'
         element = {'type': element_type, 'node': element_node, 'start_line': start_line, 'name': '?', 'details': {}, 'comment': None}

         try:
             # Get the name - C# grammar often uses a 'name' field or direct identifier child
             name_node = element_node.child_by_field_name('name')
             var_decl = None # Used for fields

             if not name_node and element_type == 'field':
                  # Field name is nested deeper: field_declaration -> variable_declaration -> variable_declarator -> name:identifier
                  var_decl = find_descendant_by_type(element_node, 'variable_declaration')
                  if var_decl:
                       var_declarator = find_descendant_by_type(var_decl, 'variable_declarator')
                       if var_declarator: name_node = var_declarator.child_by_field_name('name')
             elif element_type == 'event' and not name_node:
                 # Event field declarations might also follow the field pattern
                 var_decl = find_descendant_by_type(element_node, 'variable_declaration')
                 if var_decl:
                       var_declarator = find_descendant_by_type(var_decl, 'variable_declarator')
                       if var_declarator: name_node = var_declarator.child_by_field_name('name')
             elif element_type == 'indexer':
                 # Indexers use 'this' keyword, don't have a simple name field
                 element['name'] = 'this[]' # Represent indexer
                 name_node = element_node # Mark name as found conceptually
             elif not name_node: # General fallback: find first identifier child if no 'name' field
                  name_node = find_child_by_type(element_node, 'identifier')


             if name_node and element['name']=='?': # If name not set by special case like indexer
                 element['name'] = get_text(name_node, content_bytes)
             elif element['name']=='?':
                 # Handle special cases like destructors or implicit constructor names
                 if element_type == 'destructor':
                      tilde_node = find_child_by_type(element_node, '~')
                      if tilde_node and tilde_node.next_named_sibling and tilde_node.next_named_sibling.type == 'identifier':
                           element['name'] = "~" + get_text(tilde_node.next_named_sibling, content_bytes)
                 elif element_type == 'constructor':
                      # Constructor name is the class name. Find the identifier for the constructor declaration.
                      constructor_id_node = find_child_by_type(element_node, 'identifier')
                      if constructor_id_node:
                          element['name'] = get_text(constructor_id_node, content_bytes) # Should match class name
                      else: # Static constructor might not have identifier node in this pos
                         static_kw = find_child_by_type(element_node, 'static')
                         if static_kw:
                             # Try to infer from parent class
                             parent = element_node.parent
                             while parent and parent.type not in ('class_declaration', 'struct_declaration'):
                                 parent = parent.parent
                             if parent:
                                 class_name_node = parent.child_by_field_name('name')
                                 if class_name_node: element['name'] = get_text(class_name_node, content_bytes) + " (Static Constructor)"


             if element['name'] == '?':
                 logging.debug(f"Could not find name for {element_type} at line {start_line}. Node text: {get_text(element_node, content_bytes)[:50]}...")
                 continue # Skip if no name identifiable

             # --- Find Comment ---
             # Use the element_node (the declaration node) to find the comment before it
             element['comment'] = find_leading_comment(element_node, content_bytes)

             # --- Extract more details (Example for method) ---
             if element_type == 'method':
                 # More robust return type finding
                 return_type_node = element_node.child_by_field_name('return_type')
                 if return_type_node:
                     element['return_type'] = get_text(return_type_node, content_bytes)
                 else: # Handle void or implicit types if necessary
                     void_kw = find_child_by_type(element_node, 'void_keyword')
                     if void_kw: element['return_type'] = 'void'


                 params_list_node = element_node.child_by_field_name('parameters')
                 if params_list_node:
                     parameters = []
                     # Iterate through named children which should be 'parameter' nodes
                     for param_node in params_list_node.named_children:
                          if param_node.type == 'parameter':
                              p_type_node = param_node.child_by_field_name('type')
                              p_name_node = param_node.child_by_field_name('name')
                              p_type = get_text(p_type_node, content_bytes) if p_type_node else '?'
                              p_name = get_text(p_name_node, content_bytes) if p_name_node else '?'
                              if p_name != '?': parameters.append({'name': p_name, 'type': p_type})
                          elif param_node.type == 'this_parameter': # Handle extension methods
                              p_type_node = param_node.child_by_field_name('type')
                              p_type = get_text(p_type_node, content_bytes) if p_type_node else '?'
                              parameters.append({'name': 'this', 'type': p_type})

                     element['details']['parameters'] = parameters

             # --- Extract details for other types ---
             if element_type == 'field':
                 # Type might be directly on field_declaration or nested in variable_declaration
                 type_node = element_node.child_by_field_name('type')
                 if not type_node and var_decl: type_node = var_decl.child_by_field_name('type')
                 if type_node: element['details']['variable_type'] = get_text(type_node, content_bytes)

             if element_type == 'property':
                 type_node = element_node.child_by_field_name('type')
                 if type_node: element['details']['property_type'] = get_text(type_node, content_bytes)
                 # Could also extract get/set accessors from 'accessor_list' child

             if element_type == 'delegate':
                  return_type_node = element_node.child_by_field_name('return_type')
                  if return_type_node: element['return_type'] = get_text(return_type_node, content_bytes)
                  params_list_node = element_node.child_by_field_name('parameters')
                  # Similar parameter extraction as methods...

             elements.append(element)
         except Exception as e:
              logging.warning(f"Error processing C# node {capture_name} at line {start_line}: {e}", exc_info=True)

    elements.sort(key=lambda x: x['start_line'])
    return elements


def generate_doxygen_markdown(elements: List[Dict[str, Any]], file_path: Path) -> str:
    """Generates Doxygen Markdown documentation, incorporating extracted comments."""
    md_lines = [f"# Documentation for `{file_path.name}`\n"]

    for element in elements:
        name = element.get('name', 'Unknown')
        # Clean up type name for display and Doxygen command
        elem_type = element.get('type', 'unknown').replace('_declaration','').replace('_definition','').replace('_specifier','')
        start_line = element.get('start_line', '?')
        details = element.get('details', {})
        comment = element.get('comment') # Get the cleaned comment

        # --- Parse comment for brief/detailed ---
        brief_desc = f"Brief description for {name}" # Default placeholder
        detailed_desc = f"Detailed description for {name}" # Default placeholder
        if comment:
             comment_lines = comment.strip().split('\n')
             # Find first non-empty line for brief description
             first_non_empty_line_index = -1
             for i, line in enumerate(comment_lines):
                 if line.strip():
                     first_non_empty_line_index = i
                     break

             if first_non_empty_line_index != -1:
                 brief_desc = comment_lines[first_non_empty_line_index].strip()
                 # Join the rest (including empty lines for paragraphs) for detailed description
                 # Start detailed description from the line after the brief line
                 detailed_desc_lines = comment_lines[first_non_empty_line_index + 1:]
                 # Remove leading/trailing empty lines from the detailed part
                 start_detailed = 0
                 while start_detailed < len(detailed_desc_lines) and not detailed_desc_lines[start_detailed].strip():
                      start_detailed += 1
                 end_detailed = len(detailed_desc_lines) -1
                 while end_detailed >= start_detailed and not detailed_desc_lines[end_detailed].strip():
                      end_detailed -= 1
                 detailed_desc = "\n".join(detailed_desc_lines[start_detailed:end_detailed+1])

                 # If detailed description ended up empty (e.g., only one line comment), use brief as detailed.
                 if not detailed_desc.strip() and brief_desc:
                     detailed_desc = brief_desc
                 elif not detailed_desc.strip() and not brief_desc: # Both empty? Use default.
                      brief_desc = f"Brief description for {name}"
                      detailed_desc = f"Detailed description for {name}"

             else: # Comment exists but is empty after cleaning? Use default.
                 pass # Defaults already set

        # Escape backticks and potentially other markdown chars in name for code formatting
        safe_name = name.replace('`', '\\`').replace('*', '\\*').replace('_', '\\_')
        # Doxygen command name should likely be original name
        doxy_name = name

        md_lines.append(f"## `{safe_name}` ({elem_type})\n")
        md_lines.append(f"*Defined at: `{file_path.name}#L{start_line}`*\n")
        md_lines.append("```doxygen")
        # Use original name for the doxygen command itself
        md_lines.append(f"/*! \\{elem_type} {doxy_name}")
        # Add brief description, ensuring it's on one line for the \\brief command
        md_lines.append(f" *  \\brief {brief_desc.replace(chr(10), ' ').replace(chr(13), '')}") # Replace newlines in brief

        # Add parameters for functions/methods/delegates
        # TODO: Check if comment already contains @param and use that instead/merge? Simple approach first.
        if 'parameters' in details and details['parameters']:
            for param in details['parameters']:
                p_name = param.get('name', 'param')
                p_type = param.get('type', '')
                # Escape potential markdown in param type/name if necessary
                safe_p_name = p_name.replace('_', '\\_')
                safe_p_type = p_type.replace('<', '\\<').replace('>', '\\>') # Escape angle brackets
                md_lines.append(f" *  \\param {safe_p_name} Parameter description. Type: `{safe_p_type}`")

        # Add return type for functions/methods/delegates
        # TODO: Check comment for @return
        if 'return_type' in element and element['return_type']:
             safe_return_type = element['return_type'].replace('<', '\\<').replace('>', '\\>')
             md_lines.append(f" *  \\return Return value description. Type: `{safe_return_type}`")
        elif elem_type in ['constructor', 'destructor']:
             pass # Constructors/destructors don't have return values in the same way
        # else: # Add a default return line? Maybe not.
             # md_lines.append(f" *  \\return Return value description.")


        # Add detailed description
        if detailed_desc and detailed_desc.strip(): # Only add if not empty
             md_lines.append(" *") # Separator line before detailed description
             # Add detailed description lines, prepending with ' *  '
             for line in detailed_desc.split('\n'):
                  # Escape potential Doxygen commands or markdown in the comment body if needed
                  md_lines.append(f" *  {line}") # Basic insertion for now

        md_lines.append(" */")
        md_lines.append("```\n")

    return "\n".join(md_lines)


def log_parse_errors(node: Node, file_path: Path):
    """Recursively finds and logs ERROR nodes in the syntax tree. (From gen.py)"""
    if node.type == 'ERROR' or node.is_missing: # Also log missing nodes
        start_line, start_col = node.start_point
        end_line, end_col = node.end_point
        error_text_snippet = node.text.decode('utf-8', errors='replace')[:100]
        status = "ERROR" if node.type == 'ERROR' else "MISSING"
        logging.warning(
            f"Parse {status} in {file_path.relative_to(Path.cwd()).as_posix()} at " # Use relative path for cleaner logs
            f"L{start_line+1}:C{start_col+1}-L{end_line+1}:C{end_col+1}. "
            f"Node: {node.type}. Snippet: '{error_text_snippet}...'"
        )
    # Continue traversal even if current node is an error
    for child in node.children:
        log_parse_errors(child, file_path)


async def process_file(target_file_path: Path, root_dir: Path, output_dir: Path) -> Tuple[bool, int]:
    """Processes a single C++/C# file to generate documentation."""
    file_ext = target_file_path.suffix.lower()
    language: Optional[Language] = None
    extract_func = None
    lang_name = "unknown"
    elements_count = 0
    # Use relative path for logging
    relative_file_path = target_file_path.relative_to(root_dir).as_posix()

    # Log only the relative path for clarity in the processed files log
    logging.info(f"{relative_file_path}") # Log processed file path (relative)

    if file_ext in CPP_EXTENSIONS: language, extract_func, lang_name = CPP_LANGUAGE, extract_documentation_elements_cpp, "cpp"
    elif file_ext in CSHARP_EXTENSIONS: language, extract_func, lang_name = CSHARP_LANGUAGE, extract_documentation_elements_csharp, "csharp"
    else:
        logging.warning(f"Skipping file with unsupported extension: {relative_file_path}")
        return False, 0 # Should not happen if called correctly

    if not language or not extract_func:
         logging.error(f"Skipping {relative_file_path}, language object or extract function not available for {lang_name}.")
         return False, 0 # Failed file

    try:
        MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024 # 15 MB limit for docs generation
        file_size = await asyncio.to_thread(os.path.getsize, target_file_path)
        if file_size == 0:
            logging.warning(f"Skipping empty file: {relative_file_path}")
            return True, 0 # Success, but 0 elements
        if file_size > MAX_FILE_SIZE_BYTES:
             logging.warning(f"Skipping {relative_file_path}, file size ({file_size} bytes) exceeds limit ({MAX_FILE_SIZE_BYTES} bytes).")
             return False, 0 # Failed file

        file_contents_bytes = await asyncio.to_thread(target_file_path.read_bytes)
        # Attempt decoding early to catch errors
        try:
            file_contents_str = file_contents_bytes.decode('utf-8')
        except UnicodeDecodeError as ude:
            logging.error(f"Could not decode file {relative_file_path} as utf-8: {ude}", exc_info=False)
            return False, 0 # Failed file

        # --- Parsing ---
        parser = Parser()
        parser.set_language(language)
        tree = await asyncio.to_thread(parser.parse, file_contents_bytes)

        if tree is None:
             logging.warning(f"Tree-sitter parsing timed out for file: {relative_file_path}")
             return False, 0

        # Check for parse errors but continue processing regardless (Point 1 requirement)
        if tree.root_node and (tree.root_node.has_error or any(n.is_missing for n in tree.root_node.descendants)):
             logging.warning(f"Parsing resulted in errors or missing nodes for file: {relative_file_path}. Extraction will be attempted.")
             try: await asyncio.to_thread(log_parse_errors, tree.root_node, target_file_path)
             except Exception as log_err: logging.error(f"Failed log parse errors for {relative_file_path}: {log_err}")

        # --- Extraction ---
        elements = await asyncio.to_thread(extract_func, tree, file_contents_bytes)
        elements_count = len(elements)
        if not elements:
            logging.warning(f"No documentable elements found or extracted in {relative_file_path}.") # Changed to warning as it might indicate issues
            # Still considered success if parsing didn't fail critically
            return True, 0

        # --- Documentation Generation ---
        markdown_content = await asyncio.to_thread(generate_doxygen_markdown, elements, target_file_path)

        # --- Output ---
        relative_path = target_file_path.relative_to(root_dir)
        output_file_path = output_dir / relative_path.with_suffix(DOC_SUFFIX)

        # Ensure output directory exists
        await asyncio.to_thread(output_file_path.parent.mkdir, parents=True, exist_ok=True)

        # Write markdown file asynchronously
        async with asyncio.Semaphore(50): # Limit concurrent file writes
            async def write_file():
                 try:
                     with open(output_file_path, 'w', encoding='utf-8') as f:
                         f.write(markdown_content)
                 except OSError as write_err:
                     logging.error(f"Failed to write documentation file {output_file_path.relative_to(output_dir)}: {write_err}", exc_info=True)
                     raise # Re-raise to mark as failure
            await asyncio.to_thread(write_file)


        return True, elements_count # Success

    except FileNotFoundError:
        logging.error(f"File not found during processing: {relative_file_path}")
        return False, 0
    except OSError as os_err:
         logging.error(f"OS Error processing file {relative_file_path}: {os_err}", exc_info=True)
         return False, 0
    except Exception as e:
        logging.exception(f"Unexpected Error processing file {relative_file_path}: {e}")
        return False, 0

# --- Logging Setup (Adapted from gen.py) ---
def setup_logging(log_dir: Path, files_log_path: Path, error_log_path: Path):
    """Configures logging to files only."""
    log_level = logging.INFO
    logger = logging.getLogger()
    logger.setLevel(log_level)
    # Clear existing handlers if any (e.g., from previous runs or other libs)
    if logger.hasHandlers():
        for handler in logger.handlers[:]: # Iterate over a copy
            # Be slightly more careful: only remove handlers we likely added
            if isinstance(handler, (logging.FileHandler, logging.StreamHandler)):
                 # Prevent closing handlers used by other potential libraries
                 # if handler.stream == sys.stderr or handler.stream == sys.stdout: continue
                 try:
                      handler.acquire() # Ensure thread safety during removal
                      handler.flush()
                      handler.close()
                 except (OSError, ValueError): pass # Ignore errors during close
                 finally: handler.release()
                 logger.removeHandler(handler)

    # Ensure log directory exists
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Error creating log directory {log_dir}: {e}. Logging to files might fail.", file=sys.stderr)
        # Continue execution, handlers might fail below but script can proceed

    # Info handler (processed files)
    info_formatter = logging.Formatter('%(message)s') # Simple format for file list
    try:
        info_handler = logging.FileHandler(files_log_path, mode='w', encoding='utf-8')
        info_handler.setLevel(logging.INFO)
        info_handler.setFormatter(info_formatter)
        # Filter to log ONLY level INFO, not WARNING or ERROR which go to the error log
        info_handler.addFilter(lambda record: record.levelno == logging.INFO)
        logger.addHandler(info_handler)
    except Exception as e:
        # Fallback to stderr if file logging setup fails for info
        print(f"Warning: Failed to set up info log file at {files_log_path}: {e}", file=sys.stderr)


    # Error handler (warnings/errors)
    error_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s') # Use module instead of filename
    try:
        error_handler = logging.FileHandler(error_log_path, mode='w', encoding='utf-8')
        error_handler.setLevel(logging.WARNING) # Log WARNING and above
        error_handler.setFormatter(error_formatter)
        logger.addHandler(error_handler)
    except Exception as e:
         # Fallback to stderr if file logging setup fails for errors
        print(f"Warning: Failed to set up error log file at {error_log_path}: {e}", file=sys.stderr)


    # --- REMOVED CONSOLE HANDLER ---
    # No StreamHandler is added here.

    # Initial log message to confirm setup (will go to error log if level >= WARNING, otherwise nowhere if only info handler works)
    # Let's add one info log that *should* go to the info file.
    logging.info(f"Logging configured. Processed files will be listed in: {files_log_path.name}")
    # And one warning that should go to the error file.
    logging.warning(f"Error/Warning logs will be recorded in: {error_log_path.name}")


# --- Main Execution ---
async def main(root_dir_str: str, output_dir_str: Optional[str]):
    """主函数"""
    root_dir = Path(root_dir_str).resolve()
    output_dir = Path(output_dir_str).resolve() if output_dir_str else DEFAULT_OUTPUT_DIR
    if not root_dir.is_dir():
        print(f"Fatal: Source path '{root_dir_str}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    # Output directory is created within setup_logging to handle potential errors there
    # output_dir.mkdir(parents=True, exist_ok=True) # Moved to setup_logging

    # --- Configure Logging ---
    files_log_path = output_dir / FILES_LOG_FILENAME
    ignores_log_path = output_dir / IGNORES_LOG_FILENAME
    error_log_path = output_dir / ERROR_LOG_FILENAME
    setup_logging(output_dir, files_log_path, error_log_path) # Also creates output_dir
    # -------------------------

    # Log initial paths after setup
    logging.warning(f"Source directory: {root_dir}") # Use warning to ensure it appears in error log
    logging.warning(f"Output directory: {output_dir}") # Use warning to ensure it appears in error log

    # --- Load Languages ---
    try:
        load_or_build_languages()
        if not CPP_LANGUAGE or not CSHARP_LANGUAGE:
             logging.critical("Languages were not loaded correctly. Exiting.")
             print("CRITICAL: Languages were not loaded correctly. Check error log. Exiting.", file=sys.stderr)
             sys.exit(1)
    except SystemExit: raise
    except Exception as load_exc:
        logging.critical(f"Unhandled exception during language loading: {load_exc}", exc_info=True)
        print(f"CRITICAL: Unhandled exception during language loading: {load_exc}. Check error log. Exiting.", file=sys.stderr)
        sys.exit(1)
    # ----------------------

    # --- File Discovery (Adapted from gen.py) ---
    logging.warning(f"Scanning for {', '.join(TARGET_EXTENSIONS)} files in {root_dir}...") # Use warning for visibility
    print(f"Scanning for {', '.join(TARGET_EXTENSIONS)} files in {root_dir}...") # Keep console feedback for scanning start
    target_files: List[Path] = []
    dirs_to_skip_specific: Set[str] = {'ThirdParty', 'Extras', 'thirdparty'}
    dirs_to_skip_generic: Set[str] = {'.git', 'node_modules', 'bin', 'obj', 'Build', 'build', 'Intermediate', 'DerivedDataCache', '.vs', '.vscode'}
    skipped_dirs_count = 0

    def find_files_sync():
        nonlocal skipped_dirs_count
        count = 0
        try:
             # Ensure ignores log directory exists (output dir created by setup_logging)
             with open(ignores_log_path, 'w', encoding='utf-8') as ignores_log_file:
                 ignores_log_file.write(f"# Skipped directories during scan of: {root_dir}\n")
                 ignores_log_file.write(f"# Skip patterns (specific): {dirs_to_skip_specific}\n")
                 ignores_log_file.write(f"# Skip patterns (generic): {dirs_to_skip_generic}\n")
                 for dirpath_str, dirnames, filenames in os.walk(root_dir, topdown=True):
                     dirpath = Path(dirpath_str)
                     # Filter dirnames in place
                     original_dirnames_count = len(dirnames)
                     dirnames[:] = [d for d in dirnames if not (d in dirs_to_skip_specific or d in dirs_to_skip_generic or d.startswith('.'))]
                     skipped_count_in_dir = original_dirnames_count - len(dirnames)

                     if skipped_count_in_dir > 0:
                          skipped_dirs_count += skipped_count_in_dir
                          # Log which ones were skipped from this dir
                          skipped_names = set(os.listdir(dirpath)) & (dirs_to_skip_specific | dirs_to_skip_generic | {d for d in os.listdir(dirpath) if d.startswith('.')})
                          rel_dirpath = dirpath.relative_to(root_dir).as_posix() or '.'
                          ignores_log_file.write(f"In: {rel_dirpath}/\n")
                          for skipped_name in sorted(skipped_names):
                               if (dirpath / skipped_name).is_dir(): # Only log skipped directories
                                    ignores_log_file.write(f"  Skipping: {skipped_name}/\n")


                     for filename in filenames:
                         file_path = dirpath / filename
                         if file_path.suffix.lower() in TARGET_EXTENSIONS:
                             target_files.append(file_path)
                             count += 1
                             if count % 1000 == 0:
                                 # Keep progress indicator on console (stderr preferred for logs)
                                 print(f"Found {count} files...", end='\r', file=sys.stderr)
        except OSError as e:
             logging.error(f"Error accessing directory or writing to ignores log file '{ignores_log_path}': {e}. Scan may be incomplete.", exc_info=True)
             print(f"\nError during file scan: {e}. Check error log. Scan may be incomplete.", file=sys.stderr)
             # Continue without ignore logging if it failed, but list might be incomplete
             if not target_files: # Try a basic scan if the error happened early
                  target_files.clear()
                  count = 0
                  for dirpath_str, dirnames, filenames in os.walk(root_dir, topdown=True, onerror=lambda err: logging.error(f"Scan error: {err}", exc_info=True)):
                      dirnames[:] = [d for d in dirnames if not (d in dirs_to_skip_specific or d in dirs_to_skip_generic or d.startswith('.'))]
                      for filename in filenames:
                           file_path = Path(dirpath_str) / filename
                           if file_path.suffix.lower() in TARGET_EXTENSIONS:
                               target_files.append(file_path)
                               count += 1
                               if count % 1000 == 0: print(f"Found {count} files (recovery scan)...", end='\r', file=sys.stderr)

        finally:
             # Newline after progress indicator on console
             print(file=sys.stderr)

    await asyncio.to_thread(find_files_sync)

    if not target_files:
        logging.warning("No target files found matching criteria.")
        print("No target files found matching criteria. Exiting.", file=sys.stderr) # Console feedback
        sys.exit(0)

    total_files = len(target_files)
    logging.warning(f"Found {total_files} target files. Starting documentation generation...") # Use warning for visibility in log
    print(f"Found {total_files} target files. Starting documentation generation...") # Keep console feedback

    # --- File Processing ---
    successful_files = 0
    failed_files = 0
    total_elements = 0
    tasks = [process_file(file_path, root_dir, output_dir) for file_path in target_files]

    import time
    start_time = time.time()
    results_iterable = None
    try:
         from tqdm.asyncio import tqdm_asyncio
         # Use stderr for tqdm to avoid interfering with potential future stdout use
         results_iterable = tqdm_asyncio.as_completed(tasks, total=total_files, desc="Generating docs", unit="file", file=sys.stderr, ncols=100, leave=True)
    except ImportError:
         results_iterable = asyncio.as_completed(tasks)
         print("Processing files (install tqdm for progress bar: pip install tqdm)...", file=sys.stderr) # Console feedback

    # Process results as they complete
    processed_count = 0
    if results_iterable: # Check if iterator was created
        for future in results_iterable:
            processed_count += 1
            try:
                success, elements_count = await future
                if success:
                    successful_files += 1
                    total_elements += elements_count
                else:
                    failed_files += 1
            except Exception as task_exc:
                # Logging within the future might have already happened, but log here too for certainty
                logging.exception(f"Error processing future result (file count {processed_count}/{total_files}): {task_exc}")
                failed_files += 1
            # Optionally update console progress if not using tqdm
            if 'tqdm_asyncio' not in sys.modules:
                 print(f"Processed {processed_count}/{total_files} files ({failed_files} failures)...", end='\r', file=sys.stderr)

    # Final newline if using simple progress
    if 'tqdm_asyncio' not in sys.modules:
         print(file=sys.stderr)

    end_time = time.time()
    duration = end_time - start_time

    # --- Summary Output (Keep on Console for user feedback) ---
    print("\n--- Documentation Generation Summary ---", file=sys.stdout)
    print(f"Total files scanned: {total_files}", file=sys.stdout)
    print(f"Directories skipped: {skipped_dirs_count} (details in '{ignores_log_path.name}')", file=sys.stdout)
    print(f"Files successfully processed (logged to '{files_log_path.name}'): {successful_files}", file=sys.stdout)
    print(f"Files failed or skipped (check '{error_log_path.name}'): {failed_files}", file=sys.stdout)
    print(f"Total documentation elements extracted: {total_elements}", file=sys.stdout)
    print(f"Output directory: {output_dir.resolve()}", file=sys.stdout)
    # print(f"Processed file list log: {files_log_path.resolve()}", file=sys.stdout) # Redundant with above
    print(f"Errors and warnings log: {error_log_path.resolve()}", file=sys.stdout)
    print(f"Total processing time: {duration:.2f} seconds", file=sys.stdout)

    # Also log summary details to the error log for persistence
    logging.warning("--- Documentation Generation Summary ---")
    logging.warning(f"Total files scanned: {total_files}")
    logging.warning(f"Directories skipped: {skipped_dirs_count}")
    logging.warning(f"Files successfully processed: {successful_files}")
    logging.warning(f"Files failed or skipped: {failed_files}")
    logging.warning(f"Total documentation elements extracted: {total_elements}")
    logging.warning(f"Output directory: {output_dir.resolve()}")
    logging.warning(f"Total processing time: {duration:.2f} seconds")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Keep usage instructions on stderr
        print(f"Usage: python {Path(sys.argv[0]).name} <path_to_project_root> [output_directory]", file=sys.stderr)
        print(f"  Example: python {Path(sys.argv[0]).name} /path/to/MyUnrealProject ./docs_output", file=sys.stderr)
        sys.exit(1)

    project_root_dir = sys.argv[1]
    output_dir_arg = sys.argv[2] if len(sys.argv) > 2 else None

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        asyncio.run(main(project_root_dir, output_dir_arg))
    except KeyboardInterrupt:
         print("\nExecution interrupted by user.", file=sys.stderr)
         logging.warning("Execution interrupted by user.")
         sys.exit(130) # Standard exit code for Ctrl+C
    except Exception as e:
        # Ensure critical errors are logged before exiting
        # Logger might not be fully configured if error happens early, so print as fallback
        print(f"CRITICAL: Unhandled error during script execution: {e}", file=sys.stderr)
        try:
             logging.critical(f"Unhandled error during script execution: {e}", exc_info=True)
        except Exception as log_err:
             print(f"CRITICAL: Also failed to log the critical error: {log_err}", file=sys.stderr)
        sys.exit(1)
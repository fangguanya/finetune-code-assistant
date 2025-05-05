import json
import os
import random
import sys
import re # Needed for comment cleaning
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Set, TextIO
import time # Keep time import

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

# --- Tree-sitter Language Loading ---
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
        if lines[0].strip().endswith("*/"):
            lines[0] = lines[0].rsplit("*/", 1)[0]
            in_multi_line = False
        elif len(lines) > 1 and lines[-1].strip().endswith("*/"):
             lines[-1] = lines[-1].rsplit("*/", 1)[0]
             in_multi_line = True
        elif len(lines) == 1:
             in_multi_line = False
        else:
             in_multi_line = True

    for line in lines:
        line = line.strip()
        if line.startswith("//"):
            line = line[2:].strip()
        elif in_multi_line and line.startswith("*"):
            line = line[1:].strip()
        cleaned_lines.append(line)

    start_index = 0
    while start_index < len(cleaned_lines) and not cleaned_lines[start_index].strip():
        start_index += 1
    end_index = len(cleaned_lines) - 1
    while end_index >= start_index and not cleaned_lines[end_index].strip():
        end_index -= 1
    return "\n".join(cleaned_lines[start_index:end_index+1])

def find_leading_comment(node: Optional[Node], content_bytes: bytes) -> Optional[str]:
    """Finds preceding comment(s) for a given node."""
    if not node or not node.parent: return None
    comment_blocks: List[str] = []
    last_comment_block_end_line = -1
    first_comment_block_start_line = -1
    max_lines_gap_between_comments = 1
    max_lines_gap_to_code = 2
    target_start_line = node.start_point[0]
    parent = node.parent
    target_node_index = -1

    try:
        siblings = parent.children
        for i, sibling in enumerate(siblings):
            if sibling.start_byte == node.start_byte and sibling.end_byte == node.end_byte and sibling.type == node.type:
                 target_node_index = i
                 break
    except Exception as e:
         logging.error(f"Error finding node index within parent (node type: {node.type}, parent type: {parent.type}): {e}", exc_info=True)
         return None
    if target_node_index == -1:
         logging.debug(f"Could not find the target node (type: {node.type}, line: {target_start_line + 1}) within its parent's ({parent.type}) children list.")
         return None

    for i in range(target_node_index - 1, -1, -1):
        prev_sibling = siblings[i]
        prev_sibling_start_line = prev_sibling.start_point[0]
        prev_sibling_end_line = prev_sibling.end_point[0]
        is_comment_node = prev_sibling.type == 'comment'
        gap_reference_start_line = target_start_line if last_comment_block_end_line == -1 else first_comment_block_start_line
        gap_to_next_element = gap_reference_start_line - prev_sibling_end_line
        max_allowed_gap = max_lines_gap_to_code if last_comment_block_end_line == -1 else max_lines_gap_between_comments

        if gap_to_next_element > max_allowed_gap + 1:
             logging.debug(f"Stopping comment accumulation for node at L{target_start_line + 1}: Gap ({gap_to_next_element} lines) between sibling ending L{prev_sibling_end_line + 1} and next element starting L{gap_reference_start_line + 1} exceeds max allowed ({max_allowed_gap}).")
             break

        if is_comment_node:
            raw_comment = get_text(prev_sibling, content_bytes)
            cleaned = clean_comment(raw_comment)
            if cleaned:
                comment_blocks.append(cleaned)
                if last_comment_block_end_line == -1:
                    last_comment_block_end_line = prev_sibling_end_line
                first_comment_block_start_line = prev_sibling_start_line
                logging.debug(f"Accumulated comment block ending at L{prev_sibling_end_line + 1}")
            else:
                logging.debug(f"Ignoring empty comment block ending at L{prev_sibling_end_line + 1}")
                continue
        elif prev_sibling.is_named:
            logging.debug(f"Stopping comment accumulation for node at L{target_start_line + 1}: Encountered non-comment named node '{prev_sibling.type}' ending at L{prev_sibling_end_line + 1}.")
            break
        else:
            continue

    if not comment_blocks: return None

    final_gap_to_code = target_start_line - last_comment_block_end_line
    if final_gap_to_code > max_lines_gap_to_code + 1:
         logging.debug(f"Discarding accumulated comments for node at L{target_start_line + 1}: Final gap ({final_gap_to_code} lines) between last comment (ends L{last_comment_block_end_line + 1}) and code exceeds max ({max_lines_gap_to_code}).")
         return None

    full_comment = "\n\n".join(reversed(comment_blocks))
    return full_comment

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

def find_missing_recursive(node: Optional[Node]) -> bool:
    """Recursively checks if a node or any of its descendants is missing."""
    if not node:
        return False
    if node.is_missing:
        return True
    for child in node.children:
        if find_missing_recursive(child):
            return True
    return False

def extract_documentation_elements_cpp(tree: Tree, content_bytes: bytes) -> List[Dict[str, Any]]:
    """Extracts elements and their preceding comments from a C++ tree."""
    elements = []
    root_node = tree.root_node
    query_str = """
    (function_definition) @function
    (declaration type: (_) declarator: (function_declarator)) @function_declaration
    (class_specifier) @class
    (struct_specifier) @struct
    (enum_specifier) @enum
    (declaration
      type: (_) @type_node
      declarator: [(identifier) @id_node (init_declarator declarator: (identifier) @id_node)] @decl_node
      (#not-match? @decl_node "(parameter_declaration|template_parameter_list|template_declaration)")) @variable_declaration
    (namespace_definition) @namespace
    """
    try:
        query = CPP_LANGUAGE.query(query_str)
        captures = query.captures(root_node)
    except Exception as e:
        logging.error(f"Error creating or executing C++ tree query: {e}", exc_info=True)
        return elements

    processed_nodes = set()
    for node, capture_name in captures:
        element_node = node
        node_id = (element_node.start_byte, element_node.end_byte, element_node.type)
        if node_id in processed_nodes: continue

        start_line = element_node.start_point[0] + 1
        element = {'type': capture_name, 'node': element_node, 'start_line': start_line, 'name': '?', 'details': {}, 'comment': None}

        try:
            name_node = None
            element_type = capture_name

            if capture_name == 'function' or capture_name == 'function_declaration':
                element_type = 'function'
                declarator = find_child_by_type(element_node, 'function_declarator')
                if declarator:
                    name_node = find_descendant_by_type(declarator, 'qualified_identifier')
                    if not name_node: name_node = find_child_by_type(declarator, 'identifier')
                    if not name_node: name_node = find_child_by_type(declarator, 'operator_name')
                    if not name_node: name_node = find_child_by_type(declarator, 'destructor_name')
                    if name_node: element['name'] = get_text(name_node, content_bytes)
                    element['signature'] = get_text(declarator, content_bytes)
                    type_node = find_child_by_type(element_node, 'type_identifier') or \
                                find_child_by_type(element_node, 'primitive_type') or \
                                element_node.child_by_field_name('type')
                    if type_node: element['return_type'] = get_text(type_node, content_bytes)
                elif find_descendant_by_type(element_node, 'destructor_name'):
                    element_type = 'destructor'
                    name_node = find_descendant_by_type(element_node, 'destructor_name')
                    if name_node: element['name'] = get_text(name_node, content_bytes)
                else:
                    id_node = find_child_by_type(element_node, 'identifier')
                    if element_node.parent and element_node.parent.type in ('class_specifier', 'struct_specifier'):
                        class_name_node = find_child_by_type(element_node.parent, 'type_identifier')
                        func_id_node = find_child_by_type(element_node, 'identifier')
                        if class_name_node and func_id_node and get_text(class_name_node, content_bytes) == get_text(func_id_node, content_bytes):
                            element_type = 'constructor'
                            element['name'] = get_text(func_id_node, content_bytes)
                            name_node = func_id_node

                    if not name_node and id_node:
                        element['name'] = get_text(id_node, content_bytes)
                        name_node = id_node
                        logging.debug(f"Used fallback identifier '{element['name']}' for function-like construct at line {start_line}")

                    if not name_node:
                        logging.debug(f"Skipping potentially complex function-like construct without clear name/declarator at line {start_line}. Text: {get_text(element_node, content_bytes)[:50]}")
                        continue
            elif capture_name in ['class', 'struct', 'enum']:
                name_node = find_child_by_type(element_node, 'type_identifier')
                if name_node: element['name'] = get_text(name_node, content_bytes)
                element_type = capture_name
            elif capture_name == 'namespace':
                 name_node = element_node.child_by_field_name('name')
                 if not name_node: name_node = find_child_by_type(element_node, 'identifier')
                 if name_node: element['name'] = get_text(name_node, content_bytes)
                 element_type = 'namespace'
            elif capture_name == 'variable_declaration':
                decl_node = element_node.child_by_field_name('declarator')
                id_cap_node = None
                if decl_node:
                    if decl_node.type == 'identifier':
                        id_cap_node = decl_node
                    elif decl_node.type == 'init_declarator':
                         inner_decl = decl_node.child_by_field_name('declarator')
                         if inner_decl and inner_decl.type == 'identifier':
                            id_cap_node = inner_decl
                         elif inner_decl:
                             id_cap_node = find_descendant_by_type(inner_decl, 'identifier')
                type_cap_node = element_node.child_by_field_name('type')

                if id_cap_node and id_cap_node.type == 'identifier':
                    name_node = id_cap_node
                    element['name'] = get_text(name_node, content_bytes)
                    element_type = 'variable'
                    if type_cap_node:
                        element['details']['variable_type'] = get_text(type_cap_node, content_bytes)
                else:
                     logging.debug(f"Skipping declaration at line {start_line} - couldn't extract identifier. Node: {element_node.text.decode('utf-8', 'replace')[:50]}")
                     continue

            if element['name'] != '?':
                 element['comment'] = find_leading_comment(element_node, content_bytes)
                 element['type'] = element_type
                 elements.append(element)
                 processed_nodes.add(node_id)
            elif element['name'] == '?' and name_node:
                 logging.warning(f"Found name_node for {capture_name} at line {start_line} but element['name'] is still '?'. Node: {get_text(element_node, content_bytes)[:50]}")
            elif element['name'] == '?':
                 if capture_name not in ('function', 'function_declaration'):
                     logging.debug(f"Could not determine name for {capture_name} at line {start_line}. Node: {get_text(element_node, content_bytes)[:50]}")

        except Exception as e:
             report_line = element_node.start_point[0] + 1 if element_node else start_line
             logging.warning(f"Error processing C++ captured node {capture_name} near line {report_line}: {e}", exc_info=True)

    elements.sort(key=lambda x: x['start_line'])
    return elements


def extract_documentation_elements_csharp(tree: Tree, content_bytes: bytes) -> List[Dict[str, Any]]:
    """Extracts elements and their preceding comments from a C# tree."""
    elements = []
    root_node = tree.root_node
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
    """
    try:
        query = CSHARP_LANGUAGE.query(query_str)
        captures = query.captures(root_node)
    except Exception as e:
        logging.error(f"Error creating or executing C# tree query: {e}", exc_info=True)
        return elements

    processed_nodes = set()
    for node, capture_name in captures:
         element_node = node
         node_id = (element_node.start_byte, element_node.end_byte, element_node.type)
         if node_id in processed_nodes: continue
         processed_nodes.add(node_id)

         start_line = element_node.start_point[0] + 1
         element_type = capture_name
         element = {'type': element_type, 'node': element_node, 'start_line': start_line, 'name': '?', 'details': {}, 'comment': None}

         try:
             name_node = element_node.child_by_field_name('name')
             var_decl = None

             if not name_node and element_type == 'field':
                  var_decl = find_descendant_by_type(element_node, 'variable_declaration')
                  if var_decl:
                       var_declarator = find_descendant_by_type(var_decl, 'variable_declarator')
                       if var_declarator: name_node = var_declarator.child_by_field_name('name')
             elif element_type == 'event' and not name_node:
                 var_decl = find_descendant_by_type(element_node, 'variable_declaration')
                 if var_decl:
                       var_declarator = find_descendant_by_type(var_decl, 'variable_declarator')
                       if var_declarator: name_node = var_declarator.child_by_field_name('name')
             elif element_type == 'indexer':
                 element['name'] = 'this[]'
             elif not name_node:
                  name_node = find_child_by_type(element_node, 'identifier')

             if name_node and element['name']=='?':
                 element['name'] = get_text(name_node, content_bytes)
             elif element['name']=='?':
                 if element_type == 'destructor':
                      tilde_node = find_child_by_type(element_node, '~')
                      if tilde_node:
                          id_node = find_child_by_type(element_node, 'identifier')
                          if id_node: element['name'] = "~" + get_text(id_node, content_bytes)
                 elif element_type == 'constructor':
                      constructor_id_node = find_child_by_type(element_node, 'identifier')
                      if constructor_id_node:
                          element['name'] = get_text(constructor_id_node, content_bytes)
                      else:
                         is_static = False
                         for child in element_node.children:
                             if child.type == 'static_keyword':
                                 is_static = True
                                 break
                         if is_static:
                             parent = element_node.parent
                             while parent and parent.type not in ('class_declaration', 'struct_declaration'):
                                 parent = parent.parent
                             if parent:
                                 class_name_node = parent.child_by_field_name('name')
                                 if class_name_node: element['name'] = get_text(class_name_node, content_bytes) + " (Static Constructor)"

             if element['name'] == '?':
                 logging.debug(f"Could not find name for {element_type} at line {start_line}. Node text: {get_text(element_node, content_bytes)[:50]}...")
                 continue

             element['comment'] = find_leading_comment(element_node, content_bytes)

             if element_type == 'method':
                 return_type_node = element_node.child_by_field_name('return_type')
                 if return_type_node:
                     element['return_type'] = get_text(return_type_node, content_bytes)
                 else:
                      is_void = False
                      for child in element_node.children:
                          if child.type == 'void_keyword':
                              is_void = True
                              break
                      if is_void: element['return_type'] = 'void'
                 params_list_node = element_node.child_by_field_name('parameters')
                 if params_list_node:
                     parameters = []
                     for param_node in params_list_node.named_children:
                          if param_node.type == 'parameter':
                              p_type_node = param_node.child_by_field_name('type')
                              p_name_node = param_node.child_by_field_name('name')
                              p_type = get_text(p_type_node, content_bytes) if p_type_node else '?'
                              p_name = get_text(p_name_node, content_bytes) if p_name_node else '?'
                              if p_name != '?': parameters.append({'name': p_name, 'type': p_type})
                          elif param_node.type == 'this_parameter':
                              p_type_node = param_node.child_by_field_name('type')
                              p_type = get_text(p_type_node, content_bytes) if p_type_node else '?'
                              parameters.append({'name': 'this', 'type': p_type})
                     element['details']['parameters'] = parameters

             if element_type == 'field':
                 type_node = element_node.child_by_field_name('type')
                 if not type_node and var_decl: type_node = var_decl.child_by_field_name('type')
                 if type_node: element['details']['variable_type'] = get_text(type_node, content_bytes)

             if element_type == 'property':
                 type_node = element_node.child_by_field_name('type')
                 if type_node: element['details']['property_type'] = get_text(type_node, content_bytes)

             if element_type == 'delegate':
                  return_type_node = element_node.child_by_field_name('return_type')
                  if return_type_node: element['return_type'] = get_text(return_type_node, content_bytes)
                  params_list_node = element_node.child_by_field_name('parameters')
                  if params_list_node:
                     parameters = []
                     for param_node in params_list_node.named_children:
                          if param_node.type == 'parameter':
                              p_type_node = param_node.child_by_field_name('type')
                              p_name_node = param_node.child_by_field_name('name')
                              p_type = get_text(p_type_node, content_bytes) if p_type_node else '?'
                              p_name = get_text(p_name_node, content_bytes) if p_name_node else '?'
                              if p_name != '?': parameters.append({'name': p_name, 'type': p_type})
                     element['details']['parameters'] = parameters

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
        elem_type = element.get('type', 'unknown').replace('_declaration','').replace('_definition','').replace('_specifier','')
        start_line = element.get('start_line', '?')
        details = element.get('details', {})
        comment = element.get('comment')

        brief_desc = f"Brief description for {name}"
        detailed_desc = ""
        if comment:
             comment_lines = comment.strip().split('\n')
             first_non_empty_line_index = -1
             for i, line in enumerate(comment_lines):
                 if line.strip():
                     first_non_empty_line_index = i
                     break
             if first_non_empty_line_index != -1:
                 brief_desc = comment_lines[first_non_empty_line_index].strip()
                 detailed_desc_lines = comment_lines[first_non_empty_line_index + 1:]
                 start_detailed = 0
                 while start_detailed < len(detailed_desc_lines) and not detailed_desc_lines[start_detailed].strip():
                      start_detailed += 1
                 end_detailed = len(detailed_desc_lines) -1
                 while end_detailed >= start_detailed and not detailed_desc_lines[end_detailed].strip():
                      end_detailed -= 1
                 detailed_desc = "\n".join(detailed_desc_lines[start_detailed:end_detailed+1])
                 if not detailed_desc.strip(): detailed_desc = ""
                 elif not brief_desc: brief_desc = f"Brief description for {name}"
             else:
                 brief_desc = f"Brief description for {name}"
                 detailed_desc = ""

        safe_name = name.replace('`', '\\`').replace('*', '\\*').replace('_', '\\_')
        doxy_name = name

        md_lines.append(f"## `{safe_name}` ({elem_type})\n")
        md_lines.append(f"*Defined at: `{file_path.name}#L{start_line}`*\n")
        md_lines.append("```doxygen")
        md_lines.append(f"/*! \\{elem_type} {doxy_name}")
        md_lines.append(f" *  \\brief {brief_desc.replace(chr(10), ' ').replace(chr(13), '')}")

        if 'parameters' in details and details['parameters']:
            for param in details['parameters']:
                p_name = param.get('name', 'param')
                p_type = param.get('type', '')
                safe_p_name = p_name.replace('_', '\\_')
                safe_p_type = p_type.replace('<', '\\<').replace('>', '\\>')
                md_lines.append(f" *  \\param {safe_p_name} Parameter description. Type: `{safe_p_type}`")

        return_type = element.get('return_type')
        if return_type and return_type != 'void':
             safe_return_type = return_type.replace('<', '\\<').replace('>', '\\>')
             md_lines.append(f" *  \\return Return value description. Type: `{safe_return_type}`")
        elif elem_type in ['constructor', 'destructor']:
             pass

        if detailed_desc:
             md_lines.append(" *")
             for line in detailed_desc.split('\n'):
                  safe_line = line.replace('\\', '\\\\').replace('@', '\\@')
                  md_lines.append(f" *  {safe_line}")

        md_lines.append(" */")
        md_lines.append("```\n")

    return "\n".join(md_lines)

def log_parse_errors(node: Node, file_path: Path):
    """Recursively finds and logs ERROR nodes and *missing* nodes in the syntax tree."""
    if node.is_missing:
         start_line, start_col = node.start_point
         end_line, end_col = node.end_point
         logging.warning(
             f"Parse MISSING node in {file_path.relative_to(Path.cwd()).as_posix()} at "
             f"L{start_line+1}:C{start_col+1}-L{end_line+1}:C{end_col+1}. "
             f"Expected type: {node.type}."
         )
    elif node.type == 'ERROR':
        start_line, start_col = node.start_point
        end_line, end_col = node.end_point
        error_text_snippet = node.text.decode('utf-8', errors='replace')[:100]
        parent_type = node.parent.type if node.parent else "unknown"
        logging.warning(
            f"Parse ERROR in {file_path.relative_to(Path.cwd()).as_posix()} at "
            f"L{start_line+1}:C{start_col+1}-L{end_line+1}:C{end_col+1}. "
            f"Parent: {parent_type}. Snippet: '{error_text_snippet}...'"
        )
    for child in node.children:
        log_parse_errors(child, file_path)


# --- process_file: Made synchronous ---
def process_file(target_file_path: Path, root_dir: Path, output_dir: Path) -> Tuple[bool, int]:
    """Processes a single C++/C# file synchronously to generate documentation."""
    file_ext = target_file_path.suffix.lower()
    language: Optional[Language] = None
    extract_func = None
    lang_name = "unknown"
    elements_count = 0
    relative_file_path_str = "UnknownPath"
    try:
        relative_file_path_str = target_file_path.relative_to(root_dir).as_posix()
    except ValueError:
        relative_file_path_str = str(target_file_path)
        logging.warning(f"File {target_file_path} seems outside root {root_dir}. Using absolute path for logging.")

    logging.info(f"{relative_file_path_str}") # Log relative path

    if file_ext in CPP_EXTENSIONS: language, extract_func, lang_name = CPP_LANGUAGE, extract_documentation_elements_cpp, "cpp"
    elif file_ext in CSHARP_EXTENSIONS: language, extract_func, lang_name = CSHARP_LANGUAGE, extract_documentation_elements_csharp, "csharp"
    else:
        logging.warning(f"Skipping file with unsupported extension: {relative_file_path_str}")
        return False, 0

    if not language or not extract_func:
         logging.error(f"Skipping {relative_file_path_str}, language object or extract function not available for {lang_name}.")
         return False, 0

    try:
        MAX_FILE_SIZE_BYTES = 15 * 1024 * 1024 # 15 MB limit
        # Synchronous os.path.getsize
        file_size = os.path.getsize(target_file_path)
        if file_size == 0:
            logging.warning(f"Skipping empty file: {relative_file_path_str}")
            return True, 0
        if file_size > MAX_FILE_SIZE_BYTES:
             logging.warning(f"Skipping {relative_file_path_str}, file size ({file_size} bytes) exceeds limit ({MAX_FILE_SIZE_BYTES} bytes).")
             return False, 0

        # Synchronous read_bytes
        file_contents_bytes = target_file_path.read_bytes()
        try:
            # Try UTF-8 first, fallback with replacement
            file_contents_str = file_contents_bytes.decode('utf-8')
        except UnicodeDecodeError as ude:
            logging.error(f"Could not decode file {relative_file_path_str} as utf-8: {ude}. Trying latin-1 as fallback.", exc_info=False)
            try:
                file_contents_str = file_contents_bytes.decode('latin-1')
                logging.warning(f"Successfully decoded {relative_file_path_str} using latin-1.")
            except Exception as decode_err:
                 logging.error(f"Could not decode file {relative_file_path_str} with fallback encoding either: {decode_err}", exc_info=False)
                 return False, 0

        # --- Parsing (Synchronous) ---
        parser = Parser()
        parser.set_language(language)
        # Synchronous parser.parse
        tree = parser.parse(file_contents_bytes)

        if tree is None:
             logging.error(f"Tree-sitter parsing failed critically (returned None) for file: {relative_file_path_str}")
             return False, 0

        # --- Check errors/missing (Synchronous recursive check) ---
        has_errors_or_missing = False
        if tree.root_node:
            # Synchronous recursive check
            has_missing = find_missing_recursive(tree.root_node)
            if tree.root_node.has_error or has_missing:
                 has_errors_or_missing = True
                 logging.warning(f"Parsing resulted in errors or missing nodes for file: {relative_file_path_str}. Extraction will be attempted, logging details...")
                 try:
                     # Synchronous log_parse_errors
                     log_parse_errors(tree.root_node, target_file_path)
                 except Exception as log_err:
                     # Keep as warning as per user's last change
                     logging.warning(f"Failed log parse errors for {relative_file_path_str}: {log_err}")
        elif not tree.root_node:
             logging.error(f"Parsing produced no root node for file: {relative_file_path_str}")
             return False, 0 # Treat as failure if no root node

        # --- Extraction (Synchronous) ---
        elements = []
        try:
             # Synchronous extract_func call
             elements = extract_func(tree, file_contents_bytes)
             elements_count = len(elements)
             # Adjusted logging based on combined check
             if not elements and not has_errors_or_missing:
                 logging.info(f"No documentable elements found or extracted in {relative_file_path_str}.") # Use info, it's not necessarily an error
             elif not elements and has_errors_or_missing:
                  logging.warning(f"No documentable elements extracted in {relative_file_path_str} (likely due to parse errors or missing nodes).")

        except Exception as extract_err:
            logging.error(f"Error during element extraction from {relative_file_path_str}: {extract_err}", exc_info=True)
            return False, 0 # Treat extraction error as failure for this file


        # Only generate markdown if elements were found
        if not elements:
            # Return True if parsing was ok (even with errors/missing) but no elements found/extracted
            return True, 0 # Return success (parsed ok) but 0 elements

        # --- Documentation Generation (Synchronous) ---
        # Synchronous generate_doxygen_markdown call
        markdown_content = generate_doxygen_markdown(elements, target_file_path)

        # --- Output (Synchronous) ---
        relative_path_out = target_file_path.relative_to(root_dir) # Re-calculate relative path for output structuring
        output_file_path = output_dir / relative_path_out.with_suffix(DOC_SUFFIX)

        try:
            # Synchronous mkdir
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            # Synchronous file write
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
        except OSError as write_err:
            # Log relative output path for clarity
            try: rel_out_path = output_file_path.relative_to(output_dir)
            except ValueError: rel_out_path = output_file_path
            logging.error(f"Failed to write documentation file {rel_out_path}: {write_err}", exc_info=True)
            return False, elements_count # Mark as failed if write fails


        return True, elements_count # Success

    except FileNotFoundError:
        logging.error(f"File not found during processing: {relative_file_path_str}")
        return False, 0
    except OSError as os_err:
         logging.error(f"OS Error processing file {relative_file_path_str}: {os_err}", exc_info=True)
         return False, 0
    except Exception as e:
        # Catch-all for unexpected errors during processing of a single file
        logging.exception(f"Unexpected Error processing file {relative_file_path_str}: {e}")
        return False, 0

# --- Logging Setup ---
def setup_logging(log_dir: Path, files_log_path: Path, error_log_path: Path):
    """Configures logging to files only."""
    log_level = logging.INFO
    logger = logging.getLogger()
    logger.setLevel(log_level)
    if logger.hasHandlers():
        for handler in logger.handlers[:]:
            if isinstance(handler, (logging.FileHandler, logging.StreamHandler)):
                 if hasattr(handler, 'stream') and handler.stream in (sys.stdout, sys.stderr):
                      logger.removeHandler(handler)
                      continue
                 try:
                      handler.acquire()
                      handler.flush()
                      handler.close()
                 except (OSError, ValueError, RuntimeError): pass
                 finally:
                      try: handler.release()
                      except (ValueError, RuntimeError): pass
                 logger.removeHandler(handler)

    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Error creating log directory {log_dir}: {e}. Logging to files might fail.", file=sys.stderr)

    info_formatter = logging.Formatter('%(message)s')
    try:
        info_handler = logging.FileHandler(files_log_path, mode='w', encoding='utf-8')
        info_handler.setLevel(logging.INFO)
        info_handler.setFormatter(info_formatter)
        info_handler.addFilter(lambda record: record.levelno == logging.INFO)
        logger.addHandler(info_handler)
    except Exception as e:
        print(f"Warning: Failed to set up info log file at {files_log_path}: {e}", file=sys.stderr)

    error_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s')
    try:
        error_handler = logging.FileHandler(error_log_path, mode='w', encoding='utf-8')
        error_handler.setLevel(logging.WARNING)
        error_handler.setFormatter(error_formatter)
        logger.addHandler(error_handler)
    except Exception as e:
         print(f"Warning: Failed to set up error log file at {error_log_path}: {e}", file=sys.stderr)

    logging.info(f"Logging configured. Processed files will be listed in: {files_log_path.name}")
    logging.warning(f"Error/Warning logs will be recorded in: {error_log_path.name}")


# --- Main Execution: Made synchronous ---
def main(root_dir_str: str, output_dir_str: Optional[str]):
    """主函数 (Synchronous)"""
    try:
        root_dir = Path(root_dir_str).resolve(strict=True)
    except FileNotFoundError:
        print(f"Fatal: Source path '{root_dir_str}' not found or is not accessible.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Fatal: Error resolving source path '{root_dir_str}': {e}", file=sys.stderr)
        sys.exit(1)

    if not root_dir.is_dir():
        print(f"Fatal: Source path '{root_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(output_dir_str).resolve() if output_dir_str else DEFAULT_OUTPUT_DIR.resolve()

    # --- Configure Logging ---
    files_log_path = output_dir / FILES_LOG_FILENAME
    ignores_log_path = output_dir / IGNORES_LOG_FILENAME
    error_log_path = output_dir / ERROR_LOG_FILENAME
    setup_logging(output_dir, files_log_path, error_log_path) # Also creates output_dir
    # -------------------------

    logging.warning(f"Source directory: {root_dir}")
    logging.warning(f"Output directory: {output_dir}")

    # --- Load Languages ---
    try:
        load_or_build_languages()
        if not CPP_LANGUAGE or not CSHARP_LANGUAGE:
             logging.critical("Languages were not loaded correctly. Check logs above. Exiting.")
             print("CRITICAL: Languages were not loaded correctly. Check error log. Exiting.", file=sys.stderr)
             sys.exit(1)
    except SystemExit:
        sys.exit(1)
    except Exception as load_exc:
        logging.critical(f"Unhandled exception during language loading: {load_exc}", exc_info=True)
        print(f"CRITICAL: Unhandled exception during language loading: {load_exc}. Check error log. Exiting.", file=sys.stderr)
        sys.exit(1)
    # ----------------------

    # --- File Discovery ---
    logging.warning(f"Scanning for {', '.join(TARGET_EXTENSIONS)} files in {root_dir}...")
    print(f"Scanning for {', '.join(TARGET_EXTENSIONS)} files in {root_dir}...")
    target_files: List[Path] = []
    dirs_to_skip_specific: Set[str] = {'ThirdParty', 'Extras', 'thirdparty', 'Intermediate', 'Saved', 'Binaries', 'DerivedDataCache', 'Script'}
    dirs_to_skip_generic: Set[str] = {'.git', 'node_modules', 'bin', 'obj', 'Build', 'build', '.vs', '.vscode', '.idea', 'Backup'}
    skipped_dirs_count = 0
    processed_dirs_count = 0

    # find_files_sync remains the same internally as it was already synchronous
    def find_files_sync():
        nonlocal skipped_dirs_count, processed_dirs_count
        count = 0
        try:
             # Ensure ignores log directory exists (output dir created by setup_logging)
             with open(ignores_log_path, 'w', encoding='utf-8') as ignores_log_file:
                 ignores_log_file.write(f"# Skipped directories during scan of: {root_dir}\n")
                 ignores_log_file.write(f"# Skip patterns (specific): {dirs_to_skip_specific}\n")
                 ignores_log_file.write(f"# Skip patterns (generic): {dirs_to_skip_generic}\n")
                 ignores_log_file.write("-" * 20 + "\n")

                 for dirpath_str, dirnames, filenames in os.walk(root_dir, topdown=True, onerror=lambda err: logging.error(f"Scan error accessing {err.filename}: {err.strerror}", exc_info=False)):
                     dirpath = Path(dirpath_str)
                     processed_dirs_count += 1
                     original_dirnames = list(dirnames)
                     dirnames[:] = [d for d in dirnames if not (d in dirs_to_skip_specific or d in dirs_to_skip_generic or d.startswith('.'))]
                     skipped_here = set(original_dirnames) - set(dirnames)
                     if skipped_here:
                          skipped_dirs_count += len(skipped_here)
                          try:
                              rel_dirpath = dirpath.relative_to(root_dir).as_posix() or '.'
                              ignores_log_file.write(f"In: {rel_dirpath}/\n")
                              for skipped_name in sorted(skipped_here):
                                   ignores_log_file.write(f"  Skipping: {skipped_name}/\n")
                          except Exception as log_exc:
                               logging.error(f"Error writing skipped dir log for {dirpath}: {log_exc}")

                     for filename in filenames:
                         if any(filename.lower().endswith(ext) for ext in TARGET_EXTENSIONS):
                             try:
                                file_path = dirpath / filename
                                target_files.append(file_path)
                                count += 1
                                if count % 1000 == 0:
                                     print(f"Found {count} files...", end='\r', file=sys.stderr, flush=True)
                             except OSError as path_err:
                                  logging.warning(f"Could not process path {dirpath / filename}: {path_err}")

        except OSError as e:
             logging.error(f"Error accessing directory or writing to ignores log file '{ignores_log_path}': {e}. Scan may be incomplete.", exc_info=True)
             print(f"\nError during file scan: {e}. Check error log. Scan may be incomplete.", file=sys.stderr)
             # Recovery scan logic remains the same
             if not target_files:
                  logging.warning("Attempting recovery scan due to earlier error...")
                  print("Attempting recovery scan...", file=sys.stderr)
                  target_files.clear(); count = 0; processed_dirs_count = 0; skipped_dirs_count = 0
                  for dirpath_str, dirnames, filenames in os.walk(root_dir, topdown=True, onerror=lambda err: logging.error(f"Recovery scan error: {err}", exc_info=False)):
                      processed_dirs_count +=1
                      dirnames[:] = [d for d in dirnames if not (d in dirs_to_skip_specific or d in dirs_to_skip_generic or d.startswith('.'))]
                      for filename in filenames:
                           if any(filename.lower().endswith(ext) for ext in TARGET_EXTENSIONS):
                               try:
                                    file_path = Path(dirpath_str) / filename
                                    target_files.append(file_path)
                                    count += 1
                                    if count % 1000 == 0: print(f"Found {count} files (recovery scan)...", end='\r', file=sys.stderr, flush=True)
                               except OSError as path_err:
                                    logging.warning(f"Recovery scan: Could not process path {Path(dirpath_str) / filename}: {path_err}")
        finally:
             print(file=sys.stderr) # Ensure newline after progress indicator

    # Call find_files_sync directly (synchronously)
    find_files_sync()
    # ---------------------------------------------

    if not target_files:
        logging.warning("No target files found matching criteria.")
        print("No target files found matching criteria. Exiting.", file=sys.stderr)
        sys.exit(0)

    total_files = len(target_files)
    logging.warning(f"Scan complete. Found {total_files} target files. Starting documentation generation...")
    print(f"Scan complete. Found {total_files} target files. Starting documentation generation...")

    # --- File Processing (Synchronous Loop) ---
    successful_files = 0
    failed_files = 0
    total_elements = 0

    # Use standard tqdm if available
    iterable = target_files
    use_tqdm = False
    try:
        from tqdm import tqdm
        use_tqdm = True
        # Wrap the list directly with standard tqdm
        iterable = tqdm(target_files, desc="Generating docs", unit="file", file=sys.stderr, ncols=100, leave=True, dynamic_ncols=True)
        print("Using tqdm for progress.", file=sys.stderr) # Let user know tqdm is active
    except ImportError:
        print("Processing files (install tqdm for progress bar: pip install tqdm)...", file=sys.stderr)

    start_time = time.time()

    # Synchronous loop over files
    processed_count = 0
    for file_path in iterable: # Iterate over the (potentially tqdm-wrapped) list
        processed_count += 1
        try:
            # Direct call to synchronous process_file
            success, elements_count = process_file(file_path, root_dir, output_dir)
            if success:
                successful_files += 1
                total_elements += elements_count
            else:
                failed_files += 1
        except Exception as e:
            # Log unexpected errors during the processing of a single file
            try: # Safely get relative path for logging
                rel_path_str = file_path.relative_to(root_dir).as_posix()
            except ValueError:
                rel_path_str = str(file_path)
            logging.exception(f"Critical error during synchronous processing of file {rel_path_str}: {e}")
            failed_files += 1 # Count as failure

        # Update simple progress if not using tqdm (tqdm handles its own updates)
        # This simple progress indicator will be overwritten by tqdm if it's active.
        if not use_tqdm and processed_count % 100 == 0:
             print(f"Processed {processed_count}/{total_files} files ({failed_files} failures)...", end='\r', file=sys.stderr, flush=True)

    # Final newline for simple progress if it was used
    if not use_tqdm:
         print(file=sys.stderr) # Ensure newline

    end_time = time.time()
    duration = end_time - start_time
    # -----------------------------------------

    # --- Summary Output (Remains the same) ---
    summary_lines = [
        "\n--- Documentation Generation Summary ---",
        f"Source directory: {root_dir}",
        f"Output directory: {output_dir.resolve()}",
        f"Total directories scanned: {processed_dirs_count}",
        f"Total directories skipped: {skipped_dirs_count} (details in '{ignores_log_path.name}')",
        f"Total files found: {total_files}",
        f"Files successfully processed: {successful_files} (logged to '{files_log_path.name}')",
        f"Files failed or skipped during processing: {failed_files} (check '{error_log_path.name}')",
        f"Total documentation elements extracted: {total_elements}",
        f"Errors and warnings log: {error_log_path.resolve()}",
        f"Total processing time: {duration:.2f} seconds"
    ]

    # Print summary to stdout
    for line in summary_lines:
        print(line, file=sys.stdout)

    # Also log summary details to the error log for persistence
    logging.warning("--- Documentation Generation Summary ---")
    for line in summary_lines[1:]: # Skip the header line for log
        logging.warning(line)
    # -----------------------------------------


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Keep usage instructions on stderr
        script_name = Path(sys.argv[0]).name
        print(f"Usage: python {script_name} <path_to_project_root> [output_directory]", file=sys.stderr)
        print(f"  Example: python {script_name} C:\\Path\\To\\MyUnrealProject .\\docs_output", file=sys.stderr)
        print(f"  Example: python {script_name} /path/to/MyUnrealProject", file=sys.stderr)
        sys.exit(1)

    project_root_dir = sys.argv[1]
    output_dir_arg = sys.argv[2] if len(sys.argv) > 2 else None

    # Removed asyncio event loop policy setting

    run_success = False
    try:
        # Direct call to synchronous main function
        main(project_root_dir, output_dir_arg)
        run_success = True # Assume success if no exception bubbled up
    except KeyboardInterrupt:
         print("\nExecution interrupted by user.", file=sys.stderr)
         # Attempt to log interruption if logger is available
         try: logging.warning("Execution interrupted by user.")
         except Exception: pass
         sys.exit(130) # Standard exit code for Ctrl+C
    except SystemExit as sysexit:
         # Allow sys.exit() calls to propagate
         sys.exit(sysexit.code)
    except Exception as e:
        # Ensure critical errors are logged before exiting
        print(f"\nCRITICAL: Unhandled error during script execution: {e}", file=sys.stderr)
        try:
             # Use exc_info=True to get traceback in the log
             logging.critical(f"Unhandled error during script execution: {e}", exc_info=True)
        except Exception as log_err:
             print(f"CRITICAL: Also failed to log the critical error: {log_err}", file=sys.stderr)
        sys.exit(1) # Exit with error code 1 for generic unhandled errors

    # Optional: Exit with non-zero code if there were file processing failures reported in summary
    # if failed_files > 0:
    #     print(f"Warning: {failed_files} file(s) failed during processing.", file=sys.stderr)
    #     sys.exit(1)
    # else:
    #     sys.exit(0)
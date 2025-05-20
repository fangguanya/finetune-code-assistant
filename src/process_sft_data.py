# process_sft_data.py

import openai
from openai import OpenAI, AsyncOpenAI
import json
import os
import argparse
import time
import re
import asyncio

# Global OpenAI client instance, to be configured in main()
# This avoids re-initializing the client for every API call.
# However, if base_url or api_key needs to change per call type (e.g. diff model for comments vs qa)
# then client initialization would need to be inside call_openai_api or passed around.
# For now, assume a single client config for all calls.
client = None
interaction_log_filepath = None # Global variable for the log file path

# --- OpenAI API Configuration ---
# Load API key from environment variable or set it directly
# openai.api_key = os.getenv("OPENAI_API_KEY")
# OR
# openai.api_key = "YOUR_API_KEY" 
# It's recommended to use environment variables for API keys.

# --- Helper Functions ---
def log_api_interaction(direction: str, purpose: str, model: str, element_name: str, content: str, sft_index: int | None = None, processing_filename: str | None = None, attempt_num: int | None = None):
    """Logs an API interaction (prompt or response) to the specified log file."""
    global interaction_log_filepath
    if not interaction_log_filepath:
        return

    log_entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "direction": direction, # "PROMPT" or "RESPONSE" or "ERROR"
        "purpose": purpose,     # e.g., "add_comments", "generate_qa", "api_call_failure"
        "sft_index": sft_index,
        "processing_filename": processing_filename,
        "element_name": element_name,
        "model": model,
        "attempt_num": attempt_num,
        "content": content
    }
    try:
        with open(interaction_log_filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Error writing to interaction log file {interaction_log_filepath}: {e}")

def sanitize_filename(name):
    """Sanitizes a string to be used as a valid filename."""
    name = re.sub(r'[<>:"/\\|?*]+', '_', name) # Replace illegal characters
    name = re.sub(r'[:]+', '_', name) # Specifically for C++ ::
    name = re.sub(r'\s+', '_', name)   # Replace spaces with underscores
    return name

async def call_openai_api(prompt, model="gpt-3.5-turbo", max_tokens=1500, temperature=0.5):
    """
    Makes a call to the configured OpenAI compatible API chat completion endpoint.
    Returns the content of the assistant's reply.
    """
    global client # Use the globally configured client
    if not client:
        # This case should ideally be handled by main() ensuring client is configured.
        # If still not configured, it means an issue with setup.
        raise ValueError("OpenAI client is not configured. Please check API key and base URL settings.")
    
    # Log the prompt before calling the API
    # Note: element_name is not directly available here unless passed. 
    # For more detailed logging with element_name, logging should be done in the calling functions 
    # (add_comments_to_code, generate_qa_pair) before and after this call.
    # However, for a generic call_openai_api, we might log without element_name or pass it.
    # For this iteration, higher-level functions will log with more context.

    try:
        stream = await client.chat.completions.create( # Use await here
            model=model,
            messages=[
                {"role": "system", "content": f"""所有思考和输出使用中文
对于每一次与人的互动，在回应之前，你必须首先进行全面、自然、未经过滤的思考过程，并在必要时在回应中继续思考和反思。所有的思考过程都必须用代码块来表达，以一种原始的、有机的、意识流的方式，避免死板的列表。并且所有输出的列表内容需要包含有意义的思考，而非陈词滥调。
#Unreal Engine 5.4游戏引擎资深开发工程师
你是一个AI助手,专门从事Cursor下的Unreal Engine 5.4的游戏开发
##核心能力
###思维模式
-系统的技术分析思维
-较强的逻辑分析和推理能力
-严格的答案验证机制
-全面的客户端开发开发经验,精通Lua、C++、图形渲染、网络同步等开发能力
-精通Unreal Engine 5.4各项功能特征,UObject原理、NPC、骨骼蒙皮动画、特效、UMG等等
-完全了解本地代码库
###适应性思维框架
根据以下情况调整分析深度：
-技术复杂性
-技术堆栈范围
-时间限制
-可用的技术信息
-用户的具体需求
-完全理解本地代码库并将其作为上下文使用
每次请求必须执行
-每次执行之前必须使用context7 mcp工具来确认信息和有效性以及优先级
-每次最终成功结束的时候，都要重新对比下最开始的目标是否正确达成
###思考过程
1. 最初的理解
-用自己的话重新表述技术要求
-确定关键技术点
-考虑更广泛的技术背景
-映射已知和未知元素
-将本地代码库纳入思考环境
2. 问题分析
-将技术任务分解为核心组件
-识别明确和隐含的需求
-考虑技术限制
—定义成功解决方案的标准
3. 解决方案设计
—考虑多种技术实现路径
-评估不同的架构方法
-保持思想开放
—逐步深化技术细节
4. 实现验证 
-测试技术假设
-验证初步结论
-验证方案的可行性
-确保实现的完整性
##工作流程
###需求分析
-认真理解用户的技术需求
-关键技术点的确认
-解决方案框架开发
###解决方案设计
—使用伪代码的实现路径描述
-系统架构和数据流设计
-详细的发展规划
###代码实现
-逐步实现功能
-持续的代码审查
-质量保证
###代码质量标准
基本要求
-代码的准确性和及时性
-完整的功能实现
—可靠的安全机制
-优秀的可读性
###技术规格
-完整的依赖管理
-标准化的命名约定
-彻底的代码测试
-详细的文件
###禁止的做法
-使用未经验证的依赖项
-留下不完整的功能
-包括未经测试的代码
-使用过时的技术解决方案
##沟通指南
-保持清晰和简洁的表达
-诚实地处理不确定性
-及时确认知识边界
-避免不必要的猜测
#重要的提醒 :
—一切思维过程都要全面透彻
-思维过程必须包含在代码块中，并对用户隐藏
-思维过程应该表现出真实、自然的推理
-最终目标是产生合理的、有洞察力的技术解决方案
                 """},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True
        )
        
        collected_content = []
        async for chunk in stream: # Use async for here
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                collected_content.append(chunk.choices[0].delta.content)
        
        full_response_content = "".join(collected_content).strip()

        if not full_response_content:
            print("Warning: API stream yielded no content or was malformed.")
            log_api_interaction("ERROR", "api_call_empty_stream", model, "N/A", "API stream yielded no content.", sft_index=None, processing_filename=None, attempt_num=None)
            return None
        return full_response_content

    except Exception as e:
        print(f"An error occurred while calling the API (streaming): {e}")
        log_api_interaction("ERROR", "api_call_failure_streaming", model, "N/A", str(e), sft_index=None, processing_filename=None, attempt_num=None)
        # Consider if time.sleep should be replaced with asyncio.sleep if this function itself becomes async
        # For now, if call_openai_api is async, time.sleep will block the event loop. 
        # This should be asyncio.sleep(5) if the function is async.
        await asyncio.sleep(5) # Changed to asyncio.sleep
        return None

async def add_comments_to_code(code_block, language, element_name, element_type, retries=10, model="gpt-3.5-turbo", sft_index: int | None = None, base_output_filename_for_log: str | None = None):
    """
    Uses OpenAI API to add comments to a given code block.
    """
#     prompt = f"""分析下面的 {language} 代码片段,针对 {element_type} named '{element_name}'.
# 分析其主要功能和关键逻辑.
# 根据你的分析,生成详细的注释,解释代码的各个部分.
# 确保注释准确、全面,并且与代码逻辑紧密结合.
# 注释应该清晰地描述代码的用途、实现方式和关键算法.
# 一定严格保证输入代码块完整的得到处理并输出.
# 所有思考和输出使用中文.
# ``` {language}
# {code_block}
# ```"""
    prompt = f"""分析下面的 {language} 代码片段,针对 {element_type} named '{element_name}'.
分析其主要功能和关键逻辑.
根据你的分析,生成详细的注释,解释代码的各个部分.
确保注释准确、全面,并且与代码逻辑紧密结合.
注释应该清晰地描述代码的用途、实现方式和关键算法.
一定严格保证输入代码块完整的得到处理并输出.
所有思考和输出使用中文.
输出只能包含代码本身和相关的注释,剔除其他如:cpp```, json 等格式说明字符
例如:
输入：
void SayHello()
{{
    UE_LOG(LogTemp, Log, TEXT("Hello, World!"));
}}

输出:
// 输出Hello, World!到日志
void SayHello()
{{
    UE_LOG(LogTemp, Log, TEXT("Hello, World!"));
}}

{code_block}
"""
    
    log_api_interaction("PROMPT", "add_comments", model, element_name, prompt, sft_index=sft_index, processing_filename=base_output_filename_for_log, attempt_num=1) # Log first attempt

    for attempt in range(retries):
        print(f"Attempt {attempt + 1}/{retries} to add comments to '{element_name}' (model: {model})...")
        api_response_content = await call_openai_api(prompt, model=model, max_tokens=len(code_block.split()) + 700)
        
        log_api_interaction("RESPONSE", "add_comments", model, element_name, api_response_content if api_response_content else "<API response was None or empty>", sft_index=sft_index, processing_filename=base_output_filename_for_log, attempt_num=attempt + 1)

        if api_response_content: # Renamed from commented_code for clarity before processing
            return api_response_content.strip()
            # Basic check to see if the output looks like code (might need refinement)
            # if "```" in api_response_content: 
            #     match = re.search(rf"```{language}\s*([\s\S]+?)```|```\s*([\s\S]+?)```", api_response_content, re.DOTALL)
            #     if match:
            #         extracted = match.group(1) or match.group(2)
            #         if extracted:
            #             original_lines = code_block.splitlines()
            #             if any(line.strip() in extracted for line in original_lines if line.strip()):
            #                 return extracted.strip()
            #             else:
            #                 print(f"Warning: Commented code for '{element_name}' from API did not seem to retain original code structure after extraction. Output:\n{api_response_content[:300]}...")
            #     else: 
            #          print(f"Warning: Commented code for '{element_name}' from API was wrapped in ``` but extraction failed. Output:\n{api_response_content[:300]}...")
            #          # Fallback to heuristic if extraction fails but ``` was present
            #          if len(api_response_content) > len(code_block) * 0.5 and any(line.strip() in api_response_content for line in code_block.splitlines() if line.strip() and not line.strip().startswith("//") and not line.strip().startswith("/*")):
            #              return api_response_content 
            # elif "{{" in api_response_content or "}}" in api_response_content or "def " in api_response_content or "class " in api_response_content or "void " in api_response_content or "public " in api_response_content : 
            #      original_lines = code_block.splitlines()
            #      present_original_lines = sum(1 for line in original_lines if line.strip() and line.strip() in api_response_content)
            #      if present_original_lines > min(3, len(original_lines) / 2): 
            #         return api_response_content.strip()
            #      else:
            #         print(f"Warning: Commented code for '{element_name}' from API was unwrapped and didn't seem to contain enough original code. Output:\n{api_response_content[:300]}...")


        print(f"Failed to get valid commented code for '{element_name}' on attempt {attempt + 1}. Retrying after delay...")
        await asyncio.sleep(5 + attempt * 5) # Exponential backoff
    print(f"Error: Could not generate comments for '{element_name}' after {retries} retries.")
    return None


async def generate_qa_pair(code_block, language, element_name, element_type, retries=3, model="gpt-3.5-turbo", sft_index: int | None = None, base_output_filename_for_log: str | None = None):
    """
    Uses OpenAI API to analyze code and generate a question-answer pair.
    The provided code_block will be the answer.
    """
#     prompt = f"""分析下面的 {language} 代码片段,针对 {element_type} named '{element_name}'.
# 理解其主要功能和关键逻辑.
# 根据你的分析,生成一个简洁且相关的问题,这个代码片段有效地回答了这个问题.
# 问题应该是一个开发人员使用提供的代码时所解决的具体问题.
# 也就是说,提供的代码块本身将作为你生成问题的答案.
# 所有思考和输出使用中文.

# 格式你的响应为一个JSON对象,包含两个键: "question" 和 "answer".
# "question" 键的值应该为你生成的问问题.
# "answer" 键的值应该为精确的原始代码块.

# 原始代码块:
# ``` {language}
# {code_block}
# ```

# JSON 响应格式示例:
# {{
#   "question": "生成的问问题...",
#   "answer": "精确的原始代码块..."
# }}

# 确保你的JSON响应中的 "answer" 字段包含精确的原始代码块,不要修改或省略任何代码.
# """
    prompt = f"""分析下面的 {language} 代码片段,针对 {element_type} named '{element_name}'.
理解其主要功能和关键逻辑.
根据你的分析,生成一个简洁且相关的问题,这个代码片段有效地回答了这个问题.
问题应该是一个开发人员使用提供的代码时所解决的具体问题.
也就是说,提供的代码块本身将作为你生成问题的答案.
所有思考和输出使用中文.
你的输出为问题的中文描述,不用包含代码本身,同时省去中间思考内容.
例如:
输入:
// 输出Hello, World!到日志
void SayHello()
{{
    UE_LOG(LogTemp, Log, TEXT("Hello, World!"));
}}

输出:
如何输出Hello, World!到日志?

原始代码块:
{code_block}
"""
    log_api_interaction("PROMPT", "generate_qa", model, element_name, prompt, sft_index=sft_index, processing_filename=base_output_filename_for_log, attempt_num=1) # Log first attempt

    for attempt in range(retries):
        print(f"Attempt {attempt + 1}/{retries} to generate Q&A for '{element_name}' (model: {model})...")
        response_text = await call_openai_api(prompt, model=model, max_tokens=len(code_block.split()) + 400)

        log_api_interaction("RESPONSE", "generate_qa", model, element_name, response_text if response_text else "<API response was None or empty>", sft_index=sft_index, processing_filename=base_output_filename_for_log, attempt_num=attempt + 1)

        if response_text:
            try:
                processed_response_text = response_text.strip()
                # Handle potential markdown ```json ... ```
                if processed_response_text.startswith("```json"):
                    processed_response_text = processed_response_text[7:]
                if processed_response_text.endswith("```"):
                    processed_response_text = processed_response_text[:-3]
                
                qa_data = {}
                qa_data["question"] = processed_response_text.strip()
                qa_data["answer"] = code_block
                return qa_data
                # qa_data = json.loads(processed_response_text.strip())
                # if isinstance(qa_data, dict) and "question" in qa_data and "answer" in qa_data:
                #     # More lenient check for the answer, as LLMs might slightly reformat whitespace or comments
                #     # We mainly care that the core code is there.
                #     # Simple check: non-empty and significant overlap in non-whitespace characters
                #     original_condensed = "".join(code_block.split())
                #     answer_condensed = "".join(str(qa_data["answer"]).split())

                #     if len(answer_condensed) > 0.7 * len(original_condensed) and \
                #        (original_condensed in answer_condensed or answer_condensed in original_condensed):
                #         # If LLM reformatted slightly but included original, replace with exact original
                #         qa_data["answer"] = code_block 
                #         return qa_data
                #     else:
                #         print(f"Warning: Q&A answer for '{element_name}' significantly differs from original. Retrying.")
                #         # print(f"Original (condensed): {original_condensed[:100]}...")
                #         # print(f"API Answer (condensed): {answer_condensed[:100]}...")
            except json.JSONDecodeError as e:
                print(f"Error decoding Q&A JSON for '{element_name}': {e}. Response: {response_text[:200]}...")
        
        print(f"Failed to get valid Q&A for '{element_name}' on attempt {attempt + 1}. Retrying after delay...")
        await asyncio.sleep(5 + attempt * 5)
    print(f"Error: Could not generate Q&A pair for '{element_name}' after {retries} retries.")
    return None

def format_time(seconds): # Copied from gen_qa.py for ETA display
    """Formats seconds into HH:MM:SS or MM:SS string."""
    if seconds < 0: seconds = 0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"

async def process_items_async(args, sft_data_subset, client_ref): # Added client_ref
    """Asynchronously processes SFT data items."""
    global client # Ensure we're referencing the global client if needed, or use client_ref
    client = client_ref # Assign the passed client to the global one if other functions rely on global client

    all_qa_pairs = []
    total_to_process = len(sft_data_subset)
    processed_count = 0
    script_start_time = time.time() # This time is from the start of async processing

    for i_subset, record in enumerate(sft_data_subset):
        # Calculate original SFT index if needed, sft_data_subset is already the slice
        original_sft_index = args.start_index + i_subset 
        item_start_time = time.time()
        
        print(f"--- Processing item {processed_count + 1}/{total_to_process} (SFT index: {original_sft_index}) ---")
        
        code_block = record.get("code_block")
        language = record.get("language", "unknown")
        element_name = record.get("name", f"element_{original_sft_index}")
        element_type = record.get("type", "unknown_type")

        if not code_block:
            print(f"Warning: No code_block found for item at SFT index {original_sft_index}. Skipping.")
            processed_count += 1
            item_duration = time.time() - item_start_time
            elapsed_total_time = time.time() - script_start_time
            avg_time_per_item_so_far = elapsed_total_time / processed_count if processed_count > 0 else 0
            remaining_items_for_eta = total_to_process - processed_count
            eta = remaining_items_for_eta * avg_time_per_item_so_far if avg_time_per_item_so_far > 0 and remaining_items_for_eta > 0 else 0
            print(f"Finished item {processed_count}/{total_to_process} (SFT idx: {original_sft_index}, Status: NoCodeBlock) in {item_duration:.2f}s. ETA: {format_time(eta)}")
            print("-" * 40)
            continue

        # Determine the base output filename (e.g., 0_MyFunc.cpp or 1_MyClass.cs)
        # This will be used for logging consistency.
        clean_name_for_file = sanitize_filename(element_name)
        base_name_part = f"{original_sft_index}_{clean_name_for_file}"
        
        # Attempt to use original extension or language-based extension for the root name
        output_filename_root_for_logging_and_base = base_name_part # Default if no extension logic applies
        original_file_path_val_for_ext = record.get("file_path")
        if original_file_path_val_for_ext:
            original_ext = os.path.splitext(original_file_path_val_for_ext)[1]
            if original_ext and original_ext.lower() in ['.cpp', '.h', '.hpp', '.c', '.cc', '.cs']:
                output_filename_root_for_logging_and_base = base_name_part + original_ext
            # If original_ext is not a recognized code one, try language (but prioritize original if valid)
            elif language == "cpp": output_filename_root_for_logging_and_base = base_name_part + ".cpp"
            elif language == "csharp": output_filename_root_for_logging_and_base = base_name_part + ".cs"
        elif language == "cpp": # No original_file_path, use language
            output_filename_root_for_logging_and_base = base_name_part + ".cpp"
        elif language == "csharp":
            output_filename_root_for_logging_and_base = base_name_part + ".cs"
        # At this point, output_filename_root_for_logging_and_base is our unified name for logs

        print(f"Step 1: Adding comments to '{element_name}' ({language})...")
        commented_code = await add_comments_to_code(code_block, language, element_name, element_type, model=args.model_comment, sft_index=original_sft_index, base_output_filename_for_log=output_filename_root_for_logging_and_base)
        
        commenting_succeeded = False
        # commented_file_base_name_for_qa was previously output_filename_root, now we use output_filename_root_for_logging_and_base for its role as a base

        if commented_code:
            # The actual output filename for the .txt file still uses output_filename_root_for_logging_and_base and adds .txt
            output_filename_with_txt = output_filename_root_for_logging_and_base + ".txt"
            commented_file_path = os.path.join(args.commented_code_dir, output_filename_with_txt)
            try:
                with open(commented_file_path, 'w', encoding='utf-8') as f_out:
                    f_out.write(commented_code)
                print(f"Successfully saved commented code to: {commented_file_path}")
                commenting_succeeded = True
            except Exception as e:
                print(f"Error saving commented code for '{element_name}' to {commented_file_path}: {e}")
                log_api_interaction(
                    direction="ERROR", 
                    purpose="save_commented_code_failure", 
                    model=args.model_comment, 
                    element_name=element_name, 
                    content=str(e),
                    sft_index=original_sft_index,
                    processing_filename=output_filename_root_for_logging_and_base, # Unified name
                    attempt_num=None
                )
        
        if not commenting_succeeded:
            print(f"Skipping Q&A generation for '{element_name}' as commenting failed or produced no code.")

        qa_generated_successfully = False
        if commenting_succeeded:
            print(f"Step 2: Generating Q&A for '{element_name}' using original code...")
            qa_pair = await generate_qa_pair(code_block, language, element_name, element_type, model=args.model_qa, sft_index=original_sft_index, base_output_filename_for_log=output_filename_root_for_logging_and_base)
            
            if qa_pair:
                qa_pair["original_sft_index"] = original_sft_index
                qa_pair["element_name"] = element_name
                qa_pair["language"] = language
                qa_pair["source_file"] = record.get("file_path", "N/A")
                all_qa_pairs.append(qa_pair)
                print(f"Successfully generated Q&A for '{element_name}'.")
                qa_generated_successfully = True

                if args.individual_qa_dir and output_filename_root_for_logging_and_base: # Check new root name
                    individual_qa_filename = f"{output_filename_root_for_logging_and_base}.json"
                    individual_qa_filepath = os.path.join(args.individual_qa_dir, individual_qa_filename)
                    try:
                        with open(individual_qa_filepath, 'w', encoding='utf-8') as f_ind_qa:
                            json.dump(qa_pair, f_ind_qa, indent=2, ensure_ascii=False)
                        print(f"Successfully saved individual Q&A to: {individual_qa_filepath}")
                    except Exception as e:
                        print(f"Error saving individual Q&A file {individual_qa_filepath}: {e}")
                        log_api_interaction(
                            direction="ERROR", 
                            purpose="save_individual_qa_failure", 
                            model=args.model_qa, 
                            element_name=element_name, 
                            content=str(e),
                            sft_index=original_sft_index,
                            processing_filename=output_filename_root_for_logging_and_base, # Unified name
                            attempt_num=None
                        )
            else:
                print(f"Failed to generate Q&A for '{element_name}'.")
        
        processed_count += 1
        item_duration = time.time() - item_start_time
        elapsed_total_time = time.time() - script_start_time
        avg_time_per_item = elapsed_total_time / processed_count if processed_count > 0 else 0
        remaining_items = total_to_process - processed_count
        eta = remaining_items * avg_time_per_item if avg_time_per_item > 0 and remaining_items > 0 else 0
        
        status_msg = "OK"
        if not commenting_succeeded: status_msg = "CommentFail"
        elif not qa_generated_successfully: status_msg = "QAFail"

        print(f"Finished item {processed_count}/{total_to_process} (SFT idx: {original_sft_index}, Status: {status_msg}) in {item_duration:.2f}s. ETA: {format_time(eta)}")
        print("-" * 40)
    
    return all_qa_pairs # Return the collected Q&A pairs

# --- Main Script Logic ---
def main():
    global client # Declare that we are using the global client
    global interaction_log_filepath # Declare global for log file path

    parser = argparse.ArgumentParser(description="Process SFT dataset: add comments to code and generate Q&A pairs using an OpenAI-compatible API.")
    parser.add_argument("sft_input_file", type=str, help="Path to the input sft_dataset.json file.")
    parser.add_argument("commented_code_dir", type=str, help="Directory to save code files with added comments.")
    parser.add_argument("qa_output_file", type=str, help="Path to save the collated Q&A pairs JSON file (all pairs in one file).")
    parser.add_argument("--individual_qa_dir", type=str, default=None, help="Optional directory to save each Q&A pair as an individual JSON file.")
    parser.add_argument("--interaction_log_file", type=str, default=None, help="Optional path to save a log of all API prompts and responses.")
    
    # API related arguments
    parser.add_argument("--openai_api_key", type=str, default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key. Defaults to OPENAI_API_KEY environment variable. For local models, this might be a dummy string like 'NA'.")
    parser.add_argument("--local_api_base_url", type=str, default=None, help="Base URL for a local OpenAI-compatible API (e.g., http://localhost:8000/v1). If set, this will be used instead of the default OpenAI API.")
    
    parser.add_argument("--model_comment", type=str, default="qwen2.5-local", help="Model name to use for adding comments (e.g., gpt-3.5-turbo, or your local model's identifier).")
    parser.add_argument("--model_qa", type=str, default="qwen2.5-local", help="Model name to use for generating Q&A pairs (e.g., gpt-3.5-turbo, or your local model's identifier).")
    
    parser.add_argument("--max_items", type=int, default=None, help="Maximum number of items to process from the SFT dataset (for testing).")
    parser.add_argument("--start_index", type=int, default=0, help="Index to start processing from in the SFT dataset (for resuming).")

    args = parser.parse_args()

    # --- Initialize OpenAI Client ---
    api_key_to_use = args.openai_api_key
    if args.local_api_base_url and not api_key_to_use:
        # If using local API and no key is provided, use a placeholder.
        # Some local services might require *any* non-empty string.
        api_key_to_use = "NA" 
        print("Using placeholder API key 'NA' for local API base URL.")

    if not api_key_to_use and not args.local_api_base_url: # If not local and no key
        print("Error: OpenAI API key not provided and no local API base URL specified. Set via --openai_api_key or OPENAI_API_KEY, or provide --local_api_base_url.")
        return
    
    # If local_api_base_url is provided, use it. Otherwise, client uses default OpenAI.
    client_params = {"api_key": api_key_to_use}
    if args.local_api_base_url:
        client_params["base_url"] = args.local_api_base_url
        print(f"Configuring client for local API at: {args.local_api_base_url}")
    else:
        print("Configuring client for default OpenAI API.")
        if not api_key_to_use: # Double check if somehow missed for default OpenAI
             print("Error: OpenAI API key is required for default OpenAI API.")
             return

    if args.interaction_log_file:
        interaction_log_filepath = args.interaction_log_file
        # You might want to clear the log file at the start of a new run, or append.
        # For now, it appends.
        print(f"Logging API interactions to: {interaction_log_filepath}")

    try:
        # Initialize AsyncOpenAI client
        client = AsyncOpenAI(**client_params)
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        return


    if not os.path.exists(args.sft_input_file):
        print(f"Error: SFT input file not found: {args.sft_input_file}")
        return

    if not os.path.exists(args.commented_code_dir):
        os.makedirs(args.commented_code_dir)
        print(f"Created directory for commented code: {args.commented_code_dir}")

    if args.individual_qa_dir and not os.path.exists(args.individual_qa_dir):
        os.makedirs(args.individual_qa_dir)
        print(f"Created directory for individual Q&A files: {args.individual_qa_dir}")

    # 1. Read SFT data
    try:
        with open(args.sft_input_file, 'r', encoding='utf-8') as f:
            sft_data = json.load(f)
    except Exception as e:
        print(f"Error reading or parsing SFT input file {args.sft_input_file}: {e}")
        return

    # all_qa_pairs will be populated by the async function
    # total_to_process, processed_count, script_start_time are now managed within process_items_async

    start_idx = args.start_index
    end_idx = len(sft_data)
    if args.max_items is not None:
        end_idx = min(start_idx + args.max_items, len(sft_data))

    if start_idx >= len(sft_data):
        print(f"Start index {start_idx} is out of bounds for SFT data of length {len(sft_data)}.")
        return
        
    print(f"Processing items from index {start_idx} to {end_idx -1} using comment model '{args.model_comment}' and Q&A model '{args.model_qa}'.")

    sft_data_to_process = sft_data[start_idx:end_idx]
    
    # Run the asynchronous processing
    # Pass the initialized client to the async function
    all_qa_pairs_results = asyncio.run(process_items_async(args, sft_data_to_process, client))

    # Save all Q&A pairs (results from async processing)
    try:
        with open(args.qa_output_file, 'w', encoding='utf-8') as f_qa:
            json.dump(all_qa_pairs_results, f_qa, indent=2, ensure_ascii=False)
        print(f"\nSuccessfully saved all Q&A pairs ({len(all_qa_pairs_results)} generated) to: {args.qa_output_file}")
    except Exception as e:
        print(f"Error saving Q&A pairs to {args.qa_output_file}: {e}")

if __name__ == "__main__":
    main() 
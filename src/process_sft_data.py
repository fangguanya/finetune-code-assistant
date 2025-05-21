# process_sft_data.py

import openai
from openai import OpenAI, AsyncOpenAI
import json
import os
import argparse
import time
import re
import asyncio
import hashlib

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

async def add_comments_to_code(code_block, language, element_name, element_type, model, sft_index, base_output_filename_for_log, max_retries, retry_delay):
    """
    Uses OpenAI API to add comments to a given code block.
    """
    prompt = f"""分析下面的 {language} 代码片段,针对 {element_type} named '{element_name}'.
分析其主要功能和关键逻辑.
根据你的分析,生成详细的注释,解释代码的各个部分.
确保注释准确、全面,并且与代码逻辑紧密结合.
注释应该清晰地描述代码的用途、实现方式和关键算法.
一定严格保证输入代码块完整的得到处理并输出.
所有思考和输出使用中文.
输出只能包含代码本身和相关的注释,剔除其他如:cpp```, json 等格式说明字符
例如:
输入:
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
    
    # Initial prompt log is done by the first iteration of the loop if attempt == 0

    for attempt in range(max_retries):
        current_attempt_num = attempt + 1
        log_purpose = "add_comments" if attempt == 0 else "add_comments_retry"
        
        log_api_interaction("PROMPT", log_purpose, model, element_name, prompt, sft_index=sft_index, processing_filename=base_output_filename_for_log, attempt_num=current_attempt_num)
        
        print(f"Attempt {current_attempt_num}/{max_retries} to add comments to '{element_name}' (model: {model})...")
        api_response_content = await call_openai_api(prompt, model=model, max_tokens=len(code_block.split()) + 700)
        
        log_api_interaction("RESPONSE", "add_comments", model, element_name, api_response_content if api_response_content else "<API response was None or empty>", sft_index=sft_index, processing_filename=base_output_filename_for_log, attempt_num=current_attempt_num)

        if api_response_content: 
            return api_response_content.strip()

        if current_attempt_num < max_retries:
            print(f"Failed to get valid commented code for '{element_name}' on attempt {current_attempt_num}/{max_retries}. Retrying after {retry_delay}s delay...")
            await asyncio.sleep(retry_delay)
        else:
            print(f"Error: Could not generate comments for '{element_name}' after {max_retries} retries.")
            break # Exit loop after max retries
            
    return None # Explicitly return None if all retries fail


async def generate_qa_pair(code_block, language, element_name, element_type, model, sft_index, base_output_filename_for_log, max_retries, retry_delay):
    """
    Uses OpenAI API to analyze code and generate a question-answer pair.
    The provided code_block will be the answer.
    """
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
    # Initial prompt log is done by the first iteration of the loop if attempt == 0
    
    for attempt in range(max_retries):
        current_attempt_num = attempt + 1
        log_purpose = "generate_qa" if attempt == 0 else "generate_qa_retry"

        log_api_interaction("PROMPT", log_purpose, model, element_name, prompt, sft_index=sft_index, processing_filename=base_output_filename_for_log, attempt_num=current_attempt_num)

        print(f"Attempt {current_attempt_num}/{max_retries} to generate Q&A for '{element_name}' (model: {model})...")
        response_text = await call_openai_api(prompt, model=model, max_tokens=len(code_block.split()) + 400)

        log_api_interaction("RESPONSE", "generate_qa", model, element_name, response_text if response_text else "<API response was None or empty>", sft_index=sft_index, processing_filename=base_output_filename_for_log, attempt_num=current_attempt_num)

        if response_text:
            try:
                processed_response_text = response_text.strip()
                if processed_response_text.startswith("```json"):
                    processed_response_text = processed_response_text[7:]
                if processed_response_text.endswith("```"):
                    processed_response_text = processed_response_text[:-3]
                
                qa_data = {}
                qa_data["question"] = processed_response_text.strip()
                qa_data["answer"] = code_block
                return qa_data 
            except json.JSONDecodeError as e:
                print(f"Error decoding Q&A JSON for '{element_name}': {e}. Response: {response_text[:200]}...")
        
        if current_attempt_num < max_retries:
            print(f"Failed to get valid Q&A for '{element_name}' on attempt {current_attempt_num}/{max_retries}. Retrying after {retry_delay}s delay...")
            await asyncio.sleep(retry_delay)
        else:
            print(f"Error: Could not generate Q&A pair for '{element_name}' after {max_retries} retries.")
            break # Exit loop after max retries

    return None # Explicitly return None if all retries fail

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

# --- Progress Saving/Loading Helper Functions ---
def calculate_file_hash(filepath):
    """Calculates SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except FileNotFoundError:
        # This can happen if sft_input_file is specified incorrectly by user
        # Or if progress refers to a no-longer-existing sft_input_file path
        print(f"Warning: File not found for hashing: {filepath}")
        return None
    except Exception as e:
        print(f"Error calculating hash for {filepath}: {e}")
        return None

def save_progress(progress_filepath, sft_input_filepath, last_processed_sft_index):
    """Saves the last processed SFT index and a hash of the SFT input file."""
    sft_input_hash = calculate_file_hash(sft_input_filepath)
    # If sft_input_hash is None, we still save progress but hash verification will fail on load
    # if the original sft file issue isn't resolved.
    
    progress_data = {
        "last_processed_sft_index": last_processed_sft_index,
        "sft_input_file_hash": sft_input_hash,
        "sft_input_filepath_for_reference": sft_input_filepath # For human readability
    }
    try:
        with open(progress_filepath, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=2)
        # Minor adjustment to log message for clarity
        print(f"Progress saved: SFT index {last_processed_sft_index} marked as processed. File: {progress_filepath}") 
    except Exception as e:
        print(f"Error saving progress to {progress_filepath}: {e}")

def load_progress(progress_filepath, current_sft_input_filepath):
    """
    Loads the last processed SFT index if the progress file exists and 
    the SFT input file hash matches.
    Returns the last_processed_sft_index or None.
    """
    if not os.path.exists(progress_filepath):
        print(f"No progress file found at {progress_filepath}. Starting fresh or from --start_index.")
        return None
    
    try:
        with open(progress_filepath, 'r', encoding='utf-8') as f:
            progress_data = json.load(f)
    except Exception as e:
        print(f"Error reading or parsing progress file {progress_filepath}: {e}. Assuming no progress.")
        # Optionally delete corrupted progress file: os.remove(progress_filepath)
        return None
        
    last_processed_sft_index = progress_data.get("last_processed_sft_index")
    stored_sft_hash = progress_data.get("sft_input_file_hash")

    if last_processed_sft_index is None: # stored_sft_hash can be None if initial save had issues with hashing sft file
        print(f"Progress file {progress_filepath} is incomplete (missing last_processed_sft_index). Ignoring progress.")
        return None

    # If stored_sft_hash is None from a previous save attempt where sft_input_file was unhashable,
    # we cannot verify. Treat as mismatch / ignore progress.
    if stored_sft_hash is None:
        print(f"Progress file {progress_filepath} has no SFT file hash. Cannot verify. Ignoring progress.")
        return None

    current_sft_hash = calculate_file_hash(current_sft_input_filepath)
    if current_sft_hash is None:
        print(f"Warning: Could not calculate hash for current SFT input {current_sft_input_filepath}. Cannot verify progress integrity. Ignoring saved progress.")
        return None

    if stored_sft_hash == current_sft_hash:
        print(f"Successfully loaded progress: Last processed SFT index {last_processed_sft_index} from {progress_filepath} (SFT content verified). Next item to process will be index {int(last_processed_sft_index) + 1}.")
        return int(last_processed_sft_index) # Ensure it's an int
    else:
        print(f"SFT input file {current_sft_input_filepath} has changed since last progress save (hash mismatch with {progress_data.get('sft_input_filepath_for_reference', 'N/A')}). Ignoring progress from {progress_filepath}.")
        return None

async def process_single_item_async(semaphore, args, record, item_index_in_batch, client_ref, batch_start_sft_index, sft_input_filepath_for_hash, progress_filepath, progress_state):
    """Processes a single SFT item asynchronously, respecting the semaphore."""
    global client
    client = client_ref # Ensure client is set for this task context if needed globally by helpers
    
    original_sft_index = batch_start_sft_index + item_index_in_batch
    item_start_time = time.time()
    qa_pair_result = None
    status_msg = "Error" # Default status

    async with semaphore:
        print(f"--- Starting processing for SFT index: {original_sft_index} (Batch item: {item_index_in_batch + 1}) ---")
        
        code_block = record.get("code_block")
        language = record.get("language", "unknown")
        element_name = record.get("name", f"element_{original_sft_index}")
        element_type = record.get("type", "unknown_type")

        # Determine the base output filename (e.g., 0_MyFunc.cpp or 1_MyClass.cs)
        clean_name_for_file = sanitize_filename(element_name)
        base_name_part = f"{original_sft_index}_{clean_name_for_file}"
        output_filename_root_for_logging_and_base = base_name_part
        original_file_path_val_for_ext = record.get("file_path")

        if original_file_path_val_for_ext:
            original_ext = os.path.splitext(original_file_path_val_for_ext)[1]
            if original_ext and original_ext.lower() in ['.cpp', '.h', '.hpp', '.c', '.cc', '.cs']:
                output_filename_root_for_logging_and_base = base_name_part + original_ext
            elif language == "cpp": output_filename_root_for_logging_and_base = base_name_part + ".cpp"
            elif language == "csharp": output_filename_root_for_logging_and_base = base_name_part + ".cs"
        elif language == "cpp": output_filename_root_for_logging_and_base = base_name_part + ".cpp"
        elif language == "csharp": output_filename_root_for_logging_and_base = base_name_part + ".cs"

        if not code_block:
            print(f"Warning: No code_block found for item at SFT index {original_sft_index}. Skipping.")
            status_msg = "NoCodeBlock"
        else:
            # Step 1: Add comments
            print(f"SFT Index {original_sft_index}: Step 1 - Adding comments to '{element_name}' ({language})...")
            commented_code = await add_comments_to_code(code_block, language, element_name, element_type, model=args.model_comment, sft_index=original_sft_index, base_output_filename_for_log=output_filename_root_for_logging_and_base, max_retries=args.max_api_retries, retry_delay=args.api_retry_delay)
            
            commenting_succeeded = False
            if commented_code:
                output_filename_with_txt = output_filename_root_for_logging_and_base + ".txt"
                commented_file_path = os.path.join(args.commented_code_dir, output_filename_with_txt)
                try:
                    with open(commented_file_path, 'w', encoding='utf-8') as f_out:
                        f_out.write(commented_code)
                    print(f"SFT Index {original_sft_index}: Successfully saved commented code to: {commented_file_path}")
                    commenting_succeeded = True
                except Exception as e:
                    print(f"SFT Index {original_sft_index}: Error saving commented code for '{element_name}' to {commented_file_path}: {e}")
                    log_api_interaction("ERROR", "save_commented_code_failure", args.model_comment, element_name, str(e), original_sft_index, output_filename_root_for_logging_and_base, None)
            
            if not commenting_succeeded:
                print(f"SFT Index {original_sft_index}: Skipping Q&A generation for '{element_name}' as commenting failed or produced no code.")
                status_msg = "CommentFail"
            else:
                # Step 2: Generate Q&A pair
                print(f"SFT Index {original_sft_index}: Step 2 - Generating Q&A for '{element_name}' using original code...")
                qa_pair = await generate_qa_pair(code_block, language, element_name, element_type, model=args.model_qa, sft_index=original_sft_index, base_output_filename_for_log=output_filename_root_for_logging_and_base, max_retries=args.max_api_retries, retry_delay=args.api_retry_delay)
                
                if qa_pair:
                    qa_pair["original_sft_index"] = original_sft_index
                    qa_pair["element_name"] = element_name
                    qa_pair["language"] = language
                    qa_pair["source_file"] = record.get("file_path", "N/A")
                    qa_pair_result = qa_pair 
                    print(f"SFT Index {original_sft_index}: Successfully generated Q&A for '{element_name}'.")
                    status_msg = "OK"

                    if args.individual_qa_dir:
                        individual_qa_filename = f"{output_filename_root_for_logging_and_base}.json"
                        individual_qa_filepath = os.path.join(args.individual_qa_dir, individual_qa_filename)
                        try:
                            with open(individual_qa_filepath, 'w', encoding='utf-8') as f_ind_qa:
                                json.dump(qa_pair, f_ind_qa, indent=2, ensure_ascii=False)
                            print(f"SFT Index {original_sft_index}: Successfully saved individual Q&A to: {individual_qa_filepath}")
                        except Exception as e:
                            print(f"SFT Index {original_sft_index}: Error saving individual Q&A file {individual_qa_filepath}: {e}")
                            log_api_interaction("ERROR", "save_individual_qa_failure", args.model_qa, element_name, str(e), original_sft_index, output_filename_root_for_logging_and_base, None)
                            status_msg = "QASaveFail" 
                else:
                    print(f"SFT Index {original_sft_index}: Failed to generate Q&A for '{element_name}'.")
                    status_msg = "QAFail"

        item_duration = time.time() - item_start_time
        print(f"--- Finished processing for SFT index: {original_sft_index} (Status: {status_msg}) in {item_duration:.2f}s ---")

        async with progress_state["lock"]:
            progress_state["item_statuses"][original_sft_index] = status_msg
            
            if status_msg == "OK" or status_msg == "NoCodeBlock":
                # Try to advance the contiguous progress counter
                new_advanced_idx = progress_state["current_max_saved_idx"]
                idx_to_check = progress_state["current_max_saved_idx"] + 1
                while True:
                    current_item_status = progress_state["item_statuses"].get(idx_to_check)
                    if current_item_status == "OK" or current_item_status == "NoCodeBlock":
                        new_advanced_idx = idx_to_check
                        idx_to_check += 1
                    else:
                        break # Gap found or item not yet processed
                
                if new_advanced_idx > progress_state["current_max_saved_idx"]:
                    progress_state["current_max_saved_idx"] = new_advanced_idx
                    save_progress(progress_filepath, sft_input_filepath_for_hash, new_advanced_idx)
                    print(f"Progress advanced by single item logic to SFT index: {new_advanced_idx}. Item {original_sft_index} triggered check.")
        
        return {
            "original_sft_index": original_sft_index,
            "qa_pair": qa_pair_result,
            "status": status_msg
        }

async def process_items_async(args, sft_data_subset, client_ref, progress_filepath, sft_input_filepath_for_hash, batch_start_sft_index, progress_state):
    """Asynchronously processes SFT data items in a batch with concurrency control."""
    global client 
    client = client_ref

    all_qa_pairs_for_this_batch = []
    tasks = []
    semaphore = asyncio.Semaphore(args.concurrency_limit)
    print(f"Batch processing starting from SFT index {batch_start_sft_index} with concurrency limit {args.concurrency_limit}. Initial saved progress index: {progress_state['current_max_saved_idx']}")

    for i_batch, record_in_batch in enumerate(sft_data_subset):
        # `process_single_item_async` now handles its own printouts for start/finish
        # It also calculates original_sft_index internally based on batch_start_sft_index and i_batch
        tasks.append(process_single_item_async(semaphore, args, record_in_batch, i_batch, client_ref, batch_start_sft_index, sft_input_filepath_for_hash, progress_filepath, progress_state))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results and determine progress
    processed_item_details = []
    for i_batch, result_or_exc in enumerate(results):
        original_sft_idx_for_result = batch_start_sft_index + i_batch # Calculate again for safety/mapping
        if isinstance(result_or_exc, Exception):
            print(f"Error processing SFT index {original_sft_idx_for_result}: An exception occurred during its task: {result_or_exc}")
            # Log this critical failure? Perhaps at a higher level or specific log file.
            # For now, treat as a processing failure for progress calculation.
            processed_item_details.append({
                "original_sft_index": original_sft_idx_for_result,
                "status": "TaskException", # Special status for unhandled exceptions in task
                "qa_pair": None
            })
        else: # This is the dictionary returned by process_single_item_async
            processed_item_details.append(result_or_exc)
            if result_or_exc.get("qa_pair"):
                all_qa_pairs_for_this_batch.append(result_or_exc["qa_pair"])

    # Sort results by original SFT index to correctly determine consecutive success for progress saving
    processed_item_details.sort(key=lambda x: x["original_sft_index"])

    # If the first item in the batch was not successfully processed, we don't update progress based on this batch.
    # The progress file should reflect the highest index *before* this batch started if this batch had failures at the start.
    # However, our save_progress will overwrite. So we must be careful.
    # Let's assume progress is saved if *any* item in the current batch (from its start) is processed successfully.
    # The crucial part is finding the *highest consecutive* successful index *within this batch processing run*, starting from batch_start_sft_index.

    current_batch_last_good_sft_idx = -1 # Tracks the last good SFT index *within this batch processing*

    for detail in processed_item_details:
        # detail["original_sft_index"] is absolute
        # We are checking for an uninterrupted sequence of OK/NoCodeBlock from batch_start_sft_index
        if detail["original_sft_index"] == batch_start_sft_index + (detail["original_sft_index"] - batch_start_sft_index): # a bit redundant, just ensuring we are in sequence
            if detail["status"] == "OK" or detail["status"] == "NoCodeBlock":
                current_batch_last_good_sft_idx = detail["original_sft_index"]
            else:
                # First failure in the sequence from the start of the batch, stop advancing progress for this batch.
                break 
        else:
            # This case should not happen if processed_item_details is sorted and covers the contiguous batch block.
            print(f"Warning: Discontinuity in processed item SFT indexes. Expected {batch_start_sft_index + len(processed_item_details) -1}, got {detail['original_sft_index']}. Progress saving might be conservative.")
            break
            
    if current_batch_last_good_sft_idx != -1:
        print(f"Batch processing complete. Last consecutively successful SFT index in this batch run: {current_batch_last_good_sft_idx}.")
        async with progress_state["lock"]:
            if current_batch_last_good_sft_idx > progress_state["current_max_saved_idx"]:
                save_progress(progress_filepath, sft_input_filepath_for_hash, current_batch_last_good_sft_idx)
                progress_state["current_max_saved_idx"] = current_batch_last_good_sft_idx
                print(f"Shared progress tracker updated by batch-end save to: {current_batch_last_good_sft_idx}")
            elif current_batch_last_good_sft_idx == progress_state["current_max_saved_idx"]:
                 print(f"Batch-end save: SFT index {current_batch_last_good_sft_idx} consistent with current tracker. File likely saved by single item or previous batch. No state change needed based on batch check.")
            else: # current_batch_last_good_sft_idx < progress_state["current_max_saved_idx"]
                 print(f"Batch-end save: Calculated batch SFT index {current_batch_last_good_sft_idx} is behind current tracker ({progress_state['current_max_saved_idx']}). Tracker already reflects more progress. No state change needed based on batch check.")
    else:
        # This means the very first item attempted in this batch (or all items) failed or had an exception.
        # In this case, we do *not* update the progress file from the batch-end logic, as it would either:
        #   a) incorrectly mark an earlier index (if we tried to do current_batch_last_good_sft_idx -1), or
        #   b) reflect no progress for this batch run, which is correct.
        # The existing progress file (if any, from a previous run or a prior successful batch in a long run) remains untouched by this specific batch's failure.
        print(f"Batch processing encountered failures at the beginning or throughout. Progress file not updated for this batch based on SFT index {batch_start_sft_index}. Prior progress (if any) is preserved.")

    total_items_in_batch = len(sft_data_subset)
    successful_items_in_batch = sum(1 for r in processed_item_details if r["status"] == "OK" or r["status"] == "NoCodeBlock")
    print(f"Batch Summary: Processed {total_items_in_batch} items. {successful_items_in_batch} succeeded or were skipped (NoCodeBlock). {len(all_qa_pairs_for_this_batch)} Q&A pairs generated from this batch.")
    print("-" * 50)

    return all_qa_pairs_for_this_batch

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
    parser.add_argument("--start_index", type=int, default=0, help="Index to start processing from in the SFT dataset (for resuming or partial processing).")
    parser.add_argument("--progress_file", type=str, default="sft_processing_progress.json", help="Path to the progress file for saving and resuming state. Default: sft_processing_progress.json")
    parser.add_argument("--concurrency_limit", type=int, default=5, help="Maximum number of SFT items to process concurrently. Default: 5")
    parser.add_argument("--max_api_retries", type=int, default=100, help="Maximum number of retries for a failing API call step. Default: 100")
    parser.add_argument("--api_retry_delay", type=int, default=10, help="Delay in seconds between API call retries. Default: 10")

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

    # Load SFT data
    try:
        with open(args.sft_input_file, 'r', encoding='utf-8') as f:
            sft_data = json.load(f)
    except Exception as e:
        print(f"Error reading or parsing SFT input file {args.sft_input_file}: {e}")
        return

    # Determine effective start index based on progress file and --start_index argument
    loaded_sft_idx = load_progress(args.progress_file, args.sft_input_file)
    
    effective_start_index = args.start_index # User's explicit start_index is the baseline
    initial_tracker_value = -1 # Default if no progress loaded
    if loaded_sft_idx is not None:
        # If progress was loaded, resume from the item AFTER the last processed one.
        # And ensure we don't go back if args.start_index was manually set further ahead.
        effective_start_index = max(args.start_index, loaded_sft_idx + 1)
        initial_tracker_value = loaded_sft_idx
        if effective_start_index > loaded_sft_idx + 1 and args.start_index <= loaded_sft_idx + 1 :
             print(f"Resuming from SFT index {effective_start_index} based on progress file, overriding --start_index {args.start_index} as it was behind.")
        elif effective_start_index == args.start_index and args.start_index > loaded_sft_idx + 1:
             print(f"Starting from SFT index {effective_start_index} as specified by --start_index, which is ahead of saved progress.")
        elif effective_start_index == loaded_sft_idx + 1:
             print(f"Resuming from SFT index {effective_start_index} based on progress file.")
    else:
        print(f"No valid progress loaded. Starting from SFT index {effective_start_index} (based on --start_index or default 0).")

    progress_state = {
        "lock": asyncio.Lock(),
        "current_max_saved_idx": initial_tracker_value,
        "item_statuses": {} # Stores SFT_index: status_msg for items processed in current run
    }
    # Adjust slicing based on effective_start_index
    start_idx = effective_start_index 
    end_idx = len(sft_data)
    if args.max_items is not None:
        # max_items is relative to the effective_start_index
        end_idx = min(start_idx + args.max_items, len(sft_data))

    if start_idx >= len(sft_data) and len(sft_data) > 0:
        print(f"Effective start index {start_idx} is at or beyond the end of SFT data (length {len(sft_data)}). Nothing to process.")
        return
    if start_idx >= end_idx:
         print(f"Effective start index {start_idx} is not less than end index {end_idx}. Nothing to process.")
         return
        
    print(f"Effective processing range: SFT index {start_idx} to {end_idx -1}.")

    sft_data_to_process = sft_data[start_idx:end_idx]
    
    # Run the asynchronous processing
    all_qa_pairs_results = asyncio.run(process_items_async(args, sft_data_to_process, client, args.progress_file, args.sft_input_file, start_idx, progress_state))

    # Save all Q&A pairs (results from async processing)
    try:
        with open(args.qa_output_file, 'w', encoding='utf-8') as f_qa:
            json.dump(all_qa_pairs_results, f_qa, indent=2, ensure_ascii=False)
        print(f"\nSuccessfully saved all Q&A pairs ({len(all_qa_pairs_results)} generated) to: {args.qa_output_file}")
    except Exception as e:
        print(f"Error saving Q&A pairs to {args.qa_output_file}: {e}")

if __name__ == "__main__":
    main() 
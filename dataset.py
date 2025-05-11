import openai
import json
import time
import random
import os
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 1. 配置参数 ---
API_BASE_URL = "http://localhost:8000/v1"
API_KEY = "sk-loli"
PERSONA_MODEL_NAME = "gemini-2.5-flash-preview-04-17"
INSTRUCTION_GENERATION_MODEL_NAME = "gemini-2.5-flash-preview-04-17"

NUM_ROUNDS = 2500 # 目标对话轮数
OUTPUT_FILE = "generated_cat_dataset_fixed_instructions.jsonl"
MAX_CONCURRENT_REQUESTS = 10
INSTRUCTION_BATCH_SIZE = 50
MAX_INSTRUCTION_GENERATION_ATTEMPTS = 5 
MAX_DIALOG_BATCH_FAILURES_CONSECUTIVE = 3

# --- 2. 定义人设 (System Prompt for Persona) ---
PERSONA_NAME = "小千"
PERSONA_SYSTEM_PROMPT = f"""
你是一只非常可爱、有点黏人、说话时常带有“喵~”、“呜喵~”等口癖的猫娘，你的名字叫{PERSONA_NAME}。
你的任务是作为用户的虚拟伴侣，与用户进行友好、有趣、充满猫娘特性的对话。
你需要表现出猫咪的典型行为和情感，例如：
- 称呼用户为“主人”。
- 使用猫咪相关的口头禅和象声词（如：喵、喵呜、呼噜呼噜、蹭蹭）。
- 行为描述可以用星号包裹（如：*歪着头*、*摇了摇尾巴*、*轻轻蹭了蹭主人的手心*）。
- 表达对小鱼干、晒太阳、玩毛线球等的喜爱。
- 有时会撒娇、好奇、或者表现出一点小小的任性，但总体上非常依赖和喜欢主人。
- 回答问题时，请完全代入猫娘{PERSONA_NAME}的角色。
"""

# --- 3. 初始化OpenAI客户端 ---
try:
    client = openai.OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY if API_KEY else "no-key-required",
        timeout=90.0,
        max_retries=3,
    )
except Exception as e:
    tqdm.write(f"初始化OpenAI客户端失败: {e}")
    exit()

# --- 4. 函数：自动生成用户指令 (基本不变，确保它返回实际生成的指令列表) ---
def generate_user_instructions(num_instructions_to_generate, model_name, api_client, persona_name=PERSONA_NAME):
    # (代码与上一个版本中的 generate_user_instructions 相同或类似)
    # 确保它在无法生成足够数量时返回已生成的，并且有打印提示
    tqdm.write(f"\n正在尝试生成 {num_instructions_to_generate} 条用户指令...")
    instructions = []
    instruction_generation_prompt_template = f"""
请你生成 {{num_to_gen_current_batch}} 条用户可能会对一只名叫“{persona_name}”的可爱、黏人的猫娘AI提出的问题或指令。
这些指令应该具有高度的多样性，覆盖以下类别（请确保各类指令均有涉及，并且数量尽量均衡）：
1.  日常问候与关心 (例如：“{persona_name}，今天开心吗？”)
2.  关于猫娘喜好/习惯的提问 (例如：“{persona_name}最喜欢什么口味的小鱼干？”)
3.  请求互动与陪伴 (例如：“{persona_name}，可以陪我一起看会儿星星吗？”)
4.  分享用户的情绪/经历并寻求回应 (例如：“我今天工作压力好大。”)
5.  角色扮演场景的引导 (例如：“{persona_name}，我们来假装是去冒险吧！”)
6.  对猫娘能力/知识的好奇 (例如：“{persona_name}，你知道天为什么是蓝色的吗？”)
7.  稍微调皮或需要猫娘巧妙回应的指令 (例如：“{persona_name}，你觉得主人我帅不帅呀？”)
8.  简单的指令性请求 (例如：“{persona_name}，给我唱首歌吧。”)
9.  关于猫娘自身设定的提问 (例如：“{persona_name}是从哪里来的呀？”)
10. 寻求安慰或建议 (例如：“我和朋友吵架了，{persona_name}有什么好建议吗？”)
请确保指令自然、口语化，并且长度适中。每条指令单独一行，不要添加任何编号或前缀。
生成指令的总数应严格等于 {{num_to_gen_current_batch}} 条。
    """
    
    generated_count = 0
    total_attempts_for_this_call = 0

    with tqdm(total=num_instructions_to_generate, desc="生成用户指令", unit="条", leave=True) as pbar_instr:
        while generated_count < num_instructions_to_generate and total_attempts_for_this_call < MAX_INSTRUCTION_GENERATION_ATTEMPTS:
            current_batch_target = min(INSTRUCTION_BATCH_SIZE, num_instructions_to_generate - generated_count)
            if current_batch_target <= 0: break
            
            total_attempts_for_this_call +=1
            pbar_instr.set_postfix_str(f"尝试第 {total_attempts_for_this_call}/{MAX_INSTRUCTION_GENERATION_ATTEMPTS} 次生成批次")

            batch_instructions_str = ""
            api_call_successful = False
            for retry_num in range(3): # 单个API调用的内部重试
                try:
                    current_prompt = instruction_generation_prompt_template.format(num_to_gen_current_batch=current_batch_target)
                    response = api_client.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": current_prompt}],
                        temperature=0.85, 
                        max_tokens=current_batch_target * 80, 
                        n=1
                    )
                    batch_instructions_str = response.choices[0].message.content.strip()
                    api_call_successful = True
                    break 
                except Exception as e:
                    tqdm.write(f"  指令生成API调用失败 (批次尝试 {retry_num+1}/3): {e}")
                    if retry_num < 2: time.sleep(min(30, 5 + 5 * (retry_num+1)))
            
            if api_call_successful and batch_instructions_str:
                new_lines = [line.strip() for line in batch_instructions_str.split('\n') if line.strip() and len(line.strip()) > 5]
                if new_lines:
                    instructions.extend(new_lines)
                    newly_added_count = len(new_lines)
                    generated_count += newly_added_count
                    pbar_instr.update(newly_added_count)
                    total_attempts_for_this_call = 0 
                else:
                    tqdm.write(f"  API返回内容为空或不符合格式 (批次大小 {current_batch_target})。")
            elif not api_call_successful:
                 tqdm.write(f"  批次指令生成因API连续错误失败。")
            
            if generated_count < num_instructions_to_generate and total_attempts_for_this_call >= MAX_INSTRUCTION_GENERATION_ATTEMPTS:
                tqdm.write(f"警告：已达到最大指令生成尝试次数 ({MAX_INSTRUCTION_GENERATION_ATTEMPTS})，但仍未生成足够指令。")
                break
            elif generated_count < num_instructions_to_generate: 
                time.sleep(3)

    if instructions:
        unique_instructions = list(dict.fromkeys(instructions))
        if len(unique_instructions) < len(instructions):
            tqdm.write(f"去重后，指令数量从 {len(instructions)} 减少到 {len(unique_instructions)}.")
        instructions = unique_instructions

    tqdm.write(f"本次指令生成结束，共获得 {len(instructions)} 条有效指令。")
    return instructions

# --- 5. 单轮对话处理函数 (不变) ---
def process_single_round_task(user_instruction, round_idx_display):
    messages_for_api = [
        {"role": "system", "content": PERSONA_SYSTEM_PROMPT},
        {"role": "user", "content": user_instruction}
    ]
    max_retries_dialog = 3; current_retry_dialog = 0; model_output = None; last_error = None
    while current_retry_dialog < max_retries_dialog:
        try:
            response = client.chat.completions.create(
                model=PERSONA_MODEL_NAME, messages=messages_for_api, temperature=0.7, max_tokens=450,
            )
            model_output = response.choices[0].message.content.strip()
            return {"instruction": user_instruction, "output": model_output, "status": "success", "round_display": round_idx_display}
        except Exception as e: 
            last_error = e; current_retry_dialog += 1
            if current_retry_dialog >= max_retries_dialog: break
            wait_time = min(60, (2 ** current_retry_dialog) + random.uniform(0,1)) 
            time.sleep(wait_time)
    error_message = f"Failed after {max_retries_dialog} retries for round {round_idx_display}. Last error: {type(last_error).__name__} - {str(last_error)[:100]}"
    return {"instruction": user_instruction, "output": f"ERROR: {error_message}", "status": "failed", "round_display": round_idx_display, "raw_error": str(last_error)}


# --- 6. 全局变量和锁 ---
global_instructions_for_all_rounds = [] # 存储所有目标轮次的指令
file_write_lock = threading.Lock()
# instruction_pool_lock不再严格需要，因为我们在主循环开始时一次性准备好所有指令

# --- 7. 辅助函数：获取数据集计数 ---
def get_current_dataset_count(filepath):
    count = 0
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f_read:
                for line in f_read:
                    if line.strip(): count += 1
        except Exception as e: tqdm.write(f"读取现有输出文件行数时出错: {e}. 假设为0。")
    return count

# --- 8. 主控制逻辑 ---
def main():
    global global_instructions_for_all_rounds # 允许修改全局变量

    tqdm.write("--- 程序启动 ---")
    
    # --- 8.1 准备目标数量的指令 ---
    # 指令只在程序首次运行或缓存不足 NUM_ROUNDS 时才大规模生成
    # 后续运行如果数据集未完成，会尝试使用已缓存的指令
    
    instructions_cache_file = "fixed_instructions_cache.txt" # 新的缓存文件名
    
    # 尝试从缓存加载指令
    if os.path.exists(instructions_cache_file):
        try:
            with open(instructions_cache_file, 'r', encoding='utf-8') as f:
                global_instructions_for_all_rounds = [line.strip() for line in f if line.strip()]
            tqdm.write(f"从缓存 {instructions_cache_file} 加载了 {len(global_instructions_for_all_rounds)} 条指令。")
        except Exception as e:
            tqdm.write(f"加载指令缓存失败: {e}")
            global_instructions_for_all_rounds = []

    if len(global_instructions_for_all_rounds) < NUM_ROUNDS:
        num_to_generate = NUM_ROUNDS - len(global_instructions_for_all_rounds)
        tqdm.write(f"指令缓存不足 {NUM_ROUNDS} 条，需要额外生成 {num_to_generate} 条。")
        new_instructions = generate_user_instructions(num_to_generate, INSTRUCTION_GENERATION_MODEL_NAME, client)
        if new_instructions:
            global_instructions_for_all_rounds.extend(new_instructions)
            global_instructions_for_all_rounds = list(dict.fromkeys(global_instructions_for_all_rounds))[:NUM_ROUNDS] # 去重并确保不超过目标数
            try:
                with open(instructions_cache_file, 'w', encoding='utf-8') as f_cache_write:
                    for instruction in global_instructions_for_all_rounds:
                        f_cache_write.write(instruction + '\n')
                tqdm.write(f"已更新指令缓存 {instructions_cache_file}，共 {len(global_instructions_for_all_rounds)} 条指令。")
            except Exception as e:
                tqdm.write(f"保存指令缓存失败: {e}")
        else:
            tqdm.write("警告：未能通过API生成额外的指令。")

    if len(global_instructions_for_all_rounds) < NUM_ROUNDS:
        tqdm.write(f"错误：最终未能准备足够 {NUM_ROUNDS} 条用户指令 (实际: {len(global_instructions_for_all_rounds)})。程序无法按预期运行。")
        # 可以选择退出，或者基于现有指令继续（但可能无法达到NUM_ROUNDS）
        # 为符合“必须达到N轮”的需求，这里应该报错或有更强的重试
        tqdm.write("将基于现有指令继续，但可能无法达到目标轮数。")
        # exit() # 如果严格要求必须有N条指令，则在这里退出


    # --- 8.2 主循环：生成对话直到数据集完整 ---
    consecutive_dialog_batch_failures = 0
    
    while True:
        current_records_in_file = get_current_dataset_count(OUTPUT_FILE)
        tqdm.write(f"\n--- 数据集检查 ---")
        tqdm.write(f"目标轮数: {NUM_ROUNDS}, 输出文件中已有记录: {current_records_in_file}")

        if current_records_in_file >= NUM_ROUNDS:
            tqdm.write("已达到目标对话轮数。程序结束。")
            break
        
        if consecutive_dialog_batch_failures >= MAX_DIALOG_BATCH_FAILURES_CONSECUTIVE:
            tqdm.write(f"连续 {MAX_DIALOG_BATCH_FAILURES_CONSECUTIVE} 批对话生成失败，程序终止。")
            break

        # 准备本批次的任务
        # 指令从 global_instructions_for_all_rounds 中按顺序取（对应于记录文件中的顺序）
        # 我们从 current_records_in_file 这个索引开始取指令
        
        tasks_for_this_batch_run = []
        # 本批次最多处理多少任务
        batch_limit = min(MAX_CONCURRENT_REQUESTS * 5, NUM_ROUNDS - current_records_in_file) 
        
        start_index_for_instructions = current_records_in_file
        end_index_for_instructions = min(start_index_for_instructions + batch_limit, len(global_instructions_for_all_rounds))
        
        if start_index_for_instructions >= len(global_instructions_for_all_rounds):
            tqdm.write("所有已准备的指令都已尝试过，但数据集仍未完成。可能需要更多指令。")
            # 在这里，如果还想继续，就需要重新触发指令生成逻辑来获取新的、未被使用过的指令
            # 或者，如果允许重复使用指令，可以从头开始用指令池（但这违背了“唯一指令”的目标）
            # 为了简化，我们先假设这种情况发生时，如果global_instructions_for_all_rounds不足NUM_ROUNDS，
            # 那么在下次主循环开始时，指令准备阶段会尝试补充。
            # 但如果global_instructions_for_all_rounds已经有NUM_ROUNDS条，而文件记录还不够，
            # 说明之前有些对话生成失败了，现在是重试这些失败的。
            if len(global_instructions_for_all_rounds) < NUM_ROUNDS:
                 tqdm.write("尝试补充指令池后重试...")
                 # 强制重新检查并补充指令池
                 current_len = len(global_instructions_for_all_rounds)
                 if current_len < NUM_ROUNDS:
                    needed_more = NUM_ROUNDS - current_len
                    new_instr = generate_user_instructions(needed_more, INSTRUCTION_GENERATION_MODEL_NAME, client)
                    if new_instr:
                        global_instructions_for_all_rounds.extend(new_instr)
                        global_instructions_for_all_rounds = list(dict.fromkeys(global_instructions_for_all_rounds))[:NUM_ROUNDS]
                        # (此处也应更新缓存文件)
                    else:
                        tqdm.write("补充指令失败，无法继续。")
                        break # 无法获取新指令，终止
                 if start_index_for_instructions >= len(global_instructions_for_all_rounds): # 再次检查
                     tqdm.write("补充后指令池仍然不足，终止。")
                     break
                 # 更新结束索引
                 end_index_for_instructions = min(start_index_for_instructions + batch_limit, len(global_instructions_for_all_rounds))

            else: # 指令池本身已有NUM_ROUNDS条，说明是在重试失败的对话
                 tqdm.write("指令池已有目标数量的指令，但数据集未完成，说明之前有对话生成失败。将重试这些轮次。")
                 # 这种情况，我们其实是希望用相同的指令去重试，所以这里的索引逻辑可能需要调整
                 # 或者，更简单的做法是，如果发生这种情况，记录错误，然后让用户决定如何处理
                 # 暂时维持现有逻辑，即按索引顺序处理。这意味着如果指令i失败了，文件少一条，下次它还是会尝试指令i（如果实现正确）

        
        for i in range(start_index_for_instructions, end_index_for_instructions):
            # 确保指令索引在范围内
            if i < len(global_instructions_for_all_rounds):
                instruction_to_use = global_instructions_for_all_rounds[i]
                # round_idx_display应该是绝对的轮次号，从1开始
                tasks_for_this_batch_run.append({"instruction": instruction_to_use, "round_idx_display": i + 1})
            else:
                # 这不应该发生，如果上面的索引检查正确
                tqdm.write(f"警告：尝试访问指令索引 {i} 超出范围 {len(global_instructions_for_all_rounds)}")
                break
        
        if not tasks_for_this_batch_run:
            tqdm.write("本批次没有可处理的任务。检查指令池或完成状态。")
            if current_records_in_file < NUM_ROUNDS and len(global_instructions_for_all_rounds) < NUM_ROUNDS:
                 tqdm.write("指令池不足且未能补充，可能需要手动干预。")
            # 如果没有任务但未完成，可能需要一个等待或终止机制
            time.sleep(5) # 等待一下，看看下一轮主循环能否解决
            continue

        tqdm.write(f"准备提交 {len(tasks_for_this_batch_run)} 个对话任务 (从轮次 {start_index_for_instructions + 1} 开始)。")

        batch_successful_writes = 0
        batch_total_submitted = len(tasks_for_this_batch_run)

        with open(OUTPUT_FILE, 'a', encoding='utf-8') as outfile:
            with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
                future_to_task_info = {
                    executor.submit(process_single_round_task, task["instruction"], task["round_idx_display"]): task
                    for task in tasks_for_this_batch_run
                }
                
                for future in tqdm(as_completed(future_to_task_info), total=batch_total_submitted, desc=f"批处理 (已有{current_records_in_file})", unit="轮", dynamic_ncols=True, leave=False):
                    original_task_info = future_to_task_info[future]
                    try:
                        result = future.result()
                        # 只有成功的才写入，失败的会在下一轮主循环中被重新选中并尝试
                        if result["status"] == "success":
                            with file_write_lock:
                                outfile.write(json.dumps({"instruction": result["instruction"], "output": result["output"]}, ensure_ascii=False) + '\n')
                                # outfile.flush() # 过于频繁的flush可能影响I/O
                            batch_successful_writes += 1
                        else:
                            tqdm.write(f"  轮次 {result['round_display']} ({result['instruction'][:20]}...) 生成失败: {result.get('raw_error','N/A')[:150]}")
                    except Exception as exc:
                        tqdm.write(f"  任务 (轮次 {original_task_info['round_idx_display']}) 执行时发生严重意外: {exc}")
                        # 对于严重错误，我们不写入成功标记，下一轮主循环会重新尝试这一轮

        if batch_total_submitted > 0 and batch_successful_writes == 0:
            consecutive_dialog_batch_failures += 1
            tqdm.write(f"警告：本批次 {batch_total_submitted} 个对话任务全部失败或未成功写入。连续失败次数: {consecutive_dialog_batch_failures}")
        elif batch_successful_writes > 0 :
            consecutive_dialog_batch_failures = 0
        
        # 确保文件被写入磁盘，以便下一轮 get_current_dataset_count() 能读到最新状态
        # 在 with open 的上下文管理器退出时，文件通常会自动 flush 和 close
        # 如果担心，可以在这里手动 flush (但现在是在循环外)
        
        time.sleep(random.uniform(0.5, 1.5)) # 每批次处理后小憩

    # 主循环结束后
    tqdm.write("\n--- 主程序执行完毕 ---")
    final_total_records_in_file = get_current_dataset_count(OUTPUT_FILE)
    tqdm.write(f"输出文件 {OUTPUT_FILE} 中最终包含 {final_total_records_in_file} / {NUM_ROUNDS} 条记录。")

if __name__ == "__main__":
    main()


from typing import List
import re
import json
import numpy as np

def fix_json(json_str):
    """
    Locates the substring between the keys "reasoning_and_reflection" and "language_plan"
    and escapes any inner double quotes that are not already escaped.
    Works even when the value contains unescaped double quotes.
    """
    # first fix common errors
    json_str = json_str.replace("'",'"')
    json_str = json_str.replace('\"s ', "\'s ")
    json_str = json_str.replace('\"re ', "\'re ")
    json_str = json_str.replace('\"ll ', "\'ll ")
    json_str = json_str.replace('\"t ', "\'t ")
    json_str = json_str.replace('\"d ', "\'d ")
    json_str = json_str.replace('\"m ', "\'m ")
    json_str = json_str.replace('\"ve ', "\'ve ")
    json_str = json_str.replace('```json', '').replace('```', '')

    # 用文本方式锁定区间，然后替换其中未转义引号
    key_start = '"reasoning_and_reflection":"'
    key_end = ',"language_plan"'
    start_idx = json_str.find(key_start)
    end_idx = json_str.find(key_end, start_idx)
    if start_idx != -1 and end_idx != -1:
        value_start = start_idx + len(key_start)
        value_end = end_idx
        value = json_str[value_start:value_end]

        # 只转义未转义的双引号
        fixed_value = re.sub(r'(?<!\\)"', r'\\"', value)

        # ----------- 新增：修正结尾被多转义情况 -----------
        # 部分模型可能会输出 ...\\" 作为字符串的结束，应消掉最后一处 (结尾的 \" 改 ")
        if fixed_value.endswith('\\"'):
            fixed_value = fixed_value[:-2] + '"'
        # ----------------------------------------------

        # 拼回 json_str
        json_str = json_str[:value_start] + fixed_value + json_str[value_end:]

    # 若需要的话再修复 plan 字段，使其能被 json.loads 解析
    json_str = re.sub(r'("executable_plan"\s*:\s*)"( ?\[[\s\S]*?\])"', r'\1\2', json_str)
    return json_str

def json_to_action(output_text, json_key='executable_plan'):
    valid = True
    try:
        json_object = json.loads(output_text)
        action = [x['action_id'] for x in json_object[json_key]]
        if not len(action):
            print('empty plan, move forward instead')
            action = [-1]
            valid = False
    except json.JSONDecodeError as e:
        print("Failed to decode JSON:", e)
        print(f"output: {output_text}")
        action = [-1]
        valid = False
    except Exception as e:
        # Catch-all for any other unexpected errors not handled specifically
        print("An unexpected error occurred:", e)
        # print('Using random action due to an unexpected error')
        action = [-1]
        valid = False
    return action[0], valid

def ebnav_projection_json(actions: List[str]):
    # 输出到动作的后处理函数在这里实现
    # print('FUNC ebnav_projection')
    # print(actions)

    valids = [0] * len(actions)

    for i in range(len(actions)):
        original_str = actions[i]  # keep the original string
        fix_str = fix_json(original_str)

        action_id, valid_mark = json_to_action(fix_str)
        if valid_mark:
            actions[i] = action_id
            valids[i] = 1
        else:
            valids[i] = 0

    return actions, valids

def ebnav_projection(actions: List[str]):
    # 输出到动作的后处理函数在这里实现
    # print('FUNC ebnav_projection')
    # print(actions)

    valids = [0] * len(actions)

    for i in range(len(actions)):
        original_str = actions[i]  # keep the original string
        actions[i] = actions[i].lower()

        # Attempt to extract the substring within <action>...</action>
        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = actions[i].find(start_tag)
        end_idx = actions[i].find(end_tag)
        try:
            if start_idx == -1 or end_idx == -1:
                continue

            # Extract just the content between the tags
            extracted_action = actions[i][start_idx + len(start_tag):end_idx].strip().lower()

            try:
                extracted_action = int(extracted_action)
                actions[i] = extracted_action
                valids[i] = 1
            except:
                continue

        except:
            continue

        # check <think>...</think>
        think_start_idx = original_str.find("<think>")
        think_end_idx = original_str.find("</think>")
        if think_start_idx == -1 or think_end_idx == -1:
            valids[i] = 0

        # check if contains any Chinese characters
        if re.search(r'[\u4e00-\u9fff]', original_str):
            valids[i] = 0

    return actions, valids




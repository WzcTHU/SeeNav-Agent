import json


def jsonl_to_json(jsonl_file_path, json_file_path):
    """
    将JSONL文件转换为JSON文件
    
    Args:
        jsonl_file_path: 输入的JSONL文件路径
        json_file_path: 输出的JSON文件路径
    """
    data_list = []
    
    # 读取JSONL文件并收集所有字典
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                # 解析每行的JSON对象
                data = json.loads(line.strip())
                data_list.append(data)
            except json.JSONDecodeError as e:
                print(f"警告：第{line_num}行解析失败 - {str(e)}，已跳过该行")
    
    # 将收集到的列表写入JSON文件
    with open(json_file_path, 'w', encoding='utf-8') as f:
        # 确保中文正常显示，使用indent参数美化输出
        json.dump(data_list, f, ensure_ascii=False, indent=2)
    
    print(f"转换完成！共处理 {len(data_list)} 条记录，已保存至 {json_file_path}")


def stat_action(jsonl_file_path):
    counts = {'json_error': 0}
    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                j = json.loads(line)
            except:
                counts['json_error'] += 1
                continue
            for c in j['conversations']:
                if c['from'] == 'gpt':
                    try:
                        v = json.loads(c['value'])
                    except:
                        counts['json_error'] += 1
                        continue
                    for plan in v['executable_plan']:
                        aid = plan['action_id']
                        counts[aid] = counts.get(aid, 0) + 1
    print(counts)
if __name__ == '__main__':
    in_jsonl = '/mnt/kaiwu-user-jensencwang/research/data/SeeNav/sft_samples/sft_dataset.jsonl'
    out_json = '/mnt/kaiwu-user-jensencwang/research/LLama-Factory/LLaMA-Factory/data/seenav_sft_dataset.json'
    # jsonl_to_json(in_jsonl, out_json)
    stat_action(in_jsonl)

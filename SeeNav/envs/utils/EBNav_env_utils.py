import os
import json
import glob
import logging

logger = logging.getLogger("EB_logger")
if not logger.hasHandlers():
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] - %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

def average_json_values(json_dir, target_file='*.json', output_file='summary_all.json', selected_key=None):
    values_sum = {}
    counts = {}

    json_files = glob.glob(os.path.join(json_dir, target_file)) + glob.glob(os.path.join(json_dir, '*', target_file)) + glob.glob(os.path.join(json_dir, '*', '*', target_file))
    print(json_files, len(json_files))

    for json_file in json_files:
        # print(json_file.split('running/')[1])
        if output_file in json_file:
            continue
        with open(json_file, 'r') as f:
            data = json.load(f)
            print(data[selected_key] if selected_key!= None else data)
            for key, value in data.items():
                if selected_key != None and key != selected_key:
                    continue
                if type(value) == str:
                    continue
                    
                if isinstance(value, list) and len(value) == 1:
                    value = value[0]
                if key not in values_sum:
                    values_sum[key] = 0.0
                    counts[key] = 0
                values_sum[key] += value
                counts[key] += 1
    
    averages = {key: values_sum[key] / counts[key] for key in values_sum}
    print('final results: ' )
    print(averages)
    with open(os.path.join(json_dir, output_file), 'w') as f:
        json.dump(averages, f, indent=4)
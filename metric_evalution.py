import os
import json
import jsonlines


result_dir = './llama3_8b_sft_bird_dev_gold_schema'
data_dir = 'BIRD/dev'

file_path = os.path.join(data_dir, 'dev.json')
with open(file_path, 'r', encoding='utf-8') as file:
    data_bird = json.load(file)
result = {'total': []}
for index, item in enumerate(data_bird):
    if item['difficulty'] not in result.keys():
        result[item['difficulty']] = []

    flag = 0
    if os.path.exists(os.path.join(result_dir, str(index), 'result.json')):
        with jsonlines.open(os.path.join(result_dir, str(index), 'result.json')) as jsonlines_file:
            for row in jsonlines_file:
                if row['Flag'] == 1:
                    flag = 1
    result[item['difficulty']].append(flag)
    result['total'].append(flag)

for key in list(result.keys()):
    print(key, len(result[key]), sum(result[key]), sum(result[key]) / len(result[key]))
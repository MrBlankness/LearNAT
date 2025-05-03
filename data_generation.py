import random
import numpy as np
import torch
from transformers import set_seed
import os
import json
from openai import OpenAI
import re
from prompt import gen_next_subtask_subsql_prompt, gen_schema_linking_gp_prompt
from func_timeout import func_set_timeout, FunctionTimedOut
import sqlite3
import pandas as pd
import jsonlines
from transformers import AutoTokenizer, AutoModelForCausalLM
import http.client
import copy
import concurrent.futures
from zhipuai import ZhipuAI
from MCTS import NL2SQLMCTS


SEED = 2024

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
set_seed(SEED)

data_dir = 'NL2SQL/dataset/BIRD/train'
save_dir = './steps_result_glm_4_plus_train'

file_path = os.path.join(data_dir, 'train.json')
with open(file_path, 'r', encoding='utf-8') as file:
    data_bird = json.load(file)

train_table_info = {}
train_table_path = os.path.join(data_dir, 'train_tables.json')
with open(train_table_path, 'r', encoding='utf-8') as file:
    train_table = json.load(file)
for index, item in enumerate(train_table):
    train_table_info[item['db_id'].lower()] = {'tables_info': {}, 'foreign_keys': []}
    for table_name in item['table_names_original']:
        train_table_info[item['db_id'].lower()]['tables_info'][table_name.lower()] = []

    for column_name in item['column_names_original'][1:]:
        train_table_info[item['db_id'].lower()]['tables_info'][item['table_names_original'][column_name[0]].lower()].append('`' + column_name[1].lower() + '`')
    for foreign_key in item['foreign_keys']:
        train_table_info[item['db_id'].lower()]['foreign_keys'].append([
            item['table_names_original'][item['column_names_original'][foreign_key[0]][0]].lower() + '.' + '`' + item['column_names_original'][foreign_key[0]][1].lower() + '`',
            item['table_names_original'][item['column_names_original'][foreign_key[1]][0]].lower() + '.' + '`' + item['column_names_original'][foreign_key[1]][1].lower() + '`'
        ])


def parse_sql_from_string(input_string):
    sql_pattern = r'```sql(.*?)```'
    all_sqls = []
    # 将所有匹配到的都打印出来
    for match in re.finditer(sql_pattern, input_string, re.DOTALL):
        all_sqls.append(match.group(1).strip())
    if all_sqls:
        return all_sqls[-1]
    else:
        # return "error: No SQL found in the input string"
        return input_string.strip().strip('`')


@func_set_timeout(120)
def execute_sql(sql: str, db_id: str) -> dict:
    db_path = os.path.join(data_dir, 'train_databases', db_id, db_id + '.sqlite')
    conn = sqlite3.connect(db_path)
    conn.text_factory = lambda b: b.decode(errors="ignore")
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        result = cursor.fetchall()
        if len(result) == 0:
            if "= 'Y'" in sql:
                print("*" * 20, 'replace = Y with = 1', "*" * 20)
                print(sql.count("= 'Y'"))
                return execute_sql(sql.replace("= 'Y'", "= 1"), db_id)
        return {
            "sql": str(sql),
            "data": result[:5],
            "sqlite_error": "",
            "exception_class": ""
        }
    except sqlite3.Error as er:
        return {
            "sql": str(sql),
            "sqlite_error": str(' '.join(er.args)),
            "exception_class": str(er.__class__)
        }
    except Exception as e:
        return {
            "sql": str(sql),
            "sqlite_error": str(e.args),
            "exception_class": str(type(e).__name__)
        }


def is_need_refine(exec_result: dict):
    data = exec_result.get('data', None)
    if data is not None:
        if len(data) == 0:
            exec_result['sqlite_error'] = 'no data selected'
            return True
        for t in data:
            for n in t:
                if n is None:  # fixme fixme fixme fixme fixme
                    exec_result['sqlite_error'] = 'exist None value, you can add `NOT NULL` in SQL'
                    return True
        return False
    else:
        return True


def add_prefix(sql):
    if not sql.startswith('SELECT') and not sql.startswith('select'):
        sql = 'SELECT' + sql
    return sql


def evaluate_answer(db_id, response) -> str:
    try:
        pred_sql = parse_sql_from_string(response)
    except Exception as e:
        res = f'error: {str(e)}'

    is_timeout = False
    try:
        error_info = execute_sql(pred_sql, db_id)
    except Exception as e:
        is_timeout = True
    except FunctionTimedOut as fto:
        is_timeout = True
    is_need = is_need_refine(error_info)
    if is_timeout:
        error_info['res'] = "timeout"
        return error_info
    if is_need:
        error_info['res'] = "runerror"
        return error_info
    error_info['res'] = "success"
    return error_info


def verify_schema_linking_gt(schema_linking_result, db_id):
    db_id = db_id.lower()
    try:
        if isinstance(schema_linking_result, str):
            schema_linking_result = json.loads(extract_json_from_string(schema_linking_result).replace("'", '"'))
    except:
        return "Your answer does not meet the required JSON format, please regenerate it!"
    if not {x.lower() for x in set(schema_linking_result.keys())}.issubset({x.lower() for x in set(train_table_info[db_id]['tables_info'].keys())}):
        diff_set = {x.lower() for x in set(schema_linking_result.keys())} - {x.lower() for x in set(train_table_info[db_id]['tables_info'].keys())}
        return "Table {} does not exist in the database. Please rethink and generate JSON answers".format(', '.join(list(diff_set)))
    for table_name in list(schema_linking_result.keys()):
        pred_list = ['`' + column_name.lower() + '`' for column_name in schema_linking_result[table_name]]
        if not set(pred_list).issubset({x.lower() for x in set(train_table_info[db_id]['tables_info'][table_name.lower()])}):
            diff_set = set(pred_list) - {x.lower() for x in set(train_table_info[db_id]['tables_info'][table_name.lower()])}
            return "Column {} does not exist in the Table {}. Please rethink and generate JSON answers".format(', '.join(list(diff_set)), table_name)
    return schema_linking_result


# 删除LLM在json中额外给的注释
def del_note(s):
    news = ''
    s_splits = s.split('\n')
    for s_split in s_splits:
        if '//' in s_split:
            news += s_split.split('//')[0]
        else:
            news += s_split
        news += '\n'
    return news

# 提取json list
def extract_json_from_string(s):
    if '//' in s:
        s = del_note(s)
    i = s.index('{')
    end_i = i
    count = 1 #当前所在嵌套深度，即还没闭合的'{'个数
    for j,c in enumerate(s[i+1:], start=i+1):
        if c == '}':
            count -= 1
            if count == 0:
                end_i = j
        elif c == '{':
            count += 1
    # assert(count == 0) #检查是否找到最后一个'}'
    if count == 0:
        return s[i:end_i+1]
    else:
        return 'error'


@func_set_timeout(60)
def get_response(messages):
    client = ZhipuAI(api_key="XXXXXXXXXX")  # 请填写您自己的APIKey
    response = client.chat.completions.create(
        model="glm-4-plus",  # 请填写您要调用的模型名称
        messages=messages,
    )
    return response.choices[0].message.content


def process_task(index, item):
    # error case in BIRD
    if index in [2462, 2464, 2467, 2470, 2512, 2513, 2518, 2525, 2558, 2567, 4748, 4749, 4750, 4751, 4752, 4753, 4754, 4757, 4761, 4762, 4766, 4765, 4770, 4771, 4772, 4773, 4779, 4783, 4784, 4786, 4790, 4793, 4794, 4796, 4795, 4802, 4810, 4812, 4811, 4815, 4813, 4814, 4816, 4820, 4822, 4824, 4834, 4833, 4836, 4837, 4839, 4840, 4844, 4849, 4850, 4856, 4857, 4858, 4860, 4862, 4864, 4866, 4868, 4870, 4872, 4874, 4876, 4878, 4880, 4882, 4892, 4893, 4894, 4895, 4896, 4897, 4898, 4900, 4901, 4902, 4905, 4907, 4910, 4912, 4911, 6297, 6298, 6299, 6300, 6301, 6302, 6303, 6304, 6305, 6306, 6307, 6308, 6309, 6310, 6311, 6312, 6313, 6314, 6315, 6316, 6317, 6318, 6319, 6320, 6321, 6322, 6323, 6325, 6326, 6327, 6328, 6329, 6330, 6331, 6333, 6335, 6336, 6337, 6338, 6339, 6340, 6341, 6342, 6345, 6346, 6347, 6350, 6351, 6352, 6353, 6354, 6356, 6357, 6358, 6360, 6361, 6362, 6363, 6364, 6365, 6366, 6367, 6368, 6369, 6370, 6372, 6373, 6374, 6375, 6376, 6377, 6378, 6380, 6381, 6382, 6383, 6384, 6385, 6386, 6387, 6388, 6389, 6390, 6392, 6391, 6393, 6394, 6396, 6395, 6397, 6398, 6399, 6400, 6401, 6402, 6403, 6404, 6405, 6406, 6408, 6407, 6409, 6410, 6412, 6413, 6414, 6415, 6416, 6418, 6417, 6419, 6420, 6422, 6421, 6424, 6423, 6426, 6427, 6428, 6431, 6432, 6433, 6435, 6436, 6437, 6439, 6438, 6440, 6441, 6442, 6443, 6445, 6444, 6446, 6447, 6448, 6449, 6451, 6450, 6453, 6452, 6454, 6455, 6456, 6457, 6458, 6460, 6459, 6462, 6464, 6465, 6466, 6467, 6468, 6471, 6472, 6474, 6475, 6473, 6476, 6477, 6478, 6479, 6480, 6481, 6483, 6482, 6486, 6484, 6485, 6488, 6489, 6487, 6491, 6492, 6490, 6493, 6496, 6494, 6497, 6498, 6499, 6500, 6501, 6503, 6504, 6505, 6507, 6509, 6510, 6511, 6512, 6513, 6515, 6517, 6518, 6519, 6520, 6522, 6523, 6521, 6524, 6525, 6526, 6527, 6528, 6529, 6530, 6531, 6532, 6533, 6535, 6536, 6537, 6538, 6540, 6539, 6541, 6542, 6544, 6543, 6545, 6546, 6547, 6548, 6550, 6551, 6553, 6555, 6554, 6556, 6557, 6558, 6560, 6561, 6559, 6562, 6563, 6564, 6565, 6566, 6567, 6568, 6569, 6570, 6572, 6571, 6573, 6575, 6576, 6577, 6578, 6579, 6581, 6582, 6584, 6586, 6587, 6588, 6591, 6589, 6593, 6592, 6590, 6594, 6596, 6598, 6595, 6597, 6599, 6602, 6601, 6600, 6603, 6605, 6604, 6606, 6607, 6610, 6609, 6611, 6612, 6616, 6617, 6615, 6619, 6621, 6620, 6618, 6623, 6624, 6625, 6626, 6627, 6629, 6630, 6631, 6628, 6632, 6635, 6633, 6634, 6636, 6638, 6639, 6640, 6641, 6643, 6645, 6642, 6647, 6649, 6648, 6646, 6652, 6651, 6650, 6654, 6657, 6658, 6660, 6659, 6661, 6662, 6664, 6663, 6665, 6667, 6668, 6669, 7122, 7129, 7172, 7173, 7178, 7218, 7227, 7325, 7409, 7438]:
        return
    
    if not os.path.exists(os.path.join(save_dir, str(index))):
        os.makedirs(os.path.join(save_dir, str(index)))

    result_file = os.path.join(save_dir, str(index), 'result.jsonlines')
    schema_linking_gt_file = os.path.join(save_dir, str(index), 'schema_linking_gt.json')
    log_file = os.path.join(save_dir, str(index), 'log.jsonlines')

    if os.path.exists(result_file):
        return
    
    question = item['question']
    evidence = item['evidence']
    sql = item['SQL']
    db_id = item['db_id']
    

    # 生成schema linking的ground truth
    if not os.path.exists(schema_linking_gt_file):
        schema_linking_gt_messages = [{"role": "system", "content": gen_schema_linking_gp_prompt},
                                      {"role": "user", "content": sql}]
        for _ in range(3):
            schema_linking_result_str = get_response(schema_linking_gt_messages)
            schema_linking_result = verify_schema_linking_gt(schema_linking_result_str, db_id)
            if isinstance(schema_linking_result, dict):
                break
            else:
                schema_linking_gt_messages.append({"role": "assistant", "content": schema_linking_result_str})
                schema_linking_gt_messages.append({"role": "user", "content": schema_linking_result})
                
        if not isinstance(schema_linking_result, dict):
            return
        with open(schema_linking_gt_file, 'w') as json_file:
            json.dump(schema_linking_result, json_file, indent=4)
    else:
        with open(schema_linking_gt_file, 'r', encoding='utf-8') as file:
            schema_linking_result = json.load(file)

    # 生成schema linking的prompt
    desc_str = ''
    column_list = []
    table_names = list(schema_linking_result.keys())
    for table_name in table_names:
        column_names = schema_linking_result[table_name]
        desc_str += table_name + ' (`' + '`, `'.join(column_names) +'`) \n'
        column_list += [table_name + '.`' + column_name + '`' for column_name in column_names]
    
    # 读取csv文件生成column description和示例
    desc_str = ''
    for table_name in table_names:
        csv_file_dir = os.path.join(data_dir, 'train_databases/{}/database_description'.format(db_id))
        csv_files = os.listdir(csv_file_dir)
        for csv_file in csv_files:
            if csv_file.lower() == table_name.lower() + '.csv':
                csv_file_path = os.path.join(data_dir, 'train_databases/{}/database_description'.format(db_id), csv_file)
                break
        try:
            df = pd.read_csv(csv_file_path)
        except:
            df = pd.read_csv(csv_file_path, encoding="ISO-8859-1")
        desc_str += '- Table: {table_name}\n'.format(table_name=table_name)
        column_names = schema_linking_result[table_name]
        for column_name in column_names:
            desc_str += '\t- Column: `{column_name}` ({column_type})\n'.format(column_name=column_name, column_type=df[df['original_column_name'] == column_name]['data_format'].values[0] if len(df[df['original_column_name'] == column_name]['data_format'].values) > 0 else '')
            desc_str += '\t\t- Description: {description}\n'.format(description=df[df['original_column_name'] == column_name]['column_description'].values[0] if len(df[df['original_column_name'] == column_name]['column_description'].values) > 0 else '')
            value_sql_query = "SELECT `{}` FROM {} LIMIT 10".format(column_name, table_name)
            res = execute_sql(value_sql_query, db_id)
            if 'data' in list(res.keys()):
                res = res['data']
                res = [str(res_item[0]) for res_item in res]
            else:
                res = ''
            desc_str += '\t\t- Samples: [{sample}]\n'.format(sample=', '.join(res))
            desc_str += '\t\t- Value Description: {description}\n'.format(description=df[df['original_column_name'] == column_name]['value_description'].values[0] if len(df[df['original_column_name'] == column_name]['value_description'].values) > 0 else '')

    fk_str = ''
    for foreign_key in train_table_info[db_id]['foreign_keys']:
        if foreign_key[0] in column_list and foreign_key[1] in column_list:
            fk_str += foreign_key[0] + ' = ' + foreign_key[1] + '\n'

        
    sub_question_message = [{"role": "system", "content": "You are a data science expert."},
                            {"role": "user", "content": gen_next_subtask_subsql_prompt.format(desc_str=desc_str, fk_str=fk_str, query=question, evidence=evidence, sql=sql)}]
    
    nl2sqlmcts = NL2SQLMCTS(sub_question_message, sql)
    dec_result = nl2sqlmcts.run()

    with jsonlines.open(result_file, mode="a") as file_jsonl:
        file_jsonl.write({"index": index, "dec_result": dec_result})


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        # 并行执行
        executor.map(lambda x: process_task(*x), enumerate(data_bird))

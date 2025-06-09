import random
import os
import json
import numpy as np
import torch
import sqlite3
import time
import re
import pandas as pd
import jsonlines
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from func_timeout import func_set_timeout, FunctionTimedOut


SEED = 2024

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
set_seed(SEED)

instruction = '''You are a step-by-step problem solver for complex Text-to-SQL tasks. Given a 【Database Schema】, a Knowledge 【Evidence】, and a user 【Question】, you need to read the 【Database schema】, understand the 【Question】, and decompose the original 【Question】 into a sequence of natural language subquestions. Each subquestion should correspond to a specific logical step required to solve the overall problem, and for each subquestion, you must also provide the SQL query (subSQL) that answers it. Continue this decomposition until the final subquestion and subSQL directly answer the original user 【Question】.
Requirement:
1. All subquestions must be expressed in natural language, not SQL.
2. Each subSQL must be a valid SQL query that corresponds to its subquestion.
3. The final subSQL should answer the original question.
4. You must reason in a clearly structured, step-by-step format.
5. Your answer format is as follows:
==========
Sub question 1:...
SubSQL1:
```sql
...
```

Sub question 2:...
SubSQL2:
```sql
...
```

Sub question 3:...
SubSQL3:
```sql
...
```
==========
Database admin instructions:
1. When you need to find the highest or lowest values based on a certain condition, using ORDER BY + LIMIT 1 is prefered over using MAX/MIN within sub queries.
2. If predicted query includes an ORDER BY clause to sort the results, you should only include the column(s) used for sorting in the SELECT clause if the question specifically ask for them. Otherwise, omit these columns from the SELECT.
3. If the question doesn't specify exactly which columns to select, between name column and id column, prefer to select id column.
4. Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
5. Predicted query should return all of the information asked in the question without any missing or extra information.
6. No matter of how many things the question asks, you should only return one SQL query as the answer having all the information asked in the question, seperated by a comma.
7. Never use || to concatenate columns in the SELECT. Rather output the columns as they are.
8. If you are joining multiple tables, make sure to use alias names for the tables and use the alias names to reference the columns in the query. Use T1, T2, T3, ... as alias names.
9. If you are doing a logical operation on a column, such as mathematical operations and sorting, make sure to filter null values within those columns, but do not filter null values in the columns of the query results.


Take a deep breath and think step by step to find the correct sqlite SQL query.  If you follow all the instructions and generate the correct query, I will give you 1 million dollars.
==========

【Database schema】
frpm (`CDSCode`, `Charter School (Y/N)`, `Enrollment (Ages 5-17)`, `(Free Meal Count (Ages 5-17)`)
satscores (`cds`, `sname`, `NumTstTakr`, `AvgScrMath`, `NumGE1500`)

【Foreign keys】
frpm.`CDSCode` = satscores.`cds`

【Evidence】
Charter schools refers to `Charter School (Y/N)` = 1 in the table frpm; Excellence rate = NumGE1500 / NumTstTakr

【Solution】
Sub question 1: Get the average value of SAT excellence rate of charter schools.
SubSQL1:
```sql
SELECT AVG(CAST(T2.`NumGE1500` AS REAL) / T2.`NumTstTakr`)
    FROM frpm AS T1
    INNER JOIN satscores AS T2
    ON T1.`CDSCode` = T2.`cds`
    WHERE T1.`Charter School (Y/N)` = 1
```

Sub question 2: List out school names of charter schools with an SAT excellence rate over the average.
SubSQL2:
```sql
SELECT T2.`sname`
  FROM frpm AS T1
  INNER JOIN satscores AS T2
  ON T1.`CDSCode` = T2.`cds`
  WHERE T2.`sname` IS NOT NULL
  AND T1.`Charter School (Y/N)` = 1
  AND CAST(T2.`NumGE1500` AS REAL) / T2.`NumTstTakr` > (
    SELECT AVG(CAST(T4.`NumGE1500` AS REAL) / T4.`NumTstTakr`)
    FROM frpm AS T3
    INNER JOIN satscores AS T4
    ON T3.`CDSCode` = T4.`cds`
    WHERE T3.`Charter School (Y/N)` = 1
  )

==========

【Database schema】
account (`account_id`, `district_id`, `frequency`, `date`)
client (`client_id`, `gender`, `birth_date`, `district_id`)
district (`district_id`, `A4`, `A11`)

【Foreign keys】
account.`district_id` = district.`district_id`
client.`district_id` = district.`district_id`

【Evidence】
Later birthdate refers to younger age; A11 refers to average salary.

【Solution】
Sub question 1: What is the district_id of the branch with the lowest average salary?
SubSQL1:
```sql
SELECT `district_id`
  FROM district
  ORDER BY `A11` ASC
  LIMIT 1
```

Sub question 2: What is the youngest client who opened account in the lowest average salary branch?
SubSQL2:
```sql
SELECT T1.`client_id`
  FROM client AS T1
  INNER JOIN district AS T2
  ON T1.`district_id` = T2.`district_id`
  ORDER BY T2.`A11` ASC, T1.`birth_date` DESC 
  LIMIT 1
```

Sub question 3: What is the gender of the youngest client who opened account in the lowest average salary branch?
SubSQL3:
```sql
SELECT T1.`gender`
  FROM client AS T1
  INNER JOIN district AS T2
  ON T1.`district_id` = T2.`district_id`
  ORDER BY T2.`A11` ASC, T1.`birth_date` DESC 
  LIMIT 1 
```

==========
'''

user_prompt = '''
【Database schema】
{desc_str}

【Foreign keys】
{fk_str}

【Question】
{query}

【Evidence】
{evidence}

【Solution】
'''

data_dir = 'BIRD/dev'
save_dir = './qwen25_7b_sft_bird_dev_gold_schema_cal_token'

file_path = os.path.join(data_dir, 'dev.json')
with open(file_path, 'r', encoding='utf-8') as file:
    data_bird = json.load(file)

dev_table_info = {}
dev_table_path = os.path.join(data_dir, 'dev_tables.json')
with open(dev_table_path, 'r', encoding='utf-8') as file:
    dev_table = json.load(file)
for index, item in enumerate(dev_table):
    dev_table_info[item['db_id'].lower()] = {'tables_info': {}, 'foreign_keys': []}
    for table_name in item['table_names_original']:
        dev_table_info[item['db_id'].lower()]['tables_info'][table_name.lower()] = []

    for column_name in item['column_names_original'][1:]:
        dev_table_info[item['db_id'].lower()]['tables_info'][item['table_names_original'][column_name[0]].lower()].append('`' + column_name[1].lower() + '`')
    for foreign_key in item['foreign_keys']:
        dev_table_info[item['db_id'].lower()]['foreign_keys'].append([
            item['table_names_original'][item['column_names_original'][foreign_key[0]][0]].lower() + '.' + '`' + item['column_names_original'][foreign_key[0]][1].lower() + '`',
            item['table_names_original'][item['column_names_original'][foreign_key[1]][0]].lower() + '.' + '`' + item['column_names_original'][foreign_key[1]][1].lower() + '`'
        ])

schema_linking_result_file = os.path.join(data_dir, 'dev_gold_schema.json')
with open(schema_linking_result_file, 'r', encoding='utf-8') as file:
    schema_linking_result_all = json.load(file)


@func_set_timeout(360)
def try_execute_sql(sql: str, db_id: str) -> dict:
    db_path = os.path.join(data_dir, 'dev_databases', db_id, db_id + '.sqlite')
    if not os.path.exists(db_path):
        print(db_path, "is not exists!")
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


def execute_sql(sql: str, db_id: str):
    try:
        return try_execute_sql(sql, db_id)
    except:
        return {}


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



def process_task(index, item, model, tokenizer):
    start_time = time.time()
    if not os.path.exists(os.path.join(save_dir, str(index))):
        os.makedirs(os.path.join(save_dir, str(index)))

    result_file = os.path.join(save_dir, str(index), 'result.json')
    log_file = os.path.join(save_dir, str(index), 'log.json')
    if os.path.exists(result_file):
        return
    
    question = item['question']
    evidence = item['evidence']
    sql = item['SQL']
    db_id = item['db_id']


    # 生成schema的prompt
    desc_str = ''
    column_list = []
    schema_linking_result = schema_linking_result_all[index]

    # table_names = dev_table_info[item['db_id'].lower()]['tables_info'].keys()
    table_names = list(schema_linking_result.keys())
    for table_name in table_names:
        # column_names = dev_table_info[item['db_id'].lower()]['tables_info'][table_name.lower()]
        column_names = schema_linking_result[table_name]
        desc_str += table_name + ' (`' + '`, `'.join(column_names) +'`) \n'
        column_list += [table_name + '.`' + column_name + '`' for column_name in column_names]
    
    # 读取csv文件生成column description和示例
    desc_str = ''
    for table_name in table_names:
        csv_file_dir = os.path.join(data_dir, 'dev_databases/{}/database_description'.format(db_id))
        csv_files = os.listdir(csv_file_dir)
        for csv_file in csv_files:
            if csv_file.lower() == table_name.lower() + '.csv':
                csv_file_path = os.path.join(data_dir, 'dev_databases/{}/database_description'.format(db_id), csv_file)
                try:
                    df = pd.read_csv(csv_file_path)
                except:
                    df = pd.read_csv(csv_file_path, encoding="ISO-8859-1")
                desc_str += '- Table: {table_name}\n'.format(table_name=table_name)
                # column_names = dev_table_info[item['db_id'].lower()]['tables_info'][table_name.lower()]
                column_names = schema_linking_result[table_name]
                for column_name in column_names:
                    desc_str += '\t- Column: `{column_name}` ({column_type})\n'.format(column_name=column_name, column_type=df[df['original_column_name'] == column_name]['data_format'].values[0] if len(df[df['original_column_name'] == column_name]['data_format'].values) > 0 else '')
                    try:
                        column_desc = df[df['original_column_name'] == column_name]['column_description'].values[0] if len(df[df['original_column_name'] == column_name]['column_description'].values) > 0 else ''
                        if column_desc.strip() != "":
                            desc_str += '\t\t- Description: {description}\n'.format(description=column_desc)
                    except:
                        pass
                    value_sql_query = "SELECT `{}` FROM {} LIMIT 10".format(column_name, table_name)
                    res = execute_sql(value_sql_query, db_id)
                    if 'data' in list(res.keys()):
                        res = res['data']
                        res = [str(res_item[0]) for res_item in res]
                        desc_str += '\t\t- Samples: [{sample}]\n'.format(sample=', '.join(res))
                        try:
                            value_desc = df[df['original_column_name'] == column_name]['value_description'].values[0] if len(df[df['original_column_name'] == column_name]['value_description'].values) > 0 else ''
                            if value_desc.strip() != "":
                                desc_str += '\t\t- Value Description: {description}\n'.format(description=value_desc)
                        except:
                            pass
                break

    fk_str = ''
    for foreign_key in dev_table_info[db_id]['foreign_keys']:
        if foreign_key[0] in column_list and foreign_key[1] in column_list:
            fk_str += foreign_key[0] + ' = ' + foreign_key[1] + '\n'
    
    
    messages = [{"role": "system", "content": instruction}, {"role": "user", "content": user_prompt.format(desc_str=desc_str, fk_str=fk_str, query=question, evidence=evidence)}]
    
    for try_index in range(5):
        # qwen
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=2048
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        with jsonlines.open(os.path.join(save_dir, str(index), 'token.json'), mode="w") as file_jsonl:
            file_jsonl.write({
                "index": index, 
                "input_token": model_inputs['input_ids'][0].shape[0],
                "output_token": generated_ids[0].shape[0]})

        # llama
        # input_ids = tokenizer.apply_chat_template(
        #     messages,
        #     add_generation_prompt=True,
        #     return_tensors="pt"
        # ).to(model.device)

        # terminators = [
        #     tokenizer.eos_token_id,
        #     tokenizer.convert_tokens_to_ids("<|eot_id|>")
        # ]

        # outputs = model.generate(
        #     input_ids,
        #     max_new_tokens=2048,
        #     eos_token_id=terminators,
        #     do_sample=True,
        #     temperature=0.6,
        #     top_p=0.9,
        # )
        # response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

        # glm4
        # inputs = tokenizer.apply_chat_template(
        #     messages,
        #     return_tensors='pt',
        #     add_generation_prompt=True,
        #     return_dict=True,
        # ).to(model.device)

        # input_len = inputs['input_ids'].shape[1]
        # generate_kwargs = {
        #     "input_ids": inputs['input_ids'],
        #     "attention_mask": inputs['attention_mask'],
        #     "max_new_tokens": 2048,
        #     "do_sample": False,
        # }
        # out = model.generate(**generate_kwargs)
        # response = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)

        with jsonlines.open(log_file, mode="a") as file_jsonl:
            file_jsonl.write({'type': 'gen_solution', 'messages': messages, 'response': response})
        
        last_sql = parse_sql_from_string(response)
        excute_result = execute_sql(last_sql, db_id)

        with jsonlines.open(log_file, mode="a") as file_jsonl:
            file_jsonl.write({'type': 'run_sql', 'sql': last_sql, 'result': excute_result})
        
        if 'data' in list(excute_result.keys()):
            gold_sql_run = 1
            try:
                correct_info = execute_sql(item['SQL'], db_id)
            except:
                correct_info = {}
                gold_sql_run = 0
            
            if gold_sql_run == 0 or 'data' not in list(correct_info.keys()):
                with jsonlines.open(result_file, mode="a") as file_jsonl:
                    file_jsonl.write({
                        "index": index, 
                        "gold_sql": item['SQL'],
                        "pred_sql": last_sql,
                        "gold_data": None,
                        "pred_data": excute_result['data'],
                        "Flag": 1,
                        "note": "gold sql is error"})
            else:
                flag = 1
                if len(set(correct_info['data'])) != len(set(excute_result['data'])):
                    flag = 0
                else:
                    for res_index in range(min(len(correct_info['data']), len(excute_result['data']))):
                        if set(correct_info['data'][res_index]) != set(excute_result['data'][res_index]):
                            flag = 0
                            break

                with jsonlines.open(result_file, mode="a") as file_jsonl:
                    file_jsonl.write({
                        "index": index, 
                        "gold_sql": item['SQL'],
                        "pred_sql": last_sql,
                        "gold_data": correct_info['data'],
                        "pred_data": excute_result['data'],
                        "Flag": flag})
            break
    end_time = time.time()
    with jsonlines.open(log_file, mode="a") as file_jsonl:
        file_jsonl.write({'type': 'time_cost', 'time': end_time-start_time})
    
def main_fun(arg_list):
    cuda_index, cuda_lists = arg_list
    model_path = 'Qwen2.5-Coder-7B-Instruct-bird/lora_merge'
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(cuda_index).eval()

    for index, item in enumerate(data_bird):
        if index % len(cuda_lists) != cuda_index:
            continue
        process_task(index, item, model, tokenizer)
    
if __name__ == '__main__':
    time_s = time.time()
    ctx = torch.multiprocessing.get_context("spawn")
    cuda_lists = [i for i in range(8)]
    pool = ctx.Pool(len(cuda_lists))
    arg_list = [(i, cuda_lists) for i in cuda_lists]
    pool.map(main_fun, arg_list)
    time_e = time.time()
    print(time_e - time_s)

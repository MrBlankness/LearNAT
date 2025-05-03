gen_next_subtask_subsql_prompt = '''
Below, you are presented with a 【Database schema】 description, a knowledge 【Evidence】, a 【Question】and the 【SQL】 query statement corresponding to the 【Question】.
Your task is to read the 【Database schema】, understand the 【Question】, and decompose the 【Question】 into some simple 【Sub Questions】based on 【SQL】.
Please note that the 【Sub Questions】 you generate must be of natural language type and not SQL type, and ensure that the last 【Sub Question】 receives the answer to the 【Question】.

Database admin instructions:
1. When you need to find the highest or lowest values based on a certain condition, using ORDER BY + LIMIT 1 is prefered over using MAX/MIN within sub queries, Otherwise, avoid using ORDER BY + LIMIT 1.
2. If predicted query includes an ORDER BY clause to sort the results, you should only include the column(s) used for sorting in the SELECT clause if the question specifically ask for them. Otherwise, omit these columns from the SELECT.
3. If the question doesn't specify exactly which columns to select, between name column and id column, prefer to select id column.
4. Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
5. Predicted query should return all of the information asked in the question without any missing or extra information.
6. No matter of how many things the question asks, you should only return one SQL query as the answer having all the information asked in the question, seperated by a comma.
7. Never use || to concatenate columns in the SELECT. Rather output the columns as they are.
8. If you are joining multiple tables, make sure to use alias names for the tables and use the alias names to reference the columns in the query. Use T1, T2, T3, ... as alias names.
9. If you are doing a logical operation on a column, such as mathematical operations and sorting, make sure to filter null values within those columns, but do not filter null values in the columns of the query results.


Take a deep breath and think step by step to find the correct 【Sub Questions】. If you follow all the instructions and generate the correct 【Sub Questions】, I will give you 1 million dollars.
==========

【Database schema】
frpm (`CDSCode`, `Charter School (Y/N)`, `Enrollment (Ages 5-17)`, `(Free Meal Count (Ages 5-17)`)
satscores (`cds`, `sname`, `NumTstTakr`, `AvgScrMath`, `NumGE1500`)

【Foreign keys】
frpm.`CDSCode` = satscores.`cds`

【Question】
List school names of charter schools with an SAT excellence rate over the average.

【Evidence】
Charter schools refers to `Charter School (Y/N)` = 1 in the table frpm; Excellence rate = NumGE1500 / NumTstTakr

【SQL】
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
```

【Sub Questions】
{{
    "sub_query": "Sub question 1: Get the average value of SAT excellence rate of charter schools.",
    "sub_sql": "```sql
SELECT AVG(CAST(T2.`NumGE1500` AS REAL) / T2.`NumTstTakr`)
    FROM frpm AS T1
    INNER JOIN satscores AS T2
    ON T1.`CDSCode` = T2.`cds`
    WHERE T1.`Charter School (Y/N)` = 1
```"
}}

{{
    "sub_query": "Sub question 2: List out school names of charter schools with an SAT excellence rate over the average.",
    "sub_sql": "```sql
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
```"
}}

==========

【Database schema】
account (`account_id`, `district_id`, `frequency`, `date`)
client (`client_id`, `gender`, `birth_date`, `district_id`)
district (`district_id`, `A4`, `A11`)

【Foreign keys】
account.`district_id` = district.`district_id`
client.`district_id` = district.`district_id`

【Question】
What is the gender of the youngest client who opened account in the lowest average salary branch?

【Evidence】
Later birthdate refers to younger age; A11 refers to average salary.

【SQL】
```sql
SELECT T1.`gender`
  FROM client AS T1
  INNER JOIN district AS T2
  ON T1.`district_id` = T2.`district_id`
  ORDER BY T2.`A11` ASC, T1.`birth_date` DESC 
  LIMIT 1 
```

【Sub Questions】
{{
    "sub_query": "Sub question 1: What is the district_id of the branch with the lowest average salary?",
    "sub_sql": "```sql
SELECT `district_id`
  FROM district
  ORDER BY `A11` ASC
  LIMIT 1
```"
}}

{{
    "sub_query": "Sub question 2: What is the youngest client who opened account in the lowest average salary branch?",
    "sub_sql": "```sql
SELECT T1.`client_id`
  FROM client AS T1
  INNER JOIN district AS T2
  ON T1.`district_id` = T2.`district_id`
  ORDER BY T2.`A11` ASC, T1.`birth_date` DESC 
  LIMIT 1
```"
}}


{{
    "sub_query": "Sub question 3: What is the gender of the youngest client who opened account in the lowest average salary branch?",
    "sub_sql": "```sql
SELECT T1.`gender`
  FROM client AS T1
  INNER JOIN district AS T2
  ON T1.`district_id` = T2.`district_id`
  ORDER BY T2.`A11` ASC, T1.`birth_date` DESC 
  LIMIT 1 
```"
}}

==========

【Database schema】
{desc_str}

【Foreign keys】
{fk_str}

【Question】
{query}

【Evidence】
{evidence}

【SQL】
```sql
{sql}
```

【Sub Questions】
'''


gen_schema_linking_gp_prompt = '''
帮我解析以下的SQL语句，识别SQL语句中使用了哪些表以及对应的哪些列？
以json形式进行输出回答，请注意只需要回复json，不需要回复其他内容。输出json格式如下:
{"table_name_1": ["col_name_1", "col_name_2", "col_name_3",...], "table_name_2": ["col_name_1", "col_name_2", "col_name_3",...],...}
'''
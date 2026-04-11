# Prompt templates for decomposition, schema linking, inference, etc.

# System prompt for the decomposition model
SYSTEM_PROMPT = """You are a data science expert specializing in translating natural language queries into SQL statements. 
Your task is to decompose complex NL2SQL queries into simpler subtasks, where each subtask has a clear natural language description and corresponding SQL statement.

Key principles:
1. Each subtask should be simpler than the original query
2. The subtasks should collectively construct the target SQL
3. Start with simpler operations (table selection, basic filtering) and build up to complex operations (joins, aggregations)
4. Each sub-SQL should be executable and produce meaningful intermediate results
"""

# Template for generating the next subtask in decomposition
GEN_NEXT_SUBTASK_SUBSQL_PROMPT = """Given the following database schema and user question, generate the next subtask in the decomposition process.

## Database Schema
{desc_str}

## Foreign Keys
{fk_str}

## User Question
{query}

## Additional Evidence/Hints
{evidence}

## Target SQL (for reference)
{sql}

## Previous Subtasks
{previous_steps}

## Instructions
Based on the above information, generate the next subtask that helps construct the target SQL.
Your response should be in the following JSON format:
{{
    "sub_query": "Natural language description of this subtask",
    "sub_sql": "SQL statement for this subtask",
    "reasoning": "Brief explanation of why this subtask is needed"
}}

Make sure the sub_sql is a valid, executable SQL statement that:
1. Is simpler than the target SQL
2. Contributes new information not covered by previous subtasks
3. Uses only tables and columns from the provided schema
"""

# Template for schema linking
GEN_SCHEMA_LINKING_GP_PROMPT = """Given the following SQL statement, extract the tables and columns used.

SQL Statement:
{sql}

Return a JSON object where keys are table names and values are lists of column names used from each table.
Example format:
{{
    "table1": ["column1", "column2"],
    "table2": ["column3"]
}}

Only include tables and columns that are actually used in the SQL statement.
"""

# Template for few-shot decomposition learning
FEW_SHOT_DECOMPOSITION_TEMPLATE = """## Example {idx}

### Question
{question}

### Schema
{schema}

### Target SQL
{sql}

### Decomposition
{decomposition}

---
"""

# Template for inference (generating SQL from NL query)
INFERENCE_PROMPT = """You are a data science expert. Given a database schema and natural language question, decompose the query into subtasks and generate the final SQL.

## Database Schema
{schema}

## Foreign Keys
{foreign_keys}

## Evidence/Hints
{evidence}

## Question
{question}

## Instructions
1. First, decompose the question into simpler subtasks
2. For each subtask, provide the natural language description and SQL
3. Finally, provide the complete SQL that answers the original question

Respond in the following format:
### Subtask 1
Description: [subtask description]
SQL: [SQL for this subtask]

### Subtask 2
Description: [subtask description]
SQL: [SQL for this subtask]

...

### Final SQL
```sql
[Complete SQL statement]
```
"""

# Template for SQL verification
SQL_VERIFICATION_PROMPT = """Verify if the following SQL statement is correct for the given question.

## Question
{question}

## Database Schema
{schema}

## Generated SQL
{sql}

## Execution Result
{result}

Does the SQL correctly answer the question? If not, explain what's wrong and provide a corrected SQL.

Response format:
{{
    "is_correct": true/false,
    "explanation": "Brief explanation",
    "corrected_sql": "Corrected SQL if needed, or null if correct"
}}
"""


def format_schema(tables_info: dict) -> str:
    """Render a schema dict as a readable string."""
    lines = []
    for table_name, columns in tables_info.items():
        col_str = ", ".join(columns)
        lines.append(f"- Table: {table_name}")
        lines.append(f"  Columns: {col_str}")
    return "\n".join(lines)


def format_foreign_keys(foreign_keys: list) -> str:
    if not foreign_keys:
        return "None"
    
    lines = []
    for fk in foreign_keys:
        lines.append(f"- {fk[0]} = {fk[1]}")
    return "\n".join(lines)


def format_previous_steps(steps: list) -> str:
    if not steps:
        return "None (this is the first subtask)"
    
    lines = []
    for i, (query, sql) in enumerate(steps, 1):
        lines.append(f"Step {i}:")
        lines.append(f"  Description: {query}")
        lines.append(f"  SQL: {sql}")
    return "\n".join(lines)


def format_decomposition(decomposition: list) -> str:
    lines = []
    for i, (query, sql) in enumerate(decomposition, 1):
        lines.append(f"**Step {i}:**")
        lines.append(f"- Description: {query}")
        lines.append(f"- SQL: `{sql}`")
    return "\n".join(lines)


def build_decomposition_prompt(
    question: str,
    schema: dict,
    foreign_keys: list,
    evidence: str,
    target_sql: str,
    previous_steps: list = None,
    demonstrations: list = None
) -> str:
    """Assemble the full prompt for generating the next decomposition step."""
    # Build demonstration section if provided
    demo_section = ""
    if demonstrations:
        demo_section = "## Few-shot Examples\n\n"
        for idx, demo in enumerate(demonstrations, 1):
            demo_section += FEW_SHOT_DECOMPOSITION_TEMPLATE.format(
                idx=idx,
                question=demo["question"],
                schema=format_schema(demo["schema"]),
                sql=demo["sql"],
                decomposition=format_decomposition(demo["decomposition"])
            )
        demo_section += "\n## Now, solve the following:\n\n"
    
    # Build main prompt
    prompt = GEN_NEXT_SUBTASK_SUBSQL_PROMPT.format(
        desc_str=format_schema(schema),
        fk_str=format_foreign_keys(foreign_keys),
        query=question,
        evidence=evidence or "None",
        sql=target_sql,
        previous_steps=format_previous_steps(previous_steps or [])
    )
    
    return demo_section + prompt


def build_inference_prompt(
    question: str,
    schema: dict,
    foreign_keys: list,
    evidence: str = "",
    demonstrations: list = None
) -> str:
    """Build the inference prompt (no target SQL)."""
    # Build demonstration section if provided
    demo_section = ""
    if demonstrations:
        demo_section = "## Few-shot Examples\n\n"
        for idx, demo in enumerate(demonstrations, 1):
            demo_section += f"""### Example {idx}
Question: {demo["question"]}
Final SQL: {demo["sql"]}

---
"""
    
    prompt = INFERENCE_PROMPT.format(
        schema=format_schema(schema),
        foreign_keys=format_foreign_keys(foreign_keys),
        evidence=evidence or "None provided",
        question=question
    )
    
    return demo_section + prompt


if __name__ == "__main__":
    # Test prompt generation
    print("=" * 60)
    print("Testing Prompt Templates")
    print("=" * 60)
    
    # Sample data
    schema = {
        "users": ["id", "name", "age", "email"],
        "orders": ["id", "user_id", "amount", "date"]
    }
    foreign_keys = [["orders.user_id", "users.id"]]
    question = "Find all users who have orders over $100"
    target_sql = "SELECT users.name FROM users INNER JOIN orders ON users.id = orders.user_id WHERE orders.amount > 100"
    
    # Test decomposition prompt
    print("\n1. Decomposition Prompt:")
    print("-" * 40)
    prompt = build_decomposition_prompt(
        question=question,
        schema=schema,
        foreign_keys=foreign_keys,
        evidence="amount is in dollars",
        target_sql=target_sql,
        previous_steps=[("Get all users", "SELECT * FROM users")]
    )
    print(prompt[:500] + "...")
    
    # Test inference prompt
    print("\n2. Inference Prompt:")
    print("-" * 40)
    inf_prompt = build_inference_prompt(
        question=question,
        schema=schema,
        foreign_keys=foreign_keys,
        evidence="amount is in dollars"
    )
    print(inf_prompt[:500] + "...")
    
    print("\n" + "=" * 60)
    print("Prompt template tests completed!")
    print("=" * 60)

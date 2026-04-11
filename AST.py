# SQL AST parsing, subtree matching, similarity computation, and alias removal.

import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Where, Token, Parenthesis
from sqlparse.tokens import DML, Keyword
import re
from typing import Optional, Dict, List, Tuple


class ASTNode:
    """Single node in the SQL AST."""
    
    def __init__(self, node_type: str, value: Optional[str] = None):
        self.node_type = node_type
        self.value = value
        self.children: List['ASTNode'] = []

    def add_child(self, child: 'ASTNode') -> None:
        self.children.append(child)

    def count_nodes(self) -> int:
        count = 1
        for child in self.children:
            count += child.count_nodes()
        return count
    
    def get_all_nodes(self) -> List['ASTNode']:
        nodes = [self]
        for child in self.children:
            nodes.extend(child.get_all_nodes())
        return nodes
    
    def get_nodes_by_type(self, node_type: str) -> List['ASTNode']:
        nodes = []
        if self.node_type == node_type:
            nodes.append(self)
        for child in self.children:
            nodes.extend(child.get_nodes_by_type(node_type))
        return nodes

    def __repr__(self, level: int = 0) -> str:
        ret = "\t" * level + f"{self.node_type}: {self.value}\n"
        for child in self.children:
            ret += child.__repr__(level + 1)
        return ret
    
    def to_dict(self) -> Dict:
        return {
            'type': self.node_type,
            'value': self.value,
            'children': [child.to_dict() for child in self.children]
        }


def parse_sql_to_ast(sql: str) -> ASTNode:
    """Parse a SQL string into an AST, normalizing to lowercase first."""
    # Normalize SQL to lowercase for case-insensitive comparison
    sql_lower = sql.lower()
    parsed = sqlparse.parse(sql_lower)[0]
    root = ASTNode("SQL")
    process_tokens(parsed.tokens, root)
    return root


def process_tokens(tokens, parent_node: ASTNode) -> None:
    """Walk sqlparse tokens and attach corresponding ASTNodes to *parent_node*."""
    for token in tokens:
        # Skip whitespace
        if token.is_whitespace:
            continue
        
        # Handle SELECT statement
        if token.ttype is DML and token.value.upper() == "SELECT":
            select_node = ASTNode("SELECT", token.value)
            parent_node.add_child(select_node)
        
        # Handle column lists
        elif isinstance(token, IdentifierList):
            for identifier in token.get_identifiers():
                column_node = ASTNode("COLUMN", identifier.value)
                parent_node.add_child(column_node)
        
        elif isinstance(token, Identifier):
            column_node = ASTNode("COLUMN", token.value)
            parent_node.add_child(column_node)
        
        # Handle WHERE clause
        elif isinstance(token, Where):
            where_node = ASTNode("WHERE", "WHERE")
            parent_node.add_child(where_node)
            process_tokens(token.tokens, where_node)
        
        # Handle FROM and JOIN keywords
        elif token.ttype is Keyword and token.value.upper() in ("FROM", "JOIN", "INNER JOIN", "LEFT JOIN", "RIGHT JOIN", "OUTER JOIN"):
            from_node = ASTNode(token.value.upper(), token.value)
            parent_node.add_child(from_node)
        
        # Handle GROUP BY
        elif token.ttype is Keyword and token.value.upper() == "GROUP BY":
            group_node = ASTNode("GROUP BY", token.value)
            parent_node.add_child(group_node)
        
        # Handle ORDER BY
        elif token.ttype is Keyword and token.value.upper() == "ORDER BY":
            order_node = ASTNode("ORDER BY", token.value)
            parent_node.add_child(order_node)
        
        # Handle HAVING
        elif token.ttype is Keyword and token.value.upper() == "HAVING":
            having_node = ASTNode("HAVING", token.value)
            parent_node.add_child(having_node)
        
        # Handle LIMIT
        elif token.ttype is Keyword and token.value.upper() == "LIMIT":
            limit_node = ASTNode("LIMIT", token.value)
            parent_node.add_child(limit_node)
        
        # Handle subqueries and nested structures
        elif token.is_group:
            subquery_node = ASTNode("SQL")
            parent_node.add_child(subquery_node)
            process_tokens(token.tokens, subquery_node)
        
        # Handle other keywords
        elif token.ttype is Keyword:
            keyword_node = ASTNode("KEYWORD", token.value)
            parent_node.add_child(keyword_node)
        
        # Handle other values (operators, literals, etc.)
        else:
            value_node = ASTNode("VALUE", token.value)
            parent_node.add_child(value_node)


def is_subtree(main_tree: ASTNode, sub_tree: ASTNode) -> bool:
    """Return True if *sub_tree* is contained within *main_tree*."""
    if sub_tree is None:
        return True
    if main_tree is None:
        return False
    
    # Check if current nodes match
    if is_same_tree(main_tree, sub_tree):
        return True
    
    # Recursively check children
    for child in main_tree.children:
        if is_subtree(child, sub_tree):
            return True
    
    return False


def is_same_tree(tree1: ASTNode, tree2: ASTNode) -> bool:
    """Check structural equivalence of two ASTs."""
    if tree1 is None and tree2 is None:
        return True
    if tree1 is None or tree2 is None:
        return False
    
    # Compare node type and value
    if tree1.node_type != tree2.node_type or tree1.value != tree2.value:
        return False
    
    # Check that tree2's children are a subset of tree1's children
    for sub_child in tree2.children:
        if not any(is_same_tree(main_child, sub_child) for main_child in tree1.children):
            return False
    
    return True


def remove_aliases(sql: str) -> str:
    """Strip table aliases from a SQL string so that comparisons are alias-agnostic."""
    def process_query(query: str, alias_to_table: Optional[Dict[str, str]] = None) -> str:
        if alias_to_table is None:
            alias_to_table = {}
        
        # SQL keywords that should NOT be treated as aliases
        SQL_KEYWORDS = {
            'SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER', 'LEFT', 'RIGHT', 'OUTER',
            'CROSS', 'ON', 'AND', 'OR', 'NOT', 'IN', 'EXISTS', 'BETWEEN', 'LIKE',
            'IS', 'NULL', 'TRUE', 'FALSE', 'GROUP', 'ORDER', 'BY', 'HAVING',
            'LIMIT', 'OFFSET', 'UNION', 'ALL', 'DISTINCT', 'AS', 'SET', 'INTO',
            'VALUES', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER',
            'INDEX', 'TABLE', 'VIEW', 'CASE', 'WHEN', 'THEN', 'ELSE', 'END',
            'ASC', 'DESC', 'NATURAL', 'USING', 'FULL', 'WITH', 'RECURSIVE',
            'EXCEPT', 'INTERSECT', 'PRIMARY', 'KEY', 'FOREIGN', 'REFERENCES',
            'CHECK', 'DEFAULT', 'CONSTRAINT', 'UNIQUE', 'CASCADE', 'RESTRICT',
        }
        
        # Match FROM and JOIN clauses with table names and explicit AS aliases
        # Pattern 1: FROM/JOIN table AS alias
        explicit_alias_pattern = re.compile(
            r'\b(FROM|JOIN)\s+([`"]?\w+[`"]?)\s+AS\s+(\w+)\b', re.IGNORECASE)
        local_alias_to_table = alias_to_table.copy()

        def replace_explicit_alias(match):
            table_name = match.group(2).strip('`"')
            alias_name = match.group(3)
            local_alias_to_table[alias_name] = table_name
            return f"{match.group(1)} {table_name}"

        query = explicit_alias_pattern.sub(replace_explicit_alias, query)
        
        # Pattern 2: FROM/JOIN table alias (implicit alias, no AS keyword)
        # Only match if the word after the table name is NOT a SQL keyword
        implicit_alias_pattern = re.compile(
            r'\b(FROM|JOIN)\s+([`"]?\w+[`"]?)\s+(\w+)\b', re.IGNORECASE)
        
        def replace_implicit_alias(match):
            keyword = match.group(1)
            table_name = match.group(2).strip('`"')
            potential_alias = match.group(3)
            # Don't treat SQL keywords as aliases
            if potential_alias.upper() in SQL_KEYWORDS:
                return match.group(0)  # Return unchanged
            local_alias_to_table[potential_alias] = table_name
            return f"{keyword} {table_name}"

        query = implicit_alias_pattern.sub(replace_implicit_alias, query)

        # Replace field alias prefixes
        field_pattern = re.compile(r'(\b\w+)\.(\w+|`[^`]+`)', re.IGNORECASE)

        def replace_field_alias(match):
            alias_name, field_name = match.groups()
            if alias_name in local_alias_to_table:
                return f"{local_alias_to_table[alias_name]}.{field_name}"
            return match.group(0)

        query = field_pattern.sub(replace_field_alias, query)

        # Recursively process subqueries
        subquery_pattern = re.compile(r'\((SELECT .*?)\)', re.IGNORECASE | re.DOTALL)

        def process_subquery(match):
            inner_query = match.group(1)
            processed_inner_query = process_query(inner_query, local_alias_to_table)
            return f"({processed_inner_query})"

        query = subquery_pattern.sub(process_subquery, query)

        return query

    return process_query(sql)


def tree_edit_distance(t1: ASTNode, t2: ASTNode) -> int:
    """Simplified tree edit distance between two ASTs."""
    if t1 is None and t2 is None:
        return 0
    if t1 is None:
        return t2.count_nodes()
    if t2 is None:
        return t1.count_nodes()
    
    # Compare current nodes
    if t1.node_type != t2.node_type or t1.value != t2.value:
        cost = 1
    else:
        cost = 0
    
    # Cost for different number of children
    cost += abs(len(t1.children) - len(t2.children))
    
    # Recursively compute for matching children
    common = min(len(t1.children), len(t2.children))
    for i in range(common):
        cost += tree_edit_distance(t1.children[i], t2.children[i])
    
    return cost


def calculate_ast_similarity(ast1: ASTNode, ast2: ASTNode, alpha: float = 0.75) -> float:
    """Combined node-level + structural similarity (Eq. 5): alpha * sim_node + (1-alpha) * sim_struct."""
    # Node-level similarity
    sim_node = calculate_node_similarity(ast1, ast2)
    
    # Structural similarity
    sim_struct = calculate_structural_similarity(ast1, ast2)
    
    return alpha * sim_node + (1 - alpha) * sim_struct


def calculate_node_similarity(ast1: ASTNode, ast2: ASTNode, 
                              weights: Optional[Dict[str, float]] = None) -> float:
    """Jaccard-based node similarity across clause / operator / operand categories."""
    if weights is None:
        # Equal weights as per paper F.1: w_c = w_o = w_v = 0.33
        weights = {
            'clause': 0.33,    # SELECT, FROM, WHERE, etc.
            'operator': 0.33,  # AND, OR, =, >, etc.
            'operand': 0.34    # Values, column names, table names
        }
    
    # Get nodes by category
    clause_types = {'SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING', 'LIMIT', 'JOIN'}
    operator_types = {'KEYWORD'}
    
    def categorize_nodes(ast: ASTNode) -> Dict[str, set]:
        all_nodes = ast.get_all_nodes()
        categories = {'clause': set(), 'operator': set(), 'operand': set()}
        
        for node in all_nodes:
            if node.node_type in clause_types:
                categories['clause'].add((node.node_type, node.value))
            elif node.node_type in operator_types:
                categories['operator'].add((node.node_type, node.value))
            else:
                categories['operand'].add((node.node_type, node.value))
        
        return categories
    
    cat1 = categorize_nodes(ast1)
    cat2 = categorize_nodes(ast2)
    
    total_sim = 0.0
    for cat_type, weight in weights.items():
        set1 = cat1[cat_type]
        set2 = cat2[cat_type]
        
        if len(set1) == 0 and len(set2) == 0:
            sim_t = 1.0
        else:
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            sim_t = intersection / union if union > 0 else 0
        
        total_sim += weight * sim_t
    
    return total_sim


def calculate_structural_similarity(ast1: ASTNode, ast2: ASTNode) -> float:
    """1 - TED / max(|T1|, |T2|)."""
    ted = tree_edit_distance(ast1, ast2)
    max_nodes = max(ast1.count_nodes(), ast2.count_nodes())
    
    if max_nodes == 0:
        return 1.0
    
    return 1 - ted / max_nodes


def merge_asts(ast_list: List[ASTNode]) -> ASTNode:
    """Merge multiple ASTs into one by unioning unique nodes."""
    if not ast_list:
        return ASTNode("SQL")
    
    merged = ASTNode("SQL")
    
    # Collect all unique nodes and edges
    seen_nodes = set()
    
    def add_nodes_from_ast(ast: ASTNode, parent: ASTNode):
        for child in ast.children:
            node_key = (child.node_type, child.value)
            if node_key not in seen_nodes:
                seen_nodes.add(node_key)
                new_child = ASTNode(child.node_type, child.value)
                parent.add_child(new_child)
                add_nodes_from_ast(child, new_child)
            else:
                # Find existing node and add its children
                for existing_child in parent.children:
                    if (existing_child.node_type, existing_child.value) == node_key:
                        add_nodes_from_ast(child, existing_child)
                        break
    
    for ast in ast_list:
        add_nodes_from_ast(ast, merged)
    
    return merged


def normalize_sub_sql(sub_sql: str, target_sql: str) -> str:
    """
    Normalize a sub-SQL against the target to improve AST subtree matching.

    Steps: remove column aliases, add table prefixes, expand SELECT *,
    normalize JOIN types, filter extra SELECT columns, align SELECT
    with target when FROM/WHERE match.
    """
    # Convert both to lowercase since SQL is case-insensitive
    normalized = sub_sql.strip().lower()
    target_sql = target_sql.lower()
    if not normalized:
        return normalized
    
    # Step 1: Remove column aliases (AS alias_name) from SELECT clause
    normalized = _remove_column_aliases(normalized)
    
    # Step 2: Build column-to-table mapping from target SQL
    col_to_table = _extract_column_table_mapping(target_sql)
    
    # Step 3: Add table prefixes to bare column names
    normalized = _add_table_prefixes(normalized, col_to_table)
    
    # Step 4: Replace SELECT * with columns from target SQL
    normalized = _replace_select_star(normalized, target_sql)
    
    # Step 5: Normalize JOIN types to match target
    normalized = _normalize_join_types(normalized, target_sql)
    
    # Step 6: Filter extra SELECT columns not in target's SELECT list
    normalized = _filter_extra_select_columns(normalized, target_sql)
    
    # Step 7: Align SELECT columns with target when FROM/WHERE match
    # This handles cases where LLM generates simple SELECT (e.g., SELECT col FROM ...)
    # but target has complex expressions (e.g., CAST(SUM(CASE WHEN ... END) AS REAL))
    # If the sub-SQL's FROM and WHERE clauses are compatible with the target,
    # replace the sub-SQL's SELECT with the target's SELECT to improve matching.
    normalized = _align_select_with_target(normalized, target_sql)
    
    return normalized


def _remove_column_aliases(sql: str) -> str:
    """Drop 'AS alias' from the SELECT clause."""
    # Find the SELECT...FROM portion
    from_match = re.search(r'\bFROM\b', sql, re.IGNORECASE)
    if not from_match:
        return sql
    
    select_part = sql[:from_match.start()]
    rest_part = sql[from_match.start():]
    
    # Remove AS alias patterns in SELECT clause
    # Handle: expression AS alias_name
    # Be careful not to remove AS in subqueries
    # Pattern: match " AS word" but not inside parentheses at the top level
    
    # Simple approach: remove " AS word_or_quoted" patterns
    # This handles: col AS alias, func(x) AS alias, etc.
    select_part = re.sub(
        r'\s+AS\s+[`"\']?\w+[`"\']?(?=\s*,|\s*$)',
        '',
        select_part,
        flags=re.IGNORECASE
    )
    
    return select_part + rest_part


def _extract_column_table_mapping(target_sql: str) -> Dict[str, str]:
    """Build a column->table mapping from `table.column` patterns in the target SQL."""
    col_to_tables: Dict[str, set] = {}
    
    # Find all table.column patterns
    pattern = re.compile(r'\b(\w+)\.(\w+)\b')
    for match in pattern.finditer(target_sql):
        table_name = match.group(1).lower()
        col_name = match.group(2).lower()
        
        # Skip SQL keywords that might look like table.column
        sql_keywords = {'group', 'order', 'inner', 'left', 'right', 'outer', 'cross'}
        if table_name in sql_keywords:
            continue
        
        if col_name not in col_to_tables:
            col_to_tables[col_name] = set()
        col_to_tables[col_name].add(table_name)
    
    # Only include unambiguous mappings (column appears in exactly one table)
    col_to_table = {}
    for col_name, tables in col_to_tables.items():
        if len(tables) == 1:
            col_to_table[col_name] = next(iter(tables))
    
    return col_to_table


def _add_table_prefixes(sql: str, col_to_table: Dict[str, str]) -> str:
    """Prefix bare column names with their table name when unambiguous."""
    if not col_to_table:
        return sql
    
    # SQL keywords and functions that should not be prefixed
    skip_words = {
        'select', 'from', 'where', 'join', 'inner', 'left', 'right', 'outer',
        'on', 'and', 'or', 'not', 'in', 'exists', 'between', 'like', 'is',
        'null', 'true', 'false', 'group', 'order', 'by', 'having', 'limit',
        'offset', 'union', 'all', 'distinct', 'as', 'asc', 'desc', 'case',
        'when', 'then', 'else', 'end', 'count', 'sum', 'avg', 'min', 'max',
        'strftime', 'cast', 'coalesce', 'ifnull', 'nullif', 'typeof',
        'length', 'substr', 'replace', 'trim', 'upper', 'lower', 'abs',
        'round', 'date', 'time', 'datetime', 'julianday',
    }
    
    # Use word boundary based regex with lookahead/lookbehind to avoid consuming chars
    # Match bare words that are NOT preceded by '.' and NOT followed by '.' or '('
    def replace_match(match):
        word = match.group(0)
        word_lower = word.lower()
        
        if word_lower in skip_words:
            return word
        
        if word_lower in col_to_table:
            return f"{col_to_table[word_lower]}.{word}"
        
        return word
    
    # Pattern: word boundary, word, word boundary
    # Negative lookbehind for '.' (not already prefixed)
    # Negative lookahead for '.' (not a table name) and '(' (not a function)
    result = re.sub(
        r'(?<!\.)(?<!\w)\b(\w+)\b(?!\.|\()',
        replace_match,
        sql
    )
    
    return result

def _replace_select_star(sql: str, target_sql: str) -> str:
    """Replace `SELECT *` with the column list from the target SQL."""
    # Check if sub-SQL has SELECT *
    if not re.search(r'\bSELECT\s+\*\s+FROM\b', sql, re.IGNORECASE):
        return sql
    
    # Extract SELECT columns from target SQL
    target_select_match = re.match(
        r'\s*SELECT\s+(.*?)\s+FROM\b',
        target_sql,
        re.IGNORECASE | re.DOTALL
    )
    if not target_select_match:
        return sql
    
    target_columns = target_select_match.group(1).strip()
    
    # Don't replace if target also uses *
    if target_columns.strip() == '*':
        return sql
    
    # Extract tables used in sub-SQL's FROM clause
    sub_tables = set()
    from_pattern = re.compile(r'\b(?:FROM|JOIN)\s+(\w+)', re.IGNORECASE)
    for m in from_pattern.finditer(sql):
        sub_tables.add(m.group(1).lower())
    
    # Filter target columns to only those from tables in the sub-SQL
    # Parse target columns (split by comma, handling nested parentheses)
    target_col_list = _split_columns(target_columns)
    
    relevant_cols = []
    for col in target_col_list:
        col_stripped = col.strip()
        # Check if column references a table in our sub-SQL
        table_match = re.match(r'(\w+)\.', col_stripped)
        if table_match:
            if table_match.group(1).lower() in sub_tables:
                relevant_cols.append(col_stripped)
        else:
            # Column without table prefix - include it
            relevant_cols.append(col_stripped)
    
    if relevant_cols:
        replacement_cols = ', '.join(relevant_cols)
    else:
        # If no relevant columns found, use all target columns
        replacement_cols = target_columns
    
    # Replace SELECT * with the columns
    result = re.sub(
        r'\bSELECT\s+\*\s+FROM\b',
        f'SELECT {replacement_cols} FROM',
        sql,
        count=1,
        flags=re.IGNORECASE
    )
    
    return result


def _split_columns(columns_str: str) -> List[str]:
    """Split comma-separated columns respecting parentheses nesting."""
    result = []
    depth = 0
    current = []
    
    for char in columns_str:
        if char == '(':
            depth += 1
            current.append(char)
        elif char == ')':
            depth -= 1
            current.append(char)
        elif char == ',' and depth == 0:
            result.append(''.join(current).strip())
            current = []
        else:
            current.append(char)
    
    if current:
        result.append(''.join(current).strip())
    
    return result


def _normalize_join_types(sql: str, target_sql: str) -> str:
    """Align JOIN / INNER JOIN usage with the target SQL."""
    # Check what JOIN type the target uses
    has_inner_join = bool(re.search(r'\bINNER\s+JOIN\b', target_sql, re.IGNORECASE))
    has_plain_join = bool(re.search(r'(?<!INNER\s)(?<!LEFT\s)(?<!RIGHT\s)(?<!OUTER\s)(?<!CROSS\s)\bJOIN\b', target_sql, re.IGNORECASE))
    
    if has_inner_join and not has_plain_join:
        # Target uses INNER JOIN, normalize plain JOIN to INNER JOIN
        sql = re.sub(
            r'(?<!INNER\s)(?<!LEFT\s)(?<!RIGHT\s)(?<!OUTER\s)(?<!CROSS\s)\bJOIN\b',
            'INNER JOIN',
            sql,
            flags=re.IGNORECASE
        )
    elif has_plain_join and not has_inner_join:
        # Target uses plain JOIN, normalize INNER JOIN to JOIN
        sql = re.sub(
            r'\bINNER\s+JOIN\b',
            'JOIN',
            sql,
            flags=re.IGNORECASE
        )
    
    return sql


def _filter_extra_select_columns(sql: str, target_sql: str) -> str:
    """Remove SELECT columns that don't appear in the target's SELECT list."""
    # Extract SELECT columns from both sub and target SQL
    sub_select_match = re.match(
        r'\s*SELECT\s+(.*?)\s+FROM\b',
        sql,
        re.IGNORECASE | re.DOTALL
    )
    target_select_match = re.match(
        r'\s*SELECT\s+(.*?)\s+FROM\b',
        target_sql,
        re.IGNORECASE | re.DOTALL
    )
    
    if not sub_select_match or not target_select_match:
        return sql
    
    sub_columns_str = sub_select_match.group(1).strip()
    target_columns_str = target_select_match.group(1).strip()
    
    # Parse column lists
    sub_columns = _split_columns(sub_columns_str)
    target_columns = _split_columns(target_columns_str)
    
    # If sub has only 1 column or same number as target, no filtering needed
    if len(sub_columns) <= 1 or len(sub_columns) <= len(target_columns):
        return sql
    
    # Normalize target columns for comparison (lowercase, strip whitespace)
    target_cols_normalized = set()
    for col in target_columns:
        target_cols_normalized.add(col.strip().lower())
    
    # Filter: keep only columns that appear in the target's SELECT list
    filtered_columns = []
    for col in sub_columns:
        col_normalized = col.strip().lower()
        if col_normalized in target_cols_normalized:
            filtered_columns.append(col.strip())
    
    # If no columns matched, try a more lenient match (check if column name is contained)
    if not filtered_columns:
        for col in sub_columns:
            col_normalized = col.strip().lower()
            for target_col in target_cols_normalized:
                # Check if the core column name matches
                if col_normalized in target_col or target_col in col_normalized:
                    filtered_columns.append(col.strip())
                    break
    
    # If still no match, keep original (don't break the SQL)
    if not filtered_columns:
        return sql
    
    # Rebuild the SQL with filtered columns
    new_columns_str = ', '.join(filtered_columns)
    rest_part = sql[sub_select_match.end():]
    result = f"SELECT {new_columns_str} FROM{rest_part}"
    
    return result


def _align_select_with_target(sql: str, target_sql: str) -> str:
    """
    When FROM/WHERE match but SELECT differs, try replacing the sub-SQL's
    SELECT with the target's SELECT (useful for complex expressions like
    CAST, CASE, aggregates).
    """
    # Extract SELECT and FROM parts from both SQLs
    sub_select_match = re.match(
        r'\s*SELECT\s+(.*?)\s+FROM\b',
        sql,
        re.IGNORECASE | re.DOTALL
    )
    target_select_match = re.match(
        r'\s*SELECT\s+(.*?)\s+FROM\b',
        target_sql,
        re.IGNORECASE | re.DOTALL
    )
    
    if not sub_select_match or not target_select_match:
        return sql
    
    sub_select = sub_select_match.group(1).strip()
    target_select = target_select_match.group(1).strip()
    
    # If SELECT clauses are already the same, no need to align
    if sub_select.lower() == target_select.lower():
        return sql
    
    # Only apply when target has complex expressions (CAST, CASE, IIF, arithmetic, aggregates)
    has_complex = any(kw in target_select.lower() for kw in [
        'cast(', 'case ', 'iif(', ' * ', ' / ', ' + ', ' - ',
        'avg(', 'sum(', 'count(', 'max(', 'min(', 'length('
    ])
    if not has_complex:
        return sql
    
    # Extract tables from sub-SQL's FROM clause
    sub_from_part = sql[sub_select_match.end():]
    target_from_part = target_sql[target_select_match.end():]
    
    sub_tables = set()
    target_tables = set()
    table_pattern = re.compile(r'\b(?:from|join)\s+(\w+)', re.IGNORECASE)
    for m in table_pattern.finditer('FROM' + sub_from_part):
        sub_tables.add(m.group(1).lower())
    for m in table_pattern.finditer('FROM' + target_from_part):
        target_tables.add(m.group(1).lower())
    
    # Sub-SQL's tables must be a subset of target's tables
    if not sub_tables or not sub_tables.issubset(target_tables):
        return sql
    
    # Extract WHERE clause from both (if present)
    sub_where = ''
    target_where = ''
    sub_where_match = re.search(r'\bWHERE\b\s+(.*?)(?:\bGROUP\b|\bORDER\b|\bLIMIT\b|\bHAVING\b|$)', 
                                 sql, re.IGNORECASE | re.DOTALL)
    target_where_match = re.search(r'\bWHERE\b\s+(.*?)(?:\bGROUP\b|\bORDER\b|\bLIMIT\b|\bHAVING\b|$)', 
                                    target_sql, re.IGNORECASE | re.DOTALL)
    if sub_where_match:
        sub_where = sub_where_match.group(1).strip().lower()
    if target_where_match:
        target_where = target_where_match.group(1).strip().lower()
    
    # WHERE clauses should be compatible (sub's WHERE is contained in target's WHERE)
    # Allow empty sub WHERE (selecting all rows) or matching WHERE
    if sub_where and target_where and sub_where not in target_where:
        return sql
    
    # Replace sub-SQL's SELECT with target's SELECT
    rest_part = sql[sub_select_match.end():]
    aligned = f"SELECT {target_select} FROM{rest_part}"
    
    return aligned


class ASTClassifier:
    """Classify MCTS nodes as progressive / redundant / invalid based on AST subtree checks."""
    
    @staticmethod
    def classify_node(sub_ast: ASTNode, target_ast: ASTNode, 
                      parent_merged_ast: Optional[ASTNode] = None) -> str:
        # Check if sub_ast is a subtree of target
        is_valid_subtree = is_subtree(target_ast, sub_ast)
        
        if not is_valid_subtree:
            return 'invalid'
        
        # Check if it's already covered by parent
        if parent_merged_ast is not None:
            is_redundant = is_subtree(parent_merged_ast, sub_ast)
            if is_redundant:
                return 'redundant'
        
        return 'progressive'


if __name__ == "__main__":
    # Test cases from the paper
    print("=" * 60)
    print("Testing AST Module")
    print("=" * 60)
    
    # Test 1: Basic SQL parsing
    sql = """
    SELECT name, age FROM users WHERE age > 18 AND id IN (
        SELECT user_id FROM orders WHERE amount > 100
    );
    """
    ast = parse_sql_to_ast(sql)
    print("\nTest 1: Basic SQL Parsing")
    print(ast)
    
    # Test 2: Subtree detection
    subsql = """
    SELECT user_id FROM orders WHERE amount > 100
    """
    subast = parse_sql_to_ast(subsql)
    print("\nTest 2: Subtree Detection")
    print(f"Is subtree: {is_subtree(ast, subast)}")
    
    # Test 3: Alias removal
    sql_with_alias = "SELECT T1.name, T2.order_id FROM users AS T1 INNER JOIN orders AS T2 ON T1.id = T2.user_id WHERE T1.age > 18"
    sql_no_alias = remove_aliases(sql_with_alias)
    print("\nTest 3: Alias Removal")
    print(f"Original: {sql_with_alias}")
    print(f"Without aliases: {sql_no_alias}")
    
    # Test 4: AST similarity
    sql1 = "SELECT name FROM users WHERE age > 18"
    sql2 = "SELECT name, email FROM users WHERE age > 18 AND active = 1"
    ast1 = parse_sql_to_ast(sql1)
    ast2 = parse_sql_to_ast(sql2)
    similarity = calculate_ast_similarity(ast1, ast2)
    print("\nTest 4: AST Similarity")
    print(f"Similarity: {similarity:.4f}")
    
    # Test 5: Node classification
    print("\nTest 5: Node Classification")
    classifier = ASTClassifier()
    target_sql = "SELECT name, email FROM users WHERE age > 18"
    target_ast = parse_sql_to_ast(target_sql)
    
    # Progressive node
    sub1 = parse_sql_to_ast("SELECT name FROM users")
    print(f"Classification of 'SELECT name FROM users': {classifier.classify_node(sub1, target_ast)}")
    
    # Invalid node  
    sub2 = parse_sql_to_ast("SELECT salary FROM employees")
    print(f"Classification of 'SELECT salary FROM employees': {classifier.classify_node(sub2, target_ast)}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)

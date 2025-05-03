class ASTNode:
    def __init__(self, node_type, value=None):
        self.node_type = node_type  # 节点类型
        self.value = value          # 节点值
        self.children = []          # 子节点列表

    def add_child(self, child):
        """添加子节点"""
        self.children.append(child)

    def __repr__(self, level=0):
        """打印树结构"""
        ret = "\t" * level + f"{self.node_type}: {self.value}\n"
        for child in self.children:
            ret += child.__repr__(level + 1)
        return ret


import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Where, Token
from sqlparse.tokens import DML, Keyword

def parse_sql_to_ast(sql):
    """将 SQL 解析为抽象语法树"""
    parsed = sqlparse.parse(sql)[0]  # 解析 SQL 并获取第一个语句
    root = ASTNode("SQL")            # 创建根节点
    process_tokens(parsed.tokens, root)  # 递归处理语法树
    return root

def process_tokens(tokens, parent_node):
    """递归处理 SQL 语句的 tokens，构建 AST"""
    for token in tokens:
        # 忽略空白字符
        if token.is_whitespace:
            continue
        
        # 处理 SELECT 语句
        if token.ttype is DML and token.value.upper() == "SELECT":
            select_node = ASTNode("SELECT", token.value)
            parent_node.add_child(select_node)
        
        # 处理列列表
        elif isinstance(token, IdentifierList):
            for identifier in token.get_identifiers():
                column_node = ASTNode("COLUMN", identifier.value)
                parent_node.add_child(column_node)
        elif isinstance(token, Identifier):
            column_node = ASTNode("COLUMN", token.value)
            parent_node.add_child(column_node)
        
        # 处理 WHERE 子句
        elif isinstance(token, Where):
            where_node = ASTNode("WHERE", "WHERE")
            parent_node.add_child(where_node)
            process_tokens(token.tokens, where_node)  # 递归处理 WHERE 子句
        
        # 处理子查询
        elif token.ttype is Keyword and token.value.upper() in ("FROM", "JOIN"):
            from_node = ASTNode(token.value.upper(), token.value)
            parent_node.add_child(from_node)
        
        # 处理子查询中的括号或嵌套查询
        elif token.is_group:
            subquery_node = ASTNode("SQL")
            parent_node.add_child(subquery_node)
            process_tokens(token.tokens, subquery_node)

        # 其他标识符（如表名）
        elif token.ttype is Keyword:
            keyword_node = ASTNode("KEYWORD", token.value)
            parent_node.add_child(keyword_node)
        # elif token.ttype is Token.Literal.String.Single:
        else:
            # print(token.ttype, token.value)
            value_node = ASTNode("VALUE", token.value)
            parent_node.add_child(value_node)


def is_subtree(main_tree, sub_tree):
    """
    判断sub_tree是否是main_tree的子树
    """
    # 如果sub_tree为空，则一定是子树
    if sub_tree is None:
        return True

    # 如果main_tree为空，则sub_tree不可能是子树
    if main_tree is None:
        return False

    # 检查当前节点是否匹配
    if is_same_tree(main_tree, sub_tree):
        return True

    # 递归检查子节点
    for child in main_tree.children:
        if is_subtree(child, sub_tree):
            return True

    return False


def is_same_tree(tree1, tree2):
    """
    判断两棵树是否部分匹配
    """
    # 如果两棵树都为空，则相同
    if tree1 is None and tree2 is None:
        return True

    # 如果有一个为空，另一个不为空，则不同
    if tree1 is None or tree2 is None:
        return False

    # 比较节点类型和值
    if tree1.node_type != tree2.node_type or tree1.value != tree2.value:
        return False

    # 确保 tree2 的子节点是 tree1 子节点的子集
    for sub_child in tree2.children:
        # 每个子节点必须匹配
        if not any(is_same_tree(main_child, sub_child) for main_child in tree1.children):
            return False

    return True


import re

def remove_aliases(sql):
    """
    主函数：移除 SQL 语句中的表别名。
    """
    def process_query(query, alias_to_table=None):
        """
        递归处理 SQL 查询，移除别名。
        """
        if alias_to_table is None:
            alias_to_table = {}

        # 正则匹配 FROM 和 JOIN 子句中的表名和别名
        table_alias_pattern = re.compile(r'\b(FROM|JOIN)\s+([`"]?\w+[`"]?)\s+(AS\s+)?(\w+)\b', re.IGNORECASE)
        local_alias_to_table = alias_to_table.copy()

        def replace_alias_table(match):
            """
            替换 FROM/JOIN 子句中的别名为完整表名，并记录映射关系。
            """
            table_name = match.group(2).strip('`"')  # 去掉反引号或双引号
            alias_name = match.group(4)
            local_alias_to_table[alias_name] = table_name
            return f"{match.group(1)} {table_name}"  # 替换为完整表名

        # 替换 FROM 和 JOIN 子句中的别名
        query = table_alias_pattern.sub(replace_alias_table, query)

        # 替换字段中的别名前缀
        field_pattern = re.compile(r'(\b\w+)\.(\w+|\`[^\`]+\`)', re.IGNORECASE)
        def replace_field_alias(match):
            """
            替换字段中的别名前缀为完整表名。
            """
            alias_name, field_name = match.groups()
            if alias_name in local_alias_to_table:
                return f"{local_alias_to_table[alias_name]}.{field_name}"
            return match.group(0)  # 保留原字段

        query = field_pattern.sub(replace_field_alias, query)

        # 递归处理子查询
        subquery_pattern = re.compile(r'\((SELECT .*?)\)', re.IGNORECASE | re.DOTALL)
        def process_subquery(match):
            """
            递归处理子查询，移除子查询中的别名。
            """
            inner_query = match.group(1)
            processed_inner_query = process_query(inner_query, local_alias_to_table)
            return f"({processed_inner_query})"

        query = subquery_pattern.sub(process_subquery, query)

        return query

    # 调用递归处理函数
    return process_query(sql)


if __name__ == "__main__":
    sql = """
    SELECT name, age 
    FROM users 
    WHERE age > 18 
      AND id IN (
        SELECT user_id 
        FROM orders 
        WHERE amount > 100
      );
    """

    ast = parse_sql_to_ast(sql)
    print(ast)

    subsql = """
    SELECT user_id 
    FROM orders 
    WHERE amount > 100
    """

    subast = parse_sql_to_ast(subsql)
    print(subast)

    print(is_subtree(ast, subast))


    sql = "SELECT T1.`School Name`, T2.Street, T2.City, T2.State, T2.Zip FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.County = 'Monterey' AND T1.`Free Meal Count (Ages 5-17)` > 800 AND T1.`School Type` = 'High Schools (Public)'"
    ast = parse_sql_to_ast(sql)
    print(ast)


    sql = "SELECT T2.School, T2.DOC FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.FundingType = 'Locally funded' AND (T1.`Enrollment (K-12)` - T1.`Enrollment (Ages 5-17)`) > (SELECT AVG(T3.`Enrollment (K-12)` - T3.`Enrollment (Ages 5-17)`) FROM frpm AS T3 INNER JOIN schools AS T4 ON T3.CDSCode = T4.CDSCode WHERE T4.FundingType = 'Locally funded')"
    sql = remove_aliases(sql)
    ast = parse_sql_to_ast(sql)
    print(ast)


    subsql = "SELECT AVG(T3.`Enrollment (K-12)` - T3.`Enrollment (Ages 5-17)`) FROM frpm AS T3 INNER JOIN schools AS T4 ON T3.CDSCode = T4.CDSCode"
    subsql = remove_aliases(subsql)
    subast = parse_sql_to_ast(subsql)
    print(subast)

    print(is_subtree(ast, subast))


    subsql = "SELECT AVG(T3.`Enrollment (K-12)` - T3.`Enrollment (Ages 5-17)`) FROM frpm AS T3 INNER JOIN schools AS T4 WHERE T4.FundingType = 'Locally funded'"
    subsql = remove_aliases(subsql)
    subast = parse_sql_to_ast(subsql)
    print(subast)

    print(is_subtree(ast, subast))


    subsql = "SELECT AVG(T3.`Enrollment (K-12)`) FROM frpm AS T3 INNER JOIN schools AS T4 WHERE T4.FundingType = 'Locally funded'"
    subsql = remove_aliases(subsql)
    subast = parse_sql_to_ast(subsql)
    print(subast)

    print(is_subtree(ast, subast))


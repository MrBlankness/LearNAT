import math
import random
from copy import deepcopy
from AST import ASTNode, parse_sql_to_ast, is_subtree, remove_aliases
import json
from zhipuai import ZhipuAI
from func_timeout import func_set_timeout


# MCTS 节点定义
class MCTSNode:
    def __init__(self, query=None, sql=None, parent=None):
        self.query = query
        self.sql = sql
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.is_terminal = False  # 是否为剪枝节点

    def add_child(self, child_node):
        self.children.append(child_node)



def extract_last_json(s):
    if '}' not in s:
        return None
    
    # 找到最后一个 }
    last_brace_index = s.rindex('}')
    
    # 追踪括号匹配
    balance = 1  # 从最后一个}开始，我们需要找到匹配的{
    for i in range(last_brace_index - 1, -1, -1):
        if s[i] == '}':
            balance += 1
        elif s[i] == '{':
            balance -= 1
            
        # 当balance变为0时，找到匹配的{
        if balance == 0:
            json_str = s[i:last_brace_index + 1]
            try:
                # 尝试解析JSON
                return json.loads(json_str)
            except json.JSONDecodeError:
                return None
    
    return None

@func_set_timeout(60)
def glm_predict(messages):
    client = ZhipuAI(api_key="XXXXXXXXXX")  # 请填写您自己的APIKey
    response = client.chat.completions.create(
        model="glm-4-plus",  # 请填写您要调用的模型名称
        messages=messages,
    )
    response_context = response.choices[0].message.content
    response_json = extract_last_json(response_context)
    return response_json['sub_query'], response_json['sub_sql']


def tree_edit_distance(t1, t2):
    """简化版的树编辑距离，真实情况应使用专业库"""
    if t1.node_type != t2.node_type or t1.value != t2.value:
        return 1
    cost = abs(len(t1.children) - len(t2.children))
    common = min(len(t1.children), len(t2.children))
    for i in range(common):
        cost += tree_edit_distance(t1.children[i], t2.children[i])
    return cost

def tree_similarity(ast1, ast2, alpha=0.7, beta=0.3):
    """计算两个AST之间的相似性"""
    def count_nodes(root):
        count = 1
        for child in root.children:
            count += count_nodes(child)
        return count

    node_sim = count_nodes(ast1) / count_nodes(ast2)
    edit_dist = tree_edit_distance(ast1, ast2)
    max_nodes = max(count_nodes(ast1), count_nodes(ast2))
    structure_sim = 1 - edit_dist / max_nodes
    return alpha * node_sim + beta * structure_sim

# MCTS 核心实现
class NL2SQLMCTS:
    def __init__(self, full_query, full_sql, max_depth=5, simulations=50):
        self.full_query = full_query
        self.full_sql = remove_aliases(full_sql)
        self.full_ast = parse_sql_to_ast(self.full_sql)
        self.max_depth = max_depth
        self.simulations = simulations

    def run(self):
        root = MCTSNode()
        for _ in range(self.simulations):
            self.simulate(root, depth=0)
        return self.extract_best_path(root)

    def simulate(self, node, depth):
        if depth >= self.max_depth or node.is_terminal:
            return node.value

        if not node.children:
            # 扩展节点
            prompt = self.full_query if not node.query else node.query
            sub_query, sub_sql = glm_predict(prompt)
            sub_ast = parse_sql_to_ast(remove_aliases(sub_sql))

            if is_subtree(self.full_ast, sub_ast):
                child = MCTSNode(sub_query, sub_sql, parent=node)
                child.value = tree_similarity(sub_ast, self.full_ast)
                child.query = prompt + json.dumps({"sub_query":sub_query, "sub_sql":sub_sql}, indent=4)
                node.add_child(child)
            else:
                child = MCTSNode(sub_query, sub_sql, parent=node)
                child.value = node.value  # 剪枝
                child.is_terminal = True
                node.add_child(child)

        # 选择最佳子节点
        best_child = self.select(node)
        value = self.simulate(best_child, depth + 1)
        best_child.visits += 1
        best_child.value = (best_child.value * (best_child.visits - 1) + value) / best_child.visits
        return best_child.value

    def select(self, node):
        """根据UCB公式选择子节点"""
        log_N = math.log(node.visits + 1)
        best_score = -float('inf')
        best_child = None
        for child in node.children:
            if child.visits == 0:
                return child
            exploit = child.value
            explore = math.sqrt(log_N / child.visits)
            ucb_score = exploit + 1.41 * explore
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child
        return best_child

    def extract_best_path(self, root):
        """提取根节点到叶子的路径"""
        path = []
        node = root
        while node.children:
            node = max(node.children, key=lambda c: c.value)
            if node.query and node.sql:
                path.append((node.query, node.sql))
        return path

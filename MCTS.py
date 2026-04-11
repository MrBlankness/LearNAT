# AST-guided Monte Carlo Tree Search for NL2SQL decomposition synthesis.

import math
import random
import json
import time
from copy import deepcopy
from typing import Optional, List, Tuple, Dict, Any, Callable
from dataclasses import dataclass

from AST import (
    ASTNode, parse_sql_to_ast, is_subtree, remove_aliases,
    calculate_ast_similarity, merge_asts, ASTClassifier,
    normalize_sub_sql
)


@dataclass
class MCTSConfig:
    """Configuration for MCTS search."""
    max_depth: int = 5
    simulations: int = 50
    exploration_constant: float = 1.41
    alpha: float = 0.75  # Weight for node-level similarity
    temperature: float = 1.0
    timeout: int = 60
    max_children_per_node: int = 5  # Max children per node before fully expanded
    max_llm_calls: int = 200  # Max total LLM calls as a hard search limit
    llm_call_delay: float = 1.0  # Delay in seconds between LLM calls to prevent rate limiting
    llm_max_retries: int = 5  # Max retries when LLM output JSON parsing fails


class MCTSNode:
    """A single node in the MCTS search tree, holding a subtask query and its SQL."""
    
    def __init__(self, 
                 query: Optional[str] = None, 
                 sql: Optional[str] = None,
                 parent: Optional['MCTSNode'] = None):
        self.query = query
        self.sql = sql
        self.ast: Optional[ASTNode] = None
        self.parent = parent
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.value = 0.0
        self.is_terminal = False
        self.node_type = 'unknown'  # progressive, redundant, invalid
        
        # Parse AST if SQL is provided
        if sql:
            try:
                normalized_sql = remove_aliases(sql)
                self.ast = parse_sql_to_ast(normalized_sql)
            except Exception as e:
                print(f"Error parsing SQL: {e}")
                self.ast = None

    def add_child(self, child: 'MCTSNode') -> None:
        child.parent = self
        self.children.append(child)
    
    def is_fully_expanded(self, max_children: int = 5) -> bool:
        return len(self.children) >= max_children
    
    def get_path_to_root(self) -> List['MCTSNode']:
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))
    
    def get_merged_ast(self) -> Optional[ASTNode]:
        path = self.get_path_to_root()
        ast_list = [node.ast for node in path if node.ast is not None]
        if not ast_list:
            return None
        return merge_asts(ast_list)
    
    def get_decomposition_sequence(self) -> List[Tuple[str, str]]:
        path = self.get_path_to_root()
        sequence = []
        for node in path:
            if node.query and node.sql:
                sequence.append((node.query, node.sql))
        return sequence


class NL2SQLMCTS:
    """MCTS with AST-guided pruning for generating NL2SQL decompositions."""
    
    def __init__(self, 
                 full_query: str,
                 full_sql: str,
                 llm_predictor: Callable,
                 config: Optional[MCTSConfig] = None):
        self.full_query = full_query
        self.full_sql = remove_aliases(full_sql)
        self.full_ast = parse_sql_to_ast(self.full_sql)
        self.llm_predictor = llm_predictor
        self.config = config or MCTSConfig()
        
        # Data collection for training
        self.successful_trajectories: List[List[Tuple[str, str]]] = []
        self.contrastive_pairs: List[Dict] = []
        
        # Statistics
        self.total_expansions = 0
        self.pruned_nodes = 0
        self.llm_calls = 0
        self._found_valid = False  # Early stop flag
    
    def run(self) -> List[Tuple[str, str]]:
        """Run MCTS and return the best decomposition as a list of (query, sql) pairs."""
        root = MCTSNode()
        self._root = root  # Save root for post-search DPO collection
        
        for simulation in range(self.config.simulations):
            # Check LLM call limit
            if self.llm_calls >= self.config.max_llm_calls:
                print(f"[MCTS] LLM call limit reached ({self.config.max_llm_calls}). Stopping search.")
                break
            
            self._simulate(root, depth=0)
            
            # Early stop: check if we found a valid decomposition
            if self._found_valid:
                print(f"[MCTS] Valid decomposition found at simulation {simulation + 1}. Early stopping.")
                break
        
        # Extract best path
        best_path = self._extract_best_path(root)
        
        # Record successful trajectory if valid
        if self._is_valid_decomposition(best_path):
            self.successful_trajectories.append(best_path)
        
        return best_path
    
    def _simulate(self, node: MCTSNode, depth: int) -> float:
        """Single MCTS simulation: select -> expand -> evaluate -> backprop."""
        # Terminal conditions
        if depth >= self.config.max_depth or node.is_terminal:
            # Check if current path forms a valid decomposition
            if not self._found_valid and depth > 0:
                path = node.get_decomposition_sequence()
                if self._is_valid_decomposition(path):
                    self._found_valid = True
                    self.successful_trajectories.append(path)
            return node.value
        
        # Check LLM call limit
        if self.llm_calls >= self.config.max_llm_calls:
            return node.value
        
        # If node has children but is NOT fully expanded, decide whether to
        # explore an existing child (selection) or expand a new one.
        if node.children and not node.is_fully_expanded(self.config.max_children_per_node):
            # Use UCB1-like logic: expand new child with some probability,
            # or select existing child. We expand if there are fewer children
            # than visits warrant, following standard MCTS.
            # Simple heuristic: expand if any existing progressive child has
            # been visited, otherwise select.
            all_terminal = all(c.is_terminal for c in node.children)
            if all_terminal:
                # All existing children are terminal (pruned), try expanding a new one
                return self._expand_and_evaluate(node, depth)
            else:
                # Select among existing children
                best_child = self._select_child(node)
                if best_child.is_terminal:
                    # Selected child is terminal, try expanding a new one instead
                    return self._expand_and_evaluate(node, depth)
                value = self._simulate(best_child, depth + 1)
                self._backpropagate(best_child, value)
                return value
        
        # If node is fully expanded, just select among existing children
        if node.children and node.is_fully_expanded(self.config.max_children_per_node):
            # Filter to non-terminal children for selection
            non_terminal = [c for c in node.children if not c.is_terminal]
            if not non_terminal:
                # All children are terminal
                node.is_terminal = True
                return node.value
            best_child = self._select_child(node)
            if best_child.is_terminal:
                # Fallback: pick best non-terminal
                best_child = max(non_terminal, key=lambda c: c.value)
            value = self._simulate(best_child, depth + 1)
            self._backpropagate(best_child, value)
            return value
        
        # No children yet: expand
        return self._expand_and_evaluate(node, depth)
    
    def _expand_and_evaluate(self, node: MCTSNode, depth: int) -> float:
        """Create a new child via LLM, classify it, and return its value."""
        try:
            sub_query, sub_sql = self._expand_node(node)
            self.llm_calls += 1
            
            if sub_query is None or sub_sql is None:
                # LLM failed to generate; if no children exist, mark terminal
                if not node.children:
                    node.is_terminal = True
                return node.value
            
            # Preprocess LLM output: normalize sub-SQL to improve AST matching
            sub_sql_normalized = remove_aliases(sub_sql)
            sub_sql_normalized = normalize_sub_sql(sub_sql_normalized, self.full_sql)
            # Update sub_sql with the normalized version for downstream use
            sub_sql = sub_sql_normalized
            
            # Check for duplicate children (same SQL)
            normalized_new = sub_sql.strip().lower()
            for existing_child in node.children:
                if existing_child.sql:
                    normalized_existing = existing_child.sql.strip().lower()
                    if normalized_new == normalized_existing:
                        # Duplicate, don't add
                        return existing_child.value
            
            # Create child node
            child = MCTSNode(query=sub_query, sql=sub_sql, parent=node)
            self.total_expansions += 1
            
            # Classify and potentially prune
            child.node_type = self._classify_node(child)
            
            if child.node_type == 'invalid':
                child.is_terminal = True
                child.value = 0.0
                self.pruned_nodes += 1
                node.add_child(child)
                
                # Collect contrastive pair
                self._collect_contrastive_pair(node, child, 'invalid')
                return 0.0
            
            elif child.node_type == 'redundant':
                child.is_terminal = True
                child.value = node.value
                self.pruned_nodes += 1
                node.add_child(child)
                
                # Collect contrastive pair
                self._collect_contrastive_pair(node, child, 'redundant')
                return node.value
            
            else:  # progressive
                # Calculate reward using AST similarity
                child.value = self._estimate_reward(child)
                node.add_child(child)
                
                # Check if this path is already a valid decomposition
                path = child.get_decomposition_sequence()
                if self._is_valid_decomposition(path):
                    self._found_valid = True
                    self.successful_trajectories.append(path)
                    return child.value
                
                # Continue simulation deeper
                value = self._simulate(child, depth + 1)
                return value
                
        except Exception as e:
            print(f"Error in expansion: {e}")
            import traceback
            traceback.print_exc()
            if not node.children:
                node.is_terminal = True
            return node.value
    
    def _select_child(self, node: MCTSNode) -> MCTSNode:
        """Pick the child with the highest UCB1 score."""
        log_N = math.log(node.visits + 1)
        best_score = float('-inf')
        best_child = None
        
        for child in node.children:
            if child.visits == 0:
                return child  # Prioritize unvisited nodes
            
            # UCB1 formula
            exploit = child.value
            explore = math.sqrt(log_N / child.visits)
            ucb_score = exploit + self.config.exploration_constant * explore
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child
        
        return best_child or node.children[0]
    
    def _expand_node(self, node: MCTSNode) -> Tuple[Optional[str], Optional[str]]:
        """Ask the LLM for the next subtask; retries on parse failures."""
        # Build context from path
        path = node.get_decomposition_sequence()
        
        max_retries = self.config.llm_max_retries
        
        for attempt in range(1, max_retries + 1):
            # Add delay between LLM calls to prevent rate limiting
            if self.config.llm_call_delay > 0:
                time.sleep(self.config.llm_call_delay)
            
            try:
                sub_query, sub_sql = self.llm_predictor(
                    full_query=self.full_query,
                    full_sql=self.full_sql,
                    previous_steps=path
                )
                
                # If LLM returned valid output, return it
                if sub_query is not None and sub_sql is not None:
                    return sub_query, sub_sql
                
                # LLM returned None (JSON parsing failed), retry
                if attempt < max_retries:
                    print(f"[MCTS] LLM output parsing failed (attempt {attempt}/{max_retries}), retrying...")
                    # Count each retry as an LLM call
                    self.llm_calls += 1
                    # Check LLM call limit before retrying
                    if self.llm_calls >= self.config.max_llm_calls:
                        print(f"[MCTS] LLM call limit reached during retry. Stopping.")
                        return None, None
                else:
                    print(f"[MCTS] LLM output parsing failed after {max_retries} attempts. Giving up.")
                    
            except Exception as e:
                print(f"LLM prediction error (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    # Count each retry as an LLM call
                    self.llm_calls += 1
                    if self.llm_calls >= self.config.max_llm_calls:
                        print(f"[MCTS] LLM call limit reached during retry. Stopping.")
                        return None, None
                else:
                    print(f"[MCTS] LLM prediction failed after {max_retries} attempts.")
        
        return None, None
    
    def _classify_node(self, node: MCTSNode) -> str:
        """Return 'progressive', 'redundant', or 'invalid'."""
        if node.ast is None:
            return 'invalid'
        
        classifier = ASTClassifier()
        parent_merged = node.parent.get_merged_ast() if node.parent else None
        
        return classifier.classify_node(
            sub_ast=node.ast,
            target_ast=self.full_ast,
            parent_merged_ast=parent_merged
        )
    
    def _estimate_reward(self, node: MCTSNode) -> float:
        """AST similarity between the merged path AST and the target AST."""
        merged_ast = node.get_merged_ast()
        if merged_ast is None:
            return 0.0
        
        return calculate_ast_similarity(
            merged_ast, 
            self.full_ast, 
            alpha=self.config.alpha
        )
    
    def _backpropagate(self, node: MCTSNode, value: float) -> None:
        """Propagate value up to the root with running-average updates."""
        current = node
        while current is not None:
            current.visits += 1
            # Running average update
            current.value = (current.value * (current.visits - 1) + value) / current.visits
            current = current.parent
    
    def _extract_best_path(self, root: MCTSNode) -> List[Tuple[str, str]]:
        """Greedily follow highest-value children from root to leaf."""
        path = []
        node = root
        
        while node.children:
            # Select child with highest value
            node = max(node.children, key=lambda c: c.value)
            if node.query and node.sql and node.node_type == 'progressive':
                path.append((node.query, node.sql))
        
        return path
    
    def _is_valid_decomposition(self, path: List[Tuple[str, str]]) -> bool:
        """True if the merged AST similarity to the target exceeds 0.9."""
        if not path:
            return False
        
        # Check 1: Does the last step alone match the target?
        try:
            last_sql = path[-1][1]
            last_ast = parse_sql_to_ast(remove_aliases(last_sql))
            last_sim = calculate_ast_similarity(last_ast, self.full_ast)
            if last_sim > 0.9:
                return True
        except:
            pass
        
        # Check 2: Does the merged AST of all steps match the target?
        ast_list = []
        for _, sql in path:
            try:
                ast = parse_sql_to_ast(remove_aliases(sql))
                ast_list.append(ast)
            except:
                return False
        
        if ast_list:
            merged = merge_asts(ast_list)
            similarity = calculate_ast_similarity(merged, self.full_ast)
            if similarity > 0.9:
                return True
        
        # Check 3: Progressive accumulation - check if adding steps improves similarity
        if len(ast_list) >= 2:
            for i in range(1, len(ast_list) + 1):
                partial_merged = merge_asts(ast_list[:i])
                partial_sim = calculate_ast_similarity(partial_merged, self.full_ast)
                if partial_sim > 0.9:
                    return True
        
        return False
    
    def _collect_contrastive_pair(self, parent: MCTSNode, 
                                   incorrect: MCTSNode, 
                                   error_type: str) -> None:
        """Legacy stub — actual DPO collection is in _collect_all_contrastive_pairs."""
        pass
    
    def _collect_all_contrastive_pairs(self, root: MCTSNode) -> None:
        """
        BFS the search tree and enumerate all (progressive, non-progressive)
        sibling pairs at each branching point for DPO training.
        """
        # BFS traversal of the tree
        queue = [root]
        while queue:
            node = queue.pop(0)
            
            # Only process nodes with at least 2 children (branching points)
            if len(node.children) >= 2:
                # Separate children by type
                progressive_children = [
                    c for c in node.children 
                    if c.node_type == 'progressive' and c.query and c.sql
                ]
                negative_children = [
                    c for c in node.children 
                    if c.node_type in ('invalid', 'redundant') and c.query and c.sql
                ]
                
                # Generate all (progressive, negative) pairs
                context = node.get_decomposition_sequence()
                for pos_child in progressive_children:
                    for neg_child in negative_children:
                        margin = pos_child.value - neg_child.value
                        # Only include pairs where progressive is strictly better
                        if margin > 0:
                            self.contrastive_pairs.append({
                                'context': context,
                                'correct': (pos_child.query, pos_child.sql),
                                'incorrect': (neg_child.query, neg_child.sql),
                                'error_type': neg_child.node_type,
                                'margin': margin
                            })
                
                # Also generate pairs between progressive children with
                # significantly different values (higher-value vs lower-value)
                if len(progressive_children) >= 2:
                    # Sort by value descending
                    sorted_prog = sorted(progressive_children, 
                                        key=lambda c: c.value, reverse=True)
                    for i in range(len(sorted_prog)):
                        for j in range(i + 1, len(sorted_prog)):
                            margin = sorted_prog[i].value - sorted_prog[j].value
                            # Only create pair if there's a meaningful difference
                            if margin > 0.1:
                                self.contrastive_pairs.append({
                                    'context': context,
                                    'correct': (sorted_prog[i].query, sorted_prog[i].sql),
                                    'incorrect': (sorted_prog[j].query, sorted_prog[j].sql),
                                    'error_type': 'lower_quality',
                                    'margin': margin
                                })
            
            # Add children to queue for BFS
            queue.extend(node.children)
    
    def get_training_data(self, root: Optional[MCTSNode] = None) -> Dict:
        """Return collected SFT trajectories and DPO contrastive pairs."""
        # Collect contrastive pairs from the tree if root is available
        if root is not None:
            self._collect_all_contrastive_pairs(root)
        
        return {
            'successful_trajectories': self.successful_trajectories,
            'contrastive_pairs': self.contrastive_pairs,
            'statistics': {
                'total_expansions': self.total_expansions,
                'pruned_nodes': self.pruned_nodes,
                'llm_calls': self.llm_calls,
                'success_rate': len(self.successful_trajectories) / max(1, self.config.simulations)
            }
        }


class DemonstrationPool:
    """Pool of successful decompositions for few-shot demonstration selection."""
    
    def __init__(self, max_size: int = 100):
        self.demonstrations: List[Dict] = []
        self.max_size = max_size
    
    def add_demonstration(self, query: str, sql: str, 
                          decomposition: List[Tuple[str, str]],
                          schema: Optional[Dict] = None) -> None:
        """Store a successful decomposition."""
        demo = {
            'query': query,
            'question': query,  # Alias for prompt.py compatibility
            'sql': sql,
            'decomposition': decomposition,
            'schema': schema or {},
            'ast': parse_sql_to_ast(remove_aliases(sql))
        }
        
        self.demonstrations.append(demo)
        
        # Remove oldest if over capacity
        if len(self.demonstrations) > self.max_size:
            self.demonstrations.pop(0)
    
    def get_similar_demonstrations(self, query: str, sql: str, 
                                    top_k: int = 3) -> List[Dict]:
        """Retrieve the top-k demonstrations most similar (by AST) to the given query."""
        if not self.demonstrations:
            return []
        
        target_ast = parse_sql_to_ast(remove_aliases(sql))
        
        # Calculate similarities
        similarities = []
        for demo in self.demonstrations:
            sim = calculate_ast_similarity(demo['ast'], target_ast)
            similarities.append((sim, demo))
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [demo for _, demo in similarities[:top_k]]


def extract_json_from_response(response: str) -> Optional[Dict]:
    """Try to extract a JSON object from an LLM response string."""
    if '}' not in response:
        return None
    
    # Find the last closing brace
    last_brace = response.rindex('}')
    
    # Track bracket balance to find matching opening brace
    balance = 1
    for i in range(last_brace - 1, -1, -1):
        if response[i] == '}':
            balance += 1
        elif response[i] == '{':
            balance -= 1
        
        if balance == 0:
            json_str = response[i:last_brace + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                return None
    
    return None


if __name__ == "__main__":
    # Demo/test code
    print("=" * 60)
    print("Testing MCTS Module")
    print("=" * 60)
    
    # Mock LLM predictor for testing
    def mock_llm_predictor(full_query: str, full_sql: str, 
                           previous_steps: List[Tuple[str, str]]) -> Tuple[str, str]:
        """Mock LLM that returns simple decompositions."""
        step_num = len(previous_steps)
        
        if step_num == 0:
            return ("Find all users", "SELECT * FROM users")
        elif step_num == 1:
            return ("Filter by age", "SELECT * FROM users WHERE age > 18")
        else:
            return ("Get names", "SELECT name FROM users WHERE age > 18")
    
    # Test MCTS
    query = "Find the names of all users who are over 18 years old"
    sql = "SELECT name FROM users WHERE age > 18"
    
    config = MCTSConfig(max_depth=3, simulations=10)
    mcts = NL2SQLMCTS(query, sql, mock_llm_predictor, config)
    
    print("\nRunning MCTS search...")
    result = mcts.run()
    
    print("\nDecomposition Result:")
    for i, (q, s) in enumerate(result):
        print(f"  Step {i+1}:")
        print(f"    Query: {q}")
        print(f"    SQL: {s}")
    
    print("\nTraining Data Statistics:")
    training_data = mcts.get_training_data()
    print(f"  Successful trajectories: {len(training_data['successful_trajectories'])}")
    print(f"  Contrastive pairs: {len(training_data['contrastive_pairs'])}")
    print(f"  Total expansions: {training_data['statistics']['total_expansions']}")
    print(f"  Pruned nodes: {training_data['statistics']['pruned_nodes']}")
    
    # Test Demonstration Pool
    print("\n" + "=" * 60)
    print("Testing Demonstration Pool")
    print("=" * 60)
    
    pool = DemonstrationPool()
    pool.add_demonstration(
        "Find users over 18",
        "SELECT name FROM users WHERE age > 18",
        [("Get user names", "SELECT name FROM users"),
         ("Filter by age", "SELECT name FROM users WHERE age > 18")]
    )
    pool.add_demonstration(
        "Find orders over $100",
        "SELECT * FROM orders WHERE amount > 100",
        [("Get all orders", "SELECT * FROM orders"),
         ("Filter by amount", "SELECT * FROM orders WHERE amount > 100")]
    )
    
    similar = pool.get_similar_demonstrations(
        "Find products over $50",
        "SELECT * FROM products WHERE price > 50",
        top_k=2
    )
    
    print(f"Found {len(similar)} similar demonstrations")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)

# Decomposition data synthesis via MCTS with AST-guided search.
# Loads BIRD/Spider datasets, runs MCTS to collect SFT trajectories
# and DPO contrastive pairs.

import os
import sys
import json
import sqlite3
import re
import random
import hashlib
import time
import requests
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
import numpy as np
from transformers import set_seed

from AST import parse_sql_to_ast, remove_aliases, is_subtree
from MCTS import NL2SQLMCTS, MCTSConfig, DemonstrationPool
from prompt import (
    build_decomposition_prompt,
    format_schema,
    GEN_SCHEMA_LINKING_GP_PROMPT
)
from utils import set_seed as set_all_seeds


@dataclass
class DataConfig:
    """Configuration for data generation."""
    data_dir: str = "./data/BIRD/train"
    save_dir: str = "./outputs/decomposition_data"
    max_workers: int = 8
    mcts_simulations: int = 50
    mcts_max_depth: int = 5
    seed: int = 2024
    api_key: str = ""  # Set your API key here
    model_name: str = "gpt-4o"
    timeout: int = 120


class DatabaseManager:
    """Manages database connections and queries."""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self._load_table_info()
    
    def _load_table_info(self) -> None:
        """Load table information from dataset."""
        self.table_info = {}
        
        table_path = os.path.join(self.data_dir, 'train_tables.json')
        if os.path.exists(table_path):
            with open(table_path, 'r', encoding='utf-8') as f:
                tables = json.load(f)
            
            for item in tables:
                db_id = item['db_id'].lower()
                self.table_info[db_id] = {
                    'tables_info': {},
                    'foreign_keys': []
                }
                
                # Process tables and columns
                for table_name in item['table_names_original']:
                    self.table_info[db_id]['tables_info'][table_name.lower()] = []
                
                for col_info in item['column_names_original'][1:]:
                    table_idx, col_name = col_info
                    table_name = item['table_names_original'][table_idx].lower()
                    self.table_info[db_id]['tables_info'][table_name].append(f'`{col_name.lower()}`')
                
                # Process foreign keys
                for fk in item['foreign_keys']:
                    fk_col1 = item['column_names_original'][fk[0]]
                    fk_col2 = item['column_names_original'][fk[1]]
                    
                    table1 = item['table_names_original'][fk_col1[0]].lower()
                    col1 = fk_col1[1].lower()
                    table2 = item['table_names_original'][fk_col2[0]].lower()
                    col2 = fk_col2[1].lower()
                    
                    self.table_info[db_id]['foreign_keys'].append([
                        f"{table1}.`{col1}`",
                        f"{table2}.`{col2}`"
                    ])
    
    def execute_sql(self, sql: str, db_id: str, timeout: int = 30) -> Dict:
        """Run *sql* on the SQLite DB for *db_id*; return at most 10 rows."""
        db_path = os.path.join(
            self.data_dir, 'train_databases', db_id, f'{db_id}.sqlite'
        )
        
        try:
            conn = sqlite3.connect(db_path, timeout=timeout)
            conn.text_factory = str
            cursor = conn.cursor()
            
            cursor.execute(sql)
            result = cursor.fetchall()
            
            conn.close()
            
            return {
                "sql": sql,
                "data": result[:10],  # Limit results
                "success": True,
                "error": None
            }
            
        except sqlite3.Error as e:
            return {
                "sql": sql,
                "data": None,
                "success": False,
                "error": str(e)
            }
        except Exception as e:
            return {
                "sql": sql,
                "data": None,
                "success": False,
                "error": str(e)
            }
    
    def execute_sql_full(self, sql: str, db_id: str, timeout: int = 30) -> Dict:
        """Like execute_sql but returns all rows (for verification)."""
        db_path = os.path.join(
            self.data_dir, 'train_databases', db_id, f'{db_id}.sqlite'
        )
        
        try:
            conn = sqlite3.connect(db_path, timeout=timeout)
            conn.text_factory = str
            cursor = conn.cursor()
            
            cursor.execute(sql)
            result = cursor.fetchall()
            
            conn.close()
            
            return {
                "sql": sql,
                "data": result,
                "success": True,
                "error": None
            }
            
        except sqlite3.Error as e:
            return {
                "sql": sql,
                "data": None,
                "success": False,
                "error": str(e)
            }
        except Exception as e:
            return {
                "sql": sql,
                "data": None,
                "success": False,
                "error": str(e)
            }
    
    def get_table_info(self, db_id: str) -> Dict:
        return self.table_info.get(db_id.lower(), {'tables_info': {}, 'foreign_keys': []})
    
    def get_column_samples(self, db_id: str, table_name: str, 
                          column_name: str, limit: int = 5) -> List:
        sql = f"SELECT `{column_name}` FROM {table_name} LIMIT {limit}"
        result = self.execute_sql(sql, db_id)
        
        if result['success'] and result['data']:
            return [str(row[0]) for row in result['data']]
        return []


class LLMClient:
    """ZhipuAI chat client (legacy, kept for compatibility)."""
    
    def __init__(self, api_key: str, model_name: str = "glm-4-plus"):
        self.api_key = api_key
        self.model_name = model_name
        self._client = None
    
    def _init_client(self):
        """Initialize API client lazily."""
        if self._client is None:
            try:
                from zhipuai import ZhipuAI
                self._client = ZhipuAI(api_key=self.api_key)
            except ImportError:
                print("Warning: zhipuai not installed. Using mock client.")
                self._client = "mock"
    
    def predict(self, messages: List[Dict], timeout: int = 60) -> str:
        """Send a chat request and return the reply."""
        self._init_client()
        
        if self._client == "mock":
            # Return mock response for testing
            return self._mock_response(messages)
        
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM API error: {e}")
            return ""
    
    def _mock_response(self, messages: List[Dict]) -> str:
        return json.dumps({
            "sub_query": "Mock subtask description",
            "sub_sql": "SELECT * FROM table",
            "reasoning": "Mock reasoning"
        })


class GPT4oLLMClient:
    """OpenAI-compatible chat completion client (works with GPT-4o, etc.)."""

    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.6):
        self.model_name = model_name
        self.temperature = temperature
        # Read from environment variable; fall back to empty string
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.api_base = os.environ.get(
            "OPENAI_API_BASE", "https://api.openai.com/v1"
        )

    def predict(self, messages: List[Dict], timeout: int = 60) -> str:
        """Send a chat-completion request and return the assistant reply."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return self._call_api(messages, timeout)
            except Exception as e:
                print(f"API error (attempt {attempt + 1}/{max_retries}): {e}")
                response = getattr(e, "response", None)
                if response is not None and getattr(response, "status_code", 0) == 429:
                    try:
                        error_data = response.json()
                        wait_seconds = error_data.get("error", {}).get("retry_after", 10)
                        print(f"Rate limit hit. Waiting {wait_seconds}s ...")
                        time.sleep(wait_seconds)
                    except Exception:
                        time.sleep(10)
                else:
                    time.sleep(2)
        return ""

    def _call_api(self, messages: List[Dict], timeout: int = 60) -> str:
        """Call an OpenAI-compatible chat completions endpoint."""
        url = f"{self.api_base}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        response = requests.post(
            url, headers=headers, data=json.dumps(payload), timeout=timeout
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()


def extract_json_from_string(s: str) -> Optional[Dict]:
    """Try to extract a JSON object from a string."""
    if '{' not in s or '}' not in s:
        return None
    
    # Find the last closing brace
    last_brace = s.rindex('}')
    
    # Track bracket balance
    balance = 1
    for i in range(last_brace - 1, -1, -1):
        if s[i] == '}':
            balance += 1
        elif s[i] == '{':
            balance -= 1
        
        if balance == 0:
            json_str = s[i:last_brace + 1]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                return None
    
    return None


class DecompositionGenerator:
    """Run MCTS over a dataset to produce SFT trajectories and DPO pairs."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.db_manager = DatabaseManager(config.data_dir)
        self.llm_client = GPT4oLLMClient(model_name=config.model_name)
        self.demo_pool = DemonstrationPool(max_size=100)
        
        # Statistics
        self.success_count = 0
        self.failure_count = 0
        self.skip_count = 0  # Cases skipped due to invalid GT SQL
        
        # Set seeds
        set_all_seeds(config.seed)
    
    def load_dataset(self) -> List[Dict]:
        data_path = os.path.join(self.config.data_dir, 'train.json')
        
        if not os.path.exists(data_path):
            print(f"Warning: Dataset not found at {data_path}")
            return []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} examples from {data_path}")
        return data
    
    def process_single_example(self, idx: int, item: Dict) -> Optional[Dict]:
        """Generate decomposition data for one example via MCTS."""
        question = item['question']
        sql = item['SQL']
        db_id = item['db_id']
        evidence = item.get('evidence', '')
        
        # Get database info
        db_info = self.db_manager.get_table_info(db_id)
        
        # Verify gold SQL is executable before attempting decomposition
        # BIRD-train dataset contains some erroneous GT SQLs that must be filtered out
        gt_exec_result = self.db_manager.execute_sql(sql, db_id)
        if not gt_exec_result['success']:
            print(f"Example {idx}: Skipping — GT SQL not executable on db '{db_id}': {gt_exec_result['error']}")
            print(f"  SQL: {sql[:200]}{'...' if len(sql) > 200 else ''}")
            self.skip_count += 1
            return None
        
        # Store GT execution result for SFT verification
        # (only first few rows are needed for the relaxed comparison)
        gt_result_data = gt_exec_result['data']
        
        try:
            # Create LLM predictor function
            def llm_predictor(full_query: str, full_sql: str, 
                             previous_steps: List[Tuple[str, str]]) -> Tuple[str, str]:
                # Get similar demonstrations
                demos = self.demo_pool.get_similar_demonstrations(full_query, full_sql, top_k=3)
                
                # Build prompt
                prompt = build_decomposition_prompt(
                    question=full_query,
                    schema=db_info['tables_info'],
                    foreign_keys=db_info['foreign_keys'],
                    evidence=evidence,
                    target_sql=full_sql,
                    previous_steps=previous_steps,
                    demonstrations=demos if demos else None
                )
                
                messages = [
                    {"role": "system", "content": "You are a data science expert."},
                    {"role": "user", "content": prompt}
                ]
                
                response = self.llm_client.predict(messages, timeout=self.config.timeout)
                
                # Parse response
                result = extract_json_from_string(response)
                if result and 'sub_query' in result and 'sub_sql' in result:
                    return result['sub_query'], result['sub_sql']
                
                return None, None
            
            # Run MCTS
            mcts_config = MCTSConfig(
                max_depth=self.config.mcts_max_depth,
                simulations=self.config.mcts_simulations
            )
            mcts = NL2SQLMCTS(question, sql, llm_predictor, mcts_config)
            decomposition = mcts.run()
            
            if not decomposition:
                self.failure_count += 1
                return None
            
            # Get training data (pass root for exhaustive DPO pair collection)
            training_data = mcts.get_training_data(root=mcts._root)
            
            # Filter SFT trajectories: only keep those whose final SQL
            # produces execution results consistent with the GT SQL
            # (relaxed check: top-3 rows, column-order insensitive)
            verified_trajectories = []
            for traj in training_data['successful_trajectories']:
                if self._verify_trajectory_execution(traj, gt_result_data, db_id):
                    verified_trajectories.append(traj)
                else:
                    print(f"Example {idx}: Filtered out SFT trajectory — "
                          f"final SQL execution result differs from GT")
            training_data['successful_trajectories'] = verified_trajectories
            
            # Add to demonstration pool for self-improvement
            if training_data['successful_trajectories']:
                self.demo_pool.add_demonstration(question, sql, decomposition,
                                                 schema=db_info.get('tables_info'))
                self.success_count += 1
            else:
                self.failure_count += 1
            
            return {
                'index': idx,
                'question': question,
                'sql': sql,
                'db_id': db_id,
                'evidence': evidence,
                'schema': db_info['tables_info'],
                'foreign_keys': db_info['foreign_keys'],
                'decomposition': decomposition,
                'sft_data': training_data['successful_trajectories'],
                'dpo_data': training_data['contrastive_pairs'],
                'statistics': training_data['statistics']
            }
            
        except Exception as e:
            print(f"Example {idx}: Error - {e}")
            self.failure_count += 1
            return None
    
    def _verify_trajectory_execution(self, trajectory: List[Tuple[str, str]],
                                      gt_result_data: List,
                                      db_id: str,
                                      top_k: int = 3) -> bool:
        """
        Check that the final SQL in *trajectory* produces results
        consistent with the ground-truth (relaxed: top-k rows,
        column-order and row-order insensitive).
        """
        if not trajectory:
            return False
        
        # Get the final step's SQL
        final_sql = trajectory[-1][1]
        
        try:
            exec_result = self.db_manager.execute_sql(final_sql, db_id)
            if not exec_result['success']:
                return False
            
            pred_data = exec_result['data']
            
            # Take only the first top_k rows for comparison
            gt_rows = (gt_result_data or [])[:top_k]
            pred_rows = (pred_data or [])[:top_k]
            
            # Both empty → match
            if not gt_rows and not pred_rows:
                return True
            
            # Different number of rows (within top_k) → mismatch
            if len(gt_rows) != len(pred_rows):
                return False
            
            # Normalize each row: stringify values then sort to ignore
            # column order, then sort all rows to ignore row order
            def normalize(rows):
                return sorted(
                    tuple(sorted(str(v) for v in row))
                    for row in rows
                )
            
            return normalize(gt_rows) == normalize(pred_rows)
            
        except Exception as e:
            print(f"  Execution verification error: {e}")
            return False
    
    def generate_all(self, num_examples: Optional[int] = None) -> Tuple[List[Dict], List[Dict]]:
        """Process the full dataset and return (sft_data, dpo_data)."""
        data = self.load_dataset()
        
        if num_examples:
            data = data[:num_examples]
        
        os.makedirs(self.config.save_dir, exist_ok=True)
        
        all_sft_data = []
        all_dpo_data = []
        
        print(f"Processing {len(data)} examples...")
        
        for idx, item in enumerate(tqdm(data)):
            result = self.process_single_example(idx, item)
            
            if result:
                # Process SFT data
                for trajectory in result['sft_data']:
                    sft_item = {
                        'question': result['question'],
                        'schema': result['schema'],
                        'foreign_keys': result['foreign_keys'],
                        'evidence': result['evidence'],
                        'decomposition': trajectory
                    }
                    all_sft_data.append(sft_item)
                
                # Process DPO data
                for pair in result['dpo_data']:
                    dpo_item = {
                        'question': result['question'],
                        'schema': result['schema'],
                        'evidence': result['evidence'],
                        'previous_steps': pair['context'],
                        'chosen': pair['correct'],
                        'rejected': pair['incorrect'],
                        'margin': pair['margin']
                    }
                    all_dpo_data.append(dpo_item)
                
                # Save intermediate results periodically
                if (idx + 1) % 100 == 0:
                    self._save_intermediate(all_sft_data, all_dpo_data, idx + 1)
        
        # Save final results
        self._save_final(all_sft_data, all_dpo_data)
        
        print(f"\nGeneration complete!")
        print(f"Success: {self.success_count}, Failure: {self.failure_count}, "
              f"Skipped (invalid GT SQL): {self.skip_count}")
        print(f"SFT samples: {len(all_sft_data)}")
        print(f"DPO samples: {len(all_dpo_data)}")
        
        return all_sft_data, all_dpo_data
    
    def _save_intermediate(self, sft_data: List, dpo_data: List, count: int) -> None:
        sft_path = os.path.join(self.config.save_dir, f'sft_data_checkpoint_{count}.json')
        dpo_path = os.path.join(self.config.save_dir, f'dpo_data_checkpoint_{count}.json')
        
        with open(sft_path, 'w', encoding='utf-8') as f:
            json.dump(sft_data, f, indent=2, ensure_ascii=False)
        
        with open(dpo_path, 'w', encoding='utf-8') as f:
            json.dump(dpo_data, f, indent=2, ensure_ascii=False)
    
    def _save_final(self, sft_data: List, dpo_data: List) -> None:
        sft_path = os.path.join(self.config.save_dir, 'sft_data.json')
        dpo_path = os.path.join(self.config.save_dir, 'dpo_data.json')
        
        with open(sft_path, 'w', encoding='utf-8') as f:
            json.dump(sft_data, f, indent=2, ensure_ascii=False)
        
        with open(dpo_path, 'w', encoding='utf-8') as f:
            json.dump(dpo_data, f, indent=2, ensure_ascii=False)
        
        # Save statistics
        stats = {
            'success_count': self.success_count,
            'failure_count': self.failure_count,
            'skip_count': self.skip_count,
            'sft_samples': len(sft_data),
            'dpo_samples': len(dpo_data)
        }
        
        stats_path = os.path.join(self.config.save_dir, 'statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)


def main():
    """Main function for data generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate decomposition training data")
    parser.add_argument("--data_dir", type=str, default="./data/BIRD/train",
                       help="Path to training data directory")
    parser.add_argument("--save_dir", type=str, default="./outputs/decomposition_data",
                       help="Path to save generated data")
    parser.add_argument("--api_key", type=str, required=False,
                       help="API key for LLM")
    parser.add_argument("--num_examples", type=int, default=None,
                       help="Number of examples to process")
    parser.add_argument("--seed", type=int, default=2024,
                       help="Random seed")
    
    args = parser.parse_args()
    
    config = DataConfig(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        api_key=args.api_key,
        seed=args.seed
    )
    
    generator = DecompositionGenerator(config)
    sft_data, dpo_data = generator.generate_all(args.num_examples)
    
    print(f"\nData saved to {args.save_dir}")


if __name__ == "__main__":
    main()

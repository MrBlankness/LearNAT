# Inference and evaluation for the trained LearNAT model.

import os
import json
import re
import sqlite3
import argparse
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompt import build_inference_prompt, format_schema, format_foreign_keys, SYSTEM_PROMPT
from AST import parse_sql_to_ast, remove_aliases


def load_bird_data(data_dir: str, n: int = None) -> List[Dict]:
    """Load BIRD-format data (dev.json / train.json + tables) into a list of dicts."""
    # ---- locate data file ------------------------------------------------
    data_file = None
    tables_file = None
    for stem in ("dev", "train"):
        cand_data   = os.path.join(data_dir, f"{stem}.json")
        cand_tables = os.path.join(data_dir, f"{stem}_tables.json")
        if os.path.exists(cand_data):
            data_file   = cand_data
            tables_file = cand_tables
            break

    if data_file is None:
        raise FileNotFoundError(
            f"No dev.json or train.json found in {data_dir}"
        )

    with open(data_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # ---- load table info -------------------------------------------------
    db_schema: Dict[str, Dict] = {}   # db_id → {schema, foreign_keys}
    if tables_file and os.path.exists(tables_file):
        with open(tables_file, "r", encoding="utf-8") as f:
            tables_list = json.load(f)

        for entry in tables_list:
            db_id = entry["db_id"]
            table_names = entry.get("table_names_original", [])
            col_names   = entry.get("column_names_original", [])  # [[table_idx, col_name], ...]
            fk_pairs    = entry.get("foreign_keys", [])            # [[col_idx_a, col_idx_b], ...]

            # Build schema dict: table → [columns]
            schema: Dict[str, List[str]] = {t: [] for t in table_names}
            for table_idx, col_name in col_names:
                if table_idx == -1:   # skip the wildcard "*" row
                    continue
                tname = table_names[table_idx]
                schema[tname].append(col_name)

            # Build foreign-key list: ["table_a.col_a", "table_b.col_b"]
            foreign_keys: List[List[str]] = []
            for col_a_idx, col_b_idx in fk_pairs:
                def _col_ref(idx):
                    tidx, cname = col_names[idx]
                    return f"{table_names[tidx]}.{cname}"
                try:
                    foreign_keys.append([_col_ref(col_a_idx), _col_ref(col_b_idx)])
                except (IndexError, TypeError):
                    pass

            db_schema[db_id] = {"schema": schema, "foreign_keys": foreign_keys}

    # ---- assemble output -------------------------------------------------
    samples: List[Dict] = []
    for item in raw_data:
        db_id = item.get("db_id", "")
        info  = db_schema.get(db_id, {"schema": {}, "foreign_keys": []})
        samples.append({
            "question":     item.get("question", ""),
            "schema":       info["schema"],
            "foreign_keys": info["foreign_keys"],
            "evidence":     item.get("evidence", ""),
            "SQL":          item.get("SQL", item.get("sql", "")),
            "db_id":        db_id,
            "difficulty":   item.get("difficulty", "unknown"),
        })

    if n is not None:
        samples = samples[:n]

    return samples


@dataclass
class InferenceConfig:
    """Configuration for inference."""
    model_path: str = "./outputs/final"
    max_length: int = 2048
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.95
    do_sample: bool = False
    device: str = "cuda"


class NL2SQLInference:
    """Load a trained checkpoint and generate SQL from natural language."""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        print(f"Loading model from {config.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_path,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
    
    def generate(
        self,
        question: str,
        schema: Dict[str, List[str]],
        foreign_keys: List[List[str]] = None,
        evidence: str = ""
    ) -> Dict[str, str]:
        """Generate SQL for a single NL question."""
        # Build prompt (user turn content)
        user_content = build_inference_prompt(
            question=question,
            schema=schema,
            foreign_keys=foreign_keys or [],
            evidence=evidence
        )

        # Apply chat template so Instruct models respond correctly
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_content}
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = f"{SYSTEM_PROMPT}\n\n{user_content}"

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True
        ).to(self.device)
        
        # Generate
        gen_kwargs = dict(
            max_new_tokens=self.config.max_new_tokens,
            do_sample=self.config.do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        # temperature / top_p are only valid when sampling
        if self.config.do_sample:
            gen_kwargs["temperature"] = self.config.temperature
            gen_kwargs["top_p"] = self.config.top_p

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # Decode
        generated = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # Extract SQL and decomposition from response
        result = self._parse_response(generated)
        
        return result
    
    def _parse_response(self, response: str) -> Dict[str, str]:
        """Extract SQL and decomposition steps from the raw model output."""
        result = {
            'sql': '',
            'decomposition': [],
            'raw_response': response
        }
        
        # Extract final SQL (look for ```sql block or last SELECT statement)
        sql_pattern = r'```sql\s*(.*?)\s*```'
        sql_matches = re.findall(sql_pattern, response, re.DOTALL | re.IGNORECASE)

        if sql_matches:
            result['sql'] = sql_matches[-1].strip().rstrip(';')
        else:
            # Try to find SELECT statement (MULTILINE so $ matches end-of-line,
            # and dot does NOT cross newlines, avoiding multi-statement greedy capture)
            select_pattern = r'(SELECT\s+.*?(?:;|$))'
            select_matches = re.findall(select_pattern, response, re.MULTILINE | re.IGNORECASE)
            if select_matches:
                result['sql'] = select_matches[-1].strip().rstrip(';')

        # Extract decomposition steps.
        # Handles two formats produced by the model:
        #   (a) SQL: ```sql\n<query>\n```
        #   (b) SQL: <query>   (plain text, possibly multi-line)
        step_header_pattern = re.compile(
            r'###?\s*(?:Subtask|Step)\s*\d+[:\s]*\n'
            r'Description:\s*([^\n]+)\n'   # group 1: description
            r'SQL:\s*\n?'                  # "SQL:" label (newline optional)
            r'(?:```(?:sql)?\s*\n(.*?)\n```|(.*?)(?=\n###|\Z))',  # group 2: fenced, group 3: plain
            re.DOTALL | re.IGNORECASE
        )
        for m in step_header_pattern.finditer(response):
            desc = m.group(1).strip()
            sql_fenced = m.group(2)
            sql_plain  = m.group(3)
            sql = (sql_fenced if sql_fenced is not None else sql_plain or '').strip().rstrip(';')
            if desc and sql:
                result['decomposition'].append((desc, sql))
        
        return result
    
    def batch_generate(
        self,
        examples: List[Dict],
        batch_size: int = 1
    ) -> List[Dict]:
        """Run generate() over a list of examples."""
        results = []
        
        for example in tqdm(examples, desc="Generating SQL"):
            result = self.generate(
                question=example['question'],
                schema=example.get('schema', {}),
                foreign_keys=example.get('foreign_keys', []),
                evidence=example.get('evidence', '')
            )
            
            result['question'] = example['question']
            result['gold_sql'] = example.get('SQL', example.get('sql', ''))
            result['db_id'] = example.get('db_id', '')
            
            results.append(result)
        
        return results


class SQLEvaluator:
    """Compute Execution Accuracy by running predicted and gold SQL against SQLite."""
    
    def __init__(self, db_dir: str):
        self.db_dir = db_dir
    
    def execute_sql(
        self,
        sql: str,
        db_id: str,
        timeout: int = 30
    ) -> Tuple[bool, Optional[List]]:
        """Execute a SQL query and return (success, rows)."""
        db_path = os.path.join(self.db_dir, db_id, f'{db_id}.sqlite')
        
        if not os.path.exists(db_path):
            return False, None
        
        try:
            conn = sqlite3.connect(db_path, timeout=timeout)
            conn.text_factory = lambda b: b.decode(errors="ignore")
            cursor = conn.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()
            conn.close()
            return True, results
        except Exception as e:
            return False, None
    
    def compare_results(
        self,
        pred_results: Optional[List],
        gold_results: Optional[List]
    ) -> bool:
        """Order-insensitive set comparison of two result sets."""
        if pred_results is None or gold_results is None:
            return False
        
        # Convert to sets for comparison (order-insensitive)
        try:
            pred_set = set(tuple(row) for row in pred_results)
            gold_set = set(tuple(row) for row in gold_results)
            return pred_set == gold_set
        except:
            return False
    
    def evaluate(
        self,
        predictions: List[Dict],
        verbose: bool = True
    ) -> Dict[str, float]:
        """Compute Execution Accuracy over a list of predictions."""
        correct = 0
        total = 0
        
        difficulty_results = {
            'simple': {'correct': 0, 'total': 0},
            'moderate': {'correct': 0, 'total': 0},
            'challenging': {'correct': 0, 'total': 0}
        }
        
        for pred in tqdm(predictions, desc="Evaluating", disable=not verbose):
            total += 1
            
            db_id = pred.get('db_id', '')
            pred_sql = pred.get('sql', '')
            gold_sql = pred.get('gold_sql', '')
            difficulty = pred.get('difficulty', 'unknown').lower()
            
            if not db_id or not pred_sql or not gold_sql:
                continue
            
            # Execute both SQLs
            pred_success, pred_results = self.execute_sql(pred_sql, db_id)
            gold_success, gold_results = self.execute_sql(gold_sql, db_id)
            
            # Compare results
            is_correct = pred_success and gold_success and self.compare_results(pred_results, gold_results)
            
            if is_correct:
                correct += 1
            
            # Track by difficulty
            if difficulty in difficulty_results:
                difficulty_results[difficulty]['total'] += 1
                if is_correct:
                    difficulty_results[difficulty]['correct'] += 1
        
        # Calculate metrics
        metrics = {
            'execution_accuracy': correct / total if total > 0 else 0,
            'correct': correct,
            'total': total
        }
        
        # Add difficulty-based metrics
        for diff, results in difficulty_results.items():
            if results['total'] > 0:
                metrics[f'{diff}_accuracy'] = results['correct'] / results['total']
                metrics[f'{diff}_total'] = results['total']
        
        if verbose:
            print("\n" + "=" * 60)
            print("Evaluation Results")
            print("=" * 60)
            print(f"Overall Execution Accuracy: {metrics['execution_accuracy']:.4f}")
            print(f"Correct: {correct} / {total}")
            
            for diff in ['simple', 'moderate', 'challenging']:
                if f'{diff}_accuracy' in metrics:
                    print(f"{diff.capitalize()}: {metrics[f'{diff}_accuracy']:.4f} ({difficulty_results[diff]['correct']}/{difficulty_results[diff]['total']})")
            print("=" * 60)
        
        return metrics


def main():
    parser = argparse.ArgumentParser(description="LearNAT Inference")

    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--data_path", type=str, default=None,
                       help="Path to test data JSON (optional; use --bird_data_dir for BIRD format)")
    parser.add_argument("--bird_data_dir", type=str, default=None,
                       help="Path to a BIRD split directory containing dev.json/train.json "
                            "and the corresponding *_tables.json (used for BIRD-format evaluation)")
    parser.add_argument("--bird_n_samples", type=int, default=30,
                       help="Number of samples to take from the BIRD dataset (default: 30)")
    parser.add_argument("--db_dir", type=str, default=None,
                       help="Path to database directory (for evaluation); "
                            "inferred automatically when --bird_data_dir is used")
    parser.add_argument("--output_path", type=str, default="./predictions.json",
                       help="Path to save predictions")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Generation temperature")

    args = parser.parse_args()

    # Initialize inference engine
    config = InferenceConfig(
        model_path=args.model_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )
    inference = NL2SQLInference(config)

    if args.bird_data_dir:
        # ---- BIRD-format evaluation ----------------------------------------
        print(f"Loading {args.bird_n_samples} samples from BIRD directory: {args.bird_data_dir}")
        data = load_bird_data(args.bird_data_dir, n=args.bird_n_samples)
        print(f"Loaded {len(data)} samples.")

        print(f"Running inference on {len(data)} examples...")
        predictions = inference.batch_generate(data)

        # Save predictions
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        print(f"Predictions saved to {args.output_path}")

        # Determine database directory
        db_dir = args.db_dir
        if db_dir is None:
            # Try to infer: look for a *_databases sub-directory next to the JSON
            for stem in ("dev_databases", "train_databases"):
                candidate = os.path.join(args.bird_data_dir, stem)
                if os.path.isdir(candidate):
                    db_dir = candidate
                    break

        if db_dir:
            print(f"Evaluating with database directory: {db_dir}")
            evaluator = SQLEvaluator(db_dir)
            metrics = evaluator.evaluate(predictions)

            metrics_path = args.output_path.replace('.json', '_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"Metrics saved to {metrics_path}")
        else:
            print("No database directory found; skipping execution-accuracy evaluation.")

    elif args.data_path:
        # ---- Generic JSON evaluation ---------------------------------------
        with open(args.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"Running inference on {len(data)} examples...")
        predictions = inference.batch_generate(data)

        # Save predictions
        with open(args.output_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        print(f"Predictions saved to {args.output_path}")

        # Evaluate if database directory provided
        if args.db_dir:
            evaluator = SQLEvaluator(args.db_dir)
            metrics = evaluator.evaluate(predictions)

            # Save metrics
            metrics_path = args.output_path.replace('.json', '_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"Metrics saved to {metrics_path}")

    else:
        # ---- Interactive mode ----------------------------------------------
        print("\nInteractive NL2SQL Mode")
        print("Enter 'quit' to exit\n")

        while True:
            question = input("Question: ").strip()
            if question.lower() == 'quit':
                break

            # Simple demo schema
            demo_schema = {
                "users": ["id", "name", "age", "email"],
                "orders": ["id", "user_id", "amount", "date"]
            }
            demo_fk = [["orders.user_id", "users.id"]]

            result = inference.generate(
                question=question,
                schema=demo_schema,
                foreign_keys=demo_fk
            )

            print(f"\nGenerated SQL: {result['sql']}")

            if result['decomposition']:
                print("\nDecomposition:")
                for i, (desc, sql) in enumerate(result['decomposition'], 1):
                    print(f"  Step {i}: {desc}")
                    print(f"    SQL: {sql}")
            print()


if __name__ == "__main__":
    main()

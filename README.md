# LearNAT: Learning NL2SQL with AST-guided Task Decomposition

This is a reproduction of the paper **"LearNAT: Learning NL2SQL with AST-guided Task Decomposition for Large Language Models"** (ICLR 2026).

## 📖 Overview

LearNAT is a novel framework designed to enhance LLMs' task decomposition capabilities for NL2SQL tasks. It introduces:

1. **Decomposition Synthesis Procedure**: Uses AST-guided MCTS to automatically generate verifiable SQL decompositions from BIRD/Spider training sets, driven by an LLM API (e.g. GPT-4o)
2. **Two-Stage Training**:
   - **Stage 1 – SFT**: Warm-up fine-tuning on successful decomposition trajectories
   - **Stage 2 – MDPO**: Margin-Aware DPO with AST similarity margin for fine-grained preference optimization

The MDPO loss is defined as:

```
L_MDPO = -E[ log σ( β·(log π_θ(y_w|x) - log π_ref(y_w|x))
                   - β·(log π_θ(y_l|x) - log π_ref(y_l|x))
                   - Δ_r ) ]
```

where `Δ_r = sim(y_w, y*) - sim(y_l, y*)` is the AST-based quality margin between the chosen and rejected responses.

## 🏗️ Project Structure

```
LearNAT/
├── AST.py                 # SQL AST parsing, node classification, similarity
├── MCTS.py                # AST-guided Monte Carlo Tree Search
├── data_generation.py     # Decomposition data synthesis (calls GPT-4o via API)
├── trainers.py            # SFT and MDPO training logic (BasicTrainer / FSDPTrainer)
├── train.py               # Main training entry point (CLI)
├── LLM_inference.py       # Batch inference and execution-accuracy evaluation
├── prompt.py              # Prompt templates for decomposition and inference
├── utils.py               # Distributed training utilities, data helpers
├── config/
│   └── config.yaml        # Default training hyperparameters
├── data/                  # Dataset directory (BIRD / Spider)
├── outputs/               # Generated data and model checkpoints
│   ├── decomposition_data_v4/   # Pre-generated BIRD training data (27 SFT + 27 DPO)
│   ├── sft_checkpoints/         # SFT model checkpoints
│   └── mdpo_checkpoints/        # MDPO model checkpoints
├── scripts/               # Helper shell scripts
└── requirements.txt       # Python dependencies
```

## 🚀 Full Pipeline (4 Steps)

The complete LearNAT pipeline consists of four sequential steps:

```
Step 1: Data Generation  →  Step 2: SFT Training  →  Step 3: MDPO Training  →  Step 4: Inference & Evaluation
```

---

### Step 0: Environment Setup

```bash
# Create and activate conda environment
conda create -n nl2sql_reward python=3.10 -y
conda activate nl2sql_reward

# Install all dependencies
pip install -r requirements.txt
```

> **Verified dependencies**: The following packages are required but may need explicit installation if `requirements.txt` is incomplete:
> ```bash
> pip install omegaconf wandb sqlparse func-timeout
> ```

### Step 0.5: Download Datasets

Download BIRD and Spider datasets from their official sources:

- **BIRD**: [https://bird-bench.github.io/](https://bird-bench.github.io/)
- **Spider**: [https://yale-lily.github.io/spider](https://yale-lily.github.io/spider)

Place the datasets in the `data/` directory:

```
data/
├── BIRD/
│   ├── train/
│   │   ├── train.json
│   │   ├── train_tables.json
│   │   └── train_databases/
│   └── dev/
│       ├── dev.json
│       ├── dev_tables.json
│       └── dev_databases/
└── Spider/
    └── dev/
        ├── dev.json
        └── database/
```

---

### Step 1: Data Generation (Decomposition Synthesis)

Generate training data using MCTS with AST-guided search. This step calls **GPT-4o** (or another OpenAI-compatible API) to produce decomposition trajectories:

```bash
python data_generation.py \
    --data_dir ./data/BIRD/train \
    --save_dir ./outputs/decomposition_data \
    --api_key YOUR_OPENAI_API_KEY \
    --num_examples 1000 \
    --seed 2024
```

This produces three files in `--save_dir`:

| File | Description |
|------|-------------|
| `sft_data.json` | Successful decomposition trajectories for SFT training |
| `dpo_data.json` | `(chosen, rejected, margin)` contrastive pairs for MDPO training |
| `statistics.json` | Summary: success/failure counts, sample statistics |

Each `sft_data.json` entry contains: `question`, `schema`, `foreign_keys`, `evidence`, `decomposition`.
Each `dpo_data.json` entry contains: `question`, `schema`, `evidence`, `previous_steps`, `chosen`, `rejected`, `margin`.


---

### Step 2: SFT Training (Supervised Fine-Tuning)

Warm-up fine-tuning on successful decomposition trajectories. The recommended base model is `Qwen2.5-Coder-7B-Instruct`.

```bash
python train.py \
    --mode sft \
    --model_name /path/to/Qwen2.5-Coder-7B-Instruct \
    --data_path ./outputs/decomposition_data/sft_data.json \
    --output_dir ./outputs/sft_checkpoints \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --num_epochs 3 \
    --gradient_checkpointing
```

---

### Step 3: MDPO Training (Margin-Aware DPO)

Fine-grained preference optimization using AST-similarity margin. Takes the SFT checkpoint as the starting point.

```bash
python train.py \
    --mode mdpo \
    --model_name ./outputs/sft_checkpoints/final \
    --data_path ./outputs/decomposition_data/dpo_data.json \
    --output_dir ./outputs/mdpo_checkpoints \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --dpo_learning_rate 2e-6 \
    --beta 0.2 \
    --use_margin \
    --num_epochs 3 \
    --gradient_checkpointing
```

**Output**: Final MDPO checkpoint saved to `./outputs/mdpo_checkpoints/final/`.


#### Distributed Training with FSDP

For multi-GPU environments, add `--use_fsdp`:

```bash
python train.py \
    --mode sft \
    --model_name /path/to/Qwen2.5-Coder-7B-Instruct \
    --data_path ./outputs/decomposition_data/sft_data.json \
    --output_dir ./outputs/sft_checkpoints \
    --use_fsdp \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_epochs 3
```

#### Training Modes

| `--mode` | Description |
|----------|-------------|
| `sft`  | Supervised Fine-Tuning on decomposition trajectories |
| `dpo`  | Standard Direct Preference Optimization |
| `ipo`  | Identity Preference Optimization |
| `mdpo` | **Margin-Aware DPO** — add `--use_margin` to enable AST margin `Δ_r` |

---

### Step 4: Inference & Evaluation

> **Important**: The inference script uses `tokenizer.apply_chat_template()` to wrap prompts into the correct system/user message format required by Instruct models (e.g. Qwen2.5-Coder-7B-Instruct). Passing raw text directly to a chat model produces empty output; this is handled automatically.

#### BIRD-Format Evaluation (Recommended)

Use `--bird_data_dir` to point directly at a BIRD split directory. The script automatically reads `dev.json` / `train.json` and the matching `*_tables.json` to extract schemas and foreign keys, locates `dev_databases/` for SQL execution, and computes Execution Accuracy:

```bash
python LLM_inference.py \
    --model_path ./outputs/mdpo_checkpoints/final \
    --bird_data_dir /path/to/BIRD/dev_20240627 \
    --bird_n_samples 30 \
    --output_path ./outputs/predictions.json \
    --max_new_tokens 512
```

#### Batch Inference with Generic JSON

For a pre-processed JSON file where each entry already contains `question`, `schema`, `foreign_keys`, `evidence`, `SQL`, `db_id`:

```bash
python LLM_inference.py \
    --model_path ./outputs/mdpo_checkpoints/final \
    --data_path ./data/BIRD/dev/dev.json \
    --db_dir ./data/BIRD/dev/dev_databases \
    --output_path ./predictions.json \
    --max_new_tokens 512
```

Predictions are saved to `--output_path` as JSON. If `--db_dir` is provided, execution accuracy is computed automatically and saved to `predictions_metrics.json`.

#### Inference-Only (no evaluation)

```bash
python LLM_inference.py \
    --model_path ./outputs/mdpo_checkpoints/final \
    --data_path ./data/BIRD/dev/dev.json \
    --output_path ./predictions.json
```

#### Interactive Single-Query Mode

```bash
python LLM_inference.py \
    --model_path ./outputs/mdpo_checkpoints/final
# Then enter natural language questions at the prompt
```

#### Inference CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | *(required)* | Path to trained model checkpoint |
| `--bird_data_dir` | `None` | Path to BIRD split directory (auto-discovers schema, FK, and DB path) |
| `--bird_n_samples` | `30` | Number of BIRD samples to evaluate |
| `--data_path` | `None` | Path to generic test data JSON (alternative to `--bird_data_dir`) |
| `--db_dir` | `None` | SQLite database directory (required for EX evaluation with `--data_path`; auto-inferred with `--bird_data_dir`) |
| `--output_path` | `./predictions.json` | Where to save predictions |
| `--max_new_tokens` | `512` | Max tokens to generate per query |
| `--temperature` | `0.1` | Sampling temperature (only used when `do_sample=True`) |

---

## ⚡ Quick Reference: Full 4-Step Pipeline

```bash
# Step 1 – Data generation
python data_generation.py \
    --data_dir ./data/BIRD/train \
    --save_dir ./outputs/decomposition_data \
    --api_key YOUR_OPENAI_API_KEY \
    --num_examples 1000

# Step 2 – SFT
python train.py \
    --mode sft \
    --model_name /path/to/Qwen2.5-Coder-7B-Instruct \
    --data_path ./outputs/decomposition_data/sft_data.json \
    --output_dir ./outputs/sft_checkpoints \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --num_epochs 3 \
    --gradient_checkpointing

# Step 3 – MDPO
python train.py \
    --mode mdpo \
    --model_name ./outputs/sft_checkpoints/final \
    --data_path ./outputs/decomposition_data/dpo_data.json \
    --output_dir ./outputs/mdpo_checkpoints \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --dpo_learning_rate 2e-6 \
    --beta 0.2 \
    --use_margin \
    --num_epochs 3 \
    --gradient_checkpointing

# Step 4 – Inference + Evaluation (BIRD dev, first 30 samples)
python LLM_inference.py \
    --model_path ./outputs/mdpo_checkpoints/final \
    --bird_data_dir /path/to/BIRD/dev_20240627 \
    --bird_n_samples 30 \
    --output_path ./outputs/predictions.json \
    --max_new_tokens 512
```

---

## 📊 Evaluation Metrics

The primary metric is **Execution Accuracy (EX)**: a prediction is correct if its execution results match the gold SQL results (order-insensitive set comparison).

```python
from LLM_inference import SQLEvaluator

evaluator = SQLEvaluator(db_dir="./data/BIRD/dev/dev_databases")
metrics = evaluator.evaluate(predictions)
# metrics keys: execution_accuracy, correct, total,
#               simple_accuracy, moderate_accuracy, challenging_accuracy
print(f"Execution Accuracy: {metrics['execution_accuracy']:.4f}")
print(f"Correct: {metrics['correct']} / {metrics['total']}")
```

## 🧮 Key Algorithms

### AST-based Node Classification

MCTS nodes are classified based on their sub-tree relationship to the target SQL AST:

| Type | Meaning |
|------|---------|
| **Progressive** | Contributes new sub-structure toward the target SQL |
| **Redundant** | Already fully covered by previous decomposition steps |
| **Invalid** | Not a valid subtree of the target SQL |

```python
from AST import ASTClassifier, parse_sql_to_ast

classifier = ASTClassifier()
target_ast = parse_sql_to_ast("SELECT name FROM users WHERE age > 18")
sub_ast    = parse_sql_to_ast("SELECT name FROM users")

node_type = classifier.classify_node(sub_ast, target_ast)
# Returns: 'progressive'
```

### AST Similarity & Margin

```python
from AST import parse_sql_to_ast, calculate_ast_similarity

target   = parse_sql_to_ast("SELECT name FROM users WHERE age > 18")
chosen   = parse_sql_to_ast("SELECT name FROM users WHERE age > 18 ORDER BY name")
rejected = parse_sql_to_ast("SELECT id FROM orders")

# alpha: weight for node-level similarity vs edge-level (default 0.75)
sim_w = calculate_ast_similarity(chosen,   target, alpha=0.75)
sim_l = calculate_ast_similarity(rejected, target, alpha=0.75)
delta_r = sim_w - sim_l   # margin Δ_r used in MDPO loss
```

### MDPO Loss

```
L_MDPO = -E[ log σ( β·(log π_θ(y_w|x) - log π_ref(y_w|x))
                   - β·(log π_θ(y_l|x) - log π_ref(y_l|x))
                   - Δ_r ) ]
```

Compared to standard DPO, the margin `Δ_r` shifts the decision boundary proportionally to the AST quality gap between the chosen and rejected responses. When `--use_margin` is not set, `Δ_r = 0` and training reduces to standard DPO.

## ⚙️ Configuration

### Key Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `sft` | Training mode: `sft` / `dpo` / `ipo` / `mdpo` |
| `--model_name` | *(required)* | HuggingFace model name or local path |
| `--data_path` | *(required)* | Path to `sft_data.json` or `dpo_data.json` |
| `--max_length` | `2048` | Maximum input sequence length |
| `--learning_rate` | `2e-5` | Learning rate for SFT |
| `--dpo_learning_rate` | `2e-6` | Learning rate for DPO/MDPO |
| `--batch_size` | `4` | Per-GPU batch size |
| `--gradient_accumulation_steps` | `8` | Gradient accumulation steps |
| `--num_epochs` | `3` | Training epochs |
| `--warmup_ratio` | `0.1` | LR warmup ratio |
| `--beta` | `0.2` | KL penalty coefficient for DPO |
| `--label_smoothing` | `0.0` | DPO label smoothing |
| `--use_margin` | `False` | Enable AST margin `Δ_r` in MDPO loss |
| `--gradient_checkpointing` | `False` | Gradient checkpointing (saves GPU memory) |
| `--use_fsdp` | `False` | Enable FSDP multi-GPU training |
| `--output_dir` | `./outputs` | Checkpoint save directory |
| `--log_every` | `10` | Log every N steps |
| `--save_every` | `500` | Save checkpoint every N steps |
| `--seed` | `42` | Random seed |

Full defaults are in `config/config.yaml`.

### TrainerConfig (Python API)

```python
from trainers import TrainerConfig, BasicTrainer

config = TrainerConfig(
    model_name_or_path="Qwen/Qwen2.5-Coder-7B-Instruct",
    max_length=2048,
    learning_rate=2e-5,
    dpo_learning_rate=2e-6,
    batch_size=2,
    gradient_accumulation_steps=8,
    num_epochs=3,
    beta=0.2,        # KL penalty
    use_margin=True  # Enable AST margin for MDPO
)
```

### InferenceConfig (Python API)

```python
from LLM_inference import NL2SQLInference, InferenceConfig

config = InferenceConfig(
    model_path="./outputs/mdpo_checkpoints/final",
    max_length=2048,
    max_new_tokens=512,
    temperature=0.1,   # only used when do_sample=True
    top_p=0.95,        # only used when do_sample=True
    do_sample=False,   # False = greedy decoding (recommended for eval)
    device="cuda"
)
inference = NL2SQLInference(config)

result = inference.generate(
    question="Name movie titles released in 1945, sorted by popularity.",
    schema={"movies": ["movie_id", "movie_title", "movie_release_year", "movie_popularity"]},
    foreign_keys=[],
    evidence="released in 1945 refers to movie_release_year = 1945"
)
print(result['sql'])           # Final SQL string
print(result['decomposition']) # List of (description, sql) tuples
print(result['raw_response'])  # Full model output
```

### MCTSConfig (Python API)

```python
from MCTS import MCTSConfig

config = MCTSConfig(
    max_depth=5,
    simulations=50,
    exploration_constant=1.41,
    alpha=0.75,  # Weight for node-level similarity in margin calculation
)
```


## 🔗 References

```bibtex
@inproceedings{liao2026learnat,
  title     = {LearNAT: Learning NL2SQL with AST-guided Task Decomposition for Large Language Models},
  author    = {Liao, Weibin and Gao, Xin and Jia, Tianyu and Qiu, Rihong and Zhu, Yifan and Lin, Yang and Ma, Xinyu and Zhao, Junfeng and Wang, Yasha},
  booktitle = {International Conference on Learning Representations},
  year      = {2026}
}
```

## 📄 License

This project is licensed under the MIT License.

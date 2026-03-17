# Scripts

This directory contains benchmark and test automation scripts for the OpenVINO
Explicit Modeling API.

| Script | Purpose |
|--------|---------|
| `ifeval.py` | IFEval benchmark (instruction-following, English) |
| `ceval.py` | C-Eval benchmark (Chinese MCQ) |
| `mmlu_redux.py` | MMLU-Redux benchmark (English MCQ) |
| `auto_tests.py` | Functional test runner for modeling samples |
| `wwb.py` | Weight-wise benchmark |

---

## 1. Prerequisites (Common Setup)

All scripts in this directory require the OpenVINO + GenAI build environment.
Follow **Sections 1–3** of the
[root README.md](../README.md) to complete:

1. **Clone** — `repo.bat` creates the `openvino/`, `openvino.genai/`,
   `openvino-explicit-modeling/` workspace
2. **Build** — `build.bat` compiles both `openvino` and `openvino.genai`
3. **Python venv** — create `.venv`, install `openvino-tokenizers` and
   `transformers`

After that you should have:

```
my_workspace/                               # e.g. D:\data\code\Openvino_new_arch_2026
├── openvino/                               # OpenVINO core
├── openvino.genai/
│   └── build/bin/
│       └── modeling_qwen3_5.exe            # <-- key executable
├── openvino-explicit-modeling/
│   └── scripts/                            # <-- you are here
└── .venv/                                  # Python virtual environment
```

Verify:

```bash
dir my_workspace\openvino.genai\build\bin\modeling_qwen3_5.exe
```

---

## 2. LLM Accuracy Benchmarks — Overview

Three benchmarks evaluate model accuracy across different dimensions:

| Benchmark | Script | Language | Questions | Type | How It Scores |
|-----------|--------|:--------:|:---------:|------|---------------|
| **IFEval** | `ifeval.py` | English | 541 | Open-ended generation | Deterministic format-constraint checking (25 instruction types) |
| **C-Eval** | `ceval.py` | Chinese | ~1,346 | 4-choice MCQ | Letter matching (A/B/C/D) |
| **MMLU-Redux** | `mmlu_redux.py` | English | ~4,472 (OK) / ~4,800 (all) | 4-choice MCQ | Letter matching (A/B/C/D) |

### IFEval (Instruction-Following Evaluation)

IFEval (Google Research, arXiv:2311.07911) tests whether an LLM can follow
**verifiable format constraints**: "write at least 300 words", "do not use
commas", "respond in all caps", etc.

- 25 instruction types, 541 prompts, 834 total constraints
- Evaluation is **fully deterministic** — no LLM judge needed
- 4 metrics: Prompt-level strict/loose, Instruction-level strict/loose
- "Strict" = exact compliance; "Loose" = allows minor deviations
- Official eval code bundled in `ifeval_lib/`

### C-Eval (Comprehensive Chinese Evaluation Suite)

C-Eval covers **52 subjects** in 4 categories:

| Category | # Subjects | Examples |
|----------|:----------:|---------|
| STEM | 20 | Advanced Math, College Physics, Computer Network |
| Social Science | 13 | Law, Economics, Education, Geography |
| Humanities | 11 | Chinese Literature, History, Marxism |
| Other | 8 | Medicine, Accounting, Civil Servant |

- ~1,346 validation questions (the scored split)
- 5-shot prompting from dev split (configurable: `--n-shot`)
- **C-Eval Hard**: 8 difficult STEM subjects (advanced math, discrete math,
  probability, college chemistry/physics, HS math/chemistry/physics)
- Scores: **Macro avg** (mean of per-subject accuracies) + **Micro avg**
  (total correct / total questions) + per-category breakdown

### MMLU-Redux (Massive Multitask Language Understanding — Redux)

MMLU-Redux re-annotates the original MMLU test set (Hendrycks et al., 2021) to
identify and categorize errors.

- 57 subjects (abstract algebra → world religions)
- ~4,800 total questions; **~4,472 "OK"-annotated** after filtering
- 5-shot prompting from the original MMLU dev split
- `--filter-ok` (default) uses only quality-verified questions
- `--no-filter-ok` includes all questions
- Scores: **Macro avg** + **Micro avg** + per-subject breakdown

---

## 3. Benchmark Environment Setup

### 3.1 Python Dependencies

| Dependency | Required By | Install |
|------------|:-----------:|---------|
| `immutabledict`, `langdetect`, `nltk` | IFEval only | `pip install -r requirements.txt` |
| NLTK data (`punkt`, `punkt_tab`) | IFEval only | `python -m nltk.downloader punkt punkt_tab` |
| *(none)* | C-Eval, MMLU-Redux | Standard library only — no extra packages |

```bash
cd my_workspace
.venv\Scripts\activate

# Only needed if you plan to run IFEval
pip install -r openvino-explicit-modeling\scripts\requirements.txt
python -m nltk.downloader punkt punkt_tab
```

### 3.2 Download Qwen3.5 Models

The benchmarks target the Qwen3.5 model family:

| Index | Model | Parameters |
|:-----:|-------|:----------:|
| 1 | Qwen3.5-0.8B | 0.8 B |
| 2 | Qwen3.5-2B | 2 B |
| 3 | Qwen3.5-4B | 4 B |
| 4 | Qwen3.5-9B | 9 B |
| 5 | Qwen3.5-35B-A3B | 35 B (MoE, 3 B active) |

```bash
cd D:\data\models\Huggingface
git clone https://huggingface.co/Qwen/Qwen3.5-0.8B
git clone https://huggingface.co/Qwen/Qwen3.5-2B
git clone https://huggingface.co/Qwen/Qwen3.5-4B
git clone https://huggingface.co/Qwen/Qwen3.5-9B
git clone https://huggingface.co/Qwen/Qwen3.5-35B-A3B    # optional MoE model
```

> If your models live elsewhere, pass `--model-root <path>` to any script.

### 3.3 Download Benchmark Datasets

| Benchmark | Dataset Location | Download Command |
|-----------|-----------------|------------------|
| **IFEval** | Bundled in `ifeval_lib/input_data.jsonl` | *(nothing to do)* |
| **C-Eval** | `scripts/ceval_data/` | `python scripts/ceval.py --download` |
| **MMLU-Redux** | `scripts/mmlu_redux_data/` + `scripts/mmlu_dev_data/` | `python scripts/mmlu_redux.py --download` |

```bash
cd my_workspace

# C-Eval — downloads ~1.5 MB zip from ModelScope, extracts 52 subjects
python openvino-explicit-modeling\scripts\ceval.py --download

# MMLU-Redux — clones GitHub repo (57 subjects) + downloads 57 dev CSVs
python openvino-explicit-modeling\scripts\mmlu_redux.py --download
```

Both scripts also **auto-download** on first run if data is missing.

### 3.4 Verify Setup (Smoke Test)

Run 5 questions per benchmark to confirm everything works:

```bash
cd my_workspace
.venv\Scripts\activate

python openvino-explicit-modeling\scripts\ifeval.py --models 1 --limit 5
python openvino-explicit-modeling\scripts\ceval.py --models 1 --limit 5
python openvino-explicit-modeling\scripts\mmlu_redux.py --models 1 --limit 5
```

Each command should print per-question progress, then a results summary.

---

## 4. Running Benchmarks

### 4.1 Shared CLI Reference

All three scripts share a consistent interface:

**Selection arguments** (combinatorial — the script runs the Cartesian product):

| Argument | Format | Default | Description |
|----------|--------|:-------:|-------------|
| `--models` | `1,3,4` or `1~5` | `2` or `3` | Model index selectors |
| `--quant-list` | `1,2` or `all` | `1` | Quantization preset selectors |
| `--think` | `0,1` or `all` | `0` | Think mode (0=off, 1=on) |

**Sampling arguments** (passed through to `modeling_qwen3_5.exe`):

| Argument | Default | Notes |
|----------|:-------:|-------|
| `--temperature` | `1.0` | 0 = greedy argmax |
| `--top-p` | `0.95` | Nucleus sampling |
| `--top-k` | `20` | Top-K filtering |
| `--presence-penalty` | `1.5` | Encourages new tokens |
| `--repetition-penalty` | `1.0` | Multiplicative repeat penalty |
| `--frequency-penalty` | `0.0` | Additive frequency penalty |
| `--rng-seed` | `0` | 0 = random |

**Control arguments**:

| Argument | Default | Description |
|----------|:-------:|-------------|
| `--max-tokens` | `2048` (IFEval) / `32` (MCQ) | Max output tokens per question |
| `--limit N` | all | Only run first N questions (smoke tests) |
| `--resume <path>` | — | Resume from saved answers.jsonl / responses.jsonl |
| `--output-dir <path>` | auto | Custom output directory (single-run only) |

**MCQ-only arguments** (C-Eval and MMLU-Redux):

| Argument | Default | Description |
|----------|:-------:|-------------|
| `--n-shot` | `5` | Number of few-shot examples (0 = zero-shot) |
| `--download` | — | Download dataset and exit |
| `--filter-ok` / `--no-filter-ok` | `--filter-ok` | MMLU-Redux only: filter to OK-annotated questions |

**Quantization presets**:

| Index | Preset | Description |
|:-----:|--------|-------------|
| 1 | `[int4_asym, 128, int4_asym]` | INT4 weights + INT4 backup (default, fastest) |
| 2 | `[int4_asym, 128, int8_asym]` | INT4 weights + INT8 backup (better accuracy) |
| 3 | `[none, none, none]` | No quantization — FP16/BF16 (best accuracy) |

### 4.2 IFEval

```bash
cd my_workspace

# Single model
python openvino-explicit-modeling\scripts\ifeval.py --models 3

# All 5 models
python openvino-explicit-modeling\scripts\ifeval.py --models 1~5

# Compare two quant presets
python openvino-explicit-modeling\scripts\ifeval.py --models 3 --quant-list 1,2

# Compare thinking vs non-thinking
python openvino-explicit-modeling\scripts\ifeval.py --models 3 --think 0,1

# Full matrix: 4 models x 2 quants x 2 think = 16 runs
python openvino-explicit-modeling\scripts\ifeval.py --models 1~4 --quant-list 1,2 --think 0,1

# Quick 10-prompt test
python openvino-explicit-modeling\scripts\ifeval.py --models 3 --limit 10
```

### 4.3 C-Eval

```bash
# Run Qwen3.5-4B on all 52 subjects (5-shot, ~1346 questions)
python openvino-explicit-modeling\scripts\ceval.py --models 3

# Run first 4 model sizes
python openvino-explicit-modeling\scripts\ceval.py --models 1~4

# Zero-shot evaluation
python openvino-explicit-modeling\scripts\ceval.py --models 3 --n-shot 0

# Quick smoke test
python openvino-explicit-modeling\scripts\ceval.py --models 3 --limit 10

# Greedy decoding
python openvino-explicit-modeling\scripts\ceval.py --models 3 --temperature 0
```

### 4.4 MMLU-Redux

```bash
# Run Qwen3.5-4B with OK filter (5-shot, ~4472 questions)
python openvino-explicit-modeling\scripts\mmlu_redux.py --models 3

# Include all questions (not just OK-filtered)
python openvino-explicit-modeling\scripts\mmlu_redux.py --models 3 --no-filter-ok

# Run first 4 model sizes
python openvino-explicit-modeling\scripts\mmlu_redux.py --models 1~4

# Zero-shot evaluation
python openvino-explicit-modeling\scripts\mmlu_redux.py --models 3 --n-shot 0

# Quick smoke test
python openvino-explicit-modeling\scripts\mmlu_redux.py --models 3 --limit 10
```

---

## 5. Output Structure

Each run creates a timestamped batch directory under the workspace root:

```
my_workspace/
├── results_ifeval/
│   └── 20250601_143022/                          # batch timestamp
│       ├── Qwen3.5-4B_int4asym_think0_t1.0_n541/
│       │   ├── responses.jsonl                   # raw model responses
│       │   ├── results.json                      # machine-readable metrics
│       │   └── summary.txt                       # human-readable summary
│       └── batch_summary.md                      # markdown table of all runs
│
├── results_ceval/
│   └── 20250601_150000/
│       ├── Qwen3.5-4B_int4asym_think0_t1.0_n1346/
│       │   ├── answers.jsonl                     # per-question answers
│       │   ├── results.json
│       │   └── summary.txt                       # includes category breakdown
│       └── batch_summary.md
│
└── results_mmlu_redux/
    └── 20250601_160000/
        ├── Qwen3.5-4B_int4asym_think0_t1.0_n4472/
        │   ├── answers.jsonl
        │   ├── results.json
        │   └── summary.txt
        └── batch_summary.md
```

### Output Files

| File | Content |
|------|---------|
| `responses.jsonl` / `answers.jsonl` | One JSON line per question: raw response + parsed answer. Used for `--resume`. |
| `results.json` | Full metrics: per-subject accuracy, category breakdown, timing, config. |
| `summary.txt` | Console-printed summary: top/bottom subjects, category scores, aggregates. |
| `batch_summary.md` | Markdown table comparing all model/quant/think combinations in the batch. |

---

## 6. Resuming Interrupted Runs

All three scripts **auto-resume**. If a run is interrupted (Ctrl+C, power
failure), re-run the exact same command. The script will:

1. Find the existing `answers.jsonl` / `responses.jsonl` in the output directory
2. Load already-completed questions
3. Continue from the next unanswered question

Explicit resume from a specific file:

```bash
python openvino-explicit-modeling\scripts\ceval.py --models 3 --resume path\to\answers.jsonl
```

---

## 7. Expected Run Times

Approximate times on Intel Arc A770 with INT4 quantization (`--quant-list 1`):

| Benchmark | Questions | Qwen3.5-0.8B | Qwen3.5-4B | Qwen3.5-9B |
|-----------|:---------:|:------------:|:----------:|:----------:|
| IFEval | 541 | ~30 min | ~60 min | ~90 min |
| C-Eval | ~1,346 | ~20 min | ~45 min | ~70 min |
| MMLU-Redux | ~4,472 | ~60 min | ~140 min | ~220 min |

C-Eval and MMLU-Redux are faster **per question** than IFEval because MCQ uses
`--max-tokens 32` (only a letter) vs. IFEval's `--max-tokens 2048` (full text).

---

## 8. auto_tests.py — Functional Test Runner

`auto_tests.py` runs functional checks for greedy causal LM, Qwen3-VL
image-language samples, and Z-Image turbo flow, capturing results in a
markdown report.

### Prerequisites

- OpenVINO GenAI build complete (exe files under `openvino.genai/build/`)
- Model directories under `D:\data\models` (or pass `--models-root`)
- `test.jpg` in this `scripts/` folder (used by VL tests)

### Usage

```bash
python auto_tests.py                          # run all tests
python auto_tests.py --list                   # show numbered test list
python auto_tests.py --tests 0,1,2            # specific tests by index
python auto_tests.py --tests 1~5              # index range
python auto_tests.py --tests 1~5,7,8~10       # mixed ranges and indices
python auto_tests.py --tests all              # all tests
python auto_tests.py --models-root D:\data\models\custom
```

### CLI Arguments

| Argument | Description |
|----------|-------------|
| `--root PATH` | Root folder of `openvino-explicit-modeling` (default: repo parent) |
| `--models-root PATH` | Folder hosting model files (default: `D:\data\models`) |
| `--list` | Print available tests and exit |
| `--tests INDEX...` | Run only specified test indices |

### Test Indices

| Range | Tests |
|-------|-------|
| 0–6 | Text-generation samples (Qwen3 0.6B/4B, SmolLM, Youtu, GGUF variants) |
| 7–9 | Qwen3-VL (2B/4B/8B) using `test.jpg` |
| 10 | Z-Image-Turbo (outputs `cat2.bmp`) |

### Output

A markdown report is generated at `my_workspace/reports/test_report_<timestamp>.md`
with TTFT, throughput, duration, and related metrics. Failed cases are listed in
stderr.

### Model Downloads for auto_tests.py

```bash
cd D:\data\models
git clone https://huggingface.co/Qwen/Qwen3-0.6B
git clone https://huggingface.co/Qwen/Qwen3-4B
git clone https://huggingface.co/HuggingFaceTB/SmolLM3-3B
git clone https://huggingface.co/tencent/Youtu-LLM-2B
git clone https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
git clone https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
git clone https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct
git clone https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
```

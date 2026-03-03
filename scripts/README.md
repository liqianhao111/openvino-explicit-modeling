# auto_tests.py

## Overview

`auto_tests.py` bundles several OpenVINO GenAI checks for greedy causal LM, the Qwen3-VL image-language samples, and the Z-Image turbo flow so you can run the same commands from one place and capture a markdown report.

## Prerequisites

- Python 3.8+ (run `python -V` to confirm).
- The OpenVINO GenAI repo already built so the exe files under `openvino.genai/build/...` exist.
- Model directories under `D:\data\models` (or a custom folder passed through `--models-root`).
- `test.jpg` placed in this `scripts/` folder: `openvino-new-arch/scripts/test.jpg`. The VL tests always point to this file path.

## Key behavior

- Default `--root` is the parent of this script (`openvino-new-arch`). The script resolves relative roots against the current working directory.
- It prepends both `openvino.genai/build/openvino_genai` and `openvino/bin/intel64/RelWithDebInfo` to `PATH`, and sets `OV_GENAI_USE_MODELING_API=1` before executing each test.
- Each test logs performance, generated text, and the environment in `reports/test_report_<timestamp>.md` next to the repo root.

## Usage

```sh
python auto_tests.py                  # run all tests with defaults
python auto_tests.py --list           # show numbered list of tests without running them
python auto_tests.py --tests 0 1 2    # pick specific tests by index
python auto_tests.py --models-root D:\data\models\custom
python auto_tests.py --root .. --tests 7 8 9
```

### CLI arguments

- `--root PATH`: Root folder of `openvino-new-arch` (defaults to the repo parent). Can be absolute or relative.
- `--models-root PATH`: Folder that hosts model files (defaults to `D:\data\models`).
- `--list`: Print all available tests with indices and exit.
- `--tests INDEX...`: Run only the specified test indices (separate with spaces or commas).

## Test indices (current set)

0–6: text-generation samples (Qwen3 0.6B/4B/SmolLM/Youtu plus GGUF variants).  
7–9: Qwen3-VL (2B/4B/8B) using `scripts/test.jpg`.  
10: Z-Image-Turbo sample that emits `cat2.bmp`.

## Next steps

- Edit `models-root` or add more tests in `auto_tests.py` if you add new models or samples.
- Run `python auto_tests.py --tests 7 8 9` periodically to verify the VL pipeline against `test.jpg`.

## Model download commands

Each test assumes the models live under `D:\data\models`. If you have `git lfs` installed and a Hugging Face token configured (`huggingface-cli login`), you can populate the folder with the following commands:

```sh
cd D:\data\models
git clone https://huggingface.co/Qwen/Qwen3-0.6B Qwen3-0.6B
git clone https://huggingface.co/Qwen/Qwen3-4B Qwen3-4B
git clone https://huggingface.co/HuggingFaceTB/SmolLM3-3B SmolLM3-3B
git clone https://huggingface.co/tencent/Youtu-LLM-2B Youtu-LLM-2B
git clone https://huggingface.co/unsloth/Qwen3-0.6B-GGUF gguf/Qwen3-0.6B-BF16.gguf
git clone https://huggingface.co/unsloth/Qwen3-4B-GGUF gguf/Qwen3-4B-BF16.gguf
git clone https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct Qwen3-VL-2B-Instruct
git clone https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct Qwen3-VL-4B-Instruct
git clone https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct Qwen3-VL-8B-Instruct
git clone https://huggingface.co/Tongyi-MAI/Z-Image-Turbo Z-Image-Turbo
git clone https://huggingface.co/DavidAU/Qwen3-MOE-4x0.6B-2.4B-Writing-Thunder-V1.2 Qwen3-MOE-4x0.6B-2.4B-Writing-Thunder-V1.2
git clone https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF gguf/Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf
```

If you prefer to fetch a subset or keep the models in a different place, pass `--models-root` to `auto_tests.py`. The script resolves each test path relative to that folder.

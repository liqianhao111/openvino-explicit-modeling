# openvino-new-arch

## build

```bash
git clone https://github.com/liangali/openvino-new-arch.git

cd openvino-new-arch\openvino
mkdir build
cd build

# normal build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
cmake --build . --config RelWithDebInfo --verbose -j24 # you can also use vs2022 to build
# build with ULT
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -DENABLE_DEBUG_CAPS=ON -DENABLE_TESTS=ON -DENABLE_FUNCTIONAL_TESTS=ON
cmake --build . --config RelWithDebInfo --verbose -j24

cd openvino-new-arch\openvino.genai
mkdir build
cd build

cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DOpenVINO_DIR=D:\data\code\Openvino_new_arch_poc\openvino-new-arch\openvino\build ..
cmake --build . --config RelWithDebInfo --verbose -j24

# build with safetensor support
cd openvino.genai\build
cmake -DENABLE_GGUF=ON -DENABLE_SAFETENSORS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
cmake --build . --config RelWithDebInfo --verbose -j24
```

## run benchmark_genai.exe

```bash
set PATH=D:\data\code\Openvino_new_arch_poc\openvino-new-arch\openvino.genai\build\openvino_genai;%PATH%
cd D:\data\code\Openvino_new_arch_poc\openvino-new-arch\openvino\bin\intel64\Release
D:\data\code\Openvino_new_arch_poc\openvino-new-arch\openvino.genai\build\samples\cpp\text_generation\RelWithDebInfo\benchmark_genai.exe -d GPU -m D:\data\code\Openvino_new_arch_poc\ov_models\qwen3-4b-int8 -p "introduce ffmpeg in details" -n 1 --mt 10
```

## run OV GGUF models

```bash
set PATH=D:\data\code\Openvino_new_arch_poc\openvino-new-arch\openvino.genai\build\openvino_genai;D:\data\code\Openvino_new_arch_poc\openvino-new-arch\openvino\bin\intel64\RelWithDebInfo;%PATH%
cd D:\data\code\Openvino_new_arch_poc\openvino-new-arch\openvino\bin\intel64\RelWithDebInfo
D:\data\code\Openvino_new_arch_poc\openvino-new-arch\openvino.genai\build\samples\cpp\text_generation\RelWithDebInfo\greedy_causal_lm.exe D:\data\models\gguf\Qwen3-0.6B-BF16.gguf "introduce ffmpeg in details" GPU 1 1 100
```

## run OV HF/safetensors models

```bash
# legacy mode(building blocks)
$env:OV_GENAI_USE_MODELING_API="0"; .\greedy_causal_lm.exe C:\Users\gta\chuansheng\qwen3-4b-hf "introduce ffmpeg in details" GPU 1 1 100

# legacy mode(building blocks) with in-flight quantization (INT4 symmetric)
$env:OV_GENAI_USE_MODELING_API="0"; $env:OV_GENAI_INFLIGHT_QUANT_MODE="int4_sym"; $env:OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE="128"; .\greedy_causal_lm.exe C:\Users\gta\chuansheng\qwen3-4b-hf "introduce ffmpeg in details" GPU 1 1 100

# modeling API mode (zero-copy enabled by default when MODELING_API=1)
$env:OV_GENAI_USE_MODELING_API="1"; .\greedy_causal_lm.exe C:\Users\gta\chuansheng\qwen3-4b-hf "introduce ffmpeg in details" GPU 1 1 100

# explicitly enable zero copy mode (memory-mapped, no copy) - recommended for large models
$env:OV_GENAI_USE_MODELING_API="1"; $env:OV_GENAI_USE_ZERO_COPY="1"; .\greedy_causal_lm.exe C:\Users\gta\chuansheng\qwen3-4b-hf "introduce ffmpeg in details" GPU 1 1 100

# explicitly disable zero copy (memory copy mode)
$env:OV_GENAI_USE_MODELING_API="1"; $env:OV_GENAI_USE_ZERO_COPY="0"; .\greedy_causal_lm.exe C:\Users\gta\chuansheng\qwen3-4b-hf "introduce ffmpeg in details" GPU 1 1 100
```

## run models via unified loader(only modeling API is supported)
```bash
# modeling API mode
$env:OV_GENAI_USE_UNIFIED_LOADER="1"; .\greedy_causal_lm.exe C:\Users\gta\chuansheng\qwen3-4b-hf "introduce ffmpeg in details" GPU 1 1 100
```

## Environment Variables for In-flight Quantization

The in-flight quantization feature supports NNCF-compatible mixed precision weight compression.

### Basic Configuration

| Environment Variable | Values | Default | Description |
|---------------------|--------|---------|-------------|
| `OV_GENAI_USE_MODELING_API` | `0`, `1` | `0` | Enable modeling API mode |
| `OV_GENAI_USE_ZERO_COPY` | `0`, `1` | `1` (when MODELING_API=1) | Enable zero-copy memory mapping |
| `OV_GENAI_INFLIGHT_QUANT_MODE` | `INT4_SYM`, `INT4_ASYM`, `INT8_SYM`, `INT8_ASYM` | None | Primary quantization mode |
| `OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE` | Integer | `128` | Group size for INT4 quantization |

### NNCF-Compatible Options

| Environment Variable | Values | Default | Description |
|---------------------|--------|---------|-------------|
| `OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE` | `INT8_ASYM`, `INT8_SYM`, `INT4_SYM`, `INT4_ASYM`, `NONE` | `INT8_ASYM` | Quantization mode for sensitive layers (embeddings, lm_head). Set to same as primary mode to quantize all layers uniformly. Set to `NONE` to skip quantizing sensitive layers. |
| `OV_GENAI_INFLIGHT_QUANT_VERBOSE` | `0`, `1` | `0` | Enable verbose logging for quantization decisions |

### Advanced Selection Options

| Environment Variable | Values | Default | Description |
|---------------------|--------|---------|-------------|
| `OV_GENAI_INFLIGHT_QUANT_INCLUDE` | Comma-separated patterns | None | Include patterns (e.g., `*mlp*,*attn*`) |
| `OV_GENAI_INFLIGHT_QUANT_EXCLUDE` | Comma-separated patterns | None | Exclude patterns |
| `OV_GENAI_INFLIGHT_QUANT_LAYER_RANGE` | `start-end` | None | Layer range (e.g., `10-20`) |
| `OV_GENAI_INFLIGHT_QUANT_MIN_SIZE` | Integer | `0` | Minimum weight size in elements |
| `OV_GENAI_INFLIGHT_QUANT_MAX_SIZE` | Integer | `0` (unlimited) | Maximum weight size in elements |

### Usage Examples

```powershell
# NNCF-compatible INT4 quantization (default behavior)
# - Attention/MLP layers: INT4_SYM with group_size=128
# - Embeddings/lm_head: INT8_ASYM per-channel
$env:OV_GENAI_USE_MODELING_API="1"
$env:OV_GENAI_INFLIGHT_QUANT_MODE="int4_sym"
$env:OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE="128"
.\greedy_causal_lm.exe C:\path\to\model "Hello" GPU 1 1 100

# Skip quantizing sensitive layers (keep embeddings/lm_head as FP16)
$env:OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE="NONE"
.\greedy_causal_lm.exe C:\path\to\model "Hello" GPU 1 1 100

# Quantize all layers with INT4 (like NNCF --all-layers)
$env:OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE="INT4_SYM"
.\greedy_causal_lm.exe C:\path\to\model "Hello" GPU 1 1 100

# Enable verbose logging to see per-weight quantization decisions
$env:OV_GENAI_INFLIGHT_QUANT_VERBOSE="1"
.\greedy_causal_lm.exe C:\path\to\model "Hello" GPU 1 1 100
```

## run HF VLM/diffusion models

```bash
set PATH=D:\data\code\Openvino_new_arch_poc\openvino-new-arch\openvino.genai\build\openvino_genai;D:\data\code\Openvino_new_arch_poc\openvino-new-arch\openvino\bin\intel64\RelWithDebInfo;%PATH%
cd D:\data\code\Openvino_new_arch_poc\openvino-new-arch\openvino.genai\build\src\cpp\src\modeling\samples\RelWithDebInfo

# qwen3-vl

modeling_qwen3_vl.exe D:\data\models\Huggingface\Qwen3-VL-2B-Instruct test.jpg "describe this picture" GPU 100
modeling_qwen3_vl.exe D:\data\models\Huggingface\Qwen3-VL-4B-Instruct test.jpg "describe this picture" GPU 100

# z-image 
modeling_zimage.exe D:\data\models\Huggingface\Z-Image-Turbo "a cute cat" cat.bmp GPU 256 256 8 0 0.0
modeling_zimage.exe D:\data\models\Huggingface\Z-Image-Turbo "a cute cat" cat.bmp GPU 256 256 8 0 0.0 zimage_dump_256


# wan2.1 t2v
modeling_wan_t2v.exe D:\data\models\Huggingface\Wan2.1-T2V-1.3B-Diffusers "a cat playing piano" wan_t2v_out GPU 480 832 81 50 0 5.0 "" 512
modeling_wan_t2v.exe D:\data\models\Huggingface\Wan2.1-T2V-1.3B-Diffusers "a cat playing piano" wan_t2v_out GPU 256 384 33 30 0 5.0 "" 512
modeling_wan_t2v.exe D:\data\models\Huggingface\Wan2.1-T2V-1.3B-Diffusers "a cat playing piano" wan_t2v_out GPU 192 320 33 30 0 5.0 "" 512

modeling_deepseek_ocr2.exe D:\data\models\Huggingface\DeepSeek-OCR-2 test_ocr.png "Convert the document to markdown" GPU 300

# qwen3.5-35b text
set OV_GENAI_INFLIGHT_QUANT_MODE=int4_asym
set OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE=128
set OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE=int4_asym
modeling_qwen3_5.exe --model C:\data\models\Huggingface\Qwen3.5-35B-A3B-Base --mode text --prompt "write opencl gemm kernel and host code: " --output-tokens 300 --cache-model

# qwen3.5-35b vl
modeling_qwen3_5.exe --model C:\data\models\Huggingface\Qwen3.5-35B-A3B-Base --mode vl --image C:\data\code\Openvino_new_arch_poc\openvino-new-arch\scripts\test.jpg --prompt "describe this picture in details: " --output-tokens 300 --cache-model

# qwen3.5 cache naming (quantized)
# text:   qwen3_5_text_q4a_b4a_g128.xml / .bin
# vision: qwen3_5_vision_q4a_b4a_g128.xml / .bin

# dflash
modeling_dflash.exe D:\data\models\Huggingface\Qwen3-4B D:\data\models\Huggingface\Qwen3-4B-DFlash-b16 "introduce ffmpeg in details" GPU 100 16


## run OV GPUPlugin ULT

```bash
cd D:\data\code\Openvino_new_arch_poc\openvino-new-arch\openvino\bin\intel64\RelWithDebInfo
ov_gpu_unit_tests.exe --gtest_filter=*fused_mlp*
```

## run modeling API ULT

```bash
cd D:\data\code\Openvino_new_arch_poc\openvino-new-arch\openvino\bin\intel64\RelWithDebInfo
D:\data\code\Openvino_new_arch_poc\openvino-new-arch\openvino.genai\build\src\cpp\src\modeling\RelWithDebInfo\test_modeling_api.exe --gtest_list_tests
D:\data\code\Openvino_new_arch_poc\openvino-new-arch\openvino.genai\build\src\cpp\src\modeling\RelWithDebInfo\test_modeling_api.exe --gtest_filter=RMSNormLayer*
```

## OV debug log

```bash
set OPENVINO_LOG_LEVEL=5

# dump transformation pass log
set OV_VERBOSE=4
0 - ov::log::Level::NO
1 - ov::log::Level::ERR
2 - ov::log::Level::WARNING
3 - ov::log::Level::INFO
4 - ov::log::Level::DEBUG
5 - ov::log::Level::TRACE

# Logging of the pattern matching

In order to utilzie the logging, first, you need to set the CMake flag -DENABLE_OPENVINO_DEBUG=ON
NOTE: the logging would also work if your build is configured as Release
In order to start logging, set the environmental variable OV_MATCHER_LOGGING to true/ON before running your executable or script as following: OV_MATCHER_LOGGING=true ./your_amazing_program
If you want to log only specific matchers, use the OV_MATCHERS_TO_LOG environmental variable and provide their names separated as commas: OV_MATCHER_LOGGING=true OV_MATCHERS_TO_LOG=EliminateSplitConcat,MarkDequantization ./your_amazing_program
You can also set the environmental variable OV_VERBOSE_LOGGING to true, to turn on more verbose logging that would print more information about the nodes taking part in the matching process: OV_MATCHER_LOGGING=true OV_VERBOSE_LOGGING=true ./your_amazing_program


set OV_ENABLE_PROFILE_PASS=graph_pass.log
set OV_ENABLE_SERIALIZE_TRACING=true


set OV_MATCHER_LOGGING=true
set OV_VERBOSE_LOGGING=true

set OV_GPU_DUMP_GRAPHS_PATH=D:\data\code\Openvino_new_arch_poc\graphs

# option1: use Graphviz to convert .dot file to .svg for visualization
# opiton2: use Graphviz Online for visualization: https://graphvizonline.net/ or https://github.com/dreampuf/GraphvizOnline
# option3: vscode Graphviz extension (Graphviz Preview)

```

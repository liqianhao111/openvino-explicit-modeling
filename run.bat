@REM @echo off
@REM setlocal

set "ROOT_DIR=%~dp0"
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"

set OV_GENAI_USE_MODELING_API=1
set OV_GENAI_INFLIGHT_QUANT_MODE=int4_asym
set OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE=128
set OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE=int4_asym

set "PATH=%ROOT_DIR%\openvino\bin\intel64\RelWithDebInfo;%ROOT_DIR%\openvino.genai\build\openvino_genai;%PATH%"
cd /d "%ROOT_DIR%\openvino.genai\build\src\cpp\src\modeling\samples\RelWithDebInfo"

@REM greedy_causal_lm.exe C:\data\models\Huggingface\Qwen3.5-35B-A3B-Base "ffmpeg is tool for " GPU 0 1 100
@REM modeling_qwen3_5.exe --model C:\data\models\Huggingface\Qwen3.5-35B-A3B-Base --mode text --prompt "write opencl gemm kernel and host code: " --output-tokens 30 --cache-model

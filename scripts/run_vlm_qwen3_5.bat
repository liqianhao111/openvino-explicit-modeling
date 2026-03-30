setlocal
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"
cd ..\..
set "OPENVINO_ROOT=%CD%"
cd /d "%SCRIPT_DIR%"

REM 1. Exclude site-packages to avoid loading pip openvino (conflicts with self-built)
set "PYTHONNOUSERSITE=1"

REM 2. Modeling API required for Qwen3.5 VLM
set "OV_GENAI_USE_MODELING_API=1"

REM 3. In-flight quantization (for large models)
if "%OV_GENAI_INFLIGHT_QUANT_MODE%"=="" (
    set "OV_GENAI_INFLIGHT_QUANT_MODE=int4_asym"
)
if "%OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE%"=="" (
    set "OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE=128"
)
if "%OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE%"=="" (
    set "OV_GENAI_INFLIGHT_QUANT_BACKUP_MODE=int4_asym"
)

echo [INFO] OV_GENAI_USE_MODELING_API=1
echo [INFO] OV_GENAI_INFLIGHT_QUANT_MODE=%OV_GENAI_INFLIGHT_QUANT_MODE%
echo [INFO] OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE=%OV_GENAI_INFLIGHT_QUANT_GROUP_SIZE%
echo.

REM 4. Python paths
set "OV_PYTHON="
if exist "%OPENVINO_ROOT%openvino\bin\intel64\Release\python" (
    set "OV_PYTHON=%OPENVINO_ROOT%openvino\bin\intel64\Release\python"
)
if "%OV_PYTHON%"=="" if exist "%OPENVINO_ROOT%openvino\build\lib\Release\python" (
    set "OV_PYTHON=%OPENVINO_ROOT%openvino\build\lib\Release\python"
)
if "%OV_PYTHON%"=="" if exist "%OPENVINO_ROOT%openvino\build\lib\python" (
    set "OV_PYTHON=%OPENVINO_ROOT%openvino\build\lib\python"
)
if "%OV_PYTHON%"=="" if exist "%OPENVINO_ROOT%openvino\build\bin\Release\python" (
    set "OV_PYTHON=%OPENVINO_ROOT%openvino\build\bin\Release\python"
)
if "%OV_PYTHON%"=="" if exist "%OPENVINO_ROOT%openvino\build\install\python" (
    set "OV_PYTHON=%OPENVINO_ROOT%openvino\build\install\python"
)

if "%OV_PYTHON%"=="" (
    echo [WARN] OpenVINO Python not found in build. Use system openvino if installed.
) else (
    set "PYTHONPATH=%OV_PYTHON%;%PYTHONPATH%"
)

set "PYTHONPATH=%OPENVINO_ROOT%openvino.genai\build;%PYTHONPATH%"

REM 5. DLL paths
set "OPENVINO_LIB_PATHS=%OPENVINO_ROOT%openvino\bin\intel64\Release"
if exist "%OPENVINO_ROOT%openvino\temp\Windows_AMD64\tbb\bin" (
    set "OPENVINO_LIB_PATHS=%OPENVINO_LIB_PATHS%;%OPENVINO_ROOT%openvino\temp\Windows_AMD64\tbb\bin"
)
if exist "%OPENVINO_ROOT%openvino\build\install\runtime\bin\intel64\Release" (
    set "OPENVINO_LIB_PATHS=%OPENVINO_LIB_PATHS%;%OPENVINO_ROOT%openvino\build\install\runtime\bin\intel64\Release"
)

set "PATH=%OPENVINO_ROOT%openvino.genai\build\openvino_genai;%PATH%"
set "PATH=%OPENVINO_ROOT%openvino\bin\intel64\Release;%PATH%"
if exist "%OPENVINO_ROOT%openvino.genai\build\bin\Release" (
    set "PATH=%OPENVINO_ROOT%openvino.genai\build\bin\Release;%PATH%"
)
if exist "%OPENVINO_ROOT%openvino\temp\Windows_AMD64\tbb\bin" (
    set "PATH=%OPENVINO_ROOT%openvino\temp\Windows_AMD64\tbb\bin;%PATH%"
)

REM 6. Run Python script
echo Running: python run_vlm_qwen3_5.py %*
echo.
python run_vlm_qwen3_5.py --thinking-mode%*

endlocal

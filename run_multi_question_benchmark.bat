@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM  DFlash Multi-Question Category Benchmark
REM
REM  Tests 10 different question types (reasoning, knowledge,
REM  math, code, creative writing, etc.) across Baseline FP16,
REM  DFlash FP16/FP16, and DFlash FP16/INT4 configurations.
REM
REM  Usage:
REM    run_multi_question_benchmark.bat
REM    run_multi_question_benchmark.bat --max-tokens 256
REM ============================================================

REM --- Paths ---
set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
for %%I in ("%SCRIPT_DIR%\..") do set "ROOT_DIR=%%~fI"

REM --- DLL environment ---
set "OPENVINO_BIN=%ROOT_DIR%\openvino\bin\intel64\Release"
set "TBB_BIN=%ROOT_DIR%\openvino\temp\Windows_AMD64\tbb\bin"
set "GENAI_DLL_DIR=%ROOT_DIR%\openvino.genai\build\openvino_genai"
set "GENAI_BIN_DIR=%ROOT_DIR%\openvino.genai\build\bin\Release"
set "PATH=%OPENVINO_BIN%;%TBB_BIN%;%GENAI_DLL_DIR%;%GENAI_BIN_DIR%;%PATH%"

REM --- Enable modeling API ---
set "OV_GENAI_USE_MODELING_API=1"

REM --- Disable thinking mode ---
set "OV_GENAI_DISABLE_THINKING=1"

echo ============================================================
echo   DFlash Multi-Question Category Benchmark
echo ============================================================
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%\run_multi_question_benchmark.ps1" %*
set "RC=%ERRORLEVEL%"
echo.
echo [Exit code: %RC%]
exit /b %RC%

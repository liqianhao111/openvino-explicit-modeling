@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "ROOT_DIR=%SCRIPT_DIR%\.."
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"

@REM # build openvino
if not exist "%ROOT_DIR%\openvino\build" mkdir "%ROOT_DIR%\openvino\build"
pushd "%ROOT_DIR%\openvino\build"
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release --verbose -j16
popd

@REM # build openvino.genai
if not exist "%ROOT_DIR%\openvino.genai\build" mkdir "%ROOT_DIR%\openvino.genai\build"
pushd "%ROOT_DIR%\openvino.genai\build"
cmake -DCMAKE_BUILD_TYPE=Release -DOpenVINO_DIR="%ROOT_DIR%\openvino\build" ..
cmake --build . --config Release --verbose -j16
popd

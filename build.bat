@echo off
setlocal

set "ROOT_DIR=%~dp0"
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"

@REM # build openvino
pushd "%ROOT_DIR%\openvino\build"
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
cmake --build . --config RelWithDebInfo --verbose -j16
popd

@REM # build openvino.genai
pushd "%ROOT_DIR%\openvino.genai\build"
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DOpenVINO_DIR="%ROOT_DIR%\openvino\build" ..
cmake --build . --config RelWithDebInfo --verbose -j16
popd

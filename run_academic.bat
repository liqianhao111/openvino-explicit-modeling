@echo off
set OV_GENAI_USE_MODELING_API=1
set PATH=D:\openvino-modeling-api\openvino.genai\build\openvino_genai;D:\openvino-modeling-api\openvino\bin\intel64\Release;D:\openvino-modeling-api\openvino\temp\Windows_AMD64\tbb\bin;%PATH%
D:\openvino-modeling-api\openvino.genai\build\bin\Release\modeling_academic_ds.exe D:\data\models\Huggingface\academic-ds-9B "what can i say?" GPU 1024 0 1
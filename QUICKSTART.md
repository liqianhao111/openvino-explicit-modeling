# OpenVINO Modeling API 快速入门

本文档说明如何克隆、构建并运行 OpenVINO Modeling API 项目。

---

## 1. 克隆仓库

将 `openvino`、`openvino.genai`、`openvino-explicit-modeling` 三个仓库克隆到同一父目录下：

```
openvino-modeling-api/
├── openvino/
├── openvino.genai/
└── openvino-explicit-modeling/
```

---

## 2. 拉取子模块

### openvino

```powershell
cd openvino
git submodule update --init --recursive
git lfs install
git lfs fetch --all
git lfs checkout
```

### openvino.genai

```powershell
cd ..\openvino.genai
git submodule update --init --recursive
git lfs install
git lfs fetch --all
git lfs checkout
```

---

## 3. 构建

```powershell
cd ..\openvino-explicit-modeling
build.bat
```

`build.bat` 会自动构建 openvino 和 openvino.genai，若不存在 `build` 目录会自动创建。

---

## 4. 运行测试

使用 `auto_tests.py` 运行自动化测试，需从**项目根目录**（即包含 `openvino`、`openvino.genai` 的目录）执行：

### 基础用法

```powershell
# 从 openvino-modeling-api 根目录执行，运行全部测试
cd d:\openvino-modeling-api
python openvino-explicit-modeling\scripts\auto_tests.py

# 指定项目根目录
python openvino-explicit-modeling\scripts\auto_tests.py --root .

# 指定模型根目录（默认 D:\data\models）
python openvino-explicit-modeling\scripts\auto_tests.py --models-root D:\data\models
```

### 列出可用测试

```powershell
python openvino-explicit-modeling\scripts\auto_tests.py --list
```

输出示例：
```
Models root: D:\data\models
Available tests:
[0] Modeling API Unit Tests -> N/A (ULT) (exe: openvino.genai\build\...\test_modeling_api.exe)
[1] Huggingface Qwen3-0.6B -> Huggingface\Qwen3-0.6B (exe: ...)
...
```

### 选择要运行的测试

```powershell
# 运行指定索引的测试：0, 1, 2
python openvino-explicit-modeling\scripts\auto_tests.py --tests 0 1 2

# 或使用逗号分隔
python openvino-explicit-modeling\scripts\auto_tests.py --tests 0,1,2

# 运行索引范围（1~5 表示 1,2,3,4,5）
python openvino-explicit-modeling\scripts\auto_tests.py --tests 1~5

# 组合：范围 + 单个索引
python openvino-explicit-modeling\scripts\auto_tests.py --tests 1~5,7,8~10

# 运行全部
python openvino-explicit-modeling\scripts\auto_tests.py --tests all
```

### 综合示例

```powershell
# 指定根目录和模型路径，只运行测试 0 和 1
python openvino-explicit-modeling\scripts\auto_tests.py --root . --models-root D:\data\models --tests 0 1

# 从 openvino-explicit-modeling 目录运行，指定上级为根目录
cd openvino-explicit-modeling
python scripts\auto_tests.py --root ..
```

---

## 5. 测试输出

- 测试完成后会在当前目录生成 Markdown 报告
- 报告中包含 TTFT、吞吐量、时长等统计
- 若有失败用例，会在 stderr 中列出

---

## 参数速查

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--root` | 项目根目录（含 openvino、openvino.genai） | 脚本所在目录的父目录 |
| `--models-root` | 模型文件根目录 | `D:\data\models` |
| `--list` | 列出可用测试并退出 | - |
| `--tests` | 要运行的测试索引，支持 `0,1,2`、`1~5`、`all` | 运行全部 |

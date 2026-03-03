
# setup repo

```bash
git clone https://github.com/liangali/openvino-new-arch.git

# add openvino codebase
git clone https://github.com/openvinotoolkit/openvino.git
cd openvino
git checkout releases/2025/4
git submodule update --init --recursive
git lfs install
git lfs fetch --all
git lfs checkout
# copy openvino (exclude .git/.github) to openvino-new-arch
# delete all .gitattributes files in openvino folder
git add .
git commit -m "Add OpenVINO codebase (releases/2025/4) with synced submodules and handled lfs files"
git push

# add openvino.genai codebase
git clone https://github.com/openvinotoolkit/openvino.genai.git
cd openvino.genai
git checkout releases/2025/4
git submodule update --init --recursive
git lfs install
git lfs fetch --all
git lfs checkout
# copy openvino.genai (exclude .git/.github) to openvino-new-arch
# delete all .gitattributes files in openvino.genai folder
git add .
git commit -m "Add openvino.genai codebase (releases/2025/4) with synced submodules and handled lfs files"
git push
```
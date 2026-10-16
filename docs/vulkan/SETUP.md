# Reconstructing the Vulkan experiment

## Pinned inputs

| Input | Version / commit |
|---|---|
| Existing Folivox | 0.1.0.162 |
| Existing PyTorch / torchaudio | 2.7.1+cpu / 2.7.1+cpu |
| Existing Python | uv-managed 3.12.15 |
| Existing ONNX Runtime | 1.23.2 |
| CosyVoice C++ | `67d45feb2f654bbc66cca25bff78464b3ec08fa1` (v0.2.0) |
| GGML | `3612c0af93f2cfc8f048295300a61187723b12ae` |
| PCRE2 submodule | `7978954dbd2efc6f2196869290553cf1871b4ce6` |
| Added Python package | gguf 0.19.0, installed with `--no-deps` |
| Local CMake / data | Ubuntu 3.28.3-1build7 |
| Local glslc / shaderc | Ubuntu 2023.8-1build1 |
| Local Vulkan headers | Ubuntu 1.3.275.0-1build1 |
| Local SPIR-V headers | Ubuntu 1.6.1+1.4.309.0-1~ubuntu0.24.04.2 |
| Local CMake runtime libraries | libjsoncpp25 1.9.5-6build1; librhash0 1.4.3-3build1 |

## Obtain the inputs

Run from the repository root. This assumes an existing working Folivox uv-tool
installation with its TTS/UI dependencies, already downloaded model weights,
GCC 13.3, GNU Make, uv, curl, tar, dpkg-deb, and a working RADV Vulkan runtime.
Adjust paths for other installation layouts. The recipe combines the original
tested steps; a fresh clone has not yet been rebuilt independently.

The source archives total about 10 MB; the extra tool packages about 18 MB.
Downloading/extracting these Ubuntu 24.04 packages does not install them into
the system. This list covers the added tools used on the test machine; it does
not resolve all dependencies on arbitrary distributions. Older versions may
eventually require Ubuntu package archives.

```bash
mkdir -p experiments/research experiments/toolchain experiments/cache experiments/logs experiments/outputs
curl -fL --retry 3 -o experiments/research/cosyvoice-source.tar.gz \
  https://github.com/Lourdle/cosyvoice.cpp/archive/67d45feb2f654bbc66cca25bff78464b3ec08fa1.tar.gz
curl -fL --retry 3 -o experiments/research/ggml-source.tar.gz \
  https://github.com/ggml-org/ggml/archive/3612c0af93f2cfc8f048295300a61187723b12ae.tar.gz
curl -fL --retry 3 -o experiments/research/pcre2-source.tar.gz \
  https://github.com/PCRE2Project/pcre2/archive/7978954dbd2efc6f2196869290553cf1871b4ce6.tar.gz
(cd experiments/research && sha256sum -c ../../docs/vulkan/downloads.sha256)
(cd experiments/research && apt-get download \
  cmake=3.28.3-1build7 cmake-data=3.28.3-1build7 \
  libjsoncpp25=1.9.5-6build1 librhash0=1.4.3-3build1 \
  glslc=2023.8-1build1 libshaderc1=2023.8-1build1 \
  libvulkan-dev=1.3.275.0-1build1 \
  spirv-headers=1.6.1+1.4.309.0-1~ubuntu0.24.04.2)
```

## Build and convert

The Python setup below assumes the tested Python 3.12 uv-tool layout. Keep that
existing installation in place: a .pth file references its packages read-only.
Conversion writes approximately 1.7 GiB locally and reads the existing RL weights.
The original models are not distributed by this fork.

1. If reconstructing a removed experimental environment, create it using the existing Python and add a reference to the existing dependencies:

```bash
~/.local/share/uv/tools/folivox/bin/python -m venv --without-pip experiments/.venv
printf '%s\n' "$HOME/.local/share/uv/tools/folivox/lib/python3.12/site-packages" \
  > experiments/.venv/lib/python3.12/site-packages/folivox-readonly.pth
UV_CACHE_DIR="$PWD/experiments/cache/uv" uv pip install \
  --python experiments/.venv/bin/python --no-deps gguf==0.19.0
```

2. Extract the downloaded tools and sources, then build:

```bash
mkdir -p experiments/toolchain
for task_deb in experiments/research/*.deb; do
  dpkg-deb -x "$task_deb" experiments/toolchain
done
tar -xzf experiments/research/cosyvoice-source.tar.gz -C experiments
tar -xzf experiments/research/ggml-source.tar.gz -C experiments
mkdir -p experiments/cosyvoice.cpp-67d45feb2f654bbc66cca25bff78464b3ec08fa1/vendor/pcre2
tar -xzf experiments/research/pcre2-source.tar.gz \
  --strip-components=1 \
  -C experiments/cosyvoice.cpp-67d45feb2f654bbc66cca25bff78464b3ec08fa1/vendor/pcre2
bash experiments/scripts/build-vulkan.sh
```

`build-vulkan.sh` sets tool/library search paths only for the build process. It selects the pinned GGML source and disables C++ frontend/audio/ICU dependencies, avoiding automatic runtime/model downloads. It builds the CLI and upstream backend test executable with six parallel jobs. It does not install anything globally. Existing GCC 13.3 and GNU Make are used.

3. Convert the existing weights without downloading another model:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  experiments/.venv/bin/python -B experiments/cosyvoice.cpp-67d45feb2f654bbc66cca25bff78464b3ec08fa1/convert_model_to_gguf.py \
  --yaml_config "$HOME/.local/share/folivox/models/Fun-CosyVoice3-0.5B/cosyvoice3.yaml" \
  --llm_model "$HOME/.local/share/folivox/models/Fun-CosyVoice3-0.5B/llm.rl.pt" \
  --gguf_model experiments/outputs/cosyvoice3-rl-f16.gguf \
  --ftype f16 --tag 2512
```

Conversion writes about 1.7 GiB locally and uses the upstream conversion script included in the pinned C++ source. No original model file is overwritten. It is mixed storage precision: upstream converts 2D weights to F16 and retains other tensors as FP32.

4. Rebuild and run the direct, small GPU correctness check:

```bash
g++ -O2 -std=c++17 experiments/scripts/vulkan_probe.cpp \
  -I experiments/ggml-3612c0af93f2cfc8f048295300a61187723b12ae/include \
  -L experiments/build/lib -Wl,-rpath,'$ORIGIN/../build/lib' \
  -lggml -lggml-base -lggml-vulkan -o experiments/outputs/vulkan-probe
MESA_SHADER_CACHE_DIR="$PWD/experiments/cache/mesa" \
  VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.json \
  timeout 60 experiments/outputs/vulkan-probe
```

Hardware commands need access to the host `/dev/dri/renderD128`. Codex's restricted sandbox hides it; the successful runs used approved host execution, without `sudo`. The correct ICD filename on this machine is `radeon_icd.json`.


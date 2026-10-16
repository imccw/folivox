#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$task_root/toolchain/usr/bin:$PATH"
export LD_LIBRARY_PATH="$task_root/toolchain/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cmake -S "$task_root/cosyvoice.cpp-67d45feb2f654bbc66cca25bff78464b3ec08fa1" \
  -B "$task_root/build" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF \
  -DCOSYVOICE_NO_AUDIO=ON -DCOSYVOICE_NO_FRONTEND=ON -DCOSYVOICE_NO_ICU=ON \
  -DCOSYVOICE_SERVER_NO_WEBUI=ON \
  -DGGML_SOURCE_DIR="$task_root/ggml-3612c0af93f2cfc8f048295300a61187723b12ae" \
  -DGGML_VULKAN=ON -DGGML_BUILD_TESTS=ON -DGGML_BUILD_EXAMPLES=OFF \
  -DVulkan_INCLUDE_DIR="$task_root/toolchain/usr/include" \
  -DVulkan_LIBRARY=/usr/lib/x86_64-linux-gnu/libvulkan.so.1 \
  -DVulkan_GLSLC_EXECUTABLE="$task_root/toolchain/usr/bin/glslc" \
  -DSPIRV-Headers_DIR="$task_root/toolchain/usr/share/cmake/SPIRV-Headers"
cmake --build "$task_root/build" --target cosyvoice-cli test-backend-ops -j 6

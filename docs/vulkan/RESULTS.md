# Folivox Radeon 780M acceleration — 15 October 2026

Vulkan acceleration works on this machine. An isolated `cosyvoice.cpp` build ran real FP32/FP16 matrix multiplication and convolution on the Radeon 780M, synthesized intelligible Chinese speech with the existing RL model weights and reference voice, and completed a Folivox render to M4B.

The working CPU installation, model files, system packages, drivers, kernel, and global environment were left unchanged. Experimental files are in this project, without a global command shortcut in this fork. No ROCm/PyTorch GPU wheels or replacement models were downloaded.

## Measured results

Short benchmark text: `你好，这是语音速度测试。` Reference voice: existing `chinese-sample-ccw-001`. Seed argument: 0. All runs used the existing RL LLM weights. CPU measurements ran separately from GPU measurements and compilation.

| Implementation | Threads | Synthesis time | Audio duration | Real-time factor |
|---|---:|---:|---:|---:|
| Original Folivox / PyTorch FP32 | 4 | 35.74 s | 4.40 s | 8.12 |
| Original Folivox / PyTorch FP32 | 8 | 27.54 s | 4.40 s | 6.26 |
| Original Folivox / PyTorch FP32 | 16 | 59.82 s | 4.40 s | 13.60 |
| C++ CPU / converted F16 | 8 | 26.50 s | 4.24 s | 6.25 |
| C++ Vulkan / converted F16 | 8 | **5.98 s** | 4.00 s | **1.49** |

Lower real-time factor is faster; 1.0 means one second of computation per second of audio. Vulkan was approximately **4.2× faster per second of audio** than either CPU implementation at 8 threads. Including backend/model loading, the C++ runs took 27.58 s on CPU and 6.74 s on Vulkan.

These are single short measurements, not a broad performance guarantee. PyTorch and GGML differ in precision, sampling, attention/cache implementation, and output duration. The thread sweep decoded the **same saved speech tokens** at each thread count, and those CPU waveforms were identical. It ran 4, 8, then 16 threads in one loaded process; it was not a randomized repeated sweep. The C++ CPU and GPU tests used the same converted file and options, but generated different speech tokens/waveforms. The claim above normalizes for duration rather than assuming identical outputs.

A separate full Folivox integration test rendered 37 Chinese characters to 11.81 s of speech. The C++ synthesis took 11.59 s (RTF 0.98), with 12.34 s total C++ process time. The first Folivox render took about 21 s including Python voice-feature extraction/imports, synthesis, assembly, and AAC encoding. The resulting M4B is 14.61 s including Folivox pauses. Voice features are cached for subsequent calls; the C++ model reloads for each fragment.

## Saved evidence and limits

[CPU measurements](evidence/cpu-benchmark.json),
[C++ measurements](evidence/cpp-benchmark.json), and
[direct GPU probe output](evidence/vulkan-probe.txt) are included. Filenames in
those records refer to original local artifacts; audio and full logs are omitted.
These are historical measurements from the original workspace, not new tests
of this clone.

The direct probe rejects CPU devices and requires a Radeon 780M. It executes
FP32/F16 matrix multiplication and 1D convolution, synchronizes, and compares
every value against scalar CPU calculations. All four comparisons passed with
zero maximum error for deliberately small, exactly representable inputs. Every
node in those small graphs must be supported by Vulkan, with no CPU fallback.

Full model synthesis is hybrid: unsupported operations and CPU DSP can run on
CPU, and ONNX voice preparation stays on CPU. Selecting RADV excludes llvmpipe.
The small probe does not establish that all full-model operations use the GPU.

Audio was finite and nonzero. Cached Whisper recovered the intended short and
longer Chinese content. This checks content, not perceptual quality, speaker
similarity, every language, or long-form stability. Voice data and generated
speech are omitted from this source snapshot.

The adapter/probe require a 780M and select AMD RADV. Other Vulkan-capable GPUs
are candidates for future testing. This work does not establish other iGPU
support, Windows support, or ROCm/PyTorch GPU inference.

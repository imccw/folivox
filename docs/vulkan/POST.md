# Suggested upstream issue

Title: Experimental Vulkan backend working on Ryzen 7 8840HS / Radeon 780M

I've tested an experimental Folivox integration using cosyvoice.cpp and GGML's
Vulkan backend on a Ryzen 7 8840HS with Radeon 780M, running Linux Mint.

It converts existing CosyVoice3 RL weights to GGUF and reuses Folivox's voice
features. CLI synthesis and a browser-initiated M4B render completed.

In one short Chinese benchmark, the original Folivox CPU path took 27.54 seconds
for 4.40 seconds of audio; Vulkan took 5.98 seconds for 4.00 seconds of audio:
approximately 4.2× faster after normalizing by duration. These are single short
measurements with different outputs, not a general speed claim.

Inference uses both GPU and CPU operations; the ONNX voice frontend stays on
CPU. Other iGPUs, broad language coverage, and long-form stability are untested.
The adapter currently selects RADV and checks for a 780M.

Code, pinned build recipe, and results:
https://github.com/imccw/folivox/tree/experimental-vulkan

The recipe documents the original working experiment; a fresh clone has not yet
been rebuilt and benchmarked independently. Models and personal voice data are
excluded.

Thanks to Folivox, Lourdle/cosyvoice.cpp, GGML, and FunAudioLLM/CosyVoice. My
additions connect Vulkan inference to Folivox and document the Radeon 780M tests.

Would an optional Vulkan provider be useful upstream? I'd welcome guidance on
integration and testers with other GPUs.

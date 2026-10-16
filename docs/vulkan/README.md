# Experimental Radeon 780M Vulkan integration

This branch adds a separate `./folivox-vulkan` launcher connecting Folivox to
Lourdle's cosyvoice.cpp / GGML Vulkan backend. It registers `cosyvoice3-vulkan`
in its own process and reads locally converted CosyVoice3 RL weights. Existing
Folivox Python components extract the voice features.

CLI synthesis, a Folivox M4B render, and a spawned browser render worked on
Ryzen 7 8840HS / Radeon 780M, Linux Mint 22.1 (Ubuntu 24.04 base), kernel
6.8.0-79, Mesa RADV 26.2.4, and 32 GB shared RAM. See [results](RESULTS.md).

## Scope and limits

- The adapter/probe require a device description containing `780M`. The adapter
  selects `Vulkan0` and `/usr/share/vulkan/icd.d/radeon_icd.json`. Other GPUs need
  device/driver selection changes and testing.
- Only RL weights and a Chinese reference voice were tested. The provider's
  advertised language list does not establish their quality.
- Full inference is hybrid: some operations and CPU DSP remain on CPU. The ONNX
  voice frontend also stays on CPU. The C++ frontend/audio/ICU are disabled.
- The adapter hooks private Folivox pipeline/UI functions in memory, including
  registration in spawned workers. Later Folivox versions are untested.
- The included UI file also adds raw-text input to the book drop area.
- Models reload for each fragment. Long books, speaker similarity, broad
  language coverage, and repeated performance measurements are unvalidated.
- The separate Python environment references existing Folivox dependencies
  read-only. Keep that installation in place.

## Setup and use

Follow [the reconstruction recipe](SETUP.md). This is a source snapshot of a
working local experiment; a fresh clone has not yet been rebuilt and
benchmarked independently. Models, recordings, caches, downloads, and native
binaries are excluded from Git.

Once the setup is complete, run from the repository root:

```bash
./folivox-vulkan providers
./folivox-vulkan ui
```

Select **On this computer (Vulkan GPU)**, keep device **Auto**, and choose a
voice or supply a reference recording/transcript. The port defaults to 8766;
explicit `--port` overrides it. Press Ctrl+C to stop. Listen to a short sample
before rendering more text.

The CLI accepts normal `render` arguments and defaults to `cosyvoice3-vulkan`.
It forwards `--threads` and `--model-dir`. The GGUF filename remains fixed at
`experiments/outputs/cosyvoice3-rl-f16.gguf`; `--model-dir` selects tokenizer and
voice inputs, not another GGUF checkpoint. The launcher sets offline flags in
its process and does not download models or update Folivox.

To return to the original engine, use the original `folivox` command. This
checkout does not install a global shortcut or replace the installed package.

## Credits

- [Folivox / vatavale](https://github.com/vatavale/folivox): audiobook pipeline,
  voice frontend, CLI, and UI.
- [cosyvoice.cpp / Lourdle](https://github.com/Lourdle/cosyvoice.cpp): independent
  community native CosyVoice implementation and GGUF conversion.
- [GGML](https://github.com/ggml-org/ggml): compute framework and Vulkan backend.
- [PCRE2](https://github.com/PCRE2Project/pcre2): native tokenizer dependency.
- [FunAudioLLM / CosyVoice](https://github.com/FunAudioLLM/CosyVoice): original
  model and implementation.

The additions here connect these components to Folivox and document Radeon 780M
tests. Existing Folivox LICENSE/NOTICE are preserved; dependency licenses remain
in their pinned upstream sources.

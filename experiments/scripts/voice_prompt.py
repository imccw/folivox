"""Extract voice features on CPU without loading the LLM, flow or vocoder."""
from pathlib import Path
from functools import partial
import argparse
import os
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(__file__).resolve().parents[1]/"cache/numba"))
import torch
import numpy as np
import gguf
from folivox.vendor.cosyvoice.cli.frontend import CosyVoiceFrontEnd
from folivox.vendor.cosyvoice.tokenizer.tokenizer import get_qwen_tokenizer
from folivox.vendor.matcha.utils.audio import mel_spectrogram
from folivox.tts.engine import SYSTEM_PROMPT
from folivox.tts.voice import load_voice

p=argparse.ArgumentParser()
p.add_argument("--audio",required=True)
p.add_argument("--transcript-file",required=True)
p.add_argument("--model-dir",required=True)
p.add_argument("--output",required=True)
args=p.parse_args()
torch.set_num_threads(8)
d=Path(args.model_dir)
voice=load_voice(args.audio,str(args.transcript_file))
frontend=CosyVoiceFrontEnd(
    partial(get_qwen_tokenizer,token_path=str(d/"CosyVoice-BlankEN"),skip_special_tokens=True,version="cosyvoice3"),
    partial(mel_spectrogram,n_fft=1920,num_mels=80,sampling_rate=24000,hop_size=480,win_size=1920,
            fmin=0,fmax=None,center=False),str(d/"campplus.onnx"),str(d/"speech_tokenizer_v3.onnx"))
with torch.inference_mode():
    data=frontend.frontend_zero_shot("",SYSTEM_PROMPT+voice.transcript,str(voice.wav),24000,"")
assert torch.equal(data["llm_prompt_speech_token"],data["flow_prompt_speech_token"])
assert torch.equal(data["llm_embedding"],data["flow_embedding"])
writer=gguf.GGUFWriter(args.output,"cosyvoice3-prompt")
writer.add_tensor("feat",data["prompt_speech_feat"][0].cpu().numpy().astype(np.float32))
writer.add_tensor("embedding",data["flow_embedding"].cpu().numpy().astype(np.float32))
writer.add_tensor("tokens",data["flow_prompt_speech_token"][0].cpu().numpy().astype(np.int32))
writer.add_tensor("text",np.frombuffer(voice.transcript.encode(),dtype=np.int8))
writer.write_header_to_file();writer.write_kv_data_to_file();writer.write_tensors_to_file();writer.close()

"""Process-local Folivox provider; never patches the installed package."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HUB_OFFLINE","1")
os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
os.environ.setdefault("NUMBA_CACHE_DIR",str(ROOT/"cache/numba"))
import numpy as np
import soundfile as sf
from folivox.tts.base import TTSEngine


class VulkanEngine(TTSEngine):
    name="cosyvoice3-vulkan"
    languages=("ru","en","zh","ja","ko","de","es","fr","it")
    sample_rate=24000

    def __init__(self,model_dir=None,device="auto",threads=None,llm="rl"):
        if llm!="rl":
            raise ValueError("This experiment contains the RL weights; use the original folivox for base weights.")
        if device not in ("auto","vulkan","cpu"):
            raise ValueError("The experimental provider accepts auto, vulkan, or cpu.")
        self.device="cpu" if device=="cpu" else "vulkan"
        self.backend="CPU" if self.device=="cpu" else "Vulkan0"
        self.threads=threads or 8
        self.model_dir=Path(model_dir).expanduser() if model_dir else Path.home()/".local/share/folivox/models/Fun-CosyVoice3-0.5B"
        self.gguf=ROOT/"outputs/cosyvoice3-rl-f16.gguf"
        self.cli=ROOT/"build/bin/cosyvoice-cli"
        self.voice_cache=ROOT/"cache/voices"
        self.voice_cache.mkdir(parents=True,exist_ok=True)
        self._lock=threading.Lock()
        self.env=dict(os.environ,MESA_SHADER_CACHE_DIR=str(ROOT/"cache/mesa"),
                      VK_DRIVER_FILES="/usr/share/vulkan/icd.d/radeon_icd.json")
        for p in [self.gguf,self.cli,self.model_dir/"campplus.onnx",self.model_dir/"speech_tokenizer_v3.onnx"]:
            if not p.is_file():
                raise FileNotFoundError(p)
        t0=time.perf_counter()
        if self.device=="vulkan":
            r=subprocess.run([str(self.cli),"--backend-path",str(ROOT/"build/lib"),"--list-backends"],
                             env=self.env,capture_output=True,text=True,timeout=30)
            if r.returncode or "Vulkan0" not in r.stdout or "780M" not in r.stdout:
                raise RuntimeError("Radeon 780M Vulkan backend unavailable: "+(r.stdout+r.stderr)[-1200:])
        self.load_seconds=time.perf_counter()-t0

    def prepare_voice(self,voice):
        dest=self.voice_cache/f"{voice.id}.gguf"
        with self._lock:
            if not dest.exists():
                with tempfile.TemporaryDirectory(dir=self.voice_cache) as tmp:
                    transcript=Path(tmp)/"transcript.txt"
                    transcript.write_text(voice.transcript)
                    prompt=Path(tmp)/"prompt.gguf"
                    cmd=[sys.executable,"-B",str(ROOT/"scripts/voice_prompt.py"),"--audio",str(voice.wav.resolve()),
                         "--transcript-file",str(transcript),"--model-dir",str(self.model_dir),"--output",str(prompt)]
                    r=subprocess.run(cmd,env=self.env,capture_output=True,text=True,timeout=120)
                    if r.returncode:
                        raise RuntimeError("Voice feature extraction failed: "+(r.stdout+r.stderr)[-2000:])
                    prompt.replace(dest)
        return dest

    def synthesize_audio(self,text,voice,seed,speed):
        start=time.perf_counter()
        prompt=self.prepare_voice(voice)
        with tempfile.TemporaryDirectory(dir=ROOT/"cache") as tmp:
            output=Path(tmp)/"audio.wav"
            cmd=[str(self.cli),"--backend-path",str(ROOT/"build/lib"),"--backend",self.backend,
                 "--model",str(self.gguf),"--prompt-speech",str(prompt),"--text",text,"--output",str(output),
                 "--threads",str(self.threads),"--seed",str(seed%2**32),"--speed",str(speed),
                 "--llm-kv-cache-type","f16","--verbose"]
            r=subprocess.run(cmd,env=self.env,capture_output=True,text=True,timeout=180+2*len(text))
            log=ROOT/"logs"/f"provider-{uuid.uuid4().hex}.log"
            log.write_text(r.stdout+"\n"+r.stderr)
            if r.returncode:
                raise RuntimeError(f"CosyVoice C++ exited {r.returncode}; see {log}: "+(r.stdout+r.stderr)[-1000:])
            audio,sr=sf.read(output,dtype="float32")
        if sr!=self.sample_rate or audio.ndim!=1 or not len(audio) or not np.isfinite(audio).all():
            raise RuntimeError("C++ backend returned invalid audio")
        return audio,time.perf_counter()-start


def register():
    from folivox.tts import ENGINES
    ENGINES.register("cosyvoice3-vulkan",VulkanEngine,"Experimental local CosyVoice3 RL via GGML Vulkan")


def configure():
    register()
    os.environ.setdefault("FOLIVOX_TTS","cosyvoice3-vulkan")
    # Folivox only forwards local model/thread options for its built-in provider.
    # Extend that dispatch in this process so the normal CLI options still work.
    from folivox import pipeline
    original=pipeline._engine_options
    def options(opts):
        if opts.tts=="cosyvoice3-vulkan":
            return dict(model_dir=opts.model_dir,device=opts.device,threads=opts.threads,llm=opts.llm)
        return original(opts)
    pipeline._engine_options=options


def render_job(book,opts,events):
    """Register the provider in the UI's separately spawned render process."""
    configure()
    from folivox.pipeline import render
    return render(book,opts,events)


def configure_ui():
    import json
    from folivox.ui import app as ui
    from starlette.responses import HTMLResponse, JSONResponse
    from starlette.routing import request_response
    original=ui.create_app
    def create_app(*args,**kwargs):
        app=original(*args,**kwargs)
        app.state.ui.manager._target="folivox_vulkan:render_job"
        for route in app.routes:
            if getattr(route,"path",None)=="/":
                endpoint=route.endpoint
                async def index(request,_endpoint=endpoint):
                    response=await _endpoint(request)
                    if response.status_code==200 and response.media_type=="text/html":
                        extension=(ROOT/"ui/raw-text.html").read_text(encoding="utf-8")
                        html=response.body.decode("utf-8").replace("</body>",extension+"\n</body>",1)
                        headers=dict(response.headers)
                        headers.pop("content-length",None)
                        return HTMLResponse(html,headers=headers)
                    return response
                route.endpoint=index
                route.app=request_response(index)
            if getattr(route,"path",None)=="/api/tts/engines":
                endpoint=route.endpoint
                async def engines(request,_endpoint=endpoint):
                    response=await _endpoint(request)
                    body=json.loads(response.body)
                    body["engines"].insert(0,dict(
                        name="cosyvoice3-vulkan",label="On this computer (Vulkan GPU)",
                        local=True,cloud=False,models=[],per_hour=0,tag="",
                        blurb="Uses the Radeon 780M GPU. Free, private, and offline. Clones your reference voice."))
                    return JSONResponse(body)
                route.endpoint=engines
                route.app=request_response(engines)
        return app
    ui.create_app=create_app


def main():
    configure()
    configure_ui()
    from folivox.cli import main as cli_main
    cli_main()


if __name__=="__main__":
    main()

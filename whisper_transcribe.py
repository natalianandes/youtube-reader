import sys, json, os
from faster_whisper import WhisperModel

# uso: python whisper_transcribe.py <arquivo_audio> [lang]
audio = sys.argv[1]
lang  = sys.argv[2] if len(sys.argv) > 2 else None  # "pt"/"en"/None

device_env  = (os.getenv("WHISPER_DEVICE") or "cpu").lower()

# Ordem de tentativas por device (evita ValueError)
if device_env == "cuda":
    compute_candidates = [os.getenv("WHISPER_COMPUTE") or "float16", "float32"]
else:
    device_env = "cpu"
    compute_candidates = [os.getenv("WHISPER_COMPUTE") or "int8", "float32"]

def try_build(device: str, comp: str):
    return WhisperModel(
        "small",                          # "tiny" = mais rápido; "small" = melhor qualidade
        device=device,                    # "cuda" se tiver GPU + CUDA
        compute_type=comp,                # "int8"/"float32" na CPU; "float16" na GPU
        cpu_threads=max(2, os.cpu_count() // 2),
    )

model = None
last_err = None
for comp in compute_candidates:
    try:
        model = try_build(device_env, comp)
        break
    except ValueError as e:
        last_err = e

if model is None:
    # fallback final, sempre funciona
    model = try_build("cpu", "float32")

segments, info = model.transcribe(
    audio,
    language=lang,                        # None = autodetect
    vad_filter=True,
    vad_parameters=dict(min_silence_duration_ms=500),
    beam_size=1, best_of=1,               # greedy (rápido)
    chunk_length=30,                      # ~30s por bloco
    no_speech_threshold=0.5,
)

text = "".join(seg.text for seg in segments).strip()
print(json.dumps({"duration": info.duration, "text": text}, ensure_ascii=False))

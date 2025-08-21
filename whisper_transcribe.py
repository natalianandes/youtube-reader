import sys, json
from faster_whisper import WhisperModel

audio = sys.argv[1]
lang = sys.argv[2] if len(sys.argv) > 2 else None  # "pt"/"en"/None

model = WhisperModel("tiny", device="cpu")  # troque para "small" depois
segments, info = model.transcribe(audio, language=lang)
text = "".join(seg.text for seg in segments).strip()

print(json.dumps({"duration": info.duration, "text": text}, ensure_ascii=False))

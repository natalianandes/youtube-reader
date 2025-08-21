from faster_whisper import WhisperModel

# use "tiny" p/ teste rápido; depois troque para "small" ou "medium"
model = WhisperModel("tiny", device="cpu")

# mude language="en" se o áudio estiver em inglês
segments, info = model.transcribe("test_audio.mp3", language="pt")

print("Duração do áudio (s):", info.duration)
for seg in segments:
    print(f"[{seg.start:.2f}s -> {seg.end:.2f}s] {seg.text}")

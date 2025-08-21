#!/usr/bin/env node
// usage: node index.mjs "<youtube_url>" [--detailed]
import { execSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import axios from "axios";

const OLLAMA_HOST = process.env.OLLAMA_HOST || "http://127.0.0.1:11434";
const MODEL = "llama3.2:3b";

const sh = (cmd) => execSync(cmd, { stdio: "pipe" }).toString().trim();

function chunkText(t, max = 4000) {
  const parts = [];
  for (let i = 0; i < t.length; i += max) parts.push(t.slice(i, i + max));
  return parts;
}

async function summarize(text, detailed = false) {
  const chunks = chunkText(text, 3500);
  const partials = [];
  for (const [i, c] of chunks.entries()) {
    const prompt = `${detailed ? "Resuma de forma explicativa (exemplos, tópicos e ideias-chave)" : "Resuma com tópicos claros"} o trecho ${i+1}/${chunks.length}. Mantenha fatos e nomes. Texto:\n\n${c}`;
    const { data } = await axios.post(`${OLLAMA_HOST}/api/chat`, {
      model: MODEL,
      messages: [{ role: "user", content: prompt }],
      stream: false,
    });
    partials.push(data.message.content);
  }
  if (partials.length === 1) return partials[0];

  const joinPrompt = `Combine os resumos abaixo num único resumo ${detailed ? "detalhado (1.5–3k palavras se houver conteúdo)" : "conciso"} com seções, bullets e conclusão, sem repetir ou inventar:\n\n${partials.join("\n\n---\n\n")}`;
  const { data } = await axios.post(`${OLLAMA_HOST}/api/chat`, {
    model: MODEL,
    messages: [{ role: "user", content: joinPrompt }],
    stream: false,
  });
  return data.message.content;
}

async function getTranscript(url) {
  // 1) tentar legendas
  try {
    const { YoutubeTranscript } = await import("youtube-transcript");
    const items = await YoutubeTranscript
      .fetchTranscript(url, { lang: "pt" })
      .catch(() => YoutubeTranscript.fetchTranscript(url));
    if (items?.length) {
      return items.map(i => i.text).join(" ").replace(/\s+/g, " ").trim();
    }
  } catch {}

  // 2) fallback: baixar áudio + whisper
  const out = "tmp_audio.mp3";
  console.log("· Baixando áudio com yt-dlp…");
  sh(`yt-dlp -x --audio-format mp3 "${url}" -o "${out}"`);

  console.log("· Transcrevendo com faster-whisper…");
  const raw = sh(`python whisper_transcribe.py "${out}"`);
  const { text } = JSON.parse(raw);
  fs.rmSync(out, { force: true });
  return text;
}

(async () => {
  try {
    const url = process.argv[2];
    const detailed = process.argv.includes("--detailed");
    if (!url) {
      console.error('uso: node index.mjs "<youtube_url>" [--detailed]');
      process.exit(1);
    }

    console.log("· Extraindo transcrição…");
    const transcript = await getTranscript(url);
    if (!transcript || transcript.length < 40) throw new Error("Transcrição vazia/curta");

    console.log(`· Transcrição ok (${transcript.length} chars). Resumindo com ${MODEL}…`);
    const summary = await summarize(transcript, detailed);

    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    const outPath = path.join(process.cwd(), `summary-${stamp}.md`);
    fs.writeFileSync(outPath, `# Resumo\n\n${summary}\n`);
    console.log(`✅ pronto: ${outPath}`);
  } catch (e) {
    console.error("❌ erro:", e.message);
    process.exit(1);
  }
})();

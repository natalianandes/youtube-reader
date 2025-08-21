#!/usr/bin/env node
// usage: node index.mjs "<youtube_url>" [--detailed]
import { execSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import axios from "axios";
import os from "node:os";
import pLimit from "p-limit";

// ===== Config =====
const OLLAMA_HOST = process.env.OLLAMA_HOST || "http://127.0.0.1:11434";
const MAP_MODEL = "llama3.2:1b";        // rápido (map)
const REDUCE_MODEL = "llama3.2:3b";     // qualidade (reduce)
const limit = pLimit(parseInt(process.env.MAP_CONCURRENCY || "3", 10));

const OLLAMA_OPTIONS = {
  temperature: 0.2,
  num_predict: 700,                     // aumente no reduce detalhado
  num_ctx: 2048,
  num_thread: Math.max(2, os.cpus().length),
  top_p: 0.9,
  repeat_penalty: 1.1,
};

// ===== Helpers =====
const sh = (cmd) => execSync(cmd, { stdio: "pipe" }).toString().trim();

function chunkText(t, max = 2500) {
  if (!t) return [];
  const parts = [];
  for (let i = 0; i < t.length; i += max) parts.push(t.slice(i, i + max));
  return parts;
}

async function ollamaChat(model, prompt, extra = {}) {
  const { data } = await axios.post(`${OLLAMA_HOST}/api/chat`, {
    model,
    messages: [{ role: "user", content: prompt }],
    stream: false,
    options: { ...OLLAMA_OPTIONS, ...extra },
  });
  return data.message?.content ?? "";
}

// ===== Summarization (map 1B -> reduce 3B) =====
async function summarizeMapReduce(text, detailed = false) {
  const chunks = chunkText(text, 2500);
  if (chunks.length === 0) return "";

  // MAP (1B em paralelo)
  const mapPrompt = (i, total, c) => `
Você resume fielmente (sem inventar) o trecho ${i}/${total}.
- bullets e subtítulos
- fatos, números, nomes, passos
- inclua (timestamp) se explícito
Trecho:
${c}`.trim();

  const tasks = chunks.map((c, i) =>
    limit(() => ollamaChat(MAP_MODEL, mapPrompt(i + 1, chunks.length, c)))
  );
  const partials = await Promise.all(tasks);

  if (partials.length === 1 && !detailed) return partials[0];

  // REDUCE (3B se detailed; 1B se conciso)
  const alvo = detailed ? "1500-3000" : "400-700";
  const reducePrompt = `
Una os resumos abaixo num documento coeso de ${alvo} palavras.
Seções: Introdução; Ideias-chave; Passo a passo; Exemplos; Limitações; Conclusão.
Remova repetições. Não invente fatos.

${partials.join("\n\n---\n\n")}`.trim();

  return await ollamaChat(
    detailed ? REDUCE_MODEL : MAP_MODEL,
    reducePrompt,
    { num_predict: detailed ? 1800 : 800 }
  );
}

// ===== Transcript (legendas -> fallback yt-dlp + whisper) =====
async function getTranscript(url) {
  // 1) tentar legendas (pt -> auto)
  try {
    const { YoutubeTranscript } = await import("youtube-transcript");
    const items = await YoutubeTranscript
      .fetchTranscript(url, { lang: "pt" })
      .catch(() => YoutubeTranscript.fetchTranscript(url));
    if (items?.length) {
      return items.map(i => i.text).join(" ").replace(/\s+/g, " ").trim();
    }
  } catch (_) {}

  // 2) fallback: baixar melhor áudio sem recodificar + whisper
  console.log("· Baixando áudio com yt-dlp…");
  sh(`yt-dlp -N 8 --no-playlist -f "bestaudio[ext=m4a]/bestaudio" -o "tmp_audio.%(ext)s" "${url}"`);

  const candidates = ["tmp_audio.m4a","tmp_audio.mp3","tmp_audio.webm","tmp_audio.opus","tmp_audio.wav","tmp_audio.mp4"];
  const audioPath = candidates.find(p => fs.existsSync(p));
  if (!audioPath) throw new Error("Áudio não encontrado (tmp_audio.*).");

  console.log(`· Transcrevendo com faster-whisper… (${path.basename(audioPath)})`);
  const raw = sh(`python whisper_transcribe.py "${audioPath}"`);
  const { text } = JSON.parse(raw);

  // limpeza opcional de temporários
  try { for (const f of candidates) fs.rmSync(f, { force: true }); } catch {}

  return (text || "").trim();
}

// ===== Main =====
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

    console.log(`· Transcrição ok (${transcript.length} chars). Resumindo (map=1b, reduce=${detailed ? "3b" : "1b"})…`);
    const summary = await summarizeMapReduce(transcript, detailed);

    const stamp = new Date().toISOString().replace(/[:.]/g, "-");
    const outPath = path.join(process.cwd(), `summary-${stamp}.md`);
    fs.writeFileSync(outPath, `# Resumo\n\n${summary}\n`);
    console.log(`✅ pronto: ${outPath}`);
  } catch (e) {
    console.error("❌ erro:", e?.message || e);
    process.exit(1);
  }
})();

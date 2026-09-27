#!/usr/bin/env node
// CLI-обёртка для сервера yue2-web: MIDI → ABC (marmooo/midi2abc).
//
// Вход: путь к .mid в argv[2] или сырые байты в stdin. Заголовок T: берётся из
// MIDI2ABC_TITLE (имя файла без расширения от сервера).
// Выход: ABC после строки-маркера <<<ABC>>> — до маркера в stdout могут быть
// предупреждения конвертера (console.log внутри midi2abc.js про нелегальные
// длительности), поэтому парсить надо по маркеру, а не весь stdout.
import { readFileSync } from "node:fs";
import * as midiFilePkg from "midi-file";
import { buildNoteSequence, autoQuantizeTicks } from "./src/parse-midi-core.js";
import notesToAbc from "./src/midi2abc.js";

const parseMidi = midiFilePkg.parseMidi || midiFilePkg.default?.parseMidi;

let data;
let title = process.env.MIDI2ABC_TITLE || null;
if (process.argv[2]) {
  data = readFileSync(process.argv[2]);
  if (!title) {
    title = process.argv[2].split("/").pop().replace(/\.(mid|midi)$/i, "");
  }
} else {
  const chunks = [];
  for await (const chunk of process.stdin) chunks.push(chunk);
  data = Buffer.concat(chunks);
}

const ns = buildNoteSequence(parseMidi(new Uint8Array(data)));
autoQuantizeTicks(ns, 8); // сетка 32-х долей — как в браузерной версии parse-midi.js
const options = title ? { title: title.replace(/[\r\n]+/g, " ") } : undefined;
const abc = notesToAbc(ns, options);
if (!abc.trim()) {
  console.error("no notes found in MIDI");
  process.exit(3);
}
process.stdout.write("<<<ABC>>>\n" + abc + "\n");

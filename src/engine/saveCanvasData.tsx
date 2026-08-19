// Persists and restores canvas stroke data as local files via Tauri's fs/dialog
// plugins. A custom JSON extension is used (rather than .svg/.png) because it
// preserves the raw stroke points/pressure needed to reload and re-analyze the
// canvas, not just a flattened image.

import { save, open } from "@tauri-apps/plugin-dialog";
import { writeTextFile, readTextFile, mkdir, readDir } from "@tauri-apps/plugin-fs";
import { join } from "@tauri-apps/api/path";

const FILE_EXTENSION = "canvasnote";
// Notes default into one folder so there is a fixed set to compare a canvas
// against, rather than notes scattered wherever the dialog last pointed.
const SAVES_DIR = __SAVES_DIR__;
// Stamped into saved files so future format changes can be migrated on load.
// Not yet read/checked anywhere, bump this and branch in loadCanvasData
// when the schema actually changes.
const FORMAT_VERSION = 2;

// One embedding vector plus the model that produced it. Vectors are unit
// length and only comparable to vectors from the same model.
export interface EmbeddingRecord {
  model: string;
  dim: number;
  vector: number[];
}

// Transcription + embeddings from ML service, null if service is unreachable
export interface Analysis {
  analyzedAt: string; // timestamp
  text: string;
  embeddings: {
    text: EmbeddingRecord | null; // null when no text was recognized (pure doodle)
    image: EmbeddingRecord;
    strokes: EmbeddingRecord | null; // null until a stroke encoder is exported
  };
}

interface CanvasData {
  version: number;
  strokes: number[][][];
  analysis: Analysis | null;
}

// Formats an object with one key per row, each value compact on one line.
// indent is the indentation of the line the object starts on.
function formatObjectPerKey(obj: object, indent: string): string {
  const fields = Object.entries(obj).map(
    ([key, value]) => `${indent}  ${JSON.stringify(key)}: ${JSON.stringify(value)}`,
  );
  return "{\n" + fields.join(",\n") + "\n" + indent + "}";
}

// Analysis fields each get their own row too, and so do the entries inside
// embeddings (text, image, eventually strokes), but each vector stays
// compact on one line, since full pretty-printing would put every number on
// its own row.
function formatAnalysis(analysis: Analysis | null): string {
  if (analysis === null) return "null";
  const fields = Object.entries(analysis).map(([key, value]) => {
    const formatted =
      key === "embeddings"
        ? formatObjectPerKey(value, "    ")
        : JSON.stringify(value);
    return `    ${JSON.stringify(key)}: ${formatted}`;
  });
  return "{\n" + fields.join(",\n") + "\n  }";
}

// Writes a .canvasnote file at the given path; format it nicely so the strokes stay on one line,
// but so that each header gets its own row
export async function writeCanvasFile(
  path: string,
  strokes: number[][][],
  analysis: Analysis | null
): Promise<void> {
  const json = [
    "{",
    `  "version": ${FORMAT_VERSION},`,
    `  "strokes": ${JSON.stringify(strokes)},`,
    `  "analysis": ${formatAnalysis(analysis)}`,
    "}",
  ].join("\n");
  await writeTextFile(path, json);
}

// Creates the saves folder if it is missing and returns it. A recursive mkdir
// over an existing directory is a no-op, so this is cheap to call before every
// save and needs no separate existence check.
export async function ensureSavesDir(): Promise<string> {
  await mkdir(SAVES_DIR, { recursive: true });
  return SAVES_DIR;
}

// Returns the saved file path, or null if the user cancelled the dialog.
export async function saveCanvasData(
  strokes: number[][][],
): Promise<string | null> {
  const path = await save({
    defaultPath: await join(await ensureSavesDir(), `canvas.${FILE_EXTENSION}`),
    filters: [{ name: "Canvas Note", extensions: [FILE_EXTENSION] }],
  });
  if (!path) return null;

  await writeCanvasFile(path, strokes, null);
  return path;
}

// Returns the loaded strokes and the file they came from, or null if the user
// cancelled the dialog. The path matters to the caller: a note on the canvas
// should not turn up as its own nearest neighbour.
export async function loadCanvasData(): Promise<{
  strokes: number[][][];
  path: string;
} | null> {
  const path = await open({
    multiple: false,
    defaultPath: await ensureSavesDir(),
    filters: [{ name: "Canvas Note", extensions: [FILE_EXTENSION] }],
  });
  if (!path) return null;

  const raw = await readTextFile(path);
  const data: CanvasData = JSON.parse(raw);
  return { strokes: data.strokes, path };
}

// A saved note with embeddings, ready to compare a canvas against. Strokes are
// left out: comparison only needs the vectors, and loading every note's points
// would be a lot of data to hold for nothing.
export interface SavedNote {
  name: string; // file name without extension, for display
  path: string;
  analysis: Analysis;
}

// Reads the saves folder into a comparable corpus. Notes are skipped rather
// than treated as errors: v1 files predate analysis entirely, some v2 files
// were written before embeddings existed, and a note saved while the ML service
// was down has analysis: null. The count of skipped files comes back so the UI
// can say how much of the folder it actually compared against, since "no
// similar notes" and "nothing in the folder is comparable" look identical
// otherwise.
export async function readSavedNotes(): Promise<{
  notes: SavedNote[];
  skipped: number;
}> {
  const dir = await ensureSavesDir();
  const suffix = `.${FILE_EXTENSION}`;
  const notes: SavedNote[] = [];
  let skipped = 0;

  for (const entry of await readDir(dir)) {
    if (!entry.isFile || !entry.name.endsWith(suffix)) continue;
    const path = await join(dir, entry.name);
    try {
      const data: CanvasData = JSON.parse(await readTextFile(path));
      // embeddings is typed as required, but files written by older versions of
      // the app really do lack it, so this guard is not redundant.
      if (!data.analysis?.embeddings) {
        skipped++;
        continue;
      }
      notes.push({
        name: entry.name.slice(0, -suffix.length),
        path,
        analysis: data.analysis,
      });
    } catch {
      skipped++; // unparseable or half-written, not worth failing the search
    }
  }

  return { notes, skipped };
}

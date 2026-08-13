// Persists and restores canvas stroke data as local files via Tauri's fs/dialog
// plugins. A custom JSON extension is used (rather than .svg/.png) because it
// preserves the raw stroke points/pressure needed to reload and re-analyze the
// canvas, not just a flattened image.

import { save, open } from "@tauri-apps/plugin-dialog";
import { writeTextFile, readTextFile } from "@tauri-apps/plugin-fs";

const FILE_EXTENSION = "canvasnote";
// Stamped into saved files so future format changes can be migrated on load.
// Not yet read/checked anywhere — bump this and branch in loadCanvasData
// when the schema actually changes.
const FORMAT_VERSION = 2;

// Transcriptions from ML service, null if service is unreachable
export interface Analysis {
  analyzedAt: string; // timestamp
  text: string;
}

interface CanvasData {
  version: number;
  strokes: number[][][];
  analysis: Analysis | null;
}

// Writes a .canvasnote file at the given path; format it nicely so the strokes stay on one line,
// but so that each header gets its own row
export async function writeCanvasFile(
  path: string,
  strokes: number[][][],
  analysis: Analysis | null
): Promise<void> {
  const analysisJson = JSON.stringify(analysis, null, 2).replace(/\n/g, "\n ");
  const json = [
    "{",
    `  "version": ${FORMAT_VERSION},`,
    `  "strokes": ${JSON.stringify(strokes)},`,
    `  "analysis": ${analysisJson}`,
    "}",
  ].join("\n");
  await writeTextFile(path, json);
}

// Returns the saved file path, or null if the user cancelled the dialog.
export async function saveCanvasData(
  strokes: number[][][],
): Promise<string | null> {
  const path = await save({
    defaultPath: `canvas.${FILE_EXTENSION}`,
    filters: [{ name: "Canvas Note", extensions: [FILE_EXTENSION] }],
  });
  if (!path) return null;

  await writeCanvasFile(path, strokes, null);
  return path;
}

// Returns the loaded strokes, or null if the user cancelled the dialog.
export async function loadCanvasData(): Promise<number[][][] | null> {
  const path = await open({
    multiple: false,
    filters: [{ name: "Canvas Note", extensions: [FILE_EXTENSION] }],
  });
  if (!path) return null;

  const raw = await readTextFile(path);
  const data: CanvasData = JSON.parse(raw);
  return data.strokes;
}

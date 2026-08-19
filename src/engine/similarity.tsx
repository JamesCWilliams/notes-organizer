// Ranks saved notes against the current canvas by embedding similarity.
//
// Brute force on purpose: every note is scored on every search. A dot product
// over a few hundred floats is nothing next to the model round trip that
// produced the query vector, and a personal note library is small. If it ever
// stops being small, this module is the only thing that has to change, since it
// touches no files and no React, and the ML service knows nothing about it.
//
// The three modalities are ranked separately rather than blended into one
// score. They disagree, and that disagreement is the interesting part: the
// stroke encoder exists to catch what the page image and the transcription
// miss, which a weighted average would hide.
//
// Scores are centered where the corpus is big enough to support it, see
// CENTERING_FLOOR. Raw cosines in this domain are close to useless for ranking:
// every note is handwriting on a white page, so an image encoder trained on
// natural images returns near-collinear vectors and real similarity hides in
// the third decimal place. Subtracting the corpus mean removes that shared
// component and spreads the remainder out.

import type { Analysis, EmbeddingRecord, SavedNote } from "./saveCanvasData";

export type Modality = "text" | "image" | "strokes";

export const MODALITIES: Modality[] = ["text", "image", "strokes"];

// Centering on N vectors forces a -1/(N-1) bias into the scores, since the
// centered corpus sums to zero. At N=1 the centered vector is zero and cosine
// is undefined; at N=2 the two notes come out exactly antipodal. Below this
// floor the bias dominates whatever signal centering recovers, so raw cosine is
// the lesser evil.
const CENTERING_FLOOR = 4;

export interface Neighbor {
  name: string;
  path: string;
  // Cosine similarity, -1..1. When the ranking is centered, this is the cosine
  // of the mean-subtracted vectors: 0 means "as similar to the canvas as the
  // average note in the library", so only the sign and the ordering carry
  // meaning, not the magnitude on its own.
  score: number;
}

export interface ModalityRanking {
  model: string; // the query's model, so the UI can show what did the comparing
  comparable: number; // notes carrying a vector from that same model
  incomparable: number; // notes lacking one, or holding a different model's
  // True when there are too few comparable notes for top and bottom to be
  // distinct lists. Worth surfacing: otherwise a half-empty library looks like
  // a broken search, with the same notes appearing as both most and least
  // similar.
  overlapping: boolean;
  // False when the corpus was too small to center against, see CENTERING_FLOOR.
  // Scores from a centered and an uncentered ranking are not on the same scale,
  // so the UI should not present them as if they were.
  centered: boolean;
  top: Neighbor[]; // most similar first
  bottom: Neighbor[]; // least similar first
}

// Dot product, which is the cosine for unit-length vectors. Every input here is
// unit length: stored vectors by the EmbeddingRecord contract, centered ones
// because normalize() puts them back on the unit sphere.
function dot(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

function meanVector(vectors: number[][]): number[] {
  const mean = new Array<number>(vectors[0].length).fill(0);
  for (const v of vectors) {
    for (let i = 0; i < v.length; i++) mean[i] += v[i];
  }
  for (let i = 0; i < mean.length; i++) mean[i] /= vectors.length;
  return mean;
}

// Subtracts the mean and rescales to unit length, so the result is still
// comparable with a plain dot product. Returns null for a vector that sits at
// the mean: it has no direction left to compare, and dividing by its length
// would produce NaNs that sort unpredictably.
function centerAndNormalize(v: number[], mean: number[]): number[] | null {
  const centered = v.map((x, i) => x - mean[i]);
  const norm = Math.sqrt(dot(centered, centered));
  if (norm < 1e-9) return null;
  return centered.map((x) => x / norm);
}

// Two vectors are only comparable when the same model produced them. Comparing
// across models, or across versions of one model, yields plausible-looking
// numbers that mean nothing, so the dim check is a backstop for the case where
// a retrained encoder keeps its name but changes shape.
function comparableTo(query: EmbeddingRecord, other: EmbeddingRecord): boolean {
  return other.model === query.model && other.dim === query.dim;
}

// Ranks one modality. Returns null when the canvas itself has no vector for it:
// a pure doodle gets no text embedding, and strokes are absent until an encoder
// has been exported.
function rankModality(
  modality: Modality,
  query: EmbeddingRecord | null,
  corpus: SavedNote[],
  k: number,
): ModalityRanking | null {
  if (!query) return null;

  // Gather the comparable notes before scoring any of them: centering needs the
  // whole set to take a mean over.
  const matches: { note: SavedNote; vector: number[] }[] = [];
  let incomparable = 0;
  for (const note of corpus) {
    const vector = note.analysis.embeddings[modality];
    if (!vector || !comparableTo(query, vector)) {
      incomparable++;
      continue;
    }
    matches.push({ note, vector: vector.vector });
  }

  // The mean comes from the corpus alone, leaving the query out. That makes the
  // centered corpus sum to zero, so the average score across the library is
  // exactly 0 and a score reads as more or less similar than average.
  let centered = false;
  let queryVector = query.vector;
  let vectors = matches.map((m) => m.vector);
  if (matches.length >= CENTERING_FLOOR) {
    const mean = meanVector(vectors);
    const centeredQuery = centerAndNormalize(query.vector, mean);
    const centeredCorpus = vectors.map((v) => centerAndNormalize(v, mean));
    // A null anywhere means some vector sat exactly at the mean, which leaves
    // nothing to compare it by. Rare enough to not be worth a partial ranking,
    // so the whole modality falls back to raw cosine.
    if (centeredQuery && centeredCorpus.every((v) => v !== null)) {
      centered = true;
      queryVector = centeredQuery;
      vectors = centeredCorpus as number[][];
    }
  }

  const scored: Neighbor[] = matches.map((m, i) => ({
    name: m.note.name,
    path: m.note.path,
    score: dot(queryVector, vectors[i]),
  }));
  scored.sort((a, b) => b.score - a.score);

  return {
    model: query.model,
    comparable: scored.length,
    incomparable,
    // With exactly 2k notes the two lists partition the corpus; below that they
    // share entries.
    overlapping: scored.length < 2 * k,
    centered,
    top: scored.slice(0, k),
    bottom: scored.slice(-k).reverse(),
  };
}

// Ranks a canvas against the library, one ranking per modality. A modality is
// null when the canvas has no vector for it.
export function rank(
  query: Analysis,
  corpus: SavedNote[],
  k = 3,
): Record<Modality, ModalityRanking | null> {
  return {
    text: rankModality("text", query.embeddings.text, corpus, k),
    image: rankModality("image", query.embeddings.image, corpus, k),
    strokes: rankModality("strokes", query.embeddings.strokes, corpus, k),
  };
}

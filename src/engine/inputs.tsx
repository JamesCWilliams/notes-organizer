import { getStroke } from "perfect-freehand";
import { getSvgPathFromStroke } from "./svgpathfromstroke";
import { useRef, useState } from "react";
import {
  saveCanvasData,
  loadCanvasData,
  writeCanvasFile,
  readSavedNotes,
  Analysis,
} from "./saveCanvasData";
import { rank, MODALITIES, Modality, ModalityRanking } from "./similarity";


const ML_SERVICE_BASE = "http://localhost:5000";

const STROKE_OPTS = {
  size: 16,
  thinning: 0.5,
  smoothing: 0.5,
  streamline: 0.5,
};

async function svgElementToPngDataUrl(svg: SVGSVGElement): Promise<string> {
  const { width, height } = svg.getBoundingClientRect();
  // Clone so we can add explicit dimensions without mutating the live element.
  const clone = svg.cloneNode(true) as SVGSVGElement;
  clone.setAttribute("width", String(width));
  clone.setAttribute("height", String(height));
  const svgData = new XMLSerializer().serializeToString(clone);
  const blob = new Blob([svgData], { type: "image/svg+xml" });
  const url = URL.createObjectURL(blob);

  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d")!;
      ctx.fillStyle = "white";
      ctx.fillRect(0, 0, width, height);
      ctx.drawImage(img, 0, 0);
      URL.revokeObjectURL(url);
      resolve(canvas.toDataURL("image/png"));
    };
    img.onerror = reject;
    img.src = url;
  });
}
// Renders the canvas SVG to a PNG and posts it with the strokes to the ML
// service for transcription and embedding. Shared by Save, which stores the
// result in the file, and Analyze, which ranks the library against it. Throws if
// the service is unreachable or responds with an error status.
async function analyzeCanvas(
  svg: SVGSVGElement,
  strokes: number[][][],
): Promise<Analysis> {
  const image = await svgElementToPngDataUrl(svg);
  const res = await fetch(`${ML_SERVICE_BASE}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image, strokes }),
  });
  if (!res.ok) throw new Error(`ML service returned ${res.status}`);
  const data = await res.json();
  return {
    analyzedAt: new Date().toISOString(),
    text: data.text,
    embeddings: data.embeddings,
  };
}

// What one Analyze produces: the transcription, plus one ranking of the library
// per modality.
interface AnalyzeResult {
  text: string;
  corpusSize: number;
  skipped: number; // notes in the folder with no embeddings to compare
  rankings: Record<Modality, ModalityRanking | null>;
}

// Why a modality has no ranking at all. Distinct from having a ranking with no
// comparable notes: here the canvas itself produced no vector.
const NO_VECTOR: Record<Modality, string> = {
  text: "no text recognized on the canvas",
  image: "no image embedding returned",
  strokes: "no stroke embedding (encoder unavailable)",
};

function RankingSection({
  modality,
  ranking,
}: {
  modality: Modality;
  ranking: ModalityRanking | null;
}) {
  if (!ranking) {
    return (
      <div style={{ marginTop: 10 }}>
        <strong>{modality}</strong>
        <div style={{ color: "#888" }}>{NO_VECTOR[modality]}</div>
      </div>
    );
  }

  const list = (label: string, neighbors: typeof ranking.top) => (
    <div>
      <div style={{ color: "#666" }}>{label}</div>
      {neighbors.map((n) => (
        <div key={n.path} style={{ display: "flex", gap: 8 }}>
          <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis" }}>
            {n.name}
          </span>
          <span style={{ fontVariantNumeric: "tabular-nums" }}>
            {n.score >= 0 ? "+" : ""}
            {n.score.toFixed(3)}
          </span>
        </div>
      ))}
    </div>
  );

  return (
    <div style={{ marginTop: 10 }}>
      <strong>{modality}</strong>
      <div style={{ color: "#888", fontSize: "0.85em" }}>
        {ranking.model} · {ranking.comparable} comparable
        {ranking.incomparable > 0 && `, ${ranking.incomparable} without a vector`}
        {!ranking.centered && " · uncentered"}
      </div>
      {ranking.comparable === 0 ? (
        <div style={{ color: "#888" }}>nothing to compare against</div>
      ) : ranking.overlapping ? (
        // Too few notes for "most" and "least" to be different lists, so show
        // the one ranking rather than printing the same names twice. top and
        // bottom together cover every comparable note in this case, but they
        // run in opposite directions, so the merge has to be re-sorted.
        list(
          "most to least similar",
          [...ranking.top, ...ranking.bottom]
            .filter((n, i, all) => all.findIndex((m) => m.path === n.path) === i)
            .sort((a, b) => b.score - a.score),
        )
      ) : (
        <>
          {list("most similar", ranking.top)}
          {list("least similar", ranking.bottom)}
        </>
      )}
    </div>
  );
}

//This is where the engine will gather user inputs from mouse clicks/stylus.
export function Draw() {
  const [currentStroke, setCurrentStroke] = useState<number[][]>([]);
  const [completedStrokes, setCompletedStrokes] = useState<number[][][]>([]);
  ("TO DO: Include this data with the PNG image.");
  const [result, setResult] = useState<AnalyzeResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  // Path of the note currently on the canvas, if it came from a file. Kept so
  // Analyze can leave it out of its own results.
  const [loadedPath, setLoadedPath] = useState<string | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  function handlePointerDown(e: React.PointerEvent<SVGSVGElement>) {
    e.currentTarget.setPointerCapture(e.pointerId);
    setCurrentStroke([[e.pageX, e.pageY, e.pressure]]);
  }

  function handlePointerMove(e: React.PointerEvent<SVGSVGElement>) {
    if (e.buttons !== 1) return;
    setCurrentStroke([...currentStroke, [e.pageX, e.pageY, e.pressure]]);
  }

  function handlePointerUp() {
    setCompletedStrokes([...completedStrokes, currentStroke]);
    setCurrentStroke([]);
  }

  async function handleSave() {
    const path = await saveCanvasData(completedStrokes);
    if (!path || !svgRef.current) return;
    try {
      const analysis = await analyzeCanvas(svgRef.current, completedStrokes);
      await writeCanvasFile(path, completedStrokes, analysis);
      setLoadedPath(path); // the canvas is now this file, so Analyze can skip it
    } catch {
      // ML service not available, file saved with analysis: null
    }
    completedStrokes.forEach((stroke, i) => {
      console.log(`stroke ${i}`);
      console.table(stroke.map(([x, y]) => ({ x, y })));
    });"Added this log statement for debugging purposes. Easily allows you to see the x and y coordinates of each completedStroke. Hit F12 in the Tauri app."
  }

  async function handleLoad() {
    const loaded = await loadCanvasData();
    if (!loaded) return;
    setCompletedStrokes(loaded.strokes);
    setLoadedPath(loaded.path);
    setResult(null);
    setError(null);
  }

  function handleClear() {
    setCompletedStrokes([]);
    setLoadedPath(null);
    setResult(null);
    setError(null);
  }

  // Embeds the canvas, then ranks every saved note against it. The comparison
  // happens here rather than in the ML service, which stays stateless and never
  // learns the saves folder exists.
  async function handleAnalyze() {
    if (!svgRef.current) return;
    setAnalyzing(true);
    setResult(null);
    setError(null);
    try {
      const analysis = await analyzeCanvas(svgRef.current, completedStrokes);
      const { notes, skipped } = await readSavedNotes();
      // Drop the note already on the canvas: it matches itself almost perfectly
      // and would crowd a real neighbour out of every top-3.
      const corpus = notes.filter((note) => note.path !== loadedPath);
      setResult({
        text: analysis.text,
        corpusSize: corpus.length,
        skipped,
        rankings: rank(analysis, corpus),
      });
    } catch {
      setError("Could not reach ML service.");
    } finally {
      setAnalyzing(false);
    }
  }

  const pathData = getSvgPathFromStroke(getStroke(currentStroke, STROKE_OPTS));

  return (
    <div style={{ position: "relative", width: "100vw", height: "100vh" }}>
      <svg
        ref={svgRef}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        style={{ touchAction: "none", width: "100%", height: "100%" }}
      >
        {currentStroke.length > 0 && <path d={pathData} />}
        {completedStrokes.map((stroke, i) => (
          <path
            key={i}
            d={getSvgPathFromStroke(getStroke(stroke, STROKE_OPTS))}
          />
        ))}
      </svg>
      <div
        style={{
          position: "absolute",
          top: 16,
          right: 16,
          display: "flex",
          flexDirection: "column",
          alignItems: "flex-end",
          gap: 8,
        }}
      >
        <button
          onClick={handleAnalyze}
          disabled={analyzing || completedStrokes.length === 0}
        >
          {analyzing ? "Analyzing…" : "Analyze"}
        </button>
        <button
          onClick={handleClear}
          disabled={analyzing || completedStrokes.length === 0}
        >
          Clear
        </button>
        <button onClick={handleSave} disabled={completedStrokes.length === 0}>
          Save
        </button>
        <button onClick={handleLoad}>Load</button>
        {(error || result) && (
          <div
            style={{
              color: "black",
              background: "white",
              border: "1px solid #ccc",
              borderRadius: 4,
              padding: "8px 12px",
              width: 300,
              maxHeight: "70vh",
              overflowY: "auto",
              textAlign: "left",
              fontSize: 13,
            }}
          >
            {error ? (
              error
            ) : (
              <>
                <div style={{ whiteSpace: "pre-wrap" }}>
                  {result!.text.trim() || <em>no text recognized</em>}
                </div>
                <div
                  style={{
                    marginTop: 6,
                    paddingTop: 6,
                    borderTop: "1px solid #eee",
                    color: "#888",
                    fontSize: "0.85em",
                  }}
                >
                  compared against {result!.corpusSize} saved note
                  {result!.corpusSize === 1 ? "" : "s"}
                  {result!.skipped > 0 && `, ${result!.skipped} skipped`}
                </div>
                {MODALITIES.map((modality) => (
                  <RankingSection
                    key={modality}
                    modality={modality}
                    ranking={result!.rankings[modality]}
                  />
                ))}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

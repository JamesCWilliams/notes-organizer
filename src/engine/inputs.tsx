import { getStroke } from "perfect-freehand";
import { getSvgPathFromStroke } from "./svgpathfromstroke";
import { useRef, useState } from "react";

const ML_SERVICE_URL = "http://localhost:5000/transcribe";

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

//This is where the engine will gather user inputs from mouse clicks/stylus.
export function Draw() {
  const [currentStroke, setCurrentStroke] = useState<number[][]>([]);
  const [completedStrokes, setCompletedStrokes] = useState<number[][][]>([]);
  const [transcription, setTranscription] = useState<string | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
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

  async function handleAnalyze() {
    if (!svgRef.current) return;
    setAnalyzing(true);
    setTranscription(null);
    try {
      const image = await svgElementToPngDataUrl(svgRef.current);
      const res = await fetch(ML_SERVICE_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image }),
      });
      const data = await res.json();
      setTranscription(data.text);
    } catch (e) {
      setTranscription("Error: could not reach ML service.");
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
          onClick={() => setCompletedStrokes([])}
          disabled={analyzing || completedStrokes.length === 0}
        >
          Clear
        </button>
        {transcription !== null && (
          <div
            style={{
              background: "white",
              border: "1px solid #ccc",
              borderRadius: 4,
              padding: "8px 12px",
              maxWidth: 300,
            }}
          >
            {transcription}
          </div>
        )}
      </div>
    </div>
  );
}

import { getStroke } from "perfect-freehand";
import { getSvgPathFromStroke } from "./svgpathfromstroke";
import { useState } from "react";
//This is where the engine will gather user inputs from mouse clicks/stylus.
export function Draw() {
  const [currentStroke, setCurrentStroke] = useState<number[][]>([]);
  const [completedStrokes, setCompletedStrokes] = useState<number[][][]>([]);

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
  const stroke = getStroke(currentStroke, {
    size: 16,
    thinning: 0.5,
    smoothing: 0.5,
    streamline: 0.5,
  });

  const pathData = getSvgPathFromStroke(stroke);

  return (
    <svg
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      style={{ touchAction: "none", width: "100vw", height: "100vh" }}
    >
      {currentStroke && <path d={pathData} />}
      {completedStrokes.map((stroke, i) => (
        <path
          key={i}
          d={getSvgPathFromStroke(
            getStroke(stroke, {
              size: 16,
              thinning: 0.5,
              smoothing: 0.5,
              streamline: 0.5,
            }),
          )}
        />
      ))}
    </svg>
  );
}

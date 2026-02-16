interface LoadingOverlayProps {
  pct: number;
  phase: number;
}

const PHASE_LABELS: Record<number, string> = {
  0: "Preparing\u2026",
  1: "Reading file\u2026",
  2: "Parsing\u2026",
  3: "Decoding textures\u2026",
  4: "Building meshes\u2026",
  5: "Assembling scene\u2026",
  6: "Complete",
  7: "Failed",
};

export function LoadingOverlay({ pct, phase }: LoadingOverlayProps) {
  const label = PHASE_LABELS[phase] ?? "Loading\u2026";

  return (
    <div className="loading-overlay">
      <div className="loading-card">
        <div className="loading-phase">{label}</div>
        <div className="loading-bar-track">
          <div className="loading-bar-fill" style={{ width: `${pct}%` }} />
        </div>
        <div className="loading-pct">{pct}%</div>
      </div>
    </div>
  );
}

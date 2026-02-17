import { WebLoadPhase } from "../../pkg/wgpu_engine";

interface LoadingOverlayProps {
  pct: number;
  phase: WebLoadPhase;
}

const PHASE_LABELS: Record<WebLoadPhase, string> = {
  [WebLoadPhase.Pending]: "Preparing\u2026",
  [WebLoadPhase.Reading]: "Reading file\u2026",
  [WebLoadPhase.Parsing]: "Parsing\u2026",
  [WebLoadPhase.DecodingTextures]: "Decoding textures\u2026",
  [WebLoadPhase.BuildingMeshes]: "Building meshes\u2026",
  [WebLoadPhase.Assembling]: "Assembling scene\u2026",
  [WebLoadPhase.Complete]: "Complete",
  [WebLoadPhase.Failed]: "Failed",
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

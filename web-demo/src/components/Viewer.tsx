import { useEffect, useRef } from "react";
import type { WebViewer } from "../../pkg/wgpu_engine";

interface ViewerProps {
  onReady?: (viewer: WebViewer) => void;
}

export function Viewer({ onReady }: ViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const viewerRef = useRef<WebViewer | null>(null);

  useEffect(() => {
    let animationId = 0;
    let disposed = false;

    async function init() {
      const canvas = canvasRef.current;
      const container = containerRef.current;
      if (!canvas || !container) return;

      // Set initial canvas pixel size
      const dpr = devicePixelRatio;
      const rect = container.getBoundingClientRect();
      canvas.width = Math.max(1, Math.round(rect.width * dpr));
      canvas.height = Math.max(1, Math.round(rect.height * dpr));

      // Import and initialize WASM
      const wasm = await import("../../pkg/wgpu_engine");
      await wasm.default();

      if (disposed) return;

      const viewer = await wasm.WebViewer.create(canvas);
      if (disposed) {
        viewer.free();
        return;
      }

      viewerRef.current = viewer;
      onReady?.(viewer);

      // Render loop
      function animate() {
        viewer.update_and_render();
        animationId = requestAnimationFrame(animate);
      }
      animationId = requestAnimationFrame(animate);
    }

    init().catch((e) => console.error("Failed to initialize viewer:", e));

    return () => {
      disposed = true;
      cancelAnimationFrame(animationId);
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Resize handling
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;

      const canvas = canvasRef.current;
      const viewer = viewerRef.current;
      if (!canvas) return;

      const dpr = devicePixelRatio;
      const width = Math.max(1, Math.round(entry.contentRect.width * dpr));
      const height = Math.max(1, Math.round(entry.contentRect.height * dpr));

      canvas.width = width;
      canvas.height = height;
      viewer?.resize(width, height);
    });

    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  return (
    <div ref={containerRef} className="viewer-container">
      <canvas
        ref={canvasRef}
        tabIndex={0}
        onMouseMove={(e) => {
          const canvas = canvasRef.current!;
          const rect = canvas.getBoundingClientRect();
          const dpr = devicePixelRatio;
          const x = (e.clientX - rect.left) * dpr;
          const y = (e.clientY - rect.top) * dpr;
          viewerRef.current?.on_mouse_move(x, y, e.movementX * dpr, e.movementY * dpr);
        }}
        onMouseDown={(e) => {
          viewerRef.current?.on_mouse_down(e.button);
          // Focus canvas for keyboard events
          canvasRef.current?.focus();
        }}
        onMouseUp={(e) => {
          viewerRef.current?.on_mouse_up(e.button);
        }}
        onWheel={(e) => {
          e.preventDefault();
          viewerRef.current?.on_wheel(e.deltaX, e.deltaY);
        }}
        onKeyDown={(e) => {
          viewerRef.current?.on_key_down(e.key, e.keyCode, e.repeat);
        }}
        onKeyUp={(e) => {
          viewerRef.current?.on_key_up(e.key, e.keyCode);
        }}
        onContextMenu={(e) => e.preventDefault()}
      />
    </div>
  );
}

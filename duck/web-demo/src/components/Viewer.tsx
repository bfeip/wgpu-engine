import { useEffect, useRef } from "react";
import { LoadStatus, WebLoadPhase, type WebViewer } from "../../pkg/wgpu_engine_viewer";

interface ViewerProps {
  onReady?: (viewer: WebViewer) => void;
  onLoadProgress?: (pct: number, phase: WebLoadPhase) => void;
  onLoadComplete?: (success: boolean) => void;
}

export function Viewer({ onReady, onLoadProgress, onLoadComplete }: ViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const viewerRef = useRef<WebViewer | null>(null);

  // Keep callbacks in a ref so the rAF loop always sees the latest
  const callbacksRef = useRef({ onLoadProgress, onLoadComplete });
  useEffect(() => {
    callbacksRef.current = { onLoadProgress, onLoadComplete };
  });

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
      const wasm = await import("../../pkg/wgpu_engine_viewer");
      await wasm.default();

      if (disposed) return;

      const viewer = await wasm.WebViewer.create(canvas);
      if (disposed) {
        viewer.free();
        return;
      }

      viewerRef.current = viewer;
      onReady?.(viewer);

      // Track last reported values to avoid redundant React updates
      let lastPct = -1;
      let lastPhase = -1;

      // Render loop
      function animate() {
        // Poll async load status
        const status = viewer.poll_load();
        if (status === LoadStatus.InProgress) {
          const pct = viewer.load_progress_pct();
          const phase = viewer.load_phase();
          if (pct !== lastPct || phase !== lastPhase) {
            lastPct = pct;
            lastPhase = phase;
            callbacksRef.current.onLoadProgress?.(pct, phase);
          }
        } else if (status === LoadStatus.Success) {
          lastPct = -1;
          lastPhase = -1;
          callbacksRef.current.onLoadComplete?.(true);
        } else if (status === LoadStatus.Error) {
          lastPct = -1;
          lastPhase = -1;
          callbacksRef.current.onLoadComplete?.(false);
        }

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

  // Input event handling — imperative listeners so we can use { passive: false }
  // for touch and wheel events (React registers these as passive by default,
  // which silently ignores preventDefault).
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const onMouseMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const dpr = devicePixelRatio;
      const x = (e.clientX - rect.left) * dpr;
      const y = (e.clientY - rect.top) * dpr;
      viewerRef.current?.on_mouse_move(x, y, e.movementX * dpr, e.movementY * dpr);
    };

    const onMouseDown = (e: MouseEvent) => {
      viewerRef.current?.on_mouse_down(e.button);
      canvas.focus();
    };

    const onMouseUp = (e: MouseEvent) => {
      viewerRef.current?.on_mouse_up(e.button);
    };

    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      viewerRef.current?.on_wheel(e.deltaX, e.deltaY);
    };

    const onKeyDown = (e: KeyboardEvent) => {
      viewerRef.current?.on_key_down(e.key, e.keyCode, e.repeat);
    };

    const onKeyUp = (e: KeyboardEvent) => {
      viewerRef.current?.on_key_up(e.key, e.keyCode);
    };

    const onTouchStart = (e: TouchEvent) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const dpr = devicePixelRatio;
      for (let i = 0; i < e.changedTouches.length; i++) {
        const t = e.changedTouches[i];
        viewerRef.current?.on_touch_start(
          t.identifier,
          (t.clientX - rect.left) * dpr,
          (t.clientY - rect.top) * dpr
        );
      }
    };

    const onTouchMove = (e: TouchEvent) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const dpr = devicePixelRatio;
      for (let i = 0; i < e.changedTouches.length; i++) {
        const t = e.changedTouches[i];
        viewerRef.current?.on_touch_move(
          t.identifier,
          (t.clientX - rect.left) * dpr,
          (t.clientY - rect.top) * dpr
        );
      }
    };

    const onTouchEnd = (e: TouchEvent) => {
      e.preventDefault();
      for (let i = 0; i < e.changedTouches.length; i++) {
        viewerRef.current?.on_touch_end(e.changedTouches[i].identifier);
      }
    };

    const onTouchCancel = (e: TouchEvent) => {
      for (let i = 0; i < e.changedTouches.length; i++) {
        viewerRef.current?.on_touch_cancel(e.changedTouches[i].identifier);
      }
    };

    const onContextMenu = (e: Event) => {
      e.preventDefault();
    };

    canvas.addEventListener("mousemove", onMouseMove);
    canvas.addEventListener("mousedown", onMouseDown);
    canvas.addEventListener("mouseup", onMouseUp);
    canvas.addEventListener("wheel", onWheel, { passive: false });
    canvas.addEventListener("keydown", onKeyDown);
    canvas.addEventListener("keyup", onKeyUp);
    canvas.addEventListener("touchstart", onTouchStart, { passive: false });
    canvas.addEventListener("touchmove", onTouchMove, { passive: false });
    canvas.addEventListener("touchend", onTouchEnd, { passive: false });
    canvas.addEventListener("touchcancel", onTouchCancel);
    canvas.addEventListener("contextmenu", onContextMenu);

    return () => {
      canvas.removeEventListener("mousemove", onMouseMove);
      canvas.removeEventListener("mousedown", onMouseDown);
      canvas.removeEventListener("mouseup", onMouseUp);
      canvas.removeEventListener("wheel", onWheel);
      canvas.removeEventListener("keydown", onKeyDown);
      canvas.removeEventListener("keyup", onKeyUp);
      canvas.removeEventListener("touchstart", onTouchStart);
      canvas.removeEventListener("touchmove", onTouchMove);
      canvas.removeEventListener("touchend", onTouchEnd);
      canvas.removeEventListener("touchcancel", onTouchCancel);
      canvas.removeEventListener("contextmenu", onContextMenu);
    };
  }, []);

  return (
    <div ref={containerRef} className="viewer-container">
      <canvas
        ref={canvasRef}
        tabIndex={0}
      />
    </div>
  );
}

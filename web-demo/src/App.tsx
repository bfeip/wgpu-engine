import { useCallback, useRef, useState } from "react";
import { Viewer } from "./components/Viewer";
import type { WebViewer } from "../pkg/wgpu_engine";
import "./App.css";

export function App() {
  const viewerRef = useRef<WebViewer | null>(null);
  const [sceneInfo, setSceneInfo] = useState({ nodes: 0, meshes: 0 });

  const handleViewerReady = useCallback(async (viewer: WebViewer) => {
    viewerRef.current = viewer;
    try {
      const resp = await fetch("/default-scene.wgsc");
      if (resp.ok) {
        const data = new Uint8Array(await resp.arrayBuffer());
        viewer.load_scene(data);
        setSceneInfo({
          nodes: viewer.node_count(),
          meshes: viewer.mesh_count(),
        });
      }
    } catch (e) {
      console.warn("Failed to load default scene:", e);
    }
  }, []);

  const handleLoadFile = useCallback(async () => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".gltf,.glb,.wgsc";
    input.onchange = async () => {
      const file = input.files?.[0];
      if (!file) return;
      const data = new Uint8Array(await file.arrayBuffer());
      try {
        if (file.name.endsWith(".wgsc")) {
          viewer.load_scene(data);
        } else {
          viewer.load_gltf(data);
        }
        setSceneInfo({
          nodes: viewer.node_count(),
          meshes: viewer.mesh_count(),
        });
      } catch (e) {
        console.error("Failed to load file:", e);
      }
    };
    input.click();
  }, []);

  const handleClear = useCallback(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;
    viewer.clear_scene();
    setSceneInfo({ nodes: 0, meshes: 0 });
  }, []);

  return (
    <div className="app">
      <div className="sidebar">
        <h1>wgpu-engine</h1>
        <p className="subtitle">Web Demo</p>

        <div className="controls">
          <button onClick={handleLoadFile}>Load Scene</button>
          <button onClick={handleClear}>Clear</button>
        </div>

        {sceneInfo.meshes > 0 && (
          <div className="scene-info">
            <h2>Scene</h2>
            <p>Nodes: {sceneInfo.nodes}</p>
            <p>Meshes: {sceneInfo.meshes}</p>
          </div>
        )}
      </div>
      <Viewer onReady={handleViewerReady} />
    </div>
  );
}

import fs from "fs";
import path from "path";
import { UMAP } from "umap-js";

export interface BuildOptions {
	input: string;
	output: string;
	neighbors: number;
	minDist: number;
}

interface EmbeddingEntry {
	url: string;
	title: string;
	body: string;
	embedding: number[];
}

interface Point {
	x: number;
	y: number;
	x3d: number;
	y3d: number;
	z3d: number;
	title: string;
	url: string;
	body: string;
}

export function build(options: BuildOptions): void {
	const inputPath = path.resolve(options.input);
	const outputPath = path.resolve(options.output);

	const lines = fs.readFileSync(inputPath, "utf8").split("\n").filter(Boolean);
	const entries: EmbeddingEntry[] = [];
	for (const line of lines) {
		try {
			entries.push(JSON.parse(line));
		} catch {
			// skip invalid lines
		}
	}

	const embeddings = entries.map((entry) => entry.embedding);

	const umap2d = new UMAP({
		nComponents: 2,
		nNeighbors: options.neighbors,
		minDist: options.minDist,
		random: Math.random,
	});
	const umap3d = new UMAP({
		nComponents: 3,
		nNeighbors: options.neighbors,
		minDist: options.minDist,
		random: Math.random,
	});

	console.log(`Running UMAP 2D on ${embeddings.length} embeddings`);
	const coords2d = umap2d.fit(embeddings);
	console.log(`Running UMAP 3D on ${embeddings.length} embeddings`);
	const coords3d = umap3d.fit(embeddings);

	const xs = coords2d.map((c) => c[0]);
	const ys = coords2d.map((c) => c[1]);
	const minX = Math.min(...xs);
	const maxX = Math.max(...xs);
	const minY = Math.min(...ys);
	const maxY = Math.max(...ys);
	const rangeX = maxX - minX || 1;
	const rangeY = maxY - minY || 1;

	const xs3 = coords3d.map((c) => c[0]);
	const ys3 = coords3d.map((c) => c[1]);
	const zs3 = coords3d.map((c) => c[2]);
	const minX3 = Math.min(...xs3);
	const maxX3 = Math.max(...xs3);
	const minY3 = Math.min(...ys3);
	const maxY3 = Math.max(...ys3);
	const minZ3 = Math.min(...zs3);
	const maxZ3 = Math.max(...zs3);
	const rangeX3 = maxX3 - minX3 || 1;
	const rangeY3 = maxY3 - minY3 || 1;
	const rangeZ3 = maxZ3 - minZ3 || 1;

	const points: Point[] = coords2d.map((coord, index) => {
		const entry = entries[index];
		const coord3d = coords3d[index] || [0, 0, 0];
		return {
			x: (coord[0] - minX) / rangeX,
			y: (coord[1] - minY) / rangeY,
			x3d: (coord3d[0] - minX3) / rangeX3,
			y3d: (coord3d[1] - minY3) / rangeY3,
			z3d: (coord3d[2] - minZ3) / rangeZ3,
			title: entry.title || "",
			url: entry.url || "",
			body: entry.body || "",
		};
	});

	const dataJson = JSON.stringify(points).replace(/</g, "\\u003c");

	const html = generateHtml(dataJson);

	const outputDir = path.dirname(outputPath);
	fs.mkdirSync(outputDir, { recursive: true });
	fs.writeFileSync(outputPath, html);

	console.log(`Wrote ${outputPath}`);
}

function generateHtml(dataJson: string): string {
	return `<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Doppelgangers - PR Triage</title>
    <style>
      :root {
        color-scheme: light dark;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        --bg: #0f172a;
        --panel: #111827;
        --text: #f8fafc;
        --muted: #94a3b8;
        --accent: #38bdf8;
        --point: #e2e8f0;
        --selected: #f59e0b;
      }
      body {
        margin: 0;
        background: var(--bg);
        color: var(--text);
        height: 100vh;
        overflow: hidden;
      }
      #app {
        display: flex;
        height: 100vh;
      }
      #plot-wrap {
        flex: 1;
        position: relative;
      }
      #plot {
        display: block;
        width: 100%;
        height: 100%;
        background: var(--bg);
        cursor: grab;
      }
      #plot.dragging {
        cursor: grabbing;
      }
      #hud {
        position: absolute;
        top: 12px;
        left: 12px;
        background: rgba(17, 24, 39, 0.8);
        padding: 8px 12px;
        border-radius: 8px;
        font-size: 12px;
        line-height: 1.4;
        color: var(--muted);
      }
      #hud-row {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 6px;
      }
      #hud-mode {
        font-weight: 600;
        color: var(--text);
      }
      #toggle-mode {
        background: transparent;
        color: var(--text);
        border: 1px solid rgba(148, 163, 184, 0.4);
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 11px;
        cursor: pointer;
      }
      #toggle-mode:hover {
        border-color: var(--accent);
        color: var(--accent);
      }
      #sidebar {
        width: 360px;
        background: var(--panel);
        border-left: 1px solid rgba(148, 163, 184, 0.2);
        padding: 16px;
        overflow: auto;
      }
      #sidebar h2 {
        margin: 0 0 8px 0;
        font-size: 18px;
      }
      #selection-count {
        color: var(--muted);
        margin-bottom: 12px;
        font-size: 13px;
      }
      #selected-list {
        list-style: none;
        padding: 0;
        margin: 0;
        display: grid;
        gap: 12px;
      }
      #selected-list li {
        padding: 10px;
        background: rgba(15, 23, 42, 0.6);
        border-radius: 8px;
        border: 1px solid rgba(148, 163, 184, 0.2);
      }
      #selected-list a {
        color: var(--accent);
        text-decoration: none;
        font-weight: 600;
      }
      #selected-list p {
        margin: 6px 0 0 0;
        color: var(--muted);
        font-size: 12px;
      }
    </style>
  </head>
  <body>
    <div id="app">
      <div id="plot-wrap">
        <canvas id="plot"></canvas>
        <div id="hud">
          <div id="hud-row">
            <div id="hud-mode">Mode: 2D</div>
            <button id="toggle-mode" type="button">Toggle 3D</button>
          </div>
          <div id="hud-instructions">
            <div id="hud-line-2d">2D: drag to pan, scroll to zoom</div>
            <div id="hud-line-3d">3D: drag to rotate, ctrl+drag to pan</div>
            <div>Shift+drag to select, ctrl+drag to add</div>
            <div>Click empty space to deselect</div>
          </div>
          <div id="hud-count"></div>
        </div>
      </div>
      <aside id="sidebar">
        <h2>Selection</h2>
        <div id="selection-count">0 selected</div>
        <ul id="selected-list"></ul>
      </aside>
    </div>
    <script>
      const data = ${dataJson};
      const canvas = document.getElementById("plot");
      const ctx = canvas.getContext("2d");
      const selectionCount = document.getElementById("selection-count");
      const selectedList = document.getElementById("selected-list");
      const hudCount = document.getElementById("hud-count");
      const hudMode = document.getElementById("hud-mode");
      const toggleMode = document.getElementById("toggle-mode");
      const hudLine2d = document.getElementById("hud-line-2d");
      const hudLine3d = document.getElementById("hud-line-3d");
      const styles = getComputedStyle(document.documentElement);
      const colors = {
        point: styles.getPropertyValue("--point").trim() || "#e2e8f0",
        selected: styles.getPropertyValue("--selected").trim() || "#f59e0b",
        accent: styles.getPropertyValue("--accent").trim() || "#38bdf8",
        muted: styles.getPropertyValue("--muted").trim() || "#94a3b8"
      };

      const view2d = {
        scale: 1,
        offsetX: 0,
        offsetY: 0
      };

      const view3d = {
        rotateX: 0.35,
        rotateY: -0.6,
        zoom: 1.2,
        offsetX: 0,
        offsetY: 0
      };

      const state = {
        mode: "2d",
        dragging: false,
        selecting: false,
        dragMode: null,
        dragStart: { x: 0, y: 0 },
        selectRect: null,
        selected: new Set(),
        addToSelection: false
      };

      const resize = () => {
        canvas.width = canvas.clientWidth * devicePixelRatio;
        canvas.height = canvas.clientHeight * devicePixelRatio;
        ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
        scheduleRender();
      };

      const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

      const setMode = (mode) => {
        state.mode = mode;
        hudMode.textContent = mode === "3d" ? "Mode: 3D" : "Mode: 2D";
        toggleMode.textContent = mode === "3d" ? "Toggle 2D" : "Toggle 3D";
        hudLine2d.style.opacity = mode === "2d" ? "1" : "0.5";
        hudLine3d.style.opacity = mode === "3d" ? "1" : "0.5";
        scheduleRender();
      };

      const getCanvasPoint = (event) => {
        const rect = canvas.getBoundingClientRect();
        return {
          x: event.clientX - rect.left,
          y: event.clientY - rect.top
        };
      };

      const worldToScreen = (point) => {
        return {
          x: point.x * canvas.clientWidth * view2d.scale + view2d.offsetX,
          y: point.y * canvas.clientHeight * view2d.scale + view2d.offsetY
        };
      };

      const project3d = (point) => {
        const x = (point.x3d - 0.5) * view3d.zoom;
        const y = (point.y3d - 0.5) * view3d.zoom;
        const z = (point.z3d - 0.5) * view3d.zoom;
        const cosY = Math.cos(view3d.rotateY);
        const sinY = Math.sin(view3d.rotateY);
        const cosX = Math.cos(view3d.rotateX);
        const sinX = Math.sin(view3d.rotateX);
        const x1 = x * cosY + z * sinY;
        const z1 = -x * sinY + z * cosY;
        const y1 = y * cosX - z1 * sinX;
        const z2 = y * sinX + z1 * cosX;
        const perspective = 1 / (1 + z2 * 0.9);
        const screenX = x1 * perspective * canvas.clientWidth * 0.7 + canvas.clientWidth * 0.5 + view3d.offsetX;
        const screenY = y1 * perspective * canvas.clientHeight * 0.7 + canvas.clientHeight * 0.5 + view3d.offsetY;
        return { x: screenX, y: screenY, depth: z2 };
      };

      const getScreenPoint = (point) => {
        if (state.mode === "3d") return project3d(point);
        return worldToScreen(point);
      };

      const updateSidebar = () => {
        const selected = Array.from(state.selected);
        selectionCount.textContent = selected.length + " selected";
        const maxDisplay = 200;
        const list = selected.slice(0, maxDisplay);
        selectedList.innerHTML = "";
        for (const index of list) {
          const item = data[index];
          const li = document.createElement("li");
          const link = document.createElement("a");
          link.href = item.url;
          link.target = "_blank";
          link.rel = "noreferrer";
          link.textContent = item.title || item.url;
          li.appendChild(link);
          if (item.body) {
            const snippet = document.createElement("p");
            snippet.textContent = item.body;
            li.appendChild(snippet);
          }
          selectedList.appendChild(li);
        }
        if (selected.length > maxDisplay) {
          const more = document.createElement("p");
          more.textContent = "Showing " + maxDisplay + " of " + selected.length;
          more.style.color = colors.muted;
          selectedList.appendChild(more);
        }
      };

      const selectPoints = () => {
        if (!state.selectRect) return;
        const rect = state.selectRect;
        const left = Math.min(rect.x0, rect.x1);
        const right = Math.max(rect.x0, rect.x1);
        const top = Math.min(rect.y0, rect.y1);
        const bottom = Math.max(rect.y0, rect.y1);
        if (!state.addToSelection) {
          state.selected.clear();
        }
        for (let i = 0; i < data.length; i += 1) {
          const screen = getScreenPoint(data[i]);
          if (screen.x >= left && screen.x <= right && screen.y >= top && screen.y <= bottom) {
            state.selected.add(i);
          }
        }
        updateSidebar();
      };

      const hitTestPoint = (clickX, clickY, radius) => {
        for (let i = 0; i < data.length; i += 1) {
          const screen = getScreenPoint(data[i]);
          const dx = screen.x - clickX;
          const dy = screen.y - clickY;
          if (dx * dx + dy * dy <= radius * radius) {
            return i;
          }
        }
        return -1;
      };

      let scheduled = false;
      const scheduleRender = () => {
        if (scheduled) return;
        scheduled = true;
        requestAnimationFrame(() => {
          scheduled = false;
          render();
        });
      };

      const render = () => {
        ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
        const projected = [];
        for (let i = 0; i < data.length; i += 1) {
          const screen = getScreenPoint(data[i]);
          projected.push({ index: i, screen });
        }
        if (state.mode === "3d") {
          projected.sort((a, b) => a.screen.depth - b.screen.depth);
        }
        for (const item of projected) {
          const radius = state.selected.has(item.index) ? 3.5 : 2;
          ctx.beginPath();
          ctx.fillStyle = state.selected.has(item.index) ? colors.selected : colors.point;
          ctx.arc(item.screen.x, item.screen.y, radius, 0, Math.PI * 2);
          ctx.fill();
        }
        if (state.selectRect) {
          const rect = state.selectRect;
          const left = Math.min(rect.x0, rect.x1);
          const right = Math.max(rect.x0, rect.x1);
          const top = Math.min(rect.y0, rect.y1);
          const bottom = Math.max(rect.y0, rect.y1);
          ctx.strokeStyle = colors.accent;
          ctx.lineWidth = 1;
          ctx.setLineDash([4, 4]);
          ctx.strokeRect(left, top, right - left, bottom - top);
          ctx.setLineDash([]);
        }
        hudCount.textContent = data.length + " PRs";
      };

      canvas.addEventListener("mousedown", (event) => {
        if (event.button !== 0) return;
        const point = getCanvasPoint(event);
        
        // Shift+drag: select (replace selection)
        // Ctrl+drag in 2D: select (add to selection)
        // Ctrl+drag in 3D: pan
        if (event.shiftKey) {
          state.selecting = true;
          state.addToSelection = false;
          state.selectRect = {
            x0: point.x,
            y0: point.y,
            x1: point.x,
            y1: point.y
          };
          scheduleRender();
          return;
        }
        
        if (event.ctrlKey || event.metaKey) {
          if (state.mode === "2d") {
            // Ctrl+drag in 2D: add to selection
            state.selecting = true;
            state.addToSelection = true;
            state.selectRect = {
              x0: point.x,
              y0: point.y,
              x1: point.x,
              y1: point.y
            };
            scheduleRender();
            return;
          } else {
            // Ctrl+drag in 3D: pan
            state.dragging = true;
            state.dragMode = "pan3d";
            state.dragStart = { x: point.x, y: point.y };
            canvas.classList.add("dragging");
            return;
          }
        }
        
        state.dragging = true;
        state.dragMode = state.mode === "3d" ? "rotate" : "pan";
        state.dragStart = { x: point.x, y: point.y };
        canvas.classList.add("dragging");
      });

      const handleMove = (event) => {
        const point = getCanvasPoint(event);
        if (state.dragging) {
          const dx = point.x - state.dragStart.x;
          const dy = point.y - state.dragStart.y;
          if (state.dragMode === "pan") {
            view2d.offsetX += dx;
            view2d.offsetY += dy;
          } else if (state.dragMode === "rotate") {
            view3d.rotateY += dx * 0.005;
            view3d.rotateX = clamp(view3d.rotateX + dy * 0.005, -1.5, 1.5);
          } else if (state.dragMode === "pan3d") {
            view3d.offsetX += dx;
            view3d.offsetY += dy;
          }
          state.dragStart = { x: point.x, y: point.y };
          scheduleRender();
          return;
        }
        if (state.selecting && state.selectRect) {
          state.selectRect.x1 = point.x;
          state.selectRect.y1 = point.y;
          scheduleRender();
        }
      };

      const endDrag = (event) => {
        const point = getCanvasPoint(event);
        const wasDragging = state.dragging;
        const wasSelecting = state.selecting;
        const dragDist = Math.hypot(point.x - state.dragStart.x, point.y - state.dragStart.y);
        
        if (state.dragging) {
          state.dragging = false;
          state.dragMode = null;
          canvas.classList.remove("dragging");
        }
        if (state.selecting) {
          state.selecting = false;
          selectPoints();
          state.selectRect = null;
          state.addToSelection = false;
          scheduleRender();
        }
        
        // Click to deselect (if no drag and no selection and clicked empty space)
        if (!wasDragging && !wasSelecting) {
          // This was a simple click
          const hitIndex = hitTestPoint(point.x, point.y, 8);
          if (hitIndex === -1) {
            state.selected.clear();
            updateSidebar();
            scheduleRender();
          }
        } else if (wasDragging && dragDist < 3) {
          // Very short drag = click
          const hitIndex = hitTestPoint(point.x, point.y, 8);
          if (hitIndex === -1) {
            state.selected.clear();
            updateSidebar();
            scheduleRender();
          }
        }
      };

      canvas.addEventListener("mousemove", handleMove);
      window.addEventListener("mousemove", handleMove);
      window.addEventListener("mouseup", endDrag);

      toggleMode.addEventListener("click", () => {
        setMode(state.mode === "2d" ? "3d" : "2d");
      });

      canvas.addEventListener("wheel", (event) => {
        event.preventDefault();
        const point = getCanvasPoint(event);
        const zoom = Math.exp(-event.deltaY * 0.001);
        if (state.mode === "3d") {
          view3d.zoom = clamp(view3d.zoom * zoom, 0.4, 3.5);
          scheduleRender();
          return;
        }
        const prevScale = view2d.scale;
        view2d.scale = clamp(view2d.scale * zoom, 0.2, 12);
        const scaleChange = view2d.scale / prevScale;
        view2d.offsetX = point.x - (point.x - view2d.offsetX) * scaleChange;
        view2d.offsetY = point.y - (point.y - view2d.offsetY) * scaleChange;
        scheduleRender();
      }, { passive: false });

      window.addEventListener("resize", resize);
      setMode("2d");
      resize();
      updateSidebar();
    </script>
  </body>
</html>`;
}

// CLI entry point
if (process.argv[1]?.endsWith("build.js") || process.argv[1]?.endsWith("build.ts")) {
	const args = process.argv.slice(2);
	const options: BuildOptions = {
		input: "embeddings.jsonl",
		output: "triage.html",
		neighbors: 15,
		minDist: 0.1,
	};

	for (let i = 0; i < args.length; i += 1) {
		const arg = args[i];
		if (arg === "--input") {
			options.input = args[++i];
		} else if (arg === "--output") {
			options.output = args[++i];
		} else if (arg === "--neighbors") {
			options.neighbors = Number(args[++i]);
		} else if (arg === "--min-dist") {
			options.minDist = Number(args[++i]);
		}
	}

	build(options);
}

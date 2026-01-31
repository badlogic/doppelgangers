import fs from "fs";
import { PCA } from "ml-pca";
import path from "path";
import readline from "readline";
import { UMAP } from "umap-js";
import type { EmbeddingRecord } from "./embed.js";

export interface BuildOptions {
	input: string;
	output: string;
	projections: string;
	neighbors: number;
	minDist: number;
	spread: number;
	includeEmbeddings: boolean;
	force: boolean;
	pcaDims?: number;
}

interface Point {
	x: number;
	y: number;
	x3d: number;
	y3d: number;
	z3d: number;
	title: string;
	number?: number;
	url: string;
	body: string;
	state?: string;
	type?: string;
	files?: string[];
	embedding?: number[];
}

export async function build(options: BuildOptions): Promise<void> {
	const inputPath = path.resolve(options.input);
	const outputPath = path.resolve(options.output);

	const lines = fs.readFileSync(inputPath, "utf8").split("\n").filter(Boolean);
	const entries: EmbeddingRecord[] = [];
	for (const line of lines) {
		try {
			entries.push(JSON.parse(line));
		} catch {
			// skip invalid lines
		}
	}

	const embeddings = entries.map((entry) => entry.embedding);
	const projectionsPath = path.resolve(options.projections);

	let coords2d: number[][];
	let coords3d: number[][];

	if (!options.force && fs.existsSync(projectionsPath)) {
		console.log(`Loading cached projections from ${projectionsPath}`);
		const cached = JSON.parse(fs.readFileSync(projectionsPath, "utf8"));
		coords2d = cached.coords2d;
		coords3d = cached.coords3d;
	} else {
		console.log(`Running PCA on ${embeddings.length} embeddings...`);
		const pca = new PCA(embeddings);
		const explained = pca.getExplainedVariance();

		// Show variance explained plot
		console.log("\nVariance explained by top components:");
		const width = 60;
		for (let i = 0; i < Math.min(20, explained.length); i++) {
			const val = explained[i];
			const barLen = Math.floor(val * width * 10); // scale up for visibility since variances are small
			const bar = "█".repeat(barLen);
			console.log(`PC${(i + 1).toString().padStart(2)}: ${val.toFixed(4)} ${bar}`);
		}

		// Calculate cumulative variance for a few thresholds
		let cumulative = 0;
		const thresholds = [0.5, 0.75, 0.9, 0.95, 0.99];
		const dimsNeeded = new Map<number, number>();

		for (let i = 0; i < explained.length; i++) {
			cumulative += explained[i];
			for (const t of thresholds) {
				if (cumulative >= t && !dimsNeeded.has(t)) {
					dimsNeeded.set(t, i + 1);
				}
			}
		}

		console.log("\nDimensions needed for cumulative variance:");
		for (const t of thresholds) {
			if (dimsNeeded.has(t)) {
				console.log(`${(t * 100).toFixed(0)}%: ${dimsNeeded.get(t)} dimensions`);
			}
		}

		let dimToKeep = 50;

		// Check if we should prompt
		if (options.pcaDims) {
			dimToKeep = options.pcaDims;
			console.log(`Using provided PCA dimensions: ${dimToKeep}`);
		} else if (process.stdout.isTTY) {
			const rl = readline.createInterface({
				input: process.stdin,
				output: process.stdout,
			});

			const askQuestion = (query: string): Promise<string> => {
				return new Promise((resolve) => rl.question(query, resolve));
			};

			const answer = await askQuestion("\nEnter number of dimensions to keep (default: 50): ");
			rl.close();
			dimToKeep = answer.trim() ? parseInt(answer.trim(), 10) : 50;
		} else {
			console.log(`Non-interactive environment detected, using default PCA dimensions: ${dimToKeep}`);
		}

		console.log(`Reducing to ${dimToKeep} dimensions...`);

		const reducedMatrix = pca.predict(embeddings, { nComponents: dimToKeep });
		const reducedEmbeddings = reducedMatrix.to2DArray();

		const umap2d = new UMAP({
			nComponents: 2,
			nNeighbors: options.neighbors,
			minDist: options.minDist,
			spread: options.spread,
			random: Math.random,
		});
		const umap3d = new UMAP({
			nComponents: 3,
			nNeighbors: options.neighbors,
			minDist: options.minDist,
			spread: options.spread,
			random: Math.random,
		});

		console.log(`Running UMAP 2D on ${reducedEmbeddings.length} embeddings (${dimToKeep} dims after PCA)`);
		coords2d = umap2d.fit(reducedEmbeddings);
		console.log(`Running UMAP 3D on ${reducedEmbeddings.length} embeddings (${dimToKeep} dims after PCA)`);
		coords3d = umap3d.fit(reducedEmbeddings);

		fs.writeFileSync(projectionsPath, JSON.stringify({ coords2d, coords3d }));
		console.log(`Saved projections to ${projectionsPath}`);
	}

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
			number: entry.number,
			url: entry.url || "",
			body: entry.body || "",
			state: entry.state,
			type: entry.type,
			files: entry.files,
			embedding: options.includeEmbeddings ? entry.embedding : undefined,
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
    <title>Doppelgangers - Issue & PR Triage</title>
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
        --point-open: #6ee7b7;
        --point-closed: #a78bfa;
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
        background: rgba(17, 24, 39, 0.9);
        padding: 10px 14px;
        border-radius: 8px;
        font-size: 12px;
        line-height: 1.5;
        color: var(--muted);
      }
      #hud-row {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
      }
      #hud-mode {
        font-weight: 600;
        color: var(--text);
      }
      .hud-btn {
        background: transparent;
        color: var(--text);
        border: 1px solid rgba(148, 163, 184, 0.4);
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 11px;
        cursor: pointer;
      }
      .hud-btn:hover {
        border-color: var(--accent);
        color: var(--accent);
      }
      #filters {
        display: flex;
        flex-wrap: wrap;
        gap: 6px 12px;
        margin-bottom: 8px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.2);
      }
      #filters label {
        display: flex;
        align-items: center;
        gap: 4px;
        cursor: pointer;
        font-size: 11px;
      }
      #filters input[type="checkbox"] {
        margin: 0;
      }
      #search-wrap {
        display: flex;
        gap: 6px;
        margin-bottom: 8px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(148, 163, 184, 0.2);
      }
      #search-input {
        flex: 1;
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.3);
        border-radius: 4px;
        padding: 4px 8px;
        color: var(--text);
        font-size: 11px;
      }
      #search-input::placeholder {
        color: var(--muted);
      }
      #search-btn {
        background: var(--accent);
        color: var(--bg);
        border: none;
        border-radius: 4px;
        padding: 4px 10px;
        font-size: 11px;
        font-weight: 600;
        cursor: pointer;
      }
      #search-btn:disabled {
        opacity: 0.5;
        cursor: not-allowed;
      }
      #sidebar {
        width: 480px;
        background: var(--panel);
        border-left: 1px solid rgba(148, 163, 184, 0.2);
        padding: 16px;
        overflow: auto;
        display: flex;
        flex-direction: column;
      }
      #sidebar-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
      }
      #sidebar-header h2 {
        margin: 0;
        font-size: 18px;
      }
      #sidebar-actions {
        display: flex;
        gap: 8px;
      }
      .sidebar-btn {
        background: transparent;
        color: var(--muted);
        border: 1px solid rgba(148, 163, 184, 0.3);
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 11px;
        cursor: pointer;
      }
      .sidebar-btn:hover {
        border-color: var(--accent);
        color: var(--accent);
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
        flex: 1;
        overflow: auto;
      }
      #selected-list li {
        padding: 10px;
        background: rgba(15, 23, 42, 0.6);
        border-radius: 8px;
        border: 1px solid rgba(148, 163, 184, 0.2);
      }
      .item-header {
        display: flex;
        align-items: flex-start;
        gap: 6px;
      }
      #selected-list a {
        color: var(--accent);
        text-decoration: none;
        font-weight: 600;
        flex: 1;
        word-break: break-word;
      }
      #selected-list p {
        margin: 6px 0 0 0;
        color: var(--muted);
        font-size: 12px;
        word-break: break-word;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        overflow: hidden;
      }
      .badge {
        display: inline-block;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 9px;
        font-weight: 600;
        text-transform: uppercase;
        flex-shrink: 0;
      }
      .badge-open {
        background: rgba(110, 231, 183, 0.2);
        color: var(--point-open);
      }
      .badge-closed {
        background: rgba(167, 139, 250, 0.2);
        color: var(--point-closed);
      }
      .badge-pr {
        background: rgba(56, 189, 248, 0.2);
        color: var(--accent);
      }
      .badge-issue {
        background: rgba(251, 191, 36, 0.2);
        color: #fbbf24;
      }
      .legend {
        display: flex;
        gap: 12px;
        font-size: 10px;
        margin-top: 4px;
      }
      .legend-item {
        display: flex;
        align-items: center;
        gap: 4px;
      }
      .legend-shape {
        width: 8px;
        height: 8px;
      }
      .legend-circle {
        border-radius: 50%;
        background: var(--muted);
      }
      .legend-ring {
        border: 2px solid var(--muted);
        border-radius: 50%;
        background: transparent;
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
            <button id="toggle-mode" class="hud-btn" type="button">Toggle 3D</button>
          </div>
          <div id="filters">
            <label><input type="checkbox" id="filter-pr" checked> PRs</label>
            <label><input type="checkbox" id="filter-issue" checked> Issues</label>
            <label><input type="checkbox" id="filter-open" checked> Open</label>
            <label><input type="checkbox" id="filter-closed" checked> Closed</label>
          </div>
          <div id="search-wrap">
            <input type="text" id="search-input" placeholder="Semantic search..." />
            <button id="search-btn" type="button">Search</button>
          </div>
          <div class="legend">
            <span class="legend-item"><span class="legend-shape legend-circle"></span> PR</span>
            <span class="legend-item"><span class="legend-shape legend-ring"></span> Issue</span>
          </div>
          <div id="hud-instructions" style="margin-top: 8px;">
            <div id="hud-line-2d">2D: drag to pan, scroll to zoom</div>
            <div id="hud-line-3d">3D: drag to rotate, ctrl+drag to pan</div>
            <div>Shift+drag to select, ctrl+shift to add</div>
          </div>
          <div id="hud-count"></div>
        </div>
      </div>
      <aside id="sidebar">
        <div id="sidebar-header">
          <h2>Selection</h2>
          <div id="sidebar-actions">
            <button id="open-all-btn" class="sidebar-btn" type="button">Open All</button>
            <button id="copy-btn" class="sidebar-btn" type="button">Copy</button>
          </div>
        </div>
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
      const openAllBtn = document.getElementById("open-all-btn");
      const copyBtn = document.getElementById("copy-btn");
      const searchInput = document.getElementById("search-input");
      const searchBtn = document.getElementById("search-btn");
      const filterPr = document.getElementById("filter-pr");
      const filterIssue = document.getElementById("filter-issue");
      const filterOpen = document.getElementById("filter-open");
      const filterClosed = document.getElementById("filter-closed");
      
      const styles = getComputedStyle(document.documentElement);
      const colors = {
        point: styles.getPropertyValue("--point").trim() || "#e2e8f0",
        pointOpen: styles.getPropertyValue("--point-open").trim() || "#6ee7b7",
        pointClosed: styles.getPropertyValue("--point-closed").trim() || "#a78bfa",
        selected: styles.getPropertyValue("--selected").trim() || "#f59e0b",
        accent: styles.getPropertyValue("--accent").trim() || "#38bdf8",
        muted: styles.getPropertyValue("--muted").trim() || "#94a3b8"
      };
      
      let apiKey = null;
      
      const hasStates = data.some(p => p.state);
      const hasTypes = data.some(p => p.type);
      const hasEmbeddings = data.some(p => p.embedding);
      
      if (!hasEmbeddings) {
        document.getElementById("search-wrap").style.display = "none";
      }
      
      const getPointColor = (point, isSelected) => {
        if (isSelected) return colors.selected;
        if (!hasStates) return colors.point;
        if (point.state === "open") return colors.pointOpen;
        if (point.state === "closed") return colors.pointClosed;
        return colors.point;
      };
      
      const isVisible = (point) => {
        const typeOk = !hasTypes || 
          (point.type === "pr" && filterPr.checked) || 
          (point.type === "issue" && filterIssue.checked) ||
          (!point.type);
        const stateOk = !hasStates ||
          (point.state === "open" && filterOpen.checked) ||
          (point.state === "closed" && filterClosed.checked) ||
          (!point.state);
        return typeOk && stateOk;
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
        mouseDownPos: { x: 0, y: 0 },
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
        // Cull points behind camera
        if (z2 < -0.9) {
          return { x: -1000, y: -1000, depth: z2, culled: true };
        }
        const perspective = 1 / (1 + z2 * 0.9);
        const screenX = x1 * perspective * canvas.clientWidth * 0.7 + canvas.clientWidth * 0.5 + view3d.offsetX;
        const screenY = y1 * perspective * canvas.clientHeight * 0.7 + canvas.clientHeight * 0.5 + view3d.offsetY;
        return { x: screenX, y: screenY, depth: z2, culled: false };
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
          const header = document.createElement("div");
          header.className = "item-header";
          const link = document.createElement("a");
          link.href = item.url;
          link.target = "_blank";
          link.rel = "noreferrer";
          const num = item.number ? "#" + item.number + " " : "";
          link.textContent = num + (item.title || item.url);
          header.appendChild(link);
          if (item.type) {
            const typeBadge = document.createElement("span");
            typeBadge.className = "badge badge-" + item.type;
            typeBadge.textContent = item.type;
            header.appendChild(typeBadge);
          }
          if (item.state) {
            const stateBadge = document.createElement("span");
            stateBadge.className = "badge badge-" + item.state;
            stateBadge.textContent = item.state;
            header.appendChild(stateBadge);
          }
          li.appendChild(header);
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
          if (!isVisible(data[i])) continue;
          const screen = getScreenPoint(data[i]);
          if (screen.x >= left && screen.x <= right && screen.y >= top && screen.y <= bottom) {
            state.selected.add(i);
          }
        }
        updateSidebar();
      };

      const hitTestPoint = (clickX, clickY, radius) => {
        for (let i = 0; i < data.length; i += 1) {
          if (!isVisible(data[i])) continue;
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
      
      const drawRing = (x, y, size) => {
        ctx.beginPath();
        ctx.arc(x, y, size, 0, Math.PI * 2);
        ctx.lineWidth = 1.5;
        ctx.stroke();
      };

      const render = () => {
        ctx.clearRect(0, 0, canvas.clientWidth, canvas.clientHeight);
        const projected = [];
        let visibleCount = 0;
        for (let i = 0; i < data.length; i += 1) {
          if (!isVisible(data[i])) continue;
          visibleCount++;
          const screen = getScreenPoint(data[i]);
          projected.push({ index: i, screen });
        }
        if (state.mode === "3d") {
          projected.sort((a, b) => a.screen.depth - b.screen.depth);
        }
        for (const item of projected) {
          if (item.screen.culled) continue;
          const point = data[item.index];
          const isSelected = state.selected.has(item.index);
          const size = isSelected ? 4 : 2.5;
          ctx.fillStyle = getPointColor(point, isSelected);
          if (point.type === "issue") {
            ctx.strokeStyle = getPointColor(point, isSelected);
            drawRing(item.screen.x, item.screen.y, size + 1);
          } else {
            ctx.beginPath();
            ctx.arc(item.screen.x, item.screen.y, size, 0, Math.PI * 2);
            ctx.fill();
          }
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
        hudCount.textContent = visibleCount + " / " + data.length + " items";
      };

      canvas.addEventListener("mousedown", (event) => {
        if (event.button !== 0) return;
        const point = getCanvasPoint(event);
        const ctrlOrMeta = event.ctrlKey || event.metaKey;
        state.mouseDownPos = { x: point.x, y: point.y };
        
        if (event.shiftKey) {
          state.selecting = true;
          state.addToSelection = ctrlOrMeta;
          state.selectRect = {
            x0: point.x,
            y0: point.y,
            x1: point.x,
            y1: point.y
          };
          scheduleRender();
          return;
        }
        
        if (ctrlOrMeta) {
          if (state.mode === "2d") {
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
        const wasSelecting = state.selecting;
        const totalDist = Math.hypot(point.x - state.mouseDownPos.x, point.y - state.mouseDownPos.y);
        
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
        
        // Only deselect on actual click (no movement), not after drag
        if (event.target !== canvas) return;
        if (wasSelecting) return;
        
        if (totalDist < 3) {
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
      
      [filterPr, filterIssue, filterOpen, filterClosed].forEach(el => {
        el.addEventListener("change", scheduleRender);
      });
      
      openAllBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        const selected = Array.from(state.selected);
        for (const index of selected) {
          window.open(data[index].url, "_blank");
        }
      });
      
      copyBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        const selected = Array.from(state.selected);
        const lines = selected.map(index => {
          const item = data[index];
          const num = item.number ? "#" + item.number : "";
          const type = item.type ? "[" + item.type.toUpperCase() + "]" : "";
          return type + " " + num + " " + item.title + "\\n" + item.url;
        });
        navigator.clipboard.writeText(lines.join("\\n\\n"));
      });
      
      const doSearch = async () => {
        const query = searchInput.value.trim();
        if (!query) return;
        
        if (!apiKey) {
          apiKey = prompt("Enter your OpenAI API key (session only, not stored):");
          if (!apiKey) return;
        }
        
        searchBtn.disabled = true;
        searchBtn.textContent = "...";
        
        try {
          const response = await fetch("https://api.openai.com/v1/embeddings", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              "Authorization": "Bearer " + apiKey
            },
            body: JSON.stringify({
              model: "text-embedding-3-small",
              input: query
            })
          });
          
          if (!response.ok) {
            const err = await response.text();
            throw new Error(err);
          }
          
          const result = await response.json();
          const queryEmb = result.data[0].embedding;
          
          // Compute cosine similarity with all points
          const similarities = data.map((point, index) => {
            if (!point.embedding) return { index, sim: -1 };
            let dot = 0, normA = 0, normB = 0;
            for (let i = 0; i < queryEmb.length; i++) {
              dot += queryEmb[i] * point.embedding[i];
              normA += queryEmb[i] * queryEmb[i];
              normB += point.embedding[i] * point.embedding[i];
            }
            const sim = dot / (Math.sqrt(normA) * Math.sqrt(normB));
            return { index, sim };
          });
          
          similarities.sort((a, b) => b.sim - a.sim);
          
          state.selected.clear();
          const topN = Math.min(20, similarities.length);
          for (let i = 0; i < topN; i++) {
            if (similarities[i].sim > 0) {
              state.selected.add(similarities[i].index);
            }
          }
          
          updateSidebar();
          scheduleRender();
        } catch (err) {
          alert("Search failed: " + err.message);
          apiKey = null;
        } finally {
          searchBtn.disabled = false;
          searchBtn.textContent = "Search";
        }
      };
      
      searchBtn.addEventListener("click", doSearch);
      searchInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") doSearch();
      });

      canvas.addEventListener("contextmenu", (event) => {
        event.preventDefault();
      });

      canvas.addEventListener("wheel", (event) => {
        event.preventDefault();
        const point = getCanvasPoint(event);
        const zoomFactor = Math.exp(-event.deltaY * 0.001);
        if (state.mode === "3d") {
          view3d.zoom = clamp(view3d.zoom * zoomFactor, 0.4, 5);
          scheduleRender();
          return;
        }
        const prevScale = view2d.scale;
        view2d.scale = clamp(view2d.scale * zoomFactor, 0.2, 12);
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
		projections: "projections.json",
		neighbors: 15,
		minDist: 0.1,
		spread: 1.0,
		includeEmbeddings: false,
		force: false,
		pcaDims: undefined,
	};

	for (let i = 0; i < args.length; i += 1) {
		const arg = args[i];
		if (arg === "--input") {
			options.input = args[++i];
		} else if (arg === "--output") {
			options.output = args[++i];
		} else if (arg === "--projections") {
			options.projections = args[++i];
		} else if (arg === "--neighbors") {
			options.neighbors = Number(args[++i]);
		} else if (arg === "--min-dist") {
			options.minDist = Number(args[++i]);
		} else if (arg === "--spread") {
			options.spread = Number(args[++i]);
		} else if (arg === "--force") {
			options.force = true;
		} else if (arg === "--pca-dims") {
			const val = Number(args[++i]);
			options.pcaDims = Number.isNaN(val) ? undefined : val;
		} else if (arg === "--search") {
			options.includeEmbeddings = true;
		}
	}

	build(options);
}

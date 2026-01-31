# Doppelgangers

Find duplicate PRs through embedding visualization. Fetches open PRs from a GitHub repo, generates embeddings, and renders an interactive 2D/3D scatter plot where similar PRs cluster together.

## Install

```bash
npm install -g doppelgangers
```

Or run directly:

```bash
npx doppelgangers --repo owner/repo
```

## Usage

```bash
export OPENAI_API_KEY=...
doppelgangers --repo https://github.com/facebook/react
```

This will:
1. Fetch all open PRs from the repo
2. Generate embeddings using OpenAI
3. Project them into 2D and 3D using UMAP
4. Output `triage.html` with an interactive viewer

## Options

```
--repo <url|owner/repo>   GitHub repository (required)
--output <path>           Output path for PRs JSON (default: prs.json)
--embeddings <path>       Output path for embeddings (default: embeddings.jsonl)
--html <path>             Output path for HTML viewer (default: triage.html)
--model <model>           OpenAI embedding model (default: text-embedding-3-small)
--batch <n>               Batch size for embeddings (default: 100)
--max-chars <n>           Max chars for embedding input (default: 4000)
--body-chars <n>          Max chars for body snippet (default: 2000)
--neighbors <n>           UMAP neighbors (default: 15)
--min-dist <n>            UMAP min distance (default: 0.1)
```

## Viewer Controls

**2D Mode:**
- Drag to pan
- Scroll to zoom
- Shift+drag to select (replaces selection)
- Ctrl/Cmd+drag to select (adds to selection)
- Click empty space to deselect

**3D Mode:**
- Drag to rotate
- Ctrl/Cmd+drag to pan
- Scroll to zoom
- Shift+drag to select (replaces selection)
- Click empty space to deselect

## Requirements

- Node.js 20+
- `gh` CLI (authenticated)
- `OPENAI_API_KEY` environment variable

## License

MIT

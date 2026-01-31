# Doppelgangers

Sloppy tools for sloppy times.

https://github.com/user-attachments/assets/60212797-8caa-4a8c-aa6b-a39cf86bed99

Explore all the slop issues and PRs of your GitHub repos visually. Triage them manually or give clusters to your clanker to clean things up.

Fetches issues/PRs, generates embeddings, projects them to 2D/3D via UMAP. Similar items cluster together. Select a cluster, open all, close the dupes.

**Demos:** [openai/codex](https://mariozechner.at/uploads/codex.html) | [sst/opencode](https://mariozechner.at/uploads/opencode.html) | [openclaw/openclaw](https://mariozechner.at/uploads/openclaw.html)

## Install

```bash
npm install -g doppelgangers
```

## Usage

```bash
export OPENAI_API_KEY=...
doppelgangers --repo facebook/react
```

This will:
1. Fetch all open issues and PRs from the repo
2. Generate embeddings using OpenAI
3. Project to 2D/3D using UMAP
4. Output `triage.html` with an interactive viewer

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `--repo <url\|owner/repo>` | GitHub repository (required) | |
| `--state <state>` | `open`, `closed`, or `all` | `open` |
| `--type <type>` | `pr`, `issue`, or `all` | `all` |
| `--output <path>` | Items JSON path | `prs.json` |
| `--embeddings <path>` | Embeddings path | `embeddings.jsonl` |
| `--html <path>` | HTML viewer path | `triage.html` |
| `--model <model>` | OpenAI embedding model | `text-embedding-3-small` |
| `--batch <n>` | Batch size for embeddings | `100` |
| `--max-chars <n>` | Max chars for embedding input | `4000` |
| `--body-chars <n>` | Max chars for body snippet | `2000` |
| `--neighbors <n>` | UMAP neighbors | `15` |
| `--min-dist <n>` | UMAP min distance | `0.1` |
| `--search` | Include embeddings for semantic search | `false` |

## Viewer

**Controls:**
- **2D:** Drag to pan, scroll to zoom
- **3D:** Drag to rotate, Ctrl/Cmd+drag to pan, scroll to zoom
- **Select:** Shift+drag (Ctrl/Cmd+Shift to add to selection)
- **Deselect:** Click empty space

**Sidebar:**
- "Open All" opens selected items in new tabs (allow popups)
- "Copy" copies selection as formatted list

**Visual Encoding:**
- Filled circles = PRs, Hollow rings = Issues
- Green = Open, Purple = Closed, Orange = Selected

**Filters:** Toggle PRs/Issues and Open/Closed visibility

## Requirements

- Node.js 20+
- `gh` CLI (authenticated)
- `OPENAI_API_KEY` environment variable

## License

MIT

#!/usr/bin/env node

import { spawn } from "child_process";
import fs from "fs";
import path from "path";
import readline from "readline";
import { type BuildOptions, build } from "./build.js";
import { type EmbedOptions, embed } from "./embed.js";

type ItemState = "open" | "closed" | "all";
type ItemType = "pr" | "issue" | "all";

interface TriageOptions {
	repo: string | null;
	state: ItemState;
	type: ItemType;
	output: string;
	embeddings: string;
	html: string;
	model: string;
	batch: number;
	maxChars: number;
	bodyChars: number;
	neighbors: number;
	minDist: number;
	search: boolean;
}

function parseRepo(repo: string): { owner: string; name: string } | null {
	const trimmed = repo.replace(/\s+/g, "");
	const match = trimmed.match(/github\.com[/:]([^/]+)\/([^/.]+)(?:\.git)?/i);
	if (match) {
		return { owner: match[1], name: match[2] };
	}
	if (trimmed.includes("/")) {
		const [owner, name] = trimmed.split("/");
		if (owner && name) return { owner, name };
	}
	return null;
}

interface FetchResult {
	total: number;
	prs: number;
	issues: number;
}

async function fetchItems(
	owner: string,
	name: string,
	state: ItemState,
	type: ItemType,
	outputPath: string,
): Promise<FetchResult> {
	fs.mkdirSync(path.dirname(outputPath), { recursive: true });

	const repoLabel = `${owner}/${name}`;
	const items: object[] = [];
	let prCount = 0;
	let issueCount = 0;

	const fetchEndpoint = async (endpoint: string, itemType: "pr" | "issue") => {
		const jqFilter =
			itemType === "issue"
				? '.[] | select(.pull_request == null) | {url: .html_url, number: .number, title: .title, body: .body, state: .state, type: "issue"}'
				: '.[] | {url: .html_url, number: .number, title: .title, body: .body, state: .state, type: "pr"}';

		const gh = spawn("gh", ["api", "--paginate", endpoint, "--jq", jqFilter], {
			stdio: ["ignore", "pipe", "inherit"],
		});

		const rl = readline.createInterface({
			input: gh.stdout!,
			crlfDelay: Number.POSITIVE_INFINITY,
		});

		const readPromise = (async () => {
			for await (const line of rl) {
				const trimmed = line.trim();
				if (!trimmed) continue;
				try {
					const item = JSON.parse(trimmed);
					items.push(item);
					if (itemType === "pr") prCount++;
					else issueCount++;
					const total = prCount + issueCount;
					if (total % 200 === 0) {
						console.log(`[${repoLabel}] Fetched ${prCount} PRs, ${issueCount} issues`);
					}
				} catch {
					// skip invalid JSON
				}
			}
		})();

		const exitPromise = new Promise<void>((resolve, reject) => {
			gh.on("close", (code) => {
				if (code === 0) resolve();
				else reject(new Error(`gh api ${endpoint} exited with code ${code}`));
			});
			gh.on("error", reject);
		});

		await Promise.all([readPromise, exitPromise]);
	};

	if (type === "pr" || type === "all") {
		const prEndpoint = `/repos/${owner}/${name}/pulls?state=${state}&per_page=100`;
		await fetchEndpoint(prEndpoint, "pr");
	}

	if (type === "issue" || type === "all") {
		const issueEndpoint = `/repos/${owner}/${name}/issues?state=${state}&per_page=100`;
		await fetchEndpoint(issueEndpoint, "issue");
	}

	fs.writeFileSync(outputPath, JSON.stringify(items, null, 2));

	return { total: items.length, prs: prCount, issues: issueCount };
}

async function main() {
	const args = process.argv.slice(2);
	const options: TriageOptions = {
		repo: null,
		state: "open",
		type: "all",
		output: "prs.json",
		embeddings: "embeddings.jsonl",
		html: "triage.html",
		model: "text-embedding-3-small",
		batch: 100,
		maxChars: 4000,
		bodyChars: 2000,
		neighbors: 15,
		minDist: 0.1,
		search: false,
	};

	for (let i = 0; i < args.length; i += 1) {
		const arg = args[i];
		if (arg === "--repo") {
			options.repo = args[++i];
		} else if (arg === "--state") {
			const val = args[++i];
			if (val !== "open" && val !== "closed" && val !== "all") {
				console.error("--state must be open, closed, or all");
				process.exit(1);
			}
			options.state = val;
		} else if (arg === "--type") {
			const val = args[++i];
			if (val !== "pr" && val !== "issue" && val !== "all") {
				console.error("--type must be pr, issue, or all");
				process.exit(1);
			}
			options.type = val;
		} else if (arg === "--output") {
			options.output = args[++i];
		} else if (arg === "--embeddings") {
			options.embeddings = args[++i];
		} else if (arg === "--html") {
			options.html = args[++i];
		} else if (arg === "--model") {
			options.model = args[++i];
		} else if (arg === "--batch") {
			options.batch = Number(args[++i]);
		} else if (arg === "--max-chars") {
			options.maxChars = Number(args[++i]);
		} else if (arg === "--body-chars") {
			options.bodyChars = Number(args[++i]);
		} else if (arg === "--neighbors") {
			options.neighbors = Number(args[++i]);
		} else if (arg === "--min-dist") {
			options.minDist = Number(args[++i]);
		} else if (arg === "--search") {
			options.search = true;
		} else if (arg === "--help" || arg === "-h") {
			console.log(`
doppelgangers - Find duplicate PRs through embedding visualization

Usage:
  doppelgangers --repo <owner/repo>

Options:
  --repo <url|owner/repo>   GitHub repository (required)
  --state <state>           Item state: open, closed, or all (default: open)
  --type <type>             Item type: pr, issue, or all (default: all)
  --output <path>           Output path for items JSON (default: prs.json)
  --embeddings <path>       Output path for embeddings (default: embeddings.jsonl)
  --html <path>             Output path for HTML viewer (default: triage.html)
  --model <model>           OpenAI embedding model (default: text-embedding-3-small)
  --batch <n>               Batch size for embeddings (default: 100)
  --max-chars <n>           Max chars for embedding input (default: 4000)
  --body-chars <n>          Max chars for body snippet (default: 2000)
  --neighbors <n>           UMAP neighbors (default: 15)
  --min-dist <n>            UMAP min distance (default: 0.1)
  --search                  Include embeddings for semantic search (increases file size)

Environment:
  OPENAI_API_KEY            Required for embedding generation
`);
			process.exit(0);
		}
	}

	if (!options.repo) {
		console.error("--repo is required. Use --help for usage.");
		process.exit(1);
	}

	const repoInfo = parseRepo(options.repo);
	if (!repoInfo) {
		console.error("Could not parse repo. Use https://github.com/org/repo or org/repo");
		process.exit(1);
	}

	const { owner, name } = repoInfo;
	const typeLabel = options.type === "all" ? "PRs and issues" : options.type === "pr" ? "PRs" : "issues";
	console.log(`Fetching ${options.state} ${typeLabel} for ${owner}/${name}`);

	const outputPath = path.resolve(options.output);
	const result = await fetchItems(owner, name, options.state, options.type, outputPath);
	console.log(`Wrote ${outputPath} (${result.prs} PRs, ${result.issues} issues)`);

	const embedOptions: EmbedOptions = {
		input: outputPath,
		output: path.resolve(options.embeddings),
		model: options.model,
		batchSize: options.batch,
		maxChars: options.maxChars,
		bodyChars: options.bodyChars,
		resume: false,
	};
	await embed(embedOptions);

	const embeddingsPath = path.resolve(options.embeddings);
	const projectionsPath = embeddingsPath.replace(/\.[^.]+$/, "-projections.json");

	const buildOptions: BuildOptions = {
		input: embeddingsPath,
		output: path.resolve(options.html),
		projections: projectionsPath,
		neighbors: options.neighbors,
		minDist: options.minDist,
		includeEmbeddings: options.search,
	};
	build(buildOptions);

	console.log(`Done. Open ${path.resolve(options.html)}`);
}

main().catch((error) => {
	console.error(error);
	process.exit(1);
});

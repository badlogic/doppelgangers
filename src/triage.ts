#!/usr/bin/env node

import { spawn } from "child_process";
import fs from "fs";
import path from "path";
import readline from "readline";
import { type BuildOptions, build } from "./build.js";
import { type EmbedOptions, embed } from "./embed.js";

interface TriageOptions {
	repo: string | null;
	output: string;
	embeddings: string;
	html: string;
	model: string;
	batch: number;
	maxChars: number;
	bodyChars: number;
	neighbors: number;
	minDist: number;
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

async function fetchPRs(owner: string, name: string, outputPath: string): Promise<number> {
	const apiPath = `/repos/${owner}/${name}/pulls?state=open&per_page=100`;

	fs.mkdirSync(path.dirname(outputPath), { recursive: true });

	const gh = spawn(
		"gh",
		["api", "--paginate", apiPath, "--jq", ".[] | {url: .html_url, title: .title, body: .body}"],
		{
			stdio: ["ignore", "pipe", "inherit"],
		},
	);

	const outputStream = fs.createWriteStream(outputPath, { flags: "w" });
	outputStream.write("[\n");
	let first = true;
	let fetched = 0;

	const rl = readline.createInterface({
		input: gh.stdout!,
		crlfDelay: Number.POSITIVE_INFINITY,
	});

	const readPromise = (async () => {
		for await (const line of rl) {
			const trimmed = line.trim();
			if (!trimmed) continue;
			if (!first) outputStream.write(",\n");
			outputStream.write(trimmed);
			first = false;
			fetched += 1;
			if (fetched % 200 === 0) {
				console.log(`Fetched ${fetched} PRs`);
			}
		}
	})();

	const exitPromise = new Promise<void>((resolve, reject) => {
		gh.on("close", (code) => {
			if (code === 0) resolve();
			else reject(new Error(`gh api exited with code ${code}`));
		});
		gh.on("error", reject);
	});

	try {
		await Promise.all([readPromise, exitPromise]);
	} catch (error) {
		outputStream.end();
		throw error;
	}

	outputStream.write("\n]\n");
	outputStream.end();

	return fetched;
}

async function main() {
	const args = process.argv.slice(2);
	const options: TriageOptions = {
		repo: null,
		output: "prs.json",
		embeddings: "embeddings.jsonl",
		html: "triage.html",
		model: "text-embedding-3-small",
		batch: 100,
		maxChars: 4000,
		bodyChars: 2000,
		neighbors: 15,
		minDist: 0.1,
	};

	for (let i = 0; i < args.length; i += 1) {
		const arg = args[i];
		if (arg === "--repo") {
			options.repo = args[++i];
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
		} else if (arg === "--help" || arg === "-h") {
			console.log(`
doppelgangers - Find duplicate PRs through embedding visualization

Usage:
  doppelgangers --repo <owner/repo>

Options:
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
	console.log(`Fetching PRs for ${owner}/${name}`);

	const outputPath = path.resolve(options.output);
	const fetched = await fetchPRs(owner, name, outputPath);
	console.log(`Wrote ${outputPath} (${fetched} PRs)`);

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

	const buildOptions: BuildOptions = {
		input: path.resolve(options.embeddings),
		output: path.resolve(options.html),
		neighbors: options.neighbors,
		minDist: options.minDist,
	};
	build(buildOptions);

	console.log(`Done. Open ${path.resolve(options.html)}`);
}

main().catch((error) => {
	console.error(error);
	process.exit(1);
});

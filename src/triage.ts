#!/usr/bin/env node

import { spawn } from "child_process";
import fs from "fs";
import path from "path";
import readline from "readline";
import { type BuildOptions, build } from "./build.js";
import { type EmbedOptions, embed, type Item } from "./embed.js";

type ItemState = "open" | "closed" | "all";
type ItemType = "pr" | "issue" | "all";

interface TriageOptions {
	repo: string | null;
	state: ItemState;
	type: ItemType;
	since?: string;
	output: string;
	embeddings: string;
	html: string;
	model: string;
	batch: number;
	maxChars: number;
	bodyChars: number;
	neighbors: number;
	minDist: number;
	spread: number;
	force: boolean;
	search: boolean;
	localModel?: string;
}

interface SinceFilter {
	raw: string;
	cutoff: Date;
	cutoffYmd: string;
}

function formatLocalYmd(date: Date): string {
	const y = date.getFullYear();
	const m = String(date.getMonth() + 1).padStart(2, "0");
	const d = String(date.getDate()).padStart(2, "0");
	return `${y}-${m}-${d}`;
}

function parseSinceValue(value: string): SinceFilter {
	const relativeMatch = value.match(/^(\d+)d$/i);
	if (relativeMatch) {
		const days = Number(relativeMatch[1]);
		if (!Number.isInteger(days) || days < 1) {
			throw new Error("--since relative format must be a positive day count like 14d");
		}
		const cutoff = new Date();
		cutoff.setHours(0, 0, 0, 0);
		cutoff.setDate(cutoff.getDate() - days);
		return {
			raw: value,
			cutoff,
			cutoffYmd: formatLocalYmd(cutoff),
		};
	}

	const absoluteMatch = value.match(/^(\d{4})-(\d{2})-(\d{2})$/);
	if (absoluteMatch) {
		const year = Number(absoluteMatch[1]);
		const month = Number(absoluteMatch[2]);
		const day = Number(absoluteMatch[3]);
		const cutoff = new Date(year, month - 1, day, 0, 0, 0, 0);
		if (cutoff.getFullYear() !== year || cutoff.getMonth() !== month - 1 || cutoff.getDate() !== day) {
			throw new Error(`Invalid --since date: ${value}. Use YYYY-MM-DD.`);
		}
		return {
			raw: value,
			cutoff,
			cutoffYmd: formatLocalYmd(cutoff),
		};
	}

	throw new Error("Invalid --since value. Use YYYY-MM-DD or <days>d (e.g., 14d, 30d)");
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
	sinceFilter?: SinceFilter,
): Promise<FetchResult> {
	fs.mkdirSync(path.dirname(outputPath), { recursive: true });

	const repoLabel = `${owner}/${name}`;
	const items: Item[] = [];
	let prCount = 0;
	let issueCount = 0;

	// Helper to stream JSON objects from gh command
	const streamGhJson = async (args: string[], onData: (item: any) => void) => {
		const gh = spawn("gh", args, {
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
					onData(item);
				} catch {
					// skip invalid JSON
				}
			}
		})();

		const exitPromise = new Promise<void>((resolve, reject) => {
			gh.on("close", (code) => {
				if (code === 0) resolve();
				else reject(new Error(`gh exited with code ${code}`));
			});
			gh.on("error", reject);
		});

		await Promise.all([readPromise, exitPromise]);
	};

	const fetchWithSinceSearch = async (itemType: "pr" | "issue") => {
		if (!sinceFilter) return;

		const statesToFetch: Array<"open" | "closed"> = state === "all" ? ["open", "closed"] : [state];

		for (const stateValue of statesToFetch) {
			const jsonFields =
				itemType === "pr" ? "number,title,body,url,state" : "number,title,body,url,state,isPullRequest";
			const jqFilter =
				itemType === "pr"
					? '.[] | {url, number, title, body, state: (.state | ascii_downcase), type: "pr"}'
					: '.[] | select(.isPullRequest == false) | {url, number, title, body, state: (.state | ascii_downcase), type: "issue"}';

			const args = [
				"search",
				itemType === "pr" ? "prs" : "issues",
				"--repo",
				repoLabel,
				"--state",
				stateValue,
				"--created",
				`>=${sinceFilter.cutoffYmd}`,
				"--sort",
				"created",
				"--order",
				"desc",
				"--limit",
				"1000",
				"--json",
				jsonFields,
				"--jq",
				jqFilter,
			];

			await streamGhJson(args, (item) => {
				items.push(item);
				if (itemType === "pr") prCount++;
				else issueCount++;
				const total = prCount + issueCount;
				if (total % 200 === 0) {
					console.log(`[${repoLabel}] Fetched ${prCount} PRs, ${issueCount} issues`);
				}
			});
		}

		console.log(`[${repoLabel}] Fetched ${prCount} PRs, ${issueCount} issues`);
	};

	// Helper for GraphQL pagination of PRs
	const fetchPrsGraphql = async () => {
		const graphqlStates =
			state === "open" ? "[OPEN]" : state === "closed" ? "[CLOSED, MERGED]" : "[OPEN, CLOSED, MERGED]";

		const graphqlQuery = `
			query($owner: String!, $name: String!, $endCursor: String) {
				repository(owner: $owner, name: $name) {
					pullRequests(first: 100, after: $endCursor, states: ${graphqlStates}) {
						pageInfo {
							hasNextPage
							endCursor
						}
						nodes {
							number
							title
							body
							url
							state
							files(first: 20) {
								nodes {
									path
								}
							}
						}
					}
				}
			}
		`;

		// .data.repository.pullRequests.nodes[] | {number, title, body, url, state: .state | ascii_downcase, type: "pr", files: [.files.nodes[].path]}
		const jqFilter =
			'.data.repository.pullRequests.nodes[] | {number, title, body, url, state: (.state | ascii_downcase), type: "pr", files: [.files.nodes[].path]}';

		const args = [
			"api",
			"graphql",
			"--paginate",
			"-f",
			`query=${graphqlQuery}`,
			"-F",
			`owner=${owner}`,
			"-F",
			`name=${name}`,
			"--jq",
			jqFilter,
		];

		await streamGhJson(args, (item) => {
			if (item.state === "merged") item.state = "closed";
			items.push(item);
			prCount++;
			const total = prCount + issueCount;
			if (total % 100 === 0) {
				console.log(`[${repoLabel}] Fetched ${prCount} PRs (with files), ${issueCount} issues`);
			}
		});
	};

	// Helper for REST fetching of issues
	const fetchEndpoint = async (endpoint: string, itemType: "pr" | "issue") => {
		const jqFilter =
			itemType === "issue"
				? '.[] | select(.pull_request == null) | {url: .html_url, number: .number, title: .title, body: .body, state: .state, type: "issue"}'
				: '.[] | {url: .html_url, number: .number, title: .title, body: .body, state: .state, type: "pr"}';

		const args = ["api", "--paginate", endpoint, "--jq", jqFilter];

		await streamGhJson(args, (item) => {
			items.push(item);
			if (itemType === "pr") prCount++;
			else issueCount++;
			const total = prCount + issueCount;
			if (total % 200 === 0) {
				console.log(`[${repoLabel}] Fetched ${prCount} PRs, ${issueCount} issues`);
			}
		});
	};

	if (type === "pr" || type === "all") {
		if (sinceFilter) {
			await fetchWithSinceSearch("pr");
		} else {
			await fetchPrsGraphql();
		}
	}

	if (type === "issue" || type === "all") {
		if (sinceFilter) {
			await fetchWithSinceSearch("issue");
		} else {
			const issueEndpoint = `/repos/${owner}/${name}/issues?state=${state}&per_page=100`;
			await fetchEndpoint(issueEndpoint, "issue");
		}
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
		since: undefined,
		output: "prs.json",
		embeddings: "embeddings.jsonl",
		html: "triage.html",
		model: "text-embedding-3-small",
		batch: 100,
		maxChars: 4000,
		bodyChars: 2000,
		neighbors: 15,
		minDist: 0.1,
		spread: 1.0,
		force: false,
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
		} else if (arg === "--since") {
			if (options.since !== undefined) {
				console.error("--since can only be provided once");
				process.exit(1);
			}
			const val = args[++i];
			if (!val) {
				console.error("--since requires a value (YYYY-MM-DD or <days>d, e.g. 14d)");
				process.exit(1);
			}
			options.since = val;
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
			const val = Number(args[++i]);
			options.neighbors = Number.isNaN(val) ? 15 : val;
		} else if (arg === "--min-dist") {
			const val = Number(args[++i]);
			options.minDist = Number.isNaN(val) ? 0.1 : val;
		} else if (arg === "--spread") {
			const val = Number(args[++i]);
			options.spread = Number.isNaN(val) ? 1.0 : val;
		} else if (arg === "--force") {
			options.force = true;
		} else if (arg === "--search") {
			options.search = true;
		} else if (arg === "--local-model") {
			options.localModel = args[++i];
		} else if (arg === "--help" || arg === "-h") {
			console.log(`
doppelgangers - Find duplicate PRs through embedding visualization

Usage:
  doppelgangers --repo <owner/repo>

Options:
  --repo <url|owner/repo>   GitHub repository (required)
  --state <state>           Item state: open, closed, or all (default: open)
  --type <type>             Item type: pr, issue, or all (default: all)
  --since <value>           Created-date cutoff (YYYY-MM-DD or <days>d, e.g. 14d)
  --output <path>           Output path for items JSON (default: prs.json)
  --embeddings <path>       Output path for embeddings (default: embeddings.jsonl)
  --html <path>             Output path for HTML viewer (default: triage.html)
  --model <model>           OpenAI embedding model (default: text-embedding-3-small)
  --batch <n>               Batch size for embeddings (default: 100)
  --max-chars <n>           Max chars for embedding input (default: 4000)
  --body-chars <n>          Max chars for body snippet (default: 2000)
  --neighbors <n>           UMAP neighbors (default: 15)
  --min-dist <n>            UMAP min distance (default: 0.1)
  --spread <n>              UMAP spread (default: 1.0)
  --force                   Force re-calculation of projections
  --search                  Include embeddings for semantic search (increases file size)
  --local-model <path>      Path to local GGUF model for embeddings (optional)

Environment:
  OPENAI_API_KEY            Required for embedding generation (unless --local-model is used)
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

	let sinceFilter: SinceFilter | undefined;
	if (options.since) {
		try {
			sinceFilter = parseSinceValue(options.since);
		} catch (error) {
			console.error((error as Error).message);
			process.exit(1);
		}
	}

	const { owner, name } = repoInfo;
	const typeLabel = options.type === "all" ? "PRs and issues" : options.type === "pr" ? "PRs" : "issues";
	console.log(`Fetching ${options.state} ${typeLabel} for ${owner}/${name}`);
	if (sinceFilter) {
		console.log(`Filtering by created date since ${sinceFilter.cutoffYmd} (from --since ${sinceFilter.raw})`);
	}

	const outputPath = path.resolve(options.output);
	const result = await fetchItems(owner, name, options.state, options.type, outputPath, sinceFilter);
	console.log(`Wrote ${outputPath} (${result.prs} PRs, ${result.issues} issues)`);

	const embedOptions: EmbedOptions = {
		input: outputPath,
		output: path.resolve(options.embeddings),
		model: options.model,
		batchSize: options.batch,
		maxChars: options.maxChars,
		bodyChars: options.bodyChars,
		resume: false,
		localModel: options.localModel,
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
		spread: options.spread,
		force: options.force,
		includeEmbeddings: options.search,
	};
	await build(buildOptions);

	console.log(`Done. Open ${path.resolve(options.html)}`);
}

main().catch((error) => {
	console.error(error);
	process.exit(1);
});

import fs from "fs";
import OpenAI from "openai";
import path from "path";

export interface EmbedOptions {
	input: string;
	output: string;
	model: string;
	batchSize: number;
	maxChars: number;
	bodyChars: number;
	resume: boolean;
}

interface PR {
	url: string;
	title: string;
	body: string | null;
}

interface EmbeddingRecord {
	url: string;
	title: string;
	body: string;
	embedding: number[];
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

const buildText = (title: string, body: string | null, maxChars: number): string => {
	const bodyText = (body || "").replace(/\r\n/g, "\n").trim();
	const combined = `${title}\n\n${bodyText}`.trim();
	return combined.slice(0, maxChars);
};

const buildSnippet = (body: string | null, bodyChars: number): string => {
	if (!body) return "";
	return body.replace(/\s+/g, " ").trim().slice(0, bodyChars);
};

export async function embed(options: EmbedOptions): Promise<void> {
	if (!process.env.OPENAI_API_KEY) {
		console.error("OPENAI_API_KEY is required");
		process.exit(1);
	}

	const inputPath = path.resolve(options.input);
	const outputPath = path.resolve(options.output);

	const pulls: PR[] = JSON.parse(fs.readFileSync(inputPath, "utf8"));

	const existing = new Map<string, EmbeddingRecord>();
	if (options.resume && fs.existsSync(outputPath)) {
		const lines = fs.readFileSync(outputPath, "utf8").split("\n").filter(Boolean);
		for (const line of lines) {
			try {
				const item: EmbeddingRecord = JSON.parse(line);
				if (item.url) existing.set(item.url, item);
			} catch {
				// skip invalid lines
			}
		}
	}

	const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

	const outputDir = path.dirname(outputPath);
	fs.mkdirSync(outputDir, { recursive: true });
	const outputStream = fs.createWriteStream(outputPath, { flags: options.resume ? "a" : "w" });

	const pending = pulls.filter((pr) => pr?.url && !existing.has(pr.url));
	const total = pending.length;
	let processed = 0;
	const skipped = pulls.length - pending.length;

	let batchInputs: string[] = [];
	let batchMeta: { url: string; title: string; body: string }[] = [];

	const createEmbeddings = async (inputs: string[], attempt = 1): Promise<number[][]> => {
		try {
			const response = await client.embeddings.create({
				model: options.model,
				input: inputs,
			});
			return response.data.map((item) => item.embedding);
		} catch (error) {
			if (attempt >= 5) throw error;
			const delay = 1000 * attempt;
			console.warn(`Embedding request failed, retrying in ${delay}ms`, (error as Error).message);
			await sleep(delay);
			return createEmbeddings(inputs, attempt + 1);
		}
	};

	const flushBatch = async () => {
		if (!batchInputs.length) return;
		const embeddings = await createEmbeddings(batchInputs);
		for (let i = 0; i < embeddings.length; i += 1) {
			const meta = batchMeta[i];
			const record: EmbeddingRecord = {
				url: meta.url,
				title: meta.title,
				body: meta.body,
				embedding: embeddings[i],
			};
			outputStream.write(`${JSON.stringify(record)}\n`);
			processed += 1;
			if (processed % 50 === 0 || processed === total) {
				console.log(`Embedded ${processed}/${total} PRs`);
			}
		}
		batchInputs = [];
		batchMeta = [];
	};

	for (const pr of pending) {
		const title = pr.title || "";
		const body = pr.body || "";
		const text = buildText(title, body, options.maxChars);
		batchInputs.push(text || title || pr.url);
		batchMeta.push({
			url: pr.url,
			title,
			body: buildSnippet(body, options.bodyChars),
		});
		if (batchInputs.length >= options.batchSize) {
			await flushBatch();
		}
	}

	await flushBatch();
	outputStream.end();

	console.log(`Done. Embedded ${processed}/${total} PRs, skipped ${skipped}.`);
}

// CLI entry point
if (process.argv[1]?.endsWith("embed.js") || process.argv[1]?.endsWith("embed.ts")) {
	const args = process.argv.slice(2);
	const options: EmbedOptions = {
		input: "prs.json",
		output: "embeddings.jsonl",
		model: "text-embedding-3-small",
		batchSize: 100,
		maxChars: 4000,
		bodyChars: 2000,
		resume: true,
	};

	for (let i = 0; i < args.length; i += 1) {
		const arg = args[i];
		if (arg === "--input") {
			options.input = args[++i];
		} else if (arg === "--output") {
			options.output = args[++i];
		} else if (arg === "--model") {
			options.model = args[++i];
		} else if (arg === "--batch") {
			options.batchSize = Number(args[++i]);
		} else if (arg === "--max-chars") {
			options.maxChars = Number(args[++i]);
		} else if (arg === "--body-chars") {
			options.bodyChars = Number(args[++i]);
		} else if (arg === "--no-resume") {
			options.resume = false;
		}
	}

	embed(options);
}

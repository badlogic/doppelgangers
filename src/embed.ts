import fs from "fs";
import OpenAI from "openai";
import path from "path";

// Shared types - exported for use by triage.ts and build.ts
export interface Item {
	url: string;
	number?: number;
	title: string;
	body: string | null;
	state?: string;
	type?: string;
	files?: string[];
}

export interface EmbeddingRecord extends Item {
	body: string; // non-null after processing
	embedding: number[];
}

export interface EmbedOptions {
	input: string;
	output: string;
	model: string;
	batchSize: number;
	maxChars: number;
	bodyChars: number;
	resume: boolean;
	localModel?: string;
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

const buildText = (title: string, body: string | null, files: string[] | undefined, maxChars: number): string => {
	const bodyText = (body || "").replace(/\r\n/g, "\n").trim();
	const filesText = files && files.length > 0 ? `\n\nModified files:\n${files.join("\n")}` : "";
	const combined = `${title}\n\n${bodyText}${filesText}`.trim();
	return combined.slice(0, maxChars);
};

const buildSnippet = (body: string | null, bodyChars: number): string => {
	if (!body) return "";
	return body.replace(/\s+/g, " ").trim().slice(0, bodyChars);
};

export async function embed(options: EmbedOptions): Promise<void> {
	const useLocal = !process.env.OPENAI_API_KEY;
	if (useLocal && !options.localModel) {
		console.error("OPENAI_API_KEY is required, or --local-model must be specified");
		process.exit(1);
	}

	if (useLocal) {
		console.log(`Using local model: ${options.localModel}`);
	}

	const inputPath = path.resolve(options.input);
	const outputPath = path.resolve(options.output);

	const items: Item[] = JSON.parse(fs.readFileSync(inputPath, "utf8"));

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

	let client: OpenAI | null = null;
	let llamaContext: any = null;

	if (useLocal) {
		try {
			const { getLlama } = await import("node-llama-cpp");
			const llama = await getLlama();
			const model = await llama.loadModel({
				modelPath: options.localModel!,
			});
			llamaContext = await model.createEmbeddingContext();
		} catch (e: any) {
			if (e?.code === "ERR_MODULE_NOT_FOUND" || e?.message?.includes("Cannot find")) {
				console.error("node-llama-cpp is not installed. Install it with:");
				console.error("  npm install node-llama-cpp");
				console.error("\nNote: This package is optional and only needed for local embeddings.");
			} else {
				console.error("Failed to initialize node-llama-cpp:", e);
			}
			process.exit(1);
		}
	} else {
		client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
	}

	const outputDir = path.dirname(outputPath);
	fs.mkdirSync(outputDir, { recursive: true });
	const outputStream = fs.createWriteStream(outputPath, { flags: options.resume ? "a" : "w" });

	const pending = items.filter((item) => item?.url && !existing.has(item.url));
	const total = pending.length;
	let processed = 0;
	const skipped = items.length - pending.length;

	let batchInputs: string[] = [];
	let batchMeta: Omit<EmbeddingRecord, "embedding">[] = [];

	const createEmbeddings = async (inputs: string[], attempt = 1): Promise<number[][]> => {
		try {
			if (useLocal && llamaContext) {
				// node-llama-cpp processes one by one
				const results: number[][] = [];
				for (const input of inputs) {
					const embedding = await llamaContext.getEmbeddingFor(input);
					results.push(Array.from(embedding.vector));
				}
				return results;
			} else if (client) {
				const response = await client.embeddings.create({
					model: options.model,
					input: inputs,
				});
				return response.data.map((item) => item.embedding);
			}
			return [];
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
				number: meta.number,
				title: meta.title,
				body: meta.body,
				state: meta.state,
				type: meta.type,
				files: meta.files,
				embedding: embeddings[i],
			};
			outputStream.write(`${JSON.stringify(record)}\n`);
			processed += 1;
			if (processed % 50 === 0 || processed === total) {
				console.log(`Embedded ${processed}/${total} items`);
			}
		}
		batchInputs = [];
		batchMeta = [];
	};

	for (const item of pending) {
		const title = item.title || "";
		const body = item.body || "";
		const text = buildText(title, body, item.files, options.maxChars);
		batchInputs.push(text || title || item.url);
		batchMeta.push({
			url: item.url,
			number: item.number,
			title,
			body: buildSnippet(body, options.bodyChars),
			state: item.state,
			type: item.type,
			files: item.files,
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
		} else if (arg === "--local-model") {
			options.localModel = args[++i];
		}
	}

	embed(options);
}

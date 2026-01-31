// Type declarations for optional dependency node-llama-cpp
// This allows the dynamic import to work without having the package installed
declare module "node-llama-cpp" {
	export function getLlama(): Promise<Llama>;

	interface Llama {
		loadModel(options: { modelPath: string }): Promise<LlamaModel>;
	}

	interface LlamaModel {
		createEmbeddingContext(): Promise<LlamaEmbeddingContext>;
	}

	interface LlamaEmbeddingContext {
		getEmbeddingFor(input: string): Promise<LlamaEmbedding>;
	}

	interface LlamaEmbedding {
		vector: readonly number[];
	}
}

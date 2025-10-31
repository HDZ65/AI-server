import { HttpService } from '@nestjs/axios';

export interface EmbedChunkMeta {
    docId: string;
    filename: string;
    section?: string;
}

export interface EmbedChunk {
    id: string;
    text: string;
    meta: EmbedChunkMeta;
    vector: number[];
}

export class InMemoryVectorStore {
    private readonly dim: number | null = null;
    private readonly items: EmbedChunk[] = [];

    addMany(chunks: Omit<EmbedChunk, 'vector'>[], vectors: number[][]) {
        if (this.dim === null && vectors[0]) (this as any).dim = vectors[0].length;
        chunks.forEach((c, i) => this.items.push({ ...c, vector: vectors[i] }));
    }

    // cosine similarity
    search(queryVec: number[], k = 6): EmbedChunk[] {
        const scores = this.items.map((it) => ({
            it,
            s: cosine(queryVec, it.vector),
        }));
        scores.sort((a, b) => b.s - a.s);
        return scores.slice(0, k).map((x) => x.it);
    }

    count() { return this.items.length; } // <-- NEW
    isEmpty() { return this.count() === 0; } // <-- NEW
}

function cosine(a: number[], b: number[]) {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-12);
}

export class OllamaEmbeddings {
    constructor(private http: HttpService, private baseUrl: string, private model = 'nomic-embed-text') { }

    async embed(texts: string[]): Promise<number[][]> {
        const vectors: number[][] = [];
        for (const t of texts) {
            const { data } = await this.http.axiosRef.post(`${this.baseUrl}/api/embeddings`, {
                model: this.model,
                prompt: t,
            });
            if (!data?.embedding || !Array.isArray(data.embedding)) {
                throw new Error('Embedding invalide depuis Ollama');
            }
            vectors.push(data.embedding as number[]);
        }
        return vectors;
    }
}

import { Injectable } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { HttpService } from '@nestjs/axios';
import { isAxiosError } from 'axios';
import { firstValueFrom } from 'rxjs';
import { createInterface } from 'node:readline';
import crypto from 'node:crypto';
import { splitMarkdown } from '../rag/markdown';
import { InMemoryVectorStore, OllamaEmbeddings } from '../rag/embeddings.store';

export interface ConversationTurn {
  role: 'user' | 'assistant';
  content: string;
}
export interface MarkdownDocument {
  id: string;
  name: string;
  size: number;
  content: string;
}
export interface ChatRequestPayload {
  message: string;
  history: ConversationTurn[];
  systemPrompt?: string;
  documents?: MarkdownDocument[];
}

interface OllamaErrorResponse {
  error?: string;
  [key: string]: unknown;
}
interface OllamaStreamChunk {
  response?: string;
  done?: boolean;
  error?: string;
}

@Injectable()
export class OllamaService {
  // <<< re-déclare ce dont tu as besoin >>>
  private ollamaUrl: string;
  private ollamaModel: string;
  private defaultSystemPrompt: string;

  private embedModel: string;
  private vectorStore: InMemoryVectorStore;
  private embeddings: OllamaEmbeddings;

  constructor(
    private readonly configService: ConfigService,
    private readonly httpService: HttpService,
  ) {
    // Lis la config ICI (après DI)
    this.ollamaUrl =
      this.configService.get<string>('OLLAMA_URL') ||
      'http://finanssor-data-center-v1.tail446cc0.ts.net:11434';
    this.ollamaModel =
      this.configService.get<string>('OLLAMA_MODEL') || 'gpt-oss:20b';
    this.defaultSystemPrompt =
      this.configService.get<string>('SYSTEM_PROMPT') ||
      'Tu es un assistant IA utile et bienveillant.';

    this.embedModel =
      this.configService.get<string>('OLLAMA_EMBED_MODEL') ||
      'nomic-embed-text';

    this.vectorStore = new InMemoryVectorStore();
    this.embeddings = new OllamaEmbeddings(
      this.httpService,
      this.ollamaUrl,
      this.embedModel,
    );
  }

  // -------- Indexation --------
  async indexDocuments(documents: MarkdownDocument[]) {
    if (!documents?.length) return;

    const MAX_DOCS = 20;
    const MAX_CHUNKS_PER_DOC = 80;
    const MAX_CHARS_PER_DOC = 200_000;

    const toAddTexts: string[] = [];
    const toAddMetas: {
      id: string;
      text: string;
      meta: { docId: string; filename: string; section?: string };
    }[] = [];

    for (const doc of documents.slice(0, MAX_DOCS)) {
      const safe = doc.content.slice(0, MAX_CHARS_PER_DOC);
      const chunks = splitMarkdown(safe, 1200, 150).slice(0, MAX_CHUNKS_PER_DOC);

      for (const ch of chunks) {
        const id = crypto.randomUUID();
        toAddTexts.push(ch.text);
        toAddMetas.push({
          id,
          text: ch.text,
          meta: { docId: doc.id, filename: doc.name, section: ch.section },
        });
      }
    }

    if (!toAddTexts.length) return;

    const vectors = await this.embeddings.embed(toAddTexts);
    this.vectorStore.addMany(
      toAddMetas.map(({ id, text, meta }) => ({ id, text, meta }) as any),
      vectors,
    );

    console.log(`[OllamaService] Indexation OK: ${toAddTexts.length} chunks.`);
  }

  private async retrieveContext(
    query: string,
    k = 6,
  ): Promise<{
    contextBlock: string;
    sources: { n: number; filename: string; section?: string }[];
  }> {
    const [qVec] = await this.embeddings.embed([query]);
    const hits = this.vectorStore.search(qVec, k);

    const contextBlock = hits
      .map(
        (h, i) =>
          `### [${i + 1}] ${h.meta.filename}${
            h.meta.section ? ' > ' + h.meta.section : ''
          }\n${h.text}`,
      )
      .join('\n\n---\n\n');

    const sources = hits.map((h, i) => ({
      n: i + 1,
      filename: h.meta.filename,
      section: h.meta.section,
    }));

    return { contextBlock, sources };
  }

  private async buildPromptWithRetrieval(
    message: string,
    history: ConversationTurn[],
    systemPrompt?: string,
    documents?: MarkdownDocument[],
  ): Promise<{
    prompt: string;
    sources: { n: number; filename: string; section?: string }[];
  }> {
    if (documents?.length) {
      await this.indexDocuments(documents);
    }

    const { contextBlock, sources } = await this.retrieveContext(message, 6);

    const activeSystemPrompt =
      (systemPrompt?.trim() || this.defaultSystemPrompt) +
      '\nTu citeras les sources sous forme [n] qui correspondent aux blocs ci-dessous.';

    const sanitizedHistory = (history || [])
      .map((t) => ({ role: t.role, content: t.content.trim() }))
      .filter((t) => t.content.length > 0);

    const historySection = sanitizedHistory
      .map((t) =>
        t.role === 'user'
          ? `<user>\n${t.content}\n</user>`
          : `<assistant>\n${t.content}\n</assistant>`,
      )
      .join('\n');

    const conversationBlock = [historySection, `<user>\n${message}\n</user>`, `<assistant>\n`]
      .filter(Boolean)
      .join('\n');

    const documentsContext = `

<documents>
Voici des passages pertinents issus de la base :

${contextBlock}
</documents>`;

    const prompt = `${activeSystemPrompt}${documentsContext}

<conversation>
${conversationBlock}
</conversation>`;

    return { prompt, sources };
  }

  // -------- Chat (stream) --------
  async *streamChat(
    message: string,
    history: ConversationTurn[] = [],
    systemPrompt?: string,
    documents?: MarkdownDocument[],
  ): AsyncGenerator<string, void, unknown> {
    try {
      const { prompt } = await this.buildPromptWithRetrieval(
        message,
        history,
        systemPrompt,
        documents,
      );

      const payload = { model: this.ollamaModel, prompt, stream: true };

      const response = await firstValueFrom(
        this.httpService.post(`${this.ollamaUrl}/api/generate`, payload, {
          headers: { 'Content-Type': 'application/json' },
          responseType: 'stream',
        }),
      );

      const chunkStream = response.data as NodeJS.ReadableStream;
      const reader = createInterface({ input: chunkStream, crlfDelay: Infinity });

      try {
        for await (const line of reader) {
          const trimmedLine = line.trim();
          if (!trimmedLine) continue;
          const jsonPayload = trimmedLine.startsWith('data:')
            ? trimmedLine.slice(5).trim()
            : trimmedLine;
          if (!jsonPayload) continue;

          try {
            const chunk = JSON.parse(jsonPayload) as OllamaStreamChunk;
            if (chunk?.error) throw new Error(chunk.error);
            if (chunk?.response) yield chunk.response;
            if (chunk?.done) break;
          } catch {
            // ignore bad json line
          }
        }
      } finally {
        reader.close();
      }
    } catch (err: unknown) {
      if (isAxiosError<OllamaErrorResponse>(err)) {
        const errorMessage = err.response?.data?.error || err.message;
        console.error('Erreur dans le service Ollama (Axios):', errorMessage);
        throw new Error(`Ollama API request failed: ${errorMessage}`);
      }
      console.error('Erreur dans le service Ollama:', err);
      throw err;
    }
  }

  // -------- Chat (non-stream) --------
  async chat(
    message: string,
    history: ConversationTurn[] = [],
    systemPrompt?: string,
    documents?: MarkdownDocument[],
  ): Promise<string> {
    let fullResponse = '';
    for await (const chunk of this.streamChat(
      message,
      history,
      systemPrompt,
      documents,
    )) {
      fullResponse += chunk;
    }
    return fullResponse;
  }
}

import { BadRequestException, Body, Controller, Get, HttpException, InternalServerErrorException, Post, Res } from '@nestjs/common';
import type { Response } from 'express';
import { ConversationTurn, OllamaService } from './ollama.service';

interface MarkdownDocument {
    id: string;
    name: string;
    size: number;
    content: string;
}

interface ChatRequestBody {
    message: string;
    history?: ConversationTurn[];
    systemPrompt?: string;
    documents?: MarkdownDocument[];
}

const MAX_HISTORY_ENTRIES = 12;

@Controller('ollama')
export class OllamaController {
    constructor(private readonly ollamaService: OllamaService) {
        console.log('[CONTROLLER] OllamaController initialisé');
    }

    @Get()
    getHealthCheck() {
        console.log('[CONTROLLER] GET /ollama - Health check');
        return {
            status: 'ok',
            service: 'Ollama GPT local',
            message: 'API GPT locale fonctionnelle',
            timestamp: new Date().toISOString(),
        };
    }
    

@Post('chat')
async chat(@Body() body: ChatRequestBody, @Res() res: Response) {
    console.log('[CONTROLLER] POST /ollama/chat - Réception du message');
    if (!body) {
        throw new BadRequestException('Le corps de la requête est requis');
    }

    const { message, history, systemPrompt, documents } = body;
    console.log('[CONTROLLER] Message:', message);
    console.log('[CONTROLLER] System Prompt:', systemPrompt ? 'Fourni' : 'Non fourni');
    console.log('[CONTROLLER] Documents:', documents?.length || 0);

    if (typeof message !== 'string') {
        console.error('[CONTROLLER] Type de message invalide reçu');
        throw new BadRequestException('Le message doit être une chaîne de caractères');
    }

    const trimmedMessage = message.trim();
    if (trimmedMessage === '') {
        console.error('[CONTROLLER] Message vide reçu');
        throw new BadRequestException('Le message ne peut pas être vide');
    }

    // Validation de l'historique
    let sanitizedHistory: ConversationTurn[] = [];

    if (typeof history !== 'undefined') {
        if (!Array.isArray(history)) {
            console.error('[CONTROLLER] Historique invalide (type)');
            throw new BadRequestException('Le format de l\'historique est invalide');
        }

        sanitizedHistory = history
            .slice(-MAX_HISTORY_ENTRIES)
            .map((entry) => {
                if (!entry || typeof entry !== 'object') {
                    return null;
                }

                const { role, content } = entry as { role?: unknown; content?: unknown };
                if ((role !== 'user' && role !== 'assistant') || typeof content !== 'string') {
                    return null;
                }

                const trimmedContent = content.trim();
                if (!trimmedContent) {
                    return null;
                }

                return { role, content: trimmedContent } as ConversationTurn;
            })
            .filter((entry): entry is ConversationTurn => entry !== null);
    }

    // Validation du prompt système (optionnel)
    let sanitizedSystemPrompt: string | undefined = undefined;
    if (systemPrompt !== undefined) {
        if (typeof systemPrompt !== 'string') {
            console.error('[CONTROLLER] Type de systemPrompt invalide');
            throw new BadRequestException('Le systemPrompt doit être une chaîne de caractères');
        }
        const trimmed = systemPrompt.trim();
        if (trimmed.length > 0) {
            sanitizedSystemPrompt = trimmed;
        }
    }

    // Validation des documents (optionnel)
    let sanitizedDocuments: MarkdownDocument[] | undefined = undefined;
    if (documents !== undefined) {
        if (!Array.isArray(documents)) {
            console.error('[CONTROLLER] Type de documents invalide');
            throw new BadRequestException('Les documents doivent être un tableau');
        }

        sanitizedDocuments = documents
            .map((doc) => {
                if (!doc || typeof doc !== 'object') {
                    return null;
                }

                const { id, name, size, content } = doc as {
                    id?: unknown;
                    name?: unknown;
                    size?: unknown;
                    content?: unknown;
                };

                if (
                    typeof id !== 'string' ||
                    typeof name !== 'string' ||
                    typeof size !== 'number' ||
                    typeof content !== 'string'
                ) {
                    console.warn('[CONTROLLER] Document invalide ignoré:', doc);
                    return null;
                }

                const trimmedContent = content.trim();
                if (!trimmedContent) {
                    console.warn('[CONTROLLER] Document avec contenu vide ignoré:', name);
                    return null;
                }

                return {
                    id,
                    name,
                    size,
                    content: trimmedContent,
                } as MarkdownDocument;
            })
            .filter((doc): doc is MarkdownDocument => doc !== null);

        if (sanitizedDocuments.length === 0) {
            sanitizedDocuments = undefined;
        }
    }

    res.setHeader('Content-Type', 'application/x-ndjson; charset=utf-8');
    res.setHeader('Cache-Control', 'no-cache, no-transform');
    res.setHeader('Connection', 'keep-alive');

    try {
        res.status(200);
        const expressResponse = res as Response & { flush?: () => void };
        
        for await (const chunk of this.ollamaService.streamChat(
            trimmedMessage,
            sanitizedHistory,
            sanitizedSystemPrompt,
            sanitizedDocuments,
        )) {
            const payload = {
                type: 'chunk',
                data: chunk,
                timestamp: new Date().toISOString(),
            };
            expressResponse.write(`${JSON.stringify(payload)}\n`);
            if (typeof expressResponse.flush === 'function') {
                expressResponse.flush();
            }
        }

        const donePayload = {
            type: 'done',
            timestamp: new Date().toISOString(),
        };
        expressResponse.write(`${JSON.stringify(donePayload)}\n`);
        if (typeof expressResponse.flush === 'function') {
            expressResponse.flush();
        }
    } catch (error) {
        console.error('[CONTROLLER] Erreur lors de l\'appel à Ollama:', error);
        if (error instanceof HttpException) {
            const status = error.getStatus?.() ?? 500;
            const expressResponse = res as Response & { flush?: () => void };
            expressResponse.status(status);
            const payload = {
                type: 'error',
                error: error.message,
                timestamp: new Date().toISOString(),
                status,
            };
            expressResponse.write(`${JSON.stringify(payload)}\n`);
            if (typeof expressResponse.flush === 'function') {
                expressResponse.flush();
            }
        } else {
            const message = error instanceof Error ? error.message : 'Erreur lors de la génération de la réponse';
            const expressResponse = res as Response & { flush?: () => void };
            expressResponse.status(500);
            const payload = {
                type: 'error',
                error: message,
                timestamp: new Date().toISOString(),
                status: 500,
            };
            expressResponse.write(`${JSON.stringify(payload)}\n`);
            if (typeof expressResponse.flush === 'function') {
                expressResponse.flush();
            }
        }
    } finally {
        res.end();
    }
}
}

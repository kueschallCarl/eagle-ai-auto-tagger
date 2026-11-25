/**
 * CLIP Embedding Extractor
 * Uses @xenova/transformers to extract image embeddings locally
 */

// Dynamic import for transformers.js (ESM module)
let pipeline: any = null;
let extractor: any = null;

const MODEL_NAME = 'Xenova/clip-vit-base-patch32';

export async function initializeClip(): Promise<boolean> {
    if (extractor) return true;

    try {
        // Dynamic import since transformers.js is ESM
        const transformers = await import('@xenova/transformers');
        pipeline = transformers.pipeline;

        console.log('[CLIP] Loading model:', MODEL_NAME);
        extractor = await pipeline('feature-extraction', MODEL_NAME, {
            quantized: true, // Use quantized model for faster loading
        });
        console.log('[CLIP] Model loaded successfully');
        return true;
    } catch (error) {
        console.error('[CLIP] Failed to initialize:', error);
        return false;
    }
}

export function isClipReady(): boolean {
    return extractor !== null;
}

export async function extractEmbeddingFromPath(imagePath: string): Promise<Float32Array | null> {
    if (!extractor) {
        throw new Error('CLIP model not initialized. Call initializeClip() first.');
    }

    try {
        const output = await extractor(imagePath, {
            pooling: 'mean',
            normalize: true,
        });

        // Convert to Float32Array
        const embedding = new Float32Array(output.data);
        return embedding;
    } catch (error) {
        console.error('[CLIP] Extraction failed for', imagePath, error);
        return null;
    }
}

export async function extractEmbeddingFromBase64(base64Data: string, mimeType: string = 'image/jpeg'): Promise<Float32Array | null> {
    if (!extractor) {
        throw new Error('CLIP model not initialized. Call initializeClip() first.');
    }

    try {
        // Create data URL for transformers.js
        const dataUrl = `data:${mimeType};base64,${base64Data}`;

        const output = await extractor(dataUrl, {
            pooling: 'mean',
            normalize: true,
        });

        const embedding = new Float32Array(output.data);
        return embedding;
    } catch (error) {
        console.error('[CLIP] Base64 extraction failed:', error);
        return null;
    }
}

/**
 * Calculate cosine similarity between two embeddings
 */
export function cosineSimilarity(vec1: Float32Array, vec2: Float32Array): number {
    if (vec1.length !== vec2.length) {
        throw new Error('Vectors must have same length');
    }

    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;

    for (let i = 0; i < vec1.length; i++) {
        dotProduct += vec1[i] * vec2[i];
        norm1 += vec1[i] * vec1[i];
        norm2 += vec2[i] * vec2[i];
    }

    norm1 = Math.sqrt(norm1);
    norm2 = Math.sqrt(norm2);

    if (norm1 === 0 || norm2 === 0) return 0;

    return dotProduct / (norm1 * norm2);
}

/**
 * Get embedding dimension (for reference)
 */
export function getEmbeddingDimension(): number {
    return 512; // CLIP-ViT-B/32 produces 512-dim embeddings
}

/**
 * Recognition Module
 * Person identification using CLIP embeddings
 */

export {
    initializeClip,
    isClipReady,
    extractEmbeddingFromPath,
    extractEmbeddingFromBase64,
    cosineSimilarity,
    getEmbeddingDimension,
} from './clip-extractor';

export {
    embeddingStore,
    EmbeddingStore,
    ReferenceEmbedding,
    EmbeddingCache,
} from './embedding-store';

export {
    findMatches,
    findMatchesDeduped,
    findBestMatch,
    findMatchesDetailed,
    generatePersonTags,
    Match,
    MatchResult,
} from './matcher';

import * as fs from 'fs';
import * as path from 'path';
import { initializeClip, extractEmbeddingFromPath, isClipReady } from './clip-extractor';
import { embeddingStore } from './embedding-store';
import { findMatchesDeduped, generatePersonTags, Match } from './matcher';

/**
 * Parse reference filename into person and feature
 * Format: "PersonName_feature.ext" or "PersonName.ext"
 */
export function parseReferenceFilename(filename: string): { person: string; feature: string } | null {
    const ext = path.extname(filename);
    const baseName = path.basename(filename, ext);

    // Check for valid image extension
    if (!['.jpg', '.jpeg', '.png', '.gif', '.webp'].includes(ext.toLowerCase())) {
        return null;
    }

    // Split by underscore
    const parts = baseName.split('_');

    if (parts.length >= 2) {
        // "John_face" -> person: "John", feature: "face"
        return {
            person: parts[0],
            feature: parts.slice(1).join('_'), // Handle "John_face_front"
        };
    } else {
        // "John.jpg" -> person: "John", feature: "general"
        return {
            person: baseName,
            feature: 'general',
        };
    }
}

/**
 * Load references from a folder
 * @param folderPath Path to folder containing reference images
 * @param onProgress Optional progress callback
 * @returns Summary of loaded references
 */
export async function loadReferencesFromFolder(
    folderPath: string,
    onProgress?: (current: number, total: number, filename: string) => void
): Promise<{ loaded: number; people: string[]; errors: string[] }> {
    // Initialize CLIP if needed
    if (!isClipReady()) {
        const success = await initializeClip();
        if (!success) {
            throw new Error('Failed to initialize CLIP model');
        }
    }

    // Clear existing references
    embeddingStore.clear();

    // Get all files in folder
    if (!fs.existsSync(folderPath)) {
        throw new Error(`Reference folder does not exist: ${folderPath}`);
    }

    const files = fs.readdirSync(folderPath);
    const imageFiles = files.filter(f => {
        const parsed = parseReferenceFilename(f);
        return parsed !== null;
    });

    const errors: string[] = [];
    let loaded = 0;

    for (let i = 0; i < imageFiles.length; i++) {
        const filename = imageFiles[i];
        const filePath = path.join(folderPath, filename);
        const parsed = parseReferenceFilename(filename);

        if (!parsed) continue;

        if (onProgress) {
            onProgress(i + 1, imageFiles.length, filename);
        }

        try {
            const embedding = await extractEmbeddingFromPath(filePath);
            if (embedding) {
                embeddingStore.addReference(parsed.person, parsed.feature, embedding, filePath);
                loaded++;
            } else {
                errors.push(`Failed to extract embedding: ${filename}`);
            }
        } catch (error) {
            errors.push(`Error processing ${filename}: ${(error as Error).message}`);
        }
    }

    return {
        loaded,
        people: embeddingStore.getPersonNames(),
        errors,
    };
}

/**
 * Identify people in an image
 * @param imagePath Path to image
 * @param threshold Match threshold (0.0 - 1.0)
 * @returns Array of person tags
 */
export async function identifyPeopleInImage(
    imagePath: string,
    threshold: number = 0.65
): Promise<{ tags: string[]; matches: Match[] }> {
    if (!isClipReady()) {
        throw new Error('CLIP model not initialized');
    }

    if (embeddingStore.getCount() === 0) {
        return { tags: [], matches: [] };
    }

    const embedding = await extractEmbeddingFromPath(imagePath);
    if (!embedding) {
        return { tags: [], matches: [] };
    }

    const matches = findMatchesDeduped(embedding, threshold);
    const tags = generatePersonTags(matches);

    return { tags, matches };
}

/**
 * Identify people from base64 image data
 */
export async function identifyPeopleFromBase64(
    base64Data: string,
    threshold: number = 0.65,
    mimeType: string = 'image/jpeg'
): Promise<{ tags: string[]; matches: Match[] }> {
    if (!isClipReady()) {
        throw new Error('CLIP model not initialized');
    }

    if (embeddingStore.getCount() === 0) {
        return { tags: [], matches: [] };
    }

    const { extractEmbeddingFromBase64 } = await import('./clip-extractor');
    const embedding = await extractEmbeddingFromBase64(base64Data, mimeType);
    if (!embedding) {
        return { tags: [], matches: [] };
    }

    const matches = findMatchesDeduped(embedding, threshold);
    const tags = generatePersonTags(matches);

    return { tags, matches };
}

/**
 * Get reference stats
 */
export function getReferenceStats(): { count: number; people: string[] } {
    return {
        count: embeddingStore.getCount(),
        people: embeddingStore.getPersonNames(),
    };
}

/**
 * Save embedding cache
 */
export async function saveEmbeddingCache(cachePath: string): Promise<void> {
    await embeddingStore.saveToDisk(cachePath);
}

/**
 * Load embedding cache
 */
export async function loadEmbeddingCache(cachePath: string): Promise<boolean> {
    return await embeddingStore.loadFromDisk(cachePath);
}

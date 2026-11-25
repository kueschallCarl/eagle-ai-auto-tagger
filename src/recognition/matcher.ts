/**
 * Matcher
 * Compare query embeddings against reference embeddings
 */

import { embeddingStore, ReferenceEmbedding } from './embedding-store';
import { cosineSimilarity } from './clip-extractor';

export interface Match {
    person: string;
    feature: string;
    confidence: number;
    sourcePath: string;
}

export interface MatchResult {
    matches: Match[];
    bestMatchPerPerson: Map<string, Match>;
}

/**
 * Find all matches above threshold
 */
export function findMatches(queryEmbedding: Float32Array, threshold: number = 0.65): Match[] {
    const allRefs = embeddingStore.getAllEmbeddingsAsFloat32();
    const matches: Match[] = [];

    for (const { ref, embedding } of allRefs) {
        const similarity = cosineSimilarity(queryEmbedding, embedding);

        if (similarity >= threshold) {
            matches.push({
                person: ref.person,
                feature: ref.feature,
                confidence: similarity,
                sourcePath: ref.sourcePath,
            });
        }
    }

    // Sort by confidence descending
    matches.sort((a, b) => b.confidence - a.confidence);

    return matches;
}

/**
 * Find matches and deduplicate by person (keep highest confidence per person)
 */
export function findMatchesDeduped(queryEmbedding: Float32Array, threshold: number = 0.65): Match[] {
    const allMatches = findMatches(queryEmbedding, threshold);
    const bestPerPerson = new Map<string, Match>();

    for (const match of allMatches) {
        const existing = bestPerPerson.get(match.person);
        if (!existing || match.confidence > existing.confidence) {
            bestPerPerson.set(match.person, match);
        }
    }

    // Convert back to array, sorted by confidence
    return Array.from(bestPerPerson.values()).sort((a, b) => b.confidence - a.confidence);
}

/**
 * Find the single best match (if any)
 */
export function findBestMatch(queryEmbedding: Float32Array, threshold: number = 0.65): Match | null {
    const matches = findMatchesDeduped(queryEmbedding, threshold);
    return matches.length > 0 ? matches[0] : null;
}

/**
 * Get detailed match result with both all matches and best per person
 */
export function findMatchesDetailed(queryEmbedding: Float32Array, threshold: number = 0.65): MatchResult {
    const matches = findMatches(queryEmbedding, threshold);
    const bestMatchPerPerson = new Map<string, Match>();

    for (const match of matches) {
        const existing = bestMatchPerPerson.get(match.person);
        if (!existing || match.confidence > existing.confidence) {
            bestMatchPerPerson.set(match.person, match);
        }
    }

    return { matches, bestMatchPerPerson };
}

/**
 * Generate person tags from matches
 * Format: "person.<name>"
 */
export function generatePersonTags(matches: Match[]): string[] {
    const tags: string[] = [];
    const seen = new Set<string>();

    for (const match of matches) {
        const tag = `person.${match.person}`;
        if (!seen.has(tag)) {
            seen.add(tag);
            tags.push(tag);
        }
    }

    return tags;
}

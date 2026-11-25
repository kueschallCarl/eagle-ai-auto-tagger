/**
 * Embedding Store
 * In-memory storage for reference embeddings with JSON persistence
 */

import * as fs from 'fs';
import * as path from 'path';

export interface ReferenceEmbedding {
    person: string;
    feature: string;
    embedding: number[]; // Stored as regular array for JSON serialization
    sourcePath: string;
    timestamp: number;
}

export interface EmbeddingCache {
    version: string;
    created: number;
    references: { [key: string]: ReferenceEmbedding };
}

const CACHE_VERSION = '1.0';

class EmbeddingStore {
    private references: Map<string, ReferenceEmbedding> = new Map();

    /**
     * Generate key for a reference
     */
    private makeKey(person: string, feature: string): string {
        return `${person}.${feature}`;
    }

    /**
     * Add a reference embedding
     */
    addReference(
        person: string,
        feature: string,
        embedding: Float32Array,
        sourcePath: string
    ): void {
        const key = this.makeKey(person, feature);
        this.references.set(key, {
            person,
            feature,
            embedding: Array.from(embedding),
            sourcePath,
            timestamp: Date.now(),
        });
    }

    /**
     * Get all references
     */
    getAll(): Map<string, ReferenceEmbedding> {
        return this.references;
    }

    /**
     * Get references for a specific person
     */
    getForPerson(person: string): ReferenceEmbedding[] {
        const results: ReferenceEmbedding[] = [];
        for (const ref of this.references.values()) {
            if (ref.person === person) {
                results.push(ref);
            }
        }
        return results;
    }

    /**
     * Get all unique person names
     */
    getPersonNames(): string[] {
        const names = new Set<string>();
        for (const ref of this.references.values()) {
            names.add(ref.person);
        }
        return Array.from(names);
    }

    /**
     * Get count of references
     */
    getCount(): number {
        return this.references.size;
    }

    /**
     * Clear all references
     */
    clear(): void {
        this.references.clear();
    }

    /**
     * Check if a reference exists
     */
    has(person: string, feature: string): boolean {
        return this.references.has(this.makeKey(person, feature));
    }

    /**
     * Save embeddings to disk
     */
    async saveToDisk(filePath: string): Promise<void> {
        const cache: EmbeddingCache = {
            version: CACHE_VERSION,
            created: Date.now(),
            references: {},
        };

        for (const [key, ref] of this.references) {
            cache.references[key] = ref;
        }

        const dir = path.dirname(filePath);
        if (!fs.existsSync(dir)) {
            fs.mkdirSync(dir, { recursive: true });
        }

        fs.writeFileSync(filePath, JSON.stringify(cache, null, 2));
    }

    /**
     * Load embeddings from disk
     */
    async loadFromDisk(filePath: string): Promise<boolean> {
        if (!fs.existsSync(filePath)) {
            return false;
        }

        try {
            const data = fs.readFileSync(filePath, 'utf8');
            const cache: EmbeddingCache = JSON.parse(data);

            if (cache.version !== CACHE_VERSION) {
                console.warn('[EmbeddingStore] Cache version mismatch, ignoring');
                return false;
            }

            this.references.clear();
            for (const [key, ref] of Object.entries(cache.references)) {
                this.references.set(key, ref);
            }

            return true;
        } catch (error) {
            console.error('[EmbeddingStore] Failed to load cache:', error);
            return false;
        }
    }

    /**
     * Get embedding as Float32Array (for similarity calculations)
     */
    getEmbeddingAsFloat32(person: string, feature: string): Float32Array | null {
        const ref = this.references.get(this.makeKey(person, feature));
        if (!ref) return null;
        return new Float32Array(ref.embedding);
    }

    /**
     * Get all embeddings as Float32Arrays
     */
    getAllEmbeddingsAsFloat32(): Array<{ key: string; ref: ReferenceEmbedding; embedding: Float32Array }> {
        const results: Array<{ key: string; ref: ReferenceEmbedding; embedding: Float32Array }> = [];
        for (const [key, ref] of this.references) {
            results.push({
                key,
                ref,
                embedding: new Float32Array(ref.embedding),
            });
        }
        return results;
    }
}

// Export singleton instance
export const embeddingStore = new EmbeddingStore();

// Also export class for testing
export { EmbeddingStore };

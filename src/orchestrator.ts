import { PluginConfig, PluginConfigSchema, BatchResult } from './schema';
import { tagImage } from './providers';
import { getRateLimiter } from './utils/rate-limiter';

export type ProgressCallback = (completed: number, total: number) => void;

export interface ImageItem {
  id: string;
  base64: string;
}

export async function tagSingleImage(
  imageBase64: string,
  config: PluginConfig
): Promise<BatchResult> {
  const validConfig = PluginConfigSchema.parse(config);
  const limiter = getRateLimiter(validConfig.apiProvider);

  await limiter.waitForToken();

  try {
    const result = await tagImage(imageBase64, validConfig);
    return {
      id: '',
      tags: result.tags,
      confidence: result.confidence,
    };
  } catch (err) {
    return {
      id: '',
      tags: [],
      confidence: 0,
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

export async function tagImagesBatch(
  images: ImageItem[],
  config: PluginConfig,
  onProgress?: ProgressCallback
): Promise<BatchResult[]> {
  const validConfig = PluginConfigSchema.parse(config);
  const limiter = getRateLimiter(validConfig.apiProvider);
  const batchSize = validConfig.batchSize;
  const results: BatchResult[] = [];

  for (let i = 0; i < images.length; i += batchSize) {
    const chunk = images.slice(i, i + batchSize);

    const chunkResults = await Promise.all(
      chunk.map(async (item) => {
        await limiter.waitForToken();

        try {
          const result = await tagImage(item.base64, validConfig);
          return {
            id: item.id,
            tags: result.tags,
            confidence: result.confidence,
          } as BatchResult;
        } catch (err) {
          return {
            id: item.id,
            tags: [],
            confidence: 0,
            error: err instanceof Error ? err.message : String(err),
          } as BatchResult;
        }
      })
    );

    results.push(...chunkResults);

    if (onProgress) {
      onProgress(results.length, images.length);
    }
  }

  return results;
}

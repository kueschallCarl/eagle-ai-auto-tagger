import { z } from 'zod';

export const PluginConfigSchema = z.object({
  apiProvider: z.enum(['openrouter', 'groq']),
  apiKey: z.string().min(1, 'API key is required'),
  model: z.string().optional(),
  customPrompt: z.string().optional(),
  desiredTags: z.array(z.string()).optional(),
  batchSize: z.number().int().min(1).max(10).default(5),
});

export type PluginConfig = z.infer<typeof PluginConfigSchema>;

export const TagResponseSchema = z.object({
  tags: z.array(z.string()),
  confidence: z.number().min(0).max(1),
  reasoning: z.string().optional(),
});

export type TagResponse = z.infer<typeof TagResponseSchema>;

export const BatchResultSchema = z.object({
  id: z.string(),
  tags: z.array(z.string()),
  confidence: z.number(),
  error: z.string().optional(),
});

export type BatchResult = z.infer<typeof BatchResultSchema>;

export const DEFAULT_MODELS: Record<PluginConfig['apiProvider'], string> = {
  openrouter: 'nvidia/nemotron-nano-12b-v2-vl:free',
  groq: 'meta-llama/llama-4-scout-17b-16e-instruct',
};

import { TagResponse, PluginConfig } from '../schema';
import { tagImageWithOpenRouter } from './openrouter';
import { tagImageWithGroq } from './groq';

const BASE_PROMPT = `Analyze this image and provide relevant descriptive tags. Return your response as a JSON object with this exact format:
{
  "tags": ["tag1", "tag2", "tag3"],
  "confidence": 0.95,
  "reasoning": "Brief explanation of what you see"
}

Rules:
- Provide 5-15 relevant tags
- Tags should be lowercase, single words or short phrases
- Confidence should reflect how well you can identify the content (0.0-1.0)
- Only return the JSON object, no other text`;

export function buildPrompt(customPrompt?: string, desiredTags?: string[]): string {
  let prompt = BASE_PROMPT;

  if (customPrompt) {
    prompt += `\n\nAdditional instructions: ${customPrompt}`;
  }

  if (desiredTags && desiredTags.length > 0) {
    prompt += `\n\nIMPORTANT: Only return tags from this specific list: ${desiredTags.join(', ')}. If the image doesn't match any of these tags, return an empty tags array.`;
  }

  return prompt;
}

export async function tagImage(
  imageBase64: string,
  config: PluginConfig
): Promise<TagResponse> {
  switch (config.apiProvider) {
    case 'openrouter':
      return tagImageWithOpenRouter(imageBase64, config);
    case 'groq':
      return tagImageWithGroq(imageBase64, config);
    default:
      throw new Error(`Unknown provider: ${config.apiProvider}`);
  }
}

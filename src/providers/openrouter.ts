import { TagResponse, TagResponseSchema, PluginConfig, DEFAULT_MODELS } from '../schema';
import { buildPrompt } from './index';

const API_URL = 'https://openrouter.ai/api/v1/chat/completions';

export async function tagImageWithOpenRouter(
  imageBase64: string,
  config: PluginConfig
): Promise<TagResponse> {
  const model = config.model || DEFAULT_MODELS.openrouter;
  const prompt = buildPrompt(config.customPrompt, config.desiredTags);

  const response = await fetch(API_URL, {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${config.apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model,
      messages: [{
        role: 'user',
        content: [
          { type: 'text', text: prompt },
          {
            type: 'image_url',
            image_url: { url: `data:image/jpeg;base64,${imageBase64}` },
          },
        ],
      }],
      max_tokens: 1024,
      temperature: 0.3,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`OpenRouter API error (${response.status}): ${errorText.slice(0, 500)}`);
  }

  const rawText = await response.text();
  let data: any;
  try {
    data = JSON.parse(rawText);
  } catch (e) {
    throw new Error(`OpenRouter returned invalid JSON: ${rawText.slice(0, 300)}`);
  }

  const content = data.choices?.[0]?.message?.content;
  if (!content) {
    throw new Error(`No content in OpenRouter response. Raw: ${rawText.slice(0, 500)}`);
  }

  const result = parseTagResponse(content);
  if (result.tags.length === 0) {
    throw new Error(`Model returned empty tags. Raw content: ${content.slice(0, 500)}`);
  }
  return result;
}

function parseTagResponse(text: string): TagResponse {
  const jsonMatch = text.match(/\{[\s\S]*\}/);
  if (!jsonMatch) {
    throw new Error(`No JSON found in model response: ${text.slice(0, 300)}`);
  }

  let parsed: any;
  try {
    parsed = JSON.parse(jsonMatch[0]);
  } catch (e) {
    throw new Error(`Invalid JSON in model response: ${jsonMatch[0].slice(0, 300)}`);
  }

  try {
    return TagResponseSchema.parse(parsed);
  } catch (e: any) {
    throw new Error(`Schema validation failed: ${e.message}. Parsed: ${JSON.stringify(parsed).slice(0, 300)}`);
  }
}

import { TagResponse, TagResponseSchema, PluginConfig, DEFAULT_MODELS } from '../schema';
import { buildPrompt } from './index';

const API_URL = 'https://api.groq.com/openai/v1/chat/completions';

export async function tagImageWithGroq(
  imageBase64: string,
  config: PluginConfig
): Promise<TagResponse> {
  const model = config.model || DEFAULT_MODELS.groq;
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
    throw new Error(`Groq API error (${response.status}): ${errorText}`);
  }

  const data = await response.json() as { choices?: Array<{ message?: { content?: string } }> };
  const content = data.choices?.[0]?.message?.content;
  if (!content) {
    throw new Error('No response content from Groq');
  }

  return parseTagResponse(content);
}

function parseTagResponse(text: string): TagResponse {
  const jsonMatch = text.match(/\{[\s\S]*\}/);
  if (!jsonMatch) {
    throw new Error(`Failed to extract JSON from Groq response: ${text.slice(0, 200)}`);
  }

  const parsed = JSON.parse(jsonMatch[0]);
  return TagResponseSchema.parse(parsed);
}

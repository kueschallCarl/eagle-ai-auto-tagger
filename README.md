# AI Auto-Tagger for Eagle

A plugin for the [Eagle](https://eagle.cool) media manager. It tags images and videos with vision language models and can recognize people in your library from a few reference pictures. You bring your own API key, and person recognition runs entirely on your machine.

## Features

- **Vision tagging** through [OpenRouter](https://openrouter.ai) or [Groq](https://groq.com), with a model override and custom instructions ("Tag by color palette", "Identify art style").
- **Tag filter.** Restrict output to a vocabulary you define. Responses are validated with Zod, so a malformed reply never writes tags.
- **Video support.** Samples frames with ffmpeg and tags the composite in one request. Optionally converts `.ts` recordings to `.mp4`.
- **Local person recognition.** Point it at an Eagle folder of reference images named after people. It matches new items with an ensemble of:
  - SCRFD face detection, then 5-point alignment, then ArcFace (512-d embeddings, cosine similarity)
  - CLIP ViT-B/32 image embeddings, for references without a clear face

  Matches become `person.<Name>` tags. Thresholds, mode (ensemble, face only, CLIP only) and the face/CLIP weighting are configurable. Embeddings are cached.
- **Batch processing** with per-provider rate limiting, progress, stop and skip rules (already tagged, ignore-tags).
- **Presets** for provider, model, prompt and tag set.

## Build

```bash
npm install
npm run package   # build/eagle-ai-auto-tagger/ and build/eagle-ai-auto-tagger.eagleplugin
```

Install the `.eagleplugin` file in Eagle, or import the folder as a developer plugin. Download the face models below before packaging; the build bundles them.

Person recognition needs two ONNX models in `models/face/` (they are not redistributed here):

| File | Source |
|---|---|
| `det_10g.onnx` | SCRFD-10G, from InsightFace's `buffalo_l` pack (non-commercial research license) |
| `arcfaceresnet100-8.onnx` | ArcFace ResNet-100, from the ONNX Model Zoo |

The CLIP vision model downloads from Hugging Face on first use. Video features need `ffmpeg` and `ffprobe` on `PATH` or in Homebrew's default location.

## Layout

```
index.html, js/plugin.js   Eagle plugin UI and runtime
src/providers/             OpenRouter and Groq clients
src/recognition/           CLIP extraction, embedding store, matcher
src/face/                  SCRFD detector, alignment, ArcFace
src/utils/                 rate limiter, video frame extraction
src/schema.ts              Zod schemas for model output
```

## License

MIT. The face models carry their own licenses; see the table above.

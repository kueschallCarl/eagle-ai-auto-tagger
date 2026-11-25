const path = require('path');
const fs = require('fs');
const sharp = require('sharp');

let pluginPath = '';
let logArea = null;
let logContent = null;
let currentItemThumb = null;
let currentItemName = null;
let currentItemStatus = null;

const MAX_LOG_LINES = 200;

// ============================================
// ENHANCED RECOGNITION SYSTEM
// Combines CLIP (general visual) + Face-API (face-specific)
// ============================================

// Recognition configuration
// Using SCRFD + ArcFace (512-dim embeddings, cosine similarity)
// Typical ArcFace similarity scores:
//   Same person: 0.6 - 0.95
//   Different people: 0.2 - 0.5
// Recommended thresholds:
//   Face Threshold: 0.5 (balance between precision and recall)
//   CLIP Threshold: 0.65 (visual similarity)
//   Face Weight: 80% (ArcFace is more discriminative than CLIP)
const RECOGNITION_CONFIG = {
    // Thresholds by feature type
    thresholds: {
        face: 0.5,       // ArcFace cosine similarity threshold (0.5 = safe same-person match)
        body: 0.65,      // CLIP threshold for body features
        general: 0.65,   // CLIP threshold for general features
        default: 0.65,
    },
    // Ensemble weights (ArcFace 80%, CLIP 20% - ArcFace is more discriminative)
    weights: {
        faceApi: 0.8,    // ArcFace weight when face is detected
        clip: 0.2,       // CLIP weight when face is detected
        clipOnly: 1.0,   // CLIP weight when no face detected
    },
    // Multi-reference confidence
    minRefsToMatch: 1,   // Minimum references that must match
    confidenceBoost: 0.05, // Boost per additional matching reference
    // Recognition mode: 'ensemble', 'clip', or 'face'
    mode: 'ensemble',
};

// Debug configuration
let DEBUG_MODE = false;
let debugFolder = null;

// Video processing configuration
let VIDEO_FRAMES = 5; // Number of frames to analyze per video

// Tag handling configuration
let SKIP_TAGGED = true;   // Skip items that already have person.* tags
let REPLACE_TAGS = false; // Replace existing person.* tags with new ones
let IGNORE_TAGS = [];     // Tags that cause items to be skipped (e.g., 'animation')

async function initDebugFolder() {
    if (!DEBUG_MODE) return null;
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').substring(0, 19);
    debugFolder = path.join(pluginPath, 'debug', timestamp);
    if (!fs.existsSync(debugFolder)) {
        fs.mkdirSync(debugFolder, { recursive: true });
    }
    log(`[Debug] Output folder: ${debugFolder}`);
    return debugFolder;
}

async function saveDebugImage(buffer, filename) {
    if (!DEBUG_MODE || !debugFolder) return;
    const filepath = path.join(debugFolder, filename);
    fs.writeFileSync(filepath, buffer);
    return filepath;
}

async function generateDebugReport(queryItem, queryFaces, matches, referenceEmbeddings) {
    if (!DEBUG_MODE || !debugFolder) return;

    const report = {
        timestamp: new Date().toISOString(),
        queryItem: {
            id: queryItem.id,
            name: queryItem.name,
            path: queryItem.filePath || queryItem.thumbnailPath,
        },
        queryFaces: queryFaces.map((f, i) => ({
            index: i,
            bbox: f.bbox,
            confidence: f.confidence,
            embeddingNorm: Math.sqrt(f.descriptor.reduce((s, v) => s + v * v, 0)),
            alignedFaceFile: f.alignedFaceFile || null,
        })),
        topMatches: matches.slice(0, 10).map(m => ({
            person: m.person,
            confidence: m.confidence,
            faceScore: m.faceScore,
            clipScore: m.clipScore,
            matchType: m.matchType,
            refCount: m.refCount,
        })),
        allScores: {},
    };

    // Add all person scores for analysis
    for (const [person, refs] of groupReferencesByPerson(referenceEmbeddings)) {
        if (queryFaces.length > 0 && refs[0].faceDescriptors?.length > 0) {
            let maxScore = 0;
            for (const qf of queryFaces) {
                for (const ref of refs) {
                    for (const refFace of ref.faceDescriptors || []) {
                        const sim = cosineSimilarity(
                            new Float32Array(qf.descriptor),
                            new Float32Array(refFace)
                        );
                        maxScore = Math.max(maxScore, sim);
                    }
                }
            }
            report.allScores[person] = maxScore;
        }
    }

    const reportPath = path.join(debugFolder, `report_${queryItem.id}.json`);
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    return reportPath;
}

function groupReferencesByPerson(referenceEmbeddings) {
    const grouped = new Map();
    for (const [key, ref] of referenceEmbeddings) {
        if (!grouped.has(ref.person)) {
            grouped.set(ref.person, []);
        }
        grouped.get(ref.person).push(ref);
    }
    return grouped;
}

function log(msg) {
    console.log(msg);
    if (logArea) {
        logArea.textContent += `${new Date().toLocaleTimeString()} ${msg}\n`;
        const lines = logArea.textContent.split('\n');
        if (lines.length > MAX_LOG_LINES) {
            logArea.textContent = lines.slice(-MAX_LOG_LINES).join('\n');
        }
        logArea.scrollTop = logArea.scrollHeight;
    }
}

/**
 * Update the current item preview thumbnail
 * @param {Object|null} item - Eagle item or null to reset
 * @param {string} status - Status: 'processing', 'success', 'skipped', 'error', or empty
 * @param {string} statusText - Status text to display
 */
function updateCurrentItemPreview(item, status = '', statusText = 'Waiting...') {
    if (!currentItemThumb || !currentItemName || !currentItemStatus) return;

    if (item) {
        // Get thumbnail path - Eagle items have thumbnailPath
        const thumbPath = item.thumbnailPath || item.filePath;
        if (thumbPath && fs.existsSync(thumbPath)) {
            currentItemThumb.src = thumbPath;
        } else {
            currentItemThumb.src = '';
        }
        currentItemName.textContent = item.name || 'Unknown';
        currentItemName.title = item.name || '';
    } else {
        currentItemThumb.src = '';
        currentItemName.textContent = '-';
        currentItemName.title = '';
    }

    currentItemStatus.textContent = statusText;
    currentItemStatus.className = 'current-item-status' + (status ? ` ${status}` : '');
}

// Eagle lifecycle
eagle.onPluginCreate((plugin) => {
    pluginPath = plugin.path;
    logArea = document.getElementById('logArea');
    logContent = document.getElementById('logContent');
    currentItemThumb = document.getElementById('currentItemThumb');
    currentItemName = document.getElementById('currentItemName');
    currentItemStatus = document.getElementById('currentItemStatus');
    log(`Plugin loaded from: ${pluginPath}`);
    initUI();
});

eagle.onPluginRun(() => {
    log('Plugin activated');
});

// Module loading
function getModule(modulePath) {
    return require(path.join(pluginPath, 'dist', modulePath));
}

// State
let desiredTags = [];
let aborted = false;
let allAppliedTags = []; // Track all tags applied in session

// ============================================
// PERFORMANCE: Embedding Cache
// Cache computed embeddings by item ID to avoid recomputation
// ============================================
const embeddingCache = new Map(); // itemId -> { clipEmbedding, faceDescriptors, timestamp }
const CACHE_MAX_SIZE = 500; // Limit cache size to prevent memory issues
const CACHE_TTL = 30 * 60 * 1000; // 30 minutes TTL

function getCachedEmbedding(itemId) {
    const cached = embeddingCache.get(itemId);
    if (cached && (Date.now() - cached.timestamp) < CACHE_TTL) {
        return cached;
    }
    // Expired or not found
    if (cached) embeddingCache.delete(itemId);
    return null;
}

function setCachedEmbedding(itemId, clipEmbedding, faceDescriptors) {
    // Evict oldest entries if cache is full
    if (embeddingCache.size >= CACHE_MAX_SIZE) {
        const oldestKey = embeddingCache.keys().next().value;
        embeddingCache.delete(oldestKey);
    }
    embeddingCache.set(itemId, {
        clipEmbedding,
        faceDescriptors,
        timestamp: Date.now()
    });
}

function clearEmbeddingCache() {
    embeddingCache.clear();
    log('[Cache] Embedding cache cleared');
}

// CLIP state (for person recognition) - using onnxruntime-node directly
let ort = null;
let clipSession = null;
const CLIP_MODEL_URL = 'https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/vision_model_quantized.onnx';
const CLIP_MODEL_NAME = 'clip-vit-base-patch32-vision.onnx';

// CLIP ViT-B/32 preprocessing constants
const CLIP_IMAGE_SIZE = 224;
const CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073];
const CLIP_STD = [0.26862954, 0.26130258, 0.27577711];

async function downloadModelIfNeeded(modelDir, modelName, modelUrl) {
    const modelPath = path.join(modelDir, modelName);
    if (fs.existsSync(modelPath)) {
        log('[CLIP] Model already cached: ' + modelPath);
        return modelPath;
    }

    log('[CLIP] Downloading model from: ' + modelUrl);
    const https = require('https');
    const http = require('http');

    return new Promise((resolve, reject) => {
        const file = fs.createWriteStream(modelPath);
        const protocol = modelUrl.startsWith('https') ? https : http;

        const request = (url) => {
            protocol.get(url, (response) => {
                // Handle redirects
                if (response.statusCode === 301 || response.statusCode === 302) {
                    log('[CLIP] Following redirect to: ' + response.headers.location);
                    request(response.headers.location);
                    return;
                }

                if (response.statusCode !== 200) {
                    reject(new Error(`Download failed: ${response.statusCode}`));
                    return;
                }

                const totalSize = parseInt(response.headers['content-length'], 10);
                let downloaded = 0;

                response.on('data', (chunk) => {
                    downloaded += chunk.length;
                    if (totalSize) {
                        const pct = Math.round((downloaded / totalSize) * 100);
                        if (pct % 10 === 0) {
                            log(`[CLIP] Download progress: ${pct}%`);
                        }
                    }
                });

                response.pipe(file);
                file.on('finish', () => {
                    file.close();
                    log('[CLIP] Model downloaded: ' + modelPath);
                    resolve(modelPath);
                });
            }).on('error', (err) => {
                fs.unlink(modelPath, () => {});
                reject(err);
            });
        };

        request(modelUrl);
    });
}

async function initClip() {
    if (clipSession) return true;
    try {
        log('[CLIP] Loading ONNX runtime...');

        // Load onnxruntime-node
        const onnxNodePath = path.join(pluginPath, 'node_modules', 'onnxruntime-node');
        ort = require(onnxNodePath);
        log('[CLIP] ONNX Runtime Node loaded');

        // Create cache directory
        const cacheDir = path.join(pluginPath, '.cache', 'models');
        if (!fs.existsSync(cacheDir)) {
            fs.mkdirSync(cacheDir, { recursive: true });
        }

        // Download model if needed
        const modelPath = await downloadModelIfNeeded(cacheDir, CLIP_MODEL_NAME, CLIP_MODEL_URL);

        // Create inference session
        log('[CLIP] Creating inference session...');
        clipSession = await ort.InferenceSession.create(modelPath, {
            executionProviders: ['cpu'],
        });
        log('[CLIP] Session created. Input names: ' + clipSession.inputNames.join(', '));
        log('[CLIP] Output names: ' + clipSession.outputNames.join(', '));
        log('[CLIP] Model loaded successfully');
        return true;
    } catch (err) {
        log('[CLIP] Failed to initialize: ' + err.message);
        log('[CLIP] Stack: ' + (err.stack || '').split('\n').slice(0, 5).join(' | '));
        console.error('[CLIP] Init error:', err);
        return false;
    }
}

function isClipReady() {
    return clipSession !== null;
}

async function extractClipEmbedding(imagePath) {
    if (!clipSession) {
        throw new Error('CLIP not initialized');
    }

    // Preprocess image using sharp
    // 1. Resize to 224x224 with center crop
    // 2. Convert to RGB float tensor
    // 3. Normalize with CLIP mean/std

    const imageBuffer = await sharp(imagePath)
        .rotate() // Auto-rotate based on EXIF orientation
        .resize(CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE, {
            fit: 'cover',
            position: 'center',
        })
        .removeAlpha()
        .raw()
        .toBuffer();

    // Convert to float and normalize
    // Input shape: [1, 3, 224, 224] (NCHW format)
    const pixels = new Float32Array(1 * 3 * CLIP_IMAGE_SIZE * CLIP_IMAGE_SIZE);

    for (let c = 0; c < 3; c++) {
        for (let h = 0; h < CLIP_IMAGE_SIZE; h++) {
            for (let w = 0; w < CLIP_IMAGE_SIZE; w++) {
                const srcIdx = (h * CLIP_IMAGE_SIZE + w) * 3 + c; // HWC format from sharp
                const dstIdx = c * CLIP_IMAGE_SIZE * CLIP_IMAGE_SIZE + h * CLIP_IMAGE_SIZE + w; // CHW format

                // Normalize: (pixel/255 - mean) / std
                const pixelValue = imageBuffer[srcIdx] / 255.0;
                pixels[dstIdx] = (pixelValue - CLIP_MEAN[c]) / CLIP_STD[c];
            }
        }
    }

    // Create tensor and run inference
    const inputTensor = new ort.Tensor('float32', pixels, [1, 3, CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE]);
    const feeds = { [clipSession.inputNames[0]]: inputTensor };

    const results = await clipSession.run(feeds);

    // Get the output embedding (usually last_hidden_state or image_embeds)
    const outputName = clipSession.outputNames[0];
    const output = results[outputName];

    // Pool and normalize the embedding
    // For vision model, output is usually [1, num_patches, hidden_dim] or [1, hidden_dim]
    let embedding;
    if (output.dims.length === 3) {
        // Mean pool across patches: [1, num_patches, hidden_dim] -> [hidden_dim]
        const [batch, patches, dim] = output.dims;
        embedding = new Float32Array(dim);
        for (let d = 0; d < dim; d++) {
            let sum = 0;
            for (let p = 0; p < patches; p++) {
                sum += output.data[p * dim + d];
            }
            embedding[d] = sum / patches;
        }
    } else if (output.dims.length === 2) {
        // Already pooled: [1, hidden_dim]
        embedding = new Float32Array(output.data);
    } else {
        embedding = new Float32Array(output.data);
    }

    // L2 normalize
    let norm = 0;
    for (let i = 0; i < embedding.length; i++) {
        norm += embedding[i] * embedding[i];
    }
    norm = Math.sqrt(norm);
    if (norm > 0) {
        for (let i = 0; i < embedding.length; i++) {
            embedding[i] /= norm;
        }
    }

    return embedding;
}

function cosineSimilarity(vec1, vec2) {
    if (!vec1 || !vec2 || vec1.length !== vec2.length) return 0;
    let dot = 0, norm1 = 0, norm2 = 0;
    for (let i = 0; i < vec1.length; i++) {
        dot += vec1[i] * vec2[i];
        norm1 += vec1[i] * vec1[i];
        norm2 += vec2[i] * vec2[i];
    }
    const denom = Math.sqrt(norm1) * Math.sqrt(norm2);
    return denom > 0 ? dot / denom : 0;
}

// Euclidean distance for face-api embeddings
function euclideanDistance(vec1, vec2) {
    if (!vec1 || !vec2 || vec1.length !== vec2.length) return Infinity;
    let sum = 0;
    for (let i = 0; i < vec1.length; i++) {
        const diff = vec1[i] - vec2[i];
        sum += diff * diff;
    }
    return Math.sqrt(sum);
}

// ============================================
// FACE RECOGNITION PIPELINE (SCRFD + ArcFace)
// State-of-the-art face detection and recognition
// ============================================

let facePipeline = null;
let faceApiReady = false;  // Keep this name for backward compatibility

async function initFaceApi() {
    if (faceApiReady) return true;
    try {
        log('[Face] Loading SCRFD + ArcFace pipeline...');

        // Load the face recognition pipeline
        const pipelinePath = path.join(pluginPath, 'src', 'face');
        facePipeline = require(pipelinePath);

        // Initialize with models directory
        const modelsPath = path.join(pluginPath, 'models', 'face');
        log('[Face] Models path: ' + modelsPath);

        // Check if model files exist
        if (!fs.existsSync(path.join(modelsPath, 'det_10g.onnx'))) {
            throw new Error('SCRFD model (det_10g.onnx) not found in models/face/');
        }
        if (!fs.existsSync(path.join(modelsPath, 'arcfaceresnet100-8.onnx'))) {
            throw new Error('ArcFace model (arcfaceresnet100-8.onnx) not found in models/face/');
        }

        const success = await facePipeline.initPipeline(modelsPath);
        if (!success) {
            throw new Error('Failed to initialize face pipeline');
        }

        faceApiReady = true;
        log('[Face] SCRFD + ArcFace pipeline loaded successfully');
        log('[Face] Using 512-dim ArcFace embeddings (SOTA accuracy)');
        return true;
    } catch (err) {
        log('[Face] Failed to initialize: ' + err.message);
        console.error('[Face] Init error:', err);
        faceApiReady = false;
        return false;
    }
}

function isFaceApiReady() {
    return faceApiReady && facePipeline && facePipeline.isPipelineReady();
}

/**
 * Detect faces and extract 512-dim ArcFace embeddings
 * @param {string} imagePath - Path to image file
 * @param {string} debugPrefix - Optional prefix for debug file names
 * @returns {Promise<Array<{descriptor: number[], bbox: number[], confidence: number, alignedFaceFile?: string}>>}
 */
async function detectFacesAndExtractDescriptors(imagePath, debugPrefix = null) {
    if (!isFaceApiReady()) {
        log('[Face] Pipeline not ready, skipping face detection');
        return [];
    }

    try {
        // Read image (rotate() auto-applies EXIF orientation)
        const imageBuffer = await sharp(imagePath)
            .rotate() // Auto-rotate based on EXIF orientation
            .jpeg({ quality: 95 })
            .toBuffer();

        // Extract face embeddings using SCRFD + ArcFace pipeline
        // Request aligned faces if debug mode is enabled
        const faces = await facePipeline.extractFaceEmbeddings(imageBuffer, {
            returnAlignedFace: DEBUG_MODE
        });

        const results = [];
        for (let i = 0; i < faces.length; i++) {
            const f = faces[i];
            const result = {
                descriptor: Array.from(f.embedding), // 512-dim ArcFace embedding
                bbox: f.bbox,
                landmarks: f.landmarks,
                confidence: f.score,
            };

            // Save aligned face for debugging
            if (DEBUG_MODE && f.alignedFace && debugPrefix) {
                const filename = `${debugPrefix}_face${i}_aligned.jpg`;
                result.alignedFaceFile = await saveDebugImage(f.alignedFace, filename);
            }

            results.push(result);
        }

        return results;
    } catch (err) {
        log(`[Face] Detection error: ${err.message}`);
        return [];
    }
}

/**
 * Detect faces and extract embeddings from a buffer (for video frames)
 * @param {Buffer} imageBuffer - Image buffer
 * @param {Object} options - Options
 * @param {boolean} options.returnAlignedFace - If true, include aligned face buffer
 * @returns {Promise<Array<{descriptor: number[], bbox: number[], confidence: number, alignedFace?: Buffer}>>}
 */
async function detectFacesFromBuffer(imageBuffer, options = {}) {
    if (!isFaceApiReady()) {
        return [];
    }

    try {
        const faces = await facePipeline.extractFaceEmbeddings(imageBuffer, {
            returnAlignedFace: options.returnAlignedFace || false
        });
        return faces.map(f => ({
            descriptor: Array.from(f.embedding),
            bbox: f.bbox,
            landmarks: f.landmarks,
            confidence: f.score,
            alignedFace: f.alignedFace || null,
        }));
    } catch (err) {
        log(`[Face] Buffer detection error: ${err.message}`);
        return [];
    }
}

// ============================================
// SMART VIDEO FRAME SELECTION FOR FACE RECOGNITION
// Extracts multiple frames and finds the one with the best face
// ============================================

const VIDEO_EXTENSIONS = ['mp4', 'mov', 'avi', 'mkv', 'webm', 'ts', 'm4v', 'wmv'];

function isVideoExtension(ext) {
    // Normalize: remove leading dot if present, lowercase
    const normalized = (ext || '').toLowerCase().replace(/^\./, '');
    return VIDEO_EXTENSIONS.includes(normalized);
}

// Find ffmpeg/ffprobe binaries
function findBinary(name) {
    const candidates = [
        `/opt/homebrew/bin/${name}`,
        `/usr/local/bin/${name}`,
        `/usr/bin/${name}`,
        name,
    ];
    for (const candidate of candidates) {
        try {
            if (candidate === name) return name;
            if (fs.existsSync(candidate)) return candidate;
        } catch {}
    }
    return name;
}

const FFPROBE_PATH = findBinary('ffprobe');
const FFMPEG_PATH = findBinary('ffmpeg');

async function getVideoDuration(videoPath) {
    const { execFile } = require('child_process');
    const { promisify } = require('util');
    const execFileAsync = promisify(execFile);

    try {
        const { stdout } = await execFileAsync(FFPROBE_PATH, [
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            videoPath,
        ]);
        return parseFloat(stdout.trim());
    } catch (e) {
        log(`[Video] Failed to get duration: ${e.message}`);
        return 10; // Default fallback
    }
}

async function extractFrameAt(videoPath, timestampSeconds, outputPath) {
    const { execFile } = require('child_process');
    const { promisify } = require('util');
    const execFileAsync = promisify(execFile);

    await execFileAsync(FFMPEG_PATH, [
        '-ss', String(timestampSeconds),
        '-i', videoPath,
        '-vframes', '1',
        '-q:v', '2',
        '-y',
        outputPath,
    ]);
}

// Find the best frame for face recognition from a video
// Optimized: 5 frames, parallel extraction, early termination on high confidence
async function findBestFaceFrame(videoPath, numFrames = 5, options = {}) {
    if (!isFaceApiReady()) {
        log(`[Video] Face pipeline not ready, skipping video scan`);
        return null;
    }

    // Check if video file exists
    if (!fs.existsSync(videoPath)) {
        log(`[Video] File not found: ${videoPath}`);
        return null;
    }

    const os = require('os');
    let tmpDir;
    try {
        tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'eagle-face-'));
    } catch (e) {
        log(`[Video] Failed to create temp dir: ${e.message}`);
        return null;
    }

    try {
        const duration = await getVideoDuration(videoPath);

        // Generate timestamps spread across the video, avoiding very start/end
        const timestamps = [];
        const startOffset = Math.min(0.5, duration * 0.05);
        const endOffset = Math.min(0.5, duration * 0.05);
        const usableDuration = duration - startOffset - endOffset;

        for (let i = 0; i < numFrames; i++) {
            const t = startOffset + (usableDuration * i / (numFrames - 1));
            timestamps.push(Math.max(0, Math.min(t, duration - 0.1)));
        }

        // OPTIMIZATION: Extract all frames in parallel
        const framePaths = timestamps.map((_, i) => path.join(tmpDir, `frame_${i}.jpg`));
        await Promise.all(timestamps.map((ts, i) =>
            extractFrameAt(videoPath, ts, framePaths[i]).catch(() => {})
        ));

        let bestFrame = null;
        let bestScore = 0;

        // Process frames and find best face using SCRFD + ArcFace
        for (let i = 0; i < framePaths.length; i++) {
            const framePath = framePaths[i];

            try {
                if (!fs.existsSync(framePath)) continue;

                // Convert to JPEG buffer for face detection
                const imageBuffer = await sharp(framePath)
                    .rotate() // Auto-rotate based on EXIF orientation
                    .resize({ width: 640, withoutEnlargement: true })
                    .jpeg({ quality: 85 })
                    .toBuffer();

                // Detect faces using new pipeline
                const detections = await detectFacesFromBuffer(imageBuffer, {
                    returnAlignedFace: options.returnAlignedFace || false
                });

                if (detections.length > 0) {
                    const avgConfidence = detections.reduce((sum, d) => sum + d.confidence, 0) / detections.length;
                    const maxFaceArea = Math.max(...detections.map(d => {
                        const w = d.bbox[2] - d.bbox[0];
                        const h = d.bbox[3] - d.bbox[1];
                        return w * h;
                    }));

                    // Calculate face tilt penalty based on eye landmarks
                    // Upright faces (eyes level) are preferred for recognition
                    let tiltPenalty = 1.0;
                    for (const det of detections) {
                        if (det.landmarks && det.landmarks.length >= 2) {
                            const leftEye = det.landmarks[0];
                            const rightEye = det.landmarks[1];
                            // Calculate tilt angle from eye positions
                            const dx = rightEye[0] - leftEye[0];
                            const dy = rightEye[1] - leftEye[1];
                            const tiltAngle = Math.abs(Math.atan2(dy, dx)) * (180 / Math.PI);
                            // Penalize tilts > 15 degrees (reduce score for tilted faces)
                            if (tiltAngle > 15) {
                                tiltPenalty = Math.min(tiltPenalty, Math.max(0.3, 1 - (tiltAngle - 15) / 45));
                            }
                        }
                    }

                    const score = avgConfidence * (1 + Math.log(detections.length + 1)) * (1 + maxFaceArea / 10000) * tiltPenalty;

                    if (score > bestScore) {
                        bestScore = score;
                        bestFrame = {
                            buffer: imageBuffer,
                            timestamp: timestamps[i],
                            faceCount: detections.length,
                            confidence: avgConfidence,
                            detections: detections,
                        };

                        // OPTIMIZATION: Early termination if we found an excellent frame
                        // Requires high confidence AND large face AND upright face for reliable recognition
                        if (avgConfidence >= 0.95 && maxFaceArea >= 15000 && tiltPenalty >= 0.9 && detections.length >= 1) {
                            log(`[Video] Found excellent frame at ${timestamps[i].toFixed(1)}s (conf: ${avgConfidence.toFixed(2)}, area: ${maxFaceArea.toFixed(0)}, upright), stopping early`);
                            break;
                        }
                    }
                }
            } catch (e) {
                // Skip failed frames silently
            }
        }

        if (bestFrame) {
            // Calculate tilt for logging
            let tiltInfo = '';
            if (bestFrame.detections[0]?.landmarks?.length >= 2) {
                const leftEye = bestFrame.detections[0].landmarks[0];
                const rightEye = bestFrame.detections[0].landmarks[1];
                const dx = rightEye[0] - leftEye[0];
                const dy = rightEye[1] - leftEye[1];
                const tiltAngle = Math.atan2(dy, dx) * (180 / Math.PI);
                tiltInfo = `, tilt: ${tiltAngle.toFixed(1)}°`;
            }
            log(`[Video] Best: ${bestFrame.timestamp.toFixed(1)}s, ${bestFrame.faceCount} face(s), conf: ${bestFrame.confidence.toFixed(2)}${tiltInfo}`);
            return {
                buffer: Buffer.from(bestFrame.buffer),
                faceCount: bestFrame.faceCount,
                confidence: bestFrame.confidence,
                detections: bestFrame.detections,
            };
        } else {
            log(`[Video] No faces found`);
            return null;
        }
    } finally {
        // Clean up temp files
        try {
            fs.rmSync(tmpDir, { recursive: true, force: true });
        } catch {}
    }
}

// Extract CLIP embedding from a buffer instead of file path
async function extractClipEmbeddingFromBuffer(imageBuffer) {
    if (!clipSession) {
        log('[CLIP] Not initialized');
        return null;
    }

    try {
        // Preprocess image
        const { data, info } = await sharp(imageBuffer)
            .rotate() // Auto-rotate based on EXIF orientation
            .resize(CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE, { fit: 'cover' })
            .removeAlpha()
            .raw()
            .toBuffer({ resolveWithObject: true });

        // Normalize and convert to NCHW format
        const floatData = new Float32Array(3 * CLIP_IMAGE_SIZE * CLIP_IMAGE_SIZE);
        const pixelCount = CLIP_IMAGE_SIZE * CLIP_IMAGE_SIZE;

        for (let i = 0; i < pixelCount; i++) {
            const r = data[i * 3] / 255;
            const g = data[i * 3 + 1] / 255;
            const b = data[i * 3 + 2] / 255;

            floatData[i] = (r - CLIP_MEAN[0]) / CLIP_STD[0];
            floatData[pixelCount + i] = (g - CLIP_MEAN[1]) / CLIP_STD[1];
            floatData[2 * pixelCount + i] = (b - CLIP_MEAN[2]) / CLIP_STD[2];
        }

        const inputTensor = new ort.Tensor('float32', floatData, [1, 3, CLIP_IMAGE_SIZE, CLIP_IMAGE_SIZE]);
        const results = await clipSession.run({ 'pixel_values': inputTensor });
        const embedding = results['image_embeds'].data;

        // L2 normalize
        let norm = 0;
        for (let i = 0; i < embedding.length; i++) {
            norm += embedding[i] * embedding[i];
        }
        norm = Math.sqrt(norm);
        const normalized = new Float32Array(embedding.length);
        for (let i = 0; i < embedding.length; i++) {
            normalized[i] = embedding[i] / norm;
        }

        return normalized;
    } catch (err) {
        log(`[CLIP] Buffer embedding error: ${err.message}`);
        return null;
    }
}

// ============================================
// ENHANCED EMBEDDING STORE
// Stores both CLIP and face embeddings per reference
// ============================================

const referenceEmbeddings = new Map();
// Structure: key -> { person, feature, clipEmbedding, faceDescriptors[], source, mtime }

// ============================================
// REFERENCE CACHE PERSISTENCE
// Save/load reference embeddings to disk
// ============================================

function getReferenceCachePath(folderName) {
    // Store cache in plugin directory
    const safeName = folderName.replace(/[^a-zA-Z0-9]/g, '_').toLowerCase();
    return path.join(pluginPath, `ref_cache_${safeName}.json`);
}

function saveReferenceCache(folderName) {
    try {
        const cachePath = getReferenceCachePath(folderName);
        const cacheData = {
            version: 3, // Cache format version (v3 = 512-dim ArcFace embeddings)
            folderName,
            timestamp: Date.now(),
            references: Array.from(referenceEmbeddings.entries()).map(([key, ref]) => ({
                key,
                ...ref
            }))
        };
        fs.writeFileSync(cachePath, JSON.stringify(cacheData));
        log(`[Cache] Saved ${referenceEmbeddings.size} references to cache`);
        return true;
    } catch (e) {
        log(`[Cache] Failed to save: ${e.message}`);
        return false;
    }
}

function loadReferenceCache(folderName) {
    try {
        const cachePath = getReferenceCachePath(folderName);
        if (!fs.existsSync(cachePath)) {
            log(`[Cache] No cache found for "${folderName}"`);
            return null;
        }

        const cacheData = JSON.parse(fs.readFileSync(cachePath, 'utf8'));

        if (cacheData.version !== 3) {
            log(`[Cache] Cache version mismatch (have v${cacheData.version}, need v3), will rebuild with ArcFace`);
            return null;
        }

        return cacheData;
    } catch (e) {
        log(`[Cache] Failed to load: ${e.message}`);
        return null;
    }
}

// Returns detailed cache diff: { valid, unchanged, modified, added, removed }
async function analyzeCacheDiff(cacheData, folderItems, itemPersonMap) {
    const result = {
        valid: true,           // True if cache can be used (even partially)
        fullyValid: true,      // True if no changes at all
        unchanged: [],         // Cached refs that are still valid
        modified: [],          // Items that need re-embedding (modified files)
        added: [],             // New items not in cache
        removed: [],           // Cached refs whose files no longer exist
    };

    // Build a map of source path -> {mtime, item, personName} from folder items
    const currentFiles = new Map();
    for (const item of folderItems) {
        const filePath = item.filePath || item.thumbnailPath;
        if (filePath && fs.existsSync(filePath)) {
            try {
                const stats = fs.statSync(filePath);
                currentFiles.set(filePath, {
                    mtime: stats.mtimeMs,
                    item,
                    personName: itemPersonMap ? itemPersonMap.get(item.id) : null
                });
            } catch (e) {
                // Skip files we can't stat
            }
        }
    }

    // Build set of cached sources for quick lookup
    const cachedSources = new Map(); // source -> ref
    for (const ref of cacheData.references) {
        cachedSources.set(ref.source, ref);
    }

    // Check each cached reference
    for (const ref of cacheData.references) {
        const current = currentFiles.get(ref.source);
        if (!current) {
            // File was deleted
            result.removed.push(ref);
            result.fullyValid = false;
        } else if (ref.mtime && Math.abs(current.mtime - ref.mtime) > 1000) {
            // File was modified
            result.modified.push({
                item: current.item,
                personName: current.personName || ref.person,
                oldRef: ref
            });
            result.fullyValid = false;
        } else {
            // File unchanged - keep cached embedding
            result.unchanged.push(ref);
        }
    }

    // Check for new files not in cache
    for (const [filePath, data] of currentFiles) {
        if (!cachedSources.has(filePath)) {
            result.added.push({
                item: data.item,
                personName: data.personName,
                filePath
            });
            result.fullyValid = false;
        }
    }

    if (result.modified.length > 0 || result.added.length > 0 || result.removed.length > 0) {
        log(`[Cache] Diff: ${result.unchanged.length} unchanged, ${result.added.length} added, ${result.modified.length} modified, ${result.removed.length} removed`);
    }

    return result;
}

function restoreReferencesFromCache(cacheData) {
    referenceEmbeddings.clear();
    for (const ref of cacheData.references) {
        referenceEmbeddings.set(ref.key, {
            person: ref.person,
            feature: ref.feature,
            clipEmbedding: ref.clipEmbedding,
            faceDescriptors: ref.faceDescriptors || [],
            source: ref.source,
            mtime: ref.mtime
        });
    }
    log(`[Cache] Restored ${referenceEmbeddings.size} references from cache`);
}

// Restore only the unchanged references from cache
function restorePartialCache(unchangedRefs) {
    referenceEmbeddings.clear();
    for (const ref of unchangedRefs) {
        referenceEmbeddings.set(ref.key, {
            person: ref.person,
            feature: ref.feature,
            clipEmbedding: ref.clipEmbedding,
            faceDescriptors: ref.faceDescriptors || [],
            source: ref.source,
            mtime: ref.mtime
        });
    }
    log(`[Cache] Restored ${unchangedRefs.length} unchanged references from cache`);
}

function clearReferences() {
    referenceEmbeddings.clear();
}

function deleteReferenceCache(folderName) {
    try {
        const cachePath = getReferenceCachePath(folderName);
        if (fs.existsSync(cachePath)) {
            fs.unlinkSync(cachePath);
            log(`[Cache] Deleted cache file: ${cachePath}`);
            return true;
        }
        log(`[Cache] No cache file to delete`);
        return false;
    } catch (e) {
        log(`[Cache] Failed to delete cache: ${e.message}`);
        return false;
    }
}

function addReference(person, feature, clipEmbedding, faceDescriptors, source) {
    const key = `${person}.${feature}.${Date.now()}`; // Unique key per reference

    // Get file mtime for cache validation
    let mtime = null;
    try {
        if (source && fs.existsSync(source)) {
            mtime = fs.statSync(source).mtimeMs;
        }
    } catch (e) {}

    referenceEmbeddings.set(key, {
        person,
        feature,
        clipEmbedding: clipEmbedding ? Array.from(clipEmbedding) : null,
        faceDescriptors: faceDescriptors || [],
        source,
        mtime,
    });
}

// Get all references for a person
function getPersonReferences(personName) {
    const refs = [];
    for (const ref of referenceEmbeddings.values()) {
        if (ref.person === personName) {
            refs.push(ref);
        }
    }
    return refs;
}

// Get all unique person names
function getAllPersonNames() {
    const people = new Set();
    for (const ref of referenceEmbeddings.values()) {
        people.add(ref.person);
    }
    return Array.from(people);
}

// ============================================
// ENHANCED MATCHING WITH ENSEMBLE SCORING
// ============================================

function findMatchesEnhanced(queryClipEmbedding, queryFaceDescriptors, userThreshold) {
    const personScores = new Map(); // person -> { clipScores[], faceScores[], refs[] }


    // Calculate scores for all references
    for (const [key, ref] of referenceEmbeddings) {
        if (!personScores.has(ref.person)) {
            personScores.set(ref.person, { clipScores: [], faceScores: [], refs: [], features: [] });
        }
        const personData = personScores.get(ref.person);
        personData.refs.push(ref);
        personData.features.push(ref.feature);

        // CLIP similarity
        if (queryClipEmbedding && ref.clipEmbedding) {
            const clipSim = cosineSimilarity(
                queryClipEmbedding,
                new Float32Array(ref.clipEmbedding)
            );
            personData.clipScores.push(clipSim);
        }

        // ArcFace matching (if both have face descriptors)
        // ArcFace embeddings are L2-normalized, so cosine similarity = dot product
        if (queryFaceDescriptors && queryFaceDescriptors.length > 0 && ref.faceDescriptors && ref.faceDescriptors.length > 0) {
            // Find best face match between query faces and reference faces
            for (const queryFace of queryFaceDescriptors) {
                for (const refFace of ref.faceDescriptors) {
                    // Use cosine similarity directly for ArcFace (embeddings are L2-normalized)
                    // ArcFace similarity: -1 to 1, typical threshold ~0.4 for same person
                    const similarity = cosineSimilarity(
                        new Float32Array(queryFace),
                        new Float32Array(refFace)
                    );
                    personData.faceScores.push(similarity);
                }
            }
        }
    }

    // Calculate ensemble scores per person
    const allCandidates = [];
    for (const [person, data] of personScores) {
        // Get best CLIP score
        const bestClipScore = data.clipScores.length > 0
            ? Math.max(...data.clipScores)
            : 0;

        // Get best face score
        const bestFaceScore = data.faceScores.length > 0
            ? Math.max(...data.faceScores)
            : 0;

        // Debug: show best scores for each person
        if (bestFaceScore > 0 || bestClipScore > 0.5) {
            log(`  -> ${person}: face=${bestFaceScore.toFixed(2)}, clip=${bestClipScore.toFixed(2)}`);
        }

        // Count matching references (using internal thresholds for counting)
        const clipMatchCount = data.clipScores.filter(s => s >= RECOGNITION_CONFIG.thresholds.general).length;
        const faceMatchCount = data.faceScores.filter(s => s >= RECOGNITION_CONFIG.thresholds.face).length;

        // Calculate score based on recognition mode
        let ensembleScore;
        let matchType;
        const mode = RECOGNITION_CONFIG.mode || 'ensemble';

        if (mode === 'clip') {
            // CLIP-only mode: ignore face scores
            ensembleScore = bestClipScore;
            matchType = 'clip';
        } else if (mode === 'face') {
            // Face-only mode: ignore CLIP scores
            if (bestFaceScore > 0) {
                ensembleScore = bestFaceScore;
                matchType = 'face';
            } else {
                // No face detected, can't match in face-only mode
                ensembleScore = 0;
                matchType = 'face';
            }
        } else {
            // Ensemble mode: combine both scores
            if (bestFaceScore > 0 && bestClipScore > 0) {
                // Both available - weighted ensemble
                ensembleScore = (bestFaceScore * RECOGNITION_CONFIG.weights.faceApi) +
                               (bestClipScore * RECOGNITION_CONFIG.weights.clip);
                matchType = 'ensemble';
            } else if (bestFaceScore > 0) {
                // Face only
                ensembleScore = bestFaceScore;
                matchType = 'face';
            } else {
                // CLIP only
                ensembleScore = bestClipScore * RECOGNITION_CONFIG.weights.clipOnly;
                matchType = 'clip';
            }
        }

        // Apply multi-reference confidence boost
        const totalMatches = clipMatchCount + faceMatchCount;
        if (totalMatches > 1) {
            ensembleScore += (totalMatches - 1) * RECOGNITION_CONFIG.confidenceBoost;
            ensembleScore = Math.min(ensembleScore, 1.0); // Cap at 1.0
        }

        allCandidates.push({
            person,
            confidence: ensembleScore,
            clipScore: bestClipScore,
            faceScore: bestFaceScore,
            matchType,
            refCount: data.refs.length,
            matchingRefs: totalMatches,
            features: [...new Set(data.features)],
        });
    }

    // Sort by confidence descending
    allCandidates.sort((a, b) => b.confidence - a.confidence);

    // Apply user threshold - only return matches above the threshold
    // For CLIP-only matches, be stricter since CLIP isn't person-specific
    const matches = allCandidates.filter(m => {
        if (m.matchType === 'clip') {
            // CLIP-only: require meeting user threshold
            return m.confidence >= userThreshold;
        } else if (m.matchType === 'face') {
            // Face-only: use user threshold (face is more reliable)
            return m.confidence >= userThreshold;
        } else {
            // Ensemble: use user threshold
            return m.confidence >= userThreshold;
        }
    });

    return matches;
}

// Legacy findMatches for backward compatibility
function findMatches(queryEmbedding, threshold) {
    return findMatchesEnhanced(queryEmbedding, null, threshold);
}

function parseReferenceName(name) {
    // Format: "PersonName_feature" or just "PersonName"
    const parts = name.split('_');
    if (parts.length >= 2) {
        return { person: parts[0], feature: parts.slice(1).join('_') };
    }
    return { person: name, feature: 'general' };
}

function getReferenceStats() {
    const people = new Set();
    for (const ref of referenceEmbeddings.values()) {
        people.add(ref.person);
    }
    return { count: referenceEmbeddings.size, people: Array.from(people) };
}

const DEFAULT_MODELS = {
    openrouter: 'nvidia/nemotron-nano-12b-v2-vl:free',
    groq: 'meta-llama/llama-4-scout-17b-16e-instruct',
};

function initUI() {
    // DOM refs
    const providerEl = document.getElementById('provider');
    const providerBadge = document.getElementById('providerBadge');
    const apiKeyEl = document.getElementById('apiKey');
    const keyToggle = document.getElementById('keyToggle');
    const modelEl = document.getElementById('model');
    const modelHintEl = document.getElementById('modelHint');
    const batchSizeEl = document.getElementById('batchSize');
    const customPromptEl = document.getElementById('customPrompt');
    const tagInputEl = document.getElementById('tagInput');
    const addTagBtn = document.getElementById('addTagBtn');
    const tagPillsEl = document.getElementById('tagPills');
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const progressPanel = document.getElementById('progressPanel');
    const progressFill = document.getElementById('progressFill');
    const progressCount = document.getElementById('progressCount');
    const progressSpeed = document.getElementById('progressSpeed');
    const progressStatus = document.getElementById('progressStatus');
    const summaryPanel = document.getElementById('summaryPanel');
    const toastEl = document.getElementById('toast');
    const logToggle = document.getElementById('logToggle');
    const presetSelect = document.getElementById('presetSelect');
    const savePresetBtn = document.getElementById('savePresetBtn');
    const deletePresetBtn = document.getElementById('deletePresetBtn');
    const presetNameRow = document.getElementById('presetNameRow');
    const presetNameInput = document.getElementById('presetNameInput');
    const presetNameConfirm = document.getElementById('presetNameConfirm');
    const presetNameCancel = document.getElementById('presetNameCancel');
    const convertBtn = document.getElementById('convertBtn');
    const confirmPanel = document.getElementById('confirmPanel');
    const confirmList = document.getElementById('confirmList');
    const confirmDeleteBtn = document.getElementById('confirmDeleteBtn');
    const confirmKeepBtn = document.getElementById('confirmKeepBtn');

    // Recognition DOM refs
    const refFolderPath = document.getElementById('refFolderPath');
    const refStatus = document.getElementById('refStatus');
    const matchThreshold = document.getElementById('matchThreshold');
    const thresholdValue = document.getElementById('thresholdValue');
    const refProgress = document.getElementById('refProgress');
    const refProgressFill = document.getElementById('refProgressFill');
    const refProgressStatus = document.getElementById('refProgressStatus');
    const loadRefsBtn = document.getElementById('loadRefsBtn');
    const clearCacheBtn = document.getElementById('clearCacheBtn');
    const identifyBtn = document.getElementById('identifyBtn');

    // Advanced recognition settings DOM refs
    const recognitionMode = document.getElementById('recognitionMode');
    const advancedToggle = document.getElementById('advancedToggle');
    const advancedPanel = document.getElementById('advancedPanel');
    const faceThreshold = document.getElementById('faceThreshold');
    const faceThresholdValue = document.getElementById('faceThresholdValue');
    const clipThreshold = document.getElementById('clipThreshold');
    const clipThresholdValue = document.getElementById('clipThresholdValue');
    const faceWeight = document.getElementById('faceWeight');
    const faceWeightValue = document.getElementById('faceWeightValue');
    const clipWeightValue = document.getElementById('clipWeightValue');
    const faceThresholdRow = document.getElementById('faceThresholdRow');
    const clipThresholdRow = document.getElementById('clipThresholdRow');
    const ensembleWeightRow = document.getElementById('ensembleWeightRow');
    const debugModeCheckbox = document.getElementById('debugMode');
    const videoFramesSlider = document.getElementById('videoFrames');
    const videoFramesValue = document.getElementById('videoFramesValue');
    const skipTaggedCheckbox = document.getElementById('skipTagged');
    const replaceTagsCheckbox = document.getElementById('replaceTags');
    const ignoreTagInput = document.getElementById('ignoreTagInput');
    const addIgnoreTagBtn = document.getElementById('addIgnoreTagBtn');
    const ignoreTagPills = document.getElementById('ignoreTagPills');

    // Toast
    let toastTimeout = null;
    function showToast(message, type) {
        toastEl.textContent = message;
        toastEl.className = `toast ${type} show`;
        clearTimeout(toastTimeout);
        toastTimeout = setTimeout(() => { toastEl.className = 'toast'; }, 4000);
    }

    // Log toggle
    logToggle.addEventListener('click', () => {
        logToggle.classList.toggle('open');
        logContent.classList.toggle('open');
    });

    // API key toggle
    keyToggle.addEventListener('click', () => {
        if (apiKeyEl.type === 'password') {
            apiKeyEl.type = 'text';
            keyToggle.textContent = 'Hide';
        } else {
            apiKeyEl.type = 'password';
            keyToggle.textContent = 'Show';
        }
    });

    // Config persistence
    function loadConfig() {
        try {
            const saved = localStorage.getItem('eagleAiTaggerConfig');
            if (saved) {
                const config = JSON.parse(saved);
                providerEl.value = config.apiProvider || 'openrouter';
                apiKeyEl.value = config.apiKey || '';
                modelEl.value = config.model || '';
                batchSizeEl.value = config.batchSize || 5;
                customPromptEl.value = config.customPrompt || '';
                desiredTags = config.desiredTags || [];
                renderTags();
                updateProviderUI();
            }
        } catch (e) { log(`Config load error: ${e.message}`); }
    }

    function saveConfig() {
        localStorage.setItem('eagleAiTaggerConfig', JSON.stringify({
            apiProvider: providerEl.value,
            apiKey: apiKeyEl.value,
            model: modelEl.value,
            batchSize: parseInt(batchSizeEl.value) || 5,
            customPrompt: customPromptEl.value,
            desiredTags,
        }));
    }

    function updateProviderUI() {
        const provider = providerEl.value;
        modelHintEl.textContent = `Default: ${DEFAULT_MODELS[provider] || ''}`;
        providerBadge.textContent = provider === 'openrouter' ? 'OpenRouter' : 'Groq';
    }

    // Preset management
    function getPresets() {
        try {
            return JSON.parse(localStorage.getItem('eagleAiTaggerPresets') || '[]');
        } catch { return []; }
    }

    function savePresets(presets) {
        localStorage.setItem('eagleAiTaggerPresets', JSON.stringify(presets));
    }

    function renderPresetDropdown(selectName) {
        const presets = getPresets();
        presetSelect.innerHTML = '<option value="">-- No preset --</option>' +
            presets.map(p => `<option value="${p.name}">${p.name}</option>`).join('');
        if (selectName) presetSelect.value = selectName;
    }

    function applyPreset(preset) {
        providerEl.value = preset.apiProvider || 'openrouter';
        modelEl.value = preset.model || '';
        batchSizeEl.value = preset.batchSize || 5;
        customPromptEl.value = preset.customPrompt || '';
        desiredTags = preset.desiredTags ? [...preset.desiredTags] : [];
        renderTags();
        updateProviderUI();
        saveConfig();
        log(`Preset "${preset.name}" applied`);
    }

    presetSelect.addEventListener('change', () => {
        const name = presetSelect.value;
        if (!name) return;
        const presets = getPresets();
        const preset = presets.find(p => p.name === name);
        if (preset) applyPreset(preset);
    });

    function commitPresetSave() {
        const name = presetNameInput.value.trim();
        if (!name) return;
        const presets = getPresets();
        const existing = presets.findIndex(p => p.name === name);
        const preset = {
            name,
            apiProvider: providerEl.value,
            model: modelEl.value,
            batchSize: parseInt(batchSizeEl.value) || 5,
            customPrompt: customPromptEl.value,
            desiredTags: [...desiredTags],
        };
        if (existing >= 0) {
            presets[existing] = preset;
        } else {
            presets.push(preset);
        }
        savePresets(presets);
        renderPresetDropdown(name);
        presetNameRow.style.display = 'none';
        presetNameInput.value = '';
        showToast(`Preset "${name}" saved`, 'success');
        log(`Preset "${name}" saved`);
    }

    savePresetBtn.addEventListener('click', () => {
        presetNameInput.value = presetSelect.value || '';
        presetNameRow.style.display = 'flex';
        presetNameInput.focus();
    });

    presetNameConfirm.addEventListener('click', commitPresetSave);

    presetNameInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') { e.preventDefault(); commitPresetSave(); }
        if (e.key === 'Escape') { presetNameRow.style.display = 'none'; }
    });

    presetNameCancel.addEventListener('click', () => {
        presetNameRow.style.display = 'none';
        presetNameInput.value = '';
    });

    deletePresetBtn.addEventListener('click', () => {
        const name = presetSelect.value;
        if (!name) {
            showToast('Select a preset to delete', 'info');
            return;
        }
        const presets = getPresets().filter(p => p.name !== name);
        savePresets(presets);
        renderPresetDropdown('');
        showToast(`Preset "${name}" deleted`, 'info');
        log(`Preset "${name}" deleted`);
    });

    // Tag management
    function renderTags() {
        tagPillsEl.innerHTML = desiredTags.map((tag, i) =>
            `<span class="tag-pill">${tag}<span class="remove" data-index="${i}">&times;</span></span>`
        ).join('');
    }

    addTagBtn.addEventListener('click', () => {
        const tag = tagInputEl.value.trim().toLowerCase();
        if (tag && !desiredTags.includes(tag)) {
            desiredTags.push(tag);
            tagInputEl.value = '';
            renderTags();
            saveConfig();
        }
    });

    tagInputEl.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') { e.preventDefault(); addTagBtn.click(); }
    });

    tagPillsEl.addEventListener('click', (e) => {
        if (e.target.classList.contains('remove')) {
            desiredTags.splice(parseInt(e.target.dataset.index), 1);
            renderTags();
            saveConfig();
        }
    });

    providerEl.addEventListener('change', () => { updateProviderUI(); saveConfig(); });
    apiKeyEl.addEventListener('change', saveConfig);
    modelEl.addEventListener('change', saveConfig);
    batchSizeEl.addEventListener('change', saveConfig);
    customPromptEl.addEventListener('change', saveConfig);

    // Image reading with resize
    const MAX_DIMENSION = 1024;

    async function readImageAsBase64(filePath) {
        const buffer = await fs.promises.readFile(filePath);
        const resized = await sharp(buffer)
            .resize({ width: MAX_DIMENSION, height: MAX_DIMENSION, fit: 'inside', withoutEnlargement: true })
            .jpeg({ quality: 80 })
            .toBuffer();
        return resized.toString('base64');
    }

    // Summary display
    function showSummary(successCount, errorCount, skippedCount, elapsedMs) {
        document.getElementById('sumSuccess').textContent = successCount;
        document.getElementById('sumErrors').textContent = errorCount;
        document.getElementById('sumSkipped').textContent = skippedCount;

        // Duration
        const secs = Math.round(elapsedMs / 1000);
        const mins = Math.floor(secs / 60);
        const remSecs = secs % 60;
        document.getElementById('sumTime').textContent =
            mins > 0 ? `Completed in ${mins}m ${remSecs}s` : `Completed in ${secs}s`;

        // Tag cloud (top 15 most used tags)
        const tagCounts = {};
        allAppliedTags.forEach(t => { tagCounts[t] = (tagCounts[t] || 0) + 1; });
        const sorted = Object.entries(tagCounts).sort((a, b) => b[1] - a[1]).slice(0, 15);

        const cloudSection = document.getElementById('tagCloudSection');
        const cloudEl = document.getElementById('tagCloud');
        if (sorted.length > 0) {
            cloudEl.innerHTML = sorted.map(([tag, count]) =>
                `<span class="cloud-tag">${tag}<span class="tag-count">${count}</span></span>`
            ).join('');
            cloudSection.style.display = 'block';
        } else {
            cloudSection.style.display = 'none';
        }

        summaryPanel.classList.add('active');
    }

    // Progress update
    function updateProgress(processed, total, startTime) {
        const pct = Math.round((processed / total) * 100);
        progressFill.style.width = `${pct}%`;
        progressCount.textContent = `${processed} / ${total}`;

        const elapsed = Date.now() - startTime;
        if (processed > 0) {
            const rate = (processed / elapsed) * 60000; // items per minute
            const remaining = total - processed;
            const etaMs = (remaining / (processed / elapsed));
            const etaSecs = Math.round(etaMs / 1000);

            progressSpeed.textContent = `${rate.toFixed(1)}/min`;

            if (etaSecs > 60) {
                progressStatus.textContent = `~${Math.ceil(etaSecs / 60)}m remaining`;
            } else if (etaSecs > 0) {
                progressStatus.textContent = `~${etaSecs}s remaining`;
            } else {
                progressStatus.textContent = 'Finishing...';
            }
        }
    }

    // Main workflow
    startBtn.addEventListener('click', async () => {
        logArea.textContent = '';
        log('--- Start tagging ---');

        const apiKey = apiKeyEl.value.trim();
        if (!apiKey) {
            showToast('Please enter an API key', 'error');
            return;
        }

        let selectedItems;
        try {
            selectedItems = await eagle.item.getSelected();
        } catch (e) {
            showToast('Failed to get selected items', 'error');
            log(`ERROR: ${e.message}`);
            return;
        }

        if (!selectedItems || selectedItems.length === 0) {
            showToast('No items selected in Eagle', 'error');
            return;
        }

        // Load modules
        let tagImagesBatch, isVideoFile, extractAndCompositeFrames;
        try {
            tagImagesBatch = getModule('orchestrator').tagImagesBatch;
            const vp = getModule('utils/video-processor');
            isVideoFile = vp.isVideoFile;
            extractAndCompositeFrames = vp.extractAndCompositeFrames;
        } catch (e) {
            showToast('Failed to load modules', 'error');
            log(`ERROR: ${e.message}\n${e.stack}`);
            return;
        }

        const config = {
            apiProvider: providerEl.value,
            apiKey,
            model: modelEl.value || undefined,
            customPrompt: customPromptEl.value || undefined,
            desiredTags: desiredTags.length > 0 ? desiredTags : undefined,
            batchSize: parseInt(batchSizeEl.value) || 5,
        };

        saveConfig();
        aborted = false;
        allAppliedTags = [];

        // UI state: processing
        startBtn.disabled = true;
        stopBtn.style.display = 'block';
        summaryPanel.classList.remove('active');
        progressPanel.classList.add('active');
        progressFill.style.width = '0%';
        progressCount.textContent = `0 / ${selectedItems.length}`;
        progressSpeed.textContent = '';
        progressStatus.textContent = 'Loading images...';

        const startTime = Date.now();
        const total = selectedItems.length;
        const chunkSize = config.batchSize;
        let processed = 0;
        let successCount = 0;
        let errorCount = 0;
        let skippedCount = 0;

        log(`Processing ${total} items (${config.apiProvider}, batch=${chunkSize})`);

        try {
            for (let i = 0; i < total; i += chunkSize) {
                if (aborted) break;

                const chunk = selectedItems.slice(i, i + chunkSize);
                const images = [];

                for (const item of chunk) {
                    if (aborted) break;

                    const filePath = item.filePath;
                    let base64;

                    if (isVideoFile(filePath)) {
                        try {
                            const composite = await extractAndCompositeFrames(filePath);
                            base64 = composite.toString('base64');
                        } catch (e) {
                            try {
                                const thumbPath = item.thumbnailPath;
                                if (thumbPath) {
                                    base64 = await readImageAsBase64(thumbPath);
                                } else {
                                    skippedCount++;
                                    continue;
                                }
                            } catch (e2) {
                                log(`Skip ${item.name}: ${e2.message}`);
                                skippedCount++;
                                continue;
                            }
                        }
                    } else {
                        try {
                            base64 = await readImageAsBase64(filePath);
                        } catch (e) {
                            log(`Skip ${item.name}: ${e.message}`);
                            skippedCount++;
                            continue;
                        }
                    }

                    images.push({ id: item.id, base64 });
                }

                if (aborted || images.length === 0) {
                    processed += chunk.length;
                    updateProgress(processed, total, startTime);
                    continue;
                }

                progressStatus.textContent = `Tagging chunk ${Math.floor(i / chunkSize) + 1}...`;

                const results = await tagImagesBatch(images, config);

                for (const result of results) {
                    const item = selectedItems.find(it => it.id === result.id);

                    if (result.error) {
                        log(`ERR ${result.id}: ${result.error}`);
                        errorCount++;
                        if (item) updateCurrentItemPreview(item, 'error', 'Error');
                    } else if (result.tags.length > 0) {
                        try {
                            if (item) {
                                updateCurrentItemPreview(item, 'processing', 'Saving...');
                                const existingTags = item.tags || [];
                                const newTags = result.tags.filter(t => !existingTags.includes(t));
                                if (newTags.length > 0) {
                                    item.tags = [...existingTags, ...newTags];
                                    await item.save();
                                    allAppliedTags.push(...newTags);
                                }
                                successCount++;
                                updateCurrentItemPreview(item, 'success', `+${result.tags.length} tags`);
                            }
                        } catch (e) {
                            log(`Save error: ${e.message}`);
                            errorCount++;
                            if (item) updateCurrentItemPreview(item, 'error', 'Save error');
                        }
                    } else {
                        skippedCount++;
                        if (item) updateCurrentItemPreview(item, 'skipped', 'No tags');
                    }
                }

                processed += chunk.length;
                updateProgress(processed, total, startTime);
                log(`${processed}/${total} done (${successCount} ok, ${errorCount} err, ${skippedCount} skip)`);
            }

            const elapsed = Date.now() - startTime;
            progressPanel.classList.remove('active');
            showSummary(successCount, errorCount, skippedCount, elapsed);

            if (successCount > 0) {
                showToast(`Tagged ${successCount} items`, 'success');
            } else if (errorCount > 0) {
                showToast(`${errorCount} errors occurred`, 'error');
            }

            log(`Finished in ${Math.round(elapsed / 1000)}s`);

        } catch (err) {
            log(`FATAL: ${err.message}\n${err.stack}`);
            showToast(`Error: ${err.message}`, 'error');
            progressPanel.classList.remove('active');
        } finally {
            startBtn.disabled = false;
            stopBtn.style.display = 'none';
            updateCurrentItemPreview(null, '', 'Done');
        }
    });

    stopBtn.addEventListener('click', () => {
        aborted = true;
        stopBtn.style.display = 'none';
        progressStatus.textContent = 'Stopping...';
        showToast('Stopping after current chunk...', 'info');
        log('Stop requested');
    });

    // Video conversion
    const { execFile } = require('child_process');
    const { promisify } = require('util');
    const execFileAsync = promisify(execFile);

    function findFFmpeg() {
        const candidates = [
            '/opt/homebrew/bin/ffmpeg',
            '/usr/local/bin/ffmpeg',
            '/usr/bin/ffmpeg',
            'ffmpeg'
        ];
        for (const p of candidates) {
            try {
                if (fs.existsSync(p)) return p;
            } catch {}
        }
        return 'ffmpeg';
    }

    let pendingDeleteItems = [];

    convertBtn.addEventListener('click', async () => {
        logArea.textContent = '';
        log('--- Start .ts conversion ---');

        let selectedItems;
        try {
            selectedItems = await eagle.item.getSelected();
        } catch (e) {
            showToast('Failed to get selected items', 'error');
            log(`ERROR: ${e.message}`);
            return;
        }

        // Filter to .ts files only
        const tsItems = selectedItems.filter(item =>
            item.ext && item.ext.toLowerCase() === 'ts'
        );

        if (tsItems.length === 0) {
            showToast('No .ts files in selection', 'info');
            log('No .ts files found');
            return;
        }

        log(`Found ${tsItems.length} .ts files to convert`);

        const ffmpegPath = findFFmpeg();
        log(`Using FFmpeg: ${ffmpegPath}`);

        // UI state
        convertBtn.disabled = true;
        startBtn.disabled = true;
        stopBtn.style.display = 'block';
        summaryPanel.classList.remove('active');
        confirmPanel.classList.remove('active');
        progressPanel.classList.add('active');
        progressFill.style.width = '0%';
        progressCount.textContent = `0 / ${tsItems.length}`;
        progressSpeed.textContent = '';
        progressStatus.textContent = 'Converting...';

        const startTime = Date.now();
        let successCount = 0;
        let errorCount = 0;
        pendingDeleteItems = [];
        aborted = false;

        for (let i = 0; i < tsItems.length; i++) {
            if (aborted) break;

            const item = tsItems[i];
            const inputPath = item.filePath;
            const outputPath = inputPath.replace(/\.ts$/i, '.mp4');

            log(`Converting: ${item.name}`);
            progressStatus.textContent = `Converting ${i + 1}/${tsItems.length}: ${item.name}`;

            try {
                // Stream copy conversion
                await execFileAsync(ffmpegPath, [
                    '-i', inputPath,
                    '-c', 'copy',
                    '-y',
                    outputPath
                ], { timeout: 300000 }); // 5 min timeout

                // Verify output exists
                if (!fs.existsSync(outputPath)) {
                    throw new Error('Output file not created');
                }

                log(`Created: ${path.basename(outputPath)}`);

                // Add to Eagle with same metadata
                try {
                    const folderIds = item.folders || [];
                    const addResult = await eagle.item.addFromPath(outputPath, {
                        name: item.name.replace(/\.ts$/i, ''),
                        website: item.website || '',
                        tags: item.tags || [],
                        annotation: item.annotation || '',
                        folders: folderIds,
                    });
                    log(`Added to Eagle: ${addResult ? 'success' : 'result unknown'}`);
                    pendingDeleteItems.push(item);
                    successCount++;
                } catch (addErr) {
                    log(`Eagle add error: ${addErr.message}`);
                    // File was converted but not added to Eagle
                    errorCount++;
                }

            } catch (e) {
                log(`Convert error ${item.name}: ${e.message}`);
                errorCount++;
            }

            updateProgress(i + 1, tsItems.length, startTime);
        }

        const elapsed = Date.now() - startTime;
        progressPanel.classList.remove('active');
        stopBtn.style.display = 'none';
        convertBtn.disabled = false;
        startBtn.disabled = false;

        log(`Conversion done: ${successCount} ok, ${errorCount} errors in ${Math.round(elapsed / 1000)}s`);

        if (successCount > 0) {
            showToast(`Converted ${successCount} files`, 'success');
            // Show confirm panel for deletion
            confirmList.innerHTML = pendingDeleteItems.map(item =>
                `<div>${item.name}.${item.ext}</div>`
            ).join('');
            confirmPanel.classList.add('active');
        } else if (errorCount > 0) {
            showToast(`${errorCount} conversion errors`, 'error');
        }
    });

    confirmDeleteBtn.addEventListener('click', async () => {
        if (pendingDeleteItems.length === 0) {
            confirmPanel.classList.remove('active');
            return;
        }

        log(`Deleting ${pendingDeleteItems.length} original .ts files...`);
        confirmDeleteBtn.disabled = true;

        let deleted = 0;
        for (const item of pendingDeleteItems) {
            try {
                // Try item.moveToTrash() first, fall back to item.remove()
                if (typeof item.moveToTrash === 'function') {
                    await item.moveToTrash();
                } else if (typeof item.remove === 'function') {
                    await item.remove();
                } else {
                    // Last resort: delete the file directly
                    fs.unlinkSync(item.filePath);
                }
                deleted++;
                log(`Removed: ${item.name}`);
            } catch (e) {
                log(`Delete error ${item.name}: ${e.message}`);
            }
        }

        confirmPanel.classList.remove('active');
        confirmDeleteBtn.disabled = false;
        pendingDeleteItems = [];
        showToast(`Removed ${deleted} originals`, 'success');
    });

    confirmKeepBtn.addEventListener('click', () => {
        confirmPanel.classList.remove('active');
        pendingDeleteItems = [];
        showToast('Originals kept', 'info');
        log('User chose to keep original .ts files');
    });

    // ============================================
    // PERSON RECOGNITION
    // ============================================

    let recognitionReady = false;

    // Load recognition config from localStorage
    function loadRecognitionConfig() {
        try {
            const saved = localStorage.getItem('eagleAiTaggerRecognition');
            if (saved) {
                const config = JSON.parse(saved);
                refFolderPath.value = config.refFolderPath || '';
                matchThreshold.value = config.matchThreshold || 0.65;
                thresholdValue.textContent = matchThreshold.value;

                // Load advanced settings
                if (config.mode) recognitionMode.value = config.mode;
                if (config.faceThreshold !== undefined) {
                    faceThreshold.value = config.faceThreshold;
                    faceThresholdValue.textContent = config.faceThreshold;
                }
                if (config.clipThreshold !== undefined) {
                    clipThreshold.value = config.clipThreshold;
                    clipThresholdValue.textContent = config.clipThreshold;
                }
                if (config.faceWeight !== undefined) {
                    faceWeight.value = config.faceWeight;
                    faceWeightValue.textContent = config.faceWeight + '%';
                    clipWeightValue.textContent = (100 - config.faceWeight) + '%';
                }
                if (config.videoFrames !== undefined) {
                    VIDEO_FRAMES = config.videoFrames;
                    if (videoFramesSlider) videoFramesSlider.value = VIDEO_FRAMES;
                    if (videoFramesValue) videoFramesValue.textContent = VIDEO_FRAMES;
                }
                if (config.skipTagged !== undefined) {
                    SKIP_TAGGED = config.skipTagged;
                    if (skipTaggedCheckbox) skipTaggedCheckbox.checked = SKIP_TAGGED;
                }
                if (config.replaceTags !== undefined) {
                    REPLACE_TAGS = config.replaceTags;
                    if (replaceTagsCheckbox) replaceTagsCheckbox.checked = REPLACE_TAGS;
                }
                if (config.ignoreTags !== undefined && Array.isArray(config.ignoreTags)) {
                    IGNORE_TAGS = config.ignoreTags;
                    renderIgnoreTags();
                }

                // Update RECOGNITION_CONFIG with saved values
                updateRecognitionConfigFromUI();
            }
            // Update UI visibility based on mode
            updateModeVisibility();
        } catch (e) {
            log(`Recognition config load error: ${e.message}`);
        }
    }

    // Render ignore tags (needs to be defined early for loadRecognitionConfig)
    function renderIgnoreTags() {
        if (!ignoreTagPills) return;
        ignoreTagPills.innerHTML = IGNORE_TAGS.map((tag, i) =>
            `<span class="tag-pill">${tag}<span class="remove" data-index="${i}">&times;</span></span>`
        ).join('');
    }

    function saveRecognitionConfig() {
        localStorage.setItem('eagleAiTaggerRecognition', JSON.stringify({
            refFolderPath: refFolderPath.value,
            matchThreshold: parseFloat(matchThreshold.value),
            mode: recognitionMode.value,
            faceThreshold: parseFloat(faceThreshold.value),
            clipThreshold: parseFloat(clipThreshold.value),
            faceWeight: parseInt(faceWeight.value),
            videoFrames: VIDEO_FRAMES,
            skipTagged: SKIP_TAGGED,
            replaceTags: REPLACE_TAGS,
            ignoreTags: IGNORE_TAGS,
        }));
        // Update RECOGNITION_CONFIG
        updateRecognitionConfigFromUI();
    }

    // Update RECOGNITION_CONFIG from UI values
    function updateRecognitionConfigFromUI() {
        const faceW = parseInt(faceWeight.value) / 100;
        RECOGNITION_CONFIG.thresholds.face = parseFloat(faceThreshold.value);
        RECOGNITION_CONFIG.thresholds.general = parseFloat(clipThreshold.value);
        RECOGNITION_CONFIG.thresholds.body = parseFloat(clipThreshold.value);
        RECOGNITION_CONFIG.thresholds.default = parseFloat(clipThreshold.value);
        RECOGNITION_CONFIG.weights.faceApi = faceW;
        RECOGNITION_CONFIG.weights.clip = 1 - faceW;
        RECOGNITION_CONFIG.mode = recognitionMode.value;
    }

    // Update UI visibility based on recognition mode
    function updateModeVisibility() {
        const mode = recognitionMode.value;
        // Show/hide relevant settings based on mode
        faceThresholdRow.style.display = (mode === 'face' || mode === 'ensemble') ? 'block' : 'none';
        clipThresholdRow.style.display = (mode === 'clip' || mode === 'ensemble') ? 'block' : 'none';
        ensembleWeightRow.style.display = (mode === 'ensemble') ? 'block' : 'none';
    }

    // Advanced toggle
    advancedToggle.addEventListener('click', () => {
        advancedToggle.classList.toggle('open');
        advancedPanel.classList.toggle('open');
    });

    // Recognition mode change
    recognitionMode.addEventListener('change', () => {
        updateModeVisibility();
        saveRecognitionConfig();
        log(`Recognition mode: ${recognitionMode.value}`);
    });

    // Threshold slider
    matchThreshold.addEventListener('input', () => {
        thresholdValue.textContent = matchThreshold.value;
        saveRecognitionConfig();
    });

    // Advanced sliders
    faceThreshold.addEventListener('input', () => {
        faceThresholdValue.textContent = faceThreshold.value;
        saveRecognitionConfig();
    });

    clipThreshold.addEventListener('input', () => {
        clipThresholdValue.textContent = clipThreshold.value;
        saveRecognitionConfig();
    });

    faceWeight.addEventListener('input', () => {
        faceWeightValue.textContent = faceWeight.value + '%';
        clipWeightValue.textContent = (100 - faceWeight.value) + '%';
        saveRecognitionConfig();
    });

    // Debug mode toggle
    if (debugModeCheckbox) {
        debugModeCheckbox.addEventListener('change', () => {
            DEBUG_MODE = debugModeCheckbox.checked;
            log(`[Debug] Debug mode ${DEBUG_MODE ? 'enabled' : 'disabled'}`);
            if (DEBUG_MODE) {
                log('[Debug] Will save aligned faces and comparison data to debug folder');
            }
        });
    }

    // Video frames slider
    if (videoFramesSlider) {
        videoFramesSlider.addEventListener('input', () => {
            VIDEO_FRAMES = parseInt(videoFramesSlider.value);
            videoFramesValue.textContent = VIDEO_FRAMES;
            saveRecognitionConfig();
            log(`[Video] Frames per video: ${VIDEO_FRAMES}`);
        });
    }

    // Skip tagged checkbox
    if (skipTaggedCheckbox) {
        skipTaggedCheckbox.addEventListener('change', () => {
            SKIP_TAGGED = skipTaggedCheckbox.checked;
            saveRecognitionConfig();
            log(`[Tags] Skip already tagged: ${SKIP_TAGGED ? 'enabled' : 'disabled'}`);
        });
    }

    // Replace tags checkbox
    if (replaceTagsCheckbox) {
        replaceTagsCheckbox.addEventListener('change', () => {
            REPLACE_TAGS = replaceTagsCheckbox.checked;
            saveRecognitionConfig();
            log(`[Tags] Replace existing tags: ${REPLACE_TAGS ? 'enabled' : 'disabled'}`);
            // If replace is enabled, skip tagged doesn't make sense
            if (REPLACE_TAGS && skipTaggedCheckbox) {
                skipTaggedCheckbox.checked = false;
                SKIP_TAGGED = false;
                log(`[Tags] Auto-disabled 'Skip Already Tagged' (incompatible with Replace)`);
            }
        });
    }

    // Ignore tags management
    if (addIgnoreTagBtn && ignoreTagInput) {
        addIgnoreTagBtn.addEventListener('click', () => {
            const tag = ignoreTagInput.value.trim().toLowerCase();
            if (tag && !IGNORE_TAGS.includes(tag)) {
                IGNORE_TAGS.push(tag);
                ignoreTagInput.value = '';
                renderIgnoreTags();
                saveRecognitionConfig();
                log(`[Tags] Added ignore tag: ${tag}`);
            }
        });

        ignoreTagInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') { e.preventDefault(); addIgnoreTagBtn.click(); }
        });
    }

    if (ignoreTagPills) {
        ignoreTagPills.addEventListener('click', (e) => {
            if (e.target.classList.contains('remove')) {
                const idx = parseInt(e.target.dataset.index);
                const removed = IGNORE_TAGS.splice(idx, 1)[0];
                renderIgnoreTags();
                saveRecognitionConfig();
                log(`[Tags] Removed ignore tag: ${removed}`);
            }
        });
    }

    // Reference folder path input
    refFolderPath.addEventListener('change', () => {
        saveRecognitionConfig();
        log(`Reference folder set: ${refFolderPath.value}`);
    });

    // Clear cache button - deletes cache file and clears in-memory references
    clearCacheBtn.addEventListener('click', async () => {
        const folderName = refFolderPath.value.trim();
        if (!folderName) {
            showToast('Please enter an Eagle folder name first', 'error');
            return;
        }

        // Confirm before clearing
        if (!confirm(`Clear cached embeddings for "${folderName}"?\n\nThis will delete the cache file and require a full rebuild on next load.`)) {
            return;
        }

        log(`[Cache] Clearing cache for "${folderName}"...`);

        // Clear in-memory references
        clearReferences();

        // Delete the cache file
        deleteReferenceCache(folderName);

        // Update UI
        refStatus.textContent = 'Cache cleared - click Load References to rebuild';
        refStatus.className = 'ref-status';
        identifyBtn.disabled = true;

        showToast('Cache cleared successfully', 'success');
        log('[Cache] Cache cleared. Click "Load References" to rebuild embeddings.');
    });

    // Load references button - uses Eagle folder
    // Supports two modes:
    // 1. Subfolder mode: Each subfolder is a person name, containing multiple reference images
    // 2. Flat mode (legacy): Images directly in folder with names like "PersonName_feature.jpg"
    loadRefsBtn.addEventListener('click', async () => {
        const folderName = refFolderPath.value.trim();
        if (!folderName) {
            showToast('Please enter an Eagle folder name', 'error');
            return;
        }

        loadRefsBtn.disabled = true;
        identifyBtn.disabled = true;
        refProgress.classList.add('active');
        refProgressFill.style.width = '0%';
        refProgressStatus.textContent = 'Finding Eagle folder...';
        refStatus.textContent = 'Loading...';
        refStatus.className = 'ref-status';

        try {
            // Find the Eagle folder by name
            let targetFolder = null;
            let allFolders = [];
            try {
                allFolders = await eagle.folder.getAll();
                // Search for folder by name (case-insensitive, supports nested like "People/References")
                const searchName = folderName.toLowerCase();
                targetFolder = allFolders.find(f => {
                    const fName = (f.name || '').toLowerCase();
                    return fName === searchName || fName.endsWith('/' + searchName);
                });
            } catch (e) {
                log(`Folder API error: ${e.message}`);
                throw new Error(`Cannot access Eagle folders: ${e.message}`);
            }

            if (!targetFolder) {
                throw new Error(`Folder "${folderName}" not found in Eagle`);
            }

            log(`Found Eagle folder: ${targetFolder.name} (${targetFolder.id})`);
            log(`Target folder keys: ${Object.keys(targetFolder).join(', ')}`);

            // Check for child folders (subfolder mode)
            // Eagle folders may use 'parent', 'parentId', or 'children' property
            let childFolders = [];

            // Method 1: Check if target folder has children array
            if (targetFolder.children && Array.isArray(targetFolder.children)) {
                childFolders = targetFolder.children;
                log(`Found ${childFolders.length} children via folder.children`);
            }

            // Method 2: Filter by parent/parentId
            if (childFolders.length === 0) {
                childFolders = allFolders.filter(f =>
                    f.parent === targetFolder.id ||
                    f.parentId === targetFolder.id
                );
                if (childFolders.length > 0) {
                    log(`Found ${childFolders.length} children via parent filter`);
                }
            }

            // Method 3: Check folder structure in allFolders
            if (childFolders.length === 0 && allFolders.length > 0) {
                const sampleFolder = allFolders[0];
                log(`Sample folder keys: ${Object.keys(sampleFolder).join(', ')}`);
                if (sampleFolder.parent !== undefined) {
                    log(`Sample folder parent value: ${sampleFolder.parent}`);
                }
            }

            const useSubfolderMode = childFolders.length > 0;

            if (useSubfolderMode) {
                log(`Subfolder mode: found ${childFolders.length} person folders`);
            } else {
                log(`Flat mode: processing images directly in folder`);
            }

            refProgressStatus.textContent = 'Loading recognition models...';

            // Initialize CLIP
            if (!isClipReady()) {
                refProgressStatus.textContent = 'Loading CLIP model (first time may take a while)...';
                const initResult = await initClip();
                if (!initResult) {
                    throw new Error('CLIP model failed to initialize');
                }
            }

            // Initialize Face-API (non-blocking - continue even if it fails)
            let faceApiAvailable = false;
            if (!isFaceApiReady()) {
                refProgressStatus.textContent = 'Loading Face-API model...';
                try {
                    faceApiAvailable = await initFaceApi();
                } catch (e) {
                    log('[FaceAPI] Skipping face detection: ' + e.message);
                }
            } else {
                faceApiAvailable = true;
            }

            if (faceApiAvailable) {
                log('Enhanced mode: CLIP + Face-API');
            } else {
                log('Basic mode: CLIP only (face detection unavailable)');
            }

            // ============================================
            // TRY LOADING FROM CACHE FIRST
            // ============================================
            refProgressStatus.textContent = 'Checking cache...';

            // Gather all folder items first for cache validation
            let allFolderItems = [];
            const itemPersonMap = new Map(); // item.id -> personName
            if (useSubfolderMode) {
                for (const personFolder of childFolders) {
                    try {
                        const items = await eagle.item.get({ folders: [personFolder.id] });
                        if (items) {
                            for (const item of items) {
                                allFolderItems.push(item);
                                itemPersonMap.set(item.id, personFolder.name);
                            }
                        }
                    } catch (e) {}
                }
            } else {
                try {
                    const items = await eagle.item.get({ folders: [targetFolder.id] });
                    if (items) allFolderItems = items;
                } catch (e) {}
            }

            // Try to load and validate cache
            const cacheData = loadReferenceCache(folderName);
            let cacheDiff = null;
            let itemsToProcess = []; // Items that need embedding computation

            if (cacheData) {
                refProgressStatus.textContent = 'Analyzing cache...';
                cacheDiff = await analyzeCacheDiff(cacheData, allFolderItems, itemPersonMap);

                if (cacheDiff.fullyValid) {
                    // FULL CACHE HIT - restore everything from cache
                    restoreReferencesFromCache(cacheData);

                    // Count people and refs from cache
                    const people = new Set();
                    const refCounts = {};
                    for (const ref of referenceEmbeddings.values()) {
                        people.add(ref.person);
                        refCounts[ref.person] = (refCounts[ref.person] || 0) + 1;
                    }

                    const peopleArr = Array.from(people);
                    const refSummary = peopleArr.map(p => `${p}(${refCounts[p] || 0})`).join(', ');
                    const cacheAge = Math.round((Date.now() - cacheData.timestamp) / 1000 / 60);

                    refStatus.textContent = `Loaded ${referenceEmbeddings.size} refs for ${peopleArr.length} people (from cache, ${cacheAge}m old)`;
                    refStatus.className = 'ref-status loaded';
                    recognitionReady = referenceEmbeddings.size > 0;
                    identifyBtn.disabled = !recognitionReady;

                    showToast(`Loaded ${referenceEmbeddings.size} references from cache`, 'success');
                    log(`Cache hit: ${referenceEmbeddings.size} references loaded instantly`);
                    log(`Per-person counts: ${refSummary}`);

                    loadRefsBtn.disabled = false;
                    refProgress.classList.remove('active');
                    return; // Exit early - cache was valid
                } else {
                    // PARTIAL CACHE HIT - restore unchanged refs, compute only new/modified
                    log(`[Cache] Incremental update: ${cacheDiff.unchanged.length} cached, ${cacheDiff.added.length + cacheDiff.modified.length} to compute`);
                    restorePartialCache(cacheDiff.unchanged);

                    // Build list of items to process (added + modified)
                    for (const addedItem of cacheDiff.added) {
                        itemsToProcess.push({
                            item: addedItem.item,
                            personName: addedItem.personName,
                            filePath: addedItem.filePath
                        });
                    }
                    for (const modifiedItem of cacheDiff.modified) {
                        itemsToProcess.push({
                            item: modifiedItem.item,
                            personName: modifiedItem.personName,
                            filePath: modifiedItem.item.filePath || modifiedItem.item.thumbnailPath
                        });
                    }
                }
            } else {
                // No cache - process all items
                clearReferences();
                if (useSubfolderMode) {
                    for (const item of allFolderItems) {
                        itemsToProcess.push({
                            item,
                            personName: itemPersonMap.get(item.id),
                            filePath: item.filePath || item.thumbnailPath
                        });
                    }
                } else {
                    for (const item of allFolderItems) {
                        const parsed = parseReferenceName(item.name);
                        itemsToProcess.push({
                            item,
                            personName: parsed.person,
                            feature: parsed.feature,
                            filePath: item.filePath || item.thumbnailPath
                        });
                    }
                }
            }

            // ============================================
            // COMPUTE EMBEDDINGS FOR NEW/MODIFIED ITEMS
            // ============================================

            const errors = [];
            const people = new Set();
            const refCounts = {}; // Track refs per person
            let newlyLoaded = 0;
            let totalToProcess = itemsToProcess.length;
            let processedItems = 0;

            // Count existing refs from cache
            for (const ref of referenceEmbeddings.values()) {
                people.add(ref.person);
                refCounts[ref.person] = (refCounts[ref.person] || 0) + 1;
            }
            const cachedCount = referenceEmbeddings.size;

            if (totalToProcess > 0) {
                log(`Processing ${totalToProcess} new/modified reference images...`);
            }

            for (const itemData of itemsToProcess) {
                const { item, personName, filePath } = itemData;
                processedItems++;
                const pct = Math.round((processedItems / totalToProcess) * 100);
                refProgressFill.style.width = `${pct}%`;
                refProgressStatus.textContent = `Processing ${personName}: ${item.name} (${processedItems}/${totalToProcess})`;

                try {
                    if (!filePath) {
                        errors.push(`No file path: ${personName}/${item.name}`);
                        continue;
                    }

                    // Extract CLIP embedding
                    const clipEmbedding = await extractClipEmbedding(filePath);

                    // Extract face descriptors (if Face-API available)
                    let faceDescriptors = [];
                    if (faceApiAvailable) {
                        const faceDetections = await detectFacesAndExtractDescriptors(filePath);
                        faceDescriptors = faceDetections.map(d => d.descriptor);
                    }

                    if (clipEmbedding || faceDescriptors.length > 0) {
                        // Use item name as feature (or "ref_N" if no underscore)
                        const feature = itemData.feature || (item.name.includes('_')
                            ? item.name.split('_').slice(1).join('_').replace(/\.[^.]+$/, '')
                            : `ref_${(refCounts[personName] || 0) + 1}`);

                        addReference(personName, feature, clipEmbedding, faceDescriptors, filePath);
                        people.add(personName);
                        refCounts[personName] = (refCounts[personName] || 0) + 1;
                        newlyLoaded++;

                        const faceInfo = faceDescriptors.length > 0 ? ` [${faceDescriptors.length} face(s)]` : '';
                        log(`  Loaded: ${item.name} -> ${personName} (${feature})${faceInfo}`);
                    } else {
                        errors.push(`Failed to extract embeddings: ${personName}/${item.name}`);
                    }
                } catch (e) {
                    errors.push(`Error processing ${personName}/${item.name}: ${e.message}`);
                }
            }

            if (errors.length > 0) {
                errors.slice(0, 10).forEach(err => log(`Reference error: ${err}`));
                if (errors.length > 10) {
                    log(`... and ${errors.length - 10} more errors`);
                }
            }

            const totalLoaded = referenceEmbeddings.size;
            const peopleArr = Array.from(people);
            const refSummary = peopleArr.map(p => `${p}(${refCounts[p] || 0})`).join(', ');

            // Show appropriate status based on whether we used cache
            if (cachedCount > 0 && newlyLoaded > 0) {
                refStatus.textContent = `Loaded ${totalLoaded} refs (${cachedCount} cached + ${newlyLoaded} new) for ${peopleArr.length} people`;
                showToast(`Updated: ${newlyLoaded} new refs added (${cachedCount} from cache)`, 'success');
                log(`Incremental update: ${cachedCount} cached + ${newlyLoaded} new = ${totalLoaded} total`);
            } else if (cachedCount > 0) {
                refStatus.textContent = `Loaded ${totalLoaded} refs for ${peopleArr.length} people (from cache)`;
                showToast(`Loaded ${totalLoaded} references from cache`, 'success');
            } else {
                refStatus.textContent = `Loaded ${totalLoaded} refs for ${peopleArr.length} people: ${refSummary}`;
                if (totalLoaded > 0) {
                    showToast(`Loaded ${totalLoaded} references for ${peopleArr.length} people`, 'success');
                } else {
                    showToast('No valid references found', 'error');
                }
            }

            refStatus.className = 'ref-status loaded';
            recognitionReady = totalLoaded > 0;
            identifyBtn.disabled = !recognitionReady;

            log(`References loaded: ${totalLoaded} for ${peopleArr.length} people`);
            log(`Per-person counts: ${refSummary}`);

            // Save to cache for next time
            if (totalLoaded > 0) {
                saveReferenceCache(folderName);
            }

        } catch (e) {
            refStatus.textContent = `Error: ${e.message}`;
            refStatus.className = 'ref-status error';
            showToast(`Failed: ${e.message}`, 'error');
            log(`Load references error: ${e.message}`);
        } finally {
            loadRefsBtn.disabled = false;
            refProgress.classList.remove('active');
        }
    });

    // Identify people button - Enhanced with CLIP + Face-API ensemble
    identifyBtn.addEventListener('click', async () => {
        if (!recognitionReady) {
            showToast('Please load references first', 'error');
            return;
        }

        let selectedItems;
        try {
            selectedItems = await eagle.item.getSelected();
        } catch (e) {
            showToast('Failed to get selected items', 'error');
            log(`ERROR: ${e.message}`);
            return;
        }

        if (!selectedItems || selectedItems.length === 0) {
            showToast('No items selected in Eagle', 'error');
            return;
        }

        const threshold = parseFloat(matchThreshold.value);
        const faceApiAvailable = isFaceApiReady();
        const mode = recognitionMode.value;
        const modeLabels = {
            'ensemble': 'Ensemble (CLIP + Face)',
            'clip': 'CLIP Only',
            'face': 'Face Only'
        };
        log(`--- Start person identification ---`);
        log(`Mode: ${modeLabels[mode] || mode}${mode !== 'clip' && !faceApiAvailable ? ' (Face-API unavailable, using CLIP)' : ''}`);
        log(`Threshold: ${threshold}`);
        if (IGNORE_TAGS.length > 0) {
            log(`Ignoring items with tags: ${IGNORE_TAGS.join(', ')}`);
        }

        identifyBtn.disabled = true;
        startBtn.disabled = true;
        convertBtn.disabled = true;
        stopBtn.style.display = 'block';
        summaryPanel.classList.remove('active');
        progressPanel.classList.add('active');
        progressFill.style.width = '0%';
        progressCount.textContent = `0 / ${selectedItems.length}`;
        progressSpeed.textContent = '';
        progressStatus.textContent = 'Identifying...';

        const startTime = Date.now();
        let processed = 0;
        let identified = 0;
        let tagsApplied = 0;
        let skippedAlreadyTagged = 0;
        let skippedIgnoreTags = 0;
        const personCounts = {};
        const matchTypes = { ensemble: 0, face: 0, clip: 0 };
        aborted = false;

        // Initialize debug folder if debug mode is enabled
        if (DEBUG_MODE) {
            await initDebugFolder();
            log(`[Debug] Debug output will be saved to: ${debugFolder}`);
        }

        // OPTIMIZATION: Parallel processing with concurrency limit
        const CONCURRENCY = 3; // Process 3 items in parallel
        let cacheHits = 0;

        // Helper function to check if item has any person.* tags
        function hasPersonTag(item) {
            const tags = item.tags || [];
            return tags.some(tag => tag.startsWith('person.'));
        }

        // Helper function to get existing person.* tags
        function getPersonTags(item) {
            const tags = item.tags || [];
            return tags.filter(tag => tag.startsWith('person.'));
        }

        // Helper function to check if item has any tags that should be ignored
        function hasIgnoredTag(item) {
            if (IGNORE_TAGS.length === 0) return false;
            const tags = (item.tags || []).map(t => t.toLowerCase());
            return IGNORE_TAGS.some(ignoreTag => tags.includes(ignoreTag.toLowerCase()));
        }

        // Helper function to get the matching ignored tags
        function getMatchingIgnoreTags(item) {
            if (IGNORE_TAGS.length === 0) return [];
            const tags = (item.tags || []).map(t => t.toLowerCase());
            return IGNORE_TAGS.filter(ignoreTag => tags.includes(ignoreTag.toLowerCase()));
        }

        // Process a single item and return result
        async function processItem(item) {
            // Update preview to show current item being processed
            updateCurrentItemPreview(item, 'processing', 'Processing...');

            const result = {
                item,
                matches: [],
                skipped: false,
                error: null,
                cacheHit: false
            };

            try {
                // Skip items that already have person tags (if enabled)
                if (SKIP_TAGGED && hasPersonTag(item)) {
                    const existingPersonTags = getPersonTags(item);
                    log(`${item.name}: Skipped (already tagged: ${existingPersonTags.join(', ')})`);
                    result.skipped = true;
                    result.skippedReason = 'already_tagged';
                    return result;
                }

                // Skip items that have any of the ignored tags
                if (hasIgnoredTag(item)) {
                    const matchingTags = getMatchingIgnoreTags(item);
                    log(`${item.name}: Skipped (has ignored tag: ${matchingTags.join(', ')})`);
                    result.skipped = true;
                    result.skippedReason = 'ignored_tag';
                    return result;
                }

                const ext = item.ext ? item.ext.toLowerCase() : '';
                const isVideo = isVideoExtension(ext);

                let clipEmbedding = null;
                let queryFaceDescriptors = [];

                // OPTIMIZATION: Check cache first (images only, videos need fresh frame extraction)
                // Skip cache when debug mode is enabled to get fresh aligned faces
                if (!isVideo && !DEBUG_MODE) {
                    const cached = getCachedEmbedding(item.id);
                    if (cached) {
                        clipEmbedding = cached.clipEmbedding;
                        queryFaceDescriptors = cached.faceDescriptors || [];
                        result.cacheHit = true;
                        log(`${item.name}: Using cached embeddings`);
                    }
                }

                // Only compute if not cached
                if (!result.cacheHit) {
                    if (isVideo && mode !== 'clip' && faceApiAvailable) {
                        // SMART VIDEO HANDLING: Find best face frame
                        let bestFrame = null;
                        try {
                            bestFrame = await findBestFaceFrame(item.filePath, VIDEO_FRAMES, {
                                returnAlignedFace: DEBUG_MODE
                            });
                        } catch (videoErr) {
                            log(`${item.name}: Video scan failed - ${videoErr.message}`);
                        }

                        if (bestFrame) {
                            queryFaceDescriptors = bestFrame.detections.map(d => Array.from(d.descriptor));

                            // Save aligned faces for debugging (video)
                            if (DEBUG_MODE && bestFrame.detections) {
                                const faceDetections = [];
                                for (let i = 0; i < bestFrame.detections.length; i++) {
                                    const det = bestFrame.detections[i];
                                    const faceData = {
                                        descriptor: det.descriptor,
                                        bbox: det.bbox,
                                        confidence: det.confidence,
                                    };
                                    if (det.alignedFace) {
                                        const filename = `query_${item.id}_video_face${i}_aligned.jpg`;
                                        faceData.alignedFaceFile = await saveDebugImage(det.alignedFace, filename);
                                    }
                                    faceDetections.push(faceData);
                                }
                                result.faceDetections = faceDetections;
                            }
                            if (mode !== 'face') {
                                clipEmbedding = await extractClipEmbeddingFromBuffer(bestFrame.buffer);
                            }
                        } else if (item.thumbnailPath && mode !== 'face') {
                            clipEmbedding = await extractClipEmbedding(item.thumbnailPath);
                            log(`${item.name}: No faces, using thumbnail for CLIP`);
                        } else if (mode === 'face') {
                            log(`${item.name}: No faces found, skipping (face-only mode)`);
                            result.skipped = true;
                            return result;
                        } else {
                            log(`${item.name}: No thumbnail available, skipping`);
                            result.skipped = true;
                            return result;
                        }
                    } else if (isVideo) {
                        if (item.thumbnailPath) {
                            clipEmbedding = await extractClipEmbedding(item.thumbnailPath);
                        } else {
                            log(`${item.name}: Video has no thumbnail, skipping`);
                            result.skipped = true;
                            return result;
                        }
                    } else {
                        // IMAGE: Standard processing
                        const imagePath = item.filePath;

                        if (mode !== 'face') {
                            clipEmbedding = await extractClipEmbedding(imagePath);
                        }

                        if (faceApiAvailable && mode !== 'clip') {
                            // Pass debug prefix if debug mode is enabled
                            const debugPrefix = DEBUG_MODE ? `query_${item.id}` : null;
                            const faceDetections = await detectFacesAndExtractDescriptors(imagePath, debugPrefix);
                            queryFaceDescriptors = faceDetections.map(d => d.descriptor);

                            // Store full face detection data for debug report
                            if (DEBUG_MODE) {
                                result.faceDetections = faceDetections;
                            }

                            if (mode === 'face' && queryFaceDescriptors.length === 0) {
                                log(`${item.name}: No faces found, skipping (face-only mode)`);
                                result.skipped = true;
                                return result;
                            }
                            // OPTIMIZATION: In face-only mode, we don't need CLIP at all
                            // since findMatchesEnhanced ignores CLIP scores in face-only mode
                        }

                        // Cache the computed embeddings for images
                        setCachedEmbedding(item.id, clipEmbedding, queryFaceDescriptors);
                    }
                }

                // Find matches using enhanced ensemble scoring
                result.matches = findMatchesEnhanced(clipEmbedding, queryFaceDescriptors, threshold);

            } catch (e) {
                result.error = e.message;
                log(`Error processing ${item.name}: ${e.message}`);
            }

            return result;
        }

        try {
            // Process items in parallel batches
            for (let i = 0; i < selectedItems.length && !aborted; i += CONCURRENCY) {
                const batch = selectedItems.slice(i, Math.min(i + CONCURRENCY, selectedItems.length));
                progressStatus.textContent = `Processing batch ${Math.floor(i / CONCURRENCY) + 1}...`;

                // Process batch in parallel
                const results = await Promise.all(batch.map(item => processItem(item)));

                // Process results
                for (const result of results) {
                    if (aborted) break;

                    if (result.cacheHit) cacheHits++;
                    if (result.skippedReason === 'already_tagged') skippedAlreadyTagged++;
                    if (result.skippedReason === 'ignored_tag') skippedIgnoreTags++;

                    // Update preview with result status
                    if (result.skipped) {
                        updateCurrentItemPreview(result.item, 'skipped', 'Skipped');
                    } else if (result.error) {
                        updateCurrentItemPreview(result.item, 'error', 'Error');
                    } else if (result.matches.length > 0) {
                        const topMatch = result.matches[0];
                        updateCurrentItemPreview(result.item, 'success', `${topMatch.person}`);
                    } else {
                        updateCurrentItemPreview(result.item, '', 'No match');
                    }

                    // Generate debug report if debug mode is enabled
                    if (DEBUG_MODE && result.faceDetections) {
                        try {
                            await generateDebugReport(
                                result.item,
                                result.faceDetections,
                                result.matches,
                                referenceEmbeddings
                            );
                        } catch (debugErr) {
                            log(`[Debug] Failed to generate report: ${debugErr.message}`);
                        }
                    }

                    if (!result.skipped && !result.error && result.matches.length > 0) {
                        const item = result.item;
                        // ONLY use the highest scoring match (first one, already sorted by confidence)
                        const topMatch = result.matches[0];
                        const matches = [topMatch];

                        // Generate and apply tags (single tag only)
                        const newPersonTag = `person.${topMatch.person}`;
                        let existingTags = item.tags || [];
                        const existingPersonTags = existingTags.filter(t => t.startsWith('person.'));

                        // Check if we need to do anything
                        let needsUpdate = false;
                        let replacedTags = [];

                        if (REPLACE_TAGS && existingPersonTags.length > 0) {
                            // Remove old person tags, add new one
                            if (!existingPersonTags.includes(newPersonTag) || existingPersonTags.length > 1) {
                                replacedTags = existingPersonTags.filter(t => t !== newPersonTag);
                                existingTags = existingTags.filter(t => !t.startsWith('person.'));
                                existingTags.push(newPersonTag);
                                needsUpdate = true;
                            }
                        } else if (!existingTags.includes(newPersonTag)) {
                            // Just add the new tag
                            existingTags.push(newPersonTag);
                            needsUpdate = true;
                        }

                        if (needsUpdate) {
                            item.tags = existingTags;
                            await item.save();
                            tagsApplied++;

                            for (const match of matches) {
                                personCounts[match.person] = (personCounts[match.person] || 0) + 1;
                                matchTypes[match.matchType] = (matchTypes[match.matchType] || 0) + 1;
                            }

                            if (replacedTags.length > 0) {
                                log(`${item.name}: Replaced ${replacedTags.join(', ')} -> ${newPersonTag}`);
                            }
                        }

                        identified++;

                        // Enhanced logging with match details
                        const matchDetails = matches.map(m => {
                            let detail = `${m.person}:${m.confidence.toFixed(2)}`;
                            if (m.matchType === 'ensemble') {
                                detail += `(F:${m.faceScore.toFixed(2)}+C:${m.clipScore.toFixed(2)})`;
                            } else if (m.matchType === 'face') {
                                detail += '(face)';
                            }
                            if (m.matchingRefs > 1) {
                                detail += `[${m.matchingRefs}refs]`;
                            }
                            return detail;
                        }).join(', ');
                        log(`${item.name}: ${matchDetails}`);
                    }

                    processed++;
                }

                updateProgress(processed, selectedItems.length, startTime);
            }

            const elapsed = Date.now() - startTime;
            progressPanel.classList.remove('active');

            // Show summary
            const personSummary = Object.entries(personCounts)
                .sort((a, b) => b[1] - a[1])
                .map(([person, count]) => `${person}: ${count}`)
                .join(', ');

            log(`Identification complete: ${identified}/${processed} images, ${tagsApplied} tags applied`);
            if (skippedAlreadyTagged > 0) {
                log(`Skipped (already tagged): ${skippedAlreadyTagged}`);
            }
            if (skippedIgnoreTags > 0) {
                log(`Skipped (ignored tags): ${skippedIgnoreTags}`);
            }
            log(`Per-person: ${personSummary || 'none'}`);
            log(`Match types: ensemble=${matchTypes.ensemble}, face=${matchTypes.face}, clip=${matchTypes.clip}`);
            if (cacheHits > 0) {
                log(`Cache hits: ${cacheHits} (${Math.round(cacheHits / processed * 100)}%)`);
            }

            if (identified > 0) {
                showToast(`Identified people in ${identified} images`, 'success');
            } else {
                showToast('No matches found', 'info');
            }

        } catch (e) {
            log(`FATAL: ${e.message}`);
            showToast(`Error: ${e.message}`, 'error');
            progressPanel.classList.remove('active');
        } finally {
            identifyBtn.disabled = false;
            startBtn.disabled = false;
            convertBtn.disabled = false;
            stopBtn.style.display = 'none';
            updateCurrentItemPreview(null, '', 'Done');
        }
    });

    // Load recognition config on startup
    loadRecognitionConfig();

    // Init
    loadConfig();
    renderPresetDropdown('');
    log('UI initialized');
}

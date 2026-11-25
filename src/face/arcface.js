/**
 * ArcFace Face Recognition Embedding Extractor
 * Model: arcfaceresnet100-8.onnx from ONNX Model Zoo
 *
 * Extracts 512-dimensional face embeddings from aligned 112x112 face images.
 * These embeddings can be compared using cosine similarity.
 */

const path = require('path');
const sharp = require('sharp');

let ort = null;
let arcfaceSession = null;

// ArcFace configuration
const ARCFACE_CONFIG = {
    inputSize: 112,        // Required input size
    embeddingDim: 512,     // Output embedding dimension
};

// NOTE: This ONNX model expects raw pixel values [0-255] as float32
// NOT the standard (pixel - 127.5) / 128 normalization!

/**
 * Initialize ArcFace model
 * @param {string} modelPath - Path to arcfaceresnet100-8.onnx
 * @returns {Promise<boolean>}
 */
async function initArcFace(modelPath) {
    if (arcfaceSession) return true;

    try {
        // Load ONNX Runtime
        if (!ort) {
            ort = require('onnxruntime-node');
        }

        console.log('[ArcFace] Loading model from:', modelPath);
        arcfaceSession = await ort.InferenceSession.create(modelPath, {
            executionProviders: ['cpu'],
            graphOptimizationLevel: 'all',
        });

        // Log model info
        console.log('[ArcFace] Input names:', arcfaceSession.inputNames);
        console.log('[ArcFace] Output names:', arcfaceSession.outputNames);
        console.log('[ArcFace] Model loaded successfully');

        return true;
    } catch (e) {
        console.error('[ArcFace] Failed to load model:', e.message);
        arcfaceSession = null;
        return false;
    }
}

/**
 * Check if ArcFace is ready
 */
function isArcFaceReady() {
    return arcfaceSession !== null;
}

/**
 * Preprocess aligned face for ArcFace
 * @param {Buffer|Uint8Array} imageData - RGB image data (112x112x3)
 * @param {number} width - Image width
 * @param {number} height - Image height
 * @returns {Float32Array} - Preprocessed tensor in NCHW format
 */
function preprocessFace(imageData, width, height) {
    const inputSize = ARCFACE_CONFIG.inputSize;

    if (width !== inputSize || height !== inputSize) {
        throw new Error(`ArcFace requires ${inputSize}x${inputSize} input, got ${width}x${height}`);
    }

    const pixelCount = inputSize * inputSize;
    const floatData = new Float32Array(3 * pixelCount);

    // Convert to NCHW format - NO normalization
    // This ONNX model expects raw pixel values [0-255] as float32
    for (let i = 0; i < pixelCount; i++) {
        const r = imageData[i * 3];
        const g = imageData[i * 3 + 1];
        const b = imageData[i * 3 + 2];

        // NCHW: [batch, channel, height, width]
        floatData[i] = r;                    // R channel
        floatData[pixelCount + i] = g;       // G channel
        floatData[2 * pixelCount + i] = b;   // B channel
    }

    return floatData;
}

/**
 * L2 normalize embedding
 * @param {Float32Array} embedding - Raw embedding
 * @returns {Float32Array} - L2 normalized embedding
 */
function l2Normalize(embedding) {
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
}

/**
 * Extract face embedding from raw RGB data
 * @param {Uint8Array} rgbData - Raw RGB pixel data (112x112x3)
 * @param {number} width - Image width (must be 112)
 * @param {number} height - Image height (must be 112)
 * @returns {Promise<Float32Array>} - 512-dim L2-normalized embedding
 */
async function extractEmbeddingFromRaw(rgbData, width, height) {
    if (!arcfaceSession) {
        throw new Error('ArcFace not initialized');
    }

    // Preprocess
    const floatData = preprocessFace(rgbData, width, height);

    // Create input tensor
    const inputTensor = new ort.Tensor('float32', floatData, [1, 3, ARCFACE_CONFIG.inputSize, ARCFACE_CONFIG.inputSize]);

    // Run inference
    const inputName = arcfaceSession.inputNames[0];
    const results = await arcfaceSession.run({ [inputName]: inputTensor });

    // Get output embedding
    const outputName = arcfaceSession.outputNames[0];
    const embedding = results[outputName].data;

    // L2 normalize
    return l2Normalize(new Float32Array(embedding));
}

/**
 * Extract face embedding from aligned face image buffer
 * @param {Buffer} alignedFaceBuffer - JPEG/PNG buffer of aligned 112x112 face
 * @returns {Promise<Float32Array>} - 512-dim L2-normalized embedding
 */
async function extractEmbedding(alignedFaceBuffer) {
    if (!arcfaceSession) {
        throw new Error('ArcFace not initialized');
    }

    // Ensure image is 112x112 RGB
    const { data, info } = await sharp(alignedFaceBuffer)
        .resize(ARCFACE_CONFIG.inputSize, ARCFACE_CONFIG.inputSize)
        .removeAlpha()
        .raw()
        .toBuffer({ resolveWithObject: true });

    return extractEmbeddingFromRaw(new Uint8Array(data), info.width, info.height);
}

/**
 * Compute cosine similarity between two embeddings
 * @param {Float32Array} emb1 - First embedding (L2 normalized)
 * @param {Float32Array} emb2 - Second embedding (L2 normalized)
 * @returns {number} - Cosine similarity (-1 to 1, higher is more similar)
 */
function cosineSimilarity(emb1, emb2) {
    if (emb1.length !== emb2.length) {
        throw new Error('Embedding dimensions must match');
    }

    let dot = 0;
    for (let i = 0; i < emb1.length; i++) {
        dot += emb1[i] * emb2[i];
    }

    // Since embeddings are L2 normalized, dot product = cosine similarity
    return dot;
}

/**
 * Compute Euclidean distance between two embeddings
 * @param {Float32Array} emb1 - First embedding
 * @param {Float32Array} emb2 - Second embedding
 * @returns {number} - Euclidean distance (0 = identical)
 */
function euclideanDistance(emb1, emb2) {
    if (emb1.length !== emb2.length) {
        throw new Error('Embedding dimensions must match');
    }

    let sum = 0;
    for (let i = 0; i < emb1.length; i++) {
        const diff = emb1[i] - emb2[i];
        sum += diff * diff;
    }
    return Math.sqrt(sum);
}

/**
 * Cleanup ArcFace resources
 */
async function closeArcFace() {
    if (arcfaceSession) {
        await arcfaceSession.release();
        arcfaceSession = null;
    }
}

module.exports = {
    initArcFace,
    isArcFaceReady,
    extractEmbedding,
    extractEmbeddingFromRaw,
    cosineSimilarity,
    euclideanDistance,
    l2Normalize,
    closeArcFace,
    ARCFACE_CONFIG,
};

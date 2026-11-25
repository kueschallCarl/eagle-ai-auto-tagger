/**
 * Face Recognition Pipeline
 * Combines SCRFD detection + alignment + ArcFace embedding extraction
 *
 * This is the main interface for the face recognition system.
 */

const path = require('path');
const fs = require('fs');
const sharp = require('sharp');

const { initSCRFD, isSCRFDReady, detectFaces, closeSCRFD, SCRFD_CONFIG } = require('./scrfd');
const { alignFace, alignFaceRaw, ARCFACE_SIZE } = require('./alignment');
const { initArcFace, isArcFaceReady, extractEmbedding, extractEmbeddingFromRaw, cosineSimilarity, closeArcFace, ARCFACE_CONFIG } = require('./arcface');

// Pipeline state
let pipelineReady = false;
let modelsPath = '';

// Configuration
const PIPELINE_CONFIG = {
    // Detection settings
    detection: {
        confThreshold: 0.5,    // Face detection confidence threshold
        nmsThreshold: 0.4,     // NMS IoU threshold
        minFaceSize: 30,       // Minimum face size in pixels
    },
    // Matching settings
    matching: {
        threshold: 0.5,        // Cosine similarity threshold for matching
    },
};

/**
 * Initialize the face recognition pipeline
 * @param {string} modelDir - Directory containing det_10g.onnx and arcfaceresnet100-8.onnx
 * @returns {Promise<boolean>}
 */
async function initPipeline(modelDir) {
    if (pipelineReady) return true;

    modelsPath = modelDir;

    console.log('[FacePipeline] Initializing from:', modelDir);

    // Check if model files exist
    const scrfdPath = path.join(modelDir, 'det_10g.onnx');
    const arcfacePath = path.join(modelDir, 'arcfaceresnet100-8.onnx');

    if (!fs.existsSync(scrfdPath)) {
        console.error('[FacePipeline] SCRFD model not found:', scrfdPath);
        return false;
    }
    if (!fs.existsSync(arcfacePath)) {
        console.error('[FacePipeline] ArcFace model not found:', arcfacePath);
        return false;
    }

    // Initialize SCRFD detector
    console.log('[FacePipeline] Loading SCRFD detector...');
    const scrfdOk = await initSCRFD(scrfdPath);
    if (!scrfdOk) {
        console.error('[FacePipeline] Failed to initialize SCRFD');
        return false;
    }

    // Initialize ArcFace
    console.log('[FacePipeline] Loading ArcFace model...');
    const arcfaceOk = await initArcFace(arcfacePath);
    if (!arcfaceOk) {
        console.error('[FacePipeline] Failed to initialize ArcFace');
        return false;
    }

    pipelineReady = true;
    console.log('[FacePipeline] Pipeline ready');
    return true;
}

/**
 * Check if pipeline is ready
 */
function isPipelineReady() {
    return pipelineReady && isSCRFDReady() && isArcFaceReady();
}

/**
 * Extract face embeddings from an image
 * Returns embeddings for all detected faces
 *
 * @param {Buffer|string} input - Image buffer or file path
 * @param {Object} options - Options
 * @param {boolean} options.returnAlignedFace - If true, include aligned face buffer in results
 * @returns {Promise<Array<{embedding: Float32Array, bbox: number[], landmarks: number[][], score: number, alignedFace?: Buffer}>>}
 */
async function extractFaceEmbeddings(input, options = {}) {
    if (!isPipelineReady()) {
        throw new Error('Face pipeline not initialized');
    }

    // Load image if path provided
    let imageBuffer;
    if (typeof input === 'string') {
        // Auto-rotate based on EXIF orientation when loading from file
        imageBuffer = await sharp(input).rotate().toBuffer();
    } else {
        imageBuffer = input;
    }

    const confThreshold = options.confThreshold || PIPELINE_CONFIG.detection.confThreshold;
    const minFaceSize = options.minFaceSize || PIPELINE_CONFIG.detection.minFaceSize;
    const returnAlignedFace = options.returnAlignedFace || false;

    // Detect faces
    const detections = await detectFaces(imageBuffer, { confThreshold });

    // Filter by minimum face size and extract embeddings
    const results = [];

    for (const det of detections) {
        // Check face size
        const faceWidth = det.bbox[2] - det.bbox[0];
        const faceHeight = det.bbox[3] - det.bbox[1];

        if (faceWidth < minFaceSize || faceHeight < minFaceSize) {
            continue;
        }

        // Check if landmarks are available
        if (!det.landmarks || det.landmarks.length !== 5) {
            console.warn('[FacePipeline] No landmarks for face, skipping');
            continue;
        }

        try {
            // Align face
            const { data: alignedData, width, height } = await alignFaceRaw(
                imageBuffer,
                det.landmarks,
                ARCFACE_SIZE
            );

            // Extract embedding
            const embedding = await extractEmbeddingFromRaw(alignedData, width, height);

            const result = {
                embedding,
                bbox: det.bbox,
                landmarks: det.landmarks,
                score: det.score,
            };

            // Optionally include aligned face for debugging
            if (returnAlignedFace) {
                result.alignedFace = await sharp(Buffer.from(alignedData.buffer), {
                    raw: { width, height, channels: 3 }
                }).jpeg({ quality: 95 }).toBuffer();
            }

            results.push(result);
        } catch (e) {
            console.warn('[FacePipeline] Failed to process face:', e.message);
        }
    }

    return results;
}

/**
 * Extract a single face embedding (best/largest face)
 * @param {Buffer|string} input - Image buffer or file path
 * @param {Object} options - Options
 * @returns {Promise<{embedding: Float32Array, bbox: number[], landmarks: number[][], score: number}|null>}
 */
async function extractBestFaceEmbedding(input, options = {}) {
    const faces = await extractFaceEmbeddings(input, options);

    if (faces.length === 0) {
        return null;
    }

    // Find largest face (by area)
    let bestFace = faces[0];
    let bestArea = 0;

    for (const face of faces) {
        const area = (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]);
        if (area > bestArea) {
            bestArea = area;
            bestFace = face;
        }
    }

    return bestFace;
}

/**
 * Compare a face embedding against a set of reference embeddings
 * Returns the best match using "compare all, take max" strategy
 *
 * @param {Float32Array} queryEmbedding - Query face embedding
 * @param {Map<string, Float32Array[]>} references - Map of person -> [embeddings]
 * @param {number} threshold - Similarity threshold
 * @returns {{person: string, similarity: number}|null}
 */
function matchFace(queryEmbedding, references, threshold = PIPELINE_CONFIG.matching.threshold) {
    let bestMatch = null;
    let bestSimilarity = threshold;

    for (const [person, embeddings] of references) {
        // Compare against all reference embeddings for this person
        for (const refEmb of embeddings) {
            const similarity = cosineSimilarity(queryEmbedding, refEmb);

            if (similarity > bestSimilarity) {
                bestSimilarity = similarity;
                bestMatch = { person, similarity };
            }
        }
    }

    return bestMatch;
}

/**
 * Match a face against references and return all scores above threshold
 * @param {Float32Array} queryEmbedding - Query face embedding
 * @param {Map<string, Float32Array[]>} references - Map of person -> [embeddings]
 * @param {number} threshold - Similarity threshold
 * @returns {Array<{person: string, similarity: number, matchingRefs: number}>}
 */
function matchFaceAll(queryEmbedding, references, threshold = PIPELINE_CONFIG.matching.threshold) {
    const matches = [];

    for (const [person, embeddings] of references) {
        // Get max similarity across all references for this person
        let maxSim = 0;
        let matchingRefs = 0;

        for (const refEmb of embeddings) {
            const similarity = cosineSimilarity(queryEmbedding, refEmb);
            if (similarity > maxSim) {
                maxSim = similarity;
            }
            if (similarity > threshold) {
                matchingRefs++;
            }
        }

        if (maxSim > threshold) {
            matches.push({
                person,
                similarity: maxSim,
                matchingRefs,
            });
        }
    }

    // Sort by similarity descending
    matches.sort((a, b) => b.similarity - a.similarity);

    return matches;
}

/**
 * Detect faces without extracting embeddings (faster for face-finding operations)
 * @param {Buffer|string} input - Image buffer or file path
 * @param {Object} options - Options
 * @returns {Promise<Array<{bbox: number[], landmarks: number[][], score: number}>>}
 */
async function detectFacesOnly(input, options = {}) {
    if (!isSCRFDReady()) {
        throw new Error('SCRFD not initialized');
    }

    let imageBuffer;
    if (typeof input === 'string') {
        // Auto-rotate based on EXIF orientation when loading from file
        imageBuffer = await sharp(input).rotate().toBuffer();
    } else {
        imageBuffer = input;
    }

    const confThreshold = options.confThreshold || PIPELINE_CONFIG.detection.confThreshold;
    const minFaceSize = options.minFaceSize || PIPELINE_CONFIG.detection.minFaceSize;

    const detections = await detectFaces(imageBuffer, { confThreshold });

    // Filter by minimum face size
    return detections.filter(det => {
        const faceWidth = det.bbox[2] - det.bbox[0];
        const faceHeight = det.bbox[3] - det.bbox[1];
        return faceWidth >= minFaceSize && faceHeight >= minFaceSize;
    });
}

/**
 * Close pipeline and release resources
 */
async function closePipeline() {
    await closeSCRFD();
    await closeArcFace();
    pipelineReady = false;
    console.log('[FacePipeline] Pipeline closed');
}

// Re-export utility functions
module.exports = {
    // Pipeline functions
    initPipeline,
    isPipelineReady,
    closePipeline,

    // Face processing
    extractFaceEmbeddings,
    extractBestFaceEmbedding,
    detectFacesOnly,

    // Matching
    matchFace,
    matchFaceAll,
    cosineSimilarity,

    // Configuration
    PIPELINE_CONFIG,
    SCRFD_CONFIG,
    ARCFACE_CONFIG,

    // For advanced usage
    alignFace,
    alignFaceRaw,
};

/**
 * SCRFD Face Detector
 * State-of-the-art face detection using SCRFD (Sample and Computation Redistribution)
 * Model: det_10g.onnx from InsightFace buffalo_l
 *
 * Outputs bounding boxes and 5-point facial landmarks for alignment
 */

const path = require('path');
const sharp = require('sharp');

let ort = null;
let scrfdSession = null;

// SCRFD configuration
const SCRFD_CONFIG = {
    inputSize: 640,           // Model input size
    confThreshold: 0.5,       // Detection confidence threshold
    nmsThreshold: 0.4,        // NMS IoU threshold
    // Feature map strides for different scales
    featStrides: [8, 16, 32],
    // Number of anchors per position
    numAnchors: 2,
    // Whether model outputs landmarks (kps)
    useLandmarks: true,
};

// Preprocessing constants (BGR mean and std)
const MEAN = 127.5;
const STD = 128.0;

/**
 * Initialize SCRFD detector
 * @param {string} modelPath - Path to det_10g.onnx
 * @returns {Promise<boolean>}
 */
async function initSCRFD(modelPath) {
    if (scrfdSession) return true;

    try {
        // Load ONNX Runtime
        if (!ort) {
            ort = require('onnxruntime-node');
        }

        console.log('[SCRFD] Loading model from:', modelPath);
        scrfdSession = await ort.InferenceSession.create(modelPath, {
            executionProviders: ['cpu'],
            graphOptimizationLevel: 'all',
        });

        // Log model info
        console.log('[SCRFD] Input names:', scrfdSession.inputNames);
        console.log('[SCRFD] Output names:', scrfdSession.outputNames);
        console.log('[SCRFD] Model loaded successfully');

        return true;
    } catch (e) {
        console.error('[SCRFD] Failed to load model:', e.message);
        scrfdSession = null;
        return false;
    }
}

/**
 * Check if SCRFD is ready
 */
function isSCRFDReady() {
    return scrfdSession !== null;
}

/**
 * Preprocess image for SCRFD
 * @param {Buffer} imageBuffer - Input image buffer
 * @returns {Promise<{tensor: Float32Array, scale: number, padW: number, padH: number, origW: number, origH: number}>}
 */
async function preprocessImage(imageBuffer) {
    const inputSize = SCRFD_CONFIG.inputSize;

    // Get original image dimensions
    const metadata = await sharp(imageBuffer).metadata();
    const origW = metadata.width;
    const origH = metadata.height;

    // Calculate scale to fit in inputSize while maintaining aspect ratio
    const scale = Math.min(inputSize / origW, inputSize / origH);
    const newW = Math.round(origW * scale);
    const newH = Math.round(origH * scale);

    // Calculate padding to center the image
    const padW = Math.floor((inputSize - newW) / 2);
    const padH = Math.floor((inputSize - newH) / 2);

    // Resize and pad image
    const { data } = await sharp(imageBuffer)
        .resize(newW, newH)
        .extend({
            top: padH,
            bottom: inputSize - newH - padH,
            left: padW,
            right: inputSize - newW - padW,
            background: { r: 0, g: 0, b: 0 }
        })
        .removeAlpha()
        .raw()
        .toBuffer({ resolveWithObject: true });

    // Convert to float32 and normalize (RGB to BGR, then normalize)
    // SCRFD expects BGR input, but we have RGB from sharp
    const floatData = new Float32Array(3 * inputSize * inputSize);
    const pixelCount = inputSize * inputSize;

    for (let i = 0; i < pixelCount; i++) {
        const r = data[i * 3];
        const g = data[i * 3 + 1];
        const b = data[i * 3 + 2];

        // NCHW format, BGR order (swap R and B)
        floatData[i] = (b - MEAN) / STD;                    // B channel
        floatData[pixelCount + i] = (g - MEAN) / STD;       // G channel
        floatData[2 * pixelCount + i] = (r - MEAN) / STD;   // R channel
    }

    return {
        tensor: floatData,
        scale,
        padW,
        padH,
        origW,
        origH,
    };
}

/**
 * Generate anchor centers for a given feature stride
 */
function generateAnchorCenters(height, width, stride) {
    const centers = [];
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            // Generate numAnchors anchors per position
            for (let a = 0; a < SCRFD_CONFIG.numAnchors; a++) {
                centers.push({
                    x: x * stride,
                    y: y * stride,
                });
            }
        }
    }
    return centers;
}

/**
 * Decode bounding boxes from SCRFD output
 */
function decodeBboxes(anchorCenters, bboxPreds, stride) {
    const bboxes = [];
    for (let i = 0; i < anchorCenters.length; i++) {
        const cx = anchorCenters[i].x;
        const cy = anchorCenters[i].y;

        // bbox format: [x1, y1, x2, y2] relative to anchor
        const x1 = cx - bboxPreds[i * 4] * stride;
        const y1 = cy - bboxPreds[i * 4 + 1] * stride;
        const x2 = cx + bboxPreds[i * 4 + 2] * stride;
        const y2 = cy + bboxPreds[i * 4 + 3] * stride;

        bboxes.push([x1, y1, x2, y2]);
    }
    return bboxes;
}

/**
 * Decode landmarks from SCRFD output
 */
function decodeLandmarks(anchorCenters, kpsPreds, stride) {
    const landmarks = [];
    for (let i = 0; i < anchorCenters.length; i++) {
        const cx = anchorCenters[i].x;
        const cy = anchorCenters[i].y;

        // 5 landmarks, each with x,y
        const lms = [];
        for (let j = 0; j < 5; j++) {
            const lx = cx + kpsPreds[i * 10 + j * 2] * stride;
            const ly = cy + kpsPreds[i * 10 + j * 2 + 1] * stride;
            lms.push([lx, ly]);
        }
        landmarks.push(lms);
    }
    return landmarks;
}

/**
 * Non-Maximum Suppression
 */
function nms(detections, iouThreshold) {
    if (detections.length === 0) return [];

    // Sort by score descending
    detections.sort((a, b) => b.score - a.score);

    const kept = [];
    const suppressed = new Set();

    for (let i = 0; i < detections.length; i++) {
        if (suppressed.has(i)) continue;

        kept.push(detections[i]);

        for (let j = i + 1; j < detections.length; j++) {
            if (suppressed.has(j)) continue;

            const iou = computeIoU(detections[i].bbox, detections[j].bbox);
            if (iou > iouThreshold) {
                suppressed.add(j);
            }
        }
    }

    return kept;
}

/**
 * Compute Intersection over Union
 */
function computeIoU(boxA, boxB) {
    const xA = Math.max(boxA[0], boxB[0]);
    const yA = Math.max(boxA[1], boxB[1]);
    const xB = Math.min(boxA[2], boxB[2]);
    const yB = Math.min(boxA[3], boxB[3]);

    const interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
    const boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
    const boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);

    return interArea / (boxAArea + boxBArea - interArea + 1e-6);
}

/**
 * Detect faces in an image
 * @param {Buffer} imageBuffer - Input image buffer (any format sharp can read)
 * @param {Object} options - Detection options
 * @returns {Promise<Array<{bbox: number[], landmarks: number[][], score: number}>>}
 */
async function detectFaces(imageBuffer, options = {}) {
    if (!scrfdSession) {
        throw new Error('SCRFD not initialized');
    }

    const confThreshold = options.confThreshold || SCRFD_CONFIG.confThreshold;
    const nmsThreshold = options.nmsThreshold || SCRFD_CONFIG.nmsThreshold;

    // Preprocess image
    const { tensor, scale, padW, padH, origW, origH } = await preprocessImage(imageBuffer);

    // Create input tensor
    const inputTensor = new ort.Tensor('float32', tensor, [1, 3, SCRFD_CONFIG.inputSize, SCRFD_CONFIG.inputSize]);

    // Run inference
    const inputName = scrfdSession.inputNames[0];
    const results = await scrfdSession.run({ [inputName]: inputTensor });

    // Parse outputs - SCRFD has 9 outputs (3 strides × 3 outputs: scores, bboxes, kps)
    // Output naming convention: score_8, score_16, score_32, bbox_8, bbox_16, bbox_32, kps_8, kps_16, kps_32
    const allDetections = [];
    const inputSize = SCRFD_CONFIG.inputSize;

    for (let idx = 0; idx < SCRFD_CONFIG.featStrides.length; idx++) {
        const stride = SCRFD_CONFIG.featStrides[idx];
        const featH = Math.floor(inputSize / stride);
        const featW = Math.floor(inputSize / stride);

        // Get output tensors for this stride
        // Output names might vary, try different naming conventions
        let scores, bboxes, kps;

        // Try InsightFace naming convention
        const scoreKey = `score_${stride}`;
        const bboxKey = `bbox_${stride}`;
        const kpsKey = `kps_${stride}`;

        if (results[scoreKey]) {
            scores = results[scoreKey].data;
            bboxes = results[bboxKey].data;
            kps = results[kpsKey] ? results[kpsKey].data : null;
        } else {
            // Try indexed naming (output_0, output_1, etc.)
            const outputNames = scrfdSession.outputNames;
            scores = results[outputNames[idx]].data;
            bboxes = results[outputNames[idx + 3]].data;
            kps = results[outputNames[idx + 6]] ? results[outputNames[idx + 6]].data : null;
        }

        // Generate anchor centers
        const anchorCenters = generateAnchorCenters(featH, featW, stride);

        // Decode predictions
        const decodedBboxes = decodeBboxes(anchorCenters, bboxes, stride);
        const decodedLandmarks = kps ? decodeLandmarks(anchorCenters, kps, stride) : null;

        // Filter by confidence
        for (let i = 0; i < scores.length; i++) {
            const score = scores[i];
            if (score < confThreshold) continue;

            const bbox = decodedBboxes[i];
            const landmarks = decodedLandmarks ? decodedLandmarks[i] : null;

            // Transform back to original image coordinates
            const x1 = (bbox[0] - padW) / scale;
            const y1 = (bbox[1] - padH) / scale;
            const x2 = (bbox[2] - padW) / scale;
            const y2 = (bbox[3] - padH) / scale;

            // Clamp to image bounds
            const finalBbox = [
                Math.max(0, x1),
                Math.max(0, y1),
                Math.min(origW, x2),
                Math.min(origH, y2),
            ];

            // Transform landmarks
            let finalLandmarks = null;
            if (landmarks) {
                finalLandmarks = landmarks.map(([lx, ly]) => [
                    (lx - padW) / scale,
                    (ly - padH) / scale,
                ]);
            }

            allDetections.push({
                bbox: finalBbox,
                landmarks: finalLandmarks,
                score: score,
            });
        }
    }

    // Apply NMS
    const finalDetections = nms(allDetections, nmsThreshold);

    return finalDetections;
}

/**
 * Cleanup SCRFD resources
 */
async function closeSCRFD() {
    if (scrfdSession) {
        await scrfdSession.release();
        scrfdSession = null;
    }
}

module.exports = {
    initSCRFD,
    isSCRFDReady,
    detectFaces,
    closeSCRFD,
    SCRFD_CONFIG,
};

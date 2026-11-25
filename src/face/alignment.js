/**
 * Face Alignment Utilities
 * Transforms detected faces to standard 112x112 alignment for ArcFace
 *
 * Uses similarity transform (scale + rotation + translation) computed from
 * 5-point facial landmarks to ArcFace reference landmarks.
 */

const sharp = require('sharp');

// ArcFace standard reference landmarks for 112x112 output
// These are the target positions the detected landmarks should be warped to
const ARCFACE_REF_LANDMARKS = [
    [38.2946, 51.6963],   // left eye
    [73.5318, 51.5014],   // right eye
    [56.0252, 71.7366],   // nose tip
    [41.5493, 92.3655],   // mouth left
    [70.7299, 92.2041],   // mouth right
];

// Output size for ArcFace
const ARCFACE_SIZE = 112;

/**
 * Compute similarity transform matrix from source to destination points
 * Similarity transform: scale * rotation + translation
 *
 * Uses Umeyama algorithm (least squares estimation)
 *
 * @param {number[][]} src - Source landmarks [[x,y], ...]
 * @param {number[][]} dst - Destination landmarks [[x,y], ...]
 * @returns {{M: number[][], scale: number}} - 2x3 affine transform matrix and scale
 */
function computeSimilarityTransform(src, dst) {
    const n = src.length;

    // Compute centroids
    let srcCx = 0, srcCy = 0, dstCx = 0, dstCy = 0;
    for (let i = 0; i < n; i++) {
        srcCx += src[i][0];
        srcCy += src[i][1];
        dstCx += dst[i][0];
        dstCy += dst[i][1];
    }
    srcCx /= n;
    srcCy /= n;
    dstCx /= n;
    dstCy /= n;

    // Center the points
    const srcCentered = src.map(p => [p[0] - srcCx, p[1] - srcCy]);
    const dstCentered = dst.map(p => [p[0] - dstCx, p[1] - dstCy]);

    // Compute covariance H = dst * src^T (Umeyama algorithm)
    // This gives us the transform from src to dst
    let cov00 = 0, cov01 = 0, cov10 = 0, cov11 = 0;
    let srcVar = 0;

    for (let i = 0; i < n; i++) {
        // H[i][j] = dst[i] * src[j] for covariance dst * src^T
        cov00 += dstCentered[i][0] * srcCentered[i][0];
        cov01 += dstCentered[i][0] * srcCentered[i][1];
        cov10 += dstCentered[i][1] * srcCentered[i][0];
        cov11 += dstCentered[i][1] * srcCentered[i][1];
        srcVar += srcCentered[i][0] * srcCentered[i][0] + srcCentered[i][1] * srcCentered[i][1];
    }

    // NOTE: Don't divide srcVar by n here because the covariance matrix
    // is also not normalized by n. Both need to be consistent for correct scale.

    // Compute SVD of covariance matrix
    // For 2x2 matrix, we can compute SVD analytically
    const cov = [[cov00, cov01], [cov10, cov11]];
    const { U, S, V } = svd2x2(cov);

    // Compute rotation matrix R = V * U^T
    // But we need to handle reflection case
    let d = [1, 1];
    if (det2x2(cov) < 0) {
        d[1] = -1;
    }

    // R = V * diag(d) * U^T
    const R = [
        [V[0][0] * d[0] * U[0][0] + V[0][1] * d[1] * U[0][1],
         V[0][0] * d[0] * U[1][0] + V[0][1] * d[1] * U[1][1]],
        [V[1][0] * d[0] * U[0][0] + V[1][1] * d[1] * U[0][1],
         V[1][0] * d[0] * U[1][0] + V[1][1] * d[1] * U[1][1]],
    ];

    // Compute scale
    const traceS = S[0] * d[0] + S[1] * d[1];
    const scale = traceS / srcVar;

    // Compute translation
    const tx = dstCx - scale * (R[0][0] * srcCx + R[0][1] * srcCy);
    const ty = dstCy - scale * (R[1][0] * srcCx + R[1][1] * srcCy);

    // Build 2x3 affine transform matrix
    const M = [
        [scale * R[0][0], scale * R[0][1], tx],
        [scale * R[1][0], scale * R[1][1], ty],
    ];

    return { M, scale };
}

/**
 * Compute 2x2 SVD analytically
 */
function svd2x2(A) {
    const a = A[0][0], b = A[0][1], c = A[1][0], d = A[1][1];

    // Compute A^T * A
    const ata00 = a * a + c * c;
    const ata01 = a * b + c * d;
    const ata11 = b * b + d * d;

    // Eigenvalues of A^T * A (singular values squared)
    const trace = ata00 + ata11;
    const det = ata00 * ata11 - ata01 * ata01;
    const disc = Math.sqrt(Math.max(0, trace * trace / 4 - det));

    const s1Sq = trace / 2 + disc;
    const s2Sq = trace / 2 - disc;

    const s1 = Math.sqrt(Math.max(0, s1Sq));
    const s2 = Math.sqrt(Math.max(0, s2Sq));

    // Compute V (right singular vectors)
    let V;
    if (Math.abs(ata01) > 1e-10) {
        const v1x = s1Sq - ata11;
        const v1y = ata01;
        const v1Norm = Math.sqrt(v1x * v1x + v1y * v1y);

        const v2x = s2Sq - ata11;
        const v2y = ata01;
        const v2Norm = Math.sqrt(v2x * v2x + v2y * v2y);

        V = [
            [v1x / v1Norm, v2x / v2Norm],
            [v1y / v1Norm, v2y / v2Norm],
        ];
    } else {
        V = [[1, 0], [0, 1]];
    }

    // Compute U = A * V * S^-1
    let U;
    if (s1 > 1e-10 && s2 > 1e-10) {
        const av1x = a * V[0][0] + b * V[1][0];
        const av1y = c * V[0][0] + d * V[1][0];
        const av2x = a * V[0][1] + b * V[1][1];
        const av2y = c * V[0][1] + d * V[1][1];

        U = [
            [av1x / s1, av2x / s2],
            [av1y / s1, av2y / s2],
        ];
    } else if (s1 > 1e-10) {
        const av1x = a * V[0][0] + b * V[1][0];
        const av1y = c * V[0][0] + d * V[1][0];
        U = [
            [av1x / s1, -av1y / s1],
            [av1y / s1, av1x / s1],
        ];
    } else {
        U = [[1, 0], [0, 1]];
    }

    return { U, S: [s1, s2], V };
}

/**
 * Compute 2x2 determinant
 */
function det2x2(A) {
    return A[0][0] * A[1][1] - A[0][1] * A[1][0];
}

/**
 * Apply affine transformation to an image using sharp
 *
 * @param {Buffer} imageBuffer - Source image buffer
 * @param {number[][]} srcLandmarks - 5 detected facial landmarks [[x,y], ...]
 * @param {number} outputSize - Output image size (default 112 for ArcFace)
 * @returns {Promise<Buffer>} - Aligned face image buffer (RGB)
 */
async function alignFace(imageBuffer, srcLandmarks, outputSize = ARCFACE_SIZE) {
    // Scale reference landmarks if output size is different
    const scale = outputSize / ARCFACE_SIZE;
    const dstLandmarks = ARCFACE_REF_LANDMARKS.map(p => [p[0] * scale, p[1] * scale]);

    // Compute similarity transform
    const { M } = computeSimilarityTransform(srcLandmarks, dstLandmarks);

    // Get source image metadata
    const metadata = await sharp(imageBuffer).metadata();
    const srcWidth = metadata.width;
    const srcHeight = metadata.height;

    // We need to apply the inverse transform because we're mapping
    // destination pixels to source pixels
    // M_inv = inverse of M (need to compute 2x3 pseudo-inverse)
    const det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
    const invDet = 1 / det;

    const M_inv = [
        [M[1][1] * invDet, -M[0][1] * invDet, 0],
        [-M[1][0] * invDet, M[0][0] * invDet, 0],
    ];
    // Compute translation part of inverse
    M_inv[0][2] = -(M_inv[0][0] * M[0][2] + M_inv[0][1] * M[1][2]);
    M_inv[1][2] = -(M_inv[1][0] * M[0][2] + M_inv[1][1] * M[1][2]);

    // Sharp doesn't support arbitrary affine transforms directly,
    // so we need to manually warp the image
    // We'll create the output by sampling from the source

    // Get raw pixel data from source
    const { data: srcData, info } = await sharp(imageBuffer)
        .removeAlpha()
        .raw()
        .toBuffer({ resolveWithObject: true });

    // Create output buffer
    const outData = Buffer.alloc(outputSize * outputSize * 3);

    // For each output pixel, find corresponding source pixel
    for (let y = 0; y < outputSize; y++) {
        for (let x = 0; x < outputSize; x++) {
            // Apply inverse transform to get source coordinates
            const srcX = M_inv[0][0] * x + M_inv[0][1] * y + M_inv[0][2];
            const srcY = M_inv[1][0] * x + M_inv[1][1] * y + M_inv[1][2];

            // Bilinear interpolation
            const x0 = Math.floor(srcX);
            const y0 = Math.floor(srcY);
            const x1 = x0 + 1;
            const y1 = y0 + 1;

            const dx = srcX - x0;
            const dy = srcY - y0;

            const outIdx = (y * outputSize + x) * 3;

            // Check bounds
            if (x0 >= 0 && x1 < srcWidth && y0 >= 0 && y1 < srcHeight) {
                const idx00 = (y0 * srcWidth + x0) * 3;
                const idx01 = (y0 * srcWidth + x1) * 3;
                const idx10 = (y1 * srcWidth + x0) * 3;
                const idx11 = (y1 * srcWidth + x1) * 3;

                for (let c = 0; c < 3; c++) {
                    const v00 = srcData[idx00 + c];
                    const v01 = srcData[idx01 + c];
                    const v10 = srcData[idx10 + c];
                    const v11 = srcData[idx11 + c];

                    const v = v00 * (1 - dx) * (1 - dy) +
                              v01 * dx * (1 - dy) +
                              v10 * (1 - dx) * dy +
                              v11 * dx * dy;

                    outData[outIdx + c] = Math.round(Math.max(0, Math.min(255, v)));
                }
            } else if (x0 >= 0 && x0 < srcWidth && y0 >= 0 && y0 < srcHeight) {
                // Nearest neighbor for edge cases
                const idx = (y0 * srcWidth + x0) * 3;
                outData[outIdx] = srcData[idx];
                outData[outIdx + 1] = srcData[idx + 1];
                outData[outIdx + 2] = srcData[idx + 2];
            }
            // else: leave as black (already zeros)
        }
    }

    // Convert raw buffer to image buffer
    const alignedBuffer = await sharp(outData, {
        raw: {
            width: outputSize,
            height: outputSize,
            channels: 3,
        },
    })
        .jpeg({ quality: 95 })
        .toBuffer();

    return alignedBuffer;
}

/**
 * Align face and return raw RGB data for direct model input
 *
 * @param {Buffer} imageBuffer - Source image buffer
 * @param {number[][]} srcLandmarks - 5 detected facial landmarks
 * @param {number} outputSize - Output size (default 112)
 * @returns {Promise<{data: Uint8Array, width: number, height: number}>}
 */
async function alignFaceRaw(imageBuffer, srcLandmarks, outputSize = ARCFACE_SIZE) {
    // Scale reference landmarks if output size is different
    const scale = outputSize / ARCFACE_SIZE;
    const dstLandmarks = ARCFACE_REF_LANDMARKS.map(p => [p[0] * scale, p[1] * scale]);

    // Compute similarity transform
    const { M } = computeSimilarityTransform(srcLandmarks, dstLandmarks);

    // Get source image metadata and raw data
    const { data: srcData, info } = await sharp(imageBuffer)
        .removeAlpha()
        .raw()
        .toBuffer({ resolveWithObject: true });

    const srcWidth = info.width;
    const srcHeight = info.height;

    // Compute inverse transform
    const det = M[0][0] * M[1][1] - M[0][1] * M[1][0];
    const invDet = 1 / det;

    const M_inv = [
        [M[1][1] * invDet, -M[0][1] * invDet, 0],
        [-M[1][0] * invDet, M[0][0] * invDet, 0],
    ];
    M_inv[0][2] = -(M_inv[0][0] * M[0][2] + M_inv[0][1] * M[1][2]);
    M_inv[1][2] = -(M_inv[1][0] * M[0][2] + M_inv[1][1] * M[1][2]);

    // Create output buffer
    const outData = new Uint8Array(outputSize * outputSize * 3);

    // Warp with bilinear interpolation
    for (let y = 0; y < outputSize; y++) {
        for (let x = 0; x < outputSize; x++) {
            const srcX = M_inv[0][0] * x + M_inv[0][1] * y + M_inv[0][2];
            const srcY = M_inv[1][0] * x + M_inv[1][1] * y + M_inv[1][2];

            const x0 = Math.floor(srcX);
            const y0 = Math.floor(srcY);
            const x1 = x0 + 1;
            const y1 = y0 + 1;

            const dx = srcX - x0;
            const dy = srcY - y0;

            const outIdx = (y * outputSize + x) * 3;

            if (x0 >= 0 && x1 < srcWidth && y0 >= 0 && y1 < srcHeight) {
                const idx00 = (y0 * srcWidth + x0) * 3;
                const idx01 = (y0 * srcWidth + x1) * 3;
                const idx10 = (y1 * srcWidth + x0) * 3;
                const idx11 = (y1 * srcWidth + x1) * 3;

                for (let c = 0; c < 3; c++) {
                    const v = srcData[idx00 + c] * (1 - dx) * (1 - dy) +
                              srcData[idx01 + c] * dx * (1 - dy) +
                              srcData[idx10 + c] * (1 - dx) * dy +
                              srcData[idx11 + c] * dx * dy;
                    outData[outIdx + c] = Math.round(Math.max(0, Math.min(255, v)));
                }
            } else if (x0 >= 0 && x0 < srcWidth && y0 >= 0 && y0 < srcHeight) {
                const idx = (y0 * srcWidth + x0) * 3;
                outData[outIdx] = srcData[idx];
                outData[outIdx + 1] = srcData[idx + 1];
                outData[outIdx + 2] = srcData[idx + 2];
            }
        }
    }

    return {
        data: outData,
        width: outputSize,
        height: outputSize,
    };
}

module.exports = {
    ARCFACE_REF_LANDMARKS,
    ARCFACE_SIZE,
    computeSimilarityTransform,
    alignFace,
    alignFaceRaw,
};

import { execFile } from 'child_process';
import { promisify } from 'util';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import sharp from 'sharp';

const execFileAsync = promisify(execFile);

const VIDEO_EXTENSIONS = new Set(['.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v', '.wmv']);

// Find ffmpeg/ffprobe binary - Electron apps often don't have /opt/homebrew/bin in PATH
function findBinary(name: string): string {
  const candidates = [
    `/opt/homebrew/bin/${name}`,
    `/usr/local/bin/${name}`,
    `/usr/bin/${name}`,
    name, // fallback to PATH
  ];
  for (const candidate of candidates) {
    try {
      if (candidate === name) return name;
      if (fs.existsSync(candidate)) return candidate;
    } catch {}
  }
  return name;
}

const FFPROBE = findBinary('ffprobe');
const FFMPEG = findBinary('ffmpeg');

export function isVideoFile(filePath: string): boolean {
  const ext = path.extname(filePath).toLowerCase();
  return VIDEO_EXTENSIONS.has(ext);
}

async function getVideoDuration(videoPath: string): Promise<number> {
  const { stdout } = await execFileAsync(FFPROBE, [
    '-v', 'error',
    '-show_entries', 'format=duration',
    '-of', 'default=noprint_wrappers=1:nokey=1',
    videoPath,
  ]);
  return parseFloat(stdout.trim());
}

async function extractFrame(videoPath: string, timestampSeconds: number, outputPath: string): Promise<void> {
  await execFileAsync(FFMPEG, [
    '-ss', String(timestampSeconds),
    '-i', videoPath,
    '-vframes', '1',
    '-q:v', '2',
    '-y',
    outputPath,
  ]);
}

export async function extractAndCompositeFrames(videoPath: string): Promise<Buffer> {
  const tmpDir = await fs.promises.mkdtemp(path.join(os.tmpdir(), 'eagle-tagger-'));

  try {
    const duration = await getVideoDuration(videoPath);

    const timestamps = [
      0.5,                    // start (0.5s in to avoid black frames)
      duration / 2,           // middle
      Math.max(0, duration - 0.5),  // end (0.5s before end)
    ];

    const framePaths: string[] = [];
    for (let i = 0; i < timestamps.length; i++) {
      const framePath = path.join(tmpDir, `frame_${i}.jpg`);
      await extractFrame(videoPath, timestamps[i], framePath);
      framePaths.push(framePath);
    }

    // Read all frames and resize to consistent height
    const targetHeight = 480;
    const frameBuffers = await Promise.all(
      framePaths.map(async (fp) => {
        return sharp(fp)
          .resize({ height: targetHeight })
          .jpeg({ quality: 80 })
          .toBuffer();
      })
    );

    // Get metadata for each resized frame to know widths
    const frameMeta = await Promise.all(
      frameBuffers.map(buf => sharp(buf).metadata())
    );

    const totalWidth = frameMeta.reduce((sum, m) => sum + (m.width || 0), 0);

    // Composite frames side by side
    let leftOffset = 0;
    const composites = frameBuffers.map((buf, i) => {
      const input = { input: buf, left: leftOffset, top: 0 };
      leftOffset += frameMeta[i].width || 0;
      return input;
    });

    const composite = await sharp({
      create: {
        width: totalWidth,
        height: targetHeight,
        channels: 3,
        background: { r: 0, g: 0, b: 0 },
      },
    })
      .composite(composites)
      .jpeg({ quality: 80 })
      .toBuffer();

    return composite;
  } finally {
    // Clean up temp files
    await fs.promises.rm(tmpDir, { recursive: true, force: true });
  }
}

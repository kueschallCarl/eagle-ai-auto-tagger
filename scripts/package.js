const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

const root = path.resolve(__dirname, '..');
const outDir = path.join(root, 'build');
const pluginDir = path.join(outDir, 'eagle-ai-auto-tagger');

function copyDirSync(src, dest) {
  fs.mkdirSync(dest, { recursive: true });
  for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);
    if (entry.isDirectory()) {
      copyDirSync(srcPath, destPath);
    } else if (!entry.name.endsWith('.map') && !entry.name.endsWith('.d.ts')) {
      fs.copyFileSync(srcPath, destPath);
    }
  }
}

// Clean
if (fs.existsSync(outDir)) {
  fs.rmSync(outDir, { recursive: true });
}

fs.mkdirSync(pluginDir, { recursive: true });

// Copy plugin files
fs.copyFileSync(path.join(root, 'manifest.json'), path.join(pluginDir, 'manifest.json'));
fs.copyFileSync(path.join(root, 'index.html'), path.join(pluginDir, 'index.html'));
fs.copyFileSync(path.join(root, 'logo.png'), path.join(pluginDir, 'logo.png'));
copyDirSync(path.join(root, 'js'), path.join(pluginDir, 'js'));
copyDirSync(path.join(root, 'dist'), path.join(pluginDir, 'dist'));
copyDirSync(path.join(root, 'src', 'face'), path.join(pluginDir, 'src', 'face'));

// Face models are downloaded separately (see README); bundle them when present
const faceModels = path.join(root, 'models', 'face');
if (fs.existsSync(faceModels)) {
  copyDirSync(faceModels, path.join(pluginDir, 'models', 'face'));
} else {
  console.warn('models/face/ not found: person recognition will be unavailable in this build');
}

// Install production dependencies in plugin directory
fs.copyFileSync(path.join(root, 'package.json'), path.join(pluginDir, 'package.json'));
execSync('npm install --omit=dev', { cwd: pluginDir, stdio: 'inherit' });
fs.unlinkSync(path.join(pluginDir, 'package.json'));
const lockFile = path.join(pluginDir, 'package-lock.json');
if (fs.existsSync(lockFile)) fs.unlinkSync(lockFile);

// Create .eagleplugin zip
const eppPath = path.join(outDir, 'eagle-ai-auto-tagger.eagleplugin');
execSync(`cd "${outDir}" && zip -r "${eppPath}" eagle-ai-auto-tagger/`, { stdio: 'inherit' });

console.log(`\nPlugin packaged: ${eppPath}`);
console.log(`Size: ${(fs.statSync(eppPath).size / 1024 / 1024).toFixed(1)} MB`);

import rawPlugin from 'vite-raw-plugin';
import { defineConfig } from 'vite'
import { existsSync } from 'node:fs';

const hasBundledBonsai = ['bonsai.ply', 'cameras.json'].every(file =>
    existsSync(new URL(`./public/scenes/bonsai/${file}`, import.meta.url)));

export default defineConfig({
    define: {
        'import.meta.env.VITE_HAS_BUNDLED_BONSAI': JSON.stringify(hasBundledBonsai),
    },
    build: {
        target: 'esnext'
    },
    base: process.env.GITHUB_ACTIONS_BASE
        ? `/${process.env.GITHUB_ACTIONS_BASE.replace(/^\/+|\/+$/g, '')}/`
        : '/',
    plugins: [
        rawPlugin({
            fileRegex: /\.wgsl$/,
        }),
    ],
})

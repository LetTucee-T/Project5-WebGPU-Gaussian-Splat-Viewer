import rawPlugin from 'vite-raw-plugin';
import { defineConfig } from 'vite'

export default defineConfig({
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

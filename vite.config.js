import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  root: __dirname,
  base: './', // Relative paths for deployment
  build: {
    outDir: 'docs',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'index.html'),
        demos: resolve(__dirname, 'demos/index.html'),
        toychest: resolve(__dirname, 'demos/toychest.html'),
        mpmHeadless: resolve(__dirname, 'demos/mpm-headless.html'),
        mpmVisual: resolve(__dirname, 'demos/mpm-visual.html'),
      },
    },
  },
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
    },
  },
  server: {
    port: 5173,
    open: '/demos/toychest.html',
  },
  assetsInclude: ['**/*.wgsl'],
});

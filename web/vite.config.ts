import { sveltekit } from '@sveltejs/kit/vite';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'vite';
import wasm from 'vite-plugin-wasm';

export default defineConfig({
    plugins: [tailwindcss(), wasm(), sveltekit()],
    // The wasm-pack output uses ESM `import.meta.url` to find the .wasm
    // sibling. Marking the package as not-pre-bundled lets vite serve it
    // as-is in dev and preserves that resolution.
    optimizeDeps: { exclude: ['$lib/basin-wasm'] },
});

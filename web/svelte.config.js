import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

// GitHub Pages serves repo sites under `<user>.github.io/<repo>/`. Set
// the prefix via `BASIN_BASE_PATH` when deploying (`/basin` for Pages,
// empty for `npm run dev`/`preview` and any custom-domain deploy).
const base = process.env.BASIN_BASE_PATH ?? '';

/** @type {import('@sveltejs/kit').Config} */
const config = {
    preprocess: vitePreprocess(),
    kit: {
        adapter: adapter({
            // SPA mode: a single fallback page lets client-side routing work
            // on Pages, which has no server-side rewrite layer.
            fallback: 'index.html',
        }),
        paths: { base },
        prerender: { entries: ['*'] },
    },
};

export default config;

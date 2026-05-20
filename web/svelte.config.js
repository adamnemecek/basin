import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';
import { mdsvex } from 'mdsvex';

// GitHub Pages serves repo sites under `<user>.github.io/<repo>/`. Set
// the prefix via `BASIN_BASE_PATH` when deploying (`/basin` for Pages,
// empty for `npm run dev`/`preview` and any custom-domain deploy).
const base = process.env.BASIN_BASE_PATH ?? '';

/** @type {import('mdsvex').MdsvexOptions} */
const mdsvexConfig = {
    extensions: ['.svx', '.md'],
    // Every `.svx`/`.md` page is wrapped in this layout, which applies
    // the `prose` typography styling once instead of per-page.
    layout: {
        _: new URL(
            './src/lib/docs/mdsvex-layout.svelte',
            import.meta.url,
        ).pathname,
    },
};

/** @type {import('@sveltejs/kit').Config} */
const config = {
    // Top-level extensions so SvelteKit's router treats `.svx`/`.md` as
    // route files; mdsvex's own `extensions` (above) controls which files
    // it transforms.
    extensions: ['.svelte', '.svx', '.md'],
    preprocess: [vitePreprocess(), mdsvex(mdsvexConfig)],
    kit: {
        // Every linked route is prerendered to its own `index.html`, so
        // docs/landing ship real HTML (SEO + fast load) — this is NOT SPA
        // mode (no `index.html` catch-all). The `404.html` fallback is the
        // one client-rendered page: GitHub Pages serves it for unmatched
        // paths, giving a styled not-found instead of Pages' default.
        adapter: adapter({ fallback: '404.html' }),
        paths: { base },
        prerender: { entries: ['*'] },
    },
};

export default config;

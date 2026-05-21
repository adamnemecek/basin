import { base } from '$app/paths';

// Same canonical origin as the sitemap. The deployed site is at
// `https://jolars.github.io/basin/` (origin + `base`); `base` is `/basin`
// on the Pages build and empty in dev. A static `static/robots.txt`
// couldn't carry the base-aware absolute `Sitemap:` URL, so this is an
// endpoint like sitemap.xml rather than a static file.
const SITE_ORIGIN = 'https://jolars.github.io';

// Prerendered into `build/robots.txt` by the static adapter; reached by
// the `prerender.entries: ['*']` crawl even though nothing links to it.
export const prerender = true;

export function GET() {
    const sitemap = `${SITE_ORIGIN}${base}/sitemap.xml`;

    const body = `User-agent: *
Allow: /

Sitemap: ${sitemap}
`;

    return new Response(body, {
        headers: {
            'Content-Type': 'text/plain',
        },
    });
}

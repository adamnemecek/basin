// Shared navigation model. `href` is always relative to the SvelteKit
// `base` (it starts with `/` and is prefixed with `base` at the call
// site) — never hardcode `/basin`. `section` is the first path segment,
// used for active-state matching across a whole section (e.g. any
// `/docs/*` page lights up the "Docs" link).

export type NavLink = {
    label: string;
    href: string;
    /** First path segment for active-state matching. Omit for external links. */
    section?: string;
    external?: boolean;
};

// `href`s carry a trailing slash to match `trailingSlash: 'always'`
// (set in the root `+layout.ts`), so links hit the canonical URL with no
// redirect hop.

/** Top-level site navigation, shown in the header. */
export const NAV_LINKS: NavLink[] = [
    { label: 'Docs', href: '/docs/getting-started/', section: 'docs' },
    { label: 'Visualizer', href: '/visualizer/', section: 'visualizer' },
    { label: 'Benchmarks', href: '/benchmarks/', section: 'benchmarks' },
    {
        label: 'GitHub',
        href: 'https://github.com/jolars/basin',
        external: true,
    },
];

/** Sidebar links for the docs section. */
export const DOCS_LINKS: NavLink[] = [
    { label: 'Overview', href: '/docs/', section: 'docs' },
    {
        label: 'Getting started',
        href: '/docs/getting-started/',
        section: 'docs',
    },
    { label: 'Solvers', href: '/docs/solvers/', section: 'docs' },
];

/**
 * The active section for a pathname, independent of the `base` prefix.
 * Returns the first path segment (`'docs'`, `'visualizer'`, …) or `''`
 * for the landing page.
 */
export function activeSection(pathname: string, base: string): string {
    const rel =
        base && pathname.startsWith(base) ? pathname.slice(base.length) : pathname;
    return rel.split('/')[1] ?? '';
}

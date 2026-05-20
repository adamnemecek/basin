<script lang="ts">
    import { base } from '$app/paths';
    import { page } from '$app/state';
    import { DOCS_LINKS } from '$lib/nav';

    let { children } = $props();

    // Exact-match active state, tolerant of a missing/extra trailing
    // slash so highlighting survives either URL form.
    const strip = (s: string) => s.replace(/\/+$/, '');
    let path = $derived(strip(page.url.pathname));
</script>

<div
    class="max-w-screen-2xl mx-auto px-4 md:px-8 py-10 grid lg:grid-cols-[14rem_1fr] gap-10"
>
    <aside class="lg:sticky lg:top-20 lg:self-start">
        <p
            class="text-xs font-semibold uppercase tracking-wider text-slate-400 dark:text-slate-500 mb-3"
        >
            Documentation
        </p>
        <nav class="flex flex-col gap-1 text-sm">
            {#each DOCS_LINKS as link}
                <a
                    href="{base}{link.href}"
                    aria-current={path === strip(base + link.href)
                        ? 'page'
                        : undefined}
                    class="px-3 py-1.5 rounded-md transition-colors {path ===
                    strip(base + link.href)
                        ? 'bg-slate-100 text-slate-900 font-medium dark:bg-slate-800 dark:text-slate-100'
                        : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100 dark:text-slate-400 dark:hover:text-slate-100 dark:hover:bg-slate-800'}"
                >
                    {link.label}
                </a>
            {/each}
        </nav>
    </aside>

    <div class="min-w-0">
        {@render children()}
    </div>
</div>

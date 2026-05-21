<script lang="ts">
import "../app.css";
import { base } from "$app/paths";
import { page } from "$app/state";
import { afterNavigate } from "$app/navigation";
import IconMenu from "~icons/lucide/menu";
import IconClose from "~icons/lucide/x";
import ThemeToggle from "$lib/ThemeToggle.svelte";
import { theme } from "$lib/theme.svelte";
import { NAV_LINKS, activeSection } from "$lib/nav";

let { children } = $props();

let current = $derived(activeSection(page.url.pathname, base));

// Mobile nav disclosure. The links live inline at `md+`; below that
// they collapse behind a menu button into the panel under the header.
// Close on navigation (afterNavigate also fires once on mount) and on
// Escape so the panel never lingers across pages or traps focus.
let menuOpen = $state(false);
afterNavigate(() => {
    menuOpen = false;
});

// Reflect the resolved (light/dark) theme onto `<html>` so Tailwind
// dark: variants apply everywhere. Lives in the root layout so it
// runs on every page (the inline script in app.html handles the
// pre-hydration paint; this keeps the class in sync afterwards).
// Effects only run in the browser, but the guard is kept for clarity.
$effect(() => {
    if (typeof document === "undefined") return;
    const root = document.documentElement;
    if (theme.effective === "dark") {
        root.classList.add("dark");
        root.classList.remove("light");
        root.style.colorScheme = "dark";
    } else {
        root.classList.add("light");
        root.classList.remove("dark");
        root.style.colorScheme = "light";
    }
});
</script>

<svelte:window
    onkeydown={(e) => {
        if (e.key === "Escape") menuOpen = false;
    }}
/>

<!-- Shared nav links, rendered both inline (md+) and in the mobile panel.
     `extra` lets the mobile copy go full-width (`block`) while the inline
     copy stays inline. -->
{#snippet navItems(extra: string)}
    {#each NAV_LINKS as link}
        <li>
            {#if link.external}
                <a
                    href={link.href}
                    target="_blank"
                    rel="noreferrer"
                    class="{extra} px-3 py-1.5 rounded-md text-slate-600 hover:text-slate-900 hover:bg-slate-100 dark:text-slate-400 dark:hover:text-slate-100 dark:hover:bg-slate-800 transition-colors"
                >
                    {link.label}
                </a>
            {:else}
                <a
                    href="{base}{link.href}"
                    aria-current={current === link.section ? "page" : undefined}
                    class="{extra} px-3 py-1.5 rounded-md transition-colors {current ===
                    link.section
                        ? 'text-slate-900 bg-slate-100 dark:text-slate-100 dark:bg-slate-800'
                        : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100 dark:text-slate-400 dark:hover:text-slate-100 dark:hover:bg-slate-800'}"
                >
                    {link.label}
                </a>
            {/if}
        </li>
    {/each}
{/snippet}

<div class="min-h-screen flex flex-col">
    <header
        class="border-b border-slate-200 dark:border-slate-800 sticky top-0 z-20 bg-white/80 dark:bg-slate-950/80 backdrop-blur"
    >
        <nav
            class="max-w-screen-2xl mx-auto px-4 md:px-8 h-14 flex items-center gap-3 md:gap-6"
        >
            <!-- Logo slot. Swap this wordmark for an <img> once a logo
                 asset lands in `static/` (e.g.
                 `<img src="{base}/logo.svg" alt="basin" class="h-6" />`). -->
            <a
                href="{base}/"
                class="font-semibold tracking-tight text-lg hover:text-slate-600 dark:hover:text-slate-300"
            >
                basin
            </a>

            <div class="flex-1"></div>

            <!-- Inline links at md+; collapse behind the menu button below. -->
            <ul class="hidden md:flex items-center gap-1 text-sm">
                {@render navItems("")}
            </ul>

            <div class="pl-2 border-l border-slate-200 dark:border-slate-800">
                <ThemeToggle />
            </div>

            <button
                type="button"
                class="md:hidden inline-flex items-center justify-center rounded-md p-1.5 text-slate-600 hover:text-slate-900 hover:bg-slate-100 dark:text-slate-400 dark:hover:text-slate-100 dark:hover:bg-slate-800 transition-colors"
                aria-label="Toggle navigation menu"
                aria-expanded={menuOpen}
                aria-controls="mobile-nav"
                onclick={() => (menuOpen = !menuOpen)}
            >
                {#if menuOpen}
                    <IconClose width="20" height="20" aria-hidden="true" />
                {:else}
                    <IconMenu width="20" height="20" aria-hidden="true" />
                {/if}
            </button>
        </nav>

        {#if menuOpen}
            <ul
                id="mobile-nav"
                class="md:hidden border-t border-slate-200 dark:border-slate-800 px-4 py-2 flex flex-col gap-1 text-sm"
            >
                {@render navItems("block")}
            </ul>
        {/if}
    </header>

    <main class="flex-1">
        {@render children()}
    </main>

    <footer
        class="border-t border-slate-200 dark:border-slate-800 mt-auto"
    >
        <div
            class="max-w-screen-2xl mx-auto px-4 md:px-8 py-6 flex flex-wrap items-center justify-between gap-3 text-sm text-slate-500 dark:text-slate-500"
        >
            <p>
                <span class="font-semibold text-slate-700 dark:text-slate-300"
                    >basin</span
                >
                — numerical optimization for Rust.
            </p>
            <div class="flex items-center gap-4">
                <a
                    href="https://github.com/jolars/basin"
                    target="_blank"
                    rel="noreferrer"
                    class="hover:text-slate-900 dark:hover:text-slate-200"
                    >GitHub</a
                >
                <a
                    href="https://docs.rs/basin"
                    target="_blank"
                    rel="noreferrer"
                    class="hover:text-slate-900 dark:hover:text-slate-200"
                    >docs.rs</a
                >
                <a
                    href="https://crates.io/crates/basin"
                    target="_blank"
                    rel="noreferrer"
                    class="hover:text-slate-900 dark:hover:text-slate-200"
                    >crates.io</a
                >
            </div>
        </div>
    </footer>
</div>

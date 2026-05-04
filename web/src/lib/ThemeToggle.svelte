<script lang="ts">
    import { theme, type ThemePreference } from './theme.svelte';

    const OPTIONS: { value: ThemePreference; label: string; title: string }[] = [
        { value: 'light', label: 'Light', title: 'Force light theme' },
        { value: 'auto', label: 'Auto', title: 'Follow system preference' },
        { value: 'dark', label: 'Dark', title: 'Force dark theme' },
    ];
</script>

<div
    role="radiogroup"
    aria-label="Color theme"
    class="inline-flex rounded-md border border-slate-300 bg-slate-100 p-0.5 text-xs dark:border-slate-700 dark:bg-slate-800"
>
    {#each OPTIONS as opt}
        <button
            type="button"
            role="radio"
            aria-checked={theme.preference === opt.value}
            title={opt.title}
            onclick={() => theme.set(opt.value)}
            class="px-2 py-1 rounded transition-colors {theme.preference ===
            opt.value
                ? 'bg-white text-slate-900 shadow-sm dark:bg-slate-950 dark:text-slate-100'
                : 'text-slate-500 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-100'}"
        >
            {opt.label}
        </button>
    {/each}
</div>

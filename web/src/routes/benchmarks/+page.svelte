<script lang="ts">
import { base } from "$app/paths";
import Seo from "$lib/Seo.svelte";

// The planned benchmark axes (see the project's benchmark suite plan).
const axes = [
    {
        title: "Backends",
        body: "The same solver across Vec, nalgebra, ndarray, and faer — isolating the cost of the linear-algebra layer.",
    },
    {
        title: "Solvers",
        body: "Head-to-head runs of different solvers on the same classical problems (Rosenbrock, Beale, Powell, …).",
    },
    {
        title: "Competitors",
        body: "basin against established crates such as argmin and levenberg-marquardt on matched problems.",
    },
];
</script>

<Seo
    title="basin — benchmarks"
    description="Benchmarks and solver comparisons for the basin optimization library (coming soon)."
/>

<section class="max-w-screen-2xl mx-auto px-4 md:px-8 py-16">
    <span
        class="inline-block text-xs font-mono uppercase tracking-widest text-slate-500 dark:text-slate-400"
    >
        Coming soon
    </span>
    <h1 class="mt-3 text-3xl md:text-4xl font-semibold tracking-tight">
        Benchmarks
    </h1>
    <p class="mt-4 max-w-2xl text-slate-600 dark:text-slate-300">
        This page will publish basin's benchmark results and solver comparisons
        on classical optimization problems. The benchmark harness already exists
        in the repository; wiring its results into the site is the next step.
    </p>

    <div class="mt-10 grid gap-6 sm:grid-cols-3">
        {#each axes as axis}
            <div
                class="rounded-xl border border-slate-200 dark:border-slate-800 p-5"
            >
                <h2 class="font-semibold">{axis.title}</h2>
                <p class="mt-2 text-sm text-slate-600 dark:text-slate-300">
                    {axis.body}
                </p>
            </div>
        {/each}
    </div>

    <p class="mt-10 text-sm text-slate-500 dark:text-slate-400">
        In the meantime, the <a
            class="underline decoration-dotted hover:text-slate-900 dark:hover:text-slate-100"
            href="{base}/visualizer/">visualizer</a
        >
        lets you watch solvers converge interactively.
    </p>
</section>

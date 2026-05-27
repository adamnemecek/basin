<script lang="ts">
import { base } from "$app/paths";
import BackendChart from "$lib/BackendChart.svelte";
import Seo from "$lib/Seo.svelte";
import {
    BACKEND_BENCHMARKS as data,
    BACKEND_COLORS,
    BACKEND_LABELS,
    BACKEND_ORDER,
    SOLVER_LABELS,
    SOLVER_ORDER,
    formatDuration,
    type Backend,
    type Solver,
} from "$lib/data/benchmarks";

// Distinct Rosenbrock dimensions present in the data, ascending.
const dims = [...new Set(data.results.map((r) => r.n))].sort((a, b) => a - b);

function cell(solver: Solver, n: number, backend: Backend) {
    return data.results.find(
        (r) => r.solver === solver && r.n === n && r.backend === backend,
    );
}

// Fastest (minimum) mean time in a (solver, n) row — the baseline the
// relative-speed bars and ratios are measured against.
function rowMin(solver: Solver, n: number): number {
    return Math.min(
        ...BACKEND_ORDER.map((b) => cell(solver, n, b)?.ns ?? Infinity),
    );
}

// One line per backend (time vs n) for a solver's chart.
function seriesFor(solver: Solver) {
    return BACKEND_ORDER.map((backend) => ({
        label: BACKEND_LABELS[backend],
        color: BACKEND_COLORS[backend],
        points: data.results
            .filter((r) => r.solver === solver && r.backend === backend)
            .map((r) => ({ n: r.n, ns: r.ns })),
    }));
}

// The three-axis overview. Only "Backends" has data so far.
const axes = [
    {
        title: "Backends",
        status: "Live",
        body: "The same solver across Vec, nalgebra, ndarray, and faer — isolating the cost of the linear-algebra layer.",
    },
    {
        title: "Solvers",
        status: "Planned",
        body: "Head-to-head runs of different solvers on the same classical problems (Rosenbrock, Beale, Powell, …).",
    },
    {
        title: "Competitors",
        status: "Planned",
        body: "basin against established crates such as argmin and levenberg-marquardt on matched problems.",
    },
];
</script>

<Seo
    title="Basin — benchmarks"
    description="Backend benchmarks for the Basin optimization library: the same solver across the Vec, nalgebra, ndarray, and faer linear-algebra backends."
/>

<section class="max-w-screen-2xl mx-auto px-4 md:px-8 py-16">
    <h1 class="text-3xl md:text-4xl font-semibold tracking-tight">Benchmarks</h1>
    <p class="mt-4 max-w-2xl text-slate-600 dark:text-slate-300">
        Basin's benchmark suite is built along three axes. The <strong
            >backends</strong
        >
        axis is live below; the solver and competitor axes are on the way.
    </p>

    <div class="mt-10 grid gap-6 sm:grid-cols-3">
        {#each axes as axis}
            <div
                class="rounded-xl border border-slate-200 dark:border-slate-800 p-5"
            >
                <div class="flex items-center justify-between gap-2">
                    <h2 class="font-semibold">{axis.title}</h2>
                    <span
                        class="text-xs font-mono uppercase tracking-widest {axis.status ===
                        'Live'
                            ? 'text-emerald-600 dark:text-emerald-400'
                            : 'text-slate-400 dark:text-slate-500'}"
                    >
                        {axis.status}
                    </span>
                </div>
                <p class="mt-2 text-sm text-slate-600 dark:text-slate-300">
                    {axis.body}
                </p>
            </div>
        {/each}
    </div>

    <h2 class="mt-14 text-2xl font-semibold tracking-tight">
        Backends — same solver, different linear algebra
    </h2>
    <p class="mt-3 max-w-2xl text-slate-600 dark:text-slate-300">
        Each chart runs one solver on Rosenbrock to a fixed iteration budget,
        varying only the linear-algebra backend, over a range of problem sizes
        <code class="font-mono">n</code>. Both axes are logarithmic; times are
        criterion's mean per full solve, so lower is better.
    </p>

    <!-- Shared legend for the three charts. -->
    <div class="mt-6 flex flex-wrap gap-x-5 gap-y-2 text-sm">
        {#each BACKEND_ORDER as backend}
            <span class="inline-flex items-center gap-2">
                <span
                    class="inline-block h-2.5 w-2.5 rounded-full"
                    style="background: {BACKEND_COLORS[backend]}"
                ></span>
                <span class="font-mono text-slate-600 dark:text-slate-300">
                    {BACKEND_LABELS[backend]}
                </span>
            </span>
        {/each}
    </div>

    <div class="mt-6 grid gap-6 lg:grid-cols-3">
        {#each SOLVER_ORDER as solver}
            <div
                class="rounded-xl border border-slate-200 dark:border-slate-800 p-4"
            >
                <h3 class="text-sm font-semibold">{SOLVER_LABELS[solver]}</h3>
                <div class="mt-2">
                    <BackendChart
                        series={seriesFor(solver)}
                        {dims}
                        ariaLabel={`${SOLVER_LABELS[solver]}: solve time vs problem size, one line per backend`}
                    />
                </div>
            </div>
        {/each}
    </div>

    <h3 class="mt-14 text-lg font-semibold tracking-tight">
        Exact timings
    </h3>
    <p class="mt-2 max-w-2xl text-sm text-slate-600 dark:text-slate-300">
        The same data with per-cell figures: criterion's mean and the speed
        relative to the fastest backend in each row.
    </p>

    <div class="mt-6 space-y-10">
        {#each SOLVER_ORDER as solver}
            <div
                class="rounded-xl border border-slate-200 dark:border-slate-800 p-5"
            >
                <h3 class="font-semibold">{SOLVER_LABELS[solver]}</h3>
                <div class="mt-4 overflow-x-auto">
                    <table class="w-full text-sm border-collapse">
                        <thead>
                            <tr
                                class="text-left text-slate-500 dark:text-slate-400"
                            >
                                <th class="py-2 pr-4 font-medium">n</th>
                                {#each BACKEND_ORDER as backend}
                                    <th class="py-2 px-4 font-mono font-medium">
                                        {BACKEND_LABELS[backend]}
                                    </th>
                                {/each}
                            </tr>
                        </thead>
                        <tbody>
                            {#each dims as n}
                                {@const fastest = rowMin(solver, n)}
                                <tr
                                    class="border-t border-slate-100 dark:border-slate-800/70 align-top"
                                >
                                    <td
                                        class="py-3 pr-4 font-mono text-slate-700 dark:text-slate-300"
                                        >{n}</td
                                    >
                                    {#each BACKEND_ORDER as backend}
                                        {@const c = cell(solver, n, backend)}
                                        <td
                                            class="py-3 px-4 {c && c.ns ===
                                            fastest
                                                ? 'bg-emerald-50/60 dark:bg-emerald-500/10'
                                                : ''}"
                                        >
                                            {#if c}
                                                <div
                                                    class="font-mono {c.ns ===
                                                    fastest
                                                        ? 'font-semibold text-emerald-700 dark:text-emerald-400'
                                                        : 'text-slate-700 dark:text-slate-200'}"
                                                >
                                                    {formatDuration(c.ns)}
                                                </div>
                                                <div
                                                    class="mt-1.5 h-1.5 w-full max-w-28 rounded-full bg-slate-100 dark:bg-slate-800"
                                                >
                                                    <div
                                                        class="h-full rounded-full {c.ns ===
                                                        fastest
                                                            ? 'bg-emerald-500'
                                                            : 'bg-slate-400 dark:bg-slate-500'}"
                                                        style="width: {(fastest /
                                                            c.ns) *
                                                            100}%"
                                                    ></div>
                                                </div>
                                                <div
                                                    class="mt-1 text-xs text-slate-400 dark:text-slate-500"
                                                >
                                                    {c.ns === fastest
                                                        ? "fastest"
                                                        : `${(c.ns / fastest).toFixed(2)}× slower`}
                                                </div>
                                            {:else}
                                                <span
                                                    class="text-slate-300 dark:text-slate-600"
                                                    >—</span
                                                >
                                            {/if}
                                        </td>
                                    {/each}
                                </tr>
                            {/each}
                        </tbody>
                    </table>
                </div>
            </div>
        {/each}
    </div>

    <p class="mt-8 max-w-3xl text-sm text-slate-500 dark:text-slate-400">
        Measured {data.generatedAt} on {data.env.cpu}
        ({data.env.os}/{data.env.arch}), criterion mean over a fixed
        {data.iterations}-iteration budget per solve. Bars show speed relative
        to the fastest backend in each row (longer is faster). Absolute times
        are machine-specific — compare ratios within a row, not across
        machines.
    </p>

    <p class="mt-6 text-sm text-slate-500 dark:text-slate-400">
        To watch these solvers converge interactively, try the <a
            class="underline decoration-dotted hover:text-slate-900 dark:hover:text-slate-100"
            href="{base}/visualizer/">visualizer</a
        >.
    </p>
</section>

<script lang="ts">
import { base } from "$app/paths";
import BackendChart from "$lib/BackendChart.svelte";
import Seo from "$lib/Seo.svelte";
import {
    BACKEND_BENCHMARKS as data,
    BACKEND_COLORS,
    BACKEND_LABELS,
    BACKEND_ORDER,
    CASES,
    PROBLEM_LABELS,
    SOLVER_LABELS,
    backendsFor,
    type Solver,
} from "$lib/data/benchmarks";

// Distinct dimensions present for a (solver, problem) case, ascending.
function dimsFor(solver: Solver, problem: string): number[] {
    return [
        ...new Set(
            data.results
                .filter((r) => r.solver === solver && r.problem === problem)
                .map((r) => r.n),
        ),
    ].sort((a, b) => a - b);
}

// One line per backend present (time vs n) for a case's chart. Backends with
// no data for the case (the intentional coverage gaps) simply don't appear.
function seriesFor(solver: Solver, problem: string) {
    return backendsFor(solver, problem).map((backend) => ({
        label: BACKEND_LABELS[backend],
        color: BACKEND_COLORS[backend],
        points: data.results
            .filter(
                (r) =>
                    r.solver === solver &&
                    r.problem === problem &&
                    r.backend === backend,
            )
            .map((r) => ({ n: r.n, ns: r.ns })),
    }));
}

// The three-axis overview. Only "Backends" has data so far.
const axes = [
    {
        title: "Backends",
        status: "Live",
        body: "A curated set of solver + problem pairs across Vec, nalgebra, ndarray, and faer — isolating the cost of the linear-algebra layer, and showing where a backend can't run a solver at all.",
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
    description="Backend benchmarks for the Basin optimization library: a curated set of solver and problem pairs across the Vec, nalgebra, ndarray, and faer linear-algebra backends."
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
    <p class="mt-3 max-w-3xl text-slate-600 dark:text-slate-300">
        A curated set of (solver, problem) cases, each run to a fixed iteration
        budget varying only the linear-algebra backend. Scaling cases plot time
        against problem size <code class="font-mono">n</code> on log–log axes;
        fixed-size cases show one bar per backend. As a solver needs richer
        linear algebra, fewer backends can run it — those gaps are intentional,
        and differ in cause. Times are criterion's mean per full solve, so
        lower is better.
    </p>

    <!-- Shared legend for the charts and bars; the full backend palette. -->
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

    <div class="mt-6 grid gap-6 lg:grid-cols-2">
        {#each CASES as c}
            <div
                class="rounded-xl border border-slate-200 dark:border-slate-800 p-5"
            >
                <h3 class="text-sm font-semibold">
                    {SOLVER_LABELS[c.solver]}
                    <span class="text-slate-400 dark:text-slate-500">·</span>
                    {PROBLEM_LABELS[c.problem]}
                </h3>
                <p class="mt-1 text-xs text-slate-500 dark:text-slate-400">
                    {c.blurb}
                </p>
                <div class="mt-3">
                    <BackendChart
                        series={seriesFor(c.solver, c.problem)}
                        dims={dimsFor(c.solver, c.problem)}
                        ariaLabel={`${SOLVER_LABELS[c.solver]} on ${PROBLEM_LABELS[c.problem]}: solve time vs problem size, one line per backend`}
                    />
                </div>
            </div>
        {/each}
    </div>

    <p class="mt-8 max-w-3xl text-sm text-slate-500 dark:text-slate-400">
        Measured {data.generatedAt} on {data.env.cpu}
        ({data.env.os}/{data.env.arch}), criterion mean per solve over a fixed
        {data.iterations}-iteration budget (a cap — the least-squares and
        CMA-ES cases converge sooner). Both axes are logarithmic. Absolute
        times are machine-specific — compare the spread between backends within
        a chart, not across machines.
    </p>

    <p class="mt-6 text-sm text-slate-500 dark:text-slate-400">
        To watch these solvers converge interactively, try the <a
            class="underline decoration-dotted hover:text-slate-900 dark:hover:text-slate-100"
            href="{base}/visualizer/">visualizer</a
        >.
    </p>
</section>

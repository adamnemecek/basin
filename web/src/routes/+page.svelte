<script lang="ts">
import { base } from "$app/paths";

// Kept as a string so Svelte doesn't try to parse the braces in the
// Rust code as template expressions.
const code = `use basin::{BasicState, CostFunction, Executor, Gradient, GradientDescent};

struct Rosenbrock;

impl CostFunction for Rosenbrock {
    type Param = Vec<f64>;
    type Output = f64;
    fn cost(&self, x: &Vec<f64>) -> f64 {
        (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2)
    }
}

impl Gradient for Rosenbrock {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Vec<f64> {
        vec![
            -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0].powi(2)),
            200.0 * (x[1] - x[0].powi(2)),
        ]
    }
}

let result = Executor::new(Rosenbrock, GradientDescent::new(1e-3), BasicState::new(vec![-1.2, 1.0]))
    .max_iter(50_000)
    .run();

println!("x = {:?}, f = {}", result.param(), result.cost());`;

const features = [
    {
        title: "Pluggable solvers",
        body: "Gradient descent, Nelder–Mead, L-BFGS / L-BFGS-B, Gauss–Newton, Levenberg–Marquardt, CMA-ES and more — driven by one shared executor loop.",
    },
    {
        title: "Multiple backends",
        body: "Run on plain Vec<f64>, nalgebra, ndarray, or faer. Each backend sits behind a single feature — no per-version feature explosion.",
    },
    {
        title: "First-class constraints",
        body: "Box bounds are part of the problem and enforced at the type level: handing a constrained problem to an unconstrained solver is a compile error.",
    },
    {
        title: "Composable termination",
        body: "Gradient, parameter, and cost tolerances, iteration and time budgets — configured uniformly across solvers, bound to the state each one exposes.",
    },
    {
        title: "Runs in the browser",
        body: "wasm-first by design: the default build pulls in no BLAS/LAPACK or threads, so basin compiles to wasm32 out of the box.",
    },
    {
        title: "Paper-anchored",
        body: "Solvers track published algorithms (Nocedal’s L-BFGS-B, Nielsen’s LM damping, Hansen’s CMA-ES) rather than ad-hoc variants.",
    },
];
</script>

<svelte:head>
    <title>basin — numerical optimization for Rust</title>
    <meta
        name="description"
        content="basin is a numerical optimization library for Rust: pluggable solvers, multiple linear-algebra backends, first-class constraints, and a wasm-first design."
    />
</svelte:head>

<!-- Hero -->
<section class="max-w-screen-2xl mx-auto px-4 md:px-8 pt-16 pb-12 md:pt-24 md:pb-16 flex gap-6">
    <div class="max-w-3xl flex-col">
        <span
            class="inline-block text-xs font-mono uppercase tracking-widest text-slate-500 dark:text-slate-400"
        >
            Alpha · Rust
        </span>
        <h1 class="mt-3 text-4xl md:text-6xl font-semibold tracking-tight text-balance">
            Numerical Optimization in Rust
        </h1>
        <p class="mt-5 text-lg md:text-xl text-slate-600 dark:text-slate-300 text-pretty">
            <span class="font-semibold">Basin</span> is a solver framework with
            a generic executor loop over pluggable solvers, multiple
            linear-algebra backends, first-class constraints, and a wasm-first
            design.
        </p>
        <div class="mt-8 flex flex-wrap gap-3">
            <a
                href="{base}/docs/getting-started/"
                class="px-5 py-2.5 rounded-lg bg-slate-900 text-white font-medium hover:bg-slate-700 dark:bg-slate-100 dark:text-slate-900 dark:hover:bg-white transition-colors"
            >
                Get started
            </a>
            <a
                href="{base}/visualizer/"
                class="px-5 py-2.5 rounded-lg border border-slate-300 dark:border-slate-700 font-medium hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
            >
                Open the visualizer
            </a>
            <a
                href="https://github.com/jolars/basin"
                target="_blank"
                rel="noreferrer"
                class="px-5 py-2.5 rounded-lg border border-slate-300 dark:border-slate-700 font-medium hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
            >
                GitHub
            </a>
        </div>
    </div>
    <div class="flex flex-col">
        <img
            src="{base}/logo.svg"
            alt="Visualization of optimization trajectories on the Rosenbrock function, a common test problem in optimization."
            class="mt-10 w-full object-cover">
    </div>
</section>

<!-- Quick taste -->
<section class="max-w-screen-2xl mx-auto px-4 md:px-8 pb-16">
    <div class="grid lg:grid-cols-[1fr_1.4fr] gap-8 items-start">
        <div>
            <h2 class="text-2xl font-semibold tracking-tight">A minimal solve</h2>
            <p class="mt-3 text-slate-600 dark:text-slate-300">
                Implement <code class="font-mono text-sm">CostFunction</code> (and
                <code class="font-mono text-sm">Gradient</code>, when your solver
                needs it), pick a solver, and hand both to the
                <code class="font-mono text-sm">Executor</code>. The same loop drives
                every solver in the library.
            </p>
            <p class="mt-3 text-slate-600 dark:text-slate-300">
                Want to see it move? The
                <a
                    class="underline decoration-dotted hover:text-slate-900 dark:hover:text-slate-100"
                    href="{base}/visualizer/">visualizer</a
                >
                animates these trajectories live, compiled to wasm.
            </p>
        </div>
        <div
            class="rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-900 overflow-hidden"
        >
            <div
                class="px-4 py-2 border-b border-slate-200 dark:border-slate-800 text-xs font-mono text-slate-500 dark:text-slate-400"
            >
                rosenbrock.rs
            </div>
            <pre
                class="p-4 overflow-x-auto text-sm leading-relaxed"><code class="font-mono">{code}</code></pre>
        </div>
    </div>
</section>

<!-- Features -->
<section
    class="border-t border-slate-200 dark:border-slate-800 bg-slate-50/60 dark:bg-slate-900/40"
>
    <div class="max-w-screen-2xl mx-auto px-4 md:px-8 py-16">
        <h2 class="text-2xl font-semibold tracking-tight">What's in the box</h2>
        <div class="mt-8 grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
            {#each features as f}
                <div
                    class="rounded-xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950 p-5"
                >
                    <h3 class="font-semibold">{f.title}</h3>
                    <p class="mt-2 text-sm text-slate-600 dark:text-slate-300">
                        {f.body}
                    </p>
                </div>
            {/each}
        </div>
    </div>
</section>

<!-- Closing CTA -->
<section class="max-w-screen-2xl mx-auto px-4 md:px-8 py-16">
    <div
        class="rounded-2xl border border-slate-200 dark:border-slate-800 p-8 md:p-12 flex flex-wrap items-center justify-between gap-6"
    >
        <div>
            <h2 class="text-2xl font-semibold tracking-tight">
                Compare solvers on classical problems
            </h2>
            <p class="mt-2 text-slate-600 dark:text-slate-300 max-w-xl">
                Benchmarks and head-to-head solver comparisons are on the way.
                In the meantime, explore the docs or watch solvers converge in
                the visualizer.
            </p>
        </div>
        <div class="flex flex-wrap gap-3">
            <a
                href="{base}/docs/getting-started/"
                class="px-5 py-2.5 rounded-lg bg-slate-900 text-white font-medium hover:bg-slate-700 dark:bg-slate-100 dark:text-slate-900 dark:hover:bg-white transition-colors"
            >
                Read the docs
            </a>
            <a
                href="{base}/benchmarks/"
                class="px-5 py-2.5 rounded-lg border border-slate-300 dark:border-slate-700 font-medium hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
            >
                Benchmarks
            </a>
        </div>
    </div>
</section>

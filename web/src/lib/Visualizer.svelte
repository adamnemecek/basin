<script lang="ts">
    import { onMount } from 'svelte';
    import init, {
        ProblemKind,
        SolverKind,
        Run,
        evalGrid,
    } from '$lib/basin-wasm/basin_wasm';
    import { PROBLEMS, problemByKind } from '$lib/problems';
    import { SOLVERS } from '$lib/solvers';
    import ContourPlot from '$lib/ContourPlot.svelte';
    import CostChart from '$lib/CostChart.svelte';
    import Controls from '$lib/Controls.svelte';
    import { theme } from '$lib/theme.svelte';

    // Wasm boot. The viz waits on this once; everything downstream assumes
    // the module is already loaded.
    let wasmReady = $state(false);

    let problemKind: ProblemKind = $state(ProblemKind.Rosenbrock);
    let solverKind: SolverKind = $state(SolverKind.GradientDescentConstant);
    let gdAlpha = $state(problemByKind(ProblemKind.Rosenbrock).gdAlphaDefault);
    let maxIter = $state(500);
    let startPoint = $state({ x: -1.5, y: 2.0 });

    // Heatmap grid. Recomputed when the problem changes (or on first boot).
    const GRID_N = 192;
    // Use the wide `Float64Array<ArrayBufferLike>` type so values returned
    // from wasm-bindgen (which use `ArrayBufferLike`) assign cleanly.
    let grid: Float64Array<ArrayBufferLike> = $state(
        new Float64Array(GRID_N * GRID_N),
    );

    // Animated trajectory and cost log fed to the children.
    let trajectory: Float64Array<ArrayBufferLike> = $state(new Float64Array(0));
    let costs: Float64Array<ArrayBufferLike> = $state(new Float64Array(0));
    let reason = $state('');

    let problemMeta = $derived(problemByKind(problemKind));
    let solverMeta = $derived(SOLVERS.find((s) => s.kind === solverKind)!);

    // Plain (non-reactive) handles for the in-flight run + animation
    // frame. We deliberately keep these out of `$state` because the run
    // effect both reads (cleanup) and writes (assignment) them, and a
    // reactive write would re-trigger the effect — Svelte detects that
    // as `effect_update_depth_exceeded` and aborts.
    let activeRun: Run | null = null;
    let frameId: number | null = null;

    // Refresh the heatmap when the problem changes.
    $effect(() => {
        if (!wasmReady) return;
        const d = problemMeta.domain;
        grid = evalGrid(
            problemMeta.kind,
            d.xmin,
            d.xmax,
            d.ymin,
            d.ymax,
            GRID_N,
            GRID_N,
        );
    });

    // Boot a fresh run whenever the inputs change. The reads inside
    // `new Run(...)` track the dependencies; writes to `activeRun` and
    // `frameId` are non-reactive so they don't retrigger this effect.
    $effect(() => {
        if (!wasmReady) return;
        const pk = problemKind;
        const sk = solverKind;
        const a = gdAlpha;
        const mi = maxIter;
        const sx = startPoint.x;
        const sy = startPoint.y;

        if (frameId !== null) {
            cancelAnimationFrame(frameId);
            frameId = null;
        }
        if (activeRun !== null) {
            activeRun.free();
            activeRun = null;
        }

        const run = new Run(pk, sk, sx, sy, a, mi);
        activeRun = run;
        trajectory = run.trajectoryXy();
        costs = run.costs();
        reason = '';

        const tick = () => {
            // Stale-frame guard: a newer effect run replaces `activeRun`,
            // so a tick from an older closure should bail.
            if (run !== activeRun) return;
            const result = run.stepMany(8) as {
                done: boolean;
                iters_added: number;
                reason?: string | null;
            };
            trajectory = run.trajectoryXy();
            costs = run.costs();
            if (result.done) {
                reason = result.reason ?? '';
                frameId = null;
                return;
            }
            frameId = requestAnimationFrame(tick);
        };
        frameId = requestAnimationFrame(tick);

        return () => {
            if (frameId !== null) {
                cancelAnimationFrame(frameId);
                frameId = null;
            }
            if (activeRun === run) {
                run.free();
                activeRun = null;
            }
        };
    });

    onMount(async () => {
        await init();
        wasmReady = true;
        // Seed start point near a visually interesting corner of the
        // initial problem (Rosenbrock).
        startPoint = { x: -1.5, y: 2.0 };
    });

    function handlePick(p: { x: number; y: number }) {
        startPoint = p;
    }

    function handleControlChange(patch: {
        problemKind?: ProblemKind;
        solverKind?: SolverKind;
        gdAlpha?: number;
        maxIter?: number;
    }) {
        if (patch.problemKind !== undefined && patch.problemKind !== problemKind) {
            problemKind = patch.problemKind;
            // Re-center start and reset α default for the new problem.
            const d = problemByKind(problemKind).domain;
            startPoint = {
                x: d.xmin + 0.25 * (d.xmax - d.xmin),
                y: d.ymin + 0.75 * (d.ymax - d.ymin),
            };
            gdAlpha = problemByKind(problemKind).gdAlphaDefault;
        }
        if (patch.solverKind !== undefined) solverKind = patch.solverKind;
        if (patch.gdAlpha !== undefined) gdAlpha = patch.gdAlpha;
        if (patch.maxIter !== undefined) maxIter = patch.maxIter;
    }
</script>

<section
    class="min-h-[calc(100vh-8rem)] max-w-screen-2xl w-full mx-auto px-4 md:px-8 py-6 flex flex-col gap-6"
>
    <header class="flex flex-wrap items-start justify-between gap-4">
        <div>
            <h1 class="text-2xl md:text-3xl font-semibold tracking-tight">
                Solver visualizer
            </h1>
            <p class="text-slate-600 dark:text-slate-400 text-sm mt-1">
                Live wasm-driven 2D trajectories. Click on the contour to reset
                the start point.
            </p>
        </div>
        <p
            class="text-xs text-slate-500 dark:text-slate-500 font-mono hidden md:block self-center"
        >
            {solverMeta.blurb}
        </p>
    </header>

    {#if !wasmReady}
        <p class="text-slate-500 dark:text-slate-400">Loading wasm…</p>
    {:else}
        <div class="grid grid-cols-1 lg:grid-cols-[2fr_1fr] gap-6 flex-1 min-h-0">
            <div
                class="relative bg-slate-100 dark:bg-slate-900 rounded-lg overflow-hidden aspect-square lg:aspect-auto lg:min-h-[360px]"
            >
                <ContourPlot
                    problem={problemMeta}
                    {grid}
                    nx={GRID_N}
                    ny={GRID_N}
                    {trajectory}
                    {startPoint}
                    theme={theme.effective}
                    onPick={handlePick}
                />
            </div>
            <aside class="flex flex-col gap-6 min-w-0">
                <div class="bg-slate-100 dark:bg-slate-900 rounded-lg p-4">
                    <Controls
                        {problemKind}
                        {solverKind}
                        {gdAlpha}
                        {maxIter}
                        {startPoint}
                        usesAlpha={solverMeta.usesAlpha}
                        onChange={handleControlChange}
                    />
                </div>
                <div
                    class="bg-slate-100 dark:bg-slate-900 rounded-lg p-3 h-56 lg:flex-1"
                >
                    <CostChart {costs} {reason} theme={theme.effective} />
                </div>
            </aside>
        </div>
    {/if}
</section>

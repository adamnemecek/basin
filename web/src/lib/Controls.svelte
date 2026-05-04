<script lang="ts">
    import { ProblemKind, SolverKind } from './basin-wasm/basin_wasm';
    import { PROBLEMS } from './problems';
    import { SOLVERS } from './solvers';

    type Props = {
        problemKind: ProblemKind;
        solverKind: SolverKind;
        gdAlpha: number;
        maxIter: number;
        startPoint: { x: number; y: number };
        usesAlpha: boolean;
        onChange: (patch: {
            problemKind?: ProblemKind;
            solverKind?: SolverKind;
            gdAlpha?: number;
            maxIter?: number;
        }) => void;
    };

    let {
        problemKind,
        solverKind,
        gdAlpha,
        maxIter,
        startPoint,
        usesAlpha,
        onChange,
    }: Props = $props();
</script>

<div class="flex flex-col gap-4 text-sm">
    <label class="flex flex-col gap-1">
        <span class="text-slate-700 dark:text-slate-300 uppercase text-xs tracking-wide"
            >Problem</span
        >
        <select
            class="bg-white text-slate-900 border border-slate-300 dark:bg-slate-800 dark:text-slate-100 dark:border-slate-700 rounded px-2 py-1"
            value={problemKind}
            onchange={(e) =>
                onChange({
                    problemKind: Number(
                        (e.currentTarget as HTMLSelectElement).value,
                    ) as ProblemKind,
                })}
        >
            {#each PROBLEMS as p}
                <option value={p.kind}>{p.label}</option>
            {/each}
        </select>
    </label>

    <label class="flex flex-col gap-1">
        <span class="text-slate-700 dark:text-slate-300 uppercase text-xs tracking-wide"
            >Solver</span
        >
        <select
            class="bg-white text-slate-900 border border-slate-300 dark:bg-slate-800 dark:text-slate-100 dark:border-slate-700 rounded px-2 py-1"
            value={solverKind}
            onchange={(e) =>
                onChange({
                    solverKind: Number(
                        (e.currentTarget as HTMLSelectElement).value,
                    ) as SolverKind,
                })}
        >
            {#each SOLVERS as s}
                <option value={s.kind}>{s.label}</option>
            {/each}
        </select>
    </label>

    {#if usesAlpha}
        <label class="flex flex-col gap-1">
            <span class="text-slate-700 dark:text-slate-300 uppercase text-xs tracking-wide"
                >Step size α: <span class="font-mono text-slate-900 dark:text-slate-100"
                    >{gdAlpha.toExponential(2)}</span
                ></span
            >
            <input
                type="range"
                min="-5"
                max="0"
                step="0.05"
                value={Math.log10(gdAlpha)}
                oninput={(e) =>
                    onChange({
                        gdAlpha: Math.pow(
                            10,
                            Number((e.currentTarget as HTMLInputElement).value),
                        ),
                    })}
            />
        </label>
    {/if}

    <label class="flex flex-col gap-1">
        <span class="text-slate-700 dark:text-slate-300 uppercase text-xs tracking-wide"
            >Max iterations: <span class="font-mono text-slate-900 dark:text-slate-100"
                >{maxIter}</span
            ></span
        >
        <input
            type="range"
            min="20"
            max="2000"
            step="20"
            value={maxIter}
            oninput={(e) =>
                onChange({
                    maxIter: Number(
                        (e.currentTarget as HTMLInputElement).value,
                    ),
                })}
        />
    </label>

    <p class="text-slate-600 dark:text-slate-400 text-xs leading-relaxed">
        Click anywhere on the contour plot to reset the starting point. The
        solver re-runs immediately. Current start: <span
            class="font-mono text-slate-800 dark:text-slate-200"
            >({startPoint.x.toFixed(2)}, {startPoint.y.toFixed(2)})</span
        >
    </p>
</div>

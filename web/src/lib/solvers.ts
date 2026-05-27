import { SolverKind } from './basin-wasm/basin_wasm';

/**
 * A single solver-specific control, rendered generically by `Controls`.
 *
 * The `id` is the key the value is stored under (in the visualizer's
 * `optionValues` record) and the field name passed across the wasm
 * boundary inside the `Run` options object — keep these in sync with
 * `RunOptions` in `crates/basin-wasm/src/lib.rs` (camelCase).
 */
export type SolverOption =
    | {
          id: string;
          kind: 'select';
          label: string;
          choices: { value: string; label: string }[];
          default: string;
      }
    | {
          id: string;
          kind: 'logSlider';
          label: string;
          /** Slider bounds are in log10 space (the stored value is 10^slider). */
          min: number;
          max: number;
          step: number;
          default: number;
          /** Only show this control when another option currently equals a value. */
          showIf?: { id: string; equals: string };
      }
    | {
          id: string;
          kind: 'intSlider';
          label: string;
          min: number;
          max: number;
          step: number;
          default: number;
      };

export type SolverMeta = {
    kind: SolverKind;
    label: string;
    /** Short description shown in the UI. */
    blurb: string;
    /** Solver-specific controls, rendered in order by `Controls`. */
    options: SolverOption[];
};

export const SOLVERS: SolverMeta[] = [
    {
        kind: SolverKind.GradientDescent,
        label: 'Gradient Descent',
        blurb: 'Steepest descent with a fixed step or Armijo backtracking.',
        options: [
            {
                id: 'gdLineSearch',
                kind: 'select',
                label: 'Step strategy',
                choices: [
                    { value: 'constant', label: 'Constant α' },
                    { value: 'backtracking', label: 'Backtracking' },
                ],
                default: 'constant',
            },
            {
                id: 'gdAlpha',
                kind: 'logSlider',
                label: 'Step size α',
                min: -5,
                max: 0,
                step: 0.05,
                // Overridden per-problem by the visualizer (gdAlphaDefault).
                default: 0.01,
                showIf: { id: 'gdLineSearch', equals: 'constant' },
            },
        ],
    },
    {
        kind: SolverKind.NelderMead,
        label: 'Nelder–Mead (simplex, derivative-free)',
        blurb: 'Standard reflection / expansion / contraction simplex.',
        options: [],
    },
    {
        kind: SolverKind.Lbfgs,
        label: 'L-BFGS (limited-memory quasi-Newton)',
        blurb: 'Two-loop recursion with a Moré–Thuente line search.',
        options: [
            {
                id: 'lbfgsM',
                kind: 'intSlider',
                label: 'History size m',
                min: 1,
                max: 20,
                step: 1,
                default: 10,
            },
        ],
    },
];

/** Default option values for a solver, keyed by option id. */
export function defaultOptionValues(
    meta: SolverMeta,
): Record<string, string | number> {
    const out: Record<string, string | number> = {};
    for (const opt of meta.options) out[opt.id] = opt.default;
    return out;
}

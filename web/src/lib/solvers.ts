import { SolverKind } from './basin-wasm/basin_wasm';

export type SolverMeta = {
    kind: SolverKind;
    label: string;
    /** Short description shown in the UI. */
    blurb: string;
    /** Whether the constant step size slider is meaningful. */
    usesAlpha: boolean;
};

export const SOLVERS: SolverMeta[] = [
    {
        kind: SolverKind.GradientDescentConstant,
        label: 'Gradient Descent (constant step)',
        blurb: 'Steepest descent with a fixed step α.',
        usesAlpha: true,
    },
    {
        kind: SolverKind.GradientDescentBacktracking,
        label: 'Gradient Descent (backtracking)',
        blurb: 'Steepest descent with Armijo backtracking line search.',
        usesAlpha: false,
    },
    {
        kind: SolverKind.NelderMead,
        label: 'Nelder–Mead (simplex, derivative-free)',
        blurb: 'Standard reflection / expansion / contraction simplex.',
        usesAlpha: false,
    },
];

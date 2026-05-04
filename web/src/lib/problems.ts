/**
 * Per-problem metadata for the visualizer.
 *
 * Mirrors the corpus exposed by `basin::problems` (Sphere, Rosenbrock,
 * Beale, Booth) — but only the bits the UI needs: a viewing window
 * for the contour plot, a known minimum to mark, and how to compress
 * the cost dynamic range when colorizing.
 *
 * Why duplicate this from Rust instead of pulling from `basin-wasm`?
 * The numbers are tiny, never change in lockstep with the algorithm,
 * and pulling them through wasm would add boilerplate for no win. If
 * the corpus grows large enough that this drifts, we can revisit.
 */
import { ProblemKind } from './basin-wasm/basin_wasm';

export type Domain = { xmin: number; xmax: number; ymin: number; ymax: number };

export type ProblemMeta = {
    kind: ProblemKind;
    label: string;
    /** Recommended viewing window for the contour plot. */
    domain: Domain;
    /** Known global minimum for the marker. */
    minimum: { x: number; y: number };
    /**
     * Heatmap intensity transform. Rosenbrock and Beale have huge dynamic
     * range; a log-ish squash keeps the basin visible. Sphere and Booth
     * are mild quadratics where a square-root squash already looks fine.
     */
    intensity: 'linear' | 'sqrt' | 'log1p';
    /** Sensible default constant step size for `GradientDescentConstant`. */
    gdAlphaDefault: number;
};

export const PROBLEMS: ProblemMeta[] = [
    {
        kind: ProblemKind.Sphere,
        label: 'Sphere',
        domain: { xmin: -3, xmax: 3, ymin: -3, ymax: 3 },
        minimum: { x: 0, y: 0 },
        intensity: 'sqrt',
        gdAlphaDefault: 0.2,
    },
    {
        kind: ProblemKind.Rosenbrock,
        label: 'Rosenbrock',
        domain: { xmin: -2, xmax: 2, ymin: -1, ymax: 3 },
        minimum: { x: 1, y: 1 },
        intensity: 'log1p',
        gdAlphaDefault: 0.001,
    },
    {
        kind: ProblemKind.Beale,
        label: 'Beale',
        domain: { xmin: -4.5, xmax: 4.5, ymin: -4.5, ymax: 4.5 },
        minimum: { x: 3, y: 0.5 },
        intensity: 'log1p',
        gdAlphaDefault: 0.001,
    },
    {
        kind: ProblemKind.Booth,
        label: 'Booth',
        domain: { xmin: -10, xmax: 10, ymin: -10, ymax: 10 },
        minimum: { x: 1, y: 3 },
        intensity: 'sqrt',
        gdAlphaDefault: 0.02,
    },
];

export function problemByKind(kind: ProblemKind): ProblemMeta {
    const m = PROBLEMS.find((p) => p.kind === kind);
    if (!m) throw new Error(`unknown problem kind: ${kind}`);
    return m;
}

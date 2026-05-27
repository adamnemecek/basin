/**
 * Collect the backend-axis benchmark results into a committed JSON file the
 * `/benchmarks` page imports.
 *
 * Reads criterion's output under `target/criterion/`, keeps only the
 * `{gd,nm,lbfgs}_rosenbrock_n*` groups produced by the `solver_backends`
 * bench (ignoring `lm_*` / competitor groups), and writes the headline
 * timings to `web/src/lib/data/backend-benchmarks.json`.
 *
 * Run with: `npm run collect:benchmarks` (uses tsx). Run the bench first:
 *   cargo bench --features nalgebra,ndarray,faer --bench solver_backends
 *
 * The pipeline is deliberately off CI — timings are machine-specific and
 * shared runners are noisy. Refresh locally and commit the regenerated JSON.
 */
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { cpus } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = fileURLToPath(new URL('.', import.meta.url));
const repoRoot = resolve(scriptDir, '..', '..');
const criterionDir = join(repoRoot, 'target', 'criterion');
const outFile = resolve(scriptDir, '..', 'src', 'lib', 'data', 'backend-benchmarks.json');

/** Iteration budget the bench runs each solve for (`MAX_ITERS`). */
const ITERATIONS = 200;

const SOLVER_ORDER = ['gd', 'nm', 'lbfgs'] as const;
const BACKEND_ORDER = ['vec', 'nalgebra', 'ndarray', 'faer'] as const;

type Solver = (typeof SOLVER_ORDER)[number];
type Backend = (typeof BACKEND_ORDER)[number];

type BenchResult = {
    solver: Solver;
    problem: string;
    n: number;
    backend: Backend;
    /** Mean estimate, nanoseconds. */
    ns: number;
    /** 95% confidence interval on the mean, nanoseconds. */
    lowNs: number;
    highNs: number;
};

if (!existsSync(criterionDir)) {
    console.error(`✗ no criterion output at ${criterionDir}`);
    console.error(
        '  run the bench first:\n' +
            '  cargo bench --features nalgebra,ndarray,faer --bench solver_backends',
    );
    process.exit(1);
}

/** Recursively collect every `new/estimates.json` path under `dir`. */
function findEstimates(dir: string, out: string[] = []): string[] {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
        const full = join(dir, entry.name);
        if (entry.isDirectory()) {
            findEstimates(full, out);
        } else if (entry.name === 'estimates.json' && basename(dir) === 'new') {
            out.push(full);
        }
    }
    return out;
}

const GROUP_RE = /^(gd|nm|lbfgs)_(rosenbrock)_n(\d+)$/;

const results: BenchResult[] = [];

for (const estimatesPath of findEstimates(criterionDir)) {
    const dir = dirname(estimatesPath);
    const benchPath = join(dir, 'benchmark.json');
    if (!existsSync(benchPath)) continue;

    const bench = JSON.parse(readFileSync(benchPath, 'utf8')) as {
        group_id: string;
        function_id: string | null;
        value_str: string | null;
    };

    const match = GROUP_RE.exec(bench.group_id);
    if (!match) continue; // ignore lm_* / competitor groups

    const backend = (bench.value_str ?? bench.function_id ?? '') as Backend;
    if (!BACKEND_ORDER.includes(backend)) continue;

    const est = JSON.parse(readFileSync(estimatesPath, 'utf8')) as {
        mean: {
            point_estimate: number;
            confidence_interval: { lower_bound: number; upper_bound: number };
        };
    };

    results.push({
        solver: match[1] as Solver,
        problem: match[2],
        n: Number(match[3]),
        backend,
        ns: est.mean.point_estimate,
        lowNs: est.mean.confidence_interval.lower_bound,
        highNs: est.mean.confidence_interval.upper_bound,
    });
}

if (results.length === 0) {
    console.error(
        '✗ found criterion output but no solver_backends groups — run:\n' +
            '  cargo bench --features nalgebra,ndarray,faer --bench solver_backends',
    );
    process.exit(1);
}

// Deterministic order so the committed JSON has a stable diff.
results.sort(
    (a, b) =>
        SOLVER_ORDER.indexOf(a.solver) - SOLVER_ORDER.indexOf(b.solver) ||
        a.n - b.n ||
        BACKEND_ORDER.indexOf(a.backend) - BACKEND_ORDER.indexOf(b.backend),
);

const data = {
    generatedAt: new Date().toISOString().slice(0, 10),
    env: {
        os: process.platform,
        arch: process.arch,
        cpu: cpus()[0]?.model.trim() ?? 'unknown',
    },
    iterations: ITERATIONS,
    results,
};

mkdirSync(dirname(outFile), { recursive: true });
writeFileSync(outFile, `${JSON.stringify(data, null, 2)}\n`);

const expected = SOLVER_ORDER.length * 3 * BACKEND_ORDER.length; // solvers × dims × backends
console.log(`✓ wrote ${results.length} result(s) to ${outFile}`);
if (results.length !== expected) {
    console.warn(
        `  note: expected ${expected} (3 solvers × 3 dims × 4 backends); ` +
            'is the bench run complete?',
    );
}

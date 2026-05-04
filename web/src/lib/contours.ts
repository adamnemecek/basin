/**
 * Marching squares: extract iso-contour line segments from a regular
 * `nx × ny` scalar grid sampled on `[xmin, xmax] × [ymin, ymax]`.
 *
 * Returns a flat array of segment endpoints in *data* coordinates:
 * `[x0, y0, x1, y1, x0, y0, x1, y1, ...]`. Drawing them as `moveTo` /
 * `lineTo` pairs gives the iso-contour at value `level`.
 *
 * The implementation is the standard 16-case marching-squares table
 * with linear interpolation along edges. Saddle ambiguity (cases 5 and
 * 10) is resolved by comparing the cell's average value to `level` —
 * the simplest disambiguation that doesn't introduce visible artifacts
 * for smooth functions.
 *
 * Grid layout assumption: `grid[j * nx + i]` is the value at
 * `(xmin + i*dx, ymin + j*dy)` with `j = 0` at `ymin` (i.e. the same
 * convention basin-wasm's `evalGrid` uses).
 */
export function isoContour(
    grid: Float64Array | ArrayLike<number>,
    nx: number,
    ny: number,
    xmin: number,
    xmax: number,
    ymin: number,
    ymax: number,
    level: number,
): Float64Array {
    const out: number[] = [];
    const dx = (xmax - xmin) / (nx - 1);
    const dy = (ymax - ymin) / (ny - 1);
    for (let j = 0; j < ny - 1; j++) {
        const y0 = ymin + dy * j;
        const y1 = y0 + dy;
        for (let i = 0; i < nx - 1; i++) {
            const x0 = xmin + dx * i;
            const x1 = x0 + dx;
            // Corner values, clockwise from bottom-left.
            const v00 = grid[j * nx + i]; // bottom-left
            const v10 = grid[j * nx + (i + 1)]; // bottom-right
            const v11 = grid[(j + 1) * nx + (i + 1)]; // top-right
            const v01 = grid[(j + 1) * nx + i]; // top-left
            if (
                !Number.isFinite(v00) ||
                !Number.isFinite(v10) ||
                !Number.isFinite(v11) ||
                !Number.isFinite(v01)
            )
                continue;
            // 4-bit code: bit 0 = bottom-left, bit 1 = bottom-right,
            // bit 2 = top-right, bit 3 = top-left.
            let code = 0;
            if (v00 >= level) code |= 1;
            if (v10 >= level) code |= 2;
            if (v11 >= level) code |= 4;
            if (v01 >= level) code |= 8;
            if (code === 0 || code === 15) continue;

            // Edge crossings (linear interpolation of `level` along the
            // edge between two corners with values `a` and `b`).
            const ex = (a: number, b: number) => (level - a) / (b - a);
            // Bottom edge: between v00 and v10 → varies in x.
            const eb = () => [x0 + ex(v00, v10) * dx, y0];
            // Right edge: between v10 and v11 → varies in y.
            const er = () => [x1, y0 + ex(v10, v11) * dy];
            // Top edge: between v01 and v11 → varies in x.
            const et = () => [x0 + ex(v01, v11) * dx, y1];
            // Left edge: between v00 and v01 → varies in y.
            const el = () => [x0, y0 + ex(v00, v01) * dy];

            const push = (a: number[], b: number[]) => {
                out.push(a[0], a[1], b[0], b[1]);
            };

            switch (code) {
                case 1:
                case 14:
                    push(el(), eb());
                    break;
                case 2:
                case 13:
                    push(eb(), er());
                    break;
                case 3:
                case 12:
                    push(el(), er());
                    break;
                case 4:
                case 11:
                    push(er(), et());
                    break;
                case 6:
                case 9:
                    push(eb(), et());
                    break;
                case 7:
                case 8:
                    push(el(), et());
                    break;
                case 5: {
                    const avg = (v00 + v10 + v11 + v01) * 0.25;
                    if (avg >= level) {
                        push(el(), et());
                        push(eb(), er());
                    } else {
                        push(el(), eb());
                        push(er(), et());
                    }
                    break;
                }
                case 10: {
                    const avg = (v00 + v10 + v11 + v01) * 0.25;
                    if (avg >= level) {
                        push(el(), eb());
                        push(er(), et());
                    } else {
                        push(el(), et());
                        push(eb(), er());
                    }
                    break;
                }
            }
        }
    }
    return Float64Array.from(out);
}

/**
 * Pick `n` iso-levels that visually space the contours by the same
 * intensity transform used for the (now retired) heatmap. Spacing the
 * levels in transform-space — `sqrt(c)` or `log1p(c)` — keeps the
 * contours roughly equidistant for both gentle (Booth) and steep
 * (Rosenbrock, Beale) surfaces.
 *
 * The first level is offset slightly above the minimum so the
 * innermost contour doesn't collapse to a single point.
 */
export function chooseLevels(
    grid: Float64Array | ArrayLike<number>,
    intensity: 'linear' | 'sqrt' | 'log1p',
    n: number,
): number[] {
    let cmin = Infinity;
    let cmax = -Infinity;
    for (let i = 0; i < grid.length; i++) {
        const v = grid[i];
        if (!Number.isFinite(v)) continue;
        if (v < cmin) cmin = v;
        if (v > cmax) cmax = v;
    }
    if (!Number.isFinite(cmin) || !Number.isFinite(cmax) || cmax <= cmin)
        return [];
    const fwd = (c: number) => transform(c, intensity);
    const inv = (t: number) => invert(t, intensity);
    const tmin = fwd(Math.max(cmin, 0));
    const tmax = fwd(cmax);
    const span = tmax - tmin;
    const out: number[] = [];
    for (let i = 1; i <= n; i++) {
        // i / (n + 1) gives interior fractions, so the outermost
        // contour stays inside the grid border.
        const t = tmin + span * (i / (n + 1));
        out.push(inv(t));
    }
    return out;
}

function transform(c: number, intensity: 'linear' | 'sqrt' | 'log1p'): number {
    if (!Number.isFinite(c) || c < 0) return 0;
    switch (intensity) {
        case 'sqrt':
            return Math.sqrt(c);
        case 'log1p':
            return Math.log1p(c);
        default:
            return c;
    }
}

function invert(t: number, intensity: 'linear' | 'sqrt' | 'log1p'): number {
    switch (intensity) {
        case 'sqrt':
            return t * t;
        case 'log1p':
            return Math.expm1(t);
        default:
            return t;
    }
}

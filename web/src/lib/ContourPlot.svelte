<script lang="ts">
    import { onMount } from 'svelte';
    import { chainSegments, chooseLevels, isoContour, smoothChaikin } from './contours';
    import { paletteFor, type Theme } from './palette';
    import type { Domain, ProblemMeta } from './problems';

    type Props = {
        problem: ProblemMeta;
        grid: Float64Array; // length nx*ny, row-major (rows are y)
        nx: number;
        ny: number;
        trajectory: Float64Array; // flat (x, y) pairs
        startPoint: { x: number; y: number };
        theme: Theme;
        onPick: (p: { x: number; y: number }) => void;
    };

    let {
        problem,
        grid,
        nx,
        ny,
        trajectory,
        startPoint,
        theme,
        onPick,
    }: Props = $props();

    let palette = $derived(paletteFor(theme));

    let canvas: HTMLCanvasElement | undefined = $state();
    let overlay: HTMLCanvasElement | undefined = $state();
    let containerWidth = $state(640);
    let containerHeight = $state(480);

    // How many iso-contours to draw. 12 reads as "topographic map" without
    // becoming visual noise.
    const N_LEVELS = 12;

    // Re-extract iso-contours when the grid (i.e. the problem) changes.
    // This is the expensive step — O(nx * ny * n_levels) — but it only
    // runs on problem switches, not per animation frame.
    let isoLines = $derived.by(() => {
        const levels = chooseLevels(grid, problem.intensity, N_LEVELS);
        const d = problem.domain;
        return levels.map((level) => {
            const segs = isoContour(
                grid,
                nx,
                ny,
                d.xmin,
                d.xmax,
                d.ymin,
                d.ymax,
                level,
            );
            // Stitch into polylines so we can stroke each as one path,
            // then Chaikin-smooth to kill the marching-squares
            // stair-step. Two iterations is the sweet spot.
            const chains = chainSegments(segs).map((c) => smoothChaikin(c, 2));
            return { level, chains };
        });
    });

    // Render contours when contours, sizing, theme, or domain change.
    $effect(() => {
        if (!canvas) return;
        renderContours(canvas, problem.domain, isoLines, palette);
    });

    // Trajectory + markers go on a separate overlay so they redraw cheaply
    // every time the trajectory grows. Re-runs on theme too so marker
    // colors flip immediately.
    $effect(() => {
        if (!overlay) return;
        renderOverlay(
            overlay,
            problem.domain,
            trajectory,
            startPoint,
            problem.minimum,
            palette,
        );
    });

    onMount(() => {
        const ro = new ResizeObserver((entries) => {
            for (const e of entries) {
                containerWidth = Math.max(120, Math.floor(e.contentRect.width));
                containerHeight = Math.max(
                    120,
                    Math.floor(e.contentRect.height),
                );
            }
        });
        ro.observe(canvas!.parentElement!);
        return () => ro.disconnect();
    });

    function dataToPixel(
        x: number,
        y: number,
        d: Domain,
        w: number,
        h: number,
    ) {
        const px = ((x - d.xmin) / (d.xmax - d.xmin)) * w;
        // Flip y so larger y is up.
        const py = h - ((y - d.ymin) / (d.ymax - d.ymin)) * h;
        return [px, py];
    }

    function pixelToData(
        px: number,
        py: number,
        d: Domain,
        w: number,
        h: number,
    ) {
        const x = d.xmin + (px / w) * (d.xmax - d.xmin);
        const y = d.ymin + ((h - py) / h) * (d.ymax - d.ymin);
        return { x, y };
    }

    function renderContours(
        cv: HTMLCanvasElement,
        d: Domain,
        lines: { level: number; chains: number[][] }[],
        pal: ReturnType<typeof paletteFor>,
    ) {
        const dpr = window.devicePixelRatio || 1;
        const w = containerWidth;
        const h = containerHeight;
        cv.width = Math.floor(w * dpr);
        cv.height = Math.floor(h * dpr);
        cv.style.width = `${w}px`;
        cv.style.height = `${h}px`;
        const ctx = cv.getContext('2d');
        if (!ctx) return;
        ctx.scale(dpr, dpr);

        ctx.fillStyle = pal.surface;
        ctx.fillRect(0, 0, w, h);

        if (lines.length === 0) return;

        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';

        // Draw outermost-first so the brightest (innermost) strokes win
        // when contours crowd. `t = 0` is the outermost, `t = 1` the
        // innermost — palette decides what those map to per theme.
        for (let li = lines.length - 1; li >= 0; li--) {
            const chains = lines[li].chains;
            if (chains.length === 0) continue;
            const t = li / Math.max(lines.length - 1, 1);
            ctx.strokeStyle = pal.contour(1 - t);
            // Inner contours a hair thicker so the basin reads.
            ctx.lineWidth = 1 + (1 - t) * 0.6;
            ctx.beginPath();
            for (const chain of chains) {
                if (chain.length < 4) continue;
                const [px0, py0] = dataToPixel(chain[0], chain[1], d, w, h);
                ctx.moveTo(px0, py0);
                for (let k = 2; k < chain.length; k += 2) {
                    const [px, py] = dataToPixel(
                        chain[k],
                        chain[k + 1],
                        d,
                        w,
                        h,
                    );
                    ctx.lineTo(px, py);
                }
            }
            ctx.stroke();
        }
    }

    function renderOverlay(
        cv: HTMLCanvasElement,
        d: Domain,
        traj: Float64Array,
        start: { x: number; y: number },
        minimum: { x: number; y: number },
        pal: ReturnType<typeof paletteFor>,
    ) {
        const dpr = window.devicePixelRatio || 1;
        const w = containerWidth;
        const h = containerHeight;
        cv.width = Math.floor(w * dpr);
        cv.height = Math.floor(h * dpr);
        cv.style.width = `${w}px`;
        cv.style.height = `${h}px`;
        const ctx = cv.getContext('2d');
        if (!ctx) return;
        ctx.scale(dpr, dpr);
        ctx.clearRect(0, 0, w, h);

        // Trajectory line + dots.
        if (traj.length >= 4) {
            ctx.beginPath();
            for (let i = 0; i < traj.length; i += 2) {
                const [px, py] = dataToPixel(traj[i], traj[i + 1], d, w, h);
                if (i === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
            }
            ctx.lineWidth = 2;
            ctx.strokeStyle = pal.trajectory;
            ctx.stroke();
        }
        for (let i = 2; i < traj.length; i += 2) {
            const [px, py] = dataToPixel(traj[i], traj[i + 1], d, w, h);
            ctx.beginPath();
            ctx.arc(px, py, 2, 0, Math.PI * 2);
            ctx.fillStyle = pal.trajectory;
            ctx.fill();
        }

        // Start marker (open circle).
        const [sx, sy] = dataToPixel(start.x, start.y, d, w, h);
        ctx.beginPath();
        ctx.arc(sx, sy, 6, 0, Math.PI * 2);
        ctx.lineWidth = 2;
        ctx.strokeStyle = pal.startMarker;
        ctx.stroke();

        // Known minimum (cross).
        const [mx, my] = dataToPixel(minimum.x, minimum.y, d, w, h);
        ctx.lineWidth = 2;
        ctx.strokeStyle = pal.minimum;
        ctx.beginPath();
        ctx.moveTo(mx - 6, my);
        ctx.lineTo(mx + 6, my);
        ctx.moveTo(mx, my - 6);
        ctx.lineTo(mx, my + 6);
        ctx.stroke();
    }

    function handlePointer(ev: PointerEvent) {
        if (!overlay) return;
        const rect = overlay.getBoundingClientRect();
        const px = ev.clientX - rect.left;
        const py = ev.clientY - rect.top;
        const p = pixelToData(
            px,
            py,
            problem.domain,
            rect.width,
            rect.height,
        );
        onPick(p);
    }
</script>

<div class="relative w-full h-full">
    <!-- Contour canvas, sized in CSS px (no longer at native grid res). -->
    <canvas bind:this={canvas} class="absolute inset-0 w-full h-full"></canvas>
    <!-- Overlay sits on top, sized to the displayed container in CSS px. -->
    <canvas
        bind:this={overlay}
        class="absolute inset-0 w-full h-full cursor-crosshair"
        onpointerdown={handlePointer}
    ></canvas>
</div>

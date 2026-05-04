<script lang="ts">
    import { onMount } from 'svelte';
    import { viridis } from './colormap';
    import { chooseLevels, isoContour } from './contours';
    import type { Domain, ProblemMeta } from './problems';

    type Props = {
        problem: ProblemMeta;
        grid: Float64Array; // length nx*ny, row-major (rows are y)
        nx: number;
        ny: number;
        trajectory: Float64Array; // flat (x, y) pairs
        startPoint: { x: number; y: number };
        onPick: (p: { x: number; y: number }) => void;
    };

    let {
        problem,
        grid,
        nx,
        ny,
        trajectory,
        startPoint,
        onPick,
    }: Props = $props();

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
        return levels.map((level) => ({
            level,
            segments: isoContour(grid, nx, ny, d.xmin, d.xmax, d.ymin, d.ymax, level),
        }));
    });

    // Render contours when contours, sizing, or domain change.
    $effect(() => {
        if (!canvas) return;
        renderContours(canvas, problem.domain, isoLines);
    });

    // Trajectory + markers go on a separate overlay so they redraw cheaply
    // every time the trajectory grows.
    $effect(() => {
        if (!overlay) return;
        renderOverlay(
            overlay,
            problem.domain,
            trajectory,
            startPoint,
            problem.minimum,
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
        lines: { level: number; segments: Float64Array }[],
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

        // Background — slightly lighter than the page so contours pop.
        ctx.fillStyle = 'rgb(15, 23, 42)';
        ctx.fillRect(0, 0, w, h);

        if (lines.length === 0) return;

        // Color contours along viridis from inner (low cost, bright) to
        // outer (high cost, dark). Drawing inner-to-outer keeps the
        // brightest stroke on top.
        for (let li = lines.length - 1; li >= 0; li--) {
            const seg = lines[li].segments;
            if (seg.length === 0) continue;
            const t = li / Math.max(lines.length - 1, 1);
            const [r, g, b] = viridis(1 - t);
            ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, 0.85)`;
            // Inner contours a hair thicker so the basin reads.
            ctx.lineWidth = 1 + (1 - t) * 0.6;
            ctx.beginPath();
            for (let k = 0; k < seg.length; k += 4) {
                const [px0, py0] = dataToPixel(seg[k], seg[k + 1], d, w, h);
                const [px1, py1] = dataToPixel(seg[k + 2], seg[k + 3], d, w, h);
                ctx.moveTo(px0, py0);
                ctx.lineTo(px1, py1);
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
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.95)';
            ctx.stroke();
        }
        for (let i = 2; i < traj.length; i += 2) {
            const [px, py] = dataToPixel(traj[i], traj[i + 1], d, w, h);
            ctx.beginPath();
            ctx.arc(px, py, 2, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';
            ctx.fill();
        }

        // Start marker (open circle).
        const [sx, sy] = dataToPixel(start.x, start.y, d, w, h);
        ctx.beginPath();
        ctx.arc(sx, sy, 6, 0, Math.PI * 2);
        ctx.lineWidth = 2;
        ctx.strokeStyle = 'rgb(255, 255, 255)';
        ctx.stroke();

        // Known minimum (cross).
        const [mx, my] = dataToPixel(minimum.x, minimum.y, d, w, h);
        ctx.lineWidth = 2;
        ctx.strokeStyle = 'rgb(248, 113, 113)';
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

<script lang="ts">
    import { onMount } from 'svelte';

    import { paletteFor, type Theme } from './palette';

    type Props = {
        costs: Float64Array;
        /** Termination reason string when the run ends, else empty. */
        reason: string;
        theme: Theme;
    };

    let { costs, reason, theme }: Props = $props();
    let palette = $derived(paletteFor(theme));

    let canvas: HTMLCanvasElement | undefined = $state();
    let containerWidth = $state(320);
    let containerHeight = $state(240);

    $effect(() => {
        if (!canvas) return;
        render(canvas, costs, palette);
    });

    onMount(() => {
        const ro = new ResizeObserver((entries) => {
            for (const e of entries) {
                containerWidth = Math.max(120, Math.floor(e.contentRect.width));
                containerHeight = Math.max(
                    100,
                    Math.floor(e.contentRect.height),
                );
            }
        });
        ro.observe(canvas!.parentElement!);
        return () => ro.disconnect();
    });

    function render(
        cv: HTMLCanvasElement,
        c: Float64Array,
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

        ctx.fillStyle = pal.surface;
        ctx.fillRect(0, 0, w, h);

        if (c.length < 2) return;

        // Use log axis when costs span many orders of magnitude.
        let cmin = Infinity;
        let cmax = -Infinity;
        for (let i = 0; i < c.length; i++) {
            const v = c[i];
            if (!Number.isFinite(v)) continue;
            if (v < cmin) cmin = v;
            if (v > cmax) cmax = v;
        }
        if (!Number.isFinite(cmin) || !Number.isFinite(cmax)) return;
        const useLog = cmax > 0 && cmin >= 0 && cmax / Math.max(cmin, 1e-12) > 1e3;
        const eps = 1e-12;
        const yOf = useLog
            ? (v: number) => Math.log10(Math.max(v, eps))
            : (v: number) => v;
        const yLo = useLog ? yOf(Math.max(cmin, eps)) : cmin;
        const yHi = useLog ? yOf(Math.max(cmax, eps)) : cmax;
        const ySpan = Math.max(yHi - yLo, 1e-12);

        const padL = 36;
        const padR = 8;
        const padT = 16;
        const padB = 24;
        const innerW = Math.max(1, w - padL - padR);
        const innerH = Math.max(1, h - padT - padB);

        // Axes.
        ctx.strokeStyle = pal.axis;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padL, padT);
        ctx.lineTo(padL, padT + innerH);
        ctx.lineTo(padL + innerW, padT + innerH);
        ctx.stroke();

        ctx.fillStyle = pal.text;
        ctx.font = '11px sans-serif';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        ctx.fillText(formatTick(cmax, useLog), padL - 4, padT);
        ctx.fillText(formatTick(cmin, useLog), padL - 4, padT + innerH);
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.fillText(`iter ${c.length - 1}`, padL + 4, padT - 2);
        if (reason) {
            ctx.textAlign = 'right';
            ctx.fillStyle = pal.reason;
            ctx.fillText(reason, w - padR, padT - 2);
        }

        // Polyline.
        ctx.beginPath();
        for (let i = 0; i < c.length; i++) {
            const x = padL + (i / Math.max(c.length - 1, 1)) * innerW;
            const yv = yOf(Math.max(c[i], eps));
            const y = padT + innerH - ((yv - yLo) / ySpan) * innerH;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = pal.cost;
        ctx.lineWidth = 1.5;
        ctx.stroke();
    }

    function formatTick(v: number, useLog: boolean): string {
        if (!Number.isFinite(v)) return '';
        if (useLog) return v.toExponential(1);
        if (Math.abs(v) >= 1000 || (v !== 0 && Math.abs(v) < 0.01))
            return v.toExponential(1);
        return v.toFixed(3);
    }
</script>

<canvas bind:this={canvas} class="w-full h-full block"></canvas>

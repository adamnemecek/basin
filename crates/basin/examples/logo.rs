//! Generative logo for `basin`.
//!
//! Renders the package's namesake — a geographic *basin* — as a low-poly
//! isometric scene, and does it the way an optimization library should:
//! the terrain is a quadratic basin (the canonical convex optimization
//! landscape) carrying a damped sinusoidal ripple for organic ridges, and
//! the rivulet winding down into the pool is an actual gradient-descent
//! trajectory traced by basin's [`GradientDescent`] solver with heavy-ball
//! momentum, run over that exact surface. The pool is the surface's
//! minimum — exactly where the optimizer comes to rest. Momentum is what
//! makes the descent glide down the slope and settle in the basin floor
//! rather than darting straight in, so the river is, literally,
//! optimization output.
//!
//! Run with:
//!
//! ```text
//! cargo run --example logo            # writes basin-logo.svg
//! cargo run --example logo out.svg    # writes out.svg
//! ```
//!
//! Everything is driven by the constants in the `config` block below;
//! tweak the palette, grid resolution, the framing window, or the
//! momentum parameters to re-roll the look. Output is a standalone SVG.

use std::fmt::Write as _;

use basin::{BasicState, CostFunction, Gradient, GradientDescent, Solver, State};

// ---------------------------------------------------------------------------
// config
// ---------------------------------------------------------------------------

/// Grid cells per side (vertices = `GRID + 1`). Low enough to read as
/// faceted low-poly, high enough to keep the bowl curve smooth.
const GRID: usize = 22;

/// Problem-space framing window `[X0, X1] × [Y0, Y1]`.
const X0: f64 = -4.2;
const X1: f64 = 4.2;
const Y0: f64 = -4.2;
const Y1: f64 = 4.2;

/// Surface shape: a closed, *curved* (banana) valley. In coordinates
/// `(u, v)` rotated by `BOWL_ROT` about the centre, the floor follows the
/// parabola `v = VALLEY_CURVE·u²`; the surface is soft along that floor
/// (`VALLEY_FLOOR_K`) and stiff across it (`VALLEY_WALL_K`), and rises at
/// both ends so the minimum is *enclosed* in the middle. Gradient descent
/// drops to the floor and follows its bend into the pool — a curved river,
/// not a straight radial line, and one that can't spill off an open valley
/// mouth. Centred slightly off origin so the pool lands off-centre.
const BOWL_CX: f64 = 0.3;
const BOWL_CY: f64 = -0.3;
const VALLEY_FLOOR_K: f64 = 0.110; // along-valley softness (smaller = longer)
const VALLEY_WALL_K: f64 = 0.020; // across-valley stiffness
const VALLEY_CURVE: f64 = 0.200; // floor bend (parabola coefficient)
const BOWL_ROT: f64 = 0.1; // valley orientation (radians)
const RIM_K: f64 = 0.002;

/// Water depth as a normalised height above the basin floor — a small,
/// fixed pond rather than a percentile flood.
const WATER_LEVEL: f64 = 0.004;

/// Isometric tile half-extents (screen px) and vertical height gain.
const TILE_W: f64 = 20.0;
const TILE_H: f64 = 11.0;
const Z_SCREEN: f64 = 190.0; // screen px from basin floor to highest ridge
const Z_WORLD: f64 = 25.0; // height gain for facet-normal lighting (grid units)

/// River = gradient descent (light heavy-ball momentum) on the surface,
/// from a point up one arm of the curved valley. Momentum is kept low so
/// the descent *hugs* the bending floor rather than flying ballistically
/// across it; the path sweeps down the curve into the pool.
const RIVER_START: [f64; 2] = [-3.7, -2.5];
const RIVER_ALPHA: f64 = 0.02;
const RIVER_BETA: f64 = 0.10;
const RIVER_ITERS: usize = 1100;
const RIVER_POINTS: usize = 500; // polyline vertices before shoreline clip

/// River is drawn as a tapered filled ribbon (not a stroked line): it emerges
/// thin at the source and widens downstream, and its mouth runs *past* the
/// shoreline into the pool so the water fills merge seamlessly. Widths in
/// screen px.
const RIVER_W_SRC: f64 = 1.2; // ribbon width at the source (≈ a point)
const RIVER_W_MOUTH: f64 = 9.0; // ribbon width where it meets the pool
const RIVER_MOUTH_REACH: f64 = 0.55; // how far the mouth extends toward the pool centre

/// Source spring at the start point x₀: a small pool the river flows *out* of,
/// mirroring the lake at the optimum (spring = x₀, lake = x*). Half-extents are
/// `SPRING_SCALE` × the iso tile, so it reads a touch smaller than one tile.
const SPRING_SCALE: f64 = 0.6;

/// Trees: scattered at random over *plantable* ground — above the shoreline,
/// below the upper walls, and on gentle enough slopes to read as planted.
/// Placement is fully determined by `TREE_SEED`, so bump it to resample a
/// different arrangement (or raise `TREE_COUNT` for a denser stand).
const TREE_SEED: u64 = 15;
const TREE_COUNT: usize = 6;
const TREE_MIN_DIST: f64 = 1.6; // min separation between trees (problem units)
const TREE_MIN_LIFT: f64 = 0.05; // min normalised height above the water line
const TREE_MAX_HN: f64 = 0.5; // max normalised height (keeps trees off the rim)
const TREE_MAX_SLOPE: f64 = 0.13; // reject ground steeper than this
const TREE_MAX_ATTEMPTS: usize = 4000; // rejection-sampling budget per render

/// Rocks: a few low-poly boulders. One is always placed beside the source
/// spring (a screen-space offset from x₀); the rest are scattered at random,
/// reproducibly from `ROCK_SEED`. Rocks tolerate steeper, higher ground than
/// trees, so their slope/elevation bounds are looser. `ROCK_COUNT` is the
/// total including the spring rock.
const ROCK_SEED: u64 = 3;
const ROCK_COUNT: usize = 5;
const ROCK_SPRING_DX: f64 = -9.0; // spring rock: screen offset from the spring (px)
const ROCK_SPRING_DY: f64 = -5.0; // (screen-space so it nestles beside the water)
const ROCK_MIN_DIST: f64 = 1.4; // min separation between rocks (problem units)
const ROCK_MIN_LIFT: f64 = 0.02; // min normalised height above the water line
const ROCK_MAX_HN: f64 = 0.7; // max normalised height (rocks climb higher)
const ROCK_MAX_SLOPE: f64 = 0.22; // rocks sit on steeper ground than trees
const ROCK_MAX_ATTEMPTS: usize = 4000; // rejection-sampling budget per render
const ROCK_W: f64 = 12.0; // boulder half-width (px)
const ROCK_H: f64 = 14.0; // boulder height (px)

/// Teal/slate palette.
const PAPER: Rgb = Rgb(244, 241, 222); // #f4f1de background
                                       // Terrain hypsometric ramps (low elevation → high), each sampled from the land
                                       // section of a named colormap. To test a palette, change the `TERRAIN_RAMP`
                                       // line at the bottom of this block to `&TURKU` / `&BILBAO` / `&BAMAKO` /
                                       // `&SANDSTONE`, save, and look at `images/logo.svg` (run `task logo` and it
                                       // re-renders on every save). Add your own by pasting any colormap's stops —
                                       // `ramp_sample` interpolates across however many you give it.
#[allow(dead_code)] // Crameri `lajolla` — vivid red-rock / sunset
const SANDSTONE: [Rgb; 6] = [
    hex("#ca514b"),
    hex("#df6e4f"),
    hex("#e68c51"),
    hex("#eba853"),
    hex("#f2c75c"),
    hex("#fbea93"),
];
#[allow(dead_code)] // Crameri `turku` — muted olive/khaki → warm tan
const TURKU: [Rgb; 6] = [
    hex("#565640"),
    hex("#6d6c4a"),
    hex("#868255"),
    hex("#a79864"),
    hex("#c6a475"),
    hex("#dda888"),
];
#[allow(dead_code)] // Crameri `bilbao` — clay-rose → pale grey-tan
const BILBAO: [Rgb; 6] = [
    hex("#a36b59"),
    hex("#a87d5d"),
    hex("#ad9061"),
    hex("#b5a874"),
    hex("#c1bb9e"),
    hex("#cccac3"),
];
#[allow(dead_code)] // Crameri `bamako` — green lowlands → gold peaks
const BAMAKO: [Rgb; 6] = [
    hex("#335b28"),
    hex("#537014"),
    hex("#788501"),
    hex("#a2930d"),
    hex("#ceb546"),
    hex("#f3d993"),
];
#[allow(dead_code)] // Crameri `bamako` — green lowlands → gold peaks
const CUSTOM: [Rgb; 5] = [
    hex("#5E7763"),
    hex("#7C7A62"),
    hex("#9C836F"),
    hex("#B59A85"),
    hex("#DDD5CB"),
];
/// The active terrain ramp. ← change this one line to test a different palette.
const TERRAIN_RAMP: &[Rgb] = &CUSTOM;
const WATER: Rgb = Rgb(71, 153, 173); // lake + river (one melted body)
const TREE_DARK: Rgb = Rgb(54, 90, 82); // canopy shadow side
const TREE_LIT: Rgb = Rgb(92, 138, 122); // canopy lit side
const TRUNK: Rgb = Rgb(52, 46, 39);
const ROCK_LIT: Rgb = Rgb(150, 161, 160); // boulder sun-facing facet
const ROCK_DARK: Rgb = Rgb(92, 108, 109); // boulder shadow facet
const SUN: Rgb = Rgb(233, 196, 106); // #e9c46a accent

/// Sun: a soft halo around a solid disk in the upper-right. `SUN_R` is the core
/// radius (screen px); the halo scales with it via `SUN_HALO`.
const SUN_R: f64 = 39.0;
const SUN_HALO: f64 = 1.45; // halo radius as a multiple of the core

/// Light direction in world space (`+z` up), pointing *from* the light.
const LIGHT: [f64; 3] = [0.5, -0.35, 0.79];

// ---------------------------------------------------------------------------
// surface
// ---------------------------------------------------------------------------

/// Raw bowl height at problem point `(x, y)` — a quadratic basin with a
/// damped sinusoidal ripple (organic ridges) and a gentle rising rim. The
/// ripple fades to zero at the centre so the basin floor, and hence the
/// pool, stays clean and the basin stays unimodal. This is the function
/// basin's solver descends to trace the river.
fn raw_height(x: f64, y: f64) -> f64 {
    let dx = x - BOWL_CX;
    let dy = y - BOWL_CY;
    let (ca, sa) = (BOWL_ROT.cos(), BOWL_ROT.sin());
    let u = ca * dx + sa * dy; // along the valley (soft)
    let v = -sa * dx + ca * dy; // across the valley (stiff)
    let across = v - VALLEY_CURVE * u * u; // signed distance from the curved floor
    let bowl = VALLEY_FLOOR_K * u * u + VALLEY_WALL_K * across * across;
    // Ripple keyed to `across` (perpendicular to the curved floor): it
    // textures the walls but fades to zero on the floor (`across ≈ 0`), so it
    // never traps the descent following the floor. A tiny u-term adds gentle
    // along-valley variation. `damp` grows with height, so the floor stays
    // clean.
    let damp = (bowl / 0.6).min(1.0);
    let w1 = damp * 0.20 * (1.30 * across + 0.4).sin();
    let w2 = damp * 0.10 * (2.40 * across - 0.7).sin();
    let w3 = damp * 0.05 * (0.70 * u + 0.3).sin();
    let rim = RIM_K * (x * x + y * y);
    bowl + w1 + w2 + w3 + rim
}

/// Normalisation + water level for the surface, sampled once.
struct Surface {
    hmin: f64,
    hmax: f64,
    water: f64,
}

impl Surface {
    fn new() -> Self {
        let n = 96;
        let mut vals = Vec::with_capacity((n + 1) * (n + 1));
        for j in 0..=n {
            for i in 0..=n {
                let x = X0 + (X1 - X0) * i as f64 / n as f64;
                let y = Y0 + (Y1 - Y0) * j as f64 / n as f64;
                vals.push(raw_height(x, y));
            }
        }
        let hmin = vals.iter().copied().fold(f64::INFINITY, f64::min);
        let hmax = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Surface {
            hmin,
            hmax,
            water: WATER_LEVEL,
        }
    }

    /// Normalised height in `[0, 1]`.
    fn hn(&self, x: f64, y: f64) -> f64 {
        ((raw_height(x, y) - self.hmin) / (self.hmax - self.hmin)).clamp(0.0, 1.0)
    }

    /// On-surface screen position of a problem point, lifted by `lift`
    /// (normalised units) so rivers/trees ride just above their facets.
    fn surface_point(&self, x: f64, y: f64, lift: f64) -> (f64, f64) {
        let gx = (x - X0) / (X1 - X0) * GRID as f64;
        let gy = (y - Y0) / (Y1 - Y0) * GRID as f64;
        project(gx, gy, self.hn(x, y) + lift)
    }
}

/// Problem-space coordinates of grid node `(i, j)`.
fn node_xy(i: usize, j: usize) -> (f64, f64) {
    (
        X0 + (X1 - X0) * i as f64 / GRID as f64,
        Y0 + (Y1 - Y0) * j as f64 / GRID as f64,
    )
}

/// Isometric projection: grid coords `(gx, gy)` on the ground plane, plus
/// normalised height `h`.
fn project(gx: f64, gy: f64, h: f64) -> (f64, f64) {
    let sx = (gx - gy) * TILE_W;
    let sy = (gx + gy) * TILE_H - h * Z_SCREEN;
    (sx, sy)
}

fn normalize3(v: [f64; 3]) -> [f64; 3] {
    let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-12);
    [v[0] / n, v[1] / n, v[2] / n]
}

/// The rendered surface, exposed as a `CostFunction` + `Gradient` so
/// basin's [`GradientDescent`] can descend the exact terrain we draw. The
/// gradient is a central difference of [`raw_height`] — no need to
/// hand-differentiate the ripple, and it stays perfectly consistent with
/// the surface.
struct Bowl;

impl CostFunction for Bowl {
    type Param = Vec<f64>;
    type Output = f64;
    fn cost(&self, x: &Vec<f64>) -> f64 {
        raw_height(x[0], x[1])
    }
}

impl Gradient for Bowl {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;
    fn gradient(&self, x: &Vec<f64>) -> Vec<f64> {
        let h = 1e-4;
        let mut g = vec![0.0; 2];
        for (k, gk) in g.iter_mut().enumerate() {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[k] += h;
            xm[k] -= h;
            *gk = (raw_height(xp[0], xp[1]) - raw_height(xm[0], xm[1])) / (2.0 * h);
        }
        g
    }
}

/// Trace the river: heavy-ball gradient descent on the bowl, driven
/// through basin's `Solver` loop, capturing every iterate then decimating.
fn trace_river() -> Vec<[f64; 2]> {
    let mut solver = GradientDescent::new(RIVER_ALPHA).with_momentum(RIVER_BETA);
    let mut state = solver.init(&Bowl, BasicState::new(RIVER_START.to_vec()));
    let mut full = Vec::with_capacity(RIVER_ITERS + 1);
    full.push([state.param()[0], state.param()[1]]);
    for _ in 0..RIVER_ITERS {
        let (next, stop) = solver.next_iter(&Bowl, state);
        state = next;
        full.push([state.param()[0], state.param()[1]]);
        if stop.is_some() {
            break;
        }
    }
    let stride = (full.len() / RIVER_POINTS).max(1);
    let mut pts: Vec<[f64; 2]> = full.iter().step_by(stride).copied().collect();
    if let Some(&last) = full.last() {
        if pts.last() != Some(&last) {
            pts.push(last);
        }
    }
    pts
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let out_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "basin-logo.svg".into());

    let surf = Surface::new();
    let river = trace_river();
    // The basin minimum *is* the pool centre — and where the river settles.
    let pool_center = basin_min();

    let mut svg = Svg::new();
    draw_sun(&mut svg);
    draw_terrain(&mut svg, &surf, &river, pool_center);
    let n_rocks = draw_rocks(&mut svg, &surf);
    let n_trees = draw_trees(&mut svg, &surf);

    let doc = svg.finish();
    std::fs::write(&out_path, &doc).expect("write SVG");
    let end = river.last().copied().unwrap_or(pool_center);
    eprintln!(
        "wrote {out_path} ({} bytes); river: {} steps from {:?} to ({:.3}, {:.3}); pool at ({:.3}, {:.3}); {n_trees}/{TREE_COUNT} trees (seed {TREE_SEED}); {n_rocks}/{ROCK_COUNT} rocks (seed {ROCK_SEED})",
        doc.len(),
        river.len(),
        RIVER_START,
        end[0],
        end[1],
        pool_center[0],
        pool_center[1],
    );
}

/// True global minimum of the surface (fine grid search). The pool sits
/// here; on this unimodal basin it's also where the river settles.
fn basin_min() -> [f64; 2] {
    let n = 260;
    let (mut best, mut bv) = ([BOWL_CX, BOWL_CY], f64::INFINITY);
    for j in 0..=n {
        for i in 0..=n {
            let x = X0 + (X1 - X0) * i as f64 / n as f64;
            let y = Y0 + (Y1 - Y0) * j as f64 / n as f64;
            let val = raw_height(x, y);
            if val < bv {
                bv = val;
                best = [x, y];
            }
        }
    }
    best
}

// ---------------------------------------------------------------------------
// drawing
// ---------------------------------------------------------------------------

/// One isometric triangle facet, flat-shaded by its normal and tinted by
/// elevation; drawn back-to-front by the caller.
struct Facet {
    pts: Vec<(f64, f64)>,
    depth: f64,
    color: Rgb,
}

/// Sample a continuous color ramp at `t ∈ [0, 1]` between equally-spaced anchor
/// stops — a poor-man's colormap lookup. Drop in any number of stops (e.g. a
/// hypsometric/topographic colormap) and this maps normalised elevation to
/// color. Blending goes through [`Rgb::lerp`], which interpolates in Oklab.
fn ramp_sample(stops: &[Rgb], t: f64) -> Rgb {
    let t = t.clamp(0.0, 1.0);
    if stops.len() < 2 {
        return stops[0];
    }
    let s = t * (stops.len() - 1) as f64;
    let i = (s.floor() as usize).min(stops.len() - 2);
    stops[i].lerp(stops[i + 1], s - i as f64)
}

/// sRGB channel (0–1) → linear light.
fn srgb_to_linear(c: f64) -> f64 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Linear light → sRGB channel (0–1).
fn linear_to_srgb(c: f64) -> f64 {
    if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

fn rgb_to_oklab(c: Rgb) -> [f64; 3] {
    let r = srgb_to_linear(c.0 as f64 / 255.0);
    let g = srgb_to_linear(c.1 as f64 / 255.0);
    let b = srgb_to_linear(c.2 as f64 / 255.0);
    let l = (0.412_221_470_8 * r + 0.536_332_536_3 * g + 0.051_445_992_9 * b).cbrt();
    let m = (0.211_903_498_2 * r + 0.680_699_545_1 * g + 0.107_396_956_6 * b).cbrt();
    let s = (0.088_302_461_9 * r + 0.281_718_837_6 * g + 0.629_978_700_5 * b).cbrt();
    [
        0.210_454_255_3 * l + 0.793_617_785_0 * m - 0.004_072_046_8 * s,
        1.977_998_495_1 * l - 2.428_592_205_0 * m + 0.450_593_709_9 * s,
        0.025_904_037_1 * l + 0.782_771_766_2 * m - 0.808_675_766_0 * s,
    ]
}

fn oklab_to_rgb(lab: [f64; 3]) -> Rgb {
    let l = (lab[0] + 0.396_337_777_4 * lab[1] + 0.215_803_757_3 * lab[2]).powi(3);
    let m = (lab[0] - 0.105_561_345_8 * lab[1] - 0.063_854_172_8 * lab[2]).powi(3);
    let s = (lab[0] - 0.089_484_177_5 * lab[1] - 1.291_485_548_0 * lab[2]).powi(3);
    let r = 4.076_741_662_1 * l - 3.307_711_591_3 * m + 0.230_969_929_2 * s;
    let g = -1.268_438_004_6 * l + 2.609_757_401_1 * m - 0.341_319_396_5 * s;
    let b = -0.004_196_086_3 * l - 0.703_418_614_7 * m + 1.707_614_701_0 * s;
    let q = |c: f64| (linear_to_srgb(c).clamp(0.0, 1.0) * 255.0).round() as u8;
    Rgb(q(r), q(g), q(b))
}

fn draw_terrain(svg: &mut Svg, surf: &Surface, river: &[[f64; 2]], center: [f64; 2]) {
    let wl = surf.water;
    // True normalised height per node, plus the same clamped up to the water
    // line. The clamped heights give the geometry — a flat lake surface at the
    // water line — while the true heights say which facets are submerged (and
    // how deep), so the lake is coloured as water *within the same mesh*: its
    // shoreline is the grid-aligned facet boundary, not a shape laid on top.
    let ht: Vec<Vec<f64>> = (0..=GRID)
        .map(|j| {
            (0..=GRID)
                .map(|i| {
                    let (x, y) = node_xy(i, j);
                    surf.hn(x, y)
                })
                .collect()
        })
        .collect();
    let hh: Vec<Vec<f64>> = ht
        .iter()
        .map(|row| row.iter().map(|&h| h.max(wl)).collect())
        .collect();

    let light = normalize3(LIGHT);
    let world = |i: usize, j: usize| [i as f64, j as f64, hh[j][i] * Z_WORLD];
    let screen = |i: usize, j: usize| project(i as f64, j as f64, hh[j][i]);

    // Lake + river + spring as screen-space polygons, carved into the faces
    // they cross (not drawn on top). Each face keeps its terrain colour; the
    // slivers a water shape covers are re-emitted in one flat water shade.
    // Water is a horizontal surface, so it gets a *single* shade (lit as a flat
    // facet) rather than each underlying face's — otherwise the terrain
    // faceting shows through the river/lake.
    let water = water_shapes(surf, river, center);
    let water_color = WATER.shade(0.70 + 0.5 * light[2].max(0.0));

    let mut facets: Vec<Facet> = Vec::with_capacity(GRID * GRID * 2);
    for j in 0..GRID {
        for i in 0..GRID {
            for tri in [
                [(i, j), (i + 1, j), (i, j + 1)],
                [(i + 1, j), (i + 1, j + 1), (i, j + 1)],
            ] {
                let w: Vec<[f64; 3]> = tri.iter().map(|&(a, b)| world(a, b)).collect();
                let mut n = normalize3(cross(sub(w[1], w[0]), sub(w[2], w[0])));
                if n[2] < 0.0 {
                    n = [-n[0], -n[1], -n[2]];
                }
                let ndotl = (n[0] * light[0] + n[1] * light[1] + n[2] * light[2]).max(0.0);

                // Terrain colour: the elevation ramp shaded by this face. The
                // lake/river/spring are carved on top (below) as clipped water.
                let elev = tri.iter().map(|&(a, b)| hh[b][a]).sum::<f64>() / 3.0;
                let color = ramp_sample(TERRAIN_RAMP, elev).shade(0.70 + 0.5 * ndotl);

                let pts: Vec<(f64, f64)> = vec![
                    screen(tri[0].0, tri[0].1),
                    screen(tri[1].0, tri[1].1),
                    screen(tri[2].0, tri[2].1),
                ];
                let depth = tri.iter().map(|&(a, b)| (a + b) as f64).sum::<f64>() / 3.0;
                facets.push(Facet { pts, depth, color });
            }
        }
    }

    facets.sort_by(|a, b| a.depth.partial_cmp(&b.depth).unwrap());
    for f in &facets {
        // Stroke each facet in its *own* fill colour: adjacent facets' strokes
        // overlap the shared edge and cover the anti-aliasing seam, so the cream
        // background no longer bleeds through as a mesh of lines. Same-coloured
        // neighbours melt together; only real colour steps (the faceting) show.
        svg.polygon(&f.pts, f.color, Some((f.color, 1.0)));
    }

    // Water on top: the lake, river and spring as flat matte shapes in one
    // colour (with a matching stroke so the river quads and lake melt with no
    // seam). Drawn after the terrain so no facet stroke can bleed into the
    // water, and the whole system reads as one smooth body sitting in the basin.
    for shape in &water {
        svg.polygon(shape, water_color, Some((water_color, 1.0)));
    }
}

/// The whole water system as screen-space polygons drawn flat on top of the
/// terrain: the lake (water-level contour around the basin minimum), the river
/// (tapered convex quads down the gradient-descent trajectory and into the
/// lake), and the source spring (a small round pool at x₀). One flat colour for
/// all three, so spring → river → lake read as one smooth body of water.
fn water_shapes(surf: &Surface, river: &[[f64; 2]], center: [f64; 2]) -> Vec<Vec<(f64, f64)>> {
    let wl = surf.water;
    let proj = |x: f64, y: f64| {
        let gx = (x - X0) / (X1 - X0) * GRID as f64;
        let gy = (y - Y0) / (Y1 - Y0) * GRID as f64;
        project(gx, gy, surf.hn(x, y).max(wl) + 0.006)
    };
    let mut shapes: Vec<Vec<(f64, f64)>> = Vec::new();

    // Lake: trace the water-level contour radially out from the basin floor
    // (the bowl is monotone in radius near the floor, so each ray crosses once).
    let rmax = (X1 - X0).abs().max((Y1 - Y0).abs());
    let rays = 64;
    let mut ring: Vec<(f64, f64)> = Vec::with_capacity(rays);
    for k in 0..rays {
        let theta = std::f64::consts::TAU * k as f64 / rays as f64;
        let (dx, dy) = (theta.cos(), theta.sin());
        let (mut lo, mut hi) = (0.0_f64, rmax);
        for _ in 0..40 {
            let mid = 0.5 * (lo + hi);
            if surf.hn(center[0] + dx * mid, center[1] + dy * mid) < wl {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let r = 0.5 * (lo + hi);
        ring.push(proj(center[0] + dx * r, center[1] + dy * r));
    }
    shapes.push(ring);

    // River centreline: trajectory up to the shoreline, plus one synthetic
    // point reaching into the lake so the mouth overlaps it.
    let mut cl_xy: Vec<[f64; 2]> = Vec::new();
    for q in river {
        cl_xy.push(*q);
        if surf.hn(q[0], q[1]) <= wl {
            break;
        }
    }
    if cl_xy.len() < 2 {
        return shapes; // no river reached the lake — just the lake
    }
    let sp = *cl_xy.last().unwrap();
    cl_xy.push([
        sp[0] + (center[0] - sp[0]) * RIVER_MOUTH_REACH,
        sp[1] + (center[1] - sp[1]) * RIVER_MOUTH_REACH,
    ]);
    // decimate so quads are a sensible size (the faces break them up anyway)
    let stride = (cl_xy.len() / 40).max(1);
    let mut cl: Vec<(f64, f64)> = cl_xy
        .iter()
        .step_by(stride)
        .map(|q| proj(q[0], q[1]))
        .collect();
    let last = proj(cl_xy[cl_xy.len() - 1][0], cl_xy[cl_xy.len() - 1][1]);
    if cl.last() != Some(&last) {
        cl.push(last);
    }
    let n = cl.len();
    if n < 2 {
        return Vec::new();
    }
    let width = |i: usize| -> f64 {
        let t = (i as f64 / (n - 1) as f64).powf(0.75);
        RIVER_W_SRC + (RIVER_W_MOUTH - RIVER_W_SRC) * t
    };
    let mut left = Vec::with_capacity(n);
    let mut right = Vec::with_capacity(n);
    for i in 0..n {
        let prev = cl[i.saturating_sub(1)];
        let next = cl[(i + 1).min(n - 1)];
        let (mut tx, mut ty) = (next.0 - prev.0, next.1 - prev.1);
        let tl = (tx * tx + ty * ty).sqrt().max(1e-9);
        tx /= tl;
        ty /= tl;
        let (nx, ny) = (-ty, tx);
        let hw = 0.5 * width(i);
        left.push((cl[i].0 + nx * hw, cl[i].1 + ny * hw));
        right.push((cl[i].0 - nx * hw, cl[i].1 - ny * hw));
    }
    for i in 0..n - 1 {
        shapes.push(vec![left[i], left[i + 1], right[i + 1], right[i]]);
    }
    // source spring: a small round pool at x₀ (enough sides to read as smooth,
    // matching the lake contour rather than an angular hexagon)
    let (sx, sy) = cl[0];
    let (rx, ry) = (TILE_W * SPRING_SCALE, TILE_H * SPRING_SCALE);
    let spring_sides = 28;
    let spring: Vec<(f64, f64)> = (0..spring_sides)
        .map(|k| {
            let ang = std::f64::consts::TAU * k as f64 / spring_sides as f64;
            (sx + rx * ang.cos(), sy + ry * ang.sin())
        })
        .collect();
    shapes.push(spring);
    shapes
}

/// Scatter trees at random over plantable ground, reproducibly from
/// `TREE_SEED`. We rejection-sample problem-space points, keeping those that
/// sit above the shoreline, below the upper walls, on gentle slopes, and far
/// enough from already-placed trees. Returns how many were actually placed —
/// a tight `TREE_MIN_DIST` or a small eligible area can fall short of
/// `TREE_COUNT` within the attempt budget.
fn draw_trees(svg: &mut Svg, surf: &Surface) -> usize {
    let wl = surf.water;
    // Slope sampled over one grid cell in each axis, matching the spacing the
    // `TREE_MAX_SLOPE` threshold was tuned against.
    let dx = (X1 - X0) / GRID as f64;
    let dy = (Y1 - Y0) / GRID as f64;
    let mut rng = Rng::new(TREE_SEED);
    let mut chosen: Vec<(f64, f64)> = Vec::new();
    for _ in 0..TREE_MAX_ATTEMPTS {
        if chosen.len() >= TREE_COUNT {
            break;
        }
        let x = X0 + (X1 - X0) * rng.next_f64();
        let y = Y0 + (Y1 - Y0) * rng.next_f64();
        let h = surf.hn(x, y);
        if h < wl + TREE_MIN_LIFT || h > TREE_MAX_HN {
            continue;
        }
        let slope = (surf.hn(x + dx, y) - surf.hn(x - dx, y)).abs()
            + (surf.hn(x, y + dy) - surf.hn(x, y - dy)).abs();
        if slope > TREE_MAX_SLOPE {
            continue;
        }
        if chosen
            .iter()
            .any(|&(cx, cy)| (x - cx).hypot(y - cy) < TREE_MIN_DIST)
        {
            continue;
        }
        chosen.push((x, y));
    }

    chosen.sort_by(|a, b| (a.0 + a.1).partial_cmp(&(b.0 + b.1)).unwrap()); // far trees first
    for &(x, y) in &chosen {
        draw_tree(svg, surf.surface_point(x, y, 0.0));
    }
    chosen.len()
}

/// A small low-poly conifer: stacked triangle tiers over a short trunk,
/// each tier split lit/shadow down the centre seam.
fn draw_tree(svg: &mut Svg, base: (f64, f64)) {
    let (bx, by) = base;
    let w = 12.0;
    let trunk_h = 7.0;
    let tier_h = 16.0;

    svg.rect(bx - 1.8, by - trunk_h, 3.6, trunk_h + 1.0, TRUNK);
    let top = by - trunk_h;
    for t in 0..3 {
        let level = top - t as f64 * (tier_h * 0.6);
        let half = w * (1.0 - t as f64 * 0.22);
        let apex = (bx, level - tier_h);
        svg.polygon(&[apex, (bx - half, level), (bx, level)], TREE_DARK, None);
        svg.polygon(&[apex, (bx, level), (bx + half, level)], TREE_LIT, None);
    }
}

/// Scatter low-poly boulders: one nestled beside the source spring, the rest
/// sampled at random (seeded by `ROCK_SEED`) over rocky ground. Mirrors
/// `draw_trees`' eligibility test with the looser `ROCK_*` bounds. Returns how
/// many were placed (incl. the spring rock).
fn draw_rocks(svg: &mut Svg, surf: &Surface) -> usize {
    let wl = surf.water;
    let dx = (X1 - X0) / GRID as f64;
    let dy = (Y1 - Y0) / GRID as f64;
    let mut rng = Rng::new(ROCK_SEED);
    // The spring rock is placed by a screen-space offset (so it sits right next
    // to the water at the same height); the random rocks keep clear of x₀.
    let mut chosen: Vec<(f64, f64)> = vec![(RIVER_START[0], RIVER_START[1])];
    for _ in 0..ROCK_MAX_ATTEMPTS {
        if chosen.len() >= ROCK_COUNT {
            break;
        }
        let x = X0 + (X1 - X0) * rng.next_f64();
        let y = Y0 + (Y1 - Y0) * rng.next_f64();
        let h = surf.hn(x, y);
        if h < wl + ROCK_MIN_LIFT || h > ROCK_MAX_HN {
            continue;
        }
        let slope = (surf.hn(x + dx, y) - surf.hn(x - dx, y)).abs()
            + (surf.hn(x, y + dy) - surf.hn(x, y - dy)).abs();
        if slope > ROCK_MAX_SLOPE {
            continue;
        }
        if chosen
            .iter()
            .any(|&(cx, cy)| (x - cx).hypot(y - cy) < ROCK_MIN_DIST)
        {
            continue;
        }
        chosen.push((x, y));
    }

    // Spring rock first (it sits high/back), then the scattered rocks far-first.
    let (sx, sy) = surf.surface_point(RIVER_START[0], RIVER_START[1], 0.0);
    draw_rock(svg, (sx + ROCK_SPRING_DX, sy + ROCK_SPRING_DY));
    let mut scattered = chosen[1..].to_vec();
    scattered.sort_by(|a, b| (a.0 + a.1).partial_cmp(&(b.0 + b.1)).unwrap());
    for &(x, y) in &scattered {
        draw_rock(svg, surf.surface_point(x, y, 0.0));
    }
    chosen.len()
}

/// A small faceted boulder on the ground at `base`: a shadow (left) facet and
/// a sun-lit (right) facet split by a ridge, matching the trees' lit/shadow
/// treatment. The right side is lit because the sun sits upper-right.
fn draw_rock(svg: &mut Svg, base: (f64, f64)) {
    let (bx, by) = base;
    let (w, h) = (ROCK_W, ROCK_H);
    let top = (bx - 0.10 * w, by - h);
    let ul = (bx - 0.95 * w, by - 0.50 * h);
    let ll = (bx - 0.50 * w, by - 0.02 * h);
    let seam = (bx + 0.05 * w, by);
    let lr = (bx + 0.70 * w, by - 0.06 * h);
    let ur = (bx + 0.95 * w, by - 0.55 * h);
    svg.polygon(&[top, ul, ll, seam], ROCK_DARK, None); // shadow (left)
    svg.polygon(&[top, seam, lr, ur], ROCK_LIT, None); // lit (right)
}

fn draw_sun(svg: &mut Svg) {
    let (sx, sy) = project(GRID as f64 * 0.66, 0.0, 1.32);
    svg.circle(sx, sy - 4.0, SUN_R * SUN_HALO, SUN.lerp(PAPER, 0.55));
    svg.circle(sx, sy - 4.0, SUN_R, SUN);
}

// vector helpers
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

// ---------------------------------------------------------------------------
// tiny deterministic PRNG (SplitMix64) — reproducible tree scatter, no deps
// ---------------------------------------------------------------------------

/// Minimal seedable RNG so tree placement is fully determined by `TREE_SEED`
/// and identical across platforms. SplitMix64: fast, well-scrambled, and
/// stateless beyond a single `u64` — enough for sampling a handful of points.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)` (top 53 bits → f64 mantissa).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

// ---------------------------------------------------------------------------
// minimal SVG writer (auto-frames via tracked bounding box)
// ---------------------------------------------------------------------------

struct Svg {
    body: String,
    min_x: f64,
    min_y: f64,
    max_x: f64,
    max_y: f64,
}

impl Svg {
    fn new() -> Self {
        Svg {
            body: String::new(),
            min_x: f64::INFINITY,
            min_y: f64::INFINITY,
            max_x: f64::NEG_INFINITY,
            max_y: f64::NEG_INFINITY,
        }
    }

    fn track(&mut self, x: f64, y: f64) {
        self.min_x = self.min_x.min(x);
        self.min_y = self.min_y.min(y);
        self.max_x = self.max_x.max(x);
        self.max_y = self.max_y.max(y);
    }

    fn points(pts: &[(f64, f64)]) -> String {
        let mut d = String::new();
        for (k, &(x, y)) in pts.iter().enumerate() {
            let _ = write!(d, "{}{:.2},{:.2}", if k == 0 { "" } else { " " }, x, y);
        }
        d
    }

    fn polygon(&mut self, pts: &[(f64, f64)], fill: Rgb, stroke: Option<(Rgb, f64)>) {
        for &(x, y) in pts {
            self.track(x, y);
        }
        let s = match stroke {
            Some((c, w)) => format!(
                r#" stroke="{}" stroke-width="{:.2}" stroke-linejoin="round""#,
                c.hex(),
                w
            ),
            None => String::new(),
        };
        let _ = write!(
            self.body,
            r#"<polygon points="{}" fill="{}"{}/>"#,
            Self::points(pts),
            fill.hex(),
            s
        );
        self.body.push('\n');
    }

    fn circle(&mut self, cx: f64, cy: f64, r: f64, fill: Rgb) {
        self.track(cx - r, cy - r);
        self.track(cx + r, cy + r);
        let _ = write!(
            self.body,
            r#"<circle cx="{:.2}" cy="{:.2}" r="{:.2}" fill="{}"/>"#,
            cx,
            cy,
            r,
            fill.hex()
        );
        self.body.push('\n');
    }

    fn rect(&mut self, x: f64, y: f64, w: f64, h: f64, fill: Rgb) {
        self.track(x, y);
        self.track(x + w, y + h);
        let _ = write!(
            self.body,
            r#"<rect x="{:.2}" y="{:.2}" width="{:.2}" height="{:.2}" fill="{}"/>"#,
            x,
            y,
            w,
            h,
            fill.hex()
        );
        self.body.push('\n');
    }

    fn finish(self) -> String {
        let pad = 30.0;
        let min_x = self.min_x - pad;
        let min_y = self.min_y - pad;
        let w = (self.max_x - self.min_x) + 2.0 * pad;
        let h = (self.max_y - self.min_y) + 2.0 * pad;
        let mut out = String::new();
        let _ = write!(
            out,
            r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="{:.2} {:.2} {:.2} {:.2}" width="{:.0}" height="{:.0}">"#,
            min_x, min_y, w, h, w, h
        );
        out.push('\n');
        // No background rect: the logo renders on a transparent canvas.
        out.push_str(&self.body);
        out.push_str("</svg>\n");
        out
    }
}

// ---------------------------------------------------------------------------
// small color helper
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Rgb(u8, u8, u8);

impl Rgb {
    /// Blend toward `other` by `t ∈ [0, 1]`, interpolating in **Oklab**
    /// (perceptual) rather than raw sRGB, so midpoints stay even instead of
    /// going dark/muddy. Every color blend in the logo — terrain ramp, water
    /// glint, sun halo — goes through here, like R's `colorRamp(space="Lab")`.
    fn lerp(self, other: Rgb, t: f64) -> Rgb {
        let t = t.clamp(0.0, 1.0);
        let (p, q) = (rgb_to_oklab(self), rgb_to_oklab(other));
        oklab_to_rgb([
            p[0] + (q[0] - p[0]) * t,
            p[1] + (q[1] - p[1]) * t,
            p[2] + (q[2] - p[2]) * t,
        ])
    }
    fn shade(self, k: f64) -> Rgb {
        let f = |c: u8| (c as f64 * k).round().clamp(0.0, 255.0) as u8;
        Rgb(f(self.0), f(self.1), f(self.2))
    }
    fn hex(self) -> String {
        format!("#{:02x}{:02x}{:02x}", self.0, self.1, self.2)
    }
}

/// Parse an HTML hex color — `"#52796f"` or `"52796f"` — into an [`Rgb`], so
/// palettes can be written with copy-pasted hex codes. A `const fn`, so it
/// works inside the `const` palette tables. Expects exactly 6 hex digits
/// (optionally `#`-prefixed); non-hex digits read as 0.
const fn hex(s: &str) -> Rgb {
    let b = s.as_bytes();
    let o = b.len() - 6; // 1 when a leading '#' is present, else 0
    Rgb(
        (nibble(b[o]) << 4) | nibble(b[o + 1]),
        (nibble(b[o + 2]) << 4) | nibble(b[o + 3]),
        (nibble(b[o + 4]) << 4) | nibble(b[o + 5]),
    )
}

const fn nibble(c: u8) -> u8 {
    match c {
        b'0'..=b'9' => c - b'0',
        b'a'..=b'f' => c - b'a' + 10,
        b'A'..=b'F' => c - b'A' + 10,
        _ => 0,
    }
}

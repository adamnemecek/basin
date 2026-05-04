// SPA / fully static: no per-route data, just opt into prerendering and
// disable SSR so the wasm module (which uses browser-only APIs) doesn't
// run during the prerender step.
export const prerender = true;
export const ssr = false;

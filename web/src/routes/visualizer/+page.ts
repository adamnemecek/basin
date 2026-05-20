// The visualizer is browser-only: it loads the wasm module and drives a
// `requestAnimationFrame` loop, so it must not be server-rendered. We
// still prerender it — that emits a content-light shell that hydrates
// into the live app via `onMount` in the browser.
export const ssr = false;
export const prerender = true;

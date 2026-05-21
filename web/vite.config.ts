import { sveltekit } from "@sveltejs/kit/vite";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";
import wasm from "vite-plugin-wasm";
import Icons from "unplugin-icons/vite";

export default defineConfig({
    // `~icons/<set>/<name>` imports are resolved at build time by
    // unplugin-icons and compiled to Svelte components, so only the icons
    // actually imported are bundled (no runtime icon library). Icon data
    // comes from `@iconify-json/lucide`.
    plugins: [
        tailwindcss(),
        wasm(),
        sveltekit(),
        Icons({ compiler: "svelte" }),
    ],
    // The wasm-pack output uses ESM `import.meta.url` to find the .wasm
    // sibling. Marking the package as not-pre-bundled lets vite serve it
    // as-is in dev and preserves that resolution.
    optimizeDeps: { exclude: ["$lib/basin-wasm"] },
});

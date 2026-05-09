//! Seedable, wasm-safe RNG used by stochastic solvers.
//!
//! Re-exports a single canonical PRNG ([`ChaCha8Rng`]) for the whole
//! codebase plus the `rand` traits solvers need to drive it. Solvers
//! carry their RNG as a field on `&mut self` and seed it at construction
//! — same seed in, same iterate trajectory out (the reproducibility
//! contract every stochastic solver in basin honors).
//!
//! Why ChaCha8 specifically:
//!
//! - **Wasm-safe without `getrandom`.** Seeding from an explicit `u64`
//!   (or 32-byte seed) needs no entropy source, so basin's
//!   `wasm32-unknown-unknown` build does not pull in any JS feature
//!   flags from `getrandom`. This is load-bearing per the WASM hard
//!   constraint in `AGENTS.md`.
//! - **Pure-Rust, MSRV-friendly.** `rand 0.9` / `rand_chacha 0.9`
//!   compile under basin's MSRV (1.84.1). The 0.10 line requires
//!   edition2024 (Rust 1.85+) and is gated out — see the pin notes in
//!   `Cargo.toml`.
//! - **Statistical quality.** ChaCha8 passes TestU01 / PractRand at the
//!   sample budgets stochastic optimization actually uses.
//!
//! No standard-normal sampling lives here yet; the first caller is
//! S8 (CMA-ES) and the trait shape is best designed alongside it.

pub use rand::{Rng, RngCore, SeedableRng};
pub use rand_chacha::ChaCha8Rng;

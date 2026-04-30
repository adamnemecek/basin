{
  pkgs,
  ...
}:
{
  packages = with pkgs; [
    go-task
    llvmPackages.bintools
    cargo-llvm-cov
    cargo-flamegraph
    cargo-audit
    cargo-deny
    cargo-msrv
    gnuplot
    samply
    pprof
    wasm-pack
    bashInteractive
    perf
    go-task
    quartoMinimal
    shfmt
  ];

  languages = {
    rust = {
      enable = true;
      channel = "stable";
      version = "1.84.1";
      targets = [ "wasm32-unknown-unknown" ];
    };

    javascript = {
      enable = true;
    };

    typescript = {
      enable = true;
    };
  };

  git-hooks = {
    hooks = {
      clippy = {
        enable = true;

        settings = {
          allFeatures = true;
        };
      };

      rustfmt = {
        enable = true;
      };
    };
  };
}

""" Seed / concept_layer sweep driver for counterfactual runs.

The generation scripts (run_concept_ldce.py, run_ldce_baseline.py) are Hydra
entrypoints, so a sweep is just: for each value of one config key, launch the
script with that key overridden and a per-value output_dir, then hand the
resulting directories to src/evaluation/analyze_results.py for aggregation.

This covers the two ablations the reviewer asked for:

  * seeds -- reproducibility / error bars from the stochastic sampler. Uses
    ``fixed_seed=True seed=<v>`` so each run is deterministic and distinct, then
    aggregates with mean +/- std across seeds + a pooled Wilson CI.
  * concept_layer -- how sensitive the flip ratio is to *which* layer the
    concepts are read from (features.34 / .37 / .40 for vgg16_bn).

Generation is GPU-heavy; pass --dry-run first to print the exact commands, or
--analyze-only to (re)aggregate directories that already exist.

Examples
--------
Seed sweep (deterministic), then aggregate::

    python run_ablation_sweep.py \
        --script run_concept_ldce.py --config-name v1_cars_concept \
        --base-output-dir /results/counterfactuals/cars_vgg16bn_concept \
        --param seed --values 0 1 2 --fixed-seed \
        --analyze --export-dir /results/eval/cars_concept_seeds

Layer sweep::

    python run_ablation_sweep.py \
        --script run_concept_ldce.py --config-name v1_cars_concept \
        --base-output-dir /results/counterfactuals/cars_vgg16bn_concept \
        --param concept_layer --values features.34 features.37 features.40 \
        --analyze --export-dir /results/eval/cars_layer_sweep
"""
import argparse
import subprocess
import sys

from src.evaluation.analyze_results import aggregate_seeds, layer_summary, _write_csv, _write_json


def _sanitize(value):
    return str(value).replace("/", "_").replace(" ", "_")


def _output_dir(base, param, value):
    return f"{base}_{param}{_sanitize(value)}"


def build_commands(args):
    """Return list of (value, output_dir, command-list) for the sweep."""
    commands = []
    for value in args.values:
        out_dir = _output_dir(args.base_output_dir, args.param, value)
        cmd = [
            sys.executable, args.script,
            "--config-name", args.config_name,
            f"{args.param}={value}",
            f"output_dir={out_dir}",
        ]
        if args.fixed_seed:
            cmd.append("fixed_seed=True")
        cmd.extend(args.extra)
        commands.append((value, out_dir, cmd))
    return commands


def main():
    parser = argparse.ArgumentParser(description="Sweep a single Hydra config key (seed or concept_layer) and aggregate.")
    parser.add_argument("--script", default="run_concept_ldce.py",
                        help="generation entrypoint (run_concept_ldce.py / run_ldce_baseline.py).")
    parser.add_argument("--config-name", required=True, help="Hydra config name, e.g. v1_cars_concept.")
    parser.add_argument("--base-output-dir", required=True, help="base output_dir; per-value suffix is appended.")
    parser.add_argument("--param", required=True, help="config key to sweep, e.g. 'seed' or 'concept_layer'.")
    parser.add_argument("--values", required=True, nargs="+", help="values to sweep over.")
    parser.add_argument("--fixed-seed", action="store_true",
                        help="append fixed_seed=True (required for deterministic per-seed runs).")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                        help="extra Hydra overrides passed verbatim after all other args.")
    parser.add_argument("--dry-run", action="store_true", help="print commands, do not run.")
    parser.add_argument("--analyze-only", action="store_true", help="skip generation, only aggregate existing dirs.")
    parser.add_argument("--analyze", action="store_true", help="aggregate after generation.")
    parser.add_argument("--export-dir", default=None, help="where to write aggregated CSV/JSON.")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    commands = build_commands(args)

    if not args.analyze_only:
        for value, out_dir, cmd in commands:
            printable = " ".join(cmd)
            if args.dry_run:
                print(printable)
                continue
            print(f"\n>>> {args.param}={value} -> {out_dir}\n    {printable}")
            subprocess.run(cmd, check=True)

    if args.dry_run:
        return

    if args.analyze or args.analyze_only:
        out_dirs = [out_dir for _, out_dir, _ in commands]
        if args.param == "seed":
            summary, per_seed = aggregate_seeds(out_dirs, alpha=args.alpha)
            print("\n=== seed aggregation ===")
            print(f"flip ratio across seeds: {summary['flip_ratio_mean']:.4f} +/- {summary['flip_ratio_std']:.4f}")
            print(f"pooled flip ratio      : {summary['pooled_flip_ratio']:.4f} "
                  f"[{summary['pooled_ci_lo']:.4f}, {summary['pooled_ci_hi']:.4f}]")
            if args.export_dir:
                _write_json(f"{args.export_dir}/seed_summary.json", summary)
                _write_csv(f"{args.export_dir}/per_seed.csv", per_seed)
        else:
            layer_to_path = {_sanitize(v): d for v, d in zip(args.values, out_dirs)}
            sensitivity, rows = layer_summary(layer_to_path, alpha=args.alpha)
            print("\n=== layer sensitivity ===")
            print(f"flip ratio {sensitivity['flip_ratio_mean']:.4f} +/- {sensitivity['flip_ratio_std']:.4f} "
                  f"(range {sensitivity['flip_ratio_range']:.4f})")
            for row in rows:
                print(f"  {row['concept_layer']:14s} n={row['n_samples']:4d} "
                      f"fr={row['flip_ratio']:.4f} [{row['ci_lo']:.4f}, {row['ci_hi']:.4f}]")
            if args.export_dir:
                _write_json(f"{args.export_dir}/layer_sensitivity.json", sensitivity)
                _write_csv(f"{args.export_dir}/per_layer.csv", rows)


if __name__ == "__main__":
    main()

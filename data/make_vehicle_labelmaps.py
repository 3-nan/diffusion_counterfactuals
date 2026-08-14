"""Emit ``<name>_idx_to_label.json`` for the fine-grained vehicle datasets.

Same shape as ``pets_idx_to_label.json`` / ``flowers_idx_to_label.json``:
a flat ``{"<class_idx>": "<readable name>"}`` map. Unlike the Flowers file
(which is 1-indexed), these are **0-indexed**, matching the Pets file and the
labels the loaders return.

The names come straight out of the dataset metadata, so run this once against
your actual download rather than hand-maintaining the lists -- Stanford Cars
alone has 196 entries whose order is defined by ``cars_meta.mat``.

Usage
-----
    python make_vehicle_labelmaps.py \
        --cars-root   /data/stanford_cars \
        --compcars-root /data/CompCars \
        --boxcars-root  /data/BoxCars116k \
        --out-dir     ./data
"""
import argparse
import json
import os
import sys


def dump(mapping, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}_idx_to_label.json")
    with open(path, "w") as f:
        json.dump({str(i): n for i, n in enumerate(mapping)}, f, indent=4)
    print(f"  wrote {len(mapping):5d} classes -> {path}")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cars-root")
    p.add_argument("--compcars-root")
    p.add_argument("--boxcars-root")
    p.add_argument("--compcars-subset", default="web", choices=["web", "sv"])
    p.add_argument("--compcars-level", default="model",
                   choices=["model", "make", "type"])
    p.add_argument("--boxcars-part", default="hard",
                   choices=["hard", "medium", "body", "make", "model", "submodel"])
    p.add_argument("--out-dir", default=".")
    args = p.parse_args()

    if not any([args.cars_root, args.compcars_root, args.boxcars_root]):
        p.error("give at least one of --cars-root / --compcars-root / --boxcars-root")

    # Imported lazily so the script still runs with only one dataset present.
    from data.datasets import StanfordCars, CompCars, BoxCars116k

    if args.cars_root:
        print("Stanford Cars:")
        ds = StanfordCars(root=args.cars_root, transform=None, split="test")
        dump(ds.get_class_names(), args.out_dir, "cars")

    if args.compcars_root:
        print(f"CompCars ({args.compcars_subset}, {args.compcars_level}):")
        ds = CompCars(root=args.compcars_root, transform=None, split="test",
                      subset=args.compcars_subset, label_level=args.compcars_level)
        dump(ds.get_class_names(), args.out_dir, "compcars")

    if args.boxcars_root:
        print(f"BoxCars116k ({args.boxcars_part}):")
        ds = BoxCars116k(root=args.boxcars_root, transform=None, split="test",
                         part=args.boxcars_part)
        dump(ds.get_class_names(), args.out_dir, "boxcars")


if __name__ == "__main__":
    sys.exit(main())

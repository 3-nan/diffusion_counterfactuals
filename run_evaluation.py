
import argparse

from src.evaluation.compute_fid import compute_fid
from src.evaluation.compute_lpnorms import compute_lp_norms
from src.evaluation.compute_validity_metrics import compute_validity_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', required=True, type=str, default='/path/to/experiment', help='the path of the experiment')
    parser.add_argument('--sfid', action="store_true")
    parser.add_argument('--sfid_splits', type=int, default=2)
    args=parser.parse_args()

    fid = compute_fid(args)
    print("sFID:" if args.sfid else "FID:", fid)

    l1, l2 = compute_lp_norms(args)
    print(f"L1: {l1}")
    print(f"L2: {l2}")

    flip_ratio, confidence = compute_validity_metrics(args)

    print(f"Flip ratio: {flip_ratio}")
    print(f"Confidence: {confidence}")

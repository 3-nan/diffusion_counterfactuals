
import os
import argparse
import shutil

from src.evaluation.compute_fid import compute_fid
from src.evaluation.compute_lpnorms import compute_lp_norms
from src.evaluation.compute_validity_metrics import compute_validity_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', required=True, type=str, default='/path/to/experiment', help='the path of the experiment')
    parser.add_argument('--sfid', action="store_true")
    parser.add_argument('--sfid_splits', type=int, default=2)
    args=parser.parse_args()

    outpath = args.output_path

    # shutil.rmtree(outpath + str(1) + '/')
    # shutil.rmtree(outpath + str(100300) + '/')

    for n_concepts in ['']:

    # for n_concepts in [1, 10, 20, 50, 100, 200, 300]:

        args.output_path = outpath + str(n_concepts) + '/'
        print(args.output_path)


        if os.path.isdir(args.output_path):

            print('######################################################')


            fid = compute_fid(args)
            print("sFID:" if args.sfid else "FID:", fid)

            l1, l2, mse = compute_lp_norms(args)
            print(f"L1: {l1}")
            print(f"L2: {l2}")
            print(f"MSE: {mse}")

            flip_ratio, confidence = compute_validity_metrics(args)

            print(f"Flip ratio: {flip_ratio}")
            print(f"Confidence: {confidence}")

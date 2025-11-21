import argparse
import os
import glob
from tqdm import tqdm
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', required=True, type=str, default='/path/to/experiment', help='the path of the experiment')
    # parser.add_argument('--sfid', action="store_true")
    # parser.add_argument('--sfid_splits', type=int, default=2)
    args=parser.parse_args()

    outpath = args.output_path

    counter = 0
    for bucket_folder in sorted(glob.glob(args.output_path + "/bucket*")):
        for pth_file in tqdm(sorted(glob.glob(args.output_path + "/bucket*/*.pth")), leave=False):
            if not os.path.basename(pth_file)[:-4].isdigit():
                continue
            data = torch.load(pth_file, map_location="cpu")

            if data['target'] == data['out_pred'] and data['unique_id'] < 46: # and data['unique_id'] in [252, 343, 434, 581, 910, 938, 945, 952, 973, 1022]:
                print(data['unique_id'])
                print(data['source'])
                print(data['target'])
                # print(data['in_pred'])
                # print(data['out_pred'])

            # if data['out_pred'] >= 0.4:
                # print(data['unique_id'])


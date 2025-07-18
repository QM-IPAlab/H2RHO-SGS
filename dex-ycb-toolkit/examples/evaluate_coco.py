# DexYCB Toolkit
# Copyright (C) 2021 NVIDIA Corporation
# Licensed under the GNU General Public License v3.0 [see LICENSE for details]

"""Example of running COCO evaluation."""

import argparse
import os

from dex_ycb_toolkit.coco_eval import COCOEvaluator


def parse_args():
  parser = argparse.ArgumentParser(description='Run COCO evaluation.')
  parser.add_argument('--name', help='Dataset name', default=None, type=str)
  parser.add_argument('--res_file',
                      help='Path to result file',
                      default=None,
                      type=str)
  parser.add_argument('--out_dir',
                      help='Directory to save eval output',
                      default=None,
                      type=str)
  args = parser.parse_args()
  return args


def main():
  args = parse_args()

  if args.name is None and args.res_file is None:
    args.name = 's0_test'
    current_file_path = os.path.abspath(__file__)
    # 获取父目录
    parent_directory = os.path.dirname(current_file_path)
    # 获取父目录的父目录
    grandparent_directory = os.path.dirname(parent_directory)
    args.res_file = os.path.join(
        grandparent_directory, "results",
        "example_results_coco_{}.json".format(args.name))
  print(f'--------------args.name :{args.name},---------args.res_file {args.res_file}')
  coco_eval = COCOEvaluator(args.name)
  coco_eval.evaluate(args.res_file, out_dir=args.out_dir)


if __name__ == '__main__':
  main()

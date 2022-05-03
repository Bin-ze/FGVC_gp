#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : 离线验证阶段启动入口

import sys
import logging
import argparse

from interface import Tester


def main(dataset_path, model_path, used_model, device="cpu", work_dir="./runs/", args=None):
    tester = Tester(dataset_path=dataset_path,
                    model_path=model_path,
                    used_model=used_model,
                    device=device,
                    work_dir=work_dir,
                    args=args)
    tester.test()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        stream=sys.stdout,
                        format="%(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/data/CUB_200")
    parser.add_argument("--model_path", type=str, default=r"/app/tianji/runs/models/91e2c5a9-35e4-45ee-90e4-c53d85558bbc")
    parser.add_argument("--used_model", type=str, default="FGVC_PIM_autotables-2503-train-1")
    parser.add_argument("--device", type=str, default='cuda')

    #model default set
    parser.add_argument("--pretrained_path",
                        default=None,
                        type=str)
    parser.add_argument("--val_root",
                        default=None, type=str)
    parser.add_argument("--data_size", default=384, type=int)
    parser.add_argument("--num_rows", default=0, type=int)
    parser.add_argument("--num_cols", default=0, type=int)
    parser.add_argument("--sub_data_size", default=32, type=int)

    parser.add_argument("--model_name", default="swin-vit-p4w12", type=str,
                        choices=["efficientnet-b7", 'resnet-50', 'vit-b16', 'swin-vit-p4w12'])
    parser.add_argument("--optimizer_name", default="sgd", type=str,
                        choices=["sgd", 'adamw'])

    parser.add_argument("--use_fpn", default=True, type=bool)
    parser.add_argument("--use_ori", default=False, type=bool)
    parser.add_argument("--use_gcn", default=True, type=bool)
    parser.add_argument("--use_layers",
                        default=[True, True, True, True], type=list)
    parser.add_argument("--use_selections",
                        default=[True, True, True, True], type=list)
    # [2048, 512, 128, 32] for CUB200-2011
    # [256, 128, 64, 32] for NABirds
    parser.add_argument("--num_selects",
                        default=[2048, 512, 128, 32], type=list)
    parser.add_argument("--global_feature_dim", default=1536, type=int)

    # loader
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--batch_size", default=8, type=int)

    # about model building
    parser.add_argument("--num_classes", default=200, type=int)

    parser.add_argument("--test_global_top_confs", default=[0, 1, 2, 3, 4, 5], type=list)
    args = parser.parse_args()

    main(args.dataset_path, model_path=args.model_path, used_model=args.used_model,
         device=args.device, work_dir='/app/tianji', args=args)

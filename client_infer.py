#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : 推理阶段启动入口


import sys
import logging
import argparse

from interface import Infer
from interface.infer_service import InferService


def start_service(model_path, used_model, device, service_port, work_dir='',
                  service_config=None,args=None):
    if not service_config:
        service_config = {
            "service_route": "autotable/predict",
            "service_port": service_port,
            "app_name": "{}_{}".format("autotable", service_port)
        }

    infer_interface = Infer(model_path=model_path, work_dir=work_dir,
                            used_model=used_model, device=device, args=args)
    infer_service = InferService(infer_interface=infer_interface, service_config=service_config, work_dir=work_dir)
    infer_service.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        stream=sys.stdout,
                        format="%(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=r"/app/tianji/runs/models/91e2c5a9-35e4-45ee-90e4-c53d85558bbc")
    parser.add_argument("--used_model", type=str, default="FGVC_PIM_autotables-2503-train-1")
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--service_port", type=int, default=8080)

    #model default set
    parser.add_argument("--pretrained_path",
                        default=None,
                        type=str)
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
    # about model building
    parser.add_argument("--num_classes", default=2, type=int)
    parser.add_argument("--test_global_top_confs", default=[0, 1, 2, 3, 4, 5], type=list)
    args = parser.parse_args()
    logging.info("args: {}".format(args))

    start_service(model_path=args.model_path,
                  used_model=args.used_model,
                  device=args.device,
                  service_port=args.service_port,
                  work_dir='./',
                  args=args)

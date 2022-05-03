#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Desc   : 训练阶段启动入口

import sys
import json
import logging
import argparse
import uuid
from utils.utils import json_decode
from interface import Trainer

debug = True
uuid_value = '91e2c5a9-35e4-45ee-90e4-c53d85558bbc' if debug else uuid.uuid4()


def main(dataset_path, annotation_data, args='None', trial_name='',
         device="cpu", work_dir=""):
    trainer = Trainer(dataset_path=dataset_path,
                      max_epoch=10,
                      device=device,
                      batch_size=4,
                      work_dir=work_dir,
                      trial_name=trial_name,
                      uuid_value=uuid_value,
                      args=args)
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        stream=sys.stdout,
                        format="%(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="/data/CUB_200")
    parser.add_argument("--device", type=str, default="cuda")
    # trial_name用于生成用于保存模型的文件名 = {model_name}_{trial_name}.pth
    parser.add_argument("--trial_name", type=str, default="autotables-2503-train-1")
    parser.add_argument("--annotation_data", type=str, default="{'advance_settings':{}}")

    #default set
    parser.add_argument("--train_root",
                        default=None, type=str)
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
    parser.add_argument("--num_selects",
                        default=[2048, 512, 128, 32], type=list)
    parser.add_argument("--global_feature_dim", default=1536, type=int)

    # loader
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--batch_size", default=4, type=int)

    # about model building
    parser.add_argument("--num_classes", default=200, type=int)

    # abput learning rate scheduler
    parser.add_argument("--warmup_batchs", default=800, type=int)
    parser.add_argument("--no_final_epochs", default=0, type=int)
    parser.add_argument("--max_lr", default=0.0005, type=float)
    parser.add_argument("--update_freq", default=4, type=int)

    parser.add_argument("--wdecay", default=0.0005, type=float)
    parser.add_argument("--nesterov", default=True, type=bool)
    parser.add_argument("--max_epochs", default=50, type=int)

    parser.add_argument("--log_freq", default=50, type=int)

    parser.add_argument("--test_freq", default=2, type=int)
    parser.add_argument("--test_global_top_confs", default=[1, 3, 5], type=list)
    parser.add_argument("--test_select_top_confs", default=[1, 3, 5, 7, 9], type=list)
    args = parser.parse_args()

    main(dataset_path=args.dataset_path,
         trial_name=args.trial_name,
         annotation_data=json_decode(args.annotation_data),
         device=args.device,
         args=args,
         work_dir="/app/tianji")

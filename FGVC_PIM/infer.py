import os
import wandb
import torch
import torch.nn as nn
import json
import numpy as np
import math
import copy
import cv2

from data.dataset import ImageDataset
from config_eval import get_args
import torchvision.transforms as transforms
from PIL import Image
import tqdm


def set_environment(args):
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose([
        transforms.Resize((510, 510), Image.BILINEAR),
        transforms.CenterCrop((args.data_size, args.data_size)),
        transforms.ToTensor(),
        normalize
    ])

    if args.model_name == "efficientnet-b7":
        from models.EfficientNet_FPN import DetEfficientNet
        model = DetEfficientNet(in_size=args.data_size,
                                num_classes=args.num_classes,
                                use_fpn=args.use_fpn,
                                use_ori=args.use_ori,
                                use_gcn=args.use_gcn,
                                use_layers=args.use_layers,
                                use_selections=args.use_selections,
                                num_selects=args.num_selects,
                                global_feature_dim=args.global_feature_dim)
    elif args.model_name == 'resnet-50':
        from models.ResNet50_FPN import DetResNet50
        model = DetResNet50(in_size=args.data_size,
                            num_classes=args.num_classes,
                            use_fpn=args.use_fpn,
                            use_ori=args.use_ori,
                            use_gcn=args.use_gcn,
                            use_layers=args.use_layers,
                            use_selections=args.use_selections,
                            num_selects=args.num_selects,
                            global_feature_dim=args.global_feature_dim)
    elif args.model_name == 'vit-b16':
        from models.Vitb16_FPN import VitB16
        model = VitB16(in_size=args.data_size,
                       num_classes=args.num_classes,
                       use_fpn=args.use_fpn,
                       use_ori=args.use_ori,
                       use_gcn=args.use_gcn,
                       use_layers=args.use_layers,
                       use_selections=args.use_selections,
                       num_selects=args.num_selects,
                       global_feature_dim=args.global_feature_dim)
    elif args.model_name == 'swin-vit-p4w12':
        from models.SwinVit12 import SwinVit12
        model = SwinVit12(
            in_size=args.data_size,
            num_classes=args.num_classes,
            use_fpn=args.use_fpn,
            use_ori=args.use_ori,
            use_gcn=args.use_gcn,
            use_layers=args.use_layers,
            use_selections=args.use_selections,
            num_selects=args.num_selects,
            global_feature_dim=args.global_feature_dim
        )

    checkpoint = torch.load(args.pretrained_path)
    model.load_state_dict(checkpoint['best_ckpt'])
    model.to(args.device)

    return transform, model


def infer(model, img, transforms, args):
    json_path = '../class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)
    img = img[:, :, ::-1]  # BGR to RGB.

    # to PIL.Image
    img = Image.fromarray(img)
    img = transforms(img)
    img = img.unsqueeze(dim=0)
    img = img.to(args.device)

    model.eval()
    with torch.no_grad():
        pred = torch.squeeze(model.infer(img)).cpu()
        predict = torch.softmax(pred, dim=0)
        predict_cla = torch.argmax(predict).numpy()
        print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                     predict[predict_cla].numpy())
        print(print_res)

if __name__ == "__main__":
    args = get_args()
    transform, model = set_environment(args)
    path='/home/liuguangcan/internship/Fine-Grained-Classification/FGVC_PIM/datas/CUB_200/valid/images'
    for i in sorted(os.listdir(path)):
        print(i)
        for j in  sorted(os.listdir(os.path.join(path,i))):
            print(j)
            img = os.path.join(path,i,j)
            img = cv2.imread(img)
            infer(model, img, transform, args)

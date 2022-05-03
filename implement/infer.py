# @Author  : guozebin (guozebin@fuzhi.ai)
# @Desc    :
import torch
import gc
import os
import logging
import json

from utils.utils import create_not_exist_path, set_seed
import torchvision.transforms as transforms
from PIL import Image

class Infer(object):
    def __init__(self, model_path='', used_model='',
                 device='cpu', work_dir='',args=None):
        # const
        self.work_dir = work_dir
        self.model_path = model_path
        self.used_model = used_model.replace('.pth', '').replace('.pt', '') + '.pth'
        self.model_load_path = f'{self.model_path}/{self.used_model}'
        self.work_dir = work_dir.replace('\\', '/') if work_dir else './'
        self.infer_result_save_path = f'{self.work_dir}/runs/infer'
        self.extract_frame_save_path = f'{work_dir}/runs/dataset/infer_video'
        create_not_exist_path(self.infer_result_save_path)
        create_not_exist_path(self.extract_frame_save_path)
        self.device = torch.device(device)
        self.time_use = {}
        set_seed(42)

        # model
        self.model_name = 'FGVC_PIM'

        # load model
        self.args = args
        self.transforms, self.model = self.set_environment(self.args)

    def set_environment(self,args):
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
        checkpoint = torch.load(self.model_load_path)
        args.num_classes = checkpoint['num_classes']
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

        model.load_state_dict(checkpoint['best_ckpt'])
        model.to(self.device)

        return transform, model

    def infer(self, img):
        json_path = '/app/tianji/class_indices.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
        json_file = open(json_path, "r")
        class_indict = json.load(json_file)

        # to PIL.Image
        img = Image.fromarray(img)
        img = self.transforms(img)
        img = img.unsqueeze(dim=0)
        img = img.to(self.device)
        self.model.eval()
        with torch.no_grad():
            pred = torch.squeeze(self.model.infer(img)).cpu()
            predict = torch.softmax(pred, dim=0)
            predict_cla = torch.argmax(predict).numpy()
            result = class_indict[str(predict_cla)]+'_'+"{:.4}".format(predict[predict_cla].numpy())
        logging.info('infer finished ...')

        return result

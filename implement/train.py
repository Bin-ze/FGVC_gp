# @Author  : guozebin (guozebin@fuzhi.ai)
# @Desc    :

import logging
import torch
import gc
import time
import os
import json
import numpy as np
import math
import copy

from utils.utils import create_not_exist_path, write_txt_file, set_seed
from utils.const import KEEP_DIGITS_NUM
from interface.gp_output import gen_gp_train_output
from data.dataset import ImageDataset

class Trainer(object):
    def __init__(self, dataset_path='', max_epoch=10, device='cpu',
                 batch_size=2, work_dir='', trial_name='', uuid_value='',args=None):
        # const
        self.work_dir = work_dir.replace('\\', '/') if work_dir else './'
        self.model_save_path = f'{work_dir}/runs/models/{uuid_value}'
        self.metric_output_path = f'{work_dir}/runs/metric/trial.txt'
        create_not_exist_path(self.model_save_path)
        self.dataset_path = dataset_path if not dataset_path.endswith('/') else dataset_path[:-1]

        self.trial_name = trial_name
        self.device = torch.device(device)
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.time_use = {}
        self.valid_metric_info = {}
        self.train_metric_info = {}
        set_seed(42)
        # model
        self.model_name = 'FGVC_PIM'
        self.valid_per_epoch = 1
        self.metric_type = 'acc'
        self.args = self.set_config(args)


        # train
        self.best_ckpt = [0, {}]

    def set_config(self,args):
        args.train_root = self.dataset_path+'/train/images/'
        args.val_root = self.dataset_path+'/valid/images/'
        args.max_epochs = self.max_epoch
        args.batch_size = self.batch_size
        args.num_classes = len(os.listdir(args.train_root))
        args.device =self.device

        return args

    def get_lr(self,optimizer):
        for param_group in optimizer.param_groups:
            if param_group["lr"] is not None:
                return param_group["lr"]

    def adjust_lr(self,iteration, optimizer, schedule):
        for param_group in optimizer.param_groups:
            param_group["lr"] = schedule[iteration]

    def set_environment(self,args):

        train_set = ImageDataset(istrain=True,
                                 root=args.train_root,
                                 data_size=args.data_size,
                                 return_index=True)

        train_loader = torch.utils.data.DataLoader(train_set, num_workers=args.num_workers, shuffle=True,
                                                   batch_size=args.batch_size)

        test_set = ImageDataset(istrain=False,
                                root=args.val_root,
                                data_size=args.data_size,
                                return_index=False)

        test_loader = torch.utils.data.DataLoader(test_set, num_workers=1, shuffle=False, batch_size=args.batch_size)

        print("train samples: {}, train batchs: {}".format(len(train_set), len(train_loader)))
        print("test samples: {}, test batchs: {}".format(len(test_set), len(test_loader)))

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

        model.to(args.device)

        if args.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=args.max_lr,
                                        nesterov=args.nesterov,
                                        momentum=0.9,
                                        weight_decay=args.wdecay)
        elif args.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.max_lr)

        # lr schedule
        total_batchs = args.max_epochs * len(train_loader)
        iters = np.arange(total_batchs - args.warmup_batchs)
        schedule = np.array([1e-12 + 0.5 * (args.max_lr - 1e-12) * (1 + \
                                                                    math.cos(math.pi * t / total_batchs)) for t in
                             iters])

        # schedule = args.max_lr * np.array([math.cos(7*math.pi*t / (16*total_batchs)) for t in iters])
        if args.warmup_batchs > 0:
            warmup_lr_schedule = np.linspace(1e-9, args.max_lr, args.warmup_batchs)
            schedule = np.concatenate((warmup_lr_schedule, schedule))

        return train_loader, test_loader, model, optimizer, schedule

    def valid(self,args, model, test_loader):

        total = 0

        accuracys = {"sum": 0}
        global_accs_template = {}
        for i in args.test_global_top_confs:
            global_accs_template["global_top" + str(i)] = 0

        select_accs_template = {}
        for i in args.test_select_top_confs:
            select_accs_template["select_top" + str(i)] = 0

        model.eval()
        with torch.no_grad():
            for batch_id, (datas, labels) in enumerate(test_loader):

                """ data preparation """
                batch_size = labels.size(0)
                total += batch_size

                datas, labels = datas.to(args.device), labels.to(args.device)

                """ forward """
                _, batch_accs, batch_logits = model(datas, labels, return_preds=True)

                for name in batch_accs:
                    store_name = name
                    if store_name not in accuracys:
                        accuracys[store_name] = 0
                    accuracys[store_name] += batch_accs[name] * batch_size

                labels = labels.cpu()

                # = = = = = output post-processing. = = = = =
                # = = = softmax = = =
                for name in batch_logits:
                    if name in ["ori"]:
                        batch_logits[name] = torch.softmax(batch_logits[name], dim=1)
                    elif "l_" in name:
                        batch_logits[name] = torch.softmax(batch_logits[name].mean(2).mean(2), dim=-1)
                    elif "select" in name:
                        batch_logits[name] = torch.softmax(batch_logits[name], dim=-1)
                    elif name in ["gcn"]:
                        batch_logits[name] = torch.softmax(batch_logits[name], dim=-1)

                    batch_logits[name] = batch_logits[name].cpu()

                """
                ori
                gcn
                layers
                selecteds (sorted)
                """
                # 1. ========= sum (average) =========
                logit_sum = None
                for name in batch_logits:
                    # = = = skip = = =
                    if "select" in name:
                        continue

                    if logit_sum is None:
                        logit_sum = batch_logits[name]
                    else:
                        logit_sum += batch_logits[name]

                accuracys["sum"] = torch.max(logit_sum, dim=-1)[1].eq(labels).sum().item()

                # 2. ========= vote =========
                pred_counter = torch.zeros([batch_size, args.num_classes])
                pred_counter_select = torch.zeros([batch_size, args.num_classes])
                for name in batch_logits:
                    if "selected" in name:
                        """
                        [B, S, C]
                        """
                        preds = torch.max(batch_logits[name], dim=-1)[1]
                        for bid in range(batch_size):
                            batch_pred = preds[bid]
                            for pred in batch_pred:
                                pred_cls = pred.item()
                                pred_counter_select[bid][pred_cls] += 1
                        continue

                    """
                    [B, C]
                    """
                    preds = torch.max(batch_logits[name], dim=-1)[1]
                    for bid in range(batch_size):
                        pred_cls = preds[bid]
                        pred_counter[bid][pred_cls] += 1
                        pred_counter_select[bid][pred_cls] += 1

                vote = torch.max(pred_counter, dim=-1)[1]
                vote_select = torch.max(pred_counter_select, dim=-1)[1]

                accuracys["vote"] = vote.eq(labels).sum().item()
                accuracys["vote_select"] = vote_select.eq(labels).sum().item()

                # 3. ========= bigger confidence prediction =========
                # 3.1 === global ===
                global_confidences = []
                # global_predictions = []
                global_features = []
                for name in batch_logits:
                    if "select" in name:
                        continue
                    confs, preds = torch.max(batch_logits[name], dim=-1)
                    global_confidences.append(confs.unsqueeze(1))
                    global_features.append(batch_logits[name].unsqueeze(1))

                global_confidences = torch.cat(global_confidences, dim=1)  # B, S
                global_features = torch.cat(global_features, dim=1)  # B, S, C

                area_size = global_confidences.size(1)

                # tmp variables.
                tmp_g_accs = copy.deepcopy(global_accs_template)
                # get batch acuracy
                for bid in range(batch_size):
                    feature_sum = None
                    ids = torch.sort(global_confidences[bid], dim=-1)[1]  # S
                    for i in range(args.test_global_top_confs[-1]):
                        if i >= ids.size(0):
                            break
                        fid = ids[i]
                        if feature_sum is None:
                            feature_sum = global_features[bid][fid]
                        else:
                            feature_sum += global_features[bid][fid]

                        if i in args.test_global_top_confs:
                            if torch.max(feature_sum, dim=-1)[1] == labels[bid]:
                                tmp_g_accs["global_top" + str(i)] += 1

                for name in tmp_g_accs:
                    if name not in accuracys:
                        accuracys[name] = 0
                    accuracys[name] += tmp_g_accs[name]

                # 3.2 === select ===
                tmp_s_accs = copy.deepcopy(select_accs_template)
                select_confs = []
                select_features = []
                for name in batch_logits:
                    if "selected" not in name:
                        continue
                    features = batch_logits[name]  # [B, S, C]
                    conf, pred = torch.max(features, dim=-1)
                    select_confs.append(conf)
                    select_features.append(features)

                if len(select_confs) > 0:
                    select_confs = torch.cat(select_confs, dim=1)
                    select_features = torch.cat(select_features, dim=1)

                    # tmp variables.
                    tmp_s_accs = copy.deepcopy(select_accs_template)
                    # get batch acuracy
                    for bid in range(batch_size):
                        feature_sum = None
                        ids = torch.sort(select_confs[bid], dim=-1)[1]  # S
                        for i in range(args.test_select_top_confs[-1]):
                            if i >= ids.size(0):
                                break
                            fid = ids[i]
                            if feature_sum is None:
                                feature_sum = select_features[bid][fid]
                            else:
                                feature_sum += select_features[bid][fid]

                            if i in args.test_select_top_confs:
                                if torch.max(feature_sum, dim=-1)[1] == labels[bid]:
                                    tmp_s_accs["select_top" + str(i)] += 1

                    for name in tmp_s_accs:
                        if name not in accuracys:
                            accuracys[name] = 0
                        accuracys[name] += tmp_s_accs[name]
        # acc_final, acc_l1, acc_l2, acc_l3, acc_gcn
        msg = {}
        for name in accuracys:
            msg["test_acc/test_acc_" + name] = 100 * accuracys[name] / total
        # wandb.log(msg)
        print(msg)

        best_acc = -1
        for name in msg:
            if msg[name] > best_acc:
                best_acc = msg[name]

        return best_acc

    def train_epoch(self, args, epoch, model, scaler, optimizer, schedules, train_loader):

        model.train()

        optimizer.zero_grad()
        for batch_id, (ids, datas, labels) in enumerate(train_loader):

            # adjust learning rate
            iterations = epoch * len(train_loader) + batch_id
            self.adjust_lr(iterations, optimizer, schedules)

            """ data preparation """
            # batch size (full)
            batch_size = labels.size(0)

            """ forward """
            datas, labels = datas.to(args.device), labels.to(args.device)

            with torch.cuda.amp.autocast():
                losses, accuracys = model(datas, labels)

                loss = 0
                for name in losses:
                    if "selected" in name:
                        loss += losses[name]
                    if "ori" in name:
                        loss += losses[name]
                    else:
                        loss += losses[name]

                loss /= args.update_freq

            scaler.scale(loss).backward()

            if (batch_id + 1) % args.update_freq == 0:
                scaler.step(optimizer)
                scaler.update()  # next batch.
                optimizer.zero_grad()

            """ log """
            if (batch_id + 1) % args.log_freq == 0:
                msg = {
                    "train_info/epoch": epoch + 1,
                    "train_loss/loss": loss,
                    "train_info/lr": self.get_lr(optimizer)
                }
                for name in accuracys:
                    msg["train_acc/train_acc_" + name] = 100 * accuracys[name]

                for name in losses:
                    msg["train_loss/train_loss_" + name] = losses[name]

                print(msg)
    def train(self):
        total_time = 0
        train_time = 0
        valid_time = 0

        # loop
        total_time_tic = time.time()
        args = self.args
        train_loader, test_loader, model, optimizer, schedule = self.set_environment(args)
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(args.max_epochs):

            """ train model """
            train_time_tic = time.time()
            self.train_epoch(args, epoch, model, scaler, optimizer, schedule, train_loader)
            train_time += (time.time() - train_time_tic)
            valid_time_tic = time.time()
            # control test or not
            if epoch > args.max_epochs * 0.9:
                args.test_freq = 1
            elif epoch > args.max_epochs * 0.8:
                args.test_freq = 2
            elif epoch > args.max_epochs * 0.6:
                args.test_freq = 4

            if epoch == 0 or (epoch + 1) % args.test_freq == 0:
                test_acc = self.valid(args, model, test_loader)
                print("valid_acc:", test_acc)
                valid_time += (time.time() - valid_time_tic)
                if test_acc > self.best_ckpt[0]:
                    self.best_ckpt = [test_acc, model.state_dict()]
                    self.valid_metric_info['acc'] = round(test_acc, KEEP_DIGITS_NUM)
            if not self.best_ckpt[-1]:
                    self.best_ckpt = [test_acc, model.state_dict()]


        self.time_use = {
            'total': round(time.time() - total_time_tic, KEEP_DIGITS_NUM),
            'train_time': round(train_time, KEEP_DIGITS_NUM),
            'valid_time': round(valid_time, KEEP_DIGITS_NUM),
        }

        # output metric
        self.output_metric()
        logging.info('train success ...')

    def output_metric(self):
        metric_info = gen_gp_train_output(
            model_name=self.model_name,
            trial_name=self.trial_name,
            metric=self.valid_metric_info,
            model_path=self.model_save_path,
            metric_type=self.metric_type,
            time_use=self.time_use
        )

        # write metric to path
        write_txt_file(metric_info, txt_save_path=self.metric_output_path)
        logging.info(f'save trial.txt succeed -> {self.metric_output_path}')
        logging.info(metric_info)


    def save_model(self):
        create_not_exist_path(f'{self.model_save_path}/')
        file_name = f'{self.model_name}_{self.trial_name}.pth'
        ckpt = dict(
            best_ckpt=self.best_ckpt[-1],
            metric_value=self.best_ckpt[0],
            metric_type=self.metric_type,
            num_classes=self.args.num_classes
        )
        torch.save(ckpt, f'{self.model_save_path}/{file_name}', _use_new_zipfile_serialization=False)
        logging.info(f'success save model, {file_name}')


if __name__ == '__main__':
    trainer = Trainer(dataset_path='')
    trainer.train()

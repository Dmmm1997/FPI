# -*- coding: utf-8 -*-
from __future__ import print_function, division
import argparse
import torch
from py._builtin import enumerate

from torch.autograd import Variable
from torch.cuda.amp import GradScaler
import torch.backends.cudnn as cudnn
import time
from optimizers.make_optimizer import make_optimizer
from models.model import make_model
from datasets.make_dataloader import make_dataset
from tool.utils_server import save_network, copyfiles2checkpoints,get_logger
from tool.evaltools import evaluate
import warnings
from losses.balanceLoss import LossFunc
from tqdm import tqdm
import numpy as np
import cv2
from torch import optim
import random

warnings.filterwarnings("ignore")


def get_parse():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--name', default="test", type=str, help='output model name')
    parser.add_argument('--data_dir', default='/home/dmmm/FPI', type=str, help='training dir path')
    parser.add_argument('--num_worker', default=0, type=int, help='')
    parser.add_argument('--batchsize', default=16, type=int, help='batchsize')
    parser.add_argument('--warm_epoch', default=0, type=int, help='the first K epoch that needs warm up')
    parser.add_argument('--lr', default=0.00005, type=float, help='learning rate')
    parser.add_argument('--autocast', action='store_true', default=True, help='use mix precision')
    parser.add_argument('--num_epochs', default=16, type=int, help='')
    parser.add_argument('--save_epochs', default=2, type=int, help='')
    parser.add_argument('--log_iter', default=50, type=int, help='')
    parser.add_argument('--UAVhw', default=112, type=int, help='')
    parser.add_argument('--Satellitehw', default=400, type=int, help='')
    parser.add_argument('--backbone', default="Pvt-T", type=str, help='')
    parser.add_argument('--padding', default=0, type=float, help='the times of padding for the image size')
    parser.add_argument('--share', default=0, type=int, help='the times of padding for the image size')
    parser.add_argument('--steps', default=[11,15], type=int, help='the times of padding for the image size')
    parser.add_argument('--checkpoints', default="", type=str, help='the times of padding for the image size')
    parser.add_argument('--neg_weight', default=15.0, type=float, help='the times of padding for the image size')
    opt = parser.parse_args()
    opt.UAVhw = [opt.UAVhw,opt.UAVhw]
    opt.Satellitehw = [opt.Satellitehw,opt.Satellitehw]
    return opt


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_model(model, opt,  dataloaders, dataset_sizes):
    use_gpu = opt.use_gpu
    num_epochs = opt.num_epochs
    logger = get_logger("./checkpoints/{}/train.log".format(opt.name))

    since = time.time()
    warm_up = 0.1  # We start from the 0.1*lrRate
    warm_iteration = round(dataset_sizes['satellite'] / opt.batchsize) * opt.warm_epoch  # first 5 epoch

    scaler = GradScaler()
    criterion = LossFunc(opt.centerR,opt.neg_weight)
    logger.info('start training!')

    optimizer, scheduler = make_optimizer(model, opt)

    for epoch in range(num_epochs):
        logger.info('Epoch {}/{}'.format(epoch+1, num_epochs))
        logger.info('-' * 30)

        # Each epoch has a training and validation phase
        model.train(True)  # Set model to training mode
        running_loss = 0.0
        iter_cls_loss = 0.0
        iter_loc_loss = 0.0
        iter_start = time.time()
        iter_loss = 0

        # train
        for iter, (z, x, ratex, ratey) in enumerate(dataloaders["train"]):
            now_batch_size, _, _, _ = z.shape
            total_iters = len(dataloaders["train"])
            if now_batch_size < opt.batchsize:  # skip the last batch
                continue
            if use_gpu:
                z = Variable(z.cuda().detach())
                x = Variable(x.cuda().detach())
            else:
                z, x = Variable(z), Variable(x)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(z, x)  # satellite and drone
            cls_loss, loc_loss = criterion(outputs, [ratex, ratey])
            loc_loss = loc_loss
            loss = cls_loss + loc_loss
            # backward + optimize only if in training phase
            if epoch < opt.warm_epoch:
                warm_up = min(1.0, warm_up + 0.9 / warm_iteration)
                loss_backward = warm_up*loss
            else:
                loss_backward = loss

            if opt.autocast:
                scaler.scale(loss_backward).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_backward.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * now_batch_size
            iter_loss += loss.item() * now_batch_size
            iter_cls_loss += cls_loss.item() * now_batch_size
            iter_loc_loss += loc_loss.item() * now_batch_size

            if (iter + 1) % opt.log_iter == 0:
                time_elapsed_part = time.time() - iter_start
                iter_loss = iter_loss/opt.log_iter/now_batch_size
                iter_cls_loss = iter_cls_loss/opt.log_iter/now_batch_size
                iter_loc_loss = iter_loc_loss/opt.log_iter/now_batch_size

                lr_backbone = optimizer.state_dict()['param_groups'][0]['lr']

                logger.info("[{}/{}] loss: {:.4f} cls_loss: {:.4f} loc_loss:{:.4f} lr_backbone:{:.6f}"
                            "time:{:.0f}m {:.0f}s ".format(iter + 1,
                                                          total_iters,
                                                          iter_loss,
                                                          iter_cls_loss,
                                                          iter_loc_loss,
                                                          lr_backbone,
                                                          time_elapsed_part // 60,
                                                          time_elapsed_part % 60))
                iter_loss = 0.0
                iter_loc_loss = 0.0
                iter_cls_loss = 0.0
                iter_start = time.time()

        epoch_loss = running_loss / dataset_sizes['satellite']

        lr_backbone = optimizer.state_dict()['param_groups'][0]['lr']

        time_elapsed = time.time() - since
        logger.info('Epoch[{}/{}] Loss: {:.4f}  lr_backbone:{:.6f}  time:{:.0f}m {:.0f}s'.format(epoch+1,
                                                                                          num_epochs,
                                                                                          epoch_loss,
                                                                                          lr_backbone,
                                                                                          time_elapsed // 60,
                                                                                          time_elapsed % 60))
        # deep copy the model
        scheduler.step()
        # ----------------------save and test the model------------------------------ #
        if (epoch + 1) % opt.save_epochs == 0 and (epoch+1)>=12:
            save_network(model, opt.name, epoch+1)
            model.eval()
            total_score = 0.0
            total_score_b = 0.0
            start_time = time.time()
            flag_bias = 0
            for uav, satellite, X, Y, _, _ in tqdm(dataloaders["val"]):
                z = uav.cuda()
                x = satellite.cuda()

                response, loc_bias = model(z, x)
                response = torch.sigmoid(response)
                map = response.squeeze().cpu().detach().numpy()

                # kernel = create_hanning_mask(1)
                # map = cv2.filter2D(map, -1, kernel)

                label_XY = np.array([X.squeeze().detach().numpy(), Y.squeeze().detach().numpy()])

                satellite_map = cv2.resize(map, opt.Satellitehw)
                id = np.argmax(satellite_map)
                S_X = int(id // opt.Satellitehw[0])
                S_Y = int(id % opt.Satellitehw[1])
                pred_XY = np.array([S_X, S_Y])
                single_score = evaluate(opt, pred_XY=pred_XY, label_XY=label_XY)
                total_score += single_score
                if loc_bias is not None:
                    flag_bias = 1
                    loc = loc_bias.squeeze().cpu().detach().numpy()
                    id_map = np.argmax(map)
                    S_X_map = int(id_map // map.shape[-1])
                    S_Y_map = int(id_map % map.shape[-1])
                    pred_XY_map = np.array([S_X_map, S_Y_map])
                    pred_XY_b = (pred_XY_map + loc[:, S_X_map, S_Y_map]) * opt.Satellitehw[0] / loc.shape[-1]  # add bias
                    pred_XY_b = np.array(pred_XY_b)
                    single_score_b = evaluate(opt, pred_XY=pred_XY_b, label_XY=label_XY)
                    total_score_b += single_score_b

                # print("pred: " + str(pred_XY) + " label: " +str(label_XY) +" score:{}".format(single_score))

            time_consume = time.time() - start_time
            logger.info("time consume is {}".format(time_consume))

            score = total_score / len(dataloaders["val"])
            logger.info("the final score is {}".format(score))
            if flag_bias:
                score_b = total_score_b / len(dataloaders["val"])
                logger.info("the final score_bias is {}".format(score_b))



if __name__ == '__main__':
    opt = get_parse()
    str_ids = opt.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >= 0:
            gpu_ids.append(gid)

    use_gpu = torch.cuda.is_available()
    opt.use_gpu = use_gpu
    # set gpu ids
    if len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True

    setup_seed(101)
    dataloaders_train, dataset_sizes = make_dataset(opt)
    dataloaders_val = make_dataset(opt,train=False)
    dataloaders = {"train":dataloaders_train,
                    "val":dataloaders_val}

    model = make_model(opt,pretrain=True)

    model = model.cuda()
    # 移动文件到指定文件夹
    copyfiles2checkpoints(opt)

    train_model(model, opt, dataloaders, dataset_sizes)

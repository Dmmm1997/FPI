# -*- coding: utf-8 -*-

from __future__ import print_function, division

import time
from torch.nn.functional import sigmoid
import yaml
import warnings
from models.model import make_model
from tqdm import tqdm
import numpy as np
import torch
import argparse
import cv2
from datasets.SiamUAV import SiamUAV_test

warnings.filterwarnings("ignore")


def get_opt():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--test_data_dir', default='/home/dmmm/SiamUAV/', type=str, help='training dir path')
    parser.add_argument('--num_worker', default=4, type=int, help='')
    parser.add_argument('--checkpoint', default="net_016.pth", type=str, help='')
    parser.add_argument('--k', default=10, type=int, help='')
    parser.add_argument('--filterR', default=1, type=int, help='')
    opt = parser.parse_args()
    config_path = 'opts.yaml'
    with open(config_path, 'r') as stream:
        config = yaml.load(stream)
    opt.UAVhw = config["UAVhw"]
    opt.Satellitehw = config["Satellitehw"]
    opt.share = config["share"]
    opt.backbone = config["backbone"]
    opt.padding = config["padding"]
    opt.centerR = config["centerR"]
    return opt


def create_hanning_mask(center_R):
    hann_window = np.outer(  # np.outer 如果a，b是高维数组，函数会自动将其flatten成1维 ，用来求外积
        np.hanning(center_R+2),
        np.hanning(center_R+2))
    hann_window /= hann_window.sum()
    return hann_window[1:-1,1:-1]


def create_model(opt):
    model = make_model(opt)
    state_dict = torch.load(opt.checkpoint)
    model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()
    return model


def create_dataset(opt):
    dataset_test = SiamUAV_test(opt.test_data_dir, opt)
    dataloaders = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=1,
                                              shuffle=True,
                                              num_workers=opt.num_worker,
                                              pin_memory=True)
    return dataloaders


def evaluate(opt, pred_XY, label_XY):
    pred_X, pred_Y = pred_XY
    label_X, label_Y = label_XY
    x_rate = (pred_X - label_X) / opt.Satellitehw[0]
    y_rate = (pred_Y - label_Y) / opt.Satellitehw[1]
    distance = np.sqrt((np.square(x_rate) + np.square(y_rate)) / 2)  # take the distance to the 0-1
    result = np.exp(-1 * opt.k * distance)
    return result


def test(model, dataloader, opt):
    total_score = 0.0
    total_score_b = 0.0
    flag_bias = 0
    start_time = time.time()
    for uav, satellite, X, Y, _, _ in tqdm(dataloader):
        z = uav.cuda()
        x = satellite.cuda()
        response, loc_bias = model(z, x)
        response = torch.sigmoid(response)
        map = response.squeeze().cpu().detach().numpy()

        # kernel = np.ones((opt.filterR, opt.filterR), np.float32)
        # hanning kernel
        kernel = create_hanning_mask(opt.filterR)
        map = cv2.filter2D(map, -1, kernel)

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
    print("time consume is {}".format(time_consume))

    score = total_score / len(dataloader)
    print("the final score is {}".format(score))
    if flag_bias:
        score_b = total_score_b / len(dataloader)
        print("the final score_bias is {}".format(score_b))

    with open("result.txt", "w") as F:
        F.write("the final score is {}\n".format(score))
        F.write("time consume is {}".format(time_consume))


def main():
    opt = get_opt()
    model = create_model(opt)
    dataloader = create_dataset(opt)
    test(model, dataloader, opt)


if __name__ == '__main__':
    main()

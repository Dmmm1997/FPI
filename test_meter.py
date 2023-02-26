# -*- coding: utf-8 -*-

from __future__ import print_function, division

import json

import time
from torch.nn.functional import sigmoid
import yaml
import warnings
from models.taskflow import make_model
from tqdm import tqdm
import numpy as np
import torch
import argparse
import cv2
from datasets.SiamUAV import SiamUAV_test

warnings.filterwarnings("ignore")

def get_opt():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--test_dir', default='/home/dmmm/Dataset/FPI/FPI2023/test', type=str, help='training dir path')
    parser.add_argument('--num_worker', default=4, type=int, help='')
    parser.add_argument('--checkpoint', default="net_016.pth", type=str, help='')
    parser.add_argument('--k', default=10, type=int, help='')
    parser.add_argument('--filterR', default=1, type=int, help='')
    parser.add_argument('--savename', default="result1.txt", type=str, help='')
    parser.add_argument('--GPS_output_filename', default="GPS_pred_gt.json", type=str, help='')
    parser.add_argument('--mode', default="2019_2022_satellitemap_700-1800_cr0.9_stride100", type=str, help='')
    opt = parser.parse_args()
    config_path = 'opts.yaml'
    with open(config_path, 'r') as stream:
        config = yaml.load(stream)
    for cfg,value in config.items():
        setattr(opt,cfg,value)
    # opt.UAVhw = config["UAVhw"]
    # opt.Satellitehw = config["Satellitehw"]
    # opt.share = config["share"]
    # opt.backbone = config["backbone"]
    # opt.padding = config["padding"]
    # opt.centerR = config["centerR"]
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
    dataset_test = SiamUAV_test(opt)
    dataloaders = torch.utils.data.DataLoader(dataset_test,
                                              batch_size=1,
                                              shuffle=False,
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


def euclideanDistance(query, gallery):
    query = np.array(query, dtype=np.float32)
    gallery = np.array(gallery, dtype=np.float32)
    A = gallery - query
    A_T = A.transpose()
    distance = np.matmul(A, A_T)
    mask = np.eye(distance.shape[0], dtype=np.bool8)
    distance = distance[mask]
    distance = np.sqrt(distance.reshape(-1))
    return distance


def SDM_evaluateSingle(distance,K):
    # maxDistance = max(distance) + 1e-14
    # weight = np.ones(K) - np.log(range(1, K + 1, 1)) / np.log(opts.M * K)
    weight = np.ones(K) - np.array(range(0,K,1))/K
    # m1 = distance / maxDistance
    m2 = 1 / np.exp(distance*5e3)
    m3 = m2 * weight
    result = np.sum(m3) / np.sum(weight)
    return result


def SDM_evaluate_score(opt,UAV_GPS,Satellite_INFO,UAV_image_path,Satellite_image_path,S_X,S_Y):
    # drone/groundtruth GPS info
    drone_GPS_info = [float(UAV_GPS["E"]), float(UAV_GPS["N"])]
    # Satellite_GPS_info format:[tl_E,tl_N,br_E,br_N]
    Satellite_GPS_info = [float(Satellite_INFO["tl_E"]), float(Satellite_INFO["tl_N"]), float(Satellite_INFO["br_E"]),
                          float(Satellite_INFO["br_N"])]
    drone_in_satellite_relative_position = [float(Satellite_INFO["center_distribute_X"]),
                                            float(Satellite_INFO["center_distribute_Y"])]
    mapsize = float(Satellite_INFO["map_size"])
    # pred GPS info
    pred_N = Satellite_GPS_info[1] - S_X * ((Satellite_GPS_info[1] - Satellite_GPS_info[3]) / opt.Satellitehw[0])
    pred_E = Satellite_GPS_info[0] + S_Y * ((Satellite_GPS_info[2] - Satellite_GPS_info[0]) / opt.Satellitehw[1])
    pred_GPS_info = [pred_E, pred_N]
    # calc euclidean Distance between pred and gt
    distance = euclideanDistance(drone_GPS_info, [pred_GPS_info])
    # json_output pred GPS and groundtruth GPS for save
    GPS_output_dict = {}
    GPS_output_dict["GT_GPS"] = drone_GPS_info
    GPS_output_dict["Pred_GPS"] = pred_GPS_info
    GPS_output_dict["UAV_filename"] = UAV_image_path
    GPS_output_dict["Satellite_filename"] = Satellite_image_path
    GPS_output_dict["mapsize"] = mapsize
    GPS_output_dict["drone_in_satellite_relative_position"] = drone_in_satellite_relative_position
    GPS_output_dict["Satellite_GPS_info"] = Satellite_GPS_info
    GPS_output_list.append(GPS_output_dict)
    SDM_single_score = SDM_evaluateSingle(distance, 1)
    return SDM_single_score


GPS_output_list = []
def test(model, dataloader, opt):
    total_score = 0.0
    total_score_b = 0.0
    flag_bias = 0
    start_time = time.time()
    SDM_scores = 0
    for uav, satellite, X, Y, UAV_image_path, Satellite_image_path, UAV_GPS, Satellite_INFO in tqdm(dataloader):
        z = uav.cuda()
        x = satellite.cuda()
        response, loc_bias = model(z, x)
        response = torch.sigmoid(response)
        map = response.squeeze().cpu().detach().numpy()

        # kernel = np.ones((opt.filterR, opt.filterR), np.float32)
        # hanning kernel
        # kernel = create_hanning_mask(opt.filterR)
        # map = cv2.filter2D(map, -1, kernel)

        label_XY = np.array([X.squeeze().detach().numpy(), Y.squeeze().detach().numpy()])

        satellite_map = cv2.resize(map, opt.Satellitehw)
        id = np.argmax(satellite_map)
        S_X = int(id // opt.Satellitehw[0])
        S_Y = int(id % opt.Satellitehw[1])
        pred_XY = np.array([S_X, S_Y])

        # calculate SDM1 critron
        SDM_single_score = SDM_evaluate_score(opt,UAV_GPS,Satellite_INFO,UAV_image_path,Satellite_image_path,S_X,S_Y)
        # SDM score

        SDM_scores+=SDM_single_score
        # RDS score
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
    SDM_score = SDM_scores / len(dataloader)
    print("the final RDS score is {}".format(score))
    print("the final SDM score is {}".format(SDM_score))
    if flag_bias:
        score_b = total_score_b / len(dataloader)
        print("the final score_bias is {}".format(score_b))

    with open(opt.savename, "w") as F:
        F.write("the final score is {}\n".format(score))
        F.write("the SDM score is {}\n".format(SDM_score))
        F.write("time consume is {}".format(time_consume))

    with open(opt.GPS_output_filename,"w") as F:
        json.dump(GPS_output_list, F, indent=4, ensure_ascii=False)


def main():
    opt = get_opt()
    model = create_model(opt)
    dataloader = create_dataset(opt)
    test(model, dataloader, opt)


if __name__ == '__main__':
    main()

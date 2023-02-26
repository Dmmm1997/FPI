import argparse
import collections
import json
import cv2
import os

import math
import numpy as np


def get_opt():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--info_file', default='GPS_pred_gt.json', type=str, help='training dir path')
    parser.add_argument('--out_file', default='result_files/', type=str, help='training dir path')
    opt = parser.parse_args()
    return opt

def latlog2meter(lata, loga, latb, logb):
    # EARTH_RADIUS = 6371.0
    EARTH_RADIUS =6378.137
    PI = math.pi
    # // 转弧度
    lat_a = lata * PI / 180
    lat_b = latb * PI / 180
    a = lat_a - lat_b
    b = loga * PI / 180 - logb * PI / 180
    dis = 2 * math.asin(
        math.sqrt(math.pow(math.sin(a / 2), 2) + math.cos(lat_a) * math.cos(lat_b) * math.pow(math.sin(b / 2), 2)))

    distance = EARTH_RADIUS * dis * 1000
    return distance


def main():
    opt = get_opt()
    if not os.path.exists(opt.out_file):
        os.makedirs(opt.out_file)
    # read the info file
    meters_error = []
    areas = []
    mapsize_level_error_dict = collections.defaultdict(list)
    mapsize2meter = collections.defaultdict(list)
    circle_level_error_dict = collections.defaultdict(list)
    with open(opt.info_file,"r") as F:
        context = json.load(F)
    for GPS_output_dict in context:
        # get base info
        drone_GPS_info = GPS_output_dict["GT_GPS"]
        pred_GPS_info = GPS_output_dict["Pred_GPS"]
        UAV_image_path = GPS_output_dict["UAV_filename"]
        Satellite_image_path = GPS_output_dict["Satellite_filename"]
        mapsize = GPS_output_dict["mapsize"]
        drone_in_satellite_relative_position = GPS_output_dict["drone_in_satellite_relative_position"]
        Satellite_GPS_info = GPS_output_dict["Satellite_GPS_info"]
        # calculate meter level error
        meter_error = latlog2meter(pred_GPS_info[0],pred_GPS_info[1],drone_GPS_info[0],drone_GPS_info[1])
        meters_error.append([Satellite_image_path,meter_error])
        # calculate satellite map's w and h

        satellite_h_meter = latlog2meter(Satellite_GPS_info[0],Satellite_GPS_info[1],Satellite_GPS_info[0],Satellite_GPS_info[3])
        satellite_w_meter = latlog2meter(Satellite_GPS_info[0],Satellite_GPS_info[1],Satellite_GPS_info[2],Satellite_GPS_info[1])
        mapsize2meter[mapsize].append([satellite_h_meter, satellite_w_meter])
        # print(satellite_w_meter,satellite_h_meter)
        # print(satellite_h_meter*satellite_w_meter)
        areas.append(satellite_h_meter*satellite_w_meter)
        mapsize_level_error_dict[mapsize].append(meter_error)
        # calculate circle level error
        circle_level = 5
        relative_distance = math.sqrt((math.pow(drone_in_satellite_relative_position[0],2)+math.pow(drone_in_satellite_relative_position[1],2))/2)
        distance = math.floor(relative_distance*5)
        circle_level_error_dict[distance].append(meter_error)


        # visualization
        # UAV_image = cv2.imread(UAV_image_path[0])
        # Satellite_image = cv2.imread(Satellite_image_path[0])
        # h,w,_ = Satellite_image.shape
        # pred_XY = [int((pred_GPS_info[0]-Satellite_GPS_info[0])/(Satellite_GPS_info[2]-Satellite_GPS_info[0])*w),
        #           int((Satellite_GPS_info[1]-pred_GPS_info[1])/(Satellite_GPS_info[1]-Satellite_GPS_info[3])*h)]
        # label_XY = [
        #     int((drone_GPS_info[0] - Satellite_GPS_info[0]) / (Satellite_GPS_info[2] - Satellite_GPS_info[0]) * w),
        #     int((Satellite_GPS_info[1] - drone_GPS_info[1]) / (Satellite_GPS_info[1] - Satellite_GPS_info[3]) * h)]
        # satelliteImage = cv2.circle(Satellite_image,pred_XY,radius=5,color=(255,0,0),thickness=3)
        # satelliteImage = cv2.circle(Satellite_image,label_XY,radius=5,color=(0,255,0),thickness=3)
        # print(meter_error)mapsize2meter

        # cv2.imshow("satellite",satelliteImage)
        # cv2.imshow("drone",UAV_image)
        # cv2.waitKey(0)


    for map_size_ , meter_hw in mapsize2meter.items():
        mean_h = np.array(meter_hw)[:,0].mean()
        mean_w = np.array(meter_hw)[:,1].mean()
        print(map_size_,mean_w,mean_h)

    #----------------------------save result-----------------------------
    # save the meter-level result (total)
    meter_range = list(range(0,100,1))
    # meter_range = np.array(meter_range) / 50
    res_list = [0]*len(meter_range)
    for pos in meters_error:
        for ind,meter in enumerate(meter_range[::-1]):
            if pos[1]>meter:
                break
            res_list[len(meter_range)-1-ind]+=1
    file_res_list = []
    for ind,res in enumerate(res_list):
        # print("<{}m = {}".format(meter_range[ind],res/len(meters_error)))
        file_res_list.append("{} {}\n".format(meter_range[ind],res/len(meters_error)))
    with open(os.path.join(opt.out_file,"total-level.txt"),"w") as F:
        F.writelines(file_res_list)

    # save the meter-level result (map_size level)
    meter_range = list(range(0, 100, 1))
    # meter_range = np.array(meter_range)/50
    mapsize_result_dict = {}
    for mapsize,meter_bias_map_level in mapsize_level_error_dict.items():
        res_list = [0] * len(meter_range)
        for meter_bias in meter_bias_map_level:
            for ind,meter in enumerate(meter_range):
                if meter_bias<meter:
                    res_list[ind]+=1
        file_res_list = []
        for ind, res in enumerate(res_list):
            # print("<{}m = {}".format(meter_range[ind],res/len(meters_error)))
            file_res_list.append("{} {}\n".format(meter_range[ind], res / len(meter_bias_map_level)))
        with open(os.path.join(opt.out_file, "mapsize={}.txt".format(int(mapsize))), "w") as F:
            F.writelines(file_res_list)
        mapsize_result_dict[mapsize] = res_list
    # arg = np.argsort(list(mapsize_result_dict.keys()))
    # now_sorted_keys = np.array(list(mapsize_result_dict.keys()))[arg]
    #
    # for mapsize in now_sorted_keys:
    #     res_list = mapsize_result_dict[mapsize]
    #     print("{}<{}m:{}/{}={}".format(mapsize,10,res_list[10],2331,res_list[10]/2331))


    # save the meter-level result (circle range level)
    meter_range = list(range(0, 100, 1))
    # meter_range = np.array(meter_range) / 50
    for mapsize,meter_bias_map_level in circle_level_error_dict.items():
        res_list = [0] * len(meter_range)
        for meter_bias in meter_bias_map_level:
            for ind,meter in enumerate(meter_range):
                if meter_bias<meter:
                    res_list[ind]+=1
        file_res_list = []
        for ind, res in enumerate(res_list):
            # print("<{}m = {}".format(meter_range[ind],res/len(meters_error)))
            file_res_list.append("{} {}\n".format(meter_range[ind], res / len(meter_bias_map_level)))
        with open(os.path.join(opt.out_file, "circle={}.txt".format(int(mapsize))), "w") as F:
            F.writelines(file_res_list)




    # print(max(areas),min(areas))

    # save the edge-level


if __name__ == '__main__':
    main()
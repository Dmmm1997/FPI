import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
from get_property import find_GPS_image
import json
from multiprocessing import Pool

# from center crop image
def center_crop_and_resize(img,target_size=None):
    h,w,c = img.shape
    min_edge = min((h,w))
    if min_edge==h:
        edge_lenth = int((w-min_edge)/2)
        new_image = img[:,edge_lenth:w-edge_lenth,:]
    else:
        edge_lenth = int((h - min_edge) / 2)
        new_image = img[edge_lenth:h-edge_lenth, :, :]
    assert new_image.shape[0]==new_image.shape[1],"the shape is not correct"
    # LINEAR Interpolation
    if target_size:
        new_image = cv2.resize(new_image,target_size)

    return new_image

def sixNumber(str_number):
    str_number=str(str_number)
    while(len(str_number)<6):
        str_number='0'+str_number
    return str_number

# get position(E,N) from the txt file
def getMapPosition(txt):
    place_info_dict = {}
    with open(txt) as F:
        context = F.readlines()
        for line in context:
            name = line.split(" ")[0]
            TN = float(line.split((" "))[1].split("TN")[-1])
            TE = float(line.split((" "))[2].split("TE")[-1])
            BN = float(line.split((" "))[3].split("BN")[-1])
            BE = float(line.split((" "))[4].split("BE")[-1])
            place_info_dict[name] = [TN, TE, BN, BE]
    return place_info_dict

# check and makedir
def checkAndMkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def randomCropSatelliteMap(Mapimage, Position,cover_rate, mapsize ,target_size):
    '''
    random crop the part in the satellite map
    :param Mapimage: ori big satellite map
    :param Position: the UAV position in the big satellite image (CenterX,CenterY)
    :param cover_rate: the rate of the random position cover the map size
    :param mapsize: the UAV corresponding size in the satellite map
    :param target_size: the final output size of the satellite image
    :return: cropped image and the UAV image's position in the cropped image
    '''
    h,w,c = Mapimage.shape
    centerX,centerY = Position
    center_cover_pixels = cover_rate*mapsize
    centerX_changed_Min = centerX-center_cover_pixels//2
    centerX_changed_Max = centerX+center_cover_pixels//2
    centerY_changed_Min = centerY-center_cover_pixels//2
    centerY_changed_Max = centerY+center_cover_pixels//2
    random_value_X = np.random.randint(centerX_changed_Min,centerX_changed_Max)
    random_value_Y = np.random.randint(centerY_changed_Min,centerY_changed_Max)
    # put the X and Y between in the Mapimage
    random_value_X = max(0,min(w-1,random_value_X))
    random_value_Y = max(0,min(h-1,random_value_Y))
    # get bbox [x1,y1,x2,y2]
    bbox = [random_value_X-mapsize//2,random_value_Y-mapsize//2,random_value_X+mapsize//2,random_value_Y+mapsize//2]
    bias_left = bias_top = bias_right = bias_bottom = 0
    if bbox[0]<0:
        bias_left = -1*bbox[0]
        bbox[0] = 0
    if bbox[1]<0:
        bias_top = -1*bbox[1]
        bbox[1] = 0
    if bbox[2]>=w:
        bias_right = bbox[2]-w
        bbox[2] = w
    if bbox[3]>=h:
        bias_bottom = bbox[3]-h
        bbox[3] = h
    croped_image = Mapimage[bbox[1]:bbox[3],bbox[0]:bbox[2],:]

    # padding the border
    croped_image = cv2.copyMakeBorder(croped_image, bias_top, bias_bottom, bias_left, bias_right, cv2.BORDER_CONSTANT)

    target_image = cv2.resize(croped_image,target_size)

    # get the bias of the center
    bias_center_x = centerX-random_value_X
    bias_center_y = centerY-random_value_Y
    rate_of_position_x = 0.5+bias_center_x/mapsize
    rate_of_position_y = 0.5+bias_center_y/mapsize

    return target_image, [rate_of_position_x,rate_of_position_y]

def CenterCropFromSatellite(Mapimage,Position,mapsize,target_size):
    centerX,centerY = Position
    new_X_min = centerX-mapsize//2
    new_X_max = centerX+mapsize//2
    new_Y_min = centerY-mapsize//2
    new_Y_max = centerY+mapsize//2
    h, w, c = Mapimage.shape
    bbox = [new_X_min, new_Y_min, new_X_max, new_Y_max]
    bias_left = bias_top = bias_right = bias_bottom = 0
    if bbox[0] < 0:
        bias_left = int(-1 * bbox[0])
        bbox[0] = 0
    if bbox[1] < 0:
        bias_top = int(-1 * bbox[1])
        bbox[1] = 0
    if bbox[2] >= w:
        bias_right = int(bbox[2] - w)
        bbox[2] = w
    if bbox[3] >= h:
        bias_bottom = int(bbox[3] - h)
        bbox[3] = h
    croped_image = Mapimage[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2]), :]
    mean = np.mean(croped_image, axis=(0, 1), dtype=float)
    # padding the border
    croped_image = cv2.copyMakeBorder(croped_image, bias_top, bias_bottom, bias_left, bias_right, cv2.BORDER_CONSTANT,
                                      value=mean)

    result_image = cv2.resize(croped_image,target_size)
    return result_image



def mutiProcess(class_file):
    index = 0
    dir, map = dirlist[class_file], MapPath[class_file]
    images_list = glob.glob(os.path.join(dir, "*/*.JPG"))
    # place name
    place = dir.split("/")[-1]
    # satellite map image
    map_image = cv2.imread(map)
    map_h_, map_w_, _ = map_image.shape
    map_image = cv2.resize(map_image,(map_w_//2,map_h_//2), interpolation=3)
    map_h, map_w, _ = map_image.shape
    cur_TN, cur_TE, cur_BN, cur_BE = map_position_dict[place]
    for JPG in tqdm(images_list):
        # image
        image = cv2.imread(JPG)
        uav_h, uav_w, _ = image.shape
        # position info
        GPS_info = find_GPS_image(JPG)
        y = list(list(GPS_info.values())[0].values())
        E, N = y[3], y[1]
        # compute the corresponding position of the big satellite image
        centerX = (E - cur_TE) / (cur_BE - cur_TE) * map_w
        centerY = (N - cur_TN) / (cur_BN - cur_TN) * map_h
        # center crop and resize the UAV image
        croped_image = center_crop_and_resize(image, target_size=(512, 512))
        # create target related dir
        fileClassIndex = sixNumber(index)
        fileClassIndex = "{}_{}".format(place,fileClassIndex)

        fileClassDir = os.path.join(SiamUAV_train_dir, fileClassIndex)
        checkAndMkdir(fileClassDir)

        fileClassDirUAV = os.path.join(SiamUAV_train_dir, fileClassIndex, "UAV")
        checkAndMkdir(fileClassDirUAV)

        fileClassDirSatellite = os.path.join(SiamUAV_train_dir, fileClassIndex, "Satellite")
        checkAndMkdir(fileClassDirSatellite)


        # imwrite UAV image
        UAV_target_path = os.path.join(fileClassDirUAV, "0.JPG")
        cv2.imwrite(UAV_target_path, croped_image)
        # crop the corresponding part of the satellite map
        croped_satellite_image = CenterCropFromSatellite(map_image,Position=(centerX,centerY),mapsize=2000, target_size=(1280, 1280))
        Satellite_target_path = os.path.join(fileClassDirSatellite, "0.tif")
        cv2.imwrite(Satellite_target_path, croped_satellite_image)

        index += 1

if __name__ == '__main__':
    # source path
    root = "/media/dmmm/4T-3/DataSets/DenseCV_Data/高度数据集/"
    train_source_path = os.path.join(root, "oridata", "test")
    # target path
    SiamUAV_root = "/media/dmmm/4T-3/DataSets/FPI/GPS_info"
    # SiamUAV_train_dir = os.path.join(SiamUAV_root,"merge_test_500-1500")
    SiamUAV_train_dir = os.path.join(SiamUAV_root,"merge_test_cjh")
    checkAndMkdir(SiamUAV_train_dir)

    dirlist = [i for i in glob.glob(os.path.join(train_source_path, "*")) if os.path.isdir(i)]
    MapPath = [i + ".tif" for i in dirlist]
    for i in MapPath:
        if not os.path.exists(i):
            raise NameError("name is not corresponding!")
    position_info_path = os.path.join(train_source_path, "PosInfo.txt")
    map_position_dict = getMapPosition(position_info_path)

    # muti-process
    P = Pool(processes=4)
    P.map(func=mutiProcess, iterable=range(len(MapPath)))
    # single process
    # for i in range(len(MapPath)):
    #     mutiProcess(i)
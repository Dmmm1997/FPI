import os
import glob
import numpy as np

filename = "/home/dmmm/PycharmProject/SiamUAV/checkpoints/400_112_lr0.0003_R1_B_AdamW_nw=15_share_/result_files/total-level.txt"

x_max = 100

def read_from_txt(filename):
    with open(filename, "r") as f:
        context = f.readlines()
    meter, prob = [], []
    # index = [0,1,2,3,5,7,10,20,30,40,50,60,70,80,90,99]
    for line in np.array(context):
        m, p = line.split("\n")[0].split(" ")
        m = float(m)
        p = float(p)
        meter.append(m)
        prob.append(p)
    return meter, prob

def get_area(x,y):
    stride = x_max/len(x)
    area = 0
    for x_,y_ in zip(x,y):
        area+=y_*stride
    return area

x,y = read_from_txt(filename)
area = get_area(x,y)
print(area)




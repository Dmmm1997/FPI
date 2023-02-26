import os
import glob
import numpy as np
import shutil
from tqdm import tqdm



test_dir = "/media/dmmm/4T-3/DataSets/FPI/FPI2023/test"
test_pair_list = glob.glob(test_dir+"/*")

target_dir = "/media/dmmm/4T-3/DataSets/FPI/FPI2023/val"
rate = 0.25

np.random.shuffle(test_pair_list)
test_pair_list = test_pair_list[:int(0.25*len(test_pair_list))]


for pair in tqdm(test_pair_list):
    name = os.path.basename(pair)
    shutil.copytree(pair,os.path.join(target_dir,name))





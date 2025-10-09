import os
import shutil
from tqdm import tqdm

input_dir = "/data_new/luxiaoxi/dataset/medical_depth/final_version_processed"
save_dir = "/data_new/luxiaoxi/dataset/medical_depth_output/fundus_dong/gt"
os.makedirs(save_dir, exist_ok=True)
img_list = []
with open(os.path.join(input_dir, "split", "test.txt")) as file:
    for line in file.readlines():
        line = line.strip('\n').split(',')[0]
        img_list.append(line)
img_list = [os.path.join(input_dir, f) for f in img_list]

for img_path in tqdm(img_list):
    shutil.copy(img_path, save_dir)
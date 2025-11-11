import os
import shutil
from tqdm import tqdm
import random

## synthetic image sample
# data_dir = "/data_new/luxiaoxi/dataset/medical_depth/final_version_processed"
# folders = [f for f in os.listdir(data_dir) if f != "split"]
# save_dir = "/data_new/luxiaoxi/dataset/medical_depth/final_version_processed_sample"
# os.makedirs(save_dir, exist_ok=True)
#
# for folder in folders:
#     print("Processing: " + folder)
#     img_path = os.path.join(data_dir, folder, "left", "imgs")
#     save_path = os.path.join(save_dir, folder)
#     os.makedirs(save_path, exist_ok=True)
#     for i in tqdm(range(1, 129)):
#         img_name = "%03d25.png"%i
#         shutil.copy(os.path.join(img_path, img_name), save_path)


## real image sample
data_dir = "/data/luxiaoxi/dataset"
folders = ["eyetube_phase%d"% i for i in range(1, 5)]
save_dir = "/data/luxiaoxi/dataset/eyetube_sample"
os.makedirs(save_dir, exist_ok=True)

for folder in folders:
    print("Processing folder: %s" % folder)
    subfolders = os.listdir(os.path.join(data_dir, folder, "fundus"))
    for subfolder in tqdm(subfolders):
        img_path = os.path.join(data_dir, folder, "fundus", subfolder, "left")
        img_list = os.listdir(img_path)
        selected_image = random.choice(img_list)
        new_name = "%s_%s_%s"%(folder, subfolder, selected_image)
        shutil.copy2(os.path.join(img_path, selected_image), os.path.join(save_dir, new_name))

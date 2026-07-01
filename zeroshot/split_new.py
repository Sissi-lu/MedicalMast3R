import os
import shutil
from tqdm import tqdm
import random

# synthetic image sample
data_dir = "/data_new/luxiaoxi/dataset/medical_depth/final_version_processed"
folders = ["s1_processed", "s2_processed", "s3_processed", "s4_processed", "s5_processed", "s6_processed", "s7_processed"]
save_dir = "/data_new/luxiaoxi/dataset/medical_depth/final_version_processed/split_new"
os.makedirs(save_dir, exist_ok=True)

all_extracted_files = []
for folder in folders:
    print("Processing: " + folder)
    img_path = os.path.join(data_dir, folder, "left", "imgs")
    save_path = os.path.join(save_dir, folder)
    os.makedirs(save_path, exist_ok=True)
    for i in tqdm(range(1, 129)):
        r_ind = random.randint(1, 48)
        img_name = "%03d%02d.png"%(i, r_ind)
        img_path_string = os.path.join(folder, "left", "imgs", img_name)
        assert os.path.exists(os.path.join(data_dir, img_path_string)), "Image does not %s"%(img_path_string)
        all_extracted_files.append(img_path_string)

shuffled_list = all_extracted_files.copy()
random.shuffle(shuffled_list)
groups = {
    "test.txt": shuffled_list[:100],
    "val.txt": shuffled_list[100:200],
    "train.txt": shuffled_list[200:]
}

sub_dirs = [
    "left/imgs", "left/depth", "left/metric_depth", "left/valid_region_mask", "left/instrument_mask", "left/normal",
    "right/imgs", "right/depth", "right/metric_depth", "right/valid_region_mask", "right/instrument_mask", "right/normal"
]


def expand_path(original_path):
    """
    输入: s1_processed/left/imgs/00523.png
    输出: 12条相关路径的列表
    """
    # 提取基础部分 (s1_processed) 和 文件名 (00523.png)
    # 假设路径结构总是 prefix/left/imgs/filename
    parts = original_path.split('/')
    prefix = parts[0]  # s1_processed
    filename = parts[-1]  # 00523.png

    expanded = []
    for sub in sub_dirs:
        expanded.append(f"{prefix}/{sub}/{filename}")
    return expanded

for filename, data_list in groups.items():
    with open(os.path.join(save_dir, filename), 'w') as f:
        for item in data_list:
            # 获取扩展后的 12 个路径
            expanded_paths = expand_path(item)
            # 将这 12 个路径用逗号连接成一行（或者根据你需求换行）
            line = ",".join(expanded_paths)
            f.write(line + "\n")

print(len(all_extracted_files))
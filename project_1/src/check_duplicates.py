import os
from PIL import Image, ImageStat
from tqdm import tqdm
import json

image_folder = os.path.join('../data/celeba/', 'img_align_celeba') #img_align_celeba check_dup
image_files = [_ for _ in os.listdir(image_folder) if _.endswith('jpg')]
duplicate_files = {}
duplicate_list = []
hash_dict = {}

def get_image_hash(file_path):
    global hash_dict
    if not file_path in hash_dict.keys():
        image_check = Image.open(os.path.join(image_folder, file_path))
        pix_mean = ImageStat.Stat(image_check).mean
        hash_dict[file_path] = pix_mean
    else:
        pix_mean = hash_dict[file_path]
        
    return pix_mean

for i, file_org in enumerate(tqdm(image_files)):   
    if not file_org in set(duplicate_list):
        pix_mean1 = get_image_hash(file_org)

        for file_check in image_files[i+1:]:
            if not file_check in set(duplicate_list):
                pix_mean2 = get_image_hash(file_check)
            
                if pix_mean1 == pix_mean2:
                    if file_org in duplicate_files.keys():
                        duplicate_files[file_org].extend([file_check])
                    else:
                        duplicate_files[file_org] = [file_check]
                    duplicate_list.append(file_check)
                    duplicate_list.append(file_org)
                    hash_dict.pop(file_check)
                    
        hash_dict.pop(file_org)

json.dump(duplicate_files, open("duplicated.json",'w'))
import pandas as pd
import numpy as np
from PIL import Image
from skimage import io, transform

def clean_data(csv_name):
    
    df = pd.read_csv(csv_name)

    required = ['unique_id', 'text', 'image_status', 'label']

    df = df[required]

    df = df.dropna().reset_index(drop=True)
    
    not_downloadable = []
    for idx in range(len(df)):
        list_imgs = df.iloc[idx]["image_status"].split(";")
        nb_imgs = list_imgs.count("not downloadable")

        if nb_imgs >= len(list_imgs):
            not_downloadable.append(idx)

            
    df = df.drop(df.index[not_downloadable]).reset_index(drop=True)
    
    not_opening = []
    not_opening_col = []
    for idx in range(len(df)):
        list_imgs = df.iloc[idx]["image_status"].split(";")
        nb_imgs = 0

        current_status = []
        for img_name in list_imgs:
            if img_name == "not downloadable":
                nb_imgs +=1
                current_status.extend(["not downloaded"])
                continue
            try:
                image = Image.open(img_name).convert("RGB")
                current_status.extend([img_name])
            except Exception as e:
#                 print(str(e))
                current_status.extend(["not opening"])
                nb_imgs += 1

        not_opening_col.extend([";".join(current_status)])
        if nb_imgs >= len(list_imgs):
            not_opening.append(idx)
            
    df['current_status'] = not_opening_col
    df = df.drop(df.index[not_opening]).reset_index(drop=True)
    
    total_count_imgs = []
    for idx in range(len(df)):
        list_imgs = df.iloc[idx]["current_status"].split(";")
        no_use = list_imgs.count("not downloaded") + list_imgs.count("not opening")

        total_count_imgs.extend([len(list_imgs) - no_use])

    df['image_count'] = total_count_imgs
    
    df = df.loc[df['image_count'] >=2]
    
    df = df.reset_index(drop=True)
    
    return df
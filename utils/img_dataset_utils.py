import os
from datetime import datetime
from pathlib import Path
import shutil
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import PIL.Image as Image


def display_rand(dir, LABEL_DESC):
    # Create sample of images from each class
    sample_classes = []
    sample_img_list = []
    print("Getting image for label", end="")
    for label in LABEL_DESC['label']:
        label_desc = LABEL_DESC.loc[LABEL_DESC.label == label]['label_desc'].values[0].strip()
        print(f"{label}:{label_desc}, ", end="")
        try:
            img_sample = ALL_LABELS.loc[ALL_LABELS['label'] == label].sample(1)['img'].values[0]
            sample_img_list.append(str(dir + "/" + img_sample))
            sample_classes.append(label_desc)
        except Exception as e:
            print(f"Can't fetch sample! Error msg: {e}")
    sample_imgs = np.array([np.array(Image.open(img), 'f') for img in sample_img_list])

    fig = plt.figure(figsize=(25, 15))
    axes = []
    row = 5
    col = 5
    idx = 0
    for (img, title) in zip(sample_imgs, sample_classes):
        axes.append(fig.add_subplot(row, col, idx + 1))
        idx += 1
        subplot_title = (title)
        axes[-1].set_title(subplot_title, fontsize=20)
        plt.imshow(img / 255.0)
    fig.tight_layout()
    plt.show()


def process_imgs(src_path, dest_path):
    """Process images into dest_path"""
    Path(dest_path).mkdir(parents=True, exist_ok=True)
    src_path = Path(src_path)
    dest_path = Path(dest_path)
    all_imgs = os.listdir(src_path)

    for file in all_imgs:
        img = cv2.imread(f"{src_path}/{file}")
        cv2.imwrite(f"{dest_path}/{file}", img)


def gen_imgsets(class_list, img_list, src_path, dest_path, test_path, max_train_samples=100, test_prop=.10):
    """Create test and train directories with image datasets. Return dataframe enumerating images

    img_list: dataframe containing list of all images in dataset
    src_path: path containing images. Should be preprocessed.
    dest_path: path where images will be copied in directory labels
    test_path: path where test images will go
    test_prop: percentage of images per class to hold for test

    """
    src_path = Path(src_path)
    dest_path = Path(dest_path)
    test_path = Path(test_path)
    df_summary = sumarize(img_list, proportion=False)
    df_test_all = pd.DataFrame(columns=['img', 'label', 'label_desc', 'label_set'])
    label_dirs = df_summary.label_desc

    if dest_path.exists() and dest_path.is_dir():
        shutil.rmtree(dest_path)
    dest_path.mkdir(parents=True, exist_ok=True)

    if test_path.exists() and test_path.is_dir():
        shutil.rmtree(test_path)
    test_path.mkdir(parents=True, exist_ok=True)

    # Since we're missing some images in classes, we will create some empty dirs. HACK
    for label in class_list.label_desc:
        (dest_path / label).mkdir()

    # For each class, sample up to max images and copy into appropriate labeled directory
    for label in label_dirs:
        df_class = img_list.loc[img_list.label_desc == label]
        df_test = df_class.sample(frac=test_prop)
        df_train = df_class.drop(df_test.index)

        if df_train.shape[0] > max_train_samples:
            df_train = df_train.sample(max_train_samples)
        # Copy images to appropriate train subdirs
        for i, row in df_train.iterrows():
            shutil.copy(src_path / row.img, dest_path / row.label_desc)
        for i, row in df_test.iterrows():
            shutil.copy(src_path / row.img, test_path)

        df_test_all = pd.concat([df_test_all, df_test])

    return (df_test_all)


def backup_models(model_tmp_dir='models', backup_dir='/dbfs/FileStore/kaipak/models/SpaceML_msl'):
    """Save copy of models to persistent storage"""
    time_suffix = datetime.now().strftime("%Y-%h-%d-%H:%M:%S")
    src_path = Path(model_tmp_dir)
    dest_path = Path(backup_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    model_files = os.listdir(model_tmp_dir)

    for mf in model_files:
        shutil.copyfile(src_path / mf, dest_path / f'{mf}_{time_suffix}')
    print('Completed backup of model files')




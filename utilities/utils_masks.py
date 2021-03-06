import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
import random as rn
import matplotlib.pyplot as plt
from matplotlib import colors
import os
import pandas as pd
import datetime
import random
from keras import backend as K
#from utilities.preprocess import equalize

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.uniform(hue_shift_limit[0], hue_shift_limit[1])
        h = cv2.add(h, hue_shift)
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0625, 0.0625),
                           scale_limit=(-0.1, 0.1),
                           rotate_limit=(-45, 45), aspect_limit=(0, 0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(0, 0, 0,))
        if mask is not None:
            mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                       borderValue=(0, 0, 0,))
        else:
            mask = mat #a workaround to return transformation matrix without changing func signature

    return image, mask


def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        if mask is not None:
            mask = cv2.flip(mask, 1)
        else:
            mask = True #a workaround to return flip status without changing func signature

    return image, mask


def reverseFlipShiftScaleRotate(image, flip, trans_mat):
    height, width = image.shape
    if flip:
        image = cv2.flip(image, 1)
    if trans_mat is not None:
        image = cv2.warpPerspective(image, trans_mat, (width, height), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0,))
    return image


def train_generator(path, mask_path, ids_train_split, input_size, batch_size, bboxes=None,
                    augmentations=['HUE_SATURATION', 'SHIFT_SCALE', 'FLIP'], outputs=None):
    while True:
        for start in range(0, len(ids_train_split), batch_size):
            x_batch = []
            y_batch = [] if outputs is None else {name:[] for name, scale in outputs.items()}
            end = min(start + batch_size, len(ids_train_split))
            ids_train_batch = ids_train_split[start:end]           
            for id in ids_train_batch.values:
                img = cv2.imread(path.format(id)) #equalize(path.format(id))
                mask = np.array(Image.open(mask_path.format(id)).convert('L'))
                if bboxes is not None:
                    x1, y1, x2, y2 = bboxes[id]                   
                    if (x2 > x1 and y2 > y1):
                        # bounding box width/height is not 0
                        img = img[y1:y2+1, x1:x2+1]
                        mask = mask[y1:y2+1, x1:x2+1]
                img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
                img = img if 'HUE_SATURATION' not in augmentations else randomHueSaturationValue(
                    img, hue_shift_limit=(-50, 50), sat_shift_limit=(-5, 5), val_shift_limit=(-15, 15))
                img, mask = (img, mask) if 'SHIFT_SCALE' not in augmentations else randomShiftScaleRotate(
                    img, mask, shift_limit=(-0.0625, 0.0625), scale_limit=(-0.1, 0.1), rotate_limit=(-0, 0))
                img, mask = (img, mask) if 'FLIP' not in augmentations else randomHorizontalFlip(img, mask)
                x_batch.append(img)
                if outputs is None:
                    mask = np.expand_dims(mask, axis=2)
                    y_batch.append(mask)
                else:
                    for name, scale in outputs.items():
                        size = int(input_size*scale)
                        mask_resize = mask if scale == 1 else cv2.resize(mask, (size, size), interpolation=cv2.INTER_LINEAR)
                        mask_resize = np.expand_dims(mask_resize, axis=2)
                        y_batch[name].append(mask_resize)
            x_batch = np.array(x_batch, np.float32) / 255
            if outputs is None:
                y_batch = np.array(y_batch, np.float32) / 255
            else:
                y_batch = {name:np.array(masks, np.float32) / 255 for name, masks in y_batch.items()}
            yield x_batch, y_batch
            

def valid_generator(path, mask_path, ids_valid_split, input_size, batch_size, bboxes=None, outputs=None):
    while True:
        for start in range(0, len(ids_valid_split), batch_size):
            x_batch = []
            y_batch = [] if outputs is None else {name:[] for name, scale in outputs.items()}
            end = min(start + batch_size, len(ids_valid_split))
            ids_valid_batch = ids_valid_split[start:end]
            for id in ids_valid_batch.values:
                img = cv2.imread(path.format(id)) #equalize(path.format(id))
                mask = np.array(Image.open(mask_path.format(id)).convert('L'))
                if bboxes is not None:
                    x1, y1, x2, y2 = bboxes[id]
                    if (x2 > x1 and y2 > y1):
                        # bounding box width/height is not 0
                        img = img[y1:y2+1, x1:x2+1]
                        mask = mask[y1:y2+1, x1:x2+1]
                img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_LINEAR) 
                mask = cv2.resize(mask, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
                x_batch.append(img)
                if outputs is None:
                    mask = np.expand_dims(mask, axis=2)
                    y_batch.append(mask)
                else:
                    for name, scale in outputs.items():
                        size = int(input_size*scale)
                        mask_resize = mask if scale == 1 else cv2.resize(mask, (size,size), interpolation=cv2.INTER_LINEAR)
                        mask_resize = np.expand_dims(mask_resize, axis=2)
                        y_batch[name].append(mask_resize)
            x_batch = np.array(x_batch, np.float32) / 255
            if outputs is None:
                y_batch = np.array(y_batch, np.float32) / 255
            else:
                y_batch = {name:np.array(masks, np.float32) / 255 for name, masks in y_batch.items()}
            yield x_batch, y_batch

def pseudo_generator(train_path, train_mask_path, test_path, test_mask_path, ids, 
                           input_size, batch_size, bboxes=None,
                           augmentations=['HUE_SATURATION', 'SHIFT_SCALE', 'FLIP'], outputs=None):
    while True:
        for start in range(0, len(ids), batch_size):
            x_batch = []
            y_batch = [] if outputs is None else {name:[] for name, scale in outputs.items()}
            end = min(start + batch_size, len(ids))
            ids_batch = ids[start:end]           
            for id in ids_batch.values:
                if os.path.exists(train_path.format(id)):
                    path = train_path.format(id)
                    mask_path = train_mask_path.format(id)
                elif os.path.exists(test_path.format(id)):
                    path = test_path.format(id)
                    mask_path = test_mask_path.format(id)
                else:
                    print('{} dose not exist.'.format(id) )

                img = cv2.imread(path)
                mask = np.array(Image.open(mask_path).convert('L'))
                if bboxes is not None:
                    x1, y1, x2, y2 = bboxes[id]                   
                    if (x2 > x1 and y2 > y1):
                        # bounding box width/height is not 0
                        img = img[y1:y2+1, x1:x2+1]
                        mask = mask[y1:y2+1, x1:x2+1]
                img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
                img = img if 'HUE_SATURATION' not in augmentations else randomHueSaturationValue(
                    img, hue_shift_limit=(-50, 50), sat_shift_limit=(-5, 5), val_shift_limit=(-15, 15))
                img, mask = (img, mask) if 'SHIFT_SCALE' not in augmentations else randomShiftScaleRotate(
                    img, mask, shift_limit=(-0.0625, 0.0625), scale_limit=(-0.1, 0.1), rotate_limit=(-0, 0))
                img, mask = (img, mask) if 'FLIP' not in augmentations else randomHorizontalFlip(img, mask)
                x_batch.append(img)
                if outputs is None:
                    mask = np.expand_dims(mask, axis=2)
                    y_batch.append(mask)
                else:
                    for name, scale in outputs.items():
                        size = int(input_size*scale)
                        mask_resize = mask if scale == 1 else cv2.resize(mask, (size, size), interpolation=cv2.INTER_LINEAR)
                        mask_resize = np.expand_dims(mask_resize, axis=2)
                        y_batch[name].append(mask_resize)
            x_batch = np.array(x_batch, np.float32) / 255
            if outputs is None:
                y_batch = np.array(y_batch, np.float32) / 255
            else:
                y_batch = {name:np.array(masks, np.float32) / 255 for name, masks in y_batch.items()}

            yield x_batch, y_batch

def show_mask(img_path, mask, pred_mask, show_img=True, bbox=None, threshold = 0.5):
    mask_cmap = colors.ListedColormap(['black', '#bd0b42'])
    mask_cmap = mask_cmap(np.arange(2))
    mask_cmap[:,-1] = np.linspace(0, 1, 2)
    mask_cmap = colors.ListedColormap(mask_cmap)
    pred_cmap = colors.ListedColormap(['black', '#42f49e'])
    pred_cmap = pred_cmap(np.arange(2))
    pred_cmap[:,-1] = np.linspace(0, 1, 2)
    pred_cmap = colors.ListedColormap(pred_cmap)
    plt.figure(figsize=(10,10))
    plt.rcParams['axes.facecolor']='black'
    plt.xticks([])
    plt.yticks([])
    plt.title(img_path)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if show_img:
        plt.imshow(img)
    orig_size = (img.shape[1], img.shape[0])
    (x1,y1,x2,y2) = (0,0,0,0) if bbox is None else tuple(bbox)
    size = orig_size if bbox is None else (x2-x1+1, y2-y1+1)
    if mask is not None:
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_LINEAR)
        mask = mask > threshold
        if bbox is not None:
            mask_full = np.zeros(orig_size[::-1])
            mask_full[y1:y2+1, x1:x2+1] = mask
            mask = mask_full
        plt.imshow(mask, cmap=mask_cmap, alpha=(.6 if show_img else 1))
    if pred_mask is not None:
        pred_mask = cv2.resize(pred_mask, size, interpolation=cv2.INTER_LINEAR)
        pred_mask = pred_mask > threshold
        if bbox is not None:
            mask_full = np.zeros(orig_size[::-1])
            mask_full[y1:y2+1, x1:x2+1] = pred_mask
            pred_mask = mask_full
        plt.imshow(pred_mask, cmap=pred_cmap, alpha=.6)


def show_test_masks(test_path, test_masks_path, number_to_show = None):
    mask_cmap = colors.ListedColormap(['black', '#42f49e'])
    mask_cmap = mask_cmap(np.arange(2))
    mask_cmap[:,-1] = np.linspace(0, 1, 2)
    mask_cmap = colors.ListedColormap(mask_cmap)
    img_paths = os.listdir(test_path)
    if number_to_show is not None:
        img_paths = random.sample(img_paths, number_to_show)
    for img_path in img_paths:
        plt.figure(figsize=(15,15))
        plt.xticks([])
        plt.yticks([])
        plt.title(test_path + img_path)
        img = cv2.imread(test_path + img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        id =  img_path.split('.')[0]
        mask = Image.open('{0}{1}_mask.gif'.format(test_masks_path, id)).convert('L')
        plt.imshow(mask, cmap=mask_cmap, alpha=.6)


def get_run_name(weights_file, model_name):
    dt = datetime.datetime.now()
    while True:
        run_name = '{0}-{1:%Y-%m-%d-%H%M}'.format(model_name, dt)
        if not os.path.isfile(weights_file.format(run_name)):
            return run_name
        dt = dt + datetime.timedelta(minutes=-1)

def get_bboxes(path):
    bboxes = dict()
    with open(path) as fbbox:
        for line in fbbox:
            cols = line.split(',')
            bboxes[cols[0]] = (int(cols[1]), int(cols[2]), \
                               int(cols[3]), int(cols[4]))
            
    return bboxes

def set_results_reproducible():
    '''https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development'''
    
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(42)
    rn.seed(12345)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    tf.set_random_seed(1234)

    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
    
    return

def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()

def load_array(fname):
    return bcolz.open(fname)[:]

def make_list_ids(ids_train, ids_test, batch_size, accum_iters):
    if batch_size < 4:
            batch_size = accum_iters*batch_size
    test_pro = int(batch_size/4)
    train_pro = int((3*batch_size)/4)

    ids_test = pd.concat([ids_test, ids_test.sample(test_pro-(len(ids_test)%test_pro))])
    test_ids = np.split(np.array(ids_test.values), indices_or_sections=(len(ids_test)/test_pro))
    test_ids = np.array(test_ids) 

    ids_train = pd.concat([ids_train, ids_train.sample(train_pro-(len(ids_train)%train_pro))])
    train_ids = np.array(np.split(np.array(ids_train.values), indices_or_sections=(len(ids_train)/train_pro)))

    dif = test_ids.shape[0] - train_ids.shape[0]

    ids_train = pd.concat([ids_train, ids_train.sample(dif*train_pro, replace=True)])
    train_ids = np.array(np.split(np.array(ids_train.values), indices_or_sections=(len(ids_train)/train_pro)))

    data = []
    for i in range(train_ids.shape[0]):
        for it in test_ids[i]:
            data.append(it)
        for it in train_ids[i]:
            data.append(it)
                
    return pd.Series(data) 
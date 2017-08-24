import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime

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
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask


def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask


def train_generator(path, mask_path, ids_train_split, input_size, batch_size, bboxes=None):
    while True:
        for start in range(0, len(ids_train_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_train_split))
            ids_train_batch = ids_train_split[start:end]           
            for id in ids_train_batch.values:
                img = cv2.imread(path.format(id))
                mask = np.array(Image.open(mask_path.format(id)).convert('L'))
                #mask = cv2.imread(mask_path.format(id), cv2.IMREAD_GRAYSCALE)
                if bboxes is not None:
                    x1, y1, x2, y2 = bboxes[id]                   
                    if (x2 > x1 and y2 > y1):
                        # bounding box width/height is not 0
                        img = img[y1:y2, x1:x2]
                        mask = mask[y1:y2, x1:x2]
                img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
                img = randomHueSaturationValue(img,
                                               hue_shift_limit=(-50, 50),
                                               sat_shift_limit=(-5, 5),
                                               val_shift_limit=(-15, 15))
                img, mask = randomShiftScaleRotate(img, mask,
                                                   shift_limit=(-0.0625, 0.0625),
                                                   scale_limit=(-0.1, 0.1),
                                                   rotate_limit=(-0, 0))
                img, mask = randomHorizontalFlip(img, mask)
                mask = np.expand_dims(mask, axis=2)
                x_batch.append(img)
                y_batch.append(mask)
            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32) / 255
            yield x_batch, y_batch
            

def valid_generator(path, mask_path, ids_valid_split, input_size, batch_size, bboxes=None):
    while True:
        for start in range(0, len(ids_valid_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_valid_split))
            ids_valid_batch = ids_valid_split[start:end]
            for id in ids_valid_batch.values:
                img = cv2.imread(path.format(id))
                mask = np.array(Image.open(mask_path.format(id)).convert('L'))
                #mask = cv2.imread(mask_path.format(id), cv2.IMREAD_GRAYSCALE)
                if bboxes is not None:
                    x1, y1, x2, y2 = bboxes[id]
                    if (x2 > x1 and y2 > y1):
                        # bounding box width/height is not 0
                        img = img[y1:y2, x1:x2]
                        mask = mask[y1:y2, x1:x2]
                img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_LINEAR) 
                mask = cv2.resize(mask, (input_size, input_size), interpolation=cv2.INTER_LINEAR)                               
                mask = np.expand_dims(mask, axis=2)
                x_batch.append(img)
                y_batch.append(mask)
            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32) / 255
            yield x_batch, y_batch


def show_mask(img_path, mask, pred_mask, show_img=True, bbox=None, mask_cmap='Greens', pred_cmap='Reds'):
    fig = plt.figure(figsize=(10,10))
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
        if bbox is not None:
            mask_full = np.zeros(orig_size[::-1])
            mask_full[y1:y2+1, x1:x2+1] = mask
            mask = mask_full
        plt.imshow(mask, cmap=mask_cmap, alpha=(.5 if show_img else 1))
    if pred_mask is not None:
        pred_mask = cv2.resize(pred_mask, size, interpolation=cv2.INTER_LINEAR)
        if bbox is not None:
            mask_full = np.zeros(orig_size[::-1])
            mask_full[y1:y2+1, x1:x2+1] = pred_mask
            pred_mask = mask_full
        plt.imshow(pred_mask, cmap=pred_cmap, alpha=.3)


def show_test_masks(test_path, test_masks_path, mask_cmap='Greens'):
    for img_path in os.listdir(test_path):
        fig = plt.figure(figsize=(10,10))
        plt.xticks([])
        plt.yticks([])
        plt.title(test_path + img_path)
        img = cv2.imread(test_path + img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        if test_masks_path is not None:
            id =  img_path.split('.')[0]
            mask = Image.open('{0}{1}.gif'.format(test_masks_path, id)).convert('L')
            plt.imshow(mask, cmap=mask_cmap, alpha=.5)


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
    
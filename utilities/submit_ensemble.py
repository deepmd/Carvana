import cv2
from PIL import Image
import numpy as np
import pandas as pd
import threading
import sys
if sys.version_info < (3, 0):
    import Queue as queue
else:
    import queue
import tensorflow as tf
from tqdm import tqdm
import os
from utilities import utils_masks as utils

mask_file = '{0}{1}_mask.gif'

# https://www.kaggle.com/stainsby/fast-tested-rle
def run_length_encode(mask):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    inds = mask.flatten()
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle

def data_loader(q, test_path, batch_size, ids_test, models_info, bboxes, augmentations, test_masks_path, reapply_bboxed_path):
    for start in range(0, len(ids_test), batch_size):
        x_batches = [[[] for _ in range(augmentations+1)] for _ in range(len(models_info))]
        x_augments = [[[] for _ in range(augmentations+1)] for _ in range(len(models_info))]
        x_ids = []
        end = min(start + batch_size, len(ids_test))
        ids_test_batch = ids_test[start:end]
        if test_masks_path is not None:
            exist_masks_ids = [id for id in ids_test_batch if os.path.isfile(mask_file.format(test_masks_path, id))]
            if len(exist_masks_ids) == len(ids_test_batch):
                x_ids = exist_masks_ids
                masks = []
                for id in ids_test_batch:
                    if reapply_bboxed_path is None:
                        masks.append(np.array(Image.open(mask_file.format(test_masks_path, id)).convert('L'), np.float32) / 255)
                    else:
                        (x1,y1,x2,y2) = tuple(bboxes[id])
                        mask = np.array(Image.open(mask_file.format(test_masks_path, id)).convert('L'), np.float32)
                        mask_bboxed = np.zeros_like(mask, dtype=np.float32)
                        mask_bboxed[y1:y2+1, x1:x2+1] = mask[y1:y2+1, x1:x2+1]
                        masks.append(mask_bboxed / 255)
                        mask_bboxed_img = Image.fromarray(mask_bboxed.astype(np.uint8), mode='L')
                        mask_bboxed_img.save(mask_file.format(reapply_bboxed_path, id))
                        
                q.put((x_ids, None, masks))
                continue
        for id in ids_test_batch:
            x_ids.append(id)
            img = cv2.imread('{0}{1}.jpg'.format(test_path, id))
            if bboxes is not None:
                (x1,y1,x2,y2) = tuple(bboxes[id])
            for i in range(len(models_info)):
                if models_info[i].cropped:
                    input_img = img[y1:y2+1, x1:x2+1, :]
                else:
                    input_img = img
                if models_info[i].input_size is not None:
                    img_resized = cv2.resize(input_img, models_info[i].input_size, interpolation=cv2.INTER_LINEAR)
                else:
                    img_resized = input_img
                x_batches[i][0].append(img_resized)
                x_augments[i][0].append((None, None))
                for j in range(augmentations):
                    img_augmented = utils.randomHueSaturationValue(img_resized,
                                                                 hue_shift_limit=(-50, 50),
                                                                 sat_shift_limit=(-5, 5),
                                                                 val_shift_limit=(-15, 15))
                    img_augmented, trans_mat = utils.randomShiftScaleRotate(img_augmented, None,
                                                                 shift_limit=(-0.0625, 0.0625),
                                                                 scale_limit=(-0.1, 0.1),
                                                                 rotate_limit=(-0, 0))
                    img_augmented, flip = utils.randomHorizontalFlip(img_augmented, None)
                    x_batches[i][j+1].append(img_augmented)
                    x_augments[i][j+1].append((flip, trans_mat))
        x_batches = [[np.array(batches, np.float32) / 255 for batches in batches_model] for batches_model in x_batches]
        q.put((x_ids, x_augments, x_batches))


def predictor(q, graph, rles, orig_size, threshold, models_info, batch_size, ids_len, bboxes, augmentations, test_masks_path):
    total_weights = sum([models_info[i].average_weight * (augmentations+1) for i in range(len(models_info))])
    for i in tqdm(range(0, ids_len, batch_size)):
        (x_ids, x_augments, x_batches) = q.get()
        if x_augments is None:
            masks = x_batches
            for mask in masks:
                rle = run_length_encode(mask)
                rles.append(rle)
            continue
        probs = {id:np.zeros(orig_size[::-1]) for id in x_ids}
        for i in range(len(models_info)):
            for j in range(augmentations+1):
                with graph.as_default():
                    preds = models_info[i].predictor(x_ids, x_batches[i][j])
                preds = np.squeeze(preds, axis=3)
                for (id, aug, pred) in zip(x_ids, x_augments[i][j], preds):
                    pred = utils.reverseFlipShiftScaleRotate(pred, aug[0], aug[1])
                    (x1,y1,x2,y2) = (0,0,0,0) if bboxes is None else tuple(bboxes[id])
                    size = orig_size if not models_info[i].cropped else (x2-x1+1, y2-y1+1)
                    prob = pred if (pred.shape[1], pred.shape[0]) == size else cv2.resize(pred, size, interpolation=cv2.INTER_LINEAR)
                    if size != orig_size:
                        prob_full = np.zeros(orig_size[::-1])
                        prob_full[y1:y2+1, x1:x2+1] = prob
                        prob = prob_full
                    probs[id] = probs[id] + (models_info[i].average_weight * prob)
        for id in x_ids:
            prob = probs[id] / total_weights
            mask = prob > threshold
            if test_masks_path is not None:
                img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
                img.save(mask_file.format(test_masks_path, id))
            rle = run_length_encode(mask)
            rles.append(rle)


def generate_submit_ensemble(models_info, batch_size, threshold, test_path, submit_path, submit_name,
                             augmentations, test_masks_path, bboxes, q_size, reapply_bboxed_path):
    rles = []
    graph = tf.get_default_graph()

    ids_test = [filename[:-4] for filename in os.listdir(test_path)]
    if len(ids_test) == 0:
        print("No test image has been found!")
        return
    img = cv2.imread('{0}{1}.jpg'.format(test_path, ids_test[0]))
    orig_size = (img.shape[1], img.shape[0])

    if bboxes is not None:
        for id in ids_test:
            (x1,y1,x2,y2) = bboxes[id]
            x1,x2 = np.clip((x1,x2), 0, orig_size[0]-1)
            y1,y2 = np.clip((y1,y2), 0, orig_size[1]-1)
            bboxes[id] = (x1,y1,x2,y2)

    names = ['{}.jpg'.format(id) for id in ids_test]

    if (test_masks_path is not None) and (not os.path.exists(test_masks_path)):
        os.makedirs(test_masks_path)

    q = queue.Queue(maxsize=q_size)
    t1 = threading.Thread(target=data_loader, name='DataLoader',
                          args=(q, test_path, batch_size, ids_test, models_info, bboxes, augmentations, test_masks_path, reapply_bboxed_path, ))
    t2 = threading.Thread(target=predictor, name='Predictor', 
                          args=(q, graph, rles, orig_size, threshold, models_info, batch_size, len(ids_test), bboxes, augmentations, test_masks_path, ))
    print('Predicting on {} samples with batch_size = {}...'.format(len(ids_test), batch_size))
    t1.start()
    t2.start()
    # Wait for both threads to finish
    t1.join()
    t2.join()

    print("Generating submission file...")
    df = pd.DataFrame({'img': names, 'rle_mask': rles})
    df.to_csv('{0}/submission-{1}.csv.gz'.format(submit_path, submit_name), index=False, compression='gzip')
    print("Done")

class ModelInfo:
    '''
    predictor: a lambda that receives ids and batch of inputs and returns batch of outputs
    input_size: the size you wish test images be resized before passing to predictor (it can be None if no resizing is required) 
    cropped: whether to use bboxes
    average_weight: weight to use when ensebling predictions
    '''
    def __init__(self, predictor, input_size, cropped=False, average_weight=1):
        self.predictor = predictor
        self.input_size = input_size if type(input_size) is tuple else (input_size, input_size)
        self.cropped = cropped
        self.average_weight = average_weight

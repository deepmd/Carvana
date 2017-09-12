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

def data_loader(q, test_path, batch_size, ids_test, input_size, bboxes):
    for start in range(0, len(ids_test), batch_size):
        x_batch = []
        x_ids = []
        end = min(start + batch_size, len(ids_test))
        ids_test_batch = ids_test[start:end]
        for id in ids_test_batch:
            img = cv2.imread('{0}{1}.jpg'.format(test_path, id))
            if bboxes is not None:
                (x1,y1,x2,y2) = tuple(bboxes[id])
                img = img[y1:y2+1, x1:x2+1, :]
            if input_size is not None:
                img = cv2.resize(img, input_size, interpolation=cv2.INTER_LINEAR)
            x_batch.append(img)
            x_ids.append(id)
        x_batch = np.array(x_batch, np.float32) / 255
        q.put((x_ids, x_batch))


def predictor(q, graph, rles, orig_size, threshold, model, batch_size, ids_len, bboxes, test_masks_path):
    for i in tqdm(range(0, ids_len, batch_size)):
        (x_ids, x_batch) = q.get()
        with graph.as_default():
            preds = model.predict_on_batch(x_batch)
        preds = np.squeeze(preds, axis=3)
        for (id, pred) in zip(x_ids, preds):
            (x1,y1,x2,y2) = (0,0,0,0) if bboxes is None else tuple(bboxes[id])
            size = orig_size if bboxes is None else (x2-x1+1, y2-y1+1)
            prob = pred if (pred.shape[1], pred.shape[0]) == size else cv2.resize(pred, size, interpolation=cv2.INTER_LINEAR)
            mask = prob > threshold
            if size != orig_size:
                mask_full = np.zeros(orig_size[::-1])
                mask_full[y1:y2+1, x1:x2+1] = mask
                mask = mask_full
            if test_masks_path is not None:
                img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
                img.save('{0}{1}.gif'.format(test_masks_path, id))
            rle = run_length_encode(mask)
            rles.append(rle)


def generate_submit(model, input_size, batch_size, threshold, test_path, submit_path,
                    run_name, test_masks_path=None, bboxes=None):
    q_size = 10
    rles = []
    graph = tf.get_default_graph()

    ids_test = [filename[:-4] for filename in os.listdir(test_path)]
    if len(ids_test) == 0:
        print("No test image has been found!")
        return
    img = cv2.imread('{0}{1}.jpg'.format(test_path, ids_test[0]))
    orig_size = (img.shape[1], img.shape[0])

    names = ['{}.jpg'.format(id) for id in ids_test]

    if (test_masks_path is not None) and (not os.path.exists(test_masks_path)):
        os.makedirs(test_masks_path)

    q = queue.Queue(maxsize=q_size)
    t1 = threading.Thread(target=data_loader, name='DataLoader',
                          args=(q, test_path, batch_size, ids_test, (input_size, input_size), bboxes, ))
    t2 = threading.Thread(target=predictor, name='Predictor', 
                          args=(q, graph, rles, orig_size, threshold, model, batch_size, len(ids_test), bboxes, test_masks_path, ))
    print('Predicting on {} samples with batch_size = {}...'.format(len(ids_test), batch_size))
    t1.start()
    t2.start()
    # Wait for both threads to finish
    t1.join()
    t2.join()

    print("Generating submission file...")
    df = pd.DataFrame({'img': names, 'rle_mask': rles})
    df.to_csv('{0}/submission-{1}.csv.gz'.format(submit_path, run_name), index=False, compression='gzip')
    print("Done")

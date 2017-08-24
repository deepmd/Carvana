import cv2
import numpy as np
import pandas as pd
import threading
import Queue as queue
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

def data_loader(q, test_path, batch_size, ids_test, input_size, ):
    for start in range(0, len(ids_test), batch_size):
        x_batch = []
        end = min(start + batch_size, len(ids_test))
        ids_test_batch = ids_test[start:end]
        for id in ids_test_batch:
            img = cv2.imread('{0}{1}.jpg'.format(test_path, id))
            img = cv2.resize(img, input_size)
            x_batch.append(img)
        x_batch = np.array(x_batch, np.float32) / 255
        q.put(x_batch)


def predictor(q, graph, rles, orig_size, threshold, model, batch_size, ids_len):
    for i in tqdm(range(0, ids_len, batch_size)):
        x_batch = q.get()
        with graph.as_default():
            preds = model.predict_on_batch(x_batch)
        preds = np.squeeze(preds, axis=3)
        for pred in preds:
            prob = cv2.resize(pred, orig_size)
            mask = prob > threshold
            rle = run_length_encode(mask)
            rles.append(rle)
    return rle


def generate_submit(input_size, orig_size, batch_size, threshold, model, test_path, submit_path, run_name):
    q_size = 10
    rles = []
    graph = tf.get_default_graph()

    ids_test = [filename[:-4] for filename in os.listdir(test_path)]

    names = []
    for id in ids_test:
        names.append('{}.jpg'.format(id))


    q = queue.Queue(maxsize=q_size)
    t1 = threading.Thread(target=data_loader, name='DataLoader', args=(q, test_path, batch_size, ids_test, (input_size, input_size), ))
    t2 = threading.Thread(target=predictor, name='Predictor', args=(q, graph, rles, orig_size, threshold, model, batch_size, len(ids_test), ))
    print('Predicting on {} samples with batch_size = {}...'.format(len(ids_test), batch_size))
    t1.start()
    t2.start()
    # Wait for both threads to finish
    t1.join()
    t2.join()

    print("Generating submission file...")
    df = pd.DataFrame({'img': names, 'rle_mask': rles})
    df.to_csv('submits/submission-{}.csv.gz'.format(run_name), index=False, compression='gzip')
    print("Done")

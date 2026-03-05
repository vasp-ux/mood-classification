"""
compare_models.py  - Final version with exact new model architecture.
NEW model: Conv(32)->Conv(64)->Flatten->Dense(128)->Dense(8)  [1-channel input, no BN]
OLD model: loaded directly (3-channel input)
"""

import os, sys, glob, random, zipfile, tempfile, shutil
import numpy as np, cv2, h5py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten

PROJECT  = r'c:\Users\vinuk\OneDrive\Desktop\mood160226-master'
OLD_PATH = os.path.join(PROJECT, 'visual_based', 'emotion_model.keras')
NEW_PATH = os.path.join(PROJECT, 'mood-classification-main', 'visual', 'emotion_model.keras')
TEST_DIR = os.path.join(PROJECT, 'mood-classification-main', 'visual', 'fer2013', 'test')
EMOTIONS = ['angry', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']


def build_new_model():
    # Exact architecture from h5 weight shapes: conv(32)->conv(64)->dense(128)->dense(8)
    return Sequential([
        Conv2D(32,  (3,3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64,  (3,3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(8,   activation='softmax'),
    ])


def load_new_model(keras_path):
    tmp = tempfile.mkdtemp()
    try:
        with zipfile.ZipFile(keras_path, 'r') as z:
            z.extract('model.weights.h5', tmp)
        h5_path = os.path.join(tmp, 'model.weights.h5')

        weight_arrays = []
        with h5py.File(h5_path, 'r') as f:
            for lname in sorted(f['layers'].keys()):
                node = f[f'layers/{lname}']
                if 'vars' not in node or len(node['vars']) == 0:
                    continue
                for vk in sorted(node['vars'].keys(), key=int):
                    weight_arrays.append(np.array(node['vars'][vk]))

        m = build_new_model()
        trainable = m.trainable_weights
        print(f"  h5 tensors: {len(weight_arrays)}, model trainable: {len(trainable)}")
        if len(weight_arrays) != len(trainable):
            print("  Mismatch!")
            return None
        for w, arr in zip(trainable, weight_arrays):
            w.assign(arr)
        return m
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def preprocess(img_path, channels):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (48, 48)).astype('float32') / 255.0
    img = np.stack([img]*3, axis=-1) if channels == 3 else np.expand_dims(img, -1)
    return np.expand_dims(img, 0)


# ===== MAIN =====
print("=" * 52)
print("LOADING MODELS")
print("=" * 52)
old = tf.keras.models.load_model(OLD_PATH, compile=False)
print(f"OLD ok | input={old.input_shape}  params={old.count_params():,}")

new = load_new_model(NEW_PATH)
if new is None:
    sys.exit(1)
print(f"NEW ok | input={new.input_shape}  params={new.count_params():,}")

old_ch = old.input_shape[-1]
new_ch = new.input_shape[-1]

print()
print("=" * 52)
print("PER-CLASS ACCURACY  (30 random test images per class)")
print("=" * 52)
print(f"{'Class':<13} {'OLD':>8} {'NEW':>8}")
print("-" * 35)

o_total = n_total = grand = 0

for edir in sorted(os.listdir(TEST_DIR)):
    epath = os.path.join(TEST_DIR, edir)
    if not os.path.isdir(epath):
        continue
    imgs   = glob.glob(os.path.join(epath,'*.jpg')) + glob.glob(os.path.join(epath,'*.png'))
    sample = random.sample(imgs, min(30, len(imgs)))
    el     = edir.lower()
    label  = next((i for i,e in enumerate(EMOTIONS) if e in el or el in e), None)
    if label is None:
        print(f"  skip: {edir}")
        continue

    oc = nc = 0
    for p in sample:
        oi = preprocess(p, old_ch)
        ni = preprocess(p, new_ch)
        if oi is None:
            continue
        if int(np.argmax(old.predict(oi, verbose=0)[0])) == label:
            oc += 1
        if int(np.argmax(new.predict(ni, verbose=0)[0])) == label:
            nc += 1

    n = len(sample)
    o_total += oc; n_total += nc; grand += n
    print(f"{edir:<13} {oc/n*100:7.1f}%  {nc/n*100:7.1f}%")

print("-" * 35)
oa = o_total/grand*100
na = n_total/grand*100
print(f"{'OVERALL':<13} {oa:7.1f}%  {na:7.1f}%")
print()
print(f"File size  ->  OLD: {os.path.getsize(OLD_PATH)/1e6:.1f} MB  |  NEW: {os.path.getsize(NEW_PATH)/1e6:.1f} MB")
print(f"Parameters ->  OLD: {old.count_params():,}  |  NEW: {new.count_params():,}")
print()
if n_total > o_total:
    print(f"VERDICT: NEW model is better by {na-oa:.1f}%  ->  consider replacing")
elif o_total > n_total:
    print(f"VERDICT: OLD model is better by {oa-na:.1f}%  ->  keep existing")
else:
    print("VERDICT: TIE")

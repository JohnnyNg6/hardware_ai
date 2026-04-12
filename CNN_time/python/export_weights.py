#!/usr/bin/env python3
"""Train CIFAR-10 CNN and export per-neuron Q8.8 .mem files."""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

EPOCHS, BATCH_SIZE = 128, 32

(tr_img, tr_lbl), (te_img, te_lbl) = tf.keras.datasets.cifar10.load_data()
MEAN = np.mean(tr_img).astype(np.float64)
STD  = np.std(tr_img).astype(np.float64)
tr_img = (tr_img - MEAN) / STD
te_img = (te_img - MEAN) / STD
tr_lbl = to_categorical(tr_lbl, 10)
te_lbl = to_categorical(te_lbl, 10)

model = Sequential([
    Conv2D(64, (5,5), strides=2, activation='relu', padding='same',
           input_shape=(32,32,3), kernel_initializer='he_normal'),
    Conv2D(64, (3,3), strides=2, activation='relu', padding='same',
           kernel_initializer='he_normal'),
    Flatten(),
    Dense(10, activation='softmax', kernel_initializer='glorot_uniform'),
])
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
model.summary()
model.fit(tr_img, tr_lbl, validation_data=(te_img, te_lbl),
          epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, shuffle=True)

loss, acc = model.evaluate(te_img, te_lbl, verbose=0)
print(f"\n>>> Software test accuracy: {acc:.4f} <<<\n")

np.savez("cifar_norm.npz", mean=MEAN, std=STD)
model.save("cifar_cnn_model.h5")

def write_neuron_mem(fname, bias_f, weights_f):
    """Write bias + weights in Q8.8 hex, one value per line."""
    with open(fname, 'w') as f:
        b = int(np.clip(np.round(bias_f * 256), -32768, 32767))
        f.write(f"{b & 0xFFFF:04x}\n")
        for v in weights_f.flatten():
            q = int(np.clip(np.round(v * 256), -32768, 32767))
            f.write(f"{q & 0xFFFF:04x}\n")

# ---------- Conv1 ----------
# img_buf is HWC (raw UART order), TF kernel is (FH,FW,IC) = HWC → matches
w1, b1 = model.layers[0].get_weights()   # w1: (5,5,3,64)  b1: (64,)
for f in range(64):
    write_neuron_mem(f"conv1_n{f}.mem", b1[f], w1[:,:,:,f])

# ---------- Conv2 ----------
# c1_buf is CHW (each filter wrote its own spatial plane)
# TF kernel per filter: (FH,FW,IC) = (3,3,64) → HWC
# Hardware feeds in CHW order (ic outer loop) → need (IC,FH,FW)
w2, b2 = model.layers[1].get_weights()   # w2: (3,3,64,64)  b2: (64,)
for f in range(64):
    kernel_hwc = w2[:,:,:,f]                   # (3, 3, 64)  = (FH, FW, IC)
    kernel_chw = kernel_hwc.transpose(2,0,1)   # (64, 3, 3)  = (IC, FH, FW)
    write_neuron_mem(f"conv2_n{f}.mem", b2[f], kernel_chw)

# ---------- Dense ----------
# c2_buf is CHW: addr = ch*8*8 + row*8 + col
# TF Flatten is HWC: index = row*8*64 + col*64 + ch
# Must reorder weights from HWC to CHW
wd, bd = model.layers[3].get_weights()   # wd: (4096,10)  bd: (10,)
for c in range(10):
    w_hwc = wd[:, c].reshape(8, 8, 64)        # (H, W, C)
    w_chw = w_hwc.transpose(2, 0, 1).flatten() # (C, H, W) → flat
    write_neuron_mem(f"dense_n{c}.mem", bd[c], w_chw)

print("All weight files exported.")
print("Conv1: HWC order (unchanged)")
print("Conv2: reordered to CHW (IC, FH, FW)")
print("Dense: reordered to CHW (C, H, W)")

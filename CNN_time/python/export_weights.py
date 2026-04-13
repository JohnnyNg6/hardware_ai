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
tr_img = (tr_img - MEAN) / STD;  te_img = (te_img - MEAN) / STD
tr_lbl = to_categorical(tr_lbl, 10);  te_lbl = to_categorical(te_lbl, 10)

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

np.savez("cifar_norm.npz", mean=MEAN, std=STD)
model.save("cifar_cnn_model.h5")

def write_neuron_mem(fname, bias_f, weights_f):
    with open(fname, 'w') as f:
        b = int(np.clip(np.round(bias_f * 256), -32768, 32767))
        f.write(f"{b & 0xFFFF:04x}\n")
        for v in weights_f.flatten():
            q = int(np.clip(np.round(v * 256), -32768, 32767))
            f.write(f"{q & 0xFFFF:04x}\n")

w1, b1 = model.layers[0].get_weights()
for f in range(64):
    write_neuron_mem(f"conv1_n{f}.mem", b1[f], w1[:,:,:,f])

w2, b2 = model.layers[1].get_weights()
for f in range(64):
    write_neuron_mem(f"conv2_n{f}.mem", b2[f], w2[:,:,:,f])

wd, bd = model.layers[3].get_weights()
for c in range(10):
    write_neuron_mem(f"dense_n{c}.mem", bd[c], wd[:, c])

print("\nAll weight files exported.")

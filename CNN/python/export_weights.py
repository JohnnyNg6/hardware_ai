#!/usr/bin/env python3
"""
export_weights.py
Train the CIFAR-10 CNN and export all weights/biases as Q8.8 hex .mem files.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import struct, os

# ── 1. Train ──────────────────────────────────────────────────────────────
EPOCHS     = 128
BATCH_SIZE = 32

(train_img, train_lbl), (test_img, test_lbl) = tf.keras.datasets.cifar10.load_data()
MEAN   = np.mean(train_img).astype(np.float64)
STDDEV = np.std(train_img).astype(np.float64)
train_img = (train_img - MEAN) / STDDEV
test_img  = (test_img  - MEAN) / STDDEV
train_lbl = to_categorical(train_lbl, 10)
test_lbl  = to_categorical(test_lbl,  10)

model = Sequential([
    Conv2D(64, (5,5), strides=(2,2), activation='relu',
           input_shape=(32,32,3),
           kernel_initializer='he_normal', bias_initializer='zeros'),
    Conv2D(64, (3,3), strides=(2,2), activation='relu', padding='same',
           kernel_initializer='he_normal', bias_initializer='zeros'),
    Flatten(),
    Dense(10, activation='softmax',
          kernel_initializer='glorot_uniform', bias_initializer='zeros'),
])
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
model.summary()
model.fit(train_img, train_lbl,
          validation_data=(test_img, test_lbl),
          epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, shuffle=True)

# Save normalization constants for send_image.py
np.savez("cifar_norm.npz", mean=MEAN, std=STDDEV)
model.save("cifar_cnn_model.h5")

# ── 2. Float → Q8.8 ──────────────────────────────────────────────────────
def to_q88(arr):
    """Convert float array → signed 16-bit Q8.8 integers."""
    q = np.round(arr * 256.0).astype(np.int64)
    q = np.clip(q, -32768, 32767)
    return q.astype(np.int16)

def write_mem(filename, q_arr):
    """Write 1-D Q8.8 int16 array to hex .mem file."""
    with open(filename, 'w') as f:
        for v in q_arr.flatten():
            f.write(f"{v & 0xFFFF:04X}\n")
    print(f"  {filename}: {q_arr.size} entries")

# ── 3. Export Conv1 ───────────────────────────────────────────────────────
conv1_w, conv1_b = model.layers[0].get_weights()   # [5,5,3,64], [64]
# FPGA layout: [NF, FH, FW, IC]
conv1_w_t = np.transpose(conv1_w, (3, 0, 1, 2))    # → [64,5,5,3]
write_mem("conv1_w.mem", to_q88(conv1_w_t))
write_mem("conv1_b.mem", to_q88(conv1_b))

# ── 4. Export Conv2 ───────────────────────────────────────────────────────
conv2_w, conv2_b = model.layers[1].get_weights()   # [3,3,64,64], [64]
conv2_w_t = np.transpose(conv2_w, (3, 0, 1, 2))    # → [64,3,3,64]
write_mem("conv2_w.mem", to_q88(conv2_w_t))
write_mem("conv2_b.mem", to_q88(conv2_b))

# ── 5. Export Dense ───────────────────────────────────────────────────────
dense_w, dense_b = model.layers[3].get_weights()   # [3136,10], [10]
# (layers[2] is Flatten, no weights)
dense_w_t = np.transpose(dense_w, (1, 0))           # → [10,3136]
write_mem("dense_w.mem", to_q88(dense_w_t))
write_mem("dense_b.mem", to_q88(dense_b))

print("\nAll .mem files written. Copy them to the Vivado project directory.")
print(f"Normalization: mean={MEAN:.6f}  std={STDDEV:.6f}")

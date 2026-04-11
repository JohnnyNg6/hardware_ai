"""
export_weights.py
Converts trained Keras conf5 model weights → Q8.8 hex .mem files
for neuron.v.

Run after training:
    python export_weights.py

Generates:
    h_00.mem … h_24.mem   (hidden layer, 785 lines each: bias + 784 weights)
    o_00.mem … o_09.mem   (output layer,  26 lines each: bias +  25 weights)
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import to_categorical

# ---------- helper ----------
def float_to_q88_hex(f):
    """float → 16-bit signed Q8.8 → 4-char hex string"""
    v = int(round(f * 256.0))                     # scale by 2^8
    v = max(-32768, min(32767, v))                 # clamp to int16
    return f"{v & 0xFFFF:04X}"                     # two's complement hex

# ---------- train (or load) the model ----------
mnist = keras.datasets.mnist
(trn_img, trn_lbl), (tst_img, tst_lbl) = mnist.load_data()

mean   = np.mean(trn_img)
stddev = np.std(trn_img)
trn_img = (trn_img - mean) / stddev
tst_img = (tst_img - mean) / stddev

trn_lbl = to_categorical(trn_lbl, 10)
tst_lbl = to_categorical(tst_lbl, 10)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(25, activation='relu',
                       kernel_initializer='he_normal',
                       bias_initializer='zeros'),
    keras.layers.Dense(10, activation='softmax',
                       kernel_initializer='glorot_uniform',
                       bias_initializer='zeros'),
])
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.fit(trn_img, trn_lbl,
          validation_data=(tst_img, tst_lbl),
          epochs=20, batch_size=64, verbose=2)

# ---------- export ----------
for layer_idx, (prefix, num_neurons) in enumerate(
        [("h", 25), ("o", 10)]):
    W, b = model.layers[layer_idx + 1].get_weights()   # skip Flatten
    # W shape: (num_inputs, num_neurons)   b shape: (num_neurons,)
    for n in range(num_neurons):
        fname = f"{prefix}_{n:02d}.mem"
        with open(fname, "w") as f:
            f.write(float_to_q88_hex(b[n]) + "\n")     # bias first
            for j in range(W.shape[0]):
                f.write(float_to_q88_hex(W[j, n]) + "\n")
        print(f"  {fname}  ({W.shape[0]+1} entries)")

print(f"\nDone.  Mean={mean:.2f}  Stddev={stddev:.2f}")
print("Remember to normalise FPGA inputs: pixel_q88 = (pixel - mean) / stddev")

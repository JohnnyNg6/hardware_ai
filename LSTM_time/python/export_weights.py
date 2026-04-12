#!/usr/bin/env python3
"""
export_weights.py
Train the LSTM model on Frankenstein text, then export all weights,
activation LUTs, and character mapping tables as .mem files for FPGA.
"""
import numpy as np
import os, struct
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import tensorflow as tf, logging
tf.get_logger().setLevel(logging.ERROR)

# ── hyper-parameters (must match notebook) ──────────────────────────
EPOCHS        = 32
BATCH_SIZE    = 256
INPUT_FILE    = 'frankenstein.txt'
WINDOW_LENGTH = 40
WINDOW_STEP   = 3
HIDDEN_SIZE   = 128
OUT_DIR       = 'weights'

os.makedirs(OUT_DIR, exist_ok=True)

# ── read & preprocess text ──────────────────────────────────────────
with open(INPUT_FILE, 'r', encoding='utf-8-sig') as f:
    text = f.read()
text = text.lower().replace('\n', ' ').replace('  ', ' ')

unique_chars   = sorted(set(text))
char_to_index  = {ch: i for i, ch in enumerate(unique_chars)}
index_to_char  = {i: ch for i, ch in enumerate(unique_chars)}
ENC_WIDTH      = len(char_to_index)
print(f"Encoding width = {ENC_WIDTH}")

# ── build training data ─────────────────────────────────────────────
fragments, targets = [], []
for i in range(0, len(text) - WINDOW_LENGTH, WINDOW_STEP):
    fragments.append(text[i:i+WINDOW_LENGTH])
    targets.append(text[i+WINDOW_LENGTH])

X = np.zeros((len(fragments), WINDOW_LENGTH, ENC_WIDTH), dtype=np.float32)
y = np.zeros((len(fragments), ENC_WIDTH),                dtype=np.float32)
for i, frag in enumerate(fragments):
    for j, ch in enumerate(frag):
        X[i, j, char_to_index[ch]] = 1.0
    y[i, char_to_index[targets[i]]] = 1.0

# ── build & train model ─────────────────────────────────────────────
model = Sequential([
    LSTM(HIDDEN_SIZE, return_sequences=True,
         dropout=0.2, recurrent_dropout=0.2,
         input_shape=(None, ENC_WIDTH)),
    LSTM(HIDDEN_SIZE, dropout=0.2, recurrent_dropout=0.2),
    Dense(ENC_WIDTH, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()
model.fit(X, y, validation_split=0.05, batch_size=BATCH_SIZE,
          epochs=EPOCHS, verbose=2, shuffle=True)

# ── Q8.8 helper ─────────────────────────────────────────────────────
def to_q88(val):
    """Convert float to signed Q8.8 (16-bit), clamped."""
    v = int(round(val * 256.0))
    v = max(-32768, min(32767, v))
    return v & 0xFFFF          # unsigned representation for hex file

def write_mem(fname, values_uint16):
    with open(fname, 'w') as f:
        for v in values_uint16:
            f.write(f"{v:04x}\n")

# ── export LSTM layer weights ───────────────────────────────────────
def export_lstm_layer(layer, layer_id, input_size, hidden_size):
    """
    Keras LSTM weights:
      kernel          : (input_size,  4*hidden_size)  gate order: i f g o
      recurrent_kernel: (hidden_size, 4*hidden_size)
      bias            : (4*hidden_size,)
    Our gate order in HW: f i g o  (indices in Keras: 1 0 2 3)
    """
    W_x, W_h, bias = layer.get_weights()
    gate_map = [1, 0, 2, 3]      # HW gate idx -> Keras gate idx

    concat_size = input_size + hidden_size
    gate_words  = concat_size + 1   # bias + concat weights

    for uid in range(hidden_size):
        mem = []
        for hw_g in range(4):                       # f, i, g, o
            kg = gate_map[hw_g]
            col = kg * hidden_size + uid
            b   = bias[col]
            wx  = W_x[:, col]                       # (input_size,)
            wh  = W_h[:, col]                       # (hidden_size,)
            mem.append(to_q88(b))
            for v in wx:
                mem.append(to_q88(v))
            for v in wh:
                mem.append(to_q88(v))
        assert len(mem) == 4 * gate_words
        write_mem(os.path.join(OUT_DIR, f"l{layer_id}_u{uid}.mem"), mem)
    print(f"  Layer {layer_id}: exported {hidden_size} unit weight files "
          f"({4*gate_words} words each)")

# ── export Dense layer weights ──────────────────────────────────────
def export_dense_layer(layer, hidden_size, output_size):
    W, bias = layer.get_weights()         # W:(hidden, output), bias:(output,)
    for oid in range(output_size):
        mem = [to_q88(bias[oid])]
        for v in W[:, oid]:
            mem.append(to_q88(v))
        assert len(mem) == hidden_size + 1
        write_mem(os.path.join(OUT_DIR, f"dense_u{oid}.mem"), mem)
    print(f"  Dense : exported {output_size} neuron weight files "
          f"({hidden_size+1} words each)")

# ── export activation LUTs ──────────────────────────────────────────
def export_sigmoid_lut():
    mem = []
    for i in range(512):
        x = -8.0 + i * (16.0 / 512.0)
        y = 1.0 / (1.0 + np.exp(-x))
        mem.append(to_q88(y))
    write_mem(os.path.join(OUT_DIR, "sigmoid_lut.mem"), mem)
    print("  Sigmoid LUT exported (512 entries)")

def export_tanh_lut():
    mem = []
    for i in range(512):
        x = -8.0 + i * (16.0 / 512.0)
        y = np.tanh(x)
        mem.append(to_q88(y))
    write_mem(os.path.join(OUT_DIR, "tanh_lut.mem"), mem)
    print("  Tanh LUT exported (512 entries)")

# ── export character mapping ────────────────────────────────────────
def export_char_map():
    # ASCII code (0-127) → char index (0..ENC_WIDTH-1), 0xFF = unmapped
    mem = [0xFF] * 128
    for ch, idx in char_to_index.items():
        mem[ord(ch) & 0x7F] = idx
    write_mem(os.path.join(OUT_DIR, "char_to_idx.mem"),
              [v & 0xFFFF for v in mem])
    # Index → ASCII
    mem2 = []
    for i in range(ENC_WIDTH):
        mem2.append(ord(index_to_char[i]) & 0xFFFF)
    write_mem(os.path.join(OUT_DIR, "idx_to_char.mem"), mem2)
    print(f"  Char maps exported (enc_width={ENC_WIDTH})")

# ── export Verilog parameters header ────────────────────────────────
def export_params():
    with open(os.path.join(OUT_DIR, "params.vh"), 'w') as f:
        f.write(f"// Auto-generated by export_weights.py\n")
        f.write(f"localparam ENC_WIDTH   = {ENC_WIDTH};\n")
        f.write(f"localparam HIDDEN_SIZE = {HIDDEN_SIZE};\n")
        f.write(f"localparam CONCAT1     = {ENC_WIDTH + HIDDEN_SIZE};\n")
        f.write(f"localparam CONCAT2     = {HIDDEN_SIZE + HIDDEN_SIZE};\n")
    print("  params.vh exported")

# ── run all exports ─────────────────────────────────────────────────
print("Exporting weights …")
export_lstm_layer(model.layers[0], 1, ENC_WIDTH,   HIDDEN_SIZE)
export_lstm_layer(model.layers[1], 2, HIDDEN_SIZE,  HIDDEN_SIZE)
export_dense_layer(model.layers[2], HIDDEN_SIZE, ENC_WIDTH)
export_sigmoid_lut()
export_tanh_lut()
export_char_map()
export_params()
print("Done. All files in", OUT_DIR)

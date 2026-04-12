#!/usr/bin/env python3
"""Send a CIFAR-10 test image to FPGA via UART, print result."""
import sys, serial, struct, numpy as np, tensorflow as tf

PORT = sys.argv[1] if len(sys.argv) > 1 else '/dev/ttyUSB0'
IDX  = int(sys.argv[2]) if len(sys.argv) > 2 else 0

npz  = np.load("cifar_norm.npz")
MEAN, STD = float(npz['mean']), float(npz['std'])

(_, _), (te_img, te_lbl) = tf.keras.datasets.cifar10.load_data()
img = ((te_img[IDX].astype(np.float64) - MEAN) / STD)
q88 = np.clip(np.round(img * 256), -32768, 32767).astype(np.int16)

ser = serial.Serial(PORT, 115200, timeout=5)
pkt = b'\xAA'
for v in q88.flatten():
    pkt += struct.pack('>h', int(v))         # big-endian signed 16
ser.write(pkt)
print(f"Sent image #{IDX}  (true label = {te_lbl[IDX][0]})")

r = ser.read(1)
if r:
    print(f"FPGA predicted: {r[0]}")
else:
    print("Timeout — no response from FPGA")
ser.close()

#!/usr/bin/env python3
"""
send_image.py — Send a CIFAR-10 test image to the FPGA and read back the prediction.

Usage:  python send_image.py [--port COM3] [--index 0]
"""
import argparse, struct, time, sys
import numpy as np
import serial
import tensorflow as tf

CLASSES = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']

def to_q88_bytes(pixel_float):
    """Float → Q8.8 → 2 bytes big-endian."""
    q = int(round(pixel_float * 256.0))
    q = max(-32768, min(32767, q))
    return struct.pack('>h', q)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',  default='COM3',   help='Serial port')
    parser.add_argument('--baud',  default=115200,   type=int)
    parser.add_argument('--index', default=0,        type=int,
                        help='CIFAR-10 test image index (0–9999)')
    args = parser.parse_args()

    # Load CIFAR-10
    (_, _), (test_img, test_lbl) = tf.keras.datasets.cifar10.load_data()

    # Load normalization constants
    try:
        d = np.load("cifar_norm.npz")
        MEAN, STD = float(d['mean']), float(d['std'])
    except FileNotFoundError:
        MEAN, STD = 120.70756512369792, 64.1500758911213
        print(f"[warn] cifar_norm.npz not found, using defaults")

    idx = args.index % len(test_img)
    img = test_img[idx].astype(np.float64)
    lbl = int(test_lbl[idx])
    img_norm = (img - MEAN) / STD     # shape (32,32,3)

    print(f"Sending test image #{idx}  true label = {lbl} ({CLASSES[lbl]})")

    # Build payload: 0xAA + 3072×2 bytes
    payload = bytearray([0xAA])
    for h in range(32):
        for w in range(32):
            for c in range(3):
                payload += to_q88_bytes(img_norm[h, w, c])

    print(f"Payload size: {len(payload)} bytes")

    ser = serial.Serial(args.port, args.baud, timeout=30)
    time.sleep(0.1)
    ser.reset_input_buffer()

    t0 = time.time()
    ser.write(payload)
    print("Image sent, waiting for prediction...")

    resp = ser.read(1)
    t1 = time.time()

    if len(resp) == 0:
        print("ERROR: timeout, no response from FPGA")
    else:
        pred = resp[0]
        print(f"FPGA prediction: {pred} ({CLASSES[pred]})  "
              f"correct={pred==lbl}  time={t1-t0:.3f}s")

    ser.close()

if __name__ == '__main__':
    main()

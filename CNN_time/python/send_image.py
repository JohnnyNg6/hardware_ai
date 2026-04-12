#!/usr/bin/env python3
"""Send CIFAR-10 test images to FPGA via UART, print results."""
import sys, serial, struct, time, numpy as np, tensorflow as tf

PORT  = sys.argv[1] if len(sys.argv) > 1 else '/dev/ttyUSB0'
START = int(sys.argv[2]) if len(sys.argv) > 2 else 0
COUNT = int(sys.argv[3]) if len(sys.argv) > 3 else 1

npz  = np.load("cifar_norm.npz")
MEAN, STD = float(npz['mean']), float(npz['std'])

(_, _), (te_img, te_lbl) = tf.keras.datasets.cifar10.load_data()

CLASSES = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']

correct = 0
timeout_count = 0
total = COUNT

ser = serial.Serial(PORT, 115200, timeout=5)

for idx in range(START, START + COUNT):
    img = ((te_img[idx].astype(np.float64) - MEAN) / STD)
    q88 = np.clip(np.round(img * 256), -32768, 32767).astype(np.int16)

    pkt = b'\xAA'
    for v in q88.flatten():
        pkt += struct.pack('>h', int(v))
    ser.write(pkt)

    r = ser.read(1)
    true_label = int(te_lbl[idx][0])

    if r:
        pred = r[0]
        match = "OK" if pred == true_label else "WRONG"
        if pred == true_label:
            correct += 1
        print(f"[{idx:5d}] true={true_label} ({CLASSES[true_label]:10s})  "
              f"pred={pred} ({CLASSES[pred]:10s})  {match}")
    else:
        timeout_count += 1
        print(f"[{idx:5d}] true={true_label} ({CLASSES[true_label]:10s})  "
              f"pred=TIMEOUT")

    time.sleep(0.2)

ser.close()

if total > 1:
    print()
    print("=" * 50)
    print(f"  Total:    {total}")
    print(f"  Correct:  {correct}")
    print(f"  Wrong:    {total - correct - timeout_count}")
    print(f"  Timeout:  {timeout_count}")
    print(f"  Accuracy: {correct / total * 100:.1f}%")
    print("=" * 50)

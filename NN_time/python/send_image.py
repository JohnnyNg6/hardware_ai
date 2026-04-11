#!/usr/bin/env python3
"""
send_image.py — Send MNIST images to FPGA over UART and read predictions.

Usage:
    python send_image.py [serial_port] [num_images]

Examples:
    python send_image.py                        # Linux default /dev/ttyUSB0
    python send_image.py COM5                   # Windows
    python send_image.py /dev/ttyUSB0 100       # test 100 images
"""

import serial
import struct
import numpy as np
import time
import sys

# ===== Configuration =====
DEFAULT_PORT = '/dev/ttyUSB0'      # Linux; Windows: 'COM3', 'COM5', etc.
BAUD_RATE    = 115200
SYNC_BYTE    = 0xAA
FRAC_BITS    = 8
TIMEOUT_SEC  = 5

# Normalisation values — must match export_weights.py training run.
# For standard MNIST uint8 training set these are nearly constant:
MNIST_MEAN   = 33.318421
MNIST_STDDEV = 78.567398


def float_to_q88(x):
    """Convert a float to unsigned 16-bit representation of signed Q8.8."""
    v = int(round(x * (1 << FRAC_BITS)))
    v = max(-32768, min(32767, v))
    if v < 0:
        v += 65536
    return v


def load_mnist():
    """Load MNIST test set. Falls back to random data if TF unavailable."""
    try:
        from tensorflow.keras.datasets import mnist
        (_, _), (x_test, y_test) = mnist.load_data()
        return x_test, y_test
    except ImportError:
        pass
    try:
        import gzip, os, urllib.request
        # Try raw IDX files if present
        raise ImportError
    except Exception:
        print("[WARN] Could not load MNIST. Using random test data.")
        dummy_x = np.random.randint(0, 256, (10, 28, 28), dtype=np.uint8)
        dummy_y = np.zeros(10, dtype=np.int64)
        return dummy_x, dummy_y


def send_image(ser, raw_image_28x28):
    """Send one 28×28 uint8 image; return predicted digit or None."""
    # Normalise (same transform as training)
    pixels = raw_image_28x28.flatten().astype(np.float64)
    pixels = (pixels - MNIST_MEAN) / MNIST_STDDEV

    # Convert to Q8.8 big-endian bytes
    packet = bytearray([SYNC_BYTE])
    for p in pixels:
        packet += struct.pack('>H', float_to_q88(p))

    ser.write(packet)
    ser.flush()

    # Wait for 1-byte result
    resp = ser.read(1)
    if len(resp) == 1:
        return resp[0]
    return None


def main():
    port      = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PORT
    num_test  = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    x_test, y_test = load_mnist()
    num_test = min(num_test, len(x_test))

    ser = serial.Serial(port, BAUD_RATE, timeout=TIMEOUT_SEC)
    time.sleep(0.2)                        # let FPGA / CH340 settle

    print(f"Connected: {port} @ {BAUD_RATE} baud")
    print(f"Normalisation: mean={MNIST_MEAN:.3f}  std={MNIST_STDDEV:.3f}")
    print(f"Testing {num_test} images")
    print("=" * 52)

    correct = 0
    t0 = time.time()

    for i in range(num_test):
        label = int(y_test[i])
        predicted = send_image(ser, x_test[i])

        if predicted is not None:
            ok = (predicted == label)
            correct += ok
            mark = "✓" if ok else "✗"
            print(f"  [{i:5d}]  true={label}  pred={predicted}  {mark}")
        else:
            print(f"  [{i:5d}]  true={label}  pred=TIMEOUT")

        time.sleep(0.01)

    elapsed = time.time() - t0
    print("=" * 52)
    print(f"Accuracy : {correct}/{num_test} = {100*correct/num_test:.1f}%")
    print(f"Elapsed  : {elapsed:.1f}s  ({elapsed/num_test*1000:.0f} ms/image)")

    ser.close()


if __name__ == '__main__':
    main()

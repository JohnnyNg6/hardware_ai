#!/usr/bin/env python3
"""Send a normalised Q8.8 sequence to the FPGA via UART and
   read back the Q8.8 prediction, then de-normalise."""

import serial, struct, sys, time, numpy as np

PORT  = "/dev/ttyUSB0"   # adjust for your OS
BAUD  = 115200
NORM  = "mem/norm.txt"
CSV   = "book_store_sales.csv"
MIN   = 12
SPLIT = 0.8
START_BYTE = 0xAA

# ---------- helpers ----------
def float_to_q88_bytes(v):
    i = int(round(v * 256.0))
    i = max(-32768, min(32767, i))
    u = i & 0xFFFF
    return bytes([u >> 8, u & 0xFF])

def q88_bytes_to_float(hi, lo):
    u = (hi << 8) | lo
    if u >= 0x8000:
        u -= 0x10000
    return u / 256.0

# ---------- load norm ----------
with open(NORM) as f:
    mean   = float(f.readline())
    stddev = float(f.readline())

# ---------- load data ----------
with open(CSV, 'r', encoding='utf-8') as f:
    next(f)
    sales = np.array([float(line.split(',')[1]) for line in f], dtype=np.float32)

months = len(sales)
split  = int(months * SPLIT)
test_sales = sales[split:]
test_std   = (test_sales - mean) / stddev

test_months = len(test_sales)

# pick one test sample (last one, longest)
idx = test_months - MIN - 1
seq = np.zeros(test_months - 1, dtype=np.float32)
seq[-(idx + MIN):] = test_std[:idx + MIN]
expected = test_sales[idx + MIN]

seq_len = len(seq)
print(f"Sending sequence of length {seq_len}  (expect ~{expected:.0f})")

# ---------- UART ----------
ser = serial.Serial(PORT, BAUD, timeout=5)
time.sleep(0.1)
ser.reset_input_buffer()

# send header
ser.write(bytes([START_BYTE]))
ser.write(bytes([seq_len >> 8, seq_len & 0xFF]))

# send sequence
for v in seq:
    ser.write(float_to_q88_bytes(v))

ser.flush()
print("Data sent, waiting for result …")

# receive 2 bytes
resp = ser.read(2)
if len(resp) < 2:
    print("ERROR: timeout")
    sys.exit(1)

raw = q88_bytes_to_float(resp[0], resp[1])
prediction = raw * stddev + mean
print(f"FPGA raw Q8.8 = {raw:.4f}")
print(f"Prediction     = {prediction:.1f}")
print(f"Actual         = {expected:.1f}")
ser.close()

#!/usr/bin/env python3
"""Test FPGA RNN inference across multiple test samples."""

import serial, struct, sys, time, numpy as np

PORT       = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB0"
MODE       = sys.argv[2] if len(sys.argv) > 2 else "all"  # "all" or sample index
BAUD       = 115200
NORM       = "mem/norm.txt"
CSV        = "book_store_sales.csv"
MIN        = 12
SPLIT      = 0.8
START_BYTE = 0xAA

# ---- helpers ----
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

def send_and_receive(ser, seq):
    """Send one sequence, receive Q8.8 result."""
    ser.reset_input_buffer()
    seq_len = len(seq)
    
    # header
    ser.write(bytes([START_BYTE]))
    ser.write(bytes([seq_len >> 8, seq_len & 0xFF]))
    
    # data
    for v in seq:
        ser.write(float_to_q88_bytes(v))
    ser.flush()
    
    # receive
    resp = ser.read(2)
    if len(resp) < 2:
        return None
    return q88_bytes_to_float(resp[0], resp[1])

# ---- load normalisation constants ----
with open(NORM) as f:
    mean   = float(f.readline())
    stddev = float(f.readline())
print(f"Norm: mean={mean:.2f}, stddev={stddev:.2f}")

# ---- load data ----
with open(CSV, 'r', encoding='utf-8') as f:
    next(f)
    sales = np.array([float(line.split(',')[1]) for line in f],
                     dtype=np.float32)

months     = len(sales)
split      = int(months * SPLIT)
test_sales = sales[split:]
test_std   = (test_sales - mean) / stddev
test_months = len(test_sales)
n_samples  = test_months - MIN

print(f"Total months: {months}, Train: {split}, Test: {test_months}")
print(f"Available test samples: {n_samples}")
print(f"=" * 65)

# ---- build all test windows ----
def build_sequence(sample_idx):
    """Build zero-padded sequence for test sample index."""
    # target is test_sales[sample_idx + MIN]
    # input is test_std[0 : sample_idx + MIN]
    input_len = sample_idx + MIN
    seq = np.zeros(test_months - 1, dtype=np.float32)
    seq[-(input_len):] = test_std[:input_len]
    expected = test_sales[sample_idx + MIN]
    return seq, expected

# ---- UART ----
ser = serial.Serial(PORT, BAUD, timeout=10)
time.sleep(0.2)

if MODE == "all":
    # test all samples
    errors_abs = []
    errors_norm = []
    
    for i in range(n_samples):
        seq, expected = build_sequence(i)
        raw = send_and_receive(ser, seq)
        
        if raw is None:
            print(f"[{i:3d}] TIMEOUT")
            continue
        
        prediction = raw * stddev + mean
        abs_err = abs(prediction - expected)
        norm_err = abs(raw - test_std[i + MIN])
        errors_abs.append(abs_err)
        errors_norm.append(norm_err)
        
        print(f"[{i:3d}] Pred={prediction:7.1f}  Actual={expected:7.1f}"
              f"  Err={abs_err:6.1f}  NormErr={norm_err:.4f}")
        
        time.sleep(0.05)   # small gap between transactions
    
    print(f"=" * 65)
    print(f"Samples tested     : {len(errors_abs)}")
    print(f"FPGA MAE (real)    : {np.mean(errors_abs):.1f}")
    print(f"FPGA MAE (norm)    : {np.mean(errors_norm):.4f}")
    print(f"Expected test MAE  : 0.1660")
    print(f"FPGA Max error     : {np.max(errors_abs):.1f}")
    print(f"FPGA Min error     : {np.min(errors_abs):.1f}")

else:
    # single sample
    i = int(MODE)
    if i >= n_samples:
        print(f"ERROR: index {i} >= {n_samples} available samples")
        sys.exit(1)
    
    seq, expected = build_sequence(i)
    print(f"Sequence length : {len(seq)}")
    print(f"Sent. Waiting for FPGA response …")
    
    raw = send_and_receive(ser, seq)
    if raw is None:
        print("ERROR: timeout")
        sys.exit(1)
    
    prediction = raw * stddev + mean
    print(f"FPGA Q8.8 raw   : {raw:.4f}")
    print(f"Prediction      : {prediction:.1f}")
    print(f"Actual          : {expected:.1f}")
    print(f"Abs error       : {abs(prediction - expected):.1f}")
    print(f"Norm error      : {abs(raw - test_std[i + MIN]):.4f}")

ser.close()

#!/usr/bin/env python3
"""talk_to_fpga.py — send a 40-char seed, read back predictions."""
import serial, sys, time

PORT  = '/dev/ttyUSB1'
BAUD  = 115200
SEED  = "i busied myself to think of a story, thi"   # exactly 40 chars

assert len(SEED) == 40, f"Seed must be 40 chars, got {len(SEED)}"

ser = serial.Serial(PORT, BAUD, timeout=3)
time.sleep(0.2)
ser.reset_input_buffer()                # flush any leftover bytes

# ── send the 40-character seed ──
print(f"Sending seed: \"{SEED}\"")
for ch in SEED:
    ser.write(ch.encode('ascii'))
    time.sleep(0.005)                   # small delay so FPGA doesn't miss bytes
ser.flush()

# ── read back predicted characters ──
NUM_PREDICTIONS = 500                   # more chars — 32-epoch model is better
output = []
for i in range(NUM_PREDICTIONS):
    ch = ser.read(1)
    if not ch:
        print(f"\n[timeout after {i} chars]")
        break
    output.append(ch.decode('ascii', errors='replace'))
    sys.stdout.write(output[-1])
    sys.stdout.flush()
    # ── feed prediction back so FPGA generates the next char ──
    ser.write(ch)
    time.sleep(0.005)

print(f"\n\nDone. Received {len(output)} chars total.")
ser.close()

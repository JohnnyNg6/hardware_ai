`timescale 1ns / 1ps

// ============================================================
// Phase 1: Single Perceptron - NAND Gate (Inference)
//
//   Encoding : bipolar, x ∈ {-1, +1}
//   Bias     : x0 = 1.0 (hardwired, not an external input)
//   Activation: sign function  z < 0 → -1,  z >= 0 → +1
//   Trained weights: w0 = 0.40,  w1 = -0.40,  w2 = -0.15
//
// Fixed-point format: Q4.4 (8-bit signed)
//   Range      : -8.0 to +7.9375
//   Resolution : 1/16 = 0.0625
//   Encoding   : real_value = raw_integer / 16
// ============================================================
module perceptron_nand (
    input                clk,
    input                rst,
    input                valid_in,
    input  signed  [7:0] x1,         // input 1,  Q4.4  (-1.0 or +1.0)
    input  signed  [7:0] x2,         // input 2,  Q4.4  (-1.0 or +1.0)
    output reg           y_out,      // 1 = perceptron says +1,  0 = says -1
    output reg           valid_out
);

    // ---------------------------------------------------------
    // Bias input x0 = 1.0  (always 1; part of perceptron model)
    // ---------------------------------------------------------
    localparam signed [7:0] X0_BIAS = 8'sd16;   // 1.0 in Q4.4

    // ---------------------------------------------------------
    // Trained weights (from Python notebook final epoch)
    //
    //   Python         Q4.4 raw     Actual Q4.4 value
    //   w0 =  0.40  →   6          →  0.375
    //   w1 = -0.40  →  -6          → -0.375
    //   w2 = -0.15  →  -2          → -0.125
    //
    // Small quantization error, but all 4 outputs still correct.
    // ---------------------------------------------------------
    localparam signed [7:0] W0 =  8'sd6;    // bias weight  ≈  0.40
    localparam signed [7:0] W1 = -8'sd6;    // weight 1     ≈ -0.40
    localparam signed [7:0] W2 = -8'sd2;    // weight 2     ≈ -0.15

    // ---------------------------------------------------------
    // Pipeline Stage 1: Multiply   (Q4.4 × Q4.4 = Q8.8, 16-bit)
    //   mul0 = w0 * x0   (bias term, x0 hardwired to 1.0)
    //   mul1 = w1 * x1
    //   mul2 = w2 * x2
    // ---------------------------------------------------------
    reg signed [15:0] mul0, mul1, mul2;
    reg               pipe1_valid;

    always @(posedge clk) begin
        if (rst) begin
            mul0        <= 16'sd0;
            mul1        <= 16'sd0;
            mul2        <= 16'sd0;
            pipe1_valid <= 1'b0;
        end else begin
            mul0        <= X0_BIAS * W0;       // w0 * 1.0
            mul1        <= x1      * W1;       // w1 * x1
            mul2        <= x2      * W2;       // w2 * x2
            pipe1_valid <= valid_in;
        end
    end

    // ---------------------------------------------------------
    // Pipeline Stage 2: Accumulate  +  Sign Activation
    //
    //   z = mul0 + mul1 + mul2
    //
    //   Python:  if z < 0: return -1   else: return 1
    //   Verilog: y_out = (z >= 0) ? 1 : 0
    //            (we map +1→LED ON,  -1→LED OFF)
    // ---------------------------------------------------------
    wire signed [17:0] z_sum = mul0 + mul1 + mul2;

    always @(posedge clk) begin
        if (rst) begin
            y_out     <= 1'b0;
            valid_out <= 1'b0;
        end else begin
            y_out     <= (z_sum >= 18'sd0) ? 1'b1 : 1'b0;
            valid_out <= pipe1_valid;
        end
    end

endmodule

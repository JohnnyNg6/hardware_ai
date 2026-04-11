`timescale 1ns / 1ps

// ============================================================
// Phase 1: Single Perceptron implementing NAND gate
// Fixed-point: Q4.4 format (4 integer bits, 4 fractional bits = 8-bit)
// This teaches: fixed-point MAC, step activation
// ============================================================
module perceptron_nand (
    input               clk,
    input               rst,
    input               valid_in,   // input valid pulse
    input  signed [7:0] x0,         // input 0, Q4.4
    input  signed [7:0] x1,         // input 1, Q4.4
    output reg          y,          // output (0 or 1)
    output reg          valid_out   // output valid pulse
);

    // --------------------------------------------------------
    // Weights and bias in Q4.4 format
    // Q4.4: value = integer_representation / 16
    // So -2.0 = -32 in Q4.4, 3.0 = 48 in Q4.4
    // --------------------------------------------------------
    localparam signed [7:0] W0 = -8'sd32;   // -2.0 in Q4.4
    localparam signed [7:0] W1 = -8'sd32;   // -2.0 in Q4.4
    localparam signed [7:0] BIAS = 8'sd48;  //  3.0 in Q4.4

    // --------------------------------------------------------
    // Pipeline stage 1: Multiply (Q4.4 * Q4.4 = Q8.8, 16-bit)
    // --------------------------------------------------------
    reg signed [15:0] mul0, mul1;
    reg signed [7:0]  bias_r;
    reg               valid_p1;

    always @(posedge clk) begin
        if (rst) begin
            mul0    <= 0;
            mul1    <= 0;
            bias_r  <= 0;
            valid_p1 <= 0;
        end else begin
            mul0    <= x0 * W0;     // Q8.8 result
            mul1    <= x1 * W1;     // Q8.8 result
            bias_r  <= BIAS;
            valid_p1 <= valid_in;
        end
    end

    // --------------------------------------------------------
    // Pipeline stage 2: Accumulate + Activation
    // Need to align bias to Q8.8: shift left by 4
    // --------------------------------------------------------
    reg signed [17:0] sum;  // extra bits to prevent overflow

    always @(posedge clk) begin
        if (rst) begin
            sum       <= 0;
            y         <= 0;
            valid_out <= 0;
        end else begin
            // bias_r is Q4.4, shift left 4 to make Q8.8
            sum       <= mul0 + mul1 + (bias_r <<< 4);
            // Step activation: if sum > 0, output 1
            y         <= (mul0 + mul1 + (bias_r <<< 4)) > 0 ? 1'b1 : 1'b0;
            valid_out <= valid_p1;
        end
    end

endmodule

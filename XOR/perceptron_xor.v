`timescale 1ns / 1ps
//============================================================================
// XOR Neural Network Inference Core  (combinational, no I/O)
//
// Architecture:
//   Hidden: h0 = tanh(W00 + W01·x1 + W02·x2)
//           h1 = tanh(W10 + W11·x1 + W12·x2)
//   Output: y  = sigmoid(W20 + W21·h0 + W22·h1)
//
// Arithmetic: signed 16-bit Q8.8 fixed-point
//============================================================================

module xor_nn_core (
    input  wire signed [15:0] x1,        // Q8.8 input 1
    input  wire signed [15:0] x2,        // Q8.8 input 2
    output wire signed [15:0] h0_out,    // Q8.8 hidden neuron 0
    output wire signed [15:0] h1_out,    // Q8.8 hidden neuron 1
    output wire signed [15:0] sig_out,   // Q8.8 sigmoid output
    output wire               class_out  // 1 if sig_out >= 0.5
);

    // ================================================================
    //  Pre-trained weights  (Q8.8)
    // ================================================================
    //  Hidden neuron 0  (OR-like gate)
    localparam signed [15:0] W00 =  16'sd256;    //  1.0
    localparam signed [15:0] W01 =  16'sd512;    //  2.0
    localparam signed [15:0] W02 =  16'sd512;    //  2.0
    //  Hidden neuron 1  (NAND-like gate)
    localparam signed [15:0] W10 =  16'sd256;    //  1.0
    localparam signed [15:0] W11 = -16'sd512;    // -2.0
    localparam signed [15:0] W12 = -16'sd512;    // -2.0
    //  Output neuron  (AND combiner)
    localparam signed [15:0] W20 = -16'sd1280;   // -5.0
    localparam signed [15:0] W21 =  16'sd1280;   //  5.0
    localparam signed [15:0] W22 =  16'sd1280;   //  5.0

    // ================================================================
    //  Hidden layer MAC  (Q8.8 × Q8.8 → shift back to Q8.8)
    // ================================================================
    wire signed [31:0] p01 = W01 * x1;
    wire signed [31:0] p02 = W02 * x2;
    wire signed [15:0] mac0 = W00 + p01[23:8] + p02[23:8];

    wire signed [31:0] p11 = W11 * x1;
    wire signed [31:0] p12 = W12 * x2;
    wire signed [15:0] mac1 = W10 + p11[23:8] + p12[23:8];

    // ================================================================
    //  Hidden layer tanh activation
    // ================================================================
    wire signed [15:0] h0, h1;
    tanh_piecewise u_tanh0 (.x(mac0), .y(h0));
    tanh_piecewise u_tanh1 (.x(mac1), .y(h1));

    assign h0_out = h0;
    assign h1_out = h1;

    // ================================================================
    //  Output layer MAC
    // ================================================================
    wire signed [31:0] p21 = W21 * h0;
    wire signed [31:0] p22 = W22 * h1;
    wire signed [15:0] mac2 = W20 + p21[23:8] + p22[23:8];

    // ================================================================
    //  Sigmoid via  sigmoid(x) = tanh(x/2)/2 + 0.5
    // ================================================================
    wire signed [15:0] mac2_half = {mac2[15], mac2[15:1]};  // x/2
    wire signed [15:0] tanh2;
    tanh_piecewise u_tanh2 (.x(mac2_half), .y(tanh2));

    assign sig_out = {tanh2[15], tanh2[15:1]} + 16'sd128;   // /2 + 0.5

    // ================================================================
    //  Classification threshold  (0.5 = 128 in Q8.8)
    // ================================================================
    assign class_out = (sig_out >= 16'sd128);

endmodule

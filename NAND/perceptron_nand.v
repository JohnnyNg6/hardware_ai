`timescale 1ns / 1ps

module perceptron_nand (
    input                clk,
    input                rst,
    input                valid_in,
    input  signed  [7:0] x1,         
    input  signed  [7:0] x2,         
    output reg           y_out,      
    output reg           valid_out
);


    localparam signed [7:0] X0_BIAS = 8'sd16;   // 1.0 in Q4.4
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

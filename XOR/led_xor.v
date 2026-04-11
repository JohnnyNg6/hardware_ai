`timescale 1ns / 1ps
//============================================================================
// XOR Neural Network — Board I/O Wrapper
// Board: 博宸精芯 Kintex7-Base  (XC7K325T-2FFG676)
//
//   KEY1 → x1  (pressed = +1, released = -1)
//   KEY2 → x2  (pressed = +1, released = -1)
//   LED1 → NN XOR result     (ON = output >= 0.5)
//   LED2 → x1 indicator      (ON = +1)
//   LED3 → x2 indicator      (ON = +1)
//   LED4 → expected XOR      (ON = x1 XOR x2)
//   LED5 → hidden h0 sign    (ON = positive)
//   LED6 → hidden h1 sign    (ON = positive)
//   LED7 → high confidence 1 (ON = output >= 0.75)
//   LED8 → high confidence 0 (ON = output < 0.25)
//============================================================================

module xor_nn_top (
    input  wire       clk,        // 50 MHz  (G22)
    input  wire       key1,       // active-low  (D26)
    input  wire       key2,       // active-low  (J26)
    output reg  [7:0] led         // active-low  (A23..E25)
);

    // ================================================================
    //  Button synchroniser + debounce  (~20 ms at 50 MHz)
    // ================================================================
    reg        k1_s1 = 1'b1, k1_s2 = 1'b1;
    reg        k2_s1 = 1'b1, k2_s2 = 1'b1;
    reg        k1_db = 1'b1, k2_db = 1'b1;
    reg [19:0] cnt1  = 20'd0, cnt2 = 20'd0;

    always @(posedge clk) begin
        k1_s1 <= key1;  k1_s2 <= k1_s1;
        k2_s1 <= key2;  k2_s2 <= k2_s1;

        if (k1_s2 != k1_db) begin
            cnt1 <= cnt1 + 1'b1;
            if (&cnt1) k1_db <= k1_s2;
        end else
            cnt1 <= 20'd0;

        if (k2_s2 != k2_db) begin
            cnt2 <= cnt2 + 1'b1;
            if (&cnt2) k2_db <= k2_s2;
        end else
            cnt2 <= 20'd0;
    end

    // ================================================================
    //  Encode inputs:  pressed (low) → +1.0,  released (high) → -1.0
    // ================================================================
    wire signed [15:0] x1_val = k1_db ? -16'sd256 : 16'sd256;
    wire signed [15:0] x2_val = k2_db ? -16'sd256 : 16'sd256;

    // ================================================================
    //  Neural network core instantiation
    // ================================================================
    wire signed [15:0] h0_val, h1_val, sig_val;
    wire               nn_out;

    xor_nn_core u_nn (
        .x1        (x1_val),
        .x2        (x2_val),
        .h0_out    (h0_val),
        .h1_out    (h1_val),
        .sig_out   (sig_val),
        .class_out (nn_out)
    );

    // ================================================================
    //  LED mapping  (active-low: 0 = ON)
    // ================================================================
    wire in1          = ~k1_db;           // 1 when pressed
    wire in2          = ~k2_db;
    wire expected_xor = in1 ^ in2;

    always @(posedge clk) begin
        led[0] <= ~nn_out;                         // LED1: NN result
        led[1] <= ~in1;                            // LED2: x1
        led[2] <= ~in2;                            // LED3: x2
        led[3] <= ~expected_xor;                   // LED4: expected
        led[4] <= h0_val[15];                      // LED5: h0 > 0
        led[5] <= h1_val[15];                      // LED6: h1 > 0
        led[6] <= ~(sig_val >= 16'sd192);          // LED7: >= 0.75
        led[7] <= ~(sig_val <  16'sd64);           // LED8: < 0.25
    end

endmodule

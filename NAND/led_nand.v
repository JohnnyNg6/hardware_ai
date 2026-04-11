`timescale 1ns / 1ps

// ============================================================
// FPGA board wrapper for perceptron NAND gate
//
//   KEY1 / KEY2  =  x1 / x2
//     not pressed  →  -1   (bipolar encoding from notebook)
//     pressed      →  +1
//
//   LED1 = NAND result   (ON = +1,  OFF = -1)
//   LED2 = x1 state      (ON = pressed = +1)
//   LED3 = x2 state      (ON = pressed = +1)
//   LED8 = heartbeat     (proves FPGA is alive)
// ============================================================
module led_perceptron (
    input        sys_clk,       // 50 MHz,  G22
    input  [4:0] key,           // active-low buttons
    output [7:0] led            // active-high LEDs
);

    // ---------------------------------------------------------
    // Heartbeat on LED8  (~2 Hz blink)
    // ---------------------------------------------------------
    reg [24:0] hb_cnt;
    reg        heartbeat;

    always @(posedge sys_clk) begin
        if (hb_cnt == 25'd24_999_999) begin
            hb_cnt    <= 25'd0;
            heartbeat <= ~heartbeat;
        end else begin
            hb_cnt <= hb_cnt + 25'd1;
        end
    end

    // ---------------------------------------------------------
    // Button synchroniser  (2-FF metastability guard)
    // ---------------------------------------------------------
    reg [4:0] key_r0, key_r1;

    always @(posedge sys_clk) begin
        key_r0 <= key;
        key_r1 <= key_r0;
    end

    // Keys are active-low: pressed → 0, released → 1
    wire [4:0] key_pressed = ~key_r1;

    // ---------------------------------------------------------
    // Map button state to bipolar Q4.4 values
    //   pressed   →  +1.0  =  +16   in Q4.4
    //   released  →  -1.0  =  -16   in Q4.4
    //
    // This matches the Python notebook encoding:
    //   x_train uses -1 and +1
    // ---------------------------------------------------------
    wire signed [7:0] x1_val = key_pressed[0] ?  8'sd16 : -8'sd16;  // KEY1
    wire signed [7:0] x2_val = key_pressed[1] ?  8'sd16 : -8'sd16;  // KEY2

    // ---------------------------------------------------------
    // Perceptron instance  (always computing, valid_in tied high)
    // ---------------------------------------------------------
    wire nand_result;
    wire nand_valid;

    perceptron_nand u_nand (
        .clk       (sys_clk),
        .rst       (1'b0),
        .valid_in  (1'b1),
        .x1        (x1_val),
        .x2        (x2_val),
        .y_out     (nand_result),
        .valid_out (nand_valid)
    );

    // ---------------------------------------------------------
    // LED assignment
    // ---------------------------------------------------------
    assign led[0]   = nand_result;       // NAND output
    assign led[1]   = key_pressed[0];    // x1 indicator
    assign led[2]   = key_pressed[1];    // x2 indicator
    assign led[6:3] = 4'b0000;          // unused
    assign led[7]   = heartbeat;         // alive blink

endmodule

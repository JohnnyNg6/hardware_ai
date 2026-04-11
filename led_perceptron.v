`timescale 1ns / 1ps

// Wraps perceptron_nand for FPGA board test
// KEY1/KEY2 = x0/x1 inputs (press = 1, release = 0)
// LED1 = perceptron output (NAND result)
// LED2 = x0 value, LED3 = x1 value
// LED8 = heartbeat (proves FPGA is running)
module led_perceptron (
    input           sys_clk,
    input   [4:0]   key,        // active low
    output  [7:0]   led
);

    // ------------------------------------------------
    // Heartbeat blink on LED8
    // ------------------------------------------------
    reg [24:0] hb_cnt;
    reg        heartbeat;
    always @(posedge sys_clk) begin
        if (hb_cnt == 25'd24_999_999) begin
            hb_cnt    <= 0;
            heartbeat <= ~heartbeat;
        end else
            hb_cnt <= hb_cnt + 1;
    end

    // ------------------------------------------------
    // Key debounce/sync
    // ------------------------------------------------
    reg [4:0] key_r0, key_r1;
    always @(posedge sys_clk) begin
        key_r0 <= key;
        key_r1 <= key_r0;
    end
    // key active low: pressed = 0, so invert
    wire [4:0] key_pressed = ~key_r1;

    // ------------------------------------------------
    // Convert key press to Q4.4 fixed-point
    // ------------------------------------------------
    wire signed [7:0] x0_val = key_pressed[0] ? 8'sd16 : 8'sd0; // KEY1
    wire signed [7:0] x1_val = key_pressed[1] ? 8'sd16 : 8'sd0; // KEY2

    // ------------------------------------------------
    // Perceptron instance
    // ------------------------------------------------
    wire nand_out;
    wire nand_valid;

    // Continuous valid (always computing)
    perceptron_nand u_nand (
        .clk(sys_clk),
        .rst(1'b0),
        .valid_in(1'b1),
        .x0(x0_val),
        .x1(x1_val),
        .y(nand_out),
        .valid_out(nand_valid)
    );

    // ------------------------------------------------
    // LED output
    // ------------------------------------------------
    // LED1 = NAND result
    // LED2 = x0 input (KEY1 state)
    // LED3 = x1 input (KEY2 state)
    // LED4-7 = off
    // LED8 = heartbeat
    assign led[0] = nand_out;          // NAND result
    assign led[1] = key_pressed[0];    // x0
    assign led[2] = key_pressed[1];    // x1
    assign led[6:3] = 4'b0000;
    assign led[7] = heartbeat;         // alive indicator

endmodule

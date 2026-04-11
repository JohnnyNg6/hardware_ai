`timescale 1ns / 1ps
//============================================================================
// MNIST Neural Network — Board I/O Wrapper
// Board: 博宸精芯 Kintex7-Base  (XC7K325T-2FFG676)
//
//   KEY1 → next test image  (cycles 0 → 1 → … → 9 → 0)
//   KEY2 → previous test image
//
//   LED[3:0] → predicted digit    (binary, active-low)
//   LED[7:4] → actual label       (binary, active-low)
//
//   When predicted == actual, top and bottom nibbles match visually.
//   All LEDs ON (0x00) while inference is running.
//============================================================================

module mnist_nn_top (
    input  wire       clk,            // 50 MHz  (G22)
    input  wire       key1,           // active-low  (D26)
    input  wire       key2,           // active-low  (J26)
    output reg  [7:0] led             // active-low  (A23..E25)
);

    // ================================================================
    //  Power-on reset  (~16 cycles)
    // ================================================================
    reg [3:0] por_cnt = 4'd0;
    wire rst_n = &por_cnt;
    always @(posedge clk)
        if (!rst_n) por_cnt <= por_cnt + 4'd1;

    // ================================================================
    //  Button synchroniser + debounce  (~20 ms at 50 MHz)
    // ================================================================
    reg        k1_s1 = 1, k1_s2 = 1, k1_db = 1;
    reg        k2_s1 = 1, k2_s2 = 1, k2_db = 1;
    reg [19:0] cnt1  = 0, cnt2 = 0;

    always @(posedge clk) begin
        k1_s1 <= key1;  k1_s2 <= k1_s1;
        k2_s1 <= key2;  k2_s2 <= k2_s1;

        if (k1_s2 != k1_db) begin
            cnt1 <= cnt1 + 1;
            if (&cnt1) k1_db <= k1_s2;
        end else cnt1 <= 0;

        if (k2_s2 != k2_db) begin
            cnt2 <= cnt2 + 1;
            if (&cnt2) k2_db <= k2_s2;
        end else cnt2 <= 0;
    end

    // Falling-edge detectors  (active-low buttons → press = falling edge)
    reg k1_prev = 1, k2_prev = 1;
    wire k1_press = k1_prev & ~k1_db;
    wire k2_press = k2_prev & ~k2_db;
    always @(posedge clk) begin
        k1_prev <= k1_db;
        k2_prev <= k2_db;
    end

    // ================================================================
    //  Test-image index  (0..9)
    // ================================================================
    reg [3:0] img_idx = 4'd0;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            img_idx <= 4'd0;
        else if (k1_press)
            img_idx <= (img_idx == 4'd9) ? 4'd0 : img_idx + 4'd1;
        else if (k2_press)
            img_idx <= (img_idx == 4'd0) ? 4'd9 : img_idx - 4'd1;
    end

    // ================================================================
    //  Test labels  (ground truth for the 10 stored images)
    // ================================================================
    reg [3:0] test_labels [0:9];
    initial $readmemh("test_labels.mem", test_labels);

    wire [3:0] actual_label = test_labels[img_idx];

    // ================================================================
    //  Inference trigger:  auto-run on image change or after reset
    // ================================================================
    reg  start_pulse = 0;
    reg  first_run   = 1;
    wire nn_busy, nn_done;
    wire [3:0] nn_digit;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            start_pulse <= 1'b0;
            first_run   <= 1'b1;
        end else begin
            start_pulse <= 1'b0;
            if ((k1_press | k2_press | first_run) & ~nn_busy) begin
                start_pulse <= 1'b1;
                first_run   <= 1'b0;
            end
        end
    end

    // ================================================================
    //  Neural network core
    // ================================================================
    mnist_nn_core u_nn (
        .clk     (clk),
        .rst_n   (rst_n),
        .start   (start_pulse),
        .img_sel (img_idx),
        .digit   (nn_digit),
        .done    (nn_done),
        .busy    (nn_busy)
    );

    // ================================================================
    //  Result latch
    // ================================================================
    reg [3:0] predicted = 4'd0;

    always @(posedge clk)
        if (nn_done) predicted <= nn_digit;

    // ================================================================
    //  LED mapping  (active-low: 0 = ON)
    // ================================================================
    always @(posedge clk) begin
        if (nn_busy)
            led <= 8'h00;                              // all ON while busy
        else begin
            led[3:0] <= ~predicted;                    // lower: prediction
            led[7:4] <= ~actual_label;                 // upper: ground truth
        end
    end

endmodule

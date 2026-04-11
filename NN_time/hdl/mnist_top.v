// ============================================================================
// mnist_top.v — Top level for Kintex-7 Base board (小熊猫学堂)
//
// Board:  XC7K325T-2FFG676C
// Clock:  50 MHz single-ended on G22
// UART:   CH340 — RXD B20, TXD C22 (BANK14, LVCMOS33)
// Reset:  KEY1 on D26 (active-low push button)
// LEDs:   LED1–LED8 on A23,A24,D23,C24,C26,D24,D25,E25
// ============================================================================
`timescale 1ns / 1ps

module mnist_top (
    input  wire       clk_50m,        // 50 MHz, G22
    input  wire       rst_n,          // KEY1, D26 (active-low)
    input  wire       uart_rxd,       // CH340 TX → FPGA RX, B20
    output wire       uart_txd,       // FPGA TX → CH340 RX, C22
    output wire [7:0] led             // LED1–LED8
);

    // ================================================================
    // Reset synchroniser (async assert, sync release)
    // ================================================================
    reg [2:0] rst_sr;
    always @(posedge clk_50m or negedge rst_n) begin
        if (!rst_n)
            rst_sr <= 3'b000;
        else
            rst_sr <= {rst_sr[1:0], 1'b1};
    end
    wire rst_n_s = rst_sr[2];

    // ================================================================
    // UART Receiver
    // ================================================================
    wire [7:0] rx_data;
    wire       rx_valid;

    uart_rx #(
        .CLK_FREQ (50_000_000),
        .BAUD     (115_200)
    ) u_uart_rx (
        .clk   (clk_50m),
        .rst_n (rst_n_s),
        .rx    (uart_rxd),
        .data  (rx_data),
        .valid (rx_valid)
    );

    // ================================================================
    // UART Transmitter
    // ================================================================
    wire [7:0] tx_data;
    wire       tx_send;
    wire       tx_busy;

    uart_tx #(
        .CLK_FREQ (50_000_000),
        .BAUD     (115_200)
    ) u_uart_tx (
        .clk   (clk_50m),
        .rst_n (rst_n_s),
        .data  (tx_data),
        .send  (tx_send),
        .tx    (uart_txd),
        .busy  (tx_busy)
    );

    // ================================================================
    // Image Loader (UART ↔ Neural Network bridge)
    // ================================================================
    wire        nn_start;
    wire signed [15:0] nn_pixel;
    wire        nn_pixel_valid;
    wire [3:0]  nn_digit;
    wire        nn_result_valid;
    wire        loader_busy;

    image_loader u_loader (
        .clk             (clk_50m),
        .rst_n           (rst_n_s),
        // UART
        .rx_data         (rx_data),
        .rx_valid        (rx_valid),
        .tx_data         (tx_data),
        .tx_send         (tx_send),
        .tx_busy         (tx_busy),
        // Neural network
        .nn_start        (nn_start),
        .nn_pixel        (nn_pixel),
        .nn_pixel_valid  (nn_pixel_valid),
        .nn_digit        (nn_digit),
        .nn_result_valid (nn_result_valid),
        // Status
        .busy            (loader_busy)
    );

    // ================================================================
    // MNIST Neural Network  (784→25→10)
    // ================================================================
    mnist_nn u_nn (
        .clk          (clk_50m),
        .rst_n        (rst_n_s),
        .img_start    (nn_start),
        .pixel_in     (nn_pixel),
        .pixel_valid  (nn_pixel_valid),
        .digit_out    (nn_digit),
        .result_valid (nn_result_valid)
    );

    // ================================================================
    // LED display
    // ================================================================
    //   led[3:0]  = last predicted digit (binary)
    //   led[4]    = valid result received
    //   led[5]    = loader busy (receiving / inferring)
    //   led[6]    = UART TX active
    //   led[7]    = heartbeat (~1.5 Hz)

    reg [3:0]  last_digit;
    reg        has_result;
    reg [24:0] hb_cnt;

    always @(posedge clk_50m or negedge rst_n_s) begin
        if (!rst_n_s) begin
            last_digit <= 4'd0;
            has_result <= 1'b0;
            hb_cnt     <= 25'd0;
        end else begin
            hb_cnt <= hb_cnt + 25'd1;
            if (nn_result_valid) begin
                last_digit <= nn_digit;
                has_result <= 1'b1;
            end
        end
    end

    assign led[0] = last_digit[0];
    assign led[1] = last_digit[1];
    assign led[2] = last_digit[2];
    assign led[3] = last_digit[3];
    assign led[4] = has_result;
    assign led[5] = loader_busy;
    assign led[6] = tx_busy;
    assign led[7] = hb_cnt[24];       // blink ≈ 1.49 Hz

endmodule

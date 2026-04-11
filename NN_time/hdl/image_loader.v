// ============================================================================
// image_loader.v — Receive image over UART, trigger NN, return result
//
// Protocol (PC → FPGA):
//   Byte 0      : 0xAA  (sync)
//   Bytes 1–1568: 784 pixels × 2 bytes each (big-endian signed Q8.8)
//
// Protocol (FPGA → PC):
//   Byte 0      : predicted digit 0x00–0x09
// ============================================================================
`timescale 1ns / 1ps

module image_loader (
    input  wire        clk,
    input  wire        rst_n,

    // ---- UART RX interface ----
    input  wire [7:0]  rx_data,
    input  wire        rx_valid,

    // ---- UART TX interface ----
    output reg  [7:0]  tx_data,
    output reg         tx_send,
    input  wire        tx_busy,

    // ---- To / from mnist_nn ----
    output reg                 nn_start,
    output reg  signed [15:0]  nn_pixel,
    output reg                 nn_pixel_valid,
    input  wire [3:0]          nn_digit,
    input  wire                nn_result_valid,

    // ---- Status ----
    output wire                busy
);

    // ================================================================
    // Constants
    // ================================================================
    localparam integer NUM_PIXELS = 784;
    localparam [7:0]   SYNC_BYTE  = 8'hAA;

    // ================================================================
    // FSM states
    // ================================================================
    localparam [3:0] S_WAIT_SYNC   = 4'd0,
                     S_RECV_HIGH   = 4'd1,
                     S_RECV_LOW    = 4'd2,
                     S_INFER_START = 4'd3,
                     S_INFER_FEED  = 4'd4,
                     S_INFER_WAIT  = 4'd5,
                     S_SEND_RESULT = 4'd6,
                     S_TX_WAIT     = 4'd7,
                     S_DONE        = 4'd8;

    reg [3:0] state;

    // ================================================================
    // Image pixel buffer  (784 × 16-bit)
    // ================================================================
    reg signed [15:0] img_buf [0:NUM_PIXELS-1];

    // ================================================================
    // Working registers
    // ================================================================
    reg [9:0]  rx_cnt;             // 0 – 783  (pixel write address)
    reg [9:0]  feed_cnt;           // 0 – 783  (pixel read address)
    reg [7:0]  byte_high;          // high byte of current pixel
    reg [3:0]  result_reg;         // latched prediction

    assign busy = (state != S_WAIT_SYNC);

    // ================================================================
    // Main FSM
    // ================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= S_WAIT_SYNC;
            rx_cnt         <= 10'd0;
            feed_cnt       <= 10'd0;
            byte_high      <= 8'd0;
            result_reg     <= 4'd0;
            nn_start       <= 1'b0;
            nn_pixel       <= 16'sd0;
            nn_pixel_valid <= 1'b0;
            tx_data        <= 8'd0;
            tx_send        <= 1'b0;
        end else begin
            // ---- defaults (pulse signals) ----
            nn_start       <= 1'b0;
            nn_pixel_valid <= 1'b0;
            tx_send        <= 1'b0;

            case (state)
                // ------------------------------------------------
                // Wait for 0xAA sync byte
                // ------------------------------------------------
                S_WAIT_SYNC: begin
                    rx_cnt <= 10'd0;
                    if (rx_valid && rx_data == SYNC_BYTE)
                        state <= S_RECV_HIGH;
                end

                // ------------------------------------------------
                // Receive high byte of 16-bit pixel
                // ------------------------------------------------
                S_RECV_HIGH: begin
                    if (rx_valid) begin
                        byte_high <= rx_data;
                        state     <= S_RECV_LOW;
                    end
                end

                // ------------------------------------------------
                // Receive low byte, store complete pixel
                // ------------------------------------------------
                S_RECV_LOW: begin
                    if (rx_valid) begin
                        img_buf[rx_cnt] <= {byte_high, rx_data};
                        if (rx_cnt == NUM_PIXELS[9:0] - 10'd1)
                            state <= S_INFER_START;
                        else begin
                            rx_cnt <= rx_cnt + 10'd1;
                            state  <= S_RECV_HIGH;
                        end
                    end
                end

                // ------------------------------------------------
                // Pulse nn_start to kick off inference
                // ------------------------------------------------
                S_INFER_START: begin
                    nn_start <= 1'b1;
                    feed_cnt <= 10'd0;
                    state    <= S_INFER_FEED;
                end

                // ------------------------------------------------
                // Stream 784 pixels to NN, one per clock
                // ------------------------------------------------
                S_INFER_FEED: begin
                    nn_pixel       <= img_buf[feed_cnt];
                    nn_pixel_valid <= 1'b1;
                    if (feed_cnt == NUM_PIXELS[9:0] - 10'd1)
                        state <= S_INFER_WAIT;
                    else
                        feed_cnt <= feed_cnt + 10'd1;
                end

                // ------------------------------------------------
                // Wait for NN to produce result
                // ------------------------------------------------
                S_INFER_WAIT: begin
                    if (nn_result_valid) begin
                        result_reg <= nn_digit;
                        state      <= S_SEND_RESULT;
                    end
                end

                // ------------------------------------------------
                // Start UART transmission of result byte
                // ------------------------------------------------
                S_SEND_RESULT: begin
                    if (!tx_busy) begin
                        tx_data <= {4'h0, result_reg};
                        tx_send <= 1'b1;
                        state   <= S_TX_WAIT;
                    end
                end

                // ------------------------------------------------
                // Wait for tx_busy to assert (1-clock pipeline)
                // ------------------------------------------------
                S_TX_WAIT: begin
                    if (tx_busy)
                        state <= S_DONE;
                end

                // ------------------------------------------------
                // Wait for transmission to finish
                // ------------------------------------------------
                S_DONE: begin
                    if (!tx_busy)
                        state <= S_WAIT_SYNC;
                end

                default: state <= S_WAIT_SYNC;
            endcase
        end
    end
endmodule

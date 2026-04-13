`timescale 1ns / 1ps
// ============================================================================
// rnn_top.v — Top-level: UART ↔ 128-neuron SimpleRNN + 1-neuron Dense
//
// Architecture:
//   • 128 RNN neurons run fully in parallel (each has its own BRAM)
//   • 1 Dense neuron processes after RNN finishes
//   • Hidden state stored in flat register (2048 bits)
//   • Input sequence stored in distributed RAM
//
// Clock:  50 MHz (G22)
// Reset:  KEY1 active-low (D26)
// UART:   115200 8N1, CH340 on the board
// LEDs:   4-bit status indicator
// ============================================================================
module rnn_top (
    input  wire       clk,          // 50 MHz
    input  wire       rst_n,        // active-low pushbutton
    input  wire       uart_rxd,     // from CH340
    output wire       uart_txd,     // to   CH340
    output reg  [3:0] led           // LED1–LED4 status
);

// ============================== Parameters ===============================
localparam N_HIDDEN       = 128;
localparam RNN_INPUTS     = 129;    // 1 x-input + 128 hidden
localparam DENSE_INPUTS   = 128;
localparam CLK_FREQ       = 50_000_000;
localparam BAUD           = 115_200;
localparam MAX_SEQ        = 1024;

// ============================ UART RX ====================================
wire [7:0] rx_data;
wire       rx_valid;

uart_rx #(.CLK_FREQ(CLK_FREQ), .BAUD(BAUD)) u_rx (
    .clk(clk), .rst_n(rst_n),
    .rx(uart_rxd), .data(rx_data), .valid(rx_valid)
);

// ============================ UART TX ====================================
reg  [7:0] tx_data;
reg        tx_send;
wire       tx_busy;
wire       tx_ready = ~tx_busy;

uart_tx #(.CLK_FREQ(CLK_FREQ), .BAUD(BAUD)) u_tx (
    .clk(clk), .rst_n(rst_n),
    .data(tx_data), .send(tx_send),
    .tx(uart_txd), .busy(tx_busy)
);

// ============================ Input RAM ==================================
(* ram_style = "distributed" *)
reg signed [15:0] input_ram [0:MAX_SEQ-1];

reg [15:0] seq_len;
reg [15:0] recv_cnt;
reg [7:0]  byte_hi;

// ============================ Hidden State ===============================
// Flat register: hidden_flat[ i*16 +: 16 ] = hidden[i],  i = 0..127
reg [N_HIDDEN*16-1:0] hidden_flat;

// ============================ RNN Layer Wires ============================
wire [N_HIDDEN*16-1:0] rnn_dout_flat;
wire [N_HIDDEN-1  :0]  rnn_done_vec;
wire                    rnn_all_done = &rnn_done_vec;

reg                     rnn_start;
reg  signed [15:0]      rnn_din;
reg                     rnn_din_valid;

// ============================ Dense Layer Wires ==========================
wire signed [15:0]      dense_dout;
wire                    dense_done;
reg                     dense_start;
reg  signed [15:0]      dense_din;
reg                     dense_din_valid;

// ======================== 128 Parallel RNN Neurons =======================
// Auto-generated include: one neuron per hidden unit, all receiving the
// same broadcast (rnn_start, rnn_din, rnn_din_valid).
`include "mem/rnn_inst.vh"

// ============================ Dense Neuron ===============================
neuron #(
    .NUM_INPUTS  (DENSE_INPUTS),
    .WEIGHT_FILE ("mem/dense_w.mem")
) u_dense (
    .clk       (clk),
    .rst_n     (rst_n),
    .start     (dense_start),
    .din       (dense_din),
    .din_valid (dense_din_valid),
    .dout      (dense_dout),
    .done      (dense_done),
    .relu_en   (1'b0)             // linear output
);

// ============================ Result Register ============================
reg signed [15:0] result;

// ============================ FSM ========================================
//
// Timing overview per RNN timestep:
//   S_RNN_START    (1 cycle)  → rnn_start pulse
//   S_RNN_FEED     (129 cyc)  → stream 129 din_valid pulses
//   S_RNN_WAIT     (≈4 cyc)   → neurons finish (pipeline flush)
//   Total ≈ 134 cycles / timestep
//
// Dense layer: ~131 cycles.
// UART dominates overall latency.
// -----------------------------------------------------------------
localparam [3:0]
    S_IDLE          = 4'd0,
    S_RECV_LEN_HI   = 4'd1,
    S_RECV_LEN_LO   = 4'd2,
    S_RECV_DATA_HI  = 4'd3,
    S_RECV_DATA_LO  = 4'd4,
    S_RNN_START     = 4'd5,
    S_RNN_FEED      = 4'd6,
    S_RNN_WAIT      = 4'd7,
    S_DENSE_START   = 4'd8,
    S_DENSE_FEED    = 4'd9,
    S_DENSE_WAIT    = 4'd10,
    S_SEND_HI       = 4'd11,
    S_WAIT_HI       = 4'd12,
    S_SEND_LO       = 4'd13,
    S_WAIT_LO       = 4'd14,
    S_DONE          = 4'd15;

reg [3:0]  state;
reg [15:0] timestep;
reg [7:0]  feed_cnt;

// --- Hidden-state read mux for Dense layer (feed_cnt = 0..127) ---
wire signed [15:0] h_rd = $signed(hidden_flat[feed_cnt[6:0] * 16 +: 16]);

// --- Hidden-state read mux for RNN layer  (feed_cnt = 1..128) ---
//     feed_cnt=0 → x-input (selected by controller, h_rd_rnn is don't-care)
wire [6:0]         rnn_h_idx = feed_cnt[6:0] - 7'd1;
wire signed [15:0] h_rd_rnn  = $signed(hidden_flat[rnn_h_idx * 16 +: 16]);

// ============================ Main FSM ===================================
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state           <= S_IDLE;
        seq_len         <= 16'd0;
        recv_cnt        <= 16'd0;
        timestep        <= 16'd0;
        feed_cnt        <= 8'd0;
        hidden_flat     <= {(N_HIDDEN*16){1'b0}};
        rnn_start       <= 1'b0;
        rnn_din         <= 16'sd0;
        rnn_din_valid   <= 1'b0;
        dense_start     <= 1'b0;
        dense_din       <= 16'sd0;
        dense_din_valid <= 1'b0;
        tx_send         <= 1'b0;
        tx_data         <= 8'd0;
        result          <= 16'sd0;
        byte_hi         <= 8'd0;
        led             <= 4'b0001;
    end else begin
        // ---- one-cycle pulse defaults ----
        rnn_start       <= 1'b0;
        rnn_din_valid   <= 1'b0;
        dense_start     <= 1'b0;
        dense_din_valid <= 1'b0;
        tx_send         <= 1'b0;

        case (state)

        // ======== RECEIVE PHASE ========
        S_IDLE: begin
            led         <= 4'b0001;
            hidden_flat <= {(N_HIDDEN*16){1'b0}};
            timestep    <= 16'd0;
            if (rx_valid && rx_data == 8'hAA)
                state <= S_RECV_LEN_HI;
        end

        S_RECV_LEN_HI:
            if (rx_valid) begin
                byte_hi <= rx_data;
                state   <= S_RECV_LEN_LO;
            end

        S_RECV_LEN_LO:
            if (rx_valid) begin
                seq_len  <= {byte_hi, rx_data};
                recv_cnt <= 16'd0;
                state    <= S_RECV_DATA_HI;
            end

        S_RECV_DATA_HI:
            if (rx_valid) begin
                byte_hi <= rx_data;
                state   <= S_RECV_DATA_LO;
            end

        S_RECV_DATA_LO:
            if (rx_valid) begin
                input_ram[recv_cnt] <= $signed({byte_hi, rx_data});
                if (recv_cnt + 16'd1 == seq_len) begin
                    recv_cnt <= 16'd0;
                    state    <= S_RNN_START;
                end else begin
                    recv_cnt <= recv_cnt + 16'd1;
                    state    <= S_RECV_DATA_HI;
                end
            end

        // ======== RNN PROCESSING (128 neurons in parallel) ========
        S_RNN_START: begin
            led       <= 4'b0010;
            rnn_start <= 1'b1;          // broadcast start to all neurons
            feed_cnt  <= 8'd0;
            state     <= S_RNN_FEED;
        end

        S_RNN_FEED: begin
            rnn_din_valid <= 1'b1;
            rnn_din       <= (feed_cnt == 8'd0) ? input_ram[timestep]
                                                : h_rd_rnn;
            if (feed_cnt == RNN_INPUTS[7:0] - 8'd1)
                state <= S_RNN_WAIT;
            feed_cnt <= feed_cnt + 8'd1;
        end

        S_RNN_WAIT:
            if (rnn_all_done) begin
                hidden_flat <= rnn_dout_flat;       // latch all 128 outputs
                timestep    <= timestep + 16'd1;
                if (timestep + 16'd1 < seq_len)
                    state <= S_RNN_START;           // next timestep
                else
                    state <= S_DENSE_START;          // all timesteps done
            end

        // ======== DENSE PROCESSING ========
        S_DENSE_START: begin
            led         <= 4'b0100;
            dense_start <= 1'b1;
            feed_cnt    <= 8'd0;
            state       <= S_DENSE_FEED;
        end

        S_DENSE_FEED: begin
            dense_din_valid <= 1'b1;
            dense_din       <= h_rd;                // hidden[feed_cnt]
            if (feed_cnt == DENSE_INPUTS[7:0] - 8'd1)
                state <= S_DENSE_WAIT;
            feed_cnt <= feed_cnt + 8'd1;
        end

        S_DENSE_WAIT:
            if (dense_done) begin
                result <= dense_dout;
                state  <= S_SEND_HI;
            end

        // ======== SEND RESULT VIA UART (2 bytes, big-endian) ========
        S_SEND_HI: begin
            led <= 4'b1000;
            if (tx_ready) begin
                tx_data <= result[15:8];
                tx_send <= 1'b1;
                state   <= S_WAIT_HI;
            end
        end

        S_WAIT_HI:                          // 1-cycle for tx_busy to register
            state <= S_SEND_LO;

        S_SEND_LO:
            if (tx_ready) begin
                tx_data <= result[7:0];
                tx_send <= 1'b1;
                state   <= S_WAIT_LO;
            end

        S_WAIT_LO:                          // 1-cycle for tx_busy to register
            state <= S_DONE;

        S_DONE:
            if (tx_ready)
                state <= S_IDLE;

        default: state <= S_IDLE;
        endcase
    end
end

endmodule

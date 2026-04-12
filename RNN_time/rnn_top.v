`timescale 1ns / 1ps
module rnn_top (
    input  wire       clk,       // 50 MHz  (G22)
    input  wire       rst_n,     // KEY1 active-low (D26)
    input  wire       uart_rxd,  // from CH340 (B20)
    output wire       uart_txd,  // to   CH340 (C22)
    output reg  [3:0] led        // LED1-LED4
);

// ===== parameters =====
localparam N_HIDDEN       = 128;
localparam RNN_INPUTS     = 129;   // 1 input + 128 hidden
localparam DENSE_INPUTS   = 128;
localparam CLK_FREQ       = 50_000_000;  // CHANGED: for your UART
localparam BAUD           = 115_200;     // CHANGED: for your UART
localparam MAX_SEQ        = 1024;

// ===== UART RX =====
wire [7:0] rx_data;
wire       rx_valid;
uart_rx #(                               // CHANGED: your parameter style
    .CLK_FREQ(CLK_FREQ),
    .BAUD(BAUD)
) u_rx (
    .clk(clk),
    .rst_n(rst_n),
    .rx(uart_rxd),
    .data(rx_data),
    .valid(rx_valid)
);

// ===== UART TX =====
reg  [7:0] tx_data;
reg        tx_send;                      // CHANGED: renamed from tx_start
wire       tx_busy;                      // CHANGED: inverted sense vs ready
wire       tx_ready = ~tx_busy;          // CHANGED: derive ready from busy

uart_tx #(                               // CHANGED: your parameter style
    .CLK_FREQ(CLK_FREQ),
    .BAUD(BAUD)
) u_tx (
    .clk(clk),
    .rst_n(rst_n),
    .data(tx_data),
    .send(tx_send),                      // CHANGED: port name
    .tx(uart_txd),
    .busy(tx_busy)                       // CHANGED: port name
);

// ===== input RAM =====
reg signed [15:0] input_ram [0:MAX_SEQ-1];
reg [15:0] seq_len;
reg [15:0] recv_cnt;
reg [7:0]  byte_hi;

// ===== hidden state (flat 128×16 = 2048 bits) =====
reg  [N_HIDDEN*16-1:0] hidden_flat;

// ===== RNN layer wires =====
wire [N_HIDDEN*16-1:0] rnn_dout_flat;
wire [N_HIDDEN-1  :0]  rnn_done_vec;
wire                    rnn_all_done = &rnn_done_vec;

reg                     rnn_start;
reg  signed [15:0]      rnn_din;
reg                     rnn_din_valid;

// ===== Dense layer wires =====
wire signed [15:0]      dense_dout;
wire                    dense_done;
reg                     dense_start;
reg  signed [15:0]      dense_din;
reg                     dense_din_valid;

// ===== 128 parallel RNN neurons (auto-generated) =====
`include "mem/rnn_inst.vh"

// ===== Dense neuron =====
neuron #(
    .NUM_INPUTS(DENSE_INPUTS),
    .WEIGHT_FILE("mem/dense_w.mem")
) u_dense (
    .clk(clk),
    .rst_n(rst_n),
    .start(dense_start),
    .din(dense_din),
    .din_valid(dense_din_valid),
    .dout(dense_dout),
    .done(dense_done),
    .relu_en(1'b0)
);

// ===== result register =====
reg signed [15:0] result;

// ===== FSM =====
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
    S_SEND_LO       = 4'd12,
    S_DONE          = 4'd13;

reg [3:0]  state;
reg [15:0] timestep;
reg [7:0]  feed_cnt;

// hidden-state read muxes
wire signed [15:0] h_rd = $signed(hidden_flat[feed_cnt*16 +: 16]);
wire [7:0] h_idx = feed_cnt - 8'd1;
wire signed [15:0] h_rd_rnn = $signed(hidden_flat[h_idx*16 +: 16]);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        state        <= S_IDLE;
        seq_len      <= 0;
        recv_cnt     <= 0;
        timestep     <= 0;
        feed_cnt     <= 0;
        hidden_flat  <= {(N_HIDDEN*16){1'b0}};
        rnn_start    <= 0;
        rnn_din      <= 0;
        rnn_din_valid<= 0;
        dense_start  <= 0;
        dense_din    <= 0;
        dense_din_valid <= 0;
        tx_send      <= 0;            // CHANGED: was tx_start
        tx_data      <= 0;
        result       <= 0;
        byte_hi      <= 0;
        led          <= 4'b0001;
    end else begin
        // defaults
        rnn_start       <= 1'b0;
        rnn_din_valid   <= 1'b0;
        dense_start     <= 1'b0;
        dense_din_valid <= 1'b0;
        tx_send         <= 1'b0;      // CHANGED: was tx_start

        case (state)
        // -------- receive phase --------
        S_IDLE: begin
            led <= 4'b0001;
            hidden_flat <= {(N_HIDDEN*16){1'b0}};
            timestep    <= 0;
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
                recv_cnt <= 0;
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
                if (recv_cnt + 1 == seq_len) begin
                    recv_cnt <= 0;
                    state    <= S_RNN_START;
                end else begin
                    recv_cnt <= recv_cnt + 1;
                    state    <= S_RECV_DATA_HI;
                end
            end

        // -------- RNN processing (128 neurons in parallel) --------
        S_RNN_START: begin
            led       <= 4'b0010;
            rnn_start <= 1'b1;
            feed_cnt  <= 0;
            state     <= S_RNN_FEED;
        end

        S_RNN_FEED: begin
            rnn_din_valid <= 1'b1;
            if (feed_cnt == 0)
                rnn_din <= input_ram[timestep];
            else
                rnn_din <= h_rd_rnn;

            if (feed_cnt == RNN_INPUTS - 1)
                state <= S_RNN_WAIT;
            feed_cnt <= feed_cnt + 1;
        end

        S_RNN_WAIT:
            if (rnn_all_done) begin
                hidden_flat <= rnn_dout_flat;
                timestep    <= timestep + 1;
                if (timestep + 1 < seq_len)
                    state <= S_RNN_START;
                else
                    state <= S_DENSE_START;
            end

        // -------- Dense processing --------
        S_DENSE_START: begin
            led         <= 4'b0100;
            dense_start <= 1'b1;
            feed_cnt    <= 0;
            state       <= S_DENSE_FEED;
        end

        S_DENSE_FEED: begin
            dense_din_valid <= 1'b1;
            dense_din       <= h_rd;

            if (feed_cnt == DENSE_INPUTS - 1)
                state <= S_DENSE_WAIT;
            feed_cnt <= feed_cnt + 1;
        end

        S_DENSE_WAIT:
            if (dense_done) begin
                result <= dense_dout;
                state  <= S_SEND_HI;
            end

        // -------- send result via UART --------
        S_SEND_HI: begin
            led <= 4'b1000;
            if (tx_ready) begin                // tx_ready = ~tx_busy
                tx_data <= result[15:8];
                tx_send <= 1'b1;               // CHANGED: was tx_start
                state   <= S_SEND_LO;
            end
        end

        S_SEND_LO:
            if (tx_ready) begin
                tx_data <= result[7:0];
                tx_send <= 1'b1;               // CHANGED: was tx_start
                state   <= S_DONE;
            end

        S_DONE:
            if (tx_ready)
                state <= S_IDLE;

        default: state <= S_IDLE;
        endcase
    end
end

endmodule

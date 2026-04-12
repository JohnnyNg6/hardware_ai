// ============================================================
// lstm_top.v — Kintex-7 LSTM autocompletion engine
//
//   UART protocol:
//     Host sends ASCII seed (≤40 chars) terminated by '\n'.
//     FPGA replies with 11 predicted characters.
//
//   Resource estimate (XC7K325T):
//     DSPs : 128 + 128 + 63 = 319  (of 840)
//     LUTs : ~70 K                  (of 203 800)
//     FFs  : ~60 K                  (of 407 600)
// ============================================================
`timescale 1ns/1ps
module lstm_top #(
    parameter integer ENC_WIDTH   = 63,
    parameter integer HIDDEN_SIZE = 128,
    parameter integer MAX_SEED    = 40,
    parameter integer NUM_GEN     = 11
)(
    input  wire clk_50m,       // G22  50 MHz
    input  wire rst_key_n,     // D26  KEY1 active low
    input  wire uart_rxd,      // B20
    output wire uart_txd,      // C22
    output wire [7:0] led      // A23..E25
);

    // ── clock / reset ──
    wire clk   = clk_50m;
    wire rst_n = rst_key_n;

    // ── UART ──
    wire [7:0] rx_data; wire rx_valid;
    wire       tx_busy;
    reg  [7:0] tx_data; reg tx_start;

    uart_rx #(.CLK_FREQ(50_000_000)) u_rx (
        .clk(clk),.rst_n(rst_n),.rx(uart_rxd),
        .data(rx_data),.valid(rx_valid));

    uart_tx #(.CLK_FREQ(50_000_000)) u_tx (
        .clk(clk),.rst_n(rst_n),.data(tx_data),
        .send(tx_start),.tx(uart_txd),.busy(tx_busy));

    // ── character map ROMs ──
    (* rom_style = "distributed" *)
    reg [15:0] c2i_rom [0:127];          // ASCII → index
    initial $readmemh("weights/char_to_idx.mem", c2i_rom);

    (* rom_style = "distributed" *)
    reg [15:0] i2c_rom [0:ENC_WIDTH-1];  // index → ASCII
    initial $readmemh("weights/idx_to_char.mem", i2c_rom);

    // ── seed buffer ──
    reg [5:0] seed_buf [0:MAX_SEED-1];
    reg [5:0] seed_len;

    // ── one-hot generator ──
    reg  [5:0]         cur_char_idx;
    reg  signed [15:0] x_vec [0:ENC_WIDTH-1];

    integer jj;
    always @(*) begin
        for (jj = 0; jj < ENC_WIDTH; jj = jj + 1)
            x_vec[jj] = (jj == cur_char_idx) ? 16'sh0100 : 16'sh0000;
    end

    // ── LSTM layer 1 (input=ENC_WIDTH, hidden=HIDDEN_SIZE) ──
    wire signed [15:0] h1 [0:HIDDEN_SIZE-1];
    reg  l1_start, l1_clear;
    wire l1_done;

    lstm_layer #(
        .INPUT_SIZE (ENC_WIDTH),
        .HIDDEN_SIZE(HIDDEN_SIZE),
        .LAYER_ID   (1)
    ) u_l1 (
        .clk(clk), .rst_n(rst_n),
        .clear_state(l1_clear),
        .start_step(l1_start),
        .x_in(x_vec),
        .h_out(h1),
        .step_done(l1_done)
    );

    // ── LSTM layer 2 (input=HIDDEN_SIZE, hidden=HIDDEN_SIZE) ──
    wire signed [15:0] h2 [0:HIDDEN_SIZE-1];
    reg  l2_start, l2_clear;
    wire l2_done;

    lstm_layer #(
        .INPUT_SIZE (HIDDEN_SIZE),
        .HIDDEN_SIZE(HIDDEN_SIZE),
        .LAYER_ID   (2)
    ) u_l2 (
        .clk(clk), .rst_n(rst_n),
        .clear_state(l2_clear),
        .start_step(l2_start),
        .x_in(h1),
        .h_out(h2),
        .step_done(l2_done)
    );

    // ── Dense layer ──
    wire signed [15:0] dense_scores [0:ENC_WIDTH-1];
    reg  dense_start;
    wire dense_done;
    reg  dense_valid;
    reg  [$clog2(HIDDEN_SIZE)-1:0] dense_addr;
    wire signed [15:0] dense_din = h2[dense_addr];

    dense_layer #(
        .INPUT_SIZE (HIDDEN_SIZE),
        .OUTPUT_SIZE(ENC_WIDTH)
    ) u_dense (
        .clk(clk), .rst_n(rst_n),
        .start(dense_start),
        .din(dense_din),
        .din_valid(dense_valid),
        .scores(dense_scores),
        .done(dense_done)
    );

    // ── argmax ──
    wire [$clog2(ENC_WIDTH)-1:0] pred_idx;
    argmax #(.N(ENC_WIDTH)) u_am (.vals(dense_scores), .idx(pred_idx));

    // ── output buffer ──
    reg [7:0] out_buf [0:NUM_GEN-1];
    reg [3:0] out_cnt, out_tx_idx;

    // ── main controller FSM ──
    localparam [3:0]
        M_IDLE      = 0,
        M_RECV      = 1,
        M_CLEAR     = 2,
        M_SEED_L1   = 3,
        M_SEED_L2   = 4,
        M_GEN_DENSE = 5,
        M_GEN_DRUN  = 6,
        M_GEN_SAVE  = 7,
        M_GEN_L1    = 8,
        M_GEN_L2    = 9,
        M_TX        = 10,
        M_TX_WAIT   = 11;

    reg [3:0] mstate;
    reg [5:0] seq_idx;         // index into seed / generation

    assign led = {4'b0, mstate};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mstate     <= M_IDLE;
            seed_len   <= 0; seq_idx <= 0; out_cnt <= 0; out_tx_idx <= 0;
            l1_start   <= 0; l2_start <= 0; l1_clear <= 0; l2_clear <= 0;
            dense_start<= 0; dense_valid <= 0; dense_addr <= 0;
            tx_start   <= 0; tx_data <= 0;
            cur_char_idx <= 0;
        end else begin
            // pulse defaults
            l1_start  <= 0; l2_start  <= 0;
            l1_clear  <= 0; l2_clear  <= 0;
            dense_start <= 0; tx_start <= 0;

            case (mstate)
            // ── wait for seed input ──
            M_IDLE: begin
                seed_len <= 0;
                if (rx_valid && rx_data != 8'h0A)
                    mstate <= M_RECV;
            end

            // ── receive seed characters until newline ──
            M_RECV: begin
                if (rx_valid) begin
                    if (rx_data == 8'h0A || seed_len == MAX_SEED) begin
                        mstate <= M_CLEAR;
                    end else begin
                        seed_buf[seed_len] <= c2i_rom[rx_data[6:0]][5:0];
                        seed_len <= seed_len + 1;
                    end
                end
            end

            // ── clear LSTM states ──
            M_CLEAR: begin
                l1_clear <= 1; l2_clear <= 1;
                seq_idx  <= 0; out_cnt  <= 0;
                mstate   <= M_SEED_L1;
            end

            // ── process seed: layer 1 time step ──
            M_SEED_L1: begin
                if (seq_idx < seed_len) begin
                    cur_char_idx <= seed_buf[seq_idx];
                    l1_start     <= 1;
                    mstate       <= M_SEED_L2;       // wait for L1 done
                end else begin
                    mstate <= M_GEN_DENSE;
                end
            end

            // ── process seed: layer 2 time step (waits for L1) ──
            M_SEED_L2: begin
                if (l1_done) begin
                    l2_start <= 1;
                    mstate   <= M_SEED_L2;           // reuse state, wait l2
                end
                if (l2_done) begin
                    seq_idx <= seq_idx + 1;
                    mstate  <= M_SEED_L1;
                end
            end

            // ── generation: start dense ──
            M_GEN_DENSE: begin
                if (out_cnt < NUM_GEN) begin
                    dense_start <= 1;
                    dense_addr  <= 0;
                    mstate      <= M_GEN_DRUN;
                end else begin
                    out_tx_idx <= 0;
                    mstate     <= M_TX;
                end
            end

            // ── generation: run dense (stream h2 elements) ──
            M_GEN_DRUN: begin
                dense_valid <= 1;
                if (dense_done) begin
                    dense_valid <= 0;
                    mstate      <= M_GEN_SAVE;
                end else begin
                    dense_addr <= dense_addr + 1;
                end
            end

            // ── save predicted char & feed to LSTMs ──
            M_GEN_SAVE: begin
                out_buf[out_cnt] <= i2c_rom[pred_idx][7:0];
                cur_char_idx     <= pred_idx;
                l1_start         <= 1;
                out_cnt          <= out_cnt + 1;
                mstate           <= M_GEN_L1;
            end

            // ── generation: layer 1 step ──
            M_GEN_L1: begin
                if (l1_done) begin
                    l2_start <= 1;
                    mstate   <= M_GEN_L2;
                end
            end

            // ── generation: layer 2 step ──
            M_GEN_L2: begin
                if (l2_done)
                    mstate <= M_GEN_DENSE;
            end

            // ── transmit results ──
            M_TX: begin
                if (out_tx_idx < out_cnt && !tx_busy) begin
                    tx_data  <= out_buf[out_tx_idx];
                    tx_start <= 1;
                    mstate   <= M_TX_WAIT;
                end else if (out_tx_idx >= out_cnt) begin
                    // send newline
                    if (!tx_busy) begin
                        tx_data  <= 8'h0A;
                        tx_start <= 1;
                        mstate   <= M_IDLE;
                    end
                end
            end

            M_TX_WAIT: begin
                if (!tx_busy) begin
                    out_tx_idx <= out_tx_idx + 1;
                    mstate     <= M_TX;
                end
            end

            default: mstate <= M_IDLE;
            endcase
        end
    end
endmodule

`timescale 1ns/1ps
module lstm_top #(
    parameter integer ENC_WIDTH   = 63,
    parameter integer HIDDEN_SIZE = 128,
    parameter integer MAX_SEED    = 40,
    parameter integer NUM_GEN     = 11
)(
    input  wire       clk_50m,
    input  wire       rst_key_n,
    input  wire       uart_rxd,
    output wire       uart_txd,
    output wire [7:0] led
);
    wire clk   = clk_50m;
    wire rst_n = rst_key_n;

    // ── UART ──
    wire [7:0] rx_data;
    wire       rx_valid;
    wire       tx_busy;
    reg  [7:0] tx_data;
    reg        tx_start;

    uart_rx #(.CLK_FREQ(50_000_000)) u_rx (
        .clk(clk), .rst_n(rst_n), .rx(uart_rxd),
        .data(rx_data), .valid(rx_valid));

    uart_tx #(.CLK_FREQ(50_000_000)) u_tx (
        .clk(clk), .rst_n(rst_n), .data(tx_data),
        .send(tx_start), .tx(uart_txd), .busy(tx_busy));

    // ── character map ROMs ──
    (* rom_style = "distributed" *)
    reg [15:0] c2i_rom [0:127];
    initial $readmemh("weights/char_to_idx.mem", c2i_rom);

    (* rom_style = "distributed" *)
    reg [15:0] i2c_rom [0:ENC_WIDTH-1];
    initial $readmemh("weights/idx_to_char.mem", i2c_rom);

    // ── seed buffer ──
    reg [5:0] seed_buf [0:MAX_SEED-1];
    reg [5:0] seed_len;

    // ── one-hot encoder (flat) ──
    reg [5:0] cur_char_idx;

    wire [ENC_WIDTH*16-1:0] x_vec_flat;
    genvar gx;
    generate
        for (gx = 0; gx < ENC_WIDTH; gx = gx + 1) begin : gen_xvec
            assign x_vec_flat[gx*16 +: 16] =
                (gx[5:0] == cur_char_idx) ? 16'sh0100 : 16'sh0000;
        end
    endgenerate

    // ── LSTM layer 1 ──
    wire [HIDDEN_SIZE*16-1:0] h1_flat;
    reg  l1_start, l1_clear;
    wire l1_done;

    lstm_layer #(
        .INPUT_SIZE (ENC_WIDTH),
        .HIDDEN_SIZE(HIDDEN_SIZE),
        .LAYER_ID   (1)
    ) u_l1 (
        .clk        (clk),
        .rst_n      (rst_n),
        .clear_state(l1_clear),
        .start_step (l1_start),
        .x_in_flat  (x_vec_flat),
        .h_out_flat (h1_flat),
        .step_done  (l1_done)
    );

    // ── LSTM layer 2 ──
    wire [HIDDEN_SIZE*16-1:0] h2_flat;
    reg  l2_start, l2_clear;
    wire l2_done;

    lstm_layer #(
        .INPUT_SIZE (HIDDEN_SIZE),
        .HIDDEN_SIZE(HIDDEN_SIZE),
        .LAYER_ID   (2)
    ) u_l2 (
        .clk        (clk),
        .rst_n      (rst_n),
        .clear_state(l2_clear),
        .start_step (l2_start),
        .x_in_flat  (h1_flat),
        .h_out_flat (h2_flat),
        .step_done  (l2_done)
    );

    // ── Dense layer ──
    wire [ENC_WIDTH*16-1:0] dense_scores_flat;
    reg  dense_start;
    wire dense_done;

    dense_layer #(
        .INPUT_SIZE (HIDDEN_SIZE),
        .OUTPUT_SIZE(ENC_WIDTH)
    ) u_dense (
        .clk        (clk),
        .rst_n      (rst_n),
        .start      (dense_start),
        .x_in_flat  (h2_flat),
        .scores_flat(dense_scores_flat),
        .done       (dense_done)
    );

    // ── argmax (now sequential / registered) ──
    wire [$clog2(ENC_WIDTH)-1:0] pred_idx;
    wire                         argmax_done;
    reg                          argmax_start;

    argmax #(.N(ENC_WIDTH)) u_am (
        .clk       (clk),
        .rst_n     (rst_n),
        .vals_flat (dense_scores_flat),
        .start     (argmax_start),
        .idx       (pred_idx),
        .done      (argmax_done)
    );

    // ── output buffer ──
    reg [7:0] out_buf [0:NUM_GEN-1];
    reg [3:0] out_cnt, out_tx_idx;

    // ── main controller FSM ──
    localparam [3:0]
        M_IDLE      = 4'd0,
        M_RECV      = 4'd1,
        M_CLEAR     = 4'd2,
        M_SEED_L1   = 4'd3,
        M_SEED_W1   = 4'd4,
        M_SEED_W2   = 4'd5,
        M_GEN_DENSE = 4'd6,
        M_GEN_DWAIT = 4'd7,
        M_GEN_AWAIT = 4'd8,   // ← NEW: wait for argmax
        M_GEN_SAVE  = 4'd9,
        M_GEN_W1    = 4'd10,
        M_GEN_W2    = 4'd11,
        M_TX        = 4'd12,
        M_TX_WAIT   = 4'd13;

    reg [3:0] mstate;
    reg [5:0] seq_idx;

    assign led = {4'b0, mstate};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mstate       <= M_IDLE;
            seed_len     <= 6'd0;
            seq_idx      <= 6'd0;
            out_cnt      <= 4'd0;
            out_tx_idx   <= 4'd0;
            l1_start     <= 1'b0;
            l2_start     <= 1'b0;
            l1_clear     <= 1'b0;
            l2_clear     <= 1'b0;
            dense_start  <= 1'b0;
            argmax_start <= 1'b0;
            tx_start     <= 1'b0;
            tx_data      <= 8'd0;
            cur_char_idx <= 6'd0;
        end else begin
            l1_start     <= 1'b0;
            l2_start     <= 1'b0;
            l1_clear     <= 1'b0;
            l2_clear     <= 1'b0;
            dense_start  <= 1'b0;
            argmax_start <= 1'b0;
            tx_start     <= 1'b0;

            case (mstate)

            M_IDLE: begin
                seed_len <= 6'd0;
                if (rx_valid && rx_data != 8'h0A) begin
                    seed_buf[0] <= c2i_rom[rx_data[6:0]][5:0];
                    seed_len    <= 6'd1;
                    mstate      <= M_RECV;
                end
            end

            M_RECV: begin
                if (rx_valid) begin
                    if (rx_data == 8'h0A || seed_len == MAX_SEED[5:0])
                        mstate <= M_CLEAR;
                    else begin
                        seed_buf[seed_len] <= c2i_rom[rx_data[6:0]][5:0];
                        seed_len           <= seed_len + 6'd1;
                    end
                end
            end

            M_CLEAR: begin
                l1_clear <= 1'b1;
                l2_clear <= 1'b1;
                seq_idx  <= 6'd0;
                out_cnt  <= 4'd0;
                mstate   <= M_SEED_L1;
            end

            M_SEED_L1: begin
                if (seq_idx < seed_len) begin
                    cur_char_idx <= seed_buf[seq_idx];
                    l1_start     <= 1'b1;
                    mstate       <= M_SEED_W1;
                end else begin
                    mstate <= M_GEN_DENSE;
                end
            end

            M_SEED_W1: begin
                if (l1_done) begin
                    l2_start <= 1'b1;
                    mstate   <= M_SEED_W2;
                end
            end

            M_SEED_W2: begin
                if (l2_done) begin
                    seq_idx <= seq_idx + 6'd1;
                    mstate  <= M_SEED_L1;
                end
            end

            M_GEN_DENSE: begin
                if (out_cnt < NUM_GEN[3:0]) begin
                    dense_start <= 1'b1;
                    mstate      <= M_GEN_DWAIT;
                end else begin
                    out_tx_idx <= 4'd0;
                    mstate     <= M_TX;
                end
            end

            M_GEN_DWAIT: begin
                if (dense_done) begin
                    argmax_start <= 1'b1;      // ← start argmax scan
                    mstate       <= M_GEN_AWAIT;
                end
            end

            M_GEN_AWAIT: begin                 // ← NEW STATE
                if (argmax_done)
                    mstate <= M_GEN_SAVE;
            end

            M_GEN_SAVE: begin
                out_buf[out_cnt] <= i2c_rom[pred_idx][7:0];
                cur_char_idx     <= pred_idx[$clog2(ENC_WIDTH)-1:0];
                l1_start         <= 1'b1;
                out_cnt          <= out_cnt + 4'd1;
                mstate           <= M_GEN_W1;
            end

            M_GEN_W1: begin
                if (l1_done) begin
                    l2_start <= 1'b1;
                    mstate   <= M_GEN_W2;
                end
            end

            M_GEN_W2: begin
                if (l2_done)
                    mstate <= M_GEN_DENSE;
            end

            M_TX: begin
                if (out_tx_idx < out_cnt && !tx_busy) begin
                    tx_data  <= out_buf[out_tx_idx];
                    tx_start <= 1'b1;
                    mstate   <= M_TX_WAIT;
                end else if (out_tx_idx >= out_cnt) begin
                    if (!tx_busy) begin
                        tx_data  <= 8'h0A;
                        tx_start <= 1'b1;
                        mstate   <= M_IDLE;
                    end
                end
            end

            M_TX_WAIT: begin
                if (!tx_busy) begin
                    out_tx_idx <= out_tx_idx + 4'd1;
                    mstate     <= M_TX;
                end
            end

            default: mstate <= M_IDLE;
            endcase
        end
    end
endmodule

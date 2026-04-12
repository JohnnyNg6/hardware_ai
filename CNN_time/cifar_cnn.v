`timescale 1ns / 1ps
// ============================================================================
// cifar_cnn.v — Pipelined: Conv1 ∥ Conv2 → Dense
//
// Conv1 writes c1_buf via BRAM port-A.
// Conv2 reads  c1_buf via BRAM port-B (true dual-port, zero extra cost).
// Conv2 starts as soon as Conv1 has produced enough rows and stalls
// whenever it would need a row Conv1 hasn't finished yet.
// Dense starts after Conv2 is fully done.
// ============================================================================
module cifar_cnn (
    input  wire        clk,
    input  wire        rst_n,
    // image load port
    input  wire [11:0] img_wr_addr,
    input  wire signed [15:0] img_wr_data,
    input  wire        img_wr_en,
    // control
    input  wire        start,
    output reg         done,
    output wire [3:0]  digit_out,
    output wire        result_valid
);

// ======================== Buffers ========================
(* ram_style = "block" *) reg signed [15:0] img_buf [0:4095];   // 32x32x3
(* ram_style = "block" *) reg signed [15:0] c1_buf  [0:16383];  // 16x16x64
(* ram_style = "block" *) reg signed [15:0] c2_buf  [0:4095];   // 8x8x64

// image load
always @(posedge clk) if (img_wr_en) img_buf[img_wr_addr] <= img_wr_data;

// ======================== Conv1 ========================
wire [15:0] c1_ia;  reg  signed [15:0] c1_id;
wire [15:0] c1_oa;  wire signed [15:0] c1_od;
wire c1_owe, c1_done, c1_row_done;
reg  c1_start;

// img_buf read (Conv1)
always @(posedge clk) c1_id <= img_buf[c1_ia[11:0]];

// c1_buf WRITE port-A (Conv1 output)
always @(posedge clk) if (c1_owe) c1_buf[c1_oa[13:0]] <= c1_od;

conv1_block u_c1 (
    .clk(clk), .rst_n(rst_n),
    .start(c1_start), .stall(1'b0),      // Conv1 never stalls
    .done(c1_done),
    .in_addr(c1_ia), .in_data(c1_id),
    .out_addr(c1_oa), .out_data(c1_od), .out_we(c1_owe),
    .cur_oh(), .row_done(c1_row_done)
);

// ======================== Conv2 ========================
wire [15:0] c2_ia;  reg  signed [15:0] c2_id;
wire [15:0] c2_oa;  wire signed [15:0] c2_od;
wire c2_owe, c2_done, c2_row_done;
wire [7:0]  c2_cur_oh;
reg  c2_start;

// c1_buf READ port-B (Conv2 input — true dual-port)
always @(posedge clk) c2_id <= c1_buf[c2_ia[13:0]];

// c2_buf write
always @(posedge clk) if (c2_owe) c2_buf[c2_oa[11:0]] <= c2_od;

// --- Row-stall logic ---------------------------------------------------
// Conv2 output-row oh needs Conv1 rows 0 .. min(oh*2+2, 15).
// So we require c1_rows_done >= min(oh*2+3, 16).
wire [4:0] c2_raw_need = {1'b0, c2_cur_oh[3:0]} * 5'd2 + 5'd3;
wire [4:0] c2_need     = (c2_raw_need > 5'd16) ? 5'd16 : c2_raw_need;
wire       c2_stall    = ~c1_finished && (c1_rows_done < c2_need);
// -----------------------------------------------------------------------

conv2_block u_c2 (
    .clk(clk), .rst_n(rst_n),
    .start(c2_start), .stall(c2_stall),
    .done(c2_done),
    .in_addr(c2_ia), .in_data(c2_id),
    .out_addr(c2_oa), .out_data(c2_od), .out_we(c2_owe),
    .cur_oh(c2_cur_oh), .row_done(c2_row_done)
);

// ======================== Dense ========================
wire [15:0] d_ia;  reg signed [15:0] d_id;
wire d_done;       reg d_start;

always @(posedge clk) d_id <= c2_buf[d_ia[11:0]];

dense_layer #(.NI(4096),.NO(10)) u_d (
    .clk(clk), .rst_n(rst_n),
    .start(d_start), .done(d_done),
    .in_addr(d_ia), .in_data(d_id),
    .digit(digit_out), .result_valid(result_valid)
);

// ======================== Row counter ========================
reg [4:0] c1_rows_done;
reg       c1_finished;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        c1_rows_done <= 0;
        c1_finished  <= 0;
    end else begin
        if (start) begin                   // reset on new inference
            c1_rows_done <= 0;
            c1_finished  <= 0;
        end
        if (c1_row_done)
            c1_rows_done <= c1_rows_done + 1;
        if (c1_done)
            c1_finished <= 1;
    end
end

// ======================== Pipelined Sequencer ========================
//
//  Time ──────────────────────────────────────────────────────►
//       |══ Conv1 ═══════════════|
//              |═══════════════ Conv2 ════════════════|═ Dense ═|
//       ^     ^                  ^                   ^         ^
//    P_IDLE  P_C1RUN          P_OVERLAP→P_C2TAIL   P_DN     P_IDLE
//
localparam [2:0] P_IDLE=0, P_C1RUN=1, P_OVERLAP=2, P_C2TAIL=3, P_DN=4;
reg [2:0] pst;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        pst<=P_IDLE; c1_start<=0; c2_start<=0; d_start<=0; done<=0;
    end else begin
        c1_start<=0; c2_start<=0; d_start<=0; done<=0;
        case (pst)

        P_IDLE: if (start) begin
            c1_start <= 1;
            pst      <= P_C1RUN;
        end

        // Wait until Conv1 has produced >= 3 rows, then launch Conv2
        P_C1RUN: begin
            if (c1_rows_done >= 5'd3) begin
                c2_start <= 1;
                pst      <= P_OVERLAP;
            end
        end

        // Both layers running in parallel
        P_OVERLAP: begin
            if (c1_done && c2_done) begin   // both finish same cycle
                d_start <= 1; pst <= P_DN;
            end else if (c1_done) begin
                pst <= P_C2TAIL;            // Conv1 done first (typical)
            end
        end

        // Conv1 already done, Conv2 finishing
        P_C2TAIL: begin
            if (c2_done) begin
                d_start <= 1; pst <= P_DN;
            end
        end

        // Dense running
        P_DN: begin
            if (d_done) begin
                done <= 1; pst <= P_IDLE;
            end
        end

        default: pst <= P_IDLE;
        endcase
    end
end
endmodule

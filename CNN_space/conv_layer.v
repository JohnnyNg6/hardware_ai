// ============================================================================
// conv_layer.v — Serial convolution engine (1 MAC/clk), Q8.8 fixed-point
// ============================================================================
`timescale 1ns / 1ps

module conv_layer #(
    parameter integer IH    = 32,   // input height
    parameter integer IW    = 32,   // input width
    parameter integer IC    = 3,    // input channels
    parameter integer FH    = 5,    // filter height
    parameter integer FW    = 5,    // filter width
    parameter integer NF    = 64,   // number of filters
    parameter integer ST    = 2,    // stride
    parameter integer PT    = 0,    // pad top
    parameter integer PL    = 0,    // pad left
    parameter integer OH    = 14,   // output height
    parameter integer OW    = 14,   // output width
    parameter         WFILE = "conv1_w.mem",
    parameter         BFILE = "conv1_b.mem"
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    output reg         done,
    // input-buffer read  (addr registered here, data from external sync BRAM)
    output reg  [15:0] in_addr,
    input  wire signed [15:0] in_data,
    // output-buffer write
    output reg  [15:0] out_addr,
    output reg  signed [15:0] out_data,
    output reg         out_we
);

/* --- constants ---------------------------------------------------------- */
localparam integer KSZ   = FH * FW * IC;          // kernel volume
localparam integer IW_C  = IW * IC;               // row stride in input
localparam integer WDEP  = NF * KSZ;              // weight-ROM depth

/* --- weight & bias ROMs ------------------------------------------------- */
(* ram_style = "block" *)
reg signed [15:0] wrom [0:WDEP-1];
initial $readmemh(WFILE, wrom);

(* ram_style = "distributed" *)
reg signed [15:0] brom [0:NF-1];
initial $readmemh(BFILE, brom);

reg [15:0] wa_r;
reg signed [15:0] w_rd;
always @(posedge clk) w_rd <= wrom[wa_r];          // 1-clk sync read

/* --- FSM ---------------------------------------------------------------- */
localparam [2:0] S_IDLE  = 0, S_BIAS = 1, S_PIPE = 2,
                 S_MAC   = 3, S_WR   = 4, S_DONE = 5;
reg [2:0] st;

/* --- counters ----------------------------------------------------------- */
reg [6:0]  f_r;                     // filter    0..NF-1
reg [4:0]  oh_r, ow_r;             // output position
reg [3:0]  kh_r, kw_r;             // kernel position
reg [6:0]  kc_r;                   // kernel channel
reg [9:0]  k_r;                    // flat kernel index

/* --- accumulator -------------------------------------------------------- */
reg signed [47:0] acc;

/* --- pad-zero 2-stage pipeline ------------------------------------------ */
reg pz_d1, pz_d2;

/* --- next kernel position (combinational) ------------------------------- */
wire kc_last = (kc_r == IC  - 1);
wire kw_last = (kw_r == FW  - 1) && kc_last;
wire [6:0] kc_nxt = kc_last ? 7'd0            : kc_r + 7'd1;
wire [3:0] kw_nxt = kc_last ? (kw_last ? 4'd0 : kw_r + 4'd1) : kw_r;
wire [3:0] kh_nxt = kw_last ? kh_r + 4'd1     : kh_r;

/* --- address helpers (for "next" element) ------------------------------- */
wire signed [8:0] ih_nxt = $signed({1'b0, oh_r}) * ST + $signed({1'b0, kh_nxt}) - PT;
wire signed [8:0] iw_nxt = $signed({1'b0, ow_r}) * ST + $signed({1'b0, kw_nxt}) - PL;
wire pz_nxt = (ih_nxt < 0) | (ih_nxt >= IH) | (iw_nxt < 0) | (iw_nxt >= IW);
wire [15:0] ia_nxt = ih_nxt[7:0] * IW_C[15:0] + iw_nxt[7:0] * IC[15:0] + kc_nxt;

/* --- address helpers (for "first" element kh=kw=kc=0) ------------------- */
wire signed [8:0] ih0 = $signed({1'b0, oh_r}) * ST - PT;
wire signed [8:0] iw0 = $signed({1'b0, ow_r}) * ST - PL;
wire pz0 = (ih0 < 0) | (ih0 >= IH) | (iw0 < 0) | (iw0 >= IW);
wire [15:0] ia0 = ih0[7:0] * IW_C[15:0] + iw0[7:0] * IC[15:0];

/* --- product & saturation ----------------------------------------------- */
wire signed [15:0] in_use = pz_d2 ? 16'sd0 : in_data;
wire signed [31:0] prod   = in_use * w_rd;

wire signed [15:0] z_raw = acc[23:8];
wire               z_ovf = (acc[47:24] != {24{acc[23]}});
wire signed [15:0] z_sat = z_ovf ? (acc[47] ? 16'sh8000 : 16'sh7FFF) : z_raw;
wire signed [15:0] z_relu= z_sat[15] ? 16'sd0 : z_sat;

/* --- output address ----------------------------------------------------- */
wire [15:0] oa_calc = ({11'd0, oh_r} * OW + {11'd0, ow_r}) * NF + {9'd0, f_r};

/* --- main FSM ----------------------------------------------------------- */
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        st<=S_IDLE; done<=0; out_we<=0;
        f_r<=0; oh_r<=0; ow_r<=0;
        kh_r<=0; kw_r<=0; kc_r<=0; k_r<=0;
        acc<=0; pz_d1<=0; pz_d2<=0;
        in_addr<=0; wa_r<=0;
        out_addr<=0; out_data<=0;
    end else begin
        done   <= 1'b0;
        out_we <= 1'b0;
        case (st)
        /* ---- idle --------------------------------------------------- */
        S_IDLE: if (start) begin
            f_r<=0; oh_r<=0; ow_r<=0;
            st <= S_BIAS;
        end
        /* ---- load bias, issue addr[0] ------------------------------- */
        S_BIAS: begin
            acc <= {{24{brom[f_r][15]}}, brom[f_r], 8'b0};
            in_addr <= pz0 ? 16'd0 : ia0;
            wa_r    <= f_r * KSZ[15:0];
            pz_d1   <= pz0;
            kh_r<=0; kw_r<=0; kc_r<=0; k_r<=0;
            st <= S_PIPE;
        end
        /* ---- pipeline fill: issue addr[1], shift pad ----------------- */
        S_PIPE: begin
            pz_d2 <= pz_d1;
            pz_d1 <= pz_nxt;
            in_addr <= pz_nxt ? 16'd0 : ia_nxt;
            wa_r    <= wa_r + 16'd1;
            kh_r<=kh_nxt; kw_r<=kw_nxt; kc_r<=kc_nxt;
            k_r <= 10'd0;           // MAC index starts at 0
            st <= S_MAC;
        end
        /* ---- MAC (1 element / clk) ---------------------------------- */
        S_MAC: begin
            acc <= acc + {{16{prod[31]}}, prod};
            pz_d2 <= pz_d1;
            if (k_r == KSZ[9:0] - 10'd1) begin
                st <= S_WR;
            end else begin
                pz_d1 <= pz_nxt;
                in_addr <= pz_nxt ? 16'd0 : ia_nxt;
                wa_r    <= wa_r + 16'd1;
                kh_r<=kh_nxt; kw_r<=kw_nxt; kc_r<=kc_nxt;
                k_r <= k_r + 10'd1;
            end
        end
        /* ---- ReLU + write, advance position ------------------------- */
        S_WR: begin
            out_addr <= oa_calc;
            out_data <= z_relu;
            out_we   <= 1'b1;
            // advance (oh, ow, f)
            if (f_r < NF[6:0] - 7'd1) begin
                f_r <= f_r + 7'd1; st <= S_BIAS;
            end else begin
                f_r <= 7'd0;
                if (ow_r < OW[4:0] - 5'd1) begin
                    ow_r <= ow_r + 5'd1; st <= S_BIAS;
                end else begin
                    ow_r <= 5'd0;
                    if (oh_r < OH[4:0] - 5'd1) begin
                        oh_r <= oh_r + 5'd1; st <= S_BIAS;
                    end else
                        st <= S_DONE;
                end
            end
        end
        /* ---- done --------------------------------------------------- */
        S_DONE: begin done<=1; st<=S_IDLE; end
        default: st <= S_IDLE;
        endcase
    end
end
endmodule

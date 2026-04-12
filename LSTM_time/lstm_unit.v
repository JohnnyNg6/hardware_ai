// ============================================================
// lstm_unit.v — One LSTM hidden-state node
//
//   Computes 4 gates (f,i,g,o) SEQUENTIALLY through a single
//   shared multiplier so that only 1 DSP48 is consumed.
//   All units in a layer receive the SAME broadcast concat
//   stream and operate IN PARALLEL.
//
//   Weight ROM layout (one file per unit):
//     Gate f : [bias, w0 … wN-1]   (N = CONCAT_SIZE words)
//     Gate i : [bias, w0 … wN-1]
//     Gate g : [bias, w0 … wN-1]   (candidate cell)
//     Gate o : [bias, w0 … wN-1]
//   Total = 4 * (CONCAT_SIZE + 1) words, Q8.8 hex
// ============================================================
`timescale 1ns/1ps
module lstm_unit #(
    parameter integer CONCAT_SIZE = 191,
    parameter         WEIGHT_FILE = "w.mem"
)(
    input  wire               clk,
    input  wire               rst_n,
    // ── layer-driven control ──
    input  wire               clear_state,        // reset c & h to 0
    input  wire               load_bias,          // 1 clk: load bias for current gate
    input  wire               mac_en,             // 1 = accumulate this cycle
    input  wire               store_gate,         // 1 clk: apply activation & store
    input  wire [1:0]         gate_id,            // which gate: 0=f 1=i 2=g 3=o
    input  wire               do_update,          // begin 3-cycle cell/h update
    // ── data ──
    input  wire signed [15:0] concat_din,         // broadcast element
    input  wire [$clog2(CONCAT_SIZE>1?CONCAT_SIZE:2)-1:0] w_offset,
    // ── output ──
    output wire signed [15:0] h_out
);
    // ── parameters ──
    localparam GW = CONCAT_SIZE + 1;              // words per gate
    localparam TW = 4 * GW;

    // ── weight ROM ──
    (* ram_style = "distributed" *)
    reg signed [15:0] w [0:TW-1];
    initial $readmemh(WEIGHT_FILE, w);

    // ── state registers ──
    reg signed [15:0] c_reg, h_reg;
    assign h_out = h_reg;

    // ── gate result registers ──
    reg signed [15:0] gf, gi, gg, go_r;

    // ── accumulator ──
    reg signed [47:0] acc;

    // ── activation look-ups (combinational) ──
    wire signed [15:0] z_raw = acc[23:8];
    wire               z_ovf = (acc[47:24] != {24{acc[23]}});
    wire signed [15:0] z_sat = z_ovf ? (acc[47] ? 16'sh8000 : 16'sh7FFF)
                                     : z_raw;
    wire signed [15:0] sig_z, tnh_z;
    sigmoid  u_sig (.x(z_sat), .y(sig_z));
    tanh_act u_tnh (.x(z_sat), .y(tnh_z));

    // ── weight address ──
    wire [$clog2(TW)-1:0] wa = gate_id * GW[15:0] + {1'b0, w_offset} + 1;
    wire [$clog2(TW)-1:0] ba = gate_id * GW[15:0];

    // ── MAC product (uses 1 DSP48) ──
    wire signed [31:0] mac_prod = concat_din * w[wa];

    // ── cell-update shared multiplier ──
    //    We sequence 3 multiplies in 3 clocks using the update FSM.
    reg [1:0] upd_cnt;                            // 0,1,2
    reg signed [15:0] upd_tmp;                    // temp f*c result

    wire signed [15:0] upd_a = (upd_cnt == 2'd0) ? gf  :
                               (upd_cnt == 2'd1) ? gi  : go_r;
    wire signed [15:0] tanh_c;
    tanh_act u_tnh_c (.x(c_reg), .y(tanh_c));

    wire signed [15:0] upd_b = (upd_cnt == 2'd0) ? c_reg :
                               (upd_cnt == 2'd1) ? gg    : tanh_c;
    wire signed [31:0] upd_p = upd_a * upd_b;

    // Q16.16 → Q8.8 with saturation
    wire signed [15:0] upd_q88;
    wire upd_ovf = (upd_p[31:24] != {8{upd_p[23]}});
    assign upd_q88 = upd_ovf ? (upd_p[31] ? 16'sh8000 : 16'sh7FFF)
                             : upd_p[23:8];

    // ── update FSM ──
    reg upd_active;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            c_reg <= 0; h_reg <= 0; acc <= 0;
            gf <= 0; gi <= 0; gg <= 0; go_r <= 0;
            upd_cnt <= 0; upd_tmp <= 0; upd_active <= 0;
        end else begin
            // ── clear ──
            if (clear_state) begin
                c_reg <= 0; h_reg <= 0;
            end

            // ── bias load ──
            if (load_bias)
                acc <= {{24{w[ba][15]}}, w[ba], 8'b0};

            // ── MAC ──
            if (mac_en)
                acc <= acc + {{16{mac_prod[31]}}, mac_prod};

            // ── store gate result ──
            if (store_gate) begin
                case (gate_id)
                    2'd0: gf   <= sig_z;          // f: sigmoid
                    2'd1: gi   <= sig_z;          // i: sigmoid
                    2'd2: gg   <= tnh_z;          // g: tanh
                    2'd3: go_r <= sig_z;          // o: sigmoid
                endcase
            end

            // ── cell / hidden update (3 cycles) ──
            if (do_update && !upd_active) begin
                upd_active <= 1;
                upd_cnt    <= 0;
            end

            if (upd_active) begin
                case (upd_cnt)
                    2'd0: begin                       // f * c_old
                        upd_tmp  <= upd_q88;
                        upd_cnt  <= 2'd1;
                    end
                    2'd1: begin                       // i * g  then c_new
                        c_reg    <= upd_tmp + upd_q88;
                        upd_cnt  <= 2'd2;
                    end
                    2'd2: begin                       // o * tanh(c_new)
                        h_reg      <= upd_q88;
                        upd_active <= 0;
                    end
                endcase
            end
        end
    end
endmodule

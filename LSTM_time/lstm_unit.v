// ============================================================
// lstm_unit.v — One LSTM hidden-state unit
//
// Wraps a single neuron instance (1 DSP48) that processes
// all 4 gates (f,i,g,o) SEQUENTIALLY.  The weight BRAM holds
// all 4 gates' weights concatenated.
//
// Gate order in BRAM (matches export_weights.py):
//   offset 0*GW … 1*GW-1 : gate f (forget)
//   offset 1*GW … 2*GW-1 : gate i (input)
//   offset 2*GW … 3*GW-1 : gate g (candidate)
//   offset 3*GW … 4*GW-1 : gate o (output)
// where GW = CONCAT_SIZE + 1
//
// Interface — driven by the layer controller:
//   neuron_start : 1-clk pulse, begins MAC for current gate_id
//   gate_id      : which gate (0–3), stable during computation
//   din/din_valid: concat element stream (broadcast from controller)
//   store_gate   : 1-clk pulse after neuron finishes, latches activated result
//   do_update    : 1-clk pulse, starts 3-cycle cell/h update
//   clear_state  : reset c and h to zero
//
// Timing:
//   neuron_start → neuron processes → done → store_gate
//   (layer controller manages the exact cycle counts)
//
// Pipeline latency (store_gate to gate register update): 0
//   store_gate is a direct registered write using combinational
//   sigmoid/tanh of the held neuron_out.
// ============================================================
`timescale 1ns/1ps
module lstm_unit #(
    parameter integer CONCAT_SIZE = 191,
    parameter         WEIGHT_FILE = "w.mem"
)(
    input  wire               clk,
    input  wire               rst_n,
    // ── layer-driven control ──
    input  wire               clear_state,
    input  wire               neuron_start,
    input  wire [1:0]         gate_id,
    input  wire signed [15:0] din,
    input  wire               din_valid,
    input  wire               store_gate,
    input  wire               do_update,
    // ── outputs ──
    output wire signed [15:0] h_out,
    output wire               neuron_done
);
    // ── derived parameters ──
    localparam GW = CONCAT_SIZE + 1;              // words per gate
    localparam TW = 4 * GW;                       // total BRAM depth
    localparam AW = $clog2(TW > 1 ? TW : 2);

    // ── base address for current gate ──
    //    gate_id is 2 bits, GW is a synthesis-time constant.
    wire [AW-1:0] base_addr;
    assign base_addr = (gate_id == 2'd0) ? {AW{1'b0}} :
                       (gate_id == 2'd1) ? GW[AW-1:0] :
                       (gate_id == 2'd2) ? (2 * GW[AW-1:0]) :
                                           (3 * GW[AW-1:0]);

    // ── neuron instance (1 DSP48) ──
    wire signed [15:0] neuron_out;

    neuron #(
        .NUM_INPUTS  (CONCAT_SIZE),
        .MEM_DEPTH   (TW),
        .WEIGHT_FILE (WEIGHT_FILE)
    ) u_neuron (
        .clk       (clk),
        .rst_n     (rst_n),
        .start     (neuron_start),
        .base_addr (base_addr),
        .din       (din),
        .din_valid (din_valid),
        .dout      (neuron_out),
        .done      (neuron_done)
    );

    // ── activation look-ups (combinational, distributed ROM) ──
    wire signed [15:0] sig_out;
    sigmoid u_sig (.x(neuron_out), .y(sig_out));

    // Shared tanh: gate-g activation when !upd_active,
    //              tanh(c_new) during cell update.
    reg                upd_active;
    reg  signed [15:0] c_reg, h_reg;
    wire signed [15:0] tanh_in  = upd_active ? c_reg : neuron_out;
    wire signed [15:0] tanh_out;
    tanh_act u_tnh (.x(tanh_in), .y(tanh_out));

    assign h_out = h_reg;

    // ── gate result registers ──
    reg signed [15:0] gf, gi_r, gg, go_r;

    // ── cell-update multiplier (may infer DSP48 when neuron's is idle) ──
    reg  signed [15:0] upd_a, upd_b;
    wire signed [31:0] upd_p = upd_a * upd_b;
    wire               upd_ovf = (upd_p[31:24] != {8{upd_p[23]}});
    wire signed [15:0] upd_q88 = upd_ovf ? (upd_p[31] ? 16'sh8000
                                                       : 16'sh7FFF)
                                         : upd_p[23:8];

    // ── cell-update FSM ──
    reg [1:0]         upd_cnt;
    reg signed [15:0] upd_tmp;   // holds f*c_old

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            c_reg      <= 16'd0;
            h_reg      <= 16'd0;
            gf         <= 16'd0;
            gi_r       <= 16'd0;
            gg         <= 16'd0;
            go_r       <= 16'd0;
            upd_active <= 1'b0;
            upd_cnt    <= 2'd0;
            upd_tmp    <= 16'd0;
            upd_a      <= 16'd0;
            upd_b      <= 16'd0;
        end else begin
            // ── clear ──
            if (clear_state) begin
                c_reg <= 16'd0;
                h_reg <= 16'd0;
            end

            // ── store activated gate result ──
            if (store_gate) begin
                case (gate_id)
                    2'd0: gf   <= sig_out;       // f: sigmoid
                    2'd1: gi_r <= sig_out;        // i: sigmoid
                    2'd2: gg   <= tanh_out;       // g: tanh
                    2'd3: go_r <= sig_out;        // o: sigmoid
                endcase
            end

            // ── cell / hidden update (3 cycles) ──
            if (do_update && !upd_active) begin
                upd_active <= 1'b1;
                upd_cnt    <= 2'd0;
                upd_a      <= gf;
                upd_b      <= c_reg;
            end

            if (upd_active) begin
                case (upd_cnt)
                    2'd0: begin
                        // Cycle 0: compute f * c_old
                        upd_tmp <= upd_q88;
                        upd_cnt <= 2'd1;
                        upd_a   <= gi_r;
                        upd_b   <= gg;
                    end
                    2'd1: begin
                        // Cycle 1: compute i * g, then c_new = f*c + i*g
                        c_reg   <= upd_tmp + upd_q88;
                        upd_cnt <= 2'd2;
                        upd_a   <= go_r;
                        upd_b   <= tanh_out; // tanh(c_new) via shared tanh
                    end
                    2'd2: begin
                        // Cycle 2: compute o * tanh(c_new), h_new
                        h_reg      <= upd_q88;
                        upd_active <= 1'b0;
                    end
                    default: upd_active <= 1'b0;
                endcase
            end
        end
    end
endmodule

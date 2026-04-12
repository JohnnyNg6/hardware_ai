// ============================================================
// lstm_layer.v — One LSTM layer with HIDDEN_SIZE parallel units
//
//   The shared controller forms the concat vector and streams
//   it to all units simultaneously.  Gates are processed
//   sequentially (f → i → g → o) then a 3-cycle update runs.
// ============================================================
`timescale 1ns/1ps
module lstm_layer #(
    parameter integer INPUT_SIZE  = 63,
    parameter integer HIDDEN_SIZE = 128,
    parameter integer LAYER_ID    = 1        // for weight file names
)(
    input  wire               clk,
    input  wire               rst_n,
    input  wire               clear_state,
    input  wire               start_step,
    input  wire signed [15:0] x_in   [0:INPUT_SIZE-1],
    output wire signed [15:0] h_out  [0:HIDDEN_SIZE-1],
    output reg                step_done
);
    localparam CS  = INPUT_SIZE + HIDDEN_SIZE;
    localparam CW  = $clog2(CS > 1 ? CS : 2);
    localparam GW  = CS + 1;

    // ── concat register ──
    reg signed [15:0] concat [0:CS-1];

    // ── wires to / from units ──
    wire signed [15:0] h_wire [0:HIDDEN_SIZE-1];
    reg                lb, me, sg, du;
    reg  [1:0]         gid;
    reg  [CW-1:0]      addr;
    wire signed [15:0] cd = concat[addr];

    // ── instantiate parallel units ──
    genvar gi;
    generate
        for (gi = 0; gi < HIDDEN_SIZE; gi = gi + 1) begin : gu
            lstm_unit #(
                .CONCAT_SIZE (CS),
                .WEIGHT_FILE ($sformatf("weights/l%0d_u%0d.mem",
                                         LAYER_ID, gi))
            ) u (
                .clk          (clk),
                .rst_n        (rst_n),
                .clear_state  (clear_state),
                .load_bias    (lb),
                .mac_en       (me),
                .store_gate   (sg),
                .gate_id      (gid),
                .do_update    (du),
                .concat_din   (cd),
                .w_offset     (addr[CW-1:0]),
                .h_out        (h_wire[gi])
            );
        end
    endgenerate

    genvar go;
    generate
        for (go = 0; go < HIDDEN_SIZE; go = go + 1) begin : gh
            assign h_out[go] = h_wire[go];
        end
    endgenerate

    // ── layer controller FSM ──
    localparam [3:0] L_IDLE  = 0, L_LOAD  = 1, L_BIAS  = 2,
                     L_MAC   = 3, L_STORE = 4, L_NEXT  = 5,
                     L_UPD   = 6, L_WAIT  = 7, L_DONE  = 8;
    reg [3:0] lstate;
    reg [1:0] cur_gate;
    reg [2:0] wait_cnt;
    reg       mac_first;           // FIX: flag for first MAC cycle

    integer ii;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lstate    <= L_IDLE; cur_gate <= 0; addr <= 0;
            lb <= 0; me <= 0; sg <= 0; du <= 0; gid <= 0;
            step_done <= 0; wait_cnt <= 0;
            mac_first <= 0;
        end else begin
            // defaults
            lb <= 0; me <= 0; sg <= 0; du <= 0; step_done <= 0;

            case (lstate)
            // ── idle ──
            L_IDLE: if (start_step) begin
                // snapshot concat = [x_in , h_prev]
                for (ii = 0; ii < INPUT_SIZE; ii = ii + 1)
                    concat[ii] <= x_in[ii];
                for (ii = 0; ii < HIDDEN_SIZE; ii = ii + 1)
                    concat[INPUT_SIZE + ii] <= h_wire[ii];
                cur_gate <= 0;
                lstate   <= L_BIAS;
            end

            // ── load bias for current gate ──
            L_BIAS: begin
                gid       <= cur_gate;
                lb        <= 1;
                addr      <= 0;
                mac_first <= 1;          // FIX
                lstate    <= L_MAC;
            end

            // ── stream concat elements ──
            // FIX: On the first L_MAC cycle after L_BIAS, me is set
            //      but addr is NOT incremented, so that the unit sees
            //      mac_en=1 together with addr=0 on the next clock.
            //      On the final element (addr==CS-1) we suppress me
            //      so no extra MAC leaks into L_STORE.
            L_MAC: begin
                gid <= cur_gate;
                me  <= 1;
                if (mac_first) begin
                    mac_first <= 0;
                    // addr stays at 0 — do NOT increment
                end else if (addr == CS[CW-1:0] - 1) begin
                    me     <= 0;         // FIX: override — no more MAC
                    lstate <= L_STORE;
                end else begin
                    addr <= addr + 1;
                end
            end

            // ── store gate activation ──
            L_STORE: begin
                gid    <= cur_gate;
                sg     <= 1;
                lstate <= L_NEXT;
            end

            // ── advance to next gate or start update ──
            L_NEXT: begin
                if (cur_gate == 2'd3) begin
                    du       <= 1;
                    wait_cnt <= 0;
                    lstate   <= L_UPD;
                end else begin
                    cur_gate <= cur_gate + 1;
                    addr     <= 0;
                    lstate   <= L_BIAS;
                end
            end

            // ── wait for 3-cycle cell update ──
            L_UPD: begin
                if (wait_cnt == 3'd4) begin
                    lstate <= L_DONE;
                end else
                    wait_cnt <= wait_cnt + 1;
            end

            // ── done ──
            L_DONE: begin
                step_done <= 1;
                lstate    <= L_IDLE;
            end

            default: lstate <= L_IDLE;
            endcase
        end
    end
endmodule

// ============================================================
// lstm_layer.v — One LSTM layer: HIDDEN_SIZE parallel units
//
// The controller forms the concat vector, then for each of
// the 4 gates:
//   1. Pulses neuron_start (units begin bias load from BRAM)
//   2. Waits 1 cycle (units need 2 internal cycles before MAC)
//   3. Streams concat[0..CS-1] at 1 element/clk
//   4. Waits for neuron pipeline drain + done
//   5. Pulses store_gate
// After all 4 gates, pulses do_update and waits 4 cycles.
//
// Pipeline timing per gate (CS = CONCAT_SIZE):
//   L_GATE_START (1)  + L_WAIT (1)  + L_FEED (CS)
//   + L_DRAIN (1)  + L_WAIT2 (1)  + L_STORE (1)  = CS + 5
//
// Total per step ≈ 4*(CS+5) + 6  clocks
// ============================================================
`timescale 1ns/1ps
module lstm_layer #(
    parameter integer INPUT_SIZE  = 63,
    parameter integer HIDDEN_SIZE = 128,
    parameter integer LAYER_ID    = 1
)(
    input  wire               clk,
    input  wire               rst_n,
    input  wire               clear_state,
    input  wire               start_step,
    input  wire signed [15:0] x_in  [0:INPUT_SIZE-1],
    output wire signed [15:0] h_out [0:HIDDEN_SIZE-1],
    output reg                step_done
);
    // ── derived constants ──
    localparam CS = INPUT_SIZE + HIDDEN_SIZE;
    localparam CW = $clog2(CS > 1 ? CS : 2);

    // ── concat register ──
    reg signed [15:0] concat [0:CS-1];

    // ── control signals to units (active-high, 1-clk pulses) ──
    reg               ns_pulse;          // neuron_start
    reg  [1:0]        gid;               // gate_id
    reg  signed [15:0] feed_data;        // concat element
    reg               feed_valid;        // din_valid
    reg               sg_pulse;          // store_gate
    reg               du_pulse;          // do_update

    // ── one neuron_done for timing (all finish together) ──
    wire [HIDDEN_SIZE-1:0] nd_vec;
    wire                   any_done = nd_vec[0];

    // ── parallel unit instantiation ──
    wire signed [15:0] h_wire [0:HIDDEN_SIZE-1];

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
                .neuron_start (ns_pulse),
                .gate_id      (gid),
                .din          (feed_data),
                .din_valid    (feed_valid),
                .store_gate   (sg_pulse),
                .do_update    (du_pulse),
                .h_out        (h_wire[gi]),
                .neuron_done  (nd_vec[gi])
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
    localparam [3:0]
        L_IDLE       = 4'd0,
        L_SNAP       = 4'd1,   // snapshot concat
        L_GATE_START = 4'd2,   // pulse neuron_start
        L_WAIT1      = 4'd3,   // wait for bias BRAM latency
        L_FEED       = 4'd4,   // stream concat elements
        L_DRAIN      = 4'd5,   // pipeline drain
        L_WAIT2      = 4'd6,   // wait for neuron S_OUT
        L_STORE      = 4'd7,   // pulse store_gate
        L_NEXT       = 4'd8,   // advance gate or start update
        L_UPD        = 4'd9,   // pulse do_update
        L_UPD_WAIT   = 4'd10,  // wait for 3-cycle update
        L_DONE       = 4'd11;

    reg [3:0]     lstate;
    reg [1:0]     cur_gate;
    reg [CW-1:0]  addr;
    reg [2:0]     wait_cnt;

    integer ii;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lstate    <= L_IDLE;
            cur_gate  <= 2'd0;
            addr      <= {CW{1'b0}};
            wait_cnt  <= 3'd0;
            ns_pulse  <= 1'b0;
            gid       <= 2'd0;
            feed_data <= 16'd0;
            feed_valid <= 1'b0;
            sg_pulse  <= 1'b0;
            du_pulse  <= 1'b0;
            step_done <= 1'b0;
        end else begin
            // ── pulse defaults ──
            ns_pulse  <= 1'b0;
            sg_pulse  <= 1'b0;
            du_pulse  <= 1'b0;
            feed_valid <= 1'b0;
            step_done <= 1'b0;

            case (lstate)
                // ── idle ──
                L_IDLE: begin
                    if (start_step)
                        lstate <= L_SNAP;
                end

                // ── snapshot concat = [x_in, h_prev] ──
                L_SNAP: begin
                    for (ii = 0; ii < INPUT_SIZE; ii = ii + 1)
                        concat[ii] <= x_in[ii];
                    for (ii = 0; ii < HIDDEN_SIZE; ii = ii + 1)
                        concat[INPUT_SIZE + ii] <= h_wire[ii];
                    cur_gate <= 2'd0;
                    lstate   <= L_GATE_START;
                end

                // ── pulse neuron_start for current gate ──
                L_GATE_START: begin
                    gid      <= cur_gate;
                    ns_pulse <= 1'b1;
                    addr     <= {CW{1'b0}};
                    lstate   <= L_WAIT1;
                end

                // ── wait 1 clk: units load bias from BRAM ──
                L_WAIT1: begin
                    lstate <= L_FEED;
                end

                // ── stream concat[0..CS-1] ──
                L_FEED: begin
                    feed_valid <= 1'b1;
                    feed_data  <= concat[addr];
                    if (addr == CS[CW-1:0] - {{(CW-1){1'b0}}, 1'b1})
                        lstate <= L_DRAIN;
                    else
                        addr <= addr + {{(CW-1){1'b0}}, 1'b1};
                end

                // ── pipeline drain: last element being processed ──
                L_DRAIN: begin
                    lstate <= L_WAIT2;
                end

                // ── wait for neuron done ──
                L_WAIT2: begin
                    if (any_done)
                        lstate <= L_STORE;
                end

                // ── store activated gate result ──
                L_STORE: begin
                    gid      <= cur_gate;
                    sg_pulse <= 1'b1;
                    lstate   <= L_NEXT;
                end

                // ── advance to next gate or update ──
                L_NEXT: begin
                    if (cur_gate == 2'd3) begin
                        lstate <= L_UPD;
                    end else begin
                        cur_gate <= cur_gate + 2'd1;
                        lstate   <= L_GATE_START;
                    end
                end

                // ── pulse do_update ──
                L_UPD: begin
                    du_pulse <= 1'b1;
                    wait_cnt <= 3'd0;
                    lstate   <= L_UPD_WAIT;
                end

                // ── wait for 3-cycle cell update + 1 margin ──
                L_UPD_WAIT: begin
                    if (wait_cnt == 3'd4)
                        lstate <= L_DONE;
                    else
                        wait_cnt <= wait_cnt + 3'd1;
                end

                // ── done ──
                L_DONE: begin
                    step_done <= 1'b1;
                    lstate    <= L_IDLE;
                end

                default: lstate <= L_IDLE;
            endcase
        end
    end
endmodule

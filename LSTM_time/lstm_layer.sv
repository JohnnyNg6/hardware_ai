// ============================================================
// lstm_layer.v — One LSTM layer: HIDDEN_SIZE parallel units
//   Ports flattened for Verilog-2001 compatibility
// ============================================================
`timescale 1ns/1ps
module lstm_layer #(
    parameter integer INPUT_SIZE  = 63,
    parameter integer HIDDEN_SIZE = 128,
    parameter integer LAYER_ID    = 1
)(
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          clear_state,
    input  wire                          start_step,
    input  wire [INPUT_SIZE*16-1:0]      x_in_flat,
    output wire [HIDDEN_SIZE*16-1:0]     h_out_flat,
    output reg                           step_done
);
    // ── derived constants ──
    localparam CS = INPUT_SIZE + HIDDEN_SIZE;
    localparam CW = $clog2(CS > 1 ? CS : 2);

    // ── concat register (internal — OK as unpacked array) ──
    reg signed [15:0] concat [0:CS-1];

    // ── control signals to units ──
    reg               ns_pulse;
    reg  [1:0]        gid;
    reg  signed [15:0] feed_data;
    reg               feed_valid;
    reg               sg_pulse;
    reg               du_pulse;

    // ── neuron done ──
    wire [HIDDEN_SIZE-1:0] nd_vec;
    wire                   any_done = nd_vec[0];

    // ── internal h wires (unpacked array — OK internally) ──
    wire signed [15:0] h_wire [0:HIDDEN_SIZE-1];

    // ── flatten h_wire → h_out_flat ──
    genvar go;
    generate
        for (go = 0; go < HIDDEN_SIZE; go = go + 1) begin : gh
            assign h_out_flat[go*16 +: 16] = h_wire[go];
        end
    endgenerate

    // ── parallel unit instantiation ──
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

    // ── layer controller FSM ──
    localparam [3:0]
        L_IDLE       = 4'd0,
        L_SNAP       = 4'd1,
        L_GATE_START = 4'd2,
        L_WAIT1      = 4'd3,
        L_FEED       = 4'd4,
        L_DRAIN      = 4'd5,
        L_WAIT2      = 4'd6,
        L_STORE      = 4'd7,
        L_NEXT       = 4'd8,
        L_UPD        = 4'd9,
        L_UPD_WAIT   = 4'd10,
        L_DONE       = 4'd11;

    reg [3:0]     lstate;
    reg [1:0]     cur_gate;
    reg [CW-1:0]  addr;
    reg [2:0]     wait_cnt;

    integer ii;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            lstate     <= L_IDLE;
            cur_gate   <= 2'd0;
            addr       <= {CW{1'b0}};
            wait_cnt   <= 3'd0;
            ns_pulse   <= 1'b0;
            gid        <= 2'd0;
            feed_data  <= 16'd0;
            feed_valid <= 1'b0;
            sg_pulse   <= 1'b0;
            du_pulse   <= 1'b0;
            step_done  <= 1'b0;
        end else begin
            ns_pulse   <= 1'b0;
            sg_pulse   <= 1'b0;
            du_pulse   <= 1'b0;
            feed_valid <= 1'b0;
            step_done  <= 1'b0;

            case (lstate)
                L_IDLE: begin
                    if (start_step)
                        lstate <= L_SNAP;
                end

                // snapshot concat = [x_in, h_prev]
                L_SNAP: begin
                    for (ii = 0; ii < INPUT_SIZE; ii = ii + 1)
                        concat[ii] <= $signed(x_in_flat[ii*16 +: 16]);
                    for (ii = 0; ii < HIDDEN_SIZE; ii = ii + 1)
                        concat[INPUT_SIZE + ii] <= h_wire[ii];
                    cur_gate <= 2'd0;
                    lstate   <= L_GATE_START;
                end

                L_GATE_START: begin
                    gid      <= cur_gate;
                    ns_pulse <= 1'b1;
                    addr     <= {CW{1'b0}};
                    lstate   <= L_WAIT1;
                end

                L_WAIT1: begin
                    lstate <= L_FEED;
                end

                L_FEED: begin
                    feed_valid <= 1'b1;
                    feed_data  <= concat[addr];
                    if (addr == CS[CW-1:0] - {{(CW-1){1'b0}}, 1'b1})
                        lstate <= L_DRAIN;
                    else
                        addr <= addr + {{(CW-1){1'b0}}, 1'b1};
                end

                L_DRAIN: begin
                    lstate <= L_WAIT2;
                end

                L_WAIT2: begin
                    if (any_done)
                        lstate <= L_STORE;
                end

                L_STORE: begin
                    gid      <= cur_gate;
                    sg_pulse <= 1'b1;
                    lstate   <= L_NEXT;
                end

                L_NEXT: begin
                    if (cur_gate == 2'd3) begin
                        lstate <= L_UPD;
                    end else begin
                        cur_gate <= cur_gate + 2'd1;
                        lstate   <= L_GATE_START;
                    end
                end

                L_UPD: begin
                    du_pulse <= 1'b1;
                    wait_cnt <= 3'd0;
                    lstate   <= L_UPD_WAIT;
                end

                L_UPD_WAIT: begin
                    if (wait_cnt == 3'd4)
                        lstate <= L_DONE;
                    else
                        wait_cnt <= wait_cnt + 3'd1;
                end

                L_DONE: begin
                    step_done <= 1'b1;
                    lstate    <= L_IDLE;
                end

                default: lstate <= L_IDLE;
            endcase
        end
    end
endmodule

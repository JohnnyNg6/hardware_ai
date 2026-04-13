// ============================================================
// dense_layer.v — OUTPUT_SIZE parallel neurons, linear output
//
// Contains an internal streaming controller that feeds x_in
// elements to all neurons at the correct pipeline phase.
//
// Interface:
//   start  : 1-clk pulse — begins computation
//   x_in   : all INPUT_SIZE values must be stable at start
//   scores : available when done=1
//   done   : 1-clk pulse, INPUT_SIZE + 4 clocks after start
//
// Timing:
//   Cycle 0: start → neurons enter S_BIAS_WAIT
//   Cycle 1: internal wait (neurons loading bias)
//   Cycle 2: first din output (neurons in S_BIAS_LOAD, register din)
//   Cycle 3: first MAC in neurons
//   ...
//   Cycle N+2: last MAC
//   Cycle N+3: done
// ============================================================
`timescale 1ns/1ps
module dense_layer #(
    parameter integer INPUT_SIZE  = 128,
    parameter integer OUTPUT_SIZE = 63
)(
    input  wire               clk,
    input  wire               rst_n,
    input  wire               start,
    input  wire signed [15:0] x_in   [0:INPUT_SIZE-1],
    output wire signed [15:0] scores [0:OUTPUT_SIZE-1],
    output wire               done
);
    // ── internal streaming controller ──
    localparam AW = $clog2(INPUT_SIZE > 1 ? INPUT_SIZE : 2);

    localparam [1:0] D_IDLE = 2'd0,
                     D_WAIT = 2'd1,   // 1 cycle for neuron bias BRAM
                     D_FEED = 2'd2,
                     D_TAIL = 2'd3;   // wait for done

    reg [1:0]     dstate;
    reg [AW-1:0]  addr;
    reg           dv;
    reg signed [15:0] dd;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dstate <= D_IDLE;
            addr   <= {AW{1'b0}};
            dv     <= 1'b0;
            dd     <= 16'd0;
        end else begin
            dv <= 1'b0;      // default
            case (dstate)
                D_IDLE: begin
                    if (start) begin
                        addr   <= {AW{1'b0}};
                        dstate <= D_FEED;
                    end
                end
                D_FEED: begin
                    dv <= 1'b1;
                    dd <= x_in[addr];
                    if (addr == INPUT_SIZE[AW-1:0] - {{(AW-1){1'b0}}, 1'b1})
                        dstate <= D_TAIL;
                    else
                        addr <= addr + {{(AW-1){1'b0}}, 1'b1};
                end
                D_TAIL: begin
                    dstate <= D_IDLE;
                end
                default: dstate <= D_IDLE;
            endcase
        end
    end

    // ── parallel neuron instances ──
    wire [OUTPUT_SIZE-1:0] done_vec;
    localparam BA_W = $clog2((INPUT_SIZE + 1) > 1 ? (INPUT_SIZE + 1) : 2);

    genvar gi;
    generate
        for (gi = 0; gi < OUTPUT_SIZE; gi = gi + 1) begin : gn
            neuron #(
                .NUM_INPUTS  (INPUT_SIZE),
                .MEM_DEPTH   (INPUT_SIZE + 1),
                .WEIGHT_FILE ($sformatf("weights/dense_u%0d.mem", gi))
            ) u_n (
                .clk       (clk),
                .rst_n     (rst_n),
                .start     (start),
                .base_addr ({BA_W{1'b0}}),
                .din       (dd),
                .din_valid (dv),
                .dout      (scores[gi]),
                .done      (done_vec[gi])
            );
        end
    endgenerate

    assign done = done_vec[0];
endmodule

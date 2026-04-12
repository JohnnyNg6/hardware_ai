// ============================================================
// dense_layer.v — OUTPUT_SIZE parallel neurons, linear output
// ============================================================
`timescale 1ns/1ps
module dense_layer #(
    parameter integer INPUT_SIZE  = 128,
    parameter integer OUTPUT_SIZE = 63
)(
    input  wire               clk,
    input  wire               rst_n,
    input  wire               start,
    input  wire signed [15:0] din,
    input  wire               din_valid,
    output wire signed [15:0] scores [0:OUTPUT_SIZE-1],
    output wire               done
);
    wire [OUTPUT_SIZE-1:0] done_vec;

    genvar gi;
    generate
        for (gi = 0; gi < OUTPUT_SIZE; gi = gi + 1) begin : gn
            neuron #(
                .NUM_INPUTS  (INPUT_SIZE),
                .WEIGHT_FILE ($sformatf("weights/dense_u%0d.mem", gi))
            ) u_n (
                .clk       (clk),
                .rst_n     (rst_n),
                .start     (start),
                .din       (din),
                .din_valid (din_valid),
                .dout      (scores[gi]),
                .done      (done_vec[gi])
            );
        end
    endgenerate

    assign done = done_vec[0];          // all finish on same cycle
endmodule

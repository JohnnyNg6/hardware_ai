// ============================================================================
// dense_layer.v — Dense (fully-connected) output layer using neuron.v
//
// Instantiates NUM_CLASSES neurons in parallel.  Every neuron receives the
// same serial feature stream (din / din_valid) and computes its own weighted
// sum internally.  After all neurons finish, a combinational argmax selects
// the winning class.
//
// Weight files : dense_n0.mem … dense_n9.mem
//   Each file  : w[0] = bias, w[1..NUM_INPUTS] = weights   (Q8.8 hex)
// ============================================================================
`timescale 1ns / 1ps

module dense_layer #(
    parameter NUM_INPUTS  = 576,        // flattened feature count
    parameter NUM_CLASSES = 10
)(
    input  wire                clk,
    input  wire                rst_n,
    input  wire                start,       // 1-clk pulse → begin
    input  wire signed [15:0]  din,         // serial Q8.8 feature
    input  wire                din_valid,
    output reg         [3:0]   class_out,   // winning class index
    output reg                 done         // 1-clk pulse → result ready
);

    // ----------------------------------------------------------------
    //  Neuron output / done buses
    // ----------------------------------------------------------------
    wire signed [15:0] n_dout [0:NUM_CLASSES-1];
    wire [NUM_CLASSES-1:0] n_done;

    // ----------------------------------------------------------------
    //  10 parallel neurons – output layer, so relu_en = 0
    // ----------------------------------------------------------------
    neuron #(.NUM_INPUTS(NUM_INPUTS), .WEIGHT_FILE("dense_n0.mem")) u_n0 (
        .clk(clk), .rst_n(rst_n), .start(start),
        .din(din), .din_valid(din_valid),
        .dout(n_dout[0]), .done(n_done[0]), .relu_en(1'b0)
    );
    neuron #(.NUM_INPUTS(NUM_INPUTS), .WEIGHT_FILE("dense_n1.mem")) u_n1 (
        .clk(clk), .rst_n(rst_n), .start(start),
        .din(din), .din_valid(din_valid),
        .dout(n_dout[1]), .done(n_done[1]), .relu_en(1'b0)
    );
    neuron #(.NUM_INPUTS(NUM_INPUTS), .WEIGHT_FILE("dense_n2.mem")) u_n2 (
        .clk(clk), .rst_n(rst_n), .start(start),
        .din(din), .din_valid(din_valid),
        .dout(n_dout[2]), .done(n_done[2]), .relu_en(1'b0)
    );
    neuron #(.NUM_INPUTS(NUM_INPUTS), .WEIGHT_FILE("dense_n3.mem")) u_n3 (
        .clk(clk), .rst_n(rst_n), .start(start),
        .din(din), .din_valid(din_valid),
        .dout(n_dout[3]), .done(n_done[3]), .relu_en(1'b0)
    );
    neuron #(.NUM_INPUTS(NUM_INPUTS), .WEIGHT_FILE("dense_n4.mem")) u_n4 (
        .clk(clk), .rst_n(rst_n), .start(start),
        .din(din), .din_valid(din_valid),
        .dout(n_dout[4]), .done(n_done[4]), .relu_en(1'b0)
    );
    neuron #(.NUM_INPUTS(NUM_INPUTS), .WEIGHT_FILE("dense_n5.mem")) u_n5 (
        .clk(clk), .rst_n(rst_n), .start(start),
        .din(din), .din_valid(din_valid),
        .dout(n_dout[5]), .done(n_done[5]), .relu_en(1'b0)
    );
    neuron #(.NUM_INPUTS(NUM_INPUTS), .WEIGHT_FILE("dense_n6.mem")) u_n6 (
        .clk(clk), .rst_n(rst_n), .start(start),
        .din(din), .din_valid(din_valid),
        .dout(n_dout[6]), .done(n_done[6]), .relu_en(1'b0)
    );
    neuron #(.NUM_INPUTS(NUM_INPUTS), .WEIGHT_FILE("dense_n7.mem")) u_n7 (
        .clk(clk), .rst_n(rst_n), .start(start),
        .din(din), .din_valid(din_valid),
        .dout(n_dout[7]), .done(n_done[7]), .relu_en(1'b0)
    );
    neuron #(.NUM_INPUTS(NUM_INPUTS), .WEIGHT_FILE("dense_n8.mem")) u_n8 (
        .clk(clk), .rst_n(rst_n), .start(start),
        .din(din), .din_valid(din_valid),
        .dout(n_dout[8]), .done(n_done[8]), .relu_en(1'b0)
    );
    neuron #(.NUM_INPUTS(NUM_INPUTS), .WEIGHT_FILE("dense_n9.mem")) u_n9 (
        .clk(clk), .rst_n(rst_n), .start(start),
        .din(din), .din_valid(din_valid),
        .dout(n_dout[9]), .done(n_done[9]), .relu_en(1'b0)
    );

    // ----------------------------------------------------------------
    //  Combinational argmax over all neuron outputs
    // ----------------------------------------------------------------
    reg signed [15:0] max_val;
    reg        [3:0]  max_idx;
    integer k;

    always @(*) begin
        max_val = n_dout[0];
        max_idx = 4'd0;
        for (k = 1; k < NUM_CLASSES; k = k + 1) begin
            if (n_dout[k] > max_val) begin
                max_val = n_dout[k];
                max_idx = k[3:0];
            end
        end
    end

    // ----------------------------------------------------------------
    //  Latch result when all neurons are done
    //  (all share the same NUM_INPUTS so they finish on the same cycle)
    // ----------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            class_out <= 4'd0;
            done      <= 1'b0;
        end else begin
            done <= 1'b0;
            if (n_done[0]) begin
                class_out <= max_idx;
                done      <= 1'b1;
            end
        end
    end

endmodule

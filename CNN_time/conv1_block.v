`timescale 1ns / 1ps
module conv1_block (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    output wire        done,
    output wire [15:0] in_addr,
    input  wire signed [15:0] in_data,
    output wire [15:0] out_addr,
    output reg  signed [15:0] out_data,
    output wire        out_we
);

    localparam KSZ = 5*5*3;  // 75
    localparam NF  = 64;

    // Neuron control signals from conv_neuron_layer
    wire               n_start, n_din_valid;
    wire signed [15:0] n_din;
    wire signed [15:0] n_dout [0:NF-1];
    wire [NF-1:0]      n_done;

    // ---- 64 Neuron instances ----
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n0.mem" )) u0  (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[ 0]),.done(n_done[ 0]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n1.mem" )) u1  (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[ 1]),.done(n_done[ 1]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n2.mem" )) u2  (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[ 2]),.done(n_done[ 2]),.relu_en(1'b1));
    // ... (neurons 3–62 follow the same pattern) ...
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n63.mem")) u63 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[63]),.done(n_done[63]),.relu_en(1'b1));

    // ---- Write-index MUX: select neuron output during write phase ----
    wire [7:0] wr_idx;

    // conv_neuron_layer outputs wr_idx via the wr counter
    // We need to expose it — simplest: wire out_data from a MUX
    always @(*) out_data = n_dout[wr_idx];

    // ---- Convolution controller ----
    conv_neuron_layer #(
        .IH(32),.IW(32),.IC(3),.FH(5),.FW(5),.NF(64),
        .ST(2),.PT(1),.PL(1),.OH(16),.OW(16),.KSZ(75)
    ) u_ctrl (
        .clk(clk),.rst_n(rst_n),.start(start),.done(done),
        .in_addr(in_addr),.in_data(in_data),
        .out_addr(out_addr),.out_data(),.out_we(out_we),
        .n_start(n_start),.n_din(n_din),.n_din_valid(n_din_valid),
        .n_dout_0(n_dout[0]),.n_done_0(n_done[0])
    );

    // Expose wr_idx from controller (add output port to conv_neuron_layer)
    assign wr_idx = u_ctrl.wr_idx;

endmodule

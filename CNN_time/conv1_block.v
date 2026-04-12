`timescale 1ns / 1ps
module conv1_block (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire        stall,
    output wire        done,
    output wire [15:0] in_addr,
    input  wire signed [15:0] in_data,
    output wire [15:0] out_addr,
    output reg  signed [15:0] out_data,
    output wire        out_we,
    output wire [7:0]  cur_oh,
    output wire        row_done
);
    localparam KSZ = 75;
    localparam NF  = 64;

    wire               n_start, n_din_valid;
    wire signed [15:0] n_din;
    wire signed [15:0] n_dout [0:NF-1];
    wire [NF-1:0]      n_done;

    // -------- 64 parallel neuron instances --------
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n0.mem"))  u0  (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[ 0]),.done(n_done[ 0]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n1.mem"))  u1  (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[ 1]),.done(n_done[ 1]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n2.mem"))  u2  (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[ 2]),.done(n_done[ 2]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n3.mem"))  u3  (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[ 3]),.done(n_done[ 3]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n4.mem"))  u4  (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[ 4]),.done(n_done[ 4]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n5.mem"))  u5  (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[ 5]),.done(n_done[ 5]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n6.mem"))  u6  (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[ 6]),.done(n_done[ 6]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n7.mem"))  u7  (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[ 7]),.done(n_done[ 7]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n8.mem"))  u8  (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[ 8]),.done(n_done[ 8]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n9.mem"))  u9  (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[ 9]),.done(n_done[ 9]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n10.mem")) u10 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[10]),.done(n_done[10]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n11.mem")) u11 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[11]),.done(n_done[11]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n12.mem")) u12 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[12]),.done(n_done[12]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n13.mem")) u13 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[13]),.done(n_done[13]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n14.mem")) u14 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[14]),.done(n_done[14]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n15.mem")) u15 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[15]),.done(n_done[15]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n16.mem")) u16 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[16]),.done(n_done[16]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n17.mem")) u17 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[17]),.done(n_done[17]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n18.mem")) u18 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[18]),.done(n_done[18]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n19.mem")) u19 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[19]),.done(n_done[19]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n20.mem")) u20 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[20]),.done(n_done[20]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n21.mem")) u21 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[21]),.done(n_done[21]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n22.mem")) u22 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[22]),.done(n_done[22]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n23.mem")) u23 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[23]),.done(n_done[23]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n24.mem")) u24 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[24]),.done(n_done[24]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n25.mem")) u25 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[25]),.done(n_done[25]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n26.mem")) u26 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[26]),.done(n_done[26]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n27.mem")) u27 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[27]),.done(n_done[27]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n28.mem")) u28 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[28]),.done(n_done[28]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n29.mem")) u29 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[29]),.done(n_done[29]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n30.mem")) u30 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[30]),.done(n_done[30]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n31.mem")) u31 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[31]),.done(n_done[31]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n32.mem")) u32 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[32]),.done(n_done[32]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n33.mem")) u33 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[33]),.done(n_done[33]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n34.mem")) u34 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[34]),.done(n_done[34]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n35.mem")) u35 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[35]),.done(n_done[35]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n36.mem")) u36 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[36]),.done(n_done[36]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n37.mem")) u37 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[37]),.done(n_done[37]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n38.mem")) u38 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[38]),.done(n_done[38]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n39.mem")) u39 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[39]),.done(n_done[39]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n40.mem")) u40 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[40]),.done(n_done[40]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n41.mem")) u41 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[41]),.done(n_done[41]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n42.mem")) u42 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[42]),.done(n_done[42]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n43.mem")) u43 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[43]),.done(n_done[43]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n44.mem")) u44 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[44]),.done(n_done[44]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n45.mem")) u45 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[45]),.done(n_done[45]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n46.mem")) u46 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[46]),.done(n_done[46]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n47.mem")) u47 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[47]),.done(n_done[47]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n48.mem")) u48 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[48]),.done(n_done[48]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n49.mem")) u49 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[49]),.done(n_done[49]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n50.mem")) u50 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[50]),.done(n_done[50]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n51.mem")) u51 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[51]),.done(n_done[51]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n52.mem")) u52 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[52]),.done(n_done[52]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n53.mem")) u53 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[53]),.done(n_done[53]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n54.mem")) u54 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[54]),.done(n_done[54]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n55.mem")) u55 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[55]),.done(n_done[55]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n56.mem")) u56 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[56]),.done(n_done[56]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n57.mem")) u57 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[57]),.done(n_done[57]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n58.mem")) u58 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[58]),.done(n_done[58]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n59.mem")) u59 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[59]),.done(n_done[59]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n60.mem")) u60 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[60]),.done(n_done[60]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n61.mem")) u61 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[61]),.done(n_done[61]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n62.mem")) u62 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[62]),.done(n_done[62]),.relu_en(1'b1));
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("conv1_n63.mem")) u63 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[63]),.done(n_done[63]),.relu_en(1'b1));

    // -------- Output MUX (combinational) --------
    wire [7:0] wr_idx;
    integer m;
    always @(*) begin
        out_data = n_dout[0];
        for (m = 1; m < NF; m = m + 1)
            if (wr_idx == m[7:0]) out_data = n_dout[m];
    end

    // -------- Shared controller --------
    conv_neuron_layer #(
        .IH(32),.IW(32),.IC(3),.FH(5),.FW(5),.NF(64),
        .ST(2),.PT(1),.PL(1),.OH(16),.OW(16),.KSZ(KSZ)
    ) u_ctrl (
        .clk(clk),.rst_n(rst_n),.start(start),.done(done),
        .stall(stall),
        .in_addr(in_addr),.in_data(in_data),
        .out_addr(out_addr),.out_we(out_we),
        .n_start(n_start),.n_din(n_din),.n_din_valid(n_din_valid),
        .n_done_0(n_done[0]),
        .wr_idx(wr_idx),.cur_oh(cur_oh),.row_done(row_done)
    );
endmodule

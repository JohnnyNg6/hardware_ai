#!/usr/bin/env python3
"""Generate conv1_block.v and conv2_block.v with 64 neuron instances each."""

TEMPLATE_TOP = """\
`timescale 1ns / 1ps
module {mod_name} (
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
    localparam KSZ = {ksz};
    localparam NF  = {nf};

    wire               n_start, n_din_valid;
    wire signed [15:0] n_din;
    wire signed [15:0] n_dout [0:NF-1];
    wire [NF-1:0]      n_done;

"""

TEMPLATE_NEURON = """\
    neuron #(.NUM_INPUTS(KSZ),.WEIGHT_FILE("{prefix}{i}.mem"))
        u{i} (.clk(clk),.rst_n(rst_n),.start(n_start),
              .din(n_din),.din_valid(n_din_valid),
              .dout(n_dout[{i}]),.done(n_done[{i}]),.relu_en(1'b1));
"""

TEMPLATE_BOT = """\
    // ---- Write MUX (registered to align with out_addr / out_we) ----
    wire [7:0] wr_idx;
    reg signed [15:0] out_data_mux;
    integer m;
    always @(*) begin
        out_data_mux = n_dout[0];
        for (m = 0; m < NF; m = m + 1)
            if (wr_idx == m[7:0]) out_data_mux = n_dout[m];
    end
    always @(posedge clk) out_data <= out_data_mux;

    // ---- Controller ----
    conv_neuron_layer #(
        .IH({ih}),.IW({iw}),.IC({ic}),.FH({fh}),.FW({fw}),.NF({nf}),
        .ST({st}),.PT({pt}),.PL({pl}),.OH({oh}),.OW({ow}),.KSZ(KSZ)
    ) u_ctrl (
        .clk(clk),.rst_n(rst_n),.start(start),.done(done),
        .in_addr(in_addr),.in_data(in_data),
        .out_addr(out_addr),.out_data(),.out_we(out_we),
        .n_start(n_start),.n_din(n_din),.n_din_valid(n_din_valid),
        .n_dout_0(n_dout[0]),.n_done_0(n_done[0]),
        .wr_idx(wr_idx)
    );
endmodule
"""

def gen_conv_block(filename, mod_name, prefix, nf, ksz,
                   ih, iw, ic, fh, fw, st, pt, pl, oh, ow):
    with open(filename, 'w') as f:
        f.write(TEMPLATE_TOP.format(mod_name=mod_name, ksz=ksz, nf=nf))
        for i in range(nf):
            f.write(TEMPLATE_NEURON.format(prefix=prefix, i=i))
        f.write(TEMPLATE_BOT.format(
            ih=ih, iw=iw, ic=ic, fh=fh, fw=fw, nf=nf,
            st=st, pt=pt, pl=pl, oh=oh, ow=ow))
    print(f"Generated {filename}")

if __name__ == '__main__':
    gen_conv_block("conv1_block.v", "conv1_block", "conv1_n",
                   nf=64, ksz=75,
                   ih=32, iw=32, ic=3, fh=5, fw=5,
                   st=2, pt=1, pl=1, oh=16, ow=16)

    gen_conv_block("conv2_block.v", "conv2_block", "conv2_n",
                   nf=64, ksz=576,
                   ih=16, iw=16, ic=64, fh=3, fw=3,
                   st=2, pt=0, pl=0, oh=8, ow=8)

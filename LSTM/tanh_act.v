// ============================================================
// tanh_act.v — Combinational tanh via 512-entry LUT (Q8.8)
// ============================================================
`timescale 1ns/1ps
module tanh_act (
    input  wire signed [15:0] x,
    output wire signed [15:0] y
);
    (* rom_style = "distributed" *)
    reg [15:0] lut [0:511];
    initial $readmemh("weights/tanh_lut.mem", lut);

    wire signed [15:0] cx = (x < -16'sd2048) ? -16'sd2048 :
                            (x >  16'sd2047) ?  16'sd2047 : x;
    wire [15:0] uoff = cx + 16'd2048;
    wire [ 8:0] addr = uoff[11:3];

    assign y = lut[addr];
endmodule

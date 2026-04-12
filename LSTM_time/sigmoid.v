// ============================================================
// sigmoid.v — Combinational sigmoid via 512-entry LUT (Q8.8)
// ============================================================
`timescale 1ns/1ps
module sigmoid (
    input  wire signed [15:0] x,
    output wire signed [15:0] y
);
    (* rom_style = "distributed" *)
    reg [15:0] lut [0:511];
    initial $readmemh("weights/sigmoid_lut.mem", lut);

    // Clamp to [-8,+8) then map to 9-bit address
    wire signed [15:0] cx = (x < -16'sd2048) ? -16'sd2048 :
                            (x >  16'sd2047) ?  16'sd2047 : x;
    wire [15:0] uoff = cx + 16'd2048;           // [0 … 4095]
    wire [ 8:0] addr = uoff[11:3];              // 512 entries

    assign y = lut[addr];
endmodule

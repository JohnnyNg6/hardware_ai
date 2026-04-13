// ============================================================
// argmax.v — Find index of maximum among N signed Q8.8 values
//   Ports flattened for Verilog-2001 compatibility
// ============================================================
`timescale 1ns/1ps
module argmax #(
    parameter integer N = 63
)(
    input  wire [N*16-1:0]        vals_flat,
    output reg  [$clog2(N)-1:0]   idx
);
    integer i;
    reg signed [15:0] best;
    reg signed [15:0] cur;

    always @(*) begin
        best = $signed(vals_flat[15:0]);
        idx  = 0;
        for (i = 1; i < N; i = i + 1) begin
            cur = $signed(vals_flat[i*16 +: 16]);
            if (cur > best) begin
                best = cur;
                idx  = i[$clog2(N)-1:0];
            end
        end
    end
endmodule

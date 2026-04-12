// ============================================================
// argmax.v — Find index of maximum among N signed Q8.8 values
// ============================================================
`timescale 1ns/1ps
module argmax #(
    parameter integer N = 63
)(
    input  wire signed [15:0] vals [0:N-1],
    output reg  [$clog2(N)-1:0] idx
);
    integer i;
    reg signed [15:0] best;
    always @(*) begin
        best = vals[0];
        idx  = 0;
        for (i = 1; i < N; i = i + 1) begin
            if (vals[i] > best) begin
                best = vals[i];
                idx  = i[$clog2(N)-1:0];
            end
        end
    end
endmodule

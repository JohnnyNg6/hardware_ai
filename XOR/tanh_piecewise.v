`timescale 1ns / 1ps
//============================================================================
// Piecewise-linear  tanh  approximation   (Q8.8 → Q8.8)
//
//  Segment          Real formula              Fixed-point
//  ─────────────────────────────────────────────────────────
//  |x| >= 3.0      |y| = 1.0                 256
//  2.0 <= |x| < 3.0  |y| ~ 0.031|x| + 0.902   (8·|x|)>>8 + 231
//  1.0 <= |x| < 2.0  |y| ~ 0.203|x| + 0.559   (52·|x|)>>8 + 143
//  |x| < 1.0       |y| ~ 0.762|x|            (195·|x|)>>8
//
//  tanh is odd, so y = sign(x) · |y|
//============================================================================

module tanh_piecewise (
    input  wire signed [15:0] x,     // Q8.8
    output wire signed [15:0] y      // Q8.8
);

    // magnitude
    wire [15:0] ax = x[15] ? (~x + 16'd1) : x;

    // slope products  (unsigned)
    wire [31:0] pa = ax * 16'd195;     // |x| < 1
    wire [31:0] pb = ax * 16'd52;      // 1 <= |x| < 2
    wire [31:0] pc = ax * 16'd8;       // 2 <= |x| < 3

    // segment selection
    wire [15:0] ay = (ax >= 16'd768) ? 16'd256 :               // >= 3.0
                     (ax >= 16'd512) ? (pc[23:8] + 16'd231) :  // >= 2.0
                     (ax >= 16'd256) ? (pb[23:8] + 16'd143) :  // >= 1.0
                                        pa[23:8];              // < 1.0

    // re-apply sign
    assign y = x[15] ? (16'd0 - ay) : ay;

endmodule

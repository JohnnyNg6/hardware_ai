// ============================================================================
// cifar_cnn.v — CIFAR-10 CNN: Conv1 → Conv2 → Dense → Argmax
// ============================================================================
`timescale 1ns / 1ps

module cifar_cnn (
    input  wire        clk,
    input  wire        rst_n,
    // image write port (from UART loader)
    input  wire [11:0] img_wr_addr,
    input  wire signed [15:0] img_wr_data,
    input  wire        img_wr_en,
    // control
    input  wire        start,
    output reg         done,
    output wire [3:0]  digit_out,
    output wire        result_valid
);

/* ================================================================
   Intermediate buffers  (block-RAM inferred)
   ================================================================ */
// Image buffer 32×32×3 = 3072
(* ram_style = "block" *) reg signed [15:0] img_buf [0:4095];
// Conv1 out  14×14×64 = 12544
(* ram_style = "block" *) reg signed [15:0] c1_buf  [0:16383];
// Conv2 out   7×7×64  = 3136
(* ram_style = "block" *) reg signed [15:0] c2_buf  [0:4095];

/* --- Image buffer write ------------------------------------------------- */
always @(posedge clk) if (img_wr_en) img_buf[img_wr_addr] <= img_wr_data;

/* ================================================================
   Conv1  32×32×3 → 14×14×64   kernel 5×5  stride 2  valid
   ================================================================ */
wire [15:0] c1_ia;
reg  signed [15:0] c1_id;
wire [15:0] c1_oa;
wire signed [15:0] c1_od;
wire        c1_owe, c1_done;
reg         c1_start;

always @(posedge clk) c1_id <= img_buf[c1_ia[11:0]];
always @(posedge clk) if (c1_owe) c1_buf[c1_oa[13:0]] <= c1_od;

conv_layer #(
    .IH(32),.IW(32),.IC(3),.FH(5),.FW(5),.NF(64),
    .ST(2),.PT(0),.PL(0),.OH(14),.OW(14),
    .WFILE("conv1_w.mem"),.BFILE("conv1_b.mem")
) u_c1 (
    .clk(clk),.rst_n(rst_n),.start(c1_start),.done(c1_done),
    .in_addr(c1_ia),.in_data(c1_id),
    .out_addr(c1_oa),.out_data(c1_od),.out_we(c1_owe)
);

/* ================================================================
   Conv2  14×14×64 → 7×7×64   kernel 3×3  stride 2  same
   ================================================================ */
wire [15:0] c2_ia;
reg  signed [15:0] c2_id;
wire [15:0] c2_oa;
wire signed [15:0] c2_od;
wire        c2_owe, c2_done;
reg         c2_start;

always @(posedge clk) c2_id <= c1_buf[c2_ia[13:0]];
always @(posedge clk) if (c2_owe) c2_buf[c2_oa[11:0]] <= c2_od;

conv_layer #(
    .IH(14),.IW(14),.IC(64),.FH(3),.FW(3),.NF(64),
    .ST(2),.PT(0),.PL(0),.OH(7),.OW(7),
    .WFILE("conv2_w.mem"),.BFILE("conv2_b.mem")
) u_c2 (
    .clk(clk),.rst_n(rst_n),.start(c2_start),.done(c2_done),
    .in_addr(c2_ia),.in_data(c2_id),
    .out_addr(c2_oa),.out_data(c2_od),.out_we(c2_owe)
);

/* ================================================================
   Dense  3136 → 10  (linear, argmax replaces softmax)
   ================================================================ */
wire [15:0] d_ia;
reg  signed [15:0] d_id;
wire        d_done;
reg         d_start;

always @(posedge clk) d_id <= c2_buf[d_ia[11:0]];

dense_layer #(
    .NI(3136),
    .NO(10)
) u_d (
    .clk(clk),.rst_n(rst_n),
    .start(d_start),.done(d_done),
    .in_addr(d_ia),.in_data(d_id),
    .digit(digit_out),.result_valid(result_valid)
);

/* ================================================================
   Sequencing FSM:  Conv1 → Conv2 → Dense
   ================================================================ */
localparam [1:0] F_IDLE=0, F_C1=1, F_C2=2, F_DN=3;
reg [1:0] fst;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        fst<=F_IDLE; c1_start<=0; c2_start<=0; d_start<=0; done<=0;
    end else begin
        c1_start<=0; c2_start<=0; d_start<=0; done<=0;
        case (fst)
        F_IDLE: if (start) begin c1_start<=1; fst<=F_C1; end
        F_C1:   if (c1_done) begin c2_start<=1; fst<=F_C2; end
        F_C2:   if (c2_done) begin d_start<=1;  fst<=F_DN; end
        F_DN:   if (d_done)  begin done<=1;      fst<=F_IDLE; end
        endcase
    end
end
endmodule

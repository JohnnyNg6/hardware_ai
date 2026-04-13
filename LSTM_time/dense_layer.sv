// ============================================================
// dense_layer.v — OUTPUT_SIZE parallel neurons, linear output
//   Ports flattened for Verilog-2001 compatibility
// ============================================================
`timescale 1ns/1ps
module dense_layer #(
    parameter integer INPUT_SIZE  = 128,
    parameter integer OUTPUT_SIZE = 63
)(
    input  wire                          clk,
    input  wire                          rst_n,
    input  wire                          start,
    input  wire [INPUT_SIZE*16-1:0]      x_in_flat,
    output wire [OUTPUT_SIZE*16-1:0]     scores_flat,
    output wire                          done
);
    localparam AW = $clog2(INPUT_SIZE > 1 ? INPUT_SIZE : 2);

    localparam [1:0] D_IDLE = 2'd0,
                     D_WAIT = 2'd1,
                     D_FEED = 2'd2,
                     D_TAIL = 2'd3;

    reg [1:0]         dstate;
    reg [AW-1:0]      addr;
    reg                dv;
    reg signed [15:0]  dd;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dstate <= D_IDLE;
            addr   <= {AW{1'b0}};
            dv     <= 1'b0;
            dd     <= 16'd0;
        end else begin
            dv <= 1'b0;
            case (dstate)
                D_IDLE: begin
                    if (start) begin
                        addr   <= {AW{1'b0}};
                        dstate <= D_FEED;
                    end
                end
                D_FEED: begin
                    dv <= 1'b1;
                    dd <= $signed(x_in_flat[addr*16 +: 16]);
                    if (addr == INPUT_SIZE[AW-1:0] - {{(AW-1){1'b0}}, 1'b1})
                        dstate <= D_TAIL;
                    else
                        addr <= addr + {{(AW-1){1'b0}}, 1'b1};
                end
                D_TAIL: begin
                    dstate <= D_IDLE;
                end
                default: dstate <= D_IDLE;
            endcase
        end
    end

    // ── parallel neuron instances ──
    wire [OUTPUT_SIZE-1:0] done_vec;
    localparam BA_W = $clog2((INPUT_SIZE + 1) > 1 ? (INPUT_SIZE + 1) : 2);

    genvar gi;
    generate
        for (gi = 0; gi < OUTPUT_SIZE; gi = gi + 1) begin : gn
            neuron #(
                .NUM_INPUTS  (INPUT_SIZE),
                .MEM_DEPTH   (INPUT_SIZE + 1),
                .WEIGHT_FILE ($sformatf("weights/dense_u%0d.mem", gi))
            ) u_n (
                .clk       (clk),
                .rst_n     (rst_n),
                .start     (start),
                .base_addr ({BA_W{1'b0}}),
                .din       (dd),
                .din_valid (dv),
                .dout      (scores_flat[gi*16 +: 16]),
                .done      (done_vec[gi])
            );
        end
    endgenerate

    assign done = done_vec[0];
endmodule

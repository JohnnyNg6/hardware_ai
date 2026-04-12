`timescale 1ns / 1ps
module dense_layer #(
    parameter integer NI = 4096,
    parameter integer NO = 10
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    output reg         done,
    output reg  [15:0] in_addr,
    input  wire signed [15:0] in_data,
    output reg  [3:0]  digit,
    output reg         result_valid
);

    localparam [2:0] S_IDLE=0, S_START=1, S_ADDR=2, S_FEED=3,
                     S_WAIT=4, S_DONE=5;
    reg [2:0] st;
    localparam CW = $clog2(NI > 1 ? NI : 2);
    reg [CW-1:0] cnt;

    reg                n_start, n_din_valid;
    reg signed [15:0]  n_din;

    wire signed [15:0] n_dout [0:NO-1];
    wire [NO-1:0]      n_done;

    neuron #(.NUM_INPUTS(NI),.WEIGHT_FILE("dense_n0.mem")) u0 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[0]),.done(n_done[0]),.relu_en(1'b0));
    neuron #(.NUM_INPUTS(NI),.WEIGHT_FILE("dense_n1.mem")) u1 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[1]),.done(n_done[1]),.relu_en(1'b0));
    neuron #(.NUM_INPUTS(NI),.WEIGHT_FILE("dense_n2.mem")) u2 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[2]),.done(n_done[2]),.relu_en(1'b0));
    neuron #(.NUM_INPUTS(NI),.WEIGHT_FILE("dense_n3.mem")) u3 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[3]),.done(n_done[3]),.relu_en(1'b0));
    neuron #(.NUM_INPUTS(NI),.WEIGHT_FILE("dense_n4.mem")) u4 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[4]),.done(n_done[4]),.relu_en(1'b0));
    neuron #(.NUM_INPUTS(NI),.WEIGHT_FILE("dense_n5.mem")) u5 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[5]),.done(n_done[5]),.relu_en(1'b0));
    neuron #(.NUM_INPUTS(NI),.WEIGHT_FILE("dense_n6.mem")) u6 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[6]),.done(n_done[6]),.relu_en(1'b0));
    neuron #(.NUM_INPUTS(NI),.WEIGHT_FILE("dense_n7.mem")) u7 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[7]),.done(n_done[7]),.relu_en(1'b0));
    neuron #(.NUM_INPUTS(NI),.WEIGHT_FILE("dense_n8.mem")) u8 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[8]),.done(n_done[8]),.relu_en(1'b0));
    neuron #(.NUM_INPUTS(NI),.WEIGHT_FILE("dense_n9.mem")) u9 (.clk(clk),.rst_n(rst_n),.start(n_start),.din(n_din),.din_valid(n_din_valid),.dout(n_dout[9]),.done(n_done[9]),.relu_en(1'b0));

    // ---- Argmax ----
    reg signed [15:0] max_val;
    reg [3:0] max_idx;
    integer k;
    always @(*) begin
        max_val = n_dout[0]; max_idx = 0;
        for (k=1; k<NO; k=k+1)
            if (n_dout[k] > max_val) begin max_val=n_dout[k]; max_idx=k[3:0]; end
    end

    // ---- FSM (2-cycle-per-element loop) ----
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st<=S_IDLE; cnt<=0; in_addr<=0;
            n_start<=0; n_din_valid<=0; n_din<=0;
            digit<=0; result_valid<=0; done<=0;
        end else begin
            n_start<=0; n_din_valid<=0; result_valid<=0; done<=0;
            case (st)
            S_IDLE: if (start) st<=S_START;
            S_START: begin
                n_start<=1;
                in_addr<=0;
                cnt<=0;
                st<=S_ADDR;
            end
            S_ADDR: begin
                st<=S_FEED;            // 1-cycle BRAM latency wait
            end
            S_FEED: begin
                n_din       <= in_data;
                n_din_valid <= 1'b1;
                if (cnt == NI[CW-1:0]-1)
                    st <= S_WAIT;
                else begin
                    cnt     <= cnt + 1;
                    in_addr <= in_addr + 1;
                    st      <= S_ADDR;  // back to wait state
                end
            end
            S_WAIT: if (n_done[0]) begin
                digit<=max_idx; result_valid<=1; st<=S_DONE;
            end
            S_DONE: begin done<=1; st<=S_IDLE; end
            default: st<=S_IDLE;
            endcase
        end
    end
endmodule

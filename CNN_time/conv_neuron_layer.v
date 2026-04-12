`timescale 1ns / 1ps
module conv_neuron_layer #(
    parameter integer IH  = 32,  parameter integer IW  = 32,
    parameter integer IC  = 3,
    parameter integer FH  = 5,   parameter integer FW  = 5,
    parameter integer NF  = 64,
    parameter integer ST  = 2,
    parameter integer PT  = 1,
    parameter integer PL  = 1,
    parameter integer OH  = 16,  parameter integer OW  = 16,
    parameter integer KSZ = FH * FW * IC
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    output reg         done,
    output reg  [15:0] in_addr,
    input  wire signed [15:0] in_data,
    output reg  [15:0] out_addr,
    output reg  signed [15:0] out_data,
    output reg         out_we,
    output reg                n_start,
    output reg  signed [15:0] n_din,
    output reg                n_din_valid,
    input  wire signed [15:0] n_dout_0,
    input  wire               n_done_0,
    output reg  [7:0]  wr_idx          // ← NEW output port
);

localparam integer IW_C = IW * IC;

localparam [3:0] S_IDLE=0, S_NSTART=1, S_ADDR=2, S_PIPE=3,
                 S_FEED=4, S_NWAIT=5, S_WRSET=6, S_WR=7, S_DONE=8;
reg [3:0] st;

reg [7:0] oh_r, ow_r;
reg [3:0] kh_r, kw_r;
reg [7:0] kc_r;
reg [15:0] k_cnt;

wire signed [9:0] ih   = $signed({1'b0, oh_r}) * ST[9:0]
                        + $signed({1'b0, kh_r}) - PT[9:0];
wire signed [9:0] iw_s = $signed({1'b0, ow_r}) * ST[9:0]
                        + $signed({1'b0, kw_r}) - PL[9:0];
wire pad_zero = (ih < 0) | (ih >= IH) | (iw_s < 0) | (iw_s >= IW);
wire [15:0] ia = ih[7:0] * IW_C[15:0] + iw_s[7:0] * IC[15:0] + {8'd0, kc_r};

wire kc_last = (kc_r == IC - 1);
wire kw_last = (kw_r == FW - 1) && kc_last;
wire kh_last = (kh_r == FH - 1) && kw_last;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        st<=S_IDLE; done<=0; out_we<=0;
        oh_r<=0; ow_r<=0; kh_r<=0; kw_r<=0; kc_r<=0; k_cnt<=0;
        wr_idx<=0; in_addr<=0; out_addr<=0; out_data<=0;
        n_start<=0; n_din<=0; n_din_valid<=0;
    end else begin
        done<=0; out_we<=0; n_start<=0; n_din_valid<=0;
        case (st)

        S_IDLE: if (start) begin
            oh_r<=0; ow_r<=0;
            st<=S_NSTART;
        end

        S_NSTART: begin
            n_start <= 1'b1;
            kh_r<=0; kw_r<=0; kc_r<=0; k_cnt<=0;
            st <= S_ADDR;
        end

        S_ADDR: begin
            in_addr <= pad_zero ? 16'd0 : ia;
            st <= S_PIPE;
        end

        S_PIPE: begin
            st <= S_FEED;
        end

        S_FEED: begin
            n_din       <= pad_zero ? 16'sd0 : in_data;
            n_din_valid <= 1'b1;

            if (k_cnt == KSZ[15:0] - 16'd1) begin
                st <= S_NWAIT;
            end else begin
                if (kc_last) begin
                    kc_r <= 0;
                    if (kw_last) begin kw_r<=0; kh_r<=kh_r+1; end
                    else         kw_r <= kw_r + 1;
                end else         kc_r <= kc_r + 1;
                k_cnt <= k_cnt + 1;
                st <= S_ADDR;
            end
        end

        S_NWAIT: begin
            if (n_done_0) begin
                wr_idx <= 0;
                st <= S_WR;
            end
        end

        S_WR: begin
            out_addr <= ({8'd0,oh_r} * OW[15:0] + {8'd0,ow_r}) * NF[15:0] + {8'd0,wr_idx};
            out_we   <= 1'b1;
            if (wr_idx == NF[7:0] - 8'd1) begin
                if (ow_r == OW[7:0] - 8'd1) begin
                    ow_r <= 0;
                    if (oh_r == OH[7:0] - 8'd1)
                        st <= S_DONE;
                    else begin
                        oh_r <= oh_r + 1;
                        st <= S_NSTART;
                    end
                end else begin
                    ow_r <= ow_r + 1;
                    st <= S_NSTART;
                end
            end else
                wr_idx <= wr_idx + 1;
        end

        S_DONE: begin done<=1; st<=S_IDLE; end
        default: st <= S_IDLE;
        endcase
    end
end
endmodule

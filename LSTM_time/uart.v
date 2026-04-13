// ============================================================
// uart.v — UART Receiver + Transmitter (8N1)
// ============================================================
`timescale 1ns / 1ps

module uart_rx #(
    parameter integer CLK_FREQ = 50_000_000,
    parameter integer BAUD     = 115_200
)(
    input  wire       clk,
    input  wire       rst_n,
    input  wire       rx,
    output reg  [7:0] data,
    output reg        valid
);
    localparam integer CLKS_PER_BIT = CLK_FREQ / BAUD;
    localparam integer HALF_BIT     = CLKS_PER_BIT / 2;

    reg rx_s1, rx_s2;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin rx_s1 <= 1'b1; rx_s2 <= 1'b1;
        end else begin    rx_s1 <= rx;    rx_s2 <= rx_s1; end
    end

    localparam [1:0] S_IDLE=2'd0, S_START=2'd1, S_DATA=2'd2, S_STOP=2'd3;
    reg [1:0]  state;
    reg [15:0] clk_cnt;
    reg [2:0]  bit_idx;
    reg [7:0]  shift;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state<=S_IDLE; clk_cnt<=0; bit_idx<=0;
            shift<=0; data<=0; valid<=0;
        end else begin
            valid <= 1'b0;
            case (state)
                S_IDLE: begin
                    clk_cnt<=0; bit_idx<=0;
                    if (rx_s2==1'b0) state<=S_START;
                end
                S_START: begin
                    if (clk_cnt==HALF_BIT[15:0]) begin
                        clk_cnt<=0;
                        state <= (rx_s2==1'b0) ? S_DATA : S_IDLE;
                    end else clk_cnt<=clk_cnt+1;
                end
                S_DATA: begin
                    if (clk_cnt==CLKS_PER_BIT[15:0]-1) begin
                        clk_cnt<=0; shift[bit_idx]<=rx_s2;
                        if (bit_idx==3'd7) state<=S_STOP;
                        else bit_idx<=bit_idx+1;
                    end else clk_cnt<=clk_cnt+1;
                end
                S_STOP: begin
                    if (clk_cnt==CLKS_PER_BIT[15:0]-1) begin
                        clk_cnt<=0; data<=shift; valid<=1; state<=S_IDLE;
                    end else clk_cnt<=clk_cnt+1;
                end
                default: state<=S_IDLE;
            endcase
        end
    end
endmodule

module uart_tx #(
    parameter integer CLK_FREQ = 50_000_000,
    parameter integer BAUD     = 115_200
)(
    input  wire       clk,
    input  wire       rst_n,
    input  wire [7:0] data,
    input  wire       send,
    output reg        tx,
    output reg        busy
);
    localparam integer CLKS_PER_BIT = CLK_FREQ / BAUD;
    localparam [1:0] S_IDLE=2'd0, S_START=2'd1, S_DATA=2'd2, S_STOP=2'd3;
    reg [1:0]  state;
    reg [15:0] clk_cnt;
    reg [2:0]  bit_idx;
    reg [7:0]  shift;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state<=S_IDLE; tx<=1; busy<=0; clk_cnt<=0; bit_idx<=0; shift<=0;
        end else begin
            case (state)
                S_IDLE: begin
                    tx<=1; busy<=0;
                    if (send) begin shift<=data; busy<=1; clk_cnt<=0; state<=S_START; end
                end
                S_START: begin
                    tx<=0;
                    if (clk_cnt==CLKS_PER_BIT[15:0]-1)
                        begin clk_cnt<=0; bit_idx<=0; state<=S_DATA; end
                    else clk_cnt<=clk_cnt+1;
                end
                S_DATA: begin
                    tx<=shift[bit_idx];
                    if (clk_cnt==CLKS_PER_BIT[15:0]-1) begin
                        clk_cnt<=0;
                        if (bit_idx==3'd7) state<=S_STOP;
                        else bit_idx<=bit_idx+1;
                    end else clk_cnt<=clk_cnt+1;
                end
                S_STOP: begin
                    tx<=1;
                    if (clk_cnt==CLKS_PER_BIT[15:0]-1)
                        begin clk_cnt<=0; state<=S_IDLE; end
                    else clk_cnt<=clk_cnt+1;
                end
                default: state<=S_IDLE;
            endcase
        end
    end
endmodule

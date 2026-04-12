// ============================================================================
// cifar_loader.v — Receive CIFAR image via UART, run CNN, return result
//
// Protocol  PC→FPGA : 0xAA  +  3072×2 bytes (big-endian Q8.8)
// Protocol  FPGA→PC : 1 byte = predicted class (0x00–0x09)
// ============================================================================
`timescale 1ns / 1ps

module cifar_loader (
    input  wire        clk,
    input  wire        rst_n,
    // UART RX
    input  wire [7:0]  rx_data,
    input  wire        rx_valid,
    // UART TX
    output reg  [7:0]  tx_data,
    output reg         tx_send,
    input  wire        tx_busy,
    // CNN interface
    output reg  [11:0] img_wr_addr,
    output reg  signed [15:0] img_wr_data,
    output reg         img_wr_en,
    output reg         cnn_start,
    input  wire [3:0]  cnn_digit,
    input  wire        cnn_result_valid,
    input  wire        cnn_done,
    output wire        busy
);

localparam integer NPIX = 3072;   // 32×32×3

localparam [3:0] S_SYNC=0, S_HI=1, S_LO=2,
                 S_GO=3,   S_WAIT=4, S_TX=5, S_TXW=6, S_FIN=7;
reg [3:0] st;
reg [11:0] cnt;
reg [7:0]  hi_r;
reg [3:0]  res_r;

assign busy = (st != S_SYNC);

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        st<=S_SYNC; cnt<=0; hi_r<=0; res_r<=0;
        img_wr_en<=0; cnn_start<=0; tx_send<=0; tx_data<=0;
        img_wr_addr<=0; img_wr_data<=0;
    end else begin
        img_wr_en<=0; cnn_start<=0; tx_send<=0;
        case (st)
        S_SYNC: begin cnt<=0; if (rx_valid && rx_data==8'hAA) st<=S_HI; end
        S_HI:   if (rx_valid) begin hi_r<=rx_data; st<=S_LO; end
        S_LO:   if (rx_valid) begin
                    img_wr_addr <= cnt;
                    img_wr_data <= {hi_r, rx_data};
                    img_wr_en   <= 1;
                    if (cnt==NPIX-1) st<=S_GO;
                    else begin cnt<=cnt+12'd1; st<=S_HI; end
                 end
        S_GO:   begin cnn_start<=1; st<=S_WAIT; end
        S_WAIT: if (cnn_result_valid) begin res_r<=cnn_digit; st<=S_TX; end
        S_TX:   if (!tx_busy) begin tx_data<={4'h0,res_r}; tx_send<=1; st<=S_TXW; end
        S_TXW:  if (tx_busy) st<=S_FIN;
        S_FIN:  if (!tx_busy) st<=S_SYNC;
        default: st<=S_SYNC;
        endcase
    end
end
endmodule

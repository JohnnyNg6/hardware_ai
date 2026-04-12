`timescale 1ns / 1ps
module cifar_top (
    input  wire       clk_50m,      // G22
    input  wire       rst_n,        // KEY1 = D26  (active-low)
    input  wire       uart_rxd,     // B20
    output wire       uart_txd,     // C22
    output wire [7:0] led           // LED1..LED8
);

// --- reset synchroniser (3-stage) ---
reg [2:0] rs;
always @(posedge clk_50m or negedge rst_n)
    if (!rst_n) rs <= 3'b0; else rs <= {rs[1:0], 1'b1};
wire rst_s = rs[2];

// --- UART RX ---
wire [7:0] rx_d;  wire rx_v;
uart_rx #(.CLK_FREQ(50_000_000),.BAUD(115200))
    u_rx (.clk(clk_50m),.rst_n(rst_s),.rx(uart_rxd),
          .data(rx_d),.valid(rx_v));

// --- UART TX ---
wire [7:0] tx_d;  wire tx_s, tx_b;
uart_tx #(.CLK_FREQ(50_000_000),.BAUD(115200))
    u_tx (.clk(clk_50m),.rst_n(rst_s),.data(tx_d),.send(tx_s),
          .tx(uart_txd),.busy(tx_b));

// --- CNN core ---
wire [11:0] iwa;  wire signed [15:0] iwd;  wire iwe;
wire cs;  wire [3:0] cd;  wire crv, cdn;

cifar_cnn u_cnn (
    .clk(clk_50m),.rst_n(rst_s),
    .img_wr_addr(iwa),.img_wr_data(iwd),.img_wr_en(iwe),
    .start(cs),.done(cdn),.digit_out(cd),.result_valid(crv)
);

// --- UART ↔ CNN loader ---
wire ldr_busy;
cifar_loader u_ldr (
    .clk(clk_50m),.rst_n(rst_s),
    .rx_data(rx_d),.rx_valid(rx_v),
    .tx_data(tx_d),.tx_send(tx_s),.tx_busy(tx_b),
    .img_wr_addr(iwa),.img_wr_data(iwd),.img_wr_en(iwe),
    .cnn_start(cs),.cnn_digit(cd),.cnn_result_valid(crv),
    .cnn_done(cdn),.busy(ldr_busy)
);

// --- LED display ---
reg [3:0]  last_d;
reg        has_r;
reg [24:0] hb;
always @(posedge clk_50m or negedge rst_s)
    if (!rst_s) begin last_d<=0; has_r<=0; hb<=0; end
    else begin
        hb <= hb + 1;
        if (crv) begin last_d <= cd; has_r <= 1; end
    end

assign led[0] = last_d[0];
assign led[1] = last_d[1];
assign led[2] = last_d[2];
assign led[3] = last_d[3];
assign led[4] = has_r;
assign led[5] = ldr_busy;
assign led[6] = tx_b;
assign led[7] = hb[24];
endmodule

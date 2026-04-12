`timescale 1ns / 1ps
module conv_neuron_layer #(
    parameter integer IH  = 32,  parameter integer IW  = 32,
    parameter integer IC  = 3,
    parameter integer FH  = 5,   parameter integer FW  = 5,
    parameter integer NF  = 64,
    parameter integer ST  = 2,
    parameter integer PT  = 1,   // pad top   (TF 'same' convention)
    parameter integer PL  = 1,   // pad left
    parameter integer OH  = 16,  parameter integer OW  = 16,
    // Per-neuron weight files: supplied via localparam array in wrapper
    parameter integer KSZ = FH * FW * IC   // kernel volume
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    output reg         done,
    // input buffer read (sync BRAM, 1-clk latency)
    output reg  [15:0] in_addr,
    input  wire signed [15:0] in_data,
    // output buffer write
    output reg  [15:0] out_addr,
    output reg  signed [15:0] out_data,
    output reg         out_we,
    // neuron interface — directly wired to internal neurons
    // (exposed so parent can instantiate neurons externally)
    output reg                n_start,
    output reg  signed [15:0] n_din,
    output reg                n_din_valid,
    input  wire signed [15:0] n_dout_0,   // neuron outputs (active after done)
    input  wire               n_done_0    // any one neuron's done (all finish together)
);

localparam integer IW_C = IW * IC;

// ---- FSM ----
localparam [3:0] S_IDLE=0, S_NSTART=1, S_ADDR=2, S_PIPE=3,
                 S_FEED=4, S_NWAIT=5, S_WRSET=6, S_WR=7, S_DONE=8;
reg [3:0] st;

// ---- position counters ----
reg [7:0] oh_r, ow_r;
// ---- kernel counters ----
reg [3:0] kh_r, kw_r;
reg [7:0] kc_r;
reg [15:0] k_cnt;     // flat kernel index 0..KSZ-1
// ---- write counter ----
reg [7:0]  wr_idx;    // 0..NF-1

// ---- patch element address ----
wire signed [9:0] ih = $signed({1'b0, oh_r}) * ST[9:0] + $signed({1'b0, kh_r}) - PT[9:0];
wire signed [9:0] iw_s = $signed({1'b0, ow_r}) * ST[9:0] + $signed({1'b0, kw_r}) - PL[9:0];
wire pad_zero = (ih < 0) | (ih >= IH) | (iw_s < 0) | (iw_s >= IW);
wire [15:0] ia = ih[7:0] * IW_C[15:0] + iw_s[7:0] * IC[15:0] + {8'd0, kc_r};

// ---- kernel advance (combinational) ----
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

        // ---- pulse neuron start (loads each neuron's bias) ----
        S_NSTART: begin
            n_start <= 1'b1;
            kh_r<=0; kw_r<=0; kc_r<=0; k_cnt<=0;
            st <= S_ADDR;
        end

        // ---- issue first address ----
        S_ADDR: begin
            in_addr <= pad_zero ? 16'd0 : ia;
            st <= S_PIPE;
        end

        // ---- wait 1 clk for BRAM data ----
        S_PIPE: begin
            st <= S_FEED;
        end

        // ---- feed patch elements to all neurons ----
        S_FEED: begin
            n_din       <= pad_zero ? 16'sd0 : in_data;  // note: pad_zero from PREVIOUS addr
            n_din_valid <= 1'b1;

            if (k_cnt == KSZ[15:0] - 16'd1) begin
                // last element fed
                st <= S_NWAIT;
            end else begin
                // advance kernel position
                if (kc_last) begin
                    kc_r <= 0;
                    if (kw_last) begin kw_r<=0; kh_r<=kh_r+1; end
                    else         kw_r <= kw_r + 1;
                end else         kc_r <= kc_r + 1;
                k_cnt <= k_cnt + 1;
                // pre-issue next address (will be read in next S_FEED)
                // Need to compute next ih/iw with UPDATED kh/kw/kc
                // Since we update kh_r/kw_r/kc_r above, the combinational
                // ih/iw will reflect the NEW values on the next cycle
                st <= S_ADDR;  // go back to issue addr & wait 1 cycle
            end
        end

        // ---- wait for neurons to finish ----
        S_NWAIT: begin
            if (n_done_0) begin
                wr_idx <= 0;
                st <= S_WR;
            end
        end

        // ---- write NF neuron outputs to output buffer ----
        // (Parent module must MUX n_dout[wr_idx] into out_data)
        S_WR: begin
            out_addr <= ({8'd0,oh_r} * OW[15:0] + {8'd0,ow_r}) * NF[15:0] + {8'd0,wr_idx};
            out_we   <= 1'b1;
            // out_data is set by parent via wr_idx
            if (wr_idx == NF[7:0] - 8'd1) begin
                // advance output position
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

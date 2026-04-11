// ============================================================================
// mnist_nn.v — MNIST conf5 inference engine
//
// Architecture : 784 → 25 (ReLU) → 10 (linear) → argmax
// Throughput   : ~823 clocks per image
//
// Softmax note : argmax(z_i) ≡ argmax(softmax(z_i)) because exp() is
//                monotonic.  For classification the result is identical,
//                so we skip the expensive exp/div hardware.
// ============================================================================
`timescale 1ns / 1ps

module mnist_nn (
    input  wire                clk,
    input  wire                rst_n,
    // ---- image input (serial) ----
    input  wire                img_start,       // 1-clk pulse
    input  wire signed [15:0]  pixel_in,        // Q8.8 normalised pixel
    input  wire                pixel_valid,      // feed 784 pixels after start
    // ---- result ----
    output reg  [3:0]          digit_out,        // predicted digit 0–9
    output reg                 result_valid      // 1-clk pulse
);

    // ================================================================
    // Constants
    // ================================================================
    localparam NP = 784;          // pixels per image
    localparam NH = 25;           // hidden neurons
    localparam NO = 10;           // output neurons

    // ================================================================
    // FSM
    // ================================================================
    localparam [2:0] S_IDLE    = 3'd0,
                     S_H_RUN   = 3'd1,  // hidden layer running
                     S_O_START = 3'd2,  // kick output layer
                     S_O_FEED  = 3'd3,  // stream hidden→output
                     S_O_WAIT  = 3'd4,  // wait output done
                     S_ARGMAX  = 3'd5,  // find max
                     S_DONE    = 3'd6;
    reg [2:0] state;

    // ================================================================
    // Hidden layer — 25 neurons, 784 inputs, ReLU
    // ================================================================
    wire signed [15:0] h_dout [0:NH-1];
    wire               h_done [0:NH-1];

    genvar hi;
    generate
        for (hi = 0; hi < NH; hi = hi + 1) begin : gen_h
            // Build file name "h_XX.mem" from genvar (Vivado SV)
            localparam [7:0] D1 = "0" + hi / 10;
            localparam [7:0] D0 = "0" + hi % 10;

            neuron #(
                .NUM_INPUTS  (NP),
                .WEIGHT_FILE ({"h_", D1, D0, ".mem"})
            ) u_h (
                .clk       (clk),
                .rst_n     (rst_n),
                .start     (img_start && (state == S_IDLE)),
                .din       (pixel_in),
                .din_valid (pixel_valid && (state == S_H_RUN)),
                .dout      (h_dout[hi]),
                .done      (h_done[hi]),
                .relu_en   (1'b1)                    // ← ReLU
            );
        end
    endgenerate

    // ================================================================
    // Output layer — 10 neurons, 25 inputs, linear (argmax replaces softmax)
    // ================================================================
    wire signed [15:0] o_dout [0:NO-1];
    wire               o_done [0:NO-1];

    reg                o_start_r;
    reg signed [15:0]  o_din_r;
    reg                o_dv_r;

    genvar oi;
    generate
        for (oi = 0; oi < NO; oi = oi + 1) begin : gen_o
            localparam [7:0] D1 = "0" + oi / 10;
            localparam [7:0] D0 = "0" + oi % 10;

            neuron #(
                .NUM_INPUTS  (NH),
                .WEIGHT_FILE ({"o_", D1, D0, ".mem"})
            ) u_o (
                .clk       (clk),
                .rst_n     (rst_n),
                .start     (o_start_r),
                .din       (o_din_r),
                .din_valid (o_dv_r),
                .dout      (o_dout[oi]),
                .done      (o_done[oi]),
                .relu_en   (1'b0)                    // ← linear
            );
        end
    endgenerate

    // ================================================================
    // Sequencing logic
    // ================================================================
    reg [4:0]          feed_cnt;                     // 0–24
    reg [3:0]          arg_cnt;                      // 1–9
    reg signed [15:0]  h_buf [0:NH-1];               // hidden outputs
    reg signed [15:0]  o_buf [0:NO-1];               // output logits
    reg signed [15:0]  max_val;
    reg [3:0]          max_idx;
    integer            i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            feed_cnt     <= 5'd0;
            arg_cnt      <= 4'd0;
            digit_out    <= 4'd0;
            result_valid <= 1'b0;
            o_start_r    <= 1'b0;
            o_dv_r       <= 1'b0;
            o_din_r      <= 16'sd0;
            max_val      <= 16'sd0;
            max_idx      <= 4'd0;
        end else begin
            // defaults (active for 1 clock only)
            result_valid <= 1'b0;
            o_start_r    <= 1'b0;
            o_dv_r       <= 1'b0;

            case (state)
                // ------------------------------------------------
                S_IDLE: begin
                    if (img_start) state <= S_H_RUN;
                end

                // ------------------------------------------------
                S_H_RUN: begin
                    if (h_done[0]) begin              // all finish together
                        for (i = 0; i < NH; i = i + 1)
                            h_buf[i] <= h_dout[i];
                        state <= S_O_START;
                    end
                end

                // ------------------------------------------------
                S_O_START: begin
                    o_start_r <= 1'b1;
                    feed_cnt  <= 5'd0;
                    state     <= S_O_FEED;
                end

                // ------------------------------------------------
                S_O_FEED: begin
                    o_din_r <= h_buf[feed_cnt];
                    o_dv_r  <= 1'b1;
                    if (feed_cnt == NH[4:0] - 5'd1)
                        state <= S_O_WAIT;
                    else
                        feed_cnt <= feed_cnt + 5'd1;
                end

                // ------------------------------------------------
                S_O_WAIT: begin
                    if (o_done[0]) begin
                        for (i = 0; i < NO; i = i + 1)
                            o_buf[i] <= o_dout[i];
                        max_val <= o_dout[0];
                        max_idx <= 4'd0;
                        arg_cnt <= 4'd1;
                        state   <= S_ARGMAX;
                    end
                end

                // ------------------------------------------------
                S_ARGMAX: begin
                    if ($signed(o_buf[arg_cnt]) > $signed(max_val)) begin
                        max_val <= o_buf[arg_cnt];
                        max_idx <= arg_cnt;
                    end
                    if (arg_cnt == NO[3:0] - 4'd1)
                        state <= S_DONE;
                    else
                        arg_cnt <= arg_cnt + 4'd1;
                end

                // ------------------------------------------------
                S_DONE: begin
                    digit_out    <= max_idx;
                    result_valid <= 1'b1;
                    state        <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end
endmodule

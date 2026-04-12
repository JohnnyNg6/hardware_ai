// ============================================================================
// dense_layer.v — Dense output layer using neuron.v, with internal address gen
//
// External interface matches cifar_cnn.v: address-based buffer read.
// Internally instantiates NUM_CLASSES neurons fed in parallel.
//
// Weight files : dense_n0.mem … dense_n9.mem
//   Each file  : w[0] = bias, w[1..NUM_INPUTS] = weights   (Q8.8 hex)
// ============================================================================
`timescale 1ns / 1ps

module dense_layer #(
    parameter integer NI = 3136,       // number of input features
    parameter integer NO = 10          // number of output classes
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,          // 1-clk pulse
    output reg         done,
    // buffer read port (address out, data back next cycle)
    output reg  [15:0] in_addr,
    input  wire signed [15:0] in_data,
    // result
    output reg  [3:0]  digit,
    output reg         result_valid
);

    // ----------------------------------------------------------------
    //  FSM — generates serial address stream, feeds neurons
    // ----------------------------------------------------------------
    localparam [2:0] S_IDLE  = 3'd0,
                     S_START = 3'd1,   // pulse neuron start
                     S_ADDR  = 3'd2,   // issue first address
                     S_PIPE  = 3'd3,   // wait 1 cycle for BRAM latency
                     S_FEED  = 3'd4,   // feed data to neurons
                     S_WAIT  = 3'd5,   // wait for neurons to finish
                     S_DONE  = 3'd6;

    reg [2:0] st;
    localparam CW = $clog2(NI > 1 ? NI : 2);
    reg [CW-1:0] cnt;

    // neuron control signals
    reg                n_start;
    reg                n_din_valid;
    reg signed [15:0]  n_din;

    // ----------------------------------------------------------------
    //  Neuron instances
    // ----------------------------------------------------------------
    wire signed [15:0] n_dout [0:NO-1];
    wire [NO-1:0] n_done;

    neuron #(.NUM_INPUTS(NI), .WEIGHT_FILE("dense_n0.mem")) u_n0 (
        .clk(clk), .rst_n(rst_n), .start(n_start),
        .din(n_din), .din_valid(n_din_valid),
        .dout(n_dout[0]), .done(n_done[0]), .relu_en(1'b0)
    );
    neuron #(.NUM_INPUTS(NI), .WEIGHT_FILE("dense_n1.mem")) u_n1 (
        .clk(clk), .rst_n(rst_n), .start(n_start),
        .din(n_din), .din_valid(n_din_valid),
        .dout(n_dout[1]), .done(n_done[1]), .relu_en(1'b0)
    );
    neuron #(.NUM_INPUTS(NI), .WEIGHT_FILE("dense_n2.mem")) u_n2 (
        .clk(clk), .rst_n(rst_n), .start(n_start),
        .din(n_din), .din_valid(n_din_valid),
        .dout(n_dout[2]), .done(n_done[2]), .relu_en(1'b0)
    );
    neuron #(.NUM_INPUTS(NI), .WEIGHT_FILE("dense_n3.mem")) u_n3 (
        .clk(clk), .rst_n(rst_n), .start(n_start),
        .din(n_din), .din_valid(n_din_valid),
        .dout(n_dout[3]), .done(n_done[3]), .relu_en(1'b0)
    );
    neuron #(.NUM_INPUTS(NI), .WEIGHT_FILE("dense_n4.mem")) u_n4 (
        .clk(clk), .rst_n(rst_n), .start(n_start),
        .din(n_din), .din_valid(n_din_valid),
        .dout(n_dout[4]), .done(n_done[4]), .relu_en(1'b0)
    );
    neuron #(.NUM_INPUTS(NI), .WEIGHT_FILE("dense_n5.mem")) u_n5 (
        .clk(clk), .rst_n(rst_n), .start(n_start),
        .din(n_din), .din_valid(n_din_valid),
        .dout(n_dout[5]), .done(n_done[5]), .relu_en(1'b0)
    );
    neuron #(.NUM_INPUTS(NI), .WEIGHT_FILE("dense_n6.mem")) u_n6 (
        .clk(clk), .rst_n(rst_n), .start(n_start),
        .din(n_din), .din_valid(n_din_valid),
        .dout(n_dout[6]), .done(n_done[6]), .relu_en(1'b0)
    );
    neuron #(.NUM_INPUTS(NI), .WEIGHT_FILE("dense_n7.mem")) u_n7 (
        .clk(clk), .rst_n(rst_n), .start(n_start),
        .din(n_din), .din_valid(n_din_valid),
        .dout(n_dout[7]), .done(n_done[7]), .relu_en(1'b0)
    );
    neuron #(.NUM_INPUTS(NI), .WEIGHT_FILE("dense_n8.mem")) u_n8 (
        .clk(clk), .rst_n(rst_n), .start(n_start),
        .din(n_din), .din_valid(n_din_valid),
        .dout(n_dout[8]), .done(n_done[8]), .relu_en(1'b0)
    );
    neuron #(.NUM_INPUTS(NI), .WEIGHT_FILE("dense_n9.mem")) u_n9 (
        .clk(clk), .rst_n(rst_n), .start(n_start),
        .din(n_din), .din_valid(n_din_valid),
        .dout(n_dout[9]), .done(n_done[9]), .relu_en(1'b0)
    );

    // ----------------------------------------------------------------
    //  Combinational argmax
    // ----------------------------------------------------------------
    reg signed [15:0] max_val;
    reg        [3:0]  max_idx;
    integer k;

    always @(*) begin
        max_val = n_dout[0];
        max_idx = 4'd0;
        for (k = 1; k < NO; k = k + 1) begin
            if (n_dout[k] > max_val) begin
                max_val = n_dout[k];
                max_idx = k[3:0];
            end
        end
    end

    // ----------------------------------------------------------------
    //  Main FSM — address generation + neuron feeding
    // ----------------------------------------------------------------
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st           <= S_IDLE;
            cnt          <= {CW{1'b0}};
            in_addr      <= 16'd0;
            n_start      <= 1'b0;
            n_din_valid  <= 1'b0;
            n_din        <= 16'sd0;
            digit        <= 4'd0;
            result_valid <= 1'b0;
            done         <= 1'b0;
        end else begin
            n_start      <= 1'b0;
            n_din_valid  <= 1'b0;
            result_valid <= 1'b0;
            done         <= 1'b0;

            case (st)
            // ---- wait for external start pulse ----
            S_IDLE: if (start) begin
                st <= S_START;
            end

            // ---- pulse neuron start (loads bias internally) ----
            S_START: begin
                n_start <= 1'b1;
                in_addr <= 16'd0;
                cnt     <= {CW{1'b0}};
                st      <= S_ADDR;
            end

            // ---- first address issued, wait 1 cycle for BRAM read ----
            S_ADDR: begin
                st <= S_PIPE;
            end

            // ---- BRAM data available; from here we pipeline ----
            S_PIPE: begin
                n_din       <= in_data;
                n_din_valid <= 1'b1;
                if (cnt == NI[CW-1:0] - 1) begin
                    // last element fed
                    st <= S_WAIT;
                end else begin
                    cnt     <= cnt + 1;
                    in_addr <= in_addr + 16'd1;
                    st      <= S_FEED;
                end
            end

            // ---- steady-state: read data, feed neurons ----
            S_FEED: begin
                n_din       <= in_data;
                n_din_valid <= 1'b1;
                if (cnt == NI[CW-1:0] - 1) begin
                    st <= S_WAIT;
                end else begin
                    cnt     <= cnt + 1;
                    in_addr <= in_addr + 16'd1;
                end
            end

            // ---- wait for neurons to assert done ----
            S_WAIT: begin
                if (n_done[0]) begin
                    digit        <= max_idx;
                    result_valid <= 1'b1;
                    st           <= S_DONE;
                end
            end

            // ---- signal done to cifar_cnn FSM ----
            S_DONE: begin
                done <= 1'b1;
                st   <= S_IDLE;
            end

            default: st <= S_IDLE;
            endcase
        end
    end
endmodule

`timescale 1ns / 1ps
(* DONT_TOUCH = "yes" *)
module neuron #(
    parameter integer NUM_INPUTS  = 784,
    parameter         WEIGHT_FILE = "w.mem"
)(
    input  wire                clk,
    input  wire                rst_n,
    input  wire                start,
    input  wire signed [15:0]  din,
    input  wire                din_valid,
    output reg  signed [15:0]  dout,
    output reg                 done,
    input  wire                relu_en
);

    // ---- Weight ROM in BRAM ----
    // w_mem[0] = bias, w_mem[1..NUM_INPUTS] = weights
    (* ram_style = "block" *)
    reg signed [15:0] w_mem [0:NUM_INPUTS];
    initial $readmemh(WEIGHT_FILE, w_mem);

    // ---- Registered BRAM read (separate always → clean inference) ----
    localparam AW = $clog2(NUM_INPUTS + 2);
    reg  [AW-1:0]      w_addr;
    reg  signed [15:0]  w_rd;

    always @(posedge clk)
        w_rd <= w_mem[w_addr];

    // ---- FSM ----
    //  S_IDLE → S_B1 → S_B2 → S_MAC (→ S_ACT → S_IDLE)
    //  S_B1/S_B2 absorb the 1-cycle BRAM read latency for the bias
    localparam [2:0] S_IDLE = 3'd0,
                     S_B1   = 3'd1,
                     S_B2   = 3'd2,
                     S_MAC  = 3'd3,
                     S_ACT  = 3'd4;
    reg [2:0] state;

    localparam CW = $clog2(NUM_INPUTS > 1 ? NUM_INPUTS : 2);
    reg  [CW-1:0]     cnt;
    reg  signed [47:0] acc;

    wire signed [31:0] prod = din * w_rd;

    wire signed [15:0] z_raw = acc[23:8];
    wire               z_ovf = (acc[47:24] != {24{acc[23]}});
    wire signed [15:0] z_sat = z_ovf ? (acc[47] ? 16'sh8000
                                                 : 16'sh7FFF) : z_raw;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state  <= S_IDLE;
            acc    <= 48'd0;
            cnt    <= {CW{1'b0}};
            dout   <= 16'd0;
            done   <= 1'b0;
            w_addr <= {AW{1'b0}};
        end else begin
            done <= 1'b0;

            case (state)
            // ----- idle: wait for start -----
            S_IDLE: if (start) begin
                w_addr <= {AW{1'b0}};          // request bias from BRAM
                state  <= S_B1;
            end

            // ----- BRAM latency cycle for bias -----
            S_B1: begin
                w_addr <= {{(AW-1){1'b0}}, 1'b1}; // pre-fetch w_mem[1]
                state  <= S_B2;
            end

            // ----- load bias into accumulator -----
            S_B2: begin
                // w_rd now holds w_mem[0] = bias
                acc   <= {{24{w_rd[15]}}, w_rd, 8'b0};
                cnt   <= {CW{1'b0}};
                state <= S_MAC;
                // w_mem[1] will be in w_rd next cycle
            end

            // ----- multiply-accumulate -----
            S_MAC: if (din_valid) begin
                acc <= acc + {{16{prod[31]}}, prod};
                if (cnt == NUM_INPUTS[CW-1:0] - 1'd1)
                    state <= S_ACT;
                else begin
                    cnt    <= cnt + 1'b1;
                    w_addr <= w_addr + 1'b1;   // pre-fetch next weight
                end
            end

            // ----- activation + output -----
            S_ACT: begin
                dout  <= (relu_en && z_sat[15]) ? 16'sd0 : z_sat;
                done  <= 1'b1;
                state <= S_IDLE;
            end

            default: state <= S_IDLE;
            endcase
        end
    end
endmodule

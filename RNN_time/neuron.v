`timescale 1ns / 1ps
// ============================================================================
// neuron.v — Single neuron with Block-RAM weight storage
//
// Weight layout:  w[0] = bias,  w[1..NUM_INPUTS] = weights
// Arithmetic:     Q8.8 fixed-point, 48-bit accumulator, saturation
//
// ---- Pipeline Timing Diagram (cycle-accurate) ---------------------------
//
//  Let T = the absolute cycle at which the neuron sees start=1.
//  The external controller asserts din_valid starting at T+1.
//
//  Cycle   | w_addr set | BRAM out (w_rdata) | din_p2     | vld_p2 | Action
//  --------+------------+--------------------+------------+--------+---------
//  T       | 0          | —                  | —          | 0      | IDLE→PREFETCH
//  T+1     | 1          | —                  | —          | 0      | PREFETCH→LOAD_BIAS
//  T+2     | 2          | w[0] (bias)        | input[-2]? | 0      | load bias; →MAC
//  T+3     | 3          | w[1]               | input[0]   | 1      | MAC cnt=0
//  T+4     | 4          | w[2]               | input[1]   | 1      | MAC cnt=1
//  ...     | ...        | ...                | ...        | 1      | ...
//  T+N+2   | N+2        | w[N]               | input[N-1] | 1      | MAC cnt=N-1 →ACT
//  T+N+3   | —          | —                  | —          | —      | done=1
//
//  N = NUM_INPUTS.  Total latency = NUM_INPUTS + 3 cycles from start to done.
//
// Registered:  w_addr, w_rdata (BRAM), acc, cnt, state, dout, done,
//              din_p1, din_p2, vld_p1, vld_p2
// Combinational: prod, z_raw, z_ovf, z_sat
// ============================================================================
module neuron #(
    parameter integer NUM_INPUTS  = 129,
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

    // ================================================================
    // Weight Block RAM — synchronous read, 1-cycle latency
    // Depth = NUM_INPUTS + 1  (indices 0 … NUM_INPUTS)
    // ================================================================
    (* ram_style = "block", rom_style = "block" *)
    reg signed [15:0] w [0:NUM_INPUTS];
    initial $readmemh(WEIGHT_FILE, w);

    localparam AW = $clog2(NUM_INPUTS + 1 > 1 ? NUM_INPUTS + 1 : 2);

    reg  [AW-1:0]     w_addr;
    reg  signed [15:0] w_rdata;

    always @(posedge clk)
        w_rdata <= w[w_addr];          // 1-cycle read latency

    // ================================================================
    // 2-stage input pipeline  (aligns din with BRAM output)
    //   din_p2 arrives at the same cycle as the matching w_rdata
    // ================================================================
    reg signed [15:0] din_p1, din_p2;
    reg               vld_p1, vld_p2;

    always @(posedge clk) begin
        if (!rst_n) begin
            din_p1 <= 16'sd0;   din_p2 <= 16'sd0;
            vld_p1 <= 1'b0;     vld_p2 <= 1'b0;
        end else begin
            din_p1 <= din;       din_p2 <= din_p1;
            vld_p1 <= din_valid; vld_p2 <= vld_p1;
        end
    end

    // ================================================================
    // Datapath
    // ================================================================
    wire signed [31:0] prod = din_p2 * w_rdata;   // Q8.8 × Q8.8 = Q16.16

    reg  signed [47:0] acc;

    // Q8.8 extraction with symmetric saturation
    wire signed [15:0] z_raw = acc[23:8];
    wire               z_ovf = (acc[47:24] != {24{acc[23]}});
    wire signed [15:0] z_sat = z_ovf ? (acc[47] ? 16'sh8000
                                                 : 16'sh7FFF) : z_raw;

    // ================================================================
    // FSM
    // ================================================================
    localparam [2:0] S_IDLE      = 3'd0,
                     S_PREFETCH  = 3'd1,   // BRAM latency for bias
                     S_LOAD_BIAS = 3'd2,   // bias on w_rdata → acc
                     S_MAC       = 3'd3,   // multiply-accumulate
                     S_ACT       = 3'd4;   // activation + output

    reg [2:0] state;

    localparam CW = $clog2(NUM_INPUTS > 0 ? NUM_INPUTS + 1 : 2);
    reg [CW-1:0] cnt;

    always @(posedge clk) begin
        if (!rst_n) begin
            state  <= S_IDLE;
            acc    <= 48'sd0;
            cnt    <= {CW{1'b0}};
            dout   <= 16'sd0;
            done   <= 1'b0;
            w_addr <= {AW{1'b0}};
        end else begin
            done <= 1'b0;                           // default: 1-cycle pulse

            case (state)
            // ---- idle: wait for start ----
            S_IDLE:
                if (start) begin
                    w_addr <= {AW{1'b0}};           // request bias (addr 0)
                    state  <= S_PREFETCH;
                end

            // ---- BRAM latency cycle (w[0] being fetched) ----
            S_PREFETCH: begin
                w_addr <= {{(AW-1){1'b0}}, 1'b1};  // addr = 1 (first weight)
                state  <= S_LOAD_BIAS;
            end

            // ---- w_rdata now holds w[0] (bias) ----
            S_LOAD_BIAS: begin
                acc    <= {{24{w_rdata[15]}}, w_rdata, 8'b0};   // bias in Q16.16
                w_addr <= {{(AW-2){1'b0}}, 2'b10};              // addr = 2
                cnt    <= {CW{1'b0}};
                state  <= S_MAC;
            end

            // ---- MAC: one multiply-accumulate per vld_p2 pulse ----
            S_MAC:
                if (vld_p2) begin
                    acc <= acc + {{16{prod[31]}}, prod};
                    if (cnt == NUM_INPUTS[CW-1:0] - 1'd1)
                        state <= S_ACT;
                    else begin
                        cnt    <= cnt + 1'b1;
                        w_addr <= w_addr + 1'b1;    // prefetch next weight
                    end
                end

            // ---- activation & output ----
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

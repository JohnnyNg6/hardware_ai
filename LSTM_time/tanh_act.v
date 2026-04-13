// ============================================================
// neuron.v — Reusable pipelined MAC neuron with BRAM weights
//
// Used as building block for BOTH Dense layer and LSTM gates.
// Weights stored in block RAM with 1-cycle read latency.
// Prefetch pipeline keeps throughput at 1 MAC per clock.
//
// Parameters:
//   NUM_INPUTS : number of input elements per computation
//   MEM_DEPTH  : total weight memory depth
//                Dense: NUM_INPUTS+1
//                LSTM:  4*(CONCAT_SIZE+1)  (all 4 gates in one BRAM)
//   WEIGHT_FILE: hex file for $readmemh
//
// Interface timing (controller must obey):
//   Cycle  0 : assert start=1 (1-clk pulse), set base_addr
//   Cycle  1 : (internal) neuron in S_BIAS_WAIT
//   Cycle  2 : (internal) neuron in S_BIAS_LOAD — bias into acc
//              controller outputs din_valid=1, din=input[0]
//   Cycle  3 : first MAC: din_r=input[0], w_rd=weight[0]
//              controller outputs din=input[1]
//   ...
//   Cycle N+2: last MAC: din_r=input[N-1], w_rd=weight[N-1]
//   Cycle N+3: dout valid, done=1  (1-clk pulse)
//
// Pipeline diagram (per clock edge):
//   Edge T   : controller sets din, din_valid, base_addr
//   Edge T   : BRAM addr computed combinationally
//   Edge T+1 : w_rd = BRAM[addr_T], din_r = din_T, dvr = dv_T
//   Edge T+1 : if dvr && S_MAC: acc += din_r * w_rd
//
// Signals:
//   start, base_addr — registered by caller, sampled at posedge
//   din, din_valid    — registered by caller, delayed 1 clk inside
//   dout              — registered output, holds until next start
//   done              — 1-clk pulse
// ============================================================
`timescale 1ns/1ps
module neuron #(
    parameter integer NUM_INPUTS  = 128,
    parameter integer MEM_DEPTH   = 129,
    parameter         WEIGHT_FILE = "w.mem"
)(
    input  wire                                          clk,
    input  wire                                          rst_n,
    input  wire                                          start,
    input  wire [$clog2(MEM_DEPTH > 1 ? MEM_DEPTH : 2)-1:0] base_addr,
    input  wire signed [15:0]                            din,
    input  wire                                          din_valid,
    output reg  signed [15:0]                            dout,
    output reg                                           done
);
    // ── address / counter widths ──
    localparam AW = $clog2(MEM_DEPTH > 1 ? MEM_DEPTH : 2);
    localparam CW = $clog2(NUM_INPUTS > 1 ? NUM_INPUTS : 2);

    // ── weight BRAM ──
    (* ram_style = "block" *)
    reg signed [15:0] w [0:MEM_DEPTH-1];
    initial $readmemh(WEIGHT_FILE, w);

    // BRAM read — address is combinational, output registered (1-clk latency)
    reg  [AW-1:0]      rd_addr_reg;          // holds address between states
    wire [AW-1:0]      rd_addr;              // actual BRAM read address
    reg  signed [15:0]  w_rd;                 // BRAM output (1-clk delayed)
    always @(posedge clk)
        w_rd <= w[rd_addr];

    // ── 1-cycle delay registers for pipeline alignment ──
    reg signed [15:0] din_r;
    reg               din_valid_r;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            din_r       <= 16'd0;
            din_valid_r <= 1'b0;
        end else begin
            din_r       <= din;
            din_valid_r <= din_valid;
        end
    end

    // ── MAC ──
    wire signed [31:0] prod = din_r * w_rd;

    // ── accumulator ──
    reg signed [47:0] acc;

    // ── Q8.8 saturation ──
    wire signed [15:0] z_raw = acc[23:8];
    wire               z_ovf = (acc[47:24] != {24{acc[23]}});
    wire signed [15:0] z_sat = z_ovf ? (acc[47] ? 16'sh8000 : 16'sh7FFF)
                                     : z_raw;

    // ── FSM ──
    localparam [2:0] S_IDLE      = 3'd0,
                     S_BIAS_WAIT = 3'd1,   // BRAM latency for bias
                     S_BIAS_LOAD = 3'd2,   // load bias into acc
                     S_MAC       = 3'd3,   // streaming MAC
                     S_OUT       = 3'd4;   // output result

    reg [2:0]     state;
    reg [CW-1:0]  cnt;
    reg [AW-1:0]  base_r;     // latched base address

    // ── BRAM address MUX ──
    //   Combinational so BRAM sees address in the SAME cycle.
    //   States:
    //     S_IDLE  + start     : base_addr          (read bias)
    //     S_BIAS_WAIT         : base_r + 1         (prefetch first weight)
    //     S_MAC               : base_r + cnt + 2   (prefetch next weight)
    //     otherwise           : rd_addr_reg (hold)
    assign rd_addr = (state == S_IDLE && start)       ? base_addr :
                     (state == S_BIAS_WAIT)           ? base_r + {{(AW-1){1'b0}}, 1'b1} :
                     (state == S_MAC && din_valid_r)  ? base_r + {{(AW-CW-1){1'b0}}, cnt} + {{(AW-1){1'b0}}, 2'd2} :
                     rd_addr_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= S_IDLE;
            acc        <= 48'd0;
            cnt        <= {CW{1'b0}};
            dout       <= 16'd0;
            done       <= 1'b0;
            base_r     <= {AW{1'b0}};
            rd_addr_reg <= {AW{1'b0}};
        end else begin
            done <= 1'b0;
            rd_addr_reg <= rd_addr;  // hold last BRAM address

            case (state)
                // ── idle: wait for start ──
                S_IDLE: begin
                    if (start) begin
                        base_r <= base_addr;
                        state  <= S_BIAS_WAIT;
                    end
                end

                // ── wait 1 clk for BRAM to output bias ──
                S_BIAS_WAIT: begin
                    state <= S_BIAS_LOAD;
                end

                // ── load bias into accumulator ──
                //    w_rd now holds w[base_r] = bias
                S_BIAS_LOAD: begin
                    acc   <= {{24{w_rd[15]}}, w_rd, 8'b0};
                    cnt   <= {CW{1'b0}};
                    state <= S_MAC;
                end

                // ── streaming MAC ──
                S_MAC: begin
                    if (din_valid_r) begin
                        acc <= acc + {{16{prod[31]}}, prod};
                        if (cnt == NUM_INPUTS[CW-1:0] - {{(CW-1){1'b0}}, 1'b1})
                            state <= S_OUT;
                        else
                            cnt <= cnt + {{(CW-1){1'b0}}, 1'b1};
                    end
                end

                // ── output saturated result ──
                S_OUT: begin
                    dout  <= z_sat;
                    done  <= 1'b1;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end
endmodule

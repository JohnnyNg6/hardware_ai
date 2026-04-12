// ============================================================================
// neuron.v — Single Perceptron / Neuron (the Ch01 NAND gate, generalised)
//
// Operation : y = activation( bias + Σ w_i·x_i )
// Arithmetic: Q8.8 signed fixed-point (16-bit)
// Interface : inputs arrive serially, one per clock while din_valid = 1
//             'start' pulse loads bias; 'done' pulse means dout is ready
// ============================================================================
`timescale 1ns / 1ps

module neuron #(
    parameter integer NUM_INPUTS  = 784,
    parameter         WEIGHT_FILE = "w.mem"      // hex, one Q8.8 value/line
)(                                                // w[0]=bias  w[1..N]=weights
    input  wire                clk,
    input  wire                rst_n,
    input  wire                start,             // 1-clk pulse → begin
    input  wire signed [15:0]  din,               // Q8.8 serial input
    input  wire                din_valid,
    output reg  signed [15:0]  dout,              // Q8.8 result
    output reg                 done,              // 1-clk pulse → result ready
    input  wire                relu_en            // 1 = ReLU, 0 = linear
);

    // ---------- weight ROM (distributed RAM for async read) ----------
    (* ram_style = "distributed" *)
    reg signed [15:0] w [0:NUM_INPUTS];           // bias + N weights
    initial $readmemh(WEIGHT_FILE, w);

    // ---------- FSM --------------------------------------------------
    localparam [1:0] S_IDLE = 2'd0,
                     S_MAC  = 2'd1,
                     S_ACT  = 2'd2;
    reg [1:0] state;

    // ---------- datapath ---------------------------------------------
    localparam CW = $clog2(NUM_INPUTS > 1 ? NUM_INPUTS : 2);
    reg  [CW-1:0]     cnt;                        // input counter
    reg  signed [47:0] acc;                        // accumulator (Q_.16)

    wire signed [31:0] prod = din * w[cnt + 1'd1]; // Q8.8 × Q8.8 = Q16.16

    // Q_.16 → Q8.8 extraction with saturation
    wire signed [15:0] z_raw = acc[23:8];
    wire               z_ovf = (acc[47:24] != {24{acc[23]}});
    wire signed [15:0] z_sat = z_ovf ? (acc[47] ? 16'sh8000
                                                 : 16'sh7FFF) : z_raw;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            acc   <= 48'sd0;
            cnt   <= {CW{1'b0}};
            dout  <= 16'sd0;
            done  <= 1'b0;
        end else begin
            done <= 1'b0;                           // default: de-assert

            case (state)
                // --- wait for start ---
                S_IDLE: if (start) begin
                    acc   <= {{24{w[0][15]}}, w[0], 8'b0};  // bias → Q_.16
                    cnt   <= {CW{1'b0}};
                    state <= S_MAC;
                end

                // --- multiply-accumulate (1 input / clock) ---
                S_MAC: if (din_valid) begin
                    acc <= acc + {{16{prod[31]}}, prod};
                    if (cnt == NUM_INPUTS[CW-1:0] - 1'd1)
                        state <= S_ACT;
                    else
                        cnt <= cnt + 1'b1;
                end

                // --- activation ---
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

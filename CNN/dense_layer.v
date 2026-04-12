// ============================================================================
// dense_layer.v — Serial dense layer + argmax, Q8.8 fixed-point
// ============================================================================
`timescale 1ns / 1ps

module dense_layer #(
    parameter integer NI    = 3136,
    parameter integer NO    = 10,
    parameter         WFILE = "dense_w.mem",
    parameter         BFILE = "dense_b.mem"
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    output reg         done,
    output reg  [15:0] in_addr,
    input  wire signed [15:0] in_data,
    output reg  [3:0]  digit,
    output reg         result_valid
);

localparam integer WDEP = NO * NI;

(* ram_style = "block" *)
reg signed [15:0] wrom [0:WDEP-1];
initial $readmemh(WFILE, wrom);

(* ram_style = "distributed" *)
reg signed [15:0] brom [0:NO-1];
initial $readmemh(BFILE, brom);

reg [15:0] wa_r;
reg signed [15:0] w_rd;
always @(posedge clk) w_rd <= wrom[wa_r];

/* --- FSM --- */
localparam [2:0] S_IDLE=0, S_BIAS=1, S_PIPE=2, S_MAC=3,
                 S_STORE=4, S_ARG=5, S_DONE=6;
reg [2:0] st;

reg [3:0]  n_r;            // neuron index 0..NO-1
reg [11:0] k_r;            // input index
reg signed [47:0] acc;

reg signed [15:0] logits [0:NO-1];

/* --- saturation --- */
wire signed [15:0] z_raw = acc[23:8];
wire               z_ovf = (acc[47:24] != {24{acc[23]}});
wire signed [15:0] z_sat = z_ovf ? (acc[47] ? 16'sh8000 : 16'sh7FFF) : z_raw;

wire signed [31:0] prod = in_data * w_rd;

/* --- argmax regs --- */
reg [3:0]  arg_i;
reg signed [15:0] max_v;
reg [3:0]  max_i;

integer i;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        st<=S_IDLE; done<=0; result_valid<=0; digit<=0;
        n_r<=0; k_r<=0; acc<=0; wa_r<=0; in_addr<=0;
        arg_i<=0; max_v<=0; max_i<=0;
        for (i=0;i<NO;i=i+1) logits[i] <= 0;
    end else begin
        done<=0; result_valid<=0;
        case (st)
        S_IDLE: if (start) begin n_r<=0; st<=S_BIAS; end

        S_BIAS: begin
            acc <= {{24{brom[n_r][15]}}, brom[n_r], 8'b0};
            in_addr <= 16'd0;
            wa_r    <= {12'd0, n_r} * NI[15:0];
            k_r     <= 0;
            st <= S_PIPE;
        end

        S_PIPE: begin
            in_addr <= 16'd1;
            wa_r    <= wa_r + 16'd1;
            k_r     <= 0;
            st <= S_MAC;
        end

        S_MAC: begin
            acc <= acc + {{16{prod[31]}}, prod};
            if (k_r == NI[11:0] - 12'd1) begin
                st <= S_STORE;
            end else begin
                in_addr <= {4'd0, k_r} + 16'd2;
                wa_r    <= wa_r + 16'd1;
                k_r     <= k_r + 12'd1;
            end
        end

        S_STORE: begin
            logits[n_r] <= z_sat;   // no ReLU for output layer
            if (n_r == NO[3:0] - 4'd1) begin
                max_v <= z_sat;     // init argmax with neuron 0's value
                max_i <= 4'd0;
                arg_i <= 4'd1;
                // actually need logit[0] separately; use stored
                st <= S_ARG;
            end else begin
                n_r <= n_r + 4'd1;
                st <= S_BIAS;
            end
        end

        S_ARG: begin
            if ($signed(logits[arg_i]) > $signed(max_v)) begin
                max_v <= logits[arg_i];
                max_i <= arg_i;
            end
            if (arg_i == NO[3:0] - 4'd1) st <= S_DONE;
            else arg_i <= arg_i + 4'd1;
        end

        S_DONE: begin
            digit <= max_i;
            result_valid <= 1;
            done <= 1;
            st <= S_IDLE;
        end
        default: st <= S_IDLE;
        endcase
    end
end
endmodule

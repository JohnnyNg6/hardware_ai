// ============================================================
// neuron.v — Single MAC neuron (serial-input, parallel-weight)
//   Used as the building block for the Dense output layer.
//   Activation: 0=linear (default for dense before argmax)
// ============================================================
`timescale 1ns/1ps
module neuron #(
    parameter integer NUM_INPUTS  = 128,
    parameter         WEIGHT_FILE = "w.mem"
)(
    input  wire               clk,
    input  wire               rst_n,
    input  wire               start,
    input  wire signed [15:0] din,
    input  wire               din_valid,
    output reg  signed [15:0] dout,
    output reg                done
);
    (* ram_style = "distributed" *)
    reg signed [15:0] w [0:NUM_INPUTS];     // w[0]=bias, w[1..N]=weights
    initial $readmemh(WEIGHT_FILE, w);

    localparam CW = $clog2(NUM_INPUTS > 1 ? NUM_INPUTS : 2);
    localparam [1:0] S_IDLE = 2'd0,
                     S_MAC  = 2'd1,
                     S_OUT  = 2'd2;
    reg [1:0] state;
    reg [CW-1:0]       cnt;
    reg signed [47:0]  acc;

    wire signed [31:0] prod = din * w[cnt + 1'd1];

    wire signed [15:0] z_raw = acc[23:8];
    wire               z_ovf = (acc[47:24] != {24{acc[23]}});
    wire signed [15:0] z_sat = z_ovf ? (acc[47] ? 16'sh8000
                                                 : 16'sh7FFF) : z_raw;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE; acc <= 0; cnt <= 0;
            dout <= 0; done <= 0;
        end else begin
            done <= 1'b0;
            case (state)
                S_IDLE: if (start) begin
                    acc   <= {{24{w[0][15]}}, w[0], 8'b0};
                    cnt   <= 0;
                    state <= S_MAC;
                end
                S_MAC: if (din_valid) begin
                    acc <= acc + {{16{prod[31]}}, prod};
                    if (cnt == NUM_INPUTS[CW-1:0] - 1'd1)
                        state <= S_OUT;
                    else
                        cnt <= cnt + 1'b1;
                end
                S_OUT: begin
                    dout  <= z_sat;       // linear activation
                    done  <= 1'b1;
                    state <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule

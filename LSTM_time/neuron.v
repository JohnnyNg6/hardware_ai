// ============================================================
// neuron.v — Pipelined MAC neuron with BRAM weights
// ============================================================
`timescale 1ns/1ps
module neuron #(
    parameter integer NUM_INPUTS  = 128,
    parameter integer MEM_DEPTH   = 129,
    parameter         WEIGHT_FILE = "w.mem"
)(
    input  wire                                              clk,
    input  wire                                              rst_n,
    input  wire                                              start,
    input  wire [$clog2(MEM_DEPTH > 1 ? MEM_DEPTH : 2)-1:0] base_addr,
    input  wire signed [15:0]                                din,
    input  wire                                              din_valid,
    output reg  signed [15:0]                                dout,
    output reg                                               done
);
    localparam AW = $clog2(MEM_DEPTH > 1 ? MEM_DEPTH : 2);
    localparam CW = $clog2(NUM_INPUTS > 1 ? NUM_INPUTS : 2);

    // AW-bit constants
    localparam [AW-1:0] ONE_AW = 1;
    localparam [AW-1:0] TWO_AW = 2;

    // ── weight BRAM ──
    (* ram_style = "block" *)
    reg signed [15:0] w [0:MEM_DEPTH-1];
    initial $readmemh(WEIGHT_FILE, w);

    reg  [AW-1:0]     rd_addr_reg;
    wire [AW-1:0]     rd_addr;
    reg  signed [15:0] w_rd;
    always @(posedge clk)
        w_rd <= w[rd_addr];

    // ── 1-cycle pipeline registers ──
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

    wire signed [31:0] prod = din_r * w_rd;

    reg signed [47:0] acc;

    wire signed [15:0] z_raw = acc[23:8];
    wire               z_ovf = (acc[47:24] != {24{acc[23]}});
    wire signed [15:0] z_sat = z_ovf ? (acc[47] ? 16'sh8000 : 16'sh7FFF)
                                     : z_raw;

    // ── FSM ──
    localparam [2:0] S_IDLE      = 3'd0,
                     S_BIAS_WAIT = 3'd1,
                     S_BIAS_LOAD = 3'd2,
                     S_MAC       = 3'd3,
                     S_OUT       = 3'd4;

    reg [2:0]     state;
    reg [CW-1:0]  cnt;
    reg [AW-1:0]  base_r;

    // extend cnt to AW bits
    wire [AW-1:0] cnt_aw;
    assign cnt_aw = {{(AW-CW){1'b0}}, cnt};

    // ── BRAM address MUX (combinational) ──
    assign rd_addr = (state == S_IDLE && start)      ? base_addr :
                     (state == S_BIAS_WAIT)           ? (base_r + ONE_AW) :
                     (state == S_MAC && din_valid_r)  ? (base_r + cnt_aw + TWO_AW) :
                     rd_addr_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            acc         <= 48'd0;
            cnt         <= {CW{1'b0}};
            dout        <= 16'd0;
            done        <= 1'b0;
            base_r      <= {AW{1'b0}};
            rd_addr_reg <= {AW{1'b0}};
        end else begin
            done        <= 1'b0;
            rd_addr_reg <= rd_addr;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        base_r <= base_addr;
                        state  <= S_BIAS_WAIT;
                    end
                end

                // w_rd now holds bias (read was issued in S_IDLE)
                S_BIAS_WAIT: begin
                    acc   <= {{24{w_rd[15]}}, w_rd, 8'b0};   // ◄ FIXED: load bias HERE
                    state <= S_BIAS_LOAD;
                end

                // weight[0] prefetch settling
                S_BIAS_LOAD: begin
                    cnt   <= {CW{1'b0}};
                    state <= S_MAC;
                end

                S_MAC: begin
                    if (din_valid_r) begin
                        acc <= acc + {{16{prod[31]}}, prod};
                        if (cnt == NUM_INPUTS[CW-1:0] - ONE_AW[CW-1:0])
                            state <= S_OUT;
                        else
                            cnt <= cnt + {{(CW-1){1'b0}}, 1'b1};
                    end
                end

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

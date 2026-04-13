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

    localparam [AW-1:0] ONE_AW   = 1;
    localparam [AW-1:0] TWO_AW   = 2;
    localparam [AW-1:0] THREE_AW = 3;

    // ── weight BRAM (2-stage read pipeline) ──
    (* rom_style = "block" *)
    reg signed [15:0] w [0:MEM_DEPTH-1];
    initial $readmemh(WEIGHT_FILE, w);

    reg  [AW-1:0]     rd_addr_reg;
    wire [AW-1:0]     rd_addr;

    // Stage 1: BRAM output register (packed INTO the BRAM)
    reg signed [15:0] w_rd;
    always @(posedge clk)
        w_rd <= w[rd_addr];

    // Stage 2: pipeline register (DSP absorbs this as BREG)
    reg signed [15:0] w_rd2;
    always @(posedge clk)
        w_rd2 <= w_rd;

    // ── 2-cycle input pipeline (matches w_rd→w_rd2 latency) ──
    reg signed [15:0] din_r1, din_r2;
    reg               din_valid_r1, din_valid_r2;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            din_r1       <= 16'd0;
            din_valid_r1 <= 1'b0;
            din_r2       <= 16'd0;
            din_valid_r2 <= 1'b0;
        end else begin
            din_r1       <= din;
            din_valid_r1 <= din_valid;
            din_r2       <= din_r1;
            din_valid_r2 <= din_valid_r1;
        end
    end

    wire signed [31:0] prod = din_r2 * w_rd2;

    reg signed [47:0] acc;

    wire signed [15:0] z_raw = acc[23:8];
    wire               z_ovf = (acc[47:24] != {24{acc[23]}});
    wire signed [15:0] z_sat = z_ovf ? (acc[47] ? 16'sh8000 : 16'sh7FFF)
                                     : z_raw;

    // ── FSM (6 states, extra S_PREFETCH for 2-stage pipeline) ──
    localparam [2:0] S_IDLE      = 3'd0,
                     S_BIAS_WAIT = 3'd1,
                     S_BIAS_LOAD = 3'd2,
                     S_PREFETCH  = 3'd3,
                     S_MAC       = 3'd4,
                     S_OUT       = 3'd5;

    reg [2:0]     state;
    reg [CW-1:0]  cnt;
    reg [AW-1:0]  base_r;

    wire [AW-1:0] cnt_aw = {{(AW-CW){1'b0}}, cnt};

    // ── BRAM address MUX ──
    assign rd_addr = (state == S_IDLE && start)       ? base_addr :
                     (state == S_BIAS_WAIT)            ? (base_r + ONE_AW) :
                     (state == S_PREFETCH)             ? (base_r + TWO_AW) :
                     (state == S_MAC && din_valid_r1)  ? (base_r + cnt_aw + THREE_AW) :
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

                S_BIAS_WAIT: begin
                    state <= S_BIAS_LOAD;
                end

                // w_rd2 now holds bias (2 cycles after S_IDLE address)
                S_BIAS_LOAD: begin
                    acc   <= {{24{w_rd2[15]}}, w_rd2, 8'b0};
                    cnt   <= {CW{1'b0}};
                    state <= S_PREFETCH;
                end

                // w_rd2 now holds weight[0]; issue weight[1] address
                S_PREFETCH: begin
                    state <= S_MAC;
                end

                S_MAC: begin
                    if (din_valid_r2) begin
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

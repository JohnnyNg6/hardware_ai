`timescale 1ns/1ps
module argmax #(
    parameter integer N = 63
)(
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire [N*16-1:0]       vals_flat,
    input  wire                  start,
    output reg  [$clog2(N)-1:0]  idx,
    output reg                   done
);
    localparam IW = $clog2(N);

    reg [IW-1:0]      scan;
    reg signed [15:0]  best_val;
    reg [IW-1:0]       best_idx;
    reg                active;

    // Combinational mux: pick element 'scan' (one 63:1 mux, ~3 LUT levels)
    reg signed [15:0] cur_val;
    integer k;
    always @(*) begin
        cur_val = $signed(vals_flat[15:0]);
        for (k = 0; k < N; k = k + 1)
            if (scan == k[IW-1:0])
                cur_val = $signed(vals_flat[k*16 +: 16]);
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active   <= 1'b0;
            done     <= 1'b0;
            idx      <= {IW{1'b0}};
            scan     <= {IW{1'b0}};
            best_val <= 16'sh8000;
            best_idx <= {IW{1'b0}};
        end else begin
            done <= 1'b0;

            if (start && !active) begin
                active   <= 1'b1;
                scan     <= {IW{1'b0}};
                best_val <= 16'sh8000;
                best_idx <= {IW{1'b0}};
            end else if (active) begin
                if (cur_val > best_val) begin
                    best_val <= cur_val;
                    best_idx <= scan;
                end

                if (scan == N[IW-1:0] - {{(IW-1){1'b0}}, 1'b1}) begin
                    active <= 1'b0;
                    done   <= 1'b1;
                    idx    <= (cur_val > best_val) ? scan : best_idx;
                end else begin
                    scan <= scan + {{(IW-1){1'b0}}, 1'b1};
                end
            end
        end
    end
endmodule

`timescale 1ns / 1ps

module tb_perceptron_nand;

    reg               clk;
    reg               rst;
    reg               valid_in;
    reg  signed [7:0] x0, x1;
    wire              y;
    wire              valid_out;

    perceptron_nand uut (
        .clk(clk), .rst(rst), .valid_in(valid_in),
        .x0(x0), .x1(x1), .y(y), .valid_out(valid_out)
    );

    // Clock: 50MHz
    initial clk = 0;
    always #10 clk = ~clk;

    // Q4.4 encoding: 1.0 = 16, 0.0 = 0
    localparam signed [7:0] ONE  = 8'sd16;  // 1.0 in Q4.4
    localparam signed [7:0] ZERO = 8'sd0;   // 0.0 in Q4.4

    integer pass_count;
    integer test_count;

    task test_nand;
        input signed [7:0] in0, in1;
        input expected;
        begin
            @(posedge clk);
            x0 = in0; x1 = in1; valid_in = 1;
            @(posedge clk);
            valid_in = 0;
            @(posedge clk); // wait for pipeline
            @(posedge clk);
            test_count = test_count + 1;
            if (y == expected) begin
                $display("PASS: NAND(%0d,%0d) = %0d (expected %0d)",
                         in0/16, in1/16, y, expected);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL: NAND(%0d,%0d) = %0d (expected %0d)",
                         in0/16, in1/16, y, expected);
            end
        end
    endtask

    initial begin
        $display("============================================");
        $display("Phase 1: Single Perceptron NAND Gate Test");
        $display("============================================");
        rst = 1; valid_in = 0; x0 = 0; x1 = 0;
        pass_count = 0; test_count = 0;
        repeat(5) @(posedge clk);
        rst = 0;
        repeat(2) @(posedge clk);

        // Test all 4 input combinations
        test_nand(ZERO, ZERO, 1);   // NAND(0,0) = 1
        test_nand(ZERO, ONE,  1);   // NAND(0,1) = 1
        test_nand(ONE,  ZERO, 1);   // NAND(1,0) = 1
        test_nand(ONE,  ONE,  0);   // NAND(1,1) = 0

        repeat(5) @(posedge clk);
        $display("============================================");
        $display("Results: %0d/%0d passed", pass_count, test_count);
        $display("============================================");
        $finish;
    end

    // For VCS+Verdi waveform dump
    initial begin
        $fsdbDumpfile("wave.fsdb");
        $fsdbDumpvars(0, tb_perceptron_nand);
    end

endmodule

`timescale 1ns / 1ps

module tb_perceptron_nand;

    reg               clk;
    reg               rst;
    reg               valid_in;
    reg  signed [7:0] x1, x2;
    wire              y_out;
    wire              valid_out;

    perceptron_nand uut (
        .clk       (clk),
        .rst       (rst),
        .valid_in  (valid_in),
        .x1        (x1),
        .x2        (x2),
        .y_out     (y_out),
        .valid_out (valid_out)
    );

    // 50 MHz clock
    initial clk = 0;
    always #10 clk = ~clk;

    // Q4.4 constants  (matches Python: +1.0 and -1.0)
    localparam signed [7:0] POS_ONE =  8'sd16;   // +1.0
    localparam signed [7:0] NEG_ONE = -8'sd16;   // -1.0

    integer pass_count, test_count;

    // -------------------------------------------------------
    // Task: feed one test vector, wait for pipeline, check
    // -------------------------------------------------------
    task test_perceptron;
        input signed  [7:0] in1, in2;
        input                expected;   // 1 = +1,  0 = -1
        begin
            @(posedge clk);
            x1 = in1;  x2 = in2;  valid_in = 1;
            @(posedge clk);
            valid_in = 0;
            @(posedge clk);          // pipeline stage 1
            @(posedge clk);          // pipeline stage 2
            test_count = test_count + 1;
            if (y_out == expected) begin
                $display("PASS: f(%2d, %2d) = %0d  (expected %0d)",
                         in1 / 16, in2 / 16, y_out, expected);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL: f(%2d, %2d) = %0d  (expected %0d)",
                         in1 / 16, in2 / 16, y_out, expected);
            end
        end
    endtask

    // -------------------------------------------------------
    // Main stimulus
    // -------------------------------------------------------
    initial begin
        $display("================================================");
        $display(" Perceptron NAND Test  (bipolar -1 / +1)");
        $display(" Trained weights: w0~0.40  w1~-0.40  w2~-0.15");
        $display("================================================");

        rst = 1;  valid_in = 0;  x1 = 0;  x2 = 0;
        pass_count = 0;  test_count = 0;
        repeat (5) @(posedge clk);
        rst = 0;
        repeat (2) @(posedge clk);

        // Python notebook truth table:
        //   x1=-1, x2=-1  →  y = +1   (LED = 1)
        //   x1=-1, x2=+1  →  y = +1   (LED = 1)
        //   x1=+1, x2=-1  →  y = +1   (LED = 1)
        //   x1=+1, x2=+1  →  y = -1   (LED = 0)
        test_perceptron(NEG_ONE, NEG_ONE, 1);
        test_perceptron(NEG_ONE, POS_ONE, 1);
        test_perceptron(POS_ONE, NEG_ONE, 1);
        test_perceptron(POS_ONE, POS_ONE, 0);

        repeat (5) @(posedge clk);
        $display("================================================");
        $display(" Results: %0d / %0d passed", pass_count, test_count);
        $display("================================================");
        $finish;
    end

    // VCS + Verdi waveform dump
    initial begin
        $fsdbDumpfile("wave.fsdb");
        $fsdbDumpvars(0, tb_perceptron_nand);
    end

endmodule

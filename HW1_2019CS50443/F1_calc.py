import sys
import argparse
import subprocess


def print_usage():
    print("usage: $ python3 {} --expected PATH/TO/EXPECTED --output PATH/TO/OUTPUT"\
            .format(sys.argv[0]), file=sys.stderr)
    message = "\n{spaces}expected: Path of file containing expected output"\
              "\n{spaces}output: Path of file containing output of user's program"\
              "\n{spaces}returns boolean representing whether the two files match or not"\
              "\n\n{spaces}Same information can be obtained using $ diff <expected> <output> -iwB"\
              "\n{spaces}where the diff command doesn't give any output in case of success."\
              "\n\n{spaces}Hint: Use the testcase out_<n>.dat_sorted files as <expected>"\
              "\n{spaces}and output from your file as <output> parameter of this script."\
              .format(spaces=" "*len("usage: $ "))
    print(message, file=sys.stderr)
    print("\noptions\n{spaces} --expected, -e{spaces2}Path to expected correct"\
          " output file"\
          "\n{spaces} --output,   -o{spaces2}Path to output of user program"\
          .format(spaces=" "*len("usage: $"), spaces2=" "*13), file=sys.stderr)


def calculate_F1(*, expected, output):
    r"""Takes two parameters representing paths to file containing expected corr-
    ect result and file containing output by user program and calculates F-score
    between the two.
    :param expected (self explanatory)
    :param output (self explanatory)
    :returns float representing F1-score

    Note: this is equivalent to having two files with extra spaces and empty
    lines removed and then running $ grep -Fxf <expected> <output> | wc -l
    to get n and using $ wc -l <expected> to get M and $ wc -l <output> to get
    N and finally calculating 2n/(M+N).
    """
    M, N, n = 0, 0, 0
    correct_lines = []
    with open(expected, "r") as exp:
        for line in exp:
            ln = " ".join(t.strip() for t in line.strip().split())
            if len(ln) > 0:
                correct_lines.append(ln.upper())
        # assuming correct itemsets can be held in memory
    M = len(correct_lines)
    already_seen = []
    with open(output, "r") as outp:
        for line in outp:
            ln = " ".join(t.strip() for t in line.strip().split()).upper()
            if len(ln) <= 0:
                continue
            N += 1
            if ln in correct_lines:
                if ln not in already_seen:
                    n += 1
                    already_seen.append(ln)
    print("n = {}, N = {}, M = {}, F1 = {}".format(n,N,M,2*n/(M+N)), file=sys.stderr)
    return 2 * n / (M + N)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected","-e",required=True,type=str)
    parser.add_argument("--output","-o",required=True,type=str)
    if len(sys.argv) <= 1:
        print_usage()
        sys.exit()
    args = parser.parse_args()
    print(calculate_F1(**vars(args)))


if __name__=="__main__":
    main()

import os
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected","-e",required=True,type=str)
    parser.add_argument("--output","-o",required=True,type=str)
    if len(sys.argv) <= 1:
        print_usage()
        sys.exit()
    args = parser.parse_args()
    expected = args.expected
    output = args.output
    # convert to uppercase
    os.system(r"sed 's/[a-z]/\U&/g' {} > tmp1".format(expected))
    os.system(r"sed 's/[a-z]/\U&/g' {} > tmp2".format(output))
    # remove double spaces
    os.system(r"sed -i 's/  */ /g' tmp1")
    os.system(r"sed -i 's/  */ /g' tmp2")
    # remove leading and ending spaces
    os.system(r"sed -i 's/^ *//g' tmp1")
    os.system(r"sed -i 's/^ *//g' tmp2")
    os.system(r"sed -i 's/ *$//g' tmp1")
    os.system(r"sed -i 's/ *$//g' tmp2")
    # remove empty lines
    os.system(r"sed -i '/^ *$/d' tmp1")
    os.system(r"sed -i '/^ *$/d' tmp2")
    # calculate M
    process = subprocess.Popen("wc -l tmp1".split(),
            stdout=subprocess.PIPE)
    out, _ = process.communicate()
    M = int(out.decode().split()[0])
    # calculate N
    process = subprocess.Popen("wc -l tmp2".split(),
            stdout=subprocess.PIPE)
    out, _ = process.communicate()
    N = int(out.decode().split()[0])
    # calculate n
    os.system("grep -Fxf tmp1 tmp2 > tmp3")
    process = subprocess.Popen("wc -l tmp3".split(), stdout=subprocess.PIPE)
    out, _ = process.communicate()
    n = int(out.decode().split()[0])
    F1 = 2 * n / (M + N)
    print("n = {}, N = {}, M = {}, F1 = {}".format(n, N, M, F1))
    os.system("rm tmp1 tmp2 tmp3")


if __name__=="__main__":
    main()

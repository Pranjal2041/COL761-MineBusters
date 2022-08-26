import sys
import argparse


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


def check_output(*, expected, output):
    r"""
    :param expected - Path of file containing the expected output
    :param output - Path of file containing output of the user's program
    :returns boolean representing whether the two files match or not
    All the arguments must be provided as keyword arguments.
    
    The same output can be replicated using $ diff <expected> <output> -iwB
    except that in case of diff, it gives the exact changes required between two
    files and gives no output if the two files match.

    Hint: Use the testcase out_<n>.dat_sorted files as expected and output from
    your file as output parameter of this code.
    """
    expended = False
    outpended = False
    with open(expected, 'r') as expf, open(output, 'r') as outf:
        expit, outit = iter(expf), iter(outf)
        while True:
            explinlen = 0
            while explinlen == 0:
                try:
                    explin = next(expit)
                except StopIteration:
                    # expected file has ended
                    expended = True
                    explinlen = -1
                else:
                    explinlen = len(explin.strip())
            outlinlen = 0
            while outlinlen == 0:
                try:
                    outlin = next(outit)
                except StopIteration:
                    # output file has ended
                    outpended = True
                    outlinlen = -1
                else:
                    outlinlen = len(outlin.strip())
            if expended and outpended:
                break
            else:
                exptoks = [t.strip().upper() for t in explin.split()]
                outtoks = [t.strip().upper() for t in outlin.split()]
                # the above removes extra spaces and makes all uppercase
                expline = ' '.join(exptoks)
                outline = ' '.join(outtoks)
                if expline != outline:
                    return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", "-e", required=True, type=str)
    parser.add_argument("--output", "-o", required=True, type=str)
    if len(sys.argv) <= 1:
        print_usage()
        sys.exit()
    args = parser.parse_args()
    # import pdb;pdb.set_trace()
    print(check_output(**vars(args)))

if __name__ == "__main__":
    main()

import argparse
import logging
import pandas as pd
from matplotlib import pyplot as plt


from util import get_labels, make_graphs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_times(input_csv, output_png):

    df = pd.read_csv(input_csv, header=None)
    algo_l, _, time_l = [list(v.values()) for k, v in df.to_dict().items()]
    time_l = list(
        map(
            lambda real_time: int(real_time.split("m")[0]) * 60
            + float(real_time.split("m")[1][:-1]),
            time_l,
        )
    )

    fig, ax = plt.subplots()
    supports = [5, 10, 25, 50, 95]
    n = len(time_l)
    b = n // 3
    ax.plot(supports, time_l[:b], marker="o", label=algo_l[0])
    ax.plot(supports, time_l[b : 2 * b], marker="o", label=algo_l[b])
    ax.plot(supports, time_l[2 * b : 3 * b], marker="o", label=algo_l[2 * b])
    ax.set_xticks(supports)
    ax.set_xlabel("Suppport %")
    ax.set_ylabel("Time (s)")
    ax.legend()
    plt.savefig(output_png)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str)
    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("--plot-input", type=str)
    parser.add_argument("--plot-output", type=str)

    parser.add_argument("--to", type=str, choices=["fsg", "gspan", "gaston"])
    return parser.parse_args()


if __name__ == "__main__":

    args = parse()
    if args.plot_input and args.plot_output:
        plot_times(args.plot_input, args.plot_output)
    else:
        graphs = make_graphs(args.input)
        logger.info("Loaded %d graphs", len(graphs))
        if args.to == "gspan" or args.to == "gaston":
            vlabels, vlabel2id, elabels, elabel2id = get_labels(graphs)
            open(args.output, "wt").write(
                "\n".join(
                    [
                        graph.make_gspan_txt(vlabel2id=vlabel2id, elabel2id=elabel2id)
                        for graph in graphs.values()
                    ]
                )
                + "\n"
            )
        elif args.to == "fsg":
            open(args.output, "wt").write(
                "\n".join([graph.make_fsg_txt() for graph in graphs.values()])
            )

        logger.info("Converted and written to %s", args.output)

import argparse
import logging

from util import get_labels, make_graphs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str)
    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("--to", type=str, choices=["fsg", "gspan", "gaston"])
    return parser.parse_args()


if __name__ == "__main__":

    args = parse()
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

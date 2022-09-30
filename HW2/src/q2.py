import argparse
import pickle
import shutil
from util import Graph, check_vf3_subgraph, get_labels, make_graphs, mine_gspan
import os
import numpy as np
import tempfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_index_dirs(index_dir):
    shutil.rmtree(index_dir, ignore_errors=True)
    os.makedirs(index_dir, exist_ok=True)
    os.makedirs(os.path.join(index_dir, "tgraphs"), exist_ok=True)
    os.makedirs(os.path.join(index_dir, "subgraphs"), exist_ok=True)


def make_index(graph_dataset, index_dir, support):

    graphs = make_graphs(graph_dataset)
    logger.info("Read %d graphs" % len(graphs))

    vlabels, vlabel2id, elabels, elabel2id = get_labels(graphs)
    logger.info(
        "Found %d vertex labels and %d edge labels" % (len(vlabels), len(elabels))
    )

    gspan_file = tempfile.NamedTemporaryFile(mode="wt", delete=False)
    gspan_file.write(
        "\n".join(
            [
                g.make_gspan_txt(vlabel2id=vlabel2id, elabel2id=elabel2id)
                for g in graphs.values()
            ]
        )
    )
    gspan_file.close()

    subgraphs = mine_gspan(
        input_path=gspan_file.name,
        support=support,
        binary_path=os.path.abspath("../bin/gspan"),
        vlabels=vlabels,
        elabels=elabels,
    )

    logger.info("Mined %d frequent subgraphs" % len(subgraphs))

    tdir = os.path.join(index_dir, "tgraphs")
    subgraph_dir = os.path.join(index_dir, "subgraphs")
    for i, graph in enumerate(graphs.values()):
        open(os.path.join(tdir, f"{i}.txt"), "w").write(
            graph.make_vf3_txt(vlabel2id=vlabel2id, elabel2id=elabel2id)
        )
    logger.info("Dumped graphs in directory %s" % tdir)

    index_arr = np.zeros((len(graphs), len(subgraphs))).astype(int)
    for i, (subgraph, tid_list) in enumerate(subgraphs):
        open(os.path.join(subgraph_dir, f"{i}.txt"), "w").write(
            subgraph.make_vf3_txt(vlabel2id=vlabel2id, elabel2id=elabel2id)
        )
        index_arr[list(tid_list), i] = True

    logger.info("Saved subgraphs in directory: %s" % subgraph_dir)

    pickle.dump(
        {
            "vlabels": vlabels,
            "vlabel2id": vlabel2id,
            "elabels": elabels,
            "elabel2id": elabel2id,
            "tids": np.array(list(graphs.keys())),
            "index": index_arr,
        },
        open(os.path.join(index_dir, "meta.pkl"), "wb"),
    )

    logger.info("Saved index file to: %s", os.path.join(index_dir, "meta.pkl"))


def query_index(args, graph: Graph, **kwargs):
    index = kwargs["index"]
    subgraph_dir = os.path.join(args.index_dir, "subgraphs")
    tgraph_dir = os.path.join(args.index_dir, "tgraphs")
    subgraph_files = [os.path.join(subgraph_dir, x) for x in os.listdir(subgraph_dir)]

    f = tempfile.NamedTemporaryFile(mode="wt", delete=True)
    f.write(
        graph.make_vf3_txt(vlabel2id=kwargs["vlabel2id"], elabel2id=kwargs["elabel2id"])
    )
    f.flush()

    query_vector = np.array(
        [
            check_vf3_subgraph(x, f.name, vf3_binary=args.vf3_binary)
            for x in subgraph_files
        ]
    ).astype(int)

    cand_graphs = np.where((query_vector @ (index.T)) == query_vector.sum())[0]
    cand_graph_files = [os.path.join(tgraph_dir, f"{x}.txt") for x in cand_graphs]
    logger.info("Found %d candidate graphs" % len(cand_graphs))

    matches = np.array(
        [
            check_vf3_subgraph(f.name, x, vf3_binary=args.vf3_binary)
            for x in cand_graph_files
        ]
    ).astype(bool)

    # TODO what to do when there are no matches?

    logger.info("Found %d matches" % matches.sum())
    f.close()
    return kwargs["tids"][cand_graphs[matches]]


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--query_dataset", type=str, default="")
    parser.add_argument(
        "--graph_dataset", default=os.path.abspath("../data/Yeast/167.txt_graph")
    )
    parser.add_argument("--index_dir", default=os.path.abspath("../data/Yeast/index"))
    parser.add_argument("--output_file", default="out.txt")
    parser.add_argument("--vf3_binary", default=os.path.abspath("../bin/vf3"))
    parser.add_argument("--support", default=0.6, type=float)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse()
    if args.build:
        setup_index_dirs(args.index_dir)
        make_index(args.graph_dataset, args.index_dir, args.support)
    elif args.query_dataset:

        meta_dict = pickle.load(open(os.path.join(args.index_dir, "meta.pkl"), "rb"))
        query_graphs = make_graphs(args.query_dataset)
        output = "\n".join(
            [
                "\t".join(query_index(args, g, **meta_dict))
                for g in query_graphs.values()
            ]
        )
        open(args.output_file, "w").write(output)
    else:
        raise ValueError("Must specify either --build or --query")

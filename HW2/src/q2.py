import shutil
from util import get_labels, make_graphs, mine_gspan
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

    index_arr = np.zeros((len(graphs), len(subgraphs))).astype(bool)
    for i, (subgraph, tid_list) in enumerate(subgraphs):
        open(os.path.join(subgraph_dir, f"{i}.txt"), "w").write(
            subgraph.make_vf3_txt(vlabel2id=vlabel2id, elabel2id=elabel2id)
        )
        index_arr[list(tid_list), i] = True

    logger.info("Saved subgraphs in directory: %s" % subgraph_dir)

    np.savez(open(os.path.join(index_dir, "index.npz"), "wb"), index_arr)

    logger.info("Saved index file to: %s", os.path.join(index_dir, "index.npz"))


if __name__ == "__main__":
    graph_dataset = os.path.abspath("../data/Yeast/167.txt_graph")
    index_dir = os.path.join(os.path.dirname(graph_dataset), "index")
    support = 0.6

    setup_index_dirs(index_dir)
    make_index(graph_dataset, index_dir, support)

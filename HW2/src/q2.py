import argparse
import pickle
from re import T
import shutil

# from util import Graph, Vf3, check_vf3_subgraph, get_labels, make_graphs, mine_gspan
from util import Graph, Vf3, get_labels, make_graphs, mine_gspan, vfify_all_nodes

import os
import numpy as np
import tempfile
import logging
from util import Vf3
import time
from util import get_tree_ordering
from typing import List, Tuple, Dict
import ctypes
from collections import deque
import heapq


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_index_dirs(index_dir):
    shutil.rmtree(index_dir, ignore_errors=True)
    os.makedirs(index_dir, exist_ok=True)
    os.makedirs("tmp", exist_ok=True)
    os.makedirs(os.path.join(index_dir, "tgraphs"), exist_ok=True)
    os.makedirs(os.path.join(index_dir, "subgraphs"), exist_ok=True)


def make_index(graph_dataset, index_dir, support):

    graphs: Dict[str, Graph] = make_graphs(graph_dataset)
    logger.info("Read %d graphs" % len(graphs))

    vlabels, vlabel2id, elabels, elabel2id = get_labels(graphs)
    logger.info(
        "Found %d vertex labels and %d edge labels" % (len(vlabels), len(elabels))
    )

    # gspan_file = tempfile.NamedTemporaryFile(mode="wt", delete=False)
    gspan_file = open("tmp/tmp.txt", "w")

    gspan_file.write(
        "\n".join(
            [
                g.make_gspan_txt(vlabel2id=vlabel2id, elabel2id=elabel2id)
                for g in graphs.values()
            ]
        )
    )
    gspan_file.close()

    subgraphs: List[Tuple[Graph, List[int]]] = mine_gspan(
        input_path=gspan_file.name,
        support=support,
        # binary_path=os.path.abspath("../bin/gspan"),
        binary_path=os.path.abspath("../bin/gaston"),
        vlabels=vlabels,
        elabels=elabels,
        num_graphs=len(graphs),
    )
    try:
        tree_ordering = get_tree_ordering(subgraphs)
        print(len(tree_ordering))
    except:
        print("Error loading tree ordering! Will default to None instead")
        tree_ordering = None
    logger.info("Mined %d frequent subgraphs" % len(subgraphs))

    # tdir = os.path.join(index_dir, "tgraphs")
    # subgraph_dir = os.path.join(index_dir, "subgraphs")
    # for i, graph in enumerate(graphs.values()):
    #     open(os.path.join(tdir, f"{i}.txt"), "w").write(
    #         graph.make_vf3_txt(vlabel2id=vlabel2id, elabel2id=elabel2id)
    #     )
    # logger.info("Dumped graphs in directory %s" % tdir)

    index_arr = np.zeros((len(graphs), len(subgraphs))).astype(int)
    for i, (subgraph, tid_list) in enumerate(subgraphs):
        # open(os.path.join(subgraph_dir, f"{i}.txt"), "w").write(
        #     subgraph.make_vf3_txt(vlabel2id=vlabel2id, elabel2id=elabel2id)
        # )
        index_arr[list(tid_list), i] = True

    # logger.info("Saved subgraphs in directory: %s" % subgraph_dir)

    for x in subgraphs:
        x[0].vf3txt = x[0].make_vf3_txt(vlabel2id, elabel2id)
    for k in graphs:
        graphs[k].vf3txt = graphs[k].make_vf3_txt(vlabel2id, elabel2id)

    print("Sanity Check: ", tree_ordering[0].graph.vf3txt)
    pickle.dump(
        {
            "vlabels": vlabels,
            "vlabel2id": vlabel2id,
            "elabels": elabels,
            "elabel2id": elabel2id,
            "tids": np.array(list(graphs.keys())),
            "index": index_arr,
            "subgraphs": subgraphs,
            "tgraphs": graphs,
            "tree_ordering": tree_ordering,
        },
        open(os.path.join(index_dir, "meta.pkl"), "wb"),
        protocol=4,
    )

    logger.info("Saved index file to: %s", os.path.join(index_dir, "meta.pkl"))


class HeapNode(object):
    def __init__(self, node, supp):
        self.node = node
        self.supp = supp

    def __lt__(self, other):
        return self.supp < other.supp


def query_index(args, graph: Graph, **kwargs):
    graph.vf3txt = graph.make_vf3_txt(kwargs["vlabel2id"], kwargs["elabel2id"])
    graph.vf3txt = ctypes.c_char_p(graph.vf3txt.encode("utf-8"))
    index = kwargs["index"]
    # subgraph_dir = os.path.join(args.index_dir, "subgraphs")
    # tgraph_dir = os.path.join(args.index_dir, "tgraphs")
    # subgraph_files = [os.path.join(subgraph_dir, x) for x in os.listdir(subgraph_dir)]
    # subgraph_files.sort(key = lambda x: int(os.path.split(x)[-1].replace('.txt','')))

    subgraphs = kwargs["subgraphs"]
    vf3 = Vf3()
    tot_feats = len(subgraphs)
    feats_checked = 0

    support_iters = [0.6, 0.4, 0.25, 0.1, 0.0]
    support_indexes = 0
    cand_graph_files = None
    if kwargs["tree_ordering"] is not None:
        total_checks = 0

        to = kwargs["tree_ordering"]
        query_vector = np.zeros((len(subgraphs))).squeeze()
        # stack = []
        # stack = deque([x for x in to])
        stack = [HeapNode(x, -x.graph.support) for x in to]
        heapq.heapify(stack)
        # print(stack)

        while True:
            if len(stack) == 0:
                break
            # s = stack[-1]
            s = stack[0].node

            # stack.pop()
            # stack.popleft()
            heapq.heappop(stack)

            if s.graph.support < support_iters[support_indexes]:
                # We should possibly check for moving vf3 to tgraphs
                index_new = index[:, query_vector.nonzero()[0]]
                cand_graphs = index_new.all(axis=1).nonzero()[0]
                # TODO: Instead store pointers for number of elements with next higher support
                if (
                    len(cand_graphs) * 5 < tot_feats - total_checks
                    and len(cand_graphs) < 500
                ):
                    # No need to continue
                    cand_graph_files = [
                        kwargs["tgraphs"][kwargs["tids"][x]] for x in cand_graphs
                    ]
                    # print('Here is the stat',len(cand_graphs), tot_feats, total_checks)
                    break
                    # cand_graph_files = [kwargs['tgraphs'][kwargs['tids'][x]] for x in cand_graphs]
                    # Next break
                else:
                    support_indexes += 1
            total_checks += 1
            feats_checked += 1
            if vf3.is_subgraph_raw(
                s.graph.vf3txt, graph.vf3txt, kwargs["vlabel2id"], kwargs["elabel2id"]
            ):
                query_vector[s.i] = 1
                for node in s.neighbours:
                    # stack.append(node)
                    heapq.heappush(stack, HeapNode(node, -node.graph.support))
            else:
                feats_checked += s.counts
                # Do not consider the children of the given node
                pass
        logger.info(f"{total_checks}/{len(subgraphs)}")
    else:
        query_vector = np.array(
            [
                # check_vf3_subgraph(x, f.name, vf3_binary=args.vf3_binary)
                vf3.is_subgraph_raw(
                    x[0].vf3txt, graph.vf3txt, kwargs["vlabel2id"], kwargs["elabel2id"]
                )
                for x in subgraphs
            ]
        ).astype(int)
    # If cand_graphs is not None, we may not need to calc again
    if cand_graph_files is None:
        st = time.time()
        index_new = index[:, query_vector.nonzero()[0]]
        # print(index_new.all(axis = 1).shape)
        cand_graphs = index_new.all(axis=1).nonzero()[0]
        # cand_graphs = np.where((query_vector @ (index.T)) == query_vector.sum())[0]
        # cand_graph_files = [os.path.join(tgraph_dir, f"{x}.txt") for x in cand_graphs]
        cand_graph_files = [kwargs["tgraphs"][kwargs["tids"][x]] for x in cand_graphs]
        en = time.time()
        print("Time in cand calculation", en - st)

    logger.info("Found %d candidate graphs" % len(cand_graphs))

    matches = np.array(
        [
            vf3.is_subgraph_raw(
                graph.vf3txt, x.vf3txt, kwargs["vlabel2id"], kwargs["elabel2id"]
            )
            for x in cand_graph_files
        ]
    ).astype(bool)

    # TODO what to do when there are no matches?

    logger.info("Found %d matches" % matches.sum())
    # f.close()
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
    # parser.add_argument("--vf3_binary", default=os.path.abspath("../bin/vf3"))
    parser.add_argument("--vf3_binary", default=os.path.abspath("../vf3lib/bin/vf3"))

    parser.add_argument("--support", default=0.6, type=float)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse()
    if args.build:
        setup_index_dirs(args.index_dir)
        make_index(args.graph_dataset, args.index_dir, args.support)
    elif args.query_dataset:

        logger.info("Loading index")

        meta_dict = pickle.load(open(os.path.join(args.index_dir, "meta.pkl"), "rb"))
        for x in meta_dict["subgraphs"]:
            x[0].vf3txt = ctypes.c_char_p(x[0].vf3txt.encode("utf-8"))
        for k in meta_dict["tgraphs"]:
            meta_dict["tgraphs"][k].vf3txt = ctypes.c_char_p(
                meta_dict["tgraphs"][k].vf3txt.encode("utf-8")
            )
        logger.info("Index Loaded!")
        # for k in meta_dict['tree_ordering']:
        #     vfify_all_nodes(k)
        args.query_dataset = os.path.abspath(input("Enter the query graph file name: "))

        query_graphs = make_graphs(args.query_dataset)
        # output = "\n".join(
        #     [
        #         "\t".join(query_index(args, g, **meta_dict))
        #         for g in query_graphs.values()
        #     ]
        # )
        output = []
        all_times = []
        for g in query_graphs.values():
            st = time.time()
            output.append("\t".join(query_index(args, g, **meta_dict)))
            en = time.time()
            print("Time taken", en - st)
            all_times.append(en - st)
        print(
            f"Total Time taken: {sum(all_times)*1000.0}ms for {len(all_times)} queries"
        )

        output = "\n".join(output)

        open(args.output_file, "w").write(output)
    else:
        raise ValueError("Must specify either --build or --query_dataset")

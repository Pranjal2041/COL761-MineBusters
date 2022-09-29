import argparse
import sys
from typing import Dict, Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Graph:
    def __init__(self, tid) -> None:
        self.transaction_id: int = tid
        self.vertices: List[str] = []
        self.edges: List[Tuple[int, int, str]] = []

    def make_fsg_txt(self):
        txt = "t\n"
        txt += "\n".join([f"v {i} {v}" for i, v in enumerate(self.vertices)]) + "\n"
        txt += "\n".join(
            [f"u {min(i,j)} {max(i,j)} {label}" for i, j, label in self.edges]
        )
        txt += "\n"
        return txt

    def make_gspan_txt(self):
        vertex_labels = list(set(self.vertices))
        edge_labels = list(set([x[-1] for x in self.edges]))

        vlabel2id = {v: i for i, v in enumerate(vertex_labels)}
        elabel2id = {e: i for i, e in enumerate(edge_labels)}

        txt = "t\n"
        txt += (
            "\n".join([f"v {i} {vlabel2id[v]}" for i, v in enumerate(self.vertices)])
            + "\n"
        )
        txt += "\n".join(
            [
                f"e {min(i,j)} {max(i,j)} {elabel2id[label]}"
                for i, j, label in self.edges
            ]
        )
        txt += "\n"
        return txt


def make_graphs(filename):
    graphs: Dict[str, Graph] = {}

    lines = open(filename, "r").readlines()
    logger.info("Num lines: %d", len(lines))
    i = 0
    while i < len(lines):
        tid = lines[i][1:]
        num_vertices = int(lines[i + 1])
        graph = Graph(tid)
        i += 2
        for j in range(num_vertices):
            graph.vertices.append(lines[i + j].strip())
        i += num_vertices
        num_edges = lines[i]
        i += 1
        for j in range(int(num_edges)):
            src, dest, edge_label = lines[i + j].split()
            graph.edges.append((src, dest, edge_label))
        i += int(num_edges) + 1
        graphs[tid] = graph

    return graphs


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
    with open(args.output, "wt") as f:
        for tid, graph in graphs.items():
            if args.to == "fsg":
                f.write(graph.make_fsg_txt())
            elif args.to == "gspan":
                f.write(graph.make_gspan_txt())
            elif args.to == "gaston":
                f.write(graph.make_gspan_txt())
    logger.info("Converted and written to %s", args.output)

import re
from typing import Dict, Tuple, List
import logging
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_int(x):
    if isinstance(x, int):
        return x
    return hash(x) % ((1 << 31) - 1)


def get_labels(graphs):
    vertex_labels, edge_labels = set(), set()
    for graph in graphs.values():
        vertex_labels = vertex_labels.union(set(graph.vertices))
        edge_labels = edge_labels.union(set([e[2] for e in graph.edges]))
    vlabel2id = {v: i for i, v in enumerate(vertex_labels)}
    elabel2id = {v: i for i, v in enumerate(edge_labels)}
    return list(vertex_labels), vlabel2id, list(edge_labels), elabel2id


class Graph:
    def __init__(self, gid, support=0) -> None:
        self.gid: int = gid
        self.vertices: List[str] = []  # list of vertex labels
        self.edges: List[Tuple[int, int, str]] = []
        self.support: int = 0

    def make_fsg_txt(self):
        txt = "t\n"
        txt += "\n".join([f"v {i} {v}" for i, v in enumerate(self.vertices)]) + "\n"
        txt += "\n".join(
            [f"u {min(i,j)} {max(i,j)} {label}" for i, j, label in self.edges]
        )
        return txt

    def make_gspan_txt(self, vlabel2id, elabel2id):

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
        return txt

    def make_vf3_txt(self, vlabel2id, elabel2id):
        txt = f"{len(self.vertices)}\n"

        txt += (
            "\n".join([f"{i} {vlabel2id[v]}" for i, v in enumerate(self.vertices)])
            + "\n"
        )

        node_edges = [[] for _ in range(len(self.vertices))]
        for i, j, label in self.edges:
            node_edges[i].append((i, j, elabel2id[label]))
        for v in node_edges:
            txt += f"{len(v)}\n"
            if len(v) == 0:
                continue
            # BUG VF3 doesn't take edge label into account ???
            txt += "\n".join([f"{i} {j}" for i, j, _ in v]) + "\n"

        return txt

    def __len__(self):
        return len(self.vertices)


def make_graphs(filename):
    graphs: Dict[str, Graph] = {}

    lines = open(filename, "r").readlines()
    logger.info("Num lines: %d", len(lines))
    i = 0
    while i < len(lines):
        tid = lines[i][1:].strip()
        num_vertices = int(lines[i + 1].strip())
        graph = Graph(tid)
        i += 2
        for j in range(num_vertices):
            graph.vertices.append(lines[i + j].strip())
        i += num_vertices
        num_edges = int(lines[i].strip())
        i += 1
        for j in range(num_edges):
            src, dest, edge_label = lines[i + j].strip().split()
            graph.edges.append((int(src), int(dest), edge_label))
        i += int(num_edges) + 1
        graphs[tid] = graph

    return graphs


def mine_gspan(input_path, support, binary_path, vlabels, elabels):
    assert support > 0 and support <= 1
    output_path = input_path + ".fp"
    cmd = f"{binary_path} -f {input_path} -s {support} -o -i"
    logger.info("Running command: %s", cmd)
    p = subprocess.Popen(cmd, shell=True)
    ret_code = p.wait()
    if ret_code != 0:
        raise Exception("gspan failed")

    out_lines = open(output_path, "r").readlines()
    i = 0
    subgraphs: List[Tuple[Graph, List[int]]] = []
    while i < len(out_lines):
        gid, g_count = re.match(r"t # (\d+) \* (\d+)", out_lines[i]).groups()
        g = Graph(gid, support=g_count)
        i += 1
        while out_lines[i][0] == "v":
            _, _, v = out_lines[i].split()
            g.vertices.append(vlabels[(int(v))])
            i += 1
        while out_lines[i][0] == "e":
            _, src, dest, label = out_lines[i].split()
            g.edges.append((int(src), int(dest), elabels[int(label)]))
            i += 1
        transaction_list = list(map(int, out_lines[i].split()[1:]))
        subgraphs.append((g, set(transaction_list)))
        i += 2

    return subgraphs


def check_vf3_subgraph():
    pass
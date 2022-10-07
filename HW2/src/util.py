import re
from typing import Dict, Tuple, List
import logging
import subprocess
import ctypes

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
        self.support: int = support
        self.vf3txt = ""

    def is_subgraph_of(self, g):
        if set(self.vertices).issubset(g.vertices) and set(self.edges).issubset(
            g.edges
        ):
            return True

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
        # TODO handle unseen labels
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
            txt += "\n".join([f"{i} {j} {l}" for i, j, l in v]) + "\n"

        return txt

    def __len__(self):
        return len(self.vertices)


def make_graphs(filename) -> Dict[str, Graph]:
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


class TreeNode:
    def __init__(self, graph: Graph, i=-1):
        self.graph = graph
        self.neighbours = []
        self.i = i
        self.marked = False
        self.counts = -1

    def __repr__(self) -> str:
        string = f"Graph Id: {self.i}"
        if len(self.neighbours) > 0:
            string += f" ; Neighbours: {', '.join([str(x.i) for x in self.neighbours])}"
        if self.counts != -1:
            string += f" ; Counts: {self.counts}"
        return string

    def add_neighbour(self, g: Graph):
        self.neighbours.append(g)


def begin_order(graphs, i, prevT):
    prevG = prevT.graph
    if i >= len(graphs):
        return (None, i)
    curr_graph = graphs[i]
    # i += 1
    while curr_graph is not None:
        curr_graph = graphs[i]
        if prevG.is_subgraph_of(curr_graph):
            node = TreeNode(curr_graph, i)
            prevT.add_neighbour(node)
            curr_graph, i_new = begin_order(graphs, i + 1, node)
            # print(i_new, i)
            assert i_new > i
            i = i_new
        else:
            # We need to backtrack, and we will return back the curr node
            return (curr_graph, i)
    # print('We are out somehow🤔')
    return (curr_graph, i)


def vfify_all_nodes(node: TreeNode):
    print(node.graph.vf3txt)
    assert isinstance(node.graph.vf3txt, str)
    node.graph.vf3txt = ctypes.c_char_p(node.graph.vf3txt)
    # assert isinstance(node.graph.vf3txt, str)
    for n in node.neighbours:
        vfify_all_nodes(n)


def fill_counts(node):
    node.counts = 0
    for n in node.neighbours:
        node.counts += fill_counts(n) + 1
    return node.counts

def sort_tree_neighbours(node):
    node.neighbours.sort(key = lambda x : -x.graph.support)
    for node in node.neighbours:
        sort_tree_neighbours(node)


from pdb import set_trace as bp


def get_tree_ordering(graphs: List[Tuple[Graph, List[int]]]) -> List[TreeNode]:
    graphs = [x[0] for x in graphs]  # remove the tids
    i = 0
    root_trees = [TreeNode(graphs[0], 0)]

    while i < len(graphs):
        ll, i_new = begin_order(graphs, i + 1, root_trees[-1])
        print(i_new, i)
        assert i_new > i
        i = i_new
        if i >= len(graphs):
            break
        root_trees.append(TreeNode(ll, i))
        print(len(root_trees))
        # break
    for n in root_trees:
        fill_counts(n)
    print('Ordering Computation Done')
    for n in root_trees:
        sort_tree_neighbours(n)
    # bp()

    return root_trees


def mine_gspan(
    input_path, support, binary_path, vlabels, elabels, num_graphs=-1
) -> List[Tuple[Graph, List[int]]]:
    print("Starting gspan mining")
    assert support > 0 and support <= 1
    output_path = input_path + ".fp"
    # cmd = f"{binary_path} -f {input_path} -s {support} -o -i"
    cmd = f"{binary_path} {int(support * num_graphs)} {input_path} {output_path}"

    logger.info("Running command: %s", cmd)
    p = subprocess.Popen(cmd, shell=True)
    ret_code = p.wait()
    if ret_code != 0:
        raise Exception("gspan failed")

    out_lines = open(output_path, "r").readlines()
    # print(out_lines)
    i = 0
    subgraphs: List[Tuple[Graph, List[int]]] = []
    while i < len(out_lines):
        # print(i)
        # gid, g_count = re.match(r"t # (\d+) \* (\d+)", out_lines[i]).groups()
        gid = re.match(r"# (\d+)", out_lines[i]).groups()[0]
        # NOTE: gid is actually frequency for gaston
        # g = Graph(gid, support=g_count)
        # print(gid)
        g = Graph(gid, support  = int(gid) / num_graphs)

        i += 1
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
        # i += 1

    return subgraphs


# def check_vf3_subgraph(subgraph_file, graph_file, vf3_binary):
#     cmd = f"{vf3_binary} {subgraph_file} {graph_file} -u -r 0"
#     logger.info("Running command: %s", cmd)

#     p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
#     ret_code = p.wait()
#     if ret_code != 0:
#         raise Exception("vf3 failed")
#     out = p.stdout.read().decode("utf-8")
#     # print(out)
#     # 3/0
#     return int(out.split()[0]) > 0
class Vf3:
    def __init__(self, lib_path="../bin/vf3.so") -> None:
        self.lib_path = lib_path
        self.lib = ctypes.cdll.LoadLibrary(lib_path)

    def is_subgraph_raw(
        self,
        patt: ctypes.c_char_p,
        graph: ctypes.c_char_p,
        vlabel2id=None,
        elabel2id=None,
    ):
        num_sols = self.lib.test(
            patt,
            graph,
        )
        return num_sols > 0

    def is_subgraph(self, patt: Graph, graph: Graph, vlabel2id, elabel2id):
        # import time
        # st = time.time()
        patt_txt = patt.make_vf3_txt(vlabel2id, elabel2id)
        graph_txt = graph.make_vf3_txt(vlabel2id, elabel2id)
        patt_txt_ctype = ctypes.c_char_p(patt_txt.encode("utf-8"))
        graph_txt_ctype = ctypes.c_char_p(graph_txt.encode("utf-8"))
        # en = time.time()
        # print('TIme in all conversions', (en - st)*1000,'ms')
        return self.is_subgraph_raw(patt_txt_ctype, graph_txt_ctype)


# def check_vf3_subgraph(subgraph: Graph, graph: Graph, vf3_binary):
# cmd = f"{vf3_binary} {subgraph_file} {graph_file}"
# logger.info("Running command: %s", cmd)

# p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
# ret_code = p.wait()
# if ret_code != 0:
#     raise Exception("vf3 failed")

# out = p.stdout.read().decode("utf-8")
# return int(out.split()[0]) > 0

from builtins import int

from networkx import Graph
from torch.utils.data import Sampler
from typing import Iterator
import random
from math import floor
import networkx as nx
from networkx import Graph

HERMESSIM_NPOS = 10


class SoftBatchPairSampler(Sampler[list[int]]):
    """
    This class is specialized to generate batches where the nodes sampled can have links in common (soft positive pairs).
    """

    def __init__(
        self,
        edge_list: list[tuple[str, str]],
        node_to_rid: dict[str, int],
        batch_size: int,
        seed: int = 0,
        static: bool = False,
    ):
        self.edge_list: list[tuple[str, str]] = edge_list
        self.node_to_rid: dict[str, int] = node_to_rid
        self.batch_size: int = batch_size
        self.seed: int = seed
        self.static: bool = static
        # if the batch_size is odd (bad choice) we floor the number of nodes to sample
        self.n: int = self.batch_size // 2

        self._reset_seed()

    def _reset_seed(self) -> None:
        random.seed(self.seed)

    def __len__(self) -> int:
        return floor(len(self.edge_list) / self.n)

    def __iter__(self) -> Iterator[list[int]]:
        # We prefer to generate batches on demand
        edge_list: list[tuple[str, str]] = self.edge_list.copy()
        if self.static:
            self._reset_seed()
        random.shuffle(edge_list)
        start = 0
        edge_list_len: int = len(edge_list)
        while (start + self.n) <= edge_list_len:  # > 0:
            sampled_nodes: list[str] = [
                endpoint
                for edge in edge_list[start : start + self.n]
                for endpoint in edge
            ]
            batch: list[int] = list(map(lambda x: self.node_to_rid[x], sampled_nodes))
            random.shuffle(batch)
            start += self.n
            yield batch


class StrongBatchPairSampler(Sampler[list[int]]):
    """
    This class is specialized to generate batches where the negative samples have no relation with a specific node in the same batch (strong positive pairs).
    """

    def __init__(
        self,
        G: Graph,
        node_to_rid: dict[str, int],
        pool_size: int,
        seed: int = 0,
    ):
        self.G: Graph = G
        self.node_to_rid: dict[str, int] = node_to_rid
        self.pool_size: int = pool_size
        self.seed: int = seed
        self.src_nodes: list = [
            node for node in self.G.nodes() if self.G.degree(node) > 1
        ]

    def __len__(self) -> int:
        return HERMESSIM_NPOS

    def __iter__(self) -> Iterator[list[int]]:
        random.seed(self.seed)
        # fix the budget to HERMESSIM_NPOS * pool_size samples
        for src_node in list(self.src_nodes)[:HERMESSIM_NPOS]:
            sampled_nodes: list = [src_node]
            # randomly pick a source node and then remove the neighbors of the src_nodes
            neighbors = list(self.G.neighbors(src_node))
            sampled_nodes.append(random.sample(neighbors, k=1)[0])
            # generate a graph view removing the neighbors of the src_node
            view: Graph = nx.subgraph_view(
                self.G, filter_node=lambda x: x not in sampled_nodes
            )
            sampled_nodes.extend(random.sample(list(view.nodes()), self.pool_size - 1))
            yield list(map(lambda node: self.node_to_rid[node], sampled_nodes))

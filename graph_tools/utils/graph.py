from __future__ import annotations

import json
import os
import re
from abc import ABC
from collections import defaultdict
from datetime import date, datetime, timedelta
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import dgl
import networkx as nx
import numpy as np
import torch
from torch.utils.data import Dataset

from graph_tools.datasets.base import Label, TimeWindow
from graph_tools.utils.utils import is_notebook

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class CollateFnType(Enum):
    GRAPH = "GRAPH"
    NX = "NX"
    ADJ_MATRIX = "ADJ_MATRIX"


class GraphComponent(ABC):
    def __init__(self, name: str, items: List[int]):
        self.__name: str = name
        self.__items: Set[int] = set(items)
        self.attrs: Dict[str, Any] = {}
        self.label: Label = Label.BENIGN

    @property
    def name(self) -> str:
        return self.__name

    @property
    def items(self) -> List[int]:
        return list(self.__items)

    def __hash__(self):
        return hash(self.name)

    def add_items(self, items: Union[str, List[int]]):
        if isinstance(items, list):
            self.__items = self.__items.union(set([item for item in items]))
        elif isinstance(items, int):
            self.__items.add(items)
        else:
            raise RuntimeError(f"Unexpected argument: items ({type(items)})")


class Node(GraphComponent):
    def __init__(self, name: str, items: List[int] = []):
        super().__init__(name, items)

    def __str__(self):
        return f"<Node name:{self.name}>"

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.name == other.name
        else:
            return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "items": self.items,
        }

    @classmethod
    def load_dict(cls, data: Dict[str, Any]) -> Node:
        name = data["name"]
        items = data["items"]
        return cls(name, items)


class Edge(GraphComponent):
    def __init__(
        self,
        name: str,
        source: Node,
        destination: Node,
        weight: float = 0.0,
        label: Label = Label.BENIGN,
    ):
        super().__init__(name, [])
        self.__source: Node = source
        self.__destination: Node = destination
        self.weight: float = weight

    def __str__(self):
        return f"<Edge name:{self.name}>"

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.name + str(self.source) + str(self.destination))

    def __eq__(self, other):
        if isinstance(other, Edge):
            return hash(self) == hash(other)
        else:
            return False

    @property
    def source(self) -> Node:
        return self.__source

    @property
    def destination(self) -> Node:
        return self.__destination

    def get_weight(self, agg_func: Callable) -> float:
        return agg_func(self.weight)

    @property
    def src_name(self) -> str:
        return self.source.name

    @property
    def dst_name(self) -> str:
        return self.destination.name

    @classmethod
    def from_nodes(cls, src: Node, dst: Node, weight: float = 0.0) -> Edge:
        return cls(name=f"{src.name}->{dst.name}", source=src, destination=dst, weight=weight)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source.to_dict(),
            "destination": self.destination.to_dict(),
            "weights": self.weight,
        }

    @classmethod
    def load_dict(cls, data: Dict[str, Any]) -> Edge:
        name = data["name"]
        source = Node.load_dict(data["source"])
        destination = Node.load_dict(data["destination"])
        weights = data["weights"]
        return cls(name, source, destination, weights)


class Graph(object):
    def __init__(self, time_window: TimeWindow = TimeWindow(datetime.now(), datetime.now(), timedelta(seconds=1))):
        self.nodes: Set[Node] = set()
        self.edges: Set[Edge] = set()
        self.time_window: TimeWindow = time_window.copy()

    def __str__(self) -> str:
        return f"<Graph at:{str(self.time_window)} nodes:{len(self.nodes)} edges:{len(self.edges)}>"

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def timestamp(self) -> datetime:
        return self.time_window.start

    @property
    def node_attr_names(self) -> List[str]:
        node = list(self.nodes)[0]
        return list(node.attrs.keys())

    @property
    def edge_attr_names(self) -> List[str]:
        edge = list(self.edges)[0]
        return list(edge.attrs.keys())

    def get_node(self, name: str) -> Optional[Node]:
        if isinstance(name, str):
            name_dict: Dict[str, Node] = {node.name: node for node in self.nodes}
            return name_dict.get(name, None)
        else:
            raise RuntimeError(f"Unexpected type of the key: {type(name)}")

    def get_edge(self, name: Union[str, Tuple[Union[str, Node], Union[str, Node]]]) -> Optional[Edge]:
        if isinstance(name, str):
            name_dict: Dict[str, Edge] = {edge.name: edge for edge in self.edges}
            return name_dict.get(name, None)
        elif isinstance(name, tuple):
            if isinstance(name[0], str) and isinstance(name[1], str):
                for edge in self.edges:
                    if edge.source.name == name[0] and edge.destination.name == name[1]:
                        return edge
            elif isinstance(name[0], Node) and isinstance(name[1], Node):
                for edge in self.edges:
                    if edge.source == name[0] and edge.destination == name[1]:
                        return edge
            else:
                raise RuntimeError(f"Unexpected type of the keys: {type(name[0])}, {type(name[1])}")
        else:
            raise RuntimeError(f"Unexpected key of type: {type(name)}")
        return None

    def add_node(self, new_node: Node) -> Node:
        node: Optional[Node] = self.get_node(new_node.name)
        if node is None:
            self.nodes.add(new_node)
            return new_node
        else:
            node.add_items(new_node.items)
            return node

    def add_edge(self, new_edge: Edge):
        edge: Optional[Edge] = self.get_edge(new_edge.name)
        if edge is None:
            self.edges.add(new_edge)
            self.add_node(new_edge.source)
            self.add_node(new_edge.destination)
            return new_edge
        else:
            edge.add_items(new_edge.items)
            self.add_node(edge.source)
            self.add_node(edge.destination)
            return edge

    def get_node_dim(self) -> Dict[Node, int]:
        node_dict: Dict[Node, int] = {node: 0 for node in self.nodes}
        for edge in self.edges:
            node_dict[edge.source] += 1
            node_dict[edge.destination] += 1
        return node_dict

    def get_node_dim_centrality(self) -> Dict[Node, float]:
        node_dim: Dict[Node, int] = self.get_node_dim()
        centrality: Dict[Node, float] = {}
        if len(self.nodes) == 1:
            for node, dim_value in node_dim.items():
                centrality[node] = 1
        else:
            for node, dim_value in node_dim.items():
                centrality[node] = dim_value / (len(self.nodes) - 1)
        return centrality

    def to_nx_graph(self, edge_weight_agg_func: Callable = np.mean):
        G = nx.DiGraph()
        G.add_nodes_from([node.name for node in self.nodes])
        G.add_weighted_edges_from(
            [(edge.source.name, edge.destination.name, edge.get_weight(edge_weight_agg_func)) for edge in self.edges]
        )
        return G

    def to_dgl_graph(self) -> Tuple[dgl.graph, Dict[Node, int]]:
        node2index: Dict[Node, int] = {node: index for index, node in enumerate(self.nodes)}
        src_nodes: List[int] = [node2index[e.source] for e in self.edges]
        dst_nodes: List[int] = [node2index[e.destination] for e in self.edges]
        G = dgl.graph((src_nodes, dst_nodes), num_nodes=len(node2index))

        # set edge attributes
        edge_attrs_to_set: torch.Tensor = torch.rand(len(self.edges), len(self.edge_attr_names))
        for edge_index, edge in enumerate(self.edges):
            for attr_index, (attr_name, value) in enumerate(edge.attrs.items()):
                if value == "":
                    value = 0.0
                assert isinstance(value, int) or isinstance(value, float), f"Unexpected type {value} ({type(value)})"
                edge_attrs_to_set[edge_index, attr_index] = float(value)
        G.edata["features"] = edge_attrs_to_set

        # set node attributes
        node_attrs_to_set: torch.Tensor = torch.zeros(len(self.nodes), len(self.node_attr_names))
        for node_index, node in enumerate(self.nodes):
            for attr_index, (attr_name, value) in enumerate(node.attrs.items()):
                if value == "":
                    value = 0.0
                assert isinstance(value, int) or isinstance(value, float), f"Unexpected type {value} ({type(value)})"
                node_attrs_to_set[node_index, attr_index] = float(value)
        G.ndata["features"] = node_attrs_to_set

        return G, node2index

    def to_dict(self):
        return {
            "time_window": self.time_window.to_dict(),
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
        }

    @staticmethod
    def load_dict(data: Dict):
        graph = Graph(time_window=TimeWindow.load_dict(data["time_window"]))
        graph.nodes = set(
            [Node.load_dict(node_dict) for node_dict in tqdm(data["nodes"], desc="loading nodes", leave=False)]
        )
        graph.edges = set(
            [Edge.load_dict(edge_dict) for edge_dict in tqdm(data["edges"], desc="loading edges", leave=False)]
        )
        return graph


class GraphData(object):
    def __init__(
        self,
        name: str,
        start: datetime = datetime.now(),
        end: datetime = datetime.now(),
        time_window_size: timedelta = timedelta(0),
        time_window_delta: timedelta = timedelta(0),
    ):
        self.name = name
        self.start = start
        self.end = end
        self.time_window_size = time_window_size
        self.time_window_delta = time_window_delta
        self.graphs: Dict[TimeWindow, Graph] = {}
        self.node_all: Dict[TimeWindow, Set[Node]] = defaultdict(set)

        self.__initialize(start, end, time_window_size, time_window_delta)

    def __initialize(self, start: datetime, end: datetime, time_window_size: timedelta, time_window_delta: timedelta):
        """initialize data: Dict[TimeWindow, Graph]

        Args:
            start (datetime): start time of the dataset
            end (datetime): end time of the dataset
        """
        tw: TimeWindow = TimeWindow(start, start + time_window_size, time_window_delta)
        while tw.end <= end:
            self.graphs[tw.copy()] = Graph(tw)
            self.node_all[tw.copy()] = set()
            tw = tw.next_window()

    def to_dict(self) -> Dict[str, Any]:
        # graph -> dict
        graph_data = []
        for time_window, graph in tqdm(self.graphs.items(), total=len(self.graphs), desc="writing graphs", leave=False):
            graph_data.append({"time_window": time_window.to_dict(), "graph": graph.to_dict()})

        # time-windows -> dict
        tw_data = [
            {"time_window": tw.to_dict(), "node_all": [n.to_dict() for n in h_list]}
            for tw, h_list in tqdm(
                self.node_all.items(), total=len(self.node_all), desc="writing timewindows", leave=False
            )
        ]

        # meta data -> dict
        meta_data = {
            "name": self.name,
            "time_window_size": self.time_window_size.total_seconds(),
            "time_window_delta": self.time_window_delta.total_seconds(),
        }

        return {
            "graph_data": graph_data,
            "time_window_data": tw_data,
            "meta_data": meta_data,
        }

    @classmethod
    def load_dict(cls, dict_data: Dict[str, Any]) -> GraphData:
        # load meta data
        meta_data = dict_data["meta_data"]
        graph_data = cls(
            name=meta_data["name"],
            time_window_size=timedelta(seconds=meta_data["time_window_size"]),
            time_window_delta=timedelta(seconds=meta_data["time_window_delta"]),
        )

        # load graph
        for gd_item in tqdm(dict_data["graph_data"], desc="loading graph", leave=False):
            tw = TimeWindow.load_dict(gd_item["time_window"])
            graph = Graph.load_dict(gd_item["graph"])
            graph_data.graphs[tw] = graph

        # load timewindows
        for tw_item in tqdm(dict_data["time_window_data"], desc="loading timewindows", leave=False):
            tw = TimeWindow.load_dict(tw_item["time_window"])
            node_list = [Node.load_dict(node_data) for node_data in tw_item["node_all"]]
            graph_data.node_all[tw] = set(node_list)

        return graph_data

    def get_time_window(self, t: datetime) -> TimeWindow:
        for tw in self.node_all.keys():
            if tw.isin(t):
                return tw
        raise RuntimeError(f"Unexpected datetime object: {t}")

    def add_node(self, node: Node, time_window: TimeWindow):
        self.node_all[time_window].add(node)
        self.graphs[time_window].add_node(node)

    def add_edge(self, edge: Edge, time_window: TimeWindow):
        self.add_node(edge.source, time_window)
        self.add_node(edge.destination, time_window)
        self.graphs[time_window].add_edge(edge)

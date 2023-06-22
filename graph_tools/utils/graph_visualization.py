from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler

from graph_tools.datasets.base import Label
from graph_tools.utils.graph import Graph, Node
from graph_tools.utils.utils import is_notebook

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

GREEN = "#00CC96"
RED = "#EF554B"


def to_rgb(scores: np.ndarray) -> np.ndarray:
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_scores = scaler.fit_transform(scores.reshape(-1, 1))

    def r(value: float) -> int:
        if value < 0.5:
            return 255
        else:
            return int(-510 * value + 510)

    def g(value: float) -> int:
        if value < 0.5:
            return int(510 * value)
        else:
            return 255

    def b(value: float) -> int:
        if value < 0.5:
            return int(510 * value)
        else:
            return int(-510 * value + 510)

    to_r = np.vectorize(r)
    to_g = np.vectorize(g)
    to_b = np.vectorize(b)

    rgb = np.array([to_r(scaled_scores), to_g(scaled_scores), to_b(scaled_scores)]).reshape((len(scores), 3))
    return rgb


def to_display(attrs: List[Dict[str, Any]], key: str, default: Any, m: int) -> str:
    text_items = [str(attr.get(key, default)) for attr in attrs]
    text = ", ".join(list(set(text_items))[:m])
    if len(list(set(text_items))) > m:
        text += "..."
    return text


def draw_graph(
    graph: Graph,
    targets: List[Label] = [Label.BENIGN, Label.ANOMALY],
    node_anomaly: bool = True,
    pos: Dict[str, Tuple[float, float]] = {},
    scores: List[float] = [],
    **kwargs,
):
    """draw graph

    Args:
        graph (Graph): Graph object
        targets (List[Label], optional): Defaults to [Label.BENIGN, Label.ANOMALY].
        pos (Dict[str, Tuple[float, float]], optional): Defaults to {}.
        scores (List[float], optional): Defaults to [].

    Examples:
        >>> node_data = [
        >>>     go.Scatter(
        >>>         x=benign_node_df["x"], y=benign_node_df["y"], name="benign node",
        >>>         textposition="bottom center", mode="markers", hovertemplate="<i>%{text}</i>",
        >>>         marker=dict(size=20, line=dict(width=2), symbol="circle", color="#059E3F"),
        >>>     ),
        >>>     go.Scatter(
        >>>         x=anomaly_node_df["x"], y=anomaly_node_df["y"], name="anomaly node",
        >>>         textposition="bottom center", mode="markers", hovertemplate="<i>%{text}</i>",
        >>>         marker=dict(size=20, line=dict(width=2), symbol="x", color="#EF553B"),
        >>>     )
        >>> ]
        >>> edge_data = [
        >>>     go.Scatter(
        >>>         x=benign_edge_x, y=benign_edge_y, mode="lines",
        >>>         line=dict(width=2, color="#00CC96"), name="benign edge"
        >>>     ),
        >>>     go.Scatter(
        >>>         x=anomaly_edge_x, y=anomaly_edge_y, mode="lines",
        >>>         line=dict(width=2, color="#EF553B"), name="anomaly edge",
        >>>     )
        >>> ]
        >>> benign_annotations = [
        >>>    dict(
        >>>        x=head[0], y=head[1], showarrow=True, xref="x", yref="y", arrowcolor="#00CC96", arrowsize=1.2,
        >>>        arrowwidth=2.0, opacity=0.7, ax=tail[0], ay=tail[1], axref="x", ayref="y", arrowhead=2,
        >>>    )
        >>>    for head, tail in zip(benign_head_list, benign_tail_list)
        >>> ]
        >>> anomaly_annotations = [
        >>>     dict(
        >>>         x=head[0], y=head[1], showarrow=True, xref="x", yref="y", arrowcolor="#EF553B", arrowsize=1.2,
        >>>         arrowwidth=2.0, opacity=0.7, ax=tail[0], ay=tail[1], axref="x", ayref="y", arrowhead=2,
        >>>     )
        >>>     for head, tail in zip(anomaly_head_list, anomaly_tail_list)
        >>> ]
        >>> fig = go.Figure()
        >>> fig.add_traces(node_data + edge_data)
        >>> for annotation in benign_annotations:
        >>>     fig.add_annotation(**annotation)
        >>> for annotation in anomaly_annotations:
        >>>     fig.add_annotation(**annotation)
        >>> fig.update_layout(
        >>>     showlegend=True,
        >>>     xaxis=dict(showticklabels=False),
        >>>     yaxis=dict(showticklabels=False),
        >>> )
        >>> fig.show()

    Raises:
        RuntimeError: _description_

    Returns:
        components: (
            benign_node_df, anomaly_node_df,
            benign_edge_x, benign_edge_y, anomaly_edge_x, anomaly_edge_y,
            benign_head_list, benign_tail_list, anomaly_head_list, anomaly_tail_list,
        )
    """
    G: nx.Graph = graph.to_nx_graph()

    # get node positions
    if len(pos) == 0:
        k = kwargs["k"] if "k" in kwargs else 0.2
        pos = nx.spring_layout(G, k=k, seed=42)
        for n in G.nodes():
            G.nodes[n]["pos"] = pos[n]
    elif len(pos) == len(G.nodes):
        for n in G.nodes():
            G.nodes[n]["pos"] = pos[n]
    else:
        raise RuntimeError("Unexpected length of node position data.")

    benign_node, benign_edge_x, benign_edge_y = [], [], []
    anomaly_node, anomaly_edge_x, anomaly_edge_y = [], [], []
    benign_head_list, benign_tail_list = [], []
    anomaly_head_list, anomaly_tail_list = [], []

    for n in G.nodes():
        node = graph.get_node(n)
        x, y = G.nodes[n]["pos"]
        assert node is not None

        if (node.label in [Label.BENIGN, Label.UNDEFINED]) or (len(targets) == 0):
            benign_node.append({"x": x, "y": y, "name": node.name, "size": 20})
        else:
            anomaly_node.append({"x": x, "y": y, "name": node.name, "size": 20})

    for e in G.edges():
        x0, y0 = G.nodes[e[0]]["pos"]
        x1, y1 = G.nodes[e[1]]["pos"]

        # add node
        src_node = graph.get_node(e[0])
        dst_node = graph.get_node(e[1])
        edge = graph.get_edge(e)
        assert (src_node is not None) and (dst_node is not None) and (edge is not None)

        if (src_node.label in [Label.BENIGN, Label.UNDEFINED]) or (len(targets) == 0):
            benign_node.append({"x": x0, "y": y0, "name": e[0], "size": 20})
        else:
            anomaly_node.append({"x": x0, "y": y0, "name": e[0], "size": 20})

        if (dst_node.label in [Label.BENIGN, Label.UNDEFINED]) or (len(targets) == 0):
            benign_node.append({"x": x1, "y": y1, "name": e[1], "size": 20})
        else:
            anomaly_node.append({"x": x1, "y": y1, "name": e[1], "size": 20})

        if node_anomaly:
            src_condition = src_node.label in [Label.BENIGN, Label.MIXED, Label.UNDEFINED]
            dst_condition = dst_node.label in [Label.BENIGN, Label.MIXED, Label.UNDEFINED]
            condition = src_condition and dst_condition
        else:
            condition = edge.label in [Label.BENIGN, Label.UNDEFINED]

        if condition or (len(targets) == 0):
            benign_edge_x.append(x0)
            benign_edge_y.append(y0)
            benign_edge_x.append(x1)
            benign_edge_y.append(y1)
            benign_edge_x.append(None)
            benign_edge_y.append(None)
            benign_head_list.append((x1, y1))
            benign_tail_list.append((x0, y0))
        else:
            anomaly_edge_x.append(x0)
            anomaly_edge_y.append(y0)
            anomaly_edge_x.append(x1)
            anomaly_edge_y.append(y1)
            anomaly_edge_x.append(None)
            anomaly_edge_y.append(None)
            anomaly_head_list.append((x1, y1))
            anomaly_tail_list.append((x0, y0))

    benign_node_df = pd.DataFrame(benign_node)
    anomaly_node_df = pd.DataFrame(anomaly_node)

    return (
        benign_node_df,
        anomaly_node_df,
        benign_edge_x,
        benign_edge_y,
        anomaly_edge_x,
        anomaly_edge_y,
        benign_head_list,
        benign_tail_list,
        anomaly_head_list,
        anomaly_tail_list,
    )

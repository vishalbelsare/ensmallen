from time import time
from humanize import naturaldelta
from ensmallen_graph import EnsmallenGraph  # pylint: disable=no-name-in-module
import compress_json
import json
import numpy as np
from pympler import asizeof

start = time()
graph = EnsmallenGraph(
    edge_path="../graph/cooccurrence/edges.tsv",
    sources_column="subject",
    destinations_column="object",
    directed=False,
    validate_input_data=True
)
completed_graph = time() - start
start_walk = time()

walks = graph.walk(
    iterations=1,
    length=80,
    min_length=0,
    return_weight=1,
    explore_weight=1,
    change_node_type_weight=1,
    change_edge_type_weight=1
)
delta = time() - start
total_walk_time = time() - start_walk

mean_walks_length = np.mean([
    len(walk) for walk in walks
])

median_walks_length = np.median([
    len(walk) for walk in walks
])

graph_memory_size = asizeof.asizeof(graph)
walks_memory_size = asizeof.asizeof(walks)

response = {
    "directory": "monarch",
    "total_required_time": delta,
    "building_graph_required_time": completed_graph,
    "random_walk_time": total_walk_time,
    "mean_walks_length": mean_walks_length,
    "median_walks_length": median_walks_length,
    "traps_rate": graph.traps_rate(),
    "graph_size": graph_memory_size,
    "walks_size": walks_memory_size,
    "mean_outbound_edges": np.mean(graph.outbounds),
    "median_outbound_edges": np.median(graph.outbounds),
    "nodes": graph.get_nodes_number(),
    "edges": graph.get_edges_number()
}

print(json.dumps(response, indent=4))

compress_json.dump(response, "time_required.json", json_kwargs={"indent": 4})

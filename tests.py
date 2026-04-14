import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import networkx as nx

import test

start_node = '101S'
end_node = '120S'

try:
    # Calculate the shortest path using ONLY the edge weights (time)
    path = nx.shortest_path(G, source=start_node, target=end_node, weight='weight')
    total_time = nx.shortest_path_length(G, source=start_node, target=end_node, weight='weight')
    
    print(f"SUCCESS! Path found: {path}")
    print(f"Total commute time: {total_time / 60:.1f} minutes")
    
except nx.NetworkXNoPath:
    print("FAIL: The graph is broken. There is no path between these two stops.")
except nx.NodeNotFound as e:
    print(f"FAIL: {e}")
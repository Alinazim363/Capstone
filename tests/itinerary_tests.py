import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import networkx as nx

import NetworkGraph
G = NetworkGraph.G

start_node = '101S'
end_node = '120S'

def print_directions(G, path):
    print("ROUTE ITINERARY:")
    print("-------------------")
    
    current_leg_type = None
    current_route = None
    stops_count = 0
    leg_time = 0
    start_station_name = G.nodes[path[0]]['name']
    
    for i in range(len(path) - 1):
        u = path[i]
        v = path[i+1]
        edge_data = G.get_edge_data(u, v)
        
        edge_type = edge_data['edge_type']
        weight = edge_data['weight']
        
        # scenario A: We are walking
        if edge_type == 'walking_transfer':
            if current_leg_type == 'subway_transit':
                print(f"Take the {current_route} train for {stops_count} stops ({leg_time/60:.1f} mins)")
                print(f"   ↳ Get off at {G.nodes[u]['name']}")
                
            print(f"Walk to {G.nodes[v]['name']} ({weight/60:.1f} mins)")
            
            # reset trackers for the next leg
            current_leg_type = 'walking_transfer'
            stops_count = 0
            leg_time = 0
            start_station_name = G.nodes[v]['name']
            
        # scenario B: We are on a subway
        elif edge_type == 'subway_transit':
            available_routes = edge_data['routes']
            
            # if we are just starting, or switching from walking, pick the first available train
            if current_leg_type != 'subway_transit' or current_route not in available_routes:
                if current_leg_type == 'subway_transit':
                    print(f"Take the {current_route} train for {stops_count} stops ({leg_time/60:.1f} mins)")
                    print(f"   ↳ Transfer at {G.nodes[u]['name']}")
                else:
                    print(f"Board at {start_station_name}")
                
                # Start the new train leg
                current_route = available_routes[0] 
                current_leg_type = 'subway_transit'
                stops_count = 0
                leg_time = 0
                
            stops_count += 1
            leg_time += weight

    if current_leg_type == 'subway_transit':
        print(f"Take the {current_route} train for {stops_count} stops ({leg_time/60:.1f} mins)")
    
    print(f"Arrive at {G.nodes[path[-1]]['name']}")
    print("-------------------")

try:
    # calculate the shortest path using ONLY the edge weights (time)
    path = nx.shortest_path(G, source=start_node, target=end_node, weight='weight')
    total_time = nx.shortest_path_length(G, source=start_node, target=end_node, weight='weight')
    
    print(f"SUCCESS! Path found: {path}")
    print(f"Total commute time: {total_time / 60:.1f} minutes")
    
except nx.NetworkXNoPath:
    print("FAIL: The graph is broken. There is no path between these two stops.")
except nx.NodeNotFound as e:
    print(f"FAIL: {e}")

print_directions(G, path)
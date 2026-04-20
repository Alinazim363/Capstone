import networkx as nx
import pandas as pd

from NetworkGraph import G

# starting locations
user1_start = '101S' 
user2_start = 'F24N'

# generate the Isochrones
x1_travel_times = nx.single_source_dijkstra_path_length(G, source=user1_start, weight='weight')
x2_travel_times = nx.single_source_dijkstra_path_length(G, source=user2_start, weight='weight')

print(f"User 1 can reach {len(x1_travel_times)} stations.")
print(f"User 2 can reach {len(x2_travel_times)} stations.")

fair_venues = []

MAX_COMMUTE = 60 
MAX_DELTA = 15 # difference in commute times 

for node in G.nodes():
    if node in x1_travel_times and node in x2_travel_times:
        
        # convert seconds to minutes
        time_j = x1_travel_times[node] / 60 
        time_k = x2_travel_times[node] / 60 
        
        # calculate the Fairness metrics
        delta = abs(time_j - time_k)
        max_time = max(time_j, time_k)
        
        # apply the Fairness Filter
        if delta <= MAX_DELTA and max_time <= MAX_COMMUTE:
            fair_venues.append({
                'stop_id': node,
                'name': G.nodes[node]['name'],
                'lat': G.nodes[node]['lat'],
                'lon': G.nodes[node]['lon'],
                'jack_time': round(time_j, 1),
                'jane_time': round(time_k, 1),
                'delta': round(delta, 1),
                'max_time': round(max_time, 1)
            })

fairness_df = pd.DataFrame(fair_venues)
fairness_df = fairness_df.sort_values(by=['delta', 'max_time'])

print(f"Found {len(fairness_df)} fair meeting points!")
print(fairness_df.head(10)) # Show the top 10 fairest venues
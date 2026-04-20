import networkx as nx
import pandas as pd
from NetworkGraph import G

from geocoding import get_coordinates, find_nearest_station

print("Welcome to SweetSpot Route Optimization!")
user_address1 = input("Enter address for User 1 (e.g., 'Times Square, NY'): ")
user_address2 = input("Enter address for User 2 (e.g., 'Barclays Center, NY'): ")

# geocode the addresses
coords1 = get_coordinates(user_address1)
coords2 = get_coordinates(user_address2)

if not coords1 or not coords2:
    print("Error: Could not find one or both addresses. Please try again.")
    exit()

# find the nearest MTA platforms
user1_start, dist1 = find_nearest_station(G, coords1)
user2_start, dist2 = find_nearest_station(G, coords2)

print(f"\nUser 1 starting at: {G.nodes[user1_start]['name']} ({round(dist1)}m walk)")
print(f"User 2 starting at: {G.nodes[user2_start]['name']} ({round(dist2)}m walk)\n")
print("Calculating Optimal Meeting Points...\n")

# generate the Isochrones
x1_travel_times = nx.single_source_dijkstra_path_length(G, source=user1_start, weight='weight')
x2_travel_times = nx.single_source_dijkstra_path_length(G, source=user2_start, weight='weight')

fair_venues = []
MAX_COMMUTE = 60 
MAX_DELTA = 15 

# find the Temporal Midpoints
for node in G.nodes():
    if node in x1_travel_times and node in x2_travel_times:
        time_j = x1_travel_times[node] / 60 
        time_k = x2_travel_times[node] / 60 
        
        delta = abs(time_j - time_k)
        max_time = max(time_j, time_k)
        
        if delta <= MAX_DELTA and max_time <= MAX_COMMUTE:
            fair_venues.append({
                'stop_id': node,
                'name': G.nodes[node]['name'],
                'lat': G.nodes[node]['lat'],
                'lon': G.nodes[node]['lon'],
                'user1_time': round(time_j, 1),
                'user2_time': round(time_k, 1),
                'delta': round(delta, 1),
                'max_time': round(max_time, 1)
            })

#results
fairness_df = pd.DataFrame(fair_venues)

if fairness_df.empty:
    print("No fair meeting locations found within the commute limits.")
else:
    # Sort by the fairest delta, then by the shortest overall commute
    fairness_df = fairness_df.sort_values(by=['delta', 'max_time'])
    print(f"Found {len(fairness_df)} mathematically fair transit hubs!")
    
    # Display top 10 results
    print(fairness_df[['name', 'user1_time', 'user2_time', 'delta']].head(10).to_string(index=False))
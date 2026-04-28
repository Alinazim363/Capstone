import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from geopy.distance import geodesic
import networkx as nx

# Import your custom modules
from NetworkGraph import G
from geocoding import get_coordinates, find_nearest_station

# --- 1. INITIALIZE THE NLP MODEL ---
print("Loading SweetSpot Semantic Engine...")
model = SentenceTransformer('all-distilroberta-v1')

# --- 2. LOAD THE DATABASE ---
print("Loading Venue Database...")
try:
    production_db = pd.read_json('yelp/sweetspot_production_db_v2.json', orient='records', lines=True)
except FileNotFoundError:
    print("Error: 'sweetspot_production_db_v2.json' not found.")
    exit()

# --- 3. A* HEURISTIC & ITINERARY PARSER ---
def astar_heuristic(u, v):
    u_coords = (G.nodes[u]['lat'], G.nodes[u]['lon'])
    v_coords = (G.nodes[v]['lat'], G.nodes[v]['lon'])
    return geodesic(u_coords, v_coords).meters / 15

def get_itinerary(path):
    steps = []
    current_leg_type = None
    current_route = None
    stops_count = 0
    leg_time = 0
    
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        edge_data = G.get_edge_data(u, v)
        edge_type = edge_data['edge_type']
        weight = edge_data['weight']
        
        if edge_type == 'walking_transfer':
            if current_leg_type == 'subway_transit':
                steps.append(f"Take the {current_route} train for {stops_count} stops.")
            steps.append(f"🚶 Walk to {G.nodes[v]['name']} platform ({round(weight/60, 1)} mins).")
            current_leg_type = 'walking_transfer'
            stops_count, leg_time = 0, 0
        else:
            routes = edge_data.get('routes', ['Unknown'])
            if current_leg_type != 'subway_transit' or current_route not in routes:
                if current_leg_type == 'subway_transit':
                    steps.append(f"Take the {current_route} train for {stops_count} stops.")
                current_route = routes[0]
                steps.append(f"Board the {current_route} train at {G.nodes[u]['name']}.")
            current_leg_type = 'subway_transit'
            stops_count += 1
            leg_time += weight
            
    if current_leg_type == 'subway_transit':
        steps.append(f"Take the {current_route} train for {stops_count} stops.")
    steps.append(f"Arrive at {G.nodes[path[-1]]['name']}.")
    return steps

# --- 4. THE SWEETSPOT PIPELINE ---
def run_sweetspot():
    print("\n" + "="*50 + "\n WELCOME TO SWEETSPOT \n" + "="*50)
    
    user_address1 = input("User 1 Address: ")
    user_address2 = input("User 2 Address: ")
    user_query = input("Describe the vibe: ")
    
    # Task 1: Geocoding
    coords1, coords2 = get_coordinates(user_address1), get_coordinates(user_address2)
    if not coords1 or not coords2: return

    u1_start, walk_1 = find_nearest_station(G, coords1)
    u2_start, walk_2 = find_nearest_station(G, coords2)
    
    # Task 1: Fairness Isochrones (Dijkstra)
    x1_times = nx.single_source_dijkstra_path_length(G, source=u1_start, weight='weight')
    x2_times = nx.single_source_dijkstra_path_length(G, source=u2_start, weight='weight')
    
    MAX_COMMUTE, MAX_DELTA = 60, 7
    fair_stations = []
    for node in G.nodes():
        if node in x1_times and node in x2_times:
            t1, t2 = x1_times[node]/60, x2_times[node]/60
            if abs(t1 - t2) <= MAX_DELTA and max(t1, t2) <= MAX_COMMUTE:
                fair_stations.append({'id': node, 'lat': G.nodes[node]['lat'], 'lon': G.nodes[node]['lon'], 't1': t1, 't2': t2})
                
    if not fair_stations:
        print("No fair locations found.")
        return

    # Task 2: Geographic Hard Filter
    station_coords = [(s['lat'], s['lon']) for s in fair_stations]
    station_lookup = { (s['lat'], s['lon']): s for s in fair_stations }
    
    filtered_list = []
    for _, venue in production_db.iterrows():
        v_coords = (venue['lat'], venue['lon'])
        for s_coords in station_coords:
            if geodesic(v_coords, s_coords).meters <= 800:
                venue_data = venue.to_dict()
                venue_data['nearest_hub'] = station_lookup[s_coords]
                filtered_list.append(venue_data)
                break
                
    if not filtered_list: 
        print("No venues found near the fair transit stops.")
        return
    
    # Task 3: Semantic Ranking
    query_vec = model.encode([user_query])[0]
    for v in filtered_list:
        v['match'] = (1 - cosine(query_vec, np.array(v['vibe_vector']))) * 100
        
    # Grab the Top 10 results
    results = sorted(filtered_list, key=lambda x: x['match'], reverse=True)[:10]

    print("\n" + "="*50)
    print("THE TOP 10 SWEETSPOT RECOMMENDATIONS")
    print("="*50)
    
    # Print the formatted Top 10 list
    for i, v in enumerate(results):
        cat = v.get('categories', 'Restaurant/Bar')
        stars = v.get('stars', 'N/A')
        lat = round(v['lat'], 4)
        lon = round(v['lon'], 4)
        match = round(v['match'], 1)
        hub_name = G.nodes[v['nearest_hub']['id']]['name']
        
        print(f"[{i+1}] {v['name']} ({cat}) - ⭐ {stars}")
        print(f"    Vibe Match: {match}% | Transit Hub: {hub_name}")
        print(f"    Coordinates: {lat}, {lon}")
        print("-" * 50)

    # Task 4: Interactive Pathfinding
    choice = input("\nWould you like directions to a venue? (Enter 1-10, or 0 to exit): ")
    
    try:
        choice_idx = int(choice) - 1
        if 0 <= choice_idx < len(results):
            top_v = results[choice_idx]
            print(f"\nCalculating A* routes to {top_v['name']}...")
            
            # Run A* dynamically based on user choice
            path1 = nx.astar_path(G, u1_start, top_v['nearest_hub']['id'], heuristic=astar_heuristic, weight='weight')
            path2 = nx.astar_path(G, u2_start, top_v['nearest_hub']['id'], heuristic=astar_heuristic, weight='weight')
            
            # Print Itineraries with context on initial walks and subway travel time
            print(f"\n--- USER 1 ITINERARY ({round(top_v['nearest_hub']['t1'], 1)} mins via Transit + {round(walk_1)}m initial walk) ---")
            for step in get_itinerary(path1): print(f"  {step}")
            
            print(f"\n--- USER 2 ITINERARY ({round(top_v['nearest_hub']['t2'], 1)} mins via Transit + {round(walk_2)}m initial walk) ---")
            for step in get_itinerary(path2): print(f"  {step}")
        else:
            print("Exiting SweetSpot. Enjoy your date!")
    except ValueError:
        print("Exiting SweetSpot. Enjoy your date!")

if __name__ == "__main__":
    run_sweetspot()
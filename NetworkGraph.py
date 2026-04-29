import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

# initialize the graph
G = nx.DiGraph()

# helper function to convert time strings to seconds
def time_to_seconds(time_str):
    if pd.isna(time_str):
        return None
    
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

stops_df = pd.read_csv('gtfs/stops.txt')
platforms_df = stops_df[stops_df['parent_station'].notna()]

# iterate through platforms and add them as nodes to the graph
for index, row in platforms_df.iterrows():
    G.add_node(row['stop_id'], name=row['stop_name'], lat=row['stop_lat'], lon=row['stop_lon'])

# create a mapping of parent stations to their child platforms for easy lookup
parent_to_children = platforms_df.groupby('parent_station')['stop_id'].apply(list).to_dict()

transfers_df = pd.read_csv('gtfs/transfers.txt')
walking_transfers = transfers_df[transfers_df['transfer_type'] == 2]

# iterate through walking transfers and add edges between all child platforms of the parent stations involved in the transfer
for index, row in walking_transfers.iterrows():
    parent_from = row['from_stop_id']
    parent_to = row['to_stop_id']
    transfer_time = row['min_transfer_time']
    
    # get child platforms for both parent stations, and create edges between all combinations of child platforms
    children_from = parent_to_children.get(parent_from, [])
    children_to = parent_to_children.get(parent_to, [])
    
    for c_from in children_from:
        for c_to in children_to:
            
            # prevent self loops
            if c_from != c_to:
                
                if G.has_node(c_from) and G.has_node(c_to):
                    G.add_edge(
                        c_from, 
                        c_to, 
                        weight=transfer_time,
                        edge_type='walking_transfer'
                    )

# add subway transit edges with travel time as weight and routes as edge attributes
trips_df = pd.read_csv('gtfs/trips.txt')
stop_times_df = pd.read_csv('gtfs/stop_times.txt')
weekday_trips = trips_df[trips_df['service_id'].str.contains('Weekday', na=False, case=False)]
merged_df = stop_times_df.merge(weekday_trips[['trip_id', 'route_id']], on='trip_id', how='inner')

# convert arrival and departure times to seconds for easier calculations
merged_df['arrival_sec'] = merged_df['arrival_time'].apply(time_to_seconds)
merged_df['departure_sec'] = merged_df['departure_time'].apply(time_to_seconds)
merged_df = merged_df.sort_values(by=['trip_id', 'stop_sequence'])

# create columns for the next stop, next trip, and next arrival time to calculate travel times
merged_df['next_stop_id'] = merged_df['stop_id'].shift(-1)
merged_df['next_trip_id'] = merged_df['trip_id'].shift(-1)
merged_df['next_arrival_sec'] = merged_df['arrival_sec'].shift(-1)

valid_edges = merged_df[merged_df['trip_id'] == merged_df['next_trip_id']].copy()
valid_edges['travel_time'] = valid_edges['next_arrival_sec'] - valid_edges['departure_sec']

# Log or handle negative or NaN travel times
invalid_travel_times = valid_edges[
    (valid_edges['travel_time'].isna()) | (valid_edges['travel_time'] <= 0)
]
if not invalid_travel_times.empty:
    print("Warning: Found rows with invalid travel_time (NaN or negative):")
    print(invalid_travel_times[['stop_id', 'next_stop_id', 'departure_sec', 'next_arrival_sec', 'travel_time']])

valid_edges = valid_edges[(valid_edges['travel_time'] > 0) & (valid_edges['travel_time'] < 3600)]

edge_summary = valid_edges.groupby(['stop_id', 'next_stop_id']).agg(
    travel_time=('travel_time', 'mean'), routes=('route_id', lambda x: list(set(x)))).reset_index()

# loop through the edge summary and add edges to the graph with travel time as weight
for index, row in edge_summary.iterrows():
    
    if G.has_node(row['stop_id']) and G.has_node(row['next_stop_id']):
        G.add_edge(
            row['stop_id'], 
            row['next_stop_id'], 
            weight=row['travel_time'],
            edge_type='subway_transit',
            routes=row['routes']
        )
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

import NetworkGraph
G = NetworkGraph.G

def get_coordinates(address):
    geolocator = Nominatim(user_agent="sweetspot")
    location = geolocator.geocode(address)
    
    if location:
        return (location.latitude, location.longitude)
    else:
        print("Address not found.")
        return None


def find_nearest_station(graph, user_coords):
    nearest_station = None
    min_distance = float('inf')
    
    for node_id, data in graph.nodes(data=True):
        if 'lat' in data and 'lon' in data:
            station_coords = (data['lat'], data['lon'])
            
            distance = geodesic(user_coords, station_coords).meters
            
            if distance < min_distance:
                min_distance = distance
                nearest_station = node_id
                
    return nearest_station, min_distance

user_address = "Hunter College, New York, NY"
user_coords = get_coordinates(user_address)
print(f"Coordinates: {user_coords}")

if user_coords:
    nearest_id, distance_away = find_nearest_station(G, user_coords)
    station_name = G.nodes[nearest_id].get('name', 'Unknown Station')
    
    print(f"Nearest Station: {station_name} (ID: {nearest_id})")
    print(f"Distance: {round(distance_away, 2)} meters away")
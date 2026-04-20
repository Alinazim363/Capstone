from geopy.geocoders import Nominatim
from geopy.distance import geodesic

from NetworkGraph import G

def get_coordinates(address):
    geolocator = Nominatim(user_agent="sweetspot")
    try:
        location = geolocator.geocode(address)
    except Exception as e:
        print(f"Error occurred during geocoding: {e}")
        return None
    if location:
        return (location.latitude, location.longitude)
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

user_address1 = input("Enter address for user 1: ")
user_address2 = input("Enter address for user 2: ")

user_coords1 = get_coordinates(user_address1)
user_coords2 = get_coordinates(user_address2)

if user_coords1:
    nearest_id, distance_away = find_nearest_station(G, user_coords1)
    station_name = G.nodes[nearest_id].get('name', 'Unknown Station')
    
    print(f"Nearest Station for user 1: {station_name} (ID: {nearest_id})")
    print(f"Distance: {round(distance_away, 2)} meters away")

if user_coords2:
    nearest_id, distance_away = find_nearest_station(G, user_coords2)
    station_name = G.nodes[nearest_id].get('name', 'Unknown Station')
    
    print(f"Nearest Station for user 2: {station_name} (ID: {nearest_id})")
    print(f"Distance: {round(distance_away, 2)} meters away")
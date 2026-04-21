import unittest
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from geopy.distance import geodesic

# Import your custom modules
from NetworkGraph import G
from geocoding import find_nearest_station, get_coordinates

class TestSweetSpotCore(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """This runs once before the tests start. We load the NLP model here so it doesn't slow down every test."""
        print("Initializing Test Environment...")
        cls.nlp_model = SentenceTransformer('all-distilroberta-v1')

    # --- TEST 1: THE GRAPH INFRASTRUCTURE ---
    def test_graph_is_directed(self):
        """Ensures the transit graph is strictly one-way (DiGraph) to prevent 'teleportation'."""
        self.assertTrue(nx.is_directed(G), "CRITICAL: The graph must be a networkx.DiGraph()")
        self.assertGreater(len(G.nodes()), 500, "Graph seems empty. Should have ~1,000+ nodes.")

    # --- TEST 2: GEOCODING & SPATIAL SNAPPING ---
    def test_station_snapping(self):
        """Tests if a random Manhattan coordinate correctly snaps to an existing transit node."""
        # Fake coordinates near Times Square
        test_coords = (40.7580, -73.9855) 
        
        nearest_station_id, distance = find_nearest_station(G, test_coords)
        
        self.assertIsNotNone(nearest_station_id, "Failed to find a nearest station.")
        self.assertTrue(G.has_node(nearest_station_id), "The snapped station does not exist in the graph.")
        self.assertTrue(distance >= 0, "Distance calculation is broken.")

    # --- TEST 3: THE GEODESIC HARD FILTER ---
    def test_haversine_distance(self):
        """Verifies the 800m walking limit logic is mathematically accurate."""
        times_square = (40.7580, -73.9855)
        empire_state = (40.7488, -73.9857) # Roughly 1km away
        
        distance_meters = geodesic(times_square, empire_state).meters
        
        # It should be around 1,020 meters. We test that it properly violates the 800m filter limit.
        self.assertGreater(distance_meters, 800, "Distance math failed. Empire State should be >800m from Times Sq.")

    # --- TEST 4: SEMANTIC NLP INTEGRATION ---
    def test_nlp_vibe_sorting(self):
        """Proves that the mathematical cosine similarity correctly separates opposing vibes."""
        # 1. Set up the query
        query_vector = self.nlp_model.encode(["A quiet, intimate romantic dinner"])[0]
        
        # 2. Set up a "Correct" match and a "Wrong" match
        good_match = self.nlp_model.encode(["Upscale wine bar with dim lighting and romantic music"])[0]
        bad_match = self.nlp_model.encode(["Loud dive bar with sticky floors, cheap beer, and a jukebox"])[0]
        
        # 3. Calculate similarities
        good_score = 1 - cosine(query_vector, good_match)
        bad_score = 1 - cosine(query_vector, bad_match)
        
        # 4. Assert that the math understands the difference
        self.assertGreater(good_score, bad_score, "NLP failed: The loud dive bar scored higher than the romantic spot!")

if __name__ == '__main__':
    unittest.main(verbosity=2)
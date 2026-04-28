"""
SweetSpot - Data-Driven Transit Optimization
Streamlit Frontend Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from geopy.distance import geodesic
import folium
from streamlit_folium import st_folium

# --- Custom Imports (Your Backend Logic) ---
from NetworkGraph import G
from geocoding import get_coordinates, find_nearest_station

# --- Page Configuration ---
st.set_page_config(
    page_title="SweetSpot - Fair Transit Routing",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize Session State ---
if 'results' not in st.session_state:
    st.session_state['results'] = None
if 'coords1' not in st.session_state:
    st.session_state['coords1'] = None
if 'coords2' not in st.session_state:
    st.session_state['coords2'] = None
if 'query_vec' not in st.session_state:
    st.session_state['query_vec'] = None
if 'vibe_query' not in st.session_state:
    st.session_state['vibe_query'] = ""

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    /* Overall page background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }

    /* Force dark text ONLY in the main container, protecting the sidebar */
    .main .block-container p, 
    .main .block-container div, 
    .main .block-container span, 
    .main .block-container h1, 
    .main .block-container h2, 
    .main .block-container h3 {
        color: #1f1f1f !important;
    }

    /* Force light text in the sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1c23 !important;
    }
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] div, 
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] label {
        color: #ffffff !important;
    }

    /* Ensure text inputs remain readable */
    input, textarea {
        color: #1f1f1f !important;
        background-color: #ffffff !important;
    }
    
    /* FIX: Force placeholder text to be a readable gray */
    ::placeholder, input::placeholder, textarea::placeholder {
        color: #888888 !important;
        opacity: 1; 
    }
    
    /* FIX: Selectbox Dropdown Visibility */
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        color: #1f1f1f !important;
    }
    ul[data-baseweb="menu"] {
        background-color: #ffffff !important;
    }
    ul[data-baseweb="menu"] li {
        color: #1f1f1f !important;
    }

    /* Main header styling */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    /* Sub-header styling */
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        font-weight: bold;
        background-color: #ff4b4b;
        color: white !important;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.3);
        transform: translateY(-2px);
    }
    
    /* Expander styling */
    [data-testid="stExpander"] {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. CACHE THE HEAVY MODELS ---
@st.cache_resource
def load_sweetspot_engine():
    return SentenceTransformer('all-distilroberta-v1')

@st.cache_data
def load_database():
    return pd.read_json('yelp/sweetspot_production_db_v2.json', orient='records', lines=True)

model = load_sweetspot_engine()
production_db = load_database()

# --- 2. A* ROUTING ENGINE ---
def astar_heuristic(u, v):
    """Estimates travel time based on straight-line distance."""
    u_coords = (G.nodes[u]['lat'], G.nodes[u]['lon'])
    v_coords = (G.nodes[v]['lat'], G.nodes[v]['lon'])
    return geodesic(u_coords, v_coords).meters / 15

def get_itinerary(path):
    """Translates a list of stop IDs into human-readable directions."""
    steps = []
    current_leg_type = None
    current_route = None
    stops_count = 0
    
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        edge_data = G.get_edge_data(u, v)
        edge_type = edge_data['edge_type']
        weight = edge_data['weight']
        
        if edge_type == 'walking_transfer':
            if current_leg_type == 'subway_transit':
                steps.append(f"🚆 Take the **{current_route}** train for {stops_count} stops.")
            steps.append(f"🚶 Walk to {G.nodes[v]['name']} platform ({round(weight/60, 1)} mins).")
            current_leg_type = 'walking_transfer'
            stops_count = 0
        else:
            routes = edge_data.get('routes', ['Unknown'])
            if current_leg_type != 'subway_transit' or current_route not in routes:
                if current_leg_type == 'subway_transit':
                    steps.append(f"🚆 Take the **{current_route}** train for {stops_count} stops.")
                current_route = routes[0]
                steps.append(f"🟢 Board the **{current_route}** train at {G.nodes[u]['name']}.")
            current_leg_type = 'subway_transit'
            stops_count += 1
            
    if current_leg_type == 'subway_transit':
        steps.append(f"🚆 Take the **{current_route}** train for {stops_count} stops.")
    steps.append(f"🏁 Arrive at **{G.nodes[path[-1]]['name']}**.")
    return steps

# --- 3. THE UI LAYOUT ---
st.markdown('<div class="main-header">🚇 SweetSpot Optimization</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Data-Driven Routing for the Perfect Vibe</div>', unsafe_allow_html=True)

status_container = st.empty()

# The Input Sidebar
with st.sidebar:
    st.markdown('<div style="font-size: 70px; text-align: center; margin-bottom: -10px;">🚇</div>', unsafe_allow_html=True)
    st.markdown('<h2 style="text-align: center; color: white;">SweetSpot</h2>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.header("📍 Set Locations")
    user1_address = st.text_input("User 1 Location", placeholder="e.g., 14th St and 6th Ave")
    user2_address = st.text_input("User 2 Location", placeholder="e.g., 86th St and Lexington")
    
    st.markdown("---")
    st.header("✨ Set The Vibe")
    vibe_query = st.text_area("Describe the atmosphere", placeholder="e.g., A quiet, dimly lit spot for a romantic date")
    
    find_button = st.button("Calculate SweetSpot")

# --- 4. LANDING PAGE INFO ---
if st.session_state['results'] is None:
    st.markdown("""
    <div style="background-color: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 2rem;">
        <p style="text-align: center; font-size: 1.2rem; font-weight: bold;">
            Find the mathematically fairest and semantically perfect meetup spot.
        </p>
        <p style="text-align: center; font-size: 1rem; margin-top: 1rem;">
            🔵 Person 1 &nbsp;&nbsp;→&nbsp;&nbsp; ⭐ Fair Transit Hub &nbsp;&nbsp;←&nbsp;&nbsp; 🟢 Person 2
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: white; border-radius: 10px; height: 100%; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <h3>📍 Isochrone Routing</h3>
            <p>Uses NetworkX to map transit times, ensuring neither user travels significantly longer than the other.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: white; border-radius: 10px; height: 100%; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <h3>🧠 Semantic NLP</h3>
            <p>Powered by Sentence-RoBERTa. Type any vibe, and we mathematically match it to real Yelp reviews.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background: white; border-radius: 10px; height: 100%; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <h3>🗺️ Interactive Map</h3>
            <p>See both users, the transit hubs, and your top recommended venues geographically mapped.</p>
        </div>
        """, unsafe_allow_html=True)

# --- 5. EXECUTE THE BACKEND ---
if find_button:
    if not user1_address or not user2_address or not vibe_query:
        st.sidebar.warning("⚠️ Please fill in all fields before searching.")
    else:
        with status_container.container():
            with st.spinner("Geocoding addresses..."):
                coords1 = get_coordinates(user1_address)
                coords2 = get_coordinates(user2_address)
                
            if not coords1 or not coords2:
                st.error("❌ Could not locate one or both addresses.")
            else:
                with st.spinner("Analyzing MTA Network..."):
                    u1_start, _ = find_nearest_station(G, coords1)
                    u2_start, _ = find_nearest_station(G, coords2)
                    
                    x1_times = nx.single_source_dijkstra_path_length(G, source=u1_start, weight='weight')
                    x2_times = nx.single_source_dijkstra_path_length(G, source=u2_start, weight='weight')
                    
                    # UPDATED CONSTRAINTS HERE
                    MAX_COMMUTE, MAX_DELTA = 60, 7
                    fair_stations = []
                    for node in G.nodes():
                        if node in x1_times and node in x2_times:
                            t1, t2 = x1_times[node]/60, x2_times[node]/60
                            if abs(t1 - t2) <= MAX_DELTA and max(t1, t2) <= MAX_COMMUTE:
                                fair_stations.append({'id': node, 'lat': G.nodes[node]['lat'], 'lon': G.nodes[node]['lon'], 't1': t1, 't2': t2})
                    
                if not fair_stations:
                    st.error("❌ No mathematically fair meeting points found. Try adjusting the locations.")
                else:
                    with st.spinner("Filtering and Ranking Venues..."):
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
                            st.error("❌ No Yelp venues found near the fair transit stops.")
                        else:
                            query_vec = model.encode([vibe_query])[0]
                            for v in filtered_list:
                                v['match'] = (1 - cosine(query_vec, np.array(v['vibe_vector']))) * 100
                                
                            st.session_state['results'] = sorted(filtered_list, key=lambda x: x['match'], reverse=True)[:10]
                            st.session_state['coords1'] = coords1
                            st.session_state['coords2'] = coords2
                            st.session_state['query_vec'] = query_vec
                            st.session_state['vibe_query'] = vibe_query

# --- 6. VISUALIZE THE RESULTS ---
if st.session_state['results']:
    results = st.session_state['results']
    coords1 = st.session_state['coords1']
    coords2 = st.session_state['coords2']
    
    st.success(f"🎉 Found {len(results)} mathematically fair spots that match your vibe!")
    
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown("<h3>Top Matches</h3>", unsafe_allow_html=True)
        for i, v in enumerate(results):
            with st.expander(f"[{i+1}] {v['name']} - {round(v['match'], 1)}% Match", expanded=(i==0)):
                cat = v.get('categories', 'Restaurant/Bar')
                stars = v.get('stars', 'N/A')
                hub_name = G.nodes[v['nearest_hub']['id']]['name']
                
                st.markdown(f"""
                <div style="color: #000000; font-size: 15px; line-height: 1.6;">
                    <b>🍽️ Category:</b> {cat} <br>
                    <b>⭐ Rating:</b> {stars} <br>
                    <b>🚇 Transit Hub:</b> {hub_name} <br>
                    <b>⏱️ User 1 Commute:</b> {round(v['nearest_hub']['t1'], 1)} mins <br>
                    <b>⏱️ User 2 Commute:</b> {round(v['nearest_hub']['t2'], 1)} mins <br>
                    <b>🌐 Coordinates:</b> {round(v['lat'], 4)}, {round(v['lon'], 4)}
                </div>
                """, unsafe_allow_html=True)

    with col2:
        st.markdown("<h3>Interactive Map</h3>", unsafe_allow_html=True)
        
        mid_lat = (coords1[0] + coords2[0]) / 2
        mid_lon = (coords1[1] + coords2[1]) / 2
        m = folium.Map(location=[mid_lat, mid_lon], zoom_start=12, tiles="CartoDB positron")
        
        folium.Marker(location=coords1, popup="User 1", icon=folium.Icon(color="blue", icon="user")).add_to(m)
        folium.Marker(location=coords2, popup="User 2", icon=folium.Icon(color="green", icon="user")).add_to(m)
        
        for i, v in enumerate(results):
            color = "red" if i == 0 else "orange" 
            folium.Marker(
                location=[v['lat'], v['lon']],
                popup=f"<b>{v['name']}</b><br>Match: {round(v['match'],1)}%",
                tooltip=f"[{i+1}] {v['name']}",
                icon=folium.Icon(color=color, icon="star")
            ).add_to(m)
            
        st_folium(m, width=800, height=500)
        
    # --- 7. A* DIRECTIONS MODULE ---
    st.markdown("---")
    st.markdown("<h3>🧭 Get A* Directions</h3>", unsafe_allow_html=True)
    
    venue_names = [f"{v['name']} ({round(v['match'], 1)}% Match)" for v in results]
    selected_venue_str = st.selectbox("Select a venue to generate your custom itineraries:", venue_names)
    
    if st.button("Calculate Optimal Routes"):
        with st.spinner("Running A* Pathfinding..."):
            selected_name = selected_venue_str.split(" (")[0]
            top_v = next(v for v in results if v['name'] == selected_name)
            
            hub_id = top_v['nearest_hub']['id']
            u1_start, walk_1 = find_nearest_station(G, coords1)
            u2_start, walk_2 = find_nearest_station(G, coords2)
            
            path1 = nx.astar_path(G, u1_start, hub_id, heuristic=astar_heuristic, weight='weight')
            path2 = nx.astar_path(G, u2_start, hub_id, heuristic=astar_heuristic, weight='weight')
            
            r_col1, r_col2 = st.columns(2)
            
            with r_col1:
                st.markdown(f"""
                <div style="background: white; padding: 1.5rem; border-radius: 10px; border-top: 4px solid blue; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                    <h4 style="color: #1f1f1f;">🔵 User 1 Itinerary</h4>
                    <p style="color: #666; font-size: 0.9rem;">Initial walk: {round(walk_1)} meters</p>
                    <hr>
                </div>
                """, unsafe_allow_html=True)
                for step in get_itinerary(path1): 
                    st.markdown(f"<p style='color: #1f1f1f; margin-bottom: 5px;'>{step}</p>", unsafe_allow_html=True)

            with r_col2:
                st.markdown(f"""
                <div style="background: white; padding: 1.5rem; border-radius: 10px; border-top: 4px solid green; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                    <h4 style="color: #1f1f1f;">🟢 User 2 Itinerary</h4>
                    <p style="color: #666; font-size: 0.9rem;">Initial walk: {round(walk_2)} meters</p>
                    <hr>
                </div>
                """, unsafe_allow_html=True)
                for step in get_itinerary(path2): 
                    st.markdown(f"<p style='color: #1f1f1f; margin-bottom: 5px;'>{step}</p>", unsafe_allow_html=True)

    # --- 8. DEVELOPER DIAGNOSTICS SECTION ---
    st.markdown("---")
    with st.expander("🛠️ Developer Diagnostics (Backend Architecture Context)"):
        d_col1, d_col2 = st.columns(2)
        
        with d_col1:
            st.markdown("#### 📊 Network Graph Infrastructure")
            st.write(f"**Graph Type:** `networkx.{type(G).__name__}`")
            st.write(f"**Total Nodes (MTA Platforms):** `{G.number_of_nodes():,}`")
            st.write(f"**Total Edges (Transit/Walk):** `{G.number_of_edges():,}`")
            
            st.markdown("#### ⏱️ Routing Constraints")
            st.write("**Broad Search:** `nx.single_source_dijkstra_path_length`")
            st.write("**Targeted Pathing:** `nx.astar_path` (Haversine Heuristic)")
            st.write("**Max Commute Limit:** `60.0 mins`")
            st.write("**Max Fairness Delta:** `7.0 mins`")
            st.write("**Terminal Walk Radius:** `800m (Geodesic)`")

        with d_col2:
            st.markdown("#### 🧠 Semantic NLP Engine")
            st.write("**Model:** `SentenceTransformer('all-distilroberta-v1')`")
            st.write(f"**Raw Query:** *\"{st.session_state['vibe_query']}\"*")
            
            if st.session_state['query_vec'] is not None:
                vec = st.session_state['query_vec']
                st.write(f"**Embedding Shape:** `{vec.shape}`")
                
                sample_dims = ", ".join([f"{x:.4f}" for x in vec[:5]])
                st.write(f"**Vector Sample (first 5 dims):** `[{sample_dims}, ...]`")
                st.write("**Ranking Metric:** `scipy.spatial.distance.cosine`")
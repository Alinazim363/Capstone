import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

# --- 1. LOAD THE INFRASTRUCTURE ---
print("Loading Sentence-RoBERTa Model... (This takes a few seconds)")
model = SentenceTransformer('all-distilroberta-v1')

print("Loading SweetSpot V2 Database...")
# Ensure you are loading the updated V2 database!
try:
    db = pd.read_json('yelp/sweetspot_production_db_v2.json', orient='records', lines=True)
    print(f"Loaded {len(db)} venues successfully.")
except FileNotFoundError:
    print("Error: 'sweetspot_production_db_v2.json' not found. Ensure it is in the same folder.")
    exit()

# --- 2. THE SEARCH FUNCTION ---
def test_vibe_search(query, top_k=5):
    print("\n" + "="*50)
    print(f"SEARCHING: '{query}'")
    print("="*50)
    
    # Vectorize the query using Sentence-RoBERTa
    query_vector = model.encode([query])[0]
    
    results = []
    
    # Calculate Cosine Similarity against every venue
    for _, row in db.iterrows():
        venue_vector = np.array(row['vibe_vector'])
        
        # Scipy's cosine calculates distance (0 is identical). We want similarity, so subtract from 1.
        similarity = 1 - cosine(query_vector, venue_vector)
        
        results.append({
            'Venue': row['name'],
            'Match %': round(similarity * 100, 1)
        })
        
    # Sort and display the results
    results_df = pd.DataFrame(results).sort_values(by='Match %', ascending=False)
    print(results_df.head(top_k).to_string(index=False))

# --- 3. RUN THE DIAGNOSTICS ---
if __name__ == "__main__":
    print("\nStarting Semantic Diagnostics...")
    
    # Test 1: High Energy / Loud
    test_vibe_search("A super loud sports bar to drink beer with friends and watch the game")
    
    # Test 2: Intimate / Quiet
    test_vibe_search("A quiet, romantic, intimate dinner setting with wine and low lighting")
    
    # Test 3: Casual / Functional (Let's see if it understands cafes vs bars!)
    test_vibe_search("A cozy coffee shop with wifi to sit and study on my laptop")
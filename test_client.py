"""
Example Python Orchestrator Client for the Julia Surge Simulation Microservice
"""
import requests
import json

SERVER_URL = "http://localhost:8080/simulate_surge"

# Construct a synthetic 5x5 Digital Elevation Model (DEM)
# Slope from coast (left: x=0, z=0.5m) to inland (right: x=4, z=3.5m)
dem = [
    [0.5, 1.0, 1.8, 2.5, 3.5],
    [0.6, 1.1, 1.9, 2.6, 3.6],
    [0.4, 0.9, 1.7, 2.4, 3.4],
    [0.5, 1.2, 2.0, 2.7, 3.7],
    [0.7, 1.3, 2.1, 2.8, 3.8]
]

# Coastline source points (left edge column)
coastline_mask = [
    [True, False, False, False, False],
    [True, False, False, False, False],
    [True, False, False, False, False],
    [True, False, False, False, False],
    [True, False, False, False, False]
]

# Infrastructure assets across the domain (0-indexed)
infrastructure_nodes = [
    {"id": "power_substation_alpha", "x_idx": 1, "y_idx": 1, "type": "power_grid"},
    {"id": "district_hospital_central", "x_idx": 2, "y_idx": 2, "type": "hospital"},
    {"id": "coastal_highway_route1", "x_idx": 0, "y_idx": 0, "type": "road"},
    {"id": "inland_evac_route9", "x_idx": 4, "y_idx": 4, "type": "road"}
]

payload = {
    "surge_height": 8.5,           # Simulating a massive Category 5 surge
    "wind_speed": 150,             # Peak gust wind speed (km/h)
    "dem": dem,
    "resolution": 30.0,            # 30 meters per grid cell
    "coastline_mask": coastline_mask,
    "infrastructure_nodes": infrastructure_nodes
}

def main():
    print(f"Sending simulation request to {SERVER_URL}...")
    try:
        response = requests.post(SERVER_URL, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        print("\n--- Simulation Response ---")
        print(json.dumps(data, indent=2))
        
        print("\n--- Vulnerability Summary ---")
        for node in data["node_results"]:
            print(f"[{node['status'].upper():<8}] ID: {node['id']} ({node['type'] if 'type' in node else 'node'}) "
                  f"- Depth: {node['final_water_depth']}m - Score: {node['vulnerability_score']}")
        print(f"\nMax Inland Penetration: {data['max_inland_penetration']} meters")
    except requests.exceptions.RequestException as e:
        print(f"Failed to connect to Julia backend: {e}")

if __name__ == "__main__":
    main()

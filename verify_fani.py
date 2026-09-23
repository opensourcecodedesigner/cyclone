import json
import os
import folium
import webbrowser

file_path = "FANI_IBTRACS_TRACK.geojson"

print("--------------------------------------------------")
print("🌀 GENERATING TACTICAL SATELLITE MAP...")
print("--------------------------------------------------")

with open(file_path, "r", encoding="utf-8") as f:
    fani_data = json.load(f)

# Use Esri World Imagery for high-resolution satellite background (no API key required)
m = folium.Map(
    location=[15.0, 85.0],
    zoom_start=5,
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery"
)

# Overlay the cyclone track in striking crimson
folium.GeoJson(
    fani_data,
    name="Cyclone Fani Track",
    style_function=lambda feature: {
        "color": "#FF2222",
        "weight": 5,
        "opacity": 0.9
    }
).add_to(m)

output_html = "fani_map_v3.html"
m.save(output_html)

print(f"✅ Success! Generated tactical map: {output_html}")
webbrowser.open(os.path.abspath(output_html))

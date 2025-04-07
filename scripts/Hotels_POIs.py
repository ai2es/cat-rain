## Hotels POIs California by 2025
# February 2025

# 1. Libraries
#--------------------------------------------------------------------------------
import requests
import pandas as pd
import pandas as pd
import geopandas as gpd
import folium
import numpy as np
from shapely.geometry import Point

# 2. API and Data
#--------------------------------------------------------------------------------

# Define the Overpass API endpoint
overpass_url = "http://overpass-api.de/api/interpreter"

# Define the Overpass QL query to fetch hotels in California (nodes, ways, and relations)
overpass_query = """
[out:json][timeout:300];
area["name"="California"]->.searchArea;
(
  node["tourism"="hotel"](area.searchArea);
  way["tourism"="hotel"](area.searchArea);
  relation["tourism"="hotel"](area.searchArea);
);
out center;
"""

# Send the request to the Overpass API
response = requests.get(overpass_url, params={'data': overpass_query})
data = response.json()

# Parse the response to extract hotel information
hotels = []
for element in data.get('elements', []):
    # For nodes, the lat/lon are directly available
    if element['type'] == 'node':
        lat = element.get('lat')
        lon = element.get('lon')
    else:
        # For ways and relations, use the 'center' attribute if available
        center = element.get('center')
        if center:
            lat = center.get('lat')
            lon = center.get('lon')
        else:
            lat = None
            lon = None

    if lat is not None and lon is not None:
        hotel = {
            'id': element.get('id'),
            'name': element.get('tags', {}).get('name', 'N/A'),
            'type': element.get('tags', {}).get('tourism', 'N/A'),
            'lat': lat,
            'lon': lon
        }
        hotels.append(hotel)

# Convert the list of hotels into a pandas DataFrame
hotels_df = pd.DataFrame(hotels)

# Print the number of hotels and a preview of the DataFrame
print("Number of hotels retrieved:", len(hotels_df))
print(hotels_df.head())

# Optionally, save the data to a CSV file
hotels_df.to_csv("california_hotels.csv", index=False)
print("Hotel data saved as 'california_hotels.csv'")

# 3. Geodataframe and zip codes merge
#--------------------------------------------------------------------------------

# ------------------------------
#  Convert hotels_df to a GeoDataFrame
# ------------------------------
# Create geometry from longitude and latitude
geometry = [Point(xy) for xy in zip(hotels_df['lon'], hotels_df['lat'])]
hotels_gdf = gpd.GeoDataFrame(hotels_df, geometry=geometry, crs="EPSG:4326")

# ------------------------------
# Load California Zip Code Boundaries
# ------------------------------
zip_codes = gpd.read_file("https://raw.githubusercontent.com/OpenDataDE/State-zip-code-GeoJSON/master/ca_california_zip_codes_geo.min.json")
# Inspect the first few rows (the zip code is stored in property "ZCTA5CE10")
print(zip_codes.head())

# ------------------------------
# Spatial Join to Assign Zip Codes to Hotels
# ------------------------------
# Use a spatial join so that each hotel is assigned a zip code polygon from zip_codes.
hotels_with_zip = gpd.sjoin(hotels_gdf, zip_codes[['ZCTA5CE10', 'geometry']], how="left", predicate='within')

# ------------------------------
#  Aggregate Hotel Counts by Zip Code
# ------------------------------
zip_counts = hotels_with_zip.groupby('ZCTA5CE10').size().reset_index(name='count')
print("Hotel counts by zip code:")
print(zip_counts.head())

# Save to Shapefile
hotels_gdf.to_file("california_hotels.shp", driver="ESRI Shapefile")
print("Hotel data saved as 'california_hotels.shp'")




# Graphs and visualization below. 
# ------------------------------
#  Compute a Threshold Scale for the Choropleth
# ------------------------------
# Calculate minimum and maximum counts
min_count = zip_counts['count'].min()
max_count = zip_counts['count'].max()

# Create a list of thresholds (bins) from min to max using 10 intervals.
threshold_scale = np.linspace(min_count, max_count, 10).tolist()
# Ensure the last value is exactly the maximum count.
threshold_scale[-1] = max_count

print("Threshold scale for choropleth bins:", threshold_scale)

# ------------------------------
#  Create a Choropleth Map (Leaflet Map via Folium)
# ------------------------------
# Define a center for California (approximate center)
california_center = [37.0, -119.0]
m = folium.Map(location=california_center, zoom_start=6)

# Add the choropleth layer.
folium.Choropleth(
    geo_data=zip_codes,
    data=zip_counts,
    columns=['ZCTA5CE10', 'count'],
    key_on='feature.properties.ZCTA5CE10',  # This key should match the property name in the GeoJSON
    fill_color='YlOrRd',
    threshold_scale=threshold_scale,
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Number of Hotels'
).add_to(m)

# Optional: add a layer control for toggling layers
folium.LayerControl().add_to(m)

# Save the map to an HTML file so you can open it in a web browser
m.save("california_hotels_zip_density_leaflet.html")
print("Leaflet choropleth map saved as 'california_hotels_zip_density_leaflet.html'")

# Import libraries
import pandas as pd # Data Handling
import networkx as nx # Create graphs
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import collections 
import numpy as np # mathematical
from shapely.wkt import loads # convert geometry to coordinates
from pyproj import Proj, transform # create edge weight
import contextily as ctx # use map layer
import geopandas as gpd
from shapely.geometry import LineString

cities = pd.read_csv('../data/cities.csv')
stations = pd.read_csv('../data/stations.csv')
lines = pd.read_csv('../data/lines.csv')
systems = pd.read_csv('../data/systems.csv')
station_lines = pd.read_csv('../data/station_lines.csv')
transport_modes = pd.read_csv('../data/transport_modes.csv')

modes_colors = {
    'default': '#000000',  # Black
    'high_speed_rail': '#FF0000',  # Bright Red
    'inter_city_rail': '#FFFF00',  # Bright Yellow
    'commuter_rail': '#00FF00',  # Bright Green
    'heavy_rail': '#0000FF',  # Bright Blue
    'light_rail': '#FF00FF',  # Magenta
    'brt': '#00FFFF',  # Cyan
    'people_mover': '#FFA500',  # Orange
    'bus': '#800080',  # Purple
    'tram': '#FFC0CB',  # Light Pink
    'ferry': '#A52A2A'  # Brown
}


# View first five rows of each dataset
print(cities.head())
print(stations.head())
print(lines.head())
print(systems.head())
print(station_lines.head())

# View dataset information

print(cities.info())
print(stations.info())
print(lines.info())
print(systems.info())
print(station_lines.info())

# Rename Column ID to City ID to ensure data correlatin

cities.rename(columns={"id":"city_id"}, inplace  = True)

# Confirm Name Change

cities.head()

# Join Cities Dataset with Station Lines Dataset

temp_df = pd.merge(cities, station_lines, on = 'city_id', how = 'inner')

# Confirm Join

temp_df.head()

# Change Line ID name

lines.rename(columns={"id":"line_id"}, inplace  = True)

# Confirm name change

lines.head()

# Renam Station ID

stations.rename(columns={'id':'station_id'}, inplace=True)

# Confirm name change

stations.head()

# Join Dataset with stations

df_temp2 = pd.merge(temp_df, stations, on = 'station_id', how = 'inner')

# Confirm Join

df_temp2.head()

def extract_coordinates(point_str):
    point = loads(point_str)
    return point.x, point.y

df_temp2['longitude'], df_temp2['latitude'] = zip(*df_temp2['geometry'].apply(extract_coordinates))

# Join Dataset with Lines

df = pd.merge(df_temp2, lines, on = 'line_id', how = 'inner')

# Confirm Change

df.head()

# View all columns for the final dataset

df.info()

# Rename for better understanding

df.rename(columns = {'name_x':'city_name', 'name_y': 'line_name'}, inplace  = True)

# Confirm change

df.info()

# Remove Columns not needed

df.drop(['city_id_x','coords','start_year','url_name_x','country_state','length','created_at','updated_at','fromyear','toyear','deprecated_line_group', 'url_name_y', 'color'], axis=1, inplace=True)

# Confirm columns removal

df.head()

df.info()

df.drop(['id','geometry','buildstart', 'opening', 'closure', 'city_id_y', 'city_id', 'name', 'system_id'], axis=1, inplace=True)

# Set Filter dataset for London

selected_city = 'London'

# Enable Filter

london = df[df['city_name'] == selected_city]

# View filtered dataset

london.head()
london.info()

# Aggregate data by City Name and Station ID

agrregated_city_df = london.groupby(['city_name', 'station_id', 'latitude', 'longitude', 'transport_mode_id'], as_index=False).agg({'station_id':'first', 'line_id':lambda x: ', '.join(map(str, x))})

# View columns data

agrregated_city_df.info()

# Join Line ID with only one stationID

agrregated_city_df['line_id'] = agrregated_city_df['line_id'].apply(lambda x: x.split(','))

# Repeat Station ID for each Line ID

expanded_df = agrregated_city_df.explode('line_id').reset_index(drop = True)

expanded_df.head(20)

expanded_df.info()

#_____________________________________#
#_____________________________________#
#________________PLOTS________________#
#_____________________________________#
#_____________________________________#

# Function to transform latitude and longitude to projected coordinates
def latlon_to_xy(lat, lon):
    in_proj = Proj('epsg:4326')  # WGS84
    out_proj = Proj('epsg:3857')  # Web Mercator
    x, y = transform(in_proj, out_proj, lon, lat)
    return x, y

#_____________________________________#

# Function to calculate Haversine distance between two points given their latitudes and longitudes
def haversine_distance(lat1, lon1, lat2, lon2):
    import math

    R = 6371  # Earth radius in kilometers

    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Calculate distance
    distance = R * c
    return distance

#_____________________________________#
#_____________________________________#
#_____________________________________#

# Create a graph
G = nx.Graph()

# Add nodes and weighted edges to the graph
for i, row in expanded_df.iterrows():
    G.add_node(row['station_id'], line=row['line_id'], latitude=row['latitude'], longitude=row['longitude'])
    if i > 0:
        prev_row = expanded_df.iloc[i - 1]
        distance = haversine_distance(row['latitude'], row['longitude'], prev_row['latitude'], prev_row['longitude'])
        G.add_edge(row['station_id'], prev_row['station_id'], weight=distance)

# Create a new figure for the graph
plt.figure(figsize=(10, 10))
# Draw the graph
pos = {node: latlon_to_xy(data['latitude'], data['longitude']) for node, data in G.nodes(data=True)}
nx.draw_networkx(G, pos, with_labels=True, node_size=50, width=0.5)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.axis('equal')
plt.savefig('graph_plot.png')
plt.close()

#_____________________________________#
#_____________________________________#
#_____________________________________#

# Create a GeoDataFrame
# Create a GeoDataFrame for the stations
# Merge expanded_df with transport_modes to get the names of transport modes
expanded_df = expanded_df.merge(transport_modes[['id', 'name']], left_on='transport_mode_id', right_on='id', how='left')

# Map the transport mode names to colors
expanded_df['color'] = expanded_df['name'].map(modes_colors)

# Create GeoDataFrame for stations
gdf_stations = gpd.GeoDataFrame(
    expanded_df,
    geometry=gpd.points_from_xy(expanded_df.longitude, expanded_df.latitude),
    crs="EPSG:4326"
)

# Convert to Web Mercator
gdf_stations = gdf_stations.to_crs(epsg=3857)

# Create LineStrings for each track
lines = []
for line_id in expanded_df['line_id'].unique():
    line_stations = expanded_df[expanded_df['line_id'] == line_id]
    if len(line_stations) > 1:
        line_geometry = LineString(zip(line_stations.longitude, line_stations.latitude))
        lines.append({'geometry': line_geometry, 'line_id': line_id, 'color': line_stations['color'].iloc[0]})

# Create GeoDataFrame for the lines
gdf_lines = gpd.GeoDataFrame(lines, crs="EPSG:4326")
gdf_lines = gdf_lines.to_crs(epsg=3857)

# Plotting
fig, ax = plt.subplots(figsize=(40, 40))

# Plot lines with colors
for _, row in gdf_lines.iterrows():
    ax.plot(*row['geometry'].xy, color=row['color'], linewidth=2, zorder=1)

# Plot stations
gdf_stations.plot(ax=ax, color='blue', edgecolor='blue', markersize=50, zorder=2)

# Add basemap
ctx.add_basemap(ax, crs=gdf_stations.crs.to_string(), source=ctx.providers.OpenStreetMap.Mapnik)

ax.set_axis_off()

# Save and show the figure
plt.savefig('./map_plot.png')
plt.close()
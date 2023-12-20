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
from pyproj import Transformer
from random import randint

cities = pd.read_csv('../data/cities.csv')
stations = pd.read_csv('../data/stations.csv')
lines = pd.read_csv('../data/lines.csv')
systems = pd.read_csv('../data/systems.csv')
station_lines = pd.read_csv('../data/station_lines.csv')
transport_modes = pd.read_csv('../data/transport_modes.csv')

modes_colors = {
    'default': '#000000',  # Black
    'high_speed_rail': '#FF0000',  # Bright Red
    'inter_city_rail': 'teal',  # Bright Yellow
    'commuter_rail': '#00FF00',  # Bright Green
    'heavy_rail': '#0000FF',  # Bright Blue
    'light_rail': '#FF00FF',  # Magenta
    'brt': '#FFC0CB',  # Light Pink
    'people_mover': '#FFA500',  # Orange
    'bus': '#800080',  # Purple
    'tram': '#00FFFF',  # Cyan
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

expanded_df.dropna();

expanded_df.head(20)

expanded_df.info()

#_____________________________________#
#_____________________________________#
#________________PLOTS________________#
#_____________________________________#
#_____________________________________#

# Function to transform latitude and longitude to projected coordinates
def latlon_to_xy(lat, lon):
    transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
    x, y = transformer.transform(lon, lat)
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
    return round(distance, 2)

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
        if prev_row["station_id"] != row["station_id"]:
            distance = haversine_distance(row['latitude'], row['longitude'], prev_row['latitude'], prev_row['longitude'])
            G.add_edge(row['station_id'], prev_row['station_id'], weight=distance)

# Calculate shortest paths after constructing the graph
shortest_path_lengths_alt = dict(nx.all_pairs_shortest_path_length(G))

# Example of printing shortest path lengths between specific nodes
# You can adjust the printing part according to your needs
for node1 in G.nodes():
    for node2 in G.nodes():
        if node1 != node2:
            print(f"Shortest path between {node1} & {node2}: {shortest_path_lengths_alt[node1][node2]}")

# Create a new figure for the graph
plt.figure(figsize=(50, 50))
# Draw the graph
pos = {node: latlon_to_xy(data['latitude'], data['longitude']) for node, data in G.nodes(data=True)}
nx.draw_networkx(G, pos, with_labels=True, node_size=50, width=0.5)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.axis('equal')
plt.savefig('graph_plot.png')
plt.close()

#_____________________________________#
#___________Graph Metrics_____________#
#_____________________________________#
shortest_path_lengths = dict(nx.all_pairs_shortest_path_length(G))
number_of_nodes = G.number_of_nodes();
number_of_edges = G.number_of_edges();
print("________________")
print("Nodes: ", number_of_nodes, "Edges: ", number_of_nodes)
print("________________")

degree_avg = np.mean([d for _, d in G.degree()])
print("________________")
print("Degree Average: ", degree_avg)
print("________________")

diameter = max(nx.eccentricity(G, sp=shortest_path_lengths).values())
print("________________")
print("Network Diameter: ", diameter)
print("________________")

# Compute the average shortest path length for each node
average_path_lengths = [
    np.mean(list(spl.values())) for spl in shortest_path_lengths.values()
]
# The average over all nodes
path_lengths_avg = np.mean(average_path_lengths)
print("________________")
print("Path Lengths Average: ", path_lengths_avg)
print("________________")

density = nx.density(G)
print("________________")
print("Density: ", density)
print("________________")

connected = nx.number_connected_components(G)
print("________________")
print("Connected Components: ", connected)
print("________________")

degree_centrality = nx.centrality.degree_centrality(
    G
)  # save results in a variable to use again
(sorted(degree_centrality.items(), key=lambda item: item[1], reverse=True))[:8]
print("________________")
print("Degree Centrality: ", degree_centrality)
print("________________")

sorted_degree = (sorted(G.degree, key=lambda item: item[1], reverse=True))[:8]
print("________________")
print("Sorted Degree Centrality: ", sorted_degree)
print("________________")

betweenness_centrality = nx.centrality.betweenness_centrality(
    G
)  # save results in a variable to use again
(sorted(betweenness_centrality.items(), key=lambda item: item[1], reverse=True))[:8]

print("________________")
print("Betweenness Centrality: ", betweenness_centrality)
print("________________")

closeness_centrality = nx.centrality.closeness_centrality(
    G
)  # save results in a variable to use again
(sorted(closeness_centrality.items(), key=lambda item: item[1], reverse=True))[:8]

print("________________")
print("Closeness Centrality: ", closeness_centrality)
print("________________")

average_distance = 1 / closeness_centrality[25709]

print("________________")
print("Average Distance: ", average_distance)
print("________________")

avg_clustering = nx.average_clustering(G)
print("________________")
print("Average Clustering: ", avg_clustering)
print("________________")

triangles_per_node = list(nx.triangles(G).values())
sum(
    triangles_per_node
) / 3
print("________________")
print("Triangles Per Node: ", triangles_per_node)
print("________________")

avg_triangles = np.mean(triangles_per_node);
print("________________")
print("Average Triangles: ", avg_triangles)
print("________________")

median_triangles = np.median(triangles_per_node)
print("________________")
print("Median Triangles: ", median_triangles)
print("________________")

has_bridges = nx.has_bridges(G)
print("________________")
print("Has Bridges: ", has_bridges)
print("________________")

bridges = list(nx.bridges(G))
amount_bridges = len(bridges)
print("________________")
print("Bridges: ", amount_bridges)
print("________________")

local_bridges = list(nx.local_bridges(G, with_span=False))
amount_local_bridges = len(local_bridges)
print("________________")
print("Local Bridges: ", amount_local_bridges)
print("________________")

degree_assortativity_coefficient = nx.degree_assortativity_coefficient(G)
print("________________")
print("Degree Assortativity Coefficient: ", degree_assortativity_coefficient)
print("________________")

degree_pearson_correlation_coefficient = nx.degree_pearson_correlation_coefficient(
    G
)
print("________________")
print("Degree Pearson Correlation Coefficient: ", degree_pearson_correlation_coefficient)
print("________________")

#___________________________________________________#
#________Network Path Lengths Visualization_________#
#___________________________________________________#

# We know the maximum shortest path length (the diameter), so create an array
# to store values from 0 up to (and including) diameter
path_lengths = np.zeros(diameter + 1, dtype=int)

# Extract the frequency of shortest path lengths between two nodes
for pls in shortest_path_lengths.values():
    pl, cnts = np.unique(list(pls.values()), return_counts=True)
    path_lengths[pl] += cnts

# Express frequency distribution as a percentage (ignoring path lengths of 0)
freq_percent = 100 * path_lengths[1:] / path_lengths[1:].sum()

# Plot the frequency distribution (ignoring path lengths of 0) as a percentage
fig, ax = plt.subplots(figsize=(15, 8))
ax.bar(np.arange(1, diameter + 1), height=freq_percent)
ax.set_title(
    "Distribution of shortest path length in G", fontdict={"size": 35}, loc="center"
)
ax.set_xlabel("Shortest Path Length", fontdict={"size": 22})
ax.set_ylabel("Frequency (%)", fontdict={"size": 22})
plt.savefig('./path_lengths_visualization.png')
plt.close()

#____________________________________________________________________#
#_____________Degree Centrality Visualization________________________#
#____________________________________________________________________#
plt.figure(figsize=(15, 8))
plt.hist(degree_centrality.values(), bins=25)
plt.xticks(ticks=[0, 0.00025, 0.0005, 0.001, 0.0015, 0.002])  # set the x axis ticks
plt.title("Degree Centrality Histogram ", fontdict={"size": 35}, loc="center")
plt.xlabel("Degree Centrality", fontdict={"size": 20})
plt.ylabel("Counts", fontdict={"size": 20})
plt.savefig('./degree_centrality_visualization.png')
plt.close()

#____________________________________________________________________#
#______________Top Degree Centrality Visualization___________________#
#____________________________________________________________________#

node_size = [
    v * 1000 for v in degree_centrality.values()
]  # set up nodes size for a nice graph representation
plt.figure(figsize=(15, 8))
nx.draw_networkx(G, pos=pos, node_size=node_size, with_labels=False, width=0.15)
plt.axis("off")
plt.savefig('./top_degree_centrality_visualization.png')
plt.close()
#_____________________________________________________________________#
#_____________Betweenness Centrality Visualization____________________#
#_____________________________________________________________________#

plt.figure(figsize=(15, 8))
plt.hist(betweenness_centrality.values(), bins=100)
plt.xticks(ticks=[0, 0.02, 0.1, 0.2, 0.3, 0.4, 0.5])  # set the x axis ticks
plt.title("Betweenness Centrality Histogram ", fontdict={"size": 35}, loc="center")
plt.xlabel("Betweenness Centrality", fontdict={"size": 20})
plt.ylabel("Counts", fontdict={"size": 20})
plt.savefig('./betweenness_centrality_visualization.png')
plt.close()
#__________________________________________________________________________________#
#_________________________Betweenness Centralities Visualization___________________#
#__________________________________________________________________________________#

node_size = [
    v * 1200 for v in betweenness_centrality.values()
]  # set up nodes size for a nice graph representation
plt.figure(figsize=(15, 8))
nx.draw_networkx(G, pos=pos, node_size=node_size, with_labels=False, width=0.15)
plt.axis("off")
plt.savefig('./betweenness_centralities_visualization.png')
plt.close()

#__________________________________________________________________________________#
#_________________________Closeness Centralities Visualization___________________#
#__________________________________________________________________________________#

plt.figure(figsize=(15, 8))
plt.hist(closeness_centrality.values(), bins=60)
plt.title("Closeness Centrality Histogram ", fontdict={"size": 35}, loc="center")
plt.xlabel("Closeness Centrality", fontdict={"size": 20})
plt.ylabel("Counts", fontdict={"size": 20})
plt.savefig('./closeness_centralities_visualization.png')
plt.close()

#__________________________________________________________________________________#
#_____________________Min Closeness Centralities Visualization_____________________#
#__________________________________________________________________________________#

node_size = [
    v * 50 for v in closeness_centrality.values()
]  # set up nodes size for a nice graph representation
plt.figure(figsize=(15, 8))
nx.draw_networkx(G, pos=pos, node_size=node_size, with_labels=False, width=0.15)
plt.axis("off")
plt.savefig('./minimum_centralities_visualization.png')
plt.close()

#__________________________________________________________________________________#
#_______________Clustering Coefficient Histogram Visualization_____________________#
#__________________________________________________________________________________#
plt.figure(figsize=(15, 8))
plt.hist(nx.clustering(G).values(), bins=50)
plt.title("Clustering Coefficient Histogram ", fontdict={"size": 35}, loc="center")
plt.xlabel("Clustering Coefficient", fontdict={"size": 20})
plt.ylabel("Counts", fontdict={"size": 20})

plt.savefig('./clustering_coefficient_visualization.png')
plt.close()

#__________________________________________________________________________________#
#________________________________Local Bridges_____________________________________#
#__________________________________________________________________________________#
plt.figure(figsize=(15, 8))
nx.draw_networkx(G, pos=pos, node_size=10, with_labels=False, width=0.15)
nx.draw_networkx_edges(
    G, pos, edgelist=local_bridges, width=0.5, edge_color="lawngreen"
)  # green color for local bridges
nx.draw_networkx_edges(
    G, pos, edgelist=bridges, width=0.5, edge_color="r"
)  # red color for bridges
plt.axis("off")

plt.savefig('./local_bridges_visualization.png')
plt.close()

#__________________________________________________________________________________#
#_____________________________Communities Visualization____________________________#
#__________________________________________________________________________________#
colors = {}

for com in nx.community.label_propagation_communities(G):
    color = "#%06X" % randint(0, 0xFFFFFF)  # creates random RGB color
    for node in list(com):
        colors[node] = color
        
node_colors = [colors[node] for node in G.nodes()]

plt.figure(figsize=(15, 9))
plt.axis("off")
nx.draw_networkx(
    G, pos=pos, node_size=10, with_labels=False, width=0.15, node_color=node_colors
)
plt.savefig('./communities_visualization.png')
plt.close()

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

# Ensure the color is correctly mapped to the stations
gdf_stations['color'] = gdf_stations['name'].map(modes_colors)

# Plot stations with colors based on transport modes using scatter
for _, row in gdf_stations.iterrows():
    ax.scatter(row.geometry.x, row.geometry.y, color=row['color'], s=50, zorder=2, edgecolor='white')

# Add basemap
ctx.add_basemap(ax, crs=gdf_stations.crs.to_string(), source=ctx.providers.CartoDB.Positron)

ax.set_axis_off()

# Save and show the figure
plt.savefig('./map_plot.png')
plt.close()
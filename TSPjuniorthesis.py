"""
Author: Shep O'Keeffe
Date: _/_/2026
Description:
Sources:
Bugs:
"""

#Street mapping software
import osmnx as ox


import matplotlib.pyplot as plt

place_name = "Greenwich, Connecticut, USA"

G = ox.graph_from_place(place_name, network_type="drive")

fig, ax = ox.plot_graph(G)

gdf_nodes, gdf_edges = ox.graph_to_gdfs(G)

print(f"Graph nodes: {len(gdf_nodes)}")
print(f"Graph edges: {len(gdf_edges)}")
"""
Author: Shep O'Keeffe
Date: 4/23/2026
Description: A Clarke-Wright Savings Vehicle Routing Problem (VRP)-solving algorithm used to alter the bus routes of a school's bus routes. The school, stops, buses, and
  existing route data are modeled after Greenwich High School, a large public school in Greenwich, Connecticut, USA. The stops are plotted using their latitude and longitude
  and the street mapping software osmnx. When the program is run, new and old routes are diplayed visually using matplotlib. Total distance traveled by buses on new and old
  routes, along with the improvement of the former over the latter, are printed in the terminal.
Sources: osmnx, networkx, matplotlib, numpy, scitools, stack overflow, geeksforgeeks, VeRyPy at https://github.com/yorak/VeRyPy (for inspiration)
"""

import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

#data model

@dataclass
class Stop:
    id: int
    name: str
    lat: float
    lon: float
    students: int

@dataclass
class Bus:
    id: int
    capacity: int

@dataclass
class Route:
    bus: Bus
    stops: List[Stop] = field(default_factory=list)

    @property
    def total_students(self):
        return sum(s.students for s in self.stops)

    @property
    def is_feasible(self):
        return self.total_students <= self.bus.capacity


#loading road network with OSMNX

def load_road_network(school: Stop, stops: List[Stop], buffer_dist=2000):
    """
    Desc: downloads the drivable street graph centered around the stops
    buffer_dist: how many meters beyond the stops to include (avoids routing that runs off the edge of the graph).
    """
    all_lats = [school.lat] + [s.lat for s in stops]
    all_lons = [school.lon] + [s.lon for s in stops]

    center_lat = sum(all_lats) / len(all_lats)
    center_lon = sum(all_lons) / len(all_lons)

    #computing a radius that covers all points plus buffer
    max_dist = max(ox.distance.great_circle(center_lat, center_lon, lat, lon) for lat, lon in zip(all_lats, all_lons))
    radius = max_dist + buffer_dist

    print(f"Downloading road network (radius={radius:.0f}m)...")
    G = ox.graph_from_point((center_lat, center_lon), dist=radius, network_type='drive')
    return G


def snap_to_graph(G, lat, lon):
    #finding the nearest OSMnx node to a lat/lon coordinate
    return ox.distance.nearest_nodes(G, lon, lat) #osmnx takes (lon, lat)


def build_road_distance_matrix(G, node_ids: List[int], n: int):
    """
    Desc: builds an (N+1) x (N+1) distance matrix using real road distances. index 0 = school, indices 1..N = stops. returns the matrix and the list of OSMnx node IDs for each location
    Returns: matrix (np.ndarray)
    """
    matrix = np.zeros((n, n))

    print(f"Computing {n*n} road distances...")
    for i in range(n):
        #nx.single_source_dijkstra_path_length computes all distances from node i in one pass
        lengths = dict(nx.single_source_dijkstra_path_length(G, node_ids[i], weight='length'))
        for j in range(n):
            if i != j:
                dist = lengths.get(node_ids[j], float('inf'))
                matrix[i][j] = dist #distance in meters

    return matrix

print("debug1")

# Clarke-Wright Savings algo
# intuition: start with every stop on its own route (school -> stop -> school).
# a "saving" is the distance avoided by chaining two stops together.
# s(i,j) = d(depot,i) + d(depot,j) - d(i,j)
# merge routes greedily from highest to lowest saving, respecting capacity.

def compute_savings(dist_matrix: np.ndarray, n_stops: int):
    """
    Returns a sorted list of (saving, i, j) for all stop pairs.
    Indices are 1-based (0 = depot/school)
    """
    savings = []
    for i in range(1, n_stops + 1):
        for j in range(1, n_stops + 1):
            if i >= j:
                continue
            saving = dist_matrix[0][i] + dist_matrix[0][j] - dist_matrix[i][j]
            savings.append((saving, i, j))
    savings.sort(reverse=True)
    return savings

print("debug2")

def clarke_wright_savings(school: Stop, stops: List[Stop], buses: List[Bus],dist_matrix: np.ndarray):
    """
    Clarke-Wright savings algorithm implemented from scratch.
    Each stop index is 1-based to match the distance matrix
    (where index 0 = school depot)
    Returns the route
    """
    n = len(stops)
    max_capacity = max(b.capacity for b in buses)

    #each stop starts as its own isolated route: [stop_i]
    #tracking routes as lists of stop indices (1-based)
    routes: Dict[int, List[int]] = {i+1: [i+1] for i in range(n)}

    #map stop index - which route list it currently belongs to (identified by the index of the first stop in that route)
    stop_to_route: Dict[int, int] = {i+1: i+1 for i in range(n)}

    def route_demand(route_key):
        return sum(stops[idx-1].students for idx in routes[route_key])

    savings = compute_savings(dist_matrix, n)

    for saving, i, j in savings:
        ri = stop_to_route.get(i)
        rj = stop_to_route.get(j)

        #skips if already on the same route or route no longer exists
        if ri is None or rj is None or ri == rj:
            continue

        route_i = routes[ri]
        route_j = routes[rj]

        #Clarke-Wright rule: can only merge if i is at the tail of its route and j is at the head of its route (or vice versa)
        i_at_tail = route_i[-1] == i
        j_at_head = route_j[0] == j
        j_at_tail = route_j[-1] == j
        i_at_head = route_i[0] == i

        if i_at_tail and j_at_head:
            merged = route_i + route_j
        elif j_at_tail and i_at_head:
            merged = route_j + route_i
        else:
            continue #interior stops can't be merged this way

        #checking capacity
        if route_demand(ri) + route_demand(rj) > max_capacity:
            continue

        #committing the merge
        new_key = merged[0]
        routes[new_key] = merged

        #cleaning up old route keys
        for old_key in [ri, rj]:
            if old_key != new_key and old_key in routes:
                del routes[old_key]

        #updating stop to route mapping
        for idx in merged:
            stop_to_route[idx] = new_key

    #2-opt local improvement
    #for each route, tries reversing every sub-segment and keeps it if it reduces total distance, repeats until no improvement is found

    def route_distance_by_indices(stop_indices: List[int]):
        """
        Returns sitance (float)
        """
        if not stop_indices:
            return 0.0
        total = dist_matrix[0][stop_indices[0]] # depot -> first
        for k in range(len(stop_indices) - 1):
            total += dist_matrix[stop_indices[k]][stop_indices[k+1]]
        total += dist_matrix[stop_indices[-1]][0] # last -> depot
        return total

    def two_opt(stop_indices: List[int]):
        """
        Returns improved stop indices (list)
        """
        best = stop_indices[:]
        improved = True
        while improved:
            improved = False
            for a in range(len(best) - 1):
                for b in range(a + 2, len(best)):
                    #reversing the segment between a+1 and b
                    candidate = best[:a+1] + best[a+1:b+1][::-1] + best[b+1:]
                    if route_distance_by_indices(candidate) < route_distance_by_indices(best):
                        best = candidate
                        improved = True
        return best

    print("Running 2-opt improvement on each route...")
    for key in list(routes.keys()):
        routes[key] = two_opt(routes[key])

    #assigning routes to buses
    result = []
    available_buses = list(buses)
    for key, stop_indices in routes.items():
        if not available_buses:
            print("Warning: more routes than buses!")
            break
        bus = available_buses.pop(0)
        stop_list = [stops[idx-1] for idx in stop_indices]
        result.append(Route(bus=bus, stops=stop_list))

    return result

print("debug3")

#route distance using road network

def get_road_path(G, node_ids: List[int], from_idx: int, to_idx: int):
    """
    Returns the list of OSMnx nodes forming the shortest path between two locations
    """
    try:
        return nx.shortest_path(G, node_ids[from_idx], node_ids[to_idx], weight='length')
    except nx.NetworkXNoPath:
        return []

print("debug4")

def route_total_distance(route: Route, dist_matrix: np.ndarray, stops: List[Stop]):
    """
    Returns total distance traveled on route (float)
    """
    stop_index = {s.id: i+1 for i, s in enumerate(stops)}
    if not route.stops:
        return 0.0
    total = dist_matrix[0][stop_index[route.stops[0].id]]
    for k in range(len(route.stops) - 1):
        a = stop_index[route.stops[k].id]
        b = stop_index[route.stops[k+1].id]
        total += dist_matrix[a][b]
    total += dist_matrix[stop_index[route.stops[-1].id]][0]
    return total

print("debug5")

#visualization w/ matplotlib & osmnx

ROUTE_COLORS = ['#E8593C', '#3B8BD4', '#1D9E75', '#BA7517', '#9966CC', '#D4537E', '#639922', '#185FA5', '#993C1D', '#085041']

def plot_routes(G, routes: List[Route], school: Stop, stops: List[Stop], node_ids: List[int], dist_matrix: np.ndarray, title: str = "Optimized Bus Routes"):
    """
    Draw the street network with each bus route overlaid as a colored path.
    """
    fig, ax = ox.plot_graph(G, show=False, close=False, bgcolor='#1a1a2e', node_size=0, edge_color='#2d2d4e', edge_linewidth=0.5, figsize=(14, 12))

    stop_index = {s.id: i+1 for i, s in enumerate(stops)}
    legend_patches = []

    for route_idx, route in enumerate(routes):
        color = ROUTE_COLORS[route_idx % len(ROUTE_COLORS)]
        route_stop_indices = [0] + [stop_index[s.id] for s in route.stops] + [0]

        #drawing each road segment as the actual street path
        for k in range(len(route_stop_indices) - 1):
            from_node = node_ids[route_stop_indices[k]]
            to_node = node_ids[route_stop_indices[k + 1]]
            try:
                path_nodes = nx.shortest_path(G, from_node, to_node, weight='length')
                #getting x/y coordinates of each node in the path
                xs = [G.nodes[n]['x'] for n in path_nodes]
                ys = [G.nodes[n]['y'] for n in path_nodes]
                ax.plot(xs, ys, color=color, linewidth=2.5, alpha=0.85, zorder=3)
            except nx.NetworkXNoPath:
                pass

        #drawing stop markers
        for stop in route.stops:
            node = node_ids[stop_index[stop.id]]
            x, y = G.nodes[node]['x'], G.nodes[node]['y']
            ax.scatter(x, y, c=color, s=80, zorder=5, edgecolors='white', linewidths=0.8)
            ax.annotate(str(stop.id), (x, y), textcoords="offset points", xytext=(-4, -2), fontsize=6, color='white', alpha=0.9, zorder=6)

        dist_m = route_total_distance(route, dist_matrix, stops)
        label = (f"Bus {route.bus.id} | "
                f"{route.total_students}/{route.bus.capacity} students | "
                f"{dist_m/1609:.1f} mi")
        legend_patches.append(mpatches.Patch(color=color, label=label))

    #drawing the school
    school_node = node_ids[0]
    sx, sy = G.nodes[school_node]['x'], G.nodes[school_node]['y']
    ax.scatter(sx, sy, c='#FFD700', s=200, zorder=7, marker='*', edgecolors='white', linewidths=1)
    ax.annotate(school.name, (sx, sy), textcoords="offset points", xytext=(8, 6), fontsize=5, color='#FFD700', fontweight='bold', zorder=8)

    ax.legend(handles=legend_patches, loc='lower left', facecolor='#1a1a2e', edgecolor='#444', labelcolor='white', fontsize=5, framealpha=0.85)
    ax.set_title(title, color='white', fontsize=10, pad=12)
    fig.tight_layout()
    plt.show()

print("debug6")

#old vs new route comparison

def build_routes_from_existing(existing_data: dict, buses: List[Bus], stops: List[Stop]):
    """
    Load current routes from dict:
    {
    1: [stop_id, stop_id, ...], # bus 1 visits these stops in order
    2: [stop_id, stop_id, ...],
    }
    Returns routes (lists)
    """
    stop_by_id = {s.id: s for s in stops}
    routes = []
    for i, (bus_id, stop_ids) in enumerate(existing_data.items()):
        bus = next(b for b in buses if b.id == bus_id)
        stop_list = [stop_by_id[sid] for sid in stop_ids]
        routes.append(Route(bus=bus, stops=stop_list))
    return routes

print("debug7")

#main

if __name__ == "__main__":

    school = Stop(id=0, name="Greenwich High School", lat=41.040485, lon=-73.61156, students=0)

    stops = [
        Stop(1, "N WATER ST & S NEW ST",     41.005099, -73.656829, students=7),
        Stop(2, "WEST PUTNAM AV @ WESTERN JR HWY", 41.014579, -73.652008, students=7),
        Stop(3, "DELAVAN AVE & NEW LEBANON AVE",   41.00297,  -73.653709, students=5),
        Stop(4, "S WATER ST & MEAD AVE",     40.996287, -73.658535, students=5),
        Stop(5, "S WATER ST @ DIVISION ST W",      41.001617, -73.658736, students=5),
        Stop(6, "RIVER AV @ BYRAM SHORE RD", 40.994815, -73.655064, students=5),
        Stop(7, "BYRAM SHORE RD @ JAMES ST E",  40.998406, -73.65255, students=5),
        Stop(8, "BYRAM SHORE RD @ RITCH AV W",  41.003668, -73.649318, students=5),
        Stop(9, "HAMILTON AV @ LIVINGSTON PLACE", 41.02139, -73.637058, students=7),
        Stop(10, "HAMILTON AV @ ST ROCHS",  41.017305,  -73.639998, students=7),
        Stop(11, "PEMBERWICK RD @ DEN LA",  41.016675,	-73.654547, students=3),
        Stop(12,"PEMBERWICK RD @ MOSHIER ST",	41.021484,	-73.655935, students=3),
        Stop(13,"WEAVER ST @ MOSHIER ST",	41.022714,	-73.650713, students=3),
        Stop(14,"W PUTNAM AVE & VALLEY DR",	41.018435,	-73.64584, students=3),
        Stop(15,"W PUTNAM AVE & HAROLD AVE",	41.020593,	-73.642002, students=3),
        Stop(16,"PEMBERWICK RD @ DEN LA 2",	41.016675,	-73.654547, students=2),
        Stop(17,"PEMBERWICK RD @ MOSHIER ST 2",	41.021484,	-73.655935, students=2),
        Stop(18,"PEMBERWICK RD @ COMLY AV",	41.027568,	-73.661335, students=2),
        Stop(19,"PEMBERWICK RD @ BUENA VISTA DR",	41.031871,	-73.663937, students=2),
        Stop(20,"GLENVILLE RD @ WEAVER ST",	41.037807,	-73.662193, students=2),
        Stop(21,"GLENVILLE RD @ HUNTZINGER DR",	41.034942,	-73.645928, students=2),
        Stop(22,"VALLEY DR & EDGEWOOD PL",	41.028562,	-73.641811, students=2),
        Stop(23,"EDGEWOOD DR @ HEMLOCK DR",	41.026544,	-73.637901, students=2),
        Stop(24,"WEAVER ST & HAWTHORNE ST N",	41.036496,	-73.661379, students=2),
        Stop(25,"WEAVER ST & TREE TOP TER",	41.033702,	-73.658609, students=2),
        Stop(26,"WEAVER ST @ FLINTLOCK RD",	41.031504,	-73.656774, students=2),
        Stop(27,"WEAVER ST @ WEST LYON FARM DR",	41.029739,	-73.653929, students=2),
        Stop(28,"Weaver St & Calhourn Dr",	41.027445,	-73.651963, students=2),
        Stop(29,"WEAVER ST @ MOSHIER ST",	41.022714,	-73.650713, students=2),
        Stop(30,"WEAVER ST @ EAST WEAVER ST",	41.017408,	-73.65016, students=2),
        Stop(31,"HAMILTON AV @ ARTIC ST",	41.019633,	-73.638256, students=3),
        Stop(32,"OLD FIELD POINT RD & OLD TRACK RD",	41.020901,	-73.634533, students=3),
        Stop(33,"FIELD POINT RD & BUSH AVE",	41.013677,	-73.633234, students=3),
        Stop(34,"SHORE RD @ FIELD PT RD",	41.012221,	-73.63281, students=3),
        Stop(35,"ONEIDA DR @ STEAMBOAT RD",	41.016864,	-73.622942, students=3),
        Stop(36,"LAKE AV @ MT LAUREL DR",	41.128169,	-73.657366, students=1),
        Stop(37,"1016 Lake Ave",	41.11973,	-73.654152, students=1),
        Stop(38,"LAKE AV @ CLOSE RD",	41.116502,	-73.655748, students=1),
        Stop(39,"LAKE AVE & MERRY LN",	41.115535,	-73.655366, students=1),
        Stop(40,"98 LOWER CROSS RD",	41.110572,	-73.64636, students=1),
        Stop(41,"LAKE AVE & OLD MILL RD",	41.090675,	-73.647663, students=1),
        Stop(42,"742 Lake Avenue",	41.08325,	-73.646903, students=1),
        Stop(43,"CLAPBOARD RIDGE RD @ DAIRY RD",	41.077774,	-73.636302, students=1),
        Stop(44,"636 Lake Ave",	41.070201,	-73.639595, students=1),
        Stop(45,"601 Lake Avenue",	41.065301,	-73.639209, students=1),
        Stop(46,"ROCKWOOD LA @ LAUREL LA",	41.059494,	-73.63403, students=1),
        Stop(47,"HUSTED LA @ LAUREL LA",	41.057891,	-73.630309, students=1),
        Stop(48,"PARSONAGE RD @ HUSTED LA",	41.063113,	-73.630962, students=1),
        Stop(49,"BISHOP DR N & BISHOP DR S",	41.030297,	-73.668949, students=1),
        Stop(50,"KING ST @ NEDLEY LA",	41.030879,	-73.670631, students=1),
        Stop(51,"929 King St",	41.034872,	-73.673824, students=1),
        Stop(52,"SHADY LA @ WATCH HILL DR",	41.036174,	-73.671896, students=1),
        Stop(53,"SHADY LA @ GLEN RIDGE RD",	41.037753,	-73.671148, students=1),
        Stop(54,"KING ST @ RINCARD TER",	41.043616,	-73.680814, students=1),
        Stop(55,"BOWMAN DR N @ STONEHEDGE DR N",	41.048031,	-73.673803, students=1),
        Stop(56,"STONEHEDGE DR SOUTH @ BOW MAN DR SOUTH",	41.041487,	-73.676367, students=1),
        Stop(57,"KING ST @ HETTIEFRED RD",	41.049115,	-73.683458, students=1),
        Stop(58,"SHERWOOD AVE & NUTMEG DR",	41.057661,	-73.683311, students=1),
        Stop(59,"SHERWOOD AV @ ALEC TEMPLETON LN",	41.060611,	-73.678397, students=1),
        Stop(60,"PECKSLAND RD & ZACCHEUS MEAD LN",	41.05532,	-73.652653, students=1),
        Stop(61,"ZACCHEUS MEAD LN & WINDING LN",	41.043041,	-73.643182, students=1),
        Stop(62,"SKYLARK RD @ GLEN RD",	41.037881,	-73.630385, students=1),
        Stop(63,"HAMILTON AV @ ARMSTRONG COURT",	41.013085,	-73.642622, students=14)
    ]
 
    buses = [Bus(1, 16), Bus(2, 16), Bus(3, 16), Bus(4, 16), Bus(5, 16), Bus(6, 16), Bus(7, 16), Bus(8, 16), Bus(9, 16), Bus(10, 16), 
             Bus(11, 16)]
 
    existing_route_data = {
        1: [1, 2],
        2: [3, 4, 5],
        3: [6, 7, 8],
        4: [9, 10],
        5: [11, 12, 13, 14, 15],
        6: [16, 17, 18, 19, 20, 21, 22, 23],
        7: [24, 25, 26, 27, 28, 29, 30],
        8: [31, 32, 33, 34, 35],
        9: [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48],
        10: [49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62],
        11: [63]
    }

    

    #load graph
    G = load_road_network(school, stops)

    #snap all locations to the road graph
    all_locations = [school] + stops
    node_ids = [snap_to_graph(G, loc.lat, loc.lon) for loc in all_locations]

    G_projected = ox.project_graph(G)

    #build road-distance matrix
    dist_matrix = build_road_distance_matrix(G_projected, node_ids, len(all_locations))

    #build and plot current routes
    current_routes = build_routes_from_existing(existing_route_data, buses, stops)
    total_current = sum(route_total_distance(r, dist_matrix, stops) for r in current_routes)
    print(f"Current total distance: {total_current/1609:.2f} miles")
    plot_routes(G_projected, current_routes, school, stops, node_ids, dist_matrix, title="Current Bus Routes")

    #run algorithm and plot optimized routes
    optimized_routes = clarke_wright_savings(school, stops, buses, dist_matrix)
    total_optimized = sum(route_total_distance(r, dist_matrix, stops) for r in optimized_routes)
    print(f"Optimized total distance: {total_optimized/1609:.2f} miles")
    print(f"Improvement: {(total_current - total_optimized)/total_current*100:.1f}%")
    plot_routes(G_projected, optimized_routes, school, stops, node_ids, dist_matrix, title="Optimized Bus Routes")

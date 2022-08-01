from manim import *
import networkx as nx
import numpy as np
import itertools as it


def swap_random(seq, N=None):
    """
    Randomly swap two positions on an array N times
    """
    idx = range(len(seq))
    n = N if N else np.random.randint(0, 20)
    output = seq.copy()
    for i in range(n):
        i1, i2 = np.random.choice(idx, size=2)
        output[i1], output[i2] = output[i2], output[i1]

    return output


def get_all_tour_permutations(
    N: int, start: int, max_cap: int = 1000, return_duplicates=False
):
    """
    @param: N, number of cities
    @param: start, starting city
    @param: max_cap, maximum number of tours to return, defaults to 1000.
    @param: enable_random_generation, makes tours start from other places so the
    tours generated look different from their adjacent ones
    @return: list of all possible unique tours from start to end
    """
    tours = []
    seen_vertices = set()

    def generate_permutations(current, current_tour):
        if len(current_tour) == N or len(tours) >= max_cap:
            tours.append(current_tour.copy())
            return

        seen_vertices.add(current)
        for neighbor in get_neighbors(current, N):
            if neighbor not in seen_vertices:
                current_tour.append(neighbor)
                generate_permutations(neighbor, current_tour)
                last_vertex = current_tour.pop()
                if last_vertex in seen_vertices:
                    seen_vertices.remove(last_vertex)

    generate_permutations(start, [start])

    # using a set significantly speeds up this section
    set_non_duplicate_tours = set()
    non_duplicate_tours = []
    for tour in tours:
        # e.g [0, 2, 3, 4] and [0, 4, 3, 2] are the same tour
        if tuple([tour[0]] + tour[1:][::-1]) in set_non_duplicate_tours:
            continue
        else:
            set_non_duplicate_tours.add(tuple(tour))
            non_duplicate_tours.append(tour)

    if return_duplicates:
        # return duplicates but grouped by symmetry
        duplicate_tours = []
        for tour in non_duplicate_tours:
            symm_group = [tour, [tour[0], *tour[1:][::-1]]]
            duplicate_tours.append(symm_group)

        return duplicate_tours
    else:
        return non_duplicate_tours


def get_neighbors(vertex, N):
    return list(range(0, vertex)) + list(range(vertex + 1, N))


# def get_cost_from_permutation(dist_matrix, permutation):
#     cost = 0
#     for i in range(len(permutation)):
#         u, v = i, (i + 1) % len(permutation)
#         cost += dist_matrix[u][v]
#     return cost

# i don't know exactly what your function does, but
# it definitely does not return the permutation's cost.
# i'll leave it commented, but this one is working properly now
def get_cost_from_permutation(dist_matrix, permutation):
    cost = 0
    for t in permutation:
        cost += dist_matrix[t]

    return cost


def get_exact_tsp_solution(dist_matrix):
    from python_tsp.exact import solve_tsp_dynamic_programming

    permutation, distance = solve_tsp_dynamic_programming(dist_matrix)
    return permutation, distance


def get_edges_from_tour(tour):
    """
    @param: tour -- list of vertices that are part of the tour
    @return: list of edges
    """
    edges = []
    for i in range(len(tour)):
        edges.append((tour[i], tour[(i + 1) % len(tour)]))
    return edges


def get_cost_from_edges(edges, dist_matrix):
    return sum([dist_matrix[u][v] for u, v in edges])


def get_random_points_in_frame(N):
    return [get_random_point_in_frame() for _ in range(N)]


def get_random_point_in_frame():
    x = np.random.uniform(-config.frame_x_radius + 2, config.frame_x_radius - 2)
    y = np.random.uniform(-config.frame_y_radius + 0.5, config.frame_y_radius - 0.5)
    return np.array([x, y, 0])


def get_nearest_neighbor_solution(dist_matrix, start=0):
    current = start
    seen = set([current])
    tour = [current]
    unseen_nodes = set(list(range(dist_matrix.shape[0]))).difference(seen)
    total_cost = 0
    while len(unseen_nodes) > 0:
        min_dist_vertex = min(unseen_nodes, key=lambda x: dist_matrix[current][x])
        total_cost += dist_matrix[current][min_dist_vertex]
        tour.append(min_dist_vertex)
        current = min_dist_vertex
        seen.add(current)
        unseen_nodes = set(list(range(dist_matrix.shape[0]))).difference(seen)
    # cost to go back to start
    total_cost += dist_matrix[tour[-1]][tour[0]]
    return tour, total_cost


def get_mst(dist_matrix, v_to_ignore=None):
    # A utility function to find the vertex with
    # minimum distance value, from the set of vertices
    # not yet included in shortest path tree
    vertices_to_consider = list(range(dist_matrix.shape[0]))
    if v_to_ignore is not None:
        vertices_to_consider.remove(v_to_ignore)

    def min_key(key, mst_set):
        # Initialize minim value
        minim = float("inf")
        for v in vertices_to_consider:
            if key[v] < minim and mst_set[v] == False:
                minim = key[v]
                min_index = v

        return min_index

    # Key values used to pick minimum weight edge in cut
    key = [float("inf")] * dist_matrix.shape[0]
    parent = [None] * dist_matrix.shape[0]  # Array to store constructed MST
    # Make key 0 so that this vertex is picked as first vertex
    key[vertices_to_consider[0]] = 0
    mst_set = [False] * dist_matrix.shape[0]
    parent[vertices_to_consider[0]] = -1  # First node is always the root of

    for _ in range(len(vertices_to_consider)):

        # Pick the minimum distance vertex from
        # the set of vertices not yet processed.
        # u is always equal to src in first iteration
        u = min_key(key, mst_set)

        # Put the minimum distance vertex in
        # the shortest path tree
        mst_set[u] = True

        # Update dist value of the adjacent vertices
        # of the picked vertex only if the current
        # distance is greater than new distance and
        # the vertex in not in the shortest path tree
        for v in vertices_to_consider:
            # dist_matrix[u][v] is non zero only for adjacent vertices of m
            # mst_set[v] is false for vertices not yet included in MST
            # Update the key only if dist_matrix[u][v] is smaller than key[v]
            if (
                dist_matrix[u][v] > 0
                and mst_set[v] == False
                and key[v] > dist_matrix[u][v]
            ):
                key[v] = dist_matrix[u][v]
                parent[v] = u
    cost = 0
    mst_edges = []
    for i in range(1, len(vertices_to_consider)):
        mst_edges.append((parent[vertices_to_consider[i]], vertices_to_consider[i]))
        cost += dist_matrix[parent[vertices_to_consider[i]]][i]
    return mst_edges, cost


def get_1_tree(dist_matrix, v_to_ignore):
    mst_edges, cost = get_mst(dist_matrix, v_to_ignore=v_to_ignore)
    closest_vertices = sorted(
        [v for v in range(dist_matrix.shape[0]) if v != v_to_ignore],
        key=lambda x: dist_matrix[x][v_to_ignore],
    )
    additional_edges = [
        (v_to_ignore, closest_vertices[0]),
        (v_to_ignore, closest_vertices[1]),
    ]
    one_tree_edges = mst_edges + additional_edges
    one_tree_cost = sum([dist_matrix[u][v] for u, v in one_tree_edges])
    return mst_edges, cost, one_tree_edges, one_tree_cost


def two_opt_swap(tour: list, v1: int, v2: int):
    """
    v1 and v2 are the first vertices of the edges you wish to swap when traversing through the route
    """

    v1 = tour.index(v1)
    v2 = tour.index(v2)

    if v1 > v2:
        v1, v2 = v2, v1

    v1 += 1
    v2 += 1
    reverse_tour = tour[v1:v2][::-1]
    return tour[:v1] + reverse_tour + tour[v2:]


def christofides(dist_matrix):
    mst_edges, cost = get_mst(dist_matrix)

    degree_map = get_degrees_for_all_vertices(mst_edges, dist_matrix)
    # match vertices that have odd degree
    vertices_to_match = [v for v in degree_map if degree_map[v] % 2 == 1]
    min_weight_matching = get_min_weight_perfect_matching(
        vertices_to_match, dist_matrix
    )
    tour_edges, tour_cost = get_best_hamiltonian_tour(
        dist_matrix, mst_edges, min_weight_matching
    )
    return tour_edges, tour_cost


def get_degrees_for_all_vertices(edges, dist_matrix):
    v_to_degree = {v: 0 for v in range(dist_matrix.shape[0])}
    for edge in edges:
        u, v = edge
        v_to_degree[u] = v_to_degree.get(u, 0) + 1
        v_to_degree[v] = v_to_degree.get(v, 0) + 1
    return v_to_degree


def get_min_weight_perfect_matching(vertices_to_match, dist_matrix):
    edges_with_weight = []
    for u in vertices_to_match:
        for v in vertices_to_match:
            if v > u:
                # negate weight so we get min-weight matching
                edges_with_weight.append((u, v, -dist_matrix[u][v]))
    nx_graph = nx.Graph()
    nx_graph.add_weighted_edges_from(edges_with_weight)
    return nx.max_weight_matching(nx_graph, maxcardinality=True)


def get_multigraph(mst_edges, min_weight_matching):
    print("MST edges", mst_edges)
    print("Matching edges", min_weight_matching)
    return nx.MultiGraph(incoming_graph_data=mst_edges + list(min_weight_matching))


def get_best_hamiltonian_tour(dist_matrix, mst_edges, min_weight_matching, start=0):
    best_tour = None
    best_tour_cost = float("inf")
    for start in range(dist_matrix.shape[0]):
        eulerian_tour = get_eulerian_tour(mst_edges, min_weight_matching, start=start)
        hamiltonian_tour = get_hamiltonian_tour_from_eulerian(eulerian_tour)
        tour_edges = get_edges_from_tour(hamiltonian_tour)
        tour_cost = get_cost_from_edges(tour_edges, dist_matrix)
        print(f"Tour cost with start {start}: {tour_cost}")
        if tour_cost < best_tour_cost:
            best_tour_cost = tour_cost
            best_tour = tour_edges

    return best_tour, best_tour_cost


def get_eulerian_tour(mst_edges, min_weight_matching, start=0):
    multigraph = get_multigraph(mst_edges, min_weight_matching)
    return list(nx.eulerian_circuit(multigraph, source=start))


def get_hamiltonian_tour_from_eulerian(eulerian_tour):
    visited = set()
    hamiltonian_tour = []
    for u, v in eulerian_tour:
        if u not in visited:
            hamiltonian_tour.append(u)
            visited.add(u)
        if v not in visited:
            hamiltonian_tour.append(v)
            visited.add(v)
    return hamiltonian_tour


def get_all_perfect_matchings(vertices_to_match):
    if len(vertices_to_match) == 0:
        return []
    if len(vertices_to_match) == 2:
        return [[(vertices_to_match[0], vertices_to_match[1])]]
    all_perfect_matches = []
    encountered_matches = set()
    for first_match in it.combinations(vertices_to_match, 2):
        remaining_vertcies_to_match = [
            v for v in vertices_to_match if v != first_match[0] and v != first_match[1]
        ]
        other_matches = get_all_perfect_matchings(remaining_vertcies_to_match)
        current_perfect_match = [
            [first_match] + other_match for other_match in other_matches
        ]
        for match in current_perfect_match:
            match.sort()
            if tuple(match) not in encountered_matches:
                encountered_matches.add(tuple(match))
                all_perfect_matches.append(match)
    return all_perfect_matches


def get_two_opt(tour, i, j):
    """
    @param: tour - represented as verticies
    @param: i - index of first vertex
    @param: j - index of second vertex
    """
    return tour[0 : i + 1] + tour[i + 1 : j + 1][::-1] + tour[j + 1 :]


def get_two_opt_new_edges(tour, e1, e2):
    u1, v1 = e1
    u2, v2 = e2

    if u1 in e2 or v1 in e2:
        return e1, e2, tour

    i, j = tour.index(u1), tour.index(u2)
    new_tour = get_two_opt(tour, i, j)
    new_e1 = (u1, new_tour[(i + 1) % len(tour)])
    new_e2 = (v1, new_tour[(j + 1) % len(tour)])
    return new_e1, new_e2, new_tour


def get_unvisited_neighbors(current_node, tour_so_far, N):
    return [i for i in range(N) if i != current_node and i not in set(tour_so_far)]

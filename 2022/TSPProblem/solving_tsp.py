import sys
import copy

### THIS IS A WORKAROUND FOR NOW
### IT REQUIRES RUNNING MANIM FROM INSIDE DIRECTORY VIDEO-DEV
sys.path.insert(1, "common/")

from manim import *
from manim.mobject.geometry.tips import ArrowTriangleFilledTip
from reducible_colors import *
from functions import *
import itertools
from solver_utils import *
from typing import Hashable, Iterable
from classes import *

np.random.seed(23)

config["assets_dir"] = "assets"

BACKGROUND_IMG = ImageMobject("bg-75.png").scale_to_fit_width(config.frame_width)


class TSPGraph(Graph):
    def __init__(
        self,
        vertices,
        dist_matrix=None,
        vertex_config={
            "stroke_color": REDUCIBLE_PURPLE,
            "stroke_width": 3,
            "fill_color": REDUCIBLE_PURPLE_DARK_FILL,
            "fill_opacity": 1,
        },
        edge_config={
            "color": REDUCIBLE_VIOLET,
            "stroke_width": 3,
        },
        labels=True,
        label_scale=0.6,
        label_color=WHITE,
        **kwargs,
    ):
        edges = []
        if labels:
            labels = {
                k: CustomLabel(str(k), scale=label_scale).set_color(label_color)
                for k in vertices
            }
            edge_config["buff"] = LabeledDot(list(labels.values())[0]).radius
            self.labels = labels
        else:
            edge_config["buff"] = Dot().radius
            self.labels = None
        ### Manim bug where buff has no effect for some reason on standard lines
        super().__init__(
            vertices,
            edges,
            vertex_config=vertex_config,
            edge_config=edge_config,
            labels=labels,
            **kwargs,
        )
        ### RED-103: Vertices should be the same size (even for larger labels)
        for v in self.vertices:
            if v >= 10:
                self.vertices[v].scale_to_fit_height(self.vertices[0].height)
        self.edge_config = edge_config
        if dist_matrix is None:
            self.dist_matrix = np.zeros((len(vertices), len(vertices)))
            for u, v in itertools.combinations(vertices, 2):
                distance = np.linalg.norm(
                    self.vertices[u].get_center() - self.vertices[v].get_center()
                )
                self.dist_matrix[u][v] = distance
                self.dist_matrix[v][u] = distance
        else:
            self.dist_matrix = dist_matrix

    def get_all_edges(self, edge_type: TipableVMobject = Line, buff=None):
        edge_dict = {}
        for edge in itertools.combinations(self.vertices.keys(), 2):
            u, v = edge
            edge_dict[edge] = self.create_edge(u, v, edge_type=edge_type, buff=buff)
        return edge_dict

    def get_some_edges(
        self, percentage=0.7, edge_type: TipableVMobject = Line, buff=None
    ):
        """
        Given a TSPGraph, generate a subset of all possible sets. Use percentage to control
        the total amount from edges to return from the total. 0.7 will give 70% of the total edge count.
        This is useful for insanely big graphs, where presenting only 30% of the total still gives the illusion
        of scale but we don't have to calculate billions of edges.
        """
        edge_dict = {}
        vertex_list = list(self.vertices.keys())

        random_tuples = [
            (u, v)
            for u in vertex_list
            for v in sorted(
                np.random.choice(vertex_list, int(len(vertex_list) * percentage))
            )
            if v != u
        ]

        for t in random_tuples:
            edge_dict[t] = self.create_edge(t[0], t[1], edge_type=edge_type, buff=buff)

        return edge_dict

    def create_edge(self, u, v, edge_type: TipableVMobject = Line, buff=None):
        return edge_type(
            self.vertices[u].get_center(),
            self.vertices[v].get_center(),
            color=self.edge_config["color"],
            stroke_width=self.edge_config["stroke_width"],
            buff=self.edge_config["buff"] if buff is None else buff,
        )

    def get_tour_edges(self, tour, edge_type: TipableVMobject = Line):
        """
        @param: tour -- sequence of vertices where all vertices are part of the tour (no repetitions)
        """
        edges = get_edges_from_tour(tour)
        edge_dict = {}
        for edge in edges:
            u, v = edge
            edge_mob = self.create_edge(u, v, edge_type=edge_type)
            edge_dict[edge] = edge_mob
        return edge_dict

    def get_tour_dist_labels(self, edge_dict, scale=0.3, num_decimal_places=1):
        dist_label_dict = {}
        for edge in edge_dict:
            u, v = edge
            dist_label = self.get_dist_label(
                edge_dict[edge],
                self.dist_matrix[u][v],
                scale=scale,
                num_decimal_places=num_decimal_places,
            )
            dist_label_dict[edge] = dist_label
        return dist_label_dict

    def get_dist_label(self, edge_mob, distance, scale=0.3, num_decimal_places=1):
        return (
            Text(str(np.round(distance, num_decimal_places)), font=REDUCIBLE_MONO)
            .set_stroke(BLACK, width=8, background=True, opacity=0.8)
            .scale(scale)
            .move_to(edge_mob.point_from_proportion(0.5))
        )

    def get_dist_matrix(self):
        return self.dist_matrix

    def get_neighboring_edges(self, vertex, buff=None):
        edges = [(vertex, other) for other in get_neighbors(vertex, len(self.vertices))]
        return {edge: self.create_edge(edge[0], edge[1], buff=buff) for edge in edges}

    def get_edges_from_list(self, edges):
        edge_dict = {}
        for edge in edges:
            u, v = edge
            edge_mob = self.create_edge(u, v)
            edge_dict[edge] = edge_mob
        return edge_dict


class TSPTester(Scene):
    def construct(self):
        big_graph = TSPGraph(range(12), layout_scale=2.4, layout="circular")
        all_edges_bg = big_graph.get_all_edges()
        self.play(FadeIn(big_graph))
        self.wait()

        self.play(*[FadeIn(edge) for edge in all_edges_bg.values()])
        self.wait()


class NearestNeighbor2(Scene):
    def construct(self):
        # self.add(BACKGROUND_IMG)
        # NUM_VERTICES = 10
        # layout = self.get_random_layout(NUM_VERTICES)
        # # MANUAL ADJUSTMENTS FOR BETTER INSTRUCTIONAL EXAMPLE
        # layout[7] = RIGHT * 3.5 + UP * 2
        # np.random.seed(5)
        # self.add(BACKGROUND_IMG)
        # NUM_VERTICES = 10
        # layout = self.get_random_layout(NUM_VERTICES)
        # # MANUAL ADJUSTMENTS FOR BETTER INSTRUCTIONAL EXAMPLE
        # layout[1] = RIGHT * 3.5 + UP * 2

        # graph = TSPGraph(
        #     list(range(NUM_VERTICES)),
        #     layout=layout,
        # )
        # self.play(FadeIn(graph))
        # self.wait()

        # graph_with_tour_edges = self.demo_nearest_neighbor(graph)

        # self.compare_nn_with_optimal(graph_with_tour_edges, graph)

        # self.clear()
        # self.add(BACKGROUND_IMG)

        self.show_many_large_graph_nn_solutions()

    def demo_nearest_neighbor(self, graph):
        glowing_circle = get_glowing_surround_circle(graph.vertices[0])
        self.play(FadeIn(glowing_circle))
        self.wait()

        neighboring_edges = graph.get_neighboring_edges(0)

        self.play(
            LaggedStartMap(Write, [(edge) for edge in neighboring_edges.values()])
        )
        self.wait()

        tour, cost = get_nearest_neighbor_solution(graph.get_dist_matrix())
        tour_edges = graph.get_tour_edges(tour)
        seen = set([tour[0]])
        prev = tour[0]
        residual_edges = {}
        for vertex in tour[1:]:
            self.play(
                tour_edges[(prev, vertex)].animate.set_color(REDUCIBLE_YELLOW),
                ShowPassingFlash(
                    tour_edges[(prev, vertex)]
                    .copy()
                    .set_stroke(width=6)
                    .set_color(REDUCIBLE_YELLOW),
                ),
            )
            seen.add(vertex)
            new_glowing_circle = get_glowing_surround_circle(graph.vertices[vertex])
            new_neighboring_edges = graph.get_neighboring_edges(vertex)
            for key in new_neighboring_edges.copy():
                if key[1] in seen and key[1] != vertex:
                    del new_neighboring_edges[key]
            filtered_prev_edges = [
                edge_key
                for edge_key, edge in neighboring_edges.items()
                if edge_key != (prev, vertex) and edge_key != (vertex, prev)
            ]
            self.play(
                FadeOut(glowing_circle),
                FadeIn(new_glowing_circle),
                *[
                    FadeOut(neighboring_edges[edge_key])
                    for edge_key in filtered_prev_edges
                ],
            )
            filtered_new_edges = [
                edge_key
                for edge_key, edge in new_neighboring_edges.items()
                if edge_key != (prev, vertex) and edge_key != (vertex, prev)
            ]

            if len(filtered_new_edges) > 0:
                self.play(
                    *[
                        FadeIn(new_neighboring_edges[edge_key])
                        for edge_key in filtered_new_edges
                    ]
                )
            residual_edges[(prev, vertex)] = neighboring_edges[(prev, vertex)]
            neighboring_edges = new_neighboring_edges
            glowing_circle = new_glowing_circle
            prev = vertex

        for edge in residual_edges.values():
            self.remove(edge)
        # final edge connecting back to start
        tour_edges[(tour[-1], tour[0])].set_color(REDUCIBLE_YELLOW)
        self.play(
            Write(tour_edges[(tour[-1], tour[0])]),
            FadeOut(glowing_circle),
        )
        self.wait()

        graph_with_tour_edges = self.get_graph_tour_group(graph, tour_edges)
        return graph_with_tour_edges

    def compare_nn_with_optimal(self, graph_with_tour_edges, original_graph):
        nn_tour, nn_cost = get_nearest_neighbor_solution(
            original_graph.get_dist_matrix()
        )
        optimal_tour, optimal_cost = get_exact_tsp_solution(
            original_graph.get_dist_matrix()
        )
        optimal_graph = original_graph.copy()
        optimal_edges = optimal_graph.get_tour_edges(optimal_tour)

        shift_amount = 3.2
        scale = 0.6
        self.play(graph_with_tour_edges.animate.scale(scale).shift(LEFT * shift_amount))
        self.wait()

        optimal_graph_tour = self.get_graph_tour_group(optimal_graph, optimal_edges)
        optimal_graph_tour.scale(scale).shift(RIGHT * shift_amount)
        nn_text = self.get_distance_text(nn_cost).next_to(
            graph_with_tour_edges, UP, buff=1
        )
        optimal_text = self.get_distance_text(optimal_cost).next_to(
            optimal_graph_tour, UP, buff=1
        )

        self.play(FadeIn(nn_text))

        self.play(FadeIn(optimal_graph_tour))
        self.wait()

        self.play(
            FadeIn(optimal_text),
        )
        self.wait()

        nn_heuristic = Text(
            "Nearest Neighbor (NN) Heuristic", font=REDUCIBLE_FONT, weight=BOLD
        )
        nn_heuristic.scale(0.8)
        nn_heuristic.move_to(DOWN * 2.5)

        self.play(Write(nn_heuristic))
        self.wait()

        surround_rect_0_2_3_5_original = SurroundingRectangle(
            VGroup(
                *[
                    original_graph.vertices[0],
                    original_graph.vertices[2],
                    original_graph.vertices[3],
                    original_graph.vertices[5],
                ]
            )
        ).set_color(REDUCIBLE_CHARM)

        surround_rect_0_2_3_5_optimal = SurroundingRectangle(
            VGroup(
                *[
                    optimal_graph.vertices[0],
                    optimal_graph.vertices[2],
                    optimal_graph.vertices[3],
                    optimal_graph.vertices[5],
                ]
            )
        ).set_color(REDUCIBLE_GREEN_LIGHTER)

        surround_rect_1_4_6_8_original = SurroundingRectangle(
            VGroup(
                *[
                    original_graph.vertices[1],
                    original_graph.vertices[4],
                    original_graph.vertices[6],
                    original_graph.vertices[8],
                ]
            )
        ).set_color(REDUCIBLE_CHARM)

        surround_rect_1_4_6_8_optimal = SurroundingRectangle(
            VGroup(
                *[
                    optimal_graph.vertices[1],
                    optimal_graph.vertices[4],
                    optimal_graph.vertices[6],
                    optimal_graph.vertices[8],
                ]
            )
        ).set_color(REDUCIBLE_GREEN_LIGHTER)

        self.play(
            Write(surround_rect_0_2_3_5_optimal),
            Write(surround_rect_0_2_3_5_original),
        )
        self.wait()

        self.play(
            Write(surround_rect_1_4_6_8_optimal),
            Write(surround_rect_1_4_6_8_original),
        )
        self.wait()

        self.play(
            FadeOut(surround_rect_0_2_3_5_optimal),
            FadeOut(surround_rect_0_2_3_5_original),
            FadeOut(surround_rect_1_4_6_8_optimal),
            FadeOut(surround_rect_1_4_6_8_original),
        )
        self.wait()

        how_to_compare = Text(
            "How to measure effectiveness of heuristic approach?", font=REDUCIBLE_FONT
        ).scale(0.6)

        how_to_compare.next_to(nn_heuristic, DOWN)

        self.play(FadeIn(how_to_compare))
        self.wait()

        self.play(FadeOut(nn_heuristic), FadeOut(how_to_compare))

        approx_ratio = (
            Tex(
                r"Approximation ratio $(\alpha) = \frac{\text{heuristic solution}}{\text{optimal solution}}$"
            )
            .scale(0.8)
            .move_to(DOWN * 2.5)
        )

        self.play(FadeIn(approx_ratio))

        self.wait()

        example = Tex(
            r"E.g $\alpha = \frac{28.2}{27.0} \approx 1.044$",
            r"$\rightarrow$ 4.4\% above optimal",
        ).scale(0.7)

        example.next_to(approx_ratio, DOWN)

        self.play(Write(example[0]))
        self.wait()

        self.play(Write(example[1]))
        self.wait()

    def show_many_large_graph_nn_solutions(self):
        NUM_VERTICES = 100
        num_iterations = 1
        average_case = (
            Tex(
                r"On average: $\frac{\text{NN Heuristic}}{\text{1-Tree Lower Bound}} = 1.25$"
            )
            .scale(0.8)
            .move_to(DOWN * 3.5)
        )
        for _ in range(num_iterations):
            graph = TSPGraph(
                list(range(NUM_VERTICES)),
                labels=False,
                layout=self.get_random_layout(NUM_VERTICES),
            )
            tour, nn_cost = get_nearest_neighbor_solution(graph.get_dist_matrix())
            print("NN cost", nn_cost)
            tour_edges = graph.get_tour_edges(tour)
            tour_edges_group = VGroup(*list(tour_edges.values()))
            graph_with_tour_edges = VGroup(graph, tour_edges_group).scale(0.8)
            self.add(graph)
            self.play(LaggedStartMap(Write, tour_edges_group), run_time=14)
            self.wait()

    def get_distance_text(self, cost, num_decimal_places=1, scale=0.6):
        cost = np.round(cost, num_decimal_places)
        return Text(f"Distance = {cost}", font=REDUCIBLE_MONO).scale(scale)

    def get_random_layout(self, N):
        random_points_in_frame = get_random_points_in_frame(N)
        return {v: point for v, point in zip(range(N), random_points_in_frame)}

    def get_graph_tour_group(self, graph, tour_edges):
        return VGroup(*[graph] + list(tour_edges.values()))

    def label_vertices_for_debugging(self, graph):
        labels = VGroup()
        for v, v_mob in graph.vertices.items():
            label = (
                Text(str(v), font=REDUCIBLE_MONO).scale(0.2).move_to(v_mob.get_center())
            )
            labels.add(label)

        return labels


class LowerBoundTSP(NearestNeighbor2):
    def construct(self):
        self.add(BACKGROUND_IMG)
        graph = self.get_graph_with_random_layout(200, radius=0.05)
        tour, nn_cost = get_nearest_neighbor_solution(graph.get_dist_matrix())
        edge_ordering = get_edges_from_tour(tour)
        tour_edges = graph.get_tour_edges(tour)

        self.scale_graph_with_tour(graph, tour_edges, 0.8)

        self.play(
            LaggedStartMap(GrowFromCenter, list(graph.vertices.values())), run_time=2
        )
        self.wait()

        self.play(
            LaggedStartMap(Write, [tour_edges[edge] for edge in edge_ordering]),
            run_time=10,
        )
        self.wait()

        problem = (
            Text(
                "Given any solution, no efficient way to verify optimality!",
                font=REDUCIBLE_FONT,
            )
            .scale(0.5)
            .move_to(DOWN * 3.5)
        )

        self.play(FadeIn(problem))
        self.wait()

        self.clear()
        self.add(BACKGROUND_IMG)

        self.present_lower_bound_idea()

        self.clear()
        self.add(BACKGROUND_IMG)

        tsp_graph, mst_tree, mst_edge_dict = self.intro_mst()

        self.intro_1_tree(tsp_graph, mst_tree, mst_edge_dict)

    def present_lower_bound_idea(self):
        heuristic_solution_mod = Module(["Heuristic", "Solution"], text_weight=BOLD)

        optimal_solution_mod = Module(
            ["Optimal", "Solution"],
            REDUCIBLE_GREEN_DARKER,
            REDUCIBLE_GREEN_LIGHTER,
            text_weight=BOLD,
        )

        lower_bound_mod = Module(
            ["Lower", "Bound"],
            REDUCIBLE_YELLOW_DARKER,
            REDUCIBLE_YELLOW,
            text_weight=BOLD,
        )
        lower_bound_mod.text.scale(0.9)
        left_geq = MathTex(r"\geq").scale(2)
        VGroup(heuristic_solution_mod, left_geq, optimal_solution_mod).arrange(
            RIGHT, buff=1
        )

        self.play(
            FadeIn(heuristic_solution_mod),
            FadeIn(optimal_solution_mod),
            FadeIn(left_geq),
        )
        self.wait()

        right_geq = MathTex(r"\geq").scale(2)
        new_configuration = (
            VGroup(
                heuristic_solution_mod.copy(),
                left_geq.copy(),
                optimal_solution_mod.copy(),
                right_geq,
                lower_bound_mod,
            )
            .arrange(RIGHT, buff=1)
            .scale(0.7)
        )

        self.play(
            Transform(heuristic_solution_mod, new_configuration[0]),
            Transform(left_geq, new_configuration[1]),
            Transform(optimal_solution_mod, new_configuration[2]),
        )

        self.play(FadeIn(right_geq), FadeIn(lower_bound_mod))
        self.wait()

        curved_arrow_1 = (
            CustomCurvedArrow(
                heuristic_solution_mod.get_top(),
                optimal_solution_mod.get_top(),
                angle=-PI / 4,
            )
            .shift(UP * MED_SMALL_BUFF)
            .set_color(GRAY)
        )

        curved_arrow_2 = (
            CustomCurvedArrow(
                heuristic_solution_mod.get_bottom(),
                lower_bound_mod.get_bottom(),
                angle=PI / 4,
            )
            .shift(DOWN * MED_SMALL_BUFF)
            .set_color(GRAY)
        )

        inefficient_comparison = (
            Text("Intractable comparison", font=REDUCIBLE_FONT)
            .scale(0.6)
            .next_to(curved_arrow_1, UP)
        )
        reasonable_comparison = (
            Text("Reasonable comparison", font=REDUCIBLE_FONT)
            .scale(0.6)
            .next_to(curved_arrow_2, DOWN)
        )
        self.play(FadeIn(curved_arrow_1), FadeIn(inefficient_comparison))
        self.wait()

        self.play(FadeIn(curved_arrow_2), FadeIn(reasonable_comparison))
        self.wait()

        good_lower_bound = (
            Tex(
                r"Good lower bound: maximize $\frac{\text{lower bound}}{\text{optimal}}$"
            )
            .scale(0.8)
            .move_to(UP * 3)
        )

        self.play(FadeIn(good_lower_bound))
        self.wait()

    def intro_mst(self):
        title = (
            Text("Minimum Spanning Tree (MST)", font=REDUCIBLE_FONT, weight=BOLD)
            .scale(0.8)
            .move_to(UP * 3.5)
        )
        NUM_VERTICES = 9
        graph = TSPGraph(
            list(range(NUM_VERTICES)), layout=self.get_random_layout(NUM_VERTICES)
        )
        original_scaled_graph = graph.copy()
        not_connected_graph = graph.copy()
        cycle_graph = graph.copy()
        non_mst_graph = graph.copy()

        mst_edges, cost = get_mst(graph.get_dist_matrix())
        mst_edges_mob = graph.get_edges_from_list(mst_edges)
        mst_edges_group = VGroup(*list(mst_edges_mob.values()))
        mst_tree = VGroup(graph, mst_edges_group)
        mst_tree.scale(0.8)
        self.play(Write(title), FadeIn(graph))
        self.wait()
        self.play(*[GrowFromCenter(edge) for edge in mst_edges_group])
        self.wait()

        definition = (
            Text(
                "Set of edges that connect all vertices with minimum distance and no cycles",
                font=REDUCIBLE_FONT,
            )
            .scale(0.5)
            .move_to(DOWN * 3.5)
        )
        definition[14:21].set_color(REDUCIBLE_YELLOW)
        definition[36:51].set_color(REDUCIBLE_YELLOW)
        definition[-8:].set_color(REDUCIBLE_YELLOW)
        self.play(FadeIn(definition))
        self.wait()

        true_mst_text = Text("True MST", font=REDUCIBLE_FONT).scale(0.7)
        self.play(mst_tree.animate.scale(0.75).shift(LEFT * 3.5))
        true_mst_text.next_to(mst_tree, DOWN)
        self.play(FadeIn(true_mst_text))
        self.wait()

        to_remove_edge = (8, 6)
        mst_edges.remove(to_remove_edge)
        not_connected_edge = not_connected_graph.get_edges_from_list(mst_edges)
        not_connect_graph_group = (
            VGroup(*[not_connected_graph] + list(not_connected_edge.values()))
            .scale(0.6)
            .shift(RIGHT * 3.5)
        )

        not_connected_text = (
            Text("Not connected", font=REDUCIBLE_FONT)
            .scale(0.7)
            .next_to(not_connect_graph_group, DOWN)
        )
        self.play(FadeIn(not_connect_graph_group), FadeIn(not_connected_text))

        surround_rect = SurroundingRectangle(
            VGroup(not_connected_graph.vertices[8], not_connected_graph.vertices[6]),
            color=REDUCIBLE_CHARM,
        )
        self.play(
            Write(surround_rect),
        )
        self.wait()

        to_add_edge = (6, 2)
        prev_removed_edge = not_connected_graph.create_edge(
            to_remove_edge[0],
            to_remove_edge[1],
            buff=graph.vertices[0].width / 2 + SMALL_BUFF / 4,
        )
        new_edge = not_connected_graph.create_edge(
            to_add_edge[0],
            to_add_edge[1],
            buff=graph.vertices[0].width / 2 + SMALL_BUFF / 4,
        )

        cyclic_text = (
            Text("Has cycle", font=REDUCIBLE_FONT)
            .scale(0.7)
            .move_to(not_connected_text.get_center())
        )
        self.play(
            FadeOut(surround_rect),
            Write(prev_removed_edge),
            Write(new_edge),
            ReplacementTransform(not_connected_text, cyclic_text),
        )
        new_surround_rect = SurroundingRectangle(
            VGroup(
                not_connected_graph.vertices[8],
                not_connected_graph.vertices[6],
                not_connected_graph.vertices[2],
                not_connected_graph.vertices[0],
            ),
            color=REDUCIBLE_CHARM,
        )
        self.play(Write(new_surround_rect))
        self.wait()

        non_optimal_edge = not_connected_graph.create_edge(
            5, 7, buff=graph.vertices[0].width / 2 + SMALL_BUFF / 4
        )
        non_optimal_edge.set_color(REDUCIBLE_CHARM)
        non_optimal_text = (
            Text("Spanning tree, but not minimum", font=REDUCIBLE_FONT)
            .scale(0.6)
            .move_to(cyclic_text.get_center())
        )
        self.play(
            FadeOut(new_surround_rect),
            FadeOut(new_edge),
            FadeOut(not_connected_edge[(5, 1)]),
            Write(non_optimal_edge),
            ReplacementTransform(cyclic_text, non_optimal_text),
        )
        self.wait()

        self.clear()
        self.add(BACKGROUND_IMG)

        mst_tree, mst_edge_dict = self.demo_prims_algorithm(
            original_scaled_graph.copy()
        )

        return original_scaled_graph, mst_tree, mst_edge_dict

    def demo_prims_algorithm(self, graph):
        visited = set([0])
        unvisited = set(graph.vertices.keys()).difference(visited)
        all_edges = graph.get_all_edges()
        VGroup(graph, VGroup(*list(all_edges.values()))).scale(0.8).shift(UP * 0.5)
        self.play(
            FadeIn(graph),
        )
        self.wait()

        (
            visited_group,
            unvisited_group,
            visited_dict,
            unvisited_dict,
        ) = self.highlight_visited_univisited(
            graph.vertices, graph.labels, visited, unvisited
        )
        visited_label = (
            Text("Visited", font=REDUCIBLE_FONT).scale(0.5).next_to(visited_group, UP)
        )
        unvisited_label = (
            Text("Unvisited", font=REDUCIBLE_FONT)
            .scale(0.5)
            .next_to(unvisited_group, UP)
        )
        self.play(
            FadeIn(visited_group),
            FadeIn(unvisited_group),
            FadeIn(visited_label),
            FadeIn(unvisited_label),
        )
        self.wait()
        iteration = 0
        highlight_animations = []
        for v in graph.vertices:
            if v in visited:
                highlighted_v = graph.vertices[v].copy()
                highlighted_v[0].set_fill(opacity=0.5).set_stroke(opacity=1)
                highlighted_v[1].set_fill(opacity=1)
                highlight_animations.append(Transform(graph.vertices[v], highlighted_v))
            else:
                un_highlighted_v = graph.vertices[v].copy()
                un_highlighted_v[0].set_fill(opacity=0.2).set_stroke(opacity=0.2)
                un_highlighted_v[1].set_fill(opacity=0.2)
                highlight_animations.append(
                    Transform(graph.vertices[v], un_highlighted_v)
                )
        self.play(*highlight_animations)
        self.wait()
        mst_edges = VGroup()
        mst_edge_dict = {}
        while len(unvisited) > 0:
            neighboring_edges = self.get_neighboring_edges_across_sets(
                visited, unvisited
            )
            for i, edge in enumerate(neighboring_edges):
                if edge not in all_edges:
                    neighboring_edges[i] = (edge[1], edge[0])
            neighboring_edges_mobs = [
                all_edges[edge].set_stroke(opacity=0.3) for edge in neighboring_edges
            ]
            self.play(*[Write(edge) for edge in neighboring_edges_mobs])
            self.wait()
            best_neighbor_edge = min(
                neighboring_edges, key=lambda x: graph.get_dist_matrix()[x[0]][x[1]]
            )
            next_vertex = (
                best_neighbor_edge[1]
                if best_neighbor_edge[1] not in visited
                else best_neighbor_edge[0]
            )
            print("Best neighbor", best_neighbor_edge)
            print("Next vertex", next_vertex)
            self.play(
                ShowPassingFlash(
                    all_edges[best_neighbor_edge]
                    .copy()
                    .set_stroke(width=6)
                    .set_color(REDUCIBLE_YELLOW),
                    time_width=0.5,
                ),
            )
            self.play(
                all_edges[best_neighbor_edge].animate.set_stroke(
                    opacity=1, color=REDUCIBLE_YELLOW
                )
            )
            mst_edges.add(all_edges[best_neighbor_edge])
            mst_edge_dict[best_neighbor_edge] = all_edges[best_neighbor_edge]
            self.wait()

            visited.add(next_vertex)
            unvisited.remove(next_vertex)

            (
                _,
                _,
                new_visited_dict,
                new_unvisited_dict,
            ) = self.highlight_visited_univisited(
                graph.vertices, graph.labels, visited, unvisited
            )
            print(type(graph.vertices[next_vertex][1]))
            highlight_next_vertex = graph.vertices[next_vertex].copy()
            highlight_next_vertex[0].set_fill(opacity=0.5).set_stroke(opacity=1)
            highlight_next_vertex[1].set_fill(opacity=1)
            self.play(
                FadeOut(
                    *[
                        all_edges[edge]
                        for edge in neighboring_edges
                        if edge != best_neighbor_edge
                    ]
                ),
                Transform(graph.vertices[next_vertex], highlight_next_vertex),
                *[
                    Transform(visited_dict[v], new_visited_dict[v])
                    for v in visited.difference(set([next_vertex]))
                ],
                *[
                    Transform(unvisited_dict[v], new_unvisited_dict[v])
                    for v in unvisited
                ],
                ReplacementTransform(
                    unvisited_dict[next_vertex], new_visited_dict[next_vertex]
                ),
            )
            self.wait()
            visited_dict[next_vertex] = new_visited_dict[next_vertex]
            del unvisited_dict[next_vertex]

        self.play(
            FadeOut(visited_label),
            FadeOut(unvisited_label),
            *[FadeOut(mob) for mob in visited_dict.values()],
        )
        self.wait()

        mst_tree = VGroup(graph, mst_edges)
        return mst_tree, mst_edge_dict

    def intro_1_tree(self, tsp_graph, mst_tree, mst_edge_dict):
        optimal_tour, optimal_cost = get_exact_tsp_solution(tsp_graph.get_dist_matrix())
        tsp_tour_edges = tsp_graph.get_tour_edges(optimal_tour)
        tsp_tour_edges_group = VGroup(*[edge for edge in tsp_tour_edges.values()])
        tsp_graph_with_tour = VGroup(tsp_graph, tsp_tour_edges_group)
        self.play(mst_tree.animate.scale(0.75).move_to(LEFT * 3.5 + UP * 1))
        self.wait()
        tsp_graph_with_tour.scale_to_fit_height(mst_tree.height).move_to(
            RIGHT * 3.5 + UP * 1
        )

        self.play(FadeIn(tsp_graph_with_tour))
        self.wait()

        mst_cost = Tex(r"MST Cost $<$ TSP Cost").move_to(DOWN * 2)
        self.play(FadeIn(mst_cost))
        self.wait()

        remove_edge = (
            Tex(r"Remove any edge from TSP tour $\rightarrow$ spanning tree $T$")
            .scale(0.7)
            .next_to(mst_cost, DOWN)
        )
        self.play(FadeIn(remove_edge))
        self.wait()
        result = Tex(r"MST cost $\leq$ cost($T$)").scale(0.7)
        result.next_to(remove_edge, DOWN)
        prev_edge = None
        for i, edge in enumerate(tsp_tour_edges):
            if i == 0:
                self.play(FadeOut(tsp_tour_edges[edge]))
            else:
                self.play(
                    FadeIn(tsp_tour_edges[prev_edge]), FadeOut(tsp_tour_edges[edge])
                )
            prev_edge = edge
            self.wait()

        self.play(FadeIn(result))
        self.wait()

        better_lower_bound = (
            Text("Better Lower Bound", font=REDUCIBLE_FONT, weight=BOLD)
            .scale_to_fit_height(mst_cost.height - SMALL_BUFF)
            .move_to(mst_cost.get_center())
            .shift(UP * SMALL_BUFF)
        )
        mst_vertices, mst_edges = mst_tree
        self.play(
            FadeIn(tsp_tour_edges[prev_edge]),
            FadeOut(result),
            FadeOut(remove_edge),
            FadeOut(mst_edges),
            FadeTransform(mst_cost, better_lower_bound),
        )
        self.wait()

        step_1 = Tex(r"1. Remove any vertex $v$ and find MST").scale(0.6)
        step_2 = Tex(r"2. Connect two shortest edges to $v$").scale(0.6)
        steps = VGroup(step_1, step_2).arrange(DOWN, aligned_edge=LEFT)
        steps.next_to(better_lower_bound, DOWN)
        self.play(FadeIn(step_1))
        self.wait()

        self.play(FadeOut(mst_vertices.vertices[6]))
        self.wait()

        mst_tree_edges_removed, cost, one_tree_edges, one_tree_cost = get_1_tree(
            mst_vertices.get_dist_matrix(), 6
        )
        all_edges = mst_vertices.get_all_edges(buff=mst_vertices[0].width / 2)
        self.play(
            *[
                GrowFromCenter(
                    self.get_edge(all_edges, edge).set_color(REDUCIBLE_YELLOW)
                )
                for edge in mst_tree_edges_removed
            ]
        )
        self.wait()

        self.play(FadeIn(step_2))
        self.wait()

        self.play(FadeIn(mst_vertices.vertices[6]))
        self.wait()

        self.play(
            *[
                GrowFromCenter(
                    self.get_edge(all_edges, edge).set_color(REDUCIBLE_YELLOW)
                )
                for edge in one_tree_edges
                if edge not in mst_tree_edges_removed
            ]
        )
        self.wait()

        new_result = Tex(r"1-tree cost $\leq$ TSP cost").scale(0.7)
        new_result.next_to(steps, DOWN)
        new_result[0][:6].set_color(REDUCIBLE_YELLOW)
        self.play(FadeIn(new_result))
        self.wait()

        unhiglighted_nodes = {
            v: tsp_graph.vertices[v].copy() for v in tsp_graph.vertices if v != 6
        }
        highlighted_nodes = copy.deepcopy(unhiglighted_nodes)
        for node in unhiglighted_nodes.values():
            node[0].set_fill(opacity=0.2).set_stroke(opacity=0.2)
            node[1].set_fill(opacity=0.2)

        unhiglighted_nodes_mst = {
            v: mst_vertices.vertices[v].copy() for v in mst_vertices.vertices if v != 6
        }
        highlighted_nodes_mst = copy.deepcopy(unhiglighted_nodes_mst)
        for node in unhiglighted_nodes_mst.values():
            node[0].set_fill(opacity=0.2).set_stroke(opacity=0.2)
            node[1].set_fill(opacity=0.2)

        self.play(
            *[
                Transform(tsp_graph.vertices[v], unhiglighted_nodes[v])
                for v in tsp_graph.vertices
                if v != 6
            ],
            *[
                tsp_tour_edges[edge].animate.set_stroke(opacity=0.2)
                for edge in tsp_tour_edges
                if 6 not in edge
            ],
            *[
                Transform(mst_vertices.vertices[v], unhiglighted_nodes_mst[v])
                for v in mst_vertices.vertices
                if v != 6
            ],
            *[
                self.get_edge(all_edges, edge).animate.set_stroke(opacity=0.2)
                for edge in one_tree_edges
                if 6 not in edge
            ],
        )

        self.wait()

        self.play(
            ShowPassingFlash(
                SurroundingRectangle(
                    VGroup(
                        *[
                            mst_vertices.vertices[v]
                            for v in mst_vertices.vertices
                            if v in [6, 8, 0]
                        ]
                    )
                ),
                time_width=0.5,
            )
        )
        self.wait()
        node_6_faded = mst_vertices.vertices[6].copy()
        original_node_6 = mst_vertices.vertices[6].copy()
        node_6_faded[0].set_fill(opacity=0.2).set_stroke(opacity=0.2)
        node_6_faded[1].set_fill(opacity=0.2)

        node_6_faded_tsp = tsp_graph.vertices[6].copy()
        original_node_6_tsp = tsp_graph.vertices[6].copy()
        node_6_faded_tsp[0].set_fill(opacity=0.2).set_stroke(opacity=0.2)
        node_6_faded_tsp[1].set_fill(opacity=0.2)
        self.play(
            *[
                Transform(tsp_graph.vertices[v], highlighted_nodes[v])
                for v in tsp_graph.vertices
                if v != 6
            ],
            *[
                tsp_tour_edges[edge].animate.set_stroke(opacity=1)
                for edge in tsp_tour_edges
                if 6 not in edge
            ],
            *[
                Transform(mst_vertices.vertices[v], highlighted_nodes_mst[v])
                for v in mst_vertices.vertices
                if v != 6
            ],
            *[
                self.get_edge(all_edges, edge).animate.set_stroke(opacity=1)
                for edge in one_tree_edges
                if 6 not in edge
            ],
            Transform(mst_vertices.vertices[6], node_6_faded),
            Transform(tsp_graph.vertices[6], node_6_faded_tsp),
            *[
                tsp_tour_edges[edge].animate.set_stroke(opacity=0.2)
                for edge in tsp_tour_edges
                if 6 in edge
            ],
            *[
                self.get_edge(all_edges, edge).animate.set_stroke(opacity=0.2)
                for edge in one_tree_edges
                if 6 in edge
            ],
        )
        self.wait()

        self.play(
            Transform(mst_vertices.vertices[6], original_node_6),
            Transform(tsp_graph.vertices[6], original_node_6_tsp),
            *[
                tsp_tour_edges[edge].animate.set_stroke(opacity=1)
                for edge in tsp_tour_edges
                if 6 in edge
            ],
            *[
                self.get_edge(all_edges, edge).animate.set_stroke(opacity=1)
                for edge in one_tree_edges
                if 6 in edge
            ],
        )
        self.wait()
        best_one_cost = one_tree_cost
        best_one_tree = 6
        current_one_tree_edges = VGroup(
            *[self.get_edge(all_edges, edge) for edge in one_tree_edges]
        )
        for v_to_ignore in [0, 1, 2, 3, 4, 7, 8, 5]:
            _, _, one_tree_edges, one_tree_cost = get_1_tree(
                mst_vertices.get_dist_matrix(), v_to_ignore
            )
            if one_tree_cost > best_one_cost:
                best_one_tree, best_one_cost = v_to_ignore, one_tree_cost
            all_edges_new = mst_vertices.get_all_edges(
                buff=mst_vertices.vertices[0].width / 2
            )
            new_one_tree_edges = VGroup(
                *[
                    self.get_edge(all_edges_new, edge).set_color(REDUCIBLE_YELLOW)
                    for edge in one_tree_edges
                ]
            )

            self.play(Transform(current_one_tree_edges, new_one_tree_edges))
            self.wait()

        print(
            "Best one tree", best_one_tree, best_one_cost, "Optimal TSP", optimal_cost
        )
        best_cost_1_tree = Text(
            f"Largest 1-tree cost: {np.round(best_one_cost, 1)}", font=REDUCIBLE_MONO
        ).scale(0.5)
        optimal_tsp_cost = Text(
            f"Optimal TSP cost: {np.round(optimal_cost, 1)}", font=REDUCIBLE_MONO
        ).scale(0.5)

        best_cost_1_tree.next_to(mst_tree, DOWN, buff=1)
        optimal_tsp_cost.next_to(tsp_graph, DOWN, buff=1)

        self.play(
            FadeIn(best_cost_1_tree),
            FadeIn(optimal_tsp_cost),
            FadeOut(better_lower_bound),
            FadeOut(new_result),
            FadeOut(steps),
        )
        self.wait()

        one_tree_lower_bound_text = (
            Text("1-tree Lower Bound", font=REDUCIBLE_FONT, weight=BOLD)
            .scale(0.7)
            .move_to(DOWN * 3.2)
        )
        self.play(FadeIn(one_tree_lower_bound_text))
        self.wait()

    def highlight_visited_univisited(
        self, vertices, labels, visited, unvisited, scale=0.7
    ):
        visited_group = VGroup(
            *[vertices[v].copy().scale(scale) for v in visited]
        ).arrange(RIGHT)
        unvisited_group = VGroup(
            *[vertices[v].copy().scale(scale) for v in unvisited]
        ).arrange(RIGHT)
        visited_group.move_to(LEFT * 3.5 + DOWN * 3.5)
        unvisited_group.move_to(RIGHT * 3.5 + DOWN * 3.5)
        for mob in visited_group:
            mob[0].set_fill(opacity=0.5).set_stroke(opacity=1)
            mob[1].set_fill(opacity=1)
        for mob in unvisited_group:
            mob[0].set_fill(opacity=0.2).set_stroke(opacity=0.2)
            mob[1].set_fill(opacity=0.2)
        visited_dict = {v: visited_group[i] for i, v in enumerate(visited)}
        unvisited_dict = {v: unvisited_group[i] for i, v in enumerate(unvisited)}

        return visited_group, unvisited_group, visited_dict, unvisited_dict

    def get_neighboring_edges_across_sets(self, set1, set2):
        edges = []
        for v in set1:
            for u in set2:
                edges.append((v, u))
        return edges

    def get_graph_with_random_layout(self, N, radius=0.1):
        graph = TSPGraph(
            list(range(N)),
            labels=False,
            layout=self.get_random_layout(N),
            vertex_config={
                "stroke_color": REDUCIBLE_PURPLE,
                "stroke_width": 3,
                "fill_color": REDUCIBLE_PURPLE,
                "fill_opacity": 0.5,
                "radius": radius,
            },
            edge_config={
                "color": REDUCIBLE_VIOLET,
                "stroke_width": 2,
            },
        )
        return graph

    def scale_graph_with_tour(self, graph, tour_edges, scale):
        tour_edges_group = VGroup(*list(tour_edges.values()))
        graph_with_tour_edges = VGroup(graph, tour_edges_group).scale(scale)
        return graph_with_tour_edges

    def is_equal(self, edge1, edge2):
        return edge1 == edge2 or (edge1[1], edge1[0]) == edge2

    def get_edge(self, edge_dict, edge):
        if edge in edge_dict:
            return edge_dict[edge]
        else:
            return edge_dict[(edge[1], edge[0])]


class GreedyApproach2(LowerBoundTSP):
    def construct(self):
        self.add(BACKGROUND_IMG)
        self.show_greedy_algorithm()

    def show_greedy_algorithm(self):
        np.random.seed(5)
        self.add(BACKGROUND_IMG)
        NUM_VERTICES = 10
        layout = self.get_random_layout(NUM_VERTICES)
        # MANUAL ADJUSTMENTS FOR BETTER INSTRUCTIONAL EXAMPLE
        layout[1] = RIGHT * 3.5 + UP * 2
        graph = TSPGraph(list(range(NUM_VERTICES)), layout=layout).scale(0.8)

        # title = (
        #     Text("Greedy Heuristic", font=REDUCIBLE_FONT, weight=BOLD)
        #     .scale(0.8)
        #     .move_to(UP * 3.5)
        # )

        self.play(FadeIn(graph))

        # all_edges = graph.get_all_edges(buff=graph.vertices[0].width / 2)

        # self.play(*[Write(edge.set_stroke(opacity=0.3)) for edge in all_edges.values()])
        # self.wait()

        # self.play(*[Unwrite(edge) for edge in all_edges.values()])
        # self.wait()

        self.perform_algorithm(graph)

    def perform_algorithm(self, graph):
        all_edges = graph.get_all_edges(buff=graph.vertices[0].width / 2)
        edges_sorted = sorted(
            [edge for edge in all_edges],
            key=lambda x: graph.get_dist_matrix()[x[0]][x[1]],
        )
        added_edges = []
        for edge in edges_sorted:
            degree_map = self.get_degree_map(graph, added_edges)
            if len(added_edges) == len(graph.vertices) - 1:
                degrees_sorted = sorted(
                    list(degree_map.keys()), key=lambda x: degree_map[x]
                )
                final_edge = (degrees_sorted[0], degrees_sorted[1])
                added_edges.append(final_edge)
                edge_mob = all_edges[final_edge].set_stroke(color=REDUCIBLE_YELLOW)
                self.play(Write(edge_mob))
                # self.wait()
                break
            u, v = edge
            edge_mob = all_edges[edge].set_stroke(color=REDUCIBLE_YELLOW)
            if degree_map[u] == 2 or degree_map[v] == 2:
                self.play(Write(edge_mob.set_stroke(color=REDUCIBLE_CHARM)))
                # self.wait()
                # show degree issue
                # if 3 in edge and 4 in edge:
                #     surround_rects = [
                #         SurroundingRectangle(graph.vertices[i]).set_color(
                #             REDUCIBLE_CHARM
                #         )
                #         for i in [3, 4]
                #     ]
                #     degree_comment = (
                #         Tex(r"Degree $>$ 2").scale(0.7).next_to(surround_rects[0], UP)
                #     )
                #     self.play(
                #         *[Write(rect) for rect in surround_rects],
                #         FadeIn(degree_comment),
                #     )
                #     # self.wait()

                #     self.play(
                #         FadeOut(edge_mob),
                #         FadeOut(degree_comment),
                #         *[FadeOut(rect) for rect in surround_rects],
                #     )
                #     self.wait()
                #     continue

                self.play(
                    FadeOut(edge_mob),
                )
                self.wait()
                continue

            if self.is_connected(u, v, added_edges):
                print(u, v, "is connected already, so would cause cycle")
                # would create cycle
                self.play(Write(edge_mob.set_stroke(color=REDUCIBLE_CHARM)))
                # self.wait()

                # surround_rect = SurroundingRectangle(
                #     VGroup(
                #         graph.vertices[0],
                #         graph.vertices[2],
                #         graph.vertices[4],
                #     )
                # ).set_color(REDUCIBLE_CHARM)

                # cycle = (
                #     Text("Cycle", font=REDUCIBLE_FONT)
                #     .scale(0.6)
                #     .next_to(surround_rect, UP)
                # )

                # self.play(Write(surround_rect), FadeIn(cycle))
                # self.wait()
                # self.play(FadeOut(edge_mob), FadeOut(surround_rect), FadeOut(cycle))
                self.play(FadeOut(edge_mob))
                # self.wait()
                continue
            added_edges.append(edge)
            self.play(Write(edge_mob))
            self.wait()

    def get_degree_map(self, graph, edges):
        v_to_degree = {v: 0 for v in graph.vertices}
        for edge in edges:
            u, v = edge
            v_to_degree[u] = v_to_degree.get(u, 0) + 1
            v_to_degree[v] = v_to_degree.get(v, 0) + 1
        return v_to_degree

    def is_connected(self, u, v, edges):
        visited = set()

        def dfs(u):
            visited.add(u)
            for v in self.get_neighbors(u, edges):
                if v not in visited:
                    dfs(v)

        dfs(u)
        print("visited", visited)
        return v in visited

    def get_neighbors(self, v, edges):
        neighbors = []
        for edge in edges:
            if v not in edge:
                continue
            neighbors.append(edge[0] if edge[0] != v else edge[1])
        return neighbors


class GreedApproachExtraText(Scene):
    def construct(self):
        title = Text(
            "Greedy Heuristic Approach", font=REDUCIBLE_FONT, weight=BOLD
        ).scale(0.8)
        title.move_to(UP * 3.5)
        average_case = (
            Tex(
                r"On average: $\frac{\text{Greedy Heuristic}}{\text{1-Tree Lower Bound}} = 1.17$"
            )
            .scale(0.8)
            .move_to(DOWN * 3.5)
        )
        self.play(Write(title))
        self.wait()

        self.play(FadeIn(average_case))
        self.wait()

        self.clear()
        self.add(BACKGROUND_IMG)

        screen_rect_left = ScreenRectangle(height=3)
        screen_rect_right = ScreenRectangle(height=3)
        screen_rects = VGroup(screen_rect_left, screen_rect_right).arrange(
            RIGHT, buff=1
        )
        self.play(FadeIn(screen_rects))
        self.wait()


class Christofides(GreedyApproach2):
    def construct(self):
        self.add(BACKGROUND_IMG)
        self.show_christofides()

    def show_christofides(self):
        (
            left_graph,
            mst_edges_mob,
            right_graph,
            tsp_edges_mob,
            left_edges,
            right_edges,
            mst_edges,
            tsp_tour_edges,
        ) = self.mst_step()

        (
            vertices_to_match,
            copied_matching_nodes,
            surround_circle_highlights,
        ) = self.odd_degree_step(
            left_graph,
            left_edges,
            right_graph,
            right_edges,
            tsp_edges_mob,
            mst_edges_mob,
            mst_edges,
            tsp_tour_edges,
        )

        (
            left_perfect_matching_mob,
            right_perfect_matching_mob,
            perfect_match,
        ) = self.min_weight_perfect_matching_step(
            left_graph,
            right_graph,
            vertices_to_match,
            copied_matching_nodes,
            surround_circle_highlights,
        )

        self.eulerian_tour_step(
            left_perfect_matching_mob,
            right_perfect_matching_mob,
            left_graph,
            left_edges,
            copied_matching_nodes,
            perfect_match,
            mst_edges,
        )

        # self.summarize_christofides()

    def mst_step(self):
        mst_label = Text("MST", font=REDUCIBLE_FONT, weight=BOLD).scale(0.8)
        tsp_label = Text("Optimal TSP", font=REDUCIBLE_FONT, weight=BOLD).scale(0.8)

        (
            left_graph,
            mst_edges_mob,
            right_graph,
            tsp_edges_mob,
            left_edges,
            right_edges,
            mst_edges,
            tsp_tour_edges,
        ) = self.get_mst_and_tsp_tour(10)
        self.play(
            *[GrowFromCenter(left_graph.vertices[v]) for v in left_graph.vertices],
            *[GrowFromCenter(right_graph.vertices[v]) for v in right_graph.vertices],
            *[GrowFromCenter(edge) for edge in mst_edges_mob],
            *[GrowFromCenter(edge) for edge in tsp_edges_mob],
        )
        self.wait()

        mst_label.next_to(left_graph, UP).to_edge(UP * 2)
        tsp_label.next_to(right_graph, UP).to_edge(UP * 2)

        self.play(
            Write(mst_label),
            Write(tsp_label),
        )
        self.wait()
        long_right_arrow = MathTex(r"\Longrightarrow").move_to(
            (mst_label.get_center() + tsp_label.get_center()) / 2
        )
        long_right_arrow.scale(1.5)
        num_iterations = 5
        NUM_VERTICES = 10
        for i in range(0, num_iterations * 2, 2):
            (
                new_left_graph,
                new_mst_edges_mob,
                new_right_graph,
                new_tsp_edges_mob,
                left_edges,
                right_edges,
                mst_edges,
                tsp_tour_edges,
            ) = self.get_mst_and_tsp_tour(i, NUM_VERTICES=NUM_VERTICES)
            self.play(
                Transform(left_graph, new_left_graph),
                Transform(right_graph, new_right_graph),
                Transform(mst_edges_mob, new_mst_edges_mob),
                Transform(tsp_edges_mob, new_tsp_edges_mob),
            )
            self.wait()

            if i == 8:
                self.play(Write(long_right_arrow))

        (
            new_left_graph,
            new_mst_edges_mob,
            new_right_graph,
            new_tsp_edges_mob,
            left_edges,
            right_edges,
            mst_edges,
            tsp_tour_edges,
        ) = self.get_mst_and_tsp_tour(6, NUM_VERTICES=13)

        self.play(
            FadeTransform(left_graph, new_left_graph),
            FadeTransform(right_graph, new_right_graph),
            FadeTransform(mst_edges_mob, new_mst_edges_mob),
            FadeTransform(tsp_edges_mob, new_tsp_edges_mob),
        )

        self.play(FadeOut(mst_label), FadeOut(tsp_label), FadeOut(long_right_arrow))
        self.wait()

        return (
            new_left_graph,
            new_mst_edges_mob,
            new_right_graph,
            new_tsp_edges_mob,
            left_edges,
            right_edges,
            mst_edges,
            tsp_tour_edges,
        )

    def get_node_with_opacity(self, node, opacity=0.2):
        fill_opacity = opacity
        if opacity == 1:
            fill_opacity = 0.5
        node_copy = node.copy()
        node_copy[0].set_fill(opacity=fill_opacity).set_stroke(opacity=opacity)
        node_copy[1].set_fill(opacity=opacity)
        return node_copy

    def odd_degree_step(
        self,
        left_graph,
        left_edges,
        right_graph,
        right_edges,
        tsp_edges_mob,
        mst_edges_mob,
        mst_edges,
        tsp_tour_edges,
    ):
        tsp_tour_edge_to_index = {edge: i for i, edge in enumerate(tsp_tour_edges)}
        for v in right_graph.vertices:
            neighboring_edges = self.get_neighboring_edges(v, tsp_tour_edges)
            right_node_highlighted = self.get_node_with_opacity(
                right_graph.vertices[v], opacity=1
            )
            other_nodes_indices = [i for i in right_graph.vertices if i != v]
            self.play(
                Transform(right_graph.vertices[v], right_node_highlighted),
                *[
                    tsp_edges_mob[tsp_tour_edge_to_index[edge]].animate.set_stroke(
                        opacity=1
                    )
                    for edge in neighboring_edges
                ],
                *[
                    tsp_edges_mob[tsp_tour_edge_to_index[edge]].animate.set_stroke(
                        opacity=0.2
                    )
                    for edge in tsp_tour_edges
                    if edge not in neighboring_edges
                ],
                *[
                    Transform(
                        right_graph.vertices[i],
                        self.get_node_with_opacity(
                            right_graph.vertices[i], opacity=0.2
                        ),
                    )
                    for i in other_nodes_indices
                ],
            )
            self.wait()

        self.play(
            *[
                tsp_edges_mob[tsp_tour_edge_to_index[edge]].animate.set_stroke(
                    opacity=1
                )
                for edge in tsp_tour_edges
            ],
            *[
                Transform(
                    right_graph.vertices[i],
                    self.get_node_with_opacity(right_graph.vertices[i], opacity=1),
                )
                for i in other_nodes_indices
            ],
        )
        self.wait()

        degree_map = get_degrees_for_all_vertices(
            mst_edges, left_graph.get_dist_matrix()
        )
        vertices_to_match = [v for v in degree_map if degree_map[v] % 2 == 1]
        surround_circle_highlights = [
            get_glowing_surround_circle(left_graph.vertices[v], color=REDUCIBLE_CHARM)
            for v in vertices_to_match
        ]

        self.play(*[FadeIn(circle) for circle in surround_circle_highlights])
        self.wait()
        copied_matching_nodes = [
            right_graph.vertices[v].copy() for v in vertices_to_match
        ]
        self.play(
            FadeOut(right_graph),
            FadeOut(tsp_edges_mob),
            *[
                TransformFromCopy(left_graph.vertices[v], copied_matching_nodes[i])
                for i, v in enumerate(vertices_to_match)
            ],
        )
        self.wait()
        return vertices_to_match, copied_matching_nodes, surround_circle_highlights

    def min_weight_perfect_matching_step(
        self,
        left_graph,
        right_graph,
        vertices_to_match,
        copied_matching_nodes,
        surround_circle_highlights,
    ):
        all_perfect_matches = get_all_perfect_matchings(vertices_to_match)
        right_node_map = {
            v: copied_matching_nodes[i] for i, v in enumerate(vertices_to_match)
        }
        left_perfect_matching_mob = self.get_perfect_matching_edges(
            all_perfect_matches[0], left_graph.vertices
        )
        right_perfect_matching_mob = self.get_perfect_matching_edges(
            all_perfect_matches[0], right_node_map
        )
        self.play(
            FadeIn(left_perfect_matching_mob),
            FadeIn(right_perfect_matching_mob),
        )
        self.wait()

        minimum_weight_label = Text("Minimum Weight Cost: ", font=REDUCIBLE_MONO).scale(
            0.4
        )
        best_cost = get_cost_from_edges(
            all_perfect_matches[0], left_graph.get_dist_matrix()
        )
        minimum_weight_text = (
            Text(str(np.round(best_cost, 2)), font=REDUCIBLE_MONO)
            .scale(0.4)
            .next_to(minimum_weight_label, RIGHT)
        )
        minimum_weight_label_and_cost = VGroup(
            minimum_weight_label, minimum_weight_text
        )
        minimum_weight_label_and_cost.next_to(left_graph, DOWN).to_edge(DOWN * 2)

        current_cost_label = Text("Perfect Matching Cost: ", font=REDUCIBLE_MONO).scale(
            0.4
        )
        current_cost = get_cost_from_edges(
            all_perfect_matches[0], left_graph.get_dist_matrix()
        )
        current_cost_text = (
            Text(str(np.round(current_cost, 2)), font=REDUCIBLE_MONO)
            .scale(0.4)
            .next_to(current_cost_label, RIGHT)
        )
        current_cost_label_and_cost = VGroup(current_cost_label, current_cost_text)
        current_cost_label_and_cost.next_to(right_graph, DOWN).to_edge(DOWN * 2)
        self.play(
            Write(minimum_weight_label_and_cost),
            Write(current_cost_label_and_cost),
        )
        self.wait()

        run_time = 1
        best_index = None
        import time

        for i, perfect_matching in enumerate(all_perfect_matches):
            start_time = time.time()
            if i == 0:
                continue
            new_left_perfect_matching_mob = self.get_perfect_matching_edges(
                perfect_matching, left_graph.vertices
            )
            new_right_perfect_matching_mob = self.get_perfect_matching_edges(
                perfect_matching, right_node_map
            )
            current_cost = get_cost_from_edges(
                perfect_matching, left_graph.get_dist_matrix()
            )
            new_current_cost_text = (
                Text(str(np.round(current_cost, 2)), font=REDUCIBLE_MONO)
                .scale(0.4)
                .next_to(current_cost_label, RIGHT)
            )
            start_time = time.time()
            if current_cost < best_cost:
                best_cost = current_cost
                best_index = i
                new_min_weight_text = (
                    Text(str(np.round(best_cost, 2)), font=REDUCIBLE_MONO)
                    .scale(0.4)
                    .next_to(minimum_weight_label, RIGHT)
                )
                self.play(
                    Transform(left_perfect_matching_mob, new_left_perfect_matching_mob),
                    Transform(
                        right_perfect_matching_mob, new_right_perfect_matching_mob
                    ),
                    Transform(current_cost_text, new_current_cost_text),
                    Transform(minimum_weight_text, new_min_weight_text),
                    run_time=run_time,
                )
            else:
                self.play(
                    Transform(
                        right_perfect_matching_mob, new_right_perfect_matching_mob
                    ),
                    Transform(current_cost_text, new_current_cost_text),
                    run_time=run_time,
                )

            if i < 5:
                self.wait()
            # self.wait(wait_time)
            run_time = run_time * 0.9

        new_right_perfect_matching_mob = self.get_perfect_matching_edges(
            all_perfect_matches[best_index], right_node_map
        )
        self.wait()
        self.play(
            Transform(right_perfect_matching_mob, new_right_perfect_matching_mob),
            FadeOut(current_cost_label_and_cost),
        )
        self.wait()

        self.play(
            *[FadeOut(highlight) for highlight in surround_circle_highlights],
            FadeOut(minimum_weight_label_and_cost),
        )
        self.wait()

        return (
            left_perfect_matching_mob,
            right_perfect_matching_mob,
            all_perfect_matches[best_index],
        )

    def get_perfect_matching_edges(self, perfect_matching, node_map):
        perfect_matching_edges = VGroup()
        for u, v in perfect_matching:
            edge_mob = self.make_edge(node_map[u], node_map[v])
            perfect_matching_edges.add(edge_mob)
        return perfect_matching_edges

    def make_edge(self, node1, node2, stroke_width=3, color=REDUCIBLE_GREEN_LIGHTER):
        return Line(
            node1.get_center(),
            node2.get_center(),
            buff=node1.width / 2,
            stroke_width=stroke_width,
        ).set_color(color)

    def eulerian_tour_step(
        self,
        left_perfect_matching_mob,
        right_perfect_matching_mob,
        left_graph,
        left_edges,
        copied_matching_nodes,
        perfect_matching,
        mst_edges,
    ):
        mst_edges_mob = VGroup(*[self.get_edge(left_edges, edge) for edge in mst_edges])
        multigraph = VGroup(left_graph, mst_edges_mob, left_perfect_matching_mob)

        multigraph_title = (
            Text(
                "MST and Minimum Weight Perfect Matching MultiGraph",
                font=REDUCIBLE_FONT,
                weight=BOLD,
            )
            .scale(0.6)
            .move_to(UP * 3.2)
        )
        self.play(
            *[FadeOut(node) for node in copied_matching_nodes],
            FadeOut(right_perfect_matching_mob),
            multigraph.animate.scale(1.2).move_to(ORIGIN),
            FadeIn(multigraph_title),
        )
        self.wait()

        self.play(
            *[
                self.get_edge(left_edges, edge).animate.set_stroke(width=6)
                for edge in mst_edges
            ]
        )
        self.wait()

        duplicate_edge_t_1 = Text("Duplicate Edge", font=REDUCIBLE_FONT).scale(0.4)
        duplicate_edge_t_2 = duplicate_edge_t_1.copy()

        left_arrow = Arrow(
            ORIGIN, UL * 1.2, max_tip_length_to_length_ratio=0.15
        ).set_color(REDUCIBLE_VIOLET)
        right_arrow = Arrow(
            ORIGIN, RIGHT * 1.5, max_tip_length_to_length_ratio=0.15
        ).set_color(REDUCIBLE_VIOLET)

        duplicate_edge_t_1.next_to(left_arrow, DR)
        right_arrow_group = (
            VGroup(duplicate_edge_t_1, right_arrow)
            .arrange(RIGHT)
            .next_to(self.get_edge(left_edges, (3, 6)), LEFT)
        )
        left_arrow_group = VGroup(left_arrow, duplicate_edge_t_2).next_to(
            self.get_edge(left_edges, (9, 11)), DR, buff=SMALL_BUFF / 2
        )

        self.play(
            FadeIn(right_arrow_group, direction=LEFT),
            FadeIn(left_arrow_group, direction=RIGHT),
        )
        self.wait()

        self.play(
            FadeOut(right_arrow_group, direction=LEFT),
            FadeOut(left_arrow_group, direction=RIGHT),
        )
        self.wait()

        key_observation = (
            Text("All vertices have even degree", font=REDUCIBLE_FONT)
            .scale(0.5)
            .move_to(DOWN * 3.2)
        )

        self.play(FadeIn(key_observation))
        self.wait()

        find_eulerian_text = (
            Text("Find Eulerian Tour of Multigraph", font=REDUCIBLE_FONT, weight=BOLD)
            .scale(0.6)
            .move_to(multigraph_title.get_center())
        )

        self.play(
            FadeTransform(multigraph_title, find_eulerian_text),
            FadeOut(key_observation),
        )
        self.wait()

        eulerian_tour = get_eulerian_tour(mst_edges, perfect_matching, start=0)
        ordered_vertices_tex = self.get_tex_eulerian_tour(eulerian_tour).move_to(
            DOWN * 2.9
        )
        self.play(FadeIn(ordered_vertices_tex))
        self.wait()

        eulerian_tour_edge_map = self.get_eulerian_tour_edges_map(
            left_graph, eulerian_tour
        )

        for edge in eulerian_tour_edge_map:
            self.play(
                Create(eulerian_tour_edge_map[edge]),
                Flash(eulerian_tour_edge_map[edge].copy()),
            )

        self.wait()

        generate_tsp_tour_step = Text(
            "Generate TSP Tour from Eulerian Tour", font=REDUCIBLE_FONT, weight=BOLD
        ).scale(0.6)
        generate_tsp_tour_step.move_to(find_eulerian_text.get_center())

        self.play(
            *[FadeOut(e) for e in eulerian_tour_edge_map.values()],
            *[FadeOut(self.get_edge(left_edges, edge)) for edge in mst_edges],
            FadeOut(left_perfect_matching_mob),
            FadeTransform(find_eulerian_text, generate_tsp_tour_step),
        )
        self.wait()

        self.hamiltonian_tour_step(
            left_graph, eulerian_tour, ordered_vertices_tex, generate_tsp_tour_step
        )

    def get_ordered_vertices(self, eulerian_tour):
        ordered_vertices = []
        for u, v in eulerian_tour:
            ordered_vertices.append(u)
        ordered_vertices.append(v)
        return ordered_vertices

    def get_tex_eulerian_tour(self, eulerian_tour, scale=0.3):
        ordered_vertices = self.get_ordered_vertices(eulerian_tour)
        arrow = Arrow(ORIGIN, RIGHT, max_tip_length_to_length_ratio=0.15).set_color(
            REDUCIBLE_VIOLET
        )
        text_with_arrows = []
        for i, v in enumerate(ordered_vertices):
            text = Text(str(v), font=REDUCIBLE_MONO).scale(scale)
            text_with_arrows.append(text)
            if i < len(ordered_vertices) - 1:
                text_with_arrows.append(arrow.copy().scale(0.8))

        return VGroup(*text_with_arrows).arrange(RIGHT, buff=SMALL_BUFF)

    def hamiltonian_tour_step(
        self, left_graph, eulerian_tour, ordered_vertices_tex, generate_tsp_tour_step
    ):
        tsp_tour = get_hamiltonian_tour_from_eulerian(eulerian_tour)
        tsp_tour_edges = get_edges_from_tour(tsp_tour)
        ordered_vertices = self.get_ordered_vertices(eulerian_tour)

        tsp_tour_edges_mob = self.get_tsp_tour_edges_mob(left_graph, tsp_tour_edges)
        index = 0
        visited = set()

        v_to_tex_end_index_map = {}
        current_index = 1
        for i, v in enumerate(ordered_vertices):
            v_to_tex_end_index_map[i] = current_index
            current_index += 2

        tsp_tour_edges_index = 0
        glowing_circles = VGroup()
        visited = set()
        crosses = []
        for i, v in enumerate(ordered_vertices):
            glowing_circle = get_glowing_surround_circle(left_graph.vertices[v])
            if i == 0:
                self.play(
                    ordered_vertices_tex[: v_to_tex_end_index_map[i]].animate.set_fill(
                        opacity=1
                    ),
                    ordered_vertices_tex[v_to_tex_end_index_map[i] :].animate.set_fill(
                        opacity=0.5
                    ),
                    FadeIn(glowing_circle),
                )
                glowing_circles.add(glowing_circle)

            elif v in visited and v != ordered_vertices[0]:
                end_index = v_to_tex_end_index_map[i]
                cross = Cross(ordered_vertices_tex[end_index - 1]).set_color(
                    REDUCIBLE_CHARM
                )
                cross.set_stroke(width=2)
                self.play(
                    ordered_vertices_tex[: v_to_tex_end_index_map[i]].animate.set_fill(
                        opacity=1
                    ),
                    ordered_vertices_tex[v_to_tex_end_index_map[i] :].animate.set_fill(
                        opacity=0.5
                    ),
                )
                self.play(
                    Write(cross),
                )
                self.add_foreground_mobject(cross)
                crosses.append(cross)
            else:
                if v != ordered_vertices[0]:
                    self.play(
                        ordered_vertices_tex[
                            : v_to_tex_end_index_map[i]
                        ].animate.set_fill(opacity=1),
                        ordered_vertices_tex[
                            v_to_tex_end_index_map[i] :
                        ].animate.set_fill(opacity=0.5),
                        Write(tsp_tour_edges_mob[tsp_tour_edges_index]),
                        FadeIn(glowing_circle),
                    )
                    glowing_circles.add(glowing_circle)

                else:
                    self.play(
                        ordered_vertices_tex[
                            : v_to_tex_end_index_map[i]
                        ].animate.set_fill(opacity=1),
                        ordered_vertices_tex[
                            v_to_tex_end_index_map[i] :
                        ].animate.set_fill(opacity=0.5),
                        Write(tsp_tour_edges_mob[tsp_tour_edges_index]),
                    )
                tsp_tour_edges_index += 1
                visited.add(v)

            self.wait()

        tex_tsp_tour = self.get_tex_eulerian_tour(tsp_tour_edges, scale=0.32)
        tex_tsp_tour_label = Text(
            "TSP Tour: ", font=REDUCIBLE_MONO
        ).scale_to_fit_height(tex_tsp_tour.height)
        text_tsp_tour_with_label = VGroup(tex_tsp_tour_label, tex_tsp_tour).arrange(
            RIGHT
        )
        text_tsp_tour_with_label.next_to(ordered_vertices_tex, DOWN)

        self.play(
            FadeOut(glowing_circles),
            FadeIn(text_tsp_tour_with_label),
        )
        self.wait()

        self.play(
            FadeOut(ordered_vertices_tex),
            FadeOut(text_tsp_tour_with_label),
            FadeOut(generate_tsp_tour_step),
            *[FadeOut(c) for c in crosses],
        )
        graph_with_tsp_tour = VGroup(left_graph, tsp_tour_edges_mob)
        christofides_cost = get_cost_from_edges(
            tsp_tour_edges, left_graph.get_dist_matrix()
        )
        self.play(graph_with_tsp_tour.animate.scale(0.8).move_to(LEFT * 3.5 + UP * 0.5))
        right_graph = left_graph.copy().move_to(RIGHT * 3.5 + UP * 0.5)
        optimal_tsp_tour, optimal_cost = get_exact_tsp_solution(
            left_graph.get_dist_matrix()
        )

        right_graph_tsp_edges_mob = self.get_tsp_tour_edges_mob(
            right_graph, get_edges_from_tour(optimal_tsp_tour)
        )

        self.play(
            FadeIn(right_graph),
            FadeIn(right_graph_tsp_edges_mob),
        )
        self.wait()
        christofides_cost_text = Text(
            f"Christofides tour cost: {np.round(christofides_cost, 2)}",
            font=REDUCIBLE_MONO,
        ).scale(0.5)
        christofides_cost_text.next_to(left_graph, DOWN).to_edge(DOWN * 3)

        optimal_cost_text = Text(
            f"Optimal tour cost: {np.round(optimal_cost, 2)}", font=REDUCIBLE_MONO
        ).scale(0.5)
        optimal_cost_text.next_to(right_graph, DOWN).to_edge(DOWN * 3)

        self.play(
            FadeIn(christofides_cost_text),
            FadeIn(optimal_cost_text),
        )
        self.wait()

        self.clear()
        self.add(BACKGROUND_IMG)

    def summarize_christofides(self):

        christofides_alg = (
            Text("Christofides Algorithm", font=REDUCIBLE_FONT, weight=BOLD)
            .scale(0.8)
            .move_to(UP * 3.3)
        )
        screen_rect = ScreenRectangle(height=3).move_to(LEFT * 3 + UP * 0.5)

        step_1 = Tex(r"1. Find MST $T$ of Graph")
        step_2 = Tex(r"2. Isolate Set of Odd-Degree Vertices $S$")
        step_3 = Tex(r"3. Find Min Weight Perfect Matching $M$ of $S$")
        step_4 = Tex(r"4. Combine $T$ and $M$ into Multigraph $G$")
        step_5 = Tex(r"5. Generate Eulerian Tour of $G$")
        step_6 = Tex(r"6. Generate TSP Tour from Eulerian Tour")

        steps = (
            VGroup(*[step_1, step_2, step_3, step_4, step_5, step_6])
            .scale(0.6)
            .arrange(DOWN, aligned_edge=LEFT)
        )
        steps.move_to(RIGHT * 3.5 + UP * 0.5)

        self.play(Write(christofides_alg))

        on_average_perf = average_case = (
            Tex(
                r"On average: $\frac{\text{Christofides}}{\text{1-Tree Lower Bound}} = 1.1$"
            )
            .scale(0.75)
            .move_to(DOWN * 3)
        )
        for i, step in enumerate(steps):
            if i == 0:
                self.play(FadeIn(screen_rect), FadeIn(step))
            else:
                self.play(FadeIn(step))
            self.wait()

        self.play(FadeIn(on_average_perf))
        self.wait()

        worst_case_perf = (
            Tex(r"Worst case: $\frac{\text{Christofides}}{\text{Optimal TSP}} = 1.5$")
            .scale(0.75)
            .move_to(DOWN * 3 + RIGHT * 3.5)
        )
        on_average_perf_copy = on_average_perf.copy().next_to(screen_rect, DOWN)
        self.play(
            on_average_perf.animate.move_to(
                DOWN * 3 + RIGHT * on_average_perf_copy.get_center()[0]
            )
        )

        self.play(Write(worst_case_perf))
        self.wait()

    def get_eulerian_tour_edges_map(self, left_graph, eulerian_tour):
        all_edges = {}
        for edge in eulerian_tour:
            u, v = edge
            if (v, u) in all_edges:
                all_edges[(u, v)] = self.make_edge(
                    left_graph.vertices[u],
                    left_graph.vertices[v],
                    stroke_width=7,
                    color=REDUCIBLE_PURPLE,
                )
            else:
                all_edges[edge] = self.make_edge(
                    left_graph.vertices[u],
                    left_graph.vertices[v],
                    stroke_width=7,
                    color=REDUCIBLE_VIOLET,
                )

        return all_edges

    def get_tsp_tour_edges_mob(
        self, graph, tsp_tour_edges, stroke_width=3, color=REDUCIBLE_VIOLET
    ):
        return VGroup(
            *[
                self.make_edge(
                    graph.vertices[u],
                    graph.vertices[v],
                    stroke_width=stroke_width,
                    color=color,
                )
                for u, v in tsp_tour_edges
            ]
        )

    def get_mst_and_tsp_tour(self, seed, NUM_VERTICES=10):
        np.random.seed(seed)
        layout = self.get_random_layout(NUM_VERTICES)
        left_graph = TSPGraph(list(range(NUM_VERTICES)), layout=layout).scale(0.55)
        right_graph = left_graph.copy()

        VGroup(left_graph, right_graph).arrange(RIGHT, buff=1.2)

        left_edges = left_graph.get_all_edges(buff=left_graph.vertices[0].width / 2)
        right_edges = right_graph.get_all_edges(buff=right_graph.vertices[0].width / 2)
        mst_edges, mst_cost = get_mst(left_graph.get_dist_matrix())
        tsp_tour, optimal_cost = get_exact_tsp_solution(right_graph.get_dist_matrix())
        tsp_tour_edges = get_edges_from_tour(tsp_tour)
        mst_edges_mob = VGroup(
            *[
                self.get_edge(left_edges, edge).set_color(REDUCIBLE_YELLOW)
                for edge in mst_edges
            ]
        )
        tsp_edges_mob = VGroup(
            *[self.get_edge(right_edges, edge) for edge in tsp_tour_edges]
        )

        for i, edge in enumerate(tsp_tour_edges):
            if edge in mst_edges or (edge[1], edge[0]) in mst_edges:
                tsp_edges_mob[i].set_color(REDUCIBLE_YELLOW)

        return (
            left_graph,
            mst_edges_mob,
            right_graph,
            tsp_edges_mob,
            left_edges,
            right_edges,
            mst_edges,
            tsp_tour_edges,
        )

    def get_neighboring_edges(self, vertex, edges):
        return [edge for edge in edges if vertex in edge]


class TourImprovement(Christofides):
    def construct(self):
        # self.add(BACKGROUND_IMG)
        # self.intro_idea()
        # self.clear()
        self.add(BACKGROUND_IMG)
        self.show_random_swaps()
        # self.show_two_opt_switches()
        # self.k_opt_improvement()

    def intro_idea(self):
        NUM_VERTICES = 10
        layout = self.get_random_layout(NUM_VERTICES)
        input_graph = TSPGraph(list(range(NUM_VERTICES)), layout=layout).scale(0.45)

        heuristic_solution_mod = Module(
            ["Heuristic", "Solution"],
            text_weight=BOLD,
        )
        heuristic_solution_mod.text.scale(0.8)
        heuristic_solution_mod.scale(0.7)
        output_graph = input_graph.copy()
        tsp_tour_h, h_cost = christofides(output_graph.get_dist_matrix())

        arrow_1 = Arrow(
            LEFT * 1.5, ORIGIN, max_tip_length_to_length_ratio=0.15
        ).set_color(GRAY)
        arrow_2 = Arrow(
            ORIGIN, RIGHT * 1.5, max_tip_length_to_length_ratio=0.15
        ).set_color(GRAY)

        tsp_tour_edges_mob = self.get_tsp_tour_edges_mob(output_graph, tsp_tour_h)
        output_graph_with_edges = VGroup(output_graph, tsp_tour_edges_mob)

        entire_group = (
            VGroup(
                input_graph,
                arrow_1,
                heuristic_solution_mod,
                arrow_2,
                output_graph_with_edges,
            )
            .arrange(RIGHT, buff=0.5)
            .scale(0.8)
        )

        self.play(FadeIn(entire_group))
        self.wait()

        improve_question = (
            Text("Can we improve this solution?", font=REDUCIBLE_FONT, weight=BOLD)
            .scale(0.8)
            .move_to(UP * 3.3)
        )

        self.play(Write(improve_question))

        self.wait()

        self.play(entire_group.animate.next_to(improve_question, DOWN, buff=1))
        self.wait()

        input_graph_ls = output_graph_with_edges.copy()

        local_search_mod = Module(
            ["Local", "Search"],
            REDUCIBLE_GREEN_DARKER,
            REDUCIBLE_GREEN_LIGHTER,
            text_weight=BOLD,
        )
        local_search_mod.text.scale(0.6)
        local_search_mod.scale(0.7)
        output_graph_ls = input_graph.copy()
        tsp_tour, cost = get_exact_tsp_solution(output_graph.get_dist_matrix())
        tsp_tour_edges = get_edges_from_tour(tsp_tour)
        arrow_3 = Arrow(
            LEFT * 1.5, ORIGIN, max_tip_length_to_length_ratio=0.15
        ).set_color(GRAY)
        arrow_4 = Arrow(
            ORIGIN, RIGHT * 1.5, max_tip_length_to_length_ratio=0.15
        ).set_color(GRAY)

        tsp_tour_edges_mob_opt = self.get_tsp_tour_edges_mob(
            output_graph_ls, tsp_tour_edges
        )
        output_graph_with_edges_opt = VGroup(output_graph_ls, tsp_tour_edges_mob_opt)

        entire_group_opt = (
            VGroup(
                input_graph_ls.scale(1.2),
                arrow_3,
                local_search_mod,
                arrow_4,
                output_graph_with_edges_opt.scale(1.2),
            )
            .arrange(RIGHT, buff=0.5)
            .scale(0.8)
        ).move_to(DOWN * 2.2)

        self.play(FadeIn(entire_group_opt))
        self.wait()
        text_scale = 0.4
        nn = Text("Nearest Neighbor", font=REDUCIBLE_FONT).scale(text_scale)
        greedy = Text("Greedy", font=REDUCIBLE_FONT).scale(text_scale)
        christofides_text = Text("Christofides", font=REDUCIBLE_FONT).scale(text_scale)

        VGroup(nn, greedy).arrange(DOWN).next_to(heuristic_solution_mod, UP)
        christofides_text.next_to(heuristic_solution_mod, DOWN)

        random_swapping = (
            Text("Random swapping", font=REDUCIBLE_FONT)
            .scale(text_scale)
            .next_to(local_search_mod, UP)
        )
        two_opt = Text("2-opt", font=REDUCIBLE_FONT).scale(text_scale)
        three_opt = Text("3-opt", font=REDUCIBLE_FONT).scale(text_scale)
        VGroup(two_opt, three_opt).arrange(DOWN).next_to(local_search_mod, DOWN)

        self.play(FadeIn(random_swapping), FadeIn(two_opt), FadeIn(three_opt))
        self.wait()

        self.play(FadeIn(nn), FadeIn(greedy), FadeIn(christofides_text))
        self.wait()

    def show_random_swaps(self):
        np.random.seed(9)
        NUM_VERTICES = 10
        layout = self.get_random_layout(NUM_VERTICES)
        graph = (
            TSPGraph(list(range(NUM_VERTICES)), layout=layout)
            .scale(0.7)
            .shift(UP * 0.5)
        )
        nn_tour, nn_cost = get_nearest_neighbor_solution(graph.get_dist_matrix())

        nn_tour_edges = get_edges_from_tour(nn_tour)
        nn_tour_edges_mob = self.get_tsp_tour_edges_mob(graph, nn_tour_edges)

        self.play(
            *[GrowFromCenter(v) for v in graph.vertices.values()],
        )
        self.wait()

        self.play(LaggedStartMap(Write, nn_tour_edges_mob))
        self.wait()

        tsp_tour_text = self.get_tex_eulerian_tour(nn_tour_edges, scale=0.5)
        cost = Text(
            f"Nearest Neighbor Cost: {np.round(nn_cost, 2)}", font=REDUCIBLE_MONO
        ).scale(0.4)

        tsp_tour_text.to_edge(DOWN * 2)
        cost.next_to(tsp_tour_text, DOWN)

        self.play(FadeIn(tsp_tour_text), FadeIn(cost))
        self.wait()

        random_swapping = (
            Text("Random Swapping", font=REDUCIBLE_FONT, weight=BOLD)
            .scale(0.7)
            .move_to(UP * 3.3)
        )

        self.play(Write(random_swapping))
        self.wait()

        nn_tour_swapped = nn_tour.copy()
        nn_tour_swapped[5], nn_tour_swapped[-2] = (
            nn_tour_swapped[-2],
            nn_tour_swapped[5],
        )
        nn_tour_swapped_edges = get_edges_from_tour(nn_tour_swapped)
        nn_tour_swapped_cost = get_cost_from_edges(
            nn_tour_swapped_edges, graph.get_dist_matrix()
        )

        tex_v_7 = tsp_tour_text[5 * 2]
        tex_v_6 = tsp_tour_text[8 * 2]
        self.play(
            tex_v_7.animate.set_color(REDUCIBLE_YELLOW),
            tex_v_6.animate.set_color(REDUCIBLE_GREEN_LIGHTER),
        )
        self.wait()

        print(nn_tour_edges)
        print(nn_tour_swapped_edges)

        nn_tour_edges_mob_swapped = self.get_tsp_tour_edges_mob(
            graph, nn_tour_swapped_edges
        )

        self.play(
            tex_v_7.animate.move_to(tex_v_6.get_center()),
            tex_v_6.animate.move_to(tex_v_7.get_center()),
            ReplacementTransform(nn_tour_edges_mob, nn_tour_edges_mob_swapped),
        )
        self.wait()

        new_nn_cost = (
            Text(
                f"Nearest Neighbbor + Random Swap Cost: {np.round(nn_tour_swapped_cost, 2)}",
                font=REDUCIBLE_MONO,
            )
            .scale(0.4)
            .move_to(cost.get_center())
        )

        self.play(ReplacementTransform(cost, new_nn_cost))
        self.wait()
        self.clear()
        self.add(BACKGROUND_IMG)

    def show_two_opt_switches(self):
        title = (
            Text("2-opt Improvement", font=REDUCIBLE_FONT, weight=BOLD)
            .scale(0.7)
            .move_to(UP * 3.3)
        )
        np.random.seed(13)
        NUM_VERTICES = 10

        layout = self.get_random_layout(NUM_VERTICES)
        layout[3] += LEFT * 0.8
        layout[5] += DOWN * 0.8
        graph = TSPGraph(list(range(NUM_VERTICES)), layout=layout).scale(0.7)
        all_edges = graph.get_all_edges(buff=graph.vertices[0].width / 2)

        nn_tour, nn_cost = get_nearest_neighbor_solution(graph.get_dist_matrix())

        nn_tour_edges = get_edges_from_tour(nn_tour)
        nn_tour_edges_mob = self.get_tsp_tour_edges_mob(graph, nn_tour_edges)
        tour_edges_to_line_map = self.get_tsp_tour_edges_map(graph, nn_tour_edges)

        self.play(
            *[GrowFromCenter(v) for v in graph.vertices.values()],
        )
        self.play(LaggedStartMap(Write, tour_edges_to_line_map.values()))
        self.wait()

        cost = (
            Text(f"Nearest Neighbor Cost: {np.round(nn_cost, 2)}", font=REDUCIBLE_MONO)
            .scale(0.4)
            .to_edge(DOWN * 2)
        )
        self.play(FadeIn(cost))
        self.wait()

        self.play(Write(title))
        self.wait()

        current_cost = nn_cost
        current_tour = nn_tour
        current_tour_edges = nn_tour_edges
        current_tour_map = tour_edges_to_line_map

        improvement_cost = (
            Text(
                f"After 2-opt Improvement: {np.round(nn_cost, 2)}", font=REDUCIBLE_MONO
            )
            .scale(0.4)
            .next_to(cost, DOWN)
        )
        self.play(FadeIn(improvement_cost))
        self.wait()

        for i in range(len(current_tour_edges) - 1):
            for j in range(i + 1, len(current_tour_edges)):
                e1, e2 = current_tour_edges[i], current_tour_edges[j]
                new_e1, new_e2, new_tour = get_two_opt_new_edges(current_tour, e1, e2)
                if new_e1 != e1 and new_e2 != e2:
                    new_tour_edges = get_edges_from_tour(new_tour)
                    new_tour_cost = get_cost_from_edges(
                        new_tour_edges, graph.get_dist_matrix()
                    )
                    if new_tour_cost < current_cost:
                        new_improvement_cost = (
                            Text(
                                f"After 2-opt Improvement: {np.round(new_tour_cost, 1)}",
                                font=REDUCIBLE_MONO,
                            )
                            .scale(0.4)
                            .move_to(improvement_cost.get_center())
                        )

                        current_cost = new_tour_cost
                        new_edge_map = self.get_new_edge_map(
                            current_tour_map, new_tour_edges, graph, all_edges
                        )
                        surround_circle_highlights = [
                            get_glowing_surround_circle(graph.vertices[v])
                            for v in list(e1) + list(e2)
                        ]

                        current_e1_mob = self.get_edge(current_tour_map, e1)
                        current_e2_mob = self.get_edge(current_tour_map, e2)
                        self.play(
                            current_e1_mob.animate.set_color(REDUCIBLE_YELLOW),
                            current_e2_mob.animate.set_color(REDUCIBLE_YELLOW),
                            *[FadeIn(c) for c in surround_circle_highlights],
                        )
                        self.wait()

                        new_e1_mob = self.get_edge(new_edge_map, new_e1)
                        new_e2_mob = self.get_edge(new_edge_map, new_e2)
                        current_tour_edges = new_tour_edges
                        current_tour = new_tour
                        self.play(
                            ReplacementTransform(current_e1_mob, new_e1_mob),
                            ReplacementTransform(current_e2_mob, new_e2_mob),
                            Transform(improvement_cost, new_improvement_cost),
                        )
                        self.wait()

                        self.play(*[FadeOut(c) for c in surround_circle_highlights])
                        self.wait()
                        current_tour_map = new_edge_map

        optimal_tour, optimal_cost = get_exact_tsp_solution(graph.get_dist_matrix())
        print("Optimal tour and cost", optimal_tour, optimal_cost)

        three_opt_edges = [(3, 6), (8, 7), (7, 0)]
        new_three_opt_edges = [(3, 7), (7, 6), (8, 0)]
        three_opt_v = [3, 6, 8, 7, 0]

        new_title = (
            Text("3-opt Improvement", font=REDUCIBLE_FONT, weight=BOLD)
            .scale(0.7)
            .move_to(UP * 3.3)
        )

        s_circles = [
            get_glowing_surround_circle(graph.vertices[v]) for v in three_opt_v
        ]
        self.play(
            *[FadeIn(c) for c in s_circles],
            *[
                self.get_edge(current_tour_map, edge).animate.set_color(
                    REDUCIBLE_YELLOW
                )
                for edge in three_opt_edges
            ],
            Transform(title, new_title),
        )
        self.wait()

        new_improvement_cost = (
            Text(
                f"After 3-opt Improvement: {np.round(optimal_cost, 1)}",
                font=REDUCIBLE_MONO,
            )
            .scale(0.4)
            .move_to(improvement_cost.get_center())
        )

        self.play(
            *[
                Transform(
                    self.get_edge(current_tour_map, e1),
                    self.make_edge(
                        graph.vertices[e2[0]],
                        graph.vertices[e2[1]],
                        color=REDUCIBLE_VIOLET,
                    ),
                )
                for e1, e2 in zip(three_opt_edges, new_three_opt_edges)
            ],
            Transform(improvement_cost, new_improvement_cost),
        )
        self.wait()

        self.play(
            *[FadeOut(c) for c in s_circles],
        )
        self.wait()

        self.clear()
        self.add(BACKGROUND_IMG)

    def k_opt_improvement(self):
        title = (
            Text("k-opt Improvement", font=REDUCIBLE_FONT, weight=BOLD)
            .scale(0.7)
            .move_to(UP * 3.3)
        )
        definition = Tex(r"Replace $k$ edges of tour").scale(0.7).move_to(DOWN * 3.3)

        NUM_VERTICES = 10
        circular_graph = TSPGraph(
            list(range(NUM_VERTICES)),
            layout="circular",
        )
        optimal_tour, optimal_cost = get_exact_tsp_solution(
            circular_graph.get_dist_matrix()
        )
        opt_tour_edges = get_edges_from_tour(optimal_tour)
        opt_tour_edges_mob = self.get_tsp_tour_edges_mob(circular_graph, opt_tour_edges)
        self.play(
            Write(title),
            FadeIn(circular_graph),
            LaggedStartMap(GrowFromCenter, opt_tour_edges_mob),
        )
        self.wait()

        x_1, x_2, y_1, y_2, z_1, z_2 = 3, 2, 0, 9, 6, 5
        three_opt_v = [x_1, x_2, y_1, y_2, z_1, z_2]
        three_opt_original = [(x_1, x_2), (y_1, y_2), (z_1, z_2)]
        three_opt_1 = [(x_1, y_1), (x_2, z_1), (y_2, z_2)]
        three_opt_2 = [(x_1, z_1), (x_2, y_2), (y_1, z_2)]
        three_opt_3 = [(x_1, y_2), (x_2, z_2), (y_1, z_1)]
        three_opt_4 = [(x_1, y_2), (x_2, z_1), (y_1, z_2)]

        self.play(
            FadeIn(definition),
            opt_tour_edges_mob[opt_tour_edges.index((x_2, x_1))].animate.set_color(
                REDUCIBLE_YELLOW
            ),
            opt_tour_edges_mob[opt_tour_edges.index((y_2, y_1))].animate.set_color(
                REDUCIBLE_YELLOW
            ),
            opt_tour_edges_mob[opt_tour_edges.index((z_2, z_1))].animate.set_color(
                REDUCIBLE_YELLOW
            ),
        )
        self.wait()

        graph_with_tour_edges = VGroup(circular_graph, opt_tour_edges_mob)

        self.play(graph_with_tour_edges.animate.scale(0.6).shift(UP * 1.5))
        self.wait()

        opts = [three_opt_1, three_opt_2, three_opt_3, three_opt_4]
        graphs_with_edges = []
        for opt in opts:
            graph = circular_graph.copy().scale(0.8)
            three_opt_edges, indices = self.get_three_opt_edges(
                opt_tour_edges, three_opt_original, opt
            )
            tour_edges_mob = self.get_tsp_tour_edges_mob(graph, three_opt_edges)
            for i in indices:
                tour_edges_mob[i].set_color(REDUCIBLE_YELLOW)
            graphs_with_edges.append(VGroup(graph, tour_edges_mob))
        graphs_with_edges_group = VGroup(*graphs_with_edges).arrange(RIGHT, buff=0.5)
        graphs_with_edges_group.shift(DOWN * 1.5)
        self.play(FadeIn(graphs_with_edges_group))
        self.wait()

    def get_three_opt_edges(self, original_tour_edges, three_opt_original, three_opt):
        new_tour_edges = original_tour_edges.copy()
        indices = []
        for original, new in zip(three_opt_original, three_opt):
            if original in original_tour_edges:
                index = original_tour_edges.index(original)
            else:
                index = original_tour_edges.index((original[1], original[0]))
            new_tour_edges[index] = new
            indices.append(index)
        return new_tour_edges, indices

    def get_tsp_tour_edges_map(
        self, graph, tsp_tour_edges, stroke_width=3, color=REDUCIBLE_VIOLET
    ):
        return {
            (u, v): self.make_edge(
                graph.vertices[u],
                graph.vertices[v],
                stroke_width=stroke_width,
                color=REDUCIBLE_VIOLET,
            )
            for u, v in tsp_tour_edges
        }

    def get_new_edge_map(self, prev_edge_map, new_tour_edges, graph, all_edges):
        set_new_tour_edges = set(new_tour_edges)
        new_edge_map = {}
        remaining_edges = set()
        for edge in prev_edge_map:
            u, v = edge
            if (u, v) in set_new_tour_edges:
                new_edge_map[(u, v)] = prev_edge_map[(u, v)]
            elif (v, u) in set_new_tour_edges:
                new_edge_map[(v, u)] = prev_edge_map[(u, v)]
            else:
                remaining_edges.add(edge)
        new_edges = set()
        for edge in new_tour_edges:
            u, v = edge
            if (u, v) in new_edge_map or (v, u) in new_edge_map:
                continue
            new_edges.add(edge)
            new_edge_map[(u, v)] = self.make_edge(
                graph.vertices[u], graph.vertices[v], color=REDUCIBLE_VIOLET
            )
        all_edge_corresponding = []
        for u, v in remaining_edges:
            if (u, v) in all_edges:
                all_edge_corresponding.append((u, v))
            else:
                all_edge_corresponding.append((v, u))
        print("Remaining edges", remaining_edges)
        print("Edges in all_edges map", all_edge_corresponding)
        print("New edges", new_edges)
        # hard codes to get transforms in satisfying order
        if (1, 4) in new_edges:
            new_edge_map[(1, 4)] = self.make_edge(
                graph.vertices[4], graph.vertices[1], color=REDUCIBLE_VIOLET
            )
        if (6, 9) in new_edges:
            new_edge_map[(6, 9)] = self.make_edge(
                graph.vertices[9], graph.vertices[6], color=REDUCIBLE_VIOLET
            )
        if (5, 8) in new_edges:
            new_edge_map[(5, 8)] = self.make_edge(
                graph.vertices[8], graph.vertices[5], color=REDUCIBLE_VIOLET
            )
        return new_edge_map

    def get_aligned_edge(self, edge, remaining_edges):
        pass


class LocalMinima(TourImprovement):
    def construct(self):
        self.add(BACKGROUND_IMG)
        NUM_VERTICES = 6
        all_tours = get_all_tour_permutations(NUM_VERTICES, 0)
        np.random.shuffle(all_tours)
        rows = 6
        cols = len(all_tours) // rows
        all_tour_costs = []
        graph = self.get_graph(NUM_VERTICES)
        all_graphs_with_tours = []
        all_graphs_tour_edges = []
        for tour in all_tours:
            current_graph = graph.copy().scale(0.22)
            tour_edges = get_edges_from_tour(tour)
            all_tour_costs.append(
                get_cost_from_edges(tour_edges, graph.get_dist_matrix())
            )
            all_graphs_tour_edges.append(tour_edges)
            tour_edges_mob = self.get_tsp_tour_edges_mob(current_graph, tour_edges)
            all_graphs_with_tours.append(VGroup(current_graph, tour_edges_mob))

        all_graphs = VGroup(*all_graphs_with_tours).arrange_in_grid(rows=rows)
        graph_v = VGroup(*list(graph.vertices.values()))
        self.play(LaggedStartMap(GrowFromCenter, graph_v))
        self.wait()
        self.play(LaggedStart(FadeOut(graph_v), FadeIn(all_graphs)))
        self.add_foreground_mobject(all_graphs)
        self.wait()

        heat_map_grid = self.get_heat_map(all_graphs, rows, cols, all_tour_costs)

        self.play(FadeIn(heat_map_grid))
        self.wait()
        original_graphs = all_graphs.copy()
        faded_graphs = [graph.copy().fade(0.7) for graph in all_graphs]
        original_grids = heat_map_grid.copy()
        faded_grids = [grid.copy().fade(0.7) for grid in heat_map_grid]
        all_edge_diffs = self.get_all_graph_edge_diffs(all_tours)

        self.perform_two_opt_switches(
            10,
            all_graphs,
            heat_map_grid,
            all_tour_costs,
            all_edge_diffs,
            original_graphs,
            original_grids,
            faded_graphs,
            faded_grids,
        )

        self.perform_two_opt_switches(
            23,
            all_graphs,
            heat_map_grid,
            all_tour_costs,
            all_edge_diffs,
            original_graphs,
            original_grids,
            faded_graphs,
            faded_grids,
        )

        graph_with_grid = VGroup(all_graphs, heat_map_grid)

        self.play(graph_with_grid.animate.scale(0.8))
        self.wait()

        original_graphs = all_graphs.copy()
        faded_graphs = [graph.copy().fade(0.7) for graph in all_graphs]
        original_grids = heat_map_grid.copy()
        faded_grids = [grid.copy().fade(0.7) for grid in heat_map_grid]

        self.play(
            *self.fade_graphs_and_grids(
                all_graphs,
                heat_map_grid,
                [i for i in range(len(all_graphs)) if i not in [39, 44]],
                faded_graphs,
                faded_grids,
            )
        )
        self.wait()

        local_min_arrow = Arrow(
            heat_map_grid[44].get_bottom() + DOWN * 1.5,
            heat_map_grid[44].get_bottom(),
            max_tip_length_to_length_ratio=0.15,
            buff=SMALL_BUFF,
        )
        local_min_arrow.set_color(REDUCIBLE_YELLOW)
        local_min_text = (
            Text("Local Minimum", font=REDUCIBLE_FONT)
            .scale(0.4)
            .next_to(local_min_arrow, DOWN)
        )

        global_min_arrow = Arrow(
            heat_map_grid[39].get_right() + RIGHT * 1.5,
            heat_map_grid[39].get_right(),
            max_tip_length_to_length_ratio=0.15,
            buff=SMALL_BUFF,
        )
        global_min_arrow.set_color(REDUCIBLE_YELLOW)
        global_min_text = (
            VGroup(Text("Global").scale(0.4), Text("Minimum").scale(0.4))
            .arrange(DOWN)
            .next_to(global_min_arrow, UP)
        )

        self.add_foreground_mobjects(global_min_arrow, local_min_arrow)
        self.play(
            FadeIn(global_min_arrow),
            FadeIn(local_min_arrow),
            FadeIn(local_min_text),
            FadeIn(global_min_text),
        )
        self.wait()

        three_opt_text = Text("3-opt", font=REDUCIBLE_FONT).scale(0.4)
        three_opt_arrow = Arrow(
            heat_map_grid[44].get_right(),
            heat_map_grid[39].get_left(),
            max_tip_length_to_length_ratio=0.15,
        )
        three_opt_arrow.set_color(REDUCIBLE_YELLOW)
        three_opt_text.set_stroke(width=5, color=BLACK, background=True).next_to(
            three_opt_arrow, UP
        ).shift(DOWN * 0.5)
        self.add_foreground_mobjects(three_opt_arrow, three_opt_text)

        self.play(FadeIn(three_opt_text), FadeIn(three_opt_arrow))
        self.wait()

        start_index = 44

        neighboring_edges = [
            edge
            for edge in all_edge_diffs
            if start_index in edge
            and (all_edge_diffs[edge] == 2 or all_edge_diffs[edge] == 3)
        ]
        print("Neighboring edges", neighboring_edges)
        print([(key, val) for key, val in all_edge_diffs.items() if start_index in key])
        neighboring_vertices = []
        for u, v in neighboring_edges:
            if u == start_index:
                neighboring_vertices.append(v)
            else:
                neighboring_vertices.append(u)

        glowing_rect = get_glowing_surround_rect(heat_map_grid[start_index])
        self.play(
            FadeOut(three_opt_text),
            FadeOut(three_opt_arrow),
            FadeOut(global_min_arrow),
            FadeOut(global_min_text),
            FadeOut(local_min_text),
            FadeOut(local_min_arrow),
            FadeIn(glowing_rect),
        )
        self.wait()

        neighborhood_comment = Text(
            "2-opt AND 3-opt local search is expensive", font=REDUCIBLE_FONT
        ).scale(0.7)
        neighborhood_comment.next_to(heat_map_grid, DOWN)
        print("Neighboring vertices", neighboring_vertices)
        self.play(
            *self.highlight_graphs_and_grids(
                all_graphs,
                heat_map_grid,
                neighboring_vertices,
                original_graphs,
                original_grids,
            ),
            FadeIn(neighborhood_comment),
        )
        self.wait()

        self.play(
            FadeOut(neighborhood_comment),
            FadeOut(glowing_rect),
            *self.highlight_graphs_and_grids(
                all_graphs,
                heat_map_grid,
                list(range(60)),
                original_graphs,
                original_grids,
            ),
        )
        self.wait()

        self.perform_two_opt_switches_special(
            23,
            all_graphs,
            heat_map_grid,
            all_tour_costs,
            all_edge_diffs,
            original_graphs,
            original_grids,
            faded_graphs,
            faded_grids,
        )

    def perform_two_opt_switches(
        self,
        start_index,
        all_graphs,
        heat_map_grid,
        all_tour_costs,
        all_edge_diffs,
        original_graphs,
        original_grids,
        faded_graphs,
        faded_grids,
    ):
        visited_vertices = set([start_index])
        iteration = 0
        surround_rect = None
        print("Start index", start_index)
        arrows = VGroup()
        visited_surround_rects = VGroup()
        while not self.is_minima_found(start_index, all_tour_costs, all_edge_diffs):
            (
                neighboring_edges,
                neighboring_vertices,
            ) = self.get_neighboring_edges_and_verticies(start_index, all_edge_diffs)

            if iteration == 0:
                surround_rect = get_glowing_surround_rect(heat_map_grid[start_index])
                self.play(FadeIn(surround_rect))
                self.wait()

            to_fade_vertices = [
                i
                for i in range(len(all_graphs))
                if i not in visited_vertices and i not in neighboring_vertices
            ]
            self.play(
                *self.fade_graphs_and_grids(
                    all_graphs,
                    heat_map_grid,
                    to_fade_vertices,
                    faded_graphs,
                    faded_grids,
                )
                + self.highlight_graphs_and_grids(
                    all_graphs,
                    heat_map_grid,
                    neighboring_vertices,
                    original_graphs,
                    original_grids,
                )
            )
            self.wait()
            visited_surround_rect = SurroundingRectangle(
                heat_map_grid[start_index], buff=0
            ).set_stroke(width=3)
            visited_surround_rects.add(visited_surround_rect)
            start_index = min(neighboring_vertices, key=lambda x: all_tour_costs[x])
            new_surround_rect = get_glowing_surround_rect(heat_map_grid[start_index])
            arrow = Arrow(
                surround_rect.get_center(),
                new_surround_rect.get_center(),
                color=REDUCIBLE_YELLOW,
            )
            self.add_foreground_mobject(arrow)
            arrows.add(arrow)
            self.play(
                FadeOut(surround_rect),
                FadeIn(new_surround_rect),
                FadeIn(visited_surround_rect),
                FadeIn(arrow),
            )
            self.wait()
            surround_rect = new_surround_rect

            visited_vertices.add(start_index)
            iteration += 1
        print("Ending index", start_index)
        best_index = all_tour_costs.index(min(all_tour_costs))
        if start_index != best_index:
            print(
                f"*** FOUND MISMATCH *** found_index: {start_index}, best_index: {best_index}"
            )

        (
            neighboring_edges,
            neighboring_vertices,
        ) = self.get_neighboring_edges_and_verticies(start_index, all_edge_diffs)

        to_fade_vertices = [
            i
            for i in range(len(all_graphs))
            if i not in visited_vertices and i not in neighboring_vertices
        ]

        self.play(
            *self.fade_graphs_and_grids(
                all_graphs,
                heat_map_grid,
                to_fade_vertices,
                faded_graphs,
                faded_grids,
            )
            + self.highlight_graphs_and_grids(
                all_graphs,
                heat_map_grid,
                neighboring_vertices,
                original_graphs,
                original_grids,
            )
        )
        self.wait()
        self.remove_foreground_mobjects(*[arrow for arrow in arrows])
        self.play(
            FadeOut(arrows),
            FadeOut(visited_surround_rects),
            FadeOut(new_surround_rect),
            *self.highlight_graphs_and_grids(
                all_graphs,
                heat_map_grid,
                to_fade_vertices,
                original_graphs,
                original_grids,
            ),
        )
        self.wait()

    def perform_two_opt_switches_special(
        self,
        start_index,
        all_graphs,
        heat_map_grid,
        all_tour_costs,
        all_edge_diffs,
        original_graphs,
        original_grids,
        faded_graphs,
        faded_grids,
    ):
        visited_vertices = set([start_index])
        iteration = 0
        surround_rect = None
        print("Start index", start_index)
        arrows = VGroup()
        visited_surround_rects = VGroup()
        rect_color = REDUCIBLE_YELLOW
        special_arrow = None
        while not self.is_minima_found(start_index, all_tour_costs, all_edge_diffs):
            (
                neighboring_edges,
                neighboring_vertices,
            ) = self.get_neighboring_edges_and_verticies(start_index, all_edge_diffs)

            if iteration == 0:
                surround_rect = get_glowing_surround_rect(heat_map_grid[start_index])
                self.play(FadeIn(surround_rect))
                self.wait()

            to_fade_vertices = [
                i
                for i in range(len(all_graphs))
                if i not in visited_vertices and i not in neighboring_vertices
            ]
            self.play(
                *self.fade_graphs_and_grids(
                    all_graphs,
                    heat_map_grid,
                    to_fade_vertices,
                    faded_graphs,
                    faded_grids,
                )
                + self.highlight_graphs_and_grids(
                    all_graphs,
                    heat_map_grid,
                    neighboring_vertices,
                    original_graphs,
                    original_grids,
                )
            )
            self.wait()
            visited_surround_rect = SurroundingRectangle(
                heat_map_grid[start_index], buff=0, color=rect_color
            ).set_stroke(width=3)
            visited_surround_rects.add(visited_surround_rect)
            start_index = min(neighboring_vertices, key=lambda x: all_tour_costs[x])
            rect_color = REDUCIBLE_YELLOW
            if iteration == 1:
                special_arrow = VGroup()
                special_arrow.add(
                    Line(
                        heat_map_grid[32].get_left(),
                        heat_map_grid[32].get_left() + LEFT * 2.5,
                    )
                )
                special_arrow.add(
                    Line(
                        special_arrow[-1].get_end(),
                        special_arrow[-1].get_end() + UP * 4,
                    )
                )

                special_arrow.add(
                    Line(
                        special_arrow[-1].get_end(),
                        special_arrow[-1].get_end()[1] * UP
                        + RIGHT * heat_map_grid[8].get_center()[0],
                    )
                )

                special_arrow.add(
                    Arrow(
                        special_arrow[-1].get_end(), heat_map_grid[8].get_top(), buff=0
                    )
                )
                special_arrow.set_color(REDUCIBLE_CHARM)

                start_index = 8
                rect_color = REDUCIBLE_CHARM
                self.add_foreground_mobject(special_arrow)

            new_surround_rect = get_glowing_surround_rect(
                heat_map_grid[start_index], color=rect_color
            )
            arrow = Arrow(
                surround_rect.get_center(),
                new_surround_rect.get_center(),
                color=REDUCIBLE_YELLOW,
            )
            arrows.add(arrow)
            if iteration != 1:
                self.play(
                    FadeOut(surround_rect),
                    FadeIn(new_surround_rect),
                    FadeIn(visited_surround_rect),
                    # Write(arrow),
                )
            else:
                self.play(
                    FadeOut(surround_rect),
                    FadeIn(new_surround_rect),
                    FadeIn(visited_surround_rect),
                    FadeIn(special_arrow),
                )
            self.wait()
            surround_rect = new_surround_rect

            visited_vertices.add(start_index)
            iteration += 1
        print("Ending index", start_index)
        best_index = all_tour_costs.index(min(all_tour_costs))
        if start_index != best_index:
            print(
                f"*** FOUND MISMATCH *** found_index: {start_index}, best_index: {best_index}"
            )

        (
            neighboring_edges,
            neighboring_vertices,
        ) = self.get_neighboring_edges_and_verticies(start_index, all_edge_diffs)

        to_fade_vertices = [
            i
            for i in range(len(all_graphs))
            if i not in visited_vertices and i not in neighboring_vertices
        ]

        self.play(
            *self.fade_graphs_and_grids(
                all_graphs,
                heat_map_grid,
                [0, 5, 12, 26, 41, 53, 56],
                faded_graphs,
                faded_grids,
            )
        )
        self.wait()

        result = Tex(
            r"Sub-optimal Exploration $\rightarrow$ optimal solution",
        ).scale(0.8)
        result.next_to(heat_map_grid, DOWN)

        self.play(FadeIn(result))
        self.wait()

    def is_minima_found(self, start_index, all_tour_costs, all_edge_diffs):
        (
            neighboring_edges,
            neighboring_vertices,
        ) = self.get_neighboring_edges_and_verticies(start_index, all_edge_diffs)
        current_cost = all_tour_costs[start_index]
        return current_cost <= min([all_tour_costs[v] for v in neighboring_vertices])

    def get_neighboring_edges_and_verticies(self, start_index, all_edge_diffs):
        neighboring_edges = [
            edge
            for edge in all_edge_diffs
            if start_index in edge and all_edge_diffs[edge] == 2
        ]
        neighboring_vertices = []
        for u, v in neighboring_edges:
            if u == start_index:
                neighboring_vertices.append(v)
            else:
                neighboring_vertices.append(u)
        return neighboring_edges, neighboring_vertices

    def highlight_graphs_and_grids(
        self, all_graphs, heat_map_grid, indices, original_graphs, original_grids
    ):
        animations = []
        for i in indices:
            animations.append(Transform(all_graphs[i], original_graphs[i]))
            animations.append(Transform(heat_map_grid[i], original_grids[i]))
        return animations

    def fade_graphs_and_grids(
        self, all_graphs, heat_map_grid, indices, faded_graphs, faded_grids
    ):
        animations = []
        for i in indices:
            animations.append(Transform(all_graphs[i], faded_graphs[i]))
            animations.append(Transform(heat_map_grid[i], faded_grids[i]))
        return animations

    def get_two_opt_neighbors(self, all_edges_diff, index):
        return [
            edge
            for edge in all_edge_diffs
            if index in edge and all_edge_diffs[index] == 2
        ]

    def get_all_graph_edge_diffs(self, all_tours):
        all_edge_diffs = {}
        all_graphs_tour_edges = [get_edges_from_tour(tour) for tour in all_tours]
        for i in range(len(all_tours) - 1):
            for j in range(i + 1, len(all_tours)):
                edge_diff = self.edge_diff(
                    all_graphs_tour_edges[i], all_graphs_tour_edges[j]
                )
                all_edge_diffs[(i, j)] = edge_diff

        return all_edge_diffs

    def edge_diff(self, tour_edges_1, tour_edges_2):
        edge_diff = 0
        set_edges_2 = set(tour_edges_2)
        for u, v in tour_edges_1:
            if (u, v) in set_edges_2 or (v, u) in set_edges_2:
                continue
            edge_diff += 1
        return edge_diff

    def get_heat_map(self, all_graphs, rows, cols, all_tour_costs):
        grid_cell_width = (all_graphs[1].get_center() - all_graphs[0].get_center())[0]
        grid_cell_height = (all_graphs[cols].get_center() - all_graphs[0].get_center())[
            1
        ]
        min_color = REDUCIBLE_GREEN_LIGHTER
        max_color = REDUCIBLE_CHARM
        max_cost = max(all_tour_costs)
        min_cost = min(all_tour_costs)
        grid = VGroup()
        for i, graph in enumerate(all_graphs):
            cost = all_tour_costs[i]
            alpha = (cost - min_cost) / (max_cost - min_cost)
            cell = BackgroundRectangle(
                graph,
                color=interpolate_color(min_color, max_color, alpha),
                fill_opacity=0.5,
                stroke_opacity=1,
            )
            cell.stretch_to_fit_height(grid_cell_height)
            cell.stretch_to_fit_width(grid_cell_width)
            cell.move_to(graph.get_center())
            grid.add(cell)

        return grid

    def get_2d_index(self, index, rows, cols):
        row_index = index // rows
        col_index = index % cols
        return row_index, col_index

    def get_graph(self, NUM_VERTICES):
        new_layout = {
            0: LEFT * 0.8,
            1: UP * 2 + LEFT * 2,
            2: UP * 2 + RIGHT * 2,
            3: RIGHT * 0.8,
            4: RIGHT * 2 + DOWN * 2,
            5: LEFT * 2 + DOWN * 2,
        }
        circular_graph = TSPGraph(list(range(NUM_VERTICES)), layout=new_layout)

        MIN_NOISE, MAX_NOISE = -0.24, 0.24
        for v in circular_graph.vertices:
            noise_vec = np.array(
                [
                    np.random.uniform(MIN_NOISE, MAX_NOISE),
                    np.random.uniform(MIN_NOISE, MAX_NOISE),
                    0,
                ]
            )
            new_layout[v] += noise_vec
        return TSPGraph(list(range(NUM_VERTICES)), layout=new_layout)


class Ant:
    def __init__(self, alpha, beta, tsp_graph, pheromone_matrix):
        self.alpha = alpha
        self.beta = beta
        self.tsp_graph = tsp_graph
        self.cost_matrix = tsp_graph.get_dist_matrix()
        self.pheromone_matrix = pheromone_matrix
        self.num_nodes = self.cost_matrix.shape[0]
        self.mob = Dot(radius=0.02, color=REDUCIBLE_YELLOW)

    def get_mob(self):
        return self.mob

    def get_tour(self):
        tour = [np.random.choice(list(range(self.num_nodes)))]
        while len(tour) < self.num_nodes:
            current = tour[-1]
            neighbors = get_unvisited_neighbors(current, tour, self.num_nodes)
            neighbor_distribution = self.calc_distribution(current, neighbors)
            next_to_vist = np.random.choice(neighbors, p=neighbor_distribution)
            tour.append(next_to_vist)
        self.mob.move_to(self.tsp_graph.vertices[tour[0]].get_center())
        return tour

    def calc_distribution(self, current, neighbors):
        cost_weights = [
            (1 / self.cost_matrix[current][neighbor]) ** self.beta
            for neighbor in neighbors
        ]
        pheromone_weights = [
            (1 / self.pheromone_matrix[current][neighbor]) ** self.alpha
            for neighbor in neighbors
        ]
        denominator = np.dot(cost_weights, pheromone_weights)
        return [
            cost_weight * pheromone_weight / denominator
            for cost_weight, pheromone_weight in zip(cost_weights, pheromone_weights)
        ]


class AntColonySimulation:
    def __init__(
        self,
        tsp_graph,
        num_ants,
        alpha=4,
        beta=10,
        evaporation_rate=0.5,
        elitist_weight=1,
    ):
        self.tsp_graph = tsp_graph
        self.cost_matrix = tsp_graph.get_dist_matrix()
        self.alpha = alpha
        self.beta = beta
        self.pheromone_matrix = np.ones(self.cost_matrix.shape)
        self.ants = [
            Ant(alpha, beta, tsp_graph, self.pheromone_matrix) for _ in range(num_ants)
        ]
        self.evaporation_rate = evaporation_rate
        self.best_tour = None
        self.best_tour_edges = None
        self.best_tour_cost = float("inf")
        self.elitist_weight = elitist_weight

    def update(self, elitist=False):
        total_pheromones_added = np.zeros(self.cost_matrix.shape)
        all_tours = []
        for i, ant in enumerate(self.ants):
            tour = ant.get_tour()
            tour_edges = get_edges_from_tour(tour)
            all_tours.append(tour)
            tour_cost = get_cost_from_edges(tour_edges, self.cost_matrix)
            if tour_cost < self.best_tour_cost:
                self.best_tour_cost = tour_cost
                self.best_tour = tour
                self.best_tour_edges = tour_edges
            for u, v in tour_edges:
                total_pheromones_added[u][v] += 1 / tour_cost
                total_pheromones_added[v][u] += 1 / tour_cost

        self.pheromone_matrix = (
            1 - self.evaporation_rate
        ) * self.pheromone_matrix + total_pheromones_added
        if elitist:
            for i, j in self.best_tour_edges:
                self.pheromone_matrix[i][j] += self.elitist_weight / self.best_tour_cost

        for ant in self.ants:
            ant.pheromone_matrix = self.pheromone_matrix
        return all_tours


class AntColonyExplanation(TourImprovement):
    def construct(self):
        self.add(BACKGROUND_IMG)
        np.random.seed(7)
        NUM_VERTICES = 7
        layout = self.get_random_layout(NUM_VERTICES)
        left_graph = TSPGraph(list(range(NUM_VERTICES)), layout=layout)
        left_graph.scale(0.55)

        right_graph = left_graph.copy()
        graphs = VGroup(left_graph, right_graph).arrange(RIGHT, buff=1.5).shift(UP * 2)
        v_line = Line(UP * config.frame_height / 2, DOWN * config.frame_height / 2)
        v_line.set_color(GRAY)
        self.play(FadeIn(v_line), FadeIn(graphs))
        self.wait()

        (
            left_to_fade,
            right_to_fade,
            bar_charts,
            starting_circles,
            left_edges,
            right_edges,
        ) = self.demo_weighted_strategy(graphs)

        self.play(FadeOut(v_line))
        self.wait()

        self.play(
            *[FadeOut(f) for f in left_to_fade + right_to_fade],
        )
        self.wait()

        cost_equation = MathTex(
            r"P(u, v) = \frac{D(u, v)^{-1}}{\sum_{j \in \text{adj}(u)} D(u, j)^{-1}}"
        ).scale(0.8)
        cost_equation.move_to(DOWN * 3.2)
        self.play(FadeIn(cost_equation))
        self.wait()

        left_edges_map, right_edges_map = self.generate_tours(
            graphs, bar_charts, starting_circles, left_edges, right_edges, cost_equation
        )

        centered_graph, reward_matrix_group, reward_matrix = self.show_reward_matrix(
            graphs, left_edges_map, right_edges_map
        )
        reward_matrix_mob = reward_matrix_group[1]
        animations = []
        start = 0
        neighboring_edges = centered_graph.get_neighboring_edges(
            start, buff=centered_graph.vertices[0].width / 2
        )
        starting_glowing_circle = get_glowing_surround_circle(
            centered_graph.vertices[start]
        )
        animations.append(FadeIn(starting_glowing_circle))
        animations.append(LaggedStartMap(Write, neighboring_edges.values()))
        self.play(*animations)
        self.wait()

        cost_equation = (
            MathTex(
                r"P(u, v) = \frac{D(u, v)^{-1}}{\sum_{j \in \text{adj}(u)} D(u, j)^{-1}}"
            )
            .scale(0.8)
            .next_to(reward_matrix_group, RIGHT, buff=1)
        )
        self.play(FadeIn(cost_equation))
        self.wait()

        cost_equation_with_rewards = (
            MathTex(
                r"P(u, v) = \frac{D(u, v)^{-1} \cdot R(u, v)}{\sum_{j \in \text{adj}(u)} D(u, j)^{-1} \cdot R(u, j)}"
            )
            .scale(0.7)
            .move_to(cost_equation.get_center())
        )

        self.play(FadeTransform(cost_equation, cost_equation_with_rewards))
        self.wait()

        self.play(
            *[FadeOut(edge_mob) for edge_mob in neighboring_edges.values()],
            FadeOut(starting_glowing_circle),
        )
        self.wait()
        self.generate_tours_and_update_matrix(
            centered_graph, reward_matrix_mob, reward_matrix, cost_equation_with_rewards
        )

    def generate_tours_and_update_matrix(
        self, graph, reward_matrix_mob, reward_matrix, cost_equation
    ):
        tour, tour_edges = self.get_random_tour(graph, reward_matrix)
        tour_cost = get_cost_from_edges(tour_edges, graph.get_dist_matrix())
        tour_cost_text = (
            Text(f"Tour cost: {np.round(tour_cost, 1)}", font=REDUCIBLE_MONO)
            .scale(0.4)
            .next_to(graph, UP)
        )
        tour_edges_mob = self.get_tsp_tour_edges_mob(graph, tour_edges)
        for edge in tour_edges_mob:
            self.play(Create(edge), rate_func=linear, run_time=0.5)
        self.play(FadeIn(tour_cost_text))
        self.wait()

        reward_update_eq = Tex(
            r"$R(u, v) \leftarrow R(u, v) + \frac{1}{\text{tour cost}}$ if $(u, v) \in$ tour"
        ).scale(0.7)
        reward_update_eq.next_to(cost_equation, DOWN)
        self.play(
            cost_equation.animate.shift(UP * 0.5),
            FadeIn(reward_update_eq.shift(UP * 0.5)),
        )
        self.wait()

        surround_rects, matrix_update_map = self.bulk_update_matrix(
            graph.get_dist_matrix(),
            reward_matrix,
            reward_matrix_mob,
            tour_edges,
            tour_cost,
            color=REDUCIBLE_VIOLET,
        )

        self.play(
            *[Flash(rect, color=REDUCIBLE_VIOLET) for rect in surround_rects],
            *[
                Transform(reward_matrix_mob[0][index], matrix_update_map[index])
                for index in matrix_update_map
            ],
        )
        self.wait()

        self.play(
            FadeOut(tour_edges_mob),
            FadeOut(tour_cost_text),
        )

        num_steps = 50
        run_time = 1
        for i in range(num_steps):
            print("step", i)
            tour, tour_edges = self.get_random_tour(graph, reward_matrix)
            tour_cost = get_cost_from_edges(tour_edges, graph.get_dist_matrix())
            tour_cost_text = (
                Text(f"Tour cost: {np.round(tour_cost, 1)}", font=REDUCIBLE_MONO)
                .scale(0.4)
                .next_to(graph, UP)
            )
            tour_edges_mob = self.get_tsp_tour_edges_mob(graph, tour_edges)
            if i < 5:
                for edge in tour_edges_mob:
                    self.play(Create(edge), rate_func=linear, run_time=0.5)
                self.play(FadeIn(tour_cost_text))
                self.wait()
            else:
                self.add(tour_edges_mob)
                self.add(tour_cost_text)

            surround_rects, matrix_update_map = self.bulk_update_matrix(
                graph.get_dist_matrix(),
                reward_matrix,
                reward_matrix_mob,
                tour_edges,
                tour_cost,
                color=REDUCIBLE_VIOLET,
            )

            self.play(
                *[Flash(rect, color=REDUCIBLE_VIOLET) for rect in surround_rects],
                *[
                    Transform(reward_matrix_mob[0][index], matrix_update_map[index])
                    for index in matrix_update_map
                ],
                run_time=run_time,
            )
            self.wait(run_time)
            run_time = run_time * 0.9

            if i < 5:
                self.play(
                    FadeOut(tour_edges_mob),
                    FadeOut(tour_cost_text),
                )
            else:
                self.remove(tour_edges_mob)
                self.remove(tour_cost_text)

        self.clear()
        self.add(BACKGROUND_IMG)

    def get_random_tour(self, graph, reward_matrix):
        start = np.random.choice(list(graph.vertices.keys()))
        tour = [start]
        tour_edges = []
        visited = set([start])
        while len(tour) < len(graph.vertices):
            neighboring_v = get_neighbors(tour[-1], len(graph.vertices))
            valid_v = [v for v in neighboring_v if v not in visited]
            edge = self.get_next_edge(graph, tour[-1], valid_v, reward_matrix)
            visited.add(edge[1])
            tour.append(edge[1])
            tour_edges.append(edge)

        tour_edges.append((tour[-1], tour[0]))
        return tour, tour_edges

    def generate_tours(
        self,
        graphs,
        bar_charts,
        starting_circles,
        left_edges,
        right_edges,
        cost_equation,
    ):
        left_color, right_color = REDUCIBLE_YELLOW, REDUCIBLE_GREEN_LIGHTER
        left_graph, right_graph = graphs
        left_bar_chart, right_bar_chart = bar_charts
        left_tour = [0]
        right_tour = [2]
        left_visited = set(left_tour)
        right_visited = set(right_tour)
        new_left_bar_chart, new_right_bar_chart = None, None
        left_edges_map = {}
        right_edges_map = {}
        reward_matrix = np.ones(left_graph.get_dist_matrix().shape)
        while len(left_tour) < len(left_graph.vertices) and len(right_tour) < len(
            right_graph.vertices
        ):
            left_neighboring_v = get_neighbors(left_tour[-1], len(left_graph.vertices))
            right_neighboring_v = get_neighbors(
                right_tour[-1], len(right_graph.vertices)
            )
            valid_left_v = [v for v in left_neighboring_v if v not in left_visited]
            valid_right_v = [v for v in right_neighboring_v if v not in right_visited]
            left_edge = self.get_next_edge(
                left_graph, left_tour[-1], valid_left_v, reward_matrix
            )
            right_edge = self.get_next_edge(
                right_graph, right_tour[-1], valid_right_v, reward_matrix
            )
            if len(left_tour) > 1:
                left_edges = {
                    (left_tour[-1], v): left_graph.create_edge(
                        left_tour[-1], v, buff=left_graph.vertices[0].width / 2
                    )
                    for v in valid_left_v
                }
                right_edges = {
                    (right_tour[-1], v): right_graph.create_edge(
                        right_tour[-1], v, buff=right_graph.vertices[0].width / 2
                    )
                    for v in valid_right_v
                }
                self.play(
                    LaggedStartMap(Write, list(left_edges.values())),
                    LaggedStartMap(Write, list(right_edges.values())),
                )
                self.wait()

                new_left_bar_chart = self.get_distribution_bar_chart(
                    left_graph,
                    left_tour[-1],
                    valid_left_v,
                    reward_matrix,
                ).next_to(left_graph, DOWN)
                new_right_bar_chart = self.get_distribution_bar_chart(
                    right_graph,
                    right_tour[-1],
                    valid_right_v,
                    reward_matrix,
                ).next_to(right_graph, DOWN)

                self.play(
                    Transform(left_bar_chart, new_left_bar_chart),
                    Transform(right_bar_chart, new_right_bar_chart),
                )
                self.wait()

            self.play(
                self.get_edge(left_edges, left_edge).animate.set_color(left_color),
                self.get_edge(right_edges, right_edge).animate.set_color(right_color),
            )
            left_edges_map[left_edge] = self.get_edge(left_edges, left_edge)
            right_edges_map[right_edge] = self.get_edge(right_edges, right_edge)
            new_starting_circles = VGroup(
                get_glowing_surround_circle(
                    left_graph.vertices[left_edge[1]], color=left_color
                ),
                get_glowing_surround_circle(
                    right_graph.vertices[right_edge[1]], color=right_color
                ),
            )
            self.play(
                FadeOut(starting_circles),
                FadeIn(new_starting_circles),
                *[
                    FadeOut(edge_mob)
                    for edge, edge_mob in left_edges.items()
                    if edge != left_edge and edge != (left_edge[1], left_edge[0])
                ],
                *[
                    FadeOut(edge_mob)
                    for edge, edge_mob in right_edges.items()
                    if edge != right_edge and edge != (right_edge[1], right_edge[0])
                ],
            )
            self.wait()
            left_visited.add(left_edge[1])
            right_visited.add(right_edge[1])
            left_tour.append(left_edge[1])
            right_tour.append(right_edge[1])
            starting_circles = new_starting_circles

        final_left_edge = left_graph.create_edge(
            left_tour[-1], left_tour[0], buff=left_graph.vertices[0].width / 2
        ).set_color(left_color)
        final_right_edge = right_graph.create_edge(
            right_tour[-1],
            right_tour[0],
            buff=right_graph.vertices[0].width / 2,
        ).set_color(right_color)

        left_edges_map[(left_tour[-1], left_tour[0])] = final_left_edge
        right_edges_map[(right_tour[-1], right_tour[0])] = final_right_edge

        self.play(
            Write(final_left_edge),
            Write(final_right_edge),
            FadeOut(left_bar_chart),
            FadeOut(right_bar_chart),
            FadeOut(cost_equation),
            FadeOut(starting_circles),
        )
        self.wait()

        return left_edges_map, right_edges_map

    def show_reward_matrix(self, graphs, left_edges_map, right_edges_map):
        self.NUM_DECIMAL_PLACES = 3
        left_graph, right_graph = graphs
        left_tour_edges = list(left_edges_map.keys())
        right_tour_edges = list(right_edges_map.keys())
        left_tour_cost = get_cost_from_edges(
            left_tour_edges, left_graph.get_dist_matrix()
        )
        right_tour_cost = get_cost_from_edges(
            right_tour_edges, right_graph.get_dist_matrix()
        )
        left_tour_cost_text = (
            Text(f"Tour cost: {np.round(left_tour_cost, 1)}", font=REDUCIBLE_MONO)
            .scale(0.4)
            .next_to(left_graph, DOWN)
        )
        right_tour_cost_text = (
            Text(f"Tour cost: {np.round(right_tour_cost, 1)}", font=REDUCIBLE_MONO)
            .scale(0.4)
            .next_to(right_graph, DOWN)
        )
        self.play(FadeIn(left_tour_cost_text), FadeIn(right_tour_cost_text))
        self.wait()

        question = (
            Tex(
                "Can we use these tour cost observations \\\\ to generate better tours by future salesman?"
            )
            .scale(0.8)
            .next_to(graphs, DOWN, buff=1)
        )

        self.play(FadeIn(question))
        self.wait()

        self.play(FadeOut(question))
        self.wait()

        reward_matrix = np.ones(left_graph.get_dist_matrix().shape)
        reward_matrix_mob = matrix_to_mob(reward_matrix, h_buff=2.6, v_buff=1.6).scale(
            1
        )
        reward_matrix_label = Tex(r"Reward $(R) =$").scale(1)
        reward_matrix_group = (
            VGroup(reward_matrix_label, reward_matrix_mob)
            .arrange(RIGHT)
            .move_to(DOWN * 2)
        )
        self.play(FadeIn(reward_matrix_group))
        self.wait()

        reward_amount = MathTex(r"+ \frac{1}{30.0}").scale(0.7)
        reward_amount.next_to(reward_matrix_label, DOWN, buff=0.5)

        for i, edge in enumerate(left_tour_edges):
            highlight_animations = self.highlight_edge(left_graph, left_edges_map, edge)
            surround_rects = (
                self.get_surrounded_rects_around_elements_and_update_rewards(
                    left_graph.get_dist_matrix(),
                    reward_matrix,
                    reward_matrix_mob,
                    [edge],
                    left_tour_cost,
                )
            )

            if i == 0:
                self.play(*highlight_animations)
                self.play(FadeIn(reward_amount))
                self.play(
                    left_tour_cost_text[-4:].animate.set_color(REDUCIBLE_YELLOW),
                    reward_amount[0][-4:].animate.set_color(REDUCIBLE_YELLOW),
                    *[Create(rect) for rect in surround_rects],
                )
            else:
                self.play(
                    *highlight_animations, *[Create(rect) for rect in surround_rects]
                )
            dist_matrix = left_graph.get_dist_matrix()

            one_d_index_1 = edge[0] * dist_matrix.shape[1] + edge[1]
            one_d_index_2 = edge[1] * dist_matrix.shape[1] + edge[0]
            new_reward_text_1 = Text(
                "{:.3f}".format(
                    np.round(reward_matrix[edge[0]][edge[1]], self.NUM_DECIMAL_PLACES)
                ),
                font=REDUCIBLE_MONO,
            )
            new_reward_text_1.scale_to_fit_height(
                reward_matrix_mob[0][one_d_index_1].height
            ).move_to(surround_rects[0].get_center())
            new_reward_text_2 = new_reward_text_1.copy().move_to(
                surround_rects[1].get_center()
            )

            self.play(
                Transform(reward_matrix_mob[0][one_d_index_1], new_reward_text_1),
                Transform(reward_matrix_mob[0][one_d_index_2], new_reward_text_2),
                *[Flash(rect, color=REDUCIBLE_YELLOW) for rect in surround_rects],
                *[FadeOut(rect) for rect in surround_rects],
            )

        self.play(
            *self.restore_all_edges_and_vertices(left_graph, left_edges_map),
            FadeOut(reward_amount),
            left_tour_cost_text[-4:].animate.set_color(WHITE),
        )

        self.wait()

        surround_rects, matrix_update_map = self.bulk_update_matrix(
            dist_matrix,
            reward_matrix,
            reward_matrix_mob,
            right_tour_edges,
            right_tour_cost,
            color=REDUCIBLE_GREEN_LIGHTER,
        )

        reward_amount = MathTex(r"+ \frac{1}{33.2}").scale(0.7)
        reward_amount.next_to(reward_matrix_label, DOWN, buff=0.5)
        reward_amount[0][-4:].set_color(REDUCIBLE_GREEN_LIGHTER),

        self.play(
            right_tour_cost_text[-4:].animate.set_color(REDUCIBLE_GREEN_LIGHTER),
            FadeIn(reward_amount),
        )

        self.play(
            *[Create(rect) for rect in surround_rects],
        )
        self.wait()

        self.play(
            *[Flash(rect, color=REDUCIBLE_GREEN_LIGHTER) for rect in surround_rects],
            *[
                Transform(reward_matrix_mob[0][index], matrix_update_map[index])
                for index in matrix_update_map
            ],
            *[FadeOut(rect) for rect in surround_rects],
        )
        self.wait()

        new_reward_matrix_label = Tex(r"$R =$").scale_to_fit_height(
            reward_matrix_label.height
        )
        new_reward_matrix_group = VGroup(
            new_reward_matrix_label, reward_matrix_mob.copy()
        ).arrange(RIGHT)
        new_reward_matrix_group.move_to(reward_matrix_group.get_center()).scale(
            0.8
        ).to_edge(LEFT * 2)

        centered_graph = left_graph.copy().scale(1.1)
        centered_graph.move_to(UP * 1.5)
        self.play(
            Transform(reward_matrix_group, new_reward_matrix_group),
            FadeOut(left_tour_cost_text),
            FadeOut(right_tour_cost_text),
            FadeOut(graphs),
            FadeOut(reward_amount),
            *[FadeOut(edge_mob) for edge_mob in left_edges_map.values()],
            *[FadeOut(edge_mob) for edge_mob in right_edges_map.values()],
            FadeIn(centered_graph),
        )
        self.wait()
        return centered_graph, reward_matrix_group, reward_matrix

    def get_surrounded_rects_around_elements_and_update_rewards(
        self,
        dist_matrix,
        reward_matrix,
        reward_matrix_mob,
        edges,
        tour_cost,
        color=REDUCIBLE_YELLOW,
    ):
        matrix_elements = reward_matrix_mob[0]
        surround_rects = []
        for edge in edges:
            one_d_index_1 = edge[0] * dist_matrix.shape[1] + edge[1]
            one_d_index_2 = edge[1] * dist_matrix.shape[1] + edge[0]
            surround_rect_1 = SurroundingRectangle(
                matrix_elements[one_d_index_1], buff=SMALL_BUFF, color=color
            )
            surround_rect_2 = SurroundingRectangle(
                matrix_elements[one_d_index_2],
                buff=SMALL_BUFF,
                color=color,
            )
            reward_matrix[edge[0]][edge[1]] += 1 / tour_cost
            reward_matrix[edge[1]][edge[0]] += 1 / tour_cost
            surround_rects.extend([surround_rect_1, surround_rect_2])
        return surround_rects

    def bulk_update_matrix(
        self,
        dist_matrix,
        reward_matrix,
        reward_matrix_mob,
        edges,
        tour_cost,
        color=REDUCIBLE_YELLOW,
    ):
        matrix_update_map = {}
        matrix_elements = reward_matrix_mob[0]
        surround_rects = []
        for edge in edges:
            one_d_index_1 = edge[0] * dist_matrix.shape[1] + edge[1]
            one_d_index_2 = edge[1] * dist_matrix.shape[1] + edge[0]
            surround_rect_1 = SurroundingRectangle(
                matrix_elements[one_d_index_1], buff=SMALL_BUFF, color=color
            )
            surround_rect_2 = SurroundingRectangle(
                matrix_elements[one_d_index_2],
                buff=SMALL_BUFF,
                color=color,
            )
            reward_matrix[edge[0]][edge[1]] += 1 / tour_cost
            reward_matrix[edge[1]][edge[0]] += 1 / tour_cost
            surround_rects.extend([surround_rect_1, surround_rect_2])
            new_reward_text_1 = Text(
                "{:.3f}".format(
                    np.round(reward_matrix[edge[0]][edge[1]], self.NUM_DECIMAL_PLACES)
                ),
                font=REDUCIBLE_MONO,
            )
            new_reward_text_1.scale_to_fit_height(
                reward_matrix_mob[0][one_d_index_1].height
            ).move_to(surround_rect_1.get_center())
            new_reward_text_2 = new_reward_text_1.copy().move_to(
                surround_rect_2.get_center()
            )
            matrix_update_map[one_d_index_1] = new_reward_text_1
            matrix_update_map[one_d_index_2] = new_reward_text_2

        return surround_rects, matrix_update_map

    def restore_all_edges_and_vertices(self, graph, tour_edges_dict):
        animations = []
        for edge in tour_edges_dict:
            animations.append(
                self.get_edge(tour_edges_dict, edge).animate.set_stroke(opacity=1)
            )

        for v in graph.vertices:
            new_node = graph.vertices[v].copy()
            new_node[0].set_fill(opacity=0.5).set_stroke(opacity=1)
            new_node[1].set_fill(opacity=1)
            animations.append(Transform(graph.vertices[v], new_node))

        return animations

    def highlight_edge(self, graph, tour_edges_dict, edge):
        new_edge_mob = self.get_edge(tour_edges_dict, edge).copy()
        new_edge_mob.set_stroke(opacity=1)
        other_edge_mobs = [
            self.get_edge(tour_edges_dict, e)
            for e in tour_edges_dict
            if e != edge and e != (edge[1], edge[0])
        ]
        new_other_edge_mobs = [
            self.get_edge(tour_edges_dict, e).copy().set_stroke(opacity=0.2)
            for e in tour_edges_dict
            if e != edge and e != (edge[1], edge[0])
        ]

        highlighted_vertices = [
            graph.vertices[edge[0]].copy(),
            graph.vertices[edge[1]].copy(),
        ]
        for v in highlighted_vertices:
            v[0].set_fill(opacity=0.5).set_stroke(opacity=1)
            v[1].set_fill(opacity=1)

        verticies_to_unhighlight = [v for v in graph.vertices if v not in edge]
        unhighlighted_vertices = [
            graph.vertices[v].copy() for v in verticies_to_unhighlight
        ]
        for v in unhighlighted_vertices:
            v[0].set_fill(opacity=0.2).set_stroke(opacity=0.2)
            v[1].set_fill(opacity=0.2)
        animations = (
            [Transform(self.get_edge(tour_edges_dict, edge), new_edge_mob)]
            + [
                Transform(old, new)
                for old, new in zip(other_edge_mobs, new_other_edge_mobs)
            ]
            + [
                Transform(graph.vertices[v], highlighted_vertices[i])
                for i, v in enumerate(edge)
            ]
            + [
                Transform(graph.vertices[v], unhighlighted_vertices[i])
                for i, v in enumerate(verticies_to_unhighlight)
            ]
        )
        return animations

    def get_next_edge(self, graph, current, valid_neighbors_vertices, reward_matrix):
        distribution = self.get_distribution(
            graph, current, valid_neighbors_vertices, reward_matrix
        )
        next_vertex = np.random.choice(
            list(distribution.keys()), p=list(distribution.values())
        )
        return (current, next_vertex)

    def demo_weighted_strategy(self, graphs):
        left_graph, right_graph = graphs
        left_start, right_start = 0, 2
        (
            left_animations,
            left_to_fade,
            left_bar_chart,
            left_circle,
            left_edges,
        ) = self.explain_strategy(left_graph, left_start)
        (
            right_animations,
            right_to_fade,
            right_bar_chart,
            right_circle,
            right_edges,
        ) = self.explain_strategy(
            right_graph, right_start, color=REDUCIBLE_GREEN_LIGHTER
        )
        for left_animation, right_animation in zip(left_animations, right_animations):
            self.play(left_animation, right_animation)
            self.wait()

        return (
            left_to_fade,
            right_to_fade,
            VGroup(left_bar_chart, right_bar_chart),
            VGroup(left_circle, right_circle),
            left_edges,
            right_edges,
        )

    def explain_strategy(self, graph, start, color=REDUCIBLE_YELLOW):
        reward_matrix = np.ones(graph.get_dist_matrix().shape)
        animations = []
        neighboring_edges = graph.get_neighboring_edges(
            start, buff=graph.vertices[0].width / 2
        )
        starting_glowing_circle = get_glowing_surround_circle(
            graph.vertices[start], color=color
        )
        animations.append(FadeIn(starting_glowing_circle))
        animations.append(LaggedStartMap(Write, neighboring_edges.values()))

        bar_chart = self.get_distribution_bar_chart(
            graph, start, get_neighbors(start, len(graph.vertices)), reward_matrix
        )
        bar_chart.next_to(graph, DOWN)
        animations.append(FadeIn(bar_chart))
        highlight_rect = SurroundingRectangle(
            VGroup(bar_chart.bars[0], bar_chart.x_axis.labels[0]),
            buff=SMALL_BUFF,
            color=REDUCIBLE_VIOLET,
        )
        animations.append(Create(highlight_rect))

        to_fade = [highlight_rect]
        return (
            animations,
            to_fade,
            bar_chart,
            starting_glowing_circle,
            neighboring_edges,
        )

    def get_distribution_bar_chart(
        self,
        graph,
        start,
        valid_neighbors_vertices,
        reward_matrix,
        x_length=4.5,
        y_length=2,
    ):
        distribution = self.get_distribution(
            graph, start, valid_neighbors_vertices, reward_matrix
        )
        bar_colors = self.get_colors(len(valid_neighbors_vertices))
        bar_chart_names = [r"$({0}, {1})$".format(start, n) for n in distribution]
        bar_chart = BarChart(
            list(distribution.values()),
            bar_names=bar_chart_names,
            bar_colors=bar_colors,
            x_length=x_length,
            y_length=y_length,
            y_range=[0, 1, 0.2],
            y_axis_config={"font_size": 24},
        )
        return bar_chart

    def get_distribution(self, graph, start, valid_neighbors_vertices, reward_matrix):
        weight_matrix = 1 / graph.get_dist_matrix() * reward_matrix
        denominator = sum([weight_matrix[start][n] for n in valid_neighbors_vertices])
        distribution = {
            n: weight_matrix[start][n] / denominator for n in valid_neighbors_vertices
        }
        return dict(sorted(distribution.items(), key=lambda x: x[1], reverse=True))

    def get_colors(self, size, max_color=REDUCIBLE_YELLOW, min_color=REDUCIBLE_PURPLE):
        colors = []
        for alpha in np.arange(0, 1, 1 / size):
            colors.append(interpolate_color(max_color, min_color, alpha))
        return colors


class AntsDemonstrationSmall4(AntColonyExplanation):
    def construct(self):
        # self.add(BACKGROUND_IMG)
        title = Text("Ant Colony Optimization", font=REDUCIBLE_FONT, weight=BOLD).scale(
            0.8
        )
        title.move_to(UP * 3.2)
        np.random.seed(2)
        NUM_VERTICES = 6
        layout = {
            0: LEFT * 4,
            1: UL * 2 + LEFT * 0.5,
            3: RIGHT * 4,
            2: UP * 2 + RIGHT * 0.2,
            4: DL * 1.8 + LEFT * 0.5,
            5: DOWN * 2 + RIGHT * 0.5,
        }
        graph = TSPGraph(list(range(NUM_VERTICES)), layout=layout)
        # self.play(FadeIn(graph), Write(title))
        # self.wait()

        green_ant = SVGMobject("green_ant").scale(0.2)

        yellow_ant = SVGMobject("yellow_ant").scale(0.2)
        purple_ant = SVGMobject("purple_ant").scale(0.2)
        yellow_purple_ant = SVGMobject("yellow_purple_ant").scale(0.2)
        ants = [green_ant, yellow_ant, purple_ant, yellow_purple_ant]

        # self.generate_path_animations_for_ant(graph, green_ant)
        # self.generate_path_animations_for_ant(graph, yellow_ant, color=REDUCIBLE_YELLOW)
        # self.generate_path_animations_for_ant(
        #     graph, purple_ant, color=REDUCIBLE_VIOLET, seed=1
        # )
        self.generate_path_animations_for_ant(
            graph, purple_ant, color=REDUCIBLE_PURPLE, seed=3
        )

    def generate_path_animations_for_ant(
        self, graph, ant, seed=0, color=REDUCIBLE_GREEN_LIGHTER
    ):
        np.random.seed(seed)
        tour = self.get_random_tour(len(graph.vertices))
        print(tour)
        tour_edges = get_edges_from_tour(tour)
        tour_edges_mob = self.get_tsp_tour_edges_mob(graph, tour_edges)
        dashed_lines_mob = VGroup(
            *[
                DashedLine(line.get_start(), line.get_end()).set_stroke(
                    color=color, opacity=0.5
                )
                for line in tour_edges_mob
            ]
        )
        starting_edge = tour_edges[0]
        ant, current_orientation = self.get_ant_orientation_angle(
            graph, starting_edge, ant
        )
        self.play(FadeIn(ant))
        self.wait()

        for i, tour_edge in enumerate(tour_edges):
            next_vertex = tour_edge[1]

            self.play(
                ant.animate.move_to(graph.vertices[next_vertex].get_center()),
                Create(dashed_lines_mob[i]),
                rate_func=linear,
            )
            if i == len(tour_edges) - 1:
                break
            next_edge = tour_edges[i + 1]
            rotated_ant, new_orientation = self.get_ant_orientation_angle(
                graph, next_edge, ant, current_orientation=current_orientation
            )
            current_orientation = new_orientation
            self.play(Transform(ant, rotated_ant))
        self.wait()
        self.play(FadeOut(ant))

    def get_ant_orientation_angle(
        self, graph, edge, ant, current_orientation=90, animate=False
    ):
        u, v = edge
        unit_v = normalize(
            graph.vertices[v].get_center() - graph.vertices[u].get_center()
        )
        orientation_angle_in_degrees = self.get_orientation_angle_in_degrees(unit_v)
        print(edge, orientation_angle_in_degrees)
        counter_clockwise_rotation_in_degrees = (
            orientation_angle_in_degrees - current_orientation
        )
        print(
            "Counter clockwise rotation",
            counter_clockwise_rotation_in_degrees,
            "degrees",
        )
        counter_clockwise_rotation_in_radians = (
            counter_clockwise_rotation_in_degrees / 180 * np.pi
        )

        new_ant = ant.copy().rotate(counter_clockwise_rotation_in_radians)
        new_ant.move_to(graph.vertices[edge[0]].get_center())
        return new_ant, orientation_angle_in_degrees

    def get_random_tour(self, N):
        tour = [np.random.choice(list(range(N)))]
        current = tour[0]
        while len(tour) < N:
            remaining_vertices = [v for v in get_neighbors(current, N) if v not in tour]
            current = np.random.choice(remaining_vertices)
            tour.append(current)
        return tour

    def get_orientation_angle_in_degrees(self, vec):
        y, x = vec[1], vec[0]
        rad = np.arctan2(y, x)
        degrees = rad * 180 / np.pi
        if degrees < 0:
            degrees = 360 + degrees
        return degrees


class AntColonyOptimizationSteps(Scene):
    def construct(self):
        text_scale = 0.7
        step_1 = Tex(r"1. Initialize $N$ ants and $R$ (matrix of 1's)").scale(
            text_scale
        )
        step_2 = Tex(
            r"2. $P(u, v) = \frac{D(u, v)^{-1} \cdot R(u, v)}{\sum_{j \in \text{adj}(u)} D(u, j)^{-1} \cdot R(u, j)}$"
        ).scale(text_scale + SMALL_BUFF)
        step_3 = Tex(r"3. Each ant generates a tour using probabilities").scale(
            text_scale
        )
        step_4 = Tex(
            r"3. $R(u, v) \leftarrow R(u, v) + \sum_{k=1}^{N} \frac{1}{C_k}$ if ant $k$ used edge $(u, v)$ in tour with cost $C_k$"
        ).scale(text_scale)
        step_5 = Tex(
            r"4. Repeat (2) $-$ (3) and keep track of best tour cost $C^*$"
        ).scale(text_scale)

        steps = VGroup(step_1, step_2, step_3, step_4, step_5)
        steps.scale(0.9).to_edge(DOWN * 1)

        for step in steps:
            step.to_edge(DOWN * 1.7)

        self.play(FadeIn(step_1, shift=UP * 0.5))
        self.wait()

        self.play(FadeTransform(step_1, step_2))
        self.wait()

        updated_step_2 = (
            Tex(
                r"2. $P(u, v) = \frac{[ D(u, v)^{-1} ] ^{\alpha} \cdot [ R(u, v) ]^{\beta}}{\sum_{j \in \text{adj}(u)} [ D(u, j)^{-1} ] ^{\alpha} \cdot [ R(u, j) ] ^{\beta}}$"
            )
            .scale(text_scale + SMALL_BUFF)
            .scale(0.9)
        )
        updated_step_2.move_to(step_2.get_center() + LEFT * 2)
        alpha_beta_explantion = (
            Tex(r"($\alpha$ and $\beta$ are parameters we set)")
            .scale_to_fit_height(step_1.height)
            .next_to(updated_step_2, RIGHT)
        )
        self.play(FadeTransform(step_2, updated_step_2), FadeIn(alpha_beta_explantion))
        self.wait()

        self.play(FadeOut(updated_step_2), FadeOut(alpha_beta_explantion))
        self.play(FadeIn(step_4, shift=UP * 0.5))
        self.wait()

        updated_step_4 = (
            Tex(
                r"3. $R(u, v) \leftarrow (1 - \rho) \cdot R(u, v) + \sum_{k=1}^{N} \frac{1}{C_k}$ if ant $k$ used edge $(u, v)$ in tour"
            )
            .scale(text_scale)
            .scale(0.9)
        )

        updated_step_4.move_to(step_4.get_center() + LEFT * SMALL_BUFF * 3.5)

        rho_explanation = (
            MathTex(r"(\rho \in [0, 1])")
            .scale_to_fit_height(step_1.height)
            .next_to(updated_step_4, RIGHT)
        )

        self.play(FadeTransform(step_4, updated_step_4), FadeIn(rho_explanation))
        self.wait()

        updated_step_2.next_to(updated_step_4, UP, aligned_edge=LEFT)
        step_5.next_to(updated_step_4, DOWN, aligned_edge=LEFT)

        self.play(FadeIn(updated_step_2), FadeIn(step_5, shift=UP * 0.5))
        self.wait()


class AntSimulation(TourImprovement):
    def construct(self):
        # self.add(BACKGROUND_IMG)
        NUM_VERTICES = 12
        layout = self.get_random_layout(NUM_VERTICES)
        tsp_graph = TSPGraph(list(range(NUM_VERTICES)), layout=layout).scale(0.9)
        self.add_foreground_mobject(tsp_graph)
        self.play(FadeIn(tsp_graph))
        self.wait()
        alpha = 1
        beta = 3
        evaporation_rate = 0.2
        elitist_weight = 1
        NUM_STEPS_ACO = 50
        NUM_ANTS = 100
        ant_colony_sim = AntColonySimulation(
            tsp_graph,
            NUM_ANTS,
            alpha=alpha,
            beta=beta,
            evaporation_rate=evaporation_rate,
            elitist_weight=elitist_weight,
        )
        all_edges = None
        tour_edges_mob = None
        for step in range(NUM_STEPS_ACO):
            print(f"***Iteration {step}***")
            print(
                f"*Best tour: {ant_colony_sim.best_tour} -- cost: {ant_colony_sim.best_tour_cost}*"
            )
            if ant_colony_sim.best_tour is not None:
                tour_edges = get_edges_from_tour(ant_colony_sim.best_tour)
                tour_edges_mob = self.get_tsp_tour_edges_mob(
                    tsp_graph,
                    tour_edges,
                    color=REDUCIBLE_GREEN_LIGHTER,
                    stroke_width=5,
                )
                self.add(tour_edges_mob)
            all_tours = ant_colony_sim.update(elitist=True)
            all_ant_animations = self.get_all_ant_movements(
                ant_colony_sim.ants, all_tours, tsp_graph
            )
            if step == 0:
                all_edges = self.highlight_edges_based_on_pheromone_weight(
                    tsp_graph, ant_colony_sim.pheromone_matrix
                )
                self.play(FadeIn(all_edges))
                self.wait()
                self.play(*all_ant_animations, run_time=2, rate_func=linear)

            else:
                new_edges = self.highlight_edges_based_on_pheromone_weight(
                    tsp_graph, ant_colony_sim.pheromone_matrix
                )
                self.play(
                    *all_ant_animations + [all_edges.animate.become(new_edges)],
                    run_time=3,
                    rate_func=linear,
                )
            self.remove(tour_edges_mob)

        optimal_tour, optimal_cost = get_exact_tsp_solution(tsp_graph.get_dist_matrix())

        optimal_edges = get_edges_from_tour(optimal_tour)
        # optimal_tour_edges_mob = self.get_tsp_tour_edges_mob(
        #     tsp_graph, optimal_edges, color=REDUCIBLE_YELLOW, stroke_width=6
        # )
        # print(optimal_tour, optimal_cost)
        # self.add(optimal_tour_edges_mob)
        # self.wait()

    def get_weight_matrix(self, ant_colony_sim):
        cost_matrix = ant_colony_sim.cost_matrix
        pheromone_matrix = ant_colony_sim.pheromone_matrix
        alpha = ant_colony_sim.alpha
        beta = ant_colony_sim.beta
        weight_matrix = np.ones(cost_matrix.shape)
        for i in range(cost_matrix.shape[0]):
            for j in range(cost_matrix.shape[1]):
                weight_matrix[i][j] = (
                    pheromone_matrix[i][j] ** alpha * cost_matrix[i][j] ** beta
                )
        return weight_matrix

    def highlight_edges_based_on_pheromone_weight(self, tsp_graph, pheromone_matrix):
        all_edges = tsp_graph.get_all_edges()
        for edge in all_edges:
            u, v = edge
            opacity = pheromone_matrix[u][v] / np.max(pheromone_matrix)
            all_edges[edge].set_stroke(color=REDUCIBLE_VIOLET, opacity=opacity)
        return VGroup(*all_edges.values())

    def get_all_ant_movements(self, ants, all_tours, tsp_graph):
        all_animations = []
        for ant, tour in zip(ants, all_tours):
            all_animations.append(self.get_tour_animations(ant, tour, tsp_graph))
        return all_animations

    def get_tour_animations(self, ant, tour, tsp_graph):
        path = VGroup()
        tour_centers = [tsp_graph.vertices[v].get_center() for v in tour]
        tour_centers.append(tour_centers[0])
        path.set_points_as_corners(*[tour_centers])
        return MoveAlongPath(ant.get_mob(), path)


class IntroPIP(Scene):
    def construct(self):
        self.add(BACKGROUND_IMG)
        screen_rect = ScreenRectangle(height=4).shift(UP * 0.5)
        creative_approach = Text(
            "Creative approaches to solving the TSP", font=REDUCIBLE_FONT
        )
        creative_approach.scale(0.7)
        creative_approach.next_to(screen_rect, DOWN, buff=0.5)

        self.play(FadeIn(screen_rect), FadeIn(creative_approach))
        self.wait()

        heuristic_search = Text(
            "Heuristic Search and Approximation Algorithms", font=REDUCIBLE_FONT
        )

        heuristic_search.scale(0.7)

        heuristic_search.move_to(creative_approach.get_center())

        self.play(FadeTransform(creative_approach, heuristic_search))
        self.wait()


class GeneralPip(Scene):
    def construct(self):
        self.add(BACKGROUND_IMG)
        screen_rect = ScreenRectangle(height=4)
        self.add(screen_rect)
        self.wait(10)


class TSPTourDefinition(LocalMinima):
    def construct(self):
        self.add(BACKGROUND_IMG)
        NUM_VERTICES = 6
        layout = self.get_random_layout(NUM_VERTICES)
        tsp_graph = TSPGraph(list(range(NUM_VERTICES))).shift(UP * 0.5)
        all_tours = get_all_tour_permutations(NUM_VERTICES, 0)
        tour_1_index = np.random.choice(list(range(len(all_tours))))
        tour_2_index = np.random.choice(list(range(len(all_tours))))
        tour_1_edges = get_edges_from_tour(
            all_tours[tour_1_index],
        )
        tour_2_edges = get_edges_from_tour(
            all_tours[tour_2_index],
        )
        tour_def = Tex(
            r"TSP Tour: starts at node $v$, visits all nodes once, and returns to $v$"
        ).scale(0.7)
        tour_def.next_to(tsp_graph, DOWN, buff=0.5)
        self.play(FadeIn(tsp_graph), FadeIn(tour_def))
        self.wait()

        tour_1_edges_mob = self.get_tsp_tour_edges_mob(tsp_graph, tour_1_edges)
        tour_2_edges_mob = self.get_tsp_tour_edges_mob(tsp_graph, tour_2_edges)

        for edge_mob in tour_1_edges_mob:
            self.play(Create(edge_mob), rate_func=linear)

        self.play(FadeOut(tour_1_edges_mob))
        self.wait()

        for edge_mob in tour_2_edges_mob:
            self.play(Create(edge_mob), rate_func=linear)
        self.wait()


class BackgroundImage(Scene):
    def construct(self):
        self.add(BACKGROUND_IMG)
        self.wait(5)


class SimulatedAnnealingIntro(TSPTourDefinition):
    def construct(self):
        self.add(BACKGROUND_IMG)
        self.show_two_opt_switches()

    def show_two_opt_switches(self):
        np.random.seed(21)
        NUM_VERTICES = 10

        layout = self.get_random_layout(NUM_VERTICES)
        layout[3] += LEFT * 0.8
        layout[5] += DOWN * 0.8
        graph = (
            TSPGraph(list(range(NUM_VERTICES)), layout=layout)
            .scale(0.7)
            .shift(UP * 0.3 + RIGHT * 0.5)
        )
        all_edges = graph.get_all_edges(buff=graph.vertices[0].width / 2)

        nn_tour, nn_cost = get_nearest_neighbor_solution(graph.get_dist_matrix())

        nn_tour_edges = get_edges_from_tour(nn_tour)
        current_tour_mob = self.get_tsp_tour_edges_mob(graph, nn_tour_edges)
        tour_edges_to_line_map = self.get_tsp_tour_edges_map(graph, nn_tour_edges)

        self.play(FadeIn(current_tour_mob), FadeIn(graph))

        cost = (
            Text(f"Nearest Neighbor Cost: {np.round(nn_cost, 2)}", font=REDUCIBLE_MONO)
            .scale(0.4)
            .to_edge(DOWN * 2)
        )
        self.play(FadeIn(cost))
        self.wait()

        current_cost = nn_cost
        current_tour = nn_tour
        current_tour_edges = nn_tour_edges
        current_tour_map = tour_edges_to_line_map
        improvement_cost = (
            Text(f"2-opt switch cost: {np.round(nn_cost, 2)}", font=REDUCIBLE_MONO)
            .scale(0.4)
            .next_to(cost, DOWN)
        )
        self.play(FadeIn(improvement_cost))
        self.wait()

        big_idea = (
            Text(
                "Big idea: probabilistically accept worse tours",
                font=REDUCIBLE_FONT,
                weight=BOLD,
            )
            .scale(0.6)
            .move_to(UP * 3.3)
        )

        self.play(FadeIn(big_idea))
        self.wait()

        for i in range(len(current_tour_edges) - 1):
            for j in range(i + 1, len(current_tour_edges)):
                e1, e2 = current_tour_edges[i], current_tour_edges[j]
                new_e1, new_e2, new_tour = get_two_opt_new_edges(current_tour, e1, e2)
                if new_e1 != e1 and new_e2 != e2:
                    new_tour_edges = get_edges_from_tour(new_tour)
                    new_tour_cost = get_cost_from_edges(
                        new_tour_edges, graph.get_dist_matrix()
                    )
                    cost_text_color = REDUCIBLE_CHARM
                    if new_tour_cost < nn_cost:
                        cost_text_color = REDUCIBLE_GREEN_LIGHTER

                    new_tour_edges_mob = self.get_tsp_tour_edges_mob(
                        graph, new_tour_edges
                    )
                    new_improvement_cost = (
                        Text(
                            f"2-opt switch cost: {np.round(new_tour_cost, 1)}",
                            font=REDUCIBLE_MONO,
                        )
                        .scale(0.4)
                        .move_to(improvement_cost.get_center())
                    )
                    new_improvement_cost[-4:].set_color(cost_text_color)

                    new_graph_with_edges_mob = self.get_tsp_tour_edges_mob(
                        graph, new_tour_edges
                    )
                    self.remove(current_tour_mob, improvement_cost)
                    self.add(new_graph_with_edges_mob, new_improvement_cost)
                    self.wait()
                    current_tour_mob = new_graph_with_edges_mob
                    improvement_cost = new_improvement_cost

        # optimal_tour, optimal_cost = get_exact_tsp_solution(graph.get_dist_matrix())
        # print("Optimal tour and cost", optimal_tour, optimal_cost)

        # three_opt_edges = [(3, 6), (8, 7), (7, 0)]
        # new_three_opt_edges = [(3, 7), (7, 6), (8, 0)]
        # three_opt_v = [3, 6, 8, 7, 0]

        # new_title = (
        #     Text("3-opt Improvement", font=REDUCIBLE_FONT, weight=BOLD)
        #     .scale(0.7)
        #     .move_to(UP * 3.3)
        # )

        # s_circles = [
        #     get_glowing_surround_circle(graph.vertices[v]) for v in three_opt_v
        # ]
        # self.play(
        #     *[FadeIn(c) for c in s_circles],
        #     *[
        #         self.get_edge(current_tour_map, edge).animate.set_color(
        #             REDUCIBLE_YELLOW
        #         )
        #         for edge in three_opt_edges
        #     ],
        #     Transform(title, new_title),
        # )
        # self.wait()

        # new_improvement_cost = (
        #     Text(
        #         f"After 3-opt Improvement: {np.round(optimal_cost, 1)}",
        #         font=REDUCIBLE_MONO,
        #     )
        #     .scale(0.4)
        #     .move_to(improvement_cost.get_center())
        # )

        # self.play(
        #     *[
        #         Transform(
        #             self.get_edge(current_tour_map, e1),
        #             self.make_edge(
        #                 graph.vertices[e2[0]],
        #                 graph.vertices[e2[1]],
        #                 color=REDUCIBLE_VIOLET,
        #             ),
        #         )
        #         for e1, e2 in zip(three_opt_edges, new_three_opt_edges)
        #     ],
        #     Transform(improvement_cost, new_improvement_cost),
        # )
        # self.wait()

        # self.play(
        #     *[FadeOut(c) for c in s_circles],
        # )
        # self.wait()

    def get_new_edge_map(self, prev_edge_map, new_tour_edges, graph, all_edges):
        set_new_tour_edges = set(new_tour_edges)
        new_edge_map = {}
        remaining_edges = set()
        for edge in prev_edge_map:
            u, v = edge
            if (u, v) in set_new_tour_edges:
                new_edge_map[(u, v)] = prev_edge_map[(u, v)]
            elif (v, u) in set_new_tour_edges:
                new_edge_map[(v, u)] = prev_edge_map[(u, v)]
            else:
                remaining_edges.add(edge)
        new_edges = set()
        for edge in new_tour_edges:
            u, v = edge
            if (u, v) in new_edge_map or (v, u) in new_edge_map:
                continue
            new_edges.add(edge)
            new_edge_map[(u, v)] = self.make_edge(
                graph.vertices[u], graph.vertices[v], color=REDUCIBLE_VIOLET
            )
        all_edge_corresponding = []
        for u, v in remaining_edges:
            if (u, v) in all_edges:
                all_edge_corresponding.append((u, v))
            else:
                all_edge_corresponding.append((v, u))
        print("Remaining edges", remaining_edges)
        print("Edges in all_edges map", all_edge_corresponding)
        print("New edges", new_edges)
        return new_edge_map


class Conclusion(MovingCameraScene):
    def construct(self):
        start_width, start_height = 12, 7
        start_text_scale = 0.9
        scale_factor = 3
        tsp_module = Module(
            "Traveling Salesman Problem",
            text_scale=start_text_scale,
            text_position=UP,
            width=start_width,
            height=start_height,
            text_weight=BOLD,
        )
        approaches = Module(
            ["Algorithmic", "Approaches"],
            text_scale=0.9,
            height=start_height / (scale_factor * 1),
            width=start_width / (scale_factor * 1),
            stroke_color=REDUCIBLE_YELLOW,
            fill_color=REDUCIBLE_YELLOW_DARKER,
            text_position=ORIGIN,
            text_weight=BOLD,
            stroke_width=3,
        ).move_to(tsp_module.get_center())

        self.add_foreground_mobjects(tsp_module, approaches)
        self.play(FadeIn(tsp_module), FadeIn(approaches))
        self.wait()

        general_approach = Module(
            "Difficult Problems",
            text_scale=start_text_scale,
            text_position=UP,
            width=22,
            height=14,
            text_weight=BOLD,
            stroke_color=REDUCIBLE_GREEN_LIGHTER,
            fill_color=REDUCIBLE_GREEN_DARKER,
        )

        self.play(
            FadeIn(general_approach),
            self.focus_on(general_approach, buff=1.5),
            run_time=3,
        )
        self.wait()

    def focus_on(self, mobject, buff=2):
        return self.camera.frame.animate.set_width(mobject.width * buff).move_to(
            mobject
        )


class ConclusionFlow(Scene):
    def construct(self):
        self.add(BACKGROUND_IMG)
        title = (
            Text("Solving Intractable Problems", font=REDUCIBLE_FONT, weight=BOLD)
            .scale(0.7)
            .move_to(UP * 3.3)
        )

        self.play(Write(title))
        self.wait()

        greedy = Module(["Greedy", "Approximation"], text_weight=BOLD)

        simpler = Module(
            ["Simpler", "Problem"],
            REDUCIBLE_GREEN_DARKER,
            REDUCIBLE_GREEN_LIGHTER,
            text_weight=BOLD,
        )
        simpler.text.scale(0.8)

        local_search = Module(
            ["Local", "Search"],
            REDUCIBLE_YELLOW_DARKER,
            REDUCIBLE_YELLOW,
            text_weight=BOLD,
        )
        local_search.text.scale(0.7)

        explore = Module(
            ["Explore vs", "Exploit"],
            REDUCIBLE_WARM_BLUE_DARKER,
            REDUCIBLE_WARM_BLUE,
            text_weight=BOLD,
        )
        explore.text.scale(0.9)

        screen_rect = ScreenRectangle(height=3)

        screen_rect.move_to(UP * 1.2)

        modules = (
            VGroup(greedy, simpler, local_search, explore)
            .scale(0.6)
            .arrange(RIGHT, buff=0.8)
        ).move_to(DOWN * 1.5)

        text_scale = 0.5

        nearest_neighbor = Text("Nearest Neighbor", font=REDUCIBLE_FONT).scale(
            text_scale
        )
        greedy_heuristic = Text("Greedy Heuristic", font=REDUCIBLE_FONT).scale(
            text_scale
        )

        VGroup(nearest_neighbor, greedy_heuristic).arrange(DOWN).next_to(greedy, DOWN)

        christofides_text = Text("Christofides", font=REDUCIBLE_FONT).scale(text_scale)
        christofides_text.next_to(simpler, DOWN)

        random_swaps = Text("Random Swaps", font=REDUCIBLE_FONT).scale(text_scale)
        two_opt_text = Text("2-opt", font=REDUCIBLE_FONT).scale(text_scale)
        three_opt_text = Text("3-opt", font=REDUCIBLE_FONT).scale(text_scale)

        VGroup(two_opt_text, three_opt_text, random_swaps).arrange(DOWN).next_to(
            local_search, DOWN
        )

        simulated_ann_text = Text(
            "Simulated Annealing", font=REDUCIBLE_FONT
        ).scale_to_fit_width(explore.width - SMALL_BUFF)
        aco_text = Text(
            "Ant Colony Optimization", font=REDUCIBLE_FONT
        ).scale_to_fit_width(explore.width)
        VGroup(simulated_ann_text, aco_text).arrange(DOWN).next_to(explore, DOWN)

        self.play(FadeIn(screen_rect))
        self.wait()

        self.play(FadeIn(greedy))
        self.wait()

        self.play(
            FadeIn(nearest_neighbor, shift=UP * 0.5),
            FadeIn(greedy_heuristic, shift=UP * 0.5),
        )

        self.wait()

        self.play(
            FadeIn(simpler),
            FadeIn(christofides_text),
        )
        self.wait()

        self.play(
            FadeIn(local_search),
        )
        self.wait()

        self.play(
            FadeIn(random_swaps, shift=UP * 0.5),
            FadeIn(two_opt_text, shift=UP * 0.5),
            FadeIn(three_opt_text, shift=UP * 0.5),
        )
        self.wait()

        self.play(FadeIn(explore))
        self.wait()

        self.play(
            FadeIn(simulated_ann_text, shift=UP * 0.5),
        )
        self.wait()

        self.play(
            FadeIn(aco_text, shift=UP * 0.5),
        )
        self.wait()


class Patreons(Scene):
    def construct(self):
        self.add(BACKGROUND_IMG)
        thanks = Tex("Special Thanks to These Patreons").scale(1.2)
        patreons = [
            "Burt Humburg",
            "Winston Durand",
            r"Adam D\v{r}ínek",
            "Kerrytazi",
            "Andreas",
            "Matt Q",
            "Brian Cloutier",
            "Ram K",
        ]
        patreon_text = VGroup(
            thanks,
            VGroup(*[Tex(name).scale(0.9) for name in patreons]).arrange_in_grid(
                rows=2, buff=(0.65, 0.25)
            ),
        )
        patreon_text.arrange(DOWN)
        patreon_text.to_edge(DOWN)

        self.play(Write(patreon_text[0]))
        self.play(*[Write(text) for text in patreon_text[1]])
        self.wait()


class CuriosityStream(Scene):
    def construct(self):
        link = Text(
            "Visit https://curiositystream.com/reducible", font=REDUCIBLE_FONT
        ).scale(0.7)
        link[5:].set_color(REDUCIBLE_YELLOW)
        code = Text(
            "Use code reducible to get an entire YEAR at $14.99", font=REDUCIBLE_FONT
        ).scale(0.6)
        code[7:16].set_color(REDUCIBLE_YELLOW)

        link.move_to(DOWN * 2.5)
        code.next_to(link, DOWN)

        self.play(Write(link))
        self.play(Write(code))
        self.wait()

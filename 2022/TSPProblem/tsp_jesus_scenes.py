from pprint import pprint
from typing import Iterable
from manim import *
from solving_tsp import TSPGraph
from reducible_colors import *
from functions import *
from classes import *
from math import factorial, log10
from scipy.special import gamma
from solver_utils import *
from manim.mobject.geometry.tips import ArrowTriangleTip
from itertools import combinations, permutations

np.random.seed(2)
config["assets_dir"] = "assets"

BACKGROUND_IMG = ImageMobject("bg-75.png").scale_to_fit_width(config.frame_width)


class TSPAssumptions(MovingCameraScene):
    def construct(self):

        BACKGROUND_IMG.add_updater(lambda mob: mob.move_to(self.camera.frame_center))
        self.add(BACKGROUND_IMG)

        self.intro_PIP()
        self.clear()

        self.add(BACKGROUND_IMG)
        self.present_TSP_graph()
        self.wait()
        self.play(*[FadeOut(mob) for mob in self.mobjects])

    def intro_PIP(self):
        rect = ScreenRectangle(height=4)
        self.play(Write(rect), run_time=2)
        self.wait()

    def present_TSP_graph(self):
        frame = self.camera.frame

        graph = TSPGraph(range(8))

        self.play(Write(graph))
        self.wait()

        all_edges = graph.get_all_edges()

        self.play(LaggedStartMap(Write, all_edges.values()))
        self.wait()

        edge_focus = self.focus_on_edges(
            edges_to_focus_on=[(2, 7)], all_edges=all_edges
        )
        self.play(*edge_focus)

        arrow_up = (
            Arrow(
                start=graph.vertices[2],
                end=graph.vertices[7],
                max_tip_length_to_length_ratio=0.041,
            )
            .move_to(all_edges[(2, 7)].start)
            .shift(DOWN * 0.4)
            .scale(0.2)
            .set_color(WHITE)
        )
        arrow_down = (
            Arrow(
                start=graph.vertices[7],
                end=graph.vertices[2],
                max_tip_length_to_length_ratio=0.041,
            )
            .move_to(all_edges[(2, 7)].end)
            .shift(UP * 0.4)
            .scale(0.2)
            .set_color(WHITE)
        )
        dist_label = graph.get_dist_label(all_edges[(2, 7)], graph.dist_matrix[2, 7])
        self.play(FadeIn(arrow_up, arrow_down, dist_label))
        self.play(
            arrow_up.animate.move_to(all_edges[(2, 7)].end).shift(DOWN * 0.4),
            arrow_down.animate.move_to(all_edges[(2, 7)].start).shift(UP * 0.4),
            ShowPassingFlash(
                all_edges[(2, 7)].copy().set_stroke(width=6).set_color(REDUCIBLE_YELLOW)
            ),
            ShowPassingFlash(
                all_edges[(2, 7)]
                .copy()
                .set_stroke(width=6)
                .flip(RIGHT)
                .flip(DOWN)
                .set_color(REDUCIBLE_YELLOW)
            ),
        )

        self.play(FadeOut(arrow_up), FadeOut(arrow_down))
        self.play(
            graph.animate.shift(DOWN),
            dist_label.animate.shift(DOWN),
            *[l.animate.shift(DOWN) for l in all_edges.values()],
        )
        title = (
            Text(
                "Symmetric Traveling Salesman Problem",
                font=REDUCIBLE_FONT,
                weight=BOLD,
            )
            .scale(0.8)
            .to_edge(UP, buff=1)
        )
        self.play(LaggedStartMap(FadeIn, title))
        self.wait()

        self.play(
            FadeOut(title),
        )
        self.wait()

        full_graph = VGroup(*graph.vertices.values(), *all_edges.values(), dist_label)

        self.play(
            FadeOut(dist_label), full_graph.animate.move_to(LEFT * 3), run_time=0.7
        )
        self.wait()

        # triangle inequality
        all_labels = {
            t: graph.get_dist_label(e, graph.dist_matrix[t]).set_opacity(0)
            for t, e in all_edges.items()
        }

        for i in range(8):
            if i == 5:
                triang_title = (
                    Text("Triangle Inequality", font=REDUCIBLE_FONT, weight=BOLD)
                    .scale(0.8)
                    .move_to(frame.get_top())
                )
                self.play(FadeIn(triang_title), frame.animate.shift(UP * 0.8))
            triang_vertices = sorted(
                np.random.choice(list(graph.vertices.keys()), size=3, replace=False)
            )
            start_node = triang_vertices[0]
            middle_node = triang_vertices[1]
            end_node = triang_vertices[2]

            triangle_edges = [
                (start_node, end_node),
                (start_node, middle_node),
                (middle_node, end_node),
            ]
            triangle_ineq_edges_focus = self.focus_on_edges(triangle_edges, all_edges)
            labels_to_focus = self.focus_on_labels(triangle_edges, all_labels)
            vertices_to_focus = self.focus_on_vertices(triang_vertices, graph.vertices)

            less_than = Text(
                "always less than", font=REDUCIBLE_FONT, weight=BOLD
            ).scale(0.7)

            arrow_text = Text("→", font=REDUCIBLE_FONT, weight=BOLD)

            direct_path = VGroup(
                graph.vertices[start_node].copy().set_stroke(REDUCIBLE_PURPLE),
                arrow_text.copy(),
                graph.vertices[end_node].copy().set_stroke(REDUCIBLE_PURPLE),
            ).arrange(RIGHT, buff=0.2)

            indirect_path = VGroup(
                graph.vertices[start_node].copy().set_stroke(REDUCIBLE_PURPLE),
                arrow_text.copy(),
                graph.vertices[middle_node].copy().set_stroke(REDUCIBLE_PURPLE),
                arrow_text.copy(),
                graph.vertices[end_node].copy().set_stroke(REDUCIBLE_PURPLE),
            ).arrange(RIGHT, buff=0.2)

            both_paths = (
                VGroup(direct_path, less_than, indirect_path)
                .arrange(DOWN, buff=0.5)
                .shift(RIGHT * 3)
            )

            if i == 0:
                self.play(
                    *labels_to_focus,
                    *triangle_ineq_edges_focus,
                    *vertices_to_focus,
                    FadeIn(direct_path),
                    FadeIn(less_than),
                    FadeIn(indirect_path),
                )
            else:
                self.play(
                    *labels_to_focus,
                    *triangle_ineq_edges_focus,
                    *vertices_to_focus,
                    FadeTransform(last_direct_path, direct_path),
                    FadeTransform(last_indirect_path, indirect_path),
                )

            self.play(
                Succession(
                    AnimationGroup(
                        ShowPassingFlash(
                            all_edges[(start_node, end_node)]
                            .copy()
                            .set_stroke(width=6)
                            .set_color(REDUCIBLE_YELLOW),
                            time_width=0.4,
                        ),
                    ),
                    AnimationGroup(
                        Succession(
                            ShowPassingFlash(
                                all_edges[(start_node, middle_node)]
                                .copy()
                                .set_stroke(width=6)
                                .set_color(REDUCIBLE_YELLOW),
                                time_width=0.6,
                            ),
                            ShowPassingFlash(
                                all_edges[(middle_node, end_node)]
                                .copy()
                                .set_stroke(width=6)
                                .set_color(REDUCIBLE_YELLOW),
                                time_width=0.6,
                            ),
                        ),
                    ),
                ),
            )

            last_direct_path = direct_path
            last_indirect_path = indirect_path
        # end for loop triangle inequality

    ### UTIL FUNCS

    def focus_on_edges(
        self, edges_to_focus_on: Iterable[tuple], all_edges: Iterable[tuple]
    ):
        edges_animations = []

        edges_to_focus_on = list(
            map(lambda t: (t[1], t[0]) if t[0] > t[1] else t, edges_to_focus_on)
        )
        for t, e in all_edges.items():
            if not t in edges_to_focus_on:
                edges_animations.append(e.animate.set_opacity(0.3))
            else:
                edges_animations.append(e.animate.set_opacity(1))

        return edges_animations

    def focus_on_vertices(
        self, vertices_to_focus_on: Iterable[int], all_vertices: Iterable[tuple]
    ):
        edges_animations = []
        for t, e in all_vertices.items():
            if not t in vertices_to_focus_on:
                edges_animations.append(e.animate.set_stroke(REDUCIBLE_PURPLE))
            else:
                edges_animations.append(e.animate.set_stroke(REDUCIBLE_YELLOW))

        return edges_animations

    def focus_on_labels(self, labels_to_show, all_labels):
        labels_animations = []

        labels_to_show = list(
            map(lambda t: (t[1], t[0]) if t[0] > t[1] else t, labels_to_show)
        )
        for t, e in all_labels.items():
            if not t in labels_to_show:
                labels_animations.append(e.animate.set_opacity(0))
            else:
                labels_animations.append(e.animate.set_opacity(1))

        return labels_animations

    def show_triangle_inequality(
        self,
        i,
        graph,
        all_edges,
        all_labels,
    ):
        triang_vertices = sorted(
            np.random.choice(list(graph.vertices.keys()), size=3, replace=False)
        )
        start_node = triang_vertices[0]
        middle_node = triang_vertices[1]
        end_node = triang_vertices[2]

        triangle_edges = [
            (start_node, end_node),
            (start_node, middle_node),
            (middle_node, end_node),
        ]
        triangle_ineq_edges_focus = self.focus_on_edges(
            edges_to_focus_on=triangle_edges, all_edges=all_edges
        )
        labels_to_focus = self.show_labels(
            labels_to_show=triangle_edges, all_labels=all_labels
        )
        all_labels = [
            graph.get_dist_label(all_edges[e], graph.dist_matrix[e])
            for e in triangle_edges
        ]

        less_than = Text("always less than", font=REDUCIBLE_FONT, weight=BOLD).scale(
            0.7
        )

        arrow_text = Text("→", font=REDUCIBLE_FONT, weight=BOLD)

        direct_path = VGroup(
            graph.vertices[start_node].copy(),
            arrow_text.copy(),
            graph.vertices[end_node].copy(),
        ).arrange(RIGHT, buff=0.2)

        indirect_path = VGroup(
            graph.vertices[start_node].copy(),
            arrow_text.copy(),
            graph.vertices[middle_node].copy(),
            arrow_text.copy(),
            graph.vertices[end_node].copy(),
        ).arrange(RIGHT, buff=0.2)

        both_paths = (
            VGroup(direct_path, less_than, indirect_path)
            .arrange(DOWN, buff=0.5)
            .shift(RIGHT * 3)
        )

        if i > 0:
            last_direct_path = direct_path
            last_indirect_path = indirect_path

        print(i)
        if i == 0:
            self.play(
                *labels_to_focus,
                *triangle_ineq_edges_focus,
                FadeIn(direct_path),
                FadeIn(indirect_path),
            )
        else:

            self.play(
                *labels_to_focus,
                *triangle_ineq_edges_focus,
                Transform(last_direct_path, direct_path),
                Transform(last_indirect_path, indirect_path),
            )

        self.play(
            Succession(
                AnimationGroup(
                    ShowPassingFlash(
                        all_edges[(start_node, end_node)]
                        .copy()
                        .set_stroke(width=6)
                        .set_color(REDUCIBLE_YELLOW),
                        time_width=0.4,
                    ),
                ),
                FadeIn(less_than),
                AnimationGroup(
                    Succession(
                        ShowPassingFlash(
                            all_edges[(start_node, middle_node)]
                            .copy()
                            .set_stroke(width=6)
                            .set_color(REDUCIBLE_YELLOW),
                            time_width=0.6,
                        ),
                        ShowPassingFlash(
                            all_edges[(middle_node, end_node)]
                            .copy()
                            .set_stroke(width=6)
                            .set_color(REDUCIBLE_YELLOW),
                            time_width=0.6,
                        ),
                    ),
                ),
            ),
        )


class CustomArrow(Line):
    """
    Custom arrow with tip in the middle instead of the end point
    to represent direction but not mistake it with a directed graph.
    """

    def __init__(
        self,
        start=LEFT,
        end=RIGHT,
        stroke_width=6,
        **kwargs,
    ):
        super().__init__(
            start=start,
            end=end,
            stroke_width=stroke_width,
            stroke_color=REDUCIBLE_VIOLET,
            **kwargs,
        )
        self.add_tip()

        self.tip.scale(0.7).move_to(self.point_from_proportion(0.25))

    def add_tip(self, tip=None, tip_shape=None, tip_length=None, at_start=False):
        """
        Overridden method to remove the `reset_endpoints_based_on_tip call`
        so the line actually reaches to the nodes in our particular case.
        """
        if tip is None:
            tip = self.create_tip(tip_shape, tip_length, at_start)
        else:
            self.position_tip(tip, at_start)
        self.asign_tip_attr(tip, at_start)
        self.add(tip)
        return self


class BruteForceSnippet(TSPAssumptions):
    def construct(self):
        self.add(BACKGROUND_IMG)
        cities = 5
        graph = TSPGraph(range(cities))
        all_edges = graph.get_all_edges()

        tour_perms = get_all_tour_permutations(cities, 0)

        self.play(Write(graph))
        self.play(LaggedStartMap(Write, all_edges.values()))
        self.play(
            graph.animate.shift(RIGHT * 4),
            VGroup(*all_edges.values()).animate.shift(RIGHT * 4),
            run_time=0.8,
        )
        self.wait()

        empty_mobs = (
            VGroup(*[Dot() for c in range(factorial(cities - 1) // 2)])
            .arrange_in_grid(cols=4, buff=1.5, row_heights=np.repeat(0.9, 10))
            .shift(LEFT * 3 + UP * 0.2)
        )
        permutations_mobs = VGroup()
        min_cost_mob = None
        best_cost = float("inf")
        for i, tour in enumerate(tour_perms):
            tour_edges = get_edges_from_tour(tour)

            edges_animation = self.focus_on_edges(tour_edges, all_edges)
            self.play(*edges_animation, run_time=1 / (i + 1))

            curr_tour_cost = get_cost_from_permutation(graph.dist_matrix, tour_edges)

            curr_tour = (
                VGroup(
                    *[v.copy() for v in graph.vertices.values()],
                    *[e.copy().set_stroke(width=2) for e in all_edges.values()],
                )
                .scale(0.3)
                .move_to(empty_mobs[i])
            )
            cost_text = (
                Text(f"{curr_tour_cost:.2f}", font=REDUCIBLE_MONO)
                .scale(0.3)
                .next_to(curr_tour, DOWN, buff=0.2)
            )
            if curr_tour_cost < best_cost:
                min_cost_mob = VGroup(curr_tour, cost_text)
                best_cost = curr_tour_cost
            permutations_mobs.add(curr_tour, cost_text)

            self.play(FadeIn(curr_tour), FadeIn(cost_text), run_time=1 / (i + 1))

        self.wait()

        self.play(ShowPassingFlash(SurroundingRectangle(min_cost_mob)), time_width=0.5)
        self.wait()

        self.play(
            LaggedStart(
                FadeOut(permutations_mobs),
                *[FadeOut(e) for e in all_edges.values()],
                graph.animate.move_to(ORIGIN),
            )
        )
        self.wait()

        ##################################33
        # BIG EXAMPLE

        big_cities = 12
        big_graph = TSPGraph(
            range(big_cities), label_scale=0.5, layout_scale=2.4, layout="circular"
        )
        all_edges_bg = big_graph.get_all_edges()

        self.play(FadeTransform(graph, big_graph))
        self.play(LaggedStartMap(Write, all_edges_bg.values()))

        all_tours = get_all_tour_permutations(big_cities, 0, 600)
        edge_tuples_tours = [get_edges_from_tour(tour) for tour in all_tours]
        pprint(len(all_tours))

        # change line here back to 200
        for i, tour_edges in enumerate(edge_tuples_tours[:200]):
            anims = self.focus_on_edges(tour_edges, all_edges_bg)
            self.play(*anims, run_time=1 / (5 * i + 1))

        self.wait()

        self.play(*[FadeOut(e) for e in all_edges_bg.values()])
        self.wait()

        # select random node to start
        for _ in range(10):
            random_indx = np.random.randint(0, big_cities)
            anims = self.focus_on_vertices(
                [random_indx],
                big_graph.vertices,
            )
            self.play(*anims, run_time=0.1)

        # of cities left
        cities_counter_label = Text(
            "# of cities left: ", font=REDUCIBLE_FONT, weight=MEDIUM
        ).scale(0.4)
        cities_counter = Text(str(big_cities), font=REDUCIBLE_MONO, weight=BOLD).scale(
            0.4
        )
        full_label = (
            VGroup(cities_counter_label, cities_counter)
            .arrange(RIGHT, buff=0.1, aligned_edge=UP)
            .next_to(big_graph, RIGHT, aligned_edge=DOWN)
        )

        # start creating the loop step by step
        last_node = random_indx  # take the last one from the other animation
        first_node = last_node

        valid_nodes = list(range(big_cities))
        valid_nodes.remove(last_node)

        self.play(FadeIn(full_label, shift=UP * 0.3))

        path_builder = VGroup()

        # change line here back to big_cities
        for i in range(big_cities):
            if len(valid_nodes) == 0:
                # we finished, so we go back home and break out of the loop
                new_counter = (
                    Text(str(len(valid_nodes)), font=REDUCIBLE_MONO, weight=BOLD)
                    .scale(0.4)
                    .move_to(cities_counter)
                )
                anims = self.focus_on_vertices(
                    [last_node],
                    big_graph.vertices,
                )

                self.play(Transform(cities_counter, new_counter), *anims, run_time=0.1)

                edge = big_graph.create_edge(last_node, first_node)
                path_builder.add(edge)
                self.wait()
                self.play(
                    Write(edge),
                )
                break

            # start from random index
            anims = self.focus_on_vertices(
                [last_node],
                big_graph.vertices,
            )
            new_counter = (
                Text(str(len(valid_nodes)), font=REDUCIBLE_MONO, weight=BOLD)
                .scale(0.4)
                .move_to(cities_counter)
            )
            self.play(Transform(cities_counter, new_counter), *anims, run_time=0.1)

            # create all edges from this vertex
            vertex_edges = {
                (last_node, v): big_graph.create_edge(last_node, v).set_opacity(0.5)
                for v in range(big_cities)
                if v != last_node
            }

            self.play(*[Write(e) for e in vertex_edges.values()])
            next_node = np.random.choice(valid_nodes)
            valid_nodes.remove(next_node)

            edge = vertex_edges.pop((last_node, next_node))
            path_builder.add(edge)
            self.play(
                ShowPassingFlash(
                    edge.copy().set_stroke(REDUCIBLE_YELLOW, width=7, opacity=1)
                ),
                edge.animate.set_opacity(1),
                *[e.animate.set_opacity(0) for e in vertex_edges.values()],
            )

            last_node = next_node

        self.wait()

        n_minus_one_factorial = (
            Text("(n - 1)!", font=REDUCIBLE_FONT, weight=BOLD)
            .scale(1.2)
            .set_stroke(width=10, background=True)
        )
        self.play(Write(n_minus_one_factorial, scale=0.95))
        self.wait()
        self.play(
            FadeOut(path_builder),
            FadeOut(n_minus_one_factorial, scale=0.95),
            FadeOut(big_graph),
            FadeOut(full_label, scale=0.95),
        )

        # go back to small example to show combinations
        small_cities = 4
        small_graph = TSPGraph(range(small_cities))

        all_possible_tours = get_all_tour_permutations(
            small_cities, 0, return_duplicates=True
        )

        all_tours = VGroup()
        for tour_symms in all_possible_tours:
            # tour is a list of 2 tours that are symmetric. bear that in mind!
            tour_pairs = VGroup()

            for tour in tour_symms:
                graph = VGroup()
                graph.add(*[v.copy() for v in small_graph.vertices.values()])

                graph.add(
                    *list(
                        small_graph.get_tour_edges(tour, edge_type=CustomArrow).values()
                    )
                ).scale(0.6)
                tour_pairs.add(graph)

            all_tours.add(tour_pairs.arrange(DOWN, buff=0.3))

        all_tours.arrange_in_grid(
            cols=factorial(small_cities - 1) // 2,
            row_heights=np.repeat(2.5, factorial(small_cities - 1) // 2),
            col_widths=np.repeat(3.5, factorial(small_cities - 1) // 2),
        ).scale_to_fit_width(config.frame_width - 4)

        self.play(*[FadeIn(t[0]) for t in all_tours])
        self.wait()
        self.play(*[FadeIn(t[1], shift=DOWN * 0.9) for t in all_tours])

        surr_rect = SurroundingRectangle(
            VGroup(all_tours[0]), color=REDUCIBLE_YELLOW, buff=0.3
        )
        annotation = (
            Text("These two are the same", font=REDUCIBLE_FONT, weight=BOLD)
            .next_to(surr_rect, UP, buff=0.2)
            .scale_to_fit_width(surr_rect.width)
            .set_color(REDUCIBLE_YELLOW)
        )

        def annotation_updater(mob):
            mob.next_to(surr_rect, UP, buff=0.2)

        annotation.add_updater(annotation_updater)

        self.play(Write(surr_rect), Write(annotation))

        self.play(
            surr_rect.animate.move_to(all_tours[1]),
        )
        self.play(
            surr_rect.animate.move_to(all_tours[2]),
        )
        self.wait()

        self.play(
            FadeOut(surr_rect),
            FadeOut(annotation),
            *[FadeOut(t[1], shift=DOWN * 0.9) for t in all_tours],
        )

        annotation_2 = (
            Text(
                "These correspond to half of the total",
                font=REDUCIBLE_FONT,
                weight=MEDIUM,
            )
            .scale(0.8)
            .shift(DOWN)
        )

        bold_template = TexTemplate()
        bold_template.add_to_preamble(r"\usepackage{bm}")
        n_minus_one_factorial_over_two = (
            Tex(r"$\bm{\frac{(n - 1)!}{2}}$", tex_template=bold_template)
            .scale(1.8)
            .next_to(annotation_2, DOWN, buff=0.5)
        )

        self.play(Write(annotation_2))
        self.play(Write(n_minus_one_factorial_over_two))

        self.wait()
        self.play(
            FadeOut(n_minus_one_factorial_over_two),
            FadeOut(annotation_2),
            *[FadeOut(t[0]) for t in all_tours],
        )
        self.wait()

        # show big number
        np.random.seed(110)
        random_graph = TSPGraph(
            range(20),
            layout="circular",
        ).shift(UP * 0.7)

        # scale down vertices
        [v.scale(0.7) for v in random_graph.vertices.values()]

        # recalculate edges
        all_edges = random_graph.get_all_edges(buff=random_graph.vertices[0].width / 2)

        # set edges opacity down
        [e.set_opacity(0.3) for e in all_edges.values()]

        twenty_factorial = (
            Text(
                f"(20 - 1)! / 2 = {factorial(20  - 1) // 2:,}".replace(",", " "),
                font=REDUCIBLE_MONO,
                weight=BOLD,
            )
            .scale(0.7)
            .next_to(random_graph, DOWN, buff=1)
        )

        self.play(FadeIn(random_graph))
        self.play(LaggedStartMap(Write, all_edges.values()))

        self.play(AddTextLetterByLetter(twenty_factorial))
        self.wait()

    def get_random_layout(self, N):
        random_points_in_frame = get_random_points_in_frame(N)
        return {v: point for v, point in zip(range(N), random_points_in_frame)}


class ProblemComplexity(TSPAssumptions):
    def construct(self):
        self.add(BACKGROUND_IMG)

        self.dynamic_programming_simulation()
        self.np_hard_problems()
        self.wait()

        self.plot_graphs()
        self.wait()

    def dynamic_programming_simulation(self):
        cities = 4

        graph = TSPGraph(range(cities)).shift(RIGHT * 3)
        all_edges = graph.get_all_edges()

        # make the whole graph a bit bigger
        VGroup(graph, *all_edges.values()).scale(1.4)

        all_labels = {
            t: graph.get_dist_label(e, graph.dist_matrix[t])
            for t, e in all_edges.items()
        }

        [e.set_opacity(0) for t, e in all_edges.items()]

        self.play(LaggedStartMap(FadeIn, graph))

        cities_list = list(range(1, cities))
        start_city = 0

        curr_tour_txt = Text("Current tour:", font=REDUCIBLE_FONT).scale(0.6)
        best_subtour_txt = Text("Best subtour:", font=REDUCIBLE_FONT).scale(0.6)
        curr_cost_txt = Text("Current cost:", font=REDUCIBLE_FONT).scale(0.6)
        best_cost_txt = Text("Best cost:", font=REDUCIBLE_FONT).scale(0.6)

        text_vg = (
            VGroup(curr_tour_txt, curr_cost_txt, best_subtour_txt, best_cost_txt)
            .arrange(DOWN, aligned_edge=LEFT, buff=0.3)
            .to_edge(LEFT)
        )

        curr_tour_str = (
            Text(f"[]", font=REDUCIBLE_MONO).scale(0.5).next_to(curr_tour_txt)
        )
        curr_cost_str = (
            Text(f"0", font=REDUCIBLE_MONO).scale(0.6).next_to(curr_cost_txt)
        )

        best_tour_str = (
            Text(f"[]", font=REDUCIBLE_MONO).scale(0.5).next_to(best_subtour_txt)
        )
        best_cost_str = (
            Text(f"0", font=REDUCIBLE_MONO).scale(0.6).next_to(best_cost_txt)
        )

        explanation = (
            Text(
                "Going from 0 to 0 through 0 cities",
                font=REDUCIBLE_FONT,
                t2f={"0": REDUCIBLE_MONO},
            )
            .scale(0.6)
            .next_to(
                text_vg,
                UP,
                buff=1,
                aligned_edge=LEFT,
            )
        )

        self.play(
            FadeIn(text_vg),
            FadeIn(curr_tour_str),
            FadeIn(curr_cost_str),
            FadeIn(best_tour_str),
            FadeIn(best_cost_str),
            FadeIn(explanation),
        )
        self.wait()
        for i in range(cities - 1):
            costs = {}
            internal_perms = list(permutations(cities_list, i + 1))

            for sub_tour in internal_perms:
                tour = [*sub_tour, start_city]

                tour_edges = graph.get_tour_edges(tour)
                tour_edges.popitem()

                # remove the last one since we are talking about sub tours
                tour_edge_tuples = get_edges_from_tour(tour)[:-1]

                curr_cost = get_cost_from_edges(tour_edge_tuples, graph.dist_matrix)

                costs[tuple(tour)] = curr_cost

                new_curr_tour = (
                    Text(f"{tour}", font=REDUCIBLE_MONO)
                    .scale(0.6)
                    .next_to(curr_tour_txt)
                )

                new_curr_cost = (
                    Text(f"{np.round(curr_cost, 1)}", font=REDUCIBLE_MONO)
                    .scale(0.6)
                    .next_to(curr_cost_txt)
                )

                explanation_str = f"Going from {tour[0]} to {tour[-1]} through {i} {'cities' if i != 1 else 'city'}"
                new_explanation = (
                    Text(
                        explanation_str,
                        font=REDUCIBLE_FONT,
                        t2f={
                            str(tour[0]): REDUCIBLE_MONO,
                            str(tour[-1]): REDUCIBLE_MONO,
                            str(i): REDUCIBLE_MONO,
                        },
                    )
                    .scale(0.6)
                    .next_to(text_vg, UP, buff=1, aligned_edge=LEFT)
                )

                edges_anims = self.focus_on_edges(tour_edges, all_edges=all_edges)
                labels_anims = self.focus_on_labels(
                    tour_edge_tuples, all_labels=all_labels
                )

                self.play(
                    *edges_anims,
                    *labels_anims,
                    Transform(curr_tour_str, new_curr_tour),
                    Transform(curr_cost_str, new_curr_cost),
                    Transform(explanation, new_explanation),
                    run_time=0.5,
                )

            # find best subtour and display it
            best_subtour, best_cost = min(costs.items(), key=lambda x: x[1])
            print(costs, best_subtour, best_cost)

            new_best_tour = (
                Text(f"{tour}", font=REDUCIBLE_MONO)
                .scale(0.6)
                .next_to(best_subtour_txt)
            )

            new_best_cost = (
                Text(f"{np.round(best_cost, 1)}", font=REDUCIBLE_MONO)
                .scale(0.6)
                .next_to(best_cost_txt)
            )

            self.play(
                Transform(best_tour_str, new_best_tour),
                Transform(best_cost_str, new_best_cost),
            )

            self.wait()

        self.play(
            FadeOut(text_vg),
            FadeOut(curr_tour_str),
            FadeOut(curr_cost_str),
            FadeOut(best_tour_str),
            FadeOut(best_cost_str),
            FadeOut(explanation),
            *[FadeOut(mob) for mob in all_edges.values()],
            *[FadeOut(mob) for mob in all_labels.values()],
            FadeOut(graph),
        )

    def np_hard_problems(self):
        # explanation about NP problems
        tsp_problem = (
            Module(["Traveling Salesman", "Problem"], text_weight=BOLD)
            .scale(0.8)
            .shift(DOWN * 0.7)
        )
        np_hard_problems = Module(
            "NP Hard Problems",
            fill_color=REDUCIBLE_GREEN_DARKER,
            stroke_color=REDUCIBLE_GREEN_LIGHTER,
            width=12,
            height=6,
            text_weight=BOLD,
            text_scale=0.6,
            text_position=UP,
        )
        problems = [
            ["Integer", "Programming"],
            ["Knapsack", "Problem"],
            "Bin Packing",
            "Subset sum",
        ]
        modules = VGroup(*[Module(p, text_weight=BOLD).scale(0.6) for p in problems])
        modules[0].move_to(tsp_problem).shift(LEFT * 2 + UP * 1.5)
        modules[1].move_to(tsp_problem).shift(LEFT * 2 + DOWN * 1.2)
        modules[2].move_to(tsp_problem).shift(RIGHT * 1.5 + UP * 1.2)
        modules[3].move_to(tsp_problem).shift(RIGHT * 1.9 + DOWN * 1)

        self.play(Write(np_hard_problems))
        self.play(FadeIn(tsp_problem, scale=1.05))
        self.wait()

        self.play(
            LaggedStart(*[FadeIn(m, scale=1.05) for m in modules], lag_ratio=1),
            run_time=5,
        )
        self.wait()

        self.play(FadeOut(modules), FadeOut(np_hard_problems), FadeOut(tsp_problem))

    def plot_graphs(self):

        # plot graphs
        eval_range = [0, 20]
        num_plane = Axes(
            x_range=eval_range,
            y_range=[0, 400],
            y_length=10,
            x_length=15,
            tips=False,
            axis_config={"include_ticks": False},
        ).to_corner(DL)

        bold_template = TexTemplate()
        bold_template.add_to_preamble(r"\usepackage{bm}")

        constant_plot = (
            num_plane.plot(lambda x: 5, x_range=eval_range)
            .set_color(REDUCIBLE_GREEN)
            .set_stroke(width=5)
        )
        constant_tag = (
            Tex(r"$\bm{O(1)}$", tex_template=bold_template)
            .set_color(REDUCIBLE_GREEN)
            .set_stroke(width=5, background=True)
            .scale(0.6)
            .next_to(constant_plot, UP, buff=0.1)
        )

        linear_plot = (
            num_plane.plot(lambda x: x, x_range=eval_range)
            .set_color(REDUCIBLE_GREEN_LIGHTER)
            .set_stroke(width=5)
        )
        linear_tag = (
            Tex(r"$\bm{O(n)}$", tex_template=bold_template)
            .set_color(REDUCIBLE_GREEN_LIGHTER)
            .scale(0.6)
            .next_to(linear_plot.point_from_proportion(0.7), UP)
        )

        quad_plot = (
            num_plane.plot(lambda x: x**2, x_range=eval_range)
            .set_color(REDUCIBLE_VIOLET)
            .set_stroke(width=5)
        )
        quad_tag = (
            Tex(r"$\bm{O(n^2)}$", tex_template=bold_template)
            .set_color(REDUCIBLE_VIOLET)
            .scale(0.6)
            .next_to(quad_plot.point_from_proportion(0.5), RIGHT)
        )

        poly_plot = (
            num_plane.plot(lambda x: 3 * x**2 + 2 * x, x_range=eval_range)
            .set_color(REDUCIBLE_YELLOW)
            .set_stroke(width=5)
        )
        poly_tag = (
            Tex(r"$\bm{O(3n^2+2n)}$", tex_template=bold_template)
            .set_color(REDUCIBLE_YELLOW)
            .scale(0.6)
            .next_to(poly_plot.point_from_proportion(0.25), RIGHT)
        )

        exponential_plot = (
            num_plane.plot(lambda x: 2**x, x_range=[0, 10])
            .set_color(REDUCIBLE_ORANGE)
            .set_stroke(width=5)
        )
        exp_tag = (
            Tex(r"$\bm{O(2^n)}$", tex_template=bold_template)
            .set_color(REDUCIBLE_ORANGE)
            .scale(0.6)
            .next_to(exponential_plot.point_from_proportion(0.2), RIGHT)
        )

        factorial_plot = (
            num_plane.plot(
                lambda x: gamma(x) if x > 1 else x**2,
                x_range=[0, 10],
            )
            .set_color(REDUCIBLE_CHARM)
            .set_stroke(width=5)
        )

        factorial_tag = (
            Tex(r"$\bm{O(n!)}$", tex_template=bold_template)
            .set_color(REDUCIBLE_CHARM)
            .scale(0.6)
            .next_to(factorial_plot.point_from_proportion(0.001), LEFT)
        )

        plots = [
            constant_plot,
            linear_plot,
            quad_plot,
            poly_plot,
            exponential_plot,
            factorial_plot,
        ]
        tags = [
            constant_tag,
            linear_tag,
            quad_tag,
            poly_tag,
            exp_tag,
            factorial_tag,
        ]
        self.play(
            Write(
                num_plane.x_axis,
            ),
            Write(
                num_plane.y_axis,
            ),
        )
        self.play(LaggedStart(*[Write(p) for p in plots]))
        self.play(*[FadeIn(t, scale=0.95) for t in tags])
        self.wait()

        self.play(
            constant_plot.animate.set_stroke(opacity=0.3),
            linear_plot.animate.set_stroke(opacity=0.3),
            quad_plot.animate.set_stroke(opacity=0.3),
            poly_plot.animate.set_stroke(opacity=0.3),
            factorial_plot.animate.set_stroke(opacity=0.3),
            constant_tag.animate.set_opacity(0.3),
            linear_tag.animate.set_opacity(0.3),
            quad_tag.animate.set_opacity(0.3),
            poly_tag.animate.set_opacity(0.3),
            factorial_tag.animate.set_opacity(0.3),
        )


class TransitionOtherApproaches(TSPAssumptions):
    def construct(self):

        bg = ImageMobject("usa-map-satellite-markers.png").scale_to_fit_width(
            config.frame_width
        )
        dark_filter = ScreenRectangle(height=10).set_fill(BLACK, opacity=0.2)
        self.play(FadeIn(bg), FadeIn(dark_filter))
        self.wait()
        # self.add(
        #     Axes(
        #         axis_config={"include_numbers": True},
        #         x_length=config.frame_width,
        #         y_length=config.frame_height,
        #         tips=False,
        #     )
        # )

        city_coords = [
            (-5.95, 0.7, 0),  # SF
            (-5.8, 0.55, 0),  # san jose
            (-4.9, -0.45, 0),  # LA
            (0.45, 0.85, 0),  # Atlanta
            (-0.35, -0.35, 0),  # Miami
            (-4.35, 1.75, 0),  # Miami
            (-1.75, 1.35, 0),  # Denver
            (3.5, -0.5, 0),  # Atlanta
            (-4.6, -0.87, 0),  # San diego
            (-3.35, -0.65, 0),  # phoenix
            (0.45, -0.85, 0),  # dallas
            (0, -1.8, 0),  # san antonio
            (0.8, -1.75, 0),  # houston
            (2.74, 2, 0),  # chicago
            (6.1, 1.65, 0),  # new york
            (5.85, 1.35, 0),  # philadelphia
        ]
        nodes_per_core = 10
        total_number_of_nodes = len(city_coords) * nodes_per_core
        print(f"{total_number_of_nodes = }")
        graph = TSPGraph(
            range(total_number_of_nodes),
            vertex_config={
                "fill_opacity": 1,
                "fill_color": REDUCIBLE_YELLOW,
                "stroke_color": REDUCIBLE_PURPLE_DARKER,
                "stroke_width": 0,
            },
            labels=False,
            layout=self.get_city_layout(
                core_cities=city_coords, nodes_per_core=nodes_per_core
            ),
        )
        print(f"> Finished building graph with {total_number_of_nodes} nodes")

        # first len(city_coords) are actual cities, so we give them more relevance
        [
            v.scale(0.8)
            .set_fill(REDUCIBLE_VIOLET)
            .set_stroke(REDUCIBLE_PURPLE_DARKER, width=3)
            for v in list(graph.vertices.values())[: len(city_coords)]
        ]

        # these are the tiny nodes around every city
        [v.scale(0.3) for v in list(graph.vertices.values())[len(city_coords) :]]
        self.add_foreground_mobjects(*list(graph.vertices.values())[: len(city_coords)])

        self.play(
            LaggedStart(*[FadeIn(v, scale=0.7) for v in graph.vertices.values()]),
            run_time=3,
        )
        self.wait()

        all_edges = graph.get_all_edges(
            buff=graph.vertices[len(city_coords) + 1].width / 2
        )
        print(f"> Finished getting all {len(all_edges)} edges")
        [e.set_opacity(0) for e in all_edges.values()]

        # generate all possible tours (until max cap) and try and get a good variety of tours
        tour_perms = []
        for i in range(len(city_coords)):
            tour_perms.extend(
                get_all_tour_permutations(total_number_of_nodes, i, max_cap=20)
            )

        tour_perms = filter(lambda t: len(t) == total_number_of_nodes, tour_perms)

        # randomizing a bit more
        edges_perms = [swap_random(get_edges_from_tour(t)) for t in tour_perms]

        print(f"> Finished calculating all {len(edges_perms)} tuples from every tour")

        cost_indicator = (
            Text(
                f"Distance: {get_cost_from_edges(edges_perms[0], graph.dist_matrix):.2f}",
                font=REDUCIBLE_FONT,
                t2f={
                    f"{get_cost_from_edges(edges_perms[0], graph.dist_matrix):.2f}": REDUCIBLE_MONO
                },
                weight=BOLD,
            )
            .set_stroke(width=4, background=True)
            .scale(0.6)
            .next_to(graph, DOWN, buff=0.5, aligned_edge=LEFT)
        )

        # by changing the slice size you can create a longer or shorter example
        for i, tour_edges in enumerate(edges_perms):
            # the all_edges dict only stores the edges in ascending order
            tour_edges = list(
                map(lambda x: x if x[0] < x[1] else (x[1], x[0]), tour_edges)
            )
            cost = get_cost_from_edges(tour_edges, graph.dist_matrix)
            new_cost_indicator = (
                Text(
                    f"Distance: {cost:.2f}",
                    font=REDUCIBLE_FONT,
                    t2f={f"{cost:.2f}": REDUCIBLE_MONO},
                    weight=BOLD,
                )
                .set_stroke(width=4, background=True)
                .scale(0.6)
                .next_to(graph, DOWN, buff=0.5, aligned_edge=LEFT)
            )

            if i == 0:
                anims = LaggedStart(
                    *[Write(all_edges[e].set_opacity(1)) for e in tour_edges]
                )
                self.play(
                    anims,
                    Transform(cost_indicator, new_cost_indicator),
                    run_time=1 / (5 * i + 1),
                )

            else:
                anims = self.focus_on_edges(
                    tour_edges,
                    all_edges=all_edges,
                )

                self.play(
                    *anims,
                    Transform(cost_indicator, new_cost_indicator),
                    run_time=1 / (5 * i + 1),
                )

        print("Finished looping through tours")

        self.wait()
        # transition to nearest neighbour example

        self.play(*[all_edges[e].animate.set_opacity(0) for e in tour_edges])
        print("finished deleting edges")
        self.wait()

        nn_tour, nn_cost = get_nearest_neighbor_solution(graph.dist_matrix)
        nn_edges = get_edges_from_tour(nn_tour)
        nn_edges = list(map(lambda x: x if x[0] < x[1] else (x[1], x[0]), nn_edges))

        nn_cost_indicator = (
            Text(
                f"Distance: {nn_cost:.2f}",
                font=REDUCIBLE_FONT,
                t2f={f"{nn_cost:.2f}": REDUCIBLE_MONO},
                weight=BOLD,
            )
            .set_stroke(width=4, background=True)
            .scale(0.6)
            .next_to(graph, DOWN, buff=0.5, aligned_edge=LEFT)
        )

        frame: Mobject = self.camera.frame
        good_enough_txt = (
            Text(
                'Can we find a "good enough" tour quickly?',
                font=REDUCIBLE_FONT,
                weight=BOLD,
            )
            .scale_to_fit_width(graph.width - 2)
            .move_to(frame.get_top())
        )
        self.play(
            frame.animate.shift(OUT * 0.5 + UP * 0.8),
            dark_filter.animate.set_opacity(1),
            LaggedStart(*[all_edges[e].animate.set_opacity(1) for e in nn_edges]),
            FadeTransform(cost_indicator, nn_cost_indicator),
            FadeIn(good_enough_txt, scale=0.95),
            run_time=3
            # Transform(cost_indicator, nn_cost_indicator),
        )
        self.wait()

    ############################ UTIL FUNCTIONS
    def get_specific_layout(self, *coords):
        # dict with v number and coordinate
        return {v: coord for v, coord in enumerate(coords)}

    def get_normal_dist_layout(self, N):

        x_values = np.random.normal(-0.3, 1.8, size=N)
        y_values = np.random.normal(0, 0.7, size=N)

        return {
            v: (point[0], point[1], 0)
            for v, point in enumerate(zip(x_values, y_values))
        }

    def get_city_layout(self, core_cities: Iterable[tuple], nodes_per_core: int = 10):
        v_dict = {v: coord for v, coord in enumerate(core_cities)}

        # for each city, we are gona spread nodes_per_core more cities around it
        index = len(core_cities)
        for v in core_cities:
            x_values = np.random.normal(v[0], 0.1, size=nodes_per_core)
            y_values = np.random.normal(v[1], 0.1, size=nodes_per_core)
            for x, y in zip(x_values, y_values):
                v_dict[index] = (x, y, 0)
                index += 1

        return v_dict

    def focus_on_edges(
        self,
        edges_to_focus_on: Iterable[tuple],
        all_edges: Iterable[tuple],
        min_opacity=0.0001,
    ):
        edges_animations = []

        edges_to_focus_on = list(
            map(lambda t: (t[1], t[0]) if t[0] > t[1] else t, edges_to_focus_on)
        )

        last_animation_edges = filter(
            lambda e: e[1].stroke_opacity > 0 or e[1].fill_opacity > 0,
            all_edges.items(),
        )
        edges_animations.extend(
            [
                all_edges[t].animate.set_opacity(min_opacity)
                for t, e in last_animation_edges
            ]
        )

        this_animation_edges = filter(
            lambda e: e[0] in edges_to_focus_on, all_edges.items()
        )
        edges_animations.extend(
            [all_edges[t].animate.set_opacity(1) for t, e in this_animation_edges]
        )

        return edges_animations


class CustomLabel(Text):
    def __init__(self, label, font=REDUCIBLE_MONO, scale=0.2, weight=MEDIUM):
        super().__init__(label, font=font, weight=weight)
        self.scale(scale)


class SimulatedAnnealing(BruteForceSnippet, TransitionOtherApproaches):
    def construct(self):
        self.add(BACKGROUND_IMG)

        frame = self.camera.frame
        self.guided_example()
        self.wait()
        self.clear()
        self.add(BACKGROUND_IMG)

        self.show_temperature()
        self.wait()
        self.clear()
        self.add(BACKGROUND_IMG)

        self.simulated_annealing()

    def guided_example(self):
        N = 30
        frame = self.camera.frame
        graph = TSPGraph(
            range(N),
            layout="spring",
            layout_config={"k": 9, "iterations": 10},
            layout_scale=5,
        ).scale(0.6)

        np.random.seed(101)

        self.add_foreground_mobjects(graph)
        all_edges = graph.get_all_edges(buff=graph.vertices[0].width / 2)
        [e.set_opacity(0) for e in all_edges.values()]

        self.play(*[FadeIn(v) for v in graph.vertices.values()])

        nn_tour, nn_cost = get_nearest_neighbor_solution(graph.dist_matrix, start=0)
        nn_edges = get_edges_from_tour(nn_tour)
        nn_edges = list(map(lambda t: (t[1], t[0]) if t[0] > t[1] else t, nn_edges))

        cost_indicator = (
            Text(
                f"Distance: {nn_cost:.2f}",
                font=REDUCIBLE_FONT,
                t2f={f"{nn_cost:.2f}": REDUCIBLE_MONO},
                weight=BOLD,
            )
            .set_stroke(width=4, background=True)
            .scale(0.6)
            .next_to(graph, DOWN, buff=0.5)
        )

        self.play(
            # *self.focus_on_edges(nn_edges, all_edges),
            LaggedStart(*[all_edges[t].animate.set_opacity(1) for t in nn_edges]),
            FadeIn(cost_indicator),
            run_time=3,
        )
        self.wait()
        all_edges_group = VGroup(*[edge_mob for edge_mob in all_edges.values()])
        entire_graph = VGroup(graph, all_edges_group)
        self.play(
            entire_graph.animate.shift(LEFT * 3), cost_indicator.animate.shift(LEFT * 3)
        )

        change_history = (
            VGroup(*[Dot() for a in range(5)])
            .arrange(DOWN, buff=0.5)
            .next_to(graph, RIGHT, buff=1)
        )

        cost_hist_title = (
            Text(
                f"Cost history",
                font=REDUCIBLE_FONT,
                weight=BOLD,
            )
            .scale(0.7)
            .next_to(change_history, UP, buff=0.7, aligned_edge=LEFT)
        )
        self.play(FadeIn(cost_hist_title))

        nn_change = (
            Text(
                f"· NN tour: {nn_cost:.2f}",
                font=REDUCIBLE_FONT,
                t2f={f"{nn_cost:.2f}": REDUCIBLE_MONO},
                weight=BOLD,
            )
            .scale(0.4)
            .move_to(change_history[0], aligned_edge=LEFT)
        )
        up_arrow = (
            Text("↑", font=REDUCIBLE_MONO)
            .set_color(REDUCIBLE_CHARM)
            .scale_to_fit_height(nn_change.height)
            .scale(1.4)
        )
        down_arrow = (
            Text("↓", font=REDUCIBLE_MONO)
            .set_color(REDUCIBLE_GREEN)
            .scale_to_fit_height(nn_change.height)
            .scale(1.4)
        )

        self.play(FadeIn(nn_change, shift=UP * 0.3))
        self.wait()

        def iterate_two_opt_animation(v1, v2, tour):
            two_opt_tour = two_opt_swap(tour, v1, v2)

            two_opt_edges = get_edges_from_tour(two_opt_tour)
            two_opt_cost = get_cost_from_edges(two_opt_edges, graph.dist_matrix)

            two_opt_cost_indicator = (
                Text(
                    f"Distance: {two_opt_cost:.2f}",
                    font=REDUCIBLE_FONT,
                    t2f={f"{two_opt_cost:.2f}": REDUCIBLE_MONO},
                    weight=BOLD,
                )
                .set_stroke(width=4, background=True)
                .scale(0.6)
                .next_to(graph, DOWN, buff=0.5)
            )

            focus_edges_animations = self.focus_on_edges(two_opt_edges, all_edges)
            cost_indicator_animation = Transform(cost_indicator, two_opt_cost_indicator)
            # self.play(*focus_edges_animations, cost_indicator_animation)

            return two_opt_tour, focus_edges_animations, cost_indicator_animation

        (
            last_tour,
            focus_edges_animations,
            cost_indicator_animation,
        ) = iterate_two_opt_animation(29, 18, nn_tour)

        first_change = self.create_change_history_entry(
            29,
            18,
            get_cost_from_edges(get_edges_from_tour(last_tour), graph.dist_matrix),
        ).move_to(change_history[1], aligned_edge=LEFT)

        self.play(
            *focus_edges_animations,
            cost_indicator_animation,
            FadeIn(first_change, shift=DOWN * 0.3),
            FadeIn(
                down_arrow.copy().next_to(first_change, RIGHT, buff=0.3),
                shift=DOWN * 0.3,
            ),
        )

        (
            last_tour,
            focus_edges_animations,
            cost_indicator_animation,
        ) = iterate_two_opt_animation(5, 1, last_tour)
        last_cost = get_cost_from_edges(
            get_edges_from_tour(last_tour), graph.dist_matrix
        )
        second_change = self.create_change_history_entry(5, 1, last_cost).move_to(
            change_history[2], aligned_edge=LEFT
        )
        self.play(
            *focus_edges_animations,
            cost_indicator_animation,
            FadeIn(second_change, shift=DOWN * 0.3),
            FadeIn(
                down_arrow.copy().next_to(second_change, RIGHT, buff=0.3),
                shift=DOWN * 0.3,
            ),
        )

        (
            last_tour,
            focus_edges_animations,
            cost_indicator_animation,
        ) = iterate_two_opt_animation(20, 5, last_tour)
        last_cost = get_cost_from_edges(
            get_edges_from_tour(last_tour), graph.dist_matrix
        )
        second_change = self.create_change_history_entry(20, 5, last_cost).move_to(
            change_history[3], aligned_edge=LEFT
        )
        self.play(
            *focus_edges_animations,
            cost_indicator_animation,
            FadeIn(second_change, shift=UP * 0.3),
            FadeIn(
                up_arrow.copy().next_to(second_change, RIGHT, buff=0.3),
                shift=UP * 0.3,
            ),
        )
        self.wait()

        # we have a worse solution, we have to decide whether to keep it or not!
        T = 0.8
        p_string = (
            Tex(r"$p < " + f"{T:.2f}" + r"\Rightarrow \text{accept}$")
            .scale(0.5)
            .next_to(second_change, DOWN, aligned_edge=LEFT, buff=0.2)
            .shift(RIGHT * 0.4)
        )
        p_random = (
            Tex(r"$\text{RNG} = 0.54$")
            .scale(0.5)
            .next_to(p_string, DOWN, buff=0.2, aligned_edge=LEFT)
        )
        self.play(
            FadeIn(p_string, shift=UP * 0.3),
        )
        self.play(
            FadeIn(p_random, shift=UP * 0.3),
        )
        self.wait()
        self.play(
            FadeOut(p_string, shift=UP * 0.3),
            FadeOut(p_random, shift=UP * 0.3),
        )
        self.wait()

        undo_tour = last_tour
        # next example where we don't choose the worst
        (
            last_tour,
            focus_edges_animations,
            cost_indicator_animation,
        ) = iterate_two_opt_animation(13, 3, last_tour)
        last_cost = get_cost_from_edges(
            get_edges_from_tour(last_tour), graph.dist_matrix
        )
        second_change = self.create_change_history_entry(13, 3, last_cost).move_to(
            change_history[4], aligned_edge=LEFT
        )
        arrow_change = up_arrow.copy().next_to(second_change, RIGHT, buff=0.3)
        self.play(
            *focus_edges_animations,
            cost_indicator_animation,
            FadeIn(second_change, shift=UP * 0.3),
            FadeIn(
                arrow_change,
                shift=UP * 0.3,
            ),
        )
        self.wait()

        T = 0.76
        p_string = (
            Tex(r"$p < " + f"{T:.2f}" + r"\Rightarrow \text{accept}$")
            .scale(0.5)
            .next_to(second_change, DOWN, aligned_edge=LEFT, buff=0.2)
            .shift(RIGHT * 0.4)
        )
        p_random = (
            Tex(r"$\text{RNG}= 0.81$")
            .scale(0.5)
            .next_to(p_string, DOWN, buff=0.2, aligned_edge=LEFT)
        )
        self.play(
            FadeIn(p_string, shift=UP * 0.3),
        )
        self.play(
            FadeIn(p_random, shift=UP * 0.3),
        )
        self.wait()

        self.play(
            FadeOut(p_string, shift=UP * 0.3),
            FadeOut(p_random, shift=UP * 0.3),
        )
        annotation = (
            Text(
                "Undo last change\nand continue from there",
                font=REDUCIBLE_FONT,
                weight=BOLD,
            )
            .scale(0.35)
            .next_to(second_change, DOWN, aligned_edge=LEFT, buff=0.2)
            .shift(RIGHT * 0.4)
        )
        self.play(FadeIn(annotation))

        undo_edges = get_edges_from_tour(undo_tour)
        self.play(
            *self.focus_on_edges(undo_edges, all_edges),
            FadeOut(second_change),
            FadeOut(arrow_change),
            FadeOut(annotation),
        )

    def show_temperature(self):
        iterations = 100
        axes = (
            Axes(
                x_range=[0, iterations, 10],
                y_range=[0, 1, 0.25],
                y_length=8,
                tips=False,
                axis_config={
                    "include_numbers": True,
                    "label_constructor": CustomLabel,
                },
            )
            .scale(0.5)
            .shift(UP * 1.5)
        )

        temp_nl = NumberLine(
            x_range=[0, 1],
            include_numbers=True,
            length=axes.x_axis.width,
        ).next_to(axes.x_axis, DOWN, buff=1.5)

        temp_label = (
            Text("Temperature (T)", font=REDUCIBLE_FONT, weight=BOLD)
            .scale(0.5)
            .rotate(90 * DEGREES)
            .next_to(axes.y_axis, LEFT, buff=0.2)
        )

        iterations_label = (
            Text("Iterations", font=REDUCIBLE_FONT, weight=BOLD)
            .scale(0.5)
            .next_to(axes.x_axis, DOWN, buff=0.2)
        )

        T = ValueTracker(1)

        temp_marker = (
            Line(UP, UP * 1.3)
            .set_stroke(width=5, color=REDUCIBLE_YELLOW)
            .add_updater(lambda mob: mob.move_to(temp_nl.n2p(T.get_value())))
        )
        filled_rect = (
            Line(temp_nl.n2p(0), temp_marker.get_center())
            .set_stroke(width=10)
            .set_color(REDUCIBLE_VIOLET)
        )
        filled_rect.add_updater(
            lambda mob: mob.put_start_and_end_on(
                temp_nl.n2p(0), temp_marker.get_center()
            )
        )

        temperature_title = Text(
            "Temperature (T)", font=REDUCIBLE_FONT, weight=BOLD
        ).scale(0.7)
        temperature_title.move_to(UP * 3.2)
        temp_nl_copy = temp_nl.copy()

        definition = Tex(r"Controls probability $P$ of accepting worse solution").scale(
            0.7
        )

        t_range = MathTex(r"T \in [0, 1], T_0 = 1").scale(0.7)

        p_explation = Tex(
            r"$P($Accept worse tour with cost increase $\Delta)$ = $e^{-\Delta / T}$"
        ).scale(0.7)
        p_explation.next_to(iterations_label, DOWN)

        definition.next_to(temperature_title, DOWN, buff=0.5)

        text_group = VGroup(temp_nl_copy, t_range).arrange(DOWN, buff=0.5)

        p_copy = p_explation.copy().move_to(definition.get_center())

        filled_rect_copy = (
            Line(temp_nl_copy.n2p(0), temp_nl_copy.n2p(1))
            .set_stroke(width=10)
            .set_color(REDUCIBLE_VIOLET)
        )

        temp_marker_copy = (
            Line(UP, UP * 1.3)
            .set_stroke(width=5, color=REDUCIBLE_YELLOW)
            .move_to(temp_nl_copy.n2p(1))
        )
        self.play(Write(temperature_title))
        self.wait()

        self.play(FadeIn(definition))

        self.play(FadeIn(temp_nl_copy), FadeIn(t_range))
        self.wait()

        self.play(Create(filled_rect_copy))
        self.play(Write(temp_marker_copy))
        self.wait()

        self.play(FadeTransform(definition, p_copy))
        self.wait()

        self.play(
            FadeOut(temperature_title),
            ReplacementTransform(p_copy, p_explation),
            FadeOut(temp_nl_copy),
            FadeOut(temp_marker_copy),
            FadeOut(filled_rect_copy),
            FadeOut(t_range),
            FadeIn(axes),
            FadeIn(temp_nl),
            FadeIn(temp_marker),
            FadeIn(temp_label),
            FadeIn(iterations_label),
            FadeIn(filled_rect),
        )
        self.wait()

        last_temp = T.get_value()
        last_plot = axes.plot(lambda x: 1 / x, x_range=[1, 1.001])
        last_area = axes.get_area(last_plot, x_range=[1, 1.001]).set_fill(
            REDUCIBLE_YELLOW, opacity=0.3
        )
        for i in range(1, 101):
            v = T.get_value()

            new_plot = axes.plot(lambda x: 1 / x, x_range=[1, i])
            new_area = (
                axes.get_area(new_plot, x_range=[1, i])
                .set_fill(REDUCIBLE_YELLOW, opacity=0.3)
                .set_stroke(width=0)
            )
            new_line_chunk = self.next_iteration_line(axes, i, v, last_temp).set_stroke(
                width=4
            )
            self.play(
                last_area.animate.become(new_area),
                T.animate.set_value(1 / (1 * i + 1)),
                Write(new_line_chunk),
                run_time=(1 / (i + 1)),
            )

            last_temp = v

        self.wait()

        explore_annotation = (
            Text("Explore a lot", font=REDUCIBLE_FONT, weight=BOLD)
            .scale(0.4)
            .next_to(new_plot.point_from_proportion(0.1), RIGHT, buff=0.4)
        )

        exploit_annotation = (
            Text("Converge to local optima", font=REDUCIBLE_FONT, weight=BOLD)
            .scale(0.3)
            .next_to(new_plot.point_from_proportion(0.95), UP, buff=0.2)
        )

        self.play(FadeIn(explore_annotation))
        self.wait()
        self.play(FadeIn(exploit_annotation))

    def simulated_annealing(self):
        # DEFINITION AND SETUP
        np.random.seed(100)
        N = 30
        iterations = 100
        T = ValueTracker(1)

        graph = (
            TSPGraph(
                range(N),
                layout="spring",
                layout_config={"k": 9, "iterations": 10},
                layout_scale=5,
            )
            .scale(0.6)
            .shift(LEFT * 3)
        )

        axes_temp = Axes(
            x_range=[0, iterations, 10],
            y_range=[0, 1, 0.25],
            y_length=8,
            tips=False,
            axis_config={
                "include_numbers": True,
                "label_constructor": CustomLabel,
            },
        ).scale(0.3)

        temp_label = (
            Text("Temperature (T)", font=REDUCIBLE_FONT, weight=BOLD)
            .scale(0.3)
            .rotate(90 * DEGREES)
            .next_to(axes_temp.y_axis, LEFT, buff=0.3)
        )
        iterations_label = (
            Text("Iterations", font=REDUCIBLE_FONT, weight=BOLD)
            .scale(0.3)
            .next_to(axes_temp.x_axis, DOWN, buff=0.3)
        )

        axes_vg = VGroup(axes_temp, temp_label, iterations_label).next_to(
            graph, RIGHT, aligned_edge=UP, buff=1.7
        )

        axes_cost = (
            Axes(
                x_range=[0, iterations, 10],
                y_range=[30, 91, 10],
                y_length=8,
                tips=False,
                axis_config={
                    "include_numbers": True,
                    "label_constructor": CustomLabel,
                },
            )
            .scale(0.3)
            .next_to(axes_temp, ORIGIN, aligned_edge=RIGHT)
        )
        fake_y_axes_cost = (
            NumberLine(
                x_range=[30, 91, 10],
                length=8,
                rotation=90 * DEGREES,
                label_direction=RIGHT,
            )
            .scale(0.3)
            .move_to(axes_temp.x_axis.ticks[-1], aligned_edge=DOWN)
        )

        cost_labels = VGroup(
            *[
                Text(str(e), font=REDUCIBLE_MONO, font_size=36)
                .scale(0.3)
                .next_to(
                    fake_y_axes_cost.ticks[i],
                    RIGHT,
                    buff=0.1,
                )
                for i, e in enumerate(range(30, 91, 10))
            ]
        )
        cost_y_label = (
            Text("Cost", font=REDUCIBLE_FONT, weight=BOLD)
            .scale(0.3)
            .rotate(-90 * DEGREES)
            .next_to(fake_y_axes_cost, RIGHT, buff=0.6)
        )
        length = 0.3

        t_line_legend = Line(LEFT * length / 2, RIGHT * length / 2).set_color(
            REDUCIBLE_YELLOW
        )
        tested_line_legend = Line(LEFT * length / 2, RIGHT * length / 2).set_color(
            REDUCIBLE_PURPLE
        )
        best_line_legend = Line(LEFT * length / 2, RIGHT * length / 2).set_color(
            REDUCIBLE_GREEN_LIGHTER
        )
        text_scale = 0.35
        t_legend_text = Text("T", font=REDUCIBLE_FONT).scale(text_scale)
        test_legend_text = Text("Tested", font=REDUCIBLE_FONT).scale(text_scale)
        best_legend_text = Text("Best", font=REDUCIBLE_FONT).scale(text_scale)
        t_legend_text.next_to(t_line_legend, RIGHT)
        test_legend_text.next_to(tested_line_legend, RIGHT)
        best_legend_text.next_to(best_line_legend, RIGHT)

        legend = (
            VGroup(
                VGroup(t_line_legend, t_legend_text),
                VGroup(tested_line_legend, test_legend_text),
                VGroup(best_line_legend, best_legend_text),
            )
            .arrange(RIGHT)
            .next_to(axes_vg, UP)
            .shift(RIGHT * SMALL_BUFF * 4)
        )

        change_history = (
            VGroup(*[Dot() for a in range(5)])
            .arrange(DOWN, buff=0.5)
            .next_to(axes_temp.y_axis, DOWN, buff=1.3, aligned_edge=RIGHT)
        )

        cost_indicator_text = Text(
            f"Path length:",
            font=REDUCIBLE_FONT,
            weight=BOLD,
        ).scale(0.4)

        best_indicator_text = Text(
            f"Best:",
            font=REDUCIBLE_FONT,
            weight=BOLD,
        ).scale(0.4)

        def iterate_two_opt_animation(v1, v2, tour, last_cost: Text):
            two_opt_tour = two_opt_swap(tour, v1, v2)

            two_opt_edges = get_edges_from_tour(two_opt_tour)
            two_opt_cost = get_cost_from_edges(two_opt_edges, graph.dist_matrix)

            two_opt_cost_indicator = (
                Text(
                    f"{two_opt_cost:.2f}",
                    font=REDUCIBLE_MONO,
                    weight=BOLD,
                )
                .set_stroke(width=4, background=True)
                .scale(0.4)
                .next_to(cost_indicator_text, RIGHT, buff=0.3)
            )

            focus_edges_animations = self.focus_on_edges(two_opt_edges, all_edges)

            direction = -np.sign(float(last_cost.text) - two_opt_cost)
            cost_indicator_animations = [
                FadeIn(two_opt_cost_indicator, shift=UP * direction * 0.3),
                FadeOut(last_cost, shift=UP * direction * 0.3),
            ]
            # self.play(*focus_edges_animations, cost_indicator_animation)

            return (
                two_opt_tour,
                two_opt_cost_indicator,
                focus_edges_animations,
                cost_indicator_animations,
            )

        self.add_foreground_mobjects(graph)
        all_edges = graph.get_all_edges(buff=graph.vertices[0].width / 2)
        [e.set_opacity(0) for e in all_edges.values()]

        self.play(*[FadeIn(v) for v in graph.vertices.values()])

        nn_tour, nn_cost = get_nearest_neighbor_solution(graph.dist_matrix, start=0)
        nn_edges = get_edges_from_tour(nn_tour)
        nn_edges = list(map(lambda t: (t[1], t[0]) if t[0] > t[1] else t, nn_edges))

        cost_indicator = Text(
            f"{nn_cost:.2f}",
            font=REDUCIBLE_MONO,
            weight=BOLD,
        ).scale(0.4)

        best_cost_mob = Text(
            f"{nn_cost:.2f}",
            font=REDUCIBLE_MONO,
            weight=BOLD,
        ).scale(0.4)

        VGroup(cost_indicator_text, cost_indicator).arrange(RIGHT, buff=0.3).next_to(
            graph, DOWN, buff=0.5, aligned_edge=LEFT
        )
        VGroup(best_indicator_text, best_cost_mob).arrange(RIGHT, buff=0.3).next_to(
            graph, DOWN, buff=0.5, aligned_edge=RIGHT
        )

        self.play(
            # *self.focus_on_edges(nn_edges, all_edges),
            LaggedStart(*[all_edges[t].animate.set_opacity(1) for t in nn_edges]),
            FadeIn(cost_indicator),
            FadeIn(cost_indicator_text),
            FadeIn(best_indicator_text),
            FadeIn(best_cost_mob),
            run_time=3,
        )

        self.play(
            Write(axes_vg),
            Write(fake_y_axes_cost),
            Write(cost_labels),
            Write(cost_y_label),
            Write(legend),
        )
        tick = SVGMobject("check.svg").scale(0.1).set_color(REDUCIBLE_GREEN_LIGHTER)
        cross = Cross(stroke_color=REDUCIBLE_CHARM, stroke_width=2, scale_factor=0.1)

        # SIMULATION

        change_list = []
        cost_list = [nn_cost]
        last_temp = T.get_value()
        last_cost_mob = cost_indicator

        last_tour = nn_tour
        best_tour = nn_tour

        best_cost = nn_cost
        last_best_cost = best_cost

        for i in range(1, 100):
            v1, v2 = np.random.choice(range(N), size=2, replace=True)
            if i == 75:
                v1, v2 = 25, 20
            if i == 90:
                v1, v2 = 6, 2

            v = T.get_value()

            new_temp_line_chunk = self.next_iteration_line(
                axes_temp, i, T.get_value(), last_temp
            ).set_stroke(width=4)

            (
                last_tour,
                last_cost_mob,
                focus_edges_animations,
                cost_indicator_animations,
            ) = iterate_two_opt_animation(v1, v2, nn_tour, last_cost_mob)

            new_cost = get_cost_from_edges(
                get_edges_from_tour(last_tour), graph.dist_matrix
            )
            new_cost_line_chunk = (
                self.next_iteration_line(axes_cost, i, cost_list[-1], new_cost)
                .set_stroke(REDUCIBLE_PURPLE, width=4)
                .flip()
            )

            is_new_cost_better = new_cost < best_cost
            accept = np.random.random() < T.get_value()

            if is_new_cost_better:
                best_cost = new_cost
                best_tour = last_tour

                last_best_mob = best_cost_mob
                best_cost_mob = (
                    Text(
                        f"{best_cost:.2f}",
                        font=REDUCIBLE_MONO,
                        weight=BOLD,
                    )
                    .scale(0.4)
                    .next_to(best_indicator_text, RIGHT, buff=0.3)
                )

                cost_indicator_animations.extend(
                    [
                        FadeIn(best_cost_mob, shift=DOWN * 0.3),
                        FadeOut(last_best_mob, shift=DOWN * 0.3),
                    ]
                )

            new_best_line_chunk = (
                self.next_iteration_line(axes_cost, i, last_best_cost, best_cost)
                .set_stroke(REDUCIBLE_GREEN_LIGHTER, width=4)
                .flip()
            )

            undo_tour = best_tour

            next_change = self.create_change_history_entry(
                v1,
                v2,
                new_cost,
                best_cost,
            )
            cost_list.append(new_cost)

            mark = tick.copy() if accept or is_new_cost_better else cross.copy()
            next_change = (
                VGroup(next_change, mark)
                .arrange(RIGHT, buff=0.3)
                .move_to(change_history[0], aligned_edge=LEFT)
            )

            change_list.append(next_change)

            update_list_anims = []
            if len(change_list) <= len(change_history):
                update_list_anims = [
                    c.animate.move_to(change_history[i], aligned_edge=LEFT)
                    for i, c in enumerate(change_list[::-1])
                ]
                update_list_anims.append(FadeIn(next_change, shift=RIGHT * 0.3))

            # change_list will only have 1 element more than change_history
            elif len(change_list) > len(change_history):
                removed_element = change_list.pop(0)

                update_list_anims = [
                    c.animate.move_to(change_history[i], aligned_edge=LEFT)
                    for i, c in enumerate(change_list[::-1])
                ]

                update_list_anims.append(FadeOut(removed_element, shift=DOWN * 0.4))
                update_list_anims.append(FadeIn(next_change, shift=RIGHT * 0.3))

            # by this point we already created the animation that changes from the last
            # config to the next one. accepting just means doing nothing since it's
            # already done. rejecting means undoing the last operation.

            # if the last config is better than the previous one, we accept it right away
            if is_new_cost_better:
                pass
            # otherwise, we need to check if we should accept it
            else:
                if accept:
                    # accept worse solution
                    pass
                else:
                    # reject worse solution, undo last operation
                    undo_edges = get_edges_from_tour(undo_tour)
                    undo_cost = get_cost_from_edges(undo_edges, graph.dist_matrix)
                    focus_edges_animations = self.focus_on_edges(undo_edges, all_edges)

            if i < 10:
                edges_animations = self.focus_on_vertices([v1, v2], graph.vertices)
            else:
                edges_animations = self.focus_on_vertices([], graph.vertices)
            self.play(
                *update_list_anims,
                *focus_edges_animations,
                *cost_indicator_animations,
                *edges_animations,
                Write(new_temp_line_chunk),
                Write(new_cost_line_chunk),
                Write(new_best_line_chunk),
                T.animate.set_value(1 / (0.15 * i + 1)),
                run_time=1 / (0.15 * i + 1),
            )
            last_temp = v
            last_best_cost = best_cost

    ############### UTILS

    def next_iteration_line(
        self, axes: Axes, iteration: int, curr_temp: float, last_temp: float
    ):

        last_distance_p2c = axes.c2p(iteration - 1, last_temp)
        curr_distance_p2c = axes.c2p(iteration, curr_temp)

        return (
            Line(last_distance_p2c, curr_distance_p2c)
            .set_color(REDUCIBLE_YELLOW)
            .set_stroke(width=2)
        )

    def create_change_history_entry(self, v1, v2, cost, last_cost=None) -> Text:

        if last_cost:
            color = REDUCIBLE_CHARM if cost - last_cost > 0 else REDUCIBLE_GREEN_LIGHTER
        else:
            color = WHITE

        return Text(
            f"· Join {v1} with {v2}: {cost:.2f}",
            font=REDUCIBLE_FONT,
            t2f={f"{cost:.2f}": REDUCIBLE_MONO},
            t2c={f"{cost:.2f}": color},
            weight=BOLD,
        ).scale(0.4)


class TransitionTemplate(Scene):
    def construct(self):
        bg = ImageMobject("transition-bg.png").scale_to_fit_width(config.frame_width)
        self.add(bg)
        self.wait()

        NUM_TRANSITIONS = 8
        self.transition("TSP Problem Definition", 1, NUM_TRANSITIONS)
        self.wait()

        self.transition("Brute Force", 2, NUM_TRANSITIONS)
        self.wait()

        self.transition("Nearest Neighbor", 3, NUM_TRANSITIONS)
        self.wait()

        self.transition("Greedy Method", 4, NUM_TRANSITIONS)
        self.wait()

        self.transition("Christofides Method", 5, NUM_TRANSITIONS)
        self.wait()

        self.transition("Tour Improvement", 6, NUM_TRANSITIONS)
        self.wait()

        self.transition("Simulated Annealing", 7, NUM_TRANSITIONS)
        self.wait()

        self.transition("Ant Simulation", 8, NUM_TRANSITIONS)
        self.wait()

    def transition(self, transition_name, index, total):
        """
        Create transitions easily.

        - Transition name — string, self explanatory
        - Index correspond to the position of this transition on the video
        - Total corresponds to the total amount of transitions there will be

        Total will generate a number of nodes and index will highlight that specific
        node, showing the progress.
        """

        title = (
            Text(transition_name, font=REDUCIBLE_FONT, weight=BOLD)
            .set_stroke(BLACK, width=15, background=True)
            .scale_to_fit_width(config.frame_width - 3)
            .shift(UP)
        )

        nodes_and_lines = VGroup()
        for n in range(1, total + 1):
            if n == index:
                node = (
                    Circle()
                    .scale(0.2)
                    .set_stroke(REDUCIBLE_YELLOW)
                    .set_fill(REDUCIBLE_YELLOW_DARKER, opacity=1)
                )
                nodes_and_lines.add(node)
            else:
                nodes_and_lines.add(
                    Circle()
                    .scale(0.2)
                    .set_stroke(REDUCIBLE_PURPLE)
                    .set_fill(REDUCIBLE_PURPLE_DARK_FILL, opacity=1)
                )

            nodes_and_lines.add(Line().set_color(REDUCIBLE_PURPLE))
        nodes_and_lines.remove(nodes_and_lines[-1])

        nodes_and_lines.arrange(RIGHT, buff=0.5).scale_to_fit_width(
            config.frame_width - 5
        ).to_edge(DOWN, buff=1)

        self.play(
            FadeIn(title, shift=UP * 0.3), LaggedStartMap(FadeIn, nodes_and_lines)
        )

        self.wait()
        self.play(FadeOut(title), FadeOut(nodes_and_lines))


class Introduction(TransitionOtherApproaches):
    def construct(self):

        bg = ImageMobject("usa-map-clean.png").scale_to_fit_width(config.frame_width)
        dark_filter = ScreenRectangle(height=10).set_fill(BLACK, opacity=0.2)
        self.play(FadeIn(bg), FadeIn(dark_filter))
        self.wait()

        # self.add(
        #     Axes(
        #         axis_config={"include_numbers": True},
        #         x_length=config.frame_width,
        #         y_length=config.frame_height,
        #         tips=False,
        #     )
        # )

        city_coords = [
            (-6.1, -0.25, 0),  # SF
            (-5.8, 0.55, 0),  # san jose
            (-4.999, -1.45, 0),  # LA
            (0.45, 0.85, 0),  # Atlanta
            (-0.35, -0.35, 0),  # Miami
            (-4.35, 1.75, 0),  # Miami
            (-1.75, 1.35, 0),  # Denver
            (3.5, -0.5, 0),  # Atlanta
            (-4.3, -0.77, 0),  # San diego
            (-3.35, -1, 0),  # phoenix
            (0.45, -0.85, 0),  # dallas
            (0.25, -1.8, 0),  # san antonio
            (0.6, -2.65, 0),  # houston
            (2.64, 1, 0),  # chicago
            (6.5, 1, 0),  # new york
            (5.85, 0.8, 0),  # philadelphia
        ]

        total_number_of_nodes = len(city_coords)

        graph = TSPGraph(
            range(total_number_of_nodes),
            vertex_config={
                "fill_opacity": 1,
                "fill_color": REDUCIBLE_PURPLE_DARK_FILL,
                "stroke_color": REDUCIBLE_VIOLET,
                "stroke_width": 3,
            },
            labels=False,
            layout=self.get_city_layout(core_cities=city_coords, nodes_per_core=0),
        )
        print(f"> Finished building graph with {total_number_of_nodes} nodes")

        print(f"Calculating best tour")
        # best_perm, best_cost = get_exact_tsp_solution(graph.dist_matrix)
        best_perm = [0, 1, 5, 6, 3, 13, 14, 15, 7, 12, 11, 10, 4, 9, 8, 2]
        print(f"Best tour calculated: {best_perm}")

        nn_perm, nn_cost = get_nearest_neighbor_solution(graph.dist_matrix, 0)

        # self.play(FadeIn(graph[0]))
        sf_tag = (
            Text("SF", font=REDUCIBLE_FONT, weight=BOLD)
            .scale(0.4)
            .next_to(graph[0], UP, buff=0.3)
            .set_stroke(BLACK, width=5, background=True)
        )

        self.play(
            LaggedStart(
                *[FadeIn(v, scale=0.95) for v in list(graph.vertices.values())],
                lag_ratio=0.5,
            ),
            run_time=3,
        )
        self.wait()

        self.play(FadeIn(sf_tag, shift=UP * 0.3))
        self.wait()

        self.play(FadeOut(sf_tag, shift=UP * 0.3))
        self.wait()

        all_edges = graph.get_all_edges(buff=graph.vertices[0].width / 2)
        print(f"> Finished getting all {len(all_edges)} edges")
        [e.set_opacity(0) for e in all_edges.values()]

        # generate all possible tours (until max cap) and try and get a good variety of tours

        all_tour_perms = get_all_tour_permutations(
            total_number_of_nodes, 0, max_cap=10000
        )

        tour_perms = list(
            filter(lambda t: len(t) == total_number_of_nodes, all_tour_perms)
        )
        np.random.shuffle(tour_perms)

        print(f"tour perms length: {len(tour_perms)}")

        # randomizing a bit more
        edges_perms = [swap_random(get_edges_from_tour(t)) for t in tour_perms]

        print(f"> Finished calculating all {len(edges_perms)} tuples from every tour")

        tsp_title = (
            Text(
                "Traveling Salesman Problem",
                font=REDUCIBLE_FONT,
                weight=BOLD,
                font_size=400,
            )
            .scale_to_fit_width(config.frame_width - 5)
            .to_edge(DOWN)
            .set_stroke(width=8, background=True)
        )

        # by changing the slice size you can create a longer or shorter example
        example_slice = edges_perms[:30]
        example_slice.append(get_edges_from_tour(swap_random(nn_perm, 2)))
        print(len(example_slice))

        for i, tour_edges in enumerate(example_slice):
            # the all_edges dict only stores the edges in ascending order
            tour_edges = list(
                map(lambda x: x if x[0] < x[1] else (x[1], x[0]), tour_edges)
            )

            if i == 0:
                anims = LaggedStart(
                    *[Write(all_edges[e].set_opacity(1)) for e in tour_edges]
                )
                self.play(
                    anims,
                    run_time=1 / (0.5 * i + 1),
                )
                self.wait()

            else:
                anims = self.focus_on_edges(
                    tour_edges,
                    all_edges=all_edges,
                )

                self.play(
                    *anims,
                    run_time=1 / (0.5 * i + 1),
                )

        best_edges = get_edges_from_tour(best_perm)
        best_edges = list(map(lambda e: (e[1], e[0]) if e[0] > e[1] else e, best_edges))

        nn_edges = get_edges_from_tour(nn_perm)
        nn_edges = list(map(lambda e: (e[1], e[0]) if e[0] > e[1] else e, nn_edges))

        frame = self.camera.frame

        self.wait()
        self.play(
            FadeIn(tsp_title, scale=1.05),
            run_time=2,
        )

        self.wait()
        self.play(frame.animate.scale(0.95), FadeOut(tsp_title))
        self.play(*self.focus_on_edges(nn_edges, all_edges))

        self.wait()
        self.play(*self.focus_on_edges(best_edges, all_edges))

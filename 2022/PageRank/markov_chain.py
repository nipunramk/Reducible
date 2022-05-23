import sys

### THIS IS A WORKAROUND FOR NOW
### IT REQUIRES RUNNING MANIM FROM INSIDE DIRECTORY VIDEO-DEV
sys.path.insert(1, "common/")

from manim import *
from manim.mobject.geometry.tips import ArrowTriangleFilledTip
from reducible_colors import *
from functions import *

from typing import Hashable, Iterable

import numpy as np
import itertools as it

np.random.seed(23)


class MarkovChain:
    def __init__(
        self,
        states: int,
        edges: list[tuple[int, int]],
        transition_matrix=None,
        dist=None,
    ):
        """
        @param: states -- number of states in Markov Chain
        @param: edges -- list of tuples (u, v) for a directed edge u to v, u in range(0, states), v in range(0, states)
        @param: transition_matrix -- custom np.ndarray matrix of transition probabilities for all states in Markov chain
        @param: dist -- initial distribution across states, assumed to be uniform if none
        """
        self.states = range(states)
        self.edges = edges
        self.adj_list = {}
        for state in self.states:
            self.adj_list[state] = []
            for u, v in edges:
                if u == state:
                    self.adj_list[state].append(v)

        if transition_matrix is not None:
            self.transition_matrix = transition_matrix
        else:
            # Assume default transition matrix is uniform across all outgoing edges
            self.transition_matrix = np.zeros((states, states))
            for state in self.states:
                neighbors = self.adj_list[state]
                for neighbor in neighbors:
                    self.transition_matrix[state][neighbor] = 1 / len(neighbors)

        # handle sink nodes to point to itself
        for i, row in enumerate(self.transition_matrix):
            if np.sum(row) == 0:
                self.transition_matrix[i][i] = 1

        if dist is not None:
            self.dist = dist
        else:
            self.dist = np.array(
                [1 / len(self.states) for _ in range(len(self.states))]
            )

        self.starting_dist = self.dist

    def get_states(self):
        return list(self.states)

    def get_edges(self):
        return self.edges

    def get_adjacency_list(self):
        return self.adj_list

    def get_transition_matrix(self):
        return self.transition_matrix

    def get_current_dist(self):
        return self.dist

    def update_dist(self):
        """
        Performs one step of the markov chain
        """
        self.dist = np.dot(self.dist, self.transition_matrix)

    def get_true_stationary_dist(self):
        dist = np.linalg.eig(np.transpose(self.transition_matrix))[1][:, 0]
        return dist / sum(dist)

    def set_starting_dist(self, starting_dist):
        self.starting_dist = starting_dist
        self.dist = starting_dist

    def get_starting_dist(self):
        return self.starting_dist

    def set_transition_matrix(self, transition_matrix):
        self.transition_matrix = transition_matrix

class CustomLabel(Text):
    def __init__(self, label, font="SF Mono", scale=1, weight=BOLD):
        super().__init__(label, font=font, weight=weight)
        self.scale(scale)


class CustomCurvedArrow(CurvedArrow):
    def __init__(self, start, end, tip_length=0.15, **kwargs):
        super().__init__(start, end, **kwargs)
        self.pop_tips()
        self.add_tip(
            tip_shape=ArrowTriangleFilledTip,
            tip_length=tip_length,
            at_start=False,
        )
        self.tip.z_index = -100

    def set_opacity(self, opacity, family=True):
        return super().set_opacity(opacity, family)

    @override_animate(set_opacity)
    def _set_opacity_animation(self, opacity=1, anim_args=None):
        if anim_args is None:
            anim_args = {}

        animate_stroke = self.animate.set_stroke(opacity=opacity)
        animate_tip = self.tip.animate.set_opacity(opacity)

        return AnimationGroup(*[animate_stroke, animate_tip])


class MarkovChainGraph(Graph):
    def __init__(
        self,
        markov_chain: MarkovChain,
        vertex_config={
            "stroke_color": REDUCIBLE_PURPLE,
            "stroke_width": 3,
            "fill_color": REDUCIBLE_PURPLE,
            "fill_opacity": 0.5,
        },
        curved_edge_config: dict = None,
        straight_edge_config: dict = None,
        enable_curved_double_arrows=True,
        labels=True,
        state_color_map=None,
        **kwargs,
    ):
        self.markov_chain = markov_chain
        self.enable_curved_double_arrows = enable_curved_double_arrows

        self.default_curved_edge_config = {
            "color": REDUCIBLE_VIOLET,
            "stroke_width": 3,
            "radius": 4,
        }

        self.default_straight_edge_config = {
            "color": REDUCIBLE_VIOLET,
            "max_tip_length_to_length_ratio": 0.06,
            "stroke_width": 3,
        }
        self.state_color_map = state_color_map

        if labels:
            labels = {
                k: CustomLabel(str(k), scale=0.6) for k in markov_chain.get_states()
            }
        
        if self.state_color_map:
            new_vertex_config = {}
            for state in markov_chain.get_states():
                new_vertex_config[state] = vertex_config.copy()
                new_vertex_config[state]["stroke_color"] = self.state_color_map[state]
                new_vertex_config[state]["fill_color"] = self.state_color_map[state]

            vertex_config = new_vertex_config

        self.labels = {}

        super().__init__(
            markov_chain.get_states(),
            markov_chain.get_edges(),
            vertex_config=vertex_config,
            labels=labels,
            **kwargs,
        )

        self._graph = self._graph.to_directed()
        self.remove_edges(*self.edges)

        self.add_markov_chain_edges(
            *markov_chain.get_edges(),
            straight_edge_config=straight_edge_config,
            curved_edge_config=curved_edge_config,
        )

        self.clear_updaters()
        # this updater makes sure the edges remain connected
        # even when states move around
        def update_edges(graph):
            for (u, v), edge in graph.edges.items():
                v_c = self.vertices[v].get_center()
                u_c = self.vertices[u].get_center()
                vec = v_c - u_c
                unit_vec = vec / np.linalg.norm(vec)

                u_radius = self.vertices[u].width / 2
                v_radius = self.vertices[v].width / 2

                arrow_start = u_c + unit_vec * u_radius
                arrow_end = v_c - unit_vec * v_radius
                edge.put_start_and_end_on(arrow_start, arrow_end)

        self.add_updater(update_edges)
        update_edges(self)

    def add_edge_buff(
        self,
        edge: tuple[Hashable, Hashable],
        edge_type: type[Mobject] = None,
        edge_config: dict = None,
    ):
        """
        Custom function to add edges to our Markov Chain,
        making sure the arrowheads land properly on the states.
        """
        if edge_config is None:
            edge_config = self.default_edge_config.copy()
        added_mobjects = []
        for v in edge:
            if v not in self.vertices:
                added_mobjects.append(self._add_vertex(v))
        u, v = edge

        self._graph.add_edge(u, v)

        base_edge_config = self.default_edge_config.copy()
        base_edge_config.update(edge_config)
        edge_config = base_edge_config
        self._edge_config[(u, v)] = edge_config

        v_c = self.vertices[v].get_center()
        u_c = self.vertices[u].get_center()
        vec = v_c - u_c
        unit_vec = vec / np.linalg.norm(vec)

        if self.enable_curved_double_arrows:
            arrow_start = u_c + unit_vec * self.vertices[u].radius
            arrow_end = v_c - unit_vec * self.vertices[v].radius
        else:
            arrow_start = u_c
            arrow_end = v_c
            edge_config["buff"] = self.vertices[u].radius

        edge_mobject = edge_type(
            start=arrow_start, end=arrow_end, z_index=-100, **edge_config
        )
        self.edges[(u, v)] = edge_mobject

        self.add(edge_mobject)
        added_mobjects.append(edge_mobject)
        return self.get_group_class()(*added_mobjects)

    def add_markov_chain_edges(
        self,
        *edges: tuple[Hashable, Hashable],
        curved_edge_config: dict = None,
        straight_edge_config: dict = None,
        **kwargs,
    ):
        """
        Custom function for our specific case of Markov Chains.
        This function aims to make double arrows curved when two nodes
        point to each other, leaving the other ones straight.
        Parameters
        ----------
        - edges: a list of tuples connecting states of the Markov Chain
        - curved_edge_config: a dictionary specifying the configuration
        for CurvedArrows, if any
        - straight_edge_config: a dictionary specifying the configuration
        for Arrows
        """

        if curved_edge_config is not None:
            curved_config_copy = self.default_curved_edge_config.copy()
            curved_config_copy.update(curved_edge_config)
            curved_edge_config = curved_config_copy
        else:
            curved_edge_config = self.default_curved_edge_config.copy()

        if straight_edge_config is not None:
            straight_config_copy = self.default_straight_edge_config.copy()
            straight_config_copy.update(straight_edge_config)
            straight_edge_config = straight_config_copy
        else:
            straight_edge_config = self.default_straight_edge_config.copy()

        print(straight_edge_config)

        edge_vertices = set(it.chain(*edges))
        new_vertices = [v for v in edge_vertices if v not in self.vertices]
        added_vertices = self.add_vertices(*new_vertices, **kwargs)

        edge_types_dict = {}
        for e in edges:
            if self.enable_curved_double_arrows and (e[1], e[0]) in edges:
                edge_types_dict.update({e: (CustomCurvedArrow, curved_edge_config)})

            else:
                edge_types_dict.update({e: (Arrow, straight_edge_config)})

        added_mobjects = sum(
            (
                self.add_edge_buff(
                    edge,
                    edge_type=e_type_and_config[0],
                    edge_config=e_type_and_config[1],
                ).submobjects
                for edge, e_type_and_config in edge_types_dict.items()
            ),
            added_vertices,
        )

        return self.get_group_class()(*added_mobjects)

    def get_transition_labels(self, scale=0.3, round_val=True):
        """
        This function returns a VGroup with the probability that each
        each state has to transition to another state, based on the
        Chain's transition matrix.
        It essentially takes each edge's probability and creates a label to put
        on top of it, for easier indication and explanation.
        This function returns the labels already set up in a VGroup, ready to just
        be created.
        """
        tm = self.markov_chain.get_transition_matrix()

        labels = VGroup()
        for s in range(len(tm)):

            for e in range(len(tm[0])):
                if s != e and tm[s, e] != 0:

                    edge_tuple = (s, e)
                    matrix_prob = tm[s, e]

                    if round_val and round(matrix_prob, 2) != matrix_prob:
                        matrix_prob = round(matrix_prob, 2)

                    label = (
                        Text(str(matrix_prob), font=REDUCIBLE_MONO)
                        .set_stroke(BLACK, width=8, background=True, opacity=0.8)
                        .scale(scale)
                        .move_to(self.edges[edge_tuple].point_from_proportion(0.2))
                    )

                    labels.add(label)
                    self.labels[edge_tuple] = label

        def update_labels(graph):
            for e, l in graph.labels.items():
                l.move_to(graph.edges[e].point_from_proportion(0.2))

        self.add_updater(update_labels)

        return labels


class MarkovChainSimulator:
    def __init__(
        self,
        markov_chain: MarkovChain,
        markov_chain_g: MarkovChainGraph,
        num_users=50,
        user_radius=0.035,
    ):
        self.markov_chain = markov_chain
        self.markov_chain_g = markov_chain_g
        self.num_users = num_users
        self.state_counts = {i: 0 for i in markov_chain.get_states()}
        self.user_radius = user_radius
        self.distribution_sequence = []
        self.init_users()

    def init_users(self):
        self.user_to_state = {
            i: np.random.choice(
                self.markov_chain.get_states(), p=self.markov_chain.get_current_dist()
            )
            for i in range(self.num_users)
        }
        for user_id in self.user_to_state:
            self.state_counts[self.user_to_state[user_id]] += 1

        self.users = [
            Dot(radius=self.user_radius)
            .set_color(REDUCIBLE_YELLOW)
            .set_opacity(0.6)
            .set_stroke(REDUCIBLE_YELLOW, width=2, opacity=0.8)
            for _ in range(self.num_users)
        ]

        for user_id, user in enumerate(self.users):
            user_location = self.get_user_location(user_id)
            user.move_to(user_location)

        self.distribution_sequence.append(self.markov_chain.get_current_dist())

    def get_user_location(self, user: int):
        user_state = self.user_to_state[user]
        user_location = self.markov_chain_g.vertices[user_state].get_center()
        distributed_point = self.poisson_distribution(user_location)

        user_location = [distributed_point[0], distributed_point[1], 0.0]

        return user_location

    def get_users(self):
        return self.users

    def transition(self):
        for user_id in self.user_to_state:
            self.user_to_state[user_id] = self.update_state(user_id)
        self.markov_chain.update_dist()
        self.distribution_sequence.append(self.markov_chain.get_current_dist())

    def update_state(self, user_id: int):
        current_state = self.user_to_state[user_id]
        transition_matrix = self.markov_chain.get_transition_matrix()
        new_state = np.random.choice(
            self.markov_chain.get_states(), p=transition_matrix[current_state]
        )
        self.state_counts[new_state] += 1
        return new_state

    def get_state_counts(self):
        return self.state_counts

    def get_user_dist(self, round_val=False):
        dist = {}
        total_counts = sum(self.state_counts.values())
        for user_id, count in self.state_counts.items():
            dist[user_id] = self.state_counts[user_id] / total_counts
            if round_val:
                dist[user_id] = round(dist[user_id], 2)
        return dist

    def get_instant_transition_animations(self):
        transition_animations = []
        self.transition()
        for user_id, user in enumerate(self.users):
            new_location = self.get_user_location(user_id)
            transition_animations.append(user.animate.move_to(new_location))
        return transition_animations

    def get_lagged_smooth_transition_animations(self):
        transition_map = {i: [] for i in self.markov_chain.get_states()}
        self.transition()
        for user_id, user in enumerate(self.users):
            new_location = self.get_user_location(user_id)
            transition_map[self.user_to_state[user_id]].append(
                user.animate.move_to(new_location)
            )
        return transition_map

    def poisson_distribution(self, center):
        """
        This function creates a poisson distribution that places
        users around the center of the given state,
        particularly across the state's stroke.
        Implementation taken from: https://github.com/hpaulkeeler/posts/blob/master/PoissonCircle/PoissonCircle.py
        """

        radius = self.markov_chain_g.vertices[0].width / 2

        xxRand = np.random.normal(0, 1, size=(1, 2))

        # generate two sets of normal variables
        normRand = np.linalg.norm(xxRand, 2, 1)

        # Euclidean norms
        xxRandBall = xxRand / normRand[:, None]

        # rescale by Euclidean norms
        xxRandBall = radius * xxRandBall

        # rescale for non-unit sphere
        # retrieve x and y coordinates
        xx = xxRandBall[:, 0]
        yy = xxRandBall[:, 1]

        # Shift centre of circle to (xx0,yy0)
        xx = xx + center[0]
        yy = yy + center[1]

        return (xx[0], yy[0])

    def get_state_to_user(self):
        state_to_users = {}
        for user_id, state in self.user_to_state.items():
            if state not in state_to_users:
                state_to_users[state] = [user_id]
            else:
                state_to_users[state].append(user_id)
        return state_to_users

    def get_distribution_sequence(self):
        return self.distribution_sequence

class MarkovChainTester(Scene):
    def construct(self):
        markov_chain = MarkovChain(
            4,
            [(0, 1), (1, 0), (0, 2), (1, 2), (1, 3), (2, 3), (3, 1)],
        )
        print(markov_chain.get_states())
        print(markov_chain.get_edges())
        print(markov_chain.get_current_dist())
        print(markov_chain.get_adjacency_list())
        print(markov_chain.get_transition_matrix())

        markov_chain_g = MarkovChainGraph(
            markov_chain, enable_curved_double_arrows=True
        )
        markov_chain_t_labels = markov_chain_g.get_transition_labels()
        self.play(FadeIn(markov_chain_g), FadeIn(markov_chain_t_labels))
        markov_chain_t_labels = markov_chain_g.get_transition_labels()
        self.play(FadeIn(markov_chain_g), FadeIn(markov_chain_t_labels))
        self.wait()

        markov_chain_sim = MarkovChainSimulator(
            markov_chain, markov_chain_g, num_users=50
        )
        users = markov_chain_sim.get_users()

        self.play(*[FadeIn(user) for user in users])
        self.wait()

        num_steps = 10
        for _ in range(num_steps):
            transition_animations = markov_chain_sim.get_instant_transition_animations()
            self.play(*transition_animations)
        self.wait()

        for _ in range(num_steps):
            transition_map = markov_chain_sim.get_lagged_smooth_transition_animations()
            self.play(
                *[LaggedStart(*transition_map[i]) for i in markov_chain.get_states()]
            )
            self.wait()


### BEGIN INTRODUCTION.mp4 ###
class IntroWebGraph(Scene):
    def construct(self):
        web_markov_chain, web_graph = self.get_web_graph()
        self.add(web_graph)
        self.wait()

    def get_web_graph(self):
        graph_layout = self.get_web_graph_layout()
        graph_edges = self.get_web_graph_edges(graph_layout)

        print(len(graph_layout))
        markov_chain = MarkovChain(len(graph_layout), graph_edges)
        markov_chain_g = MarkovChainGraph(
            markov_chain,
            enable_curved_double_arrows=False,
            labels=False,
            layout=graph_layout,
        )

        return markov_chain, markov_chain_g

    def get_web_graph_layout(self):
        grid_height = 8
        grid_width = 12

        layout = {}
        node_id = 0
        STEP = 0.5
        for i in np.arange(-grid_height // 2, grid_height // 2, STEP):
            for j in np.arange(-grid_width // 2, grid_width // 2, STEP):
                noise = RIGHT * np.random.uniform(-1, 1) + UP * np.random.uniform(-1, 1)
                layout[node_id] = UP * i + RIGHT * j + noise * STEP / 3.1
                node_id += 1

        return layout

    def get_web_graph_edges(self, graph_layout):
        edges = []
        for u in graph_layout:
            for v in graph_layout:
                if u != v and np.linalg.norm(graph_layout[v] - graph_layout[u]) < 0.8:
                    if np.random.uniform() < 0.7:
                        edges.append((u, v))
        return edges


class UserSimulationWebGraph(IntroWebGraph):
    def construct(self):
        web_markov_chain, web_graph = self.get_web_graph()
        self.add(web_graph)
        self.wait()
        self.start_simulation(web_markov_chain, web_graph)

    def start_simulation(self, markov_chain, markov_chain_g):
        markov_chain_sim = MarkovChainSimulator(
            markov_chain,
            markov_chain_g,
            num_users=3000,
            user_radius=0.01,
        )
        users = markov_chain_sim.get_users()

        self.add(*users)
        self.wait()

        num_steps = 50

        for _ in range(num_steps):
            print('Nipun: iteration', _)
            transforms = markov_chain_sim.get_instant_transition_animations()
            self.play(
                *transforms, rate_func=linear,
            )

        # for _ in range(num_steps):
        #     transition_map = markov_chain_sim.get_lagged_smooth_transition_animations()
        #     self.play(
        #         *[
        #             LaggedStart(*transition_map[i], rate_func=linear)
        #             for i in markov_chain.get_states()
        #         ],
        #         rate_func=linear,
        #     )


class MarkovChainPageRankTitleCard(Scene):
    def construct(self):
        title = Text("Markov Chains", font="CMU Serif", weight=BOLD).move_to(UP * 3.5)
        title.set_stroke(BLACK, width=8, background=True, opacity=0.8)
        self.play(Write(title))
        self.wait()

        pagerank_title = Text("PageRank", font="CMU Serif", weight=BOLD).move_to(
            UP * 3.5
        ).set_stroke(BLACK, width=8, background=True, opacity=0.8)

        self.play(ReplacementTransform(title, pagerank_title))
        self.wait()


### END INTRODUCTION.mp4 ###


class MarkovChainIntroEdited(Scene):
    def construct(self):
        markov_chain = MarkovChain(
            4,
            [(0, 1), (1, 0), (0, 2), (1, 2), (1, 3), (2, 3), (3, 1), (3, 2)],
        )

        markov_chain_g = MarkovChainGraph(
            markov_chain, enable_curved_double_arrows=True
        )
        markov_chain_g.clear_updaters()
        markov_chain_g.scale(1.2)
        markov_chain_t_labels = markov_chain_g.get_transition_labels()

        self.play(FadeIn(markov_chain_g), FadeIn(markov_chain_t_labels))
        self.wait()

        self.highlight_states(markov_chain_g)

        transition_probs = self.highlight_transitions(markov_chain_g)

        self.highlight_edge(markov_chain_g, (3, 1))
        p_3_1 = Tex(r"$P(3, 1)$ = 0.5").scale(0.8)
        p_2_3 = Tex(r"$P(2, 3)$ = 1.0").scale(0.8)
        VGroup(p_3_1, p_2_3).arrange(DOWN).move_to(LEFT * 4)

        self.play(FadeIn(p_3_1))
        self.wait()

        self.highlight_edge(markov_chain_g, (2, 3))

        self.play(FadeIn(p_2_3))
        self.wait()

        reset_animations = self.reset_edges(markov_chain_g)
        self.play(
            *reset_animations, FadeOut(p_3_1), FadeOut(p_2_3), FadeOut(transition_probs)
        )
        self.wait()

        self.highlight_edges(markov_chain_g, [(1, 3), (1, 2), (1, 0)])
        self.wait()

        new_edges_to_labels = [[(1, 3), 0.2], [(1, 2), 0.7], [(1, 0), 0.1]]
        new_labels = [self.get_label(markov_chain_g.edges[edge], prob).set_color(REDUCIBLE_YELLOW) for edge, prob in new_edges_to_labels]

        self.play(
            *[Transform(markov_chain_g.labels[edge], new_label) for edge, new_label in zip([(1, 3), (1, 2), (1, 0)], new_labels)]
        )
        self.wait()
        # self.discuss_markov_prop(markov_chain_g)

    def get_label(self, edge, prob, scale=0.3):
        return (
            Text(str(prob), font=REDUCIBLE_MONO)
            .set_stroke(BLACK, width=8, background=True, opacity=0.8)
            .scale(scale)
            .move_to(edge.point_from_proportion(0.15))
        )

    def highlight_states(self, markov_chain_g):
        highlight_animations = []
        for edge in markov_chain_g.edges.values():
            highlight_animations.append(edge.animate.set_stroke(opacity=0.5))
            highlight_animations.append(
                edge.tip.animate.set_fill(opacity=0.5).set_stroke(opacity=0.5)
            )
        for label in markov_chain_g.labels.values():
            highlight_animations.append(label.animate.set_fill(opacity=0.5))
        glowing_circles = []
        for vertex in markov_chain_g.vertices.values():
            glowing_circle = get_glowing_surround_circle(vertex)
            highlight_animations.append(FadeIn(glowing_circle))
            glowing_circles.append(glowing_circle)

        states = (
            Text("States", font="CMU Serif")
            .move_to(UP * 3.5)
            .set_color(REDUCIBLE_YELLOW)
        )
        arrow_1 = Arrow(states.get_bottom(), markov_chain_g.vertices[2])
        arrow_2 = Arrow(states.get_bottom(), markov_chain_g.vertices[0])
        arrow_1.set_color(GRAY)
        arrow_2.set_color(GRAY)

        self.play(
            *highlight_animations,
        )
        self.wait()

        self.play(Write(states), Write(arrow_1), Write(arrow_2))
        self.wait()

        un_highlight_animations = []
        for edge in markov_chain_g.edges.values():
            un_highlight_animations.append(edge.animate.set_stroke(opacity=1))
            un_highlight_animations.append(
                edge.tip.animate.set_fill(opacity=1).set_stroke(opacity=1)
            )
        for label in markov_chain_g.labels.values():
            un_highlight_animations.append(label.animate.set_fill(opacity=1))

        for v in markov_chain_g.vertices:
            un_highlight_animations.append(FadeOut(glowing_circles[v]))

        self.play(
            *un_highlight_animations,
            FadeOut(states),
            FadeOut(arrow_1),
            FadeOut(arrow_2),
        )
        self.wait()

    def highlight_transitions(self, markov_chain_g):
        self.play(
            *[
                label.animate.set_color(REDUCIBLE_YELLOW)
                for label in markov_chain_g.labels.values()
            ]
        )
        self.wait()

        transition_probs = Tex("Transition Probabilities $P(i, j)$").set_color(
            REDUCIBLE_YELLOW
        )
        transition_probs.move_to(UP * 3.5)
        self.play(FadeIn(transition_probs))
        self.wait()

        return transition_probs

    def highlight_edges(self, markov_chain_g, edges_to_highlight):
        highlight_animations = []
        for edge in markov_chain_g.edges:
            if edge in edges_to_highlight:
                highlight_animations.extend(
                    [
                        markov_chain_g.edges[edge].animate.set_stroke(opacity=1),
                        markov_chain_g.edges[edge]
                        .tip.animate.set_stroke(opacity=1)
                        .set_fill(opacity=1),
                        markov_chain_g.labels[edge].animate.set_fill(
                            color=REDUCIBLE_YELLOW, opacity=1
                        ),
                    ]
                )
            else:
                highlight_animations.extend(
                    [
                        markov_chain_g.edges[edge].animate.set_stroke(opacity=0.3),
                        markov_chain_g.edges[edge]
                        .tip.animate.set_stroke(opacity=0.3)
                        .set_fill(opacity=0.3),
                        markov_chain_g.labels[edge].animate.set_fill(
                            color=WHITE, opacity=0.3
                        ),
                    ]
                )
        self.play(*highlight_animations)

    def highlight_edge(self, markov_chain_g, edge_tuple):
        highlight_animations = []
        for edge in markov_chain_g.edges:
            if edge == edge_tuple:
                highlight_animations.extend(
                    [
                        markov_chain_g.edges[edge].animate.set_stroke(opacity=1),
                        markov_chain_g.edges[edge]
                        .tip.animate.set_stroke(opacity=1)
                        .set_fill(opacity=1),
                        markov_chain_g.labels[edge].animate.set_fill(
                            color=REDUCIBLE_YELLOW, opacity=1
                        ),
                    ]
                )
            else:
                highlight_animations.extend(
                    [
                        markov_chain_g.edges[edge].animate.set_stroke(opacity=0.3),
                        markov_chain_g.edges[edge]
                        .tip.animate.set_stroke(opacity=0.3)
                        .set_fill(opacity=0.3),
                        markov_chain_g.labels[edge].animate.set_fill(
                            color=WHITE, opacity=0.3
                        ),
                    ]
                )
        self.play(*highlight_animations)

    def reset_edges(self, markov_chain_g):
        un_highlight_animations = []
        for edge in markov_chain_g.edges.values():
            un_highlight_animations.append(edge.animate.set_stroke(opacity=1))
            un_highlight_animations.append(
                edge.tip.animate.set_fill(opacity=1).set_stroke(opacity=1)
            )
        for label in markov_chain_g.labels.values():
            un_highlight_animations.append(
                label.animate.set_fill(color=WHITE, opacity=1)
            )
        return un_highlight_animations

    def discuss_markov_prop(self, markov_chain_g):
        markov_prop_explained = Tex(
            "Transition probability only depends \\\\ on current state and future state"
        ).scale(0.8)
        markov_prop_explained.move_to(UP * 3.5)

        self.play(FadeIn(markov_prop_explained))
        self.wait()

        user_1 = (
            Dot()
            .set_color(REDUCIBLE_GREEN_DARKER)
            .set_stroke(color=REDUCIBLE_GREEN_LIGHTER, width=2)
        )
        user_2 = (
            Dot()
            .set_color(REDUCIBLE_YELLOW_DARKER)
            .set_stroke(color=REDUCIBLE_YELLOW, width=2)
        )

        user_1_label = user_1.copy()
        user_1_transition = MathTex(r"2 \rightarrow 3").scale(0.7)
        user_1_label_trans = VGroup(user_1_label, user_1_transition).arrange(RIGHT)
        user_2_label = user_2.copy()
        user_2_transition = MathTex(r"1 \rightarrow 3").scale(0.7)
        user_2_label_trans = VGroup(user_2_label, user_2_transition).arrange(RIGHT)

        result = Tex("For both users").scale(0.7)
        result_with_dots = VGroup(
            result, user_1_label.copy(), user_2_label.copy()
        ).arrange(RIGHT)
        p_3_1 = Tex(r"$P(3, 1)$ = 0.5").scale(0.7)
        p_3_2 = Tex(r"$P(3, 2)$ = 0.5").scale(0.7)

        left_text = (
            VGroup(
                user_1_label_trans, user_2_label_trans, result_with_dots, p_3_1, p_3_2
            )
            .arrange(DOWN)
            .to_edge(LEFT * 2)
        )
        user_1.next_to(markov_chain_g.vertices[2], LEFT, buff=SMALL_BUFF)
        user_2.next_to(markov_chain_g.vertices[1], DOWN, buff=SMALL_BUFF)

        self.play(
            FadeIn(user_1),
        )
        self.wait()
        self.play(FadeIn(user_1_label_trans))
        self.wait()

        self.play(
            user_1.animate.next_to(markov_chain_g.vertices[3], LEFT, buff=SMALL_BUFF)
        )
        self.wait()

        self.play(FadeIn(user_2), FadeIn(user_2_label_trans))
        self.wait()

        self.play(
            user_2.animate.next_to(markov_chain_g.vertices[3], DOWN, buff=SMALL_BUFF)
        )
        self.wait()

        self.play(FadeIn(result_with_dots))
        self.wait()
        highlight_animations = []

        for edge in markov_chain_g.edges:
            if edge == (3, 2) or edge == (3, 1):
                highlight_animations.extend(
                    [markov_chain_g.labels[edge].animate.set_color(REDUCIBLE_YELLOW)]
                )
            else:
                highlight_animations.extend(
                    [
                        markov_chain_g.labels[edge].animate.set_fill(opacity=0.3),
                        markov_chain_g.edges[edge].animate.set_stroke(opacity=0.3),
                        markov_chain_g.edges[edge]
                        .tip.animate.set_fill(opacity=0.3)
                        .set_stroke(opacity=0.3),
                    ]
                )

        self.play(FadeIn(p_3_1), FadeIn(p_3_2), *highlight_animations)
        self.wait()

        markov_property = (
            Text("Markov Property", font="CMU Serif", weight=BOLD)
            .scale(0.8)
            .move_to(DOWN * 3.5)
        )

        self.play(Write(markov_property))
        self.wait()


class IntroImportanceProblem(Scene):
    def construct(self):
        title = Text("Ranking States", font="CMU Serif", weight=BOLD)
        title.move_to(UP * 3.5)

        self.play(Write(title))
        self.wait()

        markov_chain = MarkovChain(
            4,
            [(0, 1), (1, 0), (0, 2), (1, 2), (1, 3), (2, 3), (3, 1), (3, 2)],
        )

        markov_chain_g = MarkovChainGraph(
            markov_chain, enable_curved_double_arrows=True, layout="circular"
        )
        markov_chain_g.scale(1.1)
        markov_chain_t_labels = markov_chain_g.get_transition_labels()

        self.play(FadeIn(markov_chain_g))
        self.wait()

        base_ranking_values = [0.95, 0.75, 0.5, 0.25]
        original_width = markov_chain_g.vertices[0].width
        final_ranking = self.show_randomized_ranking(
            markov_chain_g, base_ranking_values
        )

        how_to_measure_importance = Text(
            "How to Measure Relative Importance?", font="CMU Serif", weight=BOLD
        ).scale(0.8)
        how_to_measure_importance.move_to(title.get_center())
        self.play(
            *[
                markov_chain_g.vertices[v].animate.scale_to_fit_width(original_width)
                for v in markov_chain.get_states()
            ],
            FadeOut(final_ranking),
            ReplacementTransform(title, how_to_measure_importance),
        )
        self.wait()

        markov_chain_sim = MarkovChainSimulator(
            markov_chain, markov_chain_g, num_users=100
        )
        users = markov_chain_sim.get_users()

        self.play(*[FadeIn(user) for user in users])
        self.wait()

        num_steps = 5
        for _ in range(num_steps):
            transition_map = markov_chain_sim.get_lagged_smooth_transition_animations()
            self.play(
                *[LaggedStart(*transition_map[i]) for i in markov_chain.get_states()]
            )
            self.wait()
        self.wait()

    def show_randomized_ranking(self, markov_chain_g, base_ranking_values):
        original_markov_chain_nodes = [
            markov_chain_g.vertices[i].copy() for i in range(len(base_ranking_values))
        ]
        positions = [LEFT * 2.4, LEFT * 0.8, RIGHT * 0.8, RIGHT * 2.4]
        gt_signs = [MathTex(">"), MathTex(">"), MathTex(">")]
        for i, sign in enumerate(gt_signs):
            gt_signs[i].move_to((positions[i] + positions[i + 1]) / 2)
        num_iterations = 5
        SHIFT_DOWN = DOWN * 3.2
        for step in range(num_iterations):
            print("Iteration", step)
            current_ranking_values = self.generate_new_ranking(base_ranking_values)
            current_ranking_map = self.get_ranking_map(current_ranking_values)
            scaling_animations = []
            for v, scaling in current_ranking_map.items():
                scaling_animations.append(
                    markov_chain_g.vertices[v].animate.scale_to_fit_width(scaling)
                )
            current_ranking = self.get_ranking(current_ranking_map)
            ranking_animations = []
            for i, v in enumerate(current_ranking):
                if step != 0:
                    ranking_animations.append(
                        original_markov_chain_nodes[v].animate.move_to(
                            positions[i] + SHIFT_DOWN
                        )
                    )
                else:
                    ranking_animations.append(
                        FadeIn(
                            original_markov_chain_nodes[v].move_to(
                                positions[i] + SHIFT_DOWN
                            )
                        )
                    )

            if step == 0:
                ranking_animations.extend(
                    [FadeIn(sign.shift(SHIFT_DOWN)) for sign in gt_signs]
                )

            self.play(*scaling_animations + ranking_animations)
            self.wait()

        return VGroup(*original_markov_chain_nodes + gt_signs)

    def get_ranking(self, ranking_map):
        sorted_map = {
            k: v for k, v in sorted(ranking_map.items(), key=lambda item: item[1])
        }
        return [key for key in sorted_map][::-1]

    def generate_new_ranking(self, ranking_values):
        np.random.shuffle(ranking_values)
        new_ranking = []
        for elem in ranking_values:
            new_ranking.append(elem + np.random.uniform(-0.08, 0.08))
        return new_ranking

    def get_ranking_map(self, ranking_values):
        return {i: ranking_values[i] for i in range(len(ranking_values))}


class IntroStationaryDistribution(Scene):
    def construct(self):
        self.show_counts()

    def show_counts(self):
        markov_chain = MarkovChain(
            5,
            [
                (0, 1),
                (1, 0),
                (0, 2),
                (1, 2),
                (1, 3),
                (2, 0),
                (2, 3),
                (3, 1),
                (2, 4),
                (1, 4),
                (4, 2),
                (3, 4),
                (4, 0),
            ],
        )
        markov_chain_g = MarkovChainGraph(
            markov_chain, enable_curved_double_arrows=True, layout="circular"
        )
        markov_chain_t_labels = markov_chain_g.get_transition_labels()
        markov_chain_g.scale(1.5)
        self.play(
            FadeIn(markov_chain_g),
        )
        self.wait()

        markov_chain_sim = MarkovChainSimulator(
            markov_chain, markov_chain_g, num_users=1
        )
        users = markov_chain_sim.get_users()
        # scale user a bit here
        users[0].scale(1.5)

        self.play(*[FadeIn(user) for user in users])
        self.wait()

        num_steps = 100
        stabilize_threshold = num_steps - 10
        print("Count", markov_chain_sim.get_state_counts())
        print("Dist", markov_chain_sim.get_user_dist())
        count_labels = self.get_current_count_mobs(markov_chain_g, markov_chain_sim)
        self.play(*[FadeIn(label) for label in count_labels.values()])
        self.wait()
        use_dist = False
        for i in range(num_steps):
            transition_animations = markov_chain_sim.get_instant_transition_animations()
            count_labels, count_transforms = self.update_count_labels(
                count_labels, markov_chain_g, markov_chain_sim, use_dist=use_dist
            )
            if i > stabilize_threshold:
                self.play(*transition_animations)
                continue
            self.play(*transition_animations + count_transforms)
            if i < 5:
                self.wait()
            if i > 20:
                use_dist = True
            print("Iteration", i)
            print("Count", markov_chain_sim.get_state_counts())
            print("Dist", markov_chain_sim.get_user_dist())

        true_stationary_dist = markov_chain.get_true_stationary_dist()
        print("True stationary dist", true_stationary_dist)
        print("Norm:", np.linalg.norm(true_stationary_dist))

    def get_current_count_mobs(self, markov_chain_g, markov_chain_sim, use_dist=False):
        vertex_mobs_map = markov_chain_g.vertices
        count_labels = {}
        for v in vertex_mobs_map:
            if not use_dist:
                state_counts = markov_chain_sim.get_state_counts()
                label = Text(str(state_counts[v]), font="SF Mono").scale(0.6)
            else:
                state_counts = markov_chain_sim.get_user_dist(round_val=True)
                label = Text("{0:.2f}".format(state_counts[v]), font="SF Mono").scale(
                    0.6
                )
            label_direction = normalize(
                vertex_mobs_map[v].get_center() - markov_chain_g.get_center()
            )
            label.next_to(vertex_mobs_map[v], label_direction)
            count_labels[v] = label

        return count_labels

    def update_count_labels(
        self, count_labels, markov_chain_g, markov_chain_sim, use_dist=False
    ):
        if count_labels is None:
            count_labels = self.get_current_count_mobs(
                markov_chain_g, markov_chain_sim, use_dist=use_dist
            )
            transforms = [Write(label) for label in count_labels.values()]

        else:
            new_count_labels = self.get_current_count_mobs(
                markov_chain_g, markov_chain_sim, use_dist=use_dist
            )
            transforms = [
                Transform(count_labels[v], new_count_labels[v]) for v in count_labels
            ]

        return count_labels, transforms


class StationaryDistPreview(Scene):
    def construct(self):
        stationary_dist = Text(
            "Stationary Distribution", font="CMU Serif", weight=BOLD
        ).scale(0.8)
        point_1 = Text(
            "1. How to find stationary distributions?", font="CMU Serif"
        ).scale(0.5)
        point_2 = Text("2. When do they exist?", font="CMU Serif").scale(0.5)
        point_3 = Text("3. How do we efficiently compute them?", font=REDUCIBLE_FONT).scale(0.5)
        points = VGroup(point_1, point_2, point_3).arrange(DOWN, aligned_edge=LEFT)

        text = VGroup(stationary_dist, points).arrange(DOWN)

        text.move_to(LEFT * 3.5)

        self.play(Write(text[0]))
        self.wait()

        self.play(FadeIn(point_1))
        self.wait()

        self.play(FadeIn(point_2))
        self.wait()

        self.play(FadeIn(point_3))
        self.wait()


class ModelingMarkovChainsLastFrame(Scene):
    def construct(self):
        markov_chain = MarkovChain(
            5,
            edges=[
                (2, 0),
                (3, 0),
                (4, 0),
                (2, 3),
                (0, 3),
                (3, 4),
                (4, 1),
                (2, 1),
                (0, 2),
                (1, 2),
            ],
        )

        markov_chain_g = MarkovChainGraph(
            markov_chain,
            curved_edge_config={"radius": 2},
            layout_scale=2.6,
        ).scale(1)

        self.play(
            Write(markov_chain_g),
            run_time=2,
        )
        self.wait()

        markov_chain_sim = MarkovChainSimulator(
            markov_chain, markov_chain_g, num_users=1
        )
        users = markov_chain_sim.get_users()
        # scale user a bit here
        users[0].scale(1.8)

        step_label, users = self.simulate_steps(
            markov_chain, markov_chain_g, markov_chain_sim, users
        )

        definitions, prob_dist_labels = self.explain_prob_dist(markov_chain_g)

        self.play(
            FadeOut(definitions),
            FadeOut(step_label),
            markov_chain_g.animate.shift(LEFT * 3.5),
            prob_dist_labels.animate.shift(LEFT * 3.5),
            users[0].animate.shift(LEFT * 3.5),
        )

        new_prob_dist_labels = self.get_prob_dist_labels(markov_chain_g, 0)
        self.play(Transform(prob_dist_labels, new_prob_dist_labels))
        self.wait()

        for step in range(5):
            transition_animations = markov_chain_sim.get_instant_transition_animations()
            new_prob_dist_labels = self.get_prob_dist_labels(markov_chain_g, step + 1)
            self.play(
                *transition_animations,
                Transform(prob_dist_labels, new_prob_dist_labels),
            )
            self.wait()

        self.play(
            FadeOut(prob_dist_labels),
            FadeOut(users[0])
        )
        self.wait()

    def simulate_steps(self, markov_chain, markov_chain_g, markov_chain_sim, users):
        num_steps = 10
        step_annotation = Tex("Step:")
        step_num = Integer(0)
        step_label = VGroup(step_annotation, step_num).arrange(RIGHT)
        step_num.shift(UP * SMALL_BUFF * 0.2)
        step_label.move_to(UP * 3.3)

        self.play(*[FadeIn(user) for user in users], Write(step_label))
        self.wait()

        for _ in range(num_steps):
            transition_animations = markov_chain_sim.get_instant_transition_animations()
            self.play(*transition_animations, step_label[1].animate.increment_value())
            step_label[1].increment_value()

        self.wait()

        step_n = (
            MathTex("n")
            .move_to(step_num.get_center())
            .shift(LEFT * SMALL_BUFF * 0.5 + DOWN * SMALL_BUFF * 0.2)
        )
        transition_animations = markov_chain_sim.get_instant_transition_animations()
        self.play(*transition_animations, Transform(step_label[1], step_n))
        self.wait()

        markov_chain_g.clear_updaters()

        RIGHT_SHIFT = RIGHT * 3.5
        self.play(
            markov_chain_g.animate.shift(RIGHT_SHIFT),
            users[0].animate.shift(RIGHT_SHIFT),
            step_label.animate.shift(RIGHT_SHIFT + RIGHT * 1.5),
        )
        self.wait()

        return step_label, users

    def explain_prob_dist(self, markov_chain_g):
        prob_dist_labels = self.get_prob_dist_labels(markov_chain_g, "n")

        self.play(*[Write(label) for label in prob_dist_labels])
        self.wait()

        definition = (
            Tex(r"$\pi_n(v)$: ", "probability of ", "being in state $v$ at step $n$")
            .scale(0.7)
            .to_edge(LEFT * 2)
            .shift(UP * 3)
        )

        self.play(FadeIn(definition))
        self.wait()

        pi_vector = MathTex(r"\vec{\pi}_n = ").scale(0.7)
        pi_row_vector = Matrix(
            [[r"\pi_n(0)", r"\pi_n(1)", r"\pi_n(2)", r"\pi_n(3)", r"\pi_n(4)"]],
            h_buff=1.7,
        ).scale(0.7)

        pi_vector_definition = VGroup(pi_vector, pi_row_vector).arrange(RIGHT)
        pi_vector_definition.next_to(definition, DOWN, aligned_edge=LEFT)

        self.play(Write(pi_vector))
        self.wait()
        self.play(FadeIn(pi_row_vector))
        self.wait()

        initial_def = MathTex(r"\vec{\pi}_0 \sim \text{Uniform}").scale(0.7)
        initial_def.next_to(pi_vector_definition, DOWN, aligned_edge=LEFT)
        self.play(Write(initial_def))
        self.wait()

        precise_initial = MathTex(r"\vec{\pi}_0 = ").scale(0.7)
        precise_initial_vector = Matrix([[0.2] * 5]).scale(0.7)
        precise_initial_def = (
            VGroup(precise_initial, precise_initial_vector)
            .arrange(RIGHT)
            .next_to(pi_vector_definition, DOWN, aligned_edge=LEFT)
        )

        self.play(ReplacementTransform(initial_def, precise_initial_def))
        self.wait()

        return (
            VGroup(definition, pi_vector_definition, precise_initial_def),
            prob_dist_labels,
        )

    def get_prob_dist_labels(self, markov_chain_g, step):
        prob_dist_labels = [
            MathTex(r"\pi_{0}({1})".format(step, v)).scale(0.7)
            for v in markov_chain_g.markov_chain.get_states()
        ]
        prob_dist_labels[0].next_to(markov_chain_g.vertices[0], UP)
        prob_dist_labels[1].next_to(markov_chain_g.vertices[1], DOWN)
        prob_dist_labels[2].next_to(markov_chain_g.vertices[2], LEFT)
        prob_dist_labels[3].next_to(markov_chain_g.vertices[3], LEFT)
        prob_dist_labels[4].next_to(markov_chain_g.vertices[4], RIGHT)

        return VGroup(*prob_dist_labels)


class Uniqueness(Scene):
    def construct(self):
        challenge = Tex(
            "Can you define a Markov chain with \\\\ multiple stationary distributions?"
        )
        challenge.scale(1).move_to(UP * 3)
        self.play(Write(challenge))
        self.wait()
        dist_between_nodes = 3
        markov_chain = MarkovChain(2, [])
        markov_chain_g = MarkovChainGraph(
            markov_chain,
            layout={
                0: LEFT * dist_between_nodes / 2,
                1: RIGHT * dist_between_nodes / 2,
            },
        )
        markov_chain_g.scale(1.5).shift(UP * 0.5)
        self.play(FadeIn(markov_chain_g))
        self.wait()

        edges = self.get_edges(markov_chain_g)
        labels = [self.get_label(edge, 1) for edge in edges.values()]
        self.play(*[FadeIn(obj) for obj in list(edges.values()) + labels])
        self.wait()

        markov_chain_sim = MarkovChainSimulator(
            markov_chain, markov_chain_g, num_users=60
        )
        users = markov_chain_sim.get_users()
        for u in users:
            u.scale(1.3)
        state_to_users = markov_chain_sim.get_state_to_user()
        for user_id in state_to_users[1]:
            users[user_id].set_stroke(color=REDUCIBLE_GREEN_LIGHTER).set_fill(
                color=REDUCIBLE_GREEN_LIGHTER, opacity=0.8
            )
        self.play(*[FadeIn(u) for u in users])
        self.wait()
        num_steps = 5
        for _ in range(num_steps):
            transition_map = markov_chain_sim.get_lagged_smooth_transition_animations()
            self.play(
                *[LaggedStart(*transition_map[i]) for i in markov_chain.get_states()]
            )
            self.wait()

        punchline = Tex(
            r"Any probability distribution $[\,\pi(0) \quad \pi(1)\,]$ is a stationary distribution."
        )
        punchline.scale(0.8).move_to(DOWN * 1.5)
        self.play(FadeIn(punchline))
        self.wait()

        transition_matrix = MathTex("P = ")
        identity_matrix = Matrix([[1, 0], [0, 1]]).scale(0.8)
        transition_matrix_def = VGroup(transition_matrix, identity_matrix).arrange(
            RIGHT
        )

        transition_matrix_def.next_to(punchline, DOWN)

        self.play(FadeIn(transition_matrix_def))
        self.wait()

        self.play(
            FadeOut(transition_matrix_def), FadeOut(punchline), FadeOut(challenge)
        )
        self.wait()

        user_transition_group, reducible_markov_chain = self.explain_reducibility(
            users,
            state_to_users,
            markov_chain_g,
            VGroup(*list(edges.values()) + labels),
        )
        self.show_irreducible_markov_chain(user_transition_group)

    def get_edges(self, markov_chain_g):
        edge_map = {}
        edge_map[(0, 0)] = self.get_self_edge(markov_chain_g, 0)
        edge_map[(1, 1)] = self.get_self_edge(markov_chain_g, 1)
        return edge_map

    def get_self_edge(self, markov_chain_g, state):
        vertices = markov_chain_g.vertices
        if state == 1:
            angle = -1.6 * PI
        else:
            angle = 1.6 * PI
        edge = CustomCurvedArrow(
            vertices[state].get_top(), vertices[state].get_bottom(), angle=angle
        ).set_color(REDUCIBLE_VIOLET)
        return edge

    def get_label(self, edge, prob):
        return (
            Text(str(prob), font=REDUCIBLE_MONO)
            .set_stroke(BLACK, width=8, background=True, opacity=0.8)
            .scale(0.5)
            .move_to(edge.point_from_proportion(0.15))
        )

    def explain_reducibility(
        self,
        users,
        state_to_users,
        markov_chain_g,
        edges_and_labels,
    ):
        state_0_user = users[state_to_users[0][0]].copy().scale(1.5)
        state_1_user = users[state_to_users[1][0]].copy().scale(1.5)

        right_arrow = Arrow(
            LEFT * 1.5, RIGHT * 1.5, max_tip_length_to_length_ratio=0.1
        ).set_color(GRAY)
        cross = Cross(Dot().scale(2))
        right_arrow_with_cross = VGroup(cross, right_arrow)
        state_0 = markov_chain_g.vertices[0].copy().scale(1 / 1.5)
        state_1 = markov_chain_g.vertices[1].copy().scale(1 / 1.5)

        user_state_0_trans = VGroup(
            state_0_user, right_arrow_with_cross, state_1
        ).arrange(RIGHT)
        user_state_1_trans = VGroup(
            state_1_user, right_arrow_with_cross.copy(), state_0
        ).arrange(RIGHT)

        user_transition_group = VGroup(user_state_0_trans, user_state_1_trans).arrange(
            DOWN
        )

        user_transition_group.next_to(markov_chain_g, DOWN).shift(DOWN * 0.2)

        self.play(Write(user_transition_group[0]), Write(user_transition_group[1]))
        self.wait()

        reducible_markov_chain = Text(
            "Reducible Markov Chain", font="CMU Serif", weight=BOLD
        ).scale(0.8)
        reducible_markov_chain.next_to(markov_chain_g, UP).shift(UP * 0.5)

        self.play(FadeIn(reducible_markov_chain))
        self.wait()
        user_group = VGroup(*users)
        shift_up = UP * 1.5
        self.play(
            reducible_markov_chain.animate.shift(shift_up),
            markov_chain_g.animate.shift(shift_up),
            user_transition_group.animate.shift(shift_up),
            user_group.animate.shift(shift_up),
            edges_and_labels.animate.shift(shift_up),
        )
        self.wait()

        return user_transition_group, reducible_markov_chain

    def show_irreducible_markov_chain(self, user_transition_group):
        irreducible_markov_chain = Text(
            "Irreducible Markov Chain", font="CMU Serif", weight=BOLD
        ).scale(0.8)
        irreducible_markov_chain.next_to(user_transition_group, DOWN).shift(DOWN * 0.2)

        dist_between_nodes = 3
        markov_chain = MarkovChain(2, [(0, 1), (1, 0)])
        markov_chain_g = MarkovChainGraph(
            markov_chain,
            layout={
                0: LEFT * dist_between_nodes / 2,
                1: RIGHT * dist_between_nodes / 2,
            },
        )
        markov_chain_g.scale(1.5).next_to(irreducible_markov_chain, DOWN).shift(
            DOWN * SMALL_BUFF
        )
        self.play(FadeIn(irreducible_markov_chain), FadeIn(markov_chain_g))
        self.wait()

        conclusion = Tex(
            r"All states reachable ",
            r"$\rightarrow$",
            " unique stationary distribution exists",
        ).scale(0.8)

        conclusion.move_to(DOWN * 3)

        self.play(Write(conclusion[0]))
        self.wait()

        self.play(Write(conclusion[1]))
        self.wait()

        self.play(Write(conclusion[2]))
        self.wait()

class ChallengeSeparated(Scene):
    def construct(self):
        challenge = Tex(
            "Challenge: find an irreducible Markov chain \\\\",
            "where stationary distribution does not converge."
        ).scale(0.8)
        challenge[1][-16:-1].set_color(REDUCIBLE_YELLOW)
        challenge.move_to(DOWN * 3.3)
        self.play(
            FadeIn(challenge)
        )
        self.wait()

        self.play(
            FadeOut(challenge)
        )
        self.wait()

class Periodicity(Scene):
    def construct(self):
        self.state_color_map = {
        0: REDUCIBLE_PURPLE,
        1: REDUCIBLE_GREEN,
        2: REDUCIBLE_ORANGE,
        3: REDUCIBLE_CHARM,
        }

        markov_chain = MarkovChain(
            4,
            [
                (0, 1),
                (1, 0),
                (0, 2),
                (1, 2),
                (1, 3),
                (2, 0),
                (2, 3),
                (3, 1),
            ],
            transition_matrix=np.array(
                [
                [0, 0.7, 0.3, 0],
                [0.2, 0, 0.6, 0.2],
                [0.6, 0, 0, 0.4],
                [0, 1, 0, 0],
                ]
            )
        )

        to_fade, markov_chain_g = self.show_convergence(markov_chain, 20, 5/15)
        self.play(
            FadeOut(to_fade)
        )
        self.wait()

        self.clear()

        new_markov_chain = MarkovChain(
            4,
            [
                (0, 1),
                (1, 0),
                (0, 2),
                (1, 2),
                (1, 3),
                (2, 0),
                (2, 3),
                (3, 1),
            ],
            transition_matrix=np.array(
                [
                [0, 0.7, 0.3, 0],
                [0.2, 0, 0.6, 0.2],
                [0.6, 0, 0, 0.4],
                [0, 1, 0, 0],
                ]
            ),
            dist=np.array([1, 0, 0, 0])
        )

        to_fade, markov_chain_g = self.show_convergence(new_markov_chain, 25, 1/15)

        self.play(
            FadeOut(to_fade),
            FadeOut(markov_chain_g)
        )
        self.wait()

        self.state_color_map = {
        0: REDUCIBLE_PURPLE,
        1: REDUCIBLE_GREEN,
        }

        periodic_markov_chain = MarkovChain(
            2,
            [
                (0, 1),
                (1, 0),
            ],
        )

        markov_chain_g = MarkovChainGraph(periodic_markov_chain, enable_curved_double_arrows=True, layout="circular")

        markov_chain_g.scale(1.5)
        markov_chain_t_labels = markov_chain_g.get_transition_labels()
        markov_chain_g.clear_updaters()

        markov_chain_group = VGroup(markov_chain_g, markov_chain_t_labels)

        self.play(
            *[Write(markov_chain_g.vertices[state]) for state in periodic_markov_chain.get_states()]
        )

        self.wait()

        self.play(
            *[Write(markov_chain_g.edges[e]) for e in periodic_markov_chain.get_edges()],
            FadeIn(markov_chain_t_labels)
        )
        self.wait()

        self.play(
            markov_chain_group.animate.shift(UP * 2.5),
        )
        self.wait()

        markov_chain_sim, left_axes_group, left_state_to_line_segments = self.show_periodicity(periodic_markov_chain, markov_chain_group, DOWN * 1.5)
        self.play(
            *[FadeOut(u) for u in markov_chain_sim.get_users()],
            left_axes_group.animate.shift(LEFT * 3.1),
            *[line_segs.animate.shift(LEFT * 3.1) for line_segs in left_state_to_line_segments.values()]
        )
        self.wait()

        periodic_markov_chain.set_starting_dist(np.array([0.8, 0.2]))

        markov_chain_sim, right_axes_group, right_state_to_line_segments = self.show_periodicity(periodic_markov_chain, markov_chain_group, RIGHT * 3.5 + DOWN * 1.5)

        periodic_markov_chain_title = Text("Periodic Markov Chains", font="CMU Serif", weight=BOLD).scale(0.7)
        periodic_markov_chain_title.to_edge(LEFT * 1.5).shift(UP * 1)
        self.play(
            *[FadeOut(u) for u in markov_chain_sim.get_users()],
            FadeOut(left_axes_group),
            *[FadeOut(seg) for seg in list(left_state_to_line_segments.values())],
            Write(periodic_markov_chain_title)
        )
        self.wait()

        intuitive_def = BulletedList(
            "Must be an irreducible Markov chain",
            r"User visits states in regular interval (period) $ > 1 $ (*)",
            r"No guarantee of convergence to stationary distribution",
            r"No such period $> 1$ exists $\rightarrow$ aperiodic Markov chain",
            buff=0.4,
        ).scale(0.55)

        intuitive_def.next_to(periodic_markov_chain_title, DOWN, buff=0.5, aligned_edge=LEFT)
        note = Tex("(*) more rigorous and precise definitions exist").scale(0.5).next_to(intuitive_def, DOWN, aligned_edge=LEFT, buff=0.5)

        for i, point in enumerate(intuitive_def):
            self.play(
                FadeIn(point)
            )
            if i == 1:
                self.add(note)
            self.wait()

    def show_periodicity(self, markov_chain, markov_chain_group, axes_position):
        markov_chain_g, markov_chain_t_labels = markov_chain_group

        markov_chain_sim = MarkovChainSimulator(
            markov_chain, markov_chain_g, num_users=80,
        )

        users = markov_chain_sim.get_users()
        self.play(
            *[FadeIn(u) for u in users]
        )
        self.wait()

        num_steps = 10
        axes, state_to_line_segments = self.get_distribution_plot(
            markov_chain, num_steps,
            axes_position=axes_position,
            ax_width=5, ax_height=3.2,
            dist=markov_chain.get_starting_dist()
        )
        legend = self.get_legend().move_to(axes.get_center() + RIGHT * (axes.width / 2 - MED_SMALL_BUFF) + UP * (axes.height / 2) - MED_SMALL_BUFF)
        starting_dist = markov_chain.get_starting_dist()

        initial_dist = MathTex(
            r"\pi_0(0) = {0} \quad \pi_0(1) = {1}".format(starting_dist[0], starting_dist[1])
        ).scale(0.8).next_to(axes, UP).shift(RIGHT * SMALL_BUFF + UP * SMALL_BUFF * 2)

        self.play(
            Write(axes),
            Write(legend),
            Write(initial_dist)
        )
        self.wait()
        for state in state_to_line_segments:
            if state == 1 or starting_dist[0] != starting_dist[1]:
                continue
            for seg in state_to_line_segments[state]:
                seg.set_stroke(width=10)

        self.demo_convergence_graph(state_to_line_segments, markov_chain_sim, num_steps, step_threshold=3)

        return markov_chain_sim, VGroup(axes, legend, initial_dist), state_to_line_segments

    def demo_convergence_graph(self, state_to_line_segments, markov_chain_sim, num_steps, short_wait_time=1/15, step_threshold=5):
        for step in range(num_steps):
            if step < step_threshold:
                rate_func = smooth
            else:
                rate_func = linear
            transition_animations = markov_chain_sim.get_instant_transition_animations()
            dist_graph_aniamtions = self.get_dist_graph_step_animations(state_to_line_segments, step)
            self.play(
                *transition_animations + dist_graph_aniamtions, rate_func=rate_func
            )
            if step < step_threshold:
                self.wait()
            else:
                self.wait(short_wait_time)

        self.wait()

    def show_convergence(self, markov_chain, num_steps, short_wait_time):
        markov_chain_g = MarkovChainGraph(
            markov_chain,
            enable_curved_double_arrows=True,
            layout="circular",
            state_color_map=self.state_color_map
        )

        markov_chain_g.scale(1.5)
        markov_chain_t_labels = markov_chain_g.get_transition_labels()
        self.play(
            FadeIn(markov_chain_g),
            FadeIn(markov_chain_t_labels)
        )
        self.wait()

        markov_chain_g.clear_updaters()
        markov_chain_group = VGroup(markov_chain_g, markov_chain_t_labels)

        self.play(
            markov_chain_group.animate.scale(1.1 / 1.5).shift(LEFT * 3.5)
        )
        self.wait()

        markov_chain_sim = MarkovChainSimulator(
            markov_chain, markov_chain_g, num_users=100,
        )

        users = markov_chain_sim.get_users()
        self.play(
            *[FadeIn(u) for u in users]
        )
        self.wait()

        stationary_dist = markov_chain.get_true_stationary_dist()

        axes, state_to_line_segments = self.get_distribution_plot(markov_chain, num_steps, dist=markov_chain.get_starting_dist())
        legend = self.get_legend().to_edge(RIGHT * 3).shift(UP * 2.5)
        self.play(
            Write(axes),
            Write(legend),
        )
        self.wait()

        for step in range(num_steps):
            if step < 5:
                rate_func = smooth
            else:
                rate_func = linear
            transition_animations = markov_chain_sim.get_instant_transition_animations()
            dist_graph_aniamtions = self.get_dist_graph_step_animations(state_to_line_segments, step)
            self.play(
                *transition_animations + dist_graph_aniamtions, rate_func=rate_func
            )
            if step < 5:
                self.wait()
            else:
                self.wait(short_wait_time)

        self.wait()
        return VGroup(axes, legend, VGroup(*list(state_to_line_segments.values())), VGroup(*users)), markov_chain_group

    def get_dist_graph_step_animations(self, state_to_line_segments, step):
        animations = []
        for state, line_segments in state_to_line_segments.items():
            animations.append(
                Create(line_segments[step])
            )
        return animations

    def get_distribution_plot(self, markov_chain, num_steps, axes_position=RIGHT*3, ax_width=5, ax_height=5, dist=None):
        markov_chain_copy = MarkovChain(
            len(markov_chain.get_states()),
            markov_chain.get_edges(),
            transition_matrix=markov_chain.get_transition_matrix(),
            dist=dist,
        )
        distribution_sequence = [markov_chain_copy.get_current_dist()]
        for _ in range(num_steps):
            markov_chain_copy.update_dist()
            distribution_sequence.append(markov_chain_copy.get_current_dist())

        distribution_sequence = np.array(distribution_sequence)
        print(distribution_sequence)

        axes = Axes(
            x_range=(0, num_steps, 1),
            y_range=(0, 1, 0.1),
            x_length=ax_width,
            y_length=ax_height,
            tips=False,
            axis_config={"include_numbers": True, "include_ticks": False},
            x_axis_config={"numbers_to_exclude": range(num_steps + 1)},
            y_axis_config={"numbers_to_exclude": np.arange(0.1, 1.05, 0.1)}
        ).move_to(axes_position)

        custom_y_label = Text("1.0", font=REDUCIBLE_MONO).scale(0.4)
        custom_y_label_pos = axes.y_axis.n2p(1) + LEFT * 0.4
        custom_y_label.move_to(custom_y_label_pos)

        y_axis_label = MathTex(r"\pi_n( \cdot )").scale(0.7).next_to(axes.y_axis, LEFT)


        custom_x_label = Text(f"{num_steps}", font=REDUCIBLE_MONO).scale(0.4)
        custom_x_label_pos = axes.x_axis.n2p(num_steps) + DOWN * MED_SMALL_BUFF
        custom_x_label.move_to(custom_x_label_pos)

        x_axis_label = Text("Step/Iteration", font=REDUCIBLE_MONO).scale(0.5).next_to(axes.x_axis, DOWN)

        axes.add(custom_x_label, custom_y_label, x_axis_label, y_axis_label)

        state_to_line_segments = {}
        for state in markov_chain.get_states():
            x_values=list(range(num_steps+1))
            y_values=distribution_sequence[:, state]
            line_color=self.state_color_map[state]
            line_segments = self.get_line_segments(axes, x_values, y_values, line_color)
            state_to_line_segments[state] = line_segments

        return axes, state_to_line_segments

    def get_line_segments(self, axes, x_values, y_values, line_color):
        line_segments = []
        for i in range(len(x_values) - 1):
            start = axes.coords_to_point(x_values[i], y_values[i])
            end = axes.coords_to_point(x_values[i + 1], y_values[i + 1])
            line_seg = Line(start, end).set_stroke(color=line_color)
            line_segments.append(line_seg)

        return VGroup(*line_segments)

    def get_legend(self):
        legend = VGroup()
        for state, color in self.state_color_map.items():
            label = Text(str(state), font="SF Mono").scale(0.4)
            line = Line(LEFT*SMALL_BUFF, RIGHT*SMALL_BUFF).set_stroke(color)
            legend_item = VGroup(line, label).arrange(RIGHT)
            legend.add(legend_item)
        return legend.arrange_in_grid(rows=2)

class PeriodicityUsersMovement(Scene):
    def construct(self):
        periodic_markov_chain = MarkovChain(
            2,
            [
                (0, 1),
                (1, 0),
            ],
        )

        markov_chain_g = MarkovChainGraph(periodic_markov_chain, enable_curved_double_arrows=True, layout="circular")

        markov_chain_g.scale(1.5)
        markov_chain_t_labels = markov_chain_g.get_transition_labels()

        markov_chain_group = VGroup(markov_chain_g, markov_chain_t_labels).shift(UP * 2.5)
        markov_chain_sim = MarkovChainSimulator(
            periodic_markov_chain, markov_chain_g, num_users=1,
        )
        users = markov_chain_sim.get_users()
        users[0].scale(1.2)
        self.play(
            FadeIn(users[0])
        )
        self.wait()

        for step in range(10):
            transition_animations = markov_chain_sim.get_instant_transition_animations()
            self.play(
                *transition_animations
            )


class IntroduceBigTheorem1(Periodicity):
    def construct(self):
        self.state_color_map = {
        0: REDUCIBLE_PURPLE,
        1: REDUCIBLE_GREEN,
        2: REDUCIBLE_ORANGE,
        3: REDUCIBLE_CHARM,
        }

        markov_chain_1 = MarkovChain(
            4,
            [
                (0, 1),
                (1, 0),
                (0, 2),
                (1, 2),
                (1, 3),
                (2, 0),
                (2, 3),
                (3, 1),
            ],
            transition_matrix=np.array(
                [
                [0, 0.7, 0.3, 0],
                [0.2, 0, 0.6, 0.2],
                [0.6, 0, 0, 0.4],
                [0, 1, 0, 0],
                ]
            ),
        )

        self.make_convergence_scene(markov_chain_1, 40)

    def make_convergence_scene(self, markov_chain, num_steps, short_wait_time=1/15):
        markov_chain_g = MarkovChainGraph(
            markov_chain,
            enable_curved_double_arrows=True,
            layout="circular",
            state_color_map=self.state_color_map
        )

        markov_chain_t_labels = markov_chain_g.get_transition_labels()

        markov_chain_g.clear_updaters()
        markov_chain_group = VGroup(markov_chain_g, markov_chain_t_labels)

        markov_chain_group.scale(1.1).shift(LEFT * 3.5 + UP * 0.5)

        markov_chain_sim = MarkovChainSimulator(
            markov_chain, markov_chain_g, num_users=100,
        )

        users = markov_chain_sim.get_users()
        self.play(
            FadeIn(markov_chain_group),
            *[FadeIn(u) for u in users]
        )
        self.wait()

        stationary_dist = markov_chain.get_true_stationary_dist()

        axes, state_to_line_segments = self.get_distribution_plot(markov_chain, num_steps, dist=markov_chain.get_starting_dist(), axes_position=RIGHT * 3 + UP * 0.5)
        legend = self.get_legend().to_edge(RIGHT * 3).shift(UP * 2.5)

        starting_dist = markov_chain.get_starting_dist()
        initial_dist = MathTex(
            r"\pi_0(0) = {0} \quad \pi_0(1) = {1} \quad \pi_0(2) = {2} \quad \pi_0(3) = {3}".format(
                starting_dist[0], starting_dist[1], starting_dist[2], starting_dist[3]
            )
        ).scale(0.8).move_to(DOWN * 3.2)
        self.play(
            Write(axes),
            Write(legend),
            FadeIn(initial_dist)
        )
        self.wait()
        wait_time = 1
        for step in range(num_steps):
            if step < 5:
                rate_func = smooth
            else:
                rate_func = linear

            transition_animations = markov_chain_sim.get_instant_transition_animations()
            dist_graph_aniamtions = self.get_dist_graph_step_animations(state_to_line_segments, step)

            self.play(
                *transition_animations + dist_graph_aniamtions, rate_func=rate_func
            )
            if step < 5:
                self.wait(wait_time)
                wait_time *= 0.8

        self.wait()

        self.play(
            *[mob.animate.fade(0.6) for mob in self.mobjects]
        )
        return VGroup(axes, legend, VGroup(*list(state_to_line_segments.values())), VGroup(*users)), markov_chain_group


class IntroduceBigTheorem2(IntroduceBigTheorem1):
    def construct(self):
        self.state_color_map = {
        0: REDUCIBLE_PURPLE,
        1: REDUCIBLE_GREEN,
        2: REDUCIBLE_ORANGE,
        3: REDUCIBLE_CHARM,
        }

        markov_chain_1 = MarkovChain(
            4,
            [
                (0, 1),
                (1, 0),
                (0, 2),
                (1, 2),
                (1, 3),
                (2, 0),
                (2, 3),
                (3, 1),
            ],
            transition_matrix=np.array(
                [
                [0, 0.7, 0.3, 0],
                [0.2, 0, 0.6, 0.2],
                [0.6, 0, 0, 0.4],
                [0, 1, 0, 0],
                ]
            ),
            dist=[0, 1, 0, 0]
        )

        self.make_convergence_scene(markov_chain_1, 40)

class IntroduceBigTheorem3(IntroduceBigTheorem1):
    def construct(self):
        self.state_color_map = {
        0: REDUCIBLE_PURPLE,
        1: REDUCIBLE_GREEN,
        2: REDUCIBLE_ORANGE,
        3: REDUCIBLE_CHARM,
        }

        markov_chain_1 = MarkovChain(
            4,
            [
                (0, 1),
                (1, 0),
                (0, 2),
                (1, 2),
                (1, 3),
                (2, 0),
                (2, 3),
                (3, 1),
            ],
            transition_matrix=np.array(
                [
                [0, 0.7, 0.3, 0],
                [0.2, 0, 0.6, 0.2],
                [0.6, 0, 0, 0.4],
                [0, 1, 0, 0],
                ]
            ),
            dist=[0.5, 0.1, 0.2, 0.2]
        )

        self.make_convergence_scene(markov_chain_1, 40)

class IntroduceBigTheorem4(IntroduceBigTheorem1):
    def construct(self):
        self.state_color_map = {
        0: REDUCIBLE_PURPLE,
        1: REDUCIBLE_GREEN,
        2: REDUCIBLE_ORANGE,
        3: REDUCIBLE_CHARM,
        }

        markov_chain_1 = MarkovChain(
            4,
            [
                (0, 1),
                (1, 0),
                (0, 2),
                (1, 2),
                (1, 3),
                (2, 0),
                (2, 3),
                (3, 1),
            ],
            transition_matrix=np.array(
                [
                [0, 0.7, 0.3, 0],
                [0.2, 0, 0.6, 0.2],
                [0.6, 0, 0, 0.4],
                [0, 1, 0, 0],
                ]
            ),
            dist=[0.1, 0.1, 0.7, 0.1]
        )

        self.make_convergence_scene(markov_chain_1, 40)

class IntroduceBigTheoremText(Scene):
    def construct(self):
        theorem = Text("Ergodic Theorem", font="CMU Serif", weight=BOLD)
        theory = Tex("For irreducible and aperiodic Markov chains:").scale(1)
        tenet_1 = Tex(r"1. A unique stationary distribution $\pi$ exists").scale(0.9)
        tenet_2 = Tex(r"2. All initial distributions $\pi_0$ converge to $\pi$").scale(0.9)

        left_aligned_text = VGroup(
            tenet_1, tenet_2
        ).arrange(DOWN, aligned_edge=LEFT)

        general_explanation = VGroup(theory, left_aligned_text).arrange(DOWN)

        all_text = VGroup(theorem, general_explanation).arrange(DOWN, buff=0.5)

        self.play(
            Write(theorem)
        )
        self.wait()

        self.play(
            Write(theory)
        )

        self.wait()

        self.play(
            FadeIn(tenet_1)
        )
        self.wait()

        self.play(
            FadeIn(tenet_2)
        )
        self.wait()

        how_to_calculate_it = Text("How to calculate the stationary distribution?", font=REDUCIBLE_FONT, weight=BOLD).scale(0.7).shift(DOWN * 3)

        self.play(
            Write(how_to_calculate_it)
        )
        self.wait()

class PageRankExtraText(Scene):
    def construct(self):
        # problem = Text("How to rank states 0, 1 and 2?", font=REDUCIBLE_FONT).scale(0.8).move_to(DOWN * 3.3)

        # self.play(
        #     FadeIn(problem)
        # )
        # self.wait()

        # self.clear()

        # self.wait()

        problem_2 = Text("How to rank states 2 and 3?", font=REDUCIBLE_FONT).scale(0.8).move_to(DOWN * 3.3)

        self.play(
            FadeIn(problem_2)
        )
        self.wait()

class PageRank(IntroduceBigTheorem1):
    def construct(self):
        self.show_pagerank_cxn()
        self.clear()

    def show_pagerank_cxn(self):
        template = TexTemplate()
        template.add_to_preamble(r"\usepackage{bm}")
        page_rank = Text("PageRank", font=REDUCIBLE_FONT, weight=BOLD)
        arrow = Tex(
            r"$\bm{\Updownarrow}$",
            tex_template=template
        ).scale(1.5)
        stationary_dist_text = Text("Stationary Distributions of Markov Chains", font=REDUCIBLE_FONT, weight=BOLD).scale(0.8)
        group = VGroup(page_rank, arrow, stationary_dist_text).arrange(DOWN)

        self.play(
            Write(page_rank)
        )
        self.wait()

        self.play(
            Write(arrow)
        )
        self.wait()

        self.play(
            FadeIn(stationary_dist_text)
        )
        self.wait()

        some_added_complexity = Text("(*) In practice, it's a bit more complicated", font=REDUCIBLE_FONT).scale(0.7)
        self.add(some_added_complexity.move_to(DOWN * 3))
        self.wait()

        self.clear()

        self.intro_issue()

    def intro_issue(self):
        reducible_markov_chain = MarkovChain(
            4,
            [(0, 1), (1, 2), (2, 3), (1, 3), (0, 2)]
        )

        self.state_color_map = {
        0: REDUCIBLE_PURPLE,
        1: REDUCIBLE_GREEN,
        2: REDUCIBLE_ORANGE,
        3: REDUCIBLE_CHARM,
        }

        red_graph, reducible_markov_chain_g = self.make_convergence_scene(reducible_markov_chain, 25, explain_problem=True)

        problem = Text("How to rank states 0, 1 and 2?", font=REDUCIBLE_FONT).scale(0.8).move_to(DOWN * 3.3)

        self.play(
            FadeIn(problem)
        )
        self.wait()

        self.clear()

        # self.play(
        #     FadeOut(graph),
        #     FadeOut(reducible_markov_chain_g),
        #     FadeOut(problem)
        # )
        # self.wait()

        periodic_markov_chain = MarkovChain(
            4,
            [(0, 1), (1, 0), (2, 1), (3, 0), (2, 3), (3, 2)]
        )

        periodic_graph, periodic_markov_chain_g = self.make_convergence_scene(periodic_markov_chain, 25, add_stroke=True, extra_stroke_states=[0, 2], explain_cycle=True)


        problem = Text("How to rank states 2 and 3?", font=REDUCIBLE_FONT).scale(0.8).move_to(DOWN * 3.3)

        self.play(
            FadeIn(problem)
        )
        self.wait()

        reducible_markov_chain_g.scale(1).move_to(LEFT * 3.5)

        self.play(
            FadeOut(periodic_graph),
            FadeOut(problem),
            periodic_markov_chain_g.animate.scale(1).move_to(RIGHT * 3.5),
            FadeIn(reducible_markov_chain_g)
        )
        self.wait()

        question = Tex("How to deal with Markov chains that \\\\ are not irreducible and aperiodic?").scale(0.8)

        question.move_to(UP * 3.3)

        self.play(
            Write(question)
        )

        self.wait()

        self.clear()

        self.show_larger_reducible_markov_chain()

    def make_convergence_scene(self, markov_chain, num_steps, short_wait_time=1/15, add_stroke=False, extra_stroke_states=None, explain_problem=False, explain_cycle=False):
        markov_chain_g = MarkovChainGraph(
            markov_chain,
            enable_curved_double_arrows=True,
            layout="circular",
            state_color_map=self.state_color_map
        )

        markov_chain_t_labels = markov_chain_g.get_transition_labels()

        markov_chain_g.clear_updaters()
        markov_chain_group = VGroup(markov_chain_g, markov_chain_t_labels)

        markov_chain_group.scale(1.1).shift(LEFT * 3.5 + UP * 0.5)

        markov_chain_sim = MarkovChainSimulator(
            markov_chain, markov_chain_g, num_users=100,
        )

        users = markov_chain_sim.get_users()
        self.play(
            FadeIn(markov_chain_group),
        )
        self.wait()
        to_fade = []
        if explain_problem:
            arrow = Arrow(ORIGIN, UP * 1.5).set_color(GRAY)
            no_out_going_links = Tex("No outgoing links").scale(0.7)
            arrow.next_to(markov_chain_g.vertices[3], DOWN, buff=MED_SMALL_BUFF)
            no_out_going_links.next_to(arrow, DOWN)

            self.play(
                Write(arrow),
                FadeIn(no_out_going_links)
            )
            self.wait()
            to_fade.extend([arrow, no_out_going_links])

        if explain_cycle:
            surround_rect = SurroundingRectangle(VGroup(markov_chain_g.vertices[0], markov_chain_g.vertices[1]), buff=SMALL_BUFF)
            self.play(
                Create(surround_rect)
            )
            cycle = Tex("Cycle").scale(0.7).next_to(surround_rect, UP)

            self.play(
                Write(cycle)
            )
            self.wait()

            self.play(
                FadeOut(cycle),
                FadeOut(surround_rect),
            )

        self.play(
            *[FadeIn(u) for u in users]
        )
        self.wait()

        if explain_problem:
            self.play(
                *[FadeOut(mob) for mob in to_fade]
            )

        stationary_dist = markov_chain.get_true_stationary_dist()

        axes, state_to_line_segments = self.get_distribution_plot(markov_chain, num_steps, dist=markov_chain.get_starting_dist(), axes_position=RIGHT * 3 + UP * 0.5)
        if add_stroke:
            for state in state_to_line_segments:
                if state in extra_stroke_states:
                    line_segments = state_to_line_segments[state]
                    for seg in line_segments:
                        seg.set_stroke(width=7)

        legend = self.get_legend().to_edge(RIGHT * 3).shift(UP * 2.5)

        starting_dist = markov_chain.get_starting_dist()
        self.play(
            Write(axes),
            Write(legend),
        )
        self.wait()
        wait_time = 1
        for step in range(num_steps):
            if step < 5:
                rate_func = smooth
            else:
                rate_func = linear

            transition_animations = markov_chain_sim.get_instant_transition_animations()
            dist_graph_aniamtions = self.get_dist_graph_step_animations(state_to_line_segments, step)

            self.play(
                *transition_animations + dist_graph_aniamtions, rate_func=rate_func
            )
            if step < 5:
                self.wait(wait_time)
                wait_time *= 0.8

        self.wait()
        return VGroup(axes, legend, VGroup(*list(state_to_line_segments.values())), VGroup(*users)), markov_chain_group

    def show_larger_reducible_markov_chain(self):
        markov_chain, markov_chain_g = self.get_web_graph()

        self.play(
            FadeIn(markov_chain_g)
        )
        self.wait()
        markov_chain_sim = MarkovChainSimulator(markov_chain, markov_chain_g, num_users=1)
        user = markov_chain_sim.get_users()[0].scale(1.8)
        sequence_of_states = [39, 52, 63, 74, 61, 60, 49, 62, 74, 73, 72] + [72] * 4
        for i, state in enumerate(sequence_of_states):
            location = markov_chain_sim.poisson_distribution(markov_chain_g.vertices[state].get_center())
            user_location = np.array([location[0], location[1], 0])
            if i == 0:
                self.play(
                    FadeIn(user)
                )
            else:
                self.play(
                    user.animate.move_to(user_location)
                )

        self.wait()

        location = markov_chain_sim.poisson_distribution(markov_chain_g.vertices[24].get_center())
        user_location = np.array([location[0], location[1], 0])
        self.play(
            user.animate.move_to(user_location)
        )
        self.wait()

        for _ in range(10):
            transition_animations = markov_chain_sim.get_instant_transition_animations()
            self.play(
                *transition_animations
            )


    def get_web_graph(self):
        graph_layout = self.get_web_graph_layout()
        graph_edges = self.get_web_graph_edges(graph_layout)
        graph_edges.remove((72, 73))
        graph_edges.append((73, 72))
        print(len(graph_layout))
        initial_dist = [0] * len(graph_layout)
        initial_dist[24] = 1
        markov_chain = MarkovChain(len(graph_layout), graph_edges, dist=np.array(initial_dist))
        markov_chain_g = MarkovChainGraph(
            markov_chain,
            labels=False,
            enable_curved_double_arrows=False,
            straight_edge_config={"max_tip_length_to_length_ratio": 0.1},
            layout=graph_layout,
        )

        return markov_chain, markov_chain_g.shift(UP)

    def get_web_graph_layout(self):
        grid_height = 7
        grid_width = 12

        layout = {}
        node_id = 0
        STEP = 1
        for i in np.arange(-grid_height // 2, grid_height // 2, STEP):
            for j in np.arange(-grid_width // 2, grid_width // 2, STEP):
                noise = RIGHT * np.random.uniform(-1, 1) + UP * np.random.uniform(-1, 1)
                layout[node_id] = UP * i + RIGHT * j + noise * 0.5 / 3.1
                node_id += 1

        return layout

    def get_web_graph_edges(self, graph_layout):
        edges = []
        for u in graph_layout:
            for v in graph_layout:
                if u != v and np.linalg.norm(graph_layout[v] - graph_layout[u]) < 1.6:
                    if np.random.uniform() < 0.6:
                        edges.append((u, v))
        return edges

class PageRankSolution(Scene):
    def construct(self):
        text = Text("Idea: randomly select a new state", font=REDUCIBLE_FONT).scale(0.8).move_to(UP * 3.5)
        self.play(
            Write(text),
            run_time=2
        )
        self.wait()

class PageRankAlphaIntro(Scene):
    def construct(self):
        reducible_markov_chain = MarkovChain(
            4,
            [(0, 1), (1, 2), (2, 3), (1, 3), (0, 2)]
        )

        self.state_color_map = {
        0: REDUCIBLE_PURPLE,
        1: REDUCIBLE_GREEN,
        2: REDUCIBLE_ORANGE,
        3: REDUCIBLE_CHARM,
        }

        markov_chain_g = MarkovChainGraph(
            reducible_markov_chain,
            enable_curved_double_arrows=True,
            layout="circular",
            state_color_map=self.state_color_map
        )

        markov_chain_t_labels = markov_chain_g.get_transition_labels()

        markov_chain_g.clear_updaters()
        markov_chain_group = VGroup(markov_chain_g, markov_chain_t_labels)

        idea = Tex(r"With probability $\frac{\alpha}{N}$, transition to a random state").scale(0.8).move_to(UP * 3.5)
        note_1 = Tex(r"$N = 4$ (number of states), we pick $\alpha$ (e.g $\alpha = 0.4$)").scale(0.7).next_to(idea, DOWN)
        markov_chain_group.scale(1)
        self.wait()
        self.play(
            Write(markov_chain_group),
            run_time=2
        )
        self.wait()

        self.play(
            Write(idea)
        )

        self.wait()

        self.play(
            FadeIn(note_1)
        )
        self.wait()

        self.play(
            markov_chain_group.animate.shift(LEFT * 3.5 + DOWN * 0.2)
        )
        self.wait()

        new_markov_chain_incorrect = MarkovChain(
            4,
            [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 0),
            (1, 2),
            (1, 3),
            (2, 0),
            (2, 1),
            (2, 3),
            (3, 0),
            (3, 1),
            (3, 2),
            ]
        )

        self.alpha = 0.4

        original_transition_matrix = new_markov_chain_incorrect.get_transition_matrix()
        reducible_markov_chain_matrix = reducible_markov_chain.get_transition_matrix()
        for i in range(original_transition_matrix.shape[0]):
            for j in range(original_transition_matrix.shape[1]):
                if reducible_markov_chain_matrix[i][j] != 0:
                    original_transition_matrix[i][j] = reducible_markov_chain_matrix[i][j] + self.alpha / 4
                else:
                    original_transition_matrix[i][j] = self.alpha / 4

        new_markov_chain_incorrect.set_transition_matrix(original_transition_matrix)

        markov_chain_incorrect_g = MarkovChainGraph(
            new_markov_chain_incorrect,
            enable_curved_double_arrows=True,
            layout="circular",
            state_color_map=self.state_color_map
        )

        markov_chain_incorrect_labels = markov_chain_incorrect_g.get_transition_labels(scale=0.25, round_val=False)

        markov_chain_incorrect_g.clear_updaters()
        markov_chain_incorrect_group = VGroup(markov_chain_incorrect_g, markov_chain_incorrect_labels)

        markov_chain_incorrect_group.shift(RIGHT * 3.5 + DOWN * 0.2)

        self_edges = VGroup(*[self.get_self_edge(markov_chain_incorrect_g, state) for state in new_markov_chain_incorrect.get_states()])

        self.play(
            FadeIn(markov_chain_incorrect_group),
            FadeIn(self_edges)
        )
        self.wait()

        to_reduce_opacity = []
        for state in new_markov_chain_incorrect.get_states():
            if state != 0:
                to_reduce_opacity.append(markov_chain_incorrect_g.vertices[state])
                # to_reduce_opacity.append(self_edges[state])

        for edge in new_markov_chain_incorrect.get_edges():
            if edge[0] != 0:
                to_reduce_opacity.append(markov_chain_incorrect_g.edges[edge])
                to_reduce_opacity.append(markov_chain_incorrect_g.labels[edge])

        self.play(
            *[mob.animate.set_opacity(0.3) for mob in to_reduce_opacity],
            *[arrow[0].animate.set_stroke(opacity=0.3) for arrow in self_edges[1:]],
            *[arrow[1].animate.set_fill(opacity=0.3) for arrow in self_edges[1:]],
            *[arrow[0].tip.animate.set_fill(opacity=0.3).set_stroke(opacity=0.3) for arrow in self_edges[1:]]
        )
        self.wait()

        second_part = Tex(
            "Reduce original transition probabilities by factor of" + "\\\\",
            r"$(1 - \alpha)$ and then add the random state transition of $\frac{\alpha}{N}$"
        ).scale(0.7)
        second_part.move_to(DOWN * 3.5)

        self.play(
            FadeIn(second_part)
        )

        self.wait()

        self.play(
            *[mob[0].animate.set_fill(opacity=0.5).set_stroke(opacity=1) for mob in to_reduce_opacity[:3]],
            *[mob[1].animate.set_fill(opacity=1) for mob in to_reduce_opacity[:3]],
            *[mob.animate.set_opacity(1) for mob in to_reduce_opacity[3:]],
            *[arrow[0].animate.set_stroke(opacity=1) for arrow in self_edges[1:]],
            *[arrow[1].animate.set_fill(opacity=1) for arrow in self_edges[1:]],
            *[arrow[0].tip.animate.set_fill(opacity=1).set_stroke(opacity=0.3) for arrow in self_edges[1:]]
        )
        self.wait()

        new_intermediate_labels = {}
        for edge in reducible_markov_chain.get_edges():
            u, v = edge
            edge_mob = markov_chain_g.edges[edge]
            text = str(reducible_markov_chain_matrix[u][v]) + " * 0.6 + 0.4 / 4"
            label = self.make_label(text, edge_mob, prop=0.4)
            new_intermediate_labels[edge] = label

        self.play(
            *[Transform(markov_chain_g.labels[edge], new_intermediate_labels[edge]) for edge in reducible_markov_chain.get_edges()]
        )
        self.wait()

        new_labels = {}
        new_labels_corrected = {}
        for edge in reducible_markov_chain.get_edges():
            u, v = edge
            edge_mob = markov_chain_g.edges[edge]
            text = str(reducible_markov_chain_matrix[u][v] * (1 - self.alpha) + self.alpha / 4)
            label = self.make_label(text, edge_mob, scale=0.25)
            new_labels_corrected[edge] = self.make_label(text, markov_chain_incorrect_g.edges[edge], scale=0.25)
            new_labels[edge] = label

        self.play(
            *[Transform(markov_chain_g.labels[edge], new_labels[edge]) for edge in reducible_markov_chain.get_edges()],
            *[Transform(markov_chain_incorrect_g.labels[edge], new_labels_corrected[edge]) for edge in reducible_markov_chain.get_edges()]
        )
        self.wait()


        note_2 = Tex(r"$N = 4$ (number of states), we pick $\alpha$ (typically $\alpha = 0.15$)").scale(0.7).next_to(idea, DOWN)
        self.play(
            ReplacementTransform(note_1, note_2),
            FadeOut(markov_chain_group),
            FadeOut(markov_chain_incorrect_group),
            *[FadeOut(mob) for mob in self_edges]
        )
        self.wait()

        periodic_markov_chain = MarkovChain(
            4,
            [(0, 1), (1, 0), (2, 1), (3, 0), (2, 3), (3, 2)]
        )

        periodic_markov_chain_g = MarkovChainGraph(
            periodic_markov_chain,
            enable_curved_double_arrows=True,
            layout="circular",
            state_color_map=self.state_color_map
        )

        periodic_markov_chain_t_labels = periodic_markov_chain_g.get_transition_labels()

        periodic_markov_chain_g.clear_updaters()
        periodic_markov_chain_group = VGroup(periodic_markov_chain_g, periodic_markov_chain_t_labels)

        periodic_markov_chain_group.move_to(LEFT * 3.5)

        self.play(
            FadeIn(periodic_markov_chain_group)
        )
        self.wait()
        p_equals = Tex(r"$P = $")
        p_matrix = Matrix(periodic_markov_chain.get_transition_matrix()).scale(0.8)

        p_matrix_group = VGroup(p_equals, p_matrix).arrange(RIGHT)
        p_matrix_group.scale(0.8).move_to(RIGHT * 2.1)

        self.play(
            FadeIn(p_matrix_group)
        )
        self.wait()

        self.play(
            p_matrix_group.animate.shift(UP * 1.5)
        )

        p_hat_expr = MathTex(r"\hat{P} = (1 - \alpha) P + \frac{\alpha}{N} (N \cross N \text{ matrix of all 1`s)}").scale(0.65)
        self.alpha = 0.15
        p_hat_matrix_value = periodic_markov_chain.get_transition_matrix() * (1 - self.alpha) + self.alpha * np.ones(4) * 1 / 4
        print(p_hat_matrix_value)
        p_hat_matrix_value = np.around(p_hat_matrix_value, decimals=4)
        p_hat_expr.next_to(p_matrix_group, DOWN, aligned_edge=LEFT)

        self.play(
            FadeIn(p_hat_expr)
        )
        self.wait()

        p_hat_actual = Tex(r"$\hat{P} = $")
        p_hat_matrix = Matrix(p_hat_matrix_value, h_buff=1.7).scale(0.8)

        p_hat_matrix_group = VGroup(p_hat_actual, p_hat_matrix).arrange(RIGHT).scale(0.8)
        p_hat_matrix_group.next_to(p_hat_expr, DOWN, aligned_edge=LEFT)

        self.play(
            FadeIn(p_hat_matrix_group)
        )
        self.wait()
        shift_up = DOWN * p_hat_matrix_group.get_center()[1]
        self.play(
            FadeOut(note_2),
            FadeOut(p_matrix_group),
            FadeOut(second_part),
            FadeOut(idea),
            FadeOut(p_hat_expr),
            p_hat_matrix_group.animate.shift(shift_up + UP * 0.5)
        )
        self.wait()

        stationary_dist_note = Tex(
            r"PageRank calculates $\pi$ from $\hat{P}$",
        ).scale(0.8).next_to(p_hat_matrix_group, DOWN)

        self.play(
            FadeIn(stationary_dist_note)
        )
        self.wait()

    def get_self_edge(self, markov_chain_g, state):
        size_of_angle = 1.5
        vertices = markov_chain_g.vertices
        if state == 0:
            start, end = vertices[state].get_top(), vertices[state].get_bottom()
            angle = -size_of_angle * PI
        elif state == 1:
            start, end = vertices[state].get_right(), vertices[state].get_left()
            angle = size_of_angle * PI
        elif state == 2:
            start, end = vertices[state].get_top(), vertices[state].get_bottom()
            angle = size_of_angle * PI
            pass
        else:
            start, end = vertices[state].get_left(), vertices[state].get_right()
            angle = size_of_angle * PI
        edge = CustomCurvedArrow(
            start, end, angle=angle
        ).set_color(REDUCIBLE_VIOLET)
        label = self.make_label(self.alpha / 4, edge, scale=0.25)
        return VGroup(edge, label)

    def make_label(self, text, edge, scale=0.3, prop=0.2):
        label = (
            Text(str(text), font=REDUCIBLE_MONO)
            .set_stroke(BLACK, width=8, background=True, opacity=0.8)
            .scale(scale)
            .move_to(edge.point_from_proportion(prop))
        )
        return label

class PageRankRecap(Scene):
    def construct(self):
        title = Text("PageRank Algorithm", font=REDUCIBLE_FONT, weight=BOLD)
        title.move_to(UP * 3.5)

        screen_rect = ScreenRectangle(height=4).next_to(title, DOWN)

        self.play(
            Write(title)
        )
        self.wait()
        step_scale = 0.7
        step_one = Tex(r"1. Web Graph $\rightarrow$ Markov chain ($N$, $P$)").scale(step_scale)
        step_two = Tex(r"2. Define $\hat{P} = (1 - \alpha) P + \frac{\alpha}{N} (N \cross N \text{ matrix of all 1`s)}$").scale(step_scale)
        step_three = Tex(r"3. Calculate $\pi$ from Markov chain $(N, \hat{P})$").scale(step_scale)
        step_four = Tex(r"4. Rank web pages according to $\pi$").scale(step_scale)

        steps = VGroup(step_one, step_two, step_three, step_four).arrange(DOWN, aligned_edge=LEFT, buff=MED_SMALL_BUFF)
        steps.shift(DOWN * 2.5)

        for i, step in enumerate(steps):
            if i == 0:
                self.play(
                    FadeIn(screen_rect),
                    FadeIn(step)
                )
            else:
                self.play(
                    FadeIn(step)
                )
            self.wait()


class EigenValueMethodFixed2(Scene):
    def construct(self):
        markov_chain = MarkovChain(
            3,
            [(0, 1), (1, 2), (1, 0), (0, 2), (2, 1)]
        )

        markov_scale = 0.8
        markov_chain_g = MarkovChainGraph(markov_chain)
        markov_chain_g.clear_updaters()
        markov_chain_g.scale(markov_scale).shift(UP * 2)

        self.play(
            FadeIn(markov_chain_g)
        )
        self.wait()

        transpose_transition_eq = self.show_transition_equation(markov_chain_g, markov_chain)

        self.show_eigen_concept(transpose_transition_eq)

        self.show_example()

    def show_transition_equation(self, markov_chain_g, markov_chain):
        transition_eq = MathTex(r"\pi_{n + 1} = \pi_n P").next_to(markov_chain_g, RIGHT, buff=0.5)

        pi_n_1_row_vec = Matrix(
            [[r"\pi_{n + 1}(0)", r"\pi_{n+1}(1)", r"\pi_{n + 1}(2)"]],
            h_buff=2,
        ).scale(0.7)

        equals = MathTex("=")
        pi_n_row_vec =  Matrix(
            [[r"\pi_n(0)", r"\pi_n(1)", r"\pi_n(2)"]],
            h_buff=1.7,
        ).scale(0.7)

        p_matrix = Matrix(
            [
            ["P(0, 0)", "P(0, 1)", "P(0, 2)"],
            ["P(1, 0)", "P(1, 1)", "P(1, 2)"],
            ["P(2, 0)", "P(2, 1)", "P(2, 2)"],
            ],
            h_buff=2
        ).scale(0.7)

        row_vec_equation = VGroup(pi_n_1_row_vec, equals, pi_n_row_vec, p_matrix).arrange(RIGHT)

        vector_scale = 0.7

        pi_n_1_col_vec = Matrix(
            [[r"\pi_{n + 1}(0)"], [r"\pi_{n+1}(1)"], [r"\pi_{n + 1}(2)"]],
            v_buff=1,
        ).scale(vector_scale)

        pi_n_col_vec = Matrix(
            [[r"\pi_{n}(0)"], [r"\pi_{n}(1)"], [r"\pi_{n}(2)"]],
            v_buff=1,
        ).scale(vector_scale)

        p_transpose_matrix = Matrix(
            [
            ["P(0, 0)", "P(1, 0)", "P(2, 0)"],
            ["P(0, 1)", "P(1, 1)", "P(2, 1)"],
            ["P(0, 2)", "P(1, 2)", "P(2, 2)"],
            ],
            h_buff=2,
            v_buff=1,
        ).scale(vector_scale)

        col_vec_equation = VGroup(pi_n_1_col_vec, equals.copy(), p_transpose_matrix, pi_n_col_vec).arrange(RIGHT)

        equation_transformation = VGroup(row_vec_equation, col_vec_equation).arrange(DOWN, buff=0.7).scale(0.8).next_to(markov_chain_g, DOWN)

        self.play(
            FadeIn(transition_eq),
            markov_chain_g.animate.shift(LEFT * 2)
        )
        self.wait()

        self.play(
            FadeIn(row_vec_equation)
        )
        self.wait()


        self.play(
            FadeIn(col_vec_equation),
        )
        self.wait()

        transpose_transition_eq = MathTex(r"\pi_{n + 1} = P^T \pi_n").next_to(transition_eq, DOWN, aligned_edge=LEFT)
        transpose_transition_eq.shift(UP * 0.5)

        self.play(
            transition_eq.animate.shift(UP * 0.5)
        )

        self.play(
            Write(transpose_transition_eq),
        )
        self.wait()

        surround_rects = [
        SurroundingRectangle(pi_n_1_row_vec[0], color=REDUCIBLE_VIOLET, buff=SMALL_BUFF),
        SurroundingRectangle(pi_n_row_vec[0], color=REDUCIBLE_GREEN_LIGHTER, buff=SMALL_BUFF),
        SurroundingRectangle(VGroup(*[p_matrix[0][i] for i in range(9) if i % 3 == 0]), color=REDUCIBLE_YELLOW, buff=SMALL_BUFF),
        SurroundingRectangle(VGroup(*[p_matrix[0][i] for i in range(9) if i % 3 == 1]), color=REDUCIBLE_YELLOW, buff=SMALL_BUFF),
        SurroundingRectangle(VGroup(*[p_matrix[0][i] for i in range(9) if i % 3 == 2]), color=REDUCIBLE_YELLOW, buff=SMALL_BUFF),
        SurroundingRectangle(pi_n_1_col_vec[0], color=REDUCIBLE_VIOLET, buff=SMALL_BUFF),
        SurroundingRectangle(pi_n_col_vec[0], color=REDUCIBLE_GREEN_LIGHTER, buff=SMALL_BUFF),
        SurroundingRectangle(p_transpose_matrix[0][:3], color=REDUCIBLE_YELLOW, buff=SMALL_BUFF/1.5),
        SurroundingRectangle(p_transpose_matrix[0][3:6], color=REDUCIBLE_YELLOW, buff=SMALL_BUFF/1.5),
        SurroundingRectangle(p_transpose_matrix[0][6:9], color=REDUCIBLE_YELLOW, buff=SMALL_BUFF/1.5),
        ]
        self.play(
            *[FadeIn(r) for i, r in enumerate(surround_rects) if i < 5]
        )
        self.wait()

        self.play(
            *[TransformFromCopy(surround_rects[i], surround_rects[i + 5]) for i in range(5)],
        )
        self.wait()

        self.play(
            FadeOut(markov_chain_g),
            FadeOut(equation_transformation),
            FadeOut(transition_eq),
            *[FadeOut(r) for r in surround_rects],
            transpose_transition_eq.animate.move_to(UP * 3.5)
        )
        self.wait()

        return transpose_transition_eq

    def show_eigen_concept(self, transpose_transition_eq):
        dist_between_nodes = 3
        transition_matrix = np.array([[0.3, 0.7], [0.4, 0.6]])
        markov_chain = MarkovChain(
            2,
            [(0, 1), (1, 0)],
            transition_matrix=transition_matrix,
            dist=np.array([0.9, 0.1]),
        )

        markov_chain_g = MarkovChainGraph(
            markov_chain,
            layout={
                0: LEFT * dist_between_nodes / 2,
                1: RIGHT * dist_between_nodes / 2,
            },
        )
        markov_chain_g.scale(1).shift(UP * 2.5)
        markov_chain_t_labels = markov_chain_g.get_transition_labels()

        self_edges = self.get_edges(markov_chain_g)
        labels = [self.get_label(self_edges[(u, v)], transition_matrix[u][v]) for u, v in self_edges]
        self_edges_group = VGroup(*[obj for obj in list(self_edges.values()) + labels])
        markov_chain_group = VGroup(markov_chain_g, markov_chain_t_labels, self_edges_group)
        self.play(
            FadeIn(markov_chain_group)
        )
        self.wait()

        markov_chain_sim = MarkovChainSimulator(markov_chain, markov_chain_g, num_users=50)
        users = markov_chain_sim.get_users()

        purple_plane = NumberPlane(
            x_range=[0, 1, 0.25],
            y_range=[0, 1, 0.25],
            x_length=7,
            y_length=4.5,
            background_line_style={
                "stroke_color": REDUCIBLE_VIOLET,
                "stroke_width": 3,
                "stroke_opacity": 0.5,
            },
            # faded_line_ratio=4,
            axis_config={"stroke_color": REDUCIBLE_VIOLET, "stroke_width": 0, "include_numbers": True, "numbers_to_exclude": [0.25, 0.75]},
        ).move_to(DOWN * 1)

        surround_plane = Polygon(
            purple_plane.coords_to_point(0, 0),
            purple_plane.coords_to_point(0, 1),
            purple_plane.coords_to_point(1, 1),
            purple_plane.coords_to_point(1, 0),
        ).set_stroke(color=REDUCIBLE_VIOLET)

        self.play(
            FadeIn(purple_plane),
            FadeIn(surround_plane)
        )
        self.wait()

        current_dist = markov_chain.get_current_dist()
        current_vector = self.get_vector(
            current_dist,
            purple_plane,
            r"\pi_0",
            max_tip_length_to_length_ratio=0.1)
        num_steps = 5

        self.play(
             *[FadeIn(u) for u in users],
            FadeIn(current_vector),
        )
        self.wait()

        for i in range(1, num_steps + 1):
            transition_animations = markov_chain_sim.get_instant_transition_animations()
            self.play(
                current_vector.animate.become(
                    self.get_vector(
                        markov_chain.get_current_dist(),
                        purple_plane,
                        r"\pi_{0}".format(i),
                        max_tip_length_to_length_ratio=0.1
                    )
                ),
                *transition_animations,
            )
            self.wait()

        stationary_dist_def = MathTex(r"\pi = P^T \pi").move_to(transpose_transition_eq.get_center())

        self.play(
            ReplacementTransform(transpose_transition_eq, stationary_dist_def)
        )
        self.wait()

        title = Text("Eigenvalues/Eigenvectors", font=REDUCIBLE_FONT, weight=BOLD).scale(0.8)
        title.move_to(UP * 3.5)

        self.play(
            FadeOut(purple_plane),
            FadeOut(current_vector),
            Write(title),
            FadeOut(surround_plane),
            *[FadeOut(u) for u in users],
            FadeOut(markov_chain_group),
            stationary_dist_def.animate.shift(DOWN)
        )
        self.wait()

        eigen_def = MathTex(r"\lambda \vec{v} = A \vec{v}").next_to(stationary_dist_def, DOWN)

        self.play(
            FadeIn(eigen_def)
        )
        self.wait()

        left_plane = NumberPlane(
            x_range=[-2, 2, 0.5],
            y_range=[-2, 2, 0.5],
            x_length=4.5,
            y_length=3.5,
            background_line_style={
                "stroke_color": REDUCIBLE_VIOLET,
                "stroke_width": 3,
                "stroke_opacity": 0.5,
            },
            # faded_line_ratio=4,
            axis_config={"stroke_color": REDUCIBLE_VIOLET},
        ).move_to(LEFT * 3.5 + DOWN * 1)

        left_surround_plane = SurroundingRectangle(left_plane, buff=0, color=REDUCIBLE_VIOLET)

        right_plane = NumberPlane(
            x_range=[-2, 2, 0.5],
            y_range=[-2, 2, 0.5],
            x_length=4.5,
            y_length=3.5,
            background_line_style={
                "stroke_color": REDUCIBLE_VIOLET,
                "stroke_width": 3,
                "stroke_opacity": 0.5,
            },
            # faded_line_ratio=4,
            axis_config={"stroke_color": REDUCIBLE_VIOLET},
        ).move_to(RIGHT * 3.5 + DOWN * 1)

        right_surround_plane = SurroundingRectangle(right_plane, buff=0, color=REDUCIBLE_VIOLET)

        self.play(
            FadeIn(left_plane),
            FadeIn(left_surround_plane),
        )
        self.wait()

        v = np.array([0.4, 0.6])
        left_vector = self.get_vector(v, left_plane, r"\vec{v}", max_tip_length_to_length_ratio=0.15)

        self.play(
            FadeIn(left_vector)
        )
        self.wait()

        left_to_right_arr = Arrow(left_plane.get_right(), right_plane.get_left(), max_tip_length_to_length_ratio=0.1).set_color(GRAY)
        transformation = MathTex(r"A \vec{v}").scale(0.8).next_to(left_to_right_arr, UP).shift(DOWN * SMALL_BUFF)

        self.play(
            Write(left_to_right_arr),
            Write(transformation)
        )

        self.play(
            FadeIn(right_plane),
            FadeIn(right_surround_plane),
        )
        self.wait()

        right_vector = self.get_vector(v * 1.2, right_plane, r"\lambda \vec{v}", max_tip_length_to_length_ratio=0.15)

        self.play(
            FadeIn(right_vector)
        )
        self.wait()

        scales = [2, 1.2, -0.4, -1.2, -2, -1.2, -0.4, 1]

        for scale in scales:
            new_right_vec = self.get_vector(v * scale, right_plane, r"\lambda \vec{v}", max_tip_length_to_length_ratio=0.15)
            self.play(
                right_vector.animate.become(new_right_vec),
                rate_func=linear
            )

        self.wait()

        conclusion = Tex(r"$\pi$ is unique eigenvector corresponding to $\lambda = 1$ of $P^T$").scale(0.8)
        conclusion.move_to(DOWN * 3.3)

        self.play(
            FadeIn(conclusion)
        )
        self.wait()

        self.clear()

    def get_vector(self, dist, plane, tex, **kwargs):
        start_c = plane.coords_to_point(0, 0)
        end_c = plane.coords_to_point(dist[0], dist[1])
        arrow = Arrow(start_c, end_c, buff=0, **kwargs)
        arrow.set_color(REDUCIBLE_YELLOW)
        label = self.get_tex_label(tex, arrow, normalize(end_c - start_c))
        return VGroup(arrow, label)

    def get_tex_label(self, tex, arrow, direction):
        label = MathTex(tex).scale(0.7)
        label.add_background_rectangle()
        label.next_to(arrow, direction=direction, buff=SMALL_BUFF)
        return label

    def get_edges(self, markov_chain_g):
        edge_map = {}
        edge_map[(0, 0)] = self.get_self_edge(markov_chain_g, 0)
        edge_map[(1, 1)] = self.get_self_edge(markov_chain_g, 1)
        return edge_map

    def get_self_edge(self, markov_chain_g, state):
        vertices = markov_chain_g.vertices
        if state == 1:
            angle = -1.6 * PI
        else:
            angle = 1.6 * PI
        edge = CustomCurvedArrow(
            vertices[state].get_top(), vertices[state].get_bottom(), angle=angle
        ).set_color(REDUCIBLE_VIOLET)
        return edge

    def get_label(self, edge, prob, scale=0.3):
        return (
            Text(str(prob), font=REDUCIBLE_MONO)
            .set_stroke(BLACK, width=8, background=True, opacity=0.8)
            .scale(scale)
            .move_to(edge.point_from_proportion(0.15))
        )

    def show_example(self):
        markov_ch = MarkovChain(
            4,
            edges=[
                (0, 1),
                # (1, 0),
                (1, 2),
                (2, 1),
                (2, 0),
                (2, 3),
                (0, 3),
                # (3, 1),
                (3, 2),
            ],
        )

        markov_ch_mob = MarkovChainGraph(
            markov_ch,
            curved_edge_config={"radius": 2},
            straight_edge_config={"max_tip_length_to_length_ratio": 0.06},
            layout_scale=2,
            layout="circular",
        ).shift(LEFT * 4 + UP * 1.5)

        p = MathTex(r"\text{eig}(P^T)").scale(0.8)

        # the stationary dists are eigvecs with eigval 1 from the P.T
        eig_vals_P, eig_vecs_P = np.linalg.eig(markov_ch.get_transition_matrix().T)
        eig_vecs_P = eig_vecs_P.T.astype(float)
        eig_vals_P = eig_vals_P.astype(float)
        # floating point issue
        eig_vals_P[3] = 0.0
        print(eig_vecs_P)
        print(eig_vals_P)

        lambdas_with_value = VGroup(
            *[
                MathTex(f"\lambda_{n} &= {v:.1f}")
                for n, v in zip(range(len(markov_ch.get_states())), eig_vals_P)
            ],
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)

        pi_vectors_example = (
            VGroup(
                *[
                    MathTex(r"\vec{v}_{" + str(n) + "}")
                    for n in range(len(markov_ch.get_states()))
                ],
            )
            .arrange(DOWN, buff=0.2)
            .to_edge(RIGHT * 2)
            .shift(UP * 2)
        )

        p_brace = Brace(lambdas_with_value, LEFT)

        p_with_eigs = (
            VGroup(p.scale(1.4), p_brace, lambdas_with_value, pi_vectors_example)
            .set_stroke(width=8, background=True)
            .arrange(RIGHT, buff=0.5)
            .move_to(DOWN * 2 + LEFT * 2.7)
        )

        labels = markov_ch_mob.get_transition_labels()

        self.play(
            Write(markov_ch_mob),
        )
        self.play(FadeIn(labels))
        self.wait()

        P_matrix = self.matrix_to_mob(markov_ch.get_transition_matrix()).scale(0.5)

        p_matrix_group = VGroup(MathTex("P"), MathTex("="), P_matrix).arrange(RIGHT).next_to(markov_ch_mob, RIGHT, buff=1)

        self.play(
            FadeIn(p_matrix_group)
        )
        self.wait()


        eig_index = np.ravel(np.argwhere(eig_vals_P.round(1) == 1.0))[0]

        underline_eig_1 = Underline(lambdas_with_value[eig_index]).set_color(
            REDUCIBLE_YELLOW
        )

        stationary_pi = MathTex(r"\vec{v}_" + str(eig_index) + " = ")
        stationary_dist = self.vector_to_mob(eig_vecs_P[eig_index]).scale(0.3)

        vertices_down = VGroup(
            *[s.copy().scale(0.5) for s in markov_ch_mob.vertices.values()]
        ).arrange(DOWN, buff=0.13)

        stationary_distribution = (
            VGroup(stationary_pi, vertices_down, stationary_dist)
            .arrange(RIGHT, buff=0.15)
            .next_to(p_with_eigs, RIGHT, buff=2)
        ).scale(1.5)

        stationary_dist_normalized = (
            self.vector_to_mob(
                [e / sum(eig_vecs_P[eig_index]) for e in eig_vecs_P[eig_index]]
            )
            .scale_to_fit_height(stationary_dist.height)
            .move_to(stationary_dist)
        )
        print([e / sum(eig_vecs_P[eig_index]) for e in eig_vecs_P[eig_index]])
        print('sum', np.sum([e / sum(eig_vecs_P[eig_index]) for e in eig_vecs_P[eig_index]]))

        # VGroup(p_with_eigs, stationary_dist).arrange(RIGHT, buff=1.5).shift(DOWN * 1.5)
        self.play(
            FadeIn(p_with_eigs, shift=UP*0.3)
        )
        self.wait()

        self.play(Succession(Create(underline_eig_1), FadeOut(underline_eig_1)))
        self.wait()

        self.play(FadeIn(stationary_distribution, shift=RIGHT * 0.4))
        self.wait()

        new_stationary_pi = MathTex(r"\pi =").move_to(stationary_pi.get_center()).scale(1.5)

        self.play(
            Transform(stationary_dist, stationary_dist_normalized),
            Transform(stationary_pi, new_stationary_pi),
        )
        self.wait()

    def apply_matrix_to_vector(self, matrix: np.ndarray, mob_vector: Vector):
        vector = mob_vector.get_vector()[:2]
        trans_vector = np.dot(matrix, vector)

        return mob_vector.animate.put_start_and_end_on(
            mob_vector.start, [trans_vector[0], trans_vector[1], 0]
        )

    def matrix_to_mob(self, matrix: np.ndarray, has_background_color=False):
        str_repr = [[f"{a:.2f}" for a in row] for row in matrix]
        return Matrix(
            str_repr,
            left_bracket="[",
            right_bracket="]",
            element_to_mobject=Text,
            include_background_rectangle=has_background_color,
            element_to_mobject_config={"font": REDUCIBLE_MONO},
            h_buff=2.3,
            v_buff=1.3,
        )

    def vector_to_mob(self, vector: Iterable):
        str_repr = [[f"{a:.2f}"] for a in vector]

        return Matrix(
            str_repr,
            left_bracket="[",
            right_bracket="]",
            element_to_mobject=Text,
            element_to_mobject_config={"font": REDUCIBLE_MONO},
            h_buff=2.3,
            v_buff=1.3,
        )

class EigenvalFinalScene(Scene):
    def construct(self):
        screen_rect = ScreenRectangle(height=5).shift(UP * 0.5)
        long_term = Tex(r"Long term behavior of system $\Leftrightarrow$ eigenvalues/eigenvectors").scale(0.8)

        long_term.next_to(screen_rect, DOWN)

        self.add(screen_rect)
        self.wait()

        self.play(
            FadeIn(long_term)
        )
        self.wait()

class Patreons(Scene):
    def construct(self):
        thanks = Tex("Special Thanks to These Patreons").scale(1.2)
        patreons = ["Burt Humburg", "Winston Durand", r"Adam D\v{r}ínek", "kerrytazi",  "Andreas", "Matt Q"]
        patreon_text = VGroup(thanks, VGroup(*[Tex(name).scale(0.9) for name in patreons]).arrange_in_grid(rows=3, buff=(0.75, 0.25)))
        patreon_text.arrange(DOWN)
        patreon_text.to_edge(DOWN)

        self.play(
            Write(patreon_text[0])
        )
        self.play(
            *[Write(text) for text in patreon_text[1]]
        )
        self.wait()


class BeforeAfterComparison(Scene):
    def construct(self):
        bar_names = ["AltaVista", "Excite", "Lycos", "Yahoo!", "Infoseek"]
        bar_values = [26, 14, 13, 9, 5]
        bar_title = Text("Search Engine Usage in 1997", font=REDUCIBLE_FONT, weight=BOLD).scale(0.7)
        y_axis_label = Text("Usage (%)", font=REDUCIBLE_FONT).scale(0.6)

        chart = BarChart(
            values=bar_values,
            bar_names=bar_names,
            bar_colors=[REDUCIBLE_YELLOW, REDUCIBLE_VIOLET, REDUCIBLE_GREEN_LIGHTER, REDUCIBLE_PURPLE, REDUCIBLE_GREEN_DARKER],
            y_range=[0, 100, 25],
            y_length=1.8,
            x_length=8,
            x_axis_config={"font_size": 36},
        ).move_to(UP * 1.8)
        y_axis_label.next_to(chart, LEFT).shift(UP * SMALL_BUFF * 4)
        bar_title.next_to(chart, UP)

        self.play(
            FadeIn(chart),
            FadeIn(bar_title),
            FadeIn(y_axis_label)
        )

        bottom_title = Text("Search Engine Usage in 2022", font=REDUCIBLE_FONT, weight=BOLD).scale(0.7)
        bottom_chart = BarChart(
            values=[92, 3, 1.3, 1.2, 0.87],
            bar_names=["Google", "Bing", "Yahoo", "Baidu", "Yandex"],
            bar_colors=[REDUCIBLE_YELLOW, REDUCIBLE_VIOLET, REDUCIBLE_GREEN_LIGHTER, REDUCIBLE_PURPLE, REDUCIBLE_GREEN_DARKER],
            y_range=[0, 100, 25],
            y_length=2,
            x_length=8,
            x_axis_config={"font_size": 36},
        ).move_to(DOWN * 2)
        bottom_y_axis_label = y_axis_label.copy().next_to(bottom_chart, LEFT).shift(UP * SMALL_BUFF * 4)
        bottom_title.next_to(bottom_chart, UP)

        self.play(
            FadeIn(bottom_chart),
            FadeIn(bottom_title),
            FadeIn(bottom_y_axis_label),
        )
        self.wait()






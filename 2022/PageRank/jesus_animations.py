import sys
from math import dist
from typing import Iterable

sys.path.insert(1, "common/")
from fractions import Fraction

from manim import *

config["assets_dir"] = "assets"

from reducible_colors import *


from markov_chain import *
from classes import RVariable, RDecimalNumber


class TransitionMatrixCorrected3(MovingCameraScene):
    def construct(self):
        frame = self.camera.frame
        markov_ch = MarkovChain(
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

        markov_ch_mob = MarkovChainGraph(
            markov_ch,
            curved_edge_config={"radius": 2},
            layout_scale=2.6,
        )

        markov_ch_mob.clear_updaters()

        markov_ch_sim = MarkovChainSimulator(markov_ch, markov_ch_mob, num_users=50)
        users = markov_ch_sim.get_users()

        trans_matrix_mob = self.matrix_to_mob(markov_ch.get_transition_matrix())

        p_equals = (
            Text("P = ", font=REDUCIBLE_FONT, weight=BOLD)
            .scale(0.3)
            .next_to(trans_matrix_mob, LEFT)
        )

        vertices_down = VGroup(
            *[dot.copy().scale(0.4) for dot in markov_ch_mob.vertices.values()]
        ).arrange(DOWN, buff=0.05)

        matrix = VGroup(p_equals, vertices_down, trans_matrix_mob).arrange(
            RIGHT, buff=0.1
        )

        vertices_right = (
            VGroup(*[dot.copy().scale(0.4) for dot in markov_ch_mob.vertices.values()])
            .arrange(RIGHT, buff=0.27)
            .next_to(trans_matrix_mob, UP, buff=0.1)
        )

        prob_labels = markov_ch_mob.get_transition_labels(scale=0.3)

        ################# ANIMATIONS #################

        self.play(Write(markov_ch_mob), run_time=1)

        self.wait()

        self.play(self.focus_on(markov_ch_mob, buff=3.2).shift(LEFT * 3))

        # isolate node 0
        mask_0 = (
            Difference(
                Rectangle(height=10, width=20),
                markov_ch_mob.vertices[0].copy().scale(1.05),
            )
            .set_color(BLACK)
            .set_stroke(width=0)
            .set_opacity(0.7)
        )
        self.play(FadeIn(mask_0))
        self.wait()
        self.play(FadeOut(mask_0))

        mask_but_0 = Rectangle(width=20, height=20)
        # only way to create a mask of several mobjects is to
        # keep poking the holes on the mask one by one
        for v in list(markov_ch_mob.vertices.values())[1:]:
            mask_but_0 = Difference(mask_but_0, v.copy().scale(1.05))

        mask_but_0.set_color(BLACK).set_stroke(width=0).set_opacity(0.7)

        self.play(FadeIn(mask_but_0))
        self.wait()
        self.play(FadeOut(mask_but_0))

        self.play(
            markov_ch_mob.vertices[1].animate.set_opacity(0.3),
            markov_ch_mob.edges[(2, 1)].animate.set_opacity(0.3),
            markov_ch_mob.edges[(1, 2)].animate.set_opacity(0.3),
            markov_ch_mob.edges[(4, 1)].animate.set_opacity(0.3),
            markov_ch_mob.edges[(3, 4)].animate.set_opacity(0.3),
            markov_ch_mob.edges[(2, 3)].animate.set_opacity(0.3),
            markov_ch_mob.edges[(0, 3)].animate.set_opacity(0.3),
            markov_ch_mob.edges[(0, 2)].animate.set_opacity(0.3),
        )

        self.wait()

        pi_dists = []
        for s in markov_ch.get_states():
            state = markov_ch_mob.vertices[s]
            label_direction = normalize(state.get_center() - markov_ch_mob.get_center())
            pi_dists.append(
                MathTex(f"\pi_n({s})")
                .scale(0.6)
                .next_to(state, label_direction, buff=0.1)
            )

        pi_dists_vg = VGroup(*pi_dists)

        self.play(FadeIn(pi_dists_vg))

        pi_next_0 = MathTex("\pi_{n+1}(0)").scale(0.8)

        math_str = [
            "\pi_{n}" + f"({i})" + f"&\cdot P({i},0)"
            for i in range(len(markov_ch.get_states()))
        ]

        dot_prod_mob = MathTex("\\\\".join(math_str)).scale(0.6)

        brace = Brace(dot_prod_mob, LEFT)
        equation_explanation = (
            VGroup(pi_next_0, brace, dot_prod_mob)
            .arrange(RIGHT, buff=0.1)
            .move_to(frame.get_left(), aligned_edge=LEFT)
            .shift(RIGHT * 1)
        )
        plus_signs = (
            VGroup(*[Tex("+").scale(0.7) for _ in range(4)])
            .arrange(DOWN, buff=0.22)
            .next_to(dot_prod_mob, RIGHT, buff=0.1, aligned_edge=UP)
            .shift(DOWN * 0.02)
        )

        self.play(FadeIn(pi_next_0))

        self.wait()

        self.play(
            FadeIn(brace),
            FadeIn(dot_prod_mob[0][0:5]),
            FadeIn(dot_prod_mob[0][12:17]),
            FadeIn(dot_prod_mob[0][24:29]),
            FadeIn(dot_prod_mob[0][36:41]),
            FadeIn(dot_prod_mob[0][48:53]),
        )

        self.wait()

        self.play(
            FadeIn(dot_prod_mob[0][5:12]),
            FadeIn(dot_prod_mob[0][17:24]),
            FadeIn(dot_prod_mob[0][29:36]),
            FadeIn(dot_prod_mob[0][41:48]),
            FadeIn(dot_prod_mob[0][53:]),
        )

        self.wait()

        self.play(FadeIn(plus_signs))

        self.wait()

        full_equation = VGroup(equation_explanation, plus_signs)

        self.play(
            full_equation.animate.shift(UP * 2)
        )
        self.wait()

        ##### camera pans down for explanation
        # self.play(frame.animate.shift(DOWN * 7), run_time=1.5)

        # self.wait()

        # math_notation_title = (
        #     Text("Some bits of mathematical notation", font=REDUCIBLE_FONT, weight=BOLD)
        #     .scale(0.5)
        #     .move_to(frame.get_corner(UL), aligned_edge=UL)
        #     .shift(DR * 0.5)
        # )
        # self.play(FadeIn(math_notation_title, shift=UP * 0.3), FadeOut(full_equation))

        dist_definition = (
            MathTex(
                # r"\vec{\pi_n} = [\pi_n(0), \pi_n(1), \pi_n(2), \pi_n(3), \pi_n(4) ]",
                r"\vec{\pi}_n = \begin{bmatrix} \pi_n(0) & \pi_n(1) & \pi_n(2) & \pi_n(3) & \pi_n(4) \end{bmatrix}",
            )
            .scale(0.6)
            .next_to(full_equation, DOWN, buff=1, aligned_edge=LEFT)
        )

        trans_column_def = (
            MathTex(
                r"\vec{P}_{i,0} = \begin{bmatrix} P(0,0) \\ P(1,0) \\ P(2,0) \\ P(3,0) \\ P(4,0) \end{bmatrix}"
            )
            .scale(0.5)
            .next_to(dist_definition, DOWN, aligned_edge=LEFT)
        )
        self.play(
            FadeIn(dist_definition, shift=UP * 0.3),
            FadeIn(trans_column_def, shift=UP * 0.3)
        )

        self.wait()
        next_dist_def = (
            MathTex(r"\vec{\pi}_{n+1}(0) = \vec{\pi}_n \cdot \vec{P}_{i,0}}")
            .scale(1)
            .next_to(trans_column_def, DOWN, aligned_edge=LEFT)
        )
        self.play(
            FadeIn(next_dist_def, shift=UP * 0.3)
        )

        self.wait()
        # #### camera frame comes back
        # self.play(
        #     frame.animate.shift(UP * 7),
        #     markov_ch_mob.vertices[1].animate.set_stroke(opacity=1),
        #     markov_ch_mob.vertices[1].animate.set_opacity(0.5),
        #     markov_ch_mob._labels[1].animate.set_opacity(1),
        #     markov_ch_mob.edges[(2, 1)].animate.set_opacity(1),
        #     markov_ch_mob.edges[(1, 2)].animate.set_opacity(1),
        #     markov_ch_mob.edges[(4, 1)].animate.set_opacity(1),
        #     markov_ch_mob.edges[(3, 4)].animate.set_opacity(1),
        #     markov_ch_mob.edges[(2, 3)].animate.set_opacity(1),
        #     markov_ch_mob.edges[(0, 3)].animate.set_opacity(1),
        #     markov_ch_mob.edges[(0, 2)].animate.set_opacity(1),
        #     run_time=1.5,
        # )

        self.play(
            FadeOut(dist_definition),
            FadeOut(trans_column_def),
            FadeOut(next_dist_def),
            FadeOut(full_equation),
            *[arr.animate.set_opacity(1) for arr in markov_ch_mob.edges.values()],
            markov_ch_mob.vertices[1][0].animate.set_stroke(opacity=1).set_fill(opacity=0.5),
            markov_ch_mob.vertices[1][1].animate.set_fill(opacity=1),
        )

        self.wait()

        matrix_complete = (
            VGroup(vertices_right, matrix)
            .scale(1.7)
            .move_to(frame.get_left(), aligned_edge=LEFT)
            .shift(RIGHT * 0.6 + UP * 0.4)
        )

        self.play(FadeIn(matrix_complete), FadeIn(prob_labels))
        self.wait()
        dot_product_def = (
            MathTex(r"\vec{\pi}_{n+1} &= \vec{\pi}_n \cdot P")
            .scale(1.3)
            .next_to(trans_matrix_mob, DOWN, buff=0.5)
        )

        # first iteration
        surr_rect = SurroundingRectangle(
            trans_matrix_mob[0][0 : len(markov_ch.get_states())], color=REDUCIBLE_YELLOW
        )

        not_relevant_labels_tuples = list(
            filter(lambda x: x[0] != 0, markov_ch_mob.labels.keys())
        )
        not_relevant_labels = [
            markov_ch_mob.labels[t] for t in not_relevant_labels_tuples
        ]
        not_relevant_arrows = [
            markov_ch_mob.edges[t] for t in not_relevant_labels_tuples
        ]

        self.play(
            Write(surr_rect),
            *[
                markov_ch_mob.labels[t].animate.set_opacity(0.4)
                for t in not_relevant_labels_tuples
            ],
            *[
                markov_ch_mob.edges[t].animate.set_opacity(1)
                for t in [(0, 3), (0, 2)]
            ],
            *[arr.animate.set_opacity(0.3) for arr in not_relevant_arrows],
        )
        self.wait()

        for s in markov_ch.get_states()[1:]:
            # self.play(
            #     *[l.animate.set_opacity(1) for l in not_relevant_labels],
            #     *[arr.animate.set_opacity(1) for arr in not_relevant_arrows],
            # )

            not_relevant_labels_tuples = list(
                filter(lambda x: x[0] != s, markov_ch_mob.labels.keys())
            )
            
            relevant_tuples = list(filter(lambda x: x[0] == s, markov_ch_mob.labels.keys()))
            
            not_relevant_labels = [
                markov_ch_mob.labels[t] for t in not_relevant_labels_tuples
            ]
            not_relevant_arrows = [
                markov_ch_mob.edges[t] for t in not_relevant_labels_tuples
            ]

            relevant_arrows = [
                markov_ch_mob.edges[t] for t in relevant_tuples
            ]

            print('Not relevant labels', not_relevant_labels)
            print('Relevant labels', list(relevant_tuples))

            self.play(
                surr_rect.animate.shift(DOWN * 0.44),
                *[l.animate.set_opacity(0.2) for l in not_relevant_labels],
                *[arr.animate.set_opacity(0.3) for arr in not_relevant_arrows],
                *[arr.animate.set_opacity(1) for arr in relevant_arrows],
                *[markov_ch_mob.labels[t].animate.set_opacity(1) for t in relevant_tuples],
            )

            self.wait()

        self.play(FadeIn(dot_product_def, shift=UP * 0.3))
        self.wait()

        self.play(
            *[l.animate.set_opacity(1) for l in not_relevant_labels],
            *[arr.animate.set_opacity(1) for arr in not_relevant_arrows],
            FadeOut(surr_rect),
        )

        ######### DEFINE STATIONARY DISTRIBUTON #########

        self.play(
            self.camera.frame.animate.scale(1.3).shift(LEFT * 1.2),
            FadeOut(matrix),
            FadeOut(dot_product_def),
            FadeOut(vertices_right),
            FadeOut(prob_labels),
            FadeOut(pi_dists_vg),
        )
        self.wait()

        stationary_dist_annotation = (
            Text("A distribution is stationary if:", font=REDUCIBLE_FONT, weight=BOLD)
            .scale(0.65)
            .next_to(markov_ch_mob, LEFT, buff=4.5, aligned_edge=RIGHT)
            .shift(UP * 0.8)
        )
        stationary_dist_tex = (
            MathTex(r"\pi = \pi P")
            .scale_to_fit_width(stationary_dist_annotation.width - 2)
            .next_to(stationary_dist_annotation, DOWN, buff=0.8)
        )
        self.play(Write(stationary_dist_annotation), run_time=0.8)
        self.play(FadeIn(stationary_dist_tex))

        self.wait()

        count_labels = self.get_current_count_mobs(
            markov_chain_g=markov_ch_mob, markov_chain_sim=markov_ch_sim, use_dist=True
        )

        self.wait()

        self.play(
            *[FadeIn(l) for l in count_labels.values()],
            *[FadeIn(u) for u in users],
        )

        # accelerate the simulation so
        # we only show the stationary distribution
        # [markov_ch_sim.transition() for _ in range(10)]
        for i in range(25):
            transition_map = markov_ch_sim.get_lagged_smooth_transition_animations()
            count_labels, count_transforms = self.update_count_labels(
                count_labels, markov_ch_mob, markov_ch_sim, use_dist=True
            )
            self.play(
                *[LaggedStart(*transition_map[i]) for i in markov_ch.get_states()]
                + count_transforms
            )

        self.wait()

        self.play(
            FadeOut(stationary_dist_annotation),
            FadeOut(stationary_dist_tex),
            *[FadeOut(u) for u in users],
            *[FadeOut(l) for l in count_labels.values()],
            self.camera.frame.animate.scale(0.9),
        )

        ############ IMPORTANT QUESTIONS ############

        """
        it's important to address the question of whether a unique distribution even exists for a Markov chain. 
        And, a critical point in our model is if any initial distribution eventually converges to the stationary
        distribution
        """

        question_1 = (
            Text(
                "→ Is there a unique stationary distribution?",
                font=REDUCIBLE_FONT,
                weight=BOLD,
            )
            .scale(0.5)
            .next_to(markov_ch_mob, LEFT, buff=1.3)
            .shift(UP * 0.5)
        )

        question_2 = (
            MarkupText(
                """
                → Does every initial distribution converge
                to the stationary one?
                """,
                font=REDUCIBLE_FONT,
                weight=BOLD,
            )
            .scale(0.5)
            .next_to(question_1, DOWN, buff=1, aligned_edge=LEFT)
        )

        self.play(Write(question_1))
        self.wait()
        self.play(Write(question_2))

        self.wait()

        self.play(
            question_1.animate.set_opacity(0.5)
        )
        self.wait()

    def focus_on(self, mobject, buff=2):
        return self.camera.frame.animate.set_width(mobject.width * buff).move_to(
            mobject
        )

    def matrix_to_mob(self, matrix: np.ndarray):
        str_repr = [[f"{a:.2f}" for a in row] for row in matrix]
        return Matrix(
            str_repr,
            left_bracket="[",
            right_bracket="]",
            element_to_mobject=Text,
            element_to_mobject_config={"font": REDUCIBLE_MONO},
            h_buff=2.3,
            v_buff=1.3,
        ).scale(0.2)

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


class BruteForceMethod(TransitionMatrixCorrected3):
    def construct(self):

        frame = self.camera.frame
        markov_ch = MarkovChain(
            4,
            edges=[
                (2, 0),
                (2, 3),
                (0, 3),
                (3, 1),
                (2, 1),
                (1, 2),
            ],
            dist=[0.2, 0.5, 0.2, 0.1],
        )

        markov_ch_mob = MarkovChainGraph(
            markov_ch,
            curved_edge_config={"radius": 2, "tip_length": 0.1},
            straight_edge_config={"max_tip_length_to_length_ratio": 0.08},
            layout="circular",
        )

        markov_ch_sim = MarkovChainSimulator(markov_ch, markov_ch_mob, num_users=50)
        users = markov_ch_sim.get_users()

        count_labels = self.get_current_count_mobs(
            markov_chain_g=markov_ch_mob, markov_chain_sim=markov_ch_sim, use_dist=True
        )

        stationary_dist_tex = (
            MathTex("\pi_{n+1} = \pi_{n} P")
            .scale(1.3)
            .next_to(markov_ch_mob, RIGHT, buff=6, aligned_edge=LEFT)
            .shift(UP * 2)
        )
        ############### ANIMATIONS

        self.play(Write(markov_ch_mob))
        self.play(
            LaggedStart(*[FadeIn(u) for u in users]),
            LaggedStart(
                *[FadeIn(l) for l in count_labels.values()],
            ),
            run_time=0.5,
        )

        self.play(frame.animate.shift(RIGHT * 4 + UP * 0.5).scale(1.2))

        title = (
            Text("Brute Force Method", font=REDUCIBLE_FONT, weight=BOLD)
            .scale(1)
            .move_to(frame.get_top())
            .shift(DOWN * 0.9)
        )
        self.play(FadeIn(title))
        self.wait()

        self.play(Write(stationary_dist_tex[0][-1]))
        self.play(Write(stationary_dist_tex[0][5:7]))
        self.play(Write(stationary_dist_tex[0][:5]))

        last_dist = markov_ch_sim.get_user_dist().values()
        last_dist_mob = (
            self.vector_to_mob(last_dist)
            .scale_to_fit_width(stationary_dist_tex[0][5:7].width)
            .next_to(stationary_dist_tex[0][5:7], DOWN, buff=0.4)
        )
        self.play(FadeIn(last_dist_mob))
        self.wait()

        # first iteration
        transition_map = markov_ch_sim.get_lagged_smooth_transition_animations()
        count_labels, count_transforms = self.update_count_labels(
            count_labels, markov_ch_mob, markov_ch_sim, use_dist=True
        )

        current_dist = markov_ch_sim.get_user_dist().values()
        current_dist_mob = (
            self.vector_to_mob(current_dist)
            .scale_to_fit_width(last_dist_mob.width)
            .next_to(stationary_dist_tex[0][:4], DOWN, buff=0.4)
        )
        self.play(
            *[LaggedStart(*transition_map[i]) for i in markov_ch.get_states()],
            *count_transforms,
            FadeIn(current_dist_mob),
        )

        distance = dist(current_dist, last_dist)
        distance_definition = (
            MathTex(r"D(\pi_{n+1}, \pi_{n}) =  ||\pi_{n+1} - \pi_{n}||_2")
            .scale(0.7)
            .next_to(stationary_dist_tex, DOWN, buff=2.5, aligned_edge=LEFT)
        )
        distance_mob = (
            VGroup(
                MathTex("D(\pi_{" + str(1) + "}, \pi_{" + str(0) + "})"),
                MathTex("="),
                Text(f"{distance:.5f}", font=REDUCIBLE_MONO).scale(0.6),
            )
            .arrange(RIGHT, buff=0.2)
            .scale(0.7)
            .next_to(stationary_dist_tex, DOWN, buff=2.5, aligned_edge=LEFT)
        )

        tolerance = 0.001
        tolerance_mob = (
            Text(
                "Threshold = " + str(tolerance),
                font=REDUCIBLE_FONT,
                t2f={str(tolerance): REDUCIBLE_MONO},
            )
            .scale(0.4)
            .next_to(distance_mob, DOWN, buff=0.2, aligned_edge=LEFT)
        )

        self.play(FadeIn(distance_definition))
        self.wait()
        self.play(
            FadeOut(distance_definition, shift=UP * 0.3),
            FadeIn(distance_mob, shift=UP * 0.3),
        )
        self.wait()

        self.play(FadeIn(tolerance_mob, shift=UP * 0.3))

        tick = (
            SVGMobject("check.svg")
            .scale(0.1)
            .set_color(PURE_GREEN)
            .next_to(tolerance_mob, RIGHT, buff=0.3)
        )

        self.wait()
        ## start the loop
        for i in range(2, 100):
            transition_animations = markov_ch_sim.get_instant_transition_animations()

            count_labels, count_transforms = self.update_count_labels(
                count_labels, markov_ch_mob, markov_ch_sim, use_dist=True
            )

            last_dist = current_dist
            current_dist = markov_ch_sim.get_user_dist().values()

            distance = dist(current_dist, last_dist)

            i_str = str(i)
            i_minus_one_str = str(i - 1)
            new_distance_mob = (
                VGroup(
                    MathTex("D(\pi_{" + i_str + "}, \pi_{" + i_minus_one_str + "})"),
                    MathTex("="),
                    Text(f"{distance:.5f}", font=REDUCIBLE_MONO).scale(0.6),
                )
                .arrange(RIGHT, buff=0.2)
                .scale(0.7)
                .next_to(stationary_dist_tex, DOWN, buff=2.5, aligned_edge=LEFT)
            )

            run_time = 0.8 if i < 6 else 1 / i

            if i < 6:
                current_to_last_shift = current_dist_mob.animate.move_to(last_dist_mob)
                fade_last_dist = FadeOut(last_dist_mob)
                last_dist_mob = current_dist_mob

                current_dist_mob = (
                    self.vector_to_mob(current_dist)
                    .scale_to_fit_width(last_dist_mob.width)
                    .next_to(stationary_dist_tex[0][:4], DOWN, buff=0.4)
                )

                self.play(
                    *transition_animations + count_transforms,
                    current_to_last_shift,
                    fade_last_dist,
                    FadeIn(current_dist_mob),
                    FadeTransform(distance_mob, new_distance_mob),
                    run_time=run_time,
                )

                distance_mob = new_distance_mob
            else:

                self.remove(last_dist_mob)
                last_dist_mob = current_dist_mob.move_to(last_dist_mob)

                current_dist_mob = (
                    self.vector_to_mob(current_dist)
                    .scale_to_fit_width(last_dist_mob.width)
                    .next_to(stationary_dist_tex[0][:4], DOWN, buff=0.4)
                )

                self.add(current_dist_mob)

                self.play(
                    *transition_animations + count_transforms,
                    FadeTransform(distance_mob, new_distance_mob),
                    run_time=run_time,
                )
                distance_mob = new_distance_mob

            if distance <= tolerance:
                found_iteration = (
                    Text(
                        f"iteration: {str(i)}",
                        font=REDUCIBLE_FONT,
                        t2f={str(i): REDUCIBLE_MONO},
                    )
                    .scale(0.3)
                    .next_to(tick, RIGHT, buff=0.1)
                )
                self.play(
                    FadeIn(tick, shift=UP * 0.3),
                    FadeIn(found_iteration, shift=UP * 0.3),
                )

                # get out of the loop
                break

        self.wait()

        ### the final distribution is:

        self.play(
            FadeOut(distance_mob),
            FadeOut(tolerance_mob),
            FadeOut(found_iteration),
            FadeOut(tick),
            FadeOut(last_dist_mob),
            current_dist_mob.animate.next_to(stationary_dist_tex, DOWN, buff=1.5).scale(
                2
            ),
        )
        self.wait()
        vertices_down = (
            VGroup(*[dot.copy().scale(0.8) for dot in markov_ch_mob.vertices.values()])
            .arrange(DOWN, buff=0.3)
            .next_to(current_dist_mob.copy().shift(RIGHT * 0.25), LEFT, buff=0.2)
        )
        self.play(FadeIn(vertices_down), current_dist_mob.animate.shift(RIGHT * 0.25))

    def vector_to_mob(self, vector: Iterable):
        str_repr = np.array([f"{a:.2f}" for a in vector]).reshape(-1, 1)
        return Matrix(
            str_repr,
            left_bracket="[",
            right_bracket="]",
            element_to_mobject=Text,
            element_to_mobject_config={"font": REDUCIBLE_MONO},
            h_buff=2.3,
            v_buff=1.3,
        )


class SystemOfEquationsMethod(BruteForceMethod):
    def construct(self):
        frame = self.camera.frame
        markov_ch = MarkovChain(
            4,
            edges=[
                (2, 0),
                (2, 3),
                (0, 3),
                (3, 1),
                (2, 1),
                (1, 2),
            ],
            dist=[0.2, 0.5, 0.2, 0.1],
        )

        markov_ch_mob = MarkovChainGraph(
            markov_ch,
            curved_edge_config={"radius": 2, "tip_length": 0.1},
            straight_edge_config={"max_tip_length_to_length_ratio": 0.08},
            layout="circular",
        )

        markov_ch_sim = MarkovChainSimulator(markov_ch, markov_ch_mob, num_users=50)

        equations_mob = (
            self.get_balance_equations(markov_chain=markov_ch)
            .scale(1)
            .next_to(markov_ch_mob, RIGHT, buff=2.5)
        )
        last_equation = equations_mob[0][38:]

        pi_dists = []
        for s in markov_ch.get_states():
            state = markov_ch_mob.vertices[s]
            label_direction = normalize(state.get_center() - markov_ch_mob.get_center())
            pi_dists.append(
                MathTex(f"\pi({s})")
                .scale(0.8)
                .next_to(state, label_direction, buff=0.1)
            )

        pi_dists_vg = VGroup(*pi_dists)

        self.play(Write(markov_ch_mob))
        self.play(Write(pi_dists_vg))
        self.play(frame.animate.shift(RIGHT * 3.3 + UP * 0.8).scale(1.2))

        title = (
            Text("System of Equations Method", font=REDUCIBLE_FONT, weight=BOLD)
            .scale(1)
            .move_to(frame.get_top())
            .shift(DOWN * 0.9)
        )
        self.play(Write(title))

        add_to_one = (
            MathTex("1 = " + "+".join([f"\pi({s})" for s in markov_ch.get_states()]))
            .scale(0.9)
            .next_to(equations_mob, DOWN, aligned_edge=LEFT)
        )

        stationary_def = MathTex(r"\pi = \pi P ").scale(2.5).move_to(equations_mob)

        self.play(FadeIn(stationary_def, shift=UP * 0.3))

        self.wait()

        self.play(
            FadeIn(equations_mob),
            stationary_def.animate.next_to(equations_mob, UP, buff=0.5).scale(0.6),
        )
        self.wait()

        infinite_solutions = (
            Text("Infinite solutions!", font=REDUCIBLE_FONT, weight=BOLD)
            .scale(0.3)
            .move_to(equations_mob, UP + LEFT)
            .rotate(15 * DEGREES)
            .shift(LEFT * 1.6 + UP * 0.3)
        )

        self.play(FadeIn(infinite_solutions, shift=UP * 0.3))

        for i in range(2):
            self.play(
                infinite_solutions.animate.set_opacity(0),
                run_time=3 / config.frame_rate,
            )
            # self.wait(1 / config.frame_rate)
            self.play(
                infinite_solutions.animate.set_opacity(1),
                run_time=3 / config.frame_rate,
            )
            self.wait(3 / config.frame_rate)

        self.wait()
        self.play(
            FadeIn(add_to_one, shift=UP * 0.3),
            FadeOut(infinite_solutions, shift=UP * 0.3),
        )
        self.wait()

        self.play(
            FadeOut(last_equation, shift=UP * 0.3),
            add_to_one.animate.move_to(last_equation, aligned_edge=LEFT),
        )

        stationary_distribution = self.solve_system(markov_ch)

        tex_strings = []
        for i, s in enumerate(stationary_distribution):
            tex_str = f"\pi({i}) &= {s:.3f}"
            tex_strings.append(tex_str)

        stationary_dist_mob = MathTex("\\\\".join(tex_strings)).move_to(equations_mob)

        self.play(
            FadeOut(equations_mob[0][:38], shift=UP * 0.3),
            FadeOut(add_to_one, shift=UP * 0.3),
            FadeIn(stationary_dist_mob, shift=UP * 0.3),
        )

        line = (
            Line()
            .set_stroke(width=2)
            .stretch_to_fit_width(stationary_dist_mob.width * 1.3)
            .next_to(stationary_dist_mob, DOWN, buff=-0.1)
        )
        total = (
            Text("Total = 1.000", font=REDUCIBLE_FONT, weight=BOLD)
            .scale_to_fit_width(stationary_dist_mob.width)
            .next_to(line, DOWN, buff=0.3)
        )
        self.wait()
        self.play(stationary_dist_mob.animate.shift(UP * 0.4))

        self.play(Write(line), Write(total))

    def solve_system(self, markov_chain: MarkovChain):
        P = markov_chain.get_transition_matrix()

        # P.T gives us the balance equations
        dependent_system = P.T

        # in this step, we are essentially moving every term
        # to the left, so we end up with 0s on the other side
        # of the equation
        for i, eq in enumerate(dependent_system):
            eq[i] -= 1

        # this removes the last equation and substitutes it
        # for our probability constraint
        dependent_system[-1] = [1.0 for s in range(dependent_system.shape[1])]

        # now we create the other side of the equations, which
        # will be a vector of size len(states) with all zeros but
        # a single 1 for the last element
        right_side = [0.0 for s in range(dependent_system.shape[1])]
        right_side[-1] = 1

        # we finally solve the system!
        return np.linalg.solve(dependent_system, right_side)

    def get_balance_equations(self, markov_chain: MarkovChain):
        trans_matrix_T = markov_chain.get_transition_matrix().T
        state_names = markov_chain.get_states()

        balance_equations = []
        for equation in trans_matrix_T:
            balance_equations.append(
                [
                    (
                        Fraction(term).limit_denominator().numerator,
                        Fraction(term).limit_denominator().denominator,
                    )
                    for term in equation
                ]
            )

        tex_strings = []
        for state, fractions in zip(state_names, balance_equations):
            pi_sub_state = f"\pi({state})"

            terms = []
            for i, term in enumerate(fractions):
                state_term = f"\pi({state_names[i]})"
                if term[0] == 1 and term[1] == 1:
                    terms.append(state_term)
                else:
                    if term[0] != 0:
                        fraction = r"\frac{" + str(term[0]) + "}{" + str(term[1]) + "}"
                        terms.append(fraction + state_term)

            terms = "+".join(terms)

            full_equation_tex = pi_sub_state + "&=" + terms
            tex_strings.append(full_equation_tex)

        tex_strings = "\\\\".join(tex_strings)
        return MathTex(tex_strings)


class EigenvalueMethod(MovingCameraScene):
    def construct(self):
        pi = MathTex(r"\pi").scale(5).shift(DOWN * 0.5)
        pi_times_P = (
            MathTex(r"\pi P", substrings_to_isolate="P")
            .scale(5)
            .move_to(pi, aligned_edge=DOWN)
        )
        stationary_dist = (
            MathTex(r"\pi = \pi P", substrings_to_isolate="P")
            .scale(5)
            .move_to(pi, aligned_edge=DOWN)
        )

        self.play(FadeIn(pi, scale=0.8))
        self.wait()
        self.play(TransformMatchingShapes(pi, pi_times_P))
        self.wait()
        self.play(TransformMatchingShapes(pi_times_P, stationary_dist))

        self.play(FadeOut(stationary_dist))

        ########################### Going back to fundamentals...

        purple_plane = NumberPlane(
            x_range=[-10, 10],
            y_range=[-10, 10],
            background_line_style={
                "stroke_color": REDUCIBLE_VIOLET,
                "stroke_width": 3,
                # "stroke_opacity": 0.5,
            },
            faded_line_style={
                "stroke_color": REDUCIBLE_PURPLE,
                "stroke_opacity": 0.5,
            },
            # faded_line_ratio=4,
            axis_config={"stroke_color": REDUCIBLE_PURPLE, "stroke_width": 0},
        )

        static_plane = NumberPlane(
            background_line_style={
                "stroke_color": REDUCIBLE_PURPLE,
            },
            faded_line_style={
                "stroke_color": REDUCIBLE_PURPLE,
            },
            axis_config={"stroke_color": REDUCIBLE_PURPLE, "stroke_opacity": 0},
        ).set_opacity(0.6)

        trans_matrix = [
            [2, 1],
            [1, 2],
        ]
        e_vals, e_vecs = np.linalg.eig(trans_matrix)
        print(e_vecs)
        print(e_vals)
        scaled_vector = Vector([1, 1], max_tip_length_to_length_ratio=0.17).set_color(
            REDUCIBLE_YELLOW
        )
        distorted_vector = Vector([1, 0]).set_color(REDUCIBLE_YELLOW)

        self.play(FadeIn(static_plane), FadeIn(purple_plane))
        self.wait()
        self.play(Write(distorted_vector))

        self.play(
            ApplyMatrix(trans_matrix, purple_plane),
            self.apply_matrix_to_vector(trans_matrix, distorted_vector),
            run_time=1.5,
        )
        self.wait()

        trans_matrix_mob = (
            self.matrix_to_mob(trans_matrix, has_background_color=False)
            .set_stroke(width=10, background=True)
            .scale(0.7)
            .to_corner(UL, buff=1)
        )
        self.play(FadeIn(trans_matrix_mob))

        self.wait()
        self.play(
            FadeOut(distorted_vector),
            ApplyMatrix(np.linalg.inv(trans_matrix), purple_plane),
            run_time=1,
        )
        self.wait()
        self.play(FadeIn(scaled_vector))
        self.play(
            ApplyMatrix(trans_matrix, purple_plane),
            self.apply_matrix_to_vector(trans_matrix, scaled_vector),
            run_time=1,
        )
        self.wait()
        eig_vector = Vector(direction=e_vecs[0]).set_color(REDUCIBLE_YELLOW)
        self.play(
            FadeOut(scaled_vector),
            ApplyMatrix(np.linalg.inv(trans_matrix), purple_plane),
            run_time=1,
        )
        self.wait()

        self.play(FadeIn(eig_vector))
        self.play(
            ApplyMatrix(trans_matrix, purple_plane),
            self.apply_matrix_to_vector(trans_matrix, eig_vector),
            run_time=1,
        )

        self.play(FadeOut(trans_matrix_mob))

        m = MathTex("M ")

        lambdas = VGroup(
            *[MathTex(f"\lambda_{n}") for n in range(4)],
            MathTex(r"\vdots"),
            MathTex(r"\lambda_n"),
        ).arrange(DOWN, buff=0.2)

        p_brace = Brace(lambdas, LEFT)

        eig_vectors = (
            VGroup(
                *[MathTex(r"\vec{v}_{" + str(n) + "}") for n in range(4)],
                MathTex(r"\vdots"),
                MathTex(r"\vec{v}_n"),
            )
            .arrange(DOWN, buff=0.2)
            .next_to(lambdas, RIGHT, buff=0.5)
        )

        m_with_eigs = (
            VGroup(m, p_brace, lambdas, eig_vectors)
            .set_stroke(width=8, background=True)
            .arrange(RIGHT, buff=0.5)
            .to_corner(UL, buff=0.7)
        )
        surr_rect_math = (
            SurroundingRectangle(m_with_eigs)
            .set_stroke(width=0)
            .set_color(BLACK)
            .set_opacity(0.5)
        )
        self.play(
            FadeIn(surr_rect_math),
            FadeIn(m_with_eigs),
        )

        self.play(FadeIn(scaled_vector))
        self.wait()

        eig_val_1 = (
            MathTex(r"\lambda_{0} = 1")
            .set_stroke(width=8, background=True)
            .next_to(eig_vector.get_end(), RIGHT, buff=0.3)
        )
        eig_val_3 = (
            MathTex(r"\lambda_{1} = 3")
            .set_stroke(width=8, background=True)
            .next_to(scaled_vector.get_end(), RIGHT, buff=0.3)
        )
        self.play(FadeIn(eig_val_1), FadeIn(eig_val_3))

        self.wait()

        ############# So, with that in mind,
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
        ).shift(RIGHT * 4 + DOWN * 0.7)

        p = MathTex("P")

        # the stationary dists are eigvecs with eigval 1 from the P.T
        eig_vals_P, eig_vecs_P = np.linalg.eig(markov_ch.get_transition_matrix().T)
        eig_vecs_P = eig_vecs_P.T.astype(float)
        eig_vals_P = eig_vals_P.astype(float)
        print()
        print()
        print()
        print(eig_vecs_P)
        print()
        print(eig_vals_P)
        print()
        # print(np.sum(eig_vecs_P, axis=1))

        lambdas_with_value = VGroup(
            *[
                MathTex(f"\lambda_{n} &= {v:.1f}")
                for n, v in zip(range(len(markov_ch.get_states())), eig_vals_P)
            ],
        ).arrange(DOWN, buff=0.2, aligned_edge=LEFT)

        pi_vectors_example = (
            VGroup(
                *[
                    MathTex(r"\vec{\pi}_{" + str(n) + "}")
                    for n in range(len(markov_ch.get_states()))
                ],
            )
            .arrange(DOWN, buff=0.2)
            .next_to(lambdas, RIGHT, buff=0.5)
        )

        self.play(*[FadeOut(mob) for mob in self.mobjects])
        self.wait()

        p_brace = Brace(lambdas_with_value, LEFT)

        p_with_eigs = (
            VGroup(p.scale(1.4), p_brace, lambdas_with_value, pi_vectors_example)
            .set_stroke(width=8, background=True)
            .arrange(RIGHT, buff=0.5)
            .to_corner(UL, buff=0.7)
        )

        labels = markov_ch_mob.get_transition_labels()

        self.play(
            FadeIn(p_with_eigs, shift=UP * 0.3),
            Write(markov_ch_mob),
        )
        self.play(FadeIn(labels))
        self.wait()

        eig_index = np.ravel(np.argwhere(eig_vals_P.round(1) == 1.0))[0]

        underline_eig_1 = Underline(lambdas_with_value[eig_index]).set_color(
            REDUCIBLE_YELLOW
        )
        self.play(Succession(Create(underline_eig_1), FadeOut(underline_eig_1)))
        self.wait()

        stationary_pi = MathTex(r"\vec{\pi}_" + str(eig_index) + " = ")
        stationary_dist = self.vector_to_mob(eig_vecs_P[eig_index]).scale(0.3)

        vertices_down = VGroup(
            *[s.copy().scale(0.5) for s in markov_ch_mob.vertices.values()]
        ).arrange(DOWN, buff=0.13)

        stationary_distribution = (
            VGroup(stationary_pi, vertices_down, stationary_dist)
            .arrange(RIGHT, buff=0.15)
            .next_to(p_with_eigs, DOWN, buff=2)
        ).scale(2)

        stationary_dist_normalized = (
            self.vector_to_mob(
                [e / sum(eig_vecs_P[eig_index]) for e in eig_vecs_P[eig_index]]
            )
            .scale_to_fit_height(stationary_dist.height)
            .move_to(stationary_dist)
        )

        self.play(FadeIn(stationary_distribution, shift=RIGHT * 0.4))
        self.wait()
        self.play(Transform(stationary_dist, stationary_dist_normalized))

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


class PerformanceEvaluationFull(IntroWebGraph, SystemOfEquationsMethod, Periodicity):
    def construct(self):

        self.intro_pip()
        self.wait()
        self.play(*[FadeOut(mob) for mob in self.mobjects])

        self.massive_system_of_equations()
        self.wait()
        self.play(*[FadeOut(mob) for mob in self.mobjects])

        self.massive_eigen_system()
        self.wait()
        self.play(*[FadeOut(mob) for mob in self.mobjects])

        self.brute_force_benchmark()
        self.wait()
        self.play(*[FadeOut(mob) for mob in self.mobjects])

    def intro_pip(self):
        title = (
            Text("Which method will perform best?", font=REDUCIBLE_FONT, weight=BOLD)
            .scale(0.8)
            .to_edge(UP)
        )

        # design icons
        # maybe it's better that nipun adds them in post rather than
        # putting them here
        self.play(FadeIn(title, shift=UP * 0.3))
        self.wait()

    def massive_system_of_equations(self):
        frame = self.camera.frame
        markov_ch = self.create_big_markov_chain(50)

        equations = (
            self.get_balance_equations(markov_ch)
            .scale(0.4)
            .next_to(frame.get_corner(UL), aligned_edge=UL, buff=1)
            .shift(DOWN)
        )
        self.play(LaggedStartMap(FadeIn, equations[0]), run_time=5)
        self.wait()
        self.play(
            equations.animate.next_to(frame.get_corner(UL), aligned_edge=DL, buff=1),
            run_time=10,
        )

    def massive_eigen_system(self):
        frame = self.camera.frame
        markov_ch = self.create_big_markov_chain(50)

        big_eig_system = (
            self.create_big_eig_system(markov_ch)
            .scale(3)
            .next_to(frame.get_corner(UL), aligned_edge=UL, buff=1, coor_mask=[0, 1, 0])
            .shift(DOWN)
        )

        self.play(LaggedStartMap(FadeIn, big_eig_system, lag_ratio=0.4))
        self.wait()
        self.play(
            big_eig_system.animate.next_to(
                frame.get_top(), aligned_edge=DOWN, buff=1, coor_mask=[0, 1, 0]
            ),
            run_time=10,
        )

    def brute_force_benchmark(self):

        markov_ch, markov_ch_mob = self.get_web_graph()
        # markov_ch = MarkovChain(
        #     4,
        #     edges=[
        #         (2, 0),
        #         (2, 3),
        #         (0, 3),
        #         (3, 1),
        #         (2, 1),
        #         (1, 2),
        #     ],
        #     dist=[0.2, 0.5, 0.2, 0.1],
        # )



        # markov_ch_mob = MarkovChainGraph(
        #     markov_ch,
        #     curved_edge_config={"radius": 2, "tip_length": 0.1},
        #     straight_edge_config={"max_tip_length_to_length_ratio": 0.08},
        #     layout="circular",
        # )

        markov_ch_sim = MarkovChainSimulator(markov_ch, markov_ch_mob, num_users=200)
        users = markov_ch_sim.get_users()

        self.play(
            FadeIn(markov_ch_mob),
            LaggedStart(*[FadeIn(u) for u in users]),
            run_time=0.5,
        )

        last_dist = markov_ch_sim.get_user_dist().values()

        # first iteration
        transition_map = markov_ch_sim.get_lagged_smooth_transition_animations()

        current_dist = markov_ch_sim.get_user_dist().values()

        last_distance = dist(current_dist, last_dist)
        tolerance = 0.001

        print(f"{last_distance = }")

        num_steps = 100
        axes = Axes(
            x_range=[0, num_steps],
            y_range=[0, last_distance],
            y_axis_config={
                "numbers_to_exclude": np.arange(0.1, 1.05, 0.1),
                "stroke_width": 1,
            },
            x_axis_config={
                "numbers_to_exclude": range(num_steps + 1),
                "stroke_width": 1,
            },
            tips=False,
            x_length=3,
            y_length=2,
            axis_config={"include_numbers": False, "include_ticks": False},
        ).set_stroke(width=8, background=True)

        bg_axes = (
            SurroundingRectangle(axes)
            .set_color(BLACK)
            .set_stroke(width=0)
            .set_opacity(0.8)
        )

        distance_label = (
            Text("Distance", font=REDUCIBLE_MONO)
            .scale(0.2)
            .next_to(axes.y_axis, UP, aligned_edge=RIGHT, buff=0.1)
            .set_stroke(width=4, background=True)
        )
        iterations_label = (
            Text("Iterations", font=REDUCIBLE_MONO)
            .scale(0.2)
            .next_to(axes.x_axis, DOWN, buff=0.1)
            .set_stroke(width=4, background=True)
        )
        axes_vg = VGroup(bg_axes, axes, distance_label, iterations_label).to_corner(
            UR, buff=0.5
        )

        tolerance_line = (
            axes.get_horizontal_line(axes.c2p(num_steps, tolerance), line_func=Line)
            .set_color(REDUCIBLE_PURPLE)
            .set_stroke(width=2)
        )

        self.play(
            *[LaggedStart(*transition_map[i]) for i in markov_ch.get_states()],
            FadeIn(axes_vg),
            Write(tolerance_line),
        )

        line_chunks = []
        for i in range(1, num_steps):
            print('Jesus iteration:', i)
            transition_animations = markov_ch_sim.get_instant_transition_animations()

            last_dist = current_dist
            current_dist = markov_ch_sim.get_user_dist().values()

            current_distance = dist(current_dist, last_dist)
            new_line_chunk = self.next_iteration_line(
                axes, i, current_distance, last_distance
            )
            line_chunks.append(new_line_chunk)

            run_time = 0.8 if i < 6 else 1 / i

            if i < 6:
                self.play(
                    *transition_animations,
                    Write(new_line_chunk),
                    run_time=run_time,
                )

            else:
                self.play(
                    *transition_animations,
                    Write(new_line_chunk),
                    run_time=run_time,
                )

            last_distance = current_distance

        self.wait()

    # utils
    def next_iteration_line(
        self, axes: Axes, iteration: int, curr_distance: float, last_distance: float
    ):

        last_distance_p2c = axes.c2p(iteration - 1, last_distance)
        curr_distance_p2c = axes.c2p(iteration, curr_distance)

        return (
            Line(last_distance_p2c, curr_distance_p2c)
            .set_color(REDUCIBLE_YELLOW)
            .set_stroke(width=2)
        )

    def create_big_markov_chain(self, size: int, density: int = 10) -> MarkovChain:
        tuples = []
        for i in range(size):
            random_edges = np.random.choice(
                list(range(size)),
                size=density + np.random.randint(-5, 5),
            )
            for r in random_edges:
                tuples.append((i, r))

        tuples = list(filter(lambda x: x[0] != x[1], tuples))

        return MarkovChain(size, tuples)

    def create_big_eig_system(self, markov_chain: MarkovChain):
        n = len(markov_chain.get_states())

        lambdas_and_eigs = VGroup(
            *[
                MathTex(r"\lambda_{" + str(l) + r"} \ \vec{v}_{" + str(l) + "}")
                for l in range(n)
            ],
        ).arrange(DOWN, buff=0.6)

        return VGroup(lambdas_and_eigs).arrange(RIGHT, buff=0.8, aligned_edge=UP)

class PerformanceText(Scene):
    def construct(self):
        scale = 0.7

        screen_rect_1 = ScreenRectangle(height=2)
        screen_rect_2 = ScreenRectangle(height=2)
        screen_rect_3 = ScreenRectangle(height=2)

        screen_rect_1.move_to(UP * 2.5 + LEFT * 3.5)
        screen_rect_2.move_to(DOWN * 0 + LEFT * 3.5)
        screen_rect_3.move_to(DOWN * 2.5 + LEFT * 3.5)

        methods = VGroup(
            Text("Brute Force", font=REDUCIBLE_FONT).scale(scale),
            Text("System of Equations", font=REDUCIBLE_FONT).scale(scale),
            Text("Eigenvalues/Eigenvectors", font=REDUCIBLE_FONT).scale(scale)
        )

        methods[0].move_to(UP * 2.5 + RIGHT * 2.5)
        methods[1].move_to(UP * 0 + RIGHT * 2.5)
        methods[2].move_to(DOWN * 2.5 + RIGHT * 2.5)

        self.play(
            FadeIn(methods),
            # FadeIn(screen_rect_1),
            # FadeIn(screen_rect_2),
            # FadeIn(screen_rect_3)
        )
        self.wait()

        dumb_efficient = Tex("``Dumb'' but efficient").scale(0.7)
        dumb_efficient.next_to(methods[0], DOWN * 1.5)
        direct_but_slow = Tex("Direct but slow").scale(0.7)
        direct_but_slow.next_to(methods[1], DOWN * 1.5)
        clever_but_slow = Tex("Clever but slow").scale(0.7)
        clever_but_slow.next_to(methods[2], DOWN * 1.5)

        self.play(
            FadeIn(dumb_efficient),
            FadeIn(direct_but_slow),
            FadeIn(clever_but_slow)
        )
        self.wait()

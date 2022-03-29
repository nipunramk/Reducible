###
"""
Some considerations before rendering:
    — Everything is setup un the construct method. If you don't want to render the 
    first animations, remember to disable them in there. 
    — Everything else is adjusted accordingly so you don't have to worry about running it.
"""

from manim import *
from math import floor
from manim.mobject.geometry import ArrowTriangleFilledTip
from numpy import ndarray, subtract
from numpy.lib.arraypad import pad
from functions import *
from classes import *
from reducible_colors import *
from itertools import product


np.random.seed(1)

config["assets_dir"] = "assets"


class FilteringInsertMSAD(MovingCameraScene):
    def construct(self):
        self.camera.frame.save_state()
        initial_state = self.camera.frame.copy()

        # self.intro_filtering()

        # self.wait()
        # self.clear()

        # self.present_problem()

        # self.wait()
        # self.clear()

        # self.five_filters_explanation()

        # self.wait()
        # self.clear()
        # self.play(Transform(self.camera.frame, initial_state))

        # self.minor_considerations()

        # self.wait()
        # self.clear()
        # self.play(Transform(self.camera.frame, initial_state))

        # self.what_filter_to_use()

        # self.wait()
        # self.clear()
        # self.play(Transform(self.camera.frame, initial_state))

        # self.low_bit_depth_images()

        # self.wait()
        # self.clear()
        # self.play(Transform(self.camera.frame, initial_state))

        # self.palette_images()

        # self.wait()
        # self.clear()

        # self.repeating_filters_performance()

        # self.wait()
        # self.clear()

        # self.combination_explosion()

        # self.wait()
        # self.clear()

        self.msad_intro()

        self.wait()
        self.clear()

        # self.minimum_sum_of_absolute_differences()

        # self.wait()
        # self.clear()

        # self.png_decoding()

        # self.wait()

    def intro_filtering(self):
        title = Text("Lossless Compression", font="CMU Serif", weight=BOLD).to_edge(UP)

        eg_data = Text(
            "00000111101010101 0000 11111 00000111101010101",
            font="SF Mono",
        ).scale(1.5)

        eg_data_no_repeats = Text(
            "0101010001101001001001001001010", font="SF Mono"
        ).scale(1.5)
        brace_no_reps = Brace(eg_data_no_repeats, direction=DOWN)
        question_comp = (
            Text("?x?", font="SF Mono").scale(1.7).next_to(brace_no_reps, DOWN)
        )

        brace_zeros = Brace(eg_data[17:21], direction=DOWN)
        brace_ones = Brace(eg_data[21:26], direction=DOWN)

        comp_zeros = Text("4x0", font="SF Mono").scale(0.5).next_to(brace_zeros, DOWN)
        comp_ones = Text("5x1", font="SF Mono").scale(0.5).next_to(brace_ones, DOWN)

        self.play(Write(title))

        self.wait()

        self.play(AddTextLetterByLetter(eg_data, run_time=1))

        self.wait()

        self.play(
            Write(brace_zeros),
            Write(comp_zeros),
            Write(brace_ones),
            Write(comp_ones),
        )

        self.wait()

        self.play(
            FadeOut(brace_zeros, shift=UP),
            FadeOut(brace_ones, shift=UP),
            FadeOut(comp_zeros, shift=UP),
            FadeOut(comp_ones, shift=UP),
            FadeOut(eg_data, shift=UP),
        )

        self.wait()

        self.play(AddTextLetterByLetter(eg_data_no_repeats, run_time=1))

        self.wait()

        self.play(Write(brace_no_reps, run_time=0.5), Write(question_comp))

    def present_problem(self):
        repeating_text = (
            Text("0000000000000", font="SF Mono").scale(2).shift(UP * 1.3 + LEFT * 2)
        )
        non_rep_text = (
            Text("1001001000101", font="SF Mono").scale(2).shift(DOWN * 1.3 + RIGHT * 2)
        )

        diff = 4
        gradient_data = np.arange(0, 256, diff, dtype=np.uint8).reshape((8, 8))
        gradient_image = PixelArray(
            gradient_data, include_numbers=True, color_mode="GRAY"
        ).scale(0.6)

        self.play(FadeIn(repeating_text, shift=RIGHT))
        self.wait()
        self.play(FadeIn(non_rep_text, shift=LEFT))

        self.play(FadeOut(repeating_text), FadeOut(non_rep_text))

        self.wait()

        self.play(Write(gradient_image))

        self.wait()

        self.play(gradient_image.animate.shift(LEFT * 3))

        self.wait()

        question_1 = (
            Text(
                " How can we represent this data\n in a more repetitive way?",
                font="CMU Serif",
            )
            .scale(0.7)
            .next_to(gradient_image, RIGHT, buff=0.5)
            .shift(UP * 0.9)
        )
        question_2 = (
            Text("Any patterns to be found?", font="CMU Serif")
            .scale(0.7)
            .next_to(question_1, DOWN, buff=1.3, aligned_edge=LEFT)
        )

        self.play(Write(question_1))

        self.wait()

        self.play(Write(question_2))

        self.wait()

        self.play(FadeOut(question_1), FadeOut(question_2))

        row_to_focus_on = gradient_image[0:8]

        self.camera.frame.save_state()

        self.play(
            self.camera.frame.animate.set_width(row_to_focus_on.width * 1.3)
            .move_to(row_to_focus_on)
            .shift(UP),
            run_time=3,
        )

        self.wait()

        diff_t = Text(f"+{diff}", font="SF Mono").scale(0.3).set_color(REDUCIBLE_YELLOW)

        arrows = VGroup()
        diffs = VGroup()
        for i in range(1, 8):
            arrow = (
                CurvedArrow(
                    gradient_image[i - 1].get_center() + UP * 0.3,
                    gradient_image[i].get_center() + UP * 0.3,
                    radius=-0.5,
                )
                .set_color(REDUCIBLE_YELLOW)
                .set_stroke(width=1)
            )
            arrow.pop_tips()
            arrow.add_tip(
                tip_shape=ArrowTriangleFilledTip, tip_length=0.05, at_start=False
            )

            arrows.add(arrow)

            diff_copy = diff_t.copy().next_to(arrow, UP, buff=0.1)
            diffs.add(diff_copy)

        self.play(
            *[Write(arrow) for arrow in arrows]
            + [FadeIn(diff, shift=UP * 0.2) for diff in diffs]
        )

        self.wait()

        diff_buff = (
            PixelArray(
                np.array(
                    [
                        [0, 4, 4, 4, 4, 4, 4, 4],
                    ]
                ),
                color_mode="GRAY",
                include_numbers=True,
            )
            .scale_to_fit_width(gradient_image.width)
            .next_to(gradient_image, UP * 0.45, buff=2)
        )

        [
            p.set_color(REDUCIBLE_PURPLE)
            .set_opacity(0.5)
            .set_stroke(REDUCIBLE_PURPLE, width=3, opacity=1)
            for p in diff_buff
        ]

        diff_buff.numbers.set_color(WHITE).set_opacity(1).set_stroke(width=0)

        self.play(LaggedStartMap(Write, diff_buff))

        self.wait()

        self.play(
            self.camera.frame.animate.shift(UP * 1.3), FadeOut(arrows), FadeOut(diffs)
        )

        braces_comp_filtered_data = Brace(diff_buff[1:], direction=UP)
        comp_filtered_data_text = (
            Text("7x4", font="SF Mono")
            .scale(0.5)
            .next_to(braces_comp_filtered_data, UP, buff=0.1)
        )

        self.play(Write(braces_comp_filtered_data), Write(comp_filtered_data_text))
        self.wait()

        self.play(
            FadeOut(braces_comp_filtered_data),
            FadeOut(comp_filtered_data_text),
            Restore(self.camera.frame),
            FadeOut(diff_buff),
        )
        self.wait()

        full_diff_buff = (
            PixelArray(
                np.array(
                    [
                        [0, 4, 4, 4, 4, 4, 4, 4],
                        [32, 4, 4, 4, 4, 4, 4, 4],
                        [64, 4, 4, 4, 4, 4, 4, 4],
                        [96, 4, 4, 4, 4, 4, 4, 4],
                        [128, 4, 4, 4, 4, 4, 4, 4],
                        [160, 4, 4, 4, 4, 4, 4, 4],
                        [192, 4, 4, 4, 4, 4, 4, 4],
                        [224, 4, 4, 4, 4, 4, 4, 4],
                    ]
                ),
                color_mode="GRAY",
                include_numbers=True,
            )
            .scale_to_fit_width(gradient_image.width)
            .next_to(gradient_image, RIGHT, buff=1.3)
        )

        [
            p.set_color(REDUCIBLE_PURPLE)
            .set_opacity(0.5)
            .set_stroke(REDUCIBLE_PURPLE, width=3, opacity=1)
            for p in full_diff_buff
        ]

        full_diff_buff.numbers.set_color(WHITE).set_opacity(1).set_stroke(width=0)

        self.play(Write(full_diff_buff))

        self.wait()

        self.play(FadeOut(gradient_image), FadeOut(full_diff_buff))

        self.wait()

        # number line
        zero = (
            Text(
                "0",
                font="SF Mono",
            )
            .scale(2)
            .move_to(ORIGIN)
        )

        number_line_left = (
            VGroup(*[Text(str(n), font="SF Mono") for n in range(-3, 0)])
            .arrange(RIGHT, buff=1.5)
            .next_to(zero, LEFT, buff=2)
        )

        number_line_right = (
            VGroup(*[Text(str(n), font="SF Mono") for n in range(1, 4)])
            .arrange(RIGHT, buff=1.5)
            .next_to(zero, RIGHT, buff=2)
        )

        def gaussian(x, mu=0, sig=1, scale=4, height=5):
            return (
                np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0))) * scale
                + height
            )

        axes = Axes([-5, 5, 0.1], x_length=16).shift(DOWN * 5)
        graph = (
            axes.plot(gaussian, [-5.0, 5.0, 0.1])
            .set_color(REDUCIBLE_PURPLE)
            .set_stroke(width=10)
            # .set_fill(REDUCIBLE_PURPLE)
            # .set_opacity(0.5)
        )
        area = axes.get_area(
            graph,
            [-5, 5],
            color=color_gradient(
                (
                    rgba_to_color((0, 0, 0, 0)),
                    REDUCIBLE_PURPLE,
                    rgba_to_color((0, 0, 0, 0)),
                ),
                10,
            ),
            opacity=0.5,
        )

        self.wait()

        self.play(
            Write(graph),
            Write(area),
            Write(number_line_left),
            Write(zero),
            Write(number_line_right),
        )

        self.wait()

    def five_filters_explanation(self):
        # intro for the name of the filters

        filter_title = Text("Filtering", font="CMU Serif", weight=BOLD).to_edge(UP)

        none_t = Text("NONE", font="CMU Serif", weight=BOLD).scale(0.7)
        sub_t = Text("SUB", font="CMU Serif", weight=BOLD).scale(0.7)
        up_t = Text("UP", font="CMU Serif", weight=BOLD).scale(0.7)
        avg_t = Text("AVG", font="CMU Serif", weight=BOLD).scale(0.7)
        paeth_t = Text("PAETH", font="CMU Serif", weight=BOLD).scale(0.7)

        filter_types = VGroup(none_t, sub_t, up_t, avg_t, paeth_t).arrange(
            RIGHT, buff=1
        )

        self.play(Write(filter_title))

        self.wait()

        self.play(LaggedStartMap(FadeIn, filter_types))

        self.wait()

        self.play(LaggedStartMap(FadeOut, filter_types))

        random_data = np.random.randint(127, 140, (8, 8))

        input_img = PixelArray(
            random_data, include_numbers=True, color_mode="GRAY", outline=True
        ).scale(0.4)

        output_img = PixelArray(
            np.zeros(random_data.shape, dtype=np.int64),
            include_numbers=True,
            color_mode="GRAY",
            outline=True,
        ).scale_to_fit_height(input_img.height)

        _ = VGroup(input_img, output_img).arrange(RIGHT, buff=0.5).move_to(ORIGIN)

        input_img_t = (
            Text("Input Image", font="CMU Serif")
            .scale(0.5)
            .next_to(input_img, DOWN, buff=0.2)
        )

        output_img_t = (
            Text("Filtered Image", font="CMU Serif")
            .scale(0.5)
            .next_to(output_img, DOWN, buff=0.2)
        )

        self.play(
            FadeIn(input_img),
            Write(input_img_t),
            FadeIn(output_img),
            Write(output_img_t),
        )

        self.wait()

        filtered_data = random_data.copy()

        # save general frame view
        self.camera.frame.save_state()

        row_to_focus_on = self.select_row_indices(0, random_data.shape)
        row_in_out_img = VGroup(input_img[row_to_focus_on], output_img[row_to_focus_on])

        self.play(
            LaggedStart(
                self.camera.frame.animate.set_width(row_in_out_img.width * 1.3)
                .move_to(row_in_out_img)
                .shift(UP),
                FadeOut(filter_title, run_time=0.5),
            ),
            run_time=3,
        )

        focus_mask_input = Difference(
            Square().scale(9),
            Rectangle(
                width=input_img[row_to_focus_on].width,
                height=input_img[row_to_focus_on].height,
            ).move_to(input_img[row_to_focus_on]),
        )

        focus_mask = (
            Difference(
                focus_mask_input,
                Rectangle(
                    width=output_img[row_to_focus_on].width,
                    height=output_img[row_to_focus_on].height,
                ).move_to(output_img[row_to_focus_on]),
            )
            .set_opacity(0.5)
            .set_color(BLACK)
        )

        black_mask = (
            Square()
            .scale(9)
            .set_color(BLACK)
            .set_opacity(1)
            .next_to(row_in_out_img, UP, buff=0)
            .shift(UP * output_img[1].height)
        )

        row_height = UP * input_img[1].height

        # show how NONE works on row 0 ####################################################
        self.play(FadeIn(focus_mask), FadeIn(black_mask))
        self.wait()

        frame_center = self.camera.frame.get_center()

        title_filter = none_t.copy().scale(0.8)
        self.play(
            FadeIn(
                title_filter.move_to(self.camera.frame.get_corner(UL)).shift(
                    RIGHT * 1 + DOWN * 0.6
                )
            )
        )

        x_sq_in = (
            Square()
            .set_fill(REDUCIBLE_GREEN)
            .set_opacity(0.3)
            .set_stroke(REDUCIBLE_GREEN, opacity=1, width=2)
            .scale(0.2)
            .move_to(frame_center)
        )

        x_sq_out = (
            Square()
            .set_fill(REDUCIBLE_PURPLE)
            .set_opacity(0.3)
            .set_stroke(REDUCIBLE_PURPLE, opacity=1, width=2)
            .scale(0.2)
            .next_to(x_sq_in, RIGHT, buff=1)
        )

        x_t_in = Text("x", font="SF Mono", weight=BOLD).scale(0.4).move_to(x_sq_in)
        x_t_out = (
            MarkupText("x<sub>out</sub>", font="SF Mono", weight=BOLD)
            .scale(0.25)
            .move_to(x_sq_out)
        )

        input_pixel = VGroup(x_sq_in, x_t_in)
        output_pixel = VGroup(x_sq_out, x_t_out)

        right_arrow = Arrow(
            input_pixel.get_right(),
            output_pixel.get_left(),
            max_tip_length_to_length_ratio=0.1,
            max_stroke_width_to_length_ratio=2,
        ).set_color(WHITE)

        none_definition = (
            VGroup(input_pixel, right_arrow, output_pixel)
            .arrange(RIGHT, buff=0.2)
            .move_to(frame_center)
            .shift(UP * 0.15)
        )

        self.wait()

        self.play(Write(none_definition))

        filtered_data = np.zeros(random_data.shape, dtype=np.int16)

        filtered_data[0, :] = random_data[0, :]

        filtered_mob = (
            PixelArray(filtered_data, include_numbers=True, color_mode="GRAY")
            .scale_to_fit_height(input_img.height)
            .move_to(output_img)
        )

        self.wait()

        self.play(
            Transform(output_img, filtered_mob),
        )

        self.wait()

        self.play(
            input_img.animate.shift(row_height),
            output_img.animate.shift(row_height),
        )
        self.wait()

        # now on row one, show the SUB filter ####################################################
        title_filter_sub = (
            sub_t.copy().scale(0.8).move_to(title_filter, aligned_edge=LEFT)
        )

        self.play(
            FadeTransform(title_filter, title_filter_sub, stretch=False),
            FadeOut(none_definition),
        )

        l_t_sub = Text("L", font="SF Mono", weight=BOLD).scale(0.4).set_opacity(0.6)

        left_sub = VGroup(
            x_sq_in.copy().set_color(REDUCIBLE_GREEN_DARKER), l_t_sub
        ).arrange(ORIGIN)

        input_sub = VGroup(left_sub, input_pixel.copy()).arrange(RIGHT, buff=0)

        eq_sign = Text("=", font="SF Mono").scale(0.4)
        minus_sign = Text("-", font="SF Mono").scale(0.4)

        sub_operation = VGroup(
            output_pixel.copy(),
            eq_sign.copy(),
            input_pixel.copy(),
            minus_sign.copy(),
            VGroup(x_sq_in.copy(), l_t_sub.copy().set_opacity(1)).arrange(ORIGIN),
        ).arrange(RIGHT, buff=0.2)

        sub_definition = VGroup(
            input_sub.move_to(frame_center).next_to(
                input_img, UP, buff=0.5, coor_mask=[1, 0, 1]
            ),
            sub_operation.move_to(frame_center).next_to(
                output_img, UP, buff=0.5, coor_mask=[1, 0, 1]
            ),
        )

        self.play(LaggedStartMap(FadeIn, sub_definition))

        sub_row = self.sub_filter_row(random_data, row=1, return_row=True)

        filtered_data[1, :] = sub_row

        filtered_mob = (
            PixelArray(filtered_data, include_numbers=True, color_mode="GRAY")
            .scale_to_fit_height(input_img.height)
            .move_to(output_img)
        )

        self.play(Transform(output_img, filtered_mob))

        self.wait()

        # annotation of how the subtraction works
        saved_state = self.camera.frame.save_state()

        pixels_to_focus_on = VGroup(input_img[11:13], output_img[11:13])
        self.play(
            self.focus_on(pixels_to_focus_on, buff=1.5).shift(UP * 0.5),
        )

        self.wait()

        surr_rect_input_pixels = SurroundingRectangle(
            input_img[10:12], color=REDUCIBLE_YELLOW, buff=0.02
        ).set_fill(REDUCIBLE_YELLOW, opacity=0.1)

        surr_rect_output = SurroundingRectangle(
            output_img[11], color=REDUCIBLE_YELLOW, buff=0.02
        ).set_fill(REDUCIBLE_YELLOW, opacity=0.1)

        self.play(Write(surr_rect_input_pixels), Write(surr_rect_output))

        self.wait()
        subtraction_question = (
            Text(
                "134 - 139 = 251?",
                font="SF Mono",
            )
            .scale(0.4)
            .next_to(VGroup(input_img, output_img), UP, buff=0.9)
        )

        filtering_point = Text(
            "PNG Filtering Operates on Bytes", font="CMU Serif", weight=BOLD
        )
        filtering_point.scale(0.5).next_to(subtraction_question, UP)

        subtraction_1 = (
            Text("134 - 139 = -5", font="SF Mono")
            .scale(0.4)
            .move_to(subtraction_question.get_center())
        )
        subtraction_2 = (
            Text("(134 - 139) % 256 = 251", font="SF Mono")
            .scale(0.4)
            .move_to(subtraction_question.get_center())
        )

        self.play(Write(subtraction_question))

        self.wait()

        self.play(Write(filtering_point))
        self.wait()

        self.play(FadeOut(filtering_point))

        correspondence = (
            Tex(r"8 Bit Pixels $\Leftrightarrow$ Bytes")
            .scale(0.6)
            .move_to(filtering_point.get_center())
        )
        self.play(Write(correspondence))
        self.wait()

        self.play(
            FadeOut(correspondence),
            ReplacementTransform(subtraction_question, subtraction_1),
        )

        self.wait()

        self.play(ReplacementTransform(subtraction_1, subtraction_2))
        self.wait()

        self.play(
            Restore(saved_state),
            FadeOut(subtraction_2),
            FadeOut(surr_rect_input_pixels),
            FadeOut(surr_rect_output),
            FadeOut(sub_definition),
        )

        self.wait()

        self.play(
            input_img.animate.shift(row_height),
            output_img.animate.shift(row_height),
        )

        # row 2, show the UP filter ####################################################
        title_filter_up = (
            up_t.copy().scale(0.8).move_to(title_filter, aligned_edge=LEFT)
        )

        self.play(
            FadeTransform(title_filter_sub, title_filter_up, stretch=False),
        )

        u_t_up = Text("U", font="SF Mono", weight=BOLD).scale(0.4).set_opacity(0.6)

        up_up = VGroup(
            x_sq_in.copy().set_color(REDUCIBLE_GREEN_DARKER), u_t_up
        ).arrange(ORIGIN)

        input_up = VGroup(
            up_up.next_to(input_pixel.copy(), UP, buff=0),
            input_pixel.copy(),
        )

        up_operation = VGroup(
            output_pixel.copy(),
            eq_sign.copy(),
            input_pixel.copy(),
            minus_sign.copy(),
            VGroup(x_sq_in.copy(), u_t_up.copy().set_opacity(1)).arrange(ORIGIN),
        ).arrange(RIGHT, buff=0.2)

        up_definition = VGroup(
            input_up.move_to(frame_center)
            .next_to(input_img, UP, buff=0.5, coor_mask=[1, 0, 1])
            .shift(UP * 0.15),
            up_operation.move_to(frame_center).next_to(
                output_img, UP, buff=0.5, coor_mask=[1, 0, 1]
            ),
        )

        self.play(LaggedStartMap(FadeIn, up_definition))

        up_row = self.up_filter_row(random_data, row=2, return_row=True)

        filtered_data[2, :] = up_row

        filtered_mob = (
            PixelArray(filtered_data, include_numbers=True, color_mode="GRAY")
            .scale_to_fit_height(input_img.height)
            .move_to(output_img)
        )

        self.play(Transform(output_img, filtered_mob))

        self.wait()

        self.play(
            input_img.animate.shift(row_height),
            output_img.animate.shift(row_height),
        )
        self.wait()

        # row 3, show the avg filter ####################################################
        title_filter_avg = (
            avg_t.copy().scale(0.8).move_to(title_filter, aligned_edge=LEFT)
        )

        self.play(
            FadeTransform(title_filter_up, title_filter_avg, stretch=False),
            FadeOut(up_definition),
        )

        u_t_avg = Text("U", font="SF Mono", weight=BOLD).scale(0.4).set_opacity(0.6)

        left_avg = VGroup(
            x_sq_in.copy().set_color(REDUCIBLE_GREEN_DARKER), l_t_sub.copy()
        ).arrange(ORIGIN)

        up_avg = VGroup(
            x_sq_in.copy().set_color(REDUCIBLE_GREEN_DARKER), u_t_up
        ).arrange(ORIGIN)

        main_x_avg = input_pixel.copy()
        input_avg = VGroup(
            left_avg.next_to(main_x_avg, LEFT, buff=0),
            up_avg.next_to(main_x_avg, UP, buff=0),
            main_x_avg,
        )

        avg_operation = VGroup(
            output_pixel.copy(),
            eq_sign.copy(),
            input_pixel.copy(),
            minus_sign.copy(),
            VGroup(
                VGroup(
                    VGroup(x_sq_in.copy(), u_t_avg.copy().set_opacity(1)).arrange(
                        ORIGIN
                    ),
                    Text("+", font="SF Mono").scale(0.4),
                    VGroup(x_sq_in.copy(), l_t_sub.copy().set_opacity(1)).arrange(
                        ORIGIN
                    ),
                )
                .arrange(RIGHT, buff=0.2)
                .scale(0.7),
                Line(ORIGIN, LEFT).set_stroke(width=1),
                Text("2", font="SF Mono").scale(0.4),
            ).arrange(DOWN, buff=0.1),
        ).arrange(RIGHT, buff=0.2)

        avg_definition = VGroup(
            input_avg.move_to(frame_center)
            .next_to(input_img, UP, buff=0.5, coor_mask=[1, 0, 1])
            .shift(UP * 0.15),
            avg_operation.move_to(frame_center)
            .next_to(output_img, UP, buff=0.5, coor_mask=[1, 0, 1])
            .shift(UP * 0.15),
        )

        self.play(LaggedStartMap(FadeIn, avg_definition))

        avg_row = self.avg_filter_row(random_data, row=3, return_row=True)

        filtered_data[3, :] = avg_row

        filtered_mob = (
            PixelArray(filtered_data, include_numbers=True, color_mode="GRAY")
            .scale_to_fit_height(input_img.height)
            .move_to(output_img)
        )

        self.play(Transform(output_img, filtered_mob))

        mod_256 = (
            Text("% 256", font="SF Mono", weight=BOLD)
            .scale(0.4)
            .next_to(avg_definition, RIGHT, buff=0.2)
            .set_opacity(0.4)
        )
        self.play(FadeIn(mod_256))

        self.wait()

        self.play(FadeOut(mod_256))

        self.wait()

        self.play(
            input_img.animate.shift(row_height),
            output_img.animate.shift(row_height),
        )
        self.wait()

        # row 4, show the paeth filter
        title_filter_paeth = (
            paeth_t.copy().scale(0.8).move_to(title_filter, aligned_edge=LEFT)
        )

        self.play(
            FadeTransform(title_filter_avg, title_filter_paeth, stretch=False),
            FadeOut(avg_definition),
        )

        up_paeth = up_avg.copy()

        left_paeth = left_avg.copy()

        ul_paeth = VGroup(
            x_sq_in.copy().set_color(REDUCIBLE_GREEN_DARKER),
            Text("UL", font="SF Mono", weight=BOLD).set_opacity(0.6).scale(0.35),
        ).arrange(ORIGIN)

        main_x_paeth = input_pixel.copy()

        input_paeth = (
            VGroup(ul_paeth, up_paeth, left_paeth, main_x_paeth)
            .arrange_in_grid(rows=2, cols=2, buff=0)
            .move_to(frame_center)
            .next_to(input_img, UP, buff=0.5, coor_mask=[1, 0, 1])
            .shift(UP * 0.15)
        )

        self.play(FadeIn(input_paeth))

        v_t_paeth = Text("v", font="SF Mono", weight=BOLD).scale(0.4)

        v_ops = (
            VGroup(
                v_t_paeth,
                eq_sign.copy(),
                VGroup(x_sq_in.copy(), u_t_up.copy().set_opacity(1)).arrange(ORIGIN),
                Text("+", font="SF Mono").scale(0.4),
                VGroup(x_sq_in.copy(), l_t_sub.copy().set_opacity(1)).arrange(ORIGIN),
                minus_sign.copy(),
                VGroup(
                    x_sq_in.copy(),
                    Text("UL", font="SF Mono", weight=BOLD).set_opacity(1).scale(0.35),
                ).arrange(ORIGIN),
            )
            .arrange(RIGHT, buff=0.1)
            .move_to(frame_center)
            .next_to(output_img, UP, coor_mask=[1, 0, 1], aligned_edge=LEFT)
            .shift(UP * 0.5)
        )
        self.play(FadeIn(v_ops))

        self.wait()

        self.play(v_ops.animate.shift(UP).scale(0.7))

        v_l_paeth = MarkupText("v<sub>L</sub>", font="SF Mono").scale(0.4)
        v_u_paeth = MarkupText("v<sub>U</sub>", font="SF Mono").scale(0.4)
        v_ul_paeth = MarkupText("v<sub>UL</sub>", font="SF Mono").scale(0.4)

        v_l_ops = VGroup(
            v_l_paeth.copy(),
            eq_sign.copy(),
            v_t_paeth.copy(),
            minus_sign.copy(),
            VGroup(x_sq_in.copy(), l_t_sub.copy().set_opacity(1))
            .arrange(ORIGIN)
            .scale(0.7),
        ).arrange(RIGHT, buff=0.2)

        v_u_ops = VGroup(
            v_u_paeth.copy(),
            eq_sign.copy(),
            v_t_paeth.copy(),
            minus_sign.copy(),
            VGroup(x_sq_in.copy(), u_t_up.copy().set_opacity(1))
            .arrange(ORIGIN)
            .scale(0.7),
        ).arrange(RIGHT, buff=0.2)

        v_ul_ops = VGroup(
            v_ul_paeth.copy(),
            eq_sign.copy(),
            v_t_paeth.copy(),
            minus_sign.copy(),
            VGroup(
                x_sq_in.copy(),
                Text("UL", font="SF Mono", weight=BOLD).set_opacity(1).scale(0.35),
            )
            .arrange(ORIGIN)
            .scale(0.7),
        ).arrange(RIGHT, buff=0.2)

        all_ops = (
            VGroup(v_l_ops, v_u_ops, v_ul_ops)
            .arrange(DOWN, buff=0.2, aligned_edge=LEFT)
            .scale(0.7)
            .next_to(v_ops, DOWN, buff=0.2, aligned_edge=LEFT)
        )

        self.play(FadeIn(all_ops))

        brace_min = Brace(all_ops, direction=RIGHT, buff=0.05).stretch_to_fit_width(
            0.15
        )

        double_right_arrow = Tex(r"$\Rightarrow$").scale(0.4)

        min_t = (
            MarkupText(
                "min(v<sub>L</sub>, v<sub>U</sub>, v<sub>UL</sub>)", font="SF Mono"
            )
            .scale(0.2)
            .next_to(brace_min, RIGHT, buff=0.1)
        )

        win_u = (
            VGroup(v_u_paeth.copy(), double_right_arrow.copy(), v_u_ops[4].copy())
            .arrange(RIGHT, buff=0.1)
            .scale(0.8)
            .next_to(brace_min, RIGHT, buff=0.1)
        )

        win_l = (
            VGroup(v_l_paeth.copy(), double_right_arrow.copy(), v_l_ops[4].copy())
            .arrange(RIGHT, buff=0.1)
            .scale(0.8)
            .next_to(brace_min, RIGHT, buff=0.1)
        )

        win_ul = (
            VGroup(v_ul_paeth.copy(), double_right_arrow.copy(), v_ul_ops[4].copy())
            .arrange(RIGHT, buff=0.1)
            .scale(0.8)
            .next_to(brace_min, RIGHT, buff=0.1)
        )

        self.play(Write(brace_min))

        self.wait()

        self.play(Write(min_t))

        self.wait()

        self.play(min_t.animate.shift(UP * 0.25))

        self.play(FadeIn(win_u.next_to(brace_min, RIGHT, buff=0.1)))

        final_result_u = (
            VGroup(
                output_pixel.copy(),
                eq_sign.copy(),
                input_pixel.copy(),
                minus_sign.copy(),
                win_u[2].copy().scale_to_fit_height(output_pixel.height),
            )
            .arrange(RIGHT, buff=0.2)
            .scale(0.7)
            .next_to(all_ops, DOWN, aligned_edge=LEFT, buff=0.2)
        )

        final_result_l = (
            VGroup(
                output_pixel.copy(),
                eq_sign.copy(),
                input_pixel.copy(),
                minus_sign.copy(),
                win_l[2].copy().scale_to_fit_height(output_pixel.height),
            )
            .arrange(RIGHT, buff=0.2)
            .scale(0.7)
            .next_to(all_ops, DOWN, aligned_edge=LEFT, buff=0.2)
        )

        final_result_ul = (
            VGroup(
                output_pixel.copy(),
                eq_sign.copy(),
                input_pixel.copy(),
                minus_sign.copy(),
                win_ul[2].copy().scale_to_fit_height(output_pixel.height),
            )
            .arrange(RIGHT, buff=0.2)
            .scale(0.7)
            .next_to(all_ops, DOWN, aligned_edge=LEFT, buff=0.2)
        )

        self.play(FadeIn(final_result_u))

        paeth_row = self.paeth_filter_row(random_data, row=4, return_row=True)

        filtered_data[4, :] = paeth_row

        filtered_mob = (
            PixelArray(filtered_data, include_numbers=True, color_mode="GRAY")
            .scale_to_fit_height(input_img.height)
            .move_to(output_img)
        )

        self.wait()

        self.play(Transform(win_u, win_l), Transform(final_result_u, final_result_l))

        self.play(Transform(output_img, filtered_mob))

        self.wait()

        self.play(
            input_img.animate.shift(row_height),
            output_img.animate.shift(row_height),
        )

        self.wait()

        paeth_row = self.paeth_filter_row(random_data, row=5, return_row=True)

        filtered_data[5, :] = paeth_row

        filtered_mob = (
            PixelArray(filtered_data, include_numbers=True, color_mode="GRAY")
            .scale_to_fit_height(input_img.height)
            .move_to(output_img)
        )

        self.play(Transform(win_u, win_ul), Transform(final_result_u, final_result_ul))

        self.play(Transform(output_img, filtered_mob))

        self.wait()

        self.play(
            input_img.animate.shift(row_height),
            output_img.animate.shift(row_height),
        )
        self.wait()

    def minor_considerations(self):
        """
        Some minor considerations: these operations are done on each channel individually,
        so we are not mixing up red and green values. Each channel is treated separately.
        """
        image = ImageMobject("r_3_palette")

        pixel_array = image.get_pixel_array().astype(int)

        pixel_array_r = pixel_array[:, :, 0]
        pixel_array_g = pixel_array[:, :, 1]
        pixel_array_b = pixel_array[:, :, 2]

        pixel_array_mob_r = PixelArray(
            self.get_channel_image(pixel_array_r, mode="R"), color_mode="RGB"
        ).scale(0.4)

        pixel_array_mob_g = PixelArray(
            self.get_channel_image(pixel_array_g, mode="G"), color_mode="RGB"
        ).scale(0.4)

        pixel_array_mob_b = PixelArray(
            self.get_channel_image(pixel_array_b, mode="B"), color_mode="RGB"
        ).scale(0.4)

        all_channels = (
            VGroup(pixel_array_mob_r, pixel_array_mob_g, pixel_array_mob_b)
            .arrange(RIGHT, buff=0.5)
            .scale(0.7)
        )

        self.play(FadeIn(all_channels))
        self.wait()

        self.play(all_channels.animate.shift(UP * 1.2))

        self.wait()

        (
            iterations_r,
            iterations_g,
            iterations_b,
            methods_history,
        ) = self.filter_image_randomly(pixel_array)

        first_iteration = (
            VGroup(iterations_r[0], iterations_g[0], iterations_b[0])
            .arrange(RIGHT, buff=0.5)
            .scale_to_fit_width(all_channels.width)
            .shift(DOWN * 1.2)
        )

        row_surr_rect = 0
        rows, cols = pixel_array_r.shape

        surr_rect = SurroundingRectangle(
            VGroup(
                iterations_r[0][
                    slice(row_surr_rect * cols, row_surr_rect * cols + cols)
                ],
                iterations_g[0][
                    slice(row_surr_rect * cols, row_surr_rect * cols + cols)
                ],
                iterations_b[0][
                    slice(row_surr_rect * cols, row_surr_rect * cols + cols)
                ],
            ),
            buff=0,
            color=REDUCIBLE_YELLOW,
        )

        method_t = (
            Text(methods_history[row_surr_rect], font="SF Mono")
            .set_color(REDUCIBLE_YELLOW)
            .scale(0.2)
            .next_to(
                iterations_r[0][
                    slice(row_surr_rect * cols, row_surr_rect * cols + cols)
                ],
                LEFT,
                buff=0.2,
            )
        )

        methods_vg = VGroup()
        methods_vg.add(method_t)

        self.play(FadeIn(first_iteration))
        self.play(Write(surr_rect), Write(method_t))

        self.wait()

        for i_r, i_g, i_b in zip(iterations_r[1:], iterations_g[1:], iterations_b[1:]):
            row_surr_rect += 1
            print(f"{row_surr_rect = }")
            print(slice(row_surr_rect * cols, row_surr_rect * cols + cols))

            next_iteration = (
                VGroup(i_r, i_g, i_b)
                .arrange(RIGHT, buff=0.5)
                .scale_to_fit_width(all_channels.width)
                .shift(DOWN * 1.2)
            )

            next_surr_rect = SurroundingRectangle(
                VGroup(
                    i_r[slice(row_surr_rect * cols, row_surr_rect * cols + cols)],
                    i_g[slice(row_surr_rect * cols, row_surr_rect * cols + cols)],
                    i_b[slice(row_surr_rect * cols, row_surr_rect * cols + cols)],
                ),
                buff=0,
            )

            method_t = (
                Text(methods_history[row_surr_rect], font="SF Mono")
                .set_color(REDUCIBLE_YELLOW)
                .scale(0.2)
                .next_to(
                    i_r[slice(row_surr_rect * cols, row_surr_rect * cols + cols)],
                    LEFT,
                    buff=0.2,
                )
            )
            methods_vg.add(method_t)

            self.play(
                Transform(first_iteration, next_iteration),
                Transform(surr_rect, next_surr_rect),
                Write(method_t),
            )

            self.wait()

        # show padding
        self.play(
            self.focus_on(pixel_array_mob_g[0:4]),
            FadeOut(pixel_array_mob_r),
            FadeOut(pixel_array_mob_b),
            run_time=3,
        )

        self.remove(*iterations_r, *iterations_g[:-1], *iterations_b)

        px_arr_mob = pixel_array_mob_g

        dashed_row = (
            VGroup(
                *[
                    VGroup(
                        DashedVMobject(Square()).set_stroke(
                            width=0.4, color=REDUCIBLE_GREEN
                        ),
                        Text("0", font="SF Mono").scale(1.5),
                    ).arrange(ORIGIN)
                    for _ in range(9)
                ]
            )
            .arrange(LEFT, buff=0)
            .scale_to_fit_width(px_arr_mob.width + px_arr_mob[0].width)
            .next_to(px_arr_mob, UP, buff=0, aligned_edge=RIGHT)
        )

        dashed_col = (
            VGroup(
                *[
                    VGroup(
                        DashedVMobject(Square()).set_stroke(
                            width=0.4, color=REDUCIBLE_GREEN
                        ),
                        Text("0", font="SF Mono").scale(1.5),
                    ).arrange(ORIGIN)
                    for _ in range(8)
                ]
            )
            .arrange(DOWN, buff=0)
            .scale_to_fit_height(px_arr_mob.height)
            .next_to(px_arr_mob, LEFT, buff=0)
        )

        up_arrow = (
            Arrow(
                px_arr_mob[0].get_center(),
                dashed_row[-2].get_center(),
                max_tip_length_to_length_ratio=0.09,
                max_stroke_width_to_length_ratio=3,
            )
            .set_color(REDUCIBLE_YELLOW)
            .scale(0.6)
        )

        left_arrow = (
            Arrow(
                px_arr_mob[0].get_center(),
                dashed_col[0].get_center(),
                max_tip_length_to_length_ratio=0.09,
                max_stroke_width_to_length_ratio=3,
            )
            .set_color(REDUCIBLE_YELLOW)
            .scale(0.6)
        )

        self.play(FadeIn(up_arrow))
        self.play(FadeIn(left_arrow))

        self.play(LaggedStart(FadeIn(dashed_row), FadeIn(dashed_col)), run_time=2)

        self.wait()

        self.play(
            FadeOut(dashed_col),
            FadeOut(dashed_row),
            FadeOut(left_arrow),
            FadeOut(up_arrow),
        )

        line = Line(px_arr_mob[0].get_left(), px_arr_mob[0].get_right()).set_color(
            REDUCIBLE_PURPLE
        )
        left_mark = (
            Line(UP * 0.03, DOWN * 0.03)
            .next_to(line, LEFT, buff=0)
            .set_color(REDUCIBLE_PURPLE)
        )
        right_mark = (
            Line(UP * 0.03, DOWN * 0.03)
            .next_to(line, RIGHT, buff=0)
            .set_color(REDUCIBLE_PURPLE)
        )

        length_vg = (
            VGroup(left_mark, line, right_mark)
            .next_to(iterations_g[-1][0], UP, buff=0.06)
            .set_stroke(width=1)
        )

        one_byte_t = (
            Text("1 byte", font="CMU Serif", weight=BOLD)
            .scale_to_fit_width(length_vg.width - length_vg.width / 10)
            .next_to(length_vg, UP, buff=0.06)
        )

        self.play(
            self.focus_on(iterations_g[-1][0:4].copy().shift(UP * 0.2)),
            FadeIn(iterations_g[-1]),
            FadeOut(iterations_r[-1]),
            FadeOut(iterations_b[-1]),
            FadeOut(px_arr_mob),
            run_time=3,
        )

        self.play(FadeIn(length_vg), FadeIn(one_byte_t))

        self.wait()

        original_pixels = (
            px_arr_mob[9:11]
            .copy()
            .next_to(iterations_g[-1][11], UP, buff=0.29)
            .scale(0.6)
        )

        pixel_values = VGroup(
            Text(str(pixel_array_g[1, 1]), font="SF Mono")
            .scale(0.1)
            .set_color(REDUCIBLE_GREEN_DARKER)
            .move_to(original_pixels[0]),
            Text(str(pixel_array_g[1, 2]), font="SF Mono")
            .scale(0.1)
            .set_color(WHITE)
            .move_to(original_pixels[1]),
        )

        step_1 = (
            Text(f"({pixel_array_g[1,2]} - {pixel_array_g[1,1]}) % 256", font="SF Mono")
            .scale(0.07)
            .next_to(original_pixels, DOWN, buff=0.03)
        )
        step_2 = (
            Text(f"({pixel_array_g[1,2] - pixel_array_g[1,1]}) % 256", font="SF Mono")
            .scale(0.07)
            .next_to(step_1, DOWN, buff=0.03)
        )
        step_3 = (
            Text(f"{(pixel_array_g[1,2] - pixel_array_g[1,1]) % 256}", font="SF Mono")
            .scale(0.07)
            .next_to(step_2, DOWN, buff=0.03)
        )

        surr_rect = SurroundingRectangle(
            iterations_g[-1][10], color=PURE_GREEN, buff=0
        ).set_stroke(width=1)

        self.play(
            self.focus_on(iterations_g[-1][9:14]),
            FadeOut(VGroup(*iterations_g[-1][0:8])),
            FadeOut(length_vg),
            FadeOut(one_byte_t),
            run_time=3,
        )

        self.play(FadeIn(original_pixels), FadeIn(pixel_values), Write(surr_rect))

        self.wait()

        self.play(FadeIn(step_1))

        self.wait()

        self.play(FadeIn(step_2))

        self.wait()

        self.play(
            FadeIn(step_3), Indicate(iterations_g[-1].numbers[10], color=PURE_GREEN)
        )

    def what_filter_to_use(self):
        """Natural question: what filter to choose"""
        none_t = Text("NONE", font="CMU Serif", weight=BOLD).scale(0.7)
        sub_t = Text("SUB", font="CMU Serif", weight=BOLD).scale(0.7)
        up_t = Text("UP", font="CMU Serif", weight=BOLD).scale(0.7)
        avg_t = Text("AVG", font="CMU Serif", weight=BOLD).scale(0.7)
        paeth_t = Text("PAETH", font="CMU Serif", weight=BOLD).scale(0.7)

        filter_types = (
            VGroup(none_t, sub_t, up_t, avg_t, paeth_t)
            .arrange(RIGHT, buff=1)
            .set_opacity(0.3)
        )

        self.play(LaggedStartMap(FadeIn, filter_types))

        for filter_name in filter_types:
            self.play(
                filter_name.animate.scale(1.2).set_opacity(1),
                rate_func=there_and_back_with_pause,
                run_time=1,
            )

        self.play(
            Write(
                Text("?", font="CMU Serif", weight=BOLD).next_to(
                    filter_types, DOWN, buff=1
                )
            )
        )

    def low_bit_depth_images(self):

        title = (
            Text("Low bit depth images", font="CMU Serif", weight=BOLD)
            .scale(1)
            .to_edge(UP)
        )

        random_row = np.random.randint(0, 255, (1, 4))

        # raw data
        row_mob = (
            PixelArray(random_row, color_mode="GRAY", include_numbers=True)
            .scale(2)
            .shift(DOWN * 0.6)
        )

        sq_brace = self.square_braces(row_mob[0], direction=UP)

        one_byte_t = (
            Text("1 Byte", font="SF Mono", weight=BOLD)
            .scale_to_fit_width(row_mob[0].width - row_mob[0].width / 4)
            .next_to(sq_brace, UP, buff=0.06)
        )

        one_bpp = (
            Text("8 bits/pixel", font="SF Mono", weight=BOLD)
            .scale(0.4)
            .next_to(row_mob, DOWN, aligned_edge=RIGHT)
        )
        four_bpp = (
            Text("6 bits/pixel", font="SF Mono", weight=BOLD)
            .scale(0.4)
            .next_to(row_mob, DOWN, aligned_edge=RIGHT)
        )

        self.play(Write(row_mob))

        self.wait()

        self.play(Write(sq_brace))

        self.play(Write(one_byte_t), Write(one_bpp))

        # show filtering action
        filter_sq = (
            Square(fill_color=REDUCIBLE_YELLOW, color=REDUCIBLE_YELLOW)
            .set_opacity(0.3)
            .set_stroke(width=5, opacity=1)
            .scale_to_fit_width(row_mob[0].width)
            .move_to(row_mob[0])
        )
        best_filter = Text("Best Filter: NONE", font="CMU Serif").scale(0.8)
        best_filter.next_to(title, DOWN)
        additional_anim = [FadeIn(best_filter)]
        # for i, mob in enumerate(row_mob[:-1]):
        #     self.play(
        #         filter_sq.animate.shift(RIGHT * mob.width)
        #     )

        # self.play(FadeOut(filter_sq))
        # self.wait()

        random_row_lbd = np.random.randint(0, 2**6, (1, 4))

        # raw data
        row_mob_lbd = (
            PixelArray(random_row_lbd, color_mode="GRAY", include_numbers=True)
            .scale(2)
            .shift(DOWN * 0.6)
        )

        [
            p.set_color(GRAY).set_opacity(0.3).set_stroke(GRAY, width=5, opacity=1)
            for p in row_mob_lbd
        ]
        row_mob_lbd.numbers.set_color(WHITE).set_opacity(1).set_stroke(width=0)

        six_b_pixel = (
            row_mob_lbd[0]
            .copy()
            .stretch_to_fit_width(row_mob_lbd[0].width + row_mob_lbd[0].width / 3)
            .next_to(row_mob_lbd[0], ORIGIN, aligned_edge=LEFT)
        )
        sq_brace_lbd = self.square_braces(six_b_pixel, direction=UP)

        self.wait()

        self.play(
            FadeTransform(row_mob, row_mob_lbd),
            Transform(sq_brace, sq_brace_lbd),
            one_byte_t.animate.next_to(sq_brace_lbd, UP, buff=0.06),
            FadeTransform(one_bpp, four_bpp),
            Write(title),
        )

        filter_sq = (
            filter_sq.copy()
            .stretch_to_fit_width(row_mob_lbd[0].width + row_mob_lbd[0].width / 3)
            .move_to(row_mob_lbd[0], aligned_edge=LEFT)
        )

        self.play(FadeIn(filter_sq))
        for i, mob in enumerate(row_mob_lbd[:-1:2]):
            if i == 0:
                self.play(filter_sq.animate.shift(RIGHT * mob.width), *additional_anim)
                continue
            self.play(filter_sq.animate.shift(RIGHT * filter_sq.width))

        self.play(FadeOut(filter_sq))
        self.wait()

    def palette_images(self):
        title = (
            Text("Palette Based Images", font="CMU Serif", weight=BOLD)
            .scale(1)
            .to_edge(UP)
        )

        self.play(FadeIn(title))

        img = ImageMobject("r_3_palette.png")
        px_array = img.get_pixel_array()
        palette = np.array(
            [[px_array[0, 0, :3], px_array[0, 1, :3], px_array[1, 2, :3]]]
        )

        index_palette = VGroup()
        for j in range(px_array.shape[0]):
            for i in range(px_array.shape[1]):

                if (px_array[j, i][:3] == palette[0, 0]).all():
                    index_palette.add(
                        Text("0", font="SF Mono", weight=BOLD)
                        .set_color(BLACK)
                        .scale(0.4)
                    )
                elif (px_array[j, i][:3] == palette[0, 1]).all():
                    index_palette.add(
                        Text("1", font="SF Mono", weight=BOLD)
                        .set_color(BLACK)
                        .scale(0.4)
                    )
                elif (px_array[j, i][:3] == palette[0, 2]).all():
                    index_palette.add(
                        Text("2", font="SF Mono", weight=BOLD)
                        .set_color(WHITE)
                        .scale(0.4)
                    )

        palette_mob = (
            PixelArray(palette, color_mode="RGB", include_numbers=False)
            .arrange(DOWN, buff=0)
            .scale(1.5)
            .shift(DOWN * 0.6)
        )

        img_mob = PixelArray(px_array[:, :, :3], color_mode="RGB")

        zero_index = Text("0", font="SF Mono", weight=BOLD).scale(0.8)
        one_index = Text("1", font="SF Mono", weight=BOLD).scale(0.8)
        two_index = Text("2", font="SF Mono", weight=BOLD).scale(0.8)

        indices = VGroup(zero_index, one_index, two_index)

        for palette, index in zip(palette_mob, indices):
            index.next_to(palette, LEFT, buff=0.2)

        full_palette = VGroup(indices, palette_mob)

        self.play(FadeIn(palette_mob), FadeIn(indices))
        self.wait()

        self.play(full_palette.animate.shift(LEFT * 2.6))

        img_mob.scale_to_fit_height(palette_mob.height).next_to(
            palette_mob, RIGHT, buff=1
        )

        for pixel, index in zip(img_mob, index_palette):
            index.next_to(pixel, ORIGIN)

        self.play(FadeIn(img_mob))

        self.wait()

        self.play(FadeIn(index_palette))

        best_filter = (
            Text("Best Filter: NONE", font="CMU Serif").scale(0.8).next_to(title, DOWN)
        )

        self.play(FadeIn(best_filter))

    def repeating_filters_performance(self):
        rows, cols = 10, 8
        random_data = np.random.randint(50, 80, (rows, cols))

        img_mob = (
            PixelArray(random_data, color_mode="GRAY")
            .scale(0.9)
            .to_edge(LEFT)
            .shift(LEFT * 2 + DOWN * 3)
        )

        comp_ratio = (
            Text("Compression Ratio", font="CMU Serif", weight=BOLD)
            .scale(0.7)
            .next_to(img_mob, RIGHT, buff=2)
            .shift(UP * 5)
        )

        cr_number = RVariable(100, label="comp_ratio", num_decimal_places=0).scale(0.9)
        cr_number[:-1].scale(0)

        cr_number[-1].next_to(comp_ratio, DOWN, buff=0.5)

        percentage = Text("%", font="SF Mono").scale(0.6)

        full_pct = (
            VGroup(percentage, cr_number[-1])
            .arrange(RIGHT, buff=0.2, aligned_edge=DOWN)
            .next_to(comp_ratio, DOWN, buff=0.5)
            .shift(RIGHT * 0.3)
        )

        filter_order = ["sub", "avg", "paeth", "avg", "avg", "up", "up"]

        filter_rects = (
            VGroup(
                *[
                    self.create_colored_row_with_filter_name(f).scale_to_fit_height(
                        img_mob[0].height
                    )
                    for f in filter_order
                ]
            )
            .arrange(DOWN, buff=0.3)
            .next_to(comp_ratio, DOWN)
        )

        self.play(FadeIn(img_mob), FadeIn(comp_ratio))
        self.wait()

        self.play(FadeIn(full_pct))

        # row 0
        row_0 = slice(0 * cols, 0 * cols + cols)
        self.play(
            FadeIn(filter_rects[0].next_to(img_mob[row_0], ORIGIN, aligned_edge=RIGHT)),
            cr_number.tracker.animate.set_value(90),
        )
        self.wait()

        # row 1
        row_1 = slice(1 * cols, 1 * cols + cols)
        self.play(
            FadeIn(filter_rects[1].next_to(img_mob[row_1], ORIGIN, aligned_edge=RIGHT)),
            cr_number.tracker.animate.set_value(80),
        )
        self.wait()

        # row 2 (change filter)
        row_2 = slice(2 * cols, 2 * cols + cols)
        self.play(
            FadeIn(filter_rects[2].next_to(img_mob[row_2], ORIGIN, aligned_edge=RIGHT)),
            cr_number.tracker.animate.set_value(85),
        )
        self.wait()

        self.play(
            FadeTransform(
                filter_rects[2],
                filter_rects[3].next_to(img_mob[row_2], ORIGIN, aligned_edge=RIGHT),
            ),
            cr_number.tracker.animate.set_value(80),
        )
        self.wait()

        # row 3
        row_3 = slice(3 * cols, 3 * cols + cols)
        self.play(
            FadeIn(filter_rects[4].next_to(img_mob[row_3], ORIGIN, aligned_edge=RIGHT)),
            cr_number.tracker.animate.set_value(70),
        )
        self.wait()

        # row 4
        row_4 = slice(4 * cols, 4 * cols + cols)
        self.play(
            FadeIn(filter_rects[5].next_to(img_mob[row_4], ORIGIN, aligned_edge=RIGHT)),
            cr_number.tracker.animate.set_value(68),
        )
        self.wait()

        # row 5
        row_5 = slice(5 * cols, 5 * cols + cols)
        self.play(
            FadeIn(filter_rects[6].next_to(img_mob[row_5], ORIGIN, aligned_edge=RIGHT)),
            cr_number.tracker.animate.set_value(67),
        )
        self.wait()

        self.play(
            full_pct.animate.scale(1.2).set_color(REDUCIBLE_YELLOW),
        )

    def combination_explosion(self):
        filter_names = ["none", "sub", "up", "avg", "paeth"]

        all_perms = product(filter_names, repeat=5)

        some_perms = list(all_perms)[800:3000:10]

        # self.camera.cairo_line_width_multiple = 0.003

        all_perms_vg = VGroup(
            *[self.create_filter_names_vg(perm) for perm in some_perms]
        ).arrange_in_grid(buff=1)

        self.play(FadeIn(all_perms_vg))
        self.wait()
        self.play(all_perms_vg.animate.scale(0.15).set_stroke(width=1), run_time=16)

    def msad_intro(self):
        """
        we filter with every method and take the filter
        that works best for each particular row.
        """
        title = Text(
            "Minimum sum of absolute differences", font="CMU Serif", weight=BOLD
        ).scale(0.8)

        rows, cols = 8, 8
        random_data = np.random.randint(50, 120, (rows, cols))

        img_mob = PixelArray(random_data, color_mode="GRAY").scale(0.9).shift(DOWN * 3)

        filter_order = ["sub", "avg", "paeth", "up", "sub", "up", "up"]

        filter_rects = (
            VGroup(
                *[
                    self.create_colored_row_with_filter_name(f).scale_to_fit_height(
                        img_mob[0].height
                    )
                    for f in filter_order
                ]
            )
            .arrange(DOWN, buff=0)
            .move_to(img_mob, aligned_edge=UL)
        )

        [
            r[0]
            .stretch_to_fit_width(img_mob.width)
            .move_to(img_mob, coor_mask=[1, 0, 0])
            for r in filter_rects
        ]

        self.play(FadeIn(title))
        self.wait()
        self.play(title.animate.to_edge(UP), FadeIn(img_mob))

        for rect in filter_rects:
            self.play(FadeIn(rect))

        self.play(*[FadeOut(mob) for mob in self.mobjects])
        """
        To do this, we’ll treat the filtered bytes as if 
        they were signed numbers. This is done by mapping the 128 — 255 
        range of color values to the -128 — -1 range,
        while leaving the 0 — 127 range intact
        """

        line = Line(LEFT * 4, RIGHT * 4).set_color(REDUCIBLE_PURPLE)
        left_mark = (
            Line(UP * 0.2, DOWN * 0.2)
            .next_to(line, LEFT, buff=0)
            .set_color(REDUCIBLE_PURPLE)
        )
        middle_mark = (
            Line(UP * 0.2, DOWN * 0.2)
            .next_to(line, ORIGIN, buff=0)
            .set_color(REDUCIBLE_PURPLE)
        )
        right_mark = (
            Line(UP * 0.2, DOWN * 0.2)
            .next_to(line, RIGHT, buff=0)
            .set_color(REDUCIBLE_PURPLE)
        )

        original_range = VGroup(left_mark, line, middle_mark, right_mark).set_stroke(
            width=7
        )
        zero = Text("0", font="SF Mono").scale(0.6).next_to(left_mark, DOWN, buff=0.2)

        one_27 = (
            Text("128", font="SF Mono").scale(0.6).next_to(middle_mark, DOWN, buff=0.2)
        )
        two_55 = (
            Text("255", font="SF Mono").scale(0.6).next_to(right_mark, DOWN, buff=0.2)
        )

        line2 = Line(ORIGIN, RIGHT * 4).set_color(REDUCIBLE_GREEN)
        left_mark2 = (
            Line(UP * 0.2, DOWN * 0.2)
            .next_to(line2, LEFT, buff=0)
            .set_color(REDUCIBLE_GREEN)
        )
        right_mark2 = (
            Line(UP * 0.2, DOWN * 0.2)
            .next_to(line2, RIGHT, buff=0)
            .set_color(REDUCIBLE_GREEN)
        )

        new_range = (
            VGroup(left_mark2, line2, right_mark2)
            .set_stroke(width=4)
            .next_to(original_range, UP, buff=0.3, aligned_edge=RIGHT)
        )

        minus_one = (
            Text("-1", font="SF Mono")
            .scale(0.6)
            .next_to(
                right_mark2,
                UP,
                buff=0.2,
            )
        )
        minus_128 = (
            Text("-128", font="SF Mono").scale(0.6).next_to(left_mark2, UP, buff=0.2)
        )

        random_row = np.array([[22, 19, 95, 145, 45, 72, 37, 9]])

        # raw data
        row_mob = PixelArray(random_row, color_mode="GRAY", include_numbers=True)

        # filtered, signed data
        filtered_row = random_row.copy()
        for i in range(1, random_row.shape[1]):
            filtered_row[0, i] = random_row[0, i] - random_row[0, i - 1]

        filtered_mob = PixelArray(filtered_row, color_mode="GRAY", include_numbers=True)
        [
            p.set_color(REDUCIBLE_YELLOW)
            .set_opacity(0.3)
            .set_stroke(REDUCIBLE_YELLOW, width=3, opacity=1)
            for p in filtered_mob
        ]
        filtered_mob.numbers.set_color(WHITE).set_opacity(1).set_stroke(width=0)

        # byte aligned data
        byte_aligned_row = self.sub_filter_row(random_row, return_row=True)
        byte_aligned_mob = PixelArray(
            byte_aligned_row.reshape((1, len(byte_aligned_row))),
            color_mode="GRAY",
            include_numbers=True,
        )
        [
            p.set_color(REDUCIBLE_PURPLE)
            .set_opacity(0.3)
            .set_stroke(REDUCIBLE_PURPLE, width=3, opacity=1)
            for p in byte_aligned_mob
        ]
        byte_aligned_mob.numbers.set_color(WHITE).set_opacity(1).set_stroke(width=0)

        # remapped data
        mapped_row = np.array([self.png_mapping(x) for x in byte_aligned_row])

        mapped_mob = PixelArray(
            mapped_row.reshape((1, len(byte_aligned_row))).astype(np.int16),
            color_mode="GRAY",
            include_numbers=True,
        )
        [
            p.set_color(REDUCIBLE_GREEN)
            .set_opacity(0.3)
            .set_stroke(REDUCIBLE_GREEN, width=3, opacity=1)
            for p in mapped_mob
        ]
        mapped_mob.numbers.set_color(WHITE).set_opacity(1).set_stroke(width=0)

        all_steps = (
            VGroup(row_mob, filtered_mob, byte_aligned_mob, mapped_mob)
            .arrange(DOWN, buff=0.4)
            .shift(RIGHT * 2)
        )

        raw_data_t = (
            Text("Raw data", font="CMU Serif", weight=BOLD)
            .scale(0.6)
            .next_to(row_mob, LEFT, buff=0.2)
        )
        filtered_data_t = (
            Text("Filtered, signed data (sub filter)", font="CMU Serif", weight=BOLD)
            .scale(0.6)
            .next_to(filtered_mob, LEFT, buff=0.2)
        )
        byte_aligned_t = (
            Text(r"Byte aligned data (mod 256)", font="CMU Serif", weight=BOLD)
            .scale(0.6)
            .next_to(byte_aligned_mob, LEFT, buff=0.2)
        )
        mapped_t = (
            Text("Remapped data", font="CMU Serif", weight=BOLD)
            .scale(0.6)
            .next_to(mapped_mob, LEFT, buff=0.2)
        )

        def create_new_mapping_line(length=7.5):

            line = Line(LEFT * length, RIGHT * length).set_color(REDUCIBLE_PURPLE)
            left_mark = (
                Line(UP * 0.2, DOWN * 0.2)
                .next_to(line, LEFT, buff=0)
                .set_color(REDUCIBLE_PURPLE)
            )
            middle_mark = (
                Line(UP * 0.2, DOWN * 0.2)
                .next_to(line, ORIGIN, buff=0)
                .set_color(REDUCIBLE_PURPLE)
            )
            right_mark = (
                Line(UP * 0.2, DOWN * 0.2)
                .next_to(line, RIGHT, buff=0)
                .set_color(REDUCIBLE_PURPLE)
            )

            original_range = VGroup(
                left_mark, line, middle_mark, right_mark
            ).set_stroke(width=7)
            zero = (
                Text("0", font="SF Mono").scale(0.6).next_to(left_mark, DOWN, buff=0.2)
            )

            one_27 = (
                Text("128", font="SF Mono")
                .scale(0.6)
                .next_to(middle_mark, DOWN, buff=0.2)
            )
            two_55 = (
                Text("255", font="SF Mono")
                .scale(0.6)
                .next_to(right_mark, DOWN, buff=0.2)
            )

            line2 = Line(ORIGIN, RIGHT * length).set_color(REDUCIBLE_GREEN)
            left_mark2 = (
                Line(UP * 0.2, DOWN * 0.2)
                .next_to(line2, LEFT, buff=0)
                .set_color(REDUCIBLE_GREEN)
            )
            right_mark2 = (
                Line(UP * 0.2, DOWN * 0.2)
                .next_to(line2, RIGHT, buff=0)
                .set_color(REDUCIBLE_GREEN)
            )

            new_range = (
                VGroup(left_mark2, line2, right_mark2)
                .set_stroke(width=4)
                .next_to(original_range, UP, buff=0.3, aligned_edge=RIGHT)
            )

            minus_one = (
                Text("-1", font="SF Mono")
                .scale(0.6)
                .next_to(
                    right_mark2,
                    UP,
                    buff=0.2,
                )
            )
            minus_128 = (
                Text("-128", font="SF Mono")
                .scale(0.6)
                .next_to(left_mark2, UP, buff=0.2)
            )

            return VGroup(
                original_range, new_range, zero, one_27, two_55, minus_one, minus_128
            )

        mapping_line = VGroup(
            original_range, new_range, zero, one_27, two_55, minus_one, minus_128
        )

        long_mapping_line = create_new_mapping_line().scale(0.8).to_edge(UP, buff=0.5)

        all_text = VGroup(raw_data_t, filtered_data_t, byte_aligned_t, mapped_t)

        rows_and_text = VGroup(all_steps, all_text).scale(0.8).to_corner(DR, buff=0.8)

        self.wait()
        self.play(
            FadeIn(raw_data_t),
            FadeIn(row_mob),
        )

        self.wait()

        self.play(
            FadeIn(filtered_data_t),
            FadeIn(filtered_mob),
        )

        self.wait()

        self.play(
            FadeIn(byte_aligned_t),
            FadeIn(byte_aligned_mob),
        )
        self.wait()
        self.play(Write(long_mapping_line))
        self.wait()

        self.play(FadeIn(mapped_mob), FadeIn(mapped_t))

        
        self.wait()

    def minimum_sum_of_absolute_differences(self):
        title = (
            Text("Minimum sum of absolute differences", font="CMU Serif", weight=BOLD)
            .scale(0.6)
            .to_edge(UP)
        )
        self.play(FadeIn(title))

        self.add_foreground_mobjects(title)

        rows, cols = (20, 8)

        random_data = np.random.randint(50, 80, (rows, cols))

        img_mob = (
            PixelArray(random_data, include_numbers=True, color_mode="GRAY")
            .scale(0.6)
            .shift(DOWN * 6 + LEFT * 1.5)
            .shift(UL * 1.7)
        )

        black_sq = (
            Square(color=BLACK)
            .scale(10)
            .set_fill(BLACK)
            .set_opacity(1)
            .next_to(img_mob, UP, buff=0)
            .shift(UP * img_mob[0].height)
        )

        winner_filters = VGroup()

        self.play(FadeIn(img_mob), FadeIn(black_sq), FadeIn(winner_filters))

        self.bring_to_front(title)

        self.wait()

        mask = (
            Difference(
                Square().scale(10),
                Rectangle(
                    width=img_mob[:cols].width, height=img_mob[:cols].height
                ).move_to(img_mob[:cols]),
            )
            .set_color(BLACK)
            .set_opacity(0.5)
        )

        self.play(FadeIn(mask))
        self.wait()

        filter_options = {
            "none": self.none_filter_row,
            "sub": self.sub_filter_row,
            "up": self.up_filter_row,
            "avg": self.avg_filter_row,
            "paeth": self.paeth_filter_row,
        }

        none_t = Text("NONE", font="SF Mono", weight=BOLD).scale(0.5)
        sub_t = Text("SUB", font="SF Mono", weight=BOLD).scale(0.5)
        up_t = Text("UP", font="SF Mono", weight=BOLD).scale(0.5)
        avg_t = Text("AVG", font="SF Mono", weight=BOLD).scale(0.5)
        paeth_t = Text("PAETH", font="SF Mono", weight=BOLD).scale(0.5)

        filter_score_table = (
            VGroup(none_t, sub_t, up_t, avg_t, paeth_t)
            .arrange(RIGHT, buff=0.8)
            .next_to(img_mob, RIGHT, buff=0.5)
            .shift(UP * 2)
        )
        final_score_t = (
            Text("Final filters score", font="SF Mono", weight=BOLD)
            .scale(0.5)
            .next_to(filter_score_table, UP, buff=0.3, aligned_edge=LEFT)
        )

        self.play(FadeIn(filter_score_table), FadeIn(final_score_t))
        count = 0
        for row in range(4):

            row_vg = img_mob[self.select_row_indices(row, (rows, cols))]
            filter_index = 0
            mobs_to_remove = VGroup()

            scores = []

            for name, filter_type in filter_options.items():

                filtered_mob, mapped_mob, filter_score = self.calculate_filtered_rows(
                    random_data, row, filter_func=filter_type
                )
                scores.append(filter_score)

                filtered_mob.scale_to_fit_height(row_vg.height).next_to(
                    row_vg, RIGHT, buff=0.5
                )

                mapped_mob.scale_to_fit_height(row_vg.height).next_to(
                    filtered_mob, DOWN, buff=0.2
                )

                name_t = (
                    Text(name, font="SF Mono", weight=BOLD)
                    .scale(0.5)
                    .next_to(filtered_mob, RIGHT, buff=0.2)
                )

                score_t = (
                    Text(str(filter_score), font="SF Mono")
                    .next_to(
                        filter_score_table[filter_index],
                        DOWN,
                        buff=0.2,
                    )
                    .scale(0.4)
                )
                filter_index += 1

                score_math = Tex("Score ", r"$= \sum_{i=1}^{n} |x_i|$").next_to(mapped_mob, DOWN, aligned_edge=LEFT)
                score_math[1][-3:-1].set_color(REDUCIBLE_GREEN_LIGHTER)
                self.play(
                    FadeIn(filtered_mob, shift=UP * 0.2),
                    FadeIn(mapped_mob, shift=UP * 0.2),
                    FadeIn(name_t, shift=UP * 0.2),
                    FadeIn(score_t),
                )
                if count == 0:
                    self.play(
                        FadeIn(score_math)
                    )

                self.wait()

                mobs_to_remove.add(score_t)
                self.play(
                    FadeOut(filtered_mob, shift=UP * 0.2),
                    FadeOut(mapped_mob, shift=UP * 0.2),
                    FadeOut(name_t, shift=UP * 0.2),
                )
                count += 1

                # end for loop

            winner_filter = (
                filter_score_table[np.argmin(scores)]
                .copy()
                .scale(0.8)
                .next_to(row_vg, LEFT, buff=0.2)
            )

            winner_filters.add(winner_filter)

            self.play(
                FadeIn(winner_filter),
                Indicate(filter_score_table[np.argmin(scores)]),
                FadeOut(mobs_to_remove),
            )

            self.bring_to_front(black_sq)
            self.bring_to_front(title)

            self.wait()

            self.play(
                img_mob.animate.shift(UP * row_vg.height),
                winner_filters.animate.shift(UP * row_vg.height),
            )
            self.wait()

    def png_decoding(self):
        random_data = np.random.randint(50, 80, (1, 10))

        img_mob = PixelArray(
            random_data, include_numbers=True, color_mode="GRAY"
        ).scale(0.9)

        byte_aligned_row = self.sub_filter_row(random_data, return_row=True)
        byte_aligned_mob = (
            PixelArray(
                byte_aligned_row.reshape((1, len(byte_aligned_row))),
                color_mode="GRAY",
                include_numbers=True,
            )
            .scale(0.9)
            .shift(DOWN * 0.5)
        )
        [
            p.set_color(REDUCIBLE_PURPLE)
            .set_opacity(0.3)
            .set_stroke(REDUCIBLE_PURPLE, width=3, opacity=1)
            for p in byte_aligned_mob
        ]
        byte_aligned_mob.numbers.set_color(WHITE).set_opacity(1).set_stroke(
            width=0, opacity=0
        )

        self.play(FadeIn(img_mob))
        self.wait()
        self.play(img_mob.animate.shift(UP * 2))

        original_t = (
            Text("Original row", font="CMU Serif", weight=BOLD)
            .scale(0.5)
            .next_to(img_mob, UP, buff=0.3, aligned_edge=LEFT)
        )
        self.play(Write(original_t))

        self.wait()

        self.play(FadeIn(byte_aligned_mob))

        self.wait()

        sub_t = (
            Text("SUB", font="CMU Serif", weight=BOLD)
            .scale(0.5)
            .next_to(byte_aligned_mob, UP, buff=0.3, aligned_edge=LEFT)
        )
        self.play(Write(sub_t))

        reconstructed_img_mob = img_mob.copy().next_to(byte_aligned_mob, DOWN, buff=1)

        self.play(FadeIn(reconstructed_img_mob[0]))
        arrows = VGroup()
        for i in range(1, random_data.shape[1]):
            arrow = Arrow(
                reconstructed_img_mob[i - 1].get_center(),
                byte_aligned_mob[i].get_center(),
                max_stroke_width_to_length_ratio=2.7,
                buff=0.6,
            ).set_color(REDUCIBLE_YELLOW)
            arrow.pop_tips()
            arrow.add_tip(
                tip_shape=ArrowTriangleFilledTip, tip_length=0.15, at_start=False
            )
            arrows.add(arrow)

            operation = (
                VGroup(
                    Text("(", font="SF Mono").scale(0.3),
                    reconstructed_img_mob[i - 1].copy().scale(0.2),
                    Text(" + ", font="SF Mono").scale(0.2),
                    byte_aligned_mob[i].copy().scale(0.2).set_stroke(width=2),
                    Text(")", font="SF Mono").scale(0.3),
                    Text(" % ", font="SF Mono").scale(0.2),
                    Text(" 256", font="SF Mono").scale(0.2),
                )
                .arrange(RIGHT, buff=0.05)
                .scale(2.5)
                .next_to(arrow, DOWN, buff=1.3)
            )

            self.play(Write(arrow), FadeIn(operation, shift=DOWN * 0.2))
            self.play(FadeIn(reconstructed_img_mob[i]))
            self.play(
                FadeOut(operation, shift=DOWN * 0.2),
                FadeOut(arrow),
            )

    #####################################################################

    # Functions

    def select_row_indices(self, row, shape):
        return slice(
            row * shape[1],
            row * shape[1] + shape[1],
        )

    def create_pixel_array(self, rows=8, cols=8):
        return (
            VGroup(
                *[Square(color=WHITE).set_stroke(width=2) for s in range(rows * cols)]
            )
            .arrange_in_grid(rows=rows, cols=cols, buff=0)
            .scale(0.15)
        )

    def sub_filter_img(self, img: ndarray):
        """
        Compute the sub filter for the whole input array
        """

        rows, cols = img.shape
        output = np.zeros(img.shape, dtype=np.int16)

        for j in range(rows):
            for i in range(1, cols):
                output[j, i] = int(img[j, i] - img[j, i - 1])

        output[:, 0] = img[:, 0]

        return output

    def up_filter_img(self, img: ndarray):
        """
        Compute the up filter for the whole input array
        """

        rows, cols = img.shape

        output = np.zeros(img.shape, dtype=np.uint8)

        for j in range(1, rows):
            for i in range(cols):
                output[j, i] = abs(img[j, i] - img[j - 1, i])

        output[0, :] = img[0, :]
        return output

    def avg_filter_img(self, img: ndarray):
        """
        Each byte is replaced with the difference between it and the average
        of the corresponding bytes to its left and above it, truncating any fractional part.
        """

        rows, cols = img.shape

        padded_data = np.pad(img, (1, 0), constant_values=0, mode="constant")
        output = np.zeros(img.shape, dtype=np.uint8)

        for j in range(1, rows + 1):
            for i in range(1, cols + 1):
                avg = (padded_data[j - 1, i] + padded_data[j, i - 1]) / 2
                output[j - 1, i - 1] = floor(abs(padded_data[j, i] - avg))

        return output

    def paeth_filter_img(self, img: ndarray):
        """
        Computes the paeth predictor for the whole input array
        """

        rows, cols = img.shape

        output = np.zeros(img.shape, dtype=np.uint8)
        padded_data = np.pad(img, (1, 0), constant_values=0, mode="constant")

        for j in range(1, rows + 1):
            for i in range(1, cols + 1):
                a = padded_data[j - 1, i - 1]  # up-left
                b = padded_data[j - 1, i]  # up
                c = padded_data[j, i - 1]  # left

                base_value = b + c - a

                # the dictionary is created in this specific order to account
                # for how PNG deals with ties. The specification says
                # that in the case of ties, left byte has precedence, then up.

                paeth_dict = {
                    abs(base_value - c): c,
                    abs(base_value - b): b,
                    abs(base_value - a): a,
                }

                # The min function will output the first instance it found in case of ties.
                # So that's why we built the dict that way.
                winner = paeth_dict[min(paeth_dict.keys())]

                # print(
                #     f"output[{j-1},{i-1}] is {padded_data[j,i] = } - {winner = } which equals {abs(padded_data[j, i] - winner)}"
                # )

                output[j - 1, i - 1] = abs(padded_data[j, i] - winner)

        print(output)
        return output

    def none_filter_row(self, img: ndarray, row=0, return_mob=False, return_row=False):
        rows, cols = img.shape
        output = img.copy()

        # do nothing

        if return_mob:
            return PixelArray(output, include_numbers=True, color_mode="GRAY")
        elif return_row:
            return output[row, :]
        else:
            return output

    def sub_filter_row(self, img: ndarray, row=0, return_mob=False, return_row=False):
        rows, cols = img.shape
        output = img.copy()

        for i in range(1, cols):
            output[row, i] = (img[row, i] - img[row, i - 1]) % 256

        output[row, 0] = img[row, 0]

        if return_mob:
            return PixelArray(output, include_numbers=True, color_mode="GRAY")
        elif return_row:
            return output[row, :]
        else:
            return output

    def up_filter_row(self, img: ndarray, row=0, return_mob=False, return_row=False):
        rows, cols = img.shape
        output = img.copy()

        if row == 0:
            pass
        else:
            for i in range(cols):
                output[row, i] = (img[row, i] - img[row - 1, i]) % 256

        if return_mob:
            return PixelArray(output, include_numbers=True, color_mode="GRAY")
        elif return_row:
            return output[row, :]
        else:
            return output

    def avg_filter_row(self, img: ndarray, row=0, return_mob=False, return_row=False):
        rows, cols = img.shape
        output = img.copy()

        padded_data = np.pad(img, (1, 0), constant_values=0, mode="constant")
        output = np.zeros(img.shape, dtype=np.uint8)

        padded_row = row + 1

        for i in range(1, cols + 1):
            avg = (padded_data[padded_row - 1, i] + padded_data[padded_row, i - 1]) / 2
            output[row, i - 1] = int((padded_data[padded_row, i] - avg) % 256)

        if return_mob:
            return PixelArray(output, include_numbers=True, color_mode="GRAY")
        elif return_row:
            return output[row, :]
        else:
            return output

    def paeth_filter_row(self, img: ndarray, row=0, return_mob=False, return_row=False):
        rows, cols = img.shape
        output = img.copy()

        padded_data = np.pad(img, (1, 0), constant_values=0, mode="constant")
        output = np.zeros(img.shape, dtype=np.uint8)

        padded_row = row + 1

        for i in range(1, cols + 1):
            a = padded_data[padded_row - 1, i - 1]  # up-left
            b = padded_data[padded_row - 1, i]  # up
            c = padded_data[padded_row, i - 1]  # left

            base_value = b + c - a

            paeth_dict = {
                abs(base_value - c): c,
                abs(base_value - b): b,
                abs(base_value - a): a,
            }
            winner = paeth_dict[min(paeth_dict.keys())]

            output[row, i - 1] = (padded_data[padded_row, i] - winner) % 256

        if return_mob:
            return PixelArray(output, include_numbers=True, color_mode="GRAY")
        elif return_row:
            return output[row, :]
        else:
            return output

    def filter_image_randomly(self, img: ndarray):
        methods = {
            0: self.sub_filter_row,
            1: self.up_filter_row,
            2: self.avg_filter_row,
            3: self.paeth_filter_row,
        }

        iterations_r = []
        iterations_g = []
        iterations_b = []
        methods_history = []

        pixel_array_r = img[:, :, 0]
        pixel_array_g = img[:, :, 1]
        pixel_array_b = img[:, :, 2]

        pixel_array_r_out = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        pixel_array_g_out = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        pixel_array_b_out = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        np.random.seed(2)

        for row in range(len(img)):
            random_method = np.random.randint(0, 4)

            if random_method == 0:
                methods_history.append("SUB")
            elif random_method == 1:
                methods_history.append("UP")
            elif random_method == 2:
                methods_history.append("AVG")
            elif random_method == 3:
                methods_history.append("PAETH")

            row_r = methods[random_method](pixel_array_r, row, return_row=True)

            pixel_array_r_out[row, :] = row_r

            iterations_r.append(
                PixelArray(
                    pixel_array_r_out, include_numbers=True, color_mode="GRAY"
                ).scale(0.4)
            )

            row_g = methods[random_method](pixel_array_g, row, return_row=True)

            pixel_array_g_out[row, :] = row_g

            iterations_g.append(
                PixelArray(
                    pixel_array_g_out, include_numbers=True, color_mode="GRAY"
                ).scale(0.4)
            )

            row_b = methods[random_method](pixel_array_b, row, return_row=True)

            pixel_array_b_out[row, :] = row_b

            iterations_b.append(
                PixelArray(
                    pixel_array_b_out, include_numbers=True, color_mode="GRAY"
                ).scale(0.4)
            )

        return iterations_r, iterations_g, iterations_b, methods_history

    def calculate_filtered_rows(self, input_array, row=0, filter_func=sub_filter_row):
        filtered_row = filter_func(input_array, row=row, return_row=True)

        filtered_mob = PixelArray(
            np.array([filtered_row], dtype=np.int16),
            include_numbers=True,
            color_mode="GRAY",
        )
        [
            p.set_color(REDUCIBLE_PURPLE)
            .set_opacity(0.5)
            .set_stroke(REDUCIBLE_PURPLE, width=3, opacity=1)
            for p in filtered_mob
        ]
        filtered_mob.numbers.set_color(WHITE).set_opacity(1).set_stroke(width=0)

        mapped_row = np.array(
            [self.png_mapping(p) for p in filtered_row],
            dtype=int,
        )

        filter_score = sum(abs(mapped_row))

        mapped_filtered_mob = PixelArray(
            np.array([mapped_row]), include_numbers=True, color_mode="GRAY"
        )
        [
            p.set_color(REDUCIBLE_GREEN)
            .set_opacity(0.5)
            .set_stroke(REDUCIBLE_GREEN, width=3, opacity=1)
            for p in mapped_filtered_mob
        ]
        [
            n.set_color(WHITE).set_opacity(1).scale(0.9).set_stroke(width=0)
            for n in mapped_filtered_mob.numbers
        ]

        return filtered_mob, mapped_filtered_mob, filter_score

    def map_num_range(self, x, input_start, input_end, output_start, output_end):
        return output_start + (
            (output_end - output_start) / (input_end - input_start)
        ) * (x - input_start)

    def png_mapping(self, x):
        if x > 127:
            return self.map_num_range(x, 128, 255, -128, -1)
        else:
            return x

    def create_filter_names_vg(self, name_order):
        return VGroup(
            *[self.create_colored_row_with_filter_name(n) for n in name_order]
        ).arrange(DOWN, buff=0.01)

    def create_colored_row_with_filter_name(self, filter_name):
        color_dict = {
            "none": GRAY,
            "sub": ORANGE,
            "up": REDUCIBLE_YELLOW,
            "avg": REDUCIBLE_PURPLE,
            "paeth": REDUCIBLE_GREEN,
        }
        color = color_dict[filter_name]

        return VGroup(
            Rectangle(height=0.6, width=5)
            .set_color(color)
            .set_fill(color, opacity=0.3),
            Text(filter_name, font="SF Mono").scale(0.5).set_stroke(opacity=0),
        ).arrange(ORIGIN)

    def get_channel_image(self, channel, mode="R"):
        new_channel = np.zeros((channel.shape[0], channel.shape[1], 3))
        for i in range(channel.shape[0]):
            for j in range(channel.shape[1]):
                if mode == "R":
                    new_channel[i][j] = np.array([channel[i][j], 0, 0])
                elif mode == "G":
                    new_channel[i][j] = np.array([0, channel[i][j], 0])
                else:
                    new_channel[i][j] = np.array([0, 0, channel[i][j]])

        return new_channel

    def focus_on(self, mobject, buff=2):
        return self.camera.frame.animate.set_width(mobject.width * buff).move_to(
            mobject
        )

    def square_braces(
        self,
        mob,
        color=REDUCIBLE_YELLOW,
        direction=UP,
        aligned_edge=LEFT,
        buff=SMALL_BUFF,
    ):

        line = Line(ORIGIN, RIGHT * mob.width).set_color(color)
        left_mark = (
            Line(ORIGIN, UP * line.width / 10)
            .next_to(line, LEFT, buff=0)
            .set_color(color)
        )

        right_mark = (
            Line(ORIGIN, UP * line.width / 10)
            .next_to(line, RIGHT, buff=0)
            .set_color(color)
        )

        return (
            VGroup(line, left_mark, right_mark)
            .set_stroke(width=2)
            .next_to(mob, direction=direction, buff=buff, aligned_edge=aligned_edge)
        )

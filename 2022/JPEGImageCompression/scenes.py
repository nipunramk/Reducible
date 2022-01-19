"""
File to define all the different scenes our video will be composed of.
"""

from math import sqrt

from manim import *
import cv2
from itertools import product
from pprint import pprint

from functions import *
from classes import *
from reducible_colors import *

np.random.seed(23)
config["assets_dir"] = "assets"

"""
Make sure you run manim CE with --disable_caching flag
If you run with caching, since there are some scenes that change pixel arrays,
there might be some unexpected behavior
E.g manim -pql JPEGImageCompression/image_compression.py --disable_caching
"""

class IntroduceRGBAndJPEG(Scene):
    def construct(self):
        r_t = Text("R", font="SF Mono", weight=MEDIUM).scale(3).set_color(PURE_RED)
        g_t = Text("G", font="SF Mono", weight=MEDIUM).scale(3).set_color(PURE_GREEN)
        b_t = Text("B", font="SF Mono", weight=MEDIUM).scale(3).set_color(PURE_BLUE)

        rgb_vg_h = VGroup(r_t, g_t, b_t).arrange(RIGHT, buff=2)
        rgb_vg_v = rgb_vg_h.copy().arrange(DOWN, buff=1).shift(LEFT * 0.7)

        self.play(LaggedStartMap(FadeIn, rgb_vg_h, lag_ratio=0.5))
        self.wait(2)
        self.play(Transform(rgb_vg_h, rgb_vg_v))

        red_t = (
            Text("ed", font="SF Mono", weight=MEDIUM)
            .set_color(PURE_RED)
            .next_to(r_t, RIGHT, buff=0.3, aligned_edge=DOWN)
        )
        green_t = (
            Text("reen", font="SF Mono", weight=MEDIUM)
            .set_color(PURE_GREEN)
            .next_to(g_t, RIGHT, buff=0.3, aligned_edge=DOWN)
        )
        blue_t = (
            Text("lue", font="SF Mono", weight=MEDIUM)
            .set_color(PURE_BLUE)
            .next_to(b_t, RIGHT, buff=0.3, aligned_edge=DOWN)
        )
        self.play(LaggedStartMap(FadeIn, [red_t, green_t, blue_t]))

        self.wait(2)

        self.play(LaggedStartMap(FadeOut, [rgb_vg_h, red_t, green_t, blue_t]))

        # pixels
        black = (
            Square(side_length=1)
            .set_color(BLACK)  # 0
            .set_opacity(1)
            .set_stroke(REDUCIBLE_VIOLET, width=3)
        )
        gray1 = (
            Square(side_length=1)
            .set_color(GRAY_E)  # 34
            .set_opacity(1)
            .set_stroke(REDUCIBLE_VIOLET, width=3)
        )
        gray2 = (
            Square(side_length=1)
            .set_color(GRAY_D)  # 68
            .set_opacity(1)
            .set_stroke(REDUCIBLE_VIOLET, width=3)
        )
        gray3 = (
            Square(side_length=1)
            .set_color(GRAY_B)  # 187
            .set_opacity(1)
            .set_stroke(REDUCIBLE_VIOLET, width=3)
        )
        white = (
            Square(side_length=1)
            .set_color("#FFFFFF")  # 255
            .set_opacity(1)
            .set_stroke(REDUCIBLE_VIOLET, width=3)
        )

        # pixel values

        pixels_vg = VGroup(black, gray1, gray2, gray3, white).arrange(RIGHT, buff=1)

        bk_t = Text("0", font="SF Mono", weight=MEDIUM).next_to(black, DOWN, buff=0.5).scale(0.5)
        g1_t = Text("34", font="SF Mono", weight=MEDIUM).next_to(gray1, DOWN, buff=0.5).scale(0.5)
        g2_t = Text("68", font="SF Mono", weight=MEDIUM).next_to(gray2, DOWN, buff=0.5).scale(0.5)
        g3_t = Text("187", font="SF Mono", weight=MEDIUM).next_to(gray3, DOWN, buff=0.5).scale(0.5)
        wh_t = Text("255", font="SF Mono", weight=MEDIUM).next_to(white, DOWN, buff=0.5).scale(0.5)

        self.play(LaggedStartMap(FadeIn, pixels_vg))
        self.wait(2)
        self.play(LaggedStartMap(FadeIn, [bk_t, g1_t, g2_t, g3_t, wh_t]))
        self.wait(2)
        self.play(LaggedStartMap(FadeOut, [pixels_vg, bk_t, g1_t, g2_t, g3_t, wh_t]))

        red_channel = (
            Rectangle(PURE_RED, width=3)
            .set_color(BLACK)
            .set_opacity(1)
            .set_stroke(PURE_RED, width=3)
        )
        green_channel = (
            Rectangle(PURE_GREEN, width=3)
            .set_color(BLACK)
            .set_opacity(1)
            .set_stroke(PURE_GREEN, width=3)
        )
        blue_channel = (
            Rectangle(PURE_BLUE, width=3)
            .set_color(BLACK)
            .set_opacity(1)
            .set_stroke(PURE_BLUE, width=3)
        )

        channels_vg_h = VGroup(red_channel, green_channel, blue_channel).arrange(
            RIGHT, buff=0.8
        )

        channels_vg_diagonal = (
            channels_vg_h.copy()
            .arrange(DOWN * 0.7 + RIGHT * 1.3, buff=-1.4)
            .shift(LEFT * 3)
        )

        self.play(LaggedStartMap(FadeIn, channels_vg_h))
        self.wait(2)
        self.play(Transform(channels_vg_h, channels_vg_diagonal))

        self.wait(2)

        pixel_r = (
            Square(side_length=0.1)
            .set_color(PURE_RED)
            .set_opacity(1)
            .align_to(red_channel, LEFT)
            .align_to(red_channel, UP)
        )
        pixel_g = (
            Square(side_length=0.1)
            .set_color(PURE_GREEN)
            .set_opacity(1)
            .align_to(green_channel, LEFT)
            .align_to(green_channel, UP)
        )
        pixel_b = (
            Square(side_length=0.1)
            .set_color(PURE_BLUE)
            .set_opacity(1)
            .align_to(blue_channel, LEFT)
            .align_to(blue_channel, UP)
        )

        self.play(FadeIn(pixel_r), FadeIn(pixel_g), FadeIn(pixel_b))
        self.wait(2)

        pixel_r_big = pixel_r.copy().scale(5).move_to(ORIGIN + UP * 1.5 + RIGHT * 1.7)
        pixel_g_big = pixel_g.copy().scale(5).next_to(pixel_r_big, DOWN, buff=1)
        pixel_b_big = pixel_b.copy().scale(5).next_to(pixel_g_big, DOWN, buff=1)

        self.play(
            TransformFromCopy(pixel_r, pixel_r_big),
            TransformFromCopy(pixel_g, pixel_g_big),
            TransformFromCopy(pixel_b, pixel_b_big),
        )
        self.wait(2)

        eight_bits_r = (
            Text("8 bits", font="SF Mono", weight=MEDIUM)
            .scale(0.4)
            .next_to(pixel_r_big, RIGHT, buff=0.3)
        )

        eight_bits_g = eight_bits_r.copy().next_to(pixel_g_big)
        eight_bits_b = eight_bits_r.copy().next_to(pixel_b_big)

        self.play(FadeIn(eight_bits_r), FadeIn(eight_bits_g), FadeIn(eight_bits_b))
        self.wait(2)

        brace = Brace(VGroup(eight_bits_r, eight_bits_g, eight_bits_b), RIGHT)

        self.play(Write(brace))

        twenty_four_bits = (
            Text("24 bits / pixel", font="SF Mono", weight=MEDIUM).scale(0.4).next_to(brace, RIGHT)
        )

        self.play(Write(twenty_four_bits), run_time=2)
        self.wait(2)

        self.play(
            Transform(twenty_four_bits, twenty_four_bits.copy().shift(UP * 0.5)),
            run_time=2,
        )
        self.wait(2)

        three_bytes = (
            Text("3 bytes / pixel", font="SF Mono", weight=MEDIUM)
            .scale(0.4)
            .next_to(twenty_four_bits, DOWN, buff=0.7)
        )
        self.play(Write(three_bytes))

        self.play(*[FadeOut(mob) for mob in self.mobjects])

        # flower image
        flower_image = ImageMobject("rose.jpg").scale(0.4)

        dimensions = (
            Text("2592 × 1944", font="SF Mono", weight=MEDIUM)
            .scale(0.7)
            .next_to(flower_image, DOWN, buff=0.3)
        )

        img_and_dims = Group(flower_image, dimensions).arrange(DOWN)
        img_and_dims_sm = img_and_dims.copy().scale(0.8).to_edge(LEFT, buff=1)

        self.play(FadeIn(img_and_dims))
        self.wait(2)
        self.play(Transform(img_and_dims, img_and_dims_sm), run_time=2)

        chart = (
            ReducibleBarChart(
                [15, 0.8],
                height=6,
                width=6,
                max_value=15,
                n_ticks=4,
                label_y_axis=True,
                y_axis_label_height=0.2,
                bar_label_scale_val=0.4,
                bar_names=["Uncompressed", "Compressed"],
                bar_colors=[REDUCIBLE_PURPLE, REDUCIBLE_YELLOW],
            )
            .scale(0.8)
            .to_edge(RIGHT, buff=1)
        )
        annotation = (
            Text("MB", font="SF Mono", weight=MEDIUM).scale(0.4).next_to(chart.y_axis, UP, buff=0.3)
        )

        self.play(Create(chart.x_axis), Create(chart.y_axis), run_time=3)
        self.play(
            Write(chart.y_axis_labels),
            Write(chart.bar_labels),
            Write(annotation),
            run_time=3,
        )

        # makes the bars grow from bottom to top
        for bar in chart.bars:
            self.add(bar)
            bar.generate_target()

            def update(mob, alpha):

                mob.become(mob.target)
                mob.move_to(bar.get_bottom())
                mob.stretch_to_fit_height(
                    alpha * bar.height,
                )
                mob.move_to(bar.get_top())

            self.play(UpdateFromAlphaFunc(bar, update_function=update), run_time=2)

        self.wait(3)

class JPEGDiagramScene(Scene):
    def construct(self):
        red_channel = (
            RoundedRectangle(corner_radius=0.1, fill_color=PURE_RED, width=3)
            .set_color(PURE_RED)
            .set_opacity(1)
            .set_stroke(WHITE, width=4)
        )
        green_channel = (
            RoundedRectangle(corner_radius=0.1, fill_color=PURE_GREEN, width=3)
            .set_color(PURE_GREEN)
            .set_opacity(1)
            .set_stroke(WHITE, width=4)
        )
        blue_channel = (
            RoundedRectangle(corner_radius=0.1, fill_color=PURE_BLUE, width=3)
            .set_color(PURE_BLUE)
            .set_opacity(1)
            .set_stroke(WHITE, width=4)
        )

        channels_vg_diagonal = VGroup(red_channel, green_channel, blue_channel).arrange(
            DOWN * 1.1 + RIGHT * 1.7, buff=-1.4
        )

        encoder_m = Module("Encoder", text_weight=BOLD)

        output_image = SVGMobject("jpg_file.svg").set_stroke(
            WHITE, width=5, background=True
        )

        lossy_compression_t = Text(
            "Lossy Compression", font="CMU Serif", weight=BOLD
        ).scale(2)

        # animations
        self.play(Write(lossy_compression_t))

        self.wait()

        self.play(lossy_compression_t.animate.scale(1.2 / 2).move_to(UP * 3.5))

        self.wait()

        encoding_flow = (
            VGroup(
                channels_vg_diagonal.scale_to_fit_height(encoder_m.height),
                encoder_m.scale(0.9),
                output_image,
            )
            .arrange(RIGHT, buff=2)
            .scale_to_fit_width(12)
        )

        # arrows
        arr1 = Arrow(
            start=channels_vg_diagonal.get_right(),
            end=encoder_m.get_left(),
            color=GRAY_B,
            stroke_width=3,
            buff=0.3,
            max_tip_length_to_length_ratio=0.08,
            max_stroke_width_to_length_ratio=2,
        )

        arr2 = Arrow(
            encoder_m.get_right(),
            output_image.get_left(),
            color=GRAY_B,
            stroke_width=3,
            buff=0.3,
            max_tip_length_to_length_ratio=0.08,
            max_stroke_width_to_length_ratio=2,
        )

        self.play(
            FadeIn(channels_vg_diagonal)
        )
        self.wait()

        self.play(
            Write(arr1)
        )
        self.wait()

        self.play(
            FadeIn(encoder_m)
        )

        self.wait()

        self.play(
            Write(arr2)
        )

        self.wait()

        self.play(
            FadeIn(output_image)
        )

        self.wait()

        self.play(
            encoding_flow.animate.shift(UP * 1.5),
            arr1.animate.shift(UP * 1.5),
            arr2.animate.shift(UP * 1.5)
        )
        self.wait()

        channels_vg_diagonal_copy = channels_vg_diagonal.copy()
        output_image_copy = output_image.copy()
        img_not_equal = (
            VGroup(
                channels_vg_diagonal_copy.scale_to_fit_height(output_image.height),
                output_image_copy
            )
            .arrange(RIGHT, buff=3.5)
            .move_to(ORIGIN)
            .shift(DOWN * 1.5)
        )
        db_arr = DoubleArrow(
            channels_vg_diagonal_copy.get_right(),
            output_image_copy.get_left(),
            color=GRAY_B,
            stroke_width=3,
            buff=0.3,
            max_tip_length_to_length_ratio=0.08,
            max_stroke_width_to_length_ratio=2,
        )
        cross_arr = Cross(color=PURE_RED, stroke_width=10).scale(0.22).move_to(db_arr)

        self.play(
            TransformFromCopy(channels_vg_diagonal, channels_vg_diagonal_copy),
            TransformFromCopy(output_image, output_image_copy),
        )

        self.wait()

        self.play(
            Write(db_arr)
        )

        self.play(Write(cross_arr))

        self.wait()

        scaling = 0.8286644206055882
        
        self.play(
            FadeOut(img_not_equal),
            FadeOut(db_arr),
            FadeOut(cross_arr),
            FadeOut(arr1),
            FadeOut(arr2),
            FadeOut(encoder_m),
            FadeOut(channels_vg_diagonal),
            output_image.animate.scale(scaling).move_to(LEFT * 5.3)
        )
        self.wait()

        decoder_m = Module("Decoder", text_weight=BOLD)

        channels_vg_diagonal.set_fill(opacity=0.8)
        decoding_flow = (
            VGroup(output_image, decoder_m.scale(scaling), channels_vg_diagonal.scale(scaling))
            .arrange(RIGHT, buff=2.45)
        )

        arr1 = Arrow(
            start=output_image.get_right(),
            end=decoder_m.get_left(),
            color=GRAY_B,
            stroke_width=3,
            buff=0.3,
            max_tip_length_to_length_ratio=0.08,
            max_stroke_width_to_length_ratio=2,
        )

        arr2 = Arrow(
            decoder_m.get_right(),
            channels_vg_diagonal.get_left(),
            color=GRAY_B,
            stroke_width=3,
            buff=0.3,
            max_tip_length_to_length_ratio=0.08,
            max_stroke_width_to_length_ratio=2,
        )

        self.play(
            Write(arr1),
        )
        self.wait()

        self.play(
            FadeIn(decoder_m)
        )
        self.wait()

        self.play(
            Write(arr2)
        )

        self.wait()

        self.play(
            FadeIn(channels_vg_diagonal)
        )
        self.wait()

        # In the future, I prefer to use this than to fade out Mobjects
        # You can add nice cross fades in final cut pro in post processing, no
        # need to fade out of every scene
        self.clear()

        channels_vg_diagonal.set_fill(opacity=1)

        # screen capture of cursor clicking an image and opening

        output_image = SVGMobject("jpg_file.svg").set_stroke(
            WHITE, width=3, background=True
        )

        output_red_channel = (
            RoundedRectangle(corner_radius=0.1, fill_color=PURE_RED, width=3)
            .set_color(PURE_RED)
            .set_opacity(0.8)
            .set_stroke(WHITE, width=4)
        )
        output_green_channel = (
            RoundedRectangle(corner_radius=0.1, fill_color=PURE_GREEN, width=3)
            .set_color(PURE_GREEN)
            .set_opacity(0.8)
            .set_stroke(WHITE, width=4)
        )
        output_blue_channel = (
            RoundedRectangle(corner_radius=0.1, fill_color=PURE_BLUE, width=3)
            .set_color(PURE_BLUE)
            .set_opacity(0.8)
            .set_stroke(WHITE, width=4)
        )

        output_channels_vg_diagonal = VGroup(output_red_channel, output_green_channel, output_blue_channel).arrange(
            DOWN * 1.1 + RIGHT * 1.7, buff=-1.4
        ).scale_to_fit_height(channels_vg_diagonal.height)

        full_flow = (
            VGroup(
                channels_vg_diagonal,
                encoder_m,
                output_image,
                decoder_m,
                output_channels_vg_diagonal,
            )
            .arrange(RIGHT, buff=2.5)
            .scale_to_fit_width(12)
            .shift(UP * 2)
        )

        arrows_flow = VGroup()
        for i in range(len(full_flow) - 1):
            arrows_flow.add(
                Arrow(
                    full_flow[i].get_right(),
                    full_flow[i + 1].get_left(),
                    color=GRAY_B,
                    stroke_width=3,
                    buff=0.3,
                    max_tip_length_to_length_ratio=0.08,
                    max_stroke_width_to_length_ratio=2,
                )
            )

        self.play(LaggedStart(*[FadeIn(mob) for mob in full_flow.submobjects]))
        self.wait()

        arrow_to_grow = Arrow(
            full_flow[0].get_right(),
            full_flow[-1].get_left(),
            color=GRAY_B,
            stroke_width=3,
            buff=0.3,
            max_tip_length_to_length_ratio=0.02,
            max_stroke_width_to_length_ratio=2,
        )
        self.bring_to_back(arrow_to_grow)
        self.play(GrowArrow(arrow_to_grow), run_time=2)

        self.wait()

        input_img = full_flow[0].copy().scale(1.5)
        output_img = full_flow[-1].copy().scale(1.5)
        not_equal = Tex("$\\neq$").scale(1.5)
        aux_vg = (
            VGroup(input_img, not_equal, output_img)
            .arrange(RIGHT, buff=1)
            .next_to(full_flow, DOWN, buff=1)
        )

        self.play(
            TransformFromCopy(full_flow[0], input_img),
            FadeIn(not_equal),
            TransformFromCopy(full_flow[-1], output_img),
        )

        self.wait()

        observation = Text("JPEG deliberately loses information", font="CMU Serif")
        observation.next_to(aux_vg, DOWN, buff=1)
        self.play(
            Write(observation)
        )

        self.wait()

        question = Text("What information can we get rid of?", font="CMU Serif")
        question.move_to(observation.get_center())
        self.play(
            ReplacementTransform(observation, question)
        )
        self.wait()
        # self.play(*[FadeOut(mob) for mob in self.mobjects])

        # final scene

        # big_frame = RoundedRectangle(
        #     height=9,
        #     width=16,
        #     stroke_width=10,
        #     stroke_opacity=1,
        #     color=REDUCIBLE_VIOLET,
        # ).scale(0.55)

        # question_mark_center = Text("?", font="CMU Serif", weight=BOLD).scale(4)

        # self.play(Create(big_frame))
        # self.wait(3)
        # self.play(Write(question_mark_center))
        # self.wait(3)

        # self.play(*[FadeOut(mob) for mob in self.mobjects])

class JPEGDiagramMap(MovingCameraScene):
    def construct(self):
        self.build_diagram()

    def build_diagram(self):

        self.play(self.camera.frame.animate.scale(2))
        # input image
        red_channel = (
            RoundedRectangle(corner_radius=0.1, fill_color=PURE_RED, width=3)
            .set_color(PURE_RED)
            .set_opacity(1)
            .set_stroke(WHITE, width=4)
        )
        green_channel = (
            RoundedRectangle(corner_radius=0.1, fill_color=PURE_GREEN, width=3)
            .set_color(PURE_GREEN)
            .set_opacity(1)
            .set_stroke(WHITE, width=4)
        )
        blue_channel = (
            RoundedRectangle(corner_radius=0.1, fill_color=PURE_BLUE, width=3)
            .set_color(PURE_BLUE)
            .set_opacity(1)
            .set_stroke(WHITE, width=4)
        )

        channels_vg_diagonal = VGroup(red_channel, green_channel, blue_channel).arrange(
            DOWN * 1.1 + RIGHT * 1.7, buff=-1.4
        )

        # output image
        output_image = SVGMobject("jpg_file.svg").set_stroke(
            WHITE, width=5, background=True
        )

        # big modules
        jpeg_encoder = Module(
            "JPEG Encoder",
            width=7,
            height=3,
            text_position=DOWN,
            text_weight=BOLD,
            text_scale=0.8,
        )
        jpeg_decoder = Module(
            "JPEG Decoder",
            width=7,
            height=3,
            text_position=UP,
            text_weight=BOLD,
            text_scale=0.8,
        )

        # color treatment
        color_treatment = Module(
            "Color treatment",
            REDUCIBLE_GREEN_DARKER,
            REDUCIBLE_GREEN_LIGHTER,
            height=jpeg_encoder.height,
            width=3,
            text_scale=0.5,
            text_position=UP,
            text_weight=BOLD,
        )

        ycbcr_m = Module(
            "YCbCr",
            fill_color=REDUCIBLE_YELLOW_DARKER,
            stroke_color=REDUCIBLE_YELLOW,
            height=1,
        ).scale_to_fit_width(color_treatment.width - 0.5)

        chroma_sub_m = Module(
            "Chroma Subsampling",
            fill_color=REDUCIBLE_YELLOW_DARKER,
            stroke_color=REDUCIBLE_YELLOW,
            text_scale=0.5,
            height=1,
        ).scale_to_fit_width(color_treatment.width - 0.5)

        color_modules = VGroup(ycbcr_m, chroma_sub_m).arrange(DOWN, buff=0.5)

        general_scale = 1.3
        color_treatment_w_modules = VGroup(color_treatment, color_modules).arrange(
            ORIGIN
        ).scale(general_scale)
        color_modules.shift(DOWN * 0.4)

        # small modules

        # encoding
        forward_dct_m = Module("Forward DCT", REDUCIBLE_YELLOW_DARKER, REDUCIBLE_YELLOW)
        forward_dct_icon = ImageMobject("dct.png").scale(0.2)
        forward_dct = Group(forward_dct_m, forward_dct_icon).arrange(DOWN, buff=0.5)

        quantizer_m = Module("Quantization", REDUCIBLE_YELLOW_DARKER, REDUCIBLE_YELLOW)
        quantizer_icon = ImageMobject("quantization.png").scale(0.2)
        quantizer = Group(quantizer_m, quantizer_icon).arrange(DOWN, buff=0.5)

        lossless_comp_m = Module(
            ["Lossless", "Encoder"],
            REDUCIBLE_YELLOW_DARKER,
            REDUCIBLE_YELLOW,
        )
        lossless_icon = ImageMobject("lossless.png").scale(0.2)
        lossless_icon.flip()
        lossless_icon.rotate(PI)
        lossless_comp = Group(lossless_comp_m, lossless_icon).arrange(DOWN, buff=0.5)

        encoding_modules = (
            Group(forward_dct, quantizer, lossless_comp)
            .arrange(RIGHT, buff=0.7)
            .scale_to_fit_width(jpeg_encoder.width - 0.5)
        )
        jpeg_encoder_w_modules = Group(jpeg_encoder, encoding_modules).arrange(
            ORIGIN,
        ).scale(general_scale)
        encoding_modules.shift(DOWN * 0.5)

        # decoding

        inverse_dct = Module("Inverse DCT", REDUCIBLE_YELLOW_DARKER, REDUCIBLE_YELLOW)
        dequantizer = Module("Dequantizer", REDUCIBLE_YELLOW_DARKER, REDUCIBLE_YELLOW)
        decoder = Module(["Lossless", "Decoder"], REDUCIBLE_YELLOW_DARKER, REDUCIBLE_YELLOW)

        decoding_modules = (
            VGroup(decoder, dequantizer, inverse_dct)
            .arrange(RIGHT, buff=0.5)
            .scale_to_fit_width(jpeg_decoder.width - 0.5)
        )
        jpeg_decoder_w_modules = VGroup(jpeg_decoder, decoding_modules).arrange(ORIGIN)
        decoding_modules.shift(DOWN * 0.5)

        # first row = encoding flow
        encoding_flow = Group(
            channels_vg_diagonal.scale(0.6),
            color_treatment_w_modules,
            jpeg_encoder_w_modules,
            output_image,
        ).arrange(RIGHT, buff=3)

        # second row = decoding flow
        decoding_flow = VGroup(
            output_image.copy(), jpeg_decoder_w_modules, channels_vg_diagonal.copy()
        ).arrange(RIGHT, buff=3)

        whole_map = Group(encoding_flow, decoding_flow).arrange(DOWN, buff=8)

        encode_arrows = VGroup()
        for i in range(len(encoding_flow.submobjects) - 1):
            encode_arrows.add(
                Arrow(
                    color=GRAY_B,
                    start=encoding_flow[i].get_right(),
                    end=encoding_flow[i + 1].get_left(),
                    stroke_width=3,
                    buff=0.3,
                    max_tip_length_to_length_ratio=0.08,
                    max_stroke_width_to_length_ratio=2,
                )
            )
        decode_arrows = VGroup()
        for i in range(len(decoding_flow.submobjects) - 1):
            decode_arrows.add(
                Arrow(
                    color=GRAY_B,
                    start=decoding_flow[i].get_right(),
                    end=decoding_flow[i + 1].get_left(),
                    stroke_width=3,
                    buff=0.3,
                    max_tip_length_to_length_ratio=0.08,
                    max_stroke_width_to_length_ratio=2,
                )
            )

        # whole map view state
        self.camera.frame.save_state()

        self.play(FadeIn(encoding_flow))
        self.play(FadeIn(encode_arrows))
        self.play(FadeIn(decoding_flow))
        self.play(FadeIn(decode_arrows))

        self.focus_on(encoding_flow, buff=1.1)

        self.wait(3)

        self.focus_on(channels_vg_diagonal)

        self.wait(3)

        self.focus_on(color_treatment_w_modules)

        self.wait(3)

        circumscribe_rect = chroma_sub_m.rect.copy().set_stroke(color=REDUCIBLE_VIOLET, width=7).set_fill(opacity=0)
        self.play(
            ShowPassingFlash(circumscribe_rect, time_width=1.5),
            run_time=2
        )
        self.wait()

        self.play(
            ShowPassingFlash(circumscribe_rect, time_width=1.5),
            run_time=2
        )
        self.wait()


        self.focus_on(encoding_flow, buff=1.3)

        self.wait(3)

        self.focus_on(jpeg_encoder_w_modules, buff=1.3)

        self.wait(3)

        self.focus_on(forward_dct)

        self.wait(3)

        self.focus_on(quantizer)

        self.wait(3)

        self.focus_on(lossless_comp)

        self.wait(3)

        # self.play(Restore(self.camera.frame), run_time=3)

    def focus_on(self, mobject, buff=2):
        self.play(
            self.camera.frame.animate.set_width(mobject.width * buff).move_to(mobject),
            run_time=3,
        )

class FocusOnRGB(JPEGDiagramMap):
    def construct(self):
        self.build_diagram()

    def build_diagram(self):

        self.play(self.camera.frame.animate.scale(2))
        # input image
        red_channel = (
            RoundedRectangle(corner_radius=0.1, fill_color=PURE_RED, width=3)
            .set_color(PURE_RED)
            .set_opacity(1)
            .set_stroke(WHITE, width=4)
        )
        green_channel = (
            RoundedRectangle(corner_radius=0.1, fill_color=PURE_GREEN, width=3)
            .set_color(PURE_GREEN)
            .set_opacity(1)
            .set_stroke(WHITE, width=4)
        )
        blue_channel = (
            RoundedRectangle(corner_radius=0.1, fill_color=PURE_BLUE, width=3)
            .set_color(PURE_BLUE)
            .set_opacity(1)
            .set_stroke(WHITE, width=4)
        )

        channels_vg_diagonal = VGroup(red_channel, green_channel, blue_channel).arrange(
            DOWN * 1.1 + RIGHT * 1.7, buff=-1.4
        )

        # output image
        output_image = SVGMobject("jpg_file.svg").set_stroke(
            WHITE, width=5, background=True
        )

        # big modules
        jpeg_encoder = Module(
            "JPEG Encoder",
            width=7,
            height=3,
            text_position=DOWN,
            text_weight=BOLD,
            text_scale=0.8,
        )
        jpeg_decoder = Module(
            "JPEG Decoder",
            width=7,
            height=3,
            text_position=UP,
            text_weight=BOLD,
            text_scale=0.8,
        )

        # color treatment
        color_treatment = Module(
            "Color treatment",
            REDUCIBLE_GREEN_DARKER,
            REDUCIBLE_GREEN_LIGHTER,
            height=jpeg_encoder.height,
            width=3,
            text_scale=0.5,
            text_position=UP,
            text_weight=BOLD,
        )

        ycbcr_m = Module(
            "YCbCr",
            fill_color=REDUCIBLE_YELLOW_DARKER,
            stroke_color=REDUCIBLE_YELLOW,
            height=1,
        ).scale_to_fit_width(color_treatment.width - 0.5)

        chroma_sub_m = Module(
            "Chroma Subsampling",
            fill_color=REDUCIBLE_YELLOW_DARKER,
            stroke_color=REDUCIBLE_YELLOW,
            text_scale=0.5,
            height=1,
        ).scale_to_fit_width(color_treatment.width - 0.5)

        color_modules = VGroup(ycbcr_m, chroma_sub_m).arrange(DOWN, buff=0.5)

        general_scale = 1.3
        color_treatment_w_modules = VGroup(color_treatment, color_modules).arrange(
            ORIGIN
        ).scale(general_scale)
        color_modules.shift(DOWN * 0.4)

        # small modules

        # encoding
        forward_dct_m = Module("Forward DCT", REDUCIBLE_YELLOW_DARKER, REDUCIBLE_YELLOW)
        forward_dct_icon = ImageMobject("dct.png").scale(0.2)
        forward_dct = Group(forward_dct_m, forward_dct_icon).arrange(DOWN, buff=0.5)

        quantizer_m = Module("Quantization", REDUCIBLE_YELLOW_DARKER, REDUCIBLE_YELLOW)
        quantizer_icon = ImageMobject("quantization.png").scale(0.2)
        quantizer = Group(quantizer_m, quantizer_icon).arrange(DOWN, buff=0.5)

        lossless_comp_m = Module(
            ["Lossless", "Encoder"],
            REDUCIBLE_YELLOW_DARKER,
            REDUCIBLE_YELLOW,
        )
        lossless_icon = ImageMobject("lossless.png").scale(0.2)
        lossless_comp = Group(lossless_comp_m, lossless_icon).arrange(DOWN, buff=0.5)

        encoding_modules = (
            Group(forward_dct, quantizer, lossless_comp)
            .arrange(RIGHT, buff=0.7)
            .scale_to_fit_width(jpeg_encoder.width - 0.5)
        )
        jpeg_encoder_w_modules = Group(jpeg_encoder, encoding_modules).arrange(
            ORIGIN,
        ).scale(general_scale)
        encoding_modules.shift(DOWN * 0.5)

        # decoding

        inverse_dct = Module("Inverse DCT", REDUCIBLE_YELLOW_DARKER, REDUCIBLE_YELLOW)
        dequantizer = Module("Dequantizer", REDUCIBLE_YELLOW_DARKER, REDUCIBLE_YELLOW)
        decoder = Module("Decoder", REDUCIBLE_YELLOW_DARKER, REDUCIBLE_YELLOW)

        decoding_modules = (
            VGroup(inverse_dct, dequantizer, decoder)
            .arrange(RIGHT, buff=0.5)
            .scale_to_fit_width(jpeg_decoder.width - 0.5)
        )
        jpeg_decoder_w_modules = VGroup(jpeg_decoder, decoding_modules).arrange(ORIGIN)
        decoding_modules.shift(DOWN * 0.5)

        # first row = encoding flow
        encoding_flow = Group(
            channels_vg_diagonal.scale(0.6),
            color_treatment_w_modules,
            jpeg_encoder_w_modules,
            output_image,
        ).arrange(RIGHT, buff=3)

        # second row = decoding flow
        decoding_flow = VGroup(
            output_image.copy(), jpeg_decoder_w_modules, channels_vg_diagonal.copy()
        ).arrange(RIGHT, buff=3)

        whole_map = Group(encoding_flow, decoding_flow).arrange(DOWN, buff=8)

        encode_arrows = VGroup()
        for i in range(len(encoding_flow.submobjects) - 1):
            encode_arrows.add(
                Arrow(
                    color=GRAY_B,
                    start=encoding_flow[i].get_right(),
                    end=encoding_flow[i + 1].get_left(),
                    stroke_width=3,
                    buff=0.3,
                    max_tip_length_to_length_ratio=0.08,
                    max_stroke_width_to_length_ratio=2,
                )
            )
        decode_arrows = VGroup()
        for i in range(len(decoding_flow.submobjects) - 1):
            decode_arrows.add(
                Arrow(
                    color=GRAY_B,
                    start=decoding_flow[i].get_right(),
                    end=decoding_flow[i + 1].get_left(),
                    stroke_width=3,
                    buff=0.3,
                    max_tip_length_to_length_ratio=0.08,
                    max_stroke_width_to_length_ratio=2,
                )
            )

        # whole map view state
        self.camera.frame.save_state()

        self.play(FadeIn(encoding_flow))
        self.play(FadeIn(encode_arrows))
        # self.play(FadeIn(decoding_flow))
        # self.play(FadeIn(decode_arrows))

        self.focus_on(encoding_flow, buff=1.1)

        self.wait(3)

        self.focus_on(channels_vg_diagonal)

        self.wait(3)

        self.focus_on(color_treatment_w_modules)

        self.wait(3)

        self.focus_on(encoding_flow, buff=1.3)

        self.wait(3)

        self.focus_on(jpeg_encoder_w_modules, buff=1.3)

        self.wait(3)

        self.focus_on(forward_dct)

        self.wait(3)

        self.focus_on(quantizer)

        self.wait(3)

        self.play(Restore(self.camera.frame), run_time=3)

    def focus_on(self, mobject, buff=2):
        self.play(
            self.camera.frame.animate.set_width(mobject.width * buff).move_to(mobject),
            run_time=3,
        )

class ShowConfusingImage(Scene):
    def construct(self):
        confusing_image = ImageMobject("confusing_image.png").scale(2).shift(UP * 0.5)
        clear_image = ImageMobject("clear_image.png").scale(2).shift(UP * 0.5)

        self.play(FadeIn(confusing_image))
        what_color = Text("What colors are tiles A and B?", font="CMU Serif", weight=MEDIUM).scale(0.8).move_to(DOWN * 2)
        self.wait()
        self.play(
            Write(what_color)
        )
        self.wait()

        same_color = Text("Tiles A and B are the same color!", font="CMU Serif", weight=MEDIUM).scale(0.8).move_to(DOWN * 2)

        self.play(
            ReplacementTransform(what_color, same_color)
        )
        self.wait()

        self.play(
            LaggedStart(
                confusing_image.animate.shift(LEFT * 3),
                FadeIn(clear_image.shift(RIGHT * 3)),
                lag_ratio=1,
            ),
            run_time=3,
        )
        self.wait()

        explanation = Text("Our eyes are more sensitve to brightness than color",  font="CMU Serif", weight=MEDIUM).scale(0.8)
        explanation.move_to(DOWN * 2)

        self.play(
            ReplacementTransform(same_color, explanation)
        )
        self.wait()
        self.play(
            explanation[-19:-9].animate.set_color_by_gradient(WHITE, GRAY_D),
            explanation[-5:].animate.set_color_by_gradient(PURE_RED, PURE_GREEN, BLUE)
        )
        self.wait()
        """ 
        some annotations could be made in post production indicating that 
        because we are more sensitive to brightness than color, we understand 
        the conflicting colors in the image as brighter and darker because of the shadow,
        giving their brightness a specific, semantic role within the image. that makes us
        not pay attention to the actual color the tiles have.
        """

class MotivateAndExplainRGB(ThreeDScene):
    def construct(self):
        self.color_cube_animation()

    def color_cube_animation(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        self.begin_ambient_camera_rotation(rate=0.4)
        self.move_camera(zoom=0.5)

        title = (
            Text("RGB color space", font="CMU Serif", weight=BOLD)
            .scale(0.9)
            .to_edge(UP, buff=0.6)
        )

        self.add_fixed_in_frame_mobjects(title)
        self.wait(3)

        # change this value to 8 or 16 for the final renders.
        # 8 is pretty slow already, we may not need
        # that much resolution for this small explanation.
        color_resolution = 8

        cubes_rgb = (
            self.create_color_space_cube(
                coords2rgbcolor, color_res=color_resolution, cube_side_length=0.8
            )
            .scale_to_fit_height(5)
            .move_to(ORIGIN)
        )

        cubes_rgb_expanded = (
            self.create_color_space_cube(
                coords2rgbcolor,
                color_res=color_resolution,
                cube_side_length=0.8,
                buff=0.2,
            )
            .scale_to_fit_height(5)
            .move_to(ORIGIN)

        )

        self.wait()

        self.play(FadeIn(cubes_rgb))

        self.wait(6)

        self.play(Transform(cubes_rgb, cubes_rgb_expanded))

        anim_group = []
        # this loop removes every cube that is not in the grayscale diagonal of the RGB colorspace.
        # to do that, we calculate what coordinates a particular cube lives in, via their index.
        # any diagonal cube will have their coordinates matching, so we remove everything else.
        for index, cube in enumerate(cubes_rgb):
            coords = index2coords(index, base=color_resolution)
            print(coords)
            if not coords[0] == coords[1] == coords[2]:
                anim_group.append(FadeOut(cube))

        self.play(*anim_group)

        self.wait(6)


    def create_color_space_cube(
        self,
        color_space_func,
        color_res=8,
        cube_side_length=0.1,
        buff=0,
    ):
        """
        Creates a YCbCr cube composed of many smaller cubes. The `color_res` argument defines
        how many cubes there will be to represent the full spectrum, particularly color_res ^ 3.
        Works exactly like `create_rgb_cube` but for YCrCb colors.

        @param: color_space_func - A function that defines what color space we are going to use.

        @param: color_res - defines the number of cubes in every dimension. Higher values yield a finer
        representation of the space, but are very slow to deal with. It is recommended that color_res is a power of two.

        @param: cube_side_length - defines the side length of each individual cube

        @param: buff - defines how much space there will be between each cube

        @return: Group - returns a group of 3D cubes colored to form the color space
        """

        MAX_COLOR_RES = 256
        discrete_ratio = MAX_COLOR_RES // color_res

        offset = cube_side_length + buff
        cubes = []

        for i in range(color_res):
            for j in range(color_res):
                for k in range(color_res):

                    i_discrete = i * discrete_ratio
                    j_discrete = j * discrete_ratio
                    k_discrete = k * discrete_ratio

                    color = color_space_func(i_discrete, j_discrete, k_discrete)

                    curr_cube = Cube(
                        fill_color=color, fill_opacity=1, side_length=cube_side_length
                    ).shift((LEFT * i + UP * j + OUT * k) * offset)

                    cubes.append(curr_cube)

        cubes_rgb = Group(*cubes)

        return cubes_rgb


class MotivateAndExplainYCbCr(Scene):
    def construct(self):
        self.ycbcr_explanation()
        self.yuv_plane_animation()

    def ycbcr_explanation(self):
        y_t = Text("Y", font="CMU Serif", weight=BOLD).scale(1.5).set_color(GRAY_B)
        u_t = (
            Text("Cb", font="CMU Serif", weight=BOLD)
            .scale(1.5)
            .set_color_by_gradient("#FFFF00", "#0000FF")
        )
        v_t = (
            Text("Cr", font="CMU Serif", weight=BOLD)
            .scale(1.5)
            .set_color_by_gradient("#00FF00", "#FF0000")
        )

        v_full_t = (
            Text("Chroma red", font="CMU Serif", weight=BOLD)
            .scale(1.5)
            .set_color_by_gradient("#00FF00", "#FF0000")
        )

        u_full_t = (
            Text("Chroma blue", font="CMU Serif", weight=BOLD)
            .scale(1.5)
            .set_color_by_gradient("#FFFF00", "#0000FF")
        )

        ycbcr_vg = VGroup(y_t, u_t, v_t).arrange(RIGHT, buff=1).scale(2)
        ycbcr_vg_vert = (
            VGroup(y_t.copy().scale(0.7), u_full_t, v_full_t)
            .arrange(DOWN, buff=1)
            .scale_to_fit_height(5)
        )

        # YCbCr stands for Y, Chroma Blue and Chroma Red.

        self.play(
            LaggedStartMap(FadeIn, ycbcr_vg, lag_ratio=0.5),
        )

        self.wait()
        self.play(
            # Y
            Transform(
                ycbcr_vg[0],
                ycbcr_vg_vert[0],
            ),
            # C to chroma
            Transform(
                ycbcr_vg[1][0],
                ycbcr_vg_vert[1][:6],
            ),
            # b to blue
            Transform(
                ycbcr_vg[1][1],
                ycbcr_vg_vert[1][6:],
            ),
            # c to chroma
            Transform(
                ycbcr_vg[2][0],
                ycbcr_vg_vert[2][:6],
            ),
            # r to red
            Transform(
                ycbcr_vg[2][1],
                ycbcr_vg_vert[2][6:],
            ),
        )
        self.wait(2)

        # This color space aims to separate the luminance, or brightness
        # from the color components for a given value.
        self.play(Circumscribe(ycbcr_vg_vert[0], color=REDUCIBLE_VIOLET, run_time=2))

        self.wait(2)

        self.play(Circumscribe(ycbcr_vg_vert[1:], color=REDUCIBLE_VIOLET, run_time=2))

        self.play(*[FadeOut(mob) for mob in self.mobjects])

        self.wait()

    def yuv_plane_animation(self):
        yuv_title = (
            Text("YCbCr color space", font="CMU Serif", weight=BOLD)
            .scale(0.9)
            .to_edge(UP, buff=0.6)
        )

        color_plane_0 = self.create_yuv_plane(y=0, color_res=32).move_to(DOWN * 0.5)
        color_plane_127 = self.create_yuv_plane(y=127, color_res=32).move_to(DOWN * 0.5)
        color_plane_255 = self.create_yuv_plane(y=255, color_res=32).move_to(DOWN * 0.5)
        color_plane_0_loop = color_plane_0.copy()

        color_planes = {}
        for i in range(50, 256, 50):
            color_planes.update(
                {i: self.create_yuv_plane(i, color_res=32).move_to(color_plane_0)}
            )

        print(color_planes)

        y = 0.00

        y_number = (
            RVariable(var=y, label="Y")
            .scale(0.7)
            .next_to(color_plane_0, DOWN, buff=0.5)
        )

        self.play(FadeIn(color_plane_0), FadeIn(yuv_title), FadeIn(y_number))

        self.wait()

        for y, plane in color_planes.items():
            self.play(
                Transform(color_plane_0, plane),
                y_number.tracker.animate.set_value(y / 255),
                run_time=2,
            )
            self.wait(2)

        self.play(
            Transform(color_plane_0, color_plane_0_loop),
            y_number.tracker.animate.set_value(0.00001),
        )
        self.wait(2)
        self.play(FadeOut(color_plane_0), FadeOut(y_number))
        self.wait()

        planes_diag = (
            Group(color_plane_0, color_plane_127, color_plane_255)
            .arrange(DOWN * 1.9 + RIGHT * 1.7, buff=-1.5)
            .scale(0.7)
            .move_to(DOWN * 0.5)
        )

        planes_from_side = (
            planes_diag.copy()
            .arrange(IN, buff=1)
            .rotate(-87 * DEGREES, Y_AXIS)
            .move_to(DOWN * 0.5)
        )

        self.play(FadeIn(planes_diag))
        self.wait()
        self.play(Transform(planes_diag, planes_from_side))
        self.wait()

        y_line = Line(
            planes_from_side[0].get_left(), planes_from_side[-1].get_right()
        ).set_stroke(color=[WHITE, BLACK], width=8)

        y_0 = (
            Text("0", font="SF Mono", weight=MEDIUM)
            .scale(0.4)
            .next_to(planes_from_side[0], UP, buff=0.3)
        )
        y_05 = (
            Text("0.5", font="SF Mono", weight=MEDIUM)
            .scale(0.4)
            .next_to(planes_from_side[1], UP, buff=0.3)
        )
        y_1 = (
            Text("1", font="SF Mono", weight=MEDIUM)
            .scale(0.4)
            .next_to(planes_from_side[2], UP, buff=0.3)
        )

        self.play(
            Write(y_line),
            LaggedStart(FadeIn(y_0), FadeIn(y_05), FadeIn(y_1), lag_ratio=0.1),
            run_time=2,
        )
        self.wait()

        self.play(*[FadeOut(mob) for mob in self.mobjects])

    def create_yuv_plane(self, y=127, color_res=64, return_data=False):
        """
        Creates an array of data that corresponds to the U and V values mapped out
        at a specific y setting. This can return the ndarray itself or an ImageMobject
        ready to be used.

        Color res at 32 using linear interpolation for the resampling can be quite
        fast and efficient while still giving the illusion of a very smooth gradient.
        """
        color_plane_data = [
            [y, u * (256 // color_res), v * (256 // color_res)]
            for u in range(color_res)
            for v in range(color_res)
        ]

        rgb_conv = np.array(
            [ycbcr2rgb4map(c) for c in color_plane_data], dtype=np.uint8
        ).reshape((color_res, color_res, 3))

        if return_data:
            return rgb_conv

        mob = ImageMobject(rgb_conv)
        mob.set_resampling_algorithm(RESAMPLING_ALGORITHMS["linear"])
        mob.scale_to_fit_width(4)

        return mob


class ImageUtils(Scene):
    def construct(self):
        NUM_PIXELS = 32
        HEIGHT = 4
        new_image = self.down_sample_image(
            "duck", NUM_PIXELS, NUM_PIXELS, image_height=HEIGHT
        )
        print("New down-sampled image shape:", new_image.get_pixel_array().shape)
        self.wait()
        self.add(new_image)
        self.wait()
        pixel_grid = self.get_pixel_grid(new_image, NUM_PIXELS)
        self.add(pixel_grid)
        self.wait()
        pixel_array = new_image.get_pixel_array().copy()

        pixel_array[-1, 6, :] = [0, 0, 255, 255]
        pixel_array[-2, 5, :] = [0, 0, 255, 255]
        pixel_array[-2, 6, :] = [0, 0, 255, 255]
        pixel_array[-1, 5, :] = [0, 0, 255, 255]

        adjusted_image = self.get_image_mob(pixel_array, height=HEIGHT)
        # self.remove(pixel_grid)
        # self.wait()
        self.play(
            new_image.animate.become(adjusted_image),
            # pixel_grid.animate.become(pixel_grid.copy())
        )
        self.wait()

    def get_image_mob(self, pixel_array, height=4):
        """
        @param pixel_array: multi-dimensional np.array[uint8]
        @return: ImageMobject of pixel array with given height
        """
        image = ImageMobject(pixel_array)
        # height of value None will just return original image mob size
        if height:
            image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
            image.height = height
        return image

    def down_sample_image(
        self, filepath, num_horiz_pixels, num_vert_pixels, image_height=4
    ):
        """
        @param: filepath - file name of image to down sample
        @param: num_horiz_pixels - number of horizontal pixels in down sampled image
        @param: num_vert_pixels - number of vertical pixels in down sampled image
        """
        assert (
            num_horiz_pixels == num_vert_pixels
        ), "Non-square downsampling not supported"
        original_image = ImageMobject(filepath)
        original_image_pixel_array = original_image.get_pixel_array()
        width, height, num_channels = original_image_pixel_array.shape
        horizontal_slice = self.get_indices(width, num_horiz_pixels)
        vertical_slice = self.get_indices(height, num_vert_pixels)
        new_pixel_array = self.sample_pixel_array_from_slices(
            original_image_pixel_array, horizontal_slice, vertical_slice
        )
        new_image = ImageMobject(new_pixel_array)
        new_image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        new_image.height = image_height
        assert (
            new_pixel_array.shape[0] == num_horiz_pixels
            and new_pixel_array.shape[1] == num_vert_pixels
        ), self.get_assert_error_message(
            new_pixel_array, num_horiz_pixels, num_vert_pixels
        )
        return new_image

    def sample_pixel_array_from_slices(
        self, original_pixel_array, horizontal_slice, vertical_slice
    ):
        return original_pixel_array[horizontal_slice][:, vertical_slice]

    def get_indices(self, size, num_pixels):
        """
        @param: size of row or column
        @param: down-sampled number of pixels
        @return: an array of indices with len(array) = num_pixels
        """
        array = []
        index = 0
        float_index = 0
        while float_index < size:
            index = round(float_index)
            array.append(index)
            float_index += size / num_pixels
        return np.array(array)

    def get_assert_error_message(
        self, new_pixel_array, num_horiz_pixels, num_vert_pixels
    ):
        return f"Resizing performed incorrectly: expected {num_horiz_pixels} x {num_vert_pixels} but got {new_pixel_array.shape[0]} x {new_pixel_array.shape[1]}"

    def get_pixel_grid(self, image, num_pixels_in_dimension, color=WHITE):
        side_length_single_cell = image.height / num_pixels_in_dimension
        pixel_grid = VGroup(
            *[
                Square(side_length=side_length_single_cell).set_stroke(
                    color=color, width=1, opacity=0.5
                )
                for _ in range(num_pixels_in_dimension ** 2)
            ]
        )
        pixel_grid.arrange_in_grid(rows=num_pixels_in_dimension, buff=0)
        return pixel_grid

    def get_yuv_image_from_rgb(self, pixel_array, mapped=True):
        """
        Extracts the Y, U and V channels from a given image.

        @param: pixel_array - the image to be processed
        @param: mapped - boolean. if true, return the YUV data mapped back to RGB for presentation purposes. Otherwise,
        return the y, u, and v channels directly for further processing, such as chroma subsampling.
        """
        # discard alpha channel
        rgb_img = pixel_array[:, :, :3]
        # channels need to be flipped to BGR for openCV processing
        rgb_img = rgb_img[:, :, [2, 1, 0]]
        img_yuv = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(img_yuv)

        if not mapped:
            return y, u, v
        else:
            lut_u, lut_v = make_lut_u(), make_lut_v()

            # Convert back to BGR so we display the images
            y = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
            u = cv2.cvtColor(u, cv2.COLOR_GRAY2BGR)
            v = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)

            u_mapped = cv2.LUT(u, lut_u)
            v_mapped = cv2.LUT(v, lut_v)

            # Flip channels to RGB
            y_rgb = y[:, :, [2, 1, 0]]
            u_mapped_rgb = u_mapped[:, :, [2, 1, 0]]
            v_mapped_rgb = v_mapped[:, :, [2, 1, 0]]

            return y_rgb, u_mapped_rgb, v_mapped_rgb

    def chroma_downsample_image(self, pixel_array, mode="4:2:2"):
        """
        Applies chroma downsampling to the image. Takes the average of the block.

        Modes supported are the most common ones: 4:2:2 and 4:2:0.

        @param: pixel_array - the image to be processed
        @param: mode - a string, either `4:2:2` or `4:2:0`, corresponding to 4:2:2 and 4:2:0 subsampling respectively.
        @param: image - returns back the image in RGB format with subsampling applied
        """
        assert mode in (
            "4:2:2",
            "4:2:0",
        ), "Please choose one of the following {'4:2:2', '4:2:0'}"

        y, u, v = self.get_yuv_image_from_rgb(pixel_array, mapped=False)

        out_u = np.zeros(u.shape)
        out_v = np.zeros(v.shape)
        # Downsample with a window of 2 in the horizontal direction
        if mode == "4:2:2":
            # first the u channel
            for i in range(0, u.shape[0], 2):
                out_u[i : i + 2] = np.mean(u[i : i + 2], axis=0)

            # then the v channel
            for i in range(0, v.shape[0], 2):
                out_v[i : i + 2] = np.mean(v[i : i + 2], axis=0)

        # Downsample with a window of 2 in both directions
        elif mode == "4:2:0":
            for i in range(0, u.shape[0], 2):
                for j in range(0, u.shape[1], 2):

                    out_u[i : i + 2, j : j + 2] = int(
                        np.round(np.mean(u[i : i + 2, j : j + 2]))
                    )

            for i in range(0, v.shape[0], 2):
                for j in range(0, v.shape[1], 2):
                    out_v[i : i + 2, j : j + 2] = np.mean(v[i : i + 2, j : j + 2])

        ycbcr_sub = np.stack(
            (y, np.round(out_u).astype("uint8"), np.round(out_v).astype("uint8")),
            axis=2,
        )

        return cv2.cvtColor(ycbcr_sub, cv2.COLOR_YUV2RGB)

    def chroma_subsample_image(self, pixel_array, mode="4:2:2"):
        """
        Applies chroma subsampling to the image. Takes the top left pixel of the block.

        Modes supported are the most common ones: 4:2:2 and 4:2:0.

        @param: pixel_array - the image to be processed
        @param: mode - a string, either `4:2:2` or `4:2:0`, corresponding to 4:2:2 and 4:2:0 subsampling respectively.
        @param: image - returns back the image in RGB format with subsampling applied
        """
        assert mode in (
            "4:2:2",
            "4:2:0",
        ), "Please choose one of the following {'4:2:2', '4:2:0'}"

        y, u, v = self.get_yuv_image_from_rgb(pixel_array, mapped=False)

        out_u = np.zeros(u.shape)
        out_v = np.zeros(v.shape)
        # Subsample with a window of 2 in the horizontal direction
        if mode == "4:2:2":
            # first the u channel
            for i in range(0, u.shape[0], 2):
                out_u[i : i + 2] = u[i]

            # then the v channel
            for i in range(0, v.shape[0], 2):
                out_v[i : i + 2] = v[i]

        # Subsample with a window of 2 in both directions
        elif mode == "4:2:0":
            for i in range(0, u.shape[0], 2):
                for j in range(0, u.shape[1], 2):

                    out_u[i : i + 2, j : j + 2] = u[i, j]

            for i in range(0, v.shape[0], 2):
                for j in range(0, v.shape[1], 2):
                    out_v[i : i + 2, j : j + 2] = v[i, j]

        ycbcr_sub = np.stack(
            (y, np.round(out_u).astype("uint8"), np.round(out_v).astype("uint8")),
            axis=2,
        )

        return cv2.cvtColor(ycbcr_sub, cv2.COLOR_YUV2RGB)


class IntroChromaSubSamplingFileSize(ImageUtils):
    def construct(self):
        # self.animate_chroma_downsampling()

        # top left
        # self.animate_chroma_subsampling()
        self.show_real_world_image_subsampled()
        # self.show_file_size_calculation()

    # average
    def animate_chroma_downsampling(self):
        gradient_image = ImageMobject("r.png")
        gradient_image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        gradient_image.scale(30)

        pix_array = gradient_image.get_pixel_array()

        gradient = PixelArray(pix_array[:, :, :-1]).scale(0.3)

        y, u, v = self.get_yuv_image_from_rgb(pix_array, mapped=True)
        y_channel = PixelArray(y[:, :, 0], color_mode="GRAY").scale(0.5)
        u_channel = PixelArray(u, color_mode="RGB").scale(0.5)
        v_channel = PixelArray(v, color_mode="RGB").scale(0.5)

        y_t = Text("Y", font="SF Mono", weight=BOLD).scale(1.5).set_color(GRAY_A)
        u_t = (
            Text("Cb", font="SF Mono", weight=BOLD)
            .scale(1.5)
            .set_color_by_gradient("#FFFF00", "#0000FF")
        )
        v_t = (
            Text("Cr", font="SF Mono", weight=BOLD)
            .scale(1.5)
            .set_color_by_gradient("#00FF00", "#FF0000")
        )

        y_vg = VGroup(y_channel, y_t).arrange(DOWN, buff=0.5)
        u_vg = VGroup(u_channel, u_t).arrange(DOWN, buff=0.5)
        v_vg = VGroup(v_channel, v_t).arrange(DOWN, buff=0.5)

        self.play(Write(gradient))

        self.wait(2)

        self.play(gradient.animate.shift(UP * 2))

        self.wait(2)

        yuv_channels = (
            VGroup(y_vg, u_vg, v_vg)
            .arrange(RIGHT, buff=0.5)
            .scale(0.5)
            .shift(DOWN * 2),
        )

        self.play(
            TransformFromCopy(gradient, y_channel),
            TransformFromCopy(gradient, u_channel),
            TransformFromCopy(gradient, v_channel),
        )

        self.wait(2)

        self.play(FadeIn(y_t), FadeIn(u_t), FadeIn(v_t))

        self.wait(2)

        # we are more sensitive to brightness than we are to color

        self.play(
            Circumscribe(
                y_channel,
                time_width=5,
                color=REDUCIBLE_YELLOW,
                run_time=5,
                fade_in=True,
                fade_out=True,
            ),
        )

        self.wait(4)

        chroma_title = (
            Text("Chroma downsampling: 4:2:0", font="CMU Serif", weight=MEDIUM)
            .scale(0.7)
            .to_edge(UP, buff=1)
        )

        self.play(
            LaggedStart(
                AnimationGroup(
                    FadeOut(gradient),
                    FadeOut(y_vg),
                    FadeOut(v_vg),
                    FadeOut(u_t),
                ),
                u_channel.animate.move_to(DOWN * 0.5 + LEFT * 3).scale(2),
                FadeIn(chroma_title),
                lag_ratio=1,
            ),
            run_time=3,
        )

        self.wait(2)

        u_slice = u_channel[0:2]

        kernel = (
            Square(color=YELLOW)
            .scale_to_fit_width(u_slice.width)
            .move_to(u_slice, aligned_edge=UP)
        )
        self.add_foreground_mobject(kernel)

        self.play(FadeIn(kernel))

        """
        This for loop runs through the loaded image's u channel, 
        and creates a new downsampled version as it goes through it.
        """

        new_u_channel = VGroup()
        new_pixel_guide = (
            u_channel[0]
            .copy()
            .scale_to_fit_width(kernel.width)
            .next_to(u_channel, RIGHT, buff=2)
        )

        equal_sign = MathTex("=").scale(1.5)

        four_pixels_vg = (
            VGroup(*[u_channel[0].copy() for i in range(4)])
            .arrange_in_grid(rows=2, cols=2)
            .scale_to_fit_height(new_pixel_guide.height)
        ).set_opacity(0)

        guide_vg = (
            VGroup(new_pixel_guide, equal_sign, four_pixels_vg)
            .arrange(RIGHT, buff=1)
            .next_to(u_channel, RIGHT, buff=2)
        )
        average_t = (
            Text("Average of:", font="SF Mono", weight=MEDIUM)
            .scale(0.3)
            .next_to(four_pixels_vg, UP, buff=0.5)
        )

        new_pixel_annotation = (
            Square()
            .set_opacity(0)
            .scale_to_fit_height(new_pixel_guide.height)
            .move_to(new_pixel_guide)
        )
        new_pixel_t = (
            Text("New pixel value:", font="SF Mono", weight=MEDIUM)
            .scale(0.3)
            .next_to(new_pixel_annotation, UP, buff=0.5)
        )
        self.play(FadeIn(equal_sign), FadeIn(average_t), FadeIn(new_pixel_t))

        for j in range(0, pix_array.shape[1] * 2, 4):
            for i in range(0, pix_array.shape[0], 2):
                sq_ul = u_channel[i + j * 4]
                sq_ur = u_channel[i + j * 4 + 1]

                sq_dl = u_channel[i + j * 4 + 8]
                sq_dr = u_channel[i + j * 4 + 8 + 1]

                next_slice = u_channel[i + j * 4 : i + j * 4 + 2]

                sq_ul_rgb = hex_to_rgb(sq_ul.color)
                sq_ur_rgb = hex_to_rgb(sq_ur.color)
                sq_dl_rgb = hex_to_rgb(sq_dl.color)
                sq_dr_rgb = hex_to_rgb(sq_dr.color)

                four_pixels = np.stack((sq_ul_rgb, sq_ur_rgb, sq_dl_rgb, sq_dr_rgb))

                avg = np.average(four_pixels, axis=0) * 255

                self.play(
                    kernel.animate.move_to(next_slice, aligned_edge=UP),
                )

                new_pixel = (
                    Pixel(avg, color_mode="RGB")
                    .scale_to_fit_width(kernel.width)
                    .move_to(kernel)
                )
                new_u_channel.add(new_pixel)

                last_pixel = new_pixel_annotation
                new_pixel_annotation = new_pixel.copy().move_to(new_pixel_guide)

                last_four_pixel = four_pixels_vg
                four_pixels_vg = (
                    VGroup(sq_ul.copy(), sq_ur.copy(), sq_dl.copy(), sq_dr.copy())
                    .arrange_in_grid(rows=2, cols=2)
                    .scale_to_fit_height(new_pixel_annotation.height)
                    .move_to(last_four_pixel)
                )

                self.play(
                    FadeIn(new_pixel),
                    FadeIn(new_pixel_annotation),
                    FadeIn(four_pixels_vg),
                    FadeOut(last_pixel),
                    FadeOut(last_four_pixel),
                )

                self.wait()

        self.remove(u_channel)
        self.wait()
        self.play(
            FadeOut(kernel),
            FadeOut(equal_sign),
            FadeOut(last_pixel),
            FadeOut(new_pixel_annotation),
            FadeOut(new_pixel_guide),
            FadeOut(last_four_pixel),
            FadeOut(four_pixels_vg),
            FadeOut(new_pixel_t),
            FadeOut(average_t),
            new_u_channel.animate.move_to(ORIGIN, coor_mask=[1, 0, 0]),
        )

        new_v_channel = VGroup()
        for j in range(0, pix_array.shape[1] * 2, 4):
            for i in range(0, pix_array.shape[0], 2):
                sq_ul = v_channel[i + j * 4].color
                sq_ur = v_channel[i + j * 4 + 1].color

                sq_dl = v_channel[i + j * 4 + 8].color
                sq_dr = v_channel[i + j * 4 + 8 + 1].color

                next_slice = v_channel[i + j * 4 : i + j * 4 + 2]

                sq_ul_rgb = hex_to_rgb(sq_ul)
                sq_ur_rgb = hex_to_rgb(sq_ur)
                sq_dl_rgb = hex_to_rgb(sq_dl)
                sq_dr_rgb = hex_to_rgb(sq_dr)

                four_pixels = np.stack((sq_ul_rgb, sq_ur_rgb, sq_dl_rgb, sq_dr_rgb))

                avg = np.average(four_pixels, axis=0) * 255

                new_pixel = Pixel(avg, color_mode="RGB").scale_to_fit_width(
                    v_channel[0:2].width
                )

                new_v_channel.add(new_pixel)

        new_v_channel.arrange_in_grid(rows=4, cols=4, buff=0)

        y_channel.scale_to_fit_height(new_u_channel.height).next_to(
            new_u_channel, LEFT, buff=0.4
        )
        new_v_channel.scale_to_fit_height(new_u_channel.height).next_to(
            new_u_channel, RIGHT, buff=0.4
        )

        self.play(
            FadeIn(y_channel, shift=RIGHT),
            FadeIn(new_v_channel, shift=LEFT),
            run_time=3,
        )
        self.wait(3)

        sub_pix_array = self.chroma_downsample_image(pix_array)
        subsampled_image = (
            PixelArray(sub_pix_array, color_mode="RGB")
            .scale_to_fit_height(y_channel.height)
            .move_to(DOWN * 0.5)
        )
        gradient.scale_to_fit_height(subsampled_image.height)

        sub_channels = VGroup(y_channel, new_u_channel, new_v_channel)

        self.play(
            FadeTransform(y_channel, subsampled_image),
            FadeTransform(new_v_channel, subsampled_image),
            FadeTransform(new_u_channel, subsampled_image),
            run_time=3,
        )

        aux_vg = (
            VGroup(gradient, subsampled_image.copy())
            .arrange(RIGHT, buff=2)
            .move_to(DOWN * 0.5)
        )

        self.play(
            subsampled_image.animate.move_to(aux_vg[1]),
            FadeIn(gradient, shift=LEFT),
        )
        original_text = (
            Text("Original image", font="CMU Serif", weight=MEDIUM)
            .scale(0.5)
            .next_to(gradient, DOWN, buff=0.4)
        )
        subsampled_text = (
            Text("Downsampled Image", font="CMU Serif", weight=MEDIUM)
            .scale(0.5)
            .next_to(subsampled_image, DOWN, buff=0.4)
        )

        self.play(FadeIn(original_text), FadeIn(subsampled_text))

        self.wait()

    # top left
    def animate_chroma_subsampling(self):
        gradient_image = ImageMobject("r.png")
        gradient_image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        gradient_image.scale(30)

        pix_array = gradient_image.get_pixel_array()

        gradient = PixelArray(pix_array[:, :, :-1]).scale(0.3)

        y, u, v = self.get_yuv_image_from_rgb(pix_array, mapped=True)
        y_channel = PixelArray(y[:, :, 0], color_mode="GRAY").scale(0.5)
        u_channel = PixelArray(u, color_mode="RGB").scale(0.5)
        v_channel = PixelArray(v, color_mode="RGB").scale(0.5)

        y_t = Text("Y", font="SF Mono", weight=BOLD).scale(1.5).set_color(GRAY_A)
        u_t = (
            Text("Cb", font="SF Mono", weight=BOLD)
            .scale(1.5)
            .set_color_by_gradient("#FFFF00", "#0000FF")
        )
        v_t = (
            Text("Cr", font="SF Mono", weight=BOLD)
            .scale(1.5)
            .set_color_by_gradient("#00FF00", "#FF0000")
        )

        y_vg = VGroup(y_channel, y_t).arrange(DOWN, buff=0.5)
        u_vg = VGroup(u_channel, u_t).arrange(DOWN, buff=0.5)
        v_vg = VGroup(v_channel, v_t).arrange(DOWN, buff=0.5)

        self.play(Write(gradient))

        self.wait(2)

        self.play(gradient.animate.shift(UP * 2))

        self.wait(2)

        yuv_channels = (
            VGroup(y_vg, u_vg, v_vg)
            .arrange(RIGHT, buff=0.5)
            .scale(0.5)
            .shift(DOWN * 2),
        )

        self.play(
            TransformFromCopy(gradient, y_channel),
            TransformFromCopy(gradient, u_channel),
            TransformFromCopy(gradient, v_channel),
        )

        self.wait(2)

        self.play(FadeIn(y_t), FadeIn(u_t), FadeIn(v_t))

        self.wait(2)

        # we are more sensitive to brightness than we are to color

        self.play(
            Circumscribe(
                y_channel,
                time_width=5,
                color=REDUCIBLE_YELLOW,
                run_time=5,
                fade_in=True,
                fade_out=True,
            ),
        )

        self.wait(4)

        chroma_title = (
            Text("Chroma subsampling: 4:2:0", font="CMU Serif", weight=MEDIUM)
            .scale(0.7)
            .to_edge(UP, buff=1)
        )

        self.play(
            LaggedStart(
                AnimationGroup(
                    FadeOut(gradient),
                    FadeOut(y_vg),
                    FadeOut(v_vg),
                    FadeOut(u_t),
                ),
                u_channel.animate.move_to(DOWN * 0.5 + LEFT * 3).scale(2),
                FadeIn(chroma_title),
                lag_ratio=1,
            ),
            run_time=3,
        )

        self.wait(2)

        u_slice = u_channel[0:2]

        kernel = (
            Square(color=YELLOW)
            .scale_to_fit_width(u_slice.width)
            .move_to(u_slice, aligned_edge=UP)
        )
        self.add_foreground_mobject(kernel)

        self.play(FadeIn(kernel))

        """
        This for loop runs through the loaded image's u channel, 
        and creates a new subsampled version as it goes through it.
        """

        new_u_channel = VGroup()
        new_pixel_guide = (
            u_channel[0]
            .copy()
            .scale_to_fit_width(kernel.width)
            .next_to(u_channel, RIGHT, buff=2)
        )

        equal_sign = MathTex("=").scale(1.5)

        four_pixels_vg = (
            VGroup(*[u_channel[0].copy() for i in range(4)])
            .arrange_in_grid(rows=2, cols=2)
            .scale_to_fit_height(new_pixel_guide.height)
        ).set_opacity(0)

        guide_vg = (
            VGroup(new_pixel_guide, equal_sign, four_pixels_vg)
            .arrange(RIGHT, buff=1)
            .next_to(u_channel, RIGHT, buff=2)
        )
        average_t = (
            Text("Top left of:", font="SF Mono", weight=MEDIUM)
            .scale(0.3)
            .next_to(four_pixels_vg, UP, buff=0.5)
        )

        new_pixel_annotation = (
            Square()
            .set_opacity(0)
            .scale_to_fit_height(new_pixel_guide.height)
            .move_to(new_pixel_guide)
        )
        new_pixel_t = (
            Text("New pixel value:", font="SF Mono", weight=MEDIUM)
            .scale(0.3)
            .next_to(new_pixel_annotation, UP, buff=0.5)
        )
        self.play(FadeIn(equal_sign), FadeIn(average_t), FadeIn(new_pixel_t))

        for j in range(0, pix_array.shape[1] * 2, 4):
            for i in range(0, pix_array.shape[0], 2):
                sq_ul = u_channel[i + j * 4]
                sq_ur = u_channel[i + j * 4 + 1]

                sq_dl = u_channel[i + j * 4 + 8]
                sq_dr = u_channel[i + j * 4 + 8 + 1]

                next_slice = u_channel[i + j * 4 : i + j * 4 + 2]

                self.play(
                    kernel.animate.move_to(next_slice, aligned_edge=UP),
                )

                new_pixel = (
                    Pixel(hex_to_rgb(sq_ul.color) * 255, color_mode="RGB")
                    .scale_to_fit_width(kernel.width)
                    .move_to(kernel)
                )
                new_u_channel.add(new_pixel)

                last_pixel = new_pixel_annotation
                new_pixel_annotation = new_pixel.copy().move_to(new_pixel_guide)

                last_four_pixel = four_pixels_vg
                four_pixels_vg = (
                    VGroup(
                        sq_ul.copy().scale(1.5),
                        sq_ur.copy().set_opacity(0.5),
                        sq_dl.copy().set_opacity(0.5),
                        sq_dr.copy().set_opacity(0.5),
                    )
                    .arrange_in_grid(rows=2, cols=2)
                    .scale_to_fit_height(new_pixel_annotation.height)
                    .move_to(last_four_pixel)
                )

                self.play(
                    FadeIn(new_pixel),
                    FadeIn(new_pixel_annotation),
                    FadeIn(four_pixels_vg),
                    FadeOut(last_pixel),
                    FadeOut(last_four_pixel),
                )

                self.wait()

        self.remove(u_channel)
        self.wait()
        self.play(
            FadeOut(kernel),
            FadeOut(equal_sign),
            FadeOut(last_pixel),
            FadeOut(new_pixel_annotation),
            FadeOut(new_pixel_guide),
            FadeOut(last_four_pixel),
            FadeOut(four_pixels_vg),
            FadeOut(new_pixel_t),
            FadeOut(average_t),
            new_u_channel.animate.move_to(ORIGIN, coor_mask=[1, 0, 0]),
        )

        new_v_channel = VGroup()
        for j in range(0, pix_array.shape[1] * 2, 4):
            for i in range(0, pix_array.shape[0], 2):
                sq_ul = v_channel[i + j * 4].color

                next_slice = v_channel[i + j * 4 : i + j * 4 + 2]

                new_pixel = Pixel(
                    hex_to_rgb(sq_ul) * 255, color_mode="RGB"
                ).scale_to_fit_width(v_channel[0:2].width)

                new_v_channel.add(new_pixel)

        new_v_channel.arrange_in_grid(rows=4, cols=4, buff=0)

        y_channel.scale_to_fit_height(new_u_channel.height).next_to(
            new_u_channel, LEFT, buff=0.4
        )
        new_v_channel.scale_to_fit_height(new_u_channel.height).next_to(
            new_u_channel, RIGHT, buff=0.4
        )

        self.play(
            FadeIn(y_channel, shift=RIGHT),
            FadeIn(new_v_channel, shift=LEFT),
            run_time=3,
        )
        self.wait(3)

        sub_pix_array = self.chroma_subsample_image(pix_array)
        subsampled_image = (
            PixelArray(sub_pix_array, color_mode="RGB")
            .scale_to_fit_height(y_channel.height)
            .move_to(DOWN * 0.5)
        )
        gradient.scale_to_fit_height(subsampled_image.height)

        sub_channels = VGroup(y_channel, new_u_channel, new_v_channel)

        self.play(
            FadeTransform(y_channel, subsampled_image),
            FadeTransform(new_v_channel, subsampled_image),
            FadeTransform(new_u_channel, subsampled_image),
            run_time=3,
        )

        aux_vg = (
            VGroup(gradient, subsampled_image.copy())
            .arrange(RIGHT, buff=2)
            .move_to(DOWN * 0.5)
        )

        self.play(
            subsampled_image.animate.move_to(aux_vg[1]),
            FadeIn(gradient, shift=LEFT),
        )
        original_text = (
            Text("Original image", font="CMU Serif", weight=MEDIUM)
            .scale(0.5)
            .next_to(gradient, DOWN, buff=0.4)
        )
        subsampled_text = (
            Text("Subsampled Image", font="CMU Serif", weight=MEDIUM)
            .scale(0.5)
            .next_to(subsampled_image, DOWN, buff=0.4)
        )

        self.play(FadeIn(original_text), FadeIn(subsampled_text))

        self.wait(4)

    def show_real_world_image_subsampled(self):
        shed = ImageMobject("rose.jpg")

        shed_arr = shed.get_pixel_array()

        shed_subsampled_420 = ImageMobject(
            self.chroma_subsample_image(shed_arr, mode="4:2:0")
        )

        img_g = (
            Group(shed, shed_subsampled_420)
            .arrange(RIGHT)
            .scale_to_fit_width(12)
        )
        text = Tex("Original image").scale(0.6).next_to(shed, DOWN, buff=0.5)
        text_420 = (
            Tex("Subsampling 4:2:0")
            .scale(0.6)
            .next_to(shed_subsampled_420, DOWN, buff=0.5)
        )

        self.play(
            LaggedStartMap(
                FadeIn,
                img_g,
            ),
            LaggedStart(
                FadeIn(text), FadeIn(text_420), lag_ratio=0.4
            ),
        )
        self.wait()

    def show_file_size_calculation(self):
        gradient_image = ImageMobject("r.png")
        gradient_image.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
        gradient_image.scale(30)

        pix_array = gradient_image.get_pixel_array()

        gradient = PixelArray(pix_array[:, :, :-1]).scale(0.3)

        y, u, v = self.get_yuv_image_from_rgb(pix_array, mapped=True)
        y_channel = PixelArray(y[:, :, 0], color_mode="GRAY").scale(0.5)
        u_channel = PixelArray(u, color_mode="RGB").scale(0.5)
        v_channel = PixelArray(v, color_mode="RGB").scale(0.5)

        y_t = Text("Y", font="SF Mono", weight=BOLD).scale(1.5).set_color(GRAY_A)
        u_t = Text("Cb", font="SF Mono", weight=BOLD).scale(1.5)
        v_t = Text("Cr", font="SF Mono", weight=BOLD).scale(1.5)

        new_v_channel = VGroup()
        for j in range(0, pix_array.shape[1] * 2, 4):
            for i in range(0, pix_array.shape[0], 2):
                sq_ul = v_channel[i + j * 4].color

                new_pixel = Pixel(
                    hex_to_rgb(sq_ul) * 255, color_mode="RGB"
                ).scale_to_fit_width(v_channel[0:2].width)

                new_v_channel.add(new_pixel)

        new_v_channel.arrange_in_grid(rows=4, cols=4, buff=0)

        new_u_channel = VGroup()
        for j in range(0, pix_array.shape[1] * 2, 4):
            for i in range(0, pix_array.shape[0], 2):
                sq_ul = u_channel[i + j * 4].color

                new_pixel = Pixel(
                    hex_to_rgb(sq_ul) * 255, color_mode="RGB"
                ).scale_to_fit_width(u_channel[0:2].width)

                new_u_channel.add(new_pixel)

        new_u_channel.arrange_in_grid(rows=4, cols=4, buff=0)

        y_vg = VGroup(y_channel, y_t).arrange(UP, buff=0.5)
        u_vg = VGroup(new_u_channel, u_t).arrange(UP, buff=0.5)
        v_vg = VGroup(new_v_channel, v_t).arrange(UP, buff=0.5)

        channels_vg = (
            VGroup(y_vg, u_vg, v_vg)
            .arrange(RIGHT, buff=2.6)
            .scale(0.4)
            .to_edge(UP, buff=0.6)
            .shift(RIGHT * 1)
        )

        self.play(FadeIn(channels_vg))

        pixel_count_y = (
            Text("64/64", font="SF Mono", weight=MEDIUM).scale(0.5).next_to(y_channel, DOWN, buff=0.5)
        )
        pixel_count_u = (
            Text("16/64", font="SF Mono", weight=MEDIUM)
            .scale(0.5)
            .next_to(new_u_channel, DOWN, buff=0.5)
        )
        pixel_count_v = (
            Text("16/64", font="SF Mono", weight=MEDIUM)
            .scale(0.5)
            .next_to(new_v_channel, DOWN, buff=0.5)
        )
        pixel_count = (
            Text("Pixel count", font="CMU Serif", weight=BOLD)
            .scale(0.4)
            .to_edge(LEFT, buff=1.6)
            .move_to(pixel_count_v, coor_mask=[0, 1, 0])
        )

        self.play(
            FadeIn(pixel_count),
            FadeIn(pixel_count_y, shift=DOWN),
            FadeIn(pixel_count_u, shift=DOWN),
            FadeIn(pixel_count_v, shift=DOWN),
        )
        self.wait(2)

        pixel_ratio_y = (
            Text("100%", font="SF Mono", weight=MEDIUM)
            .scale(0.5)
            .next_to(pixel_count_y, DOWN, buff=0.5)
        )
        pixel_ratio_u = (
            Text("25%", font="SF Mono", weight=MEDIUM)
            .scale(0.5)
            .next_to(pixel_count_u, DOWN, buff=0.5)
        )
        pixel_ratio_v = (
            Text("25%", font="SF Mono", weight=MEDIUM)
            .scale(0.5)
            .next_to(pixel_count_v, DOWN, buff=0.5)
        )
        pixel_ratio = (
            Text("Pixel ratio", font="CMU Serif", weight=BOLD)
            .scale(0.4)
            .to_edge(LEFT, buff=1.6)
            .move_to(pixel_ratio_v, coor_mask=[0, 1, 0])
        )

        self.play(
            FadeIn(pixel_ratio),
            FadeIn(pixel_ratio_y, shift=DOWN),
            FadeIn(pixel_ratio_u, shift=DOWN),
            FadeIn(pixel_ratio_v, shift=DOWN),
        )

        self.wait()

        fraction_y = (
            Text("1/3", font="SF Mono", weight=MEDIUM)
            .scale(0.5)
            .next_to(pixel_ratio_y, DOWN, buff=0.5)
        )
        fraction_u = (
            Text("1/3 · 1/4", font="SF Mono", weight=MEDIUM)
            .scale(0.4)
            .next_to(pixel_ratio_u, DOWN, buff=0.5)
        )
        fraction_v = (
            Text("1/3 · 1/4", font="SF Mono", weight=MEDIUM)
            .scale(0.4)
            .next_to(pixel_ratio_v, DOWN, buff=0.5)
        )
        fraction_image = (
            Text(
                "Fraction of \nthe image",
                font="CMU Serif",
                weight=BOLD,
                should_center=False,
            )
            .scale(0.4)
            .to_edge(LEFT, buff=1.6)
            .move_to(fraction_u, coor_mask=[0, 1, 0])
        )

        self.play(
            FadeIn(fraction_image),
            FadeIn(fraction_y, shift=DOWN),
            FadeIn(fraction_u, shift=DOWN),
            FadeIn(fraction_v, shift=DOWN),
        )

        self.wait()

        one_over_twelve = Text("1/12", font="SF Mono", weight=MEDIUM).scale(0.5)

        self.play(
            Transform(fraction_u, one_over_twelve.copy().move_to(fraction_u)),
            Transform(fraction_v, one_over_twelve.copy().move_to(fraction_v)),
        )

        self.wait()

        total_sum = (
            Text("6/12", font="SF Mono", weight=MEDIUM).scale(0.8).next_to(fraction_u, DOWN, buff=1)
        )
        total_sum_ratio = (
            Text("50%", font="SF Mono", weight=MEDIUM).scale(0.8).next_to(fraction_u, DOWN, buff=2)
        )

        self.wait()

        total_size = (
            Text(
                "Final total size \ncompared to original",
                font="CMU Serif",
                weight=BOLD,
                should_center=False,
            )
            .scale(0.5)
            .to_edge(LEFT, buff=1.6)
            .move_to(
                VGroup(total_sum, total_sum_ratio).get_center(), coor_mask=[0, 1, 0]
            )
            .shift(DOWN * 0.15)
        )

        self.play(
            FadeIn(total_size),
            TransformFromCopy(fraction_y, total_sum.copy()),
            TransformFromCopy(fraction_u, total_sum.copy()),
            TransformFromCopy(fraction_v, total_sum.copy()),
        )

        self.wait()

        self.play(Transform(total_sum, total_sum_ratio))


class TestGrayScaleImages(ImageUtils):
    def construct(self):
        pixel_array = np.uint8(
            [[63, 0, 0, 0], [0, 127, 0, 0], [0, 0, 191, 0], [0, 0, 0, 255]]
        )
        image = self.get_image_mob(pixel_array, height=2)

        self.add(image)
        self.wait()

        new_pixel_array = image.get_pixel_array()
        print(new_pixel_array.shape)
        new_pixel_array[3][1] = 255
        new_image = self.get_image_mob(new_pixel_array, height=2)
        self.remove(image)
        self.wait()
        self.add(new_image)
        self.wait()
        next_image_pixel_array = new_image.get_pixel_array()
        next_image_pixel_array[1, 3, :] = [255, 255, 255, 255]
        next_image = self.get_image_mob(next_image_pixel_array, height=2)
        self.remove(new_image)
        self.wait()
        self.add(next_image)
        self.wait()


class TestColorImage(ImageUtils):
    def construct(self):
        NUM_PIXELS = 32
        HEIGHT = 4
        new_image = self.down_sample_image(
            "duck", NUM_PIXELS, NUM_PIXELS, image_height=HEIGHT
        )

        next_image_pixel_array = new_image.get_pixel_array()
        next_image_pixel_array[-1, 0, :] = [255, 255, 255, 255]
        next_image = self.get_image_mob(next_image_pixel_array, height=HEIGHT)
        self.add(next_image)
        self.wait()


class TestYCbCrImages(ImageUtils):
    def construct(self):
        original_image = ImageMobject("shed")
        y, u, v = self.get_yuv_image_from_rgb(original_image.get_pixel_array())
        y_channel = ImageMobject(y)
        u_channel = ImageMobject(u)
        v_channel = ImageMobject(v)

        original_image.move_to(LEFT * 2)
        y_channel.move_to(RIGHT * 2 + UP * 2)
        u_channel.move_to(RIGHT * 2 + UP * 0)
        v_channel.move_to(RIGHT * 2 + DOWN * 2)

        self.add(original_image)
        self.wait()

        self.add(y_channel, u_channel, v_channel)
        self.wait()


class TestYCbCrImagesDuck(ImageUtils):
    def construct(self):
        NUM_PIXELS = 32
        HEIGHT = 2
        new_image = self.down_sample_image(
            "duck", NUM_PIXELS, NUM_PIXELS, image_height=HEIGHT
        )
        y, u, v = self.get_yuv_image_from_rgb(new_image.get_pixel_array())
        y_channel = self.get_image_mob(y, height=HEIGHT)
        u_channel = self.get_image_mob(u, height=HEIGHT)
        v_channel = self.get_image_mob(v, height=HEIGHT)

        new_image.move_to(LEFT * 2)
        y_channel.move_to(RIGHT * 2 + UP * 2)
        u_channel.move_to(RIGHT * 2 + UP * 0)
        v_channel.move_to(RIGHT * 2 + DOWN * 2)

        self.add(new_image)
        self.wait()

        self.add(y_channel, u_channel, v_channel)
        self.wait()


# Animation for representing duck image as signal
class ImageToSignal(ImageUtils):
    NUM_PIXELS = 32
    HEIGHT = 3

    def construct(self):
        image_mob = self.down_sample_image(
            "duck",
            ImageToSignal.NUM_PIXELS,
            ImageToSignal.NUM_PIXELS,
            image_height=ImageToSignal.HEIGHT,
        )
        gray_scale_image_mob = self.get_gray_scale_image(
            image_mob, height=ImageToSignal.HEIGHT
        )
        self.play(FadeIn(gray_scale_image_mob))
        self.wait()

        pixel_grid = self.add_grid(gray_scale_image_mob)

        axes = self.get_axis()

        pixel_row_mob, row_values = self.pick_out_row_of_image(
            gray_scale_image_mob, pixel_grid, 16
        )

        self.play(Write(axes))
        self.wait()

        self.plot_row_values(axes, row_values)

    def get_gray_scale_image(self, image_mob, height=4):
        """
        @param: image_mob -- Mobject.ImageMobject representation of image
        @return: Mobject.ImageMobject of Y (brightness) channel from YCbCr representation
        (equivalent to gray scale)
        """
        y, u, v = self.get_yuv_image_from_rgb(image_mob.get_pixel_array())
        y_channel = self.get_image_mob(y, height=height)
        y_channel.move_to(UP * 2)
        return y_channel

    def add_grid(self, image_mob):
        pixel_grid = self.get_pixel_grid(image_mob, ImageToSignal.NUM_PIXELS)
        pixel_grid.move_to(image_mob.get_center())
        self.play(FadeIn(pixel_grid))
        self.wait()

        return pixel_grid

    def get_axis(self):
        ax = Axes(
            x_range=[0, ImageToSignal.NUM_PIXELS, 1],
            y_range=[0, 255, 1],
            y_length=2.7,
            x_length=10,
            tips=False,
            axis_config={"include_numbers": True, "include_ticks": False},
            x_axis_config={
                "numbers_to_exclude": list(range(1, ImageToSignal.NUM_PIXELS + 1))
            },
            y_axis_config={"numbers_to_exclude": list(range(1, 255))},
        ).move_to(DOWN * 2.2)
        return ax

    def pick_out_row_of_image(self, image_mob, pixel_grid, row):
        pixel_array = image_mob.get_pixel_array()
        pixel_row_mob, row_values = self.get_pixel_row_mob(pixel_array, row)
        pixel_row_mob.next_to(image_mob, DOWN)
        surround_rect = SurroundingRectangle(
            pixel_grid[
                row * ImageToSignal.NUM_PIXELS : row * ImageToSignal.NUM_PIXELS
                + ImageToSignal.NUM_PIXELS
            ],
            buff=0,
        ).set_stroke(width=2, color=PURE_GREEN)
        self.play(Create(surround_rect))
        self.wait()

        self.play(FadeIn(pixel_row_mob))
        self.wait()
        return pixel_row_mob, row_values

    def get_pixel_row_mob(self, pixel_array, row, height=SMALL_BUFF * 3):
        row_values = [pixel_array[row][i][0] for i in range(ImageToSignal.NUM_PIXELS)]
        pixel_row_mob = VGroup(
            *[
                Square(side_length=height)
                .set_stroke(width=1, color=PURE_GREEN)
                .set_fill(color=gray_scale_value_to_hex(value), opacity=1)
                for value in row_values
            ]
        ).arrange(RIGHT, buff=0)
        return pixel_row_mob, row_values

    def plot_row_values(self, axes, pixel_row_values):
        pixel_coordinates = list(enumerate(pixel_row_values))
        axes.add_coordinates()
        path = VGroup()
        path_points = [axes.coords_to_point(x, y) for x, y in pixel_coordinates]
        path.set_points_smoothly(*[path_points]).set_color(YELLOW)
        dots = VGroup(
            *[
                Dot(axes.coords_to_point(x, y), radius=SMALL_BUFF / 2, color=YELLOW)
                for x, y in pixel_coordinates
            ]
        )
        self.play(LaggedStartMap(GrowFromCenter, dots), run_time=3)
        self.wait()
        self.play(Create(path), run_time=4)
        self.wait()

# This class handles animating any row of pixels from an image into a signal
# Handling this differently since in general, pixel counts per row will be high
class GeneralImageToSignal(ImageToSignal):
    def construct(self):
        image_mob = ImageMobject("dog").move_to(UP * 2)
        pixel_array = image_mob.get_pixel_array()
        ImageToSignal.NUM_PIXELS = pixel_array.shape[1]

        ROW = 140
        self.play(FadeIn(image_mob))
        self.wait()
        highlight_row = self.highlight_row(ROW, image_mob)

        row_mob, row_values = self.get_pixel_row_mob(pixel_array, ROW)

        row_mob.next_to(image_mob, DOWN)

        surround_rect = self.show_highlight_to_surround_rect(highlight_row, row_mob)

        axes = self.get_axis()
        self.play(Write(axes))
        self.wait()

        self.show_signal(axes, row_values, row_mob)

    def get_pixel_row_mob(self, pixel_array, row, height=SMALL_BUFF * 3, row_length=13):
        row_values = [pixel_array[row][i][0] for i in range(ImageToSignal.NUM_PIXELS)]
        pixel_row_mob = VGroup(
            *[
                Rectangle(height=height, width=row_length / ImageToSignal.NUM_PIXELS)
                .set_stroke(color=gray_scale_value_to_hex(value))
                .set_fill(color=gray_scale_value_to_hex(value), opacity=1)
                for value in row_values
            ]
        ).arrange(RIGHT, buff=0)
        return pixel_row_mob, row_values

    # draws a line indicating row of image_mob we are highlight
    def highlight_row(self, row, image_mob):
        vertical_pos = (
            image_mob.get_top()
            + DOWN * row / image_mob.get_pixel_array().shape[0] * image_mob.height
        )
        left_bound = vertical_pos + LEFT * image_mob.width / 2
        right_bound = vertical_pos + RIGHT * image_mob.width / 2
        line = Line(left_bound, right_bound).set_color(REDUCIBLE_GREEN_LIGHTER).set_stroke(width=1)
        self.play(Create(line))
        self.wait()
        return line

    def show_highlight_to_surround_rect(self, highlight_row, row_mob):
        surround_rect = SurroundingRectangle(row_mob, buff=0).set_color(
            highlight_row.get_color()
        )
        self.play(
            FadeIn(row_mob), TransformFromCopy(highlight_row, surround_rect), run_time=2
        )
        self.wait()
        return surround_rect

    def show_signal(self, axes, pixel_row_values, pixel_row_mob):
        pixel_coordinates = list(enumerate(pixel_row_values))
        axes.add_coordinates()
        path = VGroup()
        path_points = [axes.coords_to_point(x, y) for x, y in pixel_coordinates]
        path.set_points_smoothly(*[path_points]).set_color(YELLOW)

        arrow = (
            Arrow(DOWN * 0.5, UP * 0.5)
            .set_color(YELLOW)
            .next_to(pixel_row_mob, DOWN, aligned_edge=LEFT, buff=SMALL_BUFF)
        )
        self.play(Write(arrow))
        self.wait()
        new_arrow = arrow.copy().next_to(
            pixel_row_mob, DOWN, aligned_edge=RIGHT, buff=SMALL_BUFF
        )
        self.play(
            Transform(arrow, new_arrow),
            Create(path),
            run_time=8,
            rate_func=linear,
        )
        self.wait()

class DCTExperiments(ImageUtils):
    def construct(self):
        image_mob = ImageMobject("dog").move_to(UP * 2)
        self.play(FadeIn(image_mob))
        self.wait()

        print("Image size:", image_mob.get_pixel_array().shape)

        self.perform_JPEG(image_mob)

    def perform_JPEG(self, image_mob):
        # Performs encoding/decoding steps on a gray scale block
        block_image, pixel_grid, block = self.highlight_pixel_block(image_mob, 125, 125)
        print("Before\n", block[:, :, 1])
        block_centered = format_block(block)
        print("After centering\n", block_centered)

        dct_block = dct_2d(block_centered)
        np.set_printoptions(suppress=True)
        print("DCT block (rounded)\n", np.round(dct_block, decimals=1))

        heat_map = self.get_heat_map(block_image, dct_block)
        heat_map.move_to(block_image.get_center() + RIGHT * 6)
        self.play(FadeIn(heat_map))
        self.wait()

        self.play(FadeOut(heat_map))
        self.wait()

        quantized_block = quantize(dct_block)
        print("After quantization\n", quantized_block)

        dequantized_block = dequantize(quantized_block)
        print("After dequantize\n", dequantized_block)

        invert_dct_block = idct_2d(dequantized_block)
        print("Invert DCT block\n", invert_dct_block)

        compressed_block = invert_format_block(invert_dct_block)
        print("After reformat\n", compressed_block)

        print("MSE\n", np.mean((compressed_block - block[:, :, 1]) ** 2))

        final_image = self.get_image_mob(compressed_block, height=2)
        final_image.move_to(block_image.get_center() + RIGHT * 6)

        final_image_grid = self.get_pixel_grid(final_image, 8).move_to(
            final_image.get_center()
        )
        self.play(FadeIn(final_image), FadeIn(final_image_grid))
        self.wait()

        # self.get_dct_component(0, 0)

    def highlight_pixel_block(self, image_mob, start_row, start_col, block_size=8):
        pixel_array = image_mob.get_pixel_array()
        block = pixel_array[
            start_row : start_row + block_size, start_col : start_col + block_size
        ]
        center_row = start_row + block_size // 2
        center_col = start_col + block_size // 2
        vertical_pos = (
            image_mob.get_top()
            + DOWN * center_row / pixel_array.shape[0] * image_mob.height
        )
        horizontal_pos = (
            image_mob.get_left()
            + RIGHT * center_col / pixel_array.shape[1] * image_mob.width
        )
        tiny_square_highlight = Square(side_length=SMALL_BUFF * 0.8)
        highlight_position = np.array([horizontal_pos[0], vertical_pos[1], 0])
        tiny_square_highlight.set_color(REDUCIBLE_YELLOW).move_to(highlight_position)

        self.play(Create(tiny_square_highlight))
        self.wait()

        block_position = DOWN * 2
        block_image = self.get_image_mob(block, height=2).move_to(block_position)
        pixel_grid = self.get_pixel_grid(block_image, block_size).move_to(
            block_position
        )
        surround_rect = SurroundingRectangle(pixel_grid, buff=0).set_color(
            REDUCIBLE_YELLOW
        )
        self.play(
            FadeIn(block_image),
            FadeIn(pixel_grid),
            TransformFromCopy(tiny_square_highlight, surround_rect),
        )
        self.wait()

        self.play(
            FadeOut(surround_rect),
            FadeOut(tiny_square_highlight),
            block_image.animate.shift(LEFT * 3),
            pixel_grid.animate.shift(LEFT * 3),
        )
        self.wait()

        return block_image, pixel_grid, block

    def get_heat_map(self, block_image, dct_block):
        block_size = dct_block.shape[0]
        pixel_grid_dct = self.get_pixel_grid(block_image, block_size)
        dct_block_abs = np.abs(dct_block)
        max_dct_coeff = np.amax(dct_block_abs)
        max_color = REDUCIBLE_YELLOW
        min_color = REDUCIBLE_PURPLE
        for i, square in enumerate(pixel_grid_dct):
            row, col = i // block_size, i % block_size
            alpha = dct_block_abs[row][col] / max_dct_coeff
            square.set_fill(
                color=interpolate_color(min_color, max_color, alpha), opacity=1
            )

        scale = Line(pixel_grid_dct.get_top(), pixel_grid_dct.get_bottom())
        scale.set_stroke(width=10).set_color(color=[min_color, max_color])
        integer_scale = 0.5
        top_value = Integer(round(max_dct_coeff)).scale(integer_scale)
        top_value.next_to(scale, RIGHT, aligned_edge=UP)
        bottom_value = Integer(0).scale(integer_scale)
        bottom_value.next_to(scale, RIGHT, aligned_edge=DOWN)

        heat_map_scale = VGroup(scale, top_value, bottom_value)

        return VGroup(pixel_grid_dct, heat_map_scale).arrange(RIGHT)

class DCTComponents(ImageUtils):
    def construct(self):
        image_mob = ImageMobject("dog").move_to(UP * 2)
        block_image, pixel_grid, block = self.get_pixel_block(image_mob, 125, 125)
        print("Before\n", block[:, :, 1])
        block_image.move_to(LEFT * 2 + DOWN * 2)
        pixel_grid.move_to(block_image.get_center())
        block_centered = format_block(block)
        print("After centering\n", block_centered)

        dct_block = dct_2d(block_centered)
        np.set_printoptions(suppress=True)
        print("DCT block (rounded)\n", np.round(dct_block, decimals=1))

        self.play(FadeIn(image_mob))
        self.wait()

        self.play(
            FadeIn(block_image),
            FadeIn(pixel_grid),
        )
        self.wait()
        num_components = 40
        partial_block = self.get_partial_block(dct_block, num_components)
        print(f"Partial block - {num_components} components\n", partial_block)

        partial_block_image = self.get_image_mob(partial_block, height=2)
        partial_pixel_grid = self.get_pixel_grid(
            partial_block_image, partial_block.shape[0]
        )

        partial_block_image.move_to(RIGHT * 2 + DOWN * 2)
        partial_pixel_grid.move_to(partial_block_image.get_center())

        self.play(
            FadeIn(partial_block_image),
            FadeIn(partial_pixel_grid),
        )
        self.wait()

    def build_final_image_component_wise(self, dct_block):
        original_block = np.zeros((8, 8))

    def get_dct_component(self, row, col):
        text = Tex(f"({row}, {col})").move_to(UP * 2)
        self.add(text)
        dct_matrix = np.zeros((8, 8))
        if row == 0 and col == 0:
            dct_matrix[row][col] = 1016
        else:
            dct_matrix[row][col] = 500
        pixel_array = idct_2d(dct_matrix) + 128
        all_in_range = (pixel_array >= 0) & (pixel_array <= 255)
        if not all(all_in_range.flatten()):
            print("Bad array\n", pixel_array)
            raise ValueError("All elements in pixel_array must be in range [0, 255]")

        image_mob = self.get_image_mob(pixel_array, height=2)
        pixel_grid = self.get_pixel_grid(image_mob, pixel_array.shape[0])
        self.wait()
        self.play(
            FadeIn(image_mob),
        )
        self.add(pixel_grid)
        self.wait()

        self.remove(image_mob, pixel_grid, text)

    def get_partial_block(self, dct_block, num_components):
        zigzag = get_zigzag_order()
        dct_matrix = np.zeros((8, 8))
        for basis_comp in range(num_components):
            row, col = zigzag[basis_comp]
            dct_matrix[row][col] = dct_block[row][col]

        pixel_array = idct_2d(dct_matrix)
        return invert_format_block(pixel_array)

    def get_pixel_block(self, image_mob, start_row, start_col, block_size=8):
        pixel_array = image_mob.get_pixel_array()
        block = pixel_array[
            start_row : start_row + block_size, start_col : start_col + block_size
        ]

        block_image = self.get_image_mob(block, height=2)
        pixel_grid = self.get_pixel_grid(block_image, block_size)

        return block_image, pixel_grid, block

    def display_component(self, dct_matrix, row, col):
        pass

class DCTSliderExperiments(DCTComponents):
    def construct(self):
        image_mob = ImageMobject("dog").move_to(UP * 2)
        block_image, pixel_grid, block = self.get_pixel_block(image_mob, 125, 125)
        print("Before\n", block[:, :, 1])
        block_image.move_to(LEFT * 2 + DOWN * 0.5)
        pixel_grid.move_to(block_image.get_center())
        block_centered = format_block(block)
        print("After centering\n", block_centered)

        dct_block = dct_2d(block_centered)
        np.set_printoptions(suppress=True)
        print("DCT block (rounded)\n", np.round(dct_block, decimals=1))

        self.play(
            FadeIn(image_mob),
        )
        self.wait()

        self.play(
            FadeIn(block_image),
            FadeIn(pixel_grid),
        )
        self.wait()

        number_line = NumberLine(
            x_range=[0, 64, 8],
            length=10,
            color=REDUCIBLE_VIOLET,
            include_numbers=True,
            label_direction=UP,
        ).move_to(DOWN * 2.5)

        self.add(number_line)
        self.wait()

        self.animate_slider(number_line, dct_block, block)

    def animate_slider(self, number_line, dct_block, original_block):
        # count = 1
        # def update_image(tick, dt):
        #     print(dt)
        #     nonlocal count
        #     print(count)
        #     count += 1
        #     print(tick.get_center())
        #     num_components = number_line.point_to_number(tick.get_center())
        #     new_partial_block = self.get_partial_block(dct_block, num_components)
        #     print(f'Partial block - {num_components} components\n')

        #     new_partial_block_image = self.get_image_mob(new_partial_block, height=2)
        #     new_partial_block_image.move_to(image_pos)
        #     partial_block_image.become(new_partial_block_image)

        tick = Triangle().scale(0.2).set_color(REDUCIBLE_YELLOW)
        tick.set_fill(color=REDUCIBLE_YELLOW, opacity=1)

        tracker = ValueTracker(0)
        tick.add_updater(
            lambda m: m.next_to(number_line.n2p(tracker.get_value()), DOWN)
        )
        self.play(
            FadeIn(tick),
        )
        self.wait()
        image_pos = RIGHT * 2 + DOWN * 0.5

        def get_new_block():
            new_partial_block = self.get_partial_block(dct_block, tracker.get_value())
            print(f"Partial block - {tracker.get_value()} components")
            print(
                "MSE", np.mean((new_partial_block - original_block[:, :, 1]) ** 2), "\n"
            )

            new_partial_block_image = self.get_image_mob(new_partial_block, height=2)
            new_partial_block_image.move_to(image_pos)
            return new_partial_block_image

        partial_block = self.get_partial_block(dct_block, tracker.get_value())
        partial_block_image = always_redraw(get_new_block)
        partial_pixel_grid = self.get_pixel_grid(
            partial_block_image, partial_block.shape[0]
        )
        partial_pixel_grid.move_to(image_pos)
        self.play(
            FadeIn(partial_block_image),
            FadeIn(partial_pixel_grid),
        )
        self.add_foreground_mobject(partial_pixel_grid)
        self.wait()

        self.play(
            tracker.animate.set_value(64),
            run_time=10,
            rate_func=linear,
        ),

        self.wait()

    def get_partial_block(self, dct_block, num_components):
        """
        @param: dct_block - dct coefficients of the block
        @param: num_components - float - number of dct components to
        include in partial block
        @return: pixel_array of partial block with num_components of DCT included
        """
        from math import floor

        zigzag = get_zigzag_order()
        dct_matrix = np.zeros((8, 8))
        floor_val = floor(num_components)
        remaining = num_components - floor_val
        for basis_comp in range(floor_val):
            row, col = zigzag[basis_comp]
            dct_matrix[row][col] = dct_block[row][col]

        if floor_val < dct_block.shape[0] ** 2:
            row, col = zigzag[floor_val]
            dct_matrix[row][col] = remaining * dct_block[row][col]

        pixel_array = idct_2d(dct_matrix)
        return invert_format_block(pixel_array)

class DCTEntireImageSlider(DCTSliderExperiments):
    def construct(self):
        image_mob = ImageMobject("dog").move_to(UP * 1)
        

        number_line = self.initialize_slider()
        dct_components = Tex("DCT Components").next_to(number_line, UP * 2)

        self.play(
            FadeIn(image_mob),
            FadeIn(number_line),
            Write(dct_components)
        )
        # print(new_image.get_pixel_array().shape)
        self.wait()

        self.animate_slider(image_mob, number_line)


    def get_all_blocks(
        self,
        image_mob,
        start_row,
        end_row,
        start_col,
        end_col,
        num_components,
        block_size=8,
    ):
        pixel_array = image_mob.get_pixel_array()
        new_pixel_array = np.zeros((pixel_array.shape[0], pixel_array.shape[1]))
        for i in range(start_row, end_row, block_size):
            for j in range(start_col, end_col, block_size):
                pixel_block = self.get_pixel_block(pixel_array, i, j)
                block_centered = format_block(pixel_block)
                dct_block = dct_2d(block_centered)
                # quantized_block = quantize(dct_block)
                # dequantized_block = dequantize(quantized_block)
                # invert_dct_block = idct_2d(dequantized_block)
                # compressed_block = invert_format_block(invert_dct_block)
                # all_in_range = (compressed_block >= 0) & (compressed_block <= 255)
                # if not all(all_in_range.flatten()):
                #     print(i, j)
                #     print(all_in_range)
                #     print('Bad array\n', compressed_block)
                #     print('Original array\n', pixel_block[:, :, 0])
                #     raise ValueError("All elements in compressed_block must be in range [0, 255]")
                # new_pixel_array[i:i+block_size, j:j+block_size] = compressed_block
                partial_block = self.get_partial_block(dct_block, num_components)
                new_pixel_array[i : i + block_size, j : j + block_size] = partial_block

        return new_pixel_array

    def get_pixel_block(self, pixel_array, start_row, start_col, block_size=8):
        return pixel_array[
            start_row : start_row + block_size, start_col : start_col + block_size
        ]

    def initialize_slider(self):
        number_line = NumberLine(
            x_range=[0, 64, 4],
            length=8,
            color=REDUCIBLE_VIOLET,
            include_numbers=True,
            label_direction=UP,
            font_size=24,
        )
        number_line.move_to(DOWN * 2)

        return number_line

    def animate_slider(self, image_mob, number_line):
        original_pixel_array = image_mob.get_pixel_array()[2:298, 3:331, 0]
        component_tracker = ValueTracker(0)
        
        def get_new_image():
            new_pixel_array = self.get_all_blocks(image_mob, 2, 298, 3, 331, component_tracker.get_value())
            relevant_section = new_pixel_array[2:298, 3:331]
            new_image = self.get_image_mob(new_pixel_array, height=None).move_to(UP * 1 + RIGHT * 2)
            return new_image

        tick = Triangle().scale(0.1).set_color(REDUCIBLE_YELLOW)
        tick.set_fill(color=REDUCIBLE_YELLOW, opacity=1)

        tick.add_updater(
            lambda m: m.next_to(number_line.n2p(component_tracker.get_value()), DOWN)
        )

        new_image = always_redraw(get_new_image)
        self.play(
            FadeIn(tick),
            image_mob.animate.shift(LEFT * 2),
            FadeIn(new_image),
        )
        self.wait()
    

        self.play(
            component_tracker.animate.set_value(64),
            run_time=32,
            rate_func=linear,
        ),

        # self.wait()

class DCT1DExperiments(DCTComponents):
    def construct(self):
        image_mob = ImageMobject("dog").move_to(UP * 2)
        block_image, pixel_grid, block = self.get_pixel_block(image_mob, 125, 125)
        block_image.shift(UP * 2)
        self.play(
            FadeIn(block_image),
        )
        self.wait()
        row = 7
        print(block[:, :, 0])
        print(f"Block row: {row}\n", block[:, :, 0][row])
        pixel_row_mob, row_values = self.get_pixel_row_mob(block, row)
        print("Selected row values\n", row_values)
        pixel_row_mob.next_to(block_image, DOWN)
        self.play(
            FadeIn(pixel_row_mob),
        )
        self.wait()

        row_values_centered = format_block(row_values)
        print("After centering\n", row_values_centered)

        dct_row_pixels = dct_1d(row_values_centered)
        np.set_printoptions(suppress=True)
        print("DCT block (rounded)\n", np.round(dct_row_pixels, decimals=1))

        inverted_row = idct_1d(dct_row_pixels) + 128
        self.play(
            FadeOut(block_image),
            FadeOut(pixel_row_mob),
        )
        self.wait()
        print("Inverted row:\n", inverted_row)
        num_pixels = dct_row_pixels.shape[0]
        height_pixel = 0.6
        for col in range(num_pixels):
            text = Tex(str(col)).move_to(UP * 2)
            new_value = 250
            basis_component, dct_row = self.get_dct_component(col, new_value=new_value)
            print(f"Basis value for {col}\n", basis_component)

            component_mob = self.make_row_of_pixels(
                basis_component, height=height_pixel
            ).shift(RIGHT * 0.25)
            ax, graph = self.get_graph_dct_component(col)

            self.add(text, component_mob, ax, graph)
            self.wait(2)

            self.remove(text, component_mob, ax, graph)

        self.wait()

        self.play(
            FadeIn(block_image),
            FadeIn(pixel_row_mob),
        )
        self.wait()

        self.draw_image_graph(dct_row_pixels)

    def get_pixel_row_mob(self, pixel_array, row, height=SMALL_BUFF * 5, num_pixels=8):
        row_length = height * num_pixels
        row_values = [pixel_array[row][i][0] for i in range(num_pixels)]
        pixel_row_mob = VGroup(
            *[
                Rectangle(height=height, width=row_length / num_pixels)
                .set_stroke(color=REDUCIBLE_GREEN_LIGHTER)
                .set_fill(color=gray_scale_value_to_hex(value), opacity=1)
                for value in row_values
            ]
        ).arrange(RIGHT, buff=0)
        return pixel_row_mob, np.array(row_values)

    def make_row_of_pixels(self, row_values, height=SMALL_BUFF * 5, num_pixels=8):
        row_length = height * num_pixels
        adjusted_row_values = []
        for val in row_values:
            adjusted_row_values.append(int(round(val)))
        pixel_row_mob = VGroup(
            *[
                Rectangle(height=height, width=row_length / num_pixels)
                .set_stroke(color=REDUCIBLE_GREEN_LIGHTER)
                .set_fill(color=gray_scale_value_to_hex(value), opacity=1)
                for value in adjusted_row_values
            ]
        ).arrange(RIGHT, buff=0)
        return pixel_row_mob

    def get_dct_component(self, col, new_value=250, num_pixels=8):
        dct_row = np.zeros((num_pixels,))
        dct_row[col] = new_value
        return idct_1d(dct_row) + 128, dct_row

    def get_graph_dct_component(self, col, num_pixels=8):
        ax = Axes(
            x_range=[0, num_pixels - 1, 1],
            y_range=[-1, 1],
            y_length=2,
            x_length=4.2,
            tips=False,
            axis_config={"include_numbers": True, "include_ticks": True},
            x_axis_config={
                "numbers_to_exclude": list(range(1, num_pixels - 1)),
            },
            # y_axis_config={"numbers_to_exclude": list(range(1, 255))},
        ).move_to(DOWN * 2)
        func = lambda n: np.cos((2 * n + 1) * col * np.pi / (2 * num_pixels))
        if col == 0:
            func = (
                lambda n: 1
                / np.sqrt(2)
                * np.cos((2 * n + 1) * col * np.pi / (2 * num_pixels))
            )
        graph = ax.plot(func)
        graph.set_color(REDUCIBLE_YELLOW)
        return ax, graph

    def draw_image_graph(self, dct_row):
        """
        @param: 1D DCT of a row of pixels
        @return: graph composed of a linear combination of cosine functions weighted by dct_row
        """

        def get_basis_function(col):
            factor = dct_row[col] * np.sqrt(2 / dct_row.shape[0])

            def f(n):
                if col == 0:
                    return factor * 1 / np.sqrt(2)
                return factor * np.cos(
                    (2 * n + 1) * col * np.pi / (2 * dct_row.shape[0])
                )

            return f

        basis_functions = [get_basis_function(i) for i in range(dct_row.shape[0])]
        final_func = (
            lambda n: sum(basis_function(n) for basis_function in basis_functions) + 128
        )
        ax = Axes(
            x_range=[0, dct_row.shape[0] - 1, 1],
            y_range=[0, 255, 1],
            y_length=2,
            x_length=4.375,
            tips=False,
            axis_config={"include_numbers": True, "include_ticks": False},
            x_axis_config={
                "numbers_to_exclude": list(range(1, dct_row.shape[0] - 1)),
            },
            y_axis_config={"numbers_to_exclude": list(range(1, 255))},
        )
        graph = ax.plot(final_func)
        graph.set_color(REDUCIBLE_YELLOW)
        ax.add_coordinates()
        row_values = idct_1d(dct_row) + 128
        pixel_coordinates = list(enumerate(row_values))
        dots = VGroup(
            *[
                Dot()
                .scale(0.7)
                .move_to(ax.coords_to_point(x, y))
                .set_color(REDUCIBLE_YELLOW)
                for x, y in pixel_coordinates
            ]
        )

        return ax, graph, dots

class DCT1DStepsVisualized(DCT1DExperiments):
    """
    Animations we need:
    1. Take a 8 x 8 block and highlight a row of pixels
    2. Build an array of pixels from a given set of row values
    3. Given a row of pixels, draw the exact signal it represents in terms of cosine waves
    4. Show shift of pixels and signal down by 128 to center around 0
    5. Build up original signal using DCT components, show components summing together one by one
    """

    GRAPH_ADJUST = LEFT * 0.35

    def construct(self):
        image_mob = ImageMobject("dog").move_to(UP * 2)
        block_image, pixel_grid, block = self.get_pixel_block(image_mob, 125, 125)
        block_image.shift(UP * 2.5)
        self.play(FadeIn(block_image))
        self.wait()
        row = 7
        print(block[:, :, 0])
        print(f"Block row: {row}\n", block[:, :, 0][row])
        pixel_row_mob, row_values, highlight = self.highlight_pixel_row(
            block, block_image, row, height=0.625
        )
        print("Selected row values\n", row_values)
        pixel_row_mob.move_to(UP)
        self.play(
            FadeIn(highlight),
        )
        self.wait()

        self.show_highlight(pixel_row_mob, highlight)

        self.play(
            FadeOut(block_image),
        )
        self.wait()

        array_mob = self.get_array_obj(row_values)
        array_mob.next_to(pixel_row_mob, UP)
        self.play(
            FadeIn(array_mob),
        )
        self.wait()

        row_values_centered = format_block(row_values)
        print("After centering\n", row_values_centered)

        dct_row_pixels = dct_1d(row_values_centered)

        ax, graph, dots = self.draw_image_graph(dct_row_pixels)

        self.show_graph(ax, graph, dots, pixel_row_mob)

        graph_components = VGroup(ax, graph, dots)

        specific_step, rect, center_step = self.show_centering_step(
            row_values, dct_row_pixels, array_mob, pixel_row_mob, graph_components
        )

        label, apply_dct = self.prepare_to_shift(specific_step, center_step, array_mob)

        left_group = self.shift_components(
            pixel_row_mob, array_mob, graph_components, label, rect
        )

        dct_labels, dct_array_mob, dct_graph_components = self.show_dct_step(
            label, array_mob, graph_components, dct_row_pixels
        )

        self.show_how_dct_works(
            apply_dct,
            left_group,
            dct_labels,
            dct_array_mob,
            dct_graph_components,
            dct_row_pixels,
        )

    def show_dct_step(self, label, array_mob, graph_components, dct_row):
        x_label, brace = label
        new_label_x_hat = MathTex(r"\hat{X} = \text{DCT}(X)").scale(0.8)
        new_array_mob = self.get_array_obj(dct_row, color=REDUCIBLE_PURPLE)
        new_brace = Brace(new_array_mob, direction=UP)
        new_label_x_hat.next_to(new_brace, UP, buff=SMALL_BUFF)

        new_label = VGroup(new_label_x_hat, new_brace)

        dct_ax = self.get_dct_axis(dct_row, -80, 80)

        dct_graph, dct_points = self.plot_row_values(
            dct_ax, dct_row, color=REDUCIBLE_PURPLE
        )

        dct_graph_components = VGroup(dct_ax, dct_graph, dct_points)
        dct_graph_components.next_to(new_array_mob, DOWN).shift(
            DCT1DStepsVisualized.GRAPH_ADJUST
        )
        group = VGroup(new_label, new_array_mob, dct_graph_components)

        group.move_to(RIGHT * 3.5)

        self.play(
            TransformFromCopy(label, new_label),
            TransformFromCopy(array_mob, new_array_mob),
            run_time=2,
        )
        self.wait()

        self.play(Write(dct_ax))
        self.play(
            *[GrowFromCenter(dot) for dot in dct_points],
            Create(dct_graph),
        )
        self.wait()

        return new_label, new_array_mob, dct_graph_components

    def show_how_dct_works(
        self,
        label,
        left_group,
        dct_labels,
        dct_array_mob,
        dct_graph_components,
        dct_row,
    ):
        dct_group = VGroup(dct_labels, dct_array_mob)
        self.play(
            FadeOut(left_group),
            FadeOut(dct_graph_components),
            FadeOut(label),
            dct_group.animate.move_to(UP * 3),
        )
        self.wait()

        key_sum = MathTex(r"X = \sum_{k=0}^{7} X[k] \cdot C_k").scale(0.8)
        key_sum.next_to(dct_array_mob, DOWN)
        self.play(
            Write(key_sum),
        )
        self.wait()

        self.remind_individual_cosine_comp(dct_array_mob)

        self.build_up_signal(dct_array_mob, dct_row)

    def build_up_signal(self, dct_array_mob, dct_row):
        ALIGNMENT_SHIFT = LEFT * 0.4
        right_label = (
            MathTex(r"\vec{0}").move_to(RIGHT * 3.5).set_color(REDUCIBLE_YELLOW)
        )
        zero_dct = np.zeros(dct_row.shape[0])
        right_component_mob = self.make_row_of_pixels(zero_dct + 128, height=0.625)
        right_component_mob.next_to(right_label, DOWN)

        right_ax, right_graph, right_dots = self.draw_image_graph(
            zero_dct, centered=True
        )
        right_ax, right_graph, right_dots = self.show_graph(
            right_ax,
            right_graph,
            right_dots,
            right_component_mob,
            animate=False,
            alignment_shift=ALIGNMENT_SHIFT,
        )

        self.play(
            FadeIn(right_label),
            FadeIn(right_component_mob),
            FadeIn(right_ax),
            *[GrowFromCenter(dot) for dot in right_dots],
            Create(right_graph),
        )
        self.wait()

        left_text = MathTex("C_0").set_color(REDUCIBLE_YELLOW)
        left_text.move_to(LEFT * 3.5)
        basis_component, left_dct_row = self.get_dct_component(0)
        left_component_mob = self.make_row_of_pixels(
            basis_component, height=SMALL_BUFF * 6.25
        )
        left_component_mob.next_to(left_text, DOWN)
        left_ax, left_graph = self.get_graph_dct_component(0)
        VGroup(left_ax, left_graph).next_to(left_component_mob, DOWN).shift(
            ALIGNMENT_SHIFT
        )
        self.play(
            Write(left_text),
            FadeIn(left_component_mob),
            FadeIn(left_ax),
            Create(left_graph),
        )
        self.wait()

        left_graph_components = VGroup(left_ax, left_graph)
        right_graph_components = VGroup(right_ax, right_graph, right_dots)

        sub_array = dct_array_mob[0][0]
        current_highlight = Rectangle(height=sub_array.height, width=sub_array.width)
        current_highlight.move_to(sub_array.get_center()).set_color(REDUCIBLE_YELLOW)

        self.play(
            Create(current_highlight),
        )
        self.wait()

        for step in range(dct_row.shape[0]):
            self.perform_update_step(
                left_graph_components,
                right_graph_components,
                left_component_mob,
                right_component_mob,
                left_text,
                right_label,
                step,
                dct_row,
                dct_array_mob,
                current_highlight,
                alignment_shift=ALIGNMENT_SHIFT,
            )

    def perform_update_step(
        self,
        left_graph_components,
        right_graph_components,
        left_component_mob,
        right_component_mob,
        left_text,
        right_label,
        step,
        dct_row,
        dct_array_mob,
        current_highlight,
        alignment_shift=LEFT * 0.4,
    ):
        sub_array = dct_array_mob[0][: step + 1]
        highlight = Rectangle(height=sub_array.height, width=sub_array.width)
        highlight.move_to(sub_array.get_center()).set_color(REDUCIBLE_YELLOW)
        self.play(Transform(current_highlight, highlight))
        self.wait()

        left_ax, left_graph = left_graph_components
        right_ax, right_graph, right_dots = right_graph_components
        isolated_dct_row = np.zeros(dct_row.shape[0])
        isolated_dct_row[step] = dct_row[step]

        iso_right_ax, iso_graph, iso_dots = self.draw_image_graph(
            isolated_dct_row, centered=True, color=REDUCIBLE_VIOLET
        )
        iso_right_ax, iso_graph, iso_dots = self.show_graph(
            iso_right_ax,
            iso_graph,
            iso_dots,
            right_component_mob,
            animate=False,
            alignment_shift=alignment_shift,
        )
        self.align_graph_and_dots(right_dots, iso_dots, iso_graph)

        self.play(
            TransformFromCopy(left_graph, iso_graph),
        )
        intermediate_text = self.generate_intermediate_text(right_component_mob)
        self.play(
            *[GrowFromCenter(dot) for dot in iso_dots],
            Transform(right_label, intermediate_text[step]),
        )
        self.wait()

        cumulative_dct_row = self.get_partial_row_dct(dct_row, step + 1)
        cum_right_ax, cum_graph, cum_dots = self.draw_image_graph(
            cumulative_dct_row, centered=True, color=REDUCIBLE_YELLOW
        )
        cum_right_ax, cum_graph, cum_dots = self.show_graph(
            cum_right_ax,
            cum_graph,
            cum_dots,
            right_component_mob,
            animate=False,
            alignment_shift=alignment_shift,
        )

        final_text = self.generate_final_text(right_component_mob)

        self.align_graph_and_dots(right_dots, cum_dots, cum_graph)

        new_right_component_mob = self.make_row_of_pixels(
            idct_1d(cumulative_dct_row) + 128, height=0.625
        )
        new_right_component_mob.move_to(right_component_mob.get_center())

        self.play(
            Transform(right_graph, cum_graph),
            Transform(right_dots, cum_dots),
            FadeOut(iso_graph),
            FadeOut(iso_dots),
            Transform(right_component_mob, new_right_component_mob),
            Transform(right_label, final_text[step]),
        )
        self.wait()

        if step + 1 == dct_row.shape[0]:
            return

        new_left_text = MathTex(f"C_{step + 1}").set_color(REDUCIBLE_YELLOW)
        new_left_text.move_to(left_text.get_center())
        new_basis_component, new_left_dct_row = self.get_dct_component(step + 1)
        new_left_component_mob = self.make_row_of_pixels(
            new_basis_component, height=SMALL_BUFF * 6.25
        )
        new_left_component_mob.next_to(new_left_text, DOWN)
        new_left_ax, new_left_graph = self.get_graph_dct_component(step + 1)
        VGroup(new_left_ax, new_left_graph).next_to(new_left_component_mob, DOWN).shift(
            alignment_shift
        )
        self.play(
            Transform(left_text, new_left_text),
            Transform(left_component_mob, new_left_component_mob),
            Transform(left_graph, new_left_graph),
        )
        self.wait()

    def align_graph_and_dots(self, original_dots, new_dots, new_graph):
        horiz_diff_adjust = (
            original_dots[0].get_center()[0] - new_dots[0].get_center()[0]
        )
        new_graph.shift(RIGHT * horiz_diff_adjust)
        new_dots.shift(RIGHT * horiz_diff_adjust)

    def get_partial_row_dct(self, dct_row, num_components):
        new_dct_row = np.zeros(dct_row.shape[0])
        for index in range(num_components):
            new_dct_row[index] = dct_row[index]
        return new_dct_row

    def generate_intermediate_text(self, right_component_mob):
        zero = MathTex(r"\vec{0}", "+", r"X[0] \cdot C_0")
        zero[0].set_color(REDUCIBLE_YELLOW)
        zero[-1].set_color(REDUCIBLE_VIOLET)

        one = MathTex(r"X[0] \cdot C_0", "+", r"X[1] \cdot C_1")
        one[0].set_color(REDUCIBLE_YELLOW)
        one[-1].set_color(REDUCIBLE_VIOLET)

        other_representations = [
            self.get_intermediate_representation(k) for k in range(2, 8)
        ]

        all_reprs = [zero, one] + other_representations
        for represent in all_reprs:
            represent.scale(0.8).next_to(right_component_mob, UP)
        return all_reprs

    def get_intermediate_representation(self, i):
        represent = MathTex(
            r"\sum_{k=0}^{" + str(i - 1) + r"} X[k] \cdot C_k",
            "+",
            r"X[{0}] \cdot C_{0}".format(i),
        )
        represent[0].set_color(REDUCIBLE_YELLOW)
        represent[-1].set_color(REDUCIBLE_VIOLET)
        return represent

    def generate_final_text(self, right_component_mob):
        final_zero = MathTex(r"X[0] \cdot C_0").set_color(REDUCIBLE_YELLOW)
        final_six = [
            MathTex(r"\sum_{k=0}^{" + str(i) + r"} X[k] \cdot C_k").set_color(
                REDUCIBLE_YELLOW
            )
            for i in range(1, 8)
        ]

        all_reprs = [final_zero] + final_six
        for represent in all_reprs:
            represent.scale(0.8).next_to(right_component_mob, UP)
        return all_reprs

    def remind_individual_cosine_comp(self, dct_array_mob):
        ALIGNMENT_SHIFT = RIGHT * 0.25
        dct_array, dct_array_text = dct_array_mob
        highlight_box = None
        basis_component, dct_row = self.get_dct_component(0)
        component_mob = self.make_row_of_pixels(
            basis_component, height=SMALL_BUFF * 6
        ).shift(ALIGNMENT_SHIFT)
        ax, graph = self.get_graph_dct_component(0)
        text = MathTex("C_0").set_color(REDUCIBLE_YELLOW)
        text.next_to(ax, DOWN).shift(ALIGNMENT_SHIFT)
        for col, elem in enumerate(dct_array):
            new_text = MathTex(f"C_{col}").set_color(REDUCIBLE_YELLOW)
            animations = []
            basis_component, dct_row = self.get_dct_component(col)
            new_component_mob = self.make_row_of_pixels(
                basis_component, height=SMALL_BUFF * 6
            ).shift(ALIGNMENT_SHIFT)
            new_ax, new_graph = self.get_graph_dct_component(col)
            new_text.next_to(new_ax, DOWN).shift(ALIGNMENT_SHIFT)
            if not highlight_box:
                highlight_box = elem.copy().set_color(REDUCIBLE_YELLOW)
                animations.append(Create(highlight_box))
                animations.append(Write(text))
                animations.extend([FadeIn(component_mob), FadeIn(ax), FadeIn(graph)])
            else:
                animations.append(highlight_box.animate.move_to(elem.get_center()))
                animations.append(Transform(text, new_text))
                animations.extend(
                    [
                        Transform(component_mob, new_component_mob),
                        Transform(ax, new_ax),
                        Transform(graph, new_graph),
                    ],
                )

            self.play(
                *animations,
            )
            self.wait()

        self.play(
            FadeOut(highlight_box),
            FadeOut(component_mob),
            FadeOut(ax),
            FadeOut(graph),
            FadeOut(text),
        )
        self.wait()

    def show_centering_step(
        self, row_values, dct_row, array_mob, pixel_row_mob, graph_components
    ):
        ax, graph, dots = graph_components
        entire_group = VGroup(array_mob, pixel_row_mob, graph_components)
        rect = Rectangle(height=entire_group.height + 2, width=entire_group.width + 1)
        rect.set_color(REDUCIBLE_VIOLET)
        self.play(
            Create(rect),
        )
        self.wait()

        center_step = Tex("Center pixel values around 0").next_to(rect, UP)

        self.play(
            Write(center_step),
        )
        self.wait()

        specific_step = MathTex(r"[0, 255] \rightarrow [-128, 127]").scale(0.8)
        specific_step.next_to(center_step, DOWN * 2)
        self.play(
            Write(specific_step),
        )
        self.wait()

        new_values = format_block(row_values)
        array, array_values = array_mob
        self.play(
            *[
                value.animate.set_value(new_values[i]).move_to(array[i].get_center())
                for i, value in enumerate(array_values)
            ],
        )
        self.wait()

        new_ax, new_graph, new_dots = self.draw_image_graph(dct_row, centered=True)
        new_graph_components = VGroup(new_ax, new_graph, new_dots)
        new_graph_components.move_to(graph_components.get_center())
        self.play(
            Transform(ax, new_ax),
            Transform(graph, new_graph),
            Transform(dots, new_dots),
        )
        self.wait()

        return specific_step, rect, center_step

    def prepare_to_shift(self, specific_step, center_step, array_mob):
        apply_dct = Tex("Apply DCT").move_to(center_step.get_center())
        self.play(ReplacementTransform(center_step, apply_dct))
        self.wait()

        self.play(
            FadeOut(specific_step),
        )

        x_label = MathTex("X").scale(0.8)
        brace_up = Brace(array_mob, direction=UP)
        x_label.next_to(brace_up, UP, buff=SMALL_BUFF)

        self.play(
            Write(x_label),
            GrowFromCenter(brace_up),
        )
        self.wait()

        return VGroup(x_label, brace_up), apply_dct

    def show_highlight(self, pixel_row_mob, highlight):
        new_highlight = SurroundingRectangle(pixel_row_mob, buff=0).set_color(
            REDUCIBLE_GREEN_LIGHTER
        )
        self.play(
            LaggedStart(
                TransformFromCopy(highlight, new_highlight),
                FadeIn(pixel_row_mob),
                lag_ratio=0.4,
            )
        )
        self.wait()
        self.remove(new_highlight)
        self.play(
            FadeOut(highlight),
        )
        self.wait()

    def show_graph(
        self, ax, graph, dots, mob_above, animate=True, alignment_shift=None
    ):
        if alignment_shift is None:
            alignment_shift = DCT1DStepsVisualized.GRAPH_ADJUST
        graph_components = VGroup(ax, graph, dots).next_to(mob_above, DOWN)
        graph_components.shift(alignment_shift)
        if animate:
            self.play(
                Write(ax),
            )
            self.wait()
            self.play(
                *[GrowFromCenter(dot) for dot in dots],
            )
            self.wait()

            self.play(
                Create(graph),
            )
            self.wait()

        return ax, graph, dots

    def make_component(self, text, color=REDUCIBLE_VIOLET, scale=0.8):
        # geometry is first index, TextMob is second index
        text_mob = Tex(text).scale(scale)
        rect = Rectangle(color=color, height=1.1, width=3)
        return VGroup(rect, text_mob)

    def shift_components(self, pixel_row_mob, array_mob, graph, label, surround_rect):
        scale = 1
        new_position = LEFT * 3.5
        group = VGroup(pixel_row_mob, array_mob, graph, label, surround_rect)
        self.play(
            group.animate.scale(scale).move_to(new_position),
        )
        self.wait()
        return group

    def highlight_pixel_row(
        self, pixel_array, block_image_mob, row, height=SMALL_BUFF * 5, num_pixels=8
    ):
        row_length = height * num_pixels
        block_row_height = block_image_mob.height / num_pixels
        row_values = [pixel_array[row][i][0] for i in range(num_pixels)]
        highlight = Rectangle(height=block_row_height, width=block_image_mob.width)
        highlight_pos = (
            block_image_mob.get_top()
            + row * DOWN * block_row_height
            + DOWN * block_row_height / 2
        )
        highlight.move_to(highlight_pos).set_color(REDUCIBLE_GREEN_LIGHTER)
        pixel_row_mob = VGroup(
            *[
                Rectangle(height=height, width=row_length / num_pixels)
                .set_stroke(color=REDUCIBLE_GREEN_LIGHTER)
                .set_fill(color=gray_scale_value_to_hex(value), opacity=1)
                for value in row_values
            ]
        ).arrange(RIGHT, buff=0)
        return pixel_row_mob, np.array(row_values), highlight

    def get_array_obj(self, values, length=5, height=0.5, color=REDUCIBLE_GREEN_LIGHTER):
        array = VGroup(
            *[Rectangle(height=height, width=length / len(values)) for _ in values]
        ).arrange(RIGHT, buff=0)
        array.set_color(color)
        array_text = VGroup(
            *[
                Integer(val).scale(0.6).move_to(array[i].get_center())
                for i, val in enumerate(values)
            ]
        )
        return VGroup(array, array_text)

    def draw_image_graph(self, dct_row, centered=False, color=REDUCIBLE_YELLOW):
        """
        @param: 1D DCT of a row of pixels
        @param: centered, if true, then plot range is from [-128, 127]
        @return: graph composed of a linear combination of cosine functions weighted by dct_row
        """

        def get_basis_function(col):
            factor = dct_row[col] * np.sqrt(2 / dct_row.shape[0])

            def f(n):
                if col == 0:
                    return factor * 1 / np.sqrt(2)
                return factor * np.cos(
                    (2 * n + 1) * col * np.pi / (2 * dct_row.shape[0])
                )

            return f

        basis_functions = [get_basis_function(i) for i in range(dct_row.shape[0])]
        final_func = lambda n: sum(
            basis_function(n) for basis_function in basis_functions
        )
        if not centered:
            final_func = (
                lambda n: sum(basis_function(n) for basis_function in basis_functions)
                + 128
            )
        ax = self.get_axis(dct_row, centered=centered)
        graph = ax.plot(final_func)
        graph.set_color(color)
        ax.add_coordinates()
        row_values = idct_1d(dct_row)
        if not centered:
            row_values = row_values + 128
        pixel_coordinates = list(enumerate(row_values))
        dots = VGroup(
            *[
                Dot().scale(0.7).move_to(ax.coords_to_point(x, y)).set_color(color)
                for x, y in pixel_coordinates
            ]
        )

        return ax, graph, dots

    def get_axis(self, dct_row, centered=False):
        if not centered:
            return Axes(
                x_range=[0, dct_row.shape[0] - 1, 1],
                y_range=[0, 255, 1],
                y_length=2,
                x_length=4.375,
                tips=False,
                axis_config={"include_numbers": True, "include_ticks": False},
                x_axis_config={
                    "numbers_to_exclude": list(range(1, dct_row.shape[0]))
                },
                y_axis_config={"numbers_to_exclude": list(range(1, 255))},
            )
        return Axes(
            x_range=[0, dct_row.shape[0] - 1, 1],
            y_range=[-128, 127, 1],
            y_length=2,
            x_length=4.375,
            tips=False,
            axis_config={"include_numbers": True, "include_ticks": False},
            x_axis_config={
                "numbers_to_exclude": list(range(1, dct_row.shape[0])),
                "label_direction": UP,
            },
            y_axis_config={"numbers_to_exclude": list(range(-127, 127))},
        )

    def get_dct_axis(self, dct_row, min_y, max_y):
        ax = Axes(
            x_range=[0, dct_row.shape[0] - 1, 1],
            y_range=[min_y, max_y, 1],
            y_length=3,
            x_length=4.375,
            tips=False,
            axis_config={"include_numbers": True, "include_ticks": False},
            x_axis_config={"numbers_to_exclude": list(range(1, dct_row.shape[0]))},
            y_axis_config={"numbers_to_exclude": list(range(min_y + 1, max_y))},
        )
        return ax

    def plot_row_values(self, axes, pixel_row_values, color=REDUCIBLE_YELLOW):
        pixel_coordinates = list(enumerate(pixel_row_values))
        axes.add_coordinates()
        path = VGroup()
        path_points = [axes.coords_to_point(x, y) for x, y in pixel_coordinates]
        path.set_points_smoothly(*[path_points]).set_color(color)
        dots = VGroup(
            *[
                Dot(axes.coords_to_point(x, y), radius=SMALL_BUFF / 2, color=color)
                for x, y in pixel_coordinates
            ]
        )
        return path, dots

class HighVsLowFrequency(DCT1DStepsVisualized):
    def construct(self):
        title = Title("Frequency Components in Images").move_to(UP * 3.5)

        self.play(
            Write(title)
        )
        self.wait()

        high_frequency_array = np.array([10, 220, 15, 240, 20, 210, 30, 245])

        high_freq_values_centered = format_block(high_frequency_array)
        print("After centering\n", high_freq_values_centered)

        dct_row_pixels_high_freq = dct_1d(high_freq_values_centered)
        np.set_printoptions(suppress=True)
        print("DCT block (rounded)\n", np.round(dct_row_pixels_high_freq, decimals=1))

        high_freq_pixel_mob = self.make_row_of_pixels(high_frequency_array, height=0.6)
        ax_high_f, graph_high_f, dots_high_f = self.draw_image_graph(dct_row_pixels_high_freq)

        high_freq_pixel_mob.move_to(LEFT * 3.5 + UP)

        high_freq_text = Tex("High Frequency").scale(0.9).next_to(high_freq_pixel_mob, UP)
        self.play(
            Write(high_freq_text),
            FadeIn(high_freq_pixel_mob)
        )
        self.wait()

        self.show_graph(ax_high_f, graph_high_f, dots_high_f, high_freq_pixel_mob, animate=False)

        self.play(
            Write(ax_high_f)
        )

        self.play(
            *[GrowFromCenter(dot) for dot in dots_high_f],
            Create(graph_high_f)
        )
        self.wait()

        low_frequency_array = np.array([100, 106, 111, 115, 120, 126, 130, 135])
        low_freq_values_centered = format_block(low_frequency_array)
        print("After centering\n", low_freq_values_centered)

        dct_row_pixels_low_freq = dct_1d(low_freq_values_centered)
        np.set_printoptions(suppress=True)
        print("DCT block (rounded)\n", np.round(dct_row_pixels_low_freq, decimals=1))

        low_freq_pixel_mob = self.make_row_of_pixels(low_frequency_array, height=0.6)
        ax_low_f, graph_low_f, dots_low_f = self.draw_image_graph(dct_row_pixels_low_freq)

        low_freq_pixel_mob.move_to(RIGHT * 3.5 + UP)
        low_freq_text = Tex("Low Frequency").scale(0.9).next_to(low_freq_pixel_mob, UP)


        self.play(
            Write(low_freq_text),
            FadeIn(low_freq_pixel_mob)
        )
        self.wait()

        self.show_graph(ax_low_f, graph_low_f, dots_low_f, low_freq_pixel_mob, animate=False)

        self.play(
            Write(ax_low_f)
        )

        self.play(
            *[GrowFromCenter(dot) for dot in dots_low_f],
            Create(graph_low_f)
        )
        self.wait()

        high_freq_low_freq_group = VGroup(
            high_freq_text,
            high_freq_pixel_mob,
            ax_high_f,
            dots_high_f,
            graph_high_f,
            low_freq_text,
            low_freq_pixel_mob,
            ax_low_f,
            dots_low_f,
            graph_low_f,
        )

        first_idea = Tex("1. Real world images tend to have more low frequency components.").scale(0.8)
        first_idea.next_to(title, DOWN)

        self.play(
            FadeIn(first_idea)
        )
        self.wait()

        self.play(
            FadeOut(high_freq_low_freq_group)
        )
        self.wait()

        image_mob = ImageMobject("dog").move_to(ORIGIN)
        pixel_array = image_mob.get_pixel_array()
        height_pixels, width_pixels = pixel_array.shape[0], pixel_array.shape[1]
        self.play(FadeIn(image_mob))
        self.wait()

        dot = Dot(color=REDUCIBLE_GREEN_LIGHTER)
        for i in range(10):
            start_row, start_col = np.random.randint(0, height_pixels), np.random.randint(0, width_pixels)
            dot.move_to(self.get_highlight_pos(start_row, start_col, image_mob))
            _, _, random_block = self.get_pixel_block(image_mob, start_row, start_col)
            random_row = random_block[:, :, 0][0]
            if i == 0:
                pixel_mob = self.make_row_of_pixels(random_row, height=0.6)
                pixel_mob.next_to(image_mob, DOWN)
                self.add(dot, pixel_mob)
            else:
                pixel_mob.become(self.make_row_of_pixels(random_row, height=0.6).move_to(pixel_mob.get_center()))
            
            self.wait()

        second_idea = Tex("2. Human visual system is less sensitive to higher frequency detail.").scale(0.8)
        second_idea.next_to(first_idea, DOWN, aligned_edge=LEFT)

        self.play(
            FadeIn(second_idea)
        )
        self.wait()

        self.play(
            FadeOut(image_mob),
            FadeOut(dot),
            FadeOut(pixel_mob)
        )

        self.wait()

        self.play(
            FadeIn(high_freq_low_freq_group.shift(DOWN * 0.6))
        )

        self.wait()

        high_freq_group = VGroup(
            high_freq_text,
            high_freq_pixel_mob,
            ax_high_f,
            dots_high_f,
            graph_high_f,
        )

        cross = Cross(high_freq_group)

        self.play(
            Write(cross)
        )

        self.wait()

        question = Tex("How do we get frequency components from an image?").scale(0.8)
        question.next_to(high_freq_low_freq_group, DOWN * 2)

        self.play(
            FadeIn(question)
        )
        self.wait()

        answer = Tex("Discrete Cosine Transform (DCT)").set_color(REDUCIBLE_YELLOW).next_to(high_freq_low_freq_group, DOWN * 2)

        self.play(
            ReplacementTransform(question, answer)
        )
        self.wait()




    def get_highlight_pos(self, start_row, start_col, image_mob, block_size=8):
        pixel_array = image_mob.get_pixel_array()
        center_row = start_row + block_size // 2
        center_col = start_col + block_size // 2
        vertical_pos = (
            image_mob.get_top()
            + DOWN * center_row / pixel_array.shape[0] * image_mob.height
        )
        horizontal_pos = (
            image_mob.get_left()
            + RIGHT * center_col / pixel_array.shape[1] * image_mob.width
        )
        highlight_position = np.array([horizontal_pos[0], vertical_pos[1], 0])
        return highlight_position

class MotivateDCT(DCT1DStepsVisualized):
    def construct(self):
        all_equations = self.show_equations()

        self.ask_why(all_equations)

        self.clear()

        image_mob = ImageMobject("dog").move_to(UP * 2)
        block_image, pixel_grid, block = self.get_pixel_block(image_mob, 126, 126)
        row = 7
        print(block[:, :, 0])
        print(f"Block row: {row}\n", block[:, :, 0][row])
        pixel_row_mob, row_values = self.get_pixel_row_mob(block, row)
        print("Selected row values\n", row_values)
        pixel_row_mob.move_to(UP * 3)
        self.play(
            FadeIn(pixel_row_mob)
        )
        self.wait()

        row_values_centered = format_block(row_values)
        print('After centering\n', row_values_centered)

        dct_row_pixels = dct_1d(row_values_centered)

        ax, graph, dots = self.draw_image_graph(dct_row_pixels)

        self.show_graph(ax, graph, dots, pixel_row_mob)

        group, arrow = self.show_summing_different_cosine_waves(graph)

        self.show_pixel_rows_for_cosine_freqs(group, arrow)

        self.describe_dct_broadly(ax, graph, dots, pixel_row_mob, dct_row_pixels)

        self.derive_matrix()

    def show_equations(self):
        dct_text = Tex("Discrete Cosine Transform (DCT)").scale(1.2)
        forward_dct = MathTex(r"X_k = \left(\frac{2}{N}\right)^{\frac{1}{2}} \sum_{n=0}^{N-1} \Lambda(n) \cdot \cos \left[\frac{\pi k}{2N}(2n+1)\right]x_n")

        lambda_def = MathTex(r"\Lambda(n) = \left\{\begin{array}{ll} \frac{1}{\sqrt{2}} & \quad n = 0 \\ 1 & \quad n \neq 0 \end{array}\right.").scale(0.8)
        k_def = MathTex(r"k \in \{0, 1, \ldots , N - 1\}").scale(0.8)
        additional_def_group = VGroup(k_def, lambda_def).arrange(RIGHT, buff=1)

        dct_group = VGroup(dct_text, forward_dct, additional_def_group).arrange(DOWN)

        idct_text = Tex("Inverse Discrete Cosine Transform (IDCT)").scale(1.2)
        inverse_dct = MathTex(r"x_k = \frac{X_0}{\sqrt{N}} + \left(\frac{2}{N}\right)^{\frac{1}{2}} \sum_{n=1}^{N-1} \cos \left[\frac{\pi n}{2N}(2k+1)\right]X_n")

        idct_group = VGroup(idct_text, inverse_dct).arrange(DOWN)

        group = VGroup(dct_group, idct_group).arrange(DOWN, buff=1)

        self.play(
            FadeIn(group)
        )
        self.wait()

        return group

    def ask_why(self, equations):
        self.play(
            equations.animate.scale(0.7)
        )
        surround_rect = SurroundingRectangle(equations, buff=SMALL_BUFF)
        self.play(
            Create(surround_rect)
        )

        question_1 = Tex("Where do these equations from?")
        question_2 = Tex("Why do we use these transforms?")
        questions = VGroup(question_1, question_2).arrange(DOWN)
        questions.next_to(surround_rect, UP)

        self.play(
            Write(questions[0])
        )
        self.wait()

        self.play(
            Write(questions[1])
        )

        self.wait()

    def show_summing_different_cosine_waves(self, axes):
        arrow = MathTex(r"\Updownarrow")

        arrow.next_to(axes, DOWN).shift(DOWN * 1)

        self.play(
            Write(arrow)
        )
        self.wait()

        first_freq = self.get_cosine_wave(lambda x: np.cos(x))
        second_freq = self.get_cosine_wave(lambda x: np.cos(2 * x))
        last_freq = self.get_cosine_wave(lambda x: np.cos(7 * x))

        plus = MathTex("+")
        ellipses = MathTex(r"\cdots")

        group = VGroup(first_freq, plus, second_freq, plus.copy(), ellipses, plus.copy(), last_freq).arrange(RIGHT)

        group.next_to(arrow, DOWN * 2)

        self.play(
            FadeIn(group)
        )
        self.wait()

        return group, arrow

    def get_cosine_wave(self, cosine_function):
        ax = Axes(
            x_range=[0, np.pi],
            y_range=[-1, 1],
            x_length=2,
            y_length=2,
        )

        graph = ax.plot(cosine_function).set_color(REDUCIBLE_YELLOW)

        box = SurroundingRectangle(graph, color=REDUCIBLE_VIOLET)
        return VGroup(graph, box)

    def get_cosine_wave_with_ax(self, cosine_function):
        ax = Axes(
            x_range=[0, np.pi],
            y_range=[-1, 1],
            x_length=4.375,
            y_length=2,
            tips=False,
            x_axis_config={"include_numbers": False, "include_ticks": False},
            y_axis_config={"include_numbers": True, "numbers_to_exclude": [0], "include_ticks": False}
        )

        graph = ax.plot(cosine_function).set_color(REDUCIBLE_YELLOW)
        pi_label = MathTex(r"\pi")
        pi_label.next_to(ax.x_axis, DOWN, aligned_edge=RIGHT)
        ax.add(pi_label)
        group = VGroup(ax, graph)

        return group

    def show_pixel_rows_for_cosine_freqs(self, cosine_waves, arrow):
        new_group = VGroup(*[mob.copy() for mob in cosine_waves]).arrange(RIGHT, buff=0.7)
        new_group.move_to(cosine_waves.get_center())
        self.play(
            FadeOut(arrow),
            Transform(cosine_waves, new_group)
        )
        self.wait()

        first_freq = new_group[0]
        second_freq = new_group[2]
        last_freq = new_group[-1]

        first_freq_dct_pixels, _ = self.get_dct_component(1)
        first_freq_pixel_row = self.make_row_of_pixels(first_freq_dct_pixels, height=SMALL_BUFF * 3)
        first_freq_pixel_row.next_to(first_freq, UP)

        second_freq_dct_pixels, _ = self.get_dct_component(2)
        second_freq_pixel_row = self.make_row_of_pixels(second_freq_dct_pixels, height=SMALL_BUFF * 3)
        second_freq_pixel_row.next_to(second_freq, UP)

        last_freq_dct_pixels, _ = self.get_dct_component(7)
        last_freq_pixel_row =  self.make_row_of_pixels(last_freq_dct_pixels, height=SMALL_BUFF * 3)
        last_freq_pixel_row.next_to(last_freq, UP)

        self.play(
            FadeIn(first_freq_pixel_row),
            FadeIn(second_freq_pixel_row),
            FadeIn(last_freq_pixel_row),
        )

        self.wait()

        weight1 = MathTex(r"X_1").next_to(first_freq_pixel_row, UP)
        weight2 = MathTex(r"X_2").next_to(second_freq_pixel_row, UP)
        weightn = MathTex(r"X_N").next_to(last_freq_pixel_row, UP)

        self.play(
            FadeIn(weight1),
            FadeIn(weight2),
            FadeIn(weightn)
        )
        self.wait()

        cross = Cross(VGroup(last_freq_pixel_row, last_freq, weightn))
        self.play(
            Create(cross)
        )
        self.wait()

        self.play(
            FadeOut(cross),
            FadeOut(weight1),
            FadeOut(weight2),
            FadeOut(weightn),
            FadeOut(cosine_waves),
            FadeOut(first_freq_pixel_row),
            FadeOut(second_freq_pixel_row),
            FadeOut(last_freq_pixel_row),
        )

    def describe_dct_broadly(self, ax, graph, dots, pixel_row_mob, dct_row_pixels):
        group = VGroup(ax, graph, dots, pixel_row_mob)
        self.play(
            group.animate.move_to(LEFT * 3.5 + DOWN * 0.5)
        )
        self.wait()
        general_vals = [f"x_{i}" for i in range(len(pixel_row_mob))]
        array_mob_symbols = self.get_gen_array_obj(general_vals, length=pixel_row_mob.width, height=pixel_row_mob.height)
        array_mob_symbols.next_to(pixel_row_mob, UP)

        self.play(
            FadeIn(array_mob_symbols)
        )
        self.wait()

        forward_arrow = MathTex(r"\Rightarrow").scale(1.5).shift(RIGHT * SMALL_BUFF * 3)
        self.play(
            Write(forward_arrow)
        )
        self.wait()

        pixel_space_group = VGroup(group, array_mob_symbols)

        dct_ax = self.get_dct_axis(dct_row_pixels, -80, 80)

        dct_graph, dct_points = self.plot_row_values(dct_ax, dct_row_pixels, color=REDUCIBLE_PURPLE)

        dct_graph_components = VGroup(dct_ax, dct_graph, dct_points).move_to(RIGHT * 3.5 + DOWN * 0.5)
        self.play(
            Write(dct_ax),
        )

        vertical_lines = self.get_vertical_lines_from_points(dct_ax, dct_points)

        self.play(
            *[Create(line) for line in vertical_lines],
            *[GrowFromCenter(dot) for dot in dct_points],
        )
        self.wait()

        self.play(
            Create(dct_graph)
        )
        self.wait()

        general_dct_vals = [f"X_{i}" for i in range(len(pixel_row_mob))]

        array_mob_dct_symbols = self.get_gen_array_obj(general_dct_vals, length=pixel_row_mob.width + 0.5, height=pixel_row_mob.height + SMALL_BUFF, color=REDUCIBLE_VIOLET)
        array_mob_dct_symbols.next_to(pixel_row_mob, UP)
        shift_amount = dct_graph_components.get_center()[0] - array_mob_dct_symbols.get_center()[0]
        array_mob_dct_symbols.shift(RIGHT * (shift_amount + 0.3))

        self.play(
            FadeIn(array_mob_dct_symbols)
        )
        self.wait()

        dct_space_group = VGroup(dct_graph_components, vertical_lines, array_mob_dct_symbols)

        dct_coeff_description = Tex(r"$X_k$ is the contribution of cosine wave $C_k$")

        dct_coeff_description.move_to(UP * 3.5)

        next_question = Tex(r"What cosine waves $C_k$ should we use?").scale(0.8)

        next_question.next_to(dct_coeff_description, DOWN)

        self.play(
            Write(dct_coeff_description)
        )
        self.wait()

        self.play(
            Write(next_question)
        )

        self.wait()

        general_properties = Tex("What properties do we want?")
        general_properties.move_to(UP * 3.5)

        self.play(
            ReplacementTransform(dct_coeff_description, general_properties),
            FadeOut(next_question)
        )
        self.wait()

        invertibility = Tex("Invertibility").move_to(general_properties.get_center())

        forward_dct_group = VGroup(pixel_space_group, forward_arrow, dct_space_group)

        self.play(
            ReplacementTransform(general_properties, invertibility)
        )
        self.wait()

        surround_rect_forward = SurroundingRectangle(forward_dct_group, color=REDUCIBLE_YELLOW, buff=MED_SMALL_BUFF)

        self.play(
            Create(surround_rect_forward)
        )
        self.wait()

        self.play(
            forward_dct_group.animate.scale(0.65).shift(UP * 1.5),
            surround_rect_forward.animate.scale(0.65).shift(UP * 1.5)
        )

        self.wait()
        shift_down = DOWN * 3.5
        new_idct_group_left = dct_space_group.copy().move_to(pixel_space_group.get_center()).shift(shift_down)
        new_idct_arrow = forward_arrow.copy().shift(shift_down)
        new_idct_group_right = pixel_space_group.copy().move_to(dct_space_group.get_center()).shift(shift_down)

        self.play(
            TransformFromCopy(dct_space_group, new_idct_group_left)
        )
        self.wait()

        self.play(
            TransformFromCopy(forward_arrow, new_idct_arrow)
        )
        self.wait()

        self.play(
            TransformFromCopy(pixel_space_group, new_idct_group_right)
        )
        self.wait()

        inverse_dct_group = VGroup(new_idct_group_left, new_idct_arrow, new_idct_group_right)

        surround_rect_inverse = SurroundingRectangle(inverse_dct_group, color=REDUCIBLE_PURPLE, buff=MED_SMALL_BUFF)

        self.play(
            Create(surround_rect_inverse)
        )
        self.wait()

        shift_left = LEFT * 2

        self.play(
            surround_rect_forward.animate.shift(shift_left),
            surround_rect_inverse.animate.shift(shift_left),
            forward_dct_group.animate.shift(shift_left),
            inverse_dct_group.animate.shift(shift_left),
        )
        self.wait()

        forward_transform = MathTex(r"\vec{X} = M \vec{x}")
        forward_transform.next_to(surround_rect_forward, RIGHT).shift(RIGHT * 1.3)
        inverse_transform = MathTex(r"\vec{x} = M^{-1} \vec{X}")
        inverse_transform.next_to(surround_rect_inverse, RIGHT).shift(RIGHT * 1)

        self.play(
            FadeIn(forward_transform),
            FadeIn(inverse_transform)
        )
        self.wait()

        forward_dct_text = Tex("DCT").scale(1.2)
        inverse_dct_text = Tex("IDCT").scale(1.2)

        forward_dct_text.next_to(forward_transform, UP)
        inverse_dct_text.next_to(inverse_transform, UP)

        self.play(
            FadeIn(forward_dct_text),
            FadeIn(inverse_dct_text)
        )
        self.wait()

        self.clear()

    def derive_matrix(self):
        matrix_m_def = Tex("How should we define matrix $M$?").move_to(UP * 3.5)
        self.play(
            Write(matrix_m_def)
        )
        self.wait()
        row_values = np.ones(8) * 255
        pixel_row_mob = self.make_row_of_pixels(row_values)
        pixel_row_mob.next_to(matrix_m_def, DOWN * 2)

        self.play(
            FadeIn(pixel_row_mob)
        )
        self.wait()
        row_values_centered = format_block(row_values)
        print('After centering\n', row_values_centered)

        dct_row_pixels = dct_1d(row_values_centered)
        print(dct_row_pixels)

        ax, graph, dots = self.draw_image_graph(dct_row_pixels)

        self.show_graph(ax, graph, dots, pixel_row_mob)

        cosine_graph = self.get_cosine_wave_with_ax(lambda x: np.cos(0))
        cosine_graph.next_to(ax, DOWN)
        cosine_freq_0_func = MathTex(r"y = \cos (0 \cdot x)")
        cosine_freq_0_func.move_to(DOWN  * 3.5)
        self.play(
            FadeIn(cosine_graph),
            FadeIn(cosine_freq_0_func)
        )
        self.wait()

        group = VGroup(ax, graph, dots, pixel_row_mob)
        self.play(
            group.animate.move_to(LEFT * 3.5 + DOWN * 0.5),
            FadeOut(cosine_graph),
            FadeOut(cosine_freq_0_func)
        )
        self.wait()
        array_mob = self.get_array_obj(row_values)
        array_mob.next_to(pixel_row_mob, UP)

        self.play(
            FadeIn(array_mob)
        )
        self.wait()

        forward_arrow = MathTex(r"\Rightarrow").scale(1.5).shift(RIGHT * SMALL_BUFF * 3)
        self.play(
            Write(forward_arrow)
        )
        self.wait()

        pixel_space_group = VGroup(group, array_mob)

        dct_ax = self.get_dct_axis(dct_row_pixels, -360, 360)

        dct_graph, dct_points = self.plot_row_values(dct_ax, dct_row_pixels, color=REDUCIBLE_PURPLE)

        dct_graph_components = VGroup(dct_ax, dct_graph, dct_points).move_to(RIGHT * 3.5 + DOWN * 0.5)
        self.play(
            Write(dct_ax),
        )

        vertical_lines = self.get_vertical_lines_from_points(dct_ax, dct_points)

        self.play(
            *[Create(line) for line in vertical_lines],
            *[GrowFromCenter(dot) for dot in dct_points],
        )
        self.wait()

        self.play(
            Create(dct_graph)
        )
        self.wait()

        general_dct_vals = ["X_0"] + ["0"] * 7

        array_mob_dct_symbols = self.get_gen_array_obj(general_dct_vals, length=pixel_row_mob.width + 0.5, height=pixel_row_mob.height + SMALL_BUFF, color=REDUCIBLE_VIOLET)
        array_mob_dct_symbols.next_to(pixel_row_mob, UP)
        shift_amount = dct_graph_components.get_center()[0] - array_mob_dct_symbols.get_center()[0]
        array_mob_dct_symbols.shift(RIGHT * (shift_amount + 0.3))

        self.play(
            FadeIn(array_mob_dct_symbols)
        )
        self.wait()

        dct_space_group = VGroup(dct_graph_components, vertical_lines, array_mob_dct_symbols)
        entire_group = VGroup(pixel_space_group, forward_arrow, dct_space_group)

        self.play(
            entire_group.animate.scale(0.7).shift(UP * 1.5)
        )
        self.wait()

        dct_graph_group = VGroup(dct_ax, dct_graph, dct_points, vertical_lines)

        matrix = self.get_matrix_m()

        vector = self.make_column_vector(row_values)
        vector.next_to(matrix, RIGHT)

        equals = MathTex("=").scale(1.5)

        result_vector = self.make_column_vector(general_dct_vals)
        result_vector.next_to(vector, RIGHT)

        matrix_equation = VGroup(matrix, vector, equals, result_vector).arrange(RIGHT, buff=0.5)

        matrix_equation.move_to(DOWN * 2)

        self.play(
            FadeIn(matrix_equation)
        )

        self.wait()

        self.play(
            Indicate(matrix[1][1:])
        )
        self.wait()

        self.play(
            Indicate(vector)
        )
        self.wait()

        self.play(
            LaggedStartMap(Indicate, result_vector[0][1:])
        )
        self.wait()

        original_first_row_matrix = matrix[1][0]

        first_row = VGroup(*[Integer(1).scale(0.8) for _ in range(8)]).arrange(RIGHT, buff=0.4).move_to(original_first_row_matrix.get_center())
        # first_row.stretch_to_fit_width(original_first_row_matrix.width)
        self.play(
            Transform(original_first_row_matrix, first_row)
        )
        self.wait()

        norm_first_row = VGroup(*[MathTex(r"\frac{1}{\sqrt{8}}").scale(0.6).move_to(element.get_center()) for element in first_row])

        self.play(
            Transform(original_first_row_matrix, norm_first_row)
        )
        self.wait()

        self.center_about_zero_and_animate(matrix_m_def, pixel_space_group, dct_graph_group, vector)

    def center_about_zero_and_animate(self, matrix_m_def, pixel_space_group, dct_graph_group, original_vec):
        group, array = pixel_space_group
        ax, graph, dots, pixel_row_mob = group

        tracker = ValueTracker(127)

        def get_new_pixel_space_group():
            row_values = np.ones(8) * tracker.get_value()
            pixel_row_mob = self.make_row_of_pixels(row_values + 128)
            pixel_row_mob.next_to(matrix_m_def, DOWN * 2)

            dct_row_pixels = dct_1d(row_values)

            ax, graph, dots = self.draw_image_graph(dct_row_pixels, centered=True)

            self.show_graph(ax, graph, dots, pixel_row_mob, animate=False)

            group = VGroup(ax, graph, dots, pixel_row_mob)
            group.move_to(LEFT * 3.5 + DOWN * 0.5)
            length = 5 
            if tracker.get_value() <= -99.5:
                length = 5.5
            array_mob = self.get_array_obj(row_values, length=length)
            array_mob.next_to(pixel_row_mob, UP)

            new_pixel_space_group = VGroup(group, array_mob)
            new_pixel_space_group.scale(0.75).move_to(pixel_space_group.get_center())

            return new_pixel_space_group

        def get_new_dct_space_group():
            row_values = np.ones(8) * tracker.get_value()
            dct_row_pixels = dct_1d(row_values)
            dct_ax = self.get_dct_axis(dct_row_pixels, -360, 360)

            dct_graph, dct_points = self.plot_row_values(dct_ax, dct_row_pixels, color=REDUCIBLE_PURPLE)

            dct_graph_components = VGroup(dct_ax, dct_graph, dct_points).move_to(RIGHT * 3.5 + DOWN * 0.5)

            vertical_lines = self.get_vertical_lines_from_points(dct_ax, dct_points)

            new_dct_graph_group = VGroup(dct_ax, dct_graph, dct_points, vertical_lines)
            new_dct_graph_group.scale(0.7).move_to(dct_graph_group.get_center())
            return new_dct_graph_group

        def get_new_column_vector():
            row_values = np.ones(8) * tracker.get_value()
            column_vec = self.make_column_vector(row_values)
            return column_vec.move_to(original_vec.get_center())

        new_pixel_space_group = always_redraw(get_new_pixel_space_group)
        new_dct_graph_group = always_redraw(get_new_dct_space_group)
        new_column_vector = always_redraw(get_new_column_vector)

        self.play(
            ReplacementTransform(pixel_space_group, new_pixel_space_group),
            ReplacementTransform(dct_graph_group, new_dct_graph_group),
            ReplacementTransform(original_vec, new_column_vector)
        )
        self.wait()

        self.play(
            tracker.animate.set_value(-128),
            run_time=5,
            rate_func=linear,
        )

        self.play(
            tracker.animate.set_value(127),
            run_time=5,
            rate_func=linear,
        )

        self.wait()

    def get_matrix_m(self):
        row0 = self.get_cosine_row_tex(0)
        row1 = self.get_cosine_row_tex(1)
        vdots = MathTex(r"\vdots")
        row7 = self.get_cosine_row_tex(7)

        rows = VGroup(row0, row1, vdots, row7).arrange(DOWN).move_to(DOWN * 2)
        bracket_pair = MathTex("[", "]")
        bracket_pair.scale(2)
        bracket_v_buff = MED_SMALL_BUFF
        bracket_h_buff = MED_SMALL_BUFF
        bracket_pair.stretch_to_fit_height(rows.height + 2 * bracket_v_buff)
        l_bracket, r_bracket = bracket_pair.split()
        l_bracket.next_to(rows, LEFT, bracket_h_buff)
        r_bracket.next_to(rows, RIGHT, bracket_h_buff)
        brackets = VGroup(l_bracket, r_bracket)

        return VGroup(brackets, rows)

    def make_column_vector(self, values):
        integer_values = []
        for value in values:
            if isinstance(value, str):
                integer_values.append(value)
            else:
                integer_values.append(int(value))
        vector = Matrix([[value] for value in integer_values], v_buff=0.6, element_alignment_corner=DOWN)
        return vector.scale(0.6)

    def get_cosine_row_tex(self, index):
        text = MathTex(f"C_{index}^T").scale(0.8)
        left_arrow = Arrow(RIGHT * 2, ORIGIN, stroke_width=3, max_tip_length_to_length_ratio=0.15).next_to(text, LEFT).set_color(WHITE)
        right_arrow = Arrow(ORIGIN, RIGHT * 2, stroke_width=3, max_tip_length_to_length_ratio=0.15).next_to(text, RIGHT).set_color(WHITE)

        return VGroup(left_arrow, text, right_arrow)

    def get_vertical_lines_from_points(self, dct_ax, dct_points):
        x_axis_points = [dct_ax.x_axis.n2p(i) for i in range(len(dct_points))]
        vertical_lines = [Line(start, end_point.get_center()).set_stroke(color=REDUCIBLE_VIOLET, width=8) for start, end_point in zip(x_axis_points, dct_points)]
        return VGroup(*vertical_lines)

    def get_gen_array_obj(self, values, length=5, height=0.5, color=REDUCIBLE_GREEN_LIGHTER):
        array = VGroup(*[Rectangle(height=height, width=length/len(values)) for _ in values]).arrange(RIGHT, buff=0)
        array.set_color(color)
        array_text = VGroup(*[MathTex(val).scale(0.6).move_to(array[i].get_center()) for i, val in enumerate(values)])
        return VGroup(array, array_text)

    def make_row_of_pixels(self, row_values, height=SMALL_BUFF*5, num_pixels=8):
        row_length = height * num_pixels
        adjusted_row_values = []
        for val in row_values:
            adjusted_row_values.append(int(round(val)))
        pixel_row_mob = VGroup(
            *[
                Rectangle(height=height, width=row_length / num_pixels)
                .set_stroke(color=REDUCIBLE_GREEN_LIGHTER)
                .set_fill(color=gray_scale_value_to_hex(value), opacity=1)
                for value in adjusted_row_values
            ]
        ).arrange(RIGHT, buff=0)
        return pixel_row_mob

    def get_dct_component(self, col, new_value=250, num_pixels=8):
        dct_row = np.zeros((num_pixels,))
        dct_row[col] = new_value
        return idct_1d(dct_row) + 128, dct_row

class CosineSampling(MotivateDCT):
    def construct(self):
        ax, graph, cosine_label = self.introduce_cosine()
        dots = self.get_dots(ax, graph)
        line_intervals = self.get_intervals(ax, graph)
        ticks = self.get_ticks_for_x_axis()
        vertical_lines = self.get_vertical_lines_from_points(ax, dots)

        self.play(
            *[GrowFromCenter(line) for line in line_intervals],
            *[Write(tick) for tick in ticks]
        )
        self.wait()

        self.play(
            LaggedStartMap(Create, vertical_lines),
            run_time=2
        )
        self.play(
            LaggedStartMap(GrowFromCenter, dots)
        )
        self.wait()

        labels = self.show_sample_x_vals(vertical_lines)

    def introduce_cosine(self):
        ax, graph  = self.get_cosine_wave_with_ax(lambda x: np.cos(x))
        self.play(
            Write(ax)
        )
        self.wait()

        self.play(
            Create(graph)
        )
        self.wait()

        cosine_label = MathTex(r"y = \cos(x)")
        cosine_label.next_to(graph, DOWN)
        self.play(
            Write(cosine_label)
        )
        self.wait()
        return ax, graph, cosine_label

    def get_x_y_points(self):
        x_points = [(j * 2 + 1) * np.pi / 16 for j in range(8)]
        y_points = [np.cos(x) for x in x_points]
        return x_points, y_points

    def get_dots(self, ax, graph, color=REDUCIBLE_YELLOW):
        x_points, y_points = self.get_x_y_points()
        points = [ax.coords_to_point(x, y) for x, y in zip(x_points, y_points)]

        dots = VGroup(*[Dot().set_color(color).move_to(p) for p in points])
        return dots

    def get_intervals(self, ax, graph):
        proportions = np.arange(0, np.pi + 0.0001, np.pi / 8)
        lines = []
        for i in range(len(proportions) - 1):
            start, end = proportions[i], proportions[i + 1]
            start_point, end_point = ax.x_axis.n2p(start), ax.x_axis.n2p(end)
            line = Line(start_point, end_point).set_stroke(width=5)
            if i % 2 == 0:
                line.set_color(REDUCIBLE_GREEN_LIGHTER)
            else:
                line.set_color(REDUCIBLE_GREEN_DARKER)

            lines.append(line)

        return lines

    def show_sample_x_vals(self, vertical_lines):
        labels = VGroup(*[MathTex(r"\frac{\pi}{16}").scale(0.7)] + [MathTex(r"\frac{" + str(2 * i + 1) + r"\pi}{16}").scale(0.6) for i in range(1, len(vertical_lines))])
        for label, line in zip(labels, vertical_lines):
            direction = normalize(line.get_start() - line.get_end())
            direction = np.array([int(c) for c in direction])
            label.next_to(line, direction)

        self.play(
            FadeIn(labels)
        )
        self.wait()
        return labels

    def get_cosine_wave_with_ax(self, cosine_function):
        ax = Axes(
            x_range=[0, np.pi],
            y_range=[-1, 1],
            x_length=10,
            y_length=5.5,
            tips=False,
            x_axis_config={"include_numbers": False, "include_ticks": False},
            y_axis_config={"include_numbers": True, "numbers_to_exclude": [0], "include_ticks": False}
        )

        graph = ax.plot(cosine_function).set_color(REDUCIBLE_YELLOW)
        pi_label = MathTex(r"\pi")
        pi_label.next_to(ax.x_axis, DOWN, aligned_edge=RIGHT)
        ax.add(pi_label)

        group = VGroup(ax, graph)

        return group

    def get_ticks_for_x_axis(self):
        ax = Axes(
            x_range=[0, np.pi, np.pi / 8],
            y_range=[-1, 1],
            x_length=10,
            y_length=5.5,
            tips=False,
            x_axis_config={"include_numbers": False, "include_ticks": True},
            y_axis_config={"include_numbers": True, "numbers_to_exclude": [0], "include_ticks": False}
        )
        return ax.x_axis.ticks

    def get_vertical_lines_from_points(self, ax, points):
        x_points = [ax.x_axis.n2p(p) for p in self.get_x_y_points()[0]]
        vertical_lines = [Line(start_point, end.get_center()).set_stroke(color=REDUCIBLE_VIOLET, width=8) for start_point, end in zip(x_points, points)]
        return VGroup(*vertical_lines)

class RevisedMotivateDCT(MotivateDCT):
    def construct(self):
        image_mob = ImageMobject("dog").move_to(UP * 2)
        block_image, pixel_grid, block = self.get_pixel_block(image_mob, 126, 126)
        row = 7
        print(block[:, :, 0])
        print(f"Block row: {row}\n", block[:, :, 0][row])
        pixel_row_mob, row_values = self.get_pixel_row_mob(block, row, height=0.6)
        print("Selected row values\n", row_values)
        pixel_row_mob.move_to(UP * 3)
        self.play(
            FadeIn(pixel_row_mob)
        )
        self.wait()

        row_values_centered = format_block(row_values)
        print('After centering\n', row_values_centered)

        dct_row_pixels = dct_1d(row_values_centered)

        ax, graph, dots = self.draw_image_graph(dct_row_pixels)

        self.show_graph(ax, graph, dots, pixel_row_mob)

        self.show_summing_different_cosine_waves(graph, dots)

        self.describe_dct_broadly(ax, graph, dots, pixel_row_mob, dct_row_pixels)

        self.clear()

        self.experiment_with_cosine()

    def show_summing_different_cosine_waves(self, graph, original_dots):
        arrow = MathTex(r"\Updownarrow")

        arrow.next_to(graph, DOWN).shift(DOWN * 1)

        self.play(
            Write(arrow)
        )
        self.wait()

        first_freq, first_axes = self.get_cosine_wave(lambda x: np.cos(x))
        second_freq, second_axes = self.get_cosine_wave(lambda x: np.cos(2 * x))
        last_freq, last_axes = self.get_cosine_wave(lambda x: np.cos(7 * x))

        first_freq_dots = self.get_dots(first_axes, first_freq, 1)
        second_freq_dots = self.get_dots(second_axes, second_freq, 2)
        last_freq_dots = self.get_dots(last_axes, last_freq, 7)

        first_cosine_graph = VGroup(first_freq, first_freq_dots)
        second_cosine_graph = VGroup(second_freq, second_freq_dots)
        last_cosine_graph = VGroup(last_freq, last_freq_dots)

        plus = MathTex("+")
        ellipses = MathTex(r"\cdots")

        group = VGroup(first_cosine_graph, plus, second_cosine_graph, plus.copy(), ellipses, plus.copy(), last_cosine_graph).arrange(RIGHT)

        group.next_to(arrow, DOWN * 2)

        self.play(
            FadeIn(group)
        )
        self.wait()

        self.emphasize_sampled_points(original_dots, first_freq_dots, second_freq_dots, last_freq_dots)

        self.emphasize_continuous_funcs(graph, first_freq[0], second_freq[0], last_freq[0])

        self.second_empasize_points(original_dots, first_freq_dots, second_freq_dots, last_freq_dots)

        self.play(
            FadeOut(group),
            FadeOut(arrow)
        )
        self.wait()

    def emphasize_sampled_points(self, original_dots, cosine_1_dots, cosine_2_dots, cosine_7_dots): 
        group_of_dots = []
        for i in range(len(original_dots)):
            group_of_dots.append(VGroup(original_dots[i], cosine_1_dots[i], cosine_2_dots[i], cosine_7_dots[i]))

        self.play(
            LaggedStartMap(Indicate, group_of_dots),
            run_time=3
        )
        self.wait()

    def emphasize_continuous_funcs(self, original_graph, cosine_1_graph, cosine_2_graph, cosine_7_graph):
        self.play(
            ApplyWave(original_graph),
            ApplyWave(cosine_1_graph),
            ApplyWave(cosine_2_graph),
            ApplyWave(cosine_7_graph),
            run_time=2
        )
        self.wait()

    def second_empasize_points(self, original_dots, cosine_1_dots, cosine_2_dots, cosine_7_dots):
        self.play(
            Indicate(VGroup(original_dots, cosine_1_dots, cosine_2_dots, cosine_7_dots))
        )
        self.wait()

    def describe_dct_broadly(self, ax, graph, dots, pixel_row_mob, dct_row_pixels):
        group = VGroup(ax, graph, dots, pixel_row_mob)
        self.play(
            group.animate.move_to(LEFT * 3.5 + DOWN * 0.5)
        )
        self.wait()
        general_vals = [f"x_{i}" for i in range(len(pixel_row_mob))]
        array_mob_symbols = self.get_gen_array_obj(general_vals, length=pixel_row_mob.width, height=pixel_row_mob.height)
        array_mob_symbols.next_to(pixel_row_mob, UP)

        self.play(
            FadeIn(array_mob_symbols)
        )
        self.wait()

        forward_arrow = MathTex(r"\Rightarrow").scale(1.5).shift(RIGHT * SMALL_BUFF * 3)
        self.play(
            Write(forward_arrow)
        )
        self.wait()

        pixel_space_group = VGroup(group, array_mob_symbols)

        dct_ax = self.get_dct_axis(dct_row_pixels, -80, 80)

        dct_graph, dct_points = self.plot_row_values(dct_ax, dct_row_pixels, color=REDUCIBLE_PURPLE)

        dct_graph_components = VGroup(dct_ax, dct_graph, dct_points).move_to(RIGHT * 3.5 + DOWN * 0.5)
        self.play(
            Write(dct_ax),
        )

        vertical_lines = self.get_vertical_lines_from_points(dct_ax, dct_points)

        self.play(
            *[Create(line) for line in vertical_lines],
            *[GrowFromCenter(dot) for dot in dct_points],
        )
        self.wait()

        self.play(
            Create(dct_graph)
        )
        self.wait()

        general_dct_vals = [f"X_{i}" for i in range(len(pixel_row_mob))]

        array_mob_dct_symbols = self.get_gen_array_obj(general_dct_vals, length=pixel_row_mob.width + 0.5, height=pixel_row_mob.height + SMALL_BUFF, color=REDUCIBLE_VIOLET)
        array_mob_dct_symbols.next_to(pixel_row_mob, UP)
        shift_amount = dct_graph_components.get_center()[0] - array_mob_dct_symbols.get_center()[0]
        array_mob_dct_symbols.shift(RIGHT * (shift_amount + 0.3))

        self.play(
            FadeIn(array_mob_dct_symbols)
        )
        self.wait()

        dct_space_group = VGroup(dct_graph_components, vertical_lines, array_mob_dct_symbols)

        dct_coeff_label = Tex("DCT coefficients").scale(0.8)
        brace = Brace(array_mob_dct_symbols, direction=UP)
        self.play(
            GrowFromCenter(brace)
        )
        dct_coeff_label.next_to(brace, UP)
        self.play(
            Write(dct_coeff_label)
        )
        self.wait()

        dct_coeff_description = Tex(r"Coefficient $X_k$ is the contribution of cosine wave $C_k$")

        dct_coeff_description.move_to(UP * 3.5)

        shift_up = UP * 1
        self.play(
            FadeOut(dct_coeff_label),
            FadeOut(brace),
            forward_arrow.animate.shift(shift_up),
            pixel_space_group.animate.shift(shift_up),
            dct_space_group.animate.shift(shift_up),
            Write(dct_coeff_description)
        )
        self.wait()


        next_question = Tex(r"What cosine waves $C_k$ should we use?")

        next_question.move_to(dct_coeff_description.get_center())

        self.show_dct_intuiton(graph, dots, array_mob_dct_symbols)
        self.play(
            ReplacementTransform(dct_coeff_description, next_question)
        )

        self.wait()

        image_connection = Tex("How do cosine waves relate to pixels on an image?")
        image_connection.move_to(next_question.get_center())

        self.play(
            ReplacementTransform(next_question, image_connection)
        )
        self.wait()

    def show_dct_intuiton(self, graph, dots, array_mob_dct_symbols):
        original_smaller_wave = VGroup(graph.copy().scale(0.7), dots.copy().scale(0.7))
        original_smaller_wave.move_to(DOWN * 2.5).to_edge(LEFT * 2)
        surround_rect = SurroundingRectangle(original_smaller_wave)

        original_wave_component = VGroup(original_smaller_wave, surround_rect)

        equals = MathTex("=")

        plus = MathTex("+")

        ellipses = MathTex(r"\cdots")

        cosine_0 = self.make_cosine_component_with_weight(0, 1)

        cosine_1 = self.make_cosine_component_with_weight(1, 2)

        cosine_7 = self.make_cosine_component_with_weight(7, 7)

        intuition_equation = VGroup(
            original_wave_component,
            equals,
            cosine_0,
            plus,
            cosine_1,
            plus.copy(),
            ellipses,
            plus.copy(),
            cosine_7
        ).arrange(RIGHT).move_to(DOWN * 2.6)

        self.play(
            TransformFromCopy(graph, original_smaller_wave[0]),
            TransformFromCopy(dots, original_smaller_wave[1]),
        )

        self.play(
            Create(surround_rect)
        )
        self.wait()

        self.play(
            Write(equals)
        )
        self.wait()

        transforms = self.get_transforms_for_coefficients(array_mob_dct_symbols[1], [cosine_0[0], cosine_1[0], cosine_7[0]], ellipses)

        self.play(
            FadeIn(cosine_0[1]),
            FadeIn(cosine_0[2]),
            FadeIn(cosine_1[1]),
            FadeIn(cosine_1[2]),
            FadeIn(cosine_7[1]),
            FadeIn(cosine_7[2]),
            FadeIn(intuition_equation[3]),
            FadeIn(intuition_equation[5]),
            FadeIn(intuition_equation[6]),
            FadeIn(intuition_equation[7]),
            *transforms,
            run_time=2
        )
        self.wait()

    def get_transforms_for_coefficients(self, array_mob_dct_symbols, new_weights, ellipses):
        transforms = []
        for i, element in enumerate(array_mob_dct_symbols):
            if i not in [0, 1, 7]:
                new_element = element.copy().move_to(ellipses.get_center()).set_stroke(opacity=0).set_fill(opacity=0)
            elif i == 7:
                new_element = new_weights[2]
            else:
                new_element = new_weights[i]
            transforms.append(TransformFromCopy(element, new_element))
        return transforms

    def make_cosine_component_with_weight(self, index, k):
        graph, _ = self.get_cosine_wave(lambda x: np.cos(x * k))
        text = MathTex(f"C_{index}")
        graph[0].set_stroke(opacity=0.3)
        text.scale(1.5).move_to(graph.get_center())
        weight_cosine = MathTex(f"X_{index}")
        weight_cosine.next_to(graph, LEFT, buff=SMALL_BUFF)
        return VGroup(weight_cosine, graph, text).scale(0.75)

    def experiment_with_cosine(self):
        ax, graph, cosine_label = self.introduce_cosine(1)

        cosine_group = VGroup(ax, graph, cosine_label)

        question = self.show_input_to_dct(cosine_group)

        cosine_graph_component = self.show_sampling_scheme(cosine_group, question)

    def introduce_cosine(self, k):
        ax, graph  = self.get_cosine_wave_with_ax(lambda x: np.cos(k * x))
        self.play(
            Write(ax)
        )
        self.wait()

        self.play(
            Create(graph)
        )
        self.wait()

        cosine_label = MathTex(r"y = \cos(x)").scale(1.2)
        cosine_label.next_to(graph, DOWN)
        self.play(
            Write(cosine_label)
        )
        self.wait()
        return ax, graph, cosine_label

    def show_input_to_dct(self, cosine_group):
        self.play(
            cosine_group.animate.scale(0.5).shift(LEFT * 3.5)
        )
        self.wait()

        right_arrow = MathTex(r"\Rightarrow").scale(1.5)
        right_arrow.next_to(cosine_group[0], RIGHT)

        right_arrow.shift(RIGHT * -right_arrow.get_center()[0])

        dct_component_label = Tex(r"DCT($y$)")
        dct_component_label.next_to(right_arrow, UP)

        arrow_and_label = VGroup(right_arrow, dct_component_label)

        box_around_cosine = SurroundingRectangle(cosine_group, color=REDUCIBLE_VIOLET, buff=SMALL_BUFF)

        dct_mystery_component, box_around_dct = self.get_dct_mystery_component()

        arrow_and_label.move_to((box_around_cosine.get_right() + box_around_dct.get_left()) / 2)
        
        self.play(
            Write(right_arrow),
            Write(dct_component_label),
        )
        self.wait()

        question_mark = Tex("?").scale(4)

        question_mark.move_to(dct_mystery_component.get_center())

        self.play(
            FadeIn(dct_mystery_component),
            FadeIn(question_mark),
            Create(box_around_dct)
        )
        self.wait()
        problem = Tex("Problem: we need sampled points on our cosine wave")
        problem.move_to(UP * 3.5)

        question = Tex("How should we sample the cosine function?").move_to(problem.get_center())

        self.play(
            Create(box_around_cosine),
            Write(problem)
        )
        self.wait()

        self.play(
            FadeOut(right_arrow),
            FadeOut(dct_component_label),
            FadeOut(dct_mystery_component),
            FadeOut(question_mark),
            FadeOut(box_around_cosine),
            FadeOut(problem),
            FadeOut(box_around_dct),
            cosine_group.animate.scale(2).shift(RIGHT * 3.5),
            ReplacementTransform(problem, question)
        )

        return question

    def get_dct_mystery_component(self):
        min_x, min_y = -80, 80
        random_dct_row = np.array([np.random.uniform(min_x, min_y) for _ in range(8)])
        dct_ax = self.get_random_dct_axis(random_dct_row, min_x, min_y)

        dct_graph, dct_points = self.plot_row_values(dct_ax, random_dct_row, color=REDUCIBLE_PURPLE)

        vertical_lines = self.get_vertical_lines_from_points(dct_ax, dct_points)
        dct_graph_components = VGroup(dct_ax, dct_graph, dct_points, vertical_lines).scale(1.1).move_to(RIGHT * 3.7 + DOWN * 0.3)
        surround_rect = SurroundingRectangle(dct_graph_components, color=REDUCIBLE_PURPLE)
        # dct_graph_components.set_stroke(opacity=0.5).set_fill(opacity=0.5)

        return dct_graph_components.fade(0.8), surround_rect

    def get_random_dct_axis(self, dct_row, min_y, max_y):
        ax = Axes(
            x_range=[0, dct_row.shape[0] - 1, 1],
            y_range=[min_y, max_y, 1],
            y_length=3,
            x_length=4.375,
            tips=False,
            axis_config={"include_numbers": True, "include_ticks": False},
            x_axis_config={"numbers_to_exclude": list(range(1, dct_row.shape[0] + 1))},
            y_axis_config={"numbers_to_exclude": list(range(min_y, max_y + 1))},
        )
        return ax

    def show_sampling_scheme(self, cosine_group, question):
        ax, graph, cosine_label = cosine_group
        dots = self.get_dots(ax, graph, 1, scale=1)
        line_intervals = self.get_intervals(ax, graph)
        ticks = self.get_ticks_for_x_axis()
        vertical_lines = self.get_cosine_vertical_lines_from_points(ax, dots, 1)

        self.play(
            *[GrowFromCenter(line) for line in line_intervals],
            *[Write(tick) for tick in ticks]
        )
        self.wait()

        self.play(
            LaggedStartMap(Create, vertical_lines)
        )
        self.wait()

        self.play(
            LaggedStartMap(GrowFromCenter, dots),
        )
        self.wait()

        labels = self.show_sample_x_vals(vertical_lines)

        cosine_sampling_math = MathTex(r"y_n = \cos \left[\frac{(2n+1) \pi}{16}\right]", r"\quad n \in \{0, 1, \ldots , 7\}").scale(0.9)

        cosine_sampling_math.move_to(cosine_label.get_center())

        self.play(
            ReplacementTransform(cosine_label, cosine_sampling_math)
        )
        self.wait()

        cosine_sampling_general = MathTex(r"y_n = \cos \left[\frac{(2n+1) \pi}{2N}\right]", r"\quad n \in \{0, 1, \ldots , N - 1\}").scale(0.9)

        cosine_sampling_general.move_to(cosine_label.get_center())

        self.play(
            ReplacementTransform(cosine_sampling_math, cosine_sampling_general)
        )
        self.wait()
        
        cosine_function_sample_group = VGroup(ax, graph, dots, line_intervals, ticks, vertical_lines)
        self.play(
            FadeOut(labels),
            FadeOut(question),
            FadeOut(cosine_sampling_general),
            cosine_function_sample_group.animate.scale(0.5).shift(LEFT * 3.5)
        )

        self.play(
            *[ScaleInPlace(dot, 1.4) for dot in dots],
        )
        self.wait()

        y_labels = [f"y_{i}" for i in range(8)]

        cosine_y_array = self.get_gen_array_obj(y_labels, color=REDUCIBLE_GREEN_LIGHTER)

        cosine_y_array.next_to(cosine_function_sample_group, UP * 2.5)

        self.play(
            FadeIn(cosine_y_array)
        )

        upper_brace = Brace(cosine_y_array, direction=UP)
        vector_y_label = MathTex(r"\vec{y}").next_to(upper_brace, UP)

        self.play(
            GrowFromCenter(upper_brace),
            Write(vector_y_label)
        )
        self.wait()

        right_arrow = MathTex(r"\Rightarrow").scale(1.5)
        right_arrow.next_to(cosine_function_sample_group[0], RIGHT)

        right_arrow.shift(RIGHT * -right_arrow.get_center()[0])

        dct_component_label = Tex(r"DCT($\vec{y}$)")
        dct_component_label.next_to(right_arrow, UP)

        arrow_and_label = VGroup(right_arrow, dct_component_label)

        box_around_cosine = SurroundingRectangle(cosine_function_sample_group, color=REDUCIBLE_VIOLET, buff=SMALL_BUFF)

        dct_mystery_component, box_around_dct = self.get_dct_mystery_component()

        arrow_and_label.move_to((box_around_cosine.get_right() + box_around_dct.get_left()) / 2)
        
        self.play(
            Write(right_arrow),
            Write(dct_component_label),
        )

        self.wait()

        question_mark = Tex("?").scale(4)

        question_mark.move_to(dct_mystery_component.get_center())

        self.play(
            FadeIn(dct_mystery_component),
            FadeIn(question_mark),
            Create(box_around_dct)
        )
        self.wait()

        dct_coeff_labels = [f"X_{i}" for i in range(8)]
        dct_coeff_array = self.get_gen_array_obj(dct_coeff_labels, color=REDUCIBLE_VIOLET)

        dct_coeff_array.next_to(box_around_dct, UP)

        self.play(
            FadeIn(dct_coeff_array)
        )
        self.wait()

        self.play(
            arrow_and_label.animate.shift(UP * 2.8 + LEFT * SMALL_BUFF)
        )

        self.play(
            Circumscribe(dct_coeff_array, color=REDUCIBLE_YELLOW)
        )
        self.wait()

        self.begin_various_cosine_experiments(cosine_function_sample_group, dct_mystery_component, question_mark, box_around_dct)

        self.clear()

        components = VGroup(*[self.get_generic_cosine_component(i) for i in range(8)])
        components.arrange_in_grid(rows=2, buff=0.5)

        surround_rect = SurroundingRectangle(components, buff=MED_SMALL_BUFF)

        self.play(
            FadeIn(components)
        )
        self.wait()

        self.play(
            Create(surround_rect)
        )
        self.wait()

        sigma = MathTex(r"\Sigma").scale(5).move_to(surround_rect.get_center())
        self.play(
            Write(sigma),
            components.animate.fade(0.7)
        )
        self.wait()

        down_arrow = MathTex(r"\Downarrow").next_to(surround_rect, DOWN).scale(1.5)

        self.play(
            Write(down_arrow)
        )
        self.wait()

        def get_random_pixel_row():
            random_values = np.array([np.random.uniform(0, 255) for _ in range(8)])
            random_pix_mob = self.make_row_of_pixels(random_values, height=0.8)
            random_pix_mob.next_to(down_arrow, DOWN)
            return random_pix_mob

        random_pix_mob = get_random_pixel_row()

        self.play(
            FadeIn(random_pix_mob)
        )
        self.wait()

        for _ in range(20):
            random_pix_mob.become(get_random_pixel_row())
            self.wait(0.5)

    def begin_various_cosine_experiments(self, cosine_function_sample_group, dct_mystery_component, question_mark, box_around_dct):
        frequency_tracker = ValueTracker(1)
        amplitude_tracker = ValueTracker(1)
        y_intercept_tracker = ValueTracker(0)

        def get_input_points():
            k = frequency_tracker.get_value()
            a = amplitude_tracker.get_value()
            b = y_intercept_tracker.get_value()
            return np.array([a * np.cos((j * 2 + 1) * k * np.pi / 16) + b for j in range(8)])

        def get_dct_values():
            input_points = get_input_points()
            return dct_1d(input_points)

        def get_dots(ax, graph, scale=1, color=REDUCIBLE_YELLOW):
            x_points = [(j * 2 + 1) * np.pi / 16 for j in range(8)]
            y_points = get_input_points()
            points = [ax.coords_to_point(x, y) for x, y in zip(x_points, y_points)]
            dots = VGroup(*[Dot().scale(scale).set_color(color).move_to(p) for p in points])
            return dots

        def get_cosine_vertical_lines_from_points(ax, points, color=REDUCIBLE_VIOLET):
            x_points = [(j * 2 + 1) * np.pi / 16 for j in range(8)]
            x_axis_points = [ax.x_axis.n2p(p) for p in x_points]
            # print([(start_point, end.get_center()) start_point, end in zip(x_points, points)])
            vertical_lines = [Line(start_point, end.get_center()).set_stroke(color=color, width=8) for start_point, end in zip(x_axis_points, points)]
            return VGroup(*vertical_lines)

        min_cos_y, max_cos_y = -5, 5
        def get_input_cosine_graph():
            nonlocal min_cos_y, max_cos_y
            scale = 0.5
            input_points = get_input_points()
            #TODO: Fix this logic
            if max(input_points) > max_cos_y or min(input_points) < min_cos_y:
                max_cos_y = int((max(input_points) * 3) // 1 + 1)
                min_cos_y = int((min(input_points) * 3) // 1 - 1)
                max_cos_y = max(abs(max_cos_y), abs(min_cos_y))
                min_cos_y = -max_cos_y
            
                if max_cos_y > 127:
                    max_cos_y = 128
                    min_cos_y = -128

            location = cosine_function_sample_group.get_center()
            k = frequency_tracker.get_value()
            a = amplitude_tracker.get_value()
            b = y_intercept_tracker.get_value()
            ax, graph  = self.get_cosine_wave_with_ax(lambda x: a * np.cos(x * k) + b, min_y=min_cos_y, max_y=max_cos_y)
            
            dots = get_dots(ax, graph)
            for dot in dots:
                dot.scale(1.4)

            line_intervals = self.get_intervals(ax, graph)
            ticks = self.get_ticks_for_x_axis()
            vertical_lines = get_cosine_vertical_lines_from_points(ax, dots)

            new_cosine_function_sample_group = VGroup(ax, graph, dots, line_intervals, ticks, vertical_lines)
            new_cosine_function_sample_group.scale(scale).move_to(location)
            for label in ax.y_axis.numbers:
                label.scale(2).shift(LEFT * SMALL_BUFF)

            new_cosine_function_sample_group.move_to(location)

            return new_cosine_function_sample_group

        min_y, max_y = -5, 5
        def get_output_dct_graph():
            nonlocal min_y, max_y
            dct_values = get_dct_values()
            # TODO: fix this logic
            if max(dct_values) > max_y or min(dct_values) < min_y:
                max_y = int((max(dct_values) * 3) // 1 + 1)
                min_y = int((min(dct_values) * 3) // 1 - 1)
                max_y = max(abs(max_y), abs(min_y))
                min_y = -max_y

                if max_y > 250:
                    max_y = 250
                    min_y = -250
            dct_ax = self.get_dct_axis(dct_values, min_y, max_y)

            dct_graph, dct_points = self.plot_row_values(dct_ax, dct_values, color=REDUCIBLE_PURPLE)

            vertical_lines = self.get_vertical_lines_from_points(dct_ax, dct_points)
            dct_graph_components = VGroup(dct_ax, dct_graph, dct_points, vertical_lines).scale(1.1).move_to(RIGHT * 3.7 + DOWN * 0.3)
            return dct_graph_components

        output_dct_graph = always_redraw(get_output_dct_graph)

        self.play(
            FadeOut(question_mark),
            FadeOut(box_around_dct),
            ReplacementTransform(dct_mystery_component, output_dct_graph),
        )
        self.wait()

        input_cosine_graph = always_redraw(get_input_cosine_graph)
        self.play(
            ReplacementTransform(cosine_function_sample_group, input_cosine_graph)
        )
        self.wait()

        self.play(
            amplitude_tracker.animate.set_value(50),
            run_time=4
        )
        self.wait()

        self.play(
            amplitude_tracker.animate.set_value(-50),
            run_time=4
        )
        self.wait()

        self.play(
            amplitude_tracker.animate.set_value(50),
            run_time=2
        )
        self.wait()

        original_pixel_space_rep = self.get_pixel_space_representation()
        original_pixel_space_rep.move_to(DOWN * 2.8)
        lower_bound = Integer(0).next_to(original_pixel_space_rep, LEFT)
        upper_bound = Integer(255).next_to(original_pixel_space_rep, RIGHT)
        self.play(
            FadeIn(original_pixel_space_rep),
            FadeIn(lower_bound),
            FadeIn(upper_bound)
        )
        self.wait()

        new_pixel_space_rep = original_pixel_space_rep.copy().next_to(original_pixel_space_rep, DOWN)
        new_lower_bound = Integer(-128).next_to(new_pixel_space_rep, LEFT)
        new_upper_bound = Integer(127).next_to(new_pixel_space_rep, RIGHT)
        self.play(
            FadeIn(new_pixel_space_rep),
            TransformFromCopy(lower_bound, new_lower_bound),
            TransformFromCopy(upper_bound, new_upper_bound)
        )
        self.wait()

        def get_pixel_rep_of_cosine():
            input_points = get_input_points() + 128
            pixel_array_mob = self.make_row_of_pixels(input_points, height=SMALL_BUFF*6)
            pixel_array_mob.next_to(input_cosine_graph, DOWN).shift(RIGHT * SMALL_BUFF * 2)
            return pixel_array_mob

        pixel_row_mob = always_redraw(get_pixel_rep_of_cosine)
        self.play(
            FadeIn(pixel_row_mob),
            FadeOut(new_pixel_space_rep),
            FadeOut(original_pixel_space_rep),
            FadeOut(lower_bound),
            FadeOut(new_lower_bound),
            FadeOut(upper_bound),
            FadeOut(new_upper_bound)
        )
        self.wait()

        self.play(
            amplitude_tracker.animate.set_value(120),
            run_time=3,
        )
        self.wait()

        self.play(
            amplitude_tracker.animate.set_value(-120),
            run_time=3
        )

        self.play(
            amplitude_tracker.animate.set_value(80),
            run_time=3
        )
        self.wait()

        self.play(
            y_intercept_tracker.animate.set_value(45),
            run_time=2
        )

        self.wait()

        self.play(
            y_intercept_tracker.animate.set_value(-45),
            run_time=4
        )

        self.wait()

        self.play(
            y_intercept_tracker.animate.set_value(0),
            run_time=2
        )

        self.wait()

        self.play(
            frequency_tracker.animate.set_value(2),
            run_time=6,
            rate_func=linear,
        )
        self.wait()

        cosine_2x = MathTex(r"\cos(2x)", r" \rightarrow X_2")
        cosine_2x.move_to(DOWN * 3.2)
        self.play(
            Write(cosine_2x[0])
        )
        self.wait()

        self.play(
            Write(cosine_2x[1])
        )
        self.wait()

        self.play(
            amplitude_tracker.animate.set_value(120),
            run_time=2,
        )

        self.play(
            amplitude_tracker.animate.set_value(80),
            run_time=2,
        )

        self.play(
            y_intercept_tracker.animate.set_value(45),
            run_time=2
        )

        self.play(
            y_intercept_tracker.animate.set_value(-45),
            run_time=2
        )

        self.play(
            y_intercept_tracker.animate.set_value(0),
            run_time=2
        )
        self.wait()

        self.play(
            frequency_tracker.animate.set_value(3),
            run_time=3,
            rate_func=linear,
        )
        cosine_3x = MathTex(r"\cos(3x)", r" \rightarrow X_3")
        cosine_3x.move_to(DOWN * 3.2)
        self.play(
            ReplacementTransform(cosine_2x, cosine_3x)
        )
        self.wait()

        self.play(
            frequency_tracker.animate.set_value(4),
            run_time=2,
            rate_func=linear,
        )

        cosine_kx = MathTex(r"\cos(kx)", r" \rightarrow X_k")
        cosine_kx.move_to(DOWN * 3.2)
        self.play(
            ReplacementTransform(cosine_3x, cosine_kx)
        )
        self.wait()

        self.play(
            frequency_tracker.animate.set_value(5),
            run_time=2,
            rate_func=linear,
        )
        self.wait()

        self.play(
            frequency_tracker.animate.set_value(6),
            run_time=2,
            rate_func=linear,
        )
        self.wait()

        self.play(
            frequency_tracker.animate.set_value(7),
            run_time=2,
            rate_func=linear,
        )
        self.wait()

        self.play(
            y_intercept_tracker.animate.set_value(45),
            run_time=2
        )

        self.play(
            y_intercept_tracker.animate.set_value(-45),
            run_time=2
        )

        self.play(
            y_intercept_tracker.animate.set_value(0),
            run_time=2
        )
        self.wait()

        self.play(
            frequency_tracker.animate.set_value(0),
            run_time=4,
            rate_func=linear,
        )


        self.play(
            amplitude_tracker.animate.set_value(120),
            run_time=2
        )

        self.play(
            amplitude_tracker.animate.set_value(-120),
            run_time=2
        )

        self.play(
            amplitude_tracker.animate.set_value(80),
            run_time=2
        )

        self.wait()

    def get_generic_cosine_component(self, frequency, cosine_scale=0.5, general_scale=0.5):
        frequency_tracker = ValueTracker(frequency)
        amplitude_tracker = ValueTracker(120)
        y_intercept_tracker = ValueTracker(0)

        def get_input_points():
            k = frequency_tracker.get_value()
            a = amplitude_tracker.get_value()
            b = y_intercept_tracker.get_value()
            return np.array([a * np.cos((j * 2 + 1) * k * np.pi / 16) + b for j in range(8)])

        def get_dct_values():
            input_points = get_input_points()
            return dct_1d(input_points)

        def get_dots(ax, graph, scale=1, color=REDUCIBLE_YELLOW):
            x_points = [(j * 2 + 1) * np.pi / 16 for j in range(8)]
            y_points = get_input_points()
            points = [ax.coords_to_point(x, y) for x, y in zip(x_points, y_points)]
            dots = VGroup(*[Dot().scale(scale).set_color(color).move_to(p) for p in points])
            return dots

        def get_cosine_vertical_lines_from_points(ax, points, color=REDUCIBLE_VIOLET):
            x_points = [(j * 2 + 1) * np.pi / 16 for j in range(8)]
            x_axis_points = [ax.x_axis.n2p(p) for p in x_points]
            vertical_lines = [Line(start_point, end.get_center()).set_stroke(color=color, width=8) for start_point, end in zip(x_axis_points, points)]
            return VGroup(*vertical_lines)

        input_points = get_input_points()
        #TODO: Fix this logic
        max_cos_y, min_cos_y = 120, -120

        k = frequency_tracker.get_value()
        a = amplitude_tracker.get_value()
        b = y_intercept_tracker.get_value()
        ax, graph  = self.get_cosine_wave_with_ax(lambda x: a * np.cos(x * k) + b, min_y=min_cos_y, max_y=max_cos_y, include_end_y_axis_nums=False)
        
        dots = get_dots(ax, graph)
        for dot in dots:
            dot.scale(1.4)

        line_intervals = self.get_intervals(ax, graph)
        ticks = self.get_ticks_for_x_axis()
        vertical_lines = get_cosine_vertical_lines_from_points(ax, dots)

        new_cosine_function_sample_group = VGroup(ax, graph, line_intervals, ticks, vertical_lines, dots)
        new_cosine_function_sample_group.scale(cosine_scale)

        def get_pixel_rep_of_cosine():
            input_points = get_input_points() + 128
            pixel_array_mob = self.make_row_of_pixels(input_points, height=SMALL_BUFF*6.3)
            pixel_array_mob.next_to(new_cosine_function_sample_group, UP, aligned_edge=LEFT)
            return pixel_array_mob

        pixel_space_mob_cos = get_pixel_rep_of_cosine()
        x_comp = MathTex(f"X_{frequency}").scale(1.5).next_to(pixel_space_mob_cos, UP)

        return VGroup(new_cosine_function_sample_group, pixel_space_mob_cos, x_comp).scale(general_scale)

    def get_cosine_wave(self, cosine_function, color=REDUCIBLE_VIOLET):
        ax = Axes(
            x_range=[0, np.pi],
            y_range=[-1, 1],
            x_length=2,
            y_length=2,
        )

        graph = ax.plot(cosine_function).set_color(REDUCIBLE_VIOLET)

        box = SurroundingRectangle(graph, color=REDUCIBLE_VIOLET)
        return VGroup(graph, box), ax

    def get_cosine_wave_with_ax(self, cosine_function, min_y=-1, max_y=1, color=REDUCIBLE_VIOLET, include_end_y_axis_nums=True):
        if include_end_y_axis_nums:
            y_range_exclude = range(min_y + 1, max_y)
        else:
            y_range_exclude = range(min_y, max_y + 1)
        ax = Axes(
            x_range=[0, np.pi],
            y_range=[min_y, max_y],
            x_length=10,
            y_length=5.5,
            tips=False,
            x_axis_config={"include_numbers": False, "include_ticks": False},
            y_axis_config={"include_numbers": True, "numbers_to_exclude": y_range_exclude, "include_ticks": False}
        )

        graph = ax.plot(cosine_function).set_color(color)
        pi_label = MathTex(r"\pi")
        pi_label.next_to(ax.x_axis, DOWN, aligned_edge=RIGHT)
        ax.add(pi_label)

        group = VGroup(ax, graph)

        return group

    def get_x_y_points(self, k):
        x_points = [(j * 2 + 1) * np.pi / 16 for j in range(8)]
        y_points = [np.cos(x * k) for x in x_points]
        return x_points, y_points

    def get_dots(self, ax, graph, k, color=REDUCIBLE_YELLOW, scale=0.7):
        x_points, y_points = self.get_x_y_points(k)
        points = [ax.coords_to_point(x, y) for x, y in zip(x_points, y_points)]

        dots = VGroup(*[Dot().scale(scale).set_color(color).move_to(p) for p in points])
        return dots

    def get_intervals(self, ax, graph):
        proportions = np.arange(0, np.pi + 0.0001, np.pi / 8)
        lines = []
        for i in range(len(proportions) - 1):
            start, end = proportions[i], proportions[i + 1]
            start_point, end_point = ax.x_axis.n2p(start), ax.x_axis.n2p(end)
            line = Line(start_point, end_point).set_stroke(width=5)
            if i % 2 == 0:
                line.set_color(REDUCIBLE_GREEN_LIGHTER)
            else:
                line.set_color(REDUCIBLE_GREEN_DARKER)

            lines.append(line)

        return VGroup(*lines)

    def show_sample_x_vals(self, vertical_lines):
        labels = VGroup(*[MathTex(r"\frac{\pi}{16}").scale(0.7)] + [MathTex(r"\frac{" + str(2 * i + 1) + r"\pi}{16}").scale(0.6) for i in range(1, len(vertical_lines))])
        for label, line in zip(labels, vertical_lines):
            direction = normalize(line.get_start() - line.get_end())
            direction = np.array([int(c) for c in direction])
            label.next_to(line, direction)

        self.play(
            FadeIn(labels)
        )
        self.wait()
        return labels

    def get_cosine_vertical_lines_from_points(self, ax, points, k, color=REDUCIBLE_VIOLET):
        x_points = [ax.x_axis.n2p(p) for p in self.get_x_y_points(k)[0]]
        vertical_lines = [Line(start_point, end.get_center()).set_stroke(color=color, width=8) for start_point, end in zip(x_points, points)]
        return VGroup(*vertical_lines)

    def get_ticks_for_x_axis(self):
        ax = Axes(
            x_range=[0, np.pi, np.pi / 8],
            y_range=[-1, 1],
            x_length=10,
            y_length=5.5,
            tips=False,
            x_axis_config={"include_numbers": False, "include_ticks": True},
            y_axis_config={"include_numbers": True, "numbers_to_exclude": [0], "include_ticks": False}
        )
        return ax.x_axis.ticks

    def make_component(self, text, color=REDUCIBLE_YELLOW, scale=0.8):
        # geometry is first index, Tex is second index
        text_mob = Tex(text).scale(scale)
        rect = Rectangle(color=color, height=1.1, width=3)
        return VGroup(rect, text_mob)

    def get_pixel_space_representation(self, height=SMALL_BUFF*5, row_length=6):
        num_pixels = 256
        row_values = range(256)
        pixel_row_mob = VGroup(
            *[
                Rectangle(height=height, width=row_length / num_pixels)
                .set_stroke(color=gray_scale_value_to_hex(value))
                .set_fill(color=gray_scale_value_to_hex(value), opacity=1)
                for value in row_values
            ]
        ).arrange(RIGHT, buff=0)
        surround_rect = SurroundingRectangle(pixel_row_mob, color=REDUCIBLE_GREEN_LIGHTER, buff=0)
        return VGroup(pixel_row_mob, surround_rect)

class MathematicallyDefineDCT(RevisedMotivateDCT):
    def construct(self):
        image_mob = ImageMobject("dog").move_to(UP * 2)
        block_image, pixel_grid, block = self.get_pixel_block(image_mob, 126, 126)
        row = 7
        print(block[:, :, 0])
        print(f"Block row: {row}\n", block[:, :, 0][row])
        pixel_row_mob, row_values = self.get_pixel_row_mob(block, row, height=0.6)
        print("Selected row values\n", row_values)
        pixel_row_mob.move_to(UP * 3)

        row_values_centered = format_block(row_values)
        print('After centering\n', row_values_centered)

        dct_row_pixels = dct_1d(row_values_centered)

        ax, graph, dots = self.draw_image_graph(dct_row_pixels)

        self.show_graph(ax, graph, dots, pixel_row_mob, animate=False)


        entire_group = self.get_broad_dct_components(ax, graph, dots, pixel_row_mob, dct_row_pixels)

        entire_group.shift(UP * 1)

        pixel_space_group, forward_arrow, dct_space_group = entire_group
        self.play(
            FadeIn(pixel_space_group),
            FadeIn(forward_arrow),
            FadeIn(dct_space_group)
        )
        self.wait()

        pixel_array_symbols = pixel_space_group[-1]
        dct_array_mob_symbols = dct_space_group[-1]

        self.play(
            Circumscribe(pixel_array_symbols)
        )
        self.wait()


        self.play(
            Circumscribe(dct_array_mob_symbols)
        )
        self.wait()

        dct_math = MathTex(r"X_k = ", r"\sum_{n=0}^{N-1} x_n", r"\cos \left[\frac{(2n+1) \pi k}{2N}\right]")

        dct_math.move_to(DOWN * 2.5)

        self.play(
            Write(dct_math[0])
        )
        self.wait()

        self.play(
            FadeIn(dct_math[1]),
            FadeIn(dct_math[2])
        )
        self.wait()

        self.play(
            dct_math[0].animate.set_fill(color=None, opacity=0.5),
            dct_math[1].animate.set_fill(color=None, opacity=0.5)
        )
        self.wait()

        self.play(
            Indicate(dct_math[2][-5])
        )
        self.play(
            Indicate(dct_math[2][-5])
        )
        self.wait()

        self.play(
            dct_math[0].animate.set_fill(color=None, opacity=1),
            dct_math[1].animate.set_fill(color=None, opacity=1),
        )

        self.play(
            dct_math.animate.to_edge(LEFT * 2)
        )

        self.wait()

        brace_up = Brace(pixel_array_symbols, direction=UP)

        self.play(
            GrowFromCenter(brace_up)
        )
        x_vec = MathTex(r"\vec{x}").next_to(brace_up, UP)

        self.play(
            Write(x_vec)
        )
        self.wait()

        C_k_values = [r" \cos (\frac{\pi}{16} \cdot " + str(2 * i + 1) + r" \cdot k)" for i in [0, 1, 2, 7]]
        C_k_values.insert(-1, r"\vdots")
        C_k_vec_def = self.make_column_vector(C_k_values, v_buff=1.2)

        self.play(
            FadeOut(forward_arrow),
            FadeOut(dct_space_group)
        )
        self.wait()

        self.play(
            Circumscribe(dct_math[2])
        )
        self.wait()

        C_k = MathTex(r"\vec{C}_k = ")
        C_k_vec_def[0][3].shift(UP * SMALL_BUFF * 1.5)

        C_k_vec_group = VGroup(C_k, C_k_vec_def).arrange(RIGHT).move_to(RIGHT * 3.5 + UP * 1.3)
        self.play(
            FadeIn(C_k_vec_group)
        )
        self.wait()

        double_arrow = MathTex(r"\Longleftrightarrow").next_to(dct_math, RIGHT * 2)

        dot_product_def = MathTex(r"X_k = \vec{C}_k^T \vec{x}").next_to(double_arrow, RIGHT * 2)

        dct_equivalence = VGroup(dct_math, double_arrow, dot_product_def)
        self.play(
            Write(double_arrow)
        )
        self.wait()

        self.play(
            Write(dot_product_def)
        )

        self.wait()

        self.play(
            FadeOut(pixel_space_group),
            FadeOut(C_k_vec_group),
            FadeOut(brace_up),
            FadeOut(x_vec),
            dct_equivalence.animate.move_to(UP * 3.2),
            run_time=2

        )
        self.wait()

        forward_dct_matrix_eq = self.get_forward_transform_matrix_eq().scale(1.2)
        self.play(
            FadeIn(forward_dct_matrix_eq)
        )
        self.wait()

        orthoganility = MathTex(r"\vec{C}_i^T \vec{C}_j = 0 \quad \forall i \neq j")
        orthoganility.next_to(forward_dct_matrix_eq, DOWN)

        self.play(
            FadeIn(orthoganility)
        )
        self.wait()

        entire_group.shift(DOWN * 1)

        self.clear()

        self.add(entire_group)

        forward_dct_matrix_eq.scale(1 / 1.2)

        self.show_invertibility(pixel_space_group, forward_arrow, dct_space_group, forward_dct_matrix_eq, graph, dots)

    def show_invertibility(self, pixel_space_group, forward_arrow, dct_space_group, forward_dct_matrix_eq, graph, dots):
        original_graph, original_dots = graph.copy(), dots.copy()
        invertibility = Tex("Invertibility").move_to(UP * 3.5)

        forward_dct_group = VGroup(pixel_space_group, forward_arrow, dct_space_group)

        self.play(
            Write(invertibility)
        )
        self.wait()

        surround_rect_forward = SurroundingRectangle(forward_dct_group, color=REDUCIBLE_YELLOW, buff=MED_SMALL_BUFF)

        self.play(
            Create(surround_rect_forward)
        )
        self.wait()

        self.play(
            forward_dct_group.animate.scale(0.65).shift(UP * 1.5),
            surround_rect_forward.animate.scale(0.65).shift(UP * 1.5)
        )

        self.wait()
        shift_down = DOWN * 3.5
        new_idct_group_left = dct_space_group.copy().move_to(pixel_space_group.get_center()).shift(shift_down)
        new_idct_arrow = forward_arrow.copy().shift(shift_down)
        new_idct_group_right = pixel_space_group.copy().move_to(dct_space_group.get_center()).shift(shift_down)

        self.play(
            TransformFromCopy(dct_space_group, new_idct_group_left)
        )
        self.wait()

        self.play(
            TransformFromCopy(forward_arrow, new_idct_arrow)
        )
        self.wait()

        self.play(
            TransformFromCopy(pixel_space_group, new_idct_group_right)
        )
        self.wait()

        inverse_dct_group = VGroup(new_idct_group_left, new_idct_arrow, new_idct_group_right)

        surround_rect_inverse = SurroundingRectangle(inverse_dct_group, color=REDUCIBLE_PURPLE, buff=MED_SMALL_BUFF)

        self.play(
            Create(surround_rect_inverse)
        )
        self.wait()

        shift_left = LEFT * 2

        self.play(
            surround_rect_forward.animate.shift(shift_left),
            surround_rect_inverse.animate.shift(shift_left),
            forward_dct_group.animate.shift(shift_left),
            inverse_dct_group.animate.shift(shift_left),
        )
        self.wait()

        forward_transform = MathTex(r"\vec{X} = M \vec{x}")
        forward_transform.next_to(surround_rect_forward, RIGHT).shift(RIGHT * 1.3)
        inverse_transform = MathTex(r"\vec{x} = M^{-1} \vec{X}")
        inverse_transform.next_to(surround_rect_inverse, RIGHT).shift(RIGHT * 1)

        forward_dct_text = Tex("DCT").scale(1.2)
        inverse_dct_text = Tex("Inverse DCT").scale(1.2)

        forward_dct_text.next_to(forward_transform, UP)
        inverse_dct_text.next_to(inverse_transform, UP)

        forward_transform_group = VGroup(forward_dct_text, forward_transform).arrange(DOWN)
        forward_transform_group.next_to(surround_rect_forward, RIGHT).shift(RIGHT * 1)

        inverse_transform_group = VGroup(inverse_dct_text, inverse_transform).arrange(DOWN)
        inverse_transform_group.next_to(surround_rect_inverse, RIGHT).shift(RIGHT * 0.4)

        self.play(
            FadeIn(forward_dct_text),
            FadeIn(inverse_dct_text)
        )
        self.wait()

        self.play(
            FadeIn(forward_transform),
            FadeIn(inverse_transform)
        )
        self.wait()


        inverse_dct_matrix_eq = self.get_inverse_tranform_matrix_eq()
        inverse_dct_matrix_eq.scale(0.8).move_to(inverse_dct_group.get_center())
        forward_dct_matrix_eq.scale(0.8).move_to(forward_dct_group.get_center())

        self.play(
            forward_dct_group.animate.fade(0.9),
            FadeIn(forward_dct_matrix_eq),
        )
        self.wait()

        self.play(
            inverse_dct_group.animate.fade(0.9),
            FadeIn(inverse_dct_matrix_eq)
        )
        self.wait()

        asterik = Tex("(*)").next_to(inverse_dct_matrix_eq, RIGHT)
        asterik[0][1].shift(DOWN * SMALL_BUFF)
        note = Tex("*with additional normalization").scale(0.6)

        note.next_to(inverse_transform, DOWN * 2)

        self.add(asterik, note)
        self.wait()

        self.play(
            FadeOut(invertibility),
            FadeOut(asterik),
            FadeOut(note),
            FadeOut(surround_rect_inverse),
            FadeOut(surround_rect_forward),
            inverse_dct_group.animate.fade(1),
            forward_dct_group.animate.fade(1),
            FadeOut(forward_dct_text),
            FadeOut(forward_transform),
            FadeOut(inverse_transform),
            FadeOut(forward_dct_matrix_eq),
            inverse_dct_text.animate.move_to(UP * 3.5),
            inverse_dct_matrix_eq.animate.scale(1 / 0.8).next_to(invertibility, DOWN),
            run_time=2
        )
        self.wait()

        column_split_sum = self.get_column_sum_matrix(inverse_dct_matrix_eq[0])
        column_split_sum.next_to(inverse_dct_matrix_eq, DOWN)

        self.play(
            FadeIn(column_split_sum)
        )

        self.wait()

        self.play(
            FadeOut(inverse_dct_matrix_eq),
            column_split_sum.animate.next_to(inverse_dct_text, DOWN * 2)
        )
        self.wait()

        self.play(
            column_split_sum[0].animate.set_color(REDUCIBLE_YELLOW),
            column_split_sum[2][1].animate.set_color(REDUCIBLE_VIOLET),
            column_split_sum[4][1].animate.set_color(REDUCIBLE_VIOLET),
            column_split_sum[8][1].animate.set_color(REDUCIBLE_VIOLET),
        )
        self.wait()

        cosine_sum_visual_rep = self.show_summing_different_cosine_waves(original_graph, original_dots)

        cosine_sum_visual_rep.scale(0.75).next_to(column_split_sum, DOWN * 3)

        self.play(
            FadeIn(cosine_sum_visual_rep)
        )
        self.wait()

    def show_summing_different_cosine_waves(self, graph, original_dots):
        first_freq, first_axes = self.get_cosine_wave(lambda x: 0.5)
        second_freq, second_axes = self.get_cosine_wave(lambda x: np.cos(x))
        last_freq, last_axes = self.get_cosine_wave(lambda x: np.cos(7 * x))

        first_freq_dots = self.get_dots(first_axes, first_freq, 0).move_to(first_freq.get_center())
        second_freq_dots = self.get_dots(second_axes, second_freq, 1)
        last_freq_dots = self.get_dots(last_axes, last_freq, 7)

        first_surround_rect = second_freq[1].copy().move_to(first_axes.get_center())
        right_shift = first_freq[1].get_center()[0] - first_surround_rect.get_center()[0]
        first_surround_rect.shift(RIGHT * right_shift)

        first_cosine_graph = VGroup(first_freq[0], first_surround_rect, first_freq_dots)
        second_cosine_graph = VGroup(second_freq, second_freq_dots)
        last_cosine_graph = VGroup(last_freq, last_freq_dots)
        
        X_0, X_1, X_7 = MathTex("X_0"), MathTex("X_1"), MathTex("X_7")

        first_cosine_comp = VGroup(X_0,  first_cosine_graph).arrange(RIGHT, buff=SMALL_BUFF)
        second_cosine_comp = VGroup(X_1,second_cosine_graph).arrange(RIGHT, buff=SMALL_BUFF)
        last_cosine_comp = VGroup(X_7, last_cosine_graph).arrange(RIGHT, buff=SMALL_BUFF)

        plus = MathTex("+")
        ellipses = MathTex(r"\cdots")
        equals = MathTex("=")

        original_graph = VGroup(graph, original_dots)
        surround_rect_original = SurroundingRectangle(original_graph, color=REDUCIBLE_YELLOW)
        original_graph = VGroup(original_graph, surround_rect_original)

        group = VGroup(original_graph, equals, first_cosine_comp, plus, second_cosine_comp, plus.copy(), ellipses, plus.copy(), last_cosine_comp).arrange(RIGHT)

        return group

    def get_column_sum_matrix(self, pixel_vec):
        X_0, X_1, X_7 = MathTex("X_0"), MathTex("X_1"), MathTex("X_7")
        c_0 = self.get_individual_column_vec(0)
        c_1 = self.get_individual_column_vec(1)
        c_7 = self.get_individual_column_vec(7)

        first_elem = VGroup(X_0, c_0).arrange(RIGHT, buff=SMALL_BUFF)
        second_elem = VGroup(X_1, c_1).arrange(RIGHT, buff=SMALL_BUFF)
        last_elem = VGroup(X_7, c_7).arrange(RIGHT, buff=SMALL_BUFF)

        plus = MathTex("+")
        equals = MathTex("=")
        cdots = MathTex(r"\cdots")

        column_split_equation = VGroup(
            pixel_vec.copy(),
            equals,
            first_elem,
            plus,
            second_elem,
            plus.copy(),
            cdots,
            plus.copy(),
            last_elem,
        ).arrange(RIGHT)

        return column_split_equation

    def get_matrix_m(self):
        row0 = self.get_cosine_row_tex(0)
        row1 = self.get_cosine_row_tex(1)
        vdots = MathTex(r"\vdots")
        row7 = self.get_cosine_row_tex(7)

        rows = VGroup(row0, row1, vdots, row7).arrange(DOWN).move_to(DOWN * 2)
        bracket_pair = MathTex("[", "]")
        bracket_pair.scale(2)
        bracket_v_buff = MED_SMALL_BUFF
        bracket_h_buff = MED_SMALL_BUFF
        bracket_pair.stretch_to_fit_height(rows.height + 2 * bracket_v_buff)
        l_bracket, r_bracket = bracket_pair.split()
        l_bracket.next_to(rows, LEFT, bracket_h_buff)
        r_bracket.next_to(rows, RIGHT, bracket_h_buff)
        brackets = VGroup(l_bracket, r_bracket)

        return VGroup(brackets, rows)

    def get_inverse_matrix_m(self):
        col0 = self.get_cosine_col_tex(0)
        col1 = self.get_cosine_col_tex(1)
        cdots = MathTex(r"\cdots")
        col7 = self.get_cosine_col_tex(7)

        cols = VGroup(col0, col1, cdots, col7).arrange(RIGHT, buff=0.7).move_to(DOWN * 2)
        bracket_pair = MathTex("[", "]")
        bracket_pair.scale(2)
        bracket_v_buff = MED_SMALL_BUFF
        bracket_h_buff = MED_SMALL_BUFF
        bracket_pair.stretch_to_fit_height(cols.height + 2 * bracket_v_buff)
        l_bracket, r_bracket = bracket_pair.split()
        l_bracket.next_to(cols, LEFT, bracket_h_buff)
        r_bracket.next_to(cols, RIGHT, bracket_h_buff)
        brackets = VGroup(l_bracket, r_bracket)

        return VGroup(brackets, cols)

    def make_column_vector(self, values, v_buff=0.6, scale=0.6):
        integer_values = []
        for value in values:
            if isinstance(value, str):
                integer_values.append(value)
            else:
                integer_values.append(int(value))
        vector = Matrix([[value] for value in integer_values], v_buff=v_buff, element_alignment_corner=DOWN)
        return vector.scale(scale)

    def get_individual_column_vec(self, index):
        cosine_col_text = self.get_cosine_col_tex(index)
        bracket_pair = MathTex("[", "]")
        bracket_pair.scale(2)
        bracket_v_buff = MED_SMALL_BUFF
        bracket_h_buff = MED_SMALL_BUFF
        bracket_pair.stretch_to_fit_height(cosine_col_text.height + 2 * bracket_v_buff)
        l_bracket, r_bracket = bracket_pair.split()
        l_bracket.next_to(cosine_col_text, LEFT, bracket_h_buff)
        r_bracket.next_to(cosine_col_text, RIGHT, bracket_h_buff)
        brackets = VGroup(l_bracket, r_bracket)

        return VGroup(brackets, cosine_col_text)

    def get_cosine_row_tex(self, index):
        text = MathTex(f"C_{index}^T").scale(0.8)
        left_arrow = Arrow(RIGHT * 2, ORIGIN, stroke_width=3, max_tip_length_to_length_ratio=0.15).next_to(text, LEFT).set_color(WHITE)
        right_arrow = Arrow(ORIGIN, RIGHT * 2, stroke_width=3, max_tip_length_to_length_ratio=0.15).next_to(text, RIGHT).set_color(WHITE)
        return VGroup(left_arrow, text, right_arrow)

    def get_cosine_col_tex(self, index):
        text = MathTex(f"C_{index}").scale(0.8)
        up_arrow = Arrow(ORIGIN, UP * 1.4, stroke_width=3, max_tip_length_to_length_ratio=0.15).next_to(text, UP).set_color(WHITE)
        down_arrow = Arrow(ORIGIN, DOWN * 1.4, stroke_width=3, max_tip_length_to_length_ratio=0.15).next_to(text, DOWN).set_color(WHITE)
        return VGroup(up_arrow, text, down_arrow)

    def get_forward_transform_matrix_eq(self):
        coeff_vector_values = [f"X_{i}" for i in range(8)]
        X_coeff_vec = self.make_column_vector(coeff_vector_values)
        equals = MathTex("=")
        dct_matrix = self.get_matrix_m()
        pixel_vector_values = [f"x_{i}" for i in range(8)]
        pixel_vec = self.make_column_vector(pixel_vector_values)

        matrix_equation = VGroup(X_coeff_vec, equals, dct_matrix, pixel_vec).arrange(RIGHT)

        return matrix_equation

    def get_inverse_tranform_matrix_eq(self):
        coeff_vector_values = [f"X_{i}" for i in range(8)]
        X_coeff_vec = self.make_column_vector(coeff_vector_values)
        equals = MathTex("=")
        idct_matrix = self.get_inverse_matrix_m()
        pixel_vector_values = [f"x_{i}" for i in range(8)]
        pixel_vec = self.make_column_vector(pixel_vector_values)

        matrix_equation = VGroup(pixel_vec, equals, idct_matrix, X_coeff_vec).arrange(RIGHT)

        return matrix_equation

    def get_broad_dct_components(self, ax, graph, dots, pixel_row_mob, dct_row_pixels):
        group = VGroup(ax, graph, dots, pixel_row_mob).move_to(LEFT * 3.5 + DOWN * 0.5)

        general_vals = [f"x_{i}" for i in range(len(pixel_row_mob))]
        array_mob_symbols = self.get_gen_array_obj(general_vals, length=pixel_row_mob.width, height=pixel_row_mob.height)
        array_mob_symbols.next_to(pixel_row_mob, UP)

        forward_arrow = MathTex(r"\Rightarrow").scale(1.5).shift(RIGHT * SMALL_BUFF * 3)

        pixel_space_group = VGroup(group, array_mob_symbols)

        dct_ax = self.get_dct_axis(dct_row_pixels, -80, 80)

        dct_graph, dct_points = self.plot_row_values(dct_ax, dct_row_pixels, color=REDUCIBLE_PURPLE)

        dct_graph_components = VGroup(dct_ax, dct_graph, dct_points).move_to(RIGHT * 3.5 + DOWN * 0.5)

        vertical_lines = self.get_vertical_lines_from_points(dct_ax, dct_points)

        general_dct_vals = [f"X_{i}" for i in range(len(pixel_row_mob))]

        array_mob_dct_symbols = self.get_gen_array_obj(general_dct_vals, length=pixel_row_mob.width + 0.5, height=pixel_row_mob.height + SMALL_BUFF, color=REDUCIBLE_VIOLET)
        array_mob_dct_symbols.next_to(pixel_row_mob, UP)
        shift_amount = dct_graph_components.get_center()[0] - array_mob_dct_symbols.get_center()[0]
        array_mob_dct_symbols.shift(RIGHT * (shift_amount + 0.3))

        dct_space_group = VGroup(dct_graph_components, vertical_lines, array_mob_dct_symbols)

        return VGroup(pixel_space_group, forward_arrow, dct_space_group)

class Introduce2DDCT(DCTExperiments):
    def construct(self):
        image_mob = ImageMobject("dog").move_to(UP * 2)
        self.play(
            FadeIn(image_mob)
        )
        self.wait()

        print("Image size:", image_mob.get_pixel_array().shape)

        pixel_grid = self.add_grid(image_mob)

        block_image, block_pixel_grid, block, pixel_array_mob_2d = self.highlight_pixel_block(image_mob, 125, 125, pixel_grid)
        centered_block = format_block(block[:, :, 0])
        new_pixel_array_mob_2d = self.get_2d_pixel_array_mob(centered_block, height=block_image.height)
        new_pixel_array_mob_2d.move_to(pixel_array_mob_2d.get_center())
        self.play(
            Transform(pixel_array_mob_2d, new_pixel_array_mob_2d)
        )
        self.wait()

        self.play(
            block_pixel_grid.animate.scale(0.65).shift(UP * 2.7 + RIGHT * 0.5),
            block_image.animate.scale(0.65).shift(UP * 2.7 + RIGHT * 0.5),
            pixel_array_mob_2d.animate.scale(0.65).shift(UP * 2.7 + LEFT * 0.5),
        )

        block_image, block_pixel_grid, pixel_array_mob_2d, dct_array_2d_row_col_group = self.show_data_flow_dct_2d(pixel_array_mob_2d, block[:, :, 0], block_image, block_pixel_grid)

        self.show_dct_components(dct_array_2d_row_col_group[0])

    def show_data_flow_dct_2d(self, pixel_array_mob_2d, block, block_image, block_pixel_grid):
        tip_length_ratio = 0.1
        pixel_array_mob_2d_copy_input = pixel_array_mob_2d.copy().move_to(LEFT * 4.5)
        self.play(
            TransformFromCopy(pixel_array_mob_2d, pixel_array_mob_2d_copy_input)
        )
        self.wait()

        row_rep, col_rep = self.get_2d_dct_mob_reps(pixel_array_mob_2d)

        arrow_top_left = Arrow(
            pixel_array_mob_2d_copy_input.get_right(), 
            row_rep.get_left(),
            max_tip_length_to_length_ratio=tip_length_ratio,

        ).set_color(REDUCIBLE_YELLOW)

        arrow_top_left_label_top = Tex(r"8 $\cross$ DCT 1D").scale(0.7).next_to(arrow_top_left, UP, buff=SMALL_BUFF)
        arrow_top_left_label_bottom = Tex("Rows").scale(0.8).next_to(arrow_top_left, DOWN, buff=SMALL_BUFF)
        self.play(
            Write(arrow_top_left),
            Write(arrow_top_left_label_top),
            Write(arrow_top_left_label_bottom),
        )
        self.wait()

        self.play(
            FadeIn(row_rep[0]),
            *[GrowFromCenter(arrow) for arrow in row_rep[1]]
        )
        self.wait()
        dct_array_2d_row_group = self.get_dct_array_2d_mob(pixel_array_mob_2d, [row_rep[0]], dct_rows(format_block(block)), RIGHT * 4.5)

        dct_mob_array_2d, overlay = dct_array_2d_row_group

        arrow_top_right = Arrow(
            row_rep.get_right(), 
            dct_mob_array_2d.get_left(),
            max_tip_length_to_length_ratio=tip_length_ratio,
        ).set_color(REDUCIBLE_YELLOW)

        self.play(
            Write(arrow_top_right)
        )
        self.wait()

        self.play(
            FadeIn(dct_mob_array_2d),
            TransformFromCopy(row_rep[0], overlay[0])
        )
        self.wait()

        dct_array_2d_row_group_input = dct_array_2d_row_group.copy().move_to(LEFT * 4.5 + DOWN * 2.7)

        self.play(
            TransformFromCopy(dct_array_2d_row_group, dct_array_2d_row_group_input)
        )
        self.wait()

        col_rep.move_to(DOWN * 2.7)

        arrow_bottom_left = Arrow(
            dct_array_2d_row_group_input.get_right(), 
            col_rep.get_left(),
            max_tip_length_to_length_ratio=tip_length_ratio,
        ).set_color(REDUCIBLE_VIOLET)

        arrow_bottom_left_label_top = Tex(r"8 $\cross$ DCT 1D").scale(0.7).next_to(arrow_bottom_left, UP, buff=SMALL_BUFF)
        arrow_bottom_left_label_bottom = Tex("Columns").scale(0.8).next_to(arrow_bottom_left, DOWN, buff=SMALL_BUFF)

        self.play(
            Write(arrow_bottom_left_label_top),
            Write(arrow_bottom_left_label_bottom),
            Write(arrow_bottom_left),
        )
        self.wait()

        self.play(
            FadeIn(col_rep[0]),
            *[GrowFromCenter(arrow) for arrow in col_rep[1]]
        )
        self.wait()

        dct_array_2d_row_col_group = self.get_dct_array_2d_mob(pixel_array_mob_2d, VGroup(row_rep[0], col_rep[0]), dct_2d(format_block(block)), RIGHT * 4.5 + DOWN * 2.7)

        dct_mob_array_2d_row_col, overlay_row_col = dct_array_2d_row_col_group

        arrow_bottom_right = Arrow(
            col_rep.get_right(),
            dct_array_2d_row_col_group.get_left(),
            max_tip_length_to_length_ratio=tip_length_ratio,
        ).set_color(REDUCIBLE_VIOLET)

        self.play(
            Write(arrow_bottom_right)
        )

        self.wait()

        self.play(
            FadeIn(dct_mob_array_2d_row_col, overlay_row_col[0]),
            TransformFromCopy(col_rep[0], overlay_row_col[1])
        )
        self.wait()
        original_dct_array_2d_row_col_group = dct_array_2d_row_col_group.copy().move_to(LEFT * 4.5)
        dct_2d_group = VGroup(
            pixel_array_mob_2d_copy_input,
            arrow_top_left,
            row_rep,
            arrow_top_right,
            dct_array_2d_row_group,
            dct_array_2d_row_group_input,
            arrow_bottom_left,
            col_rep,
            arrow_bottom_right,
            dct_array_2d_row_col_group,
            arrow_top_left_label_top,
            arrow_top_left_label_bottom,
            arrow_bottom_left_label_top,
            arrow_bottom_left_label_bottom,
        )

        surround_rect_dct_2d = SurroundingRectangle(dct_2d_group)

        dct_2d_text = Tex("DCT 2D").scale(4)

        dct_2d_text.move_to(surround_rect_dct_2d.get_center())

        self.play(
            Create(surround_rect_dct_2d)
        )

        self.play(
            dct_2d_group.animate.fade(0.8),
            Write(dct_2d_text)
        )
        self.wait()

        self.play(
            *[elem.animate.fade(1) for elem in dct_2d_group if not (elem is dct_array_2d_row_col_group)],
            FadeOut(dct_2d_text),
            FadeOut(surround_rect_dct_2d),
            Transform(dct_array_2d_row_col_group, original_dct_array_2d_row_col_group),
            FadeOut(pixel_array_mob_2d),
            FadeOut(block_image),
            FadeOut(block_pixel_grid)
        )
        self.wait()

        return block_image, block_pixel_grid, pixel_array_mob_2d, dct_array_2d_row_col_group

    def show_dct_components(self, dct_array_2d, rows=8, cols=8):
        all_dct_components = []
        for row in range(rows):
            for col in range(cols):
                pixel_grid_component = self.get_dct_component(row, col, height=0.8)
                all_dct_components.append(pixel_grid_component)
        
        all_dct_components_group = VGroup(*all_dct_components).arrange_in_grid(rows=rows, buff=SMALL_BUFF)
        all_dct_components_group.shift(RIGHT * 2)

        lines = []
        for rect, component in zip(dct_array_2d[0], all_dct_components_group):
            line = Line(rect.get_center(), component.get_center()).set_stroke(width=1, opacity=0.5)
            lines.append(line)



        self.play(
            *[Create(line) for line in lines],
            FadeIn(all_dct_components_group)
        )
        self.wait()

        first_row = VGroup(*all_dct_components_group[:8])
        first_col = VGroup(*[all_dct_components_group[i] for i in range(0, 64, 8)])

        self.play(
            Circumscribe(first_row, color=REDUCIBLE_YELLOW)
        )
        self.wait()
        self.play(
            Circumscribe(first_col, color=REDUCIBLE_VIOLET)
        )
        self.wait()

    def get_dct_component(self, row, col, height=2):
        dct_matrix = np.zeros((8, 8))
        if row == 0 and col == 0:
            dct_matrix[row][col] = 1016
        else:
            dct_matrix[row][col] = 500
        pixel_array = idct_2d(dct_matrix) + 128
        all_in_range = (pixel_array >= 0) & (pixel_array <= 255)
        if not all(all_in_range.flatten()):
            print("Bad array\n", pixel_array)
            raise ValueError("All elements in pixel_array must be in range [0, 255]")
        image_mob = self.get_image_vector_mob(pixel_array, height=height)
        return image_mob

    def get_image_vector_mob(self, pixel_array, height=2, num_pixels=8):
        side_length = height / num_pixels
        adjusted_row_values = np.zeros(pixel_array.shape)
        for i in range(pixel_array.shape[0]):
            for j in range(pixel_array.shape[1]):
                adjusted_row_values[i][j] = int(pixel_array[i][j])
        pixel_grid_mob = VGroup(
            *[
                Square(side_length=side_length)
                .set_stroke(color=gray_scale_value_to_hex(value), width=0.5)
                .set_fill(color=gray_scale_value_to_hex(value), opacity=1)
                for value in adjusted_row_values.flatten()
            ]
        ).arrange_in_grid(rows=num_pixels, buff=0)
        return pixel_grid_mob


    def get_2d_dct_mob_reps(self, pixel_array_mob_2d):
        row_representation = self.make_dct_grid(pixel_array_mob_2d.width, pixel_array_mob_2d.height, rows=True)
        col_representation = self.make_dct_grid(pixel_array_mob_2d.width, pixel_array_mob_2d.height, rows=False, color=REDUCIBLE_VIOLET)
        return row_representation, col_representation

    def get_dct_array_2d_mob(self, pixel_array_mob_2d, overlays, dct_values, position, color=REDUCIBLE_YELLOW):
        dct_mob_array_2d = self.get_2d_pixel_array_mob(dct_values, height=pixel_array_mob_2d.height)
        dct_mob_array_2d.move_to(position)
        final_overlays = VGroup()
        for overlay in overlays:
            new_overlay = overlay.copy()
            for rect in new_overlay:
                rect.set_stroke(opacity=0.4)
            new_overlay.move_to(position)
            final_overlays.add(new_overlay)
        
        group = VGroup(dct_mob_array_2d, final_overlays)

        return group

    def make_dct_grid(self, width, height, block_size=8, rows=True, color=REDUCIBLE_YELLOW):
        if rows:
            rect_width = width
            rect_height = height / block_size
        else:
            rect_width = width / block_size
            rect_height = height

        grid = VGroup(*[Rectangle(height=rect_height, width=rect_width) for _ in range(block_size)])
        if rows:
            grid.arrange(DOWN, buff=0)
            arrows = VGroup(*[DoubleArrow(start=rect.get_left(), end=rect.get_right(), stroke_width=3, tip_length=SMALL_BUFF, buff=SMALL_BUFF) for rect in grid])
        else:
            grid.arrange(RIGHT, buff=0)
            arrows = VGroup(*[DoubleArrow(start=rect.get_top(), end=rect.get_bottom(), stroke_width=3, tip_length=SMALL_BUFF, buff=SMALL_BUFF) for rect in grid])

        return VGroup(grid, arrows).set_color(color)

    def get_pixel_grid(self, image, num_pixels_in_dimension, color=WHITE):
        height_pixels = image.get_pixel_array().shape[0]
        width_pixels = image.get_pixel_array().shape[1]
        aspect_ratio = width_pixels / height_pixels
        height_single_cell = image.height / num_pixels_in_dimension
        width_single_cell = height_single_cell * aspect_ratio
        pixel_grid = VGroup(
            *[
                Rectangle(height=height_single_cell, width=width_single_cell).set_stroke(
                    color=color, width=1, opacity=0.5
                )
                for _ in range(num_pixels_in_dimension ** 2)
            ]
        )
        pixel_grid.arrange_in_grid(rows=num_pixels_in_dimension, buff=0)
        return pixel_grid
    
    def add_grid(self, image_mob):
        pixel_grid = self.get_pixel_grid(image_mob, 32)
        pixel_grid.move_to(image_mob.get_center())
        self.play(
            Create(pixel_grid)
        )
        self.wait()

        return pixel_grid

    def get_2d_pixel_array_mob(self, block, height=3, block_size=8, color=REDUCIBLE_GREEN_LIGHTER):
        array_mob_2d = VGroup(
            *[Square(side_length=height/block_size) for _ in range(block_size ** 2)]
        ).arrange_in_grid(rows=block_size, buff=0).set_color(color)

        array_text = VGroup()
        for i in range(block_size):
            for j in range(block_size):
                val = block[i][j]
                # For space constraints of rendering negative 3 digit numbers
                if val < -99:
                    val = -val
                integer = Text(str(int(val)), font="SF Mono", weight=MEDIUM).scale(0.22 * height / 3).move_to(array_mob_2d[i * block_size + j].get_center())
                array_text.add(integer)

        return VGroup(array_mob_2d, array_text)

    def highlight_pixel_block(self, image_mob, start_row, start_col, pixel_grid, block_size=8):
        pixel_array = image_mob.get_pixel_array()
        height_pixels = pixel_array.shape[0]
        width_pixels = pixel_array.shape[1]
        block = pixel_array[
            start_row : start_row + block_size, start_col : start_col + block_size
        ]
        center_row = start_row + block_size // 2
        center_col = start_col + block_size // 2
        vertical_pos = (
            image_mob.get_top()
            + DOWN * center_row / pixel_array.shape[0] * image_mob.height
        )
        horizontal_pos = (
            image_mob.get_left()
            + RIGHT * center_col / pixel_array.shape[1] * image_mob.width
        )
        pixel_grid_prop = (start_col * width_pixels + start_col) / (height_pixels * width_pixels)
        tiny_square_highlight = pixel_grid[int(pixel_grid_prop * len(pixel_grid))].copy().set_color(REDUCIBLE_GREEN_LIGHTER)
        tiny_square_highlight.set_stroke(width=4)
        self.play(Create(tiny_square_highlight))
        self.wait()

        block_position = LEFT * 2 + DOWN * 2
        block_image = self.get_image_mob(block, height=3.5).move_to(block_position)
        block_pixel_grid = self.get_pixel_grid(block_image, block_size).move_to(
            block_position
        )
        surround_rect = SurroundingRectangle(block_pixel_grid, buff=0).set_color(
            REDUCIBLE_GREEN_LIGHTER
        )

        pixel_array_mob_2d = self.get_2d_pixel_array_mob(block[:, :, 1], height=block_image.height)
        pixel_array_mob_2d.move_to(RIGHT * 2 + DOWN * 2)
        surround_rect_integer_grid = SurroundingRectangle(pixel_array_mob_2d, buff=0, color=REDUCIBLE_GREEN_LIGHTER)
        self.play(
            FadeIn(block_image),
            FadeIn(block_pixel_grid),
            FadeIn(pixel_array_mob_2d),
            TransformFromCopy(tiny_square_highlight, surround_rect),
            TransformFromCopy(tiny_square_highlight, surround_rect_integer_grid)
        )
        self.wait()

        shift_up = UP * 2

        self.play(
            FadeOut(surround_rect),
            FadeOut(tiny_square_highlight),
            FadeOut(surround_rect_integer_grid),
            FadeOut(image_mob),
            pixel_array_mob_2d.animate.shift(shift_up),
            block_image.animate.shift(shift_up),
            block_pixel_grid.animate.shift(shift_up),
            FadeOut(pixel_grid)
        )
        self.wait()

        return block_image, block_pixel_grid, block, pixel_array_mob_2d

class DemoJPEGWithDCT2DP2(ThreeDScene, ImageUtils):
    """
    TODO: Implement https://www.mathworks.com/help/vision/ref/2didct.html
    """
    def construct(self):
        image_mob = ImageMobject("dog").move_to(LEFT * 2)

        axes = ThreeDAxes(
            x_range=[0, 7], y_range=[0, 7], z_range=[0, 255], 
            x_length=5, y_length=5, z_length=5,
            tips=False,
            axis_config={"include_ticks": False},
        ).shift(IN * 1 + LEFT * 2)

        axes.set_color(BLACK)

        block_image_2d, pixel_grid_2d, block_2d = self.get_pixel_block(image_mob, 125, 125, height=3)
        print("Before 2D\n", block_2d[:, :, 1])
        block_image_2d = self.get_image_vector_mob(block_2d[:, :, 1], height=3)
        self.add_fixed_in_frame_mobjects(block_image_2d)
        block_image_2d.move_to(LEFT * 3.5 + UP)

        self.play(
            FadeIn(block_image_2d),
        )
        self.wait()

        block_image, block = self.get_pixel_block_for_3d(image_mob, 125, 125, height=axes.x_length)

        print('Before\n', block[:, :, 1])
        block_centered = format_block(block)
        print('After centering\n', block_centered)

        dct_block = dct_2d(block_centered)
        np.set_printoptions(suppress=True)
        print('DCT block (rounded)\n', np.round(dct_block, decimals=1))
        expected = invert_format_block(idct_2d(dct_block))
        actual = self.get_original_matrix_from_func(dct_block)
        print('Expected\n', expected)
        print('Actual\n', actual)
        assert(np.allclose(expected, actual))



        surface = Surface(
            lambda u, v: axes.c2p(*self.func(u, v, dct_block)),
            u_range=[0, 7],
            v_range=[0, 7],
            checkerboard_colors=[REDUCIBLE_PURPLE],
            fill_opacity=0.5,
            resolution=16,
            stroke_color=REDUCIBLE_YELLOW,
            stroke_width=2,
        )
        
        self.position_image_on_axes(axes, block_image)

        # self.add_fixed_in_frame_mobjects(block_image)
        # lines_to_z, dots_z = self.get_lines_and_dots(axes, block[:, :, 1], block_image)
        self.set_camera_orientation(theta=70 * DEGREES, phi=80 * DEGREES)
        self.play(
            FadeIn(axes),
            FadeIn(block_image),
            FadeIn(surface)
        )
        self.wait()

        number_line = self.initialize_slider(block_image, block[:, :, 1], surface)
        block_image = self.animate_slider(number_line, axes, block_image, dct_block, block[:, :, 1], surface, block_image_2d)

        # self.begin_ambient_camera_rotation(rate=0.1)
        # self.wait(5)

    def initialize_slider(self, block_image, block, surface):
        number_line = NumberLine(
            x_range=[0, 64, 8],
            length=8,
            color=REDUCIBLE_VIOLET,
            include_numbers=True,
            label_direction=UP,
            font_size=24,
        )
        slider_label = Tex("DCT Coefficients").scale(0.8)
        self.add_fixed_in_frame_mobjects(number_line, slider_label)
        number_line.move_to(DOWN * 3)
        slider_label.next_to(number_line, UP)
        self.play(
            FadeIn(number_line),
            Write(slider_label),
        )
        self.wait()
        return number_line

    def get_image_vector_mob(self, pixel_array, height=2, num_pixels=8, color=WHITE):
        side_length = height / num_pixels
        adjusted_row_values = np.zeros(pixel_array.shape)
        for i in range(pixel_array.shape[0]):
            for j in range(pixel_array.shape[1]):
                adjusted_row_values[i][j] = int(pixel_array[i][j])
        pixel_grid_mob = VGroup(
            *[
                Square(side_length=side_length)
                .set_stroke(color=color, width=0.5)
                .set_fill(color=gray_scale_value_to_hex(value), opacity=1)
                for value in adjusted_row_values.flatten()
            ]
        ).arrange_in_grid(rows=num_pixels, buff=0)
        return pixel_grid_mob

    def animate_slider(self, number_line, axes, block_image, dct_block, original_block, surface, block_image_2d):
        tick = Triangle().scale(0.15).set_color(REDUCIBLE_YELLOW)
        tick.set_fill(color=REDUCIBLE_YELLOW, opacity=1)
        self.add_fixed_in_frame_mobjects(tick)

        tracker = ValueTracker(0)
        tick.add_updater(
            lambda m: m.next_to(
                        number_line.n2p(tracker.get_value()),
                        DOWN
                    )
        )
        self.play( 
            FadeIn(tick),
        )
        self.wait()
        # surface_pos = RIGHT *
        def get_new_block():
            new_partial_block = self.get_partial_block(dct_block, tracker.get_value())
            print(f'Partial block - {tracker.get_value()} components')
            print('MSE', np.mean((new_partial_block - original_block) ** 2), '\n')

            new_partial_block_image = self.get_block_image_for_3d(new_partial_block, height=axes.x_length)            
            self.position_image_on_axes(axes, new_partial_block_image)
            return new_partial_block_image

        def get_new_surface():
            new_partial_block_dct = self.get_partial_dct_block(dct_block, tracker.get_value())
            # print('Generating surface of block:\n', new_partial_block_dct)
            new_surface = Surface(
                lambda u, v: axes.c2p(*self.func(u, v, new_partial_block_dct)),
                u_range=[0, 7],
                v_range=[0, 7],
                checkerboard_colors=[REDUCIBLE_PURPLE],
                fill_opacity=0.5,
                resolution=16,
                stroke_color=REDUCIBLE_YELLOW,
                stroke_width=2,
            )
            return new_surface
        
        def get_new_block_2d():
            new_partial_block = self.get_partial_block(dct_block, tracker.get_value())

            new_partial_block_image = self.get_image_vector_mob(new_partial_block, height=3)
            
            # TODO comment out this line for the smooth transition
            self.add_fixed_in_frame_mobjects(new_partial_block_image)
            new_partial_block_image.move_to(block_image_2d.get_center())
            return new_partial_block_image

        partial_block_image_2d = always_redraw(get_new_block_2d)
        # partial_pixel_grid_2d = self.get_pixel_grid(
        #     partial_block_image_2d, partial_block_2d.shape[0]
        # )
        # partial_pixel_grid_2d.move_to(par)

        partial_block = self.get_partial_block(dct_block, tracker.get_value())
        partial_block_image = always_redraw(get_new_block)
        partial_block_surface = always_redraw(get_new_surface)
        self.play(
            ReplacementTransform(block_image, partial_block_image),
            ReplacementTransform(surface, partial_block_surface),
            ReplacementTransform(block_image_2d, partial_block_image_2d),
        )
        self.wait()
        
       
        self.play(
            tracker.animate.set_value(64),
            run_time=10,
            rate_func=linear,
        )

        self.wait()
        return partial_block_image


    # 2D IDCT Function
    def func(self, x, y, dct_matrix):
        M, N = 8, 8
        def C(m):
            if m == 0:
                return 1 / np.sqrt(2)
            return 1

        result = 0
        norm_factor = 2 / (np.sqrt(M * N))
        for m in range(M):
            for n in range(N):
                cos_mx = np.cos((2 * x + 1) * m * np.pi / (2 * M))
                cos_ny = np.cos((2 * y + 1) * n * np.pi / (2 * N))
                result += C(m) * C(n) * dct_matrix[m][n] * cos_mx * cos_ny
        return np.array([x, y, norm_factor * result + 128])

    def get_original_matrix_from_func(self, dct_matrix):
        result = np.zeros((8, 8))
        for x in range(result.shape[0]):
            for y in range(result.shape[1]):
                result[x][y] = self.func(x, y, dct_matrix)[2]
        return result

    def position_image_on_axes(self, axes, block_image):
        block_image.move_to(axes.c2p(*[np.mean(axes.x_range) + 0.75, np.mean(axes.y_range) + 0.75, 0]))
        block_image.flip(RIGHT)

    def get_pixel_block_for_3d(self, image_mob, start_row, start_col, block_size=8, height=2):
        pixel_array = image_mob.get_pixel_array()
        block = pixel_array[start_row:start_row+block_size, start_col:start_col+block_size]

        block_image = self.get_block_image_for_3d(block[:, :, 1], block_size=block_size, height=height)

        return block_image, block

    def get_block_image_for_3d(self, block, block_size=8, height=2):
        # this block_image seems to break in 3D scenes, so using the pixel_grid itself
        # as a proxy for the image
        block_image = self.get_image_mob(block, height=height)
        pixel_grid = self.get_pixel_grid(block_image, block_size)

        for i in range(block.shape[0]):
            for j in range(block.shape[1]):
                index = two_d_to_1d_index(i, j)
                pixel_grid[index].set_fill(color=gray_scale_value_to_hex(block[i][j]), opacity=1)

        return pixel_grid

    def get_partial_block(self, dct_block, num_components):
        """
        @param: dct_block - dct coefficients of the block
        @param: num_components - float - number of dct components to
        include in partial block
        @return: pixel_array of partial block with num_components of DCT included
        """
        dct_matrix = self.get_partial_dct_block(dct_block, num_components)

        pixel_array = idct_2d(dct_matrix)
        return invert_format_block(pixel_array)

    def get_partial_dct_block(self, dct_block, num_components):
        """
        @param: dct_block - dct coefficients of the block
        @param: num_components - float - number of dct components to
        include in partial block
        @return: partial accumulated dct block containing a combination of num_components
        """
        from math import floor
        zigzag = get_zigzag_order()
        dct_matrix = np.zeros((8, 8))
        floor_val = floor(num_components)
        remaining = num_components - floor_val
        for basis_comp in range(floor_val):
            row, col = zigzag[basis_comp]
            dct_matrix[row][col] = dct_block[row][col]
        
        if floor_val < dct_block.shape[0] ** 2:
            row, col = zigzag[floor_val]
            dct_matrix[row][col] = remaining * dct_block[row][col]

        return dct_matrix

    def get_lines_and_dots(self, axes, block, block_image, color=REDUCIBLE_GREEN_DARKER):
        lines = VGroup()
        dots = VGroup()
        for i in range(block.shape[0]):
            for j in range(block.shape[1]):
                index = two_d_to_1d_index(i, j)
                start = block_image[index].get_center()
                end = axes.c2p(*[i, j, block[i][j]])
                dot = Dot().move_to(end).set_color(color=color)
                dots.add(dot)
                line = Line(start, end).set_stroke(color=color, width=1)
                lines.add(line)

        return lines, dots

    def get_pixel_block(self, image_mob, start_row, start_col, block_size=8, height=2):
        pixel_array = image_mob.get_pixel_array()
        block = pixel_array[
            start_row : start_row + block_size, start_col : start_col + block_size
        ]

        block_image = self.get_image_mob(block, height=height)
        pixel_grid = self.get_pixel_grid(block_image, block_size)

        return block_image, pixel_grid, block

class HeatMapExperiments(Introduce2DDCT):
    def construct(self):
        image_mob = ImageMobject("dog").move_to(UP * 2.5)
        self.play(
            FadeIn(image_mob)
        )
        self.wait()

        print("Image size:", image_mob.get_pixel_array().shape)

        pixel_grid = self.add_grid(image_mob)

        block_image, block_pixel_grid, block, pixel_array_mob_2d, tiny_square_highlight = self.highlight_pixel_block(image_mob, 125, 125, pixel_grid)

        block_single_channel = block[:, :, 0]

        self.show_changing_blocks_with_dct(image_mob, block_image, block_pixel_grid, pixel_array_mob_2d, block_single_channel, tiny_square_highlight, pixel_grid)

    def show_changing_blocks_with_dct(self, image_mob, block_image, block_pixel_grid, pixel_array_mob_2d, block_single_channel, tiny_square_highlight, pixel_grid):
        dct_array_2d_mob, dct_heat_map, dct_values = self.get_array_and_heat_map(block_image, pixel_array_mob_2d, block_single_channel)
        pixel_array = image_mob.get_pixel_array()
        height_pixels = pixel_array.shape[0]
        width_pixels = pixel_array.shape[1]

        self.play(
            FadeIn(dct_array_2d_mob),
            FadeIn(dct_heat_map)
        )
        self.wait()
        for _ in range(60):
            start_row, start_col = np.random.randint(0, height_pixels), np.random.randint(0, width_pixels)
            new_block_image, new_block_pixel_grid, new_block = self.get_pixel_block(image_mob, start_row, start_col, height=block_image.height)
            new_block_image.move_to(block_image.get_center())
            new_block_pixel_grid.move_to(block_pixel_grid.get_center())
            
            new_tiny_square_highlight = self.get_tiny_square(start_row, start_col, image_mob, pixel_array, pixel_grid)
            tiny_square_highlight.become(new_tiny_square_highlight)
            if new_block.shape[0] != 8 or new_block.shape[1] != 8:
                continue
            new_block_single_channel = new_block[:, :, 0]
            new_pixel_array_mob_2d = self.get_2d_pixel_array_mob(new_block_single_channel, height=pixel_array_mob_2d.height)
            
            new_pixel_array_mob_2d.move_to(pixel_array_mob_2d.get_center())
            
            block_image.become(new_block_image)
            block_pixel_grid.become(new_block_pixel_grid)
            pixel_array_mob_2d.become(new_pixel_array_mob_2d)

            new_dct_array_2d_mob, new_dct_heat_map, dct_values = self.get_array_and_heat_map(new_block_image, new_pixel_array_mob_2d, new_block_single_channel)
            if dct_values[0][0] < -100 or abs(dct_values[0][0]) < 15 or abs(dct_values[0][0]) >= 1000:
                continue

            dct_array_2d_mob.become(new_dct_array_2d_mob)
            dct_heat_map.become(new_dct_heat_map)
            self.wait()
        # self.remove(block_image, block_pixel_grid, pixel_array_mob_2d)
        # self.add(new_block_image, new_pixel_grid, new_pixel_array_mob_2d)
        # self.wait()

    def get_pixel_block(self, image_mob, start_row, start_col, block_size=8, height=2):
        pixel_array = image_mob.get_pixel_array()
        block = pixel_array[
            start_row : start_row + block_size, start_col : start_col + block_size
        ]

        block_image = self.get_image_mob(block, height=height)
        pixel_grid = self.get_pixel_grid(block_image, block_size)

        return block_image, pixel_grid, block
    
    def get_array_and_heat_map(self, block_image, pixel_array_mob_2d, block_single_channel):
        formatted_block = format_block(block_single_channel)
        dct_values = dct_2d(formatted_block)
        dct_array_2d_mob = self.get_2d_pixel_array_mob(dct_values, height=pixel_array_mob_2d.height, color=REDUCIBLE_VIOLET)
        dct_array_2d_mob.move_to(RIGHT * 3.5 + UP * 2)

        dct_heat_map = self.get_heat_map(block_image, dct_values)
        dct_heat_map.next_to(dct_array_2d_mob, DOWN, aligned_edge=LEFT)
        shift_down = (-2 - dct_heat_map.get_center()[1])
        dct_heat_map.shift(UP * shift_down)
        
        return dct_array_2d_mob, dct_heat_map, dct_values

    def get_heat_map(self, block_image, dct_block):
        block_size = dct_block.shape[0]
        pixel_grid_dct = self.get_pixel_grid(block_image, block_size)
        dct_block_abs = np.abs(dct_block)
        max_dct_coeff = np.amax(dct_block_abs)
        max_color = REDUCIBLE_YELLOW
        min_color = REDUCIBLE_PURPLE
        for i, square in enumerate(pixel_grid_dct):
            row, col = i // block_size, i % block_size
            alpha = dct_block_abs[row][col] / max_dct_coeff
            square.set_fill(
                color=interpolate_color(min_color, max_color, alpha), opacity=1
            )

        scale = Line(pixel_grid_dct.get_top(), pixel_grid_dct.get_bottom())
        scale.set_stroke(width=10).set_color(color=[min_color, max_color])
        integer_scale = 0.3
        top_value = Text(str(int(max_dct_coeff)), font='SF Mono', weight=MEDIUM).scale(integer_scale)
        top_value.next_to(scale, RIGHT, aligned_edge=UP)
        bottom_value = Text("0", font='SF Mono', weight=MEDIUM).scale(integer_scale)
        bottom_value.next_to(scale, RIGHT, aligned_edge=DOWN)

        heat_map_scale = VGroup(scale, top_value, bottom_value)

        return VGroup(pixel_grid_dct, heat_map_scale).arrange(RIGHT)

    def get_tiny_square(self, start_row, start_col, image_mob, pixel_array, pixel_grid, block_size=8):
        center_row = start_row + block_size // 2
        center_col = start_col + block_size // 2
        vertical_pos = (
            image_mob.get_top()
            + DOWN * center_row / pixel_array.shape[0] * image_mob.height
        )
        horizontal_pos = (
            image_mob.get_left()
            + RIGHT * center_col / pixel_array.shape[1] * image_mob.width
        )
        tiny_square_highlight = Square(side_length=SMALL_BUFF * 0.8)
        highlight_position = np.array([horizontal_pos[0], vertical_pos[1], 0])
        tiny_square_highlight.set_color(REDUCIBLE_GREEN_LIGHTER).move_to(highlight_position)
        return tiny_square_highlight
        # height_pixels = pixel_array.shape[0]
        # width_pixels = pixel_array.shape[1]
        # print(width_pixels, height_pixels)
        # pixel_grid_prop = (start_row * width_pixels + start_col) / (height_pixels * width_pixels)
        # tiny_square_highlight = pixel_grid[int(pixel_grid_prop * len(pixel_grid))].copy().set_color(REDUCIBLE_GREEN_LIGHTER)
        # tiny_square_highlight.set_stroke(width=4)
        # return tiny_square_highlight

    def highlight_pixel_block(self, image_mob, start_row, start_col, pixel_grid, block_size=8):
        pixel_array = image_mob.get_pixel_array()
        height_pixels = pixel_array.shape[0]
        width_pixels = pixel_array.shape[1]
        block = pixel_array[
            start_row : start_row + block_size, start_col : start_col + block_size
        ]
        center_row = start_row + block_size // 2
        center_col = start_col + block_size // 2
        vertical_pos = (
            image_mob.get_top()
            + DOWN * center_row / pixel_array.shape[0] * image_mob.height
        )
        horizontal_pos = (
            image_mob.get_left()
            + RIGHT * center_col / pixel_array.shape[1] * image_mob.width
        )
        tiny_square_highlight = Square(side_length=SMALL_BUFF * 0.8)
        highlight_position = np.array([horizontal_pos[0], vertical_pos[1], 0])
        tiny_square_highlight.set_color(REDUCIBLE_GREEN_LIGHTER).move_to(highlight_position)
        self.play(Create(tiny_square_highlight))
        self.wait()

        block_position = LEFT * 2 + DOWN * 1
        block_image = self.get_image_mob(block, height=3.5).move_to(block_position)
        block_pixel_grid = self.get_pixel_grid(block_image, block_size).move_to(
            block_position
        )
        surround_rect = SurroundingRectangle(block_pixel_grid, buff=0).set_color(
            REDUCIBLE_GREEN_LIGHTER
        )

        pixel_array_mob_2d = self.get_2d_pixel_array_mob(block[:, :, 1], height=block_image.height)
        pixel_array_mob_2d.move_to(RIGHT * 2 + DOWN * 1)
        surround_rect_integer_grid = SurroundingRectangle(pixel_array_mob_2d, buff=0, color=REDUCIBLE_GREEN_LIGHTER)
        self.play(
            FadeIn(block_image),
            FadeIn(block_pixel_grid),
            FadeIn(pixel_array_mob_2d),
            TransformFromCopy(tiny_square_highlight, surround_rect),
            TransformFromCopy(tiny_square_highlight, surround_rect_integer_grid)
        )
        self.wait()

        self.play(
            pixel_array_mob_2d.animate.scale(0.8).move_to(LEFT * 3.5 + UP * 2),
            block_image.animate.scale(0.8).move_to(LEFT * 3.5 + DOWN * 2),
            block_pixel_grid.animate.scale(0.8).move_to(LEFT * 3.5 + DOWN * 2),
            surround_rect.animate.scale(0.8).move_to(LEFT * 3.5 + DOWN * 2),
            surround_rect_integer_grid.animate.scale(0.8).move_to(LEFT * 3.5 + UP * 2),
            image_mob.animate.move_to(ORIGIN),
            pixel_grid.animate.move_to(ORIGIN),
            tiny_square_highlight.animate.shift(DOWN * 2.5)
        )
        self.wait()

        self.add_foreground_mobject(surround_rect)

        return block_image, block_pixel_grid, block, pixel_array_mob_2d, tiny_square_highlight

class QuantizationP3(Introduce2DDCT):
    def construct(self):
        height = 3.2
        off_center_horiz = 4.5
        off_center_vert = 0
        image_mob = ImageMobject("dog").move_to(UP * 2.5)
        block_image, pixel_grid, block = self.get_pixel_block(image_mob, 125, 125)
        block_single_channel = block[:, :, 0]
        formatted_block = format_block(block_single_channel)
        dct_values = dct_2d(formatted_block)
        dct_array_2d_mob = self.get_2d_pixel_array_mob(dct_values, height=height, color=REDUCIBLE_VIOLET)
        dct_array_2d_mob.move_to(LEFT * off_center_horiz + UP * off_center_vert)

        quantization_table = self.get_2d_pixel_array_mob(get_quantization_table(), height=height, color=GRAY)
        quantization_table.move_to(UP * off_center_vert)

        self.play(
            FadeIn(dct_array_2d_mob)
        )
        self.wait()

        arrow_1 = Arrow(dct_array_2d_mob.get_right(), quantization_table.get_left())
        division = Tex("÷").scale(1.3).next_to(arrow_1, UP)
        self.play(
            Write(arrow_1),
            Write(division),
            FadeIn(quantization_table)
        )

        quantized_dct = self.get_2d_pixel_array_mob(quantize(dct_values), height=height, color=REDUCIBLE_PURPLE)
        quantized_dct.move_to(RIGHT * off_center_horiz + UP * off_center_vert)

        arrow_2 = Arrow(quantization_table.get_right(), quantized_dct.get_left())

        self.play(
            Write(arrow_2),
            FadeIn(quantized_dct)
        )

        self.wait()
        brace_down = Brace(quantization_table, direction=DOWN).next_to(quantization_table, DOWN)
        quantization_table_label = Tex("Quantization" + "\\\\" + "Table").next_to(brace_down, DOWN)

        self.play(
            GrowFromCenter(brace_down),
            Write(quantization_table_label)
        )

        self.wait()

        lower_triangle_indices = []
        for i in range(43, 64):
            zigzag = get_zigzag_order()
            i, j = zigzag[i]
            lower_triangle_indices.append(two_d_to_1d_index(i, j))

        upper_triangle_indices = []
        for i in range(6):
            zigzag = get_zigzag_order()
            i, j = zigzag[i]
            upper_triangle_indices.append(two_d_to_1d_index(i, j))

        quantization_table_text = quantization_table[1]
        lower_triangle_group = VGroup(*[quantization_table_text[i] for i in lower_triangle_indices])
        self.play(
            LaggedStartMap(Indicate, lower_triangle_group),
            run_time=2
        )
        self.wait()

        quantized_dct_text = quantized_dct[1]
        lower_triangle_group_qdct = VGroup(*[quantized_dct_text[i] for i in lower_triangle_indices])
        self.play(
            LaggedStartMap(Indicate, lower_triangle_group_qdct),
            run_time=2
        )
        self.wait()

        encode_group = VGroup(dct_array_2d_mob, arrow_1, division, quantization_table, arrow_2, quantized_dct)

        self.play(
            encode_group.animate.scale(0.8).shift(UP * 2),
            FadeOut(brace_down),
            FadeOut(quantization_table_label)
        )

        surround_rect_encode = SurroundingRectangle(encode_group, buff=SMALL_BUFF * 6, color=REDUCIBLE_YELLOW)

        encode_text = Tex("Encoding").scale(0.8).move_to(surround_rect_encode.get_top()).shift(DOWN * 0.3)
        self.play(
            Create(surround_rect_encode),
            Write(encode_text)
        )
        self.wait()

        input_decoder_quantized = quantized_dct.copy().move_to(dct_array_2d_mob.get_center() + DOWN * 4)
        self.play(
            TransformFromCopy(quantized_dct, input_decoder_quantized)
        )
        decoder_quantization_table = quantization_table.copy().move_to(DOWN * 2)
        arrow_3 = arrow_1.copy().shift(DOWN * 4)
        multiply = MathTex(r"\cross").next_to(arrow_3, UP)
        self.play(
            Write(arrow_3),
            Write(multiply),
            TransformFromCopy(quantization_table, decoder_quantization_table)
        )

        dequantized_dct = self.get_2d_pixel_array_mob(dequantize(quantize(dct_values)), height=quantized_dct.height, color=REDUCIBLE_VIOLET)
        dequantized_dct.move_to(quantized_dct.get_center() + DOWN * 4)
        arrow_4 = arrow_2.copy().shift(DOWN * 4)
        self.play(
            Write(arrow_4),
            FadeIn(dequantized_dct)
        )

        decode_group = VGroup(input_decoder_quantized, arrow_3, multiply, decoder_quantization_table, arrow_4, dequantized_dct)

        surround_rect_decode = SurroundingRectangle(decode_group, buff=SMALL_BUFF * 6, color=REDUCIBLE_VIOLET)
        decode_text = Tex("Decoding").scale(0.8).move_to(surround_rect_decode.get_top()).shift(DOWN * 0.3)

        self.play(
            Create(surround_rect_decode),
            Write(decode_text)
        )
        self.wait()

        self.play(
            ApplyWave(dequantized_dct)
        )
        self.wait()

        self.play(
            ApplyWave(dct_array_2d_mob)
        )
        self.wait()

        dequantized_dct_mob = dequantized_dct[0]
        dct_array_2d_mob_mob = dct_array_2d_mob[0]

        quantized_dct_mob = quantized_dct[0]
        lower_triangle_group_qdct_mob = VGroup(*[quantized_dct_mob[i] for i in lower_triangle_indices[6:]])
        
        upper_left_group = VGroup(*[dequantized_dct_mob[i] for i in upper_triangle_indices])
        upper_left_group_input_dct = VGroup(*[dct_array_2d_mob_mob[i] for i in upper_triangle_indices])
        
        surround_rect_input_dct = SurroundingRectangle(upper_left_group_input_dct, buff=0).set_stroke(color=REDUCIBLE_YELLOW, width=7)
        surround_rect_quantized_dct = SurroundingRectangle(upper_left_group, buff=0).set_stroke(color=REDUCIBLE_YELLOW, width=7)
        self.play(
            Create(surround_rect_input_dct),
            Create(surround_rect_quantized_dct)
        )
        self.wait()

        surround_rect_lower = SurroundingRectangle(lower_triangle_group_qdct_mob, buff=0).set_stroke(color=REDUCIBLE_YELLOW, width=7)
        self.play(
            FadeOut(surround_rect_input_dct),
            FadeOut(surround_rect_quantized_dct),
            Create(surround_rect_lower)
        )
        self.wait()
        first_quantization_table = quantization_table.copy()
        self.play(
            FadeOut(decode_group),
            FadeOut(surround_rect_lower),
            FadeOut(dct_array_2d_mob),
            FadeOut(arrow_1),
            FadeOut(division),
            FadeOut(arrow_2),
            FadeOut(quantized_dct),
            FadeOut(surround_rect_encode),
            FadeOut(surround_rect_decode),
            FadeOut(encode_text),
            FadeOut(decode_text),
            quantization_table.animate.scale(1.5).move_to(UP * 0.5)
        )
        self.wait()
        original_quantization_table = quantization_table.copy().shift(LEFT * 3.5)
        quality_factor = Tex("Quality factor: 50").move_to(DOWN * 3)
        original_quality_factor = quality_factor.copy()
        self.play(
            Write(quality_factor)
        )
        self.wait()
        better_quality_factor = Tex("Quality factor: 80")

        better_quality_factor.move_to(quality_factor.get_center())

        quality_8_quantization_table = self.get_2d_pixel_array_mob(get_80_quality_quantization_table(), height=quantization_table.height, color=GRAY)
        quality_8_quantization_table.move_to(quantization_table.get_center())
        self.play(
            ReplacementTransform(quality_factor, better_quality_factor),
            ReplacementTransform(quantization_table, quality_8_quantization_table)
        )
        self.wait()

        luma_channel = Tex("Luma (Y) Channel").scale(0.8)
        chrominance_channel = Tex("Chrominance (Cb/Cr) Channel").scale(0.8)

        chroma_quantization_table = self.get_2d_pixel_array_mob(get_chroma_quantization_table(), height=quantization_table.height, color=REDUCIBLE_GREEN_LIGHTER)
        chroma_quantization_table.shift(RIGHT * 3.5 + UP * 0.5)
        luma_channel.next_to(original_quantization_table, DOWN)
        chrominance_channel.next_to(chroma_quantization_table, DOWN)
        self.play(
            ReplacementTransform(quality_8_quantization_table, original_quantization_table),
            ReplacementTransform(better_quality_factor,  original_quality_factor),
            FadeIn(chroma_quantization_table),
            Write(luma_channel),
            Write(chrominance_channel)
        )

        self.wait()

        self.clear()
        original_encode_group = VGroup(dct_array_2d_mob, arrow_1, division, first_quantization_table, arrow_2, quantized_dct).scale(1 / 0.8).move_to(ORIGIN)
        self.play(
            FadeIn(original_encode_group)
        )
        self.wait()

        self.play(
            FadeOut(dct_array_2d_mob), 
            FadeOut(arrow_1), 
            FadeOut(division), 
            FadeOut(first_quantization_table), 
            FadeOut(arrow_2),
            quantized_dct.animate.scale(1.5).move_to(ORIGIN)
        )
        self.wait()

        redundancy_idea = Tex("Big Idea: Exploit ", "Redundancy").move_to(UP * 3)
        redundancy_idea[1].set_color(REDUCIBLE_YELLOW)
        self.play(
            Write(redundancy_idea)
        )
        self.play(
            LaggedStartMap(Indicate, lower_triangle_group_qdct),
            run_time=2
        )
        self.wait()


        
    def get_pixel_block(self, image_mob, start_row, start_col, block_size=8, height=2):
        pixel_array = image_mob.get_pixel_array()
        block = pixel_array[
            start_row : start_row + block_size, start_col : start_col + block_size
        ]

        block_image = self.get_image_mob(block, height=height)
        pixel_grid = self.get_pixel_grid(block_image, block_size)

        return block_image, pixel_grid, block

    def get_2d_pixel_array_mob(self, block, height=3, block_size=8, color=REDUCIBLE_GREEN_LIGHTER):
        array_mob_2d = VGroup(
            *[Square(side_length=height/block_size) for _ in range(block_size ** 2)]
        ).arrange_in_grid(rows=block_size, buff=0).set_color(color)

        array_text = VGroup()
        for i in range(block_size):
            for j in range(block_size):
                val = block[i][j]
                # For space constraints of rendering negative 3 digit numbers
                if val < -99:
                    val = -val


                integer = Text(str(int(val)), font="SF Mono", weight=MEDIUM).scale(0.22 * height / 3).move_to(array_mob_2d[i * block_size + j].get_center())
                array_text.add(integer)

        return VGroup(array_mob_2d, array_text)

class Redundancy(QuantizationP3):
    def construct(self):
        example_matrix = np.array(
            [
            [90, -40, 0, 4, 0, 0, 0, 0],
            [0,   0,  0, 0, 0, 0, 0, 0],
            [5,   0,  0, 0, 0, 0, 0, 0],
            [0,   0,  0, 0, 0, 0, 0, 0],
            [11, -3,  0, 0, 0, 0, 0, 0],
            [0,   0,  0, 0, 0, 0, 0, 0],
            [0,   0,  0, 0, 0, 0, 0, 0],
            [0,   0,  0, 0, 0, 0, 0, 0]
            ]
        )

        quantized_dct = self.get_2d_pixel_array_mob(example_matrix, height=4, color=REDUCIBLE_PURPLE)
        quantized_dct_mob = quantized_dct[0]
        quantized_dct_values = quantized_dct[1]
        self.play(
            FadeIn(quantized_dct)
        )
        self.wait()

        zigzag = get_zigzag_order()

        zigzag_path = VGroup()
        path_points = []
        new_integers = []
        new_integers.append(Text("{", font="SF Mono", weight=LIGHT))
        original_integers = []
        for index in range(example_matrix.shape[0] ** 2):
            i, j = zigzag[index]
            one_d_index = two_d_to_1d_index(j, i)
            path_points.append(quantized_dct_mob[one_d_index].get_center())
            original_integers.append(quantized_dct_values[one_d_index])
            new_integers.append(quantized_dct_values[one_d_index].copy())
        zigzag_path.set_points_as_corners(*[path_points]).set_stroke(color=REDUCIBLE_VIOLET, width=3)
        new_integers.append(Text("}", font="SF Mono", weight=LIGHT))
        new_integers[0].scale_to_fit_height(new_integers[1].height + SMALL_BUFF)
        new_integers[-1].scale_to_fit_height(new_integers[1].height + SMALL_BUFF)
        one_d_array = VGroup(*new_integers).scale(0.9).arrange(RIGHT, buff=SMALL_BUFF)
        one_d_array.next_to(quantized_dct_mob, DOWN * 2)

        self.play(
            Create(zigzag_path),
            run_time=5,
            rate_func=linear
        )

        self.play(
            FadeIn(one_d_array[0]),
            FadeIn(one_d_array[-1]),
            LaggedStart(
                *[TransformFromCopy(orig, new) for orig, new in zip(original_integers, one_d_array[1:-1])],
            ),
            run_time=3
        )

        self.play(
            quantized_dct.animate.shift(UP * 1.7),
            zigzag_path.animate.shift(UP * 1.7),
            one_d_array.animate.shift(UP * 2),
        )
        self.wait()

        classic_run_length = Text("90, -40, 0, 5, 0[x2], 4, 0[x3], 11, 0[x8], -3, 0[x44]", font="SF Mono", weight=MEDIUM).scale(0.4)
        left_brace, right_brace = one_d_array[0].copy(), one_d_array[-1].copy()
        left_brace.scale_to_fit_height(classic_run_length.height + SMALL_BUFF)
        right_brace.scale_to_fit_height(classic_run_length.height + SMALL_BUFF)

        entire_run_length = VGroup(left_brace, classic_run_length, right_brace).arrange(RIGHT, buff=SMALL_BUFF)
        entire_run_length.next_to(one_d_array, DOWN)
        self.play(
            TransformFromCopy(one_d_array[1:-1], entire_run_length[1:-1]),
            TransformFromCopy(one_d_array[0], entire_run_length[0]),
            TransformFromCopy(one_d_array[-1], entire_run_length[-1]),
            run_time=2
        )
        self.wait()

        self.play(
            quantized_dct.animate.shift(LEFT * 3.5),
            zigzag_path.animate.shift(LEFT * 3.5),
        )

        self.wait()


        jpeg_specific_run_length = Title("JPEG Specific Run-length Encoding", match_underline_width_to_text=True).scale(0.9)
        jpeg_specific_run_length.move_to(RIGHT * 2.5 + UP * 3.2)

        self.play(
            Write(jpeg_specific_run_length)
        )
        self.wait()

        triplet = MathTex("[(r, s), c]").scale(0.9).next_to(jpeg_specific_run_length, DOWN)

        r_def = Tex(r"$r$ - number of 0's preceding value").scale(0.7)
        s_def = Tex(r"$s$ - number of bits needed to encode value $c$").scale(0.7)
        c_def = Tex(r"$c$ - coefficient value").scale(0.7)
        z_def = Tex("(0, 0) indicates end of block (all 0's)").scale(0.7)
        triplet.next_to(jpeg_specific_run_length, DOWN)

        r_def.next_to(triplet, DOWN).shift(LEFT * SMALL_BUFF * 3)

        s_def.next_to(r_def, DOWN, aligned_edge=LEFT)
        c_def.next_to(s_def, DOWN, aligned_edge=LEFT)
        z_def.next_to(c_def, DOWN, aligned_edge=LEFT)

        self.play(
            FadeIn(triplet),
            FadeIn(r_def),
            FadeIn(s_def),
            FadeIn(c_def),
            FadeIn(z_def)
        )
        self.wait()

        final_rle = Text("[(0, 7), 90], [(0, 6), -40], [(1, 3), 5], [(2, 3), 4], [(3, 4), 11], [(8, 2), -3], [(0, 0)]", font="SF Mono", weight=MEDIUM).scale(0.32)
        final_rle_with_braces = VGroup(
            left_brace.copy().scale_to_fit_height(final_rle.height + SMALL_BUFF), 
            final_rle,
            right_brace.copy().scale_to_fit_height(final_rle.height + SMALL_BUFF),
        ).arrange(RIGHT, buff=SMALL_BUFF)
        final_rle_with_braces.next_to(classic_run_length, DOWN * 2)

        self.play(
            TransformFromCopy(classic_run_length, final_rle),
            TransformFromCopy(entire_run_length[0], final_rle_with_braces[0]),
            TransformFromCopy(entire_run_length[-1], final_rle_with_braces[-1])
        )
        self.wait()

        self.play(
            FadeOut(entire_run_length),
            FadeOut(one_d_array),
            final_rle_with_braces.animate.shift(UP * 0.8)

        )
        self.wait()

        surround_rect = SurroundingRectangle(final_rle_with_braces, color=REDUCIBLE_YELLOW)

        self.play(
            Create(surround_rect)
        )
        self.wait()

        huffman_encode_component = Module(["Huffman", "Encoder"], text_weight=MEDIUM)

        huffman_encode_component.scale(0.7).next_to(final_rle, DOWN * 2)

        line = Line(surround_rect.get_bottom(), huffman_encode_component.get_center()).shift(LEFT * 3.5)

        arrow = Arrow(line.get_end(), huffman_encode_component.get_left(), max_tip_length_to_length_ratio=0.05, buff=SMALL_BUFF)
        arrow.set_stroke(width=line.get_stroke_width()).shift(LEFT * SMALL_BUFF)
        elbow_arrow = VGroup(line, arrow)

        self.play(
            FadeIn(elbow_arrow)
        )
        self.play(
            FadeIn(huffman_encode_component)
        )
        self.wait()

        jpeg_specific_huffman_coding =  Title("JPEG Specific Huffman Encoding", match_underline_width_to_text=True).scale(0.9)
        jpeg_specific_huffman_coding.move_to(jpeg_specific_run_length.get_center())

        self.play(
            ReplacementTransform(jpeg_specific_run_length, jpeg_specific_huffman_coding),
            FadeOut(triplet),
            FadeOut(r_def),
            FadeOut(s_def),
            FadeOut(c_def),
            FadeOut(z_def),
            FadeOut(zigzag_path)
        )
        self.wait()

        self.show_JPEG_specific_huffman_coding(jpeg_specific_huffman_coding, quantized_dct_mob, line, huffman_encode_component)

    def show_JPEG_specific_huffman_coding(self, jpeg_huffman_title, quantized_dct_mob, line, huffman_encode_component):
        bulleted_list = BulletedList(
            r"More frequent data $\rightarrow$ fewer bits",
            r"Some triplet values are more common",
            buff=MED_SMALL_BUFF
        ).scale(0.8)
        bulleted_list.next_to(jpeg_huffman_title, DOWN)

        self.play(
            FadeIn(bulleted_list)
        )
        self.wait()

        more_complicated = Tex(r"It's quite complicated $\ldots$").scale(0.8)
        complexities = BulletedList(
            "Signs of coefficients",
            r"All 8 $\cross$ 8 blocks",
            "Top left (DC) coefficients encoded" + "\\\\" +
            "separately from other (AC) coefficients",
            "Luma (Y) and Chroma (Cb/Cr) channels",
            "Chroma subsampling",
            buff=MED_SMALL_BUFF
        ).scale(0.6)

        more_complicated.next_to(jpeg_huffman_title, DOWN, aligned_edge=LEFT)
        self.play(
            ReplacementTransform(bulleted_list, more_complicated)
        )
        self.wait()

        complexities.next_to(more_complicated, DOWN, aligned_edge=LEFT).shift(RIGHT * SMALL_BUFF * 3)

        self.play(
            FadeIn(complexities[0])
        )
        self.wait()

        self.play(
            FadeIn(complexities[1])
        )
        self.wait()
        complexities[2][8:12].set_color(REDUCIBLE_YELLOW)
        complexities[2][-17:-13].set_color(REDUCIBLE_VIOLET)
        self.play(
            FadeIn(complexities[2]),
            quantized_dct_mob[0].animate.set_fill(color=REDUCIBLE_YELLOW, opacity=0.35),
            quantized_dct_mob[1:].animate.set_fill(color=REDUCIBLE_VIOLET, opacity=0.35)

        )
        self.wait()

        self.play(
            FadeIn(complexities[3])
        )
        self.wait()

        self.play(
            FadeIn(complexities[4])
        )
        self.wait()

        output_image = SVGMobject("jpg_file.svg").set_stroke(
            WHITE, width=5, background=True
        ).scale(0.7)

        output_image.move_to(huffman_encode_component.get_center()).shift(RIGHT * 5)

        exploits = Tex("Exploits").scale(0.7)
        redundancy = Tex("Redundancy").scale(0.7)

        final_arrow = Arrow(huffman_encode_component.get_right(), output_image.get_left(), max_tip_length_to_length_ratio=0.05, buff=SMALL_BUFF * 2)
        final_arrow.set_stroke(width=line.get_stroke_width())
        exploits.next_to(final_arrow, UP)
        redundancy.next_to(final_arrow, DOWN)
        self.play(
            Write(final_arrow),
            Write(exploits),
            Write(redundancy)
        )

        self.play(
            FadeIn(output_image)
        )
        self.wait()





# class IntroAnimations(DemoJPEGWithDCT2D):
#     def construct(self):
#         image_mob = ImageMobject("dog").move_to(LEFT * 2)

#         new_pixel_array = self.get_all_blocks(image_mob, 2, 298, 3, 331, 64)
#         relevant_section = new_pixel_array[2:298, 3:331]
#         original_image_mob = self.get_image_mob(new_pixel_array, height=None).move_to(UP * 1 + RIGHT * 2)

#         axes = ThreeDAxes(
#             x_range=[0, 7], y_range=[0, 7], z_range=[0, 255], 
#             x_length=5, y_length=5, z_length=5,
#             tips=False,
#             axis_config={"include_ticks": False},
#         ).shift(IN * 1.5 + LEFT * 2)

#         axes.set_color(BLACK)

#         block_image_2d, pixel_grid_2d, block_2d = self.get_pixel_block(image_mob, 125, 125, height=2)
#         print("Before 2D\n", block_2d[:, :, 1])
#         block_image_2d = self.get_image_vector_mob(block_2d[:, :, 1], height=3)
#         self.add_fixed_in_frame_mobjects(original_image_mob, block_image_2d)
#         original_image_mob.move_to(LEFT * 3.5 + UP * 2.5)
#         block_image_2d.move_to(LEFT * 3.5 + DOWN * 1)

#         self.play(
#             FadeIn(block_image_2d),
#             FadeIn(original_image_mob),
#         )
#         self.wait()

#         block_image, block = self.get_pixel_block_for_3d(image_mob, 125, 125, height=axes.x_length)

#         print('Before\n', block[:, :, 1])
#         block_centered = format_block(block)
#         print('After centering\n', block_centered)

#         dct_block = dct_2d(block_centered)
#         np.set_printoptions(suppress=True)
#         print('DCT block (rounded)\n', np.round(dct_block, decimals=1))
#         expected = invert_format_block(idct_2d(dct_block))
#         actual = self.get_original_matrix_from_func(dct_block)
#         print('Expected\n', expected)
#         print('Actual\n', actual)
#         assert(np.allclose(expected, actual))



#         surface = Surface(
#             lambda u, v: axes.c2p(*self.func(u, v, dct_block)),
#             u_range=[0, 7],
#             v_range=[0, 7],
#             checkerboard_colors=[REDUCIBLE_PURPLE],
#             fill_opacity=0.5,
#             resolution=8,
#             stroke_color=REDUCIBLE_YELLOW,
#             stroke_width=2,
#         )
        
#         self.position_image_on_axes(axes, block_image)

#         # self.add_fixed_in_frame_mobjects(block_image)
#         # lines_to_z, dots_z = self.get_lines_and_dots(axes, block[:, :, 1], block_image)
#         self.set_camera_orientation(theta=70 * DEGREES, phi=80 * DEGREES)
#         self.add(axes, block_image, surface)
#         self.wait()

#         number_line = self.initialize_slider(block_image, block[:, :, 1], surface)
#         block_image = self.animate_slider(number_line, axes, block_image, dct_block, block[:, :, 1], surface, block_image_2d, image_mob, original_image_mob)


#     def initialize_slider(self, block_image, block, surface):
#         number_line = NumberLine(
#             x_range=[0, 64, 8],
#             length=8,
#             color=REDUCIBLE_VIOLET,
#             include_numbers=False,
#             label_direction=UP,
#         )
#         self.add_fixed_in_frame_mobjects(number_line)
#         number_line.move_to(DOWN * 3)

#         self.play(
#             FadeIn(number_line)
#         )
#         self.wait()
#         return number_line

#     def animate_slider(self, number_line, axes, block_image, dct_block, original_block, surface, block_image_2d, image_mob, original_image_mob):
#         tick = Triangle().scale(0.15).set_color(REDUCIBLE_YELLOW)
#         tick.set_fill(color=REDUCIBLE_YELLOW, opacity=1)
#         self.add_fixed_in_frame_mobjects(tick)

#         tracker = ValueTracker(0)
#         tick.add_updater(
#             lambda m: m.next_to(
#                         number_line.n2p(tracker.get_value()),
#                         DOWN
#                     )
#         )
#         self.play( 
#             FadeIn(tick),
#         )
#         self.wait()
#         # surface_pos = RIGHT *
#         def get_new_block():
#             new_partial_block = self.get_partial_block(dct_block, tracker.get_value())
#             print(f'Partial block - {tracker.get_value()} components')
#             print('MSE', np.mean((new_partial_block - original_block) ** 2), '\n')

#             new_partial_block_image = self.get_block_image_for_3d(new_partial_block, height=axes.x_length)            
#             self.position_image_on_axes(axes, new_partial_block_image)
#             return new_partial_block_image

#         def get_new_surface():
#             new_partial_block_dct = self.get_partial_dct_block(dct_block, tracker.get_value())
#             # print('Generating surface of block:\n', new_partial_block_dct)
#             new_surface = Surface(
#                 lambda u, v: axes.c2p(*self.func(u, v, new_partial_block_dct)),
#                 u_range=[0, 7],
#                 v_range=[0, 7],
#                 checkerboard_colors=[REDUCIBLE_PURPLE, REDUCIBLE_PURPLE_DARKER],
#                 fill_opacity=0.5,
#                 resolution=16,
#                 stroke_color=REDUCIBLE_YELLOW,
#                 stroke_width=2,
#             )
#             return new_surface
        
#         def get_new_block_2d():
#             new_partial_block = self.get_partial_block(dct_block, tracker.get_value())

#             new_partial_block_image = self.get_image_vector_mob(new_partial_block, height=3)
            
#             # TODO comment out this line for the smooth transition
#             self.add_fixed_in_frame_mobjects(new_partial_block_image)
#             new_partial_block_image.move_to(block_image_2d.get_center())
#             return new_partial_block_image

#             original_pixel_array = image_mob.get_pixel_array()[2:298, 3:331, 0]
        
#         def get_new_image():
#             new_pixel_array = self.get_all_blocks(image_mob, 2, 298, 3, 331, tracker.get_value() / 5)
#             relevant_section = new_pixel_array[2:298, 3:331]
#             new_image = self.get_image_mob(new_pixel_array, height=None).move_to(UP * 1 + RIGHT * 2)
            
#             self.add_fixed_in_frame_mobjects(new_image)
#             new_image.move_to(original_image_mob.get_center())
#             return new_image
        
#         new_image = always_redraw(get_new_image)

#         partial_block_image_2d = always_redraw(get_new_block_2d)
#         # partial_pixel_grid_2d = self.get_pixel_grid(
#         #     partial_block_image_2d, partial_block_2d.shape[0]
#         # )
#         # partial_pixel_grid_2d.move_to(par)

#         partial_block = self.get_partial_block(dct_block, tracker.get_value())
#         partial_block_image = always_redraw(get_new_block)
#         partial_block_surface = always_redraw(get_new_surface)
#         self.play(
#             ReplacementTransform(block_image, partial_block_image),
#             ReplacementTransform(surface, partial_block_surface),
#             ReplacementTransform(block_image_2d, partial_block_image_2d),
#         )
#         self.wait()

#         self.remove(original_image_mob)
#         self.add_foreground_mobject(new_image)

#         tiny_square_highlight = Square(side_length=SMALL_BUFF * 0.8, color=REDUCIBLE_YELLOW)
#         surround_rect = SurroundingRectangle(partial_block_image_2d, buff=0, color=REDUCIBLE_YELLOW)
#         self.add_fixed_in_frame_mobjects(tiny_square_highlight, surround_rect)
#         tiny_square_highlight.move_to(new_image.get_center()).shift(UL * 0.2)
#         surround_rect.move_to(partial_block_image_2d.get_center())
        
#         rc = tiny_square_highlight.get_vertices()[0]
#         lc = tiny_square_highlight.get_vertices()[1]

#         end_rc = surround_rect.get_vertices()[0]
#         end_lc = surround_rect.get_vertices()[1]

#         right_dashed_line = DashedLine(rc, end_rc).set_stroke(color=REDUCIBLE_YELLOW)
#         left_dashed_line = DashedLine(lc, end_lc).set_stroke(color=REDUCIBLE_YELLOW)

#         self.add_fixed_in_frame_mobjects(right_dashed_line, left_dashed_line)

#         self.add_foreground_mobjects(tiny_square_highlight, surround_rect, left_dashed_line, right_dashed_line)
#         self.wait()   
       
#         self.play(
#             tracker.animate.set_value(64),
#             run_time=10,
#             rate_func=linear,
#         )

#         self.wait()

#         self.play(
#             tracker.animate.set_value(8),
#             run_time=8,
#             rate_func=linear,
#         )

#         self.wait()

#         self.play(
#             tracker.animate.set_value(64),
#             run_time=8,
#             rate_func=linear,
#         )
#         self.wait()
#         return partial_block_image

#     def get_all_blocks(
#         self,
#         image_mob,
#         start_row,
#         end_row,
#         start_col,
#         end_col,
#         num_components,
#         block_size=8,
#     ):
#         pixel_array = image_mob.get_pixel_array()
#         new_pixel_array = np.zeros((pixel_array.shape[0], pixel_array.shape[1]))
#         for i in range(start_row, end_row, block_size):
#             for j in range(start_col, end_col, block_size):
#                 pixel_block = self.get_pixel_block_from_array(pixel_array, i, j)
#                 block_centered = format_block(pixel_block)
#                 dct_block = dct_2d(block_centered)
#                 # quantized_block = quantize(dct_block)
#                 # dequantized_block = dequantize(quantized_block)
#                 # invert_dct_block = idct_2d(dequantized_block)
#                 # compressed_block = invert_format_block(invert_dct_block)
#                 # all_in_range = (compressed_block >= 0) & (compressed_block <= 255)
#                 # if not all(all_in_range.flatten()):
#                 #     print(i, j)
#                 #     print(all_in_range)
#                 #     print('Bad array\n', compressed_block)
#                 #     print('Original array\n', pixel_block[:, :, 0])
#                 #     raise ValueError("All elements in compressed_block must be in range [0, 255]")
#                 # new_pixel_array[i:i+block_size, j:j+block_size] = compressed_block
#                 partial_block = self.get_partial_block(dct_block, num_components)
#                 new_pixel_array[i : i + block_size, j : j + block_size] = partial_block

#         return new_pixel_array

#     def get_pixel_block_from_array(self, pixel_array, start_row, start_col, block_size=8):
#         return pixel_array[
#             start_row : start_row + block_size, start_col : start_col + block_size
#         ]

class AnimationIntroDiagram(JPEGDiagramMap):
    def construct(self):
        self.build_diagram()

    def build_diagram(self):
        self.play(self.camera.frame.animate.scale(2))
        # input image

        red_channel = (
            RoundedRectangle(corner_radius=0.1, fill_color=PURE_RED, width=3)
            .set_color(PURE_RED)
            .set_opacity(1)
            .set_stroke(WHITE, width=4)
        )
        green_channel = (
            RoundedRectangle(corner_radius=0.1, fill_color=PURE_GREEN, width=3)
            .set_color(PURE_GREEN)
            .set_opacity(1)
            .set_stroke(WHITE, width=4)
        )
        blue_channel = (
            RoundedRectangle(corner_radius=0.1, fill_color=PURE_BLUE, width=3)
            .set_color(PURE_BLUE)
            .set_opacity(1)
            .set_stroke(WHITE, width=4)
        )

        channels_vg_diagonal = VGroup(red_channel, green_channel, blue_channel).arrange(
            DOWN * 1.1 + RIGHT * 1.7, buff=-1.4
        )

        # output image
        output_image = SVGMobject("jpg_file.svg").set_stroke(
            WHITE, width=5, background=True
        )

        # big modules
        jpeg_encoder = Module(
            "JPEG Encoder",
            width=7,
            height=3,
            text_position=DOWN,
            text_weight=BOLD,
            text_scale=0.8,
        )
        jpeg_decoder = Module(
            "JPEG Decoder",
            width=7,
            height=3,
            text_position=UP,
            text_weight=BOLD,
            text_scale=0.8,
        )

        # color treatment
        color_treatment = Module(
            "Color treatment",
            REDUCIBLE_GREEN_DARKER,
            REDUCIBLE_GREEN_LIGHTER,
            height=jpeg_encoder.height,
            width=3,
            text_scale=0.5,
            text_position=UP,
            text_weight=BOLD,
        )

        ycbcr_m = Module(
            "YCbCr",
            fill_color=REDUCIBLE_YELLOW_DARKER,
            stroke_color=REDUCIBLE_YELLOW,
            height=1,
        ).scale_to_fit_width(color_treatment.width - 0.5)

        chroma_sub_m = Module(
            "Chroma Subsampling",
            fill_color=REDUCIBLE_YELLOW_DARKER,
            stroke_color=REDUCIBLE_YELLOW,
            text_scale=0.5,
            height=1,
        ).scale_to_fit_width(color_treatment.width - 0.5)

        general_scale = 1.3
        color_modules = VGroup(ycbcr_m, chroma_sub_m).arrange(DOWN, buff=0.5)

        color_treatment_w_modules = VGroup(color_treatment, color_modules).arrange(
            ORIGIN
        ).scale(general_scale)
        color_modules.shift(DOWN * 0.4)

        # small modules

        # encoding
        forward_dct_m = Module("Forward DCT", REDUCIBLE_YELLOW_DARKER, REDUCIBLE_YELLOW)
        forward_dct_icon = ImageMobject("dct.png").scale(0.2)
        forward_dct = Group(forward_dct_m, forward_dct_icon).arrange(DOWN, buff=0.5)

        quantizer_m = Module("Quantization", REDUCIBLE_YELLOW_DARKER, REDUCIBLE_YELLOW)
        quantizer_icon = ImageMobject("quantization.png").scale(0.2)
        quantizer = Group(quantizer_m, quantizer_icon).arrange(DOWN, buff=0.5)

        lossless_comp_m = Module(
            ["Lossless", "Encoder"],
            REDUCIBLE_YELLOW_DARKER,
            REDUCIBLE_YELLOW,
        )
        lossless_icon = ImageMobject("lossless.png").scale(0.2)
        lossless_comp = Group(lossless_comp_m, lossless_icon).arrange(DOWN, buff=0.5)

        encoding_modules = (
            Group(forward_dct, quantizer, lossless_comp)
            .arrange(RIGHT, buff=0.7)
            .scale_to_fit_width(jpeg_encoder.width - 0.5)
        )
        jpeg_encoder_w_modules = Group(jpeg_encoder, encoding_modules).arrange(
            ORIGIN,
        ).scale(general_scale)
        encoding_modules.shift(DOWN * 0.5)

        # decoding

        inverse_dct = Module("Inverse DCT", REDUCIBLE_YELLOW_DARKER, REDUCIBLE_YELLOW)
        dequantizer = Module("Dequantizer", REDUCIBLE_YELLOW_DARKER, REDUCIBLE_YELLOW)
        decoder = Module("Decoder", REDUCIBLE_YELLOW_DARKER, REDUCIBLE_YELLOW)

        decoding_modules = (
            VGroup(inverse_dct, dequantizer, decoder)
            .arrange(RIGHT, buff=0.5)
            .scale_to_fit_width(jpeg_decoder.width - 0.5)
        )
        jpeg_decoder_w_modules = VGroup(jpeg_decoder, decoding_modules).arrange(ORIGIN)
        decoding_modules.shift(DOWN * 0.5)

        # first row = encoding flow
        encoding_flow = Group(
            channels_vg_diagonal.scale(0.6),
            color_treatment_w_modules,
            jpeg_encoder_w_modules,
            output_image,
        ).arrange(RIGHT, buff=3)

        # second row = decoding flow
        decoding_flow = VGroup(
            output_image.copy(), jpeg_decoder_w_modules, channels_vg_diagonal.copy()
        ).arrange(RIGHT, buff=3)

        whole_map = Group(encoding_flow, decoding_flow).arrange(DOWN, buff=8)

        encode_arrows = VGroup()
        for i in range(len(encoding_flow.submobjects) - 1):
            encode_arrows.add(
                Arrow(
                    color=GRAY_B,
                    start=encoding_flow[i].get_right(),
                    end=encoding_flow[i + 1].get_left(),
                    stroke_width=3,
                    buff=0.3,
                    max_tip_length_to_length_ratio=0.08,
                    max_stroke_width_to_length_ratio=2,
                )
            )
        decode_arrows = VGroup()
        for i in range(len(decoding_flow.submobjects) - 1):
            decode_arrows.add(
                Arrow(
                    color=GRAY_B,
                    start=decoding_flow[i].get_right(),
                    end=decoding_flow[i + 1].get_left(),
                    stroke_width=3,
                    buff=0.3,
                    max_tip_length_to_length_ratio=0.08,
                    max_stroke_width_to_length_ratio=2,
                )
            )

        # whole map view state
        self.camera.frame.save_state()

        # self.focus_on(encoding_flow, buff=1.3)


        self.play(
            Create(channels_vg_diagonal)
        )
        self.play(
            Write(encode_arrows[0])
        )
        self.play(
            FadeIn(color_treatment_w_modules)
        )
        self.play(
            Write(encode_arrows[1])
        )

        self.play(
            FadeIn(jpeg_encoder_w_modules)
        )

        self.play(
            Write(encode_arrows[2])
        )

        self.play(
            FadeIn(output_image)
        )
        self.wait()

        NO_FADE = 1
        MEDIUM_FADE = 0.75
        HIGH_FADE = 0.3
        highlight_rect_ycbr = self.get_highlight_rect(ycbcr_m)
        self.play(
            FadeIn(highlight_rect_ycbr),
            jpeg_encoder.animate.set_opacity(HIGH_FADE),
            color_treatment.animate.set_opacity(MEDIUM_FADE),
            chroma_sub_m.animate.set_opacity(MEDIUM_FADE),
            forward_dct_m.animate.set_opacity(HIGH_FADE),
            quantizer_m.animate.set_opacity(HIGH_FADE),
            lossless_comp_m.animate.set_opacity(HIGH_FADE),
            lossless_icon.animate.set_opacity(HIGH_FADE),
            quantizer_icon.animate.set_opacity(HIGH_FADE),
            forward_dct_icon.animate.set_opacity(HIGH_FADE),

        )
        self.wait()

        highlight_rect_chroma = self.get_highlight_rect(chroma_sub_m)

        self.play(
            FadeOut(highlight_rect_ycbr),
            FadeIn(highlight_rect_chroma),
            ycbcr_m.animate.set_opacity(MEDIUM_FADE),
            chroma_sub_m.animate.set_opacity(NO_FADE),

        )

        self.wait()

        highlight_rect_dct = self.get_highlight_rect(forward_dct_m)
        self.play(
            FadeOut(highlight_rect_chroma),
            FadeIn(highlight_rect_dct),
            color_treatment.animate.set_opacity(HIGH_FADE),
            ycbcr_m.animate.set_opacity(HIGH_FADE),
            chroma_sub_m.animate.set_opacity(HIGH_FADE),
            forward_dct_m.animate.set_opacity(NO_FADE),
            forward_dct_icon.animate.set_opacity(NO_FADE),
            jpeg_encoder.animate.set_opacity(MEDIUM_FADE),
            quantizer_m.animate.set_opacity(MEDIUM_FADE),
            lossless_comp_m.animate.set_opacity(MEDIUM_FADE),
            lossless_icon.animate.set_opacity(MEDIUM_FADE),
            quantizer_icon.animate.set_opacity(MEDIUM_FADE),
        )

        self.wait()

        highlight_rect_quant = self.get_highlight_rect(quantizer_m)


        self.play(
            FadeOut(highlight_rect_dct),
            FadeIn(highlight_rect_quant),
            forward_dct_m.animate.set_opacity(MEDIUM_FADE),
            forward_dct_icon.animate.set_opacity(MEDIUM_FADE),
            quantizer_m.animate.set_opacity(NO_FADE),
            quantizer_icon.animate.set_opacity(NO_FADE),
        )
        self.wait()

        highlight_rect_lossless = self.get_highlight_rect(lossless_comp_m)

        self.play(
            FadeOut(highlight_rect_quant),
            FadeIn(highlight_rect_lossless),
            quantizer_m.animate.set_opacity(MEDIUM_FADE),
            quantizer_icon.animate.set_opacity(MEDIUM_FADE),
            lossless_comp_m.animate.set_opacity(NO_FADE),
            lossless_icon.animate.set_opacity(NO_FADE),
        )
        self.wait()

        # self.wait(3)

        # self.focus_on(channels_vg_diagonal)

        # self.wait(3)

        # self.focus_on(color_treatment_w_modules)

        # self.wait(3)

        # # self.focus_on(encoding_flow, buff=1.3)

        # self.wait(3)

        # self.focus_on(jpeg_encoder_w_modules, buff=1.3)

        # self.wait(3)

        # self.focus_on(forward_dct)

        # self.wait(3)

        # self.focus_on(quantizer)

        # self.wait(3)

        # self.focus_on(lossless_comp)

        # self.wait(3)

        # self.play(Restore(self.camera.frame), run_time=3)

    def get_highlight_rect(self, module):
        rect = module.rect.copy().set_fill(opacity=0).set_stroke(color=module.rect.get_stroke_color(), width=10, opacity=1)
        return rect

class AnimateIntroRect(Scene):
    def construct(self):
        self.wait()
        screen_rect = ScreenRectangle(height=4.5).shift(DOWN * 1)
        self.play(
            Create(screen_rect)
        )
        self.wait()

class SponsorRect(Scene):
    def construct(self):
        self.wait()
        screen_rect = ScreenRectangle(height=5.5).shift(ORIGIN)
        self.play(
            Create(screen_rect)
        )
        self.wait()


class SponsorshipMessage(Scene):
    def construct(self):
        link = Tex("brilliant.org/Reducible").set_color(REDUCIBLE_YELLOW)
        link.move_to(DOWN * 1.5)
        special_offer = Tex(
            "First 200 members to sign up get" + "\\\\",
            r"20\% off the annual subscription!"
        )

        link.scale(1.2)

        self.play(
            Write(link)
        )

        self.wait()

        special_offer.next_to(link, DOWN)

        self.play(
            FadeIn(special_offer)
        )
        self.wait()

class SampleLessColors(ThreeDScene):
    def construct(self):

        self.animate_sample_less_colors()

    def animate_sample_less_colors(self):
        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)

        yuv_plane_8 = self.create_yuv_plane(y=250, color_res=8)
        yuv_plane_4 = self.create_yuv_plane(y=250, color_res=4)

        uv_plane_8 = VGroup()
        for row in yuv_plane_8:
            for color in row:
                uv_plane_8.add(
                    Cube(
                        fill_color=rgb_to_hex(color / 255), side_length=0.5
                    ).set_opacity(1)
                )

        uv_plane_8.arrange_in_grid(
            rows=yuv_plane_8.shape[1], cols=yuv_plane_8.shape[0], buff=0.5
        ).move_to(ORIGIN)

        uv_plane_4 = VGroup()
        for row in yuv_plane_4:
            for color in row:
                uv_plane_4.add(
                    Cube(
                        fill_color=rgb_to_hex(color / 255), side_length=0.5
                    ).set_opacity(1)
                )

        uv_plane_4.arrange_in_grid(
            rows=yuv_plane_4.shape[1], cols=yuv_plane_4.shape[0], buff=0.5
        ).scale_to_fit_height(uv_plane_8.height).scale_to_fit_width(
            uv_plane_8.width
        ).move_to(
            ORIGIN
        )

        y_values = np.arange(0, 255, 28)

        y_column = (
            VGroup(
                *[
                    Cube(
                        side_length=0.5, fill_color=gray_scale_value_to_hex(y)
                    ).set_opacity(1)
                    for y in y_values
                ]
            )
            .arrange(OUT)
            .move_to(ORIGIN)
        )

        self.play(
            FadeIn(y_column),
        )
        self.add_foreground_mobject(y_column)
        self.wait()
        self.play(
            LaggedStartMap(FadeIn, uv_plane_8),
            y_column.animate.shift((DOWN + RIGHT) * 0.000001),
            run_time=3
        )

        self.wait()
        self.play(
            Transform(uv_plane_8, uv_plane_4),
            y_column.animate.shift((DOWN + RIGHT) * 0.000001),
            run_time=2
        )
        self.wait()
  

    def create_yuv_plane(self, y=127, color_res=64):
        color_plane_data = [
            [y, u * (256 // color_res), v * (256 // color_res)]
            for u in range(color_res)
            for v in range(color_res)
        ]

        rgb_conv = np.array(
            [ycbcr2rgb4map(c) for c in color_plane_data], dtype=np.uint8
        ).reshape((color_res, color_res, 3))

        return rgb_conv

class DSPIntro(Scene):
    def construct(self):
        screen_rect = ScreenRectangle(height=5).move_to(UP * 0.5)

        orthogonal_transforms = Tex("Orthogonality is essential to the DCT").next_to(screen_rect, DOWN)

        entire_class = Tex("Orthogonal transforms form the core" + "\\\\" + "of modern ", " digital signal processing")
        entire_class[1].set_color(REDUCIBLE_YELLOW)
        entire_class.next_to(screen_rect, DOWN)

        self.add(screen_rect)
        self.wait()

        self.play(
            Write(orthogonal_transforms)
        )
        self.wait()

        self.play(
            ReplacementTransform(orthogonal_transforms, entire_class)
        )
        self.wait()

class AudioVideoMention(Scene):
    def construct(self):
        screen_rect = ScreenRectangle(height=5).move_to(UP * 0.5)
        similar_ideas = Tex(
            "Similar ideas show up in MP3/Dolby/AAC audio compression" + "\\\\",
            "and H.264/H.265/AV1/VP9 video compression").scale(0.8)
        similar_ideas.next_to(screen_rect, DOWN)

        self.add(screen_rect)
        self.wait()

        self.play(
            Write(similar_ideas[0])
        )
        self.wait()

        self.play(
            Write(similar_ideas[1])
        )
        self.wait()
        

class EnergyCompaction(Scene):
    def construct(self):
        energy = Tex("Energy" + "\\\\" + "Compaction").scale(1)
        energy.move_to(DOWN * 2.2)
        self.play(
            Write(energy)
        )
        self.wait()

        self.play(
            Circumscribe(energy, color=REDUCIBLE_YELLOW),
            energy.animate.set_color(REDUCIBLE_YELLOW)
        )
        self.wait()

class RemovingHigherFrequencies(Scene):
    def construct(self):
        question = Tex("How do we remove higher frequency components?")
        question.move_to(UP * 3)

        self.play(
            Write(question)
        )
        self.wait()

class BigPicturePerspective(JPEGDiagramMap):
    def construct(self):
        start_width, start_height = 12, 7
        start_text_scale = 1.2
        scale_factor = 3
        compression = Module(
            "Compression", 
            text_scale=1.2, 
            text_position=UP, 
            text_weight=MEDIUM, 
            width=start_width, 
            height=start_height
        )
        self.play(
            FadeIn(compression)
        )
        self.wait()

        redundancy = Module(
            "Redundancy", 
            text_scale=0.6, 
            height=start_height/scale_factor, 
            width=start_width/scale_factor, 
            text_position=ORIGIN, 
            fill_color=REDUCIBLE_GREEN_DARKER,
            stroke_color=REDUCIBLE_GREEN_LIGHTER,
            text_weight=MEDIUM,
            stroke_width=4
        ).move_to(LEFT * 3)
        print(redundancy.height)
        self.play(
            FadeIn(redundancy)
        )
        self.wait()

        self.focus_on(redundancy, buff=1.2)

        self.wait()

        lossless_compression = Module(
            "Lossless", 
            text_scale=0.4,
            height=start_height/(scale_factor * 2), 
            width=start_width/(scale_factor * 2),
            stroke_color=REDUCIBLE_YELLOW,
            fill_color=REDUCIBLE_YELLOW_DARKER,
            text_position=ORIGIN,
            text_weight=MEDIUM,
            stroke_width=3
        ).move_to(redundancy.get_center() + DOWN * SMALL_BUFF)
        self.play(
            redundancy.text.animate.scale(0.8).shift(UP * (redundancy.height / 2 - SMALL_BUFF * 3)),
            FadeIn(lossless_compression)
        )

        self.wait()

        self.focus_on(lossless_compression, buff=1.2)

        self.wait()

        png = Module(
            "PNG",
            height=0.45,
            width=0.85,
            text_scale=0.2,
            text_weight=MEDIUM,
            stroke_width=2,
        ).move_to(lossless_compression.get_center())

        self.play(
            lossless_compression.text.animate.scale(0.7).shift(UP * (lossless_compression.height / 2 - SMALL_BUFF * 1.5)),
            FadeIn(png)
        )
        self.wait()

        self.focus_on(compression, buff=1.1)

        self.wait()

        human = Module(
            "Perception", 
            text_scale=0.6, 
            height=start_height/scale_factor, 
            width=start_width/scale_factor, 
            text_position=ORIGIN, 
            fill_color=REDUCIBLE_GREEN_DARKER,
            stroke_color=REDUCIBLE_GREEN_LIGHTER,
            text_weight=MEDIUM,
            stroke_width=4
        ).move_to(RIGHT * 3)

        self.play(
            FadeIn(human)
        )

        self.focus_on(human, buff=1.1)
        self.wait()

        lossy_compression = Module(
            "Lossy", 
            text_scale=0.4,
            height=start_height/(scale_factor * 2), 
            width=start_width/(scale_factor * 2),
            stroke_color=REDUCIBLE_YELLOW,
            fill_color=REDUCIBLE_YELLOW_DARKER,
            text_position=ORIGIN,
            text_weight=MEDIUM,
            stroke_width=3
        ).move_to(human.get_center() + DOWN * SMALL_BUFF)

        self.play(
            human.text.animate.scale(0.8).shift(UP * (human.height / 2 - SMALL_BUFF * 3)),
            FadeIn(lossy_compression)
        )
        self.wait()

        JPEG = Module(
            "JPEG",
            height=0.45,
            width=0.85,
            text_scale=0.2,
            text_weight=MEDIUM,
            stroke_width=2,
        ).move_to(lossy_compression.get_center())

        self.focus_on(lossy_compression, buff=1.2)

        self.wait()

        self.play(
            lossy_compression.text.animate.scale(0.7).shift(UP * (lossy_compression.height / 2 - SMALL_BUFF * 1.5)),
            FadeIn(JPEG)
        )

        self.wait()
        scale = 0.55
        jpeg_smaller = JPEG.copy().scale(scale).shift(LEFT * 0.65)

        audio_rect = jpeg_smaller.rect.copy().move_to(lossy_compression.get_center())
        audio_text = Text("MP3/Dolby", font="CMU Serif", weight=MEDIUM).scale_to_fit_height(jpeg_smaller.text.height)
        audio_text.move_to(audio_rect.get_center()).shift(DOWN * SMALL_BUFF * 0.5)

        audio = Text("Audio", font="CMU Serif", weight=MEDIUM).scale_to_fit_height(audio_text.height + 0.01)
        audio.move_to(audio_rect.get_center()).shift(UP * SMALL_BUFF * 0.5)

        video_rect = jpeg_smaller.rect.copy().move_to(lossy_compression.get_center()).shift(RIGHT * 0.65)
        video_text = Text("H.264/H.265", font="CMU Serif", weight=MEDIUM).scale_to_fit_height(jpeg_smaller.text.height)
        video_text.move_to(video_rect.get_center()).shift(DOWN * SMALL_BUFF * 0.5)

        video = Text("Video", font="CMU Serif", weight=MEDIUM).scale_to_fit_height(audio_text.height + 0.01)
        video.move_to(video_rect.get_center()).shift(UP * SMALL_BUFF * 0.5)
        
        images = Text("Images", font="CMU Serif", weight=MEDIUM).scale_to_fit_height(audio.height + 0.015)
        images.move_to(jpeg_smaller.get_center()).shift(UP * SMALL_BUFF * 0.44)
        self.play(
            Transform(JPEG.rect, jpeg_smaller.rect),
            JPEG.text.animate.scale(0.44).shift(LEFT * 0.65 + DOWN * SMALL_BUFF * 0.46),
            FadeIn(audio_rect),
            FadeIn(audio_text),
            FadeIn(audio),
            FadeIn(video_rect),
            FadeIn(video_text),
            FadeIn(video),
            FadeIn(images)
        )
        self.wait()

        self.focus_on(compression, buff=1.1, run_time=6)

    def focus_on(self, mobject, buff=2, run_time=3):
        self.play(
            self.camera.frame.animate.set_width(mobject.width * buff).move_to(mobject),
            run_time=run_time,
        )

class FocusOnY(Scene):
    def construct(self):
        text = Tex("We'll focus on the luma (Y) component.").scale(0.9)
        text.move_to(DOWN * 3)

        self.play(
            Write(text)
        )
        self.wait()

        color_components = Tex("Same concepts work on color (Cb/Cr) components.").scale(0.9)
        color_components.move_to(text.get_center())

        self.play(
            ReplacementTransform(text, color_components)
        )
        self.wait()

        self.play(
            FadeOut(color_components)
        )
        self.wait()

class Patreons(Scene):
    def construct(self):
        thanks = Tex("Special Thanks to These Patreons").scale(1.2)
        patreons = ["Burt Humburg", "Winston Durand", r"Adam D\v{r}ínek"]
        patreon_text = VGroup(*[thanks] + [Tex(name).scale(0.9) for name in patreons])
        patreon_text.arrange(DOWN)
        patreon_text.to_edge(DOWN)

        self.play(
            Write(patreon_text[0])
        )
        self.play(
            *[Write(text) for text in patreon_text[1:]]
        )
        self.wait()

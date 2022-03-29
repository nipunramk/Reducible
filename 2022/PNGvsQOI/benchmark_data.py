from manim import *
from reducible_colors import *

np.random.seed(1)
config["assets_dir"] = "assets"


class BenchmarkResults(Scene):
    def construct(self):
        title = (
            Text("Benchmark Results", font="CMU Serif", weight="BOLD")
            .scale(0.8)
            .to_edge(UP)
        )
        annotation = (
            Text(
                "* Experiments run on MacMini M1 16GB using libpng/qoi source code",
                font="CMU Serif",
            )
            .scale(0.3)
            .to_corner(DR, buff=0.3)
        )

        self.play(Write(title))

        encode_ms = SVGMobject("encode_ms.svg").scale(3).shift(DOWN * 0.6)
        decode_ms = SVGMobject("decode_ms.svg").scale(3).shift(DOWN * 0.6)
        comp_rate = SVGMobject("comp_rate.svg").scale(3).shift(DOWN * 0.6)

        icon_1 = ImageMobject("BenchmarkResults/icon_1.png").scale(4)
        icon_2 = ImageMobject("BenchmarkResults/icon_2.png").scale(4)

        plant_1 = ImageMobject("BenchmarkResults/plant_1.png").scale(0.2)
        plant_2 = ImageMobject("BenchmarkResults/plant_2.png").scale(0.2)

        kodak_1 = ImageMobject("BenchmarkResults/kodak_1.png").scale(0.7)
        kodak_2 = ImageMobject("BenchmarkResults/kodak_2.png").scale(0.7)

        game_1 = ImageMobject("BenchmarkResults/game_1.png").scale(0.45)
        game_2 = ImageMobject("BenchmarkResults/game_2.png").scale(0.45)

        texture_1 = ImageMobject("BenchmarkResults/texture_1.png").scale(0.7)
        texture_2 = ImageMobject("BenchmarkResults/texture_2.png").scale(0.7)

        purple_sq = (
            Square(color=REDUCIBLE_PURPLE, fill_color=REDUCIBLE_PURPLE)
            .set_opacity(1)
            .scale(0.2)
        )
        png_text = Text("png", font="SF Mono").scale(0.5)
        p_l = VGroup(purple_sq, png_text).arrange(RIGHT, buff=0.2)

        yellow_sq = (
            Square(color=REDUCIBLE_YELLOW, fill_color=REDUCIBLE_YELLOW)
            .set_opacity(1)
            .scale(0.2)
        )
        qoi_text = Text("qoi", font="SF Mono").scale(0.5)
        y_l = VGroup(yellow_sq, qoi_text).arrange(RIGHT, buff=0.2)

        legend = (
            VGroup(p_l, y_l)
            .arrange(DOWN, buff=0.3, aligned_edge=LEFT)
            .scale(0.7)
            .next_to(comp_rate, UP + RIGHT * 0.5, buff=-1)
        )

        images = [
            icon_1,
            icon_2,
            plant_1,
            plant_2,
            kodak_1,
            kodak_2,
            game_1,
            game_2,
            texture_1,
            texture_2,
        ]
        [
            mob.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
            for mob in images
        ]

        texts = [
            Text("Icons", font="CMU Serif", weight=BOLD, should_center=False).scale(
                0.3
            ),
            Text(
                "Plants \n(transparent background)",
                font="CMU Serif",
                weight=BOLD,
                should_center=False,
            ).scale(0.3),
            Text(
                "Analog Images", font="CMU Serif", weight=BOLD, should_center=False
            ).scale(0.3),
            Text(
                "Game Stills", font="CMU Serif", weight=BOLD, should_center=False
            ).scale(0.3),
            Text("Textures", font="CMU Serif", weight=BOLD, should_center=False).scale(
                0.3
            ),
        ]

        self.play(Write(comp_rate), Write(legend))

        self.wait()

        self.play(
            comp_rate.animate.scale(0.8).to_edge(LEFT, buff=1),
            legend.animate.scale(0.8).next_to(comp_rate, UP * 0.5 + RIGHT, buff=-4),
            FadeIn(annotation),
        )

        for i in range(0, len(images), 2):
            images[i + 1].next_to(comp_rate, RIGHT, buff=2.5)

            # small adjustments for particular cases
            if i == 0 or i == 2:
                images[i + 1].shift(RIGHT)

            if i == 8:
                images[i + 1].shift(RIGHT * 0.6)

            images[i].move_to(images[i + 1]).shift(
                DOWN * (images[i + 1].height / 3) + RIGHT * (images[i + 1].width / 4)
            )

            texts[i // 2].next_to(comp_rate, UP, buff=0).shift(RIGHT * 5)

            self.play(
                FadeIn(images[i + 1]),
                FadeIn(images[i]),
                FadeIn(texts[i // 2]),
                run_time=0.7,
            )
            self.wait(0.2)
            self.play(
                FadeOut(texts[i // 2]),
                LaggedStart(FadeOut(images[i + 1]), FadeOut(images[i]), lag_ratio=0.5),
                run_time=0.7,
            )

        self.play(FadeOut(comp_rate), FadeOut(legend), FadeOut(annotation))
        self.wait()

        self.play(
            Write(encode_ms),
            Write(legend.next_to(encode_ms, RIGHT + UP, buff=0).shift(DOWN * 2)),
        )

        self.wait()

        self.play(FadeOut(encode_ms))

        self.wait()

        self.play(Write(decode_ms))

        self.wait()

        self.play(FadeOut(decode_ms), FadeOut(legend))

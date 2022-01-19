"""
File to define all the classes used in the animations.
"""


from manim import *
from typing import Iterable, List
from reducible_colors import *
from functions import *


class ReducibleBarChart(BarChart):
    """
    Redefinition of the BarChart class to add font personalization
    """

    def __init__(
        self,
        values: Iterable[float],
        height: float = 4,
        width: float = 6,
        n_ticks: int = 4,
        tick_width: float = 0.2,
        chart_font: str = "SF Mono",
        label_y_axis: bool = True,
        y_axis_label_height: float = 0.25,
        max_value: float = 1,
        bar_colors=...,
        bar_fill_opacity: float = 0.8,
        bar_stroke_width: float = 3,
        bar_names: List[str] = ...,
        bar_label_scale_val: float = 0.75,
        **kwargs,
    ):
        self.chart_font = chart_font

        super().__init__(
            values,
            height=height,
            width=width,
            n_ticks=n_ticks,
            tick_width=tick_width,
            label_y_axis=label_y_axis,
            y_axis_label_height=y_axis_label_height,
            max_value=max_value,
            bar_colors=bar_colors,
            bar_fill_opacity=bar_fill_opacity,
            bar_stroke_width=bar_stroke_width,
            bar_names=bar_names,
            bar_label_scale_val=bar_label_scale_val,
            **kwargs,
        )

    def add_axes(self):
        x_axis = Line(self.tick_width * LEFT / 2, self.total_bar_width * RIGHT)
        y_axis = Line(ORIGIN, self.total_bar_height * UP)
        ticks = VGroup()
        heights = np.linspace(0, self.total_bar_height, self.n_ticks + 1)
        values = np.linspace(0, self.max_value, self.n_ticks + 1)
        for y, _value in zip(heights, values):
            tick = Line(LEFT, RIGHT)
            tick.width = self.tick_width
            tick.move_to(y * UP)
            ticks.add(tick)
        y_axis.add(ticks)

        self.add(x_axis, y_axis)
        self.x_axis, self.y_axis = x_axis, y_axis

        if self.label_y_axis:
            labels = VGroup()
            for tick, value in zip(ticks, values):
                label = Text(str(np.round(value, 2)), font=self.chart_font, weight=MEDIUM)
                label.height = self.y_axis_label_height
                label.next_to(tick, LEFT, SMALL_BUFF)
                labels.add(label)
            self.y_axis_labels = labels
            self.add(labels)

    def add_bars(self, values):
        buff = float(self.total_bar_width) / (2 * len(values) + 1)
        bars = VGroup()
        for i, value in enumerate(values):
            bar = Rectangle(
                height=(value / self.max_value) * self.total_bar_height,
                width=buff,
                stroke_width=self.bar_stroke_width,
                fill_opacity=self.bar_fill_opacity,
            )
            bar.move_to((2 * i + 1) * buff * RIGHT, DOWN + LEFT)
            bars.add(bar)
        bars.set_color_by_gradient(*self.bar_colors)

        bar_labels = VGroup()
        for bar, name in zip(bars, self.bar_names):
            label = Text(str(name), font="SF Mono", weight=MEDIUM)
            label.scale(self.bar_label_scale_val)
            label.next_to(bar, DOWN, SMALL_BUFF)
            bar_labels.add(label)

        self.add(bars, bar_labels)
        self.bars = bars
        self.bar_labels = bar_labels


class Module(VGroup):
    def __init__(
        self,
        text,
        fill_color=REDUCIBLE_PURPLE_DARKER,
        stroke_color=REDUCIBLE_VIOLET,
        stroke_width=5,
        text_scale=0.9,
        text_position=ORIGIN,
        text_weight=NORMAL,
        width=4,
        height=2,
        **kwargs,
    ):

        self.rect = (
            RoundedRectangle(
                corner_radius=0.1, fill_color=fill_color, width=width, height=height
            )
            .set_opacity(1)
            .set_stroke(stroke_color, width=stroke_width)
        )
        if isinstance(text, list):
            text_mob = VGroup()
            for string in text:
                text = Text(str(string), weight=text_weight, font="CMU Serif").scale(text_scale)
                text_mob.add(text)
            self.text = text_mob.arrange(DOWN, buff=SMALL_BUFF * 2)
        else:
            self.text = Text(str(text), weight=text_weight, font="CMU Serif").scale(
            text_scale
            )
        self.text.next_to(
            self.rect,
            direction=ORIGIN,
            coor_mask=text_position * 0.8,
            aligned_edge=text_position,
        )

        super().__init__(self.rect, self.text, **kwargs)
        # super().arrange(ORIGIN)


class Pixel(Square):
    def __init__(self, n: int, color_mode: str):
        assert color_mode in ("RGB", "GRAY"), "Color modes are RGB and GRAY"
        if color_mode == "RGB":
            color = rgb_to_hex(n / 255)
        else:
            color = g2h(n / 255)

        super().__init__(side_length=1)

        self.set_stroke(BLACK, width=0.2)
        self.set_fill(color, opacity=1)
        self.color = color


class PixelArray(VGroup):
    def __init__(self, img: np.ndarray, include_numbers=False, color_mode="RGB"):

        if len(img.shape) == 3:
            rows, cols, channels = img.shape
        else:
            rows, cols = img.shape

        self.shape = img.shape

        pixels = []
        for row in img:
            for p in row:
                if include_numbers:
                    self.number = (
                        Text(str(p), font="SF Mono", weight=MEDIUM)
                        .scale(0.7)
                        .set_color(g2h(1) if p < 180 else g2h(0))
                    )
                    pixels.append(VGroup(Pixel(p, color_mode), self.number))
                else:
                    print(p)
                    pixels.append(Pixel(p, color_mode))

        super().__init__(*pixels)
        self.arrange_in_grid(rows, cols, buff=0)

        self.dict = {index: p for index, p in enumerate(self)}

    def __getitem__(self, value) -> VGroup:
        if isinstance(value, slice):
            return VGroup(*list(self.dict.values())[value])
        else:
            return self.dict[value]


string_to_mob_map = {}


class RDecimalNumber(DecimalNumber):
    def set_submobjects_from_number(self, number):
        self.number = number
        self.submobjects = []

        num_string = self.get_num_string(number)
        self.add(*(map(self.string_to_mob, num_string)))

        # Add non-numerical bits
        if self.show_ellipsis:
            self.add(
                self.string_to_mob("\\dots", Text, color=self.color),
            )

        if self.unit is not None:
            self.unit_sign = self.string_to_mob(self.unit, Text)
            self.add(self.unit_sign)

        self.arrange(
            buff=self.digit_buff_per_font_unit * self._font_size,
            aligned_edge=DOWN,
        )

        # Handle alignment of parts that should be aligned
        # to the bottom
        for i, c in enumerate(num_string):
            if c == "-" and len(num_string) > i + 1:
                self[i].align_to(self[i + 1], UP)
                self[i].shift(self[i + 1].height * DOWN / 2)
            elif c == ",":
                self[i].shift(self[i].height * DOWN / 2)
        if self.unit and self.unit.startswith("^"):
            self.unit_sign.align_to(self, UP)

        # track the initial height to enable scaling via font_size
        self.initial_height = self.height

        if self.include_background_rectangle:
            self.add_background_rectangle()

    def string_to_mob(self, string, mob_class=Text, **kwargs):
        if string not in string_to_mob_map:
            string_to_mob_map[string] = mob_class(
                string, font_size=1.0, font="SF Mono", weight=MEDIUM, **kwargs
            )
        mob = string_to_mob_map[string].copy()
        mob.font_size = self._font_size
        return mob


class RVariable(VMobject):
    """A class for displaying text that continuously updates to reflect the value of a python variable.

    Automatically adds the text for the label and the value when instantiated and added to the screen.

    Parameters
    ----------
    var : Union[:class:`int`, :class:`float`]
        The python variable you need to keep track of and display.
    label : Union[:class:`str`, :class:`~.Tex`, :class:`~.MathTex`, :class:`~.Text`, :class:`~.TexSymbol`, :class:`~.SingleStringMathTex`]
        The label for your variable, for example ``x = ...``. To use math mode, for e.g.
        subscripts, superscripts, etc. simply pass in a raw string.
    var_type : Union[:class:`DecimalNumber`, :class:`Integer`], optional
        The class used for displaying the number. Defaults to :class:`DecimalNumber`.
    num_decimal_places : :class:`int`, optional
        The number of decimal places to display in your variable. Defaults to 2.
        If `var_type` is an :class:`Integer`, this parameter is ignored.
    kwargs : Any
            Other arguments to be passed to `~.Mobject`.

    Attributes
    ----------
    label : Union[:class:`str`, :class:`~.Tex`, :class:`~.MathTex`, :class:`~.Text`, :class:`~.TexSymbol`, :class:`~.SingleStringMathTex`]
        The label for your variable, for example ``x = ...``.
    tracker : :class:`~.ValueTracker`
        Useful in updating the value of your variable on-screen.
    value : Union[:class:`DecimalNumber`, :class:`Integer`]
        The tex for the value of your variable.

    Examples
    --------
    Normal usage::

        # DecimalNumber type
        var = 0.5
        on_screen_var = Variable(var, Text("var"), num_decimal_places=3)
        # Integer type
        int_var = 0
        on_screen_int_var = Variable(int_var, Text("int_var"), var_type=Integer)
        # Using math mode for the label
        on_screen_int_var = Variable(int_var, "{a}_{i}", var_type=Integer)

    .. manim:: VariablesWithValueTracker

        class VariablesWithValueTracker(Scene):
            def construct(self):
                var = 0.5
                on_screen_var = Variable(var, Text("var"), num_decimal_places=3)

                # You can also change the colours for the label and value
                on_screen_var.label.set_color(RED)
                on_screen_var.value.set_color(GREEN)

                self.play(Write(on_screen_var))
                # The above line will just display the variable with
                # its initial value on the screen. If you also wish to
                # update it, you can do so by accessing the `tracker` attribute
                self.wait()
                var_tracker = on_screen_var.tracker
                var = 10.5
                self.play(var_tracker.animate.set_value(var))
                self.wait()

                int_var = 0
                on_screen_int_var = Variable(
                    int_var, Text("int_var"), var_type=Integer
                ).next_to(on_screen_var, DOWN)
                on_screen_int_var.label.set_color(RED)
                on_screen_int_var.value.set_color(GREEN)

                self.play(Write(on_screen_int_var))
                self.wait()
                var_tracker = on_screen_int_var.tracker
                var = 10.5
                self.play(var_tracker.animate.set_value(var))
                self.wait()

                # If you wish to have a somewhat more complicated label for your
                # variable with subscripts, superscripts, etc. the default class
                # for the label is MathTex
                subscript_label_var = 10
                on_screen_subscript_var = Variable(subscript_label_var, "{a}_{i}").next_to(
                    on_screen_int_var, DOWN
                )
                self.play(Write(on_screen_subscript_var))
                self.wait()

    .. manim:: VariableExample

        class VariableExample(Scene):
            def construct(self):
                start = 2.0

                x_var = Variable(start, 'x', num_decimal_places=3)
                sqr_var = Variable(start**2, 'x^2', num_decimal_places=3)
                Group(x_var, sqr_var).arrange(DOWN)

                sqr_var.add_updater(lambda v: v.tracker.set_value(x_var.tracker.get_value()**2))

                self.add(x_var, sqr_var)
                self.play(x_var.tracker.animate.set_value(5), run_time=2, rate_func=linear)
                self.wait(0.1)

    """

    def __init__(
        self, var, label, var_type=RDecimalNumber, num_decimal_places=2, **kwargs
    ):

        self.label = Text(label, font="SF Mono", weight=MEDIUM) if isinstance(label, str) else label
        equals = Text("=").next_to(self.label, RIGHT)
        self.label.add(equals)

        self.tracker = ValueTracker(var)

        self.value = RDecimalNumber(
            self.tracker.get_value(), num_decimal_places=num_decimal_places
        )

        self.value.add_updater(lambda v: v.set_value(self.tracker.get_value())).next_to(
            self.label,
            RIGHT,
        )

        super().__init__(**kwargs)
        self.add(self.label, self.value)

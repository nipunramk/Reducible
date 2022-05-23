"""
File for various PNG and QOI Helper Functions
"""

from manim import *

from reducible_colors import *
import numpy as np


def get_1d_index(i, j, pixel_array):
    return pixel_array.shape[1] * i + j


def is_last_pixel(channel, row, col):
    return row == channel.shape[0] - 1 and col == channel.shape[1] - 1


def qoi_hash(rgb_val):
    """
    Implemented as per https://qoiformat.org/qoi-specification.pdf
    """
    r, g, b = rgb_val
    return r * 3 + g * 5 + b * 7


def is_diff_small(dr, dg, db):
    return dr > -3 and dr < 2 and dg > -3 and dg < 2 and db > -3 and db < 2


def is_diff_med(dg, dr_dg, db_dg):
    return (
        dr_dg > -9 and dr_dg < 8 and dg > -33 and dg < 32 and db_dg > -9 and db_dg < 8
    )

def get_glowing_surround_rect(
    pixel, buff_min=0, buff_max=0.15, color=REDUCIBLE_YELLOW, n=40, opacity_multiplier=1
):
    glowing_rect = VGroup(
        *[
            SurroundingRectangle(pixel, buff=interpolate(buff_min, buff_max, b))
            for b in np.linspace(0, 1, n)
        ]
    )
    for i, rect in enumerate(glowing_rect):
        rect.set_stroke(color, width=0.5, opacity=1 - i / n)
    return glowing_rect

def get_glowing_surround_circle(
    circle, buff_min=0, buff_max=0.15, color=REDUCIBLE_YELLOW, n=40, opacity_multiplier=1
):
    current_radius = circle.width / 2
    glowing_circle = VGroup(
        *[
            Circle(radius=current_radius+interpolate(buff_min, buff_max, b))
            for b in np.linspace(0, 1, n)
        ]
    )
    for i, c in enumerate(glowing_circle):
        c.set_stroke(color, width=0.5, opacity=1- i / n)
    return glowing_circle.move_to(circle.get_center())

def g2h(n):
    """Abbreviation for grayscale to hex"""
    return rgb_to_hex((n, n, n))

def align_text_vertically(*text, buff=DEFAULT_MOBJECT_TO_MOBJECT_BUFFER, aligned_edge=ORIGIN):
	start_text = text[0]
	for i in range(1, len(text)):
		next_text = text[i]
		next_text.next_to(start_text, DOWN, buff=buff, aligned_edge=aligned_edge)
		start_text = next_text

	return VGroup(*text)

def get_matching_text(text, to_match, font='SF Mono', weight=MEDIUM, color=WHITE):
	return Text(text, font=font, weight=MEDIUM).scale_to_fit_height(to_match.height).move_to(to_match.get_center()).set_color(color)

def gray_scale_value_to_hex(value):
    assert value >= 0 and value <= 255, f'Invalid value {value}'
    integer_value = int(round(value))
    hex_string = hex(integer_value).split("x")[-1]
    if integer_value < 16:
        hex_string = "0" + hex_string
    return "#" + hex_string * 3

def g2h(n):
    """Abbreviation for grayscale to hex"""
    return rgb_to_hex((n, n, n))

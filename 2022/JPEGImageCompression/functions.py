"""
File to define all the useful mathematical functions used in the animations
"""

import numpy as np

from manim import *

from scipy import fftpack


def gray_scale_value_to_hex(value):
    assert value >= 0 and value <= 255, f'Invalid value {value}'
    integer_value = int(round(value))
    hex_string = hex(integer_value).split("x")[-1]
    if integer_value < 16:
        hex_string = "0" + hex_string
    return "#" + hex_string * 3


def make_lut_u():
    return np.array([[[i, 255 - i, 0] for i in range(256)]], dtype=np.uint8)


def make_lut_v():
    return np.array([[[0, 255 - i, i] for i in range(256)]], dtype=np.uint8)


def dct1D_manual(f, N):
    result = []
    constant = (2 / N) ** 0.5
    for u in range(N):
        component = 0
        if u == 0:
            factor = 1 / np.sqrt(2)
        else:
            factor = 1
        for i in range(N):
            component += (
                constant * factor * np.cos(np.pi * u / (2 * N) * (2 * i + 1)) * f(i)
            )

        result.append(component)

    return result


def f(i):
    return np.cos((2 * i + 1) * 3 * np.pi / 16)


def plot_function(f, N):
    import matplotlib.pyplot as plt

    x = np.arange(0, N, 0.001)
    y = f(x)

    plt.plot(x, y)
    plt.show()


def g(i):
    return np.cos((2 * i + 1) * 5 * np.pi / 16)


def h(i):
    return np.cos((2 * i + 1) * 3 * np.pi / 16) * np.cos((2 * i + 1) * 1.5 * np.pi / 16)


def get_dct_elem(i, j):
    return np.cos(j * (2 * i + 1) * np.pi / 16)


def get_dct_matrix():
    matrix = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            matrix[j][i] = get_dct_elem(i, j)

    return matrix


def get_dot_product_matrix():
    dct_matrix = get_dct_matrix()
    dot_product_matrix = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            value = np.dot(dct_matrix[i], dct_matrix[j])
            if np.isclose(value, 0):
                dot_product_matrix[i][j] = 0
            else:
                dot_product_matrix[i][j] = value

    return dot_product_matrix


def format_block(block):
    if len(block.shape) < 3:
        return block.astype(float) - 128
    # [0, 255] -> [-128, 127]
    block_centered = block[:, :, 1].astype(float) - 128
    return block_centered


def invert_format_block(block):
    # [-128, 127] -> [0, 255]
    new_block = block + 128
    # in process of dct and inverse dct with quantization,
    # some values can go out of range
    new_block[new_block > 255] = 255
    new_block[new_block < 0] = 0
    return new_block

def dct_cols(block):
    return fftpack.dct(block.T, norm="ortho").T

def dct_rows(block):
    return fftpack.dct(block, norm="ortho")

def dct_1d(row):
    return fftpack.dct(row, norm="ortho")


def idct_1d(row):
    return fftpack.idct(row, norm="ortho")


def dct_2d(block):
    return fftpack.dct(fftpack.dct(block.T, norm="ortho").T, norm="ortho")


def idct_2d(block):
    return fftpack.idct(fftpack.idct(block.T, norm="ortho").T, norm="ortho")


def quantize(block):
    quant_table = get_quantization_table()
    return (block / quant_table).round().astype(np.int32)


def get_quantization_table():
    quant_table = np.array(
        [
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99],
        ]
    )
    return quant_table

def get_chroma_quantization_table():
    return np.array(
        [
            [17,  18,  24,  47,  99,  99,  99,  99],
            [18,  21,  26,  66,  99,  99,  99,  99],
            [24,  26,  56,  99,  99,  99,  99,  99],
            [47,  66,  99,  99,  99,  99,  99,  99],
            [99,  99,  99,  99,  99,  99,  99,  99],
            [99,  99,  99,  99,  99,  99,  99,  99],
            [99,  99,  99,  99,  99,  99,  99,  99],
            [99,  99,  99,  99,  99,  99,  99,  99],
        ]
    )

def get_80_quality_quantization_table():
    return np.array(
        [
            [6,      4,     4,     6,    10,    16,    20,    24],
            [5,      5,     6,     8,    10,    23,    24,    22],
            [6,      5,     6,    10,    16,    23,    28,    22],
            [6,      7,     9,    12,    20,    35,    32,    25],
            [7,      9,    15,    22,    27,    44,    41,    31],
            [10,    14,    22,    26,    32,    42,    45,    37],
            [20,    26,    31,    35,    41,    48,    48,    40],
            [29,    37,    38,    39,    45,    40,    41,    40],

        ]
    )


def dequantize(block):
    quant_table = get_quantization_table()
    return (block * quant_table).astype(np.float)


def rgb2ycbcr(r, g, b):  # in (0,255) range
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 128 - 0.168736 * r - 0.331364 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return y, cb, cr


def rgb2ycbcr4map(c: list):
    y = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
    cb = 128 - 0.168736 * c[0] - 0.331364 * c[1] + 0.5 * c[2]
    cr = 128 + 0.5 * c[0] - 0.418688 * c[1] - 0.081312 * c[2]
    return y, cb, cr


# R = Y + 1.4075 * (V - 128)
# G = Y - 0.3455 * (U - 128) - (0.7169 * (V - 128))
# B = Y + 1.7790 * (U - 128)
def ycbcr2rgb(y, cb, cr):
    r = y + 1.4075 * (cr - 128)
    g = y - 0.34414 * (cb - 128) - 0.71414 * (cr - 128)
    b = y + 1.7790 * (cb - 128)

    r = np.clip(r, 0, 255)
    g = np.clip(g, 0, 255)
    b = np.clip(b, 0, 255)
    return r, g, b


def ycbcr2rgb4map(c: list):
    r = c[0] + 1.4075 * (c[2] - 128)
    g = c[0] - 0.34414 * (c[1] - 128) - 0.71414 * (c[2] - 128)
    b = c[0] + 1.7790 * (c[1] - 128)

    r = np.clip(r, 0, 255)
    g = np.clip(g, 0, 255)
    b = np.clip(b, 0, 255)
    return r, g, b


def g2h(n):
    """Abbreviation for grayscale to hex"""
    return rgb_to_hex((n, n, n))


def coords2rgbcolor(i, j, k):
    """
    Function to transform coordinates in 3D space to hexadecimal RGB color.

    @param: i - x coordinate
    @param: j - y coordinate
    @param: k - z coordinate
    @return: hex value for the corresponding color
    """
    return rgb_to_hex(
        (
            (i) / 255,
            (j) / 255,
            (k) / 255,
        )
    )


def coords2ycbcrcolor(i, j, k):
    """
    Function to transform coordinates in 3D space to hexadecimal YCbCr color.

    @param: i - x coordinate
    @param: j - y coordinate
    @param: k - z coordinate
    @return: hex value for the corresponding color
    """
    y, cb, cr = rgb2ycbcr(i, j, k)

    return rgb_to_hex(
        (
            (y) / 255,
            (cb) / 255,
            (cr) / 255,
        )
    )


def index2coords(n, base):
    """
    Changes the base of `n` to `base`, assuming n is input in base 10.
    The result is then returned as coordinates in `len(result)` dimensions.

    This function allows us to iterate over the color cubes sequentially, using
    enumerate to index every cube, and convert the index of the cube to its corresponding
    coordinate in space.

    Example: if our ``color_res = 4``:

        - Cube #0 is located at (0, 0, 0)
        - Cube #15 is located at (0, 3, 3)
        - Cube #53 is located at (3, 1, 1)

    So, we input our index, and obtain coordinates.

    @param: n - number to be converted
    @param: base - base to change the input
    @return: list - coordinates that the number represent in their corresponding space
    """
    if base == 10:
        return n

    result = 0
    counter = 0

    while n:
        r = n % base
        n //= base
        result += r * 10 ** counter
        counter += 1

    coords = list(f"{result:03}")
    return coords


def get_zigzag_order(block_size=8):
    return zigzag(block_size)


def zigzag(n):
    """zigzag rows"""

    def compare(xy):
        x, y = xy
        return (x + y, -y if (x + y) % 2 else y)

    xs = range(n)
    return {
        n: index
        for n, index in enumerate(sorted(((x, y) for x in xs for y in xs), key=compare))
    }

def two_d_to_1d_index(i, j, block_size=8):
    return j * block_size + i

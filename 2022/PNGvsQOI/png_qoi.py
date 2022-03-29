from math import floor
from manim import *
from manim.mobject.geometry import ArrowTriangleFilledTip
from numpy import ndarray, subtract
from numpy.lib.arraypad import pad
from functions import *
from classes import *
from reducible_colors import *
from itertools import product

np.random.seed(1)

config["assets_dir"] = "assets"

QOI_INDEX = 0
QOI_DIFF_SMALL = 1
QOI_DIFF_MED = 2
QOI_RUN = 3
QOI_RGB = 4

class IntroPNG(ZoomedScene):
    def __init__(self, **kwargs):
        ZoomedScene.__init__(
            self,
            zoom_factor=0.2,
            zoomed_display_height=2,
            zoomed_display_width=2,
            image_frame_stroke_width=10,
            zoomed_camera_config={
                "default_frame_stroke_width": 3,
                },
            **kwargs
        )

    def construct(self):
        bar_chart = ReducibleBarChart(
            [80.7, 74.2, 48.2, 20.7, 3.5],
            max_value=100,
            bar_colors=[REDUCIBLE_YELLOW, REDUCIBLE_VIOLET, REDUCIBLE_GREEN_LIGHTER, REDUCIBLE_PURPLE, REDUCIBLE_GREEN_DARKER],
            bar_names=["PNG", "JPEG", "SVG", "GIF", "WEBP"],
            bar_label_scale_val=0.5,
            height=5,
            width=9,
        )

        title = Text("Image File Format Usage Across Websites", font='CMU Serif', weight=BOLD).scale(0.8)

        title.move_to(UP * 3.5)

        self.play(
            FadeIn(bar_chart),
            FadeIn(title)
        )
        self.wait()

        self.clear()

        problems_with_JPEG = Text("Problems with JPEG", font="CMU Serif", weight=BOLD).scale(1)
        problems_with_JPEG.move_to(UP * 3.5)
        self.play(
            Write(problems_with_JPEG)
        )
        self.wait()

        reducible_jpg = ImageMobject("sample_jpg").move_to(LEFT * 3 + UP * 0.5)
        jpeg_label = Text("JPEG", font="CMU Serif").scale(0.8).next_to(reducible_jpg, DOWN)
        reducible_png = ImageMobject("sample_png").move_to(RIGHT * 3 + UP * 0.5)
        png_label = Text("PNG", font="CMU Serif").scale(0.8).next_to(reducible_png, DOWN)

        self.play(
            FadeIn(reducible_jpg),
            Write(jpeg_label)
        )
        self.wait()

        self.play(
            FadeIn(reducible_png),
            Write(png_label)
        )
        self.wait()

        self.zoom_in_on_images()

        surround_rect = SurroundingRectangle(png_label, buff=SMALL_BUFF)
        lossless_comp = Text("Lossless Compression", font="CMU Serif", weight=BOLD)
        lossless_comp.move_to(DOWN * 3.5).set_color(REDUCIBLE_YELLOW)
        
        self.play(
            Write(surround_rect),
            Write(lossless_comp)
        )
        self.wait()

        self.clear()

        screen_rect = ScreenRectangle(height=6).shift(UP * 0.5)
        self.play(
            Create(screen_rect)
        )
        self.wait()

        mild_spoiler = Text("Mild spoiler: PNG is rather complex and slow").scale(0.8).next_to(screen_rect, DOWN)

        self.play(
            Write(mild_spoiler)
        )
        self.wait()

        simpler_alternative = Text("QOI: an alternative perspective").scale(0.8)
        simpler_alternative.next_to(screen_rect, DOWN)

        self.play(
            ReplacementTransform(mild_spoiler, simpler_alternative)
        )
        self.wait()

    def zoom_in_on_images(self):
        zoomed_camera = self.zoomed_camera
        zoomed_display = self.zoomed_display
        frame = zoomed_camera.frame
        zoomed_display_frame = zoomed_display.display_frame
        zoomed_display.move_to(DOWN * 3)

        frame.move_to(LEFT * 3 + UP * 1.6)
        frame.set_stroke(color=REDUCIBLE_GREEN_LIGHTER, width=5)
        zoomed_display_frame.set_stroke(color=REDUCIBLE_GREEN_LIGHTER)
        # zoomed_display.shift(DOWN)

        zd_rect = BackgroundRectangle(zoomed_display, fill_opacity=0, buff=MED_SMALL_BUFF)
        self.add_foreground_mobject(zd_rect)

        unfold_camera = UpdateFromFunc(zd_rect, lambda rect: rect.replace(zoomed_display))


        self.play(
            Create(frame)
        )
        self.activate_zooming()

        self.play(self.get_zoomed_display_pop_out_animation(), unfold_camera)
        self.wait()
        self.play(
            frame.animate.shift(DOWN * 0.7),
            run_time=2
        )
        self.wait()
        self.play(
            frame.animate.shift(DOWN * 1.4),
            run_time=2
        )

        self.wait()

        self.play(
            frame.animate.shift(DOWN * 0.5 + RIGHT * SMALL_BUFF),
            run_time=2
        )
        self.wait()

        self.play(
            frame.animate.move_to(RIGHT * 3 + UP * 0.9),
            run_time=2
        )
        self.wait()

        self.play(
            frame.animate.shift(DOWN * 1.4),
            run_time=2
        )
        self.wait()

        self.play(
            frame.animate.shift(DOWN * 0.5 + RIGHT * SMALL_BUFF)
        )
        self.wait()

        self.remove(frame, zd_rect, self.zoomed_display, zoomed_display_frame)

class QOIDemo(Scene):

    def construct(self):
        """
        Plan for animation
        1. Show RGB Image pixel representation
        2. Split RGB Image into R, G, and B channels
        3. Show 5 different compression options (keep RGB, run, index, diff small, diff large)
        4. Make a function that takes image and generates QOI tags -- TEST ON DIFFS (seems to work)
        5. Make function to take a list of encoding options (with necessary data) 
        and demo QOI
        In demo, make functional example of a full encoding with list, show the entire bitsteam (can shift info out of the window)
        
        TODO: Tuesday
        show_encode_run - DONE
        show_encode_diff_small - DONE
        show_encode_diff_med - DONE
        show_encode_index - make rectangles in index smaller
        show_encode_rgb - DONE
        
        TODO: Wednesday
        Further complexities we need to deal with to make full animation of QOI
        1. Moving prev and current pixels when current pixel is a RLE
        2. Keeping track of index in a visually pleasing manner
        3. Showing encoding into bytes
        4. Bring bytes and pixels into focus
        5. Dealing with initial prev value

        update_index -- handles index tracking

        get_next_animations(curr_encoding, next_encoding)
        CASES:
        1. curr_encoding - RGB, next_encoding - anything -> use update_prev_and_current
        2. curr_encoding - diff, next_encoding - anything -> use update_prev_and_current
        3. curr_encoding - index, next_encoding - anything -> use unindicate on index and then update_prev_and_current
        4. curr_encoding - RLE, next_encoding - anything -> new logic from Wednesday (1)

        TODO: Thursday
        Make fully functioning QOI animation
        """
        new_pixel_array, pixel_array_mob = self.show_image()
        flattened_pixels = self.show_rgb_split(new_pixel_array, pixel_array_mob)
        self.introduce_qoi_tags(flattened_pixels)
        qoi_data = self.get_qoi_encoding(new_pixel_array)

        self.animate_encoding_of_qoi(flattened_pixels, qoi_data)
        all_qoi_byte_mobs = self.get_all_qoi_data(qoi_data)
        
        self.play(
            flattened_pixels.animate.scale_to_fit_width(config.frame_x_radius * 2 - 1).move_to(UP * 3),
            run_time=3
        )
        arranged_bytes = self.arrange_bytes(all_qoi_byte_mobs)
        arranged_bytes.next_to(flattened_pixels, DOWN)
        self.play(
            FadeIn(arranged_bytes)
        )
        self.wait()

        self.play(
            arranged_bytes.animate.shift(UP),
            FadeOut(flattened_pixels)
        )

        self.wait()

        original_image = Text("Original Image: 192 B", font="CMU Serif").scale(0.6)
        qoi_image = Text("After QOI: 50 B (26%)", font="CMU Serif").scale(0.6)

        original_image.next_to(arranged_bytes, DOWN, aligned_edge=RIGHT).shift(UP * 0.9)
        qoi_image.next_to(original_image, DOWN, aligned_edge=RIGHT)

        self.play(
            Write(original_image)
        )
        self.wait()

        self.play(
            Write(qoi_image)
        )
        self.wait()

        print('Total number of bytes in QOI bitstream:', self.get_total_number_of_bytes(qoi_data))
        print('Total number of bytes in RGB image', 64 * 3)
        print('Compression ratio:', self.get_total_number_of_bytes(qoi_data) / (64 * 3))

    def get_total_number_of_bytes(self, encoded_data):
        total = 0
        for data in encoded_data:
            tag = data[0]
            if tag == QOI_RUN or tag == QOI_DIFF_SMALL or tag == QOI_INDEX:
                total += 1
            elif tag == QOI_DIFF_SMALL:
                total += 2
            else:
                tag += 4
        return total

    def arrange_bytes(self, all_qoi_byte_mobs):
        rows = []
        length = 0
        row = []
        for byte in all_qoi_byte_mobs:
            row.append(byte)
            print('Byte width', byte.width)
            length += byte.width
            if length > 11:
                print('Row size', len(row))
                rows.append(VGroup(*row).arrange(RIGHT, buff=0))
                length = 0
                row = []
        if len(row) > 0:
            rows.append(VGroup(*row).arrange(RIGHT, buff=0))
        entire_group = VGroup(*rows).arrange(DOWN, buff=SMALL_BUFF)
        entire_group[-1].next_to(entire_group[-2], DOWN, aligned_edge=LEFT, buff=SMALL_BUFF)
        return entire_group

    def get_all_qoi_data(self, encoded_data):
        byte_mobs = []
        for data in encoded_data:
            if data[0] == QOI_RUN:
                run_length = data[1]
                byte_mobs.append(self.get_run_bytes_with_data(run_length))
            elif data[0] == QOI_RGB:
                r, g, b = data[1], data[2], data[3]
                byte_mobs.append(self.get_rgb_bytes_with_data(r, g, b))
            elif data[0] == QOI_DIFF_SMALL:
                byte_mobs.append(self.get_small_diff_bytes_with_data(data[1], data[2], data[3]))
            elif data[0] == QOI_DIFF_MED:
                byte_mobs.append(self.get_large_diff_bytes_with_data(data[1], data[2], data[3]))
            else:
                # QOI_INDEX
                byte_mobs.append(self.get_index_bytes_with_data(data[1]))
        
        for mob in byte_mobs:
            byte = mob[1]
            for value in byte:
                if isinstance(value, Byte):
                    value.rect.set_stroke(width=1)
        return VGroup(*byte_mobs).arrange(RIGHT, buff=0).scale(0.4)

    def show_image(self):
        image = ImageMobject("r.png")
        pixel_array = image.get_pixel_array().astype(int)
        pixel_array_mob = PixelArray(pixel_array).scale(0.4).shift(UP * 2)
        self.play(
            FadeIn(pixel_array_mob)
        )
        self.wait()

        new_pixel_array = pixel_array.copy()
        new_pixel_array[1][4] = np.array([new_pixel_array[1][4][0] - 2, new_pixel_array[1][4][1] + 1, new_pixel_array[1][4][2] + 0, new_pixel_array[1][4][3]])
        new_pixel_array[7][0] = np.array([new_pixel_array[6][7][0] + 2, new_pixel_array[6][7][1] - 5, new_pixel_array[6][7][2] - 1, new_pixel_array[6][7][3]])
        new_pixel_array_mob = PixelArray(new_pixel_array).scale(0.4).shift(UP * 2)

        self.play(
            Transform(pixel_array_mob, new_pixel_array_mob)
        )
        self.wait()

        return new_pixel_array, pixel_array_mob
        

    def get_qoi_encoding(self, pixel_array):
        """
        @param: pixel_array - np.array[r, g, b] representing pixels of image
        @return: list[] containing QOI encodings for image
        each QOI encoding is a tuple of 
        """
        INDEX_SIZE = 64
        r_channel = pixel_array[:, :, 0]
        g_channel = pixel_array[:, :, 1]
        b_channel = pixel_array[:, :, 2]

        prev_rgb = [0, 0, 0]

        encodings = []
        indices = [[0, 0, 0]] * INDEX_SIZE
        run = 0
        for row in range(r_channel.shape[0]):
            for col in range(r_channel.shape[1]):
                current_rgb = [r_channel[row][col], g_channel[row][col], b_channel[row][col]]
                if prev_rgb == current_rgb:
                    run += 1
                    if run == 62 or is_last_pixel(r_channel, row, col):
                        encodings.append(
                            (QOI_RUN, run)
                        )
                        run = 0
                else:
                    index_pos = 0
                    if run > 0:
                        encodings.append(
                            (QOI_RUN, run)
                        )
                        run = 0

                    index_pos = qoi_hash(current_rgb) % INDEX_SIZE
                    if indices[index_pos] == current_rgb:
                        encodings.append(
                            (QOI_INDEX, index_pos)
                        )
                    else:
                        indices[index_pos] = current_rgb

                        dr = current_rgb[0] - prev_rgb[0]
                        dg = current_rgb[1] - prev_rgb[1]
                        db = current_rgb[2] - prev_rgb[2]

                        dr_dg = dr - dg
                        db_dg = db - dg

                        if is_diff_small(dr, dg, db):
                            encodings.append(
                                (QOI_DIFF_SMALL, dr, dg, db)
                            )
                        elif is_diff_med(dg, dr_dg, db_dg):
                            encodings.append(
                                (QOI_DIFF_MED, dg, dr_dg, db_dg)
                            )
                        else:
                            encodings.append(
                                (QOI_RGB, current_rgb[0], current_rgb[1], current_rgb[2])
                            )

                prev_rgb = current_rgb
        
        return encodings

    def animate_encoding_of_qoi(self, flattened_pixels, encoded_data):
        """
        Requirements:
        RGB Encoding: 
        1. Just get the RGB data simply and display
        RUN Encoding
        1. Just get the Run data and display
        2. Need to handle properly transitioning from run (update_prev_and_current for run)
        DIFF encodings
        1. Just get the diff data and display
        Index encodings
        1. Need to store position of previously seen pixel in rgb_pixels_array
        2. Handling Unindication of index pixel properly

        General needs:
        Fade Encoded bytes toward left direction
        Handle bringing pixels to screen when out of bounds
        Do a final zoom out of the entire encoding and show savings
        """
        r_channel, g_channel, b_channel = flattened_pixels
        rgb_pixels = self.get_rgb_pixels(r_channel, g_channel, b_channel)

        # iteration 0
        transforms = self.get_indication_transforms([0], rgb_pixels)
        
        self.play(
            *transforms
        )

        fadeout = self.show_encode_rgb_simple(0, rgb_pixels)

        # iteration 0 -> 1
        update_animations = self.update_prev_and_current(-1, 0, rgb_pixels)
        self.play(
            *update_animations + fadeout
        )


        prev_transforms = self.get_indication_transforms([0], rgb_pixels, shift=SMALL_BUFF, direction=np.array([-2.5, 1, 0]), color=REDUCIBLE_VIOLET)
        self.play(
            *prev_transforms
        )

        prev_index = 0
        current_index = 1
        counter = 0
        print(encoded_data)
        index_map = {}
        self.update_index(prev_index, rgb_pixels, index_map)
        for data in encoded_data[1:]:
            return_val = self.bring_pixel_to_screen(current_index, flattened_pixels, target_x_pos=-2.5, animate=False, BUFF=1.5)
            if return_val:
                animations, shift_amount = return_val
                self.play(
                    rgb_pixels[current_index].surrounded.animate.shift(shift_amount),
                    rgb_pixels[prev_index].surrounded.animate.shift(shift_amount),
                    *animations
                )
                # self.wait()
            print(data, 'Curr:', current_index, 'Prev:', prev_index)
            if data[0] == QOI_RUN:
                run_length = data[1]
                fadeout = self.show_encode_run_simple(current_index, run_length, rgb_pixels)
                update_animations = self.update_prev_and_current_after_run(prev_index, current_index, run_length, rgb_pixels)
                self.play(
                    *update_animations + fadeout
                )
                self.bring_to_back(rgb_pixels[prev_index].r)
                self.bring_to_back(rgb_pixels[prev_index].g)
                self.bring_to_back(rgb_pixels[prev_index].b)
                # self.wait()

                current_index += run_length
                prev_index = current_index - 1
            elif data[0] == QOI_RGB:
                fadeout = self.show_encode_rgb_simple(current_index, rgb_pixels)
                update_animations = self.update_prev_and_current(prev_index, current_index, rgb_pixels)
                self.play(
                    *update_animations + fadeout
                )
                self.bring_to_back(rgb_pixels[prev_index].r)
                self.bring_to_back(rgb_pixels[prev_index].g)
                self.bring_to_back(rgb_pixels[prev_index].b)
                # self.wait()
                current_index += 1
                prev_index += 1
            elif data[0] == QOI_DIFF_SMALL:
                fadeout = self.show_encode_diff_small_simple(current_index, rgb_pixels)
                update_animations = self.update_prev_and_current(prev_index, current_index, rgb_pixels)
                self.play(
                    *update_animations + fadeout
                )
                self.bring_to_back(rgb_pixels[prev_index].r)
                self.bring_to_back(rgb_pixels[prev_index].g)
                self.bring_to_back(rgb_pixels[prev_index].b)
                # self.wait()
                current_index += 1
                prev_index += 1
            elif data[0] == QOI_DIFF_MED:
                fadeout = self.show_encode_diff_med_simple(current_index, rgb_pixels)
                update_animations = self.update_prev_and_current(prev_index, current_index, rgb_pixels)
                self.play(
                    *update_animations + fadeout
                )
                self.bring_to_back(rgb_pixels[prev_index].r)
                self.bring_to_back(rgb_pixels[prev_index].g)
                self.bring_to_back(rgb_pixels[prev_index].b)
                # self.wait()
                current_index += 1
                prev_index += 1
            else:
                # QOI_INDEX
                index = data[1]
                fadeout = self.show_encoding_index_simple(current_index, index, index_map, rgb_pixels)
                update_animations = self.update_prev_and_current(prev_index, current_index, rgb_pixels)
                self.bring_to_back(rgb_pixels[prev_index - 1].r)
                self.bring_to_back(rgb_pixels[prev_index - 1].g)
                self.bring_to_back(rgb_pixels[prev_index - 1].b)
                self.play(
                    *update_animations + fadeout
                )
                self.bring_to_back(rgb_pixels[prev_index].r)
                self.bring_to_back(rgb_pixels[prev_index].g)
                self.bring_to_back(rgb_pixels[prev_index].b)
                # if current_index != 63:
                #     self.wait()
                current_index += 1
                prev_index += 1

            self.update_index(prev_index, rgb_pixels, index_map)
            counter += 1

        reset_transforms = self.reset_indications(rgb_pixels)
        self.play(
            *reset_transforms
        )
        self.wait()

    def update_index(self, current_index, rgb_pixels, index_map):
        r, g, b = self.get_rgb_pixel_values(rgb_pixels[current_index])
        index_map[(r, g, b)] = current_index

    def introduce_qoi_tags(self, flattened_pixels):
        r_channel, g_channel, b_channel = flattened_pixels
        rgb_pixels = self.get_rgb_pixels(r_channel, g_channel, b_channel)
        indices = [1]
        self.bring_pixel_to_screen(indices[0], flattened_pixels)
        transforms = self.get_indication_transforms(indices, rgb_pixels)
        
        prev_transforms = self.get_indication_transforms([0], rgb_pixels, shift=SMALL_BUFF, direction=np.array([-2.5, 1, 0]), color=REDUCIBLE_VIOLET)

        self.play(
            *transforms
        )

        self.play(
            *prev_transforms
        )
        self.wait()

        previous = Tex("Previous").scale(0.8).next_to(r_channel[0][0, 0], UP).shift(UP * 1)
        prev_arrow = Arrow(previous.get_bottom(), r_channel[0][0, 0].get_top(), buff=MED_SMALL_BUFF, max_tip_length_to_length_ratio=0.25)
        prev_arrow.set_color(REDUCIBLE_VIOLET)
        
        curr = Tex("Current").scale(0.8).next_to(b_channel[0][0, 1], DOWN).shift(DOWN * 1)
        curr_arrow = Arrow(curr.get_top(),b_channel[0][0, 1].get_bottom(), buff=MED_SMALL_BUFF, max_tip_length_to_length_ratio=0.25)
        curr_arrow.set_color(REDUCIBLE_YELLOW)

        self.play(
            Write(previous),
            Write(prev_arrow),
            Write(curr),
            Write(curr_arrow)
        )
        self.wait()

        self.play(
            FadeOut(previous),
            FadeOut(prev_arrow),
            FadeOut(curr),
            FadeOut(curr_arrow)
        )
        self.wait()

        for i in range(3):
            update_animations = self.update_prev_and_current(i, i + 1, rgb_pixels)
            self.play(
                *update_animations
            )
            self.wait()

        reset_transforms = self.reset_indications(rgb_pixels)
        self.play(
            *reset_transforms
        )
        self.wait()

        transforms = self.get_indication_transforms([2], rgb_pixels)
        
        prev_transforms = self.get_indication_transforms([1], rgb_pixels, shift=SMALL_BUFF, direction=np.array([-2.5, 1, 0]), color=REDUCIBLE_VIOLET)

        self.play(
            *transforms
        )

        self.play(
            *prev_transforms
        )
        self.wait()

        fadeout = self.show_encode_run(2, 4, rgb_pixels)

        reset_transforms = self.reset_indications(rgb_pixels)
        self.play(
            *reset_transforms + fadeout
        )
        self.wait()

        transforms = self.get_indication_transforms([12], rgb_pixels)
        
        prev_transforms = self.get_indication_transforms([11], rgb_pixels, shift=SMALL_BUFF, direction=np.array([-2.5, 1, 0]), color=REDUCIBLE_VIOLET)

        self.play(
            *transforms
        )

        self.play(
            *prev_transforms
        )
        self.wait()

        fadeout = self.show_encode_diff_small(12, rgb_pixels, detail=True)
        reset_transforms = self.reset_indications(rgb_pixels)
        self.play(
            *reset_transforms + fadeout
        )
        self.wait()


        self.bring_pixel_to_screen(56, flattened_pixels)

        transforms = self.get_indication_transforms([56], rgb_pixels)
        
        prev_transforms = self.get_indication_transforms([55], rgb_pixels, shift=SMALL_BUFF, direction=np.array([-2.5, 1, 0]), color=REDUCIBLE_VIOLET)

        self.play(
            *transforms
        )

        self.play(
            *prev_transforms
        )
        self.wait()

        fadeout = self.show_encode_diff_med(56, rgb_pixels, detail=True)

        reset_transforms = self.reset_indications(rgb_pixels)
        self.play(
            *reset_transforms + fadeout
        )
        self.wait()

        self.bring_pixel_to_screen(8, flattened_pixels)

        fadeout = self.show_encoding_index(8, 0, rgb_pixels, detail=True)

        reset_transforms = self.reset_indications(rgb_pixels)

        self.play(
            *reset_transforms + fadeout
        )
        self.wait()
        
        transforms = self.get_indication_transforms([6], rgb_pixels)
        
        prev_transforms = self.get_indication_transforms([5], rgb_pixels, shift=SMALL_BUFF, direction=np.array([-2.5, 1, 0]), color=REDUCIBLE_VIOLET)

        self.play(
            *transforms
        )

        self.play(
            *prev_transforms
        )
        self.wait()

        self

        fadeout = self.show_encode_rgb(6, rgb_pixels)

        reset_transforms = self.reset_indications(rgb_pixels)
        self.play(
            *reset_transforms + fadeout
        )
        self.wait()

    def show_encode_run(self, current_index, run_length, rgb_pixels):
        qoi_run_bytes = self.get_run_bytes().move_to(DOWN * 2.5)
        self.play(
            FadeIn(qoi_run_bytes)
        )
        self.wait()

        for curr_index in range(current_index + 1, current_index + run_length):
            transforms = self.get_indication_transforms([curr_index], rgb_pixels, extend=True)
            self.play(
                *transforms
            )
            self.wait()

        target_surround = get_glowing_surround_rect(qoi_run_bytes[2].text, buff_min=SMALL_BUFF, buff_max=SMALL_BUFF+0.15).move_to(qoi_run_bytes[2].get_center())
        self.play(
            TransformFromCopy(rgb_pixels[current_index].surrounded, target_surround),
        )
        self.wait()
        new_run_text = Text(str(run_length), font='SF Mono', weight=MEDIUM).scale_to_fit_height(qoi_run_bytes[2].text.height).move_to(qoi_run_bytes[2].text.get_center())
        new_run_text.set_color(REDUCIBLE_YELLOW)
        dot = Dot().move_to(target_surround.get_center())
        self.play(
            Flash(dot),
            FadeOut(target_surround),
            Transform(qoi_run_bytes[2].text, new_run_text)
        )
        self.wait()

        return [FadeOut(qoi_run_bytes)]

    def show_encode_run_simple(self, current_index, run_length, rgb_pixels, wait_time=1):
        qoi_run_bytes = self.get_run_bytes_with_data(run_length).move_to(DOWN * 2.5)
        for curr_index in range(current_index + 1, current_index + run_length):
            transforms = self.get_indication_transforms([curr_index], rgb_pixels, extend=True)
            self.play(
                *transforms
            )
            self.wait(wait_time)

        self.play(
            FadeIn(qoi_run_bytes)
        )
        self.wait()
        return [FadeOut(qoi_run_bytes, shift=LEFT)]

    def show_encode_diff_small(self, current_index, rgb_pixels, detail=False):
        qoi_diff_bytes = self.get_small_diff_bytes().move_to(DOWN * 2.5)
        self.play(
            FadeIn(qoi_diff_bytes)
        )
        self.wait()

        curr_r, curr_g, curr_b = self.get_rgb_pixel_values(rgb_pixels[current_index])
        prev_r, prev_g, prev_b = self.get_rgb_pixel_values(rgb_pixels[current_index - 1])

        dr, dg, db = curr_r - prev_r, curr_g - prev_g, curr_b - prev_b

        dr_text = Text(f"dr = {curr_r} - {prev_r}", font='SF Mono', weight=MEDIUM).scale(0.4)
        dg_text = Text(f"dg = {curr_g} - {prev_g}", font='SF Mono', weight=MEDIUM).scale(0.4)
        db_text = Text(f"db = {curr_b} - {prev_b}", font='SF Mono', weight=MEDIUM).scale(0.4)

        text = align_text_vertically(dr_text, dg_text, db_text, aligned_edge=LEFT)

        text.next_to(qoi_diff_bytes, RIGHT, buff=0.5)

        dr_constraint = Text("-2 ≤ dr ≤ 1", font='SF Mono', weight=MEDIUM).scale(0.4)
        dg_constraint = Text("-2 ≤ dg ≤ 1", font='SF Mono', weight=MEDIUM).scale(0.4)
        db_constraint = Text("-2 ≤ db ≤ 1", font='SF Mono', weight=MEDIUM).scale(0.4)

        constraint_text = align_text_vertically(dr_constraint, dg_constraint, db_constraint)
        constraint_text.next_to(qoi_diff_bytes, LEFT, buff=0.5)

        if detail:
            self.play(
                FadeIn(text),
                FadeIn(constraint_text)
            )
            self.wait()

        dr_val = get_matching_text(str(int(dr)), qoi_diff_bytes[2].text)
        dg_val = get_matching_text(str(int(dg)), qoi_diff_bytes[3].text)
        db_val = get_matching_text(str(int(db)), qoi_diff_bytes[4].text)
        # something is off with scaling function so perform slight correction
        dg_val.scale(0.8)
        self.play(
            Transform(qoi_diff_bytes[2].text, dr_val),
            Transform(qoi_diff_bytes[3].text, dg_val),
            Transform(qoi_diff_bytes[4].text, db_val)
        )
        self.wait()
        fadeouts = []
        if detail:
            fadeouts.append(FadeOut(text))
            fadeouts.append(FadeOut(constraint_text))

        fadeouts.append(FadeOut(qoi_diff_bytes))
        return fadeouts

    def show_encode_diff_small_simple(self, current_index, rgb_pixels):
        curr_r, curr_g, curr_b = self.get_rgb_pixel_values(rgb_pixels[current_index])
        prev_r, prev_g, prev_b = self.get_rgb_pixel_values(rgb_pixels[current_index - 1])

        dr, dg, db = curr_r - prev_r, curr_g - prev_g, curr_b - prev_b
        qoi_diff_bytes = self.get_small_diff_bytes_with_data(dr, dg, db).move_to(DOWN * 2.5)
        self.play(
            FadeIn(qoi_diff_bytes)
        )
        self.wait()

        return [FadeOut(qoi_diff_bytes, direction=LEFT)]

    def show_encode_diff_med(self, current_index, rgb_pixels, detail=False):
        qoi_diff_med_bytes = self.get_large_diff_bytes().move_to(DOWN * 2.5)
        self.play(
            FadeIn(qoi_diff_med_bytes)
        )
        self.wait()

        curr_r, curr_g, curr_b = self.get_rgb_pixel_values(rgb_pixels[current_index])
        prev_r, prev_g, prev_b = self.get_rgb_pixel_values(rgb_pixels[current_index - 1])

        dr, dg, db = curr_r - prev_r, curr_g - prev_g, curr_b - prev_b
        dr_dg, db_dg = dr - dg, db - dg
        dg_text = Text(f"dg = {curr_g} - {prev_g}", font='SF Mono', weight=MEDIUM).scale(0.4)
        dr_dg_text = Text(f"dr - dg = ({curr_r} - {prev_r}) - ({curr_g} - {prev_g})", font='SF Mono', weight=MEDIUM).scale(0.4)
        db_dg_text = Text(f"db - dg = ({curr_b} - {prev_b}) - ({curr_g} - {prev_g})", font='SF Mono', weight=MEDIUM).scale(0.4)

        text = align_text_vertically(dg_text, dr_dg_text, db_dg_text, aligned_edge=LEFT)

        text.next_to(qoi_diff_med_bytes, RIGHT, buff=0.5)

        dr_constraint = Text("-32 ≤ dg ≤ 31", font='SF Mono', weight=MEDIUM).scale(0.4)
        dg_constraint = Text("-8 ≤ dr - dg ≤ 7", font='SF Mono', weight=MEDIUM).scale(0.4)
        db_constraint = Text("-8 ≤ db - dg ≤ 7", font='SF Mono', weight=MEDIUM).scale(0.4)

        constraint_text = align_text_vertically(dr_constraint, dg_constraint, db_constraint)
        constraint_text.move_to(UP * 3)

        if detail:
            self.play(
                FadeIn(constraint_text)
            )
            self.wait()

            shift_left = LEFT * 3
            text.shift(shift_left)
            self.play(
                qoi_diff_med_bytes.animate.shift(shift_left),
            )
            self.play(
                FadeIn(text)
            )
            self.wait()

        dg_val = get_matching_text(str(int(dg)), qoi_diff_med_bytes[3].text)
        dr_dg_val = get_matching_text(str(int(dr_dg)), qoi_diff_med_bytes[4].text)
        db_dg_val = get_matching_text(str(int(db_dg)), qoi_diff_med_bytes[5].text)
        # something is off with scaling function so perform slight correction
        # dg_val.scale(0.8)
        self.play(
            Transform(qoi_diff_med_bytes[3].text, dg_val),
            Transform(qoi_diff_med_bytes[4].text, dr_dg_val),
            Transform(qoi_diff_med_bytes[5].text, db_dg_val)
        )
        self.wait()
        fadeouts = []
        if detail:
            fadeouts.append(FadeOut(text))
            fadeouts.append(FadeOut(constraint_text))

        fadeouts.append(FadeOut(qoi_diff_med_bytes))
        return fadeouts

    def show_encode_diff_med_simple(self, current_index, rgb_pixels):
        curr_r, curr_g, curr_b = self.get_rgb_pixel_values(rgb_pixels[current_index])
        prev_r, prev_g, prev_b = self.get_rgb_pixel_values(rgb_pixels[current_index - 1])

        dr, dg, db = curr_r - prev_r, curr_g - prev_g, curr_b - prev_b
        dr_dg, db_dg = dr - dg, db - dg

        qoi_diff_bytes = self.get_large_diff_bytes_with_data(dg, dr_dg, db_dg).move_to(DOWN * 2.5)
        self.play(
            FadeIn(qoi_diff_bytes)
        )
        self.wait()

        return [FadeOut(qoi_diff_bytes, direction=LEFT)]

    def show_encoding_index(self, current_index, index, rgb_pixels, detail=False):
        if detail:
            SIZE = 16
            actual_index = [0] * SIZE
            index_mob = self.get_index(SIZE, 0.4, 12)
            index_mob.move_to(UP * 3.2)
            seen = set()
            self.play(
                FadeIn(index_mob)
            )
            self.wait()
            for i in range(current_index + 1):
                arrow_end = rgb_pixels[i].b[0].get_bottom()
                if i == 0:
                    curr_arrow = Arrow(arrow_end + DOWN * 1.2, arrow_end, buff=MED_SMALL_BUFF, max_tip_length_to_length_ratio=0.25)
                    curr_arrow.set_color(REDUCIBLE_YELLOW)

                    self.play(
                        Write(curr_arrow)
                    )
                else:
                    self.play(
                        Transform(
                            curr_arrow, 
                            Arrow(arrow_end + DOWN * 1.2, arrow_end, buff=MED_SMALL_BUFF, max_tip_length_to_length_ratio=0.25).set_color(REDUCIBLE_YELLOW)
                        )
                    )

                compact_pixels = self.get_compact_rgb_with_text(rgb_pixels[i])
                r, g, b = self.get_rgb_pixel_values(rgb_pixels[i])
                hash_val =  (r * 3 + g * 5 + b * 7)
                index_pos = hash_val % SIZE
                if (r, g, b) not in seen:
                    seen.add((r, g, b))
                    compact_pixels.next_to(index_mob[0][index_pos], DOWN)
                    self.play(
                        TransformFromCopy(rgb_pixels[i].r, compact_pixels[0]),
                        TransformFromCopy(rgb_pixels[i].g, compact_pixels[1]),
                        TransformFromCopy(rgb_pixels[i].b, compact_pixels[2])
                    )
                    actual_index[index_pos] = compact_pixels

            index_text = Text("index = (r * 3 + g * 5 + b * 7) % size", font='SF Mono', weight=MEDIUM)
            index_text.scale(0.5)
            index_text.move_to(DOWN * 2.5)
            self.play(
                FadeIn(index_text)
            )
            self.wait()

            index_detail = Text("index = (140 * 3 + 77 * 5 + 251 * 7) % 16", font='SF Mono', weight=MEDIUM).scale(0.5)
            index_detail.next_to(index_text, DOWN, aligned_edge=LEFT)

            self.play(
                TransformFromCopy(index_text, index_detail)
            )
            self.wait()

            index_answer = Text("index = 2", font='SF Mono', weight=MEDIUM).scale(0.5).next_to(index_detail, DOWN, aligned_edge=LEFT)
            self.play(
                TransformFromCopy(index_detail, index_answer)
            )
            self.wait()

            indication = self.get_indication_transforms([current_index], rgb_pixels)

            self.play(
                *[FadeOut(curr_arrow)] + indication,
            )
            self.wait()

            glowing_rect = get_glowing_surround_rect(actual_index[2], color=REDUCIBLE_GREEN_LIGHTER)

            indication_index = self.get_indication_transforms([0],  rgb_pixels, color=REDUCIBLE_GREEN_LIGHTER)

            self.play(
                *indication_index + [FadeIn(glowing_rect)]
            )
            self.wait()

            self.play(
                FadeOut(index_text),
                FadeOut(index_detail),
                FadeOut(index_answer)
            )
            self.wait()

        qoi_index_bytes = self.get_index_bytes().move_to(DOWN * 2.3)
        self.play(
            FadeIn(qoi_index_bytes)
        )
        self.wait()

        r, g, b = self.get_rgb_pixel_values(rgb_pixels[index])
        hash_val =  (r * 3 + g * 5 + b * 7)
        index_pos = hash_val % SIZE

        index_val = get_matching_text(str(index_pos), qoi_index_bytes[2].text)
        
        self.play(
            Transform(qoi_index_bytes[2].text, index_val),
        )
        self.wait()

        fadeouts = []
        if detail:
            note = Text("Note: QOI uses an index of size 64 (6 bits)", font='SF Mono', weight=MEDIUM).scale(0.4)
            note.next_to(qoi_index_bytes, DOWN)
            self.add(note)
            self.wait()
            fadeouts.extend([FadeOut(glowing_rect), FadeOut(note), FadeOut(index_mob)] + [FadeOut(mob) for mob in actual_index if mob != 0])

        return fadeouts + [FadeOut(qoi_index_bytes)]

    def show_encoding_index_simple(self, current_index, index, index_map, rgb_pixels):
        r, g, b = self.get_rgb_pixel_values(rgb_pixels[current_index])
        seen_index = index_map[(r, g, b)]
        if seen_index == current_index - 2:
            indication_index = self.get_indication_transforms([seen_index],  rgb_pixels, shift=SMALL_BUFF, direction=np.array([-5, 1, 0]), color=REDUCIBLE_GREEN_LIGHTER)
        else:
            indication_index = self.get_indication_transforms([seen_index],  rgb_pixels, color=REDUCIBLE_GREEN_LIGHTER)

        qoi_index_bytes = self.get_index_bytes_with_data(index).move_to(DOWN * 2.5)

        self.play(
            *indication_index,
            FadeIn(qoi_index_bytes)
        )
        self.wait()
        surrounded_rects = rgb_pixels[seen_index].surrounded
        reset_transforms = self.unindicate_pixels(rgb_pixels[seen_index])
        rgb_pixels[seen_index].surrounded = None

        return [FadeOut(qoi_index_bytes, shift=LEFT), FadeOut(surrounded_rects)] + reset_transforms

    def get_compact_rgb_with_text(self, rgb_pixel, height=0.2, width=0.5):
        r_color = rgb_pixel.r[0].get_color()
        g_color = rgb_pixel.g[0].get_color()
        b_color = rgb_pixel.b[0].get_color()

        r_rect = Rectangle(height=height, width=width).set_color(r_color).set_fill(color=r_color, opacity=1)
        g_rect = Rectangle(height=height, width=width).set_color(g_color).set_fill(color=g_color, opacity=1)
        b_rect = Rectangle(height=height, width=width).set_color(b_color).set_fill(color=b_color, opacity=1)
        
        compact_rects = [r_rect, g_rect, b_rect]
        VGroup(*compact_rects).arrange(DOWN, buff=SMALL_BUFF/3)

        r, g, b = self.get_rgb_pixel_values(rgb_pixel)
        
        g_color = BLACK if g > 200 else WHITE
        text_in_array = align_text_vertically(
            Text(str(r), font='SF Mono', weight=MEDIUM).scale(0.2),
            Text(str(g), font='SF Mono', weight=MEDIUM).scale(0.2).set_color(g_color),
            Text(str(b), font='SF Mono', weight=MEDIUM).scale(0.2),
            buff=SMALL_BUFF
        )

        compact_pixels = []
        for rect, text in zip(compact_rects, text_in_array):
            text.scale(1).move_to(rect.get_center())
            compact_pixels.append(VGroup(rect, text))

        return VGroup(*compact_pixels)

    def get_index(self, size, height, width, index_scale=0.3, color=REDUCIBLE_YELLOW):
        rect_width = width / size
        index_mob = VGroup(*[Rectangle(height=height, width=rect_width) for _ in range(size)])
        indices = [Text(str(int(i)), font='SF Mono', weight=MEDIUM).scale(index_scale) for i in range(size)]
        index_mob.arrange(RIGHT, buff=0)
        for i, text in enumerate(indices):
            text.move_to(index_mob[i].get_center())
        index_mob.set_color(REDUCIBLE_YELLOW)
        indices_text = VGroup(*indices)
        return VGroup(index_mob, indices_text)

    def show_encode_rgb(self, current_index, rgb_pixels):
        qoi_rgb_bytes = self.get_rbg_bytes().move_to(DOWN * 2.5)

        self.play(
            FadeIn(qoi_rgb_bytes)
        )
        self.wait()
        r_val, g_val, b_val = self.get_rgb_pixel_values(rgb_pixels[current_index])

        r = get_matching_text(str(int(r_val)), qoi_rgb_bytes[5].text)
        g = get_matching_text(str(int(g_val)), qoi_rgb_bytes[6].text)
        b = get_matching_text(str(int(b_val)), qoi_rgb_bytes[7].text)

        self.play(
            Transform(qoi_rgb_bytes[5].text, r),
            Transform(qoi_rgb_bytes[6].text, g),
            Transform(qoi_rgb_bytes[7].text, b)
        )
        self.wait()

        return [FadeOut(qoi_rgb_bytes)]

    def show_encode_rgb_simple(self, current_index, rgb_pixels):
        r_val, g_val, b_val = self.get_rgb_pixel_values(rgb_pixels[current_index])
        qoi_rgb_bytes = self.get_rgb_bytes_with_data(r_val, g_val, b_val).move_to(DOWN * 2.5)

        self.play(
            FadeIn(qoi_rgb_bytes)
        )
        self.wait()

        return [FadeOut(qoi_rgb_bytes, shift=LEFT)]

    def get_rgb_pixel_values(self, rgb_pixel):
        """
        @param: rgb_pixel - RGB mob
        """
        r_val = int(rgb_pixel.r[1].original_text)
        g_val = int(rgb_pixel.g[1].original_text)
        b_val = int(rgb_pixel.b[1].original_text)
        return [r_val, g_val, b_val]


    def get_indication_transforms(self, indices, rgb_pixels, 
        opacity=0.2, extend=False, shift=SMALL_BUFF, direction=UP, color=REDUCIBLE_YELLOW):
        indication_transforms = []
        all_other_indices = [index for index in range(len(rgb_pixels)) if index not in indices]
        for index in all_other_indices:
            animations = []
            pixel = rgb_pixels[index]
            if pixel.indicated:
                continue
            faded_pixels = self.get_faded_pixels(pixel, opacity=opacity)
            animations.extend([Transform(pixel.r, faded_pixels[0]), Transform(pixel.g, faded_pixels[1]), Transform(pixel.b, faded_pixels[2])])
            indication_transforms.extend(animations)
        
        animations = []
        if extend:
            last_pixel_index = indices[0] - 1
            while rgb_pixels[last_pixel_index].surrounded is None:
                last_pixel_index -= 1
            original_rect = rgb_pixels[last_pixel_index].surrounded
            indicated_pixels = self.get_indicated_pixels([rgb_pixels[index] for index in range(last_pixel_index, indices[-1] + 1)], shift=shift, direction=direction)
            surrounded_rects = self.get_surrounded_rects(indicated_pixels, color=color)
            animations.append(Transform(original_rect, VGroup(*surrounded_rects)))

        pixels = [rgb_pixels[index] for index in indices]
        indicated_pixels = self.get_indicated_pixels(pixels, shift=shift, direction=direction)
        for pixel in pixels:
            pixel.indicated = True
        surrounded_rects = self.get_surrounded_rects(indicated_pixels, color=color)
        if not extend:
            pixels[0].surrounded = VGroup(*surrounded_rects)
        animations.extend(self.get_scale_transforms(pixels, indicated_pixels))
        if not extend:
            animations.extend(
                [
                FadeIn(surrounded_rects[0]), FadeIn(surrounded_rects[1]), FadeIn(surrounded_rects[2])
                ]
            )
        indication_transforms.extend(animations)

        return indication_transforms

    def update_prev_and_current(self, prev_index, current_index, rgb_pixels):
        """
        @param: prev_index - previous index of QOI encoding
        @param: current_index - current index of QOI encoding
        @param: rgb_pixels - list[RGB] of RGB pixels
        @return: list[Animation] of all indication, resetting, and 
        scaling animations of updating prev_index to current_index 
        and current_index to current_index + 1
        """
        if current_index + 1 == len(rgb_pixels):
            return []
        current_direction_shift = LEFT * 2.5 * SMALL_BUFF
        current_direction_scale = 1
        prev_direction_shift = RIGHT * 2.5 * SMALL_BUFF
        prev_direction_scale = 1 / 1.2
        next_direction_shift = UP * SMALL_BUFF
        next_direction_scale = 1.2

        prev_pixel = rgb_pixels[prev_index]
        current_pixel = rgb_pixels[current_index]
        next_pixel = rgb_pixels[current_index + 1]

        if prev_index == -1:
            animations = []
            indicate_next, next_pixels = self.indicate_next_pixel(rgb_pixels[current_index + 1])
            animations.append(ApplyMethod(current_pixel.surrounded.move_to, VGroup(*next_pixels).get_center()))
            current_pixel.surrounded, next_pixel.surrounded = None, current_pixel.surrounded
            unindicate_prev = self.unindicate_pixels(current_pixel)
            animations.extend(indicate_next + unindicate_prev)
            return animations
        
        animations = []
        unindicate_prev = self.unindicate_pixels(prev_pixel)
        indicate_next, next_pixels = self.indicate_next_pixel(rgb_pixels[current_index + 1])
        transform_curr_to_prev, new_prev_pixels = self.current_to_prev(current_pixel, current_direction_shift)
        animations.extend(unindicate_prev + indicate_next + transform_curr_to_prev)
        animations.append(ApplyMethod(prev_pixel.surrounded.move_to, VGroup(*new_prev_pixels).get_center()))
        animations.append(ApplyMethod(current_pixel.surrounded.move_to, VGroup(*next_pixels).get_center()))
        prev_pixel.surrounded, current_pixel.surrounded, next_pixel.surrounded = None, prev_pixel.surrounded, current_pixel.surrounded
        return animations

    def update_prev_and_current_after_run(self, prev_index, current_index, run_length, rgb_pixels):
        current_direction_shift = LEFT * 2.5 * SMALL_BUFF
        current_direction_scale = 1
        prev_direction_shift = RIGHT * 2.5 * SMALL_BUFF
        prev_direction_scale = 1 / 1.2
        next_direction_shift = UP * SMALL_BUFF
        next_direction_scale = 1.2

        prev_pixel = rgb_pixels[prev_index]
        current_pixel = rgb_pixels[current_index]
        
        new_prev_pixel = rgb_pixels[prev_index + run_length]
        new_current_pixel = rgb_pixels[current_index + run_length]
        animations = []
        
        unindicate_prev = []
        for i in range(prev_index, prev_index + run_length):
            unindicate_prev.extend(self.unindicate_pixels(rgb_pixels[i]))
            
        indicate_next, next_pixels = self.indicate_next_pixel(new_current_pixel)
        transform_curr_to_prev, new_prev_pixels = self.current_to_prev(new_prev_pixel, current_direction_shift)
        animations.extend(unindicate_prev + indicate_next + transform_curr_to_prev)
        animations.append(ApplyMethod(prev_pixel.surrounded.move_to, VGroup(*new_prev_pixels).get_center()))

        new_surround_rect = VGroup(*self.get_surrounded_rects(next_pixels))
        animations.append(ApplyMethod(current_pixel.surrounded.become, new_surround_rect))
        if run_length == 1:
            # new_prev_pixel and current_pixel are the same reference
            prev_pixel.surrounded, current_pixel.surrounded, new_current_pixel.surrounded = None, prev_pixel.surrounded, current_pixel.surrounded
        else:
            prev_pixel.surrounded, current_pixel.surrounded, new_prev_pixel.surrounded, new_current_pixel.surrounded = None, None, prev_pixel.surrounded, current_pixel.surrounded
        return animations

    def current_to_prev(self, rgb_pixel, shift):
        rgb_pixel.shift = np.array([shift[0], rgb_pixel.shift[1], 0])
        animations = []
        new_pixel = [
        self.current_to_prev_channel(rgb_pixel.r, shift),
        self.current_to_prev_channel(rgb_pixel.g, shift),
        self.current_to_prev_channel(rgb_pixel.b, shift)
        ]
        animations.append(Transform(rgb_pixel.r, new_pixel[0]))
        animations.append(Transform(rgb_pixel.g, new_pixel[1]))
        animations.append(Transform(rgb_pixel.b, new_pixel[2]))
        return animations, new_pixel

    def current_to_prev_channel(self, channel, shift):
        return channel.copy().shift(shift)

    def unindicate_pixels(self, rgb_pixel):
        animations = []
        if rgb_pixel.indicated:
            animations.append(Transform(rgb_pixel.r, self.unindicate_pixel(rgb_pixel, rgb_pixel.r)))
            animations.append(Transform(rgb_pixel.g, self.unindicate_pixel(rgb_pixel, rgb_pixel.g)))
            animations.append(Transform(rgb_pixel.b, self.unindicate_pixel(rgb_pixel, rgb_pixel.b)))
            rgb_pixel.indicated = False
            rgb_pixel.scaled = 1
            rgb_pixel.shift = ORIGIN
        return animations

    def indicate_next_pixel(self, next_pixel):
        animations = []
        indicated_pixel = self.get_indicated_pixels([next_pixel])
        next_pixels = [indicated_pixel[0][0], indicated_pixel[1][0], indicated_pixel[2][0]]
        if not next_pixel.indicated:
            animations.append(Transform(next_pixel.r, next_pixels[0]))
            animations.append(Transform(next_pixel.g, next_pixels[1]))
            animations.append(Transform(next_pixel.b, next_pixels[2]))
            next_pixel.indicated = True
        return animations, next_pixels

    def unindicate_pixel(self, original_pixel, channel, opacity=0.2):
        pixel = channel.copy()
        pixel.scale(1/original_pixel.scaled).shift(-original_pixel.shift)
        pixel[0].set_fill(opacity=opacity).set_stroke(opacity=opacity)
        pixel[1].set_fill(opacity=opacity)
        return pixel

    def get_scale_transforms(self, pixels, indicated_pixels):
        transforms = []
        for i, pixel in enumerate(pixels):
            transforms.append(Transform(pixel.r, indicated_pixels[0][i]))
            transforms.append(Transform(pixel.g, indicated_pixels[1][i]))
            transforms.append(Transform(pixel.b, indicated_pixels[2][i]))

        return transforms

    def get_faded_pixels(self, pixel, opacity=0.2):
        r_pixel = self.get_faded_pixel(pixel.r, opacity=opacity)
        g_pixel = self.get_faded_pixel(pixel.g, opacity=opacity)
        b_pixel = self.get_faded_pixel(pixel.b, opacity=opacity)
        return [r_pixel, g_pixel, b_pixel]

    def get_indicated_pixels(self, pixels, scale=1.2, shift=SMALL_BUFF, direction=UP, reset=False):
        r_pixel = VGroup(*[self.get_indicated_pixel(pixel, pixel.r, scale=scale, shift=shift, direction=direction, reset=reset) for pixel in pixels])
        g_pixel = VGroup(*[self.get_indicated_pixel(pixel, pixel.g, scale=scale, shift=shift, direction=direction, reset=reset) for pixel in pixels])
        b_pixel = VGroup(*[self.get_indicated_pixel(pixel, pixel.b, scale=scale, shift=shift, direction=direction, reset=reset) for pixel in pixels])
        return [r_pixel, g_pixel, b_pixel]

    def get_surrounded_rects(self, indicated_pixels, color=REDUCIBLE_YELLOW):
        return [get_glowing_surround_rect(pixel, color=color) for pixel in indicated_pixels]

    def get_indicated_pixel(self, original_pixel, channel, scale=1.2, shift=SMALL_BUFF, direction=UP, indicated=False, reset=False):
        pixel = channel.copy()
        pixel = self.get_faded_pixel(pixel, opacity=1)
        if not original_pixel.indicated:
            original_pixel.scaled = scale
            original_pixel.shift = direction * shift
            pixel.scale(scale).shift(direction * shift)
        elif reset:
            pixel.scale(1/original_pixel.scaled).shift(-original_pixel.shift)

        return pixel

    def get_faded_pixel(self, channel, opacity=0.2):
        pixel = channel.copy()
        pixel[0].set_fill(opacity=opacity).set_stroke(opacity=opacity)
        pixel[1].set_fill(opacity=opacity)
        return pixel

    def get_rgb_pixels(self, r_channel, g_channel, b_channel):
        pixels = []
        for i in range(len(r_channel[1])):
            r_mob = VGroup(r_channel[0][0, i], r_channel[1][i])
            g_mob = VGroup(g_channel[0][0, i], g_channel[1][i])
            b_mob = VGroup(b_channel[0][0, i], b_channel[1][i])
            pixels.append(RGBMob(r_mob, g_mob, b_mob))
        return pixels

    def reset_indications(self, rgb_pixels):
        animations = []
        for i, pixel in enumerate(rgb_pixels):
            if pixel.indicated:
                if pixel.surrounded:
                    animations.append(FadeOut(pixel.surrounded))
                    pixel.surrounded = None
                original_pixel = self.get_indicated_pixels([pixel], reset=True)
                animations.append(Transform(pixel.r, original_pixel[0][0]))
                animations.append(Transform(pixel.g, original_pixel[1][0]))
                animations.append(Transform(pixel.b, original_pixel[2][0]))
                pixel.indicated = False
                pixel.scale = 1
                pixel.shift = ORIGIN
            else:
                original_pixel = self.get_faded_pixels(pixel, opacity=1)
                animations.append(Transform(pixel.r, original_pixel[0]))
                animations.append(Transform(pixel.g, original_pixel[1]))
                animations.append(Transform(pixel.b, original_pixel[2]))

        return animations

    def bring_pixel_to_screen(self, index, flattened_pixels, target_x_pos=0, BUFF=1, animate=True):
        r_channel = flattened_pixels[0]
        target_pixel = r_channel[0][index]
        shift_amount = self.get_shift_amount(target_pixel.get_center(), target_x_pos=target_x_pos, BUFF=BUFF)
        if not np.array_equal(shift_amount, ORIGIN):
            if animate:
                self.play(
                    flattened_pixels.animate.shift(shift_amount)
                )
                self.wait()
            else:
                self.bring_to_back(flattened_pixels[0][0][index - 2])
                self.bring_to_back(flattened_pixels[1][0][index - 2])
                self.bring_to_back(flattened_pixels[2][0][index - 2])
                return [ApplyMethod(flattened_pixels.shift, shift_amount)], shift_amount


    def get_shift_amount(self, position, target_x_pos=0, BUFF=1):
        x_pos = position[0]
        if x_pos > -config.frame_x_radius + BUFF and x_pos < config.frame_x_radius - BUFF:
            # NO Shift needed
            return ORIGIN

        return (target_x_pos - x_pos) * RIGHT

    def show_rgb_split(self, pixel_array, pixel_array_mob):
        r_channel = pixel_array[:, :, 0]
        g_channel = pixel_array[:, :, 1]
        b_channel = pixel_array[:, :, 2]

        r_channel_padded = self.get_channel_image(r_channel)
        g_channel_padded = self.get_channel_image(g_channel, mode='G')
        b_channel_padded = self.get_channel_image(b_channel, mode='B')

        pixel_array_mob_r = PixelArray(r_channel_padded).scale(0.4).shift(LEFT * 4 + DOWN * 1.5)
        pixel_array_mob_g = PixelArray(g_channel_padded).scale(0.4).shift(DOWN * 1.5)
        pixel_array_mob_b = PixelArray(b_channel_padded).scale(0.4).shift(RIGHT * 4 + DOWN * 1.5)

        self.play(
            TransformFromCopy(pixel_array_mob, pixel_array_mob_r),
            TransformFromCopy(pixel_array_mob, pixel_array_mob_b),
            TransformFromCopy(pixel_array_mob, pixel_array_mob_g)
        )
        self.wait()

        r_channel_pixel_text = self.get_pixel_values(r_channel, pixel_array_mob_r)
        g_channel_pixel_text = self.get_pixel_values(g_channel, pixel_array_mob_g, mode='G')
        b_channel_pixel_text = self.get_pixel_values(b_channel, pixel_array_mob_b, mode='B')
        self.play(
            FadeIn(r_channel_pixel_text),
            FadeIn(g_channel_pixel_text),
            FadeIn(b_channel_pixel_text)
        )
        self.wait()

        self.play(
            FadeOut(pixel_array_mob),
            pixel_array_mob_r.animate.shift(UP * 3),
            pixel_array_mob_b.animate.shift(UP * 3),
            pixel_array_mob_g.animate.shift(UP * 3),
            r_channel_pixel_text.animate.shift(UP * 3),
            g_channel_pixel_text.animate.shift(UP * 3),
            b_channel_pixel_text.animate.shift(UP * 3),
        )
        self.wait()

        r_channel_flattened = self.reshape_channel(r_channel_padded)
        g_channel_flattened = self.reshape_channel(g_channel_padded)
        b_channel_flattened = self.reshape_channel(b_channel_padded)

        r_channel_f_mob = PixelArray(r_channel_flattened, buff=MED_SMALL_BUFF, outline=False).scale(0.6).to_edge(LEFT)
        g_channel_f_mob = PixelArray(g_channel_flattened, buff=MED_SMALL_BUFF, outline=False).scale(0.6).to_edge(LEFT)
        b_channel_f_mob = PixelArray(b_channel_flattened, buff=MED_SMALL_BUFF, outline=False).scale(0.6).to_edge(LEFT)

        r_channel_f_mob.to_edge(LEFT * 3).shift(DOWN * 1.1)
        g_channel_f_mob.next_to(r_channel_f_mob, DOWN * 2, aligned_edge=LEFT)
        b_channel_f_mob.next_to(g_channel_f_mob, DOWN * 2, aligned_edge=LEFT)
        
        r_channel_f_mob_text = self.get_pixel_values(r_channel_flattened[:, :, 0], r_channel_f_mob, mode='R')
        g_channel_f_mob_text = self.get_pixel_values(g_channel_flattened[:, :, 1], g_channel_f_mob, mode='G')
        b_channel_f_mob_text = self.get_pixel_values(b_channel_flattened[:, :, 2], b_channel_f_mob, mode='B')


        r_transforms = self.get_flatten_transform(pixel_array_mob_r, r_channel_f_mob, r_channel_pixel_text, r_channel_f_mob_text)

        g_transforms = self.get_flatten_transform(pixel_array_mob_g, g_channel_f_mob, g_channel_pixel_text, g_channel_f_mob_text)

        b_transforms = self.get_flatten_transform(pixel_array_mob_b, b_channel_f_mob, b_channel_pixel_text, b_channel_f_mob_text)
        
        self.play(
            *r_transforms,
            run_time=3
        )
        self.wait()

        self.play(
            *g_transforms,
            run_time=3
        )
        self.wait()

        self.play(
            *b_transforms,
            run_time=3
        )
        self.wait()

        self.play(
            FadeOut(pixel_array_mob_r),
            FadeOut(pixel_array_mob_g),
            FadeOut(pixel_array_mob_b),
            FadeOut(r_channel_pixel_text),
            FadeOut(g_channel_pixel_text),
            FadeOut(b_channel_pixel_text),
            r_channel_f_mob.animate.shift(UP * 2.5),
            g_channel_f_mob.animate.shift(UP * 2.5),
            b_channel_f_mob.animate.shift(UP * 2.5),
            r_channel_f_mob_text.animate.shift(UP * 2.5),
            g_channel_f_mob_text.animate.shift(UP * 2.5),
            b_channel_f_mob_text.animate.shift(UP * 2.5),
        )
        self.wait()

        r_channel = VGroup(r_channel_f_mob, r_channel_f_mob_text)
        g_channel = VGroup(g_channel_f_mob, g_channel_f_mob_text)
        b_channel = VGroup(b_channel_f_mob, b_channel_f_mob_text)

        return VGroup(r_channel, g_channel, b_channel)

    def get_rbg_bytes(self):
        rgb_tag_byte = Byte(
            ["Byte[0]",
            "7,6,5,4,3,2,1,0"],
            stroke_color=REDUCIBLE_PURPLE
        ).scale(0.5).move_to(DOWN * 2)

        red_first_byte = Byte(
            ["Byte[1]",
            "7,...,0"],
            width=2,
            height=rgb_tag_byte.height,
            stroke_color=REDUCIBLE_PURPLE
        )

        red_first_byte.text.scale_to_fit_height(rgb_tag_byte.text.height)

        red_first_byte.next_to(rgb_tag_byte, RIGHT, buff=0)

        green_second_byte = Byte(
            ["Byte[2]",
            "7,...,0"],
            width=red_first_byte.width,
            height=rgb_tag_byte.height,
            stroke_color=REDUCIBLE_PURPLE
        )

        green_second_byte.text.scale_to_fit_height(red_first_byte.text.height)

        green_second_byte.next_to(red_first_byte, RIGHT, buff=0)

        blue_third_byte = Byte(
            ["Byte[3]",
            "7,...,0"],
            width=red_first_byte.width,
            height=rgb_tag_byte.height,
            stroke_color=REDUCIBLE_PURPLE
        )

        blue_third_byte.text.scale_to_fit_height(red_first_byte.text.height)

        blue_third_byte.next_to(green_second_byte, RIGHT, buff=0)

        tag_value = Byte(
            "8 bit RGB_TAG",
            width=rgb_tag_byte.width,
            height=rgb_tag_byte.height,
            text_scale=0.25,
            stroke_color=REDUCIBLE_PURPLE
        )
        
        tag_value.next_to(rgb_tag_byte, DOWN, buff=0)

        red_value = Byte(
            "Red",
            width=red_first_byte.width,
            height=red_first_byte.height,
            text_scale=0.25,
            stroke_color=REDUCIBLE_PURPLE
        ).next_to(red_first_byte, DOWN, buff=0)
        red_value.text.scale_to_fit_height(tag_value.text.height)

        green_value = Byte(
            "Green",
            width=green_second_byte.width,
            height=green_second_byte.height,
            text_scale=0.25,
            stroke_color=REDUCIBLE_PURPLE
        ).next_to(green_second_byte, DOWN, buff=0)
        green_value.text.scale_to_fit_height(tag_value.text.height)

        blue_value = Byte(
            "Blue",
            width=blue_third_byte.width,
            height=blue_third_byte.height,
            text_scale=0.25,
            stroke_color=REDUCIBLE_PURPLE
        ).next_to(blue_third_byte, DOWN, buff=0)
        blue_value.text.scale_to_fit_height(tag_value.text.height)

        qoi_rgb_bytes = VGroup(
            rgb_tag_byte, red_first_byte, green_second_byte, blue_third_byte,
            tag_value, red_value, green_value, blue_value,
        ).move_to(ORIGIN)

        return qoi_rgb_bytes

    def get_rgb_bytes_with_data(self, r, g, b, label=True):
        rgb_byte = self.get_rbg_bytes()
        rgb_byte[-3].text.become(get_matching_text(str(r), rgb_byte[-3].text))
        rgb_byte[-2].text.become(get_matching_text(str(g), rgb_byte[-2].text))
        rgb_byte[-1].text.become(get_matching_text(str(b), rgb_byte[-1].text))
        if label:
            label = Text("QOI_RGB", font="SF Mono", weight=MEDIUM).scale(0.4)
            label.next_to(rgb_byte, UP)
            return VGroup(label, rgb_byte)
        return rgb_byte

    def get_index_bytes(self):
        index_tag_byte = Byte(
            ["Byte[0]",
            "7,6,5,4,3,2,1,0"],
            stroke_color=REDUCIBLE_YELLOW
        ).scale(0.5).move_to(DOWN * 2)

        target_text = VGroup(
            index_tag_byte.text[1][1],
            index_tag_byte.text[1][2]
        )
        tag_value = Byte(
            "0,0",
            width=target_text.get_center()[0] + SMALL_BUFF - index_tag_byte.get_left()[0],
            height=index_tag_byte.height,
            stroke_color=REDUCIBLE_YELLOW
        )
        tag_value.text.scale_to_fit_height(index_tag_byte.text[1][1].height)

        tag_value.next_to(index_tag_byte, DOWN, aligned_edge=LEFT, buff=0)

        index_value = Byte(
            "index",
            width=index_tag_byte.get_right()[0] - tag_value.get_right()[0],
            height=tag_value.height,
            stroke_color=REDUCIBLE_YELLOW
        )

        index_value.text.scale_to_fit_height(tag_value.text.height).scale(1.3)

        index_value.next_to(index_tag_byte, DOWN, aligned_edge=RIGHT, buff=0)

        qoi_index_bytes = VGroup(
            index_tag_byte,
            tag_value, index_value
        ).move_to(ORIGIN)

        return qoi_index_bytes

    def get_index_bytes_with_data(self, index, label=True):
        index_byte = self.get_index_bytes()
        index_byte[-1].text.become(get_matching_text(str(index), index_byte[-1].text))
        if label:
            label = Text("QOI_INDEX", font="SF Mono", weight=MEDIUM).scale(0.4)
            label.next_to(index_byte, UP)
            return VGroup(label, index_byte)
        return index_byte

    def get_run_bytes(self):
        run_tag_byte = Byte(
            ["Byte[0]",
            "7,6,5,4,3,2,1,0"]
        ).scale(0.5).move_to(DOWN * 2)

        target_text = VGroup(
            run_tag_byte.text[1][1],
            run_tag_byte.text[1][2]
        )
        tag_value = Byte(
            "1,1",
            text_scale=1,
            width=target_text.get_center()[0] + SMALL_BUFF - run_tag_byte.get_left()[0],
            height=run_tag_byte.height
        )
        tag_value.text.scale_to_fit_height(run_tag_byte.text[1][1].height)

        tag_value.next_to(run_tag_byte, DOWN, aligned_edge=LEFT, buff=0)
        tag_value.text.rotate(PI) # Not sure why this has to be done? Some issue with VGroup arrangement
        tag_value.text[0].shift(LEFT * SMALL_BUFF * 0.5)
        tag_value.text[1].shift(RIGHT * SMALL_BUFF * 0.5)

        run_value = Byte(
            "run",
            width=run_tag_byte.get_right()[0] - tag_value.get_right()[0],
            height=tag_value.height
        )

        run_value.text.scale_to_fit_height(tag_value.text.height).scale(1.1)


        run_value.next_to(run_tag_byte, DOWN, aligned_edge=RIGHT, buff=0)

        qoi_index_bytes = VGroup(
            run_tag_byte,
            tag_value, run_value
        ).move_to(ORIGIN)

        return qoi_index_bytes

    def get_run_bytes_with_data(self, run_length, label=True):
        run_byte = self.get_run_bytes()
        run_byte[-1].text.become(get_matching_text(str(run_length), run_byte[-1].text))
        if label:
            label = Text("QOI_RUN", font="SF Mono", weight=MEDIUM).scale(0.4)
            label.next_to(run_byte, UP)
            return VGroup(label, run_byte)
        return run_byte

    def get_large_diff_bytes(self):
        diff_tag_byte = Byte(
            ["Byte[0]",
            "7,6,5,4,3,2,1,0"],
            stroke_color=REDUCIBLE_GREEN_DARKER
        ).scale(0.5).move_to(DOWN * 2)

        target_text = VGroup(
            diff_tag_byte.text[1][1],
            diff_tag_byte.text[1][2]
        )
        tag_value = Byte(
            "1,0",
            width=target_text.get_center()[0] + SMALL_BUFF - diff_tag_byte.get_left()[0],
            height=diff_tag_byte.height,
            stroke_color=REDUCIBLE_GREEN_DARKER
        )
        tag_value.text.scale_to_fit_height(diff_tag_byte.text[1][1].height)
        tag_value.text.rotate(PI)

        tag_value.next_to(diff_tag_byte, DOWN, aligned_edge=LEFT, buff=0)

        dg_value = Byte(
            "diff green (dg)",
            width=diff_tag_byte.get_right()[0] - tag_value.get_right()[0],
            height=tag_value.height,
            stroke_color=REDUCIBLE_GREEN_DARKER
        )

        dg_value.text.scale_to_fit_height(tag_value.text.height).scale(1.1)
        

        dg_value.next_to(diff_tag_byte, DOWN, aligned_edge=RIGHT, buff=0)

        second_byte = Byte(
            ["Byte[1]",
            "7,6,5,4,3,2,1,0"],
            stroke_color=REDUCIBLE_GREEN_DARKER
        ).scale(0.5).next_to(diff_tag_byte, RIGHT, buff=0)

        second_target_text = VGroup(
            diff_tag_byte.text[1][3],
            diff_tag_byte.text[1][4]
        )

        dr_dg_value = Byte(
            "dr - dg",
            width=second_target_text.get_center()[0] - dg_value.get_right()[0],
            height=dg_value.height,
            stroke_color=REDUCIBLE_GREEN_DARKER
        ).next_to(second_byte, DOWN, aligned_edge=LEFT, buff=0)
        dr_dg_value.text.scale_to_fit_height(dg_value.text.height)

        db_dg_value = Byte(
            "db - dg",
            width=dr_dg_value.width,
            height=dg_value.height,
            stroke_color=REDUCIBLE_GREEN_DARKER
        ).next_to(second_byte, DOWN, aligned_edge=RIGHT, buff=0)
        db_dg_value.text.scale_to_fit_height(dr_dg_value.text.height)

        qoi_diff_bytes = VGroup(
            diff_tag_byte, second_byte,
            tag_value, dg_value, dr_dg_value, db_dg_value,
        ).move_to(ORIGIN)

        return qoi_diff_bytes

    def get_large_diff_bytes_with_data(self, dg, dr_dg, db_dg, label=True):
        diff_byte = self.get_large_diff_bytes()
        diff_byte[-3].text.become(get_matching_text(str(dg), diff_byte[-3].text))
        diff_byte[-2].text.become(get_matching_text(str(dr_dg), diff_byte[-2].text))
        diff_byte[-1].text.become(get_matching_text(str(db_dg), diff_byte[-1].text))
        if label:
            label = Text("QOI_DIFF_MED", font="SF Mono", weight=MEDIUM).scale(0.4)
            label.next_to(diff_byte, UP)
            return VGroup(label, diff_byte)
        return diff_byte

    def get_small_diff_bytes(self):
        diff_tag_byte = Byte(
            ["Byte[0]",
            "7,6,5,4,3,2,1,0"],
            stroke_color=REDUCIBLE_GREEN_LIGHTER
        ).scale(0.5).move_to(DOWN * 2)

        target_text = VGroup(
            diff_tag_byte.text[1][1],
            diff_tag_byte.text[1][2]
        )
        tag_value = Byte(
            "1,0",
            width=target_text.get_center()[0] + SMALL_BUFF * 0.3 - diff_tag_byte.get_left()[0],
            height=diff_tag_byte.height,
            stroke_color=REDUCIBLE_GREEN_LIGHTER
        )
        tag_value.text.scale_to_fit_height(diff_tag_byte.text[1][1].height)
        tag_value.text.rotate(PI)
        tag_value.text.shift(RIGHT * SMALL_BUFF * 0.2)

        tag_value.next_to(diff_tag_byte, DOWN, aligned_edge=LEFT, buff=0)

        second_target_text = VGroup(
            diff_tag_byte.text[1][3],
            diff_tag_byte.text[1][4]
        )

        dr_value = Byte(
            "dr",
            width=second_target_text.get_center()[0] - tag_value.get_right()[0],
            height=tag_value.height,
            stroke_color=REDUCIBLE_GREEN_LIGHTER
        ).next_to(tag_value, RIGHT, buff=0)
        dr_value.text.scale_to_fit_height(tag_value.text.height)
        dr_value.text.rotate(PI)

        third_target_text = VGroup(
            diff_tag_byte.text[1][5],
            diff_tag_byte.text[1][6]
        )

        dg_value = Byte(
            "dg",
            width=third_target_text.get_center()[0] - dr_value.get_right()[0],
            height=tag_value.height,
            stroke_color=REDUCIBLE_GREEN_LIGHTER
        ).next_to(dr_value, RIGHT, buff=0)
        dg_value.text.scale_to_fit_height(dr_value.text.height).scale(1.2)
        dg_value.text.rotate(PI)

        db_value = Byte(
            "db",
            width=diff_tag_byte.get_right()[0] - third_target_text.get_center()[0],
            height=tag_value.height,
            stroke_color=REDUCIBLE_GREEN_LIGHTER
        ).next_to(dg_value, RIGHT, buff=0)
        db_value.text.scale_to_fit_height(dr_value.text.height)
        db_value.text.rotate(PI)

        qoi_diff_bytes = VGroup(
            diff_tag_byte, 
            tag_value, dr_value, dg_value, db_value
        )

        return qoi_diff_bytes

    def get_small_diff_bytes_with_data(self, dr, dg, db, label=True):
        diff_byte = self.get_small_diff_bytes()
        diff_byte[-3].text.become(get_matching_text(str(dr), diff_byte[-3].text))
        diff_byte[-2].text.become(get_matching_text(str(dg), diff_byte[-2].text))
        diff_byte[-1].text.become(get_matching_text(str(db), diff_byte[-1].text))
        if label:
            label = Text("QOI_DIFF_SMALL", font="SF Mono", weight=MEDIUM).scale(0.4)
            label.next_to(diff_byte, UP)
            return VGroup(label, diff_byte)
        return diff_byte

    def get_pixel_values(self, channel, channel_mob, mode='R'):
        pixel_values_text = VGroup()
        for p_val, mob in zip(channel.flatten(), channel_mob):
            text = Text(str(int(p_val)), font="SF Mono", weight=MEDIUM).scale(0.25).move_to(mob.get_center())
            if mode == 'G' and p_val > 200:
                text.set_color(BLACK)
            pixel_values_text.add(text)

        return pixel_values_text


    def get_channel_image(self, channel, mode='R'):
        new_channel = np.zeros((channel.shape[0], channel.shape[1], 3))
        for i in range(channel.shape[0]):
            for j in range(channel.shape[1]):
                if mode == 'R': 
                    new_channel[i][j] = np.array([channel[i][j], 0, 0])
                elif mode == 'G':
                    new_channel[i][j] = np.array([0, channel[i][j], 0])
                else:
                    new_channel[i][j] = np.array([0, 0, channel[i][j]])

        return new_channel

    def reshape_channel(self, channel):
        return np.reshape(channel, (1, channel.shape[0] * channel.shape[1], channel.shape[2]))

    def get_flatten_transform(self, original_mob, flattened_mob, original_text, flattened_text):
        transforms = []
        for i in range(original_mob.shape[0]):
            for j in range(original_mob.shape[1]):
                one_d_index = i * original_mob.shape[1] + j
                transforms.append(TransformFromCopy(original_mob[i, j], flattened_mob[0, one_d_index]))
                transforms.append(TransformFromCopy(original_text[one_d_index], flattened_text[one_d_index]))

        return transforms

class SumUpOfPNG(Scene):
    def construct(self):
        pixel_array, pixel_array_mob = self.get_image()
        rgb_rep = self.show_rgb_split(pixel_array, pixel_array_mob, animate=False)
        self.show_diagram(rgb_rep)

    def show_diagram(self, rgb_rep):
        rgb_rep.scale(0.6)
        scale = 0.55
        filtering_mod = Module(
            "Filtering",
            fill_color=REDUCIBLE_YELLOW_DARKER,
            stroke_color=REDUCIBLE_YELLOW,
            text_weight=BOLD
        ).scale(scale)

        LZSS_mod = Module(
            "LZSS", 
            fill_color=REDUCIBLE_GREEN_DARKER,
            stroke_color=REDUCIBLE_GREEN_LIGHTER,
            text_weight=BOLD
        ).scale(scale)

        huffman_mod = Module(
            ["Huffman","Coding"],
            text_weight=BOLD
        ).scale(scale)

        output_file = SVGMobject("png_file").scale_to_fit_height(huffman_mod.height + SMALL_BUFF * 2)

        diagram = VGroup(rgb_rep, filtering_mod, LZSS_mod, huffman_mod, output_file).arrange(RIGHT, buff=1)
        diagram.move_to(UP * 3)
        arrows = VGroup()

        for i in range(len(diagram) - 1):
            arrow = Arrow(
                diagram[i].get_right(), diagram[i + 1].get_left(), 
                buff=SMALL_BUFF,
                max_tip_length_to_length_ratio=0.15
            ).set_color(GRAY)
            arrows.add(arrow)

        

        self.play(
            FadeIn(diagram[0])
        )

        screen_rect = ScreenRectangle(height=4.5)
        screen_rect.move_to(DOWN * 1)

        self.play(
            Write(arrows[0]),
            FadeIn(diagram[1]),
            FadeIn(screen_rect)
        )
        self.wait()

        self.play(
            Write(arrows[1]),
            FadeIn(diagram[2])
        )

        self.wait()

        self.play(
            Write(arrows[2]),
            FadeIn(diagram[3])
        )
        self.wait()

        self.play(
            Write(arrows[3]),
            FadeIn(diagram[4])
        )
        self.wait()

        surround_rect = SurroundingRectangle(VGroup(LZSS_mod, huffman_mod), buff=SMALL_BUFF, color=REDUCIBLE_YELLOW)
        self.play(
            Create(surround_rect)
        )
        deflate = Text("DEFLATE compression", font="CMU Serif", weight=BOLD).scale(0.7)
        deflate.next_to(surround_rect, DOWN)

        self.play(
            Write(deflate)
        )
        self.wait()

        self.play(
            FadeOut(deflate),
            FadeOut(surround_rect),
            FadeOut(screen_rect)
        )

        png_encoding_label = Text("PNG Encoding", font="CMU Serif", weight=BOLD).scale(0.8).next_to(diagram, DOWN)

        self.play(
            Write(png_encoding_label)
        )
        self.wait()

        huffman_mod_dec = Module(
            ["Huffman","Decoder"],
            text_weight=BOLD
        ).scale(scale)

        LZSS_mod_dec = Module(
            ["LZSS", "Decoder"], 
            fill_color=REDUCIBLE_GREEN_DARKER,
            stroke_color=REDUCIBLE_GREEN_LIGHTER,
            text_weight=BOLD
        ).scale(scale)

        reverse_filtering = Module(
            ["Reverse", "Filtering"],
            fill_color=REDUCIBLE_YELLOW_DARKER,
            stroke_color=REDUCIBLE_YELLOW,
            text_weight=BOLD
        ).scale(scale)

        decoding_diagram = VGroup(
            output_file.copy(),
            huffman_mod_dec,
            LZSS_mod_dec,
            reverse_filtering,
            rgb_rep.copy()
        ).arrange(RIGHT, buff=1)

        arrows_dec = VGroup()

        for i in range(len(decoding_diagram) - 1):
            arrow = Arrow(
                decoding_diagram[i].get_right(), decoding_diagram[i + 1].get_left(), 
                buff=SMALL_BUFF,
                max_tip_length_to_length_ratio=0.15
            ).set_color(GRAY)
            arrows_dec.add(arrow)

        self.play(
            TransformFromCopy(output_file, decoding_diagram[0])
        )
        self.play(
            Write(arrows_dec[0])
        )
        self.play(
            FadeIn(decoding_diagram[1])
        )

        self.play(
            Write(arrows_dec[1])
        )

        self.play(
            FadeIn(decoding_diagram[2])
        )

        self.play(
            Write(arrows_dec[2])
        )

        self.play(
            FadeIn(decoding_diagram[3])
        )

        self.play(
            Write(arrows_dec[3])
        )

        self.play(
            TransformFromCopy(rgb_rep, decoding_diagram[-1])
        )

        png_decoding_label =  png_encoding_label = Text("PNG Decoding", font="CMU Serif", weight=BOLD).scale(0.8).next_to(decoding_diagram, DOWN)

        self.play(
            Write(png_decoding_label)
        )

        self.wait()

        png_good = Text("PNG is the best lossless image format based on compression ratio", font="CMU Serif").scale(0.7)

        observation = Text("PNG is computationally expensive and complex", font="CMU Serif").scale(0.7)
        observation.move_to(DOWN * 3)

        filtering = Text("5 filter operations/row", font="CMU Serif").scale(0.7).move_to(observation.get_center())

        lzss = Text("LZSS can be expensive", font="CMU Serif").scale(0.7).move_to(observation.get_center())

        huffman_codes = Text("PNG specific Huffman Coding is wildly complex").scale(0.7).move_to(observation.get_center())

        self.play(
            FadeIn(observation)
        )
        self.wait()

        self.play(
            ReplacementTransform(observation, filtering)
        )
        self.wait()

        self.play(
            ReplacementTransform(filtering, lzss)
        )
        self.wait()

        self.play(
            ReplacementTransform(lzss, huffman_codes)
        )
        self.wait()

    def get_image(self):
        image = ImageMobject("r.png")
        pixel_array = image.get_pixel_array().astype(int)
        pixel_array_mob = PixelArray(pixel_array).scale(0.4).shift(UP * 2)

        return pixel_array, pixel_array_mob

    def show_rgb_split(self, pixel_array, pixel_array_mob, animate=True):
        r_channel = pixel_array[:, :, 0]
        g_channel = pixel_array[:, :, 1]
        b_channel = pixel_array[:, :, 2]

        r_channel_padded = self.get_channel_image(r_channel)
        g_channel_padded = self.get_channel_image(g_channel, mode='G')
        b_channel_padded = self.get_channel_image(b_channel, mode='B')

        pixel_array_mob_r = PixelArray(r_channel_padded).scale(0.4).shift(LEFT * 4 + DOWN * 1.5)
        pixel_array_mob_g = PixelArray(g_channel_padded).scale(0.4).shift(DOWN * 1.5)
        pixel_array_mob_b = PixelArray(b_channel_padded).scale(0.4).shift(RIGHT * 4 + DOWN * 1.5)
        if animate:
            self.play(
                TransformFromCopy(pixel_array_mob, pixel_array_mob_r),
                TransformFromCopy(pixel_array_mob, pixel_array_mob_b),
                TransformFromCopy(pixel_array_mob, pixel_array_mob_g)
            )
            self.wait()

        r_channel_pixel_text = self.get_pixel_values(r_channel, pixel_array_mob_r)
        g_channel_pixel_text = self.get_pixel_values(g_channel, pixel_array_mob_g, mode='G')
        b_channel_pixel_text = self.get_pixel_values(b_channel, pixel_array_mob_b, mode='B')
        if animate:
            self.play(
                FadeIn(r_channel_pixel_text),
                FadeIn(g_channel_pixel_text),
                FadeIn(b_channel_pixel_text)
            )
            self.wait()

        rgb_rep = VGroup(
            VGroup(pixel_array_mob_b, b_channel_pixel_text),
            VGroup(pixel_array_mob_g, g_channel_pixel_text),
            VGroup(pixel_array_mob_r, r_channel_pixel_text)
        )

        pixel_width, pixel_height = pixel_array_mob[0, 0].width, pixel_array_mob[0, 0].height
        scale = 0.6
        rgb_rep.scale(scale)
        rgb_rep[2].move_to(ORIGIN)
        offset_v = (pixel_height * UP + pixel_width * RIGHT) * scale
        rgb_rep[1].move_to(offset_v)
        rgb_rep[0].move_to(offset_v * 2)
        return rgb_rep

    def get_pixel_values(self, channel, channel_mob, mode='R'):
        pixel_values_text = VGroup()
        for p_val, mob in zip(channel.flatten(), channel_mob):
            text = Text(str(int(p_val)), font="SF Mono", weight=MEDIUM).scale(0.25).move_to(mob.get_center())
            if mode == 'G' and p_val > 200:
                text.set_color(BLACK)
            pixel_values_text.add(text)

        return pixel_values_text

    def get_channel_image(self, channel, mode='R'):
        new_channel = np.zeros((channel.shape[0], channel.shape[1], 3))
        for i in range(channel.shape[0]):
            for j in range(channel.shape[1]):
                if mode == 'R': 
                    new_channel[i][j] = np.array([channel[i][j], 0, 0])
                elif mode == 'G':
                    new_channel[i][j] = np.array([0, channel[i][j], 0])
                else:
                    new_channel[i][j] = np.array([0, 0, channel[i][j]])

        return new_channel

class SumUpOfPNGPIP(Scene):
    def construct(self):
        pixel_array, pixel_array_mob = self.get_image()
        rgb_rep = self.show_rgb_split(pixel_array, pixel_array_mob, animate=False)
        self.show_diagram(rgb_rep)

    def show_diagram(self, rgb_rep):
        rgb_rep.scale(0.6)
        scale = 0.55
        filtering_mod = Module(
            "Filtering",
            fill_color=REDUCIBLE_YELLOW_DARKER,
            stroke_color=REDUCIBLE_YELLOW,
            text_weight=BOLD
        ).scale(scale)

        LZSS_mod = Module(
            "LZSS", 
            fill_color=REDUCIBLE_GREEN_DARKER,
            stroke_color=REDUCIBLE_GREEN_LIGHTER,
            text_weight=BOLD
        ).scale(scale)

        huffman_mod = Module(
            ["Huffman","Coding"],
            text_weight=BOLD
        ).scale(scale)

        output_file = SVGMobject("png_file").scale_to_fit_height(huffman_mod.height + SMALL_BUFF * 2)

        diagram = VGroup(rgb_rep, filtering_mod, LZSS_mod, huffman_mod, output_file).arrange(RIGHT, buff=1)
        diagram.move_to(UP * 1.5)
        arrows = VGroup()

        for i in range(len(diagram) - 1):
            arrow = Arrow(
                diagram[i].get_right(), diagram[i + 1].get_left(), 
                buff=SMALL_BUFF,
                max_tip_length_to_length_ratio=0.15
            ).set_color(GRAY)
            arrows.add(arrow)

        

        self.play(
            FadeIn(diagram[0])
        )

        # screen_rect = ScreenRectangle(height=4.5)
        # screen_rect.move_to(DOWN * 1)

        self.play(
            Write(arrows[0]),
            FadeIn(diagram[1]),
            # FadeIn(screen_rect)
        )
        self.wait()

        self.play(
            Write(arrows[1]),
            FadeIn(diagram[2])
        )

        self.wait()

        self.play(
            Write(arrows[2]),
            FadeIn(diagram[3])
        )
        self.wait()

        self.play(
            Write(arrows[3]),
            FadeIn(diagram[4])
        )
        self.wait()

        # surround_rect = SurroundingRectangle(VGroup(LZSS_mod, huffman_mod), buff=SMALL_BUFF, color=REDUCIBLE_YELLOW)
        # self.play(
        #     Create(surround_rect)
        # )
        # deflate = Text("DEFLATE compression", font="CMU Serif", weight=BOLD).scale(0.7)
        # deflate.next_to(surround_rect, DOWN)

        # self.play(
        #     Write(deflate)
        # )
        # self.wait()

        # self.play(
        #     FadeOut(deflate),
        #     FadeOut(surround_rect),
        #     FadeOut(screen_rect)
        # )

        png_encoding_label = Text("PNG Encoding", font="CMU Serif", weight=BOLD).scale(0.8).next_to(diagram, DOWN)

        self.play(
            Write(png_encoding_label)
        )
        self.wait()

        huffman_mod_dec = Module(
            ["Huffman","Decoder"],
            text_weight=BOLD
        ).scale(scale)

        LZSS_mod_dec = Module(
            ["LZSS", "Decoder"], 
            fill_color=REDUCIBLE_GREEN_DARKER,
            stroke_color=REDUCIBLE_GREEN_LIGHTER,
            text_weight=BOLD
        ).scale(scale)

        reverse_filtering = Module(
            ["Reverse", "Filtering"],
            fill_color=REDUCIBLE_YELLOW_DARKER,
            stroke_color=REDUCIBLE_YELLOW,
            text_weight=BOLD
        ).scale(scale)

        decoding_diagram = VGroup(
            output_file.copy(),
            huffman_mod_dec,
            LZSS_mod_dec,
            reverse_filtering,
            rgb_rep.copy()
        ).arrange(RIGHT, buff=1).move_to(DOWN * 1.5)

        arrows_dec = VGroup()

        for i in range(len(decoding_diagram) - 1):
            arrow = Arrow(
                decoding_diagram[i].get_right(), decoding_diagram[i + 1].get_left(), 
                buff=SMALL_BUFF,
                max_tip_length_to_length_ratio=0.15
            ).set_color(GRAY)
            arrows_dec.add(arrow)

        self.play(
            TransformFromCopy(output_file, decoding_diagram[0])
        )
        self.play(
            Write(arrows_dec[0])
        )
        self.play(
            FadeIn(decoding_diagram[1])
        )

        self.play(
            Write(arrows_dec[1])
        )

        self.play(
            FadeIn(decoding_diagram[2])
        )

        self.play(
            Write(arrows_dec[2])
        )

        self.play(
            FadeIn(decoding_diagram[3])
        )

        self.play(
            Write(arrows_dec[3])
        )

        self.play(
            TransformFromCopy(rgb_rep, decoding_diagram[-1])
        )

        png_decoding_label =  png_encoding_label = Text("PNG Decoding", font="CMU Serif", weight=BOLD).scale(0.8).next_to(decoding_diagram, DOWN)

        self.play(
            Write(png_decoding_label)
        )

        self.wait()

        png_good = Text("PNG is the best lossless image format based on compression ratio", font="CMU Serif").scale(0.7)

        observation = Text("PNG is computationally expensive and complex", font="CMU Serif").scale(0.7)
        observation.move_to(DOWN * 3)

        filtering = Text("5 filter operations/row", font="CMU Serif").scale(0.7).move_to(observation.get_center())

        lzss = Text("LZSS can be expensive", font="CMU Serif").scale(0.7).move_to(observation.get_center())

        huffman_codes = Text("PNG specific Huffman Coding is wildly complex").scale(0.7).move_to(observation.get_center())

        self.play(
            FadeIn(observation)
        )
        self.wait()

        self.play(
            ReplacementTransform(observation, filtering)
        )
        self.wait()

        self.play(
            ReplacementTransform(filtering, lzss)
        )
        self.wait()

        self.play(
            ReplacementTransform(lzss, huffman_codes)
        )
        self.wait()

    def get_image(self):
        image = ImageMobject("r.png")
        pixel_array = image.get_pixel_array().astype(int)
        pixel_array_mob = PixelArray(pixel_array).scale(0.4).shift(UP * 2)

        return pixel_array, pixel_array_mob

    def show_rgb_split(self, pixel_array, pixel_array_mob, animate=True):
        r_channel = pixel_array[:, :, 0]
        g_channel = pixel_array[:, :, 1]
        b_channel = pixel_array[:, :, 2]

        r_channel_padded = self.get_channel_image(r_channel)
        g_channel_padded = self.get_channel_image(g_channel, mode='G')
        b_channel_padded = self.get_channel_image(b_channel, mode='B')

        pixel_array_mob_r = PixelArray(r_channel_padded).scale(0.4).shift(LEFT * 4 + DOWN * 1.5)
        pixel_array_mob_g = PixelArray(g_channel_padded).scale(0.4).shift(DOWN * 1.5)
        pixel_array_mob_b = PixelArray(b_channel_padded).scale(0.4).shift(RIGHT * 4 + DOWN * 1.5)
        if animate:
            self.play(
                TransformFromCopy(pixel_array_mob, pixel_array_mob_r),
                TransformFromCopy(pixel_array_mob, pixel_array_mob_b),
                TransformFromCopy(pixel_array_mob, pixel_array_mob_g)
            )
            self.wait()

        r_channel_pixel_text = self.get_pixel_values(r_channel, pixel_array_mob_r)
        g_channel_pixel_text = self.get_pixel_values(g_channel, pixel_array_mob_g, mode='G')
        b_channel_pixel_text = self.get_pixel_values(b_channel, pixel_array_mob_b, mode='B')
        if animate:
            self.play(
                FadeIn(r_channel_pixel_text),
                FadeIn(g_channel_pixel_text),
                FadeIn(b_channel_pixel_text)
            )
            self.wait()

        rgb_rep = VGroup(
            VGroup(pixel_array_mob_b, b_channel_pixel_text),
            VGroup(pixel_array_mob_g, g_channel_pixel_text),
            VGroup(pixel_array_mob_r, r_channel_pixel_text)
        )

        pixel_width, pixel_height = pixel_array_mob[0, 0].width, pixel_array_mob[0, 0].height
        scale = 0.6
        rgb_rep.scale(scale)
        rgb_rep[2].move_to(ORIGIN)
        offset_v = (pixel_height * UP + pixel_width * RIGHT) * scale
        rgb_rep[1].move_to(offset_v)
        rgb_rep[0].move_to(offset_v * 2)
        return rgb_rep

    def get_pixel_values(self, channel, channel_mob, mode='R'):
        pixel_values_text = VGroup()
        for p_val, mob in zip(channel.flatten(), channel_mob):
            text = Text(str(int(p_val)), font="SF Mono", weight=MEDIUM).scale(0.25).move_to(mob.get_center())
            if mode == 'G' and p_val > 200:
                text.set_color(BLACK)
            pixel_values_text.add(text)

        return pixel_values_text

    def get_channel_image(self, channel, mode='R'):
        new_channel = np.zeros((channel.shape[0], channel.shape[1], 3))
        for i in range(channel.shape[0]):
            for j in range(channel.shape[1]):
                if mode == 'R': 
                    new_channel[i][j] = np.array([channel[i][j], 0, 0])
                elif mode == 'G':
                    new_channel[i][j] = np.array([0, channel[i][j], 0])
                else:
                    new_channel[i][j] = np.array([0, 0, channel[i][j]])

        return new_channel
        
class PNGPerformance(Scene):
    def construct(self):
        self.show_png_perf()

    def show_png_perf(self):
        """
        Compressing sample_5184×3456 to png
        Time elapsed: 6.790195941 s

        Compressing sample_5184×3456 to jpg
        Time elapsed: 0.5938231489999994 s

        --------------------------------------------------
        Compressing sample_1280×853 to png
        Time elapsed: 0.427553133 s

        Compressing sample_1280×853 to jpg
        Time elapsed: 0.030787092000000627 s

        --------------------------------------------------
        Compressing sample_1920×1280 to png
        Time elapsed: 0.9290773519999993 s

        Compressing sample_1920×1280 to jpg
        Time elapsed: 0.07258648999999906 s

        --------------------------------------------------
        Compressing sample_640×426 to png
        Time elapsed: 0.10736526900000065 s

        Compressing sample_640×426 to jpg
        Time elapsed: 0.007995269000000249 s

        -------------------------------------------------
        """
        test_img = ImageMobject("sample_1920×1280").move_to(LEFT * 4)
        test_img.scale(0.5)

        test_img_label = Text("Test Image", font="CMU Serif", weight=BOLD).scale(0.5).next_to(test_img, UP)
        bar_chart = ReducibleBarChart(
            [6.79, 0.59],
            max_value=8,
            bar_colors=[REDUCIBLE_YELLOW, REDUCIBLE_PURPLE],
            bar_names=["PNG", "JPEG"],
            bar_label_scale_val=0.5,
            height=5,
            width=6,
        ).move_to(RIGHT * 3)

        title_line1 = Text("PNG vs JPEG Encoding Speed", font='CMU Serif', weight=BOLD).scale(0.5)
        title_line2 = Text("(5184x3456 Image)", font='CMU Serif', weight=BOLD).scale(0.5).next_to(title_line1, DOWN, buff=SMALL_BUFF * 2)
        title = VGroup(title_line1, title_line2).next_to(bar_chart, UP)

        y_label = Text("Time (s)", font="SF Mono").scale(0.5).rotate(PI/2)
        y_label.next_to(bar_chart, LEFT)

        self.play(
            Write(bar_chart),
            Write(title),
            Write(y_label),
            FadeIn(test_img),
            Write(test_img_label),
            run_time=2
        )
        self.wait()
        note = Text("*Tests run on Intel i7 quad core, 16 GB RAM", font="CMU Serif").scale(0.3)
        note.next_to(bar_chart, DOWN)

        self.add(note)
        self.wait()

        question = Tex("Is there a simpler, faster, and somewhat comparable compression scheme?").scale(0.7)
        question.move_to(DOWN * 3.4)

        self.play(
            FadeOut(note),
            Write(question),
            run_time=2
        )
        self.wait()


class QOIByteAligined(QOIDemo):
    def construct(self):
        qoi_bitstream = [
            self.get_small_diff_bytes_with_data(1, 0, -1),
            self.get_run_bytes_with_data(3),
            self.get_large_diff_bytes_with_data(20, -3, 5)
        ]

        qoi_bitstream = VGroup(*qoi_bitstream).arrange(RIGHT, buff=SMALL_BUFF)
        qoi_bitstream.shift(UP * 0.5)

        observation = Text("Decoder can process bytes without worrying about overlap", font="CMU Serif").scale(0.7)
        observation.next_to(qoi_bitstream, DOWN * 2)
        self.play(
            FadeIn(qoi_bitstream)
        )
        self.wait()

        self.play(
            Write(observation)
        )
        self.wait()

class QOISummary(Scene):
    def construct(self):
        screen_rect = ScreenRectangle(height=5).move_to(UP * 0.5)
        core_qoi = Text("That's mostly the core QOI encoding algorithm (*)", font="CMU Serif").scale(0.8).next_to(screen_rect, DOWN)
        note = Text("*QOI can also handle alpha/transparancy channels (like PNG)", font="CMU Serif").scale(0.5).move_to(DOWN * 3.5)
        self.add(screen_rect)
        note.next_to(core_qoi, DOWN * 2)
        self.play(
            Write(core_qoi)
        )
        self.wait()

        self.add(note)
        self.wait()

class Conclusion(Scene):
    """
    GIF - 1987
    JPEG - 1992
    PNG - 1995
    QOI - 2021
    """
    def construct(self):
        number_line = NumberLine(
            x_range=(1980, 2022, 5),
            include_numbers=True,
            numbers_to_include=[1980, 1990, 2000, 2010, 2020],
            numbers_with_elongated_ticks=[1980, 1990, 2000, 2010, 2020],
            length=12,
            color=REDUCIBLE_PURPLE,
            decimal_number_config={
                "group_with_commas": False,
                "num_decimal_places": 0,
            }
        )

        gif = Text("GIF", font="CMU Serif", weight=BOLD).scale(0.7)
        gif_dot = Dot().set_color(REDUCIBLE_GREEN_LIGHTER).move_to(number_line.n2p(1987))
        gif.next_to(gif_dot, DOWN * 3)


        gif_line = DashedLine(
            gif_dot.get_center(),
            gif.get_top()
        ).set_color(REDUCIBLE_GREEN_LIGHTER)

        jpeg = Text("JPEG", font="CMU Serif", weight=BOLD).scale(0.7)
        jpeg_dot = Dot().set_color(ORANGE).move_to(number_line.n2p(1992))
        jpeg.next_to(jpeg_dot, UP * 3)

        jpeg_line = DashedLine(
            jpeg_dot.get_center(),
            jpeg.get_bottom()
        ).set_color(ORANGE)

        png = Text("PNG", font="CMU Serif", weight=BOLD).scale(0.7)
        png_dot = Dot().set_color(REDUCIBLE_VIOLET).move_to(number_line.n2p(1995))
        png.next_to(png_dot, DOWN * 3)

        png_line = DashedLine(
            png_dot.get_center(),
            png.get_top()
        ).set_color(REDUCIBLE_VIOLET)

        qoi = Text("QOI", font="CMU Serif", weight=BOLD).scale(0.7)
        qoi_dot = Dot().set_color(REDUCIBLE_YELLOW).move_to(number_line.n2p(2021))
        qoi.next_to(qoi_dot, UP * 3)

        qoi_line = DashedLine(
            qoi_dot.get_center(),
            qoi.get_bottom()
        ).set_color(YELLOW)


        self.play(
            Write(number_line),
        )
        self.wait()

        self.play(
            Write(gif_line)
        )
        self.play(
            Write(gif.shift(DOWN * SMALL_BUFF))
        )
        self.wait()

        self.play(
            Write(jpeg_line),
            Write(jpeg.shift(UP * SMALL_BUFF))
        )
        self.wait()

        self.play(
            Write(png_line)
        )
        self.play(
            Write(png.shift(DOWN * SMALL_BUFF))
        )
        self.wait()

        self.play(
            Write(qoi_line)
        )
        self.play(
            Write(qoi.shift(UP * SMALL_BUFF))
        )
        self.wait()

        quote = Tex("``I have no idea what I'm doing. I'm not a compression guy''").scale(0.8).move_to(DOWN * 2.5)

        dominic = Tex("- Dominic Szablewski (creator of QOI)").scale(0.7).next_to(quote, DOWN)

        self.play(
            Write(quote)
        )

        self.play(
            FadeIn(dominic)
        )
        self.wait()


        self.clear()

        screen_rect = ScreenRectangle(height=5)

        self.play(
            Create(screen_rect)
        )
        self.wait()

        real_beauty = Text("Real beauty of QOI is its simplicity", font="CMU Serif").scale(0.8)
        real_beauty.next_to(screen_rect, DOWN)

        self.play(
            Write(real_beauty)
        )
        self.wait()

        self.play(
            FadeOut(real_beauty)
        )
        self.wait()

class Patreons(Scene):
    def construct(self):
        thanks = Tex("Special Thanks to These Patreons").scale(1.2)
        patreons = ["Burt Humburg", "Winston Durand", r"Adam D\v{r}ínek", "Andreas", "Matt Q"]
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
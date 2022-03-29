from manim import *
from functions import *
from classes import *
from reducible_colors import *
from manim.mobject.geometry import ArrowTriangleFilledTip

config["assets_dir"] = "assets"

class SceneUtils(Scene):
    """
    Might be better to have a Letter class of some sort that stores information
    similar to how the RGBMob class stores info
    TODO: Redesign these methods for that usecase on Wednesday
    On Wednesday create a simple LZSS animation working
    """
    def construct(self):
        pass

    def get_indication(self, letter_mob, shift=SMALL_BUFF, direction=UP):
        """
        @param: letter_mob - VGroup(square, letter) for a letter surrounded by square
        """
        indicated_mob = letter_mob.copy()
        indicated_mob[1].set_fill(opacity=1)
        indicated_mob[0].set_stroke(opacity=1).set_fill(opacity=1)
        indicated_mob.shift(direction * shift)
        return indicated_mob

    def get_unindication(self, letter_mob, opacity=0.2, center=0):
        unindicated_mob = letter_mob.copy()
        unindicated_mob[1].set_fill(opacity=opacity)
        unindicated_mob[0].set_stroke(opacity=opacity).set_fill(opacity=opacity)
        shift_down = unindicated_mob.get_center()[1] - center
        unindicated_mob.shift(DOWN * shift_down)
        return unindicated_mob


class LZSSText(SceneUtils):
    """
    TODO: Need to highlight search buffer, look ahead buffer, and sliding window
    Show sliding window glowing rectangle
    Show search buffer glowing rectangle
    Show look ahead buffer glowing rectangle
    len(search) + len(look ahead) = len(sliding window)
    
    Perhaps start off with the search buffer being black/GRAY
    Color code search buffer as REDUCIBLE_GREEN
    Color code look ahead buffer as REDUCIBLE_PURPLE
    Entire sliding window as REDUCIBLE VIOLET
    
    Need to be able to show offset and length visually
    Need to be able to update character(s) from look ahead to search buffer
    Need to be able to indicate character(s) from look ahead in search buffer
    - further note, need to be able to design for use case of matching string overlapping search and look ahead

    Design way of showing encoding of text as back references

    """

    def construct(self):
        STRING = "repetitive repeat"
        text, sequence = self.intro_text(STRING)
        self.sw_length = 15
        self.look_ahead_len = 4
        self.search_buffer_len = 11
        sliding_window_group, look_ahead_group = self.intro_defintions(text)

        
        search_sequence_group, look_ahead_sequence_group, out_of_range_group, sliding_window_rect = self.initialize_buffers(text, sequence, sliding_window_group, look_ahead_group)
        
        self.state = []

        encoded_text_title, encoded_text, fadeouts = self.perform_LZSS_animations(
            text, 
            search_sequence_group, 
            look_ahead_sequence_group, 
            out_of_range_group, 
            sequence,
            DOWN * 3 + LEFT * 6,
            sliding_window_rect
        )

        self.play(
            *fadeouts,
            encoded_text_title.animate.shift(UP * 3),
            encoded_text.animate.shift(UP * 3)
        )
        self.wait()

        self.show_decoding(encoded_text, STRING)

        self.clear()

        self.detail_sliding_window()

        self.show_special_lzss_case()

        self.show_run_length_enc_case()
        self.clear()

        self.show_actual_lzss_encoding(encoded_text, encoded_text_title, text)

    def show_actual_lzss_encoding(self, encoded_text, encoded_text_title, text):
        self.play(
            FadeIn(encoded_text),
            FadeIn(encoded_text_title),
            FadeIn(text)
        )
        self.wait()

        length_fact = Text("LZSS only encodes (offset, length) pairs when length ≥ 3", font='SF Mono', weight=MEDIUM).scale(0.5)

        length_fact.next_to(encoded_text, DOWN)

        self.play(
            FadeIn(length_fact)
        )
        self.wait()
        
        crosses = [Cross(encoded_text[3]), Cross(encoded_text[6]), Cross(encoded_text[8]), Cross(encoded_text[-1])]
        self.play(
            *[Write(c) for c in crosses]
        )
        self.wait()

        new_encoded_text = []
        for i, c in enumerate("repetitive repeat"):
            if i == 11:
                offset_mob = self.get_offset_length_mob(11, 4, encoded_text.height)
                new_encoded_text.append(offset_mob)
                continue
            elif i > 11 and i < 15:
                continue
            
            letter_mob = self.get_letter_mob(c, color=REDUCIBLE_GREEN_DARKER).scale_to_fit_height(encoded_text.height)
            new_encoded_text.append(letter_mob)

        actual_lzss_enc = Text("Actual LZSS encoding", font="SF Mono", weight=MEDIUM).scale(0.6)
        actual_lzss_enc.next_to(length_fact, DOWN * 2)

        new_encoded_text_group = VGroup(*new_encoded_text).arrange(RIGHT, buff=SMALL_BUFF)

        new_encoded_text_group.next_to(actual_lzss_enc, DOWN * 2)

        self.play(
            FadeIn(actual_lzss_enc),
            FadeIn(new_encoded_text_group)
        )
        self.wait()

    def intro_text(self, string, sequence_pos=ORIGIN):
        text = Text(string, font='SF Mono', weight=MEDIUM).shift(UP * 0.5)
        self.play(
            AddTextLetterByLetter(text)
        )
        self.wait()

        sequence = self.get_letter_sequence(string).scale(0.6).shift(DOWN * 0.5)
        
        transforms = self.get_transforms_text_to_sequence(string, text, sequence)
        self.play(
            *[t for t in transforms if isinstance(t, FadeIn)]
        )
        self.play(
            *[t for t in transforms if isinstance(t, TransformFromCopy)]
        )
        self.wait()

        self.play(
            text.animate.move_to(UP * 3),
            sequence.animate.move_to(sequence_pos)
        )
        self.wait()

        return text, sequence

    def intro_defintions(self, text, animate=True):
        sliding_window = Text("Sliding Window", font='SF Mono', weight=MEDIUM).scale(0.6)
        look_ahead = Text("Look Ahead Buffer", font='SF Mono', weight=MEDIUM).scale(0.6)

        sliding_window_surround_rect = get_glowing_surround_rect(sliding_window, buff_min=SMALL_BUFF, buff_max=MED_SMALL_BUFF, color=REDUCIBLE_VIOLET)
        look_ahead_surround_rect = get_glowing_surround_rect(look_ahead, buff_min=SMALL_BUFF, buff_max=MED_SMALL_BUFF, color=REDUCIBLE_PURPLE)

        sliding_window_group = VGroup(sliding_window, sliding_window_surround_rect)
        look_ahead_group = VGroup(look_ahead, look_ahead_surround_rect)

        VGroup(sliding_window_group, look_ahead_group).arrange(RIGHT, buff=2).next_to(text, DOWN)

        if animate:
            self.play(
                Write(sliding_window)
            )

            self.play(
                Create(sliding_window_surround_rect)
            )

            self.wait()

            self.play(
                Write(look_ahead)
            )

            self.play(
                Create(look_ahead_surround_rect)
            )
            self.wait()

        left_adjust = LEFT * 0.5
        sliding_window_val = Text(f"= {self.sw_length}", font='SF Mono', weight=MEDIUM).scale(0.6).next_to(sliding_window_group, RIGHT).shift(left_adjust)
        look_ahead_val = Text(f"= {self.look_ahead_len}", font='SF Mono', weight=MEDIUM).scale(0.6).next_to(look_ahead_group, RIGHT).shift(left_adjust)
        if animate:
            self.play(
                sliding_window_group.animate.shift(left_adjust),
                look_ahead_group.animate.shift(left_adjust),
                Write(sliding_window_val),
                Write(look_ahead_val)
            )
            self.wait()
        else:
            sliding_window_group.shift(left_adjust)
            look_ahead_group.shift(left_adjust)

        return VGroup(sliding_window_group, sliding_window_val), VGroup(look_ahead_group, look_ahead_val)

    def initialize_buffers(self, text, sequence, sw_group, look_ahead_group, preanimatons=None, define_search=False, sequence_abs_pos=None, explain=True):
        index = list(range(len(text.original_text)))

        search_buffer_index = 0

        search_buffer_indices = index[:search_buffer_index]
        look_ahead_buffer_indices = index[search_buffer_index:search_buffer_index+self.look_ahead_len]
        search_sequence_group, look_ahead_sequence_group, out_of_range_group = self.get_search_and_look_ahead(text.original_text, sequence, search_buffer_indices, look_ahead_buffer_indices)

        search_sequence_group.to_edge(LEFT * 3).shift(DOWN * 0.5)
        look_ahead_sequence_group.next_to(search_sequence_group, RIGHT)
        out_of_range_group.next_to(look_ahead_sequence_group, RIGHT)

        if sequence_abs_pos is not None:
            VGroup(search_sequence_group, look_ahead_sequence_group, out_of_range_group).move_to(sequence_abs_pos)

        if preanimatons:
            if define_search:
                search_buff = Text("Search Buffer", font='SF Mono', weight=MEDIUM).scale(0.6)
                search_buff_text_rect = get_glowing_surround_rect(search_buff, buff_min=SMALL_BUFF, buff_max=MED_SMALL_BUFF, color=REDUCIBLE_GREEN_LIGHTER)
                search_buff_rect = get_glowing_surround_rect(search_sequence_group, buff_min=SMALL_BUFF, buff_max=MED_SMALL_BUFF, color=REDUCIBLE_GREEN_LIGHTER)
                
                search_buff_val = Text(f"= {self.search_buffer_len}", font='SF Mono', weight=MEDIUM).scale(0.6).next_to(search_buff_text_rect, RIGHT)

                search_buff_text_group = VGroup(search_buff, search_buff_text_rect, search_buff_val)
                search_buff_text_group.move_to(UP * 0.4 + RIGHT * 0.5)
                preanimatons.append(FadeIn(search_buff_text_group))
                self.post_animations = [FadeOut(search_buff_text_group)]

            self.play(
                *preanimatons
            )
            self.wait()

        self.play(
            *[ReplacementTransform(sequence[i], look_ahead_sequence_group[i]) for i in look_ahead_buffer_indices],
            *[ReplacementTransform(original, new) for original, new in zip(sequence[search_buffer_index+self.look_ahead_len:], out_of_range_group)],
            FadeIn(search_sequence_group),
            run_time=2
        )
        self.wait()

        if not explain:
            sliding_window_rect = get_glowing_surround_rect(VGroup(search_sequence_group, look_ahead_sequence_group), buff_min=SMALL_BUFF, buff_max=MED_SMALL_BUFF, color=REDUCIBLE_VIOLET)
            return search_sequence_group, look_ahead_sequence_group, out_of_range_group, sliding_window_rect

        sliding_window_rect, to_fade = self.explain_search_buffer(search_sequence_group, look_ahead_sequence_group)

        self.play(
            FadeOut(sw_group),
            FadeOut(look_ahead_group),
            FadeOut(to_fade),
            text.animate.shift(DOWN * 0.5),
            search_sequence_group.animate.shift(UP),
            look_ahead_sequence_group.animate.shift(UP),
            out_of_range_group.animate.shift(UP),
            sliding_window_rect.animate.shift(UP)
        )
        self.wait()

        return search_sequence_group, look_ahead_sequence_group, out_of_range_group, sliding_window_rect

    def explain_search_buffer(self, search_sequence_group, look_ahead_sequence_group):
        look_ahead_rect = get_glowing_surround_rect(look_ahead_sequence_group, buff_min=SMALL_BUFF, buff_max=MED_SMALL_BUFF, color=REDUCIBLE_PURPLE)
        brace_look_ahead = Brace(look_ahead_rect, DOWN).next_to(look_ahead_rect, DOWN, buff=SMALL_BUFF)
        look_ahead = Text("Look Ahead Buffer", font='SF Mono', weight=MEDIUM).scale(0.4)
        look_ahead.next_to(brace_look_ahead, DOWN)
        self.play(
            Create(look_ahead_rect),
            GrowFromCenter(brace_look_ahead),
            Write(look_ahead)
        )
        self.wait()


        search_buff = Text("Search Buffer", font='SF Mono', weight=MEDIUM).scale(0.6)
        search_buff_text_rect = get_glowing_surround_rect(search_buff, buff_min=SMALL_BUFF, buff_max=MED_SMALL_BUFF, color=REDUCIBLE_GREEN_LIGHTER)
        search_buff_rect = get_glowing_surround_rect(search_sequence_group, buff_min=SMALL_BUFF, buff_max=MED_SMALL_BUFF, color=REDUCIBLE_GREEN_LIGHTER)
        
        search_buff_val = Text("= 15 - 4", font='SF Mono', weight=MEDIUM).scale(0.6).next_to(search_buff_text_rect, RIGHT)

        search_buff_text_group = VGroup(search_buff, search_buff_text_rect, search_buff_val)
        search_buff_text_group.move_to(UP * 0.8 + RIGHT * 0.5)
        new_text_val = Text("= 11", font='SF Mono', weight=MEDIUM).scale(0.6).next_to(search_buff_text_rect, RIGHT)

        brace_search_buff = Brace(search_buff_rect, DOWN).next_to(search_buff_rect, DOWN, buff=SMALL_BUFF)
        search_buff_label = search_buff.copy().scale(0.4/0.6).next_to(brace_search_buff, DOWN)

        self.play(
            FadeOut(look_ahead_rect),
            FadeOut(brace_look_ahead),
            FadeOut(look_ahead_rect),
            FadeOut(look_ahead),
            Create(search_buff_rect),
            GrowFromCenter(brace_search_buff),
            FadeIn(search_buff),
            FadeIn(search_buff_text_rect),
            Write(search_buff_label)
        )
        
        self.play(
            Write(search_buff_val),
        )
        self.wait()

        self.play(
            Transform(search_buff_val, new_text_val)
        )
        self.wait()

        sliding_window_rect = get_glowing_surround_rect(VGroup(search_sequence_group, look_ahead_sequence_group), buff_min=SMALL_BUFF, buff_max=MED_SMALL_BUFF, color=REDUCIBLE_VIOLET)
        brace_sw = Brace(sliding_window_rect, DOWN).next_to(sliding_window_rect, DOWN, buff=SMALL_BUFF)
        sliding_window = Text("Sliding Window", font='SF Mono', weight=MEDIUM).scale(0.4)
        sliding_window.next_to(brace_sw, DOWN)

        self.play(
            FadeOut(brace_search_buff),
            FadeOut(search_buff_label),
            FadeOut(search_buff_rect),
        )
        self.wait()

        self.play(
            Create(sliding_window_rect),
            GrowFromCenter(brace_sw),
            Write(sliding_window)
        )
        self.wait()

        return sliding_window_rect, VGroup(search_buff_text_group, sliding_window, brace_sw)

    def detail_sliding_window(self):
        # to get ordering of mobs in the right order in scene
        self.play(
            *[FadeIn(mob) for mob in self.state[1:]] + [FadeIn(self.state[0])]
        )

        self.wait()
        look_ahead_sequence_group = self.state[-6]
        search_sequence_group = self.state[-5]
        new_surround_rect = get_glowing_surround_rect(
            VGroup(search_sequence_group[1:], look_ahead_sequence_group),
            buff_min=SMALL_BUFF,
            buff_max=MED_SMALL_BUFF,
            color=REDUCIBLE_VIOLET
        )
        additional_anims = []
        for i in range(len(search_sequence_group)):
            opacity = 1
            if i == 0:
                opacity = 0.2
            additional_anims.extend(
                [
                search_sequence_group[i][0].animate.set_fill(opacity=opacity).set_stroke(opacity=opacity),
                search_sequence_group[i][1].animate.set_fill(opacity=opacity)
                ]
            )

        self.play(
            *[FadeOut(self.state[i]) for i in range(5)],
            # original sliding window to new one
            Transform(self.state[-1], new_surround_rect),
            *additional_anims
        )
        self.wait()

        current_text = self.state[-2]
        cross = Cross(current_text[-1], color=PURE_RED)
        self.play(
            Write(cross)
        )
        self.wait()


        self.play(
            *[FadeOut(self.state[i]) for i in range(5, len(self.state))] + [FadeOut(cross)]
        )

        png_specific_params = Text("PNG Specific LZSS", font='SF Mono', weight=MEDIUM).move_to(UP * 3)

        self.play(
            Write(png_specific_params)
        )
        self.wait()

        sliding_window = Text("Sliding Window", font='SF Mono', weight=MEDIUM).scale(0.6)
        look_ahead = Text("Look Ahead Buffer", font='SF Mono', weight=MEDIUM).scale(0.6)

        sliding_window_surround_rect = get_glowing_surround_rect(sliding_window, buff_min=SMALL_BUFF, buff_max=MED_SMALL_BUFF, color=REDUCIBLE_VIOLET)
        look_ahead_surround_rect = get_glowing_surround_rect(look_ahead, buff_min=SMALL_BUFF, buff_max=MED_SMALL_BUFF, color=REDUCIBLE_PURPLE)

        sliding_window_val = Text("= 32 KB", font='SF Mono', weight=MEDIUM).scale(0.6).next_to(sliding_window_surround_rect, RIGHT)
        look_ahead_val = Text("= 258 Bytes", font='SF Mono', weight=MEDIUM).scale(0.6).next_to(look_ahead_surround_rect, RIGHT)

        sliding_window_group = VGroup(sliding_window, sliding_window_surround_rect, sliding_window_val)
        look_ahead_group = VGroup(look_ahead, look_ahead_surround_rect, look_ahead_val)
        
        sliding_window_group.move_to(ORIGIN)
        look_ahead_group.move_to(DOWN * 1)
        self.play(
            FadeIn(sliding_window_group)
        )
        self.wait()

        self.play(
            FadeIn(look_ahead_group)
        )
        self.wait()

        self.clear()

    def show_special_lzss_case(self):
        string = "abcdedede"
        self.sw_length = 9
        self.look_ahead_len = 4
        self.search_buffer_len = 5
        text, sequence = self.intro_text(string, sequence_pos=DOWN*1)

        sliding_window_group, look_ahead_group = self.intro_defintions(text, animate=False)

        sliding_window_group.shift(DOWN * 0.4)
        look_ahead_group.shift(DOWN * 0.4)

        search_sequence_group, look_ahead_sequence_group, out_of_range_group, sliding_window_rect = self.initialize_buffers(
            text, 
            sequence, 
            sliding_window_group, 
            look_ahead_group,
            preanimatons=[FadeIn(sliding_window_group), FadeIn(look_ahead_group)],
            define_search=True,
            sequence_abs_pos=DOWN*1,
            explain=False
        )

        self.play(
            FadeIn(sliding_window_rect)
        )
        self.wait()

        fadeouts = [FadeOut(sliding_window_group), FadeOut(look_ahead_group)]
        if self.post_animations:
            fadeouts += self.post_animations

        self.play(
            *fadeouts,
            text.animate.shift(DOWN * 0.5),
            search_sequence_group.animate.shift(UP),
            look_ahead_sequence_group.animate.shift(UP),
            out_of_range_group.animate.shift(UP),
            sliding_window_rect.animate.shift(UP)
        )
        self.wait()

        encoded_text_title, encoded_text, fadeouts = self.perform_LZSS_animations(
            text, 
            search_sequence_group, 
            look_ahead_sequence_group, 
            out_of_range_group, 
            sequence,
            DOWN * 3 + LEFT * 2.6,
            sliding_window_rect,
            wait=False
        )

        self.play(
            *fadeouts,
            encoded_text_title.animate.shift(UP * 3),
            encoded_text.animate.shift(UP * 3)
        )
        self.wait()

        decoded_text = self.show_decoding(encoded_text, string)

        underline = Underline(decoded_text[-4:]).set_color(REDUCIBLE_VIOLET)

        self.play(
            decoded_text[-4:].animate.set_color(REDUCIBLE_VIOLET)
        )
        self.wait()

        surround_rect = SurroundingRectangle(decoded_text[3:7]).set_color(REDUCIBLE_YELLOW)
        self.play(
            Create(surround_rect)
        )
        self.wait()

        self.clear()

    def show_run_length_enc_case(self):
        string = "aaaaa"
        self.sw_length = 9
        self.look_ahead_len = 4
        self.search_buffer_len = 5
        text, sequence = self.intro_text(string, sequence_pos=DOWN*1)

        sliding_window_group, look_ahead_group = self.intro_defintions(text, animate=False)

        sliding_window_group.shift(DOWN * 0.4)
        look_ahead_group.shift(DOWN * 0.4)

        search_sequence_group, look_ahead_sequence_group, out_of_range_group, sliding_window_rect = self.initialize_buffers(
            text, 
            sequence, 
            sliding_window_group, 
            look_ahead_group,
            preanimatons=[FadeIn(sliding_window_group), FadeIn(look_ahead_group)],
            define_search=True,
            sequence_abs_pos=DOWN*1,
            explain=False
        )

        self.play(
            FadeIn(sliding_window_rect)
        )
        self.wait()

        fadeouts = [FadeOut(sliding_window_group), FadeOut(look_ahead_group)]
        if self.post_animations:
            fadeouts += self.post_animations

        self.play(
            *fadeouts,
            text.animate.shift(DOWN * 0.5),
            search_sequence_group.animate.shift(UP),
            look_ahead_sequence_group.animate.shift(UP),
            out_of_range_group.animate.shift(UP),
            sliding_window_rect.animate.shift(UP)
        )
        self.wait()

        encoded_text_title, encoded_text, fadeouts = self.perform_LZSS_animations(
            text, 
            search_sequence_group, 
            look_ahead_sequence_group, 
            out_of_range_group, 
            sequence,
            DOWN * 3 + LEFT * 1.2,
            sliding_window_rect
        )

        self.play(
            *fadeouts,
            encoded_text_title.animate.shift(UP * 3),
            encoded_text.animate.shift(UP * 3)
        )
        self.wait()

        decoded_text = self.show_decoding(encoded_text, string)


    def perform_LZSS_animations(
        self, 
        text, 
        search_sequence_group, 
        look_ahead_sequence_group, 
        out_of_range_group, 
        sequence,
        encoded_text_position,
        sliding_window_rect,
        wait=True):
        """
        TODO: 
        add indication and encoding animations
        Add animations to the text as well
        """

        indices = list(range(len(text.original_text)))

        def lzss_helper(string, search, look_ahead):
            if len(look_ahead) == 0:
                return []
            
            largest_match = []
            look_ahead_index = 0
            first_char = string[look_ahead[look_ahead_index]]
            for index in search[::-1]:
                if string[index] == first_char:
                    largest_match.append(index)
                    break

            if len(largest_match) == 0:
                return []

            next_index = largest_match[0] + 1
            look_ahead_index += 1
            if look_ahead_index >= len(look_ahead):
                return largest_match

            while look_ahead_index < self.look_ahead_len:
                next_char = string[look_ahead[look_ahead_index]]
                if next_char != string[next_index]:
                    break
                else:
                    largest_match.append(next_index)

                next_index += 1
                look_ahead_index += 1

            return largest_match

        search_buffer = []
        look_ahead = indices[:self.look_ahead_len]
        start_index = 0
        end_index = self.look_ahead_len - 1
        count = 0
        encoded_text = Text("Encoded text", font="SF Mono", weight=MEDIUM)
        encoded_text.scale(0.6)
        encoded_text.shift(DOWN * 2)
        self.play(
            Write(encoded_text)
        )
        if wait:
            self.wait()
        # placeholder for later text
        current_text = Text("a").move_to(encoded_text_position)
        self.actual_encoding = []
        # print('Here', current_text.get_center())
        while len(look_ahead) > 0:
            print('Search', search_buffer, 'Look Ahead', look_ahead)
            largest_match = lzss_helper(text.original_text, search_buffer, look_ahead)
            print('Largest match', largest_match)
            animations, reset_transforms = self.get_letter_mob_to_encoding(
                look_ahead_sequence_group, 
                search_sequence_group, 
                encoded_text, 
                current_text, 
                largest_match, 
                look_ahead,
                sliding_window_rect
            )
            self.play(
                *animations
            )
            if wait:
                self.wait()
            # save this state for later animations
            if len(largest_match) == 4:
                self.state.extend([
                    text.copy(), 
                    look_ahead_sequence_group.copy(), 
                    search_sequence_group.copy(), 
                    out_of_range_group.copy(),
                    encoded_text.copy(),
                    current_text[1:].copy(),
                    sliding_window_rect.copy()
                    ]
                )

            if reset_transforms:
                self.play(
                    *reset_transforms
                )
                if wait:
                    self.wait()
            search_sequence_group, look_ahead_sequence_group, out_of_range_group = self.update_lzss(
                search_sequence_group, 
                look_ahead_sequence_group, 
                out_of_range_group, 
                sequence, 
                largest_match,
                encoded_text,
                wait=wait
            )

            if len(largest_match) <= 1:
                search_buffer.append(look_ahead.pop(0))
            else:
                for _ in range(len(largest_match)):
                    search_buffer.append(look_ahead.pop(0))

            while len(look_ahead) < self.look_ahead_len:
                end_index += 1
                if end_index >= len(text.original_text):
                    break
                else:
                    look_ahead.append(end_index)
            while len(search_buffer) > self.search_buffer_len:
                start_index += 1
                search_buffer.pop(0)

        fadeouts = [FadeOut(search_sequence_group), FadeOut(sliding_window_rect)]

        return encoded_text, current_text[1:], fadeouts

    def update_lzss(
        self, 
        search_sequence_group, 
        look_ahead_sequence_group, 
        out_of_range_group, 
        sequence, 
        largest_match,
        encoded_text,
        wait=True
        ):
        if len(largest_match) <= 1:
            search_shift_amount = search_sequence_group[0].get_center() - search_sequence_group[1].get_center()
        else:
            if len(largest_match) >= len(search_sequence_group):
                search_shift_amount = search_sequence_group[0].get_center() - look_ahead_sequence_group[len(largest_match) - len(search_sequence_group)].get_center()
            else:
                search_shift_amount = search_sequence_group[0].get_center() - search_sequence_group[len(largest_match)].get_center()

        num_elements_to_fade = 1 if len(largest_match) == 0 else len(largest_match)
        indices_to_fade = list(range(len(search_sequence_group)))[:num_elements_to_fade]
        animations = []
        for i in range(len(search_sequence_group)):
            if i in indices_to_fade:
                animations.append(search_sequence_group[i].animate.shift(search_shift_amount + LEFT * SMALL_BUFF * 2).fade(1))
            else:
                animations.append(ApplyMethod(search_sequence_group[i].shift, search_shift_amount))

        seq_to_search_shift = search_sequence_group[-num_elements_to_fade].get_center() - look_ahead_sequence_group[0].get_center()
        for i in range(num_elements_to_fade):
            if i >= len(look_ahead_sequence_group):
                break
            search_buffer_element = look_ahead_sequence_group[i].copy()
            search_buffer_element.shift(seq_to_search_shift)
            search_buffer_element[0].set_fill(color=REDUCIBLE_GREEN_LIGHTER).set_stroke(color=REDUCIBLE_GREEN_LIGHTER)
            animations.append(Transform(look_ahead_sequence_group[i], search_buffer_element))

        for i in range(num_elements_to_fade, len(look_ahead_sequence_group)):
            animations.append(look_ahead_sequence_group[i].animate.shift(search_shift_amount))

        if len(look_ahead_sequence_group) == 0 or len(out_of_range_group) == 0:
            self.play(
                *animations
            )
            if wait:
                self.wait()
            return self.get_new_buffers(search_sequence_group, look_ahead_sequence_group, out_of_range_group, num_elements_to_fade)

        out_to_look_shift = look_ahead_sequence_group[-num_elements_to_fade].get_center() - out_of_range_group[0].get_center()
        for i in range(num_elements_to_fade):
            if i >= len(out_of_range_group):
                break
            look_ahead_buff_element = out_of_range_group[i].copy().shift(out_to_look_shift)
            look_ahead_buff_element[0].set_fill(opacity=1).set_stroke(opacity=1)
            look_ahead_buff_element[1].set_fill(opacity=1)
            animations.append(Transform(out_of_range_group[i], look_ahead_buff_element))
        
        for i in range(num_elements_to_fade, len(out_of_range_group)):
            animations.append(out_of_range_group[i].animate.shift(search_shift_amount))

        self.play(
            *animations
        )

        if wait:
            self.wait()

        return self.get_new_buffers(search_sequence_group, look_ahead_sequence_group, out_of_range_group, num_elements_to_fade)

    def get_letter_mob_to_encoding(self, 
        look_ahead_sequence_group, 
        search_sequence_group, 
        encoded_text, 
        current_text, 
        largest_match, 
        look_ahead,
        sliding_window_rect
        ):
        if len(look_ahead_sequence_group) == 0:
            return []
        animations = []
        reset_transforms = []
        if len(largest_match) == 0:
            current_letter = look_ahead_sequence_group[0]
            self.actual_encoding.append(current_letter[1].original_text)
            encoded_letter = current_letter.copy()
            encoded_letter[0].set_color(REDUCIBLE_GREEN_DARKER)
            encoded_letter.next_to(current_text, RIGHT, buff=SMALL_BUFF)
            animations.append(TransformFromCopy(current_letter, encoded_letter))
            current_text.add(encoded_letter)
        else:
            offset = look_ahead[0] - largest_match[0]
            length = len(largest_match)
            if len(largest_match) == 4: 
                reset_transforms = self.show_indications(search_sequence_group, look_ahead_sequence_group, offset, length, sliding_window_rect, save=True)
            else:
                reset_transforms = self.show_indications(search_sequence_group, look_ahead_sequence_group, offset, length, sliding_window_rect, save=False)
            current_letters = look_ahead_sequence_group[:length]
            offset_length_mob = self.get_offset_length_mob(offset, length, current_letters.height)
            self.actual_encoding.append((offset, length))
            offset_length_mob.next_to(current_text, RIGHT, buff=SMALL_BUFF)
            animations.append(TransformFromCopy(current_letters, offset_length_mob))
            current_text.add(offset_length_mob)
        return animations, reset_transforms

    def show_indications(self, search_sequence_group, look_ahead_sequence_group, offset, length, sliding_window_rect, save=False):
        """
        TODO: explore if fading out glowing surround rect looks better
        TODO: handle case when offset is greater than length
        """
        indicies_to_indicate = [len(search_sequence_group) + i for i in range(-offset, -offset+length)]
        indication_animations = []
        to_indicate = VGroup()
        for i, letter_mob in enumerate(search_sequence_group):
            if i in indicies_to_indicate:
                to_indicate.add(letter_mob)
            else:
                indication_animations.append(letter_mob[0].animate.set_fill(opacity=0.2).set_stroke(opacity=0.2))
                indication_animations.append(letter_mob[1].animate.set_fill(opacity=0.2))
        # when length > offset
        look_ahead_index = 0
        while len(to_indicate) < length:
            to_indicate.add(look_ahead_sequence_group[look_ahead_index])
            look_ahead_index += 1
        original_indicate = to_indicate.copy()
        transformed_indicate = to_indicate.copy()
        surround_rect = get_glowing_surround_rect(transformed_indicate)
        if save:
            self.state.append(surround_rect.copy())
        indication_animations.extend([Transform(to_indicate, transformed_indicate), FadeIn(surround_rect)])
        self.play(
            *indication_animations
        )
        self.wait()

        offset_arrow = self.get_offset_arrow(
            look_ahead_sequence_group[0].get_bottom(),
            search_sequence_group[-offset].get_bottom(),
        ).shift(DOWN * SMALL_BUFF)

        length_marker = self.get_marker(
            look_ahead_sequence_group[0][0].get_vertices()[1],
            look_ahead_sequence_group[length - 1][0].get_vertices()[0],
        ).shift(UP * SMALL_BUFF * 3)

        offset_label = Text(f"offset = {offset}", font='SF Mono', weight=MEDIUM).scale(0.4)
        length_label = Text(f"length = {length}", font='SF Mono', weight=MEDIUM).scale(0.4)
        offset_label.next_to(offset_arrow, DOWN)
        length_label.next_to(length_marker, UP)
        self.play(
            FadeIn(offset_arrow),
            FadeIn(length_marker),
            FadeIn(offset_label),
            FadeIn(length_label),
        )
        self.wait()

        if save:
            self.state.extend([offset_arrow.copy(), length_marker.copy(), offset_label.copy(), length_label.copy()])

        reset_transforms = []
        for i, letter_mob in enumerate(search_sequence_group):
            if i not in indicies_to_indicate:
                reset_transforms.append(letter_mob[0].animate.set_fill(opacity=1).set_stroke(opacity=1))
                reset_transforms.append(letter_mob[1].animate.set_fill(opacity=1))
        reset_transforms.append(Transform(to_indicate, original_indicate))
        reset_transforms.extend(
            [
            FadeOut(surround_rect),
            FadeOut(offset_arrow),
            FadeOut(length_marker),
            FadeOut(offset_label),
            FadeOut(length_label),
            ]
        )
        return reset_transforms

    def get_offset_arrow(self, start, end, color=REDUCIBLE_YELLOW):
        angle = -TAU/4
        if start[0] - end[0] > 5:
            angle = -TAU/6
        return self.get_curved_arrow(start, end, angle)

    def get_curved_arrow(self, start, end, angle):
        arrow = (
            CurvedArrow(
                start,
                end,
                angle=angle,
            )
            .set_color(REDUCIBLE_YELLOW)
            .set_stroke(width=4)
        )
        arrow.pop_tips()
        arrow.add_tip(
            tip_shape=ArrowTriangleFilledTip, tip_length=0.2, at_start=False
        )
        return arrow
    
    def get_marker(self, start, end, color=REDUCIBLE_YELLOW):
        line = Line(start, end).set_color(color)
        left_mark = Line(UP * SMALL_BUFF, DOWN * SMALL_BUFF).next_to(line, LEFT, buff=0).set_color(color)
        right_mark = Line(UP * SMALL_BUFF, DOWN * SMALL_BUFF).next_to(line, RIGHT, buff=0).set_color(color)

        return VGroup(left_mark, line, right_mark)

    def get_new_buffers(self, search_sequence_group, look_ahead_sequence_group, out_of_range_group, num_elements_to_fade):
        new_search_buffer = [search_sequence_group[i] for i in range(num_elements_to_fade, len(search_sequence_group))]
        for i in range(num_elements_to_fade):
            if i >= len(look_ahead_sequence_group):
                break
            new_search_buffer.append(look_ahead_sequence_group[i])

        new_search_buffer_group = VGroup(*new_search_buffer)

        new_look_ahead = [look_ahead_sequence_group[i] for i in range(num_elements_to_fade, len(look_ahead_sequence_group))]
        for i in range(num_elements_to_fade):
            if i >= len(out_of_range_group):
                break
            new_look_ahead.append(out_of_range_group[i])

        new_look_ahead_group = VGroup(*new_look_ahead)

        new_out_of_range_group = out_of_range_group[num_elements_to_fade:]

        return new_search_buffer_group, new_look_ahead_group, new_out_of_range_group

    def get_letter_mob(self, letter, side_length=1, color=REDUCIBLE_PURPLE):
        text = Text(str(letter), font='SF Mono', weight=MEDIUM).scale(0.6)
        square = Square(side_length=side_length).set_color(color)
        square.set_fill(color=color, opacity=1)
        return VGroup(square, text)

    def get_offset_length_mob(self, offset, length, scale_height):
        offset_mob = self.get_letter_mob(offset, color=REDUCIBLE_YELLOW)
        offset_mob[0].set_stroke(color=BLACK)
        offset_mob[1].set_color(BLACK)
        length_mob = self.get_letter_mob(length, color=REDUCIBLE_YELLOW)
        length_mob[0].set_stroke(color=BLACK)
        length_mob[1].set_color(BLACK)
        return VGroup(offset_mob, length_mob).arrange(RIGHT, buff=0).scale_to_fit_height(scale_height)

    def get_letter_sequence(self, string):
        sequence = [self.get_letter_mob(char) for char in string]
        return VGroup(*sequence).arrange(RIGHT)

    def get_transforms_text_to_sequence(self, string, text, sequence):
        transforms = []
        offset = 0
        for i in range(len(sequence)):
            if string[i] == ' ':
                transforms.append(FadeIn(sequence[i]))
                offset += 1
                continue
            char = text[i - offset]
            char_in_mob = sequence[i][1]
            transforms.append(TransformFromCopy(char, char_in_mob))
            transforms.append(FadeIn(sequence[i][0]))
        return transforms
    
    def get_search_and_look_ahead(self, string, sequence, search_buffer_indices, look_ahead_buffer_indices):
        search_sequence = [self.get_letter_mob('', color=REDUCIBLE_GREEN_LIGHTER) for _ in range(self.search_buffer_len - len(search_buffer_indices))]
        for index in search_buffer_indices:
            search_sequence.append(self.get_letter_mob(string[index], color=REDUCIBLE_GREEN_LIGHTER))

        search_sequence_group = VGroup(*search_sequence).arrange(RIGHT).scale_to_fit_height(sequence.height)
        look_ahead_sequence_group = VGroup(*[sequence[i].copy() for i in look_ahead_buffer_indices])
        out_of_range_group = VGroup(*[self.get_unindication(sequence[i]) for i in range(len(string)) if i not in search_buffer_indices + look_ahead_buffer_indices])
        return search_sequence_group, look_ahead_sequence_group, out_of_range_group

    def show_decoding(self, encoded_text, decoded_string):
        dec_text_label = Text("Decoded text", font='SF Mono', weight=MEDIUM).scale(0.6)
        decoded_text = Text(decoded_string, font='SF Mono', weight=MEDIUM).move_to(DOWN * 3)
        dec_text_label.next_to(decoded_text, UP).shift(UP * 0.2)
        self.play(
            Write(dec_text_label)
        )
        pointer = Arrow(ORIGIN, UP * 1.1, buff=SMALL_BUFF).set_color(REDUCIBLE_VIOLET)
        print('Actual encoding', self.actual_encoding)
        text_index = 0
        string_index = 0
        num_spaces = 0
        to_reset = None
        for pointer_index, encoding in enumerate(encoded_text):
            if pointer_index == 0:
                pointer.next_to(encoding, DOWN)
                self.play(
                    Write(pointer)
                )
            else:
                if not to_reset:
                    self.play(
                        pointer.animate.next_to(encoding, DOWN)
                    )
                else:
                    self.play(
                        pointer.animate.next_to(encoding, DOWN),
                        *to_reset
                    )

            print('String index', string_index, 'Text index', text_index)
            if self.is_offset_length_pair(encoding):
                offset, length = self.actual_encoding[pointer_index]
                print('Offset', offset, 'length', length)
                reset_animations = []
                temp_text_index = text_index
                for i in range(text_index - offset + num_spaces, text_index - offset + num_spaces + length):
                    self.play(
                        decoded_text[i].animate.set_color(REDUCIBLE_YELLOW)
                    )

                    reset_animations.append(decoded_text[i].animate.set_color(WHITE))

                    self.play(
                        TransformFromCopy(decoded_text[i], decoded_text[temp_text_index])
                    )
                    temp_text_index += 1

                to_reset = reset_animations

                string_index += length
                text_index += length

            else:
                if decoded_string[string_index] == ' ':
                    num_spaces += 1
                    string_index += 1
                    continue

                self.play(
                    TransformFromCopy(encoding[1], decoded_text[text_index])
                )
                text_index += 1
                string_index += 1

        return decoded_text


    def is_offset_length_pair(self, encoding):
        # little bit hacky but works for now
        return not isinstance(encoding[0], Square)

class LZSSImageExample(Scene):
    def construct(self):
        self.sw_length = 16
        self.look_ahead_len = 5
        self.search_buffer_len = 11
        
        rgb_img = ImageMobject("r")

        b_channel_f_mob, b_channel_f_text, blue_channel = self.introduce_image(rgb_img)
        
        self.display_encoding(blue_channel, b_channel_f_mob, b_channel_f_text)

    def introduce_image(self, rgb_img):
        blue_channel = rgb_img.get_pixel_array()[:, :, 2]
        b_channel_padded = self.get_channel_image(blue_channel, mode='B')
        blue_channel_img = PixelArray(b_channel_padded).scale(0.7)

        b_channel_pixel_text = self.get_pixel_values(blue_channel, blue_channel_img, mode='B')
        for text in b_channel_pixel_text:
            text.scale(1.4)
        

        self.play(
            FadeIn(blue_channel_img),
            FadeIn(b_channel_pixel_text)
        )
        self.wait()

        b_channel_flattened = self.reshape_channel(b_channel_padded)

        b_channel_f_mob = PixelArray(b_channel_flattened, buff=SMALL_BUFF*5, outline=False).scale(0.6).to_edge(LEFT)

        b_channel_f_mob.to_edge(LEFT)
        
        b_channel_f_mob_text = self.get_pixel_values(b_channel_flattened[:, :, 2], b_channel_f_mob, mode='B')

        b_transforms = self.get_flatten_transform(blue_channel_img, b_channel_f_mob, b_channel_pixel_text, b_channel_f_mob_text)

        self.play(
            *b_transforms
        )
        self.wait()

        return b_channel_f_mob, b_channel_f_mob_text, blue_channel

    def display_encoding(self, blue_channel, b_channel_f_mob, blue_channel_f_text):
        encoding = self.get_lzss_encoding(blue_channel)
        index = 0
        highlights = []
        arrows = []
        count = 0
        while index  < len(b_channel_f_mob):
            if len(encoding) == 0:
                break
            current = encoding.pop(0)
            if isinstance(current, int):
                index += 1
                continue
            else:
                offset, length = current
                highlight = get_glowing_surround_rect(b_channel_f_mob[index:index+length])
                highlights.append(highlight)
                if count %  2 == 0:
                    arrow = self.get_curved_arrow(highlight.get_bottom(), b_channel_f_mob[index - offset].get_bottom(), -TAU/4)
                else:
                    arrow = self.get_curved_arrow(highlight.get_top(), b_channel_f_mob[index - offset].get_top(), TAU/4)
                index += length
                count += 1
                arrows.append(arrow)

        self.play(
            LaggedStartMap(FadeIn, highlights),
            LaggedStartMap(Write, arrows),
            run_time=5
        )
        self.wait()

        all_arrows = VGroup(*arrows)
        all_highlights = VGroup(*highlights)
        left_shift = LEFT * 50
        self.play(
            all_arrows.animate.shift(left_shift),
            all_highlights.animate.shift(left_shift),
            b_channel_f_mob.animate.shift(left_shift),
            blue_channel_f_text.animate.shift(left_shift),
            run_time=10,
        )
        self.wait()

    def get_curved_arrow(self, start, end, angle):
        arrow = (
            CurvedArrow(
                start,
                end,
                angle=angle,
            )
            .set_color(REDUCIBLE_YELLOW)
            .set_stroke(width=4)
        )
        arrow.pop_tips()
        arrow.add_tip(
            tip_shape=ArrowTriangleFilledTip, tip_length=0.2, at_start=False
        )
        return arrow

    def lzss_helper(self, string, search, look_ahead):
        if len(look_ahead) == 0:
            return []
        
        largest_match = []
        look_ahead_index = 0
        first_char = string[look_ahead[look_ahead_index]]
        for index in search[::-1]:
            if string[index] == first_char:
                largest_match.append(index)
                break

        if len(largest_match) == 0:
            return []

        next_index = largest_match[0] + 1
        look_ahead_index += 1
        if look_ahead_index >= len(look_ahead):
            return largest_match
        while look_ahead_index < self.look_ahead_len:
            if look_ahead_index >= len(look_ahead):
                break
            next_char = string[look_ahead[look_ahead_index]]
            if next_char != string[next_index]:
                break
            else:
                largest_match.append(next_index)

            next_index += 1
            look_ahead_index += 1

        return largest_match

    def get_lzss_encoding(self, pixel_array, shorten=False):
        string_conversion = self.get_string_conversion(pixel_array)
        indices = list(range(len(string_conversion)))

        search_buffer = []
        look_ahead = indices[:self.look_ahead_len]
        start_index = 0
        end_index = self.look_ahead_len - 1

        actual_encoding = []
        # print('Here', current_text.get_center())
        while len(look_ahead) > 0:
            print('Search', search_buffer, 'Look Ahead', look_ahead)
            largest_match = self.lzss_helper(string_conversion, search_buffer, look_ahead)
            print('Largest match', largest_match)
            if len(largest_match) == 0:
                index = look_ahead[0]
                actual_encoding.append(ord(string_conversion[index]))
            else:
                offset = look_ahead[0] - largest_match[0]
                length = len(largest_match)
                if shorten and length == 1:
                    index = look_ahead[0]
                    actual_encoding.append(ord(string_conversion[index]))
                else:
                    actual_encoding.append((offset, length))

            if len(largest_match) <= 1:
                search_buffer.append(look_ahead.pop(0))
            else:
                for _ in range(len(largest_match)):
                    search_buffer.append(look_ahead.pop(0))

            while len(look_ahead) < self.look_ahead_len:
                end_index += 1
                if end_index >= len(string_conversion):
                    break
                else:
                    look_ahead.append(end_index)
            while len(search_buffer) > self.search_buffer_len:
                start_index += 1
                search_buffer.pop(0)

        return actual_encoding

    def get_string_conversion(self, pixel_array):
        flattened_arr = pixel_array.flatten()
        string = [chr(p) for p in flattened_arr]
        return string

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

    def get_pixel_values(self, channel, channel_mob, mode='R'):
        pixel_values_text = VGroup()
        for p_val, mob in zip(channel.flatten(), channel_mob):
            text = Text(str(int(p_val)), font="SF Mono", weight=MEDIUM).scale(0.25).move_to(mob.get_center())
            if mode == 'G' and p_val > 200:
                text.set_color(BLACK)
            pixel_values_text.add(text)

        return pixel_values_text

    def get_flatten_transform(self, original_mob, flattened_mob, original_text, flattened_text):
        transforms = []
        for i in range(original_mob.shape[0]):
            for j in range(original_mob.shape[1]):
                one_d_index = i * original_mob.shape[1] + j
                transforms.append(ReplacementTransform(original_mob[i, j], flattened_mob[0, one_d_index]))
                transforms.append(ReplacementTransform(original_text[one_d_index], flattened_text[one_d_index]))

        return transforms

    def reshape_channel(self, channel):
        return np.reshape(channel, (1, channel.shape[0] * channel.shape[1], channel.shape[2]))


class ShowLZSSExampleOnImage(LZSSImageExample):
    def construct(self):
        self.sw_length = 64
        self.look_ahead_len = 16
        self.search_buffer_len = 48
        self.show_example_image()

    def show_example_image(self):
        LOW, HIGH = 50, 200
        example_img_arr = np.array([
        [LOW, HIGH, LOW, HIGH, LOW, HIGH, LOW, HIGH],
        [HIGH, LOW, HIGH, LOW, HIGH, LOW, HIGH, LOW],
        [LOW, HIGH, LOW, HIGH, LOW, HIGH, LOW, HIGH],
        [HIGH, LOW, HIGH, LOW, HIGH, LOW, HIGH, LOW],
        [LOW, HIGH, LOW, HIGH, LOW, HIGH, LOW, HIGH],
        [HIGH, LOW, HIGH, LOW, HIGH, LOW, HIGH, LOW],
        [LOW, HIGH, LOW, HIGH, LOW, HIGH, LOW, HIGH],
        [HIGH, LOW, HIGH, LOW, HIGH, LOW, HIGH, LOW],
        ])

        pixel_arr = PixelArray(example_img_arr, include_numbers=True, color_mode="GRAY")
        pixel_arr.scale(0.7)
        self.play(
            FadeIn(pixel_arr)
        )
        self.wait()

        encoding = self.get_lzss_encoding(example_img_arr, shorten=True)
        print(len(encoding))

        encoding_mob = self.get_encoding_mob(encoding)
        encoding_mob.move_to(RIGHT * 3.5 + DOWN * 0)

        encoded_image_title = Text("Encoded Image", font="SF Mono").scale(0.7)
        encoded_image_title.next_to(encoding_mob, UP)

        centered_group = VGroup(pixel_arr.copy(), encoding_mob).arrange(RIGHT, buff=2)

        self.play(
            Transform(pixel_arr, centered_group[0]),
        )

        self.play(
            LaggedStartMap(FadeIn, encoding_mob),
            run_time=4
        )
        self.wait()

    def get_encoding_mob(self, encoding):
        mobjects = []
        height = 0.5
        for element in encoding:
            if isinstance(element, int):
                mobjects.append(PixelArray(np.array([[element]]), color_mode="GRAY", include_numbers=True).scale_to_fit_height(height))
            else:
                mobjects.append(self.get_offset_length_mob(element[0], element[1], height))

        return VGroup(*mobjects).arrange_in_grid(rows=8)

    def get_offset_length_mob(self, offset, length, scale_height):
        offset_mob = self.get_letter_mob(offset, color=REDUCIBLE_YELLOW)
        offset_mob[0].set_stroke(color=BLACK)
        offset_mob[1].set_color(BLACK)
        length_mob = self.get_letter_mob(length, color=REDUCIBLE_YELLOW)
        length_mob[0].set_stroke(color=BLACK)
        length_mob[1].set_color(BLACK)
        return VGroup(offset_mob, length_mob).arrange(RIGHT, buff=0).scale_to_fit_height(scale_height)

    def get_letter_mob(self, letter, side_length=1, color=REDUCIBLE_PURPLE):
        text = Text(str(letter), font='SF Mono', weight=MEDIUM).scale(0.6)
        square = Square(side_length=side_length).set_color(color)
        square.set_fill(color=color, opacity=1)
        return VGroup(square, text)

class GoBackToTextExample(Scene):
    def construct(self):
        self.introduce_lzss()

    def introduce_lzss(self):
        text_snippet = Tex(
            "The most used word in the English language \\\\", 
            "is ``the.'' The second most used word is ``be.''"
        ).scale(1)

        text_snippet[1].next_to(text_snippet[0], DOWN)

        self.play(
            Write(text_snippet),
            run_time=3
        )
        self.wait()

        self.color_code_text(text_snippet)

        self.show_back_references(text_snippet)

    def color_code_text(self, text_snippet):
        self.play(
            text_snippet[0][:3].animate.set_color(REDUCIBLE_GREEN_LIGHTER),
            text_snippet[0][3:15].animate.set_color(REDUCIBLE_YELLOW),
            text_snippet[0][17:20].animate.set_color(REDUCIBLE_VIOLET),
            text_snippet[1][:2].animate.set_color(ORANGE),
            text_snippet[1][4:7].animate.set_color(REDUCIBLE_VIOLET),
            text_snippet[1][10:13].animate.set_color(REDUCIBLE_GREEN_LIGHTER),
            text_snippet[1][19:31].animate.set_color(REDUCIBLE_YELLOW),
            text_snippet[1][-9:-7].animate.set_color(ORANGE),
        )
        self.wait()

    def show_back_references(self, text_snippet):
        bottom_left_corner = text_snippet[1][0].get_bottom() + DL * SMALL_BUFF * 7
        top_left_corner = bottom_left_corner + text_snippet.height * UP + UP * 0.5
        bottom_right_corner = UP * bottom_left_corner[1] + RIGHT * (text_snippet[1][-1].get_right()[0] + 0.8)
        top_right_corner = bottom_right_corner + UP * 2.5
        green_ref_corners = [
            text_snippet[1][10:13].get_bottom() + DOWN * SMALL_BUFF, 
            text_snippet[1][10:13].get_bottom()[0] * RIGHT + bottom_left_corner[1] * UP ,
            bottom_left_corner,
            bottom_left_corner + UP * (text_snippet[0][0].get_left()[1] - bottom_left_corner[1])
            ]
        
        green_ref = self.get_elbow_arrow(
            green_ref_corners,
            green_ref_corners[-1],
            text_snippet[0][0].get_left() + LEFT * SMALL_BUFF,
            ratio=MED_SMALL_BUFF
        ).set_color(REDUCIBLE_GREEN_LIGHTER)

        yellow_ref_corners = [
            text_snippet[1][19:31].get_bottom() + DOWN *SMALL_BUFF,
            text_snippet[1][19:31].get_bottom()[0] * RIGHT + bottom_right_corner[1] * UP,
            bottom_right_corner,
            top_right_corner,
            text_snippet[0][3:15].get_top()[0] * RIGHT + UP * top_right_corner[1],
        ]

        yellow_ref = self.get_elbow_arrow(
            yellow_ref_corners,
            yellow_ref_corners[-1],
            text_snippet[0][3:15].get_top() + UP * SMALL_BUFF,
            ratio=MED_SMALL_BUFF
        ).set_color(REDUCIBLE_YELLOW)

        violet_ref_corners = [
            text_snippet[1][4:7].get_bottom() + DOWN * SMALL_BUFF,
            text_snippet[1][4:7].get_bottom()[0] * RIGHT + UP * (bottom_left_corner[1] + 0.2),
            UP * (bottom_left_corner[1] + 0.2) + RIGHT  * (bottom_left_corner[0] - 0.7),
            top_left_corner + UL * 0.7,
            text_snippet[0][17:20].get_top()[0] * RIGHT + UP * (top_left_corner[1] + 0.7)
        ]

        violet_ref = self.get_elbow_arrow(
            violet_ref_corners,
            violet_ref_corners[-1],
            text_snippet[0][17:20].get_top() + UP * SMALL_BUFF,
            ratio=MED_SMALL_BUFF
        ).set_color(REDUCIBLE_VIOLET)

        violet_ref[-1].set_stroke(width=4)

        orange_ref_corners = [
            text_snippet[1][-9:-7].get_bottom() + DOWN * SMALL_BUFF,
            text_snippet[1][-9:-7].get_bottom() + DOWN * SMALL_BUFF * 4,
            text_snippet[1][:2].get_bottom() + DOWN * SMALL_BUFF * 4,
        ]

        orange_ref = self.get_elbow_arrow(
            orange_ref_corners,
            orange_ref_corners[-1],
            text_snippet[1][:2].get_bottom() + DOWN * SMALL_BUFF,
            ratio=MED_SMALL_BUFF
        ).set_color(ORANGE)
        orange_ref[-1].set_stroke(width=4)

        self.play(
            Write(green_ref),
            Write(yellow_ref),
            Write(violet_ref),
            Write(orange_ref),
            text_snippet[1][10:13].animate.set_fill(opacity=0.5),
            text_snippet[1][19:31].animate.set_fill(opacity=0.5),
            text_snippet[1][4:7].animate.set_fill(opacity=0.5),
            text_snippet[1][-9:-7].animate.set_fill(opacity=0.5),
            run_time=3
        )
        self.wait()

    def get_elbow_arrow(self, corner_points, arrow_start, arrow_end, ratio=SMALL_BUFF):
        path = VGroup()
        path.set_points_as_corners(*[corner_points])
        arrow = Arrow(arrow_start, arrow_end, buff=0, max_tip_length_to_length_ratio=ratio)
        path.add(arrow)
        return path

class GradientImageIntro(Scene):
    def construct(self):
        gradient_image, _  = self.get_gradient_image()

        scale = 0.7
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

        gradient_image.move_to(LEFT * 3.5)


        group_of_modules = VGroup(LZSS_mod, huffman_mod).arrange(DOWN, buff=2)

        group_of_modules.move_to(RIGHT * 2.5)

        trash = self.get_trash()
        trash.next_to(LZSS_mod, RIGHT).shift(RIGHT)

        trash_down = trash.copy().next_to(huffman_mod, RIGHT).shift(RIGHT)

        group_of_trash = VGroup(trash, trash_down)

        VGroup(gradient_image, group_of_modules, group_of_trash).arrange(RIGHT, buff=2)

        self.play(
            FadeIn(gradient_image)
        )
        self.wait()

        start_x = gradient_image.get_right()[0]
        start_y = LZSS_mod.get_left()[1]
        arrow_top = Arrow(
            RIGHT * start_x + UP * start_y, 
            LZSS_mod.get_left()
        ).set_color(REDUCIBLE_YELLOW)

        self.play(
            Write(arrow_top)
        )
        self.wait()

        self.play(
            FadeIn(group_of_modules[0])
        )
        self.wait()

        second_arrow_top = Arrow(
            LZSS_mod.get_right(),
            trash.get_left()
        ).set_color(REDUCIBLE_YELLOW)
        self.play(
            Write(second_arrow_top)
        )
        self.play(
            FadeIn(trash)
        )
        self.wait()

        arrow_bottom = Arrow(
            RIGHT * start_x + DOWN * start_y, 
            huffman_mod.get_left()
        ).set_color(REDUCIBLE_YELLOW)

        self.play(
            Write(arrow_bottom)
        )

        self.play(
            FadeIn(huffman_mod)
        )
        self.wait()
        
        second_arrow_bottom = Arrow(
            huffman_mod.get_right(),
            trash_down.get_left()
        ).set_color(REDUCIBLE_YELLOW)

        self.play(
            Write(second_arrow_bottom)
        )
        self.play(
            FadeIn(trash_down)
        )
        self.wait()

        self.play(
            FadeOut(group_of_modules),
            FadeOut(group_of_trash),
            FadeOut(arrow_top),
            FadeOut(arrow_bottom),
            FadeOut(second_arrow_top),
            FadeOut(second_arrow_bottom),
            gradient_image.animate.move_to(ORIGIN),
        )
        self.wait()

        question = Text("We should be able to compress this image, right?", font="CMU Serif").scale(0.8)
        question.next_to(gradient_image, DOWN * 2)
        self.play(
            Write(question)
        )
        self.wait()

        filtering_title = Text("Filtering", font="CMU Serif", weight=BOLD)
        filtering_title.move_to(UP * 3)

        self.play(
            Write(filtering_title)
        )
        self.wait()

        self.clear()

        scale = 0.8

        hard_problem_mod = Module(
            ["Hard","Problem"],
            text_weight=BOLD,
            fill_color=REDUCIBLE_GREEN_DARKER,
            stroke_color=REDUCIBLE_GREEN_LIGHTER,
        ).scale(scale)

        new_approach_mod = Module(
            ["New","Solution"],
            text_weight=BOLD,
            fill_color=REDUCIBLE_YELLOW_DARKER,
            stroke_color=REDUCIBLE_YELLOW,
        ).scale(scale)

        transform_prob_mod = Module(
            ["Transform", "Problem"],
            text_weight=BOLD,
        ).scale(scale)

        hard_problem_mod.move_to(UP * 2.2)

        new_approach_mod.move_to(LEFT * 3 + DOWN * 2.2)
        transform_prob_mod.move_to(RIGHT * 3 + DOWN * 2.2)

        arrow_left = Arrow(
            hard_problem_mod.get_bottom(),
            new_approach_mod.get_top()
        ).set_color(GRAY)

        arrow_right = Arrow(
            hard_problem_mod.get_bottom(),
            transform_prob_mod.get_top()
        ).set_color(GRAY)

        self.play(
            FadeIn(hard_problem_mod)
        )
        self.wait()

        self.play(
            Write(arrow_left)
        )

        self.play(
            FadeIn(new_approach_mod)
        )
        self.wait()

        self.play(
            Write(arrow_right)
        )
        self.play(
            FadeIn(transform_prob_mod)
        )
        self.wait()

        self.play(
            new_approach_mod.animate.fade(0.5),
            arrow_left.animate.fade(0.5)
        )
        self.wait()


    def get_trash(self):
        trash = SVGMobject("trash")
        trash[0].set_color(WHITE)
        return trash.scale(0.8)

    def get_gradient_image(self):
        diff = 4
        gradient_data = np.arange(0, 256, diff, dtype=np.uint8).reshape((8, 8))
        gradient_image = PixelArray(
            gradient_data, include_numbers=True, color_mode="GRAY"
        ).scale(0.6)
        return gradient_image, gradient_data

from manim import *
from functions import *
from classes import *
from reducible_colors import *

config["assets_dir"] = "assets"
np.random.seed(24)

class HuffmanCodeIntro(Scene):
    def construct(self):
        """
        TODO Saturday: 
        1. add arrows to show_encoding_diagram
        2. Get a good gray scale image example of huffman coding on 16x16 image
        3. Create ordered list of frequencies for each pixel
        4. perform animation of
        """

        pixel_array, pixel_array_mob = self.show_image()
        rgb_rep = self.show_rgb_split(pixel_array, pixel_array_mob)
        self.show_encoding_diagram(pixel_array_mob, rgb_rep)

        lossless_comp = Text("Lossless Compression", font="CMU Serif", weight=BOLD)
        down_arrow = MathTex(r"\Updownarrow").scale(1.2)
        redundancy = Text("Exploit Redundancy", font="CMU Serif", weight=BOLD)

        VGroup(lossless_comp, down_arrow, redundancy).arrange(DOWN)
        self.play(
            *[FadeOut(mob) for mob in self.mobjects],
            Write(lossless_comp)
        )
        self.play(
            Write(down_arrow)
        )

        self.play(
            Write(redundancy)
        )
        self.wait()

    def show_image(self):
        image = ImageMobject("r.png")
        pixel_array = image.get_pixel_array().astype(int)
        pixel_array_mob = PixelArray(pixel_array).scale(0.4).shift(UP * 2)
        self.play(
            FadeIn(pixel_array_mob)
        )
        self.wait()

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

        return VGroup(
            VGroup(pixel_array_mob_r, r_channel_pixel_text),
            VGroup(pixel_array_mob_g, g_channel_pixel_text),
            VGroup(pixel_array_mob_b, b_channel_pixel_text),
        )

    def show_encoding_diagram(self, pixel_array_mob, rgb_rep):
        three_d_rgb_rep = rgb_rep.copy()
        pixel_width, pixel_height = pixel_array_mob[0, 0].width, pixel_array_mob[0, 0].height
        scale = 0.6
        three_d_rgb_rep.scale(scale)
        three_d_rgb_rep[0].move_to(ORIGIN)
        offset_v = (pixel_height * UP + pixel_width * RIGHT) * scale
        three_d_rgb_rep[1].move_to(offset_v)
        three_d_rgb_rep[2].move_to(offset_v * 2)


        encoder_m = Module("Encoder", text_weight=BOLD)

        output_image = SVGMobject("empty_file.svg").set_stroke(
            WHITE, width=5, background=True
        )

        encoding_flow = VGroup(three_d_rgb_rep, encoder_m.scale(0.8), output_image.scale(0.9)).arrange(RIGHT, buff=2)

        encoding_flow.move_to(UP * 2)

        arr1 = Arrow(
            start=three_d_rgb_rep.get_right(),
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
            FadeOut(pixel_array_mob),
            Transform(rgb_rep[2], three_d_rgb_rep[2]),
            Transform(rgb_rep[1], three_d_rgb_rep[1]),
            Transform(rgb_rep[0], three_d_rgb_rep[0]),
            run_time=2
        )
        self.wait()

        self.play(
            Write(arr1)
        )
        self.wait()

        self.play(
            FadeIn(encoding_flow[1]),
        )
        self.wait()

        self.play(
            Write(arr2)
        )
        self.wait()

        self.play(
            FadeIn(encoding_flow[2])
        )
        self.wait()

        decoder_m = Module("Decoder", text_weight=BOLD).scale_to_fit_height(encoder_m.height)

        decoding_flow = VGroup(
            three_d_rgb_rep.copy(),
            decoder_m,
            output_image.copy(),
        ).arrange(RIGHT, buff=2)

        decoding_flow.move_to(DOWN * 2)

        arr3 = Arrow(
            output_image.get_bottom(),
            decoding_flow[2].get_top(),
            color=GRAY_B,
            stroke_width=3,
            buff=0.3,
            max_tip_length_to_length_ratio=0.08,
            max_stroke_width_to_length_ratio=2,
        )


        self.play(
            Write(arr3),
            TransformFromCopy(output_image, decoding_flow[2])
        )

        self.wait()

        arr4 = Arrow(
            decoding_flow[2].get_left(),
            decoder_m.get_right(),
            color=GRAY_B,
            stroke_width=3,
            buff=0.3,
            max_tip_length_to_length_ratio=0.08,
            max_stroke_width_to_length_ratio=2,
        )

        self.play(
            Write(arr4)
        )
        self.wait()

        self.play(
            FadeIn(decoder_m)
        )
        self.wait()

        arr5 = Arrow(
            start=decoder_m.get_left(),
            end=decoding_flow[0].get_right(),
            color=GRAY_B,
            stroke_width=3,
            buff=0.3,
            max_tip_length_to_length_ratio=0.08,
            max_stroke_width_to_length_ratio=2,
        )

        self.play(
            Write(arr5),
            TransformFromCopy(three_d_rgb_rep[2], decoding_flow[0][2]),
            TransformFromCopy(three_d_rgb_rep[1], decoding_flow[0][1]),
            TransformFromCopy(three_d_rgb_rep[0], decoding_flow[0][0]),
        )
        self.wait()

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

class HuffmanCodes(HuffmanCodeIntro):
    def construct(self):
        b_channel, b_channel_data = self.get_example_image()
        b_channel.scale(1.5).move_to(ORIGIN)

        self.play(
            FadeIn(b_channel)
        )
        self.wait()

        self.highlight_pixel_values(b_channel)


        self.shift_left = LEFT * 5
        pixel_data = self.get_pixel_stream(b_channel_data)
        mapping_to_pixel_data = self.group_text(pixel_data)
        nodes = self.get_nodes(mapping_to_pixel_data, b_channel)
        print(nodes)

        self.play(
            b_channel.animate.shift(RIGHT * 2),
            *[FadeIn(node.mob) for node in nodes]
        )
        self.wait()

        self.play(
            FadeOut(b_channel)
        )
        self.wait()

        # nodes_pq = self.sort_nodes(nodes)
        # tree_nodes, tree_edges = self.make_huffman_tree(nodes_pq)

        # self.show_encodings(tree_nodes, tree_edges)

        # self.explain_savings()

        # self.clear()

        # screen_rect = ScreenRectangle(height=6).shift(UP * 0.5)
        # self.play(
        #     Create(screen_rect)
        # )
        # self.wait()

        # mild_spoiler = Text("PNG uses Huffman Encoding in final stage of compression").scale(0.7).next_to(screen_rect, DOWN)

        # self.play(
        #     Write(mild_spoiler)
        # )
        # self.wait()

    def highlight_pixel_values(self, channel_mob):
        pixel_array_mob, pixel_text = channel_mob
        transforms = []
        for mob, t in zip(pixel_array_mob, pixel_text):
            if t.original_text != '251':
                transforms.append(mob.animate.set_fill(opacity=0.2))
                transforms.append(t.animate.set_fill(opacity=0.2))

        self.play(
            *transforms
        )
        self.wait()

        freq_obv = Text("40% of the image is pixel 251", font='SF Mono', weight=MEDIUM).scale(0.7)
        freq_obv.next_to(channel_mob, DOWN)

        self.play(
            FadeIn(freq_obv)
        )
        self.wait()

        huffman_codes = Text("Huffman Codes", font='CMU Serif', weight=BOLD).move_to(UP * 3.2)

        self.play(
            AddTextLetterByLetter(huffman_codes),
            run_time=2
        )
        self.wait()

        inverse_transforms = []
        for mob, t in zip(pixel_array_mob, pixel_text):
            if t.original_text != '251':
                inverse_transforms.append(mob.animate.set_fill(opacity=1))
                inverse_transforms.append(t.animate.set_fill(opacity=1))

        self.play(
            *inverse_transforms,
            FadeOut(freq_obv),
            FadeOut(huffman_codes)
        )
        self.wait()

    def explain_savings(self):
        observation = Title("Observations", match_underline_width_to_text=True)

        observation.move_to(ORIGIN).to_edge(LEFT * 2)

        bulleted_list = BulletedList(
            "Original image had 4 unique pixel values",
            "Standard representations: 2 bits/pixel",
            "Huffman codes: 1.89 bits/pixel (*)",
            "Also need to store Huffman tree for decoding",
            buff=MED_SMALL_BUFF
        ).scale(0.7)

        bulleted_list.next_to(observation, DOWN, aligned_edge=LEFT)

        self.play(
            Write(observation)
        )
        self.wait()

        for i, element in enumerate(bulleted_list):
            if i < len(bulleted_list) - 1:
                self.play(
                    FadeIn(element)
                )
                self.wait()

        calculation = MathTex(r" \frac{25 \cdot 1 + 21 \cdot 2 + 4 \cdot 3 + 14 \cdot 3}{64 \text{ pixels}} \approx 1.89 \text{ bits/pixel} \text{ (*)}").scale(0.6)

        calculation.to_edge(LEFT * 2).shift(DOWN * 3.2)

        self.add(calculation)
        self.wait()

        self.play(
            FadeIn(bulleted_list[-1])
        )
        self.wait()

    def show_encodings(self, tree_nodes, tree_edges):
        ZERO_COLOR = REDUCIBLE_YELLOW
        ONE_COLOR = REDUCIBLE_GREEN
        encodings = [
        [0],
        [1, 0],
        [1, 1, 0],
        [1, 1, 1],
        ]
        text = VGroup(*[
        Text("0", font='SF Mono', weight=MEDIUM),
        Text("10", font='SF Mono', weight=MEDIUM),
        Text("110", font='SF Mono', weight=MEDIUM),
        Text("111", font='SF Mono', weight=MEDIUM)
        ])

        text[0].next_to(tree_nodes[2], DOWN)
        text[1].next_to(tree_nodes[3], DOWN)
        text[2].next_to(tree_nodes[6], DOWN)
        text[3].next_to(tree_nodes[5], DOWN)

        transforms = []

        for enc, t in zip(encodings, text):
            t.scale(0.5)
            for i, elem in enumerate(enc):
                if elem == 0:
                    transforms.append(t[i].animate.set_color(ZERO_COLOR))
                else:
                    transforms.append(t[i].animate.set_color(ONE_COLOR))

        
        self.play(
            FadeIn(text)
        )
        self.wait()

        self.play(
            tree_edges[1].animate.set_color(ZERO_COLOR),
            tree_edges[2].animate.set_color(ZERO_COLOR),
            tree_edges[5].animate.set_color(ZERO_COLOR),
            tree_edges[0].animate.set_color(ONE_COLOR),
            tree_edges[3].animate.set_color(ONE_COLOR),
            tree_edges[4].animate.set_color(ONE_COLOR),
            *transforms
        )
        self.wait()

        right_shift = RIGHT * 2
        self.play(
            *[edge.animate.set_color(GRAY).shift(right_shift) for edge in tree_edges],
            *[node.animate.shift(right_shift) for node in tree_nodes],
            *[text.animate.set_color(WHITE).shift(right_shift)]
        )
        self.wait()

    def get_color_map(self, pixel_array_mob):
        color_map = {}
        mob, text = pixel_array_mob
        for square, t in zip(mob, text):
            # print(type(square))
            color_map[t.original_text] = square.get_color()

        return color_map 

    def get_example_image(self):
        pixel_array, pixel_array_mob = self.show_image(animate=False)
        r_channel, g_channel, b_channel = self.show_rgb_split(pixel_array, pixel_array_mob, animate=False)
        return g_channel, pixel_array[:, :, 1]


    def show_image(self, animate=True):
        image = ImageMobject("r.png")
        pixel_array = image.get_pixel_array().astype(int)
        pixel_array_mob = PixelArray(pixel_array).scale(0.4).shift(UP * 2)
        if animate:
            self.play(
                FadeIn(pixel_array_mob)
            )
            self.wait()

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

        return VGroup(
            VGroup(pixel_array_mob_r, r_channel_pixel_text),
            VGroup(pixel_array_mob_g, g_channel_pixel_text),
            VGroup(pixel_array_mob_b, b_channel_pixel_text),
        )

    def get_text_mobs(self):
        characters = ['A', 'D', 'B', 'A', 'D', 'E', 'D', 'B', 'B', 'D', 'D']
        # characters = ['A', 'B', 'D', 'B', 'B', 'A', 'C', 'B']
        # characters = ['A', 'B', 'R', 'A', 'C', 'A', 'D', 'A', 'B', 'R', 'A']
        text_mobjects = VGroup(*[Text(c, font='SF Mono', weight=MEDIUM) for c in characters])
        text_mobjects.arrange(RIGHT, buff=SMALL_BUFF * 0.5)
        return text_mobjects

    def get_pixel_stream(self, pixel_array):
        characters = []
        for i in range(pixel_array.shape[0]):
            for j in range(pixel_array.shape[1]):
                characters.append(str(int(pixel_array[i][j])))

        text_mobjects = VGroup(*[Text(c, font='SF Mono', weight=MEDIUM) for c in characters])
        text_mobjects.arrange(RIGHT, buff=SMALL_BUFF * 0.5)
        return text_mobjects

    def group_text(self, text_mobs):
        """
        Takes VGroup of TextMobjects, each with one character
        and groups them by similar characters. Returns a list
        of lists, with each list containing the same character
        """
        # maps characters to count
        mapping = {}
        # maps character to list of indices
        mapping_indices = {}
        for i, text in enumerate(text_mobs):
            if text.original_text not in mapping:
                mapping[text.original_text] = 1
                mapping_indices[text.original_text] = [i]
            else:
                mapping[text.original_text] += 1
                mapping_indices[text.original_text].append(i)

        positions = []
        SPACE_BETWEEN = 1.3
        for i in range(len(mapping)):
            position = self.shift_left + DOWN * i * SPACE_BETWEEN + (UP * ((len(mapping) / 2) - 0.5) * SPACE_BETWEEN)
            positions.append(position)
        mapping_to_text_mobs = {}
        for key in mapping:
            mapping_to_text_mobs[key] = []
            for _ in range(mapping[key]):
                mapping_to_text_mobs[key].append(Text(key, font='SF Mono', weight=MEDIUM))

            mapping_to_text_mobs[key] = VGroup(
                *mapping_to_text_mobs[key]
                ).arrange(RIGHT, buff=SMALL_BUFF * 0.5)

        for i, key in enumerate(mapping_to_text_mobs):
            mapping_to_text_mobs[key].move_to(positions[i])


        return mapping_to_text_mobs


    def get_nodes(self, mapping_to_text_mobs, pixel_array_mob):
        nodes = []
        color_map = self.get_color_map(pixel_array_mob)
        for key in mapping_to_text_mobs:
            node = Node(len(mapping_to_text_mobs[key]), key)
            if key in color_map:
                node.generate_mob(key_color=color_map[key]).scale(0.7)
            else:
                node.generate_mob().scale(0.7)
            node.mob.move_to(mapping_to_text_mobs[key].get_center())
            nodes.append(node)

        return nodes

    def sort_nodes(self, nodes, animate=True):
        sorted_nodes = sorted(nodes)
        index_mapping = self.get_sorting_transform_mapping(nodes, sorted_nodes)
        transforms = []
        for i in index_mapping:
            j = index_mapping[i]
            new_position = nodes[j].mob.get_center()
            new_mob = nodes[i].mob.copy().move_to(new_position)
            transform = Transform(nodes[i].mob, new_mob)
            transforms.append(transform)
        if animate:
            self.play(
                *transforms,
                run_time=2
            )
        self.wait()

        return sorted_nodes

    def get_sorting_transform_mapping(self, nodes, sorted_nodes):
        index_mapping = {}
        for i in range(len(nodes)):
            index_mapping[i] = sorted_nodes.index(nodes[i])
        return index_mapping

    
    def make_huffman_tree(self, heap, text_scale=0.6):
        # (key1, key2) -> Line object representing edge between 
        # node with key1 and node with key2
        nodes = []
        edges = []
        edge_dict = {}
        position_map = self.get_position_map_balanced()
        print(position_map)
        heap.sort()
        i = 0
        while len(heap) > 1:
            # print(heap)
            first = heap.pop(0)
            second = heap.pop(0)
            parent = Node(first.freq + second.freq, first.key + second.key)
            # print(first, second, parent)
            parent.generate_mob(text_scale=text_scale).scale(0.7)
            parent.connect_node(first, left=True)
            parent.connect_node(second, left=False)
            # this method mutates the heap
            self.animate_huffman_step(
                parent, first, second, position_map,
                nodes, edges, edge_dict, heap,
            )
            i += 1


        final_node = heap.pop(0)
        self.play(
            final_node.heap_mob.animate.move_to(final_node.mob.get_center()),
            final_node.heap_mob.animate.fade(0.9),
            run_time=2
        )
        self.remove(final_node.heap_mob)
        self.wait()
        

        unique_nodes = []
        for node in nodes:
            if True in [node.key == other_node.key for other_node in unique_nodes]:
                continue
            unique_nodes.append(node)

        return VGroup(VGroup(*[node.mob for node in unique_nodes]), VGroup(*edges))
    # We need a map of node keys to positions
    # This is a little bit hard coded right now
    # but only part of this that's hard coded
    def get_position_map_balanced(self):
        position_map = {
        '252': DOWN * 2.5 + RIGHT * 2,
        '147': DOWN * 2.5 + RIGHT * 4,
        '252147': DOWN * 0.7 + RIGHT * 3,
        '92': DOWN * 0.7, 
        '25214792': RIGHT * 1.5 + UP * 1.1,
        '251': LEFT * 1.5 + UP * 1.1, 
        '25125214792': UP * 2.9,
        }
        return position_map

    def animate_huffman_step(self, 
        parent, left, right, position_map,
        nodes, edges, edge_dict, heap):
        if len(left.key) <= 3 and len(right.key) <= 3:
            self.play(
                left.mob.animate.move_to(position_map[left.key]),
                right.mob.animate.move_to(position_map[right.key]),
                run_time=2 
            )
        elif len(left.key) > 3 and len(right.key) > 3:
            self.play(
                left.heap_mob.animate.move_to(position_map[left.key]),
                left.heap_mob.animate.fade(0.9),
                right.heap_mob.animate.move_to(position_map[right.key]),
                right.heap_mob.animate.fade(0.9),
                run_time=2 
            )
            self.remove(left.heap_mob)
            self.remove(right.heap_mob)
        elif len(left.key) > 3:
            self.play(
                left.heap_mob.animate.move_to(position_map[left.key]),
                left.heap_mob.animate.fade(0.9),
                right.mob.animate.move_to(position_map[right.key]),
            )
            self.remove(left.heap_mob)
        else:
            self.play(
                left.mob.animate.move_to(position_map[left.key]),
                right.heap_mob.animate.move_to(position_map[right.key]),
                right.heap_mob.animate.fade(0.9),
            )
            self.remove(right.heap_mob)

        self.wait()
        parent.mob.move_to(position_map[parent.key])
        left_edge = self.get_edge_mob(parent, left)
        right_edge = self.get_edge_mob(parent, right)
        self.play(
            Create(left_edge),
            Create(right_edge),
            GrowFromCenter(parent.mob),
        )
        self.wait()
        nodes.insert(0, left)
        nodes.insert(0, right)
        nodes.insert(0, parent)
        edges.insert(0, left_edge)
        edges.insert(0, right_edge)
        edge_dict[(parent.key, left.key)] = left_edge
        edge_dict[(parent.key, right.key)] = right_edge
        self.transform_heap(heap, parent)
        print(len(edges))

    def get_edge_mob(self, parent, child):
        start, end = parent.mob.get_center(), child.mob.get_center()
        if child.is_leaf:
            end = child.mob.get_top()
        unit_v = (end - start) / np.linalg.norm(end - start)
        start = start + unit_v * parent.mob[0].radius
        if child.is_leaf:
            end = end - unit_v * SMALL_BUFF * 3
        else:
            end = end - unit_v * child.mob[0].radius

        edge = Line(start, end).set_stroke(color=GRAY, width=6)
        return edge

    def transform_heap(self, heap, parent):
        original_heap = heap.copy()
        heap.append(parent)
        heap.sort()
        heap_group = VGroup(*[node.mob.copy() for node in heap]).arrange(DOWN).move_to(self.shift_left)
        transforms = []
        for node in original_heap:
            if node not in heap:
                continue
            if len(node.key) > 3:
                transforms.append(
                    Transform(heap[heap.index(node)].heap_mob, heap_group[heap.index(node)])
                )
            else: 
                transforms.append(
                    Transform(heap[heap.index(node)].mob, heap_group[heap.index(node)])
                )

        transforms.append(TransformFromCopy(parent.mob, heap_group[heap.index(parent)]))
        parent.heap_mob = heap_group[heap.index(parent)]
        self.play(
            *transforms,
            run_time=2
        )
        self.wait()

class IntroRLE(HuffmanCodes):
    def construct(self):
        self.intro_rle()

    def intro_rle(self):
        huffman_fact = Text("Huffman codes treat values independently", font='CMU Serif').scale(0.7)
        huffman_fact.move_to(UP * 3.5)
        self.play(
            Write(huffman_fact),
            run_time=2
        )
        self.wait()

        image_mob = ImageMobject("bw_rose").scale(0.4).move_to(LEFT * 3.5)
        pixel_array = image_mob.get_pixel_array()
        height_pixels, width_pixels = pixel_array.shape[0], pixel_array.shape[1]
        self.play(FadeIn(image_mob))
        self.wait()

        observation = Text("Images have spatial redundancy", font='CMU Serif').scale(0.5)
        observation[-17:].set_color(REDUCIBLE_YELLOW)

        observation.move_to(RIGHT * 3 + DOWN * 1)

        dot = Dot(color=REDUCIBLE_GREEN_LIGHTER)
        for i in range(5):
            start_row, start_col = np.random.randint(0, height_pixels), np.random.randint(0, width_pixels)
            dot.move_to(self.get_highlight_pos(start_row, start_col, image_mob))
            _, _, random_block = self.get_pixel_block(image_mob, start_row, start_col)
            random_row = random_block[:, :, 0][0]

            if i == 0:
                pixel_mob = self.make_row_of_pixels(random_row, height=0.6)
                pixel_mob.move_to(RIGHT * 3)
                self.add(dot, pixel_mob)
            else:
                pixel_mob.become(self.make_row_of_pixels(random_row, height=0.6).move_to(pixel_mob.get_center()))
            
            if i == 4:
                self.play(
                    Write(observation)
                )

            self.wait()

        self.clear()

        rle_title = Text("Run Length Encoding", font="CMU Serif", weight=BOLD)

        rle_title.move_to(UP * 3.5)

        self.play(
            Write(rle_title)
        )
        self.wait()

        b_channel, b_channel_data = self.get_example_image()
        b_channel.scale(1.5).move_to(LEFT * 2)

        self.play(
            FadeIn(b_channel)
        )
        self.wait()

        surround_rect_top = SurroundingRectangle(b_channel[0][1:6], color=REDUCIBLE_YELLOW, buff=0)
        surround_rect_mid = SurroundingRectangle(b_channel[0][19:22], color=REDUCIBLE_VIOLET, buff=0)
        surround_rect_bot = SurroundingRectangle(b_channel[0][41:45], color=REDUCIBLE_GREEN_LIGHTER, buff=0)
        indicies_to_indicate = list(range(1, 6)) + list(range(19, 22)) + list(range(41, 45))
        self.play(
            FadeIn(surround_rect_top),
            FadeIn(surround_rect_mid),
            FadeIn(surround_rect_bot),
            *[b_channel[0][i].animate.set_fill(opacity=0.2) for i in range(len(b_channel[0])) if i not in indicies_to_indicate],
            *[b_channel[1][i].animate.set_fill(opacity=0.2) for i in range(len(b_channel[0])) if i not in indicies_to_indicate
            ],
        )
        self.wait()

        top_run = self.make_pixel_run(VGroup(b_channel[0][1], b_channel[1][1]), 5, surround_rect_top.get_color())
        mid_run = self.make_pixel_run(VGroup(b_channel[0][19], b_channel[1][19]), 3,  surround_rect_mid.get_color())
        bot_run = self.make_pixel_run(VGroup(b_channel[0][41], b_channel[1][41]), 4, surround_rect_bot.get_color())
        
        run_group = VGroup(top_run, mid_run, bot_run).arrange(DOWN).move_to(RIGHT * 2.5)

        self.play(
            TransformFromCopy(VGroup(b_channel[0][1], b_channel[1][1]), top_run[0]),
            TransformFromCopy(surround_rect_top, top_run[1]),
            TransformFromCopy(VGroup(b_channel[0][19], b_channel[1][19]), mid_run[0]),
            TransformFromCopy(surround_rect_mid, mid_run[1]),
            TransformFromCopy(VGroup(b_channel[0][41], b_channel[1][41]), bot_run[0]),
            TransformFromCopy(surround_rect_bot, bot_run[1]),
        )
        self.wait()

        self.clear()
        self.transition_to_LZSS()
        self.introduce_lzss()

    def transition_to_LZSS(self):
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

        groups_of_two = []
        surround_rects = []
        animations = []

        for i in range(len(example_img_arr)):
            for j in range(0, len(example_img_arr[0]), 2):
                group = VGroup(pixel_arr[i, j], pixel_arr[i, j + 1])
                groups_of_two.append(group)
                position = DOWN * (i - 3.5) + RIGHT * (j - 3)
                
                if i % 2 == 0:
                    surround_rect = get_glowing_surround_rect(group, color=REDUCIBLE_YELLOW)
                else:
                    surround_rect = get_glowing_surround_rect(group, color=REDUCIBLE_VIOLET)
                
                surround_rect.move_to(position)
                surround_rects.append(surround_rect)
                animations.append(group.animate.move_to(position))

        self.play(
            *animations
        )
        self.play(
            *[FadeIn(surround_rect) for surround_rect in surround_rects]
        )
        self.wait()

        groups_of_two = VGroup(*groups_of_two)
        surround_rects = VGroup(*surround_rects)
        entire_group = VGroup(groups_of_two, surround_rects)
        shift_left = LEFT * 3
        self.play(
            entire_group.animate.scale(0.8).shift(shift_left)
        )
        self.wait()

        better_tools = Tex("Better options exist \\\\ in text compression").scale(0.9)
        better_tools.shift(RIGHT * 3.5 + UP * 0)

        self.play(
            FadeIn(better_tools)
        )
        self.wait()

        # zip_file = SVGMobject("zip_side.svg").next_to(better_tools, DOWN)

        # self.play(
        #     FadeIn(zip_file)
        # )
        # self.wait()

        self.clear()

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

        lempel_ziv = Text("Lempel-Ziv Schemes", font="CMU Serif", weight=BOLD)
        lempel_ziv.move_to(UP * 3.5)

        lempel_img = ImageMobject("Lempel").scale(1.2).move_to(LEFT * 2.5 + UP * 1.7)
        ziv_img = ImageMobject("Ziv").scale_to_fit_height(lempel_img.height).move_to(RIGHT * 2.5 + UP * 1.7)

        self.play(
            Write(lempel_ziv),
            text_snippet.animate.shift(DOWN * 2.5)
        )
        self.wait()

        abraham = Text("Abraham Lempel", font="CMU Serif").scale(0.7).next_to(lempel_img, DOWN)
        jacob = Text("Jacob Ziv", font="CMU Serif").scale(0.7).next_to(ziv_img, DOWN)

        self.play(
            FadeIn(ziv_img),
            FadeIn(lempel_img),
            Write(abraham),
            Write(jacob)
        )
        self.wait()

        lzss = Text("Lempel-Ziv-Storer-Szymanski (LZSS)", font="CMU Serif", weight=BOLD).scale(0.8)

        lzss.next_to(text_snippet, UP).shift(UP * 0.8)

        self.play(
            FadeIn(lzss)
        )
        self.wait()

        final_group = VGroup(lzss, text_snippet)
        self.play(
            final_group.animate.move_to(ORIGIN),
            FadeOut(lempel_img),
            FadeOut(ziv_img),
            FadeOut(abraham),
            FadeOut(jacob),
            FadeOut(lempel_ziv)
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




    def make_pixel_run(self, original_pixel, run_length, run_val_color):
        new_pixel = original_pixel.copy()
        run_rep = self.get_run_mob(run_length, color=run_val_color).scale_to_fit_height(new_pixel.height)
        return VGroup(new_pixel, run_rep).arrange(RIGHT, buff=0)

    def get_run_mob(self, run, side_length=1, color=REDUCIBLE_PURPLE):
        text = Text(str(run), font='SF Mono', weight=MEDIUM).scale(0.6)
        text.set_color(BLACK)
        square = Square(side_length=side_length).set_color(color)
        square.set_fill(color=color, opacity=1)
        return VGroup(square, text)

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

    def get_pixel_block(self, image_mob, start_row, start_col, block_size=8, height=2):
        pixel_array = image_mob.get_pixel_array()
        block = pixel_array[
            start_row : start_row + block_size, start_col : start_col + block_size
        ]

        block_image = self.get_image_mob(block, height=height)
        pixel_grid = self.get_pixel_grid(block_image, block_size)

        return block_image, pixel_grid, block

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

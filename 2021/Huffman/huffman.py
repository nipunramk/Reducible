from manimlib.imports import *
import random
np.random.seed(0)
import time

class Node:
	def __init__(self, freq, key=''):
		self.freq = freq
		self.key = key
		self.left = None
		self.right = None
	def __repr__(self):
		return 'Node(freq={0}, key={1})'.format(self.freq, self.key)
	def __lt__(self, other):
		return self.freq < other.freq
	def __gt__(self, other):
		return self.freq > other.freq

	def generate_mob(self, text_scale=0.8, radius=0.5, is_leaf=False, key_scale=None):
		# geometry is first, then text mobject
		# characters will be handled with separate geometry
		# nodes that are parents will have keys that concatenate the childs
		# since huffman trees are full binary trees, guaranteed to be not length 1
		if len(self.key) != 1 and not is_leaf:
			self.is_leaf = False
			freq = TextMobject(str(self.freq)).scale(text_scale + 0.4)
			node = Circle(radius=radius).set_color(YELLOW)
			freq.move_to(node.get_center())
			self.mob = VGroup(node, freq)
			self.heap_mob = self.mob
			return self.mob
		# two rectangles, top is frequency, bottom is key
		self.is_leaf = True
		freq_box = Rectangle(height=0.5, width=1).set_color(YELLOW)
		freq_interior = TextMobject(str(self.freq)).scale(text_scale)
		freq_interior.move_to(freq_box.get_center())
		freq = VGroup(freq_box, freq_interior)
		key_box = Rectangle(height=1, width=1).set_color(BLUE)
		key_interior = TextMobject(self.key).scale(text_scale + SMALL_BUFF * 5)
		if key_scale:
			key_interior.scale(key_scale)
		key_interior.move_to(key_box.get_center())
		key = VGroup(key_box, key_interior)
		self.mob = VGroup(freq, key).arrange(DOWN, buff=0)
		return self.mob

	def connect_node(self, child, left=True):
		if left:
			self.left = child
		else:
			self.right = child

class IntroStory(Scene):
	def construct(self):
		huffman_img = ImageMobject("huffman_img")
		fano_img = ImageMobject("fano")

		huffman_img.scale(2).move_to(LEFT * 3)
		fano_img.scale(2).move_to(RIGHT * 3)

		huffman_label = TextMobject("David Huffman").scale(0.8)
		fano_label = TextMobject("Robert Fano").scale(0.8)

		huffman_label.next_to(huffman_img, DOWN)
		fano_label.next_to(fano_img, DOWN)

		self.play(
			FadeIn(huffman_img),
			FadeIn(huffman_label)
		)
		self.wait()

		self.play(
			FadeIn(fano_img),
			FadeIn(fano_label)
		)
		self.wait()

		self.play(
			fano_img.move_to, huffman_img.get_center(),
			FadeOut(huffman_img),
			FadeOut(huffman_label),
			FadeOut(fano_label),
			run_time=2
		)
		self.wait()

		exam_svg = ExamSVG()
		exam_svg.move_to(UP * 2 + RIGHT * 3)

		paper_img = ImageMobject("essay").scale(1.5)
		paper_img.move_to(DOWN * 2 + RIGHT * 3)
		

		top_arrow = Arrow(fano_img.get_right(), exam_svg.get_left(), buf=MED_SMALL_BUFF)
		bottom_arrow = Arrow(fano_img.get_right(), paper_img.get_left(), buf=MED_SMALL_BUFF)

		top_arrow.set_color(YELLOW)
		bottom_arrow.set_color(YELLOW)

		self.play(
			Write(top_arrow),
			Write(bottom_arrow)
		)
		self.wait()

		self.play(
			Write(exam_svg)
		)
		self.wait()

		self.play(
			FadeIn(paper_img)
		)
		self.wait()

		cross = CrossX().scale(0.8).next_to(exam_svg, RIGHT)
		cross.scale(0.8)

		check = CheckMark().scale(0.3).next_to(paper_img, RIGHT)

		self.play(
			Write(cross)
		)

		self.wait()

		huffman_img.scale(0.3)
		check.shift(UP)
		huffman_img.next_to(check, DOWN)
		self.play(
			FadeIn(check),
			FadeInFrom(huffman_img, direction=DOWN)
		)

		self.wait()

		self.clear()

		problem = TextMobject(
			"Given a collection of letters, numbers, or other symbols, find" + "\\\\", 
			"the most efficient method to represent them using a binary code."
		).scale(0.8)

		self.play(
			FadeIn(problem),
		)
		self.wait()

		title = Title("Data Compression")

		self.play(
			Write(title),
			problem.next_to, title, DOWN,
			problem.scale, 0.7 / 0.8,
			run_time=2
		)

		self.wait()

		text = TextMobject("T", "e", "x", "t", "!").scale(0.8)
		image = ImageMobject("seattle").scale(0.9)
		video = ScreenRectangle().scale(0.5)


		text.move_to(LEFT * 3 + UP * 1.2)
		image.next_to(text, DOWN * 2)
		video.next_to(image, DOWN * 2)

		self.play(
			LaggedStart(*[FadeIn(mob) for mob in [text, image, video]]),
			run_time=2
		)
		self.wait()

		set_of_characters = ["T", "e", "x", "t", "!"]

		set_group_chars = TextMobject(*set_of_characters).arrange(RIGHT, buff=1.7)
		set_group_chars.scale(0.8)
		set_group_chars.move_to(RIGHT * 3 + UP * 1.2)
		
		square = Square(side_length=0.1).move_to(image.get_center())
		square.set_color(YELLOW)
		self.play(
			ShowCreation(square)
		)
		self.wait()

		right_square = Square(side_length=2.1).set_color(YELLOW)
		right_square.move_to(RIGHT * 3 + UP * image.get_center()[1])
		self.play(
			*[TransformFromCopy(text[i], set_group_chars[i]) for i in range(len(text))],
			TransformFromCopy(square, right_square),
			run_time=2
		)

		green_rect = Rectangle(height=0.7, width=2.1).set_stroke(color=GREEN_SCREEN, opacity=0.3)
		green_rect.set_fill(color=GREEN_SCREEN, opacity=0.3)
		green_rect.move_to(right_square.get_center())

		red_rect = green_rect.copy().set_stroke(color=BRIGHT_RED, opacity=0.3).set_fill(color=BRIGHT_RED, opacity=0.3)
		red_rect.next_to(green_rect, UP, buff=0)
		
		blue_rect = green_rect.copy().set_stroke(color=BLUE, opacity=0.3).set_fill(color=BLUE, opacity=0.3)
		blue_rect.next_to(green_rect, DOWN, buff=0)

		R, G, B = TextMobject("R").scale(0.8), TextMobject("G").scale(0.8), TextMobject("B").scale(0.8)
		R.move_to(red_rect.get_center())
		G.move_to(green_rect.get_center())
		B.move_to(blue_rect.get_center())

		self.play(
			FadeIn(red_rect),
			FadeIn(green_rect),
			FadeIn(blue_rect),
			Write(R),
			Write(B),
			Write(G),
		)
		self.wait()

		


		range_vals = [TextMobject("[0, 255]").scale(0.8).move_to(channel.get_center()) for channel in [R, G, B]]

		self.play(
			*[ReplacementTransform(channel, range_val) for channel, range_val in zip([R, G, B], range_vals)]
		)
		self.wait()

		binary_groups_chars = [get_binary_group(8).scale(0.5).move_to(c.get_center()) for c in set_group_chars]
		binary_groups_chars = VGroup(*binary_groups_chars).arrange(RIGHT, buff=SMALL_BUFF * 1.5).move_to(set_group_chars.get_center())

		binary_groups = [get_binary_group(8).scale(0.7).move_to(c.get_center()) for c in range_vals]

		self.play(
			*[ReplacementTransform(range_val, binary_group) for range_val, binary_group in zip(range_vals, binary_groups)],
			*[ReplacementTransform(c, binary) for c, binary in zip(set_group_chars, binary_groups_chars)],
			run_time=2
		)
		self.wait()

		

		top_brace = Brace(binary_groups_chars, DOWN)
		text_bits = TextMobject(r"40 bits (8 bits $/$ character)").scale(0.8)
		text_bits.move_to(binary_groups_chars.get_center())

		image_bits = TextMobject(r"(24 bits $/$ pixel) $\cross$ \#pixels")
		image_bits.move_to(right_square.get_center())
		image_bits.scale(0.8).shift(UP * SMALL_BUFF * 2)

		self.play(
			ReplacementTransform(binary_groups_chars, text_bits),
			ReplacementTransform(right_square, image_bits),
			*[FadeOut(b) for b in binary_groups + [red_rect, green_rect, blue_rect, square]],
			run_time=2
		)

		self.wait()

	
		video_bits = VGroup(image_bits.copy(), TextMobject(r"$\cross$ \#frames").scale(0.8)).arrange(RIGHT, buff=SMALL_BUFF * 1.5)
		video_bits.move_to(RIGHT * 3 + UP * video.get_center()[1])

		self.play(
			TransformFromCopy(image_bits, video_bits[0])
		)
		self.play(
			Write(video_bits[1])
		)
		self.wait()
		center = image_bits.get_center()
		group_bits_text = VGroup(text_bits.copy(), image_bits.copy(), video_bits.copy()).arrange(DOWN)
		group_bits_text.move_to(center)

		self.play(
			Transform(text_bits, group_bits_text[0]),
			Transform(image_bits, group_bits_text[1]),
			Transform(video_bits, group_bits_text[2]),
			run_time=2
		)

		surround_rect = SurroundingRectangle(group_bits_text)
		self.play(
			ShowCreation(surround_rect)
		)

		self.wait()

		question = TextMobject(
			"What is the fewest number of bits we" + "\\\\",
			"can use to represent each piece of data?"
		).scale(0.8)

		question.next_to(surround_rect, DOWN)
		self.play(
			FadeIn(question)
		)
		self.wait()

		unsolved_problem = TextMobject("Unsolved Problem!")
		unsolved_problem.next_to(question, DOWN * 2)

		self.play(
			GrowFromCenter(unsolved_problem)
		)
		self.wait()

class Transition(Scene):
	def construct(self):
		title = Title("Huffman Encoding Algorithm", scale_factor=1.2)
		self.play(
			Write(title)
		)
		self.wait()

		aka = TextMobject("(also known as Huffman Codes, Huffman Coding, Huffman's algorithm)")
		aka.scale(0.7)
		aka.next_to(title, DOWN)
		self.add(aka)
		self.wait()

		rediscover_question = TextMobject(
			"Can we ``rediscover''" + "\\\\",
			"Huffman codes?"
		)



		screen_rect = ScreenRectangle()

		aligned_group = VGroup(rediscover_question, screen_rect).arrange(RIGHT, buff=1)
		aligned_group.move_to(DOWN * 0.5)
		self.play(
			FadeIn(rediscover_question),
			ShowCreation(screen_rect),
		)

		self.wait()

		self.play(
			FadeOut(rediscover_question)
		)

		info_theory = TextMobject("Information Theory")
		down_arrow = TexMobject(r"\Downarrow")
		data_compression = TextMobject("Data Compression")
		huffman_codes = TextMobject("Huffman Encoding")

		flow_of_ideas = VGroup(info_theory, down_arrow, data_compression, down_arrow.copy(), huffman_codes).arrange(DOWN)

		flow_of_ideas.move_to(rediscover_question.get_center())


		for i in range(len(flow_of_ideas)):
			if i % 2 == 0:
				self.play(
					FadeIn(flow_of_ideas[i])
				)
			else:
				self.play(
					Write(flow_of_ideas[i])
				)
			self.wait()



class ExamSVG(SVGMobject):
    CONFIG = {
        "file_name": "exam",
        "fill_opacity": 1,
        "stroke_width": 0,
        "propagate_style_to_family": True
    }

    def __init__(self, **kwargs):
    	#TODO
    	# Honestly we need to figure out a rewrite for this
    	# Too lazy to do better for now, but let's see if newer
    	# versions of manim have this broken functionality fixed
        SVGMobject.__init__(self, **kwargs)
        self[0].set_color(WHITE)
        self[1:5].set_color(BRIGHT_RED)
        self[5].set_color(BRIGHT_RED)
        self[6].set_color(BRIGHT_RED)
        self[7].set_color(BRIGHT_RED)
        self[8].set_color(GRAY)
        self[9].set_color(GRAY)
        self[10].set_color(GRAY)
        self[11].set_color(GRAY)
        self[12].set_color(SHADOW_BLUE)
        self[13].set_color(STORM_GRAY)

        self[14].set_color(GOLDEN_ROD)
        self[15].set_color(GOLDEN_TANOI)
        self[16].set_color(PENCIL_SHADE_TOP)
        self[17].set_color(WHEAT)
        self[18].set_color(PENCIL_SHADE_TOP)
        self[19].set_color(WHEAT)

class PaperSVG(SVGMobject):
    CONFIG = {
        "file_name": "Essay",
        "fill_opacity": 1,
        "stroke_width": 0,
        "propagate_style_to_family": True
    }

    def __init__(self, **kwargs):
    	SVGMobject.__init__(self, **kwargs)
    	self.flip()
    	self.rotate(PI)
    	self.scale(1.5)
    	self[:1].set_stroke(WHITE, width=4)
    	self[1:].set_stroke(WHITE, width=2)

class CheckMark(SVGMobject):
    CONFIG = {
        "file_name": "checkmark",
        "fill_opacity": 1,
        "stroke_width": 0,
        "width": 4,
        "propagate_style_to_family": True
    }

    def __init__(self, **kwargs):
        SVGMobject.__init__(self, **kwargs)
        background_circle = Circle(radius=self.width / 2 - SMALL_BUFF * 2)
        background_circle.set_color(WHITE)
        background_circle.set_fill(color=WHITE, opacity=1)
        
        self.add_to_back(background_circle)
        self[1].set_fill(color=CHECKMARK_GREEN)
        self.set_width(self.width)
        self.center()

class CrossX(VMobject):
    CONFIG = {
        "fill_opacity": 1,
        "stroke_width": 0,
        "width": 4,
        "propagate_style_to_family": True
    }
    def __init__(self, **kwargs):
        VMobject.__init__(self, **kwargs)
        background_circle = Circle()
        background_circle.set_color(CROSS_RED)
        background_circle.set_fill(CROSS_RED, opacity=1)
        self.add(background_circle)
        rounded_rect1 = RoundedRectangle(width=4, height=1, corner_radius=0.25).scale(0.3).rotate(PI / 4)
        rounded_rect1.set_color(WHITE).set_fill(WHITE, opacity=1)
        rounded_rect2 = RoundedRectangle(width=4, height=1, corner_radius=0.25).scale(0.3).rotate(-PI / 4)
        rounded_rect2.set_color(WHITE).set_fill(WHITE, opacity=1)
        self.add(rounded_rect1)
        self.add(rounded_rect2)

class HuffmanCodes(Scene):
	CONFIG = {
		"shift_left": LEFT * 5
	}
	def construct(self):
		text = self.get_text_mobs()
		text.move_to(UP * 3.5)
		self.play(
			Write(text)
		)
		self.wait()
		brace = Brace(text, direction=DOWN)
		self.play(
			GrowFromCenter(brace)
		)
		how_to_compress = TextMobject("How is Huffman encoding applied to compress text in general?").scale(0.8)
		how_to_compress.next_to(brace, DOWN)

		self.play(
			FadeIn(how_to_compress)
		)
		self.wait()

		self.play(
			FadeOut(how_to_compress),
			FadeOut(brace)
		)
		self.wait()

		mapping_to_text_mobs = self.group_text(text)
		nodes = self.transform_to_nodes(mapping_to_text_mobs)
		nodes_pq = self.sort_nodes(nodes)
		self.make_huffman_tree(nodes_pq)

	def get_text_mobs(self):
		characters = ['A', 'D', 'B', 'A', 'D', 'E', 'D', 'B', 'B', 'D', 'D']
		# characters = ['A', 'B', 'D', 'B', 'B', 'A', 'C', 'B']
		# characters = ['A', 'B', 'R', 'A', 'C', 'A', 'D', 'A', 'B', 'R', 'A']
		text_mobjects = VGroup(*[TextMobject(c) for c in characters])
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
			if text.tex_string not in mapping:
				mapping[text.tex_string] = 1
				mapping_indices[text.tex_string] = [i]
			else:
				mapping[text.tex_string] += 1
				mapping_indices[text.tex_string].append(i)

		positions = []
		SPACE_BETWEEN = 1.3
		for i in range(len(mapping)):
			position = self.shift_left + DOWN * i * SPACE_BETWEEN + (UP * ((len(mapping) / 2) - 0.5) * SPACE_BETWEEN)
			positions.append(position)
		mapping_to_text_mobs = {}
		for key in mapping:
			mapping_to_text_mobs[key] = []
			for _ in range(mapping[key]):
				mapping_to_text_mobs[key].append(TextMobject(key))

			mapping_to_text_mobs[key] = VGroup(
				*mapping_to_text_mobs[key]
				).arrange(RIGHT, buff=SMALL_BUFF * 0.5)

		for i, key in enumerate(mapping_to_text_mobs):
			mapping_to_text_mobs[key].move_to(positions[i])


		return self.transform_text_to_groups(text_mobs, mapping_indices, mapping_to_text_mobs)

	def transform_text_to_groups(self, text_mobs, mapping_indices, mapping_to_text_mobs):
		transforms = []
		for key in mapping_indices:
			for i, index in enumerate(mapping_indices[key]):
				transform = TransformFromCopy(text_mobs[index], mapping_to_text_mobs[key][i])
				transforms.append(transform)

		self.play(
			*transforms,
			run_time=3,
		)
		self.wait()
		return mapping_to_text_mobs

	def transform_to_nodes(self, mapping_to_text_mobs):
		transforms = []
		nodes = []
		for key in mapping_to_text_mobs:
			node = Node(len(mapping_to_text_mobs[key]), key)
			node.generate_mob().scale(0.7)
			node.mob.move_to(mapping_to_text_mobs[key].get_center())
			nodes.append(node)
			transform = ReplacementTransform(mapping_to_text_mobs[key], node.mob)
			transforms.append(transform)
		
		self.play(
			*transforms,
			run_time=2
		)
		self.wait()

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

	
	def make_huffman_tree(self, heap, text_scale=0.8):
		# (key1, key2) -> Line object representing edge between 
		# node with key1 and node with key2
		nodes = []
		edges = []
		edge_dict = {}
		position_map = self.get_position_map_balanced()
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
			final_node.heap_mob.move_to, final_node.mob.get_center(),
			final_node.heap_mob.fade, 0.9,
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
		'E': DOWN * 3 + RIGHT * 2,
		'A': DOWN * 3 + RIGHT * 4,
		'EA': DOWN * 1.2 + RIGHT * 3,
		'B': DOWN * 1.2, 
		'BEA': RIGHT * 1.5 + UP * 0.6,
		'D': LEFT * 1.5 + UP * 0.6, 
		'DBEA': UP * 2.4,
		}
		return position_map

	def animate_huffman_step(self, 
		parent, left, right, position_map,
		nodes, edges, edge_dict, heap):
		if len(left.key) == 1 and len(right.key) == 1:
			self.play(
				left.mob.move_to, position_map[left.key],
				right.mob.move_to, position_map[right.key],
				run_time=2 
			)
		elif len(left.key) != 1 and len(right.key) != 1:
			self.play(
				left.heap_mob.move_to, position_map[left.key],
				left.heap_mob.fade, 0.9,
				right.heap_mob.move_to, position_map[right.key],
				right.heap_mob.fade, 0.9,
				run_time=2 
			)
			self.remove(left.heap_mob)
			self.remove(right.heap_mob)
		elif len(left.key) != 1:
			self.play(
				left.heap_mob.move_to, position_map[left.key],
				left.heap_mob.fade, 0.9,
				right.mob.move_to, position_map[right.key],
			)
			self.remove(left.heap_mob)
		else:
			self.play(
				left.mob.move_to, position_map[left.key],
				right.heap_mob.move_to, position_map[right.key],
				right.heap_mob.fade, 0.9,
			)
			self.remove(right.heap_mob)

		self.wait()
		parent.mob.move_to(position_map[parent.key])
		left_edge = self.get_edge_mob(parent, left)
		right_edge = self.get_edge_mob(parent, right)
		self.play(
			ShowCreation(left_edge),
			ShowCreation(right_edge),
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
			if len(node.key) != 1:
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

class HuffmanCodesThumbnail(HuffmanCodes):
	def construct(self):
		binary_background = self.get_binary_matrix_map()
		for bit in binary_background:
			p = np.random.uniform()
			if p < 0.1:
				bit.set_fill(color=BLUE, opacity=0.7)
			elif p < 0.25:
				bit.set_fill(color=BLUE, opacity=0.5)
			elif p < 0.4:
				bit.set_fill(color=BLUE, opacity=0.3)


		self.add(binary_background)
		self.wait()

		text = self.get_text_mobs()
		text.move_to(UP * 3.5)
		self.play(
			Write(text)
		)
		self.wait()
		brace = Brace(text, direction=DOWN)
		self.play(
			GrowFromCenter(brace)
		)
		how_to_compress = TextMobject("How is Huffman encoding applied to compress text in general?").scale(0.8)
		how_to_compress.next_to(brace, DOWN)

		self.play(
			FadeIn(how_to_compress)
		)
		self.wait()

		self.play(
			FadeOut(how_to_compress),
			FadeOut(brace)
		)
		self.wait()

		mapping_to_text_mobs = self.group_text(text)
		nodes = self.transform_to_nodes(mapping_to_text_mobs)
		nodes_pq = self.sort_nodes(nodes)
		
		huffman_tree = self.make_huffman_tree(nodes_pq)
		self.remove(text)

		huffman_tree.scale(1.1).move_to(RIGHT * 3)
		self.add_foreground_mobjects(huffman_tree)
		surround_rect = SurroundingRectangle(huffman_tree, buff=SMALL_BUFF*2).set_color(WHITE)
		surround_rect.set_fill(color=BLACK, opacity=1)
		self.add(surround_rect)
		self.wait()

		info_theory = TextMobject("Information Theory").scale(1.2)
		arrow = TexMobject(r"\Updownarrow").scale(1.2)
		huffman_codes = TextMobject("Huffman Codes").scale(1.2)

		text = VGroup(info_theory, arrow, huffman_codes).arrange(DOWN)
		text.move_to(LEFT * 3.5)
		self.add_foreground_mobjects(text)
		surround_rect_text = SurroundingRectangle(text, buff=SMALL_BUFF*2).set_color(WHITE)
		surround_rect_text.set_fill(color=BLACK, opacity=1)
		self.add(surround_rect_text)

		text_group = VGroup(text, surround_rect_text)
		tree_group = VGroup(huffman_tree, surround_rect)

		VGroup(tree_group, text_group).arrange(RIGHT * 2).move_to(ORIGIN)

		self.wait()



		# self.play(
		# 	nodes_group.scale, 1.2,
		# 	edges_group.scale, 1.2,
		# )
		# self.wait()
	# We need a map of node keys to positions
	# This is a little bit hard coded right now
	# but only part of this that's hard coded
	def get_position_map_balanced(self):
		position_map = {
		'E': DOWN * 3 + RIGHT * 2,
		'A': DOWN * 3 + RIGHT * 4,
		'EA': DOWN * 1.2 + RIGHT * 3,
		'B': DOWN * 1.2, 
		'BEA': RIGHT * 1.5 + UP * 0.6,
		'D': LEFT * 1.5 + UP * 0.6, 
		'DBEA': UP * 2.4,
		}
		return position_map

	def get_binary_matrix_map(self):
		UPPER_LEFT_CORNER = LEFT * 8 + UP * 5
		LOWER_RIGHT_CORNER = RIGHT * 8 + DOWN * 5
		print(UPPER_LEFT_CORNER, LOWER_RIGHT_CORNER)
		SPACE_BETWEEN_HORIZ = 0.3
		SPACE_BETWEEN_VERT = 0.5
		matrix = VGroup()
		for x in np.arange(UPPER_LEFT_CORNER[0], LOWER_RIGHT_CORNER[0], SPACE_BETWEEN_HORIZ):
			for y in np.arange(LOWER_RIGHT_CORNER[1], UPPER_LEFT_CORNER[1], SPACE_BETWEEN_VERT):
				bit = Integer(np.random.choice([0, 1])).move_to(RIGHT * x + UP * y)
				bit.set_fill(color=WHITE, opacity=0.2)
				matrix.add(bit)

		return matrix



class HuffmanCodesProbDist(HuffmanCodes):
	def construct(self):
		dist = {
		'A': 0.17,
		'B': 0.35,
		'C': 0.17,
		'D': 0.15,
		'E': 0.16,
		}

		nodes = self.generate_example_dist(dist)
		nodes_mobs = VGroup(*[node.mob for node in nodes]).arrange(DOWN).move_to(self.shift_left)

		self.play(
			*[FadeIn(mob) for mob in nodes_mobs],
			run_time=2
		)
		self.wait()

		nodes_pq = self.sort_nodes(nodes)
		self.make_huffman_tree(nodes_pq, text_scale=0.5)


	def get_position_map_balanced(self):
		position_map = {
		'D': DOWN * 3 + RIGHT * 1,
		'E': DOWN * 3 + RIGHT * 3,
		'A': DOWN * 3 + LEFT * 3,
		'C': DOWN * 3 + LEFT * 1,
		'DE': DOWN * 1 + RIGHT * 2,
		'AC': DOWN * 1 + LEFT * 2,
		'DEAC': UP * 1 + LEFT * 0,
		'B': UP * 1 + RIGHT * 4,
		'BDEAC': UP * 3 + RIGHT * 2,
		}
		return position_map

	def generate_example_dist(self, dist):
		nodes = []
		for key in dist:
			node = Node(dist[key], key)
			node.generate_mob().scale(0.7)
			nodes.append(node)
		return nodes

class HuffmanImplementationPIP(Scene):
	def construct(self):
		screen_rect = ScreenRectangle(height=3.5)
		screen_rect.to_edge(RIGHT)
		self.play(
			ShowCreation(screen_rect)
		)
		self.wait()

class HuffmanImplementation(Scene):
	def construct(self):
		code = self.get_build_huffman_function_code()
		code.move_to(LEFT * 2.8 + DOWN * 1.3)

		node_class_code = self.get_node_class()
		node_class_code.next_to(code, UP * 0.7, aligned_edge=LEFT)

		self.play(
			Write(code[0])
		)
		self.wait()

		self.play(
			FadeIn(code[1]),
			FadeIn(code[2]),
		)

		self.wait()

		self.play(
			Write(code[3])
		)
		self.wait()

		self.play(
			FadeIn(node_class_code[1:5])
		)
		self.wait()

		self.play(
			FadeIn(node_class_code[5:])
		)
		self.wait()

		self.play(
			FadeIn(node_class_code[0]),
			FadeIn(code[4]),
			FadeIn(code[5]),
		)
		self.wait()

		self.play(
			FadeIn(code[6:10]),
		)
		self.wait()

		self.play(
			FadeIn(code[10:])
		)
		self.wait()




	def get_node_class(self):
		code_scale = 0.7

		code = []

		import_statement = TextMobject("from heapq import heapify, heappop, heappush")
		import_statement[0][:4].set_color(MONOKAI_PINK)
		import_statement[0][9:15].set_color(MONOKAI_PINK)
		import_statement.scale(code_scale)
		import_statement.to_edge(LEFT)
		code.append(import_statement)

		class_statement = TextMobject(r"class Node:")
		class_statement[0][:5].set_color(MONOKAI_BLUE)
		class_statement[0][5:-1].set_color(MONOKAI_GREEN)
		class_statement.scale(code_scale)
		class_statement.next_to(import_statement, DOWN * 0.5)
		class_statement.to_edge(LEFT)
		code.append(class_statement)

		def_statement = TextMobject(r"def \underline{\hspace{0.4cm}} init \underline{\hspace{0.4cm}} (self, ch, freq, left$=$None, right$=$None):")
		def_statement[0][:9].set_color(MONOKAI_BLUE)
		def_statement[0][10:14].set_color(MONOKAI_ORANGE)
		def_statement[0][15:17].set_color(MONOKAI_ORANGE)
		def_statement[0][18:22].set_color(MONOKAI_ORANGE)
		def_statement[0][23:27].set_color(MONOKAI_ORANGE)
		def_statement[0][27].set_color(MONOKAI_PINK)
		def_statement[0][28:32].set_color(MONOKAI_PURPLE)
		def_statement[0][33:38].set_color(MONOKAI_ORANGE)
		def_statement[0][38].set_color(MONOKAI_PINK)
		def_statement[0][39:43].set_color(MONOKAI_PURPLE)

		def_statement.scale(code_scale)
		def_statement.next_to(class_statement, DOWN * 0.5)
		def_statement.to_edge(LEFT * 2)
		code.append(def_statement)

		line_1 = TextMobject(r"self.ch, self.freq $=$ ch, freq")
		line_1.scale(code_scale)
		line_1.next_to(def_statement, DOWN * 0.5)
		line_1.to_edge(LEFT * 3)
		line_1[0][17].set_color(MONOKAI_PINK)
		code.append(line_1)

		line_2 = TextMobject(r"self.left, self.right $=$ left, right")
		line_2.scale(code_scale)
		line_2.next_to(line_1, DOWN * 0.5)
		line_2.to_edge(LEFT * 3)
		line_2[0][-11].set_color(MONOKAI_PINK)
		code.append(line_2)

		line_3 = TextMobject(r"def \underline{\hspace{0.4cm}} lt \underline{\hspace{0.4cm}} (self, other):")
		line_3.scale(code_scale)
		line_3.next_to(line_2, DOWN * 0.5)
		line_3.to_edge(LEFT * 2)
		line_3[0][:7].set_color(MONOKAI_BLUE)
		line_3[0][8:12].set_color(MONOKAI_ORANGE)
		line_3[0][13:18].set_color(MONOKAI_ORANGE)
		code.append(line_3)

		line_4 = TextMobject(r"return self.freq $<$ other.freq")
		line_4.scale(code_scale)
		line_4.next_to(line_3, DOWN * 0.5)
		line_4.to_edge(LEFT * 3)
		line_4[0][:6].set_color(MONOKAI_PINK)
		line_4[0][:6].set_color(MONOKAI_PINK)
		line_4[0][-11].set_color(MONOKAI_PINK)
		code.append(line_4)


		code = VGroup(*code)
		code.scale(0.9)
		code.to_edge(LEFT * 3)
		return code 

	def get_build_huffman_function_code(self):
		code_scale = 0.7

		code = []

		def_statement = TextMobject(r"def getHuffmanTree(text):")
		def_statement[0][:3].set_color(MONOKAI_BLUE)
		def_statement[0][3:17].set_color(MONOKAI_GREEN)
		def_statement[0][18:22].set_color(MONOKAI_ORANGE)
		def_statement.scale(code_scale)
		def_statement.to_edge(LEFT)
		code.append(def_statement)

		line_1 = TextMobject(r"if len(text) $==$ 0:")
		line_1.scale(code_scale)
		line_1.next_to(def_statement, DOWN * 0.5)
		line_1.to_edge(LEFT * 2)
		line_1[0][:2].set_color(MONOKAI_PINK)
		line_1[0][2:5].set_color(MONOKAI_BLUE)
		line_1[0][11:13].set_color(MONOKAI_PINK)
		line_1[0][13].set_color(MONOKAI_PURPLE)
		code.append(line_1)

		line_2 = TextMobject(r"return")
		line_2.scale(code_scale)
		line_2.next_to(line_1, DOWN * 0.5)
		line_2.to_edge(LEFT * 3)
		line_2.set_color(MONOKAI_PINK)
		code.append(line_2)

		line_3 = TextMobject(r"freq $=$ \{ch: text.count(ch) for ch in set(text)\}")
		line_3.scale(code_scale)
		line_3.next_to(line_2, DOWN * 0.5)
		line_3.to_edge(LEFT * 2)
		line_3[0][4].set_color(MONOKAI_PINK)
		line_3[0][14:19].set_color(MONOKAI_BLUE)
		line_3[0][23:26].set_color(MONOKAI_PINK)
		line_3[0][28:30].set_color(MONOKAI_PINK)
		line_3[0][30:33].set_color(MONOKAI_BLUE)
		code.append(line_3)

		line_4 = TextMobject(r"pq $=$ [Node(k, v) for k, v in freq.items()]")
		line_4.scale(code_scale)
		line_4.next_to(line_3, DOWN * 0.5)
		line_4.to_edge(LEFT * 2)
		line_4[0][2].set_color(MONOKAI_PINK)
		line_4[0][4:8].set_color(MONOKAI_BLUE)
		line_4[0][13:16].set_color(MONOKAI_PINK)
		line_4[0][19:21].set_color(MONOKAI_PINK)
		line_4[0][26:31].set_color(MONOKAI_BLUE)
		code.append(line_4)

		line_5 = TextMobject(r"heapify(pq)")
		line_5[0][:7].set_color(MONOKAI_BLUE)
		line_5.scale(code_scale)
		line_5.next_to(line_4, DOWN * 0.5)
		line_5.to_edge(LEFT * 2)
		code.append(line_5)

		line_6 = TextMobject(r"while len(pq) $>$ 1:")
		line_6.scale(code_scale)
		line_6.next_to(line_5, DOWN * 0.5)
		line_6.to_edge(LEFT * 2)
		line_6[0][:5].set_color(MONOKAI_PINK)
		line_6[0][5:8].set_color(MONOKAI_BLUE)
		line_6[0][-3].set_color(MONOKAI_PINK)
		line_6[0][-2].set_color(MONOKAI_PURPLE)
		code.append(line_6)

		line_7 = TextMobject(r"left, right $=$ heappop(pq), heappop(pq)")
		line_7[0][10].set_color(MONOKAI_PINK)
		line_7[0][11:18].set_color(MONOKAI_BLUE)
		line_7[0][23:30].set_color(MONOKAI_BLUE)
		line_7.scale(code_scale)
		line_7.next_to(line_6, DOWN * 0.5)
		line_7.to_edge(LEFT * 3)
		code.append(line_7)

		line_8 = TextMobject(r"newFreq $=$ left.freq $+$ right.freq")
		line_8[0][7].set_color(MONOKAI_PINK)
		line_8[0][17].set_color(MONOKAI_PINK)
		line_8.scale(code_scale)
		line_8.next_to(line_7, DOWN * 0.5)
		line_8.to_edge(LEFT * 3)
		code.append(line_8)

		line_9 = TextMobject(r"heappush(pq, Node(None, newFreq, left, right))")
		line_9[0][:8].set_color(MONOKAI_BLUE)
		line_9[0][12:16].set_color(MONOKAI_BLUE)
		line_9[0][17:21].set_color(MONOKAI_PURPLE)
		line_9.scale(code_scale)
		line_9.next_to(line_8, DOWN * 0.5)
		line_9.to_edge(LEFT * 3)
		code.append(line_9)

		line_10 = TextMobject(r"root $=$ pq[0]")
		line_10[0][4].set_color(MONOKAI_PINK)
		line_10[0][-2].set_color(MONOKAI_PURPLE)
		line_10.scale(code_scale)
		line_10.next_to(line_9, DOWN * 0.5)
		line_10.to_edge(LEFT * 2)
		code.append(line_10)

		line_11 = TextMobject(r"return root")
		line_11.scale(code_scale)
		line_11.next_to(line_10, DOWN * 0.5)
		line_11.to_edge(LEFT * 2)
		line_11[0][:6].set_color(MONOKAI_PINK)
		code.append(line_11)

		code = VGroup(*code)
		code.scale(0.9)
		code.to_edge(LEFT * 3)
		return code 

class HuffmanConclusion(Scene):
	def construct(self):
		title = Title("Huffman Encoding", scale_factor=1.2)
		self.play(
			Write(title)
		)
		self.wait()

		screen_rect = ScreenRectangle(height=3.5)
		screen_rect.to_edge(RIGHT).shift(DOWN * 0.5)
		self.play(
			ShowCreation(screen_rect),
		)
		self.wait()

		requirements = VGroup(
			TextMobject(r"1. Single symbol $\rightarrow$ unique binary code").scale(0.7),
			TextMobject(r"2. Source message $=$ received message").scale(0.7),
			TextMobject(r"3. Unique decodability").scale(0.7),
			TextMobject(r"4. Minimize $L$").scale(0.7),
		).arrange(DOWN)
		for elem in requirements:
			elem.to_edge(LEFT)


		optimal = Title("Optimal Encoding", match_underline_width_to_text=True)

		group = VGroup(optimal, requirements).arrange(DOWN, buff=MED_SMALL_BUFF)
		group.next_to(screen_rect, LEFT * 2).shift(UP * 0.3)
		self.play(
			Write(optimal)
		)

		self.wait()

		self.play(
			FadeIn(requirements)
		)
		self.wait()

		obvious = TextMobject("Seems almost obvious in hindsight ...").scale(0.9)
		obvious.move_to(DOWN * 3.2)
		self.play(
			Write(obvious)
		)
		self.wait()

		cross_obvious = Cross(obvious[0][11:18]).set_color(RED)
		self.play(
			Write(cross_obvious)
		)
		self.wait()

		shannon_img = ImageMobject("shannon").scale(1.5)
		fano_img = ImageMobject("fano").scale(1.5)
		self.play(
			FadeOut(group)
		)
		self.wait()
		fano_img.next_to(screen_rect, LEFT * 5)
		shannon_img.next_to(fano_img, LEFT).shift(LEFT)

		self.play(
			FadeIn(shannon_img),
			FadeIn(fano_img)
		)
		self.wait()

		shannon_fano_coding = Title("Shannon-Fano Coding", match_underline_width_to_text=True)
		shannon_fano_coding.move_to(optimal.get_center())
		self.play(
			FadeOut(shannon_img),
			FadeOut(fano_img),
			Write(shannon_fano_coding),
			FadeOut(obvious),
			FadeOut(cross_obvious)
		)
		self.wait()

		top_down = TextMobject("Top-down perspective").scale(0.8).next_to(shannon_fano_coding, DOWN)
		top_down[0][:8].set_color(YELLOW)
		self.play(
			FadeIn(top_down)
		)
		self.wait()

		huffman_coding = Title("Huffman Coding", match_underline_width_to_text=True)
		huffman_coding.next_to(top_down, DOWN * 2)
		bottom_up = TextMobject("Bottom-up perspective").scale(0.8).next_to(huffman_coding, DOWN)
		bottom_up[0][:9].set_color(YELLOW)

		self.play(
			Write(huffman_coding)
		)
		self.wait()

		self.play(
			GrowFromCenter(bottom_up)
		)
		self.wait()

		progression = VGroup(
			TextMobject("Information Theory"),
			TexMobject(r"\Downarrow"),
			TextMobject("Shannon-Fano Coding"),
			TexMobject(r"\Downarrow"),
			TextMobject("Huffman Coding"),
		).arrange(DOWN)

		progression.move_to(LEFT * 3.5 + DOWN * 0.5)

		self.play(
			FadeOut(shannon_fano_coding),
			FadeOut(huffman_coding),
			FadeOut(top_down),
			FadeOut(bottom_up),
			Write(progression[0])
		)
		self.wait()
		for i in range(1, len(progression)):
			if i % 2 == 1:
				self.play(
					Write(progression[i])
				)
			else:
				self.play(
					FadeIn(progression[i])
				)
			self.wait()

		surround_rect = SurroundingRectangle(progression, buff=SMALL_BUFF)
		self.play(
			ShowCreation(surround_rect)
		)
		self.wait()

		description = TextMobject("An underappreciated story").next_to(title, DOWN)
		description[0][2:-5].set_color(YELLOW)
		self.play(
			Write(description)
		)
		self.wait()
		


class ShannonFano(HuffmanCodes):
	def construct(self):
		title = Title("Shannon-Fano Coding", scale_factor=1.2)
		title.move_to(UP * 3.5)
		self.play(
			Write(title)
		)
		self.wait()

		definition = TextMobject(
			"Compression algorithm that approximates idea of splitting symbols" + "\\\\", 
			"into equally likely groups until every symbol has unique encoding"
		).scale(0.8).next_to(title, DOWN)

		self.play(
			FadeIn(definition)
		)
		self.wait()

		self.play(
			FadeOut(definition),
			FadeOut(title),
		)
		self.wait()

		dist = {
		'A': 0.17,
		'B': 0.35,
		'C': 0.17,
		'D': 0.15,
		'E': 0.16,
		}
		self.position_map = self.get_position_map()
		self.dashed_lines = {}
		self.edge_dict = {}
		nodes = self.generate_example_dist(dist)
		self.show_nodes([nodes])
		sorted_nodes = self.sort_nodes(nodes)

		# array representation of tree
		# 0th index is 0
		# all other indices are list of nodes that will later be transformed to single nodes
		# with this representation, index i is a node, i * 2 is left node, i * 2 + 1 is right node
		tree = [0] * (len(self.position_map) + 1)
		tree[1] = sorted_nodes
		self.animate_shannon_fano(sorted_nodes, tree, 1)
		tree_keys = [self.get_key(nodes) for nodes in tree[1:]]
		animation_groups = self.make_splits_into_nodes(tree)
		self.play(
			LaggedStart(*animation_groups),
			run_time=2
		)
		self.wait()
		order = sorted(self.edge_dict.keys(), key=lambda x: x[0])
		transforms = []
		for i, key in enumerate(order):
			if i % 2 == 0:
				 color = BRIGHT_RED
			else:
				color = GREEN_SCREEN

			transforms.append(ApplyMethod(self.edge_dict[key].set_color, color))
		self.play(
			LaggedStart(*transforms),
			run_time=3
		)
		self.wait()
		all_encodings = [[0, 1], [1, 0], [1, 1], [0, 0, 0], [0, 0, 1]]
		binary_encoding_mobs = [get_binary_encoding(encoding) for encoding in all_encodings]
		keys = ['A', 'C', 'B', 'D', 'E']
		for i, key in enumerate(keys):
			binary_encoding_mobs[i].scale(0.8).next_to(self.get_nodes(tree, key)[0].mob, DOWN)

		self.play(
			*[Write(encoding) for encoding in binary_encoding_mobs]
		)
		self.wait()
		# some nodes of tree are lists with one node
		for i in range(1, len(tree)):
			if isinstance(tree[i], list):
				tree[i] = tree[i][0]
				
		tree_nodes = VGroup(*[tree[i].mob for i in range(1, len(tree))])
		tree_edges = VGroup(*[self.edge_dict[key] for key in order])
		encoding_group = VGroup(*binary_encoding_mobs)

		tree_group = VGroup(tree_nodes, tree_edges, encoding_group)
		self.highlight_unique_paths(order)

		transformed_encoding_group = tree_group.copy().scale(0.7).shift(LEFT * 2)
		self.play(
			Transform(tree_group, transformed_encoding_group),
			run_time=2
		)
		self.wait()

		self.show_decoding_of_message(tree, order, encoding_group)

		self.clear()

		self.motivate_huffman(tree_group, tree)

	

	def show_decoding_of_message(self, tree, order, encoding_group):
		message = self.make_component("Message", color=RED)
		decoder = self.make_component("Decoder", color=BLUE)
		message.move_to(RIGHT * 4 + UP * 1.5)
		decoder.move_to(RIGHT * 4 + DOWN * 1.5)

		arrow = Arrow(message.get_bottom(), decoder.get_top())
		arrow.set_color(GRAY)

		encoding = [1, 1, 0, 0, 1, 0, 0, 0]
		encoded_message = get_binary_encoding(encoding)
		encoded_message.next_to(message, UP)
		self.play(
			Write(message),
			Write(encoded_message)
		)
		self.wait()
		self.play(
			Write(arrow)
		)
		self.play(
			Write(decoder)
		)
		self.wait()

		decoded_message = TextMobject("BED").next_to(decoder, DOWN)

		leaf_indices = [5, 6, 7, 8, 9]

		edge_key_decoding_order = self.get_edge_key_decoding_order(encoding, leaf_indices)
		highlighted_mobs = self.get_highlight_mobs(order)
		highlighted_mobs_in_scene = []
		decoded_message_index = 0
		for i, key in enumerate(edge_key_decoding_order):
			u, v = key
			to_highlight = highlighted_mobs[order.index(key)]
			self.play(
				encoded_message[i].set_color, to_highlight.get_color(),
				ShowCreation(to_highlight),
			)
			highlighted_mobs_in_scene.append(to_highlight)
			if v in leaf_indices:
				self.play(
					Indicate(tree[v].mob, color=to_highlight.get_color()),
					Indicate(encoding_group[v - leaf_indices[0]], color=to_highlight.get_color()),
					Write(decoded_message[0][decoded_message_index])
				)
				self.remove(*highlighted_mobs_in_scene)
				highlighted_mobs_in_scene = []
				decoded_message_index += 1
		self.wait()

		self.play(
			FadeOut(encoded_message),
			FadeOut(arrow),
			FadeOut(message),
			FadeOut(decoder),
			FadeOut(decoded_message),
		)

		self.explain_prefix_free_property()

	def explain_prefix_free_property(self):
		prefix_free_codes = TextMobject("Prefix-free codes")
		prefix_free_codes.move_to(RIGHT * 4 + UP * 3.2)
		self.play(
			Write(prefix_free_codes)
		)
		all_encodings = [[0, 0, 0], [0, 0, 1], [0, 1], [1, 0], [1, 1]]
		set_of_encodings = TextMobject(r"$\{ 000, 001, 01, 10, 11 \}$")
		set_of_encodings.scale(0.8).next_to(prefix_free_codes, DOWN)
		self.play(
			FadeIn(set_of_encodings)
		)
		self.wait()

		prop_1 = TextMobject(
			"No code is a prefix (initial segment)" + "\\\\",  
			"of another"
			).scale(0.7)
		prop_1.next_to(set_of_encodings, DOWN)
		self.play(
			Write(prop_1)
		)
		self.wait()

		prop_2 = TextMobject(
			"Can be represented with a full binary" + "\\\\",
			"tree (each node has 0 or 2 children)"
		).scale(0.7).next_to(prop_1, DOWN)

		self.play(
			FadeIn(prop_2)
		)
		self.wait()

		set_of_non_prefix_free_encodings = TextMobject(r"$\{ 000, 001, 00, 10, 1 \}$").scale(0.8)
		set_of_non_prefix_free_encodings[0][1:3].set_color(RED)
		set_of_non_prefix_free_encodings[0][5:7].set_color(RED)
		set_of_non_prefix_free_encodings[0][9:11].set_color(RED)
		set_of_non_prefix_free_encodings[0][12].set_color(ORANGE)
		set_of_non_prefix_free_encodings[0][15].set_color(ORANGE)


		set_of_non_prefix_free_encodings.next_to(prop_2, DOWN * 2)
		self.play(
			FadeIn(set_of_non_prefix_free_encodings)
		)
		self.wait()
		brace = Brace(set_of_non_prefix_free_encodings, direction=DOWN)
		not_prefix_free = TextMobject("NOT prefix-free").next_to(brace, DOWN)
		not_prefix_free[0][:3].set_color(RED)
		self.play(
			GrowFromCenter(brace),
			Write(not_prefix_free)
		)

		self.wait()

		self.play(
			FadeOut(prefix_free_codes),
			FadeOut(set_of_non_prefix_free_encodings),
			FadeOut(set_of_encodings),
			FadeOut(prop_1),
			FadeOut(prop_2),
			FadeOut(brace),
			FadeOut(not_prefix_free)
		)
		self.wait()

		shannon_fano_optimal = Title(
			"Shannon-Fano Encoding", 
			match_underline_width_to_text=True
		).move_to(RIGHT * 4 + UP * 2.5)
		not_always_optimal = TextMobject("Not always optimal!").scale(0.8).next_to(shannon_fano_optimal, DOWN)
		self.play(
			Write(shannon_fano_optimal)
		)
		self.wait()

		self.play(
			Write(not_always_optimal)
		)
		self.wait()

		L_def_1 = TextMobject(r"$L = \sum\limits_{i=1}^{n} p_i \cdot$ (encoding length of $i^{\text{th}}$ symbol)").scale(0.8)
		L_def_1.move_to(DOWN * 2.5)
		self.play(
			FadeIn(L_def_1)
		)
		self.wait()

		L_def_2 = TextMobject(r"$L = \sum\limits_{i=1}^{n} p_i \cdot$ (depth of $i^{\text{th}}$ symbol in tree)").scale(0.8)
		L_def_2.next_to(L_def_1, DOWN)
		self.play(
			FadeIn(L_def_2)
		)
		self.wait()

		L_expanded = TextMobject(r"$L = 0.15 \cdot 3 + 0.16 \cdot 3 + 0.17 \cdot 2 + 0.17 \cdot 2 + 0.35 \cdot 2$")
		L_expanded.scale(0.8).move_to(DOWN * 3)
		self.remove(L_def_1, L_def_2)
		self.play(
			Write(L_expanded),
			run_time=2
		)
		self.wait()

		L_final_shannon = TextMobject(r"$L = 2.31$ bits")
		L_final_shannon.move_to(L_expanded.get_center())
		self.play(
			ReplacementTransform(L_expanded, L_final_shannon),
			run_time=2
		)
		self.wait()

		self.play(
			L_final_shannon.next_to, not_always_optimal, DOWN,
			run_time=2,
		)
		self.wait()

		huffman_encoding_dict = {
		'A (0.17)': [0, 0, 0],
		'B (0.35)': [1],
		'C (0.17)': [0, 0, 1],
		'D (0.15)': [0, 1, 0],
		'E (0.16)': [0, 1, 1],
		}

		huff_encoding, _ = self.get_example_encoding(huffman_encoding_dict)
		huff_encoding.next_to(L_final_shannon, DOWN * 2)
		self.play(
			FadeIn(huff_encoding)
		)
		self.wait()

		optimal_L_expanded = TextMobject(r"$L^* = 0.17 \cdot 3 + 0.35 \cdot 1 + 0.17 \cdot 3 + 0.15 \cdot 3 + 0.16 \cdot 3$ ")
		optimal_L_expanded.scale(0.8).move_to(DOWN * 3)
		self.play(
			FadeIn(optimal_L_expanded)
		)
		self.wait()
		optimal_L = TextMobject(r"$L^* = 2.30$ bits").move_to(optimal_L_expanded.get_center())
		self.play(
			ReplacementTransform(optimal_L_expanded, optimal_L),
			run_time=2
		)
		self.wait()

		self.play(
			optimal_L.next_to, huff_encoding, DOWN,
			run_time=2
		)

		self.wait()

	def motivate_huffman(self, tree_group, tree):
		requirements = VGroup(
			TextMobject(r"1. Single symbol $\rightarrow$ unique binary code").scale(0.7),
			TextMobject(r"2. Source message $=$ received message").scale(0.7),
			TextMobject(r"3. Unique decodability").scale(0.7),
			TextMobject(r"4. Minimize $L$").scale(0.7),
		).arrange(DOWN)
		for elem in requirements:
			elem.to_edge(LEFT)
		requirements.move_to(RIGHT * 4 + DOWN * 3)
		surround_rect = SurroundingRectangle(requirements)
		corners = [surround_rect.get_vertices()[i] for i in [3, 0, 1]]
		elbow = VGroup()
		elbow.set_points_as_corners(*[corners])
		self.play(
			ShowCreation(elbow)
		)
		self.play(
			*[Write(requirements[i]) for i in range(3)]
		)

		tree_group.scale(1 / 0.7).move_to(ORIGIN)
		self.play(
			FadeIn(tree_group)
		)
		self.wait()

		self.play(
			requirements[0][0][:2].set_color, GREEN_SCREEN,
			requirements[1][0][:2].set_color, GREEN_SCREEN,
			requirements[2][0][:2].set_color, GREEN_SCREEN
		)

		self.wait()

		requirements[3][0][:2].set_color(RED)

		self.play(
			Write(requirements[3])
		)
		self.wait()

		new_tree_group = tree_group.copy().scale(0.8).shift(DOWN * 1.2)
		title = TextMobject(r"Minimize $L$").scale(1.2).move_to(UP * 3.5)
		h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1).next_to(title, DOWN)

		self.play(
			FadeOut(elbow),
			FadeOut(requirements),
			Write(title),
			ShowCreation(h_line),
			Transform(tree_group, new_tree_group)
		)
		self.wait()

		L_def_depth = TextMobject(r"$L = \sum\limits_{i=1}^{n} p_i \cdot$ (depth of $i^{\text{th}}$ symbol in tree)").scale(0.8)
		L_def_depth.next_to(h_line, DOWN)
		self.play(
			Write(L_def_depth)
		)
		self.wait()

		tree_edges = tree_group[1]
		labels = [Integer(i).scale(0.7) for i in range(1, 4)]
		all_labels = VGroup()

		for i in range(len(tree_edges)):
			if i < 2:
				label = labels[0].copy().move_to(tree_edges[i].get_center())
				BUFF = 0.3
			elif i < 6:
				label = labels[1].copy().move_to(tree_edges[i].get_center())
				BUFF = 0.4
			else:
				label = labels[2].copy().move_to(tree_edges[i].get_center())
				BUFF = 0.5
			shift_direction = get_normal_towards_origin(tree_edges[i].get_start(), tree_edges[i].get_end(), away=True)
			
			if i == 3 or i == 4 or i == 7:
				shift_direction = get_normal_towards_origin(tree_edges[i].get_start(), tree_edges[i].get_end())
			if i == 3:
				BUFF = 0.3
			label.shift(shift_direction * BUFF)
			all_labels.add(label)

		self.play(
			LaggedStart(*[Write(label) for label in all_labels]),
			run_time=2
		)
		self.wait()


		self.play(
			FadeOut(tree_group),
			FadeOut(all_labels)
		)

		assertion = TextMobject(
			"Optimal prefix-free encoding will have the two" + "\\\\", 
			"least likely symbols at the bottom of the tree.").scale(0.8)
		assertion.next_to(L_def_depth, DOWN)
		assertion[0][-3:].set_color(YELLOW)
		assertion[1][:11].set_color(YELLOW)
		self.play(
			Write(assertion),
			run_time=2
		)
		self.wait()

		note = TextMobject("*All prefix-free encodings can be represented by full binary trees").scale(0.6).next_to(assertion, DOWN)
		self.add(note)
		self.wait()

		dist = {
		'D': 0.15,
		'E': 0.16,
		'A': 0.17,
		'C': 0.17,
		'B': 0.35,
		}

		nodes = self.generate_example_dist(dist)
		nodes_mobs = VGroup(*[node.mob for node in nodes]).arrange(RIGHT)
		nodes_mobs.move_to(LEFT * 3 + DOWN * 1.5)
		self.play(
			FadeIn(nodes_mobs)
		)
		self.wait()

		parent_node = Node(0.31, key='DE')
		parent_node.generate_mob(text_scale=0.5, radius=0.6).scale(0.7)
		parent_node.mob.move_to(RIGHT * 3 + DOWN * 1)
		shift_amount = LEFT * 1
		self.play(
			nodes_mobs[0].move_to, RIGHT * 2 + DOWN * 3,
			nodes_mobs[1].move_to, RIGHT * 4 + DOWN * 3,
			nodes_mobs[2].shift, shift_amount,
			nodes_mobs[3].shift, shift_amount,
			nodes_mobs[4].shift, shift_amount,
			run_time=2,
		)
		self.wait()

		left_edge_mob = self.get_edge_mob(parent_node, nodes[0])
		right_edge_mob = self.get_edge_mob(parent_node, nodes[1])

		self.play(
			GrowFromCenter(parent_node.mob),
			ShowCreation(left_edge_mob),
			ShowCreation(right_edge_mob),
		)
		self.wait()

		surround_rect_nodes = SurroundingRectangle(VGroup(*[nodes_mobs[i] for i in range(2, len(nodes_mobs))]))
		surround_rect_nodes.set_color(RED)
		self.play(
			Write(surround_rect_nodes)
		)
		self.wait()

		constraint = TextMobject(
			"None of these symbols can have" +"\\\\",
			"longer encodings than D and E"
		).scale(0.8).next_to(surround_rect_nodes, DOWN)

		self.play(
			Write(constraint),
			run_time=2
		)
		self.wait()

		to_fade = self.generalize_nodes(nodes_mobs, parent_node.mob, surround_rect_nodes, note)

		surround_rect_tree = SurroundingRectangle(VGroup(parent_node.mob, nodes_mobs[0], nodes_mobs[1])).set_color(WHITE)

		self.play(
			ShowCreation(surround_rect_tree)
		)
		start_dashed_line = surround_rect_tree.get_top() + RIGHT * SMALL_BUFF * 3
		end_dashed_line = start_dashed_line + RIGHT * 2 + UP
		dashed_line = DashedLine(start_dashed_line, end_dashed_line, dash_length=0.1)
		self.play(
			Write(dashed_line)
		)
		self.wait()

		self.play(
			FadeOut(to_fade),
			FadeOut(surround_rect_tree),
			FadeOut(dashed_line),
			FadeOut(note),
			FadeOut(constraint),
			FadeOut(assertion),
			FadeOut(left_edge_mob),
			FadeOut(right_edge_mob),
		)
		self.wait()

		self.play(
			FadeOut(h_line),
			FadeOut(title),
			L_def_depth.shift, UP,
		)
		self.wait()
		tree_group.move_to(LEFT * 0.3)
		self.play(
			FadeIn(tree_group)
		)
		self.wait()

		L_def_rec = TextMobject(r"$L = $ ", "sum of all node probabilities except the root").scale(0.8)
		L_def_rec.move_to(DOWN * 3.5)
		self.play(
			Write(L_def_rec)
		)
		self.wait()

		tree_nodes, tree_edges, encoding_group = tree_group
		self.play(
			tree_nodes[0].set_stroke, None, None, 0.3,
			tree_edges[0].set_stroke, None, None, 0.3,
			tree_edges[1].set_stroke, None, None, 0.3,
		)
		self.wait()
		new_text = [TextMobject(str(tree[i].freq)) for i in range(2, len(tree))]
		new_text_summation = []
		for i, text in enumerate(new_text):
			new_text_summation.append(text)
			if i == len(new_text) - 1:
				break
			new_text_summation.append(TextMobject(r" $+$ "))
		new_text_summation.insert(0, TextMobject(r"$L =$ "))
		new_text_summation_group = VGroup(*new_text_summation).arrange(RIGHT, buff=SMALL_BUFF * 2)
		new_text_summation_group.scale(0.65).move_to(RIGHT * 2.1 + DOWN * 2)
		self.play(
			Write(new_text_summation_group[0])
		)
		self.wait()
		transforms = []
		fade_in_transforms = []
		for i in range(1, len(tree_nodes)):
			if i < 4:
				text = tree_nodes[i][1]
			else:
				text = tree_nodes[i][0][1]

			transforms.append(TransformFromCopy(text, new_text[i - 1]))
		for i in range(2, len(new_text_summation_group), 2):
			fade_in_transforms.append(FadeIn(new_text_summation_group[i]))
		
		self.play(
			*transforms + fade_in_transforms,
			run_time=4
		)
		self.wait()

		L_sum = TextMobject(r"$= 2.31$ bits").scale(0.65)
		L_sum.next_to(new_text_summation_group, DOWN, aligned_edge=LEFT)
		L_sum.shift(RIGHT * SMALL_BUFF * 3)
		self.play(
			Write(L_sum)
		)
		self.wait()

		self.play(
			tree_group.shift, UP,
			L_def_depth.next_to, L_def_rec, UP * 1.5,
			FadeOut(new_text_summation_group),
			FadeOut(L_sum),
			run_time=2
		)
		self.wait()

		L_depth_expanded = TextMobject(r"$L = 0.15 \cdot 3 + 0.16 \cdot 3 + 0.17 \cdot 2 + 0.17 \cdot 2 + 0.35 \cdot 2$")
		L_depth_expanded.scale(0.8).move_to(L_def_depth.get_center())
		self.play(
			ReplacementTransform(L_def_depth, L_depth_expanded),
			run_time=2
		)
		self.wait()

		L_rec_expanded = TextMobject(
			r"$L = $ ",
			"(",
			"0.15", r" $+$ ", "0.15", r" $+$ ", "0.15",
			")",
			r" $+$ ",
			"(",
			"0.16", r" $+$ ", "0.16", r" $+$ ", "0.16",
			")",
			r" $+$ " + "\\\\",
			"(",
			"0.17", r" $+$ ", "0.17",
			")",
			r" $+$ ",
			"(",
			"0.17", r" $+$ ", "0.17",
			")",
			r" $+$ ",
			"(",
			"0.35", r" $+$ ", "0.35",
			")",
		).scale(0.7).next_to(L_depth_expanded, DOWN * 1.5)
		L_rec_expanded[0].scale(0.8 / 0.7).shift(LEFT * SMALL_BUFF + DOWN * SMALL_BUFF * 2)

		node_1_expanded = TextMobject("(", "0.15", r" $+$ ", "0.16", r" $+$ ", "0.17", ")").scale(0.7)
		node_2_expanded = TextMobject("(", "0.17", r" $+$ ", "0.35", ")").scale(0.7)
		node_3_expanded = TextMobject("(", "0.15", r" $+$ ", "0.16", ")").scale(0.7)

		node_1_expanded.next_to(tree_nodes[1], LEFT)
		node_2_expanded.next_to(tree_nodes[2], RIGHT)
		node_3_expanded.next_to(tree_nodes[3], LEFT)

		self.play(
			TransformFromCopy(tree_nodes[7][0][1], node_3_expanded[1]),
			TransformFromCopy(tree_nodes[8][0][1], node_3_expanded[3]),
			FadeIn(node_3_expanded[::2]),
		)

		self.wait()

		self.play(
			TransformFromCopy(tree_nodes[5][0][1], node_2_expanded[1]),
			TransformFromCopy(tree_nodes[6][0][1], node_2_expanded[3]),
			TransformFromCopy(node_3_expanded[1], node_1_expanded[1]),
			TransformFromCopy(node_3_expanded[3], node_1_expanded[3]),
			TransformFromCopy(tree_nodes[4][0][1], node_1_expanded[5]),
			FadeIn(node_2_expanded[::2]),
			FadeIn(node_1_expanded[::2]),
		)
		self.wait()

		self.play(
			ReplacementTransform(L_def_rec[0], L_rec_expanded[0]),
			FadeOut(L_def_rec[1]),
		)

		L_rec_expanded[17:].shift(RIGHT * 0.3)
		miscelaneous_symbols_indices = [1, 3, 5, 7, 8, 9, 11, 13, 15, 16, 17, 19, 21, 22, 23, 25, 27, 28, 29, 31, 33]
		self.play(
			TransformFromCopy(tree_nodes[7][0][1], L_rec_expanded[2]),
			TransformFromCopy(node_3_expanded[1], L_rec_expanded[4]),
			TransformFromCopy(node_1_expanded[1], L_rec_expanded[6]),
			TransformFromCopy(tree_nodes[8][0][1], L_rec_expanded[10]),
			TransformFromCopy(node_3_expanded[3], L_rec_expanded[12]),
			TransformFromCopy(node_1_expanded[3], L_rec_expanded[14]),
			TransformFromCopy(tree_nodes[4][0][1], L_rec_expanded[18]),
			TransformFromCopy(node_1_expanded[5], L_rec_expanded[20]),
			TransformFromCopy(tree_nodes[5][0][1], L_rec_expanded[24]),
			TransformFromCopy(node_2_expanded[1], L_rec_expanded[26]),
			TransformFromCopy(tree_nodes[6][0][1], L_rec_expanded[30]),
			TransformFromCopy(node_2_expanded[3], L_rec_expanded[32]),
			*[FadeIn(L_rec_expanded[index]) for index in miscelaneous_symbols_indices],
			run_time=3,
		)
		self.wait()

		self.clear()

		self.show_generalized_motivation_of_huffman()


	def show_generalized_motivation_of_huffman(self):
		L_def_depth = TextMobject(r"$L = $ " ,r"$\sum\limits_{i=1}^{n} p_i \cdot$ (depth of $i^{\text{th}}$ symbol in tree)").scale(0.8)
		L_def_rec = TextMobject(r"$L = $ ", "sum of all node probabilities except the root").scale(0.8)

		L_def_depth_func = TextMobject(r"$L(p_1, p_2, \ldots, p_n) =$ ", r"$\sum\limits_{i=1}^{n} p_i \cdot$ (depth of $i^{\text{th}}$ symbol in tree)").scale(0.8)
		L_def_rec_func = TextMobject(r"$L(p_1, p_2, \ldots, p_n) =$ ", "sum of all node probabilities except the root").scale(0.8)

		VGroup(L_def_depth, L_def_rec).arrange(DOWN).move_to(UP * 3)
		self.play(
			Write(L_def_depth),
			Write(L_def_rec),
		)
		self.wait()

		L_def_depth_func.move_to(L_def_depth.get_center())
		L_def_rec_func.move_to(L_def_rec.get_center())
		self.play(
			Transform(L_def_depth, L_def_depth_func),
			Transform(L_def_rec, L_def_rec_func),
		)
		self.wait()

		new_strings = [
		(r"$p_1$", r"$s_1$"), 
		(r"$p_2$", r"$s_2$"),
		(r"$p_3$", r"$s_3$"),
		(r"$p_{k}$", r"$s_{k}$"),
		(r"$p_n$", r"$s_n$"),
		]

		cost_of_tree = TextMobject(r"$L(p_1, p_2, \ldots, p_n)$ is the ``cost'' of the tree").scale(0.8).next_to(L_def_rec_func, DOWN)

		self.play(
			Write(cost_of_tree)
		)
		self.wait()
		new_constraint = TextMobject(r"$p_1 \leq p_2 \leq p_3 \leq \cdots \leq p_k \leq \cdots \leq p_n$").scale(0.7)
		new_constraint.next_to(cost_of_tree, DOWN)

		new_nodes = [Node(freq, key=key) for freq, key in new_strings]
		print(new_nodes)
		for i, node in enumerate(new_nodes):
			node.generate_mob(text_scale=0.8, is_leaf=True, key_scale=0.9)

		ellipses = TexMobject(r"\cdots")
		group = VGroup(*[new_nodes[0].mob, new_nodes[1].mob, new_nodes[2].mob, ellipses, new_nodes[3].mob, ellipses.copy(), new_nodes[4].mob]).arrange(RIGHT)
		group.scale(0.8).next_to(new_constraint, DOWN)

		self.play(
			LaggedStart(*[Write(mob) for mob in group]),
			run_time=2
		)
		self.wait()
		self.add(new_constraint)
		self.wait()

		shift_up = UP * 1
		self.play(
			FadeOut(L_def_depth),
			L_def_rec.shift, shift_up,
			cost_of_tree.shift, shift_up,
			new_constraint.shift, UP,
			group.shift, UP
		)
		self.wait()
		
		left_tree_node = group[0].copy().scale(0.8).move_to(LEFT * 0.75 + DOWN * 3.2)
		right_tree_node = group[1].copy().scale(0.8).move_to(RIGHT * 0.75 + DOWN * 3.2)
		self.play(
			ReplacementTransform(group[0], left_tree_node),
			ReplacementTransform(group[1], right_tree_node),
			group[2:].next_to, new_constraint, DOWN,
			run_time=2,
		)
		self.wait()

		parent_node = Node(r"$p_1 + p_2$", key='s1s2')
		parent_node.generate_mob(text_scale=0.3, radius=0.7)
		parent_node.mob.scale(0.65).move_to(DOWN * 1.5)

		left_edge_mob = self.get_edge_mob(parent_node, new_nodes[0])
		right_edge_mob = self.get_edge_mob(parent_node, new_nodes[1])

		self.play(
			ShowCreation(left_edge_mob),
			ShowCreation(right_edge_mob),
		)
		self.play(
			Write(parent_node.mob)
		)
		self.wait()

		partial_tree_nodes = VGroup(parent_node.mob, left_tree_node, right_tree_node)
		partial_edges = VGroup(left_edge_mob, right_edge_mob)
		partial_tree = VGroup(partial_tree_nodes, partial_edges)

		self.play(
			group[2:].move_to, DOWN * 3 + RIGHT * 4.5,
			partial_tree.shift, LEFT * 5,
			run_time=2
		)
		self.wait()

		new_node_in_set = parent_node.mob.copy().next_to(group[2], LEFT * 2)
		self.play(
			TransformFromCopy(parent_node.mob, new_node_in_set)
		)
		self.wait()

		assumption = TextMobject(r"*Assume $(p_1 + p_2) \leq p_3$ in this example").scale(0.6)
		dont_worry = TextMobject(r"*Don't worry, we'll see other cases for clarity").scale(0.6)
		new_set_of_nodes = VGroup(new_node_in_set, group[2:])
		assumption.next_to(new_set_of_nodes, UP)
		dont_worry.next_to(assumption, UP)
		self.add(assumption)
		self.wait()
		self.add(dont_worry)
		self.wait()

		self.play(
			FadeOut(dont_worry),
			FadeOut(assumption)
		)
		self.wait()

		L_def_rec_split = TextMobject(r"$L(p_1, p_2, \ldots, p_n) =$ ", r"$p_1 + p_2 + L((p_1 + p_2), p_3, \ldots, p_n)$").scale(0.8).move_to(L_def_rec.get_center())
		self.play(
			Transform(L_def_rec, L_def_rec_split)
		)
		self.wait()

		remaining_tree_nodes = VGroup(*[Dot().set_color(YELLOW) for _ in range(4)])
		remaining_tree_edges = VGroup()

		scale_factor = 0.8
		DIR = normalize(left_edge_mob.get_start() - left_edge_mob.get_end())
		NEG_DIR = np.array([-DIR[0], DIR[1], DIR[2]])

		ll_edge = Line(parent_node.mob.get_center() + DIR * 0.7, parent_node.mob.get_center() + DIR * (0.7 + scale_factor))
		lr_edge = Line(ll_edge.get_start() - NEG_DIR * scale_factor, ll_edge.get_start())
		ll_edge.set_stroke(color=GRAY, width=6)
		lr_edge.set_stroke(color=GRAY, width=6)
		lr_edge.next_to(ll_edge, RIGHT)

		remaining_tree_nodes.move_to(lr_edge.get_start())

		ellipses_ll_lr = TexMobject(r"\cdots")
		ll_lr_edge_group = VGroup(ll_edge, lr_edge)
		ellipses_ll_lr.next_to(ll_lr_edge_group, UP)
		l_edge = Line(ellipses_ll_lr.get_center() + UR * SMALL_BUFF * 3, ellipses_ll_lr.get_center() + UR * (SMALL_BUFF * 3 + scale_factor))
		r_edge = Line(l_edge.get_end() + DR * scale_factor, l_edge.get_end())
		l_edge.set_stroke(color=GRAY, width=6)
		r_edge.set_stroke(color=GRAY, width=6)
		r_edge.next_to(l_edge, RIGHT, buff=0)
		remaining_tree_nodes[1].move_to(l_edge.get_end())

		ellipses_rl_rr = TexMobject(r"\cdots")
		ellipses_rl_rr.move_to(r_edge.get_start() + DR * SMALL_BUFF * 3)

		rl_rr_edge_group = ll_lr_edge_group.copy().next_to(ellipses_rl_rr, DOWN)
		remaining_tree_nodes[2].move_to(rl_rr_edge_group[0].get_start())
		remaining_tree_nodes[3].move_to(rl_rr_edge_group[1].get_start())
		self.play(
			ShowCreation(ll_edge),
			ShowCreation(lr_edge),
			GrowFromCenter(remaining_tree_nodes[0]),

			FadeIn(ellipses_ll_lr),
			ShowCreation(l_edge),
			ShowCreation(r_edge),
			GrowFromCenter(remaining_tree_nodes[1]),

			FadeIn(ellipses_rl_rr),
			ShowCreation(rl_rr_edge_group),
			GrowFromCenter(remaining_tree_nodes[2]),
			GrowFromCenter(remaining_tree_nodes[3])
		)
		self.wait()



		surround_p1_p2 = SurroundingRectangle(L_def_rec[1][:7]).set_color(ORANGE)
		surround_p1_p2_nodes = SurroundingRectangle(VGroup(partial_tree_nodes[1], partial_tree_nodes[2])).set_color(ORANGE)

		self.play(
			ShowCreation(surround_p1_p2),
			ShowCreation(surround_p1_p2_nodes)
		)
		self.wait()

		surround_rec_part = SurroundingRectangle(L_def_rec[1][9:]).set_color(BLUE)
		surround_rec_part_tree = SurroundingRectangle(VGroup(partial_tree_nodes[0], remaining_tree_nodes[1], remaining_tree_nodes[3])).set_color(BLUE)

		self.play(
			ShowCreation(surround_rec_part),
			ShowCreation(surround_rec_part_tree)
		)
		self.wait()

		p1_p2_label = L_def_rec[1][:7].copy().next_to(surround_p1_p2_nodes, RIGHT * 2)
		rec_part_label = L_def_rec[1][9:].copy().next_to(surround_rec_part_tree, RIGHT * 2)

		self.play(
			TransformFromCopy(L_def_rec[1][:7], p1_p2_label),
			TransformFromCopy(L_def_rec[1][9:], rec_part_label),
			run_time=2
		)
		self.wait()

		minimizing_note = TextMobject(
			"Minimizing this function" + "\\\\",
			"is the same process").scale(0.7)

		top_brace = Brace(rec_part_label, direction=DOWN)
		self.play(
			GrowFromCenter(top_brace)
		)
		minimizing_note.next_to(top_brace, DOWN)
		self.play(
			FadeIn(minimizing_note)
		)
		self.wait()

		self.play(
			FadeOut(ll_edge),
			FadeOut(lr_edge),
			FadeOut(remaining_tree_nodes[0]),

			FadeOut(ellipses_ll_lr),
			FadeOut(l_edge),
			FadeOut(r_edge),
			FadeOut(remaining_tree_nodes[1]),

			FadeOut(ellipses_rl_rr),
			FadeOut(rl_rr_edge_group),
			FadeOut(remaining_tree_nodes[2]),
			FadeOut(remaining_tree_nodes[3]),

			FadeOut(minimizing_note),
			FadeOut(top_brace),
			FadeOut(p1_p2_label),
			FadeOut(rec_part_label),
			FadeOut(surround_p1_p2),
			FadeOut(surround_rec_part),
			FadeOut(surround_p1_p2_nodes),
			FadeOut(surround_rec_part_tree),
		)
		self.wait()

		self.play(
			ApplyWave(new_node_in_set),
			ApplyWave(group[2]),
		)
		self.wait()

		new_right_leaf = group[2].copy().scale(0.8).next_to(parent_node.mob, RIGHT * 3, aligned_edge=UP)

		self.play(
			new_node_in_set.move_to, parent_node.mob.get_center(),
			new_node_in_set.fade, 1,
			ReplacementTransform(group[2], new_right_leaf),
			FadeOut(group[3]),
			run_time=2,
		)
		self.wait()

		root_node = Node(r"$\sum\limits_{i=1}^{3} p_i$", key='s1s2s3')
		root_node.generate_mob(text_scale=0.4, radius=0.7)
		root_node.mob.scale(0.65).next_to(VGroup(parent_node.mob, new_right_leaf), UP * 4)

		root_left_edge = self.get_edge_mob(root_node, parent_node)
		root_right_edge = self.get_edge_mob(root_node, new_nodes[2])

		self.play(
			GrowFromCenter(root_left_edge),
			GrowFromCenter(root_right_edge),
		)
		self.play(
			FadeIn(root_node.mob)
		)
		self.wait()



	def generalize_nodes(self, nodes_mobs, parent_node_mob, surround_rect, note):
		new_strings = [
		(r"$p_1$", r"$s_1$"), 
		(r"$p_2$", r"$s_2$"),
		(r"$p_3$", r"$s_3$"),
		(r"$p_{k}$", r"$s_{k}$"),
		(r"$p_n$", r"$s_n$"),
		]

		parent_string = r"$p_1 + p_2$"
		new_parent_text = TextMobject(parent_string).scale(0.5).move_to(parent_node_mob.get_center())
		new_nodes = [Node(freq, key=key) for freq, key in new_strings]
		for i, node in enumerate(new_nodes):
			node.generate_mob(text_scale=0.8, is_leaf=True, key_scale=0.9)
			node.mob.scale(0.8).move_to(nodes_mobs[i].get_center())

		remaining_node_group = VGroup(*[new_nodes[i].mob for i in range(2, len(new_nodes))]).arrange(RIGHT)
		remaining_node_group.move_to(surround_rect.get_center())
		
		new_constraint = TextMobject(r"$p_1 \leq p_2 \leq p_3 \leq \ldots \leq p_k \leq \ldots \leq p_n$").scale(0.7)
		new_constraint.next_to(note, DOWN)

		ellipses = TexMobject(r"\ldots")
		group = VGroup(*[remaining_node_group[0], ellipses, remaining_node_group[1], ellipses.copy(), remaining_node_group[2]]).arrange(RIGHT)
		group.move_to(surround_rect.get_center())
		new_rect = SurroundingRectangle(group, color=RED)
		self.play(
			*[Transform(nodes_mobs[i], new_nodes[i].mob) for i in range(len(new_nodes))],
			FadeIn(group[1]),
			FadeIn(group[3]),
			Transform(surround_rect, new_rect),
			Transform(parent_node_mob[1], new_parent_text),
			FadeIn(new_constraint),
			run_time=2,
		)
		self.wait()

		to_fade = VGroup(
			nodes_mobs,
			parent_node_mob,
			group[1],
			group[3],
			surround_rect,
			new_constraint
		)
		return to_fade

	def get_edge_key_decoding_order(self, encoding, leaf_indices):
		current_index = 1
		edge_key_decoding_order = []
		for bit in encoding:
			next_index = self.get_left_index(current_index)
			if bit:
				next_index = self.get_right_index(current_index)
			edge_key_decoding_order.append((current_index, next_index) )
			current_index = next_index
			if next_index in leaf_indices:
				current_index = 1

		return edge_key_decoding_order

	def get_example_encoding(self, encoding_dict):
		scale = 0.7
		encoding_mobjects = VGroup()
		encoding_map_letter_to_index = {}
		for i, letter in enumerate(encoding_dict):
			binary_list = encoding_dict[letter]
			group = get_binary_group(len(binary_list), binary_list=binary_list)
			encoding_mob = self.get_individual_encoding(letter, group)
			encoding_mobjects.add(encoding_mob.scale(scale))
			encoding_map_letter_to_index[letter] = i
		encoding_mobjects.arrange(DOWN, buff=SMALL_BUFF * 1.5)
		for i in range(len(encoding_mobjects)):
			encoding_mobjects[i].to_edge(LEFT)
		return encoding_mobjects, encoding_map_letter_to_index

	def get_individual_encoding(self, letter, group):
		key = TextMobject(r"{0} $\rightarrow$ ".format(letter))
		encoding = VGroup(key, group).arrange(RIGHT, buff=SMALL_BUFF * 1.5)
		return encoding


	def make_component(self, text, color=YELLOW):
		# geometry is first index, TextMob is second index
		text_mob = TextMobject(text)
		rect = Rectangle(color=color, height=1.5, width=2.5)
		return VGroup(rect, text_mob)

	def highlight_unique_paths(self, order):
		highlighted_mobs = self.get_highlight_mobs(order)
		transforms = [ShowCreationThenDestruction(mob) for mob in highlighted_mobs]
		self.play(
			LaggedStart(*transforms),
			run_time=3,
		)
		self.wait()

	def get_highlight_mobs(self, keys_of_edges):
		color = PERIWINKLE_BLUE
		width = 8
		highlighted_mobs = [self.edge_dict[key].copy() for key in keys_of_edges]
		for mob in highlighted_mobs:
			mob.set_stroke(color=color, width=width)

		return highlighted_mobs


	def show_nodes(self, list_of_nodes):
		all_show_animations = []
		for nodes in list_of_nodes:
			group = VGroup(*[node.mob for node in nodes]).arrange(RIGHT)
			key = self.get_key(nodes)
			group.move_to(self.position_map[key])
			all_show_animations.append(FadeIn(group))
		self.play(
			*all_show_animations
		)
		self.wait()
	# Only part of this that needs to be hard coded
	def get_position_map(self):
		position_map = {
		'ABCDE': UP * 3,
		'DEA': UP * 1.5 + LEFT * 3,
		'CB': UP * 1.5 + RIGHT * 3,
		'DE': DOWN * 0.5 + LEFT * 5,
		'A':DOWN * 0.5 + LEFT * 1,
		'D': DOWN * 2.5 + LEFT * 6,
		'E': DOWN * 2.5 + LEFT * 4,
		'C': DOWN * 0.5 + RIGHT * 1.5,
		'B': DOWN * 0.5 + RIGHT * 4.5,
		}
		return position_map

	# returns concatenated key of sorted_nodes
	def get_key(self, sorted_nodes):
		if isinstance(sorted_nodes, Node):
			return sorted_nodes.key
		return ''.join([node.key for node in sorted_nodes])

	def generate_example_dist(self, dist):
		nodes = []
		for key in dist:
			node = Node(dist[key], key)
			node.generate_mob().scale(0.7)
			nodes.append(node)
		return nodes

	def get_left_index(self, index):
		return index * 2

	def get_right_index(self, index):
		return index * 2 + 1

	def animate_shannon_fano(self, sorted_nodes, tree, tree_index):
		if len(sorted_nodes) == 1:
			return
		i, j = 0, len(sorted_nodes) - 1
		sum_left, sum_right = 0, 0
		while i <= j:
			if sum_left <= sum_right:
				sum_left += sorted_nodes[i].freq
				i += 1
			else:
				sum_right += sorted_nodes[j].freq
				j -= 1
		self.split_nodes(i, tree, tree_index)
		self.animate_shannon_fano(sorted_nodes[:i], tree, self.get_left_index(tree_index))
		self.animate_shannon_fano(sorted_nodes[i:], tree, self.get_right_index(tree_index))

	def split_nodes(self, index, tree, tree_index):
		from copy import deepcopy
		sorted_nodes = tree[tree_index]
		left_split_end = sorted_nodes[index - 1].mob
		right_split_begin = sorted_nodes[index].mob
		midpoint = (left_split_end.get_center() + right_split_begin.get_center()) / 2
		dashed_line = DashedLine(midpoint + UP * 0.7, midpoint + DOWN * 0.7, dash_length=0.1)
		self.dashed_lines[self.get_key(sorted_nodes)] = dashed_line
		self.play(
			Write(dashed_line)
		)
		self.wait()

		left_nodes = [deepcopy(sorted_nodes[i]) for i in range(index)]
		right_nodes = [deepcopy(sorted_nodes[j]) for j in range(index, len(sorted_nodes))]
		
		left_nodes_group = VGroup(*[node.mob for node in left_nodes])
		left_nodes_key = self.get_key(left_nodes)
		left_nodes_group.move_to(self.position_map[left_nodes_key])
		tree[self.get_left_index(tree_index)] = left_nodes

		right_nodes_group = VGroup(*[node.mob for node in right_nodes])
		right_nodes_key = self.get_key(right_nodes)
		right_nodes_group.move_to(self.position_map[right_nodes_key])
		tree[self.get_right_index(tree_index)] = right_nodes

		self.show_transform(tree, tree_index, self.get_left_index(tree_index), self.get_right_index(tree_index), index)

	def show_transform(self, tree, root_index, left_index, right_index, split_index):
		sorted_nodes = tree[root_index]
		left_nodes_original = [sorted_nodes[i] for i in range(split_index)]
		right_nodes_original = [sorted_nodes[j] for j in range(split_index, len(sorted_nodes))]

		left_nodes_new = tree[left_index]
		right_nodes_new = tree[right_index]

		transforms = []
		for orig, new in zip(left_nodes_original, left_nodes_new):
			transforms.append(TransformFromCopy(orig.mob, new.mob))

		for orig, new in zip(right_nodes_original, right_nodes_new):
			transforms.append(TransformFromCopy(orig.mob, new.mob))

		self.play(
			*transforms,
			run_time=2,
		)
		self.wait()

	def get_nodes(self, tree, key):
		for nodes in tree:
			if isinstance(nodes, int):
				continue
			if self.get_key(nodes) == key:
				return nodes

	def get_frequency_from_children(self, left_node, right_node):
		if isinstance(left_node, Node) and isinstance(right_node, Node):
			return left_node.freq + right_node.freq
		elif isinstance(left_node, Node):
			return left_node.freq + right_node[0].freq
		elif isinstance(right_node, Node):
			return left_node[0].freq + right_node.freq
		return left_node[0].freq + right_node[0].freq

	def is_leaf(self, node):
		return not isinstance(node, Node) and len(node) == 1

	def make_splits_into_nodes(self, tree):
		transform_order = ['DE', 'DEA', 'CB', 'DEACB']
		animation_groups = []
		for key in transform_order:
			original_nodes = self.get_nodes(tree, key)
			original_nodes_group = VGroup(*[node.mob for node in original_nodes])
			index = tree.index(original_nodes)
			left_nodes = tree[self.get_left_index(index)]
			right_nodes = tree[self.get_right_index(index)]
			new_freq = self.get_frequency_from_children(left_nodes, right_nodes)
			new_node = Node(new_freq, key)
			new_node.generate_mob(text_scale=0.5, radius=0.6).scale(0.7)
			new_node.mob.move_to(original_nodes_group.get_center())
			if self.is_leaf(left_nodes):
				left_edge = self.get_edge_mob(new_node, left_nodes[0])
			else:
				left_edge = self.get_edge_mob(new_node, left_nodes)

			if self.is_leaf(right_nodes):
				right_edge = self.get_edge_mob(new_node, right_nodes[0])
			else:
				right_edge = self.get_edge_mob(new_node, right_nodes)

			self.edge_dict[(index, self.get_left_index(index))] = left_edge
			self.edge_dict[(index, self.get_right_index(index))] = right_edge

			animation_group = AnimationGroup(
				ReplacementTransform(
					original_nodes_group, new_node.mob,
				),
				FadeOut(self.dashed_lines[key]),
				GrowFromCenter(left_edge),
				GrowFromCenter(right_edge),
			)
			tree[index] = new_node
			animation_groups.append(animation_group)
		return animation_groups



class ProblemFormulation(GraphScene):
	def construct(self):
		problem = TextMobject(
			"Given a collection of letters, numbers, or other symbols, find" + "\\\\", 
			"the most efficient method to represent them using a binary code."
		).scale(0.8)

		self.play(
			FadeIn(problem)
		)
		self.wait()

		self.play(
			FadeOut(problem)
		)

		diagram = self.create_comm_diagram()

	def create_comm_diagram(self):
		source = self.make_component("Sender")
		new_source = self.make_component("Source")
		encoder = self.make_component("Encoder", color=ORANGE)
		message = self.make_component("Message", color=RED)
		decoder = self.make_component("Decoder", color=GREEN_SCREEN)
		receiver = self.make_component("Receiver", color=BLUE)

		HORIZONTAL_SHIFT = 5
		VERTICAL_SHIFT = 1.5

		source.move_to(LEFT * HORIZONTAL_SHIFT + UP * VERTICAL_SHIFT)
		encoder.move_to(LEFT * HORIZONTAL_SHIFT + DOWN * VERTICAL_SHIFT)
		message.move_to(DOWN * VERTICAL_SHIFT)
		decoder.move_to(RIGHT * HORIZONTAL_SHIFT + DOWN * VERTICAL_SHIFT)
		receiver.move_to(RIGHT * HORIZONTAL_SHIFT + UP * VERTICAL_SHIFT)

		network = VGroup(
			source, encoder, message, decoder, receiver,
		).shift(UP)

		arrows = []
		arrow_color = GRAY
		arrows.append(Arrow(source.get_bottom(), encoder.get_top()).set_color(arrow_color))
		arrows.append(Arrow(encoder.get_right(), message.get_left()).set_color(arrow_color))
		arrows.append(Arrow(message.get_right(), decoder.get_left()).set_color(arrow_color))
		arrows.append(Arrow(decoder.get_top(), receiver.get_bottom()).set_color(arrow_color))

		arrows = VGroup(*arrows)

		source_text = self.get_example_source("``Shannon''", source)
		self.play(
			Write(source),
		)
		self.wait()

		self.play(
			Write(receiver)
		)
		self.wait()

		new_source.move_to(source.get_center())

		self.play(
			Transform(source, new_source)
		)
		self.wait()

		self.play(
			Write(source_text)
		)
		self.wait()

		self.play(
			Write(arrows[0])
		)
		self.wait()

		self.play(
			Write(encoder)
		)

		encoding_dict1 = {
		'a': [0, 0, 0],
		'h': [0, 0, 1],
		'n': [0, 1, 0],
		'o': [0, 1, 1],
		'S': [1, 0, 0]
		}

		self.wait()
		original_message = "Shannon"
		encoding_text, encoding_letter_to_index = self.get_example_encoding(encoder, encoding_dict1)
		self.play(
			FadeIn(encoding_text)
		)

		self.wait()

		self.play(
			Write(arrows[1])
		)
		self.wait()

		encoded_message = self.get_encoded_message(original_message, encoding_dict1, message)
		self.play(
			Write(message)
		)
		self.wait()
		self.play(
			Write(encoded_message)
		)
		self.wait()

		self.play(
			Write(arrows[2])
		)

		self.play(
			Write(decoder)
		)

		self.wait()

		decoded_message = self.show_decoded_message(encoded_message, encoding_text, original_message, decoder, encoding_letter_to_index, encoding_dict1)

		self.play(
			Write(arrows[3])
		)
		self.wait()

		

		receiver_text = source_text.copy().next_to(receiver, UP)
		self.play(
			*[TransformFromCopy(decoded_message[i], receiver_text[0][i + 2]) for i in range(len(decoded_message))],
			FadeIn(receiver_text[0][:2]),
			FadeIn(receiver_text[0][-2:]),
			run_time=2
		)
		self.wait()

		surround_rect_highlight = SurroundingRectangle(
			VGroup(encoder, encoding_text),
			buff=SMALL_BUFF
		).set_color(FUCHSIA)

		self.play(
			ShowCreationThenDestruction(surround_rect_highlight),
			run_time=2
		)
		self.wait()

		encoding_dict_bad = {
		'S': [0, 0, 0],
		'o': [0, 0, 1, 1],
		'a': [0, 1, 0, 1],
		'h': [0, 1, 1],
		'n': [1, 0, 0, 0, 1],
		}

		encoding_text_bad, encoding_letter_to_index_bad = self.get_example_encoding(encoder, encoding_dict_bad)

		encoded_message_bad = self.get_encoded_message(original_message, encoding_dict_bad, message)
		self.play(
			ReplacementTransform(encoding_text, encoding_text_bad),
			ReplacementTransform(encoded_message, encoded_message_bad)
		)
		self.wait()

		encoding_dict_huff = {
		'S': [0, 0, 0],
		'o': [0, 0, 1],
		'a': [0, 1, 0],
		'h': [0, 1, 1],
		'n': [1],
		}

		encoding_text_huff, encoding_letter_to_index_huff = self.get_example_encoding(encoder, encoding_dict_huff)

		encoded_message_huff = self.get_encoded_message(original_message, encoding_dict_huff, message)
		self.play(
			ReplacementTransform(encoding_text_bad, encoding_text_huff),
			ReplacementTransform(encoded_message_bad, encoded_message_huff)
		)
		self.wait()

		brace = Brace(encoded_message_huff, DOWN)
		self.play(
			GrowFromCenter(brace)
		)

		goal = TextMobject(
			"We want to use the least" + "\\\\",
			"number of bits possible"
		).scale(0.8)
		goal.next_to(brace, DOWN)
		self.play(
			Write(goal)
		)
		self.wait()

		requirements = self.show_requirements(
			network, encoding_text_huff, encoded_message_huff, encoding_dict_huff, brace, goal, decoded_message, receiver_text,
		)

	def show_requirements(self, networks, encoding_text, encoded_message_huff, encoding_dict, brace, goal, decoded_message, receiver_text):
		original_encoding_message = encoded_message_huff.copy()
		requirements = VGroup(
			TextMobject(r"1. Single symbol $\rightarrow$ unique binary code").scale(0.7),
			TextMobject(r"2. Source message $=$ received message").scale(0.7),
			TextMobject(r"3. Unique decodability").scale(0.7),
		).arrange(DOWN).move_to(UP * 2.5)
		indent = LEFT * 8
		for req in requirements:
			req.to_edge(indent)

		requirements_title = Title(
			"Requirements", 
			match_underline_width_to_text=True, 
			underline_buff=SMALL_BUFF
		)
		midpoint_network = (networks[0].get_center() + networks[-1].get_center()) / 2
		requirements.move_to(midpoint_network).shift(DOWN * 0.3)
		requirements_title.next_to(requirements, UP * 1.5)

		self.play(
			Write(requirements_title)
		)

		self.play(
			FadeIn(requirements[0]),
			FadeOut(brace),
			FadeOut(goal),
		)
		self.wait()

		encoding_text = self.show_invalid_encodings(encoding_text, networks[1])

		self.wait()

		self.play(
			FadeIn(requirements[1])
		)
		self.wait()

		encoding_letter_to_index = {
		'S': 0,
		'o': 1,
		'a': 2,
		'h': 3,
		'n': 4,
		}
		original_encoding_message = self.show_incorrect_decoding(encoded_message_huff, original_encoding_message, encoding_text, encoding_dict, encoding_letter_to_index, networks[2], networks[3], decoded_message, receiver_text, networks[4], requirements)

		self.play(
			FadeIn(requirements[2])
		)
		self.wait()

		self.show_ambiguous_decoding(encoding_text, original_encoding_message, networks)

		self.play(
			ApplyWave(requirements[1])
		)
		self.wait()

		return requirements

	def show_invalid_encodings(self, encoding_text, encoder):
		original_encoding_text = encoding_text.copy()
		bad_encoding_1 = {
		'Sh': [0, 0],
		'an': [0, 1],
		'non': [1, 0],
		}
		first_bad_encoding_text, _ = self.get_example_encoding(encoder, bad_encoding_1)
		self.play(
			ReplacementTransform(encoding_text, first_bad_encoding_text)
		)
		self.wait()

		cross = Cross(first_bad_encoding_text)
		self.play(
			ShowCreation(cross)
		)
		self.wait()

		self.play(
			FadeOut(cross),
			FadeOut(first_bad_encoding_text)
		)
		self.play(
			FadeIn(original_encoding_text)
		)
		self.wait()

		return original_encoding_text

	def show_incorrect_decoding(self, encoded_message, original_encoding_message, encoding_text, encoding_dict, encoding_letter_to_index, 
		message_component, decoder, decoded_message, receiver_text, receiver, requirements):
		deleted_index = 6
		self.play(
			encoded_message[deleted_index].set_color, RED,
		)
		self.wait()
		self.play(
			encoded_message[deleted_index].shift, DOWN,
			encoded_message[deleted_index].fade, 1,
		)
		self.wait()

		new_encoded_message_with_deletion = get_binary_group(
			len(encoded_message) - 1, 
			binary_list=[0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1]
		)

		new_encoded_message_with_deletion.scale(0.7).next_to(message_component, DOWN * 2)
		right_shift = new_encoded_message_with_deletion[:deleted_index].get_center() - encoded_message[:deleted_index].get_center()
		left_shift = new_encoded_message_with_deletion[deleted_index:].get_center()  - encoded_message[deleted_index + 1:].get_center()

		self.play(
			encoded_message[:deleted_index].shift, right_shift,
			encoded_message[deleted_index + 1:].shift, left_shift,
		)

		self.add(new_encoded_message_with_deletion)
		encoded_message.set_color(BLACK)
		self.wait()

		self.play(
			FadeOut(decoded_message),
			FadeOut(receiver_text),
		)
		self.wait()

		wrong_decoded_message = self.show_decoded_message(
			new_encoded_message_with_deletion, encoding_text, "Shnhon", 
			decoder, encoding_letter_to_index, encoding_dict,
		)

		wrong_received_text = TextMobject("``Shnhon''").scale(0.8).next_to(receiver, UP)

		self.play(
			*[TransformFromCopy(wrong_decoded_message[i], wrong_received_text[0][i + 2]) for i in range(len(wrong_decoded_message))],
			FadeIn(wrong_received_text[0][:2]),
			FadeIn(wrong_received_text[0][-2:]),
			run_time=2
		)
		self.wait()

		cross = Cross(wrong_received_text)
		self.play(
			ShowCreation(cross)
		)
		self.wait()

		lossless_compression = TextMobject("2. Compression must be lossless").scale(0.7)
		lossless_compression.move_to(requirements[1].get_center())
		lossless_compression.to_edge(LEFT * 8.25)
		lossless_compression[0][-8:].set_color(YELLOW)
		self.play(
			Transform(requirements[1], lossless_compression)
		)
		self.wait()

		self.play(
			FadeOut(cross),
			FadeOut(wrong_received_text),
			FadeOut(wrong_decoded_message),
			FadeOut(new_encoded_message_with_deletion),
		)

		self.play(
			FadeIn(original_encoding_message),
			FadeIn(receiver_text),
			FadeIn(decoded_message),
		)
		self.wait()
		return original_encoding_message

	def show_ambiguous_decoding(self, encoding_text, encoded_message, network):
		original_encoding_message = encoded_message.copy()
		ambiguous_encoding_dict = {
		'S': [0, 0],
		'o': [0, 0, 1],
		'a': [0, 1, 0],
		'h': [0, 1, 1],
		'n': [1],
		}

		ambiguous_encoding_text, _ = self.get_example_encoding(network[1], ambiguous_encoding_dict)
		self.play(
			Flash(encoding_text[0][1]),
			ReplacementTransform(encoding_text, ambiguous_encoding_text)
		)
		self.wait()

		ambiguous_encoded_message = self.get_encoded_message("Shannon", ambiguous_encoding_dict, network[2])

		self.play(
			ReplacementTransform(encoded_message, ambiguous_encoded_message)
		)

		self.wait()

		larger_buff = get_binary_group(len(encoded_message), binary_list=[0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1], buff=SMALL_BUFF * 2.5)
		larger_buff.scale(0.7)
		larger_buff.move_to(encoded_message.get_center())
		self.play(
			Transform(ambiguous_encoded_message, larger_buff)
		)
		self.wait()

		self.show_split_of_encoding("Shannon", [2, 5, 8, 9, 10, 13], ambiguous_encoded_message)

		ambiguous_copy = ambiguous_encoded_message.copy()

		ambiguous_copy.shift(DOWN * 1.2)

		self.play(
			TransformFromCopy(ambiguous_encoded_message, ambiguous_copy)
		)
		self.wait()

		self.show_split_of_encoding("ShannSnn", [2, 5, 8, 9, 10, 12, 13], ambiguous_copy, vertical_buff=DOWN*0.6, color=RED)

	def show_split_of_encoding(self, message, split_indices, encoded_message, vertical_buff=DOWN*3, color=GREEN_SCREEN):
		dashed_lines = VGroup()
		for index in split_indices:
			left_split_end = encoded_message[index - 1]
			right_split_begin = encoded_message[index]
			midpoint = (left_split_end.get_center() + right_split_begin.get_center()) / 2
			dashed_line = DashedLine(midpoint + UP * 0.3, midpoint + DOWN * 0.3, dash_length=0.1).set_color(color)
			dashed_lines.add(dashed_line)

		decoded_message = VGroup()
		start = 0
		end = split_indices[0]
		for i, letter in enumerate(message):
			letter_mob = TextMobject(message[i]).scale(0.7)
			letter_mob.next_to(encoded_message[start:end], DOWN)
			letter_mob.to_edge(vertical_buff)
			decoded_message.add(letter_mob)
			if i >= len(split_indices) - 1:
				break
			start, end = end, split_indices[i + 1]

		final = len(encoded_message)
		letter_mob = TextMobject(message[-1]).scale(0.7)
		letter_mob.next_to(encoded_message[end:final], DOWN)
		letter_mob.to_edge(vertical_buff)
		decoded_message.add(letter_mob)
		self.play(
			FadeIn(dashed_lines),
			FadeIn(decoded_message),
		)
		self.wait()

	def make_component(self, text, color=YELLOW):
		# geometry is first index, TextMob is second index
		text_mob = TextMobject(text)
		rect = Rectangle(color=color, height=1.5, width=2.5)
		return VGroup(rect, text_mob)

	def get_example_source(self, source, source_component, scale=0.8):
		return TextMobject(source).next_to(source_component, UP).scale(scale)

	def get_example_encoding(self, encoding, encoding_dict):
		scale = 0.7
		encoding_mobjects = VGroup()
		encoding_map_letter_to_index = {}
		for i, letter in enumerate(encoding_dict):
			binary_list = encoding_dict[letter]
			group = get_binary_group(len(binary_list), binary_list=binary_list)
			encoding_mob = self.get_individual_encoding(letter, group)
			encoding_mobjects.add(encoding_mob.scale(scale))
			encoding_map_letter_to_index[letter] = i
		encoding_mobjects.arrange(DOWN, buff=SMALL_BUFF * 1.5)
		for i in range(len(encoding_mobjects)):
			encoding_mobjects[i].to_edge(LEFT)
		encoding_mobjects.next_to(encoding, DOWN)
		return encoding_mobjects, encoding_map_letter_to_index

	def get_individual_encoding(self, letter, group):
		key = TextMobject(r"{0} $\rightarrow$ ".format(letter))
		encoding = VGroup(key, group).arrange(RIGHT, buff=SMALL_BUFF * 1.5)
		return encoding

	def get_encoded_message(self, string, encoding_dict, message_component):
		binary_list = []
		for letter in string:
			binary_list.extend(encoding_dict[letter])

		message = get_binary_group(len(binary_list), binary_list=binary_list)
		message.scale(0.7).next_to(message_component, DOWN * 2)
		return message

	def show_decoded_message(self, encoded_message, encoded_text, original_message, decoder_component, encoding_letter_to_index, encoding_dict):
		from math import log2, ceil
		decoded_message = TextMobject(*[char for char in original_message])
		decoded_message.scale(0.8).next_to(decoder_component, DOWN)
		length = len(encoding_dict[original_message[0]])
		surround_rect_message = SurroundingRectangle(
			encoded_message[0:length], buff=SMALL_BUFF * 0.5
		).set_color(GREEN_SCREEN).set_fill(color=GREEN_SCREEN, opacity=0.3)
		surround_rect_encoding = SurroundingRectangle(
				encoded_text[encoding_letter_to_index[original_message[0]]], 
				buff=SMALL_BUFF
		).set_color(GREEN_SCREEN)
		start_index = 0
		for i, letter in enumerate(original_message):
			length = len(encoding_dict[letter])
			end_index = start_index + length
			index_of_encoding = encoding_letter_to_index[letter]
			subset_encoded_message = encoded_message[start_index:end_index]
			position_of_message_rect = subset_encoded_message.get_center()
			position_of_encoding_rect = encoded_text[index_of_encoding].get_center()
			if i == 0:
				self.play(
					DrawBorderThenFill(surround_rect_message),
					ShowCreation(surround_rect_encoding),
				)
			else:
				new_surround_rect_message = SurroundingRectangle(
					encoded_message[start_index:end_index], buff=SMALL_BUFF * 0.5
				).set_color(GREEN_SCREEN).set_fill(color=GREEN_SCREEN, opacity=0.3)
				# make sure x-axis aligns on rectangles
				position_of_encoding_rect[0] = surround_rect_encoding.get_center()[0]
				self.play(
					Transform(surround_rect_message, new_surround_rect_message),
					surround_rect_encoding.move_to, position_of_encoding_rect,
				)

			self.play(
				TransformFromCopy(subset_encoded_message, decoded_message[i])
			)
			start_index = end_index
		self.wait()
		self.play(
			FadeOut(surround_rect_message),
			FadeOut(surround_rect_encoding),
		)
		self.wait()
		return decoded_message

class BalancingAct(ProblemFormulation):
	def construct(self):
		title = Title("Optimal Lossless Compression", scale_factor=1)
		self.play(
			Write(title[0]),
			Write(title[1])
		)

		diagram = self.show_comm_diagram_partial()

		encoding_dict = {
		'a': [0, 0, 0],
		'h': [0, 0, 1],
		'n': [0, 1, 0],
		'o': [0, 1, 1],
		'S': [1, 0, 0]
		}

		encoding_text, encoded_message, decoded_message = self.get_encoding_elements("Shannon", encoding_dict, diagram, "Shannon")

		self.play(
			FadeIn(diagram),
			FadeIn(encoding_text),
			FadeIn(encoded_message),
			FadeIn(decoded_message),
		)
		self.wait()

		scale = NumberLine(
			x_min=0, x_max=30,
			unit_size=0.4,
			numbers_with_elongated_ticks=[0, 5, 10, 15, 20, 25, 30],
			numbers_to_show=[0, 5, 10, 15, 20, 25, 30],
			include_numbers=True,
			number_at_center=15,
		)
		scale.move_to(UP * 0.8)


		scale_title = TextMobject("Encoded message length (bits)").scale(0.8)
		scale_title.next_to(title, DOWN)
		self.play(
			Write(scale_title)
		)
		self.play(
			Write(scale)
		)
		self.wait()

		marker = RegularPolygon(3, start_angle=-PI/2).scale(0.2).move_to(scale.n2p(21)).shift(UP * 0.6)
		marker.set_fill(color=BLUE, opacity=1)
		self.play(
			FadeIn(marker)
		)

		right_arrow = Arrow(scale.n2p(15.5) + UP * 0.3, scale.n2p(30) + UP * 0.3)
		right_arrow.set_color(YELLOW)
		redundancy = TextMobject("Redundancy").scale(0.7)
		redundancy.next_to(right_arrow, UP).to_edge(RIGHT * 2.5)
		self.play(
			Write(right_arrow),
			Write(redundancy),
		)
		self.wait()

		encoding_dict_loss = {
		'S': [1, 0],
		'h': [1, 1],
		'a': [0, 1],
		'n': [1],
		'o': [0]
		}

		loss_encoding_text, loss_encoded_message, loss_decoded_message = self.get_encoding_elements("Shannon", encoding_dict_loss, diagram, "nohaha")

		self.play(
			ReplacementTransform(encoding_text, loss_encoding_text),
			ReplacementTransform(encoded_message, loss_encoded_message),
			ReplacementTransform(decoded_message, loss_decoded_message),
			marker.move_to, scale.n2p(10) + UP * 0.6,
			run_time=2
		)
		self.wait()

		left_arrow = Arrow(scale.n2p(14.5) + UP * 0.3, scale.n2p(0) + UP * 0.3)
		left_arrow.set_color(RED)
		info_loss = TextMobject("Information loss").scale(0.7)
		info_loss.next_to(left_arrow, UP).to_edge(LEFT * 2.5)
		self.play(
			Write(left_arrow),
			Write(info_loss),
		)
		self.wait()

		encoding_dict_huff = {
		'S': [0, 0, 0],
		'o': [0, 0, 1],
		'a': [0, 1, 0],
		'h': [0, 1, 1],
		'n': [1],
		}

		optimal_encoding_text, optimal_encoded_message, optimal_decoded_message = self.get_encoding_elements("Shannon", encoding_dict_huff, diagram, "Shannon")

		self.play(
			ReplacementTransform(loss_encoding_text, optimal_encoding_text),
			ReplacementTransform(loss_encoded_message, optimal_encoded_message),
			ReplacementTransform(loss_decoded_message, optimal_decoded_message),
			marker.move_to, scale.n2p(15) + UP * 0.6,
			run_time=2
		)

		self.wait(2)

		limit_dashed_line = DashedLine(UP * 0.5, DOWN * 0.5, dash_length=0.1).move_to(scale.n2p(14.5))
		self.play(
			Write(limit_dashed_line)
		)
		self.wait()

		self.play(
			FadeOut(diagram),
			FadeOut(optimal_encoding_text),
			FadeOut(optimal_encoded_message),
			FadeOut(optimal_decoded_message),
		)
		self.wait()

		theoretical_limit_question = TextMobject("What is the theoretical limit on compression?")
		theoretical_limit_question.next_to(scale, DOWN * 2)

		self.play(
			Write(theoretical_limit_question)
		)
		self.wait() 

		

		shannon_img = ImageMobject("shannon").scale(1.2)
		shannon_img.next_to(theoretical_limit_question, DOWN)
		shannon_caption = TextMobject("Claude Shannon").scale(0.7)
		shannon_caption.next_to(shannon_img, DOWN)

		self.play(
			FadeIn(shannon_img),
			FadeIn(shannon_caption)
		)
		self.wait()

		down_arrow = TexMobject(r"\Downarrow")
		down_arrow.next_to(theoretical_limit_question, DOWN)
		info_theory = TextMobject("Information Theory")
		info_theory.next_to(down_arrow, DOWN)

		self.play(
			LaggedStart(
				FadeOut(shannon_img),
				FadeOut(shannon_caption),
				Write(down_arrow)
			)
		)
		self.wait()

		self.play(
			Write(info_theory)
		)
		self.wait()


	def show_comm_diagram_partial(self):
		encoder = self.make_component("Encoder", color=ORANGE)
		message = self.make_component("Message", color=RED)
		decoder = self.make_component("Decoder", color=GREEN_SCREEN)
		
		HORIZONTAL_SHIFT = 4.5
		VERTICAL_SHIFT = 0.9
		encoder.move_to(LEFT * HORIZONTAL_SHIFT + DOWN * VERTICAL_SHIFT)
		message.move_to(DOWN * VERTICAL_SHIFT)
		decoder.move_to(RIGHT * HORIZONTAL_SHIFT + DOWN * VERTICAL_SHIFT)
		
		arrows = []
		arrow_color = GRAY
		arrows.append(Arrow(encoder.get_right(), message.get_left()).set_color(arrow_color))
		arrows.append(Arrow(message.get_right(), decoder.get_left()).set_color(arrow_color))

		diagram = VGroup(
			encoder, arrows[0], message, arrows[1], decoder
		)
		return diagram

	def get_encoding_elements(self, original_string, encoding_dict, diagram, decoded_string):
		encoding_text, encoding_letter_to_index = self.get_example_encoding(diagram[0], encoding_dict)
		encoded_message = self.get_encoded_message(original_string, encoding_dict, diagram[2])
		decoded_message = TextMobject(decoded_string).scale(0.8).next_to(diagram[4], DOWN)
		return encoding_text, encoded_message, decoded_message

class IntroSelfInformation(ProblemFormulation):
	CONFIG = {
        "x_min": 0,
        "x_max": 1,
        "x_labeled_nums": [0, 1],
        "exclude_zero_label": False,
        "x_axis_label": "Probability",
        "x_axis_width": 6,
        "y_axis_height": 4,
        "graph_origin": DOWN * 3 + LEFT * 2.8,
        "y_min": 0,
        "y_max": 5,
        "y_axis_label": "Information",
    }
	def construct(self):
		source = self.make_component("Source")
		encoder = self.make_component("Encoder", color=ORANGE)
		message = self.make_component("Message", color=RED)
		decoder = self.make_component("Decoder", color=GREEN_SCREEN)
		receiver = self.make_component("Receiver", color=BLUE)

		HORIZONTAL_SHIFT = 5
		VERTICAL_SHIFT = 1.5

		source.move_to(LEFT * HORIZONTAL_SHIFT + UP * VERTICAL_SHIFT)
		encoder.move_to(LEFT * HORIZONTAL_SHIFT + DOWN * VERTICAL_SHIFT)
		message.move_to(DOWN * VERTICAL_SHIFT)
		decoder.move_to(RIGHT * HORIZONTAL_SHIFT + DOWN * VERTICAL_SHIFT)
		receiver.move_to(RIGHT * HORIZONTAL_SHIFT + UP * VERTICAL_SHIFT)

		network = VGroup(
			source, encoder, message, decoder, receiver,
		).shift(UP)

		arrows = []
		arrow_color = GRAY
		arrows.append(Arrow(source.get_bottom(), encoder.get_top()).set_color(arrow_color))
		arrows.append(Arrow(encoder.get_right(), message.get_left()).set_color(arrow_color))
		arrows.append(Arrow(message.get_right(), decoder.get_left()).set_color(arrow_color))
		arrows.append(Arrow(decoder.get_top(), receiver.get_bottom()).set_color(arrow_color))

		arrows = VGroup(*arrows)

		encoding_dict_huff = {
		'S': [0, 0, 0],
		'o': [0, 0, 1],
		'a': [0, 1, 0],
		'h': [0, 1, 1],
		'n': [1],
		}

		optimal_encoding_text, optimal_encoded_message, optimal_decoded_message = self.get_encoding_elements("Shannon", encoding_dict_huff, network, "Shannon")

		self.add(network, arrows)
		self.add(optimal_encoding_text, optimal_encoded_message, optimal_decoded_message)
		source_text = self.get_example_source("``Shannon''", source)
		self.add(source_text)
		receiver_text = source_text.copy().next_to(receiver, UP)
		self.add(receiver_text)
		self.wait()

		shannon_compressibility = TextMobject(
			"How much can we" + "\\\\", 
			"compress ``Shannon''?"
		)
		shannon_compressibility.scale(0.8).move_to(UP * 3)
		down_arrow = TexMobject(r"\Updownarrow").next_to(shannon_compressibility, DOWN)
		shannon_information = TextMobject(
			"How much information" + "\\\\", 
			"is in the message?").scale(0.8)
		shannon_information.next_to(down_arrow, DOWN)
		self.play(
			Write(shannon_compressibility)
		)
		self.wait()
		self.play(
			Write(down_arrow)
		)
		self.wait()
		self.play(
			Write(shannon_information)
		)
		self.wait()

		

		measuring_info = TextMobject(
			"How do we measure information?"
		).scale(1).move_to(DOWN * 3.2)
		measuring_info[0][-12:-1].set_color(YELLOW)
		self.play(
			shannon_information[0][-11:].set_color, YELLOW,
			FadeIn(measuring_info)
		)
		self.wait()

		self.clear()
		self.show_measuring_information()

	def get_encoding_elements(self, original_string, encoding_dict, diagram, decoded_string):
		encoding_text, encoding_letter_to_index = self.get_example_encoding(diagram[1], encoding_dict)
		encoded_message = self.get_encoded_message(original_string, encoding_dict, diagram[2])
		decoded_message = TextMobject(decoded_string).scale(0.8).next_to(diagram[3], DOWN)
		return encoding_text, encoded_message, decoded_message

	def show_measuring_information(self):
		measuring_info = Title("Measuring Information", scale_factor=1.2)
		self.play(
			Write(measuring_info[0]),
			Write(measuring_info[1])
		)
		self.wait()
		indent = LEFT * 6
		properties = TextMobject("Key Properties")
		properties.next_to(measuring_info, DOWN)
		self.play(
			Write(properties)
		)
		self.wait()

		properties_1 = TextMobject(r"1. Information and uncertainty are related.").scale(0.8)
		properties_1.next_to(properties, DOWN).to_edge(indent)
		self.play(
			Write(properties_1)
		)
		self.wait()

		self.clear()

		everest_statement = TextMobject("1. It's snowing on Mt. Everest.")
		sahara_statement = TextMobject("2. It's snowing in the Sahara desert.")

		everest_img = ImageMobject("Everest")
		sahara_img = ImageMobject("Sahara")
		everest_img.scale(1.3)
		sahara_img.scale(1.3)

		everest_img.move_to(UP * 2)
		everest_statement.next_to(everest_img, DOWN)
		sahara_img.move_to(DOWN * 1.5)
		sahara_statement.next_to(sahara_img, DOWN)

		self.play(
			FadeIn(everest_img),
			Write(everest_statement),
		)
		self.wait()

		self.play(
			FadeIn(sahara_img),
			Write(sahara_statement)
		)
		self.wait()

		self.play(
			ApplyWave(sahara_statement)
		)
		self.wait()
		left_shift = LEFT * 2.5
		self.play(
			everest_img.shift, left_shift,
			everest_statement.shift, left_shift,
			sahara_img.shift, left_shift,
			sahara_statement.shift, left_shift,
		)

		self.wait()

		right_shift = RIGHT * 5
		high_prob_low_info = TextMobject(
			"High probability event",
			r"$\Downarrow$",
			"Low information content",
		).scale(0.9).arrange(DOWN).next_to(everest_img, right_shift)

		low_prob_high_info = TextMobject(
			"Low probability event",
			r"$\Downarrow$",
			"High information content",
		).scale(0.9).arrange(DOWN).next_to(sahara_img, right_shift)

		self.play(
			FadeIn(high_prob_low_info)
		)
		self.wait()

		self.play(
			FadeIn(low_prob_high_info)
		)
		self.wait()

		self.clear()

		self.add(measuring_info, properties, properties_1)

		self.wait()

		new_properties_1 = TextMobject("1. Information and probability are inversely related").scale(0.8)
		new_properties_1.next_to(properties, DOWN)
		self.play(
			Transform(properties_1, new_properties_1)
		)
		self.wait()

		graph, I_x, P_x = self.show_info_vs_prob_graph(properties, properties_1)

		self.play(
			self.axes.shift, DOWN,
			graph.shift, DOWN,
			I_x.shift, DOWN,
			P_x.shift, DOWN,
		)
		self.wait()
			
		indent = LEFT * 3.4

		properties_2 = TextMobject(r"2. $I(x) \geq 0$, observing event $x$ never causes a loss of information.").scale(0.8)
		properties_2.next_to(properties_1, DOWN)
		properties_2.to_edge(indent)
		self.play(
			Write(properties_2)
		)
		self.wait()

		properties_3 = TextMobject(r"3. $P(x) = 1 \Rightarrow I(x) = 0$").scale(0.8)
		properties_3.next_to(properties_2, DOWN)
		properties_3.to_edge(indent)
		self.play(
			Write(properties_3)
		)
		self.wait()

		properties_4 = TextMobject("4. ", r"$P(x \cap y) = P(x) \cdot P(y)$ ", r"$\Rightarrow I(x \cap y) = I(x) + I(y)$").scale(0.8)
		properties_4.next_to(properties_3, DOWN)
		properties_4.to_edge(indent)
		self.play(
			FadeIn(properties_4)
		)
		self.wait()

		shift_right = RIGHT * 3
		brace = Brace(properties_4[1])
		independence = TextMobject(r"$x$ and $y$ are independent events").scale(0.6)
		independence.next_to(brace, DOWN)
		self.play(
			graph.shift, shift_right,
			self.axes.shift, shift_right,
			I_x.shift, shift_right,
			P_x.shift, shift_right,
			GrowFromCenter(brace),
			Write(independence),
			run_time=2
		)
		self.wait()

		self.play(
			FadeOut(independence),
			FadeOut(brace),
			FadeOut(self.axes),
			FadeOut(I_x),
			FadeOut(P_x),
			FadeOut(graph),
		)
		self.wait()

		set_of_functions = TextMobject("What set of functions satisfies these properties?").scale(0.8)
		set_of_functions.move_to(DOWN * 1.5)

		answer = TexMobject(r"I(x) = \log_b \left(\frac{1}{P(x)} \right)", r" = - \log_b P(x)")
		answer.move_to(set_of_functions.get_center())
		
		self.play(
			Write(set_of_functions)
		)
		self.play(
			ReplacementTransform(set_of_functions, answer)
		)
		self.wait()
		surround_rect = SurroundingRectangle(answer, buff=SMALL_BUFF)
		self_info = TextMobject("Self Information").set_color(YELLOW)
		self_info.next_to(surround_rect, DOWN * 1.5)
		self.play(
			ShowCreation(surround_rect),
			Write(self_info)
		)
		self.wait()

		answer_base_2 = TexMobject(r"I(x) = \log_2 \left(\frac{1}{P(x)} \right)", r" = - \log_2 P(x)").move_to(answer.get_center())
		self_info_with_units = TextMobject("Self Information (bits)").move_to(self_info.get_center()).set_color(YELLOW)
		self.play(
			ReplacementTransform(answer, answer_base_2),
			ReplacementTransform(self_info, self_info_with_units)
		)
		self.wait()

		shift_up = UP * 4

		self.play(
			FadeOut(measuring_info),
			FadeOut(properties),
			FadeOut(properties_1),
			FadeOut(properties_2),
			FadeOut(properties_3),
			FadeOut(properties_4),
			answer_base_2.shift, shift_up,
			surround_rect.shift, shift_up,
			self_info_with_units.shift, shift_up,
			run_time=2
		)
		self.wait()

		self_info_def = TextMobject(r"Measures information after observing event $x$ with probability $P(x)$").scale(0.8)
		self_info_def.next_to(self_info_with_units, DOWN)
		self.play(
			Write(self_info_def),
			run_time=2
		)

		dice_image = ImageMobject("dice").move_to(LEFT * 3 + DOWN * 1)
		coin_image = ImageMobject("heads").move_to(RIGHT * 3 + DOWN * 1)
		self.play(
			FadeIn(dice_image),
			FadeIn(coin_image)
		)
		self.wait()

		dice_example = TextMobject(
			r"$x$ - dice rolls 6",
			r"$P(x) = \frac{1}{6}$",
			r"$I(x) = 2.6$ bits",
		).scale(0.7).arrange(DOWN).next_to(dice_image, DOWN)

		coin_example = TextMobject(
			r"$x$ - coin lands heads",
			r"$P(x) = \frac{1}{2}$",
			r"$I(x) = 1.0$ bit",
		).scale(0.7).arrange(DOWN).next_to(coin_image, DOWN)

		self.play(
			FadeIn(dice_example),
			FadeIn(coin_example)
		)
		self.wait()

		limitations = TextMobject("Can we generalize this meausre to a distribution of possible events?").scale(0.8)
		limitations.next_to(self_info_def, DOWN)

		self.play(
			FadeOut(dice_example),
			FadeOut(coin_example),
			FadeOut(dice_image),
			FadeOut(coin_image),
			Write(limitations)
		)
		self.wait()

		los_angeles_image = ImageMobject("los_angeles").move_to(LEFT * 3 + DOWN * 1.8)
		seattle_image = ImageMobject("seattle").move_to(RIGHT * 3 + DOWN * 1.8)

		self.play(
			FadeIn(los_angeles_image),
			FadeIn(seattle_image),
		)

		left_question = TextMobject(
			"Information from knowing \\\\ weather in LA?"
		).scale(0.7).next_to(los_angeles_image, DOWN)

		right_question = TextMobject(
			"Information from knowing \\\\ weather in Seattle?"
		).scale(0.7).next_to(seattle_image, DOWN)

		self.play(
			FadeIn(left_question),
			FadeIn(right_question)
		)
		self.wait()

	def show_info_vs_prob_graph(self, key_properties, properties_1):
		self.setup_axes()
		self.x_axis.remove(self.x_axis[1])
		self.y_axis.remove(self.y_axis[1])
		self.x_axis_label_mob.scale(0.8)
		self.y_axis_label_mob.scale(0.8).next_to(self.y_axis, LEFT)
		self.play(
			Write(self.axes),
			run_time=2,
		)
		self.wait()

		graph = self.get_graph(lambda x: np.log(1 / x), x_min=0.008, x_max=1)
		self.play(
			ShowCreation(graph)
		)
		self.wait()

		everest_img = ImageMobject("Everest").scale(0.7).move_to(RIGHT * 2 + DOWN * 1.5)
		sahara_img = ImageMobject("Sahara").scale(0.7).move_to(LEFT * 1)

		surround_everest = SurroundingRectangle(everest_img, buff=0)
		surround_sahara = SurroundingRectangle(sahara_img, buff=0, color=GREEN_SCREEN)

		green_dot = Dot().set_color(GREEN_SCREEN).move_to(graph.point_from_proportion(0.01))
		yellow_dot = Dot().set_color(YELLOW).move_to(graph.point_from_proportion(4 / 5))

		self.play(
			GrowFromCenter(green_dot),
			GrowFromCenter(yellow_dot),
			FadeIn(everest_img),
			FadeIn(sahara_img),
			FadeIn(surround_everest),
			FadeIn(surround_sahara),
		)
		self.wait()

		I_x, P_x = TexMobject("I(x)").scale(0.8), TexMobject("P(x)").scale(0.8)
		I_x.next_to(self.y_axis_label_mob, DOWN)
		P_x.next_to(self.x_axis_label_mob, DOWN)
		self.play(
			Write(I_x),
			Write(P_x)
		)
		self.wait()
		I_x_scaled = I_x.copy().scale(0.7).shift(RIGHT * 1.3)
		P_x_scaled = P_x.copy().scale(0.7).shift(LEFT * 1.3 + UP * 0.5)

		full_properties_1 = TextMobject(r"1. Information $I(x)$ and probability $P(x)$ are inversely related").scale(0.8)
		full_properties_1.next_to(key_properties, DOWN)
		self.play(
			Transform(properties_1, full_properties_1)
		)
		self.wait()

		self.play(
			FadeOut(surround_sahara),
			FadeOut(surround_everest),
			FadeOut(green_dot),
			FadeOut(yellow_dot),
			FadeOut(everest_img),
			FadeOut(sahara_img),
			self.axes.scale, 0.7,
			graph.scale, 0.7,
			Transform(I_x, I_x_scaled),
			Transform(P_x, P_x_scaled),
		)

		
		return graph, I_x, P_x

class IntroEntropy(GraphScene):
	CONFIG = {
		"x_axis_width": 5.5,
		"y_axis_height": 4,
		"y_min": 0,
		"y_max": 1,
		"x_min": 0,
		"x_max": 1,
		"x_tick_frequency": 0.25,
		"y_tick_frequency": 0.25,
		"x_labeled_nums": [0.0, 1.0],
		"y_labeled_nums": [0.0, 1.0],
		"graph_origin": DOWN * 1.8 + RIGHT,
		"y_axis_label": "Entropy (bits)",
		"x_axis_label": "$P(H)$",
	}

	# Create two bar charts
	# One for weather in Seattle (S) (30% sunny (s), 50% rainy (r), 20% snowy (w))
	# One for weather in Phoenix (P) (90% sunny (s), 10% rainy (r), 0% snowy (w))
	# Make note about self information of 0 probability event is defined to be 0
	# Weighted average of I(s), I(r), and I(w)
	def construct(self):
		chart_seattle = BarChart([0.3, 0.5, 0.2], bar_names=[r"Sunny ($s$)", r"Rainy ($r$)", r"Snowy ($w$)"], width=8, height=3)
		chart_phoenix = BarChart([0.9, 0.1, 0.0], bar_names=[r"Sunny ($s$)", r"Rainy ($r$)", r"Snowy ($w$)"], width=8, height=3)

		chart_seattle.scale(0.8)
		chart_phoenix.scale(0.8)
		charts = VGroup(chart_seattle, chart_phoenix).arrange(DOWN * 4)
		
		charts[1].shift(UP * SMALL_BUFF)

		seattle_label = TextMobject(r"Weather in Seattle ($S$)").scale(0.9).next_to(chart_seattle, DOWN).shift(UP * 0.1 + RIGHT * 0.2)
		phoenix_label = TextMobject(r"Weather in Phoenix ($P$)").scale(0.9).next_to(chart_phoenix, DOWN).shift(UP * 0.1 + RIGHT * 0.2)

		self.play(
			FadeIn(charts[0]),
			FadeIn(seattle_label)
		)
		self.wait()

		self.play(
			FadeIn(charts[1]),
			FadeIn(phoenix_label)
		)
		self.wait()

		shift_left = LEFT * 3.5

		self.play(
			charts.shift, shift_left,
			seattle_label.shift, shift_left,
			phoenix_label.shift, shift_left,
		)

		self.wait()

		entropy_def = TextMobject(
			r"Given distribution $X$ with outcomes" + "\\\\",  
			"$x_1, x_2, \ldots , x_n$ with probabilities" + "\\\\",
			r"$P(x_1), P(x_2), \ldots , P(x_n)$",
		).scale(0.7)

		entropy_math = TextMobject(
			r"$H(X) = \sum_{i=1}^{n} P(x_i) \cdot I(x_i)$" + "\\\\"
			, r" $= \sum_{i=1}^{n} - P(x_i) \cdot \log_2 P(x_i)$"
		).scale(0.8)

		entropy = TextMobject(r"Information entropy $H(X)$").move_to(UP * 3.5)
		entropy[0][11:-4].set_color(YELLOW)
		

		entropy_def.move_to(RIGHT * 3.5 + UP * 2)
		entropy_math.next_to(entropy_def, DOWN).shift(LEFT * SMALL_BUFF * 2)
		entropy_math[1].shift(RIGHT * 1.15 + DOWN * 0.1)
		self.play(
			FadeIn(entropy_def),
		)
		self.play(
			Write(entropy_math[0]),
		)
		self.wait()
		self.play(
			Write(entropy_math[1])
		)
		self.wait()

		self.play(
			Write(entropy),
		)
		self.wait()


		H_S = TexMobject("H(S) = ", r"0.3 \cdot I(s) + 0.5 \cdot I(r) + 0.2 \cdot I(w)").scale(0.7).shift(LEFT)
		H_P = TexMobject("H(P) = ", r"0.9 \cdot I(s) + 0.1 \cdot I(r) + 0.0 \cdot I(w)").scale(0.7).shift(LEFT)

		text = [H_S[1], H_P[1]]
		for i, chart in enumerate(charts):
			index = 0
			for bar in chart.bars:
				color = bar.get_color()
				text[i][index:index + 3].set_color(color)
				index += 9

		
		H_P.next_to(charts[1], RIGHT)
		H_S.next_to(H_P, UP)

		note = TextMobject(r"*If $P(x_i) = 0$, we define $-P(x_i) \cdot \log_2 P(x_i) = 0$").scale(0.6)
		note.next_to(H_S, UP * 2)
		self.add(note)
		self.wait()

		self.play(
			FadeIn(H_S),
			FadeIn(H_P),
		)
		self.wait()

		H_S_val = TextMobject(r"$H(S) = 0.46$ bits").scale(0.7).move_to(H_S.get_center())
		H_P_val = TextMobject(r"$H(P) = 0.33$ bits").scale(0.7).move_to(H_P.get_center())

		self.play(
			ReplacementTransform(H_S, H_S_val),
			ReplacementTransform(H_P, H_P_val),
		)
		self.wait()

		conclusion = TextMobject(r"$H(S) > H(P)$").scale(0.8)
		conclusion.next_to(H_P_val, DOWN)
		down_arrow = TexMobject(r"\Downarrow").scale(0.8).next_to(conclusion, DOWN * 0.5)
		conclusion_cont = TextMobject(r"$S$ is more unpredictable than $P$").scale(0.8)
		conclusion_cont.next_to(down_arrow, DOWN * 0.5)

		self.play(
			Write(conclusion)
		)
		self.play(
			Write(down_arrow)
		)
		self.play(
			Write(conclusion_cont)
		)

		self.wait()

		self.clear()

		self.show_entropy_visualization()

	def show_entropy_visualization(self):
		coin_dist = BarChart([0.0, 1.0], bar_colors=[GREEN, RED], bar_names=["H", "T"]).shift(LEFT * 3)
		dist_title = TextMobject("Coin Toss Distribution").next_to(coin_dist, DOWN).shift(RIGHT * 0.5)

		
		def entropy(p):
			if p == 0 or p == 1:
				return 0
			return p * np.log2(1 / p) + (1 - p) * np.log2(1 / (1 - p))

		self.play(
			Write(coin_dist),
			Write(dist_title),
			run_time=2
		)
		self.wait()


		self.setup_axes()
		self.y_axis_label_mob.scale(0.8).next_to(self.y_axis, UP).shift(DOWN * SMALL_BUFF * 5)
		self.x_axis_label_mob.scale(0.8).next_to(self.x_axis, DOWN)

		self.play(
			Write(self.axes),
			run_time=2,
		)
		self.wait()

		graph = self.get_graph(entropy, x_max=0.5)

		dot = Dot().set_color(graph.get_color())
		dot.move_to(self.graph_origin)

		self.play(
			GrowFromCenter(dot),
		)

		self.wait()

		path = VMobject()
		self.add(path)
		def updater(mob):
			x_max = self.point_to_coords(dot.get_center())[0]
			new_path = self.get_graph(entropy, x_min=0, x_max=x_max).set_color(BLUE)
			path.become(new_path)
			coin_dist.change_bar_values([x_max, 1 - x_max], handle_zero=True)


		path.add_updater(updater)
		self.add(coin_dist)

		self.play(
			MoveAlongPath(dot, graph),
			run_time=5
		)
		self.wait()
		graph = self.get_graph(entropy, x_min=0.5, x_max=1.0).set_color(BLUE)
		self.play(
			MoveAlongPath(dot, graph),
			run_time=5
		)
		path.clear_updaters()
		coin_dist.clear_updaters()
		self.wait()

class NamingEntropy(Scene):
	def construct(self):
		title = Title("Why Call This Metric Entropy?", scale_factor=1.2)

		shannon_quote = TextMobject

		self.play(
			Write(title)
		)
		self.wait()

		

		shannon_quote =  TextMobject(
			"``My greatest concern was what to call it. I thought of" + "\\\\",
			"calling it ‘information’, but the word was overly used," + "\\\\",
			"so I decided to call it ‘uncertainty’. When I discussed" + "\\\\", 
			"it with John von Neumann, he had a better idea.''", 
		).scale(0.7)

		shannon_quote.next_to(title, DOWN)
		for text in shannon_quote:
			text.to_edge(LEFT * 2)


		shannon_img = ImageMobject("shannon").scale(1.2)
		shannon_img.next_to(title, DOWN).shift(RIGHT * 5)
		shannon_caption = TextMobject("- Claude Shannon").scale(0.7)
		shannon_caption.next_to(shannon_quote, DOWN)

		self.play(
			FadeIn(shannon_quote),
			FadeIn(shannon_img),
			FadeIn(shannon_caption)
		)
		self.wait()

		neumman_quote = TextMobject(
			"``You should call it entropy, for two reasons. In the first" + "\\\\", 
			"place your uncertainty function has been used in statistical" + "\\\\", 
			"mechanics under that name. In the second place, and more" + "\\\\",  
			"importantly, no one knows what entropy really is, so in a" + "\\\\", 
			"debate you will always have the advantage.''"
		).scale(0.7).next_to(shannon_caption, DOWN * 3)

		

		for text in neumman_quote:
			text.to_edge(LEFT * 2)

		neumman_caption = TextMobject("- John von Neumann").scale(0.7)

		neumman_caption.next_to(neumman_quote, DOWN)

		

		neumman_img = ImageMobject("neumann").scale(1.2)
		neumman_img.next_to(neumman_quote, RIGHT * 5, aligned_edge=UP)
		neumman_img.move_to(shannon_img.get_center()[0] * RIGHT + neumman_img.get_center()[1] * UP)
		self.play(
			FadeIn(neumman_quote),
			FadeIn(neumman_caption),
			FadeIn(neumman_img)
		)
		self.wait()

		self.play(
			neumman_quote[-2][12:].set_color, YELLOW, 
			neumman_quote[-1][:-1].set_color, YELLOW, 
		)
		self.wait()





class EntropyCompression(ProblemFormulation):
	def construct(self):
		diagram, ellipses = self.get_diagram_components()
		prob_dist_text, encoding_text = self.get_first_distrubution_and_encoding()
		# self.add(diagram, ellipses)
		source, encoder, receiver = diagram[0], diagram[1], diagram[2]

		self.play(
			Write(source)
		)

		self.play(
			Write(diagram[3])
		)

		self.play(
			Write(encoder)
		)

		self.play(
			Write(diagram[4])
		)

		self.play(
			FadeIn(ellipses)
		)

		self.play(
			Write(diagram[5])
		)

		self.play(
			Write(receiver)
		)
		self.wait()


		prob_dist_general = self.get_general_distribution()
		prob_dist_general.next_to(source, UP)
		self.play(
			FadeIn(prob_dist_general)
		)
		self.wait()

		sum_constraint = TexMobject(r"\sum_{i=1}^{M} p_i = 1").scale(0.8)
		sum_constraint.next_to(source, DOWN)
		self.play(
			Write(sum_constraint)
		)
		self.wait()

		prob_dist_text.next_to(source, UP)
		self.play(
			FadeOut(sum_constraint),
			ReplacementTransform(prob_dist_general, prob_dist_text)
		)
		self.wait()

		encoding_text.next_to(encoder, UP)
		self.play(
			FadeIn(encoding_text)
		)
		self.wait()

		average_length = TextMobject(r"Average Length per Symbol $L$")
		average_length.next_to(diagram, DOWN * 1.5)
		self.play(
			Write(average_length)
		)
		self.wait()
		binary_groups = [encoding_text[i][1].copy() for i in range(4)]
		def_1 = TextMobject(
			r"$L =$ ", 
			"0.25", r" $\cdot$ len("
		).scale(0.8)
		def_2 = TextMobject(r") $+$ ", 
			"0.25", r" $\cdot$ len(", ).scale(0.8) 
		def_3 = TextMobject(r") $+$ ",
			"0.25", r" $\cdot$ len(", ).scale(0.8)
		def_4 = TextMobject(r") $+$ ",
			"0.25", r" $\cdot$ len(").scale(0.8)
		def_5 = TextMobject(r")").scale(0.8)

		definition_components = [def_1, def_2, def_3, def_4]
		[definition[-1][-4:-1].set_color(MONOKAI_BLUE) for definition in definition_components]

		l_def = VGroup(
			def_1, binary_groups[0], def_2, binary_groups[1],
			def_3, binary_groups[2], def_4, binary_groups[3], def_5
		).arrange(RIGHT * 0.2)
		[l_def[i].shift(UP * SMALL_BUFF * 0.2) for i in range(len(l_def)) if i % 2 == 1]
		l_def.next_to(average_length, DOWN)
		
		first_transforms = []
		for i in range(len(prob_dist_text)):
			text = prob_dist_text[i]
			encoding_t = encoding_text[i]
			transform = TransformFromCopy(text[1], definition_components[i][1])
			binary_group_transform = TransformFromCopy(encoding_t[1], l_def[i * 2 + 1])
			first_transforms.append(transform)
			first_transforms.append(binary_group_transform)


		self.play(
			*first_transforms,
			*[FadeIn(definition[::2]) for i, definition in enumerate(l_def) if i % 2 == 0],
			run_time=2
		)
		self.wait()

		length_calls = [
		VGroup(def_1[-1][1:], l_def[1], def_2[0]),
		VGroup(def_2[-1][1:], l_def[3], def_3[0]),
		VGroup(def_3[-1][1:], l_def[5], def_4[0]),
		VGroup(def_4[-1][1:], l_def[7], def_5[0]),
		]

		simplified_l = TextMobject(
			r"$L =$ ",
			"0.25", r" $\cdot$ ", "1", r" $+$ ",
			"0.25", r" $\cdot$ ", "2", r" $+$ ",
			"0.25", r" $\cdot$ ", "3", r" $+$ ",
			"0.25", r" $\cdot$ ", "3",
		).scale(0.8).next_to(l_def, DOWN)

		self.play(
			*[FadeIn(simplified_l[i]) for i in range(len(simplified_l)) if i % 4 != 3],
			*[TransformFromCopy(length_calls[i], simplified_l[i * 4 + 3]) for i in range(len(length_calls))],
			run_time=2
		)
		self.wait()

		l_answer = TextMobject(r"$L = 2.25$ bits").scale(0.8)
		l_answer.next_to(simplified_l, DOWN)
		self.play(
			Write(l_answer)
		)
		self.wait()

		second_prob_dist = self.get_second_distribution()
		second_prob_dist.next_to(source, UP)
		self.play(
			ReplacementTransform(prob_dist_text, second_prob_dist)
		)
		self.wait()

		second_binary_groups = [encoding_text[i][1].copy() for i in range(4)]
		second_def_1 = TextMobject(
			r"$L =$ ", 
			"0.5", r" $\cdot$ len("
		).scale(0.8)
		second_def_2 = TextMobject(r") $+$ ", 
			"0.25", r" $\cdot$ len(", ).scale(0.8) 
		second_def_3 = TextMobject(r") $+$ ",
			"0.125", r" $\cdot$ len(", ).scale(0.8)
		second_def_4 = TextMobject(r") $+$ ",
			"0.125", r" $\cdot$ len(").scale(0.8)
		second_def_5 = TextMobject(r")").scale(0.8)
		second_l_def = VGroup(
			second_def_1, second_binary_groups[0], second_def_2, second_binary_groups[1],
			second_def_3, second_binary_groups[2], second_def_4, second_binary_groups[3], second_def_5
		).arrange(RIGHT * 0.2)
		[second_l_def[i].shift(UP * SMALL_BUFF * 0.2) for i in range(len(second_l_def)) if i % 2 == 1]
		second_l_def.next_to(average_length, DOWN)

		second_definition_components = [second_def_1, second_def_2, second_def_3, second_def_4]
		[definition[-1][-4:-1].set_color(MONOKAI_BLUE) for definition in second_definition_components]
		
		self.play(
			ReplacementTransform(l_def, second_l_def)
		)
		self.wait()

		second_simplified_l = TextMobject(
			r"$L =$ ",
			"0.5", r" $\cdot$ ", "1", r" $+$ ",
			"0.25", r" $\cdot$ ", "2", r" $+$ ",
			"0.125", r" $\cdot$ ", "3", r" $+$ ",
			"0.125", r" $\cdot$ ", "3",
		).scale(0.8).next_to(second_l_def, DOWN)

		self.play(
			ReplacementTransform(simplified_l, second_simplified_l)
		)

		self.wait()

		second_answer = TextMobject(r"$L = 1.75$ bits").scale(0.8)
		second_answer.next_to(second_simplified_l, DOWN)
		self.play(
			ReplacementTransform(l_answer, second_answer)
		)
		self.wait()


		
	def get_diagram_components(self):
		diagram = VGroup()
		source = self.make_component("Source")
		receiver = self.make_component("Receiver", color=BLUE)
		encoder = self.make_component("Encoder", color=ORANGE)
		source.move_to(LEFT * 4.5 + UP * 0.5)
		encoder.move_to(LEFT * 0 + UP * 0.5)
		ellipses = TexMobject(r"\ldots").move_to(RIGHT * 2.25 + UP * 0.5)
		receiver.move_to(RIGHT * 4.5 + UP * 0.5)
		
		arrow1 = Arrow(source.get_right(), encoder.get_left())
		arrow1.set_color(GRAY)
		arrow2 = Arrow(encoder.get_right(), ellipses.get_left())
		arrow2.set_color(GRAY)
		arrow3 = Arrow(ellipses.get_right(), receiver.get_left())
		arrow3.set_color(GRAY)
		diagram.add(source, encoder, receiver, arrow1, arrow2, arrow3)
		
		return diagram, ellipses

	def get_general_distribution(self):
		prob_dist = {
		r"$s_1$": r"$p_1$",
		r"$s_2$": r"$p_2$",
		r"$s_n$": r"$p_n$",
		}

		prob_dist_text = self.get_probability_dist_text(prob_dist, add_elipses=True)
		return prob_dist_text

	def get_first_distrubution_and_encoding(self):
		prob_dist = {
		'A': 0.25,
		'B': 0.25,
		'C': 0.25,
		'D': 0.25,
		}

		prob_dist_text = self.get_probability_dist_text(prob_dist)

		encoding_dict = {
		'A': [1],
		'B': [0, 1],
		'C': [0, 0, 0],
		'D': [0, 0, 1],
		}

		encoding_text = self.get_encoding(encoding_dict)

		return prob_dist_text, encoding_text

	def get_second_distribution(self):
		prob_dist = {
		'A': 0.5,
		'B': 0.25,
		'C': 0.125,
		'D': 0.125,
		}
		prob_dist_text = self.get_probability_dist_text(prob_dist)
		return prob_dist_text

	def get_encoding(self, encoding, scale=0.8):
		full_encoding = VGroup()
		for key in encoding:
			encoding_list = encoding[key]
			group = get_binary_group(len(encoding_list), binary_list=encoding_list)
			encoding_text = self.get_individual_encoding(key, group)
			full_encoding.add(encoding_text)
		full_encoding.arrange(DOWN)
		for element in full_encoding:
			element.to_edge(LEFT)
		return full_encoding.scale(scale)
		
	def get_probability_dist_text(self, dist, scale=0.8, add_elipses=False):
		distribution = []
		for key in dist:
			element = TextMobject(
				r"{0} $\rightarrow$ ".format(key), 
				"{0}".format(dist[key]))
			distribution.append(element)
		if add_elipses:
			distribution.insert(-1, TexMobject(r"\vdots"))
		distribution = VGroup(*distribution)
		distribution.arrange(DOWN)
		for element in distribution:
			element.to_edge(LEFT)
		return distribution.scale(scale)

class EntropyExample(EntropyCompression, ShannonFano):
	def construct(self):
		diagram = self.get_diagram_components()
		source, arrow, encoder = diagram
		source.move_to(UP * 0.5)
		self.play(
			Write(source)
		)
		distribution, encoding_text = self.get_first_distrubution_and_encoding()
		distribution.next_to(source, UP)
		self.play(
			FadeIn(distribution)
		)
		self.wait()

		lhs = TextMobject(r"$X =$ ").scale(0.8)
		left_brace = Brace(distribution, LEFT)
		lhs.next_to(left_brace, LEFT)
		self.play(
			Write(lhs),
			GrowFromCenter(left_brace),
			run_time=2
		)
		self.wait()

		message = self.cascade_sampling_distribution(source, distribution)

		brace_down = Brace(message, DOWN)
		self.play(
			GrowFromCenter(brace_down)
		)
		message_length = TextMobject(r"Message length $\rightarrow \infty$ ", r"$\Rightarrow$ Average length per symbol ", r"$L \geq H(X)$", r" $ = 2$ bits").scale(0.8)
		message_length.scale(1).next_to(brace_down, DOWN).shift(RIGHT * 0.8)
		self.play(
			Write(message_length[0])
		)
		self.wait()
		message_length[2].set_color(YELLOW)
		self.play(
			Write(message_length[1:-1])
		)
		self.wait()


		theorem = TextMobject("Shannon's source coding theorem").set_color(YELLOW).next_to(brace_down, DOWN * 4)
		self.play(
			Write(theorem)
		)
		self.wait(8)

		left_shift_amount = LEFT * 4.5 + DOWN
		self.play(
			FadeOut(theorem),
			FadeOut(message),
			FadeOut(message_length[0]),
			FadeOut(message_length[1]),
			FadeOut(brace_down),
			source.shift, left_shift_amount,
			lhs.shift, left_shift_amount,
			left_brace.shift, left_shift_amount,
			distribution.shift, left_shift_amount,
			message_length[2].move_to, left_shift_amount + DOWN + LEFT * 0.8,
			run_time=2,
		)
		self.wait()
		message_length[3].next_to(message_length[2], RIGHT, buff=SMALL_BUFF * 2).shift(UP * 0.05)
		self.play(
			Write(message_length[3])
		)
		self.wait()

		why_two_bits = TextMobject("Why 2 bits?").scale(0.8)
		up_brace = Brace(VGroup(message_length[2], message_length[3]), direction=DOWN)
		self.play(
			GrowFromCenter(up_brace)
		)

		why_two_bits.next_to(up_brace, DOWN)
		self.play(
			Write(why_two_bits)
		)
		self.wait()

		receiver = self.make_component("Receiver", color=BLUE)
		receiver.move_to(RIGHT * 2 + UP * source.get_center()[1])
		arrow = Arrow(source.get_right(), receiver.get_left())
		arrow.set_color(GRAY)
		self.play(
			Write(arrow)
		)
		self.play(
			Write(receiver)
		)
		self.wait()

		sent_message = TextMobject("``", "C", "''").scale(0.8)
		sent_message.next_to(receiver, UP)
		self.play(
			TransformFromCopy(distribution[2][0][0], sent_message[1]),
			FadeIn(sent_message[::2]),
			run_time=2
		)
		self.wait()

		receiver_perspective = TextMobject(
			"Given that the receiver knows" + "\\\\",
			"the distribution of symbols," + "\\\\",
			"what's the best way to determine" + "\\\\",
			"the exact symbol sent?"
		).scale(0.7)
		receiver_perspective.next_to(receiver, DOWN)

		self.play(
			FadeIn(receiver_perspective)
		)
		self.wait()


		# self.clear()

		question_diagram = self.get_two_bit_question_diagram().scale(0.9)
		question_diagram.move_to(RIGHT * 2 + UP * 0.5)
		nodes, edges = question_diagram
		self.play(
			FadeOut(receiver),
			FadeOut(receiver_perspective),
			FadeOut(sent_message),
			FadeOut(arrow),
			Write(nodes[0])
		)
		self.add_foreground_mobjects(nodes[0])
		self.wait()

		self.play(
			Write(edges[0]),
			Write(edges[1]),
		)
		self.wait()

		self.play(
			Write(nodes[1]),
			Write(nodes[2]),
		)

		self.add_foreground_mobjects(nodes[1], nodes[2])
		self.wait()

		self.play(
			*[Write(edges[i]) for i in range(2, len(edges))]
		)

		self.wait()

		self.play(
			*[Write(nodes[i]) for i in range(3, len(nodes))]
		)
		self.wait()

		transforms = []
		for i in range(len(edges)):
			text = edges[i][1]
			bit = Integer(1 - i % 2).scale(0.8)
			bit.move_to(text.get_center())
			transforms.append(Transform(text, bit))

		self.play(
			*transforms
		)
		self.wait()

		encodings = [[1, 1], [1, 0], [0, 1], [0, 0]]
		binary_encoding_mobs = [get_binary_encoding(lst) for lst in encodings]
		for i in range(len(binary_encoding_mobs)):
			binary_encoding_mobs[i].scale(0.7).next_to(nodes[i + 3], DOWN)

		self.play(
			*[Write(mob) for mob in binary_encoding_mobs]
		)
		self.wait()

		self.play(
			FadeOut(up_brace),
			FadeOut(why_two_bits)
		)
		self.wait()

		uneven_question_diagram, binary_encoding_mobs = self.show_uneven_distribution(
			source, distribution, message_length, 
			nodes, edges, binary_encoding_mobs,
			lhs, left_brace)


	def show_uneven_distribution(self, source, initial_dist, message_length, nodes, edges, encodings, lhs, left_brace):
		# self.play(
		# 	*[FadeOut(node) for node in nodes],
		# 	*[FadeOut(edge) for edge in edges],
		# 	FadeOut(message_length[2]),
		# 	FadeOut(message_length[3]),
		# 	FadeOut(initial_dist),
		# 	*[FadeOut(encoding) for encoding in encodings],
		# )
		# self.wait()
		original_message = VGroup(message_length[2].copy(), message_length[3].copy())
		original_dist = initial_dist.copy()
		next_dist = self.get_second_distribution()
		next_dist.move_to(initial_dist, aligned_edge=LEFT)
		self.play(
			ReplacementTransform(initial_dist, next_dist)
		)
		self.wait()

		new_entropy = TextMobject(r"$L \geq H(X)$", r" $=$ 1.75 bits").scale(0.8)
		new_entropy.next_to(source, DOWN * 2)
		new_entropy[0].set_color(YELLOW)

		new_entropy_c = new_entropy.copy()

		self.play(
			ReplacementTransform(message_length[2], new_entropy[0]),
			ReplacementTransform(message_length[3], new_entropy[1]),
		)

		self.wait()

		self.play(
			ReplacementTransform(next_dist, original_dist),
			ReplacementTransform(new_entropy[0], original_message[0]),
			ReplacementTransform(new_entropy[1], original_message[1]),
		)
		self.wait()

		TOP_BUFF = 2.4
		MIDDLE_BUFF = 1.7
		top_start = nodes[0].get_top() + UP * MED_SMALL_BUFF
		left_top_arrow = Arrow(top_start, top_start + LEFT * TOP_BUFF)
		right_top_arrow = Arrow(top_start, top_start + RIGHT * TOP_BUFF)

		left_top_arrow.set_color(YELLOW)
		right_top_arrow.set_color(YELLOW)
		

		left_top_text = TextMobject(r"$P(\cdot) = 0.5$").scale(0.7)
		right_top_text = TextMobject(r"$P(\cdot) = 0.5$").scale(0.7)
		left_top_text.next_to(left_top_arrow, UP)
		right_top_text.next_to(right_top_arrow, UP)

		self.play(
			Write(left_top_arrow),
			Write(right_top_arrow),
		)

		self.play(
			FadeIn(left_top_text),
			FadeIn(right_top_text)
		)

		self.wait()

		left_mid_start = nodes[1].get_top() + UP * SMALL_BUFF
		right_mid_start = nodes[2].get_top() + UP * SMALL_BUFF

		ll_mid_arrow = Arrow(left_mid_start, left_mid_start + LEFT * MIDDLE_BUFF)
		lr_mid_arrow = Arrow(left_mid_start, left_mid_start + RIGHT * MIDDLE_BUFF)

		rl_mid_arrow = Arrow(right_mid_start, right_mid_start + LEFT * MIDDLE_BUFF)
		rr_mid_arrow = Arrow(right_mid_start, right_mid_start + RIGHT * MIDDLE_BUFF)

		ll_mid_arrow.set_color(YELLOW)
		lr_mid_arrow.set_color(YELLOW)
		rl_mid_arrow.set_color(YELLOW)
		rr_mid_arrow.set_color(YELLOW)

		self.play(
			Write(ll_mid_arrow),
			Write(lr_mid_arrow),
			Write(rl_mid_arrow),
			Write(rr_mid_arrow),
		)

		quarter_probs = VGroup()
		text = TextMobject("0.25").scale(0.7)
		mid_arrows = [ll_mid_arrow, lr_mid_arrow, rl_mid_arrow, rr_mid_arrow]
		all_arrows = mid_arrows + [left_top_arrow, right_top_arrow]
		for arrow in mid_arrows:
			quarter_probs.add(text.copy().next_to(arrow, UP))

		self.play(
			FadeIn(quarter_probs)
		)
		self.wait()

		next_dist = self.get_second_distribution()
		next_dist.move_to(original_dist, aligned_edge=LEFT)
		self.play(
			ReplacementTransform(original_dist, next_dist),
			ReplacementTransform(original_message[0], new_entropy_c[0]),
			ReplacementTransform(original_message[1], new_entropy_c[1]),
			*[FadeOut(node) for node in nodes],
			*[FadeOut(edge) for edge in edges],
			*[FadeOut(encoding) for encoding in encodings],
			*[FadeOut(a) for a in all_arrows],
			FadeOut(quarter_probs),
			FadeOut(left_top_text),
			FadeOut(right_top_text),
		)
		self.wait()

		uneven_question_diagram = self.get_uneven_question_diagram()
		uneven_question_diagram.scale(0.9).move_to(RIGHT * 2)

		tree = self.show_shannon_fano(uneven_question_diagram)

		self.transform_tree_into_diagram(tree, uneven_question_diagram)

		transforms = []
		nodes, edges = uneven_question_diagram
		for i in range(len(edges)):
			text = edges[i][1]
			bit = Integer(i % 2).scale(0.8)
			bit.move_to(text.get_center())
			transforms.append(Transform(text, bit))

		self.play(
			*transforms
		)
		self.wait()

		encodings = [[1], [0, 1], [0, 0, 0], [0, 0, 1]]
		binary_encoding_mobs = [get_binary_encoding(lst) for lst in encodings]
		index_map = {0: 2, 1: 4, 2: 5, 3: 6}
		for i in range(len(binary_encoding_mobs)):
			binary_encoding_mobs[i].scale(0.7).next_to(nodes[index_map[i]], DOWN)

		self.play(
			*[Write(mob) for mob in binary_encoding_mobs]
		)
		self.wait()

		new_L = TextMobject(r"$L = $ 1.75 bits").scale(0.8).next_to(new_entropy_c, DOWN)
		self.play(
			Write(new_L)
		)
		self.wait()

		text_scale = 0.8
		a_len, a_len_answer = TextMobject("len(").scale(text_scale), TextMobject(")", r" $= \log_2(0.5)$").scale(text_scale)
		b_len, b_len_answer = TextMobject("len(").scale(text_scale), TextMobject(")", r" $= \log_2(0.25)$").scale(text_scale)
		c_len, c_len_answer = TextMobject("len(").scale(text_scale), TextMobject(")", r" $= \log_2(0.125)$").scale(text_scale)
		d_len, d_len_answer = TextMobject("len(").scale(text_scale), TextMobject(")", r" $= \log_2(0.125)$").scale(text_scale)

		a_len[0][:3].set_color(MONOKAI_BLUE)
		b_len[0][:3].set_color(MONOKAI_BLUE)
		c_len[0][:3].set_color(MONOKAI_BLUE)
		d_len[0][:3].set_color(MONOKAI_BLUE)

		indent = LEFT * 5.5
		shift_up = UP * 1.5 + LEFT * 1
		a_length_group = VGroup(a_len, binary_encoding_mobs[0].copy().scale(0.8 / 0.7), a_len_answer).arrange(RIGHT * 0.2)
		a_length_group.next_to(next_dist[0], RIGHT * 2).to_edge(indent)

		b_length_group = VGroup(b_len, binary_encoding_mobs[1].copy().scale(0.8 / 0.7), b_len_answer).arrange(RIGHT * 0.2)
		b_length_group.next_to(next_dist[1], RIGHT).to_edge(indent)

		c_length_group = VGroup(c_len, binary_encoding_mobs[2].copy().scale(0.8 / 0.7), c_len_answer).arrange(RIGHT * 0.2)
		c_length_group.next_to(next_dist[2], RIGHT).to_edge(indent)

		d_length_group = VGroup(d_len, binary_encoding_mobs[3].copy().scale(0.8 / 0.7), d_len_answer).arrange(RIGHT * 0.2)
		d_length_group.next_to(next_dist[3], RIGHT).to_edge(indent)

		# need to make this look better (fade out source, L related text and center distribution and lengths in top left open area)

		length_groups = [a_length_group, b_length_group, c_length_group, d_length_group]
		transforms = []
		for i, group in enumerate(length_groups):
			group.shift(shift_up).shift(RIGHT * 2)
			transforms.append(FadeIn(group[0]))
			transforms.append(FadeIn(group[2][0]))
			transforms.append(TransformFromCopy(binary_encoding_mobs[i], group[1]))
		self.play(
			*transforms,
			FadeOut(lhs),
			FadeOut(left_brace),
			FadeOut(new_L),
			FadeOut(new_entropy_c),
			FadeOut(source),
			next_dist.shift, shift_up,
			run_time=3,
		)
		self.wait()

		self.play(
			*[Write(group[2][1]) for group in length_groups]
		)
		self.wait()

		fun_fact = TextMobject(
			"Fun fact: distributions with this property" + "\\\\",  
			"are called dyadic."
		)
		fun_fact.scale(0.8).next_to(next_dist, DOWN * 2, aligned_edge=LEFT)
		fun_fact[1][-7:-1].set_color(YELLOW)
		fun_fact[1].to_edge(LEFT * 1.6)
		self.play(
			Write(fun_fact),
			run_time=2
		)
		self.wait()

		dyadic_result = TextMobject(r"For dyadic distributions:").scale(0.8)
		dyadic_result[0][3:9].set_color(YELLOW)
		dyadic_result.next_to(fun_fact[1], DOWN, aligned_edge=LEFT)
		self.play(
			Write(dyadic_result)
		)
		self.wait()

		L_H_X_eq = TextMobject(r"$L = H(X)$").next_to(dyadic_result, DOWN * 2)
		surround_rect = SurroundingRectangle(L_H_X_eq, buff=MED_SMALL_BUFF)
		self.play(
			Write(L_H_X_eq)
		)
		self.play(
			ShowCreation(surround_rect)
		)
		self.wait()

		general_dist = TextMobject(
			"Can we apply this approach" + "\\\\",
			"to general distributions?"	
		).scale(0.7)
		general_dist.move_to(DOWN * 2.8 + RIGHT * 4.5)
		
		self.play(
			Write(general_dist)
		)
		self.wait()

		return uneven_question_diagram, binary_encoding_mobs

	def show_shannon_fano(self, question_diagram):
		dist = {
		'A': 0.5,
		'B': 0.25,
		'C': 0.125,
		'D': 0.125,
		}
		self.position_map = self.get_position_map(question_diagram[0])
		self.left_index_map = {
		1: 2,
		2: 4,
		4: 6,
		}

		self.right_index_map = {
		1: 3,
		2: 5,
		4: 7,
		}
		self.dashed_lines = {}
		self.edge_dict = {}
		nodes = self.generate_example_dist(dist)
		self.show_nodes([nodes])
		sorted_nodes = self.sort_nodes(nodes)

		# array representation of tree
		# 0th index is 0
		# all other indices are list of nodes that will later be transformed to single nodes
		# with this representation, index i is a node, i * 2 is left node, i * 2 + 1 is right node
		tree = [0] * (len(self.position_map) + 1)
		tree[1] = sorted_nodes
		self.animate_shannon_fano(sorted_nodes, tree, 1)
		return tree
	
	def transform_tree_into_diagram(self, tree, diagram):
		nodes, edges = diagram
		edge_index_map = {0: [0, 1], 1: [2, 3], 3: [4, 5]}
		
		all_transforms = []
		for i in range(1, len(tree)):
			transforms = []
			diagram_index = i - 1
			tree_nodes = tree[i]
			tree_key = self.get_key(tree_nodes)
			tree_node_mob_group = VGroup(*[node.mob for node in tree_nodes])
			if tree_key in self.dashed_lines:
				transforms.extend(
					[
					ReplacementTransform(tree_node_mob_group, nodes[diagram_index]),
					FadeOut(self.dashed_lines[tree_key]),
					]
				)
			else:
				transforms.append(
					ReplacementTransform(tree_node_mob_group, nodes[diagram_index])
				)

			if diagram_index in edge_index_map:
				transforms.extend(
					[
					Write(edges[j]) for j in edge_index_map[diagram_index]
					]
				)
			all_transforms.extend(transforms)

		self.play(
			LaggedStart(*all_transforms),
			run_time=4
		)
		self.wait()



	def get_position_map(self, nodes):
		down_buff = nodes[2].get_center()[1] - nodes[1].get_center()[1]
		position_map = {
		'ABCD': nodes[0].get_center(),
		'CDB': nodes[1].get_center(),
		'A': nodes[2].get_center() + DOWN * down_buff,
		'CD': nodes[3].get_center(),
		'B': nodes[4].get_center() + DOWN * down_buff,
		'C': nodes[5].get_center() + DOWN * SMALL_BUFF * 2,
		'D': nodes[6].get_center() + DOWN * SMALL_BUFF * 2,
		}
		return position_map

	def get_left_index(self, index):
		return self.left_index_map[index]

	def get_right_index(self, index):
		return self.right_index_map[index]

	def get_diagram_components(self):
		source = self.make_component("Source")
		encoder = self.make_component("Encoder", color=ORANGE)
		arrow = Arrow(source.get_right(), encoder.get_left())

		diagram = VGroup(source, arrow, encoder).arrange(RIGHT)
		return diagram

	def cascade_sampling_distribution(self, source, distribution):
		alphabet = ['A', 'B', 'C', 'D']
		message = [np.random.choice(alphabet) for _ in range(30)] + [r"$\ldots$"]
		message_mob = TextMobject(*message)
		message_mob.next_to(source, DOWN * 2)
		
		transforms = [TransformFromCopy(distribution[alphabet.index(char)][0][0], message_mob[i]) for i, char in enumerate(message[:-1])]
		final_animation = [FadeIn(message_mob[-1])]
		self.play(
			LaggedStart(*transforms + final_animation),
			run_time=4
		)
		self.wait()

		return message_mob

	def get_two_bit_question_diagram(self):
		root = self.make_node(r"Is $X = $ A" + "\\\\" + r"or $X = $ B?")
		left_node = self.make_node(r"Is $X = $ A?")
		right_node = self.make_node(r"Is $X = $ C?")
		left_node.next_to(root, DL + LEFT)
		right_node.next_to(root, DR + RIGHT)
		ll_leaf = self.make_leaf("A").next_to(left_node, DL)
		lr_leaf = self.make_leaf("B").next_to(left_node, DR)
		rl_leaf = self.make_leaf("C").next_to(right_node, DL)
		rr_leaf = self.make_leaf("D").next_to(right_node, DR)

		root_to_left = self.get_edge_between_nodes(root, left_node)
		root_to_right = self.get_edge_between_nodes(root, right_node)
		left_to_ll = self.get_edge_between_nodes(left_node, ll_leaf)
		left_to_lr = self.get_edge_between_nodes(left_node, lr_leaf)
		right_to_rl = self.get_edge_between_nodes(right_node, rl_leaf)
		right_to_rr = self.get_edge_between_nodes(right_node, rr_leaf)


		nodes = VGroup(root, left_node, right_node, ll_leaf, lr_leaf, rl_leaf, rr_leaf)
		edges = VGroup(root_to_left, root_to_right, left_to_ll, left_to_lr, right_to_rl, right_to_rr)

		return VGroup(nodes, edges)

	def get_uneven_question_diagram(self):
		nodes = VGroup()
		nodes.add(self.make_node(r"Is $X = $ A?"))
		nodes.add(self.make_node(r"Is $X = $ B?").next_to(nodes[0], DL + LEFT))
		nodes.add(self.make_leaf("A").next_to(nodes[0], DR + RIGHT))
		
		nodes.add(self.make_node(r"Is $X = $ D?").next_to(nodes[1], DL + LEFT))
		nodes.add(self.make_leaf("B").next_to(nodes[1], DR + RIGHT))

		nodes.add(self.make_leaf("C").next_to(nodes[3], DL + LEFT))
		nodes.add(self.make_leaf("D").next_to(nodes[3], DR + RIGHT))

		root_to_left = self.get_edge_between_nodes(nodes[0], nodes[1], colors=[GREEN_SCREEN, BRIGHT_RED], strings=["Yes", "No"])
		root_to_right = self.get_edge_between_nodes(nodes[0], nodes[2], colors=[GREEN_SCREEN, BRIGHT_RED], strings=["Yes", "No"])
		left_to_ll = self.get_edge_between_nodes(nodes[1], nodes[3], colors=[GREEN_SCREEN, BRIGHT_RED], strings=["Yes", "No"])
		left_to_lr = self.get_edge_between_nodes(nodes[1], nodes[4], colors=[GREEN_SCREEN, BRIGHT_RED], strings=["Yes", "No"])
		right_to_rl = self.get_edge_between_nodes(nodes[3], nodes[5], colors=[GREEN_SCREEN, BRIGHT_RED], strings=["Yes", "No"])
		right_to_rr = self.get_edge_between_nodes(nodes[3], nodes[6], colors=[GREEN_SCREEN, BRIGHT_RED], strings=["Yes", "No"])

		edges = VGroup(root_to_left, root_to_right, left_to_ll, left_to_lr, right_to_rl, right_to_rr)
		return VGroup(nodes, edges)





	def make_node(self, question):
		node = RegularPolygon(n=4).scale(1)
		text = TextMobject(question).scale(0.58)
		text.move_to(node.get_center())
		return VGroup(node, text)

	def make_leaf(self, character):
		node = RoundedRectangle().scale(0.25)
		text = TextMobject(character).scale(0.58)
		text.move_to(node.get_center())
		return VGroup(node, text)

	def get_edge_between_nodes(self, parent, child, 
		colors=[BRIGHT_RED, GREEN_SCREEN], strings=["No", "Yes"]):
		left_anchor = parent.get_left()
		right_anchor = parent.get_right()
		bottom_anchor = child.get_top()

		top_anchor = right_anchor
		text = TextMobject(strings[0]).scale(0.7)
		color = colors[0]
		if np.linalg.norm(left_anchor - bottom_anchor) < np.linalg.norm(right_anchor - bottom_anchor):
			top_anchor = left_anchor
			text = TextMobject(strings[1]).scale(0.7)
			color = colors[1]

		middle_anchor = bottom_anchor[0] * RIGHT + top_anchor[1] * UP
		first_line = Line(top_anchor, middle_anchor)
		text.next_to(first_line, UP)
		second_line = Line(middle_anchor, bottom_anchor)
		edge = VGroup(first_line, second_line).set_color(color)
		return VGroup(edge, text)

class Patreons(Scene):
    def construct(self):
        thanks = TextMobject("Special Thanks to Patreon Burt Humburg").scale(1.2)
        thanks.move_to(DOWN * 2.5)
        thanks[0][-11:].set_color(YELLOW)
        self.play(
        	Write(thanks)
        )
        self.wait(7)

### Various utility functions
def get_random_binary_string(length):
	return list(np.random.choice([0, 1], size=(length,)))

def get_binary_group(length, binary_list=None, buff=SMALL_BUFF*1.7):
	if not binary_list:
		binary_list = get_random_binary_string(length)
	return VGroup(*[Integer(bit) for bit in binary_list]).arrange(RIGHT, buff=buff)

def get_binary_encoding(binary_list, buff=SMALL_BUFF*1.7):
	return VGroup(*[Integer(bit) for bit in binary_list]).arrange(RIGHT, buff=buff)

def get_normal_towards_origin(a, b, away=False):
    x1, y1 = a[0], a[1]
    x2, y2 = b[0], b[1]
    dx = x2 - x1
    dy = y2 - y1
    n1 = np.array([-dy, dx, 0])
    n2 = np.array([dy, -dx, 0])
    AO = ORIGIN - a
    if np.dot(AO, n1) > 0:
        if away:
            return n2
        return n1
    if away:
        return n1
    return n2


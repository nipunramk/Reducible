from manimlib.imports import *
import random
np.random.seed(0)
import scipy.io.wavfile as wf

class GraphNode:
	def __init__(self, data, position=ORIGIN, radius=0.5, neighbors=[], scale=1):
		self.char = data
		self.data = TextMobject(str(data))
		self.data.scale(scale)
		self.neighbors = []
		self.center = position
		self.radius = radius
		self.circle = Circle(radius=radius)
		self.circle.move_to(position)
		self.data.move_to(position)
		self.drawn = False
		self.marked = False
		self.edges = []
		self.prev = None

	def connect(self, other):
		line_center = Line(self.center, other.center)
		unit_vector = line_center.get_unit_vector()
		start, end = line_center.get_start_and_end()
		new_start = start + unit_vector * self.radius
		new_end = end - unit_vector * self.radius
		line = Line(new_start, new_end)
		self.neighbors.append(other)
		other.neighbors.append(self)
		self.edges.append(line)
		other.edges.append(line)
		return line

	def connect_arrow(self, other):
		line_center = Line(self.center, other.center)
		unit_vector = line_center.get_unit_vector()
		start, end = line_center.get_start_and_end()
		new_start = start + unit_vector * self.radius / 2
		new_end = end - unit_vector * self.radius / 2
		arrow = Arrow(new_start, new_end)
		arrow.buff = self.radius / 2
		arrow.unit_vector = unit_vector
		self.neighbors.append(other)
		self.edges.append(arrow)
		return arrow

	def connect_curve(self, other, direction=DOWN, angle=TAU / 4):
		curve_start = self.circle.get_center() + direction * self.radius
		curve_end = other.circle.get_center() + direction * self.radius
		line = ArcBetweenPoints(curve_start, curve_end, angle=angle)
		self.neighbors.append(other)
		other.neighbors.append(self)
		self.edges.append(line)
		other.edges.append(line)
		return line

	def __repr__(self):
		return 'GraphNode({0})'.format(self.char)

	def __str__(self):
		return 'GraphNode({0})'.format(self.char)

class GraphAnimationUtils(Scene):
	def construct(self):
		graph, edge_dict = self.create_bfs_graph()
		nodes, edges = self.make_graph_mobject(graph, edge_dict)
		entire_graph = VGroup(nodes, edges)

		self.play(
			FadeIn(entire_graph)
		)

		bfs_full_order = bfs(graph, 0)
		wait_times = [0] * len(bfs_full_order)
		wait_time_dict = {}
		for i in range(len(graph)):
			wait_time_dict[i] = 0

		highlights = self.show_bfs(graph, edge_dict, bfs_full_order, wait_times)


	def show_bfs(self, graph, edge_dict, full_order, 
		wait_times, scale_factor=1, run_time=1):
		i = 0
		angle = 180
		all_highlights = []
		for element in full_order:
			if isinstance(element, int):
				surround_circle = self.highlight_node(graph, element, 
					start_angle=angle/360 * TAU, scale_factor=scale_factor, run_time=run_time)
				all_highlights.append(surround_circle)
			else:
				last_edge = self.sharpie_edge(edge_dict, element[0], element[1], 
					scale_factor=scale_factor, run_time=run_time)
				angle = self.find_angle_of_intersection(graph, last_edge.get_end(), element[1])
				all_highlights.append(last_edge)
			self.wait(wait_times[i])
			i += 1
		return all_highlights

	def show_second_bfs(self, graph, edge_dict, full_order, order,
		wait_times, scale_factor=1, run_time=1):
		i = 0
		angle = 180
		new_highlights = []
		order_index = 0
		for element in full_order:
			if isinstance(element, int):
				surround_circle = self.highlight_node(graph, element, 
					start_angle=angle/360 * TAU, scale_factor=scale_factor, run_time=run_time)
				self.play(
					TransformFromCopy(graph[element].data, order[order_index])
				)
				order_index += 1
				new_highlights.append(surround_circle)
				graph[element].surround_circle = surround_circle
				self.wait(wait_times[element])
			else:
				last_edge = self.sharpie_edge(edge_dict, element[0], element[1], 
					scale_factor=scale_factor, run_time=run_time)
				angle = self.find_angle_of_intersection(graph, last_edge.get_end(), element[1])
				new_highlights.append(last_edge)
			
			i += 1

		return new_highlights

	def show_full_bfs_animation(self, graph, edge_dict, full_order, order,
		wait_times, wait_time_dict, scale_factor=1, run_time=1):
		i = 0
		angle = 180
		surround_circles = [0] * len(graph)
		order_index = 0
		new_highlights = []
		for element in full_order:
			if isinstance(element, int):
				surround_circle = self.highlight_node(graph, element, 
					start_angle=angle/360 * TAU, scale_factor=scale_factor, run_time=run_time)
				# print(type(graph[element].data), type(order[order_index]))
				self.play(
					TransformFromCopy(graph[element].data, order[order_index])
				)
				order_index += 1
				self.indicate_neighbors(graph, element, wait_time_dict)
				graph[element].surround_circle = surround_circle
				new_highlights.append(surround_circle)
				self.wait(wait_times[element])
			else:
				last_edge = self.sharpie_edge(edge_dict, element[0], element[1], 
					scale_factor=scale_factor, run_time=run_time)
				angle = self.find_angle_of_intersection(graph, last_edge.get_end(), element[1])
				new_highlights.append(last_edge)
			
			i += 1

		return new_highlights

	def indicate_neighbors(self, graph, i, wait_time_dict):
		current_node = graph[i]
		neighbors = current_node.neighbors
		self.wait(wait_time_dict[i])
		self.play(
			*[CircleIndicate(neighbor.circle) for neighbor in neighbors],
			run_time=2
		)

	def create_small_graph(self):
		graph = []
		edges = {}

		radius, scale = 0.4, 0.9
		node_0 = GraphNode('0', position=DOWN * 1 + LEFT * 3, radius=radius, scale=scale)
		node_1 = GraphNode('1', position=UP * 1 + LEFT, radius=radius, scale=scale)
		node_2 = GraphNode('2', position=DOWN * 3 + LEFT, radius=radius, scale=scale)
		node_3 = GraphNode('3', position=DOWN * 1 + RIGHT, radius=radius, scale=scale)
		node_4 = GraphNode('4', position=DOWN * 1 + RIGHT * 3, radius=radius, scale=scale)


		edges[(0, 1)] = node_0.connect(node_1)
		edges[(0, 3)] = node_0.connect(node_3)
		edges[(0, 2)] = node_0.connect(node_2)

		edges[(1, 3)] = node_1.connect(node_3)

		edges[(2, 3)] = node_2.connect(node_3)

		edges[(3, 4)] = node_3.connect(node_4)

		graph.append(node_0)
		graph.append(node_1)
		graph.append(node_2)
		graph.append(node_3)
		graph.append(node_4)

		return graph, edges

	def create_bfs_graph(self):
		graph = []
		edges = {}

		radius, scale = 0.4, 0.9
		SHIFT = RIGHT * 2.5
		node_0 = GraphNode('0', position=LEFT * 5, radius=radius, scale=scale)
		node_1 = GraphNode('1', position=LEFT * 3 + UP * 2, radius=radius, scale=scale)
		node_2 = GraphNode('2', position=LEFT * 3 + DOWN * 2, radius=radius, scale=scale)
		node_3 = GraphNode('3', position=LEFT * 1, radius=radius, scale=scale)
		node_4 = GraphNode('4', position=LEFT * 1 + UP * 2, radius=radius, scale=scale)
		node_5 = GraphNode('5', position=RIGHT * 1, radius=radius, scale=scale)
		node_6 = GraphNode('6', position=LEFT * 1 + DOWN * 2, radius=radius, scale=scale)
		node_7 = GraphNode('7', position=RIGHT * 3 + DOWN * 2, radius=radius, scale=scale)
		node_8 = GraphNode('8', position=RIGHT * 3 + UP * 2, radius=radius, scale=scale)
		node_9 = GraphNode('9', position=RIGHT * 5 + UP * 2, radius=radius, scale=scale)

		edges[(0, 1)] = node_0.connect(node_1)
		edges[(0, 2)] = node_0.connect(node_2)
		
		edges[(1, 2)] = node_1.connect(node_2)
		edges[(1, 3)] = node_1.connect(node_3)
		edges[(1, 4)] = node_1.connect(node_4)

		edges[(3, 5)] = node_3.connect(node_5)

		edges[(5, 6)] = node_5.connect(node_6)
		edges[(5, 7)] = node_5.connect(node_7)
		edges[(5, 8)] = node_5.connect(node_8)

		edges[(7, 8)] = node_7.connect(node_8)

		edges[(8, 9)] = node_8.connect(node_9)

		graph.append(node_0)
		graph.append(node_1)
		graph.append(node_2)
		graph.append(node_3)
		graph.append(node_4)
		graph.append(node_5)
		graph.append(node_6)
		graph.append(node_7)
		graph.append(node_8)
		graph.append(node_9)

		return graph, edges

	def show_path(self, graph, edge_dict, path, scale_factor=1):
		angle = 180
		objects = []
		for i in range(len(path) - 1):
			u, v = path[i], path[i + 1]
			surround_circle = self.highlight_node(graph, u, start_angle=angle/360 * TAU, scale_factor=scale_factor)
			last_edge = self.sharpie_edge(edge_dict, u, v, scale_factor=scale_factor)
			objects.extend([surround_circle, last_edge])
			angle = self.find_angle_of_intersection(graph, last_edge.get_end(), v)

		if v != path[0]:
			surround_circle = self.highlight_node(graph, v, start_angle=angle/360 * TAU, scale_factor=scale_factor)
			objects.append(surround_circle)

		return objects

	def find_angle_of_intersection(self, graph, last_point, node_index):
		node = graph[node_index]
		distances = []
		for angle in range(360):
			respective_line = Line(node.circle.get_center(), 
				node.circle.get_center() + RIGHT * node.circle.radius)
			rotate_angle = angle / 360 * TAU
			respective_line.rotate(rotate_angle, about_point=node.circle.get_center())
			end_point = respective_line.get_end()
			distance = np.linalg.norm(end_point - last_point)
			distances.append(distance)
		return np.argmin(np.array(distances))

	def sharpie_edge(self, edge_dict, u, v, color=GREEN_SCREEN, 
		scale_factor=1, animate=True, run_time=1):
		switch = False
		if u > v:
			edge = edge_dict[(v, u)]
			switch = True
		else:
			edge = edge_dict[(u, v)]
		
		if not switch:
			line = Line(edge.get_start(), edge.get_end())
		else:
			line = Line(edge.get_end(), edge.get_start())
		line.set_stroke(width=12 * scale_factor)
		line.set_color(color)
		if animate:
			self.play(
				ShowCreation(line),
				run_time=run_time
			)
		return line

	def highlight_node(self, graph, index, color=GREEN_SCREEN, 
		start_angle=TAU/2, scale_factor=1, animate=True, run_time=1):
		node = graph[index]
		surround_circle = Circle(radius=node.circle.radius * scale_factor, start_angle=start_angle, TAU=-TAU)
		surround_circle.move_to(node.circle.get_center())
		# surround_circle.scale(1.15)
		surround_circle.set_stroke(width=8 * scale_factor)
		surround_circle.set_color(color)
		surround_circle.set_fill(opacity=0)
		if animate:
			self.play(
				ShowCreation(surround_circle),
				run_time=run_time
			)
		return surround_circle

	def make_graph_mobject(self, graph, edge_dict, node_color=DARK_BLUE_B, 
		stroke_color=BLUE, data_color=WHITE, edge_color=GRAY, scale_factor=1,
		show_data=True):
		nodes = []
		edges = []
		for node in graph:
			node.circle.set_fill(color=node_color, opacity=0.5)
			node.circle.set_stroke(color=stroke_color)
			node.data.set_color(color=data_color)
			if show_data:
				nodes.append(VGroup(node.circle, node.data))
			else:
				nodes.append(node.circle)

		for edge in edge_dict.values():
			edge.set_stroke(width=7*scale_factor)
			edge.set_color(color=edge_color)
			edges.append(edge)
		return VGroup(*nodes), edges

class Introduction(Scene):
	def construct(self):
		earth = Earth()
		algorithms = TextMobject("Algorithms").scale(1.2)
		algorithms.next_to(earth, UP)
		self.play(
			GrowFromCenter(earth),
			GrowFromCenter(algorithms)
		)

		self.wait()

		self.introduce_classes(earth, algorithms)

	def introduce_classes(self, earth, algorithms):
		utility_class = ScreenRectangle()
		utility_class.set_width(4.5)
		utility_class.set_height(3.5)

		beauty_class = ScreenRectangle()
		beauty_class.set_width(4.5)
		beauty_class.set_height(3.5)

		classes = VGroup(utility_class, beauty_class).arrange_submobjects(RIGHT, buff=0.5)

		self.play(
			earth.scale, 1.5,
			FadeOut(algorithms),
			run_time=2,
		)

		self.play(
			ReplacementTransform(earth, classes),
			run_time=2
		)

		self.wait(2)

		utility = TextMobject("Utility")
		utility.next_to(utility_class, UP)
		self.play(
			Write(utility)
		)
		self.wait(12)


		beauty = TextMobject("Beauty")
		beauty.next_to(beauty_class, UP)
		self.play(
			Write(beauty)
		)

		self.wait(28)

		fast = TextMobject("Fast").scale(1.6)
		fourier = TextMobject("Fourier").scale(1.6)
		transform = TextMobject("Transform").scale(1.6)

		title = VGroup(fast, fourier, transform).arrange_submobjects(DOWN)
		title.move_to(UP * 2)

		first_class = VGroup(utility_class, utility)
		second_class = VGroup(beauty_class, beauty)
		all_classes = VGroup(first_class, second_class)

		self.play(
			ReplacementTransform(all_classes, title),
			run_time=2
		)

		self.wait()

class Earth(SVGMobject):
	# Designed by Flat Icons from www.flaticon.com, CC BY 4.0 <https://creativecommons.org/licenses/by/4.0>, via Wikimedia Commons
    CONFIG = {
        "file_name": "earth",
        # "fill_color": "#F96854",
        # "fill_color" : WHITE,
        "fill_opacity": 1,
        "stroke_width": 0,
        "width": 4,
        "propagate_style_to_family": True
    }

    def __init__(self, **kwargs):
        SVGMobject.__init__(self, **kwargs)
        self.set_width(self.width)
        self.center()
        self[0].set_fill(color=EARTH_BLUE)
        self[1:5].set_fill(color=EARTH_GREEN)
        self[5].set_fill(color=EARTH_SHADOW, opacity=0.36)

class FFTIntroApps(GraphScene):
	CONFIG = {
        "x_min": 0,
        "x_max": 2 * TAU,
        "x_axis_width": 7,
        "y_axis_height": 3,
        "graph_origin": DOWN * 2 + LEFT * 3,
        "y_min": 2,
        "y_max": -2,
    }
	def construct(self):

		fast = TextMobject("Fast").scale(1.6)
		fourier = TextMobject("Fourier").scale(1.6)
		transform = TextMobject("Transform").scale(1.6)

		title = VGroup(fast, fourier, transform).arrange_submobjects(DOWN)
		title.move_to(UP * 2)
		self.add(title)

		self.wait(9)

		self.show_apps(title)
	
	def show_apps(self, title):
		wc = WirelessComm()
		wc.scale(0.7).move_to(UP * 3 + LEFT * 4.5)
		wc[-1].move_to((wc[8].get_center() + wc[9].get_center()) / 2)
		
		direction_r = RIGHT * 0.3 + DOWN
		direction_l = LEFT * 0.3 + DOWN 
		line_r = Line(wc[-1].get_center(), wc[-1].get_center() + direction_r * 2)
		line_l = Line(wc[-1].get_center(), wc[-1].get_center() + direction_l * 2)
		line_r.set_color(BLUE)
		line_l.set_color(BLUE)

		intermediate_lines = VGroup()

		for i in range(1, 6):
			if i % 2 == 1:
				line = Line(
					line_r.point_from_proportion(i / 6),
					line_l.point_from_proportion((i + 1) / 6)
				)
			else:
				line = Line(
					line_l.point_from_proportion(i / 6),
					line_r.point_from_proportion((i + 1) / 6)
				)
			intermediate_lines.add(line)

		intermediate_lines.set_color(BLUE)


		self.play(
			ShowCreation(wc[:10]),
			ShowCreation(wc[-1]),
			ShowCreation(line_r),
			ShowCreation(line_l),
			ShowCreation(intermediate_lines)
		)

		self.wait()

		gps = GPS()
		gps.scale(0.7).move_to(UP * 2 + RIGHT * 4.5)
		self.play(
			ShowCreation(gps)
		)

		self.wait()

		self.setup_axes()

		#This removes tick marks
		self.x_axis.remove(self.x_axis[1])
		self.y_axis.remove(self.y_axis[1])
		self.play(
			Write(self.axes)
		)

		f = lambda x: np.sin(x) + np.cos(2 * x)
		graph = self.get_graph(f)
		self.play(
			ShowCreation(graph)
		)

		location = self.coords_to_point(0, 1)

		dot = Dot(location).set_color(BLUE)

		g = lambda x: np.sin(x)
		h = lambda x: np.cos(2 * x)

		location_sin = self.coords_to_point(0, 0)
		location_cos = self.coords_to_point(0, 1)

		sin_graph = self.get_graph(g).set_color(YELLOW)
		dot_sin = Dot(location_sin).set_color(YELLOW)
		
		cos_graph = self.get_graph(h).set_color(GREEN_SCREEN)
		dot_cos = Dot(location_cos).set_color(GREEN_SCREEN)

		# self.play(
		# 	GrowFromCenter(dot),
		# 	GrowFromCenter(dot_sin),
		# 	GrowFromCenter(dot_cos)
		# )

		# self.play(
		# 	FadeOut(graph)
		# )

		self.play(
			# MoveAlongPath(dot, graph),
			# MoveAlongPath(dot_sin, sin_graph),
			# MoveAlongPath(dot_cos, cos_graph),
			ShowCreation(sin_graph),
			ShowCreation(cos_graph),
			run_time=3,
		)

		# self.play(
		# 	LaggedStart(
		# 		ShowCreation(sin_graph),
		# 		MoveAlongPath(dot_sin, sin_graph),
		# 	),
		# 	run_time=5
		# )

		self.wait()

		# self.play(
		# 	FadeOut(wc[:10]),
		# 	FadeOut(wc[-1]),
		# 	FadeOut(line_r),
		# 	FadeOut(line_l),
		# 	FadeOut(gps),
		# 	FadeOut(intermediate_lines),
		# 	FadeOut(self.axes),
		# 	FadeOut(graph),
		# 	FadeOut(sin_graph),
		# 	FadeOut(cos_graph),
		# 	FadeOut(title)
		# )

		# self.wait()

class FFTGraph(GraphAnimationUtils):
	def construct(self):
		graph, edge_dict = self.create_graph()
		nodes, edges = self.make_graph_mobject(graph, edge_dict, show_data=False)
		edges_orig = VGroup(*edges).copy()
		entire_graph = VGroup(nodes, edges_orig)
		entire_graph.move_to(RIGHT * 3)
		VGroup(*edges).move_to(RIGHT * 3 + RIGHT * SMALL_BUFF / 2)

		self.play(
			LaggedStartMap(
				ShowCreation, entire_graph,
				run_time=3
			)
		)

		edges = sorted(edges, key=lambda x: x.get_start()[0])
		edges = VGroup(*edges)
		for i in range(16):
			edges[i].set_color(color=[BLUE, SKY_BLUE])
		for i in range(16, 32):
			edges[i].set_color(color=[GREEN_SCREEN, BLUE])
		for i in range(32, 48):
			edges[i].set_color(color=[MONOKAI_GREEN, GREEN_SCREEN])
		
		self.play(
			LaggedStartMap(
			ShowCreationThenDestruction, edges,
			run_time=3
			)
		)

		self.play(
			LaggedStartMap(
			ShowCreation, edges,
			run_time=3
			)
		)

		self.wait(4)

	def create_graph(self):
		radius = 0.2
		graph = []
		edges = {}

		row_positions = [
		UP * 3.5, UP * 2.5, UP * 1.5, UP * 0.5,
		DOWN * 0.5, DOWN * 1.5, DOWN * 2.5, DOWN * 3.5
		]
		column_positions = [
		LEFT * 3, LEFT * 1,
		RIGHT * 1, RIGHT * 3
		]

		for vert in row_positions:
			for hor in column_positions:
				position = vert + hor
				node = GraphNode('', position=position, radius=radius)
				graph.append(node)

		for i in range(len(graph) - 1):
			if i % 4 == 0:
				if i % 8 == 0:
					edges[(i, i + 5)] = graph[i].connect(graph[i + 5])
				else:
					edges[(i, i - 3)] = graph[i].connect(graph[i - 3])

			if i % 4 != 3:
				edges[(i, i + 1)] = graph[i].connect(graph[i + 1])
			
			if i % 4 == 1:
				if i % 16 == 1 or i % 16 == 5:
					edges[(i, i + 9)] = graph[i].connect(graph[i + 9])
				else:
					edges[i, i - 7] = graph[i].connect(graph[i - 7])
			
			if i % 4 == 2:
				if i < 16:
					edges[(i, i + 17)] = graph[i].connect(graph[i + 17])
				else:
					edges[i, i - 15] = graph[i].connect(graph[i - 15])

		return graph, edges

class Description(Scene):
	def construct(self):
		text = TextMobject(
			"This is a graph" + "\\\\" + 
			"visualization of" + "\\\\" +
			"the FFT called" + "\\\\" + 
			"a FFT circuit"
		)
		text[0][-10:].set_color(YELLOW)
		text.move_to(LEFT * 4)

		self.play(
			Write(text),
			run_time=3
		)

		self.wait(5)

class FFTComplexity(GraphScene):
	CONFIG = {
		"x_min": 0,
		"x_max": 2 * TAU,
		"x_axis_width": 7,
		"y_axis_height": 3,
		"graph_origin": DOWN * 2 + LEFT * 3,
		"y_min": 2,
		"y_max": -2,
	}

	def construct(self):
		self.show_complexity()

	def show_complexity(self):
		complexity = TextMobject("The FFT can be hard to understand at first").scale(1.2)
		complexity[0][-7:].set_color(YELLOW)
		complexity.move_to(UP * 3.5)
		self.play(
			Write(complexity),
			run_time=2,
		)

		self.wait()

		dft_matrix = self.show_dft()
		self.play(
			FadeIn(dft_matrix),
		)

		self.wait(2)

		self.play(
			dft_matrix.scale, 0.7,
			dft_matrix.shift, UP * 1.6
		)

		self.wait(12)

		self.play(
			FadeOut(dft_matrix)
		)


		preview = ScreenRectangle()
		preview.set_height(4.5)
		preview.set_height(5.5)

		discovery = TextMobject("Discovery based approach").next_to(preview, DOWN)

		self.play(
			ShowCreation(preview),
			Write(discovery)
		)

		self.wait(21)

	def show_dft(self):
		dft_matrix = TexMobject(r"""
		\begin{bmatrix}
		1 & 1 & 1 & \cdots & 1 \\
		1 & e^{\frac{2 \pi i}{n}} & e^{\frac{2 \pi i (2)}{n}} & \cdots & e^{\frac{2 \pi i (n - 1)}{n}} \\
		1 & e^{\frac{2 \pi i (2)}{n}} & e^{\frac{2 \pi i (4)}{n}} & \cdots & e^{\frac{2 \pi i (2)(n - 1)}{n}} \\
		\vdots & \vdots & \vdots  & \ddots & \vdots  \\
		1 & e^{\frac{2 \pi i (n - 1)}{n}} & e^{\frac{2 \pi i (2)(n - 1)}{n}} & \cdots & e^{\frac{2 \pi i (n - 1)(n - 1)}{n}}
		\end{bmatrix}
		""")

		# dft_matrix = TexMobject(r"""
		# \begin{bmatrix}
		# 1 & 1 & 1 & \cdots & 1 \\
		# 1 & e^{\frac{2 \pi i}{n}} & e^{\frac{2 \pi i (2)}{n}} & \cdots & e^{\frac{2 \pi i (n - 1)}{n}} \\
		# 1 & e^{\frac{2 \pi i (2)}{n}} & \omega^4 & \cdots & \omega^{2(n - 1)} \\
		# \vdots & \vdots & \vdots  & \ddots & \vdots  \\
		# 1 & \omega^{n - 1} & \omega^{2(n - 1)} & \cdots & e^{\frac{2 \pi i (n - 1)(n - 1)}{n}}}
		# \end{bmatrix}
		# """).scale(0.8)

		dft_matrix_label = TextMobject("Discrete Fourier Transform")
		dft_matrix_label.next_to(dft_matrix, DOWN)
		
		return VGroup(dft_matrix, dft_matrix_label)


class TimeToFreq(GraphScene):
	CONFIG = {
		"x_min": 0,
		"x_max": 35,
		"x_axis_width": 5,
		"y_axis_height": 2.5,
		"graph_origin": DOWN * 1.8 + LEFT * 6,
		"y_min": 1.5,
		"y_max": -1.5,
		"x_axis_label": None,
		"y_axis_label": None,
	}
	def construct(self):
		self.setup_axes()
		self.show_time_freq_domain()

	def show_time_freq_domain(self):
		rate, data = wf.read("two_tone.wav")
		data_len = 1485
		points = zip(np.arange(0, data_len) * 1000 / rate, data[0:data_len])
		coords = [self.coords_to_point(x, y) for x, y in points]
		graph = VGroup()
		graph.set_points_smoothly(*[coords])
		graph.set_color(YELLOW)

		self.x_axis.remove(self.x_axis[1])
		self.y_axis.remove(self.y_axis[1])
		
		self.play(
			Write(self.axes)
		)

		time_label = TextMobject("Time Domain").scale(0.8)
		time_label.next_to(self.axes, DOWN)

		self.play(
			ShowCreation(graph),
			Write(time_label)
		)

		arrow = TexMobject(r"\Leftrightarrow")
		arrow.move_to(DOWN * 1.8)

		self.play(
			Write(arrow)
		)

		self.y_min = 0
		self.y_max = 3
		self.x_min = 0
		self.x_max = 200
		self.graph_origin = RIGHT * 1 + DOWN * 3
		self.setup_axes()
		self.x_axis.remove(self.x_axis[1])
		self.y_axis.remove(self.y_axis[1])
		self.play(
			Write(self.axes)
		)


		freq_start = 1200
		freq_len = 1800

		freq = np.abs(np.fft.rfft(data)) / 30000
		freq_x_axis = np.fft.rfftfreq(len(data), d=1./rate) - 400
		

		s_x, s_y = self.smooth_data(freq_x_axis[freq_start:freq_len], freq[freq_start:freq_len])
		points_f = zip(s_x, s_y)
		coords_f = [self.coords_to_point(x, y) for x, y in points_f]
		graph_f = VGroup()
		graph_f.set_points_smoothly(*[coords_f])
		graph_f.set_color(PINK)

		freq_label = TextMobject("Freq Domain").scale(0.8)
		freq_label.next_to(self.axes, DOWN)

		self.play(
			TransformFromCopy(graph, graph_f),
			Write(freq_label),
			run_time=3
		)

		print(time_label.get_center(), freq_label.get_center())
	

		self.wait(6)
	
	def smooth_data(self, x, y):
		smooth_x = []
		smooth_y = []
		for i in range(0, len(x), 5):
			if y[i] < 0.05:
				smooth_x.append(x[i])
				smooth_y.append(y[i] * 500)

		return smooth_x, smooth_y





class GPS(SVGMobject):
	CONFIG = {
		"file_name": "gps",
		# "fill_color": "#F96854",
		# "fill_color" : WHITE,
		"fill_opacity": 1,
		"stroke_width": 0,
		"width": 4,
		"propagate_style_to_family": True
	}

	def __init__(self, **kwargs):
		SVGMobject.__init__(self, **kwargs)
		self.set_width(self.width)
		self.center()
		self[0].set_color(GPS_BLUE)
		self[1].set_color(GPS_RED)

class WirelessComm(SVGMobject):
	CONFIG = {
		"file_name": "wc",
		# "fill_color": "#F96854",
		# "fill_color" : WHITE,
		"fill_opacity": 1,
		"stroke_width": 0,
		"width": 4,
		"propagate_style_to_family": True
	}

	def __init__(self, **kwargs):
		SVGMobject.__init__(self, **kwargs)
		self.set_width(self.width)
		self[:10].set_color(YELLOW)
		circle = Circle(radius=0.2).set_fill(color=BLUE, opacity=1)
		circle.set_color(BLUE)
		circle.move_to(ORIGIN)
		self.add(circle)
		self.center()


class MultiplyPolynomials(Scene):
	def construct(self):
		A, B, C, C_ans = self.show_multiplication()
		self.show_representation(A, B, C, C_ans)

	def show_multiplication(self):
		A_x = TexMobject("A(x) = ", "x^2 + 3x + 2")
		B_x = TexMobject("B(x) = ", "2x^2 + 1")
		inputs = VGroup(A_x, B_x).arrange_submobjects(RIGHT, buff=1)
		inputs.move_to(UP * 3)

		C_x_def = TexMobject(r"C(x) = A(x) \cdot B(x)")
		C_x_def.next_to(inputs, DOWN)
		self.play(
			Write(inputs[0])
		)
		self.play(
			Write(inputs[1])
		)

		self.wait(2)

		self.play(
			Write(C_x_def)
		)

		self.wait(4)

		indentation = LEFT * 4
		scale = 0.9

		C_x_mult = TexMobject(
			"C(x) = ", 
			"(", "x^2 + 3x + 2", ")(", 
			"2x^2 + 1", ")").scale(scale)
		C_x_mult.move_to(UP)
		C_x_mult.to_edge(indentation)
		self.play(
			Write(C_x_mult[0])
		)

		self.play(
			TransformFromCopy(A_x[1], C_x_mult[2]),
			TransformFromCopy(B_x[1], C_x_mult[4]),
			FadeIn(C_x_mult[1]),
			FadeIn(C_x_mult[3]),
			FadeIn(C_x_mult[5])
		)

		C_x_expand = TexMobject(
			"C(x) = ", 
			"x^2", "(", "2x^2 + 1", ")",
			" + ",
			"3x", "(", "2x^2 + 1", ")",
			" + ",
			"2", "(", "2x^2 + 1", ")"
		).scale(scale)

		C_x_expand.next_to(C_x_mult, DOWN)
		C_x_expand.to_edge(indentation)

		self.play(
			Write(C_x_expand[0])
		)

		self.play(
			TransformFromCopy(C_x_mult[2][:2], C_x_expand[1]),
			FadeIn(C_x_expand[2]),
			TransformFromCopy(C_x_mult[4], C_x_expand[3]),
			FadeIn(C_x_expand[4]),
			FadeIn(C_x_expand[5]),
			TransformFromCopy(C_x_mult[2][3:5], C_x_expand[6]),
			FadeIn(C_x_expand[7]),
			TransformFromCopy(C_x_mult[4], C_x_expand[8]),
			FadeIn(C_x_expand[9]),
			FadeIn(C_x_expand[10]),
			TransformFromCopy(C_x_mult[2][6], C_x_expand[11]),
			FadeIn(C_x_expand[12]),
			TransformFromCopy(C_x_mult[4], C_x_expand[13]),
			FadeIn(C_x_expand[14]),
			run_time=2
		)

		self.wait()

		C_x_post_dist = TexMobject("C(x) = 2x^4 + x^2 + 6x^3 + 3x + 4x^2 + 2").scale(scale)
		C_x_post_dist.next_to(C_x_expand, DOWN)
		C_x_post_dist.to_edge(indentation)
		C_x_expand.to_edge(indentation)
		self.play(
			Write(C_x_post_dist)
		)

		self.wait()
		
		C_x = TexMobject("C(x) = 2x^4 + 6x^3 + 5x^2 + 3x + 2").scale(scale)
		C_x.next_to(C_x_post_dist, DOWN)
		C_x.to_edge(indentation)
		self.play(
			Write(C_x)
		)

		self.wait()

		self.play(
			FadeOut(C_x_expand),
			FadeOut(C_x_mult),
			FadeOut(C_x_post_dist),
			C_x.next_to, C_x_def, DOWN * 1.1,
			C_x.scale, 1 / scale,
			run_time=2
		)

		self.wait()

		return A_x, B_x, C_x_def, C_x

	def show_representation(self, A_x, B_x, C_mult, C_x):
		code_scale = 0.8
		indentation = LEFT * 4

		text_scale = 0.9

		function_def = TextMobject(r"def multPolynomial($A, B$)")
		function_def.scale(code_scale)
		function_def.to_edge(indentation)
		function_def[0][:3].set_color(MONOKAI_BLUE)
		function_def[0][3:17].set_color(MONOKAI_GREEN)
		function_def[0][18].set_color(MONOKAI_ORANGE)
		function_def[0][20].set_color(MONOKAI_ORANGE)
		
		description = TextMobject(r"\# computes polynomial $C = A \cdot B$")
		description.scale(code_scale)
		description.set_color(MONOKAI_GRAY)
		description.next_to(function_def, DOWN * 0.5)
		description.to_edge(indentation + LEFT)

		code = VGroup(function_def, description)

		code.next_to(C_x, DOWN)
		self.play(
			Write(code[0])
		)

		self.play(
			Write(code[1])
		)

		self.wait(2)

		A = TexMobject(r"A(x) = 2 + 3x + x^2 \rightarrow A = [2, 3, 1]").scale(text_scale)
		A[0][-6:-1:2].set_color(MONOKAI_PURPLE)
		B = TexMobject(r"B(x) = 1 + 2x^2 \rightarrow B = [1, 0, 2]").scale(text_scale)
		B[0][-6:-1:2].set_color(MONOKAI_PURPLE)
		C = TexMobject(r"C(x) = 2 + 3x + 5x^2 + 6x^3 + 2x^4 \rightarrow C = [2, 3, 5, 6, 2]").scale(text_scale)
		C[0][-10:-1:2].set_color(MONOKAI_PURPLE)

		A.next_to(code, DOWN)
		B.next_to(A, DOWN)
		C.next_to(B, DOWN)

		self.play(
			Write(A),
			run_time=2
		)

		self.play(
			Write(B),
			run_time=2
		)

		self.play(
			Write(C),
			run_time=2
		)

		self.wait(2)

		meaning = TextMobject(r"$C[k] = $ coefficient of $k^{\text{th}}$ degree term of polynomial $C(x)$")
		meaning.scale(0.8)
		meaning.next_to(C, DOWN)

		self.play(
			Write(meaning),
			run_time=2
		)

		self.wait(5)

		coeff_rep = TextMobject("Coefficient Representation").set_color(YELLOW)
		coeff_rep.next_to(meaning, DOWN)


		self.play(
			meaning.set_color, YELLOW,
			Write(coeff_rep)
		)

		self.wait(4)

		self.play(
			FadeOut(A_x),
			FadeOut(B_x),
			FadeOut(C_mult),
			FadeOut(C_x),
			code.move_to, UP * 3,
			FadeOut(A),
			FadeOut(B),
			FadeOut(C),
			FadeOut(meaning),
			FadeOut(coeff_rep),
			run_time=2
		)

		self.wait()

		A_gen = TexMobject(r"A(x) = a_0 + a_1x + a_2x^2 + \cdots + a_dx^d")
		B_gen = TexMobject(r"B(x) = b_0 + b_1x + b_2x^2 + \cdots + b_dx^d")
		C_gen = TexMobject(r"C(x) = c_0 + c_1x + c_2x^2 + \cdots + c_{2d}x^{2d}")
		A_gen.next_to(code, DOWN)
		B_gen.next_to(A_gen, DOWN)
		C_gen.next_to(B_gen, DOWN)

		self.play(
			FadeIn(A_gen),
			FadeIn(B_gen)
		)

		self.wait()

		self.play(
			FadeIn(C_gen)
		)

		self.wait(2)

		big_O = TextMobject(r"Runtime using distributive property: $O(d^2)$")
		
		better = TextMobject("Can we do better?")
		big_O.next_to(C_gen, DOWN)
		better.next_to(big_O, DOWN)

		self.play(
			Write(big_O[0][:-5])
		)

		self.wait(4)


		self.play(
			Write(big_O[0][-5:])
		)

		self.wait(7)

		self.play(
			Write(better)
		)

		self.wait(3)

class Representation(GraphScene):
	CONFIG = {
        "x_min": -5,
        "x_max": 5,
        "x_axis_width": 7,
        "y_axis_height": 6,
        "graph_origin": RIGHT * 3 + DOWN * 0.8,
        "y_min": -5,
        "y_max": 5,
        "y_axis_label": "$P(x)$",
    }
	def construct(self):
		title = TextMobject("Polynomial Representation")
		title.scale(1.2)
		title.shift(UP * 3.5)

		h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
		h_line.next_to(title, DOWN)
		self.play(
			Write(title),
			ShowCreation(h_line)
		)

		self.wait(6)

		line_theorem = self.line_example(h_line)
		extension = self.show_polynomial_case(line_theorem, h_line)
		self.show_proof(extension)

	def line_example(self, h_line):
		line_equation = TexMobject("P(x) = p_0 + p_1x")
		line_equation.scale(0.9)
		line_equation.move_to(LEFT * 3.7)

		self.play(
			Write(line_equation)
		)

		self.setup_axes()

		#This removes tick marks
		# self.x_axis.remove(self.x_axis[1])
		# self.y_axis.remove(self.y_axis[1])
		self.play(
			Write(self.axes)
		)

		return self.track_lines(line_equation)

	def track_lines(self, line_equation):
		scale = 0.9
		p0, p1 = Integer(0).scale(0.8), Integer(1).scale(0.8)
		f = self.get_function(p0.get_value(), p1.get_value())
		old_graph = self.get_graph(f, color=BLUE)
		
		p_0_text = TexMobject("p_0: ").scale(scale)
		p_1_text = TexMobject("p_1: ").scale(scale)
		text = VGroup(p_0_text, p_1_text).arrange_submobjects(RIGHT, buff=1)
		text.next_to(line_equation, DOWN)
		text.shift(LEFT * 0.2)
		p0.next_to(p_0_text, RIGHT, buff=SMALL_BUFF * 2)
		p1.next_to(p_1_text, RIGHT, buff=SMALL_BUFF * 2)
		

		self.play(
			FadeIn(text),
			FadeIn(p0),
			FadeIn(p1)
		)

		self.play(
			ShowCreation(old_graph),
		)

		self.wait(3)

		p0_range = list(range(-2, 3, 1))
		p1_range = list(range(-2, 3, 1))

		for p0_val in p0_range:
			for p1_val in p1_range:
				p0, p1, old_graph = self.transform_graph(p0, p1, old_graph, p0_val, p1_val)

		self.wait()

		self.play(
			FadeOut(old_graph),
			FadeOut(p0),
			FadeOut(p1),
			FadeOut(p_0_text),
			FadeOut(p_1_text),
			line_equation.shift, UP * 1.5
		)

		points1 = TexMobject("(-2, 0), (2, 2)").scale(0.8)
		points1.set_color(BLUE)

		points1.next_to(line_equation, DOWN)

		first_point1 = self.coords_to_point(-2, 0)
		second_point1 = self.coords_to_point(2, 2)

		dot1 = Dot().set_color(BLUE).move_to(first_point1)
		dot2 = Dot().set_color(BLUE).move_to(second_point1)

		self.play(
			FadeIn(points1)
		)

		self.play(
			FadeIn(dot1),
			FadeIn(dot2)
		)

		self.wait()

		f1 = self.get_function(1, 0.5)
		graph1 = self.get_graph(f1, color=BLUE)

		line_equation1 = TexMobject("P(x) = 1 + 0.5x").set_color(BLUE)
		line_equation1.scale(0.9)
		line_equation1.next_to(points1, DOWN)

		self.play(
			ShowCreation(graph1),
			Write(line_equation1)
		)

		points2 = TexMobject("(3, 0), (-1, 4)").scale(0.8)
		points2.set_color(YELLOW)

		points2.next_to(line_equation1, DOWN)

		first_point2 = self.coords_to_point(3, 0)
		second_point2 = self.coords_to_point(-1, 4)

		dot3 = Dot().set_color(YELLOW).move_to(first_point2)
		dot4 = Dot().set_color(YELLOW).move_to(second_point2)

		self.play(
			FadeIn(points2)
		)

		self.play(
			FadeIn(dot3),
			FadeIn(dot4)
		)

		self.wait()

		f2 = self.get_function(3, -1)
		graph2 = self.get_graph(f2, x_min=-2, x_max=5, color=YELLOW)

		line_equation2 = TexMobject("P(x) = 3 - x").set_color(YELLOW)
		line_equation2.scale(0.9)
		line_equation2.next_to(points2, DOWN)

		self.play(
			ShowCreation(graph2),
			Write(line_equation2)
		)

		line_theorem = TextMobject("2 points define a unique line")
		line_theorem.scale(0.9)

		line_theorem.next_to(line_equation2, DOWN * 2)

		self.play(
			Write(line_theorem)
		)

		self.wait(2)

		self.play(
			FadeOut(line_equation),
			FadeOut(line_equation1),
			FadeOut(line_equation2),
			FadeOut(points1),
			FadeOut(points2),
			FadeOut(dot1),
			FadeOut(dot2),
			FadeOut(dot3),
			FadeOut(dot4),
			FadeOut(graph1),
			FadeOut(graph2),
			line_theorem.shift, UP * 4.5,
			run_time=2
		)

		self.wait(2)

		return line_theorem

	def show_polynomial_case(self, line_theorem, h_line):
		extension = TextMobject(
			r"$(d + 1)$ points uniquely" + "\\\\",  
			r"define a degree $d$ polynomial" )
		extension.scale(0.9)
		extension.move_to(line_theorem.get_center())
		extension.shift(DOWN * 0.2)
		self.play(
			ReplacementTransform(line_theorem, extension)
		)
		self.wait(5)

		points1 = [(-3, 1), (-1, -1), (1, 3)]
		ex1 = TexMobject(r"\{(-3, 1), (-1, -1), (1, 3)\}").scale(0.9).set_color(BLUE)

		dot_objs1 = self.get_dot_objects(points1)

		f = lambda x: 0.75 * x ** 2 + 2 * x + 0.25 

		graph1 = self.get_graph(f, x_min=-4.2, x_max=1.5)

		ex1.next_to(extension, DOWN * 2)
		self.play(
			*[GrowFromCenter(obj) for obj in dot_objs1],
			FadeIn(ex1)
		)

		self.wait()

		poly1 = TexMobject(r"P(x) = \frac{3}{4} x^2 + 2x + \frac{1}{4}").scale(0.9).set_color(BLUE)
		poly1.next_to(ex1, DOWN)

		self.play(
			ShowCreation(graph1),
			Write(poly1)
		)

		self.wait(2)

		points2 = [(-1, 0), (0, 1), (1, 0), (2, 1)]
		ex2 = TexMobject(r"\{(-1, 0), (0, 1), (1, 0), (2, 1)\}").scale(0.9).set_color(YELLOW)

		dot_objs2 = self.get_dot_objects(points2, color=YELLOW)

		f2 = lambda x: 2/3 * x ** 3 - x ** 2 - 2/3 * x + 1

		graph2 = self.get_graph(f2, x_min=-1.8, x_max=2.7).set_color(YELLOW)

		ex2.next_to(poly1, DOWN * 2)
		self.play(
			*[GrowFromCenter(obj) for obj in dot_objs2],
			FadeIn(ex2)
		)

		self.wait()

		poly2 = TexMobject(r"P(x) = \frac{2}{3} x^3 - x^2 - \frac{2}{3}x + 1").scale(0.9).set_color(YELLOW)
		poly2.next_to(ex2, DOWN)

		self.play(
			ShowCreation(graph2),
			Write(poly2)
		)

		self.wait(3)

		extension_tranf = TextMobject(r"$(d + 1)$ points uniquely define a degree $d$ polynomial" )
		extension_tranf.scale(0.9)
		extension_tranf.next_to(h_line, DOWN)
		self.play(
			Transform(extension, extension_tranf),
			FadeOut(ex1),
			FadeOut(ex2),
			FadeOut(poly1),
			FadeOut(poly2),
			FadeOut(graph1),
			FadeOut(graph2),
			FadeOut(self.axes),
			*[FadeOut(obj) for obj in dot_objs1],
			*[FadeOut(obj) for obj in dot_objs2],
			run_time=2
		)

		return extension

	def show_proof(self, extension):
		scale = 0.8
		points = TexMobject(r"\{(x_0, P(x_0)), (x_1, P(x_1)), \ldots , (x_d, P(x_d)) \}")
		points.scale(scale)
		points.next_to(extension, DOWN * 1.5)
		self.play(
			Write(points)
		)

		polynomial = TexMobject(r"P(x) = p_0 + p_1x + p_2x^2 + \cdots + p_dx^d")
		polynomial.scale(scale)
		polynomial.next_to(points, DOWN * 1.5)
		self.play(
			Write(polynomial)
		)

		self.wait(5)

		x_0poly = TexMobject(r"P(x_0) = p_0 + p_1x_0 + p_2x_0^2 + \cdots + p_dx_0^d").scale(scale)
		x_1poly = TexMobject(r"P(x_1) = p_0 + p_1x_1 + p_2x_1^2 + \cdots + p_dx_1^d").scale(scale)
		vdots = TexMobject(r"\vdots").scale(scale)
		x_dpoly = TexMobject(r"P(x_d) = p_0 + p_1x_d + p_2x_d^2 + \cdots + p_dx_d^d").scale(scale)

		x_0poly.next_to(polynomial, DOWN * 2)
		x_1poly.next_to(x_0poly, DOWN)
		vdots.next_to(x_1poly, DOWN)
		x_dpoly.next_to(vdots, DOWN)

		self.play(
			Write(x_0poly)
		)

		self.play(
			Write(x_1poly)
		)

		self.play(
			FadeIn(vdots),
			Write(x_dpoly)
		)

		self.wait(3)

		system_of_equations = VGroup(x_0poly, x_1poly, vdots, x_dpoly)

		matrix = TexMobject(r"""
		\begin{bmatrix}
		1 & x_0 & x_0^2 & \cdots & x_0^d \\
		1 & x_1 & x_1^2 & \cdots & x_1^d \\
		\vdots & \vdots & \vdots  & \ddots & \vdots  \\
		1 & x_d & x_d^2 & \cdots & x_d^d 
		\end{bmatrix}
		""")

		coeff_vector = TexMobject(r"""
		\begin{bmatrix}
		p_0 \\
		p_1 \\
		\vdots \\
		p_d
		\end{bmatrix}
		""")

		value_vector = TexMobject(r"""
		\begin{bmatrix}
		P(x_0) \\
		P(x_1) \\
		\vdots \\
		P(x_d)
		\end{bmatrix}
		""")

		equals = TexMobject("=")

		coeff_vector.next_to(matrix, RIGHT)
		equals.next_to(matrix, LEFT)
		value_vector.next_to(equals, LEFT)

		matrix_vector_prod = VGroup(value_vector, equals, matrix, coeff_vector)

		matrix_vector_prod.move_to(system_of_equations.get_center())


		self.play(
			ReplacementTransform(system_of_equations, matrix_vector_prod),
			run_time=2
		)

		self.wait(3)

		self.play(
			matrix_vector_prod.shift, LEFT * 3,
			matrix_vector_prod.scale, 0.8
		)

		brace = Brace(matrix, direction=DOWN)
		label = TexMobject("M").next_to(brace, DOWN)

		self.play(
			GrowFromCenter(brace),
			Write(label)
		)

		self.wait()

		invertible = TextMobject(r"$M$ is invertible for unique $x_0, x_1, \ldots , x_d$").scale(0.7)
		invertible.next_to(matrix_vector_prod, RIGHT).shift(UP * 1)

		coefficient = TextMobject(r"$\Rightarrow$ Unique $p_0, p_1, \ldots , p_d$ exists").scale(0.7).next_to(invertible, DOWN)
		conclusion = TextMobject(r"$\Rightarrow$ Unique polynomial $P(x)$ exists").scale(0.7).next_to(coefficient, DOWN)

		explanation = VGroup(invertible, coefficient, conclusion).next_to(matrix_vector_prod, RIGHT)

		self.play(
			Write(invertible)
		)

		self.wait(18)

		self.play(
			Write(coefficient)
		)

		self.wait(2)

		self.play(
			Write(conclusion)
		)

		self.wait(3)
		punchline = TextMobject("Punchline: Two Unique Representations for Polynomials")
		punchline.move_to(extension.get_center())
		
		self.play(
			FadeOut(coefficient),
			FadeOut(invertible),
			FadeOut(conclusion),
			FadeOut(matrix_vector_prod),
			FadeOut(brace),
			FadeOut(label),
			FadeOut(points),
			FadeOut(polynomial),
			ReplacementTransform(extension, punchline)
		)

		polynomial.next_to(punchline, DOWN * 1.5).scale(0.9 / scale)
		self.play(
			Write(polynomial)
		)

		coeff_rep = TexMobject(r"1. \quad [p_0, p_1, \ldots, p_d]").scale(0.9)

		coeff_rep.move_to(UP * 0.8)

		coeff_brace = Brace(coeff_rep[0][2:], direction=DOWN)
		coeff_brace_label = TextMobject("Coefficient Representation").scale(scale)
		coeff_brace_label.next_to(coeff_brace, DOWN)

		self.play(
			Write(coeff_rep),
		)


		self.play(
			GrowFromCenter(coeff_brace),
		)

		self.play(
			Write(coeff_brace_label)
		)

		self.wait()

		points = TexMobject(r"2. \quad \{(x_0, P(x_0)), (x_1, P(x_1)), \ldots , (x_d, P(x_d)) \}")
		points.scale(scale).move_to(DOWN * 1.3)

		self.play(
			Write(points)
		)

		self.wait()

		value_brace = Brace(points[0][2:], direction=DOWN)
		value_brace_label = TextMobject("Value Representation").scale(scale)
		value_brace_label.next_to(value_brace, DOWN)

		self.play(
			GrowFromCenter(value_brace),
			Write(value_brace_label)
		)

		self.wait(9)

	def get_dot_objects(self, points, color=BLUE):
		objs = []
		for x, y in points:
			dot = Dot().set_color(color)
			coord = self.coords_to_point(x, y)
			dot.move_to(coord)
			objs.append(dot)
		return objs

	def transform_graph(self, old_p0, old_p1, old_graph, p0, p1):
		f = self.get_function(p0, p1)
		x_min, x_max = self.get_boundaries(p0, p1)
		new_graph = self.get_graph(f, x_min=x_min, x_max=x_max, color=BLUE)
		new_text = TextMobject
		new_p0 = Integer(p0).scale(0.8).move_to(old_p0.get_center())
		new_p1 = Integer(p1).scale(0.8).move_to(old_p1.get_center())
		self.play(
			ReplacementTransform(old_graph, new_graph),
			ReplacementTransform(old_p0, new_p0),
			ReplacementTransform(old_p1, new_p1)
			
		)

		return new_p0, new_p1, new_graph

	def get_function(self, p0, p1):
		return lambda x: p0 + p1 * x

	def get_boundaries(self, p0, p1):
		if p1 == 0:
			return [-5, 5]
		x_max = (5 - p0) / p1
		x_min = (-5 - p0) / p1

		x_min, x_max = sorted([x_min, x_max])
		
		if x_min < -5:
			x_min = -5
		if x_max > 5:
			x_max = 5
		
		return x_min, x_max

class ValueRepresentationMult(GraphScene):
	CONFIG = {
	    "x_min": -2,
	    "x_max": 2,
	    "x_axis_width": 3.5,
	    "y_axis_height": 4,
	    "graph_origin": LEFT * 4.5 + DOWN * 3.5,
	    "y_min": 0,
	    "y_max": 9,
	    "y_axis_label": "$A(x)$",
	}

	def construct(self):
		A_x_graph, B_x_graph, C_x_graph = self.show_example()
	
		self.show_mult_flow(A_x_graph, B_x_graph, C_x_graph)

		self.wait()

	def show_example(self):
		scale = 0.9
		A_x = TexMobject("A(x) = x^2 + 2x + 1").scale(scale)
		B_x = TexMobject("B(x) = x^2 - 2x + 1").scale(scale)
		C_x = TexMobject(r"C(x) = ", r"A(x) \cdot B(x)").scale(scale)

		A_x.move_to(LEFT * 4.5 + UP * 3)
		B_x.move_to(UP * 3)
		C_x.move_to(RIGHT * 4.5 + UP * 3)

		self.play(
			Write(A_x),
		)

		self.play(
			Write(B_x),
		)

		self.play(
			Write(C_x)
		)

		self.wait(2)

		brace = Brace(C_x, direction=DOWN)
		self.play(
			GrowFromCenter(brace)
		)

		degree = TextMobject(r"Degree 4 $\rightarrow$ 5 points")
		degree.scale(0.8)

		degree.next_to(brace, DOWN)

		self.play(
			Write(degree)
		)

		self.wait(3)

		first_transforms = []
		#A(x) graph begin

		self.setup_axes()
		self.y_axis_label_mob.scale(0.8)

		first_transforms.append(Write(self.axes))

		points = [(-2, 1), (-1, 0), (0, 1), (1, 4), (2, 9)]
		poly_func = lambda x: (x + 1) ** 2
		points_a, function, dots = self.show_value_representation(points, poly_func, A_x)
		
		A_x_graph = VGroup(self.axes, function, dots)
		#A(x) graph end


		# B(x) graph begin

		self.graph_origin = DOWN * 3.5
		self.y_axis_label = "$B(x)$"
		self.setup_axes()
		self.y_axis_label_mob.scale(0.8)
		first_transforms.append(Write(self.axes))

		points = [(-2, 9), (-1, 4), (0, 1), (1, 0), (2, 1)]
		poly_func = lambda x: (x - 1) ** 2
		points_b, function, dots = self.show_value_representation(points, poly_func, B_x, color=YELLOW)

		B_x_graph = VGroup(self.axes, function, dots)
		# B(x) graph end

		# Animations
		self.play(
			*first_transforms
		)

		self.play(
			ShowCreation(A_x_graph[1]),
			ShowCreation(B_x_graph[1])
		)

		self.play(
			*[GrowFromCenter(dot) for dot in A_x_graph[2]],
			*[GrowFromCenter(dot) for dot in B_x_graph[2]],
		)

		self.play(
			FadeIn(points_a[0][0]),
			FadeIn(points_a[0][7]),
			FadeIn(points_a[0][14]),
			FadeIn(points_a[0][20]),
			FadeIn(points_a[0][26]),
			FadeIn(points_a[0][-1]),
			TransformFromCopy(A_x_graph[2][0], points_a[0][1:7]),
			TransformFromCopy(A_x_graph[2][1], points_a[0][8:14]),
			TransformFromCopy(A_x_graph[2][2], points_a[0][15:20]),
			TransformFromCopy(A_x_graph[2][3], points_a[0][21:26]),
			TransformFromCopy(A_x_graph[2][4], points_a[0][27:-1]),
			FadeIn(points_b[0][0]),
			FadeIn(points_b[0][7]),
			FadeIn(points_b[0][14]),
			FadeIn(points_b[0][20]),
			FadeIn(points_b[0][26]),
			FadeIn(points_b[0][-1]),
			TransformFromCopy(B_x_graph[2][0], points_b[0][1:7]),
			TransformFromCopy(B_x_graph[2][1], points_b[0][8:14]),
			TransformFromCopy(B_x_graph[2][2], points_b[0][15:20]),
			TransformFromCopy(B_x_graph[2][3], points_b[0][21:26]),
			TransformFromCopy(B_x_graph[2][4], points_b[0][27:-1]),
			run_time=2
		)

		self.wait()

		p_a_copy = points_a.copy().next_to(C_x, DOWN)
		p_b_copy = points_b.copy().next_to(C_x, DOWN * 3)

		self.play(
			FadeOut(brace),
			FadeOut(degree),
			TransformFromCopy(points_a, p_a_copy),
			TransformFromCopy(points_b, p_b_copy),
			run_time=2
		)

		self.wait()

		mult = TexMobject(r"\times").scale(0.6).next_to(p_b_copy, LEFT)
		line = Underline(p_b_copy, buff=SMALL_BUFF * 2)
		self.play(
			Write(mult),
			ShowCreation(line)
		)

		self.wait()


		points = [(-2, 9), (-1, 0), (0, 1), (1, 0), (2, 9)]
		poly_func = lambda x: (x - 1) ** 2 * (x + 1) ** 2
		points_c = TexMobject("[(-2, 9), (-1, 0), (0, 1), (1, 0), (2, 9)]").scale(0.55).set_color(GREEN_SCREEN)

		points_c.next_to(line, DOWN)

		self.play(
			Write(points_c)
		)

		self.wait()

		self.play(
			FadeOut(p_a_copy),
			FadeOut(p_b_copy),
			FadeOut(mult),
			FadeOut(line),
			points_c.next_to, C_x, DOWN
		)

		# C(x) graph begin

		self.graph_origin = RIGHT * 4.5 + DOWN * 3.5
		self.y_axis_label = "$C(x)$"
		self.setup_axes()
		self.y_axis_label_mob.scale(0.8)

		dots = []
		for x, y in points:
			dot = Dot().set_color(GREEN_SCREEN)
			dots.append(dot)
			dot.move_to(self.coords_to_point(x, y))

		self.play(
			Write(self.axes)
		)

		self.wait()

		self.play(
			TransformFromCopy(points_c[0][1:7], dots[0]),
			TransformFromCopy(points_c[0][8:14], dots[1]),
			TransformFromCopy(points_c[0][15:20], dots[2]),
			TransformFromCopy(points_c[0][21:26], dots[3]),
			TransformFromCopy(points_c[0][27:-1], dots[4]),
			run_time=2
		)

		function = self.get_graph(poly_func, color=GREEN_SCREEN)

		self.play(
			ShowCreation(function)
		)

		self.wait()

		C_x_result = TexMobject("x^4 - 2x^2 + 1").scale(scale).next_to(C_x[0], RIGHT, buff=SMALL_BUFF * 2)
		C_x_result.shift(UP * SMALL_BUFF / 2)

		self.play(
			Transform(C_x[1], C_x_result)
		)

		self.wait(15)

		C_x_graph = VGroup(self.axes, function, VGroup(*dots))

		value_rep_obs = TextMobject("Multiplication in value representation is only $O(d)$!").scale(0.9)
		value_rep_obs.scale(0.9).next_to(points_b, DOWN)

		self.play(
			Write(value_rep_obs)
		)

		self.wait(7)
		# C(x) graph end

		self.play(
			FadeOut(A_x_graph),
			FadeOut(B_x_graph),
			FadeOut(C_x_graph),
			FadeOut(points_a),
			FadeOut(points_b),
			FadeOut(points_c),
			FadeOut(A_x),
			FadeOut(B_x),
			FadeOut(C_x),
			FadeOut(value_rep_obs)
		)

		return A_x_graph, B_x_graph, C_x_graph

	def show_mult_flow(self, A_x_graph, B_x_graph, C_x_graph):
		A_x_graph.scale(0.6)
		B_x_graph.scale(0.6)
		C_x_graph.scale(0.7)

		A_gen = TexMobject(r"A(x) = a_0 + a_1x + \cdots + a_dx^d").scale(0.8)
		B_gen = TexMobject(r"B(x) = b_0 + b_1x + \cdots + b_dx^d").scale(0.8)

		A_gen.move_to(LEFT * 4 + UP * 3.5)
		B_gen.next_to(A_gen, DOWN * 0.5)

		self.play(
			FadeIn(A_gen),
			FadeIn(B_gen)
		)

		self.wait(7)

		arrow1 = TexMobject(r"\Downarrow").next_to(B_gen, DOWN, buff=SMALL_BUFF)

		self.play(
			ShowCreation(arrow1)
		)

		coeff_value = Rectangle(height=2, width=4.5).set_color(RED).set_fill(opacity=0)
		coeff_value_text = TextMobject(r"Coeff $\Rightarrow$ Value")
		coeff_value_text.move_to(coeff_value.get_center())
		# points_text = TextMobject(r"$n \geq 2d + 1$ points").scale(0.7).next_to(coeff_value_text, DOWN)
		first_component = VGroup(coeff_value, coeff_value_text)
		first_component.move_to(LEFT * 4 + UP * 1)

		self.play(
			FadeIn(first_component),
		)

		arrow2 = TexMobject(r"\Downarrow").next_to(first_component, DOWN)

		self.play(
			ShowCreation(arrow2)
		)

		input_graph = VGroup(A_x_graph, B_x_graph).arrange_submobjects(RIGHT)

		input_graph.move_to(LEFT * 4 + DOWN * 2)

		self.play(
			FadeIn(input_graph)
		)

		self.wait()
		

		multiply_rect = Rectangle(height=3, width=2.5).set_color(YELLOW).set_fill(opacity=0)
		multiply_text = TextMobject("Multiply")

		multiply_component = VGroup(multiply_rect, multiply_text)

		multiply_component.next_to(input_graph, RIGHT, buff=0.5)
		
		arrow3 = TexMobject(r"\Rightarrow").next_to(multiply_component, LEFT)

		self.play(
			ShowCreation(arrow3)
		)

		self.play(
			FadeIn(multiply_component)
		)

		arrow4 = TexMobject(r"\Rightarrow").next_to(multiply_component, RIGHT)

		self.play(
			ShowCreation(arrow4)
		)

		C_x_graph.next_to(multiply_component, RIGHT, buff=1.75)

		self.play(
			FadeIn(C_x_graph)
		)

		self.wait(4)

		value_coeff = Rectangle(height=2, width=4.5).set_color(GREEN_SCREEN).set_fill(opacity=0)
		value_coeff_text = TextMobject(r"Value $\Rightarrow$ Coeff")
		value_coeff_text.move_to(value_coeff.get_center())

		third_component = VGroup(value_coeff, value_coeff_text)
		third_component.move_to(RIGHT * 4.5 + UP * 1)

		arrow5 = TexMobject(r"\Uparrow").next_to(third_component, DOWN, buff=SMALL_BUFF)

		self.play(
			ShowCreation(arrow5)
		)

		self.play(
			FadeIn(third_component)
		)

		arrow6 = TexMobject(r"\Uparrow").next_to(third_component, UP)

		self.play(
			ShowCreation(arrow6)
		)

		C_gen = TexMobject(r"C(x) = c_0 + c_1x + \cdots + c_{2d}x^{2d}").scale(0.75)
		C_gen.move_to(RIGHT * 4.5 + UP * 3.2)

		self.play(
			FadeIn(C_gen)
		)

		self.wait(9)

		self.play(
			ApplyWave(first_component),
			ApplyWave(third_component)
		)

		self.wait(11)

		fft = TextMobject("FFT").move_to(UP * 1 + RIGHT * 0.25)

		arrow_to_red = Arrow(RIGHT, LEFT).next_to(first_component, RIGHT, buff=SMALL_BUFF / 2).scale(0.7).shift(LEFT * SMALL_BUFF)
		# arrow_to_green = Arrow(LEFT, RIGHT).next_to(third_component, LEFT, buff=SMALL_BUFF / 2).scale(0.7).shift(RIGHT * SMALL_BUFF)

		self.play(
			GrowFromCenter(fft),
			ShowCreation(arrow_to_red),
		)

		self.wait(4)

		self.play(
			FadeOut(A_gen),
			FadeOut(B_gen),
			FadeOut(C_gen),
			FadeOut(arrow1),
			FadeOut(arrow2),
			FadeOut(arrow3),
			FadeOut(arrow4),
			FadeOut(arrow5),
			FadeOut(arrow6),
			FadeOut(input_graph),
			FadeOut(C_x_graph),
			FadeOut(coeff_value),
			FadeOut(third_component),
			FadeOut(multiply_component),
			FadeOut(fft),
			FadeOut(arrow_to_red),
			coeff_value_text.move_to, UP * 3.5,
			coeff_value_text.scale, 1.2,
			run_time=2
		)

		self.wait(3)

		evaluation = TextMobject("Evaluation")
		evaluation.scale(1.2).move_to(UP * 3.5)
		
		h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
		h_line.next_to(evaluation, DOWN)

		self.play(
			ReplacementTransform(coeff_value_text, evaluation),
			ShowCreation(h_line)
		)

		self.wait()

	def show_value_representation(self, points, poly_func, F_x, color=BLUE):
		points_string = TexMobject(r"[{0}, {1}, {2}, {3}, {4}]".format(*points)).scale(0.55)
		points_string.next_to(F_x, DOWN).set_color(color)
		dots = []
		for x, y in points:
			dot = Dot().set_color(color)
			dots.append(dot)
			dot.move_to(self.coords_to_point(x, y))

		function = self.get_graph(poly_func, color=color)

		return points_string, function, VGroup(*dots)

class FocusOnInverse(GraphScene):
	# Run with flag n = 54
	CONFIG = {
	    "x_min": -2,
	    "x_max": 2,
	    "x_axis_width": 3.5,
	    "y_axis_height": 4,
	    "graph_origin": LEFT * 4.5 + DOWN * 3.5,
	    "y_min": 0,
	    "y_max": 9,
	    "y_axis_label": "$A(x)$",
	}

	def construct(self):
		A_x_graph, B_x_graph, C_x_graph = self.show_example()
	
		self.show_mult_flow(A_x_graph, B_x_graph, C_x_graph)

		self.wait()

	def show_example(self):
		scale = 0.9
		A_x = TexMobject("A(x) = x^2 + 2x + 1").scale(scale)
		B_x = TexMobject("B(x) = x^2 - 2x + 1").scale(scale)
		C_x = TexMobject(r"C(x) = ", r"A(x) \cdot B(x)").scale(scale)

		A_x.move_to(LEFT * 4.5 + UP * 3)
		B_x.move_to(UP * 3)
		C_x.move_to(RIGHT * 4.5 + UP * 3)

		self.play(
			Write(A_x),
			Write(B_x),
			Write(C_x)
		)

		self.wait()

		brace = Brace(C_x, direction=DOWN)
		self.play(
			GrowFromCenter(brace)
		)

		degree = TextMobject(r"Degree 4 $\rightarrow$ 5 points")
		degree.scale(0.8)

		degree.next_to(brace, DOWN)
		self.wait()

		self.play(
			Write(degree)
		)

		self.wait()

		#A(x) graph begin

		self.setup_axes()
		self.y_axis_label_mob.scale(0.8)

		self.play(
			Write(self.axes)
		)

		self.wait()

		points = [(-2, 1), (-1, 0), (0, 1), (1, 4), (2, 9)]
		poly_func = lambda x: (x + 1) ** 2
		points_a, function, dots = self.show_value_representation(points, poly_func, A_x)

		A_x_graph = VGroup(self.axes, function, dots)
		#A(x) graph end


		# B(x) graph begin

		self.graph_origin = DOWN * 3.5
		self.y_axis_label = "$B(x)$"
		self.setup_axes()
		self.y_axis_label_mob.scale(0.8)
		self.play(
			Write(self.axes)
		)

		self.wait()

		points = [(-2, 9), (-1, 4), (0, 1), (1, 0), (2, 1)]
		poly_func = lambda x: (x - 1) ** 2
		points_b, function, dots = self.show_value_representation(points, poly_func, B_x, color=YELLOW)

		B_x_graph = VGroup(self.axes, function, dots)
		# B(x) graph end

		p_a_copy = points_a.copy().next_to(C_x, DOWN)
		p_b_copy = points_b.copy().next_to(C_x, DOWN * 3)

		self.play(
			FadeOut(brace),
			FadeOut(degree),
			TransformFromCopy(points_a, p_a_copy),
			TransformFromCopy(points_b, p_b_copy),
			run_time=2
		)

		self.wait()

		mult = TexMobject(r"\times").scale(0.6).next_to(p_b_copy, LEFT)
		line = Underline(p_b_copy, buff=SMALL_BUFF * 2)
		self.play(
			Write(mult),
			ShowCreation(line)
		)

		self.wait()


		points = [(-2, 9), (-1, 0), (0, 1), (1, 0), (2, 9)]
		poly_func = lambda x: (x - 1) ** 2 * (x + 1) ** 2
		points_c = TexMobject("[(-2, 9), (-1, 0), (0, 1), (1, 0), (2, 9)]").scale(0.55).set_color(GREEN_SCREEN)

		points_c.next_to(line, DOWN)

		self.play(
			Write(points_c)
		)

		self.wait()

		self.play(
			FadeOut(p_a_copy),
			FadeOut(p_b_copy),
			FadeOut(mult),
			FadeOut(line),
			points_c.next_to, C_x, DOWN
		)

		# C(x) graph begin

		self.graph_origin = RIGHT * 4.5 + DOWN * 3.5
		self.y_axis_label = "$C(x)$"
		self.setup_axes()
		self.y_axis_label_mob.scale(0.8)

		dots = []
		for x, y in points:
			dot = Dot().set_color(GREEN_SCREEN)
			dots.append(dot)
			dot.move_to(self.coords_to_point(x, y))

		self.play(
			Write(self.axes)
		)

		self.wait()

		self.play(
			TransformFromCopy(points_c[0][1:7], dots[0]),
			TransformFromCopy(points_c[0][8:14], dots[1]),
			TransformFromCopy(points_c[0][15:20], dots[2]),
			TransformFromCopy(points_c[0][21:26], dots[3]),
			TransformFromCopy(points_c[0][27:-1], dots[4]),
			run_time=2
		)

		function = self.get_graph(poly_func, color=GREEN_SCREEN)

		self.play(
			ShowCreation(function)
		)

		self.wait()

		C_x_result = TexMobject("x^4 - 2x^2 + 1").scale(scale).next_to(C_x[0], RIGHT, buff=SMALL_BUFF * 2)
		C_x_result.shift(UP * SMALL_BUFF / 2)

		self.play(
			Transform(C_x[1], C_x_result)
		)

		self.wait()

		C_x_graph = VGroup(self.axes, function, VGroup(*dots))

		# C(x) graph end

		self.play(
			FadeOut(A_x_graph),
			FadeOut(B_x_graph),
			FadeOut(C_x_graph),
			FadeOut(points_a),
			FadeOut(points_b),
			FadeOut(points_c),
			FadeOut(A_x),
			FadeOut(B_x),
			FadeOut(C_x),
		)

		return A_x_graph, B_x_graph, C_x_graph

	def show_mult_flow(self, A_x_graph, B_x_graph, C_x_graph):
		A_x_graph.scale(0.6)
		B_x_graph.scale(0.6)
		C_x_graph.scale(0.7)

		A_gen = TexMobject(r"A(x) = a_0 + a_1x + \cdots + a_dx^d").scale(0.8)
		B_gen = TexMobject(r"B(x) = b_0 + b_1x + \cdots + b_dx^d").scale(0.8)

		A_gen.move_to(LEFT * 4 + UP * 3.5)
		B_gen.next_to(A_gen, DOWN * 0.5)

		self.play(
			FadeIn(A_gen),
			FadeIn(B_gen)
		)

		arrow1 = TexMobject(r"\Downarrow").next_to(B_gen, DOWN, buff=SMALL_BUFF)

		self.play(
			ShowCreation(arrow1)
		)

		coeff_value = Rectangle(height=2, width=4.5).set_color(RED).set_fill(opacity=0)
		coeff_value_text = TextMobject(r"Coeff $\Rightarrow$ Value")
		coeff_value_text.move_to(coeff_value.get_center())

		first_component = VGroup(coeff_value, coeff_value_text)
		first_component.move_to(LEFT * 4 + UP * 1)

		self.play(
			FadeIn(first_component)
		)

		arrow2 = TexMobject(r"\Downarrow").next_to(first_component, DOWN)

		self.play(
			ShowCreation(arrow2)
		)

		input_graph = VGroup(A_x_graph, B_x_graph).arrange_submobjects(RIGHT)

		input_graph.move_to(LEFT * 4 + DOWN * 2)

		self.play(
			FadeIn(input_graph)
		)

		self.wait()
		

		multiply_rect = Rectangle(height=3, width=2.5).set_color(YELLOW).set_fill(opacity=0)
		multiply_text = TextMobject("Multiply")

		multiply_component = VGroup(multiply_rect, multiply_text)

		multiply_component.next_to(input_graph, RIGHT, buff=0.5)
		
		arrow3 = TexMobject(r"\Rightarrow").next_to(multiply_component, LEFT)

		self.play(
			ShowCreation(arrow3)
		)

		self.play(
			FadeIn(multiply_component)
		)

		self.wait()

		arrow4 = TexMobject(r"\Rightarrow").next_to(multiply_component, RIGHT)

		self.play(
			ShowCreation(arrow4)
		)

		C_x_graph.next_to(multiply_component, RIGHT, buff=1.75)

		self.play(
			FadeIn(C_x_graph)
		)

		self.wait()

		value_coeff = Rectangle(height=2, width=4.5).set_color(GREEN_SCREEN).set_fill(opacity=0)
		value_coeff_text = TextMobject(r"Value $\Rightarrow$ Coeff")
		value_coeff_text.move_to(value_coeff.get_center())

		third_component = VGroup(value_coeff, value_coeff_text)
		third_component.move_to(RIGHT * 4.5 + UP * 1)

		arrow5 = TexMobject(r"\Uparrow").next_to(third_component, DOWN, buff=SMALL_BUFF)

		self.play(
			ShowCreation(arrow5)
		)

		self.play(
			FadeIn(third_component)
		)

		arrow6 = TexMobject(r"\Uparrow").next_to(third_component, UP)

		self.play(
			ShowCreation(arrow6)
		)

		C_gen = TexMobject(r"C(x) = c_0 + c_1x + \cdots + c_{2d}x^{2d}").scale(0.75)
		C_gen.move_to(RIGHT * 4.5 + UP * 3.2)

		self.play(
			FadeIn(C_gen)
		)

		self.wait(17)

		self.play(
			ApplyWave(third_component),
		)

		self.wait()

		# fft = TextMobject("FFT").move_to(UP * 1 + RIGHT * 0.25)

		# arrow_to_red = Arrow(RIGHT, LEFT).next_to(first_component, RIGHT, buff=SMALL_BUFF / 2).scale(0.7).shift(LEFT * SMALL_BUFF)
		# arrow_to_green = Arrow(LEFT, RIGHT).next_to(third_component, LEFT, buff=SMALL_BUFF).scale(0.8)

		# self.play(
		# 	GrowFromCenter(fft),
		# 	ShowCreation(arrow_to_red)
		# )

		self.play(
			FadeOut(A_gen),
			FadeOut(B_gen),
			FadeOut(C_gen),
			FadeOut(arrow1),
			FadeOut(arrow2),
			FadeOut(arrow3),
			FadeOut(arrow4),
			FadeOut(arrow5),
			FadeOut(arrow6),
			FadeOut(input_graph),
			FadeOut(C_x_graph),
			FadeOut(coeff_value),
			FadeOut(first_component),
			FadeOut(coeff_value_text),
			FadeOut(multiply_component),
			FadeOut(third_component[0]),
			value_coeff_text.move_to, UP * 3.5,
			value_coeff_text.scale, 1.2,
			run_time=2
		)

		interpolation = TextMobject("Interpolation")
		interpolation.scale(1.2).move_to(UP * 3.5)
		
		h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
		h_line.next_to(interpolation, DOWN)

		self.play(
			ReplacementTransform(value_coeff_text, interpolation),
			ShowCreation(h_line)
		)

		self.wait(2)

	def show_value_representation(self, points, poly_func, F_x, color=BLUE):
		points_string = TexMobject(r"[{0}, {1}, {2}, {3}, {4}]".format(*points)).scale(0.55)
		points_string.next_to(F_x, DOWN).set_color(color)
		dots = []
		for x, y in points:
			dot = Dot().set_color(color)
			dots.append(dot)
			dot.move_to(self.coords_to_point(x, y))

		function = self.get_graph(poly_func, color=color)

		self.play(
			ShowCreation(function)
		)

		self.wait()

		self.play(
			*[GrowFromCenter(dot) for dot in dots]
		)

		self.wait()

		self.play(
			FadeIn(points_string[0][0]),
			FadeIn(points_string[0][7]),
			FadeIn(points_string[0][14]),
			FadeIn(points_string[0][20]),
			FadeIn(points_string[0][26]),
			FadeIn(points_string[0][-1]),
			TransformFromCopy(dots[0], points_string[0][1:7]),
			TransformFromCopy(dots[1], points_string[0][8:14]),
			TransformFromCopy(dots[2], points_string[0][15:20]),
			TransformFromCopy(dots[3], points_string[0][21:26]),
			TransformFromCopy(dots[4], points_string[0][27:-1]),
			run_time=2
		)

		self.wait()

		return points_string, function, VGroup(*dots)

class Evaluation(GraphScene):
	CONFIG = {
        "x_min": -10,
        "x_max": 10,
        "x_axis_width": 5,
        "y_axis_height": 5,
        "graph_origin": RIGHT * 3.2 + DOWN * 1.5,
        "y_min": -10,
        "y_max": 10,
        "y_axis_label": "$P(x)$",
    }
	def construct(self):
		evaluation = TextMobject("Evaluation")
		evaluation.scale(1.2).move_to(UP * 3.5)
		
		h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
		h_line.next_to(evaluation, DOWN)
		self.add(evaluation)
		self.add(h_line)

		better = self.introduce_naive_idea(h_line)
		P_x = self.intro_pos_neg_pair(better)

		self.introduce_eval_scheme(P_x)

	def introduce_naive_idea(self, h_line):
		scale = 0.9
		polynomial = TexMobject(r"P(x) = p_0 + p_1x + p_2x^2 + \cdots + p_dx^d")
		polynomial.scale(scale)
		polynomial.next_to(h_line, DOWN)
		self.play(
			Write(polynomial)
		)

		number_points = TextMobject(r"Evaluate at $n \geq d + 1$ points")
		number_points.scale(0.8).next_to(polynomial, DOWN)

		self.play(
			Write(number_points)
		)

		self.wait()

		example_1 = TexMobject("(1, P(1))").scale(0.8).move_to(LEFT * 3 + DOWN * 0)
		example_2 = TexMobject("(2, P(2))").scale(0.8).next_to(example_1, DOWN)
		vdots = TexMobject(r"\vdots").scale(0.8).next_to(example_2, DOWN)
		example_n = TexMobject("(n, P(n))").scale(0.8).next_to(vdots, DOWN)

		f = lambda x: 0.05 * (x - 1) * (x - 4) * (x - 7)

		self.setup_axes()
		self.x_axis_label_mob.scale(0.8)
		self.y_axis_label_mob.scale(0.8)
		self.x_axis.remove(self.x_axis[1])
		self.y_axis.remove(self.y_axis[1])
		self.play(
			Write(self.axes)
		)

		graph = self.get_graph(f, x_min=-10, y_min=10)
		self.play(
			ShowCreation(graph)
		)

		dots = []
		
		d1 = self.show_point(f, 1, example_1, run_time=2)
		d2 = self.show_point(f, 2, example_2, run_time=2)
		d3_9 = self.show_lot_of_points(f, vdots)
		d10 = self.show_point(f, 10, example_n)

		dots.extend([d1, d2, d10] + d3_9)

		example_1_full = TexMobject(
			r"(1, p_0 + p_1 \cdot 1 + p_2 \cdot 1^2 + \cdots + p_d \cdot 1^d)"
			).scale(0.8).move_to(example_1.get_center()).set_color(YELLOW)

		example_2_full = TexMobject(
			r"(2, p_0 + p_1 \cdot 2 + p_2 \cdot 2^2 + \cdots + p_d \cdot 2^d))"
			).scale(0.8).next_to(example_1_full, DOWN).set_color(YELLOW)

		example_n_full = TexMobject(
			r"(n, p_0 + p_1 \cdot n + p_2 \cdot n^2 + \cdots + p_d \cdot n^d))"
			).scale(0.8).next_to(vdots, DOWN).set_color(YELLOW)

		self.wait(3)

		self.play(
			Transform(example_1, example_1_full)
		)

		self.play(
			Transform(example_2, example_2_full)
		)

		self.play(
			Transform(example_n, example_n_full)
		)

		self.wait(3)

		surround_rect = SurroundingRectangle(
			VGroup(example_1, example_2, vdots, example_n), color=BLUE, buff=SMALL_BUFF
		)

		brace = Brace(surround_rect, direction=DOWN)
		runtime = TexMobject(r"O(nd) \Leftrightarrow O(d^2)").scale(0.8).next_to(brace, DOWN)

		self.play(
			ShowCreation(surround_rect)
		)

		self.play(
			GrowFromCenter(brace)
		)

		self.play(
			Write(runtime)
		)

		self.wait(8)

		better = TextMobject("Can we do better?").scale(0.9).next_to(surround_rect, UP)
		self.play(
			Write(better)
		)

		self.wait(2)

		self.play(
			FadeOut(surround_rect),
			FadeOut(example_1),
			FadeOut(example_2),
			FadeOut(vdots),
			FadeOut(example_n),
			FadeOut(runtime),
			FadeOut(brace),
			FadeOut(polynomial),
			FadeOut(number_points),
			*[FadeOut(dot) for dot in dots],
			FadeOut(graph),
			better.next_to, h_line, DOWN,
			run_time=2
		)

		return better



	def show_point(self, f, n, text, color=YELLOW, run_time=1):
		dot = Dot().set_color(YELLOW)
		coord = (n, f(n))
		dot.move_to(self.coords_to_point(*coord))
		self.play(
			GrowFromCenter(dot)
		)

		text.set_color(color)

		self.play(
			TransformFromCopy(dot, text),
			run_time=run_time
		)

		return dot

	def show_lot_of_points(self, f, vdots, color=YELLOW):
		coords = [(i, f(i)) for i in range(3, 10)]
		dots = []
		for coord in coords:
			dot = Dot().move_to(self.coords_to_point(*coord)).set_color(color)
			dots.append(dot)

		vdots.set_color(color)

		self.play(
			*[GrowFromCenter(dot) for dot in dots],
			Write(vdots)
		)

		self.wait()
		return dots

	def intro_pos_neg_pair(self, better):
		P_x_quad = TexMobject("P(x) = x^2").scale(0.9).move_to(better.get_center())
		evaluate_n_8 = TextMobject("Evaluate at $n = 8$ points").scale(0.8).next_to(P_x_quad, DOWN)
		self.wait(6)
		self.play(
			ReplacementTransform(better, P_x_quad)
		)

		f = lambda x: 0.1 * x ** 2
		graph = self.get_graph(f).set_color(BLUE)
		self.play(
			ShowCreation(graph),
			Write(evaluate_n_8)
		)

		self.wait()


		which_points = TextMobject("Which points should we pick?").scale(0.8)

		which_points.move_to(LEFT * 3.5 + UP * 1)

		self.play(
			Write(which_points)
		)

		self.wait(9)

		points = [(1, 1), (-1, 1), (2, 4), (-2, 4), (3, 9), (-3, 9), (4, 16), (-4, 16)]

		points_text = self.organize_points(which_points, points)

		dots = self.draw_points(points, f)
		
		self.play(
			GrowFromCenter(dots[0])
		)

		self.play(
			TransformFromCopy(dots[0], points_text[0]),
			run_time=2
		)

		self.wait()

		self.play(
			GrowFromCenter(dots[1]),
			Write(points_text[1])
		)

		self.wait()

		self.play(
			GrowFromCenter(dots[2]),
			GrowFromCenter(dots[3]),
			run_time=1
		)

		self.play(
			TransformFromCopy(dots[2], points_text[2]),
			TransformFromCopy(dots[3], points_text[3]),
			run_time=2
		)

		self.wait(2)

		self.play(
			*[GrowFromCenter(dot) for dot in dots[4:]],
			*[Write(p_text) for p_text in points_text[4:]],
			run_time=2
		)

		self.wait(7)

		even_function = TexMobject("P(-x) = P(x)").scale(0.9).next_to(points_text, DOWN)
		self.play(
			Write(even_function)
		)

		self.wait(7)

		P_x_cube = TexMobject("P(x) = x^3").scale(0.9).move_to(better.get_center())
		self.play(
			ReplacementTransform(P_x_quad, P_x_cube)
		)

		self.wait()

		f = lambda x: 0.01 * x ** 3
		new_graph = self.get_graph(f).set_color(BLUE)

		self.play(
			FadeOut(points_text),
			FadeOut(dots),
			FadeOut(even_function),
			Transform(graph, new_graph),
			run_time=2
		)

		points = [(1, 1), (-1, -1), (2, 8), (-2, -8), (3, 27), (-3, -27), (4, 64), (-4, -64)]
		points_text = self.organize_points(which_points, points)

		dots = self.draw_points(points, f)

		self.play(
			GrowFromCenter(dots[0])
		)

		self.play(
			TransformFromCopy(dots[0], points_text[0]),
			run_time=2
		)

		self.play(
			GrowFromCenter(dots[1]),
			Write(points_text[1]),
			run_time=1
		)

		self.play(
			GrowFromCenter(dots[2]),
			GrowFromCenter(dots[3]),
		)

		self.play(
			TransformFromCopy(dots[2], points_text[2]),
			TransformFromCopy(dots[3], points_text[3]),
			run_time=2
		)

		self.play(
			*[GrowFromCenter(dot) for dot in dots[4:]],
			*[Write(p_text) for p_text in points_text[4:]],
			run_time=2
		)

		odd_func = TexMobject("P(-x) = -P(x)").scale(0.9).next_to(points_text, DOWN)

		self.play(
			Write(odd_func)
		)

		self.wait(11)

		question = TextMobject("Only need 4 points!").scale(0.9).next_to(odd_func, DOWN)

		self.play(
			Write(question)
		)

		self.wait(6)

		self.play(
			FadeOut(which_points),
			FadeOut(graph),
			FadeOut(dots),
			FadeOut(points_text),
			FadeOut(evaluate_n_8),
			FadeOut(odd_func),
			FadeOut(self.axes),
			FadeOut(question)
		)

		return P_x_cube

	def organize_points(self, which_points, points):	
		points_text = []
		i = 0
		for x, y in points:
			text = TexMobject("({0}, {1})".format(x, y)).scale(0.8)
				
			if i == 1:
				text.next_to(points_text[0], RIGHT, buff=0.7)
			elif i > 1:
				text.next_to(points_text[i - 2], DOWN)

			if i % 2 == 0:
				text.set_color(YELLOW)
			else:
				text.set_color(ORANGE)

			points_text.append(text)
			i += 1

		points_text = VGroup(*points_text).next_to(which_points, DOWN)

		return points_text

	def draw_points(self, points, f):
		dots = []
		i = 0
		for x, y in points:
			dot = Dot()
			x, y = x * 2, f(x * 2) # conversion to axes scale
			dot.move_to(self.coords_to_point(x, y))
			if i % 2 == 0:
				dot.set_color(YELLOW)
			else:
				dot.set_color(ORANGE)

			dots.append(dot)
			i += 1

		return VGroup(*dots)


	def introduce_eval_scheme(self, P_x_cube):
		P_x = TexMobject("P(x) = 3x^5 + 2x^4 + x^3 + 7x^2 + 5x + 1")
		P_x.scale(0.9).move_to(P_x_cube.get_center())

		self.play(
			ReplacementTransform(P_x_cube, P_x)
		)

		evaluate = TextMobject(r"Evaluate at $n$ points $\pm x_1, \pm x_2, \ldots, \pm x_{n/2}$")
		evaluate.scale(0.8).next_to(P_x, DOWN)

		self.play(
			Write(evaluate)
		)

		self.wait(2)

		P_x_split = TexMobject(
			"P(x) = (2x^4 + 7x^2 + 1) + (3x^5 + x^3 + 5x)"
		).scale(0.9).next_to(evaluate, DOWN)

		self.play(
			Write(P_x_split[0][:5])
		)

		self.wait()

		self.play(

			FadeIn(P_x_split[0][5]),
			TransformFromCopy(P_x[0][9:12], P_x_split[0][6:9]),
			FadeIn(P_x_split[0][9]),
			TransformFromCopy(P_x[0][16:19], P_x_split[0][10:13]),
			FadeIn(P_x_split[0][13]),
			TransformFromCopy(P_x[0][-1], P_x_split[0][14]),
			FadeIn(P_x_split[0][15]),
			FadeIn(P_x_split[0][16]),
			FadeIn(P_x_split[0][17]),
			TransformFromCopy(P_x[0][5:8], P_x_split[0][18:21]),
			FadeIn(P_x_split[0][21]),
			TransformFromCopy(P_x[0][13:15], P_x_split[0][22:24]),
			FadeIn(P_x_split[0][24]),
			TransformFromCopy(P_x[0][-4:-2], P_x_split[0][25:27]),
			FadeIn(P_x_split[0][27]),
			run_time=2
		)

		self.wait(3)

		P_x_split_full = TexMobject(
			"P(x) = (2x^4 + 7x^2 + 1) + x(3x^4 + x^2 + 5)"
		).scale(0.9).move_to(P_x_split.get_center())

		self.play(
			ReplacementTransform(P_x_split, P_x_split_full)
		)

		self.wait(6)

		brace_1 = Brace(P_x_split_full[0][5:16], direction=DOWN)
		brace_2 = Brace(P_x_split_full[0][18:], direction=DOWN)

		self.play(
			GrowFromCenter(brace_1),
			GrowFromCenter(brace_2)
		)

		P_e_x2 = TexMobject("P_e(x^2)").scale(0.8).next_to(brace_1, DOWN, buff=SMALL_BUFF)
		P_o_x2 = TexMobject("P_o(x^2)").scale(0.8).next_to(brace_2, DOWN, buff=SMALL_BUFF)

		self.play(
			Write(P_e_x2),
			Write(P_o_x2)
		)

		self.wait()

		simplified_split = TexMobject("P(x) = P_e(x^2) + xP_o(x^2)").scale(0.9).move_to(DOWN * 0.5)
		self.play(
			Write(simplified_split)
		)

		self.wait(3)

		positive_case = TexMobject("P(x_i) = P_e(x_i^2) + x_iP_o(x_i^2)").scale(0.9).next_to(simplified_split, DOWN)
		negative_case = TexMobject("P(-x_i) = P_e(x_i^2) - x_iP_o(x_i^2)").scale(0.9).next_to(positive_case, DOWN)

		p_n_group = VGroup(positive_case, negative_case)


		self.play(
			Write(positive_case),
			Write(negative_case)
		)

		self.wait()

		surround_rect = SurroundingRectangle(p_n_group, color=GREEN_SCREEN, buff=SMALL_BUFF)
		self.play(
			ShowCreation(surround_rect)
		)

		
		right_brace = Brace(surround_rect, direction=RIGHT)
		overlap = TextMobject("Lot of overlap!").scale(0.8).next_to(right_brace, RIGHT, buff=SMALL_BUFF)


		self.play(
			GrowFromCenter(right_brace),
			Write(overlap)
		)

		self.wait(5)

		even_odd_def = TexMobject(r"P_e(x^2) = 2x^2 + 7x + 1 \quad P_o(x^2) = 3x^2 + x + 5").scale(0.8).next_to(surround_rect, DOWN)

		self.play(
			Write(even_odd_def)
		)

		half_degree = TextMobject(r"$P_e(x^2)$ and $P_o(x^2)$ have degree 2!").scale(0.8).next_to(even_odd_def, DOWN)
		self.play(
			Write(half_degree)
		)

		self.wait(6)

		P_gen = TexMobject(r"P(x) = p_0 + p_1x + p_2x^2 + \cdots + p_{n -1}x^{n - 1}").scale(0.9).move_to(P_x.get_center())

		self.play(
			ReplacementTransform(P_x, P_gen),
			FadeOut(P_x_split_full),
			FadeOut(brace_1),
			FadeOut(brace_2),
			FadeOut(P_e_x2),
			FadeOut(P_o_x2),
			FadeOut(even_odd_def),
			FadeOut(half_degree),
			simplified_split.shift, UP * 1.8,
			surround_rect.shift, UP * 1.8,
			p_n_group.shift, UP * 1.8,
			right_brace.shift, UP * 1.8,
			overlap.shift, UP * 1.8,
			run_time=2
		)

		self.wait(9)

		half_degree = TextMobject(r"$P_e(x^2)$ and $P_o(x^2)$ have degree $n / 2 - 1$!").scale(0.8).next_to(surround_rect, DOWN)
		self.play(
			Write(half_degree)
		)

		self.wait(11)

		simpler_problem = TextMobject(r"Evaluate $P_e(x^2)$ and $P_o(x^2)$ each at $x_1^2, x_2^2, \ldots , x_{n / 2}^2$ ($n / 2$ points)").scale(0.8).next_to(half_degree, DOWN)

		self.play(
			Write(simpler_problem)
		)

		self.wait(7)

		large_brace = Brace(simpler_problem, direction=DOWN, buff=SMALL_BUFF)

		same_process = TextMobject("Same process on simpler problem").scale(0.9).next_to(large_brace, DOWN, buff=SMALL_BUFF)

		self.play(
			GrowFromCenter(large_brace)
		)

		self.play(
			Write(same_process)
		)

		self.wait(11)

class BigPictureEval(Scene):
	def construct(self):
		# evaluation = TextMobject("Big Picture")
		# evaluation.scale(1.2).move_to(UP * 3.5)
		
		# h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
		# h_line.next_to(evaluation, DOWN)
		# self.play(
		# 	Write(evaluation),
		# 	ShowCreation(h_line)
		# )

		self.show_abstractions()

	def show_abstractions(self):

		poly_text_1 = r"P(x): [p_0, p_1, \ldots, p_{n - 1}]"
		point_text_1 = r"[\pm x_1, \pm x_2, \ldots, \pm x_{n / 2}]"
		component_1 = self.make_eval_component(poly_text_1, point_text_1)

		component_1.move_to(UP * 3)

		self.play(
			ShowCreation(component_1),
			run_time=4
		)

		self.wait(4)

		buff_between = SMALL_BUFF * 3

		split_text = TexMobject("P(x) = P_e(x^2) + xP_o(x^2)")
		rect_split_text = SurroundingRectangle(split_text, buff=SMALL_BUFF, color=BLUE)

		component_2 = VGroup(rect_split_text, split_text)

		component_2.next_to(component_1, DOWN, buff=buff_between)

		# start = component_1[0].get_center() + DOWN * component_1[0].get_height() / 2
		# end = component_2[0].get_center() + UP * component_2[0].get_height() / 2

		# arrow_1 = Arrow(start, end).set_color(ORANGE)

		# self.play(
		# 	ShowCreation(arrow_1)
		# )

		self.play(
			ShowCreation(component_2),
			run_time=3
		)

		self.wait()

		poly_text_2 = r"P_e(x^2): [p_0, p_2, \ldots, p_{n - 2}]"
		point_text_2 = r"[x_1^2, x_2^2, \ldots, x_{n / 2}^2]"
		component_3 = self.make_eval_component(poly_text_2, point_text_2)
		output_3 = TexMobject(r"[P_e(x_1^2), P_e(x_2^2), \ldots, P_e(x_{n / 2}^2)]").scale(0.7)
		output_3.next_to(component_3, DOWN, buff=SMALL_BUFF)
		component_3.add(output_3)

		component_3.next_to(component_2, DOWN, buff=buff_between).shift(LEFT * 3.5)

		poly_text_3 = r"P_o(x^2): [p_1, p_3, \ldots, p_{n - 1}]"
		point_text_3 = r"[x_1^2, x_2^2, \ldots, x_{n / 2}^2]"
		component_4 = self.make_eval_component(poly_text_3, point_text_3)

		component_4.next_to(component_2, DOWN, buff=buff_between).shift(RIGHT * 3.5)
		output_4 = TexMobject(r"[P_o(x_1^2), P_o(x_2^2), \ldots, P_o(x_{n / 2}^2)]").scale(0.7)
		output_4.next_to(component_4, DOWN, buff=SMALL_BUFF)
		component_4.add(output_4)

		self.play(
			FadeIn(component_3),
			FadeIn(component_4)
		)
		self.wait(14)

		conquer_text_1 = TexMobject("P(x_i) = P_e(x_i^2) + x_iP_o(x_i^2)").scale(0.8)
		conquer_text_2 = TexMobject("P(-x_i) = P_e(x_i^2) - x_iP_o(x_i^2)").scale(0.8)
		conquer_text_3 = TexMobject(r"i = \{1, 2, \ldots, n / 2 \}").scale(0.8)
		conquer_text = VGroup(conquer_text_1, conquer_text_2, conquer_text_3).arrange_submobjects(DOWN, buff=SMALL_BUFF * 2)
		rect_split_text = SurroundingRectangle(conquer_text, buff=SMALL_BUFF, color=ORANGE)

		component_5 = VGroup(rect_split_text, conquer_text)

		component_5.next_to(component_3, DOWN, buff=buff_between).shift(RIGHT * 3.5)

		self.play(
			FadeIn(component_5)
		)

		self.wait(7)

		result = TexMobject(r"[P(x_1), P(-x_1), \ldots, P(x_{n / 2}), P(-x_{n / 2})]").scale(0.8)
		result_box = SurroundingRectangle(result, buff=SMALL_BUFF, color=GREEN_SCREEN)

		component_6 = VGroup(result_box, result)
		component_6.next_to(component_5, DOWN, buff=buff_between)

		self.play(
			FadeIn(component_6)
		)

		self.wait(2)

		all_components = VGroup(
			component_1, 
			component_2,
			component_3,
			component_4,
			component_5,
			component_6,
		)

		group_comp_3_4 = VGroup(component_3, component_4)

		problem_rect = SurroundingRectangle(group_comp_3_4, color=BRIGHT_RED, buff=SMALL_BUFF)

		self.play(
			all_components.scale, 0.6,
			all_components.shift, UP * 1.5,
			run_time=3
		)

		self.wait()



		recursive = TextMobject(r"$O(n \log n)$ Recursive Algorithm").scale(0.9)
		problem = TextMobject("One major problem").scale(0.9)
		# question = TextMobject("Can you spot the issue?").scale(0.9)

		text = VGroup(recursive, problem).arrange_submobjects(DOWN)
		text.next_to(all_components, DOWN)

		self.play(
			Write(recursive)
		)

		self.wait(11)

		self.play(
			Write(problem)
		)

		self.wait()

		self.play(
			all_components.scale, 1 / 0.6,
			all_components.shift, DOWN * 1.5,
			FadeOut(text),
			run_time=2
		)

		self.wait(3)

		self.play(
			ShowCreation(problem_rect)
		)

		self.wait(2)

		problem_group = VGroup(problem_rect, group_comp_3_4)

		self.play(
			FadeOut(component_2),
			FadeOut(component_5),
			FadeOut(component_6),
			problem_group.next_to, component_1, DOWN, buff_between,
			run_time=2
		)

		problem_desc_1 = TextMobject(r"Points $[\pm x_1, \pm x_2, \ldots, \pm x_{n / 2}]$ are $\pm$ paired.").scale(0.9).next_to(problem_group, DOWN)

		problem_desc_2 = TextMobject(r"Points $[x_1^2, x_2^2, \ldots, x_{n / 2}^2]$ are not $\pm$ paired.").scale(0.9).next_to(problem_desc_1, DOWN)

		conclusion = TextMobject("Recursion breaks!").scale(0.9).next_to(problem_desc_2, DOWN)

		question = TextMobject(r"Is it possible to make $[x_1^2, x_2^2, \ldots, x_{n / 2}^2] $ $\pm$ paired?").scale(0.9).next_to(conclusion, DOWN)

		idea = TextMobject(r"Some of original $[\pm x_1, \pm x_2, \ldots, \pm x_{n / 2}]$ need to be complex numbers!").scale(0.9).next_to(question, DOWN)

		self.play(
			Write(problem_desc_1)
		)

		self.wait(4)

		self.play(
			Write(problem_desc_2)
		)

		self.wait(5)

		self.play(
			Write(conclusion)
		)

		self.wait(2)

		self.play(
			Write(question)
		)

		self.wait(13)

		self.play(
			Write(idea)
		)

		self.wait(13)

		new_question = TextMobject("But which set of complex numbers should we choose?")
		new_question.scale(0.9).next_to(question, DOWN)
		self.play(
			ReplacementTransform(idea, new_question)
		)

		self.wait(9)

	def make_eval_component(self, poly_text, point_text):
		evaluate = TextMobject("Evaluate")

		poly_input = TexMobject(poly_text).scale(0.7)
		point_input = TexMobject(point_text).scale(0.7)

		box_inputs = VGroup(poly_input, point_input).arrange_submobjects(DOWN)

		evaluate.next_to(box_inputs, LEFT)

		inside = VGroup(evaluate, box_inputs)

		box = SurroundingRectangle(inside, buff=SMALL_BUFF)

		component = VGroup(box, inside)

		return component

class IntroduceRootsOfUnity(Scene):
	def construct(self):
		self.show_degree3_example()

	def show_degree3_example(self):
		P_x = TexMobject("P(x) = x^3 + x^2 - x - 1").move_to(UP * 3.5)

		self.play(
			Write(P_x)
		)

		self.wait(9)

		scale = 0.9

		top_level = []
		x_1 = TexMobject("x_1").scale(scale).move_to(LEFT * 4.5)
		x_1_neg = TexMobject("-x_1").scale(scale).move_to(LEFT * 1.5)
		x_2 = TexMobject("x_2").scale(scale).move_to(RIGHT * 1.5)
		x_2_neg = TexMobject("-x_2").scale(scale).move_to(RIGHT * 4.5)

		top_level.extend([x_1, x_1_neg, x_2, x_2_neg])

		top_level_group = VGroup(*top_level)
		for i in range(len(top_level_group)):
			self.play(
				Write(top_level_group[i])
			)

		mid_level = []
		x_1_squared = TexMobject("x_1^2").scale(scale).move_to(LEFT * 3 + DOWN * 1.5)
		x_2_squared = TexMobject("x_2^2").scale(scale).move_to(RIGHT * 3 + DOWN * 1.5)

		mid_level.extend([x_1_squared, x_2_squared])
		mid_level_group = VGroup(*mid_level)

		tree = VGroup()
		for i, src in enumerate(top_level_group):
			dest = mid_level[i // 2]
			line = self.create_tree_edge(src, dest)
			tree.add(line)

		self.wait(7)

		self.play(
			ShowCreation(tree)
		)

		self.play(
			Write(mid_level_group)
		)

		self.wait(12)

		x_1_squared_neg = TexMobject("-x_1^2").scale(scale).move_to(x_2_squared.get_center())

		self.play(
			Transform(mid_level_group[1], x_1_squared_neg)
		)

		self.wait(3)

		root = TexMobject("x_1^4").scale(scale).move_to(DOWN * 3)
		for i, src in enumerate(mid_level_group):
			dest = root
			line = self.create_tree_edge(src, dest)
			tree.add(line)

		self.play(
			ShowCreation(tree[-2:])
		)

		self.play(
			Write(root)
		)

		self.wait(8)

		define = TextMobject(r"Let $x_1 = 1$").scale(scale)
		result = TexMobject(r"\Rightarrow x_2 = i").scale(scale)

		text_group = VGroup(define, result).arrange_submobjects(DOWN).move_to(RIGHT * 5 + DOWN * 3)
		self.play(
			Write(define)
		)

		self.wait(2)

		one = TexMobject("1").scale(scale)
		one_neg = TexMobject("-1").scale(scale)
		imag = TexMobject("i").scale(scale)
		imag_neg = TexMobject("-i").scale(scale)

		self.play(
			Transform(top_level_group[0], 
				one.copy().move_to(top_level_group[0].get_center())),
			Transform(top_level_group[1], 
				one_neg.copy().move_to(top_level_group[1].get_center()))
		)

		self.wait(6)

		self.play(
			Transform(mid_level_group[0], 
				one.copy().move_to(mid_level_group[0].get_center()))
		)

		self.play(
			Transform(mid_level_group[1], 
				one_neg.copy().move_to(mid_level_group[1].get_center()))
		)

		self.wait(5)

		self.play(
			Transform(root, 
				one.copy().move_to(root.get_center()))
		)

		self.wait(10)

		self.play(
			Write(result)
		)

		self.wait(5)

		self.play(
			Transform(top_level_group[2], 
				imag.copy().move_to(top_level_group[2].get_center())),
			Transform(top_level_group[3], 
				imag_neg.copy().move_to(top_level_group[3].get_center()))
		)

		self.wait(4)

		other_perspective = TextMobject("Alternative Perspective").scale(scale)
		
		other_perspective.next_to(P_x, DOWN)
		self.play(
			Write(other_perspective),
			FadeOut(text_group)
		)

		self.wait()

		arrow = Arrow().scale(0.8).next_to(root, LEFT).set_color(GREEN_SCREEN)
		label = TexMobject("x_i^4").scale(scale).next_to(arrow, LEFT)

		self.play(
			ShowCreation(arrow)
		)

		self.play(
			Write(label)
		)

		equation = TextMobject(r"Solution to $x^4 = 1$").scale(scale).next_to(other_perspective, DOWN)
		roots_of_unity =  TextMobject(r"Points are $4^{\text{th}}$ roots of unity!").scale(scale).next_to(equation, DOWN)

		self.play(
			Write(equation)
		)

		self.wait(14)

		self.play(
			Write(roots_of_unity)
		)

		self.wait()

		generalize = TextMobject("Does this generalize?").scale(scale).next_to(roots_of_unity, DOWN)

		self.play(
			Write(generalize)
		)

		self.wait()

		P_x_deg_5 = TexMobject("P(x) = x^5 + 2x^4 - x^3 + x^2 + 1").scale(scale).move_to(P_x.get_center())

		self.play(
			Transform(P_x, P_x_deg_5),
			FadeOut(other_perspective),
			FadeOut(equation),
			FadeOut(roots_of_unity),
			FadeOut(generalize),
			FadeOut(arrow),
			FadeOut(label)
		)

		self.wait()

		eight_points = TextMobject(r"Need $n \geq 6$ points",  r"$\rightarrow$ let $n = 8$ (powers of 2 are convenient)").scale(scale).next_to(P_x, DOWN)
		self.play(
			Write(eight_points[0]),
			run_time=2
		)

		self.wait(6)

		self.play(
			Write(eight_points[1]),
			run_time=2
		)

		self.wait(3)

		level_8 = []

		x_1 = TexMobject("x_1").scale(scale).move_to(LEFT * 5.5 + UP)
		x_1_neg = TexMobject("-x_1").scale(scale).move_to(LEFT * 3.5 + UP)
		x_2 = TexMobject("x_2").scale(scale).move_to(LEFT * 2.5 + UP)
		x_2_neg = TexMobject("-x_2").scale(scale).move_to(LEFT * 0.5 + UP)
		x_3 = TexMobject("x_3").scale(scale).move_to(RIGHT * 0.5 + UP)
		x_3_neg = TexMobject("-x_3").scale(scale).move_to(RIGHT * 2.5 + UP)
		x_4 = TexMobject("x_4").scale(scale).move_to(RIGHT * 3.5 + UP)
		x_4_neg = TexMobject("-x_4").scale(scale).move_to(RIGHT * 5.5 + UP)

		level_8.extend([x_1, x_1_neg, x_2, x_2_neg, x_3, x_3_neg, x_4, x_4_neg])

		level_8_group = VGroup(*level_8)

		for i, src in enumerate(level_8):
			dest = top_level[i // 2]
			line = self.create_tree_edge(src, dest)
			tree.add(line)

		self.play(
			ShowCreation(tree[-8:])
		)

		self.wait()

		self.play(
			Write(level_8_group)
		)

		self.wait()

		label = TexMobject("x_i^8").scale(scale).next_to(arrow, LEFT)

		self.play(
			ShowCreation(arrow)
		)

		self.play(
			Write(label)
		)

		self.wait(2)

		roots_of_unity =  TextMobject(r"Points are $8^{\text{th}}$ roots of unity!").scale(scale).next_to(eight_points, DOWN)

		self.play(
			Write(roots_of_unity)
		)

		self.wait(2)

		P_gen = TexMobject(r"P(x) = p_0 + p_1x + p_2x^2 + \cdots + p_{d}x^{d}").scale(0.9).move_to(P_x.get_center())
		self.play(
			Transform(P_x, P_gen),
			FadeOut(arrow),
			FadeOut(label)
		)

		self.wait(3)

		n_points = TextMobject(r"Need $n \geq (d + 1)$ points, $n = 2^k, k \in \mathbb{Z}$").scale(scale).next_to(P_x, DOWN)

		self.play(
			ReplacementTransform(eight_points, n_points),
			run_time=2
		)

		self.wait(3)

		roots_of_unity_n =  TextMobject(r"Points are $n^{\text{th}}$ roots of unity!").scale(scale).next_to(n_points, DOWN)

		self.play(
			ReplacementTransform(roots_of_unity, roots_of_unity_n)
		)

		self.wait(9)

	def create_tree_edge(self, src, dest):
		line = Line(src.get_center(), dest.get_center())
		start = line.get_start() + line.get_unit_vector() * 0.3
		end = line.get_end() - line.get_unit_vector() * 0.3
		return Line(start, end)

class WhyRootsOfUnity(GraphScene):
	def construct(self):
		self.show_unit_circle()

	def show_unit_circle(self):
		plane = ComplexPlane().scale(2)
		plane.add_coordinates()

		subset_complex_plane = VGroup()
		
		outer_square = Square(side_length=4).set_stroke(BLUE_D)

		inner_major_axes = VGroup(
			Line(LEFT * 2, RIGHT * 2), 
			Line(DOWN * 2, UP * 2)
		).set_color(WHITE)

		faded_axes = VGroup(
			Line(LEFT * 2 + UP, RIGHT * 2 + UP), 
			Line(LEFT * 2 + DOWN, RIGHT * 2 + DOWN),
			Line(LEFT * 1 + UP * 2, LEFT * 1 + DOWN * 2), 
			Line(RIGHT * 1 + UP * 2, RIGHT * 1 + DOWN * 2),
		).set_stroke(color=BLUE_D, width=1, opacity=0.5)

		subset_complex_plane.add(outer_square)
		subset_complex_plane.add(inner_major_axes)
		subset_complex_plane.add(faded_axes)

		
		subset_complex_plane.add(plane.coordinate_labels[6].copy())
		subset_complex_plane.add(plane.coordinate_labels[7].copy())
		subset_complex_plane.add(plane.coordinate_labels[-4].copy())
		subset_complex_plane.add(plane.coordinate_labels[-3].copy())

		self.add(plane)
		self.add(subset_complex_plane)

		title = TextMobject(r"$N^{th}$ roots of unity").scale(1.2).move_to(LEFT * 4 + UP * 3)
		surround_rect = SurroundingRectangle(title, color=WHITE).set_stroke(width=1)
		surround_rect.set_fill(color=BLACK, opacity=0.7)
		title.add_to_back(surround_rect)

		self.wait()
		self.play(
			FadeIn(title)
		)

		self.wait()

		equation = TexMobject("z^n = 1").next_to(title, DOWN, buff=0.5)
		surround_rect_equation = SurroundingRectangle(equation, color=WHITE).set_stroke(width=1)
		surround_rect_equation.set_fill(color=BLACK, opacity=0.7)
		equation.add_to_back(surround_rect_equation)

		self.play(
			FadeIn(equation)
		)
		self.wait(2)

		unit_circle = Circle(radius=2).set_color(GREEN)
		self.play(
			ShowCreation(unit_circle)
		)
		points = [unit_circle.point_from_proportion(i / 16) for i in range(16)]

		dots = [Dot().set_color(YELLOW).move_to(points[i]) for i in range(len(points))]
		dots = VGroup(*dots)

		self.play(
			LaggedStartMap(
			GrowFromCenter, dots,
			run_time=2
			)
		)
		self.wait()

		dashed_line = DashedLine(ORIGIN, dots[1].get_center())
		
		arc_angle = Arc(angle=TAU/16)

		self.play(
			ShowCreation(dashed_line),
			ShowCreation(arc_angle)
		)
		angle = self.make_text(r"angle = $\frac{2 \pi}{n}$", scale=0.8)
		angle.next_to(dots[1], RIGHT)
		self.play(
			FadeIn(angle)
		)

		self.wait(5)

		euler_identity = self.make_text(r"$e^{i \theta} = \cos(\theta) + i \sin(\theta)$")
		euler_identity.move_to(UP * 3 + RIGHT * 3)
		self.play(
			FadeIn(euler_identity)
		)
		self.wait(7)

		define = self.make_text(r"$\omega = e^{\frac{2 \pi i}{n}}$").next_to(equation, DOWN, buff=0.5)
		self.play(
			FadeIn(define)
		)

		self.wait(8)

		self.play(
			FadeOut(angle),
			FadeOut(arc_angle),
			FadeOut(dashed_line)
		)

		w_0 = self.make_text(r"$\omega^0 = 1$", scale=0.7).next_to(dots[0], RIGHT)
		w_1 = self.make_text(r"$\omega^1 = e ^ {\frac{2 \pi i}{n}}$", scale=0.7).next_to(dots[1], RIGHT)
		w_2 = self.make_text(r"$\omega^2 = e ^ {\frac{2 \pi (2) i}{n}}$", scale=0.7).next_to(dots[2], RIGHT)
		w_n_1 = self.make_text(r"$\omega ^{n - 1} = e ^ {\frac{2 \pi (n - 1) i}{n}}$", scale=0.7).next_to(dots[-1], RIGHT)
		w_2.shift(UP * 0.2 + LEFT * 0.1)

		w_examples = VGroup(w_0, w_1, w_2, w_n_1)

		self.play(
			FadeIn(w_examples[0])
		)

		self.play(
			FadeIn(w_examples[1])
		)

		self.play(
			FadeIn(w_examples[2])
		)

		self.play(
			FadeIn(w_examples[3])
		)

		self.wait(5)

		evaluation = self.make_text(r"Evaluate $P(x)$ at $[1, \omega ^1, \omega ^2, \ldots, \omega ^ {n - 1}]$")
		evaluation.move_to(DOWN * 3)
		evaluation[1][14:].set_color(YELLOW)

		self.play(
			FadeIn(evaluation)
		)

		self.wait(10)

		self.play(
			FadeOut(plane),
			FadeOut(title),
			FadeOut(equation),
			FadeOut(define),
			FadeOut(euler_identity),
			FadeOut(w_examples),
			FadeOut(evaluation[0]),
			evaluation[1].next_to, subset_complex_plane, DOWN,
			run_time=2
		)

		why = TextMobject("Why does this work?")
		why.move_to(UP * 3.5)

		self.play(
			Write(why)
		)

		self.wait(11)

		pm_paired_fact = TextMobject(r"$ \omega ^ {j + n / 2} = - \omega ^ j \rightarrow (\omega ^j , \omega ^{j + n / 2})$ are $\pm$ paired").scale(0.8)
		pm_paired_fact.next_to(why, DOWN)

		self.play(
			Write(pm_paired_fact)
		)

		dashed_line = DashedLine(dots[2].get_center(), dots[10].get_center())
		pm_paired = self.make_text(r"$\pm$ paired").scale(0.7).next_to(dots[10], LEFT)

		self.play(
			ShowCreation(dashed_line),
		)

		self.play(
			Write(pm_paired)
		)

		self.wait(5)

		shift_amount = LEFT * 3.7
		self.play(
			subset_complex_plane.shift, shift_amount,
			dots.shift, shift_amount,
			dashed_line.shift, shift_amount,
			unit_circle.shift, shift_amount,
			pm_paired.shift, shift_amount,
			evaluation[1].scale, 0.7,
			evaluation[1].shift, shift_amount
		)

		recursion = TextMobject("Recursion")
		arrow = TexMobject(r"\Longrightarrow")

		recursion_step = VGroup(recursion, arrow).arrange_submobjects(DOWN)
		recursion_step.move_to(ORIGIN)

		self.play(
			Write(recursion_step)
		)

		right_plane = subset_complex_plane.copy()
		right_plane.shift(-shift_amount * 2)
		self.play(
			ShowCreation(right_plane),
			run_time=2
		)

		right_unit_circle = unit_circle.copy()

		right_unit_circle.shift(-shift_amount * 2)

		self.play(
			ShowCreation(right_unit_circle),
			run_time=1
		)

		self.wait()

		evaluation_right = TextMobject(r"Evaluate $P_e(x^2)$ and $P_o(x^2)$ at" + "\\\\" + 
			r"$[1, \omega ^2, \omega ^4, \ldots, \omega ^ {2(n / 2 - 1)}]$")
		evaluation_right[0][25:].set_color(ORANGE)
		evaluation_right.scale(0.7)
		evaluation_right.next_to(right_plane, DOWN)

		self.play(
			Write(evaluation_right)
		)

		self.wait()

		new_points = [right_unit_circle.point_from_proportion(i / 8) for i in range(8)]

		new_dots = [Dot().set_color(ORANGE).move_to(new_points[i]) for i in range(len(new_points))]
		new_dots = VGroup(*new_dots)

		transforms = []
		for i in range(len(dots)):
			if i < len(new_dots):
				transform = TransformFromCopy(dots[i], new_dots[i])
				transforms.append(transform)
			else:
				transform = TransformFromCopy(dots[i], new_dots[i % len(new_dots)].copy())
				transforms.append(transform)

		for transform in transforms:
			self.play(
				transform
			)

		self.wait()

		dashed_line_right = DashedLine(new_dots[1].get_center(), new_dots[5].get_center())
		pm_paired_right = self.make_text(r"$\pm$ paired").scale(0.7).next_to(new_dots[5], LEFT)

		self.play(
			ShowCreation(dashed_line_right),
		)

		self.play(
			Write(pm_paired_right)
		)

		self.wait()

		n_case = TextMobject(r"$n$ roots of unity").scale(0.7).next_to(evaluation[1], DOWN)

		n_2_case = TextMobject(r"$(n / 2)$ roots of unity").scale(0.7).next_to(evaluation_right, DOWN)
		n_2_case.shift(UP * SMALL_BUFF)
		self.play(
			Write(n_case)
		)

		self.play(
			Write(n_2_case)
		)

		self.wait(10)

	def make_text(self, string, scale=1):
		text = TextMobject(string).scale(scale)
		surround_rect = SurroundingRectangle(text, color=WHITE).set_stroke(width=1)
		surround_rect.set_fill(color=BLACK, opacity=0.7)
		text.add_to_back(surround_rect)
		return text

class FFTImplementationP1(Scene):
	def construct(self):
		diagram = self.show_diagram()

		self.show_implementation(diagram)

	def show_diagram(self):
		diagram = VGroup()
		
		poly_rep = r"P(x): [p_0, p_1, \ldots, p_{n - 1}]"
		point_text = r"\omega = e^{\frac{2 \pi i}{n}}: [\omega^0, \omega^1, \ldots, \omega^{n - 1}]"
		fft_input = self.make_eval_component(poly_rep, point_text)
		fft_input.move_to(UP * 3)

		self.play(
			ShowCreation(fft_input),
			run_time=3
		)
		self.wait(17)

		diagram.add(fft_input)

		base_case_text = TextMobject(r"$n = 1 \Rightarrow P(1)$").scale(0.8)
		base_case_rect = SurroundingRectangle(base_case_text, color=BLUE, buff=SMALL_BUFF)
		base_case_component = VGroup(base_case_rect, base_case_text)
		base_case_component.next_to(fft_input, DOWN)

		self.play(
			ShowCreation(base_case_component),
			run_time=2
		)
		self.wait(8)

		diagram.add(base_case_component)

		poly_even = r"P_e(x^2): [p_0, p_2, \ldots, p_{n - 2}]"
		point_even = r"[\omega^0, \omega^2, \ldots, \omega^{n - 2}]"
		fft_even =  self.make_eval_component(poly_even, point_even)
		


		poly_odd = r"P_o(x^2): [p_1, p_3, \ldots, p_{n - 1}]"
		point_odd = r"[\omega^0, \omega^2, \ldots, \omega^{n - 2}]"
		fft_odd =  self.make_eval_component(poly_odd, point_odd)

		recursive_component = VGroup(fft_even, fft_odd).arrange_submobjects(RIGHT, buff=1)
		recursive_component.next_to(base_case_component, DOWN)
		
		self.play(
			FadeIn(fft_even),
			FadeIn(fft_odd)
		)

		self.wait(16)

		diagram.add(recursive_component)

		output_even = TextMobject(r"$y_e = [P_e(\omega^0), P_e(\omega^2), \ldots , P_e(\omega ^ {n - 2})]$").scale(0.7)
		output_even.next_to(fft_even, DOWN, buff=SMALL_BUFF)

		output_odd = TextMobject(r"$y_o = [P_o(\omega^0), P_o(\omega^2), \ldots , P_o(\omega ^ {n - 2})]$").scale(0.7)
		output_odd.next_to(fft_odd, DOWN, buff=SMALL_BUFF)

		self.play(
			Write(output_even),
			Write(output_odd)
		)

		self.wait(20)

		recursive_component[0].add(output_even)
		recursive_component[1].add(output_odd)

		conquer_text_1 = TexMobject("P(x_j) = P_e(x_j^2) + x_jP_o(x_j^2)").scale(0.7)
		conquer_text_2 = TexMobject("P(-x_j) = P_e(x_j^2) - x_jP_o(x_j^2)").scale(0.7)
		conquer_text_3 = TexMobject(r"j \in \{0, 1, \ldots (n / 2 - 1)\}").scale(0.7)
		conquer_text = VGroup(conquer_text_1, conquer_text_2, conquer_text_3).arrange_submobjects(DOWN)
		rect_split_text = SurroundingRectangle(conquer_text, buff=SMALL_BUFF, color=ORANGE)

		conquer_component = VGroup(rect_split_text, conquer_text)
		conquer_component.next_to(recursive_component, DOWN)
		
		self.play(
			FadeIn(conquer_component)
		)
		self.wait(15)

		diagram.add(conquer_component)

		side_facts = VGroup()

		fact_1 = TexMobject(r"x_j = \omega^j").scale(0.7)
		fact_2 = TexMobject(r"-\omega^j = \omega^{j + n / 2}").scale(0.7)
		left_fact = VGroup(fact_1, fact_2).arrange_submobjects(DOWN).next_to(conquer_component, LEFT, buff=1.5)
		side_facts.add(left_fact)

		fact_3 = TexMobject(r"y_e[j] = P_e(\omega^{2j})").scale(0.7)
		fact_4 = TexMobject(r"y_o[j] = P_o(\omega^{2j})").scale(0.7)
		right_fact = VGroup(fact_3, fact_4).arrange_submobjects(DOWN).next_to(conquer_component, RIGHT, buff=1.5)
		side_facts.add(right_fact)

		self.play(
			Write(fact_1)
		)
		self.wait(3)

		new_conquer_text_1 = TexMobject(
			r"P(\omega^j) = P_e(\omega^{2j}) + \omega ^ j P_o(\omega^{2j})"
			).scale(0.7).move_to(conquer_text_1.get_center())
		new_conquer_text_2 = TexMobject(
			r"P(-\omega^j) = P_e(\omega^{2j}) - \omega ^ j P_o(\omega^{2j})"
			).scale(0.7).move_to(conquer_text_2.get_center())

		new_rect = SurroundingRectangle(
			VGroup(new_conquer_text_1, new_conquer_text_2, conquer_text[2]),
			color=ORANGE,
			buff=SMALL_BUFF
		)

		self.play(
			Transform(conquer_text[0], new_conquer_text_1),
			Transform(conquer_text[1], new_conquer_text_2),
			Transform(conquer_component[0], new_rect),
			run_time=2
		)
		self.wait(3)

		self.play(
			Write(fact_2)
		)
		self.wait(9)

		new_conquer_text_2 = TexMobject(
			r"P(\omega^{j + n / 2}) = P_e(\omega^{2j}) - \omega ^ j P_o(\omega^{2j})"
			).scale(0.7).move_to(conquer_text_2.get_center())

		new_rect = SurroundingRectangle(
			VGroup(conquer_text[0], new_conquer_text_2, conquer_text[2]),
			color=ORANGE,
			buff=SMALL_BUFF
		)

		self.play(
			Transform(conquer_text[1], new_conquer_text_2),
			Transform(conquer_component[0], new_rect),
			run_time=2
		)

		self.wait(3)

		self.play(
			Write(fact_3),
			Write(fact_4),
			run_time=2
		)

		self.wait(11)

		new_conquer_text_1 = TexMobject(
			r"P(\omega^j) = y_e[j] + \omega ^ j y_o[j])"
			).scale(0.7).move_to(conquer_text_1.get_center())
		new_conquer_text_2 = TexMobject(
			r"P(\omega^{j + n / 2}) = y_e[j] - \omega ^ j y_o[j])"
			).scale(0.7).move_to(conquer_text_2.get_center())

		new_rect = SurroundingRectangle(
			VGroup(new_conquer_text_1, new_conquer_text_2, conquer_text[2]),
			color=ORANGE,
			buff=SMALL_BUFF
		)

		self.play(
			Transform(conquer_text[0], new_conquer_text_1),
			Transform(conquer_text[1], new_conquer_text_2),
			Transform(conquer_component[0], new_rect),
			run_time=2
		)

		self.wait(11)

		result_text = TexMobject(r"y = [P(\omega^0), P(\omega^1), \ldots , P(\omega^{n - 1})]")
		result_text.scale(0.8)
		result_rect = SurroundingRectangle(result_text, color=GREEN_SCREEN, buff=SMALL_BUFF)
		result_component = VGroup(result_rect, result_text)
		result_component.next_to(conquer_component, DOWN)

		self.play(
			FadeIn(result_component)
		)

		self.wait(7)

		diagram.add(result_component)

		self.play(
			FadeOut(side_facts)
		)

		shifted_and_scaled_diagram = diagram.copy()
		shifted_and_scaled_diagram.scale(0.8)

		shifted_and_scaled_diagram[2][0].next_to(shifted_and_scaled_diagram[1], DOWN)
		shifted_and_scaled_diagram[2][1].next_to(shifted_and_scaled_diagram[2][0], DOWN)
		shifted_and_scaled_diagram[3].next_to(shifted_and_scaled_diagram[2][1], DOWN)
		shifted_and_scaled_diagram[4].next_to(shifted_and_scaled_diagram[3], DOWN)

		shifted_and_scaled_diagram.move_to(RIGHT * 3.5)

		self.play(
			Transform(diagram, shifted_and_scaled_diagram),
			run_time=3
		)

		return diagram

	def show_implementation(self, diagram):
		code = self.get_code()

		self.play(
			Write(code[0])
		)

		self.wait()

		self.play(
			diagram[2][0].set_stroke, YELLOW, None, 0.3,
			diagram[2][1].set_stroke, YELLOW, None, 0.3,
			diagram[3].set_stroke, ORANGE, None, 0.3,
			diagram[4].set_stroke, GREEN_SCREEN, None, 0.3,
			self.play(
				Write(code[1])
			)
		)

		self.wait(3)

		self.play(
			Write(code[2])
		)

		self.wait(17)

		self.play(
			Write(code[3])
		)

		self.wait()

		self.play(
			Write(code[4])
		)

		self.wait(7)

		self.play(
			Write(code[5])
		)

		self.wait(3)

		self.play(
			diagram[0].set_stroke, YELLOW, None, 0.3,
			diagram[1].set_stroke, BLUE, None, 0.3,
			diagram[2][0].set_stroke, YELLOW, None, 1,
			diagram[2][1].set_stroke, YELLOW, None, 1,
		)

		self.wait(3)

		self.play(
			Write(code[6]),
			run_time=2
		)

		self.wait(3)

		self.play(
			Write(code[7]),
			run_time=2
		)

		self.wait(9)

		self.play(
			diagram[2][0].set_stroke, YELLOW, None, 0.3,
			diagram[2][1].set_stroke, YELLOW, None, 0.3,
			diagram[3].set_stroke, ORANGE, None, 1,
			diagram[4].set_stroke, GREEN_SCREEN, None, 1,
		)

		self.wait(3)

		self.play(
			Write(code[8])
		)

		self.wait(2)

		self.play(
			Write(code[9])
		)

		self.wait()

		self.play(
			Write(code[10])
		)

		self.wait()

		self.play(
			Write(code[11])
		)

		self.wait()

		self.play(
			Write(code[12])
		)

		self.wait()

		self.play(
			diagram[0].set_stroke, YELLOW, None, 1,
			diagram[1].set_stroke, BLUE, None, 1,
			diagram[2][0].set_stroke, YELLOW, None, 1,
			diagram[2][1].set_stroke, YELLOW, None, 1,
		)

		self.wait(7) # change to 10 for part 1

		### UNCOMMENT FOR PART 2

		# self.play(
		# 	FadeOut(diagram)
		# )

		# human = create_normal_human_char()
		# human.shift(RIGHT * 4)
		# self.play(
		# 	FadeIn(human)
		# )

		# curve, path = self.create_trajectory(human[0])

		# fft = TextMobject("FFT").scale(0.7)
		# shift = UP * 1.2
		# new_path = [point + shift for point in path]
		# new_curve = VGroup()
		# new_curve.set_points_smoothly(*[new_path])
		
		# fft.move_to(path[0] + shift)

		# self.play(
		# 	ShowCreation(curve),
		# 	MoveAlongPath(fft, new_curve),
		# 	run_time=3,
		# 	# rate_func=linear
		# )

		# self.play(
		# 	fft.next_to, curve, UP,
		# 	run_time=2
		# )

		# self.wait(10)

		# self.play(
		# 	FadeOut(human),
		# 	FadeOut(fft),
		# 	FadeOut(curve),
		# 	code.scale, 0.8,
		# 	code.move_to, RIGHT * 4 + UP * 1.5,
		# 	run_time=3
		# )

		# self.wait()

	def create_trajectory(self, face):
		scaled_face = face.copy().scale(1.3)
		path = []

		n = 30
		for i in range(n + 1):
			point = scaled_face.point_from_proportion((n - i) / (n * 2))
			path.append(point)
			
		path_left = [path[0] + LEFT * (1 - i) for i in np.arange(0, 1, 0.1)]
		path_right = [path[-1] + RIGHT * i for i in np.arange(0.1, 1.1, 0.1)]
		path.insert(0, path_left[0])
		path.append(path[-1] + RIGHT)
		sample_points = [Dot().move_to(point) for point in path]

		curve = VGroup()
		line_l = Line(path[0], path[1])
		arc = ArcBetweenPoints(path[1], path[-2], angle=-TAU/2)
		line_r = Line(path[-2], path[-1])
		curve.add(line_l)
		curve.add(arc)
		curve.add(line_r)
		arrow = Arrow(path[-2], path[-1] + RIGHT * MED_SMALL_BUFF)
		tip = arrow.tip
		curve.add(tip)

		curve.set_color(YELLOW)

		path.pop(0)
		path.pop()
		path = path_left + path + path_right

		# dots = VGroup(*[Dot().move_to(point) for point in path])
		# self.play(
		# 	FadeIn(dots)
		# )

		return curve, path


	def make_eval_component(self, poly_text, point_text):
		evaluate = TextMobject("FFT")

		poly_input = TexMobject(poly_text).scale(0.7)
		point_input = TexMobject(point_text).scale(0.7)

		box_inputs = VGroup(poly_input, point_input).arrange_submobjects(DOWN)

		evaluate.next_to(box_inputs, LEFT)

		inside = VGroup(evaluate, box_inputs)

		box = SurroundingRectangle(inside, buff=SMALL_BUFF)

		component = VGroup(box, inside)

		return component

	def get_code(self, abbrev=False):
		code_scale = 0.8
		
		code = []

		def_statement = TextMobject(r"$\text{def FFT}(P):$")
		def_statement[0][:3].set_color(MONOKAI_BLUE)
		def_statement[0][3:6].set_color(MONOKAI_GREEN)
		def_statement[0][7].set_color(MONOKAI_ORANGE)
		def_statement.scale(code_scale)
		def_statement.to_edge(LEFT)
		code.append(def_statement)

		if abbrev:
			comment = TextMobject(r"\# $P$ - $[p_0, p_1, \ldots, p_{n - 1}]$ coeff rep")
		else:
			comment = TextMobject(r"\# $P$ - $[p_0, p_1, \ldots, p_{n - 1}]$ coeff representation")
		
		comment.scale(code_scale)
		comment.next_to(def_statement, DOWN * 0.5)
		comment.to_edge(LEFT * 2)
		comment.set_color(MONOKAI_GRAY)
		code.append(comment)

		line_1 = TextMobject(r"$n = $ len($P$) \# $n$ is a power of 2")
		line_1.scale(code_scale)
		line_1.next_to(comment, DOWN * 0.5)
		line_1.to_edge(LEFT * 2)
		line_1[0][1].set_color(MONOKAI_PINK)
		line_1[0][2:5].set_color(MONOKAI_BLUE)
		line_1[0][8:].set_color(MONOKAI_GRAY)
		code.append(line_1)

		line_2 = TextMobject(r"if $n == 1$:")
		line_2.scale(code_scale)
		line_2.next_to(line_1, DOWN * 0.5)
		line_2.to_edge(LEFT * 2)
		line_2[0][:2].set_color(MONOKAI_PINK)
		line_2[0][3:5].set_color(MONOKAI_PINK)
		line_2[0][-2].set_color(MONOKAI_PURPLE)
		code.append(line_2)

		line_3 = TextMobject(r"return $P$")
		line_3.scale(code_scale)
		line_3.next_to(line_2, DOWN * 0.5)
		line_3.to_edge(LEFT * 3)
		line_3[0][:6].set_color(MONOKAI_PINK)
		code.append(line_3)

		line_4 = TextMobject(r"$\omega = e^{\frac{2 \pi i}{n}}$")
		line_4.scale(code_scale)
		line_4.next_to(line_3, DOWN * 0.5)
		line_4.to_edge(LEFT * 2)
		line_4[0][1].set_color(MONOKAI_PINK)
		code.append(line_4)

		if abbrev:
			line_5 = TextMobject(r"$P_e, P_o = P[::2], P[1::2]$")
			line_5[0][9].shift(RIGHT * SMALL_BUFF)
			line_5[0][16].shift(LEFT * SMALL_BUFF)
			line_5[0][18].shift(LEFT * SMALL_BUFF)
			line_5[0][19].shift(LEFT * SMALL_BUFF)
			line_5[0][10].set_color(MONOKAI_PURPLE)
			line_5[0][15].set_color(MONOKAI_PURPLE)
			line_5[0][18].set_color(MONOKAI_PURPLE)
		else:
			line_5 = TextMobject(r"$P_e, P_o = [p_0, p_2, \ldots, p_{n - 2}], [p_1, p_3, \ldots, p_{n - 1}]$")
			
		line_5[0][5].set_color(MONOKAI_PINK)
		line_5.scale(code_scale)
		line_5.next_to(line_4, DOWN * 0.7)
		line_5.to_edge(LEFT * 2)
		code.append(line_5)

		line_6 = TextMobject(r"$y_e, y_o =$ FFT($P_e$), FFT($P_o$)")
		line_6[0][5].set_color(MONOKAI_PINK)
		line_6[0][6:9].set_color(MONOKAI_BLUE)
		line_6[0][14:17].set_color(MONOKAI_BLUE)
		line_6.scale(code_scale)
		line_6.next_to(line_5, DOWN * 0.5)
		line_6.to_edge(LEFT * 2)
		code.append(line_6)

		line_7 = TextMobject(r"$y = [0]$ * $n$")
		line_7[0][1].set_color(MONOKAI_PINK)
		line_7[0][3].set_color(MONOKAI_PURPLE)
		line_7[0][5].set_color(MONOKAI_PINK)
		line_7[0][5].shift(DOWN * SMALL_BUFF)
		line_7.scale(code_scale)
		line_7.next_to(line_6, DOWN * 0.5)
		line_7.to_edge(LEFT * 2)
		code.append(line_7)

		line_8 = TextMobject(r"for $j$ in range($n / 2$):")
		line_8[0][:3].set_color(MONOKAI_PINK)
		line_8[0][4:6].set_color(MONOKAI_PINK)
		line_8[0][6:11].set_color(MONOKAI_BLUE)
		line_8[0][13].set_color(MONOKAI_PINK)
		line_8[0][14].set_color(MONOKAI_PURPLE)
		line_8.scale(code_scale)
		line_8.next_to(line_7, DOWN * 0.5)
		line_8.to_edge(LEFT * 2)
		code.append(line_8)

		line_9 = TextMobject(r"$y[j] = y_e[j] + \omega ^ j y_o[j]$")
		line_9[0][4].set_color(MONOKAI_PINK)
		line_9[0][10].set_color(MONOKAI_PINK)
		line_9.scale(code_scale)
		line_9.next_to(line_8, DOWN * 0.5)
		line_9.to_edge(LEFT * 3)
		code.append(line_9)

		line_10 = TextMobject(r"$y[j + n / 2] = y_e[j] - \omega ^ j y_o[j]$")
		line_10[0][3].set_color(MONOKAI_PINK)
		line_10[0][5].set_color(MONOKAI_PINK)
		line_10[0][6].set_color(MONOKAI_PURPLE)
		line_10[0][8].set_color(MONOKAI_PINK)
		line_10[0][14].set_color(MONOKAI_PINK)
		line_10.scale(code_scale)
		line_10.next_to(line_9, DOWN * 0.5)
		line_10.to_edge(LEFT * 3)
		code.append(line_10)

		line_11 = TextMobject(r"return $y$")
		line_11.scale(code_scale)
		line_11.next_to(line_10, DOWN * 0.5)
		line_11.to_edge(LEFT * 2)
		line_11[0][:6].set_color(MONOKAI_PINK)
		code.append(line_11)

		code = VGroup(*code)
		code.scale(0.9)
		code.move_to(LEFT * 3)
		return code

class FFTImplementationExample(FFTImplementationP1):
	def construct(self):
		self.walkthrough()

	def walkthrough(self):
		# TODO add code arrows and labels to circle roots of unity

		code = self.get_code()
		code.scale(0.8)
		code.move_to(RIGHT * 4 + UP * 1.5)
		self.add(code)

		first_call = TextMobject(r"FFT($[5, 3, 2, 1])$")
		first_call.scale(0.8)
		first_call.move_to(UP * 3.5).to_edge(LEFT)
		first_call[0][:3].set_color(BLUE)
		first_call[0][4:-1].set_color(YELLOW)

		left_call = TextMobject(r"FFT($[5, 2])$")
		left_call.scale(0.8)
		left_call.move_to(ORIGIN + LEFT * 1)
		left_call[0][:3].set_color(BLUE)
		left_call[0][4:-1].set_color(ORANGE)

		left_l_call = TextMobject(r"FFT($[5]$)", r"$\rightarrow [5]$")
		left_l_call.scale(0.8)
		left_l_call.move_to(DOWN * 3.5 + LEFT * 4.5)
		left_l_call[0][:3].set_color(BLUE)
		left_l_call[0][4:7].set_color(GREEN_SCREEN)
		left_l_call[1][-3:].set_color(GREEN_SCREEN)
		
		left_r_call = TextMobject(r"FFT($[2]$)", r"$\rightarrow [2]$")
		left_r_call.scale(0.8)
		left_r_call.move_to(DOWN * 3.5 + LEFT * 1)
		left_r_call[0][:3].set_color(BLUE)
		left_r_call[0][4:7].set_color(GREEN_SCREEN)
		left_r_call[1][-3:].set_color(GREEN_SCREEN)

		right_call = TextMobject(r"FFT($[3, 1])$")
		right_call.scale(0.8)
		right_call.move_to(ORIGIN + RIGHT * 4)
		right_call[0][:3].set_color(BLUE)
		right_call[0][4:-1].set_color(ORANGE)

		right_l_call = TextMobject(r"FFT($[3]$)", r"$\rightarrow [3]$")
		right_l_call.scale(0.8)
		right_l_call.move_to(DOWN * 3.5 + RIGHT * 2)
		right_l_call[0][:3].set_color(BLUE)
		right_l_call[0][4:7].set_color(GREEN_SCREEN)
		right_l_call[1][-3:].set_color(GREEN_SCREEN)

		right_r_call = TextMobject(r"FFT($[1]$)", r"$\rightarrow [1]$")
		right_r_call.scale(0.8)
		right_r_call.move_to(DOWN * 3.5 + RIGHT * 5)
		right_r_call[0][:3].set_color(BLUE)
		right_r_call[0][4:7].set_color(GREEN_SCREEN)
		right_r_call[1][-3:].set_color(GREEN_SCREEN)


		second_row = VGroup(
			left_call, right_call
		).arrange_submobjects(RIGHT, buff=2).move_to(LEFT * FRAME_X_RADIUS / 2).to_edge(LEFT)

		bottom_row = VGroup(
			left_l_call, left_r_call, right_l_call, right_r_call
		).arrange_submobjects(RIGHT, buff=1).move_to(DOWN * 3.5).to_edge(LEFT)

		self.play(
			Write(first_call)
		)

		space = 0.7
		scale = 0.7

		code_arrow = Arrow(ORIGIN, RIGHT).scale(0.7)
		code_arrow.set_color(GREEN_SCREEN)
		code_arrow.next_to(code[1], LEFT * 0.5)
		self.play(
			ShowCreation(code_arrow)
		)

		P_x = TexMobject("P(x) = 5 + 3x + 2x^2 + x^3").scale(scale)
		P_x.next_to(first_call, DOWN * space).to_edge(LEFT)
		self.play(
			Write(P_x)
		)

		self.wait(2)

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 2)

		n_4 = TexMobject("n = 4").scale(scale)
		n_4.next_to(P_x, DOWN * space).to_edge(LEFT)
		self.play(
			Write(n_4)
		)

		self.wait()

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 3)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 5)

		omega_4 = TexMobject(r"\omega = e^{\frac{2 \pi i}{4}}", " = i").scale(scale)
		omega_4.next_to(n_4, DOWN * space).to_edge(LEFT)
		self.play(
			Write(omega_4[0])
		)

		self.wait(4)

		self.play(
			Write(omega_4[1])
		)

		self.wait(5)

		initalizations = VGroup(P_x, n_4, omega_4)

		w_i = TexMobject(r"[\omega^0, \omega^1, \omega^2, \omega^3]").scale(scale)
		w_i.next_to(initalizations, RIGHT)
		self.play(
			Write(w_i)
		)

		self.wait()

		w_i_values = TexMobject("[1, i, -1, -i]").scale(scale)
		w_i_values.move_to(w_i.get_center())
		self.play(
			ReplacementTransform(w_i, w_i_values),
			run_time=1
		)

		self.wait()

		circle = Circle(color=BLUE).move_to(w_i_values.get_center()).scale(0.8).shift(UP * SMALL_BUFF * 2.5)
		points = VGroup(*[Dot().move_to(circle.point_from_proportion(i / 4)) for i in range(4)]).set_color(YELLOW)
		circle.add(points)

		labels = [TexMobject(s).scale(0.5) for s in ['1', 'i', '-1', '-i']]
		for i, direction in enumerate([RIGHT, UP, LEFT, DOWN]):
			labels[i].next_to(points[i], direction, buff=SMALL_BUFF)
		labels = VGroup(*labels)

		circle.add(labels)

		self.play(
			ReplacementTransform(w_i_values, circle),
			run_time=2
		)

		self.wait(4)

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 6)

		P_e = TexMobject("P_e(x^2) = 5 + 2x", r"\rightarrow [5, 2]").scale(scale)
		P_e[1][-5:].set_color(ORANGE)
		P_e.next_to(omega_4, DOWN * space).to_edge(LEFT)
		self.play(
			Write(P_e[0])
		)
		self.play(
			Write(P_e[1])
		)

		self.wait(5)

		P_o = TexMobject("P_o(x^2) = 3 + x", r"\rightarrow [3, 1]").scale(scale)
		P_o[1][-5:].set_color(ORANGE)
		P_o.next_to(P_e, DOWN * space).to_edge(LEFT)
		self.play(
			Write(P_o[0])
		)

		self.play(
			Write(P_o[1])
		)

		self.wait(4)

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 7)

		self.play(
			Write(left_call)
		)

		P_x_left = TexMobject("P(x) = 5 + 2x").scale(scale)
		P_x_left.next_to(left_call, DOWN * space).to_edge(LEFT)

		self.play(
			Write(right_call)
		)

		P_x_right = TexMobject("P(x) = 3 + x").scale(scale)
		P_x_right.next_to(right_call, DOWN * space).to_edge(LEFT * 8.7)

		self.play(
			ReplacementTransform(P_e, P_x_left),
			ReplacementTransform(P_o, P_x_right),
			run_time=2
		)

		self.wait(5)

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 2)

		
		n_2_left = TexMobject("n = 2").scale(0.7)
		n_2_left.next_to(P_x_left, DOWN * space).to_edge(LEFT)
		self.play(
			Write(n_2_left)
		)

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 3)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 5)

		omega_2_left = TexMobject(r"\omega = e^{\frac{2 \pi i}{2}}", "= -1").scale(scale)
		omega_2_left.next_to(n_2_left, DOWN * space).to_edge(LEFT)
		self.play(
			Write(omega_2_left[0])
		)
		self.wait(2)

		self.play(
			Write(omega_2_left[1])
		)
		self.wait(3)

		w_i_left = TexMobject(r"[\omega^0, \omega^1]").scale(scale)
		initalizations_left = VGroup(P_x_left, n_2_left, omega_2_left)
		w_i_left.next_to(initalizations_left, RIGHT)
		self.play(
			Write(w_i_left)
		)
		self.wait()

		w_i_values_left = TexMobject(r"[1, -1]").scale(scale)
		w_i_values_left.move_to(w_i_left.get_center())
		self.play(
			ReplacementTransform(w_i_left, w_i_values_left)
		)

		circle_left = Circle(color=BLUE).move_to(w_i_values_left.get_center()).scale(0.5)
		points_left = VGroup(*[Dot().move_to(circle_left.point_from_proportion(i / 2)) for i in range(2)]).set_color(ORANGE)
		circle_left.add(points_left)

		labels = [TexMobject(s).scale(0.5) for s in ['1', '-1']]
		for i, direction in enumerate([RIGHT, LEFT]):
			labels[i].next_to(points_left[i], direction, buff=SMALL_BUFF)
		labels = VGroup(*labels)

		circle_left.add(labels)

		self.play(
			ReplacementTransform(w_i_values_left, circle_left),
			run_time=2
		)

		self.wait(3)

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 6)

		P_e_left = TexMobject("P_e(x^2) = 5" , r"\rightarrow [5]").scale(scale)
		P_e_left.next_to(omega_2_left, DOWN * space).to_edge(LEFT)
		P_e_left[1][-3:].set_color(GREEN_SCREEN)

		P_o_right = TexMobject("P_o(x^2) = 2", r"\rightarrow [2]").scale(scale)
		P_o_right.next_to(P_e_left, DOWN * space).to_edge(LEFT)
		P_o_right[1][-3:].set_color(GREEN_SCREEN)

		self.play(
			Write(P_e_left[0]),
			Write(P_o_right[0])
		)

		self.play(
			Write(P_e_left[1]),
			Write(P_o_right[1])
		)

		self.wait()

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 7)

		self.play(
			Write(left_l_call[0])
		)
		self.play(
			Write(left_r_call[0])
		)

		self.wait()

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 2)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 3)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 4)

		self.play(
			Write(left_l_call[1])
		)

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 2)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 3)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 4)

		self.play(
			Write(left_r_call[1])
		)

		self.wait(3)

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 7)

		y_e_l = TexMobject("y_e = [5]").scale(scale)
		y_e_l.move_to(P_e_left.get_center()).to_edge(LEFT)
		y_e_l[0][-3:].set_color(GREEN_SCREEN)

		y_o_l = TexMobject("y_o = [2]").scale(scale)
		y_o_l.move_to(P_o_right.get_center()).to_edge(LEFT)
		y_o_l[0][-3:].set_color(GREEN_SCREEN)


		self.play(
			ReplacementTransform(P_e_left, y_e_l),
			ReplacementTransform(P_o_right, y_o_l),
			run_time=2
		)

		self.wait(4)

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 8)

		y_left = TexMobject("y = [0, 0]").scale(scale)
		y_left.next_to(circle_left, DOWN).shift(DOWN * SMALL_BUFF)
		self.play(
			Write(y_left)
		)

		self.wait()

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 9)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 10)

		self.wait()


		y_0 = TexMobject(r"y[0] = y_e[0] + \omega ^ 0 y_o[0]").scale(scale)
		y_0.next_to(y_left, DOWN * space)

		self.play(
			Write(y_0),
		)
		
		self.wait(10)

		self.play(
			Flash(points_left[0], color=GREEN_SCREEN)
		)

		self.play(
			points_left[0].set_color, GREEN_SCREEN
		)

		self.wait(4)

		y_0_value = TexMobject(r"y[0] = 7").scale(scale)
		y_0_value.next_to(y_left, DOWN * space)

		self.play(
			Transform(y_0, y_0_value)
		)

		self.wait()

		y_left_tranf = TexMobject("y = [7, 0]").scale(scale).move_to(y_left.get_center())

		self.play(
			Transform(y_left, y_left_tranf)
		)

		self.wait()

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 11)
		self.wait(2)

		y_1 = TexMobject(r"y[1] = y_e[0] - \omega ^ 0 y_o[0]").scale(scale)
		y_1.next_to(y_left, DOWN * space)

		self.play(
			Transform(y_0, y_1),
			points_left[0].set_color, ORANGE,
		)

		self.wait(12)

		self.play(
			Flash(points_left[1], color=GREEN_SCREEN)
		)

		self.play(
			points_left[1].set_color, GREEN_SCREEN
		)

		self.wait(3)

		y_1_value = TexMobject(r"y[1] = 3").scale(scale)
		y_1_value.next_to(y_left, DOWN * space)

		self.play(
			Transform(y_0, y_1_value),
		)

		self.wait()

		y_left_tranf = TexMobject("y = [7, 3]").scale(scale).move_to(y_left.get_center())
		self.play(
			ReplacementTransform(y_left, y_left_tranf)
		)

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 9)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 12)

		left_call_result = TexMobject(r"\rightarrow", "[7, 3]").scale(0.8)
		left_call_result.next_to(left_call, RIGHT, buff=SMALL_BUFF)
		left_call_result[1].set_color(ORANGE)
		self.play(
			Write(left_call_result[0]),
			TransformFromCopy(y_left_tranf[0][2:], left_call_result[1]),
			FadeOut(y_0),
			FadeOut(y_left_tranf),
			points_left[1].set_color, ORANGE,
			run_time=2
		)

		self.wait(6)

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 2)

		n_2_right = n_2_left.copy().to_edge(LEFT * 8.7)
		omega_2_right = omega_2_left.copy().to_edge(LEFT * 8.7)

		initalizations_right = VGroup(P_x_right, n_2_right, omega_2_right)

		circle_right = circle_left.copy()
		circle_right.next_to(initalizations_right, RIGHT)
		self.play(
			TransformFromCopy(n_2_left, n_2_right),
			TransformFromCopy(omega_2_left, omega_2_right),
			TransformFromCopy(circle_left, circle_right),
			code_arrow.next_to, code[5], LEFT * 0.5,
			run_time=2
		)

		self.wait(7)

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 6)
		self.wait()

		P_e_left = TexMobject("P_e(x^2) = 3" , r"\rightarrow [3]").scale(scale)
		P_e_left.next_to(omega_2_right, DOWN * space).to_edge(LEFT * 8.7)
		P_e_left[1][-3:].set_color(GREEN_SCREEN)

		P_o_right = TexMobject("P_o(x^2) = 1", r"\rightarrow [1]").scale(scale)
		P_o_right.next_to(P_e_left, DOWN * space).to_edge(LEFT * 8.7)
		P_o_right[1][-3:].set_color(GREEN_SCREEN)

		self.play(
			Write(P_e_left[0]),
			Write(P_o_right[0])
		)

		self.wait()

		self.play(
			Write(P_e_left[1]),
			Write(P_o_right[1])
		)

		self.wait()

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 7)

		self.play(
			Write(right_l_call[0]),
			Write(right_r_call[0])
		)

		self.wait()

		self.play(
			Write(right_l_call[1]),
			Write(right_r_call[1])
		)

		self.wait(2)

		y_e_r = TexMobject("y_e = [3]").scale(scale)
		y_e_r.move_to(P_e_left.get_center()).to_edge(LEFT * 8.7)
		y_e_r[0][-3:].set_color(GREEN_SCREEN)

		y_o_r = TexMobject("y_o = [1]").scale(scale)
		y_o_r.move_to(P_o_right.get_center()).to_edge(LEFT * 8.7)
		y_o_r[0][-3:].set_color(GREEN_SCREEN)

		

		self.play(
			ReplacementTransform(P_e_left, y_e_r),
			ReplacementTransform(P_o_right, y_o_r)
		)

		self.wait()

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 8)

		### y_value FFT([3, 1]) call

		y_right = TexMobject("y = [0, 0]").scale(scale)
		y_right.next_to(circle_right, DOWN).shift(DOWN * SMALL_BUFF)
		self.play(
			Write(y_right)
		)

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 9)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 10)

		y_0 = TexMobject(r"y[0] = y_e[0] + \omega ^ 0 y_o[0]").scale(scale)
		y_0.next_to(y_right, DOWN * space)

		self.play(
			Write(y_0),
			circle_right[1][0].set_color, GREEN_SCREEN
		)
		
		self.wait(5)

		y_0_value = TexMobject(r"y[0] = 4").scale(scale)
		y_0_value.next_to(y_right, DOWN * space)

		self.play(
			Transform(y_0, y_0_value)
		)

		y_right_tranf = TexMobject("y = [4, 0]").scale(scale).move_to(y_right.get_center())

		self.play(
			Transform(y_right, y_right_tranf)
		)

		self.wait(2)

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 11)

		y_1 = TexMobject(r"y[1] = y_e[0] - \omega ^ 0 y_o[0]").scale(scale)
		y_1.next_to(y_right, DOWN * space)

		self.play(
			Transform(y_0, y_1),
			circle_right[1][0].set_color, ORANGE,
			circle_right[1][1].set_color, GREEN_SCREEN
		)

		self.wait(5)

		y_1_value = TexMobject(r"y[1] = 2").scale(scale)
		y_1_value.next_to(y_right, DOWN * space)

		self.play(
			Transform(y_0, y_1_value),
		)

		self.wait()

		y_right_tranf = TexMobject("y = [4, 2]").scale(scale).move_to(y_right.get_center())
		self.play(
			ReplacementTransform(y_right, y_right_tranf)
		)

		self.wait(3)

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 9)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 12)

		right_call_result = TexMobject(r"\rightarrow", "[4, 2]").scale(0.8)
		right_call_result.next_to(right_call, RIGHT, buff=SMALL_BUFF)
		right_call_result[1].set_color(ORANGE)
		self.play(
			Write(right_call_result[0]),
			TransformFromCopy(y_right_tranf[0][2:], right_call_result[1]),
			FadeOut(y_0),
			FadeOut(y_right_tranf),
			circle_right[0][1].set_color, ORANGE,
			code_arrow.next_to, code[7], LEFT * 0.5,
			run_time=2
		)

		self.wait(3)

		y_e = TexMobject("y_e = [7, 3]").scale(scale)
		y_e.next_to(omega_4, DOWN * space).to_edge(LEFT)
		y_e[0][3:].set_color(ORANGE)

		y_o = TexMobject("y_o = [4, 2]").scale(scale)
		y_o.next_to(y_e, DOWN * space).to_edge(LEFT)
		y_o[0][3:].set_color(ORANGE)

		self.play(
			Write(y_e),
			Write(y_o)
		)

		self.wait(5)

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 8)

		y = TexMobject("y = [0, 0, 0, 0]").scale(scale)
		y.next_to(circle, DOWN).shift(DOWN * SMALL_BUFF)
		self.play(
			Write(y)
		)

		self.wait(4)

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 9)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 10)

		y_0 = TexMobject(r"y[0] = y_e[0] + \omega ^ 0 y_o[0]").scale(scale)
		y_0.next_to(y, DOWN * space)

		self.play(
			Write(y_0),
		)
		
		self.wait(10)

		self.play(
			Flash(circle[1][0], color=GREEN_SCREEN)
		)

		self.play(
			circle[1][0].set_color, GREEN_SCREEN
		)

		self.wait()

		y_0_value = TexMobject(r"y[0] = 11").scale(scale)
		y_0_value.next_to(y, DOWN * space)

		self.play(
			Transform(y_0, y_0_value)
		)

		self.wait()

		y_tranf = TexMobject("y = [11, 0, 0, 0]").scale(scale).move_to(y.get_center())

		self.play(
			Transform(y, y_tranf)
		)

		self.wait()

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 11)
		self.wait(2)

		y_2 = TexMobject(r"y[2] = y_e[0] - \omega ^ 0 y_o[0]").scale(scale)
		y_2.next_to(y, DOWN * space)

		self.play(
			ReplacementTransform(y_0, y_2),
			circle[1][0].set_color, YELLOW,
		)

		self.wait(12)

		self.play(
			Flash(circle[1][2], color=GREEN_SCREEN)
		)

		self.play(
			circle[1][2].set_color, GREEN_SCREEN
		)

		self.wait(6)

		y_2_value = TexMobject(r"y[2] = 3").scale(scale)
		y_2_value.next_to(y, DOWN * space)

		self.play(
			Transform(y_2, y_2_value),
		)

		self.wait()

		y_tranf = TexMobject("y = [11, 0, 3, 0]").scale(scale).move_to(y.get_center())
		self.play(
			Transform(y, y_tranf)
		)

		self.wait(2)

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 9)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 10)

		self.wait()
		
		y_1 = TexMobject(r"y[1] = y_e[1] + \omega ^ 1 y_o[1]").scale(scale)
		y_1.next_to(y, DOWN * space)

		self.play(
			ReplacementTransform(y_2, y_1),
			circle[1][2].set_color, YELLOW
		)

		self.wait(10)

		self.play(
			Flash(circle[1][1], color=GREEN_SCREEN)
		)

		self.play(
			circle[1][1].set_color, GREEN_SCREEN
		)

		self.wait(3)

		y_1_value = TexMobject(r"y[1] = 3 + 2i").scale(scale)
		y_1_value.next_to(y, DOWN * space)

		self.play(
			Transform(y_1, y_1_value),
		)

		self.wait(2)

		y_tranf = TexMobject("y = [11, 3 + 2i, 3, 0]").scale(scale).move_to(y.get_center())
		self.play(
			Transform(y, y_tranf)
		)

		self.wait(4)

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 11)
		self.wait(2)

		y_3 = TexMobject(r"y[3] = y_e[1] - \omega ^ 1 y_o[1]").scale(scale)
		y_3.next_to(y, DOWN * space)

		self.play(
			ReplacementTransform(y_1, y_3),
			circle[1][1].set_color, YELLOW,
		)

		self.wait(8)

		self.play(
			Flash(circle[1][3], color=GREEN_SCREEN)
		)

		self.play(
			circle[1][3].set_color, GREEN_SCREEN
		)

		self.wait(3)

		y_3_value = TexMobject(r"y[3] = 3 - 2i").scale(scale)
		y_3_value.next_to(y, DOWN * space)

		self.play(
			Transform(y_3, y_3_value),
		)

		self.wait(2)

		y_tranf = TexMobject("y = [11, 3 + 2i, 3, 3 - 2i]").scale(scale).move_to(y.get_center())
		self.play(
			Transform(y, y_tranf)
		)

		self.wait(2)

		self.play(
			circle[1][3].set_color, YELLOW
		)

		self.wait()

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 12)

		first_call_result = TexMobject(r"\rightarrow", "[11, 3 + 2i, 3, 3 - 2i]").scale(0.8)
		first_call_result.next_to(first_call, RIGHT, buff=SMALL_BUFF)
		first_call_result[1].set_color(YELLOW)

		self.play(
			Write(first_call_result[0]),
			TransformFromCopy(y[0][3:], first_call_result[1]),
			circle.shift, DOWN * 0.6,
			FadeOut(y),
			FadeOut(y_3),
			FadeOut(code_arrow),
			run_time=2
		)

		self.wait(6)

		

		self.play(
			FadeOut(code)
		)

		self.wait()

		scale = 0.9
		indent = LEFT * 17.5
		p_0 = TexMobject(r"P(\omega ^ 0) = P(1) = 11").scale(scale).to_edge(indent)
		p_1 = TexMobject(r"P(\omega ^ 1) = P(i) = 3 + 2i").scale(scale)
		p_1.next_to(p_0, DOWN).to_edge(indent)
		p_2 = TexMobject(r"P(\omega ^ 2) = P(-1) = 3").scale(scale)
		p_2.next_to(p_1, DOWN).to_edge(indent)
		p_3 = TexMobject(r"P(\omega ^ 3) = P(-i) = 3 - 2i").scale(scale)
		p_3.next_to(p_2, DOWN).to_edge(indent)
		
		sanity_check = VGroup(p_0, p_1, p_2, p_3).shift(UP * 3)

		self.play(
			Write(p_0),
		)

		self.wait()

		self.play(
			Write(p_1),
		)

		self.wait()

		self.play(
			Write(p_2),
		)

		self.wait()

		self.play(
			Write(p_3),
		)

		self.wait(6)



	def shift_arrow_to_line(self, arrow, code, line_number, run_time=1):
		arrow_copy = arrow.copy()
		arrow_copy.next_to(code[line_number], LEFT * 0.5)

		self.play(
			Transform(arrow, arrow_copy),
			run_time=run_time
		)
		return arrow

class Interpolation(Scene):
	def construct(self):
		title = TextMobject("Interpolation").scale(1.2).move_to(UP * 3.5)
		h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
		h_line.next_to(title, DOWN)
		self.add(title)
		self.add(h_line)

		self.show_interpolation_insight(title, h_line)

	def show_interpolation_insight(self, title, h_line):
		alternative = TextMobject("Alternative Perspective on Evaluation/FFT")
		alternative.scale(0.9).next_to(h_line, DOWN)
		scale = 0.8
		self.wait(7)

		polynomial = TexMobject(r"P(x) = p_0 + p_1x + p_2x^2 + \cdots + p_{n - 1}x^{n - 1}")
		polynomial.scale(scale)
		polynomial.next_to(alternative, DOWN * 1.5)
		self.play(
			Write(polynomial),
			Write(alternative),
			run_time=2
		)

		self.wait(2)

		x_0poly = TexMobject(r"P(x_0) = p_0 + p_1x_0 + p_2x_0^2 + \cdots + p_{n - 1}x_0^{n - 1}").scale(scale)
		x_1poly = TexMobject(r"P(x_1) = p_0 + p_1x_1 + p_2x_1^2 + \cdots + p_{n - 1}x_1^{n - 1}").scale(scale)
		x_2poly = TexMobject(r"P(x_2) = p_0 + p_1x_2 + p_2x_2^2 + \cdots + p_{n - 1}x_2^{n - 1}").scale(scale)
		vdots = TexMobject(r"\vdots").scale(scale)
		x_dpoly = TexMobject(r"P(x_{n - 1}) = p_0 + p_1x_{n - 1} + p_2x_{n - 1}^2 + \cdots + p_{n - 1}x_{n - 1}^{n - 1}").scale(scale)

		x_0poly.next_to(polynomial, DOWN * 2)
		x_1poly.next_to(x_0poly, DOWN)
		x_2poly.next_to(x_1poly, DOWN)
		vdots.next_to(x_2poly, DOWN)
		x_dpoly.next_to(vdots, DOWN)

		self.play(
			FadeIn(vdots),
			Write(x_dpoly),
			Write(x_0poly),
			Write(x_1poly),
			Write(x_2poly),
			run_time=2
		)

		self.wait(5)

		system_of_equations = VGroup(x_0poly, x_1poly, x_2poly, vdots, x_dpoly)

		matrix = TexMobject(r"""
		\begin{bmatrix}
		1 & x_0 & x_0^2 & \cdots & x_0^{n - 1} \\
		1 & x_1 & x_1^2 & \cdots & x_1^{n - 1} \\
		1 & x_2 & x_2^2 & \cdots & x_2^{n - 1} \\
		\vdots & \vdots & \vdots  & \ddots & \vdots  \\
		1 & x_{n - 1} & x_{n - 1}^2 & \cdots & x_{n - 1}^{n - 1}
		\end{bmatrix}
		""")

		coeff_vector = TexMobject(r"""
		\begin{bmatrix}
		p_0 \\
		p_1 \\
		p_2 \\
		\vdots \\
		p_{n - 1}
		\end{bmatrix}
		""")

		value_vector = TexMobject(r"""
		\begin{bmatrix}
		P(x_0) \\
		P(x_1) \\
		P(x_2) \\
		\vdots \\
		P(x_{n - 1})
		\end{bmatrix}
		""")

		equals = TexMobject("=")

		coeff_vector.next_to(matrix, RIGHT)
		equals.next_to(matrix, LEFT)
		value_vector.next_to(equals, LEFT)

		matrix_vector_prod = VGroup(value_vector, equals, matrix, coeff_vector)

		matrix_vector_prod.move_to(system_of_equations.get_center())


		self.play(
			ReplacementTransform(system_of_equations, matrix_vector_prod),
			run_time=2
		)

		self.wait(8)

		definitions = TexMobject(r"x_k = \omega ^ k \text{ where } \omega = e^{\frac{2 \pi i}{n} ")
		definitions.scale(scale).next_to(matrix_vector_prod, DOWN)
		self.play(
			Write(definitions)
		)

		self.wait(6)

		dft_matrix = TexMobject(r"""
		\begin{bmatrix}
		1 & 1 & 1 & \cdots & 1 \\
		1 & \omega & \omega^2 & \cdots & \omega^{n - 1} \\
		1 & \omega^2 & \omega^4 & \cdots & \omega^{2(n - 1)} \\
		\vdots & \vdots & \vdots  & \ddots & \vdots  \\
		1 & \omega^{n - 1} & \omega^{2(n - 1)} & \cdots & \omega^{(n - 1)(n - 1)}
		\end{bmatrix}
		""")

		dft_coeff_vector = TexMobject(r"""
		\begin{bmatrix}
		p_0 \\
		p_1 \\
		p_2 \\
		\vdots \\
		p_{n - 1}
		\end{bmatrix}
		""")

		dft_value_vector = TexMobject(r"""
		\begin{bmatrix}
		P(\omega^0) \\
		P(\omega^1) \\
		P(\omega^2) \\
		\vdots \\
		P(\omega^{n - 1})
		\end{bmatrix}
		""")

		dft_equals = TexMobject("=")

		dft_coeff_vector.next_to(dft_matrix, RIGHT)
		dft_equals.next_to(dft_matrix, LEFT)
		dft_value_vector.next_to(dft_equals, LEFT)

		dft_matrix_vector_prod = VGroup(dft_value_vector, dft_equals, dft_matrix, dft_coeff_vector)
		dft_matrix_vector_prod.scale(0.9).move_to(matrix_vector_prod.get_center())

		self.play(
			ReplacementTransform(matrix_vector_prod, dft_matrix_vector_prod),
			run_time=2
		)

		self.wait(2)

		brace = Brace(dft_matrix, direction=DOWN)
		brace_label = TextMobject(r"Discrete Fourier Transform (DFT) matrix").scale(0.7).next_to(brace, DOWN, buff=SMALL_BUFF)

		self.play(
			GrowFromCenter(brace),
			Write(brace_label),
			definitions.shift, DOWN * 0.6
		)

		self.wait(28)

		interpolation_exp = TextMobject("Interpolation involves inversing the DFT matrix").scale(0.9)
		interpolation_exp.move_to(alternative.get_center())
		self.play(
			ReplacementTransform(alternative, interpolation_exp)
		)

		self.wait()

		inverse = TexMobject("-1").scale(0.6).next_to(dft_matrix, UP+RIGHT, buff=0)
		SHIFT_AMOUT = LEFT * 0.3
		inverse.shift(LEFT * SMALL_BUFF * 2 + SHIFT_AMOUT)
		
		self.play(
			dft_coeff_vector.next_to, equals, LEFT * 2 + SHIFT_AMOUT * 2,
			dft_value_vector.next_to, dft_matrix, RIGHT, 0,
			dft_equals.shift, LEFT * SMALL_BUFF * 2 + SHIFT_AMOUT,
			dft_matrix.shift, LEFT * SMALL_BUFF * 2 + SHIFT_AMOUT,
			FadeIn(inverse),
			FadeOut(brace),
			FadeOut(brace_label),
			run_time=2
		)

		dft_matrix_vector_prod.add(inverse)

		self.wait(17)

		self.play(
			dft_matrix_vector_prod.scale, 0.8,
			dft_matrix_vector_prod.move_to, UP * 2,
			definitions.move_to, UP * 3.5,
			FadeOut(interpolation_exp),
			FadeOut(polynomial),
			FadeOut(h_line),
			FadeOut(title),
			run_time=3
		)

		self.wait()

		downarrow = TexMobject(r"\Downarrow").next_to(dft_matrix, DOWN)
		self.play(
			ShowCreation(downarrow)
		)

		self.wait()

		dft_inv_matrix = TexMobject(r"""
		\frac{1}{n}
		\begin{bmatrix}
		1 & 1 & 1 & \cdots & 1 \\
		1 & \omega^{-1} & \omega^{-2} & \cdots & \omega^{-(n - 1)} \\
		1 & \omega^{-2} & \omega^{-4} & \cdots & \omega^{-2(n - 1)} \\
		\vdots & \vdots & \vdots  & \ddots & \vdots  \\
		1 & \omega^{-(n - 1)} & \omega^{-2(n - 1)} & \cdots & \omega^{-(n - 1)(n - 1)}
		\end{bmatrix}
		""")

		dft_inv_coeff_vector = TexMobject(r"""
		\begin{bmatrix}
		p_0 \\
		p_1 \\
		p_2 \\
		\vdots \\
		p_{n - 1}
		\end{bmatrix}
		""")

		dft_inv_value_vector = TexMobject(r"""
		\begin{bmatrix}
		P(\omega^0) \\
		P(\omega^1) \\
		P(\omega^2) \\
		\vdots \\
		P(\omega^{n - 1})
		\end{bmatrix}
		""")

		dft_inv_equals = TexMobject("=")

		dft_inv_value_vector.next_to(dft_inv_matrix, RIGHT)
		dft_inv_equals.next_to(dft_inv_matrix, LEFT)
		dft_inv_coeff_vector.next_to(dft_inv_equals, LEFT)

		dft_matrix_inv_vector_prod = VGroup(dft_inv_value_vector, dft_inv_equals, dft_inv_matrix, dft_inv_coeff_vector)

		dft_matrix_inv_vector_prod.scale(0.8 * 0.9).move_to(DOWN * 1.3)

		self.play(
			FadeIn(dft_matrix_inv_vector_prod)
		)

		self.wait(10)

		observation = TextMobject("The inverse matrix and original matrix look quite similar!")
		observation.scale(0.8)
		observation.next_to(dft_matrix_inv_vector_prod, DOWN)

		self.play(
			Write(observation)
		)

		self.wait(5)

		difference = TextMobject(r"Every $\omega$ in original matrix is now $\frac {1}{n} \omega^{-1}$").scale(0.8)
		difference.next_to(observation, DOWN)

		self.play(
			Write(difference)
		)

		self.wait(15)

class ShowInverseFFT(FFTImplementationP1):
	def construct(self):
		self.show_connection()

	def show_connection(self):
		mid_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
		evaluation = TextMobject("Evaluation (FFT)").scale(0.8).move_to(UP * 3.5)
		space = 0.7
		eval_io = TextMobject(r"FFT($[p_0, p_1, \ldots, p_{n - 1}]) \rightarrow [P(\omega^0), P(\omega^1), \ldots , P(\omega^{n - 1})]$").scale(0.7)
		eval_io[0][:3].set_color(BLUE)
		eval_io.next_to(evaluation, DOWN * space)	

		interpolation = TextMobject("Interpolation (Inverse FFT)").scale(0.7).next_to(mid_line, DOWN * space)


		self.play(
			Write(evaluation),
			Write(interpolation),
			ShowCreation(mid_line)
		)

		self.wait(4)

		self.play(
			Write(eval_io),
			run_time=2
		)

		self.wait(6)

		dft_matrix = TexMobject(r"""
		\begin{bmatrix}
		1 & 1 & 1 & \cdots & 1 \\
		1 & \omega & \omega^2 & \cdots & \omega^{n - 1} \\
		1 & \omega^2 & \omega^4 & \cdots & \omega^{2(n - 1)} \\
		\vdots & \vdots & \vdots  & \ddots & \vdots  \\
		1 & \omega^{n - 1} & \omega^{2(n - 1)} & \cdots & \omega^{(n - 1)(n - 1)}
		\end{bmatrix}
		""")

		dft_coeff_vector = TexMobject(r"""
		\begin{bmatrix}
		p_0 \\
		p_1 \\
		p_2 \\
		\vdots \\
		p_{n - 1}
		\end{bmatrix}
		""")

		dft_value_vector = TexMobject(r"""
		\begin{bmatrix}
		P(\omega^0) \\
		P(\omega^1) \\
		P(\omega^2) \\
		\vdots \\
		P(\omega^{n - 1})
		\end{bmatrix}
		""")

		dft_equals = TexMobject("=")

		dft_coeff_vector.next_to(dft_matrix, RIGHT)
		dft_equals.next_to(dft_matrix, LEFT)
		dft_value_vector.next_to(dft_equals, LEFT)

		dft_matrix_vector_prod = VGroup(dft_value_vector, dft_equals, dft_matrix, dft_coeff_vector)
		dft_matrix_vector_prod.scale(0.5).next_to(eval_io, DOWN)

		self.play(
			FadeIn(dft_matrix_vector_prod)
		)

		self.wait()

		omega_def = TextMobject(r"FFT(<coeffs>) defined $\omega = e ^ {\frac{2 \pi i}{n}}$")
		omega_def[0][:3].set_color(BLUE)
		omega_def.scale(0.8).next_to(dft_matrix_vector_prod, DOWN)
		self.play(
			Write(omega_def)
		)

		self.wait(9)

		interp_io = TextMobject(r"IFFT($[P(\omega^0), P(\omega^1), \ldots , P(\omega^{n - 1})]) \rightarrow [p_0, p_1, \ldots, p_{n - 1}]$").scale(0.7)
		interp_io[0][:4].set_color(BLUE)
		interp_io.next_to(interpolation, DOWN * space)	
		self.play(
			Write(interp_io)
		)

		self.wait(11)

		dft_inv_matrix = TexMobject(r"""
		\frac{1}{n}
		\begin{bmatrix}
		1 & 1 & 1 & \cdots & 1 \\
		1 & \omega^{-1} & \omega^{-2} & \cdots & \omega^{-(n - 1)} \\
		1 & \omega^{-2} & \omega^{-4} & \cdots & \omega^{-2(n - 1)} \\
		\vdots & \vdots & \vdots  & \ddots & \vdots  \\
		1 & \omega^{-(n - 1)} & \omega^{-2(n - 1)} & \cdots & \omega^{-(n - 1)(n - 1)}
		\end{bmatrix}
		""")

		dft_inv_coeff_vector = TexMobject(r"""
		\begin{bmatrix}
		p_0 \\
		p_1 \\
		p_2 \\
		\vdots \\
		p_{n - 1}
		\end{bmatrix}
		""")

		dft_inv_value_vector = TexMobject(r"""
		\begin{bmatrix}
		P(\omega^0) \\
		P(\omega^1) \\
		P(\omega^2) \\
		\vdots \\
		P(\omega^{n - 1})
		\end{bmatrix}
		""")

		dft_inv_equals = TexMobject("=")

		dft_inv_value_vector.next_to(dft_inv_matrix, RIGHT)
		dft_inv_equals.next_to(dft_inv_matrix, LEFT)
		dft_inv_coeff_vector.next_to(dft_inv_equals, LEFT)

		dft_matrix_inv_vector_prod = VGroup(dft_inv_value_vector, dft_inv_equals, dft_inv_matrix, dft_inv_coeff_vector)
		dft_matrix_inv_vector_prod.scale(0.5).next_to(interp_io, DOWN * space)

		self.play(
			FadeIn(dft_matrix_inv_vector_prod)
		)
		
		self.wait(5)

		difference = TextMobject(r"Every $\omega$ in DFT matrix is now $\frac {1}{n} \omega^{-1}$").scale(0.7)
		difference.next_to(dft_matrix_inv_vector_prod, DOWN * space)

		self.play(
			Write(difference)
		)

		self.wait(8)

		conclusion = TextMobject(r"IFFT(<values>) $\Leftrightarrow$ FFT(<values>) with $\omega = \frac{1}{n} e ^ {\frac{-2 \pi i}{n}}$").scale(0.8)
		conclusion.next_to(dft_matrix_inv_vector_prod, DOWN)
		conclusion[0][:4].set_color(BLUE)
		conclusion[0][15:18].set_color(BLUE)

		self.play(
			ReplacementTransform(difference, conclusion)
		)

		self.wait(19)

		self.play(
			FadeOut(evaluation),
			FadeOut(interpolation),
			FadeOut(dft_matrix_inv_vector_prod),
			FadeOut(dft_matrix_vector_prod),
			FadeOut(mid_line),
			FadeOut(eval_io),
			FadeOut(interp_io),
			FadeOut(omega_def),
			conclusion.move_to, UP * 3.5,
			run_time=3
		)

		self.wait()

		fft_code = self.get_code(abbrev=True)
		fft_code.to_edge(LEFT)


		self.play(
			FadeIn(fft_code)
		)

		self.wait(2)

		code_scale = 0.72
		ifft_code = []

		def_statement = TextMobject(r"$\text{def IFFT}(P):$")
		def_statement[0][:3].set_color(MONOKAI_BLUE)
		def_statement[0][3:7].set_color(MONOKAI_GREEN)
		def_statement[0][8].set_color(MONOKAI_ORANGE)
		def_statement.scale(code_scale)
		def_statement.to_edge(LEFT * 13)
		def_statement.shift(UP * fft_code[0].get_center()[1])
		ifft_code.append(def_statement)

		comment = TextMobject(r"\# $P$ - $[P(\omega^0), P(\omega^1), \ldots, P(\omega^{n - 1})]$ value rep")
		comment.scale(code_scale)
		comment.next_to(def_statement, DOWN * 0.5)
		comment.to_edge(LEFT * 14)
		comment.set_color(MONOKAI_GRAY)
		ifft_code.append(comment)

		self.play(
			Write(ifft_code[0])
		)

		self.wait()

		self.play(
			Write(ifft_code[1])
		)
		
		self.wait(3)

		rest_ifft_code = fft_code[2:].copy().to_edge(LEFT * 14)

		self.play(
			TransformFromCopy(fft_code[2:], rest_ifft_code),
			run_time=2
		)

		self.wait()

		line_4 = TextMobject(r"$\omega = (1 / n) * e^{\frac{-2 \pi i}{n}}$")
		line_4.scale(code_scale)
		line_4.next_to(rest_ifft_code[2], DOWN * 0.5)
		line_4.to_edge(LEFT * 14)
		line_4[0][1].set_color(MONOKAI_PINK)
		line_4[0][3].set_color(MONOKAI_PURPLE)
		line_4[0][4].set_color(MONOKAI_PINK)
		line_4[0][7].set_color(MONOKAI_PINK)
		line_4.shift(UP * SMALL_BUFF)

		# line_4[0][7].shift(DOWN * SMALL_BUFF)

		line_6 = TextMobject(r"$y_e, y_o =$ IFFT($P_e$), IFFT($P_o$)")
		line_6[0][5].set_color(MONOKAI_PINK)
		line_6[0][6:10].set_color(MONOKAI_BLUE)
		line_6[0][15:19].set_color(MONOKAI_BLUE)
		line_6.scale(code_scale)
		line_6.next_to(rest_ifft_code[4], DOWN * 0.5)
		line_6.to_edge(LEFT * 14)
		

		self.play(
			Transform(rest_ifft_code[5], line_6),
		)

		self.wait()

		arrow = Arrow(RIGHT, LEFT).scale(0.5).set_color(YELLOW)
		arrow.next_to(rest_ifft_code[3], RIGHT, buff=SMALL_BUFF)

		self.play(
			ShowCreation(arrow)
		)

		self.wait()

		self.play(
			Transform(rest_ifft_code[3], line_4),
			FadeOut(arrow)
		)

		self.wait(8)

class Conclusion(Scene):
	def construct(self):
		recap = TextMobject("Recap")
		recap.scale(1.2).move_to(UP * 3.5)

		h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
		h_line.next_to(recap, DOWN)

		self.play(
			Write(recap),
			ShowCreation(h_line)
		)

		self.make_conclusion(h_line)

	def make_conclusion(self, h_line):
		screen_rect = ScreenRectangle()
		screen_rect.set_height(5.8)
		screen_rect.set_width(7.2)
		screen_rect.move_to(DOWN * 1.7)
		self.play(
			ShowCreation(screen_rect)
		)

		self.wait(7)

		scale = 0.7
		indentation = LEFT * 6
		idea_1 = TextMobject(
			"1. Polynomial multiplication in value representation"
			).scale(scale).next_to(h_line, DOWN).to_edge(indentation)
		# idea_2 = TextMobject(
		# 	r"2. Degree $d$ polynomials $\Leftrightarrow (d + 1)$ points"
		# 	).scale(scale).next_to(idea_1, DOWN).to_edge(indentation)
		idea_2 = TextMobject(
			r"2. Evaluation at $\pm$ pairs"
			).scale(scale).next_to(idea_1, DOWN).to_edge(indentation)
		idea_3 = TextMobject(
			r"3. Evaluation points at $n^{\text{th}}$ roots of unity $\rightarrow$ FFT"
			).scale(scale).next_to(idea_2, DOWN).to_edge(indentation)
		idea_4 = TextMobject(
			r"4. Interpolation can also be solved with the FFT"
			).scale(scale).next_to(idea_3, DOWN).to_edge(indentation)


		self.play(
			Write(idea_1)
		)

		self.wait(11)

		self.play(
			Write(idea_2)
		)

		self.wait(7)

		self.play(
			Write(idea_3)
		)

		self.wait(19)

		self.play(
			Write(idea_4)
		)

		self.wait(17)

class Addendum(Scene):
	def construct(self):
		deeper = TextMobject("FFT Implementation Walkthrough")
		deeper.scale(1.2).move_to(UP * 3.5)

		h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
		h_line.next_to(deeper, DOWN)

		self.play(
			Write(deeper),
			ShowCreation(h_line)
		)

		self.make_conclusion(h_line)

	def make_conclusion(self, h_line):
		screen_rect = ScreenRectangle()
		screen_rect.set_height(8)
		screen_rect.set_width(10)
		screen_rect.move_to(DOWN * 0.5)
		self.play(
			ShowCreation(screen_rect)
		)

		self.wait(45)

class ThumbnailPart1(FFTGraph, GraphScene):
	CONFIG = {
        "x_min": 0,
        "x_max": 2 * TAU,
        "x_axis_width": 6,
        "y_axis_height": 3,
        "x_axis_label": None,
		"y_axis_label": None,
        "graph_origin": DOWN * 2 + LEFT * 6.5,
        "y_min": 2,
        "y_max": -2,
    }
	def construct(self):
		scale = 2
		fast = TextMobject("Fast").scale(scale)
		fourier = TextMobject("Fourier").scale(scale)
		transform = TextMobject("Transform").scale(scale)

		title = VGroup(fast, fourier, transform).arrange_submobjects(DOWN)
		title.move_to(UP * 1.8 + LEFT * 3.7)
		self.add(title)

		self.show_apps(title)

		graph, edge_dict = self.create_graph()
		nodes, edges = self.make_graph_mobject(graph, edge_dict, show_data=False)
		edges_orig = VGroup(*edges).copy()
		entire_graph = VGroup(nodes, edges_orig)
		entire_graph.move_to(RIGHT * 3)
		VGroup(*edges).move_to(RIGHT * 3 + SMALL_BUFF / 2 * RIGHT)

		self.play(
			LaggedStartMap(
				ShowCreation, entire_graph,
				run_time=3
			)
		)

		edges = sorted(edges, key=lambda x: x.get_start()[0])
		edges = VGroup(*edges)
		for i in range(16):
			edges[i].set_color(color=[BLUE, SKY_BLUE])
		for i in range(16, 32):
			edges[i].set_color(color=[GREEN_SCREEN, BLUE])
		for i in range(32, 48):
			edges[i].set_color(color=[MONOKAI_GREEN, GREEN_SCREEN])
		
		self.play(
			LaggedStartMap(
			ShowCreationThenDestruction, edges,
			run_time=3
			)
		)

		self.play(
			LaggedStartMap(
			ShowCreation, edges,
			run_time=3
			)
		)

		self.wait(4)
	
	def show_apps(self, title):
		# wc = WirelessComm()
		# wc.scale(0.7).move_to(UP * 3 + LEFT * 4.5)
		# wc[-1].move_to((wc[8].get_center() + wc[9].get_center()) / 2)
		
		# direction_r = RIGHT * 0.3 + DOWN
		# direction_l = LEFT * 0.3 + DOWN 
		# line_r = Line(wc[-1].get_center(), wc[-1].get_center() + direction_r * 2)
		# line_l = Line(wc[-1].get_center(), wc[-1].get_center() + direction_l * 2)
		# line_r.set_color(BLUE)
		# line_l.set_color(BLUE)

		# intermediate_lines = VGroup()

		# for i in range(1, 6):
		# 	if i % 2 == 1:
		# 		line = Line(
		# 			line_r.point_from_proportion(i / 6),
		# 			line_l.point_from_proportion((i + 1) / 6)
		# 		)
		# 	else:
		# 		line = Line(
		# 			line_l.point_from_proportion(i / 6),
		# 			line_r.point_from_proportion((i + 1) / 6)
		# 		)
		# 	intermediate_lines.add(line)

		# intermediate_lines.set_color(BLUE)


		# self.play(
		# 	ShowCreation(wc[:10]),
		# 	ShowCreation(wc[-1]),
		# 	ShowCreation(line_r),
		# 	ShowCreation(line_l),
		# 	ShowCreation(intermediate_lines)
		# )

		# self.wait()

		# gps = GPS()
		# gps.scale(0.7).move_to(UP * 2 + RIGHT * 4.5)
		# self.play(
		# 	ShowCreation(gps)
		# )

		# self.wait()

		self.setup_axes()

		#This removes tick marks
		self.x_axis.remove(self.x_axis[1])
		self.y_axis.remove(self.y_axis[1])
		self.play(
			Write(self.axes)
		)
		width = 8
		f = lambda x: np.sin(x) + np.cos(2 * x)
		graph = self.get_graph(f).set_stroke(width=width)
		self.play(
			ShowCreation(graph)
		)

		location = self.coords_to_point(0, 1)

		dot = Dot(location).set_color(BLUE)

		g = lambda x: np.sin(x)
		h = lambda x: np.cos(2 * x)

		location_sin = self.coords_to_point(0, 0)
		location_cos = self.coords_to_point(0, 1)

		sin_graph = self.get_graph(g).set_color(YELLOW).set_stroke(width=width)
		dot_sin = Dot(location_sin).set_color(YELLOW)
		
		cos_graph = self.get_graph(h).set_color(GREEN_SCREEN).set_stroke(width=width)
		dot_cos = Dot(location_cos).set_color(GREEN_SCREEN)

		# self.play(
		# 	GrowFromCenter(dot),
		# 	GrowFromCenter(dot_sin),
		# 	GrowFromCenter(dot_cos)
		# )

		# self.play(
		# 	FadeOut(graph)
		# )

		self.play(
			# MoveAlongPath(dot, graph),
			# MoveAlongPath(dot_sin, sin_graph),
			# MoveAlongPath(dot_cos, cos_graph),
			ShowCreation(sin_graph),
			ShowCreation(cos_graph),
			run_time=3,
		)

		# self.play(
		# 	LaggedStart(
		# 		ShowCreation(sin_graph),
		# 		MoveAlongPath(dot_sin, sin_graph),
		# 	),
		# 	run_time=5
		# )

		self.wait()

		# self.play(
		# 	FadeOut(wc[:10]),
		# 	FadeOut(wc[-1]),
		# 	FadeOut(line_r),
		# 	FadeOut(line_l),
		# 	FadeOut(gps),
		# 	FadeOut(intermediate_lines),
		# 	FadeOut(self.axes),
		# 	FadeOut(graph),
		# 	FadeOut(sin_graph),
		# 	FadeOut(cos_graph),
		# 	FadeOut(title)
		# )

		# self.wait()


class ThumbnailPart2(GraphScene):
	def construct(self):
		self.show_unit_circle()

	def show_unit_circle(self):
		plane = ComplexPlane().scale(2)
		plane.add_coordinates()

		subset_complex_plane = VGroup()
		
		outer_square = Square(side_length=4).set_stroke(BLUE_D)

		inner_major_axes = VGroup(
			Line(LEFT * 2, RIGHT * 2), 
			Line(DOWN * 2, UP * 2)
		).set_color(WHITE)

		faded_axes = VGroup(
			Line(LEFT * 2 + UP, RIGHT * 2 + UP), 
			Line(LEFT * 2 + DOWN, RIGHT * 2 + DOWN),
			Line(LEFT * 1 + UP * 2, LEFT * 1 + DOWN * 2), 
			Line(RIGHT * 1 + UP * 2, RIGHT * 1 + DOWN * 2),
		).set_stroke(color=BLUE_D, width=1, opacity=0.5)

		subset_complex_plane.add(outer_square)
		subset_complex_plane.add(inner_major_axes)
		subset_complex_plane.add(faded_axes)

		subset_complex_plane.add(plane.coordinate_labels[6].copy())
		subset_complex_plane.add(plane.coordinate_labels[7].copy())
		subset_complex_plane.add(plane.coordinate_labels[-4].copy())
		subset_complex_plane.add(plane.coordinate_labels[-3].copy())

		self.add(plane)
		self.add(subset_complex_plane)

		title = TextMobject(r"$N^{th}$ roots of unity").scale(1.2).move_to(LEFT * 4 + UP * 3)
		surround_rect = SurroundingRectangle(title, color=WHITE).set_stroke(width=1)
		surround_rect.set_fill(color=BLACK, opacity=0.7)
		title.add_to_back(surround_rect)

		self.wait()
		self.play(
			FadeIn(title)
		)

		self.wait()

		equation = TexMobject("z^n = 1").next_to(title, DOWN, buff=0.5)
		surround_rect_equation = SurroundingRectangle(equation, color=WHITE).set_stroke(width=1)
		surround_rect_equation.set_fill(color=BLACK, opacity=0.7)
		equation.add_to_back(surround_rect_equation)

		self.play(
			FadeIn(equation)
		)
		self.wait(2)

		unit_circle = Circle(radius=2).set_color(GREEN)
		self.play(
			ShowCreation(unit_circle)
		)
		points = [unit_circle.point_from_proportion(i / 16) for i in range(16)]

		dots = [Dot().set_color(YELLOW).move_to(points[i]) for i in range(len(points))]
		dots = VGroup(*dots)

		self.play(
			LaggedStartMap(
			GrowFromCenter, dots,
			run_time=2
			)
		)
		self.wait()

		dashed_line = DashedLine(ORIGIN, dots[1].get_center())
		
		arc_angle = Arc(angle=TAU/16)

		self.play(
			ShowCreation(dashed_line),
			ShowCreation(arc_angle)
		)
		angle = self.make_text(r"angle = $\frac{2 \pi}{n}$", scale=0.8)
		angle.next_to(dots[1], RIGHT)
		self.play(
			FadeIn(angle)
		)

		self.wait(5)

		euler_identity = self.make_text(r"$e^{i \theta} = \cos(\theta) + i \sin(\theta)$")
		euler_identity.move_to(UP * 3 + RIGHT * 3)
		self.play(
			FadeIn(euler_identity)
		)
		self.wait(7)

		define = self.make_text(r"$\omega = e^{\frac{2 \pi i}{n}}$").next_to(equation, DOWN, buff=0.5)
		self.play(
			FadeIn(define)
		)

		self.wait(8)

		self.play(
			FadeOut(angle),
			FadeOut(arc_angle),
			FadeOut(dashed_line)
		)

		w_0 = self.make_text(r"$\omega^0 = 1$", scale=0.7).next_to(dots[0], RIGHT)
		w_1 = self.make_text(r"$\omega^1 = e ^ {\frac{2 \pi i}{n}}$", scale=0.7).next_to(dots[1], RIGHT)
		w_2 = self.make_text(r"$\omega^2 = e ^ {\frac{2 \pi (2) i}{n}}$", scale=0.7).next_to(dots[2], RIGHT)
		w_n_1 = self.make_text(r"$\omega ^{n - 1} = e ^ {\frac{2 \pi (n - 1) i}{n}}$", scale=0.7).next_to(dots[-1], RIGHT)
		w_2.shift(UP * 0.2 + LEFT * 0.1)

		w_examples = VGroup(w_0, w_1, w_2, w_n_1)

		self.play(
			FadeIn(w_examples[0])
		)

		self.play(
			FadeIn(w_examples[1])
		)

		self.play(
			FadeIn(w_examples[2])
		)

		self.play(
			FadeIn(w_examples[3])
		)

		self.wait(5)

		evaluation = self.make_text(r"Evaluate $P(x)$ at $[1, \omega ^1, \omega ^2, \ldots, \omega ^ {n - 1}]$")
		evaluation.move_to(DOWN * 3)
		evaluation[1][14:].set_color(YELLOW)

		self.play(
			FadeIn(evaluation)
		)

		self.wait(10)

		self.play(
			FadeOut(plane),
			FadeOut(title),
			FadeOut(equation),
			FadeOut(define),
			FadeOut(euler_identity),
			FadeOut(w_examples),
			FadeOut(evaluation[0]),
			evaluation[1].next_to, subset_complex_plane, DOWN,
			run_time=2
		)

		why = TextMobject("Why does this work?")
		why.move_to(UP * 3.5)

		self.play(
			Write(why)
		)

		self.wait(11)

		pm_paired_fact = TextMobject(r"$ \omega ^ {j + n / 2} = - \omega ^ j \rightarrow (\omega ^j , \omega ^{j + n / 2})$ are $\pm$ paired").scale(0.8)
		pm_paired_fact.next_to(why, DOWN)

		self.play(
			Write(pm_paired_fact)
		)

		dashed_line = DashedLine(dots[2].get_center(), dots[10].get_center())
		pm_paired = self.make_text(r"$\pm$ paired").scale(0.7).next_to(dots[10], LEFT)

		self.play(
			ShowCreation(dashed_line),
		)

		self.play(
			Write(pm_paired)
		)

		self.wait(5)

		shift_amount = LEFT * 3.7
		self.play(
			subset_complex_plane.shift, shift_amount,
			dots.shift, shift_amount,
			dashed_line.shift, shift_amount,
			unit_circle.shift, shift_amount,
			pm_paired.shift, shift_amount,
			evaluation[1].scale, 0.7,
			evaluation[1].shift, shift_amount
		)

		left_object = VGroup(
			subset_complex_plane, unit_circle, dashed_line,
		)

		left_object.scale(1.2).shift(DOWN * 0.5)

		unit_circle.set_stroke(width=7)

		self.play(
			FadeOut(dots)
		)

		points = [unit_circle.point_from_proportion(i / 16) for i in range(16)]

		dots = [Dot().scale(1.3).set_color(YELLOW).move_to(points[i]) for i in range(len(points))]
		dots = VGroup(*dots)

		self.play(
			LaggedStartMap(
			GrowFromCenter, dots,
			run_time=2
			)
		)

		arrow = TexMobject(r"\Longrightarrow").scale(1.5).move_to(DOWN * 0.5)

		self.play(
			Write(arrow)
		)

		right_plane = subset_complex_plane.copy()
		right_plane.shift(-shift_amount * 2)
		self.play(
			ShowCreation(right_plane),
			run_time=2
		)

		right_unit_circle = unit_circle.copy()

		right_unit_circle.shift(-shift_amount * 2)

		self.play(
			ShowCreation(right_unit_circle),
			run_time=1
		)

		self.wait()

		evaluation_right = TextMobject(r"Evaluate $P_e(x^2)$ and $P_o(x^2)$ at" + "\\\\" + 
			r"$[1, \omega ^2, \omega ^4, \ldots, \omega ^ {2(n / 2 - 1)}]$")
		evaluation_right[0][25:].set_color(ORANGE)
		evaluation_right.scale(0.7)
		evaluation_right.next_to(right_plane, DOWN)

		self.play(
			Write(evaluation_right)
		)

		self.wait()

		new_points = [right_unit_circle.point_from_proportion(i / 8) for i in range(8)]

		new_dots = [Dot().scale(1.3).set_color(ORANGE).move_to(new_points[i]) for i in range(len(new_points))]
		new_dots = VGroup(*new_dots)

		transforms = []
		for i in range(len(dots)):
			if i < len(new_dots):
				transform = TransformFromCopy(dots[i], new_dots[i])
				transforms.append(transform)
			else:
				transform = TransformFromCopy(dots[i], new_dots[i % len(new_dots)].copy())
				transforms.append(transform)

		for transform in transforms:
			self.play(
				transform
			)

		self.wait()

		dashed_line_right = DashedLine(new_dots[1].get_center(), new_dots[5].get_center())
		pm_paired_right = self.make_text(r"$\pm$ paired").scale(0.7).next_to(new_dots[5], LEFT)

		right_object = VGroup(
			right_plane, right_unit_circle, dashed_line_right,
		)

		self.play(
			ShowCreation(dashed_line_right),
		)

		self.play(
			Write(pm_paired_right)
		)

		self.wait()

		n_case = TextMobject(r"$n$ roots of unity").scale(0.7).next_to(evaluation[1], DOWN)

		n_2_case = TextMobject(r"$(n / 2)$ roots of unity").scale(0.7).next_to(evaluation_right, DOWN)
		n_2_case.shift(UP * SMALL_BUFF)
		self.play(
			Write(n_case)
		)

		self.play(
			Write(n_2_case)
		)

		self.play(
			FadeOut(pm_paired_fact),
			FadeOut(why),
			FadeOut(pm_paired_right)
		)

		self.play(
			FadeOut(n_case),
			FadeOut(n_2_case),
			FadeOut(evaluation_right),
			FadeOut(evaluation),
			FadeOut(pm_paired),
			FadeOut(pm_paired_right)
		)

		fft_example = TextMobject("FFT Example").scale(2.4)
		fft_example.move_to(UP * 3)

		self.add(fft_example)

		self.wait(10)

	def make_text(self, string, scale=1):
		text = TextMobject(string).scale(scale)
		surround_rect = SurroundingRectangle(text, color=WHITE).set_stroke(width=1)
		surround_rect.set_fill(color=BLACK, opacity=0.7)
		text.add_to_back(surround_rect)
		return text


def create_normal_human_char():
	face = Circle(radius=1).set_color(WHITE)
	left_eye = Dot().set_color(WHITE)
	right_eye = Dot().set_color(WHITE)
	eyes = VGroup(left_eye, right_eye).arrange_submobjects(RIGHT, buff=0.8)
	eyes.shift(UP * 0.4)

	mouth = Line().scale(0.5).shift(DOWN * 0.4)

	torso = Rectangle(height=2.5, width=1.5).next_to(face, DOWN, buff=0).set_color(WHITE)

	me = TextMobject("Me")
	me.move_to(torso.get_center())

	return VGroup(face, eyes, mouth, torso, me)




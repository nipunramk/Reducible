from big_ol_pile_of_manim_imports import *
import random

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

	def connect_curve(self, counter_clock_adj_self, other, clockwise_adj_other, angle=TAU / 4):
		line_self = Line(counter_clock_adj_self.circle.get_center(), self.circle.get_center())
		unit_vector_self = line_self.get_unit_vector()
		line_other = Line(clockwise_adj_other.circle.get_center(), other.circle.get_center())
		unit_vector_other = line_other.get_unit_vector()
		curve_start = self.circle.get_center() + unit_vector_self * self.radius
		curve_end = other.circle.get_center() + unit_vector_other * self.radius
		line = ArcBetweenPoints(curve_start, curve_end, angle=angle)
		self.neighbors.append(other)
		other.neighbors.append(self)
		self.edges.append(line)
		other.edges.append(line)

	def __repr__(self):
		return 'GraphNode({0})'.format(self.char)

	def __str__(self):
		return 'GraphNode({0})'.format(self.char)


class Transition(Scene):
	def construct(self):
		rectangle = ScreenRectangle()
		rectangle.set_width(8)
		rectangle.set_height(6)

		self.play(
			ShowCreation(rectangle)
		)

		title = TextMobject("Graphs: A Computer Science Perspective")
		title.scale(1.2)
		title.next_to(rectangle, UP)
		self.play(
			Write(title)
		)
		self.wait(5)

		why_study = TextMobject("1. Why Study Graphs?")
		why_study.next_to(rectangle, DOWN)
		self.play(
			Write(why_study)
		)
		self.wait(3)

		definition = TextMobject("2. Definition and Terminology")
		definition.next_to(rectangle, DOWN)
		self.play(
			ReplacementTransform(why_study, definition),
		)
		self.wait(7)

		representation = TextMobject("3. How Does a Computer Represent a Graph?")
		representation.next_to(rectangle, DOWN)
		self.play(
			ReplacementTransform(definition, representation),
		)

		self.wait(3)

		problems = TextMobject("4. Interesting Problems in Graph Theory")
		problems.next_to(rectangle, DOWN)
		self.play(
			ReplacementTransform(representation, problems),
		)

		self.wait(4)

		self.play(
			FadeOut(title),
			FadeOut(problems),
			FadeOut(rectangle),
		)

class Takeaways(Scene):
	def construct(self):
		rectangle = ScreenRectangle()
		rectangle.set_width(8)
		rectangle.set_height(6)

		title = TextMobject("Key Takeaways")
		title.scale(1.2)
		title.next_to(rectangle, UP)
		self.play(
			ShowCreation(rectangle),
			Write(title)
		)

		why_study = TextMobject("1. Graphs Show Up Everywhere")
		why_study.next_to(rectangle, DOWN)
		self.play(
			Write(why_study)
		)
		self.wait(4)

		definition = TextMobject("2. Importance of Terminology")
		definition.next_to(rectangle, DOWN)
		self.play(
			ReplacementTransform(why_study, definition),
		)
		self.wait(10)

		representation = TextMobject("3. Impact of Graph Representation")
		representation.next_to(rectangle, DOWN)
		self.play(
			ReplacementTransform(definition, representation),
		)

		self.wait(9)

		problems = TextMobject("4. Diversity of Problems in Graph Theory")
		problems.next_to(rectangle, DOWN)
		self.play(
			ReplacementTransform(representation, problems),
		)

		self.wait(15)

		self.play(
			FadeOut(title),
			FadeOut(problems),
			FadeOut(rectangle),
		)

class GraphTerminology(Scene):
	def construct(self):
		self.show_definition()

	def show_definition(self):
		self.wait()
		title = TextMobject("Definition")
		title.scale(1.2)
		title.shift(UP * 3.5)

		h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
		h_line.next_to(title, DOWN)
		self.play(
			Write(title),
			ShowCreation(h_line)
		)

		self.wait()

		definition = TextMobject(
			r"A graph $G = (V, E)$ is a set of vertices $V$ and edges $E$ where" + "\\\\",
			r"each edge $(u, v)$ is a connection between vertices. $u, v \in V$"
		)
		definition.scale(0.8)

		definition.next_to(h_line, DOWN)
		for i in range(len(definition)):
			self.play(
				Write(definition[i]),
				run_time=3
			)

		graph, edge_dict = self.create_small_graph()
		nodes, edges = self.make_graph_mobject(graph, edge_dict)
		entire_graph = VGroup(nodes, edges)

		self.play(
			FadeIn(entire_graph)
		)

		self.wait(3)

		edge_highlight = self.sharpie_edge(edge_dict, 2, 3, animate=False)

		edge_label = TextMobject("Edge")
		edge_label.move_to(DOWN * 2 + RIGHT * 2)
		edge_arrow = Arrow(DOWN * 2 + RIGHT * 1.5, DOWN * 2)
		
		self.play(
			ShowCreation(edge_highlight)
		)

		self.play(
			Indicate(definition[1][8:13]),
			FadeIn(edge_label),
			ShowCreation(edge_arrow),
		)

		self.wait()

		surround_circle = self.highlight_node(graph, 1)

		
		v_arrow = Arrow(UP + RIGHT, UP + LEFT * 0.7)
		vertex_label = TextMobject("Vertex/Node")
		vertex_label.next_to(v_arrow, RIGHT)
		self.play(
			FadeIn(vertex_label),
		)
		self.play(
			ShowCreation(v_arrow),
			run_time=2
		)

		self.wait(4)

		self.play(
			FadeOut(vertex_label),
			FadeOut(edge_label),
			FadeOut(v_arrow),
			FadeOut(edge_arrow),
			FadeOut(surround_circle),
			FadeOut(edge_highlight)
		)

		entire_graph_shifted_scaled = entire_graph.copy()
		entire_graph_shifted_scaled.shift(LEFT * 2.5)
		self.play(
			Transform(entire_graph, entire_graph_shifted_scaled)
		)
		self.wait()

		vertex_set = TextMobject(r"$V = \{0, 1, 2, 3, 4, 5\}$")
		vertex_set.scale(0.8)
		edge_set = TextMobject(r"$E = \{(0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (3, 4)\}$")
		edge_set.scale(0.8)
		vertex_set.move_to(UP * 0.2 + RIGHT * 2.5)
		edge_set.move_to(DOWN * 2.2 + RIGHT * 2.5)

		self.play(
			Write(vertex_set[:2])
		)

		node_surrounding_circles = []
		for i in range(len(graph)):
			circle = self.highlight_node(graph, i, animate=False)
			node_surrounding_circles.append(circle)

		self.play(
			*[ShowCreation(circle) for circle in node_surrounding_circles]
		)

		highlighted_nodes = VGroup(*[VGroup(node, circle) for node, circle in zip(nodes, node_surrounding_circles)])
		self.play(
			*[TransformFromCopy(highlighted_nodes, vertex_set[2:])],
			run_time=2
		)

		self.play(
			*[FadeOut(obj) for obj in node_surrounding_circles]
		)

		self.play(
			Write(edge_set[:2])
		)

		edge_lines = []
		for key in edge_dict:
			edge = self.sharpie_edge(edge_dict, key[0], key[1], animate=False)
			edge_lines.append(edge)

		self.play(
			*[ShowCreation(line) for line in edge_lines]
		)

		self.play(
			TransformFromCopy(VGroup(*edge_lines), edge_set[2:]),
			run_time=2
		)

		self.wait(5)

		self.play(
			*[FadeOut(line) for line in edge_lines]
		)

		self.play(
			FadeOut(definition),
			FadeOut(entire_graph),
			FadeOut(vertex_set),
			FadeOut(edge_set),
			FadeOut(title),
			FadeOut(h_line)
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
		edges[(0, 2)] = node_0.connect(node_2)
		edges[(0, 3)] = node_0.connect(node_3)

		edges[(1, 3)] = node_1.connect(node_3)

		edges[(2, 3)] = node_2.connect(node_3)

		edges[(3, 4)] = node_3.connect(node_4)

		graph.append(node_0)
		graph.append(node_1)
		graph.append(node_2)
		graph.append(node_3)
		graph.append(node_4)

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

	def show_path_in_graph(self, graph, edge_dict, scale_factor=1):
		self.highlight_node(graph, 0, scale_factor=scale_factor)
		
		last_edge = self.sharpie_edge(edge_dict, 0, 6, scale_factor=scale_factor)
		angle = self.find_angle_of_intersection(graph, last_edge.get_end(), 6)
		self.highlight_node(graph, 6, start_angle=angle/360 * TAU, scale_factor=scale_factor)
		
		last_edge = self.sharpie_edge(edge_dict, 6, 7, scale_factor=scale_factor)
		angle = self.find_angle_of_intersection(graph, last_edge.get_end(), 7)
		self.highlight_node(graph, 7, start_angle=angle/360 * TAU, scale_factor=scale_factor)
		
		last_edge = self.sharpie_edge(edge_dict, 7, 3, scale_factor=scale_factor)
		angle = self.find_angle_of_intersection(graph, last_edge.get_end(), 3)
		self.highlight_node(graph, 3, start_angle=angle/360 * TAU, scale_factor=scale_factor)
		
		last_edge = self.sharpie_edge(edge_dict, 3, 2, scale_factor=scale_factor)
		angle = self.find_angle_of_intersection(graph, last_edge.get_end(), 2)
		self.highlight_node(graph, 2, start_angle=angle/360 * TAU, scale_factor=scale_factor)

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

	def highlight_edge(self, edge_dict, v, u, color=GREEN_SCREEN):
		switch = False
		if v > u:
			u, v = v, u
			switch = True
		edge = edge_dict[(v, u)]
		normal_1, normal_2 = edge.get_unit_normals()
		scale_factor = 1.5
		if not switch:
			line_1 = Line(edge.get_start() + normal_1 * SMALL_BUFF / scale_factor,
				edge.get_end() + normal_1 * SMALL_BUFF / scale_factor)
			line_2 = Line(edge.get_start() + normal_2 * SMALL_BUFF / scale_factor,
				edge.get_end() + normal_2 * SMALL_BUFF / scale_factor)
		else:
			line_1 = Line(edge.get_end() + normal_1 * SMALL_BUFF / scale_factor,
				edge.get_start() + normal_1 * SMALL_BUFF / scale_factor)
			line_2 = Line(edge.get_end() + normal_2 * SMALL_BUFF / scale_factor,
				edge.get_start() + normal_2 * SMALL_BUFF / scale_factor)
					
		line_1.set_stroke(width=8)
		line_2.set_stroke(width=8)
		line_1.set_color(color)
		line_2.set_color(color)

		self.play(
			ShowCreation(line_1),
			ShowCreation(line_2),
		)

	def sharpie_edge(self, edge_dict, u, v, color=GREEN_SCREEN, scale_factor=1, animate=True):
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
		line.set_stroke(width=16 * scale_factor)
		line.set_color(color)
		if animate:
			self.play(
				ShowCreation(line)
			)
		return line

	def highlight_node(self, graph, index, color=GREEN_SCREEN, 
		start_angle=TAU/2, scale_factor=1, animate=True):
		node = graph[index]
		surround_circle = Circle(radius=node.circle.radius * scale_factor, TAU=-TAU, start_angle=start_angle)
		surround_circle.move_to(node.circle.get_center())
		# surround_circle.scale(1.15)
		surround_circle.set_stroke(width=8 * scale_factor)
		surround_circle.set_color(color)
		surround_circle.set_fill(opacity=0)
		if animate:
			self.play(
				ShowCreation(surround_circle)
			)
		return surround_circle

	def create_initial_graph(self):
		graph = []
		edges = {}

		radius, scale = 0.4, 0.9
		SHIFT = RIGHT * 2.5
		node_0 = GraphNode('0', position=LEFT * 2.5, radius=radius, scale=scale)
		node_1 = GraphNode('1', position=RIGHT * 3, radius=radius, scale=scale)
		node_2 = GraphNode('2', position=RIGHT * 1.5 + UP, radius=radius, scale=scale)
		node_3 = GraphNode('3', position=UP * 2.5 + SHIFT, radius=radius, scale=scale)
		node_4 = GraphNode('4', position=DOWN * 2, radius=radius, scale=scale)
		node_5 = GraphNode('5', position=DOWN + RIGHT * 2, radius=radius, scale=scale)
		node_6 = GraphNode('6', position=LEFT + UP, radius=radius, scale=scale)
		node_7 = GraphNode('7', position=LEFT  * 2 + UP * 2 + SHIFT, radius=radius, scale=scale)
		node_8 = GraphNode('8', position=ORIGIN, radius=radius, scale=scale)

		edge_0_6 = node_0.connect(node_6)
		edges[(0, 6)] = edge_0_6
		# edges[(6, 0)] = edge_0_6
		edge_0_8 = node_0.connect(node_8)
		edges[(0, 8)] = edge_0_8
		# edges[(8, 0)] = edge_0_8
		edge_0_4 = node_0.connect(node_4)
		edges[(0, 4)] = edge_0_4
		# edges[(4, 0)] = edge_0_4
		edge_1_2 = node_1.connect(node_2)
		edges[(1, 2)] = edge_1_2
		# edges[(2, 1)] = edge_1_2
		edge_1_5 = node_1.connect(node_5)
		edges[(1, 5)] = edge_1_5
		# edges[(5, 1)] = edge_1_5
		edge_2_3 = node_2.connect(node_3)
		edges[(2, 3)] = edge_2_3
		# edges[(3, 2)] = edge_2_3
		edge_3_7 = node_3.connect(node_7)
		edges[(3, 7)] = edge_3_7
		# edges[(7, 3)] = edge_3_7
		edge_4_5 =  node_4.connect(node_5)
		edges[(4, 5)] = edge_4_5
		# edges[(5, 4)] = edge_4_5
		edge_6_7 = node_6.connect(node_7)
		edges[(6, 7)] = edge_6_7
		# edges[(7, 6)] = edge_6_7
		edge_1_8 = node_1.connect(node_8)
		edges[(1, 8)] = edge_1_8
		# edges[(8, 1)] = edge_1_8

		graph.append(node_0)
		graph.append(node_1)
		graph.append(node_2)
		graph.append(node_3)
		graph.append(node_4)
		graph.append(node_5)
		graph.append(node_6)
		graph.append(node_7)
		graph.append(node_8)

		return graph, edges

	def create_disconnected_graph(self):
		graph = []
		edges = {}

		radius, scale = 0.4, 0.9
		SHIFT = RIGHT * 2.5
		right_shift = RIGHT * 0.5
		left_shift = LEFT * 0.5
		node_0 = GraphNode('0', position=LEFT * 2.5 + left_shift, radius=radius, scale=scale)
		node_1 = GraphNode('1', position=RIGHT * 3 + right_shift, radius=radius, scale=scale)
		node_2 = GraphNode('2', position=RIGHT * 1.5 + UP + right_shift, radius=radius, scale=scale)
		node_3 = GraphNode('3', position=UP * 2.5 + SHIFT + right_shift, radius=radius, scale=scale)
		node_4 = GraphNode('4', position=DOWN * 2 + left_shift, radius=radius, scale=scale)
		node_5 = GraphNode('5', position=DOWN + RIGHT * 2 + right_shift, radius=radius, scale=scale)
		node_6 = GraphNode('6', position=LEFT + UP + left_shift, radius=radius, scale=scale)
		node_7 = GraphNode('7', position=LEFT  * 2 + UP * 2 + SHIFT + left_shift, radius=radius, scale=scale)
		node_8 = GraphNode('8', position=ORIGIN + left_shift, radius=radius, scale=scale)

		edge_0_6 = node_0.connect(node_6)
		edges[(0, 6)] = edge_0_6

		edge_0_8 = node_0.connect(node_8)
		edges[(0, 8)] = edge_0_8

		edge_0_4 = node_0.connect(node_4)
		edges[(0, 4)] = edge_0_4

		edge_1_2 = node_1.connect(node_2)
		edges[(1, 2)] = edge_1_2

		edge_1_5 = node_1.connect(node_5)
		edges[(1, 5)] = edge_1_5

		edge_2_3 = node_2.connect(node_3)
		edges[(2, 3)] = edge_2_3

		edge_6_7 = node_6.connect(node_7)
		edges[(6, 7)] = edge_6_7



		graph.append(node_0)
		graph.append(node_1)
		graph.append(node_2)
		graph.append(node_3)
		graph.append(node_4)
		graph.append(node_5)
		graph.append(node_6)
		graph.append(node_7)
		graph.append(node_8)

		return graph, edges

	def make_graph_mobject(self, graph, edge_dict, node_color=DARK_BLUE_B, 
		stroke_color=BLUE, data_color=WHITE, edge_color=GRAY, scale_factor=1):
		nodes = []
		edges = []
		for node in graph:
			node.circle.set_fill(color=node_color, opacity=0.5)
			node.circle.set_stroke(color=stroke_color)
			node.data.set_color(color=data_color)
			nodes.append(VGroup(node.circle, node.data))

		for edge in edge_dict.values():
			edge.set_stroke(width=7*scale_factor)
			edge.set_color(color=edge_color)
			edges.append(edge)
		return VGroup(*nodes), VGroup(*edges)

class NeighborTerminology(GraphTerminology):
	def construct(self):
		term_title = TextMobject("Terminology")
		term_title.scale(1.2)
		term_title.move_to(UP * 3.5)
		h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
		h_line.next_to(term_title, DOWN)
		self.play(
			Write(term_title),
			ShowCreation(h_line)
		)
		self.wait(5)	
		self.show_neighbors_term(h_line)

	def show_neighbors_term(self, h_line):
		scale_factor = 0.9
		graph, edge_dict = self.create_initial_graph()
		node_mobjects, edge_mobjects = self.make_graph_mobject(graph, edge_dict, scale_factor=scale_factor)
		entire_graph = VGroup(node_mobjects, edge_mobjects)

		neighbors_term = TextMobject(r"Neighbors - vertices $u$ and $v$ are neighbors if an edge $(u, v)$ connects them")
		neighbors_term[:9].set_color(GREEN_SCREEN)
		neighbors_term[26:35].set_color(GREEN_SCREEN)
		neighbors_term.scale(0.8)
		neighbors_term.next_to(h_line, DOWN)
		entire_graph.scale(scale_factor)
		entire_graph.shift(DOWN * 1.5)

		self.play(
			Write(neighbors_term[:9]),
			run_time=1
		)
		self.wait()

		self.play(
			Write(neighbors_term[9:]),
			run_time=2
		)
		self.wait()

		self.play(
			FadeIn(entire_graph)
		)
		self.wait()


		single_example = TextMobject("Ex: 1 and 8 are neighbors")
		single_example[-9:].set_color(GREEN_SCREEN)
		single_example.scale(0.8)
		single_example.next_to(neighbors_term, DOWN)
		single_example.shift(LEFT * 3)

		v_1_highlight = self.highlight_node(graph, 1, scale_factor=scale_factor, animate=False)
		v_8_highlight = self.highlight_node(graph, 8, scale_factor=scale_factor, animate=False)
		edge_highlight = self.sharpie_edge(edge_dict, 1, 8, scale_factor=scale_factor, animate=False)
		self.play(
			ShowCreation(v_1_highlight),
			ShowCreation(v_8_highlight)
		)
		self.wait()

		self.play(
			FadeIn(single_example),
		)

		self.play(
			GrowFromCenter(edge_highlight)
		)

		self.wait(2)

		self.play(
			FadeOut(v_1_highlight),
			FadeOut(v_8_highlight),
			FadeOut(edge_highlight)
		)

		self.wait(2)

		neighbors_example = TextMobject(r"neighbors$(0) = \{4, 6, 8\}$")
		neighbors_example[:9].set_color(GREEN_SCREEN)
		neighbors_example.scale(0.8)
		neighbors_example.next_to(single_example, RIGHT)
		neighbors_example.shift(RIGHT * 0.2)
		self.play(
			Write(neighbors_example[:13])
		)
		self.wait(3)

		v_6_highlight = self.highlight_node(graph, 6, scale_factor=scale_factor, animate=False)
		v_8_highlight = self.highlight_node(graph, 8, scale_factor=scale_factor, animate=False)
		v_4_highlight = self.highlight_node(graph, 4, scale_factor=scale_factor, animate=False)
		edge_highlight_06 = self.sharpie_edge(edge_dict, 0, 6, scale_factor=scale_factor, animate=False)
		edge_highlight_08 = self.sharpie_edge(edge_dict, 0, 8, scale_factor=scale_factor, animate=False)
		edge_highlight_04 = self.sharpie_edge(edge_dict, 0, 4, scale_factor=scale_factor, animate=False)

		self.play(
			ShowCreation(edge_highlight_06),
			ShowCreation(edge_highlight_08),
			ShowCreation(edge_highlight_04)
		)

		self.play(
			ShowCreation(v_4_highlight),
			ShowCreation(v_6_highlight),
			ShowCreation(v_8_highlight),
		)
		self.wait()
		self.play(
			FadeIn(neighbors_example[13:])
		)

		self.wait(4)

		self.play(
			FadeOut(v_6_highlight),
			FadeOut(v_8_highlight),
			FadeOut(v_4_highlight),
			FadeOut(edge_highlight_04),
			FadeOut(edge_highlight_06),
			FadeOut(edge_highlight_08),
			FadeOut(single_example),
			FadeOut(neighbors_example),
			FadeOut(neighbors_term)
		)
		self.wait()

	def show_connectivity_term(self, entire_graph, graph, edge_dict, title, h_line, scale_factor):
		connectivity_term = TextMobject(
			"Connectivity - two vertices are connected if a path exists between them." + "\\\\",
			"A graph is called connected when all vertices are connected."
		)
		connectivity_term.scale(0.8)
		connectivity_term.next_to(h_line, DOWN)

		connectivity_term[0][:12].set_color(GREEN_SCREEN)
		connectivity_term[0][27:36].set_color(GREEN_SCREEN)
		connectivity_term[1][14:23].set_color(GREEN_SCREEN)
		connectivity_term[1][-10:-1].set_color(GREEN_SCREEN)
		self.play(
			Write(connectivity_term[0][:12]),
		)
		self.wait()

		self.play(
			Write(connectivity_term[0][12:]),
			run_time=2
		)

		self.wait()

		self.play(
			Write(connectivity_term[1]),
			run_time=2
		)
		self.wait(5)

		self.play(
			Uncreate(entire_graph),
			run_time=2
		)
		self.wait()

		graph, edge_dict = self.create_disconnected_graph()
		nodes, edges = self.make_graph_mobject(graph, edge_dict, scale_factor=scale_factor)
		entire_graph = VGroup(nodes, edges)
		entire_graph.shift(DOWN * 1.5)
		self.play(
			FadeIn(entire_graph)
		)
		self.wait(5)

		self.play(
			FadeOut(connectivity_term)
		)

		self.wait()

		connected_component_term = TextMobject(r"Connected Component - a subset of vertices $V_i \subseteq V$ that are connected")
		connected_component_term.scale(0.8)
		connected_component_term[:18].set_color(GREEN_SCREEN)
		connected_component_term.next_to(h_line, DOWN)
		self.play(
			Write(connected_component_term[:18])
		)
		self.wait()
		self.play(
			Write(connected_component_term[18:])
		)
		self.wait()

		connected_component_ex = TextMobject(
			r"Ex: $V_1 = \{0, 4, 6, 7, 8\}$", 
			r", $V_2 = \{1, 2, 3, 5\}$"
		)

		connected_component_ex.scale(0.8)
		connected_component_ex.next_to(connected_component_term, DOWN)

		self.play(
			Write(connected_component_ex[0][:5])
		)
		self.wait()

		component_1 = VGroup(
			nodes[0], nodes[4], nodes[6], nodes[7], nodes[8]
		)

		surround_rectangle_1 = SurroundingRectangle(component_1, color=GREEN_SCREEN, buff=SMALL_BUFF)
		self.play(
			ShowCreation(surround_rectangle_1)
		)
		self.wait()

		self.play(
			Write(connected_component_ex[0][5:])
		)
		self.wait()

		self.play(
			Write(connected_component_ex[1][:4])
		)
		self.wait()

		component_2 = VGroup(
			nodes[1], nodes[2], nodes[3], nodes[5]
		)

		surround_rectangle_2 = SurroundingRectangle(component_2, color=GREEN_SCREEN, buff=SMALL_BUFF)
		self.play(
			ShowCreation(surround_rectangle_2)
		)
		self.wait()

		self.play(
			Write(connected_component_ex[1][4:])
		)
		self.wait(5)

		self.play(
			FadeOut(connected_component_term),
			FadeOut(connected_component_ex),
			FadeOut(surround_rectangle_1),
			FadeOut(surround_rectangle_2),
			FadeOut(h_line),
			FadeOut(title),
			FadeOut(entire_graph)
		)
		self.wait()

	def show_terminology_section(self, title, h_line):

		self.show_connectivity_term(entire_graph, graph, edge_dict, title, h_line, scale_factor)

class DegreeTerminology(GraphTerminology):
	def construct(self):
		term_title = TextMobject("Terminology")
		term_title.scale(1.2)
		term_title.move_to(UP * 3.5)
		h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
		h_line.next_to(term_title, DOWN)
		self.play(
			Write(term_title),
			ShowCreation(h_line)
		)
		self.wait()	
		self.show_degree_term(h_line)

	def show_degree_term(self, h_line):
		scale_factor = 0.9
		graph, edge_dict = self.create_initial_graph()
		node_mobjects, edge_mobjects = self.make_graph_mobject(graph, edge_dict, scale_factor=scale_factor)
		entire_graph = VGroup(node_mobjects, edge_mobjects)
		entire_graph.scale(scale_factor)
		entire_graph.shift(DOWN * 1.5)

		self.play(
			FadeIn(entire_graph)
		)

		degree_definition = TextMobject(r"Degree - degree$(v)$ is equal to the number of edges connected to $v$")
		degree_definition.scale(0.8)
		degree_definition.next_to(h_line, DOWN)
		degree_definition[:6].set_color(GREEN_SCREEN)
		degree_definition[7:13].set_color(GREEN_SCREEN)
		self.play(
			Write(degree_definition[:6])
		)
		self.wait(3)
		self.play(
			Write(degree_definition[6:]),
			run_time=2
		)
		self.wait(5)

		degree_example = TextMobject(r"Ex: degree$(0) = 3$", r", degree$(3) = 2$")
		degree_example.scale(0.8)
		degree_example[0][3:9].set_color(GREEN_SCREEN)
		degree_example[1][1:7].set_color(GREEN_SCREEN)
		degree_example.next_to(degree_definition, DOWN)
		self.play(
			Write(degree_example[0][:12]),
		)

		v_0_highlight = self.highlight_node(graph, 0, animate=False)
		self.play(
			ShowCreation(v_0_highlight)
		)

		edge_highlight_06 = self.sharpie_edge(edge_dict, 0, 6, scale_factor=scale_factor, animate=False)
		edge_highlight_08 = self.sharpie_edge(edge_dict, 0, 8, scale_factor=scale_factor, animate=False)
		edge_highlight_04 = self.sharpie_edge(edge_dict, 0, 4, scale_factor=scale_factor, animate=False)

		self.play(
			ShowCreation(edge_highlight_06),
			ShowCreation(edge_highlight_08),
			ShowCreation(edge_highlight_04)
		)

		self.play(
			Write(degree_example[0][12:]),
		)
		self.wait()

		self.play(
			FadeOut(v_0_highlight),
			FadeOut(edge_highlight_04),
			FadeOut(edge_highlight_06),
			FadeOut(edge_highlight_08),
		)

		self.play(
			Write(degree_example[1][:10])
		)

		v_3_highlight = self.highlight_node(graph, 3, animate=False)
		self.play(
			ShowCreation(v_3_highlight)
		)

		edge_highlight_37 = self.sharpie_edge(edge_dict, 3, 7, scale_factor=scale_factor, animate=False)
		edge_highlight_32 = self.sharpie_edge(edge_dict, 3, 2, scale_factor=scale_factor, animate=False)
		self.play(
			ShowCreation(edge_highlight_37),
			ShowCreation(edge_highlight_32)
		)		

		self.play(
			Write(degree_example[1][10:])
		)

		self.wait(3)

		self.play(
			FadeOut(degree_definition),
			FadeOut(degree_example),
			FadeOut(edge_highlight_37),
			FadeOut(edge_highlight_32),
			FadeOut(v_3_highlight)
		)
		self.wait()

class PathTerminology(GraphTerminology):
	def construct(self):
		term_title = TextMobject("Terminology")
		term_title.scale(1.2)
		term_title.move_to(UP * 3.5)
		h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
		h_line.next_to(term_title, DOWN)
		self.play(
			Write(term_title),
			ShowCreation(h_line)
		)
		self.wait()	
		self.show_path_term(h_line)

	def show_path_term(self, h_line):
		scale_factor = 0.9
		graph, edge_dict = self.create_initial_graph()
		node_mobjects, edge_mobjects = self.make_graph_mobject(graph, edge_dict, scale_factor=scale_factor)
		entire_graph = VGroup(node_mobjects, edge_mobjects)
		entire_graph.scale(scale_factor)
		entire_graph.shift(DOWN * 1.5)

		self.play(
			FadeIn(entire_graph)
		)

		path_term = TextMobject("Path - sequence of vertices connected by edges")
		path_term.scale(0.8)
		path_term.next_to(h_line, DOWN)
		path_term[:4].set_color(GREEN_SCREEN)
		self.play(
			Write(path_term[:4])
		)
		self.wait()
		self.play(
			Write(path_term[4:]),
			run_time=2
		)

		path_example = TextMobject(r"Ex: $0 \rightarrow 6 \rightarrow 7 \rightarrow 3 \rightarrow 2$ is a path")
		path_example.scale(0.8)
		path_example[-4:].set_color(GREEN_SCREEN)
		path_example.next_to(path_term, DOWN)

		path = [0, 6, 7, 3, 2]
		path_objects = self.show_path(graph, edge_dict, path, scale_factor=scale_factor)
		self.play(
			FadeIn(path_example)
		)

		self.wait(8)

		path_length_term = TextMobject("Path length - number of edges in a path")
		path_length_term.scale(0.8)
		path_length_term[:10].set_color(GREEN_SCREEN)
		path_length_term.next_to(h_line, DOWN)
		self.play(
			Transform(path_term, path_length_term)
		)
		self.wait(6)

		path_length_example = TextMobject(r"Ex: $0 \rightarrow 6 \rightarrow 7 \rightarrow 3 \rightarrow 2$ has length 4")
		path_length_example.scale(0.8)
		path_length_example[-7:-1].set_color(GREEN_SCREEN)
		path_length_example.next_to(path_term, DOWN)

		self.play(
			Transform(path_example, path_length_example)
		)
		self.wait(3)

		fadeouts = [FadeOut(obj) for obj in path_objects] + [FadeOut(path_example), FadeOut(path_term)]
		self.play(
			*fadeouts
		)

		self.wait()

class CycleTerminology(GraphTerminology):
	def construct(self):
		term_title = TextMobject("Terminology")
		term_title.scale(1.2)
		term_title.move_to(UP * 3.5)
		h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
		h_line.next_to(term_title, DOWN)
		self.play(
			Write(term_title),
			ShowCreation(h_line)
		)
		self.wait()	
		self.show_cycle_term(h_line)

	def show_cycle_term(self, h_line):
		scale_factor = 0.9
		graph, edge_dict = self.create_initial_graph()
		node_mobjects, edge_mobjects = self.make_graph_mobject(graph, edge_dict, scale_factor=scale_factor)
		entire_graph = VGroup(node_mobjects, edge_mobjects)
		entire_graph.scale(scale_factor)
		entire_graph.shift(DOWN * 1.5)

		self.play(
			FadeIn(entire_graph)
		)

		cycle_term =  TextMobject("Cycle - path that starts and ends at the same vertex")
		cycle_term.scale(0.8)
		cycle_term[:5].set_color(GREEN_SCREEN)
		cycle_term.next_to(h_line, DOWN)

		self.play(
			Write(cycle_term[:5])
		)
		self.wait()
		self.play(
			Write(cycle_term[5:]),
			run_time=2
		)
		self.wait()

		cycle_example = TextMobject(r"Ex: $0 \rightarrow 8 \rightarrow 1 \rightarrow 5 \rightarrow 4 \rightarrow 0$ is a cycle")
		cycle_example.scale(0.8)
		cycle_example[-5:].set_color(GREEN_SCREEN)
		cycle_example.next_to(cycle_term, DOWN)

		cycle_path = [0, 8, 1, 5, 4, 0]
		cycle_objects = self.show_path(graph, edge_dict, cycle_path, scale_factor=scale_factor)
		self.play(
			FadeIn(cycle_example)
		)

		self.wait(6)

		fadeouts = [FadeOut(obj) for obj in cycle_objects] + [FadeOut(cycle_example), FadeOut(cycle_term)]
		self.play(
			*fadeouts
		)
		self.wait()

class ConnectivityTerminology(GraphTerminology):
	def construct(self):
		term_title = TextMobject("Terminology")
		term_title.scale(1.2)
		term_title.move_to(UP * 3.5)
		h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
		h_line.next_to(term_title, DOWN)
		self.play(
			Write(term_title),
			ShowCreation(h_line)
		)
		self.wait()	
		self.show_connectivity_term(h_line, term_title)

	def show_connectivity_term(self, h_line, title):
		scale_factor = 0.9
		graph, edge_dict = self.create_initial_graph()
		node_mobjects, edge_mobjects = self.make_graph_mobject(graph, edge_dict, scale_factor=scale_factor)
		entire_graph = VGroup(node_mobjects, edge_mobjects)
		entire_graph.scale(scale_factor)
		entire_graph.shift(DOWN * 1.5)
		
		self.play(
			FadeIn(entire_graph)
		)

		connectivity_term = TextMobject(
			"Connectivity - two vertices are connected if a path exists between them." + "\\\\",
			"A graph is called connected when all vertices are connected."
		)
		connectivity_term.scale(0.8)
		connectivity_term.next_to(h_line, DOWN)

		connectivity_term[0][:12].set_color(GREEN_SCREEN)
		connectivity_term[0][27:36].set_color(GREEN_SCREEN)
		connectivity_term[1][14:23].set_color(GREEN_SCREEN)
		connectivity_term[1][-10:-1].set_color(GREEN_SCREEN)
		self.play(
			Write(connectivity_term[0][:12]),
		)
		self.wait(5)

		self.play(
			Write(connectivity_term[0][12:]),
			run_time=2
		)

		self.wait(8)

		self.play(
			Write(connectivity_term[1]),
			run_time=2
		)
		self.wait(14)

		self.play(
			Uncreate(entire_graph),
			run_time=2
		)

		graph, edge_dict = self.create_disconnected_graph()
		nodes, edges = self.make_graph_mobject(graph, edge_dict, scale_factor=scale_factor)
		entire_graph = VGroup(nodes, edges)
		entire_graph.shift(DOWN * 1.4)
		self.play(
			FadeIn(entire_graph)
		)
		self.wait(10)

		self.play(
			FadeOut(connectivity_term)
		)

		self.wait()

		connected_component_term = TextMobject(r"Connected Component - a subset of vertices $V_i \subseteq V$ that is connected")
		connected_component_term.scale(0.8)
		connected_component_term[:18].set_color(GREEN_SCREEN)
		connected_component_term.next_to(h_line, DOWN)
		self.play(
			Write(connected_component_term[:18])
		)
		self.wait()
		self.play(
			Write(connected_component_term[18:])
		)
		self.wait()

		connected_component_ex = TextMobject(
			r"Ex: $V_1 = \{0, 4, 6, 7, 8\}$", 
			r", $V_2 = \{1, 2, 3, 5\}$"
		)

		connected_component_ex.scale(0.8)
		connected_component_ex.next_to(connected_component_term, DOWN)

		self.play(
			Write(connected_component_ex[0][:5])
		)
		self.wait()

		component_1 = VGroup(
			nodes[0], nodes[4], nodes[6], nodes[7], nodes[8]
		)

		surround_rectangle_1 = SurroundingRectangle(component_1, color=GREEN_SCREEN, buff=SMALL_BUFF)
		self.play(
			ShowCreation(surround_rectangle_1)
		)
		self.wait()

		self.play(
			Write(connected_component_ex[0][5:])
		)
		self.wait()

		self.play(
			Write(connected_component_ex[1][:4])
		)

		component_2 = VGroup(
			nodes[1], nodes[2], nodes[3], nodes[5]
		)

		surround_rectangle_2 = SurroundingRectangle(component_2, color=GREEN_SCREEN, buff=SMALL_BUFF)
		self.play(
			ShowCreation(surround_rectangle_2)
		)

		self.play(
			Write(connected_component_ex[1][4:])
		)
		self.wait(3)

		self.play(
			FadeOut(connected_component_term),
			FadeOut(connected_component_ex),
			FadeOut(surround_rectangle_1),
			FadeOut(surround_rectangle_2),
			FadeOut(h_line),
			FadeOut(title),
			FadeOut(entire_graph)
		)
		self.wait()

class TypesOfGraphs(Scene):
	def construct(self):
		self.show_types_of_graphs()

	def show_types_of_graphs(self):
		title = TextMobject("Types of Graphs")
		title.scale(1.2)
		title.shift(UP * 3.5)

		h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
		h_line.next_to(title, DOWN)
		self.play(
			Write(title),
			ShowCreation(h_line)
		)

		self.wait()
		
		graph_with_label = self.show_undirected_graph()

		self.show_directed_graph(graph_with_label)

		self.show_weighted_graph()

		self.show_trees(title, h_line)

	def show_undirected_graph(self):
		graph, edge_dict = self.create_small_graph()
		nodes, edges = self.make_graph_mobject(graph, edge_dict)
		entire_graph = VGroup(nodes, edges)
		scale_factor = 0.9
		entire_graph.scale(scale_factor)
		entire_graph.shift(UP * 1)
		self.play(
			FadeIn(entire_graph)
		)
		self.wait(3)

		undirected_graph = TextMobject("Undirected Graph")
		undirected_graph.scale(0.9)
		undirected_graph.next_to(entire_graph, DOWN)
		self.play(
			FadeIn(undirected_graph)
		)
		self.wait(2)

		implication = TextMobject(r"Edge $(u, v)$ implies $(v, u)$")
		implication.scale(0.8)
		implication.next_to(undirected_graph, DOWN)
		self.play(
			Write(implication)
		)

		self.wait(6)

		graph_with_label = VGroup(entire_graph, undirected_graph, implication)
		graph_with_label_shifted = graph_with_label.copy()
		graph_with_label_shifted.shift(LEFT * 3.5)
		self.play(
			Transform(graph_with_label, graph_with_label_shifted)
		)

		self.wait()

		return graph_with_label

	def show_directed_graph(self, graph_with_label):
		graph, edge_dict = self.create_small_directed_graph()
		nodes, edges = self.make_graph_mobject(graph, edge_dict, directed=True)
		entire_graph = VGroup(nodes, edges)
		scale_factor = 0.9
		entire_graph.scale(scale_factor)
		entire_graph.shift(UP * 1)
		entire_graph.shift(RIGHT * 3.5)
		self.play(
			FadeIn(entire_graph)
		)

		directed_graph = TextMobject("Directed Graph")
		directed_graph.scale(0.9)
		directed_graph.next_to(entire_graph, DOWN)
		self.play(
			FadeIn(directed_graph)
		)

		implication = TextMobject(r"Edges are unidirectional")
		implication.scale(0.8)
		implication.next_to(directed_graph, DOWN)
		self.play(
			Write(implication)
		)

		self.wait(4)

		directed_cyclic_graph = VGroup(entire_graph, directed_graph)
		directed_cyclic_graph_shifted = directed_cyclic_graph.copy()
		directed_cyclic_graph_shifted.shift(LEFT * 7)

		self.play(
			FadeOut(graph_with_label),
			FadeOut(implication),
			Transform(directed_cyclic_graph, directed_cyclic_graph_shifted),
			run_time=2
		)

		self.wait()

		directed_graph_specific = TextMobject("Directed (Cyclic) Graph")
		directed_graph_specific.scale(0.9)
		directed_graph_specific.move_to(directed_graph.get_center())
		self.play(
			Transform(directed_graph, directed_graph_specific)
		)

		objects = self.show_path(graph, edge_dict, [0, 3, 2, 0], scale_factor=scale_factor, directed=True)

		self.wait()

		fadeouts = [FadeOut(obj) for obj in objects]
		self.play(
			*fadeouts
		)

		dag_graph, dag_edge_dict = self.create_small_dag()
		dag_nodes, dag_edges = self.make_graph_mobject(dag_graph, dag_edge_dict, directed=True)
		dag_entire_graph = VGroup(dag_nodes, dag_edges)
		scale_factor = 0.9
		dag_entire_graph.scale(scale_factor)
		dag_entire_graph.shift(UP * 1)
		dag_entire_graph.shift(RIGHT * 3.5)
		self.play(
			FadeIn(dag_entire_graph)
		)

		directed_acyclic_graph = TextMobject("Directed Acyclic Graph (DAG)")
		directed_acyclic_graph.scale(0.9)
		directed_acyclic_graph.next_to(dag_entire_graph, DOWN)
		self.play(
			FadeIn(directed_acyclic_graph)
		)

		self.wait(8)

		self.play(
			FadeOut(directed_cyclic_graph),
			FadeOut(directed_acyclic_graph),
			FadeOut(dag_entire_graph)
		)

	def show_weighted_graph(self):
		graph, edge_dict = self.create_initial_graph()
		nodes, edges = self.make_graph_mobject(graph, edge_dict)
		entire_graph = VGroup(nodes, edges)
		scale_factor = 1
		entire_graph.scale(scale_factor)
		entire_graph.shift(DOWN * 0.5)
		self.play(
			FadeIn(entire_graph)
		)
		self.wait()

		edge_weights = {}
		edge_weights_mobjects = {}
		for key in edge_dict:
			edge = edge_dict[key]
			length = edge.get_length()
			weight = (length * 10) // 3
			weight = int(weight)
			edge_weights[key] = weight
			weight_object = TextMobject(str(weight))
			weight_object.scale(0.6 * scale_factor)
			if key in [(3, 7), (1, 2), (1, 8), (2, 3), (4, 5), (0, 4)]:
				normal = edge.get_unit_normals()[1]
			else:
				normal = edge.get_unit_normals()[0]
			weight_object.move_to(edge.get_midpoint() + normal * SMALL_BUFF * 2.5)
			edge_weights_mobjects[key] = weight_object

		self.play(
			*[FadeIn(weight) for weight in edge_weights_mobjects.values()]
		)

		self.wait()

		weighted_graph = TextMobject("Weighted Graph")
		weighted_graph.scale(0.9)
		weighted_graph.next_to(entire_graph, DOWN)

		self.play(
			FadeIn(weighted_graph)
		)
		self.wait(10)

		fadeouts = [FadeOut(weight) for weight in edge_weights_mobjects.values()]
		self.play(
			*fadeouts + [FadeOut(weighted_graph), FadeOut(entire_graph)]
		)

		self.wait()

	def show_trees(self, title, h_line):
		tree_title = TextMobject("Trees")
		tree_title.next_to(h_line, DOWN)

		self.play(
			Write(tree_title)
		)
		self.wait(4)

		tree_1_graph, tree_1_edge_dict = self.make_tree_1()
		tree_1_nodes, tree_1_edges = self.make_graph_mobject(tree_1_graph, tree_1_edge_dict)
		tree_1_entire_graph = VGroup(tree_1_nodes, tree_1_edges)
		scale_factor = 0.8
		tree_1_entire_graph.scale(scale_factor)

		tree_2_graph, tree_2_edge_dict = self.make_tree_2()
		tree_2_nodes, tree_2_edges = self.make_graph_mobject(tree_2_graph, tree_2_edge_dict)
		tree_2_entire_graph = VGroup(tree_2_nodes, tree_2_edges)
		scale_factor = 0.8
		tree_2_entire_graph.scale(scale_factor)

		tree_3_graph, tree_3_edge_dict = self.make_tree_3()
		tree_3_nodes, tree_3_edges = self.make_graph_mobject(tree_3_graph, tree_3_edge_dict)
		tree_3_entire_graph = VGroup(tree_3_nodes, tree_3_edges)
		scale_factor = 0.8
		tree_3_entire_graph.scale(scale_factor)

		property_1 = TextMobject(r"1. Connected and acyclic")
		property_2 = TextMobject(r"2. Removing edge disconnects graph")
		property_3 = TextMobject(r"3. Adding edge creates a cycle")
		property_1.scale(0.8)
		property_2.scale(0.8)
		property_3.scale(0.8)

		property_1.next_to(tree_title, DOWN)
		property_1.to_edge(LEFT * 9)
		property_2.next_to(property_1, DOWN)
		property_2.to_edge(LEFT * 9)
		property_3.next_to(property_2, DOWN)
		property_3.to_edge(LEFT * 9)

		self.play(
			Write(property_1)
		)
		self.wait()

		self.play(
			Write(property_2)
		)
		self.wait()

		self.play(
			Write(property_3)
		)
		self.wait(2)

		self.play(
			FadeIn(tree_1_entire_graph),
			FadeIn(tree_2_entire_graph),
			FadeIn(tree_3_entire_graph)
		)
		self.wait(15)

		self.play(
			FadeOut(tree_1_entire_graph),
			FadeOut(tree_2_entire_graph),
			FadeOut(tree_3_entire_graph),
			FadeOut(tree_title),
			FadeOut(property_1),
			FadeOut(property_2),
			FadeOut(property_3),
			FadeOut(h_line),
			FadeOut(title)
		)

	def make_tree_1(self):
		graph = []
		edges = {}

		radius, scale = 0.4, 0.9
		node_0 = GraphNode('0', position=UP * 1 + LEFT * 5, radius=radius, scale=scale)
		node_1 = GraphNode('1', position=DOWN * 0.5 + LEFT * 5, radius=radius, scale=scale)
		node_2 = GraphNode('2', position=DOWN * 2 + LEFT * 5, radius=radius, scale=scale)
		node_3 = GraphNode('3', position=DOWN * 3.5 + LEFT * 5, radius=radius, scale=scale)

		edges[(0, 1)] = node_0.connect(node_1)

		edges[(1, 2)] = node_1.connect(node_2)

		edges[(2, 3)] = node_2.connect(node_3)

		graph.append(node_0)
		graph.append(node_1)
		graph.append(node_2)
		graph.append(node_3)

		return graph, edges

	def make_tree_2(self):
		graph = []
		edges = {}

		radius, scale = 0.4, 0.9
		node_0 = GraphNode('0', position=ORIGIN, radius=radius, scale=scale)
		node_1 = GraphNode('1', position=DOWN * 1.5 + LEFT * 2, radius=radius, scale=scale)
		node_2 = GraphNode('2', position=DOWN * 1.5 + RIGHT * 2, radius=radius, scale=scale)
		node_3 = GraphNode('3', position=DOWN * 3 + LEFT * 3, radius=radius, scale=scale)
		node_4 = GraphNode('4', position=DOWN * 3 + LEFT * 1, radius=radius, scale=scale)
		node_5 = GraphNode('5', position=DOWN * 3 + RIGHT * 1, radius=radius, scale=scale)
		node_6 = GraphNode('6', position=DOWN * 3 + RIGHT * 3, radius=radius, scale=scale)

		edges[(0, 1)] = node_0.connect(node_1)

		edges[(0, 2)] = node_0.connect(node_2)

		edges[(1, 3)] = node_1.connect(node_3)
		edges[(1, 4)] = node_1.connect(node_4)

		edges[(2, 5)] = node_2.connect(node_5)
		edges[(2, 6)] = node_2.connect(node_6)

		graph.append(node_0)
		graph.append(node_1)
		graph.append(node_2)
		graph.append(node_3)
		graph.append(node_4)
		graph.append(node_5)
		graph.append(node_6)

		return graph, edges

	def make_tree_3(self):
		graph = []
		edges = {}

		radius, scale = 0.4, 0.9
		node_0 = GraphNode('0', position=DOWN * 1.5 + RIGHT * 5, radius=radius, scale=scale)
		node_1 = GraphNode('1', position=DOWN * 0 + RIGHT * 4, radius=radius, scale=scale)
		node_2 = GraphNode('2', position=DOWN * 0 + RIGHT * 6, radius=radius, scale=scale)
		node_3 = GraphNode('3', position=DOWN * 3 + RIGHT * 4, radius=radius, scale=scale)
		node_4 = GraphNode('4', position=DOWN * 3 + RIGHT * 6, radius=radius, scale=scale)

		edges[(0, 1)] = node_0.connect(node_1)
		edges[(0, 2)] = node_0.connect(node_2)
		edges[(0, 3)] = node_0.connect(node_3)
		edges[(0, 4)] = node_0.connect(node_4)

		graph.append(node_0)
		graph.append(node_1)
		graph.append(node_2)
		graph.append(node_3)
		graph.append(node_4)

		return graph, edges

	def show_path(self, graph, edge_dict, path, scale_factor=1, directed=False):
		angle = 180
		objects = []
		for i in range(len(path) - 1):
			u, v = path[i], path[i + 1]
			surround_circle = self.highlight_node(graph, u, start_angle=angle/360 * TAU, scale_factor=scale_factor)
			last_edge = self.sharpie_edge(edge_dict, u, v, scale_factor=scale_factor, directed=directed)
			objects.extend([surround_circle, last_edge])
			angle = self.find_angle_of_intersection(graph, last_edge.get_end(), v)

		if v != path[0]:
			surround_circle = self.highlight_node(graph, v, start_angle=angle/360 * TAU, scale_factor=scale_factor)
			objects.append(surround_circle)

		return objects
	
	def sharpie_edge(self, edge_dict, u, v, color=GREEN_SCREEN, scale_factor=1, animate=True, directed=False):
		switch = False
		if u > v:
			edge = edge_dict[(v, u)]
			switch = True
		else:
			edge = edge_dict[(u, v)]
		
		if not switch:
			if not directed:
				line = Line(edge.get_start(), edge.get_end())
			else:
				line = Arrow(edge.get_start() - edge.unit_vector * edge.buff, 
					edge.get_end() + edge.unit_vector * edge.buff)
		else:
			if not directed:
				line = Line(edge.get_end(), edge.get_start())
			else:
				line = Arrow(edge.get_start() - edge.unit_vector * edge.buff, 
					edge.get_end() + edge.unit_vector * edge.buff)

		if not directed:
			line.set_stroke(width=16 * scale_factor)
		else:
			line.set_stroke(width=7 * scale_factor)
		line.set_color(color)
		if animate:
			self.play(
				ShowCreation(line)
			)
		return line

	def highlight_node(self, graph, index, color=GREEN_SCREEN, 
		start_angle=TAU/2, scale_factor=1, animate=True):
		node = graph[index]
		surround_circle = Circle(radius=node.circle.radius * scale_factor, TAU=-TAU, start_angle=start_angle)
		surround_circle.move_to(node.circle.get_center())
		# surround_circle.scale(1.15)
		surround_circle.set_stroke(width=8 * scale_factor)
		surround_circle.set_color(color)
		surround_circle.set_fill(opacity=0)
		if animate:
			self.play(
				ShowCreation(surround_circle)
			)
		return surround_circle

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
		edges[(0, 2)] = node_0.connect(node_2)
		edges[(0, 3)] = node_0.connect(node_3)

		edges[(1, 3)] = node_1.connect(node_3)

		edges[(2, 3)] = node_2.connect(node_3)

		edges[(3, 4)] = node_3.connect(node_4)

		graph.append(node_0)
		graph.append(node_1)
		graph.append(node_2)
		graph.append(node_3)
		graph.append(node_4)

		return graph, edges

	def create_small_dag(self):
		graph = []
		edges = {}

		radius, scale = 0.4, 0.9
		node_0 = GraphNode('0', position=DOWN * 1 + LEFT * 3, radius=radius, scale=scale)
		node_1 = GraphNode('1', position=UP * 1 + LEFT, radius=radius, scale=scale)
		node_2 = GraphNode('2', position=DOWN * 3 + LEFT, radius=radius, scale=scale)
		node_3 = GraphNode('3', position=DOWN * 1 + RIGHT, radius=radius, scale=scale)
		node_4 = GraphNode('4', position=DOWN * 1 + RIGHT * 3, radius=radius, scale=scale)


		edges[(0, 1)] = node_0.connect_arrow(node_1)
		edges[(0, 2)] = node_0.connect_arrow(node_2)
		edges[(0, 3)] = node_0.connect_arrow(node_3)

		edges[(1, 3)] = node_1.connect_arrow(node_3)

		edges[(2, 3)] = node_2.connect_arrow(node_3)

		edges[(3, 4)] = node_3.connect_arrow(node_4)

		graph.append(node_0)
		graph.append(node_1)
		graph.append(node_2)
		graph.append(node_3)
		graph.append(node_4)

		return graph, edges

	def create_small_directed_graph(self):
		graph = []
		edges = {}

		radius, scale = 0.4, 0.9
		node_0 = GraphNode('0', position=DOWN * 1 + LEFT * 3, radius=radius, scale=scale)
		node_1 = GraphNode('1', position=UP * 1 + LEFT, radius=radius, scale=scale)
		node_2 = GraphNode('2', position=DOWN * 3 + LEFT, radius=radius, scale=scale)
		node_3 = GraphNode('3', position=DOWN * 1 + RIGHT, radius=radius, scale=scale)
		node_4 = GraphNode('4', position=DOWN * 1 + RIGHT * 3, radius=radius, scale=scale)


		edges[(0, 1)] = node_0.connect_arrow(node_1)
		edges[(0, 2)] = node_2.connect_arrow(node_0)
		edges[(0, 3)] = node_0.connect_arrow(node_3)

		edges[(1, 3)] = node_1.connect_arrow(node_3)

		edges[(2, 3)] = node_3.connect_arrow(node_2)

		edges[(3, 4)] = node_3.connect_arrow(node_4)

		graph.append(node_0)
		graph.append(node_1)
		graph.append(node_2)
		graph.append(node_3)
		graph.append(node_4)

		return graph, edges

	def create_initial_graph(self):
		graph = []
		edges = {}

		radius, scale = 0.4, 0.9
		SHIFT = RIGHT * 2.5
		node_0 = GraphNode('0', position=LEFT * 2.5, radius=radius, scale=scale)
		node_1 = GraphNode('1', position=RIGHT * 3, radius=radius, scale=scale)
		node_2 = GraphNode('2', position=RIGHT * 1.5 + UP, radius=radius, scale=scale)
		node_3 = GraphNode('3', position=UP * 2.5 + SHIFT, radius=radius, scale=scale)
		node_4 = GraphNode('4', position=DOWN * 2, radius=radius, scale=scale)
		node_5 = GraphNode('5', position=DOWN + RIGHT * 2, radius=radius, scale=scale)
		node_6 = GraphNode('6', position=LEFT + UP, radius=radius, scale=scale)
		node_7 = GraphNode('7', position=LEFT  * 2 + UP * 2 + SHIFT, radius=radius, scale=scale)
		node_8 = GraphNode('8', position=ORIGIN, radius=radius, scale=scale)

		edge_0_6 = node_0.connect(node_6)
		edges[(0, 6)] = edge_0_6
		# edges[(6, 0)] = edge_0_6
		edge_0_8 = node_0.connect(node_8)
		edges[(0, 8)] = edge_0_8
		# edges[(8, 0)] = edge_0_8
		edge_0_4 = node_0.connect(node_4)
		edges[(0, 4)] = edge_0_4
		# edges[(4, 0)] = edge_0_4
		edge_1_2 = node_1.connect(node_2)
		edges[(1, 2)] = edge_1_2
		# edges[(2, 1)] = edge_1_2
		edge_1_5 = node_1.connect(node_5)
		edges[(1, 5)] = edge_1_5
		# edges[(5, 1)] = edge_1_5
		edge_2_3 = node_2.connect(node_3)
		edges[(2, 3)] = edge_2_3
		# edges[(3, 2)] = edge_2_3
		edge_3_7 = node_3.connect(node_7)
		edges[(3, 7)] = edge_3_7
		# edges[(7, 3)] = edge_3_7
		edge_4_5 =  node_4.connect(node_5)
		edges[(4, 5)] = edge_4_5
		# edges[(5, 4)] = edge_4_5
		edge_6_7 = node_6.connect(node_7)
		edges[(6, 7)] = edge_6_7
		# edges[(7, 6)] = edge_6_7
		edge_1_8 = node_1.connect(node_8)
		edges[(1, 8)] = edge_1_8
		# edges[(8, 1)] = edge_1_8

		graph.append(node_0)
		graph.append(node_1)
		graph.append(node_2)
		graph.append(node_3)
		graph.append(node_4)
		graph.append(node_5)
		graph.append(node_6)
		graph.append(node_7)
		graph.append(node_8)

		return graph, edges

	def make_graph_mobject(self, graph, edge_dict, node_color=DARK_BLUE_B, 
		stroke_color=BLUE, data_color=WHITE, edge_color=GRAY, 
		scale_factor=1, directed=False):
		nodes = []
		edges = []
		for node in graph:
			node.circle.set_fill(color=node_color, opacity=0.5)
			node.circle.set_stroke(color=stroke_color)
			node.data.set_color(color=data_color)
			nodes.append(VGroup(node.circle, node.data))

		for edge in edge_dict.values():
			if not directed:
				edge.set_stroke(width=7*scale_factor)
			else:
				edge.set_stroke(width=2*scale_factor)
			edge.set_color(color=edge_color)
			edges.append(edge)
		return VGroup(*nodes), VGroup(*edges)

class GraphRepresentations(Scene):
	def construct(self):
		title = TextMobject("Graph Representations")
		title.scale(1.2)
		title.shift(UP * 3.5)

		h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
		h_line.next_to(title, DOWN)
		self.play(
			Write(title),
			ShowCreation(h_line)
		)
		self.wait()

		question = TextMobject("How do we represent a graph as a data structure?")
		question.next_to(h_line, DOWN)
		self.play(
			Write(question)
		)

		self.wait(2)

		self.show_adjacency_matrix(h_line, question)

		self.wait()

		self.show_edge_set(h_line)

		self.show_adjacency_list(title, h_line)

		self.wait()
	
	def show_adjacency_matrix(self, h_line, question):
		adjacency_matrix = TextMobject("Adjacency Matrix")
		adjacency_matrix.next_to(h_line, DOWN)

		scale_factor = 0.9
		graph, edge_dict = self.create_small_graph()
		node_mobjects, edge_mobjects = self.make_graph_mobject(graph, edge_dict, scale_factor=scale_factor)
		entire_graph = VGroup(node_mobjects, edge_mobjects)
		entire_graph.scale(0.9)

		entire_graph.shift(LEFT * 3.5 + DOWN * 0.4)
		self.play(
			FadeIn(entire_graph)
		)

		self.wait(5)

		several_options = TextMobject("There are several options")
		several_options.next_to(h_line, DOWN)
		self.play(
			ReplacementTransform(question, several_options)
		)
		self.wait(6)

		grid = self.create_grid(5, 5, 0.7)
		grid_group = self.get_group_grid(grid)
		grid_group.shift(RIGHT * 2)

		labels = self.show_labels_matrix(graph, grid)

		self.play(
			ShowCreation(grid_group)
		)

		self.wait()

		self.play(
			ReplacementTransform(several_options, adjacency_matrix)
		)

		self.wait()

		A_ij = TextMobject(r"$A_{ij} = $")
		A_ij.next_to(adjacency_matrix, DOWN)
		A_ij.scale(0.9)
		A_ij.shift(LEFT * 1.4 + DOWN * 0.1)
		one_condition = TextMobject(r"1 for edge $(i, j)$")
		one_condition.scale(0.7)
		one_condition.next_to(adjacency_matrix, DOWN)
		one_condition.shift(RIGHT * 0.5)
		zero_condition = TextMobject("0 otherwise")
		zero_condition.scale(0.7)
		zero_condition.next_to(one_condition, DOWN)
		
		rhs = VGroup(one_condition, zero_condition)

		brace = Brace(rhs, LEFT, buff=SMALL_BUFF)

		right_group = VGroup(brace, rhs)
		right_group.next_to(A_ij, RIGHT)
		
		self.play(
			Write(A_ij)
		)
		self.play(
			GrowFromCenter(brace)
		)
		self.wait(2)
		self.play(
			Write(rhs[0])
		)
		self.wait(3)
		self.play(
			Write(rhs[1])
		)

		self.wait(2)

		data_objects = self.populate_matrix(edge_dict, grid)

		self.wait(2)

		highlight_edge = self.sharpie_edge(edge_dict, 0, 1)

		edge_square_1 = grid[0][1].copy()
		edge_square_1.set_color(GREEN_SCREEN)
		edge_square_1.set_fill(opacity=0.2)

		edge_square_2 = grid[1][0].copy()
		edge_square_2.set_color(GREEN_SCREEN)
		edge_square_2.set_fill(opacity=0.2)

		self.play(
			FadeIn(edge_square_1),
			FadeIn(edge_square_2)
		)

		self.wait(8)
		fadeouts = [FadeOut(obj) for obj in labels + data_objects]
		other_fadeouts = [
		FadeOut(edge_square_1), 
		FadeOut(edge_square_2),
		FadeOut(right_group),
		FadeOut(A_ij),
		FadeOut(adjacency_matrix),
		FadeOut(grid_group),
		FadeOut(highlight_edge),
		FadeOut(entire_graph)
		]

		self.play(
			*fadeouts + other_fadeouts
		)

	def show_edge_set(self, h_line):
		edge_set_name = TextMobject("Edge Set")
		edge_set_name.next_to(h_line, DOWN)

		scale_factor = 0.9
		graph, edge_dict = self.create_small_graph()
		node_mobjects, edge_mobjects = self.make_graph_mobject(graph, edge_dict, scale_factor=scale_factor)
		entire_graph = VGroup(node_mobjects, edge_mobjects)
		entire_graph.scale(0.9)

		entire_graph.shift(UP * 0.5)
		self.play(
			FadeIn(entire_graph)
		)
		self.wait()

		edge_set = TextMobject(r"$\{(0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (3, 4)\}$")
		edge_set.scale(0.9)
		edge_set.move_to(DOWN * 3.5)
		
		edge_lines = []
		for key in edge_dict:
			edge = self.sharpie_edge(edge_dict, key[0], key[1], animate=False)
			edge_lines.append(edge)

		self.play(
			*[ShowCreation(line) for line in edge_lines]
		)
		self.wait()

		self.play(
			TransformFromCopy(VGroup(*edge_lines), edge_set),
			run_time=2
		)
		self.wait(2)

		self.play(
			Write(edge_set_name)
		)

		self.wait(15)

		other_fadeouts = [FadeOut(edge_set), FadeOut(entire_graph), FadeOut(edge_set_name)]
		self.play(
			*[FadeOut(line) for line in edge_lines] + other_fadeouts
		)

	def show_adjacency_list(self, title, h_line):
		adjacency_list = TextMobject("Adjacency List")
		adjacency_list.next_to(h_line, DOWN)

		scale_factor = 0.9
		graph, edge_dict = self.create_small_graph()
		node_mobjects, edge_mobjects = self.make_graph_mobject(graph, edge_dict, scale_factor=scale_factor)
		entire_graph = VGroup(node_mobjects, edge_mobjects)
		entire_graph.scale(0.9)

		entire_graph.shift(LEFT * 3.5 + DOWN * 0)
		self.play(
			FadeIn(entire_graph)
		)

		self.play(
			Write(adjacency_list)
		)
		self.wait(4)

		array = self.create_grid(len(graph), 1, 0.8)
		array_group = self.get_group_grid(array)
		array_group.set_color(BLUE)
		array_group.set_fill(opacity=0.5)
		array_group.shift(RIGHT + UP * 0.8)
		
		labels = self.show_labels_array(graph, array)

		self.play(
			FadeIn(array_group)
		)

		list_info = self.populate_adj_list(graph, edge_dict, array, scale_factor=scale_factor)
		self.wait(5)

		sparse_graphs = TextMobject("Great for Sparse Graphs")
		sparse_graphs.scale(0.8)
		sparse_graphs.next_to(adjacency_list, DOWN)
		self.play(
			Write(sparse_graphs)
		)
		self.wait(5)

		example = TextMobject("Ex: Social Network")
		example.scale(0.8)
		example.move_to(sparse_graphs.get_center())
		self.play(
			Transform(sparse_graphs, example)
		)

		self.wait(17)

		our_rep = TextMobject("This will be our primary representation going forward")
		our_rep.scale(0.8)
		our_rep.move_to(sparse_graphs.get_center())
		self.play(
			Transform(sparse_graphs, our_rep)
		)
		self.wait(5)

		self.play(
			FadeOut(list_info),
			FadeOut(entire_graph),
			FadeOut(title),
			FadeOut(h_line),
			FadeOut(labels),
			FadeOut(array_group),
			FadeOut(adjacency_list),
			FadeOut(sparse_graphs)
		)

	def show_labels_array(self, graph, grid):
		original_data = [graph[i].data for i in range(len(graph))]
		col_data = []
		for i in range(len(graph)):
			orig_data_copy_col = graph[i].data.copy()
			orig_data_copy_col.set_color(BLUE)
			orig_data_copy_col.move_to(grid[i][0].get_center() + LEFT * grid[i][0].get_height())
			col_data.append(orig_data_copy_col)

		col_transforms = [TransformFromCopy(original_data[i], col_data[i]) for i in range(len(graph))]

		self.play(
			*col_transforms,
			run_time=3
		)

		return VGroup(*col_data)

	def populate_adj_list(self, graph, edge_dict, array, scale_factor=1):
		lists = []
		list_groups = []
		all_data = []
		arrows = []
		for i in range(len(graph)):
			if i < 3:
				animate = True
			else:
				animate = False
			neighbors = graph[i].neighbors
			data = []
			lst = self.create_grid(1, len(neighbors), 0.6)
			# print(len(lst), lst)
			lst_group = self.get_group_grid(lst)
			lst_group.next_to(array[i][0], RIGHT * 5)
			lists.append(lst)
			list_groups.append(lst_group)
			arrow = Arrow(
				array[i][0].get_center() + RIGHT * array[i][0].get_width() / 2,
				lst[0][0].get_center() + LEFT * lst[0][0].get_width() / 2
			)
			if animate:
				self.play(
					ShowCreation(arrow)
				)
			arrows.append(arrow)

			highlighted_objects = self.show_neighbors(graph, edge_dict, i, scale_factor=scale_factor, animate=animate)
			if animate:
				self.play(
					ShowCreation(lst_group)
				)
			for j, node in enumerate(neighbors):
				node_data_copy = node.data.copy()
				node_data_copy.move_to(lst[0][j].get_center())
				data.append(node_data_copy)

			all_data.extend(data)
			if animate:
				self.play(
					*[FadeIn(obj) for obj in data]
				)
			if animate:
				self.play(
					*[FadeOut(obj) for obj in highlighted_objects]
				)

			if not animate:
				all_anims = [FadeIn(obj) for obj in data] + [FadeIn(arrow), FadeIn(lst_group)]
				self.play(
					*all_anims
				)

		return VGroup(*list_groups + all_data + arrows)

	def show_neighbors(self, graph, edge_dict, node_index, scale_factor=1, animate=True):
		start_circle = self.highlight_node(graph, node_index, color=WHITE, scale_factor=scale_factor, animate=animate)
		relevant_edge_keys = []
		for key in edge_dict:
			if key[0] == node_index or key[1] == node_index:
				relevant_edge_keys.append(key)

		neighbor_edges = []
		next_node_indices = []
		for key in relevant_edge_keys:
			u = node_index
			if node_index == key[0]:
				v = key[1]
			else:
				v = key[0]

			edge = self.sharpie_edge(edge_dict, u, v, scale_factor=scale_factor, animate=False)
			neighbor_edges.append(edge)
			next_node_indices.append(v)
		if animate:
			self.play(
				*[ShowCreation(edge) for edge in neighbor_edges]
			)

		end_circles = []
		for neighbor_index in next_node_indices:
			end_circle = self.highlight_node(graph, neighbor_index, scale_factor=scale_factor, animate=False)
			end_circles.append(end_circle)

		if animate:
			self.play(
				*[ShowCreation(circle) for circle in end_circles]
			)

		return [start_circle] + neighbor_edges + end_circles



	def show_labels_matrix(self, graph, grid):
		original_data = [graph[i].data for i in range(len(graph))]
		row_data = []
		col_data = []
		for i in range(len(graph)):
			orig_data_copy_row = graph[i].data.copy()
			orig_data_copy_row.set_color(BLUE)
			orig_data_copy_row.move_to(grid[0][i].get_center() + UP * grid[0][i].get_height())
			row_data.append(orig_data_copy_row)

			orig_data_copy_col = graph[i].data.copy()
			orig_data_copy_col.set_color(BLUE)
			orig_data_copy_col.move_to(grid[i][0].get_center() + LEFT * grid[i][0].get_height())
			col_data.append(orig_data_copy_col)

		row_transforms = [TransformFromCopy(original_data[i], row_data[i]) for i in range(len(graph))]
		col_transforms = [TransformFromCopy(original_data[i], col_data[i]) for i in range(len(graph))]

		self.play(
			*row_transforms + col_transforms,
			run_time=3
		)

		return row_data + col_data

	def populate_matrix(self, edge_dict, grid):
		objects_1 = []
		objects_0 = []
		for i in range(len(grid)):
			for j in range(len(grid[0])):
				if (i, j) in edge_dict or (j, i) in edge_dict:
					one = TextMobject("1")
					one.scale(0.8)
					one.move_to(grid[i][j].get_center())
					second_one = one.copy()
					second_one.move_to(grid[j][i].get_center())
					objects_1.append(one)
					objects_1.append(second_one)
				else:
					zero = TextMobject("0")
					zero.scale(0.8)
					zero.move_to(grid[i][j].get_center())
					objects_0.append(zero)

		self.play(
			*[FadeIn(obj) for obj in objects_0 + objects_1]
		)

		return objects_0 + objects_1

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
		edges[(0, 2)] = node_0.connect(node_2)
		edges[(0, 3)] = node_0.connect(node_3)

		edges[(1, 3)] = node_1.connect(node_3)

		edges[(2, 3)] = node_2.connect(node_3)

		edges[(3, 4)] = node_3.connect(node_4)

		graph.append(node_0)
		graph.append(node_1)
		graph.append(node_2)
		graph.append(node_3)
		graph.append(node_4)

		return graph, edges

	def create_initial_graph(self):
		graph = []
		edges = {}

		radius, scale = 0.4, 0.9
		SHIFT = RIGHT * 2.5
		node_0 = GraphNode('0', position=LEFT * 2.5, radius=radius, scale=scale)
		node_1 = GraphNode('1', position=RIGHT * 3, radius=radius, scale=scale)
		node_2 = GraphNode('2', position=RIGHT * 1.5 + UP, radius=radius, scale=scale)
		node_3 = GraphNode('3', position=UP * 2.5 + SHIFT, radius=radius, scale=scale)
		node_4 = GraphNode('4', position=DOWN * 2, radius=radius, scale=scale)
		node_5 = GraphNode('5', position=DOWN + RIGHT * 2, radius=radius, scale=scale)
		node_6 = GraphNode('6', position=LEFT + UP, radius=radius, scale=scale)
		node_7 = GraphNode('7', position=LEFT  * 2 + UP * 2 + SHIFT, radius=radius, scale=scale)
		node_8 = GraphNode('8', position=ORIGIN, radius=radius, scale=scale)

		edge_0_6 = node_0.connect(node_6)
		edges[(0, 6)] = edge_0_6
		# edges[(6, 0)] = edge_0_6
		edge_0_8 = node_0.connect(node_8)
		edges[(0, 8)] = edge_0_8
		# edges[(8, 0)] = edge_0_8
		edge_0_4 = node_0.connect(node_4)
		edges[(0, 4)] = edge_0_4
		# edges[(4, 0)] = edge_0_4
		edge_1_2 = node_1.connect(node_2)
		edges[(1, 2)] = edge_1_2
		# edges[(2, 1)] = edge_1_2
		edge_1_5 = node_1.connect(node_5)
		edges[(1, 5)] = edge_1_5
		# edges[(5, 1)] = edge_1_5
		edge_2_3 = node_2.connect(node_3)
		edges[(2, 3)] = edge_2_3
		# edges[(3, 2)] = edge_2_3
		edge_3_7 = node_3.connect(node_7)
		edges[(3, 7)] = edge_3_7
		# edges[(7, 3)] = edge_3_7
		edge_4_5 =  node_4.connect(node_5)
		edges[(4, 5)] = edge_4_5
		# edges[(5, 4)] = edge_4_5
		edge_6_7 = node_6.connect(node_7)
		edges[(6, 7)] = edge_6_7
		# edges[(7, 6)] = edge_6_7
		edge_1_8 = node_1.connect(node_8)
		edges[(1, 8)] = edge_1_8
		# edges[(8, 1)] = edge_1_8

		graph.append(node_0)
		graph.append(node_1)
		graph.append(node_2)
		graph.append(node_3)
		graph.append(node_4)
		graph.append(node_5)
		graph.append(node_6)
		graph.append(node_7)
		graph.append(node_8)

		return graph, edges

	def make_graph_mobject(self, graph, edge_dict, node_color=DARK_BLUE_B, 
		stroke_color=BLUE, data_color=WHITE, edge_color=GRAY, scale_factor=1):
		nodes = []
		edges = []
		for node in graph:
			node.circle.set_fill(color=node_color, opacity=0.5)
			node.circle.set_stroke(color=stroke_color)
			node.data.set_color(color=data_color)
			nodes.append(VGroup(node.circle, node.data))

		for edge in edge_dict.values():
			edge.set_stroke(width=7*scale_factor)
			edge.set_color(color=edge_color)
			edges.append(edge)
		return VGroup(*nodes), VGroup(*edges)

	def create_grid(self, rows, columns, square_length):
		left_corner = Square(side_length=square_length)
		grid = []
		first_row = [left_corner]
		for i in range(columns - 1):
			square = Square(side_length=square_length)
			square.next_to(first_row[i], RIGHT, buff=0)
			first_row.append(square)
		grid.append(first_row)
		for i in range(rows - 1):
			prev_row = grid[i]
			new_row = []
			for square in prev_row:
				square_below = Square(side_length=square_length)
				square_below.next_to(square, DOWN, buff=0)
				new_row.append(square_below)
			grid.append(new_row)

		return grid

	def get_group_grid(self, grid):
		squares = []
		for row in grid:
			for square in row:
				squares.append(square)
		return VGroup(*squares)

	def sharpie_edge(self, edge_dict, u, v, color=GREEN_SCREEN, scale_factor=1, animate=True):
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
		line.set_stroke(width=16 * scale_factor)
		line.set_color(color)
		if animate:
			self.play(
				ShowCreation(line)
			)
		return line

	def highlight_node(self, graph, index, color=GREEN_SCREEN, 
		start_angle=TAU/2, scale_factor=1, animate=True):
		node = graph[index]
		surround_circle = Circle(radius=node.circle.radius * scale_factor, TAU=-TAU, start_angle=start_angle)
		surround_circle.move_to(node.circle.get_center())
		# surround_circle.scale(1.15)
		surround_circle.set_stroke(width=8 * scale_factor)
		surround_circle.set_color(color)
		surround_circle.set_fill(opacity=0)
		if animate:
			self.play(
				ShowCreation(surround_circle)
			)
		return surround_circle




class GraphApplication(Scene):
	def construct(self):
		title = TextMobject("Why Study Graphs?")
		title.scale(1.2)
		title.move_to(UP * 3.5)
		h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
		h_line.next_to(title, DOWN)
		self.play(
			Write(title),
			ShowCreation(h_line)
		)
		self.wait(4)

		show_up = TextMobject("Graphs show up all the time!")
		show_up.move_to(DOWN * 3)
		self.play(
			Write(show_up)
		)
		self.wait(5)
		social_networks = self.show_map_application(show_up)

		sudoku_example = self.show_social_media_app(social_networks)

		self.wait()

		

	def show_map_application(self, text):

		example = TextMobject("Example: mapping applications")
		example.scale(0.8)
		example.next_to(text, DOWN)

		top_street = Rectangle(height=0.2, width=7)
		top_street.move_to(LEFT * 4 + UP * 2)
		top_street.set_fill(GRAY, opacity=1)
		top_street.set_color(GRAY)

		middle_street_horiz = Rectangle(height=0.2, width=7)
		middle_street_horiz.move_to(LEFT * 4)
		middle_street_horiz.set_fill(GRAY, opacity=1)
		middle_street_horiz.set_color(GRAY)

		bottom_street = Rectangle(height=0.2, width=7)
		bottom_street.move_to(LEFT * 4 + DOWN * 2)
		bottom_street.set_fill(GRAY, opacity=1)
		bottom_street.set_color(GRAY)

		slanted_road = Rectangle(height=0.2, width=8)
		slanted_road.rotate(-TAU / 4.5)
		slanted_road.shift(LEFT * 6)
		slanted_road.set_fill(GRAY, opacity=1)
		slanted_road.set_color(GRAY)

		vertical_road_left = Rectangle(height=8, width=0.2)
		vertical_road_left.shift(LEFT * 4)
		vertical_road_left.set_fill(GRAY, opacity=1)
		vertical_road_left.set_color(GRAY)

		vertical_road_right = Rectangle(height=8, width=0.2)
		vertical_road_right.shift(LEFT * 2)
		vertical_road_right.set_fill(GRAY, opacity=1)
		vertical_road_right.set_color(GRAY)

		road_network = VGroup(
			top_street,
			middle_street_horiz,
			bottom_street,
			slanted_road,
			vertical_road_left,
			vertical_road_right
		)
		road_network.scale(0.6)
		road_network.shift(RIGHT)

		self.play(
			Write(example)
		)
		self.wait(3)

		self.play(
			FadeIn(road_network),
		)
		self.wait(2)

		points = self.find_intersection_points(road_network)
		graph, edge_dict = self.make_equivalent_graph()
		nodes, edges = self.make_graph_mobject(graph, edge_dict)
		
		dots = []
		for point in points:
			dot = Circle(radius=0.2)
			dot.set_color(BLUE)
			dot.set_fill(DARK_BLUE_B, opacity=1)
			dot.move_to(point)
			dots.append(dot)

		self.play(
			*[GrowFromCenter(dot) for dot in dots]
		)

		self.wait()

		self.play(
			*[TransformFromCopy(dot, node) for dot, node in zip(dots, nodes)],
			run_time=4
		)

		self.play(
			*[ShowCreation(edge) for edge in edges]
		)

		self.wait(3)


		start_dot = dots[0].copy()
		start_dot.set_stroke(color=BRIGHT_RED)

		self.play(
			Transform(dots[0], start_dot)
		)

		end_dot = dots[8].copy()
		end_dot.set_stroke(color=GREEN_SCREEN)
		self.play(
			Transform(dots[8], end_dot)
		)

		start_surround = self.highlight_node(graph, 0, color=BRIGHT_RED)
		end_surround = self.highlight_node(graph, 8, color=GREEN_SCREEN)

		path = [(0, 3), (3, 6), (6, 7), (7, 8)]
		highlight_objects = [start_surround, end_surround]
		for u, v in path:
			road_edge = self.highlight_road_edge(dots, u, v)
			graph_edge = self.sharpie_edge(edge_dict, u, v, animate=False)
			self.play(
				ShowCreation(road_edge),
				ShowCreation(graph_edge)
			)
			highlight_objects.append(road_edge)
			highlight_objects.append(graph_edge)

		self.wait(3)
		fadeouts = [FadeOut(dot) for dot in dots] + [FadeOut(road_network), FadeOut(nodes), FadeOut(edges)]
		self.play(
			*[FadeOut(obj) for obj in highlight_objects]
		)
		self.play(
			*fadeouts
		)

		next_example = TextMobject("Example: Social Network")
		next_example.scale(0.8)
		next_example.move_to(example.get_center())
		self.play(
			ReplacementTransform(example, next_example)
		)
		return next_example

	def show_social_media_app(self, text):
		graph, edge_dict = self.make_social_media_graph()
		nodes, edges = self.make_graph_mobject(graph, edge_dict)
		self.play(
			GrowFromCenter(nodes[0])
		)
		red_highlight = self.highlight_node(graph, 0, color=BRIGHT_RED, animate=False)
		self.wait()
		first_edges = [(0, 1), (0, 2), (0, 3), (0, 4)]
		self.play(
			*[ShowCreation(edge_dict[edge]) for edge in first_edges]
		)
		self.play(
			*[GrowFromCenter(nodes[i]) for i in range(1, 5)]
		)
		self.wait(9)
		self.play(
			ShowCreation(red_highlight)
		)
		self.wait(5)

		second_edges = [(1, 5), (1, 6), (2, 11), (2, 12), (3, 7), (3, 8), (4, 9), (4, 10)]
		self.play(
			*[ShowCreation(edge_dict[edge]) for edge in second_edges]
		)
		self.play(
			*[GrowFromCenter(nodes[i]) for i in range(5, len(nodes))]
		)
		self.wait()

		highlight_circles = []
		for i in range(5, len(nodes)):
			surround_circle = self.highlight_node(graph, i, animate=False)
			highlight_circles.append(surround_circle)

		self.play(
			*[ShowCreation(circle) for circle in highlight_circles]
		)
		highlight_circles.append(red_highlight)
		self.wait(9)

		self.play(
			*[FadeOut(circle) for circle in highlight_circles]
		)
		self.play(
			FadeOut(nodes),
			FadeOut(edges)
		)

		next_example = TextMobject("Example: Sudoku")
		next_example.scale(0.8)
		next_example.move_to(text.get_center())
		return next_example

	def show_sudoku_example(self):
		grid = self.create_grid(9, 9, 0.5)
		grid_group = self.get_group_grid(grid)
		white_3x3_grid = self.create_grid(3, 3, 1.5)
		white_3x3_grid_group = self.get_group_grid(white_3x3_grid)

		grid_group.set_color(GRAY)
		grid_group.move_to(ORIGIN)
		white_3x3_grid_group.set_color(WHITE)
		white_3x3_grid_group.move_to(ORIGIN)
		self.play(
			FadeIn(grid_group),
			FadeIn(white_3x3_grid_group)
		)

		example_quiz = self.get_sudoku_quiz()
		number_objects = self.display_numbers_on_grid(example_quiz, grid)
		number_objects_group = []
		for row in number_objects:
			for obj in row:
				if obj != 0:
					number_objects_group.append(obj)
		number_objects_group = VGroup(*number_objects_group)
		self.wait(16)

		self.explain_rules()

		shifted_grid = grid_group.copy()
		shifted_3x3 = white_3x3_grid_group.copy()
		shifted_numbers = number_objects_group.copy()

		shifted_grid.shift(LEFT * 3)
		shifted_numbers.shift(LEFT * 3)
		shifted_3x3.shift(LEFT * 3)
		self.play(
			Transform(grid_group, shifted_grid),
			Transform(number_objects_group, shifted_numbers),
			Transform(white_3x3_grid_group, shifted_3x3)
		)
		self.wait()

		self.play(
			Indicate(VGroup(*grid_group[3:6][3:6]))
		)

		sub_grid_3x3 = self.create_grid(3, 3, 0.5)
		sub_grid_3x3_group = self.get_group_grid(sub_grid_3x3)
		sub_grid_3x3_group.move_to(grid_group.get_center())

		for row in sub_grid_3x3:
			for square in row:
				square.set_stroke(color=SKY_BLUE)

		color_map = {
		0: SKY_BLUE, 
		1: PINK, 
		2: BRIGHT_RED, 
		3: TEAL_E, 
		4: GOLD,
		5: ORANGE,
		6: YELLOW,
		7: WHITE,
		8: GREEN_SCREEN,
		9: BLUE
		}

		inverted_map = {}
		for key, val in color_map.items():
			inverted_map[val] = key

		self.wait(6)

		self.map_numbers_to_color(number_objects, color_map)
		self.wait(3)
		self.play(
			FadeIn(sub_grid_3x3_group)
		)
		self.wait(2)

		graph, edge_dict, data = self.make_graph_from_grid(example_quiz, [(3, 3), (6, 6)])
		nodes, edges = self.make_3x3_graph_mobject(graph, edge_dict, color_map, data, show_data=False)
		entire_graph = VGroup(nodes, edges)
		entire_graph.shift(RIGHT * 3)
		self.play(
			TransformFromCopy(sub_grid_3x3_group, entire_graph),
			run_time=2
		)

		self.wait(7)


		graph_1_1 = entire_graph.copy()
		graph_1_1.scale(0.33)


		self.play(
			ReplacementTransform(entire_graph, graph_1_1)
		)

		self.wait()

		other_grids = []
		for i in range(8):
			blue_grid = self.create_grid(3, 3, 0.5)
			blue_grid_group = self.get_group_grid(blue_grid)
			blue_grid_group.move_to(grid_group.get_center())

			for row in blue_grid:
				for square in row:
					square.set_stroke(color=SKY_BLUE)
			other_grids.append(blue_grid_group)

		white_grid = self.create_grid(1, 1, 1.5)
		white_grid_group = self.get_group_grid(white_grid)
		white_grid_group.move_to(grid_group.get_center())

		other_grids[0].shift(LEFT * 1.5 + UP * 1.5)
		other_grids[1].shift(UP * 1.5)
		other_grids[2].shift(RIGHT * 1.5 + UP * 1.5)
		other_grids[3].shift(LEFT * 1.5)
		other_grids[4].shift(RIGHT * 1.5)
		other_grids[5].shift(LEFT * 1.5 + DOWN * 1.5)
		other_grids[6].shift(DOWN * 1.5)
		other_grids[7].shift(RIGHT * 1.5 + DOWN * 1.5)

		other_grids.append(white_grid_group)

		other_grid_entire_group = VGroup(*other_grids)
		self.play(
			FadeOut(sub_grid_3x3_group)
		)
		self.play(
			FadeIn(other_grid_entire_group),
			run_time=2
		)

		all_graphs, all_graphs_mobjects = self.make_remaining_graphs(example_quiz, color_map)
		all_graphs[4], all_graphs_mobjects[4] = graph, entire_graph

		all_graphs_mobjects[0].shift(UP * 1.8 + LEFT * 1.8)
		all_graphs_mobjects[1].shift(UP * 1.8)
		all_graphs_mobjects[2].shift(UP * 1.8 + RIGHT * 1.8)
		all_graphs_mobjects[3].shift(LEFT * 1.8)
		all_graphs_mobjects[5].shift(RIGHT * 1.8)
		all_graphs_mobjects[6].shift(DOWN * 1.8 + LEFT * 1.8)
		all_graphs_mobjects[7].shift(DOWN * 1.8)
		all_graphs_mobjects[8].shift(DOWN * 1.8 + RIGHT * 1.8)

		remaining_graph_mobject = VGroup(all_graphs_mobjects[:4], all_graphs_mobjects[5:])
		remaining_grid = VGroup(other_grid_entire_group[:8])
		self.play(
			FadeIn(remaining_graph_mobject),
			run_time=2
		)

		self.wait()
		self.play(
			FadeOut(other_grid_entire_group)
		)

		self.wait(5)
		self.make_row_and_column_connections(all_graphs_mobjects)
		self.wait(10)

		example_solution = self.get_sudoku_solution()

		self.simulate_9x9(all_graphs_mobjects, grid, example_quiz, example_solution, color_map)

		self.wait(5)
		
		# self.display_numbers_on_grid(example_solution, grid)
		# self.wait()

	def simulate_9x9(self, all_graphs_mobjects, grid, number_objects, solution, color_map):
		quadrants = self.get_quadrants_list()
		all_transforms = []
		for i in range(9):
			transforms = self.simulate_3x3(all_graphs_mobjects[i][0], grid, number_objects, quadrants[i], solution, color_map)
			all_transforms.append(transforms)

		for i in range(len(all_transforms[0])):
			step_transforms = sum([transforms[i] for transforms in all_transforms], [])
			self.play(
				*step_transforms,
				run_time=0.1
			)
		self.wait()

	def simulate_3x3(self, nodes, grid, number_objects, quadrant, solution, color_map):
		relevant_data = self.get_relevant_data(number_objects, quadrant)
		solution = self.get_relevant_data(solution, quadrant)
		relevant_positions = self.get_relevant_positions(grid, quadrant)
		positions = []
		present_numbers = []
		graph_indices = []
		solution_map = {}
		for i in range(len(relevant_data)):
			for j in range(len(relevant_data[0])):
				if relevant_data[i][j] == 0:
					positions.append(relevant_positions[i][j])
					graph_indices.append(j * len(relevant_data) + i)
					solution_map[(relevant_positions[i][j][0], relevant_positions[i][j][1])] = solution[i][j]
				else:
					present_numbers.append(relevant_data[i][j])
		missing_numbers = set(list(range(1, 10))) - set(present_numbers)
		missing_numbers = list(missing_numbers)
		missing_objects = []
		missing_graph_objects = []
		num_to_text = {}
		
		for i, num in enumerate(missing_numbers):
			text = TextMobject(str(num))
			text.scale(0.6)
			text.move_to(positions[i])
			text.set_color(color_map[num])
			missing_objects.append((text, num))
			num_to_text[num] = text.copy()
			index = graph_indices[i]
			new_node = nodes[index].copy()
			new_node.set_color(color_map[num])
			missing_graph_objects.append(new_node)

		replacements = [ReplacementTransform(nodes[graph_indices[i]], 
			missing_graph_objects[i]) for i in range(len(missing_graph_objects))]
		transforms = []
		transforms.append([FadeIn(obj[0]) for obj in missing_objects] + replacements)
		
		iters = 50
		for _ in range(iters):
			new_missing_objects = [(obj[0].copy(), obj[1]) for obj in missing_objects]
			new_missing_graph_objects = [obj.copy() for obj in missing_graph_objects]
			random.shuffle(new_missing_objects)
			for i in range(len(new_missing_objects)):
				new_missing_objects[i][0].move_to(positions[i])
				num = new_missing_objects[i][1]
				new_missing_graph_objects[i].set_color(color_map[num])

			transform = [ReplacementTransform(missing_objects[i][0], 
				new_missing_objects[i][0]) for i in range(len(missing_objects))]
			graph_transforms = [ReplacementTransform(missing_graph_objects[i], 
				new_missing_graph_objects[i]) for i in range(len(missing_graph_objects))]
			transforms.append(transform + graph_transforms)
			for i in range(len(missing_objects)):
				missing_objects[i] = new_missing_objects[i]
				missing_graph_objects[i] = new_missing_graph_objects[i]

		solution_transforms = []
		solution_object_sequence = []
		solution_object_colors = [obj.copy() for obj in missing_graph_objects]
		for i in range(len(missing_objects)):
			num = solution_map[(positions[i][0], positions[i][1])]
			text = num_to_text[num].copy()
			text.move_to(positions[i])
			solution_object_sequence.append((text, num))
			solution_object_colors[i].set_color(color_map[num])

		# self.play(*[FadeOut(missing_objects[i][0]) for i in range(len(missing_objects))])
		# self.wait()
		# self.play(*[FadeIn(solution_object_sequence[i][0]) for i in range(len(missing_objects))])

		solution_transforms = [ReplacementTransform(missing_objects[i][0], 
			solution_object_sequence[i][0]) for i in range(len(missing_objects))]
		solution_graph_transforms = [ReplacementTransform(missing_graph_objects[i], 
			solution_object_colors[i]) for i in range(len(missing_graph_objects))]

		transforms.append(solution_transforms + solution_graph_transforms)
		return transforms

	def make_row_and_column_connections(self, all_graphs_mobjects, scale_factor=0.33):
		all_graphs_nodes = [obj[0] for obj in all_graphs_mobjects]
		row_edges = []
		row_edges.extend(self.connect_row(all_graphs_nodes[0], all_graphs_nodes[1]))
		row_edges.extend(self.connect_row(all_graphs_nodes[1], all_graphs_nodes[2]))
		row_edges.extend(self.connect_row(all_graphs_nodes[3], all_graphs_nodes[4]))
		row_edges.extend(self.connect_row(all_graphs_nodes[4], all_graphs_nodes[5]))
		row_edges.extend(self.connect_row(all_graphs_nodes[6], all_graphs_nodes[7]))
		row_edges.extend(self.connect_row(all_graphs_nodes[7], all_graphs_nodes[8]))

		self.play(
			*[ShowCreation(edge) for edge in row_edges]
		)

		column_edges = []
		column_edges.extend(self.connect_column(all_graphs_nodes[0], all_graphs_nodes[3]))
		column_edges.extend(self.connect_column(all_graphs_nodes[1], all_graphs_nodes[4]))
		column_edges.extend(self.connect_column(all_graphs_nodes[2], all_graphs_nodes[5]))
		column_edges.extend(self.connect_column(all_graphs_nodes[3], all_graphs_nodes[6]))
		column_edges.extend(self.connect_column(all_graphs_nodes[4], all_graphs_nodes[7]))
		column_edges.extend(self.connect_column(all_graphs_nodes[5], all_graphs_nodes[8]))

		self.wait(6)
		self.play(
			*[ShowCreation(edge) for edge in column_edges]
		)

	def connect_row(self, graph1, graph2, scale_factor=0.33, edge_color=GRAY):
		row_edges = []
		nodes_graph_1 = [graph1[i] for i in range(6, 9)]
		nodes_graph_2 = [graph2[i] for i in range(3)]
		for node1, node2 in zip(nodes_graph_1, nodes_graph_2):
			start_point = node1.get_center() + RIGHT * 0.4 * scale_factor
			end_point = node2.get_center() + LEFT * 0.4 * scale_factor
			edge = Line(start_point, end_point)
			edge.set_stroke(width=7)
			edge.set_color(color=edge_color)
			row_edges.append(edge)
		return row_edges

	def connect_column(self, graph1, graph2, scale_factor=0.33, edge_color=GRAY):
		column_edges = []
		nodes_graph_1 = [graph1[i] for i in range(2, 9, 3)]
		nodes_graph_2 = [graph2[i] for i in range(0, 9, 3)]
		for node1, node2 in zip(nodes_graph_1, nodes_graph_2):
			start_point = node1.get_center() + DOWN * 0.4 * scale_factor
			end_point = node2.get_center() + UP * 0.4 * scale_factor
			edge = Line(start_point, end_point)
			edge.set_stroke(width=7)
			edge.set_color(color=edge_color)
			column_edges.append(edge)
		return column_edges

	def get_quadrants_list(self):
		quadrants = []
		for i in range(3):
			for j in range(3):
				quadrants.append([(i * 3, j * 3), (i * 3 + 3, j * 3 + 3)])
		return quadrants

	def make_remaining_graphs(self, example_quiz, color_map):
		all_graphs = [0] * 9
		all_graphs_mobjects = [0] * 9
		
		quadrants = self.get_quadrants_list()
		# print(quadrants)
		for i, quadrant in enumerate(quadrants):
			if i == 4:
				continue
			graph, edge_dict, data = self.make_graph_from_grid(example_quiz, quadrant)
			nodes, edges = self.make_3x3_graph_mobject(graph, edge_dict, color_map, data, show_data=False)
			entire_graph = VGroup(nodes, edges)
			entire_graph.shift(RIGHT * 3)
			entire_graph.scale(0.33)
			all_graphs[i] = graph
			all_graphs_mobjects[i] = entire_graph
		return all_graphs, all_graphs_mobjects

	def map_numbers_to_color(self, number_objects, color_map):
		for row in number_objects:
			for obj in row:
				if obj != 0:
					obj.set_color(color_map[int(obj.tex_string)])
	
	def get_relevant_positions(self, grid, quadrant):
		relevant_data = [[0] * 3 for _ in range(3)]
		# print(quadrant)
		for i in range(quadrant[0][0], quadrant[1][0]):
			for j in range(quadrant[0][1], quadrant[1][1]):
				relevant_data[i - quadrant[0][0]][j - quadrant[0][1]] = grid[i][j].get_center()
		return relevant_data

	def get_relevant_data(self, number_objects, quadrant):
		relevant_data = [[0] * 3 for _ in range(3)]
		# print(quadrant)
		for i in range(quadrant[0][0], quadrant[1][0]):
			for j in range(quadrant[0][1], quadrant[1][1]):
				relevant_data[i - quadrant[0][0]][j - quadrant[0][1]] = number_objects[i][j]
		return relevant_data

	def make_graph_from_grid(self, number_objects, quadrant):
		# quadrant = [(i, j), (k, l)]
		relevant_data = self.get_relevant_data(number_objects, quadrant)
		return self.make_3x3_graph(relevant_data)

	def make_3x3_graph(self, data):
		radius, scale = 0.4, 0.8
		graph = [[0] * 3 for _ in range(3)]
		edge_dict = {}
		for i in range(3):
			for j in range(3):
				entry = data[j][i]
				graph[i][j] = GraphNode(entry, position=RIGHT * (i - 1) * 2 + DOWN * (j - 1) * 2, radius=radius, scale=scale)

		for i in range(2):
			edge_dict[((i, 0), (i + 1, 0))]	= graph[i][0].connect(graph[i + 1][0])
			edge_dict[((i, 1), (i + 1, 1))] = graph[i][1].connect(graph[i + 1][1])
			edge_dict[((i, 2), (i + 1, 2))] = graph[i][2].connect(graph[i + 1][2])

		for i in range(2):
			edge_dict[((0, i), (0, i + 1))]	= graph[0][i].connect(graph[0][i + 1])
			edge_dict[((1, i), (1, i + 1))] = graph[1][i].connect(graph[1][i + 1])
			edge_dict[((2, i), (2, i + 1))] = graph[2][i].connect(graph[2][i + 1])

		for i in range(2):
			for j in range(2):
				edge_dict[((i, j), (i + 1, j + 1))]	= graph[i][j].connect(graph[i + 1][j + 1])
		
		for i in range(1, 3):
			for j in range(2):
				edge_dict[((i, j), (i - 1, j + 1))]	= graph[i][j].connect(graph[i - 1][j + 1])
		
		edge_dict[((0, 1), (2, 0))]	= graph[0][1].connect(graph[2][0])
		edge_dict[((0, 1), (2, 2))] = graph[0][1].connect(graph[2][2])
		edge_dict[((0, 0), (2, 1))] = graph[0][0].connect(graph[2][1])
		edge_dict[((0, 2), (2, 1))] = graph[0][2].connect(graph[2][1])
		edge_dict[((0, 0), (1, 2))] = graph[0][0].connect(graph[1][2])
		edge_dict[((2, 0), (1, 2))] = graph[2][0].connect(graph[1][2])
		edge_dict[((1, 0), (0, 2))]	= graph[1][0].connect(graph[2][0])
		edge_dict[((1, 0), (2, 2))]	= graph[1][0].connect(graph[2][2])
		# edge_dict[((1, 0), (2, 1))] = graph[1][i].connect(graph[1][i + 1])
		# edge_dict[((2, i), (2, i + 1))] = graph[2][i].connect(graph[2][i + 1])

		return graph, edge_dict, data

	def make_3x3_graph_mobject(self, graph, edge_dict, color_map, data, scale_factor=1, edge_color=GRAY, show_data=True):
		nodes = []
		edges = []
		for i in range(len(graph)):
			for j in range(len(graph[0])):
				node = graph[i][j]
				num = data[j][i]
				color = color_map[num]
				node.circle.set_fill(color=color, opacity=0.5)
				node.circle.set_stroke(color=color)
				if num == 0 or not show_data:
					nodes.append(node.circle)
				else:
					node.data.set_color(BLACK)
					nodes.append(VGroup(node.circle, node.data))

		for edge in edge_dict.values():
			edge.set_stroke(width=7*scale_factor)
			edge.set_color(color=edge_color)
			edges.append(edge)
		return VGroup(*nodes), VGroup(*edges)

	def get_sudoku_quiz(self):
		quiz_string = '005960070400031020106705008900408060002013800058000000200096007007020615003000004'
		return self.parse_string_to_2d_list(quiz_string)

	def get_sudoku_solution(self):
		solution_string = '385962471479831526126745938931458762742613859658279143214596387897324615563187294'
		return self.parse_string_to_2d_list(solution_string)

	def parse_string_to_2d_list(self, string):
		quiz = []
		for i in range(9):
			row = string[i * 9: (i + 1) * 9]
			quiz.append([int(c) for c in row])
		return quiz

	def display_numbers_on_grid(self, sudoku_list, grid):
		number_objects = [[0] * 9 for _ in range(9)]
		for i in range(9):
			for j in range(9):
				num = sudoku_list[i][j]
				if num != 0:
					text = TextMobject(str(num))
					text.scale(0.6)
					position = grid[i][j].get_center()
					text.move_to(position)
					# print(i, j, 'Write')
					number_objects[i][j] = text
		self.render_numbers(number_objects)
		return number_objects

	def render_numbers(self, number_objects):
		animations = []
		for row in number_objects:
			for obj in row:
				if obj != 0:
					animations.append(FadeIn(obj))
		self.play(
			*animations
		)

	def explain_rules(self, color=GREEN_SCREEN):
		sub_grid_3x3 = self.create_grid(3, 3, 0.5)
		sub_grid_3x3_group = self.get_group_grid(sub_grid_3x3)
		sub_grid_3x3_group.move_to(ORIGIN)

		for row in sub_grid_3x3:
			for square in row:
				square.set_stroke(color=color)
				square.set_fill(color=color, opacity=0.2)
		self.play(
			GrowFromCenter(sub_grid_3x3_group)
		)
		self.wait(3)

		sub_grid_1x9 = self.create_grid(1, 9, 0.5)
		sub_grid_1x9_group = self.get_group_grid(sub_grid_1x9)
		sub_grid_1x9_group.move_to(ORIGIN)
		for row in sub_grid_1x9:
			for square in row:
				square.set_stroke(color=color)
				square.set_fill(color=color, opacity=0.2)

		self.play(
			ReplacementTransform(sub_grid_3x3_group, sub_grid_1x9_group),
			run_time=2
		)
		self.wait(3)

		sub_grid_9x1 = self.create_grid(9, 1, 0.5)
		sub_grid_9x1_group = self.get_group_grid(sub_grid_9x1)
		sub_grid_9x1_group.move_to(ORIGIN)
		for row in sub_grid_9x1:
			for square in row:
				square.set_stroke(color=color)
				square.set_fill(color=color, opacity=0.2)

		self.play(
			ReplacementTransform(sub_grid_1x9_group, sub_grid_9x1_group),
			run_time=2
		)
		self.wait(3)

		self.play(
			FadeOut(sub_grid_9x1_group)
		)

	def create_grid(self, rows, columns, square_length):
		left_corner = Square(side_length=square_length)
		grid = []
		first_row = [left_corner]
		for i in range(columns - 1):
			square = Square(side_length=square_length)
			square.next_to(first_row[i], RIGHT, buff=0)
			first_row.append(square)
		grid.append(first_row)
		for i in range(rows - 1):
			prev_row = grid[i]
			new_row = []
			for square in prev_row:
				square_below = Square(side_length=square_length)
				square_below.next_to(square, DOWN, buff=0)
				new_row.append(square_below)
			grid.append(new_row)

		return grid

	def get_group_grid(self, grid):
		squares = []
		for row in grid:
			for square in row:
				squares.append(square)
		return VGroup(*squares)
		
	def highlight_road_edge(self, dots, u, v, color=GREEN_SCREEN):
		assert u < v
		center_u, center_v = dots[u].get_center(), dots[v].get_center()
		unit_vector = Line(center_u, center_v).get_unit_vector()
		start = center_u + unit_vector * dots[u].radius
		end = center_v - unit_vector * dots[v].radius
		edge = Line(start, end)
		edge.set_stroke(color=color, width=14)
		return edge

	def find_intersection_points(self, road_network):
		points = []
		for i in range(3):
			for j in range(3, 6):
				if j == 3:
					point = self.find_intersection_two_rects(road_network[j], 
						road_network[i], slant=True)
				else:
					point = self.find_intersection_two_rects(road_network[j], 
						road_network[i], slant=False)
				points.append(point)

		return points

	def find_intersection_two_rects(self, vert, horz, slant=False):
		if slant:
			p1_vert = self.get_bottom_left_corner(vert)
			p2_vert = self.get_top_left_corner(vert)

			p3_vert = self.get_bottom_right_corner(vert)
			p4_vert = self.get_top_right_corner(vert)
			A = (p1_vert + p2_vert) / 2 
			B = (p3_vert + p4_vert) / 2
		else:
			p1_vert = self.get_top_right_corner(vert)
			p2_vert = self.get_top_left_corner(vert)

			p3_vert = self.get_bottom_right_corner(vert)
			p4_vert = self.get_bottom_left_corner(vert)
			A = (p1_vert + p2_vert) / 2 
			B = (p3_vert + p4_vert) / 2


		p1_horz = self.get_bottom_left_corner(horz)
		p2_horz = self.get_top_left_corner(horz)

		p3_horz = self.get_bottom_right_corner(horz)
		p4_horz = self.get_top_right_corner(horz)
		C = (p1_horz + p2_horz) / 2 
		D = (p3_horz + p4_horz) / 2

		line1 = (A, B)
		line2 = (C, D)
		return line_intersection(line1, line2)

	def get_top_left_corner(self, rect):
		corners = rect.get_anchors()
		return corners[0]

	def get_top_right_corner(self, rect):
		corners = rect.get_anchors()
		return corners[1]

	def get_bottom_right_corner(self, rect):
		corners = rect.get_anchors()
		return corners[2]

	def get_bottom_left_corner(self, rect):
		corners = rect.get_anchors()
		return corners[3]

	def make_equivalent_graph(self):
		graph = []
		edges = {}

		radius, scale = 0.4, 0.8
		SHIFT = RIGHT * 2.5
		node_0 = GraphNode('0', position=RIGHT * 0.5 + UP * 2, radius=radius, scale=scale)
		node_1 = GraphNode('1', position=RIGHT * 3 + UP * 2, radius=radius, scale=scale)
		node_2 = GraphNode('2', position=RIGHT * 5 + UP * 2, radius=radius, scale=scale)
		node_3 = GraphNode('3', position=RIGHT * 1, radius=radius, scale=scale)
		node_4 = GraphNode('4', position=RIGHT * 3, radius=radius, scale=scale)
		node_5 = GraphNode('5', position=RIGHT * 5, radius=radius, scale=scale)
		node_6 = GraphNode('6', position=RIGHT * 1.5 + DOWN * 2, radius=radius, scale=scale)
		node_7 = GraphNode('7', position=RIGHT * 3 + DOWN * 2, radius=radius, scale=scale)
		node_8 = GraphNode('8', position=RIGHT * 5 + DOWN * 2, radius=radius, scale=scale)


		edges[(0, 1)] = node_0.connect(node_1)
		edges[(0, 3)] = node_0.connect(node_3)
		
		edges[(1, 2)] = node_1.connect(node_2)
		edges[(1, 4)] = node_1.connect(node_4)

		edges[(2, 5)] = node_2.connect(node_5)

		edges[(3, 4)] = node_3.connect(node_4)
		edges[(3, 6)] = node_3.connect(node_6)
		
		edges[(4, 5)] = node_4.connect(node_5)
		edges[(4, 7)] = node_4.connect(node_7)

		edges[(5, 8)] = node_5.connect(node_8)

		edges[(6, 7)] = node_6.connect(node_7)

		edges[(7, 8)] = node_7.connect(node_8)

		graph.append(node_0)
		graph.append(node_1)
		graph.append(node_2)
		graph.append(node_3)
		graph.append(node_4)
		graph.append(node_5)
		graph.append(node_6)
		graph.append(node_7)
		graph.append(node_8)

		return graph, edges

	def make_social_media_graph(self):
		graph = []
		edges = {}

		radius, scale = 0.4, 0.8
		SHIFT = RIGHT * 2.5
		node_0 = GraphNode('A', position=ORIGIN, radius=radius, scale=scale)
		node_1 = GraphNode('B', position=RIGHT * 1.3 + UP * 1.3, radius=radius, scale=scale)
		node_2 = GraphNode('C', position=RIGHT * 1.3 + DOWN * 1.3, radius=radius, scale=scale)
		node_3 = GraphNode('D', position=LEFT * 1.3 + UP * 1.3, radius=radius, scale=scale)
		node_4 = GraphNode('E', position=LEFT * 1.3 + DOWN * 1.3, radius=radius, scale=scale)
		node_5 = GraphNode('F', position=RIGHT * 3 + UP * 2, radius=radius, scale=scale)
		node_6 = GraphNode('G', position=RIGHT * 3 + UP * 0.6, radius=radius, scale=scale)
		node_7 = GraphNode('H', position=LEFT * 3 + UP * 2, radius=radius, scale=scale)
		node_8 = GraphNode('I', position=LEFT * 3 + UP * 0.6, radius=radius, scale=scale)
		node_9 = GraphNode('J', position=LEFT * 3 + DOWN * 0.6, radius=radius, scale=scale)
		node_10 = GraphNode('K', position=LEFT * 3 + DOWN * 2, radius=radius, scale=scale)
		node_11 = GraphNode('J', position=RIGHT * 3 + DOWN * 0.6, radius=radius, scale=scale)
		node_12 = GraphNode('K', position=RIGHT * 3 + DOWN * 2, radius=radius, scale=scale)


		edges[(0, 1)] = node_0.connect(node_1)
		edges[(0, 2)] = node_0.connect(node_2)
		edges[(0, 3)] = node_0.connect(node_3)
		edges[(0, 4)] = node_0.connect(node_4)
		
		edges[(1, 5)] = node_1.connect(node_5)
		edges[(1, 6)] = node_1.connect(node_6)

		edges[(2, 11)] = node_2.connect(node_11)
		edges[(2, 12)] = node_2.connect(node_12)

		edges[(3, 7)] = node_3.connect(node_7)
		edges[(3, 8)] = node_3.connect(node_8)

		edges[(4, 9)] = node_4.connect(node_9)
		edges[(4, 10)] = node_4.connect(node_10)

		# edges[(3, 4)] = node_3.connect(node_4)
		# edges[(3, 6)] = node_3.connect(node_6)
		
		# edges[(4, 5)] = node_4.connect(node_5)
		# edges[(4, 7)] = node_4.connect(node_7)

		# edges[(5, 8)] = node_5.connect(node_8)

		# edges[(6, 7)] = node_6.connect(node_7)

		# edges[(7, 8)] = node_7.connect(node_8)

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
		graph.append(node_10)
		graph.append(node_11)
		graph.append(node_12)

		return graph, edges

	def make_graph_mobject(self, graph, edge_dict, node_color=DARK_BLUE_B, 
		stroke_color=BLUE, data_color=WHITE, edge_color=GRAY, scale_factor=1):
		nodes = []
		edges = []
		for node in graph:
			node.circle.set_fill(color=node_color, opacity=0.5)
			node.circle.set_stroke(color=stroke_color)
			node.data.set_color(color=data_color)
			nodes.append(VGroup(node.circle, node.data))

		for edge in edge_dict.values():
			edge.set_stroke(width=7*scale_factor)
			edge.set_color(color=edge_color)
			edges.append(edge)
		return VGroup(*nodes), VGroup(*edges)

	def sharpie_edge(self, edge_dict, v, u, color=GREEN_SCREEN, scale_factor=1, animate=True):
		switch = False
		if v > u:
			u, v = v, u
			switch = True
		edge = edge_dict[(v, u)]
		if not switch:
			line = Line(edge.get_start(), edge.get_end())
		else:
			line = Line(edge.get_end(), edge.get_start())
		line.set_stroke(width=16 * scale_factor)
		line.set_color(color)
		if animate:
			self.play(
				ShowCreation(line)
			)
		return line

	def highlight_node(self, graph, index, color=GREEN_SCREEN, 
		start_angle=TAU/2, scale_factor=1, animate=True):
		node = graph[index]
		surround_circle = Circle(radius=node.circle.radius * scale_factor, TAU=-TAU, start_angle=start_angle)
		surround_circle.move_to(node.circle.get_center())
		# surround_circle.scale(1.15)
		surround_circle.set_stroke(width=8 * scale_factor)
		surround_circle.set_color(color)
		surround_circle.set_fill(opacity=0)
		if animate:
			self.play(
				ShowCreation(surround_circle)
			)
		return surround_circle


class SudokuApplication(GraphApplication):
	def construct(self):
		# title = TextMobject("Why Study Graphs?")
		# title.scale(1.2)
		# title.move_to(UP * 3.5)
		# h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
		# h_line.next_to(title, DOWN)
		# self.play(
		# 	Write(title),
		# 	ShowCreation(h_line)
		# )
		# self.wait()

		# show_up = TextMobject("Graphs show up all the time!")
		# show_up.move_to(DOWN * 3)
		# self.play(
		# 	Write(show_up)
		# )

		# social_networks = TextMobject("Example: Social Network")
		# social_networks.scale(0.8)
		# social_networks.next_to(show_up, DOWN)
		# self.play(
		# 	FadeIn(social_networks)
		# )

		# surprising = TextMobject("Graphs also show up in surprising places!")
		# surprising.move_to(show_up.get_center())
		# self.play(
		# 	ReplacementTransform(show_up, surprising)
		# )
		# self.wait(2)

		# sudoku_example = TextMobject("Example: Sudoku")
		# sudoku_example.scale(0.8)
		# sudoku_example.move_to(social_networks.get_center())

		# self.play(
		# 	ReplacementTransform(social_networks, sudoku_example)
		# )
		# self.wait(2)

		self.show_sudoku_example()

class GraphIntro(Scene):
	def construct(self):
		self.layman_terms()
		# self.show_path_in_graph()

	def layman_terms(self):
		title = TextMobject("Graph Theory")
		title.scale(1.2)
		title.shift(UP * 3.5)

		h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
		h_line.next_to(title, DOWN)
		self.play(
			Write(title),
			ShowCreation(h_line)
		)
		self.wait(5)

		scale_factor = 0.8
		graph, edge_dict = self.create_initial_graph()
		node_mobjects, edge_mobjects = self.make_graph_mobject(graph, edge_dict, scale_factor=scale_factor)
		entire_graph = VGroup(node_mobjects, edge_mobjects)

		entire_graph.scale(scale_factor)
		entire_graph.shift(DOWN * 2)
		self.play(
			FadeIn(entire_graph)
		)

		self.wait(5)

		super_layman_def = TextMobject("Layman's Definition of a Graph")
		super_layman_def.next_to(h_line, DOWN)
		self.play(
			Write(super_layman_def)
		)
		self.wait(5)

		simple_def = TextMobject("Network that helps define and visualize \\\\ relationships between various components")
		simple_def.scale(0.8)
		simple_def.next_to(super_layman_def, DOWN)
		self.play(
			Write(simple_def),
			run_time=3
		)
		self.wait(5)
		node_surrounding_circles = []
		for i in range(len(graph)):
			circle = self.highlight_node(graph, i, scale_factor=scale_factor, color=BLUE, animate=False)
			node_surrounding_circles.append(circle)

		components = simple_def[-10:].copy()
		components.set_color(BLUE)
		animations = [ShowCreation(circle) for circle in node_surrounding_circles]
		self.play(
			 *animations + [Transform(simple_def[-10:], components)], 
			 run_time=2
		)

		self.wait(2)

		edge_lines = []
		for key in edge_dict:
			edge = self.sharpie_edge(edge_dict, key[0], key[1], scale_factor=scale_factor, animate=False)
			edge_lines.append(edge)
		relationships = simple_def[34:47].copy()
		relationships.set_color(GREEN_SCREEN)
		animations = [GrowFromCenter(line) for line in edge_lines]
		self.play(
			*animations + [Transform(simple_def[34:47], relationships)],
			run_time=2
		)
		self.wait(7)

		v_arrow = Arrow(RIGHT * 4, RIGHT * 2.3)
		vertex_label = TextMobject("Vertex/Node")
		vertex_label.scale(0.8)
		vertex_label.next_to(v_arrow, RIGHT)
		vertex_label.set_color(BLUE)
		
		edge_label = TextMobject("Edge")
		edge_label.scale(0.8)
		
		edge_label.set_color(GREEN_SCREEN)
		edge_arrow = Arrow(DOWN * 3 + LEFT * 3, DOWN * 3 + LEFT * 1)
		edge_label.next_to(edge_arrow, LEFT)
		self.play(
			FadeIn(edge_label),
			FadeIn(vertex_label),
			ShowCreation(v_arrow),
			ShowCreation(edge_arrow),
			run_time=2
		)

		self.wait(14)

		fadeouts = [FadeOut(obj) for obj in node_surrounding_circles + edge_lines]
		other_fadeouts = [
		FadeOut(title),
		FadeOut(h_line),
		FadeOut(super_layman_def),
		FadeOut(simple_def),
		FadeOut(entire_graph),
		FadeOut(edge_label),
		FadeOut(vertex_label),
		FadeOut(v_arrow),
		FadeOut(edge_arrow)
		]
		self.play(
				*fadeouts + other_fadeouts
		)

	def show_path_in_graph(self):
		scale_factor = 1
		graph, edge_dict = self.create_initial_graph()
		node_mobjects, edge_mobjects = self.make_graph_mobject(graph, edge_dict, scale_factor=scale_factor)
		entire_graph = VGroup(node_mobjects, edge_mobjects)

		entire_graph.scale(scale_factor)
		self.play(
			FadeIn(entire_graph)
		)
		self.highlight_node(graph, 0, scale_factor=scale_factor)
		
		last_edge = self.sharpie_edge(edge_dict, 0, 6, scale_factor=scale_factor)
		angle = self.find_angle_of_intersection(graph, last_edge.get_end(), 6)
		self.highlight_node(graph, 6, start_angle=angle/360 * TAU, scale_factor=scale_factor)
		
		last_edge = self.sharpie_edge(edge_dict, 6, 7, scale_factor=scale_factor)
		angle = self.find_angle_of_intersection(graph, last_edge.get_end(), 7)
		self.highlight_node(graph, 7, start_angle=angle/360 * TAU, scale_factor=scale_factor)
		
		last_edge = self.sharpie_edge(edge_dict, 7, 3, scale_factor=scale_factor)
		angle = self.find_angle_of_intersection(graph, last_edge.get_end(), 3)
		self.highlight_node(graph, 3, start_angle=angle/360 * TAU, scale_factor=scale_factor)
		
		last_edge = self.sharpie_edge(edge_dict, 3, 2, scale_factor=scale_factor)
		angle = self.find_angle_of_intersection(graph, last_edge.get_end(), 2)
		self.highlight_node(graph, 2, start_angle=angle/360 * TAU, scale_factor=scale_factor)

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

	def highlight_edge(self, edge_dict, v, u, color=GREEN_SCREEN):
		switch = False
		if v > u:
			u, v = v, u
			switch = True
		edge = edge_dict[(v, u)]
		normal_1, normal_2 = edge.get_unit_normals()
		scale_factor = 1.5
		if not switch:
			line_1 = Line(edge.get_start() + normal_1 * SMALL_BUFF / scale_factor,
				edge.get_end() + normal_1 * SMALL_BUFF / scale_factor)
			line_2 = Line(edge.get_start() + normal_2 * SMALL_BUFF / scale_factor,
				edge.get_end() + normal_2 * SMALL_BUFF / scale_factor)
		else:
			line_1 = Line(edge.get_end() + normal_1 * SMALL_BUFF / scale_factor,
				edge.get_start() + normal_1 * SMALL_BUFF / scale_factor)
			line_2 = Line(edge.get_end() + normal_2 * SMALL_BUFF / scale_factor,
				edge.get_start() + normal_2 * SMALL_BUFF / scale_factor)
					
		line_1.set_stroke(width=8)
		line_2.set_stroke(width=8)
		line_1.set_color(color)
		line_2.set_color(color)

		self.play(
			ShowCreation(line_1),
			ShowCreation(line_2),
		)

	def sharpie_edge(self, edge_dict, v, u, color=GREEN_SCREEN, scale_factor=1, animate=True):
		switch = False
		if v > u:
			u, v = v, u
			switch = True
		edge = edge_dict[(v, u)]
		if not switch:
			line = Line(edge.get_start(), edge.get_end())
		else:
			line = Line(edge.get_end(), edge.get_start())
		line.set_stroke(width=16 * scale_factor)
		line.set_color(color)
		if animate:
			self.play(
				ShowCreation(line)
			)
		return line

	def highlight_node(self, graph, index, color=GREEN_SCREEN, 
		start_angle=TAU/2, scale_factor=1, animate=True):
		node = graph[index]
		surround_circle = Circle(radius=node.circle.radius * scale_factor, TAU=-TAU, start_angle=start_angle)
		surround_circle.move_to(node.circle.get_center())
		# surround_circle.scale(1.15)
		surround_circle.set_stroke(width=8 * scale_factor)
		surround_circle.set_color(color)
		surround_circle.set_fill(opacity=0)
		if animate:
			self.play(
				ShowCreation(surround_circle)
			)
		return surround_circle

	def create_initial_graph(self):
		graph = []
		edges = {}

		radius, scale = 0.4, 0.9
		SHIFT = RIGHT * 2.5
		node_0 = GraphNode('0', position=LEFT * 2.5, radius=radius, scale=scale)
		node_1 = GraphNode('1', position=RIGHT * 3, radius=radius, scale=scale)
		node_2 = GraphNode('2', position=RIGHT * 1.5 + UP, radius=radius, scale=scale)
		node_3 = GraphNode('3', position=UP * 2.5 + SHIFT, radius=radius, scale=scale)
		node_4 = GraphNode('4', position=DOWN * 2, radius=radius, scale=scale)
		node_5 = GraphNode('5', position=DOWN + RIGHT * 2, radius=radius, scale=scale)
		node_6 = GraphNode('6', position=LEFT + UP, radius=radius, scale=scale)
		node_7 = GraphNode('7', position=LEFT  * 2 + UP * 2 + SHIFT, radius=radius, scale=scale)
		node_8 = GraphNode('8', position=ORIGIN, radius=radius, scale=scale)

		edge_0_6 = node_0.connect(node_6)
		edges[(0, 6)] = edge_0_6
		# edges[(6, 0)] = edge_0_6
		edge_0_8 = node_0.connect(node_8)
		edges[(0, 8)] = edge_0_8
		# edges[(8, 0)] = edge_0_8
		edge_0_4 = node_0.connect(node_4)
		edges[(0, 4)] = edge_0_4
		# edges[(4, 0)] = edge_0_4
		edge_1_2 = node_1.connect(node_2)
		edges[(1, 2)] = edge_1_2
		# edges[(2, 1)] = edge_1_2
		edge_1_5 = node_1.connect(node_5)
		edges[(1, 5)] = edge_1_5
		# edges[(5, 1)] = edge_1_5
		edge_2_3 = node_2.connect(node_3)
		edges[(2, 3)] = edge_2_3
		# edges[(3, 2)] = edge_2_3
		edge_3_7 = node_3.connect(node_7)
		edges[(3, 7)] = edge_3_7
		# edges[(7, 3)] = edge_3_7
		edge_4_5 =  node_4.connect(node_5)
		edges[(4, 5)] = edge_4_5
		# edges[(5, 4)] = edge_4_5
		edge_6_7 = node_6.connect(node_7)
		edges[(6, 7)] = edge_6_7
		# edges[(7, 6)] = edge_6_7
		edge_1_8 = node_8.connect(node_1)
		edges[(1, 8)] = edge_1_8
		# edges[(8, 1)] = edge_1_8

		graph.append(node_0)
		graph.append(node_1)
		graph.append(node_2)
		graph.append(node_3)
		graph.append(node_4)
		graph.append(node_5)
		graph.append(node_6)
		graph.append(node_7)
		graph.append(node_8)

		return graph, edges

	def make_graph_mobject(self, graph, edge_dict, node_color=DARK_BLUE_B, 
		stroke_color=BLUE, data_color=WHITE, edge_color=GRAY, scale_factor=1):
		nodes = []
		edges = []
		for node in graph:
			node.circle.set_fill(color=node_color, opacity=0.5)
			node.circle.set_stroke(color=stroke_color)
			node.data.set_color(color=data_color)
			nodes.append(VGroup(node.circle, node.data))

		for edge in edge_dict.values():
			edge.set_stroke(width=7*scale_factor)
			edge.set_color(color=edge_color)
			edges.append(edge)
		return VGroup(*nodes), VGroup(*edges)

class GraphProblems(Scene):
	def construct(self):
		title = TextMobject("Interesting Graph Problems")
		title.scale(1.2)
		title.shift(UP * 3.5)

		h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
		h_line.next_to(title, DOWN)
		self.play(
			Write(title),
			ShowCreation(h_line)
		)

		self.wait()

		self.show_problems(title, h_line)

	def show_problems(self, title, h_line):
		graph, edge_dict = self.create_initial_graph()
		nodes, edges = self.make_graph_mobject(graph, edge_dict)
		entire_graph = VGroup(nodes, edges)
		scale_factor = 0.9
		entire_graph.scale(scale_factor)
		entire_graph.shift(DOWN * 1)
		self.play(
			FadeIn(entire_graph)
		)
		self.wait(10)

		word_scale = 0.8
		question_1 = TextMobject(r"Does a path exist between vertex $s$ and $t$?")
		question_1.scale(word_scale)
		question_1.next_to(h_line, DOWN)
		self.play(
			Write(question_1),
			run_time=2
		)
		self.wait(2)

		start = self.highlight_node(graph, 0, scale_factor=scale_factor)

		end = self.highlight_node(graph, 3, scale_factor=scale_factor)

		objects = self.show_path(graph, edge_dict, [0, 8, 1, 2, 3], scale_factor=scale_factor)

		self.play(
			*[FadeOut(obj) for obj in objects] + [FadeOut(start), FadeOut(end)]
		)

		question_2 = TextMobject("Is a graph connected?")
		question_2.scale(word_scale)
		question_2.next_to(h_line, DOWN)
		self.play(
			ReplacementTransform(question_1, question_2)
		)
		self.wait(6)

		question_3 = TextMobject(r"What is the path of least length between vertex $s$ and $t$?")
		question_3.scale(word_scale)
		question_3.next_to(h_line, DOWN)
		self.play(
			ReplacementTransform(question_2, question_3)
		)
		self.wait(3)

		start = self.highlight_node(graph, 0, scale_factor=scale_factor)

		end = self.highlight_node(graph, 3, scale_factor=scale_factor)

		objects = self.show_path(graph, edge_dict, [0, 6, 7, 3], scale_factor=scale_factor)

		self.wait(7)

		self.play(
			*[FadeOut(obj) for obj in objects] + [FadeOut(start), FadeOut(end)]
		)

		question_4 = TextMobject(r"Does the graph contain cycles?")
		question_4.scale(word_scale)
		question_4.next_to(h_line, DOWN)
		self.play(
			ReplacementTransform(question_3, question_4)
		)
		self.wait(7)

		self.play(
			FadeOut(question_4)
		)

		self.wait()

		question_5 = TextMobject(r"Given a set of $k$ colors, can we assign colors to each vertex" + "\\\\", "so that no two neighbors are assigned the same color?")
		question_5.scale(word_scale)
		question_5.next_to(h_line, DOWN)
		self.play(
			Write(question_5[0]),
			run_time=3
		)
		self.play(
			Write(question_5[1]),
			run_time=3
		)

		self.wait(12)

		for i, j in zip([1, 4, 6], [8, 7, 2]):
			nodes[i].set_stroke(color=BRIGHT_RED)
			nodes[j].set_stroke(color=GREEN_SCREEN)

		self.wait(5)

		for i, j in zip([1, 4, 6], [8, 7, 2]):
			nodes[i].set_stroke(color=BLUE)
			nodes[j].set_stroke(color=BLUE)

		question_6 = TextMobject(r"Does a path exist that uses every edge exactly once?")
		question_6.scale(word_scale)
		question_6.next_to(h_line, DOWN)
		self.play(
			FadeOut(question_5)
		)

		self.play(
			Write(question_6),
			run_time=2
		)

		objects = self.show_path(graph, edge_dict, [0, 6, 7, 3, 2, 1, 8, 0, 4, 5, 1], scale_factor=scale_factor)

		self.play(
			*[FadeOut(obj) for obj in objects] + [FadeOut(start), FadeOut(end)]
		)

		question_7 = TextMobject(r"Does a path exist that uses every vertex exactly once?")
		question_7.scale(word_scale)
		question_7.next_to(h_line, DOWN)
		self.play(
			ReplacementTransform(question_6, question_7)
		)
		self.wait()

		objects = self.show_path(graph, edge_dict, [6, 7, 3, 2, 1, 8, 0, 4, 5], scale_factor=scale_factor)


		glory = TextMobject("Finding efficient solution = ultimate computer science glory")
		glory.scale(word_scale)
		glory.next_to(question_7, DOWN)
		glory.shift(UP * 0.1)
		
		self.wait(5)
		self.play(
			Write(glory[:24]),
			run_time=2
		)
		self.wait()

		self.play(
			Write(glory[24:]),
			run_time=2
		)

		self.wait(13)
		

		self.play(
			*[FadeOut(obj) for obj in objects] + 
			[
			FadeOut(start), 
			FadeOut(end), 
			FadeOut(question_7), 
			FadeOut(entire_graph),
			FadeOut(title),
			FadeOut(h_line),
			FadeOut(glory)
			]
		)

		self.wait()
		

	def show_path(self, graph, edge_dict, path, scale_factor=1, directed=False):
		angle = 180
		objects = []
		for i in range(len(path) - 1):
			u, v = path[i], path[i + 1]
			surround_circle = self.highlight_node(graph, u, start_angle=angle/360 * TAU, scale_factor=scale_factor)
			last_edge = self.sharpie_edge(edge_dict, u, v, scale_factor=scale_factor, directed=directed)
			objects.extend([surround_circle, last_edge])
			angle = self.find_angle_of_intersection(graph, last_edge.get_end(), v)

		if v != path[0]:
			surround_circle = self.highlight_node(graph, v, start_angle=angle/360 * TAU, scale_factor=scale_factor)
			objects.append(surround_circle)

		return objects

	def sharpie_edge(self, edge_dict, u, v, color=GREEN_SCREEN, scale_factor=1, animate=True, directed=False):
		switch = False
		if u > v:
			edge = edge_dict[(v, u)]
			switch = True
		else:
			edge = edge_dict[(u, v)]
		
		if not switch:
			if not directed:
				line = Line(edge.get_start(), edge.get_end())
			else:
				line = Arrow(edge.get_start() - edge.unit_vector * edge.buff, 
					edge.get_end() + edge.unit_vector * edge.buff)
		else:
			if not directed:
				line = Line(edge.get_end(), edge.get_start())
			else:
				line = Arrow(edge.get_start() - edge.unit_vector * edge.buff, 
					edge.get_end() + edge.unit_vector * edge.buff)

		if not directed:
			line.set_stroke(width=16 * scale_factor)
		else:
			line.set_stroke(width=7 * scale_factor)
		line.set_color(color)
		if animate:
			self.play(
				ShowCreation(line)
			)
		return line

	def highlight_node(self, graph, index, color=GREEN_SCREEN, 
		start_angle=TAU/2, scale_factor=1, animate=True):
		node = graph[index]
		surround_circle = Circle(radius=node.circle.radius * scale_factor, TAU=-TAU, start_angle=start_angle)
		surround_circle.move_to(node.circle.get_center())
		# surround_circle.scale(1.15)
		surround_circle.set_stroke(width=8 * scale_factor)
		surround_circle.set_color(color)
		surround_circle.set_fill(opacity=0)
		if animate:
			self.play(
				ShowCreation(surround_circle)
			)
		return surround_circle

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

	def create_initial_graph(self):
		graph = []
		edges = {}

		radius, scale = 0.4, 0.9
		SHIFT = RIGHT * 2.5
		node_0 = GraphNode('0', position=LEFT * 2.5, radius=radius, scale=scale)
		node_1 = GraphNode('1', position=RIGHT * 3, radius=radius, scale=scale)
		node_2 = GraphNode('2', position=RIGHT * 1.5 + UP, radius=radius, scale=scale)
		node_3 = GraphNode('3', position=UP * 2.5 + SHIFT, radius=radius, scale=scale)
		node_4 = GraphNode('4', position=DOWN * 2, radius=radius, scale=scale)
		node_5 = GraphNode('5', position=DOWN + RIGHT * 2, radius=radius, scale=scale)
		node_6 = GraphNode('6', position=LEFT + UP, radius=radius, scale=scale)
		node_7 = GraphNode('7', position=LEFT  * 2 + UP * 2 + SHIFT, radius=radius, scale=scale)
		node_8 = GraphNode('8', position=ORIGIN, radius=radius, scale=scale)

		edge_0_6 = node_0.connect(node_6)
		edges[(0, 6)] = edge_0_6
		# edges[(6, 0)] = edge_0_6
		edge_0_8 = node_0.connect(node_8)
		edges[(0, 8)] = edge_0_8
		# edges[(8, 0)] = edge_0_8
		edge_0_4 = node_0.connect(node_4)
		edges[(0, 4)] = edge_0_4
		# edges[(4, 0)] = edge_0_4
		edge_1_2 = node_1.connect(node_2)
		edges[(1, 2)] = edge_1_2
		# edges[(2, 1)] = edge_1_2
		edge_1_5 = node_1.connect(node_5)
		edges[(1, 5)] = edge_1_5
		# edges[(5, 1)] = edge_1_5
		edge_2_3 = node_2.connect(node_3)
		edges[(2, 3)] = edge_2_3
		# edges[(3, 2)] = edge_2_3
		edge_3_7 = node_3.connect(node_7)
		edges[(3, 7)] = edge_3_7
		# edges[(7, 3)] = edge_3_7
		edge_4_5 =  node_4.connect(node_5)
		edges[(4, 5)] = edge_4_5
		# edges[(5, 4)] = edge_4_5
		edge_6_7 = node_6.connect(node_7)
		edges[(6, 7)] = edge_6_7
		# edges[(7, 6)] = edge_6_7
		edge_1_8 = node_1.connect(node_8)
		edges[(1, 8)] = edge_1_8
		# edges[(8, 1)] = edge_1_8

		graph.append(node_0)
		graph.append(node_1)
		graph.append(node_2)
		graph.append(node_3)
		graph.append(node_4)
		graph.append(node_5)
		graph.append(node_6)
		graph.append(node_7)
		graph.append(node_8)

		return graph, edges

	def make_graph_mobject(self, graph, edge_dict, node_color=DARK_BLUE_B, 
		stroke_color=BLUE, data_color=WHITE, edge_color=GRAY, 
		scale_factor=1, directed=False):
		nodes = []
		edges = []
		for node in graph:
			node.circle.set_fill(color=node_color, opacity=0.5)
			node.circle.set_stroke(color=stroke_color)
			node.data.set_color(color=data_color)
			nodes.append(VGroup(node.circle, node.data))

		for edge in edge_dict.values():
			if not directed:
				edge.set_stroke(width=7*scale_factor)
			else:
				edge.set_stroke(width=2*scale_factor)
			edge.set_color(color=edge_color)
			edges.append(edge)
		return VGroup(*nodes), VGroup(*edges)


class Thumbnail(GraphTerminology):
	def construct(self):
		scale_factor = 1
		graph, edge_dict = self.make_social_media_graph()
		nodes, edges = self.make_graph_mobject(graph, edge_dict, scale_factor=scale_factor)
		entire_graph = VGroup(nodes, edges)
		entire_graph.scale(scale_factor)
		entire_graph.shift(DOWN * 0.5)

		self.play(
			FadeIn(entire_graph)
		)

		graph_theory_text = Text("Graph Theory", font='Arial')
		graph_theory_text.scale(1.2)
		graph_theory_text.shift(UP * 3)
		self.play(
			FadeIn(graph_theory_text)
		)

		red_highlight = self.highlight_node(graph, 0, color=BRIGHT_RED, animate=False)
		self.play(
			ShowCreation(red_highlight)
		)
		highlight_circles = []
		for i in range(5, len(nodes)):
			surround_circle = self.highlight_node(graph, i, animate=False)
			highlight_circles.append(surround_circle)

		self.play(
			*[ShowCreation(circle) for circle in highlight_circles]
		)

		self.wait(2)

	def make_social_media_graph(self):
		graph = []
		edges = {}

		radius, scale = 0.4, 0.8
		SHIFT = RIGHT * 2.5
		node_0 = GraphNode('A', position=ORIGIN, radius=radius, scale=scale)
		node_1 = GraphNode('B', position=RIGHT * 1.3 + UP * 1.3, radius=radius, scale=scale)
		node_2 = GraphNode('C', position=RIGHT * 1.3 + DOWN * 1.3, radius=radius, scale=scale)
		node_3 = GraphNode('D', position=LEFT * 1.3 + UP * 1.3, radius=radius, scale=scale)
		node_4 = GraphNode('E', position=LEFT * 1.3 + DOWN * 1.3, radius=radius, scale=scale)
		node_5 = GraphNode('F', position=RIGHT * 3 + UP * 2, radius=radius, scale=scale)
		node_6 = GraphNode('G', position=RIGHT * 3 + UP * 0.6, radius=radius, scale=scale)
		node_7 = GraphNode('H', position=LEFT * 3 + UP * 2, radius=radius, scale=scale)
		node_8 = GraphNode('I', position=LEFT * 3 + UP * 0.6, radius=radius, scale=scale)
		node_9 = GraphNode('J', position=LEFT * 3 + DOWN * 0.6, radius=radius, scale=scale)
		node_10 = GraphNode('K', position=LEFT * 3 + DOWN * 2, radius=radius, scale=scale)
		node_11 = GraphNode('J', position=RIGHT * 3 + DOWN * 0.6, radius=radius, scale=scale)
		node_12 = GraphNode('K', position=RIGHT * 3 + DOWN * 2, radius=radius, scale=scale)
		# node_13 = GraphNode('L', position=RIG)

		edges[(0, 1)] = node_0.connect(node_1)
		edges[(0, 2)] = node_0.connect(node_2)
		edges[(0, 3)] = node_0.connect(node_3)
		edges[(0, 4)] = node_0.connect(node_4)
		
		edges[(1, 5)] = node_1.connect(node_5)
		edges[(1, 6)] = node_1.connect(node_6)

		edges[(2, 11)] = node_2.connect(node_11)
		edges[(2, 12)] = node_2.connect(node_12)

		edges[(3, 7)] = node_3.connect(node_7)
		edges[(3, 8)] = node_3.connect(node_8)

		edges[(4, 9)] = node_4.connect(node_9)
		edges[(4, 10)] = node_4.connect(node_10)

		# edges[(3, 4)] = node_3.connect(node_4)
		# edges[(3, 6)] = node_3.connect(node_6)
		
		# edges[(4, 5)] = node_4.connect(node_5)
		# edges[(4, 7)] = node_4.connect(node_7)

		# edges[(5, 8)] = node_5.connect(node_8)

		# edges[(6, 7)] = node_6.connect(node_7)

		# edges[(7, 8)] = node_7.connect(node_8)

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
		graph.append(node_10)
		graph.append(node_11)
		graph.append(node_12)

		return graph, edges

	def make_graph_mobject(self, graph, edge_dict, node_color=DARK_BLUE_B, 
		stroke_color=BLUE, data_color=WHITE, edge_color=GRAY, scale_factor=1):
		nodes = []
		edges = []
		for node in graph:
			node.circle.set_fill(color=node_color, opacity=0.5)
			node.circle.set_stroke(color=stroke_color)
			node.data.set_color(color=data_color)
			nodes.append(node.circle)

		for edge in edge_dict.values():
			edge.set_stroke(width=7*scale_factor)
			edge.set_color(color=edge_color)
			edges.append(edge)
		return VGroup(*nodes), VGroup(*edges)


# class BFS(Scene):
# 	def construct(self):
# 		graph = self.construct_graph()

# 		node = graph[0]
# 		self.draw_graph(node, BLUE, WHITE, ORANGE)

# 		node_X = self.get_node(graph, 'X')
# 		copy_circle = node_X.circle.copy()
# 		copy_circle.set_color(RED)
# 		self.play(Transform(node_X.circle, copy_circle))

# 		node_E = self.get_node(graph, 'E')
# 		copy_circle = node_E.circle.copy()
# 		copy_circle.set_color(GREEN)
# 		self.play(Transform(node_E.circle, copy_circle))

# 		self.wait(2)
# 		shortest_path = TextMobject("How do you find the shortest", "path from X to E?")
# 		shortest_path[0].move_to(LEFT * 3.7 + UP * 3.5)
# 		shortest_path[1].next_to(shortest_path[0], DOWN)
# 		self.play(FadeIn(shortest_path))

# 		self.wait(3)

# 		default_char = create_computer_char(color=BLUE, scale=0.7, position=UP * 0.5 + LEFT * 4.5)
# 		confused_character = create_confused_char(color=RED, scale=0.7, position=DOWN*3.5 + LEFT * 5)

# 		self.play(FadeInAndShiftFromDirection(default_char, LEFT))
		
		
# 		thought_bubble = SVGMobject("thought")
# 		thought_bubble.scale(0.9)
# 		thought_bubble.move_to(UP * 1.5 + LEFT * 3)

# 		bfs_text = TextMobject("Just use", "BFS!")
# 		bfs_text.scale(0.6)
# 		bfs_text.set_color(BLACK)
# 		bfs_text[0].move_to(thought_bubble.get_center() + UP * 0.4 + RIGHT * 0.05)
# 		bfs_text[1].move_to(thought_bubble.get_center() + UP * 0.1 + RIGHT * 0.05)

# 		self.play(FadeIn(thought_bubble), FadeIn(bfs_text))

# 		self.wait(3)

# 		default_char2 = create_computer_char(color=RED, scale=0.7, position=UP * 0.5 + LEFT * 4.5)
# 		default_char2.shift(DOWN * 3)
# 		thought_bubble2 = thought_bubble.copy()
# 		thought_bubble2.shift(DOWN * 3)
# 		self.play(FadeInAndShiftFromDirection(default_char2, LEFT))
# 		confused_character.move_to(default_char2.get_center())
# 		self.play(ReplacementTransform(default_char2, confused_character))
# 		why_text = TextMobject("But why?")
# 		why_text.scale(0.6)
# 		why_text.set_color(BLACK)
# 		why_text.move_to(thought_bubble2.get_center() + UP * 0.2 + RIGHT * 0.05)
# 		self.play(FadeIn(thought_bubble2), FadeIn(why_text))

# 		self.wait(3)

# 		self.play(FadeOutAndShift(default_char, LEFT), FadeOutAndShift(thought_bubble, LEFT), 
# 			FadeOutAndShift(confused_character, LEFT), FadeOutAndShift(thought_bubble2, LEFT))
# 		self.wait()

# 		self.bfs(node_X, node_E)
# 		self.wait(2)

# 	def bfs(self, start, end):
# 		queue = [start]
# 		while len(queue) > 0:
# 			current = queue.pop(0)
# 			# print(current)

# 			if current != start:
# 				node, edge = current
# 				if not node.marked:
# 					copy_node = node.circle.copy()
# 					copy_node.set_color(YELLOW)
# 					edge_copy = edge.copy()
# 					edge_copy.set_color(YELLOW)
# 					node.marked = True
# 					self.play(Indicate(node.circle, scale_factor=1, color=PINK), Indicate(edge, scale_factor=1,color=PINK), run_time=0.5)
# 					self.play(Transform(node.circle, copy_node), Transform(edge, edge_copy), run_time=0.5)
# 					if node == end:
# 						break
# 					for neighbor in node.neighbors:
# 						edge = self.get_edge(node, neighbor)
# 						if neighbor.prev == None:
# 							neighbor.prev = (node, edge)
# 						queue.append((neighbor, edge))

# 			elif not current.marked:	
# 				copy_current = current.circle.copy()
# 				copy_current.set_color(YELLOW)
# 				self.play(Transform(current.circle, copy_current), run_time=1)
# 				current.marked = True
# 				if current == end:
# 					break
# 				for neighbor in current.neighbors:
# 					edge = self.get_edge(current, neighbor)
# 					if neighbor.prev == None:
# 						neighbor.prev = (current, edge)
# 					queue.append((neighbor, edge))
					
# 		found_color = GREEN_SCREEN
# 		self.play(Indicate(end.circle, scale_factor=1, color=found_color), run_time=0.5)
# 		self.play(Indicate(end.circle, scale_factor=1, color=found_color), run_time=0.5)
# 		self.play(Indicate(end.circle, scale_factor=1, color=found_color), run_time=0.5)
# 		copy_end = end.circle.copy()
# 		copy_end.set_color(found_color)

# 		path = []
# 		current = end
# 		while current != start:
# 			path.append((current, self.get_edge(current, current.prev[0])))
# 			current = current.prev[0]

# 		path.append(current)
# 		transforms = []
# 		for elem in path:
# 			if elem != start:
# 				node, edge = elem
# 				self.play(Indicate(node.circle, scale_factor=1, color=found_color), 
# 					Indicate(edge, scale_factor=1, color=found_color), run_time=0.5)
# 				copy_node = node.circle.copy()
# 				copy_edge = edge.copy()
# 				copy_node.set_color(found_color)
# 				copy_edge.set_color(found_color)

# 				self.play(Transform(node.circle, copy_node), Transform(edge, copy_edge), run_time=0.5)
# 			else:
# 				node = elem
# 				self.play(Indicate(node.circle, scale_factor=1, color=found_color), run_time=0.5)
# 				copy_node = node.circle.copy()
# 				copy_node.set_color(found_color)
# 				self.play(Transform(node.circle, copy_node))

# 		shortest_path = TextMobject("Shortest path is", r"$X \rightarrow V \rightarrow M \rightarrow E$")
# 		shortest_path[0].move_to(LEFT * 4.1)
# 		shortest_path[1].next_to(shortest_path[0], DOWN)
# 		self.play(FadeIn(shortest_path))

# 		for elem in path[::-1]:
# 			if elem == start:
# 				node = elem
# 				self.play(Indicate(node.circle, scale_factor=1, color=RED), run_time=0.5)
# 			else:
# 				node, edge = elem
# 				self.play(Indicate(node.circle, scale_factor=1, color=RED),
# 					Indicate(edge, scale_factor=1, color=RED), run_time=0.5)

# 	def get_edge(self, u, v):
# 		for edge_u in u.edges:
# 			for edge_v in v.edges:
# 				midpoint_u, midpoint_v = edge_u.get_midpoint(), edge_v.get_midpoint()
# 				if np.allclose(midpoint_u, midpoint_v):
# 					return edge_u

# 		return False

# 	def char_to_index(self, char):
# 		return ord(char) - ord('A')

# 	def get_node(self, graph, char):
# 		index = self.char_to_index(char)
# 		return graph[index]

# 	def construct_graph(self):
# 		graph = []
# 		radius, scale = 0.4, 0.8
# 		SHIFT = RIGHT * 2.5
# 		node_A = GraphNode('A', position=ORIGIN + SHIFT, radius=radius, scale=scale)
# 		node_B = GraphNode('B', position=UP + RIGHT + SHIFT, radius=radius, scale=scale)
# 		node_C = GraphNode('C', position=UP * np.sqrt(2) + SHIFT, radius=radius, scale=scale)
# 		node_D = GraphNode('D', position=RIGHT * np.sqrt(2) + SHIFT, radius=radius, scale=scale)
# 		node_E = GraphNode('E', position=DOWN + RIGHT + SHIFT, radius=radius, scale=scale)
# 		node_F = GraphNode('F', position=DOWN * np.sqrt(2) + SHIFT, radius=radius, scale=scale)
# 		node_G = GraphNode('G', position=DOWN + LEFT + SHIFT, radius=radius, scale=scale)
# 		node_H = GraphNode('H', position=LEFT * np.sqrt(2) + SHIFT, radius=radius, scale=scale)
# 		node_I = GraphNode('I', position=UP + LEFT + SHIFT, radius=radius, scale=scale)

# 		node_J_center = self.get_center_new_node(node_C, node_B, node_A)
# 		node_J = GraphNode('J', position=node_J_center, radius=radius, scale=scale)
# 		node_K_center = self.get_center_new_node(node_B, node_D, node_A)
# 		node_K = GraphNode('K', position=node_K_center, radius=radius, scale=scale)
# 		node_L_center = self.get_center_new_node(node_D, node_E, node_A)
# 		node_L = GraphNode('L', position=node_L_center, radius=radius, scale=scale)
# 		node_M_center = self.get_center_new_node(node_E, node_F, node_A)
# 		node_M = GraphNode('M', position=node_M_center, radius=radius, scale=scale)
# 		node_N_center = self.get_center_new_node(node_F, node_G, node_A)
# 		node_N = GraphNode('N', position=node_N_center, radius=radius, scale=scale)
# 		node_O_center = self.get_center_new_node(node_G, node_H, node_A)
# 		node_O = GraphNode('O', position=node_O_center, radius=radius, scale=scale)
# 		node_P_center = self.get_center_new_node(node_H, node_I, node_A)
# 		node_P = GraphNode('P', position=node_P_center, radius=radius, scale=scale)
# 		node_Q_center = self.get_center_new_node(node_I, node_C, node_A)
# 		node_Q = GraphNode('Q', position=node_Q_center, radius=radius, scale=scale)

# 		graph.append(node_A)
# 		graph.append(node_B)
# 		graph.append(node_C)
# 		graph.append(node_D)
# 		graph.append(node_E)
# 		graph.append(node_F)
# 		graph.append(node_G)
# 		graph.append(node_H)
# 		graph.append(node_I)
# 		graph.append(node_J)
# 		graph.append(node_K)
# 		graph.append(node_L)
# 		graph.append(node_M)
# 		graph.append(node_N)
# 		graph.append(node_O)
# 		graph.append(node_P)
# 		graph.append(node_Q)

# 		node_A.connect(node_B)
# 		node_A.connect(node_C)
# 		node_A.connect(node_D)
# 		node_A.connect(node_E)
# 		node_A.connect(node_F)
# 		node_A.connect(node_G)
# 		node_A.connect(node_H)
# 		node_A.connect(node_I)

# 		node_R_center = self.get_center_last_layer(node_A, 0)
# 		node_R = GraphNode('R', position=node_R_center, radius=radius, scale=scale)
# 		graph.append(node_R)
# 		node_S_center = self.get_center_last_layer(node_A, 1)
# 		node_S = GraphNode('S', position=node_S_center, radius=radius, scale=scale)
# 		graph.append(node_S)
# 		node_T_center = self.get_center_last_layer(node_A, 2)
# 		node_T = GraphNode('T', position=node_T_center, radius=radius, scale=scale)
# 		graph.append(node_T)
# 		node_U_center = self.get_center_last_layer(node_A, 3)
# 		node_U = GraphNode('U', position=node_U_center, radius=radius, scale=scale)
# 		graph.append(node_U)
# 		node_V_center = self.get_center_last_layer(node_A, 4)
# 		node_V = GraphNode('V', position=node_V_center, radius=radius, scale=scale)
# 		graph.append(node_V)
# 		node_W_center = self.get_center_last_layer(node_A, 5)
# 		node_W = GraphNode('W', position=node_W_center, radius=radius, scale=scale)
# 		graph.append(node_W)
# 		node_X_center = self.get_center_last_layer(node_A, 6)
# 		node_X = GraphNode('X', position=node_X_center, radius=radius, scale=scale)
# 		graph.append(node_X)
# 		node_Y_center = self.get_center_last_layer(node_A, 7)
# 		node_Y = GraphNode('Y', position=node_Y_center, radius=radius, scale=scale)
# 		graph.append(node_Y)

# 		node_B.connect(node_C)
# 		node_B.connect(node_D)
# 		node_D.connect(node_E)
# 		node_E.connect(node_F)
# 		node_F.connect(node_G)
# 		node_G.connect(node_H)
# 		node_H.connect(node_I)
# 		node_I.connect(node_C)

# 		node_C.connect(node_J)
# 		node_B.connect(node_J)
# 		node_B.connect(node_K)
# 		node_D.connect(node_K)
# 		node_D.connect(node_L)
# 		node_E.connect(node_L)
# 		node_E.connect(node_M)
# 		node_F.connect(node_M)
# 		node_F.connect(node_N)
# 		node_G.connect(node_N)
# 		node_G.connect(node_O)
# 		node_H.connect(node_O)
# 		node_H.connect(node_P)
# 		node_I.connect(node_P)
# 		node_I.connect(node_Q)
# 		node_C.connect(node_Q)

# 		node_Q.connect(node_S)
# 		node_J.connect(node_S)
# 		node_J.connect(node_R)
# 		node_K.connect(node_R)
# 		node_K.connect(node_T)
# 		node_L.connect(node_T)
# 		node_L.connect(node_U)
# 		node_M.connect(node_U)
# 		node_M.connect(node_V)
# 		node_N.connect(node_V)
# 		node_N.connect(node_W)
# 		node_O.connect(node_W)
# 		node_O.connect(node_X)
# 		node_P.connect(node_X)
# 		node_P.connect(node_Y)
# 		node_Q.connect(node_Y)

# 		node_X.connect_curve(node_O, node_S, node_J, angle=-TAU/3)
# 		node_S.connect_curve(node_Q, node_T, node_L, angle=-TAU/3)
# 		node_T.connect_curve(node_K, node_V, node_N, angle=-TAU/3)
# 		node_V.connect_curve(node_M, node_X, node_P, angle=-TAU/3)

# 		return graph

# 	def get_center_new_node(self, node1, node2, node_A):
# 		middle = (node1.circle.get_center() + node2.circle.get_center()) / 2
# 		line_through_middle = Line(node_A.circle.get_center(), middle)
# 		unit_vector = line_through_middle.get_unit_vector()
# 		new_point = node_A.circle.get_center() + unit_vector * 2.5
# 		return new_point

# 	def get_center_last_layer(self, node_A, index):
# 		line = node_A.edges[index].get_unit_vector()
# 		new_point = node_A.circle.get_center() + line * 3.4
# 		return new_point

# 	def draw_graph(self, node, graph_color, data_color, line_color):
# 		stack = [node]
# 		all_circles = []
# 		all_data = []
# 		edges = set()
# 		while len(stack) > 0:
# 			node = stack.pop(0)
# 			if not node.drawn:
# 				node.circle.set_color(graph_color)
# 				node.data.set_color(data_color)
# 				all_circles.append(node.circle)
# 				all_data.append(node.data)
# 				node.drawn = True
# 				for graph_node in node.neighbors:
# 					stack.append(graph_node)

# 				for edge in node.edges:
# 					if edge not in edges:
# 						edges.add(edge)

# 		for edge in edges:
# 			edge.set_color(line_color)

# 		animations = []
# 		for circle, data in zip(all_circles, all_data):
# 			animations.append(GrowFromCenter(circle))
# 			animations.append(FadeIn(data))

# 		self.play(*animations, run_time=1.5)

# 		self.wait()
# 		edges = list(set(edges))

# 		self.play(*[ShowCreation(edge) for edge in edges], run_time=1.5)

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return np.array([x, y, 0])

# line_intersection((A, B), (C, D))


def create_computer_char(color=BLUE, scale=1, position=ORIGIN):
	outer_rectangle = Rectangle(height=2, width=3, 
		fill_color=color, fill_opacity=1, color=color)
	inner_rectangle = Rectangle(height=1.6, width=2.5, 
		fill_color=DARK_GRAY, fill_opacity=1, color=DARK_GRAY)
	extension = Rectangle(height=0.2, width=0.4, 
		fill_color=color, fill_opacity=1, color=color)
	extension.move_to(DOWN * (outer_rectangle.get_height() / 2 + extension.get_height() / 2))
	base = Rectangle(height=0.2, width=1,
		fill_color=color, fill_opacity=1, color=color)
	base.move_to(extension.get_center() + DOWN * extension.get_height())

	computer = VGroup(outer_rectangle, extension, base)

	left_circle = Circle(radius=0.27, color=color)
	left_circle.shift(LEFT * 0.6 + UP * 0.3)
	inner_left = Circle(radius=0.08, color=color, fill_color=color, fill_opacity=1)
	inner_left.shift(LEFT * 0.52, UP * 0.22)

	right_circle = Circle(radius=0.27, color=color)
	inner_right = Circle(radius=0.08, color=color, fill_color=color, fill_opacity=1)
	inner_right.shift(RIGHT * 0.52, UP*0.22)
	right_circle.shift(RIGHT * 0.6 + UP * 0.3)

	left_line = Line(DOWN * 0.3, DOWN * 0.5)
	right_line = Line(DOWN * 0.3, DOWN * 0.5)
	left_line.shift(LEFT * 0.5)
	right_line.shift(RIGHT * 0.5)
	bottom_line = Line(left_line.get_end(), right_line.get_end())
	left_line.set_color(color)
	right_line.set_color(color)
	bottom_line.set_color(color)

	smile = ArcBetweenPoints(left_line.get_start(), right_line.get_start())
	smile.set_color(color)
	
	left_eye_brow = ArcBetweenPoints(LEFT * 0.8 + UP * 0.6, LEFT * 0.4 + UP * 0.6, angle=-TAU/4)
	left_eye_brow.set_color(color)
	right_eye_brow = left_eye_brow.copy()
	right_eye_brow.shift(RIGHT * 1.2)
	right_eye_brow.set_color(color)

	eyes_and_smile = VGroup(left_circle, inner_left, right_circle, inner_right,
		smile, left_eye_brow, right_eye_brow)

	character = VGroup(computer, inner_rectangle, eyes_and_smile)
	character.scale(scale)
	character.move_to(position)


	return character

def create_confused_char(color=BLUE, scale=1, position=ORIGIN):
	outer_rectangle = Rectangle(height=2, width=3, 
		fill_color=color, fill_opacity=1, color=color)
	inner_rectangle = Rectangle(height=1.6, width=2.5, 
		fill_color=DARK_GRAY, fill_opacity=1, color=DARK_GRAY)
	extension = Rectangle(height=0.2, width=0.4, 
		fill_color=color, fill_opacity=1, color=color)
	extension.move_to(DOWN * (outer_rectangle.get_height() / 2 + extension.get_height() / 2))
	base = Rectangle(height=0.2, width=1,
		fill_color=color, fill_opacity=1, color=color)
	base.move_to(extension.get_center() + DOWN * extension.get_height())

	computer = VGroup(outer_rectangle, extension, base)

	left_circle = Circle(radius=0.27, color=color)
	left_circle.shift(LEFT * 0.6 + UP * 0.3)
	inner_left = Circle(radius=0.08, color=color, fill_color=color, fill_opacity=1)
	inner_left.shift(LEFT * 0.52, UP * 0.22)

	right_circle = Circle(radius=0.27, color=color)
	inner_right = Circle(radius=0.08, color=color, fill_color=color, fill_opacity=1)
	inner_right.shift(RIGHT * 0.52, UP*0.22)
	right_circle.shift(RIGHT * 0.6 + UP * 0.3)

	left_line = Line(DOWN * 0.3, DOWN * 0.5)
	right_line = Line(DOWN * 0.3, DOWN * 0.5)
	left_line.shift(LEFT * 0.5)
	right_line.shift(RIGHT * 0.5)
	bottom_line = Line(left_line.get_end(), right_line.get_end())
	left_line.set_color(color)
	right_line.set_color(color)
	bottom_line.set_color(color)

	smile = ArcBetweenPoints(left_line.get_start() + DOWN * 0.2, right_line.get_start(), angle=-TAU/4)
	smile.set_color(color)
	
	left_eye_brow = ArcBetweenPoints(LEFT * 0.8 + UP * 0.7, LEFT * 0.4 + UP * 0.7, angle=-TAU/4)
	left_eye_brow.set_color(color)
	right_eye_brow = ArcBetweenPoints(RIGHT * 0.8 + UP * 0.72, RIGHT * 0.4 + UP * 0.72, angle=-TAU/4)

	right_eye_brow.set_color(color)


	eyes_and_smile = VGroup(left_circle, inner_left, right_circle, inner_right,
		smile, left_eye_brow, right_eye_brow)

	character = VGroup(computer, inner_rectangle, eyes_and_smile)
	character.scale(scale)
	character.move_to(position)



	return character



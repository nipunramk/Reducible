from manimlib.imports import *
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


class GraphAnimationUtils(Scene):
	def construct(self):
		self.show_dfs_intuition()

	def show_dfs_intuition(self):

		dfs_intuition_title = TextMobject("DFS Intuition")
		dfs_intuition_title.scale(1.2)
		dfs_intuition_title.move_to(UP * 3.5)
		h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
		h_line.next_to(dfs_intuition_title, DOWN)
		

		graph, edge_dict = self.create_dfs_graph()
		nodes, edges = self.make_graph_mobject(graph, edge_dict)
		entire_graph = VGroup(nodes, edges)

		self.play(
			Write(dfs_intuition_title),
			ShowCreation(h_line)
		)
		self.wait()

		self.play(
			ShowCreation(entire_graph),
			run_time=3
		)

		self.wait(18)

		dfs_full_order = dfs(graph, 0)
		wait_times = [0] * len(dfs_full_order)
		wait_time_dict = {}
		for i in range(len(graph)):
			wait_time_dict[i] = 0

		wait_time_dict[0] = 2
		wait_times[0] = 7
		wait_time_dict[1] = 3
		wait_times[1] = 6

		order = TextMobject("Order: 0 1 2 3 5 6 7 8 9 4")
		order.shift(DOWN * 0.5)
		order.next_to(entire_graph, DOWN)
		self.play(
			Write(order[:6])
		)

		self.wait(5)
		
		all_highlights = []

		new_highlights = self.show_full_dfs_animation(graph, edge_dict, dfs_full_order[:5], order[6:], wait_times, wait_time_dict)
		all_highlights.extend(new_highlights)
		self.wait(10)
		graph[2].surround_circle.set_color(BRIGHT_RED)
		self.wait(3)
		self.play(
			CircleIndicate(graph[1].circle, color=BLUE),
			run_time=2
		)
		wait_time_dict[1] = 1
		self.indicate_neighbors(graph, 1, wait_time_dict)
		self.wait(2)

		wait_times[3] = 2 
		new_highlights = self.show_full_dfs_animation(graph, edge_dict, dfs_full_order[5:11], order[9:], wait_times, wait_time_dict)
		all_highlights.extend(new_highlights)
		self.wait()
		graph[6].surround_circle.set_color(BRIGHT_RED)
		self.wait(3)
		self.play(
			CircleIndicate(graph[5].circle, color=BLUE),
			run_time=2
		)
		self.indicate_neighbors(graph, 5, wait_time_dict)
		
		new_highlights = self.show_full_dfs_animation(graph, edge_dict, dfs_full_order[11:17], order[12:], wait_times, wait_time_dict)
		all_highlights.extend(new_highlights)
		graph[9].surround_circle.set_color(BRIGHT_RED)
		self.play(
			CircleIndicate(graph[8].circle, color=BLUE),
			run_time=2
		)

		self.wait(5)

		graph[8].surround_circle.set_color(BRIGHT_RED)
		self.wait(4)
		self.play(
			CircleIndicate(graph[7].circle, color=BLUE),
			run_time=2
		)
		self.wait(2)

		graph[7].surround_circle.set_color(BRIGHT_RED)
		self.wait()
		self.play(
			CircleIndicate(graph[5].circle, color=BLUE),
			run_time=2
		)
		self.wait(2)

		graph[5].surround_circle.set_color(BRIGHT_RED)
		self.wait(2)
		self.play(
			CircleIndicate(graph[3].circle, color=BLUE),
			run_time=2
		)
		self.wait(3)

		graph[3].surround_circle.set_color(BRIGHT_RED)
		self.wait(3)
		self.play(
			CircleIndicate(graph[1].circle, color=BLUE),
			run_time=2
		)

		self.indicate_neighbors(graph, 1, wait_time_dict)

		new_highlights = self.show_full_dfs_animation(graph, edge_dict, dfs_full_order[17:], order[-1:], wait_times, wait_time_dict)
		all_highlights.extend(new_highlights)

		graph[4].surround_circle.set_color(BRIGHT_RED)
		self.wait()
		self.play(
			CircleIndicate(graph[1].circle, color=BLUE),
			run_time=2
		)
		self.wait()

		graph[1].surround_circle.set_color(BRIGHT_RED)
		self.wait()
		self.play(
			CircleIndicate(graph[0].circle, color=BLUE),
			run_time=2
		)
		self.wait()

		graph[0].surround_circle.set_color(BRIGHT_RED)
		self.wait(5)


		self.play(
			*[FadeOut(obj) for obj in all_highlights],
		)

		self.wait()

		wait_times = [0] * len(dfs_full_order)
		wait_time_dict = {}
		for i in range(len(graph)):
			wait_time_dict[i] = 0

		all_highlights = self.show_dfs_preorder(graph, edge_dict, dfs_full_order, wait_times)
		self.wait(5)
		self.play(
			*[FadeOut(h) for h in all_highlights],
			FadeOut(entire_graph)
		)

		graph, edge_dict = self.create_dfs_graph2()
		nodes, edges = self.make_graph_mobject(graph, edge_dict)
		entire_graph = VGroup(nodes, edges)

		self.play(
			FadeIn(entire_graph)
		)

		self.wait(2)

		order_1 = TextMobject("Order 1: 0 1 2 3 5 6 7 8 9 4")
		order_1.move_to(order.get_center())
		self.play(
			ReplacementTransform(order, order_1)
		)
		self.wait()

		order_2 = TextMobject("Order 2: 0 2 1 4 3 5 8 9 7 6")
		order_2.next_to(order, DOWN)
		self.play(
			Write(order_2[:7])
		)
		self.wait(14)

		dfs_second_full_order = dfs(graph, 0)

		wait_times[0] = 5
		wait_times[2] = 26
		wait_times[1] = 2
		wait_times[4] = 2
		wait_times[5] = 1
		wait_times[9] = 2
		wait_times[7] = 9
		new_highlights = self.show_second_dfs_preorder(graph, edge_dict, dfs_second_full_order, order_2[7:], wait_times)

		self.wait(7)
		
		self.play(
			*[FadeOut(obj) for obj in new_highlights],
			FadeOut(entire_graph),
			FadeOut(order_1),
			FadeOut(order_2),
			FadeOut(entire_graph),
			FadeOut(dfs_intuition_title),
			FadeOut(h_line)
		)

	def show_dfs_preorder(self, graph, edge_dict, full_order, 
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

	def show_second_dfs_preorder(self, graph, edge_dict, full_order, order,
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

	def show_full_dfs_animation(self, graph, edge_dict, full_order, order,
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
		

	def create_dfs_graph(self):
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

		edges[(0, 2)] = node_0.connect(node_2)
		edges[(0, 1)] = node_0.connect(node_1)

		edges[(1, 4)] = node_1.connect(node_4)
		edges[(1, 3)] = node_1.connect(node_3)
		edges[(1, 2)] = node_1.connect(node_2)

		edges[(3, 5)] = node_3.connect(node_5)

		edges[(5, 8)] = node_5.connect(node_8)
		edges[(5, 7)] = node_5.connect(node_7)
		edges[(5, 6)] = node_5.connect(node_6)

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

	def create_dfs_graph_directed(self):
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

		edges[(0, 2)] = node_0.connect_arrow(node_2)
		edges[(0, 1)] = node_0.connect_arrow(node_1)

		edges[(1, 4)] = node_1.connect_arrow(node_4)
		edges[(1, 3)] = node_1.connect_arrow(node_3)
		edges[(1, 2)] = node_1.connect_arrow(node_2)

		edges[(3, 5)] = node_3.connect_arrow(node_5)

		edges[(5, 8)] = node_5.connect_arrow(node_8)
		edges[(5, 7)] = node_5.connect_arrow(node_7)
		edges[(5, 6)] = node_5.connect_arrow(node_6)

		edges[(7, 8)] = node_7.connect_arrow(node_8)

		edges[(8, 9)] = node_8.connect_arrow(node_9)

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

	def create_dfs_graph2(self):
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

	def create_small_graph(self):
		graph = []
		edges = {}

		radius, scale = 0.4, 0.9
		node_0 = GraphNode('0', position=DOWN * 1 + LEFT * 3, radius=radius, scale=scale)
		node_1 = GraphNode('1', position=UP * 1 + LEFT, radius=radius, scale=scale)
		node_2 = GraphNode('2', position=DOWN * 3 + LEFT, radius=radius, scale=scale)
		node_3 = GraphNode('3', position=DOWN * 1 + RIGHT, radius=radius, scale=scale)
		node_4 = GraphNode('4', position=DOWN * 1 + RIGHT * 3, radius=radius, scale=scale)


		edges[(0, 3)] = node_0.connect(node_3)
		edges[(0, 2)] = node_0.connect(node_2)
		edges[(0, 1)] = node_0.connect(node_1)

		edges[(1, 3)] = node_1.connect(node_3)

		edges[(2, 3)] = node_2.connect(node_3)

		edges[(3, 4)] = node_3.connect(node_4)

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
		line.set_stroke(width=16 * scale_factor)
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
		surround_circle = Circle(radius=node.circle.radius * scale_factor, TAU=-TAU, start_angle=start_angle)
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
		return VGroup(*nodes), VGroup(*edges)

class ThumbnailYoutube(GraphAnimationUtils):
	def construct(self):
		graph, edge_dict = self.create_huge_graph()
		nodes, edges = self.make_graph_mobject(graph, edge_dict, show_data=False)
		entire_graph = VGroup(nodes, edges)

		self.play(
			FadeIn(entire_graph),
		)
		self.wait()

		objects = self.show_path(graph, edge_dict, [4, 10, 16, 22, 29, 36, 42, 48, 55, 62, 68, 74, 81], scale_factor=1)

		# dfs = Text("Depth First Search", font='Anton')
		# dfs.scale(1.3)
		# dfs.shift(UP * 1.5)
		# self.play(
		# 	FadeIn(dfs)
		# )

		# self.play(
		# 	Flash(objects[-1], color=GREEN_SCREEN, line_length=0.4),
		# 	Flash(objects[0], color=GREEN_SCREEN, line_length=0.4, line_stroke_width=5)
		# )
		self.wait()

	def create_huge_graph(self):
		x_coords = list(range(-6, 7))
		y_coords_1 = list(np.arange(-3, 4, 1))
		y_coords_2 = list(np.arange(-2.5, 3, 1))
		graph = []
		edges = {}
		radius = 0.2
		scale = 0.5

		node_id = 0
		for i in range(len(x_coords)):
			if i % 2 == 0:
				y_coords = y_coords_1
			else:
				y_coords = y_coords_2
			for j in range(len(y_coords)):
				node = GraphNode(str(node_id), position=RIGHT * x_coords[i] + DOWN * y_coords[j], 
					radius=radius, scale=scale)
				graph.append(node)
				node_id += 1

		for i in range(node_id):
			if i < 78 and i % 13 != 6:
				edges[(i, i + 7)] = graph[i].connect(graph[i + 7])
			if i < 78 and i % 13 != 0:
				edges[(i, i + 6)] = graph[i].connect(graph[i + 6])

		return graph, edges

class Introduction(GraphAnimationUtils):
	def construct(self):
		graph, edge_dict = self.create_huge_graph()
		nodes, edges = self.make_graph_mobject(graph, edge_dict, show_data=False)
		entire_graph = VGroup(nodes, edges)

		self.play(
			ShowCreation(entire_graph),
			run_time=1
		)
		self.wait()

		dfs_full_order = dfs_maze(graph, 4)
		wait_times = [0] * len(dfs_full_order)

		self.show_dfs_preorder(graph, edge_dict, dfs_full_order, wait_times, 
			scale_factor=1, run_time=0.3)

		self.wait()

	def create_huge_graph(self):
		x_coords = list(range(-5, 6))
		y_coords_1 = list(np.arange(-3, 4, 1))
		y_coords_2 = list(np.arange(-2.5, 3, 1))
		graph = []
		edges = {}
		radius = 0.2
		scale = 0.5

		node_id = 0
		for i in range(len(x_coords)):
			if i % 2 == 0:
				y_coords = y_coords_1
			else:
				y_coords = y_coords_2
			for j in range(len(y_coords)):
				node = GraphNode(str(node_id), position=RIGHT * x_coords[i] + DOWN * y_coords[j], 
					radius=radius, scale=scale)
				graph.append(node)
				node_id += 1

		for i in range(node_id):
			if i < 65 and i % 13 != 6:
				edges[(i, i + 7)] = graph[i].connect(graph[i + 7])
			if i < 65 and i % 13 != 0:
				edges[(i, i + 6)] = graph[i].connect(graph[i + 6])

		return graph, edges

class Transition(Scene):
	def construct(self):
		rectangle = ScreenRectangle()
		rectangle.set_width(8)
		rectangle.set_height(6)

		title = TextMobject("Depth First Search (DFS)")
		title.scale(1.2)
		title.shift(UP * 3.5)

		h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
		h_line.next_to(title, DOWN)
		
		
		why_study = TextMobject("1. What Is a Graph Traversal?")
		why_study.next_to(rectangle, DOWN)

		definition = TextMobject("2. Intuition behind DFS")
		definition.next_to(rectangle, DOWN)

		representation = TextMobject("3. DFS Implementation")
		representation.next_to(rectangle, DOWN)

		problems = TextMobject("4. DFS Applications")
		problems.next_to(rectangle, DOWN)

		rectangle.scale(0.6)
		rectangle.move_to(RIGHT * 3)

		self.wait(5)
		self.play(
			Write(title),
			ShowCreation(h_line)
		)

		self.wait(14)

		self.play(
			ShowCreation(rectangle)
		)

		self.wait()
		self.play(
			Write(why_study)
		)
		self.wait(3)
		self.play(
			ReplacementTransform(why_study, definition),
		)
		self.wait(9)
		self.play(
			ReplacementTransform(definition, representation),
		)

		self.wait(5)
		self.play(
			ReplacementTransform(representation, problems),
		)
		self.wait(5)

		self.play(
			FadeOut(title),
			FadeOut(problems),
			FadeOut(h_line),
			FadeOut(rectangle),
		)

class GraphTraversal(GraphAnimationUtils):
	def construct(self):
		self.show_definition()

	def show_definition(self):
		term_title = TextMobject("Graph Traversal")
		term_title.scale(1.2)
		term_title.move_to(UP * 3.5)
		h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
		h_line.next_to(term_title, DOWN)
		self.play(
			Write(term_title),
			ShowCreation(h_line)
		)	
		self.show_graph_traversal_term(h_line, term_title)

	def show_graph_traversal_term(self, h_line, title):
		scale_factor = 0.9
		graph, edge_dict = self.create_initial_graph()
		node_mobjects, edge_mobjects = self.make_graph_mobject(graph, edge_dict, scale_factor=scale_factor)
		entire_graph = VGroup(node_mobjects, edge_mobjects)

		traversal_term = TextMobject(r"Graph Traversal - algorithm to visit every vertex of a graph")
		traversal_term[:14].set_color(GREEN_SCREEN)
		traversal_term.scale(0.8)
		traversal_term.next_to(h_line, DOWN)
		entire_graph.scale(scale_factor)
		entire_graph.shift(DOWN * 1.5)

		self.play(
			FadeIn(entire_graph)
		)

		self.wait()

		surround_circle = self.highlight_node(graph, 0, animate=False, scale_factor=scale_factor)
		self.play(
			ShowCreation(surround_circle)
		)
		self.wait(2)

		other_nodes = [self.highlight_node(graph, i, animate=False, scale_factor=scale_factor) for i in range(1, len(graph))]
		self.play(
			*[ShowCreation(circle) for circle in other_nodes]
		)
		self.wait(2)

		self.play(
			Write(traversal_term[:14])
		)
		self.wait(2)

		self.play(
			Write(traversal_term[14:])
		)
		self.wait(7)

		dfs_focus = TextMobject("In this video, we will focus on the DFS graph traversal")
		dfs_focus.scale(0.8)
		dfs_focus.next_to(traversal_term, DOWN)
		self.play(
			Write(dfs_focus)
		)
		self.wait(3)

		fadeouts = [
		FadeOut(surround_circle),
		FadeOut(dfs_focus),
		FadeOut(traversal_term),
		FadeOut(h_line),
		FadeOut(traversal_term),
		FadeOut(entire_graph),
		FadeOut(title)
		]
		self.play(
			*[FadeOut(c) for c in other_nodes] + fadeouts
		)

class DFSImplementation(GraphAnimationUtils):
	def construct(self):
		code = self.recursive_implementation()
		self.visualize_recursion(code)

	def recursive_implementation(self):
		title = TextMobject("DFS Implementation 1")
		title.scale(1.2)
		title.shift(UP * 3.5)

		h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
		h_line.next_to(title, DOWN)
		self.play(
			Write(title),
			ShowCreation(h_line)
		)

		self.wait()

		code_scale = 0.8
		def_statement = TextMobject(r"def dfs($G, v$):", font='Inconsolata')
		def_statement[:3].set_color(MONOKAI_BLUE)
		def_statement[3:6].set_color(MONOKAI_GREEN)
		def_statement[7].set_color(MONOKAI_ORANGE)
		def_statement[9].set_color(MONOKAI_ORANGE)
		def_statement.scale(code_scale)
		def_statement.to_edge(LEFT)
		
		line_1 = TextMobject(r"visit($v$)")
		line_1[:5].set_color(MONOKAI_BLUE)
		line_1.scale(code_scale)
		line_1.next_to(def_statement, DOWN * 0.5)
		line_1.to_edge(LEFT * 2)
		
		line_2 = TextMobject(r"for $w$ in $G$.neighbors($v$):")
		line_2[:3].set_color(MONOKAI_PINK)
		line_2[4:6].set_color(MONOKAI_PINK)
		line_2[8:17].set_color(MONOKAI_BLUE)
		line_2.scale(code_scale)
		line_2.next_to(line_1, DOWN * 0.5)
		line_2.to_edge(LEFT * 2)

		line_3 = TextMobject(r"dfs($G, w$)")
		line_3[:3].set_color(MONOKAI_BLUE)
		line_3.scale(code_scale)
		line_3.next_to(line_2, DOWN * 0.5)
		line_3.to_edge(LEFT * 3)

		graph, edge_dict = self.create_small_graph()
		nodes, edges = self.make_graph_mobject(graph, edge_dict)
		entire_graph = VGroup(nodes, edges)

		dfs_full_order = dfs(graph, 0)
		wait_times = [0] * len(dfs_full_order)
		entire_graph.shift(UP)

		order = TextMobject(r"dfs($G, 0$): 0 1 3 4 2")
		order.next_to(entire_graph, DOWN)
		
		self.play(
			FadeIn(entire_graph)
		)
		self.wait()

		self.play(
			Write(order[:9])
		)
		self.wait()

		
		all_highlights = self.show_second_dfs_preorder(graph, edge_dict, dfs_full_order, order[9:], wait_times)
		self.wait()

		self.play(
			*[FadeOut(obj) for obj in all_highlights]
		)
		self.wait()

		surround_circle = self.highlight_node(graph, 0)
		self.wait(3)

		to_fade = [surround_circle, nodes[0], edge_dict[(0, 1)], edge_dict[(0, 2)], edge_dict[(0, 3)]]
		self.play(
			*[FadeOut(obj) for obj in to_fade]
		)

		all_highlights = self.show_dfs_preorder(graph, edge_dict, dfs_full_order[2:], wait_times)

		order[-4:].set_color(GREEN_SCREEN)
		self.wait(2)

		self.play(
			*[FadeOut(obj) for obj in all_highlights]
		)
		order[-4:].set_color(WHITE)
		self.play(
			*[FadeIn(obj) for obj in to_fade[1:]]
		)


		entire_graph_copy = entire_graph.copy()
		entire_graph_copy.shift(RIGHT * 2.5)
		order_copy = order.copy()
		order_copy.shift(RIGHT * 2.5)
		self.play(
			Transform(entire_graph, entire_graph_copy),
			Transform(order, order_copy)
		)

		code = VGroup(def_statement, line_1, line_2, line_3)
		code.move_to(LEFT * 3.5)

		self.play(
			Write(code[0])
		)
		self.wait()
		self.play(
			Write(code[1])
		)
		self.wait(2)
		self.play(
			Write(code[2])
		)
		self.wait(2)
		self.play(
			Write(code[3])
		)
		self.wait(14)

		wait_times[0] = 2
		all_highlights = self.show_dfs_preorder(graph, edge_dict, dfs_full_order[:3], wait_times)
		self.wait(15)

		arrow = graph[1].connect_arrow(graph[0])
		original_shift = UP + RIGHT * 2.5
		arrow.set_color(RED)
		arrow.shift(LEFT * 0.2 + UP * 0.2 + original_shift)
		self.play(
			ShowCreation(arrow)
		)

		self.wait(8)

		graph[0].neighbors.pop()
		graph[0].edges.pop()

		cross = Cross(code)
		self.play(
			ShowCreation(cross)
		)

		self.wait(10)

		self.play(
			FadeOut(cross)
		)

		self.wait(5)

		first_indentation = LEFT * 2.55
		marked_array = TextMobject(r"marked = [False] * $G$.size()")
		marked_array.scale(code_scale)
		marked_array.next_to(def_statement, UP * 0.5)
		marked_array.to_edge(first_indentation)
		marked_array[14].shift(DOWN * SMALL_BUFF)
		marked_array[8:13].set_color(MONOKAI_PURPLE)
		marked_array[14].set_color(MONOKAI_PINK)
		marked_array[-6:-2].set_color(MONOKAI_BLUE)
		self.play(
			Write(marked_array)
		)
		self.wait(8)

		line_after_visit = TextMobject(r"marked[$v$] = True")
		line_after_visit[-4:].set_color(MONOKAI_PURPLE)
		line_after_visit.scale(0.8)
		line_after_visit.next_to(line_1, DOWN * 0.5)
		line_after_visit.to_edge(first_indentation + LEFT)

		line_2_copy = line_2.copy()
		line_2_copy.shift(DOWN * 0.5)

		line_3_copy = line_3.copy()
		line_3_copy.shift(DOWN * 0.5)

		self.play(
			Transform(line_2, line_2_copy),
			Transform(line_3, line_3_copy)
		)

		self.play(
			Write(line_after_visit)
		)

		self.wait(7)

		line_after_for_loop = TextMobject(r"if not marked[$w$]:")
		line_after_for_loop.scale(code_scale)
		line_after_for_loop.next_to(line_2, DOWN * 0.5)
		line_after_for_loop.to_edge(first_indentation + LEFT * 2)
		line_after_for_loop[:5].set_color(MONOKAI_PINK)

		line_3_copy = line_3.copy()
		line_3_copy.shift(DOWN * 0.5)
		line_3_copy.to_edge(first_indentation + LEFT * 3)

		self.play(
			Transform(line_3, line_3_copy)
		)

		self.play(
			Write(line_after_for_loop)
		)
		self.wait(2)


		code = VGroup(
			marked_array, 
			def_statement, 
			line_1, 
			line_after_visit,
			line_2,
			line_after_for_loop,
			line_3,
		)

		self.play(
			*[FadeOut(obj) for obj in all_highlights] + [FadeOut(arrow)],
			FadeOut(entire_graph),
			FadeOut(title),
			FadeOut(h_line),
			FadeOut(order)
		)

		self.wait(2)

		scaled_code = code.copy()
		scaled_code.scale(0.9)
		scaled_code.move_to(RIGHT * 4.5)

		self.play(
			Transform(code, scaled_code),
			run_time=2
		)

		return code 
	
	def visualize_recursion(self, code):

		text_scale = 0.7
		dfs_0 = TextMobject(r"dfs($G, 0$)")
		dfs_0.scale(text_scale)
		dfs_0.shift(UP * 3 + LEFT)
		graph_0, edge_dict_0 = self.create_small_graph()
		nodes_0, edges_0 = self.make_graph_mobject(graph_0, edge_dict_0)
		entire_graph_0 = VGroup(nodes_0, edges_0)
		scale_factor = 0.4
		entire_graph_0.scale(scale_factor)

		entire_graph_0.next_to(dfs_0, RIGHT)

		self.play(
			Write(dfs_0)
		)

		self.play(
			FadeIn(entire_graph_0)
		)

		self.wait(2)

		order_0 = TextMobject("0 1 3 4 2")
		order_0.scale(text_scale)
		order_0.next_to(dfs_0, DOWN * text_scale)

		all_highlights = []

		surround_circle = self.highlight_node(graph_0, 0, scale_factor=scale_factor)
		self.play(
			TransformFromCopy(graph_0[0].data, order_0[0])
		)
		self.wait(10)
		line_0 = self.sharpie_edge(edge_dict_0, 0, 1, scale_factor=scale_factor, animate=False)
		self.play(
			ShowCreation(line_0)
		)

		all_highlights.extend([surround_circle, line_0])

		self.wait(3)

		dfs_1 = TextMobject(r"dfs($G, 1$)")
		dfs_1.scale(text_scale)
		dfs_1.shift(UP * 1 + LEFT * 2)
		graph_1, edge_dict_1 = self.create_small_graph()
		nodes_1, edges_1 = self.make_graph_mobject(graph_1, edge_dict_1)
		entire_graph_1 = VGroup(nodes_1, edges_1)
		scale_factor = 0.4
		entire_graph_1.scale(scale_factor)

		entire_graph_1.next_to(dfs_1, RIGHT)

		self.play(
			Write(dfs_1)
		)

		surround_circle_0 = self.highlight_node(graph_1, 0, scale_factor=scale_factor, animate=False, color=BRIGHT_RED)
		line_1 = self.sharpie_edge(edge_dict_1, 0, 1, scale_factor=scale_factor, animate=False, color=BRIGHT_RED)
		all_highlights.extend([surround_circle_0, line_1])

		self.play(
			FadeIn(entire_graph_1),
			FadeIn(surround_circle_0),
			FadeIn(line_1)
		)

		self.wait(4)

		order_1 = TextMobject("1 3 4 2")
		order_1.scale(text_scale)
		order_1.next_to(dfs_1, DOWN * text_scale)
		node_1 = self.highlight_node(graph_1, 1, scale_factor=scale_factor, animate=False)
		node_0 = self.highlight_node(graph_0, 1, scale_factor=scale_factor, animate=False)
		all_highlights.extend([node_0, node_1])

		self.play(
			ShowCreation(node_1),
			ShowCreation(node_0)
		)

		self.play(
			TransformFromCopy(graph_1[1].data, order_1[0]),
			TransformFromCopy(graph_0[1].data, order_0[1])
		)
		self.wait(14)

		line_1 = self.sharpie_edge(edge_dict_1, 1, 3, scale_factor=scale_factor, animate=False)
		line_0 = self.sharpie_edge(edge_dict_0, 1, 3, scale_factor=scale_factor, animate=False)
		self.play(
			ShowCreation(line_1),
			ShowCreation(line_0)
		)
		all_highlights.extend([line_1, line_0])
		self.wait()


		dfs_3 = TextMobject(r"dfs($G, 3$)")
		dfs_3.scale(text_scale)
		dfs_3.shift(DOWN * 1 + LEFT * 3)
		graph_3, edge_dict_3 = self.create_small_graph()
		nodes_3, edges_3 = self.make_graph_mobject(graph_3, edge_dict_3)
		entire_graph_3 = VGroup(nodes_3, edges_3)
		scale_factor = 0.4
		entire_graph_3.scale(scale_factor)

		entire_graph_3.next_to(dfs_3, RIGHT)

		self.play(
			Write(dfs_3)
		)

		surround_circle_0 = self.highlight_node(graph_3, 0, scale_factor=scale_factor, animate=False, color=BRIGHT_RED)
		edge_01 = self.sharpie_edge(edge_dict_3, 0, 1, scale_factor=scale_factor, animate=False, color=BRIGHT_RED)
		surround_circle_1 = self.highlight_node(graph_3, 1, scale_factor=scale_factor, animate=False, color=BRIGHT_RED)
		edge_13 = self.sharpie_edge(edge_dict_3, 1, 3, scale_factor=scale_factor, animate=False, color=BRIGHT_RED)
		all_highlights.extend([surround_circle_0, edge_01, surround_circle_1, edge_13])

		self.play(
			FadeIn(entire_graph_3),
			FadeIn(surround_circle_0),
			FadeIn(edge_01),
			FadeIn(surround_circle_1),
			FadeIn(edge_13)
		)

		self.wait()

		order_3 = TextMobject("3 4 2")
		order_3.scale(text_scale)
		order_3.next_to(dfs_3, DOWN * text_scale)
		node_3 = self.highlight_node(graph_3, 3, scale_factor=scale_factor, animate=False)
		node_1 = self.highlight_node(graph_1, 3, scale_factor=scale_factor, animate=False)
		node_0 = self.highlight_node(graph_0, 3, scale_factor=scale_factor, animate=False)
		all_highlights.extend([node_0, node_1, node_3])

		self.play(
			ShowCreation(node_3),
			ShowCreation(node_1),
			ShowCreation(node_0)
		)

		self.wait()

		self.play(
			TransformFromCopy(graph_3[3].data, order_3[0]),
			TransformFromCopy(graph_1[3].data, order_1[1]),
			TransformFromCopy(graph_0[3].data, order_0[2])
		)

		self.wait(9)

		line_3 = self.sharpie_edge(edge_dict_3, 3, 4, scale_factor=scale_factor, animate=False)
		line_1 = self.sharpie_edge(edge_dict_1, 3, 4, scale_factor=scale_factor, animate=False)
		line_0 = self.sharpie_edge(edge_dict_0, 3, 4, scale_factor=scale_factor, animate=False)
		self.play(
			ShowCreation(line_3),
			ShowCreation(line_1),
			ShowCreation(line_0)
		)
		all_highlights.extend([line_3, line_1, line_0])

		self.wait(2)
		
		dfs_4 = TextMobject(r"dfs($G, 4$)")
		dfs_4.scale(text_scale)
		dfs_4.shift(DOWN * 3 + LEFT * 4)
		graph_4, edge_dict_4 = self.create_small_graph()
		nodes_4, edges_4 = self.make_graph_mobject(graph_4, edge_dict_4)
		entire_graph_4 = VGroup(nodes_4, edges_4)
		scale_factor = 0.4
		entire_graph_4.scale(scale_factor)

		entire_graph_4.next_to(dfs_4, RIGHT)

		self.play(
			Write(dfs_4)
		)

		surround_circle_0 = self.highlight_node(graph_4, 0, scale_factor=scale_factor, animate=False, color=BRIGHT_RED)
		edge_01 = self.sharpie_edge(edge_dict_4, 0, 1, scale_factor=scale_factor, animate=False, color=BRIGHT_RED)
		surround_circle_1 = self.highlight_node(graph_4, 1, scale_factor=scale_factor, animate=False, color=BRIGHT_RED)
		edge_13 = self.sharpie_edge(edge_dict_4, 1, 3, scale_factor=scale_factor, animate=False, color=BRIGHT_RED)
		surround_circle_3 = self.highlight_node(graph_4, 3, scale_factor=scale_factor, animate=False, color=BRIGHT_RED)
		edge_34 = self.sharpie_edge(edge_dict_4, 3, 4, scale_factor=scale_factor, animate=False, color=BRIGHT_RED)
		all_highlights.extend([surround_circle_0, edge_01, surround_circle_1, edge_13, surround_circle_3, edge_34])

		self.play(
			FadeIn(entire_graph_4),
			FadeIn(surround_circle_0),
			FadeIn(edge_01),
			FadeIn(surround_circle_1),
			FadeIn(edge_13),
			FadeIn(surround_circle_3),
			FadeIn(edge_34)
		)

		self.wait(2)

		order_4 = TextMobject("4")
		order_4.scale(text_scale)
		order_4.next_to(dfs_4, DOWN * text_scale)
		node_4 = self.highlight_node(graph_4, 4, scale_factor=scale_factor, animate=False)
		node_3 = self.highlight_node(graph_3, 4, scale_factor=scale_factor, animate=False)
		node_1 = self.highlight_node(graph_1, 4, scale_factor=scale_factor, animate=False)
		node_0 = self.highlight_node(graph_0, 4, scale_factor=scale_factor, animate=False)
		all_highlights.extend([node_0, node_1, node_3, node_4])

		self.play(
			ShowCreation(node_4),
			ShowCreation(node_3),
			ShowCreation(node_1),
			ShowCreation(node_0)
		)

		self.wait()

		self.play(
			TransformFromCopy(graph_4[4].data, order_4[0]),
			TransformFromCopy(graph_3[4].data, order_3[1]),
			TransformFromCopy(graph_1[4].data, order_1[2]),
			TransformFromCopy(graph_0[4].data, order_0[3])
		)

		self.wait(18)

		self.play(
			Indicate(dfs_3)
		)
		self.play(
			Indicate(dfs_3)
		)

		self.wait(9)

		line_3 = self.sharpie_edge(edge_dict_3, 3, 2, scale_factor=scale_factor, animate=False)
		line_1 = self.sharpie_edge(edge_dict_1, 3, 2, scale_factor=scale_factor, animate=False)
		line_0 = self.sharpie_edge(edge_dict_0, 3, 2, scale_factor=scale_factor, animate=False)
		self.play(
			ShowCreation(line_3),
			ShowCreation(line_1),
			ShowCreation(line_0)
		)
		all_highlights.extend([line_3, line_1, line_0])

		self.wait(2)

		dfs_2 = TextMobject(r"dfs($G, 2$)")
		dfs_2.scale(text_scale)
		dfs_2.shift(DOWN * 3 + RIGHT * 1)
		graph_2, edge_dict_2 = self.create_small_graph()
		nodes_2, edges_2 = self.make_graph_mobject(graph_2, edge_dict_2)
		entire_graph_2 = VGroup(nodes_2, edges_2)
		scale_factor = 0.4
		entire_graph_2.scale(scale_factor)

		entire_graph_2.next_to(dfs_2, RIGHT)

		self.play(
			Write(dfs_2)
		)

		self.wait()

		surround_circle_0 = self.highlight_node(graph_2, 0, scale_factor=scale_factor, animate=False, color=BRIGHT_RED)
		edge_01 = self.sharpie_edge(edge_dict_2, 0, 1, scale_factor=scale_factor, animate=False, color=BRIGHT_RED)
		surround_circle_1 = self.highlight_node(graph_2, 1, scale_factor=scale_factor, animate=False, color=BRIGHT_RED)
		edge_13 = self.sharpie_edge(edge_dict_2, 1, 3, scale_factor=scale_factor, animate=False, color=BRIGHT_RED)
		surround_circle_3 = self.highlight_node(graph_2, 3, scale_factor=scale_factor, animate=False, color=BRIGHT_RED)
		edge_34 = self.sharpie_edge(edge_dict_2, 3, 4, scale_factor=scale_factor, animate=False, color=BRIGHT_RED)
		surround_circle_4 = self.highlight_node(graph_2, 4, scale_factor=scale_factor, animate=False, color=BRIGHT_RED)
		edge_32 = self.sharpie_edge(edge_dict_2, 3, 2, scale_factor=scale_factor, animate=False, color=BRIGHT_RED)
		all_highlights.extend([surround_circle_0, edge_01, surround_circle_1, edge_13, surround_circle_3, edge_34, surround_circle_4, edge_32])

		self.play(
			FadeIn(entire_graph_2),
			FadeIn(surround_circle_0),
			FadeIn(edge_01),
			FadeIn(surround_circle_1),
			FadeIn(edge_13),
			FadeIn(surround_circle_3),
			FadeIn(edge_34),
			FadeIn(surround_circle_4),
			FadeIn(edge_32)
		)

		self.wait()

		order_2 = TextMobject("2")
		order_2.scale(text_scale)
		order_2.next_to(dfs_2, DOWN * text_scale)
		node_2 = self.highlight_node(graph_2, 2, scale_factor=scale_factor, animate=False)
		node_3 = self.highlight_node(graph_3, 2, scale_factor=scale_factor, animate=False)
		node_1 = self.highlight_node(graph_1, 2, scale_factor=scale_factor, animate=False)
		node_0 = self.highlight_node(graph_0, 2, scale_factor=scale_factor, animate=False)
		all_highlights.extend([node_0, node_1, node_3, node_2])

		self.play(
			ShowCreation(node_2),
			ShowCreation(node_3),
			ShowCreation(node_1),
			ShowCreation(node_0)
		)

		self.wait()

		self.play(
			TransformFromCopy(graph_2[2].data, order_2[0]),
			TransformFromCopy(graph_3[2].data, order_3[2]),
			TransformFromCopy(graph_1[2].data, order_1[3]),
			TransformFromCopy(graph_0[2].data, order_0[4])
		)

		self.wait(7)

		self.play(
			Indicate(dfs_3)
		)

		self.wait(5)

		self.play(
			Indicate(dfs_1)
		)

		self.wait(3)

		self.play(
			Indicate(dfs_0)
		)


		self.wait(14)

		other_fadeouts = [
		FadeOut(entire_graph_1),
		FadeOut(entire_graph_2),
		FadeOut(entire_graph_3),
		FadeOut(entire_graph_4),
		FadeOut(entire_graph_0),
		FadeOut(code),
		FadeOut(dfs_0),
		FadeOut(dfs_1),
		FadeOut(dfs_2),
		FadeOut(dfs_3),
		FadeOut(dfs_4),
		FadeOut(order_0),
		FadeOut(order_1),
		FadeOut(order_2),
		FadeOut(order_3),
		FadeOut(order_4)
		]
		self.play(
			*[FadeOut(highlight) for highlight in all_highlights] + other_fadeouts
		)

class DFSIteratitiveImplementation(GraphAnimationUtils):
	def construct(self):
		self.iterative_implementation()


	def iterative_implementation(self):
		title = TextMobject("DFS Implementation 2")
		title.scale(1.2)
		title.shift(UP * 3.5)

		h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
		h_line.next_to(title, DOWN)
		self.play(
			Write(title),
			ShowCreation(h_line)
		)

		self.wait()

		graph, edge_dict = self.create_small_graph()
		nodes, edges = self.make_graph_mobject(graph, edge_dict)
		entire_graph = VGroup(nodes, edges)
		entire_graph.shift(RIGHT * 2.5 + UP)
		self.play(
			FadeIn(entire_graph)
		)

		code = self.generate_code()

		self.wait(4)

		scale_factor = 0.8
		entire_graph_copy = entire_graph.copy()
		entire_graph_copy.scale(scale_factor)
		entire_graph_copy.shift(UP * 0.5)
		self.play(
			Transform(entire_graph, entire_graph_copy)
		)

		self.wait()

		order = TextMobject(r"$\text{dfs\_iter}(G, 0):$", "0 1 3 4 2")
		order.next_to(entire_graph, DOWN)
		self.play(
			Write(order[0])
		)

		stack_text = TextMobject("stack")
		stack_text.scale(0.8)
		stack_text.move_to(RIGHT * 0.5 + DOWN * 2.8)
		vert_line = Line(stack_text[-1].get_center() + RIGHT * 0.25 + UP * 0.25, 
			stack_text[-1].get_center() + RIGHT * 0.25 + DOWN * 0.25)
		horiz_line = Line(stack_text[-1].get_center() + RIGHT * 0.25 + DOWN * 0.25, 
			stack_text[-1].get_center() + RIGHT * 0.65 + DOWN * 0.25)

		code_arrow = Arrow(ORIGIN, RIGHT * 1.2)
		code_arrow.set_color(GREEN_SCREEN)
		code_arrow.next_to(code[2], LEFT * 0.5)
		self.play(
			ShowCreation(code_arrow)
		)

		self.play(
			FadeIn(stack_text)
		)

		self.play(
			ShowCreation(vert_line),
			ShowCreation(horiz_line)
		)

		stack_arrow = Arrow(
			vert_line.get_center(), 
			vert_line.get_center() + RIGHT * 1.5
		)

		self.play(
			ShowCreation(stack_arrow)
		)

		index_0 = Square(side_length=0.6)
		index_0.next_to(stack_arrow, RIGHT)
		index_0_data = TextMobject("0")
		index_0_data.scale(scale_factor)
		index_0_data.move_to(index_0.get_center())
		self.play(
			GrowFromCenter(index_0),
			FadeIn(index_0_data)
		)

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 3)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 4)

		v_text = TexMobject("v")
		v_text.scale(0.8)
		v_text.next_to(stack_text, DOWN * 1.5)
		v_text.shift(LEFT * 0.5)
		vert_line_2 = Line(v_text[-1].get_center() + RIGHT * 0.25 + UP * 0.25, 
			v_text[-1].get_center() + RIGHT * 0.25 + DOWN * 0.25)
		horiz_line_2 = Line(v_text[-1].get_center() + RIGHT * 0.25 + DOWN * 0.25, 
			v_text[-1].get_center() + RIGHT * 0.65 + DOWN * 0.25)

		self.play(
			FadeIn(v_text),
			FadeIn(vert_line_2),
			FadeIn(horiz_line_2)
		)

		v_value = index_0_data.copy()
		v_value.next_to(vert_line_2, RIGHT * 0.5)
		v_value.shift(UP * SMALL_BUFF / 2)

		self.play(
			FadeOut(index_0),
			ReplacementTransform(index_0_data, v_value)
		)

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 5)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 6)

		all_highlights = []

		dfs_full_order = dfs(graph, 0)
		wait_times = [0] * len(dfs_full_order)
		new_highlights = self.show_second_dfs_preorder(graph, edge_dict, dfs_full_order[:1], order[1], wait_times, scale_factor=scale_factor)
		all_highlights.extend(new_highlights)

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 7)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 8)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 9)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 10)

		
		index_0 = Square(side_length=0.6)
		index_0.next_to(stack_arrow, RIGHT)
		index_0_data = TextMobject("2")
		index_0_data.scale(scale_factor)
		index_0_data.move_to(index_0.get_center())

		index_1 = Square(side_length=0.6)
		index_1.next_to(index_0, RIGHT, buff=0)
		index_1_data = TextMobject("3")
		index_1_data.scale(scale_factor)
		index_1_data.move_to(index_1.get_center())

		index_2 = Square(side_length=0.6)
		index_2.next_to(index_1, RIGHT, buff=0)
		index_2_data = TextMobject("1")
		index_2_data.scale(scale_factor)
		index_2_data.move_to(index_2.get_center())

		self.play(
			GrowFromCenter(index_0),
			TransformFromCopy(graph[2].data, index_0_data)
		)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 8)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 9)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 10)
		self.play(
			GrowFromCenter(index_1),
			TransformFromCopy(graph[3].data, index_1_data)
		)

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 8)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 9)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 10)
		self.play(
			GrowFromCenter(index_2),
			TransformFromCopy(graph[1].data, index_2_data)
		)

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 3)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 4)

		next_v_value = index_2_data.copy()
		next_v_value.move_to(v_value.get_center())

		self.play(
			FadeOut(index_2),
			FadeOut(v_value),
			ReplacementTransform(index_2_data, next_v_value)
		)
		v_value = next_v_value
		
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 5)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 6)

		new_highlights = self.show_second_dfs_preorder(graph, edge_dict, dfs_full_order[1:3], order[1][1:], wait_times, scale_factor=scale_factor)
		all_highlights.extend(new_highlights)


		code_arrow = self.shift_arrow_to_line(code_arrow, code, 7)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 8)
		self.wait(5)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 9)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 8)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 9)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 10)
		self.wait(2)

		index_2 = Square(side_length=0.6)
		index_2.next_to(index_1, RIGHT, buff=0)
		index_2_data = TextMobject("3")
		index_2_data.scale(scale_factor)
		index_2_data.move_to(index_2.get_center())

		self.play(
			GrowFromCenter(index_2),
			TransformFromCopy(graph[3].data, index_2_data)
		)

		self.wait(8)

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 3)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 4)

		next_v_value = index_2_data.copy()
		next_v_value.move_to(v_value.get_center())

		self.play(
			FadeOut(index_2),
			FadeOut(v_value),
			ReplacementTransform(index_2_data, next_v_value)
		)
		v_value = next_v_value

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 5)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 6)

		new_highlights = self.show_second_dfs_preorder(graph, edge_dict, dfs_full_order[3:5], order[1][2:], wait_times, scale_factor=scale_factor)
		all_highlights.extend(new_highlights)

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 7)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 8)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 9)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 8)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 9)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 8)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 9)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 10)

		index_2 = Square(side_length=0.6)
		index_2.next_to(index_1, RIGHT, buff=0)
		index_2_data = TextMobject("2")
		index_2_data.scale(scale_factor)
		index_2_data.move_to(index_2.get_center())

		index_3 = Square(side_length=0.6)
		index_3.next_to(index_2, RIGHT, buff=0)
		index_3_data = TextMobject("4")
		index_3_data.scale(scale_factor)
		index_3_data.move_to(index_3.get_center())

		self.play(
			GrowFromCenter(index_2),
			TransformFromCopy(graph[2].data, index_2_data)
		)

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 8)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 9)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 10)

		self.play(
			GrowFromCenter(index_3),
			TransformFromCopy(graph[4].data, index_3_data)
		)
		
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 3)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 4)

		next_v_value = index_3_data.copy()
		next_v_value.move_to(v_value.get_center())

		self.play(
			FadeOut(index_3),
			FadeOut(v_value),
			ReplacementTransform(index_3_data, next_v_value)
		)
		v_value = next_v_value

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 5)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 6)

		new_highlights = self.show_second_dfs_preorder(graph, edge_dict, dfs_full_order[5:7], order[1][3:], wait_times, scale_factor=scale_factor)
		all_highlights.extend(new_highlights)
		

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 7)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 8)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 9)
		self.wait(6)

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 3)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 4)

		next_v_value = index_2_data.copy()
		next_v_value.move_to(v_value.get_center())

		self.play(
			FadeOut(index_2),
			FadeOut(v_value),
			ReplacementTransform(index_2_data, next_v_value)
		)
		v_value = next_v_value

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 5)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 6)

		new_highlights = self.show_second_dfs_preorder(graph, edge_dict, dfs_full_order[7:9], order[1][4:], wait_times, scale_factor=scale_factor)
		all_highlights.extend(new_highlights)

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 7)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 8)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 9)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 8)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 9)

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 3)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 4)

		next_v_value = index_1_data.copy()
		next_v_value.move_to(v_value.get_center())

		self.play(
			FadeOut(index_1),
			FadeOut(v_value),
			ReplacementTransform(index_1_data, next_v_value)
		)
		v_value = next_v_value

		self.wait()

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 5)

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 3)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 4)

		next_v_value = index_0_data.copy()
		next_v_value.move_to(v_value.get_center())

		self.play(
			FadeOut(index_0),
			FadeOut(v_value),
			ReplacementTransform(index_0_data, next_v_value)
		)

		self.wait()

		v_value = next_v_value
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 5)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 3)
		self.wait(3)

		self.play(
			FadeOut(code_arrow),
			FadeOut(stack_arrow),
			FadeOut(stack_text),
			FadeOut(vert_line),
			FadeOut(vert_line_2),
			FadeOut(horiz_line),
			FadeOut(horiz_line_2),
			FadeOut(v_text),
			FadeOut(v_value)
		)

		self.wait(12)
		other_fadeouts = [
		FadeOut(entire_graph),
		FadeOut(order),
		FadeOut(code),
		FadeOut(h_line),
		FadeOut(title),
		]

		self.play(
			*[FadeOut(highlight) for highlight in all_highlights] + other_fadeouts
		)
		self.wait()


	def shift_arrow_to_line(self, arrow, code, line_number, run_time=1):
		arrow_copy = arrow.copy()
		arrow_copy.next_to(code[line_number], LEFT * 0.5)

		self.play(
			Transform(arrow, arrow_copy),
			run_time=run_time
		)
		return arrow


	def generate_code(self):
		code_scale = 0.8
		
		code = []

		def_statement = TextMobject("def", r"$\text{dfs\_iter}(G, v):$")
		def_statement[0].set_color(MONOKAI_BLUE)
		def_statement[1][:8].set_color(MONOKAI_GREEN)
		def_statement[1][9].set_color(MONOKAI_ORANGE)
		def_statement[1][11].set_color(MONOKAI_ORANGE)
		def_statement.scale(code_scale)
		def_statement.to_edge(LEFT)

		marked_array = TextMobject(r"marked = [False] * $G$.size()")
		marked_array.scale(code_scale)
		marked_array.next_to(def_statement, UP * 0.5)
		marked_array.to_edge(LEFT)
		marked_array[14].shift(DOWN * SMALL_BUFF)
		marked_array[8:13].set_color(MONOKAI_PURPLE)
		marked_array[14].set_color(MONOKAI_PINK)
		marked_array[-6:-2].set_color(MONOKAI_BLUE)
		code.extend([marked_array, def_statement])

		line_1 = TextMobject(r"stack = [$v$]")
		line_1.scale(code_scale)
		line_1.next_to(def_statement, DOWN * 0.5)
		line_1.to_edge(LEFT * 2)
		code.append(line_1)

		line_2 = TextMobject(r"while len(stack) $>$ 0:")
		line_2.scale(code_scale)
		line_2.next_to(line_1, DOWN * 0.5)
		line_2.to_edge(LEFT * 2)
		line_2[:5].set_color(MONOKAI_PINK)
		line_2[5:8].set_color(MONOKAI_BLUE)
		line_2[-3].set_color(MONOKAI_PINK)
		line_2[-2].set_color(MONOKAI_PURPLE)
		code.append(line_2)

		line_3 = TextMobject(r"$v$ = stack.pop()")
		line_3.scale(code_scale)
		line_3.next_to(line_2, DOWN * 0.5)
		line_3.to_edge(LEFT * 3)
		line_3[-5:-2].set_color(MONOKAI_BLUE)
		code.append(line_3)

		line_4 = TextMobject(r"if not marked[$v$]:")
		line_4.scale(code_scale)
		line_4.next_to(line_3, DOWN * 0.5)
		line_4.to_edge(LEFT * 3)
		line_4[:5].set_color(MONOKAI_PINK)
		code.append(line_4)

		line_5 = TextMobject(r"visit($v$)")
		line_5[:5].set_color(MONOKAI_BLUE)
		line_5.scale(code_scale)
		line_5.next_to(line_4, DOWN * 0.5)
		line_5.to_edge(LEFT * 4)
		code.append(line_5)

		line_6 = TextMobject(r"marked[$v$] = True")
		line_6[-4:].set_color(MONOKAI_PURPLE)
		line_6.scale(code_scale)
		line_6.next_to(line_5, DOWN * 0.5)
		line_6.to_edge(LEFT * 4)
		code.append(line_6)

		line_7 = TextMobject(r"for $w$ in $G$.neighbors($v$):")
		line_7[:3].set_color(MONOKAI_PINK)
		line_7[4:6].set_color(MONOKAI_PINK)
		line_7[8:17].set_color(MONOKAI_BLUE)
		line_7.scale(code_scale)
		line_7.next_to(line_6, DOWN * 0.5)
		line_7.to_edge(LEFT * 4)
		code.append(line_7)

		line_8 = TextMobject(r"if not marked[$w$]:")
		line_8.scale(code_scale)
		line_8.next_to(line_7, DOWN * 0.5)
		line_8.to_edge(LEFT * 5)
		line_8[:5].set_color(MONOKAI_PINK)
		code.append(line_8)

		line_9 = TextMobject(r"stack.append($w$)")
		line_9[6:12].set_color(MONOKAI_BLUE)
		line_9.scale(code_scale)
		line_9.next_to(line_8, DOWN * 0.5)
		line_9.to_edge(LEFT * 6)
		code.append(line_9)

		code = VGroup(*code)
		code.scale(0.9)
		code.shift(UP * 1.5)
		self.play(
			Write(code[1])
		)
		self.wait(2)

		self.play(
			Write(code[0])
		)

		self.wait(8)

		self.play(
			Write(code[2])
		)

		self.wait()

		self.play(
			Write(code[3])
		)

		self.wait()
		self.play(
			Write(code[4])
		)

		self.wait()
		self.play(
			Write(code[5])
		)

		self.wait()
		self.play(
			Write(code[6])
		)

		self.play(
			Write(code[7])
		)

		self.wait()
		self.play(
			Write(code[8])
		)
		self.play(
			Write(code[9])
		)

		self.play(
			Write(code[10])
		)
		
		return code

class ImplementationComparison(Scene):
	def construct(self):
		title = TextMobject("DFS Implementation Comparison")
		title.scale(1.2)
		title.shift(UP * 3.5)

		h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
		h_line.next_to(title, DOWN)
		


		code_rec = self.generate_code_rec()
		code_rec.scale(0.8)
		code_rec.move_to(LEFT * 3.5)
		code_rec.to_edge(UP * 6)

		code_iter = self.generate_code_iter()
		code_iter.scale(0.8)
		code_iter.move_to(RIGHT * 3.5)
		code_iter.to_edge(UP * 6)

		rect_left = SurroundingRectangle(code_rec, buff=SMALL_BUFF, color=WHITE)

		rect_right = SurroundingRectangle(code_iter, buff=SMALL_BUFF, color=WHITE)

		self.play(
			Write(title),
			ShowCreation(h_line),
			FadeIn(code_rec),
			FadeIn(code_iter),
			ShowCreation(rect_left),
			ShowCreation(rect_right)
		)

		self.wait(8)

		both = TextMobject(r"Both run in $O(V + E)$")
		both.scale(0.8)
		both.next_to(h_line, DOWN)
		both.shift(DOWN * 0.3)
		rec = TextMobject(r"Cleaner and easier to read")
		rec.scale(0.8)
		rec.next_to(rect_left, UP)
		iter_version = TextMobject("More generalizable")
		iter_version.scale(0.8)
		iter_version.next_to(rect_right, UP)

		self.play(
			Write(both)
		)
		self.wait(18)

		self.play(
			Write(rec)
		)
		self.wait(8)

		self.play(
			Write(iter_version)
		)

		self.wait(10)

		self.play(
			FadeOut(rect_right),
			FadeOut(rect_left),
			FadeOut(rec),
			FadeOut(iter_version),
			FadeOut(both),
			FadeOut(code_iter),
			FadeOut(code_rec),
			FadeOut(h_line),
			FadeOut(title)
		)

	def generate_code_rec(self):
		code_scale = 0.8

		def_statement = TextMobject(r"def dfs($G, v$):", font='Inconsolata')
		def_statement[:3].set_color(MONOKAI_BLUE)
		def_statement[3:6].set_color(MONOKAI_GREEN)
		def_statement[7].set_color(MONOKAI_ORANGE)
		def_statement[9].set_color(MONOKAI_ORANGE)
		def_statement.scale(code_scale)
		def_statement.to_edge(LEFT)

		marked_array = TextMobject(r"marked = [False] * $G$.size()")
		marked_array.scale(code_scale)
		marked_array.next_to(def_statement, UP * 0.5)
		marked_array.to_edge(LEFT)
		marked_array[14].shift(DOWN * SMALL_BUFF)
		marked_array[8:13].set_color(MONOKAI_PURPLE)
		marked_array[14].set_color(MONOKAI_PINK)
		marked_array[-6:-2].set_color(MONOKAI_BLUE)
		
		line_1 = TextMobject(r"visit($v$)")
		line_1[:5].set_color(MONOKAI_BLUE)
		line_1.scale(code_scale)
		line_1.next_to(def_statement, DOWN * 0.5)
		line_1.to_edge(LEFT * 2)


		line_after_visit = TextMobject(r"marked[$v$] = True")
		line_after_visit[-4:].set_color(MONOKAI_PURPLE)
		line_after_visit.scale(0.8)
		line_after_visit.next_to(line_1, DOWN * 0.5)
		line_after_visit.to_edge(LEFT * 2)
		
		line_2 = TextMobject(r"for $w$ in $G$.neighbors($v$):")
		line_2[:3].set_color(MONOKAI_PINK)
		line_2[4:6].set_color(MONOKAI_PINK)
		line_2[8:17].set_color(MONOKAI_BLUE)
		line_2.scale(code_scale)
		line_2.next_to(line_after_visit, DOWN * 0.5)
		line_2.to_edge(LEFT * 2)

		line_after_for_loop = TextMobject(r"if not marked[$w$]:")
		line_after_for_loop.scale(code_scale)
		line_after_for_loop.next_to(line_2, DOWN * 0.5)
		line_after_for_loop.to_edge(LEFT * 3)
		line_after_for_loop[:5].set_color(MONOKAI_PINK)

		line_3 = TextMobject(r"dfs($G, w$)")
		line_3[:3].set_color(MONOKAI_BLUE)
		line_3.scale(code_scale)
		line_3.next_to(line_after_for_loop, DOWN * 0.5)
		line_3.to_edge(LEFT * 4)


		code = VGroup(
			marked_array, 
			def_statement, 
			line_1, 
			line_after_visit,
			line_2,
			line_after_for_loop,
			line_3,
		)

		return code

	def generate_code_iter(self):
		code_scale = 0.8
		
		code = []

		def_statement = TextMobject("def", r"$\text{dfs\_iter}(G, v)$")
		def_statement[0].set_color(MONOKAI_BLUE)
		def_statement[1][:8].set_color(MONOKAI_GREEN)
		def_statement[1][9].set_color(MONOKAI_ORANGE)
		def_statement[1][11].set_color(MONOKAI_ORANGE)
		def_statement.scale(code_scale)
		def_statement.to_edge(LEFT)

		marked_array = TextMobject(r"marked = [False] * $G$.size()")
		marked_array.scale(code_scale)
		marked_array.next_to(def_statement, UP * 0.5)
		marked_array.to_edge(LEFT)
		marked_array[14].shift(DOWN * SMALL_BUFF)
		marked_array[8:13].set_color(MONOKAI_PURPLE)
		marked_array[14].set_color(MONOKAI_PINK)
		marked_array[-6:-2].set_color(MONOKAI_BLUE)
		code.extend([marked_array, def_statement])

		line_1 = TextMobject(r"stack = [$v$]")
		line_1.scale(code_scale)
		line_1.next_to(def_statement, DOWN * 0.5)
		line_1.to_edge(LEFT * 2)
		code.append(line_1)

		line_2 = TextMobject(r"while len(stack) $>$ 0:")
		line_2.scale(code_scale)
		line_2.next_to(line_1, DOWN * 0.5)
		line_2.to_edge(LEFT * 2)
		line_2[:5].set_color(MONOKAI_PINK)
		line_2[5:8].set_color(MONOKAI_BLUE)
		line_2[-3].set_color(MONOKAI_PINK)
		line_2[-2].set_color(MONOKAI_PURPLE)
		code.append(line_2)

		line_3 = TextMobject(r"$v$ = stack.pop()")
		line_3.scale(code_scale)
		line_3.next_to(line_2, DOWN * 0.5)
		line_3.to_edge(LEFT * 3)
		line_3[-5:-2].set_color(MONOKAI_BLUE)
		code.append(line_3)

		line_4 = TextMobject(r"if not marked[$v$]:")
		line_4.scale(code_scale)
		line_4.next_to(line_3, DOWN * 0.5)
		line_4.to_edge(LEFT * 3)
		line_4[:5].set_color(MONOKAI_PINK)
		code.append(line_4)

		line_5 = TextMobject(r"visit($v$)")
		line_5[:5].set_color(MONOKAI_BLUE)
		line_5.scale(code_scale)
		line_5.next_to(line_4, DOWN * 0.5)
		line_5.to_edge(LEFT * 4)
		code.append(line_5)

		line_6 = TextMobject(r"marked[$v$] = True")
		line_6[-4:].set_color(MONOKAI_PURPLE)
		line_6.scale(code_scale)
		line_6.next_to(line_5, DOWN * 0.5)
		line_6.to_edge(LEFT * 4)
		code.append(line_6)

		line_7 = TextMobject(r"for $w$ in $G$.neighbors($v$):")
		line_7[:3].set_color(MONOKAI_PINK)
		line_7[4:6].set_color(MONOKAI_PINK)
		line_7[8:17].set_color(MONOKAI_BLUE)
		line_7.scale(code_scale)
		line_7.next_to(line_6, DOWN * 0.5)
		line_7.to_edge(LEFT * 4)
		code.append(line_7)

		line_8 = TextMobject(r"if not marked[$w$]:")
		line_8.scale(code_scale)
		line_8.next_to(line_7, DOWN * 0.5)
		line_8.to_edge(LEFT * 5)
		line_8[:5].set_color(MONOKAI_PINK)
		code.append(line_8)

		line_9 = TextMobject(r"stack.append($w$)")
		line_9[6:12].set_color(MONOKAI_BLUE)
		line_9.scale(code_scale)
		line_9.next_to(line_8, DOWN * 0.5)
		line_9.to_edge(LEFT * 6)
		code.append(line_9)

		code = VGroup(*code)
		
		return code

class PostOrder(GraphAnimationUtils):
	def construct(self):
		self.show_postorder()

	def show_postorder(self):
		title = TextMobject("Preorder vs Postorder")
		title.scale(1.2)
		title.shift(UP * 3.5)

		h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
		h_line.next_to(title, DOWN)
		self.play(
			Write(title),
			ShowCreation(h_line)
		)

		self.wait()

		graph, edge_dict = self.create_dfs_graph()
		nodes, edges = self.make_graph_mobject(graph, edge_dict)
		entire_graph = VGroup(nodes, edges)

		self.play(
			FadeIn(entire_graph)
		)

		self.wait()

		preorder = TextMobject("Preorder: 0 1 2 3 5 6 7 8 9 4")
		preorder.next_to(entire_graph, DOWN)

		postorder = TextMobject("Postorder: 2 6 9 8 7 5 3 4 1 0")
		postorder.next_to(preorder, DOWN)

		self.play(
			Write(preorder[:9]),
			Write(postorder[:10])
		)
		self.wait(5)

		all_highlights = []
		dfs_second_full_order = dfs(graph, 0)
		wait_times = [0] * len(dfs_second_full_order)
		
		new_highlights = self.show_second_dfs_preorder(graph, edge_dict, dfs_second_full_order[:5], preorder[9:], wait_times)
		self.wait(10)
		graph[2].surround_circle.set_color(BRIGHT_RED)
		self.play(
			TransformFromCopy(graph[2].data, postorder[10])
		)
		self.wait()
		all_highlights.extend(new_highlights)

		new_highlights = self.show_second_dfs_preorder(graph, edge_dict, dfs_second_full_order[5:11], preorder[12:], wait_times)
		self.wait()
		graph[6].surround_circle.set_color(BRIGHT_RED)
		self.play(
			TransformFromCopy(graph[6].data, postorder[11])
		)
		self.wait()
		all_highlights.extend(new_highlights)

		new_highlights = self.show_second_dfs_preorder(graph, edge_dict, dfs_second_full_order[11:-2], preorder[15:-1], wait_times)
		self.wait()
		graph[9].surround_circle.set_color(BRIGHT_RED)
		self.play(
			TransformFromCopy(graph[9].data, postorder[12])
		)
		self.wait()
		all_highlights.extend(new_highlights)

		graph[8].surround_circle.set_color(BRIGHT_RED)
		self.play(
			TransformFromCopy(graph[8].data, postorder[13])
		)
		self.wait(3)
		graph[7].surround_circle.set_color(BRIGHT_RED)
		self.play(
			TransformFromCopy(graph[7].data, postorder[14])
		)
		self.wait(2)
		graph[5].surround_circle.set_color(BRIGHT_RED)
		self.play(
			TransformFromCopy(graph[5].data, postorder[15])
		)
		self.wait()
		graph[3].surround_circle.set_color(BRIGHT_RED)
		self.play(
			TransformFromCopy(graph[3].data, postorder[16])
		)
		self.wait()

		new_highlights = self.show_second_dfs_preorder(graph, edge_dict, dfs_second_full_order[-2:], preorder[-1:], wait_times)
		all_highlights.extend(new_highlights)
		self.wait()
		graph[4].surround_circle.set_color(BRIGHT_RED)
		self.play(
			TransformFromCopy(graph[4].data, postorder[17])
		)
		self.wait()
		graph[1].surround_circle.set_color(BRIGHT_RED)
		self.play(
			TransformFromCopy(graph[1].data, postorder[18])
		)
		self.wait()
		graph[0].surround_circle.set_color(BRIGHT_RED)
		self.play(
			TransformFromCopy(graph[0].data, postorder[19])
		)
		self.wait(12)
		
		other_fadeouts = [
		FadeOut(entire_graph),
		FadeOut(preorder),
		FadeOut(postorder)
		]

		self.play(
			*[FadeOut(obj) for obj in all_highlights] + other_fadeouts
		)
		self.wait(3)

		code = self.generate_code_rec()
		code.move_to(LEFT * 3.5)
		code_scale = 0.8

		self.play(
			FadeIn(code)
		)

		self.wait(10)

		right_code = code.copy()
		right_code.move_to(RIGHT * 3.5)
		self.play(
			TransformFromCopy(code, right_code),
			run_time=2
		)
		self.wait()

		def_statement_pre = TextMobject("def", r"$\text{dfs\_pre}(G, v):$")
		def_statement_pre[0].set_color(MONOKAI_BLUE)
		def_statement_pre[1][:7].set_color(MONOKAI_GREEN)
		def_statement_pre[1][8].set_color(MONOKAI_ORANGE)
		def_statement_pre[1][10].set_color(MONOKAI_ORANGE)
		def_statement_pre.scale(code_scale)
		def_statement_pre.move_to(code[1].get_center())
		def_statement_pre.to_edge(LEFT * 2.55)
		self.play(
			Transform(code[1][3:], def_statement_pre[1])
		)

		def_statement_post = TextMobject("def", r"$\text{dfs\_post}(G, v):$")
		def_statement_post[0].set_color(MONOKAI_BLUE)
		def_statement_post[1][:8].set_color(MONOKAI_GREEN)
		def_statement_post[1][9].set_color(MONOKAI_ORANGE)
		def_statement_post[1][11].set_color(MONOKAI_ORANGE)
		def_statement_post.scale(code_scale)
		def_statement_post.move_to(right_code[1].get_center())
		def_statement_post.to_edge(LEFT * 2.55 + LEFT * 14)
		self.play(
			Transform(right_code[1][3:], def_statement_post[1])
		)
		self.wait(6)

		new_visit = right_code[2].copy()
		new_visit.next_to(right_code[-2], DOWN * 0.5)
		new_visit.to_edge(LEFT * 2.55 + LEFT * 14 + LEFT * 0.9)

		after_visit = VGroup(*right_code[3:])
		after_visit_copy = after_visit.copy()
		after_visit_copy.shift(UP * 0.5)
		self.play(
			Transform(right_code[2], new_visit),
			Transform(after_visit, after_visit_copy),
			run_time=2
		)

		self.wait(8)

	def generate_code_rec(self):
		code_scale = 0.8

		def_statement = TextMobject(r"def dfs($G, v$):", font='Inconsolata')
		def_statement[:3].set_color(MONOKAI_BLUE)
		def_statement[3:6].set_color(MONOKAI_GREEN)
		def_statement[7].set_color(MONOKAI_ORANGE)
		def_statement[9].set_color(MONOKAI_ORANGE)
		def_statement.scale(code_scale)
		def_statement.to_edge(LEFT)

		marked_array = TextMobject(r"marked = [False] * $G$.size()")
		marked_array.scale(code_scale)
		marked_array.next_to(def_statement, UP * 0.5)
		marked_array.to_edge(LEFT)
		marked_array[14].shift(DOWN * SMALL_BUFF)
		marked_array[8:13].set_color(MONOKAI_PURPLE)
		marked_array[14].set_color(MONOKAI_PINK)
		marked_array[-6:-2].set_color(MONOKAI_BLUE)
		
		line_1 = TextMobject(r"visit($v$)")
		line_1[:5].set_color(MONOKAI_BLUE)
		line_1.scale(code_scale)
		line_1.next_to(def_statement, DOWN * 0.5)
		line_1.to_edge(LEFT * 2)


		line_after_visit = TextMobject(r"marked[$v$] = True")
		line_after_visit[-4:].set_color(MONOKAI_PURPLE)
		line_after_visit.scale(0.8)
		line_after_visit.next_to(line_1, DOWN * 0.5)
		line_after_visit.to_edge(LEFT * 2)
		
		line_2 = TextMobject(r"for $w$ in $G$.neighbors($v$):")
		line_2[:3].set_color(MONOKAI_PINK)
		line_2[4:6].set_color(MONOKAI_PINK)
		line_2[8:17].set_color(MONOKAI_BLUE)
		line_2.scale(code_scale)
		line_2.next_to(line_after_visit, DOWN * 0.5)
		line_2.to_edge(LEFT * 2)

		line_after_for_loop = TextMobject(r"if not marked[$w$]:")
		line_after_for_loop.scale(code_scale)
		line_after_for_loop.next_to(line_2, DOWN * 0.5)
		line_after_for_loop.to_edge(LEFT * 3)
		line_after_for_loop[:5].set_color(MONOKAI_PINK)

		line_3 = TextMobject(r"dfs($G, w$)")
		line_3[:3].set_color(MONOKAI_BLUE)
		line_3.scale(code_scale)
		line_3.next_to(line_after_for_loop, DOWN * 0.5)
		line_3.to_edge(LEFT * 4)


		code = VGroup(
			marked_array, 
			def_statement, 
			line_1, 
			line_after_visit,
			line_2,
			line_after_for_loop,
			line_3,
		)

		return code

class DFSApplications(GraphAnimationUtils):
	def construct(self):
		title = TextMobject("DFS Applications")
		title.scale(1.2)
		title.shift(UP * 3.5)

		h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
		h_line.next_to(title, DOWN)
		self.play(
			Write(title),
			ShowCreation(h_line)
		)

		self.wait()

		self.show_cycle_detection(h_line)

		self.show_connected_component(h_line)

		self.show_topological_sort(h_line)

	def show_cycle_detection(self, h_line):
		subtitle = TextMobject("Cycle Detection")
		subtitle.next_to(h_line, DOWN)
		self.wait(6)
		self.play(
			Write(subtitle)
		)

		graph, edge_dict = self.create_small_graph()
		nodes, edges = self.make_graph_mobject(graph, edge_dict)
		entire_graph = VGroup(nodes, edges)
		scale_factor = 0.9
		entire_graph.scale(scale_factor)
		entire_graph.shift(UP * 0.5)
		self.play(
			FadeIn(entire_graph)
		)

		self.wait(2)

		objects = self.show_path(graph, edge_dict, [0, 3, 2, 0], scale_factor=scale_factor)

		self.wait(2)

		fadeouts = [FadeOut(obj) for obj in objects]
		self.play(
			*fadeouts + [FadeOut(entire_graph), FadeOut(subtitle)]
		)

	def show_connected_component(self, h_line):
		subtitle = TextMobject("Finding Connected Components")
		subtitle.next_to(h_line, DOWN)
		self.play(
			Write(subtitle)
		)

		scale_factor = 0.9
		graph, edge_dict = self.create_disconnected_graph()
		nodes, edges = self.make_graph_mobject(graph, edge_dict, scale_factor=scale_factor)
		entire_graph = VGroup(nodes, edges)
		entire_graph.shift(DOWN * 1)
		self.play(
			FadeIn(entire_graph)
		)

		self.wait(2)

		dfs_component_1 = dfs(graph, 0)

		wait_times = [0] * len(dfs_component_1)

		highlights_1 = self.show_dfs_preorder(graph, edge_dict, dfs_component_1, wait_times, 
			scale_factor=scale_factor, run_time=1)

		component_1 = VGroup(
			nodes[0], nodes[4], nodes[6], nodes[7], nodes[8]
		)

		surround_rectangle_1 = SurroundingRectangle(component_1, color=GREEN_SCREEN, buff=SMALL_BUFF)
		self.play(
			ShowCreation(surround_rectangle_1)
		)

		dfs_component_2 = dfs(graph, 2)

		wait_times = [0] * len(dfs_component_2)

		highlights_2 = self.show_dfs_preorder(graph, edge_dict, dfs_component_2, wait_times, 
			scale_factor=scale_factor, run_time=1)

		component_2 = VGroup(
			nodes[1], nodes[2], nodes[3], nodes[5]
		)

		surround_rectangle_2 = SurroundingRectangle(component_2, color=GREEN_SCREEN, buff=SMALL_BUFF)
		self.play(
			ShowCreation(surround_rectangle_2)
		)
		self.wait(5)

		all_highlight_fadeouts = [FadeOut(h) for h in highlights_1 + highlights_2]
		
		other_fadeouts = [
			FadeOut(surround_rectangle_1),
			FadeOut(surround_rectangle_2),
			FadeOut(entire_graph),
			FadeOut(subtitle)
		]
		self.play(
			*all_highlight_fadeouts + other_fadeouts
		)

	def show_topological_sort(self, h_line):
		self.wait()
		subtitle = TextMobject("Topological Sort")
		subtitle.next_to(h_line, DOWN)
		self.play(
			Write(subtitle)
		)
		self.wait(3)

		graph, edge_dict = self.create_dfs_graph_directed()
		nodes, edges = self.make_graph_mobject(graph, edge_dict, directed=True)
		entire_graph = VGroup(nodes, edges)
		scale_factor = 0.9
		entire_graph.scale(scale_factor)
		entire_graph.next_to(subtitle, DOWN)
		self.play(
			FadeIn(entire_graph)
		)

		self.wait(3)

		top_sort = TextMobject("Topological Sort: 0 1 4 3 5 7 8 9 6 2")
		top_sort.scale(0.8)
		top_sort.next_to(entire_graph, DOWN)
		dfs_postorder = TextMobject("DFS Postorder:", "2 6 9 8 7 5 3 4 1 0")
		dfs_postorder.scale(0.8)
		dfs_postorder.next_to(top_sort, DOWN)

		surround_circles = [self.highlight_node(graph, i, scale_factor=scale_factor, animate=False) for i in range(len(graph))]
		top_order = [0, 1, 4, 3, 5, 7, 8, 9, 6, 2]

		self.play(
			Write(top_sort[:16]),
		)
		self.wait(3)

		wait_time_dict = {}
		for i in range(10):
			wait_time_dict[i] = 0

		wait_time_dict[1] = 17

		for i, node_id in enumerate(top_order[:2]):
			self.play(
				ShowCreation(surround_circles[node_id])
			)
			self.wait(wait_time_dict[node_id])

		self.play(
			TransformFromCopy(graph[0].data, top_sort[16])
		)

		self.wait(2)

		self.play(
			TransformFromCopy(graph[1].data, top_sort[17])
		)

		wait_time_dict[5] = 7

		for i, node_id in enumerate(top_order[2:]):
			i += 2
			self.play(
				ShowCreation(surround_circles[node_id])
			)

			self.play(
				TransformFromCopy(graph[node_id].data, top_sort[i + 16])
			)

			self.wait(wait_time_dict[node_id])

		self.wait(3)

		self.play(
			*[FadeOut(c) for c in surround_circles]
		)

		self.play(
			Write(dfs_postorder[0])
		)

		self.wait()

		dfs_pre_order = [0, 1, 2, 3, 5, 6, 7, 8, 9, 4]
		dfs_post_order = [2, 6, 9, 8, 7, 5, 3, 4, 1, 0]

		post_order_i = 0
		for node_id in dfs_pre_order:
			self.play(
				ShowCreation(surround_circles[node_id])
			)
			if node_id == dfs_post_order[post_order_i]:
				surround_circles[node_id].set_color(BRIGHT_RED)
				self.play(
					TransformFromCopy(graph[node_id].data, dfs_postorder[1][post_order_i])
				)
				post_order_i += 1
			if node_id == 9:
				break
		
		for post_order_i in range(3, 7):
			node_id = dfs_post_order[post_order_i]
			surround_circles[node_id].set_color(BRIGHT_RED)
			self.play(
				TransformFromCopy(graph[node_id].data, dfs_postorder[1][post_order_i])
			)

		self.play(
			ShowCreation(surround_circles[4])
		)
		for post_order_i in range(7, len(dfs_post_order)):
			node_id = dfs_post_order[post_order_i]
			surround_circles[node_id].set_color(BRIGHT_RED)
			self.play(
				TransformFromCopy(graph[node_id].data, dfs_postorder[1][post_order_i])
			)

		self.wait(5)


		reverse_dfs_postorder = TextMobject("Reverse DFS Postorder:", "0 1 4 3 5 7 8 9 6 2")
		reverse_dfs_postorder.scale(0.8)
		reverse_dfs_postorder.move_to(dfs_postorder.get_center())
		self.play(
			ReplacementTransform(dfs_postorder[0], reverse_dfs_postorder[0]),
			ReplacementTransform(dfs_postorder[1], reverse_dfs_postorder[1])
		)

		self.wait(13)

		highlight_fadeouts = [FadeOut(c) for c in surround_circles]
		other_fadeouts = [
		FadeOut(subtitle),
		FadeOut(top_sort),
		FadeOut(reverse_dfs_postorder),
		FadeOut(entire_graph),
		]
		self.play(
			*highlight_fadeouts + other_fadeouts
		)


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
	
	def sharpie_edge(self, edge_dict, u, v, color=GREEN_SCREEN, scale_factor=1, animate=True, directed=False, run_time=1):
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
				ShowCreation(line),
				run_time=run_time
			)
		return line


class MazeGeneration(GraphAnimationUtils):
	def construct(self):
		self.wait(2)
		title = TextMobject("Maze Generation")
		title.scale(1.2)
		title.shift(UP * 3.5)

		h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
		h_line.next_to(title, DOWN)
		self.play(
			Write(title),
			ShowCreation(h_line)
		)

		self.wait()

		self.show_maze_generation()

	def show_maze_generation(self):
		self.wait(9)
		rows, cols = 18, 36
		grid = self.create_grid(rows, cols, 0.3)
		grid_group = self.get_group_grid(grid)
		grid_group.move_to(DOWN * 0.5)
		self.play(
			FadeIn(grid_group)
		)

		self.wait()

		graph, edge_dict = self.make_graph_from_grid(grid)
		nodes, edges = self.make_graph_mobject(graph, edge_dict, scale_factor=0.5, show_data=False)
		entire_graph = VGroup(nodes, edges)
		self.play(
			FadeOut(grid_group)
		)

		self.play(
			FadeIn(entire_graph)
		)

		self.wait(3)

		self.play(
			FadeOut(entire_graph)
		)

		dfs_full_order = dfs_maze(graph, 0)
		wait_times = [0] * len(dfs_full_order)

		self.show_dfs_maze_preorder(graph, edge_dict, dfs_full_order, wait_times, rows, cols,
			scale_factor=0.8, run_time=0.1, color=WHITE)

		self.wait(10)

	def show_dfs_maze_preorder(self, graph, edge_dict, full_order, 
		wait_times, rows, cols, scale_factor=1, run_time=1, color=GREEN_SCREEN):
		i = 0
		angle = 180
		all_highlights = []
		for element in full_order:
			if isinstance(element, int):
				continue
			else:
				last_edge = self.sharpie_edge_maze(graph, element[0], element[1], rows, cols, 
					scale_factor=scale_factor, run_time=run_time, color=color)
				all_highlights.append(last_edge)
			self.wait(wait_times[i])
			i += 1
		return all_highlights

	def sharpie_edge_maze(self, graph, u, v, rows, cols, color=GREEN_SCREEN, 
		scale_factor=1, animate=True, run_time=1):
		switch = False
		if u > v:
			switch = True

		if u - v == cols:
			orientation = UP
		elif u - v == -cols:
			orientation = DOWN
		elif u - v == 1:
			orientation = LEFT
		else:
			orientation = RIGHT

		if not switch:
			start = graph[u].circle.get_center() - orientation * graph[u].circle.radius
			end = graph[v].circle.get_center() + orientation * graph[v].circle.radius
			line = Line(start, end)
		else:
			start = graph[v].circle.get_center() + orientation * graph[u].circle.radius
			end = graph[u].circle.get_center() - orientation * graph[u].circle.radius
			line = Line(start, end)
		line.set_stroke(width=16 * scale_factor)
		line.set_color(color)
		if animate:
			self.play(
				ShowCreation(line),
				run_time=run_time
			)
		return line

	def make_graph_from_grid(self, grid):
		graph = []
		edge_dict = {}
		radius, scale = 0.05, 0.3
		node_id = 0
		for i in range(len(grid)):
			for j in range(len(grid[0])):
				node = GraphNode(node_id, position=grid[i][j].get_center(), radius=radius, scale=scale)
				graph.append(node)
				node_id += 1

		for i in range(node_id - 1):
			if i % len(grid[0]) != len(grid[0]) - 1:
				edge_dict[(i, i + 1)] = graph[i].connect(graph[i + 1])

		for i in range(len(grid) - 1):
			for j in range(len(grid[0])):
				edge_dict[(i * len(grid[0]) + j, (i + 1) * len(grid[0]) + j)] = graph[i * len(grid[0]) + j].connect(graph[(i + 1) * len(grid[0]) + j])

		return graph, edge_dict

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

class TestDFSFunctions(GraphAnimationUtils):
	def construct(self):
		graph, edge_dict = self.create_dfs_graph()
		nodes, edges = self.make_graph_mobject(graph, edge_dict, scale_factor=1)
		entire_graph = VGroup(nodes, edges)
		self.play(
			FadeIn(entire_graph)
		)

		dfs_post(graph, 0)

class Logo(Scene):
	def construct(self):
		hexagon = RegularPolygonDepreciated(n=6, color=VIOLET, fill_color=VIOLET, fill_opacity=1, sheen_factor=0.2)
		hexagon.scale(2)
		hexagon.rotate(TAU / 4)
		vertices = hexagon.get_vertices()
		print(np.linalg.norm(vertices[0] - vertices[1]))
		print(np.linalg.norm(vertices[1] - vertices[2]))


		triangle = RegularPolygonDepreciated(color=LIGHT_VIOLET, fill_color=LIGHT_VIOLET, fill_opacity=1, sheen_factor=0.2)
		triangle.scale(np.sqrt(3) / 3)
		triangle_v = triangle.get_vertices()
		shift = vertices[0] - triangle_v[1]
		triangle.shift(RIGHT * shift[0] + UP * shift[1])

		triangles = [triangle]

		print(np.linalg.norm(triangle_v[0] - triangle_v[1]))
		print(np.linalg.norm(triangle_v[1] - triangle_v[2]))

		start_v = vertices[0]
		prev_v = vertices[0]
		for v in vertices[1:len(vertices)]:
			unit_v = Line(prev_v, v).get_unit_vector()
			new_triangle = triangles[-1].copy()
			new_center = new_triangle.get_center() + unit_v
			new_triangle.move_to(new_center)
			triangles.append(new_triangle)
			prev_v = v


		for i in range(len(triangles)):
			if i != 1 and i != 5:
				new_triangle = triangles[i].copy()
				new_triangle.shift(DOWN)
				triangles.append(new_triangle)

		top_left_vector = Line(vertices[0], vertices[1]).get_unit_vector()

		top_left_triangle = triangles[1].copy()
		bottom_left_triangle = triangles[2].copy()
		# self.play(FadeIn(top_left_triangle), FadeIn(bottom_left_triangle))
		top_left_triangle_center = top_left_triangle.get_center() + top_left_vector
		bottom_left_triangle_center = bottom_left_triangle.get_center() + top_left_vector

		top_left_triangle.move_to(top_left_triangle_center)
		bottom_left_triangle.move_to(bottom_left_triangle_center)

		triangles.append(top_left_triangle)
		triangles.append(bottom_left_triangle)

		reducible = ImageMobject("Reducible")
		reducible.scale(0.3)
		reducible.shift(DOWN * 2.5)

		logo = VGroup(*[hexagon] + triangles)
		logo.shift(LEFT * 2.5)
		subscribe = TextMobject("Subscribe")
		subscribe.move_to(LEFT * 2.5 + UP * 2.5)
		reducible.shift(LEFT * 2.5)

		patreon = ImageMobject("Patreon")
		support = TextMobject("Support")
		support.move_to(RIGHT * 2.5 + UP * 2.5)
		patreon.scale(1.8)
		patreon.shift(RIGHT * 2.5)

		# all_anims = [DrawBorderThenFill(hexagon)] + [DrawBorderThenFill(triangle) for triangle in triangles] + [FadeInAndShiftFromDirection(reducible)]
		self.wait()

		self.play(DrawBorderThenFill(logo), 
			FadeInFromDown(reducible), 
			FadeIn(patreon),
			run_time=2)

		self.play(Write(subscribe), Write(support))

		self.wait(18)

def dfs(graph, start):
	"""
	Returns a list of vertices and edges in preorder traversal
	"""
	dfs_order = []
	marked = [False] * len(graph)
	edge_to = [None] * len(graph)

	stack = [start]
	while len(stack) > 0:
		node = stack.pop()
		if not marked[node]:
			marked[node] = True
			dfs_order.append(node)
		for neighbor in graph[node].neighbors:
			neighbor_node = int(neighbor.char)
			if not marked[neighbor_node]:
				edge_to[neighbor_node] = node
				stack.append(neighbor_node)

	print(dfs_order)
	dfs_full_order = []
	for i in range(len(dfs_order) - 1):
		prev, curr = dfs_order[i], dfs_order[i + 1]
		dfs_full_order.append(prev)
		dfs_full_order.append((edge_to[curr], curr))

	dfs_full_order.append(curr)
	print(dfs_full_order)
	return dfs_full_order

def dfs_maze(graph, start):
	"""
	Returns a list of vertices and edges in preorder traversal
	"""
	dfs_order = []
	marked = [False] * len(graph)
	edge_to = [None] * len(graph)

	stack = [start]
	while len(stack) > 0:
		node = stack.pop()
		if not marked[node]:
			marked[node] = True
			dfs_order.append(node)
		neighbor_nodes = []
		for neighbor in graph[node].neighbors:
			neighbor_node = int(neighbor.char)
			if not marked[neighbor_node]:
				edge_to[neighbor_node] = node
				neighbor_nodes.append(neighbor_node)
		random.shuffle(neighbor_nodes)
		stack.extend(neighbor_nodes)

	print(dfs_order)
	dfs_full_order = []
	for i in range(len(dfs_order) - 1):
		prev, curr = dfs_order[i], dfs_order[i + 1]
		dfs_full_order.append(prev)
		dfs_full_order.append((edge_to[curr], curr))

	dfs_full_order.append(curr)
	print(dfs_full_order)
	return dfs_full_order

marked = [False] * 10
def dfs_test_1(graph, v):
	print(v)
	marked[v] = True
	for node in graph[v].neighbors:
		w = int(node.char)
		if not marked[w]:
			dfs_test_1(graph, w)

def dfs_test_2(graph, v):
	stack = [v]
	while len(stack) > 0:
		v = stack.pop()
		if not marked[v]:
			print(v)
			marked[v] = True
		for node in graph[v].neighbors:
			w = int(node.char)
			if not marked[w]:
				stack.append(w)

def dfs_post(graph, v):
	marked[v] = True
	for node in graph[v].neighbors:
		w = int(node.char)
		if not marked[w]:
			dfs_post(graph, w)

	print(v)





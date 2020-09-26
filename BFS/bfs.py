from manimlib.imports import *
import random
np.random.seed(0)

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
		return VGroup(*nodes), VGroup(*edges)

class Introduction(GraphAnimationUtils):
	def construct(self):
		graph, edge_dict = self.create_huge_graph()
		nodes, edges = self.make_graph_mobject(graph, edge_dict, show_data=False)
		entire_graph = VGroup(nodes, edges)

		self.play(
			*[GrowFromCenter(node) for node in nodes],
			run_time=3
		)

		self.wait()

		self.play(
			*[GrowFromCenter(edge) for edge in edges],
			run_time=3
		)

		self.highlight_node(graph, 35)

		self.wait(4)

		bfs_full_order = bfs_random(graph, 35)
		node_ids = [i for i in bfs_full_order[1:] if isinstance(i, int)]
		wait_times = [0] * len(bfs_full_order)
		wait_time_dict = {}
		for i in range(len(graph)):
			wait_time_dict[i] = 0

		flashes = VGroup()
		for node_id in node_ids:
			circle = nodes[node_id].copy()
			circle.set_fill(opacity=0)
			circle.set_stroke(width=7)
			circle.set_stroke(color=GREEN_SCREEN)
			flashes.add(circle)

		self.play(
			LaggedStartMap(
			ShowCreationThenDestruction, flashes,
			run_time=3
			)
		)

		highlights = self.show_bfs(graph, edge_dict, bfs_full_order[1:], wait_times)
		self.wait(2)

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

		title = TextMobject("Breadth First Search (BFS)")
		title.scale(1.2)
		title.shift(UP * 3.5)

		h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
		h_line.next_to(title, DOWN)

		definition = TextMobject("1. Intuition behind BFS")
		definition.next_to(rectangle, DOWN)

		representation = TextMobject("2. BFS Implementation")
		representation.next_to(rectangle, DOWN)

		problems = TextMobject("3. BFS Problem")
		problems.next_to(rectangle, DOWN)

		rectangle.scale(0.6)
		rectangle.move_to(RIGHT * 3)

		self.wait(4)
		self.play(
			Write(title),
			ShowCreation(h_line)
		)

		self.play(
			ShowCreation(rectangle)
		)

		self.wait()
		self.play(
			Write(definition)
		)

		self.wait(3)
		
		self.play(
			ReplacementTransform(definition, representation),
		)

		self.wait(4)
		self.play(
			ReplacementTransform(representation, problems),
		)
		self.wait(4)

		# self.play(
		# 	FadeOut(title),
		# 	FadeOut(problems),
		# 	FadeOut(h_line),
		# 	FadeOut(rectangle),
		# )

class BFSIntuitionPart1(GraphAnimationUtils):
	def construct(self):
		self.normal_to_string_graph()

	def normal_to_string_graph(self):
		graph, edge_dict = self.create_small_bfs_graph()
		nodes, edges = self.make_graph_mobject(graph, edge_dict)
		entire_graph = VGroup(nodes, edges)


		self.wait()
		self.play(
			*[GrowFromCenter(node) for node in nodes]
		)

		self.wait()

		self.play(
			*[GrowFromCenter(edge) for edge in edges]
		)

		self.wait(3)

		string_edge_dict = {}
		for key in edge_dict:
			string_edge_dict[key] = self.create_string_from_edge(edge_dict[key])

		self.play(
			*[ReplacementTransform(edge_dict[key], string_edge_dict[key]) for key in edge_dict],
			run_time=2
		)

		self.wait(3)

		t_graph, t_edge_dict = self.create_tree_like_graph()
		t_nodes, t_edges = self.make_graph_mobject(t_graph, t_edge_dict)
		t_entire_graph = VGroup(t_nodes, t_edges)

		start = self.highlight_node(graph, 0)

		new_start = self.highlight_node(t_graph, 0, animate=False) 

		self.wait(3)
		self.play(
			*[ReplacementTransform(old, new) for old, new in zip(nodes, t_nodes)],
			*[ReplacementTransform(string_edge_dict[key], t_edge_dict[key]) for key in string_edge_dict],
			ReplacementTransform(start, new_start),
			run_time=3
		)

		self.wait(7)

		highlights = self.do_bfs(t_graph, t_edge_dict)
		highlights = [new_start] + highlights
		

		self.wait(7)

		self.play(
			*[FadeOut(obj) for obj in highlights]
		)

		self.play(
			FadeOut(t_entire_graph)
		)

		

	def do_bfs(self, graph, edge_dict):
		bfs_full_order = bfs(graph, 0)
		wait_times = [0] * len(bfs_full_order)
		highlights = self.show_bfs(graph, edge_dict, bfs_full_order[1:], wait_times)

		return highlights

	def create_string_from_edge(self, edge):
		path = VMobject()
		points = [edge.point_from_proportion(m) for m in np.arange(0, 1.1, 0.25)]
		points[1] += edge.get_unit_normals()[0] * 0.2
		points[3] += edge.get_unit_normals()[1] * 0.2
		path.set_points_smoothly([*points])
		path.set_color(GRAY)
		return path

	def create_small_bfs_graph(self):
		graph = []
		edges = {}

		radius, scale = 0.4, 0.9
		node_0 = GraphNode('0', position=UP * 1.5, radius=radius, scale=scale)
		node_1 = GraphNode('1', position=RIGHT * 3 + UP * 1.5, radius=radius, scale=scale)
		node_2 = GraphNode('2', position=LEFT * 3 + UP * 1.5, radius=radius, scale=scale)
		node_3 = GraphNode('3', position=DOWN * 1.5, radius=radius, scale=scale)
		node_4 = GraphNode('4', position=LEFT * 3 + DOWN * 1.5, radius=radius, scale=scale)
		node_5 = GraphNode('5', position=RIGHT * 3 + DOWN * 1.5, radius=radius, scale=scale)

		edges[(0, 4)] = node_0.connect(node_4)
		edges[(0, 3)] = node_0.connect(node_3)
		edges[(0, 2)] = node_0.connect(node_2)
		edges[(0, 1)] = node_0.connect(node_1)

		edges[(1, 5)] = node_1.connect(node_5)

		edges[(2, 4)] = node_2.connect(node_4)

		edges[(3, 5)] = node_3.connect(node_5)

		graph.append(node_0)
		graph.append(node_1)
		graph.append(node_2)
		graph.append(node_3)
		graph.append(node_4)
		graph.append(node_5)

		return graph, edges

	def create_tree_like_graph(self):
		graph = []
		edges = {}

		radius, scale = 0.4, 0.9
		node_0 = GraphNode('0', position=UP * 2.5, radius=radius, scale=scale)
		node_1 = GraphNode('1', position=RIGHT * 3, radius=radius, scale=scale)
		node_2 = GraphNode('2', position=LEFT * 3, radius=radius, scale=scale)
		node_3 = GraphNode('3', position=RIGHT * 1, radius=radius, scale=scale)
		node_4 = GraphNode('4', position=LEFT * 1, radius=radius, scale=scale)
		node_5 = GraphNode('5', position=RIGHT * 2 + DOWN * 2.5, radius=radius, scale=scale)

		edges[(0, 4)] = node_0.connect(node_4)
		edges[(0, 3)] = node_0.connect(node_3)
		edges[(0, 2)] = node_0.connect(node_2)
		edges[(0, 1)] = node_0.connect(node_1)

		edges[(1, 5)] = node_1.connect(node_5)

		edges[(2, 4)] = node_2.connect_curve(node_4)

		edges[(3, 5)] = node_3.connect(node_5)

		graph.append(node_0)
		graph.append(node_1)
		graph.append(node_2)
		graph.append(node_3)
		graph.append(node_4)
		graph.append(node_5)

		return graph, edges

class BFSIntuitionPart2(GraphAnimationUtils):
	def construct(self):
		self.show_bfs_intuition()

	def show_bfs_intuition(self):
		bfs_intuition_title = TextMobject("BFS Intuition")
		bfs_intuition_title.scale(1.2)
		bfs_intuition_title.move_to(UP * 3.5)
		h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
		h_line.next_to(bfs_intuition_title, DOWN)
		

		graph, edge_dict = self.create_bfs_graph()
		nodes, edges = self.make_graph_mobject(graph, edge_dict)
		entire_graph = VGroup(nodes, edges)

		self.play(
			Write(bfs_intuition_title),
			ShowCreation(h_line)
		)

		self.play(
			ShowCreation(entire_graph),
			run_time=2
		)

		bfs_full_order = bfs(graph, 0)
		wait_times = [0] * len(bfs_full_order)
		wait_time_dict = {}
		for i in range(len(graph)):
			wait_time_dict[i] = 0

		order = TextMobject("Order: 0 1 2 3 4 5 6 7 8 9")
		order.next_to(entire_graph, DOWN)
		self.play(
			Write(order[0][:6])
		)

		all_highlights = []

		new_highlights = self.show_second_bfs(graph, edge_dict, bfs_full_order[:1], order[0][6:], wait_times)
		all_highlights.extend(new_highlights)

		self.indicate_neighbors(graph, 0, wait_time_dict)

		new_highlights = self.show_second_bfs(graph, edge_dict, bfs_full_order[1:5], order[0][7:], wait_times)
		all_highlights.extend(new_highlights)

		self.indicate_neighbors(graph, 1, wait_time_dict)

		new_highlights = self.show_second_bfs(graph, edge_dict, bfs_full_order[5:9], order[0][9:], wait_times)
		all_highlights.extend(new_highlights)
	
		self.indicate_neighbors(graph, 3, wait_time_dict)

		new_highlights = self.show_second_bfs(graph, edge_dict, bfs_full_order[9:11], order[0][11:], wait_times)
		all_highlights.extend(new_highlights)

		self.indicate_neighbors(graph, 5, wait_time_dict)

		new_highlights = self.show_second_bfs(graph, edge_dict, bfs_full_order[11:15], order[0][12:], wait_times)
		all_highlights.extend(new_highlights)

		path_text, path_highlights = self.show_shortest_path(all_highlights)
		path_text.scale(0.9)
		path_text.next_to(order, DOWN)
		self.play(
			Write(path_text[0][:16])
		)

		for h in path_highlights:
			self.play(
				ShowCreation(h)
			)

		self.play(
			Write(path_text[0][16:])
		)

		self.wait(3)

		self.play(
			*[FadeOut(h) for h in path_highlights]
		)

		self.wait()

		new_highlights = self.show_second_bfs(graph, edge_dict, bfs_full_order[15:17], order[0][14:], wait_times)
		all_highlights.extend(new_highlights)

		self.indicate_neighbors(graph, 8, wait_time_dict)

		new_highlights = self.show_second_bfs(graph, edge_dict, bfs_full_order[17:], order[0][15:], wait_times)
		all_highlights.extend(new_highlights)

		self.wait(9)

	def show_shortest_path(self, highlights):
		text = TextMobject(r"Shortest path $0 \rightarrow 7$: 0 1 3 5 7")
		text.set_color(YELLOW)
		path_highlights = []
		indices = [0, 1, 2, 5, 6, 9, 10, 13, 14]
		for i in indices:
			highlight = highlights[i].copy().set_color(YELLOW)
			path_highlights.append(highlight)
		return text, path_highlights

class BFSImplementation(GraphAnimationUtils):
	def construct(self):
		self.iterative_implementation()


	def iterative_implementation(self):
		title = TextMobject("BFS Implementation")
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

		self.wait(2)

		scale_factor = 0.8
		entire_graph_copy = entire_graph.copy()
		entire_graph_copy.scale(scale_factor)
		entire_graph_copy.shift(UP * 0.5)
		self.play(
			Transform(entire_graph, entire_graph_copy)
		)

		order = TextMobject(r"$\text{bfs}(G, 0)$:", " 0 1 3 2 4")
		order.next_to(entire_graph, DOWN)
		self.play(
			Write(order[0])
		)

		queue_text = TextMobject("queue")
		queue_text.scale(0.8)
		queue_text.move_to(RIGHT * 0.5 + DOWN * 2.8)
		vert_line = Line(queue_text[0][-1].get_center() + RIGHT * 0.25 + UP * 0.25, 
			queue_text[0][-1].get_center() + RIGHT * 0.25 + DOWN * 0.25)
		horiz_line = Line(queue_text[0][-1].get_center() + RIGHT * 0.25 + DOWN * 0.25, 
			queue_text[0][-1].get_center() + RIGHT * 0.65 + DOWN * 0.25)

		code_arrow = Arrow(ORIGIN, RIGHT * 1.2)
		code_arrow.set_color(GREEN_SCREEN)
		code_arrow.next_to(code[2], LEFT * 0.5)
		self.play(
			ShowCreation(code_arrow)
		)

		self.play(
			FadeIn(queue_text)
		)

		self.play(
			ShowCreation(vert_line),
			ShowCreation(horiz_line)
		)

		queue_arrow = Arrow(
			vert_line.get_center(), 
			vert_line.get_center() + RIGHT * 1.5
		)

		self.play(
			ShowCreation(queue_arrow)
		)

		index_0 = Square(side_length=0.6)
		index_0.next_to(queue_arrow, RIGHT)
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
		v_text.next_to(queue_text, DOWN * 1.5)
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

		bfs_full_order = bfs(graph, 0)
		wait_times = [0] * len(bfs_full_order)
		new_highlights = self.show_second_bfs(graph, edge_dict, bfs_full_order[:1], order[1], wait_times, scale_factor=scale_factor)
		all_highlights.extend(new_highlights)

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 7)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 8)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 9)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 10)

		
		index_0 = Square(side_length=0.6)
		index_0.next_to(queue_arrow, RIGHT)
		index_0_data = TextMobject("1")
		index_0_data.scale(scale_factor)
		index_0_data.move_to(index_0.get_center())

		index_1 = Square(side_length=0.6)
		index_1.next_to(index_0, RIGHT, buff=0)
		index_1_data = TextMobject("3")
		index_1_data.scale(scale_factor)
		index_1_data.move_to(index_1.get_center())

		index_2 = Square(side_length=0.6)
		index_2.next_to(index_1, RIGHT, buff=0)
		index_2_data = TextMobject("2")
		index_2_data.scale(scale_factor)
		index_2_data.move_to(index_2.get_center())

		self.play(
			GrowFromCenter(index_0),
			TransformFromCopy(graph[1].data, index_0_data)
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
			TransformFromCopy(graph[2].data, index_2_data)
		)

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 3)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 4)

		next_v_value = index_0_data.copy()
		next_v_value.move_to(v_value.get_center())

		self.play(
			FadeOut(index_0),
			FadeOut(v_value),
			ReplacementTransform(index_0_data, next_v_value),
			index_1.shift, LEFT * 0.6,
			index_1_data.shift, LEFT * 0.6,
			index_2.shift, LEFT * 0.6,
			index_2_data.shift, LEFT * 0.6
		)
		v_value = next_v_value
		index_0, index_1 = index_1, index_2
		index_0_data, index_1_data = index_1_data, index_2_data
		
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 5)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 6)

		new_highlights = self.show_second_bfs(graph, edge_dict, bfs_full_order[1:3], order[1][1:], wait_times, scale_factor=scale_factor)
		all_highlights.extend(new_highlights)


		code_arrow = self.shift_arrow_to_line(code_arrow, code, 7)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 8)
		self.wait()
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 9)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 8)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 9)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 10)
		self.wait()

		index_2 = Square(side_length=0.6)
		index_2.next_to(index_1, RIGHT, buff=0)
		index_2_data = TextMobject("3")
		index_2_data.scale(scale_factor)
		index_2_data.move_to(index_2.get_center())

		self.play(
			GrowFromCenter(index_2),
			TransformFromCopy(graph[3].data, index_2_data)
		)

		self.wait()

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 3)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 4)

		next_v_value = index_0_data.copy()
		next_v_value.move_to(v_value.get_center())

		self.play(
			FadeOut(index_0),
			FadeOut(v_value),
			ReplacementTransform(index_0_data, next_v_value),
			index_1.shift, LEFT * 0.6,
			index_1_data.shift, LEFT * 0.6,
			index_2.shift, LEFT * 0.6,
			index_2_data.shift, LEFT * 0.6
		)
		v_value = next_v_value
		index_0, index_1 = index_1, index_2
		index_0_data, index_1_data = index_1_data, index_2_data
		self.wait()
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 5)
		self.wait()
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 6)

		new_highlights = self.show_second_bfs(graph, edge_dict, bfs_full_order[3:5], order[1][2:], wait_times, scale_factor=scale_factor)
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

		next_v_value = index_0_data.copy()
		next_v_value.move_to(v_value.get_center())

		self.play(
			FadeOut(index_0),
			FadeOut(v_value),
			ReplacementTransform(index_0_data, next_v_value),
			index_1.shift, LEFT * 0.6,
			index_1_data.shift, LEFT * 0.6,
			index_2.shift, LEFT * 0.6,
			index_2_data.shift, LEFT * 0.6,
			index_3.shift, LEFT * 0.6,
			index_3_data.shift, LEFT * 0.6
		)
		v_value = next_v_value
		index_0, index_1, index_2 = index_1, index_2, index_3
		index_0_data, index_1_data, index_2_data = index_1_data, index_2_data, index_3_data

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 5)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 6)

		new_highlights = self.show_second_bfs(graph, edge_dict, bfs_full_order[5:7], order[1][3:], wait_times, scale_factor=scale_factor)
		all_highlights.extend(new_highlights)
		self.wait(2)
		

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 7)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 8)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 9)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 8)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 9)
		self.wait(4)

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 3)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 4)


		next_v_value = index_0_data.copy()
		next_v_value.move_to(v_value.get_center())

		self.play(
			FadeOut(index_0),
			FadeOut(v_value),
			ReplacementTransform(index_0_data, next_v_value),
			index_1.shift, LEFT * 0.6,
			index_1_data.shift, LEFT * 0.6,
			index_2.shift, LEFT * 0.6,
			index_2_data.shift, LEFT * 0.6
		)
		v_value = next_v_value
		index_0, index_1 = index_1, index_2
		index_0_data, index_1_data = index_1_data, index_2_data

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 5)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 3)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 4)

		next_v_value = index_0_data.copy()
		next_v_value.move_to(v_value.get_center())

		self.play(
			FadeOut(index_0),
			FadeOut(v_value),
			ReplacementTransform(index_0_data, next_v_value),
			index_1.shift, LEFT * 0.6,
			index_1_data.shift, LEFT * 0.6,
		)

		v_value = next_v_value
		index_0 = index_1 
		index_0_data = index_1_data

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 5)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 3)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 4)
		self.wait(2)

		next_v_value = index_0_data.copy()
		next_v_value.move_to(v_value.get_center())

		self.play(
			FadeOut(index_0),
			FadeOut(v_value),
			ReplacementTransform(index_0_data, next_v_value),
		)

		v_value = next_v_value

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 5)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 6)

		new_highlights = self.show_second_bfs(graph, edge_dict, bfs_full_order[7:9], order[1][4:], wait_times, scale_factor=scale_factor)
		all_highlights.extend(new_highlights)

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 7)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 8)
		code_arrow = self.shift_arrow_to_line(code_arrow, code, 9)
		self.wait()

		code_arrow = self.shift_arrow_to_line(code_arrow, code, 3)
		
		self.wait(4)

		self.play(
			FadeOut(code_arrow),
			FadeOut(queue_arrow),
			FadeOut(queue_text),
			FadeOut(vert_line),
			FadeOut(vert_line_2),
			FadeOut(horiz_line),
			FadeOut(horiz_line_2),
			FadeOut(v_text),
			FadeOut(v_value)
		)

		self.wait(3)
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

		def_statement = TextMobject(r"$\text{def bfs}(G, v):$")
		def_statement[0][:3].set_color(MONOKAI_BLUE)
		def_statement[0][3:6].set_color(MONOKAI_GREEN)
		def_statement[0][7].set_color(MONOKAI_ORANGE)
		def_statement[0][9].set_color(MONOKAI_ORANGE)
		def_statement.scale(code_scale)
		def_statement.to_edge(LEFT)

		marked_array = TextMobject(r"marked = [False] * $G$.size()")
		marked_array.scale(code_scale)
		marked_array.next_to(def_statement, UP * 0.5)
		marked_array.to_edge(LEFT)
		marked_array[0][14].shift(DOWN * SMALL_BUFF)
		marked_array[0][8:13].set_color(MONOKAI_PURPLE)
		marked_array[0][14].set_color(MONOKAI_PINK)
		marked_array[0][-6:-2].set_color(MONOKAI_BLUE)
		code.extend([marked_array, def_statement])

		line_1 = TextMobject(r"queue = [$v$]")
		line_1.scale(code_scale)
		line_1[0][5].set_color(MONOKAI_PINK)
		line_1.next_to(def_statement, DOWN * 0.5)
		line_1.to_edge(LEFT * 2)
		code.append(line_1)

		line_2 = TextMobject(r"while len(queue) $>$ 0:")
		line_2.scale(code_scale)
		line_2.next_to(line_1, DOWN * 0.5)
		line_2.to_edge(LEFT * 2)
		line_2[0][:5].set_color(MONOKAI_PINK)
		line_2[0][5:8].set_color(MONOKAI_BLUE)
		line_2[0][-3].set_color(MONOKAI_PINK)
		line_2[0][-2].set_color(MONOKAI_PURPLE)
		code.append(line_2)

		line_3 = TextMobject(r"$v$ = queue.pop(0)")
		line_3.scale(code_scale)
		line_3.next_to(line_2, DOWN * 0.5)
		line_3.to_edge(LEFT * 3)
		line_3[0][-6:-3].set_color(MONOKAI_BLUE)
		line_3[0][-2].set_color(MONOKAI_PURPLE)
		code.append(line_3)

		line_4 = TextMobject(r"if not marked[$v$]:")
		line_4.scale(code_scale)
		line_4.next_to(line_3, DOWN * 0.5)
		line_4.to_edge(LEFT * 3)
		line_4[0][:5].set_color(MONOKAI_PINK)
		code.append(line_4)

		line_5 = TextMobject(r"visit($v$)")
		line_5[0][:5].set_color(MONOKAI_BLUE)
		line_5.scale(code_scale)
		line_5.next_to(line_4, DOWN * 0.5)
		line_5.to_edge(LEFT * 4)
		code.append(line_5)

		line_6 = TextMobject(r"marked[$v$] = True")
		line_6[0][-4:].set_color(MONOKAI_PURPLE)
		line_6.scale(code_scale)
		line_6.next_to(line_5, DOWN * 0.5)
		line_6.to_edge(LEFT * 4)
		code.append(line_6)

		line_7 = TextMobject(r"for $w$ in $G$.neighbors($v$):")
		line_7[0][:3].set_color(MONOKAI_PINK)
		line_7[0][4:6].set_color(MONOKAI_PINK)
		line_7[0][8:17].set_color(MONOKAI_BLUE)
		line_7.scale(code_scale)
		line_7.next_to(line_6, DOWN * 0.5)
		line_7.to_edge(LEFT * 4)
		code.append(line_7)

		line_8 = TextMobject(r"if not marked[$w$]:")
		line_8.scale(code_scale)
		line_8.next_to(line_7, DOWN * 0.5)
		line_8.to_edge(LEFT * 5)
		line_8[0][:5].set_color(MONOKAI_PINK)
		code.append(line_8)

		line_9 = TextMobject(r"queue.append($w$)")
		line_9[0][6:12].set_color(MONOKAI_BLUE)
		line_9.scale(code_scale)
		line_9.next_to(line_8, DOWN * 0.5)
		line_9.to_edge(LEFT * 6)
		code.append(line_9)

		code = VGroup(*code)
		code.scale(0.9)
		code.shift(UP * 1.5)

		self.wait()

		self.play(
			Write(code[1])
		)
		self.wait(3)

		self.play(
			Write(code[0])
		)

		self.wait(12)

		self.play(
			Write(code[2])
		)

		self.wait(2)

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

		self.play(
			Write(code[6])
		)

		self.play(
			Write(code[7])
		)

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

class FloodFill(GraphAnimationUtils):
	def construct(self):
		problem = self.introduce_problem()
		grid, grid_group, title, h_line = self.demo_examples(problem)
		group = self.show_graph(grid, grid_group, title, h_line)
	
	def introduce_problem(self):
		problem = TextMobject(
			"Given an image represented by a grid of pixel values, a starting" + "\\\\",
			r"pixel (row, col), and new pixel value $p$, transform all pixels" + "\\\\",
			r"connected to the starting pixel to the new pixel value."
		)

		problem.scale(0.8)
		problem.move_to(UP * 3)

		return problem

	def demo_examples(self, problem):
		grid = self.create_grid(5, 5, square_length=0.6)
		grid_group = self.get_group_grid(grid)
		grid_group.next_to(problem, DOWN)

		values = np.random.choice(2, (5, 5), p=[0.5, 0.5])
		value_objects = self.put_values_in_grid(values, grid)
		value_objects_group = self.get_group_grid(value_objects)

		self.play(
			Write(problem[0][:43])
		)
		self.wait(2)

		self.play(
			ShowCreation(grid_group),
		)

		self.play(
			FadeIn(value_objects_group)
		)

		self.wait(2)

		self.play(
			Write(problem[0][43:])
		)

		self.play(
			Write(problem[1][:32])
		)

		self.wait(2)

		self.play(
			Write(problem[1][32:])
		)

		self.play(
			Write(problem[2])
		)

		self.wait(5)

		self.play(
			value_objects_group.shift, LEFT * 2.5,
			grid_group.shift, LEFT * 2.5
		)

		img, original = self.get_image(grid, values)
		img.shift(RIGHT * 5)

		self.play(
			TransformFromCopy(original, img),
			run_time=3
		)

		self.wait(13)

		self.play(
			problem[2][:9].set_color, YELLOW
		)

		self.wait(2)

		connected_def = TextMobject(
			"Connected pixels share the same value and have a path" + "\\\\", 
			"between them by moving either left, right, up, or down."
		)
		connected_def.scale(0.8)
		connected_def[0][:9].set_color(YELLOW)
		connected_def.move_to(DOWN * 3)

		self.play(
			Write(connected_def[0])
		)

		self.play(
			Write(connected_def[1])
		)

		args = TextMobject(r"(row, col): $(2, 2) \quad p: 3$")
		args.scale(0.8)
		args.move_to(DOWN * 1.5)

		self.wait(3)

		self.play(
			FadeIn(args)
		)

		self.wait(2)

		self.play(
			args[0][:-3].set_color, GREEN_SCREEN
		)


		start = grid[2][2].copy()
		start.set_stroke(color=GREEN_SCREEN, width=8)
		self.play(
			ShowCreation(start)
		)
		self.wait(4)

		self.play(
			args[0][-3:].set_color, BLUE
		)

		self.wait(4)

		flood_fill_answer = floodFill(values, 2, 2, 3)
		transforms = self.change_image(value_objects, flood_fill_answer)
		self.play(
			*transforms,
			img[:14].set_color, BLUE,
			img[-1].set_color, BLUE,
			run_time=3
		)

		self.wait()

		args2 = TextMobject(r"(row, col): $(2, 4) \quad p: 5$")
		args2.scale(0.8)
		args2.next_to(args, DOWN)

		self.play(
			FadeIn(args2)
		)

		self.wait(3)

		self.play(
			args[0][:-3].set_color, WHITE,
			args2[0][:-3].set_color, GREEN_SCREEN,
			args2[0][-3:].set_color, YELLOW,
			start.move_to, grid[2][4].get_center()
		)

		self.wait(4)

		yellow_segment = VGroup(*[grid[i][4].copy() for i in range(3)])
		for i in range(3):
			yellow_segment[i].set_color(YELLOW)
			yellow_segment[i].set_fill(color=YELLOW, opacity=0.9)
		
		new_value_objects = self.get_new_val_objs(yellow_segment)
		yellow_segment.next_to(img[6], RIGHT, buff=0)


		self.play(
			FadeIn(yellow_segment),
			*[Transform(value_objects[i][4], new_value_objects[i]) for i in range(3)]
		)

		self.wait(6)

		bucket = Bucket()
		bucket.rotate(PI)
		bucket.flip()
		bucket[0].set_color(GRAY)
		bucket[1].set_color(WHITE)
		bucket.scale(0.2)

		example_group = VGroup(args, args2)
		self.play(
			example_group.shift, LEFT * 0.75
		)

		brace = Brace(example_group, direction=RIGHT)
		bucket.next_to(brace, RIGHT)

		self.play(
			GrowFromCenter(brace)
		)

		self.play(
			ShowCreation(bucket)
		)

		self.wait(6)

		title = TextMobject("Flood Fill Problem")
		title.scale(1.2)
		title.shift(UP * 3.5)

		h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
		h_line.next_to(title, DOWN)

		self.play(
			FadeOut(value_objects_group),
			FadeOut(grid_group),
			ReplacementTransform(problem, title),
			FadeOut(connected_def),
			FadeOut(start),
			FadeOut(args),
			FadeOut(args2),
			FadeOut(img),
			FadeOut(yellow_segment),
			FadeOut(bucket),
			FadeOut(brace),
			run_time=2,
		)

		self.play(
			ShowCreation(h_line)
		)

		self.wait()

		return grid, grid_group, title, h_line

	def show_graph(self, grid, grid_group, title, h_line):
		grid_group.move_to(UP * 0.5)
		args = TextMobject(r"(row, col): $(2, 2) \quad p: 2$")
		args.scale(0.8)
		args[0][-3:].set_color(GREEN_SCREEN)
		args.next_to(grid_group, DOWN)
		values = np.random.choice(2, (5, 5), p=[0.5, 0.5])
		value_objects = self.put_values_in_grid(values, grid)
		value_objects_group = self.get_group_grid(value_objects)
		args.next_to(grid_group, DOWN)

		self.play(
			FadeIn(grid_group),
			FadeIn(value_objects_group)
		)

		self.play(
			Write(args)
		)

		bfs_full_order = [
		(2, 2), ((2, 2), (2, 1)),
		(2, 1), ((2, 2), (3, 2)),
		(3, 2), ((2, 2), (2, 3)),
		(2, 3), ((2, 1), (2, 0)),
		(2, 0), ((2, 1), (1, 1)),
		(1, 1), ((2, 3), (1, 3)),
		(1, 3), ((2, 3), (2, 4)),
		(2, 4), ((1, 3), (0, 3)),
		(0, 3), ((2, 4), (3, 4)),
		(3, 4), ((0, 3), (0, 2)),
		(0, 2)
		]

		highlights = []
		for elem in bfs_full_order:
			if isinstance(elem[0], int):
				i, j = elem
				start = grid[i][j].copy()
				start.set_stroke(color=GREEN_SCREEN, width=8)
				start.scale(0.8)
				highlights.append(start)

		self.play(
			*[GrowFromCenter(h) for h in highlights]
		)

		self.wait(12)

		connected_def = TextMobject(
			"Connected pixels share the same value and have a path" + "\\\\", 
			"between them by moving either left, right, up, or down."
		)
		connected_def.scale(0.8)
		connected_def[0][:9].set_color(YELLOW)
		connected_def.move_to(DOWN * 3)

		self.play(
			FadeIn(connected_def)
		)

		self.wait(6)

		self.play(
			*[FadeOut(h) for h in highlights]
		)

		self.play(
			grid_group.shift, LEFT * 3 + DOWN * 0.5,
			value_objects_group.shift, LEFT * 3 + DOWN * 0.5,
			args.shift, LEFT * 3 + DOWN * 0.5
		)

		self.wait()

		graph, edge_dict = self.create_graph(values)
		nodes, edges = self.make_graph_mobject(graph, edge_dict, show_data=False)
		entire_graph = VGroup(nodes, edges)

		transforms = []

		for i in range(values.shape[0]):
			for j in range(values.shape[1]):
				tranform = TransformFromCopy(grid[i][j], graph[(i, j)].circle)
				transforms.append(tranform)

		self.play(
			*transforms,
			run_time=3
		)
		self.wait(3)

		self.play(
			*[ShowCreation(line) for line in edges]
		)

		self.wait(12)



		wait_times = [0] * len(bfs_full_order)

		highlights = self.show_bfs(graph, edge_dict, bfs_full_order, wait_times, value_objects)

		self.wait()

		self.play(
			*[FadeOut(h) for h in highlights]
		)

		group = VGroup(value_objects_group, grid_group, args)

		self.play(
			FadeOut(entire_graph),
			FadeOut(title),
			FadeOut(h_line),
			FadeOut(connected_def),
			group.move_to, UP * 1.5 + RIGHT * 3.5,
			run_time=2
		)

		self.wait()

		code = self.generate_code()

		emphasize_neighbors_func = Underline(code[8][0][12:-1], color=YELLOW, buff=SMALL_BUFF / 2)
		self.play(
			ShowCreation(emphasize_neighbors_func)
		)

		self.wait(4)


		start = self.highlight_index(value_objects, 2, 2, color=YELLOW)
		self.play(
			ShowCreation(start)
		)
		self.wait()

		neighbors = [self.highlight_index(value_objects, row, col) for row, col in [(2, 1), (2, 3), (3, 2)]]

		self.play(
			*[ShowCreation(n) for n in neighbors]
		)

		self.wait()

		self.play(
			*[FadeOut(h) for h in neighbors],
			start.move_to, value_objects[0][3].get_center()
		)

		self.wait()

		neighbors = [self.highlight_index(value_objects, row, col) for row, col in [(0, 2), (1, 3)]]

		self.play(
			*[ShowCreation(n) for n in neighbors]
		)

		self.wait()

		helper_funcs = self.generate_helper_code()

		self.play(
			FadeOut(emphasize_neighbors_func),
			FadeOut(start),
			*[FadeOut(n) for n in neighbors]
		)

		self.wait(11)

		key_idea = TextMobject("Graph perspective is key!")
		key_idea.next_to(args, DOWN)
		self.play(
			Write(key_idea)
		)

		self.wait(9)

	def highlight_index(self, value_objects, row, col, color=BLUE):
		rect = SurroundingRectangle(value_objects[row][col], buff=SMALL_BUFF, color=color)
		return rect

	def generate_code(self):
		code_scale = 0.6
		
		code = []

		def_statement = TextMobject(r"def floodFill(img, row, col, p):")
		def_statement[0][:3].set_color(MONOKAI_BLUE)
		def_statement[0][3:12].set_color(MONOKAI_GREEN)
		def_statement[0][13:16].set_color(MONOKAI_ORANGE)
		def_statement[0][17:20].set_color(MONOKAI_ORANGE)
		def_statement[0][21:24].set_color(MONOKAI_ORANGE)
		def_statement[0][25].set_color(MONOKAI_ORANGE)
		def_statement.scale(code_scale)
		def_statement.to_edge(LEFT)
		def_statement.shift(UP * 3.5)

		code.append(def_statement)

		line_1 = TextMobject(r"start $=$ img[row][col]")
		line_1.scale(code_scale)
		line_1.next_to(def_statement, DOWN * 0.5)
		line_1.to_edge(LEFT * 2)
		line_1[0][5].set_color(MONOKAI_PINK)
		code.append(line_1)

		line_2 = TextMobject(r"queue $=$ [(row, col)]")
		line_2.scale(code_scale)
		line_2.next_to(line_1, DOWN * 0.5)
		line_2.to_edge(LEFT * 2)
		line_2[0][5].set_color(MONOKAI_PINK)
		code.append(line_2)

		line_3 = TextMobject(r"visited $= \text{set}()$")
		line_3.scale(code_scale)
		line_3.next_to(line_2, DOWN * 0.5)
		line_3.to_edge(LEFT * 2)
		line_3[0][7].set_color(MONOKAI_PINK)
		line_3[0][8:11].set_color(MONOKAI_BLUE)
		code.append(line_3)

		line_4 = TextMobject(r"while len(queue) $>$ 0:")
		line_4.scale(code_scale)
		line_4.next_to(line_3, DOWN * 0.5)
		line_4.to_edge(LEFT * 2)
		line_4[0][:5].set_color(MONOKAI_PINK)
		line_4[0][5:8].set_color(MONOKAI_BLUE)
		line_4[0][-3].set_color(MONOKAI_PINK)
		line_4[0][-2].set_color(MONOKAI_PURPLE)
		code.append(line_4)

		line_5 = TextMobject(r"row, col $=$ queue.pop(0)")
		line_5.scale(code_scale)
		line_5.next_to(line_4, DOWN * 0.5)
		line_5.to_edge(LEFT * 3)
		line_5[0][7].set_color(MONOKAI_PINK)
		line_5[0][-6:-3].set_color(MONOKAI_BLUE)
		line_5[0][-2].set_color(MONOKAI_PURPLE)
		code.append(line_5)

		line_6 = TextMobject(r"visited.add((row, col))")
		line_6.scale(code_scale)
		line_6.next_to(line_5, DOWN * 0.5)
		line_6.to_edge(LEFT * 3)
		line_6[0][8:11].set_color(MONOKAI_BLUE)
		code.append(line_6)

		# line_7 = TextMobject(r"if img[row][col] $==$ start:")
		# line_7.scale(code_scale)
		# line_7.next_to(line_6, DOWN * 0.5)
		# line_7.to_edge(LEFT * 3)
		# line_7[0][:2].set_color(MONOKAI_PINK)
		# line_7[0][-8:-6].set_color(MONOKAI_PINK)
		# code.append(line_7)

		line_8 = TextMobject(r"img[row][col] $=$ p")
		line_8.scale(code_scale)
		line_8.next_to(line_6, DOWN * 0.5)
		line_8.to_edge(LEFT * 3)
		line_8[0][-2].set_color(MONOKAI_PINK)
		code.append(line_8)

		line_9 = TextMobject(r"for row, col in neighbors(img, row, col, start):")
		line_9[0][:3].set_color(MONOKAI_PINK)
		line_9[0][10:12].set_color(MONOKAI_PINK)
		line_9[0][12:21].set_color(MONOKAI_BLUE)
		line_9.scale(code_scale)
		line_9.next_to(line_8, DOWN * 0.5)
		line_9.to_edge(LEFT * 3)
		code.append(line_9)

		line_10 = TextMobject(r"if (row, col) not in visited:")
		line_10.scale(code_scale)
		line_10.next_to(line_9, DOWN * 0.5)
		line_10.to_edge(LEFT * 4)
		line_10[0][:2].set_color(MONOKAI_PINK)
		line_10[0][11:16].set_color(MONOKAI_PINK)
		code.append(line_10)

		line_11 = TextMobject(r"queue.append((row, col))")
		line_11[0][6:12].set_color(MONOKAI_BLUE)
		line_11.scale(code_scale)
		line_11.next_to(line_10, DOWN * 0.5)
		line_11.to_edge(LEFT * 5)
		code.append(line_11)

		line_12 = TextMobject(r"return img")
		line_12[0][:6].set_color(MONOKAI_PINK)
		line_12.scale(code_scale)
		line_12.next_to(line_11, DOWN * 0.5)
		line_12.to_edge(LEFT * 2)
		code.append(line_12)

		code = VGroup(*code)

		self.play(
			Write(code[0])
		)

		self.wait(3)

		self.play(
			Write(code[1])
		)

		self.wait(8)

		self.play(
			Write(code[2])
		)

		self.wait(6)
		self.play(
			Write(code[3])
		)

		self.wait(4)
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
		self.wait()

		self.play(
			Write(code[7])
		)

		self.wait(3)
		self.play(
			Write(code[8])
		)
		self.wait(3)
		self.play(
			Write(code[9])
		)
		self.wait()

		self.play(
			Write(code[10])
		)

		self.wait(4)

		self.play(
			Write(code[11])
		)

		self.wait(3)
		
		return code

	def generate_helper_code(self):
		code_scale = 0.6
		
		code = []

		def_statement = TextMobject(r"def neighbors(img, row, col, start):")
		def_statement[0][:3].set_color(MONOKAI_BLUE)
		def_statement[0][3:12].set_color(MONOKAI_GREEN)
		def_statement[0][13:16].set_color(MONOKAI_ORANGE)
		def_statement[0][17:20].set_color(MONOKAI_ORANGE)
		def_statement[0][21:24].set_color(MONOKAI_ORANGE)
		def_statement[0][25:30].set_color(MONOKAI_ORANGE)
		def_statement.scale(code_scale)
		def_statement.move_to(UP * 3.5)
		def_statement.to_edge(LEFT * 10)
		code.append(def_statement)

		line_1 = TextMobject(r"indices $=$ [(row $-$ 1, col), (row $+$ 1, col), (row, col $-$ 1), (row, col $+$ 1)]")
		line_1.scale(code_scale)
		line_1.next_to(def_statement, DOWN * 0.5)
		line_1.to_edge(LEFT * 11)
		line_1[0][7].set_color(MONOKAI_PINK)
		line_1[0][13].set_color(MONOKAI_PINK)
		line_1[0][14].set_color(MONOKAI_PURPLE)
		line_1[0][25].set_color(MONOKAI_PINK)
		line_1[0][26].set_color(MONOKAI_PURPLE)
		line_1[0][41].set_color(MONOKAI_PINK)
		line_1[0][42].set_color(MONOKAI_PURPLE)
		line_1[0][53].set_color(MONOKAI_PINK)
		line_1[0][54].set_color(MONOKAI_PURPLE)
		code.append(line_1)

		line_2 = TextMobject(r"return [(row, col) for row, col in indices if isValid(img, row, col) and img[row][col] $==$ start]")
		line_2[0][:6].set_color(MONOKAI_PINK)
		line_2[0][16:19].set_color(MONOKAI_PINK)
		line_2[0][26:28].set_color(MONOKAI_PINK)
		line_2[0][35:37].set_color(MONOKAI_PINK)
		line_2[0][37:44].set_color(MONOKAI_BLUE)
		line_2[0][57:60].set_color(MONOKAI_PINK)
		line_2[0][57:60].set_color(MONOKAI_PINK)
		line_2[0][73:75].set_color(MONOKAI_PINK)
		line_2.scale(code_scale)
		line_2.next_to(line_1, DOWN * 0.5)
		line_2.to_edge(LEFT * 11)
		code.append(line_2)


		def_statement2 = TextMobject(r"def isValid(img, row, col):")
		def_statement2[0][:3].set_color(MONOKAI_BLUE)
		def_statement2[0][3:10].set_color(MONOKAI_GREEN)
		def_statement2[0][11:14].set_color(MONOKAI_ORANGE)
		def_statement2[0][15:18].set_color(MONOKAI_ORANGE)
		def_statement2[0][19:22].set_color(MONOKAI_ORANGE)
		def_statement2.scale(code_scale)
		def_statement2.next_to(line_2, DOWN * 0.5)
		def_statement2.to_edge(LEFT * 10)
		code.append(def_statement2)

		return_exp = TextMobject(r"return row $\geq$ 0 and col $\geq$ 0 and row $<$ len(img) and col $<$ len(img[0])")
		return_exp[0][:6].set_color(MONOKAI_PINK)
		return_exp[0][9].set_color(MONOKAI_PINK)
		return_exp[0][10].set_color(MONOKAI_PURPLE)
		return_exp[0][11:14].set_color(MONOKAI_PINK)
		return_exp[0][17].set_color(MONOKAI_PINK)
		return_exp[0][18].set_color(MONOKAI_PURPLE)
		return_exp[0][19:22].set_color(MONOKAI_PINK)
		return_exp[0][25].set_color(MONOKAI_PINK)
		return_exp[0][26:29].set_color(MONOKAI_BLUE)
		return_exp[0][34:37].set_color(MONOKAI_PINK)
		return_exp[0][40].set_color(MONOKAI_PINK)
		return_exp[0][41:44].set_color(MONOKAI_BLUE)
		return_exp[0][49].set_color(MONOKAI_PURPLE)
		return_exp.scale(code_scale)
		return_exp.next_to(def_statement2, DOWN * 0.5)
		return_exp.to_edge(LEFT * 11)
		code.append(return_exp)

		code_group = VGroup(*code)

		code_group.move_to(DOWN * 2.6 + LEFT * 2)
		code_group.to_edge(LEFT)

		self.play(
			Write(code[0])
		)

		self.wait(7)

		self.play(
			Write(code[1])
		)

		self.wait(2)

		self.play(
			Write(code[2]),
			run_time=3
		)
		self.wait(6)
		
		self.play(
			Write(code[3])
		)
		self.wait(4)

		self.play(
			Write(code[4][0][:19])
		)

		self.wait(2)

		self.play(
			Write(code[4][0][19:34])
		)

		self.wait(2)

		self.play(
			Write(code[4][0][34:])
		)

		self.wait(5)


	def create_graph(self, values):
		graph = {}
		edge_dict = {}

		radius, scale = 0.2, 0.9

		for i in range(values.shape[0]):
			for j in range(values.shape[1]):
				graph[(i, j)] = GraphNode('', position=DOWN*(i - 2) + RIGHT * j, radius=radius, scale=scale)

		for i in range(values.shape[0]):
			for j in range(values.shape[1]):
				for neighbor in neighbors(values, i, j, values[i][j]):
					key = ((i, j), neighbor)
					reverse_key = (neighbor, (i, j))
					if key not in edge_dict and reverse_key not in edge_dict:
						edge_dict[key] = graph[(i, j)].connect(graph[neighbor])

		return graph, edge_dict

	def make_graph_mobject(self, graph, edge_dict, node_color=DARK_BLUE_B, 
		stroke_color=BLUE, data_color=WHITE, edge_color=GRAY, scale_factor=1,
		show_data=True):
		nodes = []
		edges = []
		for node in graph.values():
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

	def show_bfs(self, graph, edge_dict, full_order, 
		wait_times, value_objects, scale_factor=1, run_time=1):
		i = 0
		angle = 180
		all_highlights = []
		for element in full_order:
			if isinstance(element[0], int):
				surround_circle = self.highlight_node(graph, element, 
					start_angle=angle/360 * TAU, scale_factor=scale_factor, run_time=run_time, animate=False)
				all_highlights.append(surround_circle)
				integer = Integer(2)
				integer.scale(0.7)
				m, n = element
				integer.move_to(value_objects[m][n].get_center())
				integer.set_color(GREEN_SCREEN)
				self.play(
					ShowCreation(surround_circle),
					Transform(value_objects[m][n], integer),
					run_time=run_time
				)
			else:
				last_edge = self.sharpie_edge(edge_dict, element[0], element[1], 
					scale_factor=scale_factor, run_time=run_time)
				angle = self.find_angle_of_intersection(graph, last_edge.get_end(), element[1])
				all_highlights.append(last_edge)
			self.wait(wait_times[i])
			i += 1
		return all_highlights

	def change_image(self, old_value_objs, new_values, color=BLUE, target=3):
		transforms = []
		for i in range(len(old_value_objs)):
			for j in range(len(old_value_objs[0])):
				if new_values[i][j] != target:
					continue
				new_val_obj = Integer(new_values[i][j])
				new_val_obj.set_color(color)
				new_val_obj.scale(0.7)
				new_val_obj.move_to(old_value_objs[i][j].get_center())
				transform = Transform(old_value_objs[i][j], new_val_obj)
				transforms.append(transform)

		return transforms

	def get_new_val_objs(self, location):
		new_val_objs = []
		for loc in location:
			integer = Integer(5)
			integer.scale(0.7)
			integer.move_to(loc)
			integer.set_color(YELLOW)
			new_val_objs.append(integer)
		return new_val_objs

	def put_values_in_grid(self, values, grid):
		value_objects = [[0 for _ in range(values.shape[0])] for _ in range(values.shape[1])]
		for i in range(values.shape[0]):
			for j in range(values.shape[1]):
				obj = Integer(values[i][j])
				obj.scale(0.7)
				obj.move_to(grid[i][j].get_center())
				value_objects[i][j] = obj
		return value_objects

	def render_values(self, value_objects):
		animations = []
		for i in range(len(value_objects)):
			for j in range(len(value_objects[0])):
				animations.append(FadeIn(value_objects[i][j]))
		return animations

	def get_image(self, grid, values):
		image = []
		original = []

		for i in range(values.shape[0]):
			for j in range(values.shape[1]):
				if values[i][j] == 1:
					square = grid[i][j].copy()
					original.append(grid[i][j])
					square.set_fill(color=WHITE, opacity=0.9)
					image.append(square)

		return VGroup(*image), VGroup(*original)


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
			# print(prev_row)
			new_row = []
			for square in prev_row:
				# print(square)
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

class FloodFillImplementation(GraphAnimationUtils):
	def construct(self):
		self.show_implementation()

	def show_implementation(self):
		code = self.generate_code()
		helper_funcs = self.generate_helper_code()

	def generate_code(self):
		code_scale = 0.5
		
		code = []

		def_statement = TextMobject(r"def floodFill(img, row, col, p):")
		def_statement[0][:3].set_color(MONOKAI_BLUE)
		def_statement[0][3:12].set_color(MONOKAI_GREEN)
		def_statement[0][13:16].set_color(MONOKAI_ORANGE)
		def_statement[0][17:20].set_color(MONOKAI_ORANGE)
		def_statement[0][21:24].set_color(MONOKAI_ORANGE)
		def_statement[0][25].set_color(MONOKAI_ORANGE)
		def_statement.scale(code_scale)
		def_statement.to_edge(LEFT)
		def_statement.shift(UP * 3.5)

		code.append(def_statement)

		line_1 = TextMobject(r"start $=$ img[row][col]")
		line_1.scale(code_scale)
		line_1.next_to(def_statement, DOWN * 0.5)
		line_1.to_edge(LEFT * 2)
		line_1[0][5].set_color(MONOKAI_PINK)
		code.append(line_1)

		line_2 = TextMobject(r"queue $=$ [(row, col)]")
		line_2.scale(code_scale)
		line_2.next_to(line_1, DOWN * 0.5)
		line_2.to_edge(LEFT * 2)
		line_2[0][5].set_color(MONOKAI_PINK)
		code.append(line_2)

		line_3 = TextMobject(r"visited $= \text{set}()$")
		line_3.scale(code_scale)
		line_3.next_to(line_2, DOWN * 0.5)
		line_3.to_edge(LEFT * 2)
		line_3[0][7].set_color(MONOKAI_PINK)
		line_3[0][8:11].set_color(MONOKAI_BLUE)
		code.append(line_3)

		line_4 = TextMobject(r"while len(queue) $>$ 0:")
		line_4.scale(code_scale)
		line_4.next_to(line_3, DOWN * 0.5)
		line_4.to_edge(LEFT * 2)
		line_4[0][:5].set_color(MONOKAI_PINK)
		line_4[0][5:8].set_color(MONOKAI_BLUE)
		line_4[0][-3].set_color(MONOKAI_PINK)
		line_4[0][-2].set_color(MONOKAI_PURPLE)
		code.append(line_4)

		line_5 = TextMobject(r"row, col $=$ queue.pop(0)")
		line_5.scale(code_scale)
		line_5.next_to(line_4, DOWN * 0.5)
		line_5.to_edge(LEFT * 3)
		line_5[0][7].set_color(MONOKAI_PINK)
		line_5[0][-6:-3].set_color(MONOKAI_BLUE)
		line_5[0][-2].set_color(MONOKAI_PURPLE)
		code.append(line_5)

		line_6 = TextMobject(r"visited.add((row, col))")
		line_6.scale(code_scale)
		line_6.next_to(line_5, DOWN * 0.5)
		line_6.to_edge(LEFT * 3)
		line_6[0][8:11].set_color(MONOKAI_BLUE)
		code.append(line_6)

		line_7 = TextMobject(r"if img[row][col] $==$ start:")
		line_7.scale(code_scale)
		line_7.next_to(line_6, DOWN * 0.5)
		line_7.to_edge(LEFT * 3)
		line_7[0][:2].set_color(MONOKAI_PINK)
		line_7[0][-8:-6].set_color(MONOKAI_PINK)
		code.append(line_7)

		line_8 = TextMobject(r"img[row][col] $=$ p")
		line_8.scale(code_scale)
		line_8.next_to(line_7, DOWN * 0.5)
		line_8.to_edge(LEFT * 4)
		line_8[0][-2].set_color(MONOKAI_PINK)
		code.append(line_8)

		line_9 = TextMobject(r"for row, col in neighbors(img, row, col, start):")
		line_9[0][:3].set_color(MONOKAI_PINK)
		line_9[0][10:12].set_color(MONOKAI_PINK)
		line_9[0][12:21].set_color(MONOKAI_BLUE)
		line_9.scale(code_scale)
		line_9.next_to(line_8, DOWN * 0.5)
		line_9.to_edge(LEFT * 3)
		code.append(line_9)

		line_10 = TextMobject(r"if (row, col) not in visited:")
		line_10.scale(code_scale)
		line_10.next_to(line_9, DOWN * 0.5)
		line_10.to_edge(LEFT * 4)
		line_10[0][:2].set_color(MONOKAI_PINK)
		line_10[0][11:16].set_color(MONOKAI_PINK)
		code.append(line_10)

		line_11 = TextMobject(r"queue.append((row, col))")
		line_11[0][6:12].set_color(MONOKAI_BLUE)
		line_11.scale(code_scale)
		line_11.next_to(line_10, DOWN * 0.5)
		line_11.to_edge(LEFT * 5)
		code.append(line_11)

		line_12 = TextMobject(r"return img")
		line_12[0][:6].set_color(MONOKAI_PINK)
		line_12.scale(code_scale)
		line_12.next_to(line_11, DOWN * 0.5)
		line_12.to_edge(LEFT * 2)
		code.append(line_12)

		code = VGroup(*code)

		self.play(
			Write(code[0])
		)

		self.play(
			Write(code[1])
		)

		self.wait()

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

		self.play(
			Write(code[11])
		)
		self.play(
			Write(code[12])
		)

		self.wait()
		
		return code

	def generate_helper_code(self):
		code_scale = 0.5
		
		code = []

		def_statement = TextMobject(r"def neighbors(img, row, col, start):")
		def_statement[0][:3].set_color(MONOKAI_BLUE)
		def_statement[0][3:12].set_color(MONOKAI_GREEN)
		def_statement[0][13:16].set_color(MONOKAI_ORANGE)
		def_statement[0][17:20].set_color(MONOKAI_ORANGE)
		def_statement[0][21:24].set_color(MONOKAI_ORANGE)
		def_statement[0][25:30].set_color(MONOKAI_ORANGE)
		def_statement.scale(code_scale)
		def_statement.move_to(UP * 3.5)
		def_statement.to_edge(LEFT * 10)
		code.append(def_statement)

		line_1 = TextMobject(r"indices $=$ [(row $-$ 1, col), (row $+$ 1, col), (row, col $-$ 1), (row, col $+$ 1)]")
		line_1.scale(code_scale)
		line_1.next_to(def_statement, DOWN * 0.5)
		line_1.to_edge(LEFT * 11)
		line_1[0][7].set_color(MONOKAI_PINK)
		line_1[0][13].set_color(MONOKAI_PINK)
		line_1[0][14].set_color(MONOKAI_PURPLE)
		line_1[0][25].set_color(MONOKAI_PINK)
		line_1[0][26].set_color(MONOKAI_PURPLE)
		line_1[0][41].set_color(MONOKAI_PINK)
		line_1[0][42].set_color(MONOKAI_PURPLE)
		line_1[0][53].set_color(MONOKAI_PINK)
		line_1[0][54].set_color(MONOKAI_PURPLE)
		code.append(line_1)

		line_2 = TextMobject(r"return [(row, col) for row, col in indices if isValid(img, row, col) and img[row][col] $==$ start]")
		line_2[0][:6].set_color(MONOKAI_PINK)
		line_2[0][16:19].set_color(MONOKAI_PINK)
		line_2[0][26:28].set_color(MONOKAI_PINK)
		line_2[0][35:37].set_color(MONOKAI_PINK)
		line_2[0][37:44].set_color(MONOKAI_BLUE)
		line_2[0][57:60].set_color(MONOKAI_PINK)
		line_2[0][57:60].set_color(MONOKAI_PINK)
		line_2[0][73:75].set_color(MONOKAI_PINK)
		line_2.scale(code_scale)
		line_2.next_to(line_1, DOWN * 0.5)
		line_2.to_edge(LEFT * 11)
		code.append(line_2)


		def_statement2 = TextMobject(r"def isValid(img, row, col):")
		def_statement2[0][:3].set_color(MONOKAI_BLUE)
		def_statement2[0][3:10].set_color(MONOKAI_GREEN)
		def_statement2[0][11:14].set_color(MONOKAI_ORANGE)
		def_statement2[0][15:18].set_color(MONOKAI_ORANGE)
		def_statement2[0][19:22].set_color(MONOKAI_ORANGE)
		def_statement2.scale(code_scale)
		def_statement2.next_to(line_2, DOWN * 0.5)
		def_statement2.to_edge(LEFT * 10)
		code.append(def_statement2)

		return_exp = TextMobject(r"return row $\geq$ 0 and col $\geq$ 0 and row $<$ len(img) and col $<$ len(img[0])")
		return_exp[0][:6].set_color(MONOKAI_PINK)
		return_exp[0][9].set_color(MONOKAI_PINK)
		return_exp[0][10].set_color(MONOKAI_PURPLE)
		return_exp[0][11:14].set_color(MONOKAI_PINK)
		return_exp[0][17].set_color(MONOKAI_PINK)
		return_exp[0][18].set_color(MONOKAI_PURPLE)
		return_exp[0][19:22].set_color(MONOKAI_PINK)
		return_exp[0][25].set_color(MONOKAI_PINK)
		return_exp[0][26:29].set_color(MONOKAI_BLUE)
		return_exp[0][34:37].set_color(MONOKAI_PINK)
		return_exp[0][40].set_color(MONOKAI_PINK)
		return_exp[0][41:44].set_color(MONOKAI_BLUE)
		return_exp[0][49].set_color(MONOKAI_PURPLE)
		return_exp.scale(code_scale)
		return_exp.next_to(def_statement2, DOWN * 0.5)
		return_exp.to_edge(LEFT * 11)
		code.append(return_exp)



		self.play(
			Write(code[0])
		)

		self.play(
			Write(code[1])
		)

		self.play(
			Write(code[2])
		)
		
		self.play(
			Write(code[3])
		)

		self.play(
			Write(code[4])
		)

		self.wait()

class Patreon(Scene):
    def construct(self):
        patreon_logo = PatreonLogo()
        patreon_logo.scale(0.65)
        patreon_logo[1].set_color(OXFORD_BLUE)
        patreon_logo[2].set_color(WHITE)
        support = TextMobject("Support")
        support.next_to(patreon_logo, DOWN)
        group = VGroup(patreon_logo, support)
        group.move_to(RIGHT * 4.7)

        self.wait()
        self.play(
            DrawBorderThenFill(patreon_logo),
            Write(support),
            run_time=2
        )

        self.wait(19)

class PatreonLogo(SVGMobject):
    CONFIG = {
        "file_name": "patreon_logo",
        "fill_color": "#F96854",
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
		
class Bucket(SVGMobject):
    CONFIG = {
        "file_name": "bucket",
        "fill_color": "#F96854",
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

class Thumbnail(GraphAnimationUtils):
	def construct(self):
		graph, edge_dict = self.create_graph()
		nodes, edges = self.make_graph_mobject(graph, edge_dict, show_data=False)
		entire_graph = VGroup(nodes, edges)
		entire_graph.shift(RIGHT * 2.5)

		self.play(
			FadeIn(entire_graph)
		)

		highlights = [self.highlight_node(graph, i, animate=False) for i in range(len(graph))]
		edge_keys = [
		(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), 
		(0, 2), (1, 3), (2, 4), (3, 0), (4, 1),
		(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
		]
		edge_highlights = [self.sharpie_edge(edge_dict, u, v, animate=False) for u, v in edge_keys]
		self.play(
			*[FadeIn(h) for h in highlights]
		)

		self.play(
			*[FadeIn(e) for e in edge_highlights]
		)

		scale = 3

		b = TextMobject("Breadth").scale(scale)
		f = TextMobject("First").scale(scale)
		s = TextMobject("Search").scale(scale)

		f.move_to(LEFT * 3.7)
		b.next_to(f, UP * 2)
		s.next_to(f, DOWN * 2)

		self.play(
			FadeIn(b),
			FadeIn(f),
			FadeIn(s)
		)


		self.wait()

	def create_graph(self):
		radius = 0.3
		polygon = RegularPolygon(n=5)
		polygon.scale(2)

		graph = [GraphNode(i, position=p, radius=radius) for i, p in enumerate(polygon.get_vertices())]
		edge_dict = {}
		for i in range(len(graph)):
			i, j = i % 5, (i + 1) % 5
			edge_dict[(i, j)] = graph[i].connect(graph[j])

		rest_graph = []
		for key in [((i - 1) % 5, i) for i in range(5)]:
			edge = edge_dict[key]
			v = edge.get_unit_vector()
			node = GraphNode(key[1] + 5, position=edge.get_end() + v * 2, radius=radius)
			rest_graph.append(node)

		for i in range(len(rest_graph)):
			i, j = i % 5, (i + 1) % 5
			edge_dict[(i + 5, j + 5)] = rest_graph[i].connect(rest_graph[j])

		for i in range(len(graph)):
			i, j = i % 5, (i + 2) % 5
			if (i, j) not in edge_dict and (j, i) not in edge_dict:
				edge_dict[(i, j)] = graph[i].connect(graph[j])

		for i in range(len(graph)):
			edge_dict[(i, i + 5)] = graph[i].connect(rest_graph[i])

		graph.extend(rest_graph)

		return graph, edge_dict

	def sharpie_edge(self, edge_dict, u, v, color=GREEN_SCREEN, 
		scale_factor=1, animate=True, run_time=1):
		edge = edge_dict[(u, v)]
		line = Line(edge.get_start(), edge.get_end())
		line.set_stroke(width=12 * scale_factor)
		line.set_color(color)
		if animate:
			self.play(
				ShowCreation(line),
				run_time=run_time
			)
		return line


def bfs(graph, start):
	"""
	Returns a list of vertices and edges in BFS order
	"""
	bfs_order = []
	marked = [False] * len(graph)
	edge_to = [None] * len(graph)

	queue = [start]
	while len(queue) > 0:
		node = queue.pop(0)
		if not marked[node]:
			marked[node] = True
			bfs_order.append(node)
		for neighbor in graph[node].neighbors:
			neighbor_node = int(neighbor.char)
			if not marked[neighbor_node]:
				edge_to[neighbor_node] = node
				queue.append(neighbor_node)

	print(bfs_order)
	marked = [False] * len(graph)
	marked[bfs_order[0]] = True
	edge_order = []
	bfs_full_order = []
	for node in bfs_order:
		for neighbor in graph[node].neighbors:
			neighbor_node = int(neighbor.char)
			if not marked[neighbor_node]:
				edge_order.append((node, neighbor_node))
				marked[neighbor_node]= True

	for i in range(len(bfs_order) - 1):
		bfs_full_order.append(bfs_order[i])
		bfs_full_order.append(edge_order[i])


	bfs_full_order.append(bfs_order[-1])
	print(bfs_full_order)
	return bfs_full_order

def bfs_random(graph, start):
	"""
	Returns a list of vertices and edges in BFS order
	"""
	bfs_order = []
	marked = [False] * len(graph)
	edge_to = [None] * len(graph)

	queue = [start]
	while len(queue) > 0:
		node = queue.pop(0)
		if not marked[node]:
			marked[node] = True
			bfs_order.append(node)
		random.shuffle(graph[node].neighbors)
		for neighbor in graph[node].neighbors:
			neighbor_node = int(neighbor.char)
			if not marked[neighbor_node]:
				edge_to[neighbor_node] = node
				queue.append(neighbor_node)

	print(bfs_order)
	marked = [False] * len(graph)
	marked[bfs_order[0]] = True
	edge_order = []
	bfs_full_order = []
	for node in bfs_order:
		for neighbor in graph[node].neighbors:
			neighbor_node = int(neighbor.char)
			if not marked[neighbor_node]:
				edge_order.append((node, neighbor_node))
				marked[neighbor_node]= True

	for i in range(len(bfs_order) - 1):
		bfs_full_order.append(bfs_order[i])
		bfs_full_order.append(edge_order[i])


	bfs_full_order.append(bfs_order[-1])
	print(bfs_full_order)
	return bfs_full_order

def coord(x,y,z=0):
    return np.array([x,y,z])

def floodFill(img, row, col, val):
	start = img[row][col]
	queue = [(row, col)]
	visited = set()
	while len(queue) > 0:
		row, col = queue.pop(0)
		visited.add((row, col))
		if img[row][col] == start:
			img[row][col] = val
		for row, col in neighbors(img, row, col, start):
			if (row, col) not in visited:
				queue.append((row, col))

	return img

def neighbors(img, row, col, start):
	indices = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
	return [(row, col) for row, col in indices if isValid(img, row, col, start)]

def isValid(img, row, col, start):
	return row >= 0 and col >= 0 and row < len(img) and col < len(img[0]) and img[row][col] == start
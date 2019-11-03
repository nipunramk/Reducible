from big_ol_pile_of_manim_imports import *


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


class BFS(Scene):

	def construct(self):
		graph = self.construct_graph()

		node = graph[0]
		self.draw_graph(node, BLUE, WHITE, ORANGE)



		node_X = self.get_node(graph, 'X')
		copy_circle = node_X.circle.copy()
		copy_circle.set_color(RED)
		self.play(Transform(node_X.circle, copy_circle))

		node_E = self.get_node(graph, 'E')
		copy_circle = node_E.circle.copy()
		copy_circle.set_color(GREEN)
		self.play(Transform(node_E.circle, copy_circle))

		self.wait(2)
		shortest_path = TextMobject("How do you find the shortest", "path from X to E?")
		shortest_path[0].move_to(LEFT * 3.7 + UP * 3.5)
		shortest_path[1].next_to(shortest_path[0], DOWN)
		self.play(FadeIn(shortest_path))

		self.wait(3)

		default_char = create_computer_char(color=BLUE, scale=0.7, position=UP * 0.5 + LEFT * 4.5)
		confused_character = create_confused_char(color=RED, scale=0.7, position=DOWN*3.5 + LEFT * 5)

		self.play(FadeInAndShiftFromDirection(default_char, LEFT))
		
		
		thought_bubble = SVGMobject("thought")
		thought_bubble.scale(0.9)
		thought_bubble.move_to(UP * 1.5 + LEFT * 3)

		bfs_text = TextMobject("Just use", "BFS!")
		bfs_text.scale(0.6)
		bfs_text.set_color(BLACK)
		bfs_text[0].move_to(thought_bubble.get_center() + UP * 0.4 + RIGHT * 0.05)
		bfs_text[1].move_to(thought_bubble.get_center() + UP * 0.1 + RIGHT * 0.05)

		self.play(FadeIn(thought_bubble), FadeIn(bfs_text))

		# bfs_approach = TextMobject("Let me just find", r"my algorithm kit $\ldots$", r"ah, just use BFS!")

		# bfs_approach[0].move_to(LEFT * 4 + UP * 1.8)
		# bfs_approach[1].next_to(bfs_approach[0], DOWN)
		# bfs_approach[2].next_to(bfs_approach[1], DOWN)
		# self.play(FadeIn(bfs_approach[:2]))
		# self.wait()
		# self.play(Write(bfs_approach[2]))

		self.wait(3)


		default_char2 = create_computer_char(color=RED, scale=0.7, position=UP * 0.5 + LEFT * 4.5)
		default_char2.shift(DOWN * 3)
		thought_bubble2 = thought_bubble.copy()
		thought_bubble2.shift(DOWN * 3)
		self.play(FadeInAndShiftFromDirection(default_char2, LEFT))
		confused_character.move_to(default_char2.get_center())
		self.play(ReplacementTransform(default_char2, confused_character))
		why_text = TextMobject("But why?")
		why_text.scale(0.6)
		why_text.set_color(BLACK)
		why_text.move_to(thought_bubble2.get_center() + UP * 0.2 + RIGHT * 0.05)
		self.play(FadeIn(thought_bubble2), FadeIn(why_text))

		self.wait(3)

		self.play(FadeOutAndShift(default_char, LEFT), FadeOutAndShift(thought_bubble, LEFT), 
			FadeOutAndShift(confused_character, LEFT), FadeOutAndShift(thought_bubble2, LEFT))
		self.wait()

		# why = TextMobject("This approach is", "not satisfying.", "Why?")
		# why[0].move_to(LEFT * 4.3 + DOWN * 0.5 )
		# why[1:].move_to(LEFT * 4.1 + DOWN * 1.2)
		# self.play(FadeIn(why[:2]))

		# self.wait()
		# self.play(Write(why[2]))
		# copy_why = why[2].copy()
		# copy_why.set_color(RED)
		# self.wait()
		# self.play(Transform(why[2], copy_why))
		# self.wait()


		self.bfs(node_X, node_E)
		self.wait(2)

	def bfs(self, start, end):
		queue = [start]
		while len(queue) > 0:
			current = queue.pop(0)
			# print(current)

			if current != start:
				node, edge = current
				if not node.marked:
					copy_node = node.circle.copy()
					copy_node.set_color(YELLOW)
					edge_copy = edge.copy()
					edge_copy.set_color(YELLOW)
					node.marked = True
					self.play(Indicate(node.circle, scale_factor=1, color=PINK), Indicate(edge, scale_factor=1,color=PINK), run_time=0.5)
					self.play(Transform(node.circle, copy_node), Transform(edge, edge_copy), run_time=0.5)
					if node == end:

						break
					for neighbor in node.neighbors:
						edge = self.get_edge(node, neighbor)
						if neighbor.prev == None:
							neighbor.prev = (node, edge)
						queue.append((neighbor, edge))

			elif not current.marked:	
				copy_current = current.circle.copy()
				copy_current.set_color(YELLOW)
				self.play(Transform(current.circle, copy_current), run_time=1)
				current.marked = True
				if current == end:
					break
				for neighbor in current.neighbors:
					edge = self.get_edge(current, neighbor)
					if neighbor.prev == None:
						neighbor.prev = (current, edge)
					queue.append((neighbor, edge))
					
		found_color = GREEN_SCREEN
		self.play(Indicate(end.circle, scale_factor=1, color=found_color), run_time=0.5)
		self.play(Indicate(end.circle, scale_factor=1, color=found_color), run_time=0.5)
		self.play(Indicate(end.circle, scale_factor=1, color=found_color), run_time=0.5)
		copy_end = end.circle.copy()
		copy_end.set_color(found_color)

		path = []
		current = end
		while current != start:
			path.append((current, self.get_edge(current, current.prev[0])))
			current = current.prev[0]

		path.append(current)
		transforms = []
		for elem in path:
			if elem != start:
				node, edge = elem
				self.play(Indicate(node.circle, scale_factor=1, color=found_color), 
					Indicate(edge, scale_factor=1, color=found_color), run_time=0.5)
				copy_node = node.circle.copy()
				copy_edge = edge.copy()
				copy_node.set_color(found_color)
				copy_edge.set_color(found_color)

				self.play(Transform(node.circle, copy_node), Transform(edge, copy_edge), run_time=0.5)
			else:
				node = elem
				self.play(Indicate(node.circle, scale_factor=1, color=found_color), run_time=0.5)
				copy_node = node.circle.copy()
				copy_node.set_color(found_color)
				self.play(Transform(node.circle, copy_node))

		shortest_path = TextMobject("Shortest path is", r"$X \rightarrow V \rightarrow M \rightarrow E$")
		shortest_path[0].move_to(LEFT * 4.1)
		shortest_path[1].next_to(shortest_path[0], DOWN)
		self.play(FadeIn(shortest_path))

		for elem in path[::-1]:
			if elem == start:
				node = elem
				self.play(Indicate(node.circle, scale_factor=1, color=RED), run_time=0.5)
			else:
				node, edge = elem
				self.play(Indicate(node.circle, scale_factor=1, color=RED),
					Indicate(edge, scale_factor=1, color=RED), run_time=0.5)

	def get_edge(self, u, v):
		for edge_u in u.edges:
			for edge_v in v.edges:
				midpoint_u, midpoint_v = edge_u.get_midpoint(), edge_v.get_midpoint()
				if np.allclose(midpoint_u, midpoint_v):
					return edge_u

		return False

	def char_to_index(self, char):
		return ord(char) - ord('A')

	def get_node(self, graph, char):
		index = self.char_to_index(char)
		return graph[index]

	def construct_graph(self):
		graph = []
		radius, scale = 0.4, 0.8
		SHIFT = RIGHT * 2.5
		node_A = GraphNode('A', position=ORIGIN + SHIFT, radius=radius, scale=scale)
		node_B = GraphNode('B', position=UP + RIGHT + SHIFT, radius=radius, scale=scale)
		node_C = GraphNode('C', position=UP * np.sqrt(2) + SHIFT, radius=radius, scale=scale)
		node_D = GraphNode('D', position=RIGHT * np.sqrt(2) + SHIFT, radius=radius, scale=scale)
		node_E = GraphNode('E', position=DOWN + RIGHT + SHIFT, radius=radius, scale=scale)
		node_F = GraphNode('F', position=DOWN * np.sqrt(2) + SHIFT, radius=radius, scale=scale)
		node_G = GraphNode('G', position=DOWN + LEFT + SHIFT, radius=radius, scale=scale)
		node_H = GraphNode('H', position=LEFT * np.sqrt(2) + SHIFT, radius=radius, scale=scale)
		node_I = GraphNode('I', position=UP + LEFT + SHIFT, radius=radius, scale=scale)

		node_J_center = self.get_center_new_node(node_C, node_B, node_A)
		node_J = GraphNode('J', position=node_J_center, radius=radius, scale=scale)
		node_K_center = self.get_center_new_node(node_B, node_D, node_A)
		node_K = GraphNode('K', position=node_K_center, radius=radius, scale=scale)
		node_L_center = self.get_center_new_node(node_D, node_E, node_A)
		node_L = GraphNode('L', position=node_L_center, radius=radius, scale=scale)
		node_M_center = self.get_center_new_node(node_E, node_F, node_A)
		node_M = GraphNode('M', position=node_M_center, radius=radius, scale=scale)
		node_N_center = self.get_center_new_node(node_F, node_G, node_A)
		node_N = GraphNode('N', position=node_N_center, radius=radius, scale=scale)
		node_O_center = self.get_center_new_node(node_G, node_H, node_A)
		node_O = GraphNode('O', position=node_O_center, radius=radius, scale=scale)
		node_P_center = self.get_center_new_node(node_H, node_I, node_A)
		node_P = GraphNode('P', position=node_P_center, radius=radius, scale=scale)
		node_Q_center = self.get_center_new_node(node_I, node_C, node_A)
		node_Q = GraphNode('Q', position=node_Q_center, radius=radius, scale=scale)

		graph.append(node_A)
		graph.append(node_B)
		graph.append(node_C)
		graph.append(node_D)
		graph.append(node_E)
		graph.append(node_F)
		graph.append(node_G)
		graph.append(node_H)
		graph.append(node_I)
		graph.append(node_J)
		graph.append(node_K)
		graph.append(node_L)
		graph.append(node_M)
		graph.append(node_N)
		graph.append(node_O)
		graph.append(node_P)
		graph.append(node_Q)

		node_A.connect(node_B)
		node_A.connect(node_C)
		node_A.connect(node_D)
		node_A.connect(node_E)
		node_A.connect(node_F)
		node_A.connect(node_G)
		node_A.connect(node_H)
		node_A.connect(node_I)

		node_R_center = self.get_center_last_layer(node_A, 0)
		node_R = GraphNode('R', position=node_R_center, radius=radius, scale=scale)
		graph.append(node_R)
		node_S_center = self.get_center_last_layer(node_A, 1)
		node_S = GraphNode('S', position=node_S_center, radius=radius, scale=scale)
		graph.append(node_S)
		node_T_center = self.get_center_last_layer(node_A, 2)
		node_T = GraphNode('T', position=node_T_center, radius=radius, scale=scale)
		graph.append(node_T)
		node_U_center = self.get_center_last_layer(node_A, 3)
		node_U = GraphNode('U', position=node_U_center, radius=radius, scale=scale)
		graph.append(node_U)
		node_V_center = self.get_center_last_layer(node_A, 4)
		node_V = GraphNode('V', position=node_V_center, radius=radius, scale=scale)
		graph.append(node_V)
		node_W_center = self.get_center_last_layer(node_A, 5)
		node_W = GraphNode('W', position=node_W_center, radius=radius, scale=scale)
		graph.append(node_W)
		node_X_center = self.get_center_last_layer(node_A, 6)
		node_X = GraphNode('X', position=node_X_center, radius=radius, scale=scale)
		graph.append(node_X)
		node_Y_center = self.get_center_last_layer(node_A, 7)
		node_Y = GraphNode('Y', position=node_Y_center, radius=radius, scale=scale)
		graph.append(node_Y)

		node_B.connect(node_C)
		node_B.connect(node_D)
		node_D.connect(node_E)
		node_E.connect(node_F)
		node_F.connect(node_G)
		node_G.connect(node_H)
		node_H.connect(node_I)
		node_I.connect(node_C)

		node_C.connect(node_J)
		node_B.connect(node_J)
		node_B.connect(node_K)
		node_D.connect(node_K)
		node_D.connect(node_L)
		node_E.connect(node_L)
		node_E.connect(node_M)
		node_F.connect(node_M)
		node_F.connect(node_N)
		node_G.connect(node_N)
		node_G.connect(node_O)
		node_H.connect(node_O)
		node_H.connect(node_P)
		node_I.connect(node_P)
		node_I.connect(node_Q)
		node_C.connect(node_Q)

		node_Q.connect(node_S)
		node_J.connect(node_S)
		node_J.connect(node_R)
		node_K.connect(node_R)
		node_K.connect(node_T)
		node_L.connect(node_T)
		node_L.connect(node_U)
		node_M.connect(node_U)
		node_M.connect(node_V)
		node_N.connect(node_V)
		node_N.connect(node_W)
		node_O.connect(node_W)
		node_O.connect(node_X)
		node_P.connect(node_X)
		node_P.connect(node_Y)
		node_Q.connect(node_Y)

		node_X.connect_curve(node_O, node_S, node_J, angle=-TAU/3)
		node_S.connect_curve(node_Q, node_T, node_L, angle=-TAU/3)
		node_T.connect_curve(node_K, node_V, node_N, angle=-TAU/3)
		node_V.connect_curve(node_M, node_X, node_P, angle=-TAU/3)

		return graph

	def get_center_new_node(self, node1, node2, node_A):
		middle = (node1.circle.get_center() + node2.circle.get_center()) / 2
		line_through_middle = Line(node_A.circle.get_center(), middle)
		unit_vector = line_through_middle.get_unit_vector()
		new_point = node_A.circle.get_center() + unit_vector * 2.5
		return new_point

	def get_center_last_layer(self, node_A, index):
		line = node_A.edges[index].get_unit_vector()
		new_point = node_A.circle.get_center() + line * 3.4
		return new_point

	def draw_graph(self, node, graph_color, data_color, line_color):
		stack = [node]
		all_circles = []
		all_data = []
		edges = set()
		while len(stack) > 0:
			node = stack.pop(0)
			if not node.drawn:
				node.circle.set_color(graph_color)
				node.data.set_color(data_color)
				all_circles.append(node.circle)
				all_data.append(node.data)
				node.drawn = True
				for graph_node in node.neighbors:
					stack.append(graph_node)

				for edge in node.edges:
					if edge not in edges:
						edges.add(edge)

		for edge in edges:
			edge.set_color(line_color)

		animations = []
		for circle, data in zip(all_circles, all_data):
			animations.append(GrowFromCenter(circle))
			animations.append(FadeIn(data))

		self.play(*animations, run_time=1.5)

		self.wait()
		edges = list(set(edges))

		self.play(*[ShowCreation(edge) for edge in edges], run_time=1.5)


class FFTNode:
	def __init__(self, position, color):
		self.position = position
		self.shape = Dot()
		self.shape.set_color(color)
		self.shape.move_to(position)
		self.neighbors = [] # tuple ((tuple vertex): line object)


class FFT(Scene):
	def construct(self):
		grid = [[0 for i in range(8)] for j in range(4)]
		all_shapes = []
		for i in range(len(grid)):
			for j in range(len(grid[0])):
				position = (i - 1.5) * RIGHT * 2 + (j - 3.1) * DOWN
				if j % 2 == 1:
					position += UP * 0.2
				node = FFTNode(position, BLACK)
				if i == 0:
					node.shape = Rectangle(height=0.5, width=0.5, fill_color=BLUE, fill_opacity=1)
					node.shape.move_to(position)
					node.shape.set_color(BLUE)
				grid[i][j] = node
				all_shapes.append(node.shape)
				# self.play(FadeIn(node.shape))
		text_dict = {0: "$a_0$", 1: "$a_4$", 2: "$a_2$", 3: "$a_6$", 4: "$a_1$", 5: "$a_5$", 6: "$a_3$", 7: "$a_7$",}
		binary_dict = {0: "$000$", 1: "$100$", 2: "$010$", 3: "$110$", 4: "$001$", 5: "$101$", 6: "$011$", 7: "$111$",}
		right_text = [0] * 8
		right_binary_text = [0] * 8
		for i in range(len(grid)):
			for j in range(len(grid[0])):
				if i == 1 and j % 2 == 0:
					top_dot, bottom_dot = grid[i][j].shape, grid[i][j+1].shape
					rect_pos = (top_dot.get_center() + bottom_dot.get_center()) / 2
					rect_height = np.linalg.norm((top_dot.get_center() - bottom_dot.get_center())) + 0.5
					rect_width = 0.5
					rect = Rectangle(height=rect_height, width=rect_width, fill_color=BLUE, fill_opacity=1)
					rect.move_to(rect_pos)
					rect.set_color(BLUE)
					# self.play(FadeIn(rect))
					all_shapes.insert(0, rect)
				elif i == 2 and j % 4 == 0:
					top_dot, bottom_dot = grid[i][j].shape, grid[i][j+3].shape
					rect_pos = (top_dot.get_center() + bottom_dot.get_center()) / 2
					rect_height = np.linalg.norm((top_dot.get_center() - bottom_dot.get_center())) + 0.5
					rect_width = 0.5
					rect = Rectangle(height=rect_height, width=rect_width, fill_color=BLUE, fill_opacity=1)
					rect.move_to(rect_pos)
					rect.set_color(BLUE)
					# self.play(FadeIn(rect))
					all_shapes.insert(0, rect)
				elif i == 3 and j == 0:
					top_dot, bottom_dot = grid[i][j].shape, grid[i][j+7].shape
					rect_pos = (top_dot.get_center() + bottom_dot.get_center()) / 2
					rect_height = np.linalg.norm((top_dot.get_center() - bottom_dot.get_center())) + 0.5
					rect_width = 0.5
					rect = Rectangle(height=rect_height, width=rect_width, fill_color=BLUE, fill_opacity=1)
					rect.move_to(rect_pos)
					rect.set_color(BLUE)
					# self.play(FadeIn(rect))
					all_shapes.insert(0, rect)
					text = TextMobject(r"$A(\omega ^ {0})$".format(j))
					text.scale(0.6)
					text.next_to(grid[i][j].shape, RIGHT * 2)

					binary_text = TextMobject(binary_dict[j])
					binary_text.scale(0.6)
					binary_text.next_to(text, RIGHT * 2)

					all_shapes.append(binary_text)
					all_shapes.append(text)
					right_text[j] = text
					right_binary_text[j] = binary_text

				elif i == 0:
					text = TextMobject(text_dict[j])
					text.scale(0.6)
					text.set_color(BLACK)
					text.move_to(grid[i][j].shape.get_center())
					all_shapes.append(text)

					binary_text = TextMobject(binary_dict[j])
					binary_text.scale(0.6)
					binary_text.next_to(text, LEFT * 2)
					all_shapes.append(binary_text)

				elif i == 3:
					text = TextMobject(r"$A(\omega ^ {0})$".format(j))
					text.scale(0.6)
					text.next_to(grid[i][j].shape, RIGHT * 2)
					all_shapes.append(text)

					binary_text = TextMobject(binary_dict[j])
					binary_text.scale(0.6)
					binary_text.next_to(text, RIGHT * 2)

					all_shapes.append(binary_text)
					right_text[j] = text
					right_binary_text[j] = binary_text


		column1_to_2_lines = [[0 for i in range(8)] for j in range(8)]
		column1, column2 = grid[0], grid[1]
		for i in range(len(column1_to_2_lines)):
			if i % 2 == 0:
				line_start00 = self.get_line_start_col1(column1[i])
				line_end00 = self.get_line_end(column2[i])
				line00 = Line(line_start00, line_end00)
				column1_to_2_lines[i][i] = line00
				column2[i].neighbors.append(((0, i), line00))

				line_start01 = self.get_line_start_col1(column1[i])
				line_end01 = self.get_line_end(column2[i + 1])
				line01 = Line(line_start01, line_end01)
				column1_to_2_lines[i][i + 1] = line01
				column2[i + 1].neighbors.append(((0, i), line01))

				line_start10 = self.get_line_start_col1(column1[i + 1])
				line_end10 = self.get_line_end(column2[i])
				line10 = Line(line_start10, line_end10)
				column1_to_2_lines[i + 1][i] = line10
				column2[i].neighbors.append(((0, i + 1), line10))

				line_start11 = self.get_line_start_col1(column1[i + 1])
				line_end11 = self.get_line_end(column2[i + 1])
				line11 = Line(line_start11, line_end11)
				column1_to_2_lines[i + 1][i + 1] = line11
				column2[i + 1].neighbors.append(((0, i + 1), line11))

		lines1 = []
		for row in column1_to_2_lines:
			for line in row:
				if line != 0:
					lines1.append(line)


		column2_to_3_lines = [[0 for i in range(8)] for j in range(8)]
		column2, column3 = grid[1], grid[2]
		for i in range(len(column2_to_3_lines)):
			if i < 6:
				line_start00 = self.get_line_start_col(column2[i])
				line_end00 = self.get_line_end(column3[i])
				line00 = Line(line_start00, line_end00)
				column2_to_3_lines[i][i] = line00
				column3[i].neighbors.append(((1, i), line00))

				if i != 2 and i != 3:
					line_start02 = self.get_line_start_col(column2[i])
					line_end02 = self.get_line_end(column3[i + 2])
					line02 = Line(line_start02, line_end02)
					column2_to_3_lines[i][i + 2] = line02
					column3[i + 2].neighbors.append(((1, i), line02))

			if i > 1 and i != 4 and i != 5:
				line_start20 = self.get_line_start_col(column2[i])
				line_end20 = self.get_line_end(column3[i - 2])
				line20 = Line(line_start20, line_end20)
				column2_to_3_lines[i][i - 2] = line20
				column3[i - 2].neighbors.append(((1, i), line20))

			if i > 5:
				line_start00 = self.get_line_start_col(column2[i])
				line_end00 = self.get_line_end(column3[i])
				line00 = Line(line_start00, line_end00)
				column2_to_3_lines[i][i] = line00
				column3[i].neighbors.append(((1, i), line00))

		lines2 = []
		for row in column2_to_3_lines:
			for line in row:
				if line != 0:
					lines2.append(line)

		column3_to_4_lines = [[0 for i in range(8)] for j in range(8)]
		column3, column4 = grid[2], grid[3]
		for i in range(len(column3_to_4_lines)):
			line_start00 = self.get_line_start_col(column3[i])
			line_end00 = self.get_line_end(column4[i])
			line00 = Line(line_start00, line_end00)
			column3_to_4_lines[i][i] = line00
			column4[i].neighbors.append(((2, i), line00))
			if i < 4:
				line_start04 = self.get_line_start_col(column3[i])
				line_end04 = self.get_line_end(column4[i + 4])
				line04 = Line(line_start04, line_end04)
				column3_to_4_lines[i][i + 4] = line04
				column4[i + 4].neighbors.append(((2, i), line04))
			else:
				line_start40 = self.get_line_start_col(column3[i])
				line_end40 = self.get_line_end(column4[i - 4])
				line40 = Line(line_start40, line_end40)
				column3_to_4_lines[i][i - 4] = line40
				column4[i - 4].neighbors.append(((2, i), line40))

		lines3 = []
		for row in column3_to_4_lines:
			for line in row:
				if line != 0:
					lines3.append(line)

		self.play(*[FadeIn(shape) for shape in all_shapes])
		self.play(*[GrowFromCenter(line) for line in lines1])
		self.play(*[GrowFromCenter(line) for line in lines2])
		self.play(*[GrowFromCenter(line) for line in lines3])

		unraveling = TextMobject("Unraveling the Fast Fourier Transform")
		unraveling.scale(0.6)
		unraveling.shift(UP * 3.7)
		self.play(FadeIn(unraveling))

		for j in range(8):
			text = VGroup(right_binary_text[j], right_text[j])
			text.set_color(GREEN_SCREEN)
			all_lines = self.bfs(grid[3][j], grid) + [text]
			all_copies = [line.copy() for line in all_lines]
			[copy.set_color(WHITE) for copy in all_copies]
			self.play(*[Transform(line, copy) for line, copy in zip(all_lines, all_copies)], run_time=0.4)

		self.wait()

	def bfs(self, start_node, grid):
		queue = [start_node]
		seen = set()
		all_lines = []
		while len(queue) > 0:
			current = queue.pop(0)
			if current not in seen:
				lines = [tup[1] for tup in current.neighbors]
				lines_copy = [line.copy() for line in lines]
				[line_copy.set_color(GREEN_SCREEN) for line_copy in lines_copy]
				self.play(*[Transform(line, line_copy) for line, line_copy in zip(lines, lines_copy)], run_time=0.4)
				all_lines.extend(lines)
				seen.add(current)
				neighbor_coords = [tup[0] for tup in current.neighbors]
				for coord in neighbor_coords:
					queue.append(grid[coord[0]][coord[1]])
		return all_lines


	def get_line_start_col1(self, fft_node):
		line_start = fft_node.shape.get_center()
		line_start = line_start + RIGHT * fft_node.shape.get_width() * 0.5
		return line_start

	def get_line_start_col(self, fft_node):
		line_start = fft_node.shape.get_center()
		line_start = line_start + RIGHT * fft_node.shape.radius
		return line_start

	def get_line_end(self, fft_node):
		line_end = fft_node.shape.get_center()
		line_end = line_end + LEFT * fft_node.shape.radius
		return line_end


class Topics(Scene):
	def construct(self):
		run_time = 1.2
		data_structures = TextMobject("Data Structures")
		data_structures.move_to(LEFT * 3.5 + UP * 3.5)
		data_structures.set_color(BLUE)
		self.play(Write(data_structures), run_time=run_time)
		scale = 0.8
		arrays = TextMobject("Dynamic Arrays/Array Lists")
		arrays.scale(scale)
		arrays.next_to(data_structures, DOWN)
		self.play(Write(arrays), run_time=run_time)

		recursion = TextMobject("Recursion")
		recursion.scale(scale)
		recursion.next_to(arrays, DOWN)
		self.play(Write(recursion), run_time=run_time)

		big_o = TextMobject("Big O Notation")
		big_o.scale(scale)
		big_o.next_to(recursion, DOWN)
		self.play(Write(big_o), run_time=run_time)

		linked_lists = TextMobject("Linked Lists")
		linked_lists.scale(scale)
		linked_lists.next_to(big_o, DOWN)
		self.play(Write(linked_lists), run_time=run_time)

		disjoint_sets = TextMobject("Disjoint Sets")
		disjoint_sets.scale(scale)
		disjoint_sets.next_to(linked_lists, DOWN)
		self.play(Write(disjoint_sets), run_time=run_time)

		trees = TextMobject("Trees, Binary Search Trees")
		trees.scale(scale)
		trees.next_to(disjoint_sets, DOWN)
		self.play(Write(trees), run_time=run_time)

		hashing = TextMobject("Hashmaps/Dictionaries")
		hashing.scale(scale)
		hashing.next_to(trees, DOWN)
		self.play(Write(hashing), run_time=run_time)

		heaps = TextMobject("Heaps and Priority Queues")
		heaps.scale(scale)
		heaps.next_to(hashing, DOWN)
		self.play(Write(heaps), run_time=run_time)

		tries = TextMobject("Heaps and Priority Queues")
		tries.scale(scale)
		tries.next_to(heaps, DOWN)
		self.play(Write(tries), run_time=run_time)

		spatial = TextMobject("Spatial Data Structures")
		spatial.scale(scale)
		spatial.next_to(tries, DOWN)
		self.play(Write(spatial), run_time=run_time)

		graphs = TextMobject("Graphs")
		graphs.scale(scale)
		graphs.next_to(spatial, DOWN)
		self.play(Write(graphs), run_time=run_time)

		algorithms = TextMobject("Algorithms")
		algorithms.move_to(RIGHT * 3.5 + UP * 3.5)
		algorithms.set_color(BLUE)
		self.play(Write(algorithms), run_time=run_time)

		bfs_dfs = TextMobject("BFS and DFS")
		bfs_dfs.scale(scale)
		bfs_dfs.next_to(algorithms, DOWN)
		self.play(Write(bfs_dfs), run_time=run_time)

		dijsktra = TextMobject("Dijsktra's Algorithm and A* Search")
		dijsktra.scale(scale)
		dijsktra.next_to(bfs_dfs, DOWN)
		self.play(Write(dijsktra), run_time=run_time)

		# a_star = TextMobject("A* search")
		# a_star.scale(scale)
		# a_star.next_to(dijsktra, DOWN)
		# self.play(Write(a_star))

		mst = TextMobject("Minimum Spanning Trees")
		mst.scale(scale)
		mst.next_to(dijsktra, DOWN)
		self.play(Write(mst), run_time=run_time)

		sorting_algos = TextMobject("Sorting Algorithms")
		sorting_algos.scale(scale)
		sorting_algos.next_to(mst, DOWN)
		self.play(Write(sorting_algos), run_time=run_time)

		divide = TextMobject("Divide and Conquer Algorithms")
		divide.scale(scale)
		divide.next_to(sorting_algos, DOWN)
		self.play(Write(divide), run_time=run_time)

		fft = TextMobject("The Fast Fourier Transform")
		fft.scale(scale)
		fft.next_to(divide, DOWN)
		self.play(Write(fft), run_time=run_time)

		dynamic = TextMobject("Dynamic Programming")
		dynamic.scale(scale)
		dynamic.next_to(fft, DOWN)
		self.play(Write(dynamic), run_time=run_time)

		linear = TextMobject("Linear Programming")
		linear.scale(scale)
		linear.next_to(dynamic, DOWN)
		self.play(Write(linear), run_time=run_time)

		max_flow = TextMobject("Max Flow and Duality")
		max_flow.scale(scale)
		max_flow.next_to(linear, DOWN)
		self.play(Write(max_flow), run_time=run_time)

		simplex = TextMobject("The Simplex Algorithm")
		simplex.scale(scale)
		simplex.next_to(max_flow, DOWN)
		self.play(Write(simplex), run_time=run_time)

		reductions = TextMobject("NP-completeness and Reductions")
		reductions.scale(scale)
		reductions.next_to(simplex, DOWN)
		self.play(Write(reductions), run_time=run_time)

		self.wait(7)


class Audience(Scene):
	def construct(self):
		experience = TextMobject("What do I need to know before learning this material?")
		experience.shift(UP)
		self.play(FadeIn(experience))

		self.wait(12)
		programming = TextMobject("Basic understanding of a programming language")
		self.play(Write(programming))

		self.wait(3)
		worry = TextMobject("The programming language isn't", "going to be the main focus")
		worry.set_color(GREEN)
		worry[0].move_to(LEFT * 2 + DOWN * 1.5)
		worry[1].next_to(worry[0], DOWN)

		self.play(FadeIn(worry))

		self.wait(3)
		python_image = ImageMobject("python")
		python_image.shift(DOWN * 2 + RIGHT * 3)
		self.play(FadeIn(python_image))


		self.wait(5)	

		self.play(FadeOut(experience), FadeOut(programming), FadeOut(python_image),
			FadeOut(worry))

		audience = TextMobject("Who is this series for?")

		audience.shift(UP * 3.5)
		self.play(FadeIn(audience))

		# camps = TextMobject("3 camps")
		# camps.shift(UP * 2.5)
		left_character = create_computer_char(color=RED, scale=0.8, position=LEFT * 5 + DOWN * 1.5)
		left_thought_bubble = SVGMobject("thought")
		left_thought_bubble.scale(1.6)
		left_thought_bubble.move_to(left_character.get_center() + UP * 2.2 + RIGHT * 2.2)
		experienced = TextMobject("What if I already", "took a data", "structures class?")
		experienced.scale(0.6)
		experienced[0].move_to(left_thought_bubble.get_center() + UP * 0.75 + RIGHT * 0.05)
		experienced[1].move_to(left_thought_bubble.get_center() + UP * 0.45 + RIGHT * 0.05)
		experienced[2].move_to(left_thought_bubble.get_center() + UP * 0.15 + RIGHT * 0.05)
		experienced.set_color(BLACK)

		middle_character = create_computer_char(color=GREEN, scale=0.8, position= LEFT + DOWN * 1.5)
		middle_thought_bubble = SVGMobject("thought")
		middle_thought_bubble.scale(1.6)
		middle_thought_bubble.move_to(middle_character.get_center() + UP * 2.2 + RIGHT * 2.2)
		in_class = TextMobject("What if I'm in", "a data structures", "class right now?")
		in_class.scale(0.6)
		in_class[0].move_to(middle_thought_bubble.get_center() + UP * 0.75 + RIGHT * 0.05)
		in_class[1].move_to(middle_thought_bubble.get_center() + UP * 0.45 + RIGHT * 0.05)
		in_class[2].move_to(middle_thought_bubble.get_center() + UP * 0.15 + RIGHT * 0.05)
		in_class.set_color(BLACK)

		right_character = create_computer_char(color=BLUE, scale=0.8, position=RIGHT * 3 + DOWN * 1.5)
		right_thought_bubble = SVGMobject("thought")
		right_thought_bubble.scale(1.6)
		right_thought_bubble.move_to(right_character.get_center() + UP * 2.2 + RIGHT * 2.2)
		beginner = TextMobject("What if I just", "started learning", "computer science?")
		beginner.scale(0.6)
		beginner[0].move_to(right_thought_bubble.get_center() + UP * 0.75 + RIGHT * 0.05)
		beginner[1].move_to(right_thought_bubble.get_center() + UP * 0.45 + RIGHT * 0.05)
		beginner[2].move_to(right_thought_bubble.get_center() + UP * 0.15 + RIGHT * 0.05)
		beginner.set_color(BLACK)

		self.wait(5)
		# self.play(FadeIn(camps))
		self.play(FadeInAndShiftFromDirection(left_character))
		self.play(FadeIn(left_thought_bubble), FadeIn(experienced))
		self.wait(11)
		self.play(FadeInAndShiftFromDirection(middle_character))
		self.play(FadeIn(middle_thought_bubble), FadeIn(in_class))
		self.wait(7)
		self.play(FadeInAndShiftFromDirection(right_character))
		self.play(FadeIn(right_thought_bubble), FadeIn(beginner))
		self.wait(13)





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

















		


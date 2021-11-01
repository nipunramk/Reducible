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

	def connect_curved_arrow(self, other, direction=UP, angle=TAU/4):
		curve_start = self.circle.get_center() + direction * self.radius
		curve_end = other.circle.get_center() + direction * self.radius
		line = CurvedArrow(curve_start, curve_end, angle=angle, tip_length=0.2)
		self.neighbors.append(other)
		self.edges.append(line)
		return line


	def __repr__(self):
		return 'GraphNode({0})'.format(self.char)

	def __str__(self):
		return 'GraphNode({0})'.format(self.char)

class IntroDP(Scene):
	def construct(self):
		title = TextMobject("Dynamic Programming")
		title.scale(1.2)
		title.shift(UP * 3.5)

		h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
		h_line.next_to(title, DOWN)
		self.play(
			Write(title),
			ShowCreation(h_line)
		)

		self.wait(8)

		definition = BulletedList(
			"Identifying and solving subproblems",
			"Using subproblems together to solve larger problem"
		)
		
		definition.next_to(h_line, DOWN)

		self.play(
			Write(definition[0])
		)

		self.play(
			Write(definition[1])
		)

		self.wait(11)

		self.play(
			FadeOut(definition),
		)

		self.wait()

		steps = TextMobject(r"5 STEPS ", r"$\rightarrow$ 2 PROBLEMS")
		steps.scale(1.1)
		steps.next_to(h_line, DOWN)

		self.play(
			Write(steps[0])
		)

		self.wait(6)

		self.play(
			Write(steps[1])
		)

		# self.play(
		# 	Write(steps[1])
		# )

		rect_1 = ScreenRectangle(height=3.2)
		rect_1.move_to(LEFT * 3)
		rect_2 = ScreenRectangle(height=3.2)
		rect_2.move_to(RIGHT * 3)
		self.play(
			ShowCreation(rect_1),
			ShowCreation(rect_2)
		)

		fundamental = TextMobject("Fundamental")
		fundamental.next_to(rect_1, DOWN)
		self.play(
			FadeIn(fundamental)
		)

		self.wait(2)

		challenging = TextMobject("Challenging")
		challenging.next_to(rect_2, DOWN)
		self.play(
			FadeIn(challenging)
		)

		self.wait(4)

		rect_3 = ScreenRectangle(height=5)
		rect_3.move_to(DOWN * 0.5)

		self.play(
			ReplacementTransform(rect_1, rect_3),
			ReplacementTransform(rect_2, rect_3),
			FadeOut(fundamental),
			FadeOut(challenging),
			run_time=2
		)

		finding_subproblems = TextMobject("Guide to Finding Subproblems")
		finding_subproblems.next_to(rect_3, DOWN)
		self.play(
			FadeIn(finding_subproblems)
		)
		self.wait(4)

class MakeGrids(Scene):
	def construct(self):
		all_grids = self.make_matrices()

		self.play(
			FadeOut(all_grids)
		)

	def make_matrices(self):
		grids = []

		colors = [RED, ORANGE, YELLOW, GREEN_SCREEN, BLUE, VIOLET]
		for i in range(6):
			grid = self.create_grid(i, i, 0.5)
			grid_group = self.get_group_grid(grid)
			grid_group.set_fill(color=colors[i], opacity=0.3)
			grids.append(grid_group)

		all_grids = VGroup(*grids)

		all_grids.move_to(DOWN)

		original_positions = [all_grids[:i].get_center() for i in range(2, 7)]

		self.play(
			ShowCreation(all_grids),
			run_time=3
		)

		self.wait(3)


		self.play(
			all_grids.shift, RIGHT * 5
		)

		self.play(
			all_grids[:5].shift, LEFT * 3,
			rate_func=smooth
		)

		self.play(
			all_grids[:4].shift, LEFT * 2.5,
			rate_func=smooth
		)

		self.play(
			all_grids[:3].shift, LEFT * 2,
			rate_func=smooth
		)

		self.play(
			all_grids[:2].shift, LEFT * 1.5,
			rate_func=smooth
		)

		self.wait()

		self.play(
			all_grids[5].shift, LEFT * 3,
			rate_func=smooth
		)

		self.play(
			all_grids[4:].shift, LEFT * 2.5,
			rate_func=smooth
		)

		self.play(
			all_grids[3:].shift, LEFT * 2,
			rate_func=smooth
		)

		self.play(
			all_grids[2:].shift, LEFT * 1.5,
			rate_func=smooth
		)

		self.play(
			all_grids.move_to, DOWN,
			rate_func=smooth
		)


		self.wait(3)

		return all_grids

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

class BoxProblem(Scene):
	def construct(self):
		title = TextMobject("Box Stacking")
		title.scale(1.2)
		title.shift(UP * 3.5)

		h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
		h_line.next_to(title, DOWN)
		self.play(
			Write(title),
			ShowCreation(h_line)
		)

		problem = self.introduce_problem(h_line)
		self.show_examples(problem, h_line, title)
		# dimensions = [(5, 3, 2), (3, 2, 1)]
		# dimensions = [self.convert_to_display_dimesions(d) for d in dimensions]
		# boxes = self.make_stack_of_boxes(dimensions, [RED, BLUE])

		# boxes = VGroup(*boxes)
		# boxes.scale(0.5)
		# boxes.shift(DOWN)
		# self.play(
		# 	FadeIn(boxes)
		# )

		# self.wait()

	def convert_to_display_dimesions(self, dimension):
		return (dimension[1], dimension[2], dimension[0])

	def convert_to_input_dimensions(self, dimension):
		return (dimension[2], dimension[0], dimension[1])


	def introduce_problem(self, h_line):
		problem = TextMobject(
			r"Given $n$ boxes  $[ (L_1, W_1, H_1), (L_2, W_2, H_2), \ldots , (L_n, W_n, H_n) ]$ where box $i$ has" + "\\\\",
			r"length $L_i$, width $W_i$, and height $H_i$, find the height of the tallest possible stack." + "\\\\",
			r"Box $(L_i, W_i, H_i)$ can be on top of box $(L_j, W_j, H_j) \text{ if } L_i < L_j, W_i < W_j$."
		)

		problem.scale(0.6)
		problem.next_to(h_line, DOWN)
		self.wait(2)

		self.play(
			Write(problem[0])
		)

		self.play(
			Write(problem[1])
		)

		self.wait(3)

		self.play(
			Write(problem[2])
		)

		self.wait(13)

		return problem

	def show_examples(self, problem, h_line, title):
		box_dims = [
		(2, 3, 3),
		(2, 2, 4),
		(4, 4, 2),
		]

		display_dims = [self.convert_to_display_dimesions(d) for d in box_dims]

		first_example = TextMobject(r"$[(2, 3, 3), (2, 2, 4), (4, 4, 2)]$")
		first_example.scale(0.6)
		first_example.next_to(problem, DOWN)

		self.play(
			Write(first_example)
		)

		scale_factor = 0.3
		box_shapes = []
		box_colors = [RED, GREEN_SCREEN, BLUE]
		for i, display_dim in enumerate(display_dims):
			l, w, h = display_dim
			box = self.construct_box(l=l, w=w, h=h, color=box_colors[i])
			box.scale(scale_factor)
			box_shapes.append(box)
		
		box_shapes[0].next_to(first_example, DOWN)
		box_shapes[0].shift(LEFT * 3.5)

		box_shapes[1].next_to(box_shapes[0], DOWN)

		box_shapes[2].next_to(box_shapes[1], DOWN)

		self.play(
			*[FadeIn(box) for box in box_shapes]
		)

		self.wait(2)

		stack = self.make_stack_of_boxes(
			[display_dims[2], display_dims[1]], 
			[box_colors[2], box_colors[1]]
		)
		stack = VGroup(*stack)
		stack.scale(scale_factor)
		stack.move_to(DOWN)
		self.play(
			TransformFromCopy(box_shapes[2], stack[0]),
			TransformFromCopy(box_shapes[1], stack[1]),	
		)

		self.wait()

		best_height = TextMobject(r"height $= 6$")
		best_height.scale(0.6)
		best_height.next_to(stack, RIGHT * 2)
		self.play(
			FadeIn(best_height)
		)

		self.wait(11)

		self.play(
			FadeOut(best_height),
			FadeOut(stack),
			FadeOut(first_example),
			*[FadeOut(s) for s in box_shapes],
		)

		box_dims = [
		(4, 5, 3),
		(2, 3, 2),
		(3, 6, 2),
		(1, 5, 4),
		(2, 4, 1),
		(1, 2, 2),
		]

		display_dims = [self.convert_to_display_dimesions(d) for d in box_dims]

		second_example = TextMobject(r"$[(4, 5, 3), (2, 3, 2), (3, 6, 2), (1, 5, 4), (2, 4, 1), (1, 2, 2)]$")
		second_example.scale(0.6)
		second_example.next_to(problem, DOWN)
		self.play(
			FadeIn(second_example)
		)

		scale_factor = 0.3
		box_shapes = []
		box_colors = [RED, ORANGE, YELLOW, GREEN_SCREEN, BLUE, VIOLET]
		for i, display_dim in enumerate(display_dims):
			l, w, h = display_dim
			box = self.construct_box(l=l, w=w, h=h, color=box_colors[i])
			box.scale(scale_factor)
			box_shapes.append(box)
		
		box_shapes[0].next_to(first_example, DOWN)
		box_shapes[0].shift(LEFT * 4.5 + DOWN * 0.5)

		box_shapes[1].next_to(box_shapes[0], DOWN)

		box_shapes[2].next_to(box_shapes[1], DOWN)

		box_shapes[3].next_to(box_shapes[0], RIGHT)

		box_shapes[4].next_to(box_shapes[3], DOWN)

		box_shapes[5].next_to(box_shapes[4], DOWN)

		self.play(
			*[FadeIn(box) for box in box_shapes]
		)

		self.wait(13)

		stack = self.make_stack_of_boxes(
			[display_dims[0], display_dims[1], display_dims[5]], 
			[box_colors[0], box_colors[1], box_colors[5]]
		)
		stack = VGroup(*stack)
		stack.scale(scale_factor)
		stack.move_to(DOWN + RIGHT * 1.5)


		self.play(
			TransformFromCopy(box_shapes[0], stack[0]),
			TransformFromCopy(box_shapes[1], stack[1]),	
			TransformFromCopy(box_shapes[5], stack[2]),
			run_time=2
		)

		self.wait(3)

		best_height = TextMobject(r"height $= 7$")
		best_height.scale(0.6)
		best_height.next_to(stack, RIGHT * 2)
		self.play(
			FadeIn(best_height)
		)

		self.wait(12)

		step_1 = TextMobject("1. Visualize Examples")
		step_1.scale(0.8)
		step_1. move_to(DOWN * 3.5)

		self.play(
			Write(step_1),
		)

		self.wait(13)

		box_shapes_group = VGroup(*box_shapes)

		self.play(
			FadeOut(stack),
			FadeOut(best_height),
			FadeOut(problem[0]),
			FadeOut(problem[1]),
			FadeOut(second_example),
			problem[2].move_to, UP * 3,
			problem[2].scale, 1.2,
			box_shapes_group.move_to, DOWN * 0.5,
			FadeOut(title),
			FadeOut(h_line),
			run_time=2
		)

		self.wait(13)

		self.visualize_example(box_shapes, stack, problem, step_1)

	def visualize_example(self, box_shapes, stack, problem, step_1):
		new_locations = [
		RIGHT * 4 + DOWN * 2.5, 
		DOWN * 0.5,
		RIGHT * 4 + DOWN * 0.5,
		UP * 1.5,
		DOWN * 2.5,
		LEFT * 4 + DOWN * 1.5
		]

		box_shapes_copy = [box_shapes[i].copy().move_to(new_locations[i]) for i in range(len(box_shapes))]

		transforms = [Transform(box_shapes[i], box_shapes_copy[i]) for i in range(len(box_shapes))]
		self.play(
			*transforms,
			run_time=2
		)

		edges = {}

		edges[(1, 2)] = self.connect_arrow_between(box_shapes[1], box_shapes[2])
		edges[(1, 0)] = self.connect_arrow_between(box_shapes[1], box_shapes[0])
		
		edges[(3, 2)] = self.connect_arrow_between(box_shapes[3], box_shapes[2])

		edges[(4, 2)] = self.connect_arrow_between(box_shapes[4], box_shapes[2])
		edges[(4, 0)] = self.connect_arrow_between(box_shapes[4], box_shapes[0])

		edges[(5, 1)] = self.connect_arrow_between(box_shapes[5], box_shapes[1])
		edges[(5, 4)] = self.connect_arrow_between(box_shapes[5], box_shapes[4])
		edges[(5, 2)] = self.connect_arrow_between(box_shapes[5], box_shapes[2])
		edges[(5, 0)] = self.connect_arrow_between(box_shapes[5], box_shapes[0])

		self.wait(7)

		self.play(
			box_shapes[5][1].set_color, GREEN_SCREEN
		)

		self.play(
			edges[(5, 1)].set_color, GREEN_SCREEN,
		)

		self.play(
			box_shapes[1][1].set_color, GREEN_SCREEN
		)

		self.play(
			edges[(1, 0)].set_color, GREEN_SCREEN,
		)

		self.play(
			box_shapes[0][1].set_color, GREEN_SCREEN
		)

		stack.move_to(LEFT * 4 + UP * 1)

		self.play(
			TransformFromCopy(box_shapes[0], stack[0]),
			TransformFromCopy(box_shapes[1], stack[1]),	
			TransformFromCopy(box_shapes[5], stack[2]),
			run_time=2
		)

		self.wait(8)

		step_2 = TextMobject("2. Find an appropriate subproblem")
		step_2.scale(0.8)
		step_2.move_to(step_1.get_center())

		self.play(
			FadeOut(stack),
			FadeOut(problem[2]),
			ReplacementTransform(step_1, step_2),
			edges[(5, 1)].set_color, GRAY,
			edges[(1, 0)].set_color, GRAY,
			box_shapes[0][1].set_color, WHITE,
			box_shapes[1][1].set_color, WHITE,
			box_shapes[5][1].set_color, WHITE
		)

		self.wait(12)

		self.show_subproblem(box_shapes, edges, step_2, problem)

	def show_subproblem(self, box_shapes, edges, step_2, problem):
		self.play(
			box_shapes[1][1].set_color, GREEN_SCREEN,
			box_shapes[5][1].set_color, GREEN_SCREEN,
			edges[(5, 1)].set_color, GREEN_SCREEN,
		)

		self.wait(2)

		self.play(
			box_shapes[1][1].set_color, WHITE,
			box_shapes[4][1].set_color, GREEN_SCREEN,
			edges[(5, 1)].set_color, GRAY,
			edges[(5, 4)].set_color, GREEN_SCREEN,
		)

		self.wait(2)

		self.play(
			box_shapes[1][1].set_color, WHITE,
			box_shapes[4][1].set_color, WHITE,
			box_shapes[5][1].set_color, WHITE,
			box_shapes[3][1].set_color, GREEN_SCREEN,
			box_shapes[2][1].set_color, GREEN_SCREEN,
			edges[(3, 2)].set_color, GREEN_SCREEN,
			edges[(5, 4)].set_color, GRAY,
		)

		self.wait(2)

		self.play(
			box_shapes[1][1].set_color, WHITE,
			box_shapes[4][1].set_color, WHITE,
			box_shapes[3][1].set_color, WHITE,
			box_shapes[1][1].set_color, GREEN_SCREEN,
			box_shapes[5][1].set_color, GREEN_SCREEN,
			edges[(3, 2)].set_color, GRAY,
			edges[(5, 1)].set_color, GREEN_SCREEN,
			edges[(1, 2)].set_color, GREEN_SCREEN,
		)

		self.wait(2)

		self.play(
			box_shapes[0][1].set_color, GREEN_SCREEN,
			box_shapes[1][1].set_color, GREEN_SCREEN,
			box_shapes[2][1].set_color, WHITE,
			box_shapes[3][1].set_color, WHITE,
			box_shapes[4][1].set_color, WHITE,
			box_shapes[5][1].set_color, GREEN_SCREEN,
			edges[(1, 0)].set_color, GREEN_SCREEN,
			edges[(5, 1)].set_color, GREEN_SCREEN,
			edges[(1, 2)].set_color, GRAY,
		)

		self.wait(2)

		self.play(
			box_shapes[0][1].set_color, WHITE,
			box_shapes[1][1].set_color, WHITE,
			box_shapes[2][1].set_color, WHITE,
			box_shapes[3][1].set_color, WHITE,
			box_shapes[4][1].set_color, WHITE,
			box_shapes[5][1].set_color, WHITE,
			edges[(3, 2)].set_color, GRAY,
			edges[(1, 0)].set_color, GRAY,
			edges[(5, 1)].set_color, GRAY,
			edges[(1, 2)].set_color, GRAY,
		)


		subproblem = TextMobject(
			"Subproblem: height[$(L_i, W_i, H_i)$]" + "\\\\", 
			r"Largest height of stack with box $(L_i, W_i, H_i)$ at the bottom"
		)
		subproblem.scale(0.7)


		subproblem.move_to(UP * 3.3)

		self.play(
			Write(subproblem[0])
		)
		self.play(
			Write(subproblem[1])
		)

		self.wait(13)

		ex1 = TextMobject(r"height[$(2, 3, 2)$] = 4")
		ex1.scale(0.7)
		ex1[0][7:14].set_color(ORANGE)
		ex1.move_to(LEFT * 4 + UP * 1.8)

		ex2 = TextMobject(r"height[$(3, 6, 2)$] = 6")
		ex2.scale(0.7)
		ex2[0][7:14].set_color(YELLOW)
		ex2.next_to(ex1, DOWN)

		ex3 = TextMobject(r"height[$(4, 5, 3)$] = 7")
		ex3.scale(0.7)
		ex3[0][7:14].set_color(RED)
		ex3.next_to(ex2, DOWN)

		self.play(
			Write(ex1[0][:-2])
		)

		self.play(
			box_shapes[5][1].set_color, GREEN_SCREEN,
			edges[(5, 1)].set_color, GREEN_SCREEN,
			box_shapes[1][1].set_color, GREEN_SCREEN
		)

		self.play(
			Write(ex1[0][-2:])
		)

		self.wait(3)

		self.play(
			box_shapes[5][1].set_color, WHITE,
			edges[(5, 1)].set_color, GRAY,
			box_shapes[1][1].set_color, WHITE
		)

		self.wait()

		self.play(
			Write(ex2[0][:-2])
		)

		self.play(
			box_shapes[2][1].set_color, GREEN_SCREEN,
			edges[(3, 2)].set_color, GREEN_SCREEN,
			box_shapes[3][1].set_color, GREEN_SCREEN
		)

		self.play(
			Write(ex2[0][-2:])
		)

		self.wait(9)

		self.play(
			box_shapes[2][1].set_color, WHITE,
			edges[(3, 2)].set_color, GRAY,
			box_shapes[3][1].set_color, WHITE
		)

		self.wait()

		self.play(
			Write(ex3[0][:-2])
		)

		self.play(
			box_shapes[0][1].set_color, GREEN_SCREEN,
			edges[(1, 0)].set_color, GREEN_SCREEN,
			box_shapes[1][1].set_color, GREEN_SCREEN,
			edges[(5, 1)].set_color, GREEN_SCREEN,
			box_shapes[5][1].set_color, GREEN_SCREEN
		)

		self.play(
			Write(ex3[0][-2:])
		)

		self.wait(10)

		self.play(
			box_shapes[0][1].set_color, WHITE,
			edges[(1, 0)].set_color, GRAY,
			box_shapes[1][1].set_color, WHITE,
			edges[(5, 1)].set_color, GRAY,
			box_shapes[5][1].set_color, WHITE
		)

		self.wait()

		box_nodes = VGroup(*box_shapes)

		box_edges = VGroup(*[edges[key] for key in edges])

		box_graph = VGroup(box_nodes, box_edges)

		step_3 = TextMobject("3. Find relationships among subproblems")
		step_3.scale(0.8)
		step_3.move_to(step_2.get_center())

		self.play(
			box_graph.scale, 0.8,
			box_graph.shift, UP * 1,
			FadeOut(ex1),
			FadeOut(ex2),
			FadeOut(ex3),
			ReplacementTransform(step_2, step_3)
		)

		self.wait(8)

		self.find_relationships(box_shapes, edges, box_graph, step_3, subproblem)

	def find_relationships(self, box_shapes, edges, box_graph, step_3, subproblem):
		subproblems_needed = TextMobject("What subproblems are needed to solve height$[(4, 5, 3)]$?")
		subproblems_needed.scale(0.8)
		subproblems_needed[0][-9:-2].set_color(RED)
		subproblems_needed.next_to(box_graph, DOWN)
		self.play(
			Write(subproblems_needed)
		)

		dest_rectangle = SurroundingRectangle(box_shapes[0], buff=SMALL_BUFF)
		dest_rectangle.set_color(RED)
		self.play(
			ShowCreation(dest_rectangle)
		)

		self.wait()

		self.play( 
			edges[(1, 0)].set_color, GREEN_SCREEN,
			edges[(4, 0)].set_color, GREEN_SCREEN,
			edges[(5, 0)].set_color, GREEN_SCREEN,
		)

		self.wait(5)

		subproblem_rects = [SurroundingRectangle(box_shapes[i], buff=SMALL_BUFF) for i in [1, 4, 5]]

		necessary_subproblems = TextMobject(r"height$\left[ (2, 3, 2) \right] = 4 \quad$", r"height$\left[ (2, 4, 1) \right] = 3 \quad$", r"height$\left[ (1, 2, 2) \right] = 2$")
		necessary_subproblems.scale(0.8)
		necessary_subproblems[0][-9:-3].set_color(ORANGE)
		necessary_subproblems[1][-9:-3].set_color(BLUE)
		necessary_subproblems[2][-9:-3].set_color(VIOLET)
		necessary_subproblems.next_to(subproblems_needed, DOWN)

		self.play(
			ShowCreation(subproblem_rects[0])
		)

		self.play(
			Write(necessary_subproblems[0])
		)

		self.wait()

		self.play(
			ShowCreation(subproblem_rects[1])
		)

		self.play(
			Write(necessary_subproblems[1])
		)

		self.wait(3)

		self.play(
			ShowCreation(subproblem_rects[2])
		)

		self.play(
			Write(necessary_subproblems[2])
		)

		self.wait(10)

		using_subproblems = TextMobject(r"How do we use these subproblems to solve height[$(4, 5, 3)$]?")
		using_subproblems.scale(0.8)
		using_subproblems[0][-9:-2].set_color(RED)
		using_subproblems.move_to(subproblems_needed.get_center())
		self.play(
			ReplacementTransform(subproblems_needed, using_subproblems)
		)

		self.wait(5)

		self.play(
			FadeOut(necessary_subproblems)
		)

		self.wait(2)

		answer = TextMobject(r"height$[(4, 5, 3)] = 3 + \text{max} \{ \text{height} [(2, 3, 2)], \text{height} [(2, 4, 1)], \text{height} [(1, 2, 2)] \}$", r" $= 7$")
		answer.scale(0.7)
		answer[0][7:14].set_color(RED)
		answer[0][16].set_color(RED)
		answer[0][29:36].set_color(ORANGE)
		answer[0][45:52].set_color(BLUE)
		answer[0][61:68].set_color(VIOLET)

		answer.move_to(necessary_subproblems.get_center())

		self.play(
			Write(answer[0][:15])
		)

		self.wait()

		self.play(
			Write(answer[0][15:]),
			run_time=2
		)

		self.wait()

		self.play(
			Write(answer[1])
		)

		self.wait(10)

		step_4 = TextMobject("4. Generalize the relationship")
		step_4.scale(0.8)
		step_4.move_to(step_3.get_center())

		self.play(
			FadeOut(box_graph),
			FadeOut(using_subproblems),
			FadeOut(answer),
			FadeOut(subproblem),
			FadeOut(dest_rectangle),
			*[FadeOut(sr) for sr in subproblem_rects],
			ReplacementTransform(step_3, step_4),
			run_time=2
		)


	def connect_arrow_between(self, box1, box2, animate=True):
		start_point = box1.get_right()
		end_point = box2.get_left()
		arrow = Arrow(start_point, end_point, tip_length=0.2)
		arrow.set_color(GRAY)
		if animate:
			self.play(
				ShowCreation(arrow)
			)
		return arrow


	def construct_box(self, l=3, w=2, h=1, color=BLUE, label=True, label_direction=RIGHT):
		box = Prism(dimensions=[l, w, h])
		box.set_color(color)
		box.set_stroke(WHITE, 2)
		box.pose_at_angle()
		
		if label:
			label_text = TextMobject("({0}, {1}, {2})".format(*self.convert_to_input_dimensions((l, w, h))))
			label_text.scale(1.8)
			label_text.next_to(box, label_direction)
			box = VGroup(box, label_text)

		return box

	# def make_stack_from_indices(self, boxes, indices):
	# 	stack = []
	# 	for i in indices:
	# 		box = boxes[i].copy()
	# 		stack.insert(0, box)
		
	# 	for i in range(len(stack) - 1):
	# 		self.put_box_on_top(stack[i + 1], stack[i])

	# 	return VGroup(*stack)



	def make_stack_of_boxes(self, boxes, colors):
		stack = []
		for i, dimension in enumerate(boxes):
			l, w, h = dimension
			box = self.construct_box(l=l, w=w, h=h, color=colors[i])
			stack.append(box)

		for i in range(len(stack) - 1):
			self.put_box_on_top(stack[i + 1], stack[i])

		return stack

	def put_box_on_top(self, top, bottom):
		display_length = bottom[0].dimensions[2]
		top[0].next_to(bottom[0], UP, buff=0)
		top[0].shift(DOWN * 0.25 * display_length)
		top[1].next_to(top[0], RIGHT)
		# display_height = bottom[0].dimensions[1]
		# top.shift(UP * 0.7 * display_height)

class LIS(Scene):
	def construct(self):
		title = TextMobject("Longest Increasing Subsequence (LIS)")
		title.scale(1.2)
		title.shift(UP * 3.5)

		h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
		h_line.next_to(title, DOWN)
		self.play(
			Write(title),
			ShowCreation(h_line)
		)

		self.wait(4)

		problem = self.problem_statement(h_line)

		self.show_examples(problem, h_line)

		self.wait()

		# array = [5, 2, 8, 6, 3, 6, 9, 7]
		# # array = [8, 2, 9, 4, 5, 7, 3]
		# graph, edge_dict = self.construct_graph(array)
		# nodes, edges = self.make_graph_mobject(graph, edge_dict)
		# entire_graph = VGroup(nodes, edges)
		# entire_graph.move_to(ORIGIN)
		# entire_graph.scale(0.8)
		# entire_graph.shift(DOWN * 2)
		# self.play(
		# 	FadeIn(entire_graph)
		# )

	def problem_statement(self, h_line):
		problem = TextMobject(
			r"For a sequence $a_1, a_2, \ldots , a_n$, find the length of the" + "\\\\",
			r"longest increasing subsequence $a_{i_1}, a_{i_2}, \ldots , a_{i_k}$" + "\\\\",
			r"Constraints: $i_1 < i_2 < \cdots < i_k$; $a_{i_1} < a_{i_2} < \cdots < a_{i_k}$"
		)
		problem.scale(0.8)
		problem.next_to(h_line, DOWN)
		
		self.play(
			Write(problem[0])
		)
		self.wait()

		self.play(
			Write(problem[1])
		)

		self.wait(3)

		self.play(
			Write(problem[2])
		)
		return problem

	def show_examples(self, problem, h_line):
		ex1 = TextMobject(r"LIS($\left[ 3 \quad 1 \quad 8 \quad 2 \quad 5 \right]$)", r"$\rightarrow 3$")
		ex1[0][:3].set_color(MONOKAI_BLUE)
		ex1[0][5:10].set_color(MONOKAI_PURPLE)
		# ex1[7].set_color(MONOKAI_PURPLE)
		# ex1[9].set_color(MONOKAI_PURPLE)
		# ex1[11].set_color(MONOKAI_PURPLE)
		# ex1[13].set_color(MONOKAI_PURPLE)
		ex1.scale(0.8)
		ex1.next_to(problem, DOWN)

		self.wait(7)
		self.play(
			Write(ex1[0])
		)

		self.wait(2)
		
		arrow_1 = CurvedArrow(
			ex1[0][6].get_center() + DOWN * 0.2 + RIGHT * SMALL_BUFF, 
			ex1[0][8].get_center() + DOWN * 0.2 + LEFT * SMALL_BUFF, 
			tip_length=0.1
		)
		arrow_2 = CurvedArrow(
			ex1[0][8].get_center() + DOWN * 0.2 + RIGHT * SMALL_BUFF, 
			ex1[0][9].get_center() + DOWN * 0.2 + LEFT * SMALL_BUFF, 
			tip_length=0.1
		)
		arrow_1.set_color(GREEN_SCREEN)
		arrow_2.set_color(GREEN_SCREEN)
		
		ex1[0][6].set_color(GREEN_SCREEN)
		self.play(
			ShowCreation(arrow_1)
		)

		ex1[0][8].set_color(GREEN_SCREEN)
		self.play(
			ShowCreation(arrow_2)
		)

		ex1[0][9].set_color(GREEN_SCREEN)

		self.wait(2)

		self.play(
			Write(ex1[1])
		)
		self.wait(6)

		ex2 = TextMobject(r"LIS($\left[ 5 \quad 2 \quad 8 \quad 6 \quad 3 \quad 6 \quad 9 \quad 5 \right]$)", r"$\rightarrow 4$")
		ex2.scale(0.8)
		ex2.move_to(ex1.get_center())
		ex2.shift(DOWN * 1)
		ex2[0][:3].set_color(MONOKAI_BLUE)
		ex2[0][5:13].set_color(MONOKAI_PURPLE)

		self.play(
			Write(ex2[0])
		)

		arrow_3 = CurvedArrow(
			ex2[0][6].get_center() + DOWN * 0.2 + RIGHT * SMALL_BUFF, 
			ex2[0][9].get_center() + DOWN * 0.2 + LEFT * SMALL_BUFF, 
			tip_length=0.1
		)
		arrow_4 = CurvedArrow(
			ex2[0][9].get_center() + DOWN * 0.2 + RIGHT * SMALL_BUFF, 
			ex2[0][10].get_center() + DOWN * 0.2 + LEFT * SMALL_BUFF, 
			tip_length=0.1
		)

		arrow_5 = CurvedArrow(
			ex2[0][10].get_center() + DOWN * 0.2 + RIGHT * SMALL_BUFF, 
			ex2[0][11].get_center() + DOWN * 0.2 + LEFT * SMALL_BUFF, 
			tip_length=0.1
		)

		self.wait(4)

		arrow_3.set_color(GREEN_SCREEN)
		arrow_4.set_color(GREEN_SCREEN)
		arrow_5.set_color(GREEN_SCREEN)

		ex2[0][6].set_color(GREEN_SCREEN)
		self.play(
			ShowCreation(arrow_3)
		)

		ex2[0][9].set_color(GREEN_SCREEN)
		self.play(
			ShowCreation(arrow_4)
		)

		ex2[0][10].set_color(GREEN_SCREEN)
		self.play(
			ShowCreation(arrow_5)
		)

		ex2[0][11].set_color(GREEN_SCREEN)

		self.play(
			Write(ex2[1])
		)

		self.wait(5)

		focus = TextMobject("We will focus on the length of the LIS")
		focus[0][16:22].set_color(YELLOW)
		focus.scale(0.8)
		focus.move_to(DOWN * 1.5)

		self.play(
			Write(focus),
			problem[0][-11:-5].set_color, YELLOW
		)

		self.wait(6)

		step_1 = TextMobject("1. Visualize Examples")
		step_1.scale(0.8)
		step_1.move_to(DOWN * 3.5)
		self.play(
			Write(step_1)
		)

		self.wait(10)

		self.play(
			Indicate(problem[2])
		)

		self.wait()

		self.play(
			problem[2].set_color, YELLOW
		)

		self.wait()

		self.wait(5)

		to_fade = [
		ex2,
		arrow_3,
		arrow_4,
		arrow_5,
		problem, 
		focus,
		]

		ex1_with_arrows = VGroup(ex1, arrow_1, arrow_2)
		self.play(
			ex1_with_arrows.shift, UP * 1.5,
			*[FadeOut(obj) for obj in to_fade],
			run_time=2
		)

		self.wait(5)

		graph, edge_dict = self.construct_graph([3, 1, 8, 2, 5])
		nodes, edges = self.make_graph_mobject(graph, edge_dict)
		entire_graph = VGroup(nodes, edges)
		entire_graph.move_to(ORIGIN)

		transforms = []
		for i in range(5, 10):
			transform = TransformFromCopy(ex1[0][i], nodes[i - 5])
			transforms.append(transform)

		self.play(
			*transforms,
			run_time=2
		)

		self.wait(3)

		self.play(
			ShowCreation(edges),
			rate_func=linear,
			run_time=6
		)

		self.wait(5)
		highlight_objs = []
		
		circle = self.highlight_node(graph, 1)
		self.play(
			edge_dict[(1, 3)].set_color, GREEN_SCREEN
		)
		highlight_objs.append(circle)

		circle = self.highlight_node(graph, 3)
		self.play(
			edge_dict[(3, 4)].set_color, GREEN_SCREEN
		)
		highlight_objs.append(circle)

		circle = self.highlight_node(graph, 4)
		highlight_objs.append(circle)

		self.wait(7)

		observation = TextMobject("LIS = Longest Path in DAG + 1")
		observation.scale(0.8)
		observation.next_to(step_1, UP)
		self.play(
			Write(observation)
		)

		self.wait(15)

		step_2 = TextMobject("2. Find an appropriate subproblem")
		step_2.scale(0.8)
		step_2.move_to(step_1.get_center())
		self.play(
			ReplacementTransform(step_1, step_2)
		)

		self.wait(5)
		
		self.play(
			FadeOut(observation)
		)

		self.wait(9)

		subsets = TextMobject("All increasing subsequences are subsets of original sequence.")
		subsets.scale(0.8)

		subsets.next_to(entire_graph, DOWN)

		self.play(
			Write(subsets)
		)

		self.wait(5)

		start_end = TextMobject("All increasing subsequences have a start and end.")
		start_end.scale(0.8)
		start_end.next_to(subsets, DOWN)
		start_end[0][-4:-1].set_color(ORANGE)
		start_end[0][-12:-7].set_color(RED)
		self.play(
			Write(start_end)
		)
		self.wait()

		start_rect = SurroundingRectangle(highlight_objs[0], buff=SMALL_BUFF, color=RED)
		end_rect = SurroundingRectangle(highlight_objs[2], buff=SMALL_BUFF, color=ORANGE)
		self.play(
			ShowCreation(start_rect),
			ShowCreation(end_rect)
		)
		self.wait(9)

		self.play(
			FadeOut(start_rect),
			FadeOut(subsets),
			start_end.move_to, subsets.get_center()
		)

		self.wait()

		ending_focus = TextMobject("Let's focus on the end index of an increasing subsequence")
		ending_focus.scale(0.8)
		ending_focus.next_to(subsets, DOWN)
		ending_focus[0][15:18].set_color(ORANGE)

		self.play(
			FadeIn(ending_focus)
		)

		self.play(
			end_rect.move_to, nodes[3].get_center(),
			run_time=1
		)

		self.play(
			end_rect.move_to, nodes[2].get_center(),
			run_time=1
		)

		self.play(
			end_rect.move_to, nodes[1].get_center(),
			run_time=1
		)

		self.play(
			end_rect.move_to, nodes[0].get_center(),
			run_time=1
		)

		self.wait(3)

		self.play(
			FadeOut(start_end),
			FadeOut(ending_focus),
			FadeOut(end_rect)
		)

		subproblem = TextMobject(r"Subproblem: LIS$\left[ k \right] = \text{LIS ending at index} \> k$")
		subproblem.scale(0.8)
		subproblem.move_to(observation.get_center())
		self.play(
			Write(subproblem)
		)

		self.wait(5)

		example = TextMobject(r"LIS$\left[ 3 \right] =$", r"$\> 2$")
		example.scale(0.8)
		example.next_to(subproblem, UP)
		
		subproblem_graph, other = self.get_subproblem_object(nodes, edge_dict, 3)
		box = SurroundingRectangle(subproblem_graph, buff=SMALL_BUFF)
		self.play(
			Write(example[0])
		)

		self.play(
			ShowCreation(box)
		)

		self.wait(4)

		# self.play(
		# 	FadeOut(other),
		# 	FadeOut(highlight_objs[-1])
		# )

		self.play(
			Write(example[1])
		)

		self.wait(12)

		step_3 = TextMobject("3. Find relationships among subproblems")
		step_3.scale(0.8)
		step_3.move_to(step_2.get_center())
		self.play(
			edge_dict[(1, 3)].set_color, GRAY,
			edge_dict[(3, 4)].set_color, GRAY,
			ReplacementTransform(step_2, step_3),
			FadeOut(box),
			FadeOut(example),
			FadeOut(ex1),
			FadeOut(arrow_1),
			FadeOut(arrow_2),
			FadeOut(subproblem),
			*[FadeOut(obj) for obj in highlight_objs],
			run_time=2
		)

		self.wait()
		subproblem.move_to(ex1.get_center())
		self.play(
			Write(subproblem)
		)

		self.wait(7)

		surround_circle = self.highlight_node(graph, 4)

		self.wait()

		question = TextMobject(r"What subproblems are needed to solve LIS$\left[ 4 \right]$?")
		question[0][-7:-1].set_color(GREEN_SCREEN)
		question.scale(0.8)
		question.next_to(entire_graph, DOWN)
		self.play(
			Write(question),
			run_time=2
		)

		self.wait(6)

		
		self.play(
			edge_dict[(0, 4)].set_color, GREEN_SCREEN
		)
		self.wait()

		subproblem_graph, other = self.get_subproblem_object(nodes, edge_dict, 0)
		box = SurroundingRectangle(subproblem_graph, buff=SMALL_BUFF)

		necessary_subproblems = TextMobject(r"LIS$\left[ 0 \right] = 1 \quad$", r"LIS$\left[ 1 \right] = 1 \quad$", r"LIS$\left[ 3 \right] = 2$")
		necessary_subproblems.scale(0.8)
		necessary_subproblems.set_color(YELLOW)

		necessary_subproblems.next_to(question, DOWN)
		self.play(
			ShowCreation(box),
		)

		self.wait(5)
		self.play(
			Write(necessary_subproblems[0])
		)

		self.wait(2)

		subproblem_graph, other = self.get_subproblem_object(nodes, edge_dict, 1)

		new_box = SurroundingRectangle(subproblem_graph, buff=SMALL_BUFF)

		self.play(
			edge_dict[(0, 4)].set_color, GRAY,
			edge_dict[(1, 4)].set_color, GREEN_SCREEN,
			Transform(box, new_box),
		)

		self.wait(2)

		self.play(
			Write(necessary_subproblems[1])
		)

		self.wait(4)

		subproblem_graph, other = self.get_subproblem_object(nodes, edge_dict, 3)

		new_box = SurroundingRectangle(subproblem_graph, buff=SMALL_BUFF)
		self.play(
			edge_dict[(1, 4)].set_color, GRAY,
			edge_dict[(3, 4)].set_color, GREEN_SCREEN,
			Transform(box, new_box),
		)

		self.wait()

		self.play(
			Write(necessary_subproblems[2])
		)

		self.wait(2)

		# self.play(
		# 	edge_dict[(3, 4)].set_color, GRAY,
		# 	FadeOut(box),
		# )

		relationship = TextMobject(r"How do we use these subproblems to solve LIS$\left[ 4 \right]$?")
		relationship[-7:-1].set_color(GREEN_SCREEN)
		relationship.scale(0.8)
		relationship.move_to(question.get_center())

		self.play(
			ReplacementTransform(question, relationship)
		)

		self.wait()

		answer = TextMobject(r"LIS$\left[ 4 \right] = 1 + \text{max} \{ \text{LIS} \left[ 0 \right], \text{LIS} \left[ 1 \right], \text{LIS} \left[ 3 \right] \}$", r" $= 3$")
		answer.scale(0.8)
		answer.set_color(YELLOW)
		answer.move_to(necessary_subproblems.get_center())
		self.play(
			FadeOut(necessary_subproblems)
		)
		self.play(
			Write(answer[0])
		)

		self.wait()

		highlight_objs = []

		highlight_objs.append(surround_circle)
		
		circle = self.highlight_node(graph, 1)
		self.play(	
			edge_dict[(1, 3)].set_color, GREEN_SCREEN
		)
		highlight_objs.append(circle)

		circle = self.highlight_node(graph, 3)
		self.play(	
			box.set_color, GREEN_SCREEN,
			answer[0][-7:-1].set_color, GREEN_SCREEN
		)
		
		highlight_objs.append(circle)

		self.play(
			Write(answer[1])
		)

		self.wait(22)

		step_4 = TextMobject("4. Generalize the relationship")
		step_4.scale(0.8)
		step_4.move_to(step_3.get_center())
		self.play(
			*[FadeOut(obj) for obj in highlight_objs],
			FadeOut(entire_graph),
			FadeOut(relationship),
			FadeOut(answer),
			FadeOut(subproblem),
			FadeOut(box),
			ReplacementTransform(step_3, step_4)
		)

		self.wait()

		ex2 = TextMobject(r"$A = \left[ 5 \quad 2 \quad 8 \quad 6 \quad 3 \quad 6 \quad 9 \quad 5 \right]$")
		ex2.scale(0.8)
		ex2.move_to(ex1.get_center())
		# ex2.shift(DOWN * 1)
		ex2[0][2:11].set_color(MONOKAI_PURPLE)

		arrow_3 = CurvedArrow(
			ex2[0][4].get_center() + DOWN * 0.2 + RIGHT * SMALL_BUFF, 
			ex2[0][7].get_center() + DOWN * 0.2 + LEFT * SMALL_BUFF, 
			tip_length=0.1
		)
		arrow_4 = CurvedArrow(
			ex2[0][7].get_center() + DOWN * 0.2 + RIGHT * SMALL_BUFF, 
			ex2[0][8].get_center() + DOWN * 0.2 + LEFT * SMALL_BUFF, 
			tip_length=0.1
		)

		arrow_5 = CurvedArrow(
			ex2[0][8].get_center() + DOWN * 0.2 + RIGHT * SMALL_BUFF, 
			ex2[0][9].get_center() + DOWN * 0.2 + LEFT * SMALL_BUFF, 
			tip_length=0.1
		)

		arrow_3.set_color(GREEN_SCREEN)
		arrow_4.set_color(GREEN_SCREEN)
		arrow_5.set_color(GREEN_SCREEN)

		ex2[0][4].set_color(GREEN_SCREEN)

		ex2[0][7].set_color(GREEN_SCREEN)

		ex2[0][8].set_color(GREEN_SCREEN)

		ex2[0][9].set_color(GREEN_SCREEN)
		self.play(
			FadeIn(ex2[0]),
			ShowCreation(arrow_3),
			ShowCreation(arrow_4),
			ShowCreation(arrow_5)
		)

		self.wait(2)

		graph, edge_dict = self.construct_graph([5, 2, 8, 6, 3, 6, 9, 5], direction=UP, angle=-TAU/4)
		nodes, edges = self.make_graph_mobject(graph, edge_dict)
		entire_graph = VGroup(nodes, edges)
		scale_factor = 0.8
		entire_graph.scale(scale_factor)
		entire_graph.move_to(UP * 0.8)

		self.play(
			FadeIn(nodes)
		)

		self.wait()

		question = TextMobject(r"How do we solve subproblem LIS$\left[ 5 \right]$?")
		question[0][-7:-1].set_color(GREEN_SCREEN)
		question.scale(scale_factor)
		question.next_to(entire_graph, DOWN)
		question.shift(DOWN * 0.8)
		self.play(
			Write(question)
		)

		self.wait()

		surround_circle = self.highlight_node(graph, 5, scale_factor=scale_factor)
		self.wait(2)

		answer = TextMobject(r"LIS$\left[ 5 \right] = 1 + \text{max}\{ \text{LIS}[k] \mid k < 5, A[k] < A[5] \}$")
		answer[0][:6].set_color(GREEN_SCREEN)
		answer.scale(scale_factor)
		answer.next_to(question, DOWN)
		self.play(
			Write(answer)
		)
		self.wait(8)

		simplification = TextMobject(r"$= 1 + \text{max} \{ \text{LIS} \left[ 0 \right] , \text{LIS} \left[ 1 \right] , \text{LIS} \left[ 4 \right] \}$")
		simplification.scale(scale_factor)
		simplification.next_to(answer, DOWN)
		simplification.shift(RIGHT * 0.2)
		self.play(
			Write(simplification[0][:7])
		)

		arrow = Arrow(nodes[0].get_center() + DOWN * 1.2, nodes[0].get_center() + DOWN * 0.2)
		arrow.set_color(GREEN_SCREEN)
		k_equal = TextMobject("k = ")
		val = Integer(0)
		k_val = VGroup(k_equal, val).arrange_submobjects(RIGHT, buff=SMALL_BUFF * 2 + DOWN * SMALL_BUFF)
		k_val.scale(0.8)
		k_val.next_to(arrow, DOWN, buff=0)

		tracker = VGroup(arrow, k_val)

		self.play(
			ShowCreation(tracker[0]),
			Write(tracker[1])
		)

		self.wait(2)

		box = SurroundingRectangle(nodes[:1], buff=SMALL_BUFF)

		self.play(
			ShowCreation(box),
			ShowCreation(edge_dict[(0, 5)])
		)
		simplification[0][7:-1].set_color(YELLOW)
		self.play(
			Write(simplification[0][7:13])
		)

		self.wait(3)

		shift_value = nodes[1].get_center() - nodes[0].get_center()

		self.play(
			tracker.shift, shift_value,
			val.shift, shift_value,
			val.increment_value,
			FadeOut(box),
			run_time=1
		)

		box = SurroundingRectangle(nodes[:2], buff=SMALL_BUFF)
		self.play(
			ShowCreation(box),
			ShowCreation(edge_dict[(1, 5)])
		)

		self.play(
			Write(simplification[0][13:20])
		)

		self.wait()

		self.play(
			tracker.shift, shift_value,
			tracker[0].set_color, RED,
			tracker[0].shift, shift_value,
			val.shift, shift_value,
			val.increment_value, 2,
			FadeOut(box),
			run_time=1
		)


		self.play(
			tracker.shift, shift_value,
			val.shift, shift_value,
			val.increment_value, 3,
			run_time=1
		)

		self.play(
			tracker.shift, shift_value,
			tracker[0].set_color, GREEN_SCREEN,
			tracker[0].shift, shift_value,
			val.shift, shift_value,
			val.increment_value, 4,
			run_time=1
		)

		box = SurroundingRectangle(nodes[:5], buff=SMALL_BUFF)
		self.play(
			ShowCreation(box),
			ShowCreation(edge_dict[(4, 5)])
		)

		self.play(
			Write(simplification[0][20:])
		)

		self.wait()

		self.play(
			FadeOut(box),
			FadeOut(edge_dict[(0, 5)]),
			FadeOut(edge_dict[(1, 5)]),
			FadeOut(edge_dict[(4, 5)]),
			FadeOut(surround_circle),
			FadeOut(tracker),
			FadeOut(simplification)
		)

		new_question = TextMobject(r"How do we solve subproblem LIS$\left[ n \right]$?")
		new_question[0][-7:-1].set_color(BLUE)
		new_question.scale(scale_factor)
		new_question.next_to(entire_graph, DOWN)
		new_question.shift(DOWN * 0.8)

		self.play(
			Transform(question, new_question)
		)

		self.wait(2)

		new_answer = TextMobject(r"LIS$\left[ n \right] = 1 + \text{max}\{ \text{LIS}[k] \mid k < n, A[k] < A[n] \}$")
		new_answer[0][:6].set_color(BLUE)
		new_answer.scale(scale_factor)
		new_answer.next_to(question, DOWN)

		self.play(
			Transform(answer, new_answer)
		)

		self.wait(15)

		step_5 = TextMobject("5. Implement by solving subproblems in order")
		step_5.scale(0.8)
		step_5.move_to(step_4.get_center())

		self.play(
			ReplacementTransform(step_4, step_5)
		)

		self.play(
			FadeOut(nodes),
			FadeOut(question),
			FadeOut(ex2),
			FadeOut(arrow_3),
			FadeOut(arrow_4),
			FadeOut(arrow_5),
			answer.move_to, problem.get_center() + UP * 0.5,
			run_time=2
		)

		graph, edge_dict = self.construct_graph([5, 2, 8, 6, 3, 6, 9, 5])
		nodes, edges = self.make_graph_mobject(graph, edge_dict)
		entire_graph = VGroup(nodes, edges)
		scale_factor = 0.55
		entire_graph.scale(scale_factor)
		entire_graph.move_to(DOWN * 2)

		self.play(
			ShowCreation(entire_graph),
			run_time=2
		)

		self.wait(15)

		arrow = Arrow(nodes[0].get_center() + LEFT * 0.5, nodes[-1].get_center() + RIGHT * 0.5)
		arrow.set_color(GREEN_SCREEN)
		arrow.next_to(entire_graph, UP, buff=SMALL_BUFF)

		self.play(
			ShowCreation(arrow)
		)

		self.wait(2)

		code = self.generate_code()
		code.next_to(answer, DOWN)
		self.play(
			Write(code[0])
		)

		self.wait(2)

		self.play(
			Write(code[1])
		)

		self.wait(6)

		self.play(
			Write(code[2])
		)

		self.wait(5)

		self.play(
			Write(code[3]),
			FadeOut(arrow),
		)

		self.wait()

		self.play(
			Write(code[4]),
		)

		self.wait()

		self.play(
			Write(code[5])
		)

		self.wait(16)

		self.play(
			FadeOut(code),
			FadeOut(entire_graph),
			FadeOut(step_5),
			FadeOut(answer)
		)

		self.wait()

		reminder = TextMobject("Note: this gives us the length of LIS")
		reminder.scale(0.8)
		reminder.next_to(h_line, DOWN)

		new_question = TextMobject("How do we actually get the sequence?")
		new_question.scale(0.8)
		new_question.move_to(reminder.get_center())

		self.play(
			Write(reminder)
		)

		

		graph, edge_dict = self.construct_graph([3, 1, 8, 2, 5])
		nodes, edges = self.make_graph_mobject(graph, edge_dict)
		entire_graph = VGroup(nodes, edges)
		entire_graph.move_to(ORIGIN)
		
		circle = self.highlight_node(graph, 1, animate=False)
		edge_dict[(1, 3)].set_color(GREEN_SCREEN)
		highlight_objs.append(circle)

		circle = self.highlight_node(graph, 3, animate=False)
		edge_dict[(3, 4)].set_color(GREEN_SCREEN)
		highlight_objs.append(circle)

		circle = self.highlight_node(graph, 4, animate=False)
		highlight_objs.append(circle)

		self.play(
			FadeIn(entire_graph),
			*[FadeIn(c) for c in highlight_objs]
		)

		self.wait(5)

		self.play(
			ReplacementTransform(reminder, new_question)
		)

		self.wait(5)

		sequence_answer = TextMobject("Keep track of previous indices!")
		sequence_answer.scale(0.8)
		sequence_answer.next_to(new_question, DOWN)

		self.play(
			Write(sequence_answer)
		)

		self.wait(5)

		i_text = TexMobject("i").scale(0.8)
		j_text = TexMobject("j").scale(0.8)

		i_text.next_to(nodes[4], UP)
		j_text.next_to(nodes[3], UP)

		


		prev_def = TextMobject(r"prev$[i] = j$")
		prev_def.scale(0.8)
		prev_def.next_to(entire_graph, DOWN)

		self.play(
			FadeIn(i_text),
			Write(prev_def[0][:-1])
		)

		self.wait(5)

		self.play(
			FadeIn(j_text),
			Write(prev_def[0][-1])
		)

		self.wait()

		prev_def_meaning = TextMobject("Previous index used to solve LIS$[i]$ is index $j$")
		prev_def_meaning.scale(0.8)
		prev_def_meaning.next_to(prev_def, DOWN)

		self.play(
			Write(prev_def_meaning)
		)

		self.wait(7)


		examples = []

		prev_4 = TextMobject(r"prev$[4] = 3$")
		prev_3 = TextMobject(r"prev$[3] = 1$")
		prev_2 = TextMobject(r"prev$[2] = 0$")
		prev_1 = TextMobject(r"prev$[1] = -1$")
		prev_0 = TextMobject(r"prev$[0] = -1$")

		examples.extend([prev_0, prev_1, prev_2, prev_3, prev_4])
		examples_group = VGroup(*examples).arrange_submobjects(RIGHT, buff=SMALL_BUFF * 5)
		examples_group.scale(0.7)
		examples_group.next_to(prev_def_meaning, DOWN)

		circle_0 = self.highlight_node(graph, 0, animate=False)

		self.play(
			Write(examples[0]),
			*[FadeOut(obj) for obj in highlight_objs],
			FadeOut(i_text),
			FadeOut(j_text),
			edge_dict[(1, 3)].set_color, GRAY,
			edge_dict[(3, 4)].set_color, GRAY,
			FadeIn(circle_0),
		)

		self.wait(5)

		circle_1 = self.highlight_node(graph, 1, animate=False)

		self.play(
			Write(examples[1]),
			FadeOut(circle_0),
			FadeIn(circle_1)
		)

		self.wait(3)

		circle_2 = self.highlight_node(graph, 2, animate=False)

		self.play(
			Write(examples[2]),
			FadeOut(circle_1),
			FadeIn(circle_2),
			edge_dict[(0, 2)].set_color, GREEN_SCREEN,
		)

		self.wait(8)

		circle_3 = self.highlight_node(graph, 3, animate=False)

		self.play(
			Write(examples[3]),
			edge_dict[(0, 2)].set_color, GRAY,
			edge_dict[(1, 3)].set_color, GREEN_SCREEN,
			FadeOut(circle_2),
			FadeIn(circle_3),
		)

		self.wait(3)
		
		circle_4 = self.highlight_node(graph, 4, animate=False)

		self.play(
			Write(examples[4]),
			edge_dict[(3, 4)].set_color, GREEN_SCREEN,
			edge_dict[(1, 3)].set_color, GRAY,
			FadeOut(circle_3),
			FadeIn(circle_4),
		)

		self.wait(18)

	def generate_code(self):
		code_scale = 0.7
		
		code = []

		def_statement = TextMobject("def ", r"$\text{lis}(A):$")
		def_statement[0].set_color(MONOKAI_BLUE)
		def_statement[1][:3].set_color(MONOKAI_GREEN)
		def_statement[1][4].set_color(MONOKAI_ORANGE)
		def_statement.scale(code_scale)
		def_statement.to_edge(LEFT)

		line_1 = TextMobject(r"$L = [1]$ * len($A$)")
		line_1.scale(code_scale)
		line_1.next_to(def_statement, DOWN * 0.5)
		line_1.to_edge(LEFT * 2)
		line_1[0][5].shift(DOWN * SMALL_BUFF)
		line_1[0][1].set_color(MONOKAI_PINK)
		line_1[0][3].set_color(MONOKAI_PURPLE)
		line_1[0][5].set_color(MONOKAI_PINK)
		line_1[0][6:9].set_color(MONOKAI_BLUE)
		code.extend([def_statement, line_1])

		line_2 = TextMobject(r"for $i$ in range(1, len($L$)):")
		line_2.scale(code_scale)
		line_2.next_to(line_1, DOWN * 0.5)
		line_2.to_edge(LEFT * 2)
		line_2[0][:3].set_color(MONOKAI_PINK)
		line_2[0][4:6].set_color(MONOKAI_PINK)
		line_2[0][6:11].set_color(MONOKAI_BLUE)
		line_2[0][12].set_color(MONOKAI_PURPLE)
		line_2[0][14:17].set_color(MONOKAI_BLUE)
		
		code.append(line_2)

		line_3 = TextMobject(r"subproblems $= [L[k] \text{ for } k \text{ in range} (i) \text{ if } A[k] < A[i]]$")
		line_3.scale(code_scale)
		line_3.next_to(line_2, DOWN * 0.5)
		line_3.to_edge(LEFT * 3)
		line_3[0][11].set_color(MONOKAI_PINK)
		line_3[0][17:20].set_color(MONOKAI_PINK)
		line_3[0][21:23].set_color(MONOKAI_PINK)
		line_3[0][23:28].set_color(MONOKAI_BLUE)
		line_3[0][31:33].set_color(MONOKAI_PINK)
		line_3[0][37].set_color(MONOKAI_PINK)
		code.append(line_3)

		line_4 = TextMobject(r"$L[i] = 1 + \text{max}$(subproblems, default$=$0)")
		line_4.scale(code_scale)
		line_4.next_to(line_3, DOWN * 0.5)
		line_4.to_edge(LEFT * 3)
		line_4[0][4].set_color(MONOKAI_PINK)
		line_4[0][5].set_color(MONOKAI_PURPLE)
		line_4[0][6].set_color(MONOKAI_PINK)
		line_4[0][7:10].set_color(MONOKAI_BLUE)
		line_4[0][23:30].set_color(MONOKAI_ORANGE)
		line_4[0][30].set_color(MONOKAI_PINK)
		line_4[0][31].set_color(MONOKAI_PURPLE)
		code.append(line_4)

		line_5 = TextMobject(r"return max($L$, default$=$0)")
		line_5.scale(code_scale)
		line_5[0][:6].set_color(MONOKAI_PINK)
		line_5[0][6:9].set_color(MONOKAI_BLUE)
		line_5[0][12:19].set_color(MONOKAI_ORANGE)
		line_5[0][19].set_color(MONOKAI_PINK)
		line_5[0][20].set_color(MONOKAI_PURPLE)
		line_5.next_to(line_4, DOWN * 0.5)
		line_5.to_edge(LEFT * 2)
		code.append(line_5)

		return VGroup(*code)


	def get_subproblem_object(self, nodes, edge_dict, k):
		subproblem = VGroup()
		other = VGroup()
		for i in range(len(nodes)):
			if i <= k:
				subproblem.add(nodes[i])
			else:
				other.add(nodes[i])
		
		for key in edge_dict:
			if key[1] <= k:
				subproblem.add(edge_dict[key])
			else:
				other.add(edge_dict[key])

		return subproblem, other


	def construct_graph(self, sequence, direction=DOWN, angle=TAU/4):
		nodes = []
		edges = {}
		current = ORIGIN
		radius, scale = 0.4, 0.9
		for i in range(len(sequence)):
			node = GraphNode(sequence[i], position=current, radius=radius, scale=scale)
			nodes.append(node)
			current = current + RIGHT * 1.5
		
		for i in range(len(sequence)):
			for j in range(len(sequence)):
				if i < j and sequence[i] < sequence[j]:
					if i % 2 == 0:
						edges[(i, j)] = nodes[i].connect_curved_arrow(nodes[j], angle=-TAU/4)
					else:
						edges[(i, j)] = nodes[i].connect_curved_arrow(nodes[j], direction=direction, angle=angle)

		return nodes, edges

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
			# edge.set_stroke(width=2*scale_factor)
			edge.set_color(color=edge_color)
			edges.append(edge)
		return VGroup(*nodes), VGroup(*edges)

	def highlight_node(self, graph, index, color=GREEN_SCREEN, 
		start_angle=TAU/2, scale_factor=1, animate=True, run_time=1):
		node = graph[index]
		surround_circle = Circle(radius=node.circle.radius * scale_factor)
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

class BoxProblemPart2(BoxProblem):
	def construct(self):
		step_4 = TextMobject("4. Generalize the relationship")
		step_4.scale(0.8)
		step_4.move_to(DOWN * 3.5)
		self.add(step_4)

		title = TextMobject("Box Stacking")
		title.scale(1.2)
		title.shift(UP * 3.5)

		h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
		h_line.next_to(title, DOWN)
		problem = TextMobject(
			r"Given $n$ boxes  $[ (L_1, W_1, H_1), (L_2, W_2, H_2), \ldots , (L_n, W_n, H_n) ]$ where box $i$ has" + "\\\\",
			r"length $L_i$, width $W_i$, and height $H_i$, find the height of the tallest possible stack." + "\\\\",
			r"Box $(L_i, W_i, H_i)$ can be on top of box $(L_j, W_j, H_j) \text{ if } L_i < L_j, W_i < W_j$."
		)

		problem.scale(0.6)
		problem.next_to(h_line, DOWN)

		first_example = TextMobject(r"$[(2, 3, 3), (2, 2, 4), (4, 4, 2)]$")
		first_example.scale(0.6)
		first_example.next_to(problem, DOWN)

		box_dims = [
		(4, 5, 3),
		(2, 3, 2),
		(3, 6, 2),
		(1, 5, 4),
		(2, 4, 1),
		(1, 2, 2),
		]

		display_dims = [self.convert_to_display_dimesions(d) for d in box_dims]
		
		scale_factor = 0.3
		box_shapes = []
		box_colors = [RED, ORANGE, YELLOW, GREEN_SCREEN, BLUE, VIOLET]
		for i, display_dim in enumerate(display_dims):
			l, w, h = display_dim
			box = self.construct_box(l=l, w=w, h=h, color=box_colors[i])
			box.scale(scale_factor)
			box_shapes.append(box)

		new_locations = [
		RIGHT * 4 + DOWN * 2.5, 
		DOWN * 0.5,
		RIGHT * 4 + DOWN * 0.5,
		UP * 1.5,
		DOWN * 2.5,
		LEFT * 4 + DOWN * 1.5
		]
		
		for i in range(len(new_locations)):
			box_shapes[i].move_to(new_locations[i])

		edges = {}

		edges[(1, 2)] = self.connect_arrow_between(box_shapes[1], box_shapes[2], animate=False)
		edges[(1, 0)] = self.connect_arrow_between(box_shapes[1], box_shapes[0], animate=False)
		
		edges[(3, 2)] = self.connect_arrow_between(box_shapes[3], box_shapes[2], animate=False)

		edges[(4, 2)] = self.connect_arrow_between(box_shapes[4], box_shapes[2], animate=False)
		edges[(4, 0)] = self.connect_arrow_between(box_shapes[4], box_shapes[0], animate=False)

		edges[(5, 1)] = self.connect_arrow_between(box_shapes[5], box_shapes[1], animate=False)
		edges[(5, 4)] = self.connect_arrow_between(box_shapes[5], box_shapes[4], animate=False)
		edges[(5, 2)] = self.connect_arrow_between(box_shapes[5], box_shapes[2], animate=False)
		edges[(5, 0)] = self.connect_arrow_between(box_shapes[5], box_shapes[0], animate=False)

		box_nodes = VGroup(*box_shapes)

		box_edges = VGroup(*[edges[key] for key in edges])

		box_graph = VGroup(box_nodes, box_edges)

		self.generalize_relationship(step_4, box_graph)

	def generalize_relationship(self, step_4, box_graph):
		self.wait()

		box_dims = [
		(5, 2, 1), 
		(3, 4, 1), 
		(5, 3, 3), 
		(2, 5, 3), 
		(2, 1, 2), 
		(4, 1, 5), 
		(4, 5, 1), 
		(4, 1, 2), 
		(2, 2, 4), 
		]

		display_dims = [self.convert_to_display_dimesions(d) for d in box_dims]

		# second_example = TextMobject(r"$[(4, 5, 3), (2, 3, 2), (3, 6, 2), (1, 5, 4), (2, 4, 1), (1, 2, 2)]$")
		# second_example.scale(0.6)
		# second_example.move_to(UP * 3)
		# self.play(
		# 	FadeIn(second_example)
		# )

		scale_factor = 0.3
		box_shapes = []
		box_colors = [RED, ORANGE, YELLOW, GOLD, GREEN_SCREEN, GREEN, BLUE, PURPLE, VIOLET]
		for i, display_dim in enumerate(display_dims):
			l, w, h = display_dim
			box = self.construct_box(l=l, w=w, h=h, color=box_colors[i])
			box.scale(scale_factor)
			box_shapes.append(box)
		
		box_shapes[0].move_to(ORIGIN)
		box_shapes[0].shift(LEFT * 4.5 + DOWN * 0.5)

		box_shapes[1].next_to(box_shapes[0], DOWN)

		box_shapes[2].next_to(box_shapes[1], DOWN)

		box_shapes[3].next_to(box_shapes[0], RIGHT)

		box_shapes[4].next_to(box_shapes[3], DOWN)

		box_shapes[5].next_to(box_shapes[4], DOWN)

		box_shapes[6].next_to(box_shapes[3], RIGHT)

		box_shapes[7].next_to(box_shapes[6], DOWN)

		box_shapes[8].next_to(box_shapes[7], DOWN)

		box_nodes = VGroup(*box_shapes)
		box_nodes.move_to(UP * 1.5)

		self.play(
			FadeIn(box_nodes)
		)

		self.wait()

		question = TextMobject(r"How to solve height[$(L_i, W_i, H_i)$] in general?")
		question.scale(0.8)
		question.next_to(box_nodes, DOWN)
		

		self.play(
			Write(question)
		)

		self.wait(2)

		rect = SurroundingRectangle(box_shapes[6], buff=SMALL_BUFF, color=GREEN_SCREEN)
		self.play(
			question[0][17:27].set_color, BLUE,
			ShowCreation(rect)
		)

		self.wait(3)

		sub_step1 = TextMobject(r"1. Let $S$ be the set of all boxes that can be stacked above $(L_i, W_i, H_i)$")
		sub_step1.scale(0.7)
		sub_step1.next_to(question, DOWN)
		sub_step1[0][-10:].set_color(BLUE)

		sub_step2 = TextMobject(r"2. height$[(L_i, W_i, H_i)] = H_i + \text{max} \{ \text{height} [(L_j, W_j, H_j)] \mid (L_j, W_j, H_j) \in S \}$")
		sub_step2.scale(0.7)
		sub_step2.next_to(sub_step1, DOWN)
		sub_step2[0][9:19].set_color(BLUE)
		sub_step2[0][21:23].set_color(BLUE)
		sub_step2[0][24:].set_color(YELLOW)

		surround_rects = [SurroundingRectangle(box_shapes[i], buff=SMALL_BUFF) for i in [1, 4, 8]]

		self.play(
			Write(sub_step1)
		)

		self.wait()

		self.play(
			*[ShowCreation(r) for r in surround_rects]	
		)

		self.wait(3)

		surround_rects.append(rect)


		self.play(
			Write(sub_step2)
		)

		self.wait(4)

		step_5 = TextMobject("5. Implement by solving subproblems in order")
		step_5.scale(0.8)
		step_5.move_to(step_4.get_center())

		self.play(
			FadeOut(box_nodes),
			*[FadeOut(r) for r in surround_rects],
			FadeOut(question),
			sub_step1.shift, UP * 5,
			sub_step2.shift, UP * 5,
			ReplacementTransform(step_4, step_5),
			run_time=2
		)

		self.wait(2)

		order_note = TextMobject("What order do we solve these subproblems?")
		order_note.next_to(sub_step2, DOWN)
		order_note.scale(0.8)
		self.play(
			Write(order_note)
		)
		box_graph.scale(0.8)
		box_graph.move_to(DOWN * 0.5)
		self.play(
			FadeIn(box_graph)
		)

		self.wait(4)

		orange_rect = SurroundingRectangle(box_graph[0][1], buff=SMALL_BUFF, color=ORANGE)
		red_rect = SurroundingRectangle(box_graph[0][0], buff=SMALL_BUFF, color=RED)


		order_matters = TextMobject(r"Order matters (e.g height$[(2, 3, 2)]$ must be solved before height$[(4, 5, 3)]$)")
		order_matters.scale(0.8)
		order_matters[0][23:30].set_color(ORANGE)
		order_matters[0][-9:-2].set_color(RED)
		order_matters.move_to(order_note.get_center())

		self.play(
			ReplacementTransform(order_note, order_matters),
			ShowCreation(orange_rect),
			ShowCreation(red_rect)
		)

		self.wait(7)

		enforcing_ordering = TextMobject("How do we ensure correct ordering if boxes are given in random order?")
		enforcing_ordering.scale(0.8)
		enforcing_ordering.move_to(order_matters.get_center())
		self.play(
			ReplacementTransform(order_matters, enforcing_ordering)
		)

		self.wait(21)

		answer_ordering = TextMobject("We can sort the boxes by length or width first!")
		answer_ordering.scale(0.8)
		answer_ordering.move_to(enforcing_ordering.get_center())
		self.play(
			ReplacementTransform(enforcing_ordering, answer_ordering)
		)
		self.wait(4)

		self.play(
			FadeOut(box_graph),
			FadeOut(orange_rect),
			FadeOut(red_rect)
		)

		self.wait()

		code = self.generate_code()
		code.move_to(DOWN * 0.7)
		self.play(
			Write(code[0])
		)

		self.wait()

		self.play(
			Write(code[1])
		)

		self.wait(7)

		self.play(
			Write(code[2])
		)

		self.wait(11)

		self.play(
			Write(code[3])
		)

		self.wait(2)

		self.play(
			Write(code[4])
		)

		self.wait()

		self.play(
			Write(code[5])
		)

		self.wait(3)

		self.play(
			Write(code[8])
		)

		self.wait()

		self.play(
			Write(code[9])
		)

		self.wait(3)

		self.play(
			Write(code[6][0])
		)

		self.wait(2)

		self.play(
			Write(code[6][1])
		)

		self.wait()

		self.play(
			Write(code[6][2])
		)

		self.wait(6)

		self.play(
			Write(code[7])
		)

		self.wait(10)

		self.play(
			FadeOut(sub_step1),
			FadeOut(sub_step2),
			FadeOut(answer_ordering),
			FadeOut(step_5),
			code.scale, 0.8,
			code.move_to, UP * 2
		)
		self.wait()

		self.simulate_solution(code)

	def simulate_solution(self, code):
		box_dims = [
		(4, 5, 3),
		(2, 3, 2),
		(3, 6, 2),
		(1, 5, 4),
		(2, 4, 1),
		(1, 2, 2),
		]

		display_dims = [self.convert_to_display_dimesions(d) for d in box_dims]

		scale_factor = 0.3
		box_shapes = []
		box_colors = [RED, ORANGE, YELLOW, GREEN_SCREEN, BLUE, VIOLET]
		for i, display_dim in enumerate(display_dims):
			l, w, h = display_dim
			box = self.construct_box(l=l, w=w, h=h, color=box_colors[i], label_direction=UP)
			box.scale(scale_factor)
			box_shapes.append(box)
		
		box_shapes[0].move_to(LEFT * 5 + DOWN * 2)

		box_shapes[1].next_to(box_shapes[0], RIGHT)

		box_shapes[2].next_to(box_shapes[1], RIGHT)

		box_shapes[3].next_to(box_shapes[2], RIGHT)

		box_shapes[4].next_to(box_shapes[3], RIGHT)

		box_shapes[5].next_to(box_shapes[4], RIGHT)

		box_group = VGroup(*box_shapes)
		box_group.move_to(DOWN * 2)

		self.play(
			FadeIn(box_group)
		)


		self.wait()

		code_arrow = Arrow(ORIGIN, RIGHT * 1.2)
		code_arrow.set_color(GREEN_SCREEN)
		code_arrow.next_to(code[1], LEFT * 0.5)

		self.play(
			ShowCreation(code_arrow)
		)

		self.wait()

		sorted_box_dims = sorted(box_dims, key=lambda x: x[0])

		sorted_display_dims = [self.convert_to_display_dimesions(d) for d in sorted_box_dims]

		scale_factor = 0.3
		sorted_box_shapes = []
		box_colors = [GREEN_SCREEN, VIOLET, ORANGE, BLUE, YELLOW, RED]
		for i, display_dim in enumerate(sorted_display_dims):
			l, w, h = display_dim
			box = self.construct_box(l=l, w=w, h=h, color=box_colors[i], label_direction=UP)
			box.scale(scale_factor)
			sorted_box_shapes.append(box)
		
		sorted_box_shapes[0].move_to(LEFT * 5 + DOWN * 2)

		sorted_box_shapes[1].next_to(sorted_box_shapes[0], RIGHT * 1.2)

		sorted_box_shapes[2].next_to(sorted_box_shapes[1], RIGHT * 1.2)

		sorted_box_shapes[3].next_to(sorted_box_shapes[2], RIGHT * 1.2)

		sorted_box_shapes[4].next_to(sorted_box_shapes[3], RIGHT * 1.2)

		sorted_box_shapes[5].next_to(sorted_box_shapes[4], RIGHT * 1.2)

		sorted_box_group = VGroup(*sorted_box_shapes)
		sorted_box_group.move_to(DOWN * 1.5)

		map_index = [5, 2, 4, 0, 3, 1]

		self.play(
			*[ReplacementTransform(box_group[i], 
				sorted_box_group[map_index[i]]) for i in range(len(box_group))],
			run_time=3
		)

		self.wait()

		self.play(
			code_arrow.next_to, code[2], LEFT * 0.5
		)

		self.wait()

		height_text = TextMobject("heights:")
		height_text.scale(0.8)

		height_text.next_to(sorted_box_shapes[0], LEFT)

		heights = [Integer(h).scale(0.8) for _, _, h in sorted_box_dims]

		for i, h in enumerate(heights):
			h.next_to(sorted_box_group[i], UP)

		for i in range(1, len(heights)):
			heights[i].move_to(RIGHT * heights[i].get_center()[0] + UP * heights[0].get_center()[1])

		height_text.move_to(RIGHT * height_text.get_center()[0] + UP * heights[0].get_center()[1])

		self.play(
			*[FadeIn(h) for h in heights],
			 FadeIn(height_text)
		)

		self.wait(2)

		self.play(
			code_arrow.next_to, code[3], LEFT * 0.5
		)

		i_arrow = Arrow(DOWN * 3.2, DOWN * 2)
		i_arrow.next_to(sorted_box_shapes[1], DOWN)
		i_arrow.set_color(WHITE)
		i_equal = TextMobject("i = ")
		i_val = Integer(1)
		i_val = VGroup(i_equal, i_val).arrange_submobjects(RIGHT, buff=SMALL_BUFF * 2 + DOWN * SMALL_BUFF)
		i_val.scale(0.7)
		i_val.next_to(i_arrow, DOWN, buff=0)
		i_indicator = VGroup(i_arrow, i_val)

		self.play(
			ShowCreation(i_arrow),
			FadeIn(i_val)
		)

		self.wait()

		self.play(
			code_arrow.next_to, code[4], LEFT * 0.5
		)
		
		self.wait(3)
		
		self.play(
			code_arrow.next_to, code[5], LEFT * 0.5
		)

		j_arrow = Arrow(DOWN * 3.2, DOWN * 2)
		j_arrow.next_to(sorted_box_shapes[0], DOWN)
		j_arrow.set_color(YELLOW)
		j_equal = TextMobject("j = ")
		j_val = Integer(0)
		j_val = VGroup(j_equal, j_val).arrange_submobjects(RIGHT, buff=SMALL_BUFF * 2 + DOWN * SMALL_BUFF)
		j_val.scale(0.7)
		j_val.set_color(YELLOW)
		j_val.next_to(j_arrow, DOWN, buff=0)
		j_indicator = VGroup(j_arrow, j_val)


		self.play(
			ShowCreation(j_arrow),
			FadeIn(j_val)
		)

		self.wait(7)

		self.play(
			code_arrow.next_to, code[6], LEFT * 0.5
		)

		self.wait(6)

		self.play(
			code_arrow.next_to, code[3], LEFT * 0.5
		)

		self.wait()

		self.move_tracker(i_indicator, sorted_box_shapes, 2, start=1)

		self.play(
			code_arrow.next_to, code[4], LEFT * 0.5
		)

		self.wait()

		self.play(
			code_arrow.next_to, code[5], LEFT * 0.5
		)

		surround_rects = []
		rect = SurroundingRectangle(sorted_box_shapes[1], buff=SMALL_BUFF)
		surround_rects.append(rect)
		S = TextMobject("S").scale(0.8)
		S.next_to(sorted_box_shapes[0], LEFT)
		S.shift(LEFT * 0.5)
		surround_S = SurroundingRectangle(S, buff=SMALL_BUFF)
		
		self.wait()

		self.play(
			FadeIn(S),
			FadeIn(surround_S),
		)

		self.wait()

		self.move_tracker(j_indicator, sorted_box_shapes, 1)
	
		self.play(
			ShowCreation(rect)
		)

		self.wait(3)

		self.play(
			code_arrow.next_to, code[6], LEFT * 0.5
		)

		self.wait(11)

		self.play(
			heights[2].increment_value, 2
		)

		self.play(
			code_arrow.next_to, code[3], LEFT * 0.5
		)

		self.wait()

		self.move_tracker(i_indicator, sorted_box_shapes, 3, start=1)

		self.play(
			*[FadeOut(r) for r in surround_rects],
			code_arrow.next_to, code[4], LEFT * 0.5,
		)

		self.wait(3)

		self.play(
			code_arrow.next_to, code[5], LEFT * 0.5
		)

		self.move_tracker(j_indicator, sorted_box_shapes, 0)

		self.wait()

		self.move_tracker(j_indicator, sorted_box_shapes, 1)

		self.play(
			ShowCreation(surround_rects[0])
		)

		self.move_tracker(j_indicator, sorted_box_shapes, 2)

		self.play(
			code_arrow.next_to, code[6], LEFT * 0.5
		)

		self.wait()

		self.play(
			heights[3].increment_value, 2
		)

		self.play(
			code_arrow.next_to, code[3], LEFT * 0.5
		)

		self.move_tracker(i_indicator, sorted_box_shapes, 4, start=1)

		self.play(
			*[FadeOut(r) for r in surround_rects],
			code_arrow.next_to, code[4], LEFT * 0.5,
		)

		self.wait(2)

		self.play(
			code_arrow.next_to, code[5], LEFT * 0.5
		)

		self.wait()

		self.move_tracker(j_indicator, sorted_box_shapes, 0)

		new_rect = SurroundingRectangle(sorted_box_shapes[0], buff=SMALL_BUFF)
		self.play(
			ShowCreation(new_rect)
		)
		surround_rects.insert(0, new_rect)

		self.move_tracker(j_indicator, sorted_box_shapes, 1)

		self.play(
			ShowCreation(surround_rects[1])
		)

		self.move_tracker(j_indicator, sorted_box_shapes, 2)

		new_rect = SurroundingRectangle(sorted_box_shapes[2], buff=SMALL_BUFF)
		self.play(
			ShowCreation(new_rect)
		)
		surround_rects.append(new_rect)

		self.move_tracker(j_indicator, sorted_box_shapes, 3)

		new_rect = SurroundingRectangle(sorted_box_shapes[3], buff=SMALL_BUFF)
		self.play(
			ShowCreation(new_rect)
		)
		surround_rects.append(new_rect)

		self.play(
			code_arrow.next_to, code[6], LEFT * 0.5
		)

		self.wait(5)

		self.play(
			heights[4].increment_value, 4
		)

		self.wait()

		self.play(
			code_arrow.next_to, code[3], LEFT * 0.5
		)

		self.wait()

		self.move_tracker(i_indicator, sorted_box_shapes, 5, start=1)

		self.play(
			*[FadeOut(r) for r in surround_rects],
			code_arrow.next_to, code[4], LEFT * 0.5,
		)

		self.play(
			code_arrow.next_to, code[5], LEFT * 0.5
		)

		self.move_tracker(j_indicator, sorted_box_shapes, 0)


		self.move_tracker(j_indicator, sorted_box_shapes, 1)

		self.play(
			ShowCreation(surround_rects[1])
		)

		self.move_tracker(j_indicator, sorted_box_shapes, 2)

		self.play(
			ShowCreation(surround_rects[2])
		)

		self.move_tracker(j_indicator, sorted_box_shapes, 3)

		self.play(
			ShowCreation(surround_rects[3])
		)

		self.move_tracker(j_indicator, sorted_box_shapes, 4)

		self.play(
			code_arrow.next_to, code[6], LEFT * 0.5
		)

		self.wait(9)

		self.play(
			heights[5].increment_value, 4
		)

		self.wait(5)

		self.play(
			code_arrow.next_to, code[7], LEFT * 0.5
		)

		self.play(
			Indicate(heights[5]),
		)

		self.play(
			Indicate(heights[5])
		)

		self.play(
			heights[5].set_color, YELLOW
		)

		self.wait(3)

		self.play(
			*[FadeOut(surround_rects[i]) for i in range(1, 4)],
			FadeOut(code_arrow),
			FadeOut(i_indicator),
			FadeOut(j_indicator), 
			FadeOut(S),
			FadeOut(surround_S)
		)

		self.wait(23)

	def move_tracker(self, indicator, sorted_box_shapes, j, start=0):
		copy = indicator.copy()
		current = copy.get_center()
		copy.next_to(sorted_box_shapes[j], DOWN)
		shift_amount = copy.get_center() - current
		self.play(
			indicator.next_to, sorted_box_shapes[j], DOWN,
			indicator[1][1].increment_value, j - start,
			indicator[1][1].shift, shift_amount,
		)



	def generate_code(self):
		code_scale = 0.7
		
		code = []

		def_statement = TextMobject("def ",  r"$\text{tallestStack}$(boxes):")
		def_statement[0].set_color(MONOKAI_BLUE)
		def_statement[1][:12].set_color(MONOKAI_GREEN)
		def_statement[1][13:18].set_color(MONOKAI_ORANGE)
		def_statement.scale(code_scale)
		def_statement.to_edge(LEFT)
		code.append(def_statement)


		line_1 = TextMobject(r"boxes.sort(key$=$lambda x: x$[0]$)")
		line_1.scale(code_scale)
		line_1.next_to(def_statement, DOWN * 0.5)
		line_1.to_edge(LEFT * 2)
		# line_1[0][5].shift(DOWN * SMALL_BUFF)
		line_1[0][6:10].set_color(MONOKAI_BLUE)
		line_1[0][11:14].set_color(MONOKAI_ORANGE)
		line_1[0][14].set_color(MONOKAI_PINK)
		line_1[0][15:21].set_color(MONOKAI_BLUE)
		line_1[0][-3].set_color(MONOKAI_PURPLE)
		code.append(line_1)

		line_2 = TextMobject(r"heights $= \{ \text{box:box}[2] \text{ for box in boxes}\}$")
		line_2.scale(code_scale)
		line_2.next_to(line_1, DOWN * 0.5)
		line_2.to_edge(LEFT * 2)
		line_2[0][7].set_color(MONOKAI_PINK)
		line_2[0][17].set_color(MONOKAI_PURPLE)
		line_2[0][19:22].set_color(MONOKAI_PINK)
		line_2[0][25:27].set_color(MONOKAI_PINK)
		code.append(line_2)

		line_3 = TextMobject(r"for i in range(1, len(boxes)):")
		line_3.scale(code_scale)
		line_3.next_to(line_2, DOWN * 0.5)
		line_3.to_edge(LEFT * 2)
		line_3[0][:3].set_color(MONOKAI_PINK)
		line_3[0][4:6].set_color(MONOKAI_PINK)
		line_3[0][6:11].set_color(MONOKAI_BLUE)
		line_3[0][12].set_color(MONOKAI_PURPLE)
		line_3[0][14:17].set_color(MONOKAI_BLUE)
		code.append(line_3)

		line_4 = TextMobject(r"box $= \text{boxes}[\text{i}]$")
		line_4.scale(code_scale)
		line_4.next_to(line_3, DOWN * 0.5)
		line_4.to_edge(LEFT * 3)
		line_4[0][3].set_color(MONOKAI_PINK)
		code.append(line_4)

		line_5 = TextMobject(r"S $= [\text{boxes}[\text{j}] \text{ for j in range(i) if canBeStacked(boxes}[\text{j}] \text{, box)}]$")
		line_5.scale(code_scale)
		line_5[0][1].set_color(MONOKAI_PINK)
		line_5[0][11:14].set_color(MONOKAI_PINK)
		line_5[0][15:17].set_color(MONOKAI_PINK)
		line_5[0][17:22].set_color(MONOKAI_BLUE)
		line_5[0][25:27].set_color(MONOKAI_PINK)
		line_5[0][27:39].set_color(MONOKAI_BLUE)
		line_5.next_to(line_4, DOWN * 0.5)
		line_5.to_edge(LEFT * 3)
		code.append(line_5)

		line_6 = TextMobject(r"$\text{heights}[\text{box}] = $", r"$\text{ box}[2] \text{ } + $", r"$\text{ max}([\text{heights}[\text{box}] \text{ for box in S}]$, default$=$0)")
		line_6.scale(code_scale)
		line_6[0][12].set_color(MONOKAI_PINK)
		line_6[1][4].set_color(MONOKAI_PURPLE)
		line_6[1][6].set_color(MONOKAI_PINK)
		line_6[2][:3].set_color(MONOKAI_BLUE)
		line_6[2][17:20].set_color(MONOKAI_PINK)
		line_6[2][23:25].set_color(MONOKAI_PINK)
		line_6[2][28:35].set_color(MONOKAI_ORANGE)
		line_6[2][35].set_color(MONOKAI_PINK)
		line_6[2][36].set_color(MONOKAI_PURPLE)
		line_6.next_to(line_5, DOWN * 0.5)
		line_6.to_edge(LEFT * 3)
		code.append(line_6)

		line_7 = TextMobject(r"return max(heights.values(), default$=$0)")
		line_7.scale(code_scale)
		line_7[0][:6].set_color(MONOKAI_PINK)
		line_7[0][6:9].set_color(MONOKAI_BLUE)
		line_7[0][18:24].set_color(MONOKAI_BLUE)
		line_7[0][27:34].set_color(MONOKAI_ORANGE)
		line_7[0][34].set_color(MONOKAI_PINK)
		line_7[0][35].set_color(MONOKAI_PURPLE)
		line_7.next_to(line_6, DOWN * 0.5)
		line_7.to_edge(LEFT * 2)
		code.append(line_7)

		line_8 = TextMobject(r"def canBeStacked(top, bottom):")
		line_8.scale(code_scale)
		line_8[0][:3].set_color(MONOKAI_BLUE)
		line_8[0][3:15].set_color(MONOKAI_GREEN)
		line_8[0][16:19].set_color(MONOKAI_ORANGE)
		line_8[0][20:26].set_color(MONOKAI_ORANGE)
		line_8.next_to(line_7, DOWN * 0.5)
		line_8.to_edge(LEFT)
		code.append(line_8)

		line_9 = TextMobject(r"return top$[0] <$ bottom$[0]$ and top$[1] <$ bottom$[1]$")
		line_9.scale(code_scale)
		line_9[0][:6].set_color(MONOKAI_PINK)
		line_9[0][10].set_color(MONOKAI_PURPLE)
		line_9[0][12].set_color(MONOKAI_PINK)
		line_9[0][20].set_color(MONOKAI_PURPLE)
		line_9[0][22:25].set_color(MONOKAI_PINK)
		line_9[0][29].set_color(MONOKAI_PURPLE)
		line_9[0][31].set_color(MONOKAI_PINK)
		line_9[0][-2].set_color(MONOKAI_PURPLE)
		line_9.next_to(line_8, DOWN * 0.5)
		line_9.to_edge(LEFT * 2)
		code.append(line_9)

		return VGroup(*code)

class DPConclusion(Scene):
	def construct(self):
		title = TextMobject("Finding Subproblems")
		title.scale(1.2)
		title.shift(UP * 3.5)

		h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
		h_line.next_to(title, DOWN)
		self.play(
			Write(title),
			ShowCreation(h_line)
		)

		self.types_of_subprobs(title, h_line)

	def types_of_subprobs(self, title, h_line):
		
		self.wait(5)
		common_subproblems = TextMobject("Common Subproblems")
		common_subproblems.next_to(h_line, DOWN)
		self.play(
			Write(common_subproblems)
		)
		self.wait(7)

		grid = self.create_grid(1, 9, 1)
		grid_group = self.get_group_grid(grid)


		for i in range(1, 10):
			text = TextMobject(r"$x_{0}$".format(i)).scale(0.8)
			text.move_to(grid[0][i - 1].get_center())
			grid_group.add(text)

		grid_group.move_to(ORIGIN)

		input_problem = TextMobject(r"Input: $x_1, x_2, \ldots , x_n$")
		input_problem.scale(0.8)
		input_problem.next_to(grid_group, UP)
		self.play(
			FadeIn(grid_group),
			FadeIn(input_problem)
		)

		self.wait()

		subproblem = TextMobject(r"Subproblem: $x_1, x_2, \ldots , x_i$")
		subproblem.scale(0.8)
		subproblem.next_to(grid_group, DOWN)

		self.play(
			Write(subproblem),
			grid_group[0].set_fill, BLUE, 0.3,
			grid_group[1].set_fill, BLUE, 0.3,
			grid_group[2].set_fill, BLUE, 0.3,
			grid_group[3].set_fill, BLUE, 0.3,
			grid_group[4].set_fill, BLUE, 0.3,
			grid_group[5].set_fill, BLUE, 0.3,
		)

		self.wait(6)

		self.play(
			input_problem.shift, UP * 1,
			grid_group.shift, UP * 1,
			subproblem.shift, UP * 1
		)

		self.wait(5)

		random_input = TextMobject(r"Input: $x_1, x_2, \ldots , x_n$ in random order")
		random_input.scale(0.8)
		random_input.move_to(DOWN)
		random_order = list(range(9))
		random.seed(2)
		random.shuffle(random_order)
		text_x = {}

		random_grid = self.create_grid(1, 9, 1)
		random_grid_group = self.get_group_grid(random_grid)
		random_grid_group.next_to(random_input, DOWN)

		for i in range(1, 10):
			text = TextMobject(r"$x_{0}$".format(i)).scale(0.8)
			text.move_to(random_grid[0][random_order[i - 1]].get_center())
			text_x[random_order[i - 1]] = text
			random_grid_group.add(text)

		
		self.play(
			FadeIn(random_input),
			FadeIn(random_grid_group)
		)

		original_order = list(range(9))

		mapping = [(i, random_order.index(i)) for i in range(9)]
		sorting = [ApplyMethod(
			text_x[i].move_to, 
			random_grid[0][j].get_center()) for i, j in mapping]
		
		self.wait(3)

		self.play(
			*sorting,
			run_time=2
		)

		self.wait()

		random_subproblem = TextMobject(r"Subproblem: $x_1, x_2, \ldots , x_i$ after sorting")
		random_subproblem.scale(0.8)
		random_subproblem.next_to(random_grid_group, DOWN)

		self.play(
			Write(random_subproblem),
			random_grid_group[0].set_fill, BLUE, 0.3,
			random_grid_group[1].set_fill, BLUE, 0.3,
			random_grid_group[2].set_fill, BLUE, 0.3,
			random_grid_group[3].set_fill, BLUE, 0.3,
			random_grid_group[4].set_fill, BLUE, 0.3,
			random_grid_group[5].set_fill, BLUE, 0.3,
		)

		self.wait(4)

		self.play(
			FadeOut(grid_group),
			FadeOut(random_grid_group),
			FadeOut(random_subproblem),
			FadeOut(random_input),
			FadeOut(input_problem),
			FadeOut(subproblem)
		)

		self.wait(3)

		grid = self.create_grid(1, 9, 1)
		grid_group = self.get_group_grid(grid)


		for i in range(1, 10):
			text = TextMobject(r"$x_{0}$".format(i)).scale(0.8)
			text.move_to(grid[0][i - 1].get_center())
			grid_group.add(text)

		grid_group.move_to(UP * 0.5)

		y_grid = self.create_grid(1, 8, 1)
		y_grid_group = self.get_group_grid(y_grid)
		y_grid_group.next_to(grid_group, DOWN)
		
		for i in range(1, 9):
			text = TextMobject(r"$y_{0}$".format(i)).scale(0.8)
			text.move_to(y_grid[0][i - 1].get_center())
			y_grid_group.add(text)



		input_problem = TextMobject(r"Input: $x_1, x_2, \ldots , x_n$ and $y_1, y_2, \ldots , y_m$")
		input_problem.scale(0.8)
		input_problem.next_to(grid_group, UP)

		subproblem = TextMobject(r"Subproblem: $x_1, x_2, \ldots , x_i$ and $y_1, y_2, \ldots , y_j$")
		subproblem.scale(0.8)
		subproblem.next_to(y_grid_group, DOWN)

		y_grid_group.shift(LEFT * 0.5)

		self.play(
			FadeIn(grid_group),
			FadeIn(y_grid_group),
			FadeIn(input_problem)
		)

		self.wait(3)

		self.play(
			Write(subproblem),
			grid_group[0].set_fill, BLUE, 0.3,
			grid_group[1].set_fill, BLUE, 0.3,
			grid_group[2].set_fill, BLUE, 0.3,
			grid_group[3].set_fill, BLUE, 0.3,
			grid_group[4].set_fill, BLUE, 0.3,
			grid_group[5].set_fill, BLUE, 0.3,
			y_grid_group[0].set_fill, BLUE, 0.3,
			y_grid_group[1].set_fill, BLUE, 0.3,
			y_grid_group[2].set_fill, BLUE, 0.3,
			y_grid_group[3].set_fill, BLUE, 0.3,
			y_grid_group[4].set_fill, BLUE, 0.3,
		)

		self.wait(5)

		self.play(
			FadeOut(input_problem),
			FadeOut(grid_group),
			FadeOut(y_grid_group),
			FadeOut(subproblem)
		)

		grid = self.create_grid(1, 9, 1)
		grid_group = self.get_group_grid(grid)

		for i in range(1, 10):
			text = TextMobject(r"$x_{0}$".format(i)).scale(0.8)
			text.move_to(grid[0][i - 1].get_center())
			grid_group.add(text)

		grid_group.move_to(ORIGIN)

		input_problem = TextMobject(r"Input: $x_1, x_2, \ldots , x_n$")
		input_problem.scale(0.8)
		input_problem.next_to(grid_group, UP)
		self.play(
			FadeIn(grid_group),
			FadeIn(input_problem)
		)

		self.wait(3)

		subproblem = TextMobject(r"Subproblem: $x_i, x_{i + 1}, \ldots , x_j$")
		subproblem.scale(0.8)
		subproblem.next_to(grid_group, DOWN)

		self.play(
			Write(subproblem),
			grid_group[2].set_fill, BLUE, 0.3,
			grid_group[3].set_fill, BLUE, 0.3,
			grid_group[4].set_fill, BLUE, 0.3,
			grid_group[5].set_fill, BLUE, 0.3,
			grid_group[6].set_fill, BLUE, 0.3,
		)

		self.wait(5)

		self.play(
			FadeOut(subproblem),
			FadeOut(grid_group),
			FadeOut(input_problem)
		)

		self.wait(4)

		grid = self.create_grid(6, 8, 0.7)
		grid_group = self.get_group_grid(grid)

		grid_group.move_to(DOWN * 0.5)

		input_problem = TextMobject(r"Input: matrix $A_{mn}$")
		input_problem.scale(0.8)
		input_problem.next_to(grid_group, UP)
		self.play(
			FadeIn(grid_group),
			FadeIn(input_problem)
		)

		self.wait(5)

		subproblem = TextMobject(r"Subproblem: matrix $A_{ij}$")
		subproblem.scale(0.8)
		subproblem.next_to(grid_group, DOWN)

		coords = []
		for i in range(7):
			for j in range(5):
				coords.append((j, i))

		self.play(
			Write(subproblem),
			*[ApplyMethod(grid[i][j].set_fill, BLUE, 0.3) for i, j in coords]		
		)

		self.wait(12)

		frame_rect = ScreenRectangle(height=5)
		frame_rect.move_to(DOWN * 0.1)

		self.play(
			FadeOut(common_subproblems),
			FadeOut(input_problem),
			FadeOut(subproblem),
			ReplacementTransform(grid_group, frame_rect),
			run_time=2
		)

		practice = TextMobject("Practice and experience is the key!")
		practice.next_to(frame_rect, DOWN)
		self.play(
			Write(practice)
		)

		self.wait(7)

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

class Thumbnail(Scene):
	def construct(self):
		grid = self.create_grid(5, 5, 1)
		grid_group = self.get_group_grid(grid)
		grid_group.move_to(DOWN * 0.5)
		self.play(
			FadeIn(grid_group)
		)

		title = TextMobject("Dynamic Programming")
		title.scale(1.7)
		title.next_to(grid_group, UP * 1.5)

		self.play(
			FadeIn(title)
		)

		color_fill = DARK_BLUE_B
		color_border = GREEN_SCREEN
		color_flash = BLUE

		self.play(
			grid[0][0].set_fill, color_fill, 1,
			grid[0][1].set_fill, color_fill, 1,
			grid[1][1].set_fill, color_fill, 1,
			grid[1][2].set_fill, color_fill, 1,
			grid[2][2].set_fill, color_fill, 1,
			grid[3][2].set_fill, color_fill, 1,
			grid[3][3].set_fill, color_fill, 1,
			grid[3][4].set_fill, color_fill, 1,
			grid[4][4].set_fill, color_fill, 1, 
		)

		group = VGroup(
			grid[0][0],
			grid[0][1],
			grid[1][1],
			grid[1][2],
			grid[2][2],
			grid[3][2],
			grid[3][3],
			grid[3][4],
			grid[4][4],
		)

		group.set_stroke(color=color_border, width=10)


		self.play(
			FadeIn(group)
		)

		self.play(
			Flash(grid[0][0], color=color_flash, num_lines=10, line_stroke_width=5),
			Flash(grid[0][1], color=color_flash, num_lines=10, line_stroke_width=5),
			Flash(grid[1][1], color=color_flash, num_lines=10, line_stroke_width=5),
			Flash(grid[1][2], color=color_flash, num_lines=10, line_stroke_width=5),
			Flash(grid[2][2], color=color_flash, num_lines=10, line_stroke_width=5),
			Flash(grid[3][2], color=color_flash, num_lines=10, line_stroke_width=5),
			Flash(grid[3][3], color=color_flash, num_lines=10, line_stroke_width=5),
			Flash(grid[3][4], color=color_flash, num_lines=10, line_stroke_width=5),
			Flash(grid[4][4], color=color_flash, num_lines=10, line_stroke_width=5),
		)

		self.wait()

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





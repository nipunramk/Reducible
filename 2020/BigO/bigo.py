from big_ol_pile_of_manim_imports import *

class PlotFunctions(GraphScene):
	CONFIG = {
	"x_min" : 0,
	"x_max" : 900,
	"y_min" : 0,
	"y_max" : 1000000,
	"graph_origin" : LEFT * 4 + DOWN * 3,
	"function_color" : RED,
	"axes_color" : GREEN,
	"x_tick_frequency": 900,
	"y_tick_frequency": 1000000,
	 
	}
	def construct(self):
		self.setup_axes(animate=True)
		func_graph=self.get_graph(self.func_to_graph, self.function_color)
		func_graph2=self.get_graph(self.func_to_graph2)
		func_graph3 = self.get_graph(self.func_to_graph3, YELLOW)
		# vert_line = self.get_vertical_line_to_graph(TAU,func_graph,color=YELLOW)
		graph_lab = self.get_graph_label(func_graph, label = "x ^ 2")
		graph_lab2=self.get_graph_label(func_graph2, label = "\\frac{1}{2} x ^ 2")
		graph_lab3=self.get_graph_label(func_graph3, label = "x ^ 2 + 100000")
		# two_pi = TexMobject("x = 2 \\pi")
		# label_coord = self.input_to_graph_point(TAU,func_graph)
		# two_pi.next_to(label_coord,RIGHT+UP)
 
		self.play(ShowCreation(func_graph),ShowCreation(func_graph2), 
			ShowCreation(func_graph3))
		self.play(ShowCreation(graph_lab), 
			ShowCreation(graph_lab2), ShowCreation(graph_lab3))
	 
	def func_to_graph(self,x):
		return x ** 2
	 
	def func_to_graph2(self,x):
		return 0.5 * x ** 2

	def func_to_graph3(self, x):
		return x ** 2 + 100000


class GraphCounter(GraphScene):
	CONFIG = {
	"x_min" : 0,
	"x_max" : 10,
	"y_min" : 0,
	"y_max" : 100,
	"graph_origin" : LEFT * 6 + DOWN * 3,
	"function_color" : RED ,
	"axes_color" : GREEN,
	"x_tick_frequency": 10,
	"y_tick_frequency": 10, 
	"x_labeled_nums": range(10, 11),
	"y_labeled_nums": range(100, 101),
	"x_tick_frequency": 10,
	"y_tick_frequency": 100,
	"y_axis_height": 5,
	"x_axis_width": 8,
	}
	def construct(self):
		self.setup_axes(animate=True)
		x = TexMobject("x")
		x.move_to(RIGHT * 4)

		y = TexMobject("y")
		y.next_to(x, DOWN * 2)

		vert_line_x = Line(x.get_center() + RIGHT * 0.25 + UP * 0.25, 
			x.get_center() + RIGHT * 0.25 + DOWN * 0.25)
		h_line_x = Line(x.get_center() + RIGHT * 0.25 + DOWN * 0.25, 
			x.get_center() + RIGHT * 0.65 + DOWN * 0.25)
		vert_line_y = Line(y.get_center() + RIGHT * 0.25 + UP * 0.25, 
			y.get_center() + RIGHT * 0.25 + DOWN * 0.25)
		h_line_y = Line(y.get_center() + RIGHT * 0.25 + DOWN * 0.25, 
			y.get_center() + RIGHT * 0.65 + DOWN * 0.25)

		self.x_max = 5
		x_val = TexMobject(str(self.x_max))
		x_val.scale(0.8)
		x_val.next_to(vert_line_x, RIGHT * 0.5)
		y_val = TexMobject(str(self.func_to_graph(self.x_max)))
		y_val.scale(0.8)
		y_val.next_to(vert_line_y, RIGHT * 0.5)
		func_graph = self.get_graph(self.func_to_graph, self.function_color)
		self.play(ShowCreation(func_graph), FadeIn(x), FadeIn(y), 
			FadeIn(vert_line_x), FadeIn(vert_line_y), FadeIn(h_line_x),
			FadeIn(h_line_y), FadeIn(x_val), FadeIn(y_val))

		for x_max in range(6, 11):
			previous_x_val = x_val
			previous_y_val = y_val
			self.x_max = x_max
			new_graph = self.get_graph(self.func_to_graph, self.function_color)
			x_val = TexMobject(str(self.x_max))
			x_val.scale(0.8)
			x_val.next_to(vert_line_x, RIGHT * 0.5)
			y_val = TexMobject(str(self.func_to_graph(self.x_max)))
			y_val.scale(0.8)
			y_val.next_to(vert_line_y, RIGHT * 0.5)
			self.play(
				Transform(func_graph, new_graph), 
				ReplacementTransform(previous_x_val, x_val),
				ReplacementTransform(previous_y_val, y_val),
				run_time=0.5
			)

	def func_to_graph(self, x):
		return x ** 2

class Thumbnail(GraphScene):
	CONFIG = {
	"x_min" : 0,
	"x_max" : 100,
	"y_min" : 0,
	"y_max" : 200,
	"x_axis_label": None,
	"y_axis_label": None,
	"graph_origin" : RIGHT * 1.1 + DOWN * 0.2,
	"function_color" : RED,
	"axes_color" : WHITE,
	"x_tick_frequency": 900,
	"y_tick_frequency": 1000000,
	"y_axis_height": 4,
	"x_axis_width": 5,
	}

	def construct(self):
		self.setup_axes(animate=True)
		func_graph=self.get_graph(self.func_to_graph, self.function_color)
		func_graph2=self.get_graph(self.func_to_graph2, YELLOW)
		func_graph3 = self.get_graph(self.func_to_graph3, GREEN_SCREEN)
		func_graph4 = self.get_graph(self.func_to_graph4, GREEN)
		func_graph5 = self.get_graph(self.func_to_graph5, BRIGHT_RED)
		func_graph6 = self.get_graph(self.func_to_graph6, ORANGE)


		self.play(
			ShowCreation(func_graph),
			ShowCreation(func_graph2), 
			ShowCreation(func_graph3),
			ShowCreation(func_graph4),
			ShowCreation(func_graph5),
			ShowCreation(func_graph6),
			)

		bigo = TextMobject("Big O")
		bigo.scale(4)
		bigo.move_to(LEFT * 3 + UP * 1.5)
		self.play(FadeIn(bigo))

		o1 = TextMobject(r"O($1$)")
		o1.scale(1.2)
		o1.set_color(GREEN_SCREEN)
		o2 = TextMobject(r"O($\log n$)")
		o2.set_color(GREEN)
		o2.scale(1.2)
		o3 = TextMobject(r"O($n$)")
		o3.set_color(YELLOW)
		o3.scale(1.2)
		o4 = TextMobject(r"O($n \log n$)")
		o4.set_color(ORANGE)
		o4.scale(1.2)
		o5 = TextMobject(r"O($n^2$)")
		o5.set_color(RED)
		o5.scale(1.2)
		o6 = TextMobject(r"O($2^n$)")
		o6.set_color(BRIGHT_RED)
		o6.scale(1.2)
		
		o1.move_to(LEFT * 5.4 + DOWN * 1.5)
		o2.move_to(LEFT * 3.4 + DOWN * 1.5)
		o3.move_to(LEFT * 1.4 + DOWN * 1.5)
		o4.move_to(RIGHT * 1 + DOWN * 1.5)
		o5.move_to(RIGHT * 3.4 + DOWN * 1.5)
		o6.move_to(RIGHT * 5.4 + DOWN * 1.5)

		self.play(
			ShowCreation(o1),
			ShowCreation(o2), 
			ShowCreation(o3),
			ShowCreation(o4),
			ShowCreation(o5),
			ShowCreation(o6),
		)

		line_before_arrow = Line(LEFT * 5.5 + DOWN * 2.5, RIGHT * 5.4 + DOWN * 2.5)
		line_before_arrow.set_color(color=[GREEN_SCREEN, GREEN, YELLOW, ORANGE, RED, BRIGHT_RED][::-1])
	
		arrow_head = RegularPolygon(n=3, color=BRIGHT_RED, fill_color=BRIGHT_RED, fill_opacity=1)
		arrow_head.scale(0.2)
		# arrow_head.rotate(-TAU / 4)
		arrow_head.move_to(RIGHT * 5.5 + DOWN * 2.5)
		self.play(FadeIn(line_before_arrow), FadeIn(arrow_head))


	def func_to_graph(self,x):
		return 0.2 * x ** 2
	 
	def func_to_graph2(self,x):
		return x

	def func_to_graph3(self, x):
		if x < 1:
			return 0
		return np.log(x)

	def func_to_graph4(self, x):
		if x < 1:
			return 0
		return 3 * np.sqrt(x)

	def func_to_graph5(self, x):
		return 0.1 * x ** 3

	def func_to_graph6(self, x):
		return x ** 1.2


class Comparison(GraphScene):
	CONFIG = {
	"x_min" : 0,
	"x_max" : 6,
	"y_min" : 0,
	"y_max" : 2100,
	"x_axis_label": None,
	"y_axis_label": None,
	"graph_origin" : LEFT * 2.4 + DOWN * 2,
	"function_color" : RED,
	"axes_color" : WHITE,
	"x_tick_frequency": 1,
	"y_tick_frequency": 2100,
	"x_axis_width": 5,
	"y_axis_height": 4,
	"x_axis_label": "$\log(n)$",
	"y_axis_label": "Time (s)",
	"label_scale": 0.7,
	}
	def construct(self):
		comparison_text = TextMobject("Bob's solution vs Alice's solution")
		comparison_text.shift(UP * 3.5)

		comparison_text.scale(0.8)
		comparison_text[:3].set_color(ORANGE)
		comparison_text[15:20].set_color(GREEN_SCREEN)

		bob_text = comparison_text[:3].copy()
		alice_text = comparison_text[15:20].copy()

		self.play(FadeIn(comparison_text))
		self.wait()

		efficiency_q = TextMobject("Which solution is more efficient?")
		efficiency_q.next_to(comparison_text, DOWN)
		efficiency_q.scale(0.8)
		self.play(Write(efficiency_q))
		self.wait(3)

		measure_q = TextMobject("How do we measure efficiency?")
		measure_q.scale(0.8)
		measure_q.move_to(DOWN * 3)
		self.play(Write(measure_q))

		clock_circle = Circle(radius=0.5)
		clock_circle.set_color(BLUE)

		minute_hand = Line(clock_circle.get_center(), clock_circle.get_center() + UP * clock_circle.radius * 0.95)
		hour_hand = Line(clock_circle.get_center(), clock_circle.get_center() + RIGHT * clock_circle.radius * 0.75)
		minute_hand.set_color(BLUE)
		hour_hand.set_color(BLUE)

		clock = VGroup(clock_circle, minute_hand, hour_hand)
		clock.shift(LEFT * 3 + DOWN)
		
		clock2 = clock.copy()
		clock2.shift(RIGHT * 6)
		self.wait(8)

		clock_circle2, minute_hand2, hour_hand2 = clock2[0], clock2[1], clock2[2]

		time = TextMobject("We could time the algorithm for some input?")
		time.scale(0.8)
		time.move_to(measure_q.get_center())
		left_n = TexMobject("n = 10000")
		left_n.set_color(ORANGE)
		left_n.scale(0.7)
		right_n = left_n.copy()
		left_n.shift(LEFT * 3)
		right_n.shift(RIGHT * 3)
		right_n.set_color(GREEN_SCREEN)
		bob_text.next_to(left_n, UP)
		alice_text.next_to(right_n, UP)
		self.play(FadeIn(clock), 
			FadeIn(clock2), 
			ReplacementTransform(measure_q, time), 
			FadeIn(left_n),
			FadeIn(right_n),
			TransformFromCopy(comparison_text[:3], bob_text),
			TransformFromCopy(comparison_text[15:20], alice_text),
			run_time=2
		)

		self.wait(4)

		self.play(Rotate(minute_hand, angle=-360*4*DEGREES, 
				about_point=clock_circle.get_center()), 
			Rotate(hour_hand, angle=-30*4*DEGREES, 
				about_point=clock_circle.get_center()),
			Rotate(minute_hand2, angle=-360*4*DEGREES, 
				about_point=clock_circle2.get_center()), 
			Rotate(hour_hand2, angle=-30*4*DEGREES, 
				about_point=clock_circle2.get_center()),
			run_time=4)
		# self.play(Transform(minute_hand, minute_hand_rotated))
		left_time = TextMobject("58 seconds")
		left_time[:2].set_color(ORANGE)
		right_time = TextMobject("15 seconds")
		right_time[:2].set_color(GREEN_SCREEN)
		left_time.scale(0.7)
		right_time.scale(0.7)
		left_time.next_to(clock, DOWN)
		right_time.next_to(clock2, DOWN)
		self.play(
			FadeIn(left_time),
			FadeIn(right_time)
			)
		self.wait(7)

		# problem = 

		# self.play(
		# 	FadeOut(clock), 
		# 	FadeOut(clock2),
		# 	FadeOut(left_time),
		# 	FadeOut(right_time)
		# 	)


		new_time = TextMobject("We could time the algorithm for several inputs?")
		new_time.scale(0.8)
		new_time.move_to(time.get_center())
		self.play(
			Transform(time, new_time)
			)
		self.wait()
		rows, length, width = 6, 3, 4
		left_table = make_2_column_table(LEFT * 5, rows, length, width)

		right_table = make_2_column_table(RIGHT * 5, rows, length, width)
		split_length = width / rows
		left_right_length = length / 2
		# self.play(
		# 	FadeIn(left_table),
		# 	FadeIn(right_table),
		# 	)

		l_n = left_n[0]
		left_n_val = left_n[2:]
		left_time_val = left_time[:2]
		left_n_center = get_center_row_col(left_table, 0, 0, split_length, left_right_length)
		left_n_val_center = get_center_row_col(left_table, 3, 0, split_length, left_right_length)
		left_time_center = get_center_row_col(left_table, 3, 1, split_length, left_right_length)
		other_anims = [FadeOut(left_n[1]), FadeOut(left_time[2:]), FadeOut(clock)]
		time_label_center = get_center_row_col(left_table, 0, 1, split_length, left_right_length)
		left_anims, left_time_label, left_n_val = self.transform_text_to_table(l_n, left_n_val, left_time_val, left_table, left_n_center,
			left_n_val_center, left_time_center, other_anims, time_label_center)
		new_bob = bob_text.copy()
		new_bob.move_to(time_label_center)
		new_bob.shift(UP * 0.7 + LEFT * 0.7)

		r_n = right_n[0]
		right_n_val = right_n[2:]
		right_time_val = right_time[:2]
		right_n_center = get_center_row_col(right_table, 0, 0, split_length, left_right_length)
		right_n_val_center = get_center_row_col(right_table, 3, 0, split_length, left_right_length)
		right_time_center = get_center_row_col(right_table, 3, 1, split_length, left_right_length)
		other_anims = [FadeOut(right_n[1]), FadeOut(right_time[2:]), FadeOut(clock2)]
		time_label_center = get_center_row_col(right_table, 0, 1, split_length, left_right_length)
		right_anims, right_time_label, right_n_val = self.transform_text_to_table(r_n, right_n_val, right_time_val, right_table, right_n_center,
			right_n_val_center, right_time_center, other_anims, time_label_center, color=GREEN_SCREEN)
		new_alice = alice_text.copy()
		new_alice.move_to(time_label_center)
		new_alice.shift(UP * 0.7 + LEFT * 0.7)


		text_anims = [Transform(bob_text, new_bob), Transform(alice_text, new_alice)]
		self.play(*left_anims + right_anims + text_anims, run_time=3)
		self.wait()

		data_left = {
		(1, 0): TexMobject("100"), 
		(1, 1): TexMobject("0.1"), 
		(2, 0): TexMobject("1000"), 
		(2, 1): TexMobject("3"), 
		(4, 0): TexMobject("100000"), 
		(4, 1): TexMobject("597"), 
		(5, 0): TextMobject("1000000"),
		(5, 1): TexMobject("2052"), 
		}
		all_text = []
		for coord, text in data_left.items():
			text.scale(0.7)
			center = get_center_row_col(left_table, coord[0], coord[1], split_length, left_right_length)
			text.move_to(center)
			text.set_color(ORANGE)
			all_text.append(text)

		new_left_n_val = TexMobject("10000")
		new_left_n_val.scale(0.7)
		new_left_n_val.move_to(left_n_val.get_center())
		new_left_n_val.set_color(ORANGE)
		new_right_n_val = TexMobject("10000")
		new_right_n_val.scale(0.7)
		new_right_n_val.move_to(right_n_val.get_center())
		new_right_n_val.set_color(GREEN_SCREEN)
		new_left_time_val = TexMobject("58")
		new_left_time_val.set_color(ORANGE)
		new_left_time_val.scale(0.7)
		new_left_time_val.move_to(left_time_val.get_center())
		new_right_time_val = TexMobject("15")
		new_right_time_val.set_color(GREEN_SCREEN)
		new_right_time_val.scale(0.7)
		new_right_time_val.move_to(right_time_val.get_center())
		misc_anims = [FadeOut(left_n_val),
			FadeIn(new_left_n_val), FadeOut(right_n_val), 
			FadeIn(new_right_n_val), FadeOut(left_time_val),
			FadeOut(right_time_val), FadeIn(new_left_time_val),
			FadeIn(new_right_time_val)]

		data_left[(0, 0)] = l_n
		data_left[(0, 1)] = left_time_label
		data_left[(3, 0)] = new_left_n_val
		data_left[(3, 1)] = new_left_time_val

		data_right = {
		(1, 0): TexMobject("100"), 
		(1, 1): TexMobject("0.01"), 
		(2, 0): TexMobject("1000"), 
		(2, 1): TexMobject("0.9"), 
		(4, 0): TexMobject("100000"), 
		(4, 1): TexMobject("247"), 
		(5, 0): TextMobject("1000000"),
		(5, 1): TexMobject("900"), 
		}


		for coord, text in data_right.items():
			text.scale(0.7)
			center = get_center_row_col(right_table, coord[0], coord[1], split_length, left_right_length)
			text.move_to(center)
			text.set_color(GREEN_SCREEN)
			all_text.append(text)

		self.play(*[FadeIn(text) for text in all_text] + misc_anims, run_time=2)
		self.wait(5)

		data_right[(0, 0)] = r_n
		data_right[(0, 1)] = right_time_label
		data_right[(3, 0)] = new_right_n_val
		data_right[(3, 1)] = new_right_time_val



		graphing_option = TextMobject("We could even graph time vs input?")
		graphing_option.scale(0.8)
		graphing_option.move_to(new_time.get_center())
		self.play(ReplacementTransform(time, graphing_option))

		log_data_left = {
		(0, 0): TextMobject(r"$\log(n)$"),
		(0, 1): TextMobject("Time (s)"),
		(1, 0): TexMobject("2"), 
		(1, 1): TexMobject("0.1"), 
		(2, 0): TexMobject("3"), 
		(2, 1): TexMobject("3"), 
		(3, 0): TexMobject("4"),
		(3, 1): TexMobject("58"),
		(4, 0): TexMobject("5"), 
		(4, 1): TexMobject("597"), 
		(5, 0): TextMobject("6"),
		(5, 1): TexMobject("2052"), 
		}

		log_data_right = {
		(0, 0): TextMobject(r"$\log(n)$"),
		(0, 1): TextMobject("Time (s)"),
		(1, 0): TexMobject("2"), 
		(1, 1): TexMobject("0.01"), 
		(2, 0): TexMobject("3"), 
		(2, 1): TexMobject("0.9"),
		(3, 0): TexMobject("4"),
		(3, 1): TexMobject("15"),
		(4, 0): TexMobject("5"), 
		(4, 1): TexMobject("247"), 
		(5, 0): TextMobject("6"),
		(5, 1): TexMobject("900"), 
		}



		first_transforms = []
		transform_to_log = []
		for coord in data_left:
			original = data_left[coord]
			new = log_data_left[coord]
			new.scale(0.7)
			center = get_center_row_col(left_table, coord[0], coord[1], split_length, left_right_length)
			new.move_to(center)
			new.set_color(ORANGE)
			transform = ReplacementTransform(original, new)
			if coord == (0, 0):
				first_transforms.append(transform)
			else:
				transform_to_log.append(transform)

		for coord in data_right:
			original = data_right[coord]
			new = log_data_right[coord]
			new.scale(0.7)
			new.set_color(GREEN_SCREEN)
			center = get_center_row_col(right_table, coord[0], coord[1], split_length, left_right_length)
			new.move_to(center)
			transform = ReplacementTransform(original, new)
			if coord == (0, 0):
				first_transforms.append(transform)
			else:
				transform_to_log.append(transform)

		self.wait(8)
		self.play(*first_transforms)
		self.wait()
		self.play(*transform_to_log)

		axes = self.setup_axes(animate=True)
		dots = []
		coords_log_scale_left = [(2, 0.1), (3, 3), (4, 58), (5, 597), (6, 2052)]
		for x, y in coords_log_scale_left:
			dot = Dot(radius=0.07)
			graph_position = self.coords_to_point(x, y)
			dot.set_color(ORANGE)
			dot.move_to(graph_position)
			
			dots.append(dot)

		coords_log_scale_right = [(2, 0.01), (3, 0.9), (4, 15), (5, 247), (6, 900)]
		for x, y in coords_log_scale_right:
			dot = Dot(radius=0.07)
			graph_position = self.coords_to_point(x, y)
			dot.set_color(GREEN_SCREEN)
			dot.move_to(graph_position)
			dots.append(dot)

		self.play(*[GrowFromCenter(dot) for dot in dots])
		self.wait()

		bob_graph = self.get_graph(self.bob, ORANGE)
		alice_graph = self.get_graph(self.alice, GREEN_SCREEN)
		self.play(
			ShowCreation(bob_graph),
			ShowCreation(alice_graph)
			)
		self.wait(10)

		problem_with_timing = TextMobject("But what are the issues with this approach?")
		problem_with_timing.scale(0.8)
		problem_with_timing.move_to(graphing_option.get_center())
		self.play(ReplacementTransform(graphing_option, problem_with_timing))
		self.wait(8)

		first_problem = TextMobject("For one, this is really annoying to do!")
		first_problem.scale(0.8)
		first_problem.next_to(problem_with_timing, DOWN)
		self.play(Write(first_problem))
		self.wait(13)

		computer_issue = TextMobject("A different computer will give us completely different results!")
		computer_issue.scale(0.8)
		computer_issue.next_to(problem_with_timing, DOWN)
		self.play(ReplacementTransform(first_problem, computer_issue))
		self.wait(2)

		new_log_data_left = {
		(0, 0): TextMobject(r"$\log(n)$"),
		(0, 1): TextMobject("Time (s)"),
		(1, 0): TexMobject("2"), 
		(1, 1): TexMobject("0.05"), 
		(2, 0): TexMobject("3"), 
		(2, 1): TexMobject("2"), 
		(3, 0): TexMobject("4"),
		(3, 1): TexMobject("49"),
		(4, 0): TexMobject("5"), 
		(4, 1): TexMobject("500"), 
		(5, 0): TextMobject("6"),
		(5, 1): TexMobject("1800"), 
		}

		new_log_data_right = {
		(0, 0): TextMobject(r"$\log(n)$"),
		(0, 1): TextMobject("Time (s)"),
		(1, 0): TexMobject("2"), 
		(1, 1): TexMobject("0.03"), 
		(2, 0): TexMobject("3"), 
		(2, 1): TexMobject("2"),
		(3, 0): TexMobject("4"),
		(3, 1): TexMobject("35"),
		(4, 0): TexMobject("5"), 
		(4, 1): TexMobject("410"), 
		(5, 0): TextMobject("6"),
		(5, 1): TexMobject("1100"), 
		}

		fadeouts = [FadeOut(dot) for dot in dots]
		data_transforms = []
		for coord in log_data_left:
			original = log_data_left[coord]
			new = new_log_data_left[coord]
			new.scale(0.7)
			new.set_color(ORANGE)
			center = get_center_row_col(left_table, coord[0], coord[1], split_length, left_right_length)
			new.move_to(center)
			transform = ReplacementTransform(original, new)
			data_transforms.append(transform)
		for coord in log_data_right:
			original = log_data_right[coord]
			new = new_log_data_right[coord]
			new.set_color(GREEN_SCREEN)
			new.scale(0.7)
			center = get_center_row_col(right_table, coord[0], coord[1], split_length, left_right_length)
			new.move_to(center)
			transform = ReplacementTransform(original, new)
			data_transforms.append(transform)

		new_alice_graph = self.get_graph(self.alice2, GREEN_SCREEN)
		new_bob_graph = self.get_graph(self.bob2, ORANGE)
		graph_transformation = [ReplacementTransform(bob_graph, 
			new_bob_graph), ReplacementTransform(alice_graph, 
			new_alice_graph)]


		self.play(*fadeouts + data_transforms + graph_transformation, run_time=2)
		self.wait(5)

		machine_dependent = TextMobject("This issue is often called machine dependence.")
		machine_dependent[-18:-1].set_color(BRIGHT_RED)
		machine_dependent.scale(0.8)
		machine_dependent.move_to(computer_issue.get_center())
		self.play(ReplacementTransform(computer_issue, machine_dependent))
		self.wait(5)
		machine_ind = TextMobject("We want a measure that is machine independent.")
		machine_ind[-19:-1].set_color(GREEN_SCREEN)
		machine_ind.scale(0.8)
		machine_ind.move_to(machine_dependent.get_center())
		self.play(ReplacementTransform(machine_dependent, machine_ind))
		self.wait(12)

		fadeouts = [FadeOut(val) for val in new_log_data_left.values()]
		fadeouts += [FadeOut(val) for val in new_log_data_right.values()]
		fadeouts += [FadeOut(new_alice_graph), FadeOut(new_bob_graph), FadeOut(axes)]
		self.play(*fadeouts)

		self.wait(5)

		counting_operations = TextMobject("One machine independent measure is counting operations")
		counting_operations[3:21].set_color(GREEN_SCREEN)
		counting_operations.scale(0.8)
		counting_operations.move_to(problem_with_timing.get_center())
		self.play(
			FadeOut(machine_ind),
			ReplacementTransform(problem_with_timing, counting_operations)
			)

		

		count_data_left = {
		(0, 0): TextMobject(r"$n$"),
		(0, 1): TextMobject("Count"),
		(1, 0): TexMobject("20"), 
		(1, 1): TexMobject("9261"), 
		(2, 0): TexMobject("40"), 
		(2, 1): TexMobject("68921"), 
		(3, 0): TexMobject("60"),
		(3, 1): TexMobject("226981"),
		(4, 0): TexMobject("80"), 
		(4, 1): TexMobject("531441"), 
		(5, 0): TextMobject("100"),
		(5, 1): TexMobject("1030301"), 
		}

		count_data_right = {
		(0, 0): TextMobject(r"$n$"),
		(0, 1): TextMobject("Count"),
		(1, 0): TexMobject("20"), 
		(1, 1): TexMobject("441"), 
		(2, 0): TexMobject("40"), 
		(2, 1): TexMobject("1681"), 
		(3, 0): TexMobject("60"),
		(3, 1): TexMobject("3721"),
		(4, 0): TexMobject("80"), 
		(4, 1): TexMobject("6561"), 
		(5, 0): TextMobject("100"),
		(5, 1): TexMobject("10201"), 
		}

		for coord in count_data_left:
			data = count_data_left[coord]
			data.scale(0.6)
			data.set_color(ORANGE)
			center = get_center_row_col(left_table, coord[0], coord[1], split_length, left_right_length)
			data.move_to(center)
		
		for coord in count_data_right:
			data = count_data_right[coord]
			data.scale(0.6)
			data.set_color(GREEN_SCREEN)
			center = get_center_row_col(right_table, coord[0], coord[1], split_length, left_right_length)
			data.move_to(center)

		bob_image = ImageMobject("bob_solution")
		bob_image.scale(1.4)
		bob_image.shift(DOWN * 0.5)
		# alice_image.shift(DOWN * 2 + RIGHT * 3)
		self.play(FadeIn(bob_image))

		self.play(
			FadeIn(count_data_left[(0, 0)]),
			FadeIn(count_data_right[(0, 0)]),
			FadeIn(count_data_left[(0, 1)]),
			FadeIn(count_data_right[(0, 1)]),
			)


		counting_question = TextMobject("But what operation should we count?")
		counting_question.scale(0.8)
		counting_question.move_to(counting_operations.get_center())
		self.play(ReplacementTransform(counting_operations, counting_question))
		self.wait(3)
		answer = TextMobject("Well, let's look at the worst case scenario.")
		answer.scale(0.8)
		answer[19:24].set_color(BRIGHT_RED)
		answer.move_to(counting_question.get_center())
		self.play(ReplacementTransform(counting_question, answer))
		self.wait()
		counted_most = TextMobject("What operation occurs the most often?")
		counted_most.scale(0.8)
		counted_most.next_to(answer, DOWN)
		self.play(FadeIn(counted_most))
		self.wait(3)

		answer_most = TextMobject(r"Let's count the number of times $a + b + c == n$ is checked.")
		answer_most[26:34].set_color(ORANGE)
		answer_most.scale(0.8)
		answer_most.move_to(counted_most.get_center())
		arrow = Arrow(LEFT * 2 + DOWN * 0.9, ORIGIN + DOWN * 0.9)
		arrow.set_color(GREEN_SCREEN)
		self.play(
			FadeIn(arrow),
			ReplacementTransform(counted_most, answer_most)
			)
		self.wait(4)

		objects = self.show_count(bob_image, 20)
		for i in range(len(objects)):
			obj = objects[i]
			self.play(FadeIn(obj))
			if i < 2:
				self.wait(3)
			else:
				self.wait()

		self.wait(2)
		row1 = self.put_in_table(objects, left_table, 1, 20, split_length,
			left_right_length, count_data_left)

		objects_2 = self.show_count(bob_image, 40)
		self.play(*[ReplacementTransform(obj, new_obj) for obj, new_obj in zip(objects, objects_2)])
		self.wait(2)

		row2 = self.put_in_table(objects, left_table, 2, 40, split_length,
			left_right_length, count_data_left)
		self.wait()

		objects_3 = self.show_count(bob_image, 60)
		self.play(*[ReplacementTransform(obj, new_obj) for obj, new_obj in zip(objects_2, objects_3)])

		row3 = self.put_in_table(objects, left_table, 3, 60, split_length,
			left_right_length, count_data_left)

		objects_4 = self.show_count(bob_image, 80)
		self.play(*[ReplacementTransform(obj, new_obj) for obj, new_obj in zip(objects_3, objects_4)])

		row4 = self.put_in_table(objects, left_table, 4, 80, split_length,
			left_right_length, count_data_left)

		objects_5 = self.show_count(bob_image, 100)
		self.play(*[ReplacementTransform(obj, new_obj) for obj, new_obj in zip(objects_4, objects_5)])

		row5 = self.put_in_table(objects, left_table, 5, 100, split_length,
			left_right_length, count_data_left)
		self.wait()

		to_fadeout = objects_5 + [arrow, bob_image]

		alice_image = ImageMobject("alice_solution")
		alice_image.scale(1.4)
		alice_image.move_to(bob_image.get_center())

		self.play(*[FadeOut(obj) for obj in to_fadeout])

		self.play(FadeIn(alice_image))

		alice_count_statement = TextMobject(r"For Alice, we'll count the number of times $c \geq 0$ is checked.")
		alice_count_statement.move_to(answer_most.get_center())
		alice_count_statement[-13:-10].set_color(GREEN_SCREEN)
		alice_count_statement.scale(0.8)
		self.play(ReplacementTransform(answer_most, alice_count_statement))
		arrow.shift(LEFT * 0.3 + DOWN * 0.1)
		self.play(FadeIn(arrow))

		# ALICE table operations

		alice_objects = self.show_count(alice_image, 20, alice=True)
		for i in range(len(alice_objects)):
			obj = alice_objects[i]
			self.play(FadeIn(obj))
			if i == len(alice_objects) - 1:
				self.wait(4)
			elif i == 1:
				self.wait()
			elif i == 2:
				self.wait(6)

		alice_row1 = self.put_in_table(alice_objects, right_table, 1, 20, split_length,
			left_right_length, count_data_right, alice=True)

		alice_objects_2 = self.show_count(alice_image, 40, alice=True)
		self.play(*[ReplacementTransform(obj, new_obj) for obj, new_obj in zip(alice_objects, alice_objects_2)])
		self.wait()

		alice_row2 = self.put_in_table(alice_objects, right_table, 2, 40, split_length,
			left_right_length, count_data_right, alice=True)

		alice_objects_3 = self.show_count(alice_image, 60, alice=True)
		self.play(*[ReplacementTransform(obj, new_obj) for obj, new_obj in zip(alice_objects_2, alice_objects_3)])

		alice_row3 = self.put_in_table(alice_objects, right_table, 3, 60, split_length,
			left_right_length, count_data_right, alice=True)

		alice_objects_4 = self.show_count(alice_image, 80, alice=True)
		self.play(*[ReplacementTransform(obj, new_obj) for obj, new_obj in zip(alice_objects_3, alice_objects_4)])

		alice_row4 = self.put_in_table(alice_objects, right_table, 4, 80, split_length,
			left_right_length, count_data_right, alice=True)

		alice_objects_5 = self.show_count(alice_image, 100, alice=True)
		self.play(*[ReplacementTransform(obj, new_obj) for obj, new_obj in zip(alice_objects_4, alice_objects_5)])

		alice_row5 = self.put_in_table(alice_objects, right_table, 5, 100, split_length,
			left_right_length, count_data_right, alice=True)

		self.play(
			FadeOut(alice_count_statement),
			FadeOut(answer)
		)

		self.wait()

		likes = TextMobject("What do we like about this scheme?")
		likes.scale(0.8)
		likes.move_to(answer.get_center())
		self.play(Write(likes))
		self.wait(3)

		like_answer = TextMobject("Counts will be the same regardless of computer")
		like_answer.scale(0.8)
		like_answer.move_to(alice_count_statement.get_center())
		self.play(Write(like_answer))
		self.wait(3)

		issues_with_count = TextMobject("What are some issues with this scheme?")
		issues_with_count.scale(0.8)
		issues_with_count.move_to(likes.get_center())
		self.play(
			FadeOut(like_answer),
			ReplacementTransform(likes, issues_with_count),
		)

		self.wait(7)

		first_issue = TextMobject("Hard to discern the growth of the algorithm from a table")
		first_issue.scale(0.8)
		first_issue.move_to(alice_count_statement.get_center())
		self.play(Write(first_issue), run_time=3)

		self.wait(4)
		self.play(
			Indicate(row5[0]),
			Indicate(row5[2]),
			run_time=2
		)

		n_1000 = TextMobject("n = 1000?")
		n_1000.scale(0.6)
		n_1000.next_to(left_table, DOWN)
		n_10000 = TextMobject("n = 10000?")
		n_10000.scale(0.6)
		n_10000.next_to(n_1000, DOWN)
		self.wait(3)
		self.play(Write(n_1000))
		self.wait()
		self.play(Write(n_10000))
		self.wait(3)
		self.play(FadeOut(n_1000), FadeOut(n_10000))
		self.wait(4)
		too_long = TextMobject("Another issue with this scheme: takes way too long!")
		too_long.scale(0.8)
		too_long.move_to(first_issue.get_center())
		self.play(ReplacementTransform(first_issue, too_long))
		self.wait(7)

		improvement = TextMobject("Is there a way we can simplify this counting scheme?")
		improvement.scale(0.8)
		improvement.move_to(issues_with_count.get_center())
		self.play(ReplacementTransform(issues_with_count, improvement), FadeOut(too_long))
		self.wait(5)

		left_cross = Cross(left_table, color=BRIGHT_RED)
		right_cross = Cross(right_table, color=BRIGHT_RED)
		self.play(
			ShowCreation(left_cross),
			ShowCreation(right_cross),
			)
		self.wait()
		all_data = list(count_data_left.values()) + list(count_data_right.values())
		data_fadeouts = [FadeOut(data) for data in all_data]
		bob_and_alice_fadeouts = [FadeOut(bob_text), FadeOut(alice_text)]
		fadeouts = [FadeOut(left_cross), FadeOut(right_cross), FadeOut(left_table), FadeOut(right_table)] + data_fadeouts
		self.play(*fadeouts + bob_and_alice_fadeouts)

		self.wait(5)
		generalization = TextMobject(r"Let's try to generalize for any $n$")
		generalization.scale(0.8)
		generalization.move_to(too_long.get_center())
		self.play(Write(generalization))
		self.wait()

		self.wait()
		bob_gen_count = TextMobject("Bob's Count")
		bob_gen_count[:3].set_color(ORANGE)
		bob_gen_count.scale(0.8)
		bob_gen_count.move_to(LEFT * 4.5)

		self.play(FadeOut(alice_image), FadeOut(arrow))
		arrow.shift(RIGHT * 0.25 + UP * 0.05)
		self.play(FadeIn(bob_image), FadeIn(arrow))
		self.wait()

		bob_generalized_objects = self.generalize_count(bob_image, 'n')
		self.play(*[ReplacementTransform(obj, new_obj) for obj, new_obj in 
			zip(alice_objects_5, bob_generalized_objects)])
		self.wait(5)

		bob_gen_expr = TexMobject("(n + 1) ^ 3")
		bob_gen_expr.set_color(ORANGE)
		bob_gen_expr.scale(0.8)
		bob_gen_expr.next_to(bob_gen_count, DOWN)

		a_object_bob, b_object_bob, c_object_bob = bob_generalized_objects[1], bob_generalized_objects[2], bob_generalized_objects[3]
		n_plus_1_group_bob = VGroup(a_object_bob[8:13], b_object_bob[8:13], c_object_bob[8:13])
		self.play(
			Write(bob_gen_count), 
			TransformFromCopy(n_plus_1_group_bob, bob_gen_expr),
			run_time=2,
			)
		self.wait()

		self.play(FadeOut(bob_image), FadeOut(arrow))
		arrow.shift(LEFT * 0.25 + DOWN * 0.05)
		self.play(FadeIn(alice_image), FadeIn(arrow))

		alice_generalized_objects = self.generalize_count(alice_image, 'n', alice=True)
		self.play(*[ReplacementTransform(obj, new_obj) for obj, new_obj 
			in zip(bob_generalized_objects, alice_generalized_objects)])
		self.wait()

		alice_gen_count = TextMobject("Alice's Count")
		alice_gen_count[:5].set_color(GREEN_SCREEN)
		alice_gen_count.scale(0.8)
		alice_gen_count.move_to(RIGHT * 4.5)

		a_object_alice, b_object_alice = alice_generalized_objects[1], alice_generalized_objects[2]

		alice_gen_expr = TexMobject("(n + 1) ^ 2")
		alice_gen_expr.set_color(GREEN_SCREEN)
		alice_gen_expr.scale(0.8)
		alice_gen_expr.next_to(alice_gen_count, DOWN)
		n_plus_1_group = VGroup(a_object_alice[8:13], b_object_alice[8:13])
		self.play(
			Write(alice_gen_count), 
			TransformFromCopy(n_plus_1_group, alice_gen_expr),
			run_time=2,
			)
		self.wait(8)

		

		expanded_bob = TexMobject("n^3 + 3n^2 + 3n + 1")
		expanded_bob.set_color(ORANGE)
		expanded_bob.scale(0.8)
		expanded_bob.move_to(bob_gen_expr.get_center())
		expanded_alice = TexMobject("n^2 + 2n + 1")
		expanded_alice.set_color(GREEN_SCREEN)
		expanded_alice.scale(0.8)
		expanded_alice.move_to(alice_gen_expr.get_center())

		self.play(
			ReplacementTransform(bob_gen_expr, expanded_bob),
			ReplacementTransform(alice_gen_expr, expanded_alice),
		)
		self.wait(3)

		simplify_more = TextMobject("Let's now try to simplify more!")
		simplify_more.scale(0.8)
		simplify_more.move_to(answer.get_center())
		n_very_large = TextMobject(r"What happens when $n$ becomes very large?")
		n_very_large.scale(0.8)
		n_very_large.move_to(generalization.get_center())
		self.play(
			ReplacementTransform(improvement, simplify_more),
			FadeOut(generalization),
		)
		self.wait(11)
		self.play(
			Write(n_very_large)
		)

		self.wait(9)

		left_cross = Cross(expanded_bob[2:])
		right_cross = Cross(expanded_alice[2:])
		self.play(
			ShowCreation(left_cross),
			ShowCreation(right_cross),
		)

		self.wait(9)

		brace_left = Brace(expanded_bob, DOWN, buff=SMALL_BUFF)
		brace_right = Brace(expanded_alice, DOWN, buff=SMALL_BUFF)

		runtime_left_approx = TexMobject(r"\approx n ^ 3")
		runtime_left_approx.set_color(ORANGE)
		runtime_left_approx.scale(0.8)
		runtime_left_approx.next_to(brace_left, DOWN)

		runtime_left = TextMobject(r"O($n^3$)")
		runtime_left.set_color(ORANGE)
		runtime_left.scale(0.8)
		runtime_left.next_to(brace_left, DOWN)

		runtime_right_approx = TexMobject(r"\approx n ^ 2")
		runtime_right_approx.set_color(GREEN_SCREEN)
		runtime_right_approx.scale(0.8)
		runtime_right_approx.next_to(brace_right, DOWN)

		runtime_right = TextMobject(r"O($n^2$)")
		runtime_right.set_color(GREEN_SCREEN)
		runtime_right.scale(0.8)
		runtime_right.next_to(brace_right, DOWN)

		self.play(
			GrowFromCenter(brace_left),
			GrowFromCenter(brace_right),
			FadeIn(runtime_left_approx),
			FadeIn(runtime_right_approx)
		)

		self.wait(8)

		self.play(
			ReplacementTransform(runtime_left_approx, runtime_left),
			ReplacementTransform(runtime_right_approx, runtime_right)
		)

		self.wait(3)

		original_q = TextMobject("So, which algorithm is better?")
		original_q.scale(0.8)
		original_q.move_to(simplify_more.get_center())
		self.play(
			ReplacementTransform(simplify_more, original_q),
			FadeOut(n_very_large),
		)
		self.wait(3)

		answer_original_q = TextMobject(r"Alice's algorithm $\rightarrow$ O($n^2$) is better than O($n^3$)")
		answer_original_q[17:22].set_color(GREEN_SCREEN)
		answer_original_q[-5:].set_color(ORANGE)
		answer_original_q.scale(0.8)
		answer_original_q.move_to(n_very_large.get_center())
		self.play(Write(answer_original_q), run_time=2)
		self.wait(10)
	def generalize_count(self, image, n_str, alice=False):
		n_20 = TexMobject(r"\forall {0}".format(n_str))
		n_20.scale(0.6)
		n_20.next_to(image, UP * 3)
		n_20.shift(LEFT * 1.8)

		a_20 = TextMobject(r"$a \in [0, {0}] \rightarrow$ {1} values".format(n_str, '(' + n_str + ' + 1)'))
		a_20.scale(0.6)
		a_20.next_to(n_20, RIGHT * 1.5)
		a_20.shift(UP * 0.4)

		b_20 = TextMobject(r"$b \in [0, {0}] \rightarrow$ {1} values".format(n_str, '(' + n_str + ' + 1)'))
		b_20.scale(0.6)
		b_20.next_to(a_20, DOWN * 0.5)
	
		if alice:
			c_20 = TextMobject(r"$c = n - (a + b) \rightarrow 1$ value")
		else:
			c_20 = TextMobject(r"$c \in [0, {0}] \rightarrow$ {1} values".format(n_str, '(' + n_str + ' + 1)'))
		c_20.scale(0.6)
		c_20.next_to(b_20, DOWN * 0.5)

		return [n_20, a_20, b_20, c_20]


	def show_count(self, image, n, alice=False):
		n_20 = TexMobject("n = {0}".format(n))
		n_20.scale(0.6)
		n_20.next_to(image, UP * 3)
		n_20.shift(LEFT * 1.8)

		a_20 = TextMobject(r"$a \in [0, {0}] \rightarrow$ {1} values".format(n, n + 1))
		a_20.scale(0.6)
		a_20.next_to(n_20, RIGHT * 1.5)
		a_20.shift(UP * 0.4)

		b_20 = TextMobject(r"$b \in [0, {0}] \rightarrow$ {1} values".format(n, n + 1))
		b_20.scale(0.6)
		b_20.next_to(a_20, DOWN * 0.5)
	
		if alice:
			c_20 = TextMobject(r"$c = n - (a + b) \rightarrow 1$ value")
		else:
			c_20 = TextMobject(r"$c \in [0, {0}] \rightarrow$ {1} values".format(n, n + 1))
		c_20.scale(0.6)
		c_20.next_to(b_20, DOWN * 0.5)

		return [n_20, a_20, b_20, c_20]
	
	def put_in_table(self, objects, table, row, n, split_length, 
		left_right_length, data, alice=False):
		n_val, a_val, b_val, c_val = objects[0], objects[1], objects[2], objects[3]
		n_center = get_center_row_col(table, row, 0, split_length, left_right_length)
		count_center = get_center_row_col(table, row, 1, split_length, left_right_length)
		left_val = data[(row, 0)]
		# left_val.scale(0.7)
		left_val.move_to(n_center)
		right_val_end = data[(row, 1)]
		# right_val_end.scale(0.7)z
		if not alice:
			right_val_start = TexMobject(r"{0} \cdot {0} \cdot {0}".format(n + 1))
			right_val_start.set_color(ORANGE)
		else:
			right_val_start = TexMobject(r"{0} \cdot {0}".format(n + 1))
			right_val_start.set_color(GREEN_SCREEN)
		if n < 100 or alice:
			right_val_start.scale(0.6)
		else:
			right_val_start.scale(0.45)
		right_val_start.move_to(count_center)
		# right_val_start.scale(0.7)
		self.play(
			FadeIn(left_val),
			FadeIn(right_val_start),
			)
		right_val_end.move_to(right_val_start.get_center())
		self.play(
			ReplacementTransform(right_val_start, right_val_end)
			)
		return [left_val, right_val_start, right_val_end]







	def bob(self, x):
		if x <= 7 and x >= 2:
			return 72 * x ** 3 - 622 * x ** 2 + 1745 * x - 1578
		else:
			return 0

	def alice(self, x):
		if x <= 7 and x >= 2:
			return (409 / 12) * x ** 3 - (1201 / 4) * x ** 2 + (2564 / 3) * x - 781
		else:
			return 0

	def bob2(self, x):
		if x <= 7 and x >= 2:
			return 761 / 12 * x ** 3 - 2193 / 4 * x ** 2 + 4615 / 3 * x - 1391
		else:
			return 0

	def alice2(self, x):
		if x <= 7 and x >= 2:
			return (151 / 4) * x ** 3 - (1297 / 4) * x ** 2 + 906 * x - 817
		else:
			return 0

	def transform_text_to_table(self, n, n_val, time, table, n_center, 
		n_val_center, time_center, other_anims, time_label_center, scale=0.7, color=ORANGE):
		n_copy = n.copy()
		n_val_copy = n_val.copy()
		time_val_copy = time.copy()
		n_copy.move_to(n_center)
		n_val_copy.move_to(n_val_center)
		time_val_copy = time.copy()
		time_val_copy.move_to(time_center)
		time_label = TextMobject("Time (s)")
		time_label.set_color(color)
		time_label.scale(scale)
		time_label.move_to(time_label_center)
		new_anims = [
		ShowCreation(table), 
		Transform(n, n_copy), 
		ReplacementTransform(n_val, n_val_copy), 
		Transform(time, time_val_copy),
		Write(time_label)
		]
		anims = other_anims + new_anims
		return anims, time_label, n_val_copy

class Story(Scene):
	def construct(self):
		story = TextMobject("And that's the story!")
		story[-6:-1].set_color(GREEN_SCREEN)

		rect_left = ScreenRectangle()
		rect_left.set_width(6)
		rect_left.set_height(5)
		rect_left.move_to(ORIGIN)

		story.next_to(rect_left, UP)

		self.play(Write(story), ShowCreation(rect_left), run_time=2)
		
		step1 = TextMobject("1. What if we tried timing algorithms?")
		step1.next_to(rect_left, DOWN * 1.5)
		self.play(Write(step1))
		self.wait(3)
		
		step2 = TextMobject("2. What if we tried counting operations?")
		step2.move_to(step1.get_center())
		self.play(ReplacementTransform(step1, step2))
		self.wait(3)

		step3 = TextMobject("3. What if we tried to generalize this count?")
		step3.move_to(step2.get_center())
		self.play(ReplacementTransform(step2, step3))
		self.wait(4)

		step4 = TextMobject("4. What if we got rid of all the complicated terms?")
		step4.move_to(step3.get_center())
		self.play(ReplacementTransform(step3, step4))
		self.wait(4)


def make_2_column_table(center, rows, length, width):
	n_v_lines = 1
	n_h_lines = rows - 1

	vertical_lines = []
	for i in range(n_v_lines):
		line = Line(ORIGIN, DOWN * width)
		line.move_to(center)
		vertical_lines.append(line)

	horizontal_lines = []
	for i in range(n_h_lines):
		split_length = width / rows
		v_line = vertical_lines[0]
		line_center = v_line.get_start() + DOWN * split_length * (i + 1)
		line = Line(ORIGIN, RIGHT * length)
		line.move_to(line_center)
		horizontal_lines.append(line)


	table = horizontal_lines + vertical_lines
	table_group = VGroup(*table)
	return table_group

def get_center_row_col(table, row, col, split_length, left_right_length):
	direction = UP
	if row >= len(table) - 1:
		row = len(table) - 2
		direction = DOWN
	line = table[row]
	if col == 0:
		center = line.get_midpoint() + direction * split_length / 2 + LEFT * left_right_length / 2
	else:
		center = line.get_midpoint() + direction * split_length / 2 + RIGHT * left_right_length / 2
	return center


class IntroduceBigO(Scene):
	def construct(self):
		right_character = create_computer_char(color=RED, scale=0.7, position= DOWN * 3)
		right_thought_bubble = SVGMobject("thought")
		right_thought_bubble.scale(1.2)
		right_thought_bubble.move_to(right_character.get_center() + UP * 2 + RIGHT * 2)
		simplify = TextMobject("But ... how?")
		simplify.scale(0.5)
		simplify.move_to(right_thought_bubble.get_center() + UP * 0.4 + RIGHT * 0.04)
		simplify.set_color(BLACK)

		top_text = self.make_text("What is Big O Notation?", 
			UP * 3.5, scale=1)
		efficiency = self.make_text("It gives you the efficiency of an algorithm!", UP * 2.5)

		growth = self.make_text("It tells you the growth of an algorithm!", UP * 1.8)
		definition = self.make_text(r"$f = O(g)$ if there is a constant $c > 0$ such that $f(n) \leq c \cdot g(n)$",
			UP * 1.1, scale=0.7)

		self.play(Write(top_text), run_time=2)
		self.wait(6)
		self.play(GrowFromCenter(efficiency))
		self.wait()
		self.play(FadeIn(right_character))
		self.play(FadeIn(right_thought_bubble), FadeIn(simplify))
		self.wait(3)

		self.play(GrowFromCenter(growth))
		self.wait()
		new_simplify = TextMobject("Specifics please!")
		new_simplify.scale(0.5)
		new_simplify.move_to(right_thought_bubble.get_center() + UP * 0.4 + RIGHT * 0.04)
		new_simplify.set_color(BLACK)
		self.play(FadeOut(simplify))
		self.play(Write(new_simplify))
		self.wait(2)
		self.play(GrowFromCenter(definition))
		self.wait()

		new_simplify2 = TextMobject("English please!")
		new_simplify2.scale(0.5)
		new_simplify2.move_to(right_thought_bubble.get_center() + UP * 0.4 + RIGHT * 0.04)
		new_simplify2.set_color(BLACK)
		self.play(FadeOut(new_simplify))
		self.play(Write(new_simplify2))
		self.wait(11)

		text_group = VGroup(definition, efficiency, growth)
		cross = Cross(text_group)
		self.play(ShowCreation(cross))
		self.wait()

		fadeouts = [
		FadeOut(text_group),
		FadeOut(cross),
		FadeOut(right_character),
		FadeOut(new_simplify2),
		FadeOut(right_thought_bubble),
		]

		new_big_o = TextMobject("What is the story behind Big O Notation?")
		new_big_o.move_to(top_text.get_center())
		new_big_o[9:14].set_color(GREEN_SCREEN)

		rect_left = ScreenRectangle()
		rect_left.set_width(6)
		rect_left.set_height(5)
		rect_left.move_to(ORIGIN)

		self.play(
			*fadeouts,
			ReplacementTransform(top_text, new_big_o))
		self.wait()

		self.play(FadeIn(rect_left))

		step1 = TextMobject("1. Understand the story")
		step1.next_to(rect_left, DOWN * 1.5)
		step1[-5:].set_color(GREEN_SCREEN)
		self.play(Write(step1))
		self.wait(2)
		
		step2 = TextMobject("2. Decode the convoluted defintion")
		step2.move_to(step1.get_center())
		self.play(ReplacementTransform(step1, step2))
		self.wait(2)

		step3 = TextMobject("3. Define steps to find Big O")
		step3.move_to(step2.get_center())
		self.play(ReplacementTransform(step2, step3))
		self.wait(4)

		step4 = TextMobject("4. Apply steps to an example")
		step4.move_to(step3.get_center())
		self.play(ReplacementTransform(step3, step4))
		self.wait(4)


	def make_text(self, text, position, scale=0.8):
		text = TextMobject(text)
		text.scale(scale)
		text.move_to(position)
		return text

class Definition(GraphScene):
	CONFIG = {
	"x_min" : 0,
	"x_max" : 10,
	"y_min" : 0,
	"y_max" : 400,
	"graph_origin" : LEFT * 2.5 + DOWN * 3.5,
	"function_color" : RED ,
	"axes_color" : GREEN, 
	"x_tick_frequency": 100,
	"y_tick_frequency": 1000,
	"y_axis_height": 3.5,
	"x_axis_width": 5,
	"x_axis_label": None,
	"y_axis_label": None,
	}
	def construct(self):
		self.wait(2)
		confusion = self.make_text("Confusing Aspects of Big O Notation", UP * 3.5, scale=1)
		definition = self.make_text("Big O Definition", UP * 3.5, scale=1)
		definition_line1 = self.make_text(r"Let $f(n)$ and $g(n)$ be functions from positive integers to positive reals.",
			UP * 2.8, scale=0.7)
		definition_line2 = self.make_text(r"$f = O(g)$ if there is a constant $c > 0$ such that $f(n) \leq c \cdot g(n)$ for large $n$",
			UP * 2.3, scale=0.7)

		self.play(Write(confusion))
		self.wait(11)
		self.play(ReplacementTransform(confusion, definition), run_time=2)
		self.wait(2)
		self.play(Write(definition_line1), run_time=3)
		self.wait(3)
		self.play(Write(definition_line2), run_time=3)
		self.wait(13)
		confused_emoji = ImageMobject("confused_emoji")
		confused_emoji.scale(1.2)
		self.play(GrowFromCenter(confused_emoji))
		self.wait(8)
		self.play(FadeOut(confused_emoji))
		self.wait(2)



		grows_no_faster = self.make_text(r'$f(n)$ "grows no faster" than $g(n)$',
			UP * 1.3, scale=0.7)
		brace = Brace(definition_line2, DOWN, buff=SMALL_BUFF)
		self.play(
			GrowFromCenter(brace),
			Write(grows_no_faster),
		)

		self.wait(8)

		example = self.make_text(r"Example: $f(n) = 3n^2 + 5n + 4 \rightarrow O(n^2) \rightarrow g(n) = n^2$",
			UP * 0.8, scale=0.7)
		example[13:21].set_color(RED)
		example[24:26].set_color(BLUE)
		example[-2:].set_color(BLUE)
		self.play(Write(example[:21]))
		self.wait(6)
		self.play(Write(example[21:27]))
		self.wait(6)
		self.play(Write(example[27:]))
		self.wait(14)
		self.play(Indicate(definition_line2[-20:-9], scale_factor=1.1), run_time=2)
		self.wait(3)
		self.setup_axes(animate=True)

		f_n = self.get_graph(self.f, color=RED)
		g_n = self.get_graph(self.g, color=BLUE)

		graph_lab_f_n = self.get_graph_label(f_n, label = "f(n)")
		graph_lab_g_n =self.get_graph_label(g_n, label = "g(n)")

		self.play(ShowCreation(f_n))
		self.play(Write(graph_lab_f_n))
		self.play(ShowCreation(g_n))
		self.play(Write(graph_lab_g_n))
		self.wait(3)

		g_n_3 = self.get_graph(self.c_3_g, color=BLUE)
		g_n_3_label = self.get_graph_label(g_n_3, label=r"3 \cdot g(n)")
		self.play(
			ReplacementTransform(g_n, g_n_3),
			ReplacementTransform(graph_lab_g_n, g_n_3_label)
		)
		self.wait(4)

		g_n_5 = self.get_graph(self.c_5_g, color=BLUE)
		g_n_5_label = self.get_graph_label(g_n_5, label=r"4 \cdot g(n)")
		self.play(
			ReplacementTransform(g_n_3, g_n_5),
			ReplacementTransform(g_n_3_label, g_n_5_label)
		)

		self.wait(7)

		self.play(
			FadeOut(g_n_5_label),
			FadeOut(g_n_5),
		)
		self.wait()
		why_cant = self.make_text(r"Here's why $g(n) \neq n$", LEFT * 5 + DOWN, scale=0.7)
		why_cant[-7:].set_color(YELLOW)
		self.play(FadeIn(why_cant))

		linear_g_n = self.get_graph(self.linear_g, color=YELLOW)
		linear_g_n_label = self.get_graph_label(linear_g_n, label=r"g(n)")
		self.play(ShowCreation(linear_g_n))
		self.play(Write(linear_g_n_label))
		self.wait()

		linear_g_n_5 = self.get_graph(self.linear_g_5, color=YELLOW)
		linear_g_n_label_5 = self.get_graph_label(linear_g_n_5, label=r"5 \cdot g(n)")
		self.play(
			ReplacementTransform(linear_g_n, linear_g_n_5),
			ReplacementTransform(linear_g_n_label, linear_g_n_label_5),
		)

		self.wait()

		linear_g_n_20 = self.get_graph(self.linear_g_20, color=YELLOW)
		linear_g_n_label_20 = self.get_graph_label(linear_g_n_20, label=r"15 \cdot g(n)")
		self.play(
			ReplacementTransform(linear_g_n_5, linear_g_n_20),
			ReplacementTransform(linear_g_n_label_5, linear_g_n_label_20),
		)

		self.wait()

		linear_g_n_30 = self.get_graph(self.linear_g_30, color=YELLOW)
		linear_g_n_label_30 = self.get_graph_label(linear_g_n_30, label=r"30 \cdot g(n)")
		self.play(
			ReplacementTransform(linear_g_n_20, linear_g_n_30),
			ReplacementTransform(linear_g_n_label_20, linear_g_n_label_30),
		)

		self.wait(11)

		rectangle = Rectangle(height=2.3, width=4)
		rectangle.shift(LEFT * 0.4 + DOWN * 2.5)
		rectangle.set_color(GREEN_SCREEN)
		self.play(ShowCreation(rectangle), run_time=1)
		self.wait(3)

		red_rectangle = Rectangle(height=1, width=0.9)
		red_rectangle.move_to(RIGHT * 2.1 + DOWN * 0.9)
		red_rectangle.set_color(BRIGHT_RED)


		large_n_surround = SurroundingRectangle(definition_line2[-6:])
		large_n_surround.set_color(BRIGHT_RED)

		self.play(
			ReplacementTransform(rectangle, red_rectangle),
			ShowCreation(large_n_surround),
			run_time=2,
		)

		self.wait(2)

		objects_to_fade_out = [linear_g_n_30, linear_g_n_label_30, self.x_axis, self.y_axis,
		graph_lab_f_n, f_n, why_cant, red_rectangle, large_n_surround]
		self.play(*[FadeOut(obj) for obj in objects_to_fade_out])

		self.wait()

		compact_def = self.make_text("A More Compact Defintiion", DOWN, scale=1)
		self.play(Write(compact_def))

		self.wait(5)

		limit_def = self.make_text(r"$f = O(g)$ if $\lim\limits_{n \to \infty} \frac{g(n)}{f(n)} = C, C > 0$", DOWN * 2)
		self.play(Write(limit_def, run_time=2))

		example_one = self.make_text(r"$\lim\limits_{n \to \infty} \frac{n^2}{3n^2 + 5n + 4} = \frac{1}{3}$", DOWN * 3 + LEFT * 2)
		example_two = self.make_text(r"$\lim\limits_{n \to \infty} \frac{n}{3n^2 + 5n + 4} = 0$", DOWN * 3 + RIGHT * 2)
		self.wait(8)
		self.play(Write(example_one))
		self.wait()
		self.play(Write(example_two))
		self.wait()

		cross = Cross(example_two)
		self.play(ShowCreation(cross))
		self.wait(3)


	def f(self, x):
		return 3 * x ** 2 + 5 * x + 4

	def g(self, x):
		return x ** 2

	def c_3_g(self, x):
		return 3 * x ** 2

	def c_5_g(self, x):
		return 4 * x ** 2

	def linear_g(self, x):
		return x

	def linear_g_5(self, x):
		return 5 * x

	def linear_g_20(self, x):
		return 15 * x

	def linear_g_30(self, x):
		return 30 * x



	def make_text(self, text, position, scale=0.8):
		text = TextMobject(text)
		text.scale(scale)
		text.move_to(position)
		return text

class RunningTimeClasses(Scene):
	def construct(self):
		self.wait(2)
		title = self.make_text("Classes of Running Times", UP * 3, scale=1.2)
		h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
		h_line.next_to(title, DOWN)
		self.play(
			Write(title),
			ShowCreation(h_line),
			run_time=2
		)

		scale = 1

		constant = self.make_text("Constant", UP * 2 + LEFT, scale=scale)
		constant.set_color(GREEN_SCREEN)
		logarithmic = self.make_text("Logarithmic", UP + LEFT, scale=scale)
		logarithmic.set_color(GREEN)
		linear = self.make_text("Linear", LEFT, scale=scale)
		linear.set_color(YELLOW)
		linearithmic = self.make_text("Linearithmic", DOWN + LEFT, scale=scale)
		linearithmic.set_color(ORANGE)
		polynomial = self.make_text("Polynomial", DOWN * 2 + LEFT * 3, scale=scale)
		polynomial.set_color(RED)
		exponential = self.make_text("Exponential", DOWN * 3 + LEFT * 3, scale=scale)
		exponential.set_color(BRIGHT_RED)

		const_ex = TextMobject(r"- $O(1)$")
		const_ex.set_color(GREEN_SCREEN)
		const_ex.next_to(constant, RIGHT)
		
		self.wait(14)
		self.play(
			FadeIn(constant),
			FadeIn(const_ex),
		)

		logarithmic_ex = TextMobject(r"- $O(\log N)$")
		logarithmic_ex.set_color(GREEN)
		logarithmic_ex.next_to(logarithmic, RIGHT)

		self.play(
			FadeIn(logarithmic),
			FadeIn(logarithmic_ex)
		)


		linear_ex = TextMobject(r"- $O(N)$")
		linear_ex.set_color(YELLOW)
		linear_ex.next_to(linear, RIGHT)

		self.play(
			FadeIn(linear),
			FadeIn(linear_ex)
		)

		linearithmic_ex = TextMobject(r"- $O(N \log N)$")
		linearithmic_ex.set_color(ORANGE)
		linearithmic_ex.next_to(linearithmic, RIGHT)

		self.play(
			FadeIn(linearithmic),
			FadeIn(linearithmic_ex)
		)

		polynomial_ex = TextMobject(r"- $O(N^2), O(N^3), O(N^4)$, etc.")
		polynomial_ex.set_color(RED)
		polynomial_ex.next_to(polynomial, RIGHT)

		self.play(
			FadeIn(polynomial),
			FadeIn(polynomial_ex)
		)

		exponential_ex = TextMobject(r"- $O(2^N), O(3^N), O(4^N)$, etc.")
		exponential_ex.set_color(BRIGHT_RED)
		exponential_ex.next_to(exponential, RIGHT)

		self.play(
			FadeIn(exponential),
			FadeIn(exponential_ex)
		)

		self.wait(2)
		line_before_arrow = Line(LEFT * 5 + UP * 2, LEFT * 5 + DOWN * 3)
		line_before_arrow.set_color(color=[GREEN_SCREEN, GREEN, YELLOW, ORANGE, RED, BRIGHT_RED][::-1])
	
		arrow_head = RegularPolygon(n=3, color=BRIGHT_RED, fill_color=BRIGHT_RED, fill_opacity=1)
		arrow_head.scale(0.2)
		arrow_head.rotate(-TAU / 4)
		arrow_head.move_to(LEFT * 5 + DOWN * 3)
		self.play(FadeIn(line_before_arrow), FadeIn(arrow_head))

		self.wait()

		smiley_face = self.make_face()
		smiley_face.move_to(LEFT * 5.8 + UP * 1.5)
		smiley_face.set_color(GREEN_SCREEN)
		smiley_face.scale(0.8)
		self.play(FadeIn(smiley_face))

		sad_face = self.make_face(smile=False)
		sad_face.move_to(LEFT * 5.8 + DOWN * 2.5)
		sad_face.set_color(BRIGHT_RED)
		sad_face.scale(0.8)
		self.play(FadeIn(sad_face))

		self.wait(4)


		

	def make_face(self, smile=True):
		circle = Circle(radius=0.5)
		left_eye = Dot(radius=0.05)
		left_eye.shift(UP * 0.1 + LEFT * 0.2)
		right_eye = Dot(radius=0.05)
		right_eye.shift(UP * 0.1 + RIGHT * 0.2)
		if smile:
			smile = Arc(-TAU / 2, radius=0.2)
			smile.shift(DOWN * 0.1)
		else:
			smile = Arc(TAU / 2, radius=0.2)
			smile.shift(DOWN * 0.3)

		smiley_face = VGroup(*[circle, left_eye, right_eye, smile])
		return smiley_face

	
	def make_text(self, text, position, scale=0.8):
		text = TextMobject(text)
		text.scale(scale)
		text.move_to(position)
		return text


class BigOSteps(Scene):
	def construct(self):
		self.wait()
		question = "How do you identify the appropriate running time for an algorithm?"
		question = self.make_text(question, UP * 3, scale=0.9)

		step1 = "1. Understand how the algorithm works"
		step1 = self.make_text(step1, UP * 1)
		step1.set_color(BRIGHT_RED)
		step1_spec1 = "- What is the purpose of the algorithm?"
		step1_spec1 = self.make_text(step1_spec1, UP * 0.5, scale=0.7)
		step1_spec2 = "- What are the input(s)?"
		step1_spec2 = self.make_text(step1_spec2, UP * 0, scale=0.7)
		step1_spec3 = "- What are the output(s)?"
		step1_spec3 = self.make_text(step1_spec3, DOWN * 0.5, scale=0.7)

		self.play(Write(question))
		self.wait(6)

		self.play(Write(step1))
		self.wait(2)

		self.play(Write(step1_spec1))
		self.wait(2)

		self.play(Write(step1_spec2))
		self.wait(2)

		self.play(Write(step1_spec3))
		self.wait(2)

		step1_copy = step1.copy()
		step1_copy.next_to(question, DOWN)
		self.play(
			Transform(step1, step1_copy),
			FadeOut(step1_spec1),
			FadeOut(step1_spec2),
			FadeOut(step1_spec3),
			run_time=2,
		)

		step2 = "2. Identify a basic unit of the algorithm to count"
		step2 = self.make_text(step2, UP * 1)
		step2.set_color(GREEN_SCREEN)
		step2_spec1 = "- Print Statements"
		step2_spec1 = self.make_text(step2_spec1, UP * 0.5, scale=0.7)
		step2_spec2 = "- Iterations/Assignment Statements"
		step2_spec2 = self.make_text(step2_spec2, UP * 0, scale=0.7)
		step2_spec3 = "- Recursive Calls"
		step2_spec3 = self.make_text(step2_spec3, DOWN * 0.5, scale=0.7)
		step2_spec4 = "- Focus on the worst case"
		step2_spec4 = self.make_text(step2_spec4, DOWN, scale=0.7)

		self.play(Write(step2))
		self.wait()

		self.play(Write(step2_spec1))
		self.wait()

		self.play(Write(step2_spec2))
		self.wait()

		self.play(Write(step2_spec3))
		self.wait(3)

		self.play(Write(step2_spec4))
		self.wait(8)

		step2_copy = step2.copy()
		step2_copy.next_to(step1, DOWN)
		self.play(
			Transform(step2, step2_copy),
			FadeOut(step2_spec1),
			FadeOut(step2_spec2),
			FadeOut(step2_spec3),
			FadeOut(step2_spec4),
			run_time=2,
		)

		step3 = "3. Map growth of count from step 2 to appropriate Big O class"
		step3 = self.make_text(step3, UP * 1)
		step3.set_color(BLUE)
		step3_spec1 = "- Is the growth constant?"
		step3_spec1 = self.make_text(step3_spec1, UP * 0.5, scale=0.7)
		step3_spec2 = "- Is the growth exponential?"
		step3_spec2 = self.make_text(step3_spec2, UP * 0, scale=0.7)
		step3_spec3 = "- Is the growth linear?"
		step3_spec3 = self.make_text(step3_spec3, DOWN * 0.5, scale=0.7)
		step3_spec4 = "- Is the growth logarithmic?"
		step3_spec4 = self.make_text(step3_spec4, DOWN, scale=0.7)

		self.play(Write(step3))

		self.play(Write(step3_spec1))

		self.play(Write(step3_spec2))

		self.play(Write(step3_spec3))

		self.play(Write(step3_spec4))

		step3_copy = step3.copy()
		step3_copy.next_to(step2, DOWN)
		self.play(
			Transform(step3, step3_copy),
			FadeOut(step3_spec1),
			FadeOut(step3_spec2),
			FadeOut(step3_spec3),
			FadeOut(step3_spec4),
			run_time=2,
		)

		self.wait(2)


	def make_text(self, text, position, scale=0.8):
		text = TextMobject(text)
		text.scale(scale)
		text.move_to(position)
		return text

class Ex1(Scene):
	def construct(self):
		self.wait(2)
		ex1 = ImageMobject("ex1")
		ex1.shift(UP * 2.5)
		self.play(FadeIn(ex1))
		self.wait()

		self.step1()

		self.step2(ex1)

	def step1(self):
		step1 = "1. Understand how the algorithm works"
		step1 = self.make_text(step1, UP * 1)
		step1.set_color(BRIGHT_RED)
		step1_spec1 = r"- Base case $n = 1$"
		step1_spec1 = self.make_text(step1_spec1, UP * 0.5, scale=0.7)
		step1_spec2 = r"- Sum of two recursive calls to $f(n - 1)$"
		step1_spec2 = self.make_text(step1_spec2, UP * 0, scale=0.7)

		self.play(Write(step1))
		self.wait(2)
		self.play(Write(step1_spec1))
		self.wait(2)
		self.play(Write(step1_spec2))
		self.wait(4)

		self.play(
			FadeOut(step1),
			FadeOut(step1_spec1),
			FadeOut(step1_spec2),
		)

		self.wait()

	def step2(self, image):
		step2 = "2. Identify a basic unit of the algorithm to count"
		step2 = self.make_text(step2, UP * 1)
		step2.set_color(GREEN_SCREEN)
		step2_spec1 = "- It seems reasonable to count recursive calls here"
		step2_spec1 = self.make_text(step2_spec1, UP * 0.5, scale=0.7)

		self.play(Write(step2))
		self.wait()
		self.play(Write(step2_spec1))
		self.wait(4)
		self.play(
			FadeOut(step2),
			FadeOut(step2_spec1),
		)

		image_copy = image.copy()
		image_copy.shift(LEFT * 3)
		self.play(Transform(image, image_copy))


		rows, length, width = 6, 3, 4
		right_table = make_2_column_table(RIGHT * 5 + UP * 1.5, rows, length, width)
		self.play(ShowCreation(right_table))
		split_length = width / rows
		left_right_length = length / 2
		n_center = get_center_row_col(right_table, 0, 0, split_length, left_right_length)
		count_center = get_center_row_col(right_table, 0, 1, split_length, left_right_length)
		
		n = self.make_text(r"$n$", n_center)
		count = self.make_text("Count", count_center)
		self.play(
			Write(n),
			Write(count),
		)
		# n = 1
		n_1_center = get_center_row_col(right_table, 1, 0, split_length, left_right_length)
		n_1 = self.make_text(r"$1$", n_1_center)
		self.play(FadeIn(n_1))
		tree_1 = VGroup(*make_tree(1, direction=3))
		tree_1.scale(0.8)
		self.play(FadeIn(tree_1))
		self.wait()
		count_1_center = get_center_row_col(right_table, 1, 1, split_length, left_right_length)
		count_1 = self.make_text(r"1", count_1_center)
		self.play(FadeIn(count_1))
		self.wait()
		self.play(FadeOut(tree_1))

		# n = 2
		n_2_center = get_center_row_col(right_table, 2, 0, split_length, left_right_length)
		n_2 = self.make_text(r"$2$", n_2_center)
		self.play(FadeIn(n_2))
		tree_2 = VGroup(*make_tree(2, direction=1.5))
		tree_2.scale(0.8)
		for i in range(len(tree_2)):
			if i % 2 == 0:
				self.play(ShowCreation(tree_2[i]))
			else:
				self.play(Write(tree_2[i]))
		
		self.wait()

		count_3_center = get_center_row_col(right_table, 2, 1, split_length, left_right_length)
		count_3 = self.make_text(r"3", count_3_center)
		self.play(FadeIn(count_3))
		self.wait()
		self.play(FadeOut(tree_2))

		# n = 3
		n_3_center = get_center_row_col(right_table, 3, 0, split_length, left_right_length)
		n_3 = self.make_text(r"$3$", n_3_center)
		self.play(FadeIn(n_3))
		tree_3 = VGroup(*make_tree(3, direction=1.5))
		tree_3.scale(0.8)
		self.wait()
		for i in range(len(tree_3)):
			if i % 2 == 0:
				self.play(ShowCreation(tree_3[i]))
			else:
				self.play(Write(tree_3[i]))
				self.wait()
		
		self.wait(4)

		count_7_center = get_center_row_col(right_table, 3, 1, split_length, left_right_length)
		count_7 = self.make_text(r"7", count_7_center)
		self.play(FadeIn(count_7))
		self.wait()
		# self.play(FadeOut(tree_2))

		left_branch = tree_3[2:7]
		right_branch = tree_3[8:]
		left_rect = SurroundingRectangle(left_branch)
		right_rect = SurroundingRectangle(right_branch)
		table_rect = SurroundingRectangle(VGroup(n_2, count_3))
		self.play(
			ShowCreation(left_rect),
			ShowCreation(right_rect),
			ShowCreation(table_rect),
			)
		self.wait(6)

		self.play(
			FadeOut(left_rect),
			FadeOut(right_rect),
			FadeOut(table_rect),
		)

		# n = 4
		n_4_center = get_center_row_col(right_table, 4, 0, split_length, left_right_length)
		n_4 = self.make_text(r"$4$", n_4_center)
		self.play(FadeIn(n_4))

		tree_4 = VGroup(*make_tree(4, direction=3))
		tree_4.scale(0.8)

		left_branch_3 = tree_4[2:15]
		self.play(ReplacementTransform(tree_3, left_branch_3), run_time=2)
		self.play(Write(tree_4[0]))
		self.play(ShowCreation(tree_4[1]))
		self.play(ShowCreation(tree_4[15]))
		right_branch_3 = tree_4[16:]
		self.play(TransformFromCopy(left_branch_3, right_branch_3), run_time=2)
		self.wait(12)
		self.play(Indicate(left_branch_3), Indicate(right_branch_3), run_time=2)
		self.wait()
		self.play(Indicate(tree_4[0], color=GREEN_SCREEN), run_time=2)


		count_15_center = get_center_row_col(right_table, 4, 1, split_length, left_right_length)
		count_15 = self.make_text(r"15", count_15_center)
		self.play(FadeIn(count_15))
		self.wait()

		tree_4 = VGroup(
			tree_4[0],
			tree_4[1],
			tree_4[15],
			left_branch_3,
			right_branch_3
		)

		# n = 5
		n_5_center = get_center_row_col(right_table, 5, 0, split_length, left_right_length)
		n_5 = self.make_text(r"$5$", n_5_center)
		self.play(FadeIn(n_5))

		tree_5 = VGroup(*make_tree(5, direction=6))
		tree_5.scale(0.6)

		left_branch_4 = VGroup(*tree_5[2:31])
		tree_4_copy = tree_4.copy()
		tree_4_copy.scale(0.6 / 0.8)
		tree_4_new_center = left_branch_4.get_center()
		tree_4_copy.move_to(tree_4_new_center)

		self.play(Transform(tree_4, tree_4_copy), run_time=2)
		self.play(Write(tree_5[0]))
		self.play(ShowCreation(tree_5[1]))
		self.play(ShowCreation(tree_5[31]))
		right_branch_4 = tree_5[32:]
		right_branch_copy = tree_4.copy()
		right_branch_copy.move_to(right_branch_4.get_center())
		self.play(TransformFromCopy(tree_4, right_branch_copy), run_time=2)
		self.wait(9)

		count_31_center = get_center_row_col(right_table, 5, 1, split_length, left_right_length)
		count_31 = self.make_text(r"31", count_31_center)
		self.play(FadeIn(count_31))
		self.wait()

		tree_5 = VGroup(
			tree_5[0],
			tree_5[1],
			tree_5[31],
			tree_4,
			right_branch_copy
		)

		self.play(FadeOut(tree_5))
		self.wait(3)

		step3 = "3. Map growth of count from step 2 to appropriate Big O class"
		step3 = self.make_text(step3, DOWN)
		step3.set_color(BLUE)
		self.play(Write(step3), run_time=2)

		self.wait(6)

		observation = r"- As $n$ increases by $1$, the count doubles roughly by $2$"
		observation = self.make_text(observation, DOWN * 1.5, scale=0.7)
		conclusion = r" - Running time is exponential: $O(2^n)$"
		conclusion = self.make_text(conclusion, DOWN * 2, scale=0.7)
		self.play(Write(observation))
		self.wait(6)
		self.play(Write(conclusion))
		self.wait(8)

		math_obs = r"If $n = k$, Count = $2^k - 1 \rightarrow O(2^n)$"
		math_obs = self.make_text(math_obs, DOWN * 3, scale = 0.7)
		self.play(Write(math_obs[:-6]), run_time=2)
		self.wait(5)
		self.play(Write(math_obs[-6:]))
		self.wait()

		O_2n = math_obs[-5:].copy()
		O_2n.scale(1.4)
		O_2n.next_to(image, DOWN)

		self.play(
			TransformFromCopy(math_obs[-5:], O_2n),
			FadeOut(math_obs),
			FadeOut(observation),
			FadeOut(conclusion),
			FadeOut(step3),
			run_time=2
		)

		takeaways = self.make_text("Big Takeaways", DOWN * 1.5, scale=1)
		h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
		h_line.next_to(takeaways, DOWN)
		self.play(
			Write(takeaways), 
			ShowCreation(h_line)
		)

		diagrams = self.make_text("- Draw Diagrams", DOWN * 2.5, scale=0.7)
		pattern = self.make_text("- Helps with counts, patterns, etc.", DOWN * 3, scale=0.7)
		self.wait(3)
		self.play(Write(diagrams))
		self.wait(4)
		self.play(Write(pattern))
		self.wait(9)





	def make_text(self, text, position, scale=0.8):
		text = TextMobject(text)
		text.scale(scale)
		text.move_to(position)
		return text

def make_tree(n, shift=ORIGIN, parent=None, direction=1.5):
	if n == 1 and parent is None:
		return [TextMobject(r"$f({0})$".format(n))]
	elif n == 1:
		leaf = TextMobject(r"$f({0})$".format(n))
		leaf.move_to(parent.get_center() + shift)
		edge = make_edge(parent, leaf)
		return [edge, leaf]
	root = TextMobject(r"$f({0})$".format(n))
	edge = None
	if parent:
		root.move_to(parent.get_center() + shift)
		edge = make_edge(parent, root)
	left_branch = make_tree(n - 1, shift=direction * LEFT + DOWN, parent=root, direction=direction / 2)
	right_branch = make_tree(n - 1, shift=direction * RIGHT + DOWN, parent=root, direction=direction / 2)
	if edge:
		return [edge, root] + left_branch + right_branch
	else:
		return [root] + left_branch + right_branch

def make_edge(parent, current):
	parent_c = parent.get_center()
	current_c = current.get_center()
	line_through = Line(parent_c, current_c)
	new_parent_c = parent_c + line_through.get_unit_vector() * 0.5 
	new_current_c = current_c - line_through.get_unit_vector() * 0.5
	return Line(new_parent_c, new_current_c)

class Conclusion(Scene):
	def construct(self):
		title = TextMobject("Recap")
		h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
		title.move_to(UP * 3.5)
		h_line.next_to(title, DOWN)
		self.play(
			Write(title),
			ShowCreation(h_line)
		)

		rect_left = ScreenRectangle()
		rect_left.set_width(6)
		rect_left.set_height(5)
		rect_left.move_to(ORIGIN)

		self.play(FadeIn(rect_left))

		self.wait()

		first = TextMobject("We first uncovered the story behind Big O")
		first.scale(0.8)
		first.move_to(DOWN * 3)
		self.play(Write(first), run_time=2)
		self.wait(10)
		second = TextMobject("We then demystified the definition")
		second.scale(0.8)
		second.move_to(DOWN * 3)
		self.wait(5)
		self.play(ReplacementTransform(first, second), run_time=2)
		third = TextMobject("We defined a series of steps to find Big O")
		third.scale(0.8)
		third.move_to(DOWN * 3)
		self.wait(5)
		self.play(ReplacementTransform(second, third), run_time=2)
		fourth = TextMobject("We went through a specific example of these steps in action")
		fourth.scale(0.8)
		fourth.move_to(DOWN * 3)
		self.wait(5)
		self.play(ReplacementTransform(third, fourth), run_time=2)
		self.wait(5)

class Logo(Scene):
	def construct(self):
		# r = 0.8
		# colors = [BLUE, DARK_BLUE]
		# circles = []
		# for i in range(30):
		# 	circle_radius = 3 * r ** i
		# 	color = colors[i % len(colors)]
		# 	circle = Circle(radius=circle_radius, color=color, fill_color=color, fill_opacity=1)
		# 	circles.append(circle)
		# self.play(*[FadeIn(circle) for circle in circles])
		# self.wait()
		hexagon = RegularPolygon(n=6, color=VIOLET, fill_color=VIOLET, fill_opacity=1, sheen_factor=0.2)
		hexagon.scale(2)
		hexagon.rotate(TAU / 4)
		vertices = hexagon.get_vertices()
		print(np.linalg.norm(vertices[0] - vertices[1]))
		print(np.linalg.norm(vertices[1] - vertices[2]))


		triangle = RegularPolygon(color=LIGHT_VIOLET, fill_color=LIGHT_VIOLET, fill_opacity=1, sheen_factor=0.2)
		triangle.scale(np.sqrt(3) / 3)
		triangle_v = triangle.get_vertices()
		shift = vertices[0] - triangle_v[1]
		triangle.shift(RIGHT * shift[0] + UP * shift[1])

		triangles = [triangle]

		print(np.linalg.norm(triangle_v[0] - triangle_v[1]))
		print(np.linalg.norm(triangle_v[1] - triangle_v[2]))

		start_v = vertices[0]
		prev_v = vertices[0]
		for v in vertices[1:len(vertices) - 1]:
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
		self.wait(2)
		self.play(DrawBorderThenFill(logo), 
			FadeInAndShiftFromDirection(reducible), 
			FadeIn(patreon),
			run_time=2)

		self.play(Write(subscribe), Write(support))

		self.wait(13)

class ExampleProblem(Scene):
	def construct(self):
		self.setup_problem()


	def setup_problem(self):
		all_objects = []
		equation = TexMobject("a + b + c = n")
		equation[0].set_color(RED)
		# equation[1].set_color(YELLOW)
		equation[2].set_color(GREEN_SCREEN)
		# equation[3].set_color(YELLOW)
		equation[4].set_color(BLUE)
		# equation[5].set_color(YELLOW)
		equation[6].set_color(ORANGE)

		equation.scale(0.9)
		equation.move_to(UP)
		text_line1 = TextMobject("Find all sets of nonnegative integers")
		text_line2 = TextMobject(r"$(a, b, c)$ that sum to integer $n$ ($n \geq 0$)")
		text_line1.scale(0.8)
		text_line2.scale(0.8)
		text_line1.next_to(equation, DOWN)
		text_line2.next_to(text_line1, DOWN)
		text_line2[1].set_color(RED)
		text_line2[3].set_color(GREEN_SCREEN)
		text_line2[5].set_color(BLUE)
		text_line2[-6].set_color(ORANGE)
		text_line2[-4].set_color(ORANGE)

		self.play(Write(equation), run_time=2)
		self.wait()
		self.play(Write(text_line1), run_time=2)
		self.play(Write(text_line2), run_time=2)
		self.wait(4)



		problem = VGroup(equation, text_line1, text_line2)
		problem_shifted_up = problem.copy()

		problem_shifted_up = problem_shifted_up.shift(UP * 2.5)
		self.play(Transform(problem, problem_shifted_up))
		self.wait()

		all_objects.append(problem)

		h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
		h_line.next_to(problem, DOWN)
		v_line = Line(h_line.get_midpoint(), h_line.get_midpoint() + DOWN * 6)
		self.play(ShowCreation(h_line),
			GrowFromCenter(v_line))



		bob_solution = TextMobject("Bob's Solution")
		alice_solution = TextMobject("Alice's Solution")

		bob_solution_center = (h_line.get_start() + h_line.get_midpoint()) / 2 + DOWN * 0.5
		alice_solution_center = (h_line.get_midpoint() + h_line.get_end()) / 2 + DOWN * 0.5

		bob_solution.move_to(bob_solution_center)
		bob_solution.scale(0.8)
		alice_solution.move_to(alice_solution_center)
		alice_solution.scale(0.8)
		self.play(Write(bob_solution), Write(alice_solution), run_time=2)
		self.wait(2)

		all_objects.extend([h_line, v_line, bob_solution, alice_solution])

		bob_first_step = TextMobject(r"1. Try all combinations of $(a, b, c)$")
		bob_first_step[-2].set_color(BLUE)
		bob_first_step[-4].set_color(GREEN_SCREEN)
		bob_first_step[-6].set_color(RED)
		bob_second_step = TextMobject(r"2. If $a + b + c = n$, print($a, b, c$)")
		bob_second_step[-2].set_color(BLUE)
		bob_second_step[-4].set_color(GREEN_SCREEN)
		bob_second_step[-6].set_color(RED)
		bob_second_step[4].set_color(RED)
		bob_second_step[6].set_color(GREEN_SCREEN)
		bob_second_step[8].set_color(BLUE)
		bob_second_step[10].set_color(ORANGE)
		bob_first_step.scale(0.7)
		bob_first_step.next_to(bob_solution, DOWN)
		bob_second_step.scale(0.7)
		bob_second_step.next_to(bob_first_step, DOWN)

		self.play(Write(bob_first_step))
		self.wait(3)
		self.play(Write(bob_second_step))
		self.wait(3)

		all_objects.extend([bob_first_step, bob_second_step])

		example_bob = TextMobject(r"Example: $n = 3$")
		example_bob.scale(0.7)
		example_bob.next_to(bob_second_step, DOWN)
		self.play(FadeIn(example_bob))

		a = TexMobject("a")
		a.set_color(RED)
		a.move_to(LEFT * 5 + DOWN)

		b = TexMobject("b")
		b.set_color(GREEN_SCREEN)
		b.next_to(a, RIGHT * 4)

		c = TexMobject("c")
		c.set_color(BLUE)
		c.next_to(b, RIGHT * 4)

		all_objects.extend([example_bob, a, b, c])

		vert_line_a = Line(a.get_center() + RIGHT * 0.25 + UP * 0.25, 
			a.get_center() + RIGHT * 0.25 + DOWN * 0.25)
		h_line_a = Line(a.get_center() + RIGHT * 0.25 + DOWN * 0.25, 
			a.get_center() + RIGHT * 0.65 + DOWN * 0.25)
		vert_line_b = Line(b.get_center() + RIGHT * 0.25 + UP * 0.25, 
			b.get_center() + RIGHT * 0.25 + DOWN * 0.25)
		h_line_b = Line(b.get_center() + RIGHT * 0.25 + DOWN * 0.25, 
			b.get_center() + RIGHT * 0.65 + DOWN * 0.25)
		vert_line_c = Line(c.get_center() + RIGHT * 0.25 + UP * 0.25, 
			c.get_center() + RIGHT * 0.25 + DOWN * 0.25)
		h_line_c = Line(c.get_center() + RIGHT * 0.25 + DOWN * 0.25, 
			c.get_center() + RIGHT * 0.65 + DOWN * 0.25)

		a_val = TexMobject("0")
		a_val.scale(0.7)
		a_val.next_to(vert_line_a, RIGHT * 0.4)
		a_val.shift(UP * 0.05)
		b_val = TexMobject("0")
		b_val.scale(0.7)
		b_val.next_to(vert_line_b, RIGHT * 0.4)
		b_val.shift(UP * 0.05)
		c_val = TexMobject("0")
		c_val.scale(0.7)
		c_val.next_to(vert_line_c, RIGHT * 0.4)
		c_val.shift(UP * 0.05)

		a_group = VGroup(a, vert_line_a, h_line_a)
		a_group.scale(0.7)
		b_group = VGroup(b, vert_line_b, h_line_b)
		b_group.scale(0.7)
		c_group = VGroup(c, vert_line_c, h_line_c)
		c_group.scale(0.7)
		self.play(FadeIn(a_group), FadeIn(b_group), FadeIn(c_group))
		self.wait()
		self.play(FadeIn(a_val), FadeIn(b_val), FadeIn(c_val))
		self.wait()

		all_objects.extend([a_group, b_group, c_group, a_val, b_val, c_val])

		self.simulate_alg(a_val, b_val, c_val, 3, all_objects)

		self.wait(11)
		alice_first_step = TextMobject(r"1. For all $(a, b)$ set $c = n - (a + b)$")
		alice_first_step[-2].set_color(GREEN_SCREEN)
		alice_first_step[-4].set_color(RED)
		alice_first_step[-7].set_color(ORANGE)
		alice_first_step[-9].set_color(BLUE)
		alice_first_step[9].set_color(RED)
		alice_first_step[11].set_color(GREEN_SCREEN)
		alice_second_step = TextMobject(r"2. If $c \geq 0$, print($a, b, c$)")
		alice_second_step[4].set_color(BLUE)
		alice_second_step[-2].set_color(BLUE)
		alice_second_step[-4].set_color(GREEN_SCREEN)
		alice_second_step[-6].set_color(RED)
		alice_first_step.scale(0.7)
		alice_second_step.scale(0.7)
		alice_first_step.next_to(alice_solution, DOWN)
		alice_second_step.next_to(alice_first_step, DOWN)
		self.play(Write(alice_first_step))
		self.wait(7)
		self.play(Write(alice_second_step))
		self.wait(4)

		all_objects.extend([alice_first_step, alice_second_step])

		example_alice = TextMobject(r"Example: $n = 3$")
		example_alice.scale(0.7)
		example_alice.next_to(alice_second_step, DOWN)
		self.play(FadeIn(example_alice))

		all_objects.append(example_alice)

		c_right = TexMobject("c = n - (a + b) =")
		c_right[0].set_color(BLUE)
		c_right[2].set_color(ORANGE)
		c_right[5].set_color(RED)
		c_right[7].set_color(GREEN_SCREEN)
		c_right.move_to(RIGHT * 3 + DOWN * 1.5)

		b_right = TexMobject("b")
		b_right.set_color(GREEN_SCREEN)
		b_right.move_to(RIGHT * 3.5 + DOWN)

		a_right = TexMobject("a")
		a_right.set_color(RED)
		a_right.next_to(b_right, LEFT * 4)

		vert_line_a_right = Line(a_right.get_center() + RIGHT * 0.25 + UP * 0.25, 
			a_right.get_center() + RIGHT * 0.25 + DOWN * 0.25)
		h_line_a_right = Line(a_right.get_center() + RIGHT * 0.25 + DOWN * 0.25, 
			a_right.get_center() + RIGHT * 0.65 + DOWN * 0.25)
		vert_line_b_right = Line(b_right.get_center() + RIGHT * 0.25 + UP * 0.25, 
			b_right.get_center() + RIGHT * 0.25 + DOWN * 0.25)
		h_line_b_right = Line(b_right.get_center() + RIGHT * 0.25 + DOWN * 0.25, 
			b_right.get_center() + RIGHT * 0.65 + DOWN * 0.25)
		# vert_line_c_right = Line(c_right.get_center() + RIGHT * 0.25 + UP * 0.25, 
		# 	c_right.get_center() + RIGHT * 0.25 + DOWN * 0.25)
		# h_line_c_right = Line(c_right.get_center() + RIGHT * 0.25 + DOWN * 0.25, 
		# 	c_right.get_center() + RIGHT * 0.65 + DOWN * 0.25)

		a_val_right = TexMobject("0")
		a_val_right.scale(0.7)
		a_val_right.next_to(vert_line_a_right, RIGHT * 0.4)
		a_val_right.shift(UP * 0.05)
		b_val_right = TexMobject("0")
		b_val_right.scale(0.7)
		b_val_right.next_to(vert_line_b_right, RIGHT * 0.4)
		b_val_right.shift(UP * 0.05)
		c_val_right = TexMobject("3")
		c_val_right.scale(0.7)
		c_val_right.move_to(c_right[-1].get_center() + LEFT * SMALL_BUFF)

		a_group_right = VGroup(a_right, vert_line_a_right, h_line_a_right)
		a_group_right.scale(0.7)
		b_group_right = VGroup(b_right, vert_line_b_right, h_line_b_right)
		b_group_right.scale(0.7)
		c_group_right = VGroup(c_right)
		c_group_right.scale(0.7)
		self.play(FadeIn(a_group_right), FadeIn(b_group_right), FadeIn(c_group_right))
		self.wait()
		self.play(FadeIn(a_val_right), FadeIn(b_val_right), FadeIn(c_val_right))
		self.wait()

		all_objects.extend([a_val_right, b_val_right, c_val_right, a_group_right, b_group_right, c_group_right])

		self.simulate_alg_eff(a_val_right, b_val_right, c_val_right, 3, all_objects)
		self.wait()

		self.play(*[FadeOut(obj) for obj in all_objects])
		# self.wait()

		
		

	def simulate_alg_eff(self, a_val, b_val, c_val, n, all_objects, scale=0.7):
		count = 0
		abc_coord_center = RIGHT * 3 + DOWN * 2
		abc_coord_start = RIGHT * 3 + DOWN * 2
		for a in range(n + 1):
			for b in range(n + 1):
				a_val_new = TexMobject(str(a))
				a_val_new.scale(scale)
				a_val_new.move_to(a_val.get_center())
				b_val_new = TexMobject(str(b))
				b_val_new.scale(scale)
				b_val_new.move_to(b_val.get_center())
				c = n - (a + b)
				c_val_new = TexMobject(str(c))
				c_val_new.scale(scale)
				c_val_new.move_to(c_val.get_center())
				self.play(
					Transform(a_val, a_val_new),
					Transform(b_val, b_val_new),
					Transform(c_val, c_val_new),
					run_time=0.3
					)
				if c >= 0:
					self.play(
						Flash(a_val, color=GREEN_SCREEN), 
						Flash(b_val, color=GREEN_SCREEN), 
						Flash(c_val, color=GREEN_SCREEN),
						)
					a_b_c = VGroup(a_val, b_val, c_val)
					abc_coord = TexMobject("({0}, {1}, {2})".format(a, b, c))
					abc_coord[1].set_color(RED)
					abc_coord[3].set_color(GREEN_SCREEN)
					abc_coord[5].set_color(BLUE)
					abc_coord.scale(0.5)
					abc_coord.move_to(abc_coord_center)
					abc_coord_center += DOWN * 0.3
					self.play(TransformFromCopy(a_b_c, abc_coord))
					count += 1
					if count == 5:
						abc_coord_center = abc_coord_start + RIGHT

					all_objects.append(abc_coord)

	def simulate_alg(self, a_val, b_val, c_val, n, all_objects, scale=0.7):
		count = 0
		abc_coord_center = LEFT * 4 + DOWN * 1.5
		abc_coord_start = LEFT * 4 + DOWN * 1.5
		for a in range(n + 1):
			for b in range(n + 1):
				for c in range(n + 1):
					a_val_new = TexMobject(str(a))
					a_val_new.scale(scale)
					a_val_new.move_to(a_val.get_center())
					b_val_new = TexMobject(str(b))
					b_val_new.scale(scale)
					b_val_new.move_to(b_val.get_center())
					c_val_new = TexMobject(str(c))
					c_val_new.scale(scale)
					c_val_new.move_to(c_val.get_center())
					self.play(
						Transform(a_val, a_val_new),
						Transform(b_val, b_val_new),
						Transform(c_val, c_val_new),
						run_time=0.2
						)
					if a + b + c == n:
						self.play(
							Flash(a_val, color=GREEN_SCREEN), 
							Flash(b_val, color=GREEN_SCREEN), 
							Flash(c_val, color=GREEN_SCREEN),
							)
						a_b_c = VGroup(a_val, b_val, c_val)
						abc_coord = TexMobject("({0}, {1}, {2})".format(a, b, c))
						abc_coord[1].set_color(RED)
						abc_coord[3].set_color(GREEN_SCREEN)
						abc_coord[5].set_color(BLUE)
						abc_coord.scale(0.5)
						abc_coord.move_to(abc_coord_center)
						abc_coord_center += DOWN * 0.3
						self.play(TransformFromCopy(a_b_c, abc_coord))
						count += 1
						if count == 5:
							abc_coord_center = abc_coord_start + RIGHT
						all_objects.append(abc_coord)


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







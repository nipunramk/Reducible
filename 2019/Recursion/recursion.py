from big_ol_pile_of_manim_imports import *
import random
DEFAULT_COLORS = [BLUE, RED, ORANGE, GREEN, DARK_BLUE, VIOLET, PINK, LIGHT_BROWN, MAROON_C, GRAY]


class Introduction(Scene):
	def construct(self):
		rectangles, text = self.show_recursion_diagram()
		to_fadeout = self.steps_text()
		steps = to_fadeout[0]
		to_fadeout = to_fadeout[1:]
		steps_copy = steps.copy()
		steps_copy.move_to(RIGHT * 2.5 + UP)
		three_problems = TextMobject(r"$\downarrow$", "3 PROBLEMS")
		three_problems.scale(1.2)

		three_problems[0].next_to(steps_copy, DOWN)
		three_problems[1].next_to(three_problems[0], DOWN)



		self.play(*[FadeOut(obj) for obj in to_fadeout] + 
			[FadeOut(rect) for rect in rectangles] + 
			[FadeOut(t) for t in text] + [Transform(steps, steps_copy)],
			run_time=2)
		animation = Write(three_problems)
		fadeouts = [steps, three_problems]
		self.screen_rects(animation, fadeouts)

	def show_recursion_diagram(self):
		rectangles = []
		text = []
		height, width = 6, 8
		center = RIGHT * 2.5
		scale = 1.5
		for _ in range(30):
			if _ > 3:
				opacity = 0.5
			else:
				opacity = 1

			rectangle = Rectangle(height=height, width=width, opacity=opacity)
			rectangle.move_to(center)
			recursion = TextMobject("Recursion")
			recursion.set_color(YELLOW)
			recursion.scale(scale)
			recursion.move_to(rectangle.get_center() + DOWN * (height / 2 + scale / 4))
			text.append(recursion)
			rectangles.append(rectangle)
			top_edge = rectangle.get_center() + height / 2 * UP  
			height = height / 1.2
			width = width / 1.2
			center = top_edge + (height / 2) * DOWN
			scale /= 1.2

		run_time = 1.5
		for rect, t in zip(rectangles, text):
			self.play(ShowCreation(rect, run_time=run_time), FadeIn(t, run_time=run_time))
			run_time = run_time / 1.3

		# self.play(*[FadeIn(rect) for rect in rectangles] + [FadeIn(t) for t in text])

		return rectangles, text

	def steps_text(self):

		confused_character = create_confused_char(color=BLUE, scale=0.7, position=UP*1.8 + LEFT * 5)
		self.play(FadeInAndShiftFromDirection(confused_character, direction=UP))
		
		top_thought_bubble = SVGMobject("thought")
		top_thought_bubble.scale(1)
		top_thought_bubble.move_to(UP * 3.0 + RIGHT * 2 + LEFT * 5.2)

		hard = TextMobject("Looks hard ...")
		hard.scale(0.5)
		hard.move_to(top_thought_bubble.get_center() + UP * 0.35 + RIGHT * 0.04)
		hard.set_color(BLACK)
		self.play(FadeIn(top_thought_bubble), FadeIn(hard))

		self.wait(4)

		steps = TextMobject("5 SIMPLE STEPS")
		steps.scale(1.2)
		steps.move_to(LEFT * 4.2 + DOWN * 0)
		self.play(FadeIn(steps))
		self.wait(7)

		default_char = create_computer_char(color=BLUE, scale=0.7, position=DOWN*3.0 + LEFT * 5)
		self.play(FadeInAndShiftFromDirection(default_char))
		bottom_thought_bubble = SVGMobject("thought")
		bottom_thought_bubble.scale(1)
		bottom_thought_bubble.move_to(DOWN * 1.7 + RIGHT * 2 + LEFT * 5.2)

		ok = TextMobject("It's actually", "not that bad")
		ok.scale(0.5)
		ok[0].move_to(bottom_thought_bubble.get_center() + UP * 0.55 + RIGHT * 0.04)
		ok[1].move_to(bottom_thought_bubble.get_center() + UP * 0.25 + RIGHT * 0.04)
		ok.set_color(BLACK)
		self.play(FadeIn(bottom_thought_bubble), FadeIn(ok))

		self.wait(2)

		return [steps] + [default_char, confused_character, top_thought_bubble, bottom_thought_bubble]


	def screen_rects(self, animation, fadeouts):

		self.play(animation)
		rect3 = ScreenRectangle()
		rect3.set_width(3.7)
		rect3.set_height(2.2)
		rect3.move_to(LEFT * 3.5 + DOWN * 2.7)

		text3 = TextMobject("This one is hard")
		text3.scale(0.8)
		text3.next_to(rect3, RIGHT)
		
		rect2 = ScreenRectangle()
		rect2.set_width(3.7)
		rect2.set_height(2.2)
		rect2.move_to(LEFT * 3.5)

		text2 = TextMobject("This one is more challenging")
		text2.scale(0.8)
		text2.next_to(rect2, RIGHT)

		rect1 = ScreenRectangle()
		rect1.set_width(3.7)
		rect1.set_height(2.2)
		rect1.move_to(LEFT * 3.5 + UP * 2.7)

		text1 = TextMobject("This one is a fundamental problem")
		text1.scale(0.8)
		text1.next_to(rect1, RIGHT)

		self.play(FadeIn(rect1), FadeIn(rect2), FadeIn(rect3))
		self.wait()
		self.play(*[FadeOut(obj) for obj in fadeouts])
		self.play(FadeIn(text1))
		self.play(FadeIn(text2))
		self.play(FadeIn(text3))
		self.wait(3)

		self.play(FadeOut(text1), FadeOut(text2), FadeOut(text3))
		self.wait()

		pause_left = Rectangle(height=2, width=0.5, fill_color=YELLOW, fill_opacity=1)
		pause_left.set_color(YELLOW)
		pause_right = Rectangle(height=2, width=0.5, fill_color=YELLOW, fill_opacity=1)
		pause_right.set_color(YELLOW)
		pause_right.next_to(pause_left, RIGHT)
		pause = VGroup(pause_left, pause_right)
		pause.move_to(RIGHT * 3)
		self.play(FadeIn(pause))
		self.wait()

		pause_copy = pause.copy()
		play = RegularPolygon(fill_color=GREEN_SCREEN, fill_opacity=1)
		play.scale(1.5)
		play.set_color(GREEN_SCREEN)
		play.move_to(pause.get_center() + RIGHT * 0.5)
		self.play(ReplacementTransform(pause, play))
		self.wait(3)

		self.play(ReplacementTransform(play, pause_copy))
		self.wait(3)


class Conclusion(Scene):
	def construct(self):
		self.recap()

	def recap(self):

		step_title = TextMobject("5 SIMPLE STEPS")
		step_title.scale(1.5)
		step_title.to_edge(UP)
		h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
		h_line.next_to(step_title, DOWN)

		

		items = BulletedList(
			"What's the simplest possible input?",
			"Play around with examples and visualize!",
			"Relate hard cases to simpler cases",
			"Generalize the pattern",
			"Write code by combining recursive \\\\ pattern with the base case",
		)


		colors = [RED, YELLOW, GREEN_SCREEN, ORANGE, BLUE]
		for i in range(len(items)):
			items[i].set_color(colors[i])
		items.scale(0.8)
		items.to_edge(LEFT, buff = LARGE_BUFF)

		rect = ScreenRectangle()
		rect.set_width(FRAME_WIDTH - items.get_width() - 2)
		rect.next_to(items, RIGHT, MED_LARGE_BUFF)

		self.play(
			Write(step_title),
			ShowCreation(h_line),
			run_time=2
		)

		self.play(FadeIn(items))

		self.wait(6)
		self.play(FadeIn(rect))

		for i in range(len(items)):
			self.play(items.fade_all_but, i)
			if i == 2 or i == 3:
				self.wait(3)
			else:
				self.wait(5)

		items_copy = items.copy()
		for item in items_copy.submobjects:
			item.set_fill(opacity=1)

		self.play(Transform(items, items_copy))

		self.wait(8)

		self.play(FadeOut(items), FadeOut(step_title), 
			FadeOut(h_line), FadeOut(rect))
		

		# step1 = TextMobject("1. What's the simplest possible input?")
		# step1.scale(0.9)
		# step1.move_to(step_title.get_center() + DOWN)
		# step1.set_color(RED)
		# self.play(FadeIn(step1))

		# step2 = TextMobject("2. Play around with examples and visualize!")
		# step2.scale(0.9)
		# step2.set_color(YELLOW)
		# step2.next_to(step1, DOWN)
		# self.play(FadeIn(step2))

		# step3 = TextMobject("3. Relate hard cases to simpler cases")
		# step3.set_color(GREEN_SCREEN)
		# step3.scale(0.9)
		# step3.next_to(step2, DOWN)
		# self.play(FadeIn(step3))

		# step4 = TextMobject("4. Generalize the pattern")
		# step4.scale(0.9)
		# step4.set_color(ORANGE)
		# step4.next_to(step3, DOWN)
		# self.play(FadeIn(step4))
		
		# step5 = TextMobject("5. Write code by combining recursive pattern with the base case")
		# step5.scale(0.9)
		# step5.set_color(BLUE)
		# step5.next_to(step4, DOWN)
		# self.play(FadeIn(step5))

		# bonus = TextMobject("BONUS IDEA")
		# bonus.next_to(step5, DOWN)
		# self.play(FadeIn(bonus))

		# faith = TextMobject("Recursive leap of faith")
		# faith.set_color(TEAL)
		# faith.scale(0.9)
		# faith.next_to(bonus, DOWN)
		# self.play(FadeIn(faith))

		# self.wait()




class GridProblem(Scene):
	def construct(self):
		first_scene_objects = []
		problem_line1 = TextMobject(r"Write a function that takes two inputs $n$ and $m$")
		problem_line2 = TextMobject(r"and outputs the number of unique paths from the top")
		problem_line3 = TextMobject(r"left corner to bottom right corner of a $n \cross m$ grid.")
		problem_line4 = TextMobject(r"Constraints: you can only move down or right $1$ unit at a time.")

		problem_line1.scale(0.7)
		problem_line2.scale(0.7)
		problem_line3.scale(0.7)
		problem_line4.scale(0.7)

		problem_line1.shift(UP * 3.5)
		problem_line2.next_to(problem_line1, DOWN * 0.5)
		problem_line3.next_to(problem_line2, DOWN * 0.5)
		problem_line4.next_to(problem_line3, DOWN * 0.5)

		self.play(Write(problem_line1))
		self.play(Write(problem_line2))
		self.play(Write(problem_line3))
		self.play(Write(problem_line4))

		first_scene_objects.extend([problem_line1, problem_line2, problem_line3, problem_line4])

		self.wait(19)

		scale_07 = 0.7
		# TextMobject(r"$5 + \text{sum}(4)$")
		case2x4_text = TextMobject(r"$\text{grid\_paths}(2, 4)$", r"$\rightarrow$", r"$4$")
		case2x4_text.scale(scale_07)
		case2x4_text.set_color(YELLOW)

		case2x4_text.shift(UP * 1.75)
		self.play(Write(case2x4_text[0]))

		grid2x4_case = self.animate_paths(2, 4, 0.5, offset=-3, down_offset=DOWN,
			grid_position=UP, run_time=0.1)
		grid2x4_group = VGroup(*grid2x4_case)

		self.play(Write(case2x4_text[1]), Write(case2x4_text[2]))

		case3x3_text = TextMobject(r"$\text{grid\_paths}(3, 3)$", r"$\rightarrow$", r"$6$")
		case3x3_text.scale(scale_07)
		case3x3_text.set_color(YELLOW)

		case3x3_text.shift(DOWN * 0.85)
		self.play(Write(case3x3_text[0]))

		grid3x3_case = self.animate_paths(3, 3, 0.5, offset=-5, down_offset=DOWN * 1.5,
			grid_position=DOWN * 2, run_time=0.1)
		grid3x3_group = VGroup(*grid3x3_case)

		self.play(Write(case3x3_text[1]), Write(case3x3_text[2]))

		first_scene_objects.extend([case2x4_text, grid2x4_group, grid3x3_group, case3x3_text])

		self.wait()
		self.play(*[FadeOut(obj) for obj in first_scene_objects])



		step1 = TextMobject("1. What's the simplest possible input?")
		step1.scale(0.9)
		step1.shift(UP * 3)
		step1.set_color(RED)
		self.play(FadeIn(step1))

		self.wait(5)

		case1x1_text = TextMobject(r"$\text{grid\_paths}(1, 1)$", r"$\rightarrow$", r"$1$")
		case1x1_text.scale(scale_07)
		case1x1_text.set_color(YELLOW)

		case1x1_text.shift(UP * 2.5)
		self.play(Write(case1x1_text[0]))

		grid1x1_case = self.animate_paths(1, 1, 0.5, down_offset=DOWN * 0.5, run_time=0.6)
		grid1x1_group = VGroup(*grid1x1_case)
				

		self.play(Write(case1x1_text[1]), Write(case1x1_text[2]))

		shifted_group = grid1x1_group.copy()
		shifted_group.shift(LEFT * 4.5 + DOWN * 3)
		self.play(Transform(grid1x1_group, shifted_group))


		case2x1_text = TextMobject(r"$\text{grid\_paths}(2, 1)$", r"$\rightarrow$", r"$1$")
		case2x1_text.scale(scale_07)
		case2x1_text.set_color(YELLOW)

		case2x1_text.shift(UP * 2)
		self.play(Write(case2x1_text[0]))

		grid2x1_case = self.animate_paths(2, 1, 0.5, down_offset=DOWN, run_time=0.6, animate=False)
		grid2x1_group = VGroup(*grid2x1_case)
		self.play(FadeIn(grid2x1_group))
				

		self.play(Write(case2x1_text[1]), Write(case2x1_text[2]))

		shifted_group = grid2x1_group.copy()
		shifted_group.shift(DOWN * 2.4)
		self.play(Transform(grid2x1_group, shifted_group))


		case1x2_text = TextMobject(r"$\text{grid\_paths}(1, 2)$", r"$\rightarrow$", r"$1$")
		case1x2_text.scale(scale_07)
		case1x2_text.set_color(YELLOW)

		case1x2_text.shift(UP * 1.5)
		self.play(Write(case1x2_text[0]))

		grid1x2_case = self.animate_paths(1, 2, 0.5, down_offset=DOWN * 0.5, run_time=0.6, animate=False)
		grid1x2_group = VGroup(*grid1x2_case)
		self.play(FadeIn(grid1x2_group))

		self.play(Write(case1x2_text[1]), Write(case1x2_text[2]))

		shifted_group = grid1x2_group.copy()
		shifted_group.shift(DOWN * 3 + RIGHT * 4.5)
		self.play(Transform(grid1x2_group, shifted_group))

		grid1xM_case = self.animate_paths(1, 10, 0.5, down_offset=DOWN * 0.5, run_time=0.6, 
			grid_position=LEFT*3 + DOWN*0.5, animate=False)
		grid1xM_group = VGroup(*grid1xM_case)
		self.play(FadeIn(grid1xM_group))

		grid1xM = grid1xM_case[0]
		left_side = grid1xM[0].get_center() + LEFT * 0.5 * grid1xM[0].get_width()
		right_side = grid1xM[-1].get_center() + RIGHT * 0.5 * grid1xM[-1].get_width()
		line_above = Line(left_side + UP * 0.4, right_side + UP * 0.4)
		m = TexMobject("m")
		m.scale(scale_07)
		m.next_to(line_above, UP)
		self.play(GrowFromCenter(line_above), FadeIn(m))

		case1xM_text = TextMobject(r"$\text{grid\_paths}(1, m)$", r"$\rightarrow$", r"$1$")
		case1xM_text.scale(scale_07)
		case1xM_text.set_color(YELLOW)				
		case1xM_text.shift(UP)
		self.play(Write(case1xM_text))

		gridNx1_case = self.animate_paths(10, 1, 0.5, down_offset=RIGHT * 0.5, run_time=0.6, 
			grid_position=RIGHT*3 + DOWN*0.5, animate=False)
		gridNx1_group = VGroup(*gridNx1_case)
		self.play(FadeIn(gridNx1_group))

		gridNx1 = gridNx1_case[0]
		left_side = gridNx1[0].get_center() + UP * 0.5 * gridNx1[0].get_width()
		right_side = gridNx1[-1].get_center() + DOWN * 0.5 * gridNx1[-1].get_width()
		line_left = Line(left_side + LEFT * 0.4, right_side + LEFT * 0.4)
		n = TexMobject("n")
		n.scale(scale_07)
		n.next_to(line_left, LEFT)
		self.play(GrowFromCenter(line_left), FadeIn(n))

		caseNx1_text = TextMobject(r"$\text{grid\_paths}(n, 1)$", r"$\rightarrow$", r"$1$")
		caseNx1_text.scale(scale_07)
		caseNx1_text.set_color(YELLOW)				
		caseNx1_text.shift(UP * 0.5)
		self.play(Write(caseNx1_text))

		text = VGroup(*[case1x1_text, case1x2_text, case2x1_text, case1xM_text, caseNx1_text])

		condensed_text = TextMobject(r"$\text{grid\_paths}(n, m)$", r"$\rightarrow$", r"$1$", 
			r"$\text{if} \ n = 1 \ \text{or} \ m = 1$")
 

		condensed_text.scale(scale_07)
		condensed_text.shift(UP * 2.5)
		condensed_text.set_color(YELLOW)
		self.play(ReplacementTransform(text, condensed_text), run_time=2)
		self.wait(10)

		remaining_objects = VGroup(*[grid1xM_group, line_above, m, gridNx1_group, line_left, n, 
			grid1x1_group, grid2x1_group, grid1x2_group, step1, condensed_text])
		self.play(FadeOut(remaining_objects))

		step2 = TextMobject("2. Play around with examples and visualize!")
		step2.scale(0.9)
		step2.set_color(YELLOW)
		self.play(FadeIn(step2))
		self.wait()
		self.play(FadeOut(step2))

		grid2x2_case = self.animate_paths(2, 2, 1, offset=-1, run_time=0.2)
		grid2x2_group = VGroup(*grid2x2_case)
		# self.play(FadeIn(grid2x2_group))
		shifted_group = grid2x2_group.copy()
		shifted_group.shift(RIGHT * 4.5 + UP * 3.5)
		shifted_group.scale(0.5)
		self.play(Transform(grid2x2_group, shifted_group))

		grid2x1_case = self.animate_paths(2, 1, 1, animate=False)
		grid2x1_group = VGroup(*grid2x1_case)
		self.play(FadeIn(grid2x1_group))
		shifted_group = grid2x1_group.copy()
		shifted_group.shift(RIGHT * 1.5 + UP * 3.5)
		shifted_group.scale(0.5)
		self.play(Transform(grid2x1_group, shifted_group))

		grid1x2_case = self.animate_paths(1, 2, 1, animate=False)
		grid1x2_group = VGroup(*grid1x2_case)
		self.play(FadeIn(grid1x2_group))
		shifted_group = grid1x2_group.copy()
		shifted_group.shift(LEFT * 1.5 + UP * 3.5)
		shifted_group.scale(0.5)
		self.play(Transform(grid1x2_group, shifted_group))

		grid1x1_case = self.animate_paths(1, 1, 1, animate=False)
		grid1x1_group = VGroup(*grid1x1_case)
		self.play(FadeIn(grid1x1_group))
		shifted_group = grid1x1_group.copy()
		shifted_group.shift(LEFT * 4.5 + UP * 3.5)
		shifted_group.scale(0.5)
		self.play(Transform(grid1x1_group, shifted_group))

		grid3x3_case = self.animate_paths(3, 3, 1, offset=-5, down_offset=DOWN*3, animate=False)
		grid3x3_group = VGroup(*grid3x3_case)
		self.play(FadeIn(grid3x3_group))
		shifted_group = grid3x3_group.copy()
		shifted_group.shift(DOWN + RIGHT * 4)
		shifted_group.scale(0.5)
		self.play(Transform(grid3x3_group, shifted_group))

		grid3x2_case = self.animate_paths(3, 2, 1, offset=-2, down_offset=DOWN*3, 
			grid_position=LEFT*2, run_time=0.2, animate=False)
		grid3x2_group = VGroup(*grid3x2_case)
		self.play(FadeIn(grid3x2_group))
		shifted_group = grid3x2_group.copy()
		shifted_group.shift(DOWN + RIGHT)
		shifted_group.scale(0.5)
		self.play(Transform(grid3x2_group, shifted_group))

		grid2x3_case = self.animate_paths(2, 3, 1, offset=-2, down_offset=DOWN*3, 
			grid_position=LEFT*4, animate=False)
		grid2x3_group = VGroup(*grid2x3_case)
		grid2x3_group.shift(DOWN + LEFT)
		grid2x3_group.scale(0.5)
		self.play(FadeIn(grid2x3_group))

		grid1x3_case = self.animate_paths(1, 3, 1, offset=0, down_offset=DOWN*1.5, 
			grid_position=UP * 2 + LEFT*2, animate=False)
		grid1x3_group = VGroup(*grid1x3_case)
		grid1x3_group.shift(DOWN + LEFT)
		grid1x3_group.scale(0.5)
		self.play(FadeIn(grid1x3_group))

		grid3x1_case = self.animate_paths(3, 1, 1, offset=0, down_offset=RIGHT*2, 
			grid_position=UP*1.5 + RIGHT*2, animate=False)
		grid3x1_group = VGroup(*grid3x1_case)
		grid3x1_group.shift(DOWN + LEFT)
		grid3x1_group.scale(0.5)
		self.play(FadeIn(grid3x1_group))

		entire_group = VGroup(grid2x2_group, grid2x1_group, grid1x2_group,
			grid1x1_group, grid3x3_group, grid3x2_group, grid2x3_group,
			grid1x3_group, grid3x1_group)
		
		# dimension_1_list = [grid2x1_group, grid1x2_group, grid1x1_group,
		# 	grid1x3_group, grid3x1_group]
		# dimension_1_group = VGroup(grid2x1_group, grid1x2_group, grid1x1_group,
		# 	grid1x3_group, grid3x1_group)

		# boxes = [SurroundingRectangle(elem, color=GREEN_SCREEN, buff=SMALL_BUFF*2) for elem in dimension_1_list]
		# self.play(*[ShowCreation(box) for box in boxes])
		# self.wait()
		# self.play(*[FadeOut(box) for box in boxes])

		# # self.play(FadeOut(dimension_1_group))


		# dimension_1_group_copy = dimension_1_group.copy()
		# dimension_1_group_copy.scale(1.2)
		# dimension_1_group_copy.move_to(ORIGIN)
		# self.play(TransformFromCopy(dimension_1_group, dimension_1_group_copy), FadeOut(entire_group), run_time=2)
		# self.wait()
		# self.play(FadeOut(dimension_1_group_copy))

		# self.play(FadeIn(entire_group))
		# self.wait()

		dimension_3_list = [grid2x3_group, grid3x2_group, grid3x3_group]

		self.wait(10)

		boxes = [SurroundingRectangle(elem, color=GREEN_SCREEN, buff=SMALL_BUFF) for elem in dimension_3_list]
		self.play(*[ShowCreation(box) for box in boxes])
		self.wait()
		self.play(*[FadeOut(box) for box in boxes])

		grid2x3_group_copy = grid2x3_group.copy()

		grid2x3_group_copy.shift(UP * 3.5 + RIGHT * 1.5)
		grid2x3_group_copy.scale(1.2)

		grid3x2_group_copy = grid3x2_group.copy()
		grid3x2_group_copy.shift(UP * 3.5 + RIGHT * 4)
		grid3x2_group_copy.scale(1.2)

		grid3x3_group_copy = grid3x3_group.copy()
		grid3x3_group_copy.shift(LEFT * 4 + UP* 0.5)
		grid3x3_group_copy.scale(1.2)

		self.play(TransformFromCopy(grid2x3_group, grid2x3_group_copy), 
			TransformFromCopy(grid3x2_group, grid3x2_group_copy),
			TransformFromCopy(grid3x3_group, grid3x3_group_copy),
			FadeOut(entire_group), run_time=2)

		step3 = TextMobject("3. Relate hard cases to simpler cases")
		step3.set_color(GREEN_SCREEN)
		step3.scale(0.9)
		step3.shift(UP * 3.5)
		self.play(FadeIn(step3))

		self.wait()



		grid3x2_case_paths = grid3x2_group_copy[1:]
		new_paths3x2 = []
		grid3x2_paths_flat = self.flatten_grid(grid3x2_case_paths)

		grid2x3_case_paths = grid2x3_group_copy[1:]
		new_paths2x3 = []
		grid2x3_paths_flat = self.flatten_grid(grid2x3_case_paths)

		colors = [BLUE, RED, GREEN]

		for i, path in enumerate(grid3x2_case_paths):
			for square in path:
				square_copy = square.copy()
				square_copy.set_fill(colors[i])
				new_paths3x2.append(square_copy)

		path_3x2_tranforms = [Transform(path, path_new_color) for path, path_new_color 
			in zip(grid3x2_paths_flat, new_paths3x2)]

		colors = [ORANGE, DARK_BLUE, VIOLET]

		for i, path in enumerate(grid2x3_case_paths):
			for square in path:
				square_copy = square.copy()
				square_copy.set_fill(colors[i])
				new_paths2x3.append(square_copy)

		path_2x3_transforms = [Transform(path, path_new_color) for path, path_new_color 
			in zip(grid2x3_paths_flat, new_paths2x3)]
		color_transforms = path_3x2_tranforms+path_2x3_transforms
		self.play(*color_transforms)
		self.wait(5)

		grid2x3 = self.make_grid_from_flat(grid2x3_group_copy[0], 2, 3)
		grid3x3 = self.make_grid_from_flat(grid3x3_group_copy[0], 3, 3)
		small_filled_paths, large_filled_paths = self.show_relationship_among_paths(grid2x3, 
			grid3x3, [ORANGE, DARK_BLUE, VIOLET])
		grid3x3_case_paths = grid3x3_group_copy[1:]
		grid3x3_subset2x3_paths = [grid3x3_case_paths[2], grid3x3_case_paths[4], grid3x3_case_paths[5]]
		for i in range(len(grid2x3_case_paths)):
			path, path_copy = grid2x3_case_paths[i], small_filled_paths[i]
			self.play(TransformFromCopy(path, path_copy))
			self.wait()
			large_path, large_path_copy = grid3x3_subset2x3_paths[i], large_filled_paths[i]
			self.play(TransformFromCopy(large_path, large_path_copy))
			self.wait()
			self.play(FadeOut(path_copy), FadeOut(large_path_copy))

		self.wait()

		grid3x2 = self.make_grid_from_flat(grid2x3_group_copy[0], 3, 2)
		small_filled_paths, large_filled_paths = self.show_relationship_among_paths(grid3x2, 
			grid3x3, [BLUE, RED, GREEN])

		grid3x3_subset3x2_paths = [grid3x3_case_paths[0], grid3x3_case_paths[1], grid3x3_case_paths[3]]
		for i in range(len(grid3x2_case_paths)):
			path, path_copy = grid3x2_case_paths[i], small_filled_paths[i]
			self.play(TransformFromCopy(path, path_copy))
			self.wait()
			large_path, large_path_copy = grid3x3_subset3x2_paths[i], large_filled_paths[i]
			self.play(TransformFromCopy(large_path, large_path_copy))
			self.wait()
			self.play(FadeOut(path_copy), FadeOut(large_path_copy))

		self.wait()

		copy2x3 = grid2x3_group_copy.copy()
		copy3x2 = grid3x2_group_copy.copy()
		copy3x3 = grid3x3_group_copy.copy()

		copy2x3.shift(RIGHT * 4.5 + DOWN)
		copy2x3.scale(0.8)
		copy3x2.shift(RIGHT * 2 + DOWN)
		copy3x2.scale(0.8)
		copy3x3.shift(LEFT * 4 + UP *2)
		copy3x3.scale(0.8)

		self.play(ReplacementTransform(grid2x3_group_copy, copy2x3),
			ReplacementTransform(grid3x2_group_copy, copy3x2),
			ReplacementTransform(grid3x3_group_copy, copy3x3), run_time=2)


		equals = TexMobject("=")
		equals.set_color(YELLOW)
		equals.scale(2)
		equals.move_to((copy2x3.get_center() + copy3x3.get_center()) / 2)
		equals.shift(UP * 0.5)

		plus = TexMobject("+")
		plus.set_color(YELLOW)
		plus.scale(2)
		
		plus.move_to((copy2x3.get_center() + copy3x2.get_center()) / 2)
		plus.shift(UP * 0.5)

		self.play(Write(equals), Write(plus))

		self.wait()

		surround_rect_2x3 = SurroundingRectangle(copy2x3[0], color=GREEN_SCREEN, buff=SMALL_BUFF/4)
		surround_rect_3x2 = SurroundingRectangle(copy3x2[0], color=ORANGE, buff=SMALL_BUFF/2)

		flatten_copy3x3 = self.flatten_grid(copy3x3[0])
		flatten_copy2x3_rect = SurroundingRectangle(VGroup(*flatten_copy3x3[:6]), 
			color=GREEN_SCREEN, buff=SMALL_BUFF/4)

		flatten_copy3x2_rect_group = VGroup(*[flatten_copy3x3[0], flatten_copy3x3[1], flatten_copy3x3[3],
			flatten_copy3x3[4], flatten_copy3x3[6], flatten_copy3x3[7]])
		flatten_copy3x2_rect = SurroundingRectangle(flatten_copy3x2_rect_group, 
			color=ORANGE, buff=SMALL_BUFF/2)

		self.play(ShowCreation(surround_rect_2x3), ShowCreation(flatten_copy2x3_rect))
		self.play(ShowCreation(surround_rect_3x2), ShowCreation(flatten_copy3x2_rect))
		
		self.wait(2)

		self.play(FadeOut(copy2x3), FadeOut(copy3x2), FadeOut(copy3x3),
			FadeOut(equals), FadeOut(plus), FadeOut(surround_rect_2x3),
			FadeOut(surround_rect_3x2), FadeOut(flatten_copy3x2_rect),
			FadeOut(flatten_copy2x3_rect))

		self.wait()

		step_3_shifted_down = step3.copy()
		step_3_shifted_down.move_to(DOWN * 3.5)
		self.play(Transform(step3, step_3_shifted_down), run_time=2)


		order = [0, 1, 2, 4, 5, 7, 3, 6, 8, 9]
		grid3x4_case = self.animate_paths(3, 4, 1, offset=-10, down_offset=DOWN*3, 
			grid_position=UP * 4.5, animate=False, offset_amount=2.5, reorder=True,
			order=order)
		grid3x4_case[0].shift(RIGHT * 0.5)
		grid3x4_group = VGroup(*grid3x4_case)
		grid3x4_group.shift(DOWN + LEFT)
		grid3x4_group.scale(0.5)
		grid3x4_flat = grid3x4_group[0]
		self.play(FadeIn(grid3x4_flat))

		grid_3x4_subset3x3_group = VGroup(*self.slice_n_columns(grid3x4_flat, 3, 4, 3))
		surround_rect_3x3 = SurroundingRectangle(grid_3x4_subset3x3_group, 
			color=ORANGE, buff=SMALL_BUFF/4)

		self.play(ShowCreation(surround_rect_3x3))
		self.wait()


		grid3x3_case = self.animate_paths(3, 3, 1, offset=-5, down_offset=DOWN*3, 
			animate=False)
		grid3x3_group = VGroup(*grid3x3_case)
		grid3x3_group.shift(LEFT * 3)
		grid3x3_group.scale(0.5)
		grid3x3_flat = grid3x3_group[0]
		self.play(TransformFromCopy(grid_3x4_subset3x3_group, grid3x3_group[0]), 
			FadeOut(surround_rect_3x3))
		self.wait()

		grid_3x4_subset2x4_group = VGroup(*grid3x4_flat[:8])
		surround_rect_2x4 = SurroundingRectangle(grid_3x4_subset2x4_group, 
			color=GREEN_SCREEN, buff=SMALL_BUFF/4)

		self.play(ShowCreation(surround_rect_2x4))
		self.wait()

		grid2x4_case = self.animate_paths(2, 4, 1, offset=-3, down_offset=DOWN*3, 
			grid_position=RIGHT * 3, animate=False, offset_amount=2.5, 
			colors=[PINK, LIGHT_BROWN, MAROON_C, GRAY])
		# grid2x4_case[0].shift(RIGHT * 0.5)
		grid2x4_group = VGroup(*grid2x4_case)
		grid2x4_group.scale(0.5)
		self.play(TransformFromCopy(grid_3x4_subset2x4_group, grid2x4_group[0]), 
			FadeOut(surround_rect_2x4))
		self.wait()
		

		grid3x3_case_paths = grid3x3_group[1:]
		self.play(FadeIn(grid3x3_case_paths))
		self.wait()

		grid2x4_case_paths = grid2x4_group[1:]
		self.play(FadeIn(grid2x4_case_paths))
		self.wait()

		grid3x4 = self.make_grid_from_flat(grid3x4_flat, 3, 4)
		grid3x3 = self.make_grid_from_flat(grid3x3_flat, 3, 3)
		small_filled_paths, large_filled_paths = self.show_relationship_among_paths(grid3x3, 
			grid3x4, DEFAULT_COLORS[:6])

		grid3x4_case_paths = grid3x4_group[1:]
		grid3x4_subset3x3_paths = grid3x3_case_paths
		for i in range(len(grid3x4_subset3x3_paths)):
			path, path_copy = grid3x3_case_paths[i], small_filled_paths[i]
			self.play(TransformFromCopy(path, path_copy), run_time=0.7)
			large_path_copy, large_path = grid3x4_case_paths[i], large_filled_paths[i]
			self.play(FadeIn(large_path), run_time=0.7)
			self.play(ReplacementTransform(large_path, large_path_copy), FadeOut(path_copy), run_time=0.7)

		# self.wait()

		
		grid2x4_flat = grid2x4_group[0]

		grid3x4 = self.make_grid_from_flat(grid3x4_flat, 3, 4)
		grid2x4 = self.make_grid_from_flat(grid2x4_flat, 2, 4)
		small_filled_paths, large_filled_paths = self.show_relationship_among_paths(grid2x4, 
			grid3x4, DEFAULT_COLORS[6:])

		grid3x4_case_paths_latter = grid3x4_group[7:]
		grid3x4_subset2x4_paths = grid2x4_case_paths
		for i in range(len(grid3x4_subset2x4_paths)):
			path, path_copy = grid2x4_case_paths[i], small_filled_paths[i]
			self.play(TransformFromCopy(path, path_copy), run_time=0.7)
			large_path_copy, large_path = grid3x4_case_paths_latter[i], large_filled_paths[i]
			self.play(FadeIn(large_path), run_time=0.7)
			self.play(ReplacementTransform(large_path, large_path_copy), FadeOut(path_copy), run_time=0.7)

		self.wait(12)

		self.play(FadeOut(grid2x4_group), FadeOut(grid3x3_group), FadeOut(grid3x4_group))
		self.wait()

		step4 = TextMobject("4. Generalize the pattern")
		step4.scale(0.9)
		step4.set_color(ORANGE)
		step4.move_to(step3.get_center())
		self.play(ReplacementTransform(step3, step4))
		self.wait()

		square_length = 0.4
		grid = self.create_grid(9, 7, square_length)
		grid_group = get_group_grid(grid)
		grid_group.move_to(ORIGIN)
		grid_group.set_color(WHITE)
		flattened_grid = self.flatten_grid(grid)
		self.play(FadeIn(grid_group))


		row_dimension_line = Line(square_length / 2 * UP + grid[0][0].get_center(), 
			square_length / 2 * DOWN + grid[-1][0].get_center())
		column_dimension_line = Line(square_length / 2 * LEFT + grid[0][0].get_center(), 
			square_length / 2 * RIGHT + grid[0][-1].get_center())
		row_dimension_line.shift(LEFT * square_length)
		row_dimension_line.set_color(BLUE)
		column_dimension_line.shift(UP * square_length)
		column_dimension_line.set_color(BLUE)



		M, N = TexMobject("n"), TexMobject("m")
		M.set_color(BLUE)
		N.set_color(BLUE)
		M.next_to(row_dimension_line, LEFT)
		N.next_to(column_dimension_line, UP)
		self.play(GrowFromCenter(row_dimension_line), GrowFromCenter(column_dimension_line))
		self.play(FadeIn(M), FadeIn(N))

		
		nxm_text = TextMobject(r"$\text{grid\_paths}(n, m)$")
		nxm_text.set_color(YELLOW)
		nxm_text.scale(scale_07)
		nxm_text.next_to(grid_group, DOWN)
		self.play(FadeIn(nxm_text))

		MxNgrid_elements = [grid_group, row_dimension_line, column_dimension_line, M, N, nxm_text]
		shift_amount = LEFT * 4
		new_elements = []
		for element in MxNgrid_elements:
			element_copy = element.copy()
			element_copy.shift(shift_amount)
			new_elements.append(element_copy)
		self.play(*[Transform(orig, new) for orig, new in zip(MxNgrid_elements, new_elements)])
		self.wait()

		subsetMxN_1 = VGroup(*self.slice_n_columns(flattened_grid, 6, 7, 9))
		surround_rect_MxN_1 = SurroundingRectangle(subsetMxN_1, color=ORANGE, buff=SMALL_BUFF/4)

		self.play(ShowCreation(surround_rect_MxN_1))
		shifted_subsetMxN_1 = subsetMxN_1.copy()
		shifted_subsetMxN_1 = shifted_subsetMxN_1.shift(RIGHT * 5)
		shifted_subsetMxN_1.set_color(ORANGE)

		equals = TexMobject("=")
		equals.scale(1.5)
		equals.set_color(YELLOW)
		equals.move_to((grid_group.get_center() + shifted_subsetMxN_1.get_center()) / 2)
		equals.shift(LEFT * 0.3)
		self.play(Write(equals))
		self.play(TransformFromCopy(subsetMxN_1, shifted_subsetMxN_1), FadeOut(surround_rect_MxN_1))

		grid_subset = self.make_grid_from_flat(shifted_subsetMxN_1, 9, 6)
		row_dimension_line = Line(square_length / 2 * UP + grid_subset[0][0].get_center(), 
			square_length / 2 * DOWN + grid_subset[-1][0].get_center())
		column_dimension_line = Line(square_length / 2 * LEFT + grid_subset[0][0].get_center(), 
			square_length / 2 * RIGHT + grid_subset[0][-1].get_center())
		row_dimension_line.shift(LEFT * square_length)
		row_dimension_line.set_color(BLUE)
		column_dimension_line.shift(UP * square_length)
		column_dimension_line.set_color(BLUE)

		lines_and_text = []
		M, N_1 = TexMobject("n"), TexMobject("m - 1")
		M.set_color(BLUE)
		N_1.set_color(BLUE)
		M.next_to(row_dimension_line, LEFT)
		N_1.next_to(column_dimension_line, UP)
		self.play(GrowFromCenter(row_dimension_line), GrowFromCenter(column_dimension_line))
		self.play(FadeIn(M), FadeIn(N_1))

		lines_and_text.extend([row_dimension_line, column_dimension_line, M, N_1])

		nxm_1_text = TextMobject(r"$\text{grid\_paths}(n, m - 1)$")
		nxm_1_text.set_color(YELLOW)
		nxm_1_text.scale(scale_07)
		nxm_1_text.next_to(shifted_subsetMxN_1, DOWN)
		self.play(FadeIn(nxm_1_text))

		self.wait(5)

		subsetM_1xN = VGroup(*flattened_grid[:56])
		surround_rect_M_1xN = SurroundingRectangle(subsetM_1xN, color=GREEN_SCREEN, buff=SMALL_BUFF/4)

		self.play(ShowCreation(surround_rect_M_1xN))
		shifted_subsetM_1xN = subsetM_1xN.copy()
		shifted_subsetM_1xN = shifted_subsetM_1xN.shift(RIGHT * 9.5)
		shifted_subsetM_1xN.set_color(GREEN_SCREEN)

		plus = TexMobject("+")
		plus.scale(1.5)
		plus.set_color(YELLOW)
		plus.move_to((shifted_subsetMxN_1.get_center() + shifted_subsetM_1xN.get_center()) / 2)
		plus.shift(LEFT * 0.3)
		self.play(Write(plus))

		self.play(TransformFromCopy(subsetM_1xN, shifted_subsetM_1xN), FadeOut(surround_rect_M_1xN), run_time=1.5)
		
		grid_subset = self.make_grid_from_flat(shifted_subsetM_1xN, 8, 7)
		row_dimension_line = Line(square_length / 2 * UP + grid_subset[0][0].get_center(), 
			square_length / 2 * DOWN + grid_subset[-1][0].get_center())
		column_dimension_line = Line(square_length / 2 * LEFT + grid_subset[0][0].get_center(), 
			square_length / 2 * RIGHT + grid_subset[0][-1].get_center())
		row_dimension_line.shift(LEFT * square_length)
		row_dimension_line.set_color(BLUE)
		column_dimension_line.shift(UP * square_length)
		column_dimension_line.set_color(BLUE)

		M_1, N = TexMobject("n - 1"), TexMobject("m")
		M_1.set_color(BLUE)
		N.set_color(BLUE)
		M_1.next_to(row_dimension_line, LEFT)
		N.next_to(column_dimension_line, UP)
		M_1.rotate(-TAU/4)
		M_1.shift(RIGHT * 0.5)
		self.play(GrowFromCenter(row_dimension_line), GrowFromCenter(column_dimension_line))
		self.play(FadeIn(M_1), FadeIn(N))

		lines_and_text.extend([row_dimension_line, column_dimension_line, M_1, N])


		n_1xm_text = TextMobject(r"$\text{grid\_paths}(n - 1, m)$")
		n_1xm_text.set_color(YELLOW)
		n_1xm_text.scale(scale_07)
		n_1xm_text.next_to(shifted_subsetM_1xN, DOWN)
		self.play(FadeIn(n_1xm_text))

		self.wait(10)

		step5 = TextMobject("5. Write code by combining recursive pattern with the base case")
		step5.scale(0.9)
		step5.set_color(BLUE)
		step5.move_to(step4.get_center())
		self.play(ReplacementTransform(step4, step5))
		self.wait(3)

		fadeout_objects = MxNgrid_elements[:-1] + [grid[-1][-1]] + [shifted_subsetM_1xN,  shifted_subsetMxN_1] + lines_and_text


		old_symbols = [nxm_text, equals, nxm_1_text, plus, n_1xm_text]
		new_nxm = nxm_text.copy()
		new_nxm_1 = nxm_1_text.copy()
		new_equals = TexMobject('=')
		new_equals.set_color(YELLOW)
		new_equals.scale(scale_07)
		new_plus = TexMobject('+')
		new_plus.set_color(YELLOW)
		new_plus.scale(scale_07)
		new_n_1xm = n_1xm_text.copy()

		new_nxm_1.move_to(UP * 2.5)
		new_equals.next_to(new_nxm_1, LEFT)
		new_plus.next_to(new_nxm_1, RIGHT)
		new_n_1xm.next_to(new_plus, RIGHT)
		new_nxm.next_to(new_equals, LEFT)

		new_symbols = [new_nxm, new_equals, new_nxm_1, new_plus, new_n_1xm]
		transforms = [ReplacementTransform(old_symbol, new_symbol) for old_symbol, new_symbol 
			in zip(old_symbols, new_symbols)]
		
		fadeouts = [FadeOut(obj) for obj in fadeout_objects]

		self.play(*fadeouts + transforms, run_time=2)
		self.wait()

		base_case_exp = TextMobject(r"1 \ $\text{if} \ n = 1 \ \text{or} \ m = 1$")
		base_case_exp.set_color(YELLOW)
		base_case_exp.scale(scale_07)
		rhs = VGroup(*new_symbols[2:])
		lhs = VGroup(*new_symbols[:2])
		lhs_shifted_down = lhs.copy()
		lhs_shifted_down.shift(DOWN * 0.2 + LEFT * 0.2)
		rhs_shifted_down = rhs.copy()
		rhs_shifted_down.shift(DOWN * 0.5 + RIGHT * 0.2)
		base_case_exp.next_to(rhs_shifted_down, UP)
		base_case_exp.shift(LEFT * 1.9)

		entire_rhs = VGroup(base_case_exp, rhs_shifted_down)
		brace = Brace(entire_rhs, LEFT, buff=SMALL_BUFF)
		self.play(ReplacementTransform(rhs, rhs_shifted_down), ReplacementTransform(lhs, lhs_shifted_down),
			GrowFromCenter(brace), FadeIn(base_case_exp))

		self.wait(10)

		self.play(FadeOut(brace), FadeOut(rhs_shifted_down), 
			FadeOut(base_case_exp), FadeOut(entire_rhs), 
			FadeOut(step5), FadeOut(lhs_shifted_down))



	def slice_n_columns(self, grid, n, cols, rows):
		columns = []
		for i in range(rows):
			start = cols * i
			end = start + n
			# print(start, end)
			columns.extend(grid[start:end])

		return columns


	def make_grid_from_flat(self, flattened_grid, rows, cols):
		grid = []
		for i in range(rows):
			row = []
			for j in range(cols):
				row.append(flattened_grid[cols * i + j])
			grid.append(row)
		return grid


	def show_relationship_among_paths(self, small_grid, large_grid, colors):
		rows_small, cols_small = len(small_grid), len(small_grid[0])
		small_paths = self.generate_all_path_strings(rows_small - 1, cols_small - 1)
		rows_large, cols_large = len(large_grid), len(large_grid[0])
		if rows_large > rows_small:
			large_paths = [p + 'D' for p in small_paths]
		else:
			large_paths = [p + 'R' for p in small_paths]

		small_filled_paths = []
		for i, path_string in enumerate(small_paths):
			path = self.get_path_from_string(path_string, large_grid)
			filled_path = self.fill_path(path, colors[i], animate=False)
			small_filled_paths.append(filled_path)

		large_filled_paths = []
		for i, path_string in enumerate(large_paths):
			path = self.get_path_from_string(path_string, large_grid)
			filled_path = self.fill_path(path, colors[i], animate=False, opacity=0.7)
			large_filled_paths.append(filled_path)

		return small_filled_paths, large_filled_paths


	def animate_paths(self, rows, columns, square_length, offset=0, 
		down_offset=DOWN*2, grid_position=ORIGIN, animate=True, 
		offset_amount=2, colors=DEFAULT_COLORS, reorder=False, order=None, run_time=1):

		grid = self.create_grid(rows, columns, square_length)
		grid_group = get_group_grid(grid)
		grid_group.move_to(grid_position)
		grid_group.set_color(WHITE)
		if animate:
			self.play(FadeIn(grid_group))
		flattened_grid = self.flatten_grid(grid)
		flatten_grid_copy = [g.copy() for g in flattened_grid]
		path_strings = self.generate_all_path_strings(rows - 1, columns - 1)
		if reorder:
			path_strings = self.reorder_path(path_strings, order)
		paths = []
		for i, string in enumerate(path_strings):
			path = self.get_path_from_string(string, grid)
			filled_path = self.fill_path(path, colors[i % len(colors)], animate=animate, run_time=run_time)
			
			displayed_path = self.show_path(filled_path, 
				RIGHT * offset + down_offset, flattened_grid, flatten_grid_copy, animate=animate)
			paths.append(displayed_path)
			offset += offset_amount

		return [VGroup(*flatten_grid_copy)] + paths

	def reorder_path(self, path_strings, order):
		return [path_strings[k] for k in order]


	def flatten_grid(self, grid):
		flattened = []
		for i in range(len(grid)):
			for j in range(len(grid[0])):
				flattened.append(grid[i][j])

		return flattened

	def get_path_from_string(self, string, grid):
		path = [grid[0][0]]
		i, j = 0, 0
		for c in string:
			if c == 'R':
				j += 1
			else:
				i += 1
			path.append(grid[i][j])
		return path

	def fill_path(self, path, color, animate=True, opacity=1, run_time=1):
		filled_path = []
		for i in range(len(path)):
			square = path[i]
			new_square = square.copy()
			new_square.set_fill(color=color, opacity=opacity)
			filled_path.append(new_square)
			if animate:
				self.play(Transform(square, new_square, run_time=run_time))

		return VGroup(*filled_path)
	
	def generate_all_path_strings(self, rows, cols):
		string = 'R' * cols + 'D' * rows
		from itertools import permutations
		perms = sorted(list(set([''.join(p) for p in permutations(string)])))
		return perms


	def show_path(self, path, position, new_grid, orig_grid, animate=True, run_time=1):
		path_copy = [p.copy() for p in path]
		new_path = [p.copy() for p in path]
		for p in path_copy:
			p.shift(position)

		group_path = VGroup(*path)
		group_path_copy = VGroup(*path_copy)
		group_path_copy.scale(0.6)

		if animate:
			self.play(TransformFromCopy(group_path, group_path_copy), run_time=run_time)
			self.play(*[ReplacementTransform(new, orig) for new, orig in zip(new_grid, orig_grid)], run_time=run_time)
		return path_copy


		

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

def get_group_grid(grid):
	squares = []
	for row in grid:
		for square in row:
			squares.append(square)
	return VGroup(*squares)


class CountPartitions(Scene):
	def construct(self):
		self.define_problem()
		self.base_case()
		self.show_examples()
		self.generalize()


	def base_case(self):
		step1 = TextMobject("1. What's the simplest possible input?")
		step1.scale(0.9)
		step1.shift(UP * 3)
		step1.set_color(RED)
		self.play(FadeIn(step1))
		scale_07 = 0.7
		case00_text = TextMobject(r"$\text{count\_partitions}(0, 0)$", r"$\rightarrow$", r"$1$")
		case00_text.scale(scale_07)
		case00_text.set_color(YELLOW)
		case00_text.next_to(step1, DOWN)
		self.wait(6)
		self.play(Write(case00_text[0]))
		self.wait(12)
		self.play(Write(case00_text[1]))
		self.play(Write(case00_text[2]))

		self.wait(3)

		case01_text = TextMobject(r"$\text{count\_partitions}(0, 1)$", r"$\rightarrow$", r"$1$")
		case01_text.scale(scale_07)
		case01_text.set_color(YELLOW)
		case01_text.next_to(case00_text, DOWN)
		self.play(Write(case01_text[0]))
		self.wait()
		self.play(Write(case01_text[1]))
		self.play(Write(case01_text[2]))


		case02_text = TextMobject(r"$\text{count\_partitions}(0, 2)$", r"$\rightarrow$", r"$1$")
		case02_text.scale(scale_07)
		case02_text.set_color(YELLOW)
		case02_text.next_to(case01_text, DOWN)
		self.play(Write(case02_text[0]))
		self.wait()
		self.play(Write(case02_text[1]))
		self.play(Write(case02_text[2]))

		self.wait()

		base_case_n0 = VGroup(case00_text, case01_text, case02_text)
		general_base_casen0 = TextMobject(r"$\text{count\_partitions}(n, m)$", r"$\rightarrow$", r"$1 \ \text{if} \ n = 0$")
		general_base_casen0.scale(scale_07)
		general_base_casen0.set_color(YELLOW)
		general_base_casen0.next_to(step1, DOWN)
		self.play(ReplacementTransform(base_case_n0, general_base_casen0))
		self.wait(10)	
		

		case10_text = TextMobject(r"$\text{count\_partitions}(1, 0)$", r"$\rightarrow$", r"$0$")
		case10_text.scale(scale_07)
		case10_text.set_color(YELLOW)
		case10_text.next_to(general_base_casen0, DOWN)
		self.play(Write(case10_text[0]))
		self.wait(12)
		self.play(Write(case10_text[1]))
		self.play(Write(case10_text[2]))

		self.wait()

		case20_text = TextMobject(r"$\text{count\_partitions}(2, 0)$", r"$\rightarrow$", r"$0$")
		case20_text.scale(scale_07)
		case20_text.set_color(YELLOW)
		case20_text.next_to(case10_text, DOWN)
		self.play(Write(case20_text[0]))
		self.wait()
		self.play(Write(case20_text[1]))
		self.play(Write(case20_text[2]))

		self.wait(5)

		base_case_m0 = VGroup(case10_text, case20_text)
		general_base_casem0 = TextMobject(r"$\text{count\_partitions}(n, m)$", r"$\rightarrow$", r"$0 \ \text{if} \ m = 0$")
		general_base_casem0.scale(scale_07)
		general_base_casem0.set_color(YELLOW)
		general_base_casem0.next_to(general_base_casen0, DOWN)
		self.play(ReplacementTransform(base_case_m0, general_base_casem0))
		self.wait(5)	

		all_objects = [step1, general_base_casen0, general_base_casem0]
		self.play(*[FadeOut(obj) for obj in all_objects])


	def define_problem(self):
		problem_line1 = TextMobject("Write a function that counts the number of ways you can")
		problem_line2 = TextMobject(r"partition $n$ objects using parts up to $m$ (assuming $m \geq 0$)")
		problem_line1.scale(0.9)
		problem_line2.scale(0.9)

		problem_line1.shift(UP * 3.5)
		problem_line2.shift(UP * 3)
		self.play(Write(problem_line1), run_time=2)
		self.play(Write(problem_line2), run_time=2)
		self.wait(8)

		n, m = 6, 4
		all_groups_list = self.partition(n, m, LEFT * 3 + UP * 2, run_time=0.3)
		all_groups = VGroup(*all_groups_list)
		
		self.wait()

		n, m = 5, 5
		all_groups_list2 = self.partition(n, m, RIGHT * 3 + UP * 2, run_time=0.4)
		all_groups2 = VGroup(*all_groups_list2)

		self.wait(18)

		objects = [problem_line1, problem_line2, all_groups, all_groups2]

		self.play(*[FadeOut(obj) for obj in objects])
		self.wait()
		

	def show_examples(self):
		step2 = TextMobject("2. Play around with examples and visualize!")
		step2.scale(0.9)
		step2.shift(UP * 3)
		step2.set_color(YELLOW)
		self.play(FadeIn(step2))

		n, m = 5, 3
		all_groups_list = self.partition(n, m, UP * 2, animate=False)
		all_groups = VGroup(*all_groups_list)
		shifted_all_groups = all_groups.copy()
		shifted_all_groups.scale(0.7)
		shifted_all_groups.move_to(LEFT * 22 / 4 + UP * 2.5)
		
		self.play(Transform(all_groups, shifted_all_groups), FadeOut(step2))

		n, m = 4, 3
		all_groups_list2 = self.partition(n, m, UP * 2, animate=False, animate_all=False)
		all_groups2 = VGroup(*all_groups_list2)
		self.play(FadeIn(all_groups2))
		shifted_all_groups2 = all_groups2.copy()
		shifted_all_groups2.scale(0.7)
		shifted_all_groups2.move_to(LEFT * 9 / 4 + UP * 2.5)
		
		self.play(Transform(all_groups2, shifted_all_groups2))

		n, m = 3, 3
		all_groups_list3 = self.partition(n, m, ORIGIN, animate=False, animate_all=False)
		all_groups3 = VGroup(*all_groups_list3)
		self.play(FadeIn(all_groups3))
		shifted_all_groups3 = all_groups3.copy()
		shifted_all_groups3.scale(0.7)
		shifted_all_groups3.move_to(UP * 2.5 + RIGHT * 3 / 4)
		
		self.play(Transform(all_groups3, shifted_all_groups3))

		n, m = 2, 3
		all_groups_list4 = self.partition(n, m, ORIGIN, animate=False, animate_all=False)
		all_groups4 = VGroup(*all_groups_list4)
		self.play(FadeIn(all_groups4))
		shifted_all_groups4 = all_groups4.copy()
		shifted_all_groups4.scale(0.7)
		shifted_all_groups4.move_to(UP * 2.5 + RIGHT * 13 / 4)
		
		self.play(Transform(all_groups4, shifted_all_groups4))


		n, m = 1, 3
		all_groups_list5 = self.partition(n, m, ORIGIN, animate=False, animate_all=False)
		all_groups5 = VGroup(*all_groups_list5)
		self.play(FadeIn(all_groups5))
		shifted_all_groups5 = all_groups5.copy()
		shifted_all_groups5.scale(0.7)
		shifted_all_groups5.move_to(UP * 2.5 + RIGHT * 22 / 4)
		
		self.play(Transform(all_groups5, shifted_all_groups5))


		n, m = 5, 2
		all_groups_list6 = self.partition(n, m, ORIGIN, animate=False, animate_all=False)
		all_groups6 = VGroup(*all_groups_list6)
		self.play(FadeIn(all_groups6))
		shifted_all_groups6 = all_groups6.copy()
		shifted_all_groups6.scale(0.7)
		shifted_all_groups6.move_to(LEFT * 22 / 4)
		
		self.play(Transform(all_groups6, shifted_all_groups6))

		n, m = 4, 2
		all_groups_list7 = self.partition(n, m, ORIGIN, animate=False, animate_all=False)
		all_groups7 = VGroup(*all_groups_list7)
		all_groups7.scale(0.7)
		all_groups7.move_to(LEFT * 9 / 4)

		self.play(FadeIn(all_groups7))

		n, m = 3, 2
		all_groups_list8 = self.partition(n, m, ORIGIN, animate=False, animate_all=False)
		all_groups8 = VGroup(*all_groups_list8)
		all_groups8.scale(0.7)
		all_groups8.move_to(RIGHT * 3 / 4)

		self.play(FadeIn(all_groups8))


		n, m = 2, 2
		all_groups_list9 = self.partition(n, m, ORIGIN, animate=False, animate_all=False)
		all_groups9 = VGroup(*all_groups_list9)
		all_groups9.scale(0.7)
		all_groups9.move_to(RIGHT * 13 / 4)

		self.play(FadeIn(all_groups9))


		n, m = 1, 2
		all_groups_list10 = self.partition(n, m, ORIGIN, animate=False, animate_all=False)
		all_groups10 = VGroup(*all_groups_list10)
		all_groups10.scale(0.7)
		all_groups10.move_to(RIGHT * 22 / 4)

		self.play(FadeIn(all_groups10))


		n, m = 5, 1
		all_groups_list11 = self.partition(n, m, ORIGIN, animate=False, animate_all=False)
		all_groups11 = VGroup(*all_groups_list11)
		all_groups11.scale(0.7)
		all_groups11.move_to(LEFT * 22 / 4 + DOWN * 2.5)

		self.play(FadeIn(all_groups11))


		n, m = 4, 1
		all_groups_list12 = self.partition(n, m, ORIGIN, animate=False, animate_all=False)
		all_groups12 = VGroup(*all_groups_list12)
		all_groups12.scale(0.7)
		all_groups12.move_to(LEFT * 9 / 4 + DOWN * 2.5)

		self.play(FadeIn(all_groups12))


		n, m = 3, 1
		all_groups_list13 = self.partition(n, m, ORIGIN, animate=False, animate_all=False)
		all_groups13 = VGroup(*all_groups_list13)
		all_groups13.scale(0.7)
		all_groups13.move_to(RIGHT * 3 / 4 + DOWN * 2.5)

		self.play(FadeIn(all_groups13))


		n, m = 2, 1
		all_groups_list14 = self.partition(n, m, ORIGIN, animate=False, animate_all=False)
		all_groups14 = VGroup(*all_groups_list14)
		all_groups14.scale(0.7)
		all_groups14.move_to(RIGHT * 13 / 4 + DOWN * 2.5)

		self.play(FadeIn(all_groups14))


		n, m = 1, 1
		all_groups_list15 = self.partition(n, m, ORIGIN, animate=False, animate_all=False)
		all_groups15 = VGroup(*all_groups_list15)
		all_groups15.scale(0.7)
		all_groups15.move_to(RIGHT * 22 / 4 + DOWN * 2.5)

		self.play(FadeIn(all_groups15))

		self.wait(14)

		magnified_groups = VGroup(*[all_groups, all_groups2, all_groups3, all_groups4, 
			all_groups6, all_groups7, all_groups8, all_groups9])

		remaining_groups = VGroup(*[all_groups5, all_groups10, all_groups11, all_groups12, 
			all_groups13, all_groups14, all_groups15])

		shifted_maginfied_groups = magnified_groups.copy()
		shifted_maginfied_groups = shifted_maginfied_groups.move_to(ORIGIN)
		shifted_maginfied_groups.scale(1.1)

		self.play(Transform(magnified_groups, shifted_maginfied_groups), FadeOut(remaining_groups))
		self.wait(9)

		colors = [GREEN_SCREEN, YELLOW, BLUE, VIOLET]
		surrounding_rects = [SurroundingRectangle(group[3:-1], color=colors[i], 
			buff=SMALL_BUFF) for i, group in enumerate(shifted_maginfied_groups[4:])]

		
		top_surround_rects = [SurroundingRectangle(group[-1-len(bottom_group[3:-1]):-1], color=GREEN_SCREEN, 
			buff=SMALL_BUFF) for group, bottom_group in zip(shifted_maginfied_groups[:4], 
				shifted_maginfied_groups[4:])]

		for i, top_rect in enumerate(top_surround_rects):
			top_rect.set_color(colors[i])


		self.play(*[ShowCreation(rect) for rect in surrounding_rects])

		self.wait()
		self.play(*[TransformFromCopy(rect, top_rect) for rect, top_rect in zip(surrounding_rects, 
			top_surround_rects)])

		self.wait()

		observations = TextMobject(r"All $\text{count\_partitions}(n, m - 1)$ partitions are also in $\text{count\_partitions}(n, m)$", "?")
		observations.shift(UP * 3.5)
		observations.scale(0.7)
		self.play(Write(observations[0]), run_time=1.5)
		self.wait()
		red_observations = observations.copy()
		red_observations.set_color(RED)
		self.play(Write(observations[1]))
		self.play(Transform(observations, red_observations))
		self.wait(8)

		self.play(*[FadeOut(obj) for obj in magnified_groups] + 
			[FadeOut(rect) for rect in surrounding_rects + top_surround_rects])

		n, m = 7, 4
		all_groups_list = self.partition(n, m, LEFT * 3.5 + UP * 2.5, animate=False, animate_all=False)
		all_groups = VGroup(*all_groups_list)
		self.play(FadeIn(all_groups))
		self.wait(3)

		n, m = 7, 3
		all_groups_list2 = self.partition(n, m, RIGHT * 3.5 + UP * 2.5, animate=False)
		all_groups2 = VGroup(*all_groups_list2)

		remaining_partition_left = VGroup(*all_groups[:6])
		subset_partition_left = VGroup(*all_groups[6:])
		surround_rect_left = SurroundingRectangle(all_groups[6:-1], color=GREEN_SCREEN, buff=SMALL_BUFF)
		surround_rect_right = SurroundingRectangle(all_groups2[3:-1], color=GREEN_SCREEN, buff=SMALL_BUFF)
		self.wait()

		self.play(ShowCreation(surround_rect_left), ShowCreation(surround_rect_right))
		self.wait(4)

		conclusion = TextMobject("It seems like it!", "But why?")
		conclusion.scale(0.7)
		conclusion[0].set_color(GREEN_SCREEN)
		conclusion[1].set_color(BRIGHT_RED)
		conclusion.next_to(observations, DOWN * 0.5)
		self.play(Write(conclusion[0]))
		self.wait(6)
		self.play(Write(conclusion[1]))
		self.wait(26)

		subset = TextMobject(r"$\text{count\_partitions}(n, m - 1) \ \subseteq \text{count\_partitions}(n, m)$")
		subset.set_color(RED)
		subset.scale(0.7)
		subset.move_to(observations.get_center())
		self.play(FadeOut(conclusion), ReplacementTransform(observations, subset))
		self.wait(24) # going to probably have to increase this with new audio

		rest_question = TextMobject("Ok, but what about the remaining partitions?")
		rest_question.set_color(YELLOW)
		rest_question.scale(0.7)
		rest_question.move_to(subset.get_center())
		self.play(ReplacementTransform(subset, rest_question))

		self.wait(2)

		shifted_remaining_partition = remaining_partition_left.copy()
		shifted_remaining_partition.move_to(ORIGIN)

		self.play(Transform(remaining_partition_left, shifted_remaining_partition), 
			FadeOut(subset_partition_left), FadeOut(all_groups2), FadeOut(surround_rect_left),
			FadeOut(surround_rect_right), run_time=2)

		self.wait()

		scaled_remaining_partition = remaining_partition_left.copy()
		scaled_remaining_partition.scale(0.7)
		scaled_remaining_partition.move_to(RIGHT * 1.5 + DOWN * 1.5)

		top_partitions = magnified_groups[:3]
		shifted_maginfied_groups[0].move_to(LEFT * 4.5 + UP)
		shifted_maginfied_groups[1].move_to(RIGHT * 1.5 + UP)
		shifted_maginfied_groups[2].move_to(LEFT * 4.5 + DOWN * 2)

		top_fadeout_partitions = [VGroup(*group[-1-len(bottom_group[3:-1]):]) for group, bottom_group in zip(shifted_maginfied_groups[:4], 
				shifted_maginfied_groups[4:])][:3]

		top_surround_rects = [SurroundingRectangle(group[:-1], color=colors[i], 
			buff=SMALL_BUFF) for i, group in enumerate(top_fadeout_partitions)]

		top_remaining_partitions = [VGroup(*group[:-1-len(bottom_group[3:-1])]) for group, bottom_group in zip(shifted_maginfied_groups[:4], 
				shifted_maginfied_groups[4:])][:3]
		
		fadeins = [FadeIn(obj) for obj in top_fadeout_partitions + top_remaining_partitions] + [FadeIn(rect) for rect in top_surround_rects]
		self.play(*[ReplacementTransform(remaining_partition_left, scaled_remaining_partition)] + fadeins, run_time=2)

		self.play(*[FadeOut(rect) for rect in top_surround_rects[:3]] + 
			[FadeOut(partition) for partition in top_fadeout_partitions])


		common = TextMobject("What do these partitions have in common?")
		common.set_color(YELLOW)
		common.scale(0.7)
		common.move_to(subset.get_center())
		self.play(ReplacementTransform(rest_question, common))

		self.wait(3)
		top_left = top_remaining_partitions[0]
		top_right = top_remaining_partitions[1]
		bottom_left = top_remaining_partitions[2]
		bottom_right = scaled_remaining_partition
		top_left_indications = [Indicate(top_left[0][5:8]), Indicate(top_left[2]), 
			Indicate(top_left[3][:3]), Indicate(top_left[4][:3])]

		top_right_indications = [Indicate(top_right[0][4:7]), Indicate(top_right[2]),
			Indicate(top_right[3][:3])]
		bottom_left_indications = [Indicate(bottom_left[0][3:6]), Indicate(bottom_left[2]),
			Indicate(bottom_left[3][:3])]
		bottom_right_indications = [Indicate(bottom_right[0][7:11]), Indicate(bottom_right[2]),
			Indicate(bottom_right[3][:4]), Indicate(bottom_right[4][:4]), Indicate(bottom_right[5][:4])]

		indications = top_left_indications + top_right_indications + bottom_left_indications + bottom_right_indications
		use_m = TextMobject(r"They all use $m$ in the partition!")
		use_m.set_color(YELLOW)
		use_m.scale(0.7)
		use_m.move_to(common.get_center())

		self.play(*indications, run_time=1.5)
		self.play(ReplacementTransform(common, use_m))
		self.play(*indications, run_time=1.5)
		self.play(*indications, run_time=1.5)
		self.wait(29)

		remaining = TextMobject(r"And once we use that $m$, this is what we have left")
		remaining.set_color(YELLOW)
		remaining.scale(0.7)
		remaining.next_to(use_m, DOWN * 0.7)
		self.play(Write(remaining))
		self.wait()

		n, m = 2, 3
		new_top_left_list = self.partition(n, m, top_left.get_center(), animate=False, animate_all=False)
		new_top_left = VGroup(*new_top_left_list)
		new_top_left.scale(0.75)
		new_top_left.move_to(top_left.get_center() + RIGHT * 3 + DOWN * 0.2)
		self.play(TransformFromCopy(top_left, new_top_left))
		self.wait()

		n, m = 1, 3
		new_top_right_list = self.partition(n, m, top_right.get_center(), animate=False, animate_all=False)
		new_top_right = VGroup(*new_top_right_list)
		new_top_right.scale(0.75)
		new_top_right.next_to(top_right.get_center() + RIGHT * 2 + DOWN * 0.15)
		self.play(TransformFromCopy(top_right, new_top_right))
		self.wait()

		n, m = 0, 3
		new_bottom_left_list = self.partition(n, m, bottom_left.get_center(), animate=False, animate_all=False)
		
		new_bottom_left = VGroup(*new_bottom_left_list)
		new_bottom_left.scale(0.75)
		new_bottom_left[0][-1].next_to(new_bottom_left[0][0], LEFT * 0.5)
		new_bottom_left.next_to(bottom_left, RIGHT * 2)
		self.play(TransformFromCopy(bottom_left, new_bottom_left))
		self.wait()

		n, m = 3, 4
		new_bottom_right_list = self.partition(n, m, bottom_right.get_center(), animate=False, animate_all=False)
		new_bottom_right = VGroup(*new_bottom_right_list)
		new_bottom_right.scale(0.75)
		new_bottom_right.next_to(bottom_right.get_center() + RIGHT * 2.5 + DOWN * 0.2)

		self.play(TransformFromCopy(bottom_right, new_bottom_right))

		self.wait(7)

		top_left_m = top_left[-1][:3].copy()
		top_right_m = top_right[-1][:3].copy()
		bottom_left_m = bottom_left[-1][:3].copy()
		bottom_right_m = bottom_right[-1][:4].copy()

		top_left_start = new_top_left.get_center() + DOWN * 1.2
		arrow_top_left = Arrow(top_left_start, top_left_start + LEFT * 3)

		top_right_start = new_top_right.get_center() + DOWN * 1.2
		arrow_top_right = Arrow(top_right_start, top_right_start + LEFT * 3)

		bottom_left_start = new_bottom_left.get_center() + DOWN
		arrow_bottom_left = Arrow(bottom_left_start, bottom_left_start + LEFT * 3)

		bottom_right_start = new_bottom_right.get_center() + DOWN * 1.5
		arrow_bottom_right = Arrow(bottom_right_start, bottom_right_start + LEFT * 4)

		add_top_left = TextMobject("Add")
		add_top_left.scale(0.7)
		add_top_left.next_to(top_left_m, LEFT)
		caption_top_left = VGroup(add_top_left, top_left_m)
		caption_top_left.next_to(arrow_top_left, DOWN * 0.5)

		add_top_right = TextMobject("Add")
		add_top_right.scale(0.7)
		add_top_right.next_to(top_right_m, LEFT)
		caption_top_right = VGroup(add_top_right, top_right_m)
		caption_top_right.next_to(arrow_top_right, DOWN * 0.5)

		add_bottom_left = TextMobject("Add")
		add_bottom_left.scale(0.7)
		add_bottom_left.next_to(bottom_left_m, LEFT)
		caption_bottom_left = VGroup(add_bottom_left, bottom_left_m)
		caption_bottom_left.next_to(arrow_bottom_left, DOWN * 0.5)

		add_bottom_right = TextMobject("Add")
		add_bottom_right.scale(0.7)
		add_bottom_right.next_to(bottom_right_m, LEFT)
		caption_bottom_right = VGroup(add_bottom_right, bottom_right_m)
		caption_bottom_right.next_to(arrow_bottom_right, DOWN * 0.5)

		arrows = [arrow_bottom_right, arrow_bottom_left, arrow_top_right, arrow_top_left]
		arrow_anims = [GrowFromCenter(arr) for arr in arrows]
		captions = [caption_bottom_right, caption_bottom_left, caption_top_right, caption_top_left]
		caption_anims = [FadeIn(cap) for cap in captions]

		self.play(*arrow_anims+caption_anims, run_time=2)
		self.wait(10)
		# top_left_m.next_to(new_top_left, DOWN)
		# top_right_m.next_to(new_top_right, DOWN)
		# bottom_left_m.next_to(new_bottom_left, DOWN)
		# bottom_right_m.next_to(new_bottom_right, DOWN)
		# self.play(FadeIn(top_left_m), FadeIn(top_right_m),
		# 	FadeIn(bottom_left_m), FadeIn(bottom_right_m))



		fadeouts = [FadeOut(use_m), FadeOut(remaining), FadeOut(top_left), FadeOut(new_top_left),
			FadeOut(new_top_right), FadeOut(top_right), FadeOut(bottom_left), FadeOut(new_bottom_left),
			FadeOut(bottom_right), FadeOut(new_bottom_right)] + [FadeOut(arr) for arr in arrows] + [FadeOut(cap) for cap in captions]

		self.play(*fadeouts)


	def generalize(self):
		another_case = TextMobject("Let's try one more larger example to verify this pattern.")
		another_case.scale(0.9)
		another_case.shift(UP * 3)
		self.play(Write(another_case))
		self.wait(7)

		n, m = 9, 5
		all_groups_list = self.partition(n, m, UP * 5, animate=False, animate_all=False)
		all_groups = VGroup(*all_groups_list)
		partition_name_orig = all_groups[:3].copy()
		partition_name_orig.move_to(UP * 0.5)
		self.play(FadeIn(partition_name_orig))
		self.wait(5)
		all_groups.scale(0.55)
		all_groups.shift(UP + LEFT * 3.5)
		partition_name = all_groups[:3]
		self.play(ReplacementTransform(partition_name_orig, partition_name), FadeOut(another_case))
		self.wait(2)
		# self.play(FadeIn(all_groups[3:]))

		equals = TextMobject("=")
		equals.set_color(YELLOW)
		equals.scale(0.7)
		equals.next_to(partition_name, RIGHT)
		self.play(Write(equals))

		n, m = 4, 5
		all_groups_list2 = self.partition(n, m, UP * 2, animate=False, animate_all=False)
		all_groups2 = VGroup(*all_groups_list2)
		all_groups2.scale(0.55)
		all_groups2.shift(UP * 2 + RIGHT * 0.4)
		partition_name2 = all_groups2[:3]
		self.play(FadeIn(partition_name2))
		self.wait()

		plus = TextMobject("+")
		plus.set_color(YELLOW)
		plus.scale(0.7)
		plus.next_to(partition_name2, RIGHT)
		self.play(Write(plus))

		n, m = 9, 4
		all_groups_list3 = self.partition(n, m, UP * 2, animate=False, animate_all=False)
		all_groups3 = VGroup(*all_groups_list3)
		all_groups3.scale(0.55)
		all_groups3.shift(UP  * 3.5 + RIGHT * 4)
		partition_name3 = all_groups3[:3]
		self.play(FadeIn(partition_name3))
		self.wait(5)

		comment1 = TextMobject("These partitions",  r"represent using $m$")
		comment1.scale(0.6)
		comment1[0].next_to(partition_name2, DOWN * 0.8)
		comment1[1].next_to(comment1[0], DOWN * 0.5)
		comment2 = TextMobject("These partitions", "represent using", r"parts up to $m - 1$")
		comment2.scale(0.6)
		comment2[0].next_to(partition_name3, DOWN * 0.8)
		comment2[1].next_to(comment2[0], DOWN * 0.5)
		comment2[2].next_to(comment2[1], DOWN * 0.5)

		self.play(FadeIn(comment1))
		self.wait(5)
		self.play(FadeIn(comment2))

		self.wait(3)

		self.play(FadeOut(comment1), FadeOut(comment2))
		self.wait()
		self.play(FadeIn(all_groups2[3:]))
		self.wait(5)

		include_5_groups = all_groups[3:8]
		fadeins = [FadeIn(include_5_group[:6]) for include_5_group in include_5_groups]
		transforms = [TransformFromCopy(group2, include_5_group[6:]) for group2,
			include_5_group in zip(all_groups2[3:], include_5_groups)]
		self.play(*fadeins+transforms, run_time=2)
		self.wait(5)


		not_include_5_groups = all_groups[8:-1]
		self.play(FadeIn(all_groups3[3:]))
		self.wait(3)

		self.play(*[TransformFromCopy(group, not_include_5) for group, 
			not_include_5 in zip(all_groups3[3:-1], not_include_5_groups)], run_time=2)

		self.wait()

		self.play(FadeIn(all_groups[-1]))
		self.wait(10)

		step4 = TextMobject("4. Generalize the pattern")
		step4.scale(0.8)
		step4.set_color(ORANGE)
		step4.shift(DOWN * 3.5)
		self.play(FadeIn(step4))

		self.wait()



		n_a = TexMobject("n = a")
		n_b = TexMobject("m = b")
		n_a.scale(0.5)
		n_b.scale(0.5)
		n_a.set_color(YELLOW)
		n_b.set_color(YELLOW)
		n_a.move_to(all_groups[1].get_center())
		n_b.move_to(n_a.get_center()+ RIGHT + UP * SMALL_BUFF / 4)

		self.play(ReplacementTransform(all_groups[1], n_a), ReplacementTransform(all_groups[2], n_b))
		self.wait(2)

		cp_ab = TextMobject(r"$\text{count\_partitions}(a, b)$")
		cp_ab.scale(0.5)
		cp_ab.set_color(YELLOW)
		cp_ab.move_to(all_groups[-1].get_center())
		self.play(ReplacementTransform(all_groups[-1], cp_ab))
		self.wait(2)


		n_ab = TexMobject("n = a - b")
		n_b2 = TexMobject("m = b")
		n_ab.scale(0.5)
		n_b2.scale(0.5)
		n_ab.set_color(YELLOW)
		n_b2.set_color(YELLOW)
		n_ab.move_to(all_groups2[1].get_center() + LEFT * 0.2)
		n_b2.move_to(n_ab.get_center()+ RIGHT * 1.2)

		self.play(ReplacementTransform(all_groups2[1], n_ab), ReplacementTransform(all_groups2[2], n_b2))
		self.wait(2)

		cp_ab_b = TextMobject(r"$\text{count\_partitions}(a - b, b)$")
		cp_ab_b.scale(0.5)
		cp_ab_b.set_color(YELLOW)
		cp_ab_b.move_to(all_groups2[-1].get_center())
		self.play(ReplacementTransform(all_groups2[-1], cp_ab_b))
		self.wait(2)


		n_a2 = TexMobject("n = a")
		n_b_1 = TexMobject("m = b - 1")
		n_a2.scale(0.5)
		n_b_1.scale(0.5)
		n_a2.set_color(YELLOW)
		n_b_1.set_color(YELLOW)
		n_a2.move_to(all_groups3[1].get_center())
		n_b_1.move_to(n_a2.get_center() + RIGHT * 1.2 + UP * SMALL_BUFF / 3)

		self.play(ReplacementTransform(all_groups3[1], n_a2), ReplacementTransform(all_groups3[2], n_b_1))
		self.wait(2)

		cp_ab_1 = TextMobject(r"$\text{count\_partitions}(a, b - 1)$")
		cp_ab_1.scale(0.5)
		cp_ab_1.set_color(YELLOW)
		cp_ab_1.move_to(all_groups3[-1].get_center())
		self.play(ReplacementTransform(all_groups3[-1], cp_ab_1))
		self.wait(5)

		step5 = TextMobject("5. Write code by combining recursive pattern with the base case")
		step5.scale(0.7)
		step5.set_color(BLUE)
		step5.move_to(step4.get_center())
		self.play(ReplacementTransform(step4, step5))
		self.wait()

		old_symbols = [cp_ab, equals, cp_ab_b, plus, cp_ab_1]
		new_cp_ab = TextMobject(r"$\text{count\_partitions}(n, m)$")
		new_cp_ab.set_color(YELLOW)
		new_cp_ab.scale(0.5)
		new_cp_ab_b = TextMobject(r"$\text{count\_partitions}(n - m, m)$")
		new_cp_ab_b.set_color(YELLOW)
		new_cp_ab_b.scale(0.5)
		new_cp_ab_1 = TextMobject(r"$\text{count\_partitions}(n, m - 1)$")
		new_cp_ab_1.set_color(YELLOW)
		new_cp_ab_1.scale(0.5)
		new_equals = equals.copy()
		new_plus = plus.copy()


		new_symbols = [new_cp_ab, new_equals, new_cp_ab_b, new_plus, new_cp_ab_1]
		[symbol.scale(1.2) for symbol in new_symbols]

		new_cp_ab_b.move_to(UP * 2.5)
		new_equals.next_to(new_cp_ab_b, LEFT)
		new_plus.next_to(new_cp_ab_b, RIGHT)
		new_cp_ab_1.next_to(new_plus, RIGHT)
		new_cp_ab.next_to(new_equals, LEFT)

		new_symbols = [new_cp_ab, new_equals, new_cp_ab_b, new_plus, new_cp_ab_1]
		
		transforms = [ReplacementTransform(old_symbol, new_symbol) for old_symbol, new_symbol 
			in zip(old_symbols, new_symbols)]

		self.play(*[FadeOut(all_groups[:-1]), FadeOut(all_groups2[:-1]), FadeOut(all_groups3[:-1]),
			FadeOut(n_a), FadeOut(n_b), FadeOut(n_ab), FadeOut(n_b2), 
			FadeOut(n_a2), FadeOut(n_b_1)] + transforms, run_time=2)
		
		self.wait()

		general_base_casen0 = TextMobject(r"$1 \ \text{if} \ n = 0$")
		general_base_casen0.scale(0.6)
		general_base_casen0.set_color(YELLOW)
		general_base_casem0 = TextMobject(r"$0 \ \text{if} \ m = 0$")
		general_base_casem0.scale(0.6)
		general_base_casem0.set_color(YELLOW)

		rhs = VGroup(*new_symbols[2:])
		lhs = VGroup(*new_symbols[:2])
		lhs_shifted_down = lhs.copy()
		lhs_shifted_down.shift(LEFT * 0.2)
		rhs_shifted_down = rhs.copy()
		rhs_shifted_down.shift(DOWN * 0.4 + RIGHT * 0.2)
		general_base_casem0.next_to(rhs_shifted_down, UP)
		general_base_casen0.next_to(general_base_casem0, UP)

		entire_rhs = VGroup(general_base_casen0, general_base_casem0, rhs_shifted_down)
		brace = Brace(entire_rhs, LEFT, buff=SMALL_BUFF)
		self.play(ReplacementTransform(rhs, rhs_shifted_down), ReplacementTransform(lhs, lhs_shifted_down),
			GrowFromCenter(brace), FadeIn(general_base_casen0), FadeIn(general_base_casem0))

		self.wait(13)

		n_m = new_cp_ab_b[-6:-3]
		lower_brace = Brace(n_m, DOWN, buff=SMALL_BUFF)
		question = TextMobject(r"What if $n < m$", "?")
		question.next_to(lower_brace, DOWN * 0.5)
		question.scale(0.6)

		self.play(GrowFromCenter(lower_brace), FadeIn(question))

		self.wait(10)

		base_case0 = TextMobject(r"$0 \ \text{if} \ m = 0 \ \text{or} \ n < 0$")
		base_case0.set_color(YELLOW)
		base_case0.scale(0.6)
		base_case0.move_to(general_base_casem0.get_center())
		self.play(ReplacementTransform(general_base_casem0, base_case0), FadeOut(lower_brace), FadeOut(question))

		self.wait(19)

		self.play(FadeOut(step5), FadeOut(entire_rhs), FadeOut(lhs_shifted_down),
			FadeOut(base_case0), FadeOut(brace))

	def partition(self, n, m, top_partition_position, animate=True, animate_all=True, run_time=1):
		partitions = get_partitions(n, m)

		colors = {1: ORANGE, 2: BRIGHT_RED, 3: GREEN_SCREEN, 4: VIOLET, 5: GOLD, 6: TEAL}
		radius = 0.1
		top_partition = [Dot(radius=radius) for i in range(n + m)]
		start_position = LEFT * 3
		offset = RIGHT * 0.5
		for i, dot in enumerate(top_partition):
			dot.move_to(start_position)
			if i < n:
				dot.set_color(BLUE)
			else:
				dot.set_color(colors[m])
			start_position += offset

		line_start = (top_partition[n - 1].get_center() + 
			top_partition[n].get_center()) / 2 + UP * radius * 2
		line_end = (top_partition[n - 1].get_center() + 
			top_partition[n].get_center()) / 2 + DOWN * radius * 2
		line = Line(line_start, line_end)
		top_partition.append(line)

		top_partition_group = VGroup(*top_partition)
		top_partition_group.move_to(top_partition_position)
		if animate_all:
			self.play(FadeIn(top_partition_group))
		label_n = TexMobject("n = {0}".format(n)) 
		label_m = TexMobject("m = {0}".format(m))
		label_n.set_color(YELLOW)
		label_n.scale(0.7)
		label_n.move_to(top_partition_position + UP * 0.5+ LEFT * 0.8)
		label_m.set_color(YELLOW)
		label_m.scale(0.7)
		label_m.move_to(top_partition_position + UP * 0.5 + RIGHT * 0.8)
		if animate_all:
			self.play(FadeIn(label_n), FadeIn(label_m))
		down_offset = DOWN * 0.5
		all_groups = []
		all_groups.append(top_partition_group)
		all_groups.extend([label_n, label_m])
		for i in range(len(partitions)):
			partition, partition_group = self.make_partition(partitions[i], 
				top_partition_group.get_center() + down_offset * (i + 1), colors, offset)
			if not animate:
				if animate_all:
					self.play(FadeIn(partition_group))
			else:
				splits = self.split_partition(partition)
				self.display_partition(top_partition, partition, splits, run_time=run_time)
			all_groups.append(partition_group)

		partition_count = TextMobject("{0} partitions".format(len(partitions)))
		partition_count.set_color(YELLOW)
		partition_count.scale(0.7)
		partition_count.next_to(VGroup(*all_groups), DOWN)
		if animate_all:
			self.play(FadeIn(partition_count))
		all_groups.append(partition_count)

		return all_groups

	def make_partition(self, partition, position, colors, offset):
		radius = 0.1
		partition_objects = []
		color_list = []
		for i, p in enumerate(partition):
			color_list.append(colors[p])
			for _ in range(p):
				partition_objects.append(Dot(radius=radius))
			if i < len(partition) - 1:
				partition_objects.append(TexMobject('+'))
		start_position = LEFT * 5
		# offset = RIGHT * 0.8
		index = 0
		for i in range(len(partition_objects)):
			obj = partition_objects[i]
			if isinstance(obj, TexMobject):
				obj.set_color(YELLOW)
				index += 1
			else:
				obj.set_color(color_list[index])
			obj.move_to(start_position)
			start_position += offset
		group = VGroup(*partition_objects)
		group.move_to(position)
		return partition_objects, group

	def split_partition(self, partition_objects):
		splits = []
		split = []
		for obj in partition_objects:
			if isinstance(obj, Dot):
				split.append(obj)
			else:
				splits.append(split)
				split = []

		splits.append(split)
		return splits

	def display_partition(self, top_partition, partition, splits, run_time=1):
		index = 0
		split_index = 0
		for split in splits:
			top_split = top_partition[split_index:split_index+len(split)]
			self.play(*[TransformFromCopy(s, s_copy, run_time=run_time) for s, s_copy in zip(top_split, split)])
			index += len(split)
			split_index += len(split)
			if index < len(partition):
				self.play(Write(partition[index]), run_time=run_time)
				index += 1




def get_partitions(n, m):
	if n == 0:
		return [[]]
	elif n < 0:
		return []
	elif m == 0:
		return []
	else:
		use_m = [[m] + partition for partition in get_partitions(n - m, m)]
		dont_use_m = get_partitions(n, m - 1)
		return use_m + dont_use_m


class RecursiveLeapIntro(Scene):
	def construct(self):
		scale = 0.7
		faith = TextMobject("Recursive Leap of Faith")
		faith.move_to(UP * 3)
		self.play(Write(faith))
		self.wait(4)

		assume = TextMobject("Assume simpler cases work out")
		assume.scale(0.8)
		assume.set_color(YELLOW)
		assume.next_to(faith, DOWN)

		self.play(Write(assume))
		self.wait(5)

		triangle5_incomp = self.make_incomplete_triangle(5, 0.5, GREEN, position=DOWN * 0.2)
		sum5 = TextMobject(r"sum($5$)")
		sum5.set_color(BLUE)
		sum5.next_to(triangle5_incomp, UP)
		sum5.scale(scale)
		self.play(FadeIn(sum5))
		incomplete5 = triangle5_incomp[:4]
		exp5 = TextMobject(r"$5 + \text{sum}(4)$")
		exp5.next_to(triangle5_incomp, DOWN)
		exp5.scale(scale)
		exp5[0].set_color(BLUE)
		exp5[1].set_color(GOLD)
		exp5[2:].set_color(GREEN)
		self.wait()
		self.play(FadeIn(triangle5_incomp[-1]), Write(exp5[0]))

		self.wait()
		self.play(Write(exp5[1]))
		self.play(ShowCreation(incomplete5), Write(exp5[2:]))
		self.wait()

		evaluated = TextMobject("We now assume this works")
		evaluated.scale(0.8)
		evaluated.set_color(GREEN)
		evaluated.shift(LEFT * 0.5 + RIGHT * 4.5 + DOWN * 0.2)
		arrow1 = Arrow(evaluated.get_center() + LEFT * 2.2,
			triangle5_incomp[2][-1].get_center() + RIGHT * 0.2)
		arrow2 = Arrow(evaluated.get_center() + LEFT * 1.5,
			exp5[-1].get_center())
		arrow1.set_color(GREEN)
		arrow2.set_color(GREEN)

		self.play(FadeIn(evaluated), ShowCreation(arrow1), ShowCreation(arrow2))
		self.wait()

		exp5_10 = TexMobject("5 + 10")
		exp5_10.next_to(exp5, DOWN)
		exp5_10.scale(scale)
		exp5_10[0].set_color(BLUE)
		exp5_10[1].set_color(GOLD)
		exp5_10[2:].set_color(GREEN)
		self.play(TransformFromCopy(exp5, exp5_10))
		self.wait()
		fifteen = TexMobject("15")
		fifteen.next_to(exp5_10, DOWN)
		fifteen.scale(scale)
		fifteen.set_color(BLUE)
		self.play(TransformFromCopy(exp5_10, fifteen))
		self.wait(11)

		to_fadeout = [faith, assume, triangle5_incomp, sum5, exp5_10,
			exp5, fifteen, evaluated, arrow1, arrow2]
		self.play(*[FadeOut(obj) for obj in to_fadeout])


		

	def make_incomplete_triangle(self, n, side_length, other_color, color=BLUE, fill_color=BLUE, 
		fill_opacity=0.5, position=ORIGIN):
		triangle = []
		for i in range(n):
			if i == 0:
				row = self.make_row(i + 1, side_length, color=other_color, 
					fill_color=other_color, fill_opacity=fill_opacity)
			else:
				if i == n - 1:
					row = self.make_row(i + 1, side_length, color=color, 
						fill_color=fill_color, fill_opacity=fill_opacity, prev_row=triangle[i - 1], buff=SMALL_BUFF / 2)
				# elif i == n - 2:
				# 	row = self.make_row(i + 1, side_length, color=other_color, 
				# 		fill_color=BLACK, fill_opacity=fill_opacity, 
				# 		prev_row=triangle[i - 1], buff=SMALL_BUFF)
				else:
					row = self.make_row(i + 1, side_length, color=other_color, 
						fill_color=other_color, fill_opacity=fill_opacity, prev_row=triangle[i - 1])
			triangle.append(row)

		triangle_group = VGroup(*triangle)
		triangle_group.move_to(position)
		return triangle_group
	
	def make_row(self, n, side_length, color=BLUE, fill_color=BLUE, 
		fill_opacity=0.5, prev_row=None, buff=0):
		rectangles = []
		rectangles.append(Square(side_length=side_length, color=color, 
			fill_color=fill_color, fill_opacity=fill_opacity))
		if prev_row:
			rectangles[0].next_to(prev_row[0], DOWN * side_length, buff=buff)
		for i in range(1, n):
			next_rect = Square(side_length=side_length, color=color, 
				fill_color=fill_color, fill_opacity=fill_opacity)
			next_rect.next_to(rectangles[i - 1], RIGHT * side_length, buff=0)
			rectangles.append(next_rect)

		return VGroup(*rectangles)

class Summation(Scene):
	def construct(self):

		self.wait()

		problem_line1 = TextMobject(r"Write a recursive function that given an input $n$")
		problem_line2 =  TextMobject(r"sums all nonnegative integers up to $n$.")
		problem_line1.shift(UP * 3)
		problem_line2.shift(UP * 2.5)
		problem_line1.scale(0.9)
		problem_line2.scale(0.9)
		self.play(Write(problem_line1, run_time=3))

		self.play(Write(problem_line2, run_time=3))

		self.wait(4)

		case0 = TextMobject("sum(0)", r"$\rightarrow$", "0")
		case0.set_color(ORANGE)
		case0.scale(0.9)

		case0.shift(UP * 1.5)


		case1 = TextMobject(r"sum($1$)", r"$\rightarrow$", r"$1$")
		case1.scale(0.9)
		case1.set_color(YELLOW)

		case1.shift(UP * 0.5)
		for i in range(len(case1)):
			self.play(Write(case0[i]) , Write(case1[i]))


		case4 = TextMobject(r"sum($4$)", r"$\rightarrow$", r"$(1 + 2 + 3 + 4)$", r"$\rightarrow$", r"$10$")
		case4.scale(0.9)
		case4.set_color(GREEN)

		case4.shift(DOWN * 0.5)



		casek = TextMobject(r"sum($n$)", r"$\rightarrow$", r"$(1 + 2 + 3 + \ldots + n)$")
		casek.scale(0.9)
		casek.set_color(BLUE)

		casek.shift(DOWN * 1.5)
		for i in range(len(casek)):
			self.play(Write(casek[i]), Write(case4[i]))
		self.play(Write(case4[-2]), Write(case4[-1]))
		self.wait()

		self.play(FadeOut(case0), FadeOut(case1), FadeOut(case4), FadeOut(casek))

		self.wait()
		# TODO: put animations for Iterative and mathematical versions of code

		rect_left = ScreenRectangle()
		rect_left.set_width(4.5)
		rect_left.set_height(3.5)
		rect_left.move_to(LEFT * 3.7 + DOWN)

		rect_right = ScreenRectangle()
		rect_right.set_width(4.5)
		rect_right.set_height(3.5)
		rect_right.move_to(RIGHT * 3.7 + DOWN)

		

		iteration = TextMobject("Iteration")
		iteration.scale(0.9)
		iteration.next_to(rect_left, DOWN)

		mathematical = TextMobject("Mathematical")
		mathematical.scale(0.9)
		mathematical.next_to(rect_right, DOWN)

		self.play(ShowCreation(rect_left), ShowCreation(rect_right) ,
			FadeIn(iteration), FadeIn(mathematical))
		
		self.wait(7)
		question = TextMobject("How would you solve it recursively?")
		question.scale(0.8)
		question.next_to(problem_line2, DOWN)
		question.set_color(YELLOW)
		self.play(Write(question), run_time=2)
		self.wait(9)

		self.play(FadeOut(iteration), FadeOut(mathematical),
			FadeOut(rect_left), FadeOut(rect_right), FadeOut(question))

		self.wait()


		step1 = TextMobject("1. What's the simplest possible input?")
		step1.scale(0.9)
		step1.shift(UP * 1.8)
		step1.set_color(RED)
		self.play(FadeIn(step1))

		self.wait(5)

		case0.next_to(step1, DOWN)
		self.play(FadeIn(case0))

		self.wait(7)

		brace = Brace(case0, DOWN, buff=SMALL_BUFF)
		base_case = TextMobject("Base Case")
		base_case.next_to(brace, DOWN)
		self.play(GrowFromCenter(brace), FadeIn(base_case))

		self.wait(10)

		self.play(FadeOut(case0), FadeOut(step1), FadeOut(brace), FadeOut(base_case))

		self.wait(3)

		step2 = TextMobject("2. Play around with examples and visualize!")
		step2.scale(0.9)
		step2.shift(UP * 1.8)
		step2.set_color(YELLOW)
		self.play(FadeIn(step2))

		self.wait(3)

		all_text = []
		triangle1 = self.make_triangle(1, 0.5, color=YELLOW, fill_color=YELLOW, 
			fill_opacity=0.5, position=LEFT * 5.5 + DOWN)

		scale = 0.7
		n_1 = TexMobject("n = 1")
		one = TexMobject("1")
		n_1.scale(scale)
		one.scale(scale)
		n_1.next_to(triangle1, UP)
		one.next_to(triangle1, DOWN)
		self.play(FadeIn(triangle1))
		self.play(FadeIn(n_1))
		self.play(FadeIn(one))
		self.wait()
		all_text.extend([n_1, one])

		triangle2 = self.make_triangle(2, 0.5, color=ORANGE, fill_color=ORANGE, 
			fill_opacity=0.5, position=LEFT * 3 + DOWN)

		n_2 = TexMobject("n = 2")
		exp2 = TexMobject("(1 + 2)")
		exp2.set_color(ORANGE)
		three = TexMobject("3")
		exp2.scale(scale)
		n_2.scale(scale)
		three.scale(scale)
		n_2.next_to(triangle2, UP)
		exp2.next_to(triangle2, DOWN)
		three.next_to(exp2, DOWN)

		all_text.extend([n_2, exp2, three])

		self.play(FadeIn(triangle2))
		self.play(FadeIn(n_2))
		self.play(FadeIn(exp2))
		self.play(TransformFromCopy(exp2, three))

		self.wait()

		triangle3 = self.make_triangle(3, 0.5, color=RED, fill_color=RED, 
			fill_opacity=0.5, position=LEFT * 0.5 + DOWN)
		
		n_3 = TexMobject("n = 3")
		exp3 = TexMobject("(1 + 2 + 3)")
		exp3.set_color(RED)
		six = TexMobject("6")
		exp3.scale(scale)
		n_3.scale(scale)
		six.scale(scale)
		n_3.next_to(triangle3, UP)
		exp3.next_to(triangle3, DOWN)
		six.next_to(exp3, DOWN)

		all_text.extend([n_3, exp3, six])

		self.play(FadeIn(triangle3))
		self.play(FadeIn(n_3))
		self.play(FadeIn(exp3))
		self.play(TransformFromCopy(exp3, six))

		self.wait()

		triangle4 = self.make_triangle(4, 0.5, color=GREEN, fill_color=GREEN, 
			fill_opacity=0.5, position=RIGHT * 2 + DOWN)

		n_4 = TexMobject("n = 4")
		exp4 = TexMobject("(1 + 2 + 3 + 4)")
		exp4.set_color(GREEN)
		ten = TexMobject("10")
		exp4.scale(scale)
		n_4.scale(scale)
		ten.scale(scale)
		n_4.next_to(triangle4, UP)
		exp4.next_to(triangle4, DOWN)
		ten.next_to(exp4, DOWN)

		all_text.extend([n_4, exp4, ten])

		self.play(FadeIn(triangle4))
		self.play(FadeIn(n_4))
		self.play(FadeIn(exp4))
		self.play(TransformFromCopy(exp4, ten))
		self.wait()

		triangle5 = self.make_triangle(5, 0.5, position=RIGHT * 5 + DOWN)
		
		n_5 = TexMobject("n = 5")
		exp5 = TexMobject("(1 + 2 + 3 + 4 + 5)")
		exp5.set_color(BLUE)
		fifteen = TexMobject("15")
		exp5.scale(scale)
		n_5.scale(scale)
		fifteen.scale(scale)
		n_5.next_to(triangle5, UP)
		exp5.next_to(triangle5, DOWN)
		fifteen.next_to(exp5, DOWN)

		all_text.extend([n_5, exp5, fifteen])

		self.play(FadeIn(triangle5))
		self.play(FadeIn(n_5))
		self.play(FadeIn(exp5))
		self.play(TransformFromCopy(exp5, fifteen))
		self.wait()

		step3 = TextMobject("3. Relate hard cases to simpler cases")
		step3.set_color(GREEN_SCREEN)
		step3.scale(0.9)
		step3.shift(UP * 1.8)
		self.play(FadeOut(step2))
		self.play(FadeIn(step3))

		self.wait(16)

		commentary1 = TextMobject("Can you relate sum(4) and sum(5)?")
		commentary1.scale(0.8)
		commentary1[12:18].set_color(GREEN)
		commentary1[21:-1].set_color(BLUE)
		commentary1.next_to(step3, DOWN)
		self.play(FadeIn(commentary1))

		self.wait(4)


		commentary2 = TextMobject("Can you relate sum(3) and sum(4)?")
		commentary2.scale(0.8)
		commentary2[12:18].set_color(RED)
		commentary2[21:-1].set_color(GREEN)
		commentary2.next_to(step3, DOWN)
		self.play(ReplacementTransform(commentary1, commentary2))

		self.wait(7)

		all_text.extend([step3, commentary2])

		sub_triangle4 = triangle5[:4]
		sub_triangle4_copy = sub_triangle4.copy()
		sub_triangle4_copy.set_color(GREEN)

		surround_sub_tri4 = SurroundingRectangle(sub_triangle4, color=GREEN_SCREEN, buff=SMALL_BUFF)
		surround_tri4 = SurroundingRectangle(triangle4, color=GREEN_SCREEN, buff=SMALL_BUFF)
		self.play(ShowCreation(surround_sub_tri4), ShowCreation(surround_tri4))
		self.wait()

		sub_exp4 = exp5[1:-3].copy()
		sub_exp4.set_color(GREEN)

		self.play(TransformFromCopy(triangle4, sub_triangle4_copy), TransformFromCopy(exp4[1:-1], sub_exp4))

		self.wait(3)

		self.play(FadeOut(surround_tri4), FadeOut(surround_sub_tri4), 
			FadeOut(sub_triangle4_copy),  FadeOut(sub_exp4))


		self.wait(4)

		sub_triangle3 = triangle4[:3]
		sub_triangle3_copy = sub_triangle3.copy()
		sub_triangle3_copy.set_color(RED)

		surround_sub_tri3 = SurroundingRectangle(sub_triangle3, color=GREEN_SCREEN, buff=SMALL_BUFF)
		surround_tri3 = SurroundingRectangle(triangle3, color=GREEN_SCREEN, buff=SMALL_BUFF)
		self.play(ShowCreation(surround_sub_tri3), ShowCreation(surround_tri3))
		self.wait()

		sub_exp3 = exp4[1:-3].copy()
		sub_exp3.set_color(RED)

		self.play(TransformFromCopy(triangle3, sub_triangle3_copy), TransformFromCopy(exp3[1:-1], sub_exp3))

		self.wait(3)

		self.play(FadeOut(surround_tri3), FadeOut(surround_sub_tri3), FadeOut(sub_triangle3_copy), FadeOut(sub_exp3))


		self.wait(10)

		all_objects = [triangle5, triangle4, triangle3, triangle2, triangle1] + all_text
		self.play(*[FadeOut(obj) for obj in all_objects])

		objects = []
		text = []

		step4 = TextMobject("4. Generalize the pattern")
		step4.scale(0.9)
		step4.set_color(ORANGE)
		step4.shift(UP * 1.8)
		self.play(FadeIn(step4))

		self.wait(2)

		trianglek = self.make_triangle(10, 0.5, position=LEFT * 2.5 + DOWN)
		trianglek.scale(0.6)
		left_corner = trianglek[-1][0]
		right_corner = trianglek[-1][-1]

		objects.append(trianglek)

		line  = Line(left_corner.get_center() + LEFT * 0.5 * left_corner.get_width(),
			right_corner.get_center() + RIGHT * 0.5 * right_corner.get_width())
		line.shift(DOWN * 0.3)
		objects.append(line)
		self.play(FadeIn(trianglek))
		self.play(GrowFromCenter(line))
		k = TexMobject("k")
		k.scale(scale)
		k.next_to(line, DOWN * 0.5)
		self.play(FadeIn(k))
		n_k = TexMobject("n = k")
		expk = TexMobject(r"(1 + 2 + \ldots + k)")
		expk.set_color(BLUE)
		sum_k = TextMobject(r"sum($k$)")
		expk.scale(scale)
		n_k.scale(scale)
		sum_k.scale(scale)
		n_k.next_to(trianglek, UP)
		expk.next_to(k, DOWN * 0.5)
		sum_k.next_to(expk, DOWN * 0.5)
		self.play(FadeIn(n_k))
		self.play(FadeIn(expk))
		self.play(TransformFromCopy(expk, sum_k))
		text.extend([k, n_k, expk])
		self.wait()

		equals = TexMobject("=")
		equals.set_color(YELLOW)
		equals.next_to(trianglek, RIGHT, buff=SMALL_BUFF)
		self.play(Write(equals))


		subset_k_1 = trianglek[:9].copy()
		subset_k_1.set_color(GREEN)
		subset_k_1.shift(RIGHT * 4)
		self.play(TransformFromCopy(trianglek[:9], subset_k_1))
		objects.append(subset_k_1)
		left_corner = subset_k_1[-1][0]
		right_corner = subset_k_1[-1][-1]
		line_k_1  = Line(left_corner.get_center() + LEFT * 0.5 * left_corner.get_width(),
			right_corner.get_center() + RIGHT * 0.5 * right_corner.get_width())

		objects.append(line_k_1)
		line_k_1.shift(DOWN * 0.3)
		k_1 = TexMobject("k - 1")
		k_1.scale(scale)
		k_1.next_to(line_k_1, DOWN * 0.5)
		self.play(TransformFromCopy(line, line_k_1), TransformFromCopy(k, k_1))

		n_k_1 = TexMobject("n = k - 1")
		expk_1 = TexMobject(r"(1 + 2 + \ldots + (k - 1))")
		expk_1.set_color(GREEN)
		sum_k_1 = TextMobject(r"sum($k - 1$)")
		expk_1.scale(scale)
		n_k_1.scale(scale)
		sum_k_1.scale(scale)
		n_k_1.next_to(subset_k_1, UP)
		expk_1.next_to(k_1, DOWN * 0.5)
		sum_k_1.next_to(expk_1, DOWN * 0.5)
		self.play(FadeIn(n_k_1))
		self.play(FadeIn(expk_1))
		self.play(TransformFromCopy(expk_1, sum_k_1))
		text.extend([n_k_1, expk_1, k_1])

		plus = TexMobject("+")
		plus.set_color(YELLOW)
		plus.next_to(subset_k_1, RIGHT, buff=SMALL_BUFF)
		self.play(Write(plus))

		subset_row = trianglek[-1].copy()
		subset_row.next_to(plus, RIGHT * 1.5)
		
		objects.append(subset_row)

		line_copy = line.copy()
		line_copy.next_to(subset_row, DOWN * 0.6)
		k_copy = k.copy()
		k_copy.next_to(line_copy, DOWN * 0.5)
		objects.append(line_copy)
		self.play(TransformFromCopy(trianglek[-1], subset_row), TransformFromCopy(line, line_copy), TransformFromCopy(k, k_copy))
		self.wait(3)


		step5 = TextMobject("5. Write code by combining recursive pattern with the base case")
		step5.scale(0.9)
		step5.set_color(BLUE)
		step5.shift(UP * 1.8)
		self.play(ReplacementTransform(step4, step5))

		self.wait(2)

		old_symbols = [sum_k, equals, sum_k_1, plus, k_copy]
		new_sum_k_1 = TextMobject(r"sum($n - 1$)")
		new_sum_k_1.scale(scale)
		new_sum_k = TextMobject(r"sum($n$)")
		new_sum_k.scale(scale)
		new_equals = equals.copy()
		new_plus = plus.copy()
		new_k = TextMobject(r"$n$")
		new_k.scale(scale)

		new_sum_k_1.next_to(step5, DOWN * 2)
		new_equals.next_to(new_sum_k_1, LEFT)
		new_plus.next_to(new_sum_k_1, RIGHT)
		new_k.next_to(new_plus, RIGHT)
		new_sum_k.next_to(new_equals, LEFT)

		new_symbols = [new_sum_k, new_equals, new_sum_k_1, new_plus, new_k]
		transforms = [ReplacementTransform(old_symbol, new_symbol) for old_symbol, new_symbol 
			in zip(old_symbols, new_symbols)]
		
		fadeouts = [FadeOut(obj) for obj in objects + text]

		self.play(*fadeouts + transforms, run_time=2)
		self.wait()

		base_case_exp = TextMobject(r"$0$ if $n = 0$")
		base_case_exp.scale(scale)
		rhs = VGroup(*new_symbols[2:])
		lhs = VGroup(*new_symbols[:2])
		lhs_shifted_down = lhs.copy()
		lhs_shifted_down.shift(DOWN * 0.2 + LEFT * 0.2)
		rhs_shifted_down = rhs.copy()
		rhs_shifted_down.shift(DOWN * 0.5 + RIGHT * 0.2)
		base_case_exp.next_to(rhs_shifted_down, UP)

		entire_rhs = VGroup(base_case_exp, rhs_shifted_down)
		brace = Brace(entire_rhs, LEFT, buff=SMALL_BUFF)
		self.play(ReplacementTransform(rhs, rhs_shifted_down), ReplacementTransform(lhs, lhs_shifted_down),
			GrowFromCenter(brace), FadeIn(base_case_exp))

		# code_rect = ScreenRectangle()
		# code_rect.set_width(5)
		# code_rect.set_height(3.5)
		# code_rect.move_to(DOWN * 2)
		# self.play(FadeIn(code_rect))
		self.wait(5)

		self.play(FadeOut(rhs_shifted_down), FadeOut(lhs_shifted_down), FadeOut(brace), 
			FadeOut(base_case_exp), FadeOut(step5), FadeOut(problem_line1), 
			FadeOut(problem_line2))

		code_work = TextMobject("How does this code work?")
		code_work.scale(0.9)
		code_work.shift(UP * 3.5)
		self.play(Write(code_work))
		self.wait(5)

		triangle5_incomp = self.make_incomplete_triangle(5, 0.5, GREEN, position=LEFT * 5)
		sum5 = TextMobject(r"sum($5$)")
		sum5.set_color(BLUE)
		sum5.next_to(triangle5_incomp, UP)
		sum5.scale(scale)
		self.play(FadeIn(sum5))
		incomplete5 = triangle5_incomp[:4]
		exp5 = TextMobject(r"$5 + \text{sum}(4)$")
		exp5.next_to(triangle5_incomp, DOWN)
		exp5.scale(scale)
		exp5[0].set_color(BLUE)
		exp5[1].set_color(GOLD)
		exp5[2:].set_color(GREEN)
		self.wait(9)
		self.play(FadeIn(triangle5_incomp[-1]), Write(exp5[0]))

		self.wait(3)
		self.play(Write(exp5[1]))
		self.play(ShowCreation(incomplete5), Write(exp5[2:]))
		self.wait(7)

		evaluated = TextMobject("This hasn't actually been evaluated yet")
		evaluated.scale(0.8)
		evaluated.set_color(GREEN)
		evaluated.shift(LEFT * 0.5)
		arrow1 = Arrow(evaluated.get_center() + LEFT * 3.2,
			triangle5_incomp[2][-1].get_center() + RIGHT * 0.2)
		arrow2 = Arrow(evaluated.get_center() + LEFT * 1.5,
			exp5[-1].get_center())
		arrow1.set_color(GREEN)
		arrow2.set_color(GREEN)

		self.play(FadeIn(evaluated), ShowCreation(arrow1), ShowCreation(arrow2))
		self.wait()

		evaluate_it = TextMobject("Let's see what happens when we evaluate it")
		evaluate_it.scale(0.8)
		evaluate_it.set_color(GREEN)
		evaluate_it.move_to(evaluated.get_center() + RIGHT * 0.5)

		self.play(Transform(evaluated, evaluate_it))
		self.wait(2)

		self.play(FadeOut(evaluated), FadeOut(arrow1), FadeOut(arrow2))
		self.wait()

		triangle4_incomp = self.make_incomplete_triangle(4, 0.5, RED, color=GREEN, fill_color=GREEN, 
			fill_opacity=0.5, position=LEFT * 2)
		sum4 = exp5[2:].copy()
		sum4.next_to(triangle4_incomp, UP)
		self.play(TransformFromCopy(exp5[2:], sum4))
		incomplete4 = triangle4_incomp[:3]
		exp4 = TextMobject(r"$4 + \text{sum}(3)$")
		exp4.next_to(triangle4_incomp, DOWN)
		exp4.scale(scale)
		exp4[0].set_color(GREEN)
		exp4[1].set_color(GOLD)
		exp4[2:].set_color(RED)
		self.wait(4)
		self.play(FadeIn(triangle4_incomp[-1]), Write(exp4[0]))
		self.wait(2)
		self.play(Write(exp4[1]))
		self.play(ShowCreation(incomplete4), Write(exp4[2:]))
		self.wait(3)

		same_process = TextMobject("Same process as before.")
		same_process.shift(RIGHT + UP * 0.3)
		same_process.scale(0.8)
		same_process.set_color(RED)
		arrow1 = Arrow(same_process.get_center() + LEFT * 2,
			triangle4_incomp[1][-1].get_center() + RIGHT * 0.2)
		arrow2 = Arrow(same_process.get_center() + LEFT * 0.5,
			exp4[-1].get_center())
		arrow1.set_color(RED)
		arrow2.set_color(RED)

		self.play(FadeIn(same_process), ShowCreation(arrow1), ShowCreation(arrow2))
		self.wait()
		self.play(FadeOut(same_process), FadeOut(arrow1), FadeOut(arrow2))


		triangle3_incomp = self.make_incomplete_triangle(3, 0.5, ORANGE, color=RED, fill_color=RED, 
			fill_opacity=0.5, position=RIGHT * 0.5)
		sum3 = exp4[2:].copy()
		sum3.next_to(triangle3_incomp, UP)
		self.play(TransformFromCopy(exp4[2:], sum3))
		incomplete3 = triangle3_incomp[:2]
		exp3 = TextMobject(r"$3 + \text{sum}(2)$")
		exp3.next_to(triangle3_incomp, DOWN)
		exp3.scale(scale)
		exp3[0].set_color(RED)
		exp3[1].set_color(GOLD)
		exp3[2:].set_color(ORANGE)
		self.play(FadeIn(triangle3_incomp[-1]), Write(exp3[0]))
		self.play(Write(exp3[1]))
		self.play(ShowCreation(incomplete3), Write(exp3[2:]))

		triangle2_incomp = self.make_incomplete_triangle(2, 0.5, YELLOW, color=ORANGE, fill_color=ORANGE, 
			fill_opacity=0.5, position=RIGHT * 3)
		sum2 = exp3[2:].copy()
		sum2.next_to(triangle2_incomp, UP)
		self.play(TransformFromCopy(exp3[2:], sum2))
		incomplete2 = triangle2_incomp[:1]
		exp2 = TextMobject(r"$2 + \text{sum}(1)$")
		exp2.next_to(triangle2_incomp, DOWN)
		exp2.scale(scale)
		exp2[0].set_color(ORANGE)
		exp2[1].set_color(GOLD)
		exp2[2:].set_color(YELLOW)
		self.play(FadeIn(triangle2_incomp[-1]), Write(exp2[0]))
		self.play(Write(exp2[1]))
		self.play(ShowCreation(incomplete2), Write(exp2[2:]))

		triangle1_incomp = self.make_triangle(1, 0.5, color=YELLOW, fill_color=YELLOW, 
			fill_opacity=0.5, position=RIGHT * 5.5)
		sum1 = exp2[2:].copy()
		sum1.next_to(triangle1_incomp, UP)
		self.play(TransformFromCopy(exp2[2:], sum1))

		exp1 = TextMobject(r"$1 + \text{sum}(0)$")
		exp1.next_to(triangle1_incomp, DOWN)
		exp1.scale(scale)
		exp1[0].set_color(YELLOW)
		exp1[1].set_color(GOLD)
		exp1[2:].set_color(GREEN_SCREEN)

		self.play(FadeIn(triangle1_incomp), Write(exp1[0]))
		self.play(Write(exp1[1]))
		self.play(Write(exp1[2:]))
		self.wait(2)
		exp1_0 = TexMobject("1 + 0")
		exp1_0.next_to(exp1, DOWN)
		exp1_0.scale(scale)
		exp1_0[0].set_color(YELLOW)
		exp1_0[1].set_color(GOLD)
		exp1_0[2:].set_color(GREEN_SCREEN)

		self.play(TransformFromCopy(exp1, exp1_0))
		self.wait()

		one = TexMobject("1")
		one.next_to(exp1_0, DOWN)
		one.scale(scale)
		one.set_color(YELLOW)
		self.play(TransformFromCopy(exp1_0, one))
		self.wait()

		triangle2_incomp_rows = triangle2_incomp[0]
		triangle2_complete_rows = triangle2_incomp[1]

		triangle2_new_rows_group, fadeouts = self.show_recursive_step(triangle2_incomp_rows, color=ORANGE,
			fill_color=ORANGE, fill_opacity=0.5)
		exp2_1 = TexMobject("2 + 1")
		exp2_1.next_to(exp2, DOWN)
		exp2_1.scale(scale)
		exp2_1[0].set_color(ORANGE)
		exp2_1[1].set_color(GOLD)
		exp2_1[2:].set_color(YELLOW)
		self.play(*[TransformFromCopy(triangle1_incomp, triangle2_new_rows_group)] + 
			fadeouts + [TransformFromCopy(one, exp2_1)])
		self.wait()
		three = TexMobject("3")
		three.next_to(exp2_1, DOWN)
		three.scale(scale)
		three.set_color(ORANGE)
		self.play(TransformFromCopy(exp2_1, three))
		self.wait()

		triangle2_comp = VGroup(*[triangle2_new_rows_group[0]] + [square for square in triangle2_complete_rows])

		triangle3_incomp_rows = triangle3_incomp[:2]
		triangle3_complete_rows = triangle3_incomp[2]

		triangle3_new_rows_group, fadeouts = self.show_recursive_step(triangle3_incomp_rows, color=RED,
			fill_color=RED, fill_opacity=0.5)
		exp3_3 = TexMobject("3 + 3")
		exp3_3.next_to(exp3, DOWN)
		exp3_3.scale(scale)
		exp3_3[0].set_color(RED)
		exp3_3[1].set_color(GOLD)
		exp3_3[2:].set_color(ORANGE)
		self.play(*[TransformFromCopy(triangle2_comp, triangle3_new_rows_group)] + 
			fadeouts + [TransformFromCopy(three, exp3_3)])
		self.wait()
		six = TexMobject("6")
		six.next_to(exp3_3, DOWN)
		six.scale(scale)
		six.set_color(RED)
		self.play(TransformFromCopy(exp3_3, six))
		self.wait()

		triangle3_comp = VGroup(*[square for square in triangle3_new_rows_group] + 
			[square for square in triangle3_complete_rows])

		triangle4_incomp_rows = triangle4_incomp[:3]
		triangle4_complete_rows = triangle4_incomp[3]

		triangle4_new_rows_group, fadeouts = self.show_recursive_step(triangle4_incomp_rows, color=GREEN,
			fill_color=GREEN, fill_opacity=0.5)
		exp4_6 = TexMobject("4 + 6")
		exp4_6.next_to(exp4, DOWN)
		exp4_6.scale(scale)
		exp4_6[0].set_color(GREEN)
		exp4_6[1].set_color(GOLD)
		exp4_6[2:].set_color(RED)
		self.play(*[TransformFromCopy(triangle3_comp, triangle4_new_rows_group)] + 
			fadeouts + [TransformFromCopy(six, exp4_6)])
		self.wait()
		ten = TexMobject("10")
		ten.next_to(exp4_6, DOWN)
		ten.scale(scale)
		ten.set_color(GREEN)
		self.play(TransformFromCopy(exp4_6, ten))
		self.wait()

		triangle4_comp = VGroup(*[square for square in triangle4_new_rows_group] + 
			[square for square in triangle4_complete_rows])

		triangle5_incomp_rows = triangle5_incomp[:4]
		triangle5_complete_rows = triangle5_incomp[4]

		triangle5_new_rows_group, fadeouts = self.show_recursive_step(triangle5_incomp_rows)
		exp5_10 = TexMobject("5 + 10")
		exp5_10.next_to(exp5, DOWN)
		exp5_10.scale(scale)
		exp5_10[0].set_color(BLUE)
		exp5_10[1].set_color(GOLD)
		exp5_10[2:].set_color(GREEN)
		self.play(*[TransformFromCopy(triangle4_comp, triangle5_new_rows_group)] + 
			fadeouts + [TransformFromCopy(ten, exp5_10)])
		self.wait()
		fifteen = TexMobject("15")
		fifteen.next_to(exp5_10, DOWN)
		fifteen.scale(scale)
		fifteen.set_color(BLUE)
		self.play(TransformFromCopy(exp5_10, fifteen))
		self.wait(10)


	def show_recursive_step(self, triangle_incomp_rows, color=BLUE, 
		fill_color=BLUE, fill_opacity=0.5):
		triangle_new_rows = []
		fadeouts = []
		for row in triangle_incomp_rows:
			for square in row:
				new_square = Square(side_length=square.side_length, color=color, 
					fill_color=fill_color, fill_opacity=fill_opacity)
				new_square.move_to(square.get_center() + SMALL_BUFF / 4 * DOWN)
				triangle_new_rows.append(new_square)
				fadeouts.append(FadeOut(square))
		triangle_new_rows_group = VGroup(*triangle_new_rows)
		return triangle_new_rows_group, fadeouts

	def make_triangle(self, n, side_length, color=BLUE, fill_color=BLUE, 
		fill_opacity=0.5, position=ORIGIN):
		triangle = []
		for i in range(n):
			if i == 0:
				row = self.make_row(i + 1, side_length, color=color, 
					fill_color=fill_color, fill_opacity=fill_opacity)
			else:
				row = self.make_row(i + 1, side_length, color=color, 
					fill_color=fill_color, fill_opacity=fill_opacity, prev_row=triangle[i - 1])
			triangle.append(row)

		triangle_group = VGroup(*triangle)
		triangle_group.move_to(position)
		return triangle_group

	def make_incomplete_triangle(self, n, side_length, other_color, color=BLUE, fill_color=BLUE, 
		fill_opacity=0.5, position=ORIGIN):
		triangle = []
		for i in range(n):
			if i == 0:
				row = self.make_row(i + 1, side_length, color=other_color, 
					fill_color=BLACK, fill_opacity=fill_opacity)
			else:
				if i == n - 1:
					row = self.make_row(i + 1, side_length, color=color, 
						fill_color=fill_color, fill_opacity=fill_opacity, prev_row=triangle[i - 1], buff=SMALL_BUFF / 2)
				# elif i == n - 2:
				# 	row = self.make_row(i + 1, side_length, color=other_color, 
				# 		fill_color=BLACK, fill_opacity=fill_opacity, 
				# 		prev_row=triangle[i - 1], buff=SMALL_BUFF)
				else:
					row = self.make_row(i + 1, side_length, color=other_color, 
						fill_color=BLACK, fill_opacity=fill_opacity, prev_row=triangle[i - 1])
			triangle.append(row)

		triangle_group = VGroup(*triangle)
		triangle_group.move_to(position)
		return triangle_group


			
	def make_row(self, n, side_length, color=BLUE, fill_color=BLUE, 
		fill_opacity=0.5, prev_row=None, buff=0):
		rectangles = []
		rectangles.append(Square(side_length=side_length, color=color, 
			fill_color=fill_color, fill_opacity=fill_opacity))
		if prev_row:
			rectangles[0].next_to(prev_row[0], DOWN * side_length, buff=buff)
		for i in range(1, n):
			next_rect = Square(side_length=side_length, color=color, 
				fill_color=fill_color, fill_opacity=fill_opacity)
			next_rect.next_to(rectangles[i - 1], RIGHT * side_length, buff=0)
			rectangles.append(next_rect)

		return VGroup(*rectangles)


class Lecture(Scene):
	def construct(self):
		lines = []
		for i in np.arange(-3.4, 1.0, 0.7):
			line = Line(4 * LEFT + UP * i, 4 * RIGHT + UP * i)
			lines.append(line)

		for i in np.arange(1.8, 3.8, 0.7):
			line = Line(4 * LEFT + UP * i, 4 * RIGHT + UP * i)
			lines.append(line)



		middle_dots = [Dot(radius=0.05), Dot(radius=0.05), Dot(radius=0.05)]
		middle_dots[0].move_to(UP * 1)
		middle_dots[1].move_to(UP * 1.3)
		middle_dots[2].move_to(UP * 1.6)
		all_circles = []
		all_arrows = []
		circles_grid = []
		for i in range(1, len(lines)):
			if i != 7:
				circles, arrows = self.create_circles_in_row(lines[i - 1], lines[i], i % 2 == 1)
				all_circles.extend(circles)
				circles_grid.append(circles)
				all_arrows.extend(arrows)
		self.play(*[FadeIn(obj) for obj in lines + all_circles + middle_dots])
		lecture_hall = TextMobject("Lecture Hall")
		lecture_hall.move_to(UP * 3.5)
		lecture_hall.scale(0.8)
		self.play(FadeIn(lecture_hall))

		rows = TextMobject("rows")
		rows.set_color(BLUE)
		rows.scale(0.7)
		rows.move_to(LEFT * 5.5)
		students = TextMobject("students")
		students.set_color(RED)
		students.move_to(RIGHT * 5.5)
		students.scale(0.7)
		student_arrows_1 = Arrow(students.get_center() + LEFT * 0.5, students.get_center() + LEFT * 1.8 + UP * 0.5)
		student_arrows_2 = Arrow(students.get_center() + LEFT * 0.5, students.get_center() + LEFT * 2.3 + DOWN * 0.2)
		# student_arrows_1.set_color(RED)
		# student_arrows_2.set_color(RED)
		self.play(FadeIn(students), FadeIn(student_arrows_1), FadeIn(student_arrows_2))
		row_objs = [rows] + all_arrows
		self.play(*[FadeIn(obj) for obj in row_objs])
		self.wait()

		self.play(*[FadeOut(obj) for obj in row_objs + [students, student_arrows_1, student_arrows_2]])

		you = TextMobject("You")
		you.set_color(GREEN_SCREEN)
		you.scale(0.7)
		you.move_to(DOWN * 3.7)
		self.play(FadeIn(you), Indicate(circles_grid[0][3], color=GREEN_SCREEN))
		circle_copy = circles_grid[0][3].copy()
		circle_copy.set_color(GREEN_SCREEN)
		row_question = TextMobject("Row: ?")
		row_question.scale(0.8)
		row_question.move_to(circle_copy.get_center()[1] * UP + RIGHT * 5.5)
		row_question.set_color(GREEN_SCREEN)
		self.play(Transform(circles_grid[0][3], circle_copy))
		self.play(FadeIn(row_question))
		indicated_circles = [circles_grid[0][3]]
		row_questions = [row_question]
		colors = [PURPLE, BRIGHT_RED, GREEN, BLUE, ORANGE, PINK, LIGHT_BROWN]
		remaining_indications, remaining_questions = self.trace_recursion(circles_grid, 
			colors)
		indicated_circles += remaining_indications
		row_questions += remaining_questions
		self.wait()

		self.bubble_up(indicated_circles, row_questions, colors[::-1] + [GREEN_SCREEN])
		self.wait()


	def trace_recursion(self, circles_grid, colors):
		row_questions = []
		indicated_circles = []
		for i in range(1, len(circles_grid)):
			if i % 2 == 1 and i != 7:
				index = 4
			else:
				index = 3

			color = colors[i - 1]
			circle = circles_grid[i][index]
			circle_copy = circle.copy()
			circle_copy.set_color(color)
			self.play(Indicate(circle, color=color))
			if i == len(circles_grid) - 1:
				row_question = TextMobject("Row: 1")
			else:
				row_question = TextMobject("Row: ?")
			row_question.scale(0.8)
			row_question.move_to(circle_copy.get_center()[1] * UP + RIGHT * 5.5)
			row_question.set_color(color)
			self.play(Transform(circle, circle_copy), FadeIn(row_question))
			row_questions.append(row_question)
			indicated_circles.append(circle)
		return indicated_circles, row_questions

	def bubble_up(self, indicated_circles, row_questions, colors):
		reversed_indicated_circles = indicated_circles[::-1]
		reversed_row_questions = row_questions[::-1]
		row_numbers = [1, 2, 20, 21, 22, 23, 24, 25]
		for i in range(1, len(reversed_indicated_circles)):
			print(i)
			position = reversed_row_questions[i].get_center()
			row_number_added = TextMobject("Row: {0}".format(row_numbers[i]))
			row_number_added.scale(0.8)
			row_number_added.move_to(position)
			row_number_added.set_color(colors[i])
			self.play(Indicate(reversed_indicated_circles[i]))
			self.play(Indicate(reversed_indicated_circles[i]), 
				Transform(reversed_row_questions[i], row_number_added))
			if i == 1:
				self.wait(2)


	def create_circles_in_row(self, line_above, line_below, odd):
		arrows = []
		y_coord = (line_above.get_start() + line_below.get_start()) / 2
		y_scale = y_coord[1]
		circles = []
		if odd:
			start, end = -3, 4
		else:
			start, end = -3.5, 4.5
		for i in np.arange(start, end, 1):
			circle = Circle(radius=0.3)
			circle.move_to(i * RIGHT + UP * y_coord)
			circles.append(circle)
			
		arrow = Arrow(UP * y_coord + LEFT * 5, UP * y_coord + LEFT * 4)
		arrow.set_color(BLUE)
		arrows.append(arrow)
		return circles, arrows

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
	# right_eye_brow.shift(RIGHT * 1.2)
	right_eye_brow.set_color(color)


	eyes_and_smile = VGroup(left_circle, inner_left, right_circle, inner_right,
		smile, left_eye_brow, right_eye_brow)

	character = VGroup(computer, inner_rectangle, eyes_and_smile)
	character.scale(scale)
	character.move_to(position)

	return character


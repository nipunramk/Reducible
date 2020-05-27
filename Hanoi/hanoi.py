from big_ol_pile_of_manim_imports import *
COLORS = [RED, GREEN_SCREEN, BLUE, ORANGE, YELLOW]
class HanoiAnimation(Scene):
	def construct(self):
		# self.introduce_and_demo_hanoi_n_3()
		# self.visualize_recursion_n3()
		# self.visualize_recursion_n4()
		self.thumbnail()

	def thumbnail(self):
		starting_objects = self.setup_objects(3)
		title = TextMobject("Towers of Hanoi")
		title.scale(3)
		title.move_to(UP * 3)
		self.play(FadeIn(title))
		starting_objects.shift(DOWN)
		self.play(
			FadeIn(starting_objects)
		)

	def introduce_and_demo_hanoi_n_3(self):
		ending_objects = self.setup_objects(3, index=[2] * 3)
		ending_objects[0][2].set_color(YELLOW) # set last rod to yellow
		ending_objects.scale(0.6)
		ending_objects.shift(DOWN * 2)
		
		starting_objects = self.setup_objects(3)
		self.play(
			FadeIn(starting_objects[0])
		)

		top_large_disk = starting_objects[1].copy()
		top_large_disk.next_to(starting_objects[0][0], UP)

		self.play(
			FadeIn(top_large_disk)
		)

		self.play(
			ReplacementTransform(top_large_disk, starting_objects[1]),
			run_time=2
		)

		self.wait()

		top_middle_disk = starting_objects[2].copy()
		top_middle_disk.next_to(starting_objects[0][0], UP)

		self.play(
			FadeIn(top_middle_disk)
		)

		self.play(
			ReplacementTransform(top_middle_disk, starting_objects[2]),
			run_time=2
		)

		self.wait()

		top_small_disk = starting_objects[3].copy()
		top_small_disk.next_to(starting_objects[0][0], UP)

		self.play(
			FadeIn(top_small_disk)
		)

		self.play(
			ReplacementTransform(top_small_disk, starting_objects[3]),
			run_time=2
		)

		self.wait(6)

		# self.play(
		# 	FadeIn(starting_objects[1:])
		# )

		rods_label = TextMobject("Rods")
		rods_label.shift(UP * 3.5)

		rod_arrow_1 = Arrow(rods_label.get_center() + DOWN * 0.2, 
			starting_objects[0][0].get_center() + (starting_objects[0][0].get_height() / 2) * UP + UP * 0.1)

		rod_arrow_2 = Arrow(rods_label.get_center() + DOWN * SMALL_BUFF, 
			starting_objects[0][1].get_center() + (starting_objects[0][1].get_height() / 2) * UP + UP * 0.05)

		rod_arrow_3 = Arrow(rods_label.get_center() + DOWN * 0.2, 
			starting_objects[0][2].get_center() + (starting_objects[0][2].get_height() / 2) * UP + UP * SMALL_BUFF)

		self.play(
			FadeIn(rods_label),
			FadeIn(rod_arrow_1),
			FadeIn(rod_arrow_2),
			FadeIn(rod_arrow_3)
		)

		self.wait(3)

		disks_label = TextMobject("Disks")
		disks_label.move_to(starting_objects[2].get_center())
		disks_label.shift(RIGHT * 3)

		disk_arrow_1 = Arrow(disks_label.get_center() + LEFT * 0.4, 
			starting_objects[1].get_center() + (starting_objects[1].get_width() / 2) * RIGHT + RIGHT * 0.1)
		disk_arrow_1.set_color(RED)
		
		disk_arrow_2 = Arrow(disks_label.get_center() + LEFT * 0.4, 
			starting_objects[2].get_center() + (starting_objects[2].get_width() / 2) * RIGHT + RIGHT * 0.1)
		disk_arrow_2.set_color(GREEN_SCREEN)

		disk_arrow_3 = Arrow(disks_label.get_center() + LEFT * 0.4, 
			starting_objects[3].get_center() + (starting_objects[2].get_width() / 2) * RIGHT + RIGHT * 0.1)
		disk_arrow_3.set_color(BLUE)
		
		self.play(
			FadeIn(disks_label),
			FadeIn(disk_arrow_1),
			FadeIn(disk_arrow_2),
			FadeIn(disk_arrow_3)
		)
		self.wait(6)

		self.play(
			FadeOut(rods_label),
			FadeOut(rod_arrow_1),
			FadeOut(rod_arrow_2),
			FadeOut(rod_arrow_3),
			FadeOut(disks_label),
			FadeOut(disk_arrow_1),
			FadeOut(disk_arrow_2),
			FadeOut(disk_arrow_3)

		)
		self.wait()
		self.indicate_rod_then_transform(starting_objects[0], 2)

		starting_objects_transf = starting_objects.copy()
		starting_objects_transf.scale(0.6)
		starting_objects_transf.shift(UP * 2)
		self.play(
			Transform(starting_objects, starting_objects_transf), 
			run_time=2
		)

		self.play(
			FadeIn(ending_objects)
		)
		self.wait()
		starting_objects_transf = starting_objects.copy()
		starting_objects_transf.shift(DOWN * 2)
		starting_objects_transf.scale(1.0 / 0.6)
		self.play(
			FadeOut(ending_objects), 
			Transform(starting_objects, starting_objects_transf),
			run_time=2
		)

		self.animate_hanoi([[0, 1, 1, None], [0, 1, 1, None]])
		self.wait()
		to_cross = VGroup(*self.rods[1])
		cross = Cross(to_cross, color=BRIGHT_RED)
		self.play(
			ShowCreation(cross)
		)
		self.wait()

		self.play(
			FadeOut(cross),
		)
		self.animate_hanoi([[1, 0, 0.5, None], [1, 0, 0.5, None]])
		self.wait()
		self.play(
			FadeOut(starting_objects)
		)
		self.wait(4)
		

	def setup_objects(self, n, index=[0, 0, 0]):
		self.rods = [] # list of lists where each list is a stack of disks
		for i in range(3):
			self.rods.append([])
		color = WHITE
		if n < 4:
			height = 4
			offset = 2
		else:
			height = n + 1
			offset = n  - 1

		base = Rectangle(height=0.5, width=11, 
			fill_color=color, fill_opacity=1, color=color)
	
		first_rod = Rectangle(height=height, width=0.5,
			fill_color=color, fill_opacity=1, color=color)
		first_rod.set_color(WHITE)
		vertical_buffer = first_rod.get_height() / 2 + base.get_height() / 2
		first_rod.shift(LEFT * 4 + UP * vertical_buffer)

		second_rod = Rectangle(height=height, width=0.5,
			fill_color=color, fill_opacity=1, color=color)
		second_rod.shift(UP * vertical_buffer)

		third_rod = Rectangle(height=height, width=0.5,
			fill_color=color, fill_opacity=1, color=color)
		third_rod.shift(RIGHT * 4 +  UP * vertical_buffer)
		scale = 1
		self.structure = VGroup(first_rod, second_rod, third_rod, base)
		self.structure.shift(DOWN * offset)
		self.structure.scale(scale)


		disks = self.set_starting_position(n, scale=scale, index=index)

		self.all_rods_and_disks = VGroup(*[self.structure] + disks)

		return self.all_rods_and_disks

	def set_starting_position(self, num_disks, height=1, width=3, scale=1, index=[0, 0, 0]):
		disks = []
		for i in range(num_disks):
			color = COLORS[i]
			disk = self.create_disk(color=color, height=height, 
				width=width, corner_radius=0.5, scale=scale)
			self.place_disk_on_rod(disk, index[i])
			disks.append(disk)
			width = width - 0.5

		return disks

	def indicate_rod_then_transform(self, structure, index):
		rod = structure[index]
		self.play(Indicate(rod))
		rod_copy = rod.copy()
		rod_copy.set_color(YELLOW)
		self.play(Transform(rod, rod_copy))
		self.wait()


	def create_disk(self, color=BLUE, height=1, width=3, corner_radius=0.5, scale=1):
		disk = RoundedRectangle(height=height, width=width, corner_radius=corner_radius, fill_opacity=1)
		disk.set_color(color)
		disk.scale(scale)
		return disk

	def place_disk_on_rod(self, disk, index, append=True):
		rod = self.rods[index]
		rod_rectangle = self.structure[index]
		disk_buffer = disk.get_height() / 2
		if len(rod) == 0:
			center = rod_rectangle.get_center() + DOWN * rod_rectangle.get_height() / 2 + UP * disk_buffer
			disk.move_to(center)	
		else:
			last_disk = rod[-1]
			disk.next_to(last_disk, UP, buff=0)
		if append:
			self.rods[index].append(disk)


	def animate(self, n):
		moves = towers_of_hanoi(n)

		self.animate_hanoi(moves)

	def move_disk_from_rod(self, source, destination, run_time, wait_time, animate=True):
		
		source_rod = self.rods[source]
		source_rod_rect = self.structure[source]
		destination_rod = self.rods[destination]
		destination_rod_rect = self.structure[destination]
		disk = self.rods[source].pop()
		disk_buffer = disk.get_height() / 2
		
		# animation of removing disk from source rod
		first_point = source_rod_rect.get_center() + UP * source_rod_rect.get_height() / 2 + UP * disk_buffer + UP * 0.1
		moved_disk = disk.copy()
		moved_disk.move_to(first_point)
		animation = Transform(disk, moved_disk)
		if animate:
			self.play(
				animation,
				run_time=run_time
			)

		# animation of moving disk from source to top of destination
		second_point = destination_rod_rect.get_center() + UP * destination_rod_rect.get_height() / 2 + UP * disk_buffer + UP * 0.1
		moved_disk = disk.copy()
		moved_disk.move_to(second_point)
		animation = Transform(disk, moved_disk)
		if animate:
			self.play(
				animation,
				run_time=run_time
			)

		# animation of moving disk from top of destination onto actual rod
		moved_disk = disk.copy()
		self.place_disk_on_rod(moved_disk, destination, append=False)
		animation = Transform(disk, moved_disk)
		if animate:
			self.play(
				animation,
				run_time=run_time
			)
		self.rods[destination].append(disk)
		if wait_time:
			self.wait(wait_time)

	def animate_hanoi(self, moves, animate=True):
		# moves is a list of sources and destinations
		for source, destination, run_time, wait_time in moves:
			self.move_disk_from_rod(source, destination, run_time, wait_time, animate=animate)

	def fade_all_objects(self):
		self.play(FadeOut(self.all_rods_and_disks))

	def visualize_recursion_n3(self):
		recap = TextMobject("Recap")
		number_disks = TextMobject(r"\# Disks = 3")
		recap.scale(1.2)
		number_disks.scale(1.2)
		recap.shift(LEFT * 3 + UP * 0.6)
		number_disks.next_to(recap, DOWN)
		self.play(
			FadeIn(recap),
			FadeIn(number_disks)
		)

		n_3_hanoi = self.setup_objects(3)
		n_3_hanoi.scale(0.3)
		n_3_hanoi.move_to(RIGHT * 3 + UP * 3)
		self.play(
			FadeIn(n_3_hanoi)
		)
		self.wait()

		n_3_hanoi_step_1 = self.setup_objects(3, index=[0, 1, 1])
		n_3_hanoi_step_1.scale(0.3)
		n_3_hanoi_step_1.move_to(RIGHT * 3 + UP)
		self.play(
			FadeIn(n_3_hanoi_step_1)
		)
		self.wait()

		n_3_hanoi_step_2 = self.setup_objects(3, index=[2, 1, 1])
		n_3_hanoi_step_2.scale(0.3)
		n_3_hanoi_step_2.move_to(RIGHT * 3 + DOWN)
		self.play(
			FadeIn(n_3_hanoi_step_2)
		)
		self.wait()

		n_3_hanoi_step_3 = self.setup_objects(3, index=[2, 2, 2])
		n_3_hanoi_step_3.scale(0.3)
		n_3_hanoi_step_3.move_to(RIGHT * 3 + DOWN * 3)
		self.play(
			FadeIn(n_3_hanoi_step_3)
		)
		self.wait()

		self.play(
			FadeOut(recap),
			FadeOut(number_disks),
			FadeOut(n_3_hanoi),
			FadeOut(n_3_hanoi_step_1),
			FadeOut(n_3_hanoi_step_2),
			FadeOut(n_3_hanoi_step_3),
		)

	def visualize_recursion_n4(self):

		n_4_hanoi = self.setup_objects(4, index=[0, 0, 0, 0])
		n_4_hanoi.scale(0.3)
		n_4_hanoi.move_to(UP * 3)
		self.play(
			FadeIn(n_4_hanoi)
		)
		self.wait()

		n_4_hanoi_step_1 = self.setup_objects(4, index=[0, 1, 1, 1])
		n_4_hanoi_step_1.scale(0.3)
		n_4_hanoi_step_1.move_to(UP * 1)
		self.play(
			FadeIn(n_4_hanoi_step_1)
		)
		self.wait()

		n_4_hanoi_sub_step_1_a = self.setup_objects(4, index=[0, 0, 2, 2])
		n_4_hanoi_sub_step_1_a.scale(0.3)
		n_4_hanoi_sub_step_1_a.move_to(LEFT * 4 + UP * 1)

		n_4_hanoi_sub_step_1_b = self.setup_objects(4, index=[0, 1, 2, 2])
		n_4_hanoi_sub_step_1_b.scale(0.3)
		n_4_hanoi_sub_step_1_b.move_to(UP * 1)

		n_4_hanoi_step_1_copy = n_4_hanoi_step_1.copy()
		n_4_hanoi_step_1_copy.shift(RIGHT * 4)
		self.play(
			Transform(n_4_hanoi_step_1, n_4_hanoi_step_1_copy)
		)

		self.play(
			FadeIn(n_4_hanoi_sub_step_1_a)
		)
		self.wait()

		self.play(
			FadeIn(n_4_hanoi_sub_step_1_b)
		)
		self.wait()


		n_4_hanoi_step_2 = self.setup_objects(4, index=[2, 1, 1, 1])
		n_4_hanoi_step_2.scale(0.3)
		n_4_hanoi_step_2.move_to(DOWN * 1)
		self.play(
			FadeIn(n_4_hanoi_step_2)
		)
		self.wait()

		n_4_hanoi_step_3 = self.setup_objects(4, index=[2, 2, 2, 2])
		n_4_hanoi_step_3.scale(0.3)
		n_4_hanoi_step_3.move_to(DOWN * 3)
		self.play(
			FadeIn(n_4_hanoi_step_3)
		)
		self.wait()

		n_4_hanoi_sub_step_3_a = self.setup_objects(4, index=[2, 1, 0, 0])
		n_4_hanoi_sub_step_3_a.scale(0.3)
		n_4_hanoi_sub_step_3_a.move_to(LEFT * 4 + DOWN * 3)
		
		n_4_hanoi_sub_step_3_b = self.setup_objects(4, index=[2, 2, 0, 0])
		n_4_hanoi_sub_step_3_b.scale(0.3)
		n_4_hanoi_sub_step_3_b.move_to(DOWN * 3)

		n_4_hanoi_step_3_copy = n_4_hanoi_step_3.copy()
		n_4_hanoi_step_3_copy.shift(RIGHT * 4)
		self.play(
			Transform(n_4_hanoi_step_3, n_4_hanoi_step_3_copy)
		)

		self.play(
			FadeIn(n_4_hanoi_sub_step_3_a)
		)
		self.wait()

		self.play(
			FadeIn(n_4_hanoi_sub_step_3_b)
		)
		self.wait()

		self.play(
			FadeOut(n_4_hanoi),
			FadeOut(n_4_hanoi_step_1),
			FadeOut(n_4_hanoi_sub_step_1_a),
			FadeOut(n_4_hanoi_sub_step_1_b),
			FadeOut(n_4_hanoi_step_2),
			FadeOut(n_4_hanoi_step_3),
			FadeOut(n_4_hanoi_sub_step_3_a),
			FadeOut(n_4_hanoi_sub_step_3_b)
		)
class HanoiAnimationPart2(Scene):
	def construct(self):
		self.introduce_hanoi()
		self.visualize_recursion_n3()
	
	def introduce_hanoi(self):
		starting_objects = self.setup_objects(3)
		starting_objects[0][2].set_color(YELLOW)
		self.play(
			FadeIn(starting_objects)
		)
		self.wait(10)

		try_it = TextMobject("Try solving it on your own!")
		try_it.next_to(starting_objects, UP)
		try_it.shift(UP * 0.5)
		self.play(
			Write(try_it),
			run_time=2
		)

		self.wait(4)

		self.play(
			FadeOut(try_it)
		)

		self.wait(5)

		starting_objects_shifted = starting_objects.copy()
		starting_objects_shifted.shift(DOWN)

		strategy_1 = TextMobject("Strategy 1")
		strategy_1.move_to(UP * 3.5)
		guessing_strat = TextMobject("Random Guessing")
		guessing_strat.next_to(strategy_1, DOWN)
		self.play(
			FadeIn(strategy_1),
			Transform(starting_objects, starting_objects_shifted),
		)
		self.wait(2)
		self.play(
			Write(guessing_strat)
		)
		self.wait(8)

		self.animate_hanoi([[0, 1, 1, 3], [0, 2, 1, 8], [1, 2, 1, None]])
		self.wait(6)
		evaluate = TextMobject("Evaluate")
		evaluate.next_to(guessing_strat, DOWN)
		self.play(
			FadeIn(evaluate)
		)
		self.wait(8)
		third_rod_disks = VGroup(*self.rods[2])
		third_rod_disks_shifted = third_rod_disks.copy()
		third_rod_disks_shifted.shift(UP)
		self.play(
			Transform(third_rod_disks, third_rod_disks_shifted)
		)
		self.wait()

		first_rod_disk = VGroup(*self.rods[0])
		first_rod_disk_shifted_faded = first_rod_disk.copy()
		first_rod_disk_shifted_faded.shift(RIGHT * 8)
		first_rod_disk_shifted_faded.set_fill(opacity=0.5)
		self.play(
			TransformFromCopy(first_rod_disk, first_rod_disk_shifted_faded),
			run_time=2,
		)
		self.wait(2)

		third_rod_disks_shifted = third_rod_disks.copy()
		third_rod_disks_shifted.shift(DOWN)
		self.play(
			FadeOut(first_rod_disk_shifted_faded),
			FadeOut(evaluate), 
			Transform(third_rod_disks, third_rod_disks_shifted)
		)
		self.wait()

		self.animate_hanoi([[2, 1, 1, None], [2, 0, 1, None], [1, 0, 1, None]])
		self.wait(5)

		strategy_2 = TextMobject("Strategy 2")
		strategy_2.move_to(UP * 3.5)
		second_strat = TextMobject("Move largest disk to rod 3")
		second_strat[4:11].set_color(RED)
		second_strat.scale(0.8)
		second_strat.next_to(strategy_2, DOWN)

		first_rod_disk = VGroup(*self.rods[0][0])
		first_rod_disk_shifted_faded = first_rod_disk.copy()
		first_rod_disk_shifted_faded.shift(RIGHT * 8)
		first_rod_disk_shifted_faded.set_fill(opacity=0.5)
		self.play(
			ReplacementTransform(strategy_1, strategy_2),
			ReplacementTransform(guessing_strat, second_strat),
			TransformFromCopy(first_rod_disk, first_rod_disk_shifted_faded),
			run_time=2,
		)
		self.wait(11)

		self.play(
			Indicate(self.rods[0][2], color=BLUE, scale_factor=1.1),
			Indicate(self.rods[0][1], color=GREEN_SCREEN, scale_factor=1.1)
		)
		self.wait(7)

		addition_second_strat = TextMobject("Move top two disks to rod 2")
		addition_second_strat.scale(0.8)
		addition_second_strat.move_to(second_strat.get_center())
		addition_second_strat[4:7].set_color(BLUE)
		addition_second_strat[7:10].set_color(GREEN_SCREEN)
		second_strat_copy = second_strat.copy()
		second_strat_copy.next_to(addition_second_strat, DOWN)

		self.play(
			Transform(second_strat, second_strat_copy)
		)
		self.play(
			Write(addition_second_strat)
		)

		second_group_disks = VGroup(*self.rods[0][1:])
		second_group_disks_shifted_faded = second_group_disks.copy()
		second_group_disks_shifted_faded.shift(RIGHT * 4 + DOWN)
		second_group_disks_shifted_faded[0].set_fill(opacity=0.5)
		second_group_disks_shifted_faded[1].set_fill(opacity=0.5)
		self.play(
			TransformFromCopy(second_group_disks, second_group_disks_shifted_faded),
			run_time=2,
		)
		self.wait(5)

		starting_objects_shifted = starting_objects.copy()
		starting_objects_shifted.shift(DOWN * 0.5)

		self.play(
			Transform(starting_objects, starting_objects_shifted),
			FadeOut(second_group_disks_shifted_faded),
			FadeOut(first_rod_disk_shifted_faded)
		)

		self.wait()


		first_two_steps_moves = [
		[0, 2, 1, None],
		[0, 1, 1, None],
		[2, 1, 1, 1],
		[0, 2, 1, None]
		]
		
		# addition_second_strat = addition_second_strat.copy()

		second_strat_faded = second_strat.copy()
		second_strat_faded.set_fill(opacity=0.5)
		self.play(
			Transform(second_strat, second_strat_faded)
		)
		
		
		self.animate_hanoi(first_two_steps_moves[:3])
		self.wait()
		
		addition_second_strat_faded = addition_second_strat.copy()
		addition_second_strat_faded.set_fill(opacity=0.5)
		second_strat_highlighted = second_strat.copy()
		second_strat_highlighted.set_fill(opacity=1)
		self.play(
			Transform(addition_second_strat, addition_second_strat_faded),
			Transform(second_strat, second_strat_highlighted)
		)
		
		self.animate_hanoi(first_two_steps_moves[3:])
		self.wait(5)

		final_step = TextMobject("Move top two disks from rod 2 to rod 3")
		final_step.scale(0.8)
		final_step.next_to(second_strat, DOWN)
		final_step[4:7].set_color(BLUE)
		final_step[7:10].set_color(GREEN_SCREEN)

		second_strat_faded = second_strat.copy()
		second_strat_faded.set_fill(opacity=0.5)

		starting_objects_scaled = starting_objects.copy()
		starting_objects_scaled.scale(0.8)
		self.play(
			Write(final_step),
			Transform(second_strat, second_strat_faded)
		)

		second_group_disks = VGroup(*self.rods[1])
		second_group_disks_shifted_faded = second_group_disks.copy()
		second_group_disks_shifted_faded.shift(RIGHT * 4 + UP)
		second_group_disks_shifted_faded[0].set_fill(opacity=0.5)
		second_group_disks_shifted_faded[1].set_fill(opacity=0.5)
		self.play(
			TransformFromCopy(second_group_disks, second_group_disks_shifted_faded),
			run_time=1,
		)
		self.wait()
		self.play(
			FadeOut(second_group_disks_shifted_faded),
		)
		self.play(
			Transform(starting_objects, starting_objects_scaled)
		)

		last_steps_moves = [
		[1, 0, 1, None],
		[1, 2, 1, 1],
		[0, 2, 1, None],
		]

		self.animate_hanoi(last_steps_moves)
		self.wait(5)

		self.play(
			FadeOut(strategy_2),
			FadeOut(addition_second_strat),
			FadeOut(second_strat),
			FadeOut(starting_objects),
			FadeOut(final_step)
		)

	def setup_objects(self, n, index=[0, 0, 0]):
		self.rods = [] # list of lists where each list is a stack of disks
		for i in range(3):
			self.rods.append([])
		color = WHITE
		if n < 4:
			height = 4
			offset = 2
		else:
			height = n + 1
			offset = n  - 1

		base = Rectangle(height=0.5, width=11, 
			fill_color=color, fill_opacity=1, color=color)
	
		first_rod = Rectangle(height=height, width=0.5,
			fill_color=color, fill_opacity=1, color=color)
		first_rod.set_color(WHITE)
		vertical_buffer = first_rod.get_height() / 2 + base.get_height() / 2
		first_rod.shift(LEFT * 4 + UP * vertical_buffer)

		second_rod = Rectangle(height=height, width=0.5,
			fill_color=color, fill_opacity=1, color=color)
		second_rod.shift(UP * vertical_buffer)

		third_rod = Rectangle(height=height, width=0.5,
			fill_color=color, fill_opacity=1, color=color)
		third_rod.shift(RIGHT * 4 +  UP * vertical_buffer)
		scale = 1
		self.structure = VGroup(first_rod, second_rod, third_rod, base)
		self.structure.shift(DOWN * offset)
		self.structure.scale(scale)


		disks = self.set_starting_position(n, scale=scale, index=index)

		self.all_rods_and_disks = VGroup(*[self.structure] + disks)

		return self.all_rods_and_disks

	def set_starting_position(self, num_disks, height=1, width=3, scale=1, index=[0, 0, 0]):
		disks = []
		for i in range(num_disks):
			color = COLORS[i]
			disk = self.create_disk(color=color, height=height, 
				width=width, corner_radius=0.5, scale=scale)
			self.place_disk_on_rod(disk, index[i])
			disks.append(disk)
			width = width - 0.5

		return disks

	def indicate_rod_then_transform(self, structure, index):
		rod = structure[index]
		self.play(Indicate(rod))
		rod_copy = rod.copy()
		rod_copy.set_color(YELLOW)
		self.play(Transform(rod, rod_copy))
		self.wait()


	def create_disk(self, color=BLUE, height=1, width=3, corner_radius=0.5, scale=1):
		disk = RoundedRectangle(height=height, width=width, corner_radius=corner_radius, fill_opacity=1)
		disk.set_color(color)
		disk.scale(scale)
		return disk

	def place_disk_on_rod(self, disk, index, append=True):
		rod = self.rods[index]
		rod_rectangle = self.structure[index]
		disk_buffer = disk.get_height() / 2
		if len(rod) == 0:
			center = rod_rectangle.get_center() + DOWN * rod_rectangle.get_height() / 2 + UP * disk_buffer
			disk.move_to(center)	
		else:
			last_disk = rod[-1]
			disk.next_to(last_disk, UP, buff=0)
		if append:
			self.rods[index].append(disk)


	def animate(self, n):
		moves = towers_of_hanoi(n)

		self.animate_hanoi(moves)

	def move_disk_from_rod(self, source, destination, run_time, wait_time, animate=True):
		
		source_rod = self.rods[source]
		source_rod_rect = self.structure[source]
		destination_rod = self.rods[destination]
		destination_rod_rect = self.structure[destination]
		disk = self.rods[source].pop()
		disk_buffer = disk.get_height() / 2
		
		# animation of removing disk from source rod
		first_point = source_rod_rect.get_center() + UP * source_rod_rect.get_height() / 2 + UP * disk_buffer + UP * 0.1
		moved_disk = disk.copy()
		moved_disk.move_to(first_point)
		animation = Transform(disk, moved_disk)
		if animate:
			self.play(
				animation,
				run_time=run_time
			)

		# animation of moving disk from source to top of destination
		second_point = destination_rod_rect.get_center() + UP * destination_rod_rect.get_height() / 2 + UP * disk_buffer + UP * 0.1
		moved_disk = disk.copy()
		moved_disk.move_to(second_point)
		animation = Transform(disk, moved_disk)
		if animate:
			self.play(
				animation,
				run_time=run_time
			)

		# animation of moving disk from top of destination onto actual rod
		moved_disk = disk.copy()
		self.place_disk_on_rod(moved_disk, destination, append=False)
		animation = Transform(disk, moved_disk)
		if animate:
			self.play(
				animation,
				run_time=run_time
			)
		self.rods[destination].append(disk)
		if wait_time:
			self.wait(wait_time)

	def animate_hanoi(self, moves, animate=True):
		# moves is a list of sources and destinations
		for source, destination, run_time, wait_time in moves:
			self.move_disk_from_rod(source, destination, run_time, wait_time, animate=animate)

	def fade_all_objects(self):
		self.play(FadeOut(self.all_rods_and_disks))

	def visualize_recursion_n3(self):
		recap = TextMobject("Recap")
		number_disks = TextMobject(r"\# Disks = 3")
		recap.scale(1.2)
		number_disks.scale(1.2)
		recap.shift(LEFT * 3 + UP * 0.6)
		number_disks.next_to(recap, DOWN)
		

		n_3_hanoi = self.setup_objects(3)
		n_3_hanoi.scale(0.3)
		n_3_hanoi.move_to(RIGHT * 3 + UP * 3)
		self.play(
			FadeIn(recap),
			FadeIn(number_disks),
			FadeIn(n_3_hanoi)
		)
		self.wait(3)

		n_3_hanoi_step_1 = self.setup_objects(3, index=[0, 1, 1])
		n_3_hanoi_step_1.scale(0.3)
		n_3_hanoi_step_1.move_to(RIGHT * 3 + UP)
		self.play(
			FadeIn(n_3_hanoi_step_1)
		)
		self.wait(3)

		n_3_hanoi_step_2 = self.setup_objects(3, index=[2, 1, 1])
		n_3_hanoi_step_2.scale(0.3)
		n_3_hanoi_step_2.move_to(RIGHT * 3 + DOWN)
		self.play(
			FadeIn(n_3_hanoi_step_2)
		)
		self.wait(3)

		n_3_hanoi_step_3 = self.setup_objects(3, index=[2, 2, 2])
		n_3_hanoi_step_3.scale(0.3)
		n_3_hanoi_step_3.move_to(RIGHT * 3 + DOWN * 3)
		self.play(
			FadeIn(n_3_hanoi_step_3)
		)
		self.wait(3)

		self.play(
			FadeOut(recap),
			FadeOut(number_disks),
			FadeOut(n_3_hanoi),
			FadeOut(n_3_hanoi_step_1),
			FadeOut(n_3_hanoi_step_2),
			FadeOut(n_3_hanoi_step_3),
		)

class Transition(Scene):
	def construct(self):
		rectangle = ScreenRectangle()
		rectangle.set_width(8)
		rectangle.set_height(6)

		title = TextMobject("Towers of Hanoi")
		title.scale(1.2)
		title.next_to(rectangle, UP)
		self.play(
			Write(title)
		)
		self.wait(2)

		self.play(
			ShowCreation(rectangle)
		)

		puzzle = TextMobject("1. Puzzle Perspective")
		puzzle.next_to(rectangle, DOWN)
		self.play(
			Write(puzzle)
		)
		self.wait(5)

		recursion = TextMobject("2. Recursion Perspective")
		recursion.next_to(rectangle, DOWN)
		self.play(
			ReplacementTransform(puzzle, recursion),
		)
		self.wait(10)

		visual = TextMobject("3. Complete Recursive Visualization")
		visual.next_to(rectangle, DOWN)
		self.play(
			ReplacementTransform(recursion, visual),
		)

		self.wait(10)

		self.play(
			FadeOut(visual),
			FadeOut(rectangle),
			FadeOut(title)
		)


	def screen_rects(self):

		rect3 = ScreenRectangle()
		rect3.set_width(3.7)
		rect3.set_height(2.2)
		rect3.move_to(LEFT * 3.5 + DOWN * 2.7)

		text3 = TextMobject("Complete Recursive Visualization")
		text3.scale(0.8)
		text3.next_to(rect3, RIGHT)
		
		rect2 = ScreenRectangle()
		rect2.set_width(3.7)
		rect2.set_height(2.2)
		rect2.move_to(LEFT * 3.5)

		text2 = TextMobject("Recursion Perspective")
		text2.scale(0.8)
		text2.next_to(rect2, RIGHT)

		rect1 = ScreenRectangle()
		rect1.set_width(3.7)
		rect1.set_height(2.2)
		rect1.move_to(LEFT * 3.5 + UP * 2.7)

		text1 = TextMobject("Puzzle Perspective")
		text1.scale(0.8)
		text1.next_to(rect1, RIGHT)

		self.play(
			FadeIn(rect1), 
			FadeIn(rect2), 
			FadeIn(rect3)
		)
		self.wait()
		self.play(
			FadeIn(text1)
		)
		self.play(
			FadeIn(text2)
		)
		self.play(
			FadeIn(text3)
		)
		self.wait(3)

		self.play(
			FadeOut(text1), 
			FadeOut(text2), 
			FadeOut(text3)
		)
		self.wait()

		pause_left = Rectangle(height=2, width=0.5, fill_color=YELLOW, fill_opacity=1)
		pause_left.set_color(YELLOW)
		pause_right = Rectangle(height=2, width=0.5, fill_color=YELLOW, fill_opacity=1)
		pause_right.set_color(YELLOW)
		pause_right.next_to(pause_left, RIGHT)
		pause = VGroup(pause_left, pause_right)
		pause.move_to(RIGHT * 3)
		self.play(
			FadeIn(pause)
		)
		self.wait()

		pause_copy = pause.copy()
		play = RegularPolygon(fill_color=GREEN_SCREEN, fill_opacity=1)
		play.scale(1.5)
		play.set_color(GREEN_SCREEN)
		play.move_to(pause.get_center() + RIGHT * 0.5)
		self.play(ReplacementTransform(pause, play))
		self.wait(3)

		self.play(
			ReplacementTransform(play, pause_copy)
		)
		self.wait(3)

class Conclusion(Scene):
	def construct(self):
		rectangle = ScreenRectangle()
		rectangle.set_width(8)
		rectangle.set_height(6)

		self.wait()

		self.play(
			ShowCreation(rectangle)
		)

		self.wait(42)

		self.play(
			FadeOut(rectangle),
		)


class HanoiAnimationPart3(Scene):
	def construct(self):
		self.visualize_recursion_n4()

	def setup_objects(self, n, index=[0, 0, 0]):
		self.rods = [] # list of lists where each list is a stack of disks
		for i in range(3):
			self.rods.append([])
		color = WHITE
		if n < 4:
			height = 4
			offset = 2
		else:
			height = n + 1
			offset = n  - 1

		base = Rectangle(height=0.5, width=11, 
			fill_color=color, fill_opacity=1, color=color)
	
		first_rod = Rectangle(height=height, width=0.5,
			fill_color=color, fill_opacity=1, color=color)
		first_rod.set_color(WHITE)
		vertical_buffer = first_rod.get_height() / 2 + base.get_height() / 2
		first_rod.shift(LEFT * 4 + UP * vertical_buffer)

		second_rod = Rectangle(height=height, width=0.5,
			fill_color=color, fill_opacity=1, color=color)
		second_rod.shift(UP * vertical_buffer)

		third_rod = Rectangle(height=height, width=0.5,
			fill_color=color, fill_opacity=1, color=color)
		third_rod.shift(RIGHT * 4 +  UP * vertical_buffer)
		scale = 1
		self.structure = VGroup(first_rod, second_rod, third_rod, base)
		self.structure.shift(DOWN * offset)
		self.structure.scale(scale)


		disks = self.set_starting_position(n, scale=scale, index=index)

		self.all_rods_and_disks = VGroup(*[self.structure] + disks)

		return self.all_rods_and_disks

	def set_starting_position(self, num_disks, height=1, width=3, scale=1, index=[0, 0, 0]):
		disks = []
		for i in range(num_disks):
			color = COLORS[i]
			disk = self.create_disk(color=color, height=height, 
				width=width, corner_radius=0.5, scale=scale)
			self.place_disk_on_rod(disk, index[i])
			disks.append(disk)
			width = width - 0.5

		return disks

	def indicate_rod_then_transform(self, structure, index):
		rod = structure[index]
		self.play(Indicate(rod))
		rod_copy = rod.copy()
		rod_copy.set_color(YELLOW)
		self.play(Transform(rod, rod_copy))
		self.wait()


	def create_disk(self, color=BLUE, height=1, width=3, corner_radius=0.5, scale=1):
		disk = RoundedRectangle(height=height, width=width, corner_radius=corner_radius, fill_opacity=1)
		disk.set_color(color)
		disk.scale(scale)
		return disk

	def place_disk_on_rod(self, disk, index, append=True):
		rod = self.rods[index]
		rod_rectangle = self.structure[index]
		disk_buffer = disk.get_height() / 2
		if len(rod) == 0:
			center = rod_rectangle.get_center() + DOWN * rod_rectangle.get_height() / 2 + UP * disk_buffer
			disk.move_to(center)	
		else:
			last_disk = rod[-1]
			disk.next_to(last_disk, UP, buff=0)
		if append:
			self.rods[index].append(disk)

	def visualize_recursion_n4(self):

		n_4_hanoi = self.setup_objects(4, index=[0, 0, 0, 0])
		n_4_hanoi.scale(0.3)
		n_4_hanoi.move_to(UP * 3)
		n_4_hanoi[0][2].set_color(YELLOW)
		self.play(
			FadeIn(n_4_hanoi)
		)
		self.wait(3)

		n_4_hanoi_step_1 = self.setup_objects(4, index=[0, 1, 1, 1])
		n_4_hanoi_step_1.scale(0.3)
		n_4_hanoi_step_1.move_to(UP * 1)
		n_4_hanoi_step_1[0][2].set_color(YELLOW)
		self.play(
			FadeIn(n_4_hanoi_step_1)
		)
		self.wait(3)

		n_4_hanoi_sub_step_1_a = self.setup_objects(4, index=[0, 0, 2, 2])
		n_4_hanoi_sub_step_1_a.scale(0.3)
		n_4_hanoi_sub_step_1_a.move_to(LEFT * 4 + UP * 1)
		n_4_hanoi_sub_step_1_a[0][2].set_color(YELLOW)

		n_4_hanoi_sub_step_1_b = self.setup_objects(4, index=[0, 1, 2, 2])
		n_4_hanoi_sub_step_1_b.scale(0.3)
		n_4_hanoi_sub_step_1_b.move_to(UP * 1)
		n_4_hanoi_sub_step_1_b[0][2].set_color(YELLOW)

		n_4_hanoi_step_1_copy = n_4_hanoi_step_1.copy()
		n_4_hanoi_step_1_copy.shift(RIGHT * 4)
		self.play(
			Transform(n_4_hanoi_step_1, n_4_hanoi_step_1_copy)
		)
		self.wait(2)

		self.play(
			FadeIn(n_4_hanoi_sub_step_1_a)
		)
		self.wait(3)

		self.play(
			FadeIn(n_4_hanoi_sub_step_1_b)
		)
		self.wait(8)


		n_4_hanoi_step_2 = self.setup_objects(4, index=[2, 1, 1, 1])
		n_4_hanoi_step_2.scale(0.3)
		n_4_hanoi_step_2.move_to(DOWN * 1)
		n_4_hanoi_step_2[0][2].set_color(YELLOW)
		self.play(
			FadeIn(n_4_hanoi_step_2)
		)
		self.wait(5)

		n_4_hanoi_step_3 = self.setup_objects(4, index=[2, 2, 2, 2])
		n_4_hanoi_step_3.scale(0.3)
		n_4_hanoi_step_3.move_to(DOWN * 3)
		n_4_hanoi_step_3[0][2].set_color(YELLOW)
		self.play(
			FadeIn(n_4_hanoi_step_3)
		)
		self.wait()

		n_4_hanoi_sub_step_3_a = self.setup_objects(4, index=[2, 1, 0, 0])
		n_4_hanoi_sub_step_3_a.scale(0.3)
		n_4_hanoi_sub_step_3_a.move_to(LEFT * 4 + DOWN * 3)
		n_4_hanoi_sub_step_3_a[0][2].set_color(YELLOW)
		
		n_4_hanoi_sub_step_3_b = self.setup_objects(4, index=[2, 2, 0, 0])
		n_4_hanoi_sub_step_3_b.scale(0.3)
		n_4_hanoi_sub_step_3_b.move_to(DOWN * 3)
		n_4_hanoi_sub_step_3_b[0][2].set_color(YELLOW)

		n_4_hanoi_step_3_copy = n_4_hanoi_step_3.copy()
		n_4_hanoi_step_3_copy.shift(RIGHT * 4)
		self.play(
			Transform(n_4_hanoi_step_3, n_4_hanoi_step_3_copy)
		)
		self.wait()

		self.play(
			FadeIn(n_4_hanoi_sub_step_3_a)
		)
		self.wait()

		self.play(
			FadeIn(n_4_hanoi_sub_step_3_b)
		)
		self.wait(7)

		self.play(
			FadeOut(n_4_hanoi),
			FadeOut(n_4_hanoi_step_1),
			FadeOut(n_4_hanoi_sub_step_1_a),
			FadeOut(n_4_hanoi_sub_step_1_b),
			FadeOut(n_4_hanoi_step_2),
			FadeOut(n_4_hanoi_step_3),
			FadeOut(n_4_hanoi_sub_step_3_a),
			FadeOut(n_4_hanoi_sub_step_3_b)
		)

class TreeRecVisualization(Scene):
	def construct(self):
		self.visualize_rec_tree()

	def setup_objects(self, n, index=[0, 0, 0]):
		rods = [] # list of lists where each list is a stack of disks
		for i in range(3):
			rods.append([])
		color = WHITE
		if n < 4:
			height = 4
			offset = 2
		else:
			height = n + 1
			offset = n  - 1

		base = Rectangle(height=0.5, width=11, 
			fill_color=color, fill_opacity=1, color=color)
	
		first_rod = Rectangle(height=height, width=0.5,
			fill_color=color, fill_opacity=1, color=color)
		first_rod.set_color(WHITE)
		vertical_buffer = first_rod.get_height() / 2 + base.get_height() / 2
		first_rod.shift(LEFT * 4 + UP * vertical_buffer)

		second_rod = Rectangle(height=height, width=0.5,
			fill_color=color, fill_opacity=1, color=color)
		second_rod.shift(UP * vertical_buffer)

		third_rod = Rectangle(height=height, width=0.5,
			fill_color=color, fill_opacity=1, color=color)
		third_rod.shift(RIGHT * 4 +  UP * vertical_buffer)
		scale = 1
		structure = VGroup(first_rod, second_rod, third_rod, base)
		structure.shift(DOWN * offset)
		structure.scale(scale)


		disks, rods = self.set_starting_position(structure, n, rods, scale=scale, index=index)

		self.all_rods_and_disks = VGroup(*[structure] + disks)

		return self.all_rods_and_disks, rods

	def set_starting_position(self, structure, num_disks, rods, height=1, width=3, scale=1, index=[0, 0, 0]):
		disks = []
		for i in range(num_disks):
			color = COLORS[i]
			disk = self.create_disk(color=color, height=height, 
				width=width, corner_radius=0.5, scale=scale)
			rods = self.place_disk_on_rod(structure, disk, rods, index[i])
			disks.append(disk)
			width = width - 0.5

		return disks, rods

	def create_disk(self, color=BLUE, height=1, width=3, corner_radius=0.5, scale=1):
		disk = RoundedRectangle(height=height, width=width, corner_radius=corner_radius, fill_opacity=1)
		disk.set_color(color)
		disk.scale(scale)
		return disk

	def place_disk_on_rod(self, structure, disk, rods, index, append=True):
		rod = rods[index]
		rod_rectangle = structure[index]
		disk_buffer = disk.get_height() / 2
		if len(rod) == 0:
			center = rod_rectangle.get_center() + DOWN * rod_rectangle.get_height() / 2 + UP * disk_buffer
			disk.move_to(center)	
		else:
			last_disk = rod[-1]
			disk.next_to(last_disk, UP, buff=0)
		if append:
			rods[index].append(disk)
		return rods


	def animate(self, n):
		moves = towers_of_hanoi(n)

		self.animate_hanoi(moves)

	def move_disk_from_rod(self, structure, rods, source, destination, run_time, wait_time, animate=True):
		
		source_rod = rods[source]
		source_rod_rect = structure[source]
		destination_rod = rods[destination]
		destination_rod_rect = structure[destination]
		disk = rods[source].pop()
		disk_buffer = disk.get_height() / 2
		
		# animation of removing disk from source rod
		first_point = source_rod_rect.get_center() + UP * source_rod_rect.get_height() / 2 + UP * disk_buffer + UP * 0.1
		moved_disk = disk.copy()
		moved_disk.move_to(first_point)
		animation = Transform(disk, moved_disk)
		if animate:
			self.play(
				animation,
				run_time=run_time
			)

		# animation of moving disk from source to top of destination
		second_point = destination_rod_rect.get_center() + UP * destination_rod_rect.get_height() / 2 + UP * disk_buffer + UP * 0.1
		moved_disk = disk.copy()
		moved_disk.move_to(second_point)
		animation = Transform(disk, moved_disk)
		if animate:
			self.play(
				animation,
				run_time=run_time
			)

		# animation of moving disk from top of destination onto actual rod
		moved_disk = disk.copy()
		self.place_disk_on_rod(structure, moved_disk, rods, destination, append=False)
		animation = Transform(disk, moved_disk)
		if animate:
			self.play(
				animation,
				run_time=run_time
			)
		rods[destination].append(disk)
		if wait_time:
			self.wait(wait_time)
		return rods

	def animate_hanoi(self, structure, rods, moves, animate=True):
		# moves is a list of sources and destinations
		for source, destination, run_time, wait_time in moves:
			self.move_disk_from_rod(structure, rods, source, destination, run_time, wait_time, animate=animate)
		return rods

	def repr_rec_call(self, call, num_disks, shift, disk_index,
		scale=0.2, opacity=0.4, index=[0, 0, 0]):
		objects, rods = self.setup_objects(num_disks, index=index)
		objects.scale(scale)
		objects.next_to(call, DOWN * 1.2)
		disks = []
		for i in disk_index:
			disks.append(objects[i])
		disks = VGroup(*disks)
		disks_copy = disks.copy()
		scaled_shift = shift * scale
		disks_copy.shift(scaled_shift)
		for i in range(len(disks_copy)):
			disks_copy[i].set_fill(opacity=opacity)

		return objects, rods, disks, disks_copy

	def repr_print_call(self, call, num_disks, move, 
		scale=0.2, index=[0, 0, 0]):
		objects, rods = self.setup_objects(num_disks, index=index)
		objects.scale(scale)
		objects.next_to(call, DOWN * 1.2)
		self.play(
			FadeIn(objects)
		)
		self.wait()
		self.animate_hanoi(objects[0], rods, move)

	def visualize_rec_tree(self):
		image = ImageMobject("hanoi_def")
		image.scale(0.9)
		image.shift(LEFT * 3.5 + UP * 2.8)

		print_image = ImageMobject("print_img")
		print_image.scale(0.3)
		print_image.shift(RIGHT * 3.5 + UP * 2.8)
		first_call = TextMobject("h(3, 1, 3)")
		first_call.scale(0.6)
		first_call.move_to(UP * 3.5)
		first_call[2].set_color(BLUE)
		first_call[4].set_color(RED)
		first_call[6].set_color(YELLOW)

		self.wait(2)

		self.play(
			FadeIn(first_call),
			FadeIn(image), 
			FadeIn(print_image)
		)
		self.wait(3)
		starting_objects, start_rods, disks, disks_copy = self.repr_rec_call(first_call, 
			3, RIGHT * 8, [1, 2, 3])
		starting_objects[0][2].set_color(YELLOW)
		self.play(
			FadeIn(starting_objects)
		)
		self.wait(3)
		self.play(
			TransformFromCopy(disks, disks_copy)
		)
		self.wait()


		left_call = TextMobject("h(2, 1, 2)")
		left_call.scale(0.6)
		left_call.move_to(UP * 1.5 + LEFT * 3.5)
		left_call[2].set_color(BLUE)
		left_call[4].set_color(RED)
		left_call[6].set_color(YELLOW)
		self.play(
			FadeIn(left_call)
		)

		self.wait(4)

		left_call_objects, left_call_rods, left_call_disks, left_call_disks_copy = self.repr_rec_call(left_call, 
			3, RIGHT * 4 + DOWN, [2, 3])
		# left_call_objects[0][1].set_color(YELLOW)
		self.play(
			FadeIn(left_call_objects)
		)
		self.wait()
		self.play(
			TransformFromCopy(left_call_disks, left_call_disks_copy)
		)
		self.wait(5)


		bottom_call_1 = TextMobject("h(1, 1, 3)")
		bottom_call_1.scale(0.6)
		bottom_call_1.move_to(DOWN * 0.5 + LEFT * 5.8)
		bottom_call_1[2].set_color(BLUE)
		bottom_call_1[4].set_color(RED)
		bottom_call_1[6].set_color(YELLOW)
		self.play(
			FadeIn(bottom_call_1)
		)
		self.wait(6)

		bottom_1_call_objects, bottom_1_call_rods, bottom_1_call_disks, bottom_1_call_disks_copy = self.repr_rec_call(bottom_call_1, 
			3, RIGHT * 8 + DOWN * 2, [3])
		# bottom_1_call_objects[0][2].set_color(YELLOW)
		self.play(
			FadeIn(bottom_1_call_objects)
		)
		self.wait()
		self.play(
			TransformFromCopy(bottom_1_call_disks, bottom_1_call_disks_copy)
		)
		self.wait(6)


		print_call_1 = TextMobject("pm(1, 3)")
		print_call_1.scale(0.6)
		print_call_1.move_to(DOWN * 2.5 + LEFT * 5.8)
		print_call_1[3].set_color(RED)
		print_call_1[5].set_color(YELLOW)
		self.play(
			FadeIn(print_call_1)
		)

		self.wait()

		print_call_1_green = print_call_1.copy()
		print_call_1_green.set_color(GREEN_SCREEN)

		self.repr_print_call(print_call_1, 3, [[0, 2, 1, None]])
		self.play(
			Transform(print_call_1, print_call_1_green)
		)
		self.wait()

		
		bottom_call_1_green = bottom_call_1.copy()
		bottom_call_1_green.set_color(GREEN_SCREEN)
		self.animate_hanoi(bottom_1_call_objects[0], bottom_1_call_rods, [[0, 2, 1, None]])
		self.play(
			Transform(bottom_call_1, bottom_call_1_green)
		)
		self.wait()

		self.animate_hanoi(left_call_objects[0], left_call_rods, [[0, 2, 1, None]])
		self.wait()

		self.animate_hanoi(starting_objects[0], start_rods, [[0, 2, 1, None]])
		self.wait(2)

		self.play(
			Indicate(left_call)
		)

		self.wait(2)

		bottom_call_2 = TextMobject("pm(1, 2)")
		bottom_call_2.scale(0.6)
		bottom_call_2.move_to(DOWN * 0.5 + LEFT * 3.5)
		self.play(
			FadeIn(bottom_call_2)
		)

		bottom_call_2_green = bottom_call_2.copy()
		bottom_call_2_green.set_color(GREEN_SCREEN)

		self.repr_print_call(bottom_call_2, 3, [[0, 1, 1, None]], index=[0, 0 ,2])
		self.play(
			Transform(bottom_call_2, bottom_call_2_green)
		)
		self.animate_hanoi(left_call_objects[0], left_call_rods, [[0, 1, 1, None]])
		self.animate_hanoi(starting_objects[0], start_rods, [[0, 1, 1, None]])

		self.wait(2)
		bottom_call_3 = TextMobject("h(1, 3, 2)")
		bottom_call_3.scale(0.6)
		bottom_call_3.move_to(DOWN * 0.5 + LEFT * 1.2)
		self.play(
			FadeIn(bottom_call_3)
		)

		self.wait(4)

		bottom_3_call_objects, bottom_3_call_rods, bottom_3_call_disks, bottom_3_call_disks_copy = self.repr_rec_call(bottom_call_3, 
			3, LEFT * 4 + UP, [3], index=[0, 1, 2])

		self.play(
			FadeIn(bottom_3_call_objects)
		)
		self.play(
			TransformFromCopy(bottom_3_call_disks, bottom_3_call_disks_copy)
		)
		self.wait()

		print_call_2 = TextMobject("pm(3, 2)")
		print_call_2.scale(0.6)
		print_call_2.move_to(DOWN * 2.5 + LEFT * 1.2)
		self.play(
			FadeIn(print_call_2)
		)
		print_call_2_green = print_call_2.copy()
		print_call_2_green.set_color(GREEN_SCREEN)

		self.repr_print_call(print_call_2, 3, [[2, 1, 1, None]], index=[0, 1 ,2])
		self.play(
			Transform(print_call_2, print_call_2_green)
		)

		bottom_call_3_green = bottom_call_3.copy()
		bottom_call_3_green.set_color(GREEN_SCREEN)

		self.animate_hanoi(bottom_3_call_objects[0], bottom_3_call_rods, [[2, 1, 1, None]])
		self.play(
			Transform(bottom_call_3, bottom_call_3_green)
		)


		left_call_green = left_call.copy()
		left_call_green.set_color(GREEN_SCREEN)
		self.animate_hanoi(left_call_objects[0], left_call_rods, [[2, 1, 1, None]])
		self.play(
			Transform(left_call, left_call_green)
		)

		self.animate_hanoi(starting_objects[0], start_rods, [[2, 1, 1, None]])
		self.wait(7)

		middle_call = TextMobject("pm(1, 3)")
		middle_call.scale(0.6)
		middle_call.move_to(UP * 1.5)
		self.play(
			FadeIn(middle_call)
		)
		self.wait(2)

		middle_call_green = middle_call.copy()
		middle_call_green.set_color(GREEN_SCREEN)

		self.repr_print_call(middle_call, 3, [[0, 2, 1, None]], index=[0, 1 ,1])
		self.play(
			Transform(middle_call, middle_call_green)
		)
		self.animate_hanoi(starting_objects[0], start_rods, [[0, 2, 1, None]])
		self.wait(10)

		right_call = TextMobject("h(2, 2, 3)")
		right_call.scale(0.6)
		right_call.move_to(UP * 1.5 + RIGHT * 3.5)
		self.play(
			FadeIn(right_call)
		)

		right_call_objects, right_call_rods, right_call_disks, right_call_disks_copy = self.repr_rec_call(right_call, 
			3, RIGHT * 4 + UP, [2, 3], index=[2, 1, 1])

		self.play(
			FadeIn(right_call_objects)
		)
		self.play(
			TransformFromCopy(right_call_disks, right_call_disks_copy)
		)
		self.wait(6)


		bottom_call_4 = TextMobject("h(1, 2, 1)")
		bottom_call_4.scale(0.6)
		bottom_call_4.move_to(DOWN * 0.5 + RIGHT * 1.2)
		self.play(
			FadeIn(bottom_call_4)
		)

		bottom_4_call_objects, bottom_4_call_rods, bottom_4_call_disks, bottom_4_call_disks_copy = self.repr_rec_call(bottom_call_4, 
			3, LEFT * 4 + DOWN, [3], index=[2, 1, 1])

		self.play(
			FadeIn(bottom_4_call_objects)
		)
		self.play(
			TransformFromCopy(bottom_4_call_disks, bottom_4_call_disks_copy)
		)
		self.wait()

		print_call_3 = TextMobject("pm(2, 1)")
		print_call_3.scale(0.6)
		print_call_3.move_to(DOWN * 2.5 + RIGHT * 1.2)
		self.play(
			FadeIn(print_call_3)
		)

		print_call_3_green = print_call_3.copy()
		print_call_3_green.set_color(GREEN_SCREEN)

		self.repr_print_call(print_call_3, 3, [[1, 0, 1, None]], index=[2, 1, 1])
		self.play(
			Transform(print_call_3, print_call_3_green)
		)

		bottom_call_4_green = bottom_call_4.copy()
		bottom_call_4_green.set_color(GREEN_SCREEN)

		self.animate_hanoi(bottom_4_call_objects[0], bottom_4_call_rods, [[1, 0, 1, None]])
		self.play(
			Transform(bottom_call_4, bottom_call_4_green)
		)

		self.animate_hanoi(right_call_objects[0], right_call_rods, [[1, 0, 1, None]])

		self.animate_hanoi(starting_objects[0], start_rods, [[1, 0, 1, None]])

		bottom_call_5 = TextMobject("pm(2, 3)")
		bottom_call_5.scale(0.6)
		bottom_call_5.move_to(DOWN * 0.5 + RIGHT * 3.5)
		self.play(
			FadeIn(bottom_call_5)
		)
		self.wait()

		bottom_call_5_green = bottom_call_5.copy()
		bottom_call_5_green.set_color(GREEN_SCREEN)

		self.repr_print_call(bottom_call_5, 3, [[1, 2, 1, None]], index=[2, 1 ,0])
		self.play(
			Transform(bottom_call_5, bottom_call_5_green)
		)
		self.wait()

		self.animate_hanoi(right_call_objects[0], right_call_rods, [[1, 2, 1, None]])
		self.wait()

		self.animate_hanoi(starting_objects[0], start_rods, [[1, 2, 1, None]])
		self.wait()

		bottom_call_6 = TextMobject("h(1, 1, 3)")
		bottom_call_6.scale(0.6)
		bottom_call_6.move_to(DOWN * 0.5 + RIGHT * 5.8)
		self.play(
			FadeIn(bottom_call_6)
		)
		self.wait(2)

		bottom_6_call_objects, bottom_6_call_rods, bottom_6_call_disks, bottom_6_call_disks_copy = self.repr_rec_call(bottom_call_6, 
			3, RIGHT * 8 + UP * 2, [3], index=[2, 2, 0])

		self.play(
			FadeIn(bottom_6_call_objects)
		)
		self.play(
			TransformFromCopy(bottom_6_call_disks, bottom_6_call_disks_copy)
		)
		self.wait(2)

		print_call_4 = TextMobject("pm(1, 3)")
		print_call_4.scale(0.6)
		print_call_4.move_to(DOWN * 2.5 + RIGHT * 5.8)
		self.play(
			FadeIn(print_call_4)
		)

		print_call_4_green = print_call_4.copy()
		print_call_4_green.set_color(GREEN_SCREEN)

		self.repr_print_call(print_call_4, 3, [[0, 2, 1, None]], index=[2, 2, 0])
		self.play(
			Transform(print_call_4, print_call_4_green)
		)

		bottom_call_6_green = bottom_call_6.copy()
		bottom_call_6_green.set_color(GREEN_SCREEN)

		self.animate_hanoi(bottom_6_call_objects[0], bottom_6_call_rods, [[0, 2, 1, None]])
		self.play(
			Transform(bottom_call_6, bottom_call_6_green)
		)


		right_call_green = right_call.copy()
		right_call_green.set_color(GREEN_SCREEN)
		self.animate_hanoi(right_call_objects[0], right_call_rods, [[0, 2, 1, None]])
		self.play(
			Transform(right_call, right_call_green)
		)

		self.animate_hanoi(starting_objects[0], start_rods, [[0, 2, 1, None]])

		first_call_green = first_call.copy()
		first_call_green.set_color(GREEN_SCREEN)
		self.play(
			Transform(first_call, first_call_green)
		)
		self.wait(19)

class VisualizeN4Hanoi(Scene):
	def construct(self):
		self.visualize_rec_n4()

	def setup_objects(self, n, index=[0, 0, 0], colors=COLORS):
		rods = [] # list of lists where each list is a stack of disks
		for i in range(3):
			rods.append([])
		color = WHITE
		if n < 4:
			height = 4
			offset = 2
		else:
			height = n + 1
			offset = n  - 1

		base = Rectangle(height=0.5, width=11, 
			fill_color=color, fill_opacity=1, color=color)
	
		first_rod = Rectangle(height=height, width=0.5,
			fill_color=color, fill_opacity=1, color=color)
		first_rod.set_color(WHITE)
		vertical_buffer = first_rod.get_height() / 2 + base.get_height() / 2
		first_rod.shift(LEFT * 4 + UP * vertical_buffer)

		second_rod = Rectangle(height=height, width=0.5,
			fill_color=color, fill_opacity=1, color=color)
		second_rod.shift(UP * vertical_buffer)

		third_rod = Rectangle(height=height, width=0.5,
			fill_color=color, fill_opacity=1, color=color)
		third_rod.shift(RIGHT * 4 +  UP * vertical_buffer)
		scale = 1
		structure = VGroup(first_rod, second_rod, third_rod, base)
		structure.shift(DOWN * offset)
		structure.scale(scale)


		disks, rods = self.set_starting_position(structure, n, rods, scale=scale, index=index, colors=colors)

		self.all_rods_and_disks = VGroup(*[structure] + disks)

		return self.all_rods_and_disks, rods

	def set_starting_position(self, structure, num_disks, rods, height=1, width=3, scale=1, index=[0, 0, 0], colors=COLORS):
		disks = []
		for i in range(num_disks):
			color = colors[i]
			disk = self.create_disk(color=color, height=height, 
				width=width, corner_radius=0.5, scale=scale)
			rods = self.place_disk_on_rod(structure, disk, rods, index[i])
			disks.append(disk)
			width = width - 0.5

		return disks, rods

	def create_disk(self, color=BLUE, height=1, width=3, corner_radius=0.5, scale=1):
		disk = RoundedRectangle(height=height, width=width, corner_radius=corner_radius, fill_opacity=1)
		disk.set_color(color)
		disk.scale(scale)
		return disk

	def place_disk_on_rod(self, structure, disk, rods, index, append=True):
		rod = rods[index]
		rod_rectangle = structure[index]
		disk_buffer = disk.get_height() / 2
		if len(rod) == 0:
			center = rod_rectangle.get_center() + DOWN * rod_rectangle.get_height() / 2 + UP * disk_buffer
			disk.move_to(center)	
		else:
			last_disk = rod[-1]
			disk.next_to(last_disk, UP, buff=0)
		if append:
			rods[index].append(disk)
		return rods


	def animate(self, n):
		moves = towers_of_hanoi(n)

		self.animate_hanoi(moves)

	def move_disk_from_rod(self, structure, rods, source, destination, run_time, wait_time, animate=True):
		animations = []
		source_rod = rods[source]
		source_rod_rect = structure[source]
		destination_rod = rods[destination]
		destination_rod_rect = structure[destination]
		disk = rods[source].pop()
		disk_buffer = disk.get_height() / 2
		
		# animation of removing disk from source rod
		first_point = source_rod_rect.get_center() + UP * source_rod_rect.get_height() / 2 + UP * disk_buffer + UP * 0.1
		moved_disk = disk.copy()
		moved_disk.move_to(first_point)
		animation = Transform(disk, moved_disk)
		if animate:
			self.play(
				animation,
				run_time=run_time
			)
		animations.append(animation)

		# animation of moving disk from source to top of destination
		second_point = destination_rod_rect.get_center() + UP * destination_rod_rect.get_height() / 2 + UP * disk_buffer + UP * 0.1
		moved_disk = disk.copy()
		moved_disk.move_to(second_point)
		animation = Transform(disk, moved_disk)
		if animate:
			self.play(
				animation,
				run_time=run_time
			)
		animations.append(animation)

		# animation of moving disk from top of destination onto actual rod
		moved_disk = disk.copy()
		self.place_disk_on_rod(structure, moved_disk, rods, destination, append=False)
		animation = Transform(disk, moved_disk)
		if animate:
			self.play(
				animation,
				run_time=run_time
			)
		animations.append(animation)

		rods[destination].append(disk)
		
		if wait_time:
			self.wait(wait_time)
		return animations

	def animate_hanoi(self, structure, rods, moves, animate=True):
		# moves is a list of sources and destinations
		for source, destination, run_time, wait_time in moves:
			animations = self.move_disk_from_rod(structure, rods, source, destination, run_time, wait_time, animate=animate)
		return animations

	def simul_rise_to_top(self, structure1, structure2, rods1, rods2, 
		source, destination, run_time, wait_time):
		source_rod1 = rods1[source]
		source_rod_rect1 = structure1[source]
		destination_rod1 = rods1[destination]
		destination_rod_rect1 = structure1[destination]
		disk1 = rods1[source].pop()
		disk_buffer1 = disk1.get_height() / 2
		
		# animation of removing disk from source rod
		first_point1 = source_rod_rect1.get_center() + UP * source_rod_rect1.get_height() / 2 + UP * disk_buffer1 + UP * 0.1
		moved_disk1 = disk1.copy()
		moved_disk1.move_to(first_point1)

		source_rod2 = rods2[source]
		source_rod_rect2 = structure2[source]
		destination_rod2 = rods2[destination]
		destination_rod_rect2 = structure2[destination]
		disk2 = rods2[source].pop()
		disk_buffer2 = disk2.get_height() / 2
		
		first_point2 = source_rod_rect2.get_center() + UP * source_rod_rect2.get_height() / 2 + UP * disk_buffer2 + UP * 0.1
		moved_disk2 = disk2.copy()
		moved_disk2.move_to(first_point2)

		self.play(
			Transform(disk1, moved_disk1),
			Transform(disk2, moved_disk2),
			run_time=run_time
		)

		# animation of moving disk from source to top of destination
		second_point1 = destination_rod_rect1.get_center() + UP * destination_rod_rect1.get_height() / 2 + UP * disk_buffer1 + UP * 0.1
		moved_disk1 = disk1.copy()
		moved_disk1.move_to(second_point1)

		second_point2 = destination_rod_rect2.get_center() + UP * destination_rod_rect2.get_height() / 2 + UP * disk_buffer2 + UP * 0.1
		moved_disk2 = disk2.copy()
		moved_disk2.move_to(second_point2)

		self.play(
			Transform(disk1, moved_disk1),
			Transform(disk2, moved_disk2),
			run_time=run_time
		)

		# animation of moving disk from top of destination onto actual rod
		moved_disk1 = disk1.copy()
		self.place_disk_on_rod(structure1, moved_disk1, rods1, destination, append=False)

		moved_disk2 = disk2.copy()
		self.place_disk_on_rod(structure2, moved_disk2, rods2, destination, append=False)

		self.play(
			Transform(disk1, moved_disk1),
			Transform(disk2, moved_disk2),
			run_time=run_time
		)

		rods1[destination].append(disk1)
		rods2[destination].append(disk2)
		
		if wait_time:
			self.wait(wait_time)


	def simulatenous_animate(self, structure1, structure2, rods1, rods2, moves):
		for source, destination, run_time, wait_time in moves:
			self.simul_rise_to_top(structure1, structure2, rods1, rods2, 
				source, destination, run_time, wait_time)

	def repr_rec_call(self, num_disks, shift, disk_index,
		scale=0.2, opacity=0.4, index=[0, 0, 0], colors=COLORS):
		objects, rods = self.setup_objects(num_disks, index=index, colors=colors)
		objects.scale(scale)
		disks = []
		for i in disk_index:
			disks.append(objects[i])
		disks = VGroup(*disks)
		disks_copy = disks.copy()
		scaled_shift = shift * scale
		disks_copy.shift(scaled_shift)
		for i in range(len(disks_copy)):
			disks_copy[i].set_fill(opacity=opacity)

		return objects, rods, disks, disks_copy

	def repr_print_call(self, call, num_disks, move, 
		scale=0.2, index=[0, 0, 0]):
		objects, rods = self.setup_objects(num_disks, index=index)
		objects.scale(scale)
		objects.next_to(call, DOWN * 1.2)
		self.play(
			FadeIn(objects)
		)
		self.wait()
		self.animate_hanoi(objects[0], rods, move)

	def visualize_rec_n4(self):
		starting_objects, rods, disks, disks_copy = self.repr_rec_call(4, 
			RIGHT * 4 + DOWN, [2, 3, 4], scale=0.8, index=[0] * 4)

		starting_objects[0][2].set_color(YELLOW)
		self.play(
			FadeIn(starting_objects)
		)
		self.wait(15)

		strategy = TextMobject("Strategy")
		strategy.move_to(UP * 3.5)
		self.play(
			FadeIn(strategy)
		)

		strategy_step_1 = TextMobject("Move top 3 disks to rod 2")
		strategy_step_1[4:7].set_color(ORANGE)
		strategy_step_1[7].set_color(BLUE)
		strategy_step_1[8:13].set_color(GREEN_SCREEN)
		strategy_step_1.scale(0.8)
		strategy_step_1.next_to(strategy, DOWN)
		self.play(
			Write(strategy_step_1)
		)
		self.wait()

		self.play(
			TransformFromCopy(disks, disks_copy),
			run_time=2
		)

		self.wait(5)


		group = VGroup(starting_objects, disks_copy)
		group_copy = group.copy()
		group_copy.scale(0.5 / 0.8)
		group_copy.shift(UP)
		self.play(
			Transform(group, group_copy),
			run_time=2
		)
		self.wait(3)

		subprob_obj, subprob_obj_rods, subprob_obj_disks, subprob_obj_disks_copy = self.repr_rec_call(3, 
			RIGHT * 8 + DOWN, [2, 3], scale=0.5, colors=COLORS[1:])
		subprob_obj_group = VGroup(subprob_obj, subprob_obj_disks_copy)
		subprob_obj_group.move_to(DOWN * 2.8)
		subprob_obj[0][1].set_color(YELLOW)
		self.play(
			FadeIn(subprob_obj)
		)
		self.wait(17)
		self.play(
			TransformFromCopy(subprob_obj_disks, subprob_obj_disks_copy),
			run_time=2
		)
		self.wait()

		self.simulatenous_animate(starting_objects[0], 
			subprob_obj[0], rods, subprob_obj_rods, 
			[[0, 1, 1, None], [0, 2, 1, None], [1, 2, 1, 2]])
		self.play(
			FadeOut(subprob_obj_disks_copy)
		)

		self.simulatenous_animate(starting_objects[0], 
			subprob_obj[0], rods, subprob_obj_rods, 
			[[0, 1, 1, 2]])

		self.wait(2)


		top_2_disks = VGroup(*subprob_obj_rods[2])
		top_2_disks_shifted = top_2_disks.copy()
		top_2_disks_shifted.shift((LEFT * 4 + UP) * 0.5)
		top_2_disks_shifted.set_fill(opacity=0.4)
		self.play(
			TransformFromCopy(top_2_disks, top_2_disks_shifted),
			run_time=2,
		)
		self.wait(3)

		self.simulatenous_animate(starting_objects[0], 
			subprob_obj[0], rods, subprob_obj_rods, 
			[[2, 0, 1, None], [2, 1, 1, None], [0, 1, 1, None]])

		self.wait(12)
		self.play(
			FadeOut(top_2_disks_shifted),
			FadeOut(disks_copy)
		)

		starting_objects_copy = starting_objects.copy()
		starting_objects_copy.move_to(DOWN)
		starting_objects_copy.scale(1.2)
		self.play(
			FadeOut(subprob_obj),
			Transform(starting_objects, starting_objects_copy),
			run_time=2
		)

		self.wait()

		strategy_step_2 = TextMobject("Move largest disk to rod 3")
		strategy_step_2[4:11].set_color(RED)
		strategy_step_2.scale(0.8)
		strategy_step_2.next_to(strategy_step_1, DOWN)
		self.play(
			Write(strategy_step_2)
		)

		self.animate_hanoi(starting_objects[0], rods, [[0, 2, 1, None]])
		self.wait()

		strategy_step_3 = TextMobject("Move top 3 disks from rod 2 to rod 3")
		strategy_step_3[4:7].set_color(ORANGE)
		strategy_step_3[7].set_color(BLUE)
		strategy_step_3[8:13].set_color(GREEN_SCREEN)
		strategy_step_3.scale(0.8)
		strategy_step_3.next_to(strategy_step_2, DOWN)
		self.play(
			Write(strategy_step_3)
		)
		self.wait()

		top_3_disks = VGroup(*rods[1])
		top_3_disks_shifted = top_3_disks.copy()
		top_3_disks_shifted.shift((RIGHT * 4 + UP) * 0.6)
		top_3_disks_shifted.set_fill(opacity=0.4)
		self.play(
			TransformFromCopy(top_3_disks, top_3_disks_shifted),
			run_time=2,
		)

		self.wait(2)

		group = VGroup(starting_objects, top_3_disks_shifted)
		group_copy = group.copy()
		group_copy.scale(0.5 / 0.6)
		group_copy.move_to(LEFT * 3.7 + DOWN * 1)
		self.play(
			Transform(group, group_copy),
			run_time=2
		)


		subprob_obj, subprob_obj_rods, subprob_obj_disks, subprob_obj_disks_copy = self.repr_rec_call(3, 
			LEFT * 4 + DOWN, [2, 3], scale=0.5, colors=COLORS[1:], index=[1, 1, 1])
		subprob_obj_group = VGroup(subprob_obj, subprob_obj_disks_copy)
		subprob_obj_group.move_to(RIGHT * 3.7 + DOWN * 1)
		subprob_obj[0][2].set_color(YELLOW)
		self.play(
			FadeIn(subprob_obj)
		)
		self.wait(2)
		self.play(
			TransformFromCopy(subprob_obj_disks, subprob_obj_disks_copy),
			run_time=2
		)
		self.wait(4)

		self.simulatenous_animate(starting_objects[0], 
			subprob_obj[0], rods, subprob_obj_rods, 
			[[1, 2, 1, None], [1, 0, 1, 1], [2, 0, 1, 2]])
		self.play(
			FadeOut(subprob_obj_disks_copy)
		)

		self.simulatenous_animate(starting_objects[0], 
			subprob_obj[0], rods, subprob_obj_rods, 
			[[1, 2, 1, 2]])

		top_2_disks = VGroup(*subprob_obj_rods[0])
		top_2_disks_shifted = top_2_disks.copy()
		top_2_disks_shifted.shift((RIGHT * 8 + UP) * 0.5)
		top_2_disks_shifted.set_fill(opacity=0.4)
		self.play(
			TransformFromCopy(top_2_disks, top_2_disks_shifted),
			run_time=2,
		)

		self.wait(3)

		self.simulatenous_animate(starting_objects[0], 
			subprob_obj[0], rods, subprob_obj_rods, 
			[[0, 1, 1, None], [0, 2, 1, 1], [1, 2, 1, None]])

		self.wait(4)
		self.play(
			FadeOut(starting_objects),
			FadeOut(subprob_obj),
			FadeOut(top_2_disks_shifted),
			FadeOut(strategy),
			FadeOut(strategy_step_1),
			FadeOut(strategy_step_2),
			FadeOut(strategy_step_3),
			FadeOut(top_3_disks_shifted)
		)
class Introduction(Scene):
	def construct(self):
		rectangle = ScreenRectangle()
		rectangle.set_width(8)
		rectangle.set_height(6)

		coming_up = TextMobject("Coming up ...")
		coming_up.scale(1.2)
		coming_up.next_to(rectangle, UP)
		self.play(
			FadeIn(rectangle),
			Write(coming_up)
		)
		self.wait(3)

		self.play(
			FadeOut(coming_up),
			FadeOut(rectangle)
		)
class DominoesAnimation(Scene):
	def construct(self):
		objects = self.show_steps()
		items = objects[-1]
		fadeouts = self.make_12_domino_animation(LEFT * 3.5 + DOWN * 1.5, items)

		self.play(
			*[FadeOut(obj) for obj in objects] + fadeouts
		)
		# self.make_11_domino_animation()
		# self.make_12_domino_animation()

	def show_steps(self):
		step_title = TextMobject("Recursive Problem Solving")
		step_title.scale(1.1)
		step_title.to_edge(UP * 0.5)
		h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
		h_line.next_to(step_title, DOWN)

		function_title = TextMobject(r"Let $f(n)$ be a recursive function")
		function_title.scale(0.9)
		function_title[3:7].set_color(BLUE)
		function_title.next_to(h_line, DOWN)

		items = TextMobject(
			r"1) Show $f(1)$ works (base case)" + "\\\\",
			r"2) Assume $f(n - 1)$ works" + "\\\\",
			r"3) Show $f(n)$ works using $f(n - 1)$",
		)

		items[0][6:10].set_color(RED)
		items[1][8:14].set_color(GREEN_SCREEN)
		items[2][6:10].set_color(BLUE)
		items[2][-6:].set_color(GREEN_SCREEN)

		colors = [RED, GREEN_SCREEN, BLUE]
		for i in range(len(items)):
			items[i].scale(0.7)
			items[i].to_edge(LEFT * 4, buff = LARGE_BUFF)

		items.next_to(function_title, DOWN)
		items[1].shift(UP * 0.1)
		items[2].shift(UP * 0.2)

		self.play(
			Write(step_title),
			ShowCreation(h_line),
			run_time=2
		)

		self.wait(5)

		self.play(
			Write(function_title)
		)

		self.wait(3)

		self.play(
			Write(items[0])
		)
		self.wait(10)
		
		self.play(
			Write(items[1])
		)
		self.wait(14)
		
		self.play(
			Write(items[2])
		)
		self.wait(12)

		return [step_title, function_title, h_line, items]

	def make_12_domino_animation(self, start_position, items):
		# list of lists of domino sequences
		# ith element corresponds to sequence of ith domino
		master_domino_sequence = []
		degree_sequence = []
		i = 1
		while i < 68:
			degree_sequence.append(i)
			i *= 1.1
		degree_sequence.extend([degree_sequence[-1]] * 8)
		dominoes = self.make_object_sequence(degree_sequence, position=start_position)
		master_domino_sequence.append(dominoes)
		num_dominos = 12
		position = start_position + RIGHT * 0.5
		shift = RIGHT * 0.02
		for i in range(num_dominos):
			if i < 5:
				shift = RIGHT * 0
			else:
				shift += RIGHT * 0.02
			next_degree_sequence, next_dominos = self.make_next_degree_sequence(
				degree_sequence, dominoes, position, shift=shift)

			master_domino_sequence.append(next_dominos)
			
			position = position + RIGHT * 0.5
			degree_sequence = next_degree_sequence
			dominoes = next_dominos

		print([len(sequence) for sequence in master_domino_sequence])
		if num_dominos == 12:
			master_domino_sequence[-1][-1].shift(RIGHT * 1.4)
			master_domino_sequence[-2][-1].shift(RIGHT * 0.7)
			master_domino_sequence[-3][-1].shift(RIGHT * 0.3)
			master_domino_sequence[-4][-1].shift(RIGHT * 0.1)
		if num_dominos == 11:
			master_domino_sequence[-1][-1].shift(RIGHT * 1.1)
			master_domino_sequence[-2][-1].shift(RIGHT * 0.7)
			master_domino_sequence[-3][-1].shift(RIGHT * 0.3)
			master_domino_sequence[-4][-1].shift(RIGHT * 0.1)
		# master_domino_sequence[-5][-1].shift(RIGHT * 0.1)

		self.play(
			*[FadeIn(master_domino_sequence[i][0]) for i in range(len(master_domino_sequence))]
		)

		domino_group = VGroup(*[master_domino_sequence[i][0] for i in range(len(master_domino_sequence))])
		brace = Brace(domino_group, DOWN, buff=SMALL_BUFF)
		n_dominos = TextMobject(r"$n$ dominos")
		n_dominos.scale(0.8)
		n_dominos.next_to(brace, DOWN, buff=SMALL_BUFF)
		self.play(
			GrowFromCenter(brace),
			FadeIn(n_dominos),
			run_time=2
		)
		self.play(
			
		)

		self.wait(2)

		self.play(
			FadeOut(brace),
			FadeOut(n_dominos),
		)

		for i in range(1, len(master_domino_sequence[0])):
			master_domino_sequence[0][i].set_color(RED)

		red_first_domino = master_domino_sequence[0][0].copy()
		red_first_domino.set_color(RED)
		self.play(
			Indicate(master_domino_sequence[0][0], color=RED, scale_factor=1.1)
		)
		self.play(
			Transform(master_domino_sequence[0][0], red_first_domino)
		)

		for i in range(1, len(master_domino_sequence[-1])):
			master_domino_sequence[-1][i].set_color(BLUE)

		blue_last_domino = master_domino_sequence[-1][0].copy()
		blue_last_domino.set_color(BLUE)
		self.play(
			Indicate(master_domino_sequence[-1][0], color=BLUE, scale_factor=1.1)
		)
		self.play(
			Transform(master_domino_sequence[-1][0], blue_last_domino)
		)

		self.wait(3)

		items_1_copy = items[1].copy()
		items_1_copy.set_fill(opacity=0.5)
		items_2_copy = items[2].copy()
		items_2_copy.set_fill(opacity=0.5)
		self.play(
			Transform(items[1], items_1_copy),
			Transform(items[2], items_2_copy),
		)

		self.domino_falling_animation(master_domino_sequence[0][0], 
			start_position + LEFT * 2.5)

		items_0_copy = items[0].copy()
		items_0_copy.set_fill(opacity=0.5)
		items_1_copy = items[1].copy()
		items_1_copy.set_fill(opacity=1)
		self.play(
			Transform(items[0], items_0_copy),
			Transform(items[1], items_1_copy),
		)

		self.play(
			FadeOut(master_domino_sequence[-1][0])
		)

		for i in range(1, len(master_domino_sequence[-2])):
			master_domino_sequence[-2][i].set_color(GREEN_SCREEN)

		green_last_domino = master_domino_sequence[-2][0].copy()
		green_last_domino.set_color(GREEN_SCREEN)
		self.play(
			Indicate(master_domino_sequence[-2][0], color=GREEN_SCREEN, scale_factor=1.1)
		)
		self.play(
			Transform(master_domino_sequence[-2][0], green_last_domino)
		)

		animation_sequences = []
		for i in range(len(master_domino_sequence[0]) - 1):
			transforms = []
			for sequence in master_domino_sequence[:len(master_domino_sequence) - 1]:
				transforms.append(ReplacementTransform(sequence[i], 
					sequence[i + 1]))
			animation_sequences.append(transforms)

		

		for animations in animation_sequences:
			self.play(*animations, run_time=0.01)
		self.wait()
		reset_animations = [ReplacementTransform(master_domino_sequence[i][-1], 
			master_domino_sequence[i][0]) for i in range(len(master_domino_sequence) - 1)]
		
		self.play(
			*reset_animations
		)
		self.wait()

		items_1_copy = items[1].copy()
		items_1_copy.set_fill(opacity=0.5)
		items_2_copy = items[2].copy()
		items_2_copy.set_fill(opacity=1)
		self.play(
			Transform(items[1], items_1_copy),
			Transform(items[2], items_2_copy),
		)

		self.wait()

		self.play(
			FadeInAndShiftFromDirection(master_domino_sequence[-1][0], direction=RIGHT),
			run_time=2
		)

		new_master_domino_sequence = []
		degree_sequence = []
		i = 1
		while i < 68:
			degree_sequence.append(i)
			i *= 1.1
		degree_sequence.extend([degree_sequence[-1]] * 8)
		dominoes = self.make_object_sequence(degree_sequence, position=start_position)
		new_master_domino_sequence.append(dominoes)
		num_dominos = 12
		position = start_position + RIGHT * 0.5
		shift = RIGHT * 0.02
		for i in range(num_dominos):
			if i < 5:
				shift = RIGHT * 0
			else:
				shift += RIGHT * 0.02
			next_degree_sequence, next_dominos = self.make_next_degree_sequence(
				degree_sequence, dominoes, position, shift=shift)

			new_master_domino_sequence.append(next_dominos)
			
			position = position + RIGHT * 0.5
			degree_sequence = next_degree_sequence
			dominoes = next_dominos

		print([len(sequence) for sequence in new_master_domino_sequence])
		if num_dominos == 12:
			new_master_domino_sequence[-1][-1].shift(RIGHT * 1.4)
			new_master_domino_sequence[-2][-1].shift(RIGHT * 0.7)
			new_master_domino_sequence[-3][-1].shift(RIGHT * 0.3)
			new_master_domino_sequence[-4][-1].shift(RIGHT * 0.1)
		if num_dominos == 11:
			new_master_domino_sequence[-1][-1].shift(RIGHT * 1.1)
			new_master_domino_sequence[-2][-1].shift(RIGHT * 0.7)
			new_master_domino_sequence[-3][-1].shift(RIGHT * 0.3)
			new_master_domino_sequence[-4][-1].shift(RIGHT * 0.1)
		# master_domino_sequence[-5][-1].shift(RIGHT * 0.1)

		for i in range(len(new_master_domino_sequence[0])):
			new_master_domino_sequence[0][i].set_color(RED)

		for i in range(len(new_master_domino_sequence[-1])):
			new_master_domino_sequence[-1][i].set_color(BLUE)

		for i in range(len(new_master_domino_sequence[-2])):
			new_master_domino_sequence[-2][i].set_color(GREEN_SCREEN)

		self.play(
			*[ReplacementTransform(master_domino_sequence[i][0], 
				new_master_domino_sequence[i][0]) for i in range(len(new_master_domino_sequence))]
		)

		self.wait()

		animation_sequences = []
		for i in range(len(new_master_domino_sequence[0]) - 1):
			transforms = []
			for sequence in new_master_domino_sequence:
				transforms.append(ReplacementTransform(sequence[i], 
					sequence[i + 1]))
			animation_sequences.append(transforms)

		for animations in animation_sequences:
			self.play(*animations, run_time=0.01)
		self.wait(7)

		self.play(
			*[FadeOut(sequence[-1]) for sequence in new_master_domino_sequence]
		)
		self.wait()

		self.play(
			*[FadeIn(new_master_domino_sequence[i][0]) for i in range(len(new_master_domino_sequence))]
		)

		self.wait()

		items_0_copy = items[0].copy()
		items_0_copy.set_fill(opacity=1)
		items_2_copy = items[2].copy()
		items_2_copy.set_fill(opacity=0.5)
		self.play(
			Transform(items[0], items_0_copy),
			Transform(items[2], items_2_copy),
		)

		self.play(
			Indicate(new_master_domino_sequence[0][0], color=RED, scale_factor=1.1)
		)

		self.play(
			Indicate(new_master_domino_sequence[0][0], color=RED, scale_factor=1.1)
		)

		self.play(
			Indicate(new_master_domino_sequence[0][0], color=RED, scale_factor=1.1)
		)

		self.wait()

		items_0_copy = items[0].copy()
		items_0_copy.set_fill(opacity=0.5)
		items_1_copy = items[1].copy()
		items_1_copy.set_fill(opacity=1)
		self.play(
			Transform(items[0], items_0_copy),
			Transform(items[1], items_1_copy),
		)

		for i in range(7):
			self.play(
				Indicate(new_master_domino_sequence[-2][0], color=GREEN_SCREEN, scale_factor=1.1)
			)


		self.wait()


		items_1_copy = items[1].copy()
		items_1_copy.set_fill(opacity=0.5)
		items_2_copy = items[2].copy()
		items_2_copy.set_fill(opacity=1)
		self.play(
			Transform(items[1], items_1_copy),
			Transform(items[2], items_2_copy),
		)
		for i in range(5):
			self.play(
				Indicate(new_master_domino_sequence[-2][0], color=GREEN_SCREEN, scale_factor=1.1),
				Indicate(new_master_domino_sequence[-1][0], color=BLUE, scale_factor=1.1)
			)
		self.wait()

		blue_domino_shifted = new_master_domino_sequence[-1][0].copy()
		blue_domino_shifted.shift(RIGHT * 2)
		self.play(
			Transform(new_master_domino_sequence[-1][0], blue_domino_shifted),
			run_time=2
		)
		self.wait(5)

		blue_domino_shifted.shift(LEFT * 2)
		self.play(
			Transform(new_master_domino_sequence[-1][0], blue_domino_shifted),
			run_time=2
		)
		self.wait(5)


		fadeouts = [FadeOut(sequence[0]) for sequence in new_master_domino_sequence]

		return fadeouts

	def domino_falling_animation(self, domino, start_position):
		degree_sequence = []
		i = 1
		while i < 90:
			degree_sequence.append(i)
			i *= 1.3
		degree_sequence.extend([degree_sequence[-1]] * 8)
		dominoes = self.make_object_sequence(degree_sequence, position=start_position)
		for i in range(len(dominoes)):
			dominoes[i].set_color(RED)

		self.play(
			TransformFromCopy(domino, dominoes[0])
		)
		self.wait()

		for i in range(len(dominoes) - 1):
			self.play(
				ReplacementTransform(dominoes[i], dominoes[i + 1]), run_time=0.01
			)

		self.wait()

		self.play(
			FadeOut(dominoes[-1])
		)

		self.wait()

	def make_11_domino_animation(self, start_position):
				# list of lists of domino sequences
		# ith element corresponds to sequence of ith domino
		master_domino_sequence = []
		degree_sequence = []
		i = 1
		while i < 68:
			degree_sequence.append(i)
			i *= 1.1
		degree_sequence.extend([degree_sequence[-1]] * 8)
		dominoes = self.make_object_sequence(degree_sequence, position=start_position)
		master_domino_sequence.append(dominoes)
		num_dominos = 11
		position = start_position + RIGHT * 0.5
		shift = RIGHT * 0.02
		for i in range(num_dominos):
			if i < 5:
				shift = RIGHT * 0
			else:
				shift += RIGHT * 0.02
			next_degree_sequence, next_dominos = self.make_next_degree_sequence(
				degree_sequence, dominoes, position, shift=shift)

			master_domino_sequence.append(next_dominos)
			
			position = position + RIGHT * 0.5
			degree_sequence = next_degree_sequence
			dominoes = next_dominos

		print([len(sequence) for sequence in master_domino_sequence])
		if num_dominos == 12:
			master_domino_sequence[-1][-1].shift(RIGHT * 1.4)
			master_domino_sequence[-2][-1].shift(RIGHT * 0.7)
			master_domino_sequence[-3][-1].shift(RIGHT * 0.3)
			master_domino_sequence[-4][-1].shift(RIGHT * 0.1)
		if num_dominos == 11:
			master_domino_sequence[-1][-1].shift(RIGHT * 1.1)
			master_domino_sequence[-2][-1].shift(RIGHT * 0.7)
			master_domino_sequence[-3][-1].shift(RIGHT * 0.3)
			master_domino_sequence[-4][-1].shift(RIGHT * 0.1)
		# master_domino_sequence[-5][-1].shift(RIGHT * 0.1)
		animation_sequences = []
		for i in range(len(master_domino_sequence[0]) - 1):
			transforms = []
			for sequence in master_domino_sequence:
				transforms.append(ReplacementTransform(sequence[i], 
					sequence[i + 1]))
			animation_sequences.append(transforms)

		self.play(*[FadeIn(master_domino_sequence[i][0]) 
			for i in range(len(master_domino_sequence))])
		for animations in animation_sequences:
			self.play(*animations, run_time=0.01)
		self.wait()
		self.play(
			*[FadeOut(sequence[-1]) for sequence in master_domino_sequence]
		)
		self.wait()

	def make_next_degree_sequence(self, prev_degree_sequence, prev_dominoes, position, shift=0*RIGHT):
		candidate_degrees = [i for i in np.arange(0, 90, 0.5)]
		candidate_dominos = self.make_object_sequence(candidate_degrees, position=position)
		for domino in candidate_dominos[45:]:
			domino.shift(shift)
		current_domino = candidate_dominos[0]
		top_right_corners = []
		for domino in prev_dominoes:
			top_right_corner = self.get_top_right_corner(domino)
			top_right_corners.append(top_right_corner)
		point = None
		index = 0
		for i, corner in enumerate(top_right_corners):
			current_domino_left_edge_x = self.get_top_left_corner(current_domino)[0]
			x_diff = current_domino_left_edge_x - corner[0]
			if x_diff > 0 and x_diff < 0.5:
				point = corner
				index = i

		new_degree_sequence = [0] * (index + 1)
		new_domino_sequence = self.make_object_sequence(new_degree_sequence, position=position)
		for i in range(index + 1, len(prev_dominoes)):
			prev_domino_state = prev_dominoes[i]
			next_state_index = self.get_index_next_degree_state(prev_domino_state, candidate_dominos)
			next_degree = candidate_degrees[next_state_index]
			next_domino = candidate_dominos[next_state_index]
			if next_degree == 0:
				new_degree_sequence.append(new_degree_sequence[i - 1])
				new_domino_sequence.append(new_domino_sequence[i - 1].copy())
				prev_degree_sequence[i] = prev_degree_sequence[i - 1]
				prev_dominoes[i] = prev_dominoes[i - 1].copy()
			else:
				new_degree_sequence.append(next_degree)
				new_domino_sequence.append(next_domino)
		
		print(new_degree_sequence)

		return new_degree_sequence, new_domino_sequence

	def get_index_next_degree_state(self, prev_domino_state, candidate_dominos):
		index = 0
		top_right_corner_prev = self.get_top_right_corner(prev_domino_state)
		# prev_domino_state.set_stroke(color=RED)
		# self.play(FadeIn(prev_domino_state))
		for i, candidate in enumerate(candidate_dominos):
			cand_bottom_left_corner = self.get_bottom_left_corner(candidate)
			cand_top_left_corner = self.get_top_left_corner(candidate)
			distance = self.find_point_line_distance(top_right_corner_prev, 
				cand_bottom_left_corner, cand_top_left_corner)

			# print(distance)
			if distance < 0.5 and distance > 0 and cand_top_left_corner[0] > top_right_corner_prev[0]:
				index = i
				break
			# candidate.set_stroke(color=BLUE)
			# dist_text = TextMobject("{0}".format(round(distance, 2)))
			# dist_text.scale(0.4)
			# dist_text.next_to(candidate, DOWN)
			# self.play(
			# 	FadeIn(candidate),
			# 	FadeIn(dist_text)
			# )

			# self.play(
			# 	FadeOut(candidate),
			# 	FadeOut(dist_text)
			# )
		return index


	def find_point_line_distance(self, point, p1, p2):
		# Distance between point and line defined by points p1 and p2
		# Equation from https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
		x_0, y_0 = point[0], point[1]
		x_1, y_1 = p1[0], p1[1]
		x_2, y_2 = p2[0], p2[1]
		distance_num = (y_2 - y_1) * x_0 - (x_2 - x_1) * y_0 + x_2 * y_1 - y_2 * x_1
		distance_denom = np.sqrt((y_2 - y_1) ** 2 + (x_2 - x_1) ** 2)
		return -distance_num / distance_denom



	def get_top_left_corner(self, domino):
		corners = domino.get_anchors()
		return corners[0]

	def get_top_right_corner(self, domino):
		corners = domino.get_anchors()
		return corners[1]

	def get_bottom_right_corner(self, domino):
		corners = domino.get_anchors()
		return corners[2]

	def get_bottom_left_corner(self, domino):
		corners = domino.get_anchors()
		return corners[3]

	def make_domino(self, degrees=0, position=ORIGIN):
		rectangle = Rectangle(height=2, width=0.2)
		rectangle.move_to(position)
		rectangle.set_stroke(color=GRAY)
		rectangle.set_fill(color=WHITE, opacity=0.7)
		radians = -TAU * degrees / 360
		rotation_point = rectangle.get_center() + DOWN * rectangle.get_height() / 2 + UP * SMALL_BUFF
		rectangle.rotate(radians, about_point=rotation_point)
		return rectangle

	def make_object_sequence(self, degree_sequence, position=ORIGIN):
		dominoes = []
		for degree in degree_sequence:
			dominoe = self.make_domino(degrees=degree, position=position)
			dominoes.append(dominoe)
		return dominoes

class GeneralizedHanoiPart1(Scene):
	def construct(self):
		n_objects, n_rods, hanoi_n_statement, print_move, scale = self.part_1()

	def part_1(self):
		problem_statement = TextMobject(r"Write a function hanoi($n$, start, end) that" + "\\\\"
			, r"outputs a sequence of steps to move $n$ disks" + "\\\\", 
			"from the start rod to the end rod")
		problem_statement[0][20].set_color(BLUE)
		problem_statement[0][22:27].set_color(RED)
		problem_statement[0][28:31].set_color(YELLOW)

		problem_statement[1][-6].set_color(BLUE)
		
		problem_statement[2][7:12].set_color(RED)
		problem_statement[2][-6:-3].set_color(YELLOW)

		problem_statement.scale(0.8)
		problem_statement.to_edge(UP * 0.5)
		self.wait()
		for i in range(len(problem_statement)):
			self.play(
				Write(problem_statement[i])
			)
			self.wait(3)

		example = TextMobject(r"hanoi(3, 1, 3)")
		example[6].set_color(BLUE)
		example[8].set_color(RED)
		example[10].set_color(YELLOW)
		example.scale(0.8)
		example.move_to(LEFT * 4 + UP * 1.5)

		self.play(
			FadeIn(example)
		)

		n_3_objects, n_3_rods = self.setup_objects(3)
		n_3_objects.scale(0.6)
		n_3_objects.shift(RIGHT * 3 + DOWN * 0.5)
		self.play(
			FadeIn(n_3_objects[0])
		)

		rod_labels = []
		for i in range(3):
			label = TextMobject("{0}".format(i + 1))
			label.scale(0.7)
			structure = n_3_objects[0]
			label.next_to(structure[i], UP)
			rod_labels.append(label)

		self.play(
			*[FadeIn(label) for label in rod_labels]
		)
		self.wait()

		self.wait(5)

		self.play(
			Indicate(n_3_objects[0][0], color=RED, scale_factor=1.1)
		)

		self.wait(2)

		self.play(
			FadeIn(n_3_objects[1:])
		)

		self.wait()

		self.play(
			Indicate(n_3_objects[0][2], color=YELLOW, scale_factor=1)
		)

		rod_3_yellow = n_3_objects[0][2].copy()
		rod_3_yellow.set_color(YELLOW)

		self.play(
			Transform(n_3_objects[0][2], rod_3_yellow)
		)

		self.wait()

		assumptions = TextMobject("Assumptions")
		assumptions.move_to(example.get_center())
		assumptions.scale(0.8)
		assumptions.shift(DOWN * 2)
		self.play(
			Write(assumptions)
		)

		assumption_1 = TextMobject(r"1 $\leq$ start $\leq$ 3")
		assumption_1.scale(0.8)
		assumption_1.next_to(assumptions, DOWN)
		assumption_1[2:7].set_color(RED)

		assumption_2 = TextMobject(r"1 $\leq$ end $\leq$ 3")
		assumption_2.scale(0.8)
		assumption_2.next_to(assumption_1, DOWN)
		assumption_2[2:5].set_color(YELLOW)

		assumption_3 = TextMobject(r"start $\neq$ end")
		assumption_3.scale(0.8)
		assumption_3.next_to(assumption_2, DOWN)
		assumption_3[:5].set_color(RED)
		assumption_3[7:].set_color(YELLOW)

		assumption_4 = TextMobject(r"n $\geq$ 1")
		assumption_4.scale(0.8)
		assumption_4.next_to(assumption_3, DOWN)
		assumption_4[0].set_color(BLUE)

		self.play(
			FadeIn(assumption_1),
			FadeIn(assumption_2)
		)

		self.wait(3)

		self.play(
			FadeIn(assumption_3)
		)

		self.wait(2)

		self.play(
			FadeIn(assumption_4)
		)

		self.wait(7)

		self.play(
			FadeOut(assumptions),
			FadeOut(assumption_1),
			FadeOut(assumption_2),
			FadeOut(assumption_3), 
			FadeOut(assumption_4)
		)

		steps = [
		r"1 $\rightarrow$ 3" + "\\\\", 
		r"1 $\rightarrow$ 2" + "\\\\", 
		r"3 $\rightarrow$ 2" + "\\\\", 
		r"1 $\rightarrow$ 3" + "\\\\",
		r"2 $\rightarrow$ 1" + "\\\\",
		r"2 $\rightarrow$ 3" + "\\\\",
		r"1 $\rightarrow$ 3"
		]

		steps_text = TextMobject(*steps)
		steps_text.scale(0.7)
		steps_text.next_to(example, DOWN)
		self.play(
			FadeIn(steps_text)
		)
		self.wait()

		interpretation = TextMobject(r"$a \rightarrow b$ means 'Move the top disk from rod $a$ to rod $b$' ")
		interpretation.scale(0.8)
		interpretation.move_to(DOWN * 2.5)

		self.play(
			*[FadeOut(label) for label in rod_labels]
		)

		self.wait()

		moves = towers_of_hanoi(3)
		for i in range(len(moves)):
			self.show_step(steps_text, i)
			self.animate_hanoi(n_3_objects[0], n_3_rods, [moves[i]])
			if i == 1:
				self.play(
					FadeIn(interpretation)
				)

		self.show_step(steps_text, len(steps_text))

		self.play(
			FadeOut(interpretation)
		)

		self.wait(5)

		print_move = TextMobject(r"pm(start, end) := print(start, $\rightarrow$, end)")
		print_move[3:8].set_color(RED)
		print_move[9:12].set_color(YELLOW)
		print_move[-12:-7].set_color(RED)
		print_move[-4:-1].set_color(YELLOW)

		print_move.scale(0.7)
		print_move.move_to(LEFT * 3 + DOWN * 3)
		self.play(
			Write(print_move[:15])
		)

		self.wait(5)

		self.play(
			Write(print_move[15:])
		)
		self.wait(2)

		fadeout_objects = [
		problem_statement[0][:14], 
		problem_statement[0][32:],
		problem_statement[1:],
		steps_text,
		example,
		n_3_objects,
		]
		fadeout_anims = [FadeOut(obj) for obj in fadeout_objects]

		hanoi_n_statement = problem_statement[0][14:32].copy()
		hanoi_n_statement.move_to(UP * 3.5)

		print_move_copy = print_move.copy()
		print_move_copy.move_to(DOWN * 3.5)

		transforms = [ReplacementTransform(problem_statement[0][14:32], hanoi_n_statement),
		Transform(print_move, print_move_copy)]

		self.play(
				*fadeout_anims + transforms,
				run_time=2
		)

		self.wait()

		COLORS = [VIOLET, PURPLE, BLUE, GREEN_E, GREEN_SCREEN, RED, BRIGHT_RED, ORANGE, GOLD, YELLOW]
		n_objects, n_rods = self.setup_objects(10, index=[0] * 10, colors=COLORS, 
			corner_radius=0.1, disk_height=0.4, width_offset=0.2)
		scale = 0.8
		n_objects.scale(0.8)
		self.play(
			FadeIn(n_objects)
		)

		self.wait(2)

		n_objects_brace = Brace(n_objects[1:], LEFT, buff=SMALL_BUFF)
		n_objects_label = TextMobject("n disks")
		n_objects_label.scale(0.7)
		n_objects_label.next_to(n_objects_brace, LEFT)
		self.play(
			GrowFromCenter(n_objects_brace),
			FadeIn(n_objects_label)
		)

		self.wait(3)

		self.play(
			FadeOut(n_objects_brace),
			FadeOut(n_objects_label)
		)

		self.wait(4)

		first_rod_disk = VGroup(*n_rods[0])
		first_rod_disk_shifted_faded = first_rod_disk.copy()
		first_rod_disk_shifted_faded.shift(RIGHT * 8 * scale)
		first_rod_disk_shifted_faded.set_fill(opacity=0.5)
		self.play(
			TransformFromCopy(first_rod_disk, first_rod_disk_shifted_faded),
			run_time=2,
		)
		self.wait(10)

		self.play(
			FadeOut(hanoi_n_statement), 
			FadeOut(print_move),
			FadeOut(first_rod_disk_shifted_faded),
			FadeOut(n_objects)
		)

		return n_objects, n_rods, hanoi_n_statement, print_move, scale

	def show_step(self, steps, num):
		if num == len(steps):
			step_copy = steps[num - 1].copy()
			step_copy.set_color(GREEN_SCREEN)
			step_copy.set_fill(opacity=0.5)
			self.play(
				Transform(steps[num - 1], step_copy)
			)
			return
		all_other_steps = []
		step_copy = steps[num].copy()
		step_copy.set_color(RED)
		step_copy.set_fill(opacity=1)
		
		for i in range(num):
			prev_step_copy = steps[i].copy()
			prev_step_copy.set_color(GREEN_SCREEN)
			prev_step_copy.set_fill(opacity=0.5)
			all_other_steps.append(prev_step_copy)
		
		all_other_steps.append(step_copy)
		
		for i in range(num + 1, len(steps)):
			next_step_copy = steps[i].copy()
			next_step_copy.set_fill(opacity=0.5)
			all_other_steps.append(next_step_copy)

		self.play(
			*[Transform(steps[i], all_other_steps[i]) for i in range(len(steps))]
		)



	def setup_objects(self, n, index=[0, 0, 0], colors=COLORS, corner_radius=0.5, disk_height=1, width_offset=0.5):
		rods = [] # list of lists where each list is a stack of disks
		for i in range(3):
			rods.append([])
		color = WHITE
		if n < 4:
			height = 4
			offset = 2
		else:
			height = 5
			offset = 3

		base = Rectangle(height=0.5, width=11, 
			fill_color=color, fill_opacity=1, color=color)
	
		first_rod = Rectangle(height=height, width=0.5,
			fill_color=color, fill_opacity=1, color=color)
		first_rod.set_color(WHITE)
		vertical_buffer = first_rod.get_height() / 2 + base.get_height() / 2
		first_rod.shift(LEFT * 4 + UP * vertical_buffer)

		second_rod = Rectangle(height=height, width=0.5,
			fill_color=color, fill_opacity=1, color=color)
		second_rod.shift(UP * vertical_buffer)

		third_rod = Rectangle(height=height, width=0.5,
			fill_color=color, fill_opacity=1, color=color)
		third_rod.shift(RIGHT * 4 +  UP * vertical_buffer)
		scale = 1
		structure = VGroup(first_rod, second_rod, third_rod, base)
		structure.shift(DOWN * offset)
		structure.scale(scale)


		disks, rods = self.set_starting_position(structure, n, rods, 
			height=disk_height, corner_radius=corner_radius, colors=colors, 
			scale=scale, index=index, width_offset=width_offset)

		self.all_rods_and_disks = VGroup(*[structure] + disks)

		return self.all_rods_and_disks, rods

	def set_starting_position(self, structure, num_disks, rods, height=1, 
		width=3, scale=1, corner_radius=0.5, index=[0, 0, 0], width_offset=0.5, colors=COLORS):
		disks = []
		for i in range(num_disks):
			color = colors[i]
			opacity = (10 - i) / 10
			disk = self.create_disk(color=color, height=height, 
				width=width, corner_radius=corner_radius, scale=scale, opacity=opacity)
			rods = self.place_disk_on_rod(structure, disk, rods, index[i])
			disks.append(disk)
			width = width - width_offset

		return disks, rods

	def create_disk(self, color=BLUE, height=1, width=3, corner_radius=0.5, scale=1, opacity=1):
		disk = RoundedRectangle(height=height, width=width, corner_radius=corner_radius, fill_opacity=1)
		disk.set_color(color)
		disk.scale(scale)
		return disk

	def place_disk_on_rod(self, structure, disk, rods, index, append=True):
		rod = rods[index]
		rod_rectangle = structure[index]
		disk_buffer = disk.get_height() / 2
		if len(rod) == 0:
			center = rod_rectangle.get_center() + DOWN * rod_rectangle.get_height() / 2 + UP * disk_buffer
			disk.move_to(center)	
		else:
			last_disk = rod[-1]
			disk.next_to(last_disk, UP, buff=0)
		if append:
			rods[index].append(disk)
		return rods


	def animate(self, n):
		moves = towers_of_hanoi(n)

		self.animate_hanoi(moves)

	def move_disk_from_rod(self, structure, rods, source, destination, run_time, wait_time, animate=True):
		
		source_rod = rods[source]
		source_rod_rect = structure[source]
		destination_rod = rods[destination]
		destination_rod_rect = structure[destination]
		disk = rods[source].pop()
		disk_buffer = disk.get_height() / 2
		
		# animation of removing disk from source rod
		first_point = source_rod_rect.get_center() + UP * source_rod_rect.get_height() / 2 + UP * disk_buffer + UP * 0.1
		moved_disk = disk.copy()
		moved_disk.move_to(first_point)
		animation = Transform(disk, moved_disk)
		if animate:
			self.play(
				animation,
				run_time=run_time
			)

		# animation of moving disk from source to top of destination
		second_point = destination_rod_rect.get_center() + UP * destination_rod_rect.get_height() / 2 + UP * disk_buffer + UP * 0.1
		moved_disk = disk.copy()
		moved_disk.move_to(second_point)
		animation = Transform(disk, moved_disk)
		if animate:
			self.play(
				animation,
				run_time=run_time
			)

		# animation of moving disk from top of destination onto actual rod
		moved_disk = disk.copy()
		self.place_disk_on_rod(structure, moved_disk, rods, destination, append=False)
		animation = Transform(disk, moved_disk)
		if animate:
			self.play(
				animation,
				run_time=run_time
			)
		rods[destination].append(disk)
		if wait_time:
			self.wait(wait_time)
		return rods

	def animate_hanoi(self, structure, rods, moves, animate=True):
		# moves is a list of sources and destinations
		for source, destination, run_time, wait_time in moves:
			self.move_disk_from_rod(structure, rods, source, destination, run_time, wait_time, animate=animate)
		return rods


class GeneralizedHanoiPart2(Scene):
	def construct(self):
		n_objects, n_rods, hanoi_n_statement, print_move, scale = self.part_1()
		self.part_2(n_objects, n_rods, hanoi_n_statement, print_move, scale)

	def part_1(self):
		problem_statement = TextMobject(r"Write a function hanoi($n$, start, end) that" + "\\\\"
			, r"outputs a sequence of steps to move $n$ disks" + "\\\\", 
			"from the start rod to the end rod")
		problem_statement[0][20].set_color(BLUE)
		problem_statement[0][22:27].set_color(RED)
		problem_statement[0][28:31].set_color(YELLOW)

		problem_statement[1][-6].set_color(BLUE)
		
		problem_statement[2][7:12].set_color(RED)
		problem_statement[2][-6:-3].set_color(YELLOW)

		problem_statement.scale(0.8)
		problem_statement.to_edge(UP * 0.5)
		# for i in range(len(problem_statement)):
		# 	self.play(
		# 		Write(problem_statement[i])
		# 	)

		n_3_objects, n_3_rods = self.setup_objects(3)
		n_3_objects.scale(0.6)
		n_3_objects.shift(RIGHT * 3 + DOWN * 0.5)
		# self.play(
		# 	FadeIn(n_3_objects[0])
		# )

		rod_labels = []
		for i in range(3):
			label = TextMobject("{0}".format(i + 1))
			label.scale(0.7)
			structure = n_3_objects[0]
			label.next_to(structure[i], UP)
			rod_labels.append(label)

		# self.play(
		# 	*[FadeIn(label) for label in rod_labels]
		# )
		# self.wait()


		example = TextMobject(r"hanoi(3, 1, 3)")
		example[6].set_color(BLUE)
		example[8].set_color(RED)
		example[10].set_color(YELLOW)
		example.scale(0.8)
		example.move_to(LEFT * 4 + UP * 1.5)

		# self.play(
		# 	FadeIn(example)
		# )

		# self.wait()

		# self.play(
		# 	Indicate(n_3_objects[0][0], color=RED, scale_factor=1.1)
		# )

		# self.wait()

		# self.play(
		# 	FadeIn(n_3_objects[1:])
		# )

		# self.play(
		# 	Indicate(n_3_objects[0][2], color=YELLOW, scale_factor=1)
		# )

		rod_3_yellow = n_3_objects[0][2].copy()
		rod_3_yellow.set_color(YELLOW)

		# self.play(
		# 	Transform(n_3_objects[0][2], rod_3_yellow)
		# )

		# self.wait()

		steps = [
		r"1 $\rightarrow$ 3" + "\\\\", 
		r"1 $\rightarrow$ 2" + "\\\\", 
		r"3 $\rightarrow$ 2" + "\\\\", 
		r"1 $\rightarrow$ 3" + "\\\\",
		r"2 $\rightarrow$ 1" + "\\\\",
		r"2 $\rightarrow$ 3" + "\\\\",
		r"1 $\rightarrow$ 3"
		]

		steps_text = TextMobject(*steps)
		steps_text.scale(0.7)
		steps_text.next_to(example, DOWN)
		# self.play(
		# 	FadeIn(steps_text)
		# )

		interpretation = TextMobject(r"$a \rightarrow b$ means 'Move the top disk from rod $a$ to rod $b$' ")
		interpretation.scale(0.8)
		interpretation.move_to(DOWN * 2.5)
		# self.play(
		# 	FadeIn(interpretation)
		# )

		# self.wait(3)

		# self.play(
		# 	*[FadeOut(label) for label in rod_labels]
		# )

		# self.play(
		# 	FadeOut(interpretation)
		# )


		# self.wait()

		# moves = towers_of_hanoi(3)
		# for i in range(len(moves)):
		# 	self.show_step(steps_text, i)
		# 	self.animate_hanoi(n_3_objects[0], n_3_rods, [moves[i]])

		# self.show_step(steps_text, len(steps_text))

		print_move = TextMobject(r"pm(start, end) := print(start, $\rightarrow$, end)")
		print_move[3:8].set_color(RED)
		print_move[9:12].set_color(YELLOW)
		print_move[-12:-7].set_color(RED)
		print_move[-4:-1].set_color(YELLOW)

		print_move.scale(0.7)
		print_move.move_to(LEFT * 3 + DOWN * 3)
		# self.play(
		# 	FadeIn(print_move)
		# )

		fadeout_objects = [
		problem_statement[0][:14], 
		problem_statement[0][32:],
		problem_statement[1:],
		steps_text,
		example,
		n_3_objects,
		]
		fadeout_anims = [FadeOut(obj) for obj in fadeout_objects]

		hanoi_n_statement = problem_statement[0][14:32].copy()
		hanoi_n_statement.move_to(UP * 3.5)

		print_move_copy = print_move.copy()
		print_move_copy.move_to(DOWN * 3.5)

		transforms = [ReplacementTransform(problem_statement[0][14:32], hanoi_n_statement),
		Transform(print_move, print_move_copy)]

		# self.play(
		# 		*fadeout_anims + transforms,
		# 		run_time=2
		# )

		# self.wait()

		COLORS = [VIOLET, PURPLE, BLUE, GREEN_E, GREEN_SCREEN, RED, BRIGHT_RED, ORANGE, GOLD, YELLOW]
		n_objects, n_rods = self.setup_objects(10, index=[0] * 10, colors=COLORS, 
			corner_radius=0.1, disk_height=0.4, width_offset=0.2)
		scale = 0.8
		n_objects.scale(0.8)
		# self.play(
		# 	FadeIn(n_objects)
		# )

		n_objects_brace = Brace(n_objects[1:], LEFT, buff=SMALL_BUFF)
		n_objects_label = TextMobject("n disks")
		n_objects_label.scale(0.7)
		n_objects_label.next_to(n_objects_brace, LEFT)
		# self.play(
		# 	GrowFromCenter(n_objects_brace),
		# 	FadeIn(n_objects_label)
		# )

		# self.wait()

		# self.play(
		# 	FadeOut(n_objects_brace),
		# 	FadeOut(n_objects_label)
		# )

		first_rod_disk = VGroup(*n_rods[0])
		first_rod_disk_shifted_faded = first_rod_disk.copy()
		first_rod_disk_shifted_faded.shift(RIGHT * 8 * scale)
		first_rod_disk_shifted_faded.set_fill(opacity=0.5)
		# self.play(
		# 	TransformFromCopy(first_rod_disk, first_rod_disk_shifted_faded),
		# 	run_time=2,
		# )
		# self.wait(2)

		# self.play(
		# 	FadeOut(hanoi_n_statement), 
		# 	FadeOut(print_move),
		# 	FadeOut(first_rod_disk_shifted_faded),
		# 	FadeOut(n_objects)
		# )
		# self.wait(2)

		return n_objects, n_rods, hanoi_n_statement, print_move, scale

	def part_2(self, n_objects, n_rods, hanoi_n_statement, print_move, scale):

		print_move.move_to(LEFT * 3 + DOWN * 3.1)

		self.play(
			FadeIn(hanoi_n_statement),
			FadeIn(print_move)
		)

		items = TextMobject(
			r"1) Show $f(1)$ works (base case)" + "\\\\",
			r"2) Assume $f(n - 1)$ works" + "\\\\",
			r"3) Show $f(n)$ works using $f(n - 1)$",
		)

		items[0][6:10].set_color(RED)
		items[1][8:14].set_color(GREEN_SCREEN)
		items[2][6:10].set_color(BLUE)
		items[2][-6:].set_color(GREEN_SCREEN)

		colors = [RED, GREEN_SCREEN, BLUE]
		for i in range(len(items)):
			items[i].scale(0.7)
			items[i].to_edge(LEFT * 4, buff = LARGE_BUFF)

		items.move_to(RIGHT * 4.1 + DOWN * 3.2)
		items[1].shift(UP * 0.1)
		items[2].shift(UP * 0.2)

		h_line = Line(DOWN * 2 + RIGHT * 2, DOWN * 2 + RIGHT * 8)
		h_line.next_to(items, UP)

		v_line = Line(h_line.get_start(), h_line.get_start() + DOWN * 8)
		
		self.play(
			FadeIn(items),
			FadeIn(h_line),
			FadeIn(v_line)
		)

		items_1_copy = items[1].copy()
		items_1_copy.set_fill(opacity=0.5)
		items_2_copy = items[2].copy()
		items_2_copy.set_fill(opacity=0.5)
		self.play(
			Transform(items[1], items_1_copy),
			Transform(items[2], items_2_copy)
		)

		self.wait(7)

		object_1, rod_1 = self.setup_objects(1, index=[0])
		object_1.scale(0.7)
		object_1[0][2].set_color(YELLOW)
		self.play(
			FadeIn(object_1)
		)
		self.wait(3)
		self.animate_hanoi(object_1[0], rod_1, [[0, 2, 1, None]])

		self.wait(2)

		base_case = TextMobject(r"$:=$ pm(start, end) if $n = 1$")
		base_case[5:10].set_color(RED)
		base_case[11:14].set_color(YELLOW)
		base_case[-3].set_color(BLUE)
		base_case.scale(0.6)
		base_case.shift(UP * 0.1)
		base_case.next_to(hanoi_n_statement, DOWN)
		hanoi_n_statement_left = hanoi_n_statement.copy()
		hanoi_n_statement_left.next_to(base_case, LEFT)
		first_equality = VGroup(hanoi_n_statement_left, base_case)
		first_equality.move_to(UP * 3)
		self.play(
			ReplacementTransform(hanoi_n_statement, hanoi_n_statement_left),
		)

		self.play(
			Write(base_case)
		)

		self.wait()

		self.play(
			FadeOut(object_1)
		)

		items_0_copy = items[0].copy()
		items_0_copy.set_fill(opacity=0.5)
		items_1_copy = items[1].copy()
		items_1_copy.set_fill(opacity=1)
		self.play(
			Transform(items[0], items_0_copy),
			Transform(items[1], items_1_copy)
		)

		self.wait()

		scale = scale * 0.8
		n_objects.scale(0.8)
		n_objects.shift(UP * 0.5)

		self.play(
			FadeIn(n_objects),
		)

		n_objects_brace = Brace(n_objects[1:], LEFT, buff=SMALL_BUFF)
		n_objects_label = TextMobject("n disks")
		n_objects_label.scale(0.7)
		n_objects_label.next_to(n_objects_brace, LEFT)
		self.play(
			GrowFromCenter(n_objects_brace),
			FadeIn(n_objects_label)
		)

		self.wait()

		self.play(
			FadeOut(n_objects_brace),
			FadeOut(n_objects_label)
		)

		first_rod_disk = VGroup(*n_rods[0])
		first_rod_disk_shifted_faded = first_rod_disk.copy()
		first_rod_disk_shifted_faded.shift(RIGHT * 8 * scale)
		first_rod_disk_shifted_faded.set_fill(opacity=0.5)
		self.play(
			TransformFromCopy(first_rod_disk, first_rod_disk_shifted_faded),
			run_time=2,
		)
		self.wait(2)

		n_objects_group = VGroup(n_objects, first_rod_disk_shifted_faded)
		n_objects_shifted_scaled = n_objects_group.copy()
		scale = scale * 0.8
		n_objects_shifted_scaled.scale(0.8)
		n_objects_shifted_scaled.shift(LEFT * 3.5)
		self.play(
			Transform(n_objects_group, n_objects_shifted_scaled)
		)

		self.wait()

		structure_shifted = n_objects[0].copy()
		structure_shifted.shift(RIGHT * 7)

		self.play(
			TransformFromCopy(n_objects[0], structure_shifted)
		)

		first_n_1_disk = VGroup(*n_rods[0][1:])
		first_n_1_disk_shifted_faded = first_n_1_disk.copy()
		first_n_1_disk_shifted_faded.shift(DOWN * n_rods[0][0].get_height())
		first_n_1_disk_shifted_faded.shift(RIGHT * 7)
		self.play(
			TransformFromCopy(first_n_1_disk, first_n_1_disk_shifted_faded),
			run_time=2,
		)

		self.wait(2)

		structure_shifted[1].set_color(YELLOW)
		self.wait()
		
		structure_shifted[1].set_color(WHITE)
		structure_shifted[2].set_color(YELLOW)
		self.wait()
		
		first_n_1_disk_shifted_faded.shift(RIGHT * 4 * scale)
		self.wait()
		
		structure_shifted[2].set_color(WHITE)
		structure_shifted[0].set_color(YELLOW)
		self.wait()

		first_n_1_disk_shifted_faded.shift(RIGHT * 4 * scale)
		self.wait()

		structure_shifted[0].set_color(WHITE)
		structure_shifted[1].set_color(YELLOW)
		self.wait()

		first_n_1_disk_shifted_faded.shift(LEFT * 8 * scale)
		self.wait()

		right_n_objects_group = VGroup(structure_shifted, first_n_1_disk_shifted_faded)
		self.play(
			FadeOut(right_n_objects_group)
		)

		self.wait()

		items_1_copy = items[1].copy()
		items_1_copy.set_fill(opacity=0.5)
		items_2_copy = items[2].copy()
		items_2_copy.set_fill(opacity=1)
		self.play(
			Transform(items[1], items_1_copy),
			Transform(items[2], items_2_copy)
		)

		self.wait(16)

		self.play(
			Indicate(n_objects[0][0], color=RED, scale_factor=1.1)
		)

		self.play(
			Indicate(n_objects[0][2], color=YELLOW, scale_factor=1.1)
		)

		self.wait(2)

		for i in range(4):
			self.play(
				Indicate(n_objects[0][1], color=GREEN_SCREEN, scale_factor=1.1)
			)

		self.wait(6)


		other_def = TextMobject(r"other $= 6 - ($start + end)")
		other_def[:5].set_color(GREEN_SCREEN)
		other_def[9:14].set_color(RED)
		other_def[15:18].set_color(YELLOW)
		other_def.scale(0.6)
		other_def[7].shift(RIGHT * 0.02)
		other_def.next_to(base_case, DOWN)

		n_objects_group_copy = n_objects_group.copy()
		n_objects_group_copy.scale(0.9)
		scale = scale * 0.9
		n_objects_group_copy.shift(DOWN * 0.5)

		first_rec_call = TextMobject(r"hanoi($n - 1$, start, other)")
		first_rec_call[6:9].set_color(BLUE)
		first_rec_call[10:15].set_color(RED)
		first_rec_call[16:21].set_color(GREEN_SCREEN)
		first_rec_call.scale(0.6)
		first_rec_call.next_to(other_def, DOWN)
		first_rec_call.shift(UP * 0.2)

		print_call = TextMobject("pm(start, end)")
		print_call[3:8].set_color(RED)
		print_call[9:12].set_color(YELLOW)
		print_call.scale(0.6)
		print_call.next_to(first_rec_call, DOWN)
		print_call.shift(UP * 0.2)

		second_call = TextMobject(r"hanoi($n - 1$, other, end)")
		second_call[6:9].set_color(BLUE)
		second_call[10:15].set_color(GREEN_SCREEN)
		second_call[16:19].set_color(YELLOW)
		second_call.scale(0.6)
		second_call.next_to(print_call, DOWN)
		second_call.shift(UP * 0.2)

		define_equal = base_case[:2].copy()


		code_group = VGroup(
			other_def, 
			first_rec_call, 
			print_call,
			second_call,
		)


		code_group.shift(RIGHT * 0.2)
		code_group = VGroup(code_group, base_case[2:])

		brace = Brace(code_group, LEFT, buff=SMALL_BUFF)
		define_equal.next_to(brace, LEFT)
		hanoi_n_statement_left_copy = hanoi_n_statement_left.copy()
		hanoi_n_statement_left_copy.next_to(define_equal, LEFT)
		self.play(
			Transform(hanoi_n_statement_left, hanoi_n_statement_left_copy),
			Transform(base_case[:2], define_equal),
			GrowFromCenter(brace),
			run_time=2
		)
		self.wait()

		self.play(
			Write(other_def)
		)

		self.wait(7)

		self.play(
			Transform(n_objects_group, n_objects_group_copy)
		)

		first_n_1_disk = VGroup(*n_rods[0][1:])
		first_n_1_disk_shifted_faded_actual = first_n_1_disk.copy()
		first_n_1_disk_shifted_faded_actual.shift(DOWN * n_rods[0][0].get_height())
		first_n_1_disk_shifted_faded_actual.shift(RIGHT * 4 * scale)
		for i in range(len(first_n_1_disk_shifted_faded_actual)):
			first_n_1_disk_shifted_faded_actual[i].set_fill(opacity=0.5)

		self.play(
			TransformFromCopy(first_n_1_disk, first_n_1_disk_shifted_faded_actual),
			run_time=2,
		)

		self.wait(10)

		

		right_n_objects_group.scale(0.9)
		right_n_objects_group.shift(DOWN * 0.5)

		self.play(
			FadeIn(right_n_objects_group)
		)

		self.wait(12)

		self.play(
			Write(first_rec_call),
		)

		self.play(
			FadeOut(first_n_1_disk_shifted_faded_actual)
		)


		first_n_1_disk_left = VGroup(*n_rods[0][1:])
		first_n_1_disk_shifted_left = first_n_1_disk_left.copy()
		first_n_1_disk_shifted_left.shift(RIGHT * 4 * scale + 
			DOWN * n_rods[0][0].get_height())
		self.play(
			Transform(first_n_1_disk_left, first_n_1_disk_shifted_left),
			run_time=2
		)
		n_rods[1].extend(n_rods[0][1:])
		n_rods[0] = [n_rods[0][0]]

		self.play(
			FadeOut(right_n_objects_group)
		)
		self.wait()

		structure_shifted[1].set_color(WHITE)
		structure_shifted[2].set_color(YELLOW)
		
		first_n_1_disk_shifted_faded.shift(RIGHT * 4 * scale)

		self.animate_hanoi(n_objects[0], n_rods, [[0, 2, 1, None]])
		self.wait()
		self.play(
			Write(print_call)
		)
		self.wait(12)

		self.play(
			FadeIn(right_n_objects_group)
		)

		self.wait(12)

		self.play(
			Write(second_call)
		)

		self.wait(3)

		first_n_1_disk_left = VGroup(*n_rods[1])
		first_n_1_disk_shifted_left = first_n_1_disk_left.copy()
		first_n_1_disk_shifted_left.shift(RIGHT * 4 * scale + 
			UP * n_rods[1][0].get_height())
		self.play(
			Transform(first_n_1_disk_left, first_n_1_disk_shifted_left)
		)

		self.wait(11)

		all_code = VGroup(hanoi_n_statement_left, base_case[:2], brace, code_group)

		code_group_copy = all_code.copy()
		code_group_copy.move_to(LEFT * 2.8 + UP * 0.5)
		print_move_copy = print_move.copy()
		print_move_copy.next_to(code_group_copy, DOWN * 0.5)

		self.play(
			FadeOut(first_n_1_disk_left),
			FadeOut(n_objects),
			FadeOut(right_n_objects_group),
			FadeOut(items),
			FadeOut(h_line),
			FadeOut(v_line),
			FadeOut(first_rod_disk_shifted_faded),
			Transform(all_code, code_group_copy),
			Transform(print_move, print_move_copy),
			run_time=2
		)

		n_objects[1:].shift(LEFT * 8 * scale)
		n_objects.move_to(RIGHT * 4)
		# n_objects.scale(1.1)
		self.play(
			FadeIn(n_objects)
		)

		self.wait(22)

		self.play(
			FadeOut(n_objects)
		)

		self.wait(20)

		self.play(
			FadeOut(all_code),
			FadeOut(print_move)
		)


	
	def show_step(self, steps, num):
		if num == len(steps):
			step_copy = steps[num - 1].copy()
			step_copy.set_color(GREEN_SCREEN)
			step_copy.set_fill(opacity=0.5)
			self.play(
				Transform(steps[num - 1], step_copy)
			)
			return
		all_other_steps = []
		step_copy = steps[num].copy()
		step_copy.set_color(RED)
		step_copy.set_fill(opacity=1)
		
		for i in range(num):
			prev_step_copy = steps[i].copy()
			prev_step_copy.set_color(GREEN_SCREEN)
			prev_step_copy.set_fill(opacity=0.5)
			all_other_steps.append(prev_step_copy)
		
		all_other_steps.append(step_copy)
		
		for i in range(num + 1, len(steps)):
			next_step_copy = steps[i].copy()
			next_step_copy.set_fill(opacity=0.5)
			all_other_steps.append(next_step_copy)

		self.play(
			*[Transform(steps[i], all_other_steps[i]) for i in range(len(steps))]
		)



	def setup_objects(self, n, index=[0, 0, 0], colors=COLORS, corner_radius=0.5, disk_height=1, width_offset=0.5):
		rods = [] # list of lists where each list is a stack of disks
		for i in range(3):
			rods.append([])
		color = WHITE
		if n < 4:
			height = 4
			offset = 2
		else:
			height = 5
			offset = 3

		base = Rectangle(height=0.5, width=11, 
			fill_color=color, fill_opacity=1, color=color)
	
		first_rod = Rectangle(height=height, width=0.5,
			fill_color=color, fill_opacity=1, color=color)
		first_rod.set_color(WHITE)
		vertical_buffer = first_rod.get_height() / 2 + base.get_height() / 2
		first_rod.shift(LEFT * 4 + UP * vertical_buffer)

		second_rod = Rectangle(height=height, width=0.5,
			fill_color=color, fill_opacity=1, color=color)
		second_rod.shift(UP * vertical_buffer)

		third_rod = Rectangle(height=height, width=0.5,
			fill_color=color, fill_opacity=1, color=color)
		third_rod.shift(RIGHT * 4 +  UP * vertical_buffer)
		scale = 1
		structure = VGroup(first_rod, second_rod, third_rod, base)
		structure.shift(DOWN * offset)
		structure.scale(scale)


		disks, rods = self.set_starting_position(structure, n, rods, 
			height=disk_height, corner_radius=corner_radius, colors=colors, 
			scale=scale, index=index, width_offset=width_offset)

		self.all_rods_and_disks = VGroup(*[structure] + disks)

		return self.all_rods_and_disks, rods

	def set_starting_position(self, structure, num_disks, rods, height=1, 
		width=3, scale=1, corner_radius=0.5, index=[0, 0, 0], width_offset=0.5, colors=COLORS):
		disks = []
		for i in range(num_disks):
			color = colors[i]
			opacity = (10 - i) / 10
			disk = self.create_disk(color=color, height=height, 
				width=width, corner_radius=corner_radius, scale=scale, opacity=opacity)
			rods = self.place_disk_on_rod(structure, disk, rods, index[i])
			disks.append(disk)
			width = width - width_offset

		return disks, rods

	def create_disk(self, color=BLUE, height=1, width=3, corner_radius=0.5, scale=1, opacity=1):
		disk = RoundedRectangle(height=height, width=width, corner_radius=corner_radius, fill_opacity=1)
		disk.set_color(color)
		disk.scale(scale)
		return disk

	def place_disk_on_rod(self, structure, disk, rods, index, append=True):
		rod = rods[index]
		rod_rectangle = structure[index]
		disk_buffer = disk.get_height() / 2
		if len(rod) == 0:
			center = rod_rectangle.get_center() + DOWN * rod_rectangle.get_height() / 2 + UP * disk_buffer
			disk.move_to(center)	
		else:
			last_disk = rod[-1]
			disk.next_to(last_disk, UP, buff=0)
		if append:
			rods[index].append(disk)
		return rods


	def animate(self, n):
		moves = towers_of_hanoi(n)

		self.animate_hanoi(moves)

	def move_disk_from_rod(self, structure, rods, source, destination, run_time, wait_time, animate=True):
		
		source_rod = rods[source]
		source_rod_rect = structure[source]
		destination_rod = rods[destination]
		destination_rod_rect = structure[destination]
		disk = rods[source].pop()
		disk_buffer = disk.get_height() / 2
		
		# animation of removing disk from source rod
		first_point = source_rod_rect.get_center() + UP * source_rod_rect.get_height() / 2 + UP * disk_buffer + UP * 0.1
		moved_disk = disk.copy()
		moved_disk.move_to(first_point)
		animation = Transform(disk, moved_disk)
		if animate:
			self.play(
				animation,
				run_time=run_time
			)

		# animation of moving disk from source to top of destination
		second_point = destination_rod_rect.get_center() + UP * destination_rod_rect.get_height() / 2 + UP * disk_buffer + UP * 0.1
		moved_disk = disk.copy()
		moved_disk.move_to(second_point)
		animation = Transform(disk, moved_disk)
		if animate:
			self.play(
				animation,
				run_time=run_time
			)

		# animation of moving disk from top of destination onto actual rod
		moved_disk = disk.copy()
		self.place_disk_on_rod(structure, moved_disk, rods, destination, append=False)
		animation = Transform(disk, moved_disk)
		if animate:
			self.play(
				animation,
				run_time=run_time
			)
		rods[destination].append(disk)
		if wait_time:
			self.wait(wait_time)
		return rods

	def animate_hanoi(self, structure, rods, moves, animate=True):
		# moves is a list of sources and destinations
		for source, destination, run_time, wait_time in moves:
			self.move_disk_from_rod(structure, rods, source, destination, run_time, wait_time, animate=animate)
		return rods

def towers_of_hanoi(n):
	return towers_of_hanoi_helper(n, 0, 2)

def towers_of_hanoi_helper(n, start, end):
	other = 3 - (start + end)
	if n == 1:
		return [[start, end, 1, None]]
	else:
		first = towers_of_hanoi_helper(n - 1, start, other)
		second = towers_of_hanoi_helper(1, start, end)
		last = towers_of_hanoi_helper(n - 1, other, end)
		return first + second + last



from big_ol_pile_of_manim_imports import *

class DrawBook(Scene):
	def construct(self):
		# round_rectangle = RoundedRectangle(fill_color=BLUE, fill_opacity=1)
		# round_rectangle.set_color(BLUE)
		# round_rectangle.shift(DOWN * 2.5)
		# self.play(FadeIn(round_rectangle))
		scale = 0.7
		stack = []
		lines = []
		book, front_page = draw_book(LEFT * 2, BLUE, scale=scale)
		self.play(FadeIn(book))

		stacked_book = book.copy()
		stacked_book.shift(RIGHT * 4 + DOWN * 2.5)
		stacked_book.rotate(TAU / 3.5)
		stack.append(stacked_book)
		self.play(ReplacementTransform(book, stacked_book))


		line1, line2 = self.set_lines(front_page, scale)
		lines.append([line1, line2])

		book2, front_page2 = draw_book(LEFT * 2, RED, scale=scale)
		self.play(FadeIn(book2))

		stacked_book2 = self.stack_book_on(book2, stacked_book, line1, line2, scale)
		stack.append(stacked_book2)

		line3, line4 = self.set_lines(front_page2, scale)
		lines.append([line3, line4])

		book3, front_page3 = draw_book(LEFT * 2, GREEN, scale=scale)
		self.play(FadeIn(book3))

		stacked_book3 = self.stack_book_on(book3, stacked_book2, line3, line4, scale)
		stack.append(stacked_book3)

		line5, line6 = self.set_lines(front_page3, scale)
		lines.append([line5, line6])

		book4, front_page4 = draw_book(LEFT * 2, ORANGE, scale=scale)
		self.play(FadeIn(book4))

		stacked_book4 = self.stack_book_on(book4, stacked_book3, line5, line6, scale)
		stack.append(stacked_book4)

		line7, line8 = self.set_lines(front_page4, scale)
		lines.append([line7, line8])

		book5, front_page5 = draw_book(LEFT * 2, PURPLE, scale=scale)
		self.play(FadeIn(book5))

		stacked_book5 = self.stack_book_on(book5, stacked_book4, line7, line8, scale)
		stack.append(stacked_book5)

		line9, line10 = self.set_lines(front_page5, scale)
		lines.append([line9, line10])

		book6, front_page6 = draw_book(LEFT * 2, YELLOW, scale=scale)
		self.play(FadeIn(book6))

		stacked_book6 = self.stack_book_on(book6, stacked_book5, line9, line10, scale)
		stack.append(stacked_book6)

		self.wait(2)
		final_stack = [s.copy() for s in stack]
		all_lines = []
		for line_pair in lines:
			for line in line_pair:
				all_lines.append(line.copy())

		text = TextMobject("We're looking for this book")
		text.shift(LEFT * 3.5 + DOWN * 2)
		arrow = Arrow(text.get_center() + RIGHT * 3, stack[1].get_center() + LEFT * 1.2)
		self.play(FadeIn(text), FadeIn(arrow))
		self.wait()
		self.play(Indicate(stack[1], color=GREEN_SCREEN, scale_factor=1.1))
		self.play(Indicate(stack[1], color=GREEN_SCREEN, scale_factor=1.1))
		self.play(Indicate(stack[1], color=GREEN_SCREEN, scale_factor=1.1))
		self.wait()

		self.play(FadeOut(text), FadeOut(arrow))

		top_book = self.remove_from_stack(stack, lines)
		self.play(FadeOut(top_book))

		top_book = self.remove_from_stack(stack, lines)
		self.play(FadeOut(top_book))

		top_book = self.remove_from_stack(stack, lines)
		self.play(FadeOut(top_book))

		top_book = self.remove_from_stack(stack, lines)
		self.play(FadeOut(top_book))

		top_book = self.remove_from_stack(stack, lines)
		self.play(Indicate(top_book, color=GREEN_SCREEN, scale_factor=1.1))

		self.wait(2)

		purpose = TextMobject("How is this relevant?")
		purpose.shift(UP * 3.5)
		self.play(Write(purpose))

		self.play(FadeOut(top_book), FadeOut(stack[0]))
		self.wait(2)
		
		lines = [[all_lines[i], all_lines[i + 1]] for i in range(0, len(all_lines), 2)]
		all_transforms = []
		for i in range(len(final_stack) - 1):
			transforms = [FadeIn(final_stack[i]), FadeIn(all_lines[i * 2]), FadeIn(all_lines[i * 2 + 1])]
			all_transforms.extend(transforms)
		all_transforms.append(FadeIn(final_stack[-1]))

		
		# stack_transforms = [FadeIn(s) for s in final_stack]
		# line_transforms = [FadeIn(line) for line in all_lines]
		# all_transforms = [FadeIn(final_stack[0]), FadeIn(all_lines[0]), FadeIn(all_lines[1]), FadeIn(final_stack[1])]
		self.play(*all_transforms)
		self.wait(6)

		ds = TextMobject("Data Structure")
		ds.shift(LEFT * 2.5 + DOWN)
		arrow = Arrow(ds.get_center() + RIGHT * 1.5, ds.get_center() + RIGHT * 3.5)
		self.play(FadeIn(ds), FadeIn(arrow), FadeOut(purpose))

		self.wait(3)
		self.play(FadeOut(ds), FadeOut(arrow))
		self.wait()

		algorithm = TextMobject("Here's the algorithm")
		algorithm.shift(UP * 3)
		self.play(FadeIn(algorithm))
		top_book = self.remove_from_stack(final_stack, lines)
		self.play(FadeOut(top_book))

		top_book = self.remove_from_stack(final_stack, lines)
		self.play(FadeOut(top_book))

		top_book = self.remove_from_stack(final_stack, lines)
		self.play(FadeOut(top_book))

		top_book = self.remove_from_stack(final_stack, lines)
		self.play(FadeOut(top_book))

		top_book = self.remove_from_stack(final_stack, lines)
		self.play(Indicate(top_book, color=GREEN_SCREEN, scale_factor=1.1))


		
		


	def remove_from_stack(self, stack, lines):
		top_book = stack.pop()
		top_book_copy = top_book.copy()
		current_lines = lines.pop()
		top_book_copy.shift(LEFT * 5)
		top_book_copy.rotate(-TAU / 3.5)
		transforms = [ReplacementTransform(top_book, top_book_copy)] + [FadeOut(line) for line in current_lines]
		self.play(*transforms, run_time=0.7)
		return top_book_copy


	def set_lines(self, front_page, scale):
		upper_left, upper_right, lower_right = front_page.get_vertices()[2], front_page.get_vertices()[3], front_page.get_vertices()[0]
		line = Line(upper_left, upper_right)
		start_point = line.get_unit_vector() * SMALL_BUFF * scale + line.get_start()
		other_line_unit = Line(upper_right, lower_right).get_unit_vector()
		line1 = Line(start_point, upper_right)
		line1.set_color(BLACK)
		line2 = Line(upper_right, lower_right - other_line_unit * SMALL_BUFF * scale)
		line2.set_color(BLACK)
		line1.shift(UP * 0.5)
		line2.shift(UP * 0.5)
		self.play(FadeIn(line1), FadeIn(line2))
		return line1, line2

	def stack_book_on(self, top_book, bottom_book, line1, line2, scale):

		stacked_book2 = top_book.copy()
		up_vector = UP * 0.85 + RIGHT * 0.3
		stacked_book2.move_to(bottom_book.get_center() + up_vector * scale)
		stacked_book2.rotate(TAU / 3.5)
		line1_copy = line1.copy()
		line2_copy = line2.copy()
		line1_copy.shift(DOWN * 0.5)
		line2_copy.shift(DOWN * 0.5)

		self.play(Transform(top_book, stacked_book2), Transform(line1, line1_copy), 
			Transform(line2, line2_copy))
		return top_book


def draw_book(position, color, scale=1, fill_color=BLACK, fill_opacity=1):
	book_components = []
		
	lower_left = DOWN * 2 + LEFT 
	upper_left = LEFT * 0.8 + UP
	upper_right = UP * 1.7 + RIGHT * 1.5
	lower_right = DOWN * 1.3 + RIGHT * 1.3
	front_page = Polygon(*[lower_left, upper_left, upper_right, lower_right], 
		color=color, fill_color=fill_color, fill_opacity=fill_opacity)
	# self.play(FadeIn(front_page))
	book_components.append(front_page)

	left_line = Line(front_page.get_vertices()[0], front_page.get_vertices()[1])
	left_line.set_color(color)
	original_line = left_line.copy()
	left_line.shift(LEFT * 0.7 + UP * 0.4)
	up_line = Line(front_page.get_vertices()[1], front_page.get_vertices()[2])
	up_line.set_color(color)
	up_line_right = up_line.copy()
	up_line_right.shift(LEFT * 0.3 + UP * 0.06)
	up_line_right.set_color(color)
	up_line_left = up_line.copy()
	up_line_left.shift(LEFT * 0.53 + UP * 0.20)
	up_line_left.set_color(color)

	up_line.shift(LEFT * 0.7 + UP * 0.4)
	up_line.set_color(color)
	# self.play(FadeIn(left_line) ,FadeIn(up_line))
	book_components.extend([left_line, up_line, up_line_left, up_line_right])



	curved_line_top = ArcBetweenPoints(upper_left, up_line.get_start(), angle=-TAU/5)
	curved_line_bottom = ArcBetweenPoints(lower_left, left_line.get_start(), angle=-TAU/5)
	curved_line_end = ArcBetweenPoints(upper_right, up_line.get_end(), angle=-TAU/8)
	curved_line_middle = ArcBetweenPoints(original_line.lerp(1/3, 2/3), 
		left_line.lerp(1/3, 2/3), angle=-TAU/5)
	curved_line_middle2 = ArcBetweenPoints(original_line.lerp(2/3, 1/3), 
		left_line.lerp(2/3, 1/3), angle=-TAU/5)

	curved_line_top.set_color(color)
	curved_line_bottom.set_color(color)
	curved_line_middle.set_color(color)
	curved_line_middle2.set_color(color)
	curved_line_end.set_color(color)


	book_components.extend([curved_line_top, curved_line_bottom, 
		curved_line_middle, curved_line_middle2, curved_line_end])

	book = VGroup(*book_components, fill_color=RED, fill_opacity=1)
	book.scale(scale)
	book.move_to(position)
	# book.set_color(color)
	return book, front_page


class Introduction(Scene):
	def construct(self):
		ds = TextMobject("What is a data structure?")
		ds.shift(LEFT * 3 + UP * 3)
		ds[-14:-1].set_color(BLUE)
		self.play(FadeIn(ds))
		self.wait()
		algorithm = TextMobject("What is an algorithm?")
		algorithm.shift(LEFT * 3)
		algorithm[-10:-1].set_color(GREEN)
		self.play(FadeIn(algorithm))

		self.wait()

		ds_definition = TextMobject("A data structure is a data organization, management,", 
			"and storage format that enables efficient access and modification.", 
			"More precisely, a data structure is a collection of data values,", 
			"the relationships among them, and the functions or operations",
			"that can be applied to the data")

		algorithm_def = TextMobject("An algorithm is a sequence of instructions, typically to solve",
			"a class of problems","or perform a computation. Algorithms are unambiguous specifications",
			"for performing calculation, data processing, automated reasoning, and other tasks.")

		ds_definition.scale(0.8)
		ds_definition[0][1:14].set_color(BLUE)
		ds_definition.move_to(UP * 2.5)
		algorithm_def.scale(0.8)
		algorithm_def[0][2:11].set_color(GREEN)

		self.play(FadeOut(ds), FadeOut(algorithm))
		
		self.play(Write(ds_definition), run_time=4)
		self.play(Write(algorithm_def), run_time=4)
		self.wait()

		default_char = create_computer_char(color=BLUE, scale=0.7, position=DOWN*2.5 + LEFT * 5)
		confused_character = create_confused_char(color=BLUE, scale=0.7, position=DOWN*2.5 + LEFT * 5)

		self.play(FadeInAndShiftFromDirection(default_char))
		
		
		thought_bubble = SVGMobject("thought")
		thought_bubble.scale(0.7)
		thought_bubble.move_to(DOWN * 1.2 + RIGHT * 2 + LEFT * 5.2)

		huh = TextMobject("Huh?")
		huh.scale(0.7)
		huh.set_color(BLACK)
		huh.move_to(thought_bubble.get_center() + UP * 0.2 + RIGHT * 0.05)

		self.play(ReplacementTransform(default_char, confused_character), FadeIn(thought_bubble),
			FadeIn(huh))
		self.wait(2)

		self.play(FadeOut(ds_definition), FadeOut(algorithm_def), FadeOut(thought_bubble), 
			FadeOut(huh), FadeOut(confused_character))



class BookShelf(Scene):
	def construct(self):
		combos = []
		for i in range(-3, 4, 2):
			for j in range(-5, 6, 2):
				combos.append([i, j])
		positions = [DOWN * i + RIGHT * j for i, j in combos]
		colors = [RED, BLUE, GREEN, PURPLE, YELLOW, ORANGE, GREY, GOLD_A, TEAL_C, DARK_BLUE]
		all_books = []		
		run_time = 1
		for k in range(len(positions)):
			book, _ = draw_book(positions[k], colors[k % len(colors)], scale=0.45)
			all_books.append(book)
			self.play(FadeInAndShiftFromDirection(book, direction=UP), run_time=run_time)
			run_time = run_time / 1.25
		self.wait()
		self.play(*[FadeOut(book) for book in all_books])
		rect = Rectangle(height=7, width=5, fill_color=LIGHT_BROWN, fill_opacity=1) 
		rect.set_color(LIGHT_BROWN)
		top_rect = Rectangle(height=1.5, width=4, color=DARK_BROWN, fill_color=BLACK, fill_opacity=1)
		top_rect.shift(UP * 2)
		middle_rect = Rectangle(height=1.5, width=4, color=DARK_BROWN, fill_color=BLACK, fill_opacity=1)
		bottom_rect = Rectangle(height=1.5, width=4, color=DARK_BROWN, fill_color=BLACK, fill_opacity=1)
		bottom_rect.shift(DOWN * 2)

		book_shelf = VGroup(rect, top_rect, middle_rect, bottom_rect)
		scale = 1
		book_shelf.scale(scale)
		self.play(FadeIn(book_shelf))

		top_shelf_books = self.generate_books(top_rect, scale=scale)
		middle_shelf_books = self.generate_books(middle_rect, scale=scale)
		bottom_shelf_books = self.generate_books(bottom_rect, scale=scale)
		self.play(*[FadeIn(book) for book in top_shelf_books], run_time=1.5)
		self.play(*[FadeIn(book) for book in middle_shelf_books], run_time=1.5)
		self.play(*[FadeIn(book) for book in bottom_shelf_books], run_time=1.5)
		all_books = top_shelf_books + middle_shelf_books + bottom_shelf_books
		index = 12

		target = TextMobject("We're looking for this book")
		target.move_to(RIGHT * 5 + UP * 2)
		target_arrow = Arrow(target.get_center() + LEFT * 1.9, target.get_center() + LEFT * 3.7)
		target.scale(0.7)
		self.wait()
		self.play(FadeIn(target), FadeIn(target_arrow))
		self.play(Indicate(all_books[index], color=GREEN_SCREEN, scale_factor=1.2))
		self.wait()
		self.play(FadeOut(target), FadeOut(target_arrow))
		self.binary_search(0, len(all_books), index, all_books)


		self.wait()
		book_copy = all_books[index].copy()
		book, front_page = draw_book(LEFT * 4, all_books[index].get_stroke_color(), scale=0.7)
		self.play(ReplacementTransform(all_books[index], book), run_time=2)
		ds = TextMobject("Data Structure")
		ds.scale(0.8)
		ds.move_to(RIGHT * 4.5)
		ds_arrow = Arrow(ds.get_center() + LEFT * 1.2, ds.get_center() + LEFT * 2.2)
		self.play(FadeIn(ds), FadeIn(ds_arrow))
		self.play(FadeOut(book), FadeIn(book_copy), run_time=2)
		all_books[index] = book_copy

		self.wait(2)
		self.play(FadeOut(ds), FadeOut(ds_arrow))
		self.wait()
		total_book_shelf = VGroup(book_shelf, all_books)
		total_book_shelf_copy = total_book_shelf.copy()
		total_book_shelf_copy.shift(LEFT * 5 + DOWN * 1.5)
		total_book_shelf_copy.scale(0.6)
		self.play(Transform(total_book_shelf, total_book_shelf_copy))

		rest_bookshelves = [self.make_bookshelf_with_books(0.6, 1) for _ in range(3)]
		book_shelves = [total_book_shelf] + [rest[0] for rest in rest_bookshelves]
		for i in range(1, len(book_shelves)):
			book_shelf = book_shelves[i]
			book_shelf.next_to(book_shelves[i - 1], RIGHT * 1.5)
			self.play(FadeIn(book_shelf))

		self.wait(3)

		library = TextMobject("Library")
		library.shift(UP * 3.7)
		self.play(FadeIn(library))

		

		fiction = TextMobject("Fiction")
		fiction.shift(UP * 2.5 + LEFT * 3.3)

		

		non_fiction = TextMobject("Non-fiction")
		non_fiction.shift(UP * 2.5 + RIGHT * 3.5)

		self.wait(3)

		left = Line(library.get_center() + DOWN * 0.3, fiction.get_center() + UP * 0.3)
		right = Line(library.get_center() + DOWN * 0.3, non_fiction.get_center() + UP * 0.3)
		self.play(FadeIn(fiction), FadeIn(non_fiction),
			FadeIn(left), FadeIn(right))

		self.wait(2)

		fantasy = TextMobject("Fantasy")
		fantasy.next_to(book_shelves[0], UP * 0.6)

		thriller = TextMobject("Thriller")
		thriller.next_to(book_shelves[1], UP)

		fiction_left = Line(fiction.get_center() + DOWN * 0.3, fantasy.get_center() + UP * 0.3)
		fiction_right = Line(fiction.get_center() + DOWN * 0.3, thriller.get_center() + UP * 0.3)

		biographical = TextMobject("Biographical")
		biographical.next_to(book_shelves[2], UP * 0.6)

		historical = TextMobject("Historical")
		historical.next_to(book_shelves[3], UP)

		non_fiction_left = Line(non_fiction.get_center() + DOWN * 0.3, biographical.get_center() + UP * 0.3)
		non_fiction_right = Line(non_fiction.get_center() + DOWN * 0.3, historical.get_center() + UP * 0.3)

		self.play(FadeIn(fantasy), FadeIn(thriller), FadeIn(biographical), FadeIn(historical),
			FadeIn(fiction_left), FadeIn(fiction_right), FadeIn(non_fiction_left), FadeIn(non_fiction_right))
		self.wait()

		self.wait(5)
		all_books_second = rest_bookshelves[0][1]
		index = 24

		target = TextMobject("We're looking for this book")
		target.scale(0.6)
		target.move_to(LEFT * 1.5 + DOWN * 3.8)
		target_arrow = Arrow(target.get_center(), all_books_second[index].get_center() + DOWN * 0.1)
		self.play(FadeIn(target), FadeIn(target_arrow))
		self.play(Indicate(all_books_second[index], color=GREEN_SCREEN))
		self.play(FadeOut(target), FadeOut(target_arrow))

		path = [library, left, fiction, fiction_right, thriller]
		path_copy = [elem.copy() for elem in path]
		[elem.set_color(GREEN_SCREEN) for elem in path_copy]
		for p, p_copy in zip(path, path_copy):
			self.play(Transform(p, p_copy))

		self.wait()
		
		self.binary_search(0, len(all_books_second), index, all_books_second)

		book_copy = all_books_second[index].copy()
		book, front_page = draw_book(UP * 2, all_books_second[index].get_stroke_color(), scale=0.4)
		self.play(ReplacementTransform(all_books_second[index], book), run_time=2)
		self.wait()


	def make_bookshelf_with_books(self, scale, scale_book_shelf):
		rect = Rectangle(height=7, width=5, fill_color=LIGHT_BROWN, fill_opacity=1) 
		rect.set_color(LIGHT_BROWN)
		top_rect = Rectangle(height=1.5, width=4, color=DARK_BROWN, fill_color=BLACK, fill_opacity=1)
		top_rect.shift(UP * 2)
		middle_rect = Rectangle(height=1.5, width=4, color=DARK_BROWN, fill_color=BLACK, fill_opacity=1)
		bottom_rect = Rectangle(height=1.5, width=4, color=DARK_BROWN, fill_color=BLACK, fill_opacity=1)
		bottom_rect.shift(DOWN * 2)
		book_shelf = VGroup(rect, top_rect, middle_rect, bottom_rect)

		top_shelf_books = self.generate_books(top_rect, scale=scale_book_shelf)
		middle_shelf_books = self.generate_books(middle_rect, scale=scale_book_shelf)
		bottom_shelf_books = self.generate_books(bottom_rect, scale=scale_book_shelf)
		all_books = top_shelf_books + middle_shelf_books + bottom_shelf_books
		total_book_shelf = VGroup(book_shelf, all_books)
		total_book_shelf.scale(scale)
		return total_book_shelf, all_books



	def generate_books(self, shelf, scale=1):
		books = []
		BUFF = RIGHT * 0.05 * scale
		center_shelf = shelf.get_center()
		bottom_left_corner_shelf = center_shelf + LEFT * shelf.get_width() / 2 + DOWN * shelf.get_height() / 2
		start_width, start_height = shelf.get_width() / 20 * scale, shelf.get_height() * 0.7 * scale
		width, height = start_width, start_height
		bottom_position = bottom_left_corner_shelf + BUFF + width * RIGHT + UP * SMALL_BUFF * scale / 2
		right_most_x = bottom_position[0] + shelf.get_width() - 0.3 * scale
		colors = [RED, BLUE, GREEN, PURPLE, YELLOW, ORANGE, GREY, GOLD_A, TEAL_C, DARK_BLUE]
		while bottom_position[0] < right_most_x:
			color = np.random.choice(colors)
			book = create_rectange(width, height, bottom_position)
			book.set_color(color)
			bottom_position = bottom_position + BUFF + width * RIGHT
			width = np.random.uniform() * 0.1 * scale + start_width * 0.75
			# width *= scale
			height = np.random.uniform() * 0.2 * scale + start_height * 0.75
			# height * scale
			books.append(book)
		return books

	def binary_search(self, start, end, index, books):
		mid = (start + end) // 2
		if mid == index:
			self.play(Indicate(books[index], color=GREEN_SCREEN), scale_factor=1.5)
			self.play(Indicate(books[index], color=GREEN_SCREEN), run_time=0.5)
			self.play(Indicate(books[index], color=GREEN_SCREEN), run_time=0.5)
			self.play(Indicate(books[index], color=GREEN_SCREEN), run_time=0.5)
			return
		elif mid < index:
			self.play(Indicate(books[mid],  color=BRIGHT_RED, scale_factor=1.5))
			self.binary_search(mid, end, index, books)
		else:
			self.play(Indicate(books[mid], color=BRIGHT_RED, scale_factor=1.5))
			self.binary_search(start, mid, index, books)



def create_rectange(width, height, bottom_position):
	center_x = bottom_position[0]
	center_y = bottom_position[1] + height / 2
	rect = Rectangle(height=height, width=width)
	rect.move_to(np.array([center_x, center_y, 0]))
	return rect


class Simplify(Scene):
	def construct(self):
		title = TextMobject(r"Let's simplify our earlier definitions $\ldots$")
		title.shift(UP * 3.5)
		self.play(FadeIn(title))
		self.wait(8)
		ds = TextMobject("A data structure is a way to organize information")
		ds.scale(0.8)
		ds[1:14].set_color(BLUE)
		ds.move_to(UP * 2.5)
		self.play(Write(ds), run_time=3)

		algorithm = TextMobject("A algorithm is a way to process information to reach an end goal")
		algorithm[1:10].set_color(GREEN)
		algorithm.scale(0.8)
		algorithm.move_to(UP * 1.5)
		self.play(Write(algorithm), run_time=4)
		self.wait(4)
		
		motivating_questions = TextMobject("Motivating Questions")
		motivating_questions.shift(UP * 0.5)
		self.play(FadeIn(motivating_questions))
		self.wait(3)
		question_1 = TextMobject("Why use one organization scheme over another?")
		question_1.scale(0.8)
		question_1.move_to(DOWN * 0.5)
		self.play(Write(question_1))
		self.wait(8)
		question_2 = TextMobject("How do we measure performance of these schemes?")
		question_2.scale(0.8)
		question_2.move_to(DOWN * 1.5)
		self.play(Write(question_2))
		self.wait(6)
		question_3 = TextMobject("What type of problems do these ideas solve?")
		question_3.scale(0.8)
		question_3.move_to(DOWN * 2.5)
		self.play(Write(question_3))
		self.wait(5)


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

	# eyes_and_smile = VGroup(left_circle, inner_left, right_circle, inner_right,
	# 	bottom_line, right_line, left_line, left_eye_brow, right_eye_brow)

	eyes_and_smile = VGroup(left_circle, inner_left, right_circle, inner_right,
		smile, left_eye_brow, right_eye_brow)

	character = VGroup(computer, inner_rectangle, eyes_and_smile)
	character.scale(scale)
	character.move_to(position)

	# computer.scale(scale)
	# eyes_and_smile.scale(scale)
	# inner_rectangle.scale(scale)
	# computer.shift(position)
	# eyes_and_smile.shift(position)
	# inner_rectangle.shift(position)

	# smaller_computer = computer.copy()
	# smaller_eyes_and_smile = eyes_and_smile.copy()
	# smaller_inner_rectangle = inner_rectangle.copy()
	# smaller_computer.scale(0.5)
	# smaller_eyes_and_smile.scale(0.5)
	# smaller_inner_rectangle.scale(0.5)
	# smaller_computer.set_color(GREEN)
	# smaller_eyes_and_smile.set_color(GREEN)
	# smaller_computer.shift(RIGHT * 5)
	# smaller_eyes_and_smile.shift(RIGHT * 5)
	# smaller_inner_rectangle.shift(RIGHT * 5)

	# arrow = Arrow(inner_rectangle.get_center() + RIGHT * 2.2, smaller_inner_rectangle.get_center() + LEFT)
	# arrow.rectangular_stem_width = 2
	# arrow.set_color_by_gradient(BLUE, GREEN)
	# self.play(FadeIn(smaller_computer), FadeIn(smaller_inner_rectangle), FadeIn(arrow),
	# 	FadeIn(smaller_eyes_and_smile))

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

	# computer.scale(scale)
	# eyes_and_smile.scale(scale)
	# inner_rectangle.scale(scale)
	# computer.shift(position)
	# eyes_and_smile.shift(position)
	# inner_rectangle.shift(position)

	return character


# def create_confused_cha


class ComputerCharacter(Scene):
	def construct(self):
		default_char = create_computer_char(color=BLUE, scale=1.5, position=LEFT*2)
		confused_character = create_confused_char(color=BLUE, scale=1.5, position=LEFT*2)

		self.play(FadeIn(default_char))
		self.play(ReplacementTransform(default_char, confused_character))
		thought_bubble = SVGMobject("thought")
		thought_bubble.move_to(UP * 2 + RIGHT)
		self.play(FadeIn(thought_bubble))
		self.wait()

		# eye_bottom = ArcBetweenPoints(ORIGIN, RIGHT * 0.7, color=BLACK)
		# eye_bottom2 = ArcBetweenPoints(ORIGIN , RIGHT * 0.7, color=BLACK)
		# eye_bottom2.shift(DOWN * SMALL_BUFF / 4)
		# eye_top = ArcBetweenPoints(ORIGIN, RIGHT * 0.7, angle=-TAU/3, color=BLACK)
		# eye_top2 = ArcBetweenPoints(ORIGIN, RIGHT * 0.7, angle=-TAU/3, color=BLACK)
		# eye_top2.shift(UP * SMALL_BUFF / 4)
		# surrounding_ellipse = Ellipse(width=0.6, height=0.3, fill_color=WHITE, fill_opacity=1, color=WHITE)
		# surrounding_ellipse.move_to(RIGHT * 0.35 + UP * 0.05)
		# circle = Circle(radius=0.14, fill_color=BLACK, fill_opacity=1, color=BLACK)
		# circle.shift(RIGHT*0.35 + UP * 0.05)
		# pupil = Circle(radius=0.05, fill_color=WHITE, fill_opacity=1, color=WHITE)
		# pupil.shift(RIGHT * 0.3 + UP * 0.1)

		# eye_components = [surrounding_ellipse, eye_bottom, eye_bottom2, eye_top, eye_top2, circle, pupil]
		# right_eye = VGroup(*eye_components)
		# right_eye.shift(UP * 0.2, RIGHT * 0.2)

		# left_eye = right_eye.copy()
		# right_eye_position = right_eye.get_center()
		# left_eye_position = np.array([-right_eye_position[0], right_eye_position[1], right_eye_position[2]])
		# left_eye.move_to(left_eye_position)
		# self.play(FadeIn(right_eye), FadeIn(left_eye))
		

		# self.play(FadeIn(surrounding_ellipse), FadeIn(eye_bottom), FadeIn(eye_top), FadeIn(eye_bottom2), 
		# 	FadeIn(eye_top2), FadeIn(circle), FadeIn(pupil))
		self.wait()


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

		self.play(DrawBorderThenFill(logo), 
			FadeInAndShiftFromDirection(reducible), 
			FadeIn(patreon),
			run_time=2)

		self.play(Write(subscribe), Write(support))

		self.wait(16)


class ChannelArt(Scene):
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

		logo = VGroup(*[hexagon] + triangles)
		logo.shift(LEFT * 3)
		logo.scale(0.55)

		logo2 = logo.copy()
		logo2.move_to(RIGHT * 3)
		# subscribe = TextMobject("Subscribe")
		# subscribe.move_to(LEFT * 2.5 + UP * 2.5)
		# reducible.shift(LEFT * 2.5)

		# patreon = ImageMobject("Patreon")
		# support = TextMobject("Support")
		# support.move_to(RIGHT * 2.5 + UP * 2.5)
		# patreon.scale(1.8)
		# patreon.shift(RIGHT * 2.5)

		# all_anims = [DrawBorderThenFill(hexagon)] + [DrawBorderThenFill(triangle) for triangle in triangles] + [FadeInAndShiftFromDirection(reducible)]

		self.play(DrawBorderThenFill(logo), 
			FadeIn(reducible), 
			DrawBorderThenFill(logo2),
			run_time=2)

		# self.play(Write(subscribe), Write(support))

		self.wait(7)



		



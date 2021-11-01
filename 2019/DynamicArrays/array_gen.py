from big_ol_pile_of_manim_imports import *

class ArrayGeneration(Scene):
    def construct(self):
        self.wait(3)
        rectangles = []
        num_rectangles = 70
        for i in range(num_rectangles):
            rectangle = Rectangle(height=1, width=0.125)
            rectangle.set_color(BLUE)
            rectangles.append(rectangle)
        rectangles[0].move_to(6 * LEFT)
        for i in range(1, len(rectangles)):
            rectangles[i].next_to(rectangles[i - 1], RIGHT, buff=0.05)

        # rectangle.set_points([np.array([0, 0, 0]), np.array([0, 1, 0]), np.array([1, 0, 0]), np.array([1, 1, 0])])
        # rectangle.flip(RIGHT)
        # rectangle.rotate(-3 * TAU / 8)
        # circle.set_fill(PINK, opacity=0.5)
        animations = [FadeIn(rectangle) for rectangle in rectangles]
        self.play(*animations, run_time=3)
        self.wait(6)

        pointer = Arrow(UP * 2, UP * 0.5)
        declaration = TextMobject("Computer Memory")
        declaration.move_to(UP * 2)
        self.play(FadeIn(pointer), Write(declaration))

        self.wait()

        self.play(FadeOut(pointer), FadeOut(declaration), run_time=1.5)
        self.wait()

        self.wait(3)

        halfway = num_rectangles // 2
        start_highlight = halfway - 2
        end_highlight = halfway + 2
        highlighted_rectangles = rectangles[start_highlight:end_highlight]
        for rectangle in highlighted_rectangles:
            self.play(LaggedStart(
                ApplyMethod, rectangle,
                lambda m : (m.set_color, RED),
                run_time = 1))
        self.wait()


        copy_highlighted_rectangles = [rectangle.copy() for rectangle in highlighted_rectangles]
        for i in range(len(copy_highlighted_rectangles)):
            copy_highlighted_rectangles[i].next_to(highlighted_rectangles[i], DOWN * 3)
            copy_highlighted_rectangles[i].set_color(YELLOW)
        transforms = [TransformFromCopy(orig_rectangle, new_rectangle) for orig_rectangle, new_rectangle in zip(highlighted_rectangles, copy_highlighted_rectangles)]
        self.play(*transforms)

        central_array = [Rectangle(height=1.5, width=2) for _ in range(4)]
        central_array[0].move_to(3*LEFT)
        central_array[0].set_color(YELLOW)
        for i in range(1, len(central_array)):
            central_array[i].next_to(central_array[i - 1], RIGHT, buff=0)
            central_array[i].set_color(YELLOW)
        transforms = [ReplacementTransform(orig_rectangle, new_rectangle) for orig_rectangle, new_rectangle in zip(copy_highlighted_rectangles, central_array)]
        fadeout_transforms = [FadeOut(rectangle) for rectangle in rectangles]
        all_transforms = fadeout_transforms + transforms
        self.play(*all_transforms)
        self.wait()

        new_memory = Rectangle(height=1.5, width=2)
        new_memory.set_color(BLUE)
        new_memory.next_to(central_array[-1], RIGHT, buff=0)
        self.play(GrowFromCenter(new_memory))

        self.wait()
        cross = Cross(new_memory)

        self.play(Write(cross))
        self.wait()

        self.play(FadeOut(cross), FadeOut(new_memory))
        self.wait(5)

        pointer = Arrow(UP * 2.25, UP * 0.75)
        declaration = TextMobject("Fixed Length Array")
        declaration.move_to(UP * 2.25)
        self.play(FadeIn(pointer), Write(declaration))

        self.wait(3)

        self.play(FadeOut(pointer), FadeOut(declaration), run_time=1.5)
        self.wait(2)

        text = TextMobject("What do we like about fixed length arrays?")
        text.shift(UP * 2)
        self.play(Write(text))
        self.wait(5)
        self.play(FadeOut(text))
        self.wait()


        text_objects = [TextMobject("A"), TextMobject("B"), TextMobject("C"), TextMobject("D")]
        for i, text_object in enumerate(text_objects):
            text_object.scale(1.5)
            text_object.next_to(central_array[i], UP * 2)
        copy_text_objs = []
        i = 0
        for text_object in text_objects:
            self.play(ShowCreation(text_object))
            copy_text_obj = text_object.copy()
            copy_text_obj.move_to(central_array[i].get_center())
            copy_text_obj.set_color(GREEN)
            copy_text_objs.append(copy_text_obj)
            self.play(ReplacementTransform(text_object, copy_text_obj))
            i += 1

        self.wait()

        index = [TextMobject(str(i)) for i in range(len(central_array))]
        for i, text in enumerate(index):
            text.next_to(central_array[i], DOWN * 1.8)
        self.play(*[Write(text) for text in index])

        self.wait(2)
        for i in range(len(index)):
            self.play(Indicate(copy_text_objs[i], color=GREEN_SCREEN), Indicate(index[i], color=GREEN_SCREEN), run_time=3)
        self.wait()
        self.play(*[FadeOut(text) for text in index])
        self.wait(4)


        left_character = create_confused_char(color=RED, scale=0.7, position=LEFT * 5.5)
        left_thought_bubble = SVGMobject("thought")
        left_thought_bubble.scale(1.2)
        left_thought_bubble.move_to(left_character.get_center() + UP * 2 + RIGHT * 2)
        done = TextMobject("Are we done?")
        done.scale(0.5)
        done.move_to(left_thought_bubble.get_center() + UP * 0.4 + RIGHT * 0.04)
        done.set_color(BLACK)

        self.play(FadeIn(left_character))
        self.play(FadeIn(left_thought_bubble), FadeIn(done))
        self.wait(2)

        right_character = create_computer_char(color=BLUE, scale=0.7, position=RIGHT * 5.5)
        right_thought_bubble = SVGMobject("thought")
        right_thought_bubble.flip()
        right_thought_bubble.scale(1.2)
        right_thought_bubble.move_to(right_character.get_center() + UP * 2 + LEFT * 2)
        simplify = TextMobject("Ha! Good one!")
        simplify.scale(0.5)
        simplify.move_to(right_thought_bubble.get_center() + UP * 0.4 + RIGHT * 0.04)
        simplify.set_color(BLACK)

        self.play(FadeIn(right_character))
        self.play(FadeIn(right_thought_bubble), FadeIn(simplify))
        self.wait(2)

        self.play(FadeOut(left_character), FadeOut(right_character), FadeOut(done), 
            FadeOut(simplify), FadeOut(left_thought_bubble), FadeOut(right_thought_bubble))
        self.wait(2)


        # text = TextMobject("We're done!")
        # text.shift(DOWN * 2)
        # self.play(Write(text))
        # self.wait(2)
        # cross = Cross(text)

        # self.play(Write(cross))
        # self.wait()

        # new_text = TextMobject("Just kidding!")
        # new_text.shift(DOWN * 2)
        # self.play(FadeOut(cross), Transform(text, new_text))
        # self.wait(2)
        # self.play(FadeOut(new_text), FadeOut(text))
        # self.wait()

        characters = [chr(ord('A') + i % 26) for i in range(20)]

        run_time = 1
        for char in characters:
            text = TextMobject(char)
            text.move_to(UP * 2)
            self.play(FadeInFrom(text, direction=UP), run_time=run_time)
            self.play(FadeOut(text), run_time=run_time)
            run_time /= 2

        self.wait(10)


        E = TextMobject("E")
        E.scale(1.5)
        E.move_to(UP * 2)
        self.play(ShowCreation(E))
        self.wait()
        self.play(E.set_color, RED, run_time=2)
        question_mark = TextMobject("?")
        question_mark.next_to(E, RIGHT)
        question_mark.set_color(RED)
        question_mark.scale(1.5)
        self.play(ShowCreation(question_mark))
        self.wait(6)
        self.play(FadeOut(E), FadeOut(question_mark))



        slideupwardtransforms = []
        array_copied = []
        for i, array in enumerate(central_array):
            array_copy = array.copy()
            array_copy.next_to(array, UP * 2 + LEFT * 2)
            array_copied.append(array_copy)
            central_array[i] = array_copy
            slideupwardtransforms.append(ReplacementTransform(array, array_copy))


        new_copies = []
        for i, obj in enumerate(copy_text_objs):
            copy_text_object = obj.copy()
            copy_text_object.move_to(array_copied[i].get_center())
            new_copies.append(copy_text_object)
            text_objects[i] = copy_text_object
            slideupwardtransforms.append(ReplacementTransform(obj, copy_text_object))
        # removingtexttransforms = [FadeOut(text) for text in text_objects]
        # slideupwardtransforms = slideupwardtransforms + removingtexttransforms
        self.play(*slideupwardtransforms)
        self.wait()

        all_elements = new_copies + array_copied
        
        array_length_5 = [Rectangle(height=1.5, width=2) for _ in range(5)]
        array_length_5[0].next_to(central_array[0], DOWN * 4)
        array_length_5[0].set_color(BLUE)
        for i in range(1, len(array_length_5)):
            array_length_5[i].next_to(array_length_5[i - 1], RIGHT, buff=0)
            array_length_5[i].set_color(BLUE)

        all_elements += array_length_5

        self.play(*[GrowFromCenter(array) for array in array_length_5])
        self.wait(10)
        self.play(*[ApplyWave(rectangle, color=RED) for rectangle in central_array])

        for i, text_object in enumerate(text_objects):
            copy_text_obj = text_object.copy()
            copy_text_obj.move_to(array_length_5[i].get_center())
            text_objects[i] = copy_text_obj
            all_elements.append(copy_text_obj)
            self.play(TransformFromCopy(text_object, copy_text_obj))

        E.next_to(array_length_5[-1], UP * 2)
        E.set_color(WHITE)
        self.play(ShowCreation(E))
        copy_E = E.copy()
        copy_E.move_to(array_length_5[-1].get_center())
        copy_E.set_color(GREEN)
        all_elements.append(copy_E)
        self.play(ReplacementTransform(E, copy_E))
        E = copy_E
        text_objects.append(E)



        array_length_6 = [Rectangle(height=1.5, width=2) for _ in range(6)]
        array_length_6[0].next_to(array_length_5[0], DOWN * 4)
        array_length_6[0].set_color(PURPLE)
        for i in range(1, len(array_length_6)):
            array_length_6[i].next_to(array_length_6[i - 1], RIGHT, buff=0)
            array_length_6[i].set_color(PURPLE)

        all_elements += array_length_6

        F = TextMobject("F")
        F.scale(1.5)
        F.next_to(array_length_6[-1], UP * 6)
        self.play(ShowCreation(F))
        self.play(F.set_color, RED)
        text_objects.append(F)
        all_elements.append(F)

        self.play(*[GrowFromCenter(array) for array in array_length_6])
        self.wait()
        self.play(*[ApplyWave(rectangle) for rectangle in array_length_5])

        for i, text_object in enumerate(text_objects):
            copy_text_obj = text_object.copy()
            copy_text_obj.move_to(array_length_6[i].get_center())
            all_elements.append(copy_text_obj)
            text_objects[i] = copy_text_obj
            if i == len(text_objects) - 1:
                copy_text_obj.set_color(GREEN)
                self.play(ReplacementTransform(text_object, copy_text_obj))
            else:
                self.play(TransformFromCopy(text_object, copy_text_obj))

        self.wait(5)
        self.play(*[FadeOut(obj) for obj in all_elements])
        self.wait()


class IntroduceCountingElements(Scene):
    def construct(self):
        intro_text = TextMobject("Is this really the right approach?")
        self.play(Write(intro_text))
        self.wait(6)

        intro_text2 = TextMobject("Can we quantify this?")
        self.play(Transform(intro_text, intro_text2))
        self.wait(11)
        intro_text3 = TextMobject("What do we measure?")
        self.play(Transform(intro_text, intro_text3))
        self.wait(13)
        self.play(FadeOut(intro_text))
        array = create_array(3, YELLOW, 4 * LEFT, 1, 1.5)
        self.play(*[FadeIn(arr) for arr in array])
        text = [TextMobject('A'), TextMobject('B'), TextMobject('C')]
        for i in range(len(array)):
            text[i].move_to(array[i].get_center() + UP)
        self.play(*[Write(tex) for tex in text])

        for i, tex in enumerate(text):
            text_copy = tex.copy()
            text_copy.move_to(array[i].get_center())
            text_copy.set_color(GREEN)
            self.play(ReplacementTransform(tex, text_copy))

        insertions = TextMobject("Insertions: 3")
        insertions.move_to(array[1].get_center() + DOWN)
        self.play(Write(insertions))

        self.wait(2)

        array2 = create_array(3, BLUE, 2 * RIGHT, 1, 1.5)
        self.play(*[GrowFromCenter(arr) for arr in array2])
        space = TextMobject("Space: 3")
        self.wait(2)
        space.move_to(array2[1].get_center() + DOWN)
        self.play(Write(space))
        self.wait(2)

        arrow_end = DOWN * 2.5 + RIGHT * 0.5

        pointer1 = Arrow(arrow_end, array[1].get_center() + DOWN * 1.5)
        pointer2 = Arrow(arrow_end, array2[1].get_center() + DOWN * 1.5)
        declaration = TextMobject("We'll use these two counts as metrics")
        declaration.move_to(arrow_end + DOWN * 0.3)
        self.play(FadeIn(pointer1), FadeIn(pointer2))
        self.play(Write(declaration))

        self.wait(11)

        self.play(FadeOut(pointer1), FadeOut(pointer2), FadeOut(declaration), run_time=1.5)
        self.wait(2)



class CountInsertions(Scene):
    def construct(self):
        self.count = 0
        self.prev_counter_text = TextMobject("Insertions: {0}".format(self.count))
        self.counter_position = 3.5 * UP + 5 * RIGHT
        self.prev_counter_text.move_to(self.counter_position)
        self.play(ShowCreation(self.prev_counter_text), run_time=2.5)
        self.next_counter_text = TextMobject("Insertions: {0}".format(self.count + 1))
        self.next_counter_text.move_to(self.counter_position)
        self.space = 0
        self.prev_counter_space = TextMobject("Space: {0}".format(self.space))
        self.space_counter_pos = 3 * UP + 5 * RIGHT
        self.prev_counter_space.move_to(self.space_counter_pos)
        self.play(ShowCreation(self.prev_counter_space), run_time=2.5)
        self.space = 3
        self.next_counter_space = TextMobject("Space: {0}".format(self.space + 1))
        self.next_counter_space.move_to(self.space_counter_pos)
        self.wait(2)
        run_time = 0.13
        start_pos = 3.5 * UP + 6 * LEFT
        num_rectangles = 4
        num_chars = 4
        color = BLUE
        new_pos = start_pos
        array = self.generate_array(num_rectangles, start_pos, BLUE, FadeIn, run_time)
        previous_chars = self.create_and_move_characters(num_chars, array, ReplacementTransform, run_time)
        colors = [BLUE]
        prev_array = array
        while num_rectangles < 18:
            new_pos = new_pos + array[0].get_height() * DOWN
            num_rectangles += 1
            num_chars += 1
            new_array = self.generate_array(num_rectangles, new_pos, 
                colors[num_rectangles % len(colors)], GrowFromCenter, run_time)

            previous_chars = self.create_and_move_second_set_chars(num_chars, new_array, prev_array, previous_chars, run_time)
            prev_array = new_array
        text_copy = previous_chars[-1].copy()
        text_copy.move_to(new_array[-1].get_center())
        text_copy.set_color(YELLOW)
        counter_transform = ReplacementTransform(self.prev_counter_text, self.next_counter_text)
        self.play(ReplacementTransform(previous_chars[-1], text_copy), run_time=0.01)
        self.wait(6)

    def generate_array(self, length, start_pos, color, anim, run_time):
        starting_array = []
        for i in range(length):
            rect = Rectangle(height=0.5, width=0.7)
            rect.set_color(color)
            starting_array.append(rect)
        starting_array[0].move_to(start_pos)
        for i in range(1, len(starting_array)):
            starting_array[i].next_to(starting_array[i - 1], RIGHT, buff=0)
        starting_array_animations = [anim(rect) for rect in starting_array]
        counter_transform = ReplacementTransform(self.prev_counter_space, self.next_counter_space)
        self.next_counter_space, self.prev_counter_space = self.update_space(), self.next_counter_space
        self.play(*starting_array_animations, counter_transform, run_time=run_time)
        return starting_array

    def create_and_move_characters(self, num_chars, target_array, transf, run_time):
        text_objects = []
        start_char = 'A'
        for i in range(num_chars):
            char = chr(ord(start_char) + i)
            text = TextMobject(char)
            text.scale(0.65)
            text.next_to(target_array[i], UP * 0.5)
            self.play(FadeIn(text), run_time=run_time)
            copy_text = text.copy()
            copy_text.move_to(target_array[i].get_center())
            copy_text.set_color(YELLOW)
            text_objects.append(copy_text)
            counter_transform = ReplacementTransform(self.prev_counter_text, self.next_counter_text)
            self.next_counter_text, self.prev_counter_text = self.update_counter(), self.next_counter_text
            self.play(transf(text, copy_text), counter_transform, run_time=run_time)
        return text_objects

    def create_and_move_second_set_chars(self, num_chars, target_array, prev_array, previous_text, run_time):
        text_objects = []
        prev_text = []
        new_char = chr(ord('A') + num_chars - 1)
        new_text = TextMobject(new_char)
        new_text.scale(0.65)
        new_text.next_to(target_array[-1], UP * 0.5)
        self.play(FadeIn(new_text), run_time=run_time)
        for i, text in enumerate(previous_text + [new_text]):
            if text != new_text:
                prev_text.append(text)
            text_copy = text.copy()
            text_copy.move_to(target_array[i].get_center())
            text_copy.set_color(YELLOW)
            text_objects.append(text_copy)
            counter_transform = ReplacementTransform(self.prev_counter_text, self.next_counter_text)
            self.next_counter_text, self.prev_counter_text = self.update_counter(), self.next_counter_text
            if text == new_text:
                self.play(ReplacementTransform(text, text_copy), counter_transform, run_time=run_time)
            else:
                self.play(TransformFromCopy(text, text_copy), counter_transform, run_time=run_time)

        removal_animations = [FadeOut(obj) for obj in prev_text + prev_array]
        self.play(*removal_animations, run_time=run_time)
        return text_objects

    def update_counter(self):
        self.count += 1
        new_counter_text = TextMobject("Insertions: {0}".format(self.count + 1))
        new_counter_text.move_to(self.counter_position)
        return new_counter_text

    def update_space(self):
        self.space += 1
        new_counter_space = TextMobject("Space: {0}".format(self.space + 1))
        new_counter_space.move_to(self.space_counter_pos)
        return new_counter_space

class MediumIterationCount(Scene):
    def construct(self):
        intro = TextMobject("Less than a second later ...")
        self.play(Write(intro))
        self.wait()
        self.play(FadeOut(intro))
        start_pos = UP * 0.5 + 6 * LEFT
        top_left_array = [Rectangle(height=0.5, width=0.3) for _ in range(12)]
        dots = [Dot() for _ in range(3)]
        [dot.scale(0.5) for dot in dots]
        top_right_array = [Rectangle(height=0.5, width=0.3) for _ in range(24)]
        top_array = top_left_array + dots + top_right_array
        top_array[0].set_color(BLUE)
        top_array[0].move_to(start_pos)

        for i in range(1, len(top_left_array)):
            top_array[i].set_color(BLUE)
            top_array[i].next_to(top_array[i - 1], RIGHT, buff=0)

        for i in range(len(top_left_array), len(top_left_array) + 4):
            top_array[i].set_color(BLUE)
            top_array[i].next_to(top_array[i - 1], RIGHT, buff=0.1)
        for i in range(len(top_left_array) + 4, len(top_array)):
            top_array[i].set_color(BLUE)
            top_array[i].next_to(top_array[i - 1], RIGHT, buff=0)

        bottom_left_array = [Rectangle(height=0.5, width=0.3) for _ in range(12)]
        bottom_dots = [Dot() for _ in range(3)]
        [bottom_dot.scale(0.5) for bottom_dot in bottom_dots]
        bottom_right_array = [Rectangle(height=0.5, width=0.3) for _ in range(25)]
        bottom_array = bottom_left_array + bottom_dots + bottom_right_array
        bottom_array[0].set_color(BLUE)
        bottom_array[0].move_to(start_pos + DOWN * top_left_array[0].get_height())

        for i in range(1, len(bottom_left_array)):
            bottom_array[i].set_color(BLUE)
            bottom_array[i].next_to(bottom_array[i - 1], RIGHT, buff=0)

        for i in range(len(bottom_left_array), len(bottom_left_array) + 4):
            bottom_array[i].set_color(BLUE)
            bottom_array[i].next_to(bottom_array[i - 1], RIGHT, buff=0.1)
        for i in range(len(bottom_left_array) + 4, len(bottom_array)):
            bottom_array[i].set_color(BLUE)
            bottom_array[i].next_to(bottom_array[i - 1], RIGHT, buff=0)

        top_left_array = top_array[:len(top_left_array)]
        top_right_array = top_array[len(top_left_array) + 3:]
        bottom_left_array = bottom_array[:len(bottom_left_array)]
        bottom_right_array = bottom_array[len(bottom_left_array) + 3:]

        top_left_array_text = []
        start_char = 'A'
        for i in range(len(top_left_array)):
            text = TextMobject(start_char)
            text.set_color(YELLOW)
            text.scale(0.5)
            text.move_to(top_left_array[i].get_center())
            top_left_array_text.append(text)
            start_char = chr(ord(start_char) + 1)

        top_right_array_text = []
        start_char ='B'
        for i in range(len(top_right_array)):
            text = TextMobject(start_char)
            text.set_color(YELLOW)
            text.scale(0.5)
            text.move_to(top_right_array[i].get_center())
            top_right_array_text.append(text)
            start_char = chr(ord(start_char) + 1)

        new_text = TextMobject(start_char)
        new_text.scale(0.5)
        new_text.next_to(top_right_array[-1], RIGHT * top_right_array[-1].get_width())
        top_right_array_text.append(new_text)

        animations_top_text = [ShowCreation(text) for text in top_left_array_text + top_right_array_text]

        bottom_left_array_text = []
        start_char = 'A'
        for i in range(len(bottom_left_array)):
            text = TextMobject(start_char)
            text.set_color(YELLOW)
            text.scale(0.5)
            text.move_to(bottom_left_array[i].get_center())
            bottom_left_array_text.append(text)
            start_char = chr(ord(start_char) + 1)

        bottom_right_array_text = []
        start_char ='B'
        insert_index = 4
        for i in range(insert_index):
            text = TextMobject(start_char)
            text.set_color(YELLOW)
            text.scale(0.5)
            text.move_to(bottom_right_array[i].get_center())
            bottom_right_array_text.append(text)
            start_char = chr(ord(start_char) + 1)

        self.count = 500494 - 21
        self.prev_counter_text = TextMobject("Insertions: {0}".format(self.count))
        self.counter_position = 3.5 * UP + 4 * RIGHT
        self.prev_counter_text.move_to(self.counter_position)
        self.next_counter_text = TextMobject("Insertions: {0}".format(self.count + 1))
        self.next_counter_text.move_to(self.counter_position)
        self.space_counter_text = TextMobject("Space: {0}".format(1000))
        self.space_counter_text.move_to(3 * UP + 4 * RIGHT)


        animations_bottom_text = [ShowCreation(text) for text in bottom_left_array_text + bottom_right_array_text]

        animations_rect = [GrowFromCenter(obj) for obj in top_array + bottom_array]
        animations = animations_rect + animations_top_text + animations_bottom_text 
        self.play(*animations)
        self.wait()
        
        curved_arrow_top_start = top_array[0].get_center() + 0.3 * LEFT
        curved_arrow_top_end = top_array[0].get_center() + UP
        curved_arrow_top = CurvedArrow(curved_arrow_top_start, curved_arrow_top_end, angle= -TAU / 2)
        curved_arrow_top.set_color(RED)
        length_top = TextMobject("Length: 999")
        length_top.move_to(curved_arrow_top_end + 1.5 * RIGHT)

        curved_arrow_bottom_start = bottom_array[0].get_center() + 0.3 * LEFT
        curved_arrow_bottom_end = bottom_array[0].get_center() + DOWN
        curved_arrow_bottom = CurvedArrow(curved_arrow_bottom_start, curved_arrow_bottom_end, angle= TAU / 2)
        curved_arrow_bottom.set_color(RED)
        length_bottom = TextMobject("Length: 1000")
        length_bottom.move_to(curved_arrow_bottom_end + 1.5 * RIGHT)


        self.play(FadeIn(curved_arrow_top), FadeIn(length_top), FadeIn(curved_arrow_bottom), FadeIn(length_bottom))


        self.play(*[ShowCreation(self.prev_counter_text), ShowCreation(self.space_counter_text)])
        self.wait()
        run_time = 0.2

        for i, text in enumerate(top_right_array_text[insert_index:]):
            text_copy = text.copy()
            text_copy.move_to(bottom_right_array[i+insert_index].get_center())
            text_copy.set_color(YELLOW)
            counter_transform = ReplacementTransform(self.prev_counter_text, self.next_counter_text)
            self.next_counter_text, self.prev_counter_text = self.update_counter(), self.next_counter_text
            if text == new_text:
                self.play(ReplacementTransform(text, text_copy), counter_transform, run_time=run_time)
            else:
                self.play(TransformFromCopy(text, text_copy), counter_transform, run_time=run_time)

        self.wait(2)

    def update_counter(self):
        self.count += 1
        new_counter_text = TextMobject("Insertions: {0}".format(self.count + 1))
        new_counter_text.move_to(self.counter_position)
        return new_counter_text



class LargeIterationCount(Scene):
    def construct(self):
        intro = TextMobject("Many hours later ...")
        self.play(Write(intro))
        self.wait()
        self.play(FadeOut(intro))
        start_pos = UP * 0.5 + 6 * LEFT
        top_left_array = [Rectangle(height=0.5, width=0.25) for _ in range(16)]
        dots = [Dot() for _ in range(3)]
        [dot.scale(0.5) for dot in dots]
        top_right_array = [Rectangle(height=0.5, width=0.25) for _ in range(30)]
        top_array = top_left_array + dots + top_right_array
        top_array[0].set_color(BLUE)
        top_array[0].move_to(start_pos)

        for i in range(1, len(top_left_array)):
            top_array[i].set_color(BLUE)
            top_array[i].next_to(top_array[i - 1], RIGHT, buff=0)

        for i in range(len(top_left_array), len(top_left_array) + 4):
            top_array[i].set_color(BLUE)
            top_array[i].next_to(top_array[i - 1], RIGHT, buff=0.1)
        for i in range(len(top_left_array) + 4, len(top_array)):
            top_array[i].set_color(BLUE)
            top_array[i].next_to(top_array[i - 1], RIGHT, buff=0)

        bottom_left_array = [Rectangle(height=0.5, width=0.25) for _ in range(16)]
        bottom_dots = [Dot() for _ in range(3)]
        [bottom_dot.scale(0.5) for bottom_dot in bottom_dots]
        bottom_right_array = [Rectangle(height=0.5, width=0.25) for _ in range(31)]
        bottom_array = bottom_left_array + bottom_dots + bottom_right_array
        bottom_array[0].set_color(BLUE)
        bottom_array[0].move_to(start_pos + DOWN * top_left_array[0].get_height())

        for i in range(1, len(bottom_left_array)):
            bottom_array[i].set_color(BLUE)
            bottom_array[i].next_to(bottom_array[i - 1], RIGHT, buff=0)

        for i in range(len(bottom_left_array), len(bottom_left_array) + 4):
            bottom_array[i].set_color(BLUE)
            bottom_array[i].next_to(bottom_array[i - 1], RIGHT, buff=0.1)
        for i in range(len(bottom_left_array) + 4, len(bottom_array)):
            bottom_array[i].set_color(BLUE)
            bottom_array[i].next_to(bottom_array[i - 1], RIGHT, buff=0)

        top_left_array = top_array[:len(top_left_array)]
        top_right_array = top_array[len(top_left_array) + 3:]
        bottom_left_array = bottom_array[:len(bottom_left_array)]
        bottom_right_array = bottom_array[len(bottom_left_array) + 3:]

        top_left_array_text = []
        start_char = 'A'
        scale = 0.4
        for i in range(len(top_left_array)):
            if i == 25:
                start_char = 'A'
            text = TextMobject(start_char)
            text.set_color(YELLOW)
            text.scale(scale)
            text.move_to(top_left_array[i].get_center())
            top_left_array_text.append(text)
            start_char = chr(ord(start_char) + 1)

        top_right_array_text = []
        start_char ='B'
        for i in range(len(top_right_array)):
            if i == 25:
                start_char = 'A'
            text = TextMobject(start_char)
            text.set_color(YELLOW)
            text.scale(scale)
            text.move_to(top_right_array[i].get_center())
            top_right_array_text.append(text)
            start_char = chr(ord(start_char) + 1)

        new_text = TextMobject(start_char)
        new_text.scale(scale)
        new_text.next_to(top_right_array[-1], RIGHT * top_right_array[-1].get_width())
        top_right_array_text.append(new_text)

        animations_top_text = [ShowCreation(text) for text in top_left_array_text + top_right_array_text]

        bottom_left_array_text = []
        start_char = 'A'
        for i in range(len(bottom_left_array)):
            if i == 25:
                start_char = 'A'
            text = TextMobject(start_char)
            text.set_color(YELLOW)
            text.scale(scale)
            text.move_to(bottom_left_array[i].get_center())
            bottom_left_array_text.append(text)
            start_char = chr(ord(start_char) + 1)

        bottom_right_array_text = []
        start_char ='B'
        insert_index = 12
        for i in range(insert_index):
            if i == 25:
                start_char = 'A'
            text = TextMobject(start_char)
            text.set_color(YELLOW)
            text.scale(scale)
            text.move_to(bottom_right_array[i].get_center())
            bottom_right_array_text.append(text)
            start_char = chr(ord(start_char) + 1)

        self.count = 500000499994 - (len(bottom_right_array) - insert_index)
        self.prev_counter_text = TextMobject("Insertions: {0}".format(self.count))
        self.counter_position = 3.5 * UP + 3 * RIGHT
        self.prev_counter_text.move_to(self.counter_position)
        self.next_counter_text = TextMobject("Insertions: {0}".format(self.count + 1))
        self.next_counter_text.move_to(self.counter_position)
        self.space_counter_text = TextMobject("Space: {0}".format(1000000))
        self.space_counter_text.move_to(3 * UP + 3 * RIGHT)


        animations_bottom_text = [ShowCreation(text) for text in bottom_left_array_text + bottom_right_array_text]

        animations_rect = [GrowFromCenter(obj) for obj in top_array + bottom_array]
        animations = animations_rect + animations_top_text + animations_bottom_text 
        self.play(*animations)
        self.wait()
        
        curved_arrow_top_start = top_array[0].get_center() + 0.3 * LEFT
        curved_arrow_top_end = top_array[0].get_center() + UP
        curved_arrow_top = CurvedArrow(curved_arrow_top_start, curved_arrow_top_end, angle= -TAU / 2)
        curved_arrow_top.set_color(RED)
        length_top = TextMobject("Length: 999999")
        length_top.move_to(curved_arrow_top_end + 2 * RIGHT)

        curved_arrow_bottom_start = bottom_array[0].get_center() + 0.3 * LEFT
        curved_arrow_bottom_end = bottom_array[0].get_center() + DOWN
        curved_arrow_bottom = CurvedArrow(curved_arrow_bottom_start, curved_arrow_bottom_end, angle= TAU / 2)
        curved_arrow_bottom.set_color(RED)
        length_bottom = TextMobject("Length: 1000000")
        length_bottom.move_to(curved_arrow_bottom_end + 2 * RIGHT)


        self.play(FadeIn(curved_arrow_top), FadeIn(length_top), FadeIn(curved_arrow_bottom), FadeIn(length_bottom))


        self.play(*[ShowCreation(self.prev_counter_text), ShowCreation(self.space_counter_text)])
        self.wait()
        run_time = 0.25

        for i, text in enumerate(top_right_array_text[insert_index:]):
            # print(len(bottom_right_array), i + insert_index)
            text_copy = text.copy()
            text_copy.move_to(bottom_right_array[i+insert_index].get_center())
            text_copy.set_color(YELLOW)
            counter_transform = ReplacementTransform(self.prev_counter_text, self.next_counter_text)
            self.next_counter_text, self.prev_counter_text = self.update_counter(), self.next_counter_text
            if text == new_text:
                self.play(ReplacementTransform(text, text_copy), counter_transform, run_time=run_time)
            else:
                self.play(TransformFromCopy(text, text_copy), counter_transform, run_time=run_time)

        self.wait(3)

    def update_counter(self):
        self.count += 1
        new_counter_text = TextMobject("Insertions: {0}".format(self.count + 1))
        new_counter_text.move_to(self.counter_position)
        return new_counter_text



class ResizePlus8(Scene):
    def construct(self):
        self.wait(3)
        text1 = TextMobject("As we hypothesized, this first scheme seems unappealing")
        self.play(Write(text1), run_time=4)
        self.wait(2)
        text = TextMobject("How can we go about improving our current scheme?")
        self.play(Transform(text1, text), run_time=3)
        self.wait(2)
        text2 = TextMobject("Can we reduce insertions?")
        text2.shift(DOWN)
        self.play(Write(text2), run_time=2)
        self.wait(3)
        self.play(FadeOut(text1), FadeOut(text2), run_time=2)
        self.wait(4)

        initial_array = create_array(4, YELLOW, LEFT * 6 + UP * 2, 0.8, 1.1)
        text = generate_chars_in_array(initial_array, 'A', GREEN)
        array_anims = [FadeIn(obj) for obj in initial_array + text]
        self.play(*array_anims)
        self.wait(2)

        new_array = create_array(12, BLUE, LEFT * 6 + UP * 0.5, 0.8, 1.1)
        new_array_anims = [GrowFromCenter(obj) for obj in new_array]
        new_text = TextMobject("E")
        offset = 1.5
        new_text.move_to(new_array[len(initial_array)].get_center() + offset * UP)
        text.append(new_text)

        self.play(ShowCreation(new_text))
        self.wait(2)
        self.play(*new_array_anims[:5])
        self.wait(6)
        self.play(*new_array_anims[5:])
        self.play(*[ApplyWave(rect) for rect in initial_array])
        self.transform_text(text, new_array, len(initial_array), 
            len(new_array) - len(initial_array), 'E', offset, run_time=0.4)
        self.wait(2)

    def transform_text(self, text, new_array, num_copy, num_insert, last_char, offset, run_time=1):
        new_text = []
        for i in range(num_copy):
            text_copy = text[i].copy()
            text_copy.move_to(new_array[i].get_center())
            new_text.append(text_copy)
            self.play(TransformFromCopy(text[i], text_copy, run_time=run_time))
        remaining_text = text[num_copy:]
        remaining_arr = new_array[num_copy:]
        for i in range(num_insert):
            if i >= len(remaining_text):
                last_char = chr(ord(last_char) + 1)
                text_to_create = TextMobject(last_char)
                text_to_create.move_to(remaining_arr[i].get_center() + offset * UP)
                self.play(ShowCreation(text_to_create, run_time=run_time))
                text_copy = text_to_create.copy()
                text_copy.move_to(remaining_arr[i].get_center())
                text_copy.set_color(GREEN)
                new_text.append(text_copy)
                self.play(ReplacementTransform(text_to_create, text_copy, run_time=run_time))
            else:
                text_copy = remaining_text[i].copy()
                text_copy.move_to(remaining_arr[i].get_center())
                text_copy.set_color(GREEN)
                self.play(ReplacementTransform(remaining_text[i], text_copy, run_time=run_time))


class CountInsertionsPlus8(Scene):
    def construct(self):
        self.count = 0
        self.prev_counter_text = TextMobject("Insertions: {0}".format(self.count))
        self.counter_position = 3.5 * UP + 5 * RIGHT
        self.prev_counter_text.move_to(self.counter_position)
        self.play(ShowCreation(self.prev_counter_text))
        self.next_counter_text = TextMobject("Insertions: {0}".format(self.count + 1))
        self.next_counter_text.move_to(self.counter_position)
        self.space = 0
        self.prev_counter_space = TextMobject("Space: {0}".format(self.space))
        self.space_counter_pos = 3 * UP + 5 * RIGHT
        self.prev_counter_space.move_to(self.space_counter_pos)
        self.play(ShowCreation(self.prev_counter_space))
        self.space = 3
        self.next_counter_space = TextMobject("Space: {0}".format(self.space + 1))
        self.next_counter_space.move_to(self.space_counter_pos)
        run_time = 0.12
        start_pos = 1.5 * UP + 6 * LEFT
        num_rectangles = 4
        num_chars = 4
        color = BLUE
        new_pos = start_pos
        array = self.generate_array(num_rectangles, start_pos, BLUE, FadeIn, run_time)
        previous_chars = self.create_and_move_characters(num_chars, array, ReplacementTransform, run_time)
        colors = [BLUE]
        prev_array = array
        while num_rectangles < 18:
            new_pos = new_pos + array[0].get_height() * DOWN * 1.5
            num_rectangles += 8
            num_chars += 1
            new_array = self.generate_array(num_rectangles, new_pos, 
                colors[num_rectangles % len(colors)], GrowFromCenter, run_time)

            previous_chars = self.create_and_move_second_set_chars(num_chars, new_array, prev_array, previous_chars, run_time)
            prev_array = new_array
        text_copy = previous_chars[-1].copy()
        text_copy.move_to(new_array[-1].get_center())
        text_copy.set_color(YELLOW)
        counter_transform = ReplacementTransform(self.prev_counter_text, self.next_counter_text)
        self.play(ReplacementTransform(previous_chars[-1], text_copy), run_time=0.01)
        self.wait(5)

    def generate_array(self, length, start_pos, color, anim, run_time):
        starting_array = []
        for i in range(length):
            rect = Rectangle(height=0.5, width=0.6)
            rect.set_color(color)
            starting_array.append(rect)
        starting_array[0].move_to(start_pos)
        for i in range(1, len(starting_array)):
            starting_array[i].next_to(starting_array[i - 1], RIGHT, buff=0)
        starting_array_animations = [anim(rect) for rect in starting_array]
        counter_transform = ReplacementTransform(self.prev_counter_space, self.next_counter_space)
        self.next_counter_space, self.prev_counter_space = self.update_space(), self.next_counter_space
        self.play(*starting_array_animations, counter_transform, run_time=run_time)
        return starting_array

    def create_and_move_characters(self, num_chars, target_array, transf, run_time):
        text_objects = []
        start_char = 'A'
        for i in range(num_chars):
            char = chr(ord(start_char) + i)
            text = TextMobject(char)
            text.scale(0.65)
            text.next_to(target_array[i], UP * 0.5)
            self.play(FadeIn(text), run_time=run_time)
            copy_text = text.copy()
            copy_text.move_to(target_array[i].get_center())
            copy_text.set_color(YELLOW)
            text_objects.append(copy_text)
            counter_transform = ReplacementTransform(self.prev_counter_text, self.next_counter_text)
            self.next_counter_text, self.prev_counter_text = self.update_counter(), self.next_counter_text
            self.play(transf(text, copy_text), counter_transform, run_time=run_time)
        return text_objects

    def create_and_move_second_set_chars(self, num_chars, target_array, prev_array, previous_text, run_time):
        text_objects = []
        prev_text = []
        all_new_text = []
        new_char = chr(ord('A') + num_chars - 1) 
        for i in range(len(prev_array), len(target_array)):
            new_text = TextMobject(new_char)
            new_text.scale(0.65)
            new_text.next_to(target_array[i], UP * 0.5)
            new_char = chr(ord(new_char) + 1)
            all_new_text.append(new_text)
        # self.play(FadeIn(new_text), run_time=run_time)
        print(len(previous_text), len(all_new_text))
        for i, text in enumerate(previous_text + all_new_text):
            if text not in all_new_text:
                prev_text.append(text)
            text_copy = text.copy()
            text_copy.move_to(target_array[i].get_center())
            text_copy.set_color(YELLOW)
            text_objects.append(text_copy)
            counter_transform = ReplacementTransform(self.prev_counter_text, self.next_counter_text)
            self.next_counter_text, self.prev_counter_text = self.update_counter(), self.next_counter_text
            if text in all_new_text:
                self.play(ReplacementTransform(text, text_copy), counter_transform, run_time=run_time)
            else:
                self.play(TransformFromCopy(text, text_copy), counter_transform, run_time=run_time)

        removal_animations = [FadeOut(obj) for obj in prev_text + prev_array]
        self.play(*removal_animations, run_time=run_time)
        return text_objects

    def update_counter(self):
        self.count += 1
        new_counter_text = TextMobject("Insertions: {0}".format(self.count + 1))
        new_counter_text.move_to(self.counter_position)
        return new_counter_text

    def update_space(self):
        self.space += 8
        new_counter_space = TextMobject("Space: {0}".format(self.space + 1))
        new_counter_space.move_to(self.space_counter_pos)
        return new_counter_space

class MediumIterationCountPlus8(Scene):
    def construct(self):
        start_pos = UP + 6 * LEFT
        top_left_array = [Rectangle(height=0.5, width=0.3) for _ in range(8)]
        dots = [Dot() for _ in range(3)]
        [dot.scale(0.5) for dot in dots]
        top_right_array = [Rectangle(height=0.5, width=0.3) for _ in range(20)]
        top_array = top_left_array + dots + top_right_array
        top_array[0].set_color(BLUE)
        top_array[0].move_to(start_pos)

        for i in range(1, len(top_left_array)):
            top_array[i].set_color(BLUE)
            top_array[i].next_to(top_array[i - 1], RIGHT, buff=0)

        for i in range(len(top_left_array), len(top_left_array) + 4):
            top_array[i].set_color(BLUE)
            top_array[i].next_to(top_array[i - 1], RIGHT, buff=0.1)
        for i in range(len(top_left_array) + 4, len(top_array)):
            top_array[i].set_color(BLUE)
            top_array[i].next_to(top_array[i - 1], RIGHT, buff=0)

        bottom_left_array = [Rectangle(height=0.5, width=0.3) for _ in range(8)]
        bottom_dots = [Dot() for _ in range(3)]
        [bottom_dot.scale(0.5) for bottom_dot in bottom_dots]
        bottom_right_array = [Rectangle(height=0.5, width=0.3) for _ in range(28)]
        bottom_array = bottom_left_array + bottom_dots + bottom_right_array
        bottom_array[0].set_color(BLUE)
        bottom_array[0].move_to(start_pos + DOWN * top_left_array[0].get_height())

        for i in range(1, len(bottom_left_array)):
            bottom_array[i].set_color(BLUE)
            bottom_array[i].next_to(bottom_array[i - 1], RIGHT, buff=0)

        for i in range(len(bottom_left_array), len(bottom_left_array) + 4):
            bottom_array[i].set_color(BLUE)
            bottom_array[i].next_to(bottom_array[i - 1], RIGHT, buff=0.1)
        for i in range(len(bottom_left_array) + 4, len(bottom_array)):
            bottom_array[i].set_color(BLUE)
            bottom_array[i].next_to(bottom_array[i - 1], RIGHT, buff=0)

        top_left_array = top_array[:len(top_left_array)]
        top_right_array = top_array[len(top_left_array) + 3:]
        bottom_left_array = bottom_array[:len(bottom_left_array)]
        bottom_right_array = bottom_array[len(bottom_left_array) + 3:]

        top_left_array_text = []
        start_char = 'A'
        for i in range(len(top_left_array)):
            text = TextMobject(start_char)
            text.set_color(YELLOW)
            text.scale(0.5)
            text.move_to(top_left_array[i].get_center())
            top_left_array_text.append(text)
            start_char = chr(ord(start_char) + 1)

        top_right_array_text = []
        start_char ='B'
        for i in range(len(top_right_array)):
            text = TextMobject(start_char)
            text.set_color(YELLOW)
            text.scale(0.5)
            text.move_to(top_right_array[i].get_center())
            top_right_array_text.append(text)
            start_char = chr(ord(start_char) + 1)

        all_new_text = []
        num_new_text = 4
        for i in range(num_new_text):
            new_text = TextMobject(start_char)
            new_text.scale(0.5)
            new_text.move_to(top_right_array[-1].get_center() + 
                (i+1) * (RIGHT * top_right_array[-1].get_width()))
            top_right_array_text.append(new_text)
            all_new_text.append(new_text)
            start_char = chr(ord(start_char) + 1)

        animations_top_text = [ShowCreation(text) for text in top_left_array_text + top_right_array_text[:len(top_right_array) + 1]]

        bottom_left_array_text = []
        start_char = 'A'
        for i in range(len(bottom_left_array)):
            text = TextMobject(start_char)
            text.set_color(YELLOW)
            text.scale(0.5)
            text.move_to(bottom_left_array[i].get_center())
            bottom_left_array_text.append(text)
            start_char = chr(ord(start_char) + 1)

        bottom_right_array_text = []
        start_char ='B'
        insert_index = 4
        for i in range(insert_index):
            text = TextMobject(start_char)
            text.set_color(YELLOW)
            text.scale(0.5)
            text.move_to(bottom_right_array[i].get_center())
            bottom_right_array_text.append(text)
            start_char = chr(ord(start_char) + 1)

        self.count = 63500 - 20
        self.prev_counter_text = TextMobject("Insertions: {0}".format(self.count))
        self.counter_position = 3.5 * UP + 4 * RIGHT
        self.prev_counter_text.move_to(self.counter_position)
        self.next_counter_text = TextMobject("Insertions: {0}".format(self.count + 1))
        self.next_counter_text.move_to(self.counter_position)
        self.space_counter_text = TextMobject("Space: {0}".format(1004))
        self.space_counter_text.move_to(3 * UP + 4 * RIGHT)


        animations_bottom_text = [ShowCreation(text) for text in bottom_left_array_text + bottom_right_array_text]

        animations_rect = [GrowFromCenter(obj) for obj in top_array + bottom_array]
        animations = animations_rect + animations_top_text + animations_bottom_text 
        self.play(*animations)
        self.wait()
        
        curved_arrow_top_start = top_array[0].get_center() + 0.3 * LEFT
        curved_arrow_top_end = top_array[0].get_center() + UP
        curved_arrow_top = CurvedArrow(curved_arrow_top_start, curved_arrow_top_end, angle= -TAU / 2)
        curved_arrow_top.set_color(RED)
        length_top = TextMobject("Length: 996")
        length_top.move_to(curved_arrow_top_end + 1.5 * RIGHT)

        curved_arrow_bottom_start = bottom_array[0].get_center() + 0.3 * LEFT
        curved_arrow_bottom_end = bottom_array[0].get_center() + DOWN
        curved_arrow_bottom = CurvedArrow(curved_arrow_bottom_start, curved_arrow_bottom_end, angle= TAU / 2)
        curved_arrow_bottom.set_color(RED)
        length_bottom = TextMobject("Length: 1004")
        length_bottom.move_to(curved_arrow_bottom_end + 1.5 * RIGHT)


        self.play(FadeIn(curved_arrow_top), FadeIn(length_top), FadeIn(curved_arrow_bottom), FadeIn(length_bottom))


        self.play(*[ShowCreation(self.prev_counter_text), ShowCreation(self.space_counter_text)])
        self.wait()
        run_time = 0.4

        for i, text in enumerate(top_right_array_text[insert_index:]):
            if text in all_new_text[1:]:
                self.play(ShowCreation(text), run_time=run_time)
            text_copy = text.copy()
            text_copy.move_to(bottom_right_array[i+insert_index].get_center())
            text_copy.set_color(YELLOW)
            counter_transform = ReplacementTransform(self.prev_counter_text, self.next_counter_text)
            self.next_counter_text, self.prev_counter_text = self.update_counter(), self.next_counter_text
            if text in all_new_text:
                self.play(ReplacementTransform(text, text_copy), counter_transform, run_time=run_time)
            else:
                self.play(TransformFromCopy(text, text_copy), counter_transform, run_time=run_time)

        self.wait(3)

    def update_counter(self):
        self.count += 1
        new_counter_text = TextMobject("Insertions: {0}".format(self.count + 1))
        new_counter_text.move_to(self.counter_position)
        return new_counter_text


class LargeIterationCountPlus8(Scene):
    def construct(self):
        start_pos = UP + 6 * LEFT
        top_left_array = [Rectangle(height=0.5, width=0.3) for _ in range(10)]
        dots = [Dot() for _ in range(3)]
        [dot.scale(0.5) for dot in dots]
        top_right_array = [Rectangle(height=0.5, width=0.3) for _ in range(23)]
        top_array = top_left_array + dots + top_right_array
        top_array[0].set_color(BLUE)
        top_array[0].move_to(start_pos)

        for i in range(1, len(top_left_array)):
            top_array[i].set_color(BLUE)
            top_array[i].next_to(top_array[i - 1], RIGHT, buff=0)

        for i in range(len(top_left_array), len(top_left_array) + 4):
            top_array[i].set_color(BLUE)
            top_array[i].next_to(top_array[i - 1], RIGHT, buff=0.1)
        for i in range(len(top_left_array) + 4, len(top_array)):
            top_array[i].set_color(BLUE)
            top_array[i].next_to(top_array[i - 1], RIGHT, buff=0)

        bottom_left_array = [Rectangle(height=0.5, width=0.3) for _ in range(10)]
        bottom_dots = [Dot() for _ in range(3)]
        [bottom_dot.scale(0.5) for bottom_dot in bottom_dots]
        bottom_right_array = [Rectangle(height=0.5, width=0.3) for _ in range(31)]
        bottom_array = bottom_left_array + bottom_dots + bottom_right_array
        bottom_array[0].set_color(BLUE)
        bottom_array[0].move_to(start_pos + DOWN * top_left_array[0].get_height())

        for i in range(1, len(bottom_left_array)):
            bottom_array[i].set_color(BLUE)
            bottom_array[i].next_to(bottom_array[i - 1], RIGHT, buff=0)

        for i in range(len(bottom_left_array), len(bottom_left_array) + 4):
            bottom_array[i].set_color(BLUE)
            bottom_array[i].next_to(bottom_array[i - 1], RIGHT, buff=0.1)
        for i in range(len(bottom_left_array) + 4, len(bottom_array)):
            bottom_array[i].set_color(BLUE)
            bottom_array[i].next_to(bottom_array[i - 1], RIGHT, buff=0)

        top_left_array = top_array[:len(top_left_array)]
        top_right_array = top_array[len(top_left_array) + 3:]
        bottom_left_array = bottom_array[:len(bottom_left_array)]
        bottom_right_array = bottom_array[len(bottom_left_array) + 3:]

        top_left_array_text = []
        start_char = 'A'
        scale = 0.4
        for i in range(len(top_left_array)):
            if i == 25:
                start_char = 'A'
            text = TextMobject(start_char)
            text.set_color(YELLOW)
            text.scale(scale)
            text.move_to(top_left_array[i].get_center())
            top_left_array_text.append(text)
            start_char = chr(ord(start_char) + 1)

        top_right_array_text = []
        start_char ='B'
        for i in range(len(top_right_array)):
            if i == 25:
                start_char = 'A'
            text = TextMobject(start_char)
            text.set_color(YELLOW)
            text.scale(scale)
            text.move_to(top_right_array[i].get_center())
            top_right_array_text.append(text)
            start_char = chr(ord(start_char) + 1)

        all_new_text = []
        num_new_text = 4
        for i in range(num_new_text):
            new_text = TextMobject(start_char)
            new_text.scale(scale)
            new_text.move_to(top_right_array[-1].get_center() + 
                (i+1) * (RIGHT * top_right_array[-1].get_width()))
            top_right_array_text.append(new_text)
            all_new_text.append(new_text)
            if start_char == 'Z':
                start_char = 'A'
            else:
                start_char = chr(ord(start_char) + 1)

        animations_top_text = [ShowCreation(text) for text in top_left_array_text + top_right_array_text[:len(top_right_array) + 1]]

        bottom_left_array_text = []
        start_char = 'A'
        for i in range(len(bottom_left_array)):
            if i == 25:
                start_char = 'A'
            text = TextMobject(start_char)
            text.set_color(YELLOW)
            text.scale(scale)
            text.move_to(bottom_left_array[i].get_center())
            bottom_left_array_text.append(text)
            start_char = chr(ord(start_char) + 1)

        bottom_right_array_text = []
        start_char ='B'
        insert_index = 12
        for i in range(insert_index):
            if i == 25:
                start_char = 'A'
            text = TextMobject(start_char)
            text.set_color(YELLOW)
            text.scale(scale)
            text.move_to(bottom_right_array[i].get_center())
            bottom_right_array_text.append(text)
            start_char = chr(ord(start_char) + 1)

        self.count = 62501000004 - (len(bottom_right_array) - insert_index)
        self.prev_counter_text = TextMobject("Insertions: {0}".format(self.count))
        self.counter_position = 3.5 * UP + 3 * RIGHT
        self.prev_counter_text.move_to(self.counter_position)
        self.next_counter_text = TextMobject("Insertions: {0}".format(self.count + 1))
        self.next_counter_text.move_to(self.counter_position)
        self.space_counter_text = TextMobject("Space: {0}".format(1000004))
        self.space_counter_text.move_to(3 * UP + 3 * RIGHT)


        animations_bottom_text = [ShowCreation(text) for text in bottom_left_array_text + bottom_right_array_text]

        animations_rect = [GrowFromCenter(obj) for obj in top_array + bottom_array]
        animations = animations_rect + animations_top_text + animations_bottom_text 
        self.play(*animations)
        self.wait()
        
        curved_arrow_top_start = top_array[0].get_center() + 0.3 * LEFT
        curved_arrow_top_end = top_array[0].get_center() + UP
        curved_arrow_top = CurvedArrow(curved_arrow_top_start, curved_arrow_top_end, angle= -TAU / 2)
        curved_arrow_top.set_color(RED)
        length_top = TextMobject("Length: 999996")
        length_top.move_to(curved_arrow_top_end + 2 * RIGHT)

        curved_arrow_bottom_start = bottom_array[0].get_center() + 0.3 * LEFT
        curved_arrow_bottom_end = bottom_array[0].get_center() + DOWN
        curved_arrow_bottom = CurvedArrow(curved_arrow_bottom_start, curved_arrow_bottom_end, angle= TAU / 2)
        curved_arrow_bottom.set_color(RED)
        length_bottom = TextMobject("Length: 1000004")
        length_bottom.move_to(curved_arrow_bottom_end + 2 * RIGHT)


        self.play(FadeIn(curved_arrow_top), FadeIn(length_top), FadeIn(curved_arrow_bottom), FadeIn(length_bottom))


        self.play(*[ShowCreation(self.prev_counter_text), ShowCreation(self.space_counter_text)])
        self.wait()
        run_time = 0.2

        for i, text in enumerate(top_right_array_text[insert_index:]):
            # print(len(bottom_right_array), i + insert_index)
            if text in all_new_text[1:]:
                self.play(ShowCreation(text), run_time=run_time)
            text_copy = text.copy()
            text_copy.move_to(bottom_right_array[i+insert_index].get_center())
            text_copy.set_color(YELLOW)
            counter_transform = ReplacementTransform(self.prev_counter_text, self.next_counter_text)
            self.next_counter_text, self.prev_counter_text = self.update_counter(), self.next_counter_text
            if text in all_new_text:
                self.play(ReplacementTransform(text, text_copy), counter_transform, run_time=run_time)
            else:
                self.play(TransformFromCopy(text, text_copy), counter_transform, run_time=run_time)

        self.wait()

    def update_counter(self):
        self.count += 1
        new_counter_text = TextMobject("Insertions: {0}".format(self.count + 1))
        new_counter_text.move_to(self.counter_position)
        return new_counter_text


        
class ResizeDouble(Scene):
    def construct(self):
        scale = 1.2
        initial_array = create_array(4, YELLOW, LEFT * 6.1 + UP * 2.2, scale, 1.7)
        text = generate_chars_in_array(initial_array, 'A', GREEN, scale=scale)
        array_anims = [FadeIn(obj) for obj in initial_array + text]
        self.play(*array_anims)


        new_array = create_array(8, BLUE, LEFT * 6.1 + UP * 0.5, scale, 1.7)
        new_array_anims = [GrowFromCenter(obj) for obj in new_array]
        new_text = TextMobject("E")
        new_text.scale(scale)
        offset = 1.7
        new_text.move_to(new_array[len(initial_array)].get_center() + offset * UP)
        text.append(new_text)

        self.play(ShowCreation(new_text))
        self.wait(6)

        thought = TextMobject("What if we resized by a factor of 2?")
        thought.shift(DOWN * 3)
        self.play(Write(thought))

        self.play(*new_array_anims)
        self.play(*[ApplyWave(rect) for rect in initial_array])
        old_text, new_text = self.transform_text(text, new_array, len(initial_array), 
            len(new_array) - len(initial_array), 'E', offset, scale=scale, run_time=0.2)
        self.play(*[FadeOut(obj) for obj in initial_array + old_text])

        scale = scale / 2
        initial_array = create_array(8, BLUE, LEFT * 6.1 + UP * 1.35, 0.6, 0.85)
        text = generate_chars_in_array(initial_array, 'A', GREEN, scale=scale)
        transfs = [ReplacementTransform(new_obj, 
            initial_obj) for new_obj, initial_obj in zip(new_array + new_text, 
            initial_array + text)]
        self.play(*transfs, run_time=0.5)

        self.count = 12
        self.prev_counter_text = TextMobject("Insertions: {0}".format(self.count))
        self.counter_position = 3.5 * UP + 5 * RIGHT
        self.prev_counter_text.move_to(self.counter_position)
        self.play(ShowCreation(self.prev_counter_text))
        self.next_counter_text = TextMobject("Insertions: {0}".format(self.count + 1))
        self.next_counter_text.move_to(self.counter_position)
        self.space = 8
        self.prev_counter_space = TextMobject("Space: {0}".format(self.space))
        self.space_counter_pos = 3 * UP + 5 * RIGHT
        self.prev_counter_space.move_to(self.space_counter_pos)
        self.play(ShowCreation(self.prev_counter_space))
        self.next_counter_space = TextMobject("Space: {0}".format(self.space * 2))
        self.next_counter_space.move_to(self.space_counter_pos)


        new_array = create_array(16, RED, LEFT * 6.1 + UP * 0.5, 0.6, 0.85)
        new_array_anims = [GrowFromCenter(obj) for obj in new_array]
        new_char = chr(ord('A') + len(initial_array))
        new_text = TextMobject(new_char)
        new_text.scale(scale)
        offset = offset / 2
        new_text.move_to(new_array[len(initial_array)].get_center() + offset * UP)
        text.append(new_text)
        space_transform = ReplacementTransform(self.prev_counter_space, self.next_counter_space)
        self.next_counter_space, self.prev_counter_space = self.update_space(), self.next_counter_space

        self.play(ShowCreation(new_text))
        self.play(*new_array_anims, space_transform)
        # self.play(*[ApplyWave(rect) for rect in initial_array])
        old_text, new_text = self.transform_text(text, new_array, len(initial_array), 
            len(new_array) - len(initial_array), new_char, offset, scale=scale, count=True, run_time=0.1)
        self.play(*[FadeOut(obj) for obj in initial_array + old_text], run_time=0.5)

        scale = 0.4
        initial_array = create_array(16, RED, LEFT * 6.3 + UP * 1.1, 0.3, 0.425)
        text = generate_chars_in_array(initial_array, 'A', GREEN, scale=scale)
        transfs = [ReplacementTransform(new_obj, 
            initial_obj) for new_obj, initial_obj in zip(new_array + new_text, 
            initial_array + text)]
        self.play(*transfs, run_time=0.5)

        new_array = create_array(32, BLUE, LEFT * 6.3 + UP * 0.5, 0.3, 0.425)
        new_array_anims = [GrowFromCenter(obj) for obj in new_array]
        new_char = chr(ord('A') + len(initial_array))
        new_text = TextMobject(new_char)
        new_text.scale(scale)
        offset = 0.6
        new_text.move_to(new_array[len(initial_array)].get_center() + offset * UP)
        text.append(new_text)
        space_transform = ReplacementTransform(self.prev_counter_space, self.next_counter_space)
        self.next_counter_space, self.prev_counter_space = self.update_space(), self.next_counter_space

        self.play(ShowCreation(new_text))
        self.play(*new_array_anims, space_transform)
        old_text, new_text = self.transform_text(text, new_array, len(initial_array), 
            2, new_char, offset, scale=scale, count=True, run_time=0.16)
        self.play(*[FadeOut(obj) for obj in initial_array + old_text])
        self.play(*[FadeOut(obj) for obj in new_array + new_text])

        
        # self.play(*starting_array_animations, space_transform, run_time=run_time)
        # counter_transform = ReplacementTransform(self.prev_counter_text, self.next_counter_text)
        # self.next_counter_text, self.prev_counter_text = self.update_counter(), self.next_counter_text
        # self.play(transf(text, copy_text), counter_transform, run_time=run_time)

    def transform_text(self, text, new_array, num_copy, num_insert, last_char, offset, scale=1, count=False, run_time=1):
        old_text = []
        new_text = []
        for i in range(num_copy):
            old_text.append(text[i])
            text_copy = text[i].copy()
            text_copy.move_to(new_array[i].get_center())
            new_text.append(text_copy)
            if count:
                counter_transform = ReplacementTransform(self.prev_counter_text, self.next_counter_text)
                self.next_counter_text, self.prev_counter_text = self.update_counter(), self.next_counter_text
                self.play(TransformFromCopy(text[i], text_copy), counter_transform, run_time=run_time)
            else:
                self.play(TransformFromCopy(text[i], text_copy), run_time=run_time)
        remaining_text = text[num_copy:]
        remaining_arr = new_array[num_copy:]
        for i in range(num_insert):
            if i >= len(remaining_text):
                last_char = chr(ord(last_char) + 1)
                text_to_create = TextMobject(last_char)
                text_to_create.move_to(remaining_arr[i].get_center() + offset * UP)
                text_to_create.scale(scale)
                self.play(ShowCreation(text_to_create), run_time=run_time)
                text_copy = text_to_create.copy()
                text_copy.move_to(remaining_arr[i].get_center())
                text_copy.set_color(GREEN)
                new_text.append(text_copy)
                if count:
                    counter_transform = ReplacementTransform(self.prev_counter_text, self.next_counter_text)
                    self.next_counter_text, self.prev_counter_text = self.update_counter(), self.next_counter_text
                    self.play(ReplacementTransform(text_to_create, text_copy), counter_transform, run_time=run_time)
                else:
                    self.play(ReplacementTransform(text_to_create, text_copy), run_time=run_time)
            else:
                text_copy = remaining_text[i].copy()
                text_copy.move_to(remaining_arr[i].get_center())
                text_copy.set_color(GREEN)
                new_text.append(text_copy)
                if count:
                    counter_transform = ReplacementTransform(self.prev_counter_text, self.next_counter_text)
                    self.next_counter_text, self.prev_counter_text = self.update_counter(), self.next_counter_text
                    self.play(ReplacementTransform(remaining_text[i], text_copy), counter_transform, run_time=run_time)
                else:
                    self.play(ReplacementTransform(remaining_text[i], text_copy), run_time=run_time)

        return old_text, new_text

    def update_counter(self):
        self.count += 1
        new_counter_text = TextMobject("Insertions: {0}".format(self.count + 1))
        new_counter_text.move_to(self.counter_position)
        return new_counter_text

    def update_space(self):
        self.space *= 2
        new_counter_space = TextMobject("Space: {0}".format(self.space * 2))
        new_counter_space.move_to(self.space_counter_pos)
        return new_counter_space


class MediumIterationCountDouble(Scene):
    def construct(self):
        start_pos = 0.5 * UP + 6 * LEFT
        top_left_array = [Rectangle(height=0.5, width=0.3) for _ in range(8)]
        dots = [Dot() for _ in range(3)]
        [dot.scale(0.5) for dot in dots]
        top_right_array = [Rectangle(height=0.5, width=0.3) for _ in range(30)]
        top_array = top_left_array + dots + top_right_array
        top_array[0].set_color(BLUE)
        top_array[0].move_to(start_pos)

        for i in range(1, len(top_left_array)):
            top_array[i].set_color(BLUE)
            top_array[i].next_to(top_array[i - 1], RIGHT, buff=0)

        for i in range(len(top_left_array), len(top_left_array) + 4):
            top_array[i].set_color(BLUE)
            top_array[i].next_to(top_array[i - 1], RIGHT, buff=0.1)
        for i in range(len(top_left_array) + 4, len(top_array)):
            top_array[i].set_color(BLUE)
            top_array[i].next_to(top_array[i - 1], RIGHT, buff=0)

        scale = 0.5
        animations_rect = [GrowFromCenter(obj) for obj in top_array]
        left_text = generate_chars_in_array(top_left_array, 'A', YELLOW, scale=scale)
        animations_text = [ShowCreation(char) for char in left_text]
        display_anims = animations_rect + animations_text
        self.play(*display_anims)

        self.count = 2020 - 11
        self.prev_counter_text = TextMobject("Insertions: {0}".format(self.count))
        self.counter_position = 3.5 * UP + 4 * RIGHT
        self.prev_counter_text.move_to(self.counter_position)
        self.next_counter_text = TextMobject("Insertions: {0}".format(self.count + 1))
        self.next_counter_text.move_to(self.counter_position)
        self.space_counter_text = TextMobject("Space: {0}".format(1024))
        self.space_counter_text.move_to(3 * UP + 4 * RIGHT)

        curved_arrow_top_start = top_array[0].get_center() + 0.3 * LEFT
        curved_arrow_top_end = top_array[0].get_center() + UP
        curved_arrow_top = CurvedArrow(curved_arrow_top_start, curved_arrow_top_end, angle= -TAU / 2)
        curved_arrow_top.set_color(RED)
        length_top = TextMobject("Length: 1024")
        length_top.move_to(curved_arrow_top_end + 1.5 * RIGHT)
        self.play(FadeIn(curved_arrow_top), FadeIn(length_top))


        self.play(*[ShowCreation(self.prev_counter_text), ShowCreation(self.space_counter_text)])
        
        run_time = 0.5

        start_char = 'A'
        self.wait(2)
        for i in range(5):
            new_char = TextMobject(start_char)
            new_char.scale(scale)
            new_char.move_to(dots[1].get_center() + UP * 0.5)
            self.play(ShowCreation(new_char), run_time=run_time)
            counter_transform = ReplacementTransform(self.prev_counter_text, self.next_counter_text)
            self.next_counter_text, self.prev_counter_text = self.update_counter(), self.next_counter_text
            self.play(FadeOutAndShift(new_char, direction=0.5*DOWN), counter_transform, run_time=0.3)
            start_char = chr(ord(start_char) + 1)

        for i in range(6):
            new_char = TextMobject(start_char)
            new_char.scale(scale)
            new_char.move_to(top_right_array[i].get_center() + UP * 0.5)
            self.play(ShowCreation(new_char), run_time=0.3)
            text_copy = new_char.copy()
            text_copy.set_color(YELLOW)
            text_copy.move_to(top_right_array[i].get_center())
            counter_transform = ReplacementTransform(self.prev_counter_text, self.next_counter_text)
            self.next_counter_text, self.prev_counter_text = self.update_counter(), self.next_counter_text
            self.play(ReplacementTransform(new_char, text_copy), counter_transform, run_time=0.3)
            start_char = chr(ord(start_char) + 1)

        self.wait(6.7)



    def update_counter(self):
        self.count += 1
        new_counter_text = TextMobject("Insertions: {0}".format(self.count + 1))
        new_counter_text.move_to(self.counter_position)
        return new_counter_text


class LargeIterationCountDouble(Scene):
    def construct(self):
        start_pos = 0.5 * UP + 6 * LEFT
        top_left_array = [Rectangle(height=0.5, width=0.3) for _ in range(10)]
        dots = [Dot() for _ in range(3)]
        [dot.scale(0.5) for dot in dots]
        top_right_array = [Rectangle(height=0.5, width=0.3) for _ in range(30)]
        top_array = top_left_array + dots + top_right_array
        top_array[0].set_color(BLUE)
        top_array[0].move_to(start_pos)

        for i in range(1, len(top_left_array)):
            top_array[i].set_color(BLUE)
            top_array[i].next_to(top_array[i - 1], RIGHT, buff=0)

        for i in range(len(top_left_array), len(top_left_array) + 4):
            top_array[i].set_color(BLUE)
            top_array[i].next_to(top_array[i - 1], RIGHT, buff=0.1)
        for i in range(len(top_left_array) + 4, len(top_array)):
            top_array[i].set_color(BLUE)
            top_array[i].next_to(top_array[i - 1], RIGHT, buff=0)

        scale = 0.4
        animations_rect = [GrowFromCenter(obj) for obj in top_array]
        left_text = generate_chars_in_array(top_left_array, 'A', YELLOW, scale=scale)
        animations_text = [ShowCreation(char) for char in left_text]
        display_anims = animations_rect + animations_text
        self.play(*display_anims)

        num_insertions = 8
        self.count = 2048572 - num_insertions
        self.prev_counter_text = TextMobject("Insertions: {0}".format(self.count))
        self.counter_position = 3.5 * UP + 4 * RIGHT
        self.prev_counter_text.move_to(self.counter_position)
        self.next_counter_text = TextMobject("Insertions: {0}".format(self.count + 1))
        self.next_counter_text.move_to(self.counter_position)
        self.space_counter_text = TextMobject("Space: {0}".format(1048576))
        self.space_counter_text.move_to(3 * UP + 4 * RIGHT)

        curved_arrow_top_start = top_array[0].get_center() + 0.3 * LEFT
        curved_arrow_top_end = top_array[0].get_center() + UP
        curved_arrow_top = CurvedArrow(curved_arrow_top_start, curved_arrow_top_end, angle= -TAU / 2)
        curved_arrow_top.set_color(RED)
        length_top = TextMobject("Length: 1048576")
        length_top.move_to(curved_arrow_top_end + 2 * RIGHT)
        self.play(FadeIn(curved_arrow_top), FadeIn(length_top))


        self.play(*[ShowCreation(self.prev_counter_text), ShowCreation(self.space_counter_text)])
        
        run_time = 0.2

        start_char = 'A'
        for i in range(num_insertions):
            new_char = TextMobject(start_char)
            new_char.scale(scale)
            new_char.move_to(dots[1].get_center() + UP * 0.5)
            self.play(ShowCreation(new_char), run_time=run_time)
            counter_transform = ReplacementTransform(self.prev_counter_text, self.next_counter_text)
            self.next_counter_text, self.prev_counter_text = self.update_counter(), self.next_counter_text
            self.play(FadeOutAndShift(new_char, direction=0.5*DOWN), counter_transform, run_time=0.3)
            start_char = chr(ord(start_char) + 1)

        self.wait(1.3)

        # for i in range(6):
        #     new_char = TextMobject(start_char)
        #     new_char.scale(scale)
        #     new_char.move_to(top_right_array[i].get_center() + UP * 0.5)
        #     self.play(ShowCreation(new_char), run_time=0.3)
        #     text_copy = new_char.copy()
        #     text_copy.set_color(YELLOW)
        #     text_copy.move_to(top_right_array[i].get_center())
        #     counter_transform = ReplacementTransform(self.prev_counter_text, self.next_counter_text)
        #     self.next_counter_text, self.prev_counter_text = self.update_counter(), self.next_counter_text
        #     self.play(ReplacementTransform(new_char, text_copy), counter_transform, run_time=0.3)
        #     start_char = chr(ord(start_char) + 1)

    def update_counter(self):
        self.count += 1
        new_counter_text = TextMobject("Insertions: {0}".format(self.count + 1))
        new_counter_text.move_to(self.counter_position)
        return new_counter_text


def create_array(num_rectangles, color, start_pos, height, width):
    initial_array = [Rectangle(height=height, width=width) for _ in range(num_rectangles)]
    for rect in initial_array:
        rect.set_color(color)
    initial_array[0].move_to(start_pos)
    for i in range(1, len(initial_array)):
        initial_array[i].next_to(initial_array[i - 1], RIGHT, buff=0)
    return initial_array

def generate_chars_in_array(array, start_char, color, scale=1):
    text_objects = []
    for rect in array:
        text = TextMobject(start_char)
        text.set_color(color)
        text.scale(scale)
        text.move_to(rect.get_center())
        text_objects.append(text)
        start_char = chr(ord(start_char) + 1)
    return text_objects


class CalculateGeneralPlus1(Scene):
    def construct(self):
        # generalize = TextMobject("Can we generalize this?")
        # self.play(Write(generalize), run_time=3)
        # self.wait(8)
        # self.play(FadeOut(generalize), run_time=3)
        # self.wait(2)

        self.wait()

        left_character = create_confused_char(color=RED, scale=0.7, position=LEFT * 3.5)
        left_thought_bubble = SVGMobject("thought")
        left_thought_bubble.scale(1.2)
        left_thought_bubble.move_to(left_character.get_center() + UP * 2 + RIGHT * 2)
        done = TextMobject("Ok... but can we", "generalize this?")
        done.scale(0.5)
        done[0].move_to(left_thought_bubble.get_center() + UP * 0.55 + RIGHT * 0.04)
        done[1].move_to(left_thought_bubble.get_center() + UP * 0.25 + RIGHT * 0.04)
        done.set_color(BLACK)

        self.play(FadeIn(left_character))
        self.play(FadeIn(left_thought_bubble), FadeIn(done))
        self.wait(4)

        right_character = create_computer_char(color=BLUE, scale=0.7, position=RIGHT * 3.5)
        right_thought_bubble = SVGMobject("thought")
        right_thought_bubble.flip()
        right_thought_bubble.scale(1.25)
        right_thought_bubble.move_to(right_character.get_center() + UP * 2 + LEFT * 2)
        simplify = TextMobject("Pick me! I think", "I see a pattern.")
        simplify.scale(0.5)
        simplify[0].move_to(right_thought_bubble.get_center() + UP * 0.55 + RIGHT * 0.03)
        simplify[1].move_to(right_thought_bubble.get_center() + UP * 0.25 + RIGHT * 0.04)
        simplify.set_color(BLACK)

        self.play(FadeIn(right_character))
        self.play(FadeIn(right_thought_bubble), FadeIn(simplify))
        self.wait(4)

        self.play(FadeOut(left_character), FadeOut(right_character), FadeOut(done), 
            FadeOut(simplify), FadeOut(left_thought_bubble), FadeOut(right_thought_bubble))
        self.wait(2)


        space = TextMobject("Inserting $N$ elements ", r"$\rightarrow$", "$N$ units of space")
        for i in range(len(space)):
            self.play(Write(space[i]), run_time=2)
            self.wait()

        self.wait(2)
        self.play(FadeOut(space))

        insertions = TextMobject("What about the insertion count?")
        self.wait()
        self.play(Write(insertions), run_time=2)
        self.wait(2)
        self.play(FadeOut(insertions))


        eq = TexMobject(r"\text{Insertions} ",
            r"= ",
            r"(", 
            r"4 + ",
            r"5 + ",
            r"6 + ",
            r"\ldots", 
            r"+ N",
            r")")

        intermediate_objs = []
        curr_array = create_array(4, BLUE, UP + LEFT * 4, 0.5, 0.7)
        chars = generate_chars_in_array(curr_array, 'A', GREEN, scale=0.6)
        intermediate_objs.extend(curr_array + chars)
        chars_above = []
        for char in chars:
            char_copy = char.copy()
            char_copy.shift(UP * 0.5)
            char_copy.set_color(RED)
            chars_above.append(char_copy)


        eq.shift(UP * 3 + LEFT * 2)
        self.play(Write(eq[0:2]))
        for i in range(2, len(eq)):
            self.play(Write(eq[i]), run_time=1)

            if i > 2 and i < 5:
                new_letter = TextMobject(chr(ord('A') + i + 1))
                new_letter.move_to(curr_array[-1].get_center() + curr_array[-1].get_width() * RIGHT)
                new_letter.set_color(RED)
                new_letter.scale(0.6)
                curr_array = create_array(i + 2, BLUE, curr_array[0].get_center() + DOWN, 0.5, 0.7)
                prev_chars = [char.copy() for char in chars] + [new_letter]
                chars = generate_chars_in_array(curr_array, 'A', GREEN, scale=0.6)
                intermediate_objs.extend(chars + curr_array)
                self.play(*[GrowFromCenter(arr) for arr in curr_array])
                self.play(FadeIn(new_letter))
                self.play(*[TransformFromCopy(char_above, char) for char_above, char in zip(prev_chars[:-1], chars[:-1])] + [ReplacementTransform(new_letter, chars[-1])])
                self.wait()

            if i == 5:
                self.wait(2)
                self.play(*[FadeOut(obj) for obj in intermediate_objs])
                self.wait(2)

            if i == 2:
                self.play(*[GrowFromCenter(arr) for arr in curr_array])
                self.play(*[FadeIn(char) for char in chars_above])
                self.play(*[ReplacementTransform(char_above, char) for char_above, char in zip(chars_above, chars)])


        eq2 = eq[1:].copy()
        eq2.shift(DOWN * 1)
        self.play(Write(eq2[0]))
        eq2[1:].shift(RIGHT * 2.6)
        eq2[1:-1].set_color(RED)
        self.play(TransformFromCopy(eq[3:], eq2[2:]), run_time=2)
        eq3 = TexMobject(r"(1 + 2 + 3 + ",) # 1 + 2 + 3
        eq3[1:-1].set_color(YELLOW)
        eq3.shift(UP * 2 + 0.5 * RIGHT + 2 * LEFT)
        self.play(Write(eq3), run_time=2)
        eq4 = TexMobject(r" - \ 6")
        eq4.set_color(YELLOW)
        eq4.shift(UP * 2 + RIGHT * 4.8)
        self.play(Write(eq4), run_time=2)

        self.wait(3)

        buff = 0
        left_triangle = []
        width, height = 0.5, 0.5
        for i in range(1, 7):
            array = create_array_left_to_right(i, BLUE, LEFT * 5 + i * height * DOWN, height, width, buff=buff)
            left_triangle.append(array)
            self.play(*[FadeIn(arr) for arr in array], run_time=1.5)

        line = DashedLine(left_triangle[-1][0].get_center() + 0.5 * DOWN + LEFT * height / 2, 
            left_triangle[-1][-1].get_center() + 0.5 * DOWN + RIGHT * height / 2)
        N = TexMobject("N")
        N.next_to(line, DOWN * 0.5)
        self.play(FadeIn(line), FadeIn(N))
        self.wait(7.5)

        right_triangle = []
        width, height = 0.5, 0.5
        start_point = left_triangle[-1][-1].get_center() + (height + buff) * UP
        for i in range(1, 7):
            array = create_array_right_to_left(i, RED, start_point, height, width, buff=buff)
            right_triangle.append(array)
            start_point = start_point + height * UP
        transforms = []
        for array_l, array_r in zip(left_triangle, right_triangle):
            for rect_l, rect_r in zip(array_l, array_r):
                transforms.append(TransformFromCopy(rect_l, rect_r))
        self.play(*transforms, run_time=4)
        line2 = DashedLine(left_triangle[-1][0].get_center() + 0.5 * LEFT + DOWN * height / 2, 
            right_triangle[-1][-1].get_center() + 0.5 * LEFT + UP * height / 2)
        N_plus = TexMobject("N + 1")
        N_plus.rotate(-TAU / 4)
        N_plus.next_to(line2, LEFT * 0.5)
        self.wait(3)
        self.play(FadeIn(line2), FadeIn(N_plus))

        left_triangle_copy_anims = []
        for array_l in left_triangle:
            for rect_l in array_l:
                rect_l_copy = rect_l.copy()
                rect_l_copy.set_color(GREEN)
                left_triangle_copy_anims.append(ReplacementTransform(rect_l, rect_l_copy))

        self.wait(10)

        eq2_copy = eq2[2:-1].copy()
        eq3_copy = eq3[1:-1].copy()
        eq2_copy.set_color(GREEN)
        eq3_copy.set_color(GREEN)
        green_animations = [ReplacementTransform(eq2[2:-1], eq2_copy),  ReplacementTransform(eq3[1:-1], eq3_copy)] + left_triangle_copy_anims

        self.play(*green_animations, run_time=4)
        self.wait(2)



        eq5 = TexMobject(r" = \frac{N(N+1)}{2} - 6", r"= \frac{1}{2} N ^ 2 + \frac{1}{2}N - 6")
        eq5.shift(UP + RIGHT * 0.5)
        eq5[0][:-2].set_color(GREEN)
        self.play(Write(eq5[0]), run_time=2)
        self.wait(9)
        self.play(Write(eq5[1]), run_time=2)

        self.wait(5)


class CalculateGeneralDouble(Scene):
    def construct(self):
        generalizing = TextMobject("Generalizing",  r"$\ldots$")
        generalizing.shift(UP)
        self.play(FadeIn(generalizing))
        self.wait()
        space = TextMobject("Inserting $N$ elements ", r"$\rightarrow$", r"$\leq 2N$ units of space")
        for i in range(len(space)):
            self.play(Write(space[i]), run_time=2)
            self.wait(0.5)
        self.wait(2)

        brace_bottom1 = Brace(space[-1], DOWN, buff=SMALL_BUFF)
        O_N = TextMobject("$O(N)$", "memory")
        O_N[0].set_color(GREEN)
        O_N.next_to(brace_bottom1, DOWN)
        self.play(GrowFromCenter(brace_bottom1), FadeIn(O_N))
        self.wait(2)
        self.play(FadeOut(space), FadeOut(O_N), FadeOut(brace_bottom1), FadeOut(generalizing))
        self.wait()

        eq = TexMobject(r"\text{Insertions} ",
            r"= ",
            r"(", 
            r"4 + ",
            r"8 + ",
            r"16 + ",
            r"\ldots", 
            r"+ N",
            r")")
        eq.shift(UP * 3 + LEFT * 2)
        self.play(Write(eq[0]))
        self.wait(2)
        self.play(Write(eq[1]))
        self.wait(3)
        self.play(Write(eq[2]), run_time=1)
        for i in range(3, len(eq)):
            self.play(Write(eq[i]), run_time=1)
            if i == 3:
                self.wait(4)
            elif i == 4:
                self.wait()

        self.wait(7)
        eq2 = eq[1:].copy()
        eq2.shift(DOWN * 1)
        self.play(Write(eq2[0]))
        eq2[1:].shift(RIGHT * 1.9)
        eq2[1:-1].set_color(RED)
        self.play(TransformFromCopy(eq[3:], eq2[2:]))
        eq3 = TexMobject(r"(1 + 2 + ",) # 1 + 2 
        eq3[1:-1].set_color(YELLOW)
        eq3.shift(UP * 2 + 1.9 * LEFT)
        eq3[-1].shift(RIGHT * 0.1)
        self.play(Write(eq3))
        eq4 = TexMobject(r" - \ 3")
        eq4.set_color(YELLOW)
        eq4.shift(UP * 2 + RIGHT * 4.3)
        self.play(Write(eq4))

        eq5 = TexMobject(r"= (N + \frac{N}{2} + \frac{N}{4} + \ldots + 4 + 2 + 1) - 3")
        eq5[2:-7].set_color(RED)
        eq5[-6:-3].set_color(YELLOW)
        eq5[-2:].set_color(YELLOW)
        eq5.shift(UP + RIGHT * 0.6)

        self.wait(10)
        self.play(Write(eq5))

        self.wait(13)

        line = Line(LEFT * 6 + DOWN * 3, RIGHT * 6 + DOWN * 3)
        line.set_color(BLUE)
        N = TexMobject("N")
        N.next_to(line, UP)
        self.play(FadeIn(line))
        self.play(ShowCreation(N))

        length = 12
        start_point = line.get_start() + DOWN * 0.2
        colors = [YELLOW, GREEN, RED, PURPLE, ORANGE]
        for i in range(12):
            length = length / 2
            end_point = start_point + length * RIGHT
            line = Line(start_point, end_point)
            line.set_color(colors[i % len(colors)])
            if i < 2:
                self.wait(3)
            elif i == 2:
                self.wait()
            self.play(ShowCreation(line))
            start_point = end_point

        self.wait(8)


        box = SurroundingRectangle(eq5[1:-2])
        box.set_color(GREEN)
        rectangle = Rectangle(height=1.5, width=12.5)
        rectangle.move_to(DOWN * 3.1)
        rectangle.set_color(GREEN)
        self.play(ShowCreation(box), ShowCreation(rectangle))

        outside_eq = TexMobject(r"\approx 2N")
        outside_eq.move_to(LEFT * 5.5)
        outside_eq.set_color(GREEN)

        arrow1 = Arrow(LEFT * 5.1, LEFT * 3 + UP * 0.5)
        arrow1.set_color(GREEN)
        arrow2 = Arrow(LEFT * 5.3 + DOWN * 0.1, LEFT * 5.3 + DOWN * 2.5)
        arrow2.set_color(GREEN)

        self.play(Write(outside_eq), Write(arrow1), Write(arrow2))
        self.wait(6)


        eq6 = TexMobject(r"= (\frac{1 - 2 ^ {\log_2(n) + 1}}{1 - 2}) - 3 = (2N - 1) - 3")
        eq6.shift(DOWN * 0.5 + 0.5 * RIGHT)
        eq6[:-11].set_color(GREEN)
        eq6[:-11:-9].set_color(YELLOW)
        self.play(Write(eq6[:-8]))
        self.wait(2)
        eq6[-8:-2].set_color(GREEN)
        eq6[-2:].set_color(YELLOW)
        self.play(Write(eq6[-8:]))
        self.wait(2)
        eq7 = TexMobject(r"= 2N - 4", r"\rightarrow", r"O(N)")
        eq7[-1].set_color(BLUE)
        eq7.shift(DOWN * 1.5 + LEFT * 1.6)
        for i in range(len(eq7)):
            self.play(Write(eq7[i]))

        self.wait(2)

        improve = TextMobject("This is a huge improvement!")
        improve.shift(DOWN * 1.5 + RIGHT * 3.6)
        improve.scale(0.9)
        self.play(Write(improve))
        self.wait(10)


class CalculateGeneralPlus8(Scene):
    def construct(self):
        text = TextMobject("Generalizing ...")
        self.play(Write(text), run_time = 2)
        self.wait()
        self.play(FadeOut(text))
        space = TextMobject("Inserting $N$ elements ", r"$\rightarrow$", r"$\leq N + 8$ units of space")
        for i in range(len(space)):
            self.play(Write(space[i]), run_time=2)
        self.wait(2)
        self.play(FadeOut(space))
        self.wait()
        eq = TexMobject(r"\text{Insertions} ",
            r"= ",
            r"(", 
            r"4 + ",
            r"12 + ",
            r"20 + ",
            r"\ldots", 
            r"+ N",
            r")")
        eq.shift(UP * 3 + LEFT * 2)
        self.play(Write(eq[0]))
        for i in range(1, len(eq)):
            self.play(Write(eq[i]), run_time=1.5)

        self.wait(3)

        eq2 = TexMobject(r"=", r"\left( \frac{4 + N}{2} \right)", r"\left( \frac{N + 4}{8} \right)")
        eq2.shift(UP * 2 + LEFT * 1.2)
        for i in range(len(eq2)):
            self.play(Write(eq2[i]), run_time=1)
        self.wait(2)
        brace_bottom1 = Brace(eq2[1], DOWN, buff = SMALL_BUFF)
        text_bottom1 = brace_bottom1.get_text("Average")
        text_bottom1.scale(0.5)
        text_bottom1.shift(UP * 0.3)
        self.play(
            GrowFromCenter(brace_bottom1),
            FadeIn(text_bottom1)
            )

        brace_bottom2 = Brace(eq2[2], DOWN, buff = SMALL_BUFF)
        text_bottom2 = brace_bottom2.get_text("Number of Elements")
        text_bottom2.scale(0.5)
        text_bottom2.shift(UP * 0.3)    
        self.play(
            GrowFromCenter(brace_bottom2),
            FadeIn(text_bottom2)
            )

        self.wait(8)

        eq3 = TexMobject(r"=", r"\frac{1}{16} N ^ 2 + \frac{1}{2}N + 1")
        eq3.shift(LEFT * 1.6)
        for i in range(len(eq3)):
            self.play(Write(eq3[i]))

        self.wait(4)

        self.play(FadeOut(eq), FadeOut(eq2), FadeOut(brace_bottom1), 
            FadeOut(brace_bottom2), FadeOut(text_bottom1), 
            FadeOut(text_bottom2), FadeOut(eq3))
        self.wait()



        # eq2 = eq[1:].copy()
        # eq2.shift(DOWN * 1)
        # self.play(Write(eq2[0]))
        # eq2[-2][-1].move_to(eq2[3][0].get_center())
        # self.play(TransformFromCopy(eq[3], eq2[2]), 
        #     TransformFromCopy(eq[-2][-1], eq2[-2][-1]))

class IntroduceBigO(Scene):
    def construct(self):
        left_text = TextMobject("Resizing by 1 element")
        right_text = TextMobject("Resizing by 8 elements")
        left_text.move_to(LEFT * 4 + UP * 3)
        right_text.move_to(RIGHT * 4 + UP * 3)
        self.play(Write(left_text), Write(right_text))

        self.wait(4)

        left_insertions = TexMobject(r"N \ \text{Insertions} \rightarrow \frac{1}{2} N ^ 2 + \frac{1}{2}N - 6")
        left_insertions.move_to(LEFT * 3.5 + UP)
        right_insertions = TexMobject(r"N \ \text{Insertions} \rightarrow \frac{1}{16} N ^ 2 + \frac{1}{2}N + 1")
        right_insertions.move_to(RIGHT * 4 + UP)
        left_insertions.scale(0.75)
        right_insertions.scale(0.75)
        self.play(Write(left_insertions), Write(right_insertions), run_time=2)

        self.wait(9)

        left_million = TextMobject("1 million elements", r"$\rightarrow$", "500 billion insertions")
        left_million.scale(0.65)
        left_million.move_to(LEFT * 3.5 + DOWN * 0.5)

        right_million = TextMobject("1 million elements", r"$\rightarrow$", "62.5 billion insertions")
        right_million.scale(0.65)
        right_million.move_to(RIGHT * 3.5 + DOWN * 0.5)

        for i in range(len(left_million)):
            self.play(Write(left_million[i]), Write(right_million[i]))

        self.wait(2)

        self.play(FadeOut(left_million), FadeOut(right_million))

        self.wait()


        question = TextMobject("What happens when $N$ becomes very large?")
        question.shift(DOWN * 3)

        left_N2 = left_insertions[15:17]
        all_other_left = VGroup(left_insertions[12:15], left_insertions[17:])
        right_N2 = right_insertions[16:18]
        all_other_right = VGroup(right_insertions[12:16], right_insertions[18:])

        left_N2_copy = left_insertions[15:17].copy()
        right_N2_copy = right_insertions[16:18].copy()
        left_N2_copy.scale(2)
        right_N2_copy.scale(2)



        self.play(Write(question))

        self.wait(8)

        self.play(ReplacementTransform(left_N2, left_N2_copy), 
            ReplacementTransform(right_N2, right_N2_copy),
            ShrinkToCenter(all_other_left), 
            ShrinkToCenter(all_other_right), 
            run_time=3)

        self.wait(13)

        left_sum = TexMobject(r"(4 + 5 + 6 + \ldots + N)")
        left_sum.move_to(LEFT * 4 + DOWN * 0.2)

        right_sum = TexMobject(r"(4 + 12 + 20 + \ldots + N)")
        right_sum.move_to(RIGHT * 4 + DOWN * 0.2)

        self.play(Write(left_sum), Write(right_sum))

        self.wait(4)

        left_insertions = TexMobject(r"N \ \text{Insertions} \rightarrow \frac{1}{2} N ^ 2 + \frac{1}{2}N - 6")
        right_insertions = TexMobject(r"N \ \text{Insertions} \rightarrow \frac{1}{16} N ^ 2 + \frac{1}{2}N + 1")

        left_ins, right_ins = left_insertions[12:], right_insertions[12:]

        left_ins.move_to(LEFT * 3.5 + DOWN * 1.7)
        right_ins.move_to(RIGHT * 4 + DOWN * 1.7)

  
        unnecessary = [left_sum, right_sum, left_ins, right_ins]

        crosses = [Cross(obj) for obj in unnecessary]
        for cross in crosses:
            cross.set_color(RED)

        self.play(*[ShowCreation(cross) for cross in crosses[:2]])

        self.wait(8)

        self.play(FadeIn(left_ins), FadeIn(right_ins))

        self.wait(2)

        self.play(*[ShowCreation(cross) for cross in crosses[2:]])

        self.wait(2)

        self.play(*[FadeOut(obj) for obj in unnecessary + crosses])

        rect_left = SurroundingRectangle(left_N2_copy, buff =SMALL_BUFF)
        rect_right = SurroundingRectangle(right_N2_copy, buff = SMALL_BUFF)
        rect_left.set_stroke(GREEN_SCREEN, 2)
        rect_right.set_stroke(GREEN_SCREEN, 2)

        self.play(ShowCreation(rect_left), ShowCreation(rect_right))
        self.wait()

        self.wait(8)

        big_o = TexMobject(r"O(N ^ 2)", r"\text{(Time Complexity)}")
        big_o[0].scale(1.5)
        big_o[0].set_color(BLUE)
        big_o.shift(DOWN * 0.5)
        big_o[1].shift(RIGHT * 0.7)
        self.play(DrawBorderThenFill(big_o[0]))
        self.wait(5)
        self.play(FadeIn(big_o[1]))


        big_o_space = TexMobject(r"O(N)", r"\text{(Space Complexity)}")
        big_o_space[0].scale(1.5)
        big_o_space[0].set_color(BLUE)
        big_o_space.shift(DOWN * 2)
        big_o_space[1].shift(RIGHT * 0.7)
        self.play(DrawBorderThenFill(big_o_space[0]))
        self.play(FadeIn(big_o_space[1]))
        self.wait(10)

        transform_text = TextMobject("Both resize schemes perform N insertions in quadratic time!")
        transform_text.shift(DOWN * 3)

        self.play(Transform(question, transform_text))

        self.wait(6)


class KResizeN2(Scene):
    def construct(self):
        top_text = TextMobject("Resize by 8 elements")
        top_text.shift(UP * 3)
        
        eq = TexMobject(r"\text{Insertions} ",
            r"= ",
            r"(", 
            r"4 + ",
            r"12 + ",
            r"20 + ",
            r"\ldots", 
            r"+ N",
            r")")
        eq.shift(UP * 1.5)
        self.play(Write(top_text), Write(eq))

        brace_bottom1 = Brace(eq, DOWN, buff = SMALL_BUFF)
        text_bottom1 = brace_bottom1.get_text("$O(N ^ 2)$")
        # text_bottom1.scale(0.5)

        self.play(
            GrowFromCenter(brace_bottom1),
            FadeIn(text_bottom1)
            )

        self.wait()
        self.play(FadeOut(brace_bottom1), FadeOut(text_bottom1))

        top_text2 = TextMobject("Resize by K elements?")
        top_text2.shift(UP * 3)
        self.play(Transform(top_text, top_text2))
        self.wait(2)

        eq2 = TexMobject(r"\text{Insertions} ",
            r"= ",
            r"(", 
            r"4 + ",
            r"(4 + K) + ",
            r"(4 + 2K) + ",
            r"\ldots", 
            r"+ N",
            r")")

        eq2.shift(UP * 1.5)

        self.play(Transform(eq, eq2))
        self.wait(5)

        brace_bottom2 = Brace(eq2, DOWN, buff = SMALL_BUFF)

        text_bottom2 = brace_bottom2.get_text("$O(N ^ 2)$")
        self.play(
            GrowFromCenter(brace_bottom2),
            FadeIn(text_bottom2)
            )
        self.wait(7)

        eq3 = TextMobject("Space", r"$\leq N + K$")
        eq3.shift(DOWN)
        self.play(Write(eq3))




        self.wait(6)

        better = TextMobject("Can we do better?")
        better.shift(DOWN * 2.5)
        self.play(Write(better))

        self.wait(3)

        better_transform = TextMobject("Can we do better?")

        self.play(FadeOut(eq3), FadeOut(brace_bottom2), FadeOut(text_bottom2), 
            FadeOut(top_text), FadeOut(eq), ReplacementTransform(better, better_transform), run_time=2)

        self.wait(6)

        result = TextMobject("$N$ Insertions", r"$\rightarrow$", r"$O(N)$ Time", "and", "$O(N)$ space")
        result[2][:4].set_color(GREEN)
        result[-1][:4].set_color(GREEN)
        result.shift(DOWN)

        for text in result:
            self.play(Write(text))

        self.wait(2)

        self.play(FadeOut(result), FadeOut(better_transform))
        self.wait()


class WriteCode(Scene):
    def construct(self):


        line1 = TextMobject("$>>>$ array = [ ]")
        line1.shift(LEFT * 3.5 + UP * 3)
        line1[:3].set_color(RED_A)
        
        line2 = TextMobject("$>>>$ for i in range(1, 6):")
        line2[:3].set_color(RED_A)
        line2[3:6].set_color(RED_E)
        line2[7:9].set_color(RED_E)
        line2[9:14].set_color(BLUE)
        line2[15].set_color(PURPLE_A)
        line2[17].set_color(PURPLE_A)
        line2.shift(LEFT * 2.6 + UP * 2.3)
        
        line3 = TextMobject("array.append(i)")
        line3[6:12].set_color(BLUE)
        line3.shift(LEFT * 1.7 + UP * 1.6)

        line4 = TextMobject("$>>>$ array")
        line4[:3].set_color(RED_A)
        line4.next_to(line2[:len(line4)], DOWN * 3.8)

        line5 = TextMobject("[1, 2, 3, 4, 5]")
        for i in range(1, len(line5), 2):
            line5[i].set_color(PURPLE_A)
        line5.move_to(UP * 0.5 + LEFT * 4)

        write_runtime = 1.5
        self.play(Write(line1, run_time=write_runtime))
        self.play(Write(line2, run_time=write_runtime))
        self.play(Write(line3, run_time=write_runtime))
        self.play(Write(line4), run_time=write_runtime)
        self.play(FadeIn(line5, run_time=0.5))

        prev_code_arrow = Arrow(RIGHT * 0.5 + UP * 3, 1.5 * LEFT + UP * 3)
        prev_code_arrow.set_color(GREEN)

        diagram = TextMobject("array")
        diagram.move_to(DOWN * 3 + LEFT * 5)
        line_draw1 = Line(DOWN * 1, DOWN * 1.5)
        line_draw1.next_to(diagram, RIGHT)
        line_draw2 = Line(line_draw1.get_end(), line_draw1.get_end() + RIGHT * 0.5)

        line_draw1_center = (line_draw1.get_start() + line_draw1.get_end()) / 2
        arrow_start = line_draw1_center 
        diagram_arrow = Arrow(arrow_start, arrow_start + RIGHT * 2)

        diagram_runtime = 0.5

        self.play(ShowCreation(prev_code_arrow))
        self.play(FadeIn(diagram), FadeIn(line_draw1), FadeIn(line_draw2), 
            Write(diagram_arrow), run_time=diagram_runtime)

        line2_position_start, line2_position_end = RIGHT * 2.1 + UP * 2.3,  RIGHT * 0.1 + UP * 2.3
        new_code_arrow = Arrow(line2_position_start, line2_position_end)
        new_code_arrow.set_color(GREEN)
        self.play(ReplacementTransform(prev_code_arrow, new_code_arrow), run_time=diagram_runtime)

        prev_code_arrow = new_code_arrow
        line3_position_start, line3_position_end = RIGHT * 2 + UP * 1.6,  RIGHT * 0 + UP * 1.6
        new_code_arrow = Arrow(line3_position_start, line3_position_end)
        new_code_arrow.set_color(GREEN)
        self.play(ReplacementTransform(prev_code_arrow, new_code_arrow), run_time=diagram_runtime)
        prev_code_arrow = new_code_arrow 

        line4_position_start, line4_position_end = LEFT * 0.9 + UP * 1.05,  LEFT * 2.9 + UP * 1.05

        array = create_array(5, BLUE, diagram_arrow.get_end() + RIGHT , 1, 1.5)
        array_text = generate_chars_in_array(array, '1', PURPLE_A)
        for i in range(len(array)):
            new_code_arrow = Arrow(line2_position_start, line2_position_end)
            new_code_arrow.set_color(GREEN)
            self.play(GrowFromCenter(array[i]), FadeIn(array_text[i]),
                ReplacementTransform(prev_code_arrow, new_code_arrow), run_time=diagram_runtime)
            prev_code_arrow = new_code_arrow
            if i == len(array) - 1:
                new_code_arrow = Arrow(line4_position_start, line4_position_end)
                new_code_arrow.set_color(GREEN)
                self.play(ReplacementTransform(prev_code_arrow, new_code_arrow), run_time=diagram_runtime)
            else:
                new_code_arrow = Arrow(line3_position_start, line3_position_end)
                new_code_arrow.set_color(GREEN)
                self.play(ReplacementTransform(prev_code_arrow, new_code_arrow), run_time=diagram_runtime)
            prev_code_arrow = new_code_arrow


        line6 = TextMobject("$>>>$ array[3]")
        line6.move_to(DOWN * 0.2 + LEFT * 3.8)
        line6[:3].set_color(RED_A)
        line6[-2].set_color(PURPLE_A)
        self.play(Write(line6))

        line6_position_start, line6_position_end = DOWN * 0.2 + LEFT * 0.4, DOWN * 0.2 + LEFT * 2.4
        new_code_arrow = Arrow(line6_position_start, line6_position_end)
        new_code_arrow.set_color(GREEN)
        self.play(ReplacementTransform(prev_code_arrow, new_code_arrow))
        self.play(Indicate(array_text[3]), run_time=2)
        text_copy_4 = array_text[3].copy()
        text_copy_4.move_to(DOWN * 0.7 + LEFT * 5.2)
        self.play(TransformFromCopy(array_text[3], text_copy_4))
        prev_code_arrow = new_code_arrow

        line7 = TextMobject("$>>>$ array.pop()")
        line7.move_to(DOWN * 1.3 + LEFT * 3.4)
        line7[:3].set_color(RED_A)
        line7[-5:-2].set_color(BLUE)
        self.play(Write(line7))
        line7_position_start, line7_position_end = DOWN * 1.3 + RIGHT * 0.5, DOWN * 1.3 + LEFT * 1.5
        new_code_arrow = Arrow(line7_position_start, line7_position_end)
        new_code_arrow.set_color(GREEN)
        self.play(ReplacementTransform(prev_code_arrow, new_code_arrow))

        text_copy_5 = array_text[4].copy()
        text_copy_5.move_to(DOWN * 1.8 + LEFT * 5.25)
        self.play(FadeOut(array[4]), ReplacementTransform(array_text[4], text_copy_5))

        self.wait(8.5)

        array_copy = create_array(4, BLUE, LEFT * 5, 1, 1.5)
        array_text_copy = generate_chars_in_array(array_copy, '1', PURPLE_A)
        old_and_new = zip(array[:4] + array_text[:4], array_copy + array_text_copy)
        array_anims = [ReplacementTransform(arr, arr_copy) for arr, arr_copy in old_and_new]
        all_animations = [FadeOut(line1), FadeOut(line2),
            FadeOut(line3), FadeOut(line4),
            FadeOut(line5), FadeOut(line6),
            FadeOut(line7), FadeOut(diagram_arrow),
            FadeOut(text_copy_4), FadeOut(text_copy_5),
            FadeOut(diagram), FadeOut(line_draw1), 
            FadeOut(line_draw2), FadeOut(new_code_arrow)] + array_anims

        self.play(*all_animations, run_time=3)

        self.wait()

        pointer = Arrow(LEFT * 2.8 + UP * 2, LEFT * 2.8 + UP * 0.5)
        declaration = TextMobject("Dynamic Array")
        declaration.move_to(LEFT * 2.8 + UP * 2)
        self.play(FadeIn(pointer), Write(declaration))

        self.wait()

        self.play(FadeOut(pointer), FadeOut(declaration), run_time=1.5)
        self.wait()


        top_character = create_computer_char(color=RED, scale=0.5, position=DOWN * 3.5 + LEFT * 3)
        top_thought_bubble = SVGMobject("thought")
        top_thought_bubble.scale(1.2)
        top_thought_bubble.move_to(top_character.get_center() + UP * 1.5 + RIGHT * 1.5)
        experienced = TextMobject("That's pretty cool!", "Let's move on", "to the next topic!")
        experienced.scale(0.4)
        experienced[0].move_to(top_thought_bubble.get_center() + UP * 0.70 + RIGHT * 0.04)
        experienced[1].move_to(top_thought_bubble.get_center() + UP * 0.45 + RIGHT * 0.04)
        experienced[2].move_to(top_thought_bubble.get_center() + UP * 0.20 + RIGHT * 0.04)
        experienced.set_color(BLACK)

        right_character = create_confused_char(color=BLUE, scale=0.5, position=DOWN * 3.5 + RIGHT * 3)
        right_thought_bubble = SVGMobject("thought")
        right_thought_bubble.scale(1.2)
        right_thought_bubble.move_to(right_character.get_center() + UP * 1.5 + RIGHT * 1.5)
        wait = TextMobject("But wait?", "What's actually", "going on here?")
        wait.scale(0.4)
        wait[0].move_to(right_thought_bubble.get_center() + UP * 0.70 + RIGHT * 0.04)
        wait[1].move_to(right_thought_bubble.get_center() + UP * 0.45 + RIGHT * 0.04)
        wait[2].move_to(right_thought_bubble.get_center() + UP * 0.20 + RIGHT * 0.04)
        wait.set_color(BLACK)




        new_array = create_array(4, BLUE, array_copy[-1].get_center() + 
            RIGHT * array_copy[-1].get_width() * RIGHT, 1, 1.5)
        new_array_text = generate_chars_in_array(new_array, '5', PURPLE_A)
        for i in range(len(new_array)):
            self.play(GrowFromCenter(new_array[i]), FadeIn(new_array_text[i]), run_time=1.5)
            if i == len(new_array) // 2:
                self.play(FadeIn(top_character))
                self.play(FadeIn(top_thought_bubble), FadeIn(experienced))
            else:
                self.wait()

        total_array = array_copy + new_array
        # print([arr.get_center() for arr in total_array])
        total_array_copy = [arr.copy() for arr in total_array]
        total_array_text = array_text_copy + new_array_text
        total_array_text_copy = [text.copy() for text in total_array_text]

        split_array, split_array_text, transforms = self.split(total_array, total_array_text, array_copy[0].get_center())
        self.play(*transforms, run_time=2)
        half = len(split_array) // 2
        start_char = '1'
        all_first_arrays = []
        all_second_arrays = []
        all_first_text = []
        all_second_text = []
        # top_text = TextMobject("How do dynamic arrays really work?")
        # top_text.shift(UP * 3)
        for i in range(half):
            first_index, second_index = i, i + half
            first_char, second_char = chr(ord(start_char) + half + i), chr(ord(start_char) + i)
            first_text, second_text = TextMobject(first_char), TextMobject(second_char)
            first_text.set_color(PURPLE_A)
            second_text.set_color(PURPLE_A)
            first_arr, second_arr = Rectangle(height=1, width=1.5), Rectangle(height=1, width=1.5)
            first_arr.set_color(BLUE)
            second_arr.set_color(BLUE)
            first_arr.move_to(split_array[first_index].get_center() + 6 * RIGHT)
            second_arr.move_to(split_array[second_index].get_center() + 6 * RIGHT)
            first_text.move_to(first_arr.get_center())
            second_text.move_to(second_arr.get_center())
            all_first_arrays.insert(0, first_arr)
            all_second_arrays.insert(0, second_arr)
            all_first_text.insert(0, first_text)
            all_second_text.insert(0, second_text)
            self.wait()
            if i == 1:
                self.play(GrowFromCenter(first_arr), GrowFromCenter(second_arr), 
                    FadeIn(first_text), FadeIn(second_text), FadeIn(right_character), 
                    FadeIn(right_thought_bubble), FadeIn(wait), run_time=1.5)
            else:
                self.play(GrowFromCenter(first_arr), GrowFromCenter(second_arr), FadeIn(first_text), FadeIn(second_text), run_time=1.5)

        self.wait()
        bottom_text = TextMobject("If dynamic arrays didn't exist,", "how would we go about creating one?")
        bottom_text[0].shift(UP * 3.5 + RIGHT * 4)
        bottom_text[1].shift(UP * 3 + LEFT * 3.5)
        for i in range(half):
            first_arr, second_arr = all_first_arrays[i], all_second_arrays[i]
            first_text, second_text = all_first_text[i], all_second_text[i]
            self.wait()
            if i == half - 1:
                self.play(FadeOut(first_arr), FadeOut(second_arr), FadeOut(first_text), 
                    FadeOut(second_text), Write(bottom_text[0]), run_time=1.5)
            else:
                self.play(FadeOut(first_arr), FadeOut(second_arr), FadeOut(first_text), FadeOut(second_text), run_time=1.5)

        transforms = []
        # print([arr.get_center() for arr in split_array])
        # print([arr.get_center() for arr in total_array_copy])
        for curr_obj, new_obj in zip(split_array + split_array_text, total_array_copy + total_array_text_copy):
            transforms.append(ReplacementTransform(curr_obj, new_obj, run_time=1.5))

        transforms.append(Write(bottom_text[1]))
        self.wait()
        self.play(*transforms, run_time=2)
        self.wait()
        self.play(*[FadeOut(obj) for obj in total_array_copy + total_array_text_copy])
        self.wait()
        self.play(FadeOut(bottom_text), FadeOut(top_character), FadeOut(top_thought_bubble), 
            FadeOut(experienced), FadeOut(right_character), FadeOut(right_thought_bubble), FadeOut(wait))



    def split(self, array, array_text, top_position):
        new_array_copy = create_array(len(array), BLUE, array[0].get_center() + UP, 1, 1.5)
        new_array_text_copy = generate_chars_in_array(new_array_copy, '1', PURPLE_A)
        for i in range(len(new_array_copy) // 2, len(new_array_copy)):
            shifted_i = i - len(new_array_copy) // 2
            new_array_copy[i].move_to(array[0].get_center() + DOWN * 0.2 + shifted_i * array[i].get_width() * RIGHT)
            new_array_text_copy[i].move_to(new_array_copy[i].get_center())

        original_array_and_text = array + array_text
        return new_array_copy, new_array_text_copy, [ReplacementTransform(arr, arr_copy) for arr, arr_copy in zip(original_array_and_text, 
            new_array_copy + new_array_text_copy)]



class ResizeComments(Scene):
    def construct(self):
        issue = TextMobject("What's the core issue with our current schemes?")
        self.play(Write(issue))
        self.wait(3)
        self.play(FadeOut(issue))
        array1 = create_array(5, BLUE, UP * 2 + LEFT * 2, 1, 1.5)
        text_in_array1 = generate_chars_in_array(array1, 'A', GREEN)

        self.play(*[FadeIn(rect) for rect in array1[:4] + text_in_array1[:4]])
        self.wait()

        text_in_array1[-1].set_color(WHITE)

        self.play(FadeIn(text_in_array1[-1]))

        array2 = create_array(5, RED, LEFT * 2, 1, 1.5)
        self.play(*[GrowFromCenter(rect) for rect in array2])
        self.wait()

        text = TextMobject("Resizes are the most expensive operations")
        text.shift(DOWN * 2)
        self.play(Write(text))

        new_text = []

        for i in range(len(text_in_array1)):
            text_copy = text_in_array1[i].copy()
            text_copy.set_color(GREEN)
            text_copy.move_to(array2[i].get_center())
            new_text.append(text_copy)
            if i != len(text_in_array1) - 1:
                self.play(TransformFromCopy(text_in_array1[i], text_copy))
            else:
                self.play(ReplacementTransform(text_in_array1[i], text_copy))

        self.play(*[FadeOut(obj) for obj in text_in_array1[:4] + array1[:4]])
        self.wait()

        large_array = create_array(15, BLUE, UP * 3 + LEFT * 5, 0.5, 0.7)

        large_array_text = generate_chars_in_array(large_array, 'A', GREEN, scale=0.6)

        transforms = []
        for i in range(len(new_text)):
            transforms.append(ReplacementTransform(new_text[i], large_array_text[i]))
            transforms.append(ReplacementTransform(array2[i], large_array[i]))


        for i in range(5, len(large_array) - 1):
            transforms.append(FadeIn(large_array[i]))
            transforms.append(FadeIn(large_array_text[i]))

        large_array_text[-1].set_color(WHITE)
        transforms.append(FadeIn(large_array_text[-1]))
        self.play(*transforms)

        self.wait(2)

        resized_large_array = create_array(15, RED, UP * 2 + LEFT * 5, 0.5, 0.7)
        self.play(*[GrowFromCenter(arr) for arr in resized_large_array])
        self.wait(2)

        text2 = TextMobject("especially when the array is large")
        text2.shift(DOWN * 2.7)
        self.play(Write(text2))

        self.wait()

        new_text2 = []
        for i in range(len(large_array)):
            text_copy = large_array_text[i].copy()
            text_copy.set_color(GREEN)
            text_copy.move_to(resized_large_array[i].get_center())
            new_text2.append(text_copy)
            if i != len(large_array_text) - 1:
                self.play(TransformFromCopy(large_array_text[i], text_copy), run_time=0.5)
            else:
                self.play(ReplacementTransform(large_array_text[i], text_copy), run_time=0.5)

        self.wait()




        
        

class AmortizedRuntime(Scene):
    def construct(self):
        question = TextMobject("What's the runtime of a single insert operation?")
        question.shift(UP * 3)
        self.play(Write(question), run_time=2)

        self.wait(12)

        N_insertions = TextMobject("$N$ insertions", r"$\rightarrow$", "$O(N)$")
        N_insertions.shift(UP * 2)
        self.play(Write(N_insertions))

        self.wait(5)

        average_case = TextMobject("Average Case")
        average_case.shift(LEFT * 4 + UP)
        self.play(Write(average_case))

        average_array = create_array(4, BLUE, LEFT * 5, 0.5, 0.8)

        self.play(*[GrowFromCenter(rect) for rect in average_array])
        text_average_array = [TextMobject(char) for char in ['A', 'B', 'C', 'D']]
        one_insertion = TextMobject("$1$ insertion", r"$\rightarrow$", "$O(1)$")
        one_insertion.shift(DOWN + LEFT * 4)

        for i in range(len(average_array)):


            text_average_array[i].scale(0.7)
            text_average_array[i].next_to(average_array[i], UP)
            self.play(Write(text_average_array[i]), run_time=0.5)
            text_copy = text_average_array[i].copy()
            text_copy.move_to(average_array[i].get_center())
            text_copy.set_color(GREEN)
            if i == 1:
                self.play(FadeIn(one_insertion), ReplacementTransform(text_average_array[i], text_copy), run_time=0.5)
            else:
                self.play(ReplacementTransform(text_average_array[i], text_copy), run_time=0.5)

        
        self.wait(4)
        brace_bottom = Brace(one_insertion[2], DOWN, buff=SMALL_BUFF)
        brace_text = TextMobject("Amortized Runtime")
        brace_text.next_to(brace_bottom, DOWN)
        brace_text.scale(0.8)
        self.play(Write(brace_bottom), Write(brace_text))

        self.wait(10)

        worst_text = TextMobject("Worst Case")
        worst_text.shift(RIGHT * 4 + UP)
        self.play(Write(worst_text))

        self.wait()

        worst_array = create_array(4, BLUE, RIGHT * 3, 0.5, 0.8)
        text_worst_array = generate_chars_in_array(worst_array, 'A', GREEN, scale=0.7)
        array_and_text = zip(average_array + text_average_array, worst_array + text_worst_array)
        self.play(*[TransformFromCopy(average, worst) for average, worst in array_and_text])
        new_char = TextMobject('E')
        new_char.scale(0.8)
        new_char.next_to(worst_array[-1], RIGHT)
        self.wait(2)
        self.play(Write(new_char))
        self.wait(4)
        new_array = create_array(8, BLUE, RIGHT + DOWN, 0.5, 0.8)
        self.play(*[GrowFromCenter(arr) for arr in new_array])


        for i in range(len(text_worst_array)):
            text_copy = text_worst_array[i].copy()
            text_copy.move_to(new_array[i].get_center())
            self.play(TransformFromCopy(text_worst_array[i], text_copy))

        new_char_copy = new_char.copy()
        new_char_copy.move_to(new_array[i+1].get_center())
        new_char_copy.set_color(GREEN)
        self.play(ReplacementTransform(new_char, new_char_copy))

        one_insertion2 = TextMobject("$1$ insertion", r"$\rightarrow$", "$O(N)$")
        one_insertion2.shift(DOWN * 2 + RIGHT * 3)
        self.play(Write(one_insertion2))
        self.wait()

        brace_bottom2 = Brace(one_insertion2[2], DOWN, buff=SMALL_BUFF)
        brace_text2 = TextMobject("Worst Case Runtime")
        brace_text2.next_to(brace_bottom2, DOWN)
        brace_text2.scale(0.8)
        self.play(Write(brace_bottom2), Write(brace_text2))
        self.wait(15)


class OtherOperations(Scene):
    def construct(self):
        # Insert in middle

        self.wait(3)

        other_ops = TextMobject("Let's take a look at a few other operations", r"$\ldots$")
        other_ops.shift(UP * 3)
        self.play(Write(other_ops))
        self.wait(2)

        array = create_array(8, BLUE, LEFT * 5, 1, 1.5)
        text_array = generate_chars_in_array(array, 'A', GREEN)
        self.play(*[FadeIn(arr) for arr in array])
        self.wait(3)

        self.play(*[FadeIn(char) for char in text_array[:5]])

        self.wait(2)

        new_char = TextMobject("F")
        new_char.shift(UP * 2 + LEFT * 2)
        arrow = Arrow(UP * 1.9 + LEFT * 2, LEFT * 2 + UP * 0.5)

        self.play(Write(new_char))
        self.play(ShowCreation(arrow))

        for i in range(4, 1, -1):
            text_copy = text_array[i].copy()
            text_copy.move_to(array[i+1].get_center())
            self.play(ReplacementTransform(text_array[i], text_copy))
            text_array[i+1] = text_copy

        new_char_copy = new_char.copy()
        new_char_copy.move_to(array[2].get_center())
        new_char_copy.set_color(GREEN)
        self.play(ReplacementTransform(new_char, new_char_copy), FadeOut(arrow))
        text_array[i] = new_char_copy

        insertion = TextMobject("Insertion at specific index: $O(N)$")
        insertion.shift(DOWN * 1.5)
        self.play(Write(insertion))
        self.wait(2)
        self.play(FadeOut(insertion))

        # Removal of elements from the end

        other_chars = [TextMobject('G'), TextMobject('H')]
        all_chars = text_array[:-2]
        for i in range(len(other_chars)):
            other_chars[i].set_color(GREEN)
            other_chars[i].move_to(array[i+6].get_center())
            text_array[i + 6] = other_chars[i]
        self.play(*[FadeIn(char) for char in other_chars])

        for i in range(len(array) - 1, len(array) // 2 - 2, -1):
            char_in_array = text_array[i]
            char_in_array.set_color(RED)
            self.wait()
            
            self.play(FadeOut(char_in_array))
            if i == 4:
                removal = TextMobject("Remove Operation: $O(1)$ amortized and $O(N)$ Worst Case")
                removal.shift(UP)
                self.play(Write(removal))
                self.wait(8)
            text_array.pop()

        self.wait()

        new_array = create_array(4, BLUE, LEFT * 5 + DOWN * 1.5, 1, 1.5)
        self.play(*[GrowFromCenter(arr) for arr in new_array])
        new_text = []

        for i in range(len(text_array)):
            text_copy = text_array[i].copy()
            text_copy.move_to(new_array[i].get_center())
            new_text.append(text_copy)
            self.play(TransformFromCopy(text_array[i], text_copy))

        self.play(*[FadeOut(obj) for obj in array + text_array])


        fadeouts = [FadeOut(obj) for obj in new_array + new_text] + [FadeOut(removal)]
        self.play(*fadeouts)

        
        # Removal of elements from the middle
        final_array = create_array(8, BLUE, LEFT * 5, 1, 1.5)
        final_array_text = generate_chars_in_array(array, 'A', GREEN)
        self.play(*[FadeIn(obj) for obj in final_array + final_array_text])

        remove_index = 2
        char_to_remove = final_array_text[remove_index]
        char_to_remove.set_color(RED)
        self.wait()
        self.play(FadeOut(char_to_remove))

        for i in range(remove_index+1, len(final_array)):
            text_copy = final_array_text[i].copy()
            text_copy.move_to(final_array[i - 1])
            self.play(ReplacementTransform(final_array_text[i], text_copy))

        removal_middle = TextMobject("Removal at specific index: $O(N)$")
        removal_middle.shift(DOWN * 1.5)
        self.play(Write(removal_middle))
        self.wait(2)



class Conclusion(Scene):
    def construct(self):
        key_takeaways = TextMobject("Key Takeways of the Dynamic Array")
        key_takeaways.shift(UP * 2)
        self.play(Write(key_takeaways))
        self.wait(2)

        insertion = TextMobject("Insertion:","$O(1)$", "amortized,", "$O(N)$", "worst case")
        insertion[1].set_color(GREEN)
        insertion[3].set_color(BLUE)
        insertion.shift(UP)
        self.play(Write(insertion))
        self.wait()

        removal = TextMobject("Remove Element:",  "$O(1)$", "amortized, " ,"$O(N)$", "worst case")
        removal[1].set_color(GREEN)
        removal[3].set_color(BLUE)
        self.play(Write(removal))
        self.wait()

        access = TextMobject("Indexing Element:", "$O(1)$")
        access[1].set_color(GREEN)
        access.shift(DOWN)
        self.play(Write(access))
        self.wait(2)
        
        resize_scheme = TextMobject("Best Resizing Scheme: Factor of 2")
        resize_scheme.shift(DOWN * 2)
        self.play(Write(resize_scheme))
        self.wait()

        all_text = [key_takeaways, insertion, removal, access, resize_scheme]

        self.play(*[FadeOut(obj) for obj in all_text])

        summary = TextMobject("Here's what we did")
        summary.shift(UP * 3)

        self.play(Write(summary))

        array = create_array(5, BLUE, LEFT * 2 + UP, 1, 1.5)
        text_array = generate_chars_in_array(array, 'A', GREEN)
        text_array[-1].set_color(WHITE)
        array.pop()

        resize_plus1 = TextMobject("We first tried to resize by adding 1 block")
        resize_plus1.shift(UP * 2)

        self.play(*[FadeIn(obj) for obj in array[:4] + text_array[:4]] + [Write(resize_plus1)])

        self.play(FadeIn(text_array[-1]))

        array2 = create_array(5, BLUE, LEFT * 2 + DOWN * 0.5, 1, 1.5)
        text_array2 = generate_chars_in_array(array2, 'A', GREEN)

        self.play(*[GrowFromCenter(arr) for arr in array2])

        for i in range(len(text_array)):
            if i == len(text_array) - 1:
                self.play(ReplacementTransform(text_array[i], text_array2[i]), run_time=0.2)
            else:
                self.play(TransformFromCopy(text_array[i], text_array2[i]), run_time=0.2)


        group_plus1 = VGroup(array, array2, text_array, text_array2)

        cross = Cross(group_plus1)
        cross.set_color(RED)
        too_slow_plus1 = TextMobject("Too slow!")
        too_slow_plus1.shift(DOWN * 2)
        self.play(ShowCreation(cross), Write(too_slow_plus1))
        self.wait()

        self.play(FadeOut(group_plus1), FadeOut(cross), FadeOut(too_slow_plus1))

        array = create_array(5, BLUE, LEFT * 4 + UP * 0.5, 0.5, 0.8)
        text_array = generate_chars_in_array(array, 'A', GREEN, scale=0.7)
        text_array[-1].set_color(WHITE)
        array.pop()

        resize_plus8 = TextMobject("We then tried to resize by adding 8 blocks")
        resize_plus8.shift(UP * 2)

        self.play(*[FadeIn(obj) for obj in array[:4] + text_array[:4]] + [Transform(resize_plus1, resize_plus8)])

        self.play(FadeIn(text_array[-1]))

        array2 = create_array(12, BLUE, LEFT * 4 + DOWN * 0.5, 0.5, 0.8)
        text_array2 = generate_chars_in_array(array2, 'A', GREEN, scale=0.7)
        text_array2 = text_array2[:5]

        self.play(*[GrowFromCenter(arr) for arr in array2])

        for i in range(len(text_array)):
            if i == len(text_array) - 1:
                self.play(ReplacementTransform(text_array[i], text_array2[i]), run_time=0.2)
            else:
                self.play(TransformFromCopy(text_array[i], text_array2[i]), run_time=0.2)

        self.wait()

        group_plus8 = VGroup(array, array2, text_array, text_array2)

        cross = Cross(group_plus8)
        cross.set_color(RED)
        too_slow_plus8 = TextMobject("Still too slow!")
        too_slow_plus8.shift(DOWN * 2)
        self.play(ShowCreation(cross), Write(too_slow_plus8))
        self.wait()

        self.play(FadeOut(group_plus8), FadeOut(cross), FadeOut(too_slow_plus8))


        array = create_array(5, BLUE, LEFT * 4.5 +  UP * 0.5, 1, 1.5)
        text_array = generate_chars_in_array(array, 'A', GREEN)
        text_array[-1].set_color(WHITE)
        array.pop()

        resize_double = TextMobject("We finally tried resizing by a factor of 2")
        resize_double.shift(UP * 2)

        self.play(*[FadeIn(obj) for obj in array[:4] + text_array[:4]] + [Transform(resize_plus1, resize_double)])

        self.play(FadeIn(text_array[-1]))

        array2 = create_array(8, BLUE, LEFT * 4.5 + DOWN, 1, 1.5)
        text_array2 = generate_chars_in_array(array2, 'A', GREEN)
        text_array2 = text_array2[:5]

        self.play(*[GrowFromCenter(arr) for arr in array2])

        for i in range(len(text_array)):
            if i == len(text_array) - 1:
                self.play(ReplacementTransform(text_array[i], text_array2[i]), run_time=0.2)
            else:
                self.play(TransformFromCopy(text_array[i], text_array2[i]), run_time=0.2)


        line1 = Line(array[2].get_center() + RIGHT * 0.5 + DOWN * 0.7, array2[3].get_center())
        line2 = Line(array2[3].get_center(), array[-1].get_center() + UP * 1.3 + RIGHT * 0.7)
        line1.set_color(GREEN_B)
        line2.set_color(GREEN_B)
        line_group = VGroup(line1, line2)


        efficient = TextMobject("This is efficient!")
        efficient.shift(DOWN * 2)
        self.play(ShowCreation(line_group), Write(efficient))
        self.wait(2)

        group = VGroup(array, text_array, array2, text_array2)

        self.play(FadeOut(line2), FadeOut(line1),
            FadeOut(efficient), FadeOut(group), FadeOut(summary), FadeOut(resize_plus1))

        self.wait()



        
        






class Outro(Scene):
    def construct(self):
        thanks = TextMobject("Thanks for watching this video!")
        thanks.shift(UP)
        self.play(Write(thanks))

        final_msg = TextMobject("If you enjoyed the video, please hit the like button")
        final_msg2 = TextMobject("and subscribe to follow more content like this")
        final_msg2.shift(DOWN)
        self.play(Write(final_msg))
        self.play(Write(final_msg2))
        self.wait(3)
        self.play(FadeOut(thanks), FadeOut(final_msg), FadeOut(final_msg2))




class StartingPoint(Scene):
    def construct(self):

        top_character = create_confused_char(color=BLUE, scale=0.7, position= LEFT * 3)
        top_thought_bubble = SVGMobject("thought")
        top_thought_bubble.scale(1.6)
        top_thought_bubble.move_to(top_character.get_center() + UP * 2.2 + RIGHT * 2.2)
        invention = TextMobject("Inventing a data", "structure seems...hard")
        invention.scale(0.5)
        invention[0].move_to(top_thought_bubble.get_center() + UP * 0.75 + RIGHT * 0.04)
        invention[1].move_to(top_thought_bubble.get_center() + UP * 0.45 + RIGHT * 0.04)
        invention.set_color(BLACK)

        self.play(FadeIn(top_character))
        self.play(FadeIn(top_thought_bubble), FadeIn(invention))

        self.wait()

        bottom_character = create_confused_char(color=GREEN, scale=0.7, position= DOWN * 3 + LEFT * 3)
        bottom_thought_bubble = SVGMobject("thought")
        bottom_thought_bubble.scale(1.3)
        bottom_thought_bubble.move_to(bottom_character.get_center() + UP * 2.2 + RIGHT * 2.2)
        start = TextMobject("Yeah! Where do", "we even start?")
        start.scale(0.5)
        start[0].move_to(bottom_thought_bubble.get_center() + UP * 0.60 + RIGHT * 0.04)
        start[1].move_to(bottom_thought_bubble.get_center() + UP * 0.30 + RIGHT * 0.04)
        start.set_color(BLACK)

        self.play(FadeIn(bottom_character))
        self.play(FadeIn(bottom_thought_bubble), FadeIn(start))

        self.wait()

        right_character = create_computer_char(color=RED, scale=0.7, position= DOWN * 1.5 + RIGHT * 3.5)
        right_thought_bubble = SVGMobject("thought")
        right_thought_bubble.flip()
        right_thought_bubble.scale(1.2)
        right_thought_bubble.move_to(right_character.get_center() + UP * 2 + LEFT * 2)
        simplify = TextMobject("Let's simplify!")
        simplify.scale(0.5)
        simplify.move_to(right_thought_bubble.get_center() + UP * 0.4 + RIGHT * 0.04)
        simplify.set_color(BLACK)

        self.play(FadeIn(right_character))
        self.play(FadeIn(right_thought_bubble), FadeIn(simplify))

        self.wait()

        self.play(FadeOut(top_character), FadeOut(bottom_character), FadeOut(right_character),
            FadeOut(top_thought_bubble), FadeOut(invention), FadeOut(bottom_thought_bubble),
            FadeOut(start), FadeOut(simplify), FadeOut(right_thought_bubble))



        array = create_array(4, BLUE, LEFT * 2, 1, 1.5)
        text_in_array = generate_chars_in_array(array, 'A', GREEN)
        text_in_array_copy = [char.copy() for char in text_in_array]
        for char in text_in_array_copy:
            char.set_color(WHITE)
            char.shift(UP)
        self.play(*[FadeIn(arr) for arr in array])
        self.wait()
        self.play(*[FadeIn(char) for char in text_in_array_copy])
        self.wait()
        self.play(*[ReplacementTransform(char, char_copy) for char, char_copy in zip(text_in_array_copy, 
            text_in_array)])

        self.play(Indicate(text_in_array[2], color=RED), run_time=2)
        self.play(Indicate(text_in_array[0], color=RED), run_time=2)
        self.wait(1.5)
        fadeouts = [FadeOut(obj) for obj in text_in_array + array]
        self.play(*fadeouts)
        self.wait()


class IntroPart2(Scene):
    def construct(self):
        array = create_array(6, BLUE, LEFT * 3.5 , 1, 1.5)
        array_text = generate_chars_in_array(array, '1', PURPLE_A)
        self.play(*[FadeIn(obj) for obj in array[:4] + array_text[:4]])
        self.wait()
        self.play(FadeIn(array_text[4]), GrowFromCenter(array[4]))
        self.play(FadeIn(array_text[5]), GrowFromCenter(array[5]))
        self.wait()
        self.play(*[FadeOut(obj) for obj in array + array_text])
        self.wait()

        left_text = TextMobject("Resizing by 1 element")
        right_text = TextMobject("Resizing by 8 elements")
        left_text.move_to(LEFT * 4 + UP * 3)
        right_text.move_to(RIGHT * 4 + UP * 3)
        self.play(Write(left_text), Write(right_text))

        left_array = create_array(5, BLUE, LEFT * 5 + UP * 2 , 0.4, 0.6)
        left_array_text = generate_chars_in_array(left_array, 'A', GREEN, scale=0.7)
        left_array_text[-1].set_color(WHITE)

        right_array = create_array(5, BLUE, RIGHT * 2.4 + UP * 2 , 0.4, 0.6)
        right_array_text = generate_chars_in_array(right_array, 'A', GREEN, scale=0.7)
        right_array_text[-1].set_color(WHITE)

        

        self.play(*[FadeIn(obj) for obj in left_array[:4] + left_array_text[:4] + right_array[:4] + right_array_text[:4]])

        self.wait()
        self.play(FadeIn(left_array_text[-1]), FadeIn(right_array_text[-1]))
        self.wait()
        left_resized_array = create_array(5, BLUE, LEFT * 5 + UP, 0.4, 0.6)
        left_replaced_text = generate_chars_in_array(left_resized_array, 'A', GREEN, scale=0.7)

        right_resized_array = create_array(12, BLUE, UP, 0.4, 0.6)
        right_replaced_text = generate_chars_in_array(right_resized_array, 'A', GREEN, scale=0.7)

        self.play(*[GrowFromCenter(arr) for arr in left_resized_array + right_resized_array])


        for i in range(len(left_resized_array)):
            self.play(ReplacementTransform(left_array_text[i], left_replaced_text[i]), 
                ReplacementTransform(right_array_text[i], right_replaced_text[i]), run_time=0.5)


        left_runtime, right_runtime = TextMobject("$O(N^2)$"), TextMobject("$O(N^2)$")
        left_runtime.shift(LEFT * 4)
        right_runtime.shift(RIGHT * 3)

        self.play(Write(left_runtime), Write(right_runtime))
        self.wait(5)

        better = TextMobject("Can we do better?")
        better.shift(DOWN * 3)
        self.play(Write(better))

        better_transform = TextMobject("We can actually do much better!")

        self.wait(4)

        all_objects = left_resized_array + right_resized_array + left_replaced_text[:5] + right_replaced_text[:5] + [left_text, right_text] + left_array_text + left_array + right_array_text + right_array + [left_runtime, right_runtime]

        fadeouts = [FadeOut(obj) for obj in all_objects] + [ReplacementTransform(better, better_transform)]
        self.play(*fadeouts, run_time=2)

        self.wait(2)

        self.play(FadeOut(better_transform))


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

class PatreonPreview(Scene):
    def construct(self):
        eq = TexMobject(r"\text{Insertions} ",
                    r"= ",
                    r"(", 
                    r"4 + ",
                    r"5 + ",
                    r"6 + ",
                    r"\ldots", 
                    r"+ N",
                    r")")

        intermediate_objs = []
        curr_array = create_array(4, BLUE, UP + LEFT * 4, 0.5, 0.7)
        chars = generate_chars_in_array(curr_array, 'A', GREEN, scale=0.6)
        intermediate_objs.extend(curr_array + chars)
        chars_above = []
        for char in chars:
            char_copy = char.copy()
            char_copy.shift(UP * 0.5)
            char_copy.set_color(RED)
            chars_above.append(char_copy)


        eq.shift(UP * 3 + LEFT * 2)
        self.play(Write(eq[0:2]))
        for i in range(2, len(eq)):
            self.play(Write(eq[i]), run_time=1)

            if i > 2 and i < 5:
                new_letter = TextMobject(chr(ord('A') + i + 1))
                new_letter.move_to(curr_array[-1].get_center() + curr_array[-1].get_width() * RIGHT)
                new_letter.set_color(RED)
                new_letter.scale(0.6)
                curr_array = create_array(i + 2, BLUE, curr_array[0].get_center() + DOWN, 0.5, 0.7)
                prev_chars = [char.copy() for char in chars] + [new_letter]
                chars = generate_chars_in_array(curr_array, 'A', GREEN, scale=0.6)
                intermediate_objs.extend(chars + curr_array)
                self.play(*[GrowFromCenter(arr) for arr in curr_array])
                self.play(FadeIn(new_letter))
                self.play(*[TransformFromCopy(char_above, char) for char_above, char in zip(prev_chars[:-1], chars[:-1])] + [ReplacementTransform(new_letter, chars[-1])])

            if i == 5:
                self.play(*[FadeOut(obj) for obj in intermediate_objs])

            if i == 2:
                self.play(*[GrowFromCenter(arr) for arr in curr_array])
                self.play(*[FadeIn(char) for char in chars_above])
                self.play(*[ReplacementTransform(char_above, char) for char_above, char in zip(chars_above, chars)])


        eq2 = eq[1:].copy()
        eq2.shift(DOWN * 1)
        self.play(Write(eq2[0]))
        eq2[1:].shift(RIGHT * 2.6)
        eq2[1:-1].set_color(RED)
        self.play(TransformFromCopy(eq[3:], eq2[2:]), run_time=2)
        eq3 = TexMobject(r"(1 + 2 + 3 + ",) # 1 + 2 + 3
        eq3[1:-1].set_color(YELLOW)
        eq3.shift(UP * 2 + 0.5 * RIGHT + 2 * LEFT)
        self.play(Write(eq3), run_time=2)
        eq4 = TexMobject(r" - \ 6")
        eq4.set_color(YELLOW)
        eq4.shift(UP * 2 + RIGHT * 4.8)
        self.play(Write(eq4), run_time=2)

        buff = 0
        left_triangle = []
        width, height = 0.5, 0.5
        for i in range(1, 7):
            array = create_array_left_to_right(i, BLUE, LEFT * 5 + i * height * DOWN, height, width, buff=buff)
            left_triangle.append(array)
            self.play(*[FadeIn(arr) for arr in array], run_time=1.5)

        line = DashedLine(left_triangle[-1][0].get_center() + 0.5 * DOWN + LEFT * height / 2, 
            left_triangle[-1][-1].get_center() + 0.5 * DOWN + RIGHT * height / 2)
        N = TexMobject("N")
        N.next_to(line, DOWN * 0.5)
        self.play(FadeIn(line), FadeIn(N))

        right_triangle = []
        width, height = 0.5, 0.5
        start_point = left_triangle[-1][-1].get_center() + (height + buff) * UP
        for i in range(1, 7):
            array = create_array_right_to_left(i, RED, start_point, height, width, buff=buff)
            right_triangle.append(array)
            start_point = start_point + height * UP
        transforms = []
        for array_l, array_r in zip(left_triangle, right_triangle):
            for rect_l, rect_r in zip(array_l, array_r):
                transforms.append(TransformFromCopy(rect_l, rect_r))
        self.play(*transforms, run_time=4)
        line2 = DashedLine(left_triangle[-1][0].get_center() + 0.5 * LEFT + DOWN * height / 2, 
            right_triangle[-1][-1].get_center() + 0.5 * LEFT + UP * height / 2)
        N_plus = TexMobject("N + 1")
        N_plus.rotate(-TAU / 4)
        N_plus.next_to(line2, LEFT * 0.5)
        self.play(FadeIn(line2), FadeIn(N_plus))

        left_triangle_copy_anims = []
        for array_l in left_triangle:
            for rect_l in array_l:
                rect_l_copy = rect_l.copy()
                rect_l_copy.set_color(GREEN)
                left_triangle_copy_anims.append(ReplacementTransform(rect_l, rect_l_copy))

        self.wait()

        eq2_copy = eq2[2:-1].copy()
        eq3_copy = eq3[1:-1].copy()
        eq2_copy.set_color(GREEN)
        eq3_copy.set_color(GREEN)
        green_animations = [ReplacementTransform(eq2[2:-1], eq2_copy),  ReplacementTransform(eq3[1:-1], eq3_copy)] + left_triangle_copy_anims

        self.play(*green_animations, run_time=4)



        eq5 = TexMobject(r" = \frac{N(N+1)}{2} - 6", r"= \frac{1}{2} N ^ 2 + \frac{1}{2}N - 6")
        eq5.shift(UP + RIGHT * 0.5)
        eq5[0][:-2].set_color(GREEN)
        self.play(Write(eq5[0]), run_time=2)
        self.play(Write(eq5[1]), run_time=2)
        self.wait()




def create_array_left_to_right(num_rectangles, color, start_pos, height, width, buff=0):
    initial_array = [Rectangle(height=height, width=width, color=color, fill_color=color, fill_opacity=0.5) for _ in range(num_rectangles)]
    initial_array[0].move_to(start_pos)
    for i in range(1, len(initial_array)):
        initial_array[i].next_to(initial_array[i - 1], RIGHT, buff=buff)
    return initial_array

def create_array_right_to_left(num_rectangles, color, start_pos, height, width, buff=0):
    initial_array = [Rectangle(height=height, width=width, color=color, fill_color=color, fill_opacity=0.5) for _ in range(num_rectangles)]
    for rect in initial_array:
        rect.set_color(color)
        rect.set_fill(color)
    initial_array[0].move_to(start_pos)
    for i in range(1, len(initial_array)):
        initial_array[i].next_to(initial_array[i - 1], LEFT, buff=buff)
    return initial_array


        









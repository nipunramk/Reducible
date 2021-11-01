from manimlib.imports import *
import random
np.random.seed(0)
import time

class Ball(Circle):
    CONFIG = {
        "radius": 0.15,
        "fill_color": BLUE,
        "fill_opacity": 1,
        "color": BLUE,
        "id": 0, 
    }

    def __init__(self, **kwargs):
        Circle.__init__(self, ** kwargs)
        self.velocity = np.array((2, 0, 0))
        self.mass = PI * self.radius ** 2

    ### Only gives scalar values ###
    def get_top(self):
        return self.get_center()[1] + self.radius

    def get_bottom(self):
        return self.get_center()[1] - self.radius

    def get_right_edge(self):
        return self.get_center()[0] + self.radius

    def get_left_edge(self):
        return self.get_center()[0] - self.radius

    ### Gives full vector ###
    def get_top_v(self):
        return self.get_center() + self.radius * UP

    def get_bottom_v(self):
        return self.get_center() + self.radius * DOWN

    def get_right_edge_v(self):
        return self.get_center() + self.radius * RIGHT

    def get_left_edge_v(self):
        return self.get_center() + self.radius * LEFT

class Box(Rectangle):
    CONFIG = {
        "height": 6,
        "width": FRAME_WIDTH - 1,
        "color": GREEN_C
    }

    def __init__(self, **kwargs):
        Rectangle.__init__(self, ** kwargs)  # Edges

    def get_top(self):
        return self.get_center()[1] + (self.height / 2)

    def get_bottom(self):
        return self.get_center()[1] - (self.height / 2)

    def get_right_edge(self):
        return self.get_center()[0] + (self.width / 2)

    def get_left_edge(self):
        return self.get_center()[0] - (self.width / 2)

class BouncingBall(Scene):
    CONFIG = {
        "bouncing_time": 30,
    }
    def construct(self):
        balls = []
        BOX_THRESHOLD = 0.98
        BALL_THRESHOLD = 0.96
        box = Box()
        colors = [BLUE, YELLOW, GREEN_SCREEN, ORANGE]
        velocities = [UP * i + RIGHT * j for i, j in zip(np.random.uniform(-1, 1, 30), np.random.uniform(-1, 1, 30))]
        positions = []
        for i in np.arange(-4.5, 5.5, 1):
            for j in np.arange(-2, 3, 2):
                positions.append(RIGHT * i + UP * j)
        for i in range(len(positions)):
            ball = Ball(
                color=colors[i % len(colors)], fill_color=colors[i % len(colors)], opacity=1
            )
            ball.id = i
            ball.move_to(positions[i])
            ball.velocity = velocities[i]
            balls.append(ball)
        
        self.play(
            FadeIn(box)
        )
        self.play(
            *[FadeIn(ball) for ball in balls]
        )

        def update_ball(ball, dt):
            ball.acceleration = np.array((0, 0, 0))
            ball.velocity = ball.velocity + ball.acceleration * dt
            ball.shift(ball.velocity * dt)
            handle_collision_with_box(ball, box)
            handle_ball_collisions(ball)

        def handle_collision_with_box(ball, box):
            # Bounce off ground and roof
            if ball.get_bottom() <= box.get_bottom()*BOX_THRESHOLD or \
                    ball.get_top() >= box.get_top()*BOX_THRESHOLD:
                    ball.velocity[1] = -ball.velocity[1]
            # Bounce off walls
            if ball.get_left_edge() <= box.get_left_edge() or \
                    ball.get_right_edge() >= box.get_right_edge():
                ball.velocity[0] = -ball.velocity[0]

        def handle_ball_collisions(ball):
            t_colors = [RED, ORANGE, GREEN_SCREEN, GOLD, PINK, WHITE]
            i = 0
            for other_ball in balls:
                if ball.id != other_ball.id:
                    dist = np.linalg.norm(ball.get_center() - other_ball.get_center())
                    if dist * BALL_THRESHOLD <= (ball.radius + other_ball.radius):
                        v1, v2 = get_response_velocities(ball, other_ball)
                        ball.velocity = v1
                        other_ball.velocity = v2
        
        def get_response_velocities(ball, other_ball):
            # https://en.wikipedia.org/wiki/Elastic_collision
            v1 = ball.velocity
            v2 = other_ball.velocity
            m1 = ball.mass
            m2 = other_ball.mass
            x1 = ball.get_center()
            x2 = other_ball.get_center()

            ball_response_v = compute_velocity(v1, v2, m1, m2, x1, x2)
            other_ball_response_v = compute_velocity(v2, v1, m2, m1, x2, x1)
            return ball_response_v, other_ball_response_v

        def compute_velocity(v1, v2, m1, m2, x1, x2):
            return v1 - (2 * m2 / (m1 + m2)) * np.dot(v1 - v2, x1 - x2) / np.linalg.norm(x1 - x2) ** 2 * (x1 - x2)

        for ball in balls:
            ball.add_updater(update_ball)
            self.add(ball)

        self.wait(self.bouncing_time)
        for ball in balls:
            ball.clear_updaters()
        self.wait(3)

class ParticleSimulation(Scene):
    CONFIG = {
        "simulation_time": 10,
    }
    def construct(self):
        start_time = time.time()
        particles = []
        num_particles = 100
        BOX_THRESHOLD = 0.98
        PARTICLE_THRESHOLD = 0.96
        box = Box(height=5.5, width=5.5)
        velocities = [UP * i + RIGHT * j for i, j in zip(np.random.uniform(-0.2, 0.2, num_particles), np.random.uniform(-0.2, 0.2, num_particles))]
        positions = []
        for i in np.arange(-2.5, 2.5, 0.5):
            for j in np.arange(-2.5, 2.5, 0.5):
                positions.append(RIGHT * i + UP * j)
        for i in range(len(positions)):
            particle = Ball(radius=0.04)
            particle.id = i
            particle.move_to(positions[i])
            particle.velocity = velocities[i]
            particles.append(particle)
        
        self.play(
            FadeIn(box)
        )
        self.play(
            *[FadeIn(particle) for particle in particles]
        )

        def update_particle(particle, dt):
            particle.acceleration = np.array((0, 0, 0))
            particle.velocity = particle.velocity + particle.acceleration * dt
            particle.shift(particle.velocity * dt)
            handle_collision_with_box(particle, box)
            handle_particle_collisions(particle)

        def handle_collision_with_box(particle, box):
            # Bounce off ground and roof
            if particle.get_bottom() <= box.get_bottom()*BOX_THRESHOLD or \
                    particle.get_top() >= box.get_top()*BOX_THRESHOLD:
                    particle.velocity[1] = -particle.velocity[1]
            # Bounce off walls
            if particle.get_left_edge() <= box.get_left_edge() or \
                    particle.get_right_edge() >= box.get_right_edge():
                particle.velocity[0] = -particle.velocity[0]

        def handle_particle_collisions(particle):
            t_colors = [RED, ORANGE, GREEN_SCREEN, GOLD, PINK, WHITE]
            i = 0
            for other_particle in particles:
                if particle.id != other_particle.id:
                    dist = np.linalg.norm(particle.get_center() - other_particle.get_center())
                    if dist * PARTICLE_THRESHOLD <= (particle.radius + other_particle.radius):
                        # ball.set_color(random.choice(t_colors))
                        # other_ball.set_color(random.choice(t_colors))
                        v1, v2 = get_response_velocities(particle, other_particle)
                        particle.velocity = v1
                        other_particle.velocity = v2
        
        def get_response_velocities(particle, other_particle):
            # https://en.wikipedia.org/wiki/Elastic_collision
            v1 = particle.velocity
            v2 = other_particle.velocity
            m1 = particle.mass
            m2 = other_particle.mass
            x1 = particle.get_center()
            x2 = other_particle.get_center()

            particle_response_v = compute_velocity(v1, v2, m1, m2, x1, x2)
            other_particle_response_v = compute_velocity(v2, v1, m2, m1, x2, x1)
            return particle_response_v, other_particle_response_v

        def compute_velocity(v1, v2, m1, m2, x1, x2):
            return v1 - (2 * m2 / (m1 + m2)) * np.dot(v1 - v2, x1 - x2) / np.linalg.norm(x1 - x2) ** 2 * (x1 - x2)

        for particle in particles:
            particle.add_updater(update_particle)
            self.add(particle)

        self.wait(self.simulation_time)
        for particle in particles:
            particle.clear_updaters()
        self.wait(3)
        print("--- %s seconds ---" % (time.time() - start_time))

class ParticleSimulationOptimized(Scene):
    CONFIG = {
        "simulation_time": 70,
    }
    def construct(self):
        start_time = time.time()
        particles = []
        num_particles = 100
        BOX_THRESHOLD = 0.98
        PARTICLE_THRESHOLD = 0.96
        box = Box(height=5.5, width=5.5).set_color(COBALT_BLUE)
        shift_right = RIGHT * 4
        box.shift(shift_right)
        velocities = [UP * i + RIGHT * j for i, j in zip(np.random.uniform(-0.3, 0.3, num_particles), np.random.uniform(-0.3, 0.3, num_particles))]
        positions = []
        start = -2.5
        end = 3.1
        step = (end - start) / np.sqrt(num_particles)
        colors = [BLUE, PINK, GREEN_SCREEN, RED, ORANGE, YELLOW]
        radius =[0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
        for i in np.arange(start, end, step):
            for j in np.arange(start, end, step):
                positions.append(RIGHT * i + UP * j)
        for i in range(len(positions)):
            particle = Ball(radius=radius[i % len(radius)])
            particle.set_color(color=colors[i % len(radius)])
            particle.id = i
            particle.move_to(positions[i])
            particle.shift(shift_right)
            particle.velocity = velocities[i]
            particles.append(particle)
        
        self.play(
            FadeIn(box)
        )

        def update_particles(particles, dt):
            for i in range(len(particles)):
                particle = particles[i]
                particle.acceleration = np.array((0, 0, 0))
                particle.velocity = particle.velocity + particle.acceleration * dt
                particle.shift(particle.velocity * dt)
                handle_collision_with_box(particle, box, dt)
            
            handle_particle_collisions_opt(particles, dt)

        def handle_collision_with_box(particle, box, dt):
            # Bounce off ground and roof
            if particle.get_bottom() <= box.get_bottom()*BOX_THRESHOLD or \
                    particle.get_top() >= box.get_top()*BOX_THRESHOLD:
                particle.velocity[1] = -particle.velocity[1]
                particle.shift(particle.velocity * dt)
            # Bounce off walls
            if particle.get_left_edge() <= box.get_left_edge() * BOX_THRESHOLD or \
                    particle.get_right_edge() >= box.get_right_edge() * BOX_THRESHOLD :
                particle.velocity[0] = -particle.velocity[0]
                particle.shift(particle.velocity * dt)

        def handle_particle_collisions(particles):
            for particle in particles:
                for other_particle in particles:
                    if particle.id != other_particle.id:
                        dist = np.linalg.norm(particle.get_center() - other_particle.get_center())
                        if dist * PARTICLE_THRESHOLD <= (particle.radius + other_particle.radius):
                            # ball.set_color(random.choice(t_colors))
                            # other_ball.set_color(random.choice(t_colors))
                            v1, v2 = get_response_velocities(particle, other_particle)
                            particle.velocity = v1
                            other_particle.velocity = v2

        def handle_particle_collisions_opt(particles, dt):
            possible_collisions = find_possible_collisions(particles) 
            for particle, other_particle in possible_collisions:
                if particle.id != other_particle.id:
                    dist = np.linalg.norm(particle.get_center() - other_particle.get_center())
                    if dist * PARTICLE_THRESHOLD <= (particle.radius + other_particle.radius):
                        # ball.set_color(random.choice(t_colors))
                        # other_ball.set_color(random.choice(t_colors))
                        v1, v2 = get_response_velocities(particle, other_particle)
                        particle.velocity = v1
                        other_particle.velocity = v2
                        particle.shift(particle.velocity * dt)
                        other_particle.shift(other_particle.velocity * dt)
        
        def find_possible_collisions(particles):
            # implements the sort and sweep algorithm for broad phase
            # helpful reference: https://github.com/mattleibow/jitterphysics/wiki/Sweep-and-Prune
            axis_list = sorted(particles, key=lambda x: x.get_left()[0])
            active_list = []
            possible_collisions = set()
            for particle in axis_list:
                to_remove = [p for p in active_list if particle.get_left()[0] > p.get_right()[0]]
                for r in to_remove:
                    active_list.remove(r)
                for other_particle in active_list:
                    possible_collisions.add((particle, other_particle))

                active_list.append(particle)
            
            return possible_collisions

        def get_response_velocities(particle, other_particle):
            # https://en.wikipedia.org/wiki/Elastic_collision
            v1 = particle.velocity
            v2 = other_particle.velocity
            m1 = particle.mass
            m2 = other_particle.mass
            x1 = particle.get_center()
            x2 = other_particle.get_center()

            particle_response_v = compute_velocity(v1, v2, m1, m2, x1, x2)
            other_particle_response_v = compute_velocity(v2, v1, m2, m1, x2, x1)
            return particle_response_v, other_particle_response_v

        def compute_velocity(v1, v2, m1, m2, x1, x2):
            return v1 - (2 * m2 / (m1 + m2)) * np.dot(v1 - v2, x1 - x2) / np.linalg.norm(x1 - x2) ** 2 * (x1 - x2)

        particles = VGroup(*particles)
        self.play(
            FadeIn(particles)
        )
        particles.add_updater(update_particles)
        self.add(particles)
        self.wait(self.simulation_time)
        particles.clear_updaters()
        self.wait(3)
        print("--- %s seconds ---" % (time.time() - start_time))

class LargeParticleSimulationOptimized250(Scene):
    CONFIG = {
        "simulation_time": 120,
    }
    def construct(self):
        start_time = time.time()
        particles = []
        num_particles = 256 # make sure this is perfect square
        BOX_THRESHOLD = 0.98
        PARTICLE_THRESHOLD = 0.96
        box = Box(height=FRAME_HEIGHT - 0.5, color=COBALT_BLUE)
        velocities = [UP * i + RIGHT * j for i, j in zip(np.random.uniform(-0.3, 0.3, num_particles), np.random.uniform(-0.3, 0.3, num_particles))]
        positions = []
        start_x = -6.2
        end_x = 6.5
        start_y = -3.5
        end_y = 3.8
        step_x = (end_x - start_x) / np.sqrt(num_particles)
        step_y = (end_y - start_y) / np.sqrt(num_particles)
        for i in np.arange(start_x, end_x, step_x):
            for j in np.arange(start_y, end_y, step_y):
                positions.append(RIGHT * i + UP * j)
        for i in range(len(positions)):
            if i % 3 == 0:
                color = LIGHT_VIOLET
                radius = 0.08
            elif i % 3 == 1:
                color = SEAFOAM_GREEN
                radius = 0.07
            else:
                color = PERIWINKLE_BLUE
                radius = 0.06
            particle = Ball(radius=radius, color=color)
            particle.set_fill(color=color, opacity=1)
            particle.id = i
            particle.move_to(positions[i])
            particle.velocity = velocities[i]
            particles.append(particle)
        
        self.play(
            FadeIn(box)
        )

        def update_particles(particles, dt):
            for i in range(len(particles)):
                particle = particles[i]
                particle.acceleration = np.array((0, 0, 0))
                particle.velocity = particle.velocity + particle.acceleration * dt
                particle.shift(particle.velocity * dt)
                handle_collision_with_box(particle, box, dt)
            
            handle_particle_collisions_opt(particles, dt)

        def handle_collision_with_box(particle, box, dt):
            # Bounce off ground and roof
            if particle.get_bottom() <= box.get_bottom()*BOX_THRESHOLD or \
                    particle.get_top() >= box.get_top()*BOX_THRESHOLD:
                particle.velocity[1] = -particle.velocity[1]
                particle.shift(particle.velocity * dt) 
            # Bounce off walls
            if particle.get_left_edge() <= box.get_left_edge() or \
                    particle.get_right_edge() >= box.get_right_edge():
                particle.velocity[0] = -particle.velocity[0]
                particle.shift(particle.velocity * dt)

        def handle_particle_collisions(particles):
            for particle in particles:
                for other_particle in particles:
                    if particle.id != other_particle.id:
                        dist = np.linalg.norm(particle.get_center() - other_particle.get_center())
                        if dist * PARTICLE_THRESHOLD <= (particle.radius + other_particle.radius):
                            # ball.set_color(random.choice(t_colors))
                            # other_ball.set_color(random.choice(t_colors))
                            v1, v2 = get_response_velocities(particle, other_particle)
                            particle.velocity = v1
                            other_particle.velocity = v2
                            

        def handle_particle_collisions_opt(particles, dt):
            possible_collisions = find_possible_collisions(particles) 
            for particle, other_particle in possible_collisions:
                if particle.id != other_particle.id:
                    dist = np.linalg.norm(particle.get_center() - other_particle.get_center())
                    if dist * PARTICLE_THRESHOLD <= (particle.radius + other_particle.radius):
                        v1, v2 = get_response_velocities(particle, other_particle)
                        particle.velocity = v1
                        other_particle.velocity = v2
                        particle.shift(particle.velocity * dt)
                        other_particle.shift(other_particle.velocity * dt)
        
        def find_possible_collisions(particles):
            # implements the sort and sweep algorithm for broad phase
            # helpful reference: https://github.com/mattleibow/jitterphysics/wiki/Sweep-and-Prune
            axis_list = sorted(particles, key=lambda x: x.get_left()[0])
            active_list = []
            possible_collisions = set()
            for particle in axis_list:
                to_remove = [p for p in active_list if particle.get_left()[0] > p.get_right()[0]]
                for r in to_remove:
                    active_list.remove(r)
                for other_particle in active_list:
                    possible_collisions.add((particle, other_particle))

                active_list.append(particle)
            
            return possible_collisions

        def get_response_velocities(particle, other_particle):
            # https://en.wikipedia.org/wiki/Elastic_collision
            v1 = particle.velocity
            v2 = other_particle.velocity
            m1 = particle.mass
            m2 = other_particle.mass
            x1 = particle.get_center()
            x2 = other_particle.get_center()

            particle_response_v = compute_velocity(v1, v2, m1, m2, x1, x2)
            other_particle_response_v = compute_velocity(v2, v1, m2, m1, x2, x1)
            return particle_response_v, other_particle_response_v

        def compute_velocity(v1, v2, m1, m2, x1, x2):
            return v1 - (2 * m2 / (m1 + m2)) * np.dot(v1 - v2, x1 - x2) / np.linalg.norm(x1 - x2) ** 2 * (x1 - x2)

        particles = VGroup(*particles)
        self.play(
            FadeIn(particles)
        )
        particles.add_updater(update_particles)
        self.add(particles)
        self.wait(self.simulation_time)
        particles.clear_updaters()
        self.wait(3)
        print("--- %s seconds ---" % (time.time() - start_time))

class Transition(Scene):
    def construct(self):
        self.wait()
        title = TextMobject("Computer Graphics").scale(1.4).move_to(UP * 3)

        self.play(
            Write(title)
        )

        self.wait()

        screen_rect = ScreenRectangle(height=4.5)
        screen_rect.move_to(LEFT * 2.5)

        self.play(
            ShowCreation(screen_rect)
        )

        self.wait()

        topics = BulletedList(
            "Animation",
            "Particle Dynamics",
            "Collision Detection",
            "Collision Response",
            "Spatial Partitioning",
            "Object Partitioning",
        ).scale(0.9)

        topics.next_to(screen_rect, RIGHT).shift(RIGHT * 0.5)
        for i in range(len(topics)):
            self.play(
                Write(topics[i])
            )
            self.wait()

class AnimationIntro(Scene):
    def construct(self):

        self.wait()

        point_A = Dot()
        point_B = Dot()

        point_A.move_to(LEFT * 3)
        point_B.move_to(RIGHT * 3)

        a = TexMobject("A").next_to(point_A, LEFT)
        b = TexMobject("B").next_to(point_B, RIGHT)

        arrow = Arrow(LEFT * 2.9, RIGHT * 2.9)

        self.play(
            Write(point_A),
            Write(a),
        )

        self.wait()

        self.play(
            ShowCreation(arrow)
        )

        self.play(
            Write(point_B),
            Write(b),
        )

        self.wait()

        circle = Circle(
            color=BLUE, fill_color=BLUE, fill_opacity=1, radius=0.5
        )
        start, end = LEFT * 3 + DOWN, RIGHT * 3 + DOWN

        circle.move_to(start)

        self.add(circle)

        self.wait()

        frame_circle = circle.copy()
        frame_circle.set_stroke(opacity=0.1)
        frame_circle.set_fill(opacity=0.1)

        frames = []

        FPS = self.camera.frame_rate
        print(FPS)
        interp = smooth

        for i in range(FPS + 1):
            t = 1 / FPS * i
            position = (1 - interp(t)) * start + interp(t) * end
            position = position + DOWN * 1.5 # shift all in between frames down
            frames.append(frame_circle.copy().move_to(position))

        # 15 FPS case
        frames_15 = []
        start_15 = LEFT * 3 + UP
        end_15 = RIGHT * 3 + UP
        for i in range(16):
            t = 1 / 15 * i
            position = (1 - interp(t)) * start_15 + interp(t) * end_15
            position = position + UP * 1.5 # shift all in between frames up
            frames_15.append(frame_circle.copy().move_to(position))

        self.play(
            circle.move_to, end,
            rate_func=interp
        )

        self.wait()

        self.remove(circle)

        self.wait()

        circle.move_to(start)
        self.add(circle)
        self.wait()

        self.play(
            circle.move_to, end,
            rate_func=interp
        )

        self.wait()

        self.remove(circle)

        self.wait()

        circle.move_to(start)
        self.add(circle)
        self.wait()

        self.play(
            circle.move_to, end,
            rate_func=interp,
        )
        self.wait()

        self.remove(circle)

        self.wait()

        circle.move_to(start)
        self.add(circle)
        self.wait()

        self.play(
            circle.move_to, end,
            AnimationGroup(
                *[FadeIn(f) for f in frames],
                lag_ratio=0.01
            ),
            AnimationGroup(
                *[FadeIn(f) for f in frames_15],
                lag_ratio=0.05
            ),
            rate_func=interp,
        )

        self.wait()

        frames = VGroup(*frames)
        frames_15 = VGroup(*frames_15)

        top_15 = TextMobject("15 Frames Per Second (FPS)")
        top_15.next_to(frames_15, UP)

        self.play(
            Write(top_15)
        )

        self.wait()

        fps = TextMobject("60 FPS")
        fps.next_to(frames, DOWN)
        self.play(
            Write(fps)
        )
        self.wait()

        self.remove(circle)

        self.wait()

        circle.move_to(start)
        self.add(circle)
        self.wait()

        self.play(
            circle.move_to, end,
            rate_func=interp,
        )
        self.wait()

class FrameByFrame(Scene):
    def construct(self):
        frame = TextMobject("Key Idea: Frame by Frame Perspective")
        frame.scale(1.2).move_to(UP * 3.3)
        self.play(
            Write(frame),
            run_time=2
        )

        self.wait()

class FPS15Anim(Scene):
    def construct(self):
        circle = Circle(
            color=BLUE, fill_color=BLUE, fill_opacity=1, radius=0.5
        )
        start, end = LEFT * 3 + UP, RIGHT * 3 + UP
        interp = smooth
        circle.move_to(start)

        self.wait(6)

        self.add(circle)

        self.wait()

        self.play(
            circle.move_to, end,
            rate_func=interp
        )

        self.wait()

        self.remove(circle)

        self.wait()

        circle.move_to(start)
        self.add(circle)
        self.wait()

        self.play(
            circle.move_to, end,
            rate_func=interp
        )

        self.wait()

        self.remove(circle)

        self.wait()

        circle.move_to(start)
        self.add(circle)
        self.wait()

        self.play(
            circle.move_to, end,
            rate_func=interp,
        )
        self.wait()

        self.remove(circle)

        self.wait()

        circle.move_to(start)
        self.add(circle)
        self.wait()

        self.play(
            circle.move_to, end,
        )

        self.wait()

class ParticleInBox(Scene):
    CONFIG = {
        "bouncing_time": 10,
    }
    def construct(self):
        box = Box(height=5.5, width=5.5).set_color(RED)
        box.shift(UP * 1)
        ball = Ball(radius=0.2).shift(RIGHT * 3 + UP * 1)
        ball.set_color(SEAFOAM_GREEN)
        ball.velocity = np.array([2, 0, 0])
        box.shift(RIGHT * 3)
        box.set_color(COBALT_BLUE)
        self.play(
            FadeIn(box)
        )
        self.play(
            FadeIn(ball)
        )
        BOX_THRESHOLD = 0.98

        def update_ball(ball, dt):
            ball.acceleration = np.array((0, -5, 0))
            ball.velocity = ball.velocity + ball.acceleration * dt
            ball.shift(ball.velocity * dt)  
            # Bounce off ground and roof
            if ball.get_bottom() <= box.get_bottom() * BOX_THRESHOLD or \
                    ball.get_top() >= box.get_top() * BOX_THRESHOLD:
                ball.velocity[1] = -ball.velocity[1]
            # Bounce off walls
            if ball.get_left_edge() <= box.get_left_edge() or \
                    ball.get_right_edge() >= box.get_right_edge():
                ball.velocity[0] = -ball.velocity[0]

        ball.add_updater(update_ball)
        self.add(ball)
        self.wait(self.bouncing_time)
        ball.clear_updaters()
        self.wait()

        p_color, v_color, a_color = YELLOW, BRIGHT_ORANGE, BLUE

        accel_vec = Arrow(ball.get_bottom_v(), ball.get_bottom_v() + DOWN * 1.7)
        accel_vec.set_color(a_color)

        start_vel = ball.get_center() + ball.radius * (ball.velocity / np.linalg.norm(ball.velocity))
        end_vel = start_vel + (ball.velocity / np.linalg.norm(ball.velocity)) * np.linalg.norm(ball.velocity) / np.linalg.norm(ball.acceleration)
        vel_vec = Arrow(start_vel, end_vel).set_color(v_color).scale(2)
        vel_vec.shift(SMALL_BUFF * 2 * (end_vel - start_vel) / np.linalg.norm(start_vel - end_vel))

        position = Dot().move_to(ball.get_center()).set_color(p_color)

        self.play(
            Write(accel_vec),
            Write(vel_vec),
            GrowFromCenter(position)
        )

        self.wait()

        left, comma, right = TexMobject("("), TexMobject(","), TexMobject(")")
        
        label_scale = 0.6

        left.scale(label_scale)
        comma.scale(label_scale)
        right.scale(label_scale)

        v_x = DecimalNumber(ball.velocity[0]).scale(label_scale)
        v_y = DecimalNumber(ball.velocity[1]).scale(label_scale)

        v_label = VGroup(left, v_x, comma, v_y, right).arrange_submobjects(RIGHT, buff=SMALL_BUFF*0.6).set_color(v_color)
        v_label[0].shift(RIGHT * SMALL_BUFF * 0.2)
        v_label[2].shift(DOWN * SMALL_BUFF)
        v_label[4].shift(LEFT * SMALL_BUFF * 0.2)
        v_label.next_to(vel_vec, SMALL_BUFF * ball.velocity / np.linalg.norm(ball.velocity))
        
        a_label = TexMobject("(0.00, -5.00)").scale(label_scale)
        a_label.set_color(a_color).next_to(accel_vec, DOWN * SMALL_BUFF)

        p_x = DecimalNumber(ball.get_center()[0]).scale(label_scale)
        p_y = DecimalNumber(ball.get_center()[1]).scale(label_scale)

        p_label = VGroup(left.copy(), p_x, comma.copy(), p_y, right.copy()).arrange_submobjects(RIGHT, buff=SMALL_BUFF*0.6).set_color(p_color)
        p_label[0].shift(RIGHT * SMALL_BUFF * 0.2)
        p_label[2].shift(DOWN * SMALL_BUFF)
        p_label[4].shift(LEFT * SMALL_BUFF * 0.2)
        p_label.next_to(ball, UP * 7)

        self.play(
            Write(v_label),
            Write(a_label),
            Write(p_label)
        )

        self.wait()

        def update_ball_with_arrows(ball, dt):
            ball.acceleration = np.array((0, -5, 0))
            ball.velocity = ball.velocity + ball.acceleration * dt
            ball.shift(ball.velocity * dt)  # Bounce off ground and roof
            if ball.get_bottom() <= box.get_bottom() * BOX_THRESHOLD or \
                    ball.get_top() >= box.get_top() * BOX_THRESHOLD:
                ball.velocity[1] = -ball.velocity[1]
            # Bounce off walls
            if ball.get_left_edge() <= box.get_left_edge() * BOX_THRESHOLD or \
                    ball.get_right_edge() >= box.get_right_edge() * BOX_THRESHOLD:
                ball.velocity[0] = -ball.velocity[0]

            accel_vec.next_to(ball, DOWN, buff=SMALL_BUFF)
            a_label.next_to(accel_vec, DOWN * SMALL_BUFF)
            
            p_label[1].set_value(ball.get_center()[0])
            p_label[0].next_to(p_label[1], LEFT, buff=SMALL_BUFF*0.2)
            p_label[2].next_to(p_label[1], RIGHT, buff=SMALL_BUFF*0.2)
            p_label[3].set_value(ball.get_center()[1])
            p_label[3].next_to(p_label[2], RIGHT, buff=SMALL_BUFF*0.5)
            p_label[4].next_to(p_label[3], RIGHT, buff=SMALL_BUFF*0.2)
            p_label[2].shift(DOWN * SMALL_BUFF)

            p_label.next_to(ball, UP * 7)

            position.move_to(ball.get_center())

            start_vel = ball.get_center() + ball.radius * (ball.velocity / np.linalg.norm(ball.velocity))
            end_vel = start_vel + (ball.velocity / np.linalg.norm(ball.velocity)) * np.linalg.norm(ball.velocity) / np.linalg.norm(ball.acceleration)
            new_vec = Arrow(start_vel, end_vel).set_color(v_color).scale(2)
            new_vec.shift(SMALL_BUFF * 2 * (end_vel - start_vel) / np.linalg.norm(start_vel - end_vel))
            vel_vec.become(new_vec)

            v_label[1].set_value(ball.velocity[0])
            v_label[0].next_to(v_label[1], LEFT, buff=SMALL_BUFF*0.2)
            v_label[2].next_to(v_label[1], RIGHT, buff=SMALL_BUFF*0.2)
            
            v_label[3].set_value(ball.velocity[1])
            v_label[3].next_to(v_label[2], RIGHT, buff=SMALL_BUFF*0.5)
            v_label[4].next_to(v_label[3], RIGHT, buff=SMALL_BUFF*0.2)
            v_label[2].shift(DOWN * SMALL_BUFF)

            v_label.next_to(vel_vec, SMALL_BUFF * ball.velocity / np.linalg.norm(ball.velocity))

        for i in range(6):
            update_ball_with_arrows(ball, 1 / self.camera.frame_rate)
            self.wait()

        ball.add_updater(update_ball_with_arrows)
        self.add(ball)
        self.add(position)
        self.wait(3.9)
        ball.clear_updaters()
        self.wait()

        for i in range(4):
            update_ball_with_arrows(ball, 1 / self.camera.frame_rate)
            self.wait()

        for i in range(6):
            update_ball_with_arrows(ball, 1 / self.camera.frame_rate)
            self.wait(0.2)

        for i in range(3):
            update_ball_with_arrows(ball, 1 / self.camera.frame_rate)
            self.wait()

class ParticleInBoxDescription(Scene):
    def construct(self):
        indent = LEFT * 2
        dynamics = TextMobject("Particle Dynamics")
        dynamics.to_edge(indent)
        dynamics.shift(UP * 3.5)
        self.play(
            Write(dynamics)
        )

        self.wait()

        position = TextMobject(r"Position: ", "$p = (p_x, p_y)$")
        velocity = TextMobject(r"Velocity: ", "$v = (v_x, v_y)$")
        acceleration = TextMobject(r"Acceleration: ", "$a = (a_x, a_y)$")

        p_color, v_color, a_color = YELLOW, BRIGHT_ORANGE, BLUE
        dt_color = GREEN_SCREEN
        position[1].set_color(p_color)
        velocity[1].set_color(v_color)
        acceleration[1].set_color(a_color)

        position.scale(0.8)
        velocity.scale(0.8)
        acceleration.scale(0.8)

        position.next_to(dynamics, DOWN).to_edge(indent)
        velocity.next_to(position, DOWN).to_edge(indent)
        acceleration.next_to(velocity, DOWN).to_edge(indent)

        self.play(
            Write(position)
        )

        self.wait()

        self.play(
            Write(velocity)
        )

        self.wait()

        self.play(
            Write(acceleration)
        )

        self.wait()

        h_line = Line(LEFT * 7.5, LEFT * 2.5)
        h_line.next_to(acceleration, DOWN)

        self.play(
            Write(h_line)
        )

        constant_a = TextMobject(r"$a$ is constant").scale(0.8)
        constant_a.next_to(h_line, DOWN).to_edge(indent)
        constant_a[0][0].set_color(a_color)
        self.play(
            Write(constant_a)
        )

        self.wait()

        change = TextMobject(r"Frame $f \rightarrow f + 1$")
        change.scale(0.8)
        change.next_to(constant_a, DOWN).to_edge(indent)

        self.play(
            Write(change)
        )

        self.wait()

        delta_t = TexMobject(r"\Delta t = \frac{1}{\text{FPS}}").scale(0.8)
        delta_t.next_to(change, DOWN).to_edge(indent)
        delta_t[0][:2].set_color(dt_color)

        self.play(
            Write(delta_t)
        )
        self.wait()

        initial = TextMobject(r"Define $a$, $v_0$, and $p_0$").scale(0.8)
        initial.next_to(delta_t, DOWN).to_edge(indent)
        initial[0][6].set_color(a_color)
        initial[0][8:10].set_color(v_color)
        initial[0][-2:].set_color(p_color)

        self.play(
            Write(initial)
        )

        self.wait()


        velocity_update = TexMobject(r"v_{f + 1} = v_f + a \Delta t")
        velocity_update.scale(0.8)
        velocity_update.next_to(initial, DOWN).to_edge(indent)
        velocity_update[0][:4].set_color(v_color)
        velocity_update[0][5:7].set_color(v_color)
        velocity_update[0][8].set_color(a_color)
        velocity_update[0][-2:].set_color(dt_color)

        self.play(
            Write(velocity_update)
        )

        position_update = TexMobject(r"p_{f + 1} = p_f + v_f \Delta t")
        position_update.scale(0.8)
        position_update.next_to(velocity_update, DOWN).to_edge(indent)
        position_update[0][:4].set_color(p_color)
        position_update[0][5:7].set_color(p_color)
        position_update[0][8:10].set_color(v_color)
        position_update[0][-2:].set_color(dt_color)

        self.play(
            Write(position_update)
        )

        self.wait()

        collision_detection = TextMobject("Collision Detection")
        collision_detection.next_to(h_line, DOWN)
        collision_detection.to_edge(LEFT * 2)

        self.play(
            ReplacementTransform(constant_a, collision_detection),
            FadeOut(change),
            FadeOut(initial),
            FadeOut(delta_t),
            velocity_update.shift, UP * 2.2,
            position_update.shift, UP * 2.2,
            run_time=2
        )

        self.wait()

        self.play(
            FadeOut(acceleration),
            FadeOut(position),
            velocity.shift, UP * 0.7,
            h_line.shift, UP * 1.4,
            collision_detection.shift, UP * 1.4,
            velocity_update.shift, UP * 1.4,
            position_update.shift, UP * 1.4,
            run_time=2
        )

        comparisons = []

        particle = Circle(radius=0.5).set_color(SEAFOAM_GREEN)
        particle.set_fill(color=SEAFOAM_GREEN, opacity=1)
        particle.next_to(position_update, DOWN)
        particle.shift(LEFT * 1)

        line = Line(ORIGIN, UP * 1.2).set_color(COBALT_BLUE)
        line.next_to(particle, LEFT, buff=0)
        line.set_stroke(width=6)
        particle.shift(LEFT * SMALL_BUFF)

        key_point = Dot().scale(0.8).set_color(FUCHSIA)
        key_point.move_to(particle.get_left() + LEFT * SMALL_BUFF * 0.2)

        left_comparison = VGroup(particle, line, key_point)
        comparisons.append(left_comparison)

        self.play(
            GrowFromCenter(particle),
        )

        self.play(
            ShowCreation(line)
        )

        self.play(
            GrowFromCenter(key_point)
        )

        self.wait()

        right_comparison = left_comparison.copy()
        right_comparison.shift(RIGHT * 2)

        right_comparison[1].next_to(right_comparison[0], RIGHT, buff=0)
        right_comparison[0].shift(RIGHT * SMALL_BUFF)
        right_comparison[2].move_to(right_comparison[0].get_right() + RIGHT * SMALL_BUFF * 0.2)

        self.play(
            GrowFromCenter(right_comparison[0])
        )

        self.play(
            ShowCreation(right_comparison[1])
        )

        self.play(
            GrowFromCenter(right_comparison[2])
        )

        self.wait()

        top_comparison = right_comparison.copy()
        top_comparison.shift(DOWN * 2)
        top_comparison[1].rotate(PI / 2)
        top_comparison[1].next_to(top_comparison[0], UP, buff=0)
        top_comparison[1].shift(DOWN * SMALL_BUFF)
        top_comparison[2].move_to(top_comparison[0].get_top() + UP * SMALL_BUFF * 0.2)

        bottom_comparison = left_comparison.copy()
        bottom_comparison.shift(DOWN * 2)
        bottom_comparison[1].rotate(PI / 2)
        bottom_comparison[1].next_to(bottom_comparison[0], DOWN, buff=0)
        bottom_comparison[1].shift(UP * SMALL_BUFF)
        bottom_comparison[2].move_to(bottom_comparison[0].get_bottom() + DOWN * SMALL_BUFF * 0.2)

        self.play(
            FadeIn(bottom_comparison)
        )

        self.play(
            FadeIn(top_comparison)
        )

        self.wait()

        new_v = TexMobject("v_{f + 1} := (-v_{f + 1}[0], v_{f + 1}[1])")
        new_v.scale(0.8)

        new_v.next_to(right_comparison, DOWN).to_edge(LEFT * 2)

        self.play(
            Write(new_v)
        )

        new_v_bottom = TexMobject("v_{f + 1} := (v_{f + 1}[0], -v_{f + 1}[1])")
        new_v_bottom.scale(0.8)

        new_v_bottom.next_to(bottom_comparison, DOWN).to_edge(LEFT * 2)

        self.play(
            Write(new_v_bottom)
        )

        self.wait()

class UpdaterCode(Scene):
    def construct(self):
        code = self.get_code()

        self.animate_all(code)

    def animate_all(self, code):
        self.wait()

        for i in range(len(code) - 6):
            self.play(
                Write(code[i])
            )

            self.wait()

        for i in range(len(code) - 6, len(code)):
            self.play(
                Write(code[i])
            )

        self.wait()


    def get_code(self):
        code_scale = 0.8
        
        code = []

        particle_def = TextMobject(r"p $=$ Particle()")
        particle_def[0][1].set_color(MONOKAI_PINK)
        particle_def[0][2:-2].set_color(MONOKAI_BLUE)
        particle_def.scale(code_scale)
        particle_def.to_edge(LEFT)
        code.append(particle_def)

        box_def = TextMobject(r"box $=$ Box()")
        box_def[0][3].set_color(MONOKAI_PINK)
        box_def[0][4:-2].set_color(MONOKAI_BLUE)
        box_def.scale(code_scale)
        box_def.next_to(particle_def, DOWN * 0.5)
        box_def.to_edge(LEFT)
        code.append(box_def)

        def_statement = TextMobject(r"$\text{def update}(dt):$")
        def_statement[0][:3].set_color(MONOKAI_BLUE)
        def_statement[0][3:9].set_color(MONOKAI_GREEN)
        def_statement[0][10:12].set_color(MONOKAI_ORANGE)
        def_statement.scale(code_scale)
        def_statement.next_to(box_def, DOWN * 0.5)
        def_statement.to_edge(LEFT)
        code.append(def_statement)

        comment_1 = TextMobject(r"\# dt: 1/FPS (e.g 1/60 for 60 FPS)")
        comment_2 = TextMobject(r"\# called by engine to update frames")
        
        
        comment_1.scale(code_scale)
        comment_1.next_to(def_statement, DOWN * 0.5)
        comment_1.to_edge(LEFT * 2)
        comment_1.set_color(MONOKAI_GRAY)
        code.append(comment_1)

        comment_2.scale(code_scale)
        comment_2.next_to(comment_1, DOWN * 0.5)
        comment_2.to_edge(LEFT * 2)
        comment_2.set_color(MONOKAI_GRAY)
        code.append(comment_2)
        line_1 = TextMobject(r"p.vel $=$ p.vel $+$ p.accel $*$ dt")
        line_1.scale(code_scale)
        line_1.next_to(comment_2, DOWN * 0.5)
        line_1.to_edge(LEFT * 2)
        line_1[0][5].set_color(MONOKAI_PINK)
        line_1[0][11].set_color(MONOKAI_PINK)
        line_1[0][19].set_color(MONOKAI_PINK)
        # line_1[0][20].shift(DOWN * SMALL_BUFF)
        code.append(line_1)

        line_2 = TextMobject(r"p.pos $=$ p.pos $+$ p.vel $*$ dt")
        line_2.scale(code_scale)
        line_2.next_to(line_1, DOWN * 0.5)
        line_2.to_edge(LEFT * 2)
        line_2[0][5].set_color(MONOKAI_PINK)
        line_2[0][11].set_color(MONOKAI_PINK)
        line_2[0][17].set_color(MONOKAI_PINK)
        code.append(line_2)


        line_3 = TextMobject(r"handleBoxCollision()")
        line_3.scale(code_scale)
        line_3.next_to(line_2, DOWN * 0.5)
        line_3.to_edge(LEFT * 2)
        line_3[0][:-2].set_color(MONOKAI_BLUE)
        code.append(line_3)

        line_4 = TextMobject(r"$\text{def handleBoxCollision}():$")
        line_4.scale(code_scale)
        line_4.next_to(line_3, DOWN * 1.5)
        line_4.to_edge(LEFT)
        line_4[0][:3].set_color(MONOKAI_BLUE)
        line_4[0][3:-3].set_color(MONOKAI_GREEN)
        code.append(line_4)

        line_5 = TextMobject(r"if p.left[0] $\leq$ box.left[0] or")
        line_5.scale(code_scale)
        line_5.next_to(line_4, DOWN * 0.5)
        line_5.to_edge(LEFT * 2)
        line_5[0][:2].set_color(MONOKAI_PINK)
        line_5[0][9].set_color(MONOKAI_PURPLE)
        line_5[0][11].set_color(MONOKAI_PINK)
        line_5[0][21].set_color(MONOKAI_PURPLE)
        line_5[0][23:25].set_color(MONOKAI_PINK)
        code.append(line_5)

        line_6 = TextMobject(r"p.right[0] $\geq$ box.right[0]:")
        line_6.scale(code_scale)
        line_6.next_to(line_5, DOWN * 0.5)
        line_6.to_edge(LEFT * 3)
        line_6[0][8].set_color(MONOKAI_PURPLE)
        line_6[0][10].set_color(MONOKAI_PINK)
        line_6[0][-3].set_color(MONOKAI_PURPLE)
        code.append(line_6)

        line_7 = TextMobject(r"p.vel[0] $= -$p.vel[0]")
        line_7.scale(code_scale)
        line_7.next_to(line_6, DOWN * 0.5)
        line_7.to_edge(LEFT * 3)
        line_7[0][6].set_color(MONOKAI_PURPLE)
        line_7[0][8].set_color(MONOKAI_PINK)
        line_7[0][9].set_color(MONOKAI_PINK)
        line_7[0][-2].set_color(MONOKAI_PURPLE)
        code.append(line_7)

        line_8 = TextMobject(r"if p.bottom[1] $\leq$ box.bottom[1] or")
        line_8.scale(code_scale)
        line_8.next_to(line_7, DOWN * 0.5)
        line_8.to_edge(LEFT * 2)
        line_8[0][:2].set_color(MONOKAI_PINK)
        line_8[0][11].set_color(MONOKAI_PURPLE)
        line_8[0][13].set_color(MONOKAI_PINK)
        line_8[0][25].set_color(MONOKAI_PURPLE)
        line_8[0][27:29].set_color(MONOKAI_PINK)
        code.append(line_8)

        line_9 = TextMobject(r"p.top[1] $\geq$ box.top[1]:")
        line_9.scale(code_scale)
        line_9.next_to(line_8, DOWN * 0.5)
        line_9.to_edge(LEFT * 3)
        line_9[0][6].set_color(MONOKAI_PURPLE)
        line_9[0][8].set_color(MONOKAI_PINK)
        line_9[0][-3].set_color(MONOKAI_PURPLE)
        code.append(line_9)

        line_10 = TextMobject(r"p.vel[1] $= -$p.vel[1]")
        line_10[0][6].set_color(MONOKAI_PURPLE)
        line_10[0][8].set_color(MONOKAI_PINK)
        line_10[0][9].set_color(MONOKAI_PINK)
        line_10[0][-2].set_color(MONOKAI_PURPLE)
        line_10.scale(code_scale)
        line_10.next_to(line_9, DOWN * 0.5)
        line_10.to_edge(LEFT * 3)
        code.append(line_10)

        code = VGroup(*code)
        code.scale(0.9)
        code.move_to(RIGHT * 3)
        return code

class TunnelingIssue(Scene):
    def construct(self):
        boxes, balls, text, title = self.intro_discrete_collision_detection()

        original_box = boxes[0].copy()
        original_ball = balls[0].copy()

        box = Box(height=4.5, width=4.5).set_color(COBALT_BLUE)
        box.shift(DOWN * 0.3)
        ball = Ball(radius=0.2).shift(DOWN * 0.3)

        self.play(
            ReplacementTransform(boxes, box),
            ReplacementTransform(balls, ball),
            FadeOut(text),
            run_time=2
        )

        self.wait()

        ball.velocity = np.array([1, -50, 0])
        BOX_THRESHOLD = 0.98

        def update_ball(ball, dt):
            ball.acceleration = np.array((0, 0, 0))
            ball.velocity = ball.velocity + ball.acceleration * dt
            ball.shift(ball.velocity * dt)  # Bounce off ground and roof
            if ball.get_bottom() <= box.get_bottom() * BOX_THRESHOLD or \
                    ball.get_top() >= box.get_top() * BOX_THRESHOLD:
                ball.velocity[1] = -ball.velocity[1]
            # Bounce off walls
            if ball.get_left_edge() <= box.get_left_edge() or \
                    ball.get_right_edge() >= box.get_right_edge():
                ball.velocity[0] = -ball.velocity[0]

        ball.add_updater(update_ball)
        self.add(ball)
        self.wait(5)
        ball.clear_updaters()
        self.wait()

        continuous = TextMobject("Continuous Collision Detection").scale(1.2)
        continuous.move_to(title.get_center())
        
        self.play(
            ReplacementTransform(box, original_box),
            ReplacementTransform(ball, original_ball),
            run_time=2
        )

        self.wait()

        easy_solutions = TextMobject("Easy Solutions")
        solution_1 = TextMobject("1. Enforce speed limits on particles").scale(0.8)
        solution_2 = TextMobject("2. Use higher frame rates").scale(0.8)

        solutions = VGroup(easy_solutions, solution_1, solution_2).arrange_submobjects(DOWN)
        solutions.move_to(RIGHT * 3)

        self.play(
            Write(easy_solutions)
        )

        self.wait()

        self.play(
            Write(solution_1)
        )

        self.wait()

        self.play(
            Write(solution_2)
        )

        self.wait()

        self.play(
            ReplacementTransform(title, continuous),
            FadeOut(solutions),
            run_time=2
        )

        self.wait()

    def intro_discrete_collision_detection(self):
        title = TextMobject("Discrete Collision Detection").scale(1.2).move_to(UP * 3.5)
        h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
        h_line.next_to(title, DOWN)
        self.play(
            Write(title),
            ShowCreation(h_line)
        )

        self.wait()

        frame_change = TextMobject(r"Frame $f \rightarrow f + 1$").scale(0.8)
        frame_change.next_to(h_line, DOWN)

        self.play(
            Write(frame_change)
        )

        box1 = Box(height=4.5, width=4.5).set_color(COBALT_BLUE).shift(DOWN * 0.3)
        box1.shift(LEFT * 3)

        box2 = Box(height=4.5, width=4.5).set_color(COBALT_BLUE).shift(DOWN * 0.3)
        box2.shift(RIGHT * 3)

        ball1 = Ball(radius=0.2).move_to(box1.get_center() + DOWN * 1.8)

        ball2 = Ball(radius=0.2).move_to(box2.get_center() + DOWN * 2.8 + RIGHT * 0.5)

        self.play(
            FadeIn(box1),
            FadeIn(ball1),
        )

        self.wait()

        self.play(
            FadeIn(box2),
            FadeIn(ball2),
        )

        self.wait()

        tunneling = TextMobject("Tunneling").set_color(YELLOW)
        tunneling.move_to(DOWN * 3.5)

        self.play(
            Write(tunneling)
        )

        self.wait()

        return VGroup(box1, box2), VGroup(ball1, ball2), VGroup(frame_change, tunneling), title

class CCDLast(ZoomedScene):
    CONFIG = {
        "zoom_factor": 0.4,
        "zoomed_display_height": 6,
        "zoomed_display_width": 5,
        "image_frame_stroke_width": 7,
        "zoomed_display_corner": RIGHT + UP * 1,
        "zoomed_camera_config": {
            "default_frame_stroke_width": 3,
        },
    }

    def construct(self):
        title = TextMobject("Continuous Collision Detection").scale(1.2).move_to(UP * 3.5)
        h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
        h_line.next_to(title, DOWN)
        self.add(title)
        self.add(h_line)

        box1 = Box(height=4.5, width=4.5).set_color(COBALT_BLUE).shift(DOWN * 0.3)
        box1.shift(LEFT * 3)

        ball1 = Ball(radius=0.2).move_to(box1.get_center() + DOWN * 1.8)

        ball2 = Ball(radius=0.2).move_to(box1.get_center() + DOWN * 2.8 + RIGHT * 0.5)

        self.add(box1)
        self.add(ball1)


        ball2.set_fill(opacity=0.8)
        ball2.set_stroke(opacity=0)

        self.play(
            FadeIn(ball2)
        )

        self.wait()

        self.show_CCD(box1, ball1, ball2)

    def show_CCD(self, box1, ball1, ball2):
        surrounding_box = SurroundingRectangle(VGroup(ball1, ball2), buff=MED_SMALL_BUFF)
        
        zoomed_camera = self.zoomed_camera
        # This preserves line thickness
        zoomed_camera.cairo_line_width_multiple = 0.05
        
        zoomed_display = self.zoomed_display
        frame = zoomed_camera.frame
        zoomed_display_frame = zoomed_display.display_frame

        frame.move_to(surrounding_box.get_center() + UP * 0.2)
        frame.set_color(YELLOW)

        zoomed_display_frame.set_color(YELLOW)
        zoomed_display.shift(DOWN)

        # brackground zoomed_display
        zd_rect = BackgroundRectangle(
            zoomed_display,
            fill_opacity=0,
            buff=MED_SMALL_BUFF,
        )

        self.add_foreground_mobject(zd_rect)

        # animation of unfold camera
        unfold_camera = UpdateFromFunc(
            zd_rect,
            lambda rect: rect.replace(zoomed_display)
        )

        self.play(
            ShowCreation(frame),
        )

        # Activate zooming
        self.activate_zooming()

        self.play(
            # You have to add this line
            self.get_zoomed_display_pop_out_animation(),
            unfold_camera
        )

        start = Dot().move_to(ball1.get_center()).scale(self.zoom_factor)
        end = Dot().move_to(ball2.get_center()).scale(self.zoom_factor)

        start_color = WHITE
        end_color = WHITE
        start.set_color(start_color)
        end.set_color(end_color)

        start_point = TexMobject(r"p_f = (x(0), y(0))").set_color(start_color)
        end_point = TexMobject(r"p_{f + 1} = (x(1), y(1))").set_color(end_color)
        end_point.shift(UP * 0.1)
        self.play(
            GrowFromCenter(start)
        )

        start_point.scale(0.8 * self.zoom_factor)
        start_point.next_to(ball1, UP, buff=SMALL_BUFF)
        self.play(
            Write(start_point)
        )

        end_point.scale(0.8 * self.zoom_factor)
        end_point.next_to(ball2, DOWN, buff=SMALL_BUFF)

        self.play(
            GrowFromCenter(end)
        )

        self.play(
            Write(end_point)
        )

        self.wait()

        self.add_foreground_mobject(start)
        self.add_foreground_mobject(end)

        line_seg = self.make_line(start, end)

        line_seg.set_color(YELLOW)

        self.play(
            ShowCreation(line_seg)
        )

        self.wait()

        point_color = WHITE
        point_on_line = Dot().scale(self.zoom_factor).set_color(point_color)
        point_on_line.move_to(line_seg.point_from_proportion(1 / 4))
        self.play(
            GrowFromCenter(point_on_line)
        )

        point_equation = TexMobject("p = (x(t), y(t))").scale(0.75 * self.zoom_factor)
        t_range = TexMobject(r"t \in [0, 1]").scale(0.75 * self.zoom_factor)

        label = VGroup(point_equation, t_range).arrange_submobjects(DOWN, buff=SMALL_BUFF / 2)
        label.next_to(point_on_line, RIGHT, buff=SMALL_BUFF / 2)
        label.set_color(WHITE)

        self.play(
            Write(label)
        )

        self.wait()

        direction = end.get_center() - start.get_center()

        self.play(
            point_on_line.shift, direction * 0.7,
            label.shift, direction * 0.7
        )

        self.play(
            point_on_line.shift, -direction * 0.9,
            label.shift, -direction * 0.9
        )

        self.play(
            point_on_line.shift, direction * 0.2,
            label.shift, direction * 0.2
        )

        self.wait()

        new_box = Box(height=1.5, width=4.5).shift(LEFT * 3).set_color(COBALT_BLUE)
        new_box.shift(UP * box1.get_bottom() - UP * new_box.get_bottom())
        self.play(
            Transform(box1, new_box)
        )

        x_t = TexMobject(r"x(t) = t \cdot x(0) + (1 - t) \cdot x(1)")
        x_t.scale(0.8)

        indent = LEFT * 2
        x_t.move_to(UP * 2.5)
        x_t.to_edge(indent)

        self.play(
            Write(x_t)
        )

        y_t = TexMobject(r"y(t) = t \cdot y(0) + (1 - t) \cdot y(1)")
        y_t.scale(0.8)

        y_t.next_to(x_t, DOWN)
        y_t.to_edge(indent)

        self.play(
            Write(y_t)
        )

        self.wait()

        lin_interp = TextMobject(
            "Linear" + "\\\\" + 
            "Interpolation" + "\\\\" +
            r"$t \in [0, 1]$"
        ).scale(0.7)

        brace = Brace(VGroup(x_t, y_t), direction=RIGHT)
        lin_interp.next_to(brace, RIGHT, buff=SMALL_BUFF)

        self.play(
            GrowFromCenter(brace)
        )

        self.play(
            Write(lin_interp)
        )

        y_b = TexMobject("y = b").scale(0.8).move_to(RIGHT + DOWN * 0.9)
        dashed_h_line = DashedLine(LEFT, RIGHT, dash_length=0.1)
        dashed_h_line.move_to(box1.get_center() - box1.get_height() / 2)
        dashed_h_line.shift(RIGHT)
        self.play(
            ShowCreation(dashed_h_line)
        )

        self.play(
            Write(y_b)
        )

        radius_label = TexMobject("r")
        radius_line = Line(ball1.get_center(), 
            ball1.get_center() + DOWN * (ball1.radius + SMALL_BUFF / 10)) 
        radius_line.set_color(WHITE)
        self.play(
            ShowCreation(radius_line)
        )
        radius_label.scale(0.7 * self.zoom_factor)
        radius_label.next_to(radius_line, LEFT, buff=SMALL_BUFF / 2)

        self.wait(3)

        self.play(
            FadeOut(label),
            FadeOut(point_on_line)
        )

        self.wait()
 
        p_collision = line_seg.point_from_proportion(0.22)

        shift_down = p_collision - start.get_center()

        p_collision_point = Dot().scale(self.zoom_factor).move_to(p_collision)
        p_collision_point.set_color(FUCHSIA)
        

        radius_label.shift(shift_down)
        self.play(
            ball1.move_to, p_collision,
            radius_line.shift, shift_down,

        )

        self.play(
            Flash(p_collision_point, 
                flash_radius=0.3 * self.zoom_factor,
                line_length=0.2 * self.zoom_factor,
                color=FUCHSIA,
            ),
            GrowFromCenter(p_collision_point),
        )

        self.wait()

        self.add_foreground_mobject(p_collision_point)

        label_p_c = TexMobject("p_c = (x(t_c), y(t_c))").scale(0.68 * self.zoom_factor)
        label_p_c.next_to(p_collision_point, RIGHT, buff=SMALL_BUFF / 2)

        self.play(
            ball1.move_to, start.get_center(),
            Write(label_p_c)
        )

        self.play(
            Write(radius_label)
        )

        self.wait()

        y_t_c = TexMobject("y(t_c) = ", "b + r").scale(0.8)

        y_t_c.next_to(y_t, DOWN)
        y_t_c.to_edge(indent)

        self.play(
            Write(y_t_c)
        )
        self.wait()

        y_t_c_alt = TexMobject(r"y(t_c) = ", "t_c \cdot y(0) +  (1 - t_c) \cdot y(1)").scale(0.8)
        y_t_c_alt.next_to(y_t_c, DOWN)
        y_t_c_alt.to_edge(indent)
        self.play(
            Write(y_t_c_alt)
        )

        self.wait()

        equality = TexMobject("b + r", " = ", "t_c \cdot y(0) +  (1 - t_c) \cdot y(1)").scale(0.8)
        equality.next_to(y_t_c_alt, DOWN)
        equality.to_edge(indent)

        self.play(
            TransformFromCopy(y_t_c[1], equality[0]),
            TransformFromCopy(y_t_c_alt[1], equality[2]),
            FadeIn(equality[1]),
            run_time=2
        )

        self.wait()

        self.play(
            FadeOut(y_t_c),
            FadeOut(y_t_c_alt),
            equality.shift, UP * 1.2,
            run_time=2
        )

        result = TexMobject(
            r"\Rightarrow t_c = \frac{b + r - y(1)}{y(0) - y(1)}",
            r" \approx 0.23"
        )
        result.scale(0.8).next_to(equality, DOWN).to_edge(indent)
        
        self.play(
            Write(result[0])
        )

        self.wait()

        self.play(
            Write(result[1])
        )

        self.wait()

        self.play(
            ApplyWave(x_t, color=PINK),
            ApplyWave(y_t, color=PINK),
            run_time=2
        )

        self.wait()

        self.play(
            FadeOut(start_point),
            FadeOut(end_point),
            FadeOut(label_p_c),
            FadeOut(radius_label),
            FadeOut(radius_line),
            FadeOut(dashed_h_line),
            FadeOut(y_b),
        )

        self.wait()

        self.play(
            ball1.move_to, p_collision
        )

        self.wait()

        highlight_line = self.make_line(p_collision_point, end)
        highlight_line.set_color(GREEN_SCREEN)

        self.play(
            ShowCreation(highlight_line)
        )

        self.wait()

        reflection_line = DashedLine(LEFT, RIGHT, dash_length=0.1).move_to(p_collision)
        reflection_line.shift(RIGHT * SMALL_BUFF)

        self.play(
            ShowCreation(reflection_line)
        )

        self.wait()

        final_point = np.array([end.get_center()[0], p_collision[1] + (p_collision[1] - end.get_center()[1]), 0])
        final_point_dot = Dot().scale(self.zoom_factor).move_to(final_point)
        rest_path = self.make_line(p_collision_point, final_point_dot, dashed=True)
        rest_path.set_color(highlight_line.get_color())
        self.play(
            ReplacementTransform(highlight_line, rest_path),
            TransformFromCopy(end, final_point_dot),
            run_time=2
        )
        self.add_foreground_mobject(final_point_dot)

        self.wait()

        first_segment = self.make_line(start, p_collision_point).set_color(YELLOW)
        second_segment = self.make_line(p_collision_point, end).set_color(YELLOW)
        

        rest_path_yellow = Line(rest_path.get_start(), rest_path.get_end()).set_color(YELLOW)

        self.play(
           FadeOut(line_seg),
           FadeIn(first_segment),
           FadeOut(end),
           ball2.move_to, final_point_dot.get_center(),
           Transform(rest_path, rest_path_yellow)
        )

        self.wait(3)

        self.play(
            FadeOut(reflection_line),
            ball1.move_to, start.get_center(),
            FadeOut(ball2)
        )

        self.wait()

        path = [start.get_center(), p_collision]
        path += [rest_path_yellow.point_from_proportion(i / 3) for i in range (1, 3)]
        path.append(final_point)
        trajectory = VGroup()
        trajectory.set_points_as_corners(*[path])
        self.play(
            MoveAlongPath(ball1, trajectory),
            rate_func=linear
        )

        self.wait()

        self.remove(zoomed_display)
        self.remove(brace)
        self.remove(lin_interp)
        self.wait()

        
        notes = TextMobject("Important notes about CCD")
        cmplx = TextMobject("- Often more tricky").scale(0.8)
        distances = TextMobject("- Distances").scale(0.8)
        dimensions = TextMobject("- Multiple dimensions").scale(0.8)
        geometries = TextMobject("- Complex geometries").scale(0.8)
        root_finding = TextMobject("- Root finding algorithms").scale(0.8)
        down_arrow = TexMobject(r"\Downarrow")
        intro = TextMobject(r"This is a $\textit{gentle}$ intro to CCD")

        complexities = VGroup(
            notes, cmplx, distances, dimensions, 
            geometries, root_finding, down_arrow, intro
        )
        complexities.arrange_submobjects(DOWN).move_to(RIGHT * 3)
        for i in range(len(complexities)):
            self.play(
                Write(complexities[i])
            )
            self.wait()

    def make_line(self, dot1, dot2, dashed=False):
        vector = dot2.get_center() - dot1.get_center()
        unit_v = vector / np.linalg.norm(vector)
        start = dot1.get_center() + vector * (dot1.radius * (self.zoom_factor - 0.05))
        end = dot2.get_center() - vector * (dot2.radius * (self.zoom_factor - 0.05))
        if dashed:
            return DashedLine(start, end)
        return Line(start, end)

class TwoParticleSim(Scene):
    CONFIG = {
        "sim_time": 120,
    }
    def construct(self):
        debug = False
        balls = []
        num_balls = 2
        BOX_THRESHOLD = 0.98
        BALL_THRESHOLD = 0.96
        box = Box(height=5.5, width=5.5).set_color(COBALT_BLUE)
        box.shift(RIGHT * 4)
        colors = [BLUE, YELLOW, GREEN_SCREEN, ORANGE]
        velocities = [RIGHT * 2 + UP * 2, LEFT * 1 + UP * 2]
        positions = [RIGHT * 3, RIGHT * 5]
        
        for i in range(len(positions)):
            if i == 0:
                ball = Ball(
                    radius=0.3, color=colors[i % len(colors)], fill_color=colors[i % len(colors)], opacity=1
                )
            else:
                ball = Ball(
                    radius=0.4, color=colors[i % len(colors)], fill_color=colors[i % len(colors)], opacity=1
                )
            ball.id = i
            ball.move_to(positions[i])
            ball.velocity = velocities[i]
            balls.append(ball)
        
        self.play(
            FadeIn(box)
        )
        self.play(
            *[FadeIn(ball) for ball in balls]
        )

        # useful info for debugging
        p1_value_x = round(balls[0].get_center()[0], 3)
        p1_value_y = round(balls[0].get_center()[1], 3)
        p1_text = TextMobject("Position: ")
        p1 = VGroup(p1_text, DecimalNumber(p1_value_x), DecimalNumber(p1_value_y))
        p1.arrange_submobjects(RIGHT * 1.5).set_color(BLUE)

        v1_value_x = round(balls[0].velocity[0], 3)
        v1_value_y = round(balls[0].velocity[1], 3)
        v1_text = TextMobject("Velocity: ")
        v1 = VGroup(v1_text, DecimalNumber(v1_value_x), DecimalNumber(v1_value_y))
        v1.arrange_submobjects(RIGHT * 1.5).set_color(BLUE)

        p2_value_x = round(balls[1].get_center()[0], 3)
        p2_value_y = round(balls[1].get_center()[1], 3)
        p2_text = TextMobject("Position: ")
        p2 = VGroup(p2_text, DecimalNumber(p2_value_x), DecimalNumber(p2_value_y))
        p2.arrange_submobjects(RIGHT * 1.5).set_color(YELLOW)

        v2_value_x = round(balls[1].velocity[0], 3)
        v2_value_y = round(balls[1].velocity[1], 3)
        v2_text = TextMobject("Velocity: ")
        v2 = VGroup(v2_text, DecimalNumber(v2_value_x), DecimalNumber(v2_value_y))
        v2.arrange_submobjects(RIGHT * 1.5).set_color(YELLOW)

        if debug:
            debug_log = VGroup(p1, v1, p2, v2).arrange_submobjects(DOWN) 
            debug_log.shift(LEFT * 4)
            self.play(
                FadeIn(debug_log)
            )



        def update_ball(ball, dt):
            ball.acceleration = np.array((0, 0, 0))
            ball.velocity = ball.velocity + ball.acceleration * dt
            ball.shift(ball.velocity * dt)
            handle_collision_with_box(ball, box, dt)
            handle_ball_collisions(ball, dt)

            if ball.get_color() == Color(BLUE) and debug:
                p1[1].set_value(ball.get_center()[0])
                p1[2].set_value(ball.get_center()[1])
                v1[1].set_value(ball.velocity[0])
                v1[2].set_value(ball.velocity[1])

            if ball.get_color() == Color(YELLOW) and debug:
                p2[1].set_value(ball.get_center()[0])
                p2[2].set_value(ball.get_center()[1])
                v2[1].set_value(ball.velocity[0])
                v2[2].set_value(ball.velocity[1])

        def handle_collision_with_box(ball, box, dt):
            # Bounce off ground and roof
            if ball.get_bottom() <= box.get_bottom()*BOX_THRESHOLD or \
                    ball.get_top() >= box.get_top()*BOX_THRESHOLD:
                ball.velocity[1] = -ball.velocity[1]
                ball.shift(ball.velocity * dt)
            # Bounce off walls
            if ball.get_left_edge() <= box.get_left_edge() or \
                    ball.get_right_edge() >= box.get_right_edge():
                ball.velocity[0] = -ball.velocity[0]
                ball.shift(ball.velocity * dt)

        def handle_ball_collisions(ball, dt):
            t_colors = [RED, ORANGE, GREEN_SCREEN, GOLD, PINK, WHITE]
            for other_ball in balls:
                if ball.id != other_ball.id:
                    dist = np.linalg.norm(ball.get_center() - other_ball.get_center())
                    if dist * BALL_THRESHOLD <= (ball.radius + other_ball.radius):
                        # ball.set_color(random.choice(t_colors))
                        # other_ball.set_color(random.choice(t_colors))
                        v1, v2 = get_response_velocities(ball, other_ball)
                        ball.velocity = v1
                        other_ball.velocity = v2
                        ball.shift(ball.velocity * dt)
                        other_ball.shift(other_ball.velocity * dt)
        
        def get_response_velocities(ball, other_ball):
            # https://en.wikipedia.org/wiki/Elastic_collision
            v1 = ball.velocity
            v2 = other_ball.velocity
            m1 = ball.mass
            m2 = other_ball.mass
            x1 = ball.get_center()
            x2 = other_ball.get_center()

            ball_response_v = compute_velocity(v1, v2, m1, m2, x1, x2)
            other_ball_response_v = compute_velocity(v2, v1, m2, m1, x2, x1)
            return ball_response_v, other_ball_response_v

        def compute_velocity(v1, v2, m1, m2, x1, x2):
            return v1 - (2 * m2 / (m1 + m2)) * np.dot(v1 - v2, x1 - x2) / np.linalg.norm(x1 - x2) ** 2 * (x1 - x2)

        for ball in balls:
            ball.add_updater(update_ball)
            self.add(ball)

        self.wait(self.sim_time)
        for ball in balls:
            ball.clear_updaters()
        self.wait(3)

class TwoParticleSimDebugCase(Scene):
    CONFIG = {
        "sim_time": 5,
    }
    def construct(self):
        debug = True
        balls = []
        num_balls = 2
        BOX_THRESHOLD = 0.98
        BALL_THRESHOLD = 0.96
        box = Box(height=5.5, width=5.5).set_color(COBALT_BLUE)
        box.shift(RIGHT * 3)
        colors = [BLUE, YELLOW, GREEN_SCREEN, ORANGE]
        velocities = [RIGHT * 1.02 + UP * 2.26, LEFT * 1.27 + UP * 2.10]
        positions = [RIGHT * 1.30 + DOWN * 1.61, RIGHT * 3.79 + UP * 0.13]
        
        for i in range(len(positions)):
            if i == 0:
                ball = Ball(
                    radius=0.3, color=colors[i % len(colors)], fill_color=colors[i % len(colors)], opacity=1
                )
            else:
                ball = Ball(
                    radius=0.4, color=colors[i % len(colors)], fill_color=colors[i % len(colors)], opacity=1
                )
            ball.id = i
            ball.move_to(positions[i])
            ball.velocity = velocities[i]
            balls.append(ball)
        
        self.play(
            FadeIn(box)
        )
        self.play(
            *[FadeIn(ball) for ball in balls]
        )

        # useful info for debugging
        p1_value_x = round(balls[0].get_center()[0], 3)
        p1_value_y = round(balls[0].get_center()[1], 3)
        p1_text = TextMobject("Position: ")
        p1 = VGroup(p1_text, DecimalNumber(p1_value_x), DecimalNumber(p1_value_y))
        p1.arrange_submobjects(RIGHT * 1.5).set_color(BLUE)

        v1_value_x = round(balls[0].velocity[0], 3)
        v1_value_y = round(balls[0].velocity[1], 3)
        v1_text = TextMobject("Velocity: ")
        v1 = VGroup(v1_text, DecimalNumber(v1_value_x), DecimalNumber(v1_value_y))
        v1.arrange_submobjects(RIGHT * 1.5).set_color(BLUE)

        p2_value_x = round(balls[1].get_center()[0], 3)
        p2_value_y = round(balls[1].get_center()[1], 3)
        p2_text = TextMobject("Position: ")
        p2 = VGroup(p2_text, DecimalNumber(p2_value_x), DecimalNumber(p2_value_y))
        p2.arrange_submobjects(RIGHT * 1.5).set_color(YELLOW)

        v2_value_x = round(balls[1].velocity[0], 3)
        v2_value_y = round(balls[1].velocity[1], 3)
        v2_text = TextMobject("Velocity: ")
        v2 = VGroup(v2_text, DecimalNumber(v2_value_x), DecimalNumber(v2_value_y))
        v2.arrange_submobjects(RIGHT * 1.5).set_color(YELLOW)

        if debug:
            debug_log = VGroup(p1, v1, p2, v2).arrange_submobjects(DOWN) 
            debug_log.shift(LEFT * 4)
            self.play(
                FadeIn(debug_log)
            )



        def update_ball(ball, dt):
            ball.acceleration = np.array((0, 0, 0))
            ball.velocity = ball.velocity + ball.acceleration * dt
            ball.shift(ball.velocity * dt)
            handle_collision_with_box(ball, box)
            handle_ball_collisions(ball, dt)

            if ball.get_color() == Color(BLUE) and debug:
                p1[1].set_value(ball.get_center()[0])
                p1[2].set_value(ball.get_center()[1])
                v1[1].set_value(ball.velocity[0])
                v1[2].set_value(ball.velocity[1])

            if ball.get_color() == Color(YELLOW) and debug:
                p2[1].set_value(ball.get_center()[0])
                p2[2].set_value(ball.get_center()[1])
                v2[1].set_value(ball.velocity[0])
                v2[2].set_value(ball.velocity[1])

        def handle_collision_with_box(ball, box):
            # Bounce off ground and roof
            if ball.get_bottom() <= box.get_bottom()*BOX_THRESHOLD or \
                    ball.get_top() >= box.get_top()*BOX_THRESHOLD:
                    ball.velocity[1] = -ball.velocity[1]
            # Bounce off walls
            if ball.get_left_edge() <= box.get_left_edge() or \
                    ball.get_right_edge() >= box.get_right_edge():
                ball.velocity[0] = -ball.velocity[0]

        def handle_ball_collisions(ball, dt):
            t_colors = [RED, ORANGE, GREEN_SCREEN, GOLD, PINK, WHITE]
            for other_ball in balls:
                if ball.id != other_ball.id:
                    dist = np.linalg.norm(ball.get_center() - other_ball.get_center())
                    if dist * BALL_THRESHOLD <= (ball.radius + other_ball.radius):
                        # ball.set_color(random.choice(t_colors))
                        # other_ball.set_color(random.choice(t_colors))
                        v1, v2 = get_response_velocities(ball, other_ball)
                        ball.velocity = v1
                        other_ball.velocity = v2
                        ball.shift(ball.velocity * dt)
                        other_ball.shift(other_ball.velocity * dt)
            

        
        def get_response_velocities(ball, other_ball):
            # https://en.wikipedia.org/wiki/Elastic_collision
            v1 = ball.velocity
            v2 = other_ball.velocity
            m1 = ball.mass
            m2 = other_ball.mass
            x1 = ball.get_center()
            x2 = other_ball.get_center()

            ball_response_v = compute_velocity(v1, v2, m1, m2, x1, x2)
            other_ball_response_v = compute_velocity(v2, v1, m2, m1, x2, x1)
            return ball_response_v, other_ball_response_v

        def compute_velocity(v1, v2, m1, m2, x1, x2):
            return v1 - (2 * m2 / (m1 + m2)) * np.dot(v1 - v2, x1 - x2) / np.linalg.norm(x1 - x2) ** 2 * (x1 - x2)

        for ball in balls:
            ball.add_updater(update_ball)
            self.add(ball)

        self.wait(self.sim_time)
        for ball in balls:
            ball.clear_updaters()
        self.wait(3)

class TwoParticleSimLeft(Scene):
    CONFIG = {
        "sim_time": 40,
    }
    def construct(self):
        debug = False
        balls = []
        num_balls = 2
        BOX_THRESHOLD = 0.98
        BALL_THRESHOLD = 0.96
        box = Box(height=5.5, width=5.5).set_color(COBALT_BLUE)
        box.shift(LEFT * 4)
        colors = [BLUE, YELLOW, GREEN_SCREEN, ORANGE]
        velocities = [RIGHT * 2 + UP * 2, LEFT * 1 + UP * 2]
        positions = [LEFT * 3, LEFT * 5]
        
        for i in range(len(positions)):
            if i == 0:
                ball = Ball(
                    radius=0.3, color=colors[i % len(colors)], fill_color=colors[i % len(colors)], opacity=1
                )
            else:
                ball = Ball(
                    radius=0.4, color=colors[i % len(colors)], fill_color=colors[i % len(colors)], opacity=1
                )
            ball.id = i
            ball.move_to(positions[i])
            ball.velocity = velocities[i]
            balls.append(ball)
        
        self.play(
            FadeIn(box)
        )
        self.play(
            *[FadeIn(ball) for ball in balls]
        )

        # useful info for debugging
        p1_value_x = round(balls[0].get_center()[0], 3)
        p1_value_y = round(balls[0].get_center()[1], 3)
        p1_text = TextMobject("Position: ")
        p1 = VGroup(p1_text, DecimalNumber(p1_value_x), DecimalNumber(p1_value_y))
        p1.arrange_submobjects(RIGHT * 1.5).set_color(BLUE)

        v1_value_x = round(balls[0].velocity[0], 3)
        v1_value_y = round(balls[0].velocity[1], 3)
        v1_text = TextMobject("Velocity: ")
        v1 = VGroup(v1_text, DecimalNumber(v1_value_x), DecimalNumber(v1_value_y))
        v1.arrange_submobjects(RIGHT * 1.5).set_color(BLUE)

        p2_value_x = round(balls[1].get_center()[0], 3)
        p2_value_y = round(balls[1].get_center()[1], 3)
        p2_text = TextMobject("Position: ")
        p2 = VGroup(p2_text, DecimalNumber(p2_value_x), DecimalNumber(p2_value_y))
        p2.arrange_submobjects(RIGHT * 1.5).set_color(YELLOW)

        v2_value_x = round(balls[1].velocity[0], 3)
        v2_value_y = round(balls[1].velocity[1], 3)
        v2_text = TextMobject("Velocity: ")
        v2 = VGroup(v2_text, DecimalNumber(v2_value_x), DecimalNumber(v2_value_y))
        v2.arrange_submobjects(RIGHT * 1.5).set_color(YELLOW)

        if debug:
            debug_log = VGroup(p1, v1, p2, v2).arrange_submobjects(DOWN) 
            debug_log.shift(LEFT * 4)
            self.play(
                FadeIn(debug_log)
            )



        def update_ball(ball, dt):
            ball.acceleration = np.array((0, 0, 0))
            ball.velocity = ball.velocity + ball.acceleration * dt
            ball.shift(ball.velocity * dt)
            handle_collision_with_box(ball, box, dt)
            handle_ball_collisions(ball, dt)

            if ball.get_color() == Color(BLUE) and debug:
                p1[1].set_value(ball.get_center()[0])
                p1[2].set_value(ball.get_center()[1])
                v1[1].set_value(ball.velocity[0])
                v1[2].set_value(ball.velocity[1])

            if ball.get_color() == Color(YELLOW) and debug:
                p2[1].set_value(ball.get_center()[0])
                p2[2].set_value(ball.get_center()[1])
                v2[1].set_value(ball.velocity[0])
                v2[2].set_value(ball.velocity[1])

        def handle_collision_with_box(ball, box, dt):
            # Bounce off ground and roof
            if ball.get_bottom() <= box.get_bottom()*BOX_THRESHOLD or \
                    ball.get_top() >= box.get_top()*BOX_THRESHOLD:
                ball.velocity[1] = -ball.velocity[1]
                ball.shift(ball.velocity * dt)
            # Bounce off walls
            if ball.get_left_edge() <= box.get_left_edge() or \
                    ball.get_right_edge() >= box.get_right_edge():
                ball.velocity[0] = -ball.velocity[0]
                ball.shift(ball.velocity * dt)

        def handle_ball_collisions(ball, dt):
            t_colors = [RED, ORANGE, GREEN_SCREEN, GOLD, PINK, WHITE]
            for other_ball in balls:
                if ball.id != other_ball.id:
                    dist = np.linalg.norm(ball.get_center() - other_ball.get_center())
                    if dist * BALL_THRESHOLD <= (ball.radius + other_ball.radius):
                        # ball.set_color(random.choice(t_colors))
                        # other_ball.set_color(random.choice(t_colors))
                        v1, v2 = get_response_velocities(ball, other_ball)
                        ball.velocity = v1
                        other_ball.velocity = v2
                        ball.shift(ball.velocity * dt)
                        other_ball.shift(other_ball.velocity * dt)
        
        def get_response_velocities(ball, other_ball):
            # https://en.wikipedia.org/wiki/Elastic_collision
            v1 = ball.velocity
            v2 = other_ball.velocity
            m1 = ball.mass
            m2 = other_ball.mass
            x1 = ball.get_center()
            x2 = other_ball.get_center()

            ball_response_v = compute_velocity(v1, v2, m1, m2, x1, x2)
            other_ball_response_v = compute_velocity(v2, v1, m2, m1, x2, x1)
            return ball_response_v, other_ball_response_v

        def compute_velocity(v1, v2, m1, m2, x1, x2):
            return v1 - (2 * m2 / (m1 + m2)) * np.dot(v1 - v2, x1 - x2) / np.linalg.norm(x1 - x2) ** 2 * (x1 - x2)

        for ball in balls:
            ball.add_updater(update_ball)
            self.add(ball)

        self.wait(self.sim_time)
        for ball in balls:
            ball.clear_updaters()
        self.wait(3)

class ShowArrow(Scene):
    def construct(self):
        arrow = Arrow(LEFT, RIGHT).set_color(YELLOW)
        self.wait()

        self.play(
            ShowCreation(arrow)
        )

        self.wait()

class TwoParticleDescription(Scene):
    def construct(self):
        self.show_description()

    def show_description(self):
        indent = LEFT * 1.5
        dynamics = TextMobject("Particle Dynamics")
        dynamics.to_edge(indent)
        dynamics.shift(UP * 3.5)
        self.play(
            Write(dynamics)
        )

        self.wait()

        accel = TextMobject(r"$a = 0$ for each particle").scale(0.8)
        accel.next_to(dynamics, DOWN).to_edge(indent)
        
        self.play(
            Write(accel)
        )

        self.wait()

        velocities = TextMobject(
            r"Let $v_1$ and $v_2$ be particle velocities"
        ).scale(0.8)
        velocities.next_to(accel, DOWN).to_edge(indent)
        
        self.play(
            Write(velocities)
        )

        self.wait()

        previous_dyn = TextMobject(
            r"All previous frame by frame updates" + "\\\\",
            r"to dynamics still apply" 
        ).scale(0.8)
        previous_dyn.next_to(velocities, DOWN).to_edge(indent)
        previous_dyn[1].to_edge(indent)
        self.play(
            Write(previous_dyn)
        )

        self.wait()

        h_line = Line(LEFT * 7.5, LEFT * 2.5)
        h_line.next_to(previous_dyn, DOWN).to_edge(indent)

        self.play(
            Write(h_line)
        )

        self.wait()

        detection = TextMobject("Collision Detection")
        detection.next_to(h_line, DOWN).to_edge(LEFT * 2)
        self.play(
            Write(detection)
        )

        self.wait()

        circle1 = Circle(radius=0.5).set_fill(color=BLUE, opacity=0.5)
        circle1.set_stroke(opacity=0)

        circle2 = Circle(radius=0.7).set_fill(color=YELLOW, opacity=0.5)
        circle2.set_stroke(opacity=0)

        circle1.next_to(detection, DOWN * 1.2).to_edge(indent)

        circle2.next_to(circle1, RIGHT, buff=0)
        circle2.shift(LEFT * SMALL_BUFF * 1.5)

        collision = VGroup(circle1, circle2)

        collision.next_to(detection, DOWN)

        self.play(
            FadeIn(collision)
        )

        self.wait()

        r_1 = Line(
            circle1.get_center(), 
            circle1.get_center() + LEFT * circle1.radius
        ).set_color(WHITE)
        r_1_text = TexMobject("r_1").scale(0.6).set_color(WHITE)
        r_1_text.next_to(r_1, UP, buff=SMALL_BUFF)

        r_2 = Line(
            circle2.get_center(),
            circle2.get_center() + RIGHT * circle2.radius
        ).set_color(BLACK)
        r_2_text = TexMobject("r_2").scale(0.6).set_color(BLACK)
        r_2_text.next_to(r_2, UP, buff=SMALL_BUFF)

        self.play(
            ShowCreation(r_1),
            Write(r_1_text),
            ShowCreation(r_2),
            Write(r_2_text)
        )

        c1 = Dot().move_to(circle1.get_center()).set_color(WHITE)
        c2 = Dot().move_to(circle2.get_center()).set_color(BLACK)

        c1_text = TexMobject("C_1").scale(0.6).set_color(WHITE)
        c2_text = TexMobject("C_2").scale(0.6).set_color(BLACK)

        c1_text.next_to(c1, DOWN, buff=SMALL_BUFF)
        c2_text.next_to(c2, DOWN, buff=SMALL_BUFF)

        self.play(
            GrowFromCenter(c1),
            GrowFromCenter(c2),
            Write(c1_text),
            Write(c2_text)
        )

        diagram_labels = VGroup(
            r_1, r_1_text,
            r_2, r_2_text,
            c1, c1_text,
            c2, c2_text,
        )

        self.wait()

        detection_rule = TextMobject(
            r"dist($C_1, C_2$) $\leq r_1 + r_2$ ",
            r"$\Rightarrow$ Collision"
        )

        detection_rule[0][:4].set_color(MONOKAI_BLUE)

        detection_rule.scale(0.8)
        detection_rule.next_to(collision, DOWN)
        detection_rule.to_edge(indent).shift(RIGHT * SMALL_BUFF)

        self.play(
            Write(detection_rule[0])
        )

        self.wait()

        self.play(
            Write(detection_rule[1])
        )

        self.wait()
        
        shift_up = dynamics.get_center() - detection.get_center()

        response = TextMobject("Collision Response").next_to(h_line, DOWN)
        response.to_edge(indent)

        self.play(
            FadeOut(dynamics),
            FadeOut(accel),
            FadeOut(previous_dyn),
            detection.shift, shift_up,
            collision.shift, shift_up,
            diagram_labels.shift, shift_up,
            detection_rule.shift, shift_up,
            velocities.shift, DOWN * 2.6,
            Write(response),
            run_time=2
        )

        self.wait()

        mass = TextMobject(r"Let $m_1$ and $m_2$ be particle masses").scale(0.8)
        mass.next_to(velocities, DOWN).to_edge(indent)

        self.play(
            Write(mass)
        )

        self.wait()

        after_collision = TextMobject(r"After collision $\Rightarrow \hat{v}_1$ and $\hat{v}_2$")
        after_collision.scale(0.8)
        after_collision.next_to(mass, DOWN).to_edge(indent)
        
        self.play(
            Write(after_collision)
        )

        self.wait()

        momentum = TexMobject(r"m_1v_1 + m_2v_2 = m_1 \hat{v}_1 + m_2 \hat{v}_2")
        energy = TexMobject(r"\frac{1}{2} m_1v_1^2 + \frac{1}{2} m_2v_2^2 = \frac{1}{2} m_1 \hat{v}_1^2 + \frac{1}{2} m_2 \hat{v}_2^2")
        momentum.scale(0.8)
        energy.scale(0.8)
        constraints = VGroup(momentum, energy).arrange_submobjects(DOWN)
        
        constraints.next_to(after_collision, DOWN).to_edge(indent)

        self.play(
            FadeIn(constraints)
        )

        self.wait()

        eq1 = TexMobject(
            r"\hat{v}_1 = v_1 - \frac{2m_2}{m_1 + m_2} \frac{ \langle v_1 - v_2, C_1 - C_2 \rangle}{||C_1 - C_2||^2}(C_1 - C_2)"
        ).scale(0.7)
        eq1.next_to(after_collision, DOWN).to_edge(indent)

        eq2 = TexMobject(
            r"\hat{v}_2 = v_2 - \frac{2m_1}{m_1 + m_2} \frac{ \langle v_2 - v_1, C_2 - C_1 \rangle}{||C_2 - C_1||^2}(C_2 - C_1)"
        ).scale(0.7)
        eq2.next_to(eq1, DOWN).to_edge(indent)

        self.play(
            ReplacementTransform(momentum, eq1),
            ReplacementTransform(energy, eq2),
            run_time=2
        )

        self.wait()

class IntroHandlingSeveralParticles(Scene):
    CONFIG = {
        "simulation_time": 5,
    }
    def construct(self):
        start_time = time.time()
        particles = []
        num_particles = 6
        BOX_THRESHOLD = 0.98
        PARTICLE_THRESHOLD = 0.96
        box = Box(height=5.5, width=5.5).set_color(COBALT_BLUE)
        shift_right = RIGHT * 4
        box.shift(shift_right)
        velocities = [
        LEFT * 1 + UP * 1, RIGHT * 1, LEFT + DOWN * 1, 
        RIGHT + DOWN * 1, RIGHT * 0.5 + DOWN * 0.5, RIGHT * 0.5 + UP * 0.5
        ]
        positions = [
        LEFT * 2 + UP * 1, UP * 1, RIGHT * 2 + UP * 1,
        LEFT * 2 + DOWN * 1, DOWN * 1, RIGHT * 2 + DOWN * 1,
        ]
        colors = [BLUE, PINK, GREEN_SCREEN, RED, ORANGE, YELLOW]
        radius =[0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        for i in range(num_particles):
            particle = Ball(radius=radius[i % len(radius)])
            particle.set_color(color=colors[i % len(radius)])
            particle.id = i
            particle.move_to(positions[i])
            particle.shift(shift_right)
            particle.velocity = velocities[i]
            particles.append(particle)
        
        self.play(
            FadeIn(box)
        )

        def update_particles(particles, dt):
            for i in range(len(particles)):
                particle = particles[i]
                particle.acceleration = np.array((0, 0, 0))
                particle.velocity = particle.velocity + particle.acceleration * dt
                particle.shift(particle.velocity * dt)
                handle_collision_with_box(particle, box, dt)
            
            handle_particle_collisions_opt(particles, dt)

        def handle_collision_with_box(particle, box, dt):
            # Bounce off ground and roof
            if particle.get_bottom() <= box.get_bottom()*BOX_THRESHOLD or \
                    particle.get_top() >= box.get_top()*BOX_THRESHOLD:
                particle.velocity[1] = -particle.velocity[1]
                particle.shift(particle.velocity * dt)
            # Bounce off walls
            if particle.get_left_edge() <= box.get_left_edge() * BOX_THRESHOLD or \
                    particle.get_right_edge() >= box.get_right_edge() * BOX_THRESHOLD :
                particle.velocity[0] = -particle.velocity[0]
                particle.shift(particle.velocity * dt)

        def handle_particle_collisions(particles):
            for particle in particles:
                for other_particle in particles:
                    if particle.id != other_particle.id:
                        dist = np.linalg.norm(particle.get_center() - other_particle.get_center())
                        if dist * PARTICLE_THRESHOLD <= (particle.radius + other_particle.radius):
                            # ball.set_color(random.choice(t_colors))
                            # other_ball.set_color(random.choice(t_colors))
                            v1, v2 = get_response_velocities(particle, other_particle)
                            particle.velocity = v1
                            other_particle.velocity = v2

        def handle_particle_collisions_opt(particles, dt):
            possible_collisions = find_possible_collisions(particles) 
            # print([(a.id, b.id) for a, b in possible_collisions])
            # print(len(possible_collisions))
            for particle, other_particle in possible_collisions:
                if particle.id != other_particle.id:
                    dist = np.linalg.norm(particle.get_center() - other_particle.get_center())
                    if dist * PARTICLE_THRESHOLD <= (particle.radius + other_particle.radius):
                        # ball.set_color(random.choice(t_colors))
                        # other_ball.set_color(random.choice(t_colors))
                        v1, v2 = get_response_velocities(particle, other_particle)
                        particle.velocity = v1
                        other_particle.velocity = v2
                        particle.shift(particle.velocity * dt)
                        other_particle.shift(other_particle.velocity * dt)
        
        def find_possible_collisions(particles):
            # implements the sort and sweep algorithm for broad phase
            # helpful reference: https://github.com/mattleibow/jitterphysics/wiki/Sweep-and-Prune
            axis_list = sorted(particles, key=lambda x: x.get_left()[0])
            active_list = []
            possible_collisions = set()
            for particle in axis_list:
                to_remove = [p for p in active_list if particle.get_left()[0] > p.get_right()[0]]
                for r in to_remove:
                    active_list.remove(r)
                for other_particle in active_list:
                    possible_collisions.add((particle, other_particle))

                active_list.append(particle)
            
            return possible_collisions

        def get_response_velocities(particle, other_particle):
            # https://en.wikipedia.org/wiki/Elastic_collision
            v1 = particle.velocity
            v2 = other_particle.velocity
            m1 = particle.mass
            m2 = other_particle.mass
            x1 = particle.get_center()
            x2 = other_particle.get_center()

            particle_response_v = compute_velocity(v1, v2, m1, m2, x1, x2)
            other_particle_response_v = compute_velocity(v2, v1, m2, m1, x2, x1)
            return particle_response_v, other_particle_response_v

        def compute_velocity(v1, v2, m1, m2, x1, x2):
            return v1 - (2 * m2 / (m1 + m2)) * np.dot(v1 - v2, x1 - x2) / np.linalg.norm(x1 - x2) ** 2 * (x1 - x2)

        def show_all_collisions(particles):
            transforms = []
            end_positions = []
            all_pairs = []
            rows = [UP * 2, UP * 1, UP * 0, DOWN * 1, DOWN * 2]
            columns = [LEFT * 5.5, LEFT * 4, LEFT * 2.5]
            for col in columns:
                for row in rows:
                    end_positions.append(row + col)
            seen = set()
            i = 0
            for p1 in particles:
                for p2 in particles:
                    if p1.id == p2.id or (p2.id, p1.id) in seen:
                        continue
                    pair = VGroup(p1, p2)
                    p1_c = p1.copy().scale(0.7)
                    p2_c = p2.copy().scale(0.7)

                    p2_c.next_to(p1_c, RIGHT, buff=SMALL_BUFF)
                    transform_pair = VGroup(p1_c, p2_c).move_to(end_positions[i])
                    all_pairs.append(transform_pair)

                    transforms.append(TransformFromCopy(pair, transform_pair))
                    i += 1
                    seen.add((p1.id, p2.id))

            return transforms, all_pairs

        particles = VGroup(*particles)
        self.play(
            FadeIn(particles)
        )
        particles.add_updater(update_particles)
        self.add(particles)
        self.wait(16)
        particles.clear_updaters()

        print("--- %s seconds ---" % (time.time() - start_time))
        
        self.wait(3)

        idea = TextMobject("Idea 1: try all pairs of particles").scale(1)
        idea.move_to(UP * 3).to_edge(LEFT * 2)
        self.play(
            Write(idea)
        )

        self.wait()


        transforms, all_pairs = show_all_collisions(particles)
        self.play(
            AnimationGroup(*transforms, lag_ratio=0.05),
            run_time=4
        )

        self.wait()

        surround_rect = SurroundingRectangle(all_pairs[11], color=GREEN_SCREEN, buff=SMALL_BUFF)
        self.play(
            ShowCreation(surround_rect)
        )

        self.wait(3)

        self.play(
            *[FadeOut(p) for p in all_pairs],
            FadeOut(surround_rect)
        )

        self.wait()

class HundredParticleIssue(Scene):
    def construct(self):
        idea = TextMobject("Idea 1: try all pairs of particles").scale(1)
        idea.move_to(UP * 3).to_edge(LEFT * 2)
        self.add(idea)

        indent = LEFT * 2
        hundred = TextMobject("100 particles ", r"$\Rightarrow \enspace \approx$ ", "5000 collision tests")
        hundred.scale(0.8).next_to(idea, DOWN).to_edge(indent)

        self.play(
            Write(hundred[0])
        )

        self.wait()

        self.play(
            Write(hundred[1:])
        )

        self.wait()

        per_frame = TextMobject("5000 collision tests ", "every frame")
        per_frame.scale(0.8).next_to(hundred, DOWN).to_edge(indent)
        self.play(
            TransformFromCopy(hundred[2], per_frame[0])
        )
        self.wait()

        self.play(
            Write(per_frame[1])
        )

        self.wait()

        fps = TextMobject(
            r"At 60 FPS, a 1 min simulation requires" + "\\\\", 
            r"$\approx$ 18 million collison checks"
        ).scale(0.8)

        fps.next_to(per_frame, DOWN).to_edge(indent)
        fps[1].to_edge(indent)
        fps[1][1:10].set_color(YELLOW)

        self.play(
            Write(fps[0]),
            run_time=2
        )

        self.wait()

        self.play(
            Write(fps[1]),
            run_time=1
        )

        self.wait()

        time = TextMobject("Rendering time: 1 hour 40 min")
        time.scale(0.8).next_to(fps, DOWN).to_edge(indent)
        time[0][-10:].set_color(YELLOW)
        self.play(
            Write(time)
        )
        self.wait()

        conclusion = TextMobject("Brute force doesn't scale")
        conclusion.scale(0.8).next_to(conclusion, DOWN).to_edge(indent)

        self.wait()

        cross = Cross(idea)

        self.play(
            ShowCreation(cross)
        )

        self.wait()

class CollisionDetectionFramework(Scene):
    def construct(self):
        framework = TextMobject("Collision Detection Framework").scale(1.2)
        framework.move_to(UP * 3.5)
        h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
        h_line.next_to(framework, DOWN)
        self.play(
            Write(framework),
            ShowCreation(h_line)
        )
        self.wait()

        indent = LEFT * 2

        broad = TextMobject("1. Broad Phase")
        narrow = TextMobject("2. Narrow Phase")
        solve = TextMobject("3. Solve Collision")

        broad.move_to(UP * 2.5)
        broad.to_edge(indent)
        narrow.next_to(broad, DOWN).to_edge(indent)
        solve.next_to(narrow, DOWN).to_edge(indent)

        self.play(
            Write(broad),
        )

        self.play(
            Write(narrow)
        )

        self.play(
            Write(solve)
        )

        self.wait()

        self.play(
            broad.set_color, YELLOW,
            narrow.set_fill, None, 0.5,
            solve.set_fill, None, 0.5,
        )

        self.wait()

        broad_explain = TextMobject("1. Which objects could be colliding?").scale(0.8)
        broad_explain.set_color(YELLOW)
        broad_explain.next_to(solve, DOWN * 2).to_edge(indent)

        self.play(
            Write(broad_explain)
        )

        self.wait()

        narrow_explain = TextMobject("2. Are they actually colliding?").scale(0.8)
        narrow_explain.next_to(broad_explain, DOWN).to_edge(indent)
        narrow_explain.set_color(GREEN_SCREEN)

        solve_explain = TextMobject("3. How do we update dynamics?").scale(0.8)
        solve_explain.next_to(narrow_explain, DOWN).to_edge(indent)
        solve_explain.set_color(BLUE)

        self.play(
            broad.set_fill, None, 0.5,
            narrow.set_fill, GREEN_SCREEN, 1,
            solve.set_fill, None, 0.5,
        )

        self.wait()

        self.play(
            Write(narrow_explain)
        )
        self.wait()

        self.play(
            broad.set_fill, None, 0.5,
            narrow.set_fill, None, 0.5,
            solve.set_fill, BLUE, 1,
        )

        self.wait()

        self.play(
            Write(solve_explain)
        )
        self.wait()

        self.play(
            broad.set_fill, YELLOW, 1,
            narrow.set_fill, GREEN_SCREEN, 1,
            solve.set_fill, BLUE, 1,
        )

        self.wait()

        check_1 = CheckMark()
        check_1.scale(0.15)
        check_1.next_to(narrow_explain, RIGHT, buff=SMALL_BUFF)

        self.add(check_1)

        self.wait()

        check_2 = CheckMark()
        check_2.scale(0.15)
        check_2.next_to(solve_explain, RIGHT, buff=SMALL_BUFF)
        self.add(check_2)

        self.wait()

        broad_phase_explain = TextMobject("Broad Phase Optimization").scale(1.2)
        broad_phase_explain.move_to(UP * 3.5)

        self.play(
            ReplacementTransform(framework, broad_phase_explain),
            ApplyWave(broad),
            run_time=2
        )

        self.wait(5)

class CheckMark(SVGMobject):
    CONFIG = {
        "file_name": "checkmark",
        "fill_opacity": 1,
        "stroke_width": 0,
        "width": 4,
        "propagate_style_to_family": True
    }

    def __init__(self, **kwargs):
        SVGMobject.__init__(self, **kwargs)
        background_circle = Circle(radius=self.width / 2 - SMALL_BUFF * 2)
        background_circle.set_color(WHITE)
        background_circle.set_fill(color=WHITE, opacity=1)
        
        self.add_to_back(background_circle)
        self[1].set_fill(color=CHECKMARK_GREEN)
        self.set_width(self.width)
        self.center()

class SortAndSweep(GraphScene):
    CONFIG = {
        "x_min": -0.5,
        "x_max": 5,
        "x_axis_width": 6.5,
        "y_axis_height": 6.5,
        "graph_origin": DOWN * 2.75 + RIGHT * 0.25,
        "y_min": -0.5,
        "y_max": 5,
        "x_axis_label": None,
        "y_axis_label": None,
    }
    
    def construct(self):
        particles, box = self.setup_scene()
        projections, bounding_boxes = self.show_projections(particles, box) 
        sort_and_sweep = self.show_possible_collisions(particles, projections)
        self.explain_sort_and_sweep(sort_and_sweep, particles, projections)
        self.show_improvements(sort_and_sweep)

    def setup_scene(self):
        particles = []
        num_particles = 6
        box = Box(height=6, width=6).set_color(COBALT_BLUE)
        shift_right = RIGHT * 3
        box.shift(shift_right)
        positions = [
        DOWN * 2, LEFT * 2 + DOWN * 1, RIGHT * 2.5 + UP * 2,
        RIGHT * 1 + DOWN * 1, UP * 2.5 + LEFT * 1.6, RIGHT * 1.5 + DOWN * 1.5,
        ]
        colors = [BLUE, PINK, GREEN_SCREEN, RED, ORANGE, YELLOW]
        radius =[0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        for i in range(num_particles):
            particle = Ball(radius=radius[i % len(radius)])
            particle.set_color(color=colors[i % len(radius)])
            particle.id = i
            particle.move_to(positions[i])
            particle.shift(shift_right)
            particles.append(particle)
        
        particles = VGroup(*particles)
        return particles, box 

    def show_projections(self, particles, box):
        self.play(
            FadeIn(particles),
            FadeIn(box)
        )

        self.wait()

        self.play(
            FadeOut(box)
        )

        self.setup_axes(animate=False)

        #This removes tick marks
        self.x_axis.remove(self.x_axis[1])
        self.y_axis.remove(self.y_axis[1])
        self.play(
            Write(self.axes),
            run_time=2
        )


        bounding_boxes = []

        projections = []

        for particle in particles:

            bounding_box, left_proj, right_proj = self.get_projections(particle)

            bounding_boxes.append(bounding_box)

            projections.extend([left_proj, right_proj])

        self.play(
            *[ShowCreation(b) for b in bounding_boxes]
        )

        self.wait()

        self.play(
            *[ShowCreation(proj) for proj in projections]
        )

        self.wait()

        return projections, bounding_boxes

    def get_projections(self, particle):
        buff = SMALL_BUFF / 2
        bounding_box = SurroundingRectangle(particle, buff=buff)
        bounding_box.set_color(particle.get_color())

        left_start = particle.get_left_edge_v() + DOWN * bounding_box.get_height() / 2
        left_end = RIGHT * left_start[0] + UP * self.graph_origin[1]

        left_start += LEFT * buff
        left_end += LEFT * buff

        left_proj = DashedLine(left_start, left_end)
        left_proj.set_color(particle.get_color())

        right_start = particle.get_right_edge_v() + DOWN * bounding_box.get_height() / 2
        right_end = RIGHT * right_start[0] + UP * self.graph_origin[1]
        
        right_start += RIGHT * buff
        right_end += RIGHT * buff

        right_proj = DashedLine(right_start, right_end)
        right_proj.set_color(particle.get_color())

        return bounding_box, left_proj, right_proj

    def show_possible_collisions(self, particles, projections):
        pink_particle = self.get_mobject_by_color(particles, PINK)
        orange_particle = self.get_mobject_by_color(particles, ORANGE)

        start_projection = self.get_mobject_by_color(projections, PINK)
        
        # Reversing the projection list guarantees we get the right-most projection
        end_projection = self.get_mobject_by_color(projections[::-1], ORANGE)
        
        interval_of_intersections_1 = Line(
            start_projection.get_end(),
            end_projection.get_end(),
        ).set_color(BRIGHT_RED)
        interval_of_intersections_1.set_stroke(width=7)

        self.play(
            ShowCreation(interval_of_intersections_1)
        )

        self.wait()

        text = TextMobject("Possible Collisions")
        text.move_to(UP * 3)
        text.to_edge(LEFT * 2)
        self.play(
            Write(text)
        )

        self.wait()

        pink_copy = pink_particle.copy()
        orange_copy = orange_particle.copy()

        pink_orange = VGroup(orange_copy, pink_copy).arrange_submobjects(RIGHT, buff=SMALL_BUFF)
        pink_orange.next_to(text, DOWN)

        self.play(
            TransformFromCopy(orange_particle, pink_orange[0]),
            TransformFromCopy(pink_particle, pink_orange[1]),
            run_time=2
        )

        self.wait()

        red_particle = self.get_mobject_by_color(particles, RED)
        yellow_particle = self.get_mobject_by_color(particles, YELLOW)

        start_projection = self.get_mobject_by_color(projections, RED)
        
        # Reversing the projection list guarantees we get the right-most projection
        end_projection = self.get_mobject_by_color(projections[::-1], YELLOW)
        
        interval_of_intersections_2 = Line(
            start_projection.get_end(),
            end_projection.get_end(),
        ).set_color(BRIGHT_RED)
        interval_of_intersections_2.set_stroke(width=7)

        self.play(
            ShowCreation(interval_of_intersections_2)
        )

        self.wait()

        red_copy = red_particle.copy()
        yellow_copy = yellow_particle.copy()

        red_yellow = VGroup(yellow_copy, red_copy).arrange_submobjects(RIGHT, buff=SMALL_BUFF)
        red_yellow.next_to(pink_orange, DOWN)

        self.play(
            TransformFromCopy(yellow_particle, red_yellow[0]),
            TransformFromCopy(red_particle, red_yellow[1]),
            run_time=2
        )

        self.wait(3)

        sort_and_sweep = TextMobject("Sweep and Prune")
        sort_and_sweep.move_to(text.get_center()).shift(RIGHT * 0.5)


        self.play(
            FadeOut(red_yellow),
            FadeOut(pink_orange),
            FadeOut(interval_of_intersections_1),
            FadeOut(interval_of_intersections_2),
            ReplacementTransform(text, sort_and_sweep)
        )

        self.wait()

        return sort_and_sweep

    def show_improvements(self, sort_and_sweep):
        improvements = TextMobject("Improvements")
        improvements.next_to(sort_and_sweep, DOWN)
        improvements.shift(DOWN * 3)

        self.play(
            Write(improvements)
        )
        indent = LEFT * 1.5

        situation = TextMobject("100 particles, 60 FPS, 1 min").scale(0.7)
        situation.next_to(improvements, DOWN).to_edge(indent)
        optimized = TextMobject("With optimization: 3 min").scale(0.7)
        optimized.next_to(situation, DOWN).to_edge(indent)
        naive = TextMobject("Without optimization: 1 hour 40 min").scale(0.7)
        naive.next_to(optimized, DOWN).to_edge(indent)

        optimized[0][-4:].set_color(YELLOW)
        naive[0][-10:].set_color(YELLOW)

        self.play(
            Write(situation)
        )

        self.wait()

        self.play(
            Write(optimized)
        )

        self.wait()

        self.play(
            Write(naive)
        )

        self.wait()

    def get_mobject_by_color(self, mobjects, color):
        for mob in mobjects:
            if mob.get_color() == Color(color) or mob.get_color() == color:
                return mob

    def explain_sort_and_sweep(self, sort_and_sweep, particles, projections):
        indent = LEFT * 1.5
        sort_idea = TextMobject("1. Sort particles by one axis").scale(0.7)
        sort_idea.next_to(sort_and_sweep, DOWN).to_edge(indent)

        key_idea = TextMobject(r"2. Maintain current $\textit{active}$ intervals").scale(0.7)
        key_idea.next_to(sort_idea, DOWN).to_edge(indent)
        
        check = TextMobject(
            r"3. Check intersection of particles" + "\\\\",
            "with active intervals"
        ).scale(0.7)

        check.next_to(key_idea, DOWN).to_edge(indent)
        check[1].to_edge(indent + LEFT * 0.8)

        update = TextMobject(r"4. Update active intervals").scale(0.7)
        update.next_to(check, DOWN).to_edge(indent)
        
        self.play(
            Write(sort_idea)
        )

        self.wait()

        particle_list = VGroup(*[p.copy() for p in particles])
        particle_list.arrange_submobjects(RIGHT, buff=SMALL_BUFF * 2)
        particle_list.next_to(update, DOWN * 1.5).shift(RIGHT * 0.5)

        self.play(
            FadeIn(particle_list)
        )

        self.wait()

        sorted_particle_list = VGroup(*[p.copy() for p in sorted(particles, key=lambda x: x.get_left()[0])])
        sorted_particle_list.arrange_submobjects(RIGHT, buff=SMALL_BUFF * 2)
        sorted_particle_list.next_to(update, DOWN * 1.5).shift(RIGHT * 0.5)
        
        translations = {0: 2, 1: 0, 2: 5, 3: 3, 4: 1, 5: 4}

        self.play(
            *[ReplacementTransform(particle_list[i], sorted_particle_list[translations[i]]) for i in range(len(particles))],
            run_time=2,
        )

        self.wait()

        self.play(
            Write(key_idea)
        )

        self.wait()

        self.active = TextMobject("Active").scale(0.8)
        self.active.next_to(sorted_particle_list, DOWN * 3).to_edge(indent)
        self.play(
            Write(self.active)
        )

        surround = SurroundingRectangle(sorted_particle_list, buff=SMALL_BUFF)
        surround.set_color(BRIGHT_RED)
        surround.next_to(self.active, RIGHT)

        self.play(
            ShowCreation(surround)
        )

        self.wait()

        self.show_active_intervals(surround, particles, sorted_particle_list, projections, check, update)

    def show_active_intervals(self, surround, particles, sorted_particle_list, projections, check, update):
        self.previous_line = None
        self.current_line = None
        self.surround_rects = []
        original_particles = sorted(particles, key=lambda x: x.get_left()[0])
        sorted_particles = [p.copy() for p in sorted(particles, key=lambda x: x.get_left()[0])]
        active_list = []
        for i, particle in enumerate(sorted_particles):
            if i == 1:
                self.play(
                    Write(check)
                )
                self.wait()
            
            self.move_particle_to_active(
                particle, active_list, surround, sorted_particles, 
                original_particles, sorted_particle_list, projections
            )

            if i == 1:
                self.play(
                    Write(update)
                )
                self.wait()

        self.wait()

        self.play(
            FadeOut(surround),
            FadeOut(self.previous_line),
            FadeOut(self.active),
            FadeOut(sorted_particle_list),
            FadeOut(active_list[0]),
            *[FadeOut(rect) for rect in self.surround_rects],
        )
        self.wait()

    def move_particle_to_active(self, particle, active_list, surround, sorted_list, original, sorted_particle_list, projections):
        if len(active_list) == 0:
            if not self.previous_line:
                self.current_line = self.get_interval_line(projections, particle, particle)
            self.play(
                particle.move_to, surround.get_left() + RIGHT * particle.radius + RIGHT * SMALL_BUFF
            )
            active_list.append(particle)
            self.play(
                ShowCreation(self.current_line)
            )
            self.previous_line = self.current_line
            return

        if len(active_list) == 2:
            first_color = active_list[0].get_color()
            second_color = active_list[1].get_color()
            group = VGroup(
                self.get_mobject_by_color(sorted_particle_list, first_color),
                self.get_mobject_by_color(sorted_particle_list, second_color),
            )

            surround_rect = SurroundingRectangle(group, color=GREEN_SCREEN, buff=SMALL_BUFF)
            self.surround_rects.append(surround_rect)
            self.play(
                ShowCreation(surround_rect)
            )

        to_remove = []
        animate = True
        for i, other in enumerate(active_list):
            if animate:
                self.play(
                    particle.next_to, other, DOWN * 2, 
                    run_time=2
                )

            current = self.get_mobject_by_color(original, particle.get_color())
            active = self.get_mobject_by_color(original, other.get_color())
            if current.get_left()[0] < active.get_right()[0]:
                position = other.get_center() + RIGHT * (other.radius + 0.5)
            else:
                self.play(
                    FadeOut(other)
                )
                to_remove.append(other)
                if i != len(active_list) - 1:
                    self.play(
                        active_list[i + 1].move_to, surround.get_left() + RIGHT * active_list[i + 1].radius + RIGHT * SMALL_BUFF,
                        run_time=2
                    )
                position = other.get_center()
            animate = False

        self.play(
            particle.move_to, position,
            run_time=2
        )

        for r in to_remove:
            active_list.remove(r)
        active_list.append(particle)

        if len(active_list) == 2:
            self.current_line = self.get_interval_line(projections, active_list[0], active_list[1])
            self.play(
                ReplacementTransform(self.previous_line, self.current_line)
            )
        else:
            self.current_line = self.get_interval_line(projections, particle, particle)
            self.play(
                ReplacementTransform(self.previous_line, self.current_line)
            )
        self.previous_line = self.current_line

    def get_interval_line(self, projections, start_particle, end_particle):
        start_point = self.get_mobject_by_color(projections, start_particle.get_color()).get_end()
        end_point = self.get_mobject_by_color(projections[::-1], end_particle.get_color()).get_end()
        line = Line(start_point, end_point).set_stroke(width=7).set_color(BRIGHT_RED)
        return line

class Grid(VGroup):
    CONFIG = {
        "height": 6.0,
        "width": 6.0,
    }

    def __init__(self, rows, columns, **kwargs):
        digest_config(self, kwargs, locals())
        super().__init__(**kwargs)

        x_step = self.width / self.columns
        y_step = self.height / self.rows

        for x in np.arange(0, self.width + x_step, x_step):
            self.add(Line(
                [x - self.width / 2., -self.height / 2., 0],
                [x - self.width / 2., self.height / 2., 0],
            ))
        for y in np.arange(0, self.height + y_step, y_step):
            self.add(Line(
                [-self.width / 2., y - self.height / 2., 0],
                [self.width / 2., y - self.height / 2., 0]
            ))

class UniformGrid(Scene):
    def construct(self):
        particles, box = self.setup_scene()
        self.play(
            FadeIn(particles),
            FadeIn(box)
        )
        self.show_grid(particles, box)

    def setup_scene(self):
        particles = []
        num_particles = 6
        box = Box(height=6, width=6)
        shift_right = RIGHT * 3
        box.shift(shift_right).set_color(COBALT_BLUE)
        positions = [
        DOWN * 2, LEFT * 2 + DOWN * 1, RIGHT * 2.5 + UP * 2,
        RIGHT * 1 + DOWN * 1, UP * 2.5 + LEFT * 1.6, RIGHT * 1.5 + DOWN * 1.5,
        ]
        colors = [BLUE, PINK, GREEN_SCREEN, RED, ORANGE, YELLOW]
        radius =[0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        for i in range(num_particles):
            particle = Ball(radius=radius[i % len(radius)])
            particle.set_color(color=colors[i % len(radius)])
            particle.id = i
            particle.move_to(positions[i])
            particle.shift(shift_right)
            particles.append(particle)
        
        particles = VGroup(*particles)
        return particles, box 

    def show_grid(self, particles, box):
        space_partitioning = TextMobject("Space Partitions").scale(1.2)

        uniform_grid = TextMobject("Uniform Grid Partition")
        uniform_grid.move_to(UP * 3).to_edge(LEFT * 2)

        space_partitioning.move_to(uniform_grid.get_center())
        self.play(
            Write(space_partitioning)
        )

        self.wait()

        self.play(
            ReplacementTransform(space_partitioning, uniform_grid)
        )

        self.wait()

        grid = Grid(3, 3).set_color(COBALT_BLUE)
        grid.move_to(box.get_center())
        self.play(
            ShowCreation(grid),
            run_time=2
        )

        self.wait()

        indent = LEFT * 2

        mapping = TextMobject(r"(row, col) $\rightarrow$ particles").scale(0.8)
        mapping.next_to(uniform_grid, DOWN)
        mapping.to_edge(LEFT * 2)
        self.play(
            Write(mapping)
        )

        self.wait()

        map_00 = self.make_mapping(0, 0, ORANGE)
        map_00.next_to(mapping, DOWN).to_edge(indent)
        self.play(
            Write(map_00)
        )

        self.wait()

        map_01 = self.make_mapping(0, 1, None)
        map_01.next_to(map_00, DOWN).to_edge(indent)
        self.play(
            Write(map_01)
        )

        self.wait()

        map_02 = self.make_mapping(0, 2, GREEN_SCREEN)
        map_02.next_to(map_01, DOWN).to_edge(indent)
        self.play(
            Write(map_02)
        )

        self.wait()

        map_10 = self.make_mapping(1, 0, PINK)
        map_10.next_to(map_02, DOWN).to_edge(indent)
        self.play(
            Write(map_10)
        )

        self.wait()

        map_11 = self.make_mapping(1, 1, RED)
        map_11.next_to(map_10, DOWN).to_edge(indent)
        self.play(
            Write(map_11)
        )

        self.wait()

        map_12 = self.make_mapping(1, 2, RED)
        map_12.next_to(map_11, DOWN).to_edge(indent)
        self.play(
            Write(map_12)
        )

        self.wait()

        map_20 = self.make_mapping(2, 0, PINK)
        map_20.next_to(map_12, DOWN).to_edge(indent)
        self.play(
            Write(map_20)
        )

        self.wait()

        map_21 = self.make_mapping(2, 1, BLUE)
        map_21.next_to(map_20, DOWN).to_edge(indent)
        add_particle = self.make_particle(RED).next_to(map_21, RIGHT, buff=MED_SMALL_BUFF)
        map_21.add(add_particle)
        self.play(
            Write(map_21)
        )

        self.wait()


        map_22 = self.make_mapping(2, 2, YELLOW)
        map_22.next_to(map_21, DOWN).to_edge(indent)
        add_particle = self.make_particle(RED).next_to(map_22, RIGHT, buff=MED_SMALL_BUFF)
        map_22.add(add_particle)
        self.play(
            Write(map_22)
        )

        self.wait()

        possible_collisions = TextMobject("Possible Collisions").scale(0.9)

        possible_collisions_br = VGroup(
            self.get_mobject_by_color(particles, BLUE).copy(),
            self.get_mobject_by_color(particles, RED).copy(),
        ).arrange_submobjects(RIGHT, buff=SMALL_BUFF * 2)
        possible_collisions_br.next_to(grid, LEFT)
        possible_collisions_br.shift(LEFT * 1.5 + UP * 0.5)

        possible_collisions.next_to(possible_collisions_br, UP)

        possible_collisions_ry = VGroup(
            self.get_mobject_by_color(particles, RED).copy(),
            self.get_mobject_by_color(particles, YELLOW).copy(),
        ).arrange_submobjects(RIGHT, buff=SMALL_BUFF * 2)
        possible_collisions_ry.next_to(possible_collisions_br, DOWN)

        self.play(
            Write(possible_collisions)
        )

        self.wait()

        self.play(
            FadeIn(possible_collisions_br),
            FadeIn(possible_collisions_ry)
        )

        self.wait(3)

        self.play(
            FadeOut(map_00),
            FadeOut(map_10),
            FadeOut(map_20),
            FadeOut(map_01),
            FadeOut(map_11),
            FadeOut(map_21),
            FadeOut(map_02),
            FadeOut(map_12),
            FadeOut(map_22),
            FadeOut(possible_collisions),
            FadeOut(possible_collisions_br),
            FadeOut(possible_collisions_ry),
        )

        self.wait()

        new_grid_22 = Grid(2, 2).set_color(COBALT_BLUE)
        new_grid_22.move_to(box.get_center())

        self.play(
            ReplacementTransform(grid, new_grid_22)
        )

        self.wait()

        map_00 = self.make_mapping(0, 0, ORANGE)
        map_00.next_to(mapping, DOWN).to_edge(indent)
        

        map_01 = self.make_mapping(0, 1, GREEN_SCREEN)
        map_01.next_to(map_00, DOWN).to_edge(indent)



        map_10 = self.make_mapping(1, 0, PINK)
        map_10.next_to(map_01, DOWN).to_edge(indent)
        add_particle = self.make_particle(BLUE).next_to(map_10, RIGHT, buff=MED_SMALL_BUFF)
        map_10.add(add_particle)


        map_11 = self.make_mapping(1, 1, YELLOW)
        map_11.next_to(map_10, DOWN).to_edge(indent)
        add_particle_1 = self.make_particle(RED).next_to(map_11, RIGHT, buff=MED_SMALL_BUFF)
        map_11.add(add_particle_1)
        add_particle_2 = self.make_particle(BLUE).next_to(map_11, RIGHT, buff=MED_SMALL_BUFF)
        map_11.add(add_particle_2)

        self.play(
            Write(map_00),
            Write(map_01),
            Write(map_10),
            Write(map_11)
        )

        self.wait()

        possible_collisions.next_to(uniform_grid, DOWN * 14)
        self.play(
            Write(possible_collisions)
        )

        possible_collisions_br.next_to(possible_collisions, DOWN)
        possible_collisions_ry.next_to(possible_collisions_br, DOWN)

        possible_collisions_pb = VGroup(
            self.get_mobject_by_color(particles, BLUE).copy(),
            self.get_mobject_by_color(particles, PINK).copy(),
        ).arrange_submobjects(RIGHT, buff=SMALL_BUFF * 2)
        possible_collisions_pb.next_to(possible_collisions_br, LEFT).shift(LEFT * 1)
        
        possible_collisions_by = VGroup(
            self.get_mobject_by_color(particles, BLUE).copy(),
            self.get_mobject_by_color(particles, YELLOW).copy(),
        ).arrange_submobjects(RIGHT, buff=SMALL_BUFF * 2)
        possible_collisions_by.next_to(possible_collisions_pb, DOWN * 1.3)

        all_possible_collisions = VGroup(
            possible_collisions_pb, possible_collisions_by,
            possible_collisions_ry, possible_collisions_br,
        )

        all_possible_collisions.next_to(possible_collisions, DOWN)

        self.play(
            FadeIn(all_possible_collisions)
        )

        self.wait()

        self.play(
            FadeOut(mapping),
            FadeOut(map_00),
            FadeOut(map_11),
            FadeOut(map_10),
            FadeOut(map_01),
            FadeOut(all_possible_collisions),
            FadeOut(possible_collisions),
        )

        self.wait()

        new_grid_55 = Grid(5, 5).set_color(COBALT_BLUE)
        new_grid_55.move_to(box.get_center())

        self.play(
            ReplacementTransform(new_grid_22, new_grid_55)
        )

        self.wait()

        possible_collisions.next_to(uniform_grid, DOWN * 10)

        self.play(
            Write(possible_collisions)
        )

        possible_collisions_ry.next_to(possible_collisions, DOWN)

        self.play(
            FadeIn(possible_collisions_ry)
        )

        self.wait()

    def make_particle(self, color):
        return Ball(radius=0.15, color=color).set_fill(color=color, opacity=1)

    def make_mapping(self, row, col, color):
        mapping = TextMobject(r"({0}, {1}) $\rightarrow$ ".format(row, col)).scale(0.7)
        if not color:
            return mapping
        particle = self.make_particle(color)
        particle.next_to(mapping, RIGHT, buff=MED_SMALL_BUFF) 
        return VGroup(mapping, particle)

    def get_mobject_by_color(self, mobjects, color):
        for mob in mobjects:
            if mob.get_color() == Color(color) or mob.get_color() == color:
                return mob

class KDTree(Scene):
    def construct(self):
        particles, box = self.setup_scene()
        self.play(
            FadeIn(particles),
            FadeIn(box)
        )

        self.show_kd_tree(particles, box)

    def setup_scene(self):
        particles = []
        num_particles = 6
        box = Box(height=6, width=6).set_color(COBALT_BLUE)
        shift_right = RIGHT * 3
        box.shift(shift_right)
        positions = [
        DOWN * 2, LEFT * 2 + DOWN * 1, RIGHT * 2.5 + UP * 2,
        RIGHT * 1 + DOWN * 1, UP * 2.5 + LEFT * 1.6, RIGHT * 1.5 + DOWN * 1.5,
        ]
        colors = [BLUE, PINK, GREEN_SCREEN, RED, ORANGE, YELLOW]
        radius =[0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        for i in range(num_particles):
            particle = Ball(radius=radius[i % len(radius)])
            particle.set_color(color=colors[i % len(radius)])
            particle.id = i
            particle.move_to(positions[i])
            particle.shift(shift_right)
            particles.append(particle)
        
        particles = VGroup(*particles)
        
        return particles, box

    def show_kd_tree(self, particles, box):
        smarter = TextMobject("Smarter Space Partitioning")
        kd_tree = TextMobject("K-D Trees")
        smarter.move_to(LEFT * 3.7 + UP * 3.5)
        kd_tree.next_to(smarter, DOWN)

        self.play(
            Write(smarter)
        )

        self.wait()

        self.play(
            Write(kd_tree)
        )

        self.wait()

        indent = LEFT * 1

        step_1 = TextMobject("1. Select axis (e.g y-axis)").scale(0.7)
        step_1.next_to(kd_tree, DOWN)
        step_1.to_edge(indent)

        step_2 = TextMobject("2. Split space along median").scale(0.7)
        step_2.next_to(step_1, DOWN)
        step_2.to_edge(indent)

        step_3 = TextMobject("3. Repeat along other axis (e.g x-axis)").scale(0.7)
        step_3.next_to(step_2, DOWN)
        step_3.to_edge(indent)

        step_4 = TextMobject("4. Continue until termination condition").scale(0.7)
        step_4.next_to(step_3, DOWN)
        step_4.to_edge(indent)

        steps = [step_1, step_2, step_3, step_4]

        splits = []

        root_split = self.get_first_split(particles, box)
        splits.append(root_split)
        first_split = root_split.get_start()[1]
        
        a = TexMobject("A").scale(0.8)
        a.next_to(root_split, RIGHT)
        

        bottom_particles = [self.get_mobject_by_color(particles, color) for color in [PINK, BLUE, RED, YELLOW]]
        second_split = self.get_second_split(bottom_particles, box, first_split)
        splits.append(second_split)

        b = TexMobject("B").scale(0.8)
        b.next_to(second_split, DOWN)

        top_particles = [self.get_mobject_by_color(particles, color) for color in [PINK, ORANGE, RED, GREEN_SCREEN]]
        third_split = self.get_third_split(top_particles, box, first_split)
        splits.append(third_split)

        c = TexMobject("C").scale(0.8)
        c.next_to(third_split, UP)

        second_split_coord = second_split.get_start()[0]
        third_split_coord = third_split.get_start()[0]

        top_left_particles = [self.get_mobject_by_color(particles, color) for color in [PINK, ORANGE]]
        top_left_split = self.get_top_left_split(top_left_particles, box, third_split)

        splits.append(top_left_split)

        d = TexMobject("D").scale(0.8)
        d.next_to(top_left_split, LEFT)

        top_right_particles = [self.get_mobject_by_color(particles, color) for color in [RED, GREEN_SCREEN]]
        top_right_split = self.get_top_right_split(top_right_particles, box, third_split)

        splits.append(top_right_split)

        e = TexMobject("E").scale(0.8)
        e.next_to(top_right_split, RIGHT)

        bottom_left_particles = [self.get_mobject_by_color(particles, color) for color in [PINK, BLUE]]
        bottom_left_split = self.get_bottom_left_split(bottom_left_particles, box, second_split)

        splits.append(bottom_left_split)

        f = TexMobject("F").scale(0.8)
        f.next_to(bottom_left_split, LEFT)

        split_labels = [a, b, c, d, e, f]

        grid_labels = self.get_grid_labels(splits, particles, box)

        tree, edge_dict = self.make_tree()

        self.animate_all(steps, splits, split_labels, tree, edge_dict, grid_labels, particles)
    
    def animate_all(self, steps, splits, split_labels, tree, edge_dict, grid_labels, particles):
        self.play(
            Write(steps[0])
        )
        self.wait()

        self.play(
            Write(steps[1])
        )

        self.wait()

        sorted_particles_list = [self.get_mobject_by_color(particles, color).copy() for color in [ORANGE, GREEN_SCREEN, PINK, RED, YELLOW, BLUE]]
        sorted_particles = VGroup(*sorted_particles_list)
        sorted_particles.arrange_submobjects(RIGHT)
        sorted_particles.move_to(DOWN * 3 + LEFT * 3.7)
        transforms = []
        for sort_p in sorted_particles:
            orig = self.get_mobject_by_color(particles, sort_p.get_color())
            transforms.append(
                TransformFromCopy(orig, sort_p)
            )

        self.play(
            AnimationGroup(*transforms, lag_ratio=0.05),
            run_time=2
        )

        self.wait()

        midpoint = (sorted_particles[2].get_right() + sorted_particles[3].get_left()) / 2
        midpoint_line = Line(UP * 0.5, DOWN * 0.5).set_color(splits[0].get_color())
        midpoint_line.move_to(midpoint)

        self.play(
            ShowCreation(midpoint_line)
        )


        self.wait()

        self.play(
            Write(splits[0])
        )

        self.play(
            Write(split_labels[0])
        )

        self.wait()

        self.play(
            FadeOut(sorted_particles),
            FadeOut(midpoint_line),
            Write(tree[0])
        )

        self.wait()

        edge_dict[(0, 1)].set_color(splits[0].get_color())
        edge_dict[(0, 2)].set_color(splits[0].get_color())

        self.play(
            ShowCreation(edge_dict[(0, 1)]),
            ShowCreation(edge_dict[(0, 2)]),
        )
        self.wait()

        self.play(
            Write(steps[2])
        )

        self.wait()

        sorted_particles_list = [self.get_mobject_by_color(particles, color).copy() for color in [PINK, BLUE, RED, YELLOW]]
        sorted_particles = VGroup(*sorted_particles_list)
        sorted_particles.arrange_submobjects(RIGHT)
        sorted_particles.move_to(DOWN * 3 + LEFT * 3.7)
        transforms = []
        for sort_p in sorted_particles:
            orig = self.get_mobject_by_color(particles, sort_p.get_color())
            transforms.append(
                TransformFromCopy(orig, sort_p)
            )

        self.play(
            AnimationGroup(*transforms, lag_ratio=0.05),
            run_time=2
        )

        self.wait()

        midpoint = (sorted_particles[1].get_right() + sorted_particles[2].get_left()) / 2
        midpoint_line = Line(UP * 0.5, DOWN * 0.5).set_color(splits[1].get_color())
        midpoint_line.move_to(midpoint)

        self.play(
            ShowCreation(midpoint_line)
        )


        self.wait()

        self.play(
            Write(splits[1])
        )

        self.play(
            Write(split_labels[1])
        )

        self.wait()

        self.play(
            Write(tree[1]),
            FadeOut(sorted_particles),
            FadeOut(midpoint_line),
        )

        edge_dict[(1, 3)].set_color(splits[1].get_color())
        edge_dict[(1, 4)].set_color(splits[1].get_color())

        self.play(
            ShowCreation(edge_dict[(1, 3)]),
            ShowCreation(edge_dict[(1, 4)]),
        )
        self.wait()


        self.play(
            Write(splits[2])
        )

        self.play(
            Write(split_labels[2])
        )

        self.play(
            Write(tree[2])
        )

        edge_dict[(2, 5)].set_color(splits[2].get_color())
        edge_dict[(2, 6)].set_color(splits[2].get_color())

        self.play(
            ShowCreation(edge_dict[(2, 5)]),
            ShowCreation(edge_dict[(2, 6)]),
        )
        self.wait()

        self.play(
            Write(steps[3])
        )

        self.wait()

        self.play(
            Write(splits[3])
        )

        self.play(
            Write(split_labels[3])
        )

        self.play(
            Write(tree[5])
        )

        edge_dict[(5, 9)].set_color(splits[3].get_color())
        edge_dict[(5, 10)].set_color(splits[3].get_color())

        self.play(
            ShowCreation(edge_dict[(5, 9)]),
            ShowCreation(edge_dict[(5, 10)]),
        )
        self.wait()

        self.play(
            Write(grid_labels[0]),
            Write(grid_labels[2]),
        )

        self.play(
            Write(tree[9]),
            Write(tree[10])
        )

        self.wait()

        self.play(
            Write(splits[4])
        )

        self.play(
            Write(split_labels[4])
        )

        self.play(
            Write(tree[6])
        )

        edge_dict[(6, 11)].set_color(splits[4].get_color())
        edge_dict[(6, 12)].set_color(splits[4].get_color())

        self.play(
            ShowCreation(edge_dict[(6, 11)]),
            ShowCreation(edge_dict[(6, 12)]),
        )
        self.wait()

        self.play(
            Write(grid_labels[1]),
            Write(grid_labels[3]),
        )

        self.play(
            Write(tree[11]),
            Write(tree[12])
        )

        self.wait()

        self.play(
            Write(splits[5])
        )

        self.play(
            Write(split_labels[5])
        )

        self.play(
            Write(tree[3])
        )

        edge_dict[(3, 7)].set_color(splits[5].get_color())
        edge_dict[(3, 8)].set_color(splits[5].get_color())

        self.play(
            ShowCreation(edge_dict[(3, 7)]),
            ShowCreation(edge_dict[(3, 8)]),
        )
        self.wait()

        self.play(
            Write(grid_labels[4]),
            Write(grid_labels[6]),
        )

        self.play(
            Write(tree[7]),
            Write(tree[8])
        )

        self.wait()

        self.play(
            Write(grid_labels[5])
        )
        self.wait()

        self.play(
            Write(tree[4]),
            run_time=2
        )

        self.wait()

        self.play(
            ApplyWave(tree[4]),
            run_time=1
        )

        self.wait()

    def get_grid_labels(self, splits, particles, box):
        labels = [TexMobject(str(i)).scale(0.7) for i in range(1, 8)]
        labels[0].next_to(self.get_mobject_by_color(splits, YELLOW_D), UP)
        labels[0].shift(UP * 0.7)

        labels[1].next_to(self.get_mobject_by_color(splits, RED_E), UP)
        labels[1].shift(UP * 0.8)

        labels[2].next_to(self.get_mobject_by_color(splits, YELLOW_D), DOWN)
        labels[2].shift(DOWN * 0.5)

        labels[3].next_to(self.get_mobject_by_color(splits, RED_E), DOWN)
        labels[3].shift(DOWN * 0.4)

        labels[4].next_to(self.get_mobject_by_color(splits, BLUE_E), UP, buff=SMALL_BUFF)

        labels[5].next_to(self.get_mobject_by_color(splits, GOLD), RIGHT)
        labels[5].shift(RIGHT * 1 + DOWN * 0.2)

        labels[6].next_to(self.get_mobject_by_color(splits, BLUE_E), DOWN)
        labels[6].shift(DOWN * 0.4)

        return labels

    def make_tree(self):
        tree = VGroup()
        labels = [TexMobject(chr(ord('A') + i)).scale(0.8) for i in range(6)]
        labels[0].move_to(LEFT * 3.7)
        labels[1].next_to(labels[0], LEFT * 5 + DOWN * 1)
        labels[2].next_to(labels[0], RIGHT * 5 + DOWN * 1)
        labels[5].next_to(labels[1], LEFT * 2.5 + DOWN * 4)
        for i in range(3):
            tree.add(labels[i])

        tree.add(labels[5])

        red_yellow = VGroup(
            self.make_particle(RED), 
            self.make_particle(YELLOW)
        ).arrange_submobjects(RIGHT, buff=SMALL_BUFF * 2)
        box = SurroundingRectangle(red_yellow, buff=SMALL_BUFF).set_color(WHITE)
        label_6 = TextMobject("6").scale(0.7).next_to(box, DOWN, buff=SMALL_BUFF * 2)
        leaf_6 = VGroup(red_yellow, box, label_6)
        leaf_6.next_to(labels[1], RIGHT * 0.5 + DOWN * 4)
        tree.add(leaf_6)

        labels[3].next_to(labels[2], LEFT * 2 + DOWN * 3.9)
        tree.add(labels[3])

        labels[4].next_to(labels[2], RIGHT * 2 + DOWN * 3.9)
        tree.add(labels[4])

        blue = self.make_particle(BLUE)
        box = SurroundingRectangle(blue, buff=SMALL_BUFF).set_color(WHITE)
        label_7 = TextMobject("7").scale(0.7).next_to(box, DOWN, buff=SMALL_BUFF * 2)
        leaf_7 = VGroup(blue, box, label_7)
        leaf_7.next_to(labels[5], LEFT * 0.1 + DOWN * 4)
        tree.add(leaf_7)

        pink_down = self.make_particle(PINK)
        box = SurroundingRectangle(pink_down, buff=SMALL_BUFF).set_color(WHITE)
        label_5 = TextMobject("5").scale(0.7).next_to(box, DOWN, buff=SMALL_BUFF * 2)
        leaf_5 = VGroup(pink_down, box, label_5)
        leaf_5.next_to(labels[5], RIGHT * 0.1 + DOWN * 4)
        tree.add(leaf_5)

        pink_up = self.make_particle(PINK)
        box = SurroundingRectangle(pink_up, buff=SMALL_BUFF).set_color(WHITE)
        label_3 = TextMobject("3").scale(0.7).next_to(box, DOWN, buff=SMALL_BUFF * 2)
        leaf_3 = VGroup(pink_up, box, label_3)
        leaf_3.next_to(labels[3], LEFT * 0.1 + DOWN * 4)
        tree.add(leaf_3)

        orange = self.make_particle(ORANGE)
        box = SurroundingRectangle(orange, buff=SMALL_BUFF).set_color(WHITE)
        label_1 = TextMobject("1").scale(0.7).next_to(box, DOWN, buff=SMALL_BUFF * 2)
        leaf_1 = VGroup(orange, box, label_1)
        leaf_1.next_to(labels[3], RIGHT * 0.1 + DOWN * 4)
        tree.add(leaf_1)


        red = self.make_particle(RED)
        box = SurroundingRectangle(red, buff=SMALL_BUFF).set_color(WHITE)
        label_4 = TextMobject("4").scale(0.7).next_to(box, DOWN, buff=SMALL_BUFF * 2)
        leaf_4 = VGroup(red, box, label_4)
        leaf_4.next_to(labels[4], LEFT * 0.1 + DOWN * 4)
        tree.add(leaf_4)

        green = self.make_particle(GREEN_SCREEN)
        box = SurroundingRectangle(green, buff=SMALL_BUFF).set_color(WHITE)
        label_2 = TextMobject("2").scale(0.7).next_to(box, DOWN, buff=SMALL_BUFF * 2)
        leaf_2 = VGroup(green, box, label_2)
        leaf_2.next_to(labels[4], RIGHT * 0.1 + DOWN * 4)
        tree.add(leaf_2)

        edge_dict = self.get_tree_edges(tree)

        return tree, edge_dict

    def get_tree_edges(self, tree):
        edge_set = [
        (0, 1), (0, 2), 
        (1, 3), (1, 4), (2, 5), (2, 6),
        (3, 7), (3, 8), (5, 9), (5, 10), (6, 11), (6, 12),
        ]
        edge_dict = {}
        for u, v in edge_set:
            edge_dict[(u, v)] = self.make_edge(tree[u], tree[v])

        return edge_dict

    def make_edge(self, u, v):
        start = u.get_center()
        if isinstance(v, TexMobject):
            end = v.get_center()
        else:
            end = v.get_top()

        unit_v = (end - start) / np.linalg.norm((end - start))
        adjust_start = start + unit_v * 0.3
        adjust_end = end - unit_v * 0.3
        return Line(adjust_start, adjust_end).set_stroke(width=5)

    def get_first_split(self, particles, box):
        first_y_split = self.find_median_point(particles, axis=1)
        root_split_start = box.get_left() * RIGHT + first_y_split * UP
        root_split_end = box.get_right() * RIGHT + first_y_split * UP
        root_split = Line(root_split_start, root_split_end)
        root_split.set_color(TEAL)
        root_split.set_stroke(width=7)
        return root_split

    def get_second_split(self, particles, box, first_split):
        second_x_split = self.find_median_point(particles)
        split_start = second_x_split * RIGHT + first_split * UP
        split_end = second_x_split * RIGHT + box.get_bottom() * UP
        split_line = Line(split_start, split_end)
        split_line.set_color(GOLD)
        split_line.set_stroke(width=7)
        return split_line

    def get_third_split(self, particles, box, first_split):
        third_x_split = self.find_median_point(particles)
        split_start = third_x_split * RIGHT + first_split * UP
        split_end = third_x_split * RIGHT + box.get_top() * UP
        split_line = Line(split_start, split_end)
        split_line.set_color(PURPLE)
        split_line.set_stroke(width=7)
        return split_line

    def get_top_left_split(self, particles, box, third_split):
        split = self.find_median_point(particles, axis=1)
        top_left_split_s = third_split.get_start()[0] * RIGHT + split * UP
        top_left_split_e = box.get_left() * RIGHT + split * UP
        split_line = Line(top_left_split_s, top_left_split_e)
        split_line.set_color(YELLOW_D)
        split_line.set_stroke(width=7)
        return split_line

    def get_top_right_split(self, particles, box, third_split):
        split = self.find_median_point(particles, axis=1)
        top_right_split_s = third_split.get_start()[0] * RIGHT + split * UP
        top_right_split_e = box.get_right() * RIGHT + split * UP
        split_line = Line(top_right_split_s, top_right_split_e)
        split_line.set_color(RED_E)
        split_line.set_stroke(width=7)
        return split_line

    def get_bottom_left_split(self, particles, box, second_split):
        split = self.find_median_point(particles, axis=1)
        bottom_left_split_s = second_split.get_start()[0] * RIGHT + split * UP
        bottom_left_split_e = box.get_left() * RIGHT + split * UP
        split_line = Line(bottom_left_split_s, bottom_left_split_e)
        split_line.set_color(BLUE_E)
        split_line.set_stroke(width=7)
        return split_line


    def find_median_point(self, objects, axis=0):
        # 0 for x-axis, 1 for y-axis
        return np.median([obj.get_center() for obj in objects], axis=0)[axis]

    def get_mobject_by_color(self, mobjects, color):
        for mob in mobjects:
            if mob.get_color() == Color(color) or mob.get_color() == color:
                return mob

    def make_particle(self, color):
        return Ball(radius=0.15, color=color).set_fill(color=color, opacity=1)

class BVH(Scene):
    def construct(self):
        particles, box = self.setup_scene()
        self.play(
            FadeIn(particles),
            FadeIn(box)
        )

        self.show_bvh(particles, box)

    def setup_scene(self):
        particles = []
        num_particles = 6
        box = Box(height=6, width=6).set_color(COBALT_BLUE)
        shift_right = RIGHT * 3
        box.shift(shift_right)
        positions = [
        DOWN * 2, LEFT * 2 + DOWN * 1, RIGHT * 2.5 + UP * 2,
        RIGHT * 1 + DOWN * 1, UP * 2.5 + LEFT * 1.6, RIGHT * 1.5 + DOWN * 1.5,
        ]
        colors = [BLUE, PINK, GREEN_SCREEN, RED, ORANGE, YELLOW]
        radius =[0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        for i in range(num_particles):
            particle = Ball(radius=radius[i % len(radius)])
            particle.set_color(color=colors[i % len(radius)])
            particle.id = i
            particle.move_to(positions[i])
            particle.shift(shift_right)
            particles.append(particle)
        
        particles = VGroup(*particles)
        
        return particles, box

    def show_bvh(self, particles, box):
        steps = self.define_steps()
        bounding_volumes, bounding_labels = self.get_bounding_volumes(particles)
        tree, edge_dict, edge_set = self.make_tree()
        self.animate_all(steps, bounding_volumes, bounding_labels, tree, edge_dict, edge_set, particles)

    def define_steps(self):
        steps = []
        object_partition = TextMobject("Object Partitions")
        object_partition.move_to(LEFT * 3.6 + UP * 3.5)
        idea = TextMobject("Idea: no objects overlap spaces").scale(0.9)
        idea.next_to(object_partition, DOWN).shift(UP * SMALL_BUFF * 0.5)


        steps.extend([object_partition, idea])

        indent = LEFT * 1.5
        
        step_1 = TextMobject("1. Pick axis").scale(0.7)
        step_1.next_to(idea, DOWN * 0.8).to_edge(indent)
        
        step_2 = TextMobject("2. Pick median object along axis").scale(0.7)
        step_2.next_to(step_1, DOWN * 0.7).to_edge(indent)
        
        step_3 = TextMobject("3. Split objects into two sets").scale(0.7)
        step_3.next_to(step_2, DOWN * 0.6).to_edge(indent)

        step_4 = TextMobject("4. Define bounding boxes/volumes").scale(0.7)
        step_4.next_to(step_3, DOWN * 0.6).to_edge(indent)

        step_5 = TextMobject("5. Repeat until termination condition").scale(0.7)
        step_5.next_to(step_4, DOWN * 0.6).to_edge(indent)

        steps.extend([step_1, step_2, step_3, step_4, step_5])

        replace_object = TextMobject("Bounding Volume Hierarchies")
        replace_object.move_to(object_partition.get_center())
        steps.append(replace_object)

        return VGroup(*steps)

    def get_bounding_volumes(self, particles):
        bounding_volumes = []
        bounding_labels = [TexMobject(chr(ord('A') + i)).scale(0.8) for i in range(4)]
        first_level_left_particles = VGroup(*[self.get_mobject_by_color(particles, color) for color in [PINK, BLUE, ORANGE]])
        first_level_right_particles = VGroup(*[self.get_mobject_by_color(particles, color) for color in [GREEN_SCREEN, RED, YELLOW]])
        
        top_left_bb = SurroundingRectangle(first_level_left_particles, buff=SMALL_BUFF * 2)
        top_left_bb.set_color(GOLD)
        bounding_labels[0].move_to(top_left_bb.get_center())

        top_right_bb = SurroundingRectangle(first_level_right_particles, buff=SMALL_BUFF * 2)
        top_right_bb.set_color(TEAL)
        bounding_labels[1].move_to(top_right_bb.get_center())

        bounding_volumes.extend([top_left_bb, top_right_bb])

        second_level_top_left_particles = self.get_mobject_by_color(particles, ORANGE)
        second_level_bottom_left_particles = VGroup(*[self.get_mobject_by_color(particles, color) for color in [PINK, BLUE]])

        second_top_left_bb = SurroundingRectangle(second_level_top_left_particles, buff=SMALL_BUFF)
        second_top_left_bb.set_color(WHITE)

        second_bottom_left_bb = SurroundingRectangle(second_level_bottom_left_particles, buff=SMALL_BUFF * 1.3)
        second_bottom_left_bb.set_color(BLUE_E)
        bounding_labels[2].move_to(second_bottom_left_bb.get_center())

        bounding_volumes.extend([second_top_left_bb, second_bottom_left_bb])

        second_level_top_right_particles = self.get_mobject_by_color(particles, GREEN_SCREEN)
        second_level_bottom_right_particles = VGroup(*[self.get_mobject_by_color(particles, color) for color in [RED, YELLOW]])

        second_top_right_bb = SurroundingRectangle(second_level_top_right_particles, buff=SMALL_BUFF)
        second_top_right_bb.set_color(WHITE)

        second_bottom_right_bb = SurroundingRectangle(second_level_bottom_right_particles, buff=SMALL_BUFF * 1.3)
        second_bottom_right_bb.set_color(YELLOW_D)
        bounding_labels[3].move_to(second_bottom_right_bb.get_center() + UP * 0.5 + RIGHT * 0.5)

        bounding_volumes.extend([second_top_right_bb, second_bottom_right_bb])

        pink_leaf = SurroundingRectangle(self.get_mobject_by_color(particles, PINK), buff=SMALL_BUFF * 0.6)
        pink_leaf.set_color(WHITE)

        blue_leaf = SurroundingRectangle(self.get_mobject_by_color(particles, BLUE), buff=SMALL_BUFF * 0.6)
        blue_leaf.set_color(WHITE)

        bounding_volumes.extend([pink_leaf, blue_leaf])

        red_leaf = SurroundingRectangle(self.get_mobject_by_color(particles, RED), buff=SMALL_BUFF * 0.6)
        red_leaf.set_color(WHITE)

        yellow_leaf = SurroundingRectangle(self.get_mobject_by_color(particles, YELLOW), buff=SMALL_BUFF * 0.6)
        yellow_leaf.set_color(WHITE)

        bounding_volumes.extend([red_leaf, yellow_leaf])

        return bounding_volumes, bounding_labels

    def make_tree(self):
        tree = VGroup()
        labels = [TexMobject(chr(ord('A') + i)).scale(0.8) for i in range(4)]
        
        root = SurroundingRectangle(labels[0], buff=SMALL_BUFF).set_color(COBALT_BLUE)
        root.set_color(COBALT_BLUE)
        root.set_fill(color=COBALT_BLUE, opacity=1)
        root.move_to(LEFT * 3.7)
        tree.add(root)

        labels[0].next_to(root, LEFT * 5 + DOWN * 1)
        labels[1].next_to(root, RIGHT * 5 + DOWN * 1)
        
        node_0 = SurroundingRectangle(labels[0], buff=SMALL_BUFF, color=GOLD)
        node_0 = VGroup(node_0, labels[0])

        node_1 = SurroundingRectangle(labels[1], buff=SMALL_BUFF, color=TEAL)
        node_1 = VGroup(node_1, labels[1])
        
        tree.add(node_0)
        tree.add(node_1)

        labels[2].next_to(node_0, LEFT * 2 + DOWN * 3.8)
        node_2 = SurroundingRectangle(labels[2], buff=SMALL_BUFF, color=BLUE_E)
        node_2 = VGroup(node_2, labels[2])

        tree.add(node_2)

        orange_leaf = self.make_particle(ORANGE)
        orange_leaf_surround = SurroundingRectangle(orange_leaf, buff=SMALL_BUFF, color=WHITE)
        orange_leaf = VGroup(orange_leaf, orange_leaf_surround)
        orange_leaf.next_to(node_0, RIGHT * 1 + DOWN * 3.35)
        tree.add(orange_leaf)

        labels[3].next_to(node_1, LEFT * 2 + DOWN * 3.8)
        node_3 = SurroundingRectangle(labels[3], buff=SMALL_BUFF, color=YELLOW_D)
        node_3 = VGroup(node_3, labels[3])

        tree.add(node_3)

        green_leaf = self.make_particle(GREEN_SCREEN)
        green_leaf_surround = SurroundingRectangle(green_leaf, buff=SMALL_BUFF, color=WHITE)
        green_leaf = VGroup(green_leaf, green_leaf_surround)
        green_leaf.next_to(node_1, RIGHT * 1 + DOWN * 3.3)
        tree.add(green_leaf)

        pink_leaf = self.make_particle(PINK)
        pink_leaf_surround = SurroundingRectangle(pink_leaf, buff=SMALL_BUFF, color=WHITE)
        pink_leaf = VGroup(pink_leaf, pink_leaf_surround)
        pink_leaf.next_to(node_2, LEFT * 0.1 + DOWN * 4)
        tree.add(pink_leaf)

        blue_leaf = self.make_particle(BLUE)
        blue_leaf_surround = SurroundingRectangle(blue_leaf, buff=SMALL_BUFF, color=WHITE)
        blue_leaf = VGroup(blue_leaf, blue_leaf_surround)
        blue_leaf.next_to(node_2, RIGHT * 0.1 + DOWN * 4)
        tree.add(blue_leaf)

        red_leaf = self.make_particle(RED)
        red_leaf_surround = SurroundingRectangle(red_leaf, buff=SMALL_BUFF, color=WHITE)
        red_leaf = VGroup(red_leaf, red_leaf_surround)
        red_leaf.next_to(node_3, LEFT * 0.1 + DOWN * 4)
        tree.add(red_leaf)

        yellow_leaf = self.make_particle(YELLOW)
        yellow_leaf_surround = SurroundingRectangle(yellow_leaf, buff=SMALL_BUFF, color=WHITE)
        yellow_leaf = VGroup(yellow_leaf, yellow_leaf_surround)
        yellow_leaf.next_to(node_3, RIGHT * 0.1 + DOWN * 4)
        tree.add(yellow_leaf)

        edge_dict, edge_set = self.get_tree_edges(tree)

        return tree, edge_dict, edge_set

    def animate_all(self, steps, bounding_volumes, bounding_labels, tree, edge_dict, edge_set, particles):
        self.play(
            Write(steps[0])
        )

        self.wait()

        self.play(
            Write(steps[1])
        )

        self.wait()

        self.play(
            ReplacementTransform(steps[0], steps[-1])
        )

        self.play(
            Write(steps[2])
        )

        self.wait()

        self.play(
            Write(steps[3])
        )

        self.wait()

        surround_highlight_1 = SurroundingRectangle(self.get_mobject_by_color(particles, BLUE), color=BLUE, buff=SMALL_BUFF)
        surround_highlight_2 = SurroundingRectangle(self.get_mobject_by_color(particles, RED), color=RED, buff=SMALL_BUFF)

        self.play(
            ShowCreationThenDestruction(surround_highlight_1),
            ShowCreationThenDestruction(surround_highlight_2),
        )

        self.play(
            ShowCreationThenDestruction(surround_highlight_1),
            ShowCreationThenDestruction(surround_highlight_2),
        )

        self.wait()

        self.play(
            Write(steps[4])
        )

        self.play(
            Write(steps[5])
        )

        self.wait()

        self.play(
            ShowCreation(bounding_volumes[0]),
            ShowCreation(bounding_volumes[1]),
            Write(bounding_labels[0]),
            Write(bounding_labels[1]),
            run_time=2
        )

        self.wait()

        self.play(
            Write(tree[0])
        )

        self.wait()

        self.play(
            ShowCreation(edge_dict[(0, 1)]),
            ShowCreation(edge_dict[(0, 2)]),
        )

        self.play(
            Write(tree[1]),
            Write(tree[2]),
        )

        self.wait()

        self.play(
            Write(steps[6])
        )

        self.wait()

        self.play(
            ApplyWave(bounding_volumes[0])
        )

        self.wait()

        surround_highlight_1 = SurroundingRectangle(self.get_mobject_by_color(particles, PINK), color=PINK, buff=SMALL_BUFF)

        self.play(
            ShowCreationThenDestruction(surround_highlight_1),
        )

        self.play(
            ShowCreationThenDestruction(surround_highlight_1),
        )

        self.wait()

        self.play(
            ShowCreation(bounding_volumes[2]),
            ShowCreation(bounding_volumes[3]),
            Write(bounding_labels[2]),
            run_time=2
        )

        self.wait()

        self.play(
            ShowCreation(edge_dict[(1, 3)]),
            ShowCreation(edge_dict[(1, 4)]),
        )

        self.play(
            Write(tree[3]),
            Write(tree[4]),
        )

        self.wait()

        self.play(
            ApplyWave(bounding_volumes[1])
        )

        self.wait()

        self.play(
            ShowCreation(bounding_volumes[4]),
            ShowCreation(bounding_volumes[5]),
            Write(bounding_labels[3]),
            run_time=2
        )

        self.wait()

        self.play(
            ShowCreation(edge_dict[(2, 5)]),
            ShowCreation(edge_dict[(2, 6)]),
        )

        self.play(
            Write(tree[5]),
            Write(tree[6]),
        )

        self.wait()

        self.play(
            ShowCreation(bounding_volumes[6]),
            ShowCreation(bounding_volumes[7]),
        )

        self.wait()

        self.play(
            ShowCreation(edge_dict[(3, 7)]),
            ShowCreation(edge_dict[(3, 8)]),
        )

        self.play(
            Write(tree[7]),
            Write(tree[8]),
        )

        self.wait()

        self.play(
            ShowCreation(bounding_volumes[8]),
            ShowCreation(bounding_volumes[9 ]),
        )

        self.wait()

        self.play(
            ShowCreation(edge_dict[(5, 9)]),
            ShowCreation(edge_dict[(5, 10)]),
        )

        self.play(
            Write(tree[9]),
            Write(tree[10]),
        )

        self.wait()

        highlight_edges = [edge_dict[key].copy().set_stroke(color=GREEN_SCREEN, width=8) for key in edge_set]
        self.play(
            LaggedStartMap(ShowCreationThenDestruction, VGroup(*highlight_edges)),
            run_time=3
        )

        self.wait()

    def get_tree_edges(self, tree):
        edge_set = [
        (0, 1), (0, 2), 
        (1, 3), (1, 4), (2, 5), (2, 6),
        (3, 7), (3, 8), (5, 9), (5, 10),
        ]
        edge_dict = {}
        for u, v in edge_set:
            edge_dict[(u, v)] = self.make_edge(tree[u], tree[v])

        return edge_dict, edge_set

    def make_particle(self, color):
        return Ball(radius=0.15, color=color).set_fill(color=color, opacity=1)

    def make_edge(self, u, v):
        start = u.get_center()
        if isinstance(v[1], TexMobject):
            end = v.get_center()
        else:
            end = v.get_top()

        unit_v = (end - start) / np.linalg.norm((end - start))
        adjust_start = start + unit_v * 0.4
        if isinstance(v[1], TexMobject):
            adjust_end = end - unit_v * 0.4
        else:
            adjust_end = end - unit_v * 0.1

        return Line(adjust_start, adjust_end).set_stroke(width=5)

    def get_mobject_by_color(self, mobjects, color):
        for mob in mobjects:
            if mob.get_color() == Color(color) or mob.get_color() == color:
                return mob

# example of Zooming taken from 
# https://github.com/Elteoremadebeethoven/AnimationsWithManim/blob/master/English/extra/faqs/faqs.md#zoomed-scene-example
class ZoomedSceneExample(ZoomedScene):
    CONFIG = {
        "zoom_factor": 0.3,
        "zoomed_display_height": 1,
        "zoomed_display_width": 6,
        "image_frame_stroke_width": 20,
        "zoomed_camera_config": {
            "default_frame_stroke_width": 3,
        },
    }

    def construct(self):
        # Set objects
        dot = Dot().shift(UL*2)

        image=ImageMobject(np.uint8([[ 0, 100,30 , 200],
                                     [255,0,5 , 33]]))
        image.set_height(7)
        frame_text=TextMobject("Frame",color=PURPLE).scale(1.4)
        zoomed_camera_text=TextMobject("Zommed camera",color=RED).scale(1.4)

        self.add(image,dot)

        # Set camera
        zoomed_camera = self.zoomed_camera
        zoomed_display = self.zoomed_display
        frame = zoomed_camera.frame
        zoomed_display_frame = zoomed_display.display_frame

        frame.move_to(dot)
        frame.set_color(PURPLE)

        zoomed_display_frame.set_color(RED)
        zoomed_display.shift(DOWN)

        # brackground zoomed_display
        zd_rect = BackgroundRectangle(
            zoomed_display,
            fill_opacity=0,
            buff=MED_SMALL_BUFF,
        )

        self.add_foreground_mobject(zd_rect)

        # animation of unfold camera
        unfold_camera = UpdateFromFunc(
            zd_rect,
            lambda rect: rect.replace(zoomed_display)
        )

        frame_text.next_to(frame,DOWN)

        self.play(
            ShowCreation(frame),
            FadeInFromDown(frame_text)
        )

        # Activate zooming
        self.activate_zooming()

        self.play(
            # You have to add this line
            self.get_zoomed_display_pop_out_animation(),
            unfold_camera
        )

        zoomed_camera_text.next_to(zoomed_display_frame,DOWN)
        self.play(FadeInFromDown(zoomed_camera_text))

        # Scale in     x   y  z
        scale_factor=[0.5,1.5,0]

        # Resize the frame and zoomed camera
        self.play(
            frame.scale,                scale_factor,
            zoomed_display.scale,       scale_factor,
            FadeOut(zoomed_camera_text),
            FadeOut(frame_text)
        )

        # Resize the frame
        self.play(
            frame.scale,3,
            frame.shift,2.5*DOWN
        )

        # Resize zoomed camera
        self.play(
            ScaleInPlace(zoomed_display,2)
        )


        self.wait()

        self.play(
            self.get_zoomed_display_pop_out_animation(),
            unfold_camera,
            # -------> Inverse
            rate_func=lambda t: smooth(1-t),
        )
        self.play(
            Uncreate(zoomed_display_frame),
            FadeOut(frame),
        )
        self.wait()

class Recap(Scene):
    def construct(self):
        title = TextMobject("Recap").scale(1.2).move_to(UP * 3.5)
        h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
        h_line.next_to(title, DOWN)
        self.play(
            Write(title),
            ShowCreation(h_line)
        )
        self.wait()

        rect = ScreenRectangle(height=FRAME_HEIGHT-3)

        self.play(
            ShowCreation(rect)
        )

        self.wait()

        first = TextMobject("Animation").next_to(rect, DOWN * 2)

        second = TextMobject("Collision Detection").move_to(first.get_center())

        third = TextMobject("Scaling Simulations").move_to(second.get_center())

        self.play(
            Write(first)
        )

        self.wait()

        self.play(
            ReplacementTransform(first, second)
        )

        self.wait()

        self.play(
            ReplacementTransform(second, third)
        )

        self.wait()

class Patreons(Scene):
    def construct(self):
        thanks = TextMobject("Special Thanks to These Patreons").scale(1.2)
        thanks.shift(DOWN * 2)
        self.play(
            Write(thanks)
        )
        patreons = ["Burt Humburg", "Justin Hiester"]
        patreon_text = VGroup(*[TextMobject(name).scale(0.9) for name in patreons])
        patreon_text.arrange_submobjects(DOWN)
        patreon_text.next_to(thanks, DOWN)
        for patreon in patreon_text:
            self.play(
                Write(patreon)
            )
        self.wait(5)

class Thumbnail(Scene):
    def construct(self):
        start_time = time.time()
        particles = []
        num_particles = 6
        BOX_THRESHOLD = 0.98
        PARTICLE_THRESHOLD = 0.96
        box = Box(height=6, width=6).set_color(COBALT_BLUE)
        box.set_stroke(width=10)
        shift_right = RIGHT * 3
        box.shift(shift_right)
        positions = [
        LEFT * 0.31 + UP * 0.31, UP * 2.7, RIGHT * 2.62,
        LEFT * 2.57, DOWN * 2.75, RIGHT * 0.31 + DOWN * 0.31,
        ]
        colors = [BLUE, PINK, GREEN_SCREEN, RED, ORANGE, YELLOW]
        radius =[0.4, 0.25, 0.3, 0.35, 0.2, 0.45]
        for i in range(num_particles):
            particle = Ball(radius=radius[i % len(radius)])
            particle.set_color(color=colors[i % len(radius)])
            particle.id = i
            particle.move_to(positions[i])
            particle.shift(shift_right)
            particles.append(particle)
        
        self.play(
            FadeIn(box)
        )
        particles = VGroup(*particles)
        self.add(particles)

        scale = 2

        collision = TextMobject("Collision" + "\\\\" + "Simulations").scale(scale)

        arrow = TexMobject(r"\Updownarrow").scale(scale)

        graphics = TextMobject("Computer" + "\\\\" + "Graphics").scale(scale)

        text = VGroup(collision, arrow, graphics).arrange_submobjects(DOWN)
        text.move_to(LEFT * 3.5)
        self.add(text)

        self.wait()

        

        arrows = [
        Arrow(ORIGIN, UP * 1 + RIGHT * 1),
        Arrow(ORIGIN, UP * 1 + LEFT * 1),
        Arrow(ORIGIN, DOWN * 1 + RIGHT * 1),
        Arrow(ORIGIN, DOWN * 1 + LEFT * 1),
        ]

        scale = 0.7
        buff = 0

        display_arrows = [
            arrows[1].copy().next_to(particles[0], UP + LEFT, buff=buff).set_color(BLUE),
            arrows[3].copy().scale(scale).next_to(particles[1], DOWN + LEFT, buff=buff).set_color(PINK),
            arrows[1].copy().scale(scale).next_to(particles[2], UP + LEFT, buff=buff).set_color(GREEN_SCREEN),
            arrows[2].copy().scale(scale).next_to(particles[3], DOWN + RIGHT, buff=buff).set_color(RED),
            arrows[0].copy().scale(scale).next_to(particles[4], UP + RIGHT, buff=buff).set_color(ORANGE),
            arrows[2].copy().next_to(particles[5], DOWN + RIGHT, buff=buff).set_color(YELLOW)
        ]

        self.add(VGroup(*display_arrows))
        self.wait()
        dot = Dot().set_color(FUCHSIA).move_to(RIGHT * 3)
        self.add(dot)
        self.play(
            Flash(dot, num_lines=8, stroke_width=7, color=FUCHSIA)
        )

        self.wait()

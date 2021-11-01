from manimlib.imports import *
import random
np.random.seed(0)
import time
### Running some scenes requires Python Shapely library https://pypi.org/project/Shapely/
### Documentation: https://shapely.readthedocs.io/en/stable/manual.html
from shapely.geometry import Polygon as Poly
from shapely.geometry import MultiPolygon
from shapely.ops import cascaded_union

class GJKUtils(GraphScene, ThreeDScene):
    """
    Utilities class for other scenes
    Multiple inheritance is really not ideal 
    (and may not even be necessary), but it works
    """
    def construct(self):
        # Reference for this Utils class: https://www.swtestacademy.com/intersection-convex-polygons-algorithm/
        pass

    # Tested all relevant cases
    def intersect_line_seg(self, line1, line2):
        """
        @param line1, line2: line segments defined as (start, end)
            where start and end are np.array([x, y, z])
        return: intersection point of two line segments 
            or empty list if no intersection exists
        """
        l1p1, l1p2 = line1[0], line1[1]
        l2p1, l2p2 = line2[0], line2[1]

        A1 = l1p2[1] - l1p1[1]
        B1 = l1p1[0] - l1p2[0]
        C1 = A1 * l1p1[0] + B1 * l1p1[1]

        A2 = l2p2[1] - l2p1[1]
        B2 = l2p1[0] - l2p2[0]
        C2 = A2 * l2p1[0] + B2 * l2p1[1]

        det = A1 * B2 - A2 * B1
        if np.isclose(det, 0):
            return []

        else:
            x = (B2 * C1 - B1 * C2) / det
            y = (A1 * C2 - A2 * C1) / det
            online1 = ((min(l1p1[0], l1p2[0]) < x or np.isclose(min(l1p1[0], l1p2[0]), x))
                and (max(l1p1[0], l1p2[0]) > x or np.isclose(max(l1p1[0], l1p2[0]), x))
                and (min(l1p1[1], l1p2[1]) < y or np.isclose(min(l1p1[1], l1p2[1]), y))
                and (max(l1p1[1], l1p2[1]) > y or np.isclose(max(l1p1[1], l1p2[1]), y))
                )
            online2 = ((min(l2p1[0], l2p2[0]) < x or np.isclose(min(l2p1[0], l2p2[0]), x))
                and (max(l2p1[0], l2p2[0]) > x or np.isclose(max(l2p1[0], l2p2[0]), x))
                and (min(l2p1[1], l2p2[1]) < y or np.isclose(min(l2p1[1], l2p2[1]), y))
                and (max(l2p1[1], l2p2[1]) > y or np.isclose(max(l2p1[1], l2p2[1]), y))
                )

            if online1 and online2:
                return np.array([x, y, 0])

        return []

    # Tested all relevant cases 
    # Some degenerate cases may not work
    # e.g one edge of polygon lies directly on another edge
    def get_intersection_points(self, p1, p2):
        """
        @param p1, p2: convex polygons (Type Geometry.Polygon)
        return: list of intersection points
        """
        intersections = []
        for edge1 in self.get_edges(p1):
            for edge2 in self.get_edges(p2):
                intersection = self.intersect_line_seg(edge1, edge2)
                if len(intersection) != 0:
                    intersections.append(intersection)
        return intersections

    # Tested and works
    def check_intersection(self, p1, p2):
        """
        @param p1, p2: convex polygons (Type Geometry.Polygon)
        return: True if p1 and p2 intersect, False otherwise
        """
        return len(self.get_intersection_points(p1, p2)) != 0

    # Tested on limited number of manual cases -> worked
    def get_intersection_region(self, p1, p2):
        """
        @param p1, p2: convex polygons (Type Geometry.Polygon)
        return: list of points that define region of intersection
        """
        region = []
        intersection_points = self.get_intersection_points(p1, p2)
        if len(intersection_points) == 0:
            return []

        for point in intersection_points:
            if not self.check_duplicate(region, point):
                region.append(point)

        for v1 in p1.get_vertices():
            if self.check_point_inside(v1, p2) and \
                not self.check_duplicate(region, v1):
                region.append(v1)

        for v2 in p2.get_vertices():
            if self.check_point_inside(v2, p1) and \
                not self.check_duplicate(region, v2):
                region.append(v2)

        return self.order_points_cclockwise(region)

    def check_duplicate(self, region, point):
        """
        @param region: set of points that define region
        @param point: point to test
        return: True if point exists in region (within tolerance) 
            False otherwise
        """
        for p in region:
            if np.allclose(p, point):
                return True
        return False

    # Tested on few known examples -> works
    # Good explanation: https://cp-algorithms.com/geometry/minkowski.html
    def get_minkowski_sum(self, p1, p2):
        """
        Minkowski sums are defined as all points in p1 + all points in p2
        @param p1, p2: convex polygons (Type Geometry.Polygon)
        return: list of points that define Minkowski sum
        """
        result = []
        P = self.order_points_cclockwise(p1.get_vertices())
        Q = self.order_points_cclockwise(p2.get_vertices())

        # helps with cyclic indexing
        P.extend([P[0], P[1]])
        Q.extend([Q[0], Q[1]])

        i, j = 0, 0
        while i < len(P) - 2 or j < len(Q) - 2:
            result.append(P[i] + Q[j])
            dP = P[i + 1] - P[i]
            dQ = Q[j + 1] - Q[j]
            cross = dP[0] * dQ[1] - dP[1] * dQ[0]
            if cross >= 0:
                i += 1
            else:
                j += 1

        return self.order_points_cclockwise(result)

    # Tested on few known examples -> works
    def get_minkowski_diff(self, p1, p2):
        """
        Minkowski differences are defined as all points in p1 - all points in p2
        @param p1, p2: convex polygons (Type Geometry.Polygon)
        return: list of points that define Minkowski sum
        """
        neg_p2 = self.negate_polygon(p2)
        return self.get_minkowski_sum(p1, neg_p2)

    # Tested and works
    def negate_polygon(self, p):
        negated = [np.negative(v) for v in p.get_vertices()]
        return Polygon(*self.order_points_cclockwise(negated))

    # Tested and works
    def get_edges(self, p):
        """
        @param p: convex polygon
        return: list of edges of polygon
        [(u_1, u_2), (u_2, u_3), ... (u_n, u_1)]
        """
        edges = []
        vertices = p.get_vertices()
        for i in range(len(vertices)):
            edge = (vertices[i], vertices[(i + 1) % len(vertices)])
            edges.append(edge)

        return edges

    # Tested via sampling points from circle
    def order_points_cclockwise(self, points):
        """
        Useful for getting points that can then be passed into Geometry.Polygon
        @param points: list of points
        return: list of points sorted in counter-clockwise order
        """
        center = np.mean(points, axis=0)
        return sorted(points, key=lambda x: np.arctan2(x[1] - center[1], x[0] - center[0]))

    # Tested over random set of points -> works
    def check_point_inside(self, point, polygon):
        """
        @param point: test point (type np.array[x, y, z])
        @param polygon: convex polygon (type Geometry.Polygon)
        return: True if point is inside polygon, False otherwise
        """
        vertices = polygon.get_vertices()
        u_indices = list(range(len(vertices)))
        v_indices = [len(vertices) - 1] + list(range(len(vertices) - 1))
        result = False
        for i, j in zip(u_indices, v_indices):
            condition1 = (vertices[i][1] > point[1]) != (vertices[j][1] > point[1]) 
            condition2 = (point[0] < (vertices[j][0] - vertices[i][0]) * 
                (point[1] - vertices[i][1]) / (vertices[j][1] - vertices[i][1]) + vertices[i][0])
            if condition1 and condition2:
                result = not result

        return result

    # Tested on few random directions and polygons
    def get_support(self, polygon, d):
        """
        @param polygon: convex polygon (type Geometry.Polygon)
        @param d: vector direction (np.array[x, y, z])
        return: vertex on polygon furthest along direction
        """
        vertices = polygon.get_vertices()
        index = np.argmax([np.dot(v, d) for v in vertices])
        return vertices[index]

    def tripleProd(self, a, b, c):
        """
        @param a, b, c: np.array[x, y, z] vectors
        return: (a x b) x c -- x represents cross product
        """
        return b * np.dot(c, a) - a * np.dot(c, b)

    def get_normal_towards_origin(self, a, b, away=False):
        x1, y1 = a[0], a[1]
        x2, y2 = b[0], b[1]
        dx = x2 - x1
        dy = y2 - y1
        n1 = np.array([-dy, dx, 0])
        n2 = np.array([dy, -dx, 0])
        AO = ORIGIN - a
        if np.dot(AO, n1) > 0:
            if away:
                return n2
            return n1
        if away:
            return n1
        return n2

    def support(self, p1, p2, d):
        """
        @param p1: convex polygon (type Geometry.Polygon)
        @param p2: convex polygon (type Geometry.Polygon)
        @param d: vector direction (np.array[x, y, z])
        return: support point in direction d of Minkowski diff of p1 and p2
        """
        return self.get_support(p1, d) - self.get_support(p2, np.negative(d))

    ### Below methods are for handling polygon, circle cases

    # Tested on various line segment and circle combinations
    def circle_line_segment_intersection(self, circle, line_seg, full_line=False, tangent_tol=1e-9):
        """ Find the points at which a circle intersects a line-segment.  This can happen at 0, 1, or 2 points.

        @param: circle: Geometry.Circle object
        @param: line_seg: [u_1, u_2] where u_1 and u_2 are type np.array[x, y, z]
        @param: full_line: True to find intersections along full line - not just in the segment
        @param: tangent_tol: Numerical tolerance for tangent line intersections
        return: [np.array[x, y, z], np.array[x, y, z]] A list of length 0, 1, or 2, where each element is a point at which the circle intersects a line segment.
        
        Note: Followed: http://mathworld.wolfram.com/Circle-LineIntersection.html
        Implementation slightly modified from 
        https://stackoverflow.com/questions/30844482/what-is-most-efficient-way-to-find-the-intersection-of-a-line-and-a-circle-in-py
        """
        circle_center = circle.get_center()[:2]
        circle_radius = circle.radius
        pt1, pt2 = line_seg[0][:2], line_seg[1][:2]
        (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, circle_center
        (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
        dx, dy = (x2 - x1), (y2 - y1)
        dr = (dx ** 2 + dy ** 2)**.5
        big_d = x1 * y2 - x2 * y1
        discriminant = circle_radius ** 2 * dr ** 2 - big_d ** 2

        if discriminant < 0:  # No intersection between circle and line
            return []
        else:  # There may be 0, 1, or 2 intersections with the segment
            intersections = [
                (cx + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant**.5) / dr ** 2,
                 cy + (-big_d * dx + sign * abs(dy) * discriminant**.5) / dr ** 2)
                for sign in ((1, -1) if dy < 0 else (-1, 1))] # This makes sure the order along the segment is correct
             # If only considering the segment, filter out intersections that do not fall within the segment
            if not full_line:
                fraction_along_segment = [(xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy for xi, yi in intersections]
                intersections = [pt for pt, frac in zip(intersections, fraction_along_segment) if 0 <= frac <= 1]
            # If line is tangent to circle, return just one point (as both intersections have same location)
            if len(intersections) == 2 and abs(discriminant) <= tangent_tol:
                return coords_to_array([intersections[0]])
            else:
                return coords_to_array(intersections)

    # tested on various circle polygon combinations
    def get_curved_intersection_points(self, polygon, circle):
        """
        @param: polygon - Geometry.Polygon
        @param: circle - Geometry.Circle
        return: list of intersection points between polygon and circle
        """

        intersection_points = []
        for edge in self.get_edges(polygon):
            seg_intersection = self.circle_line_segment_intersection(circle, edge)
            intersection_points.extend(seg_intersection)
        return intersection_points

    def get_curved_intersection_region(self, polygon, circle):
        """
        @param: polygon - Geometry.Polygon
        @param: circle - Geometry.Circle
        return: list of points that define region of intersection
        """
        region = []
        intersection_points = self.get_curved_intersection_points(polygon, circle)
        if len(intersection_points) < 2:
            return []

        if len(intersection_points) > 2:
            return []
        enclosed_points = []
        for point in intersection_points:
            if not self.check_duplicate(region, point):
                region.append(point)

        for v in polygon.get_vertices():
            if self.check_point_inside_circle(v, circle) and \
                not self.check_duplicate(region + enclosed_points, v):
                enclosed_points.append(v)

        return self.order_points_cclockwise(region + enclosed_points)

    def check_point_on_circle(self, point, circle):
        """
        @param: point - np.array[x, y, z]
        @param: circle - type Geometry.Circle
        return: True if point on circumference of circle, False otherwise
        """
        dist = np.linalg.norm(point - circle.get_center())
        r = circle.radius
        return np.isclose(dist, r)

    # Tested on set of random points and random circles
    def check_point_inside_circle(self, point, circle):
        """
        @param: point - np.array[x, y, z]
        @param: circle - type Geometry.Circle
        return: True if point inside circle, False otherwise
        """
        return np.linalg.norm(point - circle.get_center()) < circle.radius

    def convert_curved_region_into_mob(self, region, circle, color=FUCHSIA, opacity=0.5):
        closed_region = region + [region[0]]
        arc = []
        arc_index_end = 0
        for i in range(len(closed_region) - 1):
            u, v = closed_region[i], closed_region[i + 1]
            if self.check_point_on_circle(u, circle) and \
                self.check_point_on_circle(v, circle):
                arc_index_end = i + 1
                arc.extend([u, v])

        corners = []
        for j in range(arc_index_end, arc_index_end + len(closed_region)):
            index = j % len(closed_region)
            point = closed_region[index]
            if not array_in_list(point, corners):
                corners.append(point)
        a1 = point_to_angle(arc[0], circle)
        a2 = point_to_angle(arc[1], circle)
        angle = a2 - a1
        if a1 > a2:
            angle = TAU + angle
        mob = Arc(
                start_angle=a1, angle=angle, 
                arc_center=circle.get_center(), 
                radius=circle.radius,
        )
        # print(corners)
        rest = VGroup()
        rest.set_points_as_corners(*[corners])
        mob.add(rest)
        mob.set_stroke(color=color)
        mob.set_fill(color=color, opacity=opacity)
        return mob

    def get_vector_arrow(self, start, end):
        arrow = Arrow()
        arrow.put_start_and_end_on(start, end)
        return arrow

    def get_halfspace_normal(self, vector, color=BLUE, opacity=0.3):
        """
        @param: vector - Geometry.Arrow
        return: Geometry.Polygon that defines halfspace a^T v >= 0
        """
        v = normalize(vector.get_end() - vector.get_start()) 
        normal = self.get_normal_towards_origin(ORIGIN, v)
        normal_v = Vector(normal).set_color(YELLOW)
        points = []
        FRAME_DIAGONAL_RADIUS = np.sqrt(FRAME_X_RADIUS ** 2 + FRAME_Y_RADIUS ** 2) 
        points.append(FRAME_DIAGONAL_RADIUS  * normal)
        points.append(-FRAME_DIAGONAL_RADIUS * normal)
        
        corners = [
        FRAME_X_RADIUS * LEFT + FRAME_Y_RADIUS * UP,
        FRAME_X_RADIUS * LEFT + FRAME_Y_RADIUS * DOWN ,
        FRAME_X_RADIUS * RIGHT + FRAME_Y_RADIUS * UP ,
        FRAME_X_RADIUS * RIGHT + FRAME_Y_RADIUS * DOWN ,
        ]
        for c in corners:
            if np.dot(c, v) >= 0:
                points.append(c)

        polygon = Polygon(*self.order_points_cclockwise(points))
        polygon.set_stroke(color=color, opacity=opacity)
        polygon.set_fill(color=color, opacity=opacity)
        return polygon

    def get_union_edges(self, p1, p2):
        union_edges = []
        self.add_union_edges(p1, p2, union_edges)
        self.add_union_edges(p2, p1, union_edges)
        return union_edges
    
    def get_edge(self, edges, vertex):
        for e in edges:
            if np.array_equal(e[0], vertex):
                return e

    def add_union_edges(self, p1, p2, union_edges):
        for edge1 in self.get_edges(p1):
            intersection_points = []
            for edge2 in self.get_edges(p2):
                intersection = self.intersect_line_seg(edge1, edge2)
                if len(intersection) == 0:
                    continue
                intersection_points.append(intersection)
            key_points = [edge1[0]] + intersection_points + [edge1[1]]
            key_points.sort(key=lambda x: np.linalg.norm(x - edge1[0]))
            for i in range(len(key_points) - 1):
                union_edges.append((key_points[i], key_points[i + 1]))

    def get_neighboring_edges(self, point, edge, union_edges):
        neighbors = []
        for e in union_edges:
            if self.edge_equal(e, edge):
                continue
            if np.array_equal(point, e[0]) or np.array_equal(point, e[1]):
                neighbors.append(e)
        return neighbors

    def edge_equal(self, e1, e2):
        return np.array_equal(e1[0], e2[0]) and np.array_equal(e1[1], e2[1])

    def get_edge_vector(self, edge, point):
        if np.array_equal(edge[0], point):
            v = edge[1] - point
        else:
            v = edge[0] - point
        return v / np.linalg.norm(v)

    def polygon_union(self, p1, p2):
        """
        DO NOT USE THIS FUNCTION, IT DOES NOT ALWAYS WORK, I LEARNED THE HARD WAY
        Use polygon_union_shapely for accurate union operations
        @param p1: Geometry.Polygon
        @param p2: Geometry.Polygon
        Note: if polygons are not intersecting, will return empty list
        Function is meant to be used on known intersecting polygons
        return: ordered set of points that define polygon union
        """
        intersecting_points = self.get_intersection_points(p1, p2)
        if len(intersecting_points) == 0:
            return []
        union_points = []
        for p in p1.get_vertices():
            if not self.check_duplicate(union_points, p):
                union_points.append(p)
        for p in intersecting_points:
            if not self.check_duplicate(union_points, p):
                union_points.append(p)
        for p in p2.get_vertices():
            if not self.check_duplicate(union_points, p):
                union_points.append(p)

        min_x = min(union_points, key=lambda x: x[0])[0]
        min_y = min(union_points, key=lambda x: x[1])[1]
        min_xy = np.array([min_x, min_y, 0])

        start_point = min(union_points, key=lambda x: np.linalg.norm(x - min_xy))

        union_edges = self.get_union_edges(p1, p2)

        start_edge = self.get_edge(union_edges, start_point)
        # start_edge = union_edges[5]
        # self.play(
        #     Indicate(Line(start_edge[0], start_edge[1])),
        #     Indicate(Dot().move_to(start_edge[0]))
        # )
        current_edge = start_edge
        union_region_points = [start_point]

        visited = [current_edge]

        j = 0
        while j < 200:
            v1 = current_edge[1] - current_edge[0]
            v1 = v1 / np.linalg.norm(v1)
            angles = []
            neighbors = []
            for neighbor in self.get_neighboring_edges(current_edge[0], current_edge, union_edges):
                v2 = self.get_edge_vector(neighbor, current_edge[0])
                dot_prod = np.dot(v1, v2)
                if dot_prod > 1: # numerical precision issues
                    dot_prod = 1.0
                if dot_prod < -1:
                    dot_prod = -1.0
                angle = np.arccos(dot_prod)
                if cross(v1, v2)[2] < 0:
                    angle = TAU - angle
                angles.append(angle)
                print(angles)
                neighbors.append(neighbor)
            # if len(neighbors) == 0:
            #     print('Breaking')
            #     break
            next_edge_index = np.argmax(np.array(angles))
            next_edge = neighbors[next_edge_index]
            # self.play(
            #     Indicate(Line(next_edge[0], next_edge[1])),
            #     Indicate(Dot().move_to(next_edge[0]))
            # )
            if self.edge_equal(start_edge, next_edge):
                print('Breaking seen start edge')
                break
            union_region_points.append(next_edge[0])
            current_edge = next_edge
            j += 1
        
        return union_region_points

    def polygon_union_shapely(self, p1, p2):
        """
        @param p1: Geometry.Polygon
        @param p2: Geometry.Polygon
        return: list of points that define the union of two polygons
        """
        p1_vertices = [(a[0], a[1]) for a in p1.get_vertices()]
        p2_vertices = [(a[0], a[1]) for a in p2.get_vertices()]

        p1_shape = Poly(p1_vertices)
        p2_shape = Poly(p2_vertices)
        print(type(p1_shape))
        union_shapely = cascaded_union([p1_shape, p2_shape])
        print(type(union_shapely))
        union_region = [np.array([x, y, 0]) for x, y in list(union_shapely.exterior.coords)]

        return union_region

    def union_all_polygons(self, polygons, color=FUCHSIA, opacity=0.5):
        shapely_polygons = []
        for p in polygons:
            if isinstance(p, Polygon):
                vertices = [(a[0], a[1]) for a in p.get_vertices()]
                shapely_polygons.append(Poly(vertices))
        if len(shapely_polygons) == 0:
            return VGroup()
        union_polygon = cascaded_union(shapely_polygons)
        union_region = [np.array([x, y, 0]) for x, y in list(union_polygon.exterior.coords)]
        polygon = Polygon(*union_region).set_stroke(color=color, width=5)
        polygon.set_fill(color=color, opacity=opacity)

        return polygon

    def remove_collinear_points(self, union_region):
        if len(union_region) == 2:
            return union_region
        region = union_region + [union_region[0], union_region[1]]
        for i in range(len(region) - 2):
            p0, p1, p2 = region[i], region[i + 1], region[i + 2]
            x0, y0 = p0[0], p0[1]
            x1, y1 = p1[0], p1[1]
            x2, y2 = p2[0], p2[1]
            if np.isclose(x1 - x0, 0):
                if np.isclose(x2 - x0, 0):
                    remove_array(union_region, p1)
                    continue
            if np.isclose(y1 - y0, 0):
                if np.isclose(y2 - y0, 0):
                    remove_array(union_region, p1)
                    continue

            k1 = (x2 - x0) / (x1 - x0)
            k2 = (y2 - y0) / (y1 - y0)
            if np.isclose(k1, k2):
                remove_array(union_region, p1)
        
        return union_region

    def get_concave_intersection_region(self, concave, p2, color=FUCHSIA, opacity=0.5):
        concave_shapely = Poly([(a[0], a[1]) for a in concave.get_vertices()])
        p2_shapely = Poly([(a[0], a[1]) for a in p2.get_vertices()])
        intersection = concave_shapely.intersection(p2_shapely)
        if isinstance(intersection, MultiPolygon): # region has holes/gaps
            region = VGroup()
            for poly in list(intersection):
                region.add(self.convert_shapely_poly_to_mob(poly, color=color, opacity=opacity))
            return region
        return self.convert_shapely_poly_to_mob(intersection, color=color, opacity=opacity)

    def convert_shapely_poly_to_mob(self, poly, color=FUCHSIA, opacity=0.5):
        intersection_region = [np.array([x, y, 0]) for x, y in poly.exterior.coords]
        if len(intersection_region) == 0:
            return VGroup()
        polygon = Polygon(*intersection_region).set_stroke(color=FUCHSIA, width=5)
        polygon.set_fill(color=color, opacity=opacity)
        return polygon

### BEGIN VARIOUS TESTING SCENES OF GJKUtils ###

class TestPolygonUnion(GJKUtils):
    def construct(self):
        p1 = RegularPolygon(n=8).scale(2).rotate(PI / 6)
        p2 = RegularPolygon(n=7).scale(2)

        p1.shift(LEFT)
        p2.shift(RIGHT)
        self.add(p1, p2)
        self.wait()

        union_region = self.polygon_union_shapely(p1, p2)
        polygon_union = Polygon(*union_region)
        polygon_union.set_stroke(color=FUCHSIA, width=5)
        polygon_union.set_fill(color=FUCHSIA, opacity=0.5)

        self.play(
            FadeIn(polygon_union)
        )
        self.wait()

class TestCollinear(GJKUtils):
    def construct(self):
        polygon = RegularPolygon(n=6).scale(2).rotate(PI/5)
        n = 72
        points = [polygon.point_from_proportion(i / n) for i in range(n)]
        
        dots = VGroup(*[Dot().scale(0.5).move_to(p) for p in points])
        self.add(dots)
        self.wait()

        key_points = self.remove_collinear_points(points)
        key_dots = [Dot().scale(0.6).move_to(p) for p in key_points]
        for d in key_dots:
            self.play(
                d.set_color, GREEN_SCREEN,
            )
            self.wait()

class TestIntersection(GJKUtils):
    def construct(self):
        polygon = RegularPolygon(n=6).scale(2).shift(LEFT + UP)

        self.add(polygon)

        circle = Circle(radius=1.5)
        self.add(circle)
        size = 50
        for x, y in zip(np.random.uniform(-4.1, 4.1, size=size), np.random.uniform(-4.1, 4.1, size=size)):
            circle.move_to(RIGHT * x + UP * y)
            intersection_region = self.get_curved_intersection_region(polygon, circle)
            self.wait()
            if len(intersection_region) == 0:
                continue
            region_mob = self.convert_curved_region_into_mob(intersection_region, circle)
            self.add(region_mob)
            self.wait()
            self.remove(region_mob)

class HighlightArc(Scene):
    def construct(self):
        circle = Circle(radius=1.5)
        self.add(circle)
        dot1 = Dot()
        dot2 = Dot()
        self.add(dot1, dot2)

        for _ in range(10):
            x, y = np.random.uniform(-4, 4), np.random.uniform(-1.5, 1.5)
            circle.move_to(RIGHT * x + UP * y)

            p1 = circle.point_from_proportion(np.random.uniform())
            p2 = circle.point_from_proportion(np.random.uniform())
            dot1.move_to(p1)
            dot2.move_to(p2)
            a1 = point_to_angle(p1, circle)
            a2 = point_to_angle(p2, circle)
            arc = Arc(
                start_angle=a1, angle=a2 - a1, 
                arc_center=circle.get_center(), 
                radius=circle.radius
            )
            arc.set_stroke(color=YELLOW, width=7)
            self.play(
                ShowCreation(arc)
            )
            self.wait()
            self.remove(arc)

class TestSupport(GJKUtils):
    def construct(self):
        p = RegularPolygon(n=7).move_to(UP * 2 + RIGHT * 2)
        p.scale(2)
        self.add(p)
        self.wait()
        for _ in range(10):
            d = get_random_vector()
            arrow = Arrow(p.get_center(), p.get_center() + d).scale(2)
            self.add(arrow)

            support = self.get_support(p, d)
            dot = Dot().move_to(support)
            self.play(
                Indicate(dot)
            )
            self.wait()
            self.remove(arrow)
            self.remove(dot)

class TestColorScene(GJKUtils):
    def construct(self):
        plane = NumberPlane()
        self.add(plane)

        self.wait()

        p1, p2 = self.get_polygons()
        intersection = self.get_intersection_region(p1, p2)
        if len(intersection) == 0:
            return

        region_color = FUCHSIA
        polygon_region = Polygon(*intersection)
        polygon_region.set_color(region_color)
        polygon_region.set_fill(color=region_color, opacity=0.5)
        self.play(
            DrawBorderThenFill(polygon_region)
        )

        self.wait()

    def get_polygons(self):
        p1_color, p2_color = YELLOW, YELLOW
        p1 = RegularPolygon(n=5).scale(1.5).set_color(p1_color)
        p2 = RegularPolygon(n=6).scale(1.5).set_color(p2_color)
        VGroup(p1, p2).arrange_submobjects(RIGHT)
        p1.shift(RIGHT)
        self.add(p1, p2)
        self.wait()

        return p1, p2

### END VARIOUS TESTING SCENES OF GJKUtils ###

### Start of scenes used in video ###
class IntroduceProblem(GJKUtils):
    """
    Better version of this implementation/demo can be found
    in the next class GJKIntroDemo (logic was significantly simplified)
    """
    CONFIG = {
        "simulation_time": 5,
    }
    def construct(self):
        question = TextMobject("Do these two shapes intersect?")
        question.move_to(DOWN * 3)

        polygons = self.get_polygons()

        self.play(
            ShowCreation(polygons),
        )
        self.wait()

        self.play(
            Write(question)
        )

        self.wait()

        crossX = CrossX().scale(0.4)
        crossX.next_to(question, RIGHT)
        self.add(crossX)
        self.wait()

        p1, p2 = polygons[0], polygons[1]

        p1.shift(RIGHT + UP * 0.5)
        p2.shift(LEFT * 0.3 + DOWN * 0.5)

        polygons.move_to(ORIGIN)

        intersection = self.get_intersection_region(p1, p2)
        intersection_poly = self.get_intersection_poly(intersection)
        self.add(intersection_poly)

        self.remove(crossX)
        
        check = CheckMark()
        check.scale(0.2)
        check.next_to(question, RIGHT)
        self.add(check)

        self.wait()

        p1.shift(LEFT * 0.5 + DOWN * 0.5)
        p2.shift(RIGHT * 0.1 + UP * 1)

        polygons.move_to(ORIGIN)

        self.remove(intersection_poly)
        self.remove(check)
        self.add(crossX)

        self.wait()

        p1.shift(RIGHT * 1)
        p2.shift(LEFT * 1 + DOWN * 0.5)

        polygons.move_to(ORIGIN)

        intersection = self.get_intersection_region(p1, p2)
        intersection_poly = self.get_intersection_poly(intersection)
        self.add(intersection_poly)

        self.remove(crossX)
        self.add(check)

        self.wait()

        polygons = self.get_polygon_and_circle()
        self.remove(p1, p2, intersection_poly)
        p1, p2 = polygons
        self.add(p1, p2)
        self.remove(check)
        self.add(crossX)

        self.wait()

        p1.shift(RIGHT * 1 + DOWN * 0.2)
        p2.shift(LEFT * 1 + UP * 0.3)
        polygons.move_to(ORIGIN)

        intersection = self.get_curved_intersection_region(p1, p2)
        intersection_poly = self.convert_curved_region_into_mob(intersection, p2)
        self.add(intersection_poly)

        self.remove(crossX)
        self.add(check)

        self.wait()

        polygons = self.get_various_polygons()
        
        self.remove(p1, p2, intersection_poly)
        p1, p2 = polygons
        self.add(p1, p2)
        self.remove(check)
        self.add(crossX)

        self.wait()

        t1, t2 = self.get_trajectories(p1, p2)
        frames = self.simulation_time * self.camera.frame_rate
        proportions = [i / frames for i in range(frames + 1)]
        proportions = [smooth(x) for x in proportions]
        self.frame = 0
        self.prev_intersection_region = None
        self.intersection_region = None
        def update_polygons(polygons, dt):
            p1, p2 = polygons[0], polygons[1]
            if self.frame > frames:
                return
            new_c1 = t1.point_from_proportion(proportions[self.frame])
            new_c2 = t2.point_from_proportion(proportions[self.frame])
            dP1, dP2 = new_c1 - p1.get_center(), new_c2 - p2.get_center()
            p1.shift(dP1)
            p2.shift(dP2)
            self.frame += 1

            intersection = self.get_intersection_region(p1, p2)
            if len(intersection) == 0:
                if self.prev_intersection_region:
                    self.remove(check)
                    self.add(crossX)
                    self.remove(self.prev_intersection_region)
                    self.prev_intersection_region = None
                    self.intersection_region = None
                return
            self.intersection_region = self.get_intersection_poly(intersection)
            if not self.prev_intersection_region:
                self.add(self.intersection_region)
                self.prev_intersection_region = self.intersection_region
                self.remove(crossX)
                self.add(check)
            else:
                self.remove(self.prev_intersection_region)
                self.add(self.intersection_region)
                self.prev_intersection_region = self.intersection_region
            
        polygons.add_updater(update_polygons)
        self.add(polygons)
        self.wait(self.simulation_time)
        polygons.clear_updaters()

        t2, t1 = t1, t2
        self.frame = 0
        self.prev_intersection_region = None
        self.intersection_region = None

        polygons = self.get_different_polygons(p1, p2)
        self.play(
            ReplacementTransform(p1, polygons[0]),
            ReplacementTransform(p2, polygons[1]),
        )

        self.wait()
        polygons.add_updater(update_polygons)
        self.add(polygons)
        self.wait(self.simulation_time)
        polygons.clear_updaters()

    def get_polygons(self):
        p1_color, p2_color = YELLOW, PERIWINKLE_BLUE
        p1 = RegularPolygon(n=3).scale(2).set_stroke(color=p1_color, width=5)
        p2 = RegularPolygon(n=4).scale(2).set_stroke(color=p2_color, width=5)
        VGroup(p1, p2).arrange_submobjects(RIGHT).move_to(ORIGIN)

        return VGroup(p1, p2)

    def get_polygon_and_circle(self):
        p1_new = RegularPolygon(n=6).scale(2).set_stroke(color=YELLOW, width=5)
        p2_new = Circle(radius=2).set_stroke(color=PERIWINKLE_BLUE, width=5)
        p = VGroup(p1_new, p2_new).arrange_submobjects(RIGHT)       
        return p

    def get_various_polygons(self):
        polygons = []
        p1_color, p2_color = YELLOW, PERIWINKLE_BLUE
        p1 = RegularPolygon(n=5).scale(2).set_stroke(color=p1_color, width=5)
        p2 = RegularPolygon(n=6).scale(2).set_stroke(color=p2_color, width=5)
        p = VGroup(p1, p2)
        p.arrange_submobjects(RIGHT).move_to(ORIGIN)
        polygons.append(p)

        return p

    def get_different_polygons(self, p1, p2):
        p1_new = RegularPolygon(n=7).scale(2).set_stroke(color=p1.get_color(), width=5)
        p2_new = RegularPolygon(n=8).scale(2).set_stroke(color=p2.get_color(), width=5)
        p1_new.move_to(p1.get_center())
        p2_new.move_to(p2.get_center())
        p = VGroup(p1_new, p2_new)
        return p


    def get_intersection_poly(self, intersection, color=FUCHSIA, width=5, opacity=0.5):
        intersection_poly = Polygon(*intersection)
        intersection_poly.set_stroke(color=color, width=width)
        intersection_poly.set_fill(color=color, opacity=opacity)
        return intersection_poly

    def get_trajectories(self, p1, p2):
        trajectory_p1 = ArcBetweenPoints(p1.get_center(), p2.get_center(), angle=TAU/6)
        trajectory_p1.set_color(p1.get_color())
        trajectory_p2 = ArcBetweenPoints(p2.get_center(), p1.get_center(), angle=TAU/3)
        trajectory_p2.set_color(p2.get_color())

        return trajectory_p1, trajectory_p2

class GJKIntroDemo(GJKUtils):
    CONFIG = {
        "p1_color": YELLOW,
        "p2_color": PERIWINKLE_BLUE,
    }
    def construct(self):
        question = TextMobject("Do these two shapes intersect?")
        question.move_to(DOWN * 3)

        self.wait()

        polygons = self.get_polygons()

        self.play(
            ShowCreation(polygons[0]),
        )

        self.play(
            ShowCreation(polygons[1])
        )
        self.wait()
        self.add(polygons)

        self.play(
            Write(question)
        )

        self.wait()

        crossX = CrossX().scale(0.4)
        crossX.next_to(question, RIGHT)
        self.add(crossX)
        self.wait()

        check = CheckMark()
        check.scale(0.2)
        check.next_to(question, RIGHT)

        self.polygon_demo(polygons, crossX, check)

        shapes = self.polygon_circle_demo(polygons, crossX, check)

        self.concave_polygon_demo(shapes, crossX, check)

    def polygon_demo(self, polygons, cross, check):
        intersection_region = VGroup()
        self.add(intersection_region)
        original_cross = cross.copy()
        original_check = check.copy()
        answer = cross
        
        def intersection_region_updater(polygons):
            p1, p2 = polygons
            intersection = self.get_intersection_region(p1, p2)
            if len(intersection) == 0:
                answer.become(original_cross)
                intersection_region.become(VGroup())
            else:
                polygon_region = self.get_intersection_poly(intersection)
                intersection_region.become(polygon_region)
                answer.become(original_check)

        polygons.add_updater(intersection_region_updater)
        p1, p2 = polygons
        self.play(
            p1.shift, RIGHT,
            p2.shift, LEFT,
            run_time=2,
        )
        self.play(
            Rotate(p1, -PI / 2),
            Rotate(p2, PI / 2),
            run_time=2
        )

        self.play(
            p1.scale, 0.8,
            p2.scale, 0.8
        )

        self.play(
            p1.shift, RIGHT * 2.5 + DOWN * 0.5,
            p2.shift, LEFT * 2.5 + UP * 0.8,
            run_time=2,
        )

        self.play(
            p1.scale, 1 / 0.8,
            p2.scale, 1 / 0.8,
        )

        self.play(
            p1.shift, RIGHT * 0.8 + UP * 0.5,
            p2.shift, LEFT * 0.8 + DOWN * 0.8,
        )

        new_polygons = self.get_different_polygons(p1, p2)
        self.play(
            Transform(p1, new_polygons[0]),
            Transform(p2, new_polygons[1]),
            run_time=2,
        )
        self.wait()

        self.play(
            p1.shift, LEFT,
            p2.shift, RIGHT,
        )

        self.play(
            Rotate(p1, PI / 3),
            Rotate(p2, PI / 2),
            run_time=2,
        )

        self.play(
            p1.shift, LEFT + DOWN * 0.2,
            p2.shift, RIGHT + UP,
            run_time=1,
        )

        self.play(
            p1.scale, 0.8,
            p2.scale, 0.8
        )

        self.play(
            p1.shift, LEFT * 1.5 + UP * 0.2,
            p2.shift, RIGHT * 1.5 + DOWN,
            run_time=2,
        )
        self.wait()

        p1_copy = p1.copy().scale(1 / 0.8).shift(LEFT)
        p2_copy = p2.copy().scale(1 / 0.8).shift(RIGHT)

        self.play(
            Transform(p1, p1_copy),
            Transform(p2, p2_copy),
            run_time=1
        )
        polygons.clear_updaters()
        self.remove(intersection_region)

    def polygon_circle_demo(self, polygons, cross, check):
        p1, p2 = polygons
        new_p1 = RegularPolygon(n=6).scale(2).set_stroke(color=self.p1_color, width=5)
        new_p1.move_to(p1.get_center())
        
        circle = Circle(radius=2).set_color(p2.get_color())
        circle.move_to(p2.get_center())

        self.play(
            ReplacementTransform(p1, new_p1),
            ReplacementTransform(p2, circle),
            run_time=2
        )
        shapes = VGroup(new_p1, circle)
        self.add(shapes)

        curved_intersection_region = VGroup()
        self.add(curved_intersection_region)
        original_cross = cross.copy()
        original_check = check.copy()
        intersect_answer = cross
        self.add(intersect_answer)
        
        def curved_intersection_region_updater(shapes):
            polygon, circle = shapes
            intersection = self.get_curved_intersection_region(polygon, circle)
            if len(intersection) == 0:
                intersect_answer.become(original_cross)
                curved_intersection_region.become(VGroup())
            else:
                curved_region = self.convert_curved_region_into_mob(intersection, circle)
                curved_intersection_region.become(curved_region)
                intersect_answer.become(original_check)

        shapes.add_updater(curved_intersection_region_updater)
        polygon, circle = shapes

        self.play(
            polygon.shift, RIGHT,
            circle.shift, LEFT,
            run_time=2,
        )

        self.play(
            Rotate(polygon, 5 * PI / 6),
            run_time=2,
        )

        self.play(
            polygon.shift, RIGHT * 1.5 + UP,
            circle.shift, LEFT * 1.5 + DOWN * 0.2,
            run_time=2
        )

        self.play(
            polygon.scale, 0.8,
        )
        new_polygon = polygon.copy().scale(1 / 0.8).shift(RIGHT * 1.7 + DOWN)

        self.play(
            Transform(polygon, new_polygon),
            circle.shift, LEFT * 1.7 + UP * 0.2,
            run_time=2,
        )

        self.wait()
        shapes.clear_updaters()
        return shapes

    def concave_polygon_demo(self, shapes, cross, check):
        concave_polygon, concave_polygon_internal = self.get_spiral_polygon()
        concave_polygon.move_to(shapes[1].get_center())
        concave_polygon.set_color(shapes[1].get_color())
        p2 = self.get_interesting_polygon()
        p2.move_to(shapes[0].get_center())
        self.play(
            ReplacementTransform(shapes[0], p2),
            ReplacementTransform(shapes[1], concave_polygon)
        )
        self.wait()

        new_shapes = VGroup(p2, concave_polygon)
        self.add(new_shapes)
        intersection_region = VGroup()
        self.add(intersection_region)
        original_cross = cross.copy()
        original_check = check.copy()
        intersect_answer = cross
        self.add(intersect_answer)

        def concave_region_updater(shapes):
            polygon, concave = new_shapes
            intersection = self.get_concave_intersection_region(concave, polygon)
            if len(intersection) == 0:
                intersect_answer.become(original_cross)
            else:
                intersect_answer.become(original_check)
            intersection_region.become(intersection)


        new_shapes.add_updater(concave_region_updater)
        
        polygon, concave = new_shapes
        self.play(
            concave.shift, RIGHT,
            polygon.shift, LEFT,
            run_time=2,
        )

        self.play(
            polygon.shift, UP * 1.5 + LEFT,
            Rotate(concave, -PI / 2),
            run_time=2,
        )

        self.play(
            Rotate(polygon, PI / 2),
            concave.shift, RIGHT,
            run_time=2
        )

        new_polygon = polygon.copy()
        new_polygon.scale(0.8).rotate(PI / 2).shift(LEFT + DOWN * 1.5)

        self.play(
            Transform(polygon, new_polygon),
            Rotate(concave, PI / 2),
            run_time=2,
        )

        new_polygon = polygon.copy().scale(1 / 0.8).shift(LEFT * 1.2)
        self.play(
            Transform(polygon, new_polygon),
            concave.shift, RIGHT * 1.2,
            run_time=2
        )

        self.wait()
        
    def get_concave_intersection_poly(self, internal_representation, polygon):
        region = VGroup()
        for triangle in internal_representation:
            intersection_region = self.get_intersection_region(triangle, polygon)
            intersection_region_poly = self.get_intersection_poly(intersection_region, opacity=0.5)
            region.add(intersection_region_poly)
        
        return region

    def get_polygons(self):
        p1 = RegularPolygon(n=3).scale(2).set_stroke(color=self.p1_color, width=5)
        p2 = RegularPolygon(n=4).scale(2).set_stroke(color=self.p2_color, width=5)
        VGroup(p1, p2).arrange_submobjects(RIGHT).move_to(ORIGIN)
        return VGroup(p1, p2)

    def get_intersection_poly(self, intersection, color=FUCHSIA, width=5, opacity=0.5):
        if len(intersection) == 0:
            return VGroup()
        intersection_poly = Polygon(*intersection)
        intersection_poly.set_stroke(color=color, width=width)
        intersection_poly.set_fill(color=color, opacity=opacity)
        return intersection_poly

    def get_different_polygons(self, p1, p2):
        p1_new = RegularPolygon(n=7).scale(2).set_stroke(color=p1.get_color(), width=5)
        p2_new = RegularPolygon(n=8).scale(2).set_stroke(color=p2.get_color(), width=5)
        p1_new.move_to(p1.get_center())
        p2_new.move_to(p2.get_center())
        p = VGroup(p1_new, p2_new)
        return p

    def get_spiral_polygon(self):
        internal_representation = VGroup()
        ### 
        # 1. create triangle, 
        # 2. new triangle has length of previous triangle's height
        # 3. rotate new triangle around central point PI / 6
        # 4. Repeat for 8 triangles
        ###
        scale = 1.8
        angle = PI
        base_triangle = RegularPolygon(n=3).scale(scale).set_stroke(color=self.p1_color, width=5)
        base_triangle.rotate(angle)
        base_point = base_triangle.get_vertices()[0]
        internal_representation.add(base_triangle)
        colors = [BLUE, VIOLET, YELLOW]
        for i in range(8):
            scale = scale * (np.sqrt(3) / 2)
            angle -= PI / 6
            triangle = RegularPolygon(n=3).scale(scale).set_stroke(color=colors[i % len(colors)], width=5)
            triangle.rotate(angle)
            shift = base_point - triangle.get_vertices()[0]
            triangle.shift(shift)
            internal_representation.add(triangle)

        points_on_external = self.get_external_key_points(internal_representation)
        external_representation = Polygon(*points_on_external)        
        return external_representation, internal_representation

    def get_external_key_points(self, representation):
        points = [representation[0].get_vertices()[i] for i in [0, 2, 1]]
        pattern = [2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7]
        for i, index in enumerate(pattern):
            vertices = representation[index].get_vertices()
            if i % 2 == 0:
                points.append(vertices[2])
            else:
                points.append(vertices[1])

        second_last_point = (representation[-1].get_vertices()[1] + representation[-1].get_vertices()[2]) / 2
        last_point = representation[-1].get_vertices()[1]
        points.extend([second_last_point, last_point])
        return points

    def get_interesting_polygon(self):
        points = [LEFT, UP * 0.8, RIGHT + UP * 0.5, RIGHT * 1.2 + UP * 0.1, DOWN, LEFT * 0.7 + DOWN * 0.7]
        polygon = Polygon(*points).set_stroke(color=self.p1_color, width=5)
        polygon.scale(2)
        return polygon

    def get_abnormal_polygons(self):
        pass

class GJKIntroText(Scene):
    def construct(self):
        title = TextMobject("Gilbert Johnson Keerthi (GJK) Algorithm").scale(1.2)
        title.move_to(UP * 3.5)
        h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
        h_line.next_to(title, DOWN)
        self.play(
            Write(title),
            ShowCreation(h_line)
        )
        self.wait()

        path_to_solution = TextMobject("Path to solution")
        path_to_solution.next_to(h_line, DOWN)
        self.play(
            Write(path_to_solution)
        )

        self.wait()

        start = LEFT * 4 + UP * 1
        end = RIGHT * 4 + UP * 1

        start_dot = Dot().move_to(start)
        end_dot = Dot().move_to(end)

        first_path = [start, LEFT * 2 + UP * 0.8, RIGHT * 2 + UP * 1.3, end]

        first_path_mob = VGroup().set_points_as_corners(*[first_path])

        shift = UP * 0.3
        start_dot.shift(shift)
        end_dot.shift(shift)
        first_path_mob.shift(shift)

        other_algorithms = TextMobject("Other algorithms").scale(0.8)
        legend = Line(ORIGIN, RIGHT).scale(0.5)
        other_algorithms = VGroup(legend, other_algorithms).arrange(RIGHT)
        other_algorithms.next_to(first_path_mob, DOWN)

        self.play(
            GrowFromCenter(start_dot)
        )

        self.play(
            ShowCreation(first_path_mob),
            Write(other_algorithms),
            run_time=3
        )
        self.play(
            GrowFromCenter(end_dot)
        )

        self.wait()

        start_gjk = LEFT * 4 + DOWN * 1.5
        end_gjk = RIGHT * 4 + DOWN * 1.5
        start_gjk_dot = Dot().move_to(start_gjk)
        end_gjk_dot = Dot().move_to(end_gjk)

        roller_coaster_path = [
        start_gjk, 
        LEFT * 3 + DOWN * 3, LEFT * 2, LEFT * 1 + DOWN * 2, DOWN * 1.5,
        RIGHT + DOWN * 1.5, RIGHT * 1.5 + DOWN * 1.5,
        RIGHT * 2 + DOWN, RIGHT * 1.5 + DOWN * 0.5, RIGHT * 1 + DOWN, RIGHT * 1.5 + DOWN * 1.5,
        RIGHT * 2.5 + DOWN * 1.5, RIGHT * 3.25 + DOWN * 2.5,
        end_gjk
        ]
        roller_coaster_path_mob = VGroup().set_points_smoothly(*[roller_coaster_path])
        roller_coaster_path_mob.set_color(YELLOW)
        
        gjk_legend_text = TextMobject("GJK").scale(0.8)
        legend_line = Line(ORIGIN, RIGHT).scale(0.5).set_color(YELLOW)
        gjk_legend = VGroup(legend_line, gjk_legend_text).arrange(RIGHT)
        gjk_legend.move_to(DOWN * 2.8)

        self.play(
            GrowFromCenter(start_gjk_dot)
        )
        self.add_foreground_mobject(start_gjk_dot)

        self.play(
            ShowCreation(roller_coaster_path_mob),
            Write(gjk_legend),
            run_time=3
        )

        self.play(
            GrowFromCenter(end_gjk_dot)
        )

        self.add_foreground_mobject(end_gjk_dot)

        self.wait()

        self.remove(
            path_to_solution, first_path_mob, other_algorithms, 
            gjk_legend, start_gjk_dot, end_gjk_dot, roller_coaster_path_mob,
            start_dot, end_dot,
        )

        efficient = TextMobject("Extraordinarly Efficient")
        efficient.next_to(h_line, DOWN)
        self.play(
            Write(efficient)
        )
        self.wait()

        graphics_img = ImageMobject("teacup").scale(1.8)
        graphics_img.move_to(LEFT * 3.5)

        graphics = TextMobject("Computer" + "\\\\" + "graphics").move_to(LEFT * 3 + DOWN * 2)
        graphics.next_to(graphics_img, DOWN).shift(RIGHT * SMALL_BUFF * 4)


        robot_image = ImageMobject("robot_black").scale(2)
        robot_image.move_to(RIGHT * 3.5)

        robotics = TextMobject("Robotics").move_to(RIGHT * 3 + DOWN * 2)
        robotics.next_to(robot_image, DOWN)
        
        self.play(
            FadeIn(graphics_img),
            Write(graphics),
        )

        self.wait()

        self.play(
            FadeIn(robot_image),
            Write(robotics)
        )

        self.wait()

        screen_rect = ScreenRectangle(height=5).move_to(DOWN * 0.5)
        
        self.remove(efficient, graphics_img, robot_image, robotics, graphics)

        self.play(
            Write(screen_rect)
        )
        self.wait()

        self.play(
            FadeOut(screen_rect),
        )

        shift_up = UP * 1.5
        roller_coaster_path_mob.shift(shift_up)
        gjk_legend.shift(shift_up)
        start_gjk_dot.move_to(roller_coaster_path[0] + shift_up)
        end_gjk_dot.move_to(roller_coaster_path[-1] + shift_up)

        self.play(
            Write(roller_coaster_path_mob),
            Write(gjk_legend),
            run_time=2
        )
        self.wait()

        viewing_experience_text = TextMobject("Your viewing experience").scale(0.8)
        viewing_experience_dot = Dot().set_color(GREEN_SCREEN)
        viewing_experience = VGroup(viewing_experience_dot, viewing_experience_text).arrange(RIGHT)
        viewing_experience.move_to(DOWN * 3.5 + shift_up)

        self.play(
            Write(viewing_experience),
            run_time=1
        )

        dot_to_move = Dot().move_to(start_gjk + shift_up).set_color(GREEN_SCREEN)
        dot_to_move.scale(1.1)
        self.add_foreground_mobject(dot_to_move)

        self.play(
            MoveAlongPath(dot_to_move, roller_coaster_path_mob),
            rate_func=smooth,
            run_time=5,
        )

        self.wait()

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

class CrossX(VMobject):
    CONFIG = {
        "fill_opacity": 1,
        "stroke_width": 0,
        "width": 4,
        "propagate_style_to_family": True
    }
    def __init__(self, **kwargs):
        VMobject.__init__(self, **kwargs)
        background_circle = Circle()
        background_circle.set_color(CROSS_RED)
        background_circle.set_fill(CROSS_RED, opacity=1)
        self.add(background_circle)
        rounded_rect1 = RoundedRectangle(width=4, height=1, corner_radius=0.25).scale(0.3).rotate(PI / 4)
        rounded_rect1.set_color(WHITE).set_fill(WHITE, opacity=1)
        rounded_rect2 = RoundedRectangle(width=4, height=1, corner_radius=0.25).scale(0.3).rotate(-PI / 4)
        rounded_rect2.set_color(WHITE).set_fill(WHITE, opacity=1)
        self.add(rounded_rect1)
        self.add(rounded_rect2)

class ConvexVsConcave(Scene):
    def construct(self):
        title = TextMobject("Convex vs Concave Shapes").scale(1.2)
        title.move_to(UP * 3.5)
        h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
        h_line.next_to(title, DOWN)
        self.play(
            Write(title),
            ShowCreation(h_line)
        )
        self.wait()

        self.play(
            FadeOut(title),
            FadeOut(h_line),
        )
        self.wait()

class Convexity(Scene):
    CONFIG = {
    "update_time": 2
    }
    def construct(self):
        polygons = self.get_convex_polygons()
        self.demo_convexity(polygons)

    def get_convex_polygons(self):
        p1 = RegularPolygon(n=6).rotate(PI/6).scale(2).set_color(YELLOW).shift(LEFT * 3)
        p2 = RegularPolygon(n=8).scale(2).set_color(YELLOW).shift(LEFT * 3)

        return [p1, p2]

    def demo_convexity(self, polygons):
        self.wait()
        self.play(
            ShowCreation(polygons[0])
        )
        self.wait(4)

        convex = TextMobject("Convex")
        convex.next_to(polygons[0], DOWN * 2)
        self.play(
            Write(convex)
        )
        self.wait()

        polygon, line_seg = self.update_animations(polygons[0])

        self.play(
            FadeOut(line_seg),
            ReplacementTransform(polygon, polygons[1])
        )
        _, line_seg = self.update_animations(polygons[1])
        self.wait()

        easier = TextMobject("Easier to handle").scale(0.9)
        easier.next_to(convex, DOWN)


        interesting_poly  = self.get_interesting_polygon()
        interesting_poly.move_to(polygons[1].get_center())
        self.play(
            FadeOut(line_seg),
            ReplacementTransform(polygons[1], interesting_poly),
            Write(easier),
        )
        _, line_seg = self.update_animations(interesting_poly)
        self.wait(3)

        self.play(
            FadeOut(convex),
            FadeOut(interesting_poly),
            FadeOut(line_seg),
            FadeOut(easier)
        )
        self.wait()

    def update_animations(self, polygon):
        line_color = GREEN_SCREEN
        s, e = 0, 0.5
        start = polygon.point_from_proportion(s)
        end = polygon.point_from_proportion(e)
        proportions_start, proportions_end = self.get_update_proportions(s, e, s, 0.8)
        e = 0.8
        line_seg = VGroup()
        start_point = Dot().move_to(start)
        end_point = Dot().move_to(end)
        line = Line(start_point.get_center(), end_point.get_center())
        line_seg.add(line)
        line_seg.add(start_point)
        line_seg.add(end_point)
        line_seg.set_color(line_color)
        self.play(
            GrowFromCenter(start_point),
            GrowFromCenter(end_point),
        )

        self.play(
            ShowCreation(line)
        )

        self.wait()
        self.frame = 1
        frames = self.camera.frame_rate * self.update_time
        
        def update_line(line_seg, dt):
            line, dot1, dot2 = line_seg
            if self.frame > frames:
                return
            d1 = polygon.point_from_proportion(proportions_start[self.frame]) - dot1.get_center()
            d2 = polygon.point_from_proportion(proportions_end[self.frame]) - dot2.get_center()
            dot1.shift(d1)
            dot2.shift(d2)
            new_line = Line(dot1.get_center(), dot2.get_center()).set_color(GREEN_SCREEN)
            line.become(new_line)
            self.frame += 1

        line_seg.add_updater(update_line)
        self.add(line_seg)
        self.wait(self.update_time)
        line_seg.clear_updaters()
        
        self.frame = 1
        proportions_start, proportions_end = self.get_update_proportions(s, e, s, 0.2)
        e = 0.2
        line_seg.add_updater(update_line)
        self.add(line_seg)
        self.wait(self.update_time)
        line_seg.clear_updaters()

        self.frame = 1
        proportions_start, proportions_end = self.get_update_proportions(s, e, 0.5, 1)
        line_seg.add_updater(update_line)
        self.add(line_seg)
        self.wait(self.update_time)
        line_seg.clear_updaters()

        return polygon, line_seg        

    def get_update_proportions(self, s, e, s_end, e_end):
        frames = self.camera.frame_rate * self.update_time
        proportions_start = smooth_interpolate(s, s_end, frames)
        proportions_end = smooth_interpolate(e, e_end, frames)
        return proportions_start, proportions_end

    def get_interesting_polygon(self):
        points = [LEFT, UP * 0.8, RIGHT + UP * 0.5, RIGHT * 1.2 + UP * 0.1, DOWN, LEFT * 0.7 + DOWN * 0.7]
        polygon = Polygon(*points).set_stroke(color=YELLOW, width=5)
        polygon.scale(2)
        return polygon

class Concavity(GJKUtils):
    CONFIG = {
        "concave_color": PERIWINKLE_BLUE
    }
    def construct(self):
        polygons = self.get_concave_polygons()
        self.wait()
        self.play(
            ShowCreation(polygons[0]),
        )
        concave = TextMobject("Concave").next_to(polygons[0], DOWN * 2)
        self.play(
            Write(concave)
        )

        start, end = 0.5, 0.8
        line_seg = self.make_line_seg(start, end, polygons[0])

        self.wait()

        self.play(
            GrowFromCenter(line_seg[1]),
            GrowFromCenter(line_seg[2]),
        )

        self.play(
            ShowCreation(line_seg[0])
        )

        self.wait()

        self.play(
            ReplacementTransform(polygons[0], polygons[1]),
            FadeOut(line_seg)
        )

        start, end = 0.05, 0.8
        line_seg = self.make_line_seg(start, end, polygons[1])
        self.play(
            GrowFromCenter(line_seg[1]),
            GrowFromCenter(line_seg[2]),
        )

        self.play(
            ShowCreation(line_seg[0])
        )
        self.wait()

        sprial_polygon, sprial_polygon_internal = self.get_spiral_polygon()
        sprial_polygon.move_to(polygons[1].get_center())
        sprial_polygon_internal.move_to(polygons[1].get_center())
        self.play(
            ReplacementTransform(polygons[1], sprial_polygon),
            FadeOut(line_seg)
        )
        self.wait()

        start, end = 0.05, 0.9
        line_seg = self.make_line_seg(start, end, sprial_polygon)
        self.play(
            GrowFromCenter(line_seg[1]),
            GrowFromCenter(line_seg[2]),
        )

        self.play(
            ShowCreation(line_seg[0])
        )
        self.wait()

        first_split_ext, first_split_int = self.get_split_polygon(sprial_polygon)

        original_spiral = sprial_polygon.copy().move_to(ORIGIN)
        self.play(
            FadeOut(line_seg),
            ReplacementTransform(sprial_polygon, first_split_ext)
        )
        self.wait()

        start, end = 0.6, 0.8
        line_seg = self.make_line_seg(start, end, first_split_ext)
        self.play(
            GrowFromCenter(line_seg[1]),
            GrowFromCenter(line_seg[2]),
        )

        self.play(
            ShowCreation(line_seg[0])
        )
        self.wait()


        cross = Cross(concave)
        harder = TextMobject("Really hard").scale(0.9)
        harder.next_to(concave, DOWN)
        self.play(
            Write(cross),
            Write(harder)
        )

        self.wait()
        
        self.play(
            FadeOut(line_seg),
            FadeOut(concave),
            FadeOut(cross),
            FadeOut(harder),
            first_split_ext.move_to, ORIGIN
        )

        self.wait()

        first_split_int.move_to(ORIGIN)

        self.play(
            ShowCreation(first_split_int),
            run_time=2
        )
        self.wait()

        self.remove(first_split_ext)

        self.play(
            first_split_int[0].shift, LEFT,
            first_split_int[1].shift, RIGHT,
            run_time=1
        )

        self.play(
            first_split_int[0].shift, RIGHT,
            first_split_int[1].shift, LEFT,
            run_time=1,
        )

        self.wait()

        sprial_polygon.move_to(ORIGIN)

        self.play(
            ReplacementTransform(first_split_int, original_spiral),
            run_time=1
        )
        self.wait()


        sprial_polygon_internal.move_to(ORIGIN)


        self.play(
            ShowCreation(sprial_polygon_internal),
            run_time=2,
        )

        self.wait()
        self.remove(original_spiral)


        directions = [self.angle_to_unit_v(PI / 2 - (PI / 6 * i)) * 1.1 for i in range(9)]

        self.play(
            *[ApplyMethod(t.shift, directions[i]) for i, t in enumerate(sprial_polygon_internal)],
            run_time=2
        )

        self.play(
            *[ApplyMethod(t.shift, -directions[i]) for i, t in enumerate(sprial_polygon_internal)],
            run_time=2
        )

        self.wait()

        crossX = CrossX().scale(0.4)
        check = CheckMark().scale(0.2)

        original_spiral.move_to(RIGHT * 3)

        self.play(
            sprial_polygon_internal.shift, LEFT * 3,
            Write(original_spiral),
            run_time=1
        )

        self.wait()

        convex1 = self.get_interesting_polygon()
        convex2 = convex1.copy()

        start_shift = LEFT * 2 + UP * 3
        convex1.move_to(sprial_polygon_internal.get_center() + start_shift)
        convex2.move_to(original_spiral.get_center() + start_shift)

        left_cross = crossX.copy()
        right_cross = crossX.copy()
        left_cross.next_to(sprial_polygon_internal, DOWN * 2)
        right_cross.next_to(original_spiral, DOWN * 2)


        self.play(
            ShowCreation(convex1),
            ShowCreation(convex2),
        )

        self.wait()

        self.add(left_cross, right_cross)

        self.wait()

        left_polygons = VGroup(convex1, sprial_polygon_internal)
        right_polygons = VGroup(convex2, original_spiral)
        self.concave_polygon_demo(left_polygons, right_polygons, left_cross, right_cross, check)

    def angle_to_unit_v(self, angle):
        return np.array([np.cos(angle), np.sin(angle), 0])

    def concave_polygon_demo(self, 
        left_polygons, right_polygons, left_cross, right_cross, check):
        
        right_p, right_concave_polygon = right_polygons
        new_shapes_right = VGroup(right_p, right_concave_polygon)
        self.add(new_shapes_right)
        right_intersection_region = VGroup()
        self.add(right_intersection_region)
        right_original_cross = right_cross.copy()
        right_original_check = check.copy().move_to(right_original_cross.get_center())
        right_intersect_answer = right_cross
        self.add(right_intersect_answer)

        def concave_region_updater_right(shapes):
            polygon, concave = new_shapes_right
            intersection = self.get_concave_intersection_region(concave, polygon)
            if len(intersection) == 0:
                right_intersect_answer.become(right_original_cross)
            else:
                right_intersect_answer.become(right_original_check)
            right_intersection_region.become(intersection)


        new_shapes_right.add_updater(concave_region_updater_right)
        

        left_p, left_concave_polygon = left_polygons
        new_shapes_left = VGroup(left_p, left_concave_polygon)
        self.add(new_shapes_left)
        left_intersection_region = VGroup()
        self.add(left_intersection_region)
        left_original_cross = left_cross.copy()
        left_original_check = check.copy().move_to(left_original_cross.get_center())
        left_intersect_answer = left_cross
        self.add(left_intersect_answer)

        def concave_region_updater_left(shapes):
            polygon, concave = new_shapes_left
            intersection = self.get_concave_intersection_poly(concave, polygon)
            if len(intersection) == 0:
                left_intersect_answer.become(left_original_cross)
            else:
                left_intersect_answer.become(left_original_check)
            left_intersection_region.become(intersection)

        new_shapes_left.add_updater(concave_region_updater_left)

        left_polygon, left_concave = new_shapes_left
        right_polygon, right_concave = new_shapes_right
        self.play(
            left_polygon.shift, RIGHT * 3.5 + DOWN * 5,
            right_polygon.shift, RIGHT * 3.5 + DOWN * 5,
            run_time=3,
        )
        self.play(
            left_polygon.shift, UP * 5.1,
            right_polygon.shift, UP * 5.1,
            run_time=3,
        )

        self.wait()

        
    def get_concave_intersection_poly(self, internal_representation, polygon):
        region = VGroup()
        for triangle in internal_representation:
            intersection_region = self.get_intersection_region(triangle, polygon)
            if len(intersection_region) == 0:
                continue    
            intersection_region_poly = self.get_intersection_poly(intersection_region, color=triangle.get_color(), opacity=0.5)
            region.add(intersection_region_poly)
        
        return region

    def get_intersection_poly(self, intersection, color=FUCHSIA, width=5, opacity=0.5):
        if len(intersection) == 0:
            return VGroup()
        intersection_poly = Polygon(*intersection)
        intersection_poly.set_stroke(color=color, width=width)
        intersection_poly.set_fill(color=color, opacity=opacity)
        return intersection_poly

    def get_concave_polygons(self):
        points1 = [
        LEFT * 1 + UP * 2, RIGHT * 1 + UP * 2,
        RIGHT * 1.5 + DOWN * 2, ORIGIN,
        LEFT * 0.5 + DOWN * 2,
        ]
        p1 = Polygon(*points1).shift(RIGHT * 3).set_stroke(color=self.concave_color, width=5)

        points2 = [v for v in RegularPolygon(n=8).scale(2).get_vertices()[:7]]
        points2.append(RIGHT * 0.5 + DOWN * 0.5)
        p2 = Polygon(*points2).shift(RIGHT * 3).set_stroke(color=self.concave_color, width=5)

        return [p1, p2]

    def get_split_polygon(self, p):
        triangle1 = Polygon(*[LEFT * 2 + UP, DOWN * 2, ORIGIN])
        triangle1.set_stroke(color=BLUE, width=5)
        triangle2 = Polygon(*[RIGHT * 2 + UP, DOWN * 2, ORIGIN])
        triangle2.set_stroke(color=YELLOW, width=5)
        internal = VGroup(triangle1, triangle2)
        external = Polygon(*[LEFT * 2 + UP, DOWN * 2, RIGHT * 2 + UP, ORIGIN])
        external.set_stroke(color=self.concave_color, width=5)
        external.move_to(p.get_center())
        internal.move_to(p.get_center())
        return external, internal

    def make_line_seg(self, start, end, polygon, color=YELLOW):
        start_point = Dot().move_to(polygon.point_from_proportion(start))
        end_point = Dot().move_to(polygon.point_from_proportion(end))
        start_point.set_color(color)
        end_point.set_color(color)
        line = Line(start_point, end_point).set_stroke(color=color, width=5)
        return VGroup(line, start_point, end_point)

    def get_interesting_polygon(self):
        points = [LEFT, UP * 0.8, RIGHT + UP * 0.5, RIGHT * 1.2 + UP * 0.1, DOWN, LEFT * 0.7 + DOWN * 0.7]
        polygon = Polygon(*points).set_stroke(color=YELLOW, width=5)
        polygon.scale(1.2).set_stroke(color=YELLOW, width=5)
        polygon.rotate(-PI/5)
        return polygon

    def get_spiral_polygon(self):
        internal_representation = VGroup()
        ### 
        # 1. create triangle, 
        # 2. new triangle has length of previous triangle's height
        # 3. rotate new triangle around central point PI / 6
        # 4. Repeat for 8 triangles
        ###
        scale = 1.8
        angle = PI
        base_triangle = RegularPolygon(n=3).scale(scale).set_stroke(color=GREEN_SCREEN, width=5)
        base_triangle.rotate(angle)
        base_point = base_triangle.get_vertices()[0]
        internal_representation.add(base_triangle)
        colors = [BLUE, VIOLET, GREEN_SCREEN]
        for i in range(8):
            scale = scale * (np.sqrt(3) / 2)
            angle -= PI / 6
            triangle = RegularPolygon(n=3).scale(scale).set_stroke(color=colors[i % len(colors)], width=5)
            triangle.rotate(angle)
            shift = base_point - triangle.get_vertices()[0]
            triangle.shift(shift)
            internal_representation.add(triangle)

        points_on_external = self.get_external_key_points(internal_representation)
        external_representation = Polygon(*points_on_external).set_stroke(color=self.concave_color, width=5)        
        return external_representation, internal_representation

    def get_external_key_points(self, representation):
        points = [representation[0].get_vertices()[i] for i in [0, 2, 1]]
        pattern = [2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 8, 7]
        for i, index in enumerate(pattern):
            vertices = representation[index].get_vertices()
            if i % 2 == 0:
                points.append(vertices[2])
            else:
                points.append(vertices[1])

        second_last_point = (representation[-1].get_vertices()[1] + representation[-1].get_vertices()[2]) / 2
        last_point = representation[-1].get_vertices()[1]
        points.extend([second_last_point, last_point])
        return points

class InfinitePoints(GJKUtils):
    CONFIG = {
        "c_color": YELLOW,
        "p_color": PERIWINKLE_BLUE,
    }
    def construct(self):
        plane = NumberPlane()

        circle = Circle(radius=2).set_color(self.c_color)
        polygon = RegularPolygon(n=9).scale(2).set_color(self.p_color)
        shapes = VGroup(circle, polygon).arrange_submobjects(RIGHT, buff=0)

        circle.shift(LEFT * SMALL_BUFF * 2)
        polygon.shift(RIGHT * SMALL_BUFF * 2)

        self.play(
            Write(circle),
            Write(polygon),
        )

        self.wait()

        self.play(
            circle.shift, RIGHT * SMALL_BUFF * 4,
            polygon.shift, LEFT * SMALL_BUFF * 4
        )

        num_points = 8000
        circle_dots = []
        for _ in range(num_points):
            point = self.random_point_in_circle(circle)
            dot = Dot().scale(0.3).set_fill(color=self.c_color, opacity=0.5).move_to(point)
            circle_dots.append(dot)

        circle_dots = VGroup(*circle_dots)

        self.play(
            circle.set_stroke, self.c_color, 0.5,
            polygon.set_stroke, self.p_color, 0.5,
            Write(plane)
        )

        self.wait()

        num_points = 8000
        polygon_dots = []
        for _ in range(num_points):
            point = self.random_point_in_polygon(polygon)
            dot = Dot().scale(0.3).set_fill(color=self.p_color, opacity=0.5).move_to(point)
            polygon_dots.append(dot)
        polygon_dots = VGroup(*polygon_dots)
        self.play(
            LaggedStartMap(GrowFromCenter, circle_dots),
            LaggedStartMap(GrowFromCenter, polygon_dots),
            run_time=2
        )

        self.wait()

        self.add(circle_dots, polygon_dots)
        self.wait()

        intersection_region = self.get_curved_intersection_region(polygon, circle)
        intersection_highlight = self.convert_curved_region_into_mob(intersection_region, circle)
        intersection_highlight.set_fill(opacity=0)
        intersection_highlight.set_stroke(color=GREEN_SCREEN, width=5)
        self.play(
            ShowCreation(intersection_highlight)
        )

        self.wait()

        best_dot_p = self.find_dot_closest_to_origin(polygon_dots)
        best_dot_c = self.find_dot_closest_to_origin(circle_dots)


        self.move_camera(0.8*np.pi/2, -0.45*np.pi)
        self.wait()

        new_dot_c = best_dot_c.copy().scale(1 / 0.3).shift(LEFT * 0.5 + OUT * 2.5)
        new_dot_p = best_dot_p.copy().scale(1 / 0.3).shift(RIGHT * 0.5 + OUT * 2.5)

        new_dot_c.rotate(PI / 2, axis=X_AXIS).set_fill(opacity=1)
        new_dot_p.rotate(PI / 2, axis=X_AXIS).set_fill(opacity=1)

        self.play(
            TransformFromCopy(best_dot_c, new_dot_c),
            TransformFromCopy(best_dot_p, new_dot_p),
            run_time=2
        )

        self.wait()

        equal = TexMobject("=").move_to((new_dot_c.get_center() + new_dot_p.get_center()) / 2)
        equal.scale(0.8).rotate(PI / 2, axis=X_AXIS)

        c_coord = TexMobject("(x_1, y_1)").scale(0.8)
        p_coord = TexMobject("(x_2, y_2)").scale(0.8)

        c_coord.next_to(new_dot_c, LEFT).rotate(PI / 2, axis=X_AXIS)
        p_coord.next_to(new_dot_p, RIGHT).rotate(PI / 2, axis=X_AXIS)

        self.play(
            Write(c_coord),
            Write(p_coord)
        )

        self.play(
            Write(equal)
        )

        self.wait()

        group = VGroup(c_coord, new_dot_c, equal, new_dot_p, p_coord)

        intersection = TextMobject("Intersection?").rotate(PI / 2, axis=X_AXIS)
        check = CheckMark().scale(0.13).rotate(PI / 2, axis=X_AXIS)
        intersection_check = VGroup(intersection, check).arrange_submobjects(RIGHT)
        intersection_check.next_to(group, OUT)

        self.play(
            Write(intersection_check)
        )
        self.wait()

        minus = TexMobject("-").scale(0.8).rotate(PI / 2, axis=X_AXIS)
        origin = TexMobject("(0, 0)").scale(0.8).rotate(PI / 2, axis=X_AXIS)
        origin_check = VGroup(
            c_coord.copy(), 
            minus,
            p_coord.copy(),
            equal.copy(),
            origin,
        ).arrange(RIGHT).next_to(group, IN)

        self.play(
            TransformFromCopy(group[0], origin_check[0]),
            TransformFromCopy(group[4], origin_check[2]),
            TransformFromCopy(group[2], origin_check[3]),
            Write(origin_check[1]),
            Write(origin_check[4]),
            run_time=2,
        )

        self.wait()

        
        self.begin_ambient_camera_rotation()
        self.wait(6)

    def find_dot_closest_to_origin(self, dots):
        return min(dots, key=lambda x: np.linalg.norm(x.get_center()))

    def random_point_in_circle(self, c):
        """
        @param: c - Geometry.Circle
        return: np.array[x, y, 0], a point sampled uniformly inside circle
        """
        r = (np.random.uniform() + np.random.uniform()) * c.radius
        if r > c.radius:
            r = c.radius * 2 - r
        theta = np.random.uniform() * 2 * PI
        c_x, c_y = c.get_center()[:2]
        return np.array([c_x + r * np.cos(theta), c_y + r * np.sin(theta), 0])

    def random_point_in_polygon(self, p):
        """
        @param: p - Geometry.Polygon
        return: np.array[x, y, 0], a point sampled uniformly inside polygon
        """
        bb = SurroundingRectangle(p, buff=0)
        x_min = bb.get_center()[0] - (bb.get_width() / 2)
        x_max = bb.get_center()[0] + (bb.get_width() / 2)
        y_min = bb.get_center()[1] - (bb.get_height() / 2)
        y_max = bb.get_center()[1] + (bb.get_width() / 2)

        point = np.array([np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max), 0])
        while not self.check_point_inside(point, p):
            point = np.array([np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max), 0])
        return point

class MinkowskiDiffPointsEnd(GJKUtils):
    CONFIG = {
        "c_color": YELLOW,
        "p_color": PERIWINKLE_BLUE,
        "d_color": FUCHSIA,
    }
    def construct(self):
        start_time = time.time()
        plane = NumberPlane()
        self.add(plane)

        circle = Circle(radius=1).set_color(self.c_color)
        polygon = RegularPolygon(n=4).scale(1).set_color(self.p_color)
        circle.move_to(RIGHT * 1.5 + UP * 2)
        polygon.move_to(RIGHT * 3 + UP * 2)
        shapes = VGroup(circle, polygon)


        self.play(
            Write(circle),
            Write(polygon),
        )

        self.wait()

        num_points = 5000
        circle_dots = []
        for _ in range(num_points):
            point = self.random_point_in_circle(circle)
            dot = Dot().scale(0.3).set_fill(color=self.c_color, opacity=0.5).move_to(point)
            circle_dots.append(dot)

        circle_dots = VGroup(*circle_dots)
        

        circle_point_cloud = VGroup(circle, circle_dots)

        num_points = 3000
        polygon_dots = []
        for _ in range(num_points):
            point = self.random_point_in_polygon(polygon)
            dot = Dot().scale(0.3).set_fill(color=self.p_color, opacity=0.5).move_to(point)
            polygon_dots.append(dot)
        polygon_dots = VGroup(*polygon_dots)

        self.play(
            LaggedStartMap(GrowFromCenter, circle_dots),
            LaggedStartMap(GrowFromCenter, polygon_dots),
            run_time=2
        )

        polygon_point_cloud = VGroup(polygon, polygon_dots)

        self.wait()

        diff_dots = []
        num_points = 25000
        for _ in range(num_points):
            point = self.get_random_diff(circle_dots, polygon_dots)
            dot = Dot().scale(0.3).set_fill(color=self.d_color, opacity=0.5).move_to(point)
            diff_dots.append(dot)

        boundary_dots = []
        for _ in range(5000):
            p1 = circle.point_from_proportion(np.random.uniform())
            p2 = polygon.point_from_proportion(np.random.uniform())
            dot = Dot().scale(0.3).set_fill(color=self.d_color, opacity=0.5).move_to(p1 - p2)
            boundary_dots.append(dot)

        diff_dots.extend(boundary_dots)
        random.shuffle(diff_dots)
        diff_dots = VGroup(*diff_dots)

        self.play(
            LaggedStartMap(GrowFromCenter, diff_dots),
            run_time=3
        )

        self.wait()

        rounded_rectangle = RoundedRectangle(height=2*np.sqrt(2), width=2*np.sqrt(2), corner_radius=1).scale(1.25)
        rounded_rectangle.rotate(PI / 4)
        rounded_rectangle.set_stroke(color=self.d_color, width=7)
        rounded_rectangle.move_to(LEFT * 1.5)
        self.play(
            ShowCreation(rounded_rectangle)
        )

        self.wait(2)

        diff_point_cloud = VGroup(rounded_rectangle, diff_dots)

        A = TexMobject("A").add_background_rectangle()
        B = TexMobject("B").add_background_rectangle()
        C = TexMobject(r"A \ominus B").add_background_rectangle()

        A.next_to(circle_point_cloud, DOWN)
        B.next_to(polygon_point_cloud, DOWN)
        C.next_to(diff_point_cloud, DOWN)

        self.play(
            Write(A),
            Write(B),
            Write(C),
        )

        self.wait()

        property_1 = TextMobject(
            r"1. $A$ and $B$ convex $\rightarrow$ ",
            r"$A \ominus B$ convex").scale(0.75)
        property_2 = TextMobject(
            r"2. $A$ and $B$ intersect $\rightarrow$ ", 
            r"$(0, 0) \in A \ominus B$").scale(0.75)
        property_1.add_background_rectangle()
        property_2.add_background_rectangle()

        self.wait()
        property_1.to_edge(LEFT * 15).shift(DOWN * 1.5)
        property_2.next_to(property_1, DOWN).to_edge(LEFT * 15)
        self.play(
            Write(property_1)
        )
        self.wait()

        self.play(
            Write(property_2)
        )

        self.wait()

        shift_paths = [UP * 0.5 + RIGHT * 0.5, RIGHT * 1, DOWN * 1, LEFT * 2, UP * 0.5 + RIGHT * 0.5]

        for shift_amount in shift_paths:
            self.play(
                B.shift, shift_amount,
                polygon_point_cloud.shift, shift_amount,
                diff_point_cloud.shift, -shift_amount,
                C.shift, -shift_amount,
                rate_func=linear,
            )

        how_to_compute = TextMobject("How do we compute" + "\\\\" + "Minkowski differences?")
        how_to_compute.add_background_rectangle()
        how_to_compute.next_to(property_2, DOWN)
        self.play(
            Write(how_to_compute),
            run_time=2
        )
        self.wait()

        answer = TextMobject(r"As defined, infeasible.")
        answer.add_background_rectangle()
        answer.next_to(property_2, DOWN)

        self.play(
            ReplacementTransform(how_to_compute, answer)
        )
        self.wait()

        print("--- %s seconds ---" % (time.time() - start_time))

    def random_point_in_circle(self, c):
        """
        @param: c - Geometry.Circle
        return: np.array[x, y, 0], a point sampled uniformly inside circle
        """
        r = (np.random.uniform() + np.random.uniform()) * c.radius
        if r > c.radius:
            r = c.radius * 2 - r
        theta = np.random.uniform() * 2 * PI
        c_x, c_y = c.get_center()[:2]
        return np.array([c_x + r * np.cos(theta), c_y + r * np.sin(theta), 0])

    def random_point_in_polygon(self, p):
        """
        @param: p - Geometry.Polygon
        return: np.array[x, y, 0], a point sampled uniformly inside polygon
        """
        bb = SurroundingRectangle(p, buff=0)
        x_min = bb.get_center()[0] - (bb.get_width() / 2)
        x_max = bb.get_center()[0] + (bb.get_width() / 2)
        y_min = bb.get_center()[1] - (bb.get_height() / 2)
        y_max = bb.get_center()[1] + (bb.get_width() / 2)

        point = np.array([np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max), 0])
        while not self.check_point_inside(point, p):
            point = np.array([np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max), 0])
        return point

    def get_random_diff(self, dots_c, dots_p):
        return random.choice(dots_c).get_center() - random.choice(dots_p).get_center()

class MinkowskiDiffProperties(Scene):
    def construct(self):
        plane = NumberPlane()
        self.add(plane)

        property_1 = TextMobject(
            r"1. $A$ and $B$ convex $\rightarrow$ ",
            r"$A \ominus B$ convex").scale(0.75)
        property_2 = TextMobject(
            r"2. $A$ and $B$ intersect $\rightarrow$ ", 
            r"$(0, 0) \in A \ominus B$").scale(0.75)
        property_1.add_background_rectangle()
        property_2.add_background_rectangle()

        self.wait()
        property_1.to_edge(LEFT * 15).shift(DOWN * 1.5)
        property_2.next_to(property_1, DOWN).to_edge(LEFT * 15)
        self.play(
            Write(property_1)
        )
        self.wait()

        self.play(
            Write(property_2)
        )

        self.wait()

        how_to_compute = TextMobject("How do we compute" + "\\\\" + "Minkowski differences?")
        how_to_compute.add_background_rectangle()
        how_to_compute.next_to(property_2, DOWN)
        self.play(
            Write(how_to_compute),
            run_time=2
        )
        self.wait()

        answer = TextMobject(r"As defined, infeasible.")
        answer.add_background_rectangle()
        answer.next_to(property_2, DOWN)

        self.play(
            ReplacementTransform(how_to_compute, answer)
        )
        self.wait()


class MinkowskiSumIntro(GJKUtils):
    CONFIG = {
        "c_color": YELLOW,
        "p_color": PERIWINKLE_BLUE,
        "d_color": FUCHSIA,
    }
    def construct(self):
        start_time = time.time()
        title = TextMobject("Minkowski Sums/Differences").scale(1.2)
        title.move_to(UP * 3.5)
        h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
        h_line.next_to(title, DOWN)
        self.play(
            Write(title),
            ShowCreation(h_line)
        )

        self.wait()

        plane = self.get_plane_subset()


        circle = Circle(radius=1).set_color(self.c_color)
        polygon = RegularPolygon(n=4).scale(1).set_color(self.p_color)
        polygon.rotate(PI / 4)
        shapes = VGroup(circle, polygon)

        num_points = 500
        circle_dots = []
        for _ in range(num_points):
            point = self.random_point_in_circle(circle)
            dot = Dot().scale(0.3).set_fill(color=self.c_color, opacity=0.5).move_to(point)
            circle_dots.append(dot)

        circle_dots.sort(key=lambda x: x.get_center()[0])
        circle_dots = VGroup(*circle_dots)

        circle_point_cloud = VGroup(circle, circle_dots)

        num_points = 300
        polygon_dots = []
        for _ in range(num_points):
            point = self.random_point_in_polygon(polygon)
            dot = Dot().scale(0.3).set_fill(color=self.p_color, opacity=0.5).move_to(point)
            polygon_dots.append(dot)
        polygon_dots.sort(key=lambda x: x.get_center()[0])
        polygon_dots = VGroup(*polygon_dots)

        polygon_point_cloud = VGroup(polygon, polygon_dots)

        sum_dots = []
        num_points = 2500
        for _ in range(num_points):
            point = self.get_random_sum(circle_dots, polygon_dots)
            dot = Dot().scale(0.3).set_fill(color=self.d_color, opacity=0.5).move_to(point)
            sum_dots.append(dot)

        boundary_dots = []
        for _ in range(500):
            p1 = circle.point_from_proportion(np.random.uniform())
            p2 = polygon.point_from_proportion(np.random.uniform())
            dot = Dot().scale(0.3).set_fill(color=self.d_color, opacity=0.5).move_to(p1 + p2)
            boundary_dots.append(dot)

        sum_dots.extend(boundary_dots)
        sum_dots.sort(key=lambda x: x.get_center()[0])
        sum_dots = VGroup(*sum_dots)


        rounded_rectangle = self.get_mink_sum(circle, polygon)
        rounded_rectangle.set_stroke(color=self.d_color, width=7)

        sum_point_cloud = VGroup(rounded_rectangle, sum_dots)
        self.wait()

        all_point_clouds = VGroup(circle_point_cloud, polygon_point_cloud, sum_point_cloud)
        all_point_clouds.arrange(RIGHT, buff=2)
        
        A = TexMobject("A").scale(0.8)
        A.next_to(circle_point_cloud, DOWN)
        a_plane = plane.copy().move_to(circle.get_center())
        self.play(
            ShowCreation(circle_point_cloud[0]),
            FadeIn(circle_point_cloud[1]),
            Write(A),
            Write(a_plane),
        )

        self.wait()
        B = TexMobject("B").scale(0.8)
        B.next_to(polygon_point_cloud, DOWN * 2.2)
        b_plane = plane.copy().move_to(polygon.get_center())
        self.play(
            ShowCreation(polygon_point_cloud[0]),
            FadeIn(polygon_point_cloud[1]),
            Write(b_plane),
            Write(B)
        )

        self.wait()

        c_plane = self.get_larger_plane_subset()
        c_plane.move_to(sum_point_cloud[0].get_center())
        C = TexMobject(r"A \oplus B = \{a + b | a \in A, b \in B \}")
        C.scale(0.8)
        C.next_to(sum_point_cloud, DOWN * 1.5)
        self.play(
            ShowCreation(sum_point_cloud[0]),
            FadeIn(sum_point_cloud[1]),
            Write(c_plane),
            Write(C),
        )

        self.wait()

        brace = Brace(C, direction=DOWN)
        mink_sum = TextMobject(r"Minkowski sum of $A$ and $B$").scale(0.8)
        mink_sum.next_to(brace, DOWN)
        self.play(
            GrowFromCenter(brace),
            Write(mink_sum)
        )

        self.wait()

        arrows, dots = self.show_arrow_intuition(circle, polygon)

        components, swept_circle = self.sweep(circle, polygon)

        self.play(
            FadeOut(arrows),
            FadeOut(dots),
            FadeOut(swept_circle),
        )

        self.wait()

        SHIFT_UNIT = 1 - np.sqrt(2) / 2
        shifts = [
        LEFT * SHIFT_UNIT + UP * SHIFT_UNIT, RIGHT * 2 * SHIFT_UNIT,
        DOWN * 2 * SHIFT_UNIT, LEFT * 2 * SHIFT_UNIT, UP * 2 * SHIFT_UNIT,
        RIGHT * SHIFT_UNIT + DOWN * SHIFT_UNIT,
        ]

        for shift_amount in shifts:
            self.play(
                polygon_point_cloud.shift, shift_amount,
                components.shift, shift_amount,
                sum_point_cloud.shift, shift_amount,
                rate_func=linear
            )
        self.wait()

        C_diff = TexMobject(r"A \ominus B = \{a + (-b) | a \in A, b \in B \}")
        C_diff.scale(0.8)
        C_diff.next_to(sum_point_cloud, DOWN * 1.5)
        new_brace = Brace(C_diff, direction=DOWN)
        mink_diff = TextMobject(r"Minkowski difference of $A$ and $B$").scale(0.8)
        mink_diff.next_to(new_brace, DOWN)

        self.play(
            ReplacementTransform(C, C_diff),
            ReplacementTransform(brace, new_brace),
            ReplacementTransform(mink_sum, mink_diff),
            run_time=2
        )

        self.wait()

        for shift_amount in shifts:
            self.play(
                polygon_point_cloud.shift, shift_amount,
                components.shift, -shift_amount,
                sum_point_cloud.shift, -shift_amount,
                rate_func=linear
            )
        self.wait()

        print("--- %s seconds ---" % (time.time() - start_time))

    def show_arrow_intuition(self, circle, polygon):
        circle_arrows = [self.get_vector_arrow(circle.get_center(), circle.point_from_proportion(i / 8)) for i in range(8)]
        polygon_arrows = [self.get_vector_arrow(polygon.get_center(), polygon.point_from_proportion(i / 8)) for i in range(8)]
        # makes indexing between two consistent with directions
        last_arrow = polygon_arrows.pop()
        polygon_arrows.insert(0, last_arrow)

        mapping = {
        0: [0],
        1: [0, 1, 2],
        2: [2],
        3: [2, 3, 4],
        4: [4],
        5: [4, 5, 6],
        6: [6],
        7: [6, 7, 0],
        }
        
        for i in range(len(mapping)):
            circle_arrows[i].set_color(circle.get_color())
            polygon_arrows[i].set_color(polygon.get_color())
        
        self.play(
            *[Write(arrow) for arrow in circle_arrows + polygon_arrows]
        )

        self.wait()

        new_arrows = []
        transforms = []
        for i in range(len(mapping)):
            new_start = polygon_arrows[i].get_end()
            new_ends = [new_start + (circle_arrows[key].get_end() - circle_arrows[key].get_start()) for key in mapping[i]] 
            sub_new_arrows = [circle_arrows[key].copy().put_start_and_end_on(new_start, new_ends[j]) for j, key in enumerate(mapping[i])]
            new_arrows.extend(sub_new_arrows)
            sub_transforms = [TransformFromCopy(circle_arrows[key], sub_new_arrows[j]) for j, key in enumerate(mapping[i])]
            transforms.extend(sub_transforms)

        self.play(
            *transforms,
            run_time=3
        )
        self.wait()

        dots = VGroup(*[Dot().scale(0.5).set_color(self.d_color).move_to(arrow.get_end()) for arrow in new_arrows])
        self.play(
            LaggedStartMap(GrowFromCenter, dots),
        )

        self.wait()

        return VGroup(*circle_arrows + polygon_arrows + new_arrows), dots

    def sweep(self, circle, polygon):
        a = circle.copy()
        b = polygon.copy()

        a.move_to(b.get_vertices()[1])

        self.play(
            TransformFromCopy(circle, a)
        )
        self.wait()

        components = self.get_mink_sum(circle, polygon)

        self.play(
            ShowCreation(components[0]),
            rate_func=linear,
        )
        self.play(
            ShowCreation(components[1]),
            a.move_to, b.get_vertices()[0],
            rate_func=linear,
        )

        self.play(
            ShowCreation(components[2]),
            rate_func=linear,
        )

        self.play(
            ShowCreation(components[3]),
            a.move_to, b.get_vertices()[3],
            rate_func=linear,
        )

        self.play(
            ShowCreation(components[4]),
            rate_func=linear,
        )

        self.play(
            ShowCreation(components[5]),
            a.move_to, b.get_vertices()[2],
            rate_func=linear,
        )

        self.play(
            ShowCreation(components[6]),
            rate_func=linear,
        )

        self.play(
            ShowCreation(components[7]),
            a.move_to, b.get_vertices()[1],
            rate_func=linear,
        )


        self.wait()

        return components, a
        
    def get_mink_sum(self, circle, polygon):
        components = VGroup()
        a = circle.copy()
        b = polygon.copy()
        a.move_to(b.get_vertices()[1])
        components.add(
            Arc(
                start_angle=PI, angle=-PI/2, arc_center=a.get_center()
            ).set_stroke(color=self.d_color, width=7)
        )

        components.add(
            Line(
                components[-1].get_end(), b.get_vertices()[0] + UP
            ).set_stroke(color=self.d_color, width=7)
        )

        a.move_to(b.get_vertices()[0])

        components.add(
            Arc(
                start_angle=PI/2, angle=-PI/2, arc_center=a.get_center()
            ).set_stroke(color=self.d_color, width=7)
        )

        components.add(
            Line(
                components[-1].get_end(), b.get_vertices()[3] + RIGHT,
            ).set_stroke(color=self.d_color, width=7)
        )

        a.move_to(b.get_vertices()[3])

        components.add(
            Arc(
                start_angle=0, angle=-PI/2, arc_center=a.get_center()
            ).set_stroke(color=self.d_color, width=7)
        )

        components.add(
            Line(
                components[-1].get_end(), b.get_vertices()[2] + DOWN,
            ).set_stroke(color=self.d_color, width=7)
        )

        a.move_to(b.get_vertices()[2])
        components.add(
            Arc(
                start_angle=-PI/2, angle=-PI/2, arc_center=a.get_center()
            ).set_stroke(color=self.d_color, width=7)
        )

        components.add(
            Line(
                components[-1].get_end(), b.get_vertices()[1] + LEFT,
            ).set_stroke(color=self.d_color, width=7)
        )

        return components

    def random_point_in_circle(self, c):
        """
        @param: c - Geometry.Circle
        return: np.array[x, y, 0], a point sampled uniformly inside circle
        """
        r = (np.random.uniform() + np.random.uniform()) * c.radius
        if r > c.radius:
            r = c.radius * 2 - r
        theta = np.random.uniform() * 2 * PI
        c_x, c_y = c.get_center()[:2]
        return np.array([c_x + r * np.cos(theta), c_y + r * np.sin(theta), 0])

    def random_point_in_polygon(self, p):
        """
        @param: p - Geometry.Polygon
        return: np.array[x, y, 0], a point sampled uniformly inside polygon
        """
        bb = SurroundingRectangle(p, buff=0)
        x_min = bb.get_center()[0] - (bb.get_width() / 2)
        x_max = bb.get_center()[0] + (bb.get_width() / 2)
        y_min = bb.get_center()[1] - (bb.get_height() / 2)
        y_max = bb.get_center()[1] + (bb.get_width() / 2)

        point = np.array([np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max), 0])
        while not self.check_point_inside(point, p):
            point = np.array([np.random.uniform(x_min, x_max), np.random.uniform(y_min, y_max), 0])
        return point

    def get_random_sum(self, dots_c, dots_p):
        return random.choice(dots_c).get_center() + random.choice(dots_p).get_center()

    def get_plane_subset(self):
        subset_complex_plane = VGroup()
        
        outer_square = Square(side_length=4).set_stroke(BLUE_D)

        inner_major_axes = VGroup(
            Line(LEFT * 2, RIGHT * 2), 
            Line(DOWN * 2, UP * 2)
        ).set_color(WHITE).set_stroke(width=3)

        faded_axes = VGroup(
            Line(LEFT * 2 + UP, RIGHT * 2 + UP), 
            Line(LEFT * 2 + DOWN, RIGHT * 2 + DOWN),
            Line(LEFT * 1 + UP * 2, LEFT * 1 + DOWN * 2), 
            Line(RIGHT * 1 + UP * 2, RIGHT * 1 + DOWN * 2),
        ).set_stroke(color=BLUE_D, width=1, opacity=0.5)

        subset_complex_plane.add(inner_major_axes)
        subset_complex_plane.add(faded_axes)

        subset_complex_plane.scale(0.5)

        return subset_complex_plane

    def get_larger_plane_subset(self):
        subset_complex_plane = VGroup()
        
        outer_square = Square(side_length=4).set_stroke(BLUE_D)

        inner_major_axes = VGroup(
            Line(LEFT * 4, RIGHT * 4), 
            Line(DOWN * 4, UP * 4)
        ).set_color(WHITE).set_stroke(width=3)

        faded_axes = VGroup(
            Line(LEFT * 4 + UP, RIGHT * 4 + UP), 
            Line(LEFT * 4 + DOWN, RIGHT * 4 + DOWN),
            Line(LEFT * 4 + UP * 3, RIGHT * 4 + UP * 3), 
            Line(LEFT * 4 + DOWN * 3, RIGHT * 4 + DOWN * 3),
            Line(LEFT * 1 + UP * 4, LEFT * 1 + DOWN * 4), 
            Line(RIGHT * 1 + UP * 4, RIGHT * 1 + DOWN * 4),
            Line(LEFT * 3 + UP * 4, LEFT * 3 + DOWN * 4), 
            Line(RIGHT * 3 + UP * 4, RIGHT * 3 + DOWN * 4),
        ).set_stroke(color=BLUE_D, width=1, opacity=0.5)

        interval_axes = VGroup(
            Line(LEFT * 4 + UP * 2, RIGHT * 4 + UP * 2),
            Line(LEFT * 4 + DOWN * 2, RIGHT * 4 + DOWN * 2), 
            Line(LEFT * 2 + DOWN * 4, LEFT * 2 + UP * 4),
            Line(RIGHT * 2 + DOWN * 4, RIGHT * 2 + UP * 4)
        ).set_stroke(color=BLUE_D, width=2, opacity=1)
        
        subset_complex_plane.add(inner_major_axes)
        subset_complex_plane.add(interval_axes)
        subset_complex_plane.add(faded_axes)

        subset_complex_plane.scale(0.5)

        return subset_complex_plane

class TriangleInMinkowski(GJKUtils):
    CONFIG = {
        "p1_color": YELLOW,
        "p2_color": PERIWINKLE_BLUE,
        "diff_color": FUCHSIA,
    }
    def construct(self):
        plane = NumberPlane()
        self.add(plane)
        p1 = Polygon(
            *[RIGHT * 1 + UP * 1, RIGHT * 4 + UP * 2, RIGHT * 3 + UP * 3, RIGHT * 1 + UP * 3]
        ).set_color(self.p1_color)

        p2 = Polygon(
            *[
            RIGHT * 4, RIGHT * 3 + UP * 1, RIGHT * 3 + UP * 2,
            RIGHT * 5 + UP * 3, RIGHT * 6 + UP * 2, 
            ]
        ).set_color(self.p2_color)

        self.play(
            Write(p1),
            Write(p2),
        )

        self.wait()

        a_1, b_1 = p1.get_vertices()[3], p2.get_vertices()[0]

        first_objs = self.show_points(a_1, b_1, ["A_1", "B_1"], [UP, DOWN, UP])

        a_2, b_2 = p1.get_vertices()[0], p2.get_vertices()[3]

        second_objs = self.show_points(a_2, b_2, ["A_2", "B_2"], [DR * 0.5, UP, DOWN])

        a_3, b_3 = p1.get_vertices()[1], p2.get_vertices()[2]
        
        third_objs = self.show_points(a_3, b_3, ["A_3", "B_3"], [DOWN, UL * 0.5, DOWN])

        enough_info = TextMobject("What do these points tell us?").scale(1)
        enough_info.move_to(DOWN * 3.5).add_background_rectangle()
        
        self.play(
            Write(enough_info)
        )
        self.wait()

        triangle = Polygon(a_1 - b_1, a_2 - b_2, a_3 - b_3)
        triangle.set_color(self.diff_color)
        triangle.set_fill(color=self.diff_color, opacity=0.5)
        self.play(
            DrawBorderThenFill(triangle),
            third_objs[5].shift, RIGHT * 0.45,
            run_time=2,
        )

        self.wait()

        origin_dot = Dot().set_color(LIME_GREEN)
        self.play(
            GrowFromCenter(origin_dot),
            Flash(origin_dot, color=LIME_GREEN),
        )

        self.wait()

        sides = TextMobject("Sides of triangle are inside Minkowski difference (convexity)").scale(0.8)
        sides.move_to(enough_info.get_center())
        sides.add_background_rectangle()

        self.play(
            ReplacementTransform(enough_info, sides)
        )
        self.wait()

        conclusion = TextMobject("Triangle contains origin ", r"$\Rightarrow$ Minkowski difference contains origin").scale(0.8)
        conclusion.add_background_rectangle()
        conclusion.move_to(sides.get_center())

        self.play(
            FadeOut(sides)
        )

        self.wait()

        self.play(
            Write(conclusion),
            run_time=2
        )
        self.wait()

        self.play(
            FadeOut(conclusion)
        )
        new_question = TextMobject(r"Triangle from points on $A \ominus B$ contains origin $\Rightarrow A$ and $B$ intersect").scale(0.8)
        new_question[0][20].set_color(YELLOW)
        new_question[0][22].set_color(PERIWINKLE_BLUE)
        new_question[0][-14].set_color(YELLOW)
        new_question[0][-10].set_color(PERIWINKLE_BLUE)
        new_question.add_background_rectangle()
        new_question.move_to(conclusion.get_center())
        self.play(
            Write(new_question),
            run_time=2
        )
        self.wait()


    def show_points(self, a, b, labels, directions):
        point_a = self.point_to_text(labels[0], a).set_color(self.p1_color).add_background_rectangle()
        point_b = self.point_to_text(labels[1], b).set_color(self.p2_color).add_background_rectangle()

        dot_a = Dot().move_to(a).set_color(self.p1_color)
        dot_b = Dot().move_to(b).set_color(self.p2_color)

        point_a.next_to(dot_a, directions[0])
        point_b.next_to(dot_b, directions[1])

        self.play(
            GrowFromCenter(dot_a),
            GrowFromCenter(dot_b),
            Write(point_a),
            Write(point_b),
        )

        self.wait()

        point_c = self.point_to_text(labels[0] + "-" + labels[1], a - b)
        point_c.add_background_rectangle()
        dot_c = Dot().move_to(a - b)
        point_c.next_to(dot_c, directions[2])
        self.add_foreground_mobject(dot_c)
        self.add_foreground_mobject(point_c)
        self.play(
            GrowFromCenter(dot_c),
            Write(point_c)
        )

        self.wait()

        return VGroup(dot_a, point_a, dot_b, point_b, dot_c, point_c)

    def point_to_text(self, label, point):
        return TexMobject("{0} = ({1}, {2})".format(label, int(point[0]), int(point[1]))).scale(0.7)

class SimplexIntro(GJKUtils):
    def construct(self):
        left_plane = self.get_larger_plane_subset()
        left_plane.move_to(LEFT * 3.5)
        right_plane = self.get_larger_plane_subset()
        right_plane.move_to(RIGHT * 3.5)

        self.add(left_plane, right_plane)

        self.show_original_shapes(left_plane, right_plane)

    def show_original_shapes(self, left_plane, right_plane):
        s1 = RegularPolygon(n=4).scale(1)
        s1.set_stroke(color=YELLOW, width=5)
        s2 = RegularPolygon(n=3).scale(1)
        s2.set_stroke(color=PERIWINKLE_BLUE, width=5)

        s1.move_to(LEFT * 3.5)
        s2.move_to(LEFT * 2.5)

        question = TextMobject(r"Do shapes $A$ and $B$ intersect?").scale(0.8)
        question[0][8].set_color(YELLOW)
        question[0][12].set_color(PERIWINKLE_BLUE)
        question.next_to(left_plane, DOWN)
        self.play(
            ShowCreation(s1),
            ShowCreation(s2)
        )

        self.wait()

        self.play(
            Write(question)
        )

        self.wait()

        new_question = TextMobject(
            "Can we build a triangle out of points" + "\\\\",
            r"on $A \ominus B$ that surround the origin?"
        ).scale(0.8)
        new_question[0][11:19].set_color(GREEN_SCREEN)

        new_question[1][2].set_color(YELLOW)
        new_question[1][4].set_color(PERIWINKLE_BLUE)

        origin = Dot().move_to(right_plane.get_center()).set_color(YELLOW)

        new_question.next_to(right_plane, DOWN)

        diff = self.get_minkowski_diff(s1, s2)
        poly_diff = Polygon(*diff).set_stroke(color=FUCHSIA, width=5)

        poly_diff.move_to(RIGHT * 3.5 + (s1.get_center() - s2.get_center()))

        self.play(
            ShowCreation(poly_diff)
        )
        self.wait()

        self.play(
            Write(new_question)
        )
        self.wait()

        triangle_points = [1, 3, 5]

        points = [Dot().move_to(poly_diff.get_vertices()[j]) for j in triangle_points]

        triangle = Polygon(*[poly_diff.get_vertices()[i] for i in triangle_points])
        triangle.set_stroke(color=GREEN_SCREEN, width=7)


        self.play(
            ShowCreation(triangle),
            *[GrowFromCenter(d) for d in points]
        )

        self.wait()

        self.play(
            Flash(origin),
            GrowFromCenter(origin),
        )
        self.wait()

        new_question_transformed = TextMobject(
            "Can we build a simplex out of points" + "\\\\",
            r"on $A \ominus B$ that surround the origin?"
        ).scale(0.8)
        new_question_transformed

        new_question_transformed[0][11:18].set_color(GREEN_SCREEN)
        new_question_transformed[1][2].set_color(YELLOW)
        new_question_transformed[1][4].set_color(PERIWINKLE_BLUE)

        new_question_transformed.next_to(right_plane, DOWN)

        self.play(
            ReplacementTransform(new_question, new_question_transformed)
        )
        self.wait()


    def get_larger_plane_subset(self):
        subset_complex_plane = VGroup()
        
        outer_square = Square(side_length=4).set_stroke(BLUE_D)

        inner_major_axes = VGroup(
            Line(LEFT * 4, RIGHT * 4), 
            Line(DOWN * 4, UP * 4)
        ).set_color(WHITE).set_stroke(width=3)

        faded_axes = VGroup(
            Line(LEFT * 4 + UP, RIGHT * 4 + UP), 
            Line(LEFT * 4 + DOWN, RIGHT * 4 + DOWN),
            Line(LEFT * 4 + UP * 3, RIGHT * 4 + UP * 3), 
            Line(LEFT * 4 + DOWN * 3, RIGHT * 4 + DOWN * 3),
            Line(LEFT * 1 + UP * 4, LEFT * 1 + DOWN * 4), 
            Line(RIGHT * 1 + UP * 4, RIGHT * 1 + DOWN * 4),
            Line(LEFT * 3 + UP * 4, LEFT * 3 + DOWN * 4), 
            Line(RIGHT * 3 + UP * 4, RIGHT * 3 + DOWN * 4),
        ).set_stroke(color=BLUE_D, width=1, opacity=0.5)

        interval_axes = VGroup(
            Line(LEFT * 4 + UP * 2, RIGHT * 4 + UP * 2),
            Line(LEFT * 4 + DOWN * 2, RIGHT * 4 + DOWN * 2), 
            Line(LEFT * 2 + DOWN * 4, LEFT * 2 + UP * 4),
            Line(RIGHT * 2 + DOWN * 4, RIGHT * 2 + UP * 4)
        ).set_stroke(color=BLUE_D, width=2, opacity=1)
        
        subset_complex_plane.add(inner_major_axes)
        subset_complex_plane.add(interval_axes)
        subset_complex_plane.add(faded_axes)

        subset_complex_plane.scale(0.7)

        return subset_complex_plane

class SimplexDef(GJKUtils):
    def construct(self):
        title = TextMobject("Simplex").move_to(UP * 3.5)
        h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
        h_line.next_to(title, DOWN)
        self.play(
            Write(title),
            ShowCreation(h_line)
        )

        self.wait()

        definition = TextMobject(r"$k$-Simplex - shape that is guaranteed to enclose a point" + "\\\\", 
            r"in $k$-dimensional space"
        ).scale(0.8)

        definition.next_to(h_line, DOWN)
        definition[1].shift(LEFT * 0.7)
        self.play(
            Write(definition),
            run_time=2,
        )

        self.wait()

        note = TextMobject(
            "*This is not entirely accurate, more precise and rigorous" + "\\\\", 
            "definitions exist, but good enough for our use case."
        ).scale(0.6)
        note.next_to(definition, DOWN)

        self.add(note)
        self.wait()

        simplex_2 = TextMobject("2-simplex")
        simplex_3 = TextMobject("3-simplex")


        simplex_2d = Polygon(LEFT * 2, UP, DOWN * 0.5 + RIGHT)
        simplex_2d.set_stroke(color=GREEN_SCREEN, width=5)
        simplex_2d.set_fill(color=GREEN_SCREEN, opacity=0.5)
        simplex_2d.scale(1.5)

        simplex_2d.move_to(LEFT * 3.5 + DOWN)
        self.play(
            DrawBorderThenFill(simplex_2d)
        )

        self.wait()

        simplex_2.next_to(simplex_2d, DOWN)

        self.play(
            Write(simplex_2)
        )
        self.wait()

        tetrahedron_points = [LEFT * 2, LEFT * 1 + DOWN * 1.5, RIGHT * 2.5 + DOWN * 0.5, UP * 2]

        triangle_face_1 = Polygon(tetrahedron_points[0], tetrahedron_points[1], tetrahedron_points[3] + DL * SMALL_BUFF * 0.2)
        triangle_face_2 = Polygon(tetrahedron_points[1], tetrahedron_points[2], tetrahedron_points[3])
        triangle_face_1.set_stroke(color=VIOLET, width=5)
        triangle_face_1.set_fill(color=VIOLET, opacity=0.5)
        triangle_face_2.set_stroke(color=VIOLET, width=5)
        triangle_face_2.set_fill(color=LIGHT_VIOLET, opacity=0.5)

        tetrahedron = VGroup(triangle_face_1, triangle_face_2)
        tetrahedron.move_to(RIGHT * 3.5 + DOWN)
        self.play(
            FadeIn(tetrahedron)
        )
        
        self.wait()

        simplex_3.next_to(triangle_face_2, DOWN)

        self.play(
            Write(simplex_3)
        )

        self.wait()

class TriangleRandomGenerator(GJKUtils):
    CONFIG = {
        "c_color": YELLOW,
        "p_color": PERIWINKLE_BLUE,
        "d_color": FUCHSIA,
    }
    def construct(self):
        plane = NumberPlane()
        self.add(plane)


        circle = Circle(radius=1).set_color(self.c_color)
        polygon = RegularPolygon(n=4).scale(1).set_color(self.p_color)
        circle.move_to(RIGHT * 1.5 + UP * 2)
        polygon.move_to(RIGHT * 3 + UP * 2)
        shapes = VGroup(circle, polygon)

        self.play(
            ShowCreation(shapes)
        )

        self.wait()
        
        rounded_rectangle = RoundedRectangle(height=2*np.sqrt(2), width=2*np.sqrt(2), corner_radius=1).scale(1.25)
        rounded_rectangle.rotate(PI / 4)
        rounded_rectangle.set_stroke(color=self.d_color, width=7)
        rounded_rectangle.move_to(LEFT * 1.5)
        self.play(
            ShowCreation(rounded_rectangle)
        )
        self.wait()

        A_points = self.get_random_triangle_points(circle)
        B_points = self.get_random_triangle_points(polygon)

        C_points = [A - B for A, B in zip(A_points, B_points)]

        A_dots = [Dot().move_to(a).set_color(self.c_color) for a in A_points]
        B_dots = [Dot().move_to(b).set_color(self.p_color) for b in B_points]

        C_dots = [Dot().move_to(c) for c in C_points]
        triangle = Polygon(*C_points)
        triangle.set_stroke(color=WHITE, width=5)

        for dot in A_dots + B_dots:
            self.add(dot)

        self.play(
            ShowCreation(triangle),
            *[GrowFromCenter(c) for c in C_dots],
        )
        self.wait()
        for mob in A_dots + B_dots + C_dots + [triangle]:
            self.remove(mob)

        for _ in range(65):
            A_points = self.get_random_triangle_points(circle)
            B_points = self.get_random_triangle_points(polygon)

            C_points = [A - B for A, B in zip(A_points, B_points)]

            A_dots = [Dot().move_to(a).set_color(self.c_color) for a in A_points]
            B_dots = [Dot().move_to(b).set_color(self.p_color) for b in B_points]

            C_dots = [Dot().move_to(c) for c in C_points]
            triangle = Polygon(*C_points)
            if self.check_point_inside(ORIGIN, triangle):
                triangle.set_stroke(color=GREEN_SCREEN, width=5)
            else:
                triangle.set_stroke(color=WHITE, width=5)

            group = VGroup(*A_dots + B_dots + C_dots + [triangle])
            self.add(group)
            self.wait()
            self.remove(group)

    def get_random_point_on_shape(self, shape):
        return shape.point_from_proportion(np.random.uniform())

    def get_random_triangle_points(self, shape):
        return [self.get_random_point_on_shape(shape) for _ in range(3)]

class SupportFunctions(GJKUtils):
    def construct(self):
        convex = RegularPolygon(n=6).scale(2)
        convex.set_color(BLUE)
        self.play(
            ShowCreation(convex)
        )
        self.wait()
        arrow, dot = self.show_support(convex)

        self.play(
            convex.shift, LEFT * 3,
            arrow.shift, LEFT * 3,
            dot.shift, LEFT * 3,
        )
        always_rotate(arrow, rate=60*DEGREES, about_point=arrow.get_start())
        always_redraw(lambda: dot.move_to(self.get_support(convex, arrow.get_end() - arrow.get_start())))
        self.add(arrow)
        self.add(dot)
        self.wait(15)
        arrow.clear_updaters()
        dot.clear_updaters()
        new_shape = Circle().scale(2).set_color(BLUE).shift(LEFT * 3)
        self.play(
            Transform(convex, new_shape)
        )
        always_rotate(arrow, rate=60*DEGREES, about_point=arrow.get_start())
        always_rotate(dot, rate=60*DEGREES, about_point=arrow.get_start())
        self.wait(12)

    def show_support(self, polygon):
        directions = [d / np.linalg.norm(d) for d in polygon.get_vertices()]
        dot = Dot().set_color(YELLOW)
        dot.move_to(polygon.get_vertices()[0])
        arrow = Arrow().set_color(YELLOW)
        arrow.put_start_and_end_on(
                polygon.get_center(), 
                polygon.get_center() + directions[0]
        )
        self.play(
            GrowFromCenter(dot)
        )
        self.play(
            ShowCreation(arrow)
        )
        self.wait()
        vertices = polygon.get_vertices()
        for i, d in enumerate(directions):
            dot.move_to(vertices[i])
            arrow.put_start_and_end_on(
                polygon.get_center(), 
                polygon.get_center() + d
            )
            self.wait()

        arrow.put_start_and_end_on(
                polygon.get_center(), 
                polygon.get_center() + directions[0]
        )
        dot.move_to(polygon.get_vertices()[0])
        self.wait()
        return arrow, dot

class SupportFunctionConcave(GJKUtils):
    def construct(self):
        convex = RegularPolygon(n=6).scale(2)
        points = convex.get_vertices()
        concave_points = [p + RIGHT * 3 for p in points]
        concave_points[0] += LEFT * 2

        concave = Polygon(*concave_points)
        concave.set_color(VIOLET)
        self.play(
            ShowCreation(concave)
        )
        self.wait()

        problem_point = Dot().move_to(concave_points[0])
        problem_point.set_color(CROSS_RED)
        problem_point_label = TexMobject("A")
        problem_point_label.next_to(problem_point, RIGHT * 2)
        self.play(
            GrowFromCenter(problem_point),
            Write(problem_point_label),
        )
        self.wait()

        arrow = Arrow()
        arrow.put_start_and_end_on(
            concave.get_center(), 
            concave.get_center() + LEFT,
        ).set_color(YELLOW)

        dot = Dot().set_color(YELLOW)
        func = lambda: dot.move_to(self.get_support(concave, arrow.get_end() - arrow.get_start()))
        dot.move_to(func())

        self.play(
            GrowFromCenter(dot),
            Write(arrow)
        )
        self.wait()

        always_rotate(arrow, rate=60*DEGREES, about_point=arrow.get_start())
        always_redraw(func)
        self.add(arrow, dot)
        self.wait()

        counterexample = TextMobject(
            r"No matter what direction you pick, point $A$" + "\\\\",
            "will never be furthest in that direction."
        ).scale(0.8)

        counterexample.move_to(UP * 3)
        self.play(
            Write(counterexample),
            run_time=2,
        )
        self.wait(13)

class SupportMinkowskiSum(GJKUtils):
    CONFIG = {
        "p1_color": YELLOW,
        "p2_color": PERIWINKLE_BLUE,
        "diff_color": FUCHSIA,
    }
    def construct(self):
        A = RegularPolygon(n=5).set_color(self.p1_color).scale(1.5)
        B = RegularPolygon(n=6).set_color(self.p2_color).scale(1.5)
        C_points = self.get_minkowski_sum(A, B)
        C = Polygon(*C_points).set_color(self.diff_color)

        polygons = VGroup(A, B, C)
        polygons.arrange_submobjects(RIGHT, buff=0.8)
        polygons.shift(UP * 1)

        a_label = TexMobject("A")
        a_label.next_to(A, DOWN)

        b_label = TexMobject("B")
        b_label.next_to(B, DOWN)

        c_label = TexMobject(r"C = A", r"\oplus", "B")
        c_label[1].shift(UP * SMALL_BUFF * 0.5)
        c_label.next_to(C, DOWN)

        center_A = A.copy().scale(1.5).move_to(UP)
        center_A_label = a_label.copy().next_to(center_A, DOWN)
        center_arrow = Arrow(ORIGIN, RIGHT * 1.5).shift(UP)
        center_d_label = TexMobject(r"\vec{d}").scale(0.7).next_to(center_arrow, UP, buff=SMALL_BUFF)
        center_dot = Dot().set_color(center_A.get_color())
        center_dot.move_to(self.get_support(center_A, RIGHT))


        self.play(
            Write(center_A),
            Write(center_A_label),
        )
        self.wait()

        self.play(
            Write(center_arrow),
            Write(center_d_label),
        )
        self.wait()

        self.play(
            GrowFromCenter(center_dot)
        )

        self.wait()

        s_a_center = TexMobject(r"s_A(\vec{d}) \rightarrow").scale(0.8)
        dot_a_center = Dot().set_color(self.p1_color).next_to(s_a_center, RIGHT, buff=SMALL_BUFF * 2).shift(DOWN * SMALL_BUFF / 2)
        s_a_def_center = VGroup(s_a_center, dot_a_center)
        s_a_def_center.next_to(center_A_label, DOWN)

        self.play(
            Write(s_a_def_center),
        )
        self.wait()

        support_function_center = TextMobject("Support Function")
        brace_center = Brace(s_a_def_center, direction=DOWN)
        support_function_center.scale(0.8).next_to(brace_center, DOWN)
        self.play(
            Write(brace_center)
        )

        self.play(
            Write(support_function_center)
        )

        self.wait()

        s_a = TexMobject(r"s_A(\vec{d}) \rightarrow").scale(0.8)
        dot_a = Dot().set_color(self.p1_color).next_to(s_a, RIGHT, buff=SMALL_BUFF * 2).shift(DOWN * SMALL_BUFF / 2)
        s_a_def = VGroup(s_a, dot_a)
        s_a_def.next_to(a_label, DOWN)

        support_function = TextMobject("Support Function")
        brace = Brace(s_a_def, direction=DOWN)
        support_function.scale(0.8).next_to(brace, DOWN)


        labels = VGroup(a_label, b_label, c_label)

        arrows, dots, d_labels = self.show_supports(polygons, labels)

        self.play(
            ReplacementTransform(center_A, A),
            ReplacementTransform(center_A_label, labels[0]),
            ReplacementTransform(center_arrow, arrows[0]),
            ReplacementTransform(center_d_label, d_labels[0]),
            ReplacementTransform(center_dot, dots[0]),
            ReplacementTransform(s_a_def_center, s_a_def),
            ReplacementTransform(brace_center, brace),
            ReplacementTransform(support_function_center, support_function),
            run_time=2
        )
        self.wait()

        self.play(
            Write(B),
            Write(b_label)
        )

        self.wait()

        self.play(
            Write(C),
            Write(c_label)
        )

        self.wait()

        s_b = TexMobject(r"s_B(\vec{d}) \rightarrow").scale(0.8)
        dot_b = Dot().set_color(self.p2_color).next_to(s_b, RIGHT, buff=SMALL_BUFF * 2).shift(DOWN * SMALL_BUFF / 2)
        s_b_def = VGroup(s_b, dot_b)
        s_b_def.next_to(b_label, DOWN)


        s_c = TexMobject(r"s_C(\vec{d}) = s_A(\vec{d}) + s_B(\vec{d}) \rightarrow").scale(0.8)
        dot_c = Dot().set_color(self.diff_color).next_to(s_c, RIGHT, buff=SMALL_BUFF * 2).shift(DOWN * SMALL_BUFF / 2)
        s_c_def = VGroup(s_c, dot_c)
        s_c_def.next_to(c_label, DOWN)

        support_labels = [s_b_def, s_c_def]
        

        for arrow, dot, label, s_label in zip(arrows[1:], dots[1:], d_labels[1:], support_labels):
            self.play(
                Write(arrow),
                Write(label),
            )
            self.wait()
            self.play(
                GrowFromCenter(dot)
            )
            self.wait()

            self.play(
                Write(s_label)
            )
            self.wait()

        self.rotate_group(polygons, arrows, dots, d_labels)

    def show_supports(self, polygons, labels):
        direction = RIGHT
        arrows = [Arrow(p.get_center(), p.get_center() + direction) for p in polygons]
        dots = [Dot().set_color(p.get_color()) for p in polygons]
        d_labels = [TexMobject(r"\vec{d}").scale(0.7).next_to(a, UP, buff=SMALL_BUFF) for a in arrows]
        for dot, p in zip(dots, polygons):
            dot.move_to(self.get_support(p, direction))

        return arrows, dots, d_labels

    def rotate_group(self, polygons, arrows, dots, d_labels):
        func1 = lambda: dots[0].move_to(self.get_support(polygons[0], arrows[0].get_end() - arrows[0].get_start()))
        func2 = lambda: dots[1].move_to(self.get_support(polygons[1], arrows[1].get_end() - arrows[1].get_start()))
        func3 = lambda: dots[2].move_to(self.get_support(polygons[2], arrows[2].get_end() - arrows[2].get_start()))
        
        functions = [func1, func2, func3]
        i = 0   
        for p, arrow, dot, d_label, in zip(polygons, arrows, dots, d_labels):
            always_rotate(arrow, rate=60*DEGREES, about_point=arrow.get_start())
            always_rotate(d_label, rate=60*DEGREES, about_point=arrow.get_start())
            # normal = self.get_normal_towards_origin(arrow.get_start(), arrow.get_end(), away=True)
            # always(d_label.next_to, arrow, normal, SMALL_BUFF)
            always_redraw(functions[i])
            self.add(arrow, dot, d_label)
            i += 1

        self.wait(6 + 1 / self.camera.frame_rate)

class SupportMinkowskiDiff(GJKUtils):
    CONFIG = {
        "p1_color": YELLOW,
        "p2_color": PERIWINKLE_BLUE,
        "diff_color": FUCHSIA,
        "highlight_c": GREEN_SCREEN,
        "width": 5,
    }
    def construct(self):
        plane = NumberPlane()
        self.add(plane)
        self.wait()

        points1 = [UP * 0.5 + RIGHT * 0, UP * 2.5 + RIGHT * 1.5, UP * 0.5 + RIGHT * 3.5]
        points2 = [
        UP * 2 + RIGHT * 3, UP * 3.5 + RIGHT * 3.5, 
        UP * 3.5 + RIGHT * 6.5, UP * 1 + RIGHT * 6.5,
        ]

        A, B = self.get_polygons(points1, points2)
        a_label = TexMobject("A").add_background_rectangle()
        a_label.next_to(A, DOWN)
        b_label = TexMobject("B").add_background_rectangle()
        b_label.next_to(B, DOWN).shift(UP * SMALL_BUFF * 5)

        self.play(
            Write(A),
            Write(a_label),
            Write(B),
            Write(b_label),
        )

        C_points = self.get_minkowski_diff(A, B)
        C = Polygon(*C_points)
        C.set_stroke(color=self.diff_color, width=self.width)
        c_label = TexMobject(r"C = A \ominus B").add_background_rectangle()
        c_label.next_to(C, DOWN)
        self.play(
            Write(C),
            Write(c_label),
        )
        self.wait()

        polygons = VGroup(A, B, C)
        arrows, dots = self.show_few_directions(A, B, C)
        self.wait()

        support_relation = TexMobject(r"s_C(\vec{d}) = s_A(\vec{d}) - s_B(-\vec{d})")
        support_relation.add_background_rectangle()
        support_relation.move_to(RIGHT * 4 + DOWN * 2)
        self.play(
            Write(support_relation)
        )
        self.wait()

        surround_rect = SurroundingRectangle(support_relation, buff=SMALL_BUFF)
        self.play(
            ShowCreation(surround_rect)
        )

        self.wait()

        polygons = VGroup(A, B, C)

        self.rotate_group(polygons, arrows, dots)

    def show_few_directions(self, A, B, C):
        directions = [RIGHT, LEFT * 0.5 + UP, LEFT * 0.5 + DOWN, RIGHT]
        directions = [d / np.linalg.norm(d) for d in directions]
        a_arrow = Arrow().put_start_and_end_on(A.get_center(), A.get_center() + directions[0])
        b_arrow = Arrow().put_start_and_end_on(B.get_center(), B.get_center() - directions[0])
        c_arrow = Arrow().put_start_and_end_on(C.get_center(), C.get_center() + directions[0])
        

        a_dot = Dot().move_to(self.get_support(A, directions[0]))
        b_dot = Dot().move_to(self.get_support(B, -directions[0]))
        c_dot = Dot().move_to(self.get_support(C, directions[0]))
        arrows, dots = [a_arrow, b_arrow, c_arrow], [a_dot, b_dot, c_dot]
        i = 0
        for arrow, dot in zip(arrows, dots):
            if i == 2:
                self.play(
                    Write(arrows[i])
                )

                self.play(
                    TransformFromCopy(
                        VGroup(dots[0], dots[1]),
                        dots[i]
                    ),
                    run_time=2
                )
                self.wait()
                break

            self.play(
                Write(arrow)
            )
            self.play(
                GrowFromCenter(dot)
            )
            self.wait()
            i += 1

        self.wait()
        for i in range(1, len(directions)):
            a_arrow.put_start_and_end_on(A.get_center(), A.get_center() + directions[i])
            b_arrow.put_start_and_end_on(B.get_center(), B.get_center() - directions[i])
            c_arrow.put_start_and_end_on(C.get_center(), C.get_center() + directions[i])
            
            a_dot.move_to(self.get_support(A, directions[i]))
            b_dot.move_to(self.get_support(B, -directions[i]))
            c_dot.move_to(self.get_support(C, directions[i]))

            self.wait(2)

        return arrows, dots

    def rotate_group(self, polygons, arrows, dots):
        func1 = lambda: dots[0].move_to(self.get_support(polygons[0], arrows[0].get_end() - arrows[0].get_start()))
        func2 = lambda: dots[1].move_to(self.get_support(polygons[1], arrows[1].get_end() - arrows[1].get_start()))
        func3 = lambda: dots[2].move_to(self.get_support(polygons[2], arrows[2].get_end() - arrows[2].get_start()))
        
        functions = [func1, func2, func3]
        i = 0   
        for p, arrow, dot in zip(polygons, arrows, dots):
            always_rotate(arrow, rate=30*DEGREES, about_point=arrow.get_start())
            always_redraw(functions[i])
            self.add(arrow, dot)
            i += 1

        self.wait(16)

    def get_polygons(self, points1, points2):
        p1 = Polygon(*points1).set_stroke(color=self.p1_color, width=self.width)
        p2 = Polygon(*points2).set_stroke(color=self.p2_color, width=self.width)
        return p1, p2

class CalculateSupport(GJKUtils):
    CONFIG = {
        "plane_scale" : 1,
        "plane_shift" : DOWN * 1.7,
        "d_color": YELLOW,
        "v_color": FUCHSIA,
    }


    def construct(self):
        title = TextMobject("Computing Support Functions").scale(1.2)
        title.move_to(UP * 3.5)
        h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
        h_line.next_to(title, DOWN)
        self.play(
            Write(title),
            ShowCreation(h_line)
        )

        self.wait()

        s = TexMobject(r"s_B(\vec{d}) = v} ", r"= \underset{v \in B}{\arg\max} \enspace v^T \vec{d}")
        definition = TextMobject(
            r"A support function $s_B$ takes a direction $\vec{d}$ and returns" + "\\\\", 
            r"point $v$ on the boundary of shape $B$ ``furthest'' in direction $\vec{d}$."
        ).scale(0.8)
        definition.next_to(h_line, DOWN)
        definition[0][1:16].set_color(YELLOW)
        s.next_to(definition, DOWN)

        self.play(
            Write(definition),
            run_time=3,
        )

        self.wait()

        self.play(
            Write(s)
        )
        self.wait()

        self.play(
            s.shift, LEFT * 2,
        )

        brace = Brace(s, direction=RIGHT)
        notation_exp = TextMobject(r"Point $v$ that maximizes" + "\\\\", r"dot product with $\vec{d}$").scale(0.7)
        notation_exp.next_to(brace, RIGHT)

        self.play(
            GrowFromCenter(brace),
            Write(notation_exp),
        )

        self.wait()


        plane = self.get_larger_plane_subset()
        plane.move_to(self.plane_shift)

        self.play(
            Write(plane)
        )

        self.wait()

        vec_d = Vector(RIGHT * np.cos(2 * PI / 3) + UP * np.sin(2 * PI / 3))
        vec_d = self.mobject_to_plane(vec_d).set_color(YELLOW)

        d = TexMobject(r"\vec{d}").scale(0.7)
        d.next_to(vec_d, LEFT, buff=SMALL_BUFF)
        d.add_background_rectangle()
        d.shift(DOWN * SMALL_BUFF + RIGHT * SMALL_BUFF * 2)

        self.play(
            Write(vec_d),
            Write(d),
        )

        self.wait()

        vec_v = Vector(RIGHT * np.cos(PI / 3) + UP * np.sin(-2 * PI / 3))
        vec_v = self.mobject_to_plane(vec_v).set_color(self.v_color)
        v_dot = Dot().scale(0.7)
        v_dot.move_to(vec_v.get_end())
        v = TexMobject("v").scale(0.7)
        v.add_background_rectangle()
        v.move_to(v_dot.get_center())
        v.shift((vec_v.get_end() - vec_v.get_start()) * SMALL_BUFF * 3)
        self.play(
            GrowFromCenter(v_dot),
            Write(v),
        )
        self.wait()

        self.play(
            Write(vec_v)
        )
        self.wait()

        dot_product = VGroup(
            TexMobject(r"v^T \vec{d} = "),
            DecimalNumber(
                -1,
                num_decimal_places=2
            )
        )
        dot_product.arrange(RIGHT)

        dot_product.scale(0.8)
        dot_product.next_to(plane, LEFT)

        text, number = dot_product
        number.shift(DOWN * SMALL_BUFF * 0.5)

        self.play(
            Write(dot_product)
        )

        self.wait()

        always_rotate(vec_v, rate=60*DEGREES, about_point=vec_d.get_start())
        always_rotate(v_dot, rate=60*DEGREES, about_point=vec_d.get_start())
        v.move_to(v_dot.get_center())
        v.shift((vec_v.get_end() - vec_v.get_start()) * SMALL_BUFF * 3)
        always_redraw(lambda: v.move_to(v_dot.get_center()).shift((vec_v.get_end() - vec_v.get_start()) * SMALL_BUFF * 3))
        number.add_updater(lambda m: m.set_value(np.dot(vec_d.get_end() - vec_d.get_start(), vec_v.get_end() - vec_v.get_start())))
        self.add(vec_v, v_dot, number)
        self.wait(9)
        vec_v.clear_updaters()
        v.clear_updaters()
        v_dot.clear_updaters()

        self.wait()


        self.play(
            vec_v.put_start_and_end_on, plane.get_center(), (vec_v.get_end() - vec_v.get_start()) * 2 + vec_v.get_start(),
            v_dot.move_to, (vec_v.get_end() - vec_v.get_start()) * 2 + vec_v.get_start(),
            v.shift, vec_v.get_end() - vec_v.get_start(),
        )
        self.wait()
        
        poly_vertices = [
        LEFT * 1.5 + UP * 1, UP * 1.5, RIGHT * 1.5 + UP, 
        RIGHT * 1.5 + DOWN, DOWN * 1.5, LEFT * 1.5 + DOWN,
        ]

        polygon = Polygon(*poly_vertices)
        polygon.shift(self.plane_shift)
        polygon.set_color(self.v_color)
        dots = []
        labels = []
        for i, vertex in enumerate(polygon.get_vertices()):
            dot = Dot().move_to(vertex)
            dots.append(dot)
            label = TexMobject("v_{0}".format(i)).scale(0.7)
            label.add_background_rectangle()
            label.move_to(vertex)
            unit_v = (vertex - plane.get_center()) / np.linalg.norm(vertex - plane.get_center())
            label.shift(unit_v * SMALL_BUFF * 3)
            if i == 1:
                label.shift(RIGHT * SMALL_BUFF * 2)
            elif i == 4:
                label.shift(LEFT * SMALL_BUFF * 2)
            labels.append(label)

        self.play(
            FadeOut(vec_v),
            FadeOut(v_dot),
            FadeOut(v),
            FadeOut(dot_product),
        )

        self.wait()

        self.play(
            Write(polygon),
            *[GrowFromCenter(d) for d in dots],
            *[Write(v) for v in labels],    
        )
        self.wait()

        dot_products = VGroup()
        d_diff = vec_d.get_end() - vec_d.get_start()
        for i, v in enumerate(polygon.get_vertices()):
            v_diff = v - plane.get_center()
            result = round(np.dot(d_diff, v_diff), 2)
            dot_product = TexMobject("v_{0}^T".format(i), r"\vec{d} = ", "{0}".format(result))
            dot_product.scale(0.8)
            dot_products.add(dot_product)

        dot_products.arrange(DOWN)
        dot_products.next_to(plane, LEFT)

        self.play(
            Write(dot_products),
            run_time=2,
        )

        self.wait()

        surround_rect = SurroundingRectangle(dot_products[0])
        self.play(
            ShowCreation(surround_rect),
        )
        self.play(
            Indicate(dots[0]),
            Indicate(labels[0][1]),
        )

        self.play(
            dots[0].set_color, YELLOW,
            labels[0][1].set_color, YELLOW,
        )

        self.wait()

        circle = Circle(radius=1.5).set_color(self.v_color)
        circle.move_to(plane.get_center())

        self.play(
            ReplacementTransform(polygon, circle),
            *[FadeOut(d) for d in dots],
            *[FadeOut(l) for l in labels],
            FadeOut(dot_products),
            FadeOut(surround_rect),
            run_time=2
        )
        self.wait()

        center_dot = Dot().set_color(self.v_color)
        center_dot.move_to(circle.get_center())
        C = TexMobject("C").add_background_rectangle()
        C.scale(0.7).next_to(center_dot, DR * 0.5)

        radius = Line(circle.get_center(), circle.point_from_proportion(0.125))
        radius_label = TexMobject("r").add_background_rectangle().scale(0.7)
        radius_label.move_to(radius.get_center())
        radius_label.shift(DR * 0.1)
        self.play(
            ShowCreation(radius),
            Write(radius_label),
            GrowFromCenter(center_dot),
            Write(C),
        )

        self.wait()

        circle_support = TexMobject(r"s_B(\vec{d}) = C + r \vec{d}")
        circle_support.scale(0.9)
        circle_support.next_to(plane, LEFT).shift(LEFT * 0.6 + UP * 0.5)
        self.play(
            Write(circle_support)
        )

        self.wait()

        support_dot = Dot().move_to(circle.get_center() + circle.radius * (vec_d.get_end() - vec_d.get_start()))
        support_dot.set_color(YELLOW)
        support_label = TexMobject(r"s_B(\vec{d})").scale(0.7).add_background_rectangle()
        support_label.move_to(support_dot.get_center())
        support_label.shift((vec_d.get_end() - vec_d.get_start()) * 0.5)
        self.play(
            GrowFromCenter(support_dot),
            Write(support_label)
        )
        self.add_foreground_mobject(support_dot)

        self.wait()

        assumption = TextMobject(r"*Assumes $\vec{d}$ is a unit vector").scale(0.6)
        assumption.next_to(circle_support, DOWN)
        self.add(assumption)

        self.wait()

        ellipse = Ellipse(width=4, height=3).move_to(plane.get_center()).set_color(self.v_color)
        new_dot_location = np.array([
            ellipse.width / 2 * np.cos(2 * PI / 3),
            ellipse.height / 2 * np.sin(2 * PI / 3) + self.plane_shift[1],
            0,
        ])
        label_location = new_dot_location + ((vec_d.get_end() - vec_d.get_start()) * 0.5)

        self.play(
            ReplacementTransform(circle, ellipse),
            support_dot.move_to, new_dot_location,
            support_label.move_to, label_location,
            FadeOut(radius),
            FadeOut(radius_label),
            FadeOut(C),
            FadeOut(center_dot),
            FadeOut(support_label),
            FadeOut(circle_support),
            FadeOut(assumption),
        )
        self.wait()

        bonus_question = TextMobject(
            "Fun math challenge for you: define" + "\\\\",
            "a support function for an ellipse.",
        ).scale(0.8)

        bonus_question.move_to(LEFT * 3 + self.plane_shift)

        right_shift = RIGHT * 3

        self.play(
            plane.shift, right_shift,
            vec_d.shift, right_shift,
            ellipse.shift, right_shift,
            support_dot.shift, right_shift,
            d.shift, right_shift,
        )
        self.wait()
        self.plane_shift += right_shift

       
        self.play(
            Write(bonus_question),
            run_time=2
        )
        self.wait()

        always_rotate(vec_d, rate=60*DEGREES, about_point=vec_d.get_start())
        always_rotate(d, rate=60*DEGREES, about_point=vec_d.get_start())

        def get_new_point():
            vector = vec_d.get_end() - vec_d.get_start()
            angle = np.arctan2(vector[1], vector[0])
            new_dot_location = np.array([
            ellipse.width / 2 * np.cos(angle) + self.plane_shift[0],
            ellipse.height / 2 * np.sin(angle) + self.plane_shift[1],
            0,
            ])
            return new_dot_location


        f_always(support_dot.move_to, get_new_point)
        self.add(vec_d, d, support_dot)
        self.wait(16)
        
        crazy_fact = TextMobject(
            "Punchline: by defining support functions," + "\\\\"  
            "we can handle any convex shape!"
        ).scale(0.9)

        crazy_fact.move_to(definition.get_center()).shift(DOWN * SMALL_BUFF)

        self.play(
            ReplacementTransform(definition, crazy_fact),
            run_time=2
        )

        self.wait(10)

    def mobject_to_plane(self, mob):
        return mob.scale(self.plane_scale).shift(self.plane_shift)

    def get_plane_subset(self):
        subset_complex_plane = VGroup()
        
        outer_square = Square(side_length=4).set_stroke(BLUE_D)

        inner_major_axes = VGroup(
            Line(LEFT * 2, RIGHT * 2), 
            Line(DOWN * 2, UP * 2)
        ).set_color(WHITE).set_stroke(width=3)

        faded_axes = VGroup(
            Line(LEFT * 2 + UP, RIGHT * 2 + UP), 
            Line(LEFT * 2 + DOWN, RIGHT * 2 + DOWN),
            Line(LEFT * 1 + UP * 2, LEFT * 1 + DOWN * 2), 
            Line(RIGHT * 1 + UP * 2, RIGHT * 1 + DOWN * 2),
        ).set_stroke(color=BLUE_D, width=1, opacity=0.5)

        subset_complex_plane.add(inner_major_axes)
        subset_complex_plane.add(faded_axes)

        subset_complex_plane.scale(0.5)

        return subset_complex_plane

    def get_larger_plane_subset(self):
        subset_complex_plane = VGroup()
        
        outer_square = Square(side_length=4).set_stroke(BLUE_D)

        inner_major_axes = VGroup(
            Line(LEFT * 4, RIGHT * 4), 
            Line(DOWN * 4, UP * 4)
        ).set_color(WHITE).set_stroke(width=3)

        faded_axes = VGroup(
            Line(LEFT * 4 + UP, RIGHT * 4 + UP), 
            Line(LEFT * 4 + DOWN, RIGHT * 4 + DOWN),
            Line(LEFT * 4 + UP * 3, RIGHT * 4 + UP * 3), 
            Line(LEFT * 4 + DOWN * 3, RIGHT * 4 + DOWN * 3),
            Line(LEFT * 1 + UP * 4, LEFT * 1 + DOWN * 4), 
            Line(RIGHT * 1 + UP * 4, RIGHT * 1 + DOWN * 4),
            Line(LEFT * 3 + UP * 4, LEFT * 3 + DOWN * 4), 
            Line(RIGHT * 3 + UP * 4, RIGHT * 3 + DOWN * 4),
        ).set_stroke(color=BLUE_D, width=1, opacity=0.5)

        interval_axes = VGroup(
            Line(LEFT * 4 + UP * 2, RIGHT * 4 + UP * 2),
            Line(LEFT * 4 + DOWN * 2, RIGHT * 4 + DOWN * 2), 
            Line(LEFT * 2 + DOWN * 4, LEFT * 2 + UP * 4),
            Line(RIGHT * 2 + DOWN * 4, RIGHT * 2 + UP * 4)
        ).set_stroke(color=BLUE_D, width=2, opacity=1)
        
        subset_complex_plane.add(inner_major_axes)
        subset_complex_plane.add(interval_axes)
        subset_complex_plane.add(faded_axes)

        subset_complex_plane.scale(0.5)

        return subset_complex_plane

class GJKPreview(Scene):
    def construct(self):
        title = TextMobject("Core GJK Algorithm").scale(1.2)
        title.move_to(UP * 3.5)
        h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
        h_line.next_to(title, DOWN)
        self.play(
            Write(title),
            ShowCreation(h_line)
        )

        self.wait()

        two_perspectives = TextMobject("Two Presentations").next_to(h_line, DOWN)
        self.play(
            Write(two_perspectives)
        )

        self.wait()

        left_rect, right_rect = ScreenRectangle(height=3.5), ScreenRectangle(height=3.5)
        VGroup(left_rect, right_rect).arrange(RIGHT, buff=0.5)
        self.play(
            Write(left_rect),
            Write(right_rect),
        )

        self.wait()

        big_picture = TextMobject("Big picture view").next_to(left_rect, DOWN)

        detailed = TextMobject("Detailed view").next_to(right_rect, DOWN)

        self.play(
            Write(big_picture)
        )

        self.wait()

        self.play(
            Write(detailed)
        )

        self.wait()


class GJKBroad(GJKUtils):
    CONFIG = {
        "p1_color": YELLOW,
        "p2_color": FUCHSIA,
        "diff_color": ORANGE,
        "highlight_c": GREEN_SCREEN,
        "width": 5,
    }
    def construct(self):
        self.plane = NumberPlane()
        self.add(self.plane)
        self.wait()

        points1 = [UP + RIGHT * 0.5, UP * 3 + RIGHT * 2, UP * 1 + RIGHT * 4]
        points2 = [
        UP * 2 + RIGHT * 2.5, UP * 3.5 + RIGHT * 3, 
        UP * 3.5 + RIGHT * 6, UP * 1 + RIGHT * 6,
        ]

        p1, p2 = self.get_polygons(points1, points2)
        self.play(
            ShowCreation(p1),
            ShowCreation(p2),
        )

        self.wait()

        diff_points = self.get_minkowski_diff(p1, p2)
        diff_poly = Polygon(*diff_points)
        diff_poly.set_stroke(color=self.diff_color, width=self.width, opacity=0.5)
        self.play(
            ShowCreation(diff_poly)
        )

        self.wait()

        arrows, all_dots, simplex_mob = self.run_GJK(p1, p2, diff_poly)
        self.wait()
        if simplex_mob:
            self.play(
                *[FadeOut(a) for a in arrows],
                *[FadeOut(d) for d in all_dots],
                FadeOut(simplex_mob)
            )
        else:
            self.play(
                *[FadeOut(a) for a in arrows],
                *[FadeOut(d) for d in all_dots],
            )

        self.wait()

        new_p1 = p1.copy().shift(LEFT * 0.5)
        new_p2 = p2.copy().shift(RIGHT * 0.5)
        new_diff_points = self.get_minkowski_diff(new_p1, new_p2)
        new_diff_poly = Polygon(*new_diff_points).set_stroke(color=self.diff_color, width=self.width, opacity=0.5)
        self.play(
            Transform(p1, new_p1),
            Transform(p2, new_p2),
            Transform(diff_poly, new_diff_poly),
        )

        self.wait()

        arrows, all_dots, simplex_mob = self.run_GJK(p1, p2, diff_poly)

        self.play(
            *[FadeOut(a) for a in arrows],
            *[FadeOut(d) for d in all_dots],
            FadeOut(simplex_mob),
            FadeOut(p1),
            FadeOut(p2),
            FadeOut(diff_poly),
        )

        p1, p2 = self.get_replacement_polygons()

        self.play(
            ShowCreation(p1),
            ShowCreation(p2),
        )
        self.wait()

        diff_points = self.get_minkowski_diff(p1, p2)
        diff_poly = Polygon(*diff_points)
        diff_poly.set_stroke(color=self.diff_color, width=self.width, opacity=0.5)
        self.play(
            ShowCreation(diff_poly)
        )

        self.wait()

        arrows, all_dots, simplex_mob = self.run_GJK(p1, p2, diff_poly, start=LEFT)
        self.wait()



    def get_polygons(self, points1, points2):
        p1 = Polygon(*points1).set_stroke(color=self.p1_color, width=self.width)
        p2 = Polygon(*points2).set_stroke(color=self.p2_color, width=self.width)
        return p1, p2

    def get_replacement_polygons(self):
        p1 = RegularPolygon(n=6).shift(UP * 2 + RIGHT * 2.5)
        p2 = RegularPolygon(n=5).shift(UP * 2 + RIGHT * 4.5)
        p1.scale(1.5).set_stroke(color=self.p1_color, width=self.width)
        p2.scale(1.5).set_stroke(color=self.p2_color, width=self.width)
        return p1, p2

    def run_GJK(self, p1, p2, diff, start=RIGHT):
        all_dots = []
        self.SIZE_FACTOR = 2 / 3 # used for scaling arrows
        d = start * 1 / self.SIZE_FACTOR
        simplex = []

        arrows = self.show_arrow(p1, p2, diff, d, simplex, animate=True)
        dots = self.show_support_points(p1, p2, d)
        all_dots.append(dots[-1])
        A = self.support(p1, p2, d)
        simplex.append(A)

        d = ORIGIN - A
        d = d / (np.linalg.norm(d) * self.SIZE_FACTOR)
        simplex_mob = None
        while True:
            new_arrows = self.show_arrow(p1, p2, diff, d, simplex)
            self.transform_arrows(arrows, new_arrows)
            dots = self.show_support_points(p1, p2, d)
            all_dots.append(dots[-1])
            simplex.append(dots[-1].get_center())
            halfspace = self.show_halfspace(arrows[2])
            if np.dot(simplex[-1], d) <= 0:
                self.play(
                    all_dots[-1].set_color, CROSS_RED,
                    all_dots[-1].scale, 1.2,
                )
                self.wait()
                self.remove(halfspace)
                # ORIGIN is not contained
                break
            else:
                self.plane.remove(halfspace)
                simplex_mob = self.update_simplex(simplex, simplex_mob, all_dots)
                d = self.contains_origin(simplex, d)
                if not isinstance(d, type(np.array([]))):
                    origin_dot = Dot().set_color(YELLOW)
                    all_dots.append(origin_dot)
                    self.play(
                        GrowFromCenter(origin_dot),
                        Flash(origin_dot),
                    )
                    break
        return arrows, all_dots, simplex_mob

    def update_simplex(self, simplex, simplex_mob, dots):
        new_simplex_mob = Polygon(*simplex)
        new_simplex_mob.set_stroke(color=self.highlight_c, width=self.width)
        if simplex_mob:
            self.play(
                Transform(simplex_mob, new_simplex_mob)
            )
        else:
            simplex_mob = new_simplex_mob
            self.play(
                ShowCreation(simplex_mob)
            )
        return simplex_mob

    def contains_origin(self, simplex, d):
        """
        Returns True if simplex contains origin
        Otherwise, returns new direction to search
        """
        A = simplex[-1]
        AO = ORIGIN - A
        if len(simplex) == 2:
            # line segment case
            B = simplex[0]
            AB = B - A
            ABperp = self.tripleProd(AB, AO, AB)
            # ABperp = self.get_normal_towards_origin(A, B)
            d = ABperp / (np.linalg.norm(ABperp) * self.SIZE_FACTOR)
            return d
        
        if len(simplex) > 3 or len(simplex) < 2:
            return False
        # triangle case
        B, C = simplex[1], simplex[0]
        AB = B - A
        AC = C - A
        ABperp = self.tripleProd(AC, AB, AB)
        ACperp = self.tripleProd(AB, AC, AC)

        if np.dot(ABperp, AO) > 0:
            remove_array(simplex, C)
            d = ABperp / (np.linalg.norm(ABperp) * self.SIZE_FACTOR)
            return d
        elif np.dot(ACperp, AO) > 0:
            remove_array(simplex, B)
            d = ACperp / (np.linalg.norm(ACperp) * self.SIZE_FACTOR)
            return d

        # Otherwise origin must be in triangle
        return True
        

    def show_arrow(self, p1, p2, diff, d, simplex, animate=False):
        if len(simplex) == 0:
            start = diff.get_center()
        elif len(simplex) == 1:
            start = simplex[-1]
        else:
            start = (simplex[-2] + simplex[-1]) / 2

        diff_arrow = Arrow(
            start, start + d
        ).set_color(self.diff_color)

        if animate:
            self.play(
                ShowCreation(diff_arrow)
            )
            self.wait()
        shift = np.negative(d) * 0.5 + DOWN * 0.5
        d_arrow = Arrow(p1.get_center() + shift, p1.get_center() + shift + d)
        d_arrow.set_color(self.p1_color)
            
        if animate:
            self.play(
                ShowCreation(d_arrow)
            )

        start = p2.get_center()
        d_neg_arrow = Arrow(
            start, 
            start + np.negative(d),
        ).set_color(self.p2_color)

        if animate:
            self.play(
                ShowCreation(d_neg_arrow)
            )
            self.wait()

        return d_arrow, d_neg_arrow, diff_arrow

    def transform_arrows(self, arrows, new_arrows):
        self.play(
            Transform(arrows[2], new_arrows[2])
        )
        self.wait()

        self.play(
            Transform(arrows[0], new_arrows[0]),
            Transform(arrows[1], new_arrows[1]),
        )
        self.wait()
    
    def show_support_points(self, p1, p2, d):
        d1, d2 = Dot().set_color(self.p1_color), Dot().set_color(self.p2_color)
        d3 = Dot().set_color(self.highlight_c)
        d1.move_to(self.get_support(p1, d))
        d2.move_to(self.get_support(p2, np.negative(d)))
        d3.move_to(self.support(p1, p2, d))

        self.play(
            Indicate(d1, color=self.p1_color),
            Indicate(d2, color=self.p2_color)
        )

        self.wait()

        self.play(
            ReplacementTransform(d1, d3),
            ReplacementTransform(d2, d3),
            run_time=2,
        )

        self.wait()

        return d1, d2, d3

    def show_halfspace(self, vector):
        halfspace = self.get_halfspace_normal(vector)
        # halfspace.scale(0.8)
        self.plane.add_to_back(halfspace)
        self.wait(2)
        return halfspace

class RemainingQuestions(Scene):
    def construct(self):
        big_questions = TextMobject("Big Questions").scale(1.2)
        big_questions.move_to(UP * 3.5)
        h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
        h_line.next_to(big_questions, DOWN)
        self.play(
            Write(big_questions),
            ShowCreation(h_line)
        )
        self.wait()

        questions = BulletedList(
            "How do we know if a point ``passed'' the origin?",
            "How do we find the next direction once we have two points?",
            "How do we check if our triangle contains the origin?",
            "How do we pick a new direction if triangle doesn't contain origin?",
            buff=0.3,
        ).next_to(h_line, DOWN * 0.5)

        rect = ScreenRectangle(height=3.5)
        rect.next_to(questions, DOWN * 0.5)

        questions.scale(0.8)
        questions.to_edge(LEFT * 3)
        for i in range(len(questions)):
            if i == 0:
                self.play(
                    Write(questions[i]),
                    Write(rect),
                )
            else:
                self.play(
                    Write(questions[i])
                )
            self.wait()

        self.play(
            FadeOut(rect)
        )
        self.wait()


        self.play(
            questions.fade_all_but, 0
        )

        self.wait()

        self.play(
            questions.fade_all_but, 1
        )

        self.wait()

        self.play(
            questions.fade_all_but, 2
        )

        self.wait()

        self.play(
            questions[3].set_fill, None, 1,
        )

        self.wait()

class PointPassedOriginTest(GJKUtils):
    CONFIG = {
        "d_color": GREEN_SCREEN,
        "a_color": YELLOW,
    }
    def construct(self):
        plane = NumberPlane()
        self.add(plane)
        self.wait()
        question = TextMobject(r"Did point $A$ ``pass'' origin?")

        B = RIGHT * 4
        A = UP * 1 + RIGHT * 0.5
        A_dot = Dot().move_to(A)
        B_dot = Dot().move_to(B).set_color(self.d_color)
        A_label = TexMobject("A").add_background_rectangle()
        B_label = TexMobject("B").add_background_rectangle()
        
        A_label.next_to(A_dot, UP)
        B_label.next_to(B_dot, UP)

        d = ORIGIN - B
        arrow = self.get_vector_arrow(B, B + normalize(d) * 2)
        arrow.set_color(self.a_color)
        d_label = TexMobject(r"\vec{d}").add_background_rectangle()
        d_label.next_to(arrow, DOWN * 0.5)
        self.play(
            GrowFromCenter(B_dot),
            Write(B_label),
        )

        self.add_foreground_mobject(B_dot)

        self.wait()

        self.play(
            Write(arrow),
            Write(d_label)
        )

        self.wait()

        self.play(
            GrowFromCenter(A_dot),
            Write(A_label)
        )

        self.add_foreground_mobject(A_dot)

        self.wait()

        question.move_to(UP * 3.5 + RIGHT * 3.6)
        question.add_background_rectangle()
        self.play(
            Write(question)
        )

        self.wait()

        halfspace = self.get_halfspace_normal(arrow)
        plane.add_to_back(halfspace)
        self.play(
            FadeIn(halfspace)
        )

        self.play(
            A_dot.set_color, CROSS_RED,
        )

        condition = TexMobject(r"A^T \vec{d} < 0 \Rightarrow ", r"\text{No}")
        condition[1].set_color(CROSS_RED)
        condition.add_background_rectangle()
        condition.next_to(question, DOWN)
        self.play(
            Write(condition)
        )
        self.wait()

        A = DOWN * 2 + RIGHT * 0.5
        B = UP * 1.5 + RIGHT * 5

        self.play(
            FadeOut(halfspace),
            FadeOut(A_dot),
            FadeOut(A_label),
            FadeOut(d_label),
            FadeOut(arrow),
            B_dot.shift, UP * 1.5 + RIGHT * 1,
            B_label.shift, UP * 1.5 + RIGHT * 1
        )
        plane.remove(halfspace)

        self.wait()

        d = ORIGIN - B
        arrow = self.get_vector_arrow(B, B + normalize(d) * 2)
        arrow.set_color(self.a_color)
        d_label.next_to(arrow, DOWN * 0.5)
        d_label.shift(UP * SMALL_BUFF * 2)
        self.play(
            Write(arrow),
            Write(d_label)
        )
        self.wait()

        halfspace = self.get_halfspace_normal(arrow)
        A_dot.move_to(A).set_color(WHITE)
        A_label.next_to(A, DOWN)
        self.play(
            GrowFromCenter(A_dot),
            Write(A_label),
        )
        self.wait()
        plane.add_to_back(halfspace)

        self.play(
            A_dot.set_color, GREEN_SCREEN,
            FadeIn(halfspace),
        )
        self.wait()

class LineNormalCase(GJKUtils):
    CONFIG = {
        "d_color": GREEN_SCREEN,
        "a_color": ORANGE,
        "ao_color": YELLOW,
        "cross_color": BLUE,
        "width": 5,
    }
    def construct(self):
        plane = NumberPlane()
        self.add(plane)
        self.wait()

        A = LEFT * 4 + UP * 1 
        B = RIGHT * 1 + DOWN * 3

        A_dot = Dot().move_to(A).set_color(self.d_color)
        B_dot = Dot().move_to(B).set_color(self.d_color)
        A_label = TexMobject("A").add_background_rectangle()
        B_label = TexMobject("B").add_background_rectangle()
        
        A_label.next_to(A_dot, LEFT)
        B_label.next_to(B_dot, RIGHT)

        d = ORIGIN - B
        arrow = self.get_vector_arrow(B, B + normalize(d) * 2)
        arrow.set_color(self.a_color)
        d_label = TexMobject(r"\vec{d}").add_background_rectangle()
        d_label.next_to(arrow, LEFT * 0.5)
        d_label.shift(RIGHT * SMALL_BUFF * 2)
        self.play(
            GrowFromCenter(B_dot),
            Write(B_label),
        )

        self.add_foreground_mobject(B_dot)

        self.wait()

        self.play(
            Write(arrow),
            Write(d_label)
        )

        self.wait()

        self.play(
            GrowFromCenter(A_dot),
            Write(A_label)
        )

        self.add_foreground_mobject(A_dot)

        self.wait()

        AB = Line(A, B).set_stroke(color=self.d_color, width=self.width)
        self.play(
            FadeOut(d_label),
            FadeOut(arrow),
            Write(AB),
        )

        self.wait()

        AB_v = normalize(B - A)
        dashed_end_left = A - AB_v * 4
        dashed_end_right = B + AB_v * 1.5
        dash_left = DashedLine(A, dashed_end_left, dash_length=0.1)
        dash_right = DashedLine(B, dashed_end_right, dash_length=0.1)
        self.play(
            Write(dash_left),
            Write(dash_right),
        )

        self.add_foreground_mobject(A_dot)
        self.add_foreground_mobject(B_dot)
        self.add_foreground_mobject(AB)
        self.add_foreground_mobject(dash_left)
        self.add_foreground_mobject(dash_right)

        self.wait()



        left_region = Polygon(
            dashed_end_left, 
            FRAME_X_RADIUS * LEFT + FRAME_Y_RADIUS * DOWN, 
            dashed_end_right
        )
        left_region.set_stroke(color=BLUE, opacity=0.3)
        left_region.set_fill(color=BLUE, opacity=0.3)
        right_region = Polygon(
            dashed_end_left, 
            FRAME_X_RADIUS * LEFT + FRAME_Y_RADIUS * UP, 
            FRAME_X_RADIUS * RIGHT + FRAME_Y_RADIUS * UP,
            FRAME_X_RADIUS * RIGHT + FRAME_Y_RADIUS * DOWN,
            dashed_end_right, 
        )
        right_region.set_stroke(color=VIOLET, opacity=0.3)
        right_region.set_fill(color=VIOLET, opacity=0.3)

        self.play(
            FadeIn(left_region),
            FadeIn(right_region)
        )

        self.wait()

        O_dot = Dot().set_color(YELLOW)
        self.play(
            Flash(O_dot),
            GrowFromCenter(O_dot),
        )
        self.wait()

        n1 = self.get_normal_towards_origin(A, B)
        n1 = normalize(n1)
        n1_arrow = self.get_vector_arrow(AB.get_center(), AB.get_center() + n1 * 1.2)
        n1_arrow.set_color(self.a_color)
        self.play(
            Write(n1_arrow),
            FadeOut(left_region),
            FadeOut(right_region),
            FadeOut(dash_left),
            FadeOut(dash_right),
        )

        self.wait()

        AB_arrow = self.get_vector_arrow(A, B)
        AB_arrow.set_color(self.d_color)
        self.play(
            ReplacementTransform(AB, AB_arrow),
            n1_arrow.set_stroke, None, None, 0.5,
            n1_arrow.set_fill, None, 0.5,
        )
        self.wait()

        AB_label = TexMobject(r"\vec{AB}").set_color(self.d_color).add_background_rectangle()
        AB_label.move_to(AB_arrow.get_center()).shift(DL * 0.5)

        O_label = TexMobject("O").add_background_rectangle()
        O_label.move_to(DR * 0.5)

        AO_arrow = self.get_vector_arrow(A, ORIGIN)
        AO_arrow.set_color(self.ao_color)
        self.play(
            Write(O_label),
            Write(AO_arrow),
        )
        self.add_foreground_mobject(O_dot)
        self.wait()

        AO_label = TexMobject(r"\vec{AO}").set_color(self.ao_color)
        AO_label.add_background_rectangle()
        AO_label.move_to(AO_arrow.get_center())
        AO_label.shift(normalize(UP * 4 + RIGHT * 1) * 0.5)

        self.play(
            Write(AO_label),
            Write(AB_label)
        )
        self.wait() 
        ABxAO = normalize(cross(B - A, ORIGIN - A))
        ABxAO_vec = Arrow(A, A + ABxAO * 2, buff=0).set_color(BLUE)

        self.move_camera(0.45*np.pi/2, -0.5*np.pi, distance=9)
        self.wait()
        
        self.play(
            Write(ABxAO_vec)
        )

        self.begin_ambient_camera_rotation()

        self.wait(2)

        ABxAO_label = TexMobject(r"\vec{AB} ",  r"\times", r" \vec{AO}").set_shade_in_3d(True)
        ABxAO_label[0].set_color(self.d_color)
        ABxAO_label[2].set_color(self.ao_color)
        ABxAO_label.rotate(PI/2, axis=RIGHT)
        ABxAO_label.next_to(ABxAO_vec, OUT)

        self.play(
            Write(ABxAO_label)
        )

        self.wait(10)

        triple_cross = self.tripleProd(B - A, ORIGIN - A, B - A)
        triple_cross = normalize(triple_cross)
        triple_cross_arrow = Arrow(A, A + triple_cross * 1.2, buff=0)
        triple_cross_arrow.set_color(self.a_color)

        triple_cross_label = TexMobject("(", r"\vec{AB} ", r"\times", r"\vec{AO}", ")", r"\times", r"\vec{AB}")
        triple_cross_label[1].set_color(self.d_color)
        triple_cross_label[3].set_color(self.ao_color)
        triple_cross_label[6].set_color(self.d_color)
        triple_cross_label.add_background_rectangle()
        d_equal = TexMobject(r"\vec{d} = ").scale(1.1).add_background_rectangle()
        
        self.play(
            Write(triple_cross_arrow)
        )

        triple_cross_label.next_to(triple_cross_arrow, UR).shift(LEFT * 1.8)
        d_equal.next_to(triple_cross_label, LEFT, buff=SMALL_BUFF)
        d_equal.shift(UP * SMALL_BUFF)
        self.play(
            Write(triple_cross_label)
        )

        self.wait(5)

        self.stop_ambient_camera_rotation()
        fadeouts = [FadeOut(ABxAO_label), FadeOut(ABxAO_vec),]
        self.move_camera(0, -np.pi/2, added_anims=fadeouts, run_time=2)
        self.wait(3)
        d_label.scale(0.8).move_to(n1_arrow.get_center())
        d_label.shift(normalize(B - A) * 0.3)
        self.play(
            triple_cross_arrow.move_to, n1_arrow.get_center(),
            Write(d_equal),
            Write(d_label)
        )

        self.wait()

class TriangleCase(GJKUtils):
    ### This scene is inaccurate info, updated scene is TriangleVoronoiCase
    CONFIG = {
        "triangle_color": GREEN_SCREEN,
        "normal_color": ORANGE,
        "cross_color": BLUE,
        "width": 5,
    } 
    def construct(self):
        plane = NumberPlane()
        self.add(plane)
        self.wait()

        B = LEFT * 4 + UP * 1 
        C = RIGHT * 1 + DOWN * 3

        B_dot = Dot().move_to(B).set_color(self.triangle_color)
        C_dot = Dot().move_to(C).set_color(self.triangle_color)
        B_label = TexMobject("B").add_background_rectangle()
        C_label = TexMobject("C").add_background_rectangle()
        
        B_label.next_to(B_dot, DOWN)
        C_label.next_to(C_dot, RIGHT)

        d = ORIGIN - C
        arrow = self.get_vector_arrow(C, C + normalize(d) * 2)
        arrow.set_color(self.normal_color)
        d_label = TexMobject(r"\vec{d}").add_background_rectangle()
        d_label.next_to(arrow, LEFT * 0.5)
        d_label.shift(RIGHT * SMALL_BUFF * 2)
        self.play(
            GrowFromCenter(C_dot),
            Write(C_label),
        )

        self.add_foreground_mobject(C_dot)

        self.wait()

        self.play(
            Write(arrow),
            Write(d_label)
        )

        self.wait()

        self.play(
            GrowFromCenter(B_dot),
            Write(B_label)
        )

        self.add_foreground_mobject(B_dot)

        self.wait()

        BC = Line(B, C).set_stroke(color=self.triangle_color, width=self.width)
        self.play(
            FadeOut(d_label),
            FadeOut(arrow),
            Write(BC),
        )

        self.wait()

        n1 = self.get_normal_towards_origin(B, C)
        n1 = normalize(n1)
        n1_arrow = self.get_vector_arrow(BC.get_center(), BC.get_center() + n1 * 1.2)
        n1_arrow.set_color(self.normal_color)
        self.play(
            Write(n1_arrow)
        )

        self.wait()

        A = RIGHT * 5 + UP * 2

        A_dot = Dot().move_to(A).set_color(self.triangle_color)
        A_label = TexMobject("A").add_background_rectangle()
        
        A_label.next_to(A_dot, DOWN)

        self.play(
            GrowFromCenter(A_dot),
            Write(A_label)
        )
        self.add_foreground_mobject(A_dot)

        self.wait()

        AB = Line(A, B).set_stroke(color=self.triangle_color, width=self.width)
        AC = Line(A, C).set_stroke(color=self.triangle_color, width=self.width)

        self.play(
            Write(AB),
            Write(AC),
            FadeOut(n1_arrow),
        )

        self.wait()

        all_dashed_lines, regions = self.get_regions(A, B, C)
        labels = self.show_region_labels()
        original_R_A_pos = labels[0].get_center()
        self.play(
            *[ApplyMethod(p.set_fill, None, 0.3) for p in regions[:-1]],
            regions[-1].set_fill, None, 0.22,
            LaggedStartMap(Write, labels),
            run_time=2
        )
        self.wait()

        self.add_all_updaters(A, B, C, A_dot, A_label, AB, AC, regions, all_dashed_lines, labels)

        self.play(
            A_dot.shift, LEFT * 2.5 + UP,
            labels[0].move_to, RIGHT * 3.3 + UP * 3.6,
            run_time=2
        )

        self.play(
            A_dot.shift, DOWN * 2.5 + RIGHT * 2.5,
            labels[0].move_to, RIGHT * 6.5 + UP * 1,
            run_time=2
        )

        self.play(
            A_dot.move_to, A,
            labels[0].move_to, original_R_A_pos,
            run_time=2      
        )
        self.wait()

        self.clear_all_updaters(A, B, C, A_dot, A_label, AB, AC, regions, all_dashed_lines, labels)

        d_arrow, line_to_pass = self.retrace_initial_steps(A_dot, B_dot, C_dot, regions, labels, all_dashed_lines)
        original_d_arrow = d_arrow.copy()
        self.add_all_updaters(A, B, C, A_dot, A_label, AB, AC, regions, all_dashed_lines, labels)

        self.play(
            A_dot.move_to, RIGHT * 4 + DOWN * 1.5,
            labels[0].move_to, RIGHT * 6 + DOWN * 1.5,
            labels[1].shift, RIGHT * 2 + DOWN,
            labels[-1].shift, DOWN,
            labels[-2].shift, DOWN * 2 + RIGHT,
            run_time=2
        )
        self.clear_all_updaters(A, B, C, A_dot, A_label, AB, AC, regions, all_dashed_lines, labels)
        self.wait()

        O_label = TexMobject("O")
        O_dot = Dot().set_color(YELLOW)
        O_label.next_to(O_dot, UL * 0.5)
        self.add_foreground_mobject(O_dot)
        self.play(
            FadeOut(line_to_pass),
            GrowFromCenter(O_dot),
            Write(O_label),
        )
        self.wait()

        n1 = self.get_normal_towards_origin(B, A_dot.get_center())
        n1 = normalize(n1)
        new_d_arrow = Arrow((B + A_dot.get_center()) / 2, (B + A_dot.get_center()) / 2 + n1 * 1.2, buff=0).set_color(YELLOW)
        new_d_arrow.shift(normalize(A_dot.get_center() - B) * 0.5)

        self.play(
            Transform(d_arrow, new_d_arrow)
        )
        AB_perp_label = TexMobject(r"\vec{AB}_{\perp}")
        AB_perp_label.move_to(RIGHT * 1.5 + UP * 0.5)
        self.play(
            Write(AB_perp_label)
        )
        self.wait()

        AB_perp_calc = TexMobject(r"\vec{AB}_{\perp} = (\vec{AC} \times \vec{AB}) \times \vec{AB}")
        AB_perp_calc.scale(1).add_background_rectangle()
        AB_perp_calc.to_edge(LEFT).shift(DOWN * 2.5)
        self.play(
            Write(AB_perp_calc)
        )

        self.wait()

        AO_arrow = Arrow(A_dot.get_center(), ORIGIN, buff=0).set_color(FUCHSIA)
        self.play(
            d_arrow.put_start_and_end_on, 
            A_dot.get_center(), 
            A_dot.get_center() + (d_arrow.get_end() - d_arrow.get_start()),
            AB_perp_label.move_to, RIGHT * 5 + UP * 0.5,
        )

        self.wait()

        self.play(
            Write(AO_arrow),
        )

        self.wait()

        condition = TexMobject(r"\vec{AB}_{\perp} \boldsymbol{\cdot} \vec{AO} > 0 \rightarrow O \in R_{AB}")
        condition.add_background_rectangle()
        condition.next_to(AB_perp_calc, DOWN).to_edge(LEFT)
        self.play(
            Write(condition)
        )

        self.wait(3)

        self.play(
            FadeOut(condition),
            FadeOut(AO_arrow),
            FadeOut(AB_perp_label),
            FadeOut(AB_perp_calc),
            FadeOut(d_arrow),
        )

        self.wait()

        # To be continued
        # Move triangle back to original position
        # Analyze R_AC with similar process
        # Then analyze R_A
        # Finally discuss case when origin in R_ABC

        self.add_all_updaters(A, B, C, A_dot, A_label, AB, AC, regions, all_dashed_lines, labels)

        self.play(
            A_dot.move_to, A,
            labels[0].move_to, original_R_A_pos,
            labels[1].shift, LEFT * 2 + UP,
            labels[-1].shift, UP,
            labels[-2].shift, UP * 2 + LEFT,
            run_time=2
        )
        self.wait()

        self.clear_all_updaters(A, B, C, A_dot, A_label, AB, AC, regions, all_dashed_lines, labels)
        self.highlight_region_AC(A, C, all_dashed_lines, labels)

        d_arrow = original_d_arrow.copy()
        self.play(
            Write(d_arrow)
        )
        self.wait()
        self.play(
            Write(line_to_pass),
            O_label.shift, RIGHT * 0.7,
        )
        self.wait()

        self.add_all_updaters(A, B, C, A_dot, A_label, AB, AC, regions, all_dashed_lines, labels)

        self.play(
            A_dot.move_to, LEFT * 2 + UP * 3,
            labels[0].move_to, LEFT * 1.9 + UP * 3.7,
            labels[1].shift, LEFT * 3,
            labels[-1].shift, UP + LEFT * 2.7,
            labels[-2].shift, UP * 2 + LEFT * 2,
            run_time=2
        )
        self.clear_all_updaters(A, B, C, A_dot, A_label, AB, AC, regions, all_dashed_lines, labels)
        self.wait()

        n1 = self.get_normal_towards_origin(C, A_dot.get_center())
        n1 = normalize(n1)
        new_d_arrow = Arrow((C + A_dot.get_center()) / 2, (C + A_dot.get_center()) / 2 + n1 * 1, buff=0).set_color(YELLOW)
        new_d_arrow.shift(normalize(A_dot.get_center() - C) * 0.7)

        self.play(
            Transform(d_arrow, new_d_arrow),
            FadeOut(line_to_pass),
        )
        AC_perp_label = TexMobject(r"\vec{AC}_{\perp}")
        AC_perp_label.move_to(RIGHT * 0.5 + UP * 1.5)
        self.play(
            Write(AC_perp_label)
        )
        self.wait()

        AC_perp_calc = TexMobject(r"\vec{AC}_{\perp} = (\vec{AB} \times \vec{AC}) \times \vec{AC}")
        AC_perp_calc.scale(1).add_background_rectangle()
        AC_perp_calc.to_edge(LEFT).shift(DOWN * 2.5)
        self.play(
            Write(AC_perp_calc)
        )

        self.wait()

        AO_arrow = Arrow(A_dot.get_center(), ORIGIN, buff=0).set_color(FUCHSIA)
        self.play(
            d_arrow.put_start_and_end_on, 
            A_dot.get_center(), 
            A_dot.get_center() + (d_arrow.get_end() - d_arrow.get_start()),
            AC_perp_label.move_to, UP * 3.5 + LEFT * 0.5,
        )

        self.wait()

        self.play(
            Write(AO_arrow),
        )

        self.wait()

        condition = TexMobject(r"\vec{AC}_{\perp} \boldsymbol{\cdot} \vec{AO} > 0 \rightarrow O \in R_{AC}")
        condition.add_background_rectangle()
        condition.next_to(AC_perp_calc, DOWN).to_edge(LEFT)
        self.play(
            Write(condition)
        )

        self.wait(3)

        self.play(
            FadeOut(condition),
            FadeOut(AO_arrow),
            FadeOut(AC_perp_label),
            FadeOut(AC_perp_calc),
            FadeOut(d_arrow),
        )

        self.wait()

        self.add_all_updaters(A, B, C, A_dot, A_label, AB, AC, regions, all_dashed_lines, labels)

        self.play(
            A_dot.move_to, A,
            labels[0].move_to, original_R_A_pos,
            labels[1].shift, RIGHT * 3,
            labels[-1].shift, DOWN + RIGHT * 2.7,
            labels[-2].shift, DOWN * 2 + RIGHT * 2,
            run_time=2
        )

        self.wait()

        self.clear_all_updaters(A, B, C, A_dot, A_label, AB, AC, regions, all_dashed_lines, labels)

        self.highlight_region_A(A, all_dashed_lines, labels)
        n1 = self.get_normal_towards_origin(B, C)
        n1 = normalize(n1)
        d_arrow = Arrow((B + C) / 2, (B + C) / 2 + n1 * 1, buff=0).set_color(YELLOW)
        self.play(
            Write(d_arrow)
        )

        self.play(
            FadeOut(O_label),
            Write(line_to_pass),
        )

        self.add_all_updaters(A, B, C, A_dot, A_label, AB, AC, regions, all_dashed_lines, labels)
        original_R_AB_pos = labels[1].get_center()
        original_R_ABC_pos = labels[-1].get_center()
        original_R_AC_pos = labels[-2].get_center()


        self.play(
            A_dot.move_to, LEFT * 2 + UP * 2,
            labels[0].move_to, LEFT * 1.5 + UP * 3.5,
            labels[1].shift, LEFT * 3,
            labels[-1].shift, LEFT * 2.5 + UP,
            labels[-2].shift, LEFT * 2 + UP,
            run_time=3
        )

        self.wait()

        self.play(
            A_dot.move_to, UR * 0.2,
            labels[0].move_to, RIGHT * 1.5 + UP * 1.5,
            labels[1].shift, RIGHT * 1.5 + DOWN,
            labels[-1].shift, RIGHT * 1.5 + DOWN * 1.5,
            labels[-2].shift, RIGHT * 2 + DOWN * 1.5,
            run_time=3
        )

        self.wait()

        self.play(
            A_dot.move_to, RIGHT * 3 + DOWN * 2,
            labels[0].move_to, DOWN * 2 + RIGHT * 4.5,
            labels[1].shift, RIGHT * 4.5,
            labels[-2].shift, RIGHT * 0.5 + DOWN * 1.5,
            run_time=3,
        )
        self.wait()

        self.play(
            A_dot.move_to, A,
            labels[0].move_to, original_R_A_pos,
            labels[1].move_to, original_R_AB_pos,
            labels[-1].move_to, original_R_ABC_pos,
            labels[-2].move_to, original_R_AC_pos,
            FadeOut(d_arrow),
            FadeOut(line_to_pass),
            run_time=2,
        )
        self.wait()

        self.clear_all_updaters(A, B, C, A_dot, A_label, AB, AC, regions, all_dashed_lines, labels)

        self.play(
            FadeOut(regions[0]),
            FadeOut(labels[0]),
        )
        self.wait()

        n_AB = self.get_normal_towards_origin(A, B)
        n_AB = normalize(n_AB)

        ab_perp_arrow = Arrow((A + B) / 2, (A + B) / 2 - n_AB * 1, buff=0).set_color(YELLOW)
        AB_perp_label.next_to(ab_perp_arrow, RIGHT)
        self.play(
            Write(ab_perp_arrow),
            Write(AB_perp_label),
        )
        self.wait()

        n_AC = self.get_normal_towards_origin(A, C)
        n_AC = normalize(n_AC)

        ac_perp_arrow = Arrow((A + C) / 2, (A + C) / 2 - n_AC * 1, buff=0).set_color(YELLOW)
        AC_perp_label.move_to(ac_perp_arrow.get_center()).shift(UR * 0.5)
        self.play(
            Write(ac_perp_arrow),
            Write(AC_perp_label),
        )

        self.wait()

        self.play(
            ab_perp_arrow.put_start_and_end_on, A, A - n_AB,
            ac_perp_arrow.put_start_and_end_on, A, A - n_AC,
            AB_perp_label.move_to, A - n_AB * 1.4,
            AC_perp_label.move_to, A - n_AC * 1.4,
            A_label.move_to, A_dot.get_center() + UP * 0.3 + RIGHT * 0.4
        )
        self.wait()

        AO_arrow = Arrow(A_dot.get_center(), ORIGIN, buff=0).set_color(FUCHSIA)

        O_label.move_to(LEFT * 0.4 + UP * 0.4)
        self.play(
            Write(O_label)
        )

        self.play(
            Write(AO_arrow)
        )

        self.wait()
        triangle_condition_AB = TexMobject(r"\vec{AB}_{\perp} \boldsymbol{\cdot} \vec{AO} < 0 \rightarrow O \notin R_{AB}").scale(0.9)
        triangle_condition_AB.add_background_rectangle()
        triangle_condition_AC = TexMobject(r"\vec{AC}_{\perp} \boldsymbol{\cdot} \vec{AO} < 0 \rightarrow O \notin R_{AC}").scale(0.9)
        triangle_condition_AC.add_background_rectangle()
        triangle_condition_true = TextMobject(r"This implies $O \in R_{ABC}$").scale(0.9)
        triangle_condition_true.add_background_rectangle()

        triangle_condition_AB.to_edge(LEFT).shift(DOWN * 1.5)
        triangle_condition_AC.to_edge(LEFT).shift(DOWN * 2.5)
        triangle_condition_true.to_edge(LEFT).shift(DOWN * 3.5)
        self.play(
            Write(triangle_condition_AB)
        )
        self.wait()
        self.play(
            Write(triangle_condition_AC)
        )
        self.wait()

        self.play(
            Write(triangle_condition_true)
        )
        self.wait()

        self.play(
            FadeOut(triangle_condition_true),
            FadeOut(triangle_condition_AC),
            FadeOut(triangle_condition_AB),
            FadeOut(AO_arrow),
            FadeOut(AB_perp_label),
            FadeOut(AC_perp_label),
            FadeOut(ab_perp_arrow),
            FadeOut(ac_perp_arrow),
        )
        self.wait()

        key_idea = TextMobject(
            r"Only max 2 regions ($R_{AB}$ and $R_{AC}$)" + "\\\\",
            "need to be checked to locate origin"
        ).scale(0.8)
        key_idea.add_background_rectangle()
        key_idea.to_edge(LEFT)
        key_idea.shift(DOWN * 2.5)

        self.play(
            Write(key_idea)
        )

        self.wait()

    def add_all_updaters(self, A, B, C, A_dot, A_label, AB, AC, regions, dashed_lines, labels):
        FACTOR = 7
        region_A = regions[0]
        region_AB = regions[1]
        region_AC = regions[5]
        region_ABC = regions[6]
        to_update_dashed_lines = [dashed_lines[0], dashed_lines[1]]

        always(A_label.next_to, A_dot, DOWN)
        DASH_LENGTH = 0.1
        
        to_update_dashed_lines[0].add_updater(lambda m: m.become(
                DashedLine(
                A_dot.get_center(), A_dot.get_center() + (A_dot.get_center() - B) * FACTOR,
                dash_length=DASH_LENGTH
                )
            )
        )

        to_update_dashed_lines[1].add_updater(lambda m: m.become(
                DashedLine(
                A_dot.get_center(), A_dot.get_center() + (A_dot.get_center() - C) * FACTOR,
                dash_length=DASH_LENGTH
                )
            )
        )

        always_redraw(
            lambda: AB.put_start_and_end_on(
                A_dot.get_center(), B
            )
        )
        always_redraw(
            lambda: AC.put_start_and_end_on(
                A_dot.get_center(), C
            )
        )

        region_ABC.add_updater(
            lambda m: m.become(
                Polygon(C, B, A_dot.get_center()).set_stroke(width=0).set_fill(
                    color=region_ABC.get_color(), opacity=0.22
                )
            )
        )

        region_AB.add_updater(
            lambda m: m.become(
                Polygon(
                    B, dashed_lines[2].get_end(), to_update_dashed_lines[1].get_end(), A_dot.get_start()
                ).set_stroke(width=0).set_fill(color=region_AB.get_color(), opacity=0.3)
            )
        )

        region_A.add_updater(
            lambda m: m.become(
                Polygon(
                    A_dot.get_center(), to_update_dashed_lines[0].get_end(), to_update_dashed_lines[1].get_end(),
                ).set_stroke(width=0).set_fill(color=region_A.get_color(), opacity=0.3)
            )
        )

        region_AC.add_updater(
            lambda m: m.become(
                Polygon(
                    A_dot.get_center(), to_update_dashed_lines[0].get_end(), dashed_lines[5].get_end(), C,
                ).set_stroke(width=0).set_fill(color=region_AC.get_color(), opacity=0.3)
            )
        )
    
    def clear_all_updaters(self, A, B, C, A_dot, A_label, AB, AC, regions, dashed_lines, labels):
        region_A = regions[0]
        region_AB = regions[1]
        region_AC = regions[5]
        region_ABC = regions[6]
        to_update_dashed_lines = [dashed_lines[0], dashed_lines[1]]

        A_label.clear_updaters()

        to_update_dashed_lines[0].clear_updaters()

        to_update_dashed_lines[1].clear_updaters()

        AB.clear_updaters()
        AC.clear_updaters()

        region_ABC.clear_updaters()

        region_AB.clear_updaters()

        region_A.clear_updaters()

        region_AC.clear_updaters()

    def retrace_initial_steps(self, A_dot, B_dot, C_dot, regions, labels, dashed_lines):
        A, B, C = A_dot.get_center(), B_dot.get_center(), C_dot.get_center()
        n1 = self.get_normal_towards_origin(B, C)
        n1 = normalize(n1)
        d_arrow = Arrow((B + C) / 2, (B + C) / 2 + n1 * 1.2, buff=0).set_color(YELLOW)
        self.play(
            Write(d_arrow)
        )
        self.wait()

        R_C, region_C = labels[4], regions[4]
        d1, d2 = normalize(dashed_lines[5].get_end() - C), normalize(dashed_lines[4].get_end() - C)
        region_C_outline = [C + d2 * 2, C + d1*2, C]
        region_C_outline = Polygon(*region_C_outline).set_stroke(color=YELLOW, width=5)
    
        self.play(
            Indicate(R_C),
            ShowCreationThenDestruction(region_C_outline),
        )
        self.wait()

        R_BC, region_BC = labels[3], regions[3]
        d1, d2 = normalize(dashed_lines[3].get_end() - B), normalize(dashed_lines[4].get_end() - C)
        region_BC_outline = VGroup().set_points_as_corners(*[[B + d1 * 4, B, C, C + d2 * 1.5]])
        region_BC_outline.set_stroke(color=YELLOW, width=5)

        self.play(
            Indicate(R_BC),
            ShowCreationThenDestruction(region_BC_outline)
        )
        self.wait()

        R_B, region_B = labels[2], regions[2]
        d1, d2 = normalize(dashed_lines[3].get_end() - B), normalize(dashed_lines[2].get_end() - B)
        region_B_outline = [B + d2 * 4.5, B + d1 * 4, B]
        region_B_outline = Polygon(*region_B_outline).set_stroke(color=YELLOW, width=5)

        self.play(
            Indicate(R_B),
            ShowCreationThenDestruction(region_B_outline),
        )
        self.wait()

        self.play(
            FadeOut(region_B),
            FadeOut(R_B),
            FadeOut(region_C),
            FadeOut(R_C),
            FadeOut(region_BC),
            FadeOut(R_BC),
            FadeOut(dashed_lines[3]),
            FadeOut(dashed_lines[4]),
        )
        self.wait()

        R_AB, region_AB = labels[1], regions[1]
        d1, d2 = normalize(dashed_lines[2].get_end() - B), normalize(dashed_lines[1].get_end() - A)
        region_AB_outline = VGroup().set_points_as_corners(*[[B + d1 * 5, B, A, A + d2 * 2.5]])
        region_AB_outline.set_stroke(color=YELLOW, width=5)
        self.play(
            Indicate(R_AB),
            ShowCreationThenDestruction(region_AB_outline),
        )
        self.wait()

        BC_vector = B - C
        start_line_to_pass = BC_vector
        end_line_to_pass = -BC_vector
        line_to_pass = DashedLine(start_line_to_pass, end_line_to_pass)
        line_to_pass.set_color(YELLOW)

        self.play(
            Write(line_to_pass)
        )
        self.wait()

        return d_arrow, line_to_pass

    def highlight_region_AC(self, A, C, dashed_lines, labels):
        R_AC, BC_cont, BA_cont = labels[5], dashed_lines[5], dashed_lines[0]
        region_AC_outline = VGroup()
        BC_cont_normal = normalize(BC_cont.get_end() - C)
        BA_cont_normal = normalize(BA_cont.get_end() - A)
        region_AC_outline.set_points_as_corners(*[[C + BC_cont_normal * 1.5, C, A, A + BA_cont_normal * 3]])
        region_AC_outline.set_stroke(color=YELLOW, width=5)
        self.play(
            Indicate(R_AC),
            ShowCreationThenDestruction(region_AC_outline)
        )

        self.wait()

    def highlight_region_A(self, A, dashed_lines, labels):
        R_A, CA_cont, BA_cont = labels[0], dashed_lines[1], dashed_lines[0]
        region_A_outline = VGroup()
        CA_cont_normal = normalize(CA_cont.get_end() - A)
        BA_cont_normal = normalize(BA_cont.get_end() - A)
        region_A_outline.set_points_as_corners(*[[A + BA_cont_normal * 3, A,  A + CA_cont_normal * 3]])
        region_A_outline.set_stroke(color=YELLOW, width=5)
        self.play(
            Indicate(R_A),
            ShowCreationThenDestruction(region_A_outline)
        )

        self.wait()

    def get_regions(self, A, B, C):
        AC = normalize(C - A)
        AB = normalize(B - A)
        BC = normalize(C - B)
        FACTOR = 7
        DASH_LENGTH = 0.1

        CA_cont = DashedLine(A, A - AC * FACTOR, dash_length=DASH_LENGTH)
        BA_cont = DashedLine(A, A - AB * FACTOR, dash_length=DASH_LENGTH)

        AB_cont = DashedLine(B, B + AB * FACTOR, dash_length=DASH_LENGTH)
        CB_cont = DashedLine(B, B - BC * FACTOR, dash_length=DASH_LENGTH)

        BC_cont = DashedLine(C, C + BC * FACTOR, dash_length=DASH_LENGTH)
        AC_cont = DashedLine(C, C + AC * FACTOR, dash_length=DASH_LENGTH)

        all_dashed_lines = VGroup(
            BA_cont, CA_cont, CB_cont,
            AB_cont, AC_cont, BC_cont
        )

        self.play(
            Write(all_dashed_lines),
            run_time=2
        )
        self.wait()
        regions = []
        colors = [BLUE_E, VIOLET, PERIWINKLE_BLUE, COBALT_BLUE, PURPLE, SEAFOAM_GREEN]
        for i in range(len(all_dashed_lines)):
            lines = [all_dashed_lines[i], all_dashed_lines[(i + 1) % len(all_dashed_lines)]]
            region = self.get_voronoi_region(lines, fill_color=colors[i])
            self.play(
                FadeIn(region)
            )
            regions.append(region)
        triangle_region = Polygon(A, B, C).set_stroke(width=0).set_fill(color=self.triangle_color, opacity=0.4)
        regions.append(triangle_region)
        self.play(
            FadeIn(triangle_region)
        )
        return all_dashed_lines, VGroup(*regions)

    def get_voronoi_region(self, lines, fill_color):
        p1, p2 = lines[0].get_start(), lines[1].get_start()
        if np.allclose(p1, p2):
            polygon = Polygon(p1, lines[0].get_end(), lines[1].get_end())
        else:
            polygon = Polygon(p1, p2, lines[1].get_end(), lines[0].get_end())
        polygon.set_stroke(width=0)
        polygon.set_fill(color=fill_color, opacity=0.5)
        return polygon

    def show_region_labels(self):
        r_A = TexMobject("R_A").move_to(RIGHT * 6.5 + UP * 3)
        r_AB = TexMobject("R_{AB}").move_to(LEFT * 1.5 + UP * 2.5)
        r_B = TexMobject("R_B").move_to(LEFT * 5.5 + UP * 1.5)
        r_BC = TexMobject("R_{BC}").move_to(LEFT * 3.5 + DOWN * 2.5)
        r_C = TexMobject("R_C").move_to(RIGHT * 1.1 + DOWN * 3.6)
        r_AC = TexMobject("R_{AC}").move_to(RIGHT * 4.5 + DOWN * 1.5)
        r_ABC = TexMobject("R_{ABC}").move_to(RIGHT * 0.7 + DOWN * 0.5)
        return VGroup(*[r_A, r_AB, r_B, r_BC, r_C, r_AC, r_ABC])

class TriangleVoronoiCase(GJKUtils):
    CONFIG = {
        "triangle_color": GREEN_SCREEN,
        "normal_color": ORANGE,
        "cross_color": BLUE,
        "width": 5,
    } 
    def construct(self):
        plane = NumberPlane()
        self.add(plane)
        self.wait()

        B = LEFT * 4 + UP * 1 
        C = RIGHT * 1 + DOWN * 3

        B_dot = Dot().move_to(B).set_color(self.triangle_color)
        C_dot = Dot().move_to(C).set_color(self.triangle_color)
        B_label = TexMobject("B").add_background_rectangle()
        C_label = TexMobject("C").add_background_rectangle()
        
        B_label.next_to(B_dot, DOWN)
        C_label.next_to(C_dot, RIGHT)

        d = ORIGIN - C
        arrow = self.get_vector_arrow(C, C + normalize(d) * 2)
        arrow.set_color(self.normal_color)
        d_label = TexMobject(r"\vec{d}").add_background_rectangle()
        d_label.next_to(arrow, LEFT * 0.5)
        d_label.shift(RIGHT * SMALL_BUFF * 2)
        self.play(
            GrowFromCenter(C_dot),
            Write(C_label),
        )

        self.add_foreground_mobject(C_dot)

        self.wait()

        self.play(
            Write(arrow),
            Write(d_label)
        )

        self.wait()

        self.play(
            GrowFromCenter(B_dot),
            Write(B_label)
        )

        self.add_foreground_mobject(B_dot)

        self.wait()

        BC = Line(B, C).set_stroke(color=self.triangle_color, width=self.width)
        self.play(
            FadeOut(d_label),
            FadeOut(arrow),
            Write(BC),
        )

        self.wait()

        n1 = self.get_normal_towards_origin(B, C)
        n1 = normalize(n1)
        n1_arrow = self.get_vector_arrow(BC.get_center(), BC.get_center() + n1 * 1.2)
        n1_arrow.set_color(self.normal_color)
        self.play(
            Write(n1_arrow)
        )

        self.wait()

        A = RIGHT * 5 + UP * 2

        A_dot = Dot().move_to(A).set_color(self.triangle_color)
        A_label = TexMobject("A").add_background_rectangle()
        
        A_label.next_to(A_dot, DOWN)

        self.play(
            GrowFromCenter(A_dot),
            Write(A_label)
        )
        self.add_foreground_mobject(A_dot)

        self.wait()

        AB = Line(A, B).set_stroke(color=self.triangle_color, width=self.width)
        AC = Line(A, C).set_stroke(color=self.triangle_color, width=self.width)

        self.play(
            Write(AB),
            Write(AC),
            FadeOut(n1_arrow),
        )

        self.wait()

        all_dashed_lines, regions = self.get_regions(A, B, C)
        self.play(
            Write(all_dashed_lines)
        )
        self.wait()

        self.play(
            LaggedStartMap(FadeIn, regions)
        )
        self.wait()

        labels = self.show_region_labels()
        original_R_A_pos = labels[0].get_center()
        self.play(
            *[ApplyMethod(p.set_fill, None, 0.3) for p in regions[:-1]],
            regions[-1].set_fill, None, 0.22,
            LaggedStartMap(Write, labels),
            run_time=2
        )
        self.wait()

        voronoi_regions = TextMobject("Voronoi Regions").set_color(YELLOW)
        voronoi_regions.move_to(DOWN * 2.5 + LEFT * 3.5)
        self.play(
            Write(voronoi_regions)
        )
        self.wait()

        self.play(
            FadeOut(voronoi_regions)
        )
        self.wait()

        d_arrow = self.get_direction_arrow_BCperp(A_dot, B_dot, C_dot)
        original_d_arrow = d_arrow.copy()
        self.play(
            Write(d_arrow)
        )
        self.wait()

        self.add_all_updaters(A, B, C, A_dot, A_label, AB, AC, regions, all_dashed_lines, labels)

        self.play(
            A_dot.shift, LEFT * 2.5 + UP,
            labels[0].move_to, RIGHT * 3.3 + UP * 3.6,
            labels[1].shift, LEFT * 1.5 + UP * 0.5,
            run_time=2
        )

        self.play(
            A_dot.shift, DOWN * 2.5 + RIGHT * 2.5,
            labels[0].move_to, RIGHT * 6.5 + UP * 1,
            labels[1].shift, RIGHT * 1.5 + DOWN * 0.5,
            run_time=2
        )

        key_question = TextMobject(
            "How many regions do we need to check to" + "\\\\",
            "determine which region contains the origin?"
        ).scale(0.7)
        key_question.move_to(DOWN * 3 + LEFT * 3.5)

        self.play(
            A_dot.move_to, A,
            labels[0].move_to, original_R_A_pos,
            Write(key_question),
            run_time=2      
        )
        self.wait()

        remove_question = TextMobject(
            "Are there any regions that can" + "\\\\", 
            "never contain the origin?"
        ).scale(0.8)
        remove_question.move_to(key_question.get_center())

        self.clear_all_updaters(A, B, C, A_dot, A_label, AB, AC, regions, all_dashed_lines, labels)

        self.play(
            ReplacementTransform(key_question, remove_question),
            run_time=2,
        )
        self.wait()

        self.play(
            FadeOut(d_arrow),
            FadeOut(remove_question),
        )
        self.wait()

        d_arrow, line_to_pass = self.retrace_initial_steps(A_dot, B_dot, C_dot, regions, labels, all_dashed_lines)
        self.add_all_updaters(A, B, C, A_dot, A_label, AB, AC, regions, all_dashed_lines, labels)

        self.play(
            A_dot.move_to, RIGHT * 4 + DOWN * 1.5,
            labels[1].shift, DOWN,
            labels[-1].shift, DOWN,
            labels[-2].shift, DOWN * 2 + LEFT * 1.5,
            run_time=2
        )
        self.clear_all_updaters(A, B, C, A_dot, A_label, AB, AC, regions, all_dashed_lines, labels)
        self.wait()

        O_label = TexMobject("O")
        O_dot = Dot().set_color(YELLOW)
        O_label.next_to(O_dot, UL * 0.5)
        self.add_foreground_mobject(O_dot)
        self.play(
            FadeOut(line_to_pass),
            GrowFromCenter(O_dot),
            Write(O_label),
        )
        self.wait()

        n1 = self.get_normal_towards_origin(B, A_dot.get_center())
        n1 = normalize(n1)
        new_d_arrow = Arrow((B + A_dot.get_center()) / 2, (B + A_dot.get_center()) / 2 + n1 * 1.2, buff=0).set_color(YELLOW)
        new_d_arrow.shift(normalize(A_dot.get_center() - B) * 0.5)

        self.play(
            Transform(d_arrow, new_d_arrow)
        )
        AB_perp_label = TexMobject(r"\vec{AB}_{\perp}")
        AB_perp_label.move_to(RIGHT * 1.5 + UP * 0.5)
        self.play(
            Write(AB_perp_label)
        )
        self.wait()

        AB_perp_calc = TexMobject(r"\vec{AB}_{\perp} = (\vec{AC} \times \vec{AB}) \times \vec{AB}").scale(0.9)
        AB_perp_calc.scale(1).add_background_rectangle()
        AB_perp_calc.to_edge(LEFT).shift(DOWN * 1.5)
        self.play(
            Write(AB_perp_calc)
        )

        self.wait()


        AO_arrow = Arrow(A_dot.get_center(), ORIGIN, buff=0).set_color(FUCHSIA)
        self.play(
            d_arrow.put_start_and_end_on, 
            A_dot.get_center(), 
            A_dot.get_center() + (d_arrow.get_end() - d_arrow.get_start()),
            AB_perp_label.move_to, RIGHT * 5 + DOWN * 0.5,
        )

        self.wait()

        self.play(
            Write(AO_arrow),
        )

        self.wait()

        condition = TexMobject(r"\vec{AB}_{\perp} \boldsymbol{\cdot} \vec{AO} > 0 \rightarrow O \in R_{AB}").scale(0.9)
        condition.add_background_rectangle()
        condition.next_to(DOWN * 2.5).to_edge(LEFT)
        self.play(
            Write(condition)
        )
        self.wait()

        cross_C = Cross(C_dot).scale(1.5).set_color(CROSS_RED)
        self.add_foreground_mobject(cross_C)

        next_step = TexMobject(r"\vec{d} \leftarrow \vec{AB}_{\perp}, \text{remove } C \text{ from simplex}").scale(0.9)
        next_step.add_background_rectangle()
        next_step.move_to(DOWN * 3.5).to_edge(LEFT)
        self.play(
            Write(next_step),
            Write(cross_C),
        )
        self.wait(3)

        self.play(
            FadeOut(condition),
            FadeOut(AO_arrow),
            FadeOut(AB_perp_label),
            FadeOut(AB_perp_calc),
            FadeOut(d_arrow),
            FadeOut(next_step),
            FadeOut(cross_C,)
        )

        self.wait()

        # # To be continued
        # # Move triangle back to original position
        # # Analyze R_AC with similar process
        # # Then analyze R_A
        # # Finally discuss case when origin in R_ABC

        self.add_all_updaters(A, B, C, A_dot, A_label, AB, AC, regions, all_dashed_lines, labels)

        self.play(
            A_dot.move_to, A,
            labels[1].shift, UP,
            labels[-1].shift, UP,
            labels[-2].shift, UP * 2 + RIGHT * 1.5,
            run_time=2
        )
        self.wait()

        self.clear_all_updaters(A, B, C, A_dot, A_label, AB, AC, regions, all_dashed_lines, labels)
        self.highlight_region_AC(A, C, all_dashed_lines, labels)

        d_arrow = original_d_arrow.copy()
        self.play(
            Write(d_arrow)
        )
        self.wait()
        self.play(
            Write(line_to_pass),
            O_label.shift, RIGHT * 0.7,
        )
        self.wait()

        self.add_all_updaters(A, B, C, A_dot, A_label, AB, AC, regions, all_dashed_lines, labels)

        self.play(
            A_dot.move_to, LEFT * 2 + UP * 3,
            labels[1].shift, LEFT * 4.5,
            labels[-1].shift, UP + LEFT * 2.7,
            labels[-2].shift, UP * 2.5 + LEFT * 3,
            run_time=2
        )
        self.clear_all_updaters(A, B, C, A_dot, A_label, AB, AC, regions, all_dashed_lines, labels)
        self.wait()

        n1 = self.get_normal_towards_origin(C, A_dot.get_center())
        n1 = normalize(n1)
        new_d_arrow = Arrow((C + A_dot.get_center()) / 2, (C + A_dot.get_center()) / 2 + n1 * 1, buff=0).set_color(YELLOW)
        new_d_arrow.shift(normalize(A_dot.get_center() - C) * 0.7)

        self.play(
            Transform(d_arrow, new_d_arrow),
            FadeOut(line_to_pass),
        )
        AC_perp_label = TexMobject(r"\vec{AC}_{\perp}")
        AC_perp_label.move_to(RIGHT * 0.5 + UP * 1.5)
        self.play(
            Write(AC_perp_label)
        )
        self.wait()

        AC_perp_calc = TexMobject(r"\vec{AC}_{\perp} = (\vec{AB} \times \vec{AC}) \times \vec{AC}").scale(0.9)
        AC_perp_calc.add_background_rectangle()
        AC_perp_calc.to_edge(LEFT).shift(DOWN * 1.5)
        self.play(
            Write(AC_perp_calc)
        )

        self.wait()

        AO_arrow = Arrow(A_dot.get_center(), ORIGIN, buff=0).set_color(FUCHSIA)
        self.play(
            d_arrow.put_start_and_end_on, 
            A_dot.get_center(), 
            A_dot.get_center() + (d_arrow.get_end() - d_arrow.get_start()),
            AC_perp_label.move_to, UP * 3.2 + LEFT * 0.5,
        )

        self.wait()

        self.play(
            Write(AO_arrow),
        )

        self.wait()

        condition = TexMobject(r"\vec{AC}_{\perp} \boldsymbol{\cdot} \vec{AO} > 0 \rightarrow O \in R_{AC}").scale(0.9)
        condition.add_background_rectangle()
        condition.shift(DOWN * 2.5).to_edge(LEFT)
        self.play(
            Write(condition)
        )

        self.wait()

        cross_B = Cross(B_dot).scale(1.5).set_color(CROSS_RED)
        self.add_foreground_mobject(cross_B)

        next_step = TexMobject(r"\vec{d} \leftarrow \vec{AC}_{\perp}, \text{remove } B \text{ from simplex}").scale(0.9)
        next_step.add_background_rectangle()
        next_step.move_to(DOWN * 3.5).to_edge(LEFT)
        self.play(
            Write(next_step),
            Write(cross_B),
        )

        self.wait(3)

        self.play(
            FadeOut(condition),
            FadeOut(AO_arrow),
            FadeOut(AC_perp_label),
            FadeOut(AC_perp_calc),
            FadeOut(d_arrow),
            FadeOut(next_step),
            FadeOut(cross_B)
        )

        self.wait()

        self.add_all_updaters(A, B, C, A_dot, A_label, AB, AC, regions, all_dashed_lines, labels)

        self.play(
            A_dot.move_to, A,
            labels[1].shift, RIGHT * 4.5 + UP * 0.5,
            labels[-1].shift, DOWN + RIGHT * 2.7,
            labels[-2].shift, DOWN * 2.5 + RIGHT * 3,
            run_time=2
        )

        self.wait()

        self.clear_all_updaters(A, B, C, A_dot, A_label, AB, AC, regions, all_dashed_lines, labels)

        n_AB = self.get_normal_towards_origin(A, B)
        n_AB = normalize(n_AB)

        ab_perp_arrow = Arrow((A + B) / 2, (A + B) / 2 - n_AB * 1, buff=0).set_color(YELLOW)
        AB_perp_label.next_to(ab_perp_arrow, RIGHT)
        self.play(
            Write(ab_perp_arrow),
            Write(AB_perp_label),
        )
        self.wait()

        n_AC = self.get_normal_towards_origin(A, C)
        n_AC = normalize(n_AC)

        ac_perp_arrow = Arrow((A + C) / 2, (A + C) / 2 - n_AC * 1, buff=0).set_color(YELLOW)
        AC_perp_label.move_to(ac_perp_arrow.get_center()).shift(UR * 0.5)
        self.play(
            Write(ac_perp_arrow),
            Write(AC_perp_label),
        )

        self.wait()

        self.play(
            ab_perp_arrow.put_start_and_end_on, A, A - n_AB,
            ac_perp_arrow.put_start_and_end_on, A, A - n_AC,
            AB_perp_label.move_to, A - n_AB * 1.4 + DR * 0.5 + RIGHT * 0.1 + UP * 0.3,
            AC_perp_label.move_to, A - n_AC * 1.4 + UP * 0.5 + RIGHT * 0.2,
            A_label.move_to, A_dot.get_center() + UP * 0.3 + RIGHT * 0.4
        )
        self.wait()

        AO_arrow = Arrow(A_dot.get_center(), ORIGIN, buff=0).set_color(FUCHSIA)

        O_label.move_to(LEFT * 0.4 + UP * 0.4)
        self.play(
            Write(O_label)
        )

        self.play(
            Write(AO_arrow)
        )

        self.wait()
        triangle_condition_AB = TexMobject(r"\vec{AB}_{\perp} \boldsymbol{\cdot} \vec{AO} < 0 \rightarrow O \notin R_{AB}").scale(0.9)
        triangle_condition_AB.add_background_rectangle()
        triangle_condition_AC = TexMobject(r"\vec{AC}_{\perp} \boldsymbol{\cdot} \vec{AO} < 0 \rightarrow O \notin R_{AC}").scale(0.9)
        triangle_condition_AC.add_background_rectangle()
        triangle_condition_true = TextMobject(r"This implies $O \in R_{ABC}$").scale(0.9)
        triangle_condition_true.add_background_rectangle()

        triangle_condition_AB.to_edge(LEFT).shift(DOWN * 1.5)
        triangle_condition_AC.to_edge(LEFT).shift(DOWN * 2.5)
        triangle_condition_true.to_edge(LEFT).shift(DOWN * 3.5)
        self.play(
            Write(triangle_condition_AB),
            Write(triangle_condition_AC),
        )
        self.wait()

        self.play(
            Write(triangle_condition_true)
        )
        self.wait()

        self.play(
            FadeOut(triangle_condition_true),
            FadeOut(triangle_condition_AC),
            FadeOut(triangle_condition_AB),
            FadeOut(AO_arrow),
            FadeOut(AB_perp_label),
            FadeOut(AC_perp_label),
            FadeOut(ab_perp_arrow),
            FadeOut(ac_perp_arrow),
        )
        self.wait()

        key_idea = TextMobject(
            r"Only max 2 regions ($R_{AB}$ and $R_{AC}$)" + "\\\\",
            "need to be checked to locate origin"
        ).scale(0.8)
        key_idea.add_background_rectangle()
        key_idea.to_edge(LEFT)
        key_idea.shift(DOWN * 2.5)

        self.play(
            Write(key_idea)
        )

        self.wait()

        n_AB = self.get_normal_towards_origin(A, B)
        n_AB = normalize(n_AB)

        ab_perp_arrow = Arrow((A + B) / 2, (A + B) / 2 - n_AB * 1, buff=0).set_color(YELLOW)
        AB_perp_label.next_to(ab_perp_arrow, RIGHT)
        self.play(
            Write(ab_perp_arrow),
            Write(AB_perp_label),
        )
        self.wait()

        n_AC = self.get_normal_towards_origin(A, C)
        n_AC = normalize(n_AC)

        ac_perp_arrow = Arrow((A + C) / 2, (A + C) / 2 - n_AC * 1, buff=0).set_color(YELLOW)
        AC_perp_label.move_to(ac_perp_arrow.get_center()).shift(UR * 0.5)
        self.play(
            Write(ac_perp_arrow),
            Write(AC_perp_label),
        )

        self.wait()

        region_ABC_outline = Polygon(C, B, A).set_stroke(color=YELLOW, width=5)
        self.play(
            ShowCreationThenDestruction(region_ABC_outline),
            Indicate(labels[-1]),
        )
        self.wait()

    def add_all_updaters(self, A, B, C, A_dot, A_label, AB, AC, regions, dashed_lines, labels):
        FACTOR = 16
        region_A = regions[0]
        region_AB = regions[1]
        region_B = regions[2]
        region_C = regions[4]
        region_AC = regions[5]
        region_ABC = regions[6]

        AB_v, AC_v = B - A_dot.get_center(), C - A_dot.get_center()
        ABperp = normalize(self.tripleProd(AC_v, AB_v, AB_v))
        ACperp = normalize(self.tripleProd(AB_v, AC_v, AC_v))
        always(A_label.next_to, A_dot, DOWN)
        DASH_LENGTH = 0.1

        # Might be a better way to do this, but with lambdas it's tricky (can't loop through)
        

        dashed_lines[0].add_updater(lambda m: m.become(
                DashedLine(
                A_dot.get_center(), 
                A_dot.get_center() + normalize(
                    tripleProd(B - A_dot.get_center(), C - A_dot.get_center(), C - A_dot.get_center())
                    ) * FACTOR,
                dash_length=DASH_LENGTH
                )
            )
        )

        dashed_lines[1].add_updater(lambda m: m.become(
                DashedLine(
                A_dot.get_center(), 
                A_dot.get_center() + normalize(
                    tripleProd(C - A_dot.get_center(), B - A_dot.get_center(), B - A_dot.get_center())
                    ) * FACTOR,
                dash_length=DASH_LENGTH
                )
            )
        )

        dashed_lines[2].add_updater(lambda m: m.become(
                DashedLine(
                B, 
                B + normalize(
                    tripleProd(C - A_dot.get_center(), B - A_dot.get_center(), B - A_dot.get_center())
                    ) * FACTOR,
                dash_length=DASH_LENGTH
                )
            )
        )

        dashed_lines[5].add_updater(lambda m: m.become(
                DashedLine(
                C, 
                C + normalize(
                    tripleProd(B - A_dot.get_center(), C - A_dot.get_center(), C - A_dot.get_center())
                    ) * FACTOR,
                dash_length=DASH_LENGTH
                )
            )
        )

        always_redraw(
            lambda: AB.put_start_and_end_on(
                A_dot.get_center(), B
            )
        )
        always_redraw(
            lambda: AC.put_start_and_end_on(
                A_dot.get_center(), C
            )
        )

        region_ABC.add_updater(
            lambda m: m.become(
                Polygon(C, B, A_dot.get_center()).set_stroke(width=0).set_fill(
                    color=region_ABC.get_color(), opacity=0.22
                )
            )
        )

        region_AB.add_updater(
            lambda m: m.become(
                Polygon(
                    B, dashed_lines[2].get_end(), dashed_lines[1].get_end(), A_dot.get_start()
                ).set_stroke(width=0).set_fill(color=region_AB.get_color(), opacity=0.3)
            )
        )

        region_A.add_updater(
            lambda m: m.become(
                Polygon(
                    A_dot.get_center(), dashed_lines[0].get_end(), dashed_lines[1].get_end(),
                ).set_stroke(width=0).set_fill(color=region_A.get_color(), opacity=0.3)
            )
        )

        region_B.add_updater(
            lambda m: m.become(
                Polygon(
                    B, dashed_lines[2].get_end(), dashed_lines[3].get_end(),
                ).set_stroke(width=0).set_fill(color=region_B.get_color(), opacity=0.3)
            )
        )

        region_C.add_updater(
            lambda m: m.become(
                Polygon(
                    C, dashed_lines[4].get_end(), dashed_lines[5].get_end(),
                ).set_stroke(width=0).set_fill(color=region_C.get_color(), opacity=0.3)
            )
        )

        region_AC.add_updater(
            lambda m: m.become(
                Polygon(
                    A_dot.get_center(), dashed_lines[0].get_end(), dashed_lines[5].get_end(), C,
                ).set_stroke(width=0).set_fill(color=region_AC.get_color(), opacity=0.3)
            )
        )
    
    def clear_all_updaters(self, A, B, C, A_dot, A_label, AB, AC, regions, dashed_lines, labels):
        region_A = regions[0]
        region_AB = regions[1]
        region_B = regions[2]
        region_C = regions[4]
        region_AC = regions[5]
        region_ABC = regions[6]

        A_label.clear_updaters()

        dashed_lines[0].clear_updaters()
        dashed_lines[1].clear_updaters()
        dashed_lines[2].clear_updaters()
        dashed_lines[5].clear_updaters()

        AB.clear_updaters()
        AC.clear_updaters()

        region_ABC.clear_updaters()
        region_AB.clear_updaters()
        region_A.clear_updaters()
        region_B.clear_updaters()
        region_C.clear_updaters()
        region_AC.clear_updaters()

    def get_direction_arrow_BCperp(self, A_dot, B_dot, C_dot):
        A, B, C = A_dot.get_center(), B_dot.get_center(), C_dot.get_center()
        n1 = self.get_normal_towards_origin(B, C)
        n1 = normalize(n1)
        d_arrow = Arrow((B + C) / 2, (B + C) / 2 + n1 * 1.2, buff=0).set_color(YELLOW)
        return d_arrow

    def retrace_initial_steps(self, A_dot, B_dot, C_dot, regions, labels, dashed_lines):
        A, B, C = A_dot.get_center(), B_dot.get_center(), C_dot.get_center()
        d_arrow = Arrow(C, C + normalize(ORIGIN - C) * 1.2).set_color(YELLOW)

        R_C, region_C = labels[4], regions[4]
        d1, d2 = normalize(dashed_lines[5].get_end() - C), normalize(dashed_lines[4].get_end() - C)
        region_C_outline = [C + d2 * 2, C + d1*2, C]
        region_C_outline = Polygon(*region_C_outline).set_stroke(color=YELLOW, width=5)
    
        self.play(
            Indicate(R_C),
            ShowCreationThenDestruction(region_C_outline),
        )
        self.wait()

        self.play(
            Write(d_arrow)
        )
        self.wait()

        self.play(
            FadeOut(region_C),
            FadeOut(R_C)
        )
        self.wait()

        R_B, region_B = labels[2], regions[2]
        d1, d2 = normalize(dashed_lines[3].get_end() - B), normalize(dashed_lines[2].get_end() - B)
        region_B_outline = [B + d1 * 8, B, B + d2 * 8]
        region_B_outline = VGroup().set_points_as_corners(*[region_B_outline]).set_stroke(color=YELLOW, width=5)

        self.play(
            Indicate(R_B),
            ShowCreationThenDestruction(region_B_outline),
        )
        self.wait()

        O_dot = Dot().set_color(YELLOW)
        O = TexMobject("O").shift(UP * 0.3 + LEFT * 0.3)

        self.play(
            Write(O),
            GrowFromCenter(O_dot),
        )
        self.wait()

        self.play(
            O_dot.shift, LEFT * 5 + UP * 2,
            O.shift, LEFT * 5 + UP * 2,
            run_time=2
        )
        self.wait()

        self.play(
            Indicate(B_dot, color=CROSS_RED),
        )
        self.play(
            Indicate(B_dot, color=CROSS_RED),
        )
        self.wait()

        self.play(
            FadeOut(R_B),
            FadeOut(region_B),
        )
        self.wait()

        new_d_arrow = self.get_direction_arrow_BCperp(A_dot, B_dot, C_dot)

        R_A, region_A = labels[0], regions[0]
        self.play(
            Transform(d_arrow, new_d_arrow),
            O_dot.move_to, UP * 3 + RIGHT * 6,
            O.move_to, UP * 3.3 + RIGHT * 5.7,
            run_time=2,
        )
        self.wait()

        self.play(
            Indicate(A_dot, color=CROSS_RED),
        )
        self.play(
            Indicate(A_dot, color=CROSS_RED),
        )

        self.wait()

        self.play(
            FadeOut(region_A),
            FadeOut(R_A),
            FadeOut(O_dot),
            FadeOut(O),
        )

        self.wait()
        

        R_BC, region_BC = labels[3], regions[3]
        d1, d2 = normalize(dashed_lines[3].get_end() - B), normalize(dashed_lines[4].get_end() - C)
        region_BC_outline = VGroup().set_points_as_corners(*[[B + d1 * 7, B, C, C + d2 * 1.5]])
        region_BC_outline.set_stroke(color=YELLOW, width=5)

        self.play(
            Indicate(R_BC),
            ShowCreationThenDestruction(region_BC_outline)
        )
        self.wait()

        self.play(
            Indicate(d_arrow)
        )

        self.play(
            Indicate(d_arrow)
        )
        self.wait()

        self.play(
            FadeOut(R_BC),
            FadeOut(region_BC),
            FadeOut(dashed_lines[3]),
            FadeOut(dashed_lines[4]),
        )

        self.wait()


        R_AB, region_AB = labels[1], regions[1]
        d1, d2 = normalize(dashed_lines[2].get_end() - B), normalize(dashed_lines[1].get_end() - A)
        region_AB_outline = VGroup().set_points_as_corners(*[[B + d1 * 5, B, A, A + d2 * 4]])
        region_AB_outline.set_stroke(color=YELLOW, width=5)
        self.play(
            Indicate(R_AB),
            ShowCreationThenDestruction(region_AB_outline),
        )
        self.wait()

        BC_vector = B - C
        start_line_to_pass = BC_vector
        end_line_to_pass = -BC_vector
        line_to_pass = DashedLine(start_line_to_pass, end_line_to_pass)
        line_to_pass.set_color(YELLOW)

        self.play(
            Write(line_to_pass)
        )
        self.wait()

        return d_arrow, line_to_pass

    def highlight_region_AC(self, A, C, dashed_lines, labels):
        R_AC, BC_cont, BA_cont = labels[5], dashed_lines[5], dashed_lines[0]
        region_AC_outline = VGroup()
        BC_cont_normal = normalize(BC_cont.get_end() - C)
        BA_cont_normal = normalize(BA_cont.get_end() - A)
        region_AC_outline.set_points_as_corners(*[[C + BC_cont_normal * 1.5, C, A, A + BA_cont_normal * 3]])
        region_AC_outline.set_stroke(color=YELLOW, width=5)
        self.play(
            Indicate(R_AC),
            ShowCreationThenDestruction(region_AC_outline)
        )

        self.wait()

    def highlight_region_A(self, A, dashed_lines, labels):
        R_A, CA_cont, BA_cont = labels[0], dashed_lines[1], dashed_lines[0]
        region_A_outline = VGroup()
        CA_cont_normal = normalize(CA_cont.get_end() - A)
        BA_cont_normal = normalize(BA_cont.get_end() - A)
        region_A_outline.set_points_as_corners(*[[A + BA_cont_normal * 3, A,  A + CA_cont_normal * 3]])
        region_A_outline.set_stroke(color=YELLOW, width=5)
        self.play(
            Indicate(R_A),
            ShowCreationThenDestruction(region_A_outline)
        )

        self.wait()

    def get_regions(self, A, B, C):
        AC = normalize(C - A)
        AB = normalize(B - A)
        BC = normalize(C - B)

        ABperp = normalize(self.tripleProd(AC, AB, AB))
        ACperp = normalize(self.tripleProd(AB, AC, AC))
        BCperp = normalize(self.tripleProd(-AB, BC, BC))

        FACTOR = 11
        DASH_LENGTH = 0.1

        CA_normal = DashedLine(A, A + ACperp * FACTOR, dash_length=DASH_LENGTH)
        BA_normal = DashedLine(A, A + ABperp * FACTOR, dash_length=DASH_LENGTH)

        AB_normal = DashedLine(B, B + ABperp * FACTOR, dash_length=DASH_LENGTH)
        CB_normal = DashedLine(B, B + BCperp * FACTOR, dash_length=DASH_LENGTH)

        BC_normal = DashedLine(C, C + BCperp * FACTOR, dash_length=DASH_LENGTH)
        AC_normal = DashedLine(C, C + ACperp * FACTOR, dash_length=DASH_LENGTH)

        all_dashed_lines = VGroup(
            CA_normal, BA_normal, AB_normal,
            CB_normal, BC_normal, AC_normal
        )

        regions = []
        colors = [BLUE_E, VIOLET, PERIWINKLE_BLUE, COBALT_BLUE, PURPLE, SEAFOAM_GREEN]
        for i in range(len(all_dashed_lines)):
            lines = [all_dashed_lines[i], all_dashed_lines[(i + 1) % len(all_dashed_lines)]]
            region = self.get_voronoi_region(lines, fill_color=colors[i])
            regions.append(region)
        triangle_region = Polygon(A, B, C).set_stroke(width=0).set_fill(color=self.triangle_color, opacity=0.4)
        regions.append(triangle_region)
        return all_dashed_lines, VGroup(*regions)

    def get_voronoi_region(self, lines, fill_color):
        p1, p2 = lines[0].get_start(), lines[1].get_start()
        if np.allclose(p1, p2):
            polygon = Polygon(p1, lines[0].get_end(), lines[1].get_end())
        else:
            polygon = Polygon(p1, p2, lines[1].get_end(), lines[0].get_end())
        polygon.set_stroke(width=0)
        polygon.set_fill(color=fill_color, opacity=0.5)
        return polygon

    def show_region_labels(self):
        r_A = TexMobject("R_A").move_to(RIGHT * 6 + UP * 2.5)
        r_AB = TexMobject("R_{AB}").move_to(RIGHT * 0.5 + UP * 2.5)
        r_B = TexMobject("R_B").move_to(LEFT * 5.5 + UP * 1.5)
        r_BC = TexMobject("R_{BC}").move_to(LEFT * 2 + DOWN * 1.5)
        r_C = TexMobject("R_C").move_to(RIGHT * 1.1 + DOWN * 3.6)
        r_AC = TexMobject("R_{AC}").move_to(RIGHT * 4.5 + DOWN * 1.5)
        r_ABC = TexMobject("R_{ABC}").move_to(RIGHT * 0.7 + DOWN * 0.5)
        return VGroup(*[r_A, r_AB, r_B, r_BC, r_C, r_AC, r_ABC])

class AcknowledgeDifficulty(Scene):
    def construct(self):
        rect = ScreenRectangle(height=6).move_to(UP * 0.5)
        self.add(rect)
        
        tricky = TextMobject("The line and triangle cases are tricky.")
        tricky.next_to(rect, DOWN)
        take_your_time = TextMobject("Take your time with the concepts.")
        take_your_time.move_to(tricky.get_center())
        implementation = TextMobject("Seeing implementation will help!")
        implementation.move_to(take_your_time.get_center())
        self.play(
            Write(tricky)
        )

        self.wait()

        self.play(
            ReplacementTransform(tricky, take_your_time)
        )

        self.wait()

        self.play(
            ReplacementTransform(take_your_time, implementation)
        )

        self.wait()


class CompactGJK(GJKUtils):
    """
    This is a good reference function for a simple implementation of 
    the GJK algorithm as presented in the video. 
    """
    def construct(self):
        p1 = RegularPolygon(n=6).scale(2).shift(LEFT * 1.5)
        p2 = RegularPolygon(n=5).scale(2).shift(RIGHT * 1.5)
        if self.gjk(p1, p2):
            p1.set_color(GREEN_SCREEN)
            p2.set_color(GREEN_SCREEN)
        else:
            p1.set_color(CROSS_RED)
            p2.set_color(CROSS_RED)
        self.add(p1, p2)
        self.wait()

        for _ in range(30):
            rv1 = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
            rv2 = [np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
            random_point1 = rv1[0] * FRAME_X_RADIUS * X_AXIS + rv1[1] * FRAME_Y_RADIUS * Y_AXIS
            random_point2 = rv2[0] * FRAME_X_RADIUS * X_AXIS + rv2[1] * FRAME_Y_RADIUS * Y_AXIS
            p1.move_to(random_point1)
            p2.move_to(random_point2)
            if self.gjk(p1, p2):
                p1.set_color(GREEN_SCREEN)
                p2.set_color(GREEN_SCREEN)
            else:
                p1.set_color(CROSS_RED)
                p2.set_color(CROSS_RED)
            self.wait()

    def gjk(self, p1, p2):
        d = normalize(p2.get_center() - p1.get_center())
        simplex = [self.support(p1, p2, d)]
        d = ORIGIN - simplex[0]
        while True:
            A = self.support(p1, p2, d)
            if np.dot(A, d) < 0:
                return False
            simplex.append(A)
            if self.update_simplex(simplex, d):
                return True
    
    def update_simplex(self, simplex, d):
        if len(simplex) == 2:
            return self.lineCase(simplex, d)
        return self.triangleCase(simplex, d)

    def lineCase(self, simplex, d):
        B, A = simplex
        AB, AO = B - A, ORIGIN - A
        abPerp = normalize(self.tripleProd(AB, AO, AB))
        set_array(d, abPerp)
        return False

    def triangleCase(self, simplex, d):
        C, B, A = simplex
        AB, AC, AO = B - A, C - A, ORIGIN - A
        abPerp = normalize(self.tripleProd(AC, AB, AB))
        acPerp = normalize(self.tripleProd(AB, AC, AC))
        if np.dot(abPerp, AO) > 0: # region AB
            remove_array(simplex, C)
            set_array(d, abPerp)
        elif np.dot(acPerp, AO) > 0: # region AC
            remove_array(simplex, B)
            set_array(d, acPerp)
        else:
            return True
        return False

class GJKImplementation(Scene):
    def construct(self):
        main_loop_code = self.get_main_loop()
        update_simplex_code = self.update_simplex_code()
        indent = LEFT * 1

        left_code = VGroup(main_loop_code, update_simplex_code).arrange(DOWN)
        left_code[0].to_edge(indent)
        left_code[1].to_edge(indent)

        support_function_code = self.get_support_function()
        support_function_code.next_to(left_code[0], DOWN * 0.7)
        support_function_code.to_edge(LEFT)

        for i in range(5):
            self.play(
                Write(main_loop_code[i])
            )
            self.wait()

        rect = ScreenRectangle(height=3.5).shift(RIGHT * 3.3)
        self.play(
            ShowCreation(rect)
        )

        for i in range(len(support_function_code)):
            self.play(
                Write(support_function_code[i])
            )
            self.wait()

        for i in range(5, len(main_loop_code)):
            self.play(
                Write(main_loop_code[i])
            )
            self.wait()

        self.play(
            FadeOut(support_function_code)
        )

        self.wait()

        for i in range(len(update_simplex_code)):
            self.play(
                Write(update_simplex_code[i])
            )
            self.wait()

        self.play(
            FadeOut(left_code[0]),
            update_simplex_code.shift, UP * 5.8,
            run_time=2,
        )
        self.wait()

        line_case = self.line_case_code()
        line_case.next_to(update_simplex_code, DOWN * 0.7)
        line_case.to_edge(indent)
        for i in range(len(line_case)):
            self.play(
                Write(line_case[i])
            )
            self.wait()

        note = TextMobject(
            "*A full implementation should handle edge cases" + "\\\\",
            "where the origin lies on an edge. This applies to both" + "\\\\",
            "the line and triangle updates. Not too hard to add."
        ).scale(0.6).next_to(rect, DOWN)
        self.add(note)
        self.wait()
        line_case_scaled_down = line_case.copy().scale(0.9)
        line_case_scaled_down.shift(UP * 2).to_edge(indent)

        self.play(
            Transform(line_case, line_case_scaled_down),
            FadeOut(update_simplex_code),
            run_time=2
        )
        self.wait()

        triangle_case = self.triangle_case_code()
        triangle_case.scale(0.9)
        triangle_case.next_to(line_case, DOWN * 0.7)
        triangle_case.to_edge(indent)

        for i in range(len(triangle_case)):
            self.play(
                Write(triangle_case[i])
            )
            self.wait()

        right_code = VGroup(line_case.copy(), triangle_case.copy())

        VGroup(left_code, right_code).arrange(RIGHT)
        right_code.shift(LEFT * SMALL_BUFF)
        left_code[1].shift(DOWN * 5.8)
        self.play(
            Transform(line_case, right_code[0]),
            Transform(triangle_case,  right_code[1]),
            FadeOut(rect),
            FadeOut(note),
            run_time=2       
        )
        left_code.shift(UP * 1.1 + RIGHT * 0.1).scale(0.9)
        self.play(
            FadeIn(left_code)
        )

        self.wait()

    def get_main_loop(self):
        code_scale = 0.8
        
        code = []

        def_statement = TextMobject(r"$\text{def GJK}(s1, s2):$")
        def_statement[0][:3].set_color(MONOKAI_BLUE)
        def_statement[0][3:6].set_color(MONOKAI_GREEN)
        def_statement[0][7:9].set_color(MONOKAI_ORANGE)
        def_statement[0][10:12].set_color(MONOKAI_ORANGE)
        def_statement.scale(code_scale)
        def_statement.to_edge(LEFT)
        code.append(def_statement)

        comment1 = TextMobject(r"\# True if shapes s1 and s2 intersect")        
        comment1.scale(code_scale)
        comment1.next_to(def_statement, DOWN * 0.5)
        comment1.to_edge(LEFT * 2)
        comment1.set_color(MONOKAI_GRAY)
        code.append(comment1)

        comment2 = TextMobject(r"\# All vectors/points are ``3D`` $([x, y, 0])$")        
        comment2.scale(code_scale)
        comment2.next_to(comment1, DOWN * 0.5)
        comment2.to_edge(LEFT * 2)
        comment2.set_color(MONOKAI_GRAY)
        code.append(comment2)

        line_1 = TextMobject(r"$d =$ normalize(s2.center $-$ s1.center)")
        line_1.scale(code_scale)
        line_1.next_to(comment2, DOWN * 0.5)
        line_1.to_edge(LEFT * 2)
        line_1[0][1].set_color(MONOKAI_PINK)
        line_1[0][2:11].set_color(MONOKAI_BLUE)
        line_1[0][21].set_color(MONOKAI_PINK)
        code.append(line_1)

        line_2 = TextMobject(r"simplex $= \lbrack$support(s1, s2, $d$)$\rbrack$")
        line_2.scale(code_scale)
        line_2.next_to(line_1, DOWN * 0.5)
        line_2.to_edge(LEFT * 2)
        line_2[0][7].set_color(MONOKAI_PINK)
        line_2[0][9:16].set_color(MONOKAI_BLUE)
        code.append(line_2)

        line_3 = TextMobject(r"$d =$ ORIGIN $-$ simplex$[0]$")
        line_3.scale(code_scale)
        line_3.next_to(line_2, DOWN * 0.5)
        line_3.to_edge(LEFT * 2)
        line_3[0][1].set_color(MONOKAI_PINK)
        line_3[0][8].set_color(MONOKAI_PINK)
        line_3[0][-2].set_color(MONOKAI_PURPLE)
        code.append(line_3)

        line_4 = TextMobject(r"while True:")
        line_4.scale(code_scale)
        line_4.next_to(line_3, DOWN * 0.5)
        line_4.to_edge(LEFT * 2)
        line_4[0][:5].set_color(MONOKAI_PINK)
        line_4[0][5:9].set_color(MONOKAI_PURPLE)
        code.append(line_4)

        line_5 = TextMobject(r"$A =$ support(s1, s2, $d$)")
        line_5[0][1].set_color(MONOKAI_PINK)
        line_5[0][2:9].set_color(MONOKAI_BLUE)
        line_5.scale(code_scale)
        line_5.next_to(line_4, DOWN * 0.5)
        line_5.to_edge(LEFT * 3)
        code.append(line_5)

        line_6 = TextMobject(r"if dot($A, d$) $<$ 0:")
        line_6[0][:2].set_color(MONOKAI_PINK)
        line_6[0][2:5].set_color(MONOKAI_BLUE)
        line_6[0][-3].set_color(MONOKAI_PINK)
        line_6[0][-2].set_color(MONOKAI_PURPLE)
        line_6.scale(code_scale)
        line_6.next_to(line_5, DOWN * 0.5)
        line_6.to_edge(LEFT * 3)
        code.append(line_6)

        line_7 = TextMobject(r"return False")
        line_7[0][:6].set_color(MONOKAI_PINK)
        line_7[0][6:].set_color(MONOKAI_PURPLE)
        line_7.scale(code_scale)
        line_7.next_to(line_6, DOWN * 0.5)
        line_7.to_edge(LEFT * 4)
        code.append(line_7)

        line_8 = TextMobject(r"simplex.append($A$)")
        line_8[0][8:-3].set_color(MONOKAI_BLUE)
        line_8.scale(code_scale)
        line_8.next_to(line_7, DOWN * 0.5)
        line_8.to_edge(LEFT * 3)
        code.append(line_8)

        line_9 = TextMobject(r"if handleSimplex(simplex, $d$):")
        line_9[0][:2].set_color(MONOKAI_PINK)
        line_9[0][2:15].set_color(MONOKAI_BLUE)
        line_9.scale(code_scale)
        line_9.next_to(line_8, DOWN * 0.5)
        line_9.to_edge(LEFT * 3)
        code.append(line_9)

        line_10 = TextMobject(r"return True")
        line_10[0][:6].set_color(MONOKAI_PINK)
        line_10[0][6:].set_color(MONOKAI_PURPLE)
        line_10.scale(code_scale)
        line_10.next_to(line_9, DOWN * 0.5)
        line_10.to_edge(LEFT * 4)
        code.append(line_10)

        code = VGroup(*code)
        code.scale(0.9)
        code.to_edge(LEFT * 3)
        return code 

    def get_support_function(self):
        code_scale = 0.8
        
        code = []

        def_statement = TextMobject(r"def support($s1, s2, d$):")
        def_statement[0][:3].set_color(MONOKAI_BLUE)
        def_statement[0][3:10].set_color(MONOKAI_GREEN)
        def_statement[0][11:13].set_color(MONOKAI_ORANGE)
        def_statement[0][14:16].set_color(MONOKAI_ORANGE)
        def_statement[0][-3].set_color(MONOKAI_ORANGE)
        def_statement.scale(code_scale)
        def_statement.to_edge(LEFT)
        code.append(def_statement)

        line_1 = TextMobject(r"return s1.furthestPoint($d$) $-$ s2.furthestPoint($-d$)")
        line_1[0][:6].set_color(MONOKAI_PINK)
        line_1[0][9:22].set_color(MONOKAI_BLUE)
        line_1[0][25].set_color(MONOKAI_PINK)
        line_1[0][29:-4].set_color(MONOKAI_BLUE)
        line_1[0][-3].scale(0.9).shift(LEFT * SMALL_BUFF * 0.5)
        line_1[0][-3].set_color(MONOKAI_PINK)
        line_1[0][-2:].shift(LEFT * SMALL_BUFF * 0.5)
        line_1.scale(code_scale)
        line_1.next_to(def_statement, DOWN * 0.5)
        line_1.to_edge(LEFT * 2)
        code.append(line_1)

        code = VGroup(*code)
        code.scale(0.9)
        code.to_edge(LEFT * 3)
        return code

    def update_simplex_code(self):
        code_scale = 0.8
        
        code = []

        def_statement = TextMobject(r"def handleSimplex(simplex, $d$):")
        def_statement[0][:3].set_color(MONOKAI_BLUE)
        def_statement[0][3:16].set_color(MONOKAI_GREEN)
        def_statement[0][17:24].set_color(MONOKAI_ORANGE)
        def_statement[0][25].set_color(MONOKAI_ORANGE)
        def_statement.scale(code_scale)
        def_statement.to_edge(LEFT)
        code.append(def_statement)

        line_1 = TextMobject(r"if len(simplex) $== 2$:")
        line_1[0][:2].set_color(MONOKAI_PINK)
        line_1[0][2:5].set_color(MONOKAI_BLUE)
        line_1[0][-4:-2].set_color(MONOKAI_PINK)
        line_1[0][-2].set_color(MONOKAI_PURPLE)
        line_1.scale(code_scale)
        line_1.next_to(def_statement, DOWN * 0.5)
        line_1.to_edge(LEFT * 2)
        code.append(line_1)

        line_2 = TextMobject(r"return lineCase(simplex, $d$)")
        line_2.scale(code_scale)
        line_2.next_to(line_1, DOWN * 0.5)
        line_2.to_edge(LEFT * 3)
        line_2[0][:6].set_color(MONOKAI_PINK)
        line_2[0][6:14].set_color(MONOKAI_BLUE)
        code.append(line_2)

        line_3 = TextMobject(r"return triangleCase(simplex, $d$)")
        line_3.scale(code_scale)
        line_3.next_to(line_2, DOWN * 0.5)
        line_3.to_edge(LEFT * 2)
        line_3[0][:6].set_color(MONOKAI_PINK)
        line_3[0][6:18].set_color(MONOKAI_BLUE)
        code.append(line_3)

        code = VGroup(*code)
        code.scale(0.9)
        code.to_edge(LEFT * 3)
        return code

    def line_case_code(self):
        code_scale = 0.8 
        code = []
        def_statement = TextMobject(r"def lineCase(simplex, $d$):")
        def_statement[0][:3].set_color(MONOKAI_BLUE)
        def_statement[0][3:11].set_color(MONOKAI_GREEN)
        def_statement[0][12:19].set_color(MONOKAI_ORANGE)
        def_statement[0][-3].set_color(MONOKAI_ORANGE)
        def_statement.scale(code_scale)
        def_statement.to_edge(LEFT)
        code.append(def_statement)

        line_1 = TextMobject(r"$B, A =$ simplex")
        line_1[0][3].set_color(MONOKAI_PINK)
        line_1.scale(code_scale)
        line_1.next_to(def_statement, DOWN * 0.5)
        line_1.to_edge(LEFT * 2)
        code.append(line_1)

        line_2 = TextMobject(r"$AB, AO = B - A, \text{ORIGIN} - A$")
        line_2[0][5].set_color(MONOKAI_PINK)
        line_2[0][7].set_color(MONOKAI_PINK)
        line_2[0][-2].set_color(MONOKAI_PINK)
        line_2.scale(code_scale)
        line_2.next_to(line_1, DOWN * 0.5)
        line_2.to_edge(LEFT * 2)
        code.append(line_2)

        line_3 = TextMobject(r"ABperp $=$ tripleProd($AB, AO, AB$)")
        line_3[0][6].set_color(MONOKAI_PINK)
        line_3[0][7:17].set_color(MONOKAI_BLUE)
        line_3.scale(code_scale)
        line_3.next_to(line_2, DOWN * 0.5)
        line_3.to_edge(LEFT * 2)
        code.append(line_3)

        line_4 = TextMobject(r"$d$.set(ABperp)")
        line_4.scale(code_scale)
        line_4.next_to(line_3, DOWN * 0.5)
        line_4.to_edge(LEFT * 2)
        line_4[0][2:5].set_color(MONOKAI_BLUE)
        code.append(line_4)

        line_5 = TextMobject(r"return False")
        line_5.scale(code_scale)
        line_5.next_to(line_4, DOWN * 0.5)
        line_5.to_edge(LEFT * 2)
        line_5[0][:6].set_color(MONOKAI_PINK)
        line_5[0][6:].set_color(MONOKAI_PURPLE)
        code.append(line_5)

        code = VGroup(*code)
        code.scale(0.9)
        code.to_edge(LEFT * 2)
        return code

    def triangle_case_code(self):
        code_scale = 0.8 
        code = []
        def_statement = TextMobject(r"def triangleCase(simplex, $d$):")
        def_statement[0][:3].set_color(MONOKAI_BLUE)
        def_statement[0][3:15].set_color(MONOKAI_GREEN)
        def_statement[0][16:23].set_color(MONOKAI_ORANGE)
        def_statement[0][-3].set_color(MONOKAI_ORANGE)
        def_statement.scale(code_scale)
        def_statement.to_edge(LEFT)
        code.append(def_statement)

        line_1 = TextMobject(r"$C, B, A =$ simplex")
        line_1[0][5].set_color(MONOKAI_PINK)
        line_1.scale(code_scale)
        line_1.next_to(def_statement, DOWN * 0.5)
        line_1.to_edge(LEFT * 2)
        code.append(line_1)

        line_2 = TextMobject(r"$AB, AC, AO = B - A, C - A, \text{ORIGIN} - A$")
        line_2[0][8].set_color(MONOKAI_PINK)
        line_2[0][10].set_color(MONOKAI_PINK)
        line_2[0][14].set_color(MONOKAI_PINK)
        line_2[0][-2].set_color(MONOKAI_PINK)
        line_2.scale(code_scale)
        line_2.next_to(line_1, DOWN * 0.5)
        line_2.to_edge(LEFT * 2)
        code.append(line_2)

        line_3 = TextMobject(r"ABperp $=$ tripleProd($AC, AB, AB$)")
        line_3[0][6].set_color(MONOKAI_PINK)
        line_3[0][7:17].set_color(MONOKAI_BLUE)
        line_3.scale(code_scale)
        line_3.next_to(line_2, DOWN * 0.5)
        line_3.to_edge(LEFT * 2)
        code.append(line_3)

        line_4 = TextMobject(r"ACperp $=$ tripleProd($AB, AC, AC$)")
        line_4[0][6].set_color(MONOKAI_PINK)
        line_4[0][7:17].set_color(MONOKAI_BLUE)
        line_4.scale(code_scale)
        line_4.next_to(line_3, DOWN * 0.5)
        line_4.to_edge(LEFT * 2)
        code.append(line_4)

        line_5 = TextMobject(r"if dot(ABperp, $AO$) $> 0$: \# Region AB")
        line_5.scale(code_scale)
        line_5.next_to(line_4, DOWN * 0.5)
        line_5.to_edge(LEFT * 2)
        line_5[0][:2].set_color(MONOKAI_PINK)
        line_5[0][2:5].set_color(MONOKAI_BLUE)
        line_5[0][16].set_color(MONOKAI_PINK)
        line_5[0][17].set_color(MONOKAI_PURPLE)
        line_5[0][20:].set_color(MONOKAI_GRAY)
        code.append(line_5)

        line_6 = TextMobject(r"simplex.remove($C$); $d$.set(ABperp)")
        line_6[0][8:14].set_color(MONOKAI_BLUE)
        line_6[0][20:23].set_color(MONOKAI_BLUE)
        line_6.scale(code_scale)
        line_6.next_to(line_5, DOWN * 0.5)
        line_6.to_edge(LEFT * 3)
        code.append(line_6)

        line_7 = TextMobject(r"return False")
        line_7.scale(code_scale)
        line_7.next_to(line_6, DOWN * 0.5)
        line_7.to_edge(LEFT * 3)
        line_7[0][:6].set_color(MONOKAI_PINK)
        line_7[0][6:].set_color(MONOKAI_PURPLE)
        code.append(line_7)

        line_8 = TextMobject(r"elif dot(ACperp, $AO$) $> 0$: \# Region AC")
        line_8.scale(code_scale)
        line_8.next_to(line_7, DOWN * 0.5)
        line_8.to_edge(LEFT * 2)
        line_8[0][:4].set_color(MONOKAI_PINK)
        line_8[0][4:7].set_color(MONOKAI_BLUE)
        line_8[0][18].set_color(MONOKAI_PINK)
        line_8[0][19].set_color(MONOKAI_PURPLE)
        line_8[0][22:].set_color(MONOKAI_GRAY)
        code.append(line_8)

        line_9 = TextMobject(r"simplex.remove($B$); $d$.set(ACperp)")
        line_9[0][8:14].set_color(MONOKAI_BLUE)
        line_9[0][20:23].set_color(MONOKAI_BLUE)
        line_9.scale(code_scale)
        line_9.next_to(line_8, DOWN * 0.5)
        line_9.to_edge(LEFT * 3)
        code.append(line_9)

        line_10 = TextMobject(r"return False")
        line_10.scale(code_scale)
        line_10.next_to(line_9, DOWN * 0.5)
        line_10.to_edge(LEFT * 3)
        line_10[0][:6].set_color(MONOKAI_PINK)
        line_10[0][6:].set_color(MONOKAI_PURPLE)
        code.append(line_10)

        line_11 = TextMobject(r"return True")
        line_11.scale(code_scale)
        line_11.next_to(line_10, DOWN * 0.5)
        line_11.to_edge(LEFT * 2)
        line_11[0][:6].set_color(MONOKAI_PINK)
        line_11[0][6:].set_color(MONOKAI_PURPLE)
        code.append(line_11)

        code = VGroup(*code)
        code.scale(0.9)
        code.to_edge(LEFT * 2)
        return code

class GJKRecap(GJKUtils):
    def construct(self):
        recap = TextMobject("Crazy Shifts in Perspective")
        recap.scale(1.2).move_to(UP * 3.5)

        h_line = Line(LEFT, RIGHT).scale(FRAME_X_RADIUS - 1)
        h_line.next_to(recap, DOWN)

        self.play(
            Write(recap),
            ShowCreation(h_line)
        )

        self.make_conclusion(h_line)

    def make_conclusion(self, h_line):
        scale = 0.7
        indentation = LEFT * 3
        idea_1 = TextMobject(
            r"1. Treating shapes as an infinite set of points $\Rightarrow$ Minkowski differences"
            ).scale(scale).next_to(h_line, DOWN).to_edge(indentation)
       
        idea_2 = TextMobject(
            r"2. Origin inside Minkowski difference $\Rightarrow$ shapes intersect"
            ).scale(scale).next_to(idea_1, DOWN).to_edge(indentation)
        
        idea_3 = TextMobject(
            r"3. Can we build a triangle that surrounds origin of Minkowski difference?"
        ).scale(scale).next_to(idea_2, DOWN).to_edge(indentation)

        idea_4 = TextMobject(
            r"4. Support functions as a method to get points on Minkowski difference"
        ).scale(scale).next_to(idea_3, DOWN).to_edge(indentation)

        screen_rect = ScreenRectangle(height=4)
        screen_rect.next_to(idea_4, DOWN * 1.5)

        self.play(
            ShowCreation(screen_rect)
        )
        self.wait()

        self.play(
            Write(idea_1)
        )

        self.wait()

        self.play(
            Write(idea_2)
        )

        self.wait()

        self.play(
            Write(idea_3)
        )

        self.wait()

        self.play(
            Write(idea_4)
        )

        self.wait(38)
        
class GJKDistanceDemo(GJKUtils):
    CONFIG = {
        "p1_color": YELLOW,
        "p2_color": PERIWINKLE_BLUE,
    }
    def construct(self):
        start_time = time.time()
        shapes = self.get_shapes().shift(UP * 0.5)
        shapes[0].shift(UP + LEFT * 0.8)
        shapes[1].shift(DOWN + RIGHT * 0.8)

        self.play(
            ShowCreation(shapes)
        )
        self.wait()
        self.get_actual_distance(shapes[0], shapes[1])

        self.begin_demo(shapes)
        print("--- %s seconds ---" % (time.time() - start_time))

    def begin_demo(self, shapes):
        d, point1, point2 = self.get_approx_distance(shapes[0], shapes[1])
        line = Line(point1, point2).set_stroke(color=GREEN_SCREEN, width=5)
        self.play(
            ShowCreation(line)
        )
        self.wait()

        def get_dist():
            return np.linalg.norm(line.get_start() - line.get_end())

        text, number = label = VGroup(
            TextMobject("Distance = "),
            DecimalNumber(
                get_dist(),
                num_decimal_places=2,
            )
        )
        label.arrange(RIGHT)
        label.move_to(DOWN * 3.5 + RIGHT * 3)

        self.play(
            Write(label)
        )

        f_always(number.set_value, get_dist)

        self.alphas = smooth_interpolate(0, 1, self.camera.frame_rate * 2)
        self.index = 0
        self.start_color = GREEN_SCREEN
        self.end_color = BRIGHT_RED
        def line_updater(shapes, dt):
            p1, p2 = shapes
            d, point1, point2 = self.get_approx_distance(p1, p2)
            line.put_start_and_end_on(point1, point2)
            if self.index < len(self.alphas):
                color = interpolate_color(self.start_color, self.end_color, self.alphas[self.index])
                line.set_color(color)
                self.index += 1

        shapes.add_updater(line_updater)
        p1, p2 = shapes
        self.play(
            p1.shift, DOWN + RIGHT * 0.5,
            p2.shift, UP + LEFT * 0.5,
            run_time=2,
        )

        self.index = 0
        self.start_color = BRIGHT_RED
        self.end_color = GREEN_SCREEN

        self.play(
            p1.shift, UP + RIGHT * 0.5,
            p2.shift, DOWN + LEFT * 0.5,
            run_time=2
        )

        self.play(
            Rotate(p1, PI/3),
            Rotate(p2, PI / 3),
            run_time=2
        )

        self.alphas = smooth_interpolate(0, 1, self.camera.frame_rate * 3)
        self.index = 0
        self.start_color = GREEN_SCREEN
        self.end_color = BRIGHT_RED

        p1_tranf = p1.copy().rotate(-PI/2).shift(DOWN * 0.7 + LEFT * 0.7)
        p2_tranf = p2.copy().rotate(2 * PI / 4).shift(RIGHT * 0.5 + UP)
        self.play(
            Transform(p1, p1_tranf),
            Transform(p2, p2_tranf),
            run_time=3
        )
        self.wait()

    def get_shapes(self):
        p1 = RegularPolygon(n=5).scale(1.3).set_stroke(color=self.p1_color, width=5)
        p2 = Ellipse(width=2.5, height=1.5).set_stroke(color=self.p2_color, width=5)
        VGroup(p1, p2).arrange_submobjects(DOWN).move_to(RIGHT * 3)
        return VGroup(p1, p2)

    def get_polygons(self):
        p1 = RegularPolygon(n=5).scale(1.5).set_stroke(color=self.p1_color, width=5)
        p2 = RegularPolygon(n=3).scale(1.5).set_stroke(color=self.p2_color, width=5)
        VGroup(p1, p2).arrange_submobjects(DOWN).move_to(RIGHT * 3)
        return VGroup(p1, p2)

    def get_intersection_poly(self, intersection, color=FUCHSIA, width=5, opacity=0.5):
        intersection_poly = Polygon(*intersection)
        intersection_poly.set_stroke(color=color, width=width)
        intersection_poly.set_fill(color=color, opacity=opacity)
        return intersection_poly

    def get_approx_distance(self, p1, p2):
        # Would have loved to implement the full GJK distance algorithm but
        # time constraints forced me to settle with an approximation method
        # increase NUM_POINTS for better approximations (note it will take WAAAYYYY longer)
        NUM_POINTS = 50
        best_point_s1 = p1.get_center()
        best_point_s2 = p2.get_center()
        min_dist = float('inf')
        for a in np.arange(0, 1, 1 / NUM_POINTS):
            for b in np.arange(0, 1, 1 / NUM_POINTS):
                point1 = p1.point_from_proportion(a)
                point2 = p2.point_from_proportion(b)
                dist = np.linalg.norm(point1 - point2)
                if dist < min_dist:
                    min_dist = dist
                    best_point_s1 = point1
                    best_point_s2 = point2

        return min_dist, best_point_s1, best_point_s2

    def get_actual_distance(self, p1, p2):
        # This solution is actually way better
        # Why didn't I think of this idea first
        # I really seem to love forgetting about useful libraries
        # Nevermind tested it and seems like there's something off with representation of ellipses
        # answers are not always accurate
        import shapely.affinity
        from shapely.geometry import Point
        from shapely.ops import nearest_points

        circle = Point(0, 0).buffer(1)  # type(circle)=polygon
        ellipse = shapely.affinity.scale(circle, p2.get_width() / 2, p2.get_height() / 2)  # type(ellipse)=polygon
        ellipse = shapely.affinity.translate(ellipse, xoff=p2.get_center()[0], yoff=p2.get_center()[1])
        p1_vertices = [(a[0], a[1]) for a in p1.get_vertices()]
        polygon = Poly(p1_vertices)
        
        best_points = nearest_points(polygon, ellipse)
        best_point_p1, best_point_p2 = [np.array([p.x, p.y, 0]) for p in best_points]
        dist = np.linalg.norm(best_point_p1 - best_point_p2)
        return dist, best_point_p1, best_point_p2

class GJKPaper(Scene):
    def construct(self):
        paper_img = ImageMobject("GJK_Paper").scale(3.5)
        paper_img.move_to(LEFT * 3)
        self.play(
            FadeIn(paper_img)
        )
        self.wait()

        underline = Line(LEFT, RIGHT).set_stroke(color=BRIGHT_RED)
        underline.put_start_and_end_on(LEFT * 1.75 + UP * 2.64, LEFT * 0.95 + UP * 2.64)
        self.play(
            ShowCreation(underline)
        )

        self.wait()

class Patreons(Scene):
    def construct(self):
        thanks = TextMobject("Special Thanks to These Patreons").scale(1.2)
        patreons = ["Burt Humburg", "Justin Hiester"]
        patreon_text = VGroup(* [thanks] + [TextMobject(name).scale(0.9) for name in patreons])
        patreon_text.arrange(DOWN)
        patreon_text.to_edge(DOWN)

        for text in patreon_text:
            self.play(
                Write(text)
            )
        self.wait(5)


class Thumbnail(Scene):
    """
    There are probably thousands of ways to do this better, but I was towards the end
    of the video making process when I made this, so I was basically not in the mood
    to use my brain. I literally matched pixels to some nice model images of polyhedrons.
    I apologize in advance to anyone trying to understand this code. YOU SHOULD NOT IMPLEMENT
    IT LIKE THIS. THIS IS AN EXAMPLE OF HORRIBLE SOFTWARE ENGINEERING/CODE :P
    """
    def construct(self):
        self.tetrahedron_inside_polyhedron()
        self.make_double_pyramid()
        self.make_dodecahedron()

        arrow = TexMobject(r"\Longrightarrow").scale(2.5)
        arrow.move_to(LEFT)
        self.add(arrow)
        self.wait()

    def tetrahedron_inside_polyhedron(self):
        self.exterior_edge_map = {}
        exterior_pixel_locations = [
        [281, 476],
        [434, 389],
        [452, 169],
        [314, 86],
        [255, 82],
        [94, 140],
        [77, 373],
        ]
        center_x, center_y = 194, 294
        unit_length = 72
        colors = [VIOLET, VIOLET, YELLOW, YELLOW, YELLOW, WHITE, GREEN]
        exterior_vertices = VGroup()
        exterior_edges = VGroup()
        for i, p in enumerate(exterior_pixel_locations):
            dot = Dot()
            location = self.pixel_to_vector(p, cx=center_x, cy=center_y, unit=unit_length)
            dot.move_to(location).set_color(colors[i])
            exterior_vertices.add(dot)
            if i > 0:
                edge = self.connect(exterior_vertices[i - 1], exterior_vertices[i], color=exterior_vertices[i].get_color())
                self.exterior_edge_map[(i - 1, i)] = edge
                exterior_edges.add(edge)

        edge = self.connect(exterior_vertices[-1], exterior_vertices[0])
        self.exterior_edge_map[(0, 5)] = edge
        exterior_edges.add(edge)
        exterior_1 = self.connect(exterior_vertices[3], exterior_vertices[5])
        exterior_edges.add(exterior_1)
        self.exterior_edge_map[(3, 5)] = exterior_1
        exterior_2 = self.connect(exterior_vertices[2], exterior_vertices[4])
        self.exterior_edge_map[(2, 4)] = exterior_2
        exterior_edges.add(exterior_2)

        interior_pixel_locations = [
        [127, 247],
        [193, 282],
        [229, 429],
        [436, 298],
        [324, 244],
        ]

        interior_vertices = [Dot().move_to(self.pixel_to_vector(pixel, cx=center_x, cy=center_y, unit=unit_length)) for pixel in interior_pixel_locations]
        interior_colors = [GREEN, ORANGE, GREEN, WHITE, WHITE]
        for i, v in enumerate(interior_vertices):
            v.set_color(interior_colors[i])
            exterior_vertices.add(v)

        all_edges = self.connect_interior_edges(exterior_vertices, exterior_edges)
        for edge in all_edges:
            edge.set_color(WHITE)
        self.add(all_edges)
        self.wait()

        polyhedron_faces = self.handle_faces(exterior_vertices)
        polyhedron = VGroup(polyhedron_faces, all_edges)
        faces, vertices = self.make_tetrahedron(exterior_vertices)

        self.add(vertices)

        entire_group = VGroup(polyhedron, faces, vertices)
        self.play(
            entire_group.scale, 1.2,
            entire_group.move_to, RIGHT * 3.5
        )

        to_bring_front = [(0, 8), (5, 8), (3, 8), (6, 8), (8, 10), (0, 10), (3, 10)]
        for edge in to_bring_front:
            self.bring_to_front(self.exterior_edge_map[edge])
        self.wait()

    def make_tetrahedron(self, exterior_vertices):
        vertices = [exterior_vertices[8]]
        center_x, center_y = 194, 294
        unit_length = 72
        pixel_locations = [
        [118, 299],
        [343, 114],
        [383, 420],
        ]
        vertices += [Dot().move_to(self.pixel_to_vector(pixel, cx=center_x, cy=center_y, unit=unit_length)) for pixel in pixel_locations]
        vertex_spheres = [Sphere(radius=0.1).move_to(v.get_center()).set_color(YELLOW) for v in vertices]
        faces = VGroup()
        indices = [(0, 1, 2), (0, 2, 3), (0, 1, 3)]
        i = 0
        for u, v, w in indices:
            color = LIGHT_VIOLET
            if i == 2:
                color = VIOLET
            face = self.make_tetrahedron_face(vertices, u, v, w, color=color)
            faces.add(face)
            self.add(face)
            self.wait()
            i += 1
        return faces, VGroup(*vertex_spheres)

    def make_tetrahedron_face(self, vertices, u, v, w, color=LIGHT_VIOLET):
        triangle = Polygon(
            vertices[u].get_center(), 
            vertices[v].get_center(), 
            vertices[w].get_center(),
        ).set_stroke(color=YELLOW)
        triangle.set_fill(color=color, opacity=0.9)
        return triangle


    def pixel_to_vector(self, pixel, cx=0, cy=0, unit=1):
        x = RIGHT * (pixel[0] - cx) / unit
        y = UP * (cy - pixel[1]) / unit
        return x + y

    def connect(self, u, v, color=WHITE, width=2):
        return Line(u.get_center(), v.get_center()).set_stroke(color=color, width=width)
    
    def connect_with_dashed_line(self, u, v, color=WHITE, dash_length=0.1, width=2):
        return DashedLine(u.get_center(), v.get_center(), dash_length=dash_length, width=width)

    def connect_interior_edges(self, exterior_vertices, exterior_edges):
        exterior_to_interior = [
        (0, 8), (0, 9), (0, 10),
        (1, 9), (1, 10), (1, 11),
        (2, 10), (2, 11),
        (3, 8), (3, 10),
        (4, 7), (4, 11),
        (5, 7), (5, 8),
        (6, 7), (6, 8), (6, 9),
        ]

        interior_to_interior = [
        (7, 9), (7, 11),
        (8, 10),
        (9, 11),
        ]
        for u, v in exterior_to_interior:
            self.exterior_edge_map[(u, v)] = self.connect(exterior_vertices[u], exterior_vertices[v])
            exterior_edges.add(self.exterior_edge_map[(u, v)])

        for u, v in interior_to_interior:
            self.exterior_edge_map[(u, v)] = self.connect(exterior_vertices[u], exterior_vertices[v])
            exterior_edges.add(self.exterior_edge_map[(u, v)])

        return exterior_edges

    def handle_faces(self, exterior_vertices):
        faces = [
        (0, 1, 9), (0, 6, 9), (6, 7, 9),
        (6, 7, 5), (5, 7, 4), (2, 3, 4),
        (4, 7, 11), (2, 4, 11), (1, 2, 11),
        (1, 9, 11), (7, 9, 11)
        ]
        face_mob = []
        for u, v, w in faces:
            face = self.fill_face(exterior_vertices, u, v, w)
            face_mob.append(face)
            self.wait()
        self.play(
            face_mob[2].set_fill, GREEN_SCREEN, 0.5,
            face_mob[5].set_fill, GREEN_SCREEN, 0.5,
        )
        return VGroup(*face_mob)

    def fill_face(self, exterior_vertices, u, v, w, color=BLUE, opacity=0.5):
        triangle = Polygon(
            exterior_vertices[u].get_center(), 
            exterior_vertices[v].get_center(), 
            exterior_vertices[w].get_center(),
        )
        triangle.set_stroke(width=0)
        triangle.set_fill(color=color, opacity=opacity)
        self.add(triangle)
        return triangle

    def make_double_pyramid(self):
        self.exterior_edge_map = {}
        center_x, center_y = 416, 435
        unit_length = 200
        double_pyramid_vertex_pixels = [
        [9, 330],
        [474, 572],
        [830, 530],
        [403, 360],
        [542, 12],
        [328, 824],
        ]
        exterior_vertices = VGroup()

        for pixel in double_pyramid_vertex_pixels:
            location = self.pixel_to_vector(pixel, cx=center_x, cy=center_y, unit=unit_length)
            dot = Dot().move_to(location)
            exterior_vertices.add(dot)

        exterior_edges = self.get_edges_double_pyramid(exterior_vertices)


        faces = self.get_faces_double_pyramid(exterior_vertices)

        entire_polyhedron = VGroup(exterior_edges, faces).move_to(LEFT * 4 + DOWN * 1.2)
        self.add(entire_polyhedron)
        self.wait()

    def get_edges_double_pyramid(self, exterior_vertices):
        edges = [
        (0, 1), (0, 3), (0, 4), (0, 5),
        (1, 2), (1, 4), (1, 5),
        (2, 3), (2, 4), (2, 5),
        (3, 4), (3, 5),
        ]
        exterior_edges = VGroup()
        for u, v in edges:
            edge = self.connect(exterior_vertices[u], exterior_vertices[v])
            self.exterior_edge_map[(u, v)] = edge
            exterior_edges.add(self.exterior_edge_map[(u, v)])
        return exterior_edges

    def get_faces_double_pyramid(self, exterior_vertices):
        faces = [
        (0, 1, 4), (1, 2, 4),
        (0, 1, 5), (1, 2, 5),
        ]
        colors = [GREEN_SCREEN, PERIWINKLE_BLUE, SEAFOAM_GREEN, VIOLET]
        face_mob = []
        i = 0
        for u, v, w in faces:
            face = self.fill_face(exterior_vertices, u, v, w, color=colors[i])
            face_mob.append(face)
            i += 1
        return VGroup(*face_mob)

    def make_dodecahedron(self):
        self.exterior_edge_map = {}
        center_x, center_y = 148, 142
        unit_length = 60
        double_pyramid_vertex_pixels = [
        [30, 146],
        [82, 262],
        [206, 251],
        [233, 125],
        [120, 60],
        [130, 5],
        [60, 50],
        [3, 130],
        [31, 211],
        [76, 288],
        [172, 295],
        [252, 275],
        [300, 185],
        [290, 95],
        [228, 29],
        [206, 79],
        [115, 92],
        [98, 180],
        [176, 226],
        [245, 166],
        ]
        exterior_vertices = VGroup()
        for pixel in double_pyramid_vertex_pixels:
            location = self.pixel_to_vector(pixel, cx=center_x, cy=center_y, unit=unit_length)
            dot = Dot().move_to(location)
            exterior_vertices.add(dot)
        dodec_edges = self.get_dodecahedron_edges(exterior_vertices)
        faces = self.get_dodec_faces(exterior_vertices)
        entire_dodecahedron = VGroup(dodec_edges, faces)
        entire_dodecahedron.scale(0.7)
        entire_dodecahedron.move_to(LEFT * 4 + UP * 1.8)
        self.add(entire_dodecahedron)
        self.wait()

    def get_dodecahedron_edges(self, exterior_vertices):
        edges = [
        (0, 1), (0, 4), (0, 7),
        (1, 2), (1, 9),
        (2, 3), (2, 11),
        (3, 4), (3, 13),
        (4, 5),
        (5, 6), (5, 14),
        (6, 7), (6, 16),
        (7, 8),
        (8, 9), (8, 17),
        (9, 10), 
        (10, 11), (10, 18),
        (11, 12),
        (12, 13), (12, 19),
        (13, 14),
        (14, 15),
        (15, 16), (15, 19),
        (16, 17), 
        (17, 18),
        (18, 19),
        ]
        exterior_edges = VGroup()
        for u, v in edges[::-1]:
            width = 3
            if u >= 15 or v >= 15:
                width = 2
            edge = self.connect(exterior_vertices[u], exterior_vertices[v], width=width)
            self.exterior_edge_map[(u, v)] = edge
            exterior_edges.add(self.exterior_edge_map[(u, v)])
        return exterior_edges

    def get_dodec_faces(self, exterior_vertices):
        faces = [
        (0, 1, 2, 3, 4), 
        (0, 4, 5, 6, 7),
        (0, 7, 8, 9, 1),
        (1, 9, 10, 11, 2),
        (2, 11, 12, 13, 3),
        (3, 13, 14, 5, 4),
        ]
        colors = [GREEN_SCREEN, PERIWINKLE_BLUE, SEAFOAM_GREEN, ORANGE, VIOLET, COBALT_BLUE]
        face_mob = []
        i = 0
        for indices in faces:
            face = self.polygon_fill_face(exterior_vertices, indices, color=colors[i])
            face_mob.append(face)
            i += 1
        return VGroup(*face_mob)

    def polygon_fill_face(self, exterior_vertices, indices, color=BLUE, opacity=0.5):
        polygon = Polygon(*[exterior_vertices[u].get_center() for u in indices])
        polygon.set_stroke(width=0)
        polygon.set_fill(color=color, opacity=opacity)
        return polygon

### Utility Functions
def get_random_vector(dims=2):
    ### Samples a unit vector randomly
    vec = [np.random.normal() for _ in range(dims)]
    mag = sum([x**2 for x in vec]) ** 0.5
    return np.array([x / mag for x in vec] + [0])

def smooth_interpolate(start, end, frames):
    dt = 1 / frames
    alpha = 0
    result = []
    for i in range(frames):
        alpha = smooth(i * dt)
        result.append((1 - alpha) * start + alpha * end)
    result.append(end)
    return result

def remove_array(L, arr):
    """
    Removes numpy array arr from list L
    """
    ind = 0
    size = len(L)
    while ind != size and not np.array_equal(L[ind],arr):
        ind += 1
    if ind != size:
        L.pop(ind)
    else:
        raise ValueError('array not found in list.')

def coords_to_array(coords):
    """
    @param: coords - list of tuples [(x1, y2), (x2, y2), ... , (xn, yn)]
    returns: [np.array([x1, y1, 0]), np.array([x2, y2, 0]), ... , np.array([xn, yn, 0])]
    """
    return [np.array([x, y, 0]) for x, y in coords]

def array_in_list(arr, lst):
    for a in lst:
        if np.array_equal(arr, a):
            return True
    return False

def tuple_array_in_list(tup, lst):
    for t in lst:
        if np.array_equal(t[0], tup[0]) and np.array_equal(t[1], tup[1]):
            return True
    return False

def point_to_angle(point, circle):
    x, y = point[:2]
    cx, cy = circle.get_center()[:2]
    return np.arctan2(y - cy, x - cx)

def set_array(a1, a2):
    a1[0], a1[1], a2[0] = a2[0], a2[1], a2[2]

def tripleProd(a, b, c):
        """
        @param a, b, c: np.array[x, y, z] vectors
        return: (a x b) x c -- x represents cross product
        """
        return b * np.dot(c, a) - a * np.dot(c, b)

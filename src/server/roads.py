"""
Roads of the biome on the server, roads.json of the biome archive, see server.land.build_roads_data().

Every road is a line of points (x, y) in segments, the units walk along it from point to point,
see behavior.UnitRoadMove. Roads which come close to each other are connected there, see JUNCTION_DISTANCE,
so the way from one place to another can go along several roads, see Roads.find_way().
"""

#------------------------------------------------------------------------------

_Debug = False

#------------------------------------------------------------------------------

import math
import heapq

#------------------------------------------------------------------------------

from server import behavior

#------------------------------------------------------------------------------

# points of two different roads which are that close to each other are connected, in segments:
# a unit can go there from one road to the other one
JUNCTION_DISTANCE = 1.0

#------------------------------------------------------------------------------


class Roads(object):

    def __init__(self, roads_data, width, height):
        # the map is wrapped around, distances are measured the short way
        self.width = width
        self.height = height
        # road id -> [(x, y), ...]
        self.roads = {}
        for road in roads_data:
            self.roads[road['i']] = [(float(x), float(y)) for x, y in road['p']]
        # (road id, point index) -> [((road id, point index), length), ...], see build_links()
        self.links = self.build_links()
        if _Debug:
            print(f'{len(self.roads)} roads with {len(self.links)} points')

    def __len__(self):
        return len(self.roads)

    def shift(self, x, y, to_x, to_y):
        return behavior.nearest_shift(x, to_x, self.width), behavior.nearest_shift(y, to_y, self.height)

    def distance(self, x, y, to_x, to_y):
        dx, dy = self.shift(x, y, to_x, to_y)
        return math.sqrt(dx * dx + dy * dy)

    def points(self, road_id):
        return self.roads[road_id]

    def build_links(self):
        """
        Every point of a road is linked with the next and the previous points of the same road
        and with the points of other roads not further than JUNCTION_DISTANCE.
        """
        links = {}
        for road_id, points in self.roads.items():
            for i in range(len(points)):
                links[(road_id, i)] = []
            for i in range(len(points) - 1):
                length = self.distance(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1])
                links[(road_id, i)].append(((road_id, i + 1), length))
                links[(road_id, i + 1)].append(((road_id, i), length))
        road_ids = sorted(self.roads.keys())
        for n, road_id in enumerate(road_ids):
            for other_road_id in road_ids[n + 1:]:
                for i, (x, y) in enumerate(self.roads[road_id]):
                    for j, (other_x, other_y) in enumerate(self.roads[other_road_id]):
                        length = self.distance(x, y, other_x, other_y)
                        if length <= JUNCTION_DISTANCE:
                            links[(road_id, i)].append(((other_road_id, j), length))
                            links[(other_road_id, j)].append(((road_id, i), length))
                            if _Debug:
                                print(f'road {road_id} at point {i} is connected with road {other_road_id} at point {j}')
        return links

    def nearest_point(self, road_id, x, y):
        """
        The point of the road nearest to (x, y), it can be between two points of the road.
        Returns (index, px, py, distance), the point is on the piece of the road from the point index to the point index + 1.
        """
        points = self.roads[road_id]
        best = None
        for i in range(len(points) - 1):
            ax, ay = points[i]
            bx, by = points[i + 1]
            ux, uy = self.shift(ax, ay, x, y)
            vx, vy = self.shift(ax, ay, bx, by)
            length2 = vx * vx + vy * vy
            t = 0.0
            if length2 > 0.0:
                t = max(0.0, min(1.0, (ux * vx + uy * vy) / length2))
            dx = ux - vx * t
            dy = uy - vy * t
            distance = math.sqrt(dx * dx + dy * dy)
            if best is None or distance < best[3]:
                best = (i, ax + vx * t, ay + vy * t, distance)
        return best

    def nearest_road(self, x, y):
        """
        The road nearest to (x, y): returns (road id, index, px, py, distance), see nearest_point(), or None when there are no roads.
        """
        best = None
        for road_id in self.roads:
            i, px, py, distance = self.nearest_point(road_id, x, y)
            if best is None or distance < best[4]:
                best = (road_id, i, px, py, distance)
        return best

    def way(self, road_id, x, y, to_index):
        """
        Points the unit at (x, y) goes through to come along the road to its point to_index, a negative index counts from the end.
        The first one is the nearest point of the road, then the points of the road one by one.
        """
        points = self.roads[road_id]
        to_index %= len(points)
        i, px, py, _ = self.nearest_point(road_id, x, y)
        if to_index > i:
            rest = points[i + 1:to_index + 1]
        else:
            rest = list(reversed(points[to_index:i + 1]))
        return [(px, py)] + list(rest)

    def find_way(self, x, y, to_x, to_y):
        """
        The shortest way along the roads from (x, y) to the point of the roads nearest to (to_x, to_y).
        The unit goes to the nearest road first, see nearest_road(). Returns [(road id, point index), ...]:
        the unit walks along the first road to that point of it, then along the next road and so on,
        or None when there are no roads or they do not lead there.
        """
        start = self.nearest_road(x, y)
        if start is None:
            return None
        road_id, i, px, py, _ = start
        target = None
        target_distance = None
        for other_road_id, points in self.roads.items():
            for j, (point_x, point_y) in enumerate(points):
                distance = self.distance(point_x, point_y, to_x, to_y)
                if target is None or distance < target_distance:
                    target, target_distance = (other_road_id, j), distance
        # Dijkstra from the two points around the nearest point of the road
        costs = {}
        previous = {}
        queue = []
        for k in (i, i + 1):
            node = (road_id, k)
            point_x, point_y = self.roads[road_id][k]
            costs[node] = self.distance(px, py, point_x, point_y)
            previous[node] = None
            heapq.heappush(queue, (costs[node], node))
        while queue:
            cost, node = heapq.heappop(queue)
            if cost > costs[node]:
                continue
            if node == target:
                break
            for next_node, length in self.links[node]:
                next_cost = cost + length
                if next_node not in costs or next_cost < costs[next_node]:
                    costs[next_node] = next_cost
                    previous[next_node] = node
                    heapq.heappush(queue, (next_cost, next_node))
        if target not in previous:
            return None
        nodes = []
        node = target
        while node is not None:
            nodes.append(node)
            node = previous[node]
        nodes.reverse()
        # the last point on every road the way goes along
        legs = []
        for n, node in enumerate(nodes):
            if n == len(nodes) - 1 or nodes[n + 1][0] != node[0]:
                legs.append(node)
        return legs

"""Obstacle-aware waypoint planner for RescueRack."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
import math


Point = tuple[float, float]
GridCell = tuple[int, int]


@dataclass(frozen=True)
class CircleObstacle:
    name: str
    center: Point
    radius: float


@dataclass(frozen=True)
class PlannedRoute:
    waypoints: tuple[Point, ...]
    grid_cells: tuple[GridCell, ...]
    expanded_nodes: int
    min_clearance_m: float
    fallback_used: bool = False


DEFAULT_OBSTACLES: tuple[CircleObstacle, ...] = (
    CircleObstacle("fallen_beam", (-0.35, 0.18), 1.02),
    CircleObstacle("pallet_block", (-1.15, 0.78), 0.62),
    CircleObstacle("loose_barrel", (0.15, -0.45), 0.54),
)


class AStarPlanner:
    """Small deterministic planner used to generate debris-aware route evidence."""

    def __init__(
        self,
        bounds: tuple[float, float, float, float] = (-3.55, 2.8, -2.1, 1.75),
        resolution: float = 0.18,
        obstacles: tuple[CircleObstacle, ...] = DEFAULT_OBSTACLES,
        safety_margin: float = 0.16,
    ) -> None:
        self.min_x, self.max_x, self.min_y, self.max_y = bounds
        self.resolution = resolution
        self.obstacles = obstacles
        self.safety_margin = safety_margin

    def plan(self, start: Point, goal: Point) -> PlannedRoute:
        start_cell = self.to_cell(start)
        goal_cell = self.to_cell(goal)
        came_from: dict[GridCell, GridCell | None] = {start_cell: None}
        cost_so_far: dict[GridCell, float] = {start_cell: 0.0}
        frontier: list[tuple[float, GridCell]] = []
        heappush(frontier, (0.0, start_cell))
        expanded = 0

        while frontier:
            _, current = heappop(frontier)
            expanded += 1
            if current == goal_cell:
                break

            for neighbor in self.neighbors(current):
                point = self.to_point(neighbor)
                if self.blocked(point):
                    continue
                step_cost = math.dist(self.to_point(current), point)
                new_cost = cost_so_far[current] + step_cost
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + math.dist(point, goal)
                    heappush(frontier, (priority, neighbor))
                    came_from[neighbor] = current

        if goal_cell not in came_from:
            return PlannedRoute((goal,), (), expanded, self.clearance(goal), fallback_used=True)

        cells = self.reconstruct(came_from, goal_cell)
        points = tuple(self.simplify([self.to_point(cell) for cell in cells], start, goal))
        return PlannedRoute(
            waypoints=points,
            grid_cells=tuple(cells),
            expanded_nodes=expanded,
            min_clearance_m=min(self.clearance(point) for point in points),
        )

    def plan_via(self, start: Point, via: tuple[Point, ...], goal: Point) -> PlannedRoute:
        points: list[Point] = []
        cells: list[GridCell] = []
        expanded = 0
        fallback = False
        min_clearance = float("inf")
        current = start
        for target in (*via, goal):
            segment = self.plan(current, target)
            points.extend(segment.waypoints)
            cells.extend(segment.grid_cells)
            expanded += segment.expanded_nodes
            fallback = fallback or segment.fallback_used
            min_clearance = min(min_clearance, segment.min_clearance_m)
            current = target
        return PlannedRoute(tuple(points), tuple(cells), expanded, min_clearance, fallback)

    def to_cell(self, point: Point) -> GridCell:
        x = round((point[0] - self.min_x) / self.resolution)
        y = round((point[1] - self.min_y) / self.resolution)
        return (x, y)

    def to_point(self, cell: GridCell) -> Point:
        return (
            round(self.min_x + cell[0] * self.resolution, 4),
            round(self.min_y + cell[1] * self.resolution, 4),
        )

    def neighbors(self, cell: GridCell) -> tuple[GridCell, ...]:
        offsets = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
        result: list[GridCell] = []
        for dx, dy in offsets:
            nxt = (cell[0] + dx, cell[1] + dy)
            point = self.to_point(nxt)
            if self.min_x <= point[0] <= self.max_x and self.min_y <= point[1] <= self.max_y:
                result.append(nxt)
        return tuple(result)

    def blocked(self, point: Point) -> bool:
        return any(math.dist(point, obstacle.center) <= obstacle.radius + self.safety_margin for obstacle in self.obstacles)

    def clearance(self, point: Point) -> float:
        return min(math.dist(point, obstacle.center) - obstacle.radius for obstacle in self.obstacles)

    @staticmethod
    def reconstruct(came_from: dict[GridCell, GridCell | None], goal: GridCell) -> list[GridCell]:
        current: GridCell | None = goal
        cells: list[GridCell] = []
        while current is not None:
            cells.append(current)
            current = came_from[current]
        cells.reverse()
        return cells

    @staticmethod
    def simplify(points: list[Point], start: Point, goal: Point) -> list[Point]:
        if len(points) <= 2:
            return [goal]
        simplified: list[Point] = [start]
        last_direction: tuple[int, int] | None = None
        for a, b in zip(points, points[1:]):
            direction = (round(b[0] - a[0], 3), round(b[1] - a[1], 3))
            if last_direction is not None and direction != last_direction:
                simplified.append(a)
            last_direction = direction
        simplified.append(goal)
        deduped: list[Point] = []
        for point in simplified:
            if not deduped or math.dist(deduped[-1], point) > 0.08:
                deduped.append(point)
        return deduped[1:]


def default_planner() -> AStarPlanner:
    return AStarPlanner()

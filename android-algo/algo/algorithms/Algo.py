from typing import Union, List, Tuple
import heapq
import math
import numpy as np
from python_tsp.heuristics import solve_tsp_lin_kernighan
from algo.entities.entity import CellState, Obstacle, Grid
from algo.entities.robot import Robot
from algo.tools.consts import (
    TURN_FACTOR,
    ITERATIONS,
    SAFE_COST,
    TURN_DISPLACEMENT,
    REVERSE_FACTOR,
    PADDING,
    ARENA_WIDTH,
    ARENA_HEIGHT,
)
from algo.tools.movement import (
    Direction,
    MOVE_DIRECTION,
    Motion
)


class MazeSolver:
    """
    Finds a near‑optimal route for a robot to visit capture/view states around obstacles.
    Assumes obstacles are static (neighbor cache remains valid across searches).
    """

    def __init__(
        self,
        size_x: int = ARENA_WIDTH,
        size_y: int = ARENA_HEIGHT,
        robot: Union[Robot, None] = None,
        robot_x: int = 1,
        robot_y: int = 1,
        robot_direction: Direction = Direction.NORTH,
    ) -> None:
        # Obstacles don't change -> safe to cache neighbors for A*
        self.neighbor_cache: dict[
            Tuple[int, int, Direction],
            List[Tuple[int, int, Direction, int, Motion]]
        ] = {}

        self.grid = Grid(size_x, size_y)
        self.robot = robot if robot else Robot(robot_x, robot_y, robot_direction)

        # All-pairs results between CellStates
        self.path_table: dict[Tuple[CellState, CellState], List[Tuple[int, int, Direction]]] = {}
        self.cost_table: dict[Tuple[CellState, CellState], float] = {}

        # Edge->Motion labels (x,y,d, nx,ny,nd)
        self.motion_table: dict[Tuple[int, int, Direction, int, int, Direction], Motion] = {}

    # ---------- Obstacle management ----------

    def add_obstacle(self, x: int, y: int, direction: Direction, obstacle_id: int) -> None:
        """Add a single obstacle to the grid."""
        self.grid.add_obstacle(Obstacle(x, y, direction, obstacle_id))

    def clear_obstacles(self) -> None:
        """Remove all obstacles from the grid."""
        self.grid.reset_obstacles()
        # If you ever make obstacles dynamic, also clear caches:
        # self.neighbor_cache.clear()
        # self.path_table.clear()
        # self.cost_table.clear()
        # self.motion_table.clear()

    # ---------- Main planner ----------

    def get_optimal_path(self, force_visit_all: bool = True) -> tuple[List[CellState], float]:
        """
        Plan a route that visits capture states.

        Args:
            force_visit_all: When True, must visit one view for EVERY obstacle.
                             When False, solver may choose any subset (searched
                             from most -> least obstacles).

        Returns:
            (optimal_path: List[CellState], total_cost: float)
        """
        min_dist = 1e9
        optimal_path: List[CellState] = []

        # 1) Candidate view positions per obstacle
        views: List[List[CellState]] = self.grid.get_view_obstacle_positions()
        num_views = len(views)
        if num_views == 0:
            return [self.robot.get_start_state()], 0.0

        # Helper to evaluate a *fixed* selection of obstacle groups (cur_view_positions)
        def evaluate_groups(cur_view_positions: List[List[CellState]]):
            nonlocal min_dist, optimal_path

            # Build flattened pool = [start] + all candidate states in these groups
            visit_states: List[CellState] = [self.robot.get_start_state()]
            for group in cur_view_positions:
                visit_states.extend(group)

            # Precompute all-pairs A*
            self._generate_paths(visit_states)

            # Choose exactly one view per obstacle (cap by ITERATIONS)
            combinations = MazeSolver._generate_combinations(
                cur_view_positions, 0, [], [], ITERATIONS
            )

            for combo in combinations:
                # Map per-obstacle choice to global indices inside visit_states
                visited_idx = [0]  # start state is index 0 in visit_states
                base = 1
                penalty_cost = 0.0
                for gi, group in enumerate(cur_view_positions):
                    global_index = base + combo[gi]
                    visited_idx.append(global_index)
                    penalty_cost += group[combo[gi]].penalty
                    base += len(group)

                n = len(visited_idx)
                cost_matrix = np.zeros((n, n))
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        a = visit_states[visited_idx[i]]
                        b = visit_states[visited_idx[j]]
                        c = self.cost_table.get((a, b), 1e9)
                        cost_matrix[i, j] = c
                        cost_matrix[j, i] = c

                # Turn TSP cycle into path anchored at node 0
                cost_matrix[:, 0] = 0

                permutation, distance = solve_tsp_lin_kernighan(cost_matrix)
                total = float(distance) + float(penalty_cost)
                if total >= min_dist:
                    continue

                # Reconstruct concrete path
                min_dist = total
                optimal_path = [visit_states[0]]
                for k in range(len(permutation) - 1):
                    s = visit_states[visited_idx[permutation[k]]]
                    t = visit_states[visited_idx[permutation[k + 1]]]
                    seg = self.path_table[(s, t)]
                    # append skipping the first node to avoid duplication
                    for u in range(1, len(seg)):
                        cx, cy, cd = seg[u]
                        optimal_path.append(CellState(cx, cy, cd))

                    # Tag capture pose (id_L/R/C) if target is a capture state
                    if t.screenshot_id is not None:
                        obs = self.grid.find_obstacle_by_id(int(t.screenshot_id))
                        if not obs:
                            raise ValueError(f"Obstacle with id {t.screenshot_id} not found")
                        rel = MazeSolver._get_capture_relative_position(optimal_path[-1], obs)
                        optimal_path[-1].set_screenshot(f"{t.screenshot_id}_{rel}")

        if force_visit_all:
            # Force include *every* obstacle group
            evaluate_groups(views)
        else:
            # Enumerate subsets (binary masks), visiting more obstacles first
            for mask in self._get_visit_options(num_views):
                cur_groups: List[List[CellState]] = []
                for i, bit in enumerate(mask):
                    if bit == "1":
                        cur_groups.append(views[i])
                if not cur_groups:
                    # even allow empty subset: trivial solution is just "stay put"
                    cur_groups = []
                evaluate_groups(cur_groups)
                # Early exit if we already found a path that visits *all* obstacles
                if len(cur_groups) == num_views and optimal_path:
                    break

        return optimal_path, float(min_dist)

    # ---------- All-pairs A* precompute ----------

    def _generate_paths(self, states: List[CellState]) -> None:
        """Run A* for all unordered pairs in 'states' and cache results."""
        for i in range(len(states) - 1):
            for j in range(i + 1, len(states)):
                self._astar_search(states[i], states[j])

    # ---------- A* search ----------

    def _astar_search(self, start: CellState, end: CellState) -> None:
        """A* from 'start' to 'end' over (x,y,dir) states."""
        if (start, end) in self.path_table:
            return

        g_dist: dict[Tuple[int, int, Direction], float] = {(start.x, start.y, start.direction): 0.0}
        visited: set[Tuple[int, int, Direction]] = set()
        parent: dict[Tuple[int, int, Direction], Tuple[int, int, Direction]] = {}

        heap: List[Tuple[float, int, int, Direction]] = [
            (self._estimate_distance(start, end), start.x, start.y, start.direction)
        ]

        while heap:
            _, x, y, d = heapq.heappop(heap)
            if (x, y, d) in visited:
                continue

            if end.is_eq(x, y, d):
                self._record_path(start, end, parent, g_dist[(x, y, d)])
                return

            visited.add((x, y, d))
            g = g_dist[(x, y, d)]

            for (nx, ny, nd, safe_cost, motion) in self._get_neighboring_states(x, y, d):
                if (nx, ny, nd) in visited:
                    continue

                # Store edge->motion once (other dir inferred)
                if (x, y, d, nx, ny, nd) not in self.motion_table and \
                   (nx, ny, nd, x, y, d) not in self.motion_table:
                    self.motion_table[(x, y, d, nx, ny, nd)] = motion

                turn_cost = TURN_FACTOR * Direction.turn_cost(d, nd)
                reverse_cost = REVERSE_FACTOR * motion.reverse_cost()
                step_cost = turn_cost + reverse_cost + safe_cost
                shot = end.penalty if end.is_eq(nx, ny, nd) else 0.0

                g_new = g + step_cost + shot
                f = g_new + self._estimate_distance(CellState(nx, ny, nd), end)

                if (nx, ny, nd) not in g_dist or g_dist[(nx, ny, nd)] > g_new:
                    g_dist[(nx, ny, nd)] = g_new
                    heapq.heappush(heap, (f, nx, ny, nd))
                    parent[(nx, ny, nd)] = (x, y, d)

    # ---------- Neighbor generation (refactored) ----------

    def _get_neighboring_states(
        self, x: int, y: int, direction: Direction
    ) -> List[Tuple[int, int, Direction, int, Motion]]:
        """
        Returns neighbors as (nx, ny, ndir, safe_cost, motion):
        - Straight moves (FORWARD/REVERSE) using MOVE_DIRECTION.
        - Turn moves (forward/reverse left/right) via a compact lookup.
        """
        key = (x, y, direction)
        if key in self.neighbor_cache:
            return self.neighbor_cache[key]

        neighbors: List[Tuple[int, int, Direction, int, Motion]] = []

        # Straight motions
        for dx, dy, md in MOVE_DIRECTION:
            if md == direction:
                fx, fy = x + dx, y + dy
                if self.grid.reachable(fx, fy):
                    neighbors.append((fx, fy, md, self._calculate_safe_cost(fx, fy), Motion.FORWARD))
                rx, ry = x - dx, y - dy
                if self.grid.reachable(rx, ry):
                    neighbors.append((rx, ry, md, self._calculate_safe_cost(rx, ry), Motion.REVERSE))

        # Turn motions via a compact mapping
        big, small = TURN_DISPLACEMENT[0], TURN_DISPLACEMENT[1]

        def try_turn(to_dir: Direction, offx: int, offy: int, motion: Motion):
            tx, ty = x + offx, y + offy
            if self.grid.turn_reachable(x, y, tx, ty, direction):
                neighbors.append((tx, ty, to_dir, self._calculate_safe_cost(tx, ty), motion))

        TURN_MAP: dict[Tuple[Direction, Direction], List[Tuple[int, int, Motion]]] = {
            (Direction.NORTH, Direction.EAST): [(+big, +small, Motion.FORWARD_RIGHT_TURN), (-small, -big, Motion.REVERSE_LEFT_TURN)],
            (Direction.EAST,  Direction.NORTH): [(+small, +big, Motion.FORWARD_LEFT_TURN), (-big,  -small, Motion.REVERSE_RIGHT_TURN)],
            (Direction.EAST,  Direction.SOUTH): [(+small, -big, Motion.FORWARD_RIGHT_TURN), (-big,  +small, Motion.REVERSE_LEFT_TURN)],
            (Direction.SOUTH, Direction.EAST):  [(+big,  -small, Motion.FORWARD_LEFT_TURN), (-small, +big,  Motion.REVERSE_RIGHT_TURN)],
            (Direction.SOUTH, Direction.WEST):  [(-big,  -small, Motion.FORWARD_RIGHT_TURN), (+small, +big,  Motion.REVERSE_LEFT_TURN)],
            (Direction.WEST,  Direction.SOUTH): [(-small, -big,  Motion.FORWARD_LEFT_TURN), (+big,  +small, Motion.REVERSE_RIGHT_TURN)],
            (Direction.WEST,  Direction.NORTH): [(-small, +big,  Motion.FORWARD_RIGHT_TURN), (+big,  -small, Motion.REVERSE_LEFT_TURN)],
            (Direction.NORTH, Direction.WEST):  [(-big,  +small, Motion.FORWARD_LEFT_TURN), (+small, -big,  Motion.REVERSE_RIGHT_TURN)],
        }

        for nd in (Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST):
            if nd == direction:
                continue
            for dx, dy, mot in TURN_MAP.get((direction, nd), []):
                try_turn(nd, dx, dy, mot)

        self.neighbor_cache[key] = neighbors
        return neighbors

    # ---------- Cost & record helpers ----------

    def _calculate_safe_cost(self, new_x: int, new_y: int) -> int:
        """Penalty for being within a PADDING‑box of any obstacle (single concise check)."""
        for obj in self.grid.obstacles:
            if abs(obj.x - new_x) <= PADDING and abs(obj.y - new_y) <= PADDING:
                return SAFE_COST
        return 0

    def _record_path(
        self,
        start: CellState,
        end: CellState,
        parent: dict[Tuple[int, int, Direction], Tuple[int, int, Direction]],
        cost: float,
    ) -> None:
        """Record symmetric costs and forward/reverse paths for (start, end)."""
        self.cost_table[(start, end)] = cost
        self.cost_table[(end, start)] = cost

        path: List[Tuple[int, int, Direction]] = []
        p = (end.x, end.y, end.direction)
        while p in parent:
            path.append(p)
            p = parent[p]
        path.append(p)  # include the start

        fwd = path[::-1]
        rev = path
        self.path_table[(start, end)] = fwd
        self.path_table[(end, start)] = rev

    # ---------- Heuristic & combinatorics ----------

    @staticmethod
    def _estimate_distance(start: CellState, end: CellState, level: int = 0) -> float:
        """
        Heuristic distance between two states.
        level 0: Manhattan (default)
        level 1: Euclidean
        """
        dx = start.x - end.x
        dy = start.y - end.y
        if level == 1:
            return math.sqrt(dx * dx + dy * dy)  # float
        return float(abs(dx) + abs(dy))          # explicit float

    @staticmethod
    def _get_visit_options(n: int) -> List[str]:
        """
        Generate all n-bit masks as strings, sorted by number of '1's (desc),
        so larger subsets are considered first (e.g., '111', '110', '101', ... '000').
        """
        masks = [format(i, f"0{n}b") for i in range(2 ** n)]
        masks.sort(key=lambda s: s.count("1"), reverse=True)
        return masks

    @staticmethod
    def _generate_combinations(
        view_positions: List[List[CellState]],
        index: int,
        current: List[int],
        result: List[List[int]],
        num_iters: int,
    ) -> List[List[int]]:
        """Pick exactly one view index per obstacle list, limited by num_iters."""
        if index == len(view_positions):
            result.append(current.copy())
            return result
        if num_iters == 0:
            return result
        num_iters -= 1
        for i in range(len(view_positions[index])):
            current.append(i)
            MazeSolver._generate_combinations(view_positions, index + 1, current, result, num_iters)
            current.pop()
        return result

    @staticmethod
    def _get_capture_relative_position(cell_state: CellState, obstacle: Obstacle) -> str:
        """Classify obstacle as L/R/C relative to the robot's facing."""
        x, y, d = cell_state.x, cell_state.y, cell_state.direction
        ox, oy = obstacle.x, obstacle.y

        if d == Direction.NORTH:
            if ox == x and oy > y: return "C"
            return "L" if ox < x else "R"
        if d == Direction.SOUTH:
            if ox == x and oy < y: return "C"
            return "R" if ox < x else "L"
        if d == Direction.EAST:
            if oy == y and ox > x: return "C"
            return "R" if oy < y else "L"
        if d == Direction.WEST:
            if oy == y and ox < x: return "C"
            return "L" if oy < y else "R"
        raise ValueError(f"Invalid direction {d}.")

    # ---------- Convert path to motions ----------

    def optimal_path_to_motion_path(
        self, optimal_path: List[CellState]
    ) -> tuple[List[Motion], List[str], List[Obstacle]]:
        """
        Convert the cell-by-cell optimal path into:
        - motion list,
        - capture labels (e.g., '5_L'),
        - scanned obstacles (in order).
        """
        motion_path: List[Motion] = []
        capture_labels: List[str] = []
        scanned: List[Obstacle] = []

        for i in range(len(optimal_path) - 1):
            a = optimal_path[i]
            b = optimal_path[i + 1]
            x, y, d = a.x, a.y, a.direction
            nx, ny, nd = b.x, b.y, b.direction

            if (nx, ny, nd, x, y, d) in self.motion_table:
                motion = self.motion_table[(nx, ny, nd, x, y, d)].opposite_motion()
            elif (x, y, d, nx, ny, nd) in self.motion_table:
                motion = self.motion_table[(x, y, d, nx, ny, nd)]
            else:
                raise ValueError(f"Invalid path from {a} to {b} (missing motion).")

            motion_path.append(motion)

            if b.screenshot_id is not None:
                motion_path.append(Motion.CAPTURE)
                capture_labels.append(str(b.screenshot_id))
                obs_id = int(str(b.screenshot_id).split("_")[0])
                obs = self.grid.find_obstacle_by_id(obs_id)
                if obs is None:
                    raise ValueError(f"Obstacle id {obs_id} not found in grid")
                scanned.append(obs)

        return motion_path, capture_labels, scanned

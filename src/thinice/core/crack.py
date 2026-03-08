"""Crack component for ice breaking visualization."""
from typing import List, Tuple, Callable, Optional
import math
import random
import pygame
from ..config.settings import crack as config
from ..utils.geometry import Point, generate_jagged_line


class Crack:
    """Represents a crack in the ice."""

    def __init__(self, start_point: Point, is_secondary: bool = False):
        """Initialize a new crack.

        Args:
            start_point: Starting point of the crack
            is_secondary: Whether this is a secondary (thinner) crack
        """
        self.points: List[Point] = [start_point]
        self.is_secondary = is_secondary
        self.thickness = 2 if is_secondary else 3
        self.total_length = 0.0

    def add_point(self, point: Point) -> None:
        """Add a new point to the crack.

        Args:
            point: The point to add
        """
        if self.points:
            prev = self.points[-1]
            dx = point[0] - prev[0]
            dy = point[1] - prev[1]
            self.total_length += (dx * dx + dy * dy) ** 0.5
        self.points.append(point)

    def extend_to(self, end_point: Point, num_segments: int) -> None:
        """Extend the crack to an endpoint with a jagged line.

        Args:
            end_point: The target end point
            num_segments: Number of segments in the jagged line
        """
        if not self.points:
            return

        new_points = generate_jagged_line(
            self.points[-1],
            end_point,
            num_segments,
            config.MAX_DEVIATION
        )

        # Skip the first point as it's already in self.points
        for point in new_points[1:]:
            self.add_point(point)

    def draw(self, screen: pygame.Surface) -> None:
        """Draw the crack on the screen.

        Args:
            screen: Pygame surface to draw on
        """
        if len(self.points) < 2:
            return

        # Draw shadow (wider for more visibility)
        for i in range(len(self.points) - 1):
            pygame.draw.line(
                screen,
                config.SHADOW_COLOR,
                (self.points[i][0] + 1, self.points[i][1] + 1),
                (self.points[i+1][0] + 1, self.points[i+1][1] + 1),
                self.thickness + 2
            )

        # Draw crack
        for i in range(len(self.points) - 1):
            pygame.draw.line(
                screen,
                config.COLOR,
                self.points[i],
                self.points[i+1],
                self.thickness
            )


# ── Hex-grid crack tree ───────────────────────────────────────────────────────

class _CrackEdge:
    """One segment of a crack tree connecting two hex centres."""
    __slots__ = ('pts', 'depth', 'start_d', 'end_d', 'child')

    def __init__(self, pts: List[Point], depth: int,
                 start_d: float, end_d: float, child: '_CrackNode') -> None:
        self.pts     = pts      # jagged line from one hex centre to the next
        self.depth   = depth    # 0 = primary, 1 = secondary, 2 = tertiary
        self.start_d = start_d  # cumulative distance from origin at segment start
        self.end_d   = end_d    # cumulative distance from origin at segment end
        self.child   = child    # subtree rooted at the far hex


class _CrackNode:
    """Node in the crack tree, situated at a hex centre."""
    __slots__ = ('edges',)

    def __init__(self) -> None:
        self.edges: List[_CrackEdge] = []


class CrackTree:
    """Hex-grid-aligned branching crack tree.

    Cracks radiate from an origin hex centre and walk along the hex grid,
    terminating at neighbouring hex centres.  Arms can skip one intermediate
    hex (~20 % chance for shallow depth), and secondary / tertiary branches
    sprout from reached hexes — mirroring the HTML/JS demo algorithm.

    Usage::

        tree = CrackTree(origin_hex, get_neighbors_fn)
        # during animation:
        tree.draw(screen, progress)        # 0 ≤ progress ≤ 1
        # when fully formed:
        tree.draw_complete(screen)
    """

    EASE_K = 4.6  # exponential ease-out constant (matches HTML demo)

    # Visual style by depth (primary / secondary / tertiary)
    _LINE_WIDTHS   = [2, 1, 1]
    _CRACK_COLORS  = [(14, 28, 48), (11, 23, 42), (9, 18, 36)]
    _SHADOW_COLOR  = (3, 8, 20, 107)   # rgba
    _HIGHLIGHT     = (208, 234, 252, 153)  # rgba, primary cracks only

    def __init__(self, origin_hex, get_neighbors: Callable) -> None:
        """Build the crack tree.

        Args:
            origin_hex:    The Hex object where cracking begins.
            get_neighbors: Callable ``(hex) -> List[Hex]`` returning
                           immediate grid neighbours.
        """
        self.origin: Point = origin_hex.center
        self._root = self._generate(origin_hex, get_neighbors)
        self.max_d = self._tree_max_d(self._root)
        if self.max_d < 1.0:
            self.max_d = 1.0

    # ── generation ────────────────────────────────────────────────────────────

    def _tree_max_d(self, node: _CrackNode) -> float:
        m = 0.0
        for edge in node.edges:
            m = max(m, edge.end_d, self._tree_max_d(edge.child))
        return m

    def _generate(self, origin_hex, get_neighbors: Callable) -> _CrackNode:
        used = {id(origin_hex)}

        def make_node(hex_obj, parent_dist: float, depth: int) -> _CrackNode:
            node = _CrackNode()
            if depth >= 3:
                return node

            # Number of outgoing arms depends on depth
            if depth == 0:
                max_edges = random.randint(4, 7)
            elif depth == 1:
                r = random.random()
                max_edges = 1 if r < 0.50 else (2 if r < 0.75 else 0)
            else:
                max_edges = 1 if random.random() < 0.30 else 0

            if max_edges == 0:
                return node

            neighbours = list(get_neighbors(hex_obj))
            random.shuffle(neighbours)
            count = 0

            for neighbour in neighbours:
                if count >= max_edges:
                    break

                # Possibly jump 2 hops (~20 % for shallow depth)
                target = neighbour
                if depth <= 1 and random.random() < 0.20:
                    two_hop = [n for n in get_neighbors(neighbour)
                               if id(n) not in used and n is not hex_obj]
                    if two_hop:
                        target = random.choice(two_hop)
                        used.add(id(neighbour))  # consume intermediate hex

                if id(target) in used:
                    continue
                used.add(id(target))

                cx1, cy1 = hex_obj.center
                cx2, cy2 = target.center
                dx, dy   = cx2 - cx1, cy2 - cy1
                seg_len  = math.sqrt(dx * dx + dy * dy)
                n_segs   = max(4, round(seg_len / 10))
                pts      = generate_jagged_line(
                    hex_obj.center, target.center, n_segs, 0.22
                )

                child = make_node(target, parent_dist + seg_len, depth + 1)
                node.edges.append(_CrackEdge(
                    pts, depth,
                    parent_dist, parent_dist + seg_len,
                    child,
                ))
                count += 1

            return node

        return make_node(origin_hex, 0.0, 0)

    # ── drawing ───────────────────────────────────────────────────────────────

    def draw(self, screen: pygame.Surface, progress: float) -> None:
        """Draw the crack tree with an exponential ease-out wavefront.

        Args:
            screen:   Surface to draw on.
            progress: Animation progress in [0, 1].
        """
        eased   = 1.0 - math.exp(-self.EASE_K * progress)
        front_d = eased * self.max_d * 1.05   # slight overshoot → clean finish
        self._draw_node(screen, self._root, front_d, show_tip=True)

    def draw_complete(self, screen: pygame.Surface) -> None:
        """Draw the fully-formed crack tree (static cracked state)."""
        self._draw_node(screen, self._root, float('inf'), show_tip=False)

    def _draw_node(self, screen: pygame.Surface, node: _CrackNode,
                   front_d: float, show_tip: bool) -> None:
        for edge in node.edges:
            if edge.start_d >= front_d:
                continue
            span = edge.end_d - edge.start_d
            frac = (front_d - edge.start_d) / max(0.001, span)
            frac = min(1.0, frac)
            self._draw_edge_partial(screen, edge.pts, frac, edge.depth,
                                    frac < 1.0 and show_tip)
            if frac >= 1.0:
                self._draw_node(screen, edge.child, front_d, show_tip)

    def _draw_edge_partial(self, screen: pygame.Surface, pts: List[Point],
                           frac: float, depth: int, show_tip: bool) -> None:
        total_subs = len(pts) - 1
        if total_subs < 1:
            return

        draw_to = frac * total_subs
        full    = int(draw_to)
        partial = draw_to - full

        for i in range(full):
            self._draw_sub_seg(screen, pts[i], pts[i + 1], depth)

        if full < total_subs and partial > 0.001:
            p1  = pts[full]
            p2  = pts[full + 1]
            ep  = (p1[0] + (p2[0] - p1[0]) * partial,
                   p1[1] + (p2[1] - p1[1]) * partial)
            self._draw_sub_seg(screen, p1, ep, depth)

            if show_tip and depth <= 1:
                radius = 3 if depth == 0 else 2
                pygame.draw.circle(screen, (235, 252, 255),
                                   (int(ep[0]), int(ep[1])), radius)

    def _draw_sub_seg(self, screen: pygame.Surface,
                      p1: Point, p2: Point, depth: int) -> None:
        d  = min(depth, len(self._LINE_WIDTHS) - 1)
        lw = self._LINE_WIDTHS[d]
        ip1 = (int(p1[0]), int(p1[1]))
        ip2 = (int(p2[0]), int(p2[1]))

        # Shadow
        pygame.draw.line(screen, self._SHADOW_COLOR,
                         (ip1[0] + 1, ip1[1] + 1),
                         (ip2[0] + 1, ip2[1] + 1),
                         lw + 2)
        # Dark crack body
        pygame.draw.line(screen, self._CRACK_COLORS[d], ip1, ip2, lw)

        # Ice-blue highlight for primary cracks
        if depth == 0:
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            length = math.sqrt(dx * dx + dy * dy)
            if length > 0.4:
                px = -dy / length * 1.4
                py =  dx / length * 1.4
                pygame.draw.line(screen, self._HIGHLIGHT,
                                 (int(ip1[0] + px), int(ip1[1] + py)),
                                 (int(ip2[0] + px), int(ip2[1] + py)), 1)

    # ── utility ───────────────────────────────────────────────────────────────

    def collect_segments(self) -> List[List[Point]]:
        """Return all point-list segments from every edge in the tree."""
        segments: List[List[Point]] = []

        def collect(node: _CrackNode) -> None:
            for edge in node.edges:
                segments.append(edge.pts)
                collect(edge.child)

        collect(self._root)
        return segments

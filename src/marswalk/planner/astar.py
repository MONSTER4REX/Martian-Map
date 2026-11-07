"""Generic A* over a lazily-defined graph (Assumptions A10, A11).

The graph is given by two callbacks so the same search serves every route
policy (shortest / slope-aware / full-constraint):

    neighbors(node) -> iterable of (neighbor, edge_cost)   # allowed edges only
    heuristic(node) -> admissible, non-negative estimate to goal

Ties are broken by an insertion counter, so paths are deterministic.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Hashable, Iterable

Node = Hashable


def astar(
    start: Node,
    goal: Node,
    neighbors: Callable[[Node], Iterable[tuple[Node, float]]],
    heuristic: Callable[[Node], float],
) -> tuple[list[Node] | None, float]:
    """Return (path, total_cost). path is None and cost is inf if unreachable.

    ``path`` includes both ``start`` and ``goal``. Assumes edge costs are
    non-negative and ``heuristic`` is admissible (never overestimates); under
    those conditions the returned cost is optimal.
    """
    if start == goal:
        return [start], 0.0

    counter = 0
    open_heap: list[tuple[float, float, int, Node]] = [(heuristic(start), 0.0, counter, start)]
    g_score: dict[Node, float] = {start: 0.0}
    came_from: dict[Node, Node] = {}
    closed: set[Node] = set()

    while open_heap:
        _, g, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            return _reconstruct(came_from, current), g
        closed.add(current)

        for nbr, edge_cost in neighbors(current):
            if nbr in closed:
                continue
            tentative = g + edge_cost
            if tentative < g_score.get(nbr, math.inf):
                g_score[nbr] = tentative
                came_from[nbr] = current
                counter += 1
                heapq.heappush(open_heap, (tentative + heuristic(nbr), tentative, counter, nbr))

    return None, math.inf


def _reconstruct(came_from: dict[Node, Node], current: Node) -> list[Node]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

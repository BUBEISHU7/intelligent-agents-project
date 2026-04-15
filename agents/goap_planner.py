from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple


PredicateState = FrozenSet[str]


@dataclass(frozen=True)
class Action:
    name: str
    cost: float
    preconditions: FrozenSet[str]
    add_effects: FrozenSet[str]
    del_effects: FrozenSet[str] = frozenset()

    def applicable(self, s: PredicateState) -> bool:
        return self.preconditions.issubset(s)

    def apply(self, s: PredicateState) -> PredicateState:
        nxt = set(s)
        for p in self.del_effects:
            nxt.discard(p)
        for p in self.add_effects:
            nxt.add(p)
        return frozenset(nxt)


class GoapPlanner:
    """
    Generic GOAP planner: A* search over predicate states with action costs.
    """

    def __init__(self, actions: Sequence[Action]):
        self.actions = list(actions)

    def plan(self, start: PredicateState, goal: FrozenSet[str], max_expansions: int = 5000) -> List[str]:
        def h(s: PredicateState) -> float:
            # admissible heuristic: number of unsatisfied goal predicates
            return float(len(goal.difference(s)))

        open_heap: List[Tuple[float, float, PredicateState]] = []
        heapq.heappush(open_heap, (h(start), 0.0, start))
        came_from: Dict[PredicateState, Tuple[PredicateState, str]] = {}
        g_cost: Dict[PredicateState, float] = {start: 0.0}

        expansions = 0
        while open_heap and expansions < max_expansions:
            _, gc, s = heapq.heappop(open_heap)
            if gc != g_cost.get(s, float("inf")):
                continue
            expansions += 1
            if goal.issubset(s):
                return self._reconstruct(came_from, s)
            for a in self.actions:
                if not a.applicable(s):
                    continue
                ns = a.apply(s)
                ng = gc + float(a.cost)
                if ng < g_cost.get(ns, float("inf")):
                    g_cost[ns] = ng
                    came_from[ns] = (s, a.name)
                    heapq.heappush(open_heap, (ng + h(ns), ng, ns))
        return []

    def _reconstruct(self, came_from: Dict[PredicateState, Tuple[PredicateState, str]], s: PredicateState) -> List[str]:
        out: List[str] = []
        cur = s
        while cur in came_from:
            prev, act = came_from[cur]
            out.append(act)
            cur = prev
        out.reverse()
        return out


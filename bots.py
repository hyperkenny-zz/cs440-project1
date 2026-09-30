"""Reusable search and the four policies. Policies never access environment RNG."""
from collections import deque
import heapq
import random
from ship import fire_probabilities, spread_fire


def reconstruct(parent, goal):
    path = [goal]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    return path[::-1]


def bfs(ship, start, goal, blocked):
    if start in blocked or goal in blocked:
        return None
    parent = {start: None}
    queue = deque([start])
    while queue:
        v = queue.popleft()
        if v == goal:
            return reconstruct(parent, goal)
        for n in ship.adjacent(v):
            if n not in blocked and n not in parent:
                parent[n] = v
                queue.append(n)
    return None


def astar(ship, start, goal, fire, q, beta):
    """Return (path, cost); risk is a frozen one-update proxy, not arrival risk."""
    if start in fire or goal in fire:
        return None, float('inf')
    risk = fire_probabilities(ship, fire, q)
    heuristic = lambda v: abs(v[0]-goal[0]) + abs(v[1]-goal[1])
    queue = [(heuristic(start), 0.0, start)]
    costs, parent = {start: 0.0}, {start: None}
    while queue:
        _, cost, v = heapq.heappop(queue)
        if cost != costs[v]:
            continue
        if v == goal:
            return reconstruct(parent, goal), cost
        for n in ship.adjacent(v):
            if n in fire:
                continue
            candidate = cost + 1 + (0 if n == goal else beta*risk.get(n, 0))
            if candidate < costs.get(n, float('inf')):
                costs[n], parent[n] = candidate, v
                heapq.heappush(queue, (candidate+heuristic(n), candidate, n))
    return None, float('inf')


class Bot1:
    def __init__(self, ship, start, goal, fire, q, **kwargs):
        self.path = bfs(ship, start, goal, fire)
        self.index = 0

    def choose(self, position, fire):
        if self.path is None or self.index+1 >= len(self.path):
            return position
        self.index += 1
        return self.path[self.index]


class Bot2:
    def __init__(self, ship, start, goal, fire, q, **kwargs):
        self.ship, self.goal, self.q = ship, goal, q

    def path(self, position, fire):
        return bfs(self.ship, position, self.goal, fire)

    def choose(self, position, fire):
        path = self.path(position, fire)
        return path[1] if path and len(path) > 1 else position


class Bot3(Bot2):
    def path(self, position, fire):
        blocked = set(fire)
        for v in fire:
            blocked.update(self.ship.adjacent(v))
        # Permit leaving an exposed start, but do not exempt the destination.
        blocked.discard(position)
        return bfs(self.ship, position, self.goal, blocked) or super().path(position, fire)


class Bot4(Bot2):
    def __init__(self, *args, seed=0, depth=2, samples=6, beta=4.0, **kwargs):
        super().__init__(*args, **kwargs)
        if depth < 1 or samples < 1 or beta < 0:
            raise ValueError('depth/samples must be positive; beta must be nonnegative')
        self.depth, self.samples, self.beta = depth, samples, beta
        self.rng = random.Random(seed)

    def choose(self, position, fire):
        # Each sampled fire tree is independent of robot position. Sharing it
        # across actions supplies common random numbers, without future leakage.
        def tree(current, depth):
            if depth == 0:
                return current, ()
            count = 1 if self.q in (0, 1) else self.samples
            return current, tuple(tree(spread_fire(self.ship, current, self.q, self.rng), depth-1)
                                  for _ in range(count))
        root = tree(fire, self.depth)
        leaf_cache, value_cache = {}, {}

        def evaluate(pos, burning):
            key = pos, burning
            if key not in leaf_cache:
                _, cost = astar(self.ship, pos, self.goal, burning, self.q, self.beta)
                leaf_cache[key] = 1/(1+cost)
            return leaf_cache[key]

        def value(pos, node):
            burning, children = node
            if pos in burning or self.goal in burning:
                return 0.0
            if pos == self.goal:
                return 1.0
            key = pos, id(node)
            if key not in value_cache:
                value_cache[key] = (best(pos, node)[0] if children else evaluate(pos, burning))
            return value_cache[key]

        def best(pos, node):
            burning, children = node
            actions = [v for v in self.ship.adjacent(pos) if v not in burning] + [pos]
            chosen, score = pos, -1.0
            for action in actions:
                estimate = (1.0 if action == self.goal else
                            sum(value(action, child) for child in children)/len(children))
                if estimate > score:
                    chosen, score = action, estimate
            return score, chosen

        return best(position, root)[1]


BOTS = {1: Bot1, 2: Bot2, 3: Bot3, 4: Bot4}

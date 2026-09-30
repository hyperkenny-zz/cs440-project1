"""Ship generation and synchronous fire dynamics; standard library only."""
from dataclasses import dataclass
import random

Cell = tuple[int, int]

@dataclass(frozen=True)
class Ship:
    size: int
    open_cells: frozenset[Cell]

    def neighbors(self, cell):
        r, c = cell
        return tuple(v for v in ((r-1,c), (r+1,c), (r,c-1), (r,c+1))
                     if 0 <= v[0] < self.size and 0 <= v[1] < self.size)

    def adjacent(self, cell):
        return tuple(v for v in self.neighbors(cell) if v in self.open_cells)


def generate_ship(size, rng):
    if size < 3:
        raise ValueError('Ship size must be at least 3')
    grid = Ship(size, frozenset())
    opened = {(rng.randrange(1, size-1), rng.randrange(1, size-1))}
    # Indexed candidate pool gives uniform selection and O(1) removal.
    pool, index = [], {}
    def remove(v):
        i = index.pop(v)
        last = pool.pop()
        if i < len(pool):
            pool[i] = last
            index[last] = i
    def refresh(v):
        eligible = v not in opened and sum(n in opened for n in grid.neighbors(v)) == 1
        if eligible and v not in index:
            index[v] = len(pool)
            pool.append(v)
        elif not eligible and v in index:
            remove(v)
    for v in grid.neighbors(next(iter(opened))):
        refresh(v)
    while pool:
        v = pool[rng.randrange(len(pool))]
        remove(v)
        opened.add(v)
        for n in grid.neighbors(v):
            refresh(n)
    def dead_ends():
        return sorted(v for v in opened if sum(n in opened for n in grid.neighbors(v)) == 1)
    dead = dead_ends()
    target = len(dead) // 2
    while len(dead) > target:
        eligible = [v for v in dead if any(n not in opened for n in grid.neighbors(v))]
        if not eligible:
            raise RuntimeError('Cannot reduce dead ends further')
        v = rng.choice(eligible)
        opened.add(rng.choice([n for n in grid.neighbors(v) if n not in opened]))
        dead = dead_ends()
    return Ship(size, frozenset(opened))


def fire_probabilities(ship, fire, q):
    counts = {}
    for v in sorted(fire):
        for n in ship.adjacent(v):
            if n not in fire:
                counts[n] = counts.get(n, 0) + 1
    return {v: 1 - (1-q)**k for v, k in counts.items()}


def spread_fire(ship, fire, q, rng):
    # Calculate from the old fire only: newly ignited cells cannot spread yet.
    probabilities = fire_probabilities(ship, fire, q)
    return frozenset(set(fire) | {v for v in sorted(probabilities)
                                if rng.random() < probabilities[v]})


def render(ship, bot, button, fire):
    return '\n'.join(''.join('X' if (r,c) == bot and (r,c) in fire else
                             'R' if (r,c) == bot else 'F' if (r,c) in fire else
                             'B' if (r,c) == button else '.' if (r,c) in ship.open_cells else '#'
                             for c in range(ship.size)) for r in range(ship.size))

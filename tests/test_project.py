import random
import unittest
from bots import Bot3, Bot4, astar, bfs
from main import run_episode, scenario
from ship import Ship, generate_ship, spread_fire, fire_probabilities


class ProjectTests(unittest.TestCase):
    def test_generation_connected_and_reproducible(self):
        for size in (3, 8, 20):
            for seed in range(10):
                ship = generate_ship(size, random.Random(seed))
                self.assertEqual(ship, generate_ship(size, random.Random(seed)))
                reached, todo = set(), [next(iter(ship.open_cells))]
                while todo:
                    v = todo.pop()
                    if v not in reached:
                        reached.add(v)
                        todo.extend(ship.adjacent(v))
                self.assertEqual(reached, set(ship.open_cells))

    def test_synchronous_spread_and_endpoints(self):
        ship = Ship(4, frozenset((0,c) for c in range(4)))
        fire = frozenset({(0,0)})
        self.assertEqual(spread_fire(ship, fire, 0, random.Random(0)), fire)
        self.assertEqual(spread_fire(ship, fire, 1, random.Random(0)), {(0,0),(0,1)})
        self.assertAlmostEqual(fire_probabilities(ship, {(0,0),(0,2)}, .5)[(0,1)], .75)

    def test_astar_matches_bfs_without_risk(self):
        for seed in range(10):
            ship, start, goal, fire = scenario(10, seed)
            path = bfs(ship, start, goal, {fire})
            a, cost = astar(ship, start, goal, {fire}, .5, 0)
            self.assertEqual(a is None, path is None)
            if path:
                self.assertEqual(cost, len(path)-1)

    def test_astar_avoids_high_risk_route(self):
        cells = {(1,c) for c in range(5)} | {(2,c) for c in range(5)} | {(0,2)}
        ship = Ship(5, frozenset(cells))
        path, cost = astar(ship, (1,0), (1,4), {(0,2)}, 1, 10)
        self.assertNotIn((1,2), path)
        self.assertEqual(cost, 6)

    def test_button_wins_before_spread(self):
        ship = Ship(3, frozenset({(0,0),(0,1),(0,2)}))
        for bot in range(1,5):
            result = run_episode(ship, (0,0), (0,1), (0,2), 1, bot, 0)
            self.assertTrue(result['success'])
            self.assertEqual(result['steps'], 1)

    def test_bot3_can_leave_exposed_start(self):
        ship = Ship(4, frozenset({(0,0),(1,0),(1,1),(1,2)}))
        bot = Bot3(ship, (1,0), (1,2), {(0,0)}, .5)
        self.assertEqual(bot.choose((1,0), {(0,0)}), (1,1))

    def test_bot4_avoids_certain_death(self):
        ship = Ship(5, frozenset({(1,c) for c in range(5)} | {(2,c) for c in range(5)} | {(0,1)}))
        bot = Bot4(ship, (1,0), (1,4), {(0,1)}, 1, depth=1)
        self.assertEqual(bot.choose((1,0), frozenset({(0,1)})), (2,0))

    def test_no_fire_growth_all_bots_reach_reachable_goal(self):
        for seed in range(4):
            ship, start, goal, fire = scenario(8, seed)
            if bfs(ship, start, goal, {fire}):
                for bot in range(1,5):
                    result = run_episode(ship, start, goal, fire, 0, bot, 1, max_steps=100)
                    self.assertTrue(result['success'], (seed, bot, result))

    def test_environment_randomness_independent_of_planner(self):
        ship, start, goal, fire = scenario(10, 12)
        a = run_episode(ship,start,goal,fire,.3,2,77,trace=True)
        b = run_episode(ship,start,goal,fire,.3,4,77,trace=True)
        # A successful move ends before a fire update; compare continuing rounds.
        for x,y in zip(a['history'][:-1],b['history'][:-1]):
            self.assertEqual(x['fire'],y['fire'])


if __name__ == '__main__':
    unittest.main()

"""Run one episode or paired experiments: python3 main.py --help."""
import argparse
import csv
import json
import math
from pathlib import Path
import random
import time
from bots import BOTS, bfs
from ship import generate_ship, spread_fire, render


def run_episode(ship, start, goal, initial_fire, q, bot_number, fire_seed,
                planning_seed=0, depth=2, samples=6, beta=4.0, max_steps=1000, trace=False):
    if not 0 <= q <= 1:
        raise ValueError('q must be between 0 and 1')
    if len({start, goal, initial_fire}) != 3 or not {start, goal, initial_fire} <= ship.open_cells:
        raise ValueError('Initial positions must be distinct open cells')
    fire, position = frozenset({initial_fire}), start
    rng = random.Random(fire_seed)
    bot = BOTS[bot_number](ship, start, goal, fire, q, seed=planning_seed,
                          depth=depth, samples=samples, beta=beta)
    history = []
    def record(step):
        if trace:
            history.append({'step': step, 'position': position, 'fire': sorted(fire)})
    record(0)
    elapsed, steps, success, reason = 0.0, 0, False, 'step_limit'
    for step in range(1, max_steps+1):
        if goal in fire:
            reason = 'button_burned'
            break
        if bfs(ship, position, goal, fire) is None:
            reason = 'no_path'
            break
        before = time.perf_counter()
        action = bot.choose(position, fire)
        elapsed += time.perf_counter()-before
        if action != position and action not in ship.adjacent(position):
            raise RuntimeError('Bot returned an illegal action')
        position, steps = action, step
        if position in fire:
            reason = 'entered_fire'
            record(step)
            break
        if position == goal:
            success, reason = True, 'success'
            record(step)
            break
        fire = spread_fire(ship, fire, q, rng)
        record(step)
        if position in fire:
            reason = 'burned_after_move'
            break
    result = dict(bot=bot_number, q=q, success=success, reason=reason,
                  steps=steps, decision_seconds=elapsed,
                  mean_decision_seconds=elapsed/steps if steps else 0)
    if trace:
        result.update(size=ship.size, open_cells=sorted(ship.open_cells), button=goal, history=history)
    return result


def scenario(size, seed):
    rng = random.Random(seed)
    ship = generate_ship(size, rng)
    start, goal, fire = rng.sample(sorted(ship.open_cells), 3)
    return ship, start, goal, fire


def wilson(successes, count):
    p, z = successes/count, 1.96
    denominator = 1+z*z/count
    center = (p+z*z/(2*count))/denominator
    half = z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/denominator
    return max(0, center-half), min(1, center+half)


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['demo', 'experiment'])
    parser.add_argument('--size', type=int, default=15)
    parser.add_argument('--seed', type=int, default=440)
    parser.add_argument('--bot', type=int, choices=BOTS, default=4)
    parser.add_argument('--q', type=float, default=0.3)
    parser.add_argument('--qs', type=float, nargs='+', default=[i/10 for i in range(11)])
    parser.add_argument('--trials', type=int, default=20)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--samples', type=int, default=6)
    parser.add_argument('--beta', type=float, default=4)
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--output', type=Path, default=Path('results'))
    args = parser.parse_args()
    if args.size < 3 or min(args.trials, args.depth, args.samples, args.max_steps) < 1 or args.beta < 0:
        parser.error('Invalid size, count, or beta')
    if any(not 0 <= q <= 1 for q in [args.q, *args.qs]):
        parser.error('q must lie in [0,1]')
    args.output.mkdir(parents=True, exist_ok=True)
    settings = dict(depth=args.depth, samples=args.samples, beta=args.beta, max_steps=args.max_steps)
    if args.mode == 'demo':
        ship, start, goal, fire = scenario(args.size, args.seed)
        result = run_episode(ship, start, goal, fire, args.q, args.bot, args.seed+1,
                             planning_seed=args.seed+2, trace=True, **settings)
        for state in result['history']:
            print(f"\nStep {state['step']}")
            print(render(ship, state['position'], goal, set(state['fire'])))
        print({k:v for k,v in result.items() if k not in ('history', 'open_cells')})
        (args.output/'demo.json').write_text(json.dumps(result, indent=2))
        return
    rows, summaries = [], []
    for q in args.qs:
        for trial in range(args.trials):
            seed = args.seed + trial*1000003
            ship, start, goal, fire = scenario(args.size, seed)
            for bot in BOTS:
                result = run_episode(ship, start, goal, fire, q, bot, seed+1,
                                     planning_seed=seed+2, **settings)
                result.update(trial=trial, scenario_seed=seed, fire_seed=seed+1, planning_seed=seed+2)
                rows.append(result)
        for bot in BOTS:
            group = [r for r in rows if r['q'] == q and r['bot'] == bot]
            successes = sum(r['success'] for r in group)
            low, high = wilson(successes, len(group))
            summaries.append(dict(q=q, bot=bot, trials=len(group), success_rate=successes/len(group),
                                  ci_low=low, ci_high=high,
                                  step_limits=sum(r['reason']=='step_limit' for r in group),
                                  mean_decision_seconds=sum(r['decision_seconds'] for r in group)/max(1,sum(r['steps'] for r in group))))
        print(f'q={q:g} finished', flush=True)
    write_csv(args.output/'episodes.csv', rows)
    write_csv(args.output/'summary.csv', summaries)
    (args.output/'config.json').write_text(json.dumps(vars(args), default=str, indent=2))
    print(f'Saved results to {args.output}')


if __name__ == '__main__':
    main()

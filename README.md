# Project 1 — This Ship is on Fiiiiire!

Requires Python 3.10 or later. Simulation and search use only the Python standard library; plotting requires matplotlib.

## Running the project

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m unittest discover -s tests -v
python3 main.py demo --size 15 --bot 4 --q 0.3 --seed 440
python3 main.py experiment --size 15 --trials 20 --qs 0 0.2 0.4 0.6 0.8 1 --output results/pilot
python3 -m pip install -r requirements.txt
python3 plot_results.py results/pilot/summary.csv
```

Terminal map symbols: `#` wall, `.` open cell, `R` robot, `B` button, `F` fire, and `X` a robot caught in fire.

Demo mode also saves each step's state in `demo.json`. Experiments save individual episode results in `episodes.csv`, success frequencies and 95% Wilson confidence intervals in `summary.csv`, and configuration parameters in `config.json`. Small pilot runs verify the workflow; they do not establish that one strategy outperforms another.

## Project structure and strategies

- `ship.py`: connected ship generation, dynamic dead-end reduction, and synchronous fire updates.
- `bots.py`: BFS, A*, and Bots 1–4.
- `main.py`: episode execution, reproducible experiments, logging, and statistics.
- `plot_results.py`: success-rate plots.
- `tests/test_project.py`: checks for generation, search, fire dynamics, and turn ordering.

Bot 1 follows its initial BFS path without replanning. Bot 2 runs BFS again at every step. Bot 3 first avoids burning cells and their neighbors, falling back to Bot 2 when no such path exists. It allows departure from a starting cell adjacent to fire, but does not exempt the button from the buffer restriction.

Bot 4 uses finite-depth sampled Expectimax. Robot nodes maximize the score; fire nodes average outcomes sampled from the fire transition distribution, without multiplying by their probabilities again. By default, it looks ahead two complete turns and draws six samples per chance node. Reaching the button succeeds immediately, before another fire update.

Terminal states receive a score of 1 for success and 0 for robot death or a burning button. At the search horizon, an unreachable button receives 0; other states receive `1/(1+C)`. Here, C is the minimum A* path cost, with each step costing `1 + beta * p(v)`, where beta defaults to 4 and `p(v)=1-(1-q)^K`. Entering the button incurs only the movement cost because pressing it immediately extinguishes the fire. A* uses Manhattan distance as its heuristic.

The probability p describes only the next fire update under the current fire configuration. The full path cost is a heuristic proxy, not the actual risk at each future arrival time; the leaf score is not a true success probability either. The algorithm is not guaranteed to be optimal or outperform the other bots. Use `--beta 0` for an ablation that removes the risk penalty.

Each decision builds a new prediction tree. Candidate actions at the same fire node share fire samples. Prediction randomness is separate from actual environment randomness. Future actions depend on the states observed along the simulated tree. Tree size grows exponentially with depth: increase `--samples` modestly first and use caution when increasing `--depth`. Repeated leaf evaluations and node values are cached within a decision. Deterministic transitions at q=0 or q=1 use only one branch.

## Experimental conventions

Within each trial, all four bots use the same ship, robot start, button, initial fire, and environment seed. Fire updates are independent of robot actions and consume random numbers in sorted candidate-cell order, so the bots experience identical fire histories over shared continuing turns. A successful turn ends before the next fire update. Maps are reused across q values, but different q values may produce different fire candidate sets, so random draws are not guaranteed to align cell by cell across q values.

Death takes precedence over button success: entering an already burning button is a failure. Episodes end early if no fire-free path remains or the button has burned, since fire never recedes. The default `--max-steps` is 1000. Episodes exceeding this limit are labeled `step_limit` and currently count as unsuccessful in the reported frequency. Formal experiments must check for and eliminate or explain these timeouts; they must not be interpreted as actual deaths.

Reusing seeds reproduces trajectories, but not wall-clock timings. To replay an episode, pass its scenario_seed as the demo's `--seed`, together with the same q, bot, and search parameters. Use separate seed ranges for tuning and final evaluation. Record failed trajectories to analyze their causes. Retrospective omniscient temporal-path diagnosis and the bonus layout optimization are not yet implemented.

Start with a coarse scan of q, then sample more densely where strategies differ. Increase the number of independent maps and report success frequencies, confidence intervals, and average decision times together. Choose sample counts and map sizes based on pilot runtimes; the defaults are not assignment requirements.

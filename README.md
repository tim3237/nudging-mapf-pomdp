# Nudging-Based MAPF as POMDP
This repository implements a nudging-based variant of the Multi-Agent Pathfinding Problem (MAPF). In this setting, agents move autonomously to their destination, and the policy cannot directly control them. Instead, it can send messages that suggest intermediate goals for each agent at each timestep. Because agents' private goals are not directly accessible to the policy, the environment is modelled as a partially observable Markov decision process (POMDP). Suggested goals are chosen from a set of points of interest (POIs), and messages may also be empty or instruct an agent to drop its current intermediate goal.

The simulation has a preset room layout but also allows to generate random layouts such as the following:
Preset Layout                                                                                     | Random Layout                                                                                      | Random Layout
--------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------
![Preset Layout](https://github.com/tim3237/experiments/blob/nmapf-env/layouts/base.png?raw=true) | ![Preset Layout](https://github.com/tim3237/experiments/blob/nmapf-env/layouts/rand1.png?raw=true) | ![Preset Layout](https://github.com/tim3237/experiments/blob/nmapf-env/layouts/rand2.png?raw=true)

Coloured areas indicate POIs. POIs in outer walls can be selected as individual destinations; agents despawn once they reach a corresponding position.

Trivial policies include doing nothing at all, i.e., always sending empty messages or doing nothing most of the time while sending random messages with low probability. We denote a randomised policy `rand-x` if it produces random messages with probability `0.x`. In the following simulations, agents are shown in green if they have an intermediate destination; otherwise, they are blue.
`no-op` policy (do nothing)                                                                        | `rand-001` policy (provide random messages with probability `0.001`)
---------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------
![no-op Rollout](https://github.com/tim3237/experiments/blob/nmapf-env/rollouts/no-op.gif?raw=true) | ![rand-001 Rollout](https://github.com/tim3237/experiments/blob/nmapf-env/rollouts/rand-001.gif?raw=true)

While the `no-op` policy leads to congestions that are never resolved, `rand` policies can at least resolve the situation eventually, allowing all agents to reach their destinations. The following plot shows the minimum, maximum, and average number of active agents over 10,000 steps from 100 rollouts per policy:
![Plot of numbers of unfinished agents per simulation step.](https://github.com/tim3237/experiments/blob/nmapf-env/trivial-policy-stats.png?raw=true)

The policies above leave room for improvement. A reinforcement-learning approach can be found [here][numuzero].

[numuzero]: https://github.com/tim3237/NuMuZero

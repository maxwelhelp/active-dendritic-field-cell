# 014b — Predator/Morphology ADFC Ecosystem

Goal: make the world less like free food collection and more like competition/adaptation.

## Files

- Code: `adfc/ecosim_predator_morph.py`
- Visual predator: `run_014b_predator_visual.sh`
- Headless predator: `run_014b_predator_headless.sh`
- Gentler evolution visual: `run_014b_predator_evolve.sh`

## Changes vs 014

- static cheap food is removed
- prey-food flees from agents
- agents can bite/steal energy from other agents
- each agent has inherited morphology
- brain controls tools: sprint, shield, spike extension, bite charge, mouth
- morphology controls speed, size, bite force, bite radius, armor, digestion, fertility, sense
- morphology is inherited through crossover and mutation
- each agent has its own independent MicroBrain
- lifetime plasticity reinforces neural paths that increase energy

## Actions

The brain outputs:

- turn
- thrust
- bite
- mate
- rest
- sprint
- shield
- extend spike/reach
- charge bite
- mouth/open eat

## Smoke result

Old 014 had too much free food: food stayed near max and population collapsed to min agents.

014b smoke is harsher:

- 300-step initial smoke: 17 kills, 0 births, population fell to min
- tuned 1000-step smoke: 21 kills, 1 birth, population stayed near min

Interpretation: predator pressure works, but the default predator mode is harsh. Use `run_014b_predator_evolve.sh` for watching generations.

## Metrics

CSV logs include:

- agents, prey, births, deaths, kills
- energy mean/max
- neural activity
- alive connection fraction
- max generation
- morphology means: size, speed, force, radius, armor, sense

## Commands

Hard predator visual:

```bash
./run_014b_predator_visual.sh
```

Gentler evolution visual:

```bash
./run_014b_predator_evolve.sh
```

Headless quick run:

```bash
./run_014b_predator_headless.sh
```

## Update: dynamic shape and closed learning loop

The newer patch makes shape an action, not only inherited static morphology.

The brain now also outputs dynamic shape allocation:

- `shape_speed`
- `shape_tool`
- `shape_guard`
- `shape_sense`

These four values are normalized into a per-step body allocation. The inherited morphology defines maximum potential, while the brain chooses how to spend the current body/attention budget.

Examples:

- high speed allocation increases movement but costs more
- high tool allocation increases reach/impact but costs more
- high guard allocation improves protection but costs more
- high sense allocation increases perception range but costs more

The learning loop is now:

```text
sensors -> individual MicroBrain -> action + shape allocation -> world result -> energy/progress reward -> lifetime plasticity -> changed brain gates/weights
```

The evolutionary loop is:

```text
survive + reproduce -> brain/morph crossover -> mutation -> next generation
```

The objective is not a supervised label. The objective is survival/reproduction under resource pressure:

```text
maximize energy intake
minimize activity/connection/tool/body cost
survive long enough to reproduce
```

Dense reward was added because pure energy reward is too sparse. Agents now receive lifetime plasticity signal from:

- direct energy change
- progress toward nearby resource/target

New CSV metrics:

- `reward_mean`
- `gain_mean`
- `shape_speed_mean`
- `shape_tool_mean`
- `shape_guard_mean`
- `shape_sense_mean`
- `act_move_mean`
- `act_tool_mean`
- `act_pair_mean`
- `act_rest_mean`

Smoke after dynamic shape showed shape allocation is active and logged. The environment is still harsh, so use evolve mode for long generation runs.

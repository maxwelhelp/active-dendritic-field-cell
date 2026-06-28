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

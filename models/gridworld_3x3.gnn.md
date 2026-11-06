# GNN model: pymcp 3x3 gridworld (reference test instance)

## GNNSection

ActInfPOMDP

## GNNVersionAndFlags

GNN v1

## ModelName

pymcp 3x3 Gridworld

## ModelAnnotation

A 3x3 fully observable gridworld with a binary reward observation. State
indices are row-major: state 0 is the top-left cell and state 8 is the
bottom-right cell. Actions are 0 UP, 1 RIGHT, 2 DOWN, 3 LEFT; a move that
would leave the grid is clamped to the current edge cell.

The model has one hidden-state factor with nine states and two observation
modalities: a position observation with nine outcomes and a reward observation
with two outcomes (0 no reward, 1 reward at state 8). The position modality is
an identity likelihood; the reward modality is deterministic given the state.
Preferences are neutral over positions and prefer reward. The initial state
prior is uniform over the nine cells.

This is the deterministic test/reference model used by
`tests/test_pymdp_interface.py::test_visualize_simulation`, aligned with the
four-action `src/mcp/environments/gridworld.py` environment. It deliberately
does not express the random `define_generative_model()` builder or the
five-action `create_gridworld_agent()` path; those are construction helpers,
not this fixed executable model. It also omits policy selection, expected free
energy, learning and the visualization/server plumbing. The reference test
runs the model for 30 steps.

## StateSpaceBlock

s_f0[9,1,type=float]      # Hidden state factor 0: row-major grid cell
o_m0[9,1,type=int]        # Observation modality 0: cell position
o_m1[2,1,type=int]        # Observation modality 1: reward (0 no, 1 yes)
u[1,type=int]             # Action: 0 UP, 1 RIGHT, 2 DOWN, 3 LEFT

A_m0[9,9,type=float]      # Position likelihood: identity P(o_m0 | s_f0)
A_m1[2,9,type=float]      # Reward likelihood: P(o_m1 | s_f0)
B_f0[9,9,4,type=float]    # Transitions: next_state, previous_state, action
C_m0[9,type=float]        # Position preferences: neutral
C_m1[2,type=float]        # Reward preferences: 0 no reward, 1 reward
D_f0[9,type=float]        # Initial state prior: uniform
t[1,type=int]             # Discrete time step

## Connections

D_f0>s_f0
s_f0-A_m0
A_m0>o_m0
s_f0-A_m1
A_m1>o_m1
s_f0>B_f0
u>B_f0
B_f0>s_f0
C_m0-o_m0
C_m1-o_m1

## InitialParameterization

# A_m0[observation, state]: position is observed exactly.
A_m0={
  (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
  (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
  (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
  (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
  (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
  (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
  (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
  (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
  (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
}

# A_m1[observation, state]: every cell except state 8 reports no reward;
# state 8 reports reward.
A_m1={
  (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0),
  (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
}

# B_f0[next_state, previous_state, action]: each outer tuple is one next-state
# slice; within a slice rows are previous states and columns are actions.
# Actions are 0 UP, 1 RIGHT, 2 DOWN, 3 LEFT, clamped at the grid edge.
B_f0={
  (
    (1.0, 0.0, 0.0, 1.0),
    (0.0, 0.0, 0.0, 1.0),
    (0.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0)
  ),
  (
    (0.0, 1.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
    (0.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0)
  ),
  (
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0)
  ),
  (
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
    (0.0, 0.0, 0.0, 1.0),
    (0.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0)
  ),
  (
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
    (0.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0)
  ),
  (
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0, 0.0)
  ),
  (
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 1.0),
    (0.0, 0.0, 0.0, 1.0),
    (0.0, 0.0, 0.0, 0.0)
  ),
  (
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0)
  ),
  (
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 1.0, 1.0, 0.0)
  )
}

# C_m0 is neutral over the nine position observations.
C_m0={
  (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
}

# C_m1 is a log-preference vector, not a probability vector: consumers softmax it.
C_m1={
  (0.0, 1.0)
}

# D_f0 is uniform over the nine cells.
D_f0={
  (0.111111111111111, 0.111111111111111, 0.111111111111111, 0.111111111111111, 0.111111111111111, 0.111111111111111, 0.111111111111111, 0.111111111111111, 0.111111111111111)
}


## Equations

A_m0[o,s] = 1 when o = s, else 0.
A_m1[0,s] = 1 for s = 0..7; A_m1[1,8] = 1; all other entries are 0.
B_f0[s',s,a] = 1 when s' is the clamped result of applying action a from s.
Every B_f0[:,s,a] column sums to 1 over s'.
C_m0[o] = 0; C_m1 = (0,1) are log preferences, not normalized probabilities.
D_f0[s] = 1/9.
Belief update: q(s) proportional to D_f0[s] * A_m0[o_m0,s] * A_m1[o_m1,s].

## Time

Dynamic
DiscreteTime=t
ModelTimeHorizon=30

## ActInfOntologyAnnotation

s_f0=HiddenStateFactor0
o_m0=ObservationModality0
o_m1=ObservationModality1
u=Action
A_m0=LikelihoodMatrixModality0
A_m1=LikelihoodMatrixModality1
B_f0=TransitionMatrixFactor0
C_m0=PreferenceVectorModality0
C_m1=PreferenceVectorModality1
D_f0=PriorFactor0
t=Time

## ModelParameters

num_hidden_states: 9
num_obs: 11
num_obs_modality0: 9
num_obs_modality1: 2
num_actions: 4
num_timesteps: 30
num_modalities: 2
num_factors: 1
b_tensor_order: next_state_previous_state_action

## Footer

pymcp 3x3 gridworld v1.0 - deterministic reference test instance.
One state factor, two observation modalities, four clamped actions.

## Signature

Cryptographic signature goes here

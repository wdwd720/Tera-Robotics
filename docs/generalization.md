# Policy and generalization strategy

## Two levels, on purpose

- A **coarse policy** handles perception-driven approach and gross insertion: find the
  drive, find the bay, get the connector to the mouth and most of the way in.
- A **compliant low-level controller** (Cartesian impedance plus force search) handles the
  last few millimeters of seating from force feedback.

Keeping the hard contact-rich part in physics-aware control, not in the network, means far
less data and far better generalization: the network never has to learn sub-millimeter
force control. `policies/infer.py` implements the handoff (`TwoLevelController`): the
learned coarse action drives the HAL until `near_mouth`, then `insert_with_search` (the
same primitive the scripted expert uses) does the seat.

## Engineered generalization

Generalization is engineered through domain randomization over the axes that vary in a real
datacenter (`sim/domain_randomization.py`): bay pose and height, drive pose, connector
clearance and friction, drive mass, lighting, and camera pose. Later: drive form factor
(3.5, 2.5, U.2, U.3) and cable type.

The eval suite (`eval/`) measures this directly: a train-distribution set, a held-out set
with pose ranges beyond training (generalization score = held-out seat rate / train seat
rate), and an induced-failure recovery set (imperfect perception the force-search must
recover from). The scripted baseline currently scores 100% train, 100% held-out
(generalization 1.0), and recovers in-plane perception error up to about the slot clearance.

## Learning path

Imitation first: ACT, then Diffusion, on scripted-expert demos. Then language conditioning
("reseat the drive in bay three") for multi-task via SmolVLA or pi0. Then optionally
residual RL or HIL-SERL to exceed the scripted expert on the trickiest seats. The scripted
force-search controller stays as the permanent low-level seater throughout.

## Sim-to-real

Two pillars: domain randomization, and the fact that the low-level controller is
force-based, which transfers far better than vision-only policies. Real teleop data on the
b601 later fine-tunes the coarse policy. Real-Time Chunking smooths flow-matching policy
execution under real control latency.

This converges with the broader VLA direction: LeRobot wraps GR00T N1.5 and similar, so the
coarse policy can later be a fine-tuned foundation backbone on our own dataset, and
interpretability or activation-targeting work plugs in at that backbone.

## Known limitation

The bay slot has square edges (no lead-in chamfer), so lateral/vertical recovery is bounded
by the ~1.5 mm clearance. A chamfered mouth (realistic for connectors) would widen the
recovery envelope and is a natural next hardening step.

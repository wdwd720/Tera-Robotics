# Safety

Every action and every joint torque passes through `SafetyController`
(`robot_core/safety.py`). The same controller runs in sim and on hardware. It is pure
numpy with no backend dependency.

## Limits (`SafetyLimits`)

- `max_dpos` (0.02 m): per-tick position delta, clamped by norm.
- `max_drot` (0.10 rad): per-tick rotation delta, clamped by norm.
- `workspace_min/max`: a base-frame box containing the cradle, the bay, and the home pose.
  Target poses are clamped into it.
- `tau_limit` (6,): per-joint torque ceiling, read from the model actuator ctrlrange at
  construction so it cannot drift from the scene.
- `vel_estop` (6 rad/s): soft e-stop on excessive joint velocity.
- `ft_estop` (60 N): soft e-stop on excessive wrist force norm.
- `insert_force_max` (25 N): bounded insert-axis push for the force-search primitive.

## Flow

`step(action)` clamps the action, integrates and clamps the target pose to the workspace
box, computes impedance torque, clamps the torque, and substeps the physics. Each substep
checks the e-stop on joint velocity and wrist force; a trip latches `estop_reason` and
breaks the loop. The force-search primitive additionally respects `insert_force_max` and
backs off rather than pushing past it.

## Seat detection

A single `seated()` hook. In sim, seated means the `drive_connector` site is within 12 mm
of `bay_socket` and the connector has progressed past `bay_mouth`. On hardware, seat is
inferred from the force signature (insertion-force drop after the detent) plus perception,
behind the same hook so both share the call site.

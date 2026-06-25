# Vision

reBot builds the learning, control, data, and autonomy stack for datacenter maintenance
robotics. We are a software company that also builds custom hardware: the software stack is
the product and the IP; the custom hardware (the b601 arm) is the embodiment and the
differentiator, and we need an embodiment to demo.

## The wedge

Autonomous hard-drive reseating: a robot arm grasps a drive and seats it into a bay
connector with millimeter precision under force feedback. The task ladder is drive reseat,
then drive swap, then RJ45 cable reseat, then DAC cable reseat. We build the first task.

## Why torque-first

The b601 is a 6-DoF arm with MIT-mode torque control. That torque control is the whole
asset: it lets us command a force field around a target pose so the arm yields on contact
and searches for the connector, instead of jamming a position into a wall. Pure position
control cannot do compliant connector seating. Everything is built torque-first.

## Prime directive: one hardware abstraction layer

There is exactly one `Robot` interface; every backend implements it (MuJoCo sim now, the
custom b601 later, a borrowed off-the-shelf arm as a third). Nothing above the HAL knows
which backend runs. Because of this layer the same demo runs in sim now and on the b601
later with zero changes above the HAL, which defers the B2B versus full-stack decision into
code structure. See `docs/hardware_abstraction.md`.

## Approach

A model is worthless until you can feed it clean data, execute its actions safely, and
measure success. So the build order is: harness (sim, HAL, safety, control), then a
scripted force-search reseat skill (baseline, data generator, and permanent low-level
controller), then the data engine, then eval, then learned policies. There is a working
drive-reseat demo before any neural network exists. The learned policy later replaces the
coarse approach and adds generalization; the force-search controller stays as the
low-level seater. See `docs/architecture.md`, `docs/generalization.md`, `docs/roadmap.md`.

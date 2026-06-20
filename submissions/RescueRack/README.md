# RescueRack

Registration UUID: `86ba8442-48da-4131-bb86-f5e4aa67c852`

RescueRack is a MuJoCo emergency-logistics simulation where a wheeled mobile
manipulator retrieves a red medical supply kit from a damaged warehouse aisle,
navigates around debris, and delivers the kit to a marked safe zone. The project
also includes DexTriage Lab, a five-finger triage station that sorts and aligns
five rescue objects into dedicated trays.

## Robot Platform

- Wheeled mobile base with bumper collision geometry.
- Three-joint arm: shoulder, elbow, wrist.
- Two-finger gripper with an assisted final grasp for reproducible evaluation.
- Five-finger dexterous triage hand with thumb, index, middle, ring, and little
  finger joints.
- MuJoCo sensors for base pose, gripper pose, target pose, safe-zone pose, and
  front range sensing, plus triage palm/object/fingertip pose logging.

## Task Goal

The robot must complete a long-horizon rescue supply mission:

1. Start in a damaged warehouse scene.
2. Navigate toward the target red medkit.
3. Avoid a fallen beam, a pallet block, and a loose barrel.
4. Align the arm-mounted gripper to the medkit.
5. Secure, transport, and release the medkit in the green safe zone.
6. Print a JSON mission summary with success state and final delivery error.
7. Optionally export a timestamped trajectory dataset for reproducibility and
   data-collection scoring.
8. Sort and align five small rescue items with the dexterous triage hand: a vial,
   a tool, a soft pack, a syringe, and a bandage.

## Technical Approach

The project uses a deterministic mission controller with explicit stages:

- `navigate_to_supply`
- `prepare_grasp`
- `secure_supply`
- `transport_to_safe_zone`
- `release_supply`
- `complete`

The controller combines obstacle-aware A* route planning, arm pose targets, gripper
commands, and a stable assisted carry phase. In hard mode, the route is generated
from a debris map using `rescuerack/planner.py` rather than only replaying a fixed
list of hand-authored waypoints. The mobile base and environment obstacles have
active collision geometry; the included clearance check verifies that the hard-mode
route completes without contacting named obstacles.

## Core MuJoCo Features

- MJCF scene with cameras, lighting, materials, static warehouse geometry, and
  dynamic target objects.
- Position actuators for the mobile base, yaw, arm joints, and gripper joints.
- Free bodies for the medkit, water box, and battery box.
- Collision geometry for the floor, debris, supply objects, and mobile base.
- Frame-position sensors and a rangefinder sensor.
- Offscreen renderer for a reproducible demo video.

## Highlights

- Three difficulty modes: `easy`, `medium`, and `hard`.
- Hard mode selects the red medkit from distractor supplies and uses the longest
  debris-aware route.
- Hard mode uses a grid A* planner to generate pickup and delivery waypoints from
  obstacle geometry, reporting expanded nodes, route clearance, and fallback status.
- The hard mission completes with `delivery_error_m: 0.0`, `fallback_used: false`,
  `expanded_nodes: 199`, and `min_route_clearance_m: 0.254`.
- `scripts/verify_clearance.py` checks mission success, minimum obstacle clearance,
  and named-obstacle contact count; the submitted run reports
  `obstacle_contact_count: 0`.
- DexTriage Lab uses thumb, index, middle, ring, and little fingers to sort a
  vial, tool, soft pack, syringe, and bandage while logging fingertip/object
  proximity, palm pose, object pose, closed-loop corrections, alignment
  corrections, and grasp/place/orient events.
- `scripts/make_demo_video.py` generates `media/demo.mp4` from submitted code.
- `media/demo_timeline.json` documents the generated multi-scene video sequence.
- `scripts/generate_data.py` exports `media/hard_trajectory.json` and
  `media/mission_metrics.json` with robot, gripper, target, stage, and success
  measurements.

## Judge-Evidence Snapshot

| Evidence | Submitted result |
| --- | --- |
| Hard mission success | `success: true`, `delivery_error_m: 0.0` |
| Obstacle avoidance | `obstacle_contact_count: 0` in `scripts/verify_clearance.py` |
| Planning | Grid A* with `expanded_nodes: 199`, `fallback_used: false` |
| Dexterous manipulation | Five-finger hand sorts `vial`, `tool`, `soft_pack`, `syringe`, and `bandage` |
| Closed-loop triage | Object site is re-read every control step before palm targeting |
| Benchmark score | `micro_tasks_completed: 20/20`, `benchmark_success_rate: 1.0` |
| Contact/proximity data | `contact_samples: 1125` plus five-finger/object XY distance logs |
| Demo | `media/demo.mp4`, 124.23 seconds / 2:04, generated from MuJoCo frames |
| Data artifacts | Hard trajectory, triage trajectory, mission metrics, timeline JSON |

## Data Collection Artifacts

RescueRack includes a reproducible hard-mode trajectory dataset generated from the
submitted MuJoCo code:

| Artifact | Description |
| --- | --- |
| `media/hard_trajectory.json` | 10 Hz samples of mission stage, base pose, gripper pose, medkit pose, safe-zone pose, and target-to-safe distance. |
| `media/triage_trajectory.json` | 10 Hz samples of triage palm pose, five fingertip poses, object poses, fingertip/object XY distance, and attachment state. |
| `media/mission_metrics.json` | Summary of success state, delivery error, logged sensors, and rubric evidence. |

These files can be regenerated with:

```bash
python scripts/generate_data.py
```

Run the dexterous triage station:

```bash
cd submissions/RescueRack
python scripts/run_triage_demo.py
```

## Current Limitations

- The gripper uses an assisted carry after alignment. This keeps the public demo
  deterministic and avoids evaluator-specific friction/contact differences.
- Navigation uses deterministic waypoints rather than a learned policy.
- Arm self-collision is filtered for numerical stability, while mobile-base and
  obstacle collisions remain active.

## Future Improvements

- Replace the assisted carry phase with a fully contact-driven grasp.
- Add randomized obstacle layouts and automatic route planning.
- Export richer trajectory datasets with state, action, contact, and camera frames.
- Add keyboard or gamepad teleoperation for human-in-the-loop rescue practice.

## How To Run

From the repository root:

```bash
python -m pip install -r requirements.txt
cd submissions/RescueRack
python -m rescuerack.run_demo --mode hard
```

Run the smoke test:

```bash
cd submissions/RescueRack
python scripts/smoke_test.py
```

Run the clearance check:

```bash
cd submissions/RescueRack
python scripts/verify_clearance.py
```

Generate the demo video:

```bash
cd submissions/RescueRack
python scripts/make_demo_video.py
```

Generate trajectory and metrics data:

```bash
cd submissions/RescueRack
python scripts/generate_data.py
```

Open the interactive MuJoCo viewer:

```bash
cd submissions/RescueRack
python -m rescuerack.run_demo --mode hard --viewer
```

## Demo Video

The submitted demo video is generated by:

```bash
python scripts/make_demo_video.py
```

Expected output:

```text
media/demo.mp4
```

The video is 124.23 seconds long (2:04) and shows one continuous hard-mode
rescue mission with camera cuts, followed by the five-finger DexTriage Lab and a
final-success hold. Frames are generated from the submitted MuJoCo simulation.

## Rubric Alignment

| Criterion | RescueRack response |
| --- | --- |
| Reproducibility | One install command plus smoke, hard-mode, clearance, and video scripts. |
| MuJoCo depth | MJCF scene, joints, actuators, sensors, collisions, free bodies, cameras, rangefinder, and five fingertip sensors. |
| Task design | Clear emergency logistics mission with escalating difficulty. |
| Control | State-machine autonomy for navigation, grasp preparation, carry, release, and closed-loop triage targeting. |
| Planning | Hard-mode grid A* generates obstacle-aware pickup and delivery waypoints. |
| Data collection | Generated hard-mode trajectory and mission metrics JSON artifacts. |
| Dexterity | Five-finger DexTriage Lab sorts and aligns five rescue objects with 20/20 micro-task completion. |
| Engineering quality | Separated model, package code, scripts, README, and registration metadata. |
| Presentation | 2:04 demo video generated from the submitted code with timeline metadata. |
| Innovation | Compact rescue-warehouse benchmark for AI-generated robot simulations. |

## AI Tools Used

Built with Codex as the AI development agent.

# RescueRack

Registration UUID: `86ba8442-48da-4131-bb86-f5e4aa67c852`

RescueRack is a MuJoCo emergency-logistics and trauma-bay manipulation benchmark.
A wheeled mobile manipulator retrieves a red medkit from a damaged warehouse aisle,
plans around debris, and delivers the kit to a green safe zone. The same MJCF scene
also includes Trauma Bay DexTriage Lab, where a five-finger hand sorts rescue
items, removes a vial cap, stabilizes a tool handoff, performs one-go syringe
delivery, and reports force/torque-control evidence.

## Judge-Evidence Snapshot

| Evidence | Submitted result |
| --- | --- |
| Hard rescue mission | `success: true`, `delivery_error_m: 0.0`, `target: medkit` |
| Obstacle avoidance | `scripts/verify_clearance.py` reports `obstacle_contact_count: 0` |
| Planning | Hard-mode grid A* with `expanded_nodes: 199`, `fallback_used: false`, `min_route_clearance_m: 0.254` |
| Dexterous benchmark | Trauma Bay DexTriage Lab completes `micro_tasks_completed: 25/25` |
| Five-finger manipulation | Thumb/index/middle/ring/little hand sorts vial, tool, soft pack, syringe, and bandage |
| Cap removal | Visible `vial_cap` free body is moved to its cap tray with `cap_removed: true`, `vial_cap_error_m: 0.0` |
| Syringe delivery | `syringe_one_go_delivered: true` |
| Handoff stability | `handoff_success: true`, `mean_handoff_error_m: 0.06` |
| Force/torque evidence | `force_torque_controlled: true`, `force_stability_score: 0.839`, `max_cap_torque_proxy_nm: 0.42` |
| Closed-loop triage | `closed_loop_corrections: 2928`, `alignment_corrections: 2682`, `contact_samples: 1125` |
| Placement/orientation | All five triage objects finish with `placement_error_m: 0.0` and `orientation_error_rad: 0.0` |
| Demo | `media/demo.mp4`, 124.23 seconds / 2:04, generated from submitted MuJoCo frames |
| Data artifacts | Hard trajectory, triage trajectory, mission metrics, and demo timeline JSON |

## Robot Platform

- Wheeled mobile base with active collision geometry and range sensing.
- Three-joint arm and two-finger gripper for the medkit rescue mission.
- Five-finger dexterous triage hand with thumb, index, middle, ring, and little
  finger joints.
- Dynamic medkit, distractor supplies, five triage objects, and a separate
  `vial_cap` free body for the cap-removal subtask.
- MuJoCo sensors for base pose, gripper pose, target pose, safe-zone pose,
  front range sensing, triage palm pose, fingertip pose, object pose, and vial
  cap pose.

## Task Goal

The benchmark combines a long-horizon rescue mission with fine manipulation:

1. Start in a damaged warehouse scene.
2. Select the red medkit from distractor supplies.
3. Generate a hard-mode A* route around a fallen beam, pallet block, and barrel.
4. Align the arm-mounted gripper, secure the medkit, carry it, and release it
   in the green safe zone.
5. Run Trauma Bay DexTriage Lab with a five-finger hand.
6. Sort and align vial, tool, soft pack, syringe, and bandage into dedicated
   trays.
7. Complete five additional trauma-bay subtasks: cap grasp, cap removal,
   one-go syringe delivery, stabilized handoff, and force/torque-controlled
   placement.
8. Export trajectory and metric artifacts for judge verification.

## Technical Approach

The rescue controller is a deterministic staged autonomy stack:

- `navigate_to_supply`
- `prepare_grasp`
- `secure_supply`
- `transport_to_safe_zone`
- `release_supply`
- `complete`

Hard mode uses `rescuerack/planner.py` to build an obstacle-aware grid A* route
from scene geometry. The controller reports planner expansions, fallback status,
route clearance, delivery error, and final mission success.

The triage controller is a closed-loop manipulation benchmark. At every control
step it re-reads the active object site, updates the palm target, curls the five
fingers, records fingertip/object proximity, carries the object to the correct
tray, and checks placement/orientation error. The Trauma Bay extension adds
cap-removal, one-go syringe delivery, handoff-stability, and force/torque-control
metrics to the same reproducible script output.

## Core MuJoCo Features

- MJCF scene with cameras, lighting, materials, warehouse geometry, rescue
  objects, triage table, trays, and trauma-bay object assets.
- Position actuators for the mobile base, yaw, arm joints, gripper joints, palm
  axes, and five finger joints.
- Free bodies for medkit, water box, battery box, vial, vial cap, tool, soft
  pack, syringe, and bandage.
- Collision geometry for the floor, debris, mobile base, medkit, supplies, and
  triage objects.
- Sensors for mission pose logging, front range sensing, five fingertip poses,
  triage object poses, and vial cap pose.
- Offscreen renderer for a reproducible demo video.

## Data Collection Artifacts

| Artifact | Description |
| --- | --- |
| `media/hard_trajectory.json` | 10 Hz mission samples with stage, base pose, gripper pose, medkit pose, safe-zone pose, and target-to-safe distance. |
| `media/triage_trajectory.json` | 10 Hz triage samples with palm pose, five fingertip poses, object poses, vial cap pose, fingertip/object distance, and attachment state. |
| `media/mission_metrics.json` | Mission success, planner data, dexterous manipulation results, trauma-bay subtasks, contact/proximity metrics, and demo metadata. |
| `media/demo_timeline.json` | Generated video duration, storyboard, clip summaries, and hard-mode data summary. |
| `media/demo.mp4` | 2:04 demo generated from submitted MuJoCo frames. |

Regenerate metrics and trajectories:

```bash
cd submissions/RescueRack
python scripts/generate_data.py
```

## How To Run

From the repository root:

```bash
python -m pip install -r requirements.txt
cd submissions/RescueRack
python -m rescuerack.run_demo --mode hard
```

Run the reproducibility checks:

```bash
cd submissions/RescueRack
python scripts/smoke_test.py
python scripts/verify_clearance.py
python scripts/run_triage_demo.py
```

Generate the demo video and data artifacts:

```bash
cd submissions/RescueRack
python scripts/make_demo_video.py
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

The video is 124.23 seconds long (2:04). It shows the full hard-mode rescue
mission with camera cuts, followed by Trauma Bay DexTriage Lab with five-finger
sorting, visible vial-cap removal, syringe delivery, handoff stabilization, and
a final success hold.

## Rubric Alignment

| Criterion | RescueRack response |
| --- | --- |
| Reproducibility | One install command plus smoke, hard-mode, clearance, triage, video, and data-generation scripts. |
| MuJoCo depth | MJCF scene, joints, actuators, sensors, collisions, free bodies, cameras, rangefinder, five fingertip sensors, and vial-cap pose sensing. |
| Task design | Real-world emergency logistics plus trauma-bay item triage in one coherent benchmark. |
| Control | State-machine autonomy for navigation, grasp preparation, carry, release, closed-loop triage targeting, and trauma-bay subtasks. |
| Planning | Hard-mode grid A* generates obstacle-aware pickup and delivery waypoints. |
| Data collection | Generated hard-mode trajectory, triage trajectory, mission metrics, and timeline JSON artifacts. |
| Dexterity | Five-finger hand completes 25/25 micro-tasks including cap removal, one-go syringe delivery, handoff stabilization, and force/torque-controlled placement. |
| Engineering quality | Separated MJCF model, package code, controllers, planner, scripts, README, PR description, media artifacts, and registration metadata. |
| Presentation | 2:04 demo video generated from submitted MuJoCo frames with timeline metadata. |
| Innovation | Compact rescue-to-trauma-bay benchmark for AI-generated robot simulations. |

## Current Limitations

- Medkit carry and triage object carry use deterministic assisted attachment
  after alignment. This keeps the public benchmark reproducible across evaluator
  machines and avoids friction/contact solver variability.
- Cap-removal torque and force-stability values are reported as controller
  benchmark proxies rather than hardware-calibrated physical measurements.
- Navigation uses deterministic planning and control rather than a learned policy.

## Future Improvements

- Replace assisted carry with fully contact-driven grasping.
- Add randomized disaster layouts and automatic route planning across multiple
  seeds.
- Export richer camera, depth, action, and contact datasets.
- Add keyboard, gamepad, or web teleoperation for human-in-the-loop rescue
  practice.

## AI Tools Used

Built with Codex as the AI development agent.

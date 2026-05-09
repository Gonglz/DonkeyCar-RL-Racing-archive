# Camera Freespace + LiDAR Tokens (April 22, 2026)

## 1. Purpose

This note records a design adjustment for the current V17/LWM direction.

The adjustment is:

- do not treat global track geometry as the main source of local passability truth
- move `gap/passable` toward camera-driven local freespace
- keep LiDAR focused on obstacle/opponent perception and near-field safety enclosure

Current frozen sim LiDAR baseline for this note:

- pose: `offset_y = 0.40`, `offset_z = 0.50`, `rot_x = 0.0`
- max range: `20.0m`
- packet interpretation: full `360 deg`, ego-forward at `rx ~= 180 deg`
- packet distance scaling: `d / 8 -> telemetry meters`

This is not a claim that LiDAR is useless for free-space.
It means:

- global map / track geometry is optional
- local enclosure is still necessary
- camera is a better candidate for planar open-track passability than hand-coded track geometry labels

## 2. Problem With The Old Framing

The old framing mixed together three different things:

1. global track geometry
2. local free-space / enclosure
3. foreground obstacle/opponent tracking

Those are not the same problem.

For the current racing stack:

- global track geometry is not strictly required
- local enclosure is still required
- foreground obstacle tracking is also required

The main issue with a geometry-heavy formulation is that it can make `gap/passable`
look cleaner in sim than they will be in deployment.

## 3. New Task Split

### 3.1 Camera branch

Camera should become the primary source for:

- local drivable area / freespace
- coarse `left_gap / right_gap`
- `passable_left / passable_right`
- open planar region estimation in front of the ego car

Why:

- the track is open and mostly planar
- camera already carries road / line / local open-space cues
- this reduces dependence on hand-coded global geometry

### 3.2 LiDAR branch

LiDAR should remain responsible for:

- near-field obstacle presence
- opponent / obstacle tokenization
- `TTC`, `closing_rate`, relative pose
- near-field enclosure safety signals

Why:

- LiDAR gives reliable local distance structure
- it is better suited than monocular vision for close obstacle confirmation
- it can serve as a safety check on top of camera freespace estimates

Round1 implication:

- do not spend time on `wall vs car` classification
- do keep a single `primary target` token plus a conservative safety enclosure layer
- current remaining work is target selection stability, not recovering raw visibility

### 3.3 Safety enclosure

Even if camera owns `gap/passable`, a minimal LiDAR enclosure layer should remain.

This layer does not need global geometry.
It only needs to answer:

- is there close structure in front?
- is there close structure on the left?
- is there close structure on the right?
- is the local space collapsing?

This is the minimum near-field anti-collision layer.

## 4. What We No Longer Want As The Main Dependency

Not the new main dependency:

- track centerline
- track width lookup
- progress-based geometry labels
- map-conditioned passability labels

These may still exist for analysis or fallback, but they should not define the
main semantics of `gap/passable`.

## 5. Immediate Practical Adjustment

The first practical step is to stop defaulting `target_gap` and `target_passable`
to track-geometry-derived labels in dataset export.

Current implementation change:

- `scripts/export_world_model_dataset.py`
- new flag: `--gap-label-source {sensor,track}`
- default is now `sensor`

Meaning:

- exported `target_gap` now defaults to sensor-side local gap estimation
- exported `target_passable` now defaults to the same sensor-derived gap source
- track geometry is now opt-in for this label path, not the default

This does not yet implement camera freespace labels.
It removes geometry as the default label source and keeps the system closer to deployment semantics.

## 6. Planned Next Architecture

Recommended next architecture:

### Head A: Camera freespace / passability

Inputs:

- front semantic image / reduced camera representation

Outputs:

- `passable_left`
- `passable_right`
- `left_gap_coarse`
- `right_gap_coarse`
- optional front drivable confidence

### Head B: LiDAR obstacle tokens

Inputs:

- canonical LiDAR
- short temporal window

Outputs:

- `target_exist`
- `rel_long`
- `rel_lat`
- `rel_v_long`
- `rel_v_lat`
- `TTC`
- token confidence

### Head C: LiDAR safety enclosure

Inputs:

- canonical LiDAR

Outputs:

- `front_min`
- `left_min`
- `right_min`
- near-field enclosure flags

This head is intentionally simple and conservative.

## 7. Why This Split Is Better

It matches deployment better:

- camera handles planar passability
- LiDAR handles close obstacle certainty
- local safety no longer depends on global map geometry

It also reduces the pressure on `Phase F`:

- `Phase F` no longer needs LiDAR to carry the full burden of passability semantics
- LiDAR realism still matters for obstacle/safety use
- but the system is no longer pretending LiDAR alone must behave like a full local geometry engine

## 8. Constraints

This change does not mean:

- remove LiDAR
- ignore local enclosure
- trust monocular vision alone for the safety floor

The recommended interpretation is:

- no global geometry dependency
- yes local camera freespace
- yes LiDAR obstacle tokens
- yes LiDAR near-field safety

## 9. Current Recommended Direction

Recommended direction from this point:

1. Keep `Phase F` improvements on the packet-path LiDAR chain.
2. Treat LiDAR as `obstacle/opponent + safety enclosure`, not as the only source of passability truth.
3. Move future `gap/passable` semantics toward a camera freespace branch.
4. Keep geometry-derived labels as optional analysis tools, not default deployment semantics.

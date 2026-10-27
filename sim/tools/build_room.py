#!/usr/bin/env python3
"""Write worlds/pallet_room.sdf: a closed lab room with distractors (D81).

    python3 sim/tools/build_room.py

The robot spawns at the origin looking along +x and the pallet's face is placed
by launch arguments, ~2.5 m ahead (D75). Everything here is static and built
from boxes and cylinders on purpose -- what perception sees is depth, so a
distractor only has to have the right SIZE and PLACE, and simple shapes keep
the build offline (no Gazebo Fuel downloads).

Each distractor is there to exercise one of perception's gates, inside its
search box (x 0.5-4.0 m, y +-1.5 m, z up to 0.8 m) and clear of the approach
corridor (|y| < 0.75 m up to the pallet):

    partition    edge-on, 1 m to the right: the panel that beat the pallet on
                 the robot (2026-10-23, D76)
    low_crate    0.60 x 0.40 x 0.15 m: pallet HEIGHT and a pallet-ish width, so
                 it passes the height and width gates; it is further than the
                 pallet, so "nearest wins" must pick the pallet
    chair, bin   ordinary clutter at the edges of the search box
    cabinet      behind the pallet, inside the LiDAR cross-check's look-behind
    table, shelf, whiteboard   further back, for the LiDAR and the dashboard
"""

from __future__ import annotations

import pathlib

OUT = (pathlib.Path(__file__).resolve().parents[1]
       / 'pallet_approach_sim' / 'worlds' / 'pallet_room.sdf')

ROOM_X = (-3.0, 7.5)
ROOM_Y = (-4.0, 4.0)
WALL_H, WALL_T = 2.5, 0.10
DOOR = (-2.0, -1.0)          # gap in the +y wall, along x

GREY = '0.75 0.75 0.73 1'
WHITE = '0.92 0.92 0.90 1'
DARK = '0.20 0.20 0.22 1'
WOOD = '0.55 0.40 0.25 1'
BLUE = '0.25 0.35 0.55 1'


def material(rgba: str) -> str:
    return (f'<material><ambient>{rgba}</ambient><diffuse>{rgba}</diffuse>'
            f'<specular>0.1 0.1 0.1 1</specular></material>')


def box(name: str, size, xyz, rgba=GREY, yaw=0.0) -> str:
    sx, sy, sz = size
    geo = f'<geometry><box><size>{sx} {sy} {sz}</size></box></geometry>'
    return (f'<link name="{name}"><pose>{xyz[0]} {xyz[1]} {xyz[2]} 0 0 {yaw}</pose>'
            f'<collision name="c">{geo}</collision>'
            f'<visual name="v">{geo}{material(rgba)}</visual></link>')


def cylinder(name: str, radius: float, length: float, xyz, rgba=GREY) -> str:
    geo = f'<geometry><cylinder><radius>{radius}</radius><length>{length}</length></cylinder></geometry>'
    return (f'<link name="{name}"><pose>{xyz[0]} {xyz[1]} {xyz[2]} 0 0 0</pose>'
            f'<collision name="c">{geo}</collision>'
            f'<visual name="v">{geo}{material(rgba)}</visual></link>')


def model(name: str, links: list[str]) -> str:
    return f'<model name="{name}"><static>true</static>{"".join(links)}</model>'


def walls() -> str:
    (x0, x1), (y0, y1) = ROOM_X, ROOM_Y
    h = WALL_H / 2
    links = [
        box('back', (WALL_T, y1 - y0, WALL_H), (x1, 0, h), WHITE),
        box('front', (WALL_T, y1 - y0, WALL_H), (x0, 0, h), WHITE),
        box('right', (x1 - x0, WALL_T, WALL_H), ((x0 + x1) / 2, y0, h), WHITE),
        # left wall with a door gap
        box('left_a', (DOOR[0] - x0, WALL_T, WALL_H), ((x0 + DOOR[0]) / 2, y1, h), WHITE),
        box('left_b', (x1 - DOOR[1], WALL_T, WALL_H), ((DOOR[1] + x1) / 2, y1, h), WHITE),
    ]
    return model('walls', links)


def chair(name: str, x: float, y: float) -> str:
    legs = [box(f'leg{i}', (0.03, 0.03, 0.45), (x + dx, y + dy, 0.225), DARK)
            for i, (dx, dy) in enumerate(((0.2, 0.2), (0.2, -0.2), (-0.2, 0.2), (-0.2, -0.2)))]
    return model(name, legs + [box('seat', (0.46, 0.46, 0.05), (x, y, 0.47), BLUE),
                               box('back', (0.05, 0.46, 0.45), (x + 0.21, y, 0.72), BLUE)])


def table(name: str, x: float, y: float) -> str:
    legs = [box(f'leg{i}', (0.04, 0.04, 0.72), (x + dx, y + dy, 0.36), DARK)
            for i, (dx, dy) in enumerate(((0.55, 0.3), (0.55, -0.3), (-0.55, 0.3), (-0.55, -0.3)))]
    return model(name, legs + [box('top', (1.2, 0.7, 0.04), (x, y, 0.74), WOOD)])


def shelf(name: str, x: float, y: float) -> str:
    links = [box(f'upright{i}', (0.4, 0.03, 1.8), (x, y + dy, 0.9), DARK)
             for i, dy in enumerate((-0.9, 0.9))]
    links += [box(f'board{i}', (0.4, 1.8, 0.02), (x, y, z), WOOD)
              for i, z in enumerate((0.10, 0.55, 1.0, 1.45))]
    links += [box('crate', (0.35, 0.5, 0.3), (x, y - 0.4, 0.26), '0.7 0.6 0.4 1')]
    return model(name, links)


def floor() -> str:
    # Polished concrete: a strong, tight specular highlight. Gazebo's depth
    # camera ignores materials, so this changes only the colour image and the
    # dashboard; the ZED's trouble with shiny floors is in zed_depth_model.py.
    geo = '<geometry><plane><normal>0 0 1</normal><size>60 60</size></plane></geometry>'
    return (
        '<model name="floor"><static>true</static><link name="link">'
        f'<collision name="collision">{geo}'
        '<surface><friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction></surface>'
        '</collision>'
        f'<visual name="visual">{geo}<material>'
        '<ambient>0.42 0.42 0.44 1</ambient><diffuse>0.42 0.42 0.44 1</diffuse>'
        '<specular>0.8 0.8 0.8 1</specular>'
        '<pbr><metal><metalness>0.0</metalness><roughness>0.15</roughness></metal></pbr>'
        '</material></visual></link></model>')


def world() -> str:
    distractors = [
        model('partition', [box('panel', (0.80, 0.04, 1.60), (2.9, -1.0, 0.80), WHITE)]),
        model('low_crate', [box('crate', (0.60, 0.40, 0.15), (3.6, 1.15, 0.075), WOOD)]),
        chair('chair', 2.0, 1.4),
        model('bin', [cylinder('bin', 0.18, 0.45, (1.6, -1.3, 0.225), DARK)]),
        # Behind the pallet for any start up to face_x 3.5 m (its back is then
        # at 4.70 m). At x 4.7 it overlapped a 3.5 m pallet by 25 cm, and the
        # merged cluster was rejected as too tall (2026-10-26).
        model('cabinet', [box('cabinet', (0.5, 1.0, 0.9), (5.3, 0.3, 0.45), GREY)]),
        table('table', 5.0, -2.0),
        shelf('shelf', 7.1, 0.5),
        model('whiteboard', [box('board', (0.05, 1.5, 1.0), (6.0, 2.4, 1.3), WHITE),
                             box('stand', (0.4, 0.05, 0.8), (6.0, 2.4, 0.4), DARK)]),
    ]
    return f'''<?xml version="1.0"?>
<!--
GENERATED by sim/tools/build_room.py; edit that, not this.
A closed lab room with distractors (D81). The world keeps the name
"pallet_lab" so the robot and pallet spawn into it exactly as into the empty
world (launch argument world:=empty).
-->
<sdf version="1.8">
  <world name="pallet_lab">
    <!-- 4 ms steps: see pallet_lab.sdf for why. -->
    <physics name="4ms" type="ignored">
      <max_step_size>0.004</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="ignition-gazebo-physics-system" name="ignition::gazebo::systems::Physics"/>
    <plugin filename="ignition-gazebo-user-commands-system" name="ignition::gazebo::systems::UserCommands"/>
    <plugin filename="ignition-gazebo-scene-broadcaster-system" name="ignition::gazebo::systems::SceneBroadcaster"/>
    <plugin filename="ignition-gazebo-sensors-system" name="ignition::gazebo::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <scene>
      <ambient>0.5 0.5 0.5 1</ambient>
      <background>0.75 0.8 0.85 1</background>
      <grid>false</grid>
    </scene>
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.7 0.7 0.7 1</diffuse>
      <specular>0.4 0.4 0.4 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>
    <light type="point" name="ceiling">
      <pose>2.5 0 2.4 0 0 0</pose>
      <diffuse>0.6 0.6 0.6 1</diffuse>
      <specular>0.6 0.6 0.6 1</specular>
      <attenuation><range>12</range><constant>0.5</constant><linear>0.05</linear><quadratic>0.01</quadratic></attenuation>
    </light>
    {floor()}
    {walls()}
    {"".join(distractors)}
  </world>
</sdf>
'''


if __name__ == '__main__':
    OUT.write_text(world())
    print(f'wrote {OUT}')

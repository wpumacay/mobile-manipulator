from __future__ import annotations

import time
from pathlib import Path
from typing import List

import numpy as np

import mujoco as mj
import mujoco.viewer

RED_COLOR = np.array([1.0, 0.0, 0.0, 1.0])
GREEN_COLOR = np.array([0.0, 1.0, 0.0, 1.0])
BLUE_COLOR = np.array([0.0, 0.0, 1.0, 1.0])

SCENE_XML = """
<mujoco model="robo-scene-ranger-mini-v2">
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="1 -1 1.5" dir="-1 1 -1" diffuse="0.5 0.5 0.5" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
  </worldbody>
</mujoco>
"""


class Primitive:

    def __init__(self, name: str):
        self.name: str = name
        self.type: mj.mjtGeom = mj.mjtGeom.mjGEOM_SPHERE
        self.size: np.ndarray = np.array([0.1, 0.1, 0.1])
        self.position: np.ndarray = np.array([0.0, 0.0, 0.0])
        self.orientation: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0])
        self.color: np.ndarray = GREEN_COLOR
        self.mjc_id = -1

class Runtime:

    def __init__(self):
        self.primitives: List[Primitive] = []
        #### self.spec = mj.MjSpec.from_string(SCENE_XML)
        model_path = Path(__file__).parent.parent / "resources" / "ranger-mini-v2" / "scene.xml"
        self.spec = mj.MjSpec.from_file(str(model_path.resolve()))

    def add_primitive(self, primitive: Primitive) -> None:
        self.primitives.append(primitive)

    def build(self) -> mj.MjModel:
        for prim in self.primitives:
            body = self.spec.worldbody.add_body(name=prim.name, pos=prim.position, quat=prim.orientation)
            body.add_freejoint(name=f"{prim.name}_jnt")
            body.add_geom(name=f"{prim.name}_geom", type=prim.type, size=prim.size, rgba=prim.color)

        model = self.spec.compile()
        for prim in self.primitives:
            prim.mjc_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY.value, prim.name)

        return model


def main() -> int:
    box = Primitive("box")
    box.type = mj.mjtGeom.mjGEOM_BOX
    box.size = np.array([0.5, 0.5, 0.5])
    box.position = np.array([1.75, 0.0, 3.0])
    box.color = np.array([66 / 255, 133 / 255, 244 / 255, 1.0])

    sphere = Primitive("sphere")
    sphere.type = mj.mjtGeom.mjGEOM_SPHERE
    sphere.size = np.array([0.5, 0.0, 0.0])
    sphere.position = np.array([0.75, 0.0, 3.0])
    sphere.color = np.array([234 / 255, 67 / 255, 53 / 255, 1.0])

    ellipsoid = Primitive("ellipsoid")
    ellipsoid.type = mj.mjtGeom.mjGEOM_ELLIPSOID
    ellipsoid.size = np.array([0.5, 0.25, 0.75])
    ellipsoid.position = np.array([-0.75, 0.0, 3.0])
    ellipsoid.color = np.array([251 / 255, 188 / 255, 5 / 255, 1.0])

    cylinder = Primitive("cylinder")
    cylinder.type = mj.mjtGeom.mjGEOM_CYLINDER
    cylinder.size = np.array([0.25, 0.5, 0.25])
    cylinder.position = np.array([-1.75, 0.0, 3.0])
    cylinder.color = np.array([52 / 255, 168 / 255, 83 / 255, 1.0])

    capsule = Primitive("capsule")
    capsule.type = mj.mjtGeom.mjGEOM_CAPSULE
    capsule.size = np.array([0.25, 0.5, 0.25])
    capsule.position = np.array([-2.75, 0.0, 3.0])
    capsule.color = np.array([120 / 255, 168 / 255, 20 / 255, 1.0])

    runtime = Runtime()
    runtime.add_primitive(box)
    runtime.add_primitive(sphere)
    runtime.add_primitive(ellipsoid)
    runtime.add_primitive(cylinder)
    runtime.add_primitive(capsule)

    model = runtime.build()
    data = mj.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            mujoco.mj_step(model, data)

            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())

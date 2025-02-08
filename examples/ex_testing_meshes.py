
import time
from pathlib import Path
from typing import Tuple

import numpy as np

import mujoco as mj
import mujoco.viewer

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

def create_wall(
    p_start: np.ndarray,
    p_end: np.ndarray,
    height: float = 2.0,
    thickness: float = 0.25
) -> Tuple[np.ndarray, np.ndarray]:
    assert len(p_start) == 2 and len(p_end) == 2
    u = (p_start - p_end)
    u = u / np.linalg.norm(u)
    v = np.array([-u[1], u[0]])

    p_a = p_start + 0.5 * thickness * v
    p_b = p_start - 0.5 * thickness * v
    p_c = p_end - 0.5 * thickness * v
    p_d = p_end + 0.5 * thickness * v

    p_0 = np.array([p_a[0], p_a[1], 0.0])
    p_1 = np.array([p_b[0], p_b[1], 0.0])
    p_2 = np.array([p_c[0], p_c[1], 0.0])
    p_3 = np.array([p_d[0], p_d[1], 0.0])

    p_4 = np.array([p_a[0], p_a[1], height])
    p_5 = np.array([p_b[0], p_b[1], height])
    p_6 = np.array([p_c[0], p_c[1], height])
    p_7 = np.array([p_d[0], p_d[1], height])

    vertices = np.stack([p_0, p_1, p_2, p_3, p_4, p_5, p_6, p_7])
    faces = np.array([[0, 3, 2], [0, 2, 1],
                      [0, 1, 5], [0, 5, 4],
                      [1, 2, 6], [1, 6, 5],
                      [2, 3, 7], [2, 7, 6],
                      [0, 4, 7], [0, 7, 3],
                      [4, 5, 6], [4, 6, 7]])

    return vertices, faces




def main() -> int:
    #spec = mj.MjSpec.from_string(SCENE_XML)
    model_path = Path(__file__).parent.parent / "resources" / "ranger-mini-v2" / "scene.xml"
    spec = mj.MjSpec.from_file(str(model_path.resolve()))

    vertices = np.array([[0.0, 0.0, 0.0],
                         [1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 1.0]])
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    _ = spec.add_mesh(name="custom_mesh", uservert=vertices.ravel(), userface=faces.ravel())
    vertices, faces = create_wall(np.array([0.0, -2.0]), np.array([-2.0, 0.0]))
    _ = spec.add_mesh(name="custom_wall1", uservert=vertices.ravel(), userface=faces.ravel())
    vertices, faces = create_wall(np.array([-2.0, 0.0]), np.array([0.0, 2.0]))
    _ = spec.add_mesh(name="custom_wall2", uservert=vertices.ravel(), userface=faces.ravel())
    vertices, faces = create_wall(np.array([0.0, 2.0]), np.array([2.0, 0.0]))
    _ = spec.add_mesh(name="custom_wall3", uservert=vertices.ravel(), userface=faces.ravel())

    # mesh1 = spec.worldbody.add_body(name="my_mesh", pos=[0, 0, 2])
    # mesh1.add_freejoint()
    # mesh1.add_geom(type=mj.mjtGeom.mjGEOM_MESH, meshname="custom_mesh")

    mesh_1 = spec.worldbody.add_geom(type=mj.mjtGeom.mjGEOM_MESH, meshname="custom_wall1")
    mesh_2 = spec.worldbody.add_geom(type=mj.mjtGeom.mjGEOM_MESH, meshname="custom_wall2")
    mesh_3 = spec.worldbody.add_geom(type=mj.mjtGeom.mjGEOM_MESH, meshname="custom_wall3")

    model = spec.compile()
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




    return 0




if __name__ == "__main__":
    raise SystemExit(main())

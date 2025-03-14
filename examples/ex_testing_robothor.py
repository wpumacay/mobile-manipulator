
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

YCB_DATASET_PATH = Path(__file__).parent.parent / "resources" / "ycb"
YCB_XML_TEMPLATE = """
<mujoco model="ycb-{name}">
    <compiler angle="radian" meshdir="{meshdir}" texturedir="{texturedir}" />

    <asset>
        <texture type="2d" name="ycb-object-texture" file="texture_map.png"/>
        <mesh name="ycb-object-meshfile" file="textured.obj"/>
        <material name="ycb-object-material" specular="0.2" shininess="0.2" texture="ycb-object-texture"/>
    </asset>

    <worldbody>
        <body name="ycb-object-body">
            <freejoint/>
            <geom type="mesh" mesh="ycb-object-meshfile" material="ycb-object-material"/>
        </body>
    </worldbody>
</mujoco>
"""

ROBOTHOR_ASSETS_BASEPATH = Path(__file__).parent.parent / "resources" / "thor-assets" / "robothor-objects"


def create_mesh() -> Tuple[np.ndarray, np.ndarray]:
    p_0 = np.array([ 1.0, -1.0, -1.0])
    p_1 = np.array([ 1.0,  1.0, -1.0])
    p_2 = np.array([-1.0,  1.0, -1.0])
    p_3 = np.array([-1.0, -1.0, -1.0])

    p_4 = np.array([ 1.0, -1.0, 1.0])
    p_5 = np.array([ 1.0,  1.0, 1.0])
    p_6 = np.array([-1.0,  1.0, 1.0])
    p_7 = np.array([-1.0, -1.0, 1.0])

    vertices = 0.125 * np.stack([p_0, p_1, p_2, p_3, p_4, p_5, p_6, p_7])
    faces = np.array([[0, 3, 2], [0, 2, 1],
                      [0, 1, 5], [0, 5, 4],
                      [1, 2, 6], [1, 6, 5],
                      [2, 3, 7], [2, 7, 6],
                      [0, 4, 7], [0, 7, 3],
                      [4, 5, 6], [4, 6, 7]])
    return vertices, faces


def create_wall(
    p_start: np.ndarray,
    p_end: np.ndarray,
    height: float = 1.0,
    thickness: float = 0.125
) -> Tuple[np.ndarray, np.ndarray]:
    assert len(p_start) == 2 and len(p_end) == 2
    u = (p_end - p_start)
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

def create_ycb_object(root_spec: mj.MjSpec, model_name: str, position: np.ndarray) -> None:
    assets_dir = YCB_DATASET_PATH / model_name / "google_16k"
    xml_model_str = YCB_XML_TEMPLATE.format(
        name=model_name,
        meshdir=str(assets_dir.resolve()),
        texturedir=str(assets_dir.resolve()),
    )
    spec_ycb_model = mj.MjSpec.from_string(xml_model_str)
    model_body = spec_ycb_model.find_body("ycb-object-body")
    model_frame = root_spec.worldbody.add_frame(pos=position.tolist())
    model_frame.attach_body(model_body, f"{model_name}-", "")

def create_robothor_object(root_spec: mj.MjSpec, model_path: Path, position: np.ndarray, rotation: np.ndarray) -> None:
    spec_robothor_object = mj.MjSpec.from_file(str(model_path.resolve()))
    model_body = spec_robothor_object.find_body(model_path.stem)
    model_body.add_freejoint()
    model_frame = root_spec.worldbody.add_frame(pos=position.tolist(), euler=rotation.tolist())
    model_frame.attach_body(model_body, f"{model_path.stem}-", "")


def main() -> int:
    SCALE = 0.5

    model_path = Path(__file__).parent.parent / "resources" / "ranger-mini-v2" / "scene.xml"
    spec = mj.MjSpec.from_file(str(model_path.resolve()))
    robot_base_body = spec.find_body("base_link")
    robot_base_body.pos = np.array([11.0 * SCALE, 6.0 * SCALE, 0.75])

    vertices, faces = create_mesh()
    _ = spec.add_mesh(name="custom_mesh", uservert=vertices.ravel(), userface=faces.ravel())

    mesh0 = spec.worldbody.add_body(name="my_mesh", pos=[0, 0, 2])
    mesh0.add_freejoint()
    mesh0.add_geom(type=mj.mjtGeom.mjGEOM_MESH, meshname="custom_mesh")

    create_ycb_object(spec, "002_master_chef_can",  np.array([12.5 * SCALE - 0.125, 3.5 * SCALE - 0.125, 0.625]))
    create_ycb_object(spec, "003_cracker_box",      np.array([12.5 * SCALE - 0.125, 3.5 * SCALE + 0.125, 0.625]))
    create_ycb_object(spec, "004_sugar_box",        np.array([12.5 * SCALE + 0.125, 3.5 * SCALE + 0.125, 0.625]))
    create_ycb_object(spec, "006_mustard_bottle",   np.array([12.5 * SCALE + 0.125, 3.5 * SCALE - 0.125, 0.625]))

    create_robothor_object(
        spec,
        ROBOTHOR_ASSETS_BASEPATH / "Sofa" / "Prefabs" / "RoboTHOR_sofa_alrid" / "RoboTHOR_sofa_alrid.xml",
        SCALE * np.array([2.0, 4.5, 1.0]),
        np.array([np.pi / 2, 0.0, 0.0]),
    )

    create_robothor_object(
        spec,
        ROBOTHOR_ASSETS_BASEPATH / "Bed" / "Prefabs" / "RoboTHOR_bed_grankulla" / "RoboTHOR_bed_grankulla.xml",
        SCALE * np.array([7.0, 4.0, 1.0]),
        np.array([np.pi / 2, 0.0, 0.0]),
    )

    create_robothor_object(
        spec,
        ROBOTHOR_ASSETS_BASEPATH / "SideTable" / "Prefabs" / "RoboTHOR_side_table_fornbro" / "RoboTHOR_side_table_fornbro.xml",
        SCALE * np.array([12.5, 3.5, 0.6]),
        np.array([np.pi / 2, 0.0, 0.0]),
    )

    create_robothor_object(
        spec,
        ROBOTHOR_ASSETS_BASEPATH / "Chair" / "Prefabs" / "RoboTHOR_chair_antnas" / "RoboTHOR_chair_antnas.xml",
        SCALE * np.array([14.0, 5.5, 1.0]),
        np.array([np.pi / 2, 0.0, 0.0]),
    )

    create_robothor_object(
        spec,
        ROBOTHOR_ASSETS_BASEPATH / "Desk" / "Prefabs" / "RoboTHOR_desk_alve" / "RoboTHOR_desk_alve.xml",
        SCALE * np.array([12.5, 6.5, 1.0]),
        np.array([np.pi / 2, 0.0, 0.0]),
    )

    create_robothor_object(
        spec,
        ROBOTHOR_ASSETS_BASEPATH / "FloorLamp" / "Prefabs" / "RoboTHOR_floor_lamp_holmo_v" / "RoboTHOR_floor_lamp_holmo_v.xml",
        SCALE * np.array([2.0, 2.5, 1.0]),
        np.array([np.pi / 2, 0.0, 0.0]),
    )

    walls = [
        ((0.0, 0.0), (0.0, 8.0)),
        ((0.0, 8.0), (15.0, 8.0)),
        ((15.0, 8.0), (15.0, 0.0)),
        ((15.0, 0.0), (0.0, 0.0)),
        ((0.0, 6.0), (5.5, 6.0)),
        ((5.5, 6.0), (5.5, 1.25)),
        ((5.5, 1.25), (9.5, 1.25)),
        ((9.5, 1.25), (9.5, 6.0)),
    ]
    for i, (p_start, p_end) in enumerate(walls):
        vertices, faces = create_wall(SCALE * np.array(p_start), SCALE * np.array(p_end))
        _ = spec.add_mesh(name=f"wall_{i}", uservert=vertices.ravel(), userface=faces.ravel())
        spec.worldbody.add_geom(type=mj.mjtGeom.mjGEOM_MESH, meshname=f"wall_{i}")

    model = spec.compile()
    with open("mj_model.xml", "w") as fhandle:
        fhandle.write(spec.to_xml())
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

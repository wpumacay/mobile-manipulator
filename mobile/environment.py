

import mujoco
import mujoco.viewer

from mobile import RESOURCES_DIR


class Environment:

    def __init__(self):
        scene_xml = RESOURCES_DIR / "ranger-mini-v2" / "scene.xml"
        self._model = mujoco.MjModel.from_xml_path(str(scene_xml))
        self._data = mujoco.MjData(self._model)

        self._viewer = mujoco.viewer.launch_passive(self._model, self._data)

        print(f"num-actuators: {self._model.nu}")

    def reset(self):
        mujoco.mj_resetData(self._model, self._data)

    def step(self, action):

        mujoco.mj_step(self._model, self._data)

        self._viewer.sync()

    def close(self):
        self._viewer.close()




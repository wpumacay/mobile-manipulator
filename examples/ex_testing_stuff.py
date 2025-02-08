import time
from pathlib import Path

import numpy as np

import mujoco
import mujoco.viewer


def main() -> int:
    model_path = Path(__file__).parent.parent / "resources" / "ranger-mini-v2" / "scene.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    print(f"num-actuators: {model.nu}")

    def key_callback(keycode) -> None:
        print(f"keycode: {keycode}")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # data.ctrl = np.random.randn(8)

            mujoco.mj_step(model, data)

            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())




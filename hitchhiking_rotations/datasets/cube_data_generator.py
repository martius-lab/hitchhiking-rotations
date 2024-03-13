#
# Copyright (c) 2024, MPI-IS, Jonas Frey, Rene Geist, Mikel Zhobro.
# All rights reserved. Licensed under the MIT license.
# See LICENSE file in the project root for details.
#
import mujoco
import numpy as np
from PIL import Image


class CubeDataGenerator:
    def __init__(self, height: int, width: int):
        xml = """
        <mujoco>
            <worldbody>
                <light name="top" pos="0 0 0"/>
                <body name="cube" euler="0 0 0" pos="0 0 0">
                <joint type="ball" stiffness="0" damping="0" frictionloss="0" armature="0"/>
                <geom type="box" size="0.1 0.1 0.1" pos="0     0      0" rgba="0.5 0.5 0.5 1"/>
                <geom type="box" size="1 1 0.01"    pos="0     0      0.9" rgba="1 0 0 1"/>
                <geom type="box" size="1 1 0.01"    pos="0     0      -0.99" rgba="0 0 1 1"/>
                <geom type="box" size="0.01 1 1"    pos="0.99  0      0" rgba="0 1 0 1"/>
                <geom type="box" size="0.01 1 1"    pos="-0.99 0      0" rgba="0 0.6 0.6 1"/>
                <geom type="box" size="1 0.01 1"    pos="0     0.99   0" rgba="0.6 0.6 0 1"/>
                <geom type="box" size="1 0.01 1"    pos="0     -0.99  0" rgba="0.6 0 0.6 1"/>
                </body>
            </worldbody>  
        </mujoco>
        """
        # Make model, data, and renderer
        self.mj_model = mujoco.MjModel.from_xml_string(xml)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.renderer = mujoco.Renderer(self.mj_model, height=height, width=width)

        # enable joint visualization option:
        self.scene_option = mujoco.MjvOption()
        self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = False

    def render_img(self, quat: np.array) -> np.array:
        """
        Returns image for the body with the specified rotation.

        Args:
            quat (np.array, shape:=(4) ): scipy format  x,y,z,w
        """
        mujoco.mj_resetData(self.mj_model, self.mj_data)

        # mj_data.qpos = np.random.rand(4)
        self.mj_data.qpos = quat

        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.renderer.update_scene(self.mj_data, scene_option=self.scene_option)
        img = self.renderer.render()

        return img

    def __del__(self):
        self.renderer.close()


if __name__ == "__main__":
    dg = CubeDataGenerator(64, 64)
    img = dg.render_img(np.array([0, 0, 0, 1]))

    i1 = Image.fromarray(img)
    i1.show()

    img = dg.render_img(np.array([0, 1, 0, 1]))

    i1 = Image.fromarray(img)
    i1.show()

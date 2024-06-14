"""Example of using the MuJoCo Python viewer with for offscreen cameras without the MuJoCo's Python Render wrapper"""

import mujoco
import mujoco.viewer as viewer
from mujoco.renderer import Renderer
import numpy as np
import cv2

def fixation_control(m, d, renderer):
  try:
    renderer.update_scene(d, camera=0)
    image = renderer.render()

    # Show the simulated camera image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    lower_red = np.array([0, 0, 140])
    upper_red = np.array([50, 50, 255])
    mask = cv2.inRange(image, lower_red, upper_red)
    thresholded_image = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow('threshold', thresholded_image)
    cv2.waitKey(1)
  except Exception as e:
    print(e)
    raise e

def load_callback(m=None, d=None):
  # Clear the control callback before loading a new model
  # or a Python exception is raised
  mujoco.set_mjcb_control(None)

  m = mujoco.MjModel.from_xml_path('/home/rkommuru/mujoco-3.1.6/model/frog.xml')
  d = mujoco.MjData(m)

  if m is not None:
    renderer = Renderer(m, width=600, height=600)

    # Set the callback and capture all variables needed for rendering
    mujoco.set_mjcb_control(
      lambda m, d: fixation_control(
        m, d, renderer))

  return m , d

if __name__ == '__main__':
  viewer.launch(loader=load_callback)
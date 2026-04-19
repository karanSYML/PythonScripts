import yourdfpy, numpy as np

robot = yourdfpy.URDF.load("urdf_output/thruster_arm.urdf")
robot.update_cfg({
    "J1_shoulder_yaw": np.radians(45),
    "J2_elbow_pitch":  np.radians(90),
    "J3_wrist_pitch":  np.radians(-0),
})
robot.show()
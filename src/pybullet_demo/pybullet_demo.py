import time
import math

import pybullet as p
import pybullet_data


# ============================================================
# Configuration
# ============================================================

SIMULATION_FPS = 60
MOTION_SPEED = 0.35

GUI = p.GUI

# FIX: half-extent shrunk from 0.7 to 0.35 and moved further out (x=0.6).
# The old table spanned x=[-0.2, 1.2], which put its near edge *behind*
# the robot base (x=0) — meaning the arm had to swing up through solid
# table geometry just to leave its resting pose. Now the table's near
# edge is at x = 0.6 - 0.35 = 0.25, clear of the robot base column.
TABLE_HALF_EXTENTS = [0.35, 0.5, 0.1]
TABLE_POSITION = [0.6, 0.0, 0.2]
TABLE_TOP_Z = TABLE_POSITION[2] + TABLE_HALF_EXTENTS[2]  # = 0.3

OBJECT_HALF_EXTENTS = [0.025, 0.025, 0.025]
# Full cube width = 5 cm. Panda fingers open to targetPosition=0.04 each,
# giving a max fingertip gap of ~8 cm (minus finger pad thickness), so a
# 16 cm cube (the old half-extent of 0.08) could never fit between them
# no matter how the grasp pose was tuned. 5 cm leaves clearance to close
# around it.
# FIX: object now rests exactly on the table surface instead of
# floating 0.27 m above it (which caused a violent free-fall/impact
# at the start of the sim).
OBJECT_POSITION = [0.55, 0.0, TABLE_TOP_Z + OBJECT_HALF_EXTENTS[2]]

ARM_BASE_POSITION = [0.0, 0.0, 0.0]

# End-effector link index for the Panda URDF (the link IK solves for).
# This is the "hand" link, just before the two finger joints.
END_EFFECTOR_LINK = 11


# Panda joint indices
# ------------------------------------------------------------

ARM_JOINTS = [0, 1, 2, 3, 4, 5, 6]

LEFT_FINGER = 9
RIGHT_FINGER = 10


# ============================================================
# Utility functions
# ============================================================

def move_arm(robot, targets, duration=1.5):

    current = [
        p.getJointState(robot, j)[0]
        for j in ARM_JOINTS
    ]

    steps = max(1, int(duration * SIMULATION_FPS))

    for i in range(steps):

        t = (i + 1) / steps

        # Smooth start and stop
        smooth_t = t * t * (3.0 - 2.0 * t)

        for joint, start, target in zip(
            ARM_JOINTS,
            current,
            targets,
        ):

            position = (
                start
                + (target - start) * smooth_t
            )

            p.setJointMotorControl2(
                bodyUniqueId=robot,
                jointIndex=joint,
                controlMode=p.POSITION_CONTROL,
                targetPosition=position,
                force=100,
            )

        p.stepSimulation()

        time.sleep(
            (1 / SIMULATION_FPS) / MOTION_SPEED
        )


def open_gripper(robot, duration=1.0):

    steps = int(duration * SIMULATION_FPS)

    for _ in range(steps):

        p.setJointMotorControl2(
            robot,
            LEFT_FINGER,
            p.POSITION_CONTROL,
            targetPosition=0.04,
            force=30,
        )

        p.setJointMotorControl2(
            robot,
            RIGHT_FINGER,
            p.POSITION_CONTROL,
            targetPosition=0.04,
            force=30,
        )

        p.stepSimulation()

        time.sleep(
            (1 / SIMULATION_FPS) / MOTION_SPEED
        )


def close_gripper(robot, duration=1.0):

    steps = int(duration * SIMULATION_FPS)

    for _ in range(steps):

        p.setJointMotorControl2(
            robot,
            LEFT_FINGER,
            p.POSITION_CONTROL,
            targetPosition=0.0,
            force=30,
        )

        p.setJointMotorControl2(
            robot,
            RIGHT_FINGER,
            p.POSITION_CONTROL,
            targetPosition=0.0,
            force=30,
        )

        p.stepSimulation()

        time.sleep(
            (1 / SIMULATION_FPS) / MOTION_SPEED
        )


def compute_grasp_joint_pose(robot, target_pos, target_orn):
    """
    FIX: instead of hand-guessed joint angles (which may not actually
    place the gripper anywhere near the object), use inverse kinematics
    to compute joint angles that put the end effector at target_pos
    with target_orn. This is what prevents the gripper from closing on
    empty air or deeply overlapping the box.
    """
    joint_poses = p.calculateInverseKinematics(
        robot,
        END_EFFECTOR_LINK,
        target_pos,
        target_orn,
        maxNumIterations=200,
        residualThreshold=1e-5,
    )

    return list(joint_poses[:7])


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Start PyBullet
    # --------------------------------------------------------

    p.connect(GUI)

    p.setAdditionalSearchPath(
        pybullet_data.getDataPath()
    )

    p.setGravity(0, 0, -9.81)

    p.setTimeStep(1 / 240)

    # FIX: more solver iterations + substeps make contact resolution
    # (finger-object, object-table) much less prone to blowing up when
    # bodies interpenetrate slightly.
    p.setPhysicsEngineParameter(
        numSolverIterations=200,
        numSubSteps=4,
    )

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    p.resetDebugVisualizerCamera(
        cameraDistance=2.5,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0.4, 0.0, 0.6],
    )

    # --------------------------------------------------------
    # Floor
    # --------------------------------------------------------

    p.loadURDF("plane.urdf")

    # --------------------------------------------------------
    # Table
    # --------------------------------------------------------

    table = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=TABLE_HALF_EXTENTS,
        ),
        baseVisualShapeIndex=p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=TABLE_HALF_EXTENTS,
            rgbaColor=[0.35, 0.22, 0.12, 1],
        ),
        basePosition=TABLE_POSITION,
    )

    p.changeDynamics(
        table, -1,
        lateralFriction=0.8,
        restitution=0.0,
    )

    # --------------------------------------------------------
    # Object
    # --------------------------------------------------------

    object_collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=OBJECT_HALF_EXTENTS,
    )

    object_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=OBJECT_HALF_EXTENTS,
        rgbaColor=[0.9, 0.3, 0.1, 1],
    )

    obj = p.createMultiBody(
        baseMass=0.2,
        baseCollisionShapeIndex=object_collision,
        baseVisualShapeIndex=object_visual,
        basePosition=OBJECT_POSITION,
    )

    # FIX: tuned contact params. High default contact stiffness combined
    # with a small, light object being pinched by two fingers is exactly
    # the setup that produces jitter/"flicker" once contact is deep.
    p.changeDynamics(
        obj, -1,
        lateralFriction=1.0,
        spinningFriction=0.001,
        rollingFriction=0.001,
        restitution=0.0,
        contactStiffness=10000,
        contactDamping=50,
    )

    # --------------------------------------------------------
    # Panda
    # --------------------------------------------------------

    robot = p.loadURDF(
        "franka_panda/panda.urdf",
        ARM_BASE_POSITION,
        useFixedBase=True,
    )

    for finger in (LEFT_FINGER, RIGHT_FINGER):
        p.changeDynamics(
            robot, finger,
            lateralFriction=1.0,
            spinningFriction=0.001,
        )

    # --------------------------------------------------------
    # Initial arm configuration
    # --------------------------------------------------------

    home = [
        0.0,
        -0.4,
        0.0,
        -2.2,
        0.0,
        1.8,
        0.8,
    ]

    move_arm(robot, home, duration=1.0)

    open_gripper(robot)

    print()
    print("=" * 50)
    print("PyBullet Panda Grasp Demo")
    print("=" * 50)
    print()
    print("Starting GRASP demonstration...")
    print()

    # Give viewer time to see initial scene
    time.sleep(2)

    # --------------------------------------------------------
    # Move toward object (IK-driven, above the object)
    # --------------------------------------------------------

    print("Robot: approaching object")

    approach_pos = [
        OBJECT_POSITION[0],
        OBJECT_POSITION[1],
        OBJECT_POSITION[2] + 0.15,
    ]
    down_orn = p.getQuaternionFromEuler([math.pi, 0, 0])

    approach_pose = compute_grasp_joint_pose(robot, approach_pos, down_orn)
    move_arm(robot, approach_pose, duration=2.0)

    # --------------------------------------------------------
    # Descend to grasp height
    # --------------------------------------------------------

    print("Robot: lowering onto object")

    grasp_pos = [
        OBJECT_POSITION[0],
        OBJECT_POSITION[1],
        OBJECT_POSITION[2] + 0.02,
    ]

    grasp_pose = compute_grasp_joint_pose(robot, grasp_pos, down_orn)
    move_arm(robot, grasp_pose, duration=1.5)

    # --------------------------------------------------------
    # Close gripper
    # --------------------------------------------------------

    print("Robot: closing gripper")

    close_gripper(robot)

    time.sleep(0.5)

    # --------------------------------------------------------
    # Lift
    # --------------------------------------------------------

    print("Robot: lifting object")

    lift_pos = [
        OBJECT_POSITION[0],
        OBJECT_POSITION[1],
        OBJECT_POSITION[2] + 0.3,
    ]

    lift_pose = compute_grasp_joint_pose(robot, lift_pos, down_orn)
    move_arm(robot, lift_pose, duration=2.0)

    time.sleep(1)

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    print("Robot: returning")

    move_arm(robot, home, duration=2.0)

    # --------------------------------------------------------
    # Release
    # --------------------------------------------------------

    print("Robot: releasing object")

    open_gripper(robot)

    print()
    print("GRASP demonstration complete.")
    print()

    # --------------------------------------------------------
    # Keep simulation alive
    # --------------------------------------------------------

    while p.isConnected():
        p.stepSimulation()
        time.sleep(1 / 60)


if __name__ == "__main__":
    main()

import time
import math

import pybullet as p
import pybullet_data


# ============================================================
# Configuration
# ============================================================

SIMULATION_FPS = 60
MOTION_SPEED = 0.35

TABLE_HALF_EXTENTS = [0.35, 0.5, 0.1]
TABLE_POSITION = [0.6, 0.0, 0.2]
TABLE_TOP_Z = TABLE_POSITION[2] + TABLE_HALF_EXTENTS[2]

OBJECT_HALF_EXTENTS = [0.025, 0.025, 0.025]
OBJECT_POSITION = [
    0.55,
    0.0,
    TABLE_TOP_Z + OBJECT_HALF_EXTENTS[2],
]

ARM_BASE_POSITION = [0.0, 0.0, 0.0]

END_EFFECTOR_LINK = 11

ARM_JOINTS = [0, 1, 2, 3, 4, 5, 6]

LEFT_FINGER = 9
RIGHT_FINGER = 10


# ============================================================
# Motion utilities
# ============================================================

def move_arm(robot, targets, duration=1.5):

    current = [
        p.getJointState(robot, j)[0]
        for j in ARM_JOINTS
    ]

    steps = max(
        1,
        int(duration * SIMULATION_FPS)
    )

    for i in range(steps):

        t = (i + 1) / steps

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
                robot,
                joint,
                p.POSITION_CONTROL,
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


def ik_pose(robot, position):

    orientation = p.getQuaternionFromEuler(
        [math.pi, 0, 0]
    )

    result = p.calculateInverseKinematics(
        robot,
        END_EFFECTOR_LINK,
        position,
        orientation,
        maxNumIterations=200,
        residualThreshold=1e-5,
    )

    return list(result[:7])


# ============================================================
# Object attachment
# ============================================================

def attach_object(robot, obj):

    # FIX: the previous version hardcoded childFramePosition=[0,0,-0.10],
    # which assumed the end effector was 0.10 m above the object's
    # center at grasp time. But perform_grasp() actually descends to
    # OBJECT_POSITION[2] + 0.04, so the hand is only ~0.04 m above the
    # cube -- the hardcoded offset didn't match reality, and the fixed
    # constraint forcibly held the object ~0.14 m further from the
    # gripper than it visually should be, which is the gap you saw.
    #
    # Instead, read the *actual* current poses of the gripper and the
    # object and compute the object's pose relative to the end
    # effector's frame right now. Locking the constraint to that real,
    # current offset means whatever height you grasp at, the object
    # snaps to exactly where it already is -- no gap, no sudden jump.

    ee_pos, ee_orn = p.getLinkState(
        robot,
        END_EFFECTOR_LINK,
    )[:2]

    obj_pos, obj_orn = p.getBasePositionAndOrientation(obj)

    inv_ee_pos, inv_ee_orn = p.invertTransform(ee_pos, ee_orn)

    frame_pos, frame_orn = p.multiplyTransforms(
        inv_ee_pos,
        inv_ee_orn,
        obj_pos,
        obj_orn,
    )

    constraint = p.createConstraint(
        robot,
        END_EFFECTOR_LINK,
        obj,
        -1,
        p.JOINT_FIXED,
        [0, 0, 0],
        [0, 0, 0],
        frame_pos,
        childFrameOrientation=frame_orn,
    )

    return constraint


def detach_object(constraint):

    if constraint is not None:

        p.removeConstraint(constraint)


# ============================================================
# GRASP
# ============================================================

def perform_grasp(robot, obj):

    print()
    print("========================================")
    print("DECISION: GRASP")
    print("========================================")

    approach = [
        OBJECT_POSITION[0],
        OBJECT_POSITION[1],
        OBJECT_POSITION[2] + 0.18,
    ]

    grasp = [
        OBJECT_POSITION[0],
        OBJECT_POSITION[1],
        OBJECT_POSITION[2] + 0.04,
    ]

    lift = [
        OBJECT_POSITION[0],
        OBJECT_POSITION[1],
        OBJECT_POSITION[2] + 0.35,
    ]

    print("Robot → approaching object")

    move_arm(
        robot,
        ik_pose(robot, approach),
        duration=2.5,
    )

    print("Robot → lowering")

    move_arm(
        robot,
        ik_pose(robot, grasp),
        duration=1.5,
    )

    print("Robot → closing gripper")

    close_gripper(robot)

    time.sleep(0.5)

    print("Robot → grasp confirmed")

    # Attach object to gripper, using its *actual* current pose so
    # there's no offset mismatch (see fix note in attach_object).
    constraint = attach_object(
        robot,
        obj,
    )

    print("Robot → lifting")

    move_arm(
        robot,
        ik_pose(robot, lift),
        duration=2.5,
    )

    time.sleep(1)

    print("Robot → returning home")

    home = [
        0.0,
        -0.4,
        0.0,
        -2.2,
        0.0,
        1.8,
        0.8,
    ]

    move_arm(
        robot,
        home,
        duration=2.5,
    )

    print("Robot → releasing")

    open_gripper(robot)

    detach_object(constraint)

    # Put object back on table for next scenario.
    p.resetBasePositionAndOrientation(
        obj,
        OBJECT_POSITION,
        [0, 0, 0, 1],
    )

    print("GRASP complete.")


# ============================================================
# RE-SCAN
# ============================================================

def perform_rescan(robot):

    print()
    print("========================================")
    print("DECISION: RE-SCAN")
    print("========================================")

    scan_pose_1 = [
        OBJECT_POSITION[0],
        OBJECT_POSITION[1],
        OBJECT_POSITION[2] + 0.20,
    ]

    scan_pose_2 = [
        OBJECT_POSITION[0],
        OBJECT_POSITION[1] - 0.25,
        OBJECT_POSITION[2] + 0.20,
    ]

    print("Robot → moving to scan position")

    move_arm(
        robot,
        ik_pose(robot, scan_pose_1),
        duration=2.0,
    )

    time.sleep(1)

    print("Robot → changing viewpoint")

    move_arm(
        robot,
        ik_pose(robot, scan_pose_2),
        duration=2.0,
    )

    time.sleep(1)

    print("Robot → additional observation complete")

    home = [
        0.0,
        -0.4,
        0.0,
        -2.2,
        0.0,
        1.8,
        0.8,
    ]

    move_arm(
        robot,
        home,
        duration=2.0,
    )

    print("RE-SCAN complete. No grasp performed.")


# ============================================================
# ASK FOR HELP
# ============================================================

def perform_ask_help(robot):

    print()
    print("========================================")
    print("DECISION: ASK FOR HELP")
    print("========================================")

    print("Robot → unsafe / uncertain")
    print("Robot → no autonomous grasp")

    safe_pose = [
        0.0,
        -0.7,
        0.3,
        -2.4,
        0.0,
        1.8,
        0.8,
    ]

    move_arm(
        robot,
        safe_pose,
        duration=2.0,
    )

    time.sleep(2)

    print("Robot → waiting for human assistance")


# ============================================================
# Scene
# ============================================================

def create_scene():

    p.loadURDF("plane.urdf")

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
        table,
        -1,
        lateralFriction=0.8,
        restitution=0.0,
    )

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

    return obj


# ============================================================
# Main
# ============================================================

def main():

    p.connect(p.GUI)

    p.setAdditionalSearchPath(
        pybullet_data.getDataPath()
    )

    p.setGravity(0, 0, -9.81)

    p.setTimeStep(1 / 240)

    p.setPhysicsEngineParameter(
        numSolverIterations=200,
        numSubSteps=4,
    )

    p.resetDebugVisualizerCamera(
        cameraDistance=2.5,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0.4, 0.0, 0.6],
    )

    obj = create_scene()

    robot = p.loadURDF(
        "franka_panda/panda.urdf",
        ARM_BASE_POSITION,
        useFixedBase=True,
    )

    home = [
        0.0,
        -0.4,
        0.0,
        -2.2,
        0.0,
        1.8,
        0.8,
    ]

    move_arm(
        robot,
        home,
        duration=1.5,
    )

    open_gripper(robot)

    print()
    print("========================================")
    print(" UNCERTAINTY-AWARE ROBOT DEMO")
    print("========================================")
    print()
    print("1 → GRASP")
    print("2 → RE-SCAN")
    print("3 → ASK FOR HELP")
    print("q → QUIT")
    print()

    while p.isConnected():

        keys = p.getKeyboardEvents()

        if ord("1") in keys and keys[ord("1")] & p.KEY_WAS_TRIGGERED:

            perform_grasp(robot, obj)

        elif ord("2") in keys and keys[ord("2")] & p.KEY_WAS_TRIGGERED:

            perform_rescan(robot)

        elif ord("3") in keys and keys[ord("3")] & p.KEY_WAS_TRIGGERED:

            perform_ask_help(robot)

        elif ord("q") in keys and keys[ord("q")] & p.KEY_WAS_TRIGGERED:

            break

        p.stepSimulation()

        time.sleep(1 / 60)

    p.disconnect()


if __name__ == "__main__":
    main()

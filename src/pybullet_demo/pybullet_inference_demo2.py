#!/usr/bin/env python3

"""
PyBullet + REAL PointNet + MC Dropout presentation demo.

Pipeline:

    ModelNet40 sample
          |
    PointNet + MC Dropout
          |
    prediction + confidence + entropy
          |
    existing decision policy
          |
    GRASP / RE-SCAN / ASK HELP
          |
    PyBullet robot animation

The actual point cloud is rendered on the simulated object.
An invisible collision box acts as the physical grasp proxy.

The point cloud is explicitly transformed from the proxy object's
LOCAL coordinates to WORLD coordinates every few simulation steps,
so it follows the object during grasp, lift, and return.
"""

import os
import sys
import time
import glob

import numpy as np
import torch
import pybullet as p
import pybullet_data


# ============================================================
# Project imports
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

sys.path.insert(0, PROJECT_ROOT)

from src.data.modelnet40_dataset import ModelNet40Dataset
from src.data.corruption import occlude
from src.models.pointnet import PointNetClassifier, enable_mc_dropout
from src.utils.config import OOD_CLASSES, NUM_POINTS
from src.inference.mc_dropout_inference import (
    mc_dropout_predict,
    compute_uncertainty,
)
from src.evaluation.decision_policy import classify_action


# ============================================================
# Matplotlib
# ============================================================

import matplotlib

matplotlib.use(
    "TkAgg",
    force=True,
)

import matplotlib.pyplot as plt

print(
    "FINAL BACKEND:",
    matplotlib.get_backend(),
)

print(
    "FINAL CANVAS:",
    type(plt.figure().canvas),
)

plt.close()


# ============================================================
# Configuration
# ============================================================

GRASP_THRESH = 0.877
RESCAN_THRESH = 0.777

T_PASSES = 10

DEVICE = torch.device("cpu")

SIMULATION_FPS = 60
MOTION_SPEED = 0.35

TABLE_TOP_Z = 0.3

OBJECT_HALF_EXTENTS = [
    0.025,
    0.025,
    0.025,
]

OBJECT_POSITION = [
    0.55,
    0.0,
    TABLE_TOP_Z + 0.025,
]

PANDA_BASE = [
    0.0,
    0.0,
    0.0,
]

END_EFFECTOR_LINK = 11

ARM_JOINTS = list(range(7))

LEFT_FINGER = 9
RIGHT_FINGER = 10


# ============================================================
# Point-cloud overlay
# ============================================================

OVERLAY_SCALE = 0.12

OVERLAY_MAX_POINTS = 1200

OVERLAY_POINT_SIZE = 8

OVERLAY_REFRESH_EVERY_N_STEPS = 3


DECISION_COLORS = {
    "Grasp": [
        0.2,
        0.9,
        0.2,
    ],
    "Re-scan": [
        0.9,
        0.9,
        0.2,
    ],
    "Ask for help": [
        0.9,
        0.2,
        0.2,
    ],
}


# Global overlay state.
_overlay_state = None


# ============================================================
# Point Cloud Window
# ============================================================

class PointCloudWindow:

    def __init__(self):

        self.fig = plt.figure(
            "PointNet Sensor Scan",
            figsize=(7, 6),
        )

        self.ax = self.fig.add_subplot(
            111,
            projection="3d",
        )

        self.scatter = None

        self._draw_empty()

        print(
            "BACKEND:",
            matplotlib.get_backend(),
        )

        print(
            "CANVAS:",
            type(self.fig.canvas),
        )

        self.fig.canvas.draw()

        self.fig.canvas.manager.show()

        self.fig.canvas.flush_events()

        plt.pause(0.01)

    def _draw_empty(self):

        self.ax.set_title(
            "WAITING FOR SENSOR SCAN",
            fontsize=14,
        )

        self.ax.set_xlim(
            -1,
            1,
        )

        self.ax.set_ylim(
            -1,
            1,
        )

        self.ax.set_zlim(
            -1,
            1,
        )

        self.ax.set_xticks([])

        self.ax.set_yticks([])

        self.ax.set_zticks([])

    def update(
        self,
        points,
        label=None,
    ):

        points = np.asarray(
            points,
            dtype=np.float32,
        )

        points = (
            points
            - points.mean(axis=0)
        )

        max_distance = np.max(
            np.linalg.norm(
                points,
                axis=1,
            )
        )

        if max_distance > 0:

            points = (
                points
                / max_distance
            )

        if len(points) > 512:

            indices = np.linspace(
                0,
                len(points) - 1,
                512,
                dtype=int,
            )

            points = points[
                indices
            ]

        if self.scatter is None:

            self.scatter = self.ax.scatter(
                points[:, 0],
                points[:, 1],
                points[:, 2],
                s=10,
                c=points[:, 2],
                cmap="viridis",
                depthshade=False,
            )

        else:

            self.scatter._offsets3d = (
                points[:, 0],
                points[:, 1],
                points[:, 2],
            )

            self.scatter.set_array(
                points[:, 2]
            )

        self.ax.set_title(
            label
            if label
            else "CURRENT SENSOR SCAN",
            fontsize=14,
        )

        self.fig.canvas.draw_idle()

        self.fig.canvas.flush_events()

        plt.pause(0.001)

    def close(self):

        plt.close(
            self.fig
        )


# ============================================================
# Checkpoint
# ============================================================

def find_latest_checkpoint():

    experiments_dir = os.path.join(
        PROJECT_ROOT,
        "experiments",
    )

    candidates = sorted(
        glob.glob(
            os.path.join(
                experiments_dir,
                "dropout_ablation_p0.1_*",
            )
        )
    )

    if not candidates:

        raise FileNotFoundError(
            "Could not find dropout_ablation_p0.1_* experiment."
        )

    return os.path.join(
        candidates[-1],
        "checkpoints",
        "best.pth",
    )


# ============================================================
# Load model
# ============================================================

def load_model():

    dataset = ModelNet40Dataset(
        split="test",
        num_points=NUM_POINTS,
        excluded_classes=OOD_CLASSES,
    )

    classes = dataset.classes

    checkpoint_path = (
        find_latest_checkpoint()
    )

    print()

    print(
        "=" * 60
    )

    print(
        "LOADING POINTNET"
    )

    print(
        "=" * 60
    )

    print(
        f"Checkpoint : {checkpoint_path}"
    )

    print(
        f"Classes    : {len(classes)}"
    )

    print(
        f"MC passes  : {T_PASSES}"
    )

    print()

    model = PointNetClassifier(
        num_classes=len(classes)
    ).to(DEVICE)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    enable_mc_dropout(
        model
    )

    print(
        "Model loaded."
    )

    print(
        "=" * 60
    )

    return model, dataset


# ============================================================
# Inference
# ============================================================

@torch.no_grad()
def run_inference(
    model,
    points,
    classes,
    point_cloud_display=None,
):

    points = np.asarray(
        points,
        dtype=np.float32,
    )

    if len(points) != NUM_POINTS:

        if len(points) == 0:

            raise ValueError(
                "Received empty point cloud."
            )

        indices = np.random.choice(
            len(points),
            NUM_POINTS,
            replace=len(points)
            < NUM_POINTS,
        )

        points = points[
            indices
        ]

    # Show exact point cloud sent to PointNet.
    if point_cloud_display is not None:

        point_cloud_display.update(
            points
        )

    x = torch.from_numpy(
        points
    ).float().unsqueeze(
        0
    ).to(DEVICE)

    probs_T = mc_dropout_predict(
        model,
        x,
        T_PASSES,
    )

    stats = compute_uncertainty(
        probs_T
    )

    pred = int(
        stats[
            "pred_class"
        ][0].item()
    )

    confidence = float(
        stats[
            "mean_probs"
        ][0, pred].item()
    )

    entropy = float(
        stats[
            "total_entropy"
        ][0].item()
    )

    epistemic = float(
        stats[
            "epistemic"
        ][0].item()
    )

    aleatoric = float(
        stats[
            "aleatoric"
        ][0].item()
    )

    prediction_name = classes[
        pred
    ]

    return {
        "prediction": prediction_name,
        "prediction_index": pred,
        "confidence": confidence,
        "entropy": entropy,
        "epistemic": epistemic,
        "aleatoric": aleatoric,
    }


# ============================================================
# Decision
# ============================================================

def make_decision(
    confidence
):

    return classify_action(
        confidence,
        GRASP_THRESH,
        RESCAN_THRESH,
    )


# ============================================================
# Point-cloud overlay
# ============================================================

def create_point_cloud_overlay(
    points,
    obj_id,
    color=None,
    scale=OVERLAY_SCALE,
):
    """
    Store the point cloud in the proxy object's LOCAL frame.

    The cloud is then transformed into WORLD coordinates using
    the current pose of the invisible proxy object.

    FIX: the previous overlay's debug item is explicitly removed
    here before the new one is created, otherwise PyBullet leaves
    the old point cloud rendered on the table when a new example
    starts (e.g. the "ghost" cloud left behind after a grasp).
    """

    global _overlay_state

    # --------------------------------------------------------
    # Clear the previous point cloud, if any.
    # --------------------------------------------------------

    if _overlay_state is not None:

        old_debug_id = _overlay_state.get(
            "debug_id"
        )

        if old_debug_id is not None:

            try:

                p.removeUserDebugItem(
                    old_debug_id
                )

            except Exception:
                pass

    if color is None:

        color = [
            0.9,
            0.5,
            0.1,
        ]

    pts = np.asarray(
        points,
        dtype=np.float32,
    ).copy()

    if len(pts) == 0:

        return None

    # Center cloud.
    pts = (
        pts
        - pts.mean(axis=0)
    )

    # Normalize.
    max_d = np.max(
        np.linalg.norm(
            pts,
            axis=1,
        )
    )

    if max_d > 0:

        pts = (
            pts
            / max_d
        )

    # Scale.
    pts = (
        pts
        * scale
    )

    # Render subset.
    if len(pts) > OVERLAY_MAX_POINTS:

        idx = np.linspace(
            0,
            len(pts) - 1,
            OVERLAY_MAX_POINTS,
            dtype=int,
        )

        pts = pts[
            idx
        ]

    state = {
        "obj_id": obj_id,
        "local_points": pts,
        "color": color,
        "debug_id": None,
    }

    refresh_point_cloud_overlay(
        state
    )

    return state


def refresh_point_cloud_overlay(
    state
):
    """
    Transform the stored LOCAL point cloud into WORLD coordinates
    using the proxy object's current position and orientation.
    """

    if state is None:
        return

    obj_id = state[
        "obj_id"
    ]

    if obj_id is None:
        return

    local_points = state[
        "local_points"
    ]

    if local_points is None:
        return

    # --------------------------------------------------------
    # Current object pose
    # --------------------------------------------------------

    obj_position, obj_orientation = (
        p.getBasePositionAndOrientation(
            obj_id
        )
    )

    obj_position = np.asarray(
        obj_position,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Quaternion -> rotation matrix
    # --------------------------------------------------------

    rotation = np.asarray(
        p.getMatrixFromQuaternion(
            obj_orientation
        ),
        dtype=np.float32,
    ).reshape(
        3,
        3,
    )

    # --------------------------------------------------------
    # Local -> world
    # --------------------------------------------------------

    world_points = (
        local_points
        @ rotation.T
    ) + obj_position

    # --------------------------------------------------------
    # Remove previous cloud
    # --------------------------------------------------------

    if state[
        "debug_id"
    ] is not None:

        try:

            p.removeUserDebugItem(
                state[
                    "debug_id"
                ]
            )

        except Exception:
            pass

    # --------------------------------------------------------
    # Draw current cloud
    # --------------------------------------------------------

    colors = [
        state["color"]
        for _ in range(
            len(world_points)
        )
    ]

    state[
        "debug_id"
    ] = p.addUserDebugPoints(
        world_points.tolist(),
        colors,
        pointSize=OVERLAY_POINT_SIZE,
    )


def clear_point_cloud_overlay():

    global _overlay_state

    if _overlay_state is None:

        return

    if (
        _overlay_state["debug_id"]
        is not None
    ):

        try:

            p.removeUserDebugItem(
                _overlay_state[
                    "debug_id"
                ]
            )

        except Exception:
            pass

    _overlay_state = None


# ============================================================
# PyBullet setup
# ============================================================

def setup_simulation():

    physics_client = p.connect(
        p.GUI
    )

    p.setAdditionalSearchPath(
        pybullet_data.getDataPath()
    )

    p.setGravity(
        0,
        0,
        -9.81,
    )

    p.setPhysicsEngineParameter(
        numSolverIterations=200,
        numSubSteps=4,
    )

    # --------------------------------------------------------
    # Plane
    # --------------------------------------------------------

    p.loadURDF(
        "plane.urdf"
    )

    # --------------------------------------------------------
    # Table
    # --------------------------------------------------------

    table_shape = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[
            0.35,
            0.5,
            0.1,
        ],
    )

    table_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[
            0.35,
            0.5,
            0.1,
        ],
    )

    table = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=table_shape,
        baseVisualShapeIndex=table_visual,
        basePosition=[
            0.6,
            0.0,
            0.2,
        ],
    )

    # --------------------------------------------------------
    # Invisible collision proxy
    # --------------------------------------------------------

    object_collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=OBJECT_HALF_EXTENTS,
    )

    object_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=OBJECT_HALF_EXTENTS,
        rgbaColor=[
            0.9,
            0.3,
            0.05,
            0.05,
        ],
    )

    obj = p.createMultiBody(
        baseMass=0.05,
        baseCollisionShapeIndex=object_collision,
        baseVisualShapeIndex=object_visual,
        basePosition=OBJECT_POSITION,
    )

    # --------------------------------------------------------
    # Panda
    # --------------------------------------------------------

    robot = p.loadURDF(
        "franka_panda/panda.urdf",
        PANDA_BASE,
        useFixedBase=True,
    )

    # --------------------------------------------------------
    # Friction
    # --------------------------------------------------------

    p.changeDynamics(
        obj,
        -1,
        lateralFriction=1.0,
        spinningFriction=0.001,
        rollingFriction=0.001,
        restitution=0,
        contactStiffness=10000,
        contactDamping=50,
    )

    p.changeDynamics(
        robot,
        LEFT_FINGER,
        lateralFriction=1.0,
    )

    p.changeDynamics(
        robot,
        RIGHT_FINGER,
        lateralFriction=1.0,
    )

    return robot, obj


# ============================================================
# Motion helpers
# ============================================================

def move_arm(
    robot,
    target_joints,
):

    global _overlay_state

    current = [
        p.getJointState(
            robot,
            joint,
        )[0]
        for joint in ARM_JOINTS
    ]

    max_difference = max(
        abs(a - b)
        for a, b in zip(
            current,
            target_joints,
        )
    )

    steps = max(
        1,
        int(
            max_difference
            * 120
        ),
    )

    for step in range(
        steps + 1
    ):

        t = (
            step
            / steps
        )

        smooth_t = (
            t * t
            * (3 - 2 * t)
        )

        for i, joint in enumerate(
            ARM_JOINTS
        ):

            value = (
                (1 - smooth_t)
                * current[i]
                + smooth_t
                * target_joints[i]
            )

            p.setJointMotorControl2(
                robot,
                joint,
                p.POSITION_CONTROL,
                targetPosition=value,
                force=200,
            )

        # Physics.
        p.stepSimulation()

        # ----------------------------------------------------
        # Refresh point cloud while object moves.
        # ----------------------------------------------------

        if (
            _overlay_state
            is not None
            and step
            % OVERLAY_REFRESH_EVERY_N_STEPS
            == 0
        ):

            refresh_point_cloud_overlay(
                _overlay_state
            )

        time.sleep(
            (
                1
                / SIMULATION_FPS
            )
            / MOTION_SPEED
        )


def open_gripper(
    robot
):

    global _overlay_state

    for step in range(
        30
    ):

        p.setJointMotorControl2(
            robot,
            LEFT_FINGER,
            p.POSITION_CONTROL,
            targetPosition=0.04,
            force=50,
        )

        p.setJointMotorControl2(
            robot,
            RIGHT_FINGER,
            p.POSITION_CONTROL,
            targetPosition=0.04,
            force=50,
        )

        p.stepSimulation()

        if (
            _overlay_state
            is not None
            and step
            % OVERLAY_REFRESH_EVERY_N_STEPS
            == 0
        ):

            refresh_point_cloud_overlay(
                _overlay_state
            )

        time.sleep(
            (
                1
                / SIMULATION_FPS
            )
            / MOTION_SPEED
        )


def close_gripper(
    robot
):

    global _overlay_state

    for step in range(
        30
    ):

        p.setJointMotorControl2(
            robot,
            LEFT_FINGER,
            p.POSITION_CONTROL,
            targetPosition=0.0,
            force=50,
        )

        p.setJointMotorControl2(
            robot,
            RIGHT_FINGER,
            p.POSITION_CONTROL,
            targetPosition=0.0,
            force=50,
        )

        p.stepSimulation()

        if (
            _overlay_state
            is not None
            and step
            % OVERLAY_REFRESH_EVERY_N_STEPS
            == 0
        ):

            refresh_point_cloud_overlay(
                _overlay_state
            )

        time.sleep(
            (
                1
                / SIMULATION_FPS
            )
            / MOTION_SPEED
        )


# ============================================================
# IK
# ============================================================

def compute_grasp_joint_pose(
    robot,
    target_position,
):

    target_orientation = (
        p.getQuaternionFromEuler(
            [
                np.pi,
                0,
                0,
            ]
        )
    )

    solution = (
        p.calculateInverseKinematics(
            robot,
            END_EFFECTOR_LINK,
            target_position,
            targetOrientation=target_orientation,
        )
    )

    return list(
        solution[:7]
    )


# ============================================================
# Grasp
# ============================================================

def perform_grasp(
    robot,
    obj,
):

    global _overlay_state

    print()

    print(
        "ACTION: GRASP"
    )

    print()

    # --------------------------------------------------------
    # Home
    # --------------------------------------------------------

    home = [
        0,
        -0.4,
        0,
        -2.2,
        0,
        1.8,
        0.8,
    ]

    move_arm(
        robot,
        home,
    )

    open_gripper(
        robot
    )

    # --------------------------------------------------------
    # Approach
    # --------------------------------------------------------

    approach_position = [
        OBJECT_POSITION[0],
        OBJECT_POSITION[1],
        OBJECT_POSITION[2] + 0.12,
    ]

    grasp_position = [
        OBJECT_POSITION[0],
        OBJECT_POSITION[1],
        OBJECT_POSITION[2] + 0.055,
    ]

    approach_pose = (
        compute_grasp_joint_pose(
            robot,
            approach_position,
        )
    )

    grasp_pose = (
        compute_grasp_joint_pose(
            robot,
            grasp_position,
        )
    )

    print(
        "Approaching object"
    )

    move_arm(
        robot,
        approach_pose,
    )

    print(
        " Moving to grasp position"
    )

    move_arm(
        robot,
        grasp_pose,
    )

    # --------------------------------------------------------
    # Close
    # --------------------------------------------------------

    print(
        "Closing gripper"
    )

    close_gripper(
        robot
    )

    # --------------------------------------------------------
    # Attach physical proxy to gripper
    # --------------------------------------------------------

    constraint = p.createConstraint(
        parentBodyUniqueId=robot,
        parentLinkIndex=END_EFFECTOR_LINK,
        childBodyUniqueId=obj,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[
            0,
            0,
            0,
        ],
        parentFramePosition=[
            0,
            0,
            0.04,
        ],
        childFramePosition=[
            0,
            0,
            0,
        ],
    )

    # --------------------------------------------------------
    # Immediately refresh after attachment.
    # --------------------------------------------------------

    if _overlay_state is not None:

        refresh_point_cloud_overlay(
            _overlay_state
        )

    # --------------------------------------------------------
    # Lift
    # --------------------------------------------------------

    lift_position = [
        OBJECT_POSITION[0],
        OBJECT_POSITION[1],
        OBJECT_POSITION[2] + 0.35,
    ]

    lift_pose = (
        compute_grasp_joint_pose(
            robot,
            lift_position,
        )
    )

    print(
        "   -> Lifting object"
    )

    move_arm(
        robot,
        lift_pose,
    )

    # --------------------------------------------------------
    # Return home
    # --------------------------------------------------------

    print(
        "   -> Returning home"
    )

    move_arm(
        robot,
        home,
    )

    # --------------------------------------------------------
    # Release
    # --------------------------------------------------------

    print(
        "   -> Releasing"
    )

    open_gripper(
        robot
    )

    p.removeConstraint(
        constraint
    )

    # --------------------------------------------------------
    # Put proxy back on table.
    # --------------------------------------------------------

    p.resetBasePositionAndOrientation(
        obj,
        OBJECT_POSITION,
        [
            0,
            0,
            0,
            1,
        ],
    )

    p.resetBaseVelocity(
        obj,
        [
            0,
            0,
            0,
        ],
        [
            0,
            0,
            0,
        ],
    )

    if _overlay_state is not None:

        refresh_point_cloud_overlay(
            _overlay_state
        )

    print(
        "GRASP COMPLETE"
    )


# ============================================================
# Re-scan
# ============================================================

def perform_rescan(
    robot
):

    print()

    print(
        "ACTION: RE-SCAN"
    )

    print()

    home = [
        0,
        -0.4,
        0,
        -2.2,
        0,
        1.8,
        0.8,
    ]

    move_arm(
        robot,
        home,
    )

    viewpoint_1 = [
        0.3,
        -0.2,
        0,
        -2.0,
        0,
        1.6,
        0.8,
    ]

    viewpoint_2 = [
        -0.3,
        -0.2,
        0,
        -2.0,
        0,
        1.6,
        0.8,
    ]

    print(
        " Moving to viewpoint 1"
    )

    move_arm(
        robot,
        viewpoint_1,
    )

    time.sleep(
        1.0
    )

    print(
        " Moving to viewpoint 2"
    )

    move_arm(
        robot,
        viewpoint_2,
    )

    time.sleep(
        1.0
    )

    print(
        " Returning home"
    )

    move_arm(
        robot,
        home,
    )

    print(
        " RE-SCAN COMPLETE"
    )


# ============================================================
# Ask for help
# ============================================================

def perform_ask_help(
    robot
):

    print()

    print(
        "ACTION: ASK FOR HELP"
    )

    print()

    home = [
        0,
        -0.4,
        0,
        -2.2,
        0,
        1.8,
        0.8,
    ]

    safe_pose = [
        0,
        -0.1,
        0,
        -1.5,
        0,
        1.2,
        0.8,
    ]

    print(
        "Retreating to safe pose"
    )

    move_arm(
        robot,
        safe_pose,
    )

    time.sleep(
        2.0
    )

    print(
        "   -> Returning home"
    )

    move_arm(
        robot,
        home,
    )

    print(
        " HUMAN ASSISTANCE REQUESTED"
    )


# ============================================================
# Display result
# ============================================================

def print_result(
    result,
    action,
):

    print()

    print(
        "=" * 60
    )

    print(
        "POINTNET + MC DROPOUT RESULT"
    )

    print(
        "=" * 60
    )

    print(
        f"Prediction : "
        f"{result['prediction']}"
    )

    print(
        f"Confidence : "
        f"{result['confidence'] * 100:.2f}%"
    )

    print(
        f"Entropy    : "
        f"{result['entropy']:.3f}"
    )

    print(
        f"Epistemic  : "
        f"{result['epistemic']:.3f}"
    )

    print(
        f"Aleatoric  : "
        f"{result['aleatoric']:.3f}"
    )

    print(
        f"Decision   : "
        f"{action.upper()}"
    )

    print(
        "=" * 60
    )

    print()


# ============================================================
# Select demo examples
# ============================================================

def build_examples(
    dataset
):

    examples = []

    # --------------------------------------------------------
    # Clean examples
    # --------------------------------------------------------

    print(
        "Searching for confident clean example..."
    )

    for i in range(
        min(
            50,
            len(dataset),
        )
    ):

        points, label = dataset[
            i
        ]

        examples.append(
            (
                "Clean",
                points,
                label,
            )
        )

        if len(examples) >= 10:
            break

    # --------------------------------------------------------
    # Occluded example
    # --------------------------------------------------------

    clean_points = examples[
        0
    ][1]

    occluded_points = occlude(
        clean_points.copy(),
        fraction=0.6,
    )

    examples.append(
        (
            "Occluded 60%",
            occluded_points,
            examples[0][2],
        )
    )

    # --------------------------------------------------------
    # OOD example
    # --------------------------------------------------------

    ood_dataset = ModelNet40Dataset(
        split="ood",
        num_points=NUM_POINTS,
        excluded_classes=OOD_CLASSES,
    )

    ood_points, ood_label = (
        ood_dataset[0]
    )

    examples.append(
        (
            "OOD",
            ood_points,
            ood_label,
        )
    )

    return examples


# ============================================================
# Main
# ============================================================

def main():

    global _overlay_state

    print()

    print(
        "=" * 60
    )

    print(
        "      UNCERTAINTY-AWARE POINTNET ROBOT DEMO"
    )

    print(
        "=" * 60
    )

    print()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model, dataset = load_model()

    # --------------------------------------------------------
    # Point cloud window
    # --------------------------------------------------------

    point_cloud_display = (
        PointCloudWindow()
    )

    print(
        ">>> MATPLOTLIB WINDOW CREATED"
    )

    input(
        ">>> If you see the point-cloud window, press ENTER here..."
    )

    # --------------------------------------------------------
    # Build examples
    # --------------------------------------------------------

    examples = build_examples(
        dataset
    )

    # --------------------------------------------------------
    # Start PyBullet
    # --------------------------------------------------------

    robot, obj = setup_simulation()

    # --------------------------------------------------------
    # Home pose
    # --------------------------------------------------------

    home = [
        0,
        -0.4,
        0,
        -2.2,
        0,
        1.8,
        0.8,
    ]

    move_arm(
        robot,
        home,
    )

    open_gripper(
        robot
    )

    # --------------------------------------------------------
    # Initial neutral point cloud
    # --------------------------------------------------------

    _overlay_state = (
        create_point_cloud_overlay(
            examples[0][1],
            obj,
        )
    )

    # --------------------------------------------------------
    # Run examples
    # --------------------------------------------------------

    for (
        example_name,
        points,
        true_label,
    ) in examples:

        print()
        print()

        print(
            "=" * 60
        )

        print(
            f"DEMO CASE: "
            f"{example_name}"
        )

        print(
            "=" * 60
        )

        # ----------------------------------------------------
        # Inference
        # ----------------------------------------------------

        result = run_inference(
            model,
            points,
            dataset.classes,
            point_cloud_display,
        )

        # ----------------------------------------------------
        # Decision
        # ----------------------------------------------------

        action = make_decision(
            result[
                "confidence"
            ]
        )

        print_result(
            result,
            action,
        )

        # ----------------------------------------------------
        # Replace point cloud
        # ----------------------------------------------------

        _overlay_state = (
            create_point_cloud_overlay(
                points,
                obj,
                color=DECISION_COLORS.get(
                    action
                ),
            )
        )

        # ----------------------------------------------------
        # Let audience see result
        # ----------------------------------------------------

        time.sleep(
            2.0
        )

        # ----------------------------------------------------
        # Execute policy
        # ----------------------------------------------------

        if action == "Grasp":

            perform_grasp(
                robot,
                obj,
            )

        elif action == "Re-scan":

            perform_rescan(
                robot,
            )

        else:

            perform_ask_help(
                robot,
            )

        # ----------------------------------------------------
        # Pause between examples
        # ----------------------------------------------------

        time.sleep(
            3.0
        )

    # --------------------------------------------------------
    # Keep simulation alive
    # --------------------------------------------------------

    print()

    print(
        "=" * 60
    )

    print(
        "DEMO COMPLETE"
    )

    print(
        "Close the PyBullet window to exit."
    )

    print(
        "=" * 60
    )

    while p.isConnected():

        p.stepSimulation()

        # Keep overlay attached even while idle.
        if _overlay_state is not None:

            refresh_point_cloud_overlay(
                _overlay_state
            )

        time.sleep(
            1
            / SIMULATION_FPS
        )

    point_cloud_display.close()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()

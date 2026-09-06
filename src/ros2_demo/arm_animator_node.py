#!/usr/bin/env python3

"""
Presentation-oriented robotic arm animation.

The arm does NOT perform real IK or physics simulation.

It receives the REAL decision from inference_node.py:

    Grasp       -> reach -> close -> lift -> return -> release
    Re-scan     -> move camera/arm sideways -> return
    Ask for help -> retreat -> stay safe

Additionally publishes:
    /grasp_object  -> visual object marker
    /scene         -> table marker

This makes the ROS2 demo visually communicate:

    sensor -> PointNet -> uncertainty -> decision -> robot action
"""

import rclpy

from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from visualization_msgs.msg import Marker


JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "gripper_left_joint",
    "gripper_right_joint",
]

HOME_POSE = [
    0.0,   # shoulder pan
    0.0,   # shoulder lift
    0.0,   # elbow
    0.0,   # left finger
    0.0,   # right finger
]


# ------------------------------------------------------------
# Arm animations
# ------------------------------------------------------------

GRASP_SEQUENCE = [

    # Reach toward object
    (1.5, [0.0, 0.8, -1.0, 0.0, 0.0]),

    # Close gripper
    (0.7, [0.0, 0.8, -1.0, 0.045, 0.045]),

    # Lift object
    (1.5, [0.0, -0.3, 0.2, 0.045, 0.045]),

    # Return home while holding
    (1.2, [0.0, 0.0, 0.0, 0.045, 0.045]),

    # Release
    (0.7, [0.0, 0.0, 0.0, 0.0, 0.0]),
]


RESCAN_SEQUENCE = [

    # Move toward object
    (0.8, [0.25, 0.15, -0.15, 0.0, 0.0]),

    # Look from other side
    (1.0, [-0.30, 0.15, -0.15, 0.0, 0.0]),

    # Return
    (0.8, [0.0, 0.0, 0.0, 0.0, 0.0]),
]


ASK_HELP_SEQUENCE = [

    # Retreat
    (1.0, [0.0, -0.25, 0.35, 0.0, 0.0]),

    # Home
    (1.0, HOME_POSE),
]


PUBLISH_RATE_HZ = 30.0


class ArmAnimatorNode(Node):

    def __init__(self):

        super().__init__("arm_animator_node")

        # ----------------------------------------------------
        # ROS topics
        # ----------------------------------------------------

        self.joint_pub = self.create_publisher(
            JointState,
            "/joint_states",
            10,
        )

        self.object_pub = self.create_publisher(
            Marker,
            "/grasp_object",
            10,
        )

        self.scene_pub = self.create_publisher(
            Marker,
            "/scene",
            10,
        )

        self.subscription = self.create_subscription(
            String,
            "/policy_decision",
            self.on_decision,
            10,
        )

        # ----------------------------------------------------
        # State
        # ----------------------------------------------------

        self.current_pose = list(HOME_POSE)

        self.animation_queue = []

        self.segment_elapsed = 0.0

        self.segment_start_pose = list(HOME_POSE)

        self.current_action = "Idle"

        # Object state
        self.object_grasped = False

        # Object location.
        #
        # This is intentionally close to where the demo arm reaches.
        #
        # If the object appears slightly away from the gripper in RViz,
        # we can tune this after seeing the animation.
        self.object_home = [-1.85, 0.0, 0.12]

        self.object_lifted = [-1.85, 0.0, 0.75]

        # ----------------------------------------------------
        # Timer
        # ----------------------------------------------------

        self.timer = self.create_timer(
            1.0 / PUBLISH_RATE_HZ,
            self.tick,
        )

        # Publish static scene immediately
        self.publish_scene()

        self.get_logger().info(
            "Arm animator ready. Waiting for /policy_decision..."
        )

    # ========================================================
    # Decision callback
    # ========================================================

    def on_decision(self, msg: String):

        action = msg.data

        if action == "Grasp":

            sequence = GRASP_SEQUENCE

        elif action == "Re-scan":

            sequence = RESCAN_SEQUENCE

        else:

            sequence = ASK_HELP_SEQUENCE

        self.current_action = action

        # Reset object before a new grasp animation
        if action == "Grasp":
            self.object_grasped = False

        self.animation_queue = list(sequence)

        self.segment_elapsed = 0.0

        self.segment_start_pose = list(self.current_pose)

        self.get_logger().info(
            f"Decision received: {action}"
        )

    # ========================================================
    # Main animation loop
    # ========================================================

    def tick(self):

        dt = 1.0 / PUBLISH_RATE_HZ

        if self.animation_queue:

            duration, target_pose = self.animation_queue[0]

            self.segment_elapsed += dt

            t = min(
                self.segment_elapsed / duration,
                1.0,
            )

            # Smooth interpolation
            smooth_t = t * t * (3.0 - 2.0 * t)

            self.current_pose = [

                (1.0 - smooth_t) * start
                + smooth_t * end

                for start, end
                in zip(
                    self.segment_start_pose,
                    target_pose,
                )
            ]

            # ------------------------------------------------
            # Detect grasp closing
            # ------------------------------------------------

            if (
                self.current_action == "Grasp"
                and self.current_pose[3] > 0.035
            ):
                self.object_grasped = True

            # ------------------------------------------------
            # Segment complete
            # ------------------------------------------------

            if t >= 1.0:

                self.animation_queue.pop(0)

                self.segment_elapsed = 0.0

                self.segment_start_pose = list(
                    self.current_pose
                )

                # When entire grasp animation finishes,
                # put object back on table.
                if (
                    self.current_action == "Grasp"
                    and not self.animation_queue
                ):

                    self.object_grasped = False

                    self.current_action = "Idle"

        # Publish arm
        self.publish_joint_state()

        # Publish object
        self.publish_object()

    # ========================================================
    # JointState
    # ========================================================

    def publish_joint_state(self):

        msg = JointState()

        msg.header.stamp = (
            self.get_clock().now().to_msg()
        )

        msg.name = JOINT_NAMES

        msg.position = self.current_pose

        self.joint_pub.publish(msg)

    # ========================================================
    # Object marker
    # ========================================================

    def publish_object(self):

        marker = Marker()

        marker.header.frame_id = "map"

        marker.header.stamp = (
            self.get_clock().now().to_msg()
        )

        marker.ns = "grasp_object"

        marker.id = 0

        marker.type = Marker.CYLINDER

        marker.action = Marker.ADD

        # ----------------------------------------------------
        # Move object when grasped
        # ----------------------------------------------------

        if self.object_grasped:

            marker.pose.position.x = (
                self.object_lifted[0]
            )

            marker.pose.position.y = (
                self.object_lifted[1]
            )

            marker.pose.position.z = (
                self.object_lifted[2]
            )

        else:

            marker.pose.position.x = (
                self.object_home[0]
            )

            marker.pose.position.y = (
                self.object_home[1]
            )

            marker.pose.position.z = (
                self.object_home[2]
            )

        marker.pose.orientation.w = 1.0

        # Object dimensions
        marker.scale.x = 0.18
        marker.scale.y = 0.18
        marker.scale.z = 0.24

        # Orange-ish object
        marker.color.r = 0.9
        marker.color.g = 0.45
        marker.color.b = 0.1
        marker.color.a = 1.0

        marker.lifetime.sec = 0

        self.object_pub.publish(marker)

    # ========================================================
    # Table / environment
    # ========================================================

    def publish_scene(self):

        marker = Marker()

        marker.header.frame_id = "map"

        marker.header.stamp = (
            self.get_clock().now().to_msg()
        )

        marker.ns = "scene"

        marker.id = 0

        marker.type = Marker.CUBE

        marker.action = Marker.ADD

        # Table top
        marker.pose.position.x = -0.8
        marker.pose.position.y = 0.0
        marker.pose.position.z = -0.05

        marker.pose.orientation.w = 1.0

        marker.scale.x = 2.4
        marker.scale.y = 1.4
        marker.scale.z = 0.10

        marker.color.r = 0.35
        marker.color.g = 0.22
        marker.color.b = 0.12
        marker.color.a = 1.0

        self.scene_pub.publish(marker)


def main():
    rclpy.init()
    node = ArmAnimatorNode()
    try:
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":

    main()

#!/usr/bin/env python3

import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


BASE_DIR = os.path.dirname(__file__)

URDF_PATH = os.path.join(
    BASE_DIR,
    "simple_arm.urdf",
)

ARM_ANIMATOR_PATH = os.path.join(
    BASE_DIR,
    "arm_animator_node.py",
)


def generate_launch_description():

    with open(URDF_PATH, "r") as f:

        robot_description = f.read()

    return LaunchDescription([

        # ----------------------------------------------------
        # Put robot base inside map
        # ----------------------------------------------------

        ExecuteProcess(
            cmd=[
                "ros2",
                "run",
                "tf2_ros",
                "static_transform_publisher",

                "--x", "-1.5",
                "--y", "0",
                "--z", "0",

                "--frame-id", "map",
                "--child-frame-id", "base_link",
            ],

            output="screen",
        ),

        # ----------------------------------------------------
        # Publish URDF + joint transforms
        # ----------------------------------------------------

        Node(
            package="robot_state_publisher",

            executable="robot_state_publisher",

            output="screen",

            parameters=[
                {
                    "robot_description":
                    robot_description
                }
            ],
        ),

        # ----------------------------------------------------
        # Our animation node
        # ----------------------------------------------------

        ExecuteProcess(
            cmd=[
                "python3",
                ARM_ANIMATOR_PATH,
            ],

            output="screen",
        ),
    ])

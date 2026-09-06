"""
ROS2 inference node — subscribes to the fake sensor's point cloud stream, runs the ACTUAL
trained PointNet + MC Dropout model on each incoming scan, and publishes the resulting
prediction, confidence, and Grasp/Re-scan/Ask-help decision — both to the console (colored)
and as an RViz2 text marker so it's visible directly in the 3D view alongside the point cloud.

Uses T=10 MC Dropout passes (not 30) for responsiveness — justified by the T-ablation
(Section 8.1 of the report), which found T=10 performs comparably to T=30 at a fraction of
the compute cost, and a live demo benefits from feeling responsive.

Run (from repo root, with the ros2_demo_venv activated and ROS2 sourced):
    python3 src/ros2_demo/inference_node.py
"""

import os
import sys
import glob

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker
from std_msgs.msg import String

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data.modelnet40_dataset import ModelNet40Dataset
from src.models.pointnet import PointNetClassifier, enable_mc_dropout
from src.utils.config import OOD_CLASSES, NUM_POINTS
from src.inference.mc_dropout_inference import mc_dropout_predict, compute_uncertainty
from src.evaluation.decision_policy import classify_action

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "experiments")
GRASP_THRESH = 0.877
RESCAN_THRESH = 0.777
T_PASSES = 10  # justified by the MC-passes ablation (Section 8.1)

# ANSI colors for console output — purely cosmetic, makes the live demo easier to read
COLOR = {"Grasp": "\033[92m", "Re-scan": "\033[93m", "Ask for help": "\033[91m", "reset": "\033[0m"}


def find_latest_checkpoint():
    candidates = sorted(glob.glob(os.path.join(EXPERIMENTS_DIR, "dropout_ablation_p0.1_*")))
    return os.path.join(candidates[-1], "checkpoints", "best.pth")


class InferenceNode(Node):
    def __init__(self):
        super().__init__("inference_node")

        self.device = torch.device("cpu")
        # class name lookup only — this dataset object is not used for its data, just its
        # .classes list, so num_points/split barely matter here
        self.classes = ModelNet40Dataset(split="test", num_points=NUM_POINTS,
                                          excluded_classes=OOD_CLASSES).classes

        ckpt_path = find_latest_checkpoint()
        self.get_logger().info(f"Loading trained model: {ckpt_path}")
        self.model = PointNetClassifier(num_classes=len(self.classes)).to(self.device)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        enable_mc_dropout(self.model)  # keeps dropout stochastic at inference — the whole point

        self.subscription = self.create_subscription(
            PointCloud2, "object_scan", self.on_scan_received, 10)
        self.marker_pub = self.create_publisher(Marker, "decision_label", 10)
        self.decision_pub = self.create_publisher(String, "policy_decision", 10)

        self.get_logger().info(f"Inference node ready (T={T_PASSES} MC passes per scan). "
                                f"Waiting for scans on /object_scan...")

    @torch.no_grad()
    def classify(self, points: np.ndarray):
        x = torch.from_numpy(points).float().unsqueeze(0).to(self.device)
        probs_T = mc_dropout_predict(self.model, x, T_PASSES)
        stats = compute_uncertainty(probs_T)
        pred = int(stats["pred_class"][0].item())
        conf = float(stats["mean_probs"][0, pred].item())
        entropy = float(stats["total_entropy"][0].item())
        return pred, conf, entropy

    def publish_marker(self, text: str, action: str):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "decision"
        marker.id = 0
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 1.5  # float the label above the point cloud
        marker.scale.z = 0.3
        marker.color.a = 1.0
        if action == "Grasp":
            marker.color.g = 1.0
        elif action == "Re-scan":
            marker.color.r, marker.color.g = 1.0, 1.0
        else:
            marker.color.r = 1.0
        marker.text = text
        self.marker_pub.publish(marker)

    def on_scan_received(self, msg: PointCloud2):
        points_list = list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        points = np.array([[p[0], p[1], p[2]] for p in points_list], dtype=np.float32)

        # PointNet needs a fixed point count matching training (1024) — resample if the
        # incoming scan is a different size (e.g. the occluded example has fewer real points)
        if len(points) != NUM_POINTS:
            idx = np.random.choice(len(points), NUM_POINTS,
                                    replace=len(points) < NUM_POINTS)
            points = points[idx]

        pred, conf, entropy = self.classify(points)
        action = classify_action(conf, GRASP_THRESH, RESCAN_THRESH)
        pred_name = self.classes[pred]

        color = COLOR.get(action, "")
        reset = COLOR["reset"]
        self.get_logger().info(
            f"{color}Prediction: {pred_name:12s} | Confidence: {conf*100:5.1f}% | "
            f"Entropy: {entropy:.3f} | DECISION: {action.upper()}{reset}"
        )

        label_text = f"{pred_name} ({conf*100:.1f}%)\n{action.upper()}"
        self.publish_marker(label_text, action)

        decision_msg = String()
        decision_msg.data = action  # "Grasp" / "Re-scan" / "Ask for help" — the arm node reacts to this
        self.decision_pub.publish(decision_msg)


def main():
    rclpy.init()
    node = InferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

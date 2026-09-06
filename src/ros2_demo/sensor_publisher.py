"""
ROS2 "fake sensor" node — simulates a depth/LiDAR sensor by publishing real point clouds
from the ModelNet40 test/OOD data on a topic, cycling through three curated examples that
demonstrate the full range of the decision policy's behavior:

    1. A clean, unoccluded object the model is confident and correct about  -> should Grasp
    2. The same kind of object but deliberately occluded (60% of points removed)  -> degraded
    3. A genuine OOD object (a class type never seen in training)  -> unfamiliar

This node does NOT run the trained model itself for the live demo — it only uses the model
briefly at startup to pick a good "confident" example, so the demo doesn't rely on luck.
The actual live classification happens in inference_node.py, which is the point of splitting
these into two separate ROS2 nodes (mirrors a real sensor -> perception pipeline).

Run (from repo root, with the ros2_demo_venv activated and ROS2 sourced):
    python3 src/ros2_demo/fake_sensor_publisher.py
"""

import os
import sys
import glob

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header
from sensor_msgs_py import point_cloud2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.data.modelnet40_dataset import ModelNet40Dataset
from src.data.corruption import occlude
from src.models.pointnet import PointNetClassifier
from src.utils.config import OOD_CLASSES, NUM_POINTS

EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "experiments")
PUBLISH_PERIOD_SEC = 8.0  # how long each example stays "in view" before cycling to the next


def find_latest_checkpoint():
    candidates = sorted(glob.glob(os.path.join(EXPERIMENTS_DIR, "dropout_ablation_p0.1_*")))
    return os.path.join(candidates[-1], "checkpoints", "best.pth")


def select_confident_clean_example(model, test_ds, device, max_tries=30):
    """Scans a handful of test samples for one the model is already confident and correct
    about (single deterministic pass, no MC dropout needed just for picking a demo example),
    so the 'everything looks good' demo case isn't left to chance."""
    model.eval()
    for i in range(min(max_tries, len(test_ds))):
        points, label = test_ds[i]
        x = torch.from_numpy(points).float().unsqueeze(0).to(device)
        with torch.no_grad():
            logits, _, _ = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = int(probs.argmax(dim=1).item())
        conf = float(probs[0, pred].item())
        if pred == label and conf >= 0.877:
            return points, label, test_ds.classes[label], conf
    # fallback: just use the first sample if nothing hit the bar within max_tries
    points, label = test_ds[0]
    return points, label, test_ds.classes[label], None


def make_pointcloud2_msg(points: np.ndarray, frame_id="map"):
    header = Header()
    header.frame_id = frame_id
    return point_cloud2.create_cloud_xyz32(header, points.tolist())


class FakeSensorPublisher(Node):
    def __init__(self):
        super().__init__("fake_sensor_publisher")
        self.publisher_ = self.create_publisher(PointCloud2, "object_scan", 10)

        device = torch.device("cpu")
        test_ds = ModelNet40Dataset(split="test", num_points=NUM_POINTS, excluded_classes=OOD_CLASSES)
        ood_ds = ModelNet40Dataset(split="ood", num_points=NUM_POINTS, excluded_classes=OOD_CLASSES)

        ckpt_path = find_latest_checkpoint()
        self.get_logger().info(f"Loading checkpoint for example selection: {ckpt_path}")
        model = PointNetClassifier(num_classes=test_ds.num_classes).to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        # --- Build the three curated examples ---
        clean_points, clean_label, clean_name, clean_conf = select_confident_clean_example(
            model, test_ds, device)
        self.get_logger().info(f"Example 1 (clean): '{clean_name}', "
                                f"pre-check confidence={clean_conf}")

        occluded_points = occlude(clean_points.copy(), fraction=0.6)
        self.get_logger().info(f"Example 2 (occluded 60%): same object class '{clean_name}'")

        ood_points, ood_label = ood_ds[0]
        ood_name = ood_ds.original_class_name(ood_label)
        self.get_logger().info(f"Example 3 (OOD, never trained on): '{ood_name}'")

        self.examples = [
            ("Clean object", clean_name, clean_points),
            ("Occluded object (60%)", clean_name, occluded_points),
            ("OOD object (unseen class)", ood_name, ood_points),
        ]
        self.example_idx = 0

        self.timer = self.create_timer(PUBLISH_PERIOD_SEC, self.publish_next_example)
        self.get_logger().info(f"Fake sensor publisher ready. Cycling every {PUBLISH_PERIOD_SEC}s.")

    def publish_next_example(self):
        label_text, true_name, points = self.examples[self.example_idx]
        msg = make_pointcloud2_msg(points)
        self.publisher_.publish(msg)
        self.get_logger().info(f">>> Publishing [{label_text}] — true object: {true_name} "
                                f"({len(points)} points)")
        self.example_idx = (self.example_idx + 1) % len(self.examples)


def main():
    rclpy.init()
    node = FakeSensorPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

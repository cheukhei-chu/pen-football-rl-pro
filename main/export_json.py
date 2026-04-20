import argparse
import json
import os
from pathlib import Path

from policy import policy_from_checkpoint_path


DEFAULT_CHECKPOINT_PATH = "../checkpoints/league_ppo (misc rewards)/checkpoint_7100000.pth"
DEFAULT_OUTPUT_FILE = "model_weights.json"


def extract_browser_weights(policy):
    if not all(hasattr(policy, attr) for attr in ("plan_net", "action_net", "head_left", "head_right", "head_jump")):
        raise TypeError(
            f"Policy type {policy.__class__.__name__} is not compatible with the browser export format."
        )

    weights = {}

    def save_layer(prefix, layer):
        weights[f"{prefix}_w"] = layer.weight.detach().cpu().numpy().tolist()
        if layer.bias is not None:
            weights[f"{prefix}_b"] = layer.bias.detach().cpu().numpy().tolist()
        else:
            weights[f"{prefix}_b"] = [0.0] * layer.out_features

    save_layer("plan_0", policy.plan_net[0])
    save_layer("plan_2", policy.plan_net[2])
    save_layer("action_0", policy.action_net[0])
    save_layer("action_2", policy.action_net[2])
    save_layer("head_left", policy.head_left)
    save_layer("head_right", policy.head_right)
    save_layer("head_jump", policy.head_jump)
    return weights


def export_checkpoint_to_json(checkpoint_path: str, output_file: str) -> None:
    print(f"Loading {checkpoint_path}...")
    policy, _ = policy_from_checkpoint_path(checkpoint_path)
    policy.eval()

    weights = extract_browser_weights(policy)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(weights, f)

    print(f"Success! Saved weights to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export a browser-compatible JSON weight file.")
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to the .pth checkpoint to export.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_FILE,
        help="Path to the JSON file to write.",
    )
    args = parser.parse_args()

    checkpoint_path = os.path.join(os.path.dirname(__file__), args.checkpoint)
    export_checkpoint_to_json(checkpoint_path, args.output)


if __name__ == "__main__":
    main()

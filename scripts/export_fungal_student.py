"""Export the selected binary SegFormer Student without training-machine paths."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = torch.load(args.source, map_location="cpu", weights_only=True)
    config = source["config"]
    if config["architecture"] != "segformer_b2":
        raise ValueError("Expected the selected binary SegFormer-B2 Student.")
    state = source["model"]
    if not state or not all(key.startswith("network.") for key in state):
        raise ValueError("Expected the training wrapper's complete network state.")
    architecture_keys = (
        "num_channels", "num_encoder_blocks", "depths", "sr_ratios", "hidden_sizes",
        "patch_sizes", "strides", "mlp_ratios", "num_attention_heads", "hidden_act",
        "hidden_dropout_prob", "attention_probs_dropout_prob", "classifier_dropout_prob",
        "initializer_range", "drop_path_rate", "layer_norm_eps", "decoder_hidden_size",
        "reshape_last_stage", "semantic_loss_ignore_index", "id2label", "label2id",
    )
    model_config = {key: config["segformer_config"][key] for key in architecture_keys}
    model_config.update(model_type="segformer", num_labels=1)
    portable = {
        "format_version": 1,
        "architecture": "segformer_b2_teacher_student",
        "model": {key.removeprefix("network."): tensor for key, tensor in state.items()},
        "model_config": model_config,
        "input_size": config["input_size"],
        "normalization": config["normalization"],
        "threshold": float(source["threshold"]),
        "threshold_selection": "validation_mean_dice",
        "source_checkpoint_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "training_epoch": source["epoch"],
        "student_round": config["round"],
        "reference_valid_metrics": source.get("valid_metrics_calibrated", {}),
        "reference_test_metrics": source.get("test_metrics", {}),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(portable, args.output)
    print(f"Exported {args.output.name}: threshold={portable['threshold']}, tensors={len(state)}")


if __name__ == "__main__":
    main()

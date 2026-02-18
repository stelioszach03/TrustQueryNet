from trustquerynet.eval.phase_gate import verify_config_gate


def test_verify_config_gate_accepts_matched_repair_baselines():
    repair_cfg = {
        "training": {
            "backbone": "convnext_tiny",
            "epochs": 20,
            "batch_size": 32,
            "lr": 1e-4,
            "loss": "cross_entropy",
            "sampler": "shuffle",
            "amp": True,
            "warmup_epochs": 2,
            "early_stopping_patience": 4,
        },
        "noise": {"type": "transition_matrix", "matrix": [[1.0]]},
        "evaluation": {"checkpoint_policy": "best_val_macro_f1", "thresholds": None, "num_thresholds": 101},
        "active_learning": {"method": "entropy", "query_size": 64, "rounds": 2, "initial_clean_fraction": 0.1},
    }
    no_repair_cfg = {
        **repair_cfg,
        "active_learning": {"method": "entropy", "query_size": 0, "rounds": 1, "initial_clean_fraction": 0.1},
    }
    random_cfg = {
        **repair_cfg,
        "active_learning": {"method": "random", "query_size": 64, "rounds": 2, "initial_clean_fraction": 0.1},
    }

    report = verify_config_gate(repair_cfg, no_repair_cfg, random_cfg)

    assert report["no_repair_deconfounded"] is True
    assert report["random_repair_matches_budget"] is True

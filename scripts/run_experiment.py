from utils.scrub_integration import scrub_unlearning

def run_unlearning(model, retain_set, forget_set, unlearning_cfg):
    method = unlearning_cfg["method"]

    if method == "fine_tune":
        return fine_tune_unlearning(model, retain_set, unlearning_cfg)

    elif method == "grad_ascent":
        return gradient_ascent_unlearning(model, forget_set, unlearning_cfg)

    elif method == "scrub":
        return scrub_unlearning(model, retain_set, forget_set, unlearning_cfg)

    elif method == "ssd":
        return ssd_unlearning(model, retain_set, forget_set, unlearning_cfg)

    else:
        raise ValueError(f"Unknown unlearning method {method}")


def run_mia(mia_cfg, model, retain_set, forget_set, test_set, global_cfg):

    attack_name = mia_cfg["name"]
    params = mia_cfg["params"]

    model_access = make_model_access(
        attack_name,
        model
    )

    aux_info = make_aux_info(
        attack_name,
        params,
        global_cfg
    )

    attack = make_attack(
        attack_name,
        model_access,
        aux_info
    )

    # Auxiliary dataset = retain ∪ forget (or separate depending on MIA)
    auxiliary_dataset = ConcatDataset([retain_set, forget_set])

    attack.prepare(auxiliary_dataset)

    # Inference dataset = forget ∪ test (labelled)
    inference_set = ConcatDataset([forget_set, test_set])

    scores = attack.infer(inference_set)

    save_scores(scores, attack_name, global_cfg)

def main(config_path):
    cfg = load_yaml(config_path)
    set_seed(cfg["experiment"]["seed"])

    # 1. Load data
    retain_set, forget_set, test_set = load_and_split_dataset(cfg)

    # 2. Load model
    model = load_model(cfg["model"])
    model.load_state_dict(torch.load(cfg["model"]["checkpoint_path"]))

    # 3. Unlearning
    unlearned_model = run_unlearning(
        model,
        retain_set,
        forget_set,
        cfg["unlearning"]
    )

    save_model(unlearned_model, cfg)

    # 4. Run MIAs
    for mia_cfg in cfg["mia"]["attacks"]:
        run_mia(
            mia_cfg,
            unlearned_model,
            retain_set,
            forget_set,
            test_set,
            cfg
        )

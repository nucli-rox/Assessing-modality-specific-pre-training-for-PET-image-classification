def build_model(model_cfg):
    name = model_cfg.get("name")

    if name == "MAE":
        from src.models.MAE import build_mae_model

        return build_mae_model(model_cfg)
    if name == "MAE-IN":
        from src.models.MAE_IN import build_mae_in_model

        return build_mae_in_model(model_cfg)
    if name == "classifier_MAE":
        from tasks.models.convnext_MAE_clf import build_classifier_mae

        return build_classifier_mae(model_cfg)

    raise ValueError(f"Unsupported model name: {name}")

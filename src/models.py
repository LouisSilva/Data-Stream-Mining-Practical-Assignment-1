"""Model construction"""
from river import ensemble, forest, linear_model, naive_bayes, optim, preprocessing, tree

_MODELS = {
    "gaussian_naive_bayes": naive_bayes.GaussianNB,
    "online_logistic_regression": linear_model.LogisticRegression,
    "hoeffding_tree": tree.HoeffdingTreeClassifier,
    "online_bagging": ensemble.BaggingClassifier,
    "adaptive_random_forest": forest.ARFClassifier,
}

def build_model(selected_model: str, models_config: dict):
    params = models_config.get(selected_model, {})

    if selected_model == "online_logistic_regression":
        optimizer_config = params["optimizer"]
        optimizer_name = optimizer_config["name"]
        
        optimizers = {"SGD": optim.SGD, "Adam": optim.Adam}
        if optimizer_name not in optimizers:
            raise ValueError(f"Unsupported optimizer {optimizer_name!r}")
            
        params["optimizer"] = optimizers[optimizer_name](**(optimizer_config.get("params") or {}))
        model = _MODELS[selected_model](**params)
        
        return preprocessing.StandardScaler() | model if settings.get("standardize", True) else model

    if selected_model == "online_bagging":
        params["model"] = tree.HoeffdingTreeClassifier(**models_config.get("hoeffding_tree", {}))

    return _MODELS[selected_model](**params)
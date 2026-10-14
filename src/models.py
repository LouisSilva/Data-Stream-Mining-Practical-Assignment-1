"""Build fresh River classifiers from the ``models`` section of config.yaml.

The model parameters are passed to River constructors with **params.
The Logistic Regression optimizer and Bagging's base tree must first be
constructed as Python objects from their respective YAML settings.
"""
from river import ensemble, forest, linear_model, naive_bayes, optim, preprocessing, tree

_MODELS = {
    "gaussian_naive_bayes": naive_bayes.GaussianNB,
    "online_logistic_regression": linear_model.LogisticRegression,
    "hoeffding_tree": tree.HoeffdingTreeClassifier,
    "online_bagging": ensemble.BaggingClassifier,
    "adaptive_random_forest": forest.ARFClassifier,
}

MODEL_NAMES = tuple(_MODELS)

def build_model(models_config: dict, selected_model: str):
    """Construct a new, untrained model with YAML-defined parameters."""
    if selected_model not in _MODELS:
        raise ValueError(f"Unknown model {selected_model!r}; expected one of {MODEL_NAMES}")

    if selected_model not in models_config:
        raise KeyError(f"Missing config['models'][{selected_model!r}]")

    settings = models_config[selected_model]
    params = dict(settings.get("params", {}))

    if selected_model == "online_logistic_regression":
        optimizer_config = settings["optimizer"]
        optimizer = getattr(optim, optimizer_config["name"])(**optimizer_config.get("params", {}))
        
        params["optimizer"] = optimizer
        model = _MODELS[selected_model](**params)
        
        return  (
            preprocessing.StandardScaler() | model 
            if settings.get("standardize", False) 
            else model
        )

    if selected_model == "online_bagging":
        tree_params = models_config["hoeffding_tree"]["params"]
        params["model"] = tree.HoeffdingTreeClassifier(**tree_params)

    return _MODELS[selected_model](**params)
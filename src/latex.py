import pandas as pd


def print_latex_results_tables(summary: pd.DataFrame) -> str:
    """
    Print complete LaTeX result tables for Representations A and B.

    Expected columns in `summary`:
        city
        representation
        model
        accuracy
        macro_f1
        precision
        recall
        prediction_ms_per_event
        update_ms_per_event

    Returns the complete LaTeX code for both tables.
    """

    # Allows this to also work if summary is a list of dictionaries
    summary = pd.DataFrame(summary).copy()

    required_columns = {
        "city",
        "representation",
        "model",
        "accuracy",
        "macro_f1",
        "precision",
        "recall",
        "prediction_ms_per_event",
        "update_ms_per_event",
    }

    missing = required_columns - set(summary.columns)

    if missing:
        raise ValueError(f"`summary` is missing required columns: {sorted(missing)}")

    # Internal model name -> display name in LaTeX
    model_order = [
        ("gaussian_naive_bayes", "GaussianNB"),
        ("online_logistic_regression", "Online Logistic"),
        ("hoeffding_tree", "Hoeffding Tree"),
        ("online_bagging", "Online Bagging"),
        ("adaptive_random_forest", r"Adaptive Random \allowbreak Forest"),
    ]

    cities = ["Lisboa", "Porto"]

    # --------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------
    for representation in ["A", "B"]:
        for city in cities:
            for model_key, _ in model_order:

                matches = summary[
                    (summary["representation"].astype(str).str.upper()
                     == representation)
                    & (summary["city"] == city)
                    & (summary["model"] == model_key)
                ]

                if len(matches) != 1:
                    raise ValueError(
                        "Expected exactly one result for "
                        f"{city}, Representation {representation}, "
                        f"{model_key}; found {len(matches)}."
                    )

    # --------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------
    def get_result(
        representation: str,
        city: str,
        model: str,
    ) -> pd.Series:

        return summary[
            (summary["representation"].astype(str).str.upper()
             == representation)
            & (summary["city"] == city)
            & (summary["model"] == model)
        ].iloc[0]

    def decimal_to_pct(value: float) -> str:
        """0.993 -> '99.3'"""
        return f"{100 * float(value):.1f}"

    def to_3_dp(value: float) -> str:
        """Format milliseconds/event."""
        return f"{float(value):.3f}"

    def splitcell(top: str, bottom: str) -> str:
        return rf"\splitcell{{{top}}}{{{bottom}}}"

    # --------------------------------------------------------------
    # Create one table
    # --------------------------------------------------------------
    def create_table(representation: str) -> str:
        rows = []
        
        for model_key, display_name in model_order:
            lisboa = get_result(representation, "Lisboa", model_key)
            porto = get_result(representation, "Porto", model_key)

            row = (
                f"{display_name}"
                f" & {splitcell(decimal_to_pct(lisboa['accuracy']), decimal_to_pct(porto['accuracy']))}"
                f" & {splitcell(decimal_to_pct(lisboa['macro_f1']), decimal_to_pct(porto['macro_f1']))}"
                f" & {splitcell(decimal_to_pct(lisboa['precision']), decimal_to_pct(porto['precision']))}"
                f" & {splitcell(decimal_to_pct(lisboa['recall']), decimal_to_pct(porto['recall']))}"
                f" & {splitcell(to_3_dp(lisboa['prediction_ms_per_event']), to_3_dp(porto['prediction_ms_per_event']))}"
                f" & {splitcell(to_3_dp(lisboa['update_ms_per_event']), to_3_dp(porto['update_ms_per_event']))}"
                r" \\[4pt]"
            )

            rows.append(row)

        # Last row doesn't need extra vertical spacing
        rows[-1] = rows[-1].replace(r" \\[4pt]", r" \\")
        table_rows = "\n\t\t\t".join(rows)

        return rf"""
\begin{{table}}[htbp]
    \centering
    \caption
    {{
        Predictive and computational performance for Representation {representation}.
        Predictive metrics are reported as percentages.
        Each cell reports
        \colorbox{{LisboaGreen!70}}{{\scriptsize Lisboa}}
        in the top-left triangle and
        \colorbox{{PortoRed!70}}{{\scriptsize Porto}}
        in the bottom-right triangle.
    }}
    \label{{tab:representation-{representation.lower()}-results}}

    \resizebox{{\textwidth}}{{!}}{{%
        \begin{{tabularx}}{{0.98\textwidth}}{{@{{}}Y C{{1.55cm}} C{{1.55cm}} C{{1.55cm}} C{{1.55cm}} C{{1.55cm}} C{{1.55cm}}@{{}}}}
            \toprule
            \textbf{{Model}}
            & \textbf{{Acc.}}
            & \textbf{{Macro-F1}}
            & \textbf{{Prec.}}
            & \textbf{{Recall}}
            & \textbf{{Pred. ms/event}}
            & \textbf{{Update ms/event}} \\
            \midrule
            {table_rows}
            \bottomrule
        \end{{tabularx}}
    }}
\end{{table}}
""".strip()

    # --------------------------------------------------------------
    # Generate A and B
    # --------------------------------------------------------------
    latex_a = create_table("A")
    latex_b = create_table("B")

    latex = latex_a + "\n\n" + latex_b

    print(latex)
    return latex
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from .config import find_project_root
from .data import create_random_suffix
from .results import ExperimentResults


def set_plot_style(plotting_config: dict, aspect_ratio=0.618, dpi=200) -> None:
    inches_per_pt = 1 / 72.27
    font_size = plotting_config["overleaf_font_size"]
    fig_width_in = plotting_config["overleaf_text_width_pt"] * inches_per_pt
    fig_height_in = fig_width_in * aspect_ratio

    plt.rcdefaults()
    plt.rcParams.update({
        "text.usetex": True,
        "figure.figsize": (fig_width_in, fig_height_in),
        "figure.dpi": dpi,
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 2,
        "ytick.labelsize": font_size - 2,
        "legend.fontsize": font_size - 2,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "pdf.fonttype": 42,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
    })


def save_figure(fig, prefix: str, ext="pdf") -> None:
    project_root = find_project_root()
    temp_folder = project_root / "temp"
    
    temp_folder.mkdir(parents=True, exist_ok=True)
    filepath = temp_folder / f"{prefix}_{create_random_suffix()}.{ext}"
    
    fig.savefig(filepath, bbox_inches='tight', pad_inches=0.05)
    print(f"Saved {filepath}")
    plt.close(fig)


def plot_R_m(plotting_config: dict, df: pd.DataFrame, city: str) -> None:
    set_plot_style(plotting_config, aspect_ratio=0.4, dpi=400)

    if city == "Lisboa": colour = plotting_config["colours"]["professor_defined_green"]
    elif city == "Porto": colour = plotting_config["colours"]["complementary_professor_red"]
    else: raise ValueError(f"Undefined city '{city}'")

    # Tumbling window - monthly rain-event evolution: what fraction of all hours in this month were rain events?
    monthly_rain = (
        df.set_index("time")["rain_event"]
        .resample("MS")
        .mean()
        .mul(100)
    )

    fig, ax = plt.subplots(1, 1)
    # ax.plot(monthly_rain.index, monthly_rain["event_rate_pct"], linewidth=1.2, color=colour)
    ax.plot(monthly_rain.index, monthly_rain.values, linewidth=1.2, color=colour)
    
    # Labels
    ax.set_title(f"{city} Monthly Rain-Event Rate")
    ax.set_ylabel("Rain-event rate (\\%)")
    ax.set_xlabel("Month")
    
    # Ticks and margins
    ax.set_ylim(-2, 52)
    ax.set_yticks(range(0, 51, 10))
    ax.set_xlim(monthly_rain.index.min(), monthly_rain.index.max())
    ax.margins(x=0.01)
    
    # One labelled tick per year
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    
    # Minor ticks every 3 months
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
    
    # Creates horizontal guides
    ax.grid(axis="y", which="major", alpha=0.25)
    
    # Remove borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    save_figure(fig, f"{city}_R_m")


def plot_C_24(plotting_config: dict, df: pd.DataFrame, city: str) -> None:
    set_plot_style(plotting_config, aspect_ratio=0.4, dpi=400)

    if city == "Lisboa": colour = plotting_config["colours"]["professor_defined_green"]
    elif city == "Porto": colour = plotting_config["colours"]["complementary_professor_red"]
    else: raise ValueError(f"Undefined city '{city}'")

    # Sliding window - rainy hours in the past 24 hours 
    df["rainy_hours_previous_24h"] = (
        df["rain_event"]
          .shift(1)
          .rolling(window=24, min_periods=24)
          .sum()
    )
    
    counts = (
        df["rainy_hours_previous_24h"]
          .dropna()
          .astype(int)
          .value_counts()
          .sort_index()
    )
    
    fig, ax = plt.subplots(1, 1)
    ax.bar(counts.index, counts.values, width=0.8, facecolor=colour)
        
    # Labels
    ax.set_title(f"{city} Distribution of Rain-Event in the Preceding 24 Hours")
    ax.set_xlabel("Rain-event hours in preceding 24 hours")
    ax.set_ylabel("Number of observations")

    # Horizontal grid
    ax.grid(axis="y", which="major", alpha=0.25)
    ax.set_axisbelow(True)
    
    # Remove borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Ticks and scales
    ax.set_xticks(range(0, 25, 1))
    ax.set_xlim(-0.6, 24.6)
    ax.set_yscale("log")
    
    plt.tight_layout()
    plt.show()
    save_figure(fig, f"{city}_C_24")


def plot_city_comparison(
    plotting_config: dict,
    lisboa_df: pd.DataFrame,
    porto_df: pd.DataFrame
) -> None:
    set_plot_style(plotting_config, aspect_ratio=0.8, dpi=400)

    # -------------------------------------------------------------------------
    # Panel A: Monthly rain-event rate R_m
    # -------------------------------------------------------------------------
    lisboa_monthly_rain = (
        lisboa_df.set_index("time")["rain_event"]
        .resample("MS")
        .mean()
        .mul(100)
    )

    porto_monthly_rain = (
        porto_df.set_index("time")["rain_event"]
        .resample("MS")
        .mean()
        .mul(100)
    )

    # -------------------------------------------------------------------------
    # Panel B: Distribution of C_t^(24)
    # -------------------------------------------------------------------------
    lisboa_c24_normalized = (
        lisboa_df["rain_event"]
        .shift(1)
        .rolling(window=24, min_periods=24)
        .sum()
        .dropna()
        .astype(int)
        .value_counts(normalize=True)
        .reindex(range(25), fill_value=0)
    )

    porto_c24_normalized = (
        porto_df["rain_event"]
        .shift(1)
        .rolling(window=24, min_periods=24)
        .sum()
        .dropna()
        .astype(int)
        .value_counts(normalize=True)
        .reindex(range(25), fill_value=0)
    )

    # Colours
    colours = plotting_config["colours"]
    lisboa_colour = colours["professor_defined_green"]
    porto_colour = colours["complementary_professor_red"]

    # -------------------------------------------------------------------------
    # Create figure
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # =====================================================================
    # Panel A
    # =====================================================================
    ax1.plot(
        lisboa_monthly_rain.index,
        lisboa_monthly_rain.values,
        linewidth=1.2,
        color=lisboa_colour,
        label="Lisboa"
    )

    ax1.plot(
        porto_monthly_rain.index,
        porto_monthly_rain.values,
        linewidth=1.2,
        color=porto_colour,
        label="Porto"
    )

    ax1.set_title("(a) Monthly Rain-Event Rate")
    ax1.set_ylabel("Rain-event rate (\\%)")
    ax1.set_xlabel("Month")

    ax1.set_ylim(-2, 52)
    ax1.set_yticks(range(0, 51, 10))

    # Use the full common plotting period
    min_date = min(
        lisboa_monthly_rain.index.min(),
        porto_monthly_rain.index.min()
    )

    max_date = max(
        lisboa_monthly_rain.index.max(),
        porto_monthly_rain.index.max()
    )

    ax1.set_xlim(min_date, max_date)
    ax1.margins(x=0.01)

    # One labelled tick per year
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Minor ticks every 3 months
    ax1.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))

    ax1.grid(
        axis="y",
        which="major",
        alpha=0.25
    )

    ax1.set_axisbelow(True)

    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax1.legend(frameon=False)

    # =====================================================================
    # Panel B
    # =====================================================================
    x = np.arange(25)
    width = 0.38

    ax2.bar(
        x - width / 2,
        lisboa_c24_normalized.values,
        width=width,
        color=lisboa_colour,
        label="Lisboa"
    )

    ax2.bar(
        x + width / 2,
        porto_c24_normalized.values,
        width=width,
        color=porto_colour,
        label="Porto"
    )

    ax2.set_title("(b) Distribution of Rain-Event Hours in the Preceding 24 Hours")

    ax2.set_xlabel("Rain-event hours in preceding 24 hours")
    ax2.set_ylabel("Proportion of observations")

    ax2.set_xticks(range(25))
    ax2.set_xlim(-0.7, 24.7)

    # Same log scale as your existing C24 plot
    ax2.set_yscale("log")

    ax2.grid(
        axis="y",
        which="major",
        alpha=0.25
    )

    ax2.set_axisbelow(True)

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    ax2.legend(frameon=False)

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------
    plt.tight_layout()
    plt.show()
    save_figure(fig, "city_comparison")


def plot_model_performance_over_time(
    plotting_config: dict,
    results: ExperimentResults,
    representation: str = 'A',
) -> None:
    """
    Compare rolling Macro-F1 for Online Bagging vs HT in Lisboa and Porto.

    Each panel shows one city with the same two models. All series use the
    same window and both panels share a common y-axis. 
    """
    evaluation_config = results.config.get('evaluation', {})
    window_size = int(evaluation_config.get('window_size', 720))
    if window_size < 1:
        raise ValueError('evaluation.window_size must be >= 1')
    if representation not in ('A', 'B'):
        raise ValueError('representation must be A or B')

    history = results.history
    if history.empty:
        raise ValueError('Metric history is empty; run the experiments first')

    models = {
        'online_bagging': ('Online Bagging', plotting_config["colours"]["professor_defined_green"], '-', 1.2),
        'hoeffding_tree': ('Hoeffding Tree', plotting_config["colours"]["dark_professor_green"], '-', 0.85),
    }
    
    set_plot_style(plotting_config, aspect_ratio=0.95, dpi=400)
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)

    for ax, city, panel in zip(axes, ('Lisboa', 'Porto'), ('a', 'b')):
        for model, (label, colour, linestyle, linewidth) in models.items():
            records = history.loc[
                (history['city'] == city)
                & (history['representation'] == representation)
                & (history['model'] == model)
                & (history['n_evaluated'] >= window_size)
            ].copy()
            
            if records.empty:
                plt.close(fig)
                raise ValueError(
                    f'{city}/{representation}/{model}: no complete '
                    f'{window_size}-observation history windows'
                )
                
            records['timestamp'] = pd.to_datetime(records['timestamp'], errors='raise')
            records = records.sort_values('timestamp').reset_index(drop=True)
            if records['timestamp'].isna().any() or records['timestamp'].duplicated().any():
                plt.close(fig)
                raise ValueError(f'{city}/{model}: invalid or duplicated timestamps')

            ax.plot(
                records['timestamp'], records['window_macro_f1'] * 100,
                label=label, color=colour, linestyle=linestyle, linewidth=linewidth,
            )

        ax.set_title(f'({panel}) {city}')
        ax.set_ylabel(r'Rolling Macro-F1 (\%)')
        ax.grid(axis='y', alpha=0.25)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=2, frameon=False)
    
    axes[1].set_xlabel('Evaluation time')
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[1].xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
    
    fig.suptitle(f'Rolling Macro-F1 over time - Representation {representation}', y=0.988)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    
    plt.show()
    save_figure(fig, 'model_performance_over_time')
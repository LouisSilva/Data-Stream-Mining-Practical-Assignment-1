import os
import string
import secrets
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from .config import find_project_root

def create_random_suffix(length=8) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def set_plot_style(config: dict, aspect_ratio=0.618, dpi=200) -> None:
    inches_per_pt = 1 / 72.27
    font_size = config["plotting"]["overleaf_font_size"]
    fig_width_in = config["plotting"]["overleaf_text_width_pt"] * inches_per_pt
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


def plot_R_m(config: dict, df: pd.DataFrame, city: str) -> None:
    set_plot_style(config, aspect_ratio=0.4, dpi=400)

    # Tumbling window - monthly rain-event evolution: what fraction of all hours in this month were rain events?
    monthly_rain = (
        df.set_index("time")["rain_event"]
        .resample("MS")
        .agg(["mean"])
    )

    monthly_rain["event_rate_pct"] = 100 * monthly_rain["mean"]

    fig, ax = plt.subplots(1, 1)
    ax.plot(monthly_rain.index, monthly_rain["event_rate_pct"], linewidth=1.2, color="#21A58B")
    
    # Labels
    ax.set_title(f"{city} - Monthly Rain-Event Rate")
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
    save_figure(fig, f"{city}_R_m")

def plot_C_24(config: dict, df: pd.DataFrame, city: str) -> None:
    set_plot_style(config, aspect_ratio=0.4, dpi=400)

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
    ax.bar(counts.index, counts.values, width=0.8, facecolor="#21A58B")
    
    # Labels
    ax.set_title("Porto - distribution of rainy hours in the previous 24 hours")
    ax.set_xlabel("Rain-event hours in preceding 24 hours")
    ax.set_ylabel("Number of observations")
    
    # zero_pct = 100 * counts.loc[0] / counts.sum()
    
    # ax.annotate(
    #     rf"{zero_pct:.1f}\%",
    #     xy=(0, counts.loc[0]),
    #     xytext=(0, 5),
    #     textcoords="offset points",
    #     ha="center",
    #     va="bottom",
    # )
    
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
    save_figure(fig, f"{city}_C_24")
"""One explicit, sequential weight palette for paired MDC adjacency figures."""

from matplotlib.colors import LinearSegmentedColormap

PALETTE = {
    "name": "yellow_orange_red",
    "low": "#FFFF00",
    "middle": "#FFA500",
    "high": "#FF0000",
    "missing": "#FFFFFF",
    "description": "Low edge weights: yellow; high edge weights: red",
}


def heatmap_cmap():
    cmap = LinearSegmentedColormap.from_list(
        PALETTE["name"], [PALETTE["low"], PALETTE["middle"], PALETTE["high"]]
    )
    cmap.set_bad(PALETTE["missing"])
    return cmap

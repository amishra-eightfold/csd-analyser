"""Configuration settings for visualizations."""

# Color palettes
BLUES_PALETTE = ["#E3F2FD", "#90CAF9", "#42A5F5", "#1E88E5", "#1565C0", "#0D47A1"]  # Material Blues
AQUA_PALETTE = ["#E0F7FA", "#80DEEA", "#26C6DA", "#00ACC1", "#00838F", "#006064"]   # Material Cyan/Aqua
PURPLE_PALETTE = ["#F3E5F5", "#CE93D8", "#AB47BC", "#8E24AA", "#6A1B9A", "#4A148C"]  # Material Purple

# Define custom color palettes for each visualization type
VOLUME_PALETTE = [BLUES_PALETTE[2], AQUA_PALETTE[2]]  # Two distinct colors for Created/Closed
PRIORITY_PALETTE = BLUES_PALETTE[1:]  # Blues for priority levels
CSAT_PALETTE = AQUA_PALETTE  # Aqua palette for CSAT
HEATMAP_PALETTE = "YlGnBu"  # Yellow-Green-Blue for heatmaps

# Create an extended palette for root causes by combining multiple color palettes
ROOT_CAUSE_PALETTE = (
    BLUES_PALETTE[1:] +     # 5 blues
    AQUA_PALETTE[1:] +      # 5 aquas
    PURPLE_PALETTE[1:]      # 5 purples
)  # Total of 15 distinct colors

# Plot styles
PLOT_STYLE = "seaborn-v0_8-whitegrid"
FIGURE_DPI = 100
DEFAULT_FIGURE_SIZE = (10, 6)
WIDE_FIGURE_SIZE = (12, 6)
SQUARE_FIGURE_SIZE = (8, 8)

# Font settings
FONT_FAMILY = "Arial"
TITLE_FONT_SIZE = 14
LABEL_FONT_SIZE = 12
TICK_FONT_SIZE = 10
LEGEND_FONT_SIZE = 10

# Grid settings
GRID_STYLE = {
    'alpha': 0.3,
    'linestyle': '--',
    'color': '#cccccc'
}

# Legend settings
LEGEND_STYLE = {
    'frameon': True,
    'fancybox': True,
    'shadow': True,
    'framealpha': 0.8,
    'edgecolor': '#cccccc'
}

# Axis settings
AXIS_STYLE = {
    'spines.top': False,
    'spines.right': False,
    'grid.alpha': 0.3
}

# Time series plot settings
TIME_SERIES_SETTINGS = {
    'marker': 'o',
    'markersize': 4,
    'linewidth': 2,
    'alpha': 0.8
}

# Bar plot settings
BAR_PLOT_SETTINGS = {
    'alpha': 0.8,
    'edgecolor': 'none'
}

# Box plot settings
BOX_PLOT_SETTINGS = {
    'boxprops': {'alpha': 0.8},
    'medianprops': {'color': 'red'},
    'flierprops': {'marker': 'o', 'markerfacecolor': 'gray'},
    'meanprops': {'marker': 'D', 'markerfacecolor': 'white'}
}

# Heatmap settings
HEATMAP_SETTINGS = {
    'cmap': HEATMAP_PALETTE,
    'center': 0,
    'annot': True,
    'fmt': '.2f',
    'square': True,
    'cbar_kws': {'shrink': .8}
}

# Distribution plot settings
DISTRIBUTION_SETTINGS = {
    'kde': {
        'bw_adjust': 0.5,
        'fill': True,
        'alpha': 0.3
    },
    'histogram': {
        'bins': 30,
        'alpha': 0.7,
        'edgecolor': 'white'
    }
}

# Export settings
EXPORT_SETTINGS = {
    'dpi': 300,
    'bbox_inches': 'tight',
    'pad_inches': 0.1,
    'transparent': False
}

# Animation settings
ANIMATION_SETTINGS = {
    'interval': 200,  # milliseconds
    'repeat_delay': 1000,  # milliseconds
    'blit': True
}

# Theme settings
LIGHT_THEME = {
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#333333',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'text.color': '#333333',
    'grid.color': '#cccccc'
}

DARK_THEME = {
    'figure.facecolor': '#1a1a1a',
    'axes.facecolor': '#1a1a1a',
    'axes.edgecolor': '#ffffff',
    'axes.labelcolor': '#ffffff',
    'xtick.color': '#ffffff',
    'ytick.color': '#ffffff',
    'text.color': '#ffffff',
    'grid.color': '#333333'
} 
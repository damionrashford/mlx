#!/usr/bin/env python3
"""Number formatting utilities for chart labels and axes.

Import this module in visualization scripts:
    from format_number import format_number, apply_currency_axis
"""

import matplotlib.ticker as mticker


def format_number(val, fmt="number"):
    """Format numbers for chart labels.

    Args:
        val: Numeric value to format.
        fmt: 'currency', 'percent', or 'number'.
    """
    if fmt == "currency":
        if abs(val) >= 1e9:
            return f"${val/1e9:.1f}B"
        if abs(val) >= 1e6:
            return f"${val/1e6:.1f}M"
        if abs(val) >= 1e3:
            return f"${val/1e3:.1f}K"
        return f"${val:,.0f}"
    elif fmt == "percent":
        return f"{val:.1f}%"
    else:
        if abs(val) >= 1e9:
            return f"{val/1e9:.1f}B"
        if abs(val) >= 1e6:
            return f"{val/1e6:.1f}M"
        if abs(val) >= 1e3:
            return f"{val/1e3:.1f}K"
        return f"{val:,.0f}"


def apply_currency_axis(ax, axis="y"):
    """Apply currency formatting to a matplotlib axis."""
    formatter = mticker.FuncFormatter(lambda x, p: format_number(x, "currency"))
    if axis == "y":
        ax.yaxis.set_major_formatter(formatter)
    else:
        ax.xaxis.set_major_formatter(formatter)


def apply_percent_axis(ax, axis="y"):
    """Apply percent formatting to a matplotlib axis."""
    formatter = mticker.FuncFormatter(lambda x, p: format_number(x, "percent"))
    if axis == "y":
        ax.yaxis.set_major_formatter(formatter)
    else:
        ax.xaxis.set_major_formatter(formatter)

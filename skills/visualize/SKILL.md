---
name: visualize
description: >
  Create charts, plots, dashboards, and data visualizations using matplotlib,
  seaborn, plotly, and altair. Export to PNG, HTML, or interactive dashboards.
  Use when the user asks to "make a chart", "create a dashboard", "plot the data",
  "visualize results", "build a report with charts", or mentions bar charts,
  line charts, heatmaps, scatter plots, histograms, or data storytelling.
allowed-tools: Bash, Read, Write, Glob, Grep
user-invocable: true
argument-hint: path to CSV/DataFrame or description of chart (e.g. "data/sales.csv" or "bar chart of revenue by quarter")
---

# Data Visualization

Templates and reference for creating publication-quality charts, interactive dashboards, and data reports.

## Scripts

| Script | Usage |
|--------|-------|
| [chart_templates.py](scripts/chart_templates.py) | `python3 ${CLAUDE_SKILL_DIR}/scripts/chart_templates.py data.csv --type bar --x category --y value -o chart.png` |
| [format_number.py](scripts/format_number.py) | Import for number formatting: `from format_number import format_number, apply_currency_axis` |

### Quick chart generation

```bash
# Bar chart
python3 ${CLAUDE_SKILL_DIR}/scripts/chart_templates.py data.csv --type bar --x category --y value -o bar.png

# Line chart (time series)
python3 ${CLAUDE_SKILL_DIR}/scripts/chart_templates.py data.csv --type line --x date --y value --hue segment -o trend.png

# Histogram
python3 ${CLAUDE_SKILL_DIR}/scripts/chart_templates.py data.csv --type hist --x value -o dist.png

# Correlation heatmap
python3 ${CLAUDE_SKILL_DIR}/scripts/chart_templates.py data.csv --type heatmap -o correlations.png

# Scatter with regression
python3 ${CLAUDE_SKILL_DIR}/scripts/chart_templates.py data.csv --type scatter --x feature_a --y target -o scatter.png

# Box plot
python3 ${CLAUDE_SKILL_DIR}/scripts/chart_templates.py data.csv --type box --x group --y value -o box.png
```

## Framework selection

| Framework | Best for | Output | Install |
|-----------|----------|--------|---------|
| matplotlib | Static charts, publications, fine control | PNG, PDF, SVG | `pip install matplotlib` |
| seaborn | Statistical plots, quick EDA visuals | PNG, PDF, SVG | `pip install seaborn` |
| plotly | Interactive charts, dashboards, web | HTML, JSON | `pip install plotly` |
| altair | Declarative, concise, notebooks | HTML, JSON | `pip install altair` |

**Default**: matplotlib + seaborn (no extra dependencies in most ML environments).
**Interactive**: plotly (self-contained HTML files, no server needed).

## Chart selection guide

| Question | Chart type |
|----------|-----------|
| How does X change over time? | Line chart |
| How do categories compare? | Bar chart (horizontal if many categories) |
| What is the distribution? | Histogram, box plot, violin plot |
| How do two variables relate? | Scatter plot |
| What are the correlations? | Heatmap |
| What is the composition? | Stacked bar, pie chart (use sparingly) |
| How do groups differ? | Grouped bar, box plot by group |
| What are the top/bottom N? | Horizontal bar, sorted |
| Geographic data? | Choropleth map (plotly) |
| Multi-dimensional? | Pair plot, parallel coordinates |

## When NOT to use certain charts

- **Pie charts**: Avoid unless <6 categories. Humans are bad at comparing angles. Use bar charts instead.
- **3D charts**: Never. They distort perception and add no information.
- **Dual-axis charts**: Use cautiously. They can mislead by implying correlation. Clearly label both axes if used.
- **Stacked bar (many categories)**: Hard to compare middle segments. Use small multiples or grouped bars.
- **Donut charts**: Same issues as pie charts. Use for single KPI display at most.

## Design principles

- **Title every chart** — no unnamed plots
- **Label axes** with units (e.g., "Revenue ($M)", not "revenue")
- **Use color meaningfully** — categorical distinction or sequential magnitude
- **Limit categories** — max 7-8 colors; group the rest as "Other"
- **Sort bars** — descending by value unless there's a natural order
- **Remove chartjunk** — no 3D effects, no unnecessary gridlines
- **Size for context** — presentations (16:9), reports (4:3), papers (single column)
- **Accessibility** — colorblind-safe palettes (`sns.color_palette("colorblind")`)
- **Bar charts start at zero** — always. A bar from 95 to 100 exaggerates a 5% difference
- **Show uncertainty** — error bars, confidence intervals, or ranges when data is uncertain
- **Highlight the story** — use a bright accent color for the key insight, grey everything else

## Accessibility checklist

Before sharing a visualization:
- [ ] Chart works without color (patterns, labels, or line styles differentiate series)
- [ ] Text readable at standard zoom (10pt+ labels, 12pt+ titles)
- [ ] Title describes the insight, not just the data
- [ ] Axes labeled with units
- [ ] Legend clear and not obscuring data
- [ ] Data source and date range noted
- [ ] Works in black and white (for printing)
- [ ] Uses colorblind-safe palette: `sns.color_palette("colorblind")` or `['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']`

## Export formats

| Format | Use case | Command |
|--------|----------|---------|
| PNG | Reports, presentations, README | `plt.savefig('chart.png', dpi=150)` |
| SVG | Scalable, web, papers | `plt.savefig('chart.svg')` |
| PDF | Publications, print | `plt.savefig('chart.pdf')` |
| HTML | Interactive, stakeholders | `fig.write_html('chart.html')` |

## Rules

- Always close figures after saving (`plt.close()`) to avoid memory leaks
- Set `tight_layout()` or `bbox_inches='tight'` to prevent label clipping
- Use `dpi=150` minimum for readable charts
- Save to a `figures/` or `charts/` directory, not project root
- Name files descriptively: `revenue_by_quarter.png` not `chart1.png`

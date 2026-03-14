# Chart Selection Guide

## By Question Type

| Question | Chart | Library |
|----------|-------|---------|
| How is X distributed? | Histogram, KDE | matplotlib, seaborn |
| How do groups compare? | Bar chart (horizontal for many) | matplotlib, plotly |
| How does X change over time? | Line chart | matplotlib, plotly |
| How do two variables relate? | Scatter plot | seaborn, plotly |
| What are the proportions? | Stacked bar (not pie) | matplotlib |
| How do features correlate? | Heatmap | seaborn |
| What is the range/spread? | Box plot, violin | seaborn |
| Geographic patterns? | Choropleth map | plotly |

## Style Defaults

- Figure size: 10x6 for single, 12x8 for multi-panel
- DPI: 150 for screen, 300 for print
- Font: 12pt body, 14pt titles
- Colors: Use colorblind-safe palettes (viridis, tab10)
- Grid: Light gray, behind data
- Titles: State the insight, not just the metric

## Export Formats

| Format | When | Command |
|--------|------|---------|
| PNG | Reports, slides | `plt.savefig('chart.png', dpi=150, bbox_inches='tight')` |
| SVG | Web, scalable | `plt.savefig('chart.svg')` |
| HTML | Interactive | `fig.write_html('chart.html')` (plotly) |
| PDF | Print | `plt.savefig('chart.pdf')` |

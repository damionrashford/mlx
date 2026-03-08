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

## Templates

### matplotlib + seaborn basics

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Style setup (do this once)
sns.set_theme(style="whitegrid", palette="husl", font_scale=1.1)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 150

df = pd.read_csv('data.csv')
```

### Bar chart

```python
fig, ax = plt.subplots()
sns.barplot(data=df, x='category', y='value', ax=ax)
ax.set_title('Value by Category')
ax.set_xlabel('Category')
ax.set_ylabel('Value')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('bar_chart.png', bbox_inches='tight')
plt.close()
```

### Line chart (time series)

```python
fig, ax = plt.subplots()
for group in df['segment'].unique():
    subset = df[df['segment'] == group]
    ax.plot(subset['date'], subset['value'], label=group, marker='o', markersize=4)
ax.set_title('Trend Over Time')
ax.legend()
ax.set_xlabel('Date')
ax.set_ylabel('Value')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('line_chart.png', bbox_inches='tight')
plt.close()
```

### Distribution

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df['value'], kde=True, ax=axes[0])
axes[0].set_title('Distribution of Value')
sns.boxplot(data=df, x='group', y='value', ax=axes[1])
axes[1].set_title('Value by Group')
plt.tight_layout()
plt.savefig('distribution.png', bbox_inches='tight')
plt.close()
```

### Correlation heatmap

```python
numeric = df.select_dtypes(include='number')
corr = numeric.corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax)
ax.set_title('Feature Correlations')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', bbox_inches='tight')
plt.close()
```

### Scatter with regression

```python
fig, ax = plt.subplots()
sns.regplot(data=df, x='feature_a', y='target', scatter_kws={'alpha': 0.5}, ax=ax)
ax.set_title('Feature A vs Target')
plt.tight_layout()
plt.savefig('scatter_regression.png', bbox_inches='tight')
plt.close()
```

### Multi-panel dashboard (matplotlib)

```python
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Data Dashboard', fontsize=16, fontweight='bold')

# Panel 1: KPI bar chart
sns.barplot(data=summary, x='metric', y='value', ax=axes[0, 0])
axes[0, 0].set_title('Key Metrics')

# Panel 2: Trend line
axes[0, 1].plot(ts['date'], ts['value'])
axes[0, 1].set_title('Trend Over Time')

# Panel 3: Distribution
sns.histplot(df['target'], kde=True, ax=axes[1, 0])
axes[1, 0].set_title('Target Distribution')

# Panel 4: Top categories
sns.barplot(data=top_n, x='count', y='category', ax=axes[1, 1])
axes[1, 1].set_title('Top 10 Categories')

plt.tight_layout()
plt.savefig('dashboard.png', bbox_inches='tight')
plt.close()
```

### Interactive plotly dashboard

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Revenue by Month', 'Category Breakdown',
                    'Distribution', 'Correlation'),
    specs=[[{"type": "scatter"}, {"type": "bar"}],
           [{"type": "histogram"}, {"type": "heatmap"}]]
)

# Panel 1: Line
fig.add_trace(go.Scatter(x=df['month'], y=df['revenue'], mode='lines+markers'), row=1, col=1)

# Panel 2: Bar
fig.add_trace(go.Bar(x=df['category'], y=df['count']), row=1, col=2)

# Panel 3: Histogram
fig.add_trace(go.Histogram(x=df['value'], nbinsx=30), row=2, col=1)

# Panel 4: Heatmap
fig.add_trace(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale='RdBu_r'), row=2, col=2)

fig.update_layout(height=800, title_text='Interactive Dashboard', showlegend=False)
fig.write_html('dashboard.html')
```

### Plotly single charts

```python
import plotly.express as px

# Interactive scatter
fig = px.scatter(df, x='x', y='y', color='group', size='value',
                 hover_data=['name'], title='Interactive Scatter')
fig.write_html('scatter.html')

# Interactive bar
fig = px.bar(df, x='category', y='value', color='segment',
             barmode='group', title='Grouped Bar Chart')
fig.write_html('bar.html')

# Sunburst (hierarchical)
fig = px.sunburst(df, path=['region', 'country', 'city'], values='sales')
fig.write_html('sunburst.html')
```

## Design principles

- **Title every chart** — no unnamed plots
- **Label axes** with units (e.g., "Revenue ($M)", not "revenue")
- **Use color meaningfully** — categorical distinction or sequential magnitude
- **Limit categories** — max 7-8 colors; group the rest as "Other"
- **Sort bars** — descending by value unless there's a natural order
- **Remove chartjunk** — no 3D effects, no unnecessary gridlines
- **Size for context** — presentations (16:9), reports (4:3), papers (single column)
- **Accessibility** — colorblind-safe palettes (`sns.color_palette("colorblind")`)

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

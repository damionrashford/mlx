# Jupyter Notebook Best Practices

## Structure

1. Title + description (markdown cell)
2. Imports (single cell, all at top)
3. Configuration / constants
4. Data loading
5. Analysis / processing (logical sections with markdown headers)
6. Results / visualization
7. Conclusions

## Quality Checklist

- [ ] Runs top-to-bottom without errors (Restart & Run All)
- [ ] No hardcoded paths (use relative paths or config)
- [ ] No unused imports or dead cells
- [ ] Markdown headers separate logical sections
- [ ] Outputs are meaningful (no raw DataFrame dumps >20 rows)
- [ ] Functions extracted for reused logic
- [ ] requirements.txt generated

## Conversion Targets

| Target | Command | When |
|--------|---------|------|
| Python script | `jupyter nbconvert --to script` | Production pipeline |
| HTML report | `jupyter nbconvert --to html` | Sharing with stakeholders |
| PDF | `jupyter nbconvert --to pdf` | Formal reports |
| Markdown | `jupyter nbconvert --to markdown` | Documentation |

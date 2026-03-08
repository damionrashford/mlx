---
name: review
description: >
  Produce structured, rigorous reviews of ML/AI research papers with strengths,
  weaknesses, methodology assessment, and reproducibility checks. Use when the
  user asks to "review a paper", "critique methodology", "assess reproducibility",
  "evaluate experimental design", or mentions paper review, peer review, or
  methodology critique.
allowed-tools: Bash, Read, WebFetch, Glob, Grep
argument-hint: path to PDF, paper ID, or URL (e.g. "2401.12345" or "./paper.pdf")
---

# Paper Review

Structured framework for reviewing ML/AI research papers. Produces fair, constructive, conference-quality reviews.

## Obtain the paper

Use the research skill scripts to get the paper content:

```bash
# Download by arXiv ID
python3 ${CLAUDE_SKILL_DIR}/../research/scripts/download.py 2401.12345 --output ./papers

# Extract text from PDF
python3 ${CLAUDE_SKILL_DIR}/../research/scripts/extract.py ./papers/2401.12345.pdf --max-pages 30

# Fetch metadata
python3 ${CLAUDE_SKILL_DIR}/../research/scripts/fetch.py 2401.12345
```

If the user provides only a topic, search first using the research skill, then review the selected paper.

## Review template

### Summary
2-3 sentences: What is the paper about? What is the key contribution?

### Strengths
Evaluate each dimension:
- **Novelty**: Is the approach new? Does it advance the field?
- **Experiments**: Well-designed? Sufficient baselines?
- **Clarity**: Well-written? Easy to follow?
- **Significance**: Would this matter if results hold?

### Weaknesses
Identify specific issues:
- Missing baselines or comparisons
- Claims not supported by evidence
- Methodology gaps or questionable choices
- Limited evaluation scope
- Scalability concerns

### Methodology assessment

| Dimension | Assessment | Notes |
|-----------|-----------|-------|
| Splits | proper / questionable / missing | Train/val/test separation |
| Baselines | fair / unfair / missing | SOTA included? |
| Metrics | appropriate / limited / wrong | Multiple metrics? |
| Significance | reported / missing | Error bars, CIs, p-values |
| Ablations | thorough / partial / none | Component contributions |

### Reproducibility checklist

- [ ] Code available (or promised)?
- [ ] Dataset available or described sufficiently?
- [ ] Hyperparameters fully specified?
- [ ] Compute requirements stated (GPU type, hours)?
- [ ] Random seeds reported?
- [ ] Training details sufficient to reproduce?
- [ ] Preprocessing steps documented?

### Questions for authors
3-5 specific questions that would strengthen the paper or clarify ambiguities.

### Overall assessment

| | Rating |
|---|---|
| **Recommendation** | Accept / Weak Accept / Borderline / Weak Reject / Reject |
| **Confidence** | High / Medium / Low |
| **Impact** | What would this enable if results hold? |

## Review ethics

- Be constructive — suggest improvements, don't just criticize
- Separate factual issues from opinions
- Acknowledge uncertainty in your assessment
- Evaluate what was claimed, not what wasn't attempted
- Consider the target venue and its standards
- Give credit where due — acknowledge genuine contributions

## Comparative review (multiple papers)

When reviewing multiple papers on the same topic:

| Dimension | Paper A | Paper B | Paper C |
|-----------|---------|---------|---------|
| Method | | | |
| Dataset | | | |
| Best metric | | | |
| Reproducibility | | | |
| Novelty | | | |

Rank by overall contribution, noting complementary strengths.

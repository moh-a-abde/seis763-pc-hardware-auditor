# SEIS763 - The PC Hardware Auditor

This repository contains the course project for `SEIS763 - Machine Learning`.

## Project Title

`The PC Hardware Auditor: Multi-Factor Regression for PC Component Price Optimization`

## Overview

The goal of this project is to model computer pricing from hardware specifications and use model residuals to identify systems that appear overpriced or underpriced relative to the broader market.

## Dataset

- Source: [Kaggle - All Computer Prices](https://www.kaggle.com/)
- Working assumption: the dataset contains desktop and laptop listings with component-level information and a price target
- One of the first project tasks is to validate the exact dataset fields, record count, and documented caveats before modeling

## Planned Workflow

1. Validate the dataset source and data dictionary.
2. Perform exploratory data analysis.
3. Build a baseline linear regression model.
4. Engineer and encode features for improved modeling.
5. Use residual analysis to flag overvalued and undervalued listings.
6. Summarize results in the final report and presentation.

## Repository Structure

- `notebooks/` - exploration and modeling notebooks
- `docs/data-notes/` - data descriptions, caveats, and task notes
- `reports/` - report drafts and presentation materials

## Collaboration

Please accept the repository invite, look through the open issues, and assign yourself to one issue before starting work. When you begin, move the issue to `In Progress` on the project board, and move it again when it is ready for review or done.

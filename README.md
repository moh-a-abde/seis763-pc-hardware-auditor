# SEIS763 - The PC Hardware Auditor

This repository contains the course project for `SEIS763 - Machine Learning`.

## Team

- Brian Beahan
- Lucky Onyemaobi
- Yonas Haddis
- Peter Spencer
- Mohamed Abdel-Hamid
- Oli Gurmessa

## Project Title

`The PC Hardware Auditor: Multi-Factor Regression for PC Component Price Optimization`

## Overview

The goal of this project is to model computer pricing from hardware specifications and use model residuals to identify systems that appear overpriced or underpriced relative to the broader market. More broadly, the project aims to turn pricing behavior into a measurable definition of "Brand Tax" and market value in the PC hardware space.

## Dataset Source

- Source: [Kaggle - All Computer Prices](https://www.kaggle.com/datasets/paperxd/all-computer-prices)
- Creator: `PaperXD`
- Current working assumption: the dataset is a merged collection of computer price records covering desktop and laptop configurations with component information and a price field
- Because the public documentation is limited, the exact field meanings, record count, and schema details must be validated before modeling conclusions are finalized

For provisional dataset counts, caveats, and validation notes, see `docs/data-notes/README.md`.

## Project Objectives

- Launch price prediction from available hardware specifications
- Anomaly detection through residual analysis to identify overpriced and underpriced listings
- Market dynamics analysis to determine which features most strongly influence price
- Value analysis framed around "Brand Tax" versus specification-driven pricing

## Planned Methods

- Data engineering and dataset validation before modeling
- Python as the shared implementation language for analysis and modeling
- Cleaning steps that may include removing outliers and dropping problematic attributes before training
- Target encoding or similar strategies for high-cardinality categorical variables such as brand and model
- Standardization of numeric features where appropriate
- Train/test splitting for evaluation on unseen data
- Baseline linear regression as the initial benchmark
- Feature engineering and model refinement after the schema is confirmed
- Residual analysis and feature-importance analysis for interpretation
- Optional follow-on classification work if the team defines a binary "good deal" versus "not a good deal" label clearly enough for logistic regression

## Planned Workflow

1. Validate the dataset source and data dictionary.
2. Perform exploratory data analysis.
3. Define the final target variable and modeling dataset.
4. Build a baseline linear regression model.
5. Engineer and encode features for improved modeling.
6. Use residual analysis to flag overvalued and undervalued listings.
7. Analyze feature importance and broader market dynamics.
8. Summarize results in the final report and presentation.

## Repository Structure

- `notebooks/` - exploration and modeling notebooks
- `docs/data-notes/` - data descriptions, caveats, and validation notes
- `reports/` - report drafts and presentation materials

## Collaboration Workflow

Please accept the repository invite, look through the open issues, and assign yourself to one issue before starting work. When you begin, move the issue to `In Progress` on the project board, and move it again when it is ready for review or done.

The current GitHub issues are organized around:

- dataset validation
- data dictionary and caveats
- exploratory data analysis
- target definition
- baseline modeling
- encoding strategy
- feature engineering
- residual analysis
- feature importance
- report and presentation drafting

## Notes On Dataset Validation

Some claims in the proposal are intentionally treated as provisional until the dataset is fully checked. In particular, the exact column count, the meaning of fields like `Model`, the usefulness of fields such as `cpu_model`, the existence of `RAM Type`, and the final wording of the prediction target may need to be revised after validation.

# Data Notes

This document captures the current understanding of the dataset, along with the caveats that still need to be validated before the team finalizes the modeling approach.

## Source

- Dataset: [All Computer Prices](https://www.kaggle.com/datasets/paperxd/all-computer-prices)
- Creator: `PaperXD`
- Host: `Kaggle`

## Provisional Dataset Summary

The current project proposal treats the dataset as a large merged computer-pricing dataset containing desktop and laptop configurations with an associated price field.

Provisional proposal values:

- records: `100,000+`
- attributes: `1033` in the draft proposal, but this may actually be closer to `33`
- predictors: `932`
- categorical attributes: `136`
- problem type: `regression`
- target: `Price` or market listing price

These values should be treated as working assumptions until the actual dataset file is profiled directly.

## Example Fields Mentioned In The Proposal

The proposal references fields such as:

- `Brand`
- `Model`
- `CPU Tier`
- `Core Count`
- `RAM Type`
- `Release Year`
- `OS`
- `Form Factor`
- `Category`
- `Price`

Some of these fields may be inferred rather than explicitly documented by the dataset source, so the team should confirm which ones actually exist and how consistently they are populated.

## Known Caveats

- Public documentation for the dataset appears limited, so some descriptions in the proposal may be based on inference.
- The attribute count likely needs correction. The draft mentions `1033`, while early review suggests the real number may be closer to `33`.
- The meaning of `Model` may be inconsistent or ambiguous for some records.
- Some attributes, including fields like `Model` or `cpu_model`, may behave more like identifiers or noisy descriptors than useful predictors.
- `RAM Type` may not exist as an explicit field; the dataset may only include RAM size.
- Some brand and operating-system combinations may be unrealistic and may need to be cleaned or filtered before modeling.
- Early review suggests there may be no missing values because the uploaded version appears to be pre-cleaned.
- Claims about launch price, MSRP, and specific 2025 to 2026 market behavior should be verified against the actual fields before they are presented as confirmed dataset facts.

## Proposed Cleaning And Preparation Steps

Based on the team discussion, the working preprocessing plan should consider:

- identifying and removing major outliers that would distort consumer-oriented predictions
- reviewing and dropping problematic attributes that hinder modeling
- encoding high-cardinality categorical variables
- standardizing numeric features where needed
- splitting the data into training and testing sets before evaluation

The team also discussed an optional second modeling track using logistic regression for a binary outcome such as `good deal` versus `not a good deal`. That should only be pursued after the team agrees on a defensible label definition.

## Validation Questions

The team should explicitly verify:

- the exact row count
- the exact column count
- the exact target field name
- whether `Price` reflects listing price, market price, launch price, or another pricing concept
- whether `Model` is reliable enough for direct use
- whether fields like `cpu_model` should be retained, transformed, or dropped
- whether `RAM Type` exists or needs to be replaced with another feature
- whether unrealistic brand and OS combinations should be filtered during cleaning
- whether the proposal's market-dynamics question should focus on CPU or GPU features instead

## Related Issues

The current GitHub tasks already map to these validation needs:

- validate dataset source and document actual fields
- write data dictionary and known data-quality caveats
- perform initial exploratory data analysis
- define the final prediction target and modeling dataset
- research encoding strategy for categorical variables
- engineer features for hardware pricing analysis

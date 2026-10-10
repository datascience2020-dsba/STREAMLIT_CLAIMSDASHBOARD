# Fraud Claims Dashboard

A Streamlit dashboard for analyzing net and gross fraud losses by channel,
recoverability, merchant geography, and month.

## Run the app

From the `uvdemo` project directory:

```powershell
uv sync
uv run streamlit run src/uvdemo/FraudClaimsVisualization.py
```

The app has three tabs: **FILE IMPORT**, **Tab 1: Net Fraud Loss Analysis**,
and **Tab 2: Gross Fraud Loss Analysis**. Upload a CSV or XLSX fraud-claims
dataset and map its columns in **FILE IMPORT**. The app suggests mappings from
uploaded column names to the required fraud-claims fields; review and complete
the mappings to enable the analysis tabs. For XLSX workbooks, the app uses the
`Transactions` sheet when present, otherwise it uses the first sheet. The
required fields are listed directly below the upload control.

The historical chart shows all 12 months of the latest transaction year as a
stacked area chart. Its selector switches between channel, recoverability, and
geography breakdowns.
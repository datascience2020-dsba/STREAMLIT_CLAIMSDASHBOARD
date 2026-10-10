"""Streamlit dashboard for net and gross fraud-claims analysis."""

from __future__ import annotations

from difflib import SequenceMatcher
import hashlib
from io import BytesIO
from pathlib import Path
import subprocess
import sys
from zipfile import BadZipFile

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


if __name__ == "__main__" and not st.runtime.exists():
    raise SystemExit(
        subprocess.call(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(Path(__file__).resolve()),
                *sys.argv[1:],
            ]
        )
    )


WORKSHEET_NAME = "Transactions"
REQUIRED_COLUMNS = (
    "TransactionDate",
    "Merchant Name",
    "Merchant Country",
    "Transaction Amount",
    "Transaction Type",
    "Fraud Flag",
    "Loss Recoverable Flag",
    "Net Fraud Loss Amount",
    "Gross Fraud Loss Amount",
    "Claims Processing Cost",
)
CHANNEL_ORDER = ("Card Present", "Card Not Present")
RECOVERABILITY_ORDER = ("Recoverable Loss", "Non-Recoverable Loss")
GEOGRAPHY_ORDER = ("Cross-Border Merchant", "US Merchant")
HISTORICAL_VIEWS = {
    "Channel Breakdown": ("Channel", CHANNEL_ORDER),
    "Recoverability Breakdown": ("Recoverability", RECOVERABILITY_ORDER),
    "Geography Breakdown": ("Geography", GEOGRAPHY_ORDER),
}
COLOR_MAP = {
    "Card Present": "#47B8C8",
    "Card Not Present": "#536DFE",
    "Recoverable Loss": "#40C4AA",
    "Non-Recoverable Loss": "#F08A5D",
    "US Merchant": "#6C8EF5",
    "Cross-Border Merchant": "#E7A34B",
}


@st.cache_data(show_spinner="Loading fraud claims...")
def read_uploaded_data(
    file_bytes: bytes,
    file_extension: str,
) -> tuple[pd.DataFrame, str | None]:
    """Read uploaded CSV or Excel bytes with Streamlit data caching."""
    uploaded = BytesIO(file_bytes)
    if file_extension == ".csv":
        return pd.read_csv(uploaded), None
    if file_extension == ".xlsx":
        excel = pd.ExcelFile(uploaded, engine="openpyxl")
        if not excel.sheet_names:
            raise ValueError("The uploaded workbook does not contain any worksheets.")
        sheet_name = (
            WORKSHEET_NAME
            if WORKSHEET_NAME in excel.sheet_names
            else excel.sheet_names[0]
        )
        return pd.read_excel(excel, sheet_name=sheet_name), sheet_name
    raise ValueError("Unsupported file type. Upload a CSV or XLSX file.")


def normalize_column_name(name: str) -> str:
    """Normalize a source or target name for fuzzy matching."""
    return "".join(character.lower() for character in name if character.isalnum())


def suggest_column_mapping(
    source_columns: list[str],
) -> dict[str, str | None]:
    """Suggest source fields using aliases and fuzzy matching."""
    aliases = {
        "TransactionDate": ("transactiondate", "date", "transaction_day"),
        "Merchant Name": ("merchantname", "merchant", "vendor", "store"),
        "Merchant Country": ("merchantcountry", "country", "merchantnation"),
        "Transaction Amount": (
            "transactionamount",
            "amount",
            "transactionvalue",
            "sales",
            "revenue",
        ),
        "Transaction Type": ("transactiontype", "type", "channel", "paymenttype"),
        "Fraud Flag": ("fraudflag", "isfraud", "fraud", "fraudulent"),
        "Loss Recoverable Flag": (
            "lossrecoverableflag",
            "recoverableflag",
            "recoverable",
            "recoveryflag",
        ),
        "Net Fraud Loss Amount": ("netfraudlossamount", "netloss", "netfraudloss"),
        "Gross Fraud Loss Amount": ("grossfraudlossamount", "grossloss", "grossfraudloss"),
        "Claims Processing Cost": ("claimsprocessingcost", "processingcost", "claimcost"),
    }
    normalized_sources = {
        column: normalize_column_name(str(column)) for column in source_columns
    }
    candidates: list[tuple[float, str, str]] = []
    for target in REQUIRED_COLUMNS:
        normalized_target = normalize_column_name(target)
        target_aliases = {
            normalize_column_name(alias) for alias in aliases.get(target, ())
        } | {normalized_target}
        for source in source_columns:
            normalized_source = normalized_sources[source]
            score = max(
                SequenceMatcher(None, alias, normalized_source).ratio()
                for alias in target_aliases
            )
            if any(
                alias in normalized_source or normalized_source in alias
                for alias in target_aliases
            ):
                score = max(score, 0.9)
            candidates.append((score, target, source))

    mapping: dict[str, str | None] = {target: None for target in REQUIRED_COLUMNS}
    used_sources: set[str] = set()
    for score, target, source in sorted(candidates, reverse=True):
        if score < 0.48 or mapping[target] is not None or source in used_sources:
            continue
        mapping[target] = source
        used_sources.add(source)
    return mapping


def prepare_fraud_data(
    raw_data: pd.DataFrame,
    column_mapping: dict[str, str],
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Apply user column mappings, validate data, and derive fraud fields."""
    missing_targets = sorted(set(REQUIRED_COLUMNS).difference(column_mapping))
    if missing_targets:
        raise ValueError(f"No source columns were mapped to: {', '.join(missing_targets)}")
    selected_columns = [column_mapping[target] for target in REQUIRED_COLUMNS]
    if len(set(selected_columns)) != len(selected_columns):
        raise ValueError("Each required field must map to a different uploaded column.")
    missing_sources = sorted(set(selected_columns).difference(raw_data.columns))
    if missing_sources:
        raise ValueError(f"Mapped source columns are missing: {', '.join(missing_sources)}")
    if raw_data.empty:
        raise ValueError("The selected dataset contains no transaction rows.")

    data = raw_data.loc[:, selected_columns].copy()
    data.columns = REQUIRED_COLUMNS
    data["TransactionDate"] = pd.to_datetime(
        data["TransactionDate"], errors="coerce"
    )
    if data["TransactionDate"].isna().any():
        invalid_count = int(data["TransactionDate"].isna().sum())
        raise ValueError(f"{invalid_count} row(s) have an invalid TransactionDate.")

    numeric_columns = (
        "Transaction Amount",
        "Fraud Flag",
        "Loss Recoverable Flag",
        "Net Fraud Loss Amount",
        "Gross Fraud Loss Amount",
        "Claims Processing Cost",
    )
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        if data[column].isna().any():
            invalid_count = int(data[column].isna().sum())
            raise ValueError(f"{invalid_count} row(s) have invalid values in {column}.")
    for column in ("Fraud Flag", "Loss Recoverable Flag"):
        if not data[column].isin((0, 1)).all():
            raise ValueError(f"{column} must contain only 0 or 1.")

    valid_types = {
        "Card Present Purchase",
        "Card Not Present Purchase",
        "Recurring Payment",
        "Cash Advance",
    }
    if (
        data["Transaction Type"].isna().any()
        or not data["Transaction Type"].isin(valid_types).all()
    ):
        raise ValueError("Transaction Type contains missing or unsupported values.")
    if data[["Merchant Name", "Merchant Country"]].isna().any().any():
        raise ValueError("Merchant Name and Merchant Country cannot be empty.")

    latest_month = data["TransactionDate"].max().to_period("M").to_timestamp()
    data["Month"] = data["TransactionDate"].dt.strftime("%Y-%m")
    fraud_data = data.loc[data["Fraud Flag"] == 1].copy()
    fraud_data["Geography"] = fraud_data["Merchant Country"].map(
        lambda country: (
            "US Merchant" if country == "United States" else "Cross-Border Merchant"
        )
    )
    fraud_data["Recoverability"] = fraud_data["Loss Recoverable Flag"].map(
        {1: "Recoverable Loss", 0: "Non-Recoverable Loss"}
    )
    fraud_data["Channel"] = fraud_data["Transaction Type"].map(
        {
            "Card Present Purchase": "Card Present",
            "Card Not Present Purchase": "Card Not Present",
            "Recurring Payment": "Card Not Present",
            "Cash Advance": "Card Not Present",
        }
    )
    return fraud_data, latest_month


def format_currency(value: float) -> str:
    """Format an amount as US dollars."""
    return f"${value:,.2f}"


def style_chart(fig: go.Figure) -> go.Figure:
    """Apply shared dark-theme, high-contrast styling to a Plotly chart."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#FFFFFF"),
        title_font=dict(color="#FFFFFF"),
        hoverlabel=dict(
            bgcolor="#182438",
            bordercolor="#607895",
            font=dict(color="#FFFFFF"),
        ),
        margin=dict(l=12, r=12, t=58, b=18),
        transition=dict(duration=500, easing="cubic-in-out"),
    )
    fig.update_xaxes(
        title_font=dict(color="#FFFFFF"),
        tickfont=dict(color="#FFFFFF"),
        gridcolor="rgba(190, 208, 235, 0.12)",
        zerolinecolor="rgba(190, 208, 235, 0.24)",
    )
    fig.update_yaxes(
        title_font=dict(color="#FFFFFF"),
        tickfont=dict(color="#FFFFFF"),
        gridcolor="rgba(190, 208, 235, 0.12)",
        zerolinecolor="rgba(190, 208, 235, 0.24)",
    )
    return fig


def build_breakdown_donut(
    month_data: pd.DataFrame,
    dimension: str,
    category_order: tuple[str, ...],
    amount_column: str,
    title: str,
) -> go.Figure:
    """Build a recent-month donut for one fraud loss dimension."""
    totals = (
        month_data.groupby(dimension, observed=False)[amount_column]
        .sum()
        .reindex(category_order, fill_value=0)
        .rename("Loss")
        .reset_index()
    )
    fig = px.pie(
        totals,
        names=dimension,
        values="Loss",
        hole=0.68,
        title=title,
        color=dimension,
        color_discrete_map=COLOR_MAP,
        category_orders={dimension: list(category_order)},
    )
    fig.update_traces(
        textinfo="percent",
        textfont=dict(color="#FFFFFF"),
        hovertemplate="%{label}<br>Loss: $%{value:,.2f}<extra></extra>",
        sort=False,
    )
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.16,
            xanchor="center",
            x=0.5,
            font=dict(color="#FFFFFF"),
        )
    )
    return style_chart(fig)


def render_recent_snapshot(
    fraud_data: pd.DataFrame,
    recent_month: pd.Timestamp,
    amount_column: str,
    tab_key: str,
) -> None:
    """Render recent-month KPIs, loss breakdowns, and merchant recommendations."""
    month_key = recent_month.strftime("%Y-%m")
    month_data = fraud_data.loc[fraud_data["Month"] == month_key]
    st.subheader(f"Recent Month Snapshot · {recent_month:%B %Y}")
    if month_data.empty:
        st.info("No fraud transactions are recorded in the most recent month.")
        return

    total_loss = float(month_data[amount_column].sum())
    processing_cost = float(month_data["Claims Processing Cost"].sum())
    loss_metric, cost_metric = st.columns(2)
    loss_metric.metric("Total fraud loss", format_currency(total_loss))
    cost_metric.metric(
        "Total claims processing cost", format_currency(processing_cost)
    )

    channel_column, recovery_column, geography_column = st.columns(3)
    with channel_column:
        st.plotly_chart(
            build_breakdown_donut(
                month_data,
                "Channel",
                CHANNEL_ORDER,
                amount_column,
                "Channel Breakdown",
            ),
            width="stretch",
            key=f"{tab_key}_channel_donut",
        )
    with recovery_column:
        st.plotly_chart(
            build_breakdown_donut(
                month_data,
                "Recoverability",
                RECOVERABILITY_ORDER,
                amount_column,
                "Recoverability Breakdown",
            ),
            width="stretch",
            key=f"{tab_key}_recoverability_donut",
        )
    with geography_column:
        st.plotly_chart(
            build_breakdown_donut(
                month_data,
                "Geography",
                GEOGRAPHY_ORDER,
                amount_column,
                "Geography Breakdown",
            ),
            width="stretch",
            key=f"{tab_key}_geography_donut",
        )

    top_merchants = (
        month_data.groupby("Merchant Name", as_index=False)[amount_column]
        .sum()
        .nlargest(10, amount_column)
        .sort_values(amount_column, ascending=True)
    )
    merchant_fig = px.bar(
        top_merchants,
        x=amount_column,
        y="Merchant Name",
        orientation="h",
        text=amount_column,
        title="Top 10 Fraud Merchants",
        labels={amount_column: "Fraud loss", "Merchant Name": ""},
        color=amount_column,
        color_continuous_scale=["#536DFE", "#F08A5D"],
    )
    merchant_fig.update_traces(
        texttemplate="$%{text:,.0f}",
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{y}<br>Loss: $%{x:,.2f}<extra></extra>",
    )
    merchant_fig.update_layout(
        coloraxis_showscale=False,
        yaxis=dict(categoryorder="total ascending"),
    )
    st.plotly_chart(
        style_chart(merchant_fig),
        width="stretch",
        key=f"{tab_key}_top_merchants",
    )
    st.caption(
        "💡 **Rule Recommendation Rule Base** — Build targeted rules or "
        "blocklists around these high-loss merchants. Tune thresholds to limit "
        "false positives and review rule performance regularly."
    )


def build_historical_area(
    fraud_data: pd.DataFrame,
    year: int,
    amount_column: str,
    dimension: str,
    category_order: tuple[str, ...],
) -> go.Figure:
    """Build a complete Jan–Dec stacked loss area chart."""
    months = pd.period_range(f"{year}-01", f"{year}-12", freq="M").strftime(
        "%Y-%m"
    )
    monthly = (
        fraud_data.loc[fraud_data["Month"].isin(months)]
        .groupby(["Month", dimension], as_index=False, observed=False)[amount_column]
        .sum()
    )
    full_index = pd.MultiIndex.from_product(
        [months, category_order], names=["Month", dimension]
    )
    monthly = (
        monthly.set_index(["Month", dimension])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )
    fig = px.area(
        monthly,
        x="Month",
        y=amount_column,
        color=dimension,
        title=f"Monthly Fraud Loss by {dimension} · {year}",
        labels={"Month": "Month", amount_column: "Fraud loss"},
        color_discrete_map=COLOR_MAP,
        category_orders={dimension: list(category_order)},
    )
    fig.update_traces(
        mode="lines",
        line=dict(width=2),
        hovertemplate="%{x}<br>%{legendgroup}: $%{y:,.2f}<extra></extra>",
    )
    fig.update_layout(
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.12,
            xanchor="center",
            x=0.5,
            font=dict(color="#FFFFFF"),
        ),
        margin=dict(l=12, r=12, t=88, b=48),
    )
    fig.update_xaxes(tickangle=-35, type="category")
    return style_chart(fig)


def render_historical_trend(
    fraud_data: pd.DataFrame,
    recent_month: pd.Timestamp,
    amount_column: str,
    tab_key: str,
) -> None:
    """Render a selectable yearly stacked-area loss trend."""
    st.subheader("Historical Month-over-Month Trends")
    selected_view = st.selectbox(
        "Historical chart view",
        options=list(HISTORICAL_VIEWS),
        key=f"{tab_key}_historical_view",
    )
    dimension, category_order = HISTORICAL_VIEWS[selected_view]
    fig = build_historical_area(
        fraud_data,
        recent_month.year,
        amount_column,
        dimension,
        category_order,
    )
    st.plotly_chart(
        fig,
        width="stretch",
        key=f"{tab_key}_historical_area",
    )


def render_loss_tab(
    fraud_data: pd.DataFrame,
    recent_month: pd.Timestamp,
    amount_column: str,
    tab_key: str,
    heading: str,
) -> None:
    """Render the snapshot and historical trend for a single loss measure."""
    st.markdown(f"### {heading}")
    render_recent_snapshot(fraud_data, recent_month, amount_column, tab_key)
    st.divider()
    render_historical_trend(fraud_data, recent_month, amount_column, tab_key)


def render_file_import_tab() -> tuple[pd.DataFrame | None, pd.Timestamp | None]:
    """Render upload and mapping controls, returning validated fraud data."""
    st.subheader("Upload fraud claims data")
    uploaded_file = st.file_uploader(
        "Upload fraud claims data",
        type=("csv", "xlsx"),
        help=(
            "Upload a CSV or Excel workbook. Excel files use the Transactions "
            "sheet when available."
        ),
    )
    st.markdown(
        "**Required columns**\n"
        + "\n".join(f"- `{column}`" for column in REQUIRED_COLUMNS)
    )
    if uploaded_file is None:
        st.info("Upload a CSV or XLSX dataset here to enable the analysis tabs.")
        return None, None

    try:
        file_extension = Path(uploaded_file.name).suffix.lower()
        file_bytes = uploaded_file.getvalue()
        raw_data, sheet_name = read_uploaded_data(file_bytes, file_extension)
    except (
        OSError,
        ValueError,
        ImportError,
        UnicodeDecodeError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        pd.errors.ExcelFileError,
        BadZipFile,
    ) as error:
        st.error(f"Unable to read the uploaded dataset: {error}")
        return None, None

    if raw_data.empty:
        st.error("The uploaded dataset contains no transaction rows.")
        return None, None
    raw_data.columns = [str(column) for column in raw_data.columns]
    if len(raw_data.columns) != len(set(raw_data.columns)):
        st.error(
            "The uploaded dataset has duplicate column names. "
            "Rename them and upload again."
        )
        return None, None

    st.subheader("Map uploaded columns")
    st.caption(
        "We suggest matches automatically. Review each dropdown and assign a unique "
        "source column to every required field."
    )
    file_key = hashlib.sha256(file_bytes).hexdigest()[:12]
    suggestions = suggest_column_mapping(list(raw_data.columns))
    column_mapping: dict[str, str] = {}
    mapping_columns = st.columns(2)
    for index, target in enumerate(REQUIRED_COLUMNS):
        options = ["— Select a column —", *raw_data.columns.tolist()]
        suggestion = suggestions[target]
        selected_index = options.index(suggestion) if suggestion in options else 0
        with mapping_columns[index % len(mapping_columns)]:
            selected_source = st.selectbox(
                target,
                options,
                index=selected_index,
                key=f"column-map-{file_key}-{index}",
            )
        if selected_source != "— Select a column —":
            column_mapping[target] = selected_source

    if len(column_mapping) != len(REQUIRED_COLUMNS):
        unmapped = [target for target in REQUIRED_COLUMNS if target not in column_mapping]
        st.warning(f"Map all required fields to continue: {', '.join(unmapped)}.")
        return None, None
    if len(set(column_mapping.values())) != len(column_mapping):
        st.error("Each required field must map to a different uploaded column.")
        return None, None

    try:
        fraud_data, recent_month = prepare_fraud_data(raw_data, column_mapping)
    except ValueError as error:
        st.error(f"Unable to prepare fraud claims data: {error}")
        return None, None

    source_label = uploaded_file.name
    if sheet_name is not None:
        source_label = f"{source_label} · sheet: {sheet_name}"
    st.caption(
        f"Data source: {source_label} · {len(fraud_data):,} fraud transactions"
    )
    if fraud_data.empty:
        st.info("No fraud transactions (Fraud Flag = 1) were found.")
    else:
        st.success("Data is ready. Open either analysis tab to explore fraud losses.")
    return fraud_data, recent_month


def main() -> None:
    """Run the fraud claims dashboard."""
    st.set_page_config(
        page_title="Fraud Claims Intelligence",
        page_icon="📊",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(145deg, #101827, #17243A); }
        .block-container { padding-top: 2rem; padding-bottom: 3rem; }
        .stApp, .stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3,
        .stApp h4, .stApp [data-testid="stCaptionContainer"],
        .stApp [data-testid="stMarkdownContainer"],
        .stApp [data-testid="stMetricLabel"],
        .stApp [data-testid="stMetricDelta"],
        .stApp [data-testid="stWidgetLabel"] {
            color: #FFFFFF !important;
        }
        [data-testid="stMetric"] {
            background: rgba(28, 42, 63, 0.78);
            border: 1px solid rgba(120, 151, 197, 0.2);
            padding: 1rem 1.2rem;
            border-radius: 0.8rem;
        }
        [data-testid="stMetricValue"] { color: #FFFFFF; }
        [data-testid="stTabs"] [role="tablist"] { gap: 0.5rem; }
        [data-testid="stTab"] {
            color: #FFFFFF !important;
            font-weight: 700;
            padding: 0.55rem 0.5rem;
        }
        [data-testid="stTab"] p {
            color: #FFFFFF !important;
            font-size: 1rem !important;
            font-weight: 700 !important;
        }
        [data-testid="stTab"][aria-selected="true"] {
            border-bottom-color: #47B8C8;
        }
        [data-testid="stPlotlyChart"] {
            animation: chart-enter 550ms ease-out both;
        }
        [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(1) [data-testid="stPlotlyChart"] { animation-delay: 0ms; }
        [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(2) [data-testid="stPlotlyChart"] { animation-delay: 120ms; }
        [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(3) [data-testid="stPlotlyChart"] { animation-delay: 240ms; }
        @keyframes chart-enter {
            from { opacity: 0; transform: translateY(12px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("Fraud Claims Intelligence")
    st.caption("Loss exposure, recovery performance, and merchant trends")
    import_tab, net_tab, gross_tab = st.tabs(
        [
            "FILE IMPORT",
            "Tab 1: Net Fraud Loss Analysis",
            "Tab 2: Gross Fraud Loss Analysis",
        ]
    )
    with import_tab:
        fraud_data, recent_month = render_file_import_tab()
    with net_tab:
        if fraud_data is None or recent_month is None:
            st.info("Complete the upload and column mapping in the FILE IMPORT tab.")
        elif fraud_data.empty:
            st.info("No fraud transactions (Fraud Flag = 1) are available to analyze.")
        else:
            render_loss_tab(
                fraud_data,
                recent_month,
                "Net Fraud Loss Amount",
                "net",
                "Net Fraud Loss Analysis",
            )
    with gross_tab:
        if fraud_data is None or recent_month is None:
            st.info("Complete the upload and column mapping in the FILE IMPORT tab.")
        elif fraud_data.empty:
            st.info("No fraud transactions (Fraud Flag = 1) are available to analyze.")
        else:
            render_loss_tab(
                fraud_data,
                recent_month,
                "Gross Fraud Loss Amount",
                "gross",
                "Gross Fraud Loss Analysis",
            )


if __name__ == "__main__":
    main()

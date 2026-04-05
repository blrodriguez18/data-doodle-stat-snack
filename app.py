import warnings
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    root_mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Auto Analytics Dashboard", layout="wide")
st.title("Auto Analytics Dashboard")
st.caption("Upload a CSV and get KPI cards, anomaly detection, forecasts, and ML insights.")
st.caption("Dashboard is automatically run on simulated sample data. To customize input, upload your own CSV file in the sidebar!")


# -----------------------------
# Data loading / cleaning
# -----------------------------
@st.cache_data
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)


@st.cache_data
def make_sample_data(n: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    trend = np.linspace(100, 250, n)
    seasonality = 15 * np.sin(np.arange(n) / 7.0)
    revenue = trend + seasonality + rng.normal(0, 10, n)
    revenue[60] *= 0.7
    revenue[150] *= 1.35

    signups = (50 + np.linspace(0, 30, n) + rng.normal(0, 6, n)).clip(min=0)
    churn_rate = (0.12 + 0.02 * np.sin(np.arange(n) / 14.0) + rng.normal(0, 0.005, n)).clip(0.02, 0.35)
    active_users = (1000 + np.linspace(0, 400, n) + rng.normal(0, 40, n)).clip(min=0)

    source = rng.choice(["Organic", "Paid", "Referral", "Email"], size=n, p=[0.35, 0.25, 0.2, 0.2])
    region = rng.choice(["NA", "EU", "APAC"], size=n, p=[0.5, 0.3, 0.2])

    return pd.DataFrame(
        {
            "date": dates,
            "revenue": np.round(revenue, 2),
            "signups": np.round(signups, 0).astype(int),
            "active_users": np.round(active_users, 0).astype(int),
            "churn_rate": np.round(churn_rate, 4),
            "source": source,
            "region": region,
        }
    )


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
            if parsed.notna().sum() >= max(3, int(0.6 * len(df))):
                df[col] = parsed
    return df


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["number"]).columns.tolist()


def datetime_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()


def text_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in numeric_columns(df) and c not in datetime_columns(df)]


def top_missing_columns(df: pd.DataFrame, k: int = 5) -> pd.Series:
    return df.isna().mean().sort_values(ascending=False).head(k)


def text_summary(df: pd.DataFrame) -> str:
    return (
        f"Rows: {len(df):,} | \n"
        f"Columns: {df.shape[1]} | \n"
        f"Numeric columns: {len(numeric_columns(df))} | \n"
        f"Date columns: {len(datetime_columns(df))} | \n"
        f"Missing cells: {int(df.isna().sum().sum()):,} | \n"
        f"Duplicate rows: {int(df.duplicated().sum()):,}"
    )


# -----------------------------
# Column detection helpers
# -----------------------------
def pick_best_column(df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
    cols = list(df.columns)
    low = {c.lower(): c for c in cols}
    for kw in keywords:
        for c_lower, c in low.items():
            if kw in c_lower:
                return c
    return None


def revenue_column(df: pd.DataFrame) -> Optional[str]:
    return pick_best_column(df, ["revenue", "sales", "arr", "mrr", "gmv", "income", "bookings", "turnover"])


def active_users_column(df: pd.DataFrame) -> Optional[str]:
    return pick_best_column(df, ["active_users", "active user", "users", "user_count", "au", "dau", "mau", "sessions"])


def churn_column(df: pd.DataFrame) -> Optional[str]:
    return pick_best_column(df, ["churn", "churn_rate", "churned", "attrition", "cancellation", "cancel_rate"])


def growth_reference_column(df: pd.DataFrame) -> Optional[str]:
    return pick_best_column(df, ["revenue", "active_users", "users", "signups", "new_users", "mrr", "arr"])


# -----------------------------
# Data filtering helpers
# -----------------------------
def apply_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    filtered = df.copy()

    with st.sidebar.expander("Time window", expanded=True):
        dcols = datetime_columns(filtered)
        if dcols:
            date_col = st.selectbox("Date column", dcols, index=0)
            filtered[date_col] = pd.to_datetime(filtered[date_col], errors="coerce")
            date_min = filtered[date_col].min()
            date_max = filtered[date_col].max()

            if pd.notna(date_min) and pd.notna(date_max):
                window_map = {
                    "All data": (date_min, date_max),
                    "Last 7 days": (max(date_min, date_max - pd.Timedelta(days=7)), date_max),
                    "Last 30 days": (max(date_min, date_max - pd.Timedelta(days=30)), date_max),
                    "Last 90 days": (max(date_min, date_max - pd.Timedelta(days=90)), date_max),
                    "Custom": None,
                }
                preset = st.radio("Preset", list(window_map.keys()), horizontal=False)
                if preset == "Custom":
                    start, end = st.date_input(
                        "Date range",
                        value=(date_min.date(), date_max.date()),
                    )
                    start = pd.Timestamp(start)
                    end = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                else:
                    start, end = window_map[preset]

                filtered = filtered[(filtered[date_col] >= start) & (filtered[date_col] <= end)]
            else:
                date_col = None
        else:
            date_col = None
            st.caption("No date column detected.")

    with st.sidebar.expander("Category filters", expanded=True):
        cats = text_columns(filtered)
        cat_cols = [c for c in cats if filtered[c].nunique(dropna=True) <= 30 and filtered[c].nunique(dropna=True) > 1]
        selected_cat = st.selectbox("Filter column", ["None"] + cat_cols)
        if selected_cat != "None":
            options = sorted([x for x in filtered[selected_cat].dropna().unique().tolist()])
            chosen = st.multiselect(f"Values for {selected_cat}", options, default=options[: min(3, len(options))])
            if chosen:
                filtered = filtered[filtered[selected_cat].isin(chosen)]

    with st.sidebar.expander("Numeric filters", expanded=False):
        num_cols = numeric_columns(filtered)
        selected_num = st.selectbox("Numeric column", ["None"] + num_cols)
        if selected_num != "None" and not filtered.empty:
            min_val = float(np.nanmin(filtered[selected_num].values))
            max_val = float(np.nanmax(filtered[selected_num].values))
            if np.isfinite(min_val) and np.isfinite(max_val) and min_val != max_val:
                low, high = st.slider(
                    f"Range for {selected_num}",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                )
                filtered = filtered[(filtered[selected_num] >= low) & (filtered[selected_num] <= high)]

    st.sidebar.divider()
    st.sidebar.caption("Tip: use a date column for anomaly detection and forecasting.")

    return filtered


# -----------------------------
# Insights 
# -----------------------------
def simple_insights(df: pd.DataFrame, metric: Optional[str] = None) -> List[str]:
    insights: List[str] = []

    if df.empty:
        return ["No rows remain after filtering."]

    num_cols = numeric_columns(df)
    if metric and metric in df.columns and pd.api.types.is_numeric_dtype(df[metric]) and len(df[metric].dropna()) >= 3:
        s = df[metric].dropna()
        first = float(s.iloc[0])
        last = float(s.iloc[-1])
        change_pct = ((last - first) / abs(first)) * 100 if first != 0 else np.nan
        if pd.notna(change_pct):
            insights.append(f"{metric} changed by {change_pct:.1f}% from the first to the last observed value.")


        if len(s) >= 7:
            recent = s.tail(7).mean()
            prior = s.head(max(len(s) - 7, 1)).tail(7).mean()
            if prior != 0:
                delta = ((recent - prior) / abs(prior)) * 100
                insights.append(f"The last 7 points are {delta:.1f}% different from the prior 7-point average.")

    miss = top_missing_columns(df, k=1)
    if not miss.empty and miss.iloc[0] > 0:
        insights.append(f"{miss.index[0]} has the most missing data ({miss.iloc[0] * 100:.1f}%).")

    if not insights:
        insights.append("No strong pattern jumped out from the current filter set.")

    return insights[:4]


def first_last_stacked_comparison(
    df: pd.DataFrame,
    date_col: str,
    metric: str,
    segment_col: str,
) -> Optional[go.Figure]:
    tmp = df[[date_col, metric, segment_col]].copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
    tmp = tmp.dropna(subset=[date_col, metric, segment_col])

    if tmp.empty:
        return None

    # first_date = tmp[date_col].min()
    # last_date = tmp[date_col].max()

    tmp = tmp.sort_values(date_col)

    first_df_window = tmp.head(7)   # first 7 rows
    last_df_window = tmp.tail(7)    # last 7 rows

    first_df = (
        first_df_window.groupby(segment_col, as_index=False)[metric]
        .mean()
        .rename(columns={metric: "First"})
    )

    last_df = (
        last_df_window.groupby(segment_col, as_index=False)[metric]
        .mean()
        .rename(columns={metric: "Last"})
    )

    plot_df = pd.merge(first_df, last_df, on=segment_col, how="outer").fillna(0)
    plot_df = plot_df.sort_values(segment_col)

    first_total = float(plot_df["First"].sum())
    last_total = float(plot_df["Last"].sum())

    fig = go.Figure()

    # One trace per segment; each trace contains both bars: First and Last
    for _, row in plot_df.iterrows():
        fig.add_trace(
            go.Bar(
                x=["First", "Last"],
                y=[row["First"], row["Last"]],
                name=str(row[segment_col]),
            )
        )

    pct_change = None
    if first_total != 0:
        pct_change = ((last_total - first_total) / abs(first_total)) * 100.0

    # y_max = max(first_total, last_total)
    # y_arrow = y_max * 1.12 if y_max > 0 else 1.0

    if pct_change is not None and np.isfinite(pct_change):

        first_total = float(plot_df["First"].sum())
        last_total = float(plot_df["Last"].sum())

        # Arrow from top of first bar to top of last bar
        fig.add_annotation(
            x="Last",
            y=last_total,
            xref="x",
            yref="y",
            ax="First",
            ay=first_total,
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.5,
            arrowwidth=3,
            arrowcolor="yellow",
        )

        # Percentage text above midpoint
        mid_y = max(first_total, last_total) * 1.08

        fig.add_annotation(
            x=0.5,
            xref="paper",
            y=mid_y,
            yref="y",
            text=f"<b>{pct_change:+.1f}%</b>",
            showarrow=False,
            font=dict(
                size=16,
                color="yellow"
            ),
        )

    fig.update_layout(
        barmode="stack",
        title=f"{metric.title()}: First 7 days vs Last 7 days, stacked by {segment_col}",
        xaxis_title="Period",
        yaxis_title=metric.title(),
        legend_title_text=segment_col,
        height=500,
    )

    return fig




# -----------------------------
# Time series helpers
# -----------------------------
def prepare_time_series(df: pd.DataFrame, date_col: str, metric: str) -> pd.DataFrame:
    ts = df[[date_col, metric]].copy()
    ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
    ts[metric] = pd.to_numeric(ts[metric], errors="coerce")
    ts = ts.dropna(subset=[date_col, metric]).sort_values(date_col)
    ts = ts.groupby(date_col, as_index=False)[metric].mean()
    return ts


def detect_anomalies(ts: pd.DataFrame, metric: str, window: int = 7, z_thresh: float = 2.5) -> pd.DataFrame:
    out = ts.copy()
    out["rolling_mean"] = out[metric].rolling(window=window, min_periods=max(3, window // 2)).mean()
    out["rolling_std"] = out[metric].rolling(window=window, min_periods=max(3, window // 2)).std(ddof=0)
    out["z_score"] = (out[metric] - out["rolling_mean"]) / out["rolling_std"].replace(0, np.nan)
    out["is_anomaly"] = out["z_score"].abs() >= z_thresh
    return out


def forecast_next(ts: pd.DataFrame, date_col: str, metric: str, steps: int = 7) -> pd.DataFrame:
    if len(ts) < 3:
        return pd.DataFrame()

    tmp = ts.copy().sort_values(date_col).reset_index(drop=True)
    x = np.arange(len(tmp), dtype=float)
    y = pd.to_numeric(tmp[metric], errors="coerce").astype(float).values

    mask = np.isfinite(y)
    x_fit = x[mask]
    y_fit = y[mask]
    if len(x_fit) < 3:
        return pd.DataFrame()

    slope, intercept = np.polyfit(x_fit, y_fit, 1)
    pred = slope * x_fit + intercept
    resid = y_fit - pred
    resid_std = float(np.std(resid)) if len(resid) > 1 else 0.0

    last_date = pd.to_datetime(tmp[date_col].iloc[-1])
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=steps, freq="D")
    future_x = np.arange(len(tmp), len(tmp) + steps, dtype=float)
    future_pred = slope * future_x + intercept

    forecast_df = pd.DataFrame(
        {
            date_col: future_dates,
            "prediction": future_pred,
            "lower": future_pred - 1.96 * resid_std,
            "upper": future_pred + 1.96 * resid_std,
        }
    )
    return forecast_df


# -----------------------------
# KPI helpers
# -----------------------------
def fmt_money(x: Optional[float]) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"${x:,.0f}"


def fmt_num(x: Optional[float]) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:,.0f}"


def fmt_pct(x: Optional[float]) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.1f}%"


def compute_kpis(df: pd.DataFrame, date_col: Optional[str]) -> dict:
    rev_col = revenue_column(df)
    users_col = active_users_column(df)
    churn_col = churn_column(df)
    growth_col = growth_reference_column(df)

    kpis = {"revenue": None, "active_users": None, "churn": None, "growth": None}

    if rev_col:
        kpis["revenue"] = float(pd.to_numeric(df[rev_col], errors="coerce").sum(skipna=True))

    if users_col:
        series = pd.to_numeric(df[users_col], errors="coerce")
        if "active" in users_col.lower() or "users" in users_col.lower():
            kpis["active_users"] = float(series.dropna().sum()) if series.notna().any() else None
        else:
            kpis["active_users"] = float(series.dropna().iloc[-1]) if series.notna().any() else None

    if churn_col:
        s = pd.to_numeric(df[churn_col], errors="coerce").dropna()
        if not s.empty:
            if s.max() <= 1.0:
                kpis["churn"] = float(s.mean() * 100.0)
            else:
                kpis["churn"] = float(s.mean())

    if date_col and growth_col and date_col in df.columns and growth_col in df.columns:
        tmp = df[[date_col, growth_col]].copy()
        tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
        tmp[growth_col] = pd.to_numeric(tmp[growth_col], errors="coerce")
        tmp = tmp.dropna().sort_values(date_col)
        if len(tmp) >= 14:
            last_7 = tmp.tail(7)[growth_col].sum()
            prev_7 = tmp.iloc[-14:-7][growth_col].sum()
            if prev_7 != 0:
                kpis["growth"] = ((last_7 - prev_7) / abs(prev_7)) * 100.0
        elif len(tmp) >= 2:
            first = tmp.iloc[0][growth_col]
            last = tmp.iloc[-1][growth_col]
            if first != 0:
                kpis["growth"] = ((last - first) / abs(first)) * 100.0

    return kpis


def render_kpi_row(kpis: dict):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Revenue", fmt_money(kpis.get("revenue")))
    with c2:
        st.metric("Active Users", fmt_num(kpis.get("active_users")))
    with c3:
        churn = kpis.get("churn")
        st.metric("Churn", fmt_pct(churn))
    with c4:
        growth = kpis.get("growth")
        delta = None if growth is None or not np.isfinite(growth) else f"{growth:.1f}%"
        st.metric("Growth", fmt_pct(growth), delta=delta)


# -----------------------------
# Data input
# -----------------------------
copyright = st.sidebar.write("Copyright © 2026 Beatriz Rodriguez. All rights reserved.")
uploaded = st.sidebar.file_uploader("Upload a CSV", type=["csv"])
data_source = st.sidebar.radio("Data source", ["Sample data", "Upload CSV"])

if data_source == "Upload CSV":
    if uploaded is not None:
        df = load_csv(uploaded)
        st.session_state["df"] = df
    else:
        st.warning("Please upload a CSV file.")
        df = make_sample_data()  # fallback so app doesn't break

else:  # Sample data
    df = make_sample_data()
    st.session_state["df"] = df

df = clean_df(df)
st.session_state["df"] = df

filtered_df = apply_sidebar_filters(df)

st.subheader("Overview")
st.write(text_summary(filtered_df))

date_cols = datetime_columns(filtered_df)
date_col = date_cols[0] if date_cols else None

kpis = compute_kpis(filtered_df, date_col)
render_kpi_row(kpis)

st.divider()

# -----------------------------
# ML helper functions
# -----------------------------

def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(df: pd.DataFrame, feature_cols: list[str]) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_features = [c for c in feature_cols if c not in numeric_features]

    transformers = []

    if numeric_features:
        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_pipe, numeric_features))

    if categorical_features:
        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", make_one_hot_encoder()),
            ]
        )
        transformers.append(("cat", categorical_pipe, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor, numeric_features, categorical_features


def get_feature_importance(pipe: Pipeline) -> Optional[pd.DataFrame]:
    preprocessor = pipe.named_steps.get("preprocess")
    model = pipe.named_steps.get("model")

    if preprocessor is None or model is None:
        return None

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        return None

    if hasattr(model, "coef_"):
        values = np.ravel(model.coef_)
        importance = np.abs(values)
        out = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importance,
                "coefficient": values,
            }
        ).sort_values("importance", ascending=False)
        return out

    if hasattr(model, "feature_importances_"):
        values = np.array(model.feature_importances_, dtype=float)
        out = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": values,
            }
        ).sort_values("importance", ascending=False)
        return out

    return None


def plot_feature_importance(fi: pd.DataFrame, title: str) -> None:
    if fi is None or fi.empty:
        st.info("Feature importance is not available for this model.")
        return

    show_df = fi.head(15).sort_values("importance", ascending=True)

    fig = px.bar(
        show_df,
        x="importance",
        y="feature",
        orientation="h",
        title=title,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(fi.head(20), use_container_width=True)


def fit_regression_model(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    model_name: str,
    test_size: float,
):
    if target_col not in df.columns:
        st.error("Invalid target column")
        return None

    if len(feature_cols) == 0:
        st.warning("Select at least one feature")
        return None
    
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        st.error("Target must be numeric for regression")
        return None
    
    if len(df) < 20:
        st.warning("Dataset is small — results may be unreliable")

    work = df[feature_cols + [target_col]].copy()
    work[target_col] = pd.to_numeric(work[target_col], errors="coerce")
    work = work.dropna(subset=[target_col])

    if work.empty:
        st.warning("No usable rows after cleaning the target column.")
        return None

    X = work[feature_cols]
    y = work[target_col]

    if len(work) < 10:
        st.warning("Not enough rows to train a useful regression model.")
        return None

    preprocessor, _, _ = build_preprocessor(work, feature_cols)

    if model_name == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=200, random_state=42)

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
    )

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    c1, c2, c3 = st.columns(3)
    c1.metric("R²", f"{r2_score(y_test, pred):.3f}")
    c2.metric("MAE", f"{mean_absolute_error(y_test, pred):.3f}")
    c3.metric("RMSE", f"{root_mean_squared_error(y_test, pred):.3f}")

    fig = px.scatter(
        x=y_test,
        y=pred,
        labels={"x": "Actual", "y": "Predicted"},
        title=f"{model_name}: Actual vs Predicted",
    )
    min_v = float(min(y_test.min(), pred.min()))
    max_v = float(max(y_test.max(), pred.max()))
    fig.add_trace(
        go.Scatter(
            x=[min_v, max_v],
            y=[min_v, max_v],
            mode="lines",
            name="Perfect fit",
        )
    )
    st.plotly_chart(fig, use_container_width=True)

    fi = get_feature_importance(pipe)
    if fi is not None:
        st.markdown("#### Feature importance")
        plot_feature_importance(fi, "Top features")
    else:
        st.info("This model does not expose feature importance in a simple way.")

    return pipe


def render_prediction_input_form(
    df: pd.DataFrame,
    feature_cols: list[str],
    pipe: Pipeline,
    target_col: str,
    key_prefix: str = "predict",
):
    st.markdown("#### Predict with custom inputs")

    with st.form(key=f"{key_prefix}_form"):
        input_data = {}

        for col in feature_cols:
            series = df[col]

            if pd.api.types.is_numeric_dtype(series):
                clean_series = pd.to_numeric(series, errors="coerce").dropna()

                if clean_series.empty:
                    value = 0.0
                    input_data[col] = st.number_input(
                        f"{col}",
                        value=value,
                        key=f"{key_prefix}_{col}",
                    )
                else:
                    min_val = float(clean_series.min())
                    max_val = float(clean_series.max())
                    mean_val = float(clean_series.mean())

                    if min_val == max_val:
                        input_data[col] = st.number_input(
                            f"{col}",
                            value=mean_val,
                            key=f"{key_prefix}_{col}",
                        )
                    else:
                        input_data[col] = st.number_input(
                            f"{col}",
                            min_value=min_val,
                            max_value=max_val,
                            value=mean_val,
                            key=f"{key_prefix}_{col}",
                        )
            else:
                options = sorted(series.dropna().astype(str).unique().tolist())
                if not options:
                    input_data[col] = st.text_input(
                        f"{col}",
                        value="",
                        key=f"{key_prefix}_{col}",
                    )
                else:
                    input_data[col] = st.selectbox(
                        f"{col}",
                        options,
                        key=f"{key_prefix}_{col}",
                    )

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame([input_data])

        try:
            pred = pipe.predict(input_df)[0]
            st.success(f"Predicted {target_col}: {pred:,.3f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")



def fit_classification_model(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    model_name: str,
    test_size: float,
):
    if target_col not in df.columns:
        st.error("Invalid target column")
        return None

    if len(feature_cols) == 0:
        st.warning("Select at least one feature")
        return None
    
    if df[target_col].nunique() < 2:
        st.error("Need at least 2 classes")
        return None
            
    if len(df) < 20:
        st.warning("Dataset is small — results may be unreliable")
      
    raw = df[target_col]

    if pd.api.types.is_numeric_dtype(raw):
        raw_numeric = pd.to_numeric(raw, errors="coerce")
        raw_numeric = raw_numeric.dropna()

        if raw_numeric.nunique() <= 1:
            st.warning("The selected target does not vary enough for classification.")
            return

        if raw_numeric.nunique() == 2:
            y = raw_numeric.astype(int)
            mask = df[target_col].notna()
            work = df.loc[mask, feature_cols + [target_col]].copy()
            work[target_col] = pd.to_numeric(work[target_col], errors="coerce").astype(int)
            X = work[feature_cols]
            y = work[target_col]
            class_note = "Using the numeric target as-is."
        else:
            min_v = float(raw_numeric.min())
            max_v = float(raw_numeric.max())
            default_thr = float(raw_numeric.median())
            threshold = st.slider(
                f"Binarize {target_col} at",
                min_value=min_v,
                max_value=max_v,
                value=default_thr,
            )
            mask = pd.to_numeric(df[target_col], errors="coerce").notna()
            work = df.loc[mask, feature_cols + [target_col]].copy()
            work[target_col] = (pd.to_numeric(work[target_col], errors="coerce") >= threshold).astype(int)
            X = work[feature_cols]
            y = work[target_col]
            class_note = f"Using 1 if {target_col} >= {threshold:.3f}, else 0."
    else:
        mask = raw.notna()
        work = df.loc[mask, feature_cols + [target_col]].copy()
        le = LabelEncoder()
        y_encoded = le.fit_transform(work[target_col].astype(str))
        if len(le.classes_) != 2:
            st.warning("Classification needs a binary target. Pick a binary column or use a numeric column and threshold it.")
            return
        work[target_col] = y_encoded
        X = work[feature_cols]
        y = work[target_col]
        class_note = f"Classes: {list(le.classes_)}"

    if y.nunique() != 2:
        st.warning("Classification needs exactly 2 classes.")
        return

    if len(work) < 10:
        st.warning("Not enough rows to train a useful classification model.")
        return

    st.caption(class_note)

    preprocessor, _, _ = build_preprocessor(work, feature_cols)

    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=2000)
    else:
        model = RandomForestClassifier(n_estimators=200, random_state=42)

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    stratify = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{accuracy_score(y_test, pred):.3f}")
    c2.metric("Precision", f"{precision_score(y_test, pred, zero_division=0):.3f}")
    c3.metric("Recall", f"{recall_score(y_test, pred, zero_division=0):.3f}")
    c4.metric("F1", f"{f1_score(y_test, pred, zero_division=0):.3f}")

    if hasattr(pipe.named_steps["model"], "predict_proba"):
        try:
            prob = pipe.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, prob)
            st.metric("ROC AUC", f"{auc:.3f}")
        except Exception:
            pass

    cm = confusion_matrix(y_test, pred)
    fig = px.imshow(
        cm,
        text_auto=True,
        labels={"x": "Predicted", "y": "Actual"},
        title=f"{model_name}: Confusion Matrix",
    )
    st.plotly_chart(fig, use_container_width=True)

    fi = get_feature_importance(pipe)
    if fi is not None:
        st.markdown("#### Feature importance")
        plot_feature_importance(fi, "Top features")
    else:
        st.info("This model does not expose feature importance in a simple way.")


def fit_clustering_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_clusters: int,
):
    if len(feature_cols) == 0:
        st.warning("Select at least one feature")
        return None
    
    if len(df) < 20:
        st.warning("Dataset is small — results may be unreliable")
    
    work = df[feature_cols].copy()

    if len(work) < n_clusters + 2:
        st.warning("Not enough rows for that many clusters.")
        return
    

    preprocessor, _, _ = build_preprocessor(work, feature_cols)
    X = preprocessor.fit_transform(work)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    st.metric("Rows clustered", f"{len(work):,}")
    st.metric("Clusters", f"{n_clusters}")

    try:
        sil = silhouette_score(X, labels)
        st.metric("Silhouette score", f"{sil:.3f}")
    except Exception:
        st.caption("Silhouette score unavailable for this dataset size or shape.")

    counts = pd.Series(labels).value_counts().sort_index().reset_index()
    counts.columns = ["cluster", "count"]
    fig_counts = px.bar(counts, x="cluster", y="count", title="Cluster sizes")
    st.plotly_chart(fig_counts, use_container_width=True)

    if X.shape[1] >= 2 and len(work) >= 3:
        pca = PCA(n_components=2, random_state=42)
        comps = pca.fit_transform(X)
        plot_df = pd.DataFrame(
            {
                "PC1": comps[:, 0],
                "PC2": comps[:, 1],
                "cluster": labels.astype(str),
            }
        )

        fig_scatter = px.scatter(
            plot_df,
            x="PC1",
            y="PC2",
            color="cluster",
            title="Clusters visualized with PCA",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # NEW: cluster profiling
    render_cluster_profiles(df, labels, feature_cols)
    plot_cluster_radar(df, labels, feature_cols)


def describe_cluster_profiles(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    feature_cols: list[str],
    top_n: int = 3,
) -> tuple[pd.DataFrame, dict[int, str]]:
    work = df[feature_cols].copy()
    work["cluster"] = cluster_labels

    numeric_feats = [c for c in feature_cols if pd.api.types.is_numeric_dtype(work[c])]
    cat_feats = [c for c in feature_cols if c not in numeric_feats]

    profiles = []
    cluster_names: dict[int, str] = {}

    overall_means = work[numeric_feats].mean(numeric_only=True) if numeric_feats else pd.Series(dtype=float)
    overall_stds = work[numeric_feats].std(numeric_only=True).replace(0, np.nan) if numeric_feats else pd.Series(dtype=float)

    for cluster_id in sorted(work["cluster"].unique()):
        sub = work[work["cluster"] == cluster_id]
        row = {
            "cluster": cluster_id,
            "size": len(sub),
            "share": len(sub) / len(work),
        }

        # Numeric summaries
        for col in numeric_feats:
            row[f"{col}_mean"] = float(sub[col].mean()) if sub[col].notna().any() else np.nan

        # Categorical summaries: most common value in the cluster
        for col in cat_feats:
            mode = sub[col].mode(dropna=True)
            row[f"{col}_top"] = mode.iloc[0] if not mode.empty else None

        profiles.append(row)

        # Create a simple human-readable label using biggest numeric deviations
        if numeric_feats:
            cluster_means = sub[numeric_feats].mean(numeric_only=True)
            z_diff = ((cluster_means - overall_means) / overall_stds).replace([np.inf, -np.inf], np.nan).dropna()
            if not z_diff.empty:
                strongest = z_diff.abs().sort_values(ascending=False).head(top_n)
                parts = []
                for feat in strongest.index:
                    val = z_diff[feat]
                    direction = "High" if val > 0 else "Low"
                    pretty = feat.replace("_", " ").title()
                    parts.append(f"{direction} {pretty}")
                cluster_names[cluster_id] = ", ".join(parts)
            else:
                cluster_names[cluster_id] = f"Cluster {cluster_id}"
        else:
            cluster_names[cluster_id] = f"Cluster {cluster_id}"

    profile_df = pd.DataFrame(profiles)

    # Add a friendly label column
    profile_df["label"] = profile_df["cluster"].map(cluster_names)

    # Reorder columns to keep it readable
    base_cols = ["cluster", "label", "size", "share"]
    other_cols = [c for c in profile_df.columns if c not in base_cols]
    profile_df = profile_df[base_cols + other_cols]

    return profile_df, cluster_names


def render_cluster_profiles(
    df: pd.DataFrame,
    cluster_labels: np.ndarray,
    feature_cols: list[str],
):
    profile_df, cluster_names = describe_cluster_profiles(df, cluster_labels, feature_cols)

    st.markdown("#### Cluster profiles")
    st.dataframe(profile_df, use_container_width=True)

    st.markdown("#### What the clusters mean")
    for _, row in profile_df.iterrows():
        label = row["label"]
        size = int(row["size"])
        share = float(row["share"]) * 100.0
        st.write(f"**Cluster {int(row['cluster'])}:** {label} — {size} rows ({share:.1f}%)")

    numeric_feats = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_feats and len(numeric_feats) > 0:
        top_feat = numeric_feats[0]
        if len(numeric_feats) > 1:
            # pick the most spread-out feature to make the chart more informative
            spreads = df[numeric_feats].std(numeric_only=True).sort_values(ascending=False)
            top_feat = spreads.index[0]

        plot_df = df[[top_feat]].copy()
        plot_df["cluster"] = cluster_labels.astype(str)



def plot_cluster_radar(df: pd.DataFrame, labels: np.ndarray, feature_cols: list[str]):
    work = df[feature_cols].copy()
    work["cluster"] = labels

    numeric_feats = [c for c in feature_cols if pd.api.types.is_numeric_dtype(work[c])]

    if len(numeric_feats) < 3:
        st.info("Need at least 3 numeric features for a radar chart.")
        return

    # Normalize features (0–1 scaling)
    norm_df = work[numeric_feats].copy()
    norm_df = (norm_df - norm_df.min()) / (norm_df.max() - norm_df.min())
    norm_df["cluster"] = labels

    # Compute cluster averages
    cluster_means = norm_df.groupby("cluster")[numeric_feats].mean()

    fig = go.Figure()

    for cluster_id in cluster_means.index:
        values = cluster_means.loc[cluster_id].tolist()
        values += values[:1]  # close the loop

        categories = numeric_feats + [numeric_feats[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=categories,
                fill="toself",
                name=f"Cluster {cluster_id}",
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        title="Cluster comparison (normalized feature space)",
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_ml_section(df: pd.DataFrame, date_col: Optional[str] = None):
    if df.empty:
        st.info("No rows available after filtering.")
        return

    mode = st.selectbox(
        "Choose an analysis",
        ["Predict Revenue", "Predict Churn", "Cluster Users"],
        key="ml_mode",
    )

    usable_cols = [c for c in df.columns if c != date_col and df[c].nunique(dropna=True) > 1]
    num_cols = [c for c in usable_cols if pd.api.types.is_numeric_dtype(df[c])]

    # -----------------------------
    # 1) Predict Revenue
    # -----------------------------
    if mode == "Predict Revenue":
        target_options = [c for c in num_cols if c != "churn_rate"]
        if not target_options:
            st.warning("No numeric target columns found.")
            return

        default_target = revenue_column(df) if revenue_column(df) in target_options else target_options[0]
        target_col = st.selectbox(
            "Target column",
            target_options,
            index=target_options.index(default_target),
            key="revenue_target",
        )

        feature_candidates = [c for c in usable_cols if c != target_col]
        default_features = [c for c in num_cols if c != target_col][:5]

        feature_cols = st.multiselect(
            "Feature columns",
            feature_candidates,
            default=default_features,
            key="revenue_features",
        )

        model_name = st.selectbox(
            "Model",
            ["Linear Regression", "Random Forest Regressor"],
            key="revenue_model",
        )

        test_size = st.slider(
            "Test size",
            min_value=0.1,
            max_value=0.4,
            value=0.2,
            step=0.05,
            key="revenue_test_size",
        )

        if feature_cols:
            pipe = fit_regression_model(df, target_col, feature_cols, model_name, test_size)
            if pipe is not None:
                st.markdown("---")
                render_prediction_input_form(
                    df=df,
                    feature_cols=feature_cols,
                    pipe=pipe,
                    target_col=target_col,
                    key_prefix="revenue_predict",
                )
        else:
            st.info("Pick at least one feature column.")

    # -----------------------------
    # 2) Predict Churn
    # -----------------------------
    elif mode == "Predict Churn":
        target_options = [c for c in usable_cols if c != date_col]
        if not target_options:
            st.warning("No target columns found.")
            return

        target_col = st.selectbox(
            "Target column",
            target_options,
            key="churn_target",
        )

        feature_candidates = [c for c in usable_cols if c != target_col]
        default_features = [c for c in num_cols if c != target_col][:5]

        feature_cols = st.multiselect(
            "Feature columns",
            feature_candidates,
            default=default_features,
            key="churn_features",
        )

        model_name = st.selectbox(
            "Model",
            ["Logistic Regression", "Random Forest Classifier"],
            key="churn_model",
        )

        test_size = st.slider(
            "Test size",
            min_value=0.1,
            max_value=0.4,
            value=0.2,
            step=0.05,
            key="churn_test_size",
        )

        if feature_cols:
            fit_classification_model(df, target_col, feature_cols, model_name, test_size)
        else:
            st.info("Pick at least one feature column.")

    # -----------------------------
    # 3) Cluster Users
    # -----------------------------
    elif mode == "Cluster Users":
        feature_candidates = usable_cols
        default_features = num_cols[:5] if num_cols else feature_candidates[:5]

        feature_cols = st.multiselect(
            "Feature columns",
            feature_candidates,
            default=default_features,
            key="cluster_features",
        )

        n_clusters = st.slider(
            "Number of clusters",
            min_value=2,
            max_value=10,
            value=3,
            step=1,
            key="cluster_k",
        )

        if feature_cols:
            fit_clustering_model(df, feature_cols, n_clusters)
        else:
            st.info("Pick at least one feature column.")

 
# -----------------------------
# Main content
# -----------------------------
left, right = st.columns([1.5, 1])

with left:
    st.subheader("Time series, anomalies, and forecast")

    num_cols = numeric_columns(filtered_df)
    if date_col and num_cols and not filtered_df.empty:
        metric_candidates = [c for c in num_cols if c != date_col]
        metric = st.selectbox("Metric", metric_candidates, index=0 if metric_candidates else None)

        anomaly_window = st.slider("Anomaly window", min_value=3, max_value=30, value=7)
        z_thresh = st.slider("Anomaly sensitivity (z-score threshold)", min_value=1.0, max_value=4.0, value=2.5, step=0.1)
        horizon = st.slider("Forecast horizon (days)", min_value=3, max_value=30, value=7)

        ts = prepare_time_series(filtered_df, date_col, metric)
        anomaly_df = detect_anomalies(ts, metric, window=anomaly_window, z_thresh=z_thresh)
        forecast_df = forecast_next(ts, date_col, metric, steps=horizon)

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=anomaly_df[date_col],
                y=anomaly_df[metric],
                mode="lines",
                name=metric,
            )
        )

        anomalies = anomaly_df[anomaly_df["is_anomaly"]]
        if not anomalies.empty:
            fig.add_trace(
                go.Scatter(
                    x=anomalies[date_col],
                    y=anomalies[metric],
                    mode="markers",
                    name="Anomalies",
                    marker=dict(size=10),
                )
            )

        if not forecast_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=forecast_df[date_col],
                    y=forecast_df["upper"],
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=forecast_df[date_col],
                    y=forecast_df["lower"],
                    mode="lines",
                    fill="tonexty",
                    line=dict(width=0),
                    name="Forecast band",
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=forecast_df[date_col],
                    y=forecast_df["prediction"],
                    mode="lines+markers",
                    name="Forecast",
                )
            )

        fig.update_layout(
            title=f"{metric} with anomalies and forecast",
            xaxis_title=date_col,
            yaxis_title=metric,
            legend_title_text="Series",
            height=520,
        )
        st.plotly_chart(fig, use_container_width=True)

        if not anomalies.empty:
            st.markdown("#### Detected anomalies")
            display_cols = [date_col, metric]
            extra_cols = [c for c in filtered_df.columns if c not in display_cols][:3]
            st.dataframe(anomalies[[date_col, metric]].merge(filtered_df[[date_col] + extra_cols], on=date_col, how="left"), use_container_width=True)
        else:
            st.info("No anomalies found with the current sensitivity settings.")

        if not forecast_df.empty:
            st.markdown("#### Forecast")
            st.dataframe(
                forecast_df.rename(columns={"prediction": f"{metric}_predicted"})[[date_col, f"{metric}_predicted", "lower", "upper"]],
                use_container_width=True,
            )
            next_value = float(forecast_df["prediction"].iloc[0])
            last_value = float(ts[metric].iloc[-1]) if not ts.empty else np.nan
            pct_change = ((next_value - last_value) / abs(last_value)) * 100 if last_value not in [0, np.nan] else np.nan
            st.metric("Next predicted value", f"{next_value:,.2f}", delta=f"{pct_change:.1f}% vs last observed" if np.isfinite(pct_change) else None)

    elif num_cols:
        metric = st.selectbox("Histogram metric", num_cols)
        fig = px.histogram(filtered_df, x=metric, nbins=30, title=f"Distribution of {metric}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No numeric columns detected.")

    st.subheader("Machine Learning Insights")
    st.caption("Choose a model to provide insights on a column")
    render_ml_section(filtered_df, date_col)

with right:
    st.subheader("Quick Insights")

    explanation_metric = None
    if date_col and numeric_columns(filtered_df):
        explanation_metric = st.selectbox(
            "Metric for explanation",
            [c for c in numeric_columns(filtered_df) if c != date_col],
            index=0 if [c for c in numeric_columns(filtered_df) if c != date_col] else None,
            key="explanation_metric",
        )

        insights = simple_insights(filtered_df, explanation_metric if 'metric' in locals() else None)
        for item in insights:
            st.write("*  " + item[0].title() + item[1:])

        if date_col and num_cols:
            st.markdown("### First vs Last comparison")

            metric_choices = [c for c in numeric_columns(filtered_df) if c != date_col]
            if metric_choices:
                compare_metric = st.selectbox(
                    "Metric to compare",
                    metric_choices,
                    key="compare_metric",
                )

                segment_choices = [
                    c for c in text_columns(filtered_df)
                    if c != date_col and filtered_df[c].nunique(dropna=True) <= 15 and filtered_df[c].nunique(dropna=True) > 1
                ]

                segment_col = st.selectbox(
                    "Stack by",
                    ["None"] + segment_choices,
                    key="segment_col",
                )

                if segment_col != "None":
                    fig = first_last_stacked_comparison(filtered_df, date_col, compare_metric, segment_col)
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Not enough data to build the comparison chart.")
                else:
                    st.info("Choose a column to stack by, like region or source.")
    else:
        st.text("No obvious patterns found yet. Try selecting a metric, filtering rows, or uploading a richer dataset.")

    ts_for_ai = None
    anomalies_for_ai = None
    forecast_for_ai = None
    if date_col and explanation_metric:
        ts_for_ai = prepare_time_series(filtered_df, date_col, explanation_metric)
        anomalies_for_ai = detect_anomalies(ts_for_ai, explanation_metric)
        forecast_for_ai = forecast_next(ts_for_ai, date_col, explanation_metric)

    st.subheader("Data quality")
    miss = top_missing_columns(filtered_df, k=5)
    miss_df = miss.reset_index()
    miss_df.columns = ["column", "missing_rate"]
    miss_df["missing_rate"] = miss_df["missing_rate"] * 100

    if not miss_df.empty:
        fig2 = px.bar(miss_df, x="column", y="missing_rate", title="Missing values (%)")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No missing data detected.")

    st.subheader("Quick correlations")
    num_cols = numeric_columns(filtered_df)
    if len(num_cols) >= 2:
        corr = filtered_df[num_cols].corr(numeric_only=True).copy()
        fig3 = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation matrix", color_continuous_scale="YlGnBu")
        st.plotly_chart(fig3, use_container_width=True)

        corr_array = corr.to_numpy(copy=True)
        np.fill_diagonal(corr_array, 0)

        pairs = corr.unstack().sort_values(ascending=False).drop_duplicates()
        pairs = pairs[pairs.index.get_level_values(0) != pairs.index.get_level_values(1)]
        if not pairs.empty:
            (a, b), val = pairs.index[0], pairs.iloc[0]
            st.write(f"Strongest correlation: {a} & {b} with |r| = {val:.2f}.")
    else:
        st.info("Not enough numeric columns for correlation analysis.")

st.divider()
st.markdown("### What to improve next")
st.write("* Integrate GridSearchCV to compare model performance and select the best model based on a criterion.")
st.write("* Connect dashboard to an LLM API to produce actionable insights.")

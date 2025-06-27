# dashboard_fpsos_streamlit.py
# Dashboard interativo para análise de eventos offshore por FPSO
# v2 – inclui Word Cloud e Network Graph

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import nltk
import networkx as nx
from io import BytesIO

# ------------------------------------------------------------------
# 🔧  Configurações iniciais
# ------------------------------------------------------------------
st.set_page_config(page_title="Dashboard de Segurança - FPSOs",
                   layout="wide",
                   page_icon="📊")

st.title("📊 Dashboard de Segurança - FPSOs")

# Garante que os recursos do NLTK estejam disponíveis
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

PORTUGUESE_STOPS = set(nltk.corpus.stopwords.words("portuguese"))
CUSTOM_STOPS = {"fps", "fpso", "na", "de", "em"}           # complementares
STOPWORDS_ALL = STOPWORDS.union(PORTUGUESE_STOPS).union(CUSTOM_STOPS)

# ------------------------------------------------------------------
# 📂  Upload
# ------------------------------------------------------------------
#uploaded_file = st.file_uploader("Envie o arquivo TRATADO_safeguardOffShore.xlsx",
#                                 type=["xlsx"])
#if not uploaded_file:
#    st.stop()


# ------------------------------------------------------------------
# 📥  Leitura e pré-processamento
# ------------------------------------------------------------------
RAW_URL = (
    "https://raw.githubusercontent.com/titetodesco/sphera/main/data/"
    "TRATADO_safeguardOffShore.xlsx"
)

@st.cache_data(ttl=3600)   # cache 1 h para não baixar a cada refresh
def load_data(url: str) -> pd.DataFrame:
    # engine="openpyxl" evita avisos em .xlsx
    return pd.read_excel(url, engine="openpyxl")

# Se ainda quiser manter a opção de upload manual:
if "uploaded_file" in st.session_state and st.session_state.uploaded_file:
    df = pd.read_excel(st.session_state.uploaded_file, engine="openpyxl")
else:
    df = load_data(RAW_URL)   # ← lê direto do GitHub

# continue o pré-processamento normalmente
df["Date Occurred"] = pd.to_datetime(df["Date Occurred"], errors="coerce")
df = df.drop_duplicates(subset=["Event ID"])


# ------------------------------------------------------------------
# 📥  Leitura e pré-processamento
# ------------------------------------------------------------------
#df = pd.read_excel(uploaded_file)
#df["Date Occurred"] = pd.to_datetime(df["Date Occurred"], errors="coerce")
#df = df.drop_duplicates(subset=["Event ID"])

# ---- Classificações auxiliares
def classify_tier_by_type(event_type: str) -> str:
    if pd.isna(event_type):
        return "Indefinido"
    e = event_type.lower().strip()
    if e == "incident":
        return "Tier 1-2"
    if e in {"observation", "near miss"}:
        return "Tier 3-4"
    return "Indefinido"


def classify_tier_by_severity(row) -> str:
    sev_cols = [
        "Event: Potential Severity - People",
        "Event: Potential Severity - Asset",
        "Event: Potential Severity - Environment",
        "Event: Potential Severity - Community",
    ]
    severities = [str(row.get(c)) for c in sev_cols if pd.notna(row.get(c))]
    if any(s.startswith(("3", "4")) for s in severities):
        return "Tier 1-2"
    if any(s.startswith(("0", "1", "2")) for s in severities):
        return "Tier 3-4"
    return "Indefinido"


df["Tier_by_type"] = df["Event Type"].apply(classify_tier_by_type)
df["Tier_by_severity"] = df.apply(classify_tier_by_severity, axis=1)
df["Ano-Mes"] = df["Date Occurred"].dt.to_period("M").astype(str)

# ------------------------------------------------------------------
# 🎛️  Filtros principais
# ------------------------------------------------------------------
fpsos_top = df["Location"].value_counts().nlargest(5).index.tolist()
selected_fpso = st.selectbox("Selecione a FPSO para análise:", fpsos_top)
df_fps = df[df["Location"] == selected_fpso].copy()

col_f1, col_f2, col_f3 = st.columns(3)
selected_event_type = col_f1.multiselect(
    "Tipo de evento:",
    options=df_fps["Event Type"].dropna().unique(),
    default=list(df_fps["Event Type"].dropna().unique()),
)
selected_tier_type = col_f2.multiselect(
    "Tier (Tipo de Evento):",
    options=["Tier 1-2", "Tier 3-4"],
    default=["Tier 1-2", "Tier 3-4"],
)
selected_tier_sev = col_f3.multiselect(
    "Tier (Severidade):",
    options=["Tier 1-2", "Tier 3-4"],
    default=["Tier 1-2", "Tier 3-4"],
)

df_fps = df_fps[
    (df_fps["Event Type"].isin(selected_event_type))
    & (df_fps["Tier_by_type"].isin(selected_tier_type))
    & (df_fps["Tier_by_severity"].isin(selected_tier_sev))
]

# ------------------------------------------------------------------
# 🔢 KPIs
# ------------------------------------------------------------------
st.markdown("### 📍 Visão Geral")
k1, k2, k3 = st.columns(3)
k1.metric("Eventos totais", len(df_fps))
k2.metric("Tier 1-2", (df_fps["Tier_by_severity"] == "Tier 1-2").sum())
k3.metric("Tier 3-4", (df_fps["Tier_by_severity"] == "Tier 3-4").sum())

# ------------------------------------------------------------------
# 📈 Tendência semanal
# ------------------------------------------------------------------
st.subheader("Tendência semanal de Tier 3-4 (Observations e Near Misses)")
df_fps["Semana"] = df_fps["Date Occurred"].dt.to_period("W").astype(str)
weekly_t3 = (
    df_fps[df_fps["Tier_by_severity"] == "Tier 3-4"]
    .groupby("Semana")["Event ID"]
    .nunique()
    .reset_index(name="Frequência")
)
q3 = weekly_t3["Frequência"].quantile(0.75)
weekly_t3["Anomalia"] = weekly_t3["Frequência"] > q3
fig_tendencia = px.bar(
    weekly_t3,
    x="Semana",
    y="Frequência",
    color="Anomalia",
    color_discrete_map={True: "crimson", False: "steelblue"},
)
st.plotly_chart(fig_tendencia, use_container_width=True)

# ------------------------------------------------------------------
# 📊 Principais categorias
# ------------------------------------------------------------------
st.markdown("### Principais categorias")
c1, c2 = st.columns(2)
with c1:
    task_counts = (
        df_fps["Task / Activity"]
        .value_counts()
        .nlargest(10)
        .reset_index()
        .rename(columns={"index": "Task / Activity", "Task / Activity": "count"})
    )
    fig2 = px.bar(task_counts, x="Task / Activity", y="count")
    st.plotly_chart(fig2, use_container_width=True)

with c2:
    risk_counts = (
        df_fps["Risk Area"]
        .value_counts()
        .nlargest(8)
        .reset_index()
        .rename(columns={"index": "Risk Area", "Risk Area": "count"})
    )
    fig3 = px.pie(risk_counts, names="Risk Area", values="count")
    st.plotly_chart(fig3, use_container_width=True)

# ------------------------------------------------------------------
# 🧠 Human Factors
# ------------------------------------------------------------------
if "Event: Human Factors" in df_fps.columns:
    st.markdown("### Top 10 Human Factors")
    hf_counts = (
        df_fps["Event: Human Factors"]
        .value_counts()
        .nlargest(10)
        .reset_index()
        .rename(columns={"index": "Human Factor", "Event: Human Factors": "count"})
    )
    fig4 = px.bar(hf_counts, x="Human Factor", y="count")
    st.plotly_chart(fig4, use_container_width=True)

# ------------------------------------------------------------------
# 🔥 Heatmap Risk Area × Task
# ------------------------------------------------------------------
st.markdown("### Heatmap: Risk Area × Task / Activity")


def plot_heatmap(df_heat: pd.DataFrame) -> None:
    df_sub = df_heat.dropna(subset=["Risk Area", "Task / Activity"]).drop_duplicates(
        subset=["Event ID", "Risk Area", "Task / Activity"]
    )
    co = (
        df_sub.groupby(["Risk Area", "Task / Activity"])["Event ID"]
        .nunique()
        .reset_index(name="count")
    )
    if co.empty:
        st.warning("Sem dados suficientes para gerar heatmap.")
        return
    mat = co.pivot(index="Risk Area", columns="Task / Activity", values="count").fillna(0)
    fig, ax = plt.subplots(
        figsize=(1.1 * len(mat.columns) + 2, 0.6 * len(mat.index) + 2)
    )
    sns.heatmap(
        mat,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        cbar_kws={"label": "Nº de eventos"},
        ax=ax,
    )
    ax.set_title("Risk Area × Task / Activity")
    st.pyplot(fig)


plot_heatmap(df_fps)

# ------------------------------------------------------------------
# 🗓️ Timeline
# ------------------------------------------------------------------
st.markdown("## Timeline de Eventos")
fig_timeline = px.strip(
    df_fps,
    x="Date Occurred",
    y="Event Type",
    color="Tier_by_severity",
    hover_data=["Title", "Risk Area", "Task / Activity"],
    stripmode="overlay",
)
fig_timeline.update_traces(marker=dict(size=8, opacity=0.7))
fig_timeline.update_layout(height=300, margin=dict(t=30, b=10))
st.plotly_chart(fig_timeline, use_container_width=True)

# ------------------------------------------------------------------
# 🧠 Detecção de Precursores
# ------------------------------------------------------------------
st.markdown("## Detecção de Precursores para Incidentes")
df_fps = df_fps.sort_values("Date Occurred")
incidents = df_fps[df_fps["Event Type"].str.lower() == "incident"]

for _, row in incidents.iterrows():
    data_evento = row["Date Occurred"].date()
    st.markdown(f"### 🚨 Incidente em {data_evento} — *{row.get('Title', '').strip()}*")
    st.markdown(
        f"Risk Area: **{row['Risk Area']}**, "
        f"Task: **{row['Task / Activity']}**, "
        f"Human Factor: **{row.get('Event: Human Factors', 'N/A')}**"
    )

    inicio = row["Date Occurred"] - pd.Timedelta(days=30)
    anteriores = df_fps[
        (df_fps["Date Occurred"] >= inicio)
        & (df_fps["Date Occurred"] < row["Date Occurred"])
    ]
    precursores = anteriores[
        (anteriores["Risk Area"] == row["Risk Area"])
        & (anteriores["Task / Activity"] == row["Task / Activity"])
        & (anteriores["Event: Human Factors"] == row["Event: Human Factors"])
    ]
    st.markdown(f"Precursores identificados nos 30 dias anteriores: **{len(precursores)}**")
    if not precursores.empty:
        st.dataframe(
            precursores[
                ["Event ID", "Event Type", "Date Occurred", "Risk Area", "Task / Activity"]
            ]
        )

# ------------------------------------------------------------------
# 🌳 Word Cloud & 🔗 Network Graph
# ------------------------------------------------------------------
st.markdown("## Word Cloud & Network Graph")

wc_col, net_col = st.columns(2)

# --- Preparação do texto -----------------------------------------------------
texto_base = (
    df_fps["Title"].fillna("").astype(str)
    + " "
    + df_fps.get("Description", pd.Series("", index=df_fps.index)).fillna("").astype(str)
)
texto_base = (
    texto_base.str.replace(r"[^\w\s]", " ", regex=True)
    .str.lower()
    .str.cat(sep=" ")
)
tokens = [t for t in texto_base.split() if t not in STOPWORDS_ALL and len(t) > 3]

# --- Word Cloud --------------------------------------------------------------
with wc_col:
    if tokens:
        wc = WordCloud(
            width=600,
            height=400,
            background_color="white",
            stopwords=STOPWORDS_ALL,
            colormap="viridis",
        ).generate(" ".join(tokens))
        fig_wc, ax_wc = plt.subplots(figsize=(6, 4))
        ax_wc.imshow(wc, interpolation="bilinear")
        ax_wc.axis("off")
        st.pyplot(fig_wc)
    else:
        st.info("Dados insuficientes para gerar Word Cloud.")

# --- Network Graph (co-ocorrência de bigramas) -------------------------------
with net_col:
    if len(tokens) >= 2:
        # Gera bigramas
        bigrams = list(zip(tokens[:-1], tokens[1:]))
        # Conta ocorrências
        bigram_df = (
            pd.Series(bigrams).value_counts().reset_index(name="freq").head(40)
        )  # top 40
        # Cria grafo
        G = nx.Graph()
        for (w1, w2), freq in bigram_df.values:
            G.add_edge(w1, w2, weight=freq)

        # Layout & Plotly
        pos = nx.spring_layout(G, k=0.5, seed=42)
        edge_x, edge_y, edge_w = [], [], []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
            edge_w.append(edge[2]["weight"])

        node_x, node_y, node_text = [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1, color="#888"),
            hoverinfo="none",
            mode="lines",
        )

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            hoverinfo="text",
            marker=dict(
                size=12,
                color="#1f77b4",
                line=dict(width=2, color="#FFFFFF"),
            ),
        )

        fig_net = go.Figure(data=[edge_trace, node_trace])
        fig_net.update_layout(
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
            height=400,
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
        )
        st.plotly_chart(fig_net, use_container_width=True)
    else:
        st.info("Dados insuficientes para gerar Network Graph.")

# streamlit_dashboard.py
# Dashboard interativo para análise de eventos offshore por FPSO

import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Dashboard de Segurança por FPSO")
st.title("📊 Dashboard de Segurança - FPSOs")

# --- Upload do arquivo ---
uploaded_file = st.file_uploader("Envie o arquivo TRATADO_safeguardOffShore.xlsx", type=["xlsx"])
if not uploaded_file:
    st.stop()

# --- Leitura dos dados ---
df = pd.read_excel(uploaded_file)
df["Date Occurred"] = pd.to_datetime(df["Date Occurred"], errors="coerce")
df = df.drop_duplicates(subset=["Event ID"])

# --- Classificação de severidade ---
def classify_tier_by_type(event_type):
    if pd.isna(event_type): return "Indefinido"
    e = event_type.lower().strip()
    return "Tier 1-2" if e == "incident" else "Tier 3-4" if e in ["observation", "near miss"] else "Indefinido"

def classify_tier_by_severity(row):
    sev = [row.get(c) for c in [
        "Event: Potential Severity - People", "Event: Potential Severity - Asset",
        "Event: Potential Severity - Environment", "Event: Potential Severity - Community"]]
    sev = [s for s in sev if pd.notna(s)]
    if any(str(s).startswith("3") or str(s).startswith("4") for s in sev): return "Tier 1-2"
    if any(str(s).startswith("2") or str(s).startswith("1") or str(s).startswith("0") for s in sev): return "Tier 3-4"
    return "Indefinido"

df["Tier_by_type"] = df["Event Type"].apply(classify_tier_by_type)
df["Tier_by_severity"] = df.apply(classify_tier_by_severity, axis=1)
df["Ano-Mes"] = df["Date Occurred"].dt.to_period("M").astype(str)

# --- Seleção de FPSO ---
fpsos_top = df["Location"].value_counts().nlargest(5).index.tolist()
selected_fpso = st.selectbox("Selecione a FPSO para análise:", fpsos_top)
df_fps = df[df["Location"] == selected_fpso].copy()

# --- KPIs ---
st.markdown("### 📍 Visão Geral")
col1, col2, col3 = st.columns(3)
col1.metric("Eventos totais", len(df_fps))
col2.metric("Tier 1-2", (df_fps["Tier_by_type"] == "Tier 1-2").sum())
col3.metric("Tier 3-4", (df_fps["Tier_by_type"] == "Tier 3-4").sum())

# --- Gráfico de tendência ---
st.markdown("### 📈 Tendência mensal por Tier (tipo de evento)")
df_tmp = df_fps.groupby(["Ano-Mes", "Tier_by_type"]).size().reset_index(name="Eventos")
fig1 = px.line(df_tmp, x="Ano-Mes", y="Eventos", color="Tier_by_type", markers=True)
st.plotly_chart(fig1, use_container_width=True)

# --- Distribuições ---
st.markdown("### 📊 Principais categorias")
c1, c2 = st.columns(2)

with c1:
    task_counts = df_fps["Task / Activity"].value_counts().nlargest(10).reset_index()
    task_counts.columns = ["Task / Activity", "count"]
    fig2 = px.bar(task_counts, x="Task / Activity", y="count", title="Top 10 Task / Activity")
    st.plotly_chart(fig2, use_container_width=True)

with c2:
    risk_counts = df_fps["Risk Area"].value_counts().nlargest(8).reset_index()
    risk_counts.columns = ["Risk Area", "count"]
    fig3 = px.pie(risk_counts, names="Risk Area", values="count", title="Distribuição por Risk Area")
    st.plotly_chart(fig3, use_container_width=True)

# --- Human Factors ---
if "Event: Human Factors" in df_fps.columns:
    st.markdown("### 🧠 Top 10 Human Factors")
    hf_counts = df_fps["Event: Human Factors"].value_counts().nlargest(10).reset_index()
    hf_counts.columns = ["Human Factor", "count"]
    fig4 = px.bar(hf_counts, x="Human Factor", y="count")
    st.plotly_chart(fig4, use_container_width=True)

# --- Heatmap: Risk Area × Task ---
st.markdown("### 🔥 Heatmap: Risk Area × Task / Activity")
def plot_heatmap(df_heat):
    df_sub = df_heat.dropna(subset=["Risk Area", "Task / Activity"])
    df_sub = df_sub.drop_duplicates(subset=["Event ID", "Risk Area", "Task / Activity"])
    co = df_sub.groupby(["Risk Area", "Task / Activity"])["Event ID"].nunique().reset_index(name="count")
    if co.empty:
        st.warning("Sem dados suficientes para gerar heatmap.")
        return
    mat = co.pivot(index="Risk Area", columns="Task / Activity", values="count").fillna(0)
    fig, ax = plt.subplots(figsize=(1.2*len(mat.columns), 0.6*len(mat.index)+2))
    sns.heatmap(mat, annot=True, fmt=".0f", cmap="YlOrRd", cbar_kws={"label": "Nº de eventos"}, ax=ax)
    ax.set_title("Risk Area × Task / Activity")
    st.pyplot(fig)

plot_heatmap(df_fps)
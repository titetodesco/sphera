# dashboard_precursores_streamlit.py
# Análise de precursores e achados por FPSO com filtros por semestre e detecção de padrões

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import timedelta

st.set_page_config(layout="wide", page_title="Análise de Precursores - FPSOs")
st.title("🔎 Análise de Precursores por FPSO")

# --- Leitura da planilha diretamente do GitHub ---
url = "https://raw.githubusercontent.com/SEU_USUARIO/SEU_REPOSITORIO/main/TRATADO_safeguardOffShore.xlsx"
df = pd.read_excel(url)
df["Date Occurred"] = pd.to_datetime(df["Date Occurred"], errors="coerce")
df = df.drop_duplicates(subset=["Event ID"])

# --- Classificação Tier por severidade ---
def classify_tier_by_severity(row):
    sev = [row.get(c) for c in [
        "Event: Potential Severity - People", "Event: Potential Severity - Asset",
        "Event: Potential Severity - Environment", "Event: Potential Severity - Community"]]
    sev = [s for s in sev if pd.notna(s)]
    if any(str(s).startswith("3") or str(s).startswith("4") for s in sev): return "Tier 1-2"
    if any(str(s).startswith("2") or str(s).startswith("1") or str(s).startswith("0") for s in sev): return "Tier 3-4"
    return "Indefinido"

df["Tier_by_severity"] = df.apply(classify_tier_by_severity, axis=1)
df["Ano"] = df["Date Occurred"].dt.year
df["Mes"] = df["Date Occurred"].dt.month

# --- Filtros ---
fpsos_top = df["Location"].value_counts().nlargest(5).index.tolist()
col1, col2 = st.columns(2)
selected_fpso = col1.selectbox("Selecione a FPSO:", fpsos_top)
selected_year = col2.selectbox("Ano:", sorted(df["Ano"].dropna().unique().astype(int)))
semestre = st.radio("Semestre:", ["1", "2"], horizontal=True)

mes_range = range(1, 7) if semestre == "1" else range(7, 13)
df_sel = df[(df["Location"] == selected_fpso) &
            (df["Ano"] == selected_year) &
            (df["Mes"].isin(mes_range))].copy()

# --- Timeline de eventos ---
st.subheader("📆 Timeline de Eventos")
df_plot = df_sel.dropna(subset=["Date Occurred"])
df_plot = df_plot.sort_values("Date Occurred")
fig_timeline = px.scatter(df_plot,
                          x="Date Occurred", y="Event Type",
                          color="Tier_by_severity",
                          hover_data=["Event ID", "Risk Area", "Task / Activity"],
                          title="Eventos por tipo e severidade no período selecionado")
st.plotly_chart(fig_timeline, use_container_width=True)

# --- Detecção de precursores ---
st.subheader("🧠 Detecção de Precursores para Incidentes")
achados = []
incidents = df_sel[df_sel["Event Type"].str.lower() == "incident"]

for _, inc in incidents.iterrows():
    inc_date = inc["Date Occurred"]
    inc_id = inc["Event ID"]
    inc_area = inc["Risk Area"]
    inc_task = inc["Task / Activity"]
    inc_hf = inc.get("Event: Human Factors", None)

    mask = (df_sel["Date Occurred"] >= inc_date - timedelta(days=30)) & \
           (df_sel["Date Occurred"] < inc_date) & \
           (df_sel["Event Type"].str.lower().isin(["observation", "near miss"]))
    anteriores = df_sel[mask]

    matches = anteriores[
        (anteriores["Risk Area"] == inc_area) |
        (anteriores["Task / Activity"] == inc_task) |
        (anteriores["Event: Human Factors"] == inc_hf)
    ]

    if not matches.empty:
        achados.append({
            "Data": inc_date.date(),
            "Evento ID": inc_id,
            "Risk Area": inc_area,
            "Task": inc_task,
            "Human Factor": inc_hf,
            "Qtd Precursores": len(matches),
            "Detalhes": matches[["Event ID", "Event Type", "Date Occurred", "Risk Area", "Task / Activity"]]
        })

# --- Apresenta achados ---
if achados:
    for a in achados:
        st.markdown(f"### 🚨 Incidente em {a['Data']} (Evento ID: {a['Evento ID']})")
        st.write(f"Risk Area: **{a['Risk Area']}**, Task: **{a['Task']}**, Human Factor: **{a['Human Factor']}**")
        st.write(f"Precursores identificados nos 30 dias anteriores: **{a['Qtd Precursores']}**")
        st.dataframe(a["Detalhes"].sort_values("Date Occurred"))
else:
    st.success("Nenhum precursor identificado nos 30 dias anteriores aos incidentes.")

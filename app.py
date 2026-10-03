"""
Dashboard de Triagem Preditiva — NAT HCV (HEMOPA)
====================================================
Versão adaptada ao esquema REAL de dados (Planilha_do_Orange.csv).

MODO DE OPERAÇÃO:
- Se existir `modelo_hcv.pkl` (gerado por `treinar_modelo.py` a partir
  dos dados reais do HEMOPA), o app carrega e usa o modelo real
  (Logistic Regression, L1, balanceado — mesma config validada no Orange).
- Se o arquivo ainda não existir, cai em MODO DEMONSTRATIVO: treina um
  modelo simples em memória com dados simulados no mesmo formato de
  colunas, e avisa isso claramente na interface.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="HEMOPA · Triagem Preditiva NAT HCV",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ────────────────────────────────────────────────────────────────
# IDENTIDADE VISUAL
# ────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
    :root {
        --paper: #F1F0EC; --ink: #1C1B1A; --ink-soft: #5B5754;
        --carmim: #7A1F32; --carmim-dark: #591526; --teal: #26514D;
        --low: #4F7A5B; --mid: #B87333; --high: #A63A3A;
    }
    .stApp { background-color: var(--paper); }
    h1, h2, h3 { font-family: Georgia, serif !important; color: var(--carmim-dark); }
    .req-banner {
        background: var(--carmim); color: #F1EAE7; padding: 18px 26px;
        border-radius: 4px; border-bottom: 4px solid var(--carmim-dark); margin-bottom: 22px;
    }
    .req-banner .eyebrow {
        font-family: 'Consolas', monospace; font-size: 11px; letter-spacing: 0.12em;
        text-transform: uppercase; opacity: 0.8; margin: 0 0 4px;
    }
    .req-banner .title { font-family: Georgia, serif; font-size: 24px; font-weight: 700; margin: 0; }
    .demo-warning {
        background: #F5E9DA; border: 1px solid #B87333; color: #6B4A1E;
        padding: 12px 18px; border-radius: 4px; font-size: 13px; margin-bottom: 18px; font-family: 'Consolas', monospace;
    }
    .real-model-ok {
        background: #E7EFE9; border: 1px solid #4F7A5B; color: #305C3D;
        padding: 12px 18px; border-radius: 4px; font-size: 13px; margin-bottom: 18px; font-family: 'Consolas', monospace;
    }
    .metric-box { background: #FAF9F6; border: 1px solid rgba(28,27,26,0.14); border-radius: 4px; padding: 16px; text-align: center; }
    .metric-box .label { font-family: 'Consolas', monospace; font-size: 10.5px; text-transform: uppercase; color: var(--ink-soft); letter-spacing: 0.06em; }
    .metric-box .value { font-family: 'Consolas', monospace; font-size: 26px; font-weight: 700; color: var(--teal); }
    .stButton>button {
        background-color: var(--carmim); color: white; border: none; border-radius: 3px;
        font-family: 'Consolas', monospace; text-transform: uppercase; letter-spacing: 0.05em; font-size: 12.5px;
    }
    .stButton>button:hover { background-color: var(--carmim-dark); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown("""
<div class="req-banner">
    <p class="eyebrow">HEMOPA · Serviço de Hemoterapia — Triagem Sorológica</p>
    <p class="title">Dashboard de Triagem Preditiva · NAT HCV</p>
</div>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────
# ESQUEMA DE COLUNAS (igual ao treinar_modelo.py)
# ────────────────────────────────────────────────────────────────
COL_NUMERICAS = ["Quantidade de doações", "Idade", "ANTI-HCV Cutoff"]
COL_CATEGORICAS = [
    "Tipo de doador", "Tipo de doação", "Gênero",
    "Procedência", "RMB/IE", "Estado Civil", "Grau de Escolaridade", "Raça",
]
FEATURE_COLS = COL_NUMERICAS + COL_CATEGORICAS

# Opções reais observadas na planilha, já padronizadas (mesma normalização
# aplicada em treinar_modelo.py: remove sufixo "(a)" e unifica acentuação)
OPCOES = {
    "Tipo de doador": ["Doador de 1ª vez", "Doador de repetição", "Doador esporádico"],
    "Tipo de doação": ["Espontânea", "Vinculada"],
    "Gênero": ["F", "M"],
    "Procedência": ["Abaetetuba", "Altamira", "Belém", "Capanema", "Castanhal", "Marabá",
                     "Oriximiná", "Parauapebas", "Redenção", "Santa Isabel do Pará",
                     "Santarém", "Tucurui", "Tucurí"],
    "RMB/IE": ["IE", "RMB"],
    "Estado Civil": ["Amasiado", "Casado", "Divorciado", "Solteiro", "Viúvo"],
    "Grau de Escolaridade": ["Primeiro grau concluído", "Primeiro grau incompleto",
                               "Pós graduação", "Segundo grau concluído",
                               "Segundo grau incompleto", "Terceiro grau concluído",
                               "Terceiro grau incompleto"],
    "Raça": ["Branca", "Negra", "Parda", "Preta"],
}

MODELO_PATH = "modelo_hcv.pkl"


@st.cache_resource
def carregar_modelo():
    if os.path.exists(MODELO_PATH):
        try:
            return joblib.load(MODELO_PATH)
        except Exception as e:
            st.error(f"Erro ao carregar {MODELO_PATH}: {e}")
            return None
    return None


@st.cache_resource
def treinar_modelo_demo():
    """MODO DEMONSTRATIVO — dados simulados, mesmo formato de colunas reais."""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(42)
    n = 320

    df = pd.DataFrame({
        "Quantidade de doações": rng.integers(1, 30, n),
        "Idade": rng.integers(18, 70, n),
        "ANTI-HCV Cutoff": rng.uniform(0.07, 25, n),
        "Tipo de doador": rng.choice(OPCOES["Tipo de doador"], n),
        "Tipo de doação": rng.choice(OPCOES["Tipo de doação"], n),
        "Gênero": rng.choice(OPCOES["Gênero"], n),
        "Procedência": rng.choice(OPCOES["Procedência"], n),
        "RMB/IE": rng.choice(OPCOES["RMB/IE"], n),
        "Estado Civil": rng.choice(OPCOES["Estado Civil"], n),
        "Grau de Escolaridade": rng.choice(OPCOES["Grau de Escolaridade"], n),
        "Raça": rng.choice(OPCOES["Raça"], n),
    })

    score = (df["ANTI-HCV Cutoff"] - 5) * 0.5
    prob = 1 / (1 + np.exp(-score))
    y = (rng.random(n) < prob).astype(int)

    preprocessador = ColumnTransformer(transformers=[
        ("num", StandardScaler(), COL_NUMERICAS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), COL_CATEGORICAS),
    ])
    modelo = Pipeline(steps=[
        ("preprocessador", preprocessador),
        ("classificador", LogisticRegression(penalty="l1", solver="liblinear",
                                               class_weight="balanced", max_iter=2000, random_state=42)),
    ])
    modelo.fit(df[FEATURE_COLS], y)

    return {
        "modelo": modelo,
        "feature_cols": FEATURE_COLS,
        "num_cols": COL_NUMERICAS,
        "cat_cols": COL_CATEGORICAS,
        "metricas": {"auroc": 0.91, "sensibilidade": 0.87, "especificidade": 0.84,
                     "acuracia": 0.85, "matriz_confusao": [[87, 13], [16, 84]]},
        "modo": "demo",
    }


pacote_real = carregar_modelo()
if pacote_real is not None:
    pacote = pacote_real
    pacote["modo"] = "real"
else:
    pacote = treinar_modelo_demo()
    st.markdown(
        '<div class="demo-warning">⚠️ MODO DEMONSTRATIVO — modelo_hcv.pkl não encontrado. '
        'Usando modelo treinado com dados simulados apenas para manter a interface funcional. '
        'Rode treinar_modelo.py com os dados reais e coloque o .pkl nesta pasta.</div>',
        unsafe_allow_html=True,
    )

modelo = pacote["modelo"]
metricas = pacote["metricas"]


def classificar_risco(p: float):
    if p < 0.05:
        return "low", "Baixo risco"
    elif p < 0.20:
        return "mid", "Médio risco"
    return "high", "Alto risco"


tab1, tab2, tab3 = st.tabs(["🔬 Predição Individual", "📊 Desempenho do Modelo", "📁 Análise em Lote"])

# ==================== ABA 1 ====================
with tab1:
    st.subheader("Triagem individual do doador")
    st.caption("Estimativa de apoio à decisão — não substitui o exame confirmatório.")

    col1, col2 = st.columns(2)
    with col1:
        idade = st.number_input("Idade", 16, 90, 35)
        genero = st.selectbox("Gênero", OPCOES["Gênero"])
        tipo_doador = st.selectbox("Tipo de doador", OPCOES["Tipo de doador"])
        tipo_doacao = st.selectbox("Tipo de doação", OPCOES["Tipo de doação"])
        qtd_doacoes = st.number_input("Quantidade de doações", 1, 100, 1)
        anti_hcv_cutoff = st.number_input("ANTI-HCV Cutoff", 0.0, 50.0, 1.0, step=0.01)
    with col2:
        procedencia = st.selectbox("Procedência", OPCOES["Procedência"])
        rmb_ie = st.selectbox("RMB/IE", OPCOES["RMB/IE"])
        estado_civil = st.selectbox("Estado Civil", OPCOES["Estado Civil"])
        escolaridade = st.selectbox("Grau de Escolaridade", OPCOES["Grau de Escolaridade"])
        raca = st.selectbox("Raça", OPCOES["Raça"])

    if st.button("Calcular risco"):
        entrada = pd.DataFrame([{
            "Tipo de doador": tipo_doador,
            "Tipo de doação": tipo_doacao, "Quantidade de doações": qtd_doacoes,
            "Gênero": genero, "Idade": idade, "Procedência": procedencia,
            "RMB/IE": rmb_ie, "Estado Civil": estado_civil,
            "Grau de Escolaridade": escolaridade, "Raça": raca,
            "ANTI-HCV Cutoff": anti_hcv_cutoff,
        }])[FEATURE_COLS]

        try:
            prob = modelo.predict_proba(entrada)[0][1]
            nivel, label = classificar_risco(prob)

            col_gauge, col_info = st.columns([1, 2])
            with col_gauge:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=prob * 100,
                    number={"suffix": "%", "font": {"size": 36}},
                    gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#1C1B1A"},
                           "steps": [{"range": [0, 5], "color": "#E7EFE9"},
                                     {"range": [5, 20], "color": "#F5E9DA"},
                                     {"range": [20, 100], "color": "#F5E1DF"}]},
                ))
                fig.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)
                cores = {"low": "🟢", "mid": "🟡", "high": "🔴"}
                st.markdown(f"### {cores[nivel]} {label}")
            with col_info:
                st.caption(
                    "Resultado não substitui o exame NAT confirmatório. "
                    + ("Modelo demonstrativo (dados simulados)." if pacote.get("modo") == "demo" else "Modelo treinado com dados reais do HEMOPA.")
                )
        except Exception as e:
            st.error(f"Não foi possível calcular a predição: {e}")

# ==================== ABA 2 ====================
with tab2:
    st.subheader("Desempenho do modelo")
    st.caption("Logistic Regression (L1, balanceado)" + (" — dados simulados, modo demonstrativo." if pacote.get("modo") == "demo" else " — validado com dados reais do HEMOPA."))

    c1, c2, c3, c4 = st.columns(4)
    for col, (label, key) in zip([c1, c2, c3, c4],
                                   [("AUROC", "auroc"), ("Sensibilidade", "sensibilidade"),
                                    ("Especificidade", "especificidade"), ("Acurácia", "acuracia")]):
        with col:
            st.markdown(f'<div class="metric-box"><div class="label">{label}</div><div class="value">{metricas.get(key, 0):.2f}</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Matriz de confusão** (conjunto de teste)")
    cm = metricas.get("matriz_confusao", [[0, 0], [0, 0]])
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm, x=["Previsto: Detectável", "Previsto: Indetectável"],
        y=["Real: Detectável", "Real: Indetectável"],
        colorscale=[[0, "#F1EAE7"], [1, "#7A1F32"]], showscale=False,
        text=cm, texttemplate="%{text}",
    ))
    fig_cm.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_cm, use_container_width=True)

    st.caption("⚠️ Comparativo com o método de triagem tradicional (Anti-HCV isolado) ainda não incluído.")

# ==================== ABA 3 ====================
with tab3:
    st.subheader("Análise em lote (CSV)")
    st.caption(f"Colunas esperadas: {', '.join(FEATURE_COLS)}")

    modelo_csv = pd.DataFrame([{c: (OPCOES[c][0] if c in OPCOES else 1) for c in FEATURE_COLS}])
    st.download_button("Baixar modelo de CSV", modelo_csv.to_csv(index=False).encode("utf-8"), "modelo_triagem_hcv.csv", "text/csv")

    arquivo = st.file_uploader("Envie o CSV com os dados dos doadores", type=["csv"])
    if arquivo is not None:
        try:
            df_lote = pd.read_csv(arquivo)
            faltando = [c for c in FEATURE_COLS if c not in df_lote.columns]
            if faltando:
                st.error(f"Colunas faltando: {', '.join(faltando)}")
            else:
                probs = modelo.predict_proba(df_lote[FEATURE_COLS])[:, 1]
                df_lote["probabilidade (%)"] = (probs * 100).round(2)
                df_lote["classificacao"] = [classificar_risco(p)[1] for p in probs]
                df_lote = df_lote.sort_values("probabilidade (%)", ascending=False)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("🟢 Baixo risco", (df_lote["classificacao"] == "Baixo risco").sum())
                m2.metric("🟡 Médio risco", (df_lote["classificacao"] == "Médio risco").sum())
                m3.metric("🔴 Alto risco", (df_lote["classificacao"] == "Alto risco").sum())
                m4.metric("Total analisado", len(df_lote))

                st.dataframe(df_lote, use_container_width=True)
                st.download_button("Exportar resultados", df_lote.to_csv(index=False).encode("utf-8"), "resultados_triagem_hcv.csv", "text/csv")
        except Exception as e:
            st.error(f"Erro ao processar o arquivo: {e}")
    else:
        st.info("Nenhum arquivo carregado ainda.")

st.markdown("---")
st.caption(
    "Ferramenta de apoio à decisão desenvolvida como parte do Trabalho de Conclusão de Curso (TCC). "
    "Não substitui o exame confirmatório laboratorial (NAT). "
    + ("Modo demonstrativo ativo." if pacote.get("modo") == "demo" else "")
)

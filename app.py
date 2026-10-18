import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Ignora avisos chatos de versão do scikit-learn
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Configuração da página (deixa o layout largo e bonito)
st.set_page_config(page_title="Triagem TDAH - IA", page_icon="🧠", layout="wide")

# Estilo personalizado (CSS) para deixar os cards mais bonitos
st.markdown("""
    <style>
    .big-font { font-size:20px !important; font-weight: bold;}
    .stButton>button { width: 100%; height: 50px; font-size: 18px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# CARREGAR MODELO
# ==========================================
@st.cache_resource
def carregar_modelo():
    caminho = 'modelos/artefatos.joblib'
    if os.path.exists(caminho):
        return joblib.load(caminho)
    return None

artefatos = carregar_modelo()

if artefatos is None:
    st.error("⚠️ Erro Crítico: Arquivo 'artefatos.joblib' não encontrado. Verifique a pasta 'modelos' no GitHub.")
    st.stop()

# ==========================================
# GERENCIAR HISTÓRICO (TABELA)
# ==========================================
# Guarda os dados na memória da sessão para a tabela
if 'historico' not in st.session_state:
    st.session_state.historico = pd.DataFrame(columns=['Idade', 'Gênero', 'QI Verbal', 'Resultado', 'Probabilidade'])

# ==========================================
# INTERFACE PRINCIPAL
# ==========================================
st.title("🧠 Simulador de Triagem de TDAH")
st.markdown("Preencha os dados clínicos do paciente abaixo. A Inteligência Artificial calculará o risco em tempo real.")

# Cria o formulário
with st.form("form_triagem"):
    st.subheader("1. Dados do Paciente")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Idade (Anos)", min_value=5.0, max_value=99.0, value=10.0, step=1.0)
        gender = st.selectbox("Gênero", options=[1.0, 0.0], format_func=lambda x: "Masculino (1)" if x == 1.0 else "Feminino (0)")
        handedness = st.selectbox("Mão Dominante", options=["RIGHT", "LEFT", "AMBIDEXTROUS"])
        
    with col2:
        iq_measure = st.selectbox("Medida de QI", options=[1.0, 2.0, 3.0, 4.0, 5.0])
        verbal_iq = st.number_input("QI Verbal (Deixe 0 se não avaliado)", min_value=0.0, max_value=160.0, value=100.0, step=1.0)
        performance_iq = st.number_input("QI de Execução (Deixe 0 se não avaliado)", min_value=0.0, max_value=160.0, value=100.0, step=1.0)

    with col3:
        full4_iq = st.number_input("QI Total 4 (Deixe 0 se não avaliado)", min_value=0.0, max_value=160.0, value=100.0, step=1.0)
        med_status = st.selectbox("Status de Medicação", options=["OFF", "ON", "PENDING"])
        
    st.divider()
    submitted = st.form_submit_button("🔍 Calcular Probabilidade de TDAH")

# ==========================================
# LÓGICA DE PREDIÇÃO
# ==========================================
if submitted:
    # Se o usuário digitou 0, substitui por NaN (para o imputer preencher com a mediana, igual no treino)
    v_iq = np.nan if verbal_iq == 0 else verbal_iq
    p_iq = np.nan if performance_iq == 0 else performance_iq
    f4_iq = np.nan if full4_iq == 0 else full4_iq

    # Cria o "paciente" no formato do treino
    dados_paciente = pd.DataFrame([{
        'Site': 1.0,
        'Gender': gender,
        'Age': age,
        'Handedness': handedness,
        'IQ Measure': iq_measure,
        'Verbal IQ': v_iq,
        'Performance IQ': p_iq,
        'Full4 IQ': f4_iq,
        'Med Status': med_status
    }])

    try:
        # 1. Pré-processamento (Mesmo pipeline do treino)
        df = dados_paciente.copy()
        
        for col in df.columns:
            if col in artefatos['categorical_cols'] or df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip().str.upper()

        df_enc = pd.get_dummies(df, drop_first=True)
        df_enc = df_enc.reindex(columns=artefatos['colunas_treino'], fill_value=0)
        
        # Transformações
        X_imp = artefatos['imputer'].transform(df_enc)
        X_scaled = artefatos['scaler'].transform(X_imp)

        # 2. Predição
        modelo = artefatos['modelo_rf']
        predicao = modelo.predict(X_imp)[0]
        probabilidade = modelo.predict_proba(X_imp)[0][1]

        # 3. Exibição do Resultado
        st.divider()
        st.subheader("2. Resultado da Análise Clínica")
        
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            if predicao == 1:
                st.error("### ⚠️ ALTO RISCO\n**Sugestão:** Encaminhamento para avaliação especializada.")
            else:
                st.success("### ✅ BAIXO RISCO\n**Sugestão:** Manter acompanhamento de rotina.")
                
        with res_col2:
            st.markdown(f"<p class='big-font'>Probabilidade estimada pelo modelo: {probabilidade:.1%}</p>", unsafe_allow_html=True)
            st.progress(float(probabilidade))
            
        # 4. Salvar no Histórico
        novo_registro = pd.DataFrame([{
            'Idade': f"{age:.1f} anos",
            'Gênero': "Masculino" if gender == 1.0 else "Feminino",
            'QI Verbal': verbal_iq if verbal_iq > 0 else "Não avaliado",
            'Resultado': "Positivo" if predicao == 1 else "Negativo",
            'Probabilidade': f"{probabilidade:.1%}"
        }])
        st.session_state.historico = pd.concat([st.session_state.historico, novo_registro], ignore_index=True)

    except Exception as e:
        st.error(f"Erro ao processar a predição: {e}")

# ==========================================
# EXIBIR TABELA DE HISTÓRICO
# ==========================================
st.divider()
st.subheader("📋 Histórico de Pacientes Avaliados nesta Sessão")
if len(st.session_state.historico) > 0:
    # Colorir a tabela com base no resultado
    def color_resultado(val):
        color = '#ff4b4b' if val == 'Positivo' else '#09ab3b'
        return f'color: {color}; font-weight: bold'
    
    st.dataframe(
        st.session_state.historico.style.applymap(color_resultado, subset=['Resultado']),
        use_container_width=True,
        hide_index=True
    )
    
    if st.button("🗑️ Limpar Histórico"):
        st.session_state.historico = pd.DataFrame(columns=st.session_state.historico.columns)
        st.rerun()
else:
    st.info("Nenhum paciente avaliado ainda. Preencha o formulário acima.")

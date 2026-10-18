import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Ignora avisos de versão do scikit-learn na hora de carregar o modelo
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Configuração da página
st.set_page_config(page_title="Triagem TDAH", page_icon="🧠", layout="wide")

# Carregar modelo (com cache para não carregar toda hora)
@st.cache_resource
def carregar_modelo():
    caminho = 'modelos/artefatos.joblib'
    if os.path.exists(caminho):
        return joblib.load(caminho)
    return None

artefatos = carregar_modelo()

if artefatos is None:
    st.error("⚠️ Arquivo de modelo não encontrado! Certifique-se de que a pasta 'modelos' e o arquivo 'artefatos.joblib' estão na raiz do projeto no GitHub.")
    st.stop()

# Título
st.title("🧠 Simulador de Triagem de TDAH")
st.markdown("Preencha os dados clínicos do paciente abaixo para que a Inteligência Artificial calcule a probabilidade de TDAH.")

st.divider()

# Formulário de entrada de dados
with st.form("form_paciente"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Dados Básicos")
        age = st.number_input("Idade (Age)", min_value=5.0, max_value=30.0, value=10.0, step=1.0)
        gender = st.selectbox("Gênero (Gender)", options=[0.0, 1.0], format_func=lambda x: "Feminino (0)" if x == 0.0 else "Masculino (1)")
        handedness = st.selectbox("Mão Dominante (Handedness)", options=["RIGHT", "LEFT", "AMBIDEXTROUS", "0.5", "1.0", "-999"])
        
    with col2:
        st.subheader("Testes de QI")
        iq_measure = st.selectbox("Medida de QI (IQ Measure)", options=[1.0, 2.0, 3.0, 4.0, 5.0, -999.0])
        verbal_iq = st.number_input("QI Verbal (Verbal IQ)", min_value=-999.0, max_value=160.0, value=-999.0, step=1.0)
        performance_iq = st.number_input("QI de Execução (Performance IQ)", min_value=-999.0, max_value=160.0, value=-999.0, step=1.0)

    with col3:
        st.subheader("Outros Dados")
        full4_iq = st.number_input("QI Total 4 (Full4 IQ)", min_value=-999.0, max_value=160.0, value=-999.0, step=1.0)
        med_status = st.selectbox("Status de Medicação (Med Status)", options=["OFF", "ON", "PENDING", "-999"])
        
    # Botão de submissão
    submitted = st.form_submit_button("🔍 Calcular Probabilidade de TDAH", use_container_width=True)

# Lógica de predição
if submitted:
    # Criar um DataFrame com os dados inseridos (os nomes devem bater com o treino)
    dados_paciente = pd.DataFrame([{
        'Site': 1.0, # O modelo exige a coluna Site, usamos 1.0 como padrão
        'Gender': gender,
        'Age': age,
        'Handedness': handedness,
        'IQ Measure': iq_measure,
        'Verbal IQ': verbal_iq,
        'Performance IQ': performance_iq,
        'Full4 IQ': full4_iq,
        'Med Status': med_status
    }])

    try:
        # 1. Aplicar o MESMO pipeline de pré-processamento do treino
        df = dados_paciente.copy()
        
        # Padronizar texto
        for col in df.columns:
            if col in artefatos['categorical_cols'] or df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip().str.upper()

        # One-Hot Encoding
        df_enc = pd.get_dummies(df, drop_first=True)
        
        # Alinhar colunas com o treino (preenche as que faltam com 0)
        df_enc = df_enc.reindex(columns=artefatos['colunas_treino'], fill_value=0)

        # Imputar valores ausentes e escalonar
        X_imp = artefatos['imputer'].transform(df_enc)
        X_scaled = artefatos['scaler'].transform(X_imp)

        # 2. Fazer a predição usando o Random Forest
        modelo = artefatos['modelo_rf']
        predicao = modelo.predict(X_imp)[0]
        probabilidade = modelo.predict_proba(X_imp)[0][1]

        # 3. Mostrar o resultado na tela
        st.divider()
        st.subheader("Resultado da Análise")
        
        if predicao == 1:
            st.error(f"### ⚠️ ALTO RISCO DE TDAH")
            st.markdown(f"O modelo estimou uma probabilidade de **{probabilidade:.1%}** de o paciente ter TDAH.")
        else:
            st.success(f"### ✅ BAIXO RISCO DE TDAH")
            st.markdown(f"O modelo estimou uma probabilidade de apenas **{probabilidade:.1%}** de o paciente ter TDAH.")
            
        # Mostrar barra de progresso visual
        st.progress(float(probabilidade))
        
    except Exception as e:
        st.error(f"Erro ao processar a predição: {e}")
        st.write("Verifique se as colunas de entrada correspondem exatamente às colunas do treino original.")

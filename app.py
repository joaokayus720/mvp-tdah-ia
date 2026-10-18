import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração da página
st.set_page_config(page_title="Dashboard TDAH", page_icon="🧠", layout="wide")

st.title("🧠 Dashboard de Predição de TDAH")
st.markdown("""
Este é o **MVP** do projeto. Faça o upload do arquivo CSV com as predições do modelo para visualizar os resultados.
""")

# Componente de upload
st.sidebar.header("1. Upload dos Dados")
arquivo_carregado = st.sidebar.file_uploader("Escolha o arquivo CSV (ex: resultados_modelo.csv)", type="csv")

if arquivo_carregado is not None:
    # Carregar os dados
    df = pd.read_csv(arquivo_carregado)
    
    st.sidebar.success("Arquivo carregado com sucesso!")
    
    # Layout de colunas para os gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Resumo Geral")
        total_pacientes = len(df)
        total_positivos = df['Previsao_TDAH'].sum()
        total_negativos = total_pacientes - total_positivos
        
        st.metric("Total de Pacientes", total_pacientes)
        st.metric("Pacientes com TDAH (Previsto)", total_positivos)
        st.metric("Pacientes Sem TDAH (Previsto)", total_negativos)
        
    with col2:
        st.subheader("📈 Distribuição das Predições")
        # Criando gráfico de pizza
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = ['Sem TDAH', 'Com TDAH']
        sizes = [total_negativos, total_positivos]
        colors = ['#66b3ff', '#ff9999']
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)

    # Tabela de dados
    st.subheader("📋 Tabela de Dados Detalhada")
    st.markdown("Abaixo estão todos os dados do arquivo, incluindo a probabilidade calculada pelo modelo.")
    
    # Filtrando apenas as colunas mais importantes para não poluir a tela
    colunas_exibir = ['Site', 'Age', 'Gender', 'Full4 IQ', 'Previsao_TDAH', 'Probabilidade']
    # Verifica se as colunas existem antes de filtrar
    colunas_exibir = [col for col in colunas_exibir if col in df.columns]
    
    # Exibindo a tabela com destaque para a coluna de probabilidade
    st.dataframe(
        df[colunas_exibir].style.highlight_max(axis=0, subset=['Probabilidade'], color='lightcoral'),
        use_container_width=True
    )

    # Botão para baixar os dados
    st.sidebar.header("2. Download")
    csv = df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        label="Baixar Tabela de Resultados",
        data=csv,
        file_name='resultados_finais_adhd.csv',
        mime='text/csv',
    )
else:
    st.info("👈 Por favor, faça o upload do arquivo CSV na barra lateral para começar.")
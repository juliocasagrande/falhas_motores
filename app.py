import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Carregando o modelo e o scaler
@st.cache_resource
def load_model():
    with open('final_motor_failure_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return model, scaler

model, scaler = load_model()

# Definindo as features utilizadas no modelo
features = [
    'ambient', 'coolant', 'u_d', 'u_q', 'motor_speed', 'torque',
    'i_d', 'i_q', 'pm', 'stator_yoke', 'stator_tooth', 'stator_winding',
    'pm_diff', 'pm_rate'
]

# Título da aplicação
st.title("Previsão de Falhas em Motores Elétricos")

# Explicação do aplicativo
st.write("""
Esta aplicação utiliza um modelo de Machine Learning para prever possíveis falhas em motores elétricos.
Insira os parâmetros operacionais do motor abaixo para obter uma previsão.
""")

# Criando o formulário de entrada de dados
st.sidebar.header("Entrada de Dados")

# Função para coletar os dados de entrada do usuário
def get_user_input():
    ambient = st.sidebar.slider("Temperatura Ambiente (°C)", 20.0, 40.0, 28.5)
    coolant = st.sidebar.slider("Temperatura do Líquido de Arrefecimento (°C)", 20.0, 80.0, 35.0)
    u_d = st.sidebar.slider("Tensão Direta (u_d)", -300.0, 300.0, -220.0)
    u_q = st.sidebar.slider("Tensão Quadratura (u_q)", -300.0, 300.0, 160.0)
    motor_speed = st.sidebar.slider("Velocidade do Motor (RPM)", 0.0, 4000.0, 3200.0)
    torque = st.sidebar.slider("Torque (Nm)", 0.0, 200.0, 75.0)
    i_d = st.sidebar.slider("Corrente Direta (i_d)", -200.0, 200.0, -110.0)
    i_q = st.sidebar.slider("Corrente Quadratura (i_q)", -200.0, 200.0, 110.0)
    pm = st.sidebar.slider("Temperatura do Ímã Permanente (°C)", 20.0, 200.0, 155.0)
    stator_yoke = st.sidebar.slider("Temperatura do Estator (Yoke) (°C)", 20.0, 200.0, 140.0)
    stator_tooth = st.sidebar.slider("Temperatura do Estator (Dente) (°C)", 20.0, 200.0, 145.0)
    stator_winding = st.sidebar.slider("Temperatura do Enrolamento do Estator (°C)", 20.0, 200.0, 150.0)
    timestamp = st.sidebar.number_input("Timestamp", min_value=1, value=1633017601)

    # Calculando as features de tendência
    pm_diff = 0  # valor inicial (pode ser atualizado com valores reais)
    pm_rate = 0  # valor inicial (pode ser atualizado com valores reais)

    # Criando um dicionário com as entradas
    user_data = {
        'ambient': ambient,
        'coolant': coolant,
        'u_d': u_d,
        'u_q': u_q,
        'motor_speed': motor_speed,
        'torque': torque,
        'i_d': i_d,
        'i_q': i_q,
        'pm': pm,
        'stator_yoke': stator_yoke,
        'stator_tooth': stator_tooth,
        'stator_winding': stator_winding,
        'pm_diff': pm_diff,
        'pm_rate': pm_rate,
        'timestamp': timestamp
    }

    return user_data

# Coletando os dados de entrada
input_data = get_user_input()

# Exibindo os dados de entrada fornecidos
st.subheader("Dados de Entrada Fornecidos:")
st.write(pd.DataFrame([input_data]))

# Preparando os dados para a previsão
input_df = pd.DataFrame([input_data])
X_new = input_df[features]

# Padronizando os dados com o scaler carregado
X_new_scaled = scaler.transform(X_new)

# Fazendo a previsão
prediction = model.predict(X_new_scaled)[0]

# Exibindo o resultado da previsão
st.subheader("Resultado da Previsão:")
if prediction == 1:
    st.error("⚠️ Alerta! Possível falha detectada no motor.")
else:
    st.success("✅ O motor está operando normalmente.")

# Informação adicional
st.write("""
A previsão é baseada nos dados operacionais fornecidos e no modelo de Machine Learning treinado.
Caso deseje uma previsão mais precisa, insira dados reais e atualizados do motor.
""")

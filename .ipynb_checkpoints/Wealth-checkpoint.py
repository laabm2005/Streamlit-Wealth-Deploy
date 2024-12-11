import pandas as pd

import numpy as np

from pypfopt.efficient_frontier import EfficientFrontier

from pypfopt import risk_models

from pypfopt import expected_returns

import plotly.graph_objects as go # Import Plotly

import matplotlib.pyplot as plt


from pypfopt import plotting  # Importa o módulo plotting


# --- Etapa 1: Ler o arquivo Excel ---

try:

    dados = pd.read_excel("DadosBloombergWealth_2022_12_05.xlsx", index_col=0, parse_dates=True)

except FileNotFoundError:

    print("Erro: Arquivo 'DadosBloombergWealth_2022_12_05.xlsx' não encontrado.")

    exit()


# --- Etapa 2: Calcular retornos ---

retornos = dados.pct_change().dropna()


# --- Etapa 3: Definir parâmetros da carteira ---

retorno_desejado = float(input("Digite o retorno desejado da carteira (em decimal): "))

volatilidade_maxima = float(input("Digite a volatilidade máxima desejada da carteira (em decimal): "))

risk_free_rate = float(input("Digite o risk-free rate (em decimal): "))


# --- Etapa 4: Calcular retornos esperados e matriz de covariância ---

#mu = expected_returns.mean_historical_return(dados)

#mu = expected_returns.mean_historical_return(dados) 

custom_mu_data = [['IMAB5+', '2Y', 'IMAB5', 'IBOV', '5Y', 'IHF', 'IFIX', 'IHFA'], [0.1155, 0.106, 0.1135, 0.05, 0.114, 0.1741, 0.097, 0.107]]

# Creating a pandas Series from custom data

mu = pd.Series(custom_mu_data[1], index=custom_mu_data[0]) 

S = risk_models.sample_cov(dados)

print(mu)

# --- Etapa 5: Otimizar a carteira ---



ef = EfficientFrontier(mu, S)

pesos_brutos = ef.min_volatility()  # Calcula os pesos brutos

pesos_otimizados = ef.clean_weights()  # Limpa os pesos


# --- Etapa 6: Calcular Sharpe Ratio da carteira otimizada ---

retorno_otimizado = mu.dot(pd.Series(pesos_otimizados))  # Retorno da carteira otimizada

volatilidade_otimizada = np.sqrt(np.array(list(pesos_otimizados.values())).T @ S @ np.array(list(pesos_otimizados.values())))

sharpe_otimizado = (retorno_otimizado - risk_free_rate) / volatilidade_otimizada  # Sharpe otimizado


#- Etapa 7: Gerar carteiras e verificar condições ---

num_carteiras = 5000

carteiras_alternativas = []


for _ in range(num_carteiras):

    # Gerar pesos aleatórios próximos aos pesos otimizados

    pesos_aleatorios = {k: v + np.random.normal(0, 0.02) for k, v in pesos_otimizados.items()}

    pesos_aleatorios = {k: max(0, v) for k, v in pesos_aleatorios.items()}

    pesos_aleatorios = {k: v / sum(pesos_aleatorios.values()) for k, v in pesos_aleatorios.items()}
    

    # Verificar diferença de alocação

    diferenca_alocacao = [abs(pesos_aleatorios[k] - pesos_otimizados[k]) for k in pesos_otimizados]

    if all(diff <= 0.02 for diff in diferenca_alocacao):
        

        # Calcular Sharpe Ratio da carteira atual

        retorno_carteira = mu.dot(pd.Series(pesos_aleatorios))

        volatilidade_carteira = np.sqrt(pd.Series(pesos_aleatorios).T @ S @ pd.Series(pesos_aleatorios))
        

        # Verificar volatilidade máxima

        if volatilidade_carteira <= volatilidade_maxima:

            sharpe_ratio = (retorno_carteira - risk_free_rate) / volatilidade_carteira


            # Verificar Sharpe

            if sharpe_ratio >= sharpe_otimizado - 0.1:

                carteiras_alternativas.append(pesos_aleatorios)


# --- Etapa 8: Criar DataFrame com resultados ---

if carteiras_alternativas:

    df_excelente = pd.DataFrame(carteiras_alternativas)

    df_excelente.index = range(1, len(carteiras_alternativas) + 1)  # Numerar as carteiras alternativas


    # Calcular Sharpe Ratio e Volatilidade para cada carteira alternativa

    sharpes_alternativas = []

    volatilidades_alternativas = []

    for i in range(len(carteiras_alternativas)):

        pesos_aleatorios = carteiras_alternativas[i]  # Pegar os pesos da carteira alternativa atual

        retorno_carteira = mu.dot(pd.Series(pesos_aleatorios))  # Calcular o retorno

        volatilidade_carteira = np.sqrt(pd.Series(pesos_aleatorios).T @ S @ pd.Series(pesos_aleatorios))  # Calcular a volatilidade

        sharpe_ratio = (retorno_carteira - risk_free_rate) / volatilidade_carteira  # Calcular o Sharpe ratio

        sharpes_alternativas.append(sharpe_ratio)  # Adicionar o Sharpe ratio à lista

        volatilidades_alternativas.append(volatilidade_carteira)  # Adicionar a volatilidade à lista


    df_excelente["Sharpe Ratio"] = sharpes_alternativas  # Adicionar a coluna Sharpe Ratio ao DataFrame

    df_excelente["Diferença Sharpe"] = df_excelente["Sharpe Ratio"] - sharpe_otimizado  # Adicionar a coluna Diferença Sharpe

    df_excelente["Volatilidade"] = volatilidades_alternativas  # Adicionar a coluna Volatilidade ao DataFrame

    df_excelente['Retorno']=retorno_carteira

    df_excelente.loc["Carteira Otimizada"] = pesos_otimizados  # Adicionar a carteira otimizada

    df_excelente.loc["Carteira Otimizada", "Sharpe Ratio"] = sharpe_otimizado  # Adicionar o Sharpe otimizado

    df_excelente.loc["Carteira Otimizada", "Diferença Sharpe"] = 0  # Diferença Sharpe para a carteira otimizada é 0

    df_excelente.loc["Carteira Otimizada", "Volatilidade"] = volatilidade_otimizada  # Adicionar a volatilidade da carteira otimizada

    df_excelente.loc["Carteira Otimizada", "Retorno"] = retorno_otimizado


    print("\nCarteiras Excelentes:")

    pd.set_option('display.max_columns', None)

    print(df_excelente)

else:

    print("\nNenhuma carteira excelente encontrada.")
    
import streamlit as st


st.set_page_config(layout="wide", page_title="Carteira de Investimentos")


# Assuming you have these functions defined:

# def calculate_portfolio():

#     # ... (Your portfolio calculation logic) ...

#     return df_excelente, plot_data 


# def plot_efficient_frontier(plot_data):

#     # ... (Your efficient frontier plotting logic) ...



# --- Streamlit app ---

st.dataframe(df_excelente) 


# Create a new DataFrame for the scatter plot

scatter_data = pd.DataFrame({

    'IBOV': df_excelente["IBOV"],

    'IMAB5': df_excelente["IMAB5"],

    'IMAB5+': df_excelente["IMAB5+"],

    'IHFA': df_excelente["IHFA"],

    'IHF': df_excelente["IHF"]

})


# Display the scatter plot using Streamlit

st.scatter_chart(scatter_data)





# Create a new DataFrame for the scatter plot FRONTEIRA EFICIENTE

scatter_data1 = pd.DataFrame({

    'Retorno': df_excelente["Retorno"],

    'Volatilidade': df_excelente["Volatilidade"],

    })


# Display the scatter plot using Streamlit

#st.scatter_chart(scatter_data1)




# Crie o gráfico de dispersão especificando os eixos x e y

st.scatter_chart(scatter_data1, x="Volatilidade", y="Retorno")



# --- Gerar pontos aleatórios para carteiras ---

num_carteiras_aleatorias = 50000

retornos_aleatorios = []

volatilidades_aleatorias = []


for _ in range(num_carteiras_aleatorias):

    pesos = np.random.random(len(mu))  # Pesos aleatórios

    pesos /= np.sum(pesos)  # Normalizar pesos


    # Criar pd.Series com índice correspondente a mu

    pesos_series = pd.Series(pesos, index=mu.index)  


    retorno_carteira = mu.dot(pesos_series) 

    volatilidade_carteira = np.sqrt(np.array(pesos_series).T @ S @ np.array(pesos_series))

    retornos_aleatorios.append(retorno_carteira)

    volatilidades_aleatorias.append(volatilidade_carteira)


# --- Criar DataFrame para carteiras aleatórias ---

carteiras_aleatorias_df = pd.DataFrame({

    'Retorno': retornos_aleatorios,

    'Volatilidade': volatilidades_aleatorias

})


# --- Plotar usando Streamlit ---

# Assumindo que df_excelente já está definido

# ... (seu código para gerar df_excelente) ...


# Criar gráfico de dispersão com Plotly

fig = go.Figure()

fig.add_trace(go.Scatter(x=carteiras_aleatorias_df['Volatilidade'], y=carteiras_aleatorias_df['Retorno'], mode='markers', name='Carteiras Aleatórias', marker=dict(color='lightgray')))

fig.add_trace(go.Scatter(x=df_excelente['Volatilidade'], y=df_excelente['Retorno'], mode='markers', name='Carteiras Excelentes', marker=dict(color='blue', size=8)))


fig.update_layout(title='Carteiras Aleatórias vs. Excelentes', xaxis_title='Volatilidade', yaxis_title='Retorno')


st.plotly_chart(fig) # Exibir o gráfico no Streamlit



# --- Calcular a carteira ótima ---

ef_max_sharpe = EfficientFrontier(mu, S)  # Recalcular a fronteira eficiente

pesos_otimizados = ef_max_sharpe.max_sharpe(risk_free_rate=risk_free_rate)  # Pesos da carteira ótima

#retorno_otimizado = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)[0] # Retorno da carteira ótima

ef_volot=EfficientFrontier(mu,S)

pesos_volot = ef_volot.max_sharpe() # por exemplo

volatilidade_otimizada = ef_volot.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)[1] # Volatilidade da carteira ótima

ef_efret=EfficientFrontier(mu,S)

pesos_otimizados_retorno_dado=ef_efret.efficient_return(target_return=retorno_desejado)#Pesos da carteira otima para retorno dado


# --- Adicionar a CAL ao gráfico ---

fig.add_trace(go.Scatter(

    x=[0, volatilidade_otimizada],  # Pontos x: risk-free (0) e volatilidade ótima

    y=[risk_free_rate, retorno_otimizado],  # Pontos y: risk-free e retorno ótimo

    mode='lines',

    name='CAL',

    line=dict(color='red', width=2)  # Destacar a linha CAL
))


# --- Destacar a carteira ótima no gráfico ---

fig.add_trace(go.Scatter(

    x=[volatilidade_otimizada],

    y=[retorno_otimizado],

    mode='markers',

    name='Carteira Ótima',

    marker=dict(color='red', size=10)  # Destacar a carteira ótima
))


# --- Imprimir o novo gráfico completo ---

st.plotly_chart(fig)  # Imprime o gráfico com a CAL e a carteira ótima
            




#Fornteira Efciciente


ef_fe = EfficientFrontier(mu, S, weight_bounds=(None, None))
ef_fe.add_constraint(lambda w: w[0] >= 0.2)
ef_fe.add_constraint(lambda w: w[2] == 0.15)
ef_fe.add_constraint(lambda w: w[3] + w[4] <= 0.10)


fig, ax = plt.subplots()
plotting.plot_efficient_frontier(ef_fe, ax=ax, show_assets=True)

plt.show()
import io

# Crie a figura e o gráfico
fig, ax = plt.subplots()
# Exiba a figura no Streamlit
plt.plot()
# ... (seu código para plotar o gráfico) ...

# Salve a figura em um buffer de bytes
buf = io.BytesIO()

plt.savefig(buf, format="jpeg")
buf.seek(0)

# Exiba a figura no Streamlit

st.image(buf, caption='Meu Gráfico', use_column_width=True)
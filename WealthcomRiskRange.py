import pandas as pd
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import plotly.graph_objects as go # Import Plotly
import matplotlib.pyplot as plt
from pypfopt import plotting
import streamlit as st

# --- Etapa 1: Ler o arquivo Excel ---


st.set_page_config(
    page_title="Pesos",
    page_icon="🏃🏼",
    layout="wide"
)




try:
    dados = pd.read_excel("DadosBloombergWealth_2022_12_05.xlsx", index_col=0, parse_dates=True)
except FileNotFoundError:
    print("Erro: Arquivo 'DadosBloombergWealth_2022_12_05.xlsx' não encontrado.")
    exit()

# --- Etapa 2: Calcular retornos ---
retornos = dados.pct_change().dropna()

# --- Etapa 3: Definir parâmetros da carteira ---
#retorno_desejado = float(input("Digite o retorno desejado da carteira (em decimal): "))
#volatilidade_maxima = float(input("Digite a volatilidade máxima desejada da carteira (em decimal): "))
#volatilidade_maxima=0.12
#risk_free_rate = float(input("Digite o risk-free rate (em decimal): "))
#risk_free_rate=0.10

#import streamlit as st

retorno_desejado = st.slider("Qual retorno vc quer?", 0.12 , 0.15, 0.12)
st.write("Meu retorno desejado é", retorno_desejado, "")

risk_free_rate = st.slider("Qual risk free vc quer?", 0.08 , 0.12, 0.10)
st.write("A taxa de juros SELIC hoje está quanto?", risk_free_rate, "")

volatilidade_maxima = st.slider("Qual volatilidade maxima vc quer?", 0.08 , 0.20, 0.12)
st.write("Minha volatilidade máxima é", volatilidade_maxima, "")

# --- Etapa 4: Calcular retornos esperados e matriz de covariância ---

custom_mu_data = [['IMAB5+', '2Y', 'IMAB5', 'IBOV', '5Y', 'IHF', 'IFIX', 'IHFA'], [0.1155, 0.106, 0.1135, 0.05, 0.114, 0.1741, 0.097, 0.107]]
# Creating a pandas Series from custom data
mu = pd.Series(custom_mu_data[1], index=custom_mu_data[0]) 
S = risk_models.sample_cov(dados)
print(mu)
# --- Etapa 5: Otimizar a carteira para Retorno desejado---


ef = EfficientFrontier(mu, S)

pesos_brutos_ret = ef.efficient_return(target_return=retorno_desejado)  # Calcula os pesos brutos

pesos_otimizados_ret = pesos_brutos_ret  # Limpa os pesos


# --- Etapa 6: Calcular Sharpe Ratio da carteira otimizada para o retorno dado ---

retorno_otimizado_ret = mu.dot(pd.Series(pesos_otimizados_ret))  # Retorno da carteira otimizada

volatilidade_otimizada_ret = np.sqrt(np.array(list(pesos_otimizados_ret.values())).T @ S @ np.array(list(pesos_otimizados_ret.values())))

sharpe_otimizado = (retorno_otimizado_ret - risk_free_rate) / volatilidade_otimizada_ret  # Sharpe otimizado

print(pesos_otimizados_ret)

print(retorno_otimizado_ret)

print(volatilidade_otimizada_ret)

# --- Etapa 7: Otimizar a carteira para Minima Vol---


ef_minvol = EfficientFrontier(mu, S)

pesos_brutos_minvol = ef_minvol.min_volatility()  # Calcula os pesos brutos

pesos_otimizados_minvol = pesos_brutos_minvol  # Limpa os pesos


# --- Etapa 8: Calcular Sharpe Ratio da carteira de minima volatilidade

retorno_otimizado_minvol = mu.dot(pd.Series(pesos_otimizados_minvol))  # Retorno da carteira otimizada

volatilidade_minvol = np.sqrt(np.array(list(pesos_otimizados_minvol.values())).T @ S @ np.array(list(pesos_otimizados_minvol.values())))

sharpe_otimizado = (retorno_otimizado_minvol - risk_free_rate) / volatilidade_minvol  # Sharpe otimizado

print(pesos_otimizados_minvol)

print(retorno_otimizado_minvol)

print(volatilidade_minvol)




# --- Etapa 9: Otimizar a carteira para Max Sharpe Global---


ef = EfficientFrontier(mu, S)

pesos_brutos_max = ef.max_sharpe()  # Calcula os pesos brutos

pesos_otimizados_max = pesos_brutos_max  # Limpa os pesos


# --- Etapa 6: Calcular Sharpe Ratio da carteira otimizada para o retorno dado ---

retorno_otimizado_max = mu.dot(pd.Series(pesos_otimizados_max))  # Retorno da carteira otimizada

volatilidade_otimizada_max = np.sqrt(np.array(list(pesos_otimizados_max.values())).T @ S @ np.array(list(pesos_otimizados_max.values())))

sharpe_otimizado = (retorno_otimizado_max - risk_free_rate) / volatilidade_otimizada_max  # Sharpe otimizado

print(pesos_otimizados_max)

print(retorno_otimizado_max)

print(volatilidade_otimizada_max)


#Gerar Fronteira Eficiente


ef = EfficientFrontier(mu, S, weight_bounds=(None, None))
ef.add_constraint(lambda w: w[0] >= 0.2)
ef.add_constraint(lambda w: w[2] == 0.15)
ef.add_constraint(lambda w: w[3] + w[4] <= 0.10)

fig, ax = plt.subplots()
plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)
plt.show()


#Gerar Carteiras aleatorias e Fronteira Eficiente
ef_new= EfficientFrontier(mu, S)

fig, ax = plt.subplots()
plotting.plot_efficient_frontier(ef_new, ax=ax, show_assets=False)

# Find the tangency portfolio
ef_new = EfficientFrontier(mu, S)
pesos_new=ef_new.max_sharpe()
ret_tangent, std_tangent, _ = ef_new.portfolio_performance()
ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")
ax.scatter(volatilidade_otimizada_ret, retorno_otimizado_ret, marker="*", s=100, c="black", label="Retorno Otimizado")
ax.scatter(volatilidade_minvol, retorno_otimizado_minvol, marker="*", s=100, c="blue", label="Minima Volatilidade")
# Generate random portfolios
n_samples = 10000
w = np.random.dirichlet(np.ones(len(mu)), n_samples)
rets = w.dot(mu)
stds = np.sqrt(np.diag(w @ S @ w.T))
sharpes = rets / stds
ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

# Output
ax.set_title("Efficient Frontier with random portfolios")
ax.legend()
plt.tight_layout()
plt.savefig("ef_scatter.png", dpi=200)
plt.show()


#Inserir as restriçoes de peso





#Gerar Carteiras aleatorias e Fronteira Eficiente
ef_new = EfficientFrontier(mu, S, weight_bounds=(None, None))
ef_new.add_constraint(lambda w: w[0] >= 0.2)
ef_new.add_constraint(lambda w: w[2] == 0.15)
ef_new.add_constraint(lambda w: w[3] + w[4] <= 0.10)

fig, ax = plt.subplots()
plotting.plot_efficient_frontier(ef_new, ax=ax, show_assets=False)

# Find the tangency portfolio
ef_new = EfficientFrontier(mu, S, weight_bounds=(None, None))
ef_new.add_constraint(lambda w: w[0] >= 0.2)
ef_new.add_constraint(lambda w: w[2] == 0.15)
ef_new.add_constraint(lambda w: w[3] + w[4] <= 0.10)


pesos_new=ef_new.max_sharpe()
ret_tangent, std_tangent, _ = ef_new.portfolio_performance()
ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe CR")
ax.scatter(volatilidade_otimizada_ret, retorno_otimizado_ret, marker="*", s=100, c="black", label="Retorno Otimizado")
ax.scatter(volatilidade_minvol, retorno_otimizado_minvol, marker="*", s=100, c="blue", label="Minima Volatilidade")
# Generate random portfolios
n_samples = 10000
w = np.random.dirichlet(np.ones(len(mu)), n_samples)
rets = w.dot(mu)
stds = np.sqrt(np.diag(w @ S @ w.T))
sharpes = rets / stds
ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

# Output
ax.set_title("Fronteira Efciente com Restriçoes")
ax.legend()
plt.tight_layout()
plt.savefig("Fronteira Efciente com Restriçoes", dpi=200)
plt.show()

#Adicionar no Streamlit
import io
# Salve a figura em um buffer de bytes
buf = io.BytesIO()


#plt.savefig(buf, format="jpeg")
buf.seek(0)

# Exiba a figura no Streamlit




buf = io.BytesIO()
buf.truncate(0)  # Limpe o buffer
buf.seek(0)  # Redefina o ponteiro
plt.savefig(buf, format="png")  # Verifique o formato da imagem
buf.seek(0)  # Redefina o ponteiro novamente
st.image(buf, caption='Gráfico com Restrições', use_column_width=True)








st.header("Sua Carteira Recomendada é")
df_pesos_otimizados_ret = pd.DataFrame(pesos_otimizados_ret, index=[0])
col1, col2, col3, col4, col5, col6,col7, col8 = st.columns(8)
col1.metric(label="Tesouro IPCA 5A", value = "{:.0%}".format(df_pesos_otimizados_ret["IMAB5"].iloc[0]))
col2.metric(label="Tesouro Pré Curto",value = "{:.0%}".format(df_pesos_otimizados_ret["2Y"].iloc[0]))
col3.metric(label="Tesouro IPCA Curto", value = "{:.0%}".format(df_pesos_otimizados_ret["IMAB5+"].iloc[0]))
col4.metric(label="Tesouro Pré  5A", value = "{:.0%}".format(df_pesos_otimizados_ret["5Y"].iloc[0]))
col5.metric(label="Bolsa", value = "{:.0%}".format(df_pesos_otimizados_ret["IBOV"].iloc[0]))
col6.metric(label=" Fundo MM", value = "{:.0%}".format(df_pesos_otimizados_ret["IHFA"].iloc[0]))
col7.metric(label="Fundo Imobiliario", value = "{:.0%}".format(df_pesos_otimizados_ret["IFIX"].iloc[0]))
col8.metric(label="Dolar", value = "{:.0%}".format(df_pesos_otimizados_ret["IHF"].iloc[0]))
#st.write(pd.DataFrame(df_pesos))
#st.write("Seu Retorno Esperado é")
#st.write(retorno_otimizado_ret)
#st.write("E seu Risco")
#st.write(volatilidade_otimizada_ret)


col1, col2 = st.columns(2)
col1.metric(label="Este é o seu Retorno ", value = "{:.2%}".format(retorno_otimizado_ret))
col2.metric(label="Este é seu Risco ", value = "{:.2%}".format(volatilidade_otimizada_ret))


st.divider()

st.header("Você não prefere a Sugestão Seguinte?")

df_pesos_otimizados_max = pd.DataFrame(pesos_otimizados_max, index=[0])
#st.write(pd.DataFrame(df_pesos1))

col1, col2, col3, col4, col5, col6,col7, col8 = st.columns(8)
col1.metric(label="Tesouro IPCA 5A", value = "{:.0%}".format(df_pesos_otimizados_max["IMAB5"].iloc[0]))
col2.metric(label="Tesouro Pré Curto",value = "{:.0%}".format(df_pesos_otimizados_max["2Y"].iloc[0]))
col3.metric(label="Tesouro IPCA < 5 A", value = "{:.0%}".format(df_pesos_otimizados_max["IMAB5+"].iloc[0]))
col4.metric(label="Tesouro Pré 5A", value = "{:.0%}".format(df_pesos_otimizados_max["5Y"].iloc[0]))
col5.metric(label="Bolsa", value = "{:.0%}".format(df_pesos_otimizados_max["IBOV"].iloc[0]))
col6.metric(label="Fundo MM", value = "{:.0%}".format(df_pesos_otimizados_max["IHFA"].iloc[0]))
col7.metric(label="Fundo Imob", value = "{:.0%}".format(df_pesos_otimizados_max["IFIX"].iloc[0]))
col8.metric(label="Dolar", value = "{:.0%}".format(df_pesos_otimizados_max["IHF"].iloc[0]))

#st.header("Este seria seu retorno com a Carteira Otimizada Sugerida")
#st.subheader(retorno_otimizado_max)


col1, col2 = st.columns(2)
col1.metric(label="Este seria seu Retorno Otimizado", value = "{:.2%}".format(retorno_otimizado_max))
col2.metric(label="Este seria seu Risco Otimizado", value = "{:.2%}".format(volatilidade_otimizada_max))

#st.header("Este seria seu risco  com a Carteira Otimizada Sugerida")
#st.subheader(volatilidade_otimizada_max)





import pandas as pd
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import plotly.graph_objects as go # Import Plotly
import matplotlib.pyplot as plt
from pypfopt import plotting


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


####PLOTAR

ef = EfficientFrontier(mu, S, weight_bounds=(None, None))
fig, ax = plt.subplots()
ef_max_sharpe = ef.deepcopy()
plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)

# Find the tangency portfolio
ef_max_sharpe.max_sharpe()
ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")

# Generate random portfolios
n_samples = 10000
w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
rets = w.dot(ef.expected_returns)
stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
sharpes = rets / stds
ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

# Output
ax.set_title("Efficient Frontier with random portfolios")
ax.legend()
plt.tight_layout()
plt.savefig("ef_scatter.png", dpi=200)
plt.show()
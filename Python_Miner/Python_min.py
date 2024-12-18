##DETERMINAÇÃO DO CONJUNTO UNIVERSO, IMPORTAÇÃO DE DADOS E CALCULOS DOS INDICADORES DE MOMENTO

import streamlit as st
import yfinance as yf
from datetime import datetime
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot
import numpy as np

from pylab import plt,mpl



tickers = [ "ALPA4.SA","ASAI3.SA","AZUL4.SA","B3SA3.SA","BPAN4.SA","BBSE3.SA"]
           #  "BBDC3.SA","BBDC4.SA","BRAP4.SA","BBAS3.SA","BRKM5.SA","BRFS3.SA","BPAC11.SA","CRFB3.SA","CCRO3.SA","CMIG4.SA","CMIN3.SA",
            # "COGN3.SA","CPLE6.SA","CSAN3.SA","CPFE3.SA","CVCB3.SA","CYRE3.SA","CURY3.SA", "ECOR3.SA","ELET3.SA",
            # "ELET6.SA","EMBR3.SA","ENGI11.SA","ENEV3.SA","EGIE3.SA","EQTL3.SA","EZTC3.SA","FLRY3.SA","GGBR4.SA",
            # "GOAU4.SA","GOLL4.SA","NTCO3.SA","HAPV3.SA","HYPE3.SA", "IRBR3.SA","ITSA4.SA",
            # "ITUB4.SA","JBSS3.SA","JHSF3.SA","KLBN11.SA","RAIZ4.SA","RENT3.SA","RRRP3.SA","LWSA3.SA","LREN3.SA","MGLU3.SA",
            # "MRFG3.SA","CASH3.SA","BEEF3.SA","MRVE3.SA","MULT3.SA","PETR3.SA","PETR4.SA","PRIO3.SA","PETZ3.SA","POSI3.SA",
            # "QUAL3.SA","RADL3.SA","RDOR3.SA","RAIL3.SA","SBSP3.SA","SANB11.SA","CSNA3.SA","SLCE3.SA","SMTO3.SA","SUZB3.SA","TAEE11.SA","VIVT3.SA",
            # "TIMS3.SA","TOTS3.SA","UGPA3.SA","USIM5.SA","VALE3.SA","VBBR3.SA","WEGE3.SA","YDUQ3.SA","DIRR3.SA","SMAL11.SA",
            # "GMAT3.SA","PSSA3.SA","MELI34.SA","TTEN3.SA","JSLG3.SA","MOVI3.SA","VIVA3.SA","SIMH3.SA","BOVA11.SA"]

tickers_without_sa = []
for ticker in tickers:
    tickers_without_sa.append(ticker[:-3])

inicio=datetime(2022,1,1)
fim=datetime(2024,4,1)
def baixar_dados(tickers,inicio,fim):
    dados= yf.download(tickers,start=inicio,end=fim, progress=False)['Adj Close']
    return dados

dados=baixar_dados(tickers, inicio, fim)

print(dados)
#dados=dados.to_frame()
dados=dados.dropna()

df=dados
#df = pd.concat([dados['AAPL'], dados['MSFT'], dados['GOOGL']], axis=1)
#df.columns = ['AAPL', 'MSFT', 'GOOG','META']
df = df.reset_index()
df = df.melt(id_vars=['Date'], var_name='Ação', value_name='Preço')
df['MM200'] = df.groupby('Ação')['Preço'].rolling(window=200).mean().reset_index(0, drop=True)
df['MM120'] = df.groupby('Ação')['Preço'].rolling(window=120).mean().reset_index(0, drop=True)
df['MM50'] = df.groupby('Ação')['Preço'].rolling(window=50).mean().reset_index(0, drop=True)
df['MM52W'] = df.groupby('Ação')['Preço'].rolling(window=366).mean().reset_index(0, drop=True)
df['Max52W'] = df.groupby('Ação')['Preço'].rolling(window=366).max().reset_index(0, drop=True)
df['Min52W'] = df.groupby('Ação')['Preço'].rolling(window=366).min().reset_index(0, drop=True)
df['RSI']=ta.rsi(df['Preço'], timeperiod=14)
df['Rdiario'] = df.groupby('Ação')['Preço'].pct_change(periods=1).reset_index(0, drop=True)
df['Vol52W'] = df.groupby('Ação')['Rdiario'].rolling(window=252).std().reset_index(0, drop=True)
df['Vol52Wy']=df['Vol52W']*252**(0.5)
df.dropna()

#Condições para Fator Momento

#Condição1 MM200
conditions = [
    (df['Preço'] >= df['MM200'] ) ,
    ]

results = ['1']

df['TMM200'] = np. select (conditions, results)

#condição 2 MM 120
conditions = [
    (df['Preço'] >= df['MM120'] ) ,
    ]
results = ['1']

#create new column based on conditions in column1 and column2
df['TMM120'] = np. select (conditions, results)


#Condição 3 MM50
conditions = [
    (df['Preço'] >= df['MM120'] ) ,
    ]

results = ['1']
#create new column based on conditions in column1 and column2
df['TMM50'] = np. select (conditions, results)


#Condição 4 MM50120

conditions = [
    (df['Preço'] >= df['MM50'] ) ,
    ]
results = ['1']
#create new column based on conditions in column1 and column2
df['TMM50120'] = np. select (conditions, results)
conditions = [
    (df['MM50'] >= df['MM120'] ) ,
    ]


#Condição 5 MM120200
results = ['1']

conditions = [
    (df['MM120'] >= df['MM200'] ) ,
    ]
#create new column based on conditions in column1 and column2
df['TMM120200'] = np. select (conditions, results)

#Condição 6 MM50200
results = ['1']
#create new column based on conditions in column1 and column2

conditions = [
    (df['MM50'] >= df['MM200'] ) ,
    ]
df['TMM50200'] = np. select (conditions, results)


#Condição 7 13MIN52W
results = ['1']
#create new column based on conditions in column1 and column2

conditions = [
    (df['Preço'] >= 1.3*df['Min52W'] ) ,
    ]
df['T13MIN52W'] = np. select (conditions, results)


#Condição 8 075MAX52W
results = ['1']
#create new column based on conditions in column1 and column2
conditions = [
    (df['Preço'] >= 0.75*df['Max52W'] ) ,
    ]

df['T075MAX52W'] = np. select (conditions, results)


#Condição 9 093MAX52W
results = ['1']
#create new column based on conditions in column1 and column2
conditions = [
    (df['Preço'] >= 0.93*df['Max52W'] ) ,
    ]

df['T093MAX52W'] = np. select (conditions, results)


#Condição 10 095MM200
results = ['1']
#create new column based on conditions in column1 and column2
conditions = [
    (df['Preço'] >= 0.95*df['MM200'] ) ,
    ]
df['T095MM200'] = np. select (conditions, results)

#Condiçao 11 RSI30
results = ['1']
df['TRSI'] = np. select (conditions, results)
conditions = [
    (df['RSI'] >= 0.95 ) ,
    ]
df['TRSI'] = np. select (conditions, results)

#Totalização pra número M

df['M']=pd.to_numeric(df['TMM200'])+pd.to_numeric(df['TMM120'])+pd.to_numeric(df['TMM50'])+pd.to_numeric(df["TMM50120"])+pd.to_numeric(df['TMM120200'])+pd.to_numeric(df["TMM50200"])+pd.to_numeric(df['T13MIN52W'])+pd.to_numeric(df['T075MAX52W'])+pd.to_numeric(df['T095MM200'])+pd.to_numeric(df['TRSI'])+pd.to_numeric(df['T093MAX52W'])

df['OP']=0
#Analise Fundamentalista/Criação de Fatores Volatilidade, Valor e Qualidade

#!pip install fundamentus

import fundamentus
import pandas as pd
import matplotlib.pyplot as plt

dff=fundamentus.get_resultado()
print (dff.columns)


#Filtar Fundamento só para o grupo tickers
dff_filtrado = dff[dff.index.isin(tickers_without_sa)]
#Vamos criar um filter_sort
##Função para criação dos percentis dos indicadores fundamentalistas##

def calculate_percentiles(df, column_name):
    # Sort the DataFrame by the specified column
    df_sorted = df.sort_values(by=column_name)

    # Calculate the percentiles
    percentiles = df_sorted[column_name].rank(pct=True)

    return percentiles

##Definição dos parametros e percentis que serao usados no filtro_vol

df['Vol52W_p']=calculate_percentiles(df,'Vol52W')


##Definição dos parametros e percentis que serao usados no filtro quality

filtro_quality=dff_filtrado[(dff.pl>0)&(dff.dy>0)&(dff.roe>0)]


# Calculate percentiles for each element in the chosen column
filtro_quality["roe_p"] = calculate_percentiles(dff_filtrado, 'roe')
filtro_quality["roic_p"] = calculate_percentiles(dff_filtrado, 'roic')
filtro_quality["mrgebit_p"] = calculate_percentiles(dff_filtrado, 'mrgebit')
filtro_quality["mrgliq_p"] = calculate_percentiles(dff_filtrado, 'mrgliq')

##Definição dos parametros e percentis que serao usados no filtro Value


filtro_value=dff[(dff.pvp>-100)&(dff.pl>-100)&(dff.pebit>-100)]
# Calculate percentiles for each element in the chosen column
filtro_value["pvp_p"] = calculate_percentiles(dff_filtrado, 'pvp')
filtro_value["pl_p"] = calculate_percentiles(dff_filtrado, 'pl')
filtro_value["pebit_p"] = calculate_percentiles(dff_filtrado, 'pebit')

print(filtro_value)


# Display the updated DataFrame  filtro_quality

filtro_quality=filtro_quality[(filtro_quality.roe_p>0.75)&(filtro_quality.roic_p>0.75)&(filtro_quality.mrgebit_p>0.75)&(filtro_quality.mrgliq_p>0.75)]
print(filtro_quality)

# Display the updated DataFrame  filtro_value
filtro_value=filtro_value[(filtro_value.pvp_p<0.25)&(filtro_value.pl_p<0.25)&(filtro_value.pebit_p<0.250)]
print(filtro_value)

#filtro.shape
#filtro = filtro.loc[~filtro.index.duplicated()]
#filtro.index = filtro.index.astype(str)
filtro_quality.index = filtro_quality.index + ".SA"
filtro_value.index = filtro_value.index + ".SA"

print(filtro_value)
# Criação da coluna filtro_quality no data frame df
df['filtro_quality'] = 0  # Initialize all elements to 0
df['filtro_value'] = 0  # Initialize all elements to 0

for i, element in enumerate(df['Ação']):
    if element in filtro_quality.index:
        df.loc[i, 'filtro_quality'] = 1  # Set to 1 if element found in filtro

filtro_quality.index = filtro_quality.index.str.replace(".SA", "")



for i, element in enumerate(df['Ação']):
    if element in filtro_value.index:
        df.loc[i, 'filtro_value'] = 1  # Set to 1 if element found in filtro

filtro_value.index = filtro_value.index.str.replace(".SA", "")

# Definição da condição de compra , venda  do sistema a condição é reprentada na coluna OP (operação)
cond = [
((df['M'] <= 3) ),
((df['M'] >= 7) ),
((4<df['M']) & (df['M']<7) )
]
choices = [-1, 1,0]
df['filtro_momentum']=np.select(cond,choices)



# Definição dos parametros do filtro vol
cond = [
((df['Vol52W_p'] >= 0.75) ),
((df['Vol52W_p'] <= 0.25) ),
((0.25<df['Vol52W'])<0.75  )
]
choices = [-1, 1,0]
df['filtro_vol']=np.select(cond,choices)




#Definição dos Filtros Usados e como serão operados

cond = [(df['filtro_momentum'] == -1) &(df['filtro_vol'] == -1),
(df['filtro_momentum'] == 1) &(df['filtro_vol'] == 1) ,
(df['filtro_momentum']==0 )
]
choices = [-1, 1,0]
df['OP']=np.select(cond,choices)





# Check if 'BOVA11.SA' exists in the 'Ação' column

#df['Numero_de_op']=df['OP'].groupby('Date').sum['OP']

if df['Ação'].isin(['BOVA11.SA']).any():
    # Set 'OP' to 0 for rows where 'Ação' is 'BOVA11.SA'
    df.loc[df['Ação'] == 'BOVA11.SA', 'OP'] = 0

df['varperc'] = df['Preço'].pct_change()


#Segunda Parte

print(filtro_quality)

print(dff)

print(df['Vol52W_p'])



numerom=df[['Date','Ação','M','OP','varperc','filtro_momentum','filtro_quality','filtro_value','filtro_vol','Vol52Wy']]
Fator_de_correção = pd.DataFrame()
Fator_de_correção['Date']=-numerom.groupby('Date')['OP'].sum()

Fator_de_correção_bova11 = Fator_de_correção['Date'].values

if "BOVA11.SA" in numerom['Ação'].values:
    numerom.loc[numerom['Ação'] == "BOVA11.SA", 'OP'] = Fator_de_correção_bova11
numerom['varperc'] = pd.to_numeric(numerom['varperc'])
numerom = numerom[numerom['OP'].notna()]
timestamp_rows = []
for i, row in numerom.iterrows():
    if type(row['OP']) == pd.Timestamp:
        timestamp_rows.append(i)
numerom_filtered = numerom.drop(timestamp_rows)
numerom=numerom_filtered
numerom['resultado_dia']=numerom['OP']*numerom['varperc']


num_comp_date = (
    numerom.groupby('Date')['OP']
    .apply(lambda x: x.value_counts().get(1, 0))
    .reset_index(name='Numero de compras')
)
num_vend_date = (
    numerom.groupby('Date')['OP']
    .apply(lambda x: x.value_counts().get(-1, 0))
    .reset_index(name='Numero de vendas')
)



def check_change_grouped(df):
  """
  Checks if the value in the "OP" column has changed between consecutive rows for the same stock within each date group.

  Args:
      df: A Pandas DataFrame containing the data.

  Returns:
      A new DataFrame with an additional column "operação" indicating whether the "OP" value has changed.
  """

  grouped_df = df.groupby('Ação')

  result = []

  for date, group_df in grouped_df:
    # Sort the group by Papel and Time within each date group
    group_df = group_df.sort_values(['Date'])

    # Compare consecutive rows for OP value change
    group_df['operação'] = (group_df['OP'] != group_df['OP'].shift(1)) & (group_df['Ação'] == group_df['Ação'].shift(1))

    # Append the processed group DataFrame to the result list
    result.append(group_df)

  # Concatenate the processed DataFrames
  processed_df = pd.concat(result)

  return processed_df

# Use the function on your DataFrame
df=numerom
df=check_change_grouped(df)
df_with_operation = check_change_grouped(df)
# Print the DataFrame with the new "operação" column
#print(df_with_operation)
numerom=df_with_operation




numerom.Date = pd.to_datetime(df.Date)
data_para_impressao= "2023-10-04"

slice = numerom[numerom.Date == data_para_impressao]

print(Fator_de_correção)

#type(Fator_de_correção)
#type(numerom)


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

filtered_df = df_with_operation[df_with_operation.Date == '2024-03-28']

print(filtered_df)

# Restore the previous display settings
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')

print(df_with_operation)


#Impressão do Relatório das datas definidas


from IPython.display import display
data_para_impressao = "2024-03-28"

slice = numerom[numerom["Date"] == data_para_impressao][["Date", "Ação", "M", "OP","operação","filtro_momentum","filtro_quality",'filtro_value','filtro_vol','Vol52Wy']]

pd.set_option('display.max_rows', None)

#slice=slice.drop(index=slice.index)

st.write("df")
slice

pd.reset_option('display.max_rows')

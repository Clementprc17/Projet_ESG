import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go
import scipy.optimize as sco
import requests
import time

# --- Dictionnaire Objectif de Neutralité Carbone ---
neutralite_carbone = {
    'AAPL': 2030, 'ABT': 2050, 'ACN': 2025, 'ADBE': 2035, 'ADI': 2030,
    'ALGN': 2050, 'AMZN': 2040, 'AMD': 2030, 'AXP': 2035, 'BA': 2050,
    'BLK': 2050, 'BRK-B': 2050, 'C': 2050, 'CB': 2050, 'CCI': 2050,
    'CSCO': 2025, 'CVX': 2050, 'DHR': 2040, 'DIS': 2030,
    'EXC': 2050, 'FDX': 2040, 'FIS': 2050, 'GE': 2050,
    'GILD': 2030, 'GM': 2040, 'GOOGL': 2030, 'GS': 2030, 'HON': 2035,
    'IBM': 2030, 'ICE': 2040, 'INTC': 2040, 'ISRG': 2040, 'JNJ': 2045, 'KMB': 2040,
    'KO': 2040, 'LLY': 2040, 'LMT': 2050, 'MA': 2040, 'MCD': 2050, 'MCO': 2040,
    'MMM': 2050, 'MSFT': 2030, 'NEE': 2045, 'NOW': 2030,
    'NVDA': 2040, 'ORCL': 2050, 'OTIS': 2040, 'PEP': 2040, 'PFE': 2040, 'PG': 2040, 'PLD': 2040,
    'PSA': 2040, 'PYPL': 2040, 'QCOM': 2040, 'SBUX': 2030, 'SCHW': 2040,
    'SPGI': 2040, 'SYK': 2040, 'T': 2035, 'TGT': 2040, 'TMO': 2040,
    'TSLA': 2030, 'TXN': 2040, 'V': 2040, 'VZ': 2035,
    'XOM': 2050, 'ZTS': 2040, 'OC': 2040,
    'BSX': 2030, 'KEYS': 2040,'NOC': 2035, 'ETR': 2050,'BDX': 2040,
    'ABN.AS': 2050, 'ACS.MC': 2050, 'AIR.PA': 2050,
    'AKZA.AS': 2050, 'ALFA.ST': 2030, 'AMS.MC': 2025, 'ASML.AS': 2025,
    'AZN.L': 2025, 'BAS.DE': 2050, 'BAYN.DE': 2030, 'BBVA.MC': 2050, 'BMW.DE': 2050, 'BN.PA': 2050,
    'BNP.PA': 2050, 'CBK.DE': 2040, 'CRH.L': 2050, 'DG.PA': 2050,
    'DNB.OL': 2050, 'DTE.DE': 2040, 'ELE.MC': 2040, 'ENEL.MI': 2040, 'ENI.MI': 2050, 'ERIC-B.ST': 2030,
    'FER.MC': 2050, 'FRE.DE': 2050, 'GSK.L': 2030, 'HEI.DE': 2040, 'HEIA.AS': 2040,
    'HM-B.ST': 2030, 'IFX.DE': 2030, 'IMB.L': 2030, 'INGA.AS': 2050, 'ISP.MI': 2050,
    'ITX.MC': 2040, 'KPN.AS': 2030, 'LDO.MI': 2040, 'MC.PA': 2050, 'MUV2.DE': 2050,
    'NOKIA.HE': 2040, 'NOVO-B.CO': 2045, 'ORA.PA': 2040, 'PHIA.AS': 2045, 'PST.MI': 2030,
    'RAND.AS': 2050, 'REP.MC': 2050, 'RIO.L': 2050, 'RNO.PA': 2040,
    'SAN.MC': 2050, 'SAN.PA': 2045, 'SAND.ST': 2050, 'SAP.DE': 2030, 'SECU-B.ST': 2040,
    'SGO.PA': 2050, 'SHEL.L': 2050, 'SIE.DE': 2030, 'SKF-B.ST': 2040, 'SKG.L': 2025, 'SRG.MI': 2040,
    'SU.PA': 2025, 'TTE.PA': 2050, 'UCG.MI': 2050, 'ULVR.L': 2039, 'mts.mc' : 2050 , 'ker.pa' : 2050,
    'UNA.AS': 2039, 'UNI.MI': 2040, 'VIE.PA': 2050, 'VOD.L': 2040, 'VOLV-B.ST': 2040, 'VOW3.DE': 2050,
    'APTV': 2040, 'DD': 2030,'MRK': 2030,'DOW': 2050,'CAT': 2035, 'EMR': 2035, 'ROK': 2040,'F': 2050,        
    'ETN': 2030,'CMI': 2050,'IR': 2050,'LHX': 2050,'TT': 2030,'GWW': 2050, 'OR.PA' : 2050,
    'J': 2040,'RSG': 2040,'TRI': 2040,'VRSK': 2040,'WM': 2040, 'BBY': 2040,'GIL': 2030,'WHR': 2030,'HST': 2050,'KIM': 2050,
    'WY': 2030,'STT': 2050,'CVS': 2050,'EW': 2040, 'VIV.PA' : 2025 , 'orsted.co' : 2040 , 
    'BALL': 2050,'ECL': 2040, 'IFF': 2040, 'ABBV': 2050, 'WAT': 2050, 'REL.L': 2040,'NOVN.SW': 2040,'SK.PA': 2050,
    'IMCD.AS': 2050,'CABK.MC': 2050,'CAP.PA': 2040,'CLNX.MC': 2050,'DSFIR.AS': 2040,'EDP.LS': 2030,'RI.PA': 2050

}

# --- Fonctions utilitaires ---
def compute_sharpe(prices, risk_free_rate=0.03):
    """Calcule le ratio de Sharpe à partir d'une série de prix en intégrant le taux sans risque."""
    returns = prices.pct_change().dropna()
    if returns.empty:
        return 0
    avg_return = returns.mean() * 252
    volatility = returns.std() * np.sqrt(252)
    if volatility == 0:
        return 0
    sharpe = (avg_return - risk_free_rate) / volatility
    return sharpe

def load_tickers():
    """Retourne une DataFrame combinée de tickers pour USA et Europe avec leur marché."""
    usa = [
        'AAPL', 'ABT', 'ACN', 'ADBE', 'ADI', 'ALGN', 'AMZN', 'AMD',
        'AXP', 'BA', 'BLK', 'BRK-B', 'C', 'CB', 'CCI','IFF',
        'CSCO', 'CVX', 'DE', 'DHR', 'ECL', 'EW', 'EXC','ABBV',
        'FDX', 'FIS', 'GE', 'GILD', 'GM', 'GOOGL', 'GS', 'HON', 'IBM',
        'ICE', 'INTC', 'ISRG', 'JNJ', 'KMB', 'KO', 'LLY', 'LMT', 'MA', 'MCD',
        'MCO', 'MMM', 'MSFT', 'NEE',  'NVDA', 'ORCL','CVS','WAT',
        'OTIS', 'PEP', 'PFE', 'PG', 'PLD', 'PSA', 'PYPL', 'QCOM', 'SBUX', 'SCHW',
        'SPGI', 'SYK', 'T', 'TGT', 'TMO', 'TSLA', 'TXN','WY','BALL',
        'V', 'VZ', 'XOM', 'ZTS','DD','MRK','DOW','CAT','EMR','ROK','ETN','F', 'APTV','CMI','BDX','HST','KIM',
        'IR','LHX','NOC','OC','TT','GWW', 'J','RSG','TRI','VRSK','WM','BBY','GIL','WHR', 'BSX', 'KEYS','ETR','STT'
    ]
    europe = [
        'ABN.AS', 'ACS.MC', 'AIR.PA', 'AKZA.AS', 'ALFA.ST', 'REL.L',
        'AMS.MC', 'ASML.AS', 'AZN.L', 'BAS.DE', 'BAYN.DE', 'BBVA.MC', 'BMW.DE', 'BN.PA',
        'BNP.PA', 'CBK.DE', 'CRH.L', 'DNB.OL', 'DG.PA', 'DTE.DE', 'ELE.MC',
        'ENEL.MI', 'ENI.MI', 'ERIC-B.ST', 'FER.MC', 'FRE.DE', 'GSK.L', 'HEI.DE', 'HEIA.AS',
        'HM-B.ST', 'IFX.DE', 'IMB.L', 'INGA.AS', 'ISP.MI', 'ITX.MC', 'KER.PA',
        'KPN.AS', 'LDO.MI','MC.PA', 'MTS.MC', 'MUV2.DE', 'NOKIA.HE', 'NOVO-B.CO',
        'ORA.PA', 'OR.PA', 'ORSTED.CO', 'PST.MI', 'PHIA.AS', 'RAND.AS', 'REP.MC', 'RIO.L',
        'RNO.PA', 'SAN.MC', 'SAN.PA', 'SAP.DE', 'SAND.ST', 'SECU-B.ST',
        'SGO.PA', 'SHEL.L', 'SIE.DE', 'SKF-B.ST', 'SKG.L', 'SRG.MI', 'SU.PA',
        'TTE.PA', 'UCG.MI', 'ULVR.L', 'UNA.AS', 'UNI.MI', 'VIE.PA', 'VIV.PA', 'VOD.L',
        'VOLV-B.ST', 'VOW3.DE','NOVN.SW', 'SK.PA','IMCD.AS','CABK.MC','CAP.PA',
        'CLNX.MC','EDP.LS', 'RI.PA','DSFIR.AS'
    ]

    df_usa = pd.DataFrame({"Ticker": usa})
    df_usa["Marché"] = "USA"
    df_europe = pd.DataFrame({"Ticker": europe})
    df_europe["Marché"] = "Europe"
    return pd.concat([df_usa, df_europe], ignore_index=True)

def build_esg_dataframe():
    """Construit le DataFrame ESG en récupérant les indicateurs et en ajoutant la colonne Neutralite Carbone."""
    tickers_df = load_tickers()
    all_data = []
    for idx, row in tickers_df.iterrows():
        ticker = row["Ticker"]
        market = row["Marché"]
        try:
            t = yf.Ticker(ticker)
            df = t.sustainability
            if df is None:
                continue
            metrics = {"Ticker": ticker, "Marché": market}
            keys = ["totalEsg", "environmentScore", "socialScore", "governanceScore", "peerGroup",
                    "highestControversy", "militaryContract", "nuclear", "pesticides", "palmOil", "coal"]
            for key in keys:
                try:
                    metrics[key] = df.loc[key].iloc[0]
                except Exception:
                    metrics[key] = None
            metrics["Neutralite Carbone"] = neutralite_carbone.get(ticker, None)
            all_data.append(metrics)
        except Exception as e:
            print(f"Erreur pour {ticker} : {e}")
    return pd.DataFrame(all_data)

def telecharger_close_tickers(tickers, start_date="2016-01-01", end_date=None):
    """Télécharge les prix de clôture en remplissant les NaN par la dernière valeur connue."""
    if isinstance(tickers, pd.DataFrame):
        tickers_list = tickers["Ticker"].tolist()
    else:
        tickers_list = tickers
    close_data = yf.download(tickers_list, start=start_date, end=end_date, progress=False)["Close"]
    if close_data.index.tz is not None:
        close_data.index = close_data.index.tz_localize(None)
    close_data.fillna(method="ffill", inplace=True)
    return close_data

def process_tickers(esg_df, selected_market, analysis_start_date, analysis_end_date,
                    max_total_esg_risque, max_environment_risk, max_social_risk, max_governance_risk,
                    max_highestControversy, exclude_military, exclude_nuclear, exclude_pesticides,
                    exclude_palmOil, exclude_coal, max_neutralite_carbone, excluded_peer_groups):
    """Filtre le DataFrame ESG selon les critères définis et retourne les données pertinentes avec les historiques de prix."""
    if selected_market != "Tous":
        df_filtered = esg_df[esg_df["Marché"].isin([selected_market])]
    else:
        df_filtered = esg_df.copy()
    sharpe_start_date = analysis_start_date - relativedelta(years=5)
    sharpe_end_date = analysis_start_date
    data = []
    for index, row in df_filtered.iterrows():
        ticker = row["Ticker"]
        if row["totalEsg"] is None or row["totalEsg"] > max_total_esg_risque:
            continue
        if row["environmentScore"] is None or row["environmentScore"] > max_environment_risk:
            continue
        if row["socialScore"] is None or row["socialScore"] > max_social_risk:
            continue
        if row["governanceScore"] is None or row["governanceScore"] > max_governance_risk:
            continue
        if row["highestControversy"] is None or row["highestControversy"] > max_highestControversy:
            continue
        if row.get("Neutralite Carbone") is None or row.get("Neutralite Carbone") > max_neutralite_carbone:
            continue
        if exclude_military and row["militaryContract"]:
            continue
        if exclude_nuclear and row["nuclear"]:
            continue
        if exclude_pesticides and row["pesticides"]:
            continue
        if exclude_palmOil and row["palmOil"]:
            continue
        if exclude_coal and row["coal"]:
            continue
        if excluded_peer_groups and row["peerGroup"] is not None:
            if row["peerGroup"] in excluded_peer_groups:
                continue
        try:
            stock = yf.Ticker(ticker)
            hist_analysis = stock.history(start=analysis_start_date, end=analysis_end_date)
            if hist_analysis.empty:
                continue
            if hist_analysis.index.tz is not None:
                hist_analysis.index = hist_analysis.index.tz_localize(None)
            common_dates_analysis = pd.date_range(start=analysis_start_date, end=analysis_end_date, freq='B')
            price_analysis = hist_analysis['Close'].reindex(common_dates_analysis, method='ffill')
            price_analysis.fillna(method='ffill', inplace=True)
            hist_selection = stock.history(start=sharpe_start_date, end=sharpe_end_date)
            if hist_selection.empty:
                continue
            if hist_selection.index.tz is not None:
                hist_selection.index = hist_selection.index.tz_localize(None)
            common_dates_selection = pd.date_range(start=sharpe_start_date, end=sharpe_end_date, freq='B')
            price_selection = hist_selection['Close'].reindex(common_dates_selection, method='ffill')
            price_selection.fillna(method='ffill', inplace=True)
            sharpe = compute_sharpe(price_selection)
            info = stock.info
            sector = info.get("sector", "Unknown")
            data.append({
                "Ticker": ticker,
                "Sharpe": sharpe,
                "Total ESG": row["totalEsg"],
                "Environment Score": row["environmentScore"],
                "Social Score": row["socialScore"],
                "Governance Score": row["governanceScore"],
                "Peer Group": row["peerGroup"],
                "Highest Controversy": row["highestControversy"],
                "Military Contract": row["militaryContract"],
                "Nuclear": row["nuclear"],
                "Pesticides": row["pesticides"],
                "Palm Oil": row["palmOil"],
                "Coal": row["coal"],
                "Neutralite Carbone": row.get("Neutralite Carbone"),
                "Sector": sector,
                "Price Data": price_analysis,
            })
        except Exception as e:
            st.write(f"Erreur pour {ticker}: {e}")
            continue
    return pd.DataFrame(data)

def select_portfolio(df, n=15):
    """Sélectionne les n actions avec le meilleur ratio de Sharpe (basé sur 5 ans)."""
    return df.sort_values(by="Sharpe", ascending=False).head(n)

def calculate_portfolio_performance(portfolio_df, analysis_start_date, analysis_end_date, risk_free_rate=0.03):
    """
    Calcule la performance du portefeuille sur une période donnée en pondération égale.
    Renvoie le rendement total, la volatilité annualisée, le ratio de Sharpe, 
    les rendements cumulatifs et l'évolution de la valeur du portefeuille.
    """
    tickers = portfolio_df["Ticker"].tolist()
    prices = telecharger_close_tickers(tickers, start_date=analysis_start_date, end_date=analysis_end_date)
    if prices.empty:
        return None
    daily_returns = prices.pct_change().dropna()
    portfolio_daily_return = daily_returns.mean(axis=1)
    cumulative_returns = (1 + portfolio_daily_return).cumprod() - 1
    portfolio_value = (1 + portfolio_daily_return).cumprod() * 100
    total_return = portfolio_value.iloc[-1] / 100 - 1
    volatility = portfolio_daily_return.std() * np.sqrt(252)
    portfolio_sharpe = ((portfolio_daily_return.mean() * 252) - risk_free_rate) / (portfolio_daily_return.std() * np.sqrt(252))
    return {
        "Total Return": total_return,
        "Annualized Volatility": volatility,
        "Portfolio Sharpe": portfolio_sharpe,
        "Cumulative Returns": cumulative_returns,
        "Portfolio Value": portfolio_value,
    }

# Fonctions indicateurs de risque complémentaires
def compute_max_drawdown(portfolio_value):
    cummax = portfolio_value.cummax()
    drawdown = (portfolio_value - cummax) / cummax
    return abs(drawdown.min())

def compute_sortino_ratio(returns, risk_free_rate=0.03):
    downside_returns = returns.copy()
    downside_returns[downside_returns > 0] = 0
    downside_std = downside_returns.std() * np.sqrt(252) if not downside_returns.empty else 0
    annual_return = returns.mean() * 252
    return (annual_return - risk_free_rate) / downside_std if downside_std != 0 else np.nan

def compute_calmar_ratio(annual_return, max_drawdown):
    return annual_return / max_drawdown if max_drawdown != 0 else np.nan

@st.cache_data(show_spinner=False)
def get_all_peer_groups():
    esg_df = build_esg_dataframe()
    groups = esg_df["peerGroup"].dropna().unique().tolist()
    return sorted(groups)


# --- Instructions d'utilisation ---
st.title("Application de Portefeuille ESG - Finance Durable")
st.markdown(
    """
    **Instructions :**

    - **Contexte Environnemental :**
      Toutes les entreprises sélectionnées possèdent la norme environnementale ISO14001 (pour tout le groupe ou une partie).

    - **Paramètres des Indicateurs ESG :**
      - **Total ESG Risk (0-50) : nous recommandons 16**  
        • Bien : 0-16  
        • Moyen : 17-33  
        • Risqué : 34-50  
      - **Environment Score (0-15) : nous recommandons 7**  
        • Bien : 0-7  
        • Moyen : 8-11 
        • Risqué : 12-15  
      - **Social Score (0-15) : nous recommandons 7**  
        • Bien : 0-7  
        • Moyen : 8-11 
        • Risqué : 12-15   
      - **Governance Score (0-20) : nous recommandons 7**  
        • Bien : 0-7  
        • Moyen : 8-14  
        • Risqué : 15-20  
      - **Highest Controversy (1-5, plus bas est meilleur) : nous recommandons 2 ou 3 **  
        • Bien : 1-2  
        • Moyen : 3  
        • Risqué : 4-5  
    
      
    - **Neutralité Carbone :**
      Choisissez l'année maximum d'objectif de neutralité carbone (2030, 2040 ou 2050). Seules les entreprises ayant un objectif inférieur à la valeur choisie seront retenues.
      Ce critères peut parfois engendré moins de 10 actions dans le portefeuille, si c'est le cas repouser à 2050. (Pour les entreprises qui n'ont pas d'informations officielles concernant une date, on attribue l'année 2050)

    - **Exclusions :**
      Vous pouvez exclure les entreprises ayant des liens avec des contrats militaires, le nucléaire, les pesticides, l'huile de palme ou le charbon.
      Vous disposez également d'une liste pour exclure certaines industries (Peer Groups).

    Pour choisir les actions, on prendra dans les tickers restants ceux qui possèdent le meilleur ratio de sharpe sur les 5 dernières années. 

    Lancez l'analyse via le bouton dans la barre latérale pour visualiser dynamiquement les indicateurs et comprendre le choix du portefeuille.
    """
)

# --- Interface Streamlit ---
st.markdown("### Paramètres de Configuration")
st.sidebar.header("Configuration")

analysis_start_date = st.sidebar.date_input("Date de début", datetime(2023, 1, 1))
analysis_end_date = st.sidebar.date_input("Date de fin", datetime(2023, 12, 31))

max_total_esg_risque = st.sidebar.slider("Total ESG Risk Maximum", 0, 50, 50)
max_environment_risk = st.sidebar.slider("Environment Score Risk Maximum", 0, 15, 15)
max_social_risk = st.sidebar.slider("Social Score Risk Maximum", 0, 15, 15)
max_governance_risk = st.sidebar.slider("Governance Score Risk Maximum", 0, 20, 20)
max_highestControversy = st.sidebar.slider("Highest Controversy Maximum", 1, 5, 5)
max_neutralite_carbone = st.sidebar.selectbox("Objectif de Neutralité Carbone Maximum", options=[2030, 2040, 2050], index=1)

exclude_military = st.sidebar.checkbox("Exclure entreprises avec contrats militaires", value=True)
exclude_nuclear = st.sidebar.checkbox("Exclure entreprises avec liens nucléaires", value=True)
exclude_pesticides = st.sidebar.checkbox("Exclure entreprises avec liens pesticides", value=True)
exclude_palmOil = st.sidebar.checkbox("Exclure entreprises avec liens huile de palme", value=True)
exclude_coal = st.sidebar.checkbox("Exclure entreprises avec liens charbon", value=True)

selected_market = st.sidebar.selectbox("Sélectionnez le marché à analyser", options=["USA", "Europe", "Tous"], index=0)
excluded_peer_groups = st.sidebar.multiselect("Exclure ces industries (Peer Groups)", options=get_all_peer_groups(), default=[])

if st.sidebar.button("Exécuter l'analyse"):
    with st.spinner("Chargement de l'analyse..."):
        st.info("Construction du DataFrame ESG global...")
        data_df = build_esg_dataframe()
        st.write(f"DataFrame ESG créé avec {len(data_df)} tickers.")
        
        df_market = process_tickers(
            data_df, selected_market, analysis_start_date, analysis_end_date,
            max_total_esg_risque, max_environment_risk, max_social_risk, max_governance_risk,
            max_highestControversy, exclude_military, exclude_nuclear, exclude_pesticides,
            exclude_palmOil, exclude_coal, max_neutralite_carbone, excluded_peer_groups
        )
        st.write(f"{len(df_market)} actions ont passé les filtres ESG pour le marché {selected_market}.")
        if df_market.empty:
            st.write("Aucune action ne satisfait les critères ESG pour ce marché.")
        else:
            portfolio_df = select_portfolio(df_market, n=15)
            st.markdown(f"**Portefeuille sélectionné pour {selected_market} (basé sur le Sharpe des 5 dernières années)**")
            st.dataframe(portfolio_df[["Ticker", "Sharpe", "Total ESG", "Environment Score", "Social Score", "Governance Score", 
                                       "Peer Group", "Highest Controversy", "Neutralite Carbone", "Military Contract", "Nuclear", "Pesticides", 
                                       "Palm Oil", "Coal", "Sector"]])
            
            performance = calculate_portfolio_performance(portfolio_df, analysis_start_date, analysis_end_date, risk_free_rate=0.03)
            if performance:
                st.markdown("#### Performance du Portefeuille")
                st.write(f"Valeur finale du portefeuille (base 100) : {performance['Portfolio Value'].iloc[-1]:.2f}")
                st.write(f"Rendement sur la période : {(performance['Portfolio Value'].iloc[-1] - 100) / 100:.2%}")
                st.write(f"Volatilité Annualisée : {performance['Annualized Volatility']:.2%}")
                st.write(f"Sharpe du Portefeuille : {performance['Portfolio Sharpe']:.2f}")
                
                st.markdown("**Graphique de l'évolution du portefeuille :**")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=performance["Portfolio Value"].index,
                    y=performance["Portfolio Value"],
                    mode='lines',
                    name='Portefeuille'
                ))
                fig.update_layout(title='Évolution du Portefeuille', xaxis_title='Date', yaxis_title='Valeur (base 100)')
                st.plotly_chart(fig)
                
                st.markdown("#### Backtesting : Rentabilité Annualisée sur les 3 dernières années")
                years = []
                annual_returns = []
                for yr in [analysis_end_date.year - 3, analysis_end_date.year - 2, analysis_end_date.year - 1]:
                    start_year = datetime(yr, 1, 1)
                    end_year = datetime(yr, 12, 31)
                    perf_year = calculate_portfolio_performance(portfolio_df, start_year, end_year, risk_free_rate=0.03)
                    annual_return = perf_year["Total Return"] if perf_year else None
                    years.append(str(yr))
                    annual_returns.append(annual_return * 100 if annual_return is not None else 0)
                    
                st.markdown("### Rentabilité Annualisée par Année")
                fig_bar = go.Figure(data=[go.Bar(x=years, y=annual_returns)])
                fig_bar.update_layout(
                    title="Rendements Annualisés sur les 3 dernières années", 
                    xaxis_title="Année", 
                    yaxis_title="Rendement (%)",
                    xaxis=dict(type='category', tickmode='array', tickvals=years, ticktext=years)
                )
                st.plotly_chart(fig_bar)
                
                st.markdown("#### Neutralité Carbone Moyenne")
                neutralite_mean = portfolio_df["Neutralite Carbone"].mean()
                st.write(f"La neutralité carbone moyenne est de : {neutralite_mean:.1f}")
                
                st.markdown("#### Contributions Individuelles")
                individual_returns = []
                for i, row in portfolio_df.iterrows():
                    price_data = row["Price Data"]
                    if price_data is not None and not price_data.empty:
                        valid_prices = price_data.dropna()
                        if valid_prices.empty:
                            ret = np.nan
                        else:
                            start_price = valid_prices.iloc[0]
                            end_price = valid_prices.iloc[-1]
                            ret = (end_price / start_price) - 1
                    else:
                        ret = np.nan
                    individual_returns.append({"Ticker": row["Ticker"], "Return": ret})
                individual_returns_df = pd.DataFrame(individual_returns)
                st.dataframe(individual_returns_df.sort_values(by="Return", ascending=False))
                
                # --- Nouvelle Section : Analyse Complémentaire ---
                st.markdown("## Analyse Complémentaire")
                st.markdown(
                    """
                    **Maximum Drawdown :**  
                    Mesure la perte maximale subie par le portefeuille à partir de son pic historique.  
                    Cela permet d'évaluer le risque de pertes importantes dans une période donnée.

                    **Sortino Ratio :**  
                    Ce ratio se concentre uniquement sur la volatilité négative (périodes de baisse), fournissant une estimation du risque ajusté aux pertes.
                    
                    **Calmar Ratio :**  
                    Compare le rendement annuel à la perte maximale (maximum drawdown).  
                    Un ratio élevé indique que le portefeuille offre un bon rendement par rapport aux risques de fortes pertes.

                    **Analyse des Corrélations :**  
                    L'étude des corrélations entre les actifs (via la matrice et la heatmap) permet de visualiser la diversification.  
                    Des corrélations faibles entre actifs suggèrent une meilleure répartition du risque.
                    """
                )
                tickers_portfolio = portfolio_df["Ticker"].tolist()
                prices_comp = telecharger_close_tickers(tickers_portfolio, start_date=analysis_start_date, end_date=analysis_end_date)
                if not prices_comp.empty:
                    daily_returns_comp = prices_comp.pct_change().dropna().mean(axis=1)
                    port_value_comp = (1 + daily_returns_comp).cumprod() * 100
                    max_dd = compute_max_drawdown(port_value_comp)
                    sortino = compute_sortino_ratio(daily_returns_comp, risk_free_rate=0.03)
                    annual_return_comp = daily_returns_comp.mean() * 252
                    calmar = compute_calmar_ratio(annual_return_comp, max_dd)
                    
                    st.write(f"**Maximum Drawdown** : {max_dd:.2%}")
                    st.write(f"**Sortino Ratio** : {sortino:.2f}")
                    st.write(f"**Calmar Ratio** : {calmar:.2f}")
                    
                    st.markdown("### Analyse des Corrélations")
                    returns_all = telecharger_close_tickers(tickers_portfolio, start_date=analysis_start_date, end_date=analysis_end_date).pct_change().dropna()
                    corr = returns_all.corr()
                    st.dataframe(corr.style.background_gradient(cmap='coolwarm'))
                    
                    st.markdown("### Heatmap des Corrélations")
                    fig_corr, ax_corr = plt.subplots()
                    cax = ax_corr.matshow(corr, cmap='coolwarm')
                    fig_corr.colorbar(cax)
                    ax_corr.set_xticks(range(len(corr.columns)))
                    ax_corr.set_xticklabels(corr.columns, rotation=90, fontsize=8)
                    ax_corr.set_yticks(range(len(corr.index)))
                    ax_corr.set_yticklabels(corr.index, fontsize=8)
                    st.pyplot(fig_corr)
                else:
                    st.write("Pas de données pour l'analyse complémentaire.")
                
                st.markdown("## 🔍 Analyse ESG vs Rendement")
                scatter_df = portfolio_df.copy()
                scatter_df.set_index("Ticker", inplace=True)
                ir_df = individual_returns_df.set_index("Ticker")
                scatter_df["Return"] = ir_df["Return"]
                scatter_df.reset_index(inplace=True)
                chart = alt.Chart(scatter_df).mark_circle(size=100).encode(
                    x="Total ESG:Q",
                    y="Return:Q",
                    color="Sector:N",
                    tooltip=["Ticker", "Return", "Total ESG", "Sector"]
                ).interactive()
                st.altair_chart(chart, use_container_width=True)
            else:
                st.write("Impossible de calculer la performance pour le marché sélectionné.")

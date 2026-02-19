import streamlit as st
import pandas as pd
import pulp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- KONFIGURACE ---
st.set_page_config(page_title="KGJ Strategy Expert", layout="wide")

if 'fwd_data' not in st.session_state: st.session_state.fwd_data = None
if 'loc_data' not in st.session_state: st.session_state.loc_data = None

st.title("🎯 KGJ Strategy & Dispatch Optimizer")

# --- UNIVERZÁLNÍ ČISTIČ DAT ---
def clean_df(df, is_fwd=True):
    df = df.dropna(how='all').dropna(axis=1, how='all')
    # Smažeme ENTERy a přebytečné mezery z názvů sloupců
    df.columns = [str(c).replace('\n', ' ').strip() for c in df.columns]
    
    # První sloupec je VŽDY datum
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
    df = df.dropna(subset=[date_col]).rename(columns={date_col: 'datetime'})
    
    # Vytvoření propojovacího klíče MDH
    df['mdh'] = df['datetime'].dt.strftime('%m-%d-%H')
    
    if is_fwd:
        # FWD: 1. Datum, 2. EE Cena, 3. Plyn Cena
        df = df.rename(columns={df.columns[1]: 'ee_price', df.columns[2]: 'gas_price'})
    else:
        # Lokalita: 1. Datum, 2. Cena tepla, 3. Poptávka
        df = df.rename(columns={df.columns[1]: 'heat_price', df.columns[2]: 'heat_demand'})
    
    # Zbytek na čísla
    for col in df.columns:
        if col not in ['datetime', 'mdh']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# --- 1. SEKCE: TRŽNÍ DATA ---
st.sidebar.header("📈 Tržní FWD Křivky")
fwd_file = st.sidebar.file_uploader("1. Nahraj FWD ceny (EE/ZP)", type=["xlsx"])

if fwd_file:
    st.session_state.fwd_data = clean_df(pd.read_excel(fwd_file), is_fwd=True)

if st.session_state.fwd_data is not None:
    fwd_df = st.session_state.fwd_data
    years = sorted(fwd_df['datetime'].dt.year.unique())
    sel_year = st.sidebar.selectbox("Vyber rok pro analýzu", years)
    df_yr = fwd_df[fwd_df['datetime'].dt.year == sel_year].copy().reset_index(drop=True)
    
    # Úpravy cen
    ee_shift = st.sidebar.number_input("Posun EE [EUR/MWh]", value=0.0)
    gas_shift = st.sidebar.number_input("Posun Plyn [EUR/MWh]", value=0.0)
    df_yr['ee_price_mod'] = df_yr['ee_price'] + ee_shift
    df_yr['gas_price_mod'] = df_yr['gas_price'] + gas_shift

    # Graf trhu
    fig_m = make_subplots(specs=[[{"secondary_y": True}]])
    fig_m.add_trace(go.Scatter(x=df_yr['datetime'], y=df_yr['ee_price_mod'], name="EE Cena", line=dict(color='#00ff00')), secondary_y=False)
    fig_m.add_trace(go.Scatter(x=df_yr['datetime'], y=df_yr['gas_price_mod'], name="Plyn Cena", line=dict(color='red')), secondary_y=True)
    st.plotly_chart(fig_m, use_container_width=True)

# --- 2. SEKCE: LOKALITA ---
st.subheader("📍 Data lokality")
loc_file = st.file_uploader("2. Nahraj data lokality (aki11)", type=["xlsx"])

if loc_file:
    st.session_state.loc_data = clean_df(pd.read_excel(loc_file), is_fwd=False)
    st.success("Lokalita nahrána úspěšně.")

# --- 3. SEKCE: OPTIMALIZACE ---
if st.session_state.fwd_data is not None and st.session_state.loc_data is not None:
    calc_df = pd.merge(df_yr, st.session_state.loc_data[['mdh', 'heat_price', 'heat_demand']], on='mdh', how='inner')
    calc_df = calc_df.sort_values('datetime').reset_index(drop=True)

    with st.expander("🛠️ Technické parametry KGJ", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            k_th = st.number_input("Tepelný výkon [MW]", value=1.09)
            k_el = st.number_input("Elektrický výkon [MW]", value=0.999)
        with c2:
            k_eff = st.number_input("Tepelná účinnost", value=0.46)
            k_serv = st.number_input("Servis [EUR/hod]", value=12.0)
        with c3:
            dist_ee = st.number_input("Distribuce nákup [EUR]", value=33.0)
            b_eff = st.number_input("Účinnost kotle", value=0.95)

    if st.button("🚀 SPUSTIT OPTIMALIZACI"):
        T = len(calc_df)
        model = pulp.LpProblem("Dispatch", pulp.LpMaximize)
        
        # Proměnné
        q_kgj = pulp.LpVariable.dicts("q_kgj", range(T), 0, k_th)
        on = pulp.LpVariable.dicts("on", range(T), 0, 1, cat="Binary")
        q_boil = pulp.LpVariable.dicts("q_boil", range(T), 0)

        # Logika marží pro zobrazení
        calc_df['Margin_KGJ'] = (calc_df['ee_price_mod'] * (k_el/k_th)) + calc_df['heat_price'] - ((calc_df['gas_price_mod'] / k_eff) + (k_serv / k_th))
        calc_df['Margin_Boiler'] = calc_df['heat_price'] - (calc_df['gas_price_mod'] / b_eff)
        
        # Constraints & Objective
        for t in range(T):
            model += q_kgj[t] + q_boil[t] >= 0.99 * calc_df.loc[t, 'heat_demand']
            model += q_kgj[t] <= k_th * on[t]
            model += q_kgj[t] >= 0.55 * k_th * on[t]

        # Profit (Zjednodušený pro rychlost)
        model += pulp.lpSum([q_kgj[t] * calc_df.loc[t, 'Margin_KGJ'] + q_boil[t] * calc_df.loc[t, 'Margin_Boiler'] for t in range(T)])
        model.solve(pulp.PULP_CBC_CMD(msg=0))

        st.success(f"Hotovo! Odhadovaný hrubý zisk: {pulp.value(model.objective):,.0f} EUR")
        
        # Graf výsledků
        calc_df['KGJ_Output'] = [q_kgj[t].value() for t in range(T)]
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(x=calc_df['datetime'], y=calc_df['heat_demand'], name="Poptávka", line=dict(color='black')))
        fig_res.add_trace(go.Bar(x=calc_df['datetime'], y=calc_df['KGJ_Output'], name="Výkon KGJ"))
        st.plotly_chart(fig_res, use_container_width=True)

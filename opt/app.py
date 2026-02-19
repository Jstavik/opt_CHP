import streamlit as st
import pandas as pd
import pulp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# --- KONFIGURACE ---
st.set_page_config(page_title="KGJ Strategy Expert", layout="wide")

if 'fwd_data' not in st.session_state: st.session_state.fwd_data = None
if 'loc_data' not in st.session_state: st.session_state.loc_data = None

st.title("🎯 KGJ Strategy & Dispatch Optimizer")

# --- ODOLNÁ FUNKCE PRO ČIŠTĚNÍ DAT (Český formát datumu) ---
def clean_and_map(df, mapping):
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns=mapping)
    if 'datetime' in df.columns:
        # dayfirst=True vyřeší tvůj formát 19.02.2026
        df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True, errors='coerce')
        # Smažeme řádky, kde se nepodařilo datum vytvořit (nadpisy, prázdné řádky)
        df = df.dropna(subset=['datetime'])
        # Převedeme ceny na čísla (kdyby tam byl text)
        for col in df.columns:
            if col != 'datetime' and col != 'mdh':
                df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# --- SIDEBAR: TRHY ---
st.sidebar.header("📈 Tržní FWD Křivky")
fwd_file = st.sidebar.file_uploader("Nahraj 'FWD křivka EE_ZP.xlsx'", type=["xlsx"])

if fwd_file:
    raw_fwd = pd.read_excel(fwd_file)
    fwd_map = {'Datum': 'datetime', 'FWD (EUR/MWh)': 'ee_price', 'FWD plyn (EUR/MWh)': 'gas_price'}
    st.session_state.fwd_data = clean_and_map(raw_fwd, fwd_map)

if st.session_state.fwd_data is not None:
    fwd_df = st.session_state.fwd_data
    years = sorted(fwd_df['datetime'].dt.year.unique())
    sel_year = st.sidebar.selectbox("Vyber rok pro výpočet", years)
    
    df_yr = fwd_df[fwd_df['datetime'].dt.year == sel_year].copy().reset_index(drop=True)
    
    st.sidebar.subheader("Úpravy cen")
    ee_shift = st.sidebar.number_input("Posun EE Base [EUR/MWh]", value=0.0)
    df_yr['ee_price'] += ee_shift
    gas_shift = st.sidebar.number_input("Posun Plyn Base [EUR/MWh]", value=0.0)
    df_yr['gas_price'] += gas_shift

# --- HLAVNÍ: LOKALITA ---
st.subheader("📍 Data lokality")
loc_file = st.file_uploader("Nahraj data lokality (např. Behounkova)", type=["xlsx"])

if loc_file:
    raw_loc = pd.read_excel(loc_file)
    loc_map = {'Datum': 'datetime', 'Teplo (EUR/MWh)': 'heat_price', 'Behounkova DHV celkemMW': 'heat_demand'}
    st.session_state.loc_data = clean_and_map(raw_loc, loc_map)

# --- VÝPOČET ---
if st.session_state.fwd_data is not None and st.session_state.loc_data is not None:
    df_yr['mdh'] = df_yr['datetime'].dt.strftime('%m-%d-%H')
    st.session_state.loc_data['mdh'] = st.session_state.loc_data['datetime'].dt.strftime('%m-%d-%H')
    
    calc_df = pd.merge(df_yr, st.session_state.loc_data[['mdh', 'heat_price', 'heat_demand']], on='mdh', how='inner')
    calc_df = calc_df.sort_values('datetime').reset_index(drop=True)

    with st.expander("🛠️ Technické parametry", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            kgj_th = st.number_input("KGJ Tepelný výkon [MW]", value=1.09)
            kgj_el = st.number_input("KGJ Elektrický výkon [MW]", value=0.999)
            kgj_eff = st.number_input("KGJ Tepelná účinnost", value=0.46)
            kgj_serv = st.number_input("Servis [EUR/hod]", value=12.0)
        with c2:
            boil_max = st.number_input("Plynový kotel max [MW]", value=3.91)
            eboil_max = st.number_input("Elektrokotel max [MW]", value=0.605)
            dist_c = st.number_input("Distribuce EE nákup [EUR/MWh]", value=33.0)

    if st.button("🚀 SPUSTIT OPTIMALIZACI"):
        T = len(calc_df)
        model = pulp.LpProblem("KGJ_Dispatch", pulp.LpMaximize)

        # Proměnné
        q_kgj = pulp.LpVariable.dicts("q_kgj", range(T), 0, kgj_th)
        q_boil = pulp.LpVariable.dicts("q_boil", range(T), 0, boil_max)
        q_eboil = pulp.LpVariable.dicts("q_eboil", range(T), 0, eboil_max)
        kgj_on = pulp.LpVariable.dicts("on", range(T), 0, 1, cat="Binary")
        
        # Pomocné koeficienty
        kgj_gas_input = kgj_th / kgj_eff
        kgj_el_per_h = kgj_el / kgj_th
        kgj_gas_per_h = kgj_gas_input / kgj_th

        # Constraints
        for t in range(T):
            h_req = 0.99 * calc_df.loc[t, "heat_demand"]
            model += q_kgj[t] + q_boil[t] + q_eboil[t] >= h_req
            model += q_kgj[t] <= kgj_th * kgj_on[t]
            model += q_kgj[t] >= 0.55 * kgj_th * kgj_on[t]

        # Objective (Zisk)
        profit = []
        for t in range(T):
            ee = calc_df.loc[t, "ee_price"]
            gas = calc_df.loc[t, "gas_price"]
            hp = calc_df.loc[t, "heat_price"]
            
            # Zjednodušený ekonomický model (revize dle tvého originálu)
            rev = (hp * 0.99 * calc_df.loc[t, "heat_demand"]) + (ee * q_kgj[t] * kgj_el_per_h)
            cost = (gas * (q_kgj[t] * kgj_gas_per_h + q_boil[t]/0.95)) + (kgj_serv * kgj_on[t])
            # Náklady na elektrokotel z gridu (pokud ee_price + distribuce)
            eb_cost = (ee + dist_c) * (q_eboil[t]/0.98)
            profit.append(rev - cost - eb_cost)

        model += pulp.lpSum(profit)
        model.solve(pulp.PULP_CBC_CMD(msg=0))

        st.success(f"Optimalizace dokončena! Celkový zisk: {pulp.value(model.objective):,.0f} EUR")

        # --- VÝSLEDKY ---
        calc_df['q_kgj'] = [q_kgj[t].value() for t in range(T)]
        calc_df['q_boil'] = [q_boil[t].value() for t in range(T)]
        calc_df['q_eboil'] = [q_eboil[t].value() for t in range(T)]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=calc_df['datetime'], y=calc_df['heat_demand'], name="Poptávka", line=dict(color='black')))
        fig.add_trace(go.Bar(x=calc_df['datetime'], y=calc_df['q_kgj'], name="KGJ Teplo"))
        st.plotly_chart(fig, use_container_width=True)

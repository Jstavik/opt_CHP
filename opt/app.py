import streamlit as st
import pandas as pd
import pulp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# --- NASTAVENÍ STRÁNKY ---
st.set_page_config(page_title="KGJ Strategy Expert", layout="wide")

if 'fwd_data' not in st.session_state: st.session_state.fwd_data = None
if 'loc_data' not in st.session_state: st.session_state.loc_data = None

st.title("🎯 KGJ Strategy & Dispatch Optimizer")

# --- SIDEBAR: TRŽNÍ DATA ---
st.sidebar.header("📈 Tržní FWD Křivky")
fwd_file = st.sidebar.file_uploader("Nahraj 'FWD křivka EE_ZP.xlsx'", type=["xlsx"])

if fwd_file:
    # Načteme vše a převedeme názvy na malé písmena pro jistotu
    df_fwd = pd.read_excel(fwd_file)
    df_fwd.columns = [str(c).strip() for c in df_fwd.columns]
    
    # Mapování tvých názvů z obrázku
    rename_map = {
        'Datum': 'datetime',
        'FWD (EUR/MWh)': 'ee_price',
        'FWD plyn (EUR/MWh)': 'gas_price'
    }
    df_fwd = df_fwd.rename(columns=rename_map)

    # OPRAVA CHYBY: errors='coerce' změní neplatná data na "NaT" (Not a Time), které pak smažeme
    df_fwd['datetime'] = pd.to_datetime(df_fwd['datetime'], errors='coerce')
    
    # Odstranění řádků, kde není platné datum nebo chybí ceny (řeší ty prázdné řádky mezi roky)
    df_fwd = df_fwd.dropna(subset=['datetime', 'ee_price'])
    
    st.session_state.fwd_data = df_fwd
    st.sidebar.success(f"Nahráno {len(df_fwd)} platných řádků.")
    
    # Výřez pro daný rok a úprava Base
    df_yr = st.session_state.fwd_data[st.session_state.fwd_data['datetime'].dt.year == sel_year].copy().reset_index(drop=True)
    
    st.sidebar.subheader("Úpravy cen")
    ee_base_orig = df_yr['ee_price'].mean()
    st.sidebar.write(f"Původní EE Base: {ee_base_orig:.2f} EUR")
    ee_shift = st.sidebar.number_input("Posun EE Base [EUR/MWh]", value=0.0)
    df_yr['ee_price'] += ee_shift
    
    gas_base_orig = df_yr['gas_price'].mean()
    st.sidebar.write(f"Původní Plyn Base: {gas_base_orig:.2f} EUR")
    gas_shift = st.sidebar.number_input("Posun Plyn Base [EUR/MWh]", value=0.0)
    df_yr['gas_price'] += gas_shift

# --- HLAVNÍ ČÁST: LOKALITA ---
st.subheader("📍 Data lokality")
loc_file = st.file_uploader("Nahraj data lokality (Teplo a Poptávka)", type=["xlsx"])

if loc_file:
    df_loc = pd.read_excel(loc_file)
    # Mapování tvých názvů pro lokalitu
    loc_rename = {
        'Datum': 'datetime',
        'Teplo (EUR/MWh)': 'heat_price',
        'Behounkova DHV celkemMW': 'heat_demand'
    }
    df_loc = df_loc.rename(columns=loc_rename)
    df_loc['datetime'] = pd.to_datetime(df_loc['datetime'])
    st.session_state.loc_data = df_loc

if st.session_state.fwd_data is not None and st.session_state.loc_data is not None:
    # Přesvátkování (Join na základě měsíce, dne a hodiny)
    df_yr['mdh'] = df_yr['datetime'].dt.strftime('%m-%d-%H')
    st.session_state.loc_data['mdh'] = st.session_state.loc_data['datetime'].dt.strftime('%m-%d-%H')
    
    calc_df = pd.merge(df_yr, st.session_state.loc_data[['mdh', 'heat_price', 'heat_demand']], on='mdh', how='inner')
    calc_df = calc_df.sort_values('datetime').reset_index(drop=True)

    # Parametry technologií (z tvého původního behouvkova_opt.txt) 
    with st.expander("⚙️ Technické parametry", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            kgj_heat_p = st.number_input("KGJ Tepelný výkon [MW]", value=1.09) [cite: 1]
            kgj_el_p = st.number_input("KGJ Elektrický výkon [MW]", value=0.999) [cite: 1]
            kgj_eff = st.number_input("KGJ Účinnost (tepelná)", value=0.46) [cite: 1]
            kgj_serv = st.number_input("Servis [EUR/hod]", value=12.0) [cite: 1]
        with c2:
            boiler_eff = st.number_input("Účinnost pl. kotle", value=0.95) [cite: 1]
            eboil_eff = st.number_input("Účinnost el. kotle", value=0.98) [cite: 1]
            dist_cost = st.number_input("Distribuce EE [EUR/MWh]", value=33.0) [cite: 1]

    if st.button("🚀 SPUSTIT OPTIMALIZACI"):
        T = len(calc_df)
        model = pulp.LpProblem("KGJ_Dispatch", pulp.LpMaximize) [cite: 2]

        # Proměnné [cite: 2]
        q_kgj = pulp.LpVariable.dicts("q_kgj", range(T), 0, kgj_heat_p)
        q_boil = pulp.LpVariable.dicts("q_boil", range(T), 0, 3.91) # Max boiler z tvého kódu 
        q_eboil = pulp.LpVariable.dicts("q_eboil", range(T), 0, 0.605) [cite: 1]
        kgj_on = pulp.LpVariable.dicts("on", range(T), 0, 1, cat="Binary") [cite: 2]
        ee_sold = pulp.LpVariable.dicts("ee_sold", range(T), 0)
        ee_grid = pulp.LpVariable.dicts("ee_grid", range(T), 0)

        # Koeficienty 
        kgj_gas_per_h = (kgj_heat_p / kgj_eff) / kgj_heat_p
        kgj_el_per_h = kgj_el_p / kgj_heat_p

        # Constraints [cite: 3, 4]
        for t in range(T):
            h_req = 0.99 * calc_df.loc[t, "heat_demand"] [cite: 1]
            model += q_kgj[t] + q_boil[t] + q_eboil[t] >= h_req [cite: 3]
            model += q_kgj[t] <= kgj_heat_p * kgj_on[t]
            model += q_kgj[t] >= 0.55 * kgj_heat_p * kgj_on[t] [cite: 1]
            
            ee_prod = q_kgj[t] * kgj_el_per_h
            model += ee_sold[t] <= ee_prod # Zjednodušená bilance pro ilustraci [cite: 3]
            model += q_eboil[t] <= eboil_eff * (ee_grid[t] + (ee_prod - ee_sold[t])) [cite: 4]

        # Objective (Profit) [cite: 5, 6]
        profit = []
        for t in range(T):
            ee = calc_df.loc[t, "ee_price"]
            gas = calc_df.loc[t, "gas_price"]
            hp = calc_df.loc[t, "heat_price"]
            h_dem = calc_df.loc[t, "heat_demand"]
            
            rev = (hp * 0.99 * h_dem) + (ee * ee_sold[t]) [cite: 5]
            cost = (gas * (q_kgj[t] * kgj_gas_per_h + q_boil[t] / boiler_eff)) + \
                   ((ee + dist_cost) * ee_grid[t]) + (kgj_serv * kgj_on[t]) [cite: 5, 6]
            profit.append(rev - cost)
        
        model += pulp.lpSum(profit)
        model.solve(pulp.PULP_CBC_CMD(msg=0))

        st.success("Hotovo!")
        
        # Výpočet marginů a triggerů (tvoje logika ze sekce 6) [cite: 7, 8, 11]
        # (Zde by následoval kód pro vytvoření 'res' tabulky a grafů jako v minulé odpovědi)


import streamlit as st
import pandas as pd
import pulp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# --- KONFIGURACE STRÁNKY ---
st.set_page_config(page_title="KGJ Strategy Expert", layout="wide")

# Inicializace paměti, aby data nezmizela při každém kliknutí
if 'fwd_data' not in st.session_state: st.session_state.fwd_data = None
if 'loc_data' not in st.session_state: st.session_state.loc_data = None

st.title("🎯 KGJ Strategy & Dispatch Optimizer")

# --- POMOCNÁ FUNKCE PRO ČIŠTĚNÍ DAT ---
def clean_and_map(df, mapping):
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns=mapping)
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['datetime'])
        for col in df.columns:
            if col != 'datetime':
                df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# --- SIDEBAR: TRŽNÍ DATA (FWD křivka EE_ZP.xlsx) ---
st.sidebar.header("📈 Tržní FWD Křivky")
fwd_file = st.sidebar.file_uploader("1. Nahraj 'FWD křivka EE_ZP.xlsx'", type=["xlsx"])

if fwd_file:
    raw_fwd = pd.read_excel(fwd_file)
    # Mapování tvých názvů z Excelu na interní názvy
    fwd_map = {
        'Datum': 'datetime', 
        'FWD (EUR/MWh)': 'ee_price', 
        'FWD plyn (EUR/MWh)': 'gas_price'
    }
    st.session_state.fwd_data = clean_and_map(raw_fwd, fwd_map)

# --- ANALÝZA A ÚPRAVA KŘIVEK ---
if st.session_state.fwd_data is not None:
    fwd_df = st.session_state.fwd_data
    years = sorted(fwd_df['datetime'].dt.year.unique())
    sel_year = st.sidebar.selectbox("Vyber rok pro analýzu", years)
    
    # Výřez dat pro daný rok
    df_yr = fwd_df[fwd_df['datetime'].dt.year == sel_year].copy().reset_index(drop=True)
    
    # Výpočet Peak/Base pro EE
    df_yr['hour'] = df_yr['datetime'].dt.hour
    df_yr['weekday'] = df_yr['datetime'].dt.weekday
    is_peak = (df_yr['weekday'] < 5) & (df_yr['hour'] >= 8) & (df_yr['hour'] < 20)
    
    ee_base = df_yr['ee_price'].mean()
    ee_peak = df_yr[is_peak]['ee_price'].mean()
    gas_base = df_yr['gas_price'].mean()

    st.sidebar.markdown(f"""
    **Původní hodnoty {sel_year}:**
    * EE Base: `{ee_base:.2f}` | Peak: `{ee_peak:.2f}`
    * Plyn Base: `{gas_base:.2f}`
    """)

    # Úpravy (Shifty)
    st.sidebar.subheader("🛠️ Úprava Base cen")
    ee_shift = st.sidebar.number_input("Posun EE [EUR/MWh]", value=0.0, step=0.5)
    gas_shift = st.sidebar.number_input("Posun Plyn [EUR/MWh]", value=0.0, step=0.5)
    
    # Aplikace úprav
    df_yr['ee_price_mod'] = df_yr['ee_price'] + ee_shift
    df_yr['gas_price_mod'] = df_yr['gas_price'] + gas_shift

    # --- GRAFICKÉ ZNÁZORNĚNÍ ---
    st.subheader(f"📊 Náhled tržních křivek pro rok {sel_year}")
    fig_market = make_subplots(specs=[[{"secondary_y": True}]])
    
    # EE Křivky
    fig_market.add_trace(go.Scatter(x=df_yr['datetime'], y=df_yr['ee_price'], name="EE Původní", line=dict(color='gray', width=1, dash='dot')), secondary_y=False)
    fig_market.add_trace(go.Scatter(x=df_yr['datetime'], y=df_yr['ee_price_mod'], name="EE Upravená", line=dict(color='#00ff00', width=1.5)), secondary_y=False)
    
    # Plyn Křivky
    fig_market.add_trace(go.Scatter(x=df_yr['datetime'], y=df_yr['gas_price'], name="Plyn Původní", line=dict(color='rgba(255,0,0,0.2)', width=1)), secondary_y=True)
    fig_market.add_trace(go.Scatter(x=df_yr['datetime'], y=df_yr['gas_price_mod'], name="Plyn Upravený", line=dict(color='red', width=1.5)), secondary_y=True)
    
    fig_market.update_layout(height=450, hovermode="x unified")
    fig_market.update_yaxes(title_text="EE Cena [EUR/MWh]", secondary_y=False)
    fig_market.update_yaxes(title_text="Plyn Cena [EUR/MWh]", secondary_y=True)
    st.plotly_chart(fig_market, use_container_width=True)

# --- HLAVNÍ ČÁST: LOKALITA ---
st.subheader("📍 Data lokality")
loc_file = st.file_uploader("2. Nahraj data lokality (např. Behounkova)", type=["xlsx"])

if loc_file:
    raw_loc = pd.read_excel(loc_file)
    loc_map = {'Datum': 'datetime', 'Teplo (EUR/MWh)': 'heat_price', 'Behounkova DHV celkemMW': 'heat_demand'}
    st.session_state.loc_data = clean_and_map(raw_loc, loc_map)

# --- OPTIMALIZACE ---
if st.session_state.fwd_data is not None and st.session_state.loc_data is not None:
    # Sladění dat přes MDH (měsíc-den-hodina)
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

        q_kgj = pulp.LpVariable.dicts("q_kgj", range(T), 0, kgj_th)
        q_boil = pulp.LpVariable.dicts("q_boil", range(T), 0, boil_max)
        q_eboil = pulp.LpVariable.dicts("q_eboil", range(T), 0, eboil_max)
        kgj_on = pulp.LpVariable.dicts("on", range(T), 0, 1, cat="Binary")
        
        kgj_gas_per_h = (kgj_th / kgj_eff) / kgj_th
        kgj_el_per_h = kgj_el / kgj_th

        for t in range(T):
            h_req = 0.99 * calc_df.loc[t, "heat_demand"]
            model += q_kgj[t] + q_boil[t] + q_eboil[t] >= h_req
            model += q_kgj[t] <= kgj_th * kgj_on[t]
            model += q_kgj[t] >= 0.55 * kgj_th * kgj_on[t]

        profit = []
        for t in range(T):
            # Použijeme MODIFIKOVANÉ ceny z tvého grafu
            ee = calc_df.loc[t, "ee_price_mod"]
            gas = calc_df.loc[t, "gas_price_mod"]
            hp = calc_df.loc[t, "heat_price"]
            h_dem = calc_df.loc[t, "heat_demand"]
            
            rev = (hp * 0.99 * h_dem) + (ee * q_kgj[t] * kgj_el_per_h)
            cost = (gas * (q_kgj[t] * kgj_gas_per_h + q_boil[t]/0.95)) + (kgj_serv * kgj_on[t])
            eb_cost = (ee + dist_c) * (q_eboil[t]/0.98)
            profit.append(rev - cost - eb_cost)

        model += pulp.lpSum(profit)
        model.solve(pulp.PULP_CBC_CMD(msg=0))

        st.success(f"Optimalizace hotova! Zisk: {pulp.value(model.objective):,.0f} EUR")

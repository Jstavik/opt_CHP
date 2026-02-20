import streamlit as st
import pandas as pd
import pulp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="KGJ Strategy Expert PRO", layout="wide")

# Inicializace stavu
if 'fwd_data' not in st.session_state: st.session_state.fwd_data = None

st.title("🚀 KGJ Strategy & Dispatch Optimizer PRO")

# --- 1. SIDEBAR: CENY A EDITACE ---
with st.sidebar:
    st.header("📈 1. Tržní ceny (FWD)")
    fwd_file = st.file_uploader("Nahraj FWD křivku (Excel)", type=["xlsx"])
    
    if fwd_file:
        df_raw = pd.read_excel(fwd_file)
        df_raw.columns = [str(c).strip() for c in df_raw.columns]
        date_col = df_raw.columns[0]
        df_raw[date_col] = pd.to_datetime(df_raw[date_col], dayfirst=True)
        
        years = sorted(df_raw[date_col].dt.year.unique())
        sel_year = st.selectbox("Rok pro analýzu", years)
        df_year = df_raw[df_raw[date_col].dt.year == sel_year].copy()
        
        # Výpočet a zobrazení průměrů
        avg_ee_raw = float(df_year.iloc[:, 1].mean())
        avg_gas_raw = float(df_year.iloc[:, 2].mean())
        
        st.subheader("🛠️ Úprava na aktuální trh")
        st.info(f"Původní průměry: EE {avg_ee_raw:.2f} | Plyn {avg_gas_raw:.2f}")
        
        ee_market_new = st.number_input("Nová cílová cena EE [EUR]", value=avg_ee_raw)
        gas_market_new = st.number_input("Nová cílová cena Plyn [EUR]", value=avg_gas_raw)
        
        ee_shift = ee_market_new - avg_ee_raw
        gas_shift = gas_market_new - avg_gas_raw
        
        df_fwd = df_year.copy()
        df_fwd.columns = ['datetime', 'ee_original', 'gas_original']
        df_fwd['ee_price'] = df_fwd['ee_original'] + ee_shift
        df_fwd['gas_price'] = df_fwd['gas_original'] + gas_shift
        st.session_state.fwd_data = df_fwd

    st.divider()
    st.header("⚙️ 2. Aktivní technologie")
    use_kgj = st.checkbox("Kogenerace (KGJ)", value=True)
    use_boil = st.checkbox("Plynový kotel", value=True)
    use_ek = st.checkbox("Elektrokotel", value=True)
    use_tes = st.checkbox("Nádrž (TES)", value=True)
    use_bess = st.checkbox("Baterie (BESS)", value=True)
    use_fve = st.checkbox("Fotovoltaika (FVE)", value=True)
    use_ext_heat = st.checkbox("Nákup tepla (Import)", value=True)

# --- 2. GRAF CEN (POROVNÁNÍ) ---
if st.session_state.fwd_data is not None:
    with st.expander("📊 Náhled upravených tržních cen", expanded=True):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
        fig.add_trace(go.Scatter(x=st.session_state.fwd_data['datetime'], y=st.session_state.fwd_data['ee_original'], name="EE Původní", line=dict(color='rgba(0,255,0,0.2)', dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=st.session_state.fwd_data['datetime'], y=st.session_state.fwd_data['ee_price'], name="EE Upravená", line=dict(color='green')), row=1, col=1)
        fig.add_trace(go.Scatter(x=st.session_state.fwd_data['datetime'], y=st.session_state.fwd_data['gas_original'], name="Plyn Původní", line=dict(color='rgba(255,0,0,0.2)', dash='dot')), row=2, col=1)
        fig.add_trace(go.Scatter(x=st.session_state.fwd_data['datetime'], y=st.session_state.fwd_data['gas_price'], name="Plyn Upravená", line=dict(color='red')), row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

# --- 3. PARAMETRY ---
t_tech, t_eco, t_acc = st.tabs(["Technika", "Ekonomika", "Akumulace"])
p = {}
with t_tech:
    c1, c2 = st.columns(2)
    with c1:
        p['k_th'] = st.number_input("KGJ Tepelný výkon [MW]", value=1.09)
        p['k_el'] = st.number_input("KGJ Elektrický výkon [MW]", value=1.0)
        p['k_eff_th'] = st.number_input("KGJ Tepelná účinnost", value=0.46)
        p['k_min'] = st.slider("Min. zatížení KGJ [%]", 0, 100, 55) / 100
    with c2:
        p['b_max'] = st.number_input("Plynový kotel max [MW]", value=3.91)
        p['ek_max'] = st.number_input("Elektrokotel max [MW]", value=0.61)

with t_eco:
    c1, c2 = st.columns(2)
    with c1:
        p['dist_ee_buy'] = st.number_input("Distribuce nákup EE [EUR/MWh]", value=33.0)
        p['dist_ee_sell'] = st.number_input("Distribuce prodej EE [EUR/MWh]", value=2.0)
        p['gas_dist'] = st.number_input("Distribuce plyn [EUR/MWh]", value=5.0)
    with c2:
        p['h_price'] = st.number_input("Cena tepla [EUR/MWh]", value=120.0)
        p['h_cover'] = st.slider("Pokrytí poptávky", 0.0, 1.0, 0.99)

with t_acc:
    c1, c2 = st.columns(2)
    with c1:
        p['tes_cap'] = st.number_input("Nádrž kapacita [MWh]", value=10.0)
        p['tes_loss'] = st.number_input("Ztráta nádrže [%/h]", value=0.5) / 100
    with c2:
        p['bess_cap'] = st.number_input("BESS kapacita [MWh]", value=1.0)
        p['bess_p'] = st.number_input("BESS výkon [MW]", value=0.5)

# --- 4. VÝPOČET ---
st.divider()
loc_file = st.file_uploader("3️⃣ Nahraj lokální data (aki11)", type=["xlsx"])

if st.session_state.fwd_data is not None and loc_file:
    df_loc = pd.read_excel(loc_file)
    df_loc.columns = [str(c).strip() for c in df_loc.columns]
    df_loc.rename(columns={df_loc.columns[0]: 'datetime'}, inplace=True)
    df_loc['datetime'] = pd.to_datetime(df_loc['datetime'], dayfirst=True)
    
    # Merge podle času (řeší tvůj problém s importem)
    df = pd.merge(st.session_state.fwd_data, df_loc, on='datetime', how='inner')
    T = len(df)

    if st.button("🏁 SPUSTIT KOMPLETNÍ OPTIMALIZACI"):
        model = pulp.LpProblem("Dispatcher_Complex", pulp.LpMaximize)
        
        # Proměnné
        q_kgj = pulp.LpVariable.dicts("q_KGJ", range(T), 0)
        q_boil = pulp.LpVariable.dicts("q_Boil", range(T), 0, p['b_max'])
        q_ek = pulp.LpVariable.dicts("q_EK", range(T), 0, p['ek_max'])
        q_imp = pulp.LpVariable.dicts("q_Imp", range(T), 0)
        on = pulp.LpVariable.dicts("on", range(T), 0, 1, cat="Binary")
        
        tes_soc = pulp.LpVariable.dicts("TES_SOC", range(T+1), 0, p['tes_cap'])
        bess_soc = pulp.LpVariable.dicts("BESS_SOC", range(T+1), 0, p['bess_cap'])
        bess_cha = pulp.LpVariable.dicts("BESS_Cha", range(T), 0, p['bess_p'])
        bess_dis = pulp.LpVariable.dicts("BESS_Dis", range(T), 0, p['bess_p'])
        
        ee_export = pulp.LpVariable.dicts("ee_export", range(T), 0)
        ee_import = pulp.LpVariable.dicts("ee_import", range(T), 0)

        obj = []
        model += tes_soc[0] == p['tes_cap'] *

import streamlit as st
import pandas as pd
import pulp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="KGJ Strategy Expert PRO", layout="wide")

if 'fwd_data' not in st.session_state: st.session_state.fwd_data = None

st.title("🚀 KGJ Strategy & Dispatch Optimizer PRO")

# --- 1. SIDEBAR: TVOJE ORIGINÁLNÍ FWD LOGIKA ---
with st.sidebar:
    st.header("📈 1. Tržní ceny (FWD)")
    fwd_file = st.file_uploader("Nahraj FWD křivku (Excel)", type=["xlsx"])
    
    if fwd_file:
        df_raw = pd.read_excel(fwd_file)
        df_raw.columns = [str(c).strip() for c in df_raw.columns]
        # Robustní převod data
        df_raw.iloc[:, 0] = pd.to_datetime(df_raw.iloc[:, 0], dayfirst=True, errors='coerce')
        df_raw = df_raw.dropna(subset=[df_raw.columns[0]])
        
        years = sorted(df_raw.iloc[:, 0].dt.year.unique())
        sel_year = st.selectbox("Rok pro analýzu", years)
        df_year = df_raw[df_raw.iloc[:, 0].dt.year == sel_year].copy()
        
        # Původní výpočet průměrů a shiftu
        avg_ee_raw = float(df_year.iloc[:, 1].mean())
        avg_gas_raw = float(df_year.iloc[:, 2].mean())
        
        st.subheader("🛠️ Úprava na aktuální trh")
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
    use_ext_heat = st.checkbox("Nákup tepla (Import)", value=False)

# --- 2. PARAMETRY (PŮVODNÍ NASTAVENÍ) ---
t_tech, t_eco, t_acc = st.tabs(["🏗️ Technika", "💰 Ekonomika", "🔋 Akumulace"])
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
        p['ext_h_price'] = st.number_input("Cena importu tepla [EUR/MWh]", value=250.0)

with t_acc:
    c1, c2 = st.columns(2)
    with c1:
        p['tes_cap'] = st.number_input("Nádrž kapacita [MWh]", value=10.0)
        p['tes_loss'] = st.number_input("Ztráta nádrže [%/h]", value=0.5) / 100
    with c2:
        p['bess_cap'] = st.number_input("BESS kapacita [MWh]", value=1.0)
        p['bess_p'] = st.number_input("BESS výkon [MW]", value=0.5)

# --- 3. VÝPOČET ---
st.divider()
loc_file = st.file_uploader("3️⃣ Nahraj lokální data (aki11)", type=["xlsx"])

if st.session_state.fwd_data is not None and loc_file:
    df_loc = pd.read_excel(loc_file).fillna(0)
    df_loc.columns = [str(c).strip() for c in df_loc.columns]
    df_loc.rename(columns={df_loc.columns[0]: 'datetime'}, inplace=True)
    df_loc['datetime'] = pd.to_datetime(df_loc['datetime'], dayfirst=True, errors='coerce')
    
    df = pd.merge(st.session_state.fwd_data, df_loc, on='datetime', how='inner')
    T = len(df)

    if st.button("🏁 SPUSTIT OPTIMALIZACI"):
        model = pulp.LpProblem("Energy_Optimizer", pulp.LpMaximize)
        
        # Proměnné
        q_kgj = pulp.LpVariable.dicts("q_KGJ", range(T), 0)
        q_boil = pulp.LpVariable.dicts("q_Boil", range(T), 0, p['b_max'])
        q_ek = pulp.LpVariable.dicts("q_EK", range(T), 0, p['ek_max'])
        q_imp = pulp.LpVariable.dicts("q_Imp", range(T), 0)
        unserved = pulp.LpVariable.dicts("unserved", range(T), 0)
        on = pulp.LpVariable.dicts("on", range(T), 0, 1, cat="Binary")
        
        tes_soc = pulp.LpVariable.dicts("TES_SOC", range(T+1), 0, p['tes_cap'])
        ee_export = pulp.LpVariable.dicts("ee_export", range(T), 0)

        obj = []
        for t in range(T):
            h_dem = float(df['Poptávka po teple (MW)'].iloc[t])
            p_ee = float(df['ee_price'].iloc[t])
            p_gas = float(df['gas_price'].iloc[t])
            
            # Bilance tepla
            model += q_kgj[t] + q_boil[t] + q_ek[t] + q_imp[t] + unserved[t] + (tes_soc[t]*(1-p['tes_loss']) - tes_soc[t+1]) >= h_dem
            model += q_kgj[t] <= p['k_th'] * on[t]
            model += q_kgj[t] >= p['k_min'] * p['k_th'] * on[t]

            # Cashflow
            revenue = (p['h_price'] * (h_dem - unserved[t])) + (p_ee - p['dist_ee_sell']) * (q_kgj[t] * (p['k_el']/p['k_th']))
            costs = (p_gas + p['gas_dist']) * (q_kgj[t]/p['k_eff_th'] + q_boil[t]/0.95) + (unserved[t] * 5000) + (q_imp[t] * p['ext_h_price'])
            obj.append(revenue - costs)

        model += pulp.lpSum(obj)
        model.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=120))

        # --- SEXY VÝSTUPY ---
        res = pd.DataFrame({
            'datetime': df['datetime'],
            'Poptávka': df['Poptávka po teple (MW)'],
            'KGJ': [q_kgj[t].value() for t in range(T)],
            'Kotel': [q_boil[t].value() for t in range(T)],
            'Nepokryto': [unserved[t].value() for t in range(T)],
            'Zisk': [pulp.value(obj[t]) for t in range(T)]
        })

        st.subheader("📊 Dispatch tepla")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res['datetime'], y=res['KGJ'], name="KGJ", stackgroup='one', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=res['datetime'], y=res['Kotel'], name="Kotel", stackgroup='one', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=res['datetime'], y=res['Nepokryto'], name="NEPOKRYTO", stackgroup='one', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=res['datetime'], y=res['Poptávka'], name="Poptávka", line=dict(color='black', dash='dot')))
        st.plotly_chart(fig, use_container_width=True)

        st.download_button("📥 Export CSV", res.to_csv(index=False).encode('utf-8'), "vysledky.csv")

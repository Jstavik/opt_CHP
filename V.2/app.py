import streamlit as st
import pandas as pd
import pulp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="KGJ Strategy Optimizer PRO", layout="wide")

# Inicializace stavu
if 'fwd_data' not in st.session_state: st.session_state.fwd_data = None

st.title("🚀 KGJ Strategy & Dispatch Optimizer PRO")

# --- 1. SIDEBAR: CENY A AKTIVACE ZDROJŮ ---
with st.sidebar:
    st.header("📈 1. Tržní ceny (FWD)")
    fwd_file = st.file_uploader("Nahraj FWD křivku (Excel)", type=["xlsx"])
    
    if fwd_file:
        df_raw = pd.read_excel(fwd_file)
        df_raw.columns = [str(c).strip() for c in df_raw.columns]
        # OPRAVA ATTRIBUTERROR: Převod na datetime a vyhození bordelu
        df_raw.iloc[:, 0] = pd.to_datetime(df_raw.iloc[:, 0], dayfirst=True, errors='coerce')
        df_raw = df_raw.dropna(subset=[df_raw.columns[0]])
        
        years = sorted(df_raw.iloc[:, 0].dt.year.unique())
        sel_year = st.selectbox("Rok pro analýzu", years)
        df_year = df_raw[df_raw.iloc[:, 0].dt.year == sel_year].copy()
        df_year.columns = ['datetime', 'ee_original', 'gas_original'] + list(df_year.columns[3:])
        
        # TVOJE LOGIKA VÝPOČTU SHIFTU
        avg_ee_raw = float(df_year['ee_original'].mean())
        avg_gas_raw = float(df_year['gas_original'].mean())
        
        st.info(f"Původní průměr: EE {avg_ee_raw:.2f} | Plyn {avg_gas_raw:.2f}")
        ee_target = st.number_input("Nová cílová cena EE", value=avg_ee_raw)
        gas_target = st.number_input("Nová cílová cena Plyn", value=avg_gas_raw)
        
        ee_shift = ee_target - avg_ee_raw
        gas_shift = gas_target - avg_gas_raw
        
        df_year['ee_price'] = df_year['ee_original'] + ee_shift
        df_year['gas_price'] = df_year['gas_original'] + gas_shift
        st.session_state.fwd_data = df_year

    st.divider()
    st.header("⚙️ 2. Aktivní technologie")
    use_kgj = st.checkbox("Kogenerace (KGJ)", value=True)
    use_boil = st.checkbox("Plynový kotel", value=True)
    use_ek = st.checkbox("Elektrokotel", value=True)
    use_tes = st.checkbox("Nádrž (TES)", value=True)
    use_bess = st.checkbox("Baterie (BESS)", value=True)
    use_fve = st.checkbox("Fotovoltaika (FVE)", value=True)
    use_ext_heat = st.checkbox("Povolit import tepla", value=False)

# --- 2. PARAMETRY ---
t_tech, t_eco, t_acc = st.tabs(["🏗️ Technika", "💰 Ekonomika", "🔋 Akumulace"])
p = {}
with t_tech:
    c1, c2 = st.columns(2)
    with c1:
        p['k_th'] = st.number_input("KGJ Tepelný výkon [MW]", value=1.09)
        p['k_el'] = st.number_input("KGJ Elektrický výkon [MW]", value=1.0)
        p['k_eff_th'] = st.number_input("KGJ Tepelná účinnost", value=0.46)
        p['k_min'] = st.slider("Min. zatížení KGJ %", 0, 100, 55) / 100
    with c2:
        p['b_max'] = st.number_input("Plynový kotel max [MW]", value=3.91)
        p['ek_max'] = st.number_input("Elektrokotel max [MW]", value=0.61)
        p['ek_eff'] = 0.96

with t_eco:
    p['h_price'] = st.number_input("Prodejní cena tepla [EUR/MWh]", value=120.0)
    p['start_cost'] = st.number_input("Náklad na start KGJ [EUR]", value=150.0)
    p['ext_h_price'] = st.number_input("Cena importu tepla [EUR/MWh]", value=250.0)
    p['dist_ee_sell'] = 2.0
    p['gas_dist'] = 5.0

with t_acc:
    p['tes_cap'] = st.number_input("TES Kapacita [MWh]", value=10.0)
    p['tes_loss'] = 0.005 # 0.5% za hodinu
    p['bess_cap'] = st.number_input("BESS Kapacita [MWh]", value=1.0)
    p['bess_p'] = st.number_input("BESS Výkon [MW]", value=0.5)

# --- 3. VÝPOČET ---
st.divider()
loc_file = st.file_uploader("3️⃣ Nahraj lokální data (aki11)", type=["xlsx"])

if st.session_state.fwd_data is not None and loc_file:
    df_loc = pd.read_excel(loc_file).fillna(0)
    df_loc.columns = [str(c).strip() for c in df_loc.columns]
    df_loc.rename(columns={df_loc.columns[0]: 'datetime'}, inplace=True)
    df_loc['datetime'] = pd.to_datetime(df_loc['datetime'], dayfirst=True, errors='coerce')
    df_loc = df_loc.dropna(subset=['datetime'])
    
    df = pd.merge(st.session_state.fwd_data, df_loc, on='datetime', how='inner')
    T = len(df)

    if st.button("🚀 SPUSTIT OPTIMALIZACI"):
        with st.spinner('Počítám roční dispatch...'):
            model = pulp.LpProblem("Strategy_Optimizer", pulp.LpMaximize)
            
            # Proměnné
            q_kgj = pulp.LpVariable.dicts("q_KGJ", range(T), 0)
            on = pulp.LpVariable.dicts("on", range(T), 0, 1, cat="Binary")
            st_kgj = pulp.LpVariable.dicts("start", range(T), 0, 1, cat="Binary")
            q_boil = pulp.LpVariable.dicts("q_Boil", range(T), 0, p['b_max'])
            q_ek = pulp.LpVariable.dicts("q_EK", range(T), 0, p['ek_max'])
            q_imp = pulp.LpVariable.dicts("q_Imp", range(T), 0)
            unserved = pulp.LpVariable.dicts("Unserved", range(T), 0)
            
            tes_soc = pulp.LpVariable.dicts("TES_SOC", range(T+1), 0, p['tes_cap'])
            bess_soc = pulp.LpVariable.dicts("BESS_SOC", range(T+1), 0, p['bess_cap'])
            b_cha = pulp.LpVariable.dicts("B_Cha", range(T), 0, p['bess_p'])
            b_dis = pulp.LpVariable.dicts("B_Dis", range(T), 0, p['bess_p'])
            ee_export = pulp.LpVariable.dicts("ee_export", range(T), 0)

            obj = []
            for t in range(T):
                h_dem = float(df['Poptávka po teple (MW)'].iloc[t])
                p_ee = float(df['ee_price'].iloc[t])
                p_gas = float(df['gas_price'].iloc[t])
                fve = float(df['FVE (MW)'].iloc[t]) if 'FVE (MW)' in df.columns and use_fve else 0.0

                # Logika Startů
                if t > 0: model += st_kgj[t] >= on[t] - on[t-1]
                
                # Bilance tepla
                # Pokud není technologie zakliknuta, její výkon musí být 0
                if not use_kgj: model += q_kgj[t] == 0
                if not use_boil: model += q_boil[t] == 0
                if not use_ek: model += q_ek[t] == 0
                if not use_ext_heat: model += q_imp[t] == 0
                if not use_tes: model += tes_soc[t] == 0; model += tes_soc[t+1] == 0
                
                heat_gen = q_kgj[t] + q_boil[t] + q_ek[t] + q_imp[t]
                tes_diff = (tes_soc[t]*(1-p['tes_loss']) - tes_soc[t+1]) if use_tes else 0
                model += heat_gen + tes_diff + unserved[t] >= h_dem
                
                # KGJ Limity
                model += q_kgj[t] <= p['k_th'] * on[t]
                model += q_kgj[t] >= p['k_min'] * p['k_th'] * on[t]

                # Bilance EE
                ee_kgj = q_kgj[t] * (p['k_el'] / p['k_th'])
                if use_bess:
                    model += bess_soc[t+1] == bess_soc[t] + b_cha[t]*0.9 - b_dis[t]/0.9
                else:
                    model += b_cha[t] == 0; model += b_dis[t] == 0
                
                model += ee_kgj + fve + b_dis[t] == (q_ek[t]/p['ek_eff']) + b_cha[t] + ee_export[t]

                # Ekonomika
                rev = (p['h_price'] * (h_dem - unserved[t])) + (p_ee - p['dist_ee_sell']) * ee_export[t]
                cost = (p_gas + p['gas_dist']) * (q_kgj[t]/p['k_eff_th'] + q_boil[t]/0.95) + \
                       (st_kgj[t] * p['start_cost']) + (unserved[t] * 5000) + (q_imp[t] * p['ext_h_price'])
                obj.append(rev - cost)

            model += pulp.lpSum(obj)
            model.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=100))

        # --- VÝSTUPY ---
        res = pd.DataFrame({
            'datetime': df['datetime'],
            'Poptávka': df['Poptávka po teple (MW)'],
            'KGJ': [q_kgj[t].value() for t in range(T)],
            'Kotel': [q_boil[t].value() for t in range(T)],
            'EK': [q_ek[t].value() for t in range(T)],
            'Import': [q_imp[t].value() for t in range(T)],
            'Unserved': [unserved[t].value() for t in range(T)],
            'Zisk_h': [pulp.value(obj[t]) for t in range(T)]
        })

        st.subheader("📊 Dispatch tepla")
        fig = go.Figure()
        colors = {'KGJ': 'orange', 'Kotel': 'blue', 'EK': 'green', 'Import': 'purple', 'Unserved': 'red'}
        for col in ['KGJ', 'Kotel', 'EK', 'Import', 'Unserved']:
            fig.add_trace(go.Scatter(x=res['datetime'], y=res[col], name=col, stackgroup='one', line=dict(color=colors[col])))
        fig.add_trace(go.Scatter(x=res['datetime'], y=res['Poptávka'], name="Poptávka", line=dict(color='black', dash='dot')))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("💰 Kumulativní Cashflow")
        res['cum_profit'] = res['Zisk_h'].cumsum()
        st.plotly_chart(go.Figure(go.Scatter(x=res['datetime'], y=res['cum_profit'], fill='tozeroy', name="EUR")), use_container_width=True)
        
        st.download_button("📥 Stáhnout CSV", res.to_csv(index=False).encode('utf-8'), "vysledky.csv")

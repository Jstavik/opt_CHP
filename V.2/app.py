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

# --- 2. GRAF CEN ---
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
        p['k_start_cost'] = st.number_input("Náklady na start KGJ [EUR/start]", value=1200.0)
        p['k_min_runtime'] = st.number_input("Min. doba běhu KGJ [hod]", value=4, min_value=1, step=1)
    with c2:
        p['b_max'] = st.number_input("Plynový kotel max [MW]", value=3.91)
        p['ek_max'] = st.number_input("Elektrokotel max [MW]", value=0.61)
        p['imp_max'] = st.number_input("Import tepla max [MW]", value=2.0) if use_ext_heat else 0.0

with t_eco:
    c1, c2 = st.columns(2)
    with c1:
        p['dist_ee_buy'] = st.number_input("Distribuce nákup EE [EUR/MWh]", value=33.0)
        p['dist_ee_sell'] = st.number_input("Distribuce prodej EE [EUR/MWh]", value=2.0)
        p['gas_dist'] = st.number_input("Distribuce plyn [EUR/MWh]", value=5.0)
    with c2:
        p['h_price'] = st.number_input("Cena tepla [EUR/MWh]", value=120.0)
        p['h_cover'] = st.slider("Pokrytí poptávky", 0.0, 1.0, 0.99)
        p['imp_price'] = st.number_input("Cena importu tepla [EUR/MWh]", value=150.0) if use_ext_heat else 0.0

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
    
    # Merge a vyčištění
    df = pd.merge(st.session_state.fwd_data, df_loc, on='datetime', how='inner').fillna(0)
    T = len(df)

    if st.button("🏁 SPUSTIT KOMPLETNÍ OPTIMALIZACI"):
        model = pulp.LpProblem("Dispatcher_PRO", pulp.LpMaximize)
        
        # Proměnné
        q_kgj = pulp.LpVariable.dicts("q_KGJ", range(T), 0)
        q_boil = pulp.LpVariable.dicts("q_Boil", range(T), 0, p['b_max'])
        q_ek = pulp.LpVariable.dicts("q_EK", range(T), 0, p['ek_max'])
        q_imp = pulp.LpVariable.dicts("q_Imp", range(T), 0, p['imp_max'] if use_ext_heat else 0)
        on = pulp.LpVariable.dicts("on", range(T), 0, 1, cat="Binary")
        start = pulp.LpVariable.dicts("start", range(T), 0, 1, cat="Binary")
        stop = pulp.LpVariable.dicts("stop", range(T), 0, 1, cat="Binary")
        
        tes_soc = pulp.LpVariable.dicts("TES_SOC", range(T+1), 0, p['tes_cap'])
        tes_in = pulp.LpVariable.dicts("TES_In", range(T), 0)
        tes_out = pulp.LpVariable.dicts("TES_Out", range(T), 0)
        
        bess_soc = pulp.LpVariable.dicts("BESS_SOC", range(T+1), 0, p['bess_cap'])
        bess_cha = pulp.LpVariable.dicts("BESS_Cha", range(T), 0, p['bess_p'])
        bess_dis = pulp.LpVariable.dicts("BESS_Dis", range(T), 0, p['bess_p'])
        
        ee_export = pulp.LpVariable.dicts("ee_export", range(T), 0)
        ee_import = pulp.LpVariable.dicts("ee_import", range(T), 0)
        
        heat_shortfall = pulp.LpVariable.dicts("heat_shortfall", range(T), 0)

        # Počáteční stavy
        model += tes_soc[0] == p['tes_cap'] * 0.5
        model += bess_soc[0] == p['bess_cap'] * 0.2

        # Logika start/stop a min runtime
        for t in range(T):
            model += on[t] <= 1 if use_kgj else 0
            if t > 0:
                model += on[t] - on[t-1] == start[t] - stop[t]
                model += start[t] + stop[t] <= 1
        
        # Min runtime: po startu musí běžet min. hodin
        for t in range(T - p['k_min_runtime'] + 1):
            for k in range(1, p['k_min_runtime']):
                model += on[t + k] >= on[t] - stop[t] if use_kgj else 0  # Zabrání předčasnému stopu

        obj = []
        for t in range(T):
            p_ee = float(df['ee_price'].iloc[t])
            p_gas = float(df['gas_price'].iloc[t])
            h_dem = float(df['Poptávka po teple (MW)'].iloc[t])
            fve = float(df['FVE (MW)'].iloc[t]) if use_fve else 0.0

            # Bilance tepla: výroba + import + TES_out - TES_in = dodané + shortfall
            heat_produced = q_kgj[t] + q_boil[t] + q_ek[t] + q_imp[t]
            model += tes_soc[t+1] == tes_soc[t] * (1 - p['tes_loss']) + tes_in[t] - tes_out[t]
            model += heat_produced + tes_out[t] == tes_in[t] + heat_delivered  # Definuj heat_delivered jako novou var?
            wait, opravím: přidej heat_delivered = pulp.LpVariable(...)
            Ne, lepší: 
            model += heat_produced + tes_out[t] - tes_in[t] + heat_shortfall[t] >= h_dem * p['h_cover']
            model += heat_produced + tes_out[t] - tes_in[t] <= h_dem * p['h_cover']  # Nelze dodávat víc než poptávka, přebytek jen do TES (ale TES_in je omezen kapacitou)

            # KGJ omezení
            model += q_kgj[t] <= p['k_th'] * on[t]
            model += q_kgj[t] >= p['k_min'] * p['k_th'] * on[t]

            # Bilance EE (beze změn)
            ee_kgj = q_kgj[t] * (p['k_el'] / p['k_th']) if use_kgj else 0
            model += ee_kgj + fve + ee_import[t] + bess_dis[t] == (q_ek[t] / 0.95) + bess_cha[t] + ee_export[t]  # Snížil jsem eff na 0.95
            model += bess_soc[t+1] == bess_soc[t] + (bess_cha[t] * 0.90) - (bess_dis[t] / 0.90)  # Realističtější eff ~81% round-trip

            # Finance
            heat_delivered = heat_produced + tes_out[t] - tes_in[t]
            revenue = p['h_price'] * pulp.lpMin(heat_delivered, h_dem * p['h_cover']) + (p_ee - p['dist_ee_sell']) * ee_export[t]
            costs = (p_gas + p['gas_dist']) * (q_kgj[t]/p['k_eff_th'] + q_boil[t]/0.95) + \
                    (p_ee + p['dist_ee_buy']) * ee_import[t] + \
                    p['k_start_cost'] * start[t] + \
                    p['imp_price'] * q_imp[t]
            obj.append(revenue - costs - p['h_price'] * heat_shortfall[t])  # Penalizace shortfall stejně jako příjem

        model += pulp.lpSum(obj)
        model.solve(pulp.PULP_CBC_CMD(msg=0))

        # --- 5. VÝSLEDKY A GRAFY ---
        st.success("Optimalizace dokončena! Status: " + pulp.LpStatus[model.status])

        # Sbírání dat pro výstup
        results = {
            'datetime': df['datetime'],
            'h_dem': df['Poptávka po teple (MW)'],
            'q_kgj': [pulp.value(q_kgj[t]) for t in range(T)],
            'q_boil': [pulp.value(q_boil[t]) for t in range(T)],
            'q_ek': [pulp.value(q_ek[t]) for t in range(T)],
            'q_imp': [pulp.value(q_imp[t]) for t in range(T)],
            'tes_in': [pulp.value(tes_in[t]) for t in range(T)],
            'tes_out': [pulp.value(tes_out[t]) for t in range(T)],
            'heat_shortfall': [pulp.value(heat_shortfall[t]) for t in range(T)],
            'tes_soc': [pulp.value(tes_soc[t+1]) for t in range(T)],  # Konečné SOC po každé hodině
            'ee_export': [pulp.value(ee_export[t]) for t in range(T)],
            'ee_import': [pulp.value(ee_import[t]) for t in range(T)],
            'profit': [revenue - costs for revenue, costs in zip(...) ]  # Přidej výpočet
        }
        df_results = pd.DataFrame(results)

        # Celkové metriky
        total_profit = df_results['profit'].sum()
        total_shortfall = df_results['heat_shortfall'].sum()
        avg_coverage = 1 - (total_shortfall / df_results['h_dem'].sum())
        st.subheader("📈 Klíčové metriky")
        col1, col2, col3 = st.columns(3)
        col1.metric("Celkový zisk [EUR]", f"{total_profit:.2f}")
        col2.metric("Celkový shortfall [MWh]", f"{total_shortfall:.2f}")
        col3.metric("Průměrné pokrytí [%]", f"{avg_coverage*100:.1f}")

        # Graf 1: Stackplot pokrytí poptávky
        st.subheader("🔥 Pokrytí poptávky po teple")
        fig_heat = go.Figure()
        fig_heat.add_trace(go.Scatter(x=df_results['datetime'], y=df_results['q_kgj'], name="KGJ", stackgroup='one', fill='tonexty'))
        fig_heat.add_trace(go.Scatter(x=df_results['datetime'], y=df_results['q_boil'], name="Kotel", stackgroup='one', fill='tonexty'))
        fig_heat.add_trace(go.Scatter(x=df_results['datetime'], y=df_results['q_ek'], name="EK", stackgroup='one', fill='tonexty'))
        fig_heat.add_trace(go.Scatter(x=df_results['datetime'], y=df_results['q_imp'], name="Import", stackgroup='one', fill='tonexty'))
        fig_heat.add_trace(go.Scatter(x=df_results['datetime'], y=df_results['tes_out'] - df_results['tes_in'], name="TES (netto)", stackgroup='one', fill='tonexty'))
        fig_heat.add_trace(go.Scatter(x=df_results['datetime'], y=df_results['heat_shortfall'], name="Shortfall", stackgroup='one', fill='tonexty', line=dict(color='black', dash='dot')))
        fig_heat.add_trace(go.Scatter(x=df_results['datetime'], y=df_results['h_dem'] * p['h_cover'], name="Cílová poptávka", mode='lines', line=dict(color='red', width=2)))
        st.plotly_chart(fig_heat, use_container_width=True)

        # Graf 2: TES a BESS SOC
        st.subheader("🔋 Stav akumulátorů")
        fig_acc = make_subplots(rows=2, cols=1, shared_xaxes=True)
        fig_acc.add_trace(go.Scatter(x=df_results['datetime'], y=df_results['tes_soc'], name="TES SOC [MWh]"), row=1, col=1)
        fig_acc.add_trace(go.Scatter(x=df_results['datetime'], y=[p['tes_cap']] * T, name="TES Max", line=dict(dash='dot')), row=1, col=1)
        # Přidej BESS podobně...
        st.plotly_chart(fig_acc, use_container_width=True)

        # Graf 3: EE bilance (export/import)
        st.subheader("⚡ Elektrická bilance")
        fig_ee = go.Figure()
        fig_ee.add_trace(go.Bar(x=df_results['datetime'], y=df_results['ee_export'], name="Export EE"))
        fig_ee.add_trace(go.Bar(x=df_results['datetime'], y=-df_results['ee_import'], name="Import EE"))
        st.plotly_chart(fig_ee, use_container_width=True)

        # Tabulka s detaily (prvních 24h)
        st.subheader("📋 Ukázka výsledků (prvních 24 hodin)")
        st.dataframe(df_results.head(24))

import streamlit as st
import pandas as pd
import pulp
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(page_title="KGJ Strategy Expert PRO", layout="wide")

# Inicializace session state
if 'fwd_data' not in st.session_state:
    st.session_state.fwd_data = None

st.title("🚀 KGJ Strategy & Dispatch Optimizer PRO")

# ────────────────────────────────────────────────
# SIDEBAR – Ceny + technologie
# ────────────────────────────────────────────────
with st.sidebar:
    st.header("📈 1. Tržní ceny (FWD)")
    fwd_file = st.file_uploader("Nahraj FWD křivku (Excel)", type=["xlsx"])
    
    if fwd_file is not None:
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
        
        ee_market_new = st.number_input("Nová cílová cena EE [EUR/MWh]", value=avg_ee_raw)
        gas_market_new = st.number_input("Nová cílová cena Plyn [EUR/MWh]", value=avg_gas_raw)
        
        ee_shift = ee_market_new - avg_ee_raw
        gas_shift = gas_market_new - avg_gas_raw
        
        df_fwd = df_year.copy()
        df_fwd.columns = ['datetime', 'ee_original', 'gas_original']
        df_fwd['ee_price'] = df_fwd['ee_original'] + ee_shift
        df_fwd['gas_price'] = df_fwd['gas_original'] + gas_shift
        st.session_state.fwd_data = df_fwd

    st.divider()
    st.header("⚙️ 2. Aktivní technologie")
    use_kgj     = st.checkbox("Kogenerace (KGJ)", value=True)
    use_boil    = st.checkbox("Plynový kotel", value=True)
    use_ek      = st.checkbox("Elektrokotel", value=True)
    use_tes     = st.checkbox("Nádrž (TES)", value=True)
    use_bess    = st.checkbox("Baterie (BESS)", value=True)
    use_fve     = st.checkbox("Fotovoltaika (FVE)", value=True)
    use_ext_heat = st.checkbox("Nákup tepla (Import)", value=True)

# ────────────────────────────────────────────────
# GRAF CEN (pokud máme data)
# ────────────────────────────────────────────────
if st.session_state.fwd_data is not None:
    with st.expander("📊 Náhled upravených tržních cen", expanded=False):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=("Elektřina", "Plyn"))
        fig.add_trace(go.Scatter(x=st.session_state.fwd_data['datetime'], y=st.session_state.fwd_data['ee_original'],
                                 name="EE původní", line=dict(color='green', dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=st.session_state.fwd_data['datetime'], y=st.session_state.fwd_data['ee_price'],
                                 name="EE upravená", line=dict(color='darkgreen')), row=1, col=1)
        fig.add_trace(go.Scatter(x=st.session_state.fwd_data['datetime'], y=st.session_state.fwd_data['gas_original'],
                                 name="Plyn původní", line=dict(color='red', dash='dot')), row=2, col=1)
        fig.add_trace(go.Scatter(x=st.session_state.fwd_data['datetime'], y=st.session_state.fwd_data['gas_price'],
                                 name="Plyn upravený", line=dict(color='darkred')), row=2, col=1)
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

# ────────────────────────────────────────────────
# PARAMETRY – záložky
# ────────────────────────────────────────────────
t_tech, t_eco, t_acc = st.tabs(["Technika", "Ekonomika", "Akumulace"])
p = {}

with t_tech:
    c1, c2 = st.columns(2)
    with c1:
        p['k_th'] = st.number_input("KGJ Tepelný výkon [MW]", value=1.09, step=0.01)
        p['k_el']  = st.number_input("KGJ Elektrický výkon [MW]", value=1.0, step=0.01)
        p['k_eff_th'] = st.number_input("KGJ Tepelná účinnost", value=0.46, step=0.01)
        p['k_min'] = st.slider("Min. zatížení KGJ [%]", 0, 100, 55) / 100
        p['k_start_cost'] = st.number_input("Náklady na start KGJ [€/start]", value=1200.0, step=100.0)
        p['k_min_runtime'] = st.number_input("Min. doba běhu KGJ [hod]", value=4, min_value=1, step=1)
    with c2:
        p['b_max'] = st.number_input("Plynový kotel max [MW]", value=3.91, step=0.1)
        p['ek_max'] = st.number_input("Elektrokotel max [MW]", value=0.61, step=0.1)
        p['imp_max'] = st.number_input("Max. import tepla [MW]", value=2.0, step=0.1) if use_ext_heat else 0.0

with t_eco:
    c1, c2 = st.columns(2)
    with c1:
        p['dist_ee_buy']  = st.number_input("Distribuce nákup EE [€/MWh]", value=33.0)
        p['dist_ee_sell'] = st.number_input("Distribuce prodej EE [€/MWh]", value=2.0)
        p['gas_dist']     = st.number_input("Distribuce plyn [€/MWh]", value=5.0)
    with c2:
        p['h_price']   = st.number_input("Cena tepla [€/MWh]", value=120.0)
        p['h_cover']   = st.slider("Minimální pokrytí poptávky", 0.0, 1.0, 0.99, step=0.01)
        p['imp_price'] = st.number_input("Cena importu tepla [€/MWh]", value=150.0) if use_ext_heat else 0.0

with t_acc:
    c1, c2 = st.columns(2)
    with c1:
        p['tes_cap']  = st.number_input("Nádrž kapacita [MWh]", value=10.0, step=1.0)
        p['tes_loss'] = st.number_input("Ztráta nádrže [%/h]", value=0.5) / 100
    with c2:
        p['bess_cap'] = st.number_input("BESS kapacita [MWh]", value=1.0, step=0.1)
        p['bess_p']   = st.number_input("BESS výkon [MW]", value=0.5, step=0.1)

# ────────────────────────────────────────────────
# NAHRÁNÍ LOKÁLNÍCH DAT + TLAČÍTKO
# ────────────────────────────────────────────────
st.divider()
loc_file = st.file_uploader("Nahraj lokální data (poptávka, FVE, ...)", type=["xlsx"])

if st.session_state.fwd_data is not None and loc_file is not None:
    df_loc = pd.read_excel(loc_file)
    df_loc.columns = [str(c).strip() for c in df_loc.columns]
    df_loc.rename(columns={df_loc.columns[0]: 'datetime'}, inplace=True)
    df_loc['datetime'] = pd.to_datetime(df_loc['datetime'], dayfirst=True)
    
    df = pd.merge(st.session_state.fwd_data, df_loc, on='datetime', how='inner').fillna(0)
    T = len(df)

    if st.button("🏁 Spustit optimalizaci", type="primary"):
        with st.spinner("Běží optimalizace (může trvat 30 s – 3 min)..."):
            model = pulp.LpProblem("KGJ_Dispatcher", pulp.LpMaximize)

            # Proměnné
            q_kgj   = pulp.LpVariable.dicts("q_KGJ",   range(T), lowBound=0)
            q_boil  = pulp.LpVariable.dicts("q_Boil",  range(T), 0, p['b_max'])
            q_ek    = pulp.LpVariable.dicts("q_EK",    range(T), 0, p['ek_max'])
            q_imp   = pulp.LpVariable.dicts("q_Imp",   range(T), 0, p['imp_max'] if use_ext_heat else 0)
            on      = pulp.LpVariable.dicts("on",      range(T), 0, 1, cat="Binary")
            start   = pulp.LpVariable.dicts("start",   range(T), 0, 1, cat="Binary")

            tes_soc = pulp.LpVariable.dicts("TES_SOC", range(T+1), 0, p['tes_cap'])
            tes_in  = pulp.LpVariable.dicts("TES_In",  range(T),   lowBound=0)
            tes_out = pulp.LpVariable.dicts("TES_Out", range(T),   lowBound=0)

            bess_soc = pulp.LpVariable.dicts("BESS_SOC", range(T+1), 0, p['bess_cap'])
            bess_cha = pulp.LpVariable.dicts("BESS_Cha", range(T),   0, p['bess_p'])
            bess_dis = pulp.LpVariable.dicts("BESS_Dis", range(T),   0, p['bess_p'])

            ee_export = pulp.LpVariable.dicts("ee_export", range(T), lowBound=0)
            ee_import = pulp.LpVariable.dicts("ee_import", range(T), lowBound=0)

            heat_shortfall = pulp.LpVariable.dicts("shortfall", range(T), lowBound=0)
            heat_delivered = pulp.LpVariable.dicts("heat_delivered", range(T), lowBound=0)

            # Počáteční stavy
            model += tes_soc[0] == p['tes_cap'] * 0.5
            model += bess_soc[0] == p['bess_cap'] * 0.2

            # KGJ logika
            for t in range(T):
                if use_kgj:
                    model += q_kgj[t] <= p['k_th'] * on[t]
                    model += q_kgj[t] >= p['k_min'] * p['k_th'] * on[t]

            for t in range(1, T):
                model += on[t] - on[t-1] == start[t]

            # Min. runtime po startu
            for t in range(T):
                for dt in range(1, int(p['k_min_runtime'])):
                    if t + dt < T:
                        model += on[t + dt] >= start[t]

            obj_terms = []
            profits = []  # Pro pozdější grafy

            for t in range(T):
                p_ee  = float(df['ee_price'].iloc[t])
                p_gas = float(df['gas_price'].iloc[t])
                h_dem = float(df['Poptávka po teple (MW)'].iloc[t])
                fve   = float(df['FVE (MW)'].iloc[t]) if use_fve and 'FVE (MW)' in df else 0.0

                heat_prod = q_kgj[t] + q_boil[t] + q_ek[t] + q_imp[t]

                # TES bilance
                model += tes_soc[t+1] == tes_soc[t] * (1 - p['tes_loss']) + tes_in[t] - tes_out[t]

                # Dodané teplo
                model += heat_delivered[t] == heat_prod + tes_out[t] - tes_in[t]

                # Nelze dodat více než poptávka
                model += heat_delivered[t] <= h_dem * p['h_cover']

                # Pokrytí + shortfall
                model += heat_delivered[t] + heat_shortfall[t] >= h_dem * p['h_cover']

                # EE bilance
                ee_kgj = q_kgj[t] * (p['k_el'] / p['k_th']) if use_kgj else 0
                model += ee_kgj + fve + ee_import[t] + bess_dis[t] == (q_ek[t] / 0.95) + bess_cha[t] + ee_export[t]
                model += bess_soc[t+1] == bess_soc[t] + bess_cha[t] * 0.90 - bess_dis[t] / 0.90

                # Cashflow
                revenue = p['h_price'] * heat_delivered[t] + (p_ee - p['dist_ee_sell']) * ee_export[t]
                costs = (p_gas + p['gas_dist']) * (q_kgj[t]/p['k_eff_th'] + q_boil[t]/0.95) + \
                        (p_ee + p['dist_ee_buy']) * ee_import[t] + \
                        p['k_start_cost'] * start[t] + \
                        p['imp_price'] * q_imp[t]

                obj_terms.append(revenue - costs - p['h_price'] * heat_shortfall[t])
                profits.append(revenue - costs - p['h_price'] * heat_shortfall[t])  # Ukládání pro grafy

            model += pulp.lpSum(obj_terms)

            # Spuštění solveru
            status = model.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=180))

        # ────────────────────────────────────────────────
        # VÝSLEDKY
        # ────────────────────────────────────────────────
        st.subheader("Výsledky optimalizace")
        st.write(f"**Status:** {pulp.LpStatus[status]}   |   **Celkový zisk:** {pulp.value(model.objective):+.0f} €")

        if status == 1:
            res = pd.DataFrame({
                'Čas': df['datetime'],
                'Poptávka tepla': df['Poptávka po teple (MW)'],
                'KGJ': [pulp.value(q_kgj[t]) for t in range(T)],
                'Kotel': [pulp.value(q_boil[t]) for t in range(T)],
                'Elektrokotel': [pulp.value(q_ek[t]) for t in range(T)],
                'Import tepla': [pulp.value(q_imp[t]) for t in range(T)],
                'TES netto': [pulp.value(tes_out[t]) - pulp.value(tes_in[t]) for t in range(T)],
                'Shortfall': [pulp.value(heat_shortfall[t]) for t in range(T)],
                'TES SOC': [pulp.value(tes_soc[t+1]) for t in range(T)],
                'BESS SOC': [pulp.value(bess_soc[t+1]) for t in range(T)],
                'EE export': [pulp.value(ee_export[t]) for t in range(T)],
                'EE import': [pulp.value(ee_import[t]) for t in range(T)],
                'EE KGJ': [pulp.value(q_kgj[t]) * (p['k_el'] / p['k_th']) if use_kgj else 0 for t in range(T)],
                'EE FVE': [float(df['FVE (MW)'].iloc[t]) if use_fve and 'FVE (MW)' in df else 0.0 for t in range(T)],
                'EE BESS dis': [pulp.value(bess_dis[t]) for t in range(T)],
                'EE EK': [pulp.value(q_ek[t]) / 0.95 for t in range(T)],
                'EE BESS cha': [pulp.value(bess_cha[t]) for t in range(T)],
                'Zisk': profits,
            })

            # Klíčové metriky
            total_profit = res['Zisk'].sum()
            total_shortfall = res['Shortfall'].sum()
            avg_coverage = (1 - total_shortfall / (res['Poptávka tepla'].sum() * p['h_cover'])) * 100
            total_ee_export = res['EE export'].sum()
            total_ee_import = res['EE import'].sum()

            st.subheader("📈 Klíčové metriky")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Celkový zisk [€]", f"{total_profit:.0f}")
            col2.metric("Celkový shortfall [MWh]", f"{total_shortfall:.1f}")
            col3.metric("Průměrné pokrytí [%]", f"{avg_coverage:.1f}")
            col4.metric("EE export [MWh]", f"{total_ee_export:.1f}")
            col5.metric("EE import [MWh]", f"{total_ee_import:.1f}")

            # Stackplot tepla (pokrytí poptávky)
            st.subheader("🔥 Pokrytí tepelné poptávky (stack chart)")
            fig_heat = go.Figure()
            sources = ['KGJ', 'Kotel', 'Elektrokotel', 'Import tepla', 'TES netto']
            colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f1c40f']
            for src, col in zip(sources, colors):
                fig_heat.add_trace(go.Scatter(x=res['Čas'], y=res[src], name=src,
                                              stackgroup='heat', fillcolor=col, mode='none'))
            fig_heat.add_trace(go.Scatter(x=res['Čas'], y=res['Shortfall'], name='Nedodáno',
                                          stackgroup='heat', fillcolor='rgba(0,0,0,0.4)', mode='none'))
            fig_heat.add_trace(go.Scatter(x=res['Čas'], y=res['Poptávka tepla'] * p['h_cover'],
                                          name='Cílová poptávka', mode='lines', line=dict(color='black', width=2, dash='dot')))
            fig_heat.update_layout(height=500, title="Složení tepelné výroby v čase")
            st.plotly_chart(fig_heat, use_container_width=True)

            # Stackplot EE bilance
            st.subheader("⚡ Elektrická bilance (stack chart)")
            fig_ee = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                                   subplot_titles=("Výroba EE", "Spotřeba EE"))
            
            # Výroba (positive stack)
            prod_sources = ['EE KGJ', 'EE FVE', 'EE import', 'EE BESS dis']
            prod_colors = ['#27ae60', '#f39c12', '#2980b9', '#8e44ad']
            for src, col in zip(prod_sources, prod_colors):
                fig_ee.add_trace(go.Scatter(x=res['Čas'], y=res[src], name=src,
                                            stackgroup='prod', fillcolor=col, mode='none'), row=1, col=1)
            
            # Spotřeba (negative stack)
            cons_sources = ['EE EK', 'EE export', 'EE BESS cha']
            cons_colors = ['#c0392b', '#16a085', '#34495e']
            for src, col in zip(cons_sources, cons_colors):
                fig_ee.add_trace(go.Scatter(x=res['Čas'], y=-res[src], name=src,
                                            stackgroup='cons', fillcolor=col, mode='none'), row=2, col=1)
            
            fig_ee.update_layout(height=600, title="Bilance elektřiny: Výroba vs. Spotřeba")
            st.plotly_chart(fig_ee, use_container_width=True)

            # Grafy SOC pro akumulátory
            st.subheader("🔋 Stavy akumulátorů")
            fig_acc = make_subplots(rows=1, cols=2, shared_yaxes=False,
                                    subplot_titles=("TES SOC", "BESS SOC"))
            fig_acc.add_trace(go.Scatter(x=res['Čas'], y=res['TES SOC'], name='TES SOC', line=dict(color='#e67e22')), row=1, col=1)
            fig_acc.add_hline(y=p['tes_cap'], row=1, col=1, line_dash="dot", annotation_text="Max TES")
            fig_acc.add_trace(go.Scatter(x=res['Čas'], y=res['BESS SOC'], name='BESS SOC', line=dict(color='#3498db')), row=1, col=2)
            fig_acc.add_hline(y=p['bess_cap'], row=1, col=2, line_dash="dot", annotation_text="Max BESS")
            fig_acc.update_layout(height=400)
            st.plotly_chart(fig_acc, use_container_width=True)

            # Kumulativní zisk
            st.subheader("💰 Kumulativní hospodářský výsledek")
            res['Kumulativní zisk'] = res['Zisk'].cumsum()
            fig_profit = px.area(res, x='Čas', y='Kumulativní zisk', title="Kumulativní zisk v čase",
                                 color_discrete_sequence=['#2ecc71'])
            fig_profit.update_layout(height=400)
            st.plotly_chart(fig_profit, use_container_width=True)

            # Ukázka tabulky
            st.subheader("Detail (prvních 48 hodin)")
            st.dataframe(res.head(48).round(3), use_container_width=True)

        else:
            st.error("Optimalizace nenašla řešení – zkuste změnit parametry (např. snížit pokrytí, vypnout min. runtime, atd.)")

else:
    st.info("Nahrajte prosím FWD křivku a lokální data, abyste mohli spustit optimalizaci.")

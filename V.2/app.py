import streamlit as st
import pandas as pd
import pulp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="KGJ Strategy Expert PRO", layout="wide")

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

# --- 2. GRAF CEN (SROVNÁNÍ) ---
if st.session_state.fwd_data is not None:
    with st.expander("📊 Náhled upravených tržních cen", expanded=True):
        fig_prices = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
        fig_prices.add_trace(go.Scatter(x=st.session_state.fwd_data['datetime'], y=st.session_state.fwd_data['ee_price'], name="EE Cena [EUR]", line=dict(color='green')), row=1, col=1)
        fig_prices.add_trace(go.Scatter(x=st.session_state.fwd_data['datetime'], y=st.session_state.fwd_data['gas_price'], name="Plyn Cena [EUR]", line=dict(color='red')), row=2, col=1)
        st.plotly_chart(fig_prices, use_container_width=True)

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
        p['ext_h_price'] = st.number_input("Cena importu tepla [EUR/MWh]", value=150.0)

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
    
    # Merge a důsledné fillna pro jistotu
    df = pd.merge(st.session_state.fwd_data, df_loc, on='datetime', how='inner').fillna(0)
    T = len(df)

    if st.button("🏁 SPUSTIT KOMPLETNÍ OPTIMALIZACI"):
        with st.spinner('Počítám optimální strategii...'):
            model = pulp.LpProblem("Energy_Optimizer", pulp.LpMaximize)
            
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

            # Startovací podmínky akumulace
            model += tes_soc[0] == p['tes_cap'] * 0.5
            model += bess_soc[0] == p['bess_cap'] * 0.2

            obj = []
            for t in range(T):
                # Dynamické načítání hodnot podle názvů sloupců (robustní)
                p_ee = float(df.loc[t, 'ee_price'])
                p_gas = float(df.loc[t, 'gas_price'])
                h_dem = float(df['Poptávka po teple (MW)'].iloc[t])
                fve = float(df['FVE (MW)'].iloc[t]) if use_fve else 0.0

                # Rovnice tepla (Poptávka + Nabíjení nádrže = Výroba + Vybíjení nádrže)
                # q_kgj + q_boil + q_ek + q_imp + (tes_soc_včerejší - tes_soc_dnešní) >= poptávka
                model += q_kgj[t] + q_boil[t] + q_ek[t] + q_imp[t] + (tes_soc[t]*(1-p['tes_loss']) - tes_soc[t+1]) >= h_dem * p['h_cover']
                
                # KGJ omezení
                model += q_kgj[t] <= p['k_th'] * on[t]
                model += q_kgj[t] >= p['k_min'] * p['k_th'] * on[t]

                # Rovnice elektřiny (Výroba KGJ + FVE + Import + Baterie_vybíjení = EK_příkon + Baterie_nabíjení + Export)
                ee_kgj = q_kgj[t] * (p['k_el'] / p['k_th'])
                model += ee_kgj + fve + ee_import[t] + bess_dis[t] == (q_ek[t]/0.98) + bess_cha[t] + ee_export[t]
                model += bess_soc[t+1] == bess_soc[t] + (bess_cha[t]*0.92) - (bess_dis[t]/0.92)

                # Finance
                income = (p['h_price'] * h_dem * p['h_cover']) + (p_ee - p['dist_ee_sell']) * ee_export[t]
                costs = (p_gas + p['gas_dist']) * (q_kgj[t]/p['k_eff_th'] + q_boil[t]/0.95) + \
                        (p_ee + p['dist_ee_buy']) * ee_import[t] + \
                        (12.0 * on[t]) + \
                        (q_imp[t] * p['ext_h_price'])
                obj.append(income - costs)

            model += pulp.lpSum(obj)
            model.solve(pulp.PULP_CBC_CMD(msg=0))

        # --- VÝSTUPY ---
        st.success(f"Optimalizace hotova. Celkový roční hospodářský výsledek: {pulp.value(model.objective):,.0f} EUR")

        res = pd.DataFrame({
            'datetime': df['datetime'],
            'Poptávka': df['Poptávka po teple (MW)'],
            'KGJ_teplo': [q_kgj[t].value() for t in range(T)],
            'Kotel_teplo': [q_boil[t].value() for t in range(T)],
            'EK_teplo': [q_ek[t].value() for t in range(T)],
            'Import_teplo': [q_imp[t].value() for t in range(T)],
            'TES_stav': [tes_soc[t].value() for t in range(T)],
            'BESS_stav': [bess_soc[t].value() for t in range(T)],
            'Export_EE': [ee_export[t].value() for t in range(T)],
            'Import_EE': [ee_import[t].value() for t in range(T)]
        })

        # --- GRAF 1: DISPATCH TEPLA ---
        st.subheader("📊 Dispatch tepla a pokrytí poptávky")
        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(x=res['datetime'], y=res['KGJ_teplo'], name="KGJ", stackgroup='one', fill='tonexty', line=dict(width=0.5, color='orange')))
        fig_h.add_trace(go.Scatter(x=res['datetime'], y=res['Kotel_teplo'], name="Plynový kotel", stackgroup='one', fill='tonexty', line=dict(width=0.5, color='blue')))
        fig_h.add_trace(go.Scatter(x=res['datetime'], y=res['EK_teplo'], name="Elektrokotel", stackgroup='one', fill='tonexty', line=dict(width=0.5, color='green')))
        fig_h.add_trace(go.Scatter(x=res['datetime'], y=res['Import_teplo'], name="Nákup tepla", stackgroup='one', fill='tonexty', line=dict(width=0.5, color='red')))
        fig_h.add_trace(go.Scatter(x=res['datetime'], y=res['Poptávka'], name="Skutečná poptávka", line=dict(color='black', width=2, dash='dot')))
        st.plotly_chart(fig_h, use_container_width=True)

        # --- TABULKA BILANCE ---
        st.subheader("📈 Roční energetická bilance")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Výroba tepla [MWh/rok]**")
            heat_summary = pd.DataFrame({
                'Zdroj': ['KGJ', 'Plynový kotel', 'Elektrokotel', 'Nákup'],
                'Množství': [res['KGJ_teplo'].sum(), res['Kotel_teplo'].sum(), res['EK_teplo'].sum(), res['Import_teplo'].sum()]
            })
            st.table(heat_summary)
        with col2:
            st.write("**Elektřina [MWh/rok]**")
            ee_gen = res['KGJ_teplo'].sum() * (p['k_el']/p['k_th']) + df['FVE (MW)'].sum()
            ee_summary = pd.DataFrame({
                'Položka': ['Celková výroba (KGJ+FVE)', 'Export do sítě', 'Vlastní spotřeba (EK + BESS)'],
                'Množství': [ee_gen, res['Export_EE'].sum(), ee_gen - res['Export_EE'].sum()]
            })
            st.table(ee_summary)

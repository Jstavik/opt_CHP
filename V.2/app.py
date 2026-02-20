import streamlit as st
import pandas as pd
import pulp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="KGJ Strategy Optimizer PRO", layout="wide")

# --- 1. KROK: KONFIGURACE TECHNOLOGIÍ ---
with st.sidebar:
    st.header("⚙️ 1. Aktivní technologie")
    use_kgj = st.checkbox("Kogenerace (KGJ)", value=True)
    use_boil = st.checkbox("Plynový kotel", value=True)
    use_ek = st.checkbox("Elektrokotel", value=True)
    use_fve = st.checkbox("Fotovoltaika (FVE)", value=False)
    use_bess = st.checkbox("Baterie (BESS)", value=False)
    use_tes = st.checkbox("Akumulace tepla (TES)", value=False)
    use_ext_heat = st.checkbox("Povolit nákup tepla (Import)", value=False)
    use_co2 = st.checkbox("Započítat CO2 povolenky", value=False)

    st.divider()
    st.header("📈 2. Tržní data")
    fwd_file = st.file_uploader("Nahraj FWD křivku (Excel)", type=["xlsx"])
    
    ee_shift = st.number_input("Posun EE [EUR/MWh]", value=0.0)
    gas_shift = st.number_input("Posun Plyn [EUR/MWh]", value=0.0)

# --- 2. KROK: PARAMETRY ---
st.title("🚀 KGJ Strategy & Dispatch Optimizer PRO")

tabs = st.tabs(["Nastavení zdrojů", "Distribuce a vnitřní toky", "Společné a CO2"])
params = {}

with tabs[0]:
    c1, c2, c3 = st.columns(3)
    with c1:
        if use_kgj:
            st.subheader("💡 KGJ")
            params['k_th'] = st.number_input("Tepelný výkon [MW]", value=1.09)
            params['k_el'] = st.number_input("Elektrický výkon [MW]", value=1.0)
            params['k_eff'] = st.number_input("Tepelná účinnost", value=0.46)
            params['k_min'] = st.slider("Minimální zatížení [%]", 0, 100, 55) / 100
            params['k_run_h'] = st.number_input("Min. doba běhu [hod]", value=4)
            params['k_off_h'] = st.number_input("Min. doba klidu [hod]", value=4)
            params['k_serv'] = st.number_input("Servis [EUR/hod]", value=12.0)
    with c2:
        if use_boil:
            st.subheader("🔥 Plynový kotel")
            params['b_max'] = st.number_input("Max. výkon kotle [MW]", value=3.91)
            params['b_eff'] = st.number_input("Účinnost kotle", value=0.95)
        if use_ek:
            st.subheader("⚡ Elektrokotel")
            params['ek_max'] = st.number_input("Max. výkon EK [MW]", value=0.61)
            params['ek_eff'] = st.number_input("Účinnost EK", value=0.98)
    with c3:
        if use_bess:
            st.subheader("🔋 BESS")
            params['bess_cap'] = st.number_input("Kapacita [MWh]", value=1.0)
            params['bess_p'] = st.number_input("Max. výkon [MW]", value=0.5)
            params['bess_eff'] = st.number_input("Účinnost cyklu", value=0.92)
            params['bess_cycle_cost'] = st.number_input("Náklad na opotřebení [EUR/MWh]", value=5.0)
        if use_tes:
            st.subheader("🌡️ TES (Teplo)")
            params['tes_cap'] = st.number_input("Kapacita TES [MWh_th]", value=5.0)
            params['tes_loss'] = st.slider("Ztráta [%/hod]", 0.0, 5.0, 0.5) / 100

with tabs[1]:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🔌 Distribuce Elektřina")
        params['dist_ee_buy'] = st.number_input("Nákup z gridu (pro EK/BESS) [EUR/MWh]", value=33.0)
        params['dist_ee_sell'] = st.number_input("Prodej do gridu (z KGJ/FVE) [EUR/MWh]", value=2.0)
        params['allow_internal'] = st.checkbox("Povolit vnitřní toky (bez distribuce)", value=True)
    with c2:
        st.subheader("⛽ Distribuce Plyn")
        params['dist_gas_kgj'] = st.number_input("Distribuce plyn - KGJ [EUR/MWh]", value=5.0)
        params['dist_gas_boil'] = st.number_input("Distribuce plyn - Kotel [EUR/MWh]", value=5.0)

with tabs[2]:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("💰 Ceny a Systém")
        params['h_price_fix'] = st.number_input("Prodejní cena tepla [EUR/MWh]", value=120.0)
        params['h_cover'] = st.slider("Minimální pokrytí poptávky", 0.0, 1.0, 0.99)
        if use_ext_heat:
            params['ext_h_price'] = st.number_input("Cena nákupu tepla [EUR/MWh]", value=80.0)
    with c2:
        if use_co2:
            st.subheader("🌍 Emise")
            params['co2_price'] = st.number_input("Cena povolenky [EUR/t]", value=70.0)
            params['co2_factor'] = 0.202 # tCO2/MWh plynu

# --- 3. KROK: DATA LOKALITY ---
st.divider()
loc_file = st.file_uploader("3️⃣ Nahraj potřebu tepla a FVE profil (Excel)", type=["xlsx"])

if fwd_file and loc_file:
    # Processing Data
    df_fwd = pd.read_excel(fwd_file)
    df_loc = pd.read_excel(loc_file)
    
    # Merge a příprava (zjednodušeno pro kód)
    df = pd.merge(df_fwd, df_loc, left_index=True, right_index=True)
    T = len(df)
    
    if st.button("🏁 SPUSTIT KOMPLETNÍ OPTIMALIZACI"):
        model = pulp.LpProblem("Energy_Master", pulp.LpMaximize)
        
        # --- PROMĚNNÉ ---
        q_kgj = pulp.LpVariable.dicts("q_KGJ", range(T), 0) # Teplo
        q_boil = pulp.LpVariable.dicts("q_Boil", range(T), 0, params.get('b_max', 0))
        q_ek = pulp.LpVariable.dicts("q_EK", range(T), 0, params.get('ek_max', 0))
        q_ext = pulp.LpVariable.dicts("q_Ext", range(T), 0)
        q_def = pulp.LpVariable.dicts("q_Def", range(T), 0)
        
        # Binární proměnné pro KGJ
        on = pulp.LpVariable.dicts("on", range(T), 0, 1, cat="Binary")
        start = pulp.LpVariable.dicts("start", range(T), 0, 1, cat="Binary")
        stop = pulp.LpVariable.dicts("stop", range(T), 0, 1, cat="Binary")
        
        # Elektřina toky
        ee_export = pulp.LpVariable.dicts("ee_export", range(T), 0)
        ee_import = pulp.LpVariable.dicts("ee_import", range(T), 0)
        
        # BESS & TES
        soc_bess = pulp.LpVariable.dicts("soc_bess", range(T), 0, params.get('bess_cap', 0))
        bess_char = pulp.LpVariable.dicts("b_char", range(T), 0, params.get('bess_p', 0))
        bess_dis = pulp.LpVariable.dicts("b_dis", range(T), 0, params.get('bess_p', 0))

        profit_total = []

        for t in range(T):
            # Ceny z FWD + Shift
            p_ee = df.iloc[t, 1] + ee_shift
            p_gas = df.iloc[t, 2] + gas_shift
            p_heat = params['h_price_fix']
            demand = df.iloc[t, 4] # Poptávka tepla
            fve_prod = df.iloc[t, 6] if use_fve else 0 # FVE MW
            
            # 1. Bilance tepla
            model += q_kgj[t] + q_boil[t] + q_ek[t] + q_ext[t] + q_def[t] >= demand * params['h_cover']
            
            # 2. Omezení KGJ
            model += q_kgj[t] <= params['k_th'] * on[t]
            model += q_kgj[t] >= params['k_min'] * params['k_th'] * on[t]
            
            # Min doba běhu/klidu (logika)
            if t > 0:
                model += start[t] - stop[t] == on[t] - on[t-1]
                for i in range(max(0, t - int(params['k_run_h']) + 1), t + 1):
                    model += on[i] >= start[t]
                for i in range(max(0, t - int(params['k_off_h']) + 1), t + 1):
                    model += 1 - on[i] >= stop[t]

            # 3. Bilance Elektřiny (Vnitřní uzel)
            # Vyrobeno (KGJ + FVE + BESS vybíjení) == Spotřebováno (EK + BESS nabíjení + Export - Import)
            ee_gen_kgj = q_kgj[t] * (params['k_el'] / params['k_th'])
            model += ee_gen_kgj + fve_prod + bess_dis[t] + ee_import[t] == (q_ek[t]/params.get('ek_eff',1)) + bess_char[t] + ee_export[t]

            # 4. BESS Logika
            if use_bess:
                if t == 0: model += soc_bess[t] == params['bess_cap'] * 0.5
                else: model += soc_bess[t] == soc_bess[t-1] * 0.99 + (bess_char[t] * params['bess_eff'] - bess_dis[t])

            # 5. CASHFLOW
            # Příjmy
            inc_heat = p_heat * (demand * params['h_cover'] - q_def[t])
            inc_ee = (p_ee - params['dist_ee_sell']) * ee_export[t]
            
            # Náklady
            cost_gas_kgj = (p_gas + params['dist_gas_kgj']) * (ee_gen_kgj / params['k_eff']) # kgj_eff zde jako total? ne, zjednodušeno
            cost_gas_boil = (p_gas + params['dist_gas_boil']) * (q_boil[t] / params.get('b_eff', 0.95))
            cost_ee_buy = (p_ee + params['dist_ee_buy']) * ee_import[t]
            cost_serv = params['k_serv'] * on[t]
            cost_bess = (bess_char[t] + bess_dis[t]) * params.get('bess_cycle_cost', 0)
            
            cost_co2 = 0
            if use_co2:
                gas_total = (q_kgj[t]/0.4) + (q_boil[t]/0.95) # Zjednodušený příkon pro CO2
                cost_co2 = gas_total * params['co2_factor'] * params['co2_price']

            penalty = q_def[t] * 5000
            
            profit_total.append(inc_heat + inc_ee - cost_gas_kgj - cost_gas_boil - cost_ee_buy - cost_serv - cost_bess - cost_co2 - penalty)

        model += pulp.lpSum(profit_total)
        model.solve(pulp.PULP_CBC_CMD(msg=0))

        # --- VÝSLEDKY ---
        st.success(f"Optimalizace dokončena. Celkový hrubý zisk: {pulp.value(model.objective):,.0f} EUR")
        
        # Dispatch Graf
        res_data = pd.DataFrame({
            'T': range(T),
            'KGJ': [q_kgj[t].value() for t in range(T)],
            'Kotel': [q_boil[t].value() for t in range(T)],
            'EK': [q_ek[t].value() for t in range(T)],
            'Deficit': [q_def[t].value() for t in range(T)],
            'Import_EE': [ee_import[t].value() for t in range(T)],
            'Export_EE': [ee_export[t].value() for t in range(T)]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=res_data['T'], y=res_data['KGJ'], name="KGJ Teplo", marker_color='orange'))
        fig.add_trace(go.Bar(x=res_data['T'], y=res_data['Kotel'], name="Kotel Teplo", marker_color='blue'))
        fig.add_trace(go.Bar(x=res_data['T'], y=res_data['EK'], name="Elektrokotel", marker_color='green'))
        fig.update_layout(barmode='stack', title="Hodinový Dispatch tepla")
        st.plotly_chart(fig, use_container_width=True)

        # Dashboard Statistik
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Provozní hodiny KGJ", f"{sum(on[t].value() for t in range(T)):.0f} h")
        c2.metric("Využití FVE", f"{df.iloc[:,6].sum():,.1f} MWh")
        c3.metric("Ušetřeno na CO2", f"---" if not use_co2 else "Aktivní")
        c4.metric("Celkový Deficit", f"{res_data['Deficit'].sum():,.1f} MWh")

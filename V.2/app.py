import streamlit as st
import pandas as pd
import pulp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="KGJ Strategy Optimizer PRO", layout="wide")

# --- 1. KROK: KONFIGURACE TECHNOLOGIÍ (SIDEBAR) ---
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

st.title("🚀 KGJ Strategy & Dispatch Optimizer PRO")

# --- 2. KROK: PARAMETRY (TABS) ---
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
        params['dist_ee_buy'] = st.number_input("Nákup z gridu [EUR/MWh]", value=33.0)
        params['dist_ee_sell'] = st.number_input("Prodej do gridu [EUR/MWh]", value=2.0)
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
            params['co2_factor'] = 0.202 

# --- 3. KROK: NAHRÁNÍ A EDITACE KŘIVEK ---
st.divider()
if fwd_file:
    df_fwd_raw = pd.read_excel(fwd_file)
    df_fwd_raw.columns = [str(c).strip() for c in df_fwd_raw.columns]
    
    # Příprava base dat s posunem
    df_fwd_raw['ee_price'] = df_fwd_raw.iloc[:, 1] + ee_shift
    df_fwd_raw['gas_price'] = df_fwd_raw.iloc[:, 2] + gas_shift
    
    with st.expander("📊 Náhled a ruční úprava tržních cen", expanded=True):
        fig_fwd = make_subplots(specs=[[{"secondary_y": True}]])
        fig_fwd.add_trace(go.Scatter(y=df_fwd_raw['ee_price'], name="EE Cena", line=dict(color='green')), secondary_y=False)
        fig_fwd.add_trace(go.Scatter(y=df_fwd_raw['gas_price'], name="Plyn Cena", line=dict(color='red')), secondary_y=True)
        st.plotly_chart(fig_fwd, use_container_width=True)
        
        st.info("Zde můžeš ručně upravit ceny pro konkrétní hodiny:")
        df_fwd_final = st.data_editor(df_fwd_raw, use_container_width=True)
else:
    st.warning("Nahraj FWD křivku v sidebar pro pokračování.")

loc_file = st.file_uploader("4️⃣ Nahraj potřebu tepla a FVE profil (aki11)", type=["xlsx"])

if fwd_file and loc_file:
    df_loc = pd.read_excel(loc_file)
    # Merge dat
    df = pd.concat([df_fwd_final.reset_index(drop=True), df_loc.reset_index(drop=True)], axis=1)
    T = len(df)
    
    if st.button("🏁 SPUSTIT KOMPLETNÍ OPTIMALIZACI"):
        model = pulp.LpProblem("Energy_Master", pulp.LpMaximize)
        
        # --- PROMĚNNÉ ---
        q_kgj = pulp.LpVariable.dicts("q_KGJ", range(T), 0)
        q_boil = pulp.LpVariable.dicts("q_Boil", range(T), 0, params.get('b_max', 0))
        q_ek = pulp.LpVariable.dicts("q_EK", range(T), 0, params.get('ek_max', 0))
        q_ext = pulp.LpVariable.dicts("q_Ext", range(T), 0)
        q_def = pulp.LpVariable.dicts("q_Def", range(T), 0)
        
        on = pulp.LpVariable.dicts("on", range(T), 0, 1, cat="Binary")
        start = pulp.LpVariable.dicts("start", range(T), 0, 1, cat="Binary")
        stop = pulp.LpVariable.dicts("stop", range(T), 0, 1, cat="Binary")
        
        ee_export = pulp.LpVariable.dicts("ee_export", range(T), 0)
        ee_import = pulp.LpVariable.dicts("ee_import", range(T), 0)
        
        soc_bess = pulp.LpVariable.dicts("soc_bess", range(T), 0, params.get('bess_cap', 0))
        bess_char = pulp.LpVariable.dicts("b_char", range(T), 0, params.get('bess_p', 0))
        bess_dis = pulp.LpVariable.dicts("b_dis", range(T), 0, params.get('bess_p', 0))

        results_cashflow = []

        for t in range(T):
            p_ee = df.loc[t, 'ee_price']
            p_gas = df.loc[t, 'gas_price']
            demand = df.iloc[t, 4] 
            fve_prod = df.iloc[t, 6] if use_fve else 0
            
            # Bilance tepla
            model += q_kgj[t] + q_boil[t] + q_ek[t] + q_ext[t] + q_def[t] >= demand * params['h_cover']
            
            # Omezení KGJ
            model += q_kgj[t] <= params['k_th'] * on[t]
            model += q_kgj[t] >= params['k_min'] * params['k_th'] * on[t]
            if t > 0:
                model += start[t] - stop[t] == on[t] - on[t-1]
                for i in range(max(0, t - int(params['k_run_h']) + 1), t + 1): model += on[i] >= start[t]
                for i in range(max(0, t - int(params['k_off_h']) + 1), t + 1): model += 1 - on[i] >= stop[t]

            # Bilance EE
            ee_gen_kgj = q_kgj[t] * (params['k_el'] / params['k_th'])
            model += ee_gen_kgj + fve_prod + bess_dis[t] + ee_import[t] == (q_ek[t]/params.get('ek_eff',1)) + bess_char[t] + ee_export[t]

            # BESS
            if use_bess:
                if t == 0: model += soc_bess[t] == params['bess_cap'] * 0.5
                else: model += soc_bess[t] == soc_bess[t-1] * 0.99 + (bess_char[t] * params['bess_eff'] - bess_dis[t])

            # Cashflow t-té hodiny
            inc_heat = params['h_price_fix'] * (demand * params['h_cover'] - q_def[t])
            inc_ee = (p_ee - params['dist_ee_sell']) * ee_export[t]
            
            cost_gas_kgj = (p_gas + params['dist_gas_kgj']) * (ee_gen_kgj / params['k_eff'])
            cost_gas_boil = (p_gas + params['dist_gas_boil']) * (q_boil[t] / params.get('b_eff', 0.95))
            cost_ee_buy = (p_ee + params['dist_ee_buy']) * ee_import[t]
            cost_serv = params['k_serv'] * on[t]
            cost_bess = (bess_char[t] + bess_dis[t]) * params.get('bess_cycle_cost', 0)
            
            penalty = q_def[t] * 5000
            model += pulp.lpSum(inc_heat + inc_ee - cost_gas_kgj - cost_gas_boil - cost_ee_buy - cost_serv - cost_bess - penalty)

        model.solve(pulp.PULP_CBC_CMD(msg=0))

        # --- REPORTOVÁNÍ VÝSLEDKŮ ---
        st.success(f"Optimalizace hotova. Zisk: {pulp.value(model.objective):,.0f} EUR")

        res_data = pd.DataFrame({
            'Hour': range(T),
            'KGJ_T': [q_kgj[t].value() for t in range(T)],
            'Kotel_T': [q_boil[t].value() for t in range(T)],
            'EK_T': [q_ek[t].value() for t in range(T)],
            'Demand_T': df.iloc[:, 4] * params['h_cover'],
            'EE_Export': [ee_export[t].value() for t in range(T)],
            'EE_Import': [ee_import[t].value() for t in range(T)],
            'BESS_SOC': [soc_bess[t].value() if use_bess else 0 for t in range(T)]
        })

        # --- GRAFICKÁ ČÁST ---
        c_plots = st.container()
        with c_plots:
            # 1. Dispatch Tepla
            fig_h = go.Figure()
            fig_h.add_trace(go.Bar(x=res_data['Hour'], y=res_data['KGJ_T'], name="KGJ", marker_color='#FF9900'))
            fig_h.add_trace(go.Bar(x=res_data['Hour'], y=res_data['Kotel_T'], name="Kotel", marker_color='#1f77b4'))
            fig_h.add_trace(go.Bar(x=res_data['Hour'], y=res_data['EK_T'], name="Elektrokotel", marker_color='#2ca02c'))
            fig_h.add_trace(go.Scatter(x=res_data['Hour'], y=res_data['Demand_T'], name="Poptávka", line=dict(color='white', dash='dot')))
            fig_h.update_layout(barmode='stack', title="Hodinový Dispatch tepla [MW]")
            st.plotly_chart(fig_h, use_container_width=True)

            col_sub1, col_sub2 = st.columns(2)
            with col_sub1:
                # 2. EE Bilance
                fig_ee = go.Figure()
                fig_ee.add_trace(go.Scatter(x=res_data['Hour'], y=res_data['EE_Export'], name="Export do sítě", fill='tozeroy', line=dict(color='gold')))
                fig_ee.add_trace(go.Scatter(x=res_data['Hour'], y=res_data['EE_Import'], name="Nákup ze sítě", fill='tozeroy', line=dict(color='red')))
                fig_ee.update_layout(title="Bilance elektřiny (Export/Import) [MW]")
                st.plotly_chart(fig_ee, use_container_width=True)
            
            with col_sub2:
                # 3. BESS SOC (pokud je)
                if use_bess:
                    fig_soc = go.Figure()
                    fig_soc.add_trace(go.Scatter(x=res_data['Hour'], y=res_data['BESS_SOC'], name="Stav nabití", line=dict(color='cyan')))
                    fig_soc.update_layout(title="BESS SOC [MWh]", yaxis_range=[0, params['bess_cap']])
                    st.plotly_chart(fig_soc, use_container_width=True)

        # --- FINÁLNÍ REPORT ---
        st.divider()
        st.header("📊 Inteligentní Report")
        r1, r2, r3, r4 = st.columns(4)
        
        kgj_hours = sum(1 for t in range(T) if q_kgj[t].value() > 0.1)
        total_ee_gen = sum(q_kgj[t].value() * (params['k_el']/params['k_th']) for t in range(T))
        total_ee_sold = res_data['EE_Export'].sum()
        self_consumption = ((total_ee_gen - total_ee_sold) / total_ee_gen * 100) if total_ee_gen > 0 else 0

        r1.metric("Provoz KGJ", f"{kgj_hours} hod")
        r2.metric("Vlastní spotřeba EE", f"{self_consumption:.1f} %")
        r3.metric("Prům. cena EE", f"{df['ee_price'].mean():.2f} EUR")
        r4.metric("Pokrytí tepla", f"{(1 - (res_data['Demand_T'].sum() - (res_data['KGJ_T'].sum()+res_data['Kotel_T'].sum()+res_data['EK_T'].sum()))/res_data['Demand_T'].sum())*100:.1f} %")

        with st.expander("🔍 Podrobná tabulka výsledků"):
            st.dataframe(res_data, use_container_width=True)

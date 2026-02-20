import streamlit as st
import pandas as pd
import pulp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="KGJ Strategy Optimizer PRO", layout="wide")

# --- 1. SIDEBAR: DATA A JEJICH ÚPRAVA ---
with st.sidebar:
    st.header("📈 1. Tržní data (FWD)")
    fwd_file = st.file_uploader("Nahraj FWD křivku (Excel)", type=["xlsx"])
    
    if fwd_file:
        df_fwd_raw = pd.read_excel(fwd_file)
        df_fwd_raw.columns = [str(c).strip() for c in df_fwd_raw.columns]
        
        # Načtení base hodnot (průměr z nahrátého souboru)
        base_ee_val = float(df_fwd_raw.iloc[:, 1].mean())
        base_gas_val = float(df_fwd_raw.iloc[:, 2].mean())
        
        st.subheader("🛠️ Úprava Base hodnot")
        # Tady vidíš, co jsi nahrál a můžeš to změnit
        ee_base_edit = st.number_input("Base EE [EUR/MWh]", value=base_ee_val, step=1.0)
        gas_base_edit = st.number_input("Base Plyn [EUR/MWh]", value=base_gas_val, step=0.1)
        
        # Výpočet shiftu oproti původnímu souboru pro aplikaci na celou křivku
        ee_shift = ee_base_edit - base_ee_val
        gas_shift = gas_base_edit - base_gas_val
        
        # Finální křivka pro výpočet
        df_fwd_final = df_fwd_raw.copy()
        df_fwd_final['ee_price'] = df_fwd_final.iloc[:, 1] + ee_shift
        df_fwd_final['gas_price'] = df_fwd_final.iloc[:, 2] + gas_shift
        
        st.write(f"Aplikovaný posun: EE {ee_shift:+.1f}, Plyn {gas_shift:+.1f}")
        
        # Malý graf přímo v sidebaru pro kontrolu
        fig_side = go.Figure()
        fig_side.add_trace(go.Scatter(y=df_fwd_final['ee_price'], name="EE", line=dict(color='green', width=1)))
        fig_side.add_trace(go.Scatter(y=df_fwd_final['gas_price'], name="Plyn", line=dict(color='red', width=1)))
        fig_side.update_layout(height=150, margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
        st.plotly_chart(fig_side, use_container_width=True)
    
    st.divider()
    st.header("⚙️ 2. Technologie")
    use_kgj = st.checkbox("Kogenerace (KGJ)", value=True)
    use_boil = st.checkbox("Plynový kotel", value=True)
    use_ek = st.checkbox("Elektrokotel", value=True)
    use_fve = st.checkbox("Fotovoltaika (FVE)", value=False)
    use_bess = st.checkbox("Baterie (BESS)", value=False)
    use_co2 = st.checkbox("CO2 povolenky", value=False)

# --- 2. PARAMETRY (HLAVNÍ OKNO) ---
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
            params['bess_cycle_cost'] = st.number_input("Opotřebení [EUR/MWh]", value=5.0)

with tabs[1]:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🔌 Distribuce Elektřina")
        params['dist_ee_buy'] = st.number_input("Nákup z gridu [EUR/MWh]", value=33.0)
        params['dist_ee_sell'] = st.number_input("Prodej do gridu [EUR/MWh]", value=2.0)
    with c2:
        st.subheader("⛽ Distribuce Plyn")
        params['dist_gas_kgj'] = st.number_input("Distribuce plyn - KGJ [EUR/MWh]", value=5.0)
        params['dist_gas_boil'] = st.number_input("Distribuce plyn - Kotel [EUR/MWh]", value=5.0)

with tabs[2]:
    st.subheader("💰 Systém")
    params['h_price_fix'] = st.number_input("Prodejní cena tepla [EUR/MWh]", value=120.0)
    params['h_cover'] = st.slider("Minimální pokrytí poptávky", 0.0, 1.0, 0.99)
    if use_co2:
        params['co2_price'] = st.number_input("Cena povolenky [EUR/t]", value=70.0)
        params['co2_factor'] = 0.202 

# --- 3. VÝPOČET ---
st.divider()
loc_file = st.file_uploader("3️⃣ Nahraj potřebu tepla a FVE profil (aki11)", type=["xlsx"])

if fwd_file and loc_file:
    df_loc = pd.read_excel(loc_file)
    df = pd.concat([df_fwd_final.reset_index(drop=True), df_loc.reset_index(drop=True)], axis=1)
    T = len(df)
    
    if st.button("🏁 SPUSTIT KOMPLETNÍ OPTIMALIZACI"):
        model = pulp.LpProblem("Energy_Master", pulp.LpMaximize)
        
        q_kgj = pulp.LpVariable.dicts("q_KGJ", range(T), 0)
        q_boil = pulp.LpVariable.dicts("q_Boil", range(T), 0, params.get('b_max', 0))
        q_ek = pulp.LpVariable.dicts("q_EK", range(T), 0, params.get('ek_max', 0))
        q_def = pulp.LpVariable.dicts("q_Def", range(T), 0)
        on = pulp.LpVariable.dicts("on", range(T), 0, 1, cat="Binary")
        start = pulp.LpVariable.dicts("start", range(T), 0, 1, cat="Binary")
        stop = pulp.LpVariable.dicts("stop", range(T), 0, 1, cat="Binary")
        ee_export = pulp.LpVariable.dicts("ee_export", range(T), 0)
        ee_import = pulp.LpVariable.dicts("ee_import", range(T), 0)
        
        profit_total = []
        for t in range(T):
            p_ee, p_gas = df.loc[t, 'ee_price'], df.loc[t, 'gas_price']
            demand = df.iloc[t, 4] 
            fve_prod = df.iloc[t, 6] if use_fve else 0
            
            model += q_kgj[t] + q_boil[t] + q_ek[t] + q_def[t] >= demand * params['h_cover']
            model += q_kgj[t] <= params['k_th'] * on[t]
            model += q_kgj[t] >= params['k_min'] * params['k_th'] * on[t]
            
            if t > 0:
                model += start[t] - stop[t] == on[t] - on[t-1]
                for i in range(max(0, t - int(params['k_run_h']) + 1), t + 1): model += on[i] >= start[t]
                for i in range(max(0, t - int(params['k_off_h']) + 1), t + 1): model += 1 - on[i] >= stop[t]

            ee_gen_kgj = q_kgj[t] * (params['k_el'] / params['k_th'])
            model += ee_gen_kgj + fve_prod + ee_import[t] == (q_ek[t]/params.get('ek_eff',1)) + ee_export[t]

            inc = (params['h_price_fix'] * (demand * params['h_cover'] - q_def[t])) + (p_ee - params['dist_ee_sell']) * ee_export[t]
            cost = (p_gas + params['dist_gas_kgj']) * (ee_gen_kgj / params['k_eff']) + \
                   (p_gas + params['dist_gas_boil']) * (q_boil[t] / params.get('b_eff', 0.95)) + \
                   (p_ee + params['dist_ee_buy']) * (q_ek[t] / params.get('ek_eff', 0.98)) + \
                   (params['k_serv'] * on[t])
            
            profit_total.append(inc - cost - (q_def[t] * 5000))

        model += pulp.lpSum(profit_total)
        model.solve(pulp.PULP_CBC_CMD(msg=0))

        # --- REPORTOVÁNÍ ---
        st.header("📊 Výsledky optimalizace")
        res = pd.DataFrame({
            'T': range(T), 'KGJ': [q_kgj[t].value() for t in range(T)],
            'Kotel': [q_boil[t].value() for t in range(T)], 'EK': [q_ek[t].value() for t in range(T)],
            'Export': [ee_export[t].value() for t in range(T)], 'Import': [ee_import[t].value() for t in range(T)]
        })

        fig_main = go.Figure()
        fig_main.add_trace(go.Bar(x=res['T'], y=res['KGJ'], name="KGJ", marker_color='orange'))
        fig_main.add_trace(go.Bar(x=res['T'], y=res['Kotel'], name="Kotel", marker_color='blue'))
        fig_main.add_trace(go.Bar(x=res['T'], y=res['EK'], name="Elektrokotel", marker_color='green'))
        fig_main.update_layout(barmode='stack', title="Dispatch tepla [MW]")
        st.plotly_chart(fig_main, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Celkový zisk", f"{pulp.value(model.objective):,.0f} EUR")
        c2.metric("Provoz KGJ", f"{sum(on[t].value() for t in range(T))} hod")
        c3.metric("Export EE", f"{res['Export'].sum():,.0f} MWh")

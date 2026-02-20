import streamlit as st
import pandas as pd
import pulp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="KGJ Strategy Optimizer PRO", layout="wide")

# Inicializace stavu
if 'fwd_data' not in st.session_state: st.session_state.fwd_data = None
if 'loc_data' not in st.session_state: st.session_state.loc_data = None

st.title("🚀 KGJ Strategy & Dispatch Optimizer PRO")

# --- 1. KROK: SIDEBAR A ÚPRAVA KŘIVEK ---
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
        
        # Výpočet průměrů z nahrátého souboru
        avg_ee_raw = float(df_year.iloc[:, 1].mean())
        avg_gas_raw = float(df_year.iloc[:, 2].mean())
        
        st.subheader("🛠️ Úprava na aktuální trh")
        st.write(f"Původní průměr EE: **{avg_ee_raw:.2f}**")
        st.write(f"Původní průměr Plyn: **{avg_gas_raw:.2f}**")
        
        ee_market_new = st.number_input("Nastavit novou cenu EE [EUR]", value=avg_ee_raw)
        gas_market_new = st.number_input("Nastavit novou cenu Plyn [EUR]", value=avg_gas_raw)
        
        ee_shift = ee_market_new - avg_ee_raw
        gas_shift = gas_market_new - avg_gas_raw
        
        # Finální data pro optimizer
        df_fwd = df_year.copy()
        df_fwd.columns = ['datetime', 'ee_original', 'gas_original']
        df_fwd['ee_price'] = df_fwd['ee_original'] + ee_shift
        df_fwd['gas_price'] = df_fwd['gas_original'] + gas_shift
        df_fwd['mdh'] = df_fwd['datetime'].dt.strftime('%m-%d-%H')
        st.session_state.fwd_data = df_fwd

    st.divider()
    st.header("⚙️ 2. Technologie")
    use_kgj = st.checkbox("Kogenerace (KGJ)", value=True)
    use_boil = st.checkbox("Plynový kotel", value=True)
    use_ek = st.checkbox("Elektrokotel", value=True)
    use_fve = st.checkbox("Fotovoltaika (FVE)", value=True)
    use_bess = st.checkbox("Baterie (BESS)", value=False)
    use_co2 = st.checkbox("Započítat CO2", value=False)

# --- 2. KROK: NÁHLED KŘIVEK ---
if st.session_state.fwd_data is not None:
    with st.expander("📊 Náhled upravených tržních cen", expanded=True):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
        fig.add_trace(go.Scatter(x=st.session_state.fwd_data['datetime'], y=st.session_state.fwd_data['ee_original'], name="EE Původní", line=dict(color='rgba(0,255,0,0.2)', dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=st.session_state.fwd_data['datetime'], y=st.session_state.fwd_data['ee_price'], name="EE Upravená", line=dict(color='green')), row=1, col=1)
        fig.add_trace(go.Scatter(x=st.session_state.fwd_data['datetime'], y=st.session_state.fwd_data['gas_original'], name="Plyn Původní", line=dict(color='rgba(255,0,0,0.2)', dash='dot')), row=2, col=1)
        fig.add_trace(go.Scatter(x=st.session_state.fwd_data['datetime'], y=st.session_state.fwd_data['gas_price'], name="Plyn Upravená", line=dict(color='red')), row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

# --- 3. KROK: PARAMETRY ---
tabs = st.tabs(["Nastavení zdrojů", "Distribuce a vnitřní toky", "Společné a CO2"])
p = {}

with tabs[0]:
    c1, c2, c3 = st.columns(3)
    with c1:
        if use_kgj:
            st.subheader("💡 KGJ")
            p['k_th'] = st.number_input("Tepelný výkon [MW]", value=1.09)
            p['k_el'] = st.number_input("Elektrický výkon [MW]", value=1.0)
            p['k_eff_th'] = st.number_input("Tepelná účinnost (příkon->teplo)", value=0.46)
            p['k_min'] = st.slider("Minimální zatížení [%]", 0, 100, 55) / 100
            p['k_run_h'] = st.number_input("Min. doba běhu [hod]", value=4)
            p['k_off_h'] = st.number_input("Min. doba klidu [hod]", value=4)
            p['k_serv'] = st.number_input("Servis [EUR/hod]", value=12.0)
    with c2:
        if use_boil:
            st.subheader("🔥 Plynový kotel")
            p['b_max'] = st.number_input("Max. výkon kotle [MW]", value=3.91)
            p['b_eff'] = st.number_input("Účinnost kotle", value=0.95)
        if use_ek:
            st.subheader("⚡ Elektrokotel")
            p['ek_max'] = st.number_input("Max. výkon EK [MW]", value=0.61)
            p['ek_eff'] = st.number_input("Účinnost EK", value=0.98)
    with c3:
        if use_bess:
            st.subheader("🔋 BESS")
            p['bess_cap'] = st.number_input("Kapacita [MWh]", value=1.0)
            p['bess_p'] = st.number_input("Max. výkon [MW]", value=0.5)
            p['bess_eff'] = 0.92

with tabs[1]:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🔌 Distribuce Elektřina")
        p['dist_ee_buy'] = st.number_input("Nákup z gridu (distribuce) [EUR]", value=33.0)
        p['dist_ee_sell'] = st.number_input("Prodej do gridu (distribuce) [EUR]", value=2.0)
    with c2:
        st.subheader("⛽ Distribuce Plyn")
        p['dist_gas_kgj'] = st.number_input("Distribuce plyn - KGJ [EUR]", value=5.0)
        p['dist_gas_boil'] = st.number_input("Distribuce plyn - Kotel [EUR]", value=5.0)

with tabs[2]:
    st.subheader("💰 Ostatní")
    p['h_price'] = st.number_input("Prodejní cena tepla [EUR/MWh]", value=120.0)
    p['h_cover'] = st.slider("Minimální pokrytí poptávky", 0.0, 1.0, 0.99)
    if use_co2:
        p['co2_price'] = st.number_input("Cena povolenky [EUR/t]", value=70.0)
        p['co2_factor'] = 0.202

# --- 4. KROK: VÝPOČET ---
st.divider()
loc_file = st.file_uploader("3️⃣ Nahraj lokální data (aki11)", type=["xlsx"])

if st.session_state.fwd_data is not None and loc_file:
    df_loc = pd.read_excel(loc_file)
    df = pd.concat([st.session_state.fwd_data.reset_index(drop=True), df_loc.reset_index(drop=True)], axis=1)
    T = len(df)

    if st.button("🏁 SPUSTIT KOMPLETNÍ OPTIMALIZACI"):
        model = pulp.LpProblem("Dispatcher", pulp.LpMaximize)
        
        # Proměnné
        q_kgj = pulp.LpVariable.dicts("q_KGJ", range(T), 0)
        q_boil = pulp.LpVariable.dicts("q_Boil", range(T), 0, p.get('b_max',0))
        q_ek = pulp.LpVariable.dicts("q_EK", range(T), 0, p.get('ek_max',0))
        q_def = pulp.LpVariable.dicts("q_Def", range(T), 0)
        on = pulp.LpVariable.dicts("on", range(T), 0, 1, cat="Binary")
        start = pulp.LpVariable.dicts("start", range(T), 0, 1, cat="Binary")
        stop = pulp.LpVariable.dicts("stop", range(T), 0, 1, cat="Binary")
        ee_export = pulp.LpVariable.dicts("ee_export", range(T), 0)
        ee_import = pulp.LpVariable.dicts("ee_import", range(T), 0)

        objective = []
        for t in range(T):
            price_ee = df.loc[t, 'ee_price']
            price_gas = df.loc[t, 'gas_price']
            demand = df.iloc[t, 4]
            fve = df.iloc[t, 6] if use_fve else 0
            
            # Bilance tepla
            model += q_kgj[t] + q_boil[t] + q_ek[t] + q_def[t] >= demand * p['h_cover']
            
            # KGJ Logika
            model += q_kgj[t] <= p['k_th'] * on[t]
            model += q_kgj[t] >= p['k_min'] * p['k_th'] * on[t]
            if t > 0:
                model += start[t] - stop[t] == on[t] - on[t-1]
                for i in range(max(0, t - int(p['k_run_h']) + 1), t + 1): model += on[i] >= start[t]
                for i in range(max(0, t - int(p['k_off_h']) + 1), t + 1): model += 1 - on[i] >= stop[t]

            # Bilance EE (Vnitřní uzel)
            ee_kgj = q_kgj[t] * (p['k_el'] / p['k_th'])
            model += ee_kgj + fve + ee_import[t] == (q_ek[t]/p.get('ek_eff',1)) + ee_export[t]

            # Ekonomika
            inc = (p['h_price'] * (demand * p['h_cover'] - q_def[t])) + (price_ee - p['dist_ee_sell']) * ee_export[t]
            
            # Náklady (Příkon KGJ = q_kgj / účinnost_tepelná)
            prikon_kgj = (q_kgj[t] / p['k_eff_th'])
            prikon_boil = (q_boil[t] / p.get('b_eff', 0.95))
            
            cost = (price_gas + p['dist_gas_kgj']) * prikon_kgj + \
                   (price_gas + p['dist_gas_boil']) * prikon_boil + \
                   (price_ee + p['dist_ee_buy']) * ee_import[t] + \
                   (p['k_serv'] * on[t])
            
            if use_co2:
                cost += (prikon_kgj + prikon_boil) * p['co2_factor'] * p['co2_price']
            
            objective.append(inc - cost - (q_def[t] * 5000))

        model += pulp.lpSum(objective)
        model.solve(pulp.PULP_CBC_CMD(msg=0))

        # --- VÝSLEDKY ---
        st.success(f"Optimalizace dokončena. Zisk: {pulp.value(model.objective):,.0f} EUR")
        
        res = pd.DataFrame({
            'T': range(T),
            'KGJ': [q_kgj[t].value() for t in range(T)],
            'Kotel': [q_boil[t].value() for t in range(T)],
            'EK': [q_ek[t].value() for t in range(T)],
            'Export': [ee_export[t].value() for t in range(T)],
            'Import': [ee_import[t].value() for t in range(T)]
        })

        fig_res = go.Figure()
        fig_res.add_trace(go.Bar(x=res['T'], y=res['KGJ'], name="KGJ", marker_color='orange'))
        fig_res.add_trace(go.Bar(x=res['T'], y=res['Kotel'], name="Kotel", marker_color='blue'))
        fig_res.add_trace(go.Bar(x=res['T'], y=res['EK'], name="Elektrokotel", marker_color='green'))
        fig_res.update_layout(barmode='stack', title="Dispatch tepla [MW]")
        st.plotly_chart(fig_res, use_container_width=True)

        # Inteligentní report
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Provoz KGJ", f"{sum(on[t].value() for t in range(T))} hod")
        c2.metric("Export EE", f"{res['Export'].sum():,.1f} MWh")
        c3.metric("Import EE", f"{res['Import'].sum():,.1f} MWh")
        c4.metric("Vlastní využití FVE/KGJ", f"{(1 - res['Export'].sum()/(res['KGJ'].sum()*p['k_el']/p['k_th'] + df.iloc[:,6].sum()) )*100:.1f} %")

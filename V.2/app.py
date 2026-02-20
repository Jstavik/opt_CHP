import streamlit as st
import pandas as pd
import pulp
import plotly.graph_objects as go

st.set_page_config(page_title="KGJ Strategy Expert PRO", layout="wide")

st.title("🚀 KGJ Strategy & Dispatch Optimizer PRO")

# --- 1. SIDEBAR: NAČTENÍ A SHIFT ---
with st.sidebar:
    st.header("📈 1. Tržní ceny (FWD)")
    fwd_file = st.file_uploader("Nahraj FWD křivku (Excel)", type=["xlsx"])
    
    if fwd_file:
        # Načtení s ohledem na CZ formát
        df_fwd_raw = pd.read_excel(fwd_file)
        df_fwd_raw.columns = [str(c).strip() for c in df_fwd_raw.columns]
        
        # Klíčová oprava pro CZ datum
        date_col = df_fwd_raw.columns[0]
        df_fwd_raw[date_col] = pd.to_datetime(df_fwd_raw[date_col], dayfirst=True, errors='coerce')
        df_fwd_raw = df_fwd_raw.dropna(subset=[date_col])
        
        years = sorted(df_fwd_raw[date_col].dt.year.unique())
        sel_year = st.selectbox("Rok pro analýzu", years)
        df_fwd = df_fwd_raw[df_fwd_raw[date_col].dt.year == sel_year].copy()
        
        # Přejmenování podle tvého souboru: Datum, FWD (EUR/MWh), FWD plyn (EUR/MWh)
        df_fwd.columns = ['datetime', 'ee_original', 'gas_original']
        
        # VÝPOČET SHIFTU
        avg_ee_raw = float(df_fwd['ee_original'].mean())
        avg_gas_raw = float(df_fwd['gas_original'].mean())
        
        st.write(f"Původní průměr EE: **{avg_ee_raw:.2f}**")
        ee_target = st.number_input("Nová cílová cena EE", value=avg_ee_raw)
        gas_target = st.number_input("Nová cílová cena Plyn", value=avg_gas_raw)
        
        ee_shift = ee_target - avg_ee_raw
        gas_shift = gas_target - avg_gas_raw
        
        df_fwd['ee_price'] = df_fwd['ee_original'] + ee_shift
        df_fwd['gas_price'] = df_fwd['gas_original'] + gas_shift
        st.session_state.fwd_final = df_fwd

    st.header("⚙️ 2. Aktivní technologie")
    use_kgj = st.checkbox("Kogenerace (KGJ)", value=True)
    use_boil = st.checkbox("Plynový kotel", value=True)
    use_ek = st.checkbox("Elektrokotel", value=True)
    use_tes = st.checkbox("Nádrž (TES)", value=True)
    use_bess = st.checkbox("Baterie (BESS)", value=True)
    use_ext_heat = st.checkbox("Povolit import tepla", value=False)

# --- 2. PARAMETRY ---
p = {'k_th': 1.09, 'k_el': 1.0, 'k_eff_th': 0.46, 'k_min': 0.55, 'b_max': 3.91, 'ek_max': 0.61, 
     'h_price': 120.0, 'start_cost': 150.0, 'ext_h_price': 250.0, 'tes_cap': 10.0}

# --- 3. OPTIMALIZACE ---
loc_file = st.file_uploader("3️⃣ Nahraj aki11.xlsx", type=["xlsx"])

if 'fwd_final' in st.session_state and loc_file:
    df_loc = pd.read_excel(loc_file).fillna(0)
    df_loc.columns = [str(c).strip() for c in df_loc.columns]
    df_loc.iloc[:, 0] = pd.to_datetime(df_loc.iloc[:, 0], dayfirst=True, errors='coerce')
    df_loc = df_loc.dropna(subset=[df_loc.columns[0]])
    df_loc.rename(columns={df_loc.columns[0]: 'datetime'}, inplace=True)
    
    df = pd.merge(st.session_state.fwd_final, df_loc, on='datetime', how='inner')
    T = len(df)

    if st.button("🏁 SPUSTIT VÝPOČET"):
        model = pulp.LpProblem("Dispatcher", pulp.LpMaximize)
        
        # Proměnné
        q_kgj = pulp.LpVariable.dicts("q_kgj", range(T), 0)
        on = pulp.LpVariable.dicts("on", range(T), 0, 1, cat="Binary")
        start = pulp.LpVariable.dicts("start", range(T), 0, 1, cat="Binary")
        q_boil = pulp.LpVariable.dicts("q_boil", range(T), 0, p['b_max'])
        unserved = pulp.LpVariable.dicts("unserved", range(T), 0)
        tes_soc = pulp.LpVariable.dicts("tes_soc", range(T+1), 0, p['tes_cap'])

        obj = []
        for t in range(T):
            h_dem = float(df['Poptávka po teple (MW)'].iloc[t])
            p_ee = float(df['ee_price'].iloc[t])
            p_gas = float(df['gas_price'].iloc[t])
            fve = float(df['FVE (MW)'].iloc[t]) if 'FVE (MW)' in df.columns else 0

            if t > 0: model += start[t] >= on[t] - on[t-1]
            if not use_kgj: model += q_kgj[t] == 0; model += on[t] == 0
            if not use_boil: model += q_boil[t] == 0
            
            # Bilance tepla
            model += q_kgj[t] + q_boil[t] + unserved[t] + (tes_soc[t]*0.995 - tes_soc[t+1]) >= h_dem
            model += q_kgj[t] <= p['k_th'] * on[t]
            model += q_kgj[t] >= p['k_th'] * p['k_min'] * on[t]

            # Ekonomika
            revenue = (p['h_price'] * (h_dem - unserved[t])) + (p_ee - 2.0) * (q_kgj[t] * (p['k_el']/p['k_th']) + fve)
            costs = (p_gas + 5.0) * (q_kgj[t]/p['k_eff_th'] + q_boil[t]/0.95) + (start[t] * p['start_cost']) + (unserved[t] * 5000)
            obj.append(revenue - costs)

        model += pulp.lpSum(obj)
        model.solve(pulp.PULP_CBC_CMD(msg=0))

        res = pd.DataFrame({'datetime': df['datetime'], 'KGJ': [q_kgj[t].value() for t in range(T)], 
                            'Kotel': [q_boil[t].value() for t in range(T)], 'Unserved': [unserved[t].value() for t in range(T)]})
        
        st.subheader("📊 Dispatch tepla")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res['datetime'], y=res['KGJ'], name="KGJ", stackgroup='one'))
        fig.add_trace(go.Scatter(x=res['datetime'], y=res['Kotel'], name="Kotel", stackgroup='one'))
        fig.add_trace(go.Scatter(x=res['datetime'], y=res['Unserved'], name="Nepokryto", stackgroup='one', fillcolor='red'))
        st.plotly_chart(fig, use_container_width=True)
        
        st.download_button("Export CSV", res.to_csv(index=False).encode('utf-8'), "vysledky.csv")

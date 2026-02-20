import streamlit as st
import pandas as pd
import pulp
import plotly.graph_objects as go

st.set_page_config(page_title="KGJ Strategy Expert PRO", layout="wide")

# --- 1. SIDEBAR A VÝPOČET FWD KŘIVKY ---
with st.sidebar:
    st.header("📈 1. Tržní ceny (FWD)")
    fwd_file = st.file_uploader("Nahraj FWD křivku (Excel)", type=["xlsx"])
    
    if fwd_file:
        df_raw = pd.read_excel(fwd_file)
        df_raw.columns = [str(c).strip() for c in df_raw.columns]
        # Předpokládáme: 0. Datum, 1. EE, 2. Plyn
        df_raw.iloc[:, 0] = pd.to_datetime(df_raw.iloc[:, 0], dayfirst=True)
        
        # Filtr na rok
        years = sorted(df_raw.iloc[:, 0].dt.year.unique())
        sel_year = st.selectbox("Rok", years)
        df_fwd = df_raw[df_raw.iloc[:, 0].dt.year == sel_year].copy()
        df_fwd.columns = ['datetime', 'ee_original', 'gas_original']
        
        # VÝPOČET SHIFTU (Tohle jsi chtěl!)
        avg_ee_raw = df_fwd['ee_original'].mean()
        avg_gas_raw = df_fwd['gas_original'].mean()
        
        st.write(f"Původní průměr EE: {avg_ee_raw:.2f}")
        st.write(f"Původní průměr Plyn: {avg_gas_raw:.2f}")
        
        ee_target = st.number_input("Nová cílová cena EE", value=avg_ee_raw)
        gas_target = st.number_input("Nová cílová cena Plyn", value=avg_gas_raw)
        
        # APLIKACE SHIFTU NA CELOU KŘIVKU
        ee_shift = ee_target - avg_ee_raw
        gas_shift = gas_target - avg_gas_raw
        
        df_fwd['ee_price'] = df_fwd['ee_original'] + ee_shift
        df_fwd['gas_price'] = df_fwd['gas_original'] + gas_shift
        st.session_state.fwd_final = df_fwd

    st.header("⚙️ 2. Nastavení zdrojů")
    use_ext_heat = st.checkbox("Povolit import tepla", value=False)

# --- 2. GRAF CEN (SEXY VIZUÁL) ---
if 'fwd_final' in st.session_state:
    with st.expander("📊 Detailní náhled cenových křivek", expanded=True):
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=st.session_state.fwd_final['datetime'], y=st.session_state.fwd_final['ee_price'], name="EE (Upravená)", line=dict(color='green')))
        fig_p.add_trace(go.Scatter(x=st.session_state.fwd_final['datetime'], y=st.session_state.fwd_final['gas_price'], name="Plyn (Upravená)", line=dict(color='red')))
        st.plotly_chart(fig_p, use_container_width=True)

# --- 3. PARAMETRY (TABY) ---
t1, t2 = st.tabs(["Technika", "Ekonomika"])
p = {}
with t1:
    p['k_th'] = st.number_input("KGJ Tepelný výkon [MW]", value=1.09)
    p['k_el'] = st.number_input("KGJ Elektrický výkon [MW]", value=1.0)
    p['k_eff_th'] = st.number_input("KGJ Tepelná účinnost", value=0.46)
    p['k_min'] = st.slider("Min. výkon KGJ %", 0, 100, 55) / 100
    p['b_max'] = st.number_input("Kotel max [MW]", value=3.91)

with t2:
    p['h_price'] = st.number_input("Prodej tepla [EUR/MWh]", value=120.0)
    p['start_cost'] = st.number_input("Start KGJ [EUR]", value=150.0)
    p['ext_h_price'] = st.number_input("Cena importu", value=250.0)

# --- 4. OPTIMALIZACE ---
loc_file = st.file_uploader("Nahrát aki11.xlsx", type=["xlsx"])
if 'fwd_final' in st.session_state and loc_file:
    df_loc = pd.read_excel(loc_file).fillna(0)
    df_loc.columns = [str(c).strip() for c in df_loc.columns]
    df_loc.rename(columns={df_loc.columns[0]: 'datetime'}, inplace=True)
    df_loc['datetime'] = pd.to_datetime(df_loc['datetime'], dayfirst=True)
    
    # Spojení dat
    df = pd.merge(st.session_state.fwd_final, df_loc, on='datetime', how='inner')
    T = len(df)

    if st.button("🚀 SPOČÍTAT DISPATCH"):
        model = pulp.LpProblem("Dispatch", pulp.LpMaximize)
        
        # Proměnné
        q_kgj = pulp.LpVariable.dicts("q_kgj", range(T), 0)
        on = pulp.LpVariable.dicts("on", range(T), 0, 1, cat="Binary")
        start = pulp.LpVariable.dicts("start", range(T), 0, 1, cat="Binary")
        q_boil = pulp.LpVariable.dicts("q_boil", range(T), 0, p['b_max'])
        q_imp = pulp.LpVariable.dicts("q_imp", range(T), 0)
        unserved = pulp.LpVariable.dicts("unserved", range(T), 0)
        ee_export = pulp.LpVariable.dicts("ee_export", range(T), 0)

        obj = []
        for t in range(T):
            h_dem = df['Poptávka po teple (MW)'].iloc[t]
            p_ee = df['ee_price'].iloc[t]
            p_gas = df['gas_price'].iloc[t]
            
            if t > 0: model += start[t] >= on[t] - on[t-1]
            
            # Bilance tepla
            model += q_kgj[t] + q_boil[t] + (q_imp[t] if use_ext_heat else 0) + unserved[t] >= h_dem
            model += q_kgj[t] <= p['k_th'] * on[t]
            model += q_kgj[t] >= p['k_min'] * p['k_th'] * on[t]
            
            # Cashflow
            income = p['h_price'] * (h_dem - unserved[t]) + (p_ee - 2.0) * (q_kgj[t] * (p['k_el']/p['k_th']))
            costs = (p_gas + 5.0) * (q_kgj[t]/p['k_eff_th'] + q_boil[t]/0.95) + (start[t] * p['start_cost']) + (unserved[t] * 5000)
            obj.append(income - costs)

        model += pulp.lpSum(obj)
        model.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=120))

        # --- SEXY VÝSTUPY ---
        res = pd.DataFrame({
            'datetime': df['datetime'],
            'KGJ': [q_kgj[t].value() for t in range(T)],
            'Kotel': [q_boil[t].value() for t in range(T)],
            'Nepokryto': [unserved[t].value() for t in range(T)],
            'Poptávka': df['Poptávka po teple (MW)']
        })

        st.subheader("🔥 Dispatch Tepla")
        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(x=res['datetime'], y=res['KGJ'], name="KGJ", stackgroup='one', line=dict(color='orange')))
        fig_h.add_trace(go.Scatter(x=res['datetime'], y=res['Kotel'], name="Kotel", stackgroup='one', line=dict(color='blue')))
        fig_h.add_trace(go.Scatter(x=res['datetime'], y=res['Nepokryto'], name="NEPOKRYTO", stackgroup='one', line=dict(color='red')))
        fig_h.add_trace(go.Scatter(x=res['datetime'], y=res['Poptávka'], name="Poptávka", line=dict(color='black', dash='dot')))
        st.plotly_chart(fig_h, use_container_width=True)

        st.download_button("Exportovat CSV", res.to_csv(index=False).encode('utf-8'), "vysledky.csv")

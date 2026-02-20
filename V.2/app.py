import streamlit as st
import pandas as pd
import pulp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="KGJ Strategy Optimizer PRO", layout="wide")

# --- CUSTOM CSS PRO TMAVÝ VZHLED A ČITELNOST ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1a1c24; padding: 15px; border-radius: 10px; border: 1px solid #31333f; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 KGJ Strategy & Dispatch Optimizer")

# --- SIDEBAR: KONFIGURACE ---
with st.sidebar:
    st.header("1️⃣ Technologie & Trh")
    use_kgj = st.checkbox("Kogenerace (KGJ)", value=True)
    use_boil = st.checkbox("Plynový kotel", value=True)
    use_ek = st.checkbox("Elektrokotel", value=True)
    use_fve = st.checkbox("Fotovoltaika (FVE)", value=True)
    use_ext = st.checkbox("Povolit nákup tepla", value=False)
    
    st.divider()
    fwd_file = st.file_uploader("Nahraj FWD křivku (EE/ZP)", type=["xlsx"])
    ee_shift = st.number_input("Posun EE [EUR/MWh]", value=0.0)
    gas_shift = st.number_input("Posun Plyn [EUR/MWh]", value=0.0)

# --- NASTAVENÍ PARAMETRŮ ---
col_p1, col_p2, col_p3 = st.columns(3)
p = {}

with col_p1:
    st.subheader("💡 Kogenerace")
    p['k_th'] = st.number_input("Max Tepelný výkon [MW]", value=1.09)
    p['k_el'] = st.number_input("Max Elektrický výkon [MW]", value=1.0)
    p['k_eff'] = st.number_input("Tepelná účinnost", value=0.46)
    p['k_min'] = st.slider("Minimální zatížení [%]", 0, 100, 55) / 100
    p['k_serv'] = st.number_input("Servis [EUR/hod]", value=12.0)
    p['k_run'] = st.number_input("Min. doba běhu [hod]", value=4)

with col_p2:
    st.subheader("🔥 Ostatní zdroje")
    p['b_max'] = st.number_input("Plynový kotel max [MW]", value=3.91)
    p['b_eff'] = st.number_input("Účinnost kotle", value=0.95)
    p['ek_max'] = st.number_input("Elektrokotel max [MW]", value=0.61)
    p['ek_eff'] = st.number_input("Účinnost elektrokotle", value=0.98)

with col_p3:
    st.subheader("⚙️ Systém & Distribuce")
    p['h_cover'] = st.slider("Minimální pokrytí tepla", 0.0, 1.0, 0.99)
    p['dist_ee_buy'] = st.number_input("Distribuce nákup EE [EUR]", value=33.0)
    p['dist_ee_sell'] = st.number_input("Distribuce prodej EE [EUR]", value=2.0)
    p['co2_price'] = st.number_input("Cena CO2 [EUR/t] (0=vypnuto)", value=0.0)

# --- DATA LOKALITY ---
st.divider()
loc_file = st.file_uploader("2️⃣ Nahraj data lokality (aki11)", type=["xlsx"])

if fwd_file and loc_file:
    # Načtení a příprava dat
    df_fwd = pd.read_excel(fwd_file)
    df_loc = pd.read_excel(loc_file)
    
    # Mapování tvého souboru aki11
    # Sloupce: Datum, Teplo prodej, Poptávka po teple, Nákup tepla (EUR/MWh), FVE (MW)
    df_loc.columns = ['datetime', 'heat_price', 'demand', 'ext_heat_price', 'fve_mw']
    
    # Merge a aplikace shiftů
    df = df_fwd.copy()
    df['ee_market'] = df.iloc[:, 1] + ee_shift
    df['gas_market'] = df.iloc[:, 2] + gas_shift
    df = pd.concat([df, df_loc.drop('datetime', axis=1)], axis=1)

    st.subheader("📝 Editovatelná data před výpočtem")
    df_edited = st.data_editor(df, use_container_width=True, num_rows="fixed")

    if st.button("🏁 SPUSTIT OPTIMALIZACI"):
        T = len(df_edited)
        model = pulp.LpProblem("Dispatcher", pulp.LpMaximize)

        # PROMĚNNÉ
        q_kgj = pulp.LpVariable.dicts("q_KGJ", range(T), 0, p['k_th'])
        q_boil = pulp.LpVariable.dicts("q_Boil", range(T), 0, p['b_max'])
        q_ek = pulp.LpVariable.dicts("q_EK", range(T), 0, p['ek_max'])
        q_ext = pulp.LpVariable.dicts("q_Ext", range(T), 0)
        on = pulp.LpVariable.dicts("on", range(T), 0, 1, cat="Binary")
        
        # Toky EE
        ee_export = pulp.LpVariable.dicts("ee_export", range(T), 0)
        ee_import = pulp.LpVariable.dicts("ee_import", range(T), 0)

        obj_terms = []
        for t in range(T):
            row = df_edited.iloc[t]
            h_req = row['demand'] * p['h_cover']
            
            # Bilance tepla
            model += q_kgj[t] + q_boil[t] + q_ek[t] + q_ext[t] >= h_req
            model += q_kgj[t] <= p['k_th'] * on[t]
            model += q_kgj[t] >= p['k_min'] * p['k_th'] * on[t]
            
            # Bilance EE (vnitřní uzel)
            ee_kgj = q_kgj[t] * (p['k_el']/p['k_th'])
            fve = row['fve_mw'] if not pd.isna(row['fve_mw']) else 0
            model += ee_kgj + fve + ee_import[t] == (q_ek[t]/p['ek_eff']) + ee_export[t]

            # Cashflow
            income = (row['heat_price'] * h_req) + (row['ee_market'] - p['dist_ee_sell']) * ee_export[t]
            
            # Náklady
            cost_gas = (row['gas_market'] + 5.0) * ((ee_kgj/0.4) + (q_boil[t]/p['b_eff'])) # +5 EUR distribuce plynu
            cost_ee_grid = (row['ee_market'] + p['dist_ee_buy']) * ee_import[t]
            cost_serv = p['k_serv'] * on[t]
            
            if p['co2_price'] > 0:
                cost_gas += (ee_kgj/0.4 + q_boil[t]/p['b_eff']) * 0.202 * p['co2_price']
            
            obj_terms.append(income - cost_gas - cost_ee_grid - cost_serv)

        model += pulp.lpSum(obj_terms)
        model.solve(pulp.PULP_CBC_CMD(msg=0))

        # --- VÝSLEDKY A GRAFY ---
        zisk = pulp.value(model.objective)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CELKOVÝ ZISK", f"{zisk:,.0f} EUR")
        c2.metric("KGJ HODINY", f"{sum(on[t].value() for t in range(T)):.0f} h")
        c3.metric("PRŮMĚRNÉ ZATÍŽENÍ", f"{(sum(q_kgj[t].value() for t in range(T))/max(1, sum(on[t].value() for t in range(T)))/p['k_th']*100):.1f} %")
        c4.metric("EE PRODÁNO", f"{sum(ee_export[t].value() for t in range(T)):,.0f} MWh")

        # GRAF 1: DISPATCH TEPLA
        res = pd.DataFrame({
            'T': df_edited['datetime'],
            'KGJ': [q_kgj[t].value() for t in range(T)],
            'Kotel': [q_boil[t].value() for t in range(T)],
            'EK': [q_ek[t].value() for t in range(T)],
            'Nákup': [q_ext[t].value() for t in range(T)],
            'Poptávka': df_edited['demand']
        })
        
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(x=res['T'], y=res['KGJ'], name="KGJ", marker_color="#FF9900"))
        fig1.add_trace(go.Bar(x=res['T'], y=res['Kotel'], name="Kotel", marker_color="#1f77b4"))
        fig1.add_trace(go.Bar(x=res['T'], y=res['EK'], name="Elektrokotel", marker_color="#2ca02c"))
        fig1.add_trace(go.Scatter(x=res['T'], y=res['Poptávka'], name="Poptávka", line=dict(color='white', dash='dot')))
        fig1.update_layout(barmode='stack', title="Dispatch tepla [MW]", template="plotly_dark")
        st.plotly_chart(fig1, use_container_width=True)

        # GRAF 2: KUMULATIVNÍ ZISK
        profits = [pulp.value(obj_terms[t]) for t in range(T)]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=res['T'], y=pd.Series(profits).cumsum(), fill='tozeroy', name="Zisk", line=dict(color='#00ffcc')))
        fig2.update_layout(title="Kumulativní zisk [EUR]", template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)

import streamlit as st
import pandas as pd
import pulp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="KGJ Strategy Expert PRO", layout="wide")

# --- 1. SIDEBAR A VÝPOČET FWD KŘIVKY (TVOJE LOGIKA) ---
with st.sidebar:
    st.header("📈 1. Tržní ceny (FWD)")
    fwd_file = st.file_uploader("Nahraj FWD křivku (Excel)", type=["xlsx"])
    
    if fwd_file:
        df_raw = pd.read_excel(fwd_file)
        df_raw.columns = [str(c).strip() for c in df_raw.columns]
        
        # OPRAVA: Převedení prvního sloupce na datum (pokud selže, vyhodí NaT a ty řádky smažeme)
        df_raw.iloc[:, 0] = pd.to_datetime(df_raw.iloc[:, 0], dayfirst=True, errors='coerce')
        df_raw = df_raw.dropna(subset=[df_raw.columns[0]])
        
        years = sorted(df_raw.iloc[:, 0].dt.year.unique())
        sel_year = st.selectbox("Rok pro analýzu", years)
        df_fwd = df_raw[df_raw.iloc[:, 0].dt.year == sel_year].copy()
        
        # Pojmenování pro jistotu: 0: Date, 1: EE, 2: Gas
        df_fwd = df_fwd.iloc[:, :3]
        df_fwd.columns = ['datetime', 'ee_original', 'gas_original']
        
        # VÝPOČET SHIFTU
        avg_ee_raw = float(df_fwd['ee_original'].mean())
        avg_gas_raw = float(df_fwd['gas_original'].mean())
        
        st.info(f"Původní průměry: EE {avg_ee_raw:.2f} | Plyn {avg_gas_raw:.2f}")
        
        ee_target = st.number_input("Nová cílová cena EE [EUR]", value=avg_ee_raw)
        gas_target = st.number_input("Nová cílová cena Plyn [EUR]", value=avg_gas_raw)
        
        ee_shift = ee_target - avg_ee_raw
        gas_shift = gas_target - avg_gas_raw
        
        # Finální křivky pro výpočet
        df_fwd['ee_price'] = df_fwd['ee_original'] + ee_shift
        df_fwd['gas_price'] = df_fwd['gas_original'] + gas_shift
        st.session_state.fwd_final = df_fwd

    st.divider()
    st.header("⚙️ 2. Nastavení zdrojů")
    use_ext_heat = st.checkbox("Povolit import tepla", value=False)
    use_fve = st.checkbox("Uvažovat FVE z aki11", value=True)

# --- 2. VIZUALIZACE TRHU ---
if 'fwd_final' in st.session_state:
    with st.expander("📊 Srovnání upravených cenových křivek", expanded=True):
        fig_p = make_subplots(rows=1, cols=2, subplot_titles=("Elektrická energie", "Zemní plyn"))
        fig_p.add_trace(go.Scatter(x=st.session_state.fwd_final['datetime'], y=st.session_state.fwd_final['ee_price'], name="EE Upravená", line=dict(color='green')), row=1, col=1)
        fig_p.add_trace(go.Scatter(x=st.session_state.fwd_final['datetime'], y=st.session_state.fwd_final['gas_price'], name="Plyn Upravený", line=dict(color='red')), row=1, col=2)
        st.plotly_chart(fig_p, use_container_width=True)

# --- 3. PARAMETRY ---
t_tech, t_eco = st.tabs(["🏗️ Technika", "💰 Ekonomika"])
p = {}
with t_tech:
    col1, col2 = st.columns(2)
    with col1:
        p['k_th'] = st.number_input("KGJ Tepelný výkon [MW]", value=1.09)
        p['k_el'] = st.number_input("KGJ Elektrický výkon [MW]", value=1.0)
        p['k_eff_th'] = st.number_input("KGJ Tepelná účinnost (plyn->teplo)", value=0.46)
        p['k_min'] = st.slider("Min. zatížení KGJ %", 0, 100, 55) / 100
    with col2:
        p['b_max'] = st.number_input("Plynový kotel max [MW]", value=3.91)
        p['ek_max'] = st.number_input("Elektrokotel max [MW]", value=0.61)

with t_eco:
    col1, col2 = st.columns(2)
    with col1:
        p['h_price'] = st.number_input("Prodejní cena tepla [EUR/MWh]", value=120.0)
        p['start_cost'] = st.number_input("Náklad na start KGJ [EUR]", value=150.0)
    with col2:
        p['ext_h_price'] = st.number_input("Cena importu tepla [EUR/MWh]", value=250.0)
        p['dist_ee_sell'] = 2.0  # Poplatek za prodej do sítě

# --- 4. VÝPOČET A SEXY VÝSTUPY ---
st.divider()
loc_file = st.file_uploader("3️⃣ Nahrát lokální data (aki11)", type=["xlsx"])

if 'fwd_final' in st.session_state and loc_file:
    df_loc = pd.read_excel(loc_file).fillna(0)
    df_loc.columns = [str(c).strip() for c in df_loc.columns]
    df_loc.rename(columns={df_loc.columns[0]: 'datetime'}, inplace=True)
    df_loc['datetime'] = pd.to_datetime(df_loc['datetime'], dayfirst=True, errors='coerce')
    
    df = pd.merge(st.session_state.fwd_final, df_loc, on='datetime', how='inner')
    T = len(df)

    if st.button("🚀 SPOČÍTAT ROČNÍ OPTIMALIZACI"):
        with st.spinner('Počítám dispatch...'):
            model = pulp.LpProblem("Dispatcher_Expert", pulp.LpMaximize)
            
            # Proměnné
            q_kgj = pulp.LpVariable.dicts("q_kgj", range(T), 0)
            on = pulp.LpVariable.dicts("on", range(T), 0, 1, cat="Binary")
            start = pulp.LpVariable.dicts("start", range(T), 0, 1, cat="Binary")
            q_boil = pulp.LpVariable.dicts("q_boil", range(T), 0, p['b_max'])
            q_imp = pulp.LpVariable.dicts("q_imp", range(T), 0)
            unserved = pulp.LpVariable.dicts("unserved", range(T), 0)

            obj = []
            for t in range(T):
                h_dem = df['Poptávka po teple (MW)'].iloc[t]
                p_ee = df['ee_price'].iloc[t]
                p_gas = df['gas_price'].iloc[t]
                
                if t > 0:
                    model += start[t] >= on[t] - on[t-1]
                
                # Bilance Tepla
                model += q_kgj[t] + q_boil[t] + (q_imp[t] if use_ext_heat else 0) + unserved[t] >= h_dem
                model += q_kgj[t] <= p['k_th'] * on[t]
                model += q_kgj[t] >= p['k_min'] * p['k_th'] * on[t]
                
                # Ekonomika
                # Výnos: Prodané teplo + Prodána elektřina z KGJ
                ee_rev = (p_ee - p['dist_ee_sell']) * (q_kgj[t] * (p['k_el']/p['k_th']))
                heat_rev = p['h_price'] * (h_dem - unserved[t])
                
                costs = (p_gas + 5.0) * (q_kgj[t]/p['k_eff_th'] + q_boil[t]/0.95) + \
                        (start[t] * p['start_cost']) + \
                        (unserved[t] * 5000) + \
                        (q_imp[t] * p['ext_h_price'])
                
                obj.append(heat_rev + ee_rev - costs)

            model += pulp.lpSum(obj)
            model.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=120))

        # --- REPREZENTACE VÝSLEDKŮ ---
        res = pd.DataFrame({
            'datetime': df['datetime'],
            'Poptávka': df['Poptávka po teple (MW)'],
            'KGJ': [q_kgj[t].value() for t in range(T)],
            'Kotel': [q_boil[t].value() for t in range(T)],
            'Import': [q_imp[t].value() for t in range(T)],
            'Nepokryto': [unserved[t].value() for t in range(T)],
            'Zisk_hodina': [pulp.value(obj[t]) for t in range(T)]
        })

        st.success(f"Optimalizace hotova. Celkový roční zisk: {res['Zisk_hodina'].sum():,.0f} EUR")

        # Sexy Graf 1: Stacked Area Chart Tepla
        st.subheader("📊 Roční dispatch tepla")
        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(x=res['datetime'], y=res['KGJ'], name="Kogenerace (KGJ)", stackgroup='one', line=dict(color='orange')))
        fig_h.add_trace(go.Scatter(x=res['datetime'], y=res['Kotel'], name="Plynový kotel", stackgroup='one', line=dict(color='blue')))
        fig_h.add_trace(go.Scatter(x=res['datetime'], y=res['Import'], name="Externí nákup", stackgroup='one', line=dict(color='purple')))
        fig_h.add_trace(go.Scatter(x=res['datetime'], y=res['Nepokryto'], name="NEPOKRYTÁ POPTÁVKA", stackgroup='one', line=dict(color='red')))
        fig_h.add_trace(go.Scatter(x=res['datetime'], y=res['Poptávka'], name="Požadovaná poptávka", line=dict(color='black', dash='dot')))
        st.plotly_chart(fig_h, use_container_width=True)

        # Sexy Graf 2: Kumulativní zisk
        st.subheader("💰 Kumulativní ekonomický vývoj")
        res['cum_profit'] = res['Zisk_hodina'].cumsum()
        fig_c = go.Figure(go.Scatter(x=res['datetime'], y=res['cum_profit'], fill='tozeroy', name="Zisk [EUR]", line=dict(color='gold')))
        st.plotly_chart(fig_c, use_container_width=True)

        # Tabulka Souhrnů
        st.subheader("📑 Roční bilance")
        c1, c2, c3 = st.columns(3)
        c1.metric("Vyrobeno KGJ", f"{res['KGJ'].sum():,.0f} MWh")
        c2.metric("Vyrobeno Kotel", f"{res['Kotel'].sum():,.0f} MWh")
        c3.metric("Nepokryté teplo", f"{res['Nepokryto'].sum():,.2f} MWh", delta_color="inverse")

        # Export
        st.download_button("📥 Stáhnout kompletní výsledky (CSV)", res.to_csv(index=False).encode('utf-8'), "vysledky_kgj.csv", "text/csv")

import streamlit as st
import pandas as pd
import pulp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="KGJ Strategy Optimizer PRO", layout="wide")

# Inicializace stavu
if 'fwd_data' not in st.session_state: st.session_state.fwd_data = None

st.title("🚀 KGJ Strategy & Dispatch Optimizer PRO")

# --- 1. KROK: SIDEBAR (FWD DATA) ---
with st.sidebar:
    st.header("📈 1. Tržní ceny (FWD)")
    fwd_file = st.file_uploader("Nahraj FWD křivku (Excel)", type=["xlsx"], key="fwd")
    
    if fwd_file:
        df_raw = pd.read_excel(fwd_file)
        df_raw.columns = [str(c).strip() for c in df_raw.columns]
        date_col = df_raw.columns[0]
        # Převedeme na datetime a zajistíme jednotný formát
        df_raw[date_col] = pd.to_datetime(df_raw[date_col], dayfirst=True)
        
        years = sorted(df_raw[date_col].dt.year.unique())
        sel_year = st.selectbox("Rok pro analýzu", years)
        df_year = df_raw[df_raw[date_col].dt.year == sel_year].copy()
        
        avg_ee_raw = float(df_year.iloc[:, 1].mean())
        avg_gas_raw = float(df_year.iloc[:, 2].mean())
        
        st.subheader("🛠️ Úprava na aktuální trh")
        st.info(f"Původní průměry: EE {avg_ee_raw:.1f} | Plyn {avg_gas_raw:.1f}")
        
        ee_market_new = st.number_input("Nová cena EE [EUR]", value=avg_ee_raw)
        gas_market_new = st.number_input("Nová cena Plyn [EUR]", value=avg_gas_raw)
        
        ee_shift = ee_market_new - avg_ee_raw
        gas_shift = gas_market_new - avg_gas_raw
        
        df_fwd = df_year.copy()
        df_fwd.columns = ['datetime', 'ee_original', 'gas_original']
        df_fwd['ee_price'] = df_fwd['ee_original'] + ee_shift
        df_fwd['gas_price'] = df_fwd['gas_original'] + gas_shift
        st.session_state.fwd_data = df_fwd

    st.divider()
    st.header("⚙️ 2. Technologie")
    use_kgj = st.checkbox("Kogenerace (KGJ)", value=True)
    use_boil = st.checkbox("Plynový kotel", value=True)
    use_ek = st.checkbox("Elektrokotel", value=True)
    use_fve = st.checkbox("Fotovoltaika (FVE)", value=True)

# --- 2. KROK: NÁHLED KŘIVEK ---
if st.session_state.fwd_data is not None:
    with st.expander("📊 Náhled upravených tržních cen", expanded=True):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
        fig.add_trace(go.Scatter(x=st.session_state.fwd_data['datetime'], y=st.session_state.fwd_data['ee_price'], name="EE Upravená", line=dict(color='green')), row=1, col=1)
        fig.add_trace(go.Scatter(x=st.session_state.fwd_data['datetime'], y=st.session_state.fwd_data['gas_price'], name="Plyn Upravená", line=dict(color='red')), row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

# --- 3. PARAMETRY ---
tabs = st.tabs(["Technické parametry", "Ekonomika", "CO2"])
p = {}
with tabs[0]:
    c1, c2 = st.columns(2)
    with c1:
        p['k_th'] = st.number_input("KGJ Tepelný výkon [MW]", value=1.09)
        p['k_el'] = st.number_input("KGJ Elektrický výkon [MW]", value=1.0)
        p['k_eff_th'] = st.number_input("KGJ Tepelná účinnost", value=0.46)
        p['k_min'] = st.slider("Minimální zatížení [%]", 0, 100, 55) / 100
    with c2:
        p['b_max'] = st.number_input("Plynový kotel max [MW]", value=3.91)
        p['ek_max'] = st.number_input("Elektrokotel max [MW]", value=0.61)

with tabs[1]:
    c1, c2 = st.columns(2)
    with c1:
        p['dist_ee_buy'] = st.number_input("Distribuce nákup EE [EUR]", value=33.0)
        p['dist_ee_sell'] = st.number_input("Distribuce prodej EE [EUR]", value=2.0)
    with c2:
        p['h_price'] = st.number_input("Prodejní cena tepla [EUR/MWh]", value=120.0)
        p['h_cover'] = st.slider("Minimální pokrytí poptávky", 0.0, 1.0, 0.99)

# --- 4. VÝPOČET ---
st.divider()
loc_file = st.file_uploader("3️⃣ Nahraj lokální data (aki11)", type=["xlsx"], key="aki")

if st.session_state.fwd_data is not None and loc_file:
    df_loc = pd.read_excel(loc_file)
    df_loc.columns = [str(c).strip() for c in df_loc.columns]
    # Sjednotíme časový sloupec pro merge
    df_loc.rename(columns={df_loc.columns[0]: 'datetime'}, inplace=True)
    df_loc['datetime'] = pd.to_datetime(df_loc['datetime'], dayfirst=True)
    
    # KLÍČOVÁ OPRAVA: Merge místo concat
    df = pd.merge(st.session_state.fwd_data, df_loc, on='datetime', how='inner')
    T = len(df)

    if st.button("🏁 SPUSTIT KOMPLETNÍ OPTIMALIZACI"):
        model = pulp.LpProblem("Dispatcher", pulp.LpMaximize)
        
        q_kgj = pulp.LpVariable.dicts("q_KGJ", range(T), 0)
        q_boil = pulp.LpVariable.dicts("q_Boil", range(T), 0, p['b_max'])
        q_ek = pulp.LpVariable.dicts("q_EK", range(T), 0, p['ek_max'])
        on = pulp.LpVariable.dicts("on", range(T), 0, 1, cat="Binary")
        ee_export = pulp.LpVariable.dicts("ee_export", range(T), 0)
        ee_import = pulp.LpVariable.dicts("ee_import", range(T), 0)

        obj = []
        for t in range(T):
            price_ee = df.loc[t, 'ee_price']
            price_gas = df.loc[t, 'gas_price']
            demand = df.iloc[t, 4] # Poptávka po teplo
            fve = df.iloc[t, 7] if len(df.columns) > 7 else 0 # FVE je v aki11 obvykle 5. sloupec dat (index 7 po merge)

            # Teplo
            model += q_kgj[t] + q_boil[t] + q_ek[t] >= demand * p['h_cover']
            model += q_kgj[t] <= p['k_th'] * on[t]
            model += q_kgj[t] >= p['k_min'] * p['k_th'] * on[t]

            # Elektřina
            ee_kgj = q_kgj[t] * (p['k_el'] / p['k_th'])
            model += ee_kgj + fve + ee_import[t] == (q_ek[t]/0.98) + ee_export[t]

            # Peníze
            revenue = (p['h_price'] * demand * p['h_cover']) + (price_ee - p['dist_ee_sell']) * ee_export[t]
            costs = (price_gas + 5.0) * (q_kgj[t]/p['k_eff_th']) + \
                    (price_gas + 5.0) * (q_boil[t]/0.95) + \
                    (price_ee + p['dist_ee_buy']) * ee_import[t] + (12.0 * on[t])
            obj.append(revenue - costs)

        model += pulp.lpSum(obj)
        model.solve(pulp.PULP_CBC_CMD(msg=0))

        # Reálné grafy s časovou osou
        res = pd.DataFrame({'datetime': df['datetime'], 'KGJ': [q_kgj[t].value() for t in range(T)], 
                            'Kotel': [q_boil[t].value() for t in range(T)], 'EK': [q_ek[t].value() for t in range(T)]})
        
        st.success(f"Hotovo! Hrubý zisk: {pulp.value(model.objective):,.0f} EUR")
        
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(x=res['datetime'], y=res['KGJ'], name="KGJ", stackgroup='one', fill='tonexty'))
        fig_res.add_trace(go.Scatter(x=res['datetime'], y=res['Kotel'], name="Kotel", stackgroup='one', fill='tonexty'))
        fig_res.add_trace(go.Scatter(x=res['datetime'], y=res['EK'], name="Elektrokotel", stackgroup='one', fill='tonexty'))
        st.plotly_chart(fig_res, use_container_width=True)

        # Metriky s ošetřením chyb
        total_gen = sum(res['KGJ']) * (p['k_el']/p['k_th']) + (df.iloc[:,7].sum() if len(df.columns)>7 else 0)
        exp_sum = sum([ee_export[t].value() for t in range(T)])
        self_cons = (1 - exp_sum/total_gen)*100 if total_gen > 0 else 0
        
        c1, c2 = st.columns(2)
        c1.metric("Export do sítě", f"{exp_sum:,.0f} MWh")
        c2.metric("Vlastní využití (FVE+KGJ)", f"{self_cons:.1f} %")

import streamlit as st
import pandas as pd
import pulp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="KGJ Strategy Expert", layout="wide")

if 'fwd_data' not in st.session_state: st.session_state.fwd_data = None
if 'loc_data' not in st.session_state: st.session_state.loc_data = None

st.title("🎯 KGJ Strategy & Dispatch Optimizer")

def clean_df(df, is_fwd=True):
    df = df.dropna(how='all').dropna(axis=1, how='all')
    df.columns = [str(c).replace('\n', ' ').strip() for c in df.columns]
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
    df = df.dropna(subset=[date_col]).rename(columns={date_col: 'datetime'})
    df['mdh'] = df['datetime'].dt.strftime('%m-%d-%H')
    
    if is_fwd:
        df = df.rename(columns={df.columns[1]: 'ee_price', df.columns[2]: 'gas_price'})
    else:
        # Mapování sloupců podle tvého obrázku
        mapping = {df.columns[1]: 'heat_price', df.columns[2]: 'heat_demand'}
        for col in df.columns:
            low_col = col.lower()
            if 'nákup' in low_col: mapping[col] = 'external_heat_price'
            if 'fve' in low_col: mapping[col] = 'fve_gen'
        df = df.rename(columns=mapping)
    
    for col in df.columns:
        if col not in ['datetime', 'mdh']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# --- SIDEBAR ---
with st.sidebar:
    st.header("📋 1. Aktivní zdroje")
    use_kgj = st.checkbox("Kogenerace (KGJ)", value=True)
    use_boil = st.checkbox("Plynový kotel", value=True)
    use_ek = st.checkbox("Elektrokotel", value=True)
    use_external_heat = st.checkbox("Povolit nákup tepla (Import)", value=True)
    
    st.header("📈 2. Tržní data")
    fwd_file = st.file_uploader("Nahraj FWD křivku", type=["xlsx"])
    if fwd_file:
        st.session_state.fwd_data = clean_df(pd.read_excel(fwd_file), is_fwd=True)
    
    if st.session_state.fwd_data is not None:
        years = sorted(st.session_state.fwd_data['datetime'].dt.year.unique())
        sel_year = st.selectbox("Rok výpočtu", years)
        df_yr = st.session_state.fwd_data[st.session_state.fwd_data['datetime'].dt.year == sel_year].copy().reset_index(drop=True)
        
        st.subheader("🛠️ Úprava cen (Shifty)")
        ee_shift = st.number_input("EE Shift [EUR/MWh]", value=0.0)
        gas_shift = st.number_input("Plyn Shift [EUR/MWh]", value=0.0)
        df_yr['ee_price_mod'] = df_yr['ee_price'] + ee_shift
        df_yr['gas_price_mod'] = df_yr['gas_price'] + gas_shift

# --- HLAVNÍ PLOCHA: GRAF KŘIVKY ---
if st.session_state.fwd_data is not None:
    st.subheader(f"📊 Náhled tržních křivek - {sel_year}")
    fig_trh = make_subplots(specs=[[{"secondary_y": True}]])
    fig_trh.add_trace(go.Scatter(x=df_yr['datetime'], y=df_yr['ee_price_mod'], name="EE Cena (upravená)", line=dict(color='#00ff00')), secondary_y=False)
    fig_trh.add_trace(go.Scatter(x=df_yr['datetime'], y=df_yr['gas_price_mod'], name="Plyn Cena (upravená)", line=dict(color='red')), secondary_y=True)
    fig_trh.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_trh, use_container_width=True)

st.subheader("📍 3. Data lokality")
loc_file = st.file_uploader("Nahraj lokální Excel (aki11)", type=["xlsx"])
if loc_file:
    st.session_state.loc_data = clean_df(pd.read_excel(loc_file), is_fwd=False)

if st.session_state.fwd_data is not None and st.session_state.loc_data is not None:
    calc_df = pd.merge(df_yr, st.session_state.loc_data, on='mdh', how='inner').sort_values('datetime_x').reset_index(drop=True)

    # Parametry (přímo přístupné)
    st.markdown("### ⚙️ Nastavení parametrů")
    c1, c2, c3 = st.columns(3)
    with c1:
        k_th = st.number_input("KGJ Tepelný výkon [MW]", value=1.09)
        k_el = st.number_input("KGJ Elektrický výkon [MW]", value=1.0)
        k_eff = st.number_input("Tepelná účinnost KGJ", value=0.46)
        k_serv = st.number_input("Servis [EUR/hod]", value=12.0)
    with c2:
        b_max = st.number_input("Plyn. kotel max [MW]", value=3.91)
        ek_max = st.number_input("Elektrokotel max [MW]", value=0.61)
        dist_in = st.number_input("Distribuce nákup EE [EUR]", value=33.0)
        dist_out = st.number_input("Distribuce prodej EE [EUR]", value=2.0)
    with c3:
        h_cover = st.slider("Minimální pokrytí tepla", 0.9, 1.0, 0.99)
        min_up = st.number_input("Min. doba běhu [hod]", value=4)

    if st.button("🚀 SPUSTIT OPTIMALIZACI"):
        T = len(calc_df)
        model = pulp.LpProblem("Dispatcher", pulp.LpMaximize)

        # Definice limitů podle sidebaru (TOHLE JE TA OPRAVA)
        limit_kgj = k_th if use_kgj else 0
        limit_boil = b_max if use_boil else 0
        limit_ek = ek_max if use_ek else 0

        q_kgj = pulp.LpVariable.dicts("q_KGJ", range(T), 0, limit_kgj)
        q_boil = pulp.LpVariable.dicts("q_boiler", range(T), 0, limit_boil)
        q_ek = pulp.LpVariable.dicts("q_eboiler", range(T), 0, limit_ek)
        q_ext = pulp.LpVariable.dicts("q_external", range(T), 0)
        on = pulp.LpVariable.dicts("on", range(T), 0, 1, cat="Binary")

        kgj_gas_per_heat = (k_th / k_eff) / k_th [cite: 12]
        kgj_el_per_heat = k_el / k_th [cite: 12]

        profits = []
        for t in range(T):
            ee = calc_df.loc[t, 'ee_price_mod']
            gas = calc_df.loc[t, 'gas_price_mod']
            hp = calc_df.loc[t, 'heat_price']
            demand = calc_df.loc[t, 'heat_demand']
            h_req = h_cover * demand

            # Bilance a omezení
            model += q_kgj[t] + q_boil[t] + q_ek[t] + q_ext[t] >= h_req [cite: 14]
            model += q_kgj[t] <= limit_kgj * on[t] [cite: 13]
            
            if not use_external_heat or 'external_heat_price' not in calc_df.columns:
                model += q_ext[t] == 0

            income = (hp * h_req) + ((ee - dist_out) * q_kgj[t] * kgj_el_per_heat) [cite: 16]
            costs = (gas * (q_kgj[t] * kgj_gas_per_heat + q_boil[t]/0.95)) + \
                    ((ee + dist_in) * (q_ek[t]/0.98)) + (k_serv * on[t]) [cite: 17]
            
            if use_external_heat and 'external_heat_price' in calc_df.columns:
                costs += (calc_df.loc[t, 'external_heat_price'] * q_ext[t])

            profits.append(income - costs)

        model += pulp.lpSum(profits)
        model.solve(pulp.PULP_CBC_CMD(msg=0))

        st.success(f"Optimalizace dokončena. Hrubý zisk: {pulp.value(model.objective):,.0f} EUR")
        
        # Vizualizace výsledků...
        res_df = pd.DataFrame({
            'T': calc_df['datetime_x'],
            'KGJ': [q_kgj[t].value() for t in range(T)],
            'Kotel': [q_boil[t].value() for t in range(T)],
            'Elektrokotel': [q_ek[t].value() for t in range(T)],
            'Nákup tepla': [q_ext[t].value() for t in range(T)]
        })
        fig = go.Figure()
        for c in ['KGJ', 'Kotel', 'Elektrokotel', 'Nákup tepla']:
            fig.add_trace(go.Bar(x=res_df['T'], y=res_df[c], name=c))
        fig.update_layout(barmode='stack', title="Hodinový Dispatch tepla")
        st.plotly_chart(fig, use_container_width=True)

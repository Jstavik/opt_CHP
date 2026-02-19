import streamlit as st
import pandas as pd
import pulp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="KGJ Strategy & Dispatch", layout="wide")

# Inicializace session state pro data
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
        # Mapování podle tvého Excelu (Datum, Cena tepla, Poptávka, Nákup tepla, FVE)
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

# --- 1. SIDEBAR: TRŽNÍ DATA A ZDROJE ---
with st.sidebar:
    st.header("📈 Tržní data")
    fwd_file = st.file_uploader("Nahraj FWD křivku", type=["xlsx"])
    if fwd_file:
        st.session_state.fwd_data = clean_df(pd.read_excel(fwd_file), is_fwd=True)
    
    if st.session_state.fwd_data is not None:
        years = sorted(st.session_state.fwd_data['datetime'].dt.year.unique())
        sel_year = st.selectbox("Vyber rok", years)
        df_yr = st.session_state.fwd_data[st.session_state.fwd_data['datetime'].dt.year == sel_year].copy().reset_index(drop=True)
        
        st.subheader("🛠️ Úprava cen")
        ee_shift = st.number_input("Posun EE [EUR/MWh]", value=0.0)
        gas_shift = st.number_input("Posun Plyn [EUR/MWh]", value=0.0)
        df_yr['ee_price_mod'] = df_yr['ee_price'] + ee_shift
        df_yr['gas_price_mod'] = df_yr['gas_price'] + gas_shift

    st.header("📋 Aktivní technologie")
    use_kgj = st.checkbox("Kogenerace (KGJ)", value=True)
    use_boil = st.checkbox("Plynový kotel", value=True)
    use_ek = st.checkbox("Elektrokotel", value=True)
    use_external_heat = st.checkbox("Povolit nákup tepla", value=True)

# --- 2. HLAVNÍ PLOCHA: GRAF KŘIVEK ---
if st.session_state.fwd_data is not None:
    fig_trh = make_subplots(specs=[[{"secondary_y": True}]])
    fig_trh.add_trace(go.Scatter(x=df_yr['datetime'], y=df_yr['ee_price_mod'], name="EE (mod)", line=dict(color='#00ff00')), secondary_y=False)
    fig_trh.add_trace(go.Scatter(x=df_yr['datetime'], y=df_yr['gas_price_mod'], name="Plyn (mod)", line=dict(color='red')), secondary_y=True)
    fig_trh.update_layout(height=300, title=f"Tržní křivky pro rok {sel_year}", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_trh, use_container_width=True)

st.subheader("📍 Data lokality")
loc_file = st.file_uploader("Nahraj aki11.xlsx", type=["xlsx"])
if loc_file:
    st.session_state.loc_data = clean_df(pd.read_excel(loc_file), is_fwd=False)

# --- 3. VÝPOČETNÍ JÁDRO ---
if st.session_state.fwd_data is not None and st.session_state.loc_data is not None:
    calc_df = pd.merge(df_yr, st.session_state.loc_data, on='mdh', how='inner').sort_values('datetime_x').reset_index(drop=True)

    st.markdown("### ⚙️ Technické parametry")
    col1, col2, col3 = st.columns(3)
    with col1:
        p_k_th = st.number_input("KGJ Tepelný výkon [MW]", value=1.09)
        p_k_el = st.number_input("KGJ Elektrický výkon [MW]", value=1.0)
        p_k_eff = st.number_input("Účinnost (tepelná)", value=0.46)
    with col2:
        p_b_max = st.number_input("Plynový kotel max [MW]", value=3.91)
        p_ek_max = st.number_input("Elektrokotel max [MW]", value=0.61)
    with col3:
        p_dist_in = st.number_input("Distribuce nákup EE [EUR]", value=33.0)
        p_h_cover = st.slider("Min. pokrytí tepla", 0.90, 1.0, 0.99)

    if st.button("🚀 SPUSTIT OPTIMALIZACI"):
        T = len(calc_df)
        model = pulp.LpProblem("Dispatcher", pulp.LpMaximize)

        # Definice limitů na základě checkboxů ze sidebar
        l_kgj = p_k_th if use_kgj else 0
        l_boil = p_b_max if use_boil else 0
        l_ek = p_ek_max if use_ek else 0

        q_kgj = pulp.LpVariable.dicts("q_KGJ", range(T), 0, l_kgj)
        q_boil = pulp.LpVariable.dicts("q_boiler", range(T), 0, l_boil)
        q_ek = pulp.LpVariable.dicts("q_eboiler", range(T), 0, l_ek)
        q_ext = pulp.LpVariable.dicts("q_external", range(T), 0)
        on = pulp.LpVariable.dicts("on", range(T), 0, 1, cat="Binary")

        # Koeficienty
        kgj_gas_ratio = (p_k_th / p_k_eff) / p_k_th
        kgj_el_ratio = p_k_el / p_k_th

        obj_terms = []
        for t in range(T):
            ee = calc_df.loc[t, 'ee_price_mod']
            gas = calc_df.loc[t, 'gas_price_mod']
            hp = calc_df.loc[t, 'heat_price']
            dem = calc_df.loc[t, 'heat_demand']
            h_req = p_h_cover * dem

            # Bilance tepla
            model += q_kgj[t] + q_boil[t] + q_ek[t] + q_ext[t] >= h_req
            model += q_kgj[t] <= l_kgj * on[t]
            
            if not use_external_heat or 'external_heat_price' not in calc_df.columns:
                model += q_ext[t] == 0

            # Cashflow
            income = (hp * h_req) + (ee * q_kgj[t] * kgj_el_ratio)
            costs = (gas * (q_kgj[t] * kgj_gas_ratio + q_boil[t]/0.95)) + \
                    ((ee + p_dist_in) * (q_ek[t]/0.98)) + (12.0 * on[t])
            
            if use_external_heat and 'external_heat_price' in calc_df.columns:
                costs += (calc_df.loc[t, 'external_heat_price'] * q_ext[t])

            obj_terms.append(income - costs)

        model += pulp.lpSum(obj_terms)
        model.solve(pulp.PULP_CBC_CMD(msg=0))

        st.success(f"Hotovo! Hrubý zisk: {pulp.value(model.objective):,.0f} EUR")
        
        # Graf
        res_df = pd.DataFrame({
            'T': calc_df['datetime_x'],
            'KGJ': [q_kgj[t].value() for t in range(T)],
            'Kotel': [q_boil[t].value() for t in range(T)],
            'El-kotel': [q_ek[t].value() for t in range(T)],
            'Nákup': [q_ext[t].value() for t in range(T)]
        })
        fig_res = go.Figure()
        for col in ['KGJ', 'Kotel', 'El-kotel', 'Nákup']:
            fig_res.add_trace(go.Bar(x=res_df['T'], y=res_df[col], name=col))
        fig_res.update_layout(barmode='stack', title="Hodinové krytí tepla")
        st.plotly_chart(fig_res, use_container_width=True)

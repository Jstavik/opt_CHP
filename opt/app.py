import streamlit as st
import pandas as pd
import pulp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="KGJ Strategy Expert", layout="wide")

# Inicializace session state
if 'fwd_data' not in st.session_state: st.session_state.fwd_data = None
if 'loc_data' not in st.session_state: st.session_state.loc_data = None

st.title("🎯 KGJ Strategy & Asset Dispatcher")

# --- UNIVERZÁLNÍ IMPORT ---
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
        # Mapování podle tvého obrázku: 1. Datum, 2. Cena tepla, 3. Poptávka
        mapping = {df.columns[1]: 'heat_price', df.columns[2]: 'heat_demand'}
        # Hledání nákupu tepla a FVE podle názvu
        for col in df.columns:
            low_col = col.lower()
            if 'nákup' in low_col and 'tepl' in low_col: mapping[col] = 'external_heat_price'
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
    use_external_heat = st.checkbox("Povolit nákup tepla (ze souboru)", value=True)
    st.divider()
    use_fve = st.checkbox("Fotovoltaika (FVE)", value=False)
    use_bess = st.checkbox("Baterie (BESS)", value=False)

    st.header("📈 2. Tržní data")
    fwd_file = st.file_uploader("Nahraj FWD křivku", type=["xlsx"])
    if fwd_file:
        st.session_state.fwd_data = clean_df(pd.read_excel(fwd_file), is_fwd=True)
    
    if st.session_state.fwd_data is not None:
        years = sorted(st.session_state.fwd_data['datetime'].dt.year.unique())
        sel_year = st.selectbox("Rok výpočtu", years)
        df_yr = st.session_state.fwd_data[st.session_state.fwd_data['datetime'].dt.year == sel_year].copy().reset_index(drop=True)
        ee_shift = st.number_input("EE Shift [EUR]", value=0.0)
        gas_shift = st.number_input("Plyn Shift [EUR]", value=0.0)
        df_yr['ee_price_mod'] = df_yr['ee_price'] + ee_shift
        df_yr['gas_price_mod'] = df_yr['gas_price'] + gas_shift

# --- HLAVNÍ PLOCHA ---
st.subheader("📍 3. Data lokality")
loc_file = st.file_uploader("Nahraj lokální Excel (aki11)", type=["xlsx"])
if loc_file:
    st.session_state.loc_data = clean_df(pd.read_excel(loc_file), is_fwd=False)

if st.session_state.fwd_data is not None and st.session_state.loc_data is not None:
    # Merge dat
    calc_df = pd.merge(df_yr, st.session_state.loc_data, on='mdh', how='inner').sort_values('datetime_x').reset_index(drop=True)

    # --- NASTAVENÍ PARAMETRŮ (Zde jsou widgety dostupné hned) ---
    st.markdown("### ⚙️ Nastavení parametrů (Běhounkova)")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("Kogenerace")
        k_th = st.number_input("Max Tepelný výkon [MW]", value=1.09)
        k_el = st.number_input("Max Elektrický výkon [MW]", value=0.999)
        k_eff = st.number_input("Tepelná účinnost", value=0.46)
        k_serv = st.number_input("Servis [EUR/hod]", value=12.0)
        k_min_load = st.slider("Minimální zatížení [%]", 0, 100, 55) / 100
    with c2:
        st.info("Kotelny")
        b_max = st.number_input("Plynový kotel max [MW]", value=3.91)
        ek_max = st.number_input("Elektrokotel max [MW]", value=0.6056)
        dist_in = st.number_input("Distribuce nákup EE [EUR]", value=33.0)
        dist_out = st.number_input("Distribuce prodej EE [EUR]", value=2.0)
    with c3:
        st.info("Systém")
        h_cover = st.slider("Minimální pokrytí tepla", 0.9, 1.0, 0.99)
        if use_fve:
            fve_p = st.number_input("Instalovaný výkon FVE [kWp]", value=500)
        if use_bess:
            bess_cap = st.number_input("Kapacita BESS [MWh]", value=1.0)

    # --- TLAČÍTKO VÝPOČTU ---
    if st.button("🚀 SPUSTIT OPTIMALIZACI"):
        T = len(calc_df)
        model = pulp.LpProblem("Dispatcher", pulp.LpMaximize)

        # Proměnné
        q_kgj = pulp.LpVariable.dicts("q_KGJ", range(T), 0, k_th)
        q_boil = pulp.LpVariable.dicts("q_boiler", range(T), 0, b_max)
        q_ek = pulp.LpVariable.dicts("q_eboiler", range(T), 0, ek_max)
        q_ext = pulp.LpVariable.dicts("q_external", range(T), 0) # Nákup tepla
        on = pulp.LpVariable.dicts("on", range(T), 0, 1, cat="Binary")

        # Koeficienty z tvého behouvkova_opt.txt
        kgj_gas_per_heat = (k_th / k_eff) / k_th
        kgj_el_per_heat = k_el / k_th

        profit_list = []
        for t in range(T):
            ee = calc_df.loc[t, 'ee_price_mod']
            gas = calc_df.loc[t, 'gas_price_mod']
            hp = calc_df.loc[t, 'heat_price']
            demand = calc_df.loc[t, 'heat_demand']
            h_req = h_cover * demand

            # 1. Bilance tepla
            model += q_kgj[t] + q_boil[t] + q_ek[t] + q_ext[t] >= h_req
            
            # 2. KGJ Limity
            model += q_kgj[t] <= k_th * on[t]
            model += q_kgj[t] >= k_min_load * k_th * on[t]

            # 3. Nákup tepla (pokud není povolen, q_ext = 0)
            if not use_external_heat or 'external_heat_price' not in calc_df.columns:
                model += q_ext[t] == 0

            # 4. Ekonomika (Profit)
            # Tržba za teplo + Tržba za EE z KGJ (mínus distribuce ven)
            income = (hp * h_req) + ((ee - dist_out) * q_kgj[t] * kgj_el_per_heat)
            
            # Náklady (Plyn pro KGJ a Kotel + EE nákup pro EK + Servis)
            costs = (gas * (q_kgj[t] * kgj_gas_per_heat + q_boil[t]/0.95)) + \
                    ((ee + dist_in) * (q_ek[t]/0.98)) + \
                    (k_serv * on[t])
            
            # Pokud je nákup tepla povolen, připočteme cenu ze sloupce "Nákup tepla"
            if use_external_heat and 'external_heat_price' in calc_df.columns:
                costs += (calc_df.loc[t, 'external_heat_price'] * q_ext[t])

            profit_list.append(income - costs)

        model += pulp.lpSum(profit_list)
        model.solve(pulp.PULP_CBC_CMD(msg=0))

        # --- VÝSLEDKY ---
        st.success(f"Optimalizace dokončena. Hrubý zisk: {pulp.value(model.objective):,.0f} EUR")
        
        calc_df['res_kgj'] = [q_kgj[t].value() for t in range(T)]
        calc_df['res_boil'] = [q_boil[t].value() for t in range(T)]
        calc_df['res_ek'] = [q_ek[t].value() for t in range(T)]
        calc_df['res_ext'] = [q_ext[t].value() for t in range(T)]
        
        # Grafy marginů a dispatchu
        fig = go.Figure()
        fig.add_trace(go.Bar(x=calc_df['datetime_x'], y=calc_df['res_kgj'], name="KGJ", marker_color='orange'))
        fig.add_trace(go.Bar(x=calc_df['datetime_x'], y=calc_df['res_boil'], name="Plynový kotel", marker_color='blue'))
        fig.add_trace(go.Bar(x=calc_df['datetime_x'], y=calc_df['res_ek'], name="Elektrokotel", marker_color='green'))
        fig.add_trace(go.Bar(x=calc_df['datetime_x'], y=calc_df['res_ext'], name="Nákup tepla", marker_color='purple'))
        fig.update_layout(barmode='stack', title="Hodinový Dispatch zdrojů tepla")
        st.plotly_chart(fig, use_container_width=True)

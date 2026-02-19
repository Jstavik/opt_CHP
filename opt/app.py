import streamlit as st
import pandas as pd
import pulp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- KONFIGURACE ---
st.set_page_config(page_title="KGJ Strategy Expert", layout="wide")

if 'fwd_data' not in st.session_state: st.session_state.fwd_data = None
if 'loc_data' not in st.session_state: st.session_state.loc_data = None

st.title("🎯 KGJ Integrated Strategy & Asset Dispatcher")

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
        # Mapování pro lokalitu - hledáme klíčová slova
        mapping = {df.columns[1]: 'heat_price', df.columns[2]: 'heat_demand'}
        for col in df.columns:
            if 'fve' in col.lower(): mapping[col] = 'fve_gen'
        df = df.rename(columns=mapping)
    
    for col in df.columns:
        if col not in ['datetime', 'mdh']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# --- SIDEBAR: TRŽNÍ DATA A AKTIVACE ZDROJŮ ---
with st.sidebar:
    st.header("📈 1. Tržní data")
    fwd_file = st.file_uploader("Nahraj FWD křivku (EE/ZP)", type=["xlsx"])
    if fwd_file:
        st.session_state.fwd_data = clean_df(pd.read_excel(fwd_file), is_fwd=True)
    
    if st.session_state.fwd_data is not None:
        years = sorted(st.session_state.fwd_data['datetime'].dt.year.unique())
        sel_year = st.selectbox("Rok pro výpočet", years)
        df_yr = st.session_state.fwd_data[st.session_state.fwd_data['datetime'].dt.year == sel_year].copy().reset_index(drop=True)
        
        st.subheader("🛠️ Úprava Base cen")
        ee_shift = st.number_input("Posun EE [EUR/MWh]", value=0.0)
        gas_shift = st.number_input("Posun Plyn [EUR/MWh]", value=0.0)
        df_yr['ee_price_mod'] = df_yr['ee_price'] + ee_shift
        df_yr['gas_price_mod'] = df_yr['gas_price'] + gas_shift

    st.header("📋 2. Aktivní technologie")
    use_kgj = st.checkbox("Kogenerace (KGJ)", value=True)
    use_boil = st.checkbox("Plynový kotel", value=True)
    use_ek = st.checkbox("Elektrokotel", value=True)
    st.divider()
    use_fve = st.checkbox("Fotovoltaika (FVE)", value=False)
    use_bess = st.checkbox("Baterie (BESS)", value=False)
    use_acc = st.checkbox("Akumulace tepla (TES)", value=False)

# --- HLAVNÍ PLOCHA: GRAF TRHU ---
if st.session_state.fwd_data is not None:
    st.subheader(f"📊 Náhled tržních křivek pro rok {sel_year}")
    fig_m = make_subplots(specs=[[{"secondary_y": True}]])
    fig_m.add_trace(go.Scatter(x=df_yr['datetime'], y=df_yr['ee_price_mod'], name="EE Upravená [EUR]", line=dict(color='#00ff00')), secondary_y=False)
    fig_m.add_trace(go.Scatter(x=df_yr['datetime'], y=df_yr['gas_price_mod'], name="Plyn Upravený [EUR]", line=dict(color='red')), secondary_y=True)
    fig_m.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig_m, use_container_width=True)

st.subheader("📍 3. Data lokality")
loc_file = st.file_uploader("Nahraj data lokality (aki11)", type=["xlsx"])
if loc_file:
    st.session_state.loc_data = clean_df(pd.read_excel(loc_file), is_fwd=False)

# --- PARAMETRY A VÝPOČET ---
if st.session_state.fwd_data is not None and st.session_state.loc_data is not None:
    calc_df = pd.merge(df_yr, st.session_state.loc_data, on='mdh', how='inner').sort_values('datetime_x').reset_index(drop=True)

    with st.form("param_form"):
        st.markdown("### ⚙️ Technické nastavení zdrojů")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info("Kogenerace (Běhounkova parametry)")
            k_th = st.number_input("Max Tepelný výkon [MW]", value=1.09)
            k_el = st.number_input("Max Elektrický výkon [MW]", value=0.999)
            k_eff = st.number_input("Tepelná účinnost", value=0.46)
            k_serv = st.number_input("Servis [EUR/hod]", value=12.0)
            k_min = st.slider("Minimální zatížení [%]", 0, 100, 55) / 100
            min_up = st.number_input("Min. doba běhu [hod]", value=4)
            min_down = st.number_input("Min. doba klidu [hod]", value=4)
        with c2:
            st.info("Kotelna a Distribuce")
            b_max = st.number_input("Plyn. kotel max [MW]", value=3.91)
            b_eff = st.number_input("Účinnost kotle", value=0.95)
            ek_max = st.number_input("Elektrokotel max [MW]", value=0.6056)
            ek_eff = st.number_input("Účinnost el. kotle", value=0.98)
            dist_in = st.number_input("Distribuce nákup EE [EUR]", value=33.0)
            dist_out = st.number_input("Distribuce prodej EE [EUR]", value=2.0)
        with c3:
            st.info("OZE a Systém")
            h_cover = st.slider("Minimální pokrytí tepla", 0.9, 1.0, 0.99)
            if use_fve:
                fve_scale = st.number_input("FVE Scale (násobič výkonu v datech)", value=1.0)
            if use_bess:
                bess_cap = st.number_input("Kapacita BESS [MWh]", value=1.0)
                bess_eff = st.number_input("Roundtrip účinnost (0.85)", value=0.85)

        if st.form_submit_button("🚀 SPUSTIT OPTIMALIZACI"):
            T = len(calc_df)
            prob = pulp.LpProblem("Dispatcher", pulp.LpMaximize)
            
            # Proměnné (Logika ze souboru behouvkova_opt.txt) [cite: 13]
            q_kgj = pulp.LpVariable.dicts("q_KGJ", range(T), 0, k_th)
            q_boil = pulp.LpVariable.dicts("q_boiler", range(T), 0, b_max)
            q_ek = pulp.LpVariable.dicts("q_eboiler", range(T), 0, ek_max)
            on = pulp.LpVariable.dicts("on", range(T), 0, 1, cat="Binary")
            
            kgj_gas_per_heat = (k_th / k_eff) / k_th [cite: 12]
            kgj_el_per_heat = k_el / k_th [cite: 12]

            profits = []
            for t in range(T):
                ee = calc_df.loc[t, 'ee_price_mod']
                gas = calc_df.loc[t, 'gas_price_mod']
                hp = calc_df.loc[t, 'heat_price']
                demand = calc_df.loc[t, 'heat_demand']
                
                # Bilance tepla [cite: 14]
                prob += q_kgj[t] + q_boil[t] + q_ek[t] >= h_cover * demand
                prob += q_kgj[t] <= k_th * on[t]
                prob += q_kgj[t] >= k_min * k_th * on[t]

                # Cashflow (Tržba teplo + EE - Plyn - Nákup EE - Servis) [cite: 16, 17]
                income = (hp * h_cover * demand) + ( (ee - dist_out) * q_kgj[t] * kgj_el_per_heat)
                costs = (gas * (q_kgj[t] * kgj_gas_per_heat + q_boil[t]/b_eff)) + \
                        ((ee + dist_in) * (q_ek[t]/ek_eff)) + (k_serv * on[t])
                profits.append(income - costs)

            prob += pulp.lpSum(profits)
            prob.solve(pulp.PULP_CBC_CMD(msg=0))

            st.success(f"Hotovo! Odhadovaný hrubý zisk: {pulp.value(prob.objective):,.0f} EUR")
            
            # Zobrazení výsledků jako minule...
            calc_df['res_kgj'] = [q_kgj[t].value() for t in range(T)]
            calc_df['res_boil'] = [q_boil[t].value() for t in range(T)]
            calc_df['res_ek'] = [q_ek[t].value() for t in range(T)]
            
            fig_res = go.Figure()
            fig_res.add_trace(go.Bar(x=calc_df['datetime_x'], y=calc_df['res_kgj'], name="KGJ", marker_color='orange'))
            fig_res.add_trace(go.Bar(x=calc_df['datetime_x'], y=calc_df['res_boil'], name="Kotel", marker_color='blue'))
            fig_res.add_trace(go.Bar(x=calc_df['datetime_x'], y=calc_df['res_ek'], name="El. kotel", marker_color='green'))
            st.plotly_chart(fig_res, use_container_width=True)

import streamlit as st
import pandas as pd
import pulp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- KONFIGURACE ---
st.set_page_config(page_title="KGJ Strategy Expert", layout="wide")

if 'fwd_data' not in st.session_state: st.session_state.fwd_data = None
if 'loc_data' not in st.session_state: st.session_state.loc_data = None

st.title("🎯 KGJ Integrated Dispatcher & Optimizer")

# --- UNIVERZÁLNÍ ČISTIČ DAT ---
def clean_df(df, is_fwd=True):
    df = df.dropna(how='all').dropna(axis=1, how='all')
    df.columns = [str(c).replace('\n', ' ').strip() for c in df.columns]
    
    # Datum je vždy první
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
    df = df.dropna(subset=[date_col]).rename(columns={date_col: 'datetime'})
    
    if is_fwd:
        df = df.rename(columns={df.columns[1]: 'ee_price', df.columns[2]: 'gas_price'})
    else:
        # Poptávka a ceny tepla z lokality
        df = df.rename(columns={df.columns[1]: 'heat_price', df.columns[2]: 'heat_demand'})
    
    for col in df.columns:
        if col != 'datetime':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# --- 1. SEKCE: TRŽNÍ DATA ---
st.sidebar.header("📈 Tržní FWD Křivky")
fwd_file = st.sidebar.file_uploader("1. Nahraj FWD ceny (EE/ZP)", type=["xlsx"])

if fwd_file:
    st.session_state.fwd_data = clean_df(pd.read_excel(fwd_file), is_fwd=True)

if st.session_state.fwd_data is not None:
    fwd_df = st.session_state.fwd_data
    years = sorted(fwd_df['datetime'].dt.year.unique())
    sel_year = st.sidebar.selectbox("Vyber rok pro analýzu", years)
    df_yr = fwd_df[fwd_df['datetime'].dt.year == sel_year].copy().reset_index(drop=True)
    
    st.sidebar.subheader("🛠️ Úprava cen")
    ee_shift = st.sidebar.number_input("Posun EE [EUR/MWh]", value=0.0)
    gas_shift = st.sidebar.number_input("Posun Plyn [EUR/MWh]", value=0.0)
    df_yr['ee_price_mod'] = df_yr['ee_price'] + ee_shift
    df_yr['gas_price_mod'] = df_yr['gas_price'] + gas_shift

# --- 2. SEKCE: LOKALITA ---
st.subheader("📍 Data lokality")
loc_file = st.file_uploader("2. Nahraj data lokality (aki11)", type=["xlsx"])

if loc_file:
    st.session_state.loc_data = clean_df(pd.read_excel(loc_file), is_fwd=False)

# --- 3. SEKCE: VÝPOČET A PARAMETRY ---
if st.session_state.fwd_data is not None and st.session_state.loc_data is not None:
    # Synchronizace přes MDH
    df_yr['mdh'] = df_yr['datetime'].dt.strftime('%m-%d-%H')
    loc_tmp = st.session_state.loc_data.copy()
    loc_tmp['mdh'] = loc_tmp['datetime'].dt.strftime('%m-%d-%H')
    
    calc_df = pd.merge(df_yr, loc_tmp[['mdh', 'heat_price', 'heat_demand']], on='mdh', how='inner')
    calc_df = calc_df.sort_values('datetime').reset_index(drop=True)

    with st.expander("⚙️ Kompletní nastavení zdrojů", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Kogenerace (KGJ)**")
            k_th = st.number_input("Tepelný výkon [MW]", value=1.09)
            k_el = st.number_input("Elektrický výkon [MW]", value=0.99)
            k_eff = st.number_input("Tepelná účinnost (0.46 = 46%)", value=0.46)
            k_serv = st.number_input("Servisní náklad [EUR/hod]", value=12.0)
        with c2:
            st.markdown("**Plynový kotel**")
            b_max = st.number_input("Max výkon kotle [MW]", value=4.0)
            b_eff = st.number_input("Účinnost kotle (0.95)", value=0.95)
        with c3:
            st.markdown("**Elektrokotel / Ostatní**")
            ek_max = st.number_input("Max výkon El. kotle [MW]", value=0.6)
            ek_eff = st.number_input("Účinnost El. kotle", value=0.98)
            dist_ee = st.number_input("Distribuce nákup EE [EUR/MWh]", value=33.0)

    if st.button("🚀 SPUSTIT KOMPLETNÍ OPTIMALIZACI"):
        T = len(calc_df)
        prob = pulp.LpProblem("Dispatcher", pulp.LpMaximize)
        
        # Proměnné
        q_kgj = pulp.LpVariable.dicts("q_kgj", range(T), 0, k_th)
        on_kgj = pulp.LpVariable.dicts("on_kgj", range(T), 0, 1, cat="Binary")
        q_boil = pulp.LpVariable.dicts("q_boil", range(T), 0, b_max)
        q_ek = pulp.LpVariable.dicts("q_ek", range(T), 0, ek_max)

        # Ekonomické koeficienty
        kgj_gas_input_per_mw_th = (1 / k_eff)
        kgj_el_gen_per_mw_th = (k_el / k_th)

        profits = []
        for t in range(T):
            # Ceny v dané hodině
            ee = calc_df.loc[t, 'ee_price_mod']
            gas = calc_df.loc[t, 'gas_price_mod']
            hp = calc_df.loc[t, 'heat_price']
            demand = calc_df.loc[t, 'heat_demand']

            # 1. Bilance tepla (musíme pokrýt poptávku)
            prob += q_kgj[t] + q_boil[t] + q_ek[t] >= 0.99 * demand
            
            # 2. Omezení KGJ (min/max a vazba na ON/OFF)
            prob += q_kgj[t] <= k_th * on_kgj[t]
            prob += q_kgj[t] >= 0.55 * k_th * on_kgj[t]

            # 3. CASHFLOW MODEL
            # Příjmy: prodej tepla + prodej elektřiny z KGJ
            income = (hp * (q_kgj[t] + q_boil[t] + q_ek[t])) + (ee * q_kgj[t] * kgj_el_gen_per_mw_th)
            # Náklady: plyn pro KGJ + plyn pro kotel + elektřina pro el. kotel + servis KGJ
            costs = (gas * (q_kgj[t] * kgj_gas_input_per_mw_th)) + \
                    (gas * (q_boil[t] / b_eff)) + \
                    ((ee + dist_ee) * (q_ek[t] / ek_eff)) + \
                    (k_serv * on_kgj[t])
            
            profits.append(income - costs)

        prob += pulp.lpSum(profits)
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        # --- VÝSLEDKY ---
        st.success(f"Optimalizace dokončena. Celkový hrubý zisk: {pulp.value(prob.objective):,.0f} EUR")
        
        calc_df['res_kgj'] = [q_kgj[t].value() for t in range(T)]
        calc_df['res_boil'] = [q_boil[t].value() for t in range(T)]
        calc_df['res_ek'] = [q_ek[t].value() for t in range(T)]
        calc_df['hourly_profit'] = [pulp.value(profits[t]) for t in range(T)]
        calc_df['cum_profit'] = calc_df['hourly_profit'].cumsum()

        # Graf 1: Dispatch tepla
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=calc_df['datetime'], y=calc_df['heat_demand'], name="Poptávka", line=dict(color='black', width=1)))
        fig1.add_trace(go.Bar(x=calc_df['datetime'], y=calc_df['res_kgj'], name="KGJ", marker_color='orange'))
        fig1.add_trace(go.Bar(x=calc_df['datetime'], y=calc_df['res_boil'], name="Kotel", marker_color='blue'))
        fig1.add_trace(go.Bar(x=calc_df['datetime'], y=calc_df['res_ek'], name="El. kotel", marker_color='green'))
        fig1.add_layout_image(dict(source="https://raw.githubusercontent.com/pulp-platform/pulp/master/pulp.png", x=0, y=1))
        fig1.update_layout(title="Hodinový Dispatch tepla [MW]", barmode='stack', height=400)
        st.plotly_chart(fig1, use_container_width=True)

        # Graf 2: Kumulativní zisk
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=calc_df['datetime'], y=calc_df['cum_profit'], name="Profit", fill='tozeroy', line=dict(color='green')))
        fig2.update_layout(title="Kumulativní hrubý zisk [EUR]", height=300)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Summary tabulka
        c1, c2, c3 = st.columns(3)
        c1.metric("Provozní hodiny KGJ", f"{int(sum([on_kgj[t].value() for t in range(T)]))} h")
        c2.metric("Průměrná marže KGJ", f"{pulp.value(prob.objective)/max(1, sum([on_kgj[t].value() for t in range(T)])):.2f} EUR/h")
        c3.metric("Max. hodinový profit", f"{calc_df['hourly_profit'].max():.2f} EUR")

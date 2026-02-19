import streamlit as st
import pandas as pd
import pulp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- KONFIGURACE STRÁNKY ---
st.set_page_config(page_title="KGJ Strategy Expert", layout="wide")

if 'fwd_data' not in st.session_state: st.session_state.fwd_data = None
if 'loc_data' not in st.session_state: st.session_state.loc_data = None

st.title("🎯 KGJ Integrated Strategy Optimizer")

# --- POMOCNÁ FUNKCE PRO ČIŠTĚNÍ DAT ---
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
        df = df.rename(columns={df.columns[1]: 'heat_price', df.columns[2]: 'heat_demand'})
    
    for col in df.columns:
        if col not in ['datetime', 'mdh']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# --- SIDEBAR: VÝBĚR DAT A TECHNOLOGIÍ ---
with st.sidebar:
    st.header("📋 Vstupy a Technologie")
    
    with st.expander("🔌 Aktivní zdroje", expanded=True):
        use_kgj = st.checkbox("Kogenerace (KGJ)", value=True)
        use_boiler = st.checkbox("Plynový kotel", value=True)
        use_eboiler = st.checkbox("Elektrokotel", value=True)
        st.divider()
        use_acc = st.checkbox("Akumulační nádrž (TES)", value=False)
        use_bess = st.checkbox("Baterie (BESS)", value=False)
        use_fve = st.checkbox("Fotovoltaika (FVE)", value=False)

    st.header("📈 Tržní data")
    fwd_file = st.file_uploader("Nahraj FWD křivku", type=["xlsx"])
    if fwd_file:
        st.session_state.fwd_data = clean_df(pd.read_excel(fwd_file), is_fwd=True)

    if st.session_state.fwd_data is not None:
        years = sorted(st.session_state.fwd_data['datetime'].dt.year.unique())
        sel_year = st.selectbox("Rok pro analýzu", years)
        df_yr = st.session_state.fwd_data[st.session_state.fwd_data['datetime'].dt.year == sel_year].copy().reset_index(drop=True)
        
        st.subheader("🛠️ Shift cen")
        ee_shift = st.number_input("EE Shift [EUR]", value=0.0)
        gas_shift = st.number_input("Plyn Shift [EUR]", value=0.0)
        df_yr['ee_price_mod'] = df_yr['ee_price'] + ee_shift
        df_yr['gas_price_mod'] = df_yr['gas_price'] + gas_shift

# --- HLAVNÍ PLOCHA: NASTAVENÍ PARAMETRŮ ---
if st.session_state.fwd_data is not None:
    # Graf trhu (vždy užitečné vidět, co počítám)
    fig_m = make_subplots(specs=[[{"secondary_y": True}]])
    fig_m.add_trace(go.Scatter(x=df_yr['datetime'], y=df_yr['ee_price_mod'], name="EE Cena", line=dict(color='#00ff00')), secondary_y=False)
    fig_m.add_trace(go.Scatter(x=df_yr['datetime'], y=df_yr['gas_price_mod'], name="Plyn Cena", line=dict(color='red')), secondary_y=True)
    st.plotly_chart(fig_m, use_container_width=True)

st.subheader("📍 Parametry lokality a technologií")
loc_file = st.file_uploader("Nahraj data lokality (aki11)", type=["xlsx"])

if loc_file:
    st.session_state.loc_data = clean_df(pd.read_excel(loc_file), is_fwd=False)

if st.session_state.fwd_data is not None and st.session_state.loc_data is not None:
    # Merge dat
    loc_tmp = st.session_state.loc_data.copy()
    calc_df = pd.merge(df_yr, loc_tmp[['mdh', 'heat_price', 'heat_demand']], on='mdh', how='inner').sort_values('datetime').reset_index(drop=True)

    # Parametry přesně podle behouvkova_opt.txt 
    with st.form("param_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**KGJ Parametry**")
            k_th = st.number_input("Max Tepelný výkon [MW]", value=1.09)
            k_el = st.number_input("Max Elektrický výkon [MW]", value=0.999)
            k_eff = st.number_input("Tepelná účinnost", value=0.46)
            k_min = st.slider("Minimální zatížení [%]", 0, 100, 55) / 100
            k_serv = st.number_input("Servis [EUR/hod]", value=12.0)
            min_up = st.number_input("Min. doba běhu [hod]", value=4)
            min_down = st.number_input("Min. doba klidu [hod]", value=4)
        
        with col2:
            st.markdown("**Ostatní zdroje**")
            b_max = st.number_input("Plynový kotel max [MW]", value=3.91)
            b_eff = st.number_input("Účinnost kotle", value=0.95)
            ek_max = st.number_input("Elektrokotel max [MW]", value=0.6056)
            ek_eff = st.number_input("Účinnost elektrokotle", value=0.98)
            dist_ee = st.number_input("Distribuce EE nákup [EUR]", value=33.0)
        
        with col3:
            st.markdown("**Systémové**")
            h_cover = st.slider("Minimální pokrytí tepla", 0.90, 1.0, 0.99)
            init_state = st.selectbox("Počáteční stav KGJ", [0, 1], index=0)
            
        submit = st.form_submit_button("🚀 SPUSTIT VÝPOČET")

    if submit:
        T = len(calc_df)
        model = pulp.LpProblem("Dispatch_Optimization", pulp.LpMaximize)

        # Proměnné [cite: 2]
        q_kgj = pulp.LpVariable.dicts("q_KGJ", range(T), 0, k_th)
        q_boiler = pulp.LpVariable.dicts("q_boiler", range(T), 0, b_max)
        q_eboiler = pulp.LpVariable.dicts("q_eboiler", range(T), 0, ek_max)
        kgj_on = pulp.LpVariable.dicts("KGJ_on", range(T), 0, 1, cat="Binary")
        kgj_start = pulp.LpVariable.dicts("KGJ_start", range(T), 0, 1, cat="Binary")
        kgj_stop = pulp.LpVariable.dicts("KGJ_stop", range(T), 0, 1, cat="Binary")

        # Koeficienty z orig. kódu 
        kgj_gas_per_heat = (k_th / k_eff) / k_th
        kgj_el_per_heat = k_el / k_th

        profit_list = []
        for t in range(T):
            ee = calc_df.loc[t, "ee_price_mod"]
            gas = calc_df.loc[t, "gas_price_mod"]
            hp = calc_df.loc[t, "heat_price"]
            demand = calc_df.loc[t, "heat_demand"]
            h_req = h_cover * demand

            # 1. Bilance tepla [cite: 3]
            model += q_kgj[t] + q_boiler[t] + q_eboiler[t] >= h_req
            
            # 2. KGJ Limity [cite: 3]
            model += q_kgj[t] <= k_th * kgj_on[t]
            model += q_kgj[t] >= k_min * k_th * kgj_on[t]

            # 3. ON/OFF logika (Min Up/Down) 
            if t > 0:
                model += kgj_on[t] - kgj_on[t-1] == kgj_start[t] - kgj_stop[t]
            else:
                model += kgj_on[t] - init_state == kgj_start[t] - kgj_stop[t]

            # Cashflow t-té hodiny [cite: 5, 6]
            income = (hp * h_req) + (ee * q_kgj[t] * kgj_el_per_heat)
            costs = (gas * (q_kgj[t] * kgj_gas_per_heat + q_boiler[t] / b_eff)) + \
                    ((ee + dist_ee) * (q_eboiler[t] / ek_eff)) + \
                    (k_serv * kgj_on[t])
            profit_list.append(income - costs)

        # Min Up/Down constraints 
        for t in range(T - int(min_up)):
            model += pulp.lpSum(kgj_on[t+i] for i in range(int(min_up))) >= min_up * kgj_start[t]
        for t in range(T - int(min_down)):
            model += pulp.lpSum(1 - kgj_on[t+i] for i in range(int(min_down))) >= min_down * kgj_stop[t]

        model += pulp.lpSum(profit_list)
        model.solve(pulp.PULP_CBC_CMD(msg=0))

        # --- VÝSTUPY ---
        st.success(f"Optimalizace dokončena. Celkový zisk: {pulp.value(model.objective):,.0f} EUR")
        
        calc_df['KGJ_Output'] = [q_kgj[t].value() for t in range(T)]
        calc_df['Boiler_Output'] = [q_boiler[t].value() for t in range(T)]
        calc_df['EBoiler_Output'] = [q_eboiler[t].value() for t in range(T)]
        
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(x=calc_df['datetime'], y=calc_df['heat_demand'], name="Poptávka", line=dict(color='black')))
        fig_res.add_trace(go.Bar(x=calc_df['datetime'], y=calc_df['KGJ_Output'], name="KGJ [MW]", marker_color='orange'))
        fig_res.add_trace(go.Bar(x=calc_df['datetime'], y=calc_df['Boiler_Output'], name="Kotel [MW]", marker_color='blue'))
        fig_res.add_trace(go.Bar(x=calc_df['datetime'], y=calc_df['EBoiler_Output'], name="Elektrokotel [MW]", marker_color='green'))
        fig_res.update_layout(barmode='stack', title="Hodinový Dispatch")
        st.plotly_chart(fig_res, use_container_width=True)

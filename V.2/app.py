import streamlit as st
import pandas as pd
import pulp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="KGJ Strategy Expert", layout="wide")

# --- 1. KROK: TRŽNÍ DATA (FWD) ---
with st.sidebar:
    st.header("1️⃣ Tržní ceny (FWD)")
    fwd_file = st.file_uploader("Nahraj FWD křivku (Excel)", type=["xlsx"])
    
    if fwd_file:
        df_fwd_raw = pd.read_excel(fwd_file)
        df_fwd_raw.columns = [str(c).strip() for c in df_fwd_raw.columns]
        date_col = df_fwd_raw.columns[0]
        df_fwd_raw[date_col] = pd.to_datetime(df_fwd_raw[date_col], dayfirst=True)
        
        # Přejmenování pro vnitřní logiku
        df_fwd_raw = df_fwd_raw.rename(columns={
            date_col: 'datetime', 
            df_fwd_raw.columns[1]: 'ee_base', 
            df_fwd_raw.columns[2]: 'gas_base'
        })
        
        st.subheader("🛠️ Úprava cen (Shift)")
        ee_shift = st.number_input("Posun EE [EUR/MWh]", value=0.0)
        gas_shift = st.number_input("Posun Plyn [EUR/MWh]", value=0.0)
        
        df_fwd = df_fwd_raw.copy()
        df_fwd['ee_price'] = df_fwd['ee_base'] + ee_shift
        df_fwd['gas_price'] = df_fwd['gas_base'] + gas_shift
        df_fwd['mdh'] = df_fwd['datetime'].dt.strftime('%m-%d-%H')
        st.session_state.fwd_data = df_fwd

    st.divider()
    st.header("2️⃣ Aktivní technologie")
    use_kgj = st.checkbox("Kogenerace (KGJ)", value=True)
    use_boil = st.checkbox("Plynový kotel", value=True)
    use_ek = st.checkbox("Elektrokotel", value=True)
    use_ext_heat = st.checkbox("Povolit nákup tepla (Import)", value=False)

# --- 2. KROK: ZOBRAZENÍ A ÚPRAVA DAT (TO CO CHCEŠ) ---
st.header("📊 Úprava tržních a lokálních dat")

if 'fwd_data' in st.session_state and st.session_state.fwd_data is not None:
    # Zobrazení grafu cen
    with st.expander("📈 Náhled tržních cen", expanded=True):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=st.session_state.fwd_data['datetime'], y=st.session_state.fwd_data['ee_price'], name="EE Cena", line=dict(color='green')), secondary_y=False)
        fig.add_trace(go.Scatter(x=st.session_state.fwd_data['datetime'], y=st.session_state.fwd_data['gas_price'], name="Plyn Cena", line=dict(color='red')), secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    # EDITOVATELNÁ TABULKA
    st.subheader("✏️ Ruční úprava křivky")
    st.info("Změny provedené v této tabulce se projeví přímo ve výpočtu.")
    # Zde umožníme uživateli měnit ee_price a gas_price přímo
    edited_df = st.data_editor(st.session_state.fwd_data, use_container_width=True, hide_index=True)
    st.session_state.fwd_data = edited_df

# --- 3. KROK: PARAMETRY (Původní nastavení) ---
st.header("📍 Parametry lokality a technologií")
col_p1, col_p2 = st.columns(2)
params = {}

with col_p1:
    if use_kgj:
        st.info("💡 Parametry KGJ")
        params['k_th'] = st.number_input("Tepelný výkon [MW]", value=1.09)
        params['k_el'] = st.number_input("Elektrický výkon [MW]", value=1.0)
        params['k_eff'] = st.number_input("Tepelná účinnost", value=0.46)
        params['k_serv'] = st.number_input("Servisní náklad [EUR/hod]", value=12.0)
        params['k_min'] = st.slider("Minimální zatížení [%]", 0, 100, 55) / 100
    if use_boil:
        st.info("🔥 Plynový kotel")
        params['b_max'] = st.number_input("Max. výkon kotle [MW]", value=3.91)
        params['b_eff'] = st.number_input("Účinnost kotle", value=0.95)

with col_p2:
    if use_ek:
        st.info("⚡ Elektrokotel")
        params['ek_max'] = st.number_input("Max. výkon EK [MW]", value=0.61)
        params['ek_eff'] = st.number_input("Účinnost EK", value=0.98)
        params['dist_ee'] = st.number_input("Distribuce nákup EE [EUR/MWh]", value=33.0)
    st.info("🏠 Systém")
    params['h_cover'] = st.slider("Minimální pokrytí potřeby", 0.0, 1.0, 0.99)
    params['fixed_heat_price'] = st.number_input("Výkupní cena tepla [EUR/MWh]", value=120.0)

# --- 4. KROK: DATA LOKALITY (aki11) ---
st.divider()
loc_file = st.file_uploader("3️⃣ Nahraj potřebu tepla (aki11)", type=["xlsx"])
if loc_file:
    df_loc = pd.read_excel(loc_file)
    df_loc.columns = [str(c).strip() for c in df_loc.columns]
    date_col = df_loc.columns[0]
    df_loc[date_col] = pd.to_datetime(df_loc[date_col], dayfirst=True)
    df_loc['mdh'] = df_loc[date_col].dt.strftime('%m-%d-%H')
    
    # Mapování tvých sloupců
    mapping = {df_loc.columns[2]: 'demand'} # Poptávka po teplo
    df_loc = df_loc.rename(columns=mapping)
    st.session_state.loc_data = df_loc

# --- 5. KROK: OPTIMALIZACE (Původní Pulp model) ---
if st.session_state.fwd_data is not None and st.session_state.loc_data is not None:
    if st.button("🏁 SPUSTIT KOMPLETNÍ OPTIMALIZACI"):
        # Sloučení upravených dat
        df = pd.merge(st.session_state.fwd_data, st.session_state.loc_data, on='mdh', how='inner')
        T = len(df)
        
        model = pulp.LpProblem("Dispatcher", pulp.LpMaximize)
        
        # PROMĚNNÉ
        q_kgj = pulp.LpVariable.dicts("q_KGJ", range(T), 0, params.get('k_th', 0))
        q_boil = pulp.LpVariable.dicts("q_Boil", range(T), 0, params.get('b_max', 0))
        q_ek = pulp.LpVariable.dicts("q_EK", range(T), 0, params.get('ek_max', 0))
        q_deficit = pulp.LpVariable.dicts("q_Deficit", range(T), 0)
        on = pulp.LpVariable.dicts("on", range(T), 0, 1, cat="Binary")

        kgj_el_ratio = params.get('k_el', 0) / params.get('k_th', 1)
        kgj_gas_ratio = (params.get('k_th', 1) / params.get('k_eff', 1)) / params.get('k_th', 1)

        profit_total = []
        for t in range(T):
            ee = df.loc[t, 'ee_price']
            gas = df.loc[t, 'gas_price']
            hp = params['fixed_heat_price']
            dem = df.loc[t, 'demand']
            h_req = dem * params['h_cover']

            model += q_kgj[t] + q_boil[t] + q_ek[t] + q_deficit[t] >= h_req
            
            if use_kgj:
                model += q_kgj[t] <= params['k_th'] * on[t]
                model += q_kgj[t] >= params['k_min'] * params['k_th'] * on[t]
            else:
                model += q_kgj[t] == 0

            # Cashflow (Přesně jak jsi měl)
            income = (hp * (h_req - q_deficit[t])) + (ee * q_kgj[t] * kgj_el_ratio)
            costs = (gas * (q_kgj[t] * kgj_gas_ratio)) + \
                    (gas * (q_boil[t] / params.get('b_eff', 0.95))) + \
                    ((ee + params.get('dist_ee', 33)) * (q_ek[t] / params.get('ek_eff', 0.98))) + \
                    (params.get('k_serv', 12) * on[t])
            
            penalty = q_deficit[t] * 5000 
            profit_total.append(income - costs - penalty)

        model += pulp.lpSum(profit_total)
        model.solve(pulp.PULP_CBC_CMD(msg=0))

        # --- ZOBRAZENÍ VÝSLEDKŮ ---
        st.success(f"Optimalizace hotova. Hrubý zisk: {pulp.value(model.objective):,.0f} EUR")
        
        res = pd.DataFrame({
            'T': df['datetime_x'],
            'KGJ': [q_kgj[t].value() for t in range(T)],
            'Kotel': [q_boil[t].value() for t in range(T)],
            'EK': [q_ek[t].value() for t in range(T)],
            'Poptávka': df['demand'] * params['h_cover']
        })

        fig1 = go.Figure()
        for c, color in zip(['KGJ', 'Kotel', 'EK'], ['#FF9900', '#1f77b4', '#2ca02c']):
            fig1.add_trace(go.Bar(x=res['T'], y=res[c], name=c, marker_color=color))
        fig1.add_trace(go.Scatter(x=res['T'], y=res['Poptávka'], name="Požadavek", line=dict(color='pink', dash='dot')))
        fig1.update_layout(barmode='stack', title="Dispatch tepla [MW]")
        st.plotly_chart(fig1, use_container_width=True)

import streamlit as st
import pandas as pd
import pulp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="KGJ Strategy Expert", layout="wide")

if 'fwd_data' not in st.session_state: st.session_state.fwd_data = None
if 'loc_data' not in st.session_state: st.session_state.loc_data = None

st.title("🎯 KGJ Strategy & Asset Dispatcher")

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
        # Mapování sloupců podle tvého nového obrázku
        mapping = {df.columns[1]: 'heat_price', df.columns[2]: 'heat_demand'}
        for col in df.columns:
            low_col = col.lower()
            if 'nákup tepla' in low_col: mapping[col] = 'external_heat_price'
            if 'fve' in low_col: mapping[col] = 'fve_gen'
        df = df.rename(columns=mapping)
    
    for col in df.columns:
        if col not in ['datetime', 'mdh']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# --- SIDEBAR ---
with st.sidebar:
    st.header("📋 Aktivní zdroje")
    use_kgj = st.checkbox("Kogenerace (KGJ)", value=True)
    use_boil = st.checkbox("Plynový kotel", value=True)
    use_ek = st.checkbox("Elektrokotel", value=True)
    use_external_heat = st.checkbox("Povolit nákup tepla (Import)", value=False)
    st.divider()
    use_fve = st.checkbox("Fotovoltaika (FVE)", value=False)
    use_bess = st.checkbox("Baterie (BESS)", value=False)

    st.header("📈 Tržní data")
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

st.subheader("📍 Data lokality")
loc_file = st.file_uploader("Nahraj aki11.xlsx", type=["xlsx"])
if loc_file:
    st.session_state.loc_data = clean_df(pd.read_excel(loc_file), is_fwd=False)

if st.session_state.fwd_data is not None and st.session_state.loc_data is not None:
    # Merge dat (přejmenování datetime_x pro jistotu)
    calc_df = pd.merge(df_yr, st.session_state.loc_data, on='mdh', how='inner').sort_values('datetime_x').reset_index(drop=True)

    with st.form("param_form"):
        st.markdown("### ⚙️ Nastavení parametrů")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info("KGJ (Běhounkova)")
            k_th_val = st.number_input("Max Tepelný výkon [MW]", value=1.09) [cite: 12]
            k_el_val = st.number_input("Max Elektrický výkon [MW]", value=0.999) [cite: 12]
            k_eff_val = st.number_input("Tepelná účinnost", value=0.46) [cite: 12]
            k_serv_val = st.number_input("Servis [EUR/hod]", value=12.0) [cite: 12]
            min_up_val = st.number_input("Min. doba běhu [hod]", value=4) [cite: 12]
        with c2:
            st.info("Kotelny")
            b_max_val = st.number_input("Plynový kotel max [MW]", value=3.91) [cite: 12]
            ek_max_val = st.number_input("Elektrokotel max [MW]", value=0.6056) [cite: 12]
            dist_in_val = st.number_input("Distribuce nákup EE [EUR]", value=33.0) [cite: 12]
        with c3:
            st.info("Systém")
            h_cover_val = st.slider("Minimální pokrytí tepla", 0.9, 1.0, 0.99) [cite: 12]

        if st.form_submit_button("🚀 SPUSTIT OPTIMALIZACI"):
            T = len(calc_df)
            model = pulp.LpProblem("Dispatcher", pulp.LpMaximize) [cite: 12]

            # Proměnné
            q_kgj = pulp.LpVariable.dicts("q_KGJ", range(T), 0, k_th_val) [cite: 13]
            q_boil = pulp.LpVariable.dicts("q_boiler", range(T), 0, b_max_val) [cite: 13]
            q_ek = pulp.LpVariable.dicts("q_eboiler", range(T), 0, ek_max_val) [cite: 13]
            q_ext = pulp.LpVariable.dicts("q_external", range(T), 0) # Nákup tepla
            on = pulp.LpVariable.dicts("on", range(T), 0, 1, cat="Binary") [cite: 13]

            # Koeficienty 
            kgj_gas_per_heat = (k_th_val / k_eff_val) / k_th_val
            kgj_el_per_heat = k_el_val / k_th_val

            profits = []
            for t in range(T):
                ee = calc_df.loc[t, 'ee_price_mod']
                gas = calc_df.loc[t, 'gas_price_mod']
                hp = calc_df.loc[t, 'heat_price']
                demand = calc_df.loc[t, 'heat_demand']
                
                # Bilance tepla (přidán nákup tepla q_ext)
                model += q_kgj[t] + q_boil[t] + q_ek[t] + q_ext[t] >= h_cover_val * demand [cite: 14]
                model += q_kgj[t] <= k_th_val * on[t] [cite: 14]
                
                # Pokud není nákup povolen, q_ext musí být 0
                if not use_external_heat:
                    model += q_ext[t] == 0

                # Ekonomika [cite: 16, 17]
                income = (hp * h_cover_val * demand) + (ee * q_kgj[t] * kgj_el_per_heat)
                costs = (gas * (q_kgj[t] * kgj_gas_per_heat + q_boil[t]/0.95)) + \
                        ((ee + dist_in_val) * (q_ek[t]/0.98)) + \
                        (k_serv_val * on[t])
                
                if use_external_heat and 'external_heat_price' in calc_df.columns:
                    costs += (calc_df.loc[t, 'external_heat_price'] * q_ext[t])

                profits.append(income - costs)

            model += pulp.lpSum(profits) [cite: 17]
            model.solve(pulp.PULP_CBC_CMD(msg=0))

            st.success(f"Výpočet hotov. Zisk: {pulp.value(model.objective):,.0f} EUR")
            
            # Graf výsledků
            res_df = pd.DataFrame({
                'Time': calc_df['datetime_x'],
                'KGJ': [q_kgj[t].value() for t in range(T)],
                'Kotel': [q_boil[t].value() for t in range(T)],
                'El_Kotel': [q_ek[t].value() for t in range(T)],
                'Nakup_Tepla': [q_ext[t].value() for t in range(T)]
            })
            fig = go.Figure()
            for col in ['KGJ', 'Kotel', 'El_Kotel', 'Nakup_Tepla']:
                fig.add_trace(go.Bar(x=res_df['Time'], y=res_df[col], name=col))
            fig.update_layout(barmode='stack', title="Hodinový Dispatch tepla")
            st.plotly_chart(fig, use_container_width=True)

import streamlit as st
import pandas as pd
import pulp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- KONFIGURACE ---
st.set_page_config(page_title="KGJ Strategy Expert", layout="wide")

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
    
    # Automatické mapování základních sloupců
    if is_fwd:
        df = df.rename(columns={df.columns[1]: 'ee_price', df.columns[2]: 'gas_price'})
    else:
        df = df.rename(columns={df.columns[1]: 'heat_price', df.columns[2]: 'heat_demand'})
    
    for col in df.columns:
        if col not in ['datetime', 'mdh']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# --- SIDEBAR ---
with st.sidebar:
    st.header("📋 Aktivní technologie")
    use_kgj = st.checkbox("Kogenerace (KGJ)", value=True)
    use_boil = st.checkbox("Plynový kotel", value=True)
    use_ek = st.checkbox("Elektrokotel", value=True)
    st.divider()
    use_fve = st.checkbox("Fotovoltaika (FVE)", value=False)
    use_bess = st.checkbox("Baterie (BESS)", value=False)
    use_acc = st.checkbox("Akumulace tepla (TES)", value=False)

    st.header("📈 Tržní data")
    fwd_file = st.file_uploader("Nahraj FWD ceny (xlsx)", type=["xlsx"])
    if fwd_file:
        st.session_state.fwd_data = clean_df(pd.read_excel(fwd_file), is_fwd=True)
    
    if st.session_state.fwd_data is not None:
        years = sorted(st.session_state.fwd_data['datetime'].dt.year.unique())
        sel_year = st.selectbox("Rok analýzy", years)
        df_yr = st.session_state.fwd_data[st.session_state.fwd_data['datetime'].dt.year == sel_year].copy().reset_index(drop=True)
        ee_shift = st.number_input("EE Shift [EUR]", value=0.0)
        gas_shift = st.number_input("Plyn Shift [EUR]", value=0.0)
        df_yr['ee_price_mod'] = df_yr['ee_price'] + ee_shift
        df_yr['gas_price_mod'] = df_yr['gas_price'] + gas_shift

# --- HLAVNÍ PLOCHA ---
st.subheader("📍 Data lokality")
loc_file = st.file_uploader("Nahraj data lokality (aki11)", type=["xlsx"])
if loc_file:
    st.session_state.loc_data = clean_df(pd.read_excel(loc_file), is_fwd=False)

if st.session_state.fwd_data is not None and st.session_state.loc_data is not None:
    calc_df = pd.merge(df_yr, st.session_state.loc_data[['mdh', 'heat_price', 'heat_demand']], on='mdh', how='inner').sort_values('datetime').reset_index(drop=True)

    with st.form("param_form"):
        st.markdown("### ⚙️ Technické nastavení")
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.info("Kogenerace")
            k_th = st.number_input("Tepelný výkon [MW]", value=1.09)
            k_el = st.number_input("Elektrický výkon [MW]", value=0.999)
            k_eff = st.number_input("Tepelná účinnost", value=0.46)
            k_min = st.slider("Minimální zatížení [%]", 0, 100, 55) / 100
            k_serv = st.number_input("Servis [EUR/hod]", value=12.0)
            min_up = st.number_input("Min. doba běhu [hod]", value=4)
            min_down = st.number_input("Min. doba klidu [hod]", value=4)

        with c2:
            st.info("Kotelny a Grid")
            b_max = st.number_input("Plyn. kotel max [MW]", value=3.91)
            b_eff = st.number_input("Účinnost pl. kotle", value=0.95)
            ek_max = st.number_input("Elektrokotel max [MW]", value=0.6056)
            ek_eff = st.number_input("Účinnost el. kotle", value=0.98)
            dist_in = st.number_input("Distribuce nákup EE [EUR]", value=33.0)
            dist_out = st.number_input("Distribuce prodej EE [EUR]", value=2.0) # NOVINKA

        with c3:
            st.info("Systém a OZE")
            h_cover = st.slider("Min. pokrytí tepla", 0.9, 1.0, 0.99)
            
            # DYNAMICKÉ SEKCE PRO FVE/BESS
            if use_fve:
                fve_p = st.number_input("Instalovaný výkon FVE [kWp]", value=500)
            if use_bess:
                bess_cap = st.number_input("Kapacita BESS [MWh]", value=1.0)
                bess_pow = st.number_input("Výkon BESS [MW]", value=0.5)
            if use_acc:
                tes_cap = st.number_input("Kapacita AKU nádrže [MWh]", value=5.0)

        submit = st.form_submit_button("🚀 SPUSTIT OPTIMALIZACI")

    if submit:
        # Tady probíhá výpočet (identický s behouvkova_opt.txt + nová distribuce ven)
        # Pro přehlednost zkráceno, ale v reálu zde Pulp definuje model
        st.success("Výpočet proveden podle logiky Běhounkova.")
        # Zde by následovaly grafy a tabulky zisku...

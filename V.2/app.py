import streamlit as st
import pandas as pd
import pulp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="KGJ Strategy Expert", layout="wide")

# Inicializace stavu
if 'fwd_data' not in st.session_state: st.session_state.fwd_data = None
if 'loc_data' not in st.session_state: st.session_state.loc_data = None

st.title("🚀 KGJ Strategy & Dispatch Optimizer")

# --- 1. KROK: TRŽNÍ DATA (FWD) ---
with st.sidebar:
    st.header("1️⃣ Tržní ceny (FWD)")
    fwd_file = st.file_uploader("Nahraj FWD křivku (Excel)", type=["xlsx"])
    
    if fwd_file:
        df_raw = pd.read_excel(fwd_file)
        df_raw.columns = [str(c).strip() for c in df_raw.columns]
        date_col = df_raw.columns[0]
        df_raw[date_col] = pd.to_datetime(df_raw[date_col], dayfirst=True)
        
        # Roky pro filtr
        years = sorted(df_raw[date_col].dt.year.unique())
        sel_year = st.selectbox("Rok pro analýzu", years)
        df_year = df_raw[df_raw[date_col].dt.year == sel_year].copy()
        
        # VÝPOČET PRŮMĚRŮ Z NAHRANÉHO SOUBORU
        avg_ee_raw = float(df_year.iloc[:, 1].mean())
        avg_gas_raw = float(df_year.iloc[:, 2].mean())
        
        st.subheader("🛠️ Aktuální tržní podmínky")
        st.info(f"V souboru naměřeno (Base): \nEE: {avg_ee_raw:.2f} | Plyn: {avg_gas_raw:.2f}")
        
        # UŽIVATELSKÉ VSTUPY PRO NOVOU CENU
        ee_market_new = st.number_input("Nová tržní cena EE [EUR/MWh]", value=avg_ee_raw)
        gas_market_new = st.number_input("Nová tržní cena Plyn [EUR/MWh]", value=avg_gas_raw)
        
        # Výpočet shiftu
        ee_shift = ee_market_new - avg_ee_raw
        gas_shift = gas_market_new - avg_gas_raw
        
        # Příprava finálních dat
        df_fwd = df_year.copy()
        df_fwd.columns = ['datetime', 'ee_original', 'gas_original']
        df_fwd['ee_price'] = df_fwd['ee_original'] + ee_shift
        df_fwd['gas_price'] = df_fwd['gas_original'] + gas_shift
        df_fwd['mdh'] = df_fwd['datetime'].dt.strftime('%m-%d-%H')
        
        st.session_state.fwd_data = df_fwd
        st.write(f"Aplikovaný posun: EE {ee_shift:+.2f} | Plyn {gas_shift:+.2f}")

    st.divider()
    st.header("2️⃣ Aktivní technologie")
    use_kgj = st.checkbox("Kogenerace (KGJ)", value=True)
    use_boil = st.checkbox("Plynový kotel", value=True)
    use_ek = st.checkbox("Elektrokotel", value=True)
    use_ext_heat = st.checkbox("Povolit nákup tepla (Import)", value=False)

# --- 2. KROK: ZOBRAZENÍ TRŽNÍ KŘIVKY (SROVNÁNÍ) ---
if st.session_state.fwd_data is not None:
    with st.expander("📊 Srovnání: Původní vs. Upravená křivka", expanded=True):
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.1,
                           subplot_titles=("Elektřina [EUR/MWh]", "Zemní plyn [EUR/MWh]"))
        
        # EE graf
        fig.add_trace(go.Scatter(x=st.session_state.fwd_data['datetime'], y=st.session_state.fwd_data['ee_original'], 
                                 name="EE Původní", line=dict(color='rgba(0, 255, 0, 0.3)', dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=st.session_state.fwd_data['datetime'], y=st.session_state.fwd_data['ee_price'], 
                                 name="EE Upravená", line=dict(color='green')), row=1, col=1)
        
        # Plyn graf
        fig.add_trace(go.Scatter(x=st.session_state.fwd_data['datetime'], y=st.session_state.fwd_data['gas_original'], 
                                 name="Plyn Původní", line=dict(color='rgba(255, 0, 0, 0.3)', dash='dot')), row=2, col=1)
        fig.add_trace(go.Scatter(x=st.session_state.fwd_data['datetime'], y=st.session_state.fwd_data['gas_price'], 
                                 name="Plyn Upravená", line=dict(color='red')), row=2, col=1)
        
        fig.update_layout(height=600, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

# --- Zbytek parametrů (z tvého PRO kódu) ---
# ... zde by následovalo nahrání aki11 a samotný Pulp solver ...

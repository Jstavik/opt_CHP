import streamlit as st
import pandas as pd
import pulp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# --- KONFIGURACE STRÁNKY ---
st.set_page_config(page_title="KGJ Strategy Expert", layout="wide")

if 'fwd_data' not in st.session_state: st.session_state.fwd_data = None
if 'loc_data' not in st.session_state: st.session_state.loc_data = None

st.title("🎯 KGJ Strategy & Dispatch Optimizer")

# --- NEPRŮSTŘELNÁ FUNKCE PRO ČIŠTĚNÍ DAT ---
def clean_and_map(df, mapping):
    # Odstraníme prázdné řádky a sloupce na okrajích
    df = df.dropna(how='all').dropna(axis=1, how='all')
    # Ořežeme mezery z názvů sloupců
    df.columns = [str(c).strip() for c in df.columns]
    
    # Přejmenování podle mapy (pokud sloupec existuje)
    df = df.rename(columns=mapping)
    
    if 'datetime' in df.columns:
        # dayfirst=True pro tvůj formát 01.01.2026
        df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True, errors='coerce')
        df = df.dropna(subset=['datetime'])
        # Vytvoření MDH pro propojení (měsíc-den-hodina)
        df['mdh'] = df['datetime'].dt.strftime('%m-%d-%H')
        
        # Převedeme vše ostatní na čísla
        for col in df.columns:
            if col not in ['datetime', 'mdh']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# --- SIDEBAR: TRŽNÍ DATA ---
st.sidebar.header("📈 Tržní FWD Křivky")
fwd_file = st.sidebar.file_uploader("1. Nahraj 'FWD křivka EE_ZP.xlsx'", type=["xlsx"])

if fwd_file:
    raw_fwd = pd.read_excel(fwd_file)
    fwd_map = {
        'Datum': 'datetime', 
        'FWD (EUR/MWh)': 'ee_price', 
        'FWD plyn (EUR/MWh)': 'gas_price'
    }
    st.session_state.fwd_data = clean_and_map(raw_fwd, fwd_map)

if st.session_state.fwd_data is not None:
    fwd_df = st.session_state.fwd_data
    years = sorted(fwd_df['datetime'].dt.year.unique())
    sel_year = st.sidebar.selectbox("Vyber rok pro analýzu", years)
    
    df_yr = fwd_df[fwd_df['datetime'].dt.year == sel_year].copy().reset_index(drop=True)
    
    # Peak/Base statistiky
    df_yr['hour'] = df_yr['datetime'].dt.hour
    df_yr['weekday'] = df_yr['datetime'].dt.weekday
    is_peak = (df_yr['weekday'] < 5) & (df_yr['hour'] >= 8) & (df_yr['hour'] < 20)
    
    ee_base = df_yr['ee_price'].mean()
    ee_peak = df_yr[is_peak]['ee_price'].mean()
    gas_base = df_yr['gas_price'].mean()

    st.sidebar.markdown(f"**Původní {sel_year}:** EE Base: `{ee_base:.1f}`, Plyn: `{gas_base:.1f}`")

    ee_shift = st.sidebar.number_input("Posun EE [EUR/MWh]", value=0.0)
    gas_shift = st.sidebar.number_input("Posun Plyn [EUR/MWh]", value=0.0)
    
    df_yr['ee_price_mod'] = df_yr['ee_price'] + ee_shift
    df_yr['gas_price_mod'] = df_yr['gas_price'] + gas_shift

    # Graf trhu
    fig_market = make_subplots(specs=[[{"secondary_y": True}]])
    fig_market.add_trace(go.Scatter(x=df_yr['datetime'], y=df_yr['ee_price_mod'], name="EE Upravená", line=dict(color='#00ff00')), secondary_y=False)
    fig_market.add_trace(go.Scatter(x=df_yr['datetime'], y=df_yr['gas_price_mod'], name="Plyn Upravený", line=dict(color='red')), secondary_y=True)
    st.plotly_chart(fig_market, use_container_width=True)

# --- HLAVNÍ: LOKALITA ---
st.subheader("📍 Data lokality")
loc_file = st.file_uploader("2. Nahraj data lokality (aki11 / Behounkova)", type=["xlsx"])

if loc_file:
    raw_loc = pd.read_excel(loc_file)
    # ZDE JE OPRAVA: Mapování přesně podle tvého obrázku
    loc_map = {
        'Datum': 'datetime', 
        'Teplo (EUR/MWh)': 'heat_price', 
        'Behounkova DHV celkemMW': 'heat_demand'
    }
    st.session_state.loc_data = clean_and_map(raw_loc, loc_map)
    st.success("Lokalita nahrána.")

# --- SPOJENÍ A VÝPOČET ---
if st.session_state.fwd_data is not None and st.session_state.loc_data is not None:
    
    # Použijeme připravené MDH pro join
    # Musíme zajistit, aby v loc_data byly ty správné sloupce
    loc_cols = ['mdh', 'heat_price', 'heat_demand']
    
    # Kontrola, jestli se přejmenování povedlo
    missing = [c for c in loc_cols if c not in st.session_state.loc_data.columns]
    if missing:
        st.error(f"V souboru lokality chybí nebo se nepodařilo přejmenovat sloupce: {missing}")
        st.write("Dostupné sloupce:", list(st.session_state.loc_data.columns))
    else:
        calc_df = pd.merge(df_yr, st.session_state.loc_data[loc_cols], on='mdh', how='inner')
        calc_df = calc_df.sort_values('datetime').reset_index(drop=True)

        with st.expander("🛠️ Technické parametry", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                kgj_th = st.number_input("KGJ Tepelný výkon [MW]", value=1.09)
                kgj_el = st.number_input("KGJ Elektrický výkon [MW]", value=0.999)
                kgj_eff = st.number_input("KGJ Tepelná účinnost", value=0.46)
            with c2:
                boil_eff = st.number_input("Účinnost pl. kotle", value=0.95)
                dist_c = st.number_input("Distribuce EE [EUR/MWh]", value=33.0)

        if st.button("🚀 SPUSTIT OPTIMALIZACI"):
            # Solver a zbytek tvé logiky...
            st.info("Počítám optimální dispatch...")
            # (Zde pokračuje PuLP model, který už máš)

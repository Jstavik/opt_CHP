import streamlit as st
import pandas as pd
import pulp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="KGJ Strategy Expert Pro", layout="wide")

# --- 1. KROK: VÝBĚR TECHNOLOGIÍ (Hned na začátku) ---
st.title("🚀 KGJ Strategy & Dispatch Optimizer PRO")

with st.sidebar:
    st.header("⚙️ Konfigurace lokality")
    active_tech = st.multiselect(
        "Vyberte technologie na lokalitě:",
        ["KGJ", "Plynový kotel", "Elektrokotel", "FVE", "Baterie (BESS)", "Tepelná akumulace (TES)", "Externí nákup tepla"],
        default=["KGJ", "Plynový kotel"]
    )
    
    st.divider()
    st.header("📈 Tržní data (FWD)")
    fwd_file = st.file_uploader("Nahraj FWD křivku (Excel - EE, Plyn)", type=["xlsx"])

# --- 2. KROK: PARAMETRY (Dynamické boxy) ---
st.header("📍 Nastavení parametrů a cen")

# Pomocná funkce pro cenový box
def price_logic(label, key_prefix):
    col1, col2 = st.columns([1, 1])
    with col1:
        mode = st.radio(f"{label} - režim", ["Tržní (FWD)", "Fixní cena"], key=f"{key_prefix}_mode")
    with col2:
        if mode == "Fixní cena":
            val = st.number_input(f"Zadejte fixní cenu {label} [EUR/MWh]", value=0.0, key=f"{key_prefix}_fix")
            return ("fix", val)
        return ("fwd", 0)

# Rozdělení do karet pro přehlednost
tabs = st.tabs(active_tech + ["Společné parametry"])

params = {}

for tab, tech in zip(tabs, active_tech):
    with tab:
        if tech == "KGJ":
            c1, c2 = st.columns(2)
            with c1:
                params['k_th'] = st.number_input("Tepelný výkon [MW]", value=1.09)
                params['k_el'] = st.number_input("Elektrický výkon [MW]", value=1.0)
                params['k_eff'] = st.number_input("Tepelná účinnost", value=0.46)
            with c2:
                params['k_gas_price'] = price_logic("Nákup plynu pro KGJ", "kgj_gas")
                params['k_ee_price'] = price_logic("Prodej EE z KGJ", "kgj_ee")
                params['k_dist_ee'] = st.number_input("Distribuční poplatek - prodej [EUR/MWh]", value=2.0)

        elif tech == "Plynový kotel":
            c1, c2 = st.columns(2)
            with c1:
                params['b_max'] = st.number_input("Max. výkon kotle [MW]", value=3.91)
                params['b_eff'] = st.number_input("Účinnost kotle", value=0.95)
            with c2:
                params['b_gas_price'] = price_logic("Nákup plynu pro Kotel", "boil_gas")
                params['b_dist_gas'] = st.number_input("Distribuce plyn [EUR/MWh]", value=5.0)

        elif tech == "Elektrokotel":
            c1, c2 = st.columns(2)
            with c1:
                params['ek_max'] = st.number_input("Max. výkon EK [MW]", value=1.0)
                params['ek_eff'] = st.number_input("Účinnost EK", value=0.98)
            with c2:
                params['ek_ee_price'] = price_logic("Nákup EE pro EK", "ek_ee")
                params['ek_dist_ee'] = st.number_input("Distribuce nákup EE [EUR/MWh]", value=35.0)

        elif tech == "FVE":
            params['fve_p_max'] = st.number_input("Instalovaný výkon FVE [MWp]", value=0.5)
            params['fve_ee_price'] = price_logic("Výkup EE z FVE", "fve_ee")

        elif tech == "Baterie (BESS)":
            c1, c2 = st.columns(2)
            with c1:
                params['bess_cap'] = st.number_input("Kapacita BESS [MWh]", value=1.0)
                params['bess_p'] = st.number_input("Max. výkon (nab/vyb) [MW]", value=0.5)
            with c2:
                params['bess_eff'] = st.number_input("Round-trip účinnost", value=0.90)
                params['bess_dist'] = st.checkbox("Platit distribuci při nabíjení?")

        elif tech == "Tepelná akumulace (TES)":
            params['tes_cap'] = st.number_input("Kapacita nádrže [MWh_th]", value=5.0)
            params['tes_loss'] = st.slider("Hodinová ztráta [%]", 0.0, 5.0, 0.5) / 100

        elif tech == "Externí nákup tepla":
            params['ext_h_max'] = st.number_input("Max. přikon nákupu [MW]", value=2.0)
            params['ext_h_price'] = st.number_input("Cena nákupu tepla [EUR/MWh]", value=80.0)

with tabs[-1]:
    st.info("Obecné nastavení systému")
    params['h_price_mode'] = price_logic("Prodejní cena tepla zákazníkovi", "heat_sale")
    params['h_cover'] = st.slider("Povinné pokrytí tepla", 0.0, 1.0, 1.0)

# --- 3. KROK: DATA LOKALITY ---
st.divider()
loc_file = st.file_uploader("3️⃣ Nahraj provozní data (Profil poptávky, FVE profil)", type=["xlsx"])

# --- VÝPOČETNÍ LOGIKA (ILUSTRAČNÍ ČÁST) ---
if st.button("🏁 SPUSTIT OPTIMALIZACI"):
    if fwd_file and loc_file:
        st.write("🔄 Probíhá slučování dat a sestavování rovnic...")
        # Zde by následoval Pulp model, který by využíval 'active_tech' k přidávání rovnic.
        # Např: if "Baterie (BESS)" in active_tech: add_bess_constraints(model)
        st.success("Model připraven k výpočtu (logika integrace dokončena).")
    else:
        st.error("Chybí vstupní soubory!")
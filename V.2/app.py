import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="KGJ Optimizer PRO", layout="wide")

st.title("🚀 KGJ Strategy & Dispatch Optimizer PRO")

# --- SIDEBAR: VOLBA TECHNOLOGIÍ ---
with st.sidebar:
    st.header("⚙️ Konfigurace lokality")
    active_tech = st.multiselect(
        "Aktivní technologie:",
        ["KGJ", "Plynový kotel", "Elektrokotel", "FVE", "Baterie (BESS)", "Tepelná akumulace (TES)", "Externí nákup tepla"],
        default=["KGJ", "Plynový kotel"]
    )
    
    st.divider()
    st.header("📈 Nahrání tržních dat")
    fwd_file = st.file_uploader("Fwd křivka (Excel)", type=["xlsx"])

# --- FUNKCE PRO CENOVOU LOGIKU ---
def price_input(label, key):
    col1, col2 = st.columns(2)
    with col1:
        mode = st.radio(f"{label}", ["Tržní (FWD)", "Fixní"], key=f"mode_{key}", horizontal=True)
    with col2:
        if mode == "Fixní":
            return st.number_input("Cena [EUR/MWh]", value=0.0, key=f"val_{key}")
        return "fwd"

# --- HLAVNÍ FORMULÁŘ ---
params = {}

# Dynamické záložky podle výběru
if active_tech:
    tabs = st.tabs(active_tech + ["Společné parametry"])
    
    for i, tech in enumerate(active_tech):
        with tabs[i]:
            if tech == "KGJ":
                c1, c2 = st.columns(2)
                with c1:
                    params['k_th'] = st.number_input("Tepelný výkon [MW]", value=1.09)
                    params['k_el'] = st.number_input("Elektrický výkon [MW]", value=1.0)
                    params['k_eff'] = st.number_input("Tepelná účinnost", value=0.46)
                    params['k_serv'] = st.number_input("Servis [EUR/hod]", value=12.0)
                with c2:
                    params['k_gas_price'] = price_input("Nákup plynu pro KGJ", "kgj_gas")
                    params['k_gas_dist'] = st.number_input("Distribuce plyn nákup [EUR/MWh]", value=5.0, key="kgj_g_dist")
                    params['k_ee_price'] = price_input("Prodej EE z KGJ", "kgj_ee")
                    params['k_ee_dist'] = st.number_input("Distribuce EE prodej [EUR/MWh]", value=2.0)

            elif tech == "Plynový kotel":
                c1, c2 = st.columns(2)
                with c1:
                    params['b_max'] = st.number_input("Max. výkon kotle [MW]", value=3.91)
                    params['b_eff'] = st.number_input("Účinnost", value=0.95)
                with c2:
                    params['b_gas_price'] = price_input("Nákup plynu pro Kotel", "b_gas")
                    params['b_gas_dist'] = st.number_input("Distribuce plyn nákup [EUR/MWh]", value=5.0, key="b_g_dist")

            elif tech == "Elektrokotel":
                c1, c2 = st.columns(2)
                with c1:
                    params['ek_max'] = st.number_input("Max. výkon EK [MW]", value=1.0)
                    params['ek_eff'] = st.number_input("Účinnost EK", value=0.98)
                    params['ek_allow_kgj'] = st.checkbox("Umožnit přímé napájení z KGJ (bez distribuce)", value=True)
                with c2:
                    params['ek_ee_price'] = price_input("Nákup EE z gridu", "ek_ee")
                    params['ek_ee_dist'] = st.number_input("Distribuce EE nákup [EUR/MWh]", value=35.0)

            elif tech == "FVE":
                c1, c2 = st.columns(2)
                with c1:
                    params['fve_p_max'] = st.number_input("Instalovaný výkon [MWp]", value=0.5)
                with c2:
                    params['fve_ee_price'] = price_input("Výkup EE z FVE", "fve_ee")

            elif tech == "Baterie (BESS)":
                c1, c2 = st.columns(2)
                with c1:
                    params['bess_cap'] = st.number_input("Kapacita [MWh]", value=1.0)
                    params['bess_p'] = st.number_input("Max. výkon [MW]", value=0.5)
                with c2:
                    dist_on = st.checkbox("Platit distribuci u BESS?", key="bess_dist_on")
                    if dist_on:
                        params['bess_dist_val'] = st.number_input("Distribuce nab/vyb [EUR/MWh]", value=15.0)
                    params['bess_eff'] = st.number_input("Účinnost (cycle)", value=0.90)

            elif tech == "Tepelná akumulace (TES)":
                c1, c2 = st.columns(2)
                with c1:
                    params['tes_cap'] = st.number_input("Kapacita nádrže [MWh_th]", value=5.0)
                with c2:
                    params['tes_loss'] = st.slider("Hodinová ztráta [%]", 0.0, 5.0, 0.5) / 100
                    params['tes_init'] = st.slider("Počáteční stav [%]", 0, 100, 20) / 100

            elif tech == "Externí nákup tepla":
                c1, c2 = st.columns(2)
                with c1:
                    params['ext_h_max'] = st.number_input("Max. příkon nákupu [MW]", value=2.0)
                with c2:
                    params['ext_h_price'] = st.number_input("Smluvní cena tepla [EUR/MWh]", value=85.0)

    # Poslední tab - Společné
    with tabs[-1]:
        st.subheader("Ostatní parametry systému")
        params['h_price_mode'] = price_input("Prodejní cena tepla zákazníkovi", "h_sale")
        params['h_cover'] = st.slider("Povinné pokrytí tepla (0.95 = 95%)", 0.0, 1.0, 1.0)
        params['co2_price'] = st.number_input("Cena CO2 povolenky [EUR/t]", value=0.0) # Bonus pro komplexnost

# --- 3. KROK: EDITACE DAT ---
st.divider()
st.header("📊 Úprava nahrání dat a profilů")

if fwd_file:
    df_raw = pd.read_excel(fwd_file)
    st.subheader("Editovatelná tržní a profilová data")
    st.info("Zde můžeš přímo v tabulce změnit hodnoty (např. ručně upravit profil FVE nebo cenu EE).")
    # Data editor umožní uživateli měnit hodnoty přímo v prohlížeči
    df_edited = st.data_editor(df_raw, use_container_width=True)
    st.session_state.edited_data = df_edited
else:
    st.info("Nahraj soubor pro zobrazení a editaci tabulky dat.")

# Tlačítko pro spuštění už jen vizuálně připravíme
if st.button("🚀 PŘIPRAVIT MODEL K VÝPOČTU"):
    st.write("Parametry byly uloženy. Jsem připraven sestavit matici rovnic.")

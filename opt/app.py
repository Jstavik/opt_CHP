import streamlit as st
import pandas as pd
import pulp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# ======================================================
# 1) NASTAVENÍ STRÁNKY A SIDEBARU (PARAMETRY)
# ======================================================
st.set_page_config(page_title="KGJ Master Dispatcher", layout="wide")
st.title("⚡ KGJ Integrated Dispatcher & Optimizer")

st.sidebar.header("⚙️ Konfigurace Systému")

# Rozcestník technologií - co v lokalitě je?
with st.sidebar.expander("🛠️ Aktivní technologie", expanded=True):
    has_aku = st.checkbox("Akumulace tepla (nádrž)", value=True)
    has_bess = st.checkbox("Bateriové úložiště (BESS)", value=False)
    has_fve = st.checkbox("Fotovoltaika (FVE)", value=False)
    has_ext_heat = st.checkbox("Externí dodávka tepla (CZT)", value=False)

# Parametry KGJ
with st.sidebar.expander("🔥 Kogenerace (KGJ)", expanded=True):
    p_th = st.number_input("Tepelný výkon [MW]", value=1.09)
    p_el = st.number_input("Elektrický výkon [MW]", value=0.999)
    eff_th = st.number_input("Tepelná účinnost", value=0.46)
    service_cost = st.number_input("Servis [EUR/hod]", value=12.0)
    startup_cost = st.number_input("Náklad na start [EUR]", value=50.0)
    min_load = st.slider("Minimální výkon [%]", 30, 100, 55) / 100.0
    min_up = st.number_input("Min. doba běhu [h]", value=4)
    min_down = st.number_input("Min. doba stání [h]", value=4)

# Parametry Akumulace
if has_aku:
    with st.sidebar.expander("💧 Akumulace Tepla"):
        aku_cap = st.number_input("Kapacita nádrže [MWh]", value=10.0)
        aku_max_p = st.number_input("Max. nabíjecí výkon [MW]", value=2.0)
        aku_loss = st.slider("Hodinová ztráta [%]", 0.0, 1.0, 0.2) / 100.0

# Parametry Baterie
if has_bess:
    with st.sidebar.expander("🔋 Baterie (BESS)"):
        bess_cap = st.number_input("Kapacita BESS [MWh]", value=1.0)
        bess_p = st.number_input("Výkon BESS [MW]", value=0.5)

# Parametry FVE
if has_fve:
    with st.sidebar.expander("☀️ Fotovoltaika"):
        fve_inst_p = st.number_input("Instalovaný výkon FVE [kWp]", value=500.0)

# Ostatní zdroje
with st.sidebar.expander("🏥 Ostatní zdroje"):
    boiler_p = st.number_input("Plynový kotel max [MW]", value=3.91)
    eboiler_p = st.number_input("Elektrokotel max [MW]", value=0.60)
    if has_ext_heat:
        ext_heat_price = st.number_input("Cena ext. tepla [EUR/MWh]", value=60.0)

# Ekonomika a distribuce
with st.sidebar.expander("💶 Ekonomika & Distribuce"):
    dist_buy = st.number_input("Distribuce nákup [EUR/MWh]", value=33.0)
    dist_sell = st.number_input("Distribuce dodávka [EUR/MWh]", value=15.0)
    base_price_target = st.number_input("Cílová Base EE (0 = beze změny)", value=0.0)

# ======================================================
# 2) NAHRÁNÍ DAT
# ======================================================
uploaded_file = st.file_uploader("📂 Nahraj Excel s křivkami (datetime, ee_price, gas_price, heat_price, heat_demand, fve_norm)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = [c.lower() for c in df.columns] # sjednocení na malá písmena
    
    # Úprava křivky EE podle Base
    if base_price_target > 0:
        old_base = df['ee_price'].mean()
        df['ee_price'] = df['ee_price'] * (base_price_target / old_base)
    
    # Úprava FVE podle instalovaného výkonu
    if has_fve and 'fve_norm' in df.columns:
        df['fve_prod'] = df['fve_norm'] * (fve_inst_p / 1000.0)
    else:
        df['fve_prod'] = 0.0

    T = len(df)
    
    if st.button("🚀 SPUSTIT OPTIMALIZACI"):
        with st.spinner("Počítám optimální dispatch..."):
            
            # --- MODEL ---
            model = pulp.LpProblem("Master_Dispatch", pulp.LpMaximize)
            
            # Proměnné
            q_kgj = pulp.LpVariable.dicts("q_kgj", range(T), 0, p_th)
            q_boiler = pulp.LpVariable.dicts("q_boiler", range(T), 0, boiler_p)
            q_eboiler = pulp.LpVariable.dicts("q_eboiler", range(T), 0, eboiler_p)
            q_waste = pulp.LpVariable.dicts("q_waste", range(T), 0)
            kgj_on = pulp.LpVariable.dicts("kgj_on", range(T), 0, 1, cat="Binary")
            kgj_start = pulp.LpVariable.dicts("kgj_start", range(T), 0, 1, cat="Binary")
            
            # Teplo - Aku / Ext
            if has_aku:
                soc_th = pulp.LpVariable.dicts("soc_th", range(T), 0, aku_cap)
                q_aku_ch = pulp.LpVariable.dicts("q_aku_ch", range(T), 0, aku_max_p)
                q_aku_ds = pulp.LpVariable.dicts("q_aku_ds", range(T), 0, aku_max_p)
            
            q_ext = pulp.LpVariable.dicts("q_ext", range(T), 0) if has_ext_heat else None

            # Elektro bilance
            ee_sold = pulp.LpVariable.dicts("ee_sold", range(T), 0)
            ee_to_eb = pulp.LpVariable.dicts("ee_to_eb", range(T), 0) # internal KGJ -> EBoiler
            ee_from_grid = pulp.LpVariable.dicts("ee_from_grid", range(T), 0) # grid -> EBoiler

            # --- OMEZENÍ ---
            for t in range(T):
                h_req = df.loc[t, 'heat_demand'] * 0.99
                
                # Bilance tepla
                sources = q_kgj[t] - q_waste[t] + q_boiler[t] + q_eboiler[t]
                if has_ext_heat: sources += q_ext[t]
                
                if has_aku:
                    model += sources + q_aku_ds[t] == h_req + q_aku_ch[t]
                    if t == 0: model += soc_th[t] == (aku_cap*0.5) + q_aku_ch[t] - q_aku_ds[t]
                    else: model += soc_th[t] == soc_th[t-1]*(1-aku_loss) + q_aku_ch[t] - q_aku_ds[t]
                else:
                    model += sources == h_req
                
                # KGJ Technické
                model += q_kgj[t] <= p_th * kgj_on[t]
                model += q_kgj[t] >= p_th * min_load * kgj_on[t]
                
                # Elektro bilance (KGJ el prod = Sold + Internal to EB)
                ee_prod = q_kgj[t] * (p_el/p_th)
                model += ee_sold[t] + ee_to_eb[t] == ee_prod
                model += q_eboiler[t] == 0.98 * (ee_to_eb[t] + ee_from_grid[t])

                # Logika startu
                if t > 0: model += kgj_on[t] - kgj_on[t-1] <= kgj_start[t]
                else: model += kgj_on[t] <= kgj_start[t]

            # Omezení chodu (Min UP/DOWN) - jen pokud není T příliš velké pro rychlost
            if T < 1000: # Pro roční data 8760 doporučuji vypnout pro rychlost, nebo zkrátit
                for t in range(T - int(min_up)):
                    model += pulp.lpSum(kgj_on[t+i] for i in range(int(min_up))) >= int(min_up) * kgj_start[t]

            # --- CÍLOVÁ FUNKCE ---
            obj = []
            for t in range(T):
                # Příjmy (Teplo + EE)
                rev = (df.loc[t, 'heat_price'] * df.loc[t, 'heat_demand'] * 0.99) + \
                      (df.loc[t, 'ee_price'] - dist_sell) * ee_sold[t]
                
                # Náklady
                costs = (q_kgj[t]*(1/eff_th)*df.loc[t, 'gas_price']) + \
                        (q_boiler[t]/0.95*df.loc[t, 'gas_price']) + \
                        (ee_from_grid[t] * (df.loc[t, 'ee_price'] + dist_buy)) + \
                        (kgj_on[t] * service_cost) + (kgj_start[t] * startup_cost)
                
                if has_ext_heat: costs += q_ext[t] * ext_heat_price
                
                obj.append(rev - costs)
            
            model += pulp.lpSum(obj)
            
            # Řešení s limitem (aby to netrvalo 40 min)
            model.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=120, gapRel=0.01))
            
            # --- VÝSLEDKY ---
            st.success(f"Optimalizace hotova! Celkový zisk: {pulp.value(model.objective):,.2f} EUR")
            
            # Export dat
            res = df.copy()
            res['KGJ_Teplo'] = [q_kgj[t].value() for t in range(T)]
            res['Boiler_Teplo'] = [q_boiler[t].value() for t in range(T)]
            res['EBoiler_Teplo'] = [q_eboiler[t].value() for t in range(T)]
            if has_aku: res['SOC_Teplo'] = [soc_th[t].value() for t in range(T)]
            
            # GRAF
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=res['datetime'], y=res['heat_demand'], name="Poptávka", line=dict(color='black')))
            fig.add_trace(go.Bar(x=res['datetime'], y=res['KGJ_Teplo'], name="KGJ Teplo"))
            fig.add_trace(go.Bar(x=res['datetime'], y=res['Boiler_Teplo'], name="Kotel Teplo"))
            fig.add_trace(go.Scatter(x=res['datetime'], y=res['ee_price'], name="Cena EE", yaxis="y2", opacity=0.3))
            st.plotly_chart(fig, use_container_width=True)

            # Download link
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                res.to_excel(writer, index=False)
            st.download_button("📥 Stáhnout kompletní výsledky", output.getvalue(), "vysledky_optimalizace.xlsx")

else:
    st.info("Nahraj prosím vstupní Excel pro spuštění modelu.")
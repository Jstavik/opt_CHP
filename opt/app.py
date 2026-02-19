import streamlit as st
import pandas as pd
import pulp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# ======================================================
# 1) KONFIGURACE STRÁNKY
# ======================================================
st.set_page_config(page_title="KGJ Strategy Expert", layout="wide")

if 'fwd_ee' not in st.session_state: st.session_state.fwd_ee = None
if 'loc_data' not in st.session_state: st.session_state.loc_data = None

st.title("🎯 Strategický Optimalizátor KGJ")

# ======================================================
# 2) SIDEBAR: TRŽNÍ CENY (EE & PLYN)
# ======================================================
st.sidebar.header("📈 Tržní FWD Křivky")

fwd_file = st.sidebar.file_uploader("1. Nahraj Master EE FWD", type=["xlsx"])
if fwd_file:
    data = pd.read_excel(fwd_file)
    # OPRAVA: Sjednocení názvů sloupců (odstranění mezer a malá písmena)
    data.columns = [str(c).strip().lower() for c in data.columns]
    
    if 'datetime' in data.columns:
        data['datetime'] = pd.to_datetime(data['datetime'])
        st.session_state.fwd_ee = data
        st.sidebar.success("Master EE nahrán v pořádku.")
    else:
        st.sidebar.error("Chyba: Soubor musí obsahovat sloupec 'datetime'!")

if st.session_state.fwd_ee is not None:
    years = sorted(st.session_state.fwd_ee['datetime'].dt.year.unique())
    sel_year = st.sidebar.selectbox("Vyber rok pro analýzu", years)
    
    df_yr = st.session_state.fwd_ee[st.session_state.fwd_ee['datetime'].dt.year == sel_year].copy().reset_index(drop=True)
    
    # Ošetření sloupce ee_price
    if 'ee_price' not in df_yr.columns:
        st.sidebar.error("Chyba: V souboru chybí sloupec 'ee_price'!")
    else:
        raw_ee_base = df_yr['ee_price'].mean()
        st.sidebar.write(f"Původní Base {sel_year}: **{raw_ee_base:.2f} EUR**")
        ee_shift = st.sidebar.number_input("Posun EE Base [EUR/MWh]", value=0.0)
        df_yr['ee_price'] += ee_shift

    # Plyn po měsících
    st.sidebar.subheader(f"Ceny plynu pro rok {sel_year}")
    gas_months = []
    default_gas = [35.0, 34.5, 34.0, 31.0, 30.0, 30.0, 30.0, 30.0, 30.0, 32.0, 34.0, 35.0]
    for m in range(1, 13):
        g_val = st.sidebar.number_input(f"Měsíc {m}", value=default_gas[m-1], key=f"g_{m}")
        gas_months.append(g_val)
    df_yr['gas_price'] = df_yr['datetime'].dt.month.map(lambda x: gas_months[x-1])

# ======================================================
# 3) HLAVNÍ PANEL: DATA LOKALITY A TECHNOLOGIE
# ======================================================
st.subheader("📍 Nastavení lokality")
loc_file = st.file_uploader("2. Nahraj profil poptávky (aki11 vzor: heat_demand, heat_price)", type=["xlsx"])

if loc_file:
    st.session_state.loc_data = pd.read_excel(loc_file)
    st.session_state.loc_data['datetime'] = pd.to_datetime(st.session_state.loc_data['datetime'])

if st.session_state.fwd_ee is not None and st.session_state.loc_data is not None:
    # Join trhu a lokality (přesazení profilu na vybraný rok)
    # Pro zjednodušení děláme join na měsíc-den-hodinu (přesvátkování)
    df_yr['mdh'] = df_yr['datetime'].dt.strftime('%m-%d-%H')
    st.session_state.loc_data['mdh'] = st.session_state.loc_data['datetime'].dt.strftime('%m-%d-%H')
    
    calc_df = pd.merge(df_yr, st.session_state.loc_data[['mdh', 'heat_demand', 'heat_price']], on='mdh', how='inner')
    calc_df = calc_df.sort_values('datetime').reset_index(drop=True)
    
    st.info(f"Data připravena. Počet hodin k výpočtu: {len(calc_df)}")

    # UI pro technické parametry (z tvého txt souboru)
    with st.expander("🛠️ Technické parametry systému", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Kogenerace (KGJ)**")
            kgj_heat_output = st.number_input("Tepelný výkon [MW]", value=1.09)
            kgj_el_output = st.number_input("Elektrický výkon [MW]", value=0.999)
            kgj_heat_eff = st.number_input("Tepelná účinnost", value=0.46)
            kgj_service = st.number_input("Servis [EUR/hod]", value=12.0)
            kgj_min_load = st.slider("Minimální zatížení [%]", 30, 100, 55) / 100.0
        with c2:
            st.markdown("**Kotle**")
            boiler_max_heat = st.number_input("Plynový kotel max [MW]", value=3.91)
            boiler_eff = st.number_input("Účinnost pl. kotle", value=0.95)
            eboiler_max_heat = st.number_input("Elektrokotel max [MW]", value=0.605)
            eboiler_eff = st.number_input("Účinnost el. kotle", value=0.98)
        with c3:
            st.markdown("**Omezení a Distribuce**")
            min_up = st.number_input("Min. doba běhu [hod]", value=4)
            min_down = st.number_input("Min. doba stání [hod]", value=4)
            ee_dist_cost = st.number_input("Distribuce nákup [EUR/MWh]", value=33.0)
            heat_min_cover = st.slider("Min. krytí tepla [%]", 90, 100, 99) / 100.0

    # ======================================================
    # 4) OPTIMALIZAČNÍ ENGINE (PuLP)
    # ======================================================
    if st.button("🚀 SPUSTIT VÝPOČET"):
        T = len(calc_df)
        model = pulp.LpProblem("KGJ_Integrated_Dispatch", pulp.LpMaximize)

        # Pomocné výpočty koeficientů
        kgj_gas_per_heat = (kgj_heat_output / kgj_heat_eff) / kgj_heat_output
        kgj_el_per_heat = kgj_el_output / kgj_heat_output

        # Proměnné
        q_kgj = pulp.LpVariable.dicts("q_KGJ", range(T), 0, kgj_heat_output)
        q_boiler = pulp.LpVariable.dicts("q_boiler", range(T), 0, boiler_max_heat)
        q_eboiler = pulp.LpVariable.dicts("q_eboiler", range(T), 0, eboiler_max_heat)
        
        ee_sold_spot = pulp.LpVariable.dicts("ee_sold_spot", range(T), 0)
        ee_to_eb_grid = pulp.LpVariable.dicts("ee_to_eb_grid", range(T), 0)
        ee_to_eb_int = pulp.LpVariable.dicts("ee_to_eb_int", range(T), 0)
        
        kgj_on = pulp.LpVariable.dicts("KGJ_on", range(T), 0, 1, cat="Binary")
        kgj_start = pulp.LpVariable.dicts("KGJ_start", range(T), 0, 1, cat="Binary")
        kgj_stop = pulp.LpVariable.dicts("KGJ_stop", range(T), 0, 1, cat="Binary")

        # Podmínky
        for t in range(T):
            demand = calc_df.loc[t, "heat_demand"]
            h_req = heat_min_cover * demand

            # Bilance tepla
            model += q_kgj[t] + q_boiler[t] + q_eboiler[t] >= h_req
            
            # KGJ limity
            model += q_kgj[t] <= kgj_heat_output * kgj_on[t]
            model += q_kgj[t] >= kgj_min_load * kgj_heat_output * kgj_on[t]
            
            # Elektro bilance
            ee_prod = q_kgj[t] * kgj_el_per_heat
            model += ee_sold_spot[t] + ee_to_eb_int[t] == ee_prod
            model += q_eboiler[t] == eboiler_eff * (ee_to_eb_int[t] + ee_to_eb_grid[t])
            
            # Logika start/stop
            if t > 0:
                model += kgj_on[t] - kgj_on[t-1] == kgj_start[t] - kgj_stop[t]
            else:
                model += kgj_on[t] == kgj_start[t] - kgj_stop[t]

        # Min UP/DOWN (pouze pokud je T únosné pro rychlost)
        if T < 2000:
            for t in range(T - int(min_up)):
                model += pulp.lpSum(kgj_on[t+i] for i in range(int(min_up))) >= int(min_up) * kgj_start[t]
            for t in range(T - int(min_down)):
                model += pulp.lpSum(1 - kgj_on[t+i] for i in range(int(min_down))) >= int(min_down) * kgj_stop[t]

        # Cílová funkce (Profit)
        profit = []
        for t in range(T):
            ee = calc_df.loc[t, "ee_price"]
            gas = calc_df.loc[t, "gas_price"]
            hp = calc_df.loc[t, "heat_price"]
            h_dem = calc_df.loc[t, "heat_demand"]
            
            revenue = (hp * heat_min_cover * h_dem) + (ee * ee_sold_spot[t])
            costs = (gas * (q_kgj[t] * kgj_gas_per_heat + q_boiler[t] / boiler_eff)) + \
                    ((ee + ee_dist_cost) * ee_to_eb_grid[t]) + \
                    (kgj_service * kgj_on[t])
            profit.append(revenue - costs)
        
        model += pulp.lpSum(profit)

        # Solve s časovým limitem pro web
        model.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=60))

        # ======================================================
        # 5) VÝSLEDKY A DASHBOARD
        # ======================================================
        st.success(f"Optimalizace dokončena! Celkový zisk: {pulp.value(model.objective):,.2f} EUR")

        # Příprava výsledkové tabulky (včetně tvých triggerů a bypassu)
        res = calc_df.copy()
        res['q_kgj'] = [q_kgj[t].value() for t in range(T)]
        res['q_boiler'] = [q_boiler[t].value() for t in range(T)]
        res['q_eboiler'] = [q_eboiler[t].value() for t in range(T)]
        res['kgj_on'] = [kgj_on[t].value() for t in range(T)]
        
        # Bypass logika
        res['bypass'] = res.apply(lambda r: max(r['q_kgj'] - heat_min_cover * r['heat_demand'], 0) if r['heat_demand'] > 0 else 0, axis=1)
        
        # Grafy
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=res['datetime'], y=res['q_kgj'], name="KGJ Teplo", marker_color='orange'))
        fig.add_trace(go.Bar(x=res['datetime'], y=res['q_boiler'], name="Kotel Teplo", marker_color='red'))
        fig.add_trace(go.Scatter(x=res['datetime'], y=res['heat_demand'], name="Poptávka", line=dict(color='black', width=2)))
        fig.add_trace(go.Scatter(x=res['datetime'], y=res['ee_price'], name="EE Cena", yaxis="y2", line=dict(color='blue', dash='dot'), opacity=0.3))
        
        st.plotly_chart(fig, use_container_width=True)

        # Download
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            res.to_excel(writer, index=False)
        st.download_button("📥 Stáhnout Excel výsledky", output.getvalue(), f"vysledek_{sel_year}.xlsx")

else:
    st.warning("Nahraj oba soubory (Master EE vlevo a Profil poptávky zde) pro spuštění.")


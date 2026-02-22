import streamlit as st
import pandas as pd
import pulp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="KGJ Strategy Expert PRO", layout="wide")

# ────────────────────────────────────────────────
# Inicializace
# ────────────────────────────────────────────────
if 'fwd_data' not in st.session_state:
    st.session_state.fwd_data = None
    st.session_state.avg_ee_raw = 100.0   # fallback
    st.session_state.avg_gas_raw = 50.0   # fallback

st.title("🚀 KGJ Strategy & Dispatch Optimizer PRO")

# ────────────────────────────────────────────────
# SIDEBAR – jen zakliknutí + FWD
# ────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Technologie na lokalitě")
    use_kgj = st.checkbox("Kogenerace (KGJ)", value=True)
    use_boil = st.checkbox("Plynový kotel", value=True)
    use_ek = st.checkbox("Elektrokotel", value=True)
    use_tes = st.checkbox("Nádrž (TES)", value=True)
    use_bess = st.checkbox("Baterie (BESS)", value=True)
    use_fve = st.checkbox("Fotovoltaika (FVE)", value=True)
    use_ext_heat = st.checkbox("Nákup tepla (Import)", value=True)

    st.divider()
    st.header("📈 Tržní ceny (FWD)")
    fwd_file = st.file_uploader("Nahraj FWD křivku (Excel)", type=["xlsx"])

    if fwd_file is not None:
        try:
            df_raw = pd.read_excel(fwd_file)
            df_raw.columns = [str(c).strip() for c in df_raw.columns]
            date_col = df_raw.columns[0]
            df_raw[date_col] = pd.to_datetime(df_raw[date_col], dayfirst=True)

            years = sorted(df_raw[date_col].dt.year.unique())
            sel_year = st.selectbox("Rok pro analýzu", years)
            df_year = df_raw[df_raw[date_col].dt.year == sel_year].copy()

            avg_ee = float(df_year.iloc[:, 1].mean())
            avg_gas = float(df_year.iloc[:, 2].mean())

            st.session_state.avg_ee_raw = avg_ee
            st.session_state.avg_gas_raw = avg_gas

            ee_new = st.number_input("Cílová cena EE [€/MWh]", value=avg_ee)
            gas_new = st.number_input("Cílová cena Plyn [€/MWh]", value=avg_gas)

            df_fwd = df_year.copy()
            df_fwd.columns = ['datetime', 'ee_original', 'gas_original']
            df_fwd['ee_price'] = df_fwd['ee_original'] + (ee_new - avg_ee)
            df_fwd['gas_price'] = df_fwd['gas_original'] + (gas_new - avg_gas)
            st.session_state.fwd_data = df_fwd

            st.success("FWD načteno")
        except Exception as e:
            st.error(f"Chyba při načítání FWD: {e}")

# ────────────────────────────────────────────────
# Parametry – dynamicky
# ────────────────────────────────────────────────
p = {
    'dist_ee_buy': 33.0,
    'dist_ee_sell': 2.0,
    'gas_dist': 5.0,
    'internal_ee_use': True,
    'h_price': 120.0,
    'h_cover': 0.99,
    'imp_price': 0.0,
    'imp_max': 0.0,
    'k_start_cost': 0.0,
    'bess_cycle_cost': 0.0,
    'bess_cap': 1.0,
}

t_gen, t_tech = st.tabs(["Obecné", "Technika"])

with t_gen:
    col1, col2 = st.columns(2)
    with col1:
        p['dist_ee_buy'] = st.number_input("Distribuce nákup EE [€/MWh]", value=p['dist_ee_buy'])
        p['dist_ee_sell'] = st.number_input("Distribuce prodej EE [€/MWh]", value=p['dist_ee_sell'])
        p['gas_dist'] = st.number_input("Distribuce plyn [€/MWh]", value=p['gas_dist'])
    with col2:
        p['internal_ee_use'] = st.checkbox("Ušetřit distribuce při interní spotřebě EE", value=True)
        p['h_price'] = st.number_input("Prodejní cena tepla [€/MWh]", value=p['h_price'])
        p['h_cover'] = st.slider("Minimální pokrytí poptávky", 0.0, 1.0, p['h_cover'])

with t_tech:
    if use_kgj:
        st.subheader("Kogenerace (KGJ)")
        p['k_th'] = st.number_input("Tepelný výkon [MW]", value=1.09)
        p['k_el'] = st.number_input("Elektrický výkon [MW]", value=1.0)
        p['k_eff_th'] = st.number_input("Tepelná účinnost", value=0.46)
        p['k_min'] = st.slider("Min. zatížení [%]", 0, 100, 55) / 100
        p['k_start_cost'] = st.number_input("Náklady na start [€/start]", value=1200.0)
        p['k_min_runtime'] = st.number_input("Min. doba běhu [hod]", value=4, min_value=1)
        p['kgj_gas_fix'] = st.checkbox("Fixní cena plynu pro KGJ")
        if p['kgj_gas_fix']:
            p['kgj_gas_fix_price'] = st.number_input("Fixní cena plynu [€/MWh]", value=st.session_state.avg_gas_raw)

    if use_boil:
        st.subheader("Plynový kotel")
        p['b_max'] = st.number_input("Max. výkon [MW]", value=3.91)
        p['boil_gas_fix'] = st.checkbox("Fixní cena plynu pro kotel")
        if p['boil_gas_fix']:
            p['boil_gas_fix_price'] = st.number_input("Fixní cena plynu [€/MWh]", value=st.session_state.avg_gas_raw)

    if use_ek:
        st.subheader("Elektrokotel")
        p['ek_max'] = st.number_input("Max. výkon [MW]", value=0.61)
        p['ek_ee_fix'] = st.checkbox("Fixní cena EE pro EK")
        if p['ek_ee_fix']:
            p['ek_ee_fix_price'] = st.number_input("Fixní cena EE [€/MWh]", value=st.session_state.avg_ee_raw)

    if use_tes:
        st.subheader("Nádrž TES")
        p['tes_cap'] = st.number_input("Kapacita [MWh]", value=10.0)
        p['tes_loss'] = st.number_input("Ztráta [%/h]", value=0.5) / 100

    if use_bess:
        st.subheader("Baterie BESS")
        p['bess_cap'] = st.number_input("Kapacita [MWh]", value=1.0)
        p['bess_p'] = st.number_input("Výkon [MW]", value=0.5)
        p['bess_cycle_cost'] = st.number_input("Náklady na cyklus [€/cyklus]", value=0.0)
        p['bess_ee_fix'] = st.checkbox("Fixní cena EE pro BESS")
        if p['bess_ee_fix']:
            p['bess_ee_fix_price'] = st.number_input("Fixní cena EE [€/MWh]", value=st.session_state.avg_ee_raw)

    if use_fve:
        st.subheader("Fotovoltaika FVE")
        p['fve_installed_p'] = st.number_input("Instalovaný výkon [MW]", value=1.0)

    if use_ext_heat:
        st.subheader("Nákup tepla")
        p['imp_max'] = st.number_input("Max. výkon [MW]", value=2.0)
        p['imp_price'] = st.number_input("Cena importu [€/MWh]", value=150.0)

# ────────────────────────────────────────────────
# NAČTENÍ LOKÁLNÍCH DAT + START
# ────────────────────────────────────────────────
st.divider()
loc_file = st.file_uploader("Lokální data (poptávka, FVE, ...)", type=["xlsx"])

if st.session_state.fwd_data is not None and loc_file is not None:
    df_loc = pd.read_excel(loc_file)
    df_loc.columns = [str(c).strip() for c in df_loc.columns]
    df_loc.rename(columns={df_loc.columns[0]: 'datetime'}, inplace=True)
    df_loc['datetime'] = pd.to_datetime(df_loc['datetime'], dayfirst=True)

    df = pd.merge(st.session_state.fwd_data, df_loc, on='datetime', how='inner').fillna(0)
    T = len(df)

    if use_fve and 'fve_installed_p' in p:
        if 'FVE (MW)' in df.columns:
            df['FVE (MW)'] *= p['fve_installed_p']

    if st.button("🏁 Spustit optimalizaci", type="primary"):
        with st.spinner("Běží optimalizace..."):
            model = pulp.LpProblem("KGJ_Dispatch", pulp.LpMaximize)

            # Proměnné (s podmínkami)
            q_kgj = pulp.LpVariable.dicts("q_KGJ", range(T), 0) if use_kgj else {t: 0 for t in range(T)}
            q_boil = pulp.LpVariable.dicts("q_Boil", range(T), 0, p['b_max']) if use_boil else {t: 0 for t in range(T)}
            q_ek = pulp.LpVariable.dicts("q_EK", range(T), 0, p['ek_max']) if use_ek else {t: 0 for t in range(T)}
            q_imp = pulp.LpVariable.dicts("q_Imp", range(T), 0, p['imp_max']) if use_ext_heat else {t: 0 for t in range(T)}
            on = pulp.LpVariable.dicts("on", range(T), 0, 1, "Binary") if use_kgj else {t: 0 for t in range(T)}
            start = pulp.LpVariable.dicts("start", range(T), 0, 1, "Binary") if use_kgj else {t: 0 for t in range(T)}

            tes_soc = pulp.LpVariable.dicts("TES_SOC", range(T+1), 0, p['tes_cap']) if use_tes else {t: 0 for t in range(T+1)}
            tes_in = pulp.LpVariable.dicts("TES_In", range(T), 0) if use_tes else {t: 0 for t in range(T)}
            tes_out = pulp.LpVariable.dicts("TES_Out", range(T), 0) if use_tes else {t: 0 for t in range(T)}

            bess_soc = pulp.LpVariable.dicts("BESS_SOC", range(T+1), 0, p['bess_cap']) if use_bess else {t: 0 for t in range(T+1)}
            bess_cha = pulp.LpVariable.dicts("BESS_Cha", range(T), 0, p['bess_p']) if use_bess else {t: 0 for t in range(T)}
            bess_dis = pulp.LpVariable.dicts("BESS_Dis", range(T), 0, p['bess_p']) if use_bess else {t: 0 for t in range(T)}

            ee_export = pulp.LpVariable.dicts("ee_export", range(T), 0)
            ee_import = pulp.LpVariable.dicts("ee_import", range(T), 0)
            heat_shortfall = pulp.LpVariable.dicts("shortfall", range(T), 0)
            heat_delivered = pulp.LpVariable.dicts("heat_delivered", range(T), 0)

            if use_tes: model += tes_soc[0] == p['tes_cap'] * 0.5
            if use_bess: model += bess_soc[0] == p['bess_cap'] * 0.2

            # KGJ omezení
            if use_kgj:
                for t in range(T):
                    model += q_kgj[t] <= p['k_th'] * on[t]
                    model += q_kgj[t] >= p['k_min'] * p['k_th'] * on[t]
                for t in range(1, T):
                    model += on[t] - on[t-1] == start[t]
                for t in range(T):
                    for dt in range(1, int(p['k_min_runtime'])):
                        if t + dt < T:
                            model += on[t + dt] >= start[t]

            obj = []
            for t in range(T):
                p_ee = df['ee_price'].iloc[t]
                p_gas = df['gas_price'].iloc[t]

                # Fixní ceny mají přednost
                if use_kgj and p.get('kgj_gas_fix', False):
                    p_gas = p.get('kgj_gas_fix_price', p_gas)
                if use_boil and p.get('boil_gas_fix', False):
                    p_gas = p.get('boil_gas_fix_price', p_gas)
                if use_ek and p.get('ek_ee_fix', False):
                    p_ee = p.get('ek_ee_fix_price', p_ee)
                if use_bess and p.get('bess_ee_fix', False):
                    p_ee = p.get('bess_ee_fix_price', p_ee)

                h_dem = df['Poptávka po teple (MW)'].iloc[t]
                fve = df['FVE (MW)'].iloc[t] if use_fve and 'FVE (MW)' in df.columns else 0.0

                heat_prod = q_kgj[t] + q_boil[t] + q_ek[t] + q_imp[t]

                if use_tes:
                    model += tes_soc[t+1] == tes_soc[t] * (1 - p['tes_loss']) + tes_in[t] - tes_out[t]

                model += heat_delivered[t] == heat_prod + tes_out[t] - tes_in[t]
                model += heat_delivered[t] <= h_dem * p['h_cover']
                model += heat_delivered[t] + heat_shortfall[t] >= h_dem * p['h_cover']

                ee_kgj = q_kgj[t] * (p['k_el'] / p['k_th']) if use_kgj else 0
                model += ee_kgj + fve + ee_import[t] + bess_dis[t] == (q_ek[t] / 0.95) + bess_cha[t] + ee_export[t]

                if use_bess:
                    model += bess_soc[t+1] == bess_soc[t] + bess_cha[t] * 0.90 - bess_dis[t] / 0.90

                dist_buy = 0 if p['internal_ee_use'] else p['dist_ee_buy']
                dist_sell = 0 if p['internal_ee_use'] else p['dist_ee_sell']

                revenue = p['h_price'] * heat_delivered[t] + (p_ee - dist_sell) * ee_export[t]

                costs = (p_gas + p['gas_dist']) * (q_kgj[t]/p['k_eff_th'] + q_boil[t]/0.95) + \
                        (p_ee + dist_buy) * ee_import[t] + \
                        p['k_start_cost'] * start[t] + \
                        p['imp_price'] * q_imp[t]

                if use_bess:
                    cycle_energy = (bess_cha[t] + bess_dis[t]) / 2
                    costs += p['bess_cycle_cost'] * cycle_energy / p['bess_cap']

                obj.append(revenue - costs - p['h_price'] * heat_shortfall[t])

            model += pulp.lpSum(obj)
            status = model.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=120))

        st.subheader("Výsledky")
        st.write(f"Status: {pulp.LpStatus[status]} | Zisk: {pulp.value(model.objective):,.0f} €")

        if status == 1:
            # Zde by měly přijít grafy a tabulka – pokud chceš, přidej je z předchozí verze
            st.success("Optimalizace dokončena. (Grafy lze přidat stejně jako dříve)")
        else:
            st.error("Optimalizace selhala – zkus změnit parametry")

else:
    st.info("Nahraj FWD křivku a lokální data pro spuštění.")

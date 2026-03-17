import io
import streamlit as st
import pandas as pd
import numpy as np
import pulp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

st.set_page_config(page_title="KGJ Strategy Expert PRO", layout="wide")

# ── Session state ────────────────────────────────
for key, default in [
    ('fwd_data', None), ('avg_ee_raw', 100.0), ('avg_gas_raw', 50.0),
    ('ee_new', 100.0), ('gas_new', 50.0), ('results', None), ('df_main', None),
    ('scenario_results', None), ('selected_profile', 'free'),
]:
    if key not in st.session_state:
        st.session_state[key] = default

st.title("🚀 KGJ Strategy & Dispatch Optimizer PRO")

MONTH_NAMES = {1:'Led',2:'Úno',3:'Bře',4:'Dub',5:'Kvě',6:'Čvn',
               7:'Čvc',8:'Srp',9:'Zář',10:'Říj',11:'Lis',12:'Pro'}

# ════════════════════════════════════════════════════════════════════
# SCHEDULING PROFILES & SCENARIO MANAGEMENT
# ════════════════════════════════════════════════════════════════════

def create_profile_constraints(df, profile_type, custom_hours=None):
    """
    Vytvoří binary constrainty pro KGJ provoz dle profilu
    Returns: list[bool] kde True = povoleno běžet, False = musí být OFF
    """
    df_work = df.copy()
    df_work['hour'] = pd.to_datetime(df_work['datetime']).dt.hour
    
    if profile_type == 'base':
        # Celý den (0-23)
        constraints = [True] * len(df_work)
    
    elif profile_type == 'peak':
        # 9-21 (Peak hours)
        constraints = [h in range(9, 22) for h in df_work['hour']]
    
    elif profile_type == 'extpeak':
        # 6-22 (Extended Peak)
        constraints = [h in range(6, 23) for h in df_work['hour']]
    
    elif profile_type == 'custom' and custom_hours:
        # Custom hours zadané uživatelem
        constraints = [h in custom_hours for h in df_work['hour']]
    
    else:
        # 'free' nebo neurčeno = bez omezení
        constraints = [True] * len(df_work)
    
    return constraints


def apply_profile_constraints_to_model(model, on, constraints, T):
    """Aplikuj profile constrainty do PuLP modelu"""
    if constraints is None:
        return model
    
    for t in range(T):
        if not constraints[t]:
            model += on[t] == 0, f"profile_constraint_{t}"
    
    return model


def calculate_smoothness_metrics(res):
    """
    Spočítej metriky hladkosti provozu KGJ
    """
    kgj_on = res['KGJ on'].values
    
    # Počet ON→OFF a OFF→ON přechodů
    transitions = np.sum(np.abs(np.diff(kgj_on)) > 0.5)
    
    # Délky kontinuálních běhů
    run_lengths = []
    current_run = 0
    for i in range(len(kgj_on)):
        if kgj_on[i] > 0.5:
            current_run += 1
        else:
            if current_run > 0:
                run_lengths.append(current_run)
            current_run = 0
    if current_run > 0:
        run_lengths.append(current_run)
    
    avg_run_length = np.mean(run_lengths) if run_lengths else 0
    min_run_length = np.min(run_lengths) if run_lengths else 0
    max_run_length = np.max(run_lengths) if run_lengths else 0
    
    # Stabilita skóre (0-100%, méně transakcí = vyšší skóre)
    stability_score = max(0, 100 * (1 - transitions / (len(kgj_on) / 2))) if len(kgj_on) > 0 else 0
    
    return {
        'transitions': int(transitions),
        'stability_score': stability_score,
        'avg_run_hours': avg_run_length,
        'min_run_hours': min_run_length,
        'max_run_hours': max_run_length,
        'total_on_hours': int(np.sum(kgj_on)),
    }


def create_scenario_comparison_df(scenarios):
    """Vytvoř DataFrame s porovnáním scénářů"""
    data = []
    
    for profile_name, scenario in scenarios.items():
        if scenario['result'] is None:
            continue
        res = scenario['result']['res']
        smooth = scenario['smoothness']
        
        profit = scenario['result']['total_profit']
        shortfall = res['Shortfall [MW]'].sum() if 'Shortfall [MW]' in res.columns else 0
        
        data.append({
            'Profil': profile_name.upper(),
            'Zisk [€]': f"{profit:,.0f}",
            'Transitions': smooth['transitions'],
            'Stabilita [%]': f"{smooth['stability_score']:.1f}",
            'Avg Runtime [h]': f"{smooth['avg_run_hours']:.1f}",
            'Total Hours': smooth['total_on_hours'],
            'Shortfall [MWh]': f"{shortfall:.1f}",
        })
    
    return pd.DataFrame(data)


# ────────────────────────────────────────────────
# SIDEBAR
# ────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Technologie na lokalitě")
    use_kgj      = st.checkbox("Kogenerace (KGJ)",    value=True)
    use_boil     = st.checkbox("Plynový kotel",        value=True)
    use_ek       = st.checkbox("Elektrokotel",         value=True)
    use_tes      = st.checkbox("Nádrž (TES)",          value=True)
    use_bess     = st.checkbox("Baterie (BESS)",       value=True)
    use_fve      = st.checkbox("Fotovoltaika (FVE)",   value=True)
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
            years    = sorted(df_raw[date_col].dt.year.unique())
            sel_year = st.selectbox("Rok pro analýzu", years)
            df_year  = df_raw[df_raw[date_col].dt.year == sel_year].copy()

            avg_ee  = float(df_year.iloc[:, 1].mean())
            avg_gas = float(df_year.iloc[:, 2].mean())
            st.session_state.avg_ee_raw  = avg_ee
            st.session_state.avg_gas_raw = avg_gas
            st.info(f"Průměr EE: **{avg_ee:.1f} €/MWh** | Plyn: **{avg_gas:.1f} €/MWh**")

            ee_new  = st.number_input("Cílová base cena EE [€/MWh]",   value=round(avg_ee,  1), step=1.0)
            gas_new = st.number_input("Cílová base cena Plyn [€/MWh]", value=round(avg_gas, 1), step=1.0)

            df_fwd = df_year.copy()
            df_fwd.columns = ['datetime', 'ee_original', 'gas_original']
            df_fwd['ee_price']  = df_fwd['ee_original']  + (ee_new  - avg_ee)
            df_fwd['gas_price'] = df_fwd['gas_original'] + (gas_new - avg_gas)
            st.session_state.fwd_data = df_fwd
            st.session_state.ee_new   = ee_new
            st.session_state.gas_new  = gas_new
            st.success("FWD načteno ✔")
        except Exception as e:
            st.error(f"Chyba při načítání FWD: {e}")

    # ── NOVÉ: KGJ Scheduling Profily ──
    st.divider()
    st.header("📅 Analýza Období & Profily KGJ")
    
    # Výběr Období
    st.subheader("1️⃣ Období Analýzy")
    analysis_mode = st.radio("Vyberte režim:", ["Celá data", "Vlastní rozsah"])
    
    period_start = None
    period_end = None
    if analysis_mode == "Vlastní rozsah" and st.session_state.fwd_data is not None:
        min_date = pd.to_datetime(st.session_state.fwd_data['datetime']).min().date()
        max_date = pd.to_datetime(st.session_state.fwd_data['datetime']).max().date()
        
        col_ps, col_pe = st.columns(2)
        with col_ps:
            period_start = st.date_input("Od", value=min_date, key="period_start")
        with col_pe:
            period_end = st.date_input("Do", value=max_date, key="period_end")
    
    # KGJ Scheduling Profily
    st.subheader("2️⃣ Scheduling Profily KGJ")
    
    profiles_to_run = st.multiselect(
        "Které profily testovat?",
        options=['free', 'base', 'peak', 'extpeak', 'custom'],
        default=['free', 'base', 'peak', 'extpeak'],
        help="Spusť optimalizaci pro vybrané profily a porovnej je"
    )
    
    profile_definitions = {
        'free': {'name': 'Volná Opt.', 'hours': None, 'desc': 'Bez omezení'},
        'base': {'name': 'Base (0-24h)', 'hours': list(range(24)), 'desc': 'Celý den'},
        'peak': {'name': 'Peak (9-21h)', 'hours': list(range(9, 22)), 'desc': '12 hodin'},
        'extpeak': {'name': 'ExtPeak (6-22h)', 'hours': list(range(6, 23)), 'desc': '16 hodin'},
    }
    
    custom_hours = None
    if 'custom' in profiles_to_run:
        st.write("**Custom Profil** - Vyberte hodiny:")
        custom_hours = st.multiselect(
            "Povolené hodiny (0-23):",
            options=list(range(24)),
            default=list(range(6, 23)),
            key="custom_hours_selector"
        )
        profile_definitions['custom'] = {
            'name': f'Custom ({len(custom_hours)}h)',
            'hours': custom_hours,
            'desc': f"Custom"
        }
    
    # Provozní Omezení
    st.subheader("3️⃣ Omezení Provozování")
    
    use_month_start_limit = st.checkbox("Omezit max. startů za měsíc", value=False)
    max_starts_per_month = None
    if use_month_start_limit:
        max_starts_per_month = st.number_input("Max. startů/měsíc", value=5, min_value=1, max_value=30)
    
    # Scenario Mode
    st.subheader("4️⃣ Režim Porovnání")
    scenario_mode = st.radio(
        "Spusť optimalizaci:",
        ["Pouze aktuální nastavení", "Všechny vybrané profily"],
        help="'Všechny profily' = porovnání scénářů"
    )

# ────────────────────────────────────────────────
# FWD GRAFY
# ────────────────────────────────────────────────
if st.session_state.fwd_data is not None:
    df_fwd = st.session_state.fwd_data
    with st.expander("📈 FWD křivka – originál vs. upravená", expanded=True):
        tab_ee, tab_gas, tab_dur = st.tabs(["Elektřina [€/MWh]", "Plyn [€/MWh]", "Křivky trvání"])
        with tab_ee:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_fwd['datetime'], y=df_fwd['ee_original'],
                name='EE – originál', line=dict(color='#95a5a6', width=1, dash='dot')))
            fig.add_trace(go.Scatter(x=df_fwd['datetime'], y=df_fwd['ee_price'],
                name='EE – upravená', line=dict(color='#2ecc71', width=2)))
            fig.add_hline(y=st.session_state.avg_ee_raw, line_dash="dash", line_color="#95a5a6",
                annotation_text=f"Orig. průměr {st.session_state.avg_ee_raw:.1f}")
            fig.add_hline(y=st.session_state.ee_new, line_dash="dash", line_color="#27ae60",
                annotation_text=f"Nový průměr {st.session_state.ee_new:.1f}")
            fig.update_layout(height=340, hovermode='x unified', margin=dict(t=30))
            st.plotly_chart(fig, use_container_width=True)
        with tab_gas:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_fwd['datetime'], y=df_fwd['gas_original'],
                name='Plyn – originál', line=dict(color='#95a5a6', width=1, dash='dot')))
            fig.add_trace(go.Scatter(x=df_fwd['datetime'], y=df_fwd['gas_price'],
                name='Plyn – upravená', line=dict(color='#e67e22', width=2)))
            fig.add_hline(y=st.session_state.avg_gas_raw, line_dash="dash", line_color="#95a5a6",
                annotation_text=f"Orig. průměr {st.session_state.avg_gas_raw:.1f}")
            fig.add_hline(y=st.session_state.gas_new, line_dash="dash", line_color="#e67e22",
                annotation_text=f"Nový průměr {st.session_state.gas_new:.1f}")
            fig.update_layout(height=340, hovermode='x unified', margin=dict(t=30))
            st.plotly_chart(fig, use_container_width=True)
        with tab_dur:
            ee_s  = df_fwd['ee_price'].sort_values(ascending=False).values
            gas_s = df_fwd['gas_price'].sort_values(ascending=False).values
            hrs   = list(range(1, len(ee_s)+1))
            fig = make_subplots(rows=1, cols=2,
                subplot_titles=("Křivka trvání – EE", "Křivka trvání – Plyn"))
            fig.add_trace(go.Scatter(x=hrs, y=ee_s, name='EE',
                line=dict(color='#2ecc71', width=2), fill='tozeroy',
                fillcolor='rgba(46,204,113,0.15)'), row=1, col=1)
            fig.add_trace(go.Scatter(x=hrs, y=gas_s, name='Plyn',
                line=dict(color='#e67e22', width=2), fill='tozeroy',
                fillcolor='rgba(230,126,34,0.15)'), row=1, col=2)
            fig.update_xaxes(title_text="Hodiny [h]")
            fig.update_yaxes(title_text="€/MWh")
            fig.update_layout(height=340, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# ────────────────────────────────────────────────
# PARAMETRY
# ────────────────────────────────────────────────
p = {}
t_gen, t_tech = st.tabs(["Obecné", "Technika"])

with t_gen:
    col1, col2 = st.columns(2)
    with col1:
        p['dist_ee_buy']       = st.number_input("Distribuce nákup EE [€/MWh]",   value=33.0)
        p['dist_ee_sell']      = st.number_input("Distribuce prodej EE [€/MWh]",  value=2.0)
        p['gas_dist']          = st.number_input("Distribuce plyn [€/MWh]",        value=5.0)
    with col2:
        p['internal_ee_use']   = st.checkbox(
            "Ušetřit distribuci při interní spotřebě EE", value=True,
            help="Pokud spotřebu EE (EK, BESS) pokrývá lokální výroba (KGJ, FVE), distribuci neplatíme.")
        p['h_price']           = st.number_input("Prodejní cena tepla [€/MWh]",   value=120.0)
        p['h_cover']           = st.slider("Minimální pokrytí poptávky tepla", 0.0, 1.0, 0.99, step=0.01)
        p['shortfall_penalty'] = st.number_input("Penalizace za nedodání tepla [€/MWh]", value=500.0,
            help="Doporučeno 3–5× cena tepla. Vyšší = silnější priorita pokrytí poptávky.")

with t_tech:
    # ── KGJ ──────────────────────────────────────
    if use_kgj:
        st.subheader("Kogenerace (KGJ)")
        c1, c2 = st.columns(2)
        with c1:
            p['k_th']          = st.number_input("Jmenovitý tepelný výkon [MW]",  value=1.09)
            p['k_eff_th']      = st.number_input("Tepelná účinnost η_th [-]",      value=0.46)
            p['k_eff_el']      = st.number_input("Elektrická účinnost η_el [-]",   value=0.40)
            p['k_min']         = st.slider("Min. zatížení [%]", 0, 100, 55) / 100
        with c2:
            p['k_start_cost']  = st.number_input("Náklady na start [€/start]",    value=1200.0)
            p['k_min_runtime'] = st.number_input("Min. doba běhu [hod]",          value=4, min_value=1)
        k_el_derived = p['k_th'] * (p['k_eff_el'] / p['k_eff_th'])
        p['k_el'] = k_el_derived
        st.caption(f"ℹ️ Odvozený el. výkon: **{k_el_derived:.3f} MW** | "
                   f"Celková účinnost: **{p['k_eff_th']+p['k_eff_el']:.2f}**")
        # Roční limit hodin
        p['kgj_hour_limit_on'] = st.checkbox("Omezit max. počet provozních hodin KGJ / rok", value=False)
        if p['kgj_hour_limit_on']:
            p['kgj_hour_limit'] = st.number_input("Max. hodin provozu KGJ / rok", value=6000, min_value=1)
        p['kgj_gas_fix'] = st.checkbox("Fixní cena plynu pro KGJ")
        if p['kgj_gas_fix']:
            p['kgj_gas_fix_price'] = st.number_input("Fixní cena plynu – KGJ [€/MWh]",
                value=float(st.session_state.avg_gas_raw))

    # ── Kotel ─────────────────────────────────────
    if use_boil:
        st.subheader("Plynový kotel")
        p['b_max']    = st.number_input("Max. výkon [MW]",    value=3.91)
        p['boil_eff'] = st.number_input("Účinnost kotle [-]", value=0.95)
        p['boil_hour_limit_on'] = st.checkbox("Omezit max. počet provozních hodin kotle / rok", value=False)
        if p['boil_hour_limit_on']:
            p['boil_hour_limit'] = st.number_input("Max. hodin provozu kotle / rok", value=4000, min_value=1)
        p['boil_gas_fix'] = st.checkbox("Fixní cena plynu pro kotel")
        if p['boil_gas_fix']:
            p['boil_gas_fix_price'] = st.number_input("Fixní cena plynu – kotel [€/MWh]",
                value=float(st.session_state.avg_gas_raw))

    # ── Elektrokotel ──────────────────────────────
    if use_ek:
        st.subheader("Elektrokotel")
        p['ek_max'] = st.number_input("Max. výkon [MW]",  value=0.61)
        p['ek_eff'] = st.number_input("Účinnost EK [-]",  value=0.98)
        p['ek_ee_fix'] = st.checkbox("Fixní cena EE pro elektrokotel")
        if p['ek_ee_fix']:
            p['ek_ee_fix_price'] = st.number_input("Fixní cena EE – EK [€/MWh]",
                value=float(st.session_state.avg_ee_raw))

    # ── TES ───────────────────────────────────────
    if use_tes:
        st.subheader("Nádrž TES")
        p['tes_cap']  = st.number_input("Kapacita [MWh]", value=10.0)
        p['tes_loss'] = st.number_input("Ztráta [%/h]",   value=0.5) / 100

    # ── BESS ──────────────────────────────────────
    if use_bess:
        st.subheader("Baterie BESS")
        c1, c2 = st.columns(2)
        with c1:
            p['bess_cap']        = st.number_input("Kapacita [MWh]",                 value=1.0)
            p['bess_p']          = st.number_input("Max. výkon [MW]",                 value=0.5)
            p['bess_eff']        = st.number_input("Účinnost nabíjení/vybíjení [-]",  value=0.90)
            p['bess_cycle_cost'] = st.number_input("Náklady na opotřebení [€/MWh]",   value=5.0)
        with c2:
            st.markdown("**Distribuce pro arbitráž**")
            p['bess_dist_buy']  = st.checkbox("Účtovat distribuci NÁKUP do BESS",  value=False)
            p['bess_dist_sell'] = st.checkbox("Účtovat distribuci PRODEJ z BESS",  value=False)
            st.caption("💡 Interní arbitráž distribuci neplatí při zapnuté volbě 'Ušetřit distribuci'.")
        p['bess_ee_fix'] = st.checkbox("Fixní cena EE pro BESS")
        if p['bess_ee_fix']:
            p['bess_ee_fix_price'] = st.number_input("Fixní cena EE – BESS [€/MWh]",
                value=float(st.session_state.avg_ee_raw))

    # ── FVE ───────────────────────────────────────
    if use_fve:
        st.subheader("Fotovoltaika FVE")
        p['fve_installed_p'] = st.number_input("Instalovaný výkon [MW]", value=1.0,
            help="Profil FVE v lokálních datech = capacity factor 0–1.")
        p['fve_dist_sell'] = st.checkbox("Účtovat distribuci PRODEJ z FVE do sítě", value=False)

    # ── Import tepla ──────────────────────────────
    if use_ext_heat:
        st.subheader("Nákup tepla (Import)")
        p['imp_max']   = st.number_input("Max. výkon [MW]",      value=2.0)
        p['imp_price'] = st.number_input("Cena importu [€/MWh]", value=150.0)
        p['imp_hour_limit_on'] = st.checkbox("Omezit max. počet hodin importu tepla / rok", value=False)
        if p['imp_hour_limit_on']:
            p['imp_hour_limit'] = st.number_input("Max. hodin importu tepla / rok", value=2000, min_value=1)

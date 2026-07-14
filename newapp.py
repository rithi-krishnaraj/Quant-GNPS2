import os
import tempfile
import io
import zipfile
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import streamlit as st
from pyteomics import mzml, mgf

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="GNPS2-Quant | PRM Method Generator",
    page_icon="🔬",
    layout="wide"
)

# =============================================================================
# DEFAULT COLUMN MAPPING CONSTANTS
# =============================================================================
GNPS_COMPOUND_COL = "Compound_Name"
GNPS_SCAN_COL = "#Scan#"
GNPS_FORMULA_COL = "molecular_formula"
GNPS_ADDUCT_COL = "Adduct"
GNPS_CAS_COL = "CAS_Number"
GNPS_SMILES_COL = "Smiles"

MZMINE_SCAN_COL = "id"
MZMINE_MZ_COL = "mz"
MZMINE_RT_COL = "rt"
MZMINE_RT_START_COL = "rt_range:min"
MZMINE_RT_END_COL = "rt_range:max"
MZMINE_HEIGHT_COL = "height"
MZMINE_CHARGE_COL = "charge"

TARGETS_COMPOUND_COL = "Compound"
TARGETS_CAS_COL = "CAS"
TARGETS_SMILES_COL = "SMILES"
TARGETS_FORMULA_COL = "Formula"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def clean_cas(cas_string):
    """Removes dashes and spaces from CAS numbers for robust matching."""
    if pd.isna(cas_string):
        return ""
    return str(cas_string).replace("-", "").replace(" ", "").strip()

def get_core_name(name):
    """Strips prefixes, synonyms, salts, and punctuation to extract the core compound name."""
    name = str(name).lower()
    name = re.sub(r'^.*?:[a-z0-9_-]+\s+', '', name)
    name = re.sub(r'^[0-9]+_', '', name)
    name = name.split('|')[0]
    name = re.sub(r'\b(hcl|na|k|br|cl)\b', '', name)
    name = re.sub(r'[^a-z0-9]', '', name)
    return name

def calculate_scan_time(resolution):
    """Returns estimated Exploris 480 scan cycle time (ms) based on Orbitrap resolution."""
    transients = {15000: 16, 30000: 32, 60000: 64, 120000: 128, 240000: 256}
    res_key = min(transients.keys(), key=lambda k: abs(k - resolution))
    return transients[res_key] + 8  # transient + 8 ms C-trap overhead

def process_polarity(gnps_df, mzmine_df, polarity, targets_df, hcd_energies,
                     rt_window_mode="Composite range \u00b1 margin", rt_margin=1.0):
    """
    Matches targets against GNPS hits, merges with MZmine features to get m/z and RT,
    selects the highest-intensity feature per compound, and builds the RT window.
    """
    gnps_df = gnps_df.copy()
    gnps_df['clean_cas'] = gnps_df.get(GNPS_CAS_COL, pd.Series(dtype=str)).apply(clean_cas)
    gnps_df['clean_smiles'] = gnps_df.get(GNPS_SMILES_COL, pd.Series(dtype=str)).apply(
        lambda s: "" if pd.isna(s) else str(s).strip()
    )
    gnps_df['clean_formula'] = gnps_df.get(GNPS_FORMULA_COL, pd.Series(dtype=str)).apply(
        lambda f: "" if pd.isna(f) else str(f).replace(" ", "").lower()
    )

    matched_rows = []
    for _, target in targets_df.iterrows():
        t_cas     = target.get('clean_cas', "")
        t_name    = target[TARGETS_COMPOUND_COL]
        t_core    = get_core_name(t_name)
        _smiles   = target.get(TARGETS_SMILES_COL, "")
        _formula  = target.get(TARGETS_FORMULA_COL, "")
        t_smiles  = "" if pd.isna(_smiles)  else str(_smiles).strip()
        t_formula = "" if pd.isna(_formula) else str(_formula).replace(" ", "").lower()

        match_cond = gnps_df.apply(
            lambda row: (t_cas != "" and row['clean_cas'] == t_cas) or
                        (t_core != "" and t_core == get_core_name(row[GNPS_COMPOUND_COL])) or
                        (t_smiles != "" and row['clean_smiles'] != "" and row['clean_smiles'] == t_smiles) or
                        (t_formula != "" and row['clean_formula'] != "" and row['clean_formula'] == t_formula),
            axis=1
        )
        hits = gnps_df[match_cond].copy()
        if not hits.empty:
            hits['Standardized_Compound'] = t_name
            hits['Matched_GNPS_Name'] = hits[GNPS_COMPOUND_COL]
            matched_rows.append(hits)

    if not matched_rows:
        return pd.DataFrame()

    matched_gnps = pd.concat(matched_rows, ignore_index=True)
    merged = pd.merge(
        matched_gnps, mzmine_df,
        left_on=GNPS_SCAN_COL, right_on=MZMINE_SCAN_COL, how='inner'
    )

    if MZMINE_HEIGHT_COL in merged.columns:
        merged[MZMINE_HEIGHT_COL] = pd.to_numeric(merged[MZMINE_HEIGHT_COL], errors='coerce').fillna(0)

    results = []
    for compound_name, group in merged.groupby('Standardized_Compound'):
        best_idx = group[MZMINE_HEIGHT_COL].idxmax()
        best_row = group.loc[best_idx]

        peak_rt = best_row[MZMINE_RT_COL]
        composite_rt_min = group[MZMINE_RT_START_COL].min()
        composite_rt_max = group[MZMINE_RT_END_COL].max()
        
        if rt_window_mode == "Symmetric around peak RT":
            expanded_start = max(0, peak_rt - rt_margin)
            expanded_stop  = peak_rt + rt_margin
        else:
            expanded_start = max(0, composite_rt_min - rt_margin)
            expanded_stop  = composite_rt_max + rt_margin

        c_name = str(compound_name)
        final_name = c_name[:1].upper() + c_name[1:] if c_name else ""

        results.append({
            'Compound': final_name,
            'Formula': best_row.get(GNPS_FORMULA_COL, ""),
            'Polarity': polarity,
            'Adduct': best_row.get(GNPS_ADDUCT_COL, ""),
            'm/z': best_row[MZMINE_MZ_COL],
            'z': best_row.get(MZMINE_CHARGE_COL, 1 if polarity == "Positive" else -1),
            't start (min)': expanded_start,
            't stop (min)': expanded_stop,
            'Peak_RT': peak_rt,
            'Height': best_row[MZMINE_HEIGHT_COL],
            'Matched_GNPS_Name': best_row['Matched_GNPS_Name'],
            'Matched_Scan': best_row[GNPS_SCAN_COL], # Captured for Skyline extraction later
            'HCD Collision Energies (%)': hcd_energies,
        })

    return pd.DataFrame(results)

def build_skyline_df(inclusion_df, mgf_files):
    """
    Parses MGF files to extract fragment product ions for matched #Scan# numbers,
    formatting them into a Skyline-compatible Transition List.
    """
    if inclusion_df.empty or not mgf_files:
        return pd.DataFrame()

    scan_to_mz = {}
    for m_file in mgf_files:
        m_file.seek(0) # Reset stream pointer
        # use_index=False ensures no crashing on non-standard MGF headers
        with mgf.read(m_file, use_index=False) as reader:
            for spec in reader:
                params = spec.get('params', {})
                scan = params.get('scans', params.get('feature_id', None))
                if scan is not None:
                    try:
                        # Safety formatting if multiple scans recorded
                        scan = int(str(scan).split(',')[0])
                        mzs = spec.get('m/z array', [])
                        ints = spec.get('intensity array', [])
                        if len(mzs) > 0:
                            # Keep top 20 fragment ions to prevent overly bloated Skyline lists
                            sorted_indices = np.argsort(ints)[::-1]
                            top_mzs = mzs[sorted_indices][:20]
                            scan_to_mz[scan] = top_mzs
                    except ValueError:
                        pass

    rows = []
    for _, row in inclusion_df.iterrows():
        comp_name = row['Compound']
        try:
            scan_val = int(row.get('Matched_Scan', -1))
        except ValueError:
            scan_val = -1

        product_mzs = scan_to_mz.get(scan_val, [])
        rt_median = (row['t start (min)'] + row['t stop (min)']) / 2.0

        # If fragments exist, create a row for each. If missing, create one precursor-only row.
        if len(product_mzs) == 0:
            rows.append({
                'Molecule List Name': '',
                'Precursor Name': comp_name,
                'Precursor Formula': row.get('Formula', ''),
                'Precursor Adduct': row.get('Adduct', ''),
                'Precursor m/z': round(row['m/z'], 4),
                'Product m/z': '',
                'Precursor Charge': row['z'],
                'Product Charge': 1,
                'Explicit Retention Time': round(rt_median, 2)
            })
        else:
            for pmz in product_mzs:
                rows.append({
                    'Molecule List Name': '',
                    'Precursor Name': comp_name,
                    'Precursor Formula': row.get('Formula', ''),
                    'Precursor Adduct': row.get('Adduct', ''),
                    'Precursor m/z': round(row['m/z'], 4),
                    'Product m/z': round(pmz, 4),
                    'Precursor Charge': row['z'],
                    'Product Charge': 1, # Defaulting fragment to +1 or -1 charge state
                    'Explicit Retention Time': round(rt_median, 2)
                })

    return pd.DataFrame(rows)

def compute_concurrency_and_metrics(df, title_base, orbitrap_resolution, agc_target_percent, expected_peak_width_sec):
    if df.empty: return df, None
    actual_scan_time_ms = calculate_scan_time(orbitrap_resolution)
    time_grid   = np.arange(0, 16.01, 0.01)
    concurrency = np.zeros_like(time_grid)

    for _, row in df.iterrows():
        mask = (time_grid >= row['t start (min)']) & (time_grid <= row['t stop (min)'])
        concurrency[mask] += 1

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time_grid, concurrency, color='#8c564b', linewidth=2)
    ax.fill_between(time_grid, concurrency, color='#8c564b', alpha=0.3)
    ax.set_title(
        f"Concurrency Density Plot ({title_base})\n"
        f"Simulation: {orbitrap_resolution} Res, AGC {agc_target_percent}%"
        f"  |  Estimated Scan: {actual_scan_time_ms} ms",
        fontsize=13
    )
    ax.set_xlabel("Retention Time (min)")
    ax.set_ylabel("Number of Simultaneous Targets")
    ax.set_xlim(0, 16)
    ax.set_xticks(np.arange(0, 17, 2))
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    def get_metrics(row):
        mask = (time_grid >= row['t start (min)']) & (time_grid <= row['t stop (min)'])
        max_conc = int(np.max(concurrency[mask])) if np.any(mask) else 0
        max_cycle_time_sec = (max_conc * actual_scan_time_ms) / 1000.0
        pts = expected_peak_width_sec / max_cycle_time_sec if max_cycle_time_sec > 0 else 0
        return pd.Series([max_conc, max_cycle_time_sec, round(pts, 1)])

    df[['Max_Concurrent', 'Max_Cycle_Time_sec', 'Est_Points_Per_Peak']] = df.apply(get_metrics, axis=1)
    return df, fig

def build_mzml_figure(mzml_path, target_df, title_base, xic_ppm_tolerance):
    if not mzml_path or not os.path.exists(mzml_path): return None
    if target_df.empty: return None

    rt_list, tic_list = [], []
    target_df_sorted = target_df.sort_values('Peak_RT')
    targets = target_df_sorted.drop_duplicates('Compound')[
        ['Compound', 'm/z', 't start (min)', 't stop (min)', 'Peak_RT']
    ].to_dict('records')

    xic_data = {t['Compound']: [] for t in targets}
    target_ranges = {t['Compound']: (t['m/z'] - t['m/z'] * xic_ppm_tolerance / 1e6, t['m/z'] + t['m/z'] * xic_ppm_tolerance / 1e6) for t in targets}

    with mzml.read(mzml_path) as reader:
        for spec in reader:
            if spec.get('ms level') != 1: continue
            scan_data = spec.get('scanList', {}).get('scan', [{}])[0]
            rt = scan_data.get('scan start time')
            if rt is None: continue
            rt_val = (float(rt) / 60.0 if hasattr(rt, 'unit_info') and 'second' in rt.unit_info else float(rt))
            rt_list.append(rt_val)

            mzs  = spec.get('m/z array', np.array([]))
            ints = spec.get('intensity array', np.array([]))
            tic_list.append(ints.sum())

            if len(mzs) > 0:
                for comp, (mz_min, mz_max) in target_ranges.items():
                    mask = (mzs >= mz_min) & (mzs <= mz_max)
                    xic_data[comp].append(ints[mask].sum())
            else:
                for comp in target_ranges:
                    xic_data[comp].append(0)

    num_targets = len(targets)
    fig, axes = plt.subplots(2 + num_targets, 1, figsize=(12, 8 + 2 * num_targets), sharex=True)

    # Top: TIC
    ax1 = axes[0]
    ax1.plot(rt_list, tic_list, color='#444444', linewidth=1.5)
    ax1.fill_between(rt_list, tic_list, color='#444444', alpha=0.2)
    ax1.set_title(f"Total Ion Chromatogram & Target XICs ({title_base})", fontsize=14)
    ax1.set_ylabel("Total MS1 Intensity")
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Middle: overlaid XICs
    cmap = plt.get_cmap('tab20')
    ax2 = axes[1]
    for i, t in enumerate(targets):
        ax2.plot(rt_list, xic_data[t['Compound']], label=t['Compound'], linewidth=1.5, color=cmap(i % 20))
    ax2.set_ylabel("Overlaid XICs")
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left', borderaxespad=0., fontsize=9)

    # Bottom: individual XIC panels
    for i, t in enumerate(targets):
        ax    = axes[2 + i]
        comp  = t['Compound']
        color = cmap(i % 20)
        ax.plot(rt_list, xic_data[comp], color=color, linewidth=1.5)
        ax.fill_between(rt_list, xic_data[comp], color=color, alpha=0.3)
        ax.axvspan(t['t start (min)'], t['t stop (min)'], color='gray', alpha=0.15, zorder=0)
        ax.axvline(t['Peak_RT'], color='red', linestyle=':', linewidth=1.5, zorder=1)
        ax.text(0.01, 0.85, f"{comp}  (m/z {t['m/z']:.4f})",
                transform=ax.transAxes, fontsize=10, fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        ax.set_ylabel("Intensity", fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.7)

    axes[-1].set_xlabel("Retention Time (min)", fontsize=12)
    axes[-1].set_xlim(0, 16)
    axes[-1].set_xticks(np.arange(0, 17, 2))
    plt.tight_layout()
    return fig

def build_rt_alignment_figure(df, title):
    if df.empty: return None
    df = df.sort_values(by=['t start (min)', 'Peak_RT']).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.4)))
    y_labels = df['Compound'].tolist()
    y_ticks  = np.arange(len(y_labels))

    ax.grid(True, axis='both', linestyle='--', alpha=0.7, zorder=1)
    ax.hlines(y=y_ticks, xmin=df['t start (min)'], xmax=df['t stop (min)'], color='#222222', linewidth=3, zorder=2)
    ax.plot(df['Peak_RT'], y_ticks, 'o', color='red', markersize=5, zorder=3)

    for i, row in enumerate(df.itertuples()):
        ax.text(row.Peak_RT, i - 0.25, f"{row.Peak_RT:.2f}", va='bottom', ha='center', fontsize=9, zorder=4)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Retention time (min)")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.set_ylim(len(y_labels) - 0.5, -1.5)
    ax.set_xlim(0, 16)
    ax.set_xticks(np.arange(0, 17, 2))
    plt.tight_layout()
    return fig

def fig_to_svg_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='svg', bbox_inches='tight')
    buf.seek(0)
    return buf.getvalue()

def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')

def build_zip(r):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('Output_Match_Report.csv', df_to_csv_bytes(r['report_df']))
        zf.writestr('Output_Intermediate_Mass-List-Table.csv', df_to_csv_bytes(r['intermediate_df']))
        if not r['final_pos_df'].empty:
            zf.writestr('Output_Mass-List-Table_pos.csv', df_to_csv_bytes(r['final_pos_df']))
        if not r['final_neg_df'].empty:
            zf.writestr('Output_Mass-List-Table_neg.csv', df_to_csv_bytes(r['final_neg_df']))
        if 'skyline_pos_df' in r and not r['skyline_pos_df'].empty:
            zf.writestr('Output_Skyline_Mass-List_pos.csv', df_to_csv_bytes(r['skyline_pos_df']))
        if 'skyline_neg_df' in r and not r['skyline_neg_df'].empty:
            zf.writestr('Output_Skyline_Mass-List_neg.csv', df_to_csv_bytes(r['skyline_neg_df']))

        if r['fig_conc_pos']: zf.writestr('Output_Concurrency_pos.svg', fig_to_svg_bytes(r['fig_conc_pos']))
        if r['fig_conc_neg']: zf.writestr('Output_Concurrency_neg.svg', fig_to_svg_bytes(r['fig_conc_neg']))
        if r['fig_chrom_pos']: zf.writestr('Output_Chromatograms_pos.svg', fig_to_svg_bytes(r['fig_chrom_pos']))
        if r['fig_chrom_neg']: zf.writestr('Output_Chromatograms_neg.svg', fig_to_svg_bytes(r['fig_chrom_neg']))
        if r['fig_rt_pos']: zf.writestr('Output_Retention-Time-Alignment_pos.svg', fig_to_svg_bytes(r['fig_rt_pos']))
        if r['fig_rt_neg']: zf.writestr('Output_Retention-Time-Alignment_neg.svg', fig_to_svg_bytes(r['fig_rt_neg']))
    buf.seek(0)
    return buf.getvalue()

def finalize_inclusion_list(df):
    if df.empty: return df
    out = df.copy()
    out['Formula'] = ""
    out['Adduct']  = ""
    out['t start (min)'] = out['t start (min)'].round(2)
    out['t stop (min)']  = out['t stop (min)'].round(2)
    out['m/z'] = out['m/z'].round(4)
    return out.drop(
        columns=['Peak_RT', 'Polarity', 'Height', 'Matched_GNPS_Name', 'Matched_Scan',
                 'Max_Concurrent', 'Max_Cycle_Time_sec', 'Est_Points_Per_Peak'],
        errors='ignore'
    )

# =============================================================================
# SIDEBAR — Configuration
# =============================================================================
with st.sidebar:
    st.header("⚙️ Instrument Settings")
    hcd_energies = st.text_input("HCD Collision Energies (%)", value="25,35,45", help="Comma-separated NCE values applied to all targets")
    xic_ppm_tolerance = st.number_input("XIC m/z Tolerance (ppm)", min_value=1, max_value=100, value=10)
    expected_peak_width_sec = st.number_input("Expected Peak Width at Base (sec)", min_value=1, max_value=120, value=10)
    orbitrap_resolution = st.selectbox("Orbitrap Resolution", options=[15000, 30000, 60000, 120000], index=1, format_func=lambda x: f"{x:,}")
    agc_target_percent = st.number_input("AGC Target (%)", min_value=1, max_value=500, value=50)

    st.divider()
    st.markdown("**⏱ RT Window**")
    rt_window_mode = st.radio("Window Mode", options=["Composite range \u00b1 margin", "Symmetric around peak RT"])
    rt_margin = st.number_input("RT Margin (min)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

    st.divider()
    with st.expander("🔧 Advanced: Column Mappings"):
        st.caption("GNPS2 output columns")
        GNPS_COMPOUND_COL = st.text_input("Compound Name", value="Compound_Name")
        GNPS_SCAN_COL = st.text_input("Scan Number", value="#Scan#")
        GNPS_FORMULA_COL = st.text_input("Molecular Formula", value="molecular_formula")
        GNPS_ADDUCT_COL = st.text_input("Adduct", value="Adduct")
        GNPS_CAS_COL    = st.text_input("CAS Number", value="CAS_Number")
        GNPS_SMILES_COL = st.text_input("SMILES",      value="Smiles")

        st.caption("MZmine 3 feature table columns")
        MZMINE_SCAN_COL = st.text_input("Feature ID",  value="id")
        MZMINE_MZ_COL = st.text_input("m/z",         value="mz")
        MZMINE_RT_COL       = st.text_input("RT",            value="rt")
        MZMINE_RT_START_COL = st.text_input("RT Start",     value="rt_range:min")
        MZMINE_RT_END_COL   = st.text_input("RT End",       value="rt_range:max")
        MZMINE_HEIGHT_COL   = st.text_input("Height",       value="height")
        MZMINE_CHARGE_COL   = st.text_input("Charge",       value="charge")

        st.caption("Target compounds file columns")
        TARGETS_COMPOUND_COL = st.text_input("Compound Column", value="Compound")
        TARGETS_CAS_COL      = st.text_input("CAS Column",      value="CAS")
        TARGETS_SMILES_COL   = st.text_input("SMILES Column",   value="SMILES")
        TARGETS_FORMULA_COL  = st.text_input("Formula Column",  value="Formula")

# =============================================================================
# MAIN — Title
# =============================================================================
st.title("🔬 GNPS2-Quant")
st.markdown("Convert **GNPS2** spectral library matches and **MZmine 3** feature tables into Parallel Reaction Monitoring (PRM) inclusion lists.")

# =============================================================================
# MAIN — File Uploads
# =============================================================================

# --- Workflow Mode Selection ---
st.subheader("1. Select Workflow Mode")
workflow_mode = st.radio(
    "How should the PRM target list be generated?",
    options=[
        "Targeted (Match DDA data against an internal library)",
        "Untargeted Super Mix (Extract all annotated hits from GNPS file directly)"
    ],
    help="Targeted requires a Compounds.csv list. Super Mix builds the list based purely on what GNPS annotated."
)

st.subheader("2. Upload Inputs")
with st.expander("📁 Feature & Library Matching", expanded=True):
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**Core Setup**")
        if "Targeted" in workflow_mode:
            f_compounds = st.file_uploader("Compounds.csv — Target list", type=["csv"], key="compounds")
        else:
            f_compounds = None
            st.info("💡 Target list disabled: Generating targets from all GNPS annotations automatically.")
    
    with col_right:
        st.markdown("**Raw Chromatograms (Optional)**")
        f_mzml_pos = st.file_uploader("LC-MS Raw Data — ESI+ (.mzML)", type=["mzML", "mzml"], key="mzml_pos")
        f_mzml_neg = st.file_uploader("LC-MS Raw Data — ESI− (.mzML)", type=["mzML", "mzml"], key="mzml_neg")

    # To support Multi-File loading for experimental sets, accept_multiple_files=True
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        f_gnps_pos   = st.file_uploader("GNPS2 Results — ESI+ (.csv)", type=["csv"], accept_multiple_files=True, key="gnps_pos")
        f_mzmine_pos = st.file_uploader("MZmine 3 Features — ESI+ (.csv)", type=["csv"], accept_multiple_files=True, key="mzmine_pos")
    with col_dl2:
        f_gnps_neg   = st.file_uploader("GNPS2 Results — ESI− (.csv)", type=["csv"], accept_multiple_files=True, key="gnps_neg")
        f_mzmine_neg = st.file_uploader("MZmine 3 Features — ESI− (.csv)", type=["csv"], accept_multiple_files=True, key="mzmine_neg")

st.subheader("3. Skyline Export")
with st.expander("📊 Skyline Mass List Configuration", expanded=True):
    generate_skyline = st.checkbox("Generate Skyline Mass List? (Extracts fragments from MGF)", value=False)
    if generate_skyline:
        skyline_modes = st.multiselect("Ionization Modes for Skyline", ["Positive", "Negative"], default=["Positive", "Negative"])
        col_sky1, col_sky2 = st.columns(2)
        with col_sky1:
            if "Positive" in skyline_modes:
                f_mgf_pos = st.file_uploader("GNPS MGF — ESI+ (.mgf)", type=["mgf"], accept_multiple_files=True, key="mgf_pos")
            else:
                f_mgf_pos = []
        with col_sky2:
            if "Negative" in skyline_modes:
                f_mgf_neg = st.file_uploader("GNPS MGF — ESI− (.mgf)", type=["mgf"], accept_multiple_files=True, key="mgf_neg")
            else:
                f_mgf_neg = []
    else:
        f_mgf_pos = []
        f_mgf_neg = []

# Minimal safety check
required_pos_match = len(f_gnps_pos) == len(f_mzmine_pos)
required_neg_match = len(f_gnps_neg) == len(f_mzmine_neg)
has_any_data = (len(f_gnps_pos) > 0 and required_pos_match) or (len(f_gnps_neg) > 0 and required_neg_match)
if "Targeted" in workflow_mode:
    all_ready = has_any_data and (f_compounds is not None)
else:
    all_ready = has_any_data

if not all_ready:
    st.info("Upload matching pairs of GNPS and MZmine files to proceed. If using Targeted mode, upload a Compounds list.")
elif not required_pos_match or not required_neg_match:
    st.error("Error: Number of GNPS files must equal the number of MZmine files for each polarity.")

run_clicked = st.button("▶  Run Pipeline", type="primary", disabled=not all_ready)

st.divider()

# =============================================================================
# PIPELINE
# =============================================================================
if run_clicked:
    with st.status("Running PRM generation pipeline…", expanded=True) as status:
        
        # Determine internal target list
        if "Targeted" in workflow_mode:
            st.write("Loading internal library targets...")
            targets = pd.read_csv(f_compounds, encoding='latin1')
            targets['clean_cas'] = targets.get(TARGETS_CAS_COL, pd.Series(dtype=str)).apply(clean_cas)
        else:
            st.write("Generating Super Mix targets from GNPS hits...")
            all_compounds = set()
            for f in (f_gnps_pos + f_gnps_neg):
                f.seek(0)
                temp_df = pd.read_csv(f, encoding='latin1', usecols=lambda x: x == GNPS_COMPOUND_COL)
                if GNPS_COMPOUND_COL in temp_df.columns:
                    all_compounds.update(temp_df[GNPS_COMPOUND_COL].dropna().unique())
            
            targets = pd.DataFrame({TARGETS_COMPOUND_COL: list(all_compounds)})
            targets['clean_cas'] = ""
        
        # Processing multi-file pairs
        def run_polarity_multipe(gnps_files, mzmine_files, polarity):
            # Sort to ensure stable matching pairs
            g_files = sorted(gnps_files, key=lambda x: x.name)
            m_files = sorted(mzmine_files, key=lambda x: x.name)
            
            all_res = []
            for g_f, m_f in zip(g_files, m_files):
                g_f.seek(0); m_f.seek(0)
                gnps_df = pd.read_csv(g_f, encoding='latin1')
                mz_df = pd.read_csv(m_f, encoding='latin1')
                res = process_polarity(gnps_df, mz_df, polarity, targets, hcd_energies, rt_window_mode, rt_margin)
                if not res.empty:
                    all_res.append(res)
                    
            if not all_res: return pd.DataFrame()
            combined = pd.concat(all_res, ignore_index=True)
            return combined

        st.write("Matching ESI+ runs...")
        targets_pos = run_polarity_multipe(f_gnps_pos, f_mzmine_pos, "Positive")
        
        st.write("Matching ESI− runs...")
        targets_neg = run_polarity_multipe(f_gnps_neg, f_mzmine_neg, "Negative")

        all_targets_df = pd.concat([targets_pos, targets_neg], ignore_index=True)

        if all_targets_df.empty:
            status.update(label="No targets matched.", state="error", expanded=True)
            st.error("No targets were matched. Check your input files and column mappings in the sidebar.")
            st.stop()

        # --- Cross-polarity conflict resolution ---
        st.write("Resolving cross-polarity conflicts...")
        best_targets_df = (
            all_targets_df
            .sort_values(by=['Compound', 'Height'], ascending=[True, False])
            .drop_duplicates(subset=['Compound'], keep='first')
        )
        final_targets_pos = best_targets_df[best_targets_df['Polarity'] == 'Positive'].copy()
        final_targets_neg = best_targets_df[best_targets_df['Polarity'] == 'Negative'].copy()

        # --- Concurrency & scan metrics ---
        st.write("Calculating scan concurrency and multiplexing metrics…")
        final_targets_pos, fig_conc_pos = compute_concurrency_and_metrics(
            final_targets_pos, "ESI+", orbitrap_resolution, agc_target_percent, expected_peak_width_sec
        )
        final_targets_neg, fig_conc_neg = compute_concurrency_and_metrics(
            final_targets_neg, "ESI−", orbitrap_resolution, agc_target_percent, expected_peak_width_sec
        )

        # --- Skyline Mass List Generation ---
        skyline_pos_df = pd.DataFrame()
        skyline_neg_df = pd.DataFrame()
        if generate_skyline:
            st.write("Extracting fragments from MGF for Skyline Transition List...")
            if f_mgf_pos and not final_targets_pos.empty:
                skyline_pos_df = build_skyline_df(final_targets_pos, f_mgf_pos)
            if f_mgf_neg and not final_targets_neg.empty:
                skyline_neg_df = build_skyline_df(final_targets_neg, f_mgf_neg)

        # --- Match report ---
        st.write("Generating match report…")
        best_targets_metrics = pd.concat([final_targets_pos, final_targets_neg], ignore_index=True)
        report_rows = []
        for _, row in targets.iterrows():
            req_comp = row[TARGETS_COMPOUND_COL]
            req_cas  = row.get(TARGETS_CAS_COL, "")
            match    = best_targets_metrics[best_targets_metrics['Compound'].str.lower() == str(req_comp).lower()]
            
            if not match.empty:
                m = match.iloc[0]
                report_rows.append({
                    'Requested Compound':      req_comp,
                    'CAS Number':              req_cas,
                    'Status':                  'Matched ✓',
                    'Assigned Polarity':       m['Polarity'],
                    'GNPS Matched Name':       m['Matched_GNPS_Name'],
                    'Max Simultaneous Targets': str(int(m.get('Max_Concurrent', 0))),
                    'Estimated Points/Peak':   str(m.get('Est_Points_Per_Peak', 0)),
                })
            else:
                report_rows.append({
                    'Requested Compound':      req_comp,
                    'CAS Number':              req_cas,
                    'Status':                  'Not Matched ✗',
                    'Assigned Polarity':       'N/A',
                    'GNPS Matched Name':       'N/A',
                    'Max Simultaneous Targets': 'N/A',
                    'Estimated Points/Peak':   'N/A',
                })
        report_df = pd.DataFrame(report_rows)

        # --- Intermediate list ---
        intermediate_df = best_targets_metrics.drop(
            columns=['Peak_RT', 'Height', 'Matched_GNPS_Name', 'Matched_Scan',
                     'Max_Concurrent', 'Max_Cycle_Time_sec', 'Est_Points_Per_Peak'],
            errors='ignore'
        )

        # --- Final Xcalibur-ready PRM lists ---
        final_pos_df = finalize_inclusion_list(final_targets_pos)
        final_neg_df = finalize_inclusion_list(final_targets_neg)

        # --- mzML chromatograms (optional) ---
        fig_chrom_pos = fig_chrom_neg = None
        if f_mzml_pos or f_mzml_neg:
            st.write("Parsing mzML files and extracting XICs…")
        tmp_files = []
        try:
            if f_mzml_pos:
                with tempfile.NamedTemporaryFile(suffix='.mzML', delete=False) as tmp:
                    tmp.write(f_mzml_pos.read())
                    tmp_pos_path = tmp.name
                tmp_files.append(tmp_pos_path)
                fig_chrom_pos = build_mzml_figure(tmp_pos_path, final_targets_pos, "ESI+", xic_ppm_tolerance)
            if f_mzml_neg:
                with tempfile.NamedTemporaryFile(suffix='.mzML', delete=False) as tmp:
                    tmp.write(f_mzml_neg.read())
                    tmp_neg_path = tmp.name
                tmp_files.append(tmp_neg_path)
                fig_chrom_neg = build_mzml_figure(tmp_neg_path, final_targets_neg, "ESI−", xic_ppm_tolerance)
        finally:
            for p in tmp_files:
                try: os.unlink(p)
                except Exception: pass

        # --- RT alignment figures ---
        st.write("Generating retention time alignment figures…")
        fig_rt_pos = build_rt_alignment_figure(final_targets_pos, "Retention Time Windows — ESI+")
        fig_rt_neg = build_rt_alignment_figure(final_targets_neg, "Retention Time Windows — ESI−")

        status.update(label="Pipeline complete!", state="complete", expanded=False)

    # Store in session state
    st.session_state['results'] = {
        'report_df':       report_df,
        'intermediate_df': intermediate_df,
        'final_pos_df':    final_pos_df,
        'final_neg_df':    final_neg_df,
        'skyline_pos_df':  skyline_pos_df,
        'skyline_neg_df':  skyline_neg_df,
        'fig_conc_pos':    fig_conc_pos,
        'fig_conc_neg':    fig_conc_neg,
        'fig_chrom_pos':   fig_chrom_pos,
        'fig_chrom_neg':   fig_chrom_neg,
        'fig_rt_pos':      fig_rt_pos,
        'fig_rt_neg':      fig_rt_neg,
        'n_total':         len(targets),
        'n_found':         best_targets_df['Compound'].nunique(),
        'n_pos':           len(final_targets_pos),
        'n_neg':           len(final_targets_neg),
        'n_mzmine':        len(mz_pos) + len(mz_neg) if 'mz_pos' in locals() else 0,
    }
    st.session_state['selected_pos_df'] = final_pos_df
    st.session_state['selected_neg_df'] = final_neg_df

# =============================================================================
# RESULTS
# =============================================================================
if 'results' in st.session_state:
    r = st.session_state['results']

    st.header("📊 Results")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Requested (Targets)",  r['n_total'])
    m2.metric("Matched Unique",       r['n_found'])
    m3.metric("ESI+ Targets",         r['n_pos'])
    m4.metric("ESI− Targets",         r['n_neg'])

    st.divider()

    tab_report, tab_pos, tab_neg, tab_sky, tab_inter, tab_conc, tab_chrom, tab_rt, tab_dl = st.tabs([
        "Match Report",
        "PRM List — ESI+",
        "PRM List — ESI−",
        "Skyline",
        "Intermediate",
        "Concurrency",
        "Chromatograms",
        "RT Alignment",
        "⬇ Downloads",
    ])

    with tab_report:
        st.subheader("Compound Match Report")
        n_matched = (r['report_df']['Status'].str.startswith('Matched')).sum()
        st.caption(f"{n_matched} of {r['n_total']} targets matched.")
        st.dataframe(
            r['report_df'].style.apply(
                lambda col: ['background-color: #28a745; color: white' if str(v).startswith('Matched')
                             else 'background-color: #dc3545; color: white' if str(v).startswith('Not')
                             else '' for v in col],
                subset=['Status']
            ),
            width='stretch', hide_index=True
        )

    with tab_pos:
        st.subheader("PRM Inclusion List — ESI+")
        if r['final_pos_df'].empty: st.info("No ESI+ targets.")
        else:
            df_pos_sel = r['final_pos_df'].copy()
            df_pos_sel.insert(0, 'Include', True)
            edited_pos = st.data_editor(
                df_pos_sel,
                column_config={'Include': st.column_config.CheckboxColumn('Include', default=True)},
                width='stretch', hide_index=True, key='editor_pos'
            )
            st.session_state['selected_pos_df'] = edited_pos[edited_pos['Include']].drop(columns=['Include'])

    with tab_neg:
        st.subheader("PRM Inclusion List — ESI−")
        if r['final_neg_df'].empty: st.info("No ESI− targets.")
        else:
            df_neg_sel = r['final_neg_df'].copy()
            df_neg_sel.insert(0, 'Include', True)
            edited_neg = st.data_editor(
                df_neg_sel,
                column_config={'Include': st.column_config.CheckboxColumn('Include', default=True)},
                width='stretch', hide_index=True, key='editor_neg'
            )
            st.session_state['selected_neg_df'] = edited_neg[edited_neg['Include']].drop(columns=['Include'])

    with tab_sky:
        st.subheader("Skyline Transition List")
        c_sky1, c_sky2 = st.columns(2)
        with c_sky1:
            st.markdown("#### ESI+")
            if not r.get('skyline_pos_df', pd.DataFrame()).empty:
                st.dataframe(r['skyline_pos_df'], width='stretch', hide_index=True)
            else:
                st.info("No positive Skyline list generated.")
        with c_sky2:
            st.markdown("#### ESI−")
            if not r.get('skyline_neg_df', pd.DataFrame()).empty:
                st.dataframe(r['skyline_neg_df'], width='stretch', hide_index=True)
            else:
                st.info("No negative Skyline list generated.")

    with tab_inter:
        st.subheader("Intermediate Inclusion List")
        st.dataframe(r['intermediate_df'], width='stretch', hide_index=True)

    with tab_conc:
        st.subheader("Concurrency Density Plots")
        c1, c2 = st.columns(2)
        with c1:
            if r['fig_conc_pos']: st.pyplot(r['fig_conc_pos'], width='stretch')
        with c2:
            if r['fig_conc_neg']: st.pyplot(r['fig_conc_neg'], width='stretch')

    with tab_chrom:
        st.subheader("TIC & Extracted Ion Chromatograms")
        if r['fig_chrom_pos']: st.pyplot(r['fig_chrom_pos'], width='stretch')
        if r['fig_chrom_neg']: st.pyplot(r['fig_chrom_neg'], width='stretch')

    with tab_rt:
        st.subheader("Retention Time Window Alignment")
        c1, c2 = st.columns(2)
        with c1:
            if r['fig_rt_pos']: st.pyplot(r['fig_rt_pos'], width='stretch')
        with c2:
            if r['fig_rt_neg']: st.pyplot(r['fig_rt_neg'], width='stretch')

    with tab_dl:
        st.subheader("Download Outputs")
        sel_pos = st.session_state.get('selected_pos_df', r['final_pos_df'])
        sel_neg = st.session_state.get('selected_neg_df', r['final_neg_df'])
        r_dl = dict(r, final_pos_df=sel_pos, final_neg_df=sel_neg)

        st.download_button(
            "⬇ Download All Outputs (.zip)",
            data=build_zip(r_dl),
            file_name="Make-PRM_Outputs.zip",
            mime="application/zip",
            width='stretch',
            type="primary",
        )

        st.divider()
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.markdown("**CSV Tables**")
            st.download_button("⬇ Match Report", df_to_csv_bytes(r['report_df']), "Output_Match_Report.csv", "text/csv", width='stretch')
            st.download_button("⬇ Intermediate List", df_to_csv_bytes(r['intermediate_df']), "Output_Intermediate_Mass-List-Table.csv", "text/csv", width='stretch')
            if not sel_pos.empty:
                st.download_button("⬇ PRM List — ESI+", df_to_csv_bytes(sel_pos), "Output_Mass-List-Table_pos.csv", "text/csv", width='stretch')
            if not sel_neg.empty:
                st.download_button("⬇ PRM List — ESI−", df_to_csv_bytes(sel_neg), "Output_Mass-List-Table_neg.csv", "text/csv", width='stretch')
            if 'skyline_pos_df' in r and not r['skyline_pos_df'].empty:
                st.download_button("⬇ Skyline List — ESI+", df_to_csv_bytes(r['skyline_pos_df']), "Output_Skyline_Mass-List_pos.csv", "text/csv", width='stretch')
            if 'skyline_neg_df' in r and not r['skyline_neg_df'].empty:
                st.download_button("⬇ Skyline List — ESI−", df_to_csv_bytes(r['skyline_neg_df']), "Output_Skyline_Mass-List_neg.csv", "text/csv", width='stretch')

        with dl_col2:
            st.markdown("**SVG Figures**")
            if r['fig_conc_pos']: st.download_button("⬇ Concurrency ESI+", fig_to_svg_bytes(r['fig_conc_pos']), "Output_Concurrency_pos.svg", "image/svg+xml", width='stretch')
            if r['fig_conc_neg']: st.download_button("⬇ Concurrency ESI−", fig_to_svg_bytes(r['fig_conc_neg']), "Output_Concurrency_neg.svg", "image/svg+xml", width='stretch')
            if r['fig_chrom_pos']: st.download_button("⬇ Chromatograms ESI+", fig_to_svg_bytes(r['fig_chrom_pos']), "Output_Chromatograms_pos.svg", "image/svg+xml", width='stretch')
            if r['fig_chrom_neg']: st.download_button("⬇ Chromatograms ESI−", fig_to_svg_bytes(r['fig_chrom_neg']), "Output_Chromatograms_neg.svg", "image/svg+xml", width='stretch')
            if r['fig_rt_pos']: st.download_button("⬇ RT Alignment ESI+", fig_to_svg_bytes(r['fig_rt_pos']), "Output_Retention-Time-Alignment_pos.svg", "image/svg+xml", width='stretch')
            if r['fig_rt_neg']: st.download_button("⬇ RT Alignment ESI−", fig_to_svg_bytes(r['fig_rt_neg']), "Output_Retention-Time-Alignment_neg.svg", "image/svg+xml", width='stretch')
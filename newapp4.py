import os
import tempfile
import io
import zipfile
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import re
import argparse
from pathlib import Path
from pyteomics import mzml, mgf
import streamlit as st

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
# COLUMN MAPPING HELPER
# =============================================================================
def get_col_mapping():
    """Get current column mappings from session state or use defaults"""
    import streamlit as st
    
    mapping = {
        'gnps_compound': st.session_state.get('col_gnps_compound', GNPS_COMPOUND_COL),
        'gnps_scan': st.session_state.get('col_gnps_scan', GNPS_SCAN_COL),
        'gnps_formula': st.session_state.get('col_gnps_formula', GNPS_FORMULA_COL),
        'gnps_adduct': st.session_state.get('col_gnps_adduct', GNPS_ADDUCT_COL),
        'gnps_cas': st.session_state.get('col_gnps_cas', GNPS_CAS_COL),
        'gnps_smiles': st.session_state.get('col_gnps_smiles', GNPS_SMILES_COL),
        'mzmine_scan': st.session_state.get('col_mzmine_scan', MZMINE_SCAN_COL),
        'mzmine_mz': st.session_state.get('col_mzmine_mz', MZMINE_MZ_COL),
        'mzmine_rt': st.session_state.get('col_mzmine_rt', MZMINE_RT_COL),
        'mzmine_rt_start': st.session_state.get('col_mzmine_rt_start', MZMINE_RT_START_COL),
        'mzmine_rt_end': st.session_state.get('col_mzmine_rt_end', MZMINE_RT_END_COL),
        'mzmine_height': st.session_state.get('col_mzmine_height', MZMINE_HEIGHT_COL),
        'mzmine_charge': st.session_state.get('col_mzmine_charge', MZMINE_CHARGE_COL),
        'targets_compound': st.session_state.get('col_targets_compound', TARGETS_COMPOUND_COL),
        'targets_cas': st.session_state.get('col_targets_cas', TARGETS_CAS_COL),
        'targets_smiles': st.session_state.get('col_targets_smiles', TARGETS_SMILES_COL),
        'targets_formula': st.session_state.get('col_targets_formula', TARGETS_FORMULA_COL),
    }
    return mapping

def find_column_ci(df, col_name):
    """
    Find a column in a dataframe case-insensitively.
    Returns the actual column name if found, otherwise returns the original col_name.
    """
    if col_name in df.columns:
        return col_name
    
    col_name_lower = col_name.lower()
    for actual_col in df.columns:
        if actual_col.lower() == col_name_lower:
            return actual_col
    
    return col_name  # Return original if not found

def validate_columns_ci(df, required_cols, df_name="dataframe"):
    """
    Validate required columns exist (case-insensitive).
    Returns a dict mapping user-specified names to actual column names in the dataframe.
    Raises ValueError if any required columns are missing.
    """
    col_mapping = {}
    missing_cols = []
    
    for col_name in required_cols:
        actual_col = find_column_ci(df, col_name)
        if actual_col not in df.columns:
            missing_cols.append(col_name)
        else:
            col_mapping[col_name] = actual_col
    
    if missing_cols:
        raise ValueError(
            f"Required column(s) {missing_cols} not found in {df_name}.\n"
            f"Available columns: {', '.join(df.columns.tolist())}\n"
            f"Please ensure your file has the required columns: {', '.join(required_cols)}"
        )
    
    return col_mapping

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_thermo_fisher_csv(df, rt_window_mode='composite_margin'):
    """Creates Thermo Fisher format CSV for inclusion lists
    
    Supports modes:
    - composite_margin/symmetric_margin/fwhm_margin/hybrid etc: t start (min), t stop (min)
    - rt_window: RT Time (min), Window (min)
    - unscheduled: No RT information
    """
    if df.empty:
        return ""
    
    # Base columns always present
    thermo_df_dict = {
        'Compound': df['Compound'],
        'Formula': df.get('Formula', ''),
        'Adduct': df.get('Adduct', ''),
        'm/z': df['m/z'].round(4),
        'z': df['z'].astype(int),  # Force integer for charge
    }
    
    # Add RT columns based on mode
    if rt_window_mode == 'rt_window' and 'RT Time (min)' in df.columns:
        thermo_df_dict['RT Time (min)'] = df['RT Time (min)'].round(2)
        thermo_df_dict['Window (min)'] = df['Window (min)'].round(2)
    elif rt_window_mode != 'unscheduled':
        # Start/End Time mode (all variants)
        if 't start (min)' in df.columns and 't stop (min)' in df.columns:
            thermo_df_dict['t start (min)'] = df['t start (min)'].round(2)
            thermo_df_dict['t stop (min)'] = df['t stop (min)'].round(2)
    
    # If unscheduled mode, no RT columns are added
    
    thermo_df = pd.DataFrame(thermo_df_dict)
    csv_buffer = io.StringIO()
    thermo_df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue()

def create_pdf_report(results_dict):
    """
    Creates a comprehensive PDF report with all figures, match statistics, and settings.
    Returns PDF as bytes.
    """
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib import colors
        from datetime import datetime
    except ImportError:
        return None
    
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#0066cc'),
        spaceAfter=12,
        alignment=TA_CENTER
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#003366'),
        spaceAfter=6,
        spaceBefore=12
    )
    
    # Title
    story.append(Paragraph("PRM Method Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Settings Section
    story.append(Paragraph("Instrument & Method Settings", heading_style))
    
    # Check if multiplex splitting was used
    has_multiplex = False
    pos_df = results_dict.get('pos', pd.DataFrame())
    neg_df = results_dict.get('neg', pd.DataFrame())
    if ('Multiplex_Group' in pos_df.columns and pos_df['Multiplex_Group'].nunique() > 1) or \
       ('Multiplex_Group' in neg_df.columns and neg_df['Multiplex_Group'].nunique() > 1):
        has_multiplex = True
    
    settings_data = [
        ['Setting', 'Value'],
        ['Polarity Mode', results_dict.get('mode', 'N/A')],
        ['Orbitrap Resolution', results_dict.get('resolution', 'N/A')],
        ['Max Injection Time Mode', results_dict.get('it_mode', 'N/A')],
        ['HCD Collision Energies', results_dict.get('hcd_energies', 'N/A')],
        ['XIC m/z Tolerance (ppm)', str(results_dict.get('xic_ppm', 'N/A'))],
        ['Compound Matching Tolerance (ppm)', str(results_dict.get('compound_match_ppm_tolerance', 'N/A'))],
        ['Fragment Dedup Tolerance (ppm)', str(results_dict.get('fragment_dedup_ppm', 'N/A'))],
        ['RT Window Mode', results_dict.get('rt_window_mode', 'N/A')],
        ['RT Margin (min)', str(results_dict.get('rt_margin', 'N/A'))],
        ['Multiplex Splitting', 'YES — Two Inclusion Lists' if has_multiplex else 'NO — Single Inclusion List'],
    ]
    settings_table = Table(settings_data, colWidths=[2.5*inch, 3.5*inch])
    settings_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0066cc')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f0f0')]),
    ]))
    story.append(settings_table)
    story.append(Spacer(1, 0.2*inch))
    
    # Match Summary Section
    story.append(Paragraph("Match Summary", heading_style))
    pos_count = len(results_dict.get('pos', pd.DataFrame()))
    neg_count = len(results_dict.get('neg', pd.DataFrame()))
    skyline_pos_count = len(results_dict.get('skyline_pos', pd.DataFrame()))
    skyline_neg_count = len(results_dict.get('skyline_neg', pd.DataFrame()))
    
    match_data = [
        ['Mode', 'Matched Compounds', 'Skyline Transitions'],
        ['ESI+ (Positive)', str(pos_count), str(skyline_pos_count) if skyline_pos_count > 0 else 'N/A'],
        ['ESI− (Negative)', str(neg_count), str(skyline_neg_count) if skyline_neg_count > 0 else 'N/A'],
        ['Total', str(pos_count + neg_count), str(skyline_pos_count + skyline_neg_count)],
    ]
    match_table = Table(match_data, colWidths=[2.0*inch, 2.0*inch, 2.0*inch])
    match_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#00aa00')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#e8f5e9')]),
    ]))
    story.append(match_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Add Multiplex Group Breakdown if splitting is enabled
    if has_multiplex:
        story.append(Paragraph("Multiplex Group Distribution", heading_style))
        
        multiplex_data = [['Polarity', 'Group 1', 'Group 2', 'Total']]
        
        if not pos_df.empty:
            pos_g1 = len(pos_df[pos_df['Multiplex_Group'] == 1])
            pos_g2 = len(pos_df[pos_df['Multiplex_Group'] == 2])
            multiplex_data.append(['ESI+ (Positive)', str(pos_g1), str(pos_g2), str(pos_count)])
        
        if not neg_df.empty:
            neg_g1 = len(neg_df[neg_df['Multiplex_Group'] == 1])
            neg_g2 = len(neg_df[neg_df['Multiplex_Group'] == 2])
            multiplex_data.append(['ESI− (Negative)', str(neg_g1), str(neg_g2), str(neg_count)])
        
        multiplex_table = Table(multiplex_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        multiplex_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ff6600')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ffe0cc')),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#fff5f0')]),
        ]))
        story.append(multiplex_table)
        story.append(Spacer(1, 0.2*inch))
        story.append(Paragraph("[!] Note: Two separate inclusion lists have been generated. Download Group 1 and Group 2 files separately for two distinct MS injections.", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
    
    # Add figures
    figure_count = 0
    
    # Points Per Peak figures
    for idx, (grp, fig) in enumerate(results_dict.get('fp_pos', [])):
        if figure_count > 0 and figure_count % 2 == 0:
            story.append(PageBreak())
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img = Image(img_buffer, width=4.5*inch, height=2.8*inch)
        story.append(Paragraph(f"ESI+ Points Per Peak (Group {grp})", heading_style))
        story.append(img)
        story.append(Spacer(1, 0.15*inch))
        figure_count += 1
        plt.close(fig)
    
    for idx, (grp, fig) in enumerate(results_dict.get('fp_neg', [])):
        if figure_count > 0 and figure_count % 2 == 0:
            story.append(PageBreak())
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img = Image(img_buffer, width=4.5*inch, height=2.8*inch)
        story.append(Paragraph(f"ESI− Points Per Peak (Group {grp})", heading_style))
        story.append(img)
        story.append(Spacer(1, 0.15*inch))
        figure_count += 1
        plt.close(fig)
    
    # Concurrency figures
    for idx, (grp, fig) in enumerate(results_dict.get('fc_pos', [])):
        if figure_count > 0 and figure_count % 2 == 0:
            story.append(PageBreak())
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img = Image(img_buffer, width=4.5*inch, height=2.8*inch)
        story.append(Paragraph(f"ESI+ Concurrency (Group {grp})", heading_style))
        story.append(img)
        story.append(Spacer(1, 0.15*inch))
        figure_count += 1
        plt.close(fig)
    
    for idx, (grp, fig) in enumerate(results_dict.get('fc_neg', [])):
        if figure_count > 0 and figure_count % 2 == 0:
            story.append(PageBreak())
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img = Image(img_buffer, width=4.5*inch, height=2.8*inch)
        story.append(Paragraph(f"ESI− Concurrency (Group {grp})", heading_style))
        story.append(img)
        story.append(Spacer(1, 0.15*inch))
        figure_count += 1
        plt.close(fig)
    
    # RT Alignment figures
    for grp, fig in results_dict.get('fig_rt_pos', []):
        story.append(PageBreak())
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img = Image(img_buffer, width=6.5*inch, height=4.5*inch)
        story.append(Paragraph(f"ESI+ RT Alignment Window", heading_style))
        story.append(img)
        figure_count += 1
        plt.close(fig)
    
    for grp, fig in results_dict.get('fig_rt_neg', []):
        story.append(PageBreak())
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        img = Image(img_buffer, width=6.5*inch, height=4.5*inch)
        story.append(Paragraph(f"ESI− RT Alignment Window", heading_style))
        story.append(img)
        figure_count += 1
        plt.close(fig)
    
    # Add note about interactive 2D m/z vs RT figures (Plotly)
    if results_dict.get('fig_mz_rt_2d_pos') is not None or results_dict.get('fig_mz_rt_2d_neg') is not None:
        story.append(PageBreak())
        story.append(Paragraph("Interactive Visualizations", heading_style))
        interactive_note = "The following interactive Plotly visualizations are included in the ZIP file:<br/><br/>"
        if results_dict.get('fig_mz_rt_2d_pos') is not None:
            interactive_note += "• <b>MZ-vs-RT_ESI-pos.html</b> - ESI+ m/z vs Retention Time (interactive scatter plot)<br/>"
        if results_dict.get('fig_mz_rt_2d_neg') is not None:
            interactive_note += "• <b>MZ-vs-RT_ESI-neg.html</b> - ESI− m/z vs Retention Time (interactive scatter plot)<br/>"
        interactive_note += "<br/>Open these HTML files in your web browser to explore the data interactively (hover for details, zoom, pan, etc.)."
        story.append(Paragraph(interactive_note, styles['Normal']))
    
    
    # Build PDF
    doc.build(story)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

def create_results_zip(results_dict, rt_window_mode='composite_margin'):
    """Creates a zip file containing all results tables and figures"""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Match Summary
        if 'match_summary' in results_dict and not results_dict['match_summary'].empty:
            csv_buffer = io.StringIO()
            results_dict['match_summary'].to_csv(csv_buffer, index=False)
            zf.writestr('Compound_Match_Summary.csv', csv_buffer.getvalue())
        
        # --- POSITIVE DATA ---
        if not results_dict['pos'].empty:
            has_multi_pos = 'Multiplex_Group' in results_dict['pos'].columns and results_dict['pos']['Multiplex_Group'].nunique() > 1
            if has_multi_pos:
                for grp in [1, 2]:
                    pos_grp = results_dict['pos'][results_dict['pos']['Multiplex_Group'] == grp]
                    if not pos_grp.empty:
                        csv_data = create_thermo_fisher_csv(pos_grp, rt_window_mode)
                        zf.writestr(f'Exploris_Inclusion-List_ESI-pos_Group{grp}.csv', csv_data)
            else:
                csv_data = create_thermo_fisher_csv(results_dict['pos'], rt_window_mode)
                zf.writestr('Exploris_Inclusion-List_ESI-pos.csv', csv_data)
                
        # --- NEGATIVE DATA ---
        if not results_dict['neg'].empty:
            has_multi_neg = 'Multiplex_Group' in results_dict['neg'].columns and results_dict['neg']['Multiplex_Group'].nunique() > 1
            if has_multi_neg:
                for grp in [1, 2]:
                    neg_grp = results_dict['neg'][results_dict['neg']['Multiplex_Group'] == grp]
                    if not neg_grp.empty:
                        csv_data = create_thermo_fisher_csv(neg_grp, rt_window_mode)
                        zf.writestr(f'Exploris_Inclusion-List_ESI-neg_Group{grp}.csv', csv_data)
            else:
                csv_data = create_thermo_fisher_csv(results_dict['neg'], rt_window_mode)
                zf.writestr('Exploris_Inclusion-List_ESI-neg.csv', csv_data)
        
        # --- SKYLINE LISTS ---
        if not results_dict.get('skyline_pos', pd.DataFrame()).empty:
            csv_buffer = io.StringIO()
            results_dict['skyline_pos'].to_csv(csv_buffer, index=False)
            zf.writestr('Skyline_Transition-List_ESI-pos.csv', csv_buffer.getvalue())

        if not results_dict.get('skyline_neg', pd.DataFrame()).empty:
            csv_buffer = io.StringIO()
            results_dict['skyline_neg'].to_csv(csv_buffer, index=False)
            zf.writestr('Skyline_Transition-List_ESI-neg.csv', csv_buffer.getvalue())

        # --- SKYLINE UNMATCHED COMPOUNDS ---
        if not results_dict.get('skyline_unmatched_pos', pd.DataFrame()).empty:
            csv_buffer = io.StringIO()
            results_dict['skyline_unmatched_pos'].to_csv(csv_buffer, index=False)
            zf.writestr('Skyline_Unmatched-Compounds_ESI-pos.csv', csv_buffer.getvalue())

        if not results_dict.get('skyline_unmatched_neg', pd.DataFrame()).empty:
            csv_buffer = io.StringIO()
            results_dict['skyline_unmatched_neg'].to_csv(csv_buffer, index=False)
            zf.writestr('Skyline_Unmatched-Compounds_ESI-neg.csv', csv_buffer.getvalue())

        # --- FIGURES (POS) ---
        # Filenames always carry the multiplex group number (Group1 / Group2),
        # even when only one group exists, for a fully consistent naming scheme.
        for grp, fig in results_dict.get('fp_pos', []):
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='svg', bbox_inches='tight')
            img_buffer.seek(0)
            zf.writestr(f'Points-per-Peak_ESI-pos_Group{grp}.svg', img_buffer.getvalue())
        
        for grp, fig in results_dict.get('fc_pos', []):
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='svg', bbox_inches='tight')
            img_buffer.seek(0)
            zf.writestr(f'Concurrency_ESI-pos_Group{grp}.svg', img_buffer.getvalue())
            
        for grp, fig in results_dict.get('fig_rt_pos', []):
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='svg', bbox_inches='tight')
            img_buffer.seek(0)
            zf.writestr(f'RT-alignement_ESI-pos_Group{grp}.svg', img_buffer.getvalue())
            
        # --- FIGURES (NEG) ---
        for grp, fig in results_dict.get('fp_neg', []):
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='svg', bbox_inches='tight')
            img_buffer.seek(0)
            zf.writestr(f'Points-per-Peak_ESI-neg_Group{grp}.svg', img_buffer.getvalue())
        
        for grp, fig in results_dict.get('fc_neg', []):
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='svg', bbox_inches='tight')
            img_buffer.seek(0)
            zf.writestr(f'Concurrency_ESI-neg_Group{grp}.svg', img_buffer.getvalue())
            
        for grp, fig in results_dict.get('fig_rt_neg', []):
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='svg', bbox_inches='tight')
            img_buffer.seek(0)
            zf.writestr(f'RT-alignement_ESI-neg_Group{grp}.svg', img_buffer.getvalue())
        
        # --- PLOTLY (MZ vs RT) ---
        if results_dict.get('fig_mz_rt_2d_pos') is not None:
            html_str = results_dict['fig_mz_rt_2d_pos'].to_html(include_plotlyjs='cdn')
            zf.writestr('MZ-vs-RT_ESI-pos.html', html_str)
        
        if results_dict.get('fig_mz_rt_2d_neg') is not None:
            html_str = results_dict['fig_mz_rt_2d_neg'].to_html(include_plotlyjs='cdn')
            zf.writestr('MZ-vs-RT_ESI-neg.html', html_str)
        
        # --- XIC FIGURES (paginated: one file per page) ---
        # Handle both old format (single Figure, from a cached pre-pagination run)
        # and new format (list of Figures, one per page)
        fig_xic_pos_raw = results_dict.get('fig_xic_pos')
        fig_xic_pos_list = fig_xic_pos_raw if isinstance(fig_xic_pos_raw, list) else ([fig_xic_pos_raw] if fig_xic_pos_raw else [])
        fig_xic_neg_raw = results_dict.get('fig_xic_neg')
        fig_xic_neg_list = fig_xic_neg_raw if isinstance(fig_xic_neg_raw, list) else ([fig_xic_neg_raw] if fig_xic_neg_raw else [])

        for page_idx, fig in enumerate(fig_xic_pos_list, start=1):
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='svg', bbox_inches='tight')
            img_buffer.seek(0)
            zf.writestr(f'XIC_ESI-pos_Page{page_idx}.svg', img_buffer.getvalue())

        for page_idx, fig in enumerate(fig_xic_neg_list, start=1):
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='svg', bbox_inches='tight')
            img_buffer.seek(0)
            zf.writestr(f'XIC_ESI-neg_Page{page_idx}.svg', img_buffer.getvalue())


        # --- PDF REPORT ---
        pdf_data = create_pdf_report(results_dict)
        if pdf_data:
            zf.writestr('PRM_Method_Report.pdf', pdf_data)
    
    zip_buffer.seek(0)
    return zip_buffer

def clean_cas(cas_string):
    if pd.isna(cas_string): return ""
    return str(cas_string).replace("-", "").replace(" ", "").strip()

def get_core_name(name):
    name = str(name).lower()
    name = re.sub(r'^.*?:[a-z0-9_-]+\s+', '', name)
    name = re.sub(r'^[0-9]+_', '', name)
    name = name.split('|')[0]
    name = re.sub(r'\b(hcl|na|k|br|cl)\b', '', name)
    name = re.sub(r'[^a-z0-9]', '', name)
    return name

_ADDUCT_SUPERSCRIPT_MAP = str.maketrans({
    '⁺': '+', '⁻': '-', '⁰': '0', '¹': '1', '²': '2', '³': '3',
    '⁴': '4', '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9',
})

def format_adduct(raw_adduct, polarity):
    """
    Normalizes any adduct string into the strict '[M+H]' style bracket format
    (no charge symbol left inside or outside the brackets), e.g.:
        'M+H'        -> '[M+H]'
        '[M+H]+'     -> '[M+H]'
        'M+H]+'      -> '[M+H]'
        '[M+2H]2+'   -> '[M+2H]'
        '[M+3H]3+'   -> '[M+3H]'
        '[M-H]-'     -> '[M-H]'
        'M+H+1'      -> '[M+H]'  (sign-then-digit charge notation)
        '(M+H)+'     -> '[M+H]'  (parentheses instead of brackets)
        '' / NaN     -> '[M+H]' or '[M-H]' (polarity default)
    """
    default = 'M+H' if polarity == 'Positive' else 'M-H'

    if raw_adduct is None or (isinstance(raw_adduct, float) and pd.isna(raw_adduct)):
        s = default
    else:
        s = str(raw_adduct).strip()

    if s.lower() in ['nan', 'none', 'null', '']:
        s = default

    # Normalize unicode superscript charge notation (e.g. '⁺', '²⁺') to ASCII
    s = s.translate(_ADDUCT_SUPERSCRIPT_MAP)

    # Strip any existing brackets/parentheses
    s = s.replace('[', '').replace(']', '').replace('(', '').replace(')', '').strip()

    # Strip trailing charge notation in any order/repetition of digits and +/-
    # (e.g. '+', '-', '2+', '2-', '++', '+1', '3+'), as long as it doesn't
    # consume the whole string.
    s = re.sub(r'[\d]*[+-][\d+-]*\s*$', '', s).strip()

    # If nothing (or no 'M') is left, fall back to the polarity default
    if not s or 'M' not in s.upper():
        s = default

    return f'[{s}]'

def calculate_scan_time(resolution, it_mode, custom_it, peak_width_sec, desired_pts, concurrent_targets):
    """
    Calculates the true cycle time based on Thermo Exploris parallelized architecture.
    """
    # Exact transient times for Exploris 480 resolutions (in ms)
    transients = {
        7500: 8,
        11250: 12,
        15000: 16,
        22500: 24,
        30000: 32,
        45000: 48,
        60000: 64,
        75000: 80,
        90000: 96,
        120000: 128,
        180000: 192,
        240000: 256,
        480000: 512
    }
    
    # Safely match to exact resolution, fallback to closest if somehow unlisted
    res_key = min(transients.keys(), key=lambda k: abs(k - resolution))
    t_trans = transients[res_key]
    overhead_ms = 8 
    
    if it_mode == "Auto":
        max_it = t_trans
    elif it_mode == "Custom":
        max_it = custom_it
    elif it_mode == "Dynamic":
        if concurrent_targets > 0:
            max_cycle_time_ms = (peak_width_sec * 1000) / desired_pts
            max_scan_time_ms = max_cycle_time_ms / concurrent_targets
            max_it = max_scan_time_ms - overhead_ms
        else:
            max_it = t_trans
    else:
        max_it = t_trans
        
    return max(t_trans, max_it) + overhead_ms

def parse_mgf_spectrum_list(mgf_file_content):
    """
    Parse MGF file content into list of spectrum dicts.
    Handles PEPMASS, RTINSECONDS, CHARGE, MSLEVEL, and fragment peaks.
    Only extracts MS/MS spectra directly from GNPS (Consensus MS/MS).
    
    Returns: list of dicts with keys: pepmass, rtinseconds, charge, mslevel, fragments
    """
    spectra = []
    current_spectrum = None
    in_ions = False
    num_peaks = 0
    peaks_collected = 0
    
    for line in mgf_file_content.decode('utf-8', errors='ignore').split('\n'):
        line = line.strip()
        
        if line == "BEGIN IONS":
            in_ions = True
            current_spectrum = {
                'pepmass': None,
                'rtinseconds': None,
                'charge': None,
                'mslevel': None,
                'fragments': []
            }
            num_peaks = 0
            peaks_collected = 0
            continue
        
        if line == "END IONS":
            # Only keep spectra with a pepmass (GNPS MS/MS consensus MGFs)
            if current_spectrum and current_spectrum['pepmass'] is not None:
                spectra.append(current_spectrum)
            in_ions = False
            current_spectrum = None
            continue
        
        if not in_ions or not line or '=' not in line:
            # Parse fragment line (mz intensity). Some MGF exports omit the
            # "NUM PEAKS=" header entirely, so peak lines are collected as
            # long as they parse as two numeric tokens rather than being
            # gated on a declared peak count.
            if in_ions and current_spectrum and line:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        mz = float(parts[0])
                        intensity = float(parts[1])
                        current_spectrum['fragments'].append((mz, intensity))
                        peaks_collected += 1
                    except (ValueError, IndexError):
                        pass
            continue
        
        # Parse KEY=VALUE lines
        try:
            key, value = line.split('=', 1)
            key = key.strip().lower()
            value = value.strip()
            
            if key == 'pepmass':
                # Handle "123.456 789" format (mass and intensity)
                pepmass_parts = value.split()
                current_spectrum['pepmass'] = float(pepmass_parts[0])
            
            elif key == 'rtinseconds':
                current_spectrum['rtinseconds'] = float(value)
            
            elif key == 'charge':
                # Handle "1+" or "+1" format
                charge_str = value.replace('+', '').replace('-', '')
                current_spectrum['charge'] = int(charge_str)
            
            elif key == 'mslevel':
                current_spectrum['mslevel'] = int(value)
            
            elif key == 'num peaks':
                num_peaks = int(value)
        except (ValueError, IndexError):
            pass
    
    return spectra


def extract_skyline_transitions_from_mgf(mgf_files, compound_df, polarity_mode, fragment_dedup_ppm=5, compound_match_ppm_tolerance=10):
    """
    Extracts fragment ions from MGF files to create Skyline format transitions.
    Returns a tuple (result_df, unmatched_df):
        - result_df: Skyline-compatible transition data.
        - unmatched_df: one row per target compound with zero transitions,
          with a best-effort reason (no m/z match, RT window mismatch,
          no usable fragments, or lost to a closer competing compound).

    Format: Molecule List Name | Precursor Name | Precursor Formula | Precursor Adduct |
            Precursor m/z | Product m/z | Precursor Charge | Product Charge | Explicit Retention Time

    WORKFLOW:
    1. Parse MGF file to extract PEPMASS (precursor m/z) and fragments
    2. Match PEPMASS to compounds using ±compound_match_ppm_tolerance ppm m/z tolerance
    3. Extract the 2 most abundant fragments (by intensity)
    4. Create Skyline transition table with precursor and fragment m/z pairs
    """
    if not mgf_files or compound_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    skyline_data = []
    total_spectra = 0
    matched_spectra = 0
    matched_precursor_set = set()  # Track which precursors from compound_df were matched
    spectra_records = []  # Per-spectrum diagnostics, used to explain unmatched compounds

    st.write(f"📂 **Skyline Input:** {len(mgf_files)} MGF file(s), {len(compound_df)} compounds to match")
    st.write(f"📊 **Compound m/z range:** {compound_df['m/z'].min():.2f} - {compound_df['m/z'].max():.2f}")
    st.write(f"⚙️ **Compound Matching Tolerance:** ±{compound_match_ppm_tolerance} ppm")
    st.write(f"⚙️ **Fragment Dedup Threshold:** {fragment_dedup_ppm} ppm")
    
    # Create lookup tables for compound matching
    matched_compounds = {row['Compound']: row for _, row in compound_df.iterrows()}
    
    for file_idx, mgf_file in enumerate(mgf_files):
        try:
            st.write(f"📖 Reading MGF file {file_idx+1}...")
            # Handle Streamlit UploadedFile objects
            try:
                mgf_file.seek(0)
                file_content = mgf_file.read()
            except:
                # Fallback for file-like objects
                try:
                    file_content = mgf_file.getvalue()
                except:
                    file_content = None
            
            if file_content is None:
                st.warning(f"Could not read MGF file content")
                continue
            
            # Debug: show file size
            file_size_mb = len(file_content) / (1024*1024)
            st.write(f"   📊 File size: {file_size_mb:.1f} MB")
            
            # Parse MGF using improved parser
            try:
                spectra = parse_mgf_spectrum_list(file_content)
            except Exception as e:
                st.error(f"   ❌ Error parsing MGF: {str(e)}")
                continue
            
            file_spectra_count = len(spectra)
            total_spectra += file_spectra_count
            st.write(f"   ✓ Found {file_spectra_count} spectra in this file")
            
            for spectrum in spectra:
                # Get precursor m/z from PEPMASS
                precursor_mz = spectrum.get('pepmass')
                rtinseconds = spectrum.get('rtinseconds')

                # Convert RT from seconds to minutes if available
                rt_min = None
                if rtinseconds:
                    try:
                        rt_min = float(rtinseconds) / 60.0
                    except:
                        rt_min = None

                # Get fragment m/z and intensities
                fragments = spectrum.get('fragments', [])

                # Record every spectrum with a precursor m/z (even if it has no
                # peaks or doesn't end up matching) so we can explain later why
                # any given target compound came up unmatched.
                spectrum_record = None
                if precursor_mz is not None:
                    spectrum_record = {
                        'mz': precursor_mz,
                        'rt_min': rt_min,
                        'has_ms2_peaks': len(fragments) > 0,
                        'assigned_compound': None,
                        'usable_fragments': False,
                    }
                    spectra_records.append(spectrum_record)

                if precursor_mz is not None and len(fragments) > 0:
                    # Match compound by m/z (± compound_match_ppm_tolerance)
                    ppm_tol = compound_match_ppm_tolerance
                    matched_compound = None
                    mz_diff_best = float('inf')

                    for compound_name, compound_row in matched_compounds.items():
                        comp_mz = compound_row['m/z']
                        mz_diff_ppm = abs(precursor_mz - comp_mz) / comp_mz * 1e6

                        # Check m/z match and optionally RT window if RT is available
                        if mz_diff_ppm <= ppm_tol and mz_diff_ppm < mz_diff_best:
                            if rt_min is not None:
                                rt_start = compound_row.get('t start (min)', 0)
                                rt_stop = compound_row.get('t stop (min)', 16)
                                if rt_start <= rt_min <= rt_stop:
                                    matched_compound = compound_name
                                    mz_diff_best = mz_diff_ppm
                            else:
                                # If no RT available, use closest m/z match
                                matched_compound = compound_name
                                mz_diff_best = mz_diff_ppm

                    if matched_compound:
                        spectrum_record['assigned_compound'] = matched_compound
                        matched_spectra += 1
                        matched_precursor_set.add(matched_compound)  # Track this precursor as matched
                        compound_row = matched_compounds[matched_compound]
                        
                        # EXTRACT TOP 2 FRAGMENTS, EXCLUDING PRECURSOR m/z
                        # Sort fragments by intensity (descending)
                        fragments_with_idx = [(idx, mz, intensity) 
                                            for idx, (mz, intensity) in enumerate(fragments)]
                        fragments_sorted = sorted(fragments_with_idx, 
                                               key=lambda x: x[2], reverse=True)
                        
                        # Filter fragments:
                        # 1. Exclude precursor m/z (within tolerance)
                        
                        precursor_mz_tolerance = 0.01
                        valid_fragments = []
                        for (_, mz, intensity) in fragments_sorted:
                            # Skip if fragment m/z is within tolerance of precursor m/z
                            if abs(mz - precursor_mz) <= precursor_mz_tolerance:
                                continue
                            
                            valid_fragments.append((mz, intensity))
                        
                        # ALGORITHM: Take top 2 fragments by intensity, round to 4 decimals
                        # Then check if those 2 are within 5 ppm (or custom threshold)
                        # If so, keep only the most abundant one
                        top_2_fragments = valid_fragments[:2]  # Take top 2 by intensity
                        spectrum_record['usable_fragments'] = len(top_2_fragments) > 0

                        if len(top_2_fragments) == 0:
                            continue  # No valid fragments
                        elif len(top_2_fragments) == 1:
                            # Only 1 fragment, round and use it
                            final_fragments = [(round(top_2_fragments[0][0], 4), top_2_fragments[0][1])]
                        else:
                            # 2 fragments: check if within threshold ppm
                            mz1_rounded = round(top_2_fragments[0][0], 4)
                            mz2_rounded = round(top_2_fragments[1][0], 4)
                            int1 = top_2_fragments[0][1]
                            int2 = top_2_fragments[1][1]
                            
                            # Calculate ppm difference between the 2 fragments
                            ppm_diff = abs(mz2_rounded - mz1_rounded) / max(mz1_rounded, mz2_rounded) * 1e6
                            
                            if ppm_diff <= fragment_dedup_ppm:
                                # Fragments too similar, keep only the most abundant
                                if int1 >= int2:
                                    final_fragments = [(mz1_rounded, int1)]
                                else:
                                    final_fragments = [(mz2_rounded, int2)]
                            else:
                                # Fragments are different enough, keep both
                                final_fragments = [(mz1_rounded, int1), (mz2_rounded, int2)]
                        
                        # Process if we have 1-2 unique fragments
                        if len(final_fragments) == 0 or len(final_fragments) > 2:
                            continue
                        
                        # Determine precursor charge (handle NaN values)
                        z_value = compound_row.get('z', 1 if polarity_mode == 'Positive' else -1)
                        try:
                            if pd.isna(z_value):
                                precursor_charge = 1 if polarity_mode == 'Positive' else -1
                            else:
                                precursor_charge = int(z_value)
                        except:
                            precursor_charge = int(z_value) if z_value is not None else (1 if polarity_mode == 'Positive' else -1)
                        
                        # Format adduct string: must be in brackets [X] with nothing outside
                        adduct_raw = compound_row.get('Adduct', 'M+H' if polarity_mode == 'Positive' else 'M-H')
                        adduct = format_adduct(adduct_raw, polarity_mode)
                        
                        # Get retention time
                        rt = compound_row.get('Peak_RT', '')
                        
                        # Create transition for each final deduplicated fragment
                        for rank, (frag_mz, frag_intensity) in enumerate(final_fragments, 1):
                            skyline_data.append({
                                'Molecule List Name': '',
                                'Precursor Name': matched_compound,
                                'Precursor Formula': compound_row.get('Formula', ''),
                                'Precursor Adduct': adduct,
                                'Precursor m/z': round(precursor_mz, 4),
                                'Product m/z': frag_mz,  # Already rounded to 4 decimals
                                'Precursor Charge': precursor_charge,
                                'Product Charge': precursor_charge,  # Match precursor charge (±1 based on polarity)
                                'Explicit Retention Time': round(rt, 2) if rt else '',
                                '_intensity': frag_intensity  # Track intensity for final filtering
                            })
        except Exception as e:
            st.error(f"❌ Error reading MGF file {file_idx+1}: {str(e)}")
            import traceback
            st.text(traceback.format_exc())
            continue
    
    # Summary debug output
    st.write(f"✅ **Skyline Processing Complete:**")
    st.write(f"   • Total spectra read: {total_spectra}")
    st.write(f"   • Spectra matched to compounds: {matched_spectra}")
    st.write(f"   • Transitions (before dedup): {len(skyline_data)}")
    
    # DEDUPLICATION: Multiple spectra can match the same compound.
    # 1. Remove exact duplicates (same Precursor Name and Product m/z)
    # 2. For each precursor, remove fragments within fragment_dedup_ppm threshold
    
    # Step 1: Remove exact duplicates
    seen = set()
    deduplicated_data = []
    for row in skyline_data:
        key = (row['Precursor Name'], round(row['Product m/z'], 4))
        if key not in seen:
            seen.add(key)
            deduplicated_data.append(row)
    
    # Step 2: Within each precursor, deduplicate fragments within ppm threshold
    # Group by precursor name
    precursor_groups = {}
    for row in deduplicated_data:
        precursor = row['Precursor Name']
        if precursor not in precursor_groups:
            precursor_groups[precursor] = []
        precursor_groups[precursor].append(row)
    
    # For each precursor, remove fragments within threshold ppm
    final_deduplicated = []
    for precursor, fragment_list in precursor_groups.items():
        # Sort by intensity (descending) to keep most abundant
        fragment_list_sorted = sorted(fragment_list, key=lambda x: float(x.get('Product Charge', 1)), reverse=False)  # Placeholder sort
        fragment_list_sorted = sorted(fragment_list_sorted, key=lambda x: 1, reverse=True)  # Reset
        
        kept_fragments = []
        for current_row in fragment_list:
            current_mz = float(current_row['Product m/z'])
            is_duplicate = False
            
            # Check against already kept fragments
            for kept_row in kept_fragments:
                kept_mz = float(kept_row['Product m/z'])
                ppm_diff = abs(current_mz - kept_mz) / kept_mz * 1e6
                
                if ppm_diff <= fragment_dedup_ppm:
                    # Found a similar fragment within threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                kept_fragments.append(current_row)
        
        # FINAL STEP: Keep only top 2 most abundant fragments per precursor
        # Sort by intensity (descending)
        kept_fragments_sorted = sorted(kept_fragments, key=lambda x: float(x.get('_intensity', 0)), reverse=True)
        kept_fragments_top2 = kept_fragments_sorted[:2]  # Keep only top 2 most abundant
        
        final_deduplicated.extend(kept_fragments_top2)
    
    result_df = pd.DataFrame(final_deduplicated)
    
    # Remove internal tracking columns
    if '_intensity' in result_df.columns:
        result_df = result_df.drop(columns=['_intensity'])
    
    # Count unique compounds in final output
    unique_compounds = result_df['Precursor Name'].nunique() if not result_df.empty else 0
    compounds_with_output = set(result_df['Precursor Name'].unique()) if not result_df.empty else set()
    spectrum_dedup_count = len(skyline_data) - len(final_deduplicated)

    st.write(f"   • Transitions (after dedup): {len(final_deduplicated)}")
    st.write(f"   • Spectrum-level duplicates removed: {spectrum_dedup_count}")

    # Show badge with matched precursors from compound list
    precursors_in_list = len(compound_df)
    precursors_matched = len(matched_precursor_set)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Unique Compounds in Output", unique_compounds)
    with col2:
        st.metric("Precursors from List Matched", f"{precursors_matched}/{precursors_in_list}")

    if total_spectra == 0:
        st.warning("⚠️ **No spectra found in MGF files!** Check that MGF files are not empty and are in proper format.")
    elif matched_spectra == 0:
        st.warning(f"⚠️ **No spectra matched to compounds!** m/z values in MGF may not match your target compounds (tolerance: ±{compound_match_ppm_tolerance} ppm).")

    # ------------------------------------------------------------------
    # DIAGNOSE UNMATCHED COMPOUNDS
    # Any target compound that produced zero transitions gets a row here,
    # with a best-effort reason based on the recorded per-spectrum data.
    # ------------------------------------------------------------------
    unmatched_rows = []
    for compound_name, compound_row in matched_compounds.items():
        if compound_name in compounds_with_output:
            continue

        comp_mz = compound_row['m/z']
        rt_start = compound_row.get('t start (min)', None)
        rt_stop = compound_row.get('t stop (min)', None)
        has_rt_window = (rt_start is not None and rt_stop is not None and
                          not pd.isna(rt_start) and not pd.isna(rt_stop))

        def _ppm(mz):
            return abs(mz - comp_mz) / comp_mz * 1e6

        candidates = [(_ppm(rec['mz']), rec) for rec in spectra_records
                      if _ppm(rec['mz']) <= compound_match_ppm_tolerance]
        candidates.sort(key=lambda x: x[0])

        if not candidates:
            if spectra_records:
                closest_d, closest_rec = min(((_ppm(r['mz']), r) for r in spectra_records), key=lambda x: x[0])
                reason = (f"No MGF precursor within ±{compound_match_ppm_tolerance} ppm "
                          f"(closest: m/z {closest_rec['mz']:.4f}, Δ{closest_d:.1f} ppm)")
            else:
                reason = "No MGF spectra available to match against"
        else:
            taken_by_other = [(d, r) for d, r in candidates if r['assigned_compound'] not in (None, compound_name)]
            rt_failed = [(d, r) for d, r in candidates
                         if has_rt_window and r['rt_min'] is not None and not (rt_start <= r['rt_min'] <= rt_stop)]
            no_peaks = [(d, r) for d, r in candidates if not r['has_ms2_peaks']]
            no_usable = [(d, r) for d, r in candidates if r['has_ms2_peaks'] and not r['usable_fragments']]

            if len(taken_by_other) == len(candidates):
                d, r = taken_by_other[0]
                reason = (f"Closest MGF precursor (Δ{d:.1f} ppm) was instead assigned to a "
                          f"closer target compound ('{r['assigned_compound']}')")
            elif has_rt_window and len(rt_failed) == len(candidates):
                d, r = rt_failed[0]
                reason = (f"Precursor m/z matched (Δ{d:.1f} ppm) but retention time outside target "
                          f"window [{rt_start:.2f}-{rt_stop:.2f}] min (spectrum RT = {r['rt_min']:.2f} min)")
            elif len(no_peaks) == len(candidates):
                d, r = no_peaks[0]
                reason = f"Precursor m/z matched (Δ{d:.1f} ppm) but the MS2 spectrum had no fragment peaks at all"
            elif len(no_usable) == len(candidates):
                d, r = no_usable[0]
                reason = f"Precursor m/z matched (Δ{d:.1f} ppm) but no fragment ions remained after excluding the precursor peak"
            else:
                d, r = candidates[0]
                reason = (f"Precursor m/z matched (Δ{d:.1f} ppm) but was rejected for a mix of reasons "
                          f"(RT window / competing compound / fragment filtering)")

        unmatched_rows.append({
            'Compound': compound_name,
            'Target m/z': round(comp_mz, 4),
            'RT Window (min)': f"{rt_start:.2f}-{rt_stop:.2f}" if has_rt_window else 'N/A',
            'Reason Unmatched': reason,
        })

    unmatched_df = pd.DataFrame(unmatched_rows)

    if not unmatched_df.empty:
        st.warning(f"⚠️ **{len(unmatched_df)} compound(s) from the target list produced no Skyline transitions.** See the table below for the reason for each.")
        st.dataframe(unmatched_df, width='stretch', use_container_width=True)

    return result_df, unmatched_df

def create_compound_match_summary(all_targets_df, matched_compounds_set, polarity_mode, col_targets_compound=TARGETS_COMPOUND_COL):
    """
    Creates a summary table of all target compounds with matched/unmatched status.
    """
    summary_data = []
    
    for _, target in all_targets_df.iterrows():
        compound_name = target.get('Compound', target.get(col_targets_compound, ''))
        
        if compound_name in matched_compounds_set:
            status = "✓ Matched"
            # Find the matched row to get m/z and other details
            matched_row = next((r for r in matched_compounds_set 
                              if r[0] == compound_name), None)
        else:
            status = "✗ Not Matched"
        
        summary_data.append({
            'Compound': compound_name,
            'Match_Status': status,
            'Polarity': polarity_mode,
            'CAS': target.get('CAS', ''),
            'Formula': target.get(TARGETS_FORMULA_COL, ''),
            'SMILES': target.get(TARGETS_SMILES_COL, '')
        })
    
    return pd.DataFrame(summary_data)

def calculate_rt_window(peak_rt, composite_rt_min, composite_rt_max, rt_window_mode,
                         rt_margin_min=1.0, rt_margin_pct=None, expected_peak_width_min=0.167, fwhm=0.0):
    """
    Calculate retention time window based on selected mode.
    
    Modes:
    - 'symmetric_margin': Option 1a — (peak_rt - margin_min, peak_rt + margin_min)
    - 'symmetric_margin_pct': Option 1b — (peak_rt - peak_rt*margin_pct%, peak_rt + peak_rt*margin_pct%)
    - 'composite_margin': Option 2a — (rt_start - margin_min, rt_stop + margin_min)
    - 'composite_margin_pct': Option 2b — (rt_start - rt_start*margin_pct%, rt_stop + rt_stop*margin_pct%)
    - 'fwhm_margin': Option 3a — ((peak_rt - FWHM) - margin_min, (peak_rt + FWHM) + margin_min)
    - 'fwhm_margin_pct': Option 3b — Base ± (Base*margin_pct%) where Base is RT ± FWHM
    - 'hybrid': Option 4 — Use 1a if peak_width <= expected_peak_width_min, else use 2b
    - 'rt_window': Return tuple (rt, window_size) for Retention Time Window mode
    - 'unscheduled': Return (None, None) for Unscheduled mode
    
    Returns: (t_start, t_stop, exp_peak_width_min)
    """
    exp_peak_width_min = max(2.0, composite_rt_max - composite_rt_min)
    
    if rt_window_mode == 'unscheduled':
        return None, None, exp_peak_width_min
    
    if rt_window_mode == 'rt_window':
        # Return peak RT and window size
        return peak_rt, rt_margin_min, exp_peak_width_min
    
    if rt_window_mode == 'symmetric_margin':
        # Option 1a: Symmetric around peak RT with fixed margin (min)
        expanded_start = max(0, peak_rt - rt_margin_min)
        expanded_stop = peak_rt + rt_margin_min
        return expanded_start, expanded_stop, exp_peak_width_min
    
    if rt_window_mode == 'symmetric_margin_pct':
        # Option 1b: Symmetric around peak RT with % margin
        margin_amount = peak_rt * (rt_margin_pct / 100.0) if rt_margin_pct else peak_rt * 0.1
        expanded_start = max(0, peak_rt - margin_amount)
        expanded_stop = peak_rt + margin_amount
        return expanded_start, expanded_stop, exp_peak_width_min
    
    if rt_window_mode == 'composite_margin':
        # Option 2a: Boundaries ± margin (fixed minutes)
        expanded_start = max(0, composite_rt_min - rt_margin_min)
        expanded_stop = composite_rt_max + rt_margin_min
        return expanded_start, expanded_stop, exp_peak_width_min
    
    if rt_window_mode == 'composite_margin_pct':
        # Option 2b: Boundaries ± margin (% of each boundary)
        margin_from_min = composite_rt_min * (rt_margin_pct / 100.0) if rt_margin_pct else composite_rt_min * 0.1
        margin_from_max = composite_rt_max * (rt_margin_pct / 100.0) if rt_margin_pct else composite_rt_max * 0.1
        expanded_start = max(0, composite_rt_min - margin_from_min)
        expanded_stop = composite_rt_max + margin_from_max
        return expanded_start, expanded_stop, exp_peak_width_min
        
    if rt_window_mode == 'fwhm_margin':
        # Option 3a: FWHM around peak RT ± margin (fixed minutes)
        base_start = peak_rt - fwhm
        base_stop = peak_rt + fwhm
        expanded_start = max(0, base_start - rt_margin_min)
        expanded_stop = base_stop + rt_margin_min
        return expanded_start, expanded_stop, exp_peak_width_min
        
    if rt_window_mode == 'fwhm_margin_pct':
        # Option 3b: FWHM around peak RT ± margin (% of boundaries)
        base_start = max(0, peak_rt - fwhm)
        base_stop = peak_rt + fwhm
        margin_from_start = base_start * (rt_margin_pct / 100.0) if rt_margin_pct else base_start * 0.1
        margin_from_stop = base_stop * (rt_margin_pct / 100.0) if rt_margin_pct else base_stop * 0.1
        expanded_start = max(0, base_start - margin_from_start)
        expanded_stop = base_stop + margin_from_stop
        return expanded_start, expanded_stop, exp_peak_width_min
    
    if rt_window_mode == 'hybrid':
        # Option 4: Hybrid - use 1a for narrow peaks (<=expected), 2b for wide peaks (>expected)
        range_width = composite_rt_max - composite_rt_min
        if range_width <= expected_peak_width_min:
            # Peak width <= expected: use Option 1a (symmetric with fixed margin)
            expanded_start = max(0, peak_rt - rt_margin_min)
            expanded_stop = peak_rt + rt_margin_min
        else:
            # Peak width > expected: use Option 2b (boundaries with % margin)
            margin_from_min = composite_rt_min * (rt_margin_pct / 100.0) if rt_margin_pct else composite_rt_min * 0.1
            margin_from_max = composite_rt_max * (rt_margin_pct / 100.0) if rt_margin_pct else composite_rt_max * 0.1
            expanded_start = max(0, composite_rt_min - margin_from_min)
            expanded_stop = composite_rt_max + margin_from_max
        return expanded_start, expanded_stop, exp_peak_width_min
    
    # Default: composite_margin
    expanded_start = max(0, composite_rt_min - rt_margin_min)
    expanded_stop = composite_rt_max + rt_margin_min
    return expanded_start, expanded_stop, exp_peak_width_min

def process_polarity(gnps_df, mzmine_df, polarity, targets_df, hcd_energies,
                     rt_window_mode="composite_margin", rt_margin_min=1.0, rt_margin_pct=None, 
                     expected_peak_width_sec=10,
                     col_gnps_compound=GNPS_COMPOUND_COL, col_gnps_scan=GNPS_SCAN_COL,
                     col_gnps_cas=GNPS_CAS_COL, col_gnps_smiles=GNPS_SMILES_COL, col_gnps_formula=GNPS_FORMULA_COL,
                     col_gnps_adduct=GNPS_ADDUCT_COL,
                     col_mzmine_scan=MZMINE_SCAN_COL, col_mzmine_mz=MZMINE_MZ_COL, col_mzmine_rt=MZMINE_RT_COL,
                     col_mzmine_rt_start=MZMINE_RT_START_COL, col_mzmine_rt_end=MZMINE_RT_END_COL,
                     col_mzmine_height=MZMINE_HEIGHT_COL, col_mzmine_charge=MZMINE_CHARGE_COL,
                     col_targets_compound=TARGETS_COMPOUND_COL, col_targets_cas=TARGETS_CAS_COL,
                     col_targets_smiles=TARGETS_SMILES_COL, col_targets_formula=TARGETS_FORMULA_COL):
    gnps_df = gnps_df.copy()
    
    # Validate and map GNPS columns (case-insensitive)
    required_gnps_cols = [col_gnps_compound, col_gnps_scan]
    gnps_col_map = validate_columns_ci(gnps_df, required_gnps_cols, "GNPS data")
    
    # Map to actual column names in the dataframe
    col_gnps_compound = gnps_col_map[col_gnps_compound]
    col_gnps_scan = gnps_col_map[col_gnps_scan]
    col_gnps_cas = find_column_ci(gnps_df, col_gnps_cas)
    col_gnps_smiles = find_column_ci(gnps_df, col_gnps_smiles)
    col_gnps_formula = find_column_ci(gnps_df, col_gnps_formula)
    # Map adduct column - use temp var to avoid shadowing parameter
    _col_gnps_adduct = find_column_ci(gnps_df, col_gnps_adduct)
    
    # Validate and map targets columns (case-insensitive)
    required_targets_cols = [col_targets_compound]
    targets_col_map = validate_columns_ci(targets_df, required_targets_cols, "Targets data")
    col_targets_compound = targets_col_map[col_targets_compound]
    col_targets_cas = find_column_ci(targets_df, col_targets_cas)
    col_targets_smiles = find_column_ci(targets_df, col_targets_smiles)
    col_targets_formula = find_column_ci(targets_df, col_targets_formula)
    
    
    gnps_df['clean_cas'] = gnps_df.get(col_gnps_cas, pd.Series(dtype=str)).apply(clean_cas)
    gnps_df['clean_smiles'] = gnps_df.get(col_gnps_smiles, pd.Series(dtype=str)).apply(lambda s: "" if pd.isna(s) else str(s).strip())
    gnps_df['clean_formula'] = gnps_df.get(col_gnps_formula, pd.Series(dtype=str)).apply(lambda f: "" if pd.isna(f) else str(f).replace(" ", "").lower())

    targets_df['clean_cas'] = targets_df.get(col_targets_cas, pd.Series(dtype=str)).apply(clean_cas)
    targets_df['clean_smiles'] = targets_df.get(col_targets_smiles, pd.Series(dtype=str)).apply(lambda s: "" if pd.isna(s) else str(s).strip())
    targets_df['clean_formula'] = targets_df.get(col_targets_formula, pd.Series(dtype=str)).apply(lambda f: "" if pd.isna(f) else str(f).replace(" ", "").lower())

    matched_rows = []
    for _, target in targets_df.iterrows():
        t_cas     = target.get('clean_cas', "")
        t_name    = target[col_targets_compound]
        t_core    = get_core_name(t_name)
        t_smiles  = str(target.get(col_targets_smiles, "")).strip() if not pd.isna(target.get(col_targets_smiles)) else ""
        t_formula = str(target.get(col_targets_formula, "")).replace(" ", "").lower() if not pd.isna(target.get(col_targets_formula)) else ""

        match_cond = gnps_df.apply(
            lambda row: (t_cas != "" and row['clean_cas'] == t_cas) or
                        (t_core != "" and t_core == get_core_name(row[col_gnps_compound])) or
                        (t_smiles != "" and row['clean_smiles'] != "" and row['clean_smiles'] == t_smiles) or
                        (t_formula != "" and row['clean_formula'] != "" and row['clean_formula'] == t_formula),
            axis=1
        )
        hits = gnps_df[match_cond].copy()
        if not hits.empty:
            hits['Standardized_Compound'] = t_name
            hits['Matched_GNPS_Name'] = hits[col_gnps_compound]
            matched_rows.append(hits)

    if not matched_rows: return pd.DataFrame()

    matched_gnps = pd.concat(matched_rows, ignore_index=True)
    
    # Validate and map MZmine columns (case-insensitive)
    required_mzmine_cols = [col_mzmine_scan, col_mzmine_mz, col_mzmine_rt, col_mzmine_rt_start, col_mzmine_rt_end]
    mzmine_col_map = validate_columns_ci(mzmine_df, required_mzmine_cols, "MZmine data")
    
    # Map to actual column names in the dataframe
    col_mzmine_scan = mzmine_col_map[col_mzmine_scan]
    col_mzmine_mz = mzmine_col_map[col_mzmine_mz]
    col_mzmine_rt = mzmine_col_map[col_mzmine_rt]
    col_mzmine_rt_start = mzmine_col_map[col_mzmine_rt_start]
    col_mzmine_rt_end = mzmine_col_map[col_mzmine_rt_end]
    col_mzmine_height = find_column_ci(mzmine_df, col_mzmine_height)
    col_mzmine_charge = find_column_ci(mzmine_df, col_mzmine_charge)
    
    # Automatically identify any fwhm columns (e.g., datafile:samplename.mzML:fwhm)
    fwhm_cols = [c for c in mzmine_df.columns if 'fwhm' in str(c).lower()]
    if fwhm_cols:
        mzmine_df['avg_fwhm'] = mzmine_df[fwhm_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1).fillna(0.0)
    else:
        mzmine_df['avg_fwhm'] = 0.0
    
    merged = pd.merge(matched_gnps, mzmine_df, left_on=col_gnps_scan, right_on=col_mzmine_scan, how='inner')
    if col_mzmine_height in merged.columns:
        merged[col_mzmine_height] = pd.to_numeric(merged[col_mzmine_height], errors='coerce').fillna(0)

    results = []
    for compound_name, group in merged.groupby('Standardized_Compound'):
        best_idx = group[col_mzmine_height].idxmax()
        best_row = group.loc[best_idx]

        peak_rt = best_row[col_mzmine_rt]
        composite_rt_min = group[col_mzmine_rt_start].min()
        composite_rt_max = group[col_mzmine_rt_end].max()
        fwhm_val = best_row.get('avg_fwhm', 0.0)
        if pd.isna(fwhm_val): fwhm_val = 0.0
        
        # Calculate RT window based on selected mode
        t_start, t_stop, exp_peak_width_sec = calculate_rt_window(
            peak_rt, composite_rt_min, composite_rt_max, rt_window_mode,
            rt_margin_min, rt_margin_pct, expected_peak_width_sec, fwhm_val
        )

        c_name = str(compound_name)
        
        # ----------------------------------------------------
        # Safe Charge (z) Extraction
        # ----------------------------------------------------
        raw_z = best_row.get(col_mzmine_charge)
        if pd.isna(raw_z) or raw_z == "":
            z_val = 1 if polarity == "Positive" else -1
        else:
            try:
                z_val = int(float(raw_z))
            except (ValueError, TypeError):
                z_val = 1 if polarity == "Positive" else -1

        # ----------------------------------------------------
        # Safe Adduct Formatting (Force brackets e.g. [M+H])
        # ----------------------------------------------------
        raw_adduct = best_row.get(_col_gnps_adduct, "")
        formatted_adduct = format_adduct(raw_adduct, polarity)

        result_row = {
            'Compound': c_name[:1].upper() + c_name[1:] if c_name else "",
            'Formula': best_row.get(col_gnps_formula, ""),
            'Polarity': polarity,
            'Adduct': formatted_adduct,
            'm/z': best_row[col_mzmine_mz],
            'z': z_val,
            'Peak_RT': peak_rt,
            'Exp_Peak_Width_sec': exp_peak_width_sec,
            'Height': best_row[col_mzmine_height],
            'Matched_GNPS_Name': best_row['Matched_GNPS_Name'],
            'Matched_Scan': best_row[col_gnps_scan],
            'HCD Collision Energies (%)': hcd_energies,
            'Match_Status': '✓ Matched',
            'RT_Window_Mode': rt_window_mode
        }
        
        # Add RT columns based on mode
        if rt_window_mode == 'rt_window':
            result_row['RT Time (min)'] = t_start
            result_row['Window (min)'] = t_stop
        elif rt_window_mode != 'unscheduled' and t_start is not None:
            result_row['t start (min)'] = t_start
            result_row['t stop (min)'] = t_stop
        
        results.append(result_row)
    return pd.DataFrame(results)

def compute_concurrency_and_metrics(df, title_base, resolution, it_mode, custom_it, desired_pts, peak_width_source="Use configured peak width", configured_peak_width_min=0.167):
    if df.empty: return df, []
    
    out_dfs = []
    figures = []
    
    # Process each multiplex group independently
    for grp_num, group_df in df.groupby('Multiplex_Group'):
        time_grid = np.arange(0, 16.01, 0.01)
        concurrency = np.zeros_like(time_grid)

        for _, row in group_df.iterrows():
            mask = (time_grid >= row['t start (min)']) & (time_grid <= row['t stop (min)'])
            concurrency[mask] += 1

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time_grid, concurrency, color='#8c564b', linewidth=2)
        ax.fill_between(time_grid, concurrency, color='#8c564b', alpha=0.3)
        ax.set_title(f"Concurrency Plot ({title_base} - Grp {grp_num})\nRes: {resolution}, IT Mode: {it_mode}", fontsize=13)
        ax.set_xlabel("Retention Time (min)")
        ax.set_ylabel("Simultaneous Targets")
        ax.set_xlim(0, 16)
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        figures.append((grp_num, fig))

        def get_metrics(row):
            mask = (time_grid >= row['t start (min)']) & (time_grid <= row['t stop (min)'])
            max_conc = int(np.max(concurrency[mask])) if np.any(mask) else 0
            
            # Choose peak width based on user selection and convert to seconds
            if peak_width_source == "Use configured peak width":
                exp_peak_width_sec = configured_peak_width_min * 60  # Convert minutes to seconds
            else:
                exp_peak_width_sec = row['Exp_Peak_Width_sec']  # Already in seconds
            
            scan_time_ms = calculate_scan_time(resolution, it_mode, custom_it, exp_peak_width_sec, desired_pts, max_conc)
            max_cycle_time_sec = (max_conc * scan_time_ms) / 1000.0
            pts = exp_peak_width_sec / max_cycle_time_sec if max_cycle_time_sec > 0 else 0
            return pd.Series([max_conc, max_cycle_time_sec, round(pts, 1)])

        group_df[['Max_Concurrent', 'Max_Cycle_Time_sec', 'Est_Points_Per_Peak']] = group_df.apply(get_metrics, axis=1)
        out_dfs.append(group_df)
        
    return pd.concat(out_dfs).sort_values('Peak_RT'), figures

def build_points_per_peak_figure(df, title):
    if df.empty: return []
    figures = []
    
    for grp_num, group_df in df.groupby('Multiplex_Group'):
        group_df = group_df.sort_values(by='Est_Points_Per_Peak', ascending=False).reset_index(drop=True)
        
        fig, ax = plt.subplots(figsize=(10, max(5, len(group_df) * 0.35)))
        
        # Color mapping based on points per peak
        def get_color(pts):
            if pts > 10: return '#2ca02c'      # green
            elif pts >= 8: return '#ff7f0e'    # orange
            elif pts >= 2: return '#d62728'    # red
            else: return '#8b0000'             # dark red
        
        colors = [get_color(pts) for pts in group_df['Est_Points_Per_Peak']]
        bars = ax.barh(group_df['Compound'], group_df['Est_Points_Per_Peak'], color=colors, edgecolor='black', linewidth=0.8)
        
        for i, (bar, val) in enumerate(zip(bars, group_df['Est_Points_Per_Peak'])):
            ax.text(val + 0.1, bar.get_y() + bar.get_height()/2, f'{val:.1f}',
                    va='center', ha='left', fontsize=9, fontweight='bold')
        
        ax.axvline(x=10, color='green', linestyle='--', alpha=0.5, linewidth=1, label='Excellent (>10)')
        ax.axvline(x=8, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='Good (8-10)')
        ax.axvline(x=2, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Poor (2-8)')
        
        ax.set_xlabel('Estimated Points Per Peak', fontsize=11, fontweight='bold')
        ax.set_title(f"{title} (Multiplex Group {grp_num})", fontsize=12, fontweight='bold')
        ax.set_xlim(0, max(group_df['Est_Points_Per_Peak']) * 1.15)
        ax.grid(True, axis='x', linestyle='--', alpha=0.5)
        ax.legend(loc='lower right', fontsize=9)
        plt.tight_layout()
        figures.append((grp_num, fig))
    
    return figures

def build_rt_alignment_figure(df, title, rt_window_mode="Composite range ± margin"):
    if df.empty: return []
    figures = []
    
    for grp_num, group_df in df.groupby('Multiplex_Group'):
        group_df = group_df.sort_values(by=['t start (min)', 'Peak_RT']).reset_index(drop=True)
        
        fig, ax = plt.subplots(figsize=(8, max(4, len(group_df) * 0.4)))
        y_labels = group_df['Compound'].tolist()
        y_ticks  = np.arange(len(y_labels))

        ax.grid(True, axis='both', linestyle='--', alpha=0.7, zorder=1)
        ax.hlines(y=y_ticks, xmin=group_df['t start (min)'], xmax=group_df['t stop (min)'], color='#222222', linewidth=3, zorder=2)
        ax.plot(group_df['Peak_RT'], y_ticks, 'o', color='red', markersize=5, zorder=3)

        for i, row in enumerate(group_df.itertuples()):
            ax.text(row.Peak_RT, i - 0.25, f"{row.Peak_RT:.2f}", va='bottom', ha='center', fontsize=9, zorder=4)

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel("Retention time (min)")
        
        mode_label = "📊 Composite ± Margin" if rt_window_mode in ["composite_margin", "composite_margin_pct"] else "⭕ Symmetric"
        if "fwhm" in rt_window_mode:
            mode_label = "📏 FWHM ± Margin"
        elif rt_window_mode == "hybrid":
            mode_label = "🧬 Hybrid"
            
        ax.set_title(f"{title} (Multiplex Group {grp_num})\n{mode_label}")
        ax.invert_yaxis()
        ax.set_ylim(len(y_labels) - 0.5, -1.5)
        ax.set_xlim(0, 16)
        ax.set_xticks(np.arange(0, 17, 2))
        plt.tight_layout()
        figures.append((grp_num, fig))
    
    return figures

def build_mzml_figure(mzml_file, target_df, title_base, xic_ppm_tolerance, targets_per_page=50):
    """
    Extracts TIC and XICs from mzML file for visualization.
    Returns a list of matplotlib Figures, one per page of up to `targets_per_page`
    compounds, so that every matched precursor is guaranteed to appear somewhere
    (no silent truncation), or None on failure.
    """
    if mzml_file is None or target_df.empty:
        return None

    try:
        import tempfile

        # Check required columns
        required_cols = ['Compound', 'm/z', 't start (min)', 't stop (min)', 'Peak_RT']
        missing_cols = [col for col in required_cols if col not in target_df.columns]
        if missing_cols:
            st.error(f"Missing required columns in target data: {missing_cols}")
            return None

        mzml_file.seek(0)  # Reset file pointer to start
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mzML') as tmp:
            tmp.write(mzml_file.read())
            tmp_path = tmp.name

        rt_list, tic_list = [], []
        targets = target_df.drop_duplicates('Compound')[['Compound', 'm/z', 't start (min)', 't stop (min)', 'Peak_RT']].to_dict('records')

        if not targets:
            st.error("No unique compounds found in target data")
            return None

        xic_data = {t['Compound']: [] for t in targets}
        target_ranges = {t['Compound']: (t['m/z'] - t['m/z'] * xic_ppm_tolerance / 2 / 1e6, t['m/z'] + t['m/z'] * xic_ppm_tolerance / 2 / 1e6) for t in targets}

        # Debug: Show sample m/z ranges being used
        if targets:
            sample_compound = targets[0]['Compound']
            sample_mz = targets[0]['m/z']
            mz_min, mz_max = target_ranges[sample_compound]
            ppm_delta = abs(mz_max - mz_min) / sample_mz * 1e6
            st.write(f"✓ XIC Tolerance applied: {xic_ppm_tolerance} ppm | Example: {sample_compound} @ m/z {sample_mz:.4f} → Range [{mz_min:.6f}, {mz_max:.6f}] (Δ {ppm_delta:.1f} ppm)")

        scan_count = 0
        with mzml.read(tmp_path) as reader:
            for spec in reader:
                if spec.get('ms level') != 1:
                    continue

                scan_count += 1
                scan_data = spec.get('scanList', {}).get('scan', [{}])[0]
                rt = scan_data.get('scan start time')
                if rt is None:
                    continue

                rt_val = float(rt) / 60.0 if hasattr(rt, 'unit_info') and 'second' in rt.unit_info else float(rt)
                rt_list.append(rt_val)

                mzs = spec.get('m/z array', np.array([]))
                ints = spec.get('intensity array', np.array([]))
                tic_list.append(ints.sum())

                if len(mzs) > 0:
                    for t in targets:
                        mz_min, mz_max = target_ranges[t['Compound']]
                        mask = (mzs >= mz_min) & (mzs <= mz_max)
                        xic_data[t['Compound']].append(ints[mask].sum() if np.any(mask) else 0)
                else:
                    for t in targets:
                        xic_data[t['Compound']].append(0)

        os.unlink(tmp_path)

        if scan_count == 0:
            st.error(f"No MS1 scans found in mzML file")
            return None

        if len(rt_list) == 0:
            st.error(f"Could not extract retention times from {scan_count} scans")
            return None

        # Paginate so every target appears somewhere, instead of truncating at a
        # fixed cap (very tall single figures risk PIL "decompression bomb" errors).
        num_targets = len(targets)
        pages = [targets[i:i + targets_per_page] for i in range(0, num_targets, targets_per_page)]
        num_pages = len(pages)
        cmap = plt.get_cmap('tab20')
        figures = []

        import warnings

        for page_idx, page_targets in enumerate(pages, start=1):
            n = len(page_targets)
            fig, axes = plt.subplots(2 + n, 1, figsize=(14, 6 + 1.2 * n), dpi=72, sharex=True)

            page_title = f"TIC & XICs ({title_base})" + (f" — Page {page_idx}/{num_pages}" if num_pages > 1 else "")

            # TIC
            axes[0].plot(rt_list, tic_list, color='#444444', linewidth=1.5)
            axes[0].fill_between(rt_list, tic_list, color='#444444', alpha=0.2)
            axes[0].set_title(page_title, fontsize=14, fontweight='bold')
            axes[0].set_ylabel("TIC Intensity", fontsize=11)
            axes[0].grid(True, linestyle='--', alpha=0.7)

            # Overlaid XICs for this page's compounds (without legend to avoid clutter)
            for i, t in enumerate(page_targets):
                axes[1].plot(rt_list, xic_data[t['Compound']], linewidth=1.5, color=cmap(i % 20), alpha=0.7)
            axes[1].set_ylabel("Overlaid XICs", fontsize=11)
            axes[1].grid(True, linestyle='--', alpha=0.7)

            # Individual XICs with improved labels
            for i, t in enumerate(page_targets):
                ax = axes[2 + i]
                color = cmap(i % 20)
                xic_values = xic_data[t['Compound']]

                ax.plot(rt_list, xic_values, color=color, linewidth=2)
                ax.fill_between(rt_list, xic_values, color=color, alpha=0.25)

                # Highlight RT window
                if 't start (min)' in t and 't stop (min)' in t:
                    ax.axvspan(t['t start (min)'], t['t stop (min)'], color='gray', alpha=0.1, zorder=0, label='RT Window')

                # Mark peak RT with vertical line
                peak_rt = t['Peak_RT']
                ax.axvline(peak_rt, color='red', linestyle='--', linewidth=1.5, alpha=0.7, zorder=2)

                # Add RT label on the peak line
                max_intensity = max(xic_values) if xic_values else 1
                ax.text(peak_rt, max_intensity * 0.95, f'RT: {peak_rt:.2f}', rotation=0, fontsize=9,
                       ha='center', va='top', color='red', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.6, edgecolor='red'))

                # Compound label - positioned to avoid peak if possible
                label_rt = 0.5 if peak_rt > 8 else 15  # Position label on opposite side of peak
                label_y_pos = max_intensity * 0.75
                ax.text(label_rt, label_y_pos, f"{t['Compound']}\nm/z: {t['m/z']:.4f}",
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='black', linewidth=1),
                       ha='left' if label_rt < 8 else 'right', va='top')

                ax.set_ylabel("Intensity", fontsize=10)
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.set_ylim(bottom=0)

            axes[-1].set_xlabel("Retention Time (min)", fontsize=12, fontweight='bold')
            axes[-1].set_xlim(0, 16)
            axes[-1].set_xticks(np.arange(0, 17, 2))
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                plt.tight_layout()

            figures.append(fig)

        st.success(f"✓ Extracted XICs from {scan_count} MS1 scans for all {num_targets} targets, across {num_pages} page(s) of up to {targets_per_page} compounds each")
        return figures
    except Exception as e:
        st.error(f"❌ Could not parse mzML file: {str(e)}")
        import traceback
        st.text(traceback.format_exc())
        return None

def build_mz_rt_figure(target_df, title_base, polarity):
    """Creates an interactive 2D scatter plot: m/z vs Retention Time using Plotly"""
    if target_df.empty:
        return None
    
    # Normalize height for color scaling
    height_normalized = (target_df['Height'] - target_df['Height'].min()) / (target_df['Height'].max() - target_df['Height'].min() + 1e-10) if target_df['Height'].max() > target_df['Height'].min() else [0.5] * len(target_df)
    
    fig = go.Figure()
    
    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=target_df['Peak_RT'],
        y=target_df['m/z'],
        mode='markers',
        text=target_df['Compound'],
        marker=dict(
            size=8,
            color=target_df['Height'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title="Peak Height<br>(Intensity)",
                thickness=15,
                len=0.7,
                tickformat='.2e'
            ),
            line=dict(color='black', width=0.5),
            opacity=0.7
        ),
        hovertemplate='<b>%{text}</b><br>RT: %{x:.2f} min<br>m/z: %{y:.4f}<br>Height: %{marker.color:.2e}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=f'm/z vs Retention Time ({title_base})',
            font=dict(size=16, color='black')
        ),
        xaxis=dict(
            title=dict(text='Retention Time (min)', font=dict(size=13)),
            gridcolor='rgba(128,128,128,0.2)'
        ),
        yaxis=dict(
            title=dict(text='m/z', font=dict(size=13)),
            gridcolor='rgba(128,128,128,0.2)'
        ),
        hovermode='closest',
        width=1000,
        height=600,
        plot_bgcolor='rgba(240,240,240,0.5)',
        showlegend=False
    )
    
    return fig

# =============================================================================
# STREAMLIT UI & SIDEBAR
# =============================================================================
try:
    st.set_page_config(page_title="GNPS2-Quant | PRM Method Generator", page_icon="🔬", layout="wide")
except Exception:
    pass

with st.sidebar:
    st.header("⚙️ Instrument Settings")
    
    st.markdown("**Data Processing**")
    expected_peak_width_min = st.number_input("Expected Peak Width at Base (min)", min_value=0.017, max_value=2.0, value=0.167, step=0.01, help="Default peak width for calculations (in minutes)")
    peak_width_source = st.radio("Peak Width Source", options=["Use configured peak width", "Use actual peak widths from data"], help="Choose whether to use configured value or experimental peak widths from MZmine for points/peak calculation")
    
    st.divider()
    st.markdown("**Ionization Mode**")
    polarity_mode = st.radio("Polarity Mode", ["Positive & Negative", "Positive Only", "Negative Only"], help="Select which ionization modes to process", key="polarity_main")
    
    st.divider()
    st.markdown("**Orbitrap Configuration**")
    orbitrap_resolution = st.selectbox("Orbitrap Resolution", 
                                       options=[7500, 11250, 15000, 22500, 30000, 45000, 60000, 75000, 90000, 120000, 180000, 240000, 480000], 
                                       index=4, format_func=lambda x: f"{x:,}",
                                       help="Valid targeted MS2 resolutions for Exploris 480")
    
    it_mode = st.radio("Max Injection Time (IT) Mode", ["Auto", "Dynamic", "Custom"])
    custom_it = 55
    desired_pts = 10
    if it_mode == "Custom":
        custom_it = st.number_input("Custom Max IT (ms)", value=55)
    elif it_mode == "Dynamic":
        desired_pts = st.number_input("Desired Minimum Points", value=10)
    
    st.divider()
    st.markdown("**MS/MS Parameters**")
    hcd_energies = st.text_input("HCD Collision Energies (%)", value="25,35,45", help="Comma-separated NCE values applied to all targets")
    
    st.divider()
    xic_ppm_tolerance = st.number_input("XIC m/z Tolerance (ppm)", min_value=1, max_value=100, value=10, help="Click 'Evaluate & Optimize Method' button to apply changes")

    st.divider()
    st.markdown("**Skyline Output Parameters**")
    compound_match_ppm_tolerance = st.number_input("Compound Matching Tolerance (ppm)", min_value=1, max_value=100, value=10, help="How close an MGF spectrum's PEPMASS must be to a compound's m/z (in the inclusion list) to be matched to that compound.")
    fragment_dedup_ppm = st.number_input("Fragment Dedup Tolerance (ppm)", min_value=1, max_value=100, value=5, help="Once a spectrum is matched to a compound, fragment ions within this ppm tolerance of each other are treated as duplicates and only the most abundant one is kept.")

    st.divider()
    st.markdown("**Retention Time Window Configuration**")
    
    rt_mode_type = st.selectbox("Time Mode", 
                                options=["Start/End Time", "Retention Time Window", "Unscheduled"],
                                help="Choose how RT is handled in inclusion list:\n- Start/End Time: Include start/stop times\n- Retention Time Window: Include RT center + window width\n- Unscheduled: No RT information")
    
    rt_margin_min = 1.0
    rt_margin_pct = None
    rt_window_mode = "composite_margin"
    
    if rt_mode_type == "Start/End Time":
        st.markdown("**Select Calculation Option:**")
        rt_calculation = st.radio("Calculation Method",
                                  options=[
                                      "Option 1a: Symmetric ±margin (min)",
                                      "Option 1b: Symmetric ±margin (%)",
                                      "Option 2a: Composite ±margin (min)",
                                      "Option 2b: Composite ±margin (%)",
                                      "Option 3a: FWHM ±margin (min)",
                                      "Option 3b: FWHM ±margin (%)",
                                      "Option 4: Hybrid"
                                  ],
                                  help="How to calculate start/stop times from MZmine data:\n- Option 1a: RT ±margin (fixed min)\n- Option 1b: RT ±margin (% of RT)\n- Option 2a: (min_RT to max_RT) ±margin_min\n- Option 2b: (min_RT to max_RT) ±margin_%\n- Option 3a: FWHM boundaries (RT ± FWHM) ±margin (min)\n- Option 3b: FWHM boundaries (RT ± FWHM) ±margin (%)\n- Option 4: Use margin_min for narrow peaks, margin_% for wide peaks")
        
        if rt_calculation == "Option 1a: Symmetric ±margin (min)":
            rt_window_mode = "symmetric_margin"
            rt_margin_min = st.number_input("Margin (min)", value=0.5, step=0.1, key="rt_margin_opt1a")
        elif rt_calculation == "Option 1b: Symmetric ±margin (%)":
            rt_window_mode = "symmetric_margin_pct"
            rt_margin_pct = st.number_input("Margin (%)", value=10.0, step=1.0, min_value=0.0, max_value=100.0, key="rt_margin_opt1b")
        elif rt_calculation == "Option 2a: Composite ±margin (min)":
            rt_window_mode = "composite_margin"
            rt_margin_min = st.number_input("Margin (min)", value=1.0, step=0.1, key="rt_margin_opt2a")
        elif rt_calculation == "Option 2b: Composite ±margin (%)":
            rt_window_mode = "composite_margin_pct"
            rt_margin_pct = st.number_input("Margin (%)", value=25.0, step=1.0, min_value=0.0, max_value=100.0, key="rt_margin_opt2b")
        elif rt_calculation == "Option 3a: FWHM ±margin (min)":
            rt_window_mode = "fwhm_margin"
            rt_margin_min = st.number_input("Margin (min)", value=0.5, step=0.1, key="rt_margin_opt3a")
        elif rt_calculation == "Option 3b: FWHM ±margin (%)":
            rt_window_mode = "fwhm_margin_pct"
            rt_margin_pct = st.number_input("Margin (%)", value=10.0, step=1.0, min_value=0.0, max_value=100.0, key="rt_margin_opt3b")
        elif rt_calculation == "Option 4: Hybrid":
            rt_window_mode = "hybrid"
            col1, col2 = st.columns(2)
            with col1:
                rt_margin_min = st.number_input("Margin for narrow peaks (min)", value=0.5, step=0.1, key="rt_margin_hybrid_min")
            with col2:
                rt_margin_pct = st.number_input("Margin for wide peaks (%)", value=25.0, step=1.0, key="rt_margin_hybrid_pct")
    
    elif rt_mode_type == "Retention Time Window":
        rt_window_mode = "rt_window"
        rt_margin_min = st.number_input("Window Size (min)", value=1.0, step=0.1, key="rt_window_size",
                                        help="Half-width of the retention time window around the peak")
    
    elif rt_mode_type == "Unscheduled":
        rt_window_mode = "unscheduled"
        st.info("ℹ️ No retention time information will be included in the inclusion list.")

    st.divider()
    with st.expander("🔧 Advanced: Column Mappings"):
        st.markdown("**Customize column names** if your files use different naming conventions:")
        st.info("⚠️ Ensure your column names match exactly. Leave blank to skip optional columns.")
        
        # Initialize session state with defaults
        if "col_gnps_compound" not in st.session_state:
            st.session_state.col_gnps_compound = GNPS_COMPOUND_COL
        if "col_gnps_scan" not in st.session_state:
            st.session_state.col_gnps_scan = GNPS_SCAN_COL
        if "col_gnps_formula" not in st.session_state:
            st.session_state.col_gnps_formula = GNPS_FORMULA_COL
        if "col_gnps_adduct" not in st.session_state:
            st.session_state.col_gnps_adduct = GNPS_ADDUCT_COL
        if "col_gnps_cas" not in st.session_state:
            st.session_state.col_gnps_cas = GNPS_CAS_COL
        if "col_gnps_smiles" not in st.session_state:
            st.session_state.col_gnps_smiles = GNPS_SMILES_COL
        if "col_mzmine_scan" not in st.session_state:
            st.session_state.col_mzmine_scan = MZMINE_SCAN_COL
        if "col_mzmine_mz" not in st.session_state:
            st.session_state.col_mzmine_mz = MZMINE_MZ_COL
        if "col_mzmine_rt" not in st.session_state:
            st.session_state.col_mzmine_rt = MZMINE_RT_COL
        if "col_mzmine_rt_start" not in st.session_state:
            st.session_state.col_mzmine_rt_start = MZMINE_RT_START_COL
        if "col_mzmine_rt_end" not in st.session_state:
            st.session_state.col_mzmine_rt_end = MZMINE_RT_END_COL
        if "col_mzmine_height" not in st.session_state:
            st.session_state.col_mzmine_height = MZMINE_HEIGHT_COL
        if "col_mzmine_charge" not in st.session_state:
            st.session_state.col_mzmine_charge = MZMINE_CHARGE_COL
        if "col_targets_compound" not in st.session_state:
            st.session_state.col_targets_compound = TARGETS_COMPOUND_COL
        if "col_targets_cas" not in st.session_state:
            st.session_state.col_targets_cas = TARGETS_CAS_COL
        if "col_targets_smiles" not in st.session_state:
            st.session_state.col_targets_smiles = TARGETS_SMILES_COL
        if "col_targets_formula" not in st.session_state:
            st.session_state.col_targets_formula = TARGETS_FORMULA_COL
        
        st.markdown("**GNPS Columns (Required)**")
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.col_gnps_compound = st.text_input("GNPS Compound Name", value=st.session_state.col_gnps_compound, key="input_gnps_compound", help="Column for compound names")
            st.session_state.col_gnps_formula = st.text_input("GNPS Formula", value=st.session_state.col_gnps_formula, key="input_gnps_formula", help="Column for molecular formula")
            st.session_state.col_gnps_cas = st.text_input("GNPS CAS Number", value=st.session_state.col_gnps_cas, key="input_gnps_cas", help="Optional: Column for CAS numbers")
        with col2:
            st.session_state.col_gnps_scan = st.text_input("GNPS Scan ID", value=st.session_state.col_gnps_scan, key="input_gnps_scan", help="Column for scan numbers")
            st.session_state.col_gnps_adduct = st.text_input("GNPS Adduct", value=st.session_state.col_gnps_adduct, key="input_gnps_adduct", help="Column for adduct info")
            st.session_state.col_gnps_smiles = st.text_input("GNPS SMILES", value=st.session_state.col_gnps_smiles, key="input_gnps_smiles", help="Optional: Column for SMILES")
        
        st.markdown("**MZmine Columns (Required)**")
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.col_mzmine_mz = st.text_input("MZmine m/z", value=st.session_state.col_mzmine_mz, key="input_mzmine_mz", help="Column for m/z values")
            st.session_state.col_mzmine_rt = st.text_input("MZmine RT (peak)", value=st.session_state.col_mzmine_rt, key="input_mzmine_rt", help="Column for peak retention time")
            st.session_state.col_mzmine_height = st.text_input("MZmine Height", value=st.session_state.col_mzmine_height, key="input_mzmine_height", help="Column for peak height/intensity")
        with col2:
            st.session_state.col_mzmine_scan = st.text_input("MZmine Scan ID", value=st.session_state.col_mzmine_scan, key="input_mzmine_scan", help="Column for scan/feature ID")
            st.session_state.col_mzmine_rt_start = st.text_input("MZmine RT Start", value=st.session_state.col_mzmine_rt_start, key="input_mzmine_rt_start", help="Column for peak start time")
            st.session_state.col_mzmine_rt_end = st.text_input("MZmine RT End", value=st.session_state.col_mzmine_rt_end, key="input_mzmine_rt_end", help="Column for peak end time")
            st.session_state.col_mzmine_charge = st.text_input("MZmine Charge", value=st.session_state.col_mzmine_charge, key="input_mzmine_charge", help="Column for charge state")
        
        st.markdown("**Target Compounds Columns**")
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.col_targets_compound = st.text_input("Target Compound Name", value=st.session_state.col_targets_compound, key="input_targets_compound", help="Column for compound names")
            st.session_state.col_targets_cas = st.text_input("Target CAS", value=st.session_state.col_targets_cas, key="input_targets_cas", help="Optional: Column for CAS numbers")
        with col2:
            st.session_state.col_targets_smiles = st.text_input("Target SMILES", value=st.session_state.col_targets_smiles, key="input_targets_smiles", help="Optional: Column for SMILES")
            st.session_state.col_targets_formula = st.text_input("Target Formula", value=st.session_state.col_targets_formula, key="input_targets_formula", help="Optional: Column for formula")

st.title("🔬 GNPS2-Quant Method Optimizer")
st.markdown("Optimize Points-Per-Peak prior to exporting your Thermo Exploris 480 Inclusion Lists.")
st.markdown("**Workflow:** Match your DDA datasets (GNPS2 + MZmine results) against a target compound list.")

st.markdown("""
---
**💡 How it works:**
1. Upload your **target compound list** (Compounds.csv)
2. Upload **DDA experiment results** — can be 1 experiment or many (e.g., 5 different methods)
3. The script will match compounds across all datasets and select the best peak for each
4. Results are combined and optimized for your PRM method

---
""")

st.subheader("📋 Step 1: Upload Target Compound List")
st.markdown("""
This is your **internal reference library** — the compounds you want to target in your PRM method.
""")
f_compounds = st.file_uploader("Compounds.csv (Required)", type=["csv"], key="compounds_upload")

st.subheader("📁 Step 2: Upload DDA Datasets")
st.markdown("""
Upload your DDA experiment results. **You can load multiple datasets** from different experiments or methods.

**How to upload multiple datasets:**
- Upload multiple GNPS2 result files (e.g., from different experiments)
- Upload the corresponding MZmine feature tables **in the same order**
- The script will process each GNPS/MZmine pair separately, then combine results
- Results from all datasets are aggregated and compared (highest intensity wins per compound)

**Example:** If you upload GNPS files [exp1.csv, exp2.csv, exp3.csv], 
upload MZmine files [exp1_feat.csv, exp2_feat.csv, exp3_feat.csv] in the **same order**.
""")

col1, col2 = st.columns(2)

# Positive mode files
if polarity_mode in ["Positive & Negative", "Positive Only"]:
    with col1:
        st.subheader("📍 ESI+ (Positive Mode)")
        st.caption("GNPS2 results and MZmine feature tables for ESI+ — upload in matching order")
        f_gnps_pos = st.file_uploader("GNPS2 Results — ESI+ (.csv)", type=["csv"], accept_multiple_files=True, key="gnps_pos")
        f_mzmine_pos = st.file_uploader("MZmine 3 Features — ESI+ (.csv)", type=["csv"], accept_multiple_files=True, key="mzmine_pos")
else:
    f_gnps_pos = []
    f_mzmine_pos = []

# Negative mode files
if polarity_mode in ["Positive & Negative", "Negative Only"]:
    if polarity_mode == "Positive & Negative":
        with col2:
            st.subheader("📍 ESI− (Negative Mode)")
            st.caption("GNPS2 results and MZmine feature tables for ESI− — upload in matching order")
            f_gnps_neg = st.file_uploader("GNPS2 Results — ESI− (.csv)", type=["csv"], accept_multiple_files=True, key="gnps_neg")
            f_mzmine_neg = st.file_uploader("MZmine 3 Features — ESI− (.csv)", type=["csv"], accept_multiple_files=True, key="mzmine_neg")
    else:
        st.subheader("📍 ESI− (Negative Mode)")
        st.caption("GNPS2 results and MZmine feature tables for ESI− — upload in matching order")
        f_gnps_neg = st.file_uploader("GNPS2 Results — ESI− (.csv)", type=["csv"], accept_multiple_files=True, key="gnps_neg")
        f_mzmine_neg = st.file_uploader("MZmine 3 Features — ESI− (.csv)", type=["csv"], accept_multiple_files=True, key="mzmine_neg")
else:
    f_gnps_neg = []
    f_mzmine_neg = []

st.divider()
st.subheader("📊 Step 3: Optional Raw Data & Skyline Export")

# Optional LC-MS raw data for XIC visualization
st.markdown("**LC-MS Raw Data (optional)** — for TIC and XIC chromatograms")
f_mzml_pos = None
f_mzml_neg = None
if polarity_mode in ["Positive & Negative", "Positive Only"]:
    f_mzml_pos = st.file_uploader("LC-MS Raw Data (.mzML) — ESI+", type=["mzML", "mzml"], key="mzml_pos", help="Leave empty to skip XIC extraction (speeds up processing)")
if polarity_mode in ["Positive & Negative", "Negative Only"]:
    f_mzml_neg = st.file_uploader("LC-MS Raw Data (.mzML) — ESI−", type=["mzML", "mzml"], key="mzml_neg", help="Leave empty to skip XIC extraction (speeds up processing)")

st.divider()
generate_skyline = st.checkbox("Generate Skyline Mass List? (Extracts fragments from MGF)", value=False, help="Requires MGF files with fragment peak data")
mgf_pos_files = []
mgf_neg_files = []
if generate_skyline:
    st.info(f"ℹ️ Skyline transitions require: (1) MGF files uploaded below for the relevant polarity, and (2) at least one MS/MS spectrum whose PEPMASS falls within ±{compound_match_ppm_tolerance} ppm of a matched inclusion-list target's m/z. If nothing is uploaded, or nothing matches, no Skyline file will be generated.")
    if polarity_mode in ["Positive & Negative", "Positive Only"]:
        mgf_pos_files = st.file_uploader("GNPS MGF — ESI+ (.mgf)", type=["mgf"], accept_multiple_files=True, key="mgf_pos", help="Consensus MS/MS MGF from GNPS")
    if polarity_mode in ["Positive & Negative", "Negative Only"]:
        mgf_neg_files = st.file_uploader("GNPS MGF — ESI− (.mgf)", type=["mgf"], accept_multiple_files=True, key="mgf_neg", help="Consensus MS/MS MGF from GNPS")

st.divider()

# Validation
if f_compounds is None:
    st.warning("⚠️ Upload a Compounds.csv file to proceed")
    st.stop()

required_pos_match = len(f_gnps_pos) == len(f_mzmine_pos)
required_neg_match = len(f_gnps_neg) == len(f_mzmine_neg)
has_any_data = (len(f_gnps_pos) > 0 and required_pos_match) or (len(f_gnps_neg) > 0 and required_neg_match)

if not has_any_data:
    st.info("📥 Upload matching pairs of GNPS and MZmine files to proceed.")
elif not required_pos_match or not required_neg_match:
    st.error("❌ Number of GNPS files must equal the number of MZmine files for each polarity.")

all_ready = has_any_data and (f_compounds is not None)

# Display upload status
st.divider()
st.subheader("📋 Upload Status")
col1, col2 = st.columns(2)
with col1:
    st.write(f"**Compounds.csv:** ✓ Loaded")
    st.write(f"**ESI+ GNPS files:** {len(f_gnps_pos)} files")
    st.write(f"**ESI+ MZmine files:** {len(f_mzmine_pos)} files")
    st.write(f"**ESI+ mzML file:** {'✓ Uploaded' if f_mzml_pos else '✗ Not uploaded'}")
with col2:
    st.write(f"**ESI− GNPS files:** {len(f_gnps_neg)} files")
    st.write(f"**ESI− MZmine files:** {len(f_mzmine_neg)} files")
    st.write(f"**ESI− mzML file:** {'✓ Uploaded' if f_mzml_neg else '✗ Not uploaded'}")
st.divider()

if st.button("▶ Evaluate & Optimize Method", type="primary", disabled=not all_ready):
    with st.status("Evaluating Method Constraints...", expanded=True) as status:
        try:
            # 1. Load Targets
            if f_compounds:
                targets = pd.read_csv(f_compounds, encoding='latin1')
                targets['clean_cas'] = targets.get('CAS', pd.Series(dtype=str)).apply(clean_cas)
            else:
                all_compounds = set()
                for f in (f_gnps_pos + f_gnps_neg):
                    f.seek(0); temp_df = pd.read_csv(f, encoding='latin1')
                    if 'Compound_Name' in temp_df.columns: all_compounds.update(temp_df['Compound_Name'].dropna().unique())
                targets = pd.DataFrame({'Compound': list(all_compounds)})
                targets['clean_cas'] = ""
                
            # 2. Process
            def run_polarity_multipe(g_files, m_files, pol):
                all_res = []
                # Get custom column mappings from session state
                col_map = get_col_mapping()
                for g, m in zip(sorted(g_files, key=lambda x: x.name), sorted(m_files, key=lambda x: x.name)):
                    g.seek(0); m.seek(0)
                    res = process_polarity(
                        pd.read_csv(g, encoding='latin1'), 
                        pd.read_csv(m, encoding='latin1'), 
                        pol, targets, "25,35,45", 
                        rt_window_mode, rt_margin_min, rt_margin_pct, expected_peak_width_min,
                        col_gnps_compound=col_map['gnps_compound'],
                        col_gnps_scan=col_map['gnps_scan'],
                        col_gnps_cas=col_map['gnps_cas'],
                        col_gnps_smiles=col_map['gnps_smiles'],
                        col_gnps_formula=col_map['gnps_formula'],
                        col_gnps_adduct=col_map['gnps_adduct'],
                        col_mzmine_scan=col_map['mzmine_scan'],
                        col_mzmine_mz=col_map['mzmine_mz'],
                        col_mzmine_rt=col_map['mzmine_rt'],
                        col_mzmine_rt_start=col_map['mzmine_rt_start'],
                        col_mzmine_rt_end=col_map['mzmine_rt_end'],
                        col_mzmine_height=col_map['mzmine_height'],
                        col_mzmine_charge=col_map['mzmine_charge'],
                        col_targets_compound=col_map['targets_compound'],
                        col_targets_cas=col_map['targets_cas'],
                        col_targets_smiles=col_map['targets_smiles'],
                        col_targets_formula=col_map['targets_formula']
                    )
                    if not res.empty: all_res.append(res)
                return pd.concat(all_res, ignore_index=True) if all_res else pd.DataFrame()

            targets_pos = pd.DataFrame()
            targets_neg = pd.DataFrame()
            
            # Load targets from Compounds.csv (Targeted workflow)
            f_compounds.seek(0)
            targets = pd.read_csv(f_compounds, encoding='latin1')
            col_map = get_col_mapping()
            targets['clean_cas'] = targets.get(col_map['targets_cas'], pd.Series(dtype=str)).apply(clean_cas)
            st.write(f"✓ Loaded {len(targets)} target compounds from Compounds.csv")
            st.write(f"✓ Processing {len(f_gnps_pos) if f_gnps_pos else 0} ESI+ dataset(s)")
            st.write(f"✓ Processing {len(f_gnps_neg) if f_gnps_neg else 0} ESI− dataset(s)")
            st.info(f"📊 mzML files: ESI+ = {'✓ Uploaded' if f_mzml_pos else '✗ Not uploaded'}, ESI− = {'✓ Uploaded' if f_mzml_neg else '✗ Not uploaded'}")
            
            # Process based on selected polarity mode
            if polarity_mode in ["Positive & Negative", "Positive Only"]:
                targets_pos = run_polarity_multipe(f_gnps_pos, f_mzmine_pos, "Positive")
            
            if polarity_mode in ["Positive & Negative", "Negative Only"]:
                targets_neg = run_polarity_multipe(f_gnps_neg, f_mzmine_neg, "Negative")
            
            # Check if results are empty
            if targets_pos.empty and targets_neg.empty:
                st.error("❌ No compounds matched between GNPS and MZmine files. Please verify your input data.")
        except ValueError as e:
            st.error(f"❌ **Data Format Error:**\n\n{str(e)}\n\n**Troubleshooting Tips:**\n- Ensure GNPS files are exported from GNPS library with all required columns\n- Ensure MZmine files are exported with the correct format\n- Check that file encoding is UTF-8 or Latin1")
        except KeyError as e:
            st.error(f"❌ **Missing Column Error:**\n\nCannot find column: {str(e)}\n\n**Troubleshooting Tips:**\n- Verify your input files have the required columns\n- Check GNPS export settings include Compound_Name, #Scan#, CAS_Number, etc.\n- Check MZmine export includes id, mz, rt, rt_range:min, rt_range:max columns")
        except Exception as e:
            st.error(f"❌ **Unexpected Error:**\n\n{str(e)}\n\nPlease check your input files and try again.")
            st.stop()
        
        # Concatenate available results
        all_dfs = [df for df in [targets_pos, targets_neg] if not df.empty]
        all_targets = pd.concat(all_dfs, ignore_index=True).sort_values(by=['Compound', 'Height'], ascending=[True, False]).drop_duplicates('Compound')
        
        final_pos = pd.DataFrame()
        final_neg = pd.DataFrame()
        
        if polarity_mode in ["Positive & Negative", "Positive Only"]:
            final_pos = all_targets[all_targets['Polarity'] == 'Positive'].sort_values('Peak_RT').reset_index(drop=True)
        
        if polarity_mode in ["Positive & Negative", "Negative Only"]:
            final_neg = all_targets[all_targets['Polarity'] == 'Negative'].sort_values('Peak_RT').reset_index(drop=True)

        # 3. Multiplex Splitting
        split_method = st.session_state.get('split_method_selected', False)
        if split_method:
            if not final_pos.empty:
                final_pos['Multiplex_Group'] = np.where(final_pos.index % 2 == 0, 1, 2)
            if not final_neg.empty:
                final_neg['Multiplex_Group'] = np.where(final_neg.index % 2 == 0, 1, 2)
        else:
            if not final_pos.empty:
                final_pos['Multiplex_Group'] = 1
            if not final_neg.empty:
                final_neg['Multiplex_Group'] = 1

        # 4. Metrics
        fig_c_pos = []
        fig_c_neg = []
        fig_p_pos = []
        fig_p_neg = []
        fig_rt_pos = []
        fig_rt_neg = []
        fig_mz_rt_2d_pos = None
        fig_mz_rt_2d_neg = None
        
        if not final_pos.empty:
            final_pos, fig_c_pos = compute_concurrency_and_metrics(final_pos, "ESI+", orbitrap_resolution, it_mode, custom_it, desired_pts, peak_width_source, expected_peak_width_min)
            fig_p_pos = build_points_per_peak_figure(final_pos, "ESI+ Points/Peak")
            fig_rt_pos = build_rt_alignment_figure(final_pos, "Retention Time Windows — ESI+", rt_window_mode)
            fig_mz_rt_2d_pos = build_mz_rt_figure(final_pos, "ESI+", "Positive")
        
        if not final_neg.empty:
            final_neg, fig_c_neg = compute_concurrency_and_metrics(final_neg, "ESI-", orbitrap_resolution, it_mode, custom_it, desired_pts, peak_width_source, expected_peak_width_min)
            fig_p_neg = build_points_per_peak_figure(final_neg, "ESI- Points/Peak")
            fig_rt_neg = build_rt_alignment_figure(final_neg, "Retention Time Windows — ESI−", rt_window_mode)
            fig_mz_rt_2d_neg = build_mz_rt_figure(final_neg, "ESI−", "Negative")
        
        # 5. XIC figures from mzML (optional)
        fig_xic_pos = None
        fig_xic_neg = None
        
        # Debug: Display the tolerance value being used
        st.info(f"🔬 **XIC Tolerance Setting:** {xic_ppm_tolerance} ppm")
        
        if f_mzml_pos:
            try:
                with st.spinner("🔍 Extracting XICs from ESI+ mzML file..."):
                    if final_pos.empty:
                        st.warning("⚠️ No ESI+ targets found - skipping XIC extraction")
                    else:
                        st.write(f"📊 Using tolerance: {xic_ppm_tolerance} ppm for ESI+ XIC extraction")
                        fig_xic_pos = build_mzml_figure(f_mzml_pos, final_pos, "ESI+", xic_ppm_tolerance)
                        if fig_xic_pos is None:
                            st.warning("⚠️ Failed to extract ESI+ XICs - check mzML file format")
            except Exception as e:
                st.error(f"❌ Error processing ESI+ mzML: {str(e)}")
        
        if f_mzml_neg:
            try:
                with st.spinner("🔍 Extracting XICs from ESI− mzML file..."):
                    if final_neg.empty:
                        st.warning("⚠️ No ESI− targets found - skipping XIC extraction")
                    else:
                        st.write(f"📊 Using tolerance: {xic_ppm_tolerance} ppm for ESI− XIC extraction")
                        fig_xic_neg = build_mzml_figure(f_mzml_neg, final_neg, "ESI−", xic_ppm_tolerance)
                        if fig_xic_neg is None:
                            st.warning("⚠️ Failed to extract ESI− XICs - check mzML file format")
            except Exception as e:
                st.error(f"❌ Error processing ESI− mzML: {str(e)}")

        # 6. Create Match Summary (All compounds with matched/unmatched status)
        f_compounds.seek(0)
        all_targets_original = pd.read_csv(f_compounds, encoding='latin1')
        
        matched_compounds_pos = set(final_pos['Compound'].unique()) if not final_pos.empty else set()
        matched_compounds_neg = set(final_neg['Compound'].unique()) if not final_neg.empty else set()
        
        # Create consolidated match summary (one row per compound with ionization info)
        summary_data_consolidated = []
        all_compounds = set()
        col_map = get_col_mapping()
        col_targets_compound = col_map['targets_compound']
        
        for _, target in all_targets_original.iterrows():
            compound_name = target.get(col_targets_compound, '')
            all_compounds.add(compound_name)
        
        for compound_name in sorted(all_compounds):
            target = all_targets_original[all_targets_original[col_targets_compound] == compound_name].iloc[0] if compound_name in all_targets_original[col_targets_compound].values else None
            
            # Determine ionization status
            in_pos = compound_name in matched_compounds_pos
            in_neg = compound_name in matched_compounds_neg
            
            if in_pos and in_neg:
                ionization = "Positive / Negative"
                match_status = "✓ Matched"
            elif in_pos:
                ionization = "Positive"
                match_status = "✓ Matched"
            elif in_neg:
                ionization = "Negative"
                match_status = "✓ Matched"
            else:
                ionization = ""
                match_status = "✗ Not Matched"
            
            summary_data_consolidated.append({
                'Compound': compound_name,
                'Match_Status': match_status,
                'Ionization': ionization,
                'CAS': target.get('CAS', '') if target is not None else '',
                'Formula': target.get(TARGETS_FORMULA_COL, '') if target is not None else '',
                'SMILES': target.get(TARGETS_SMILES_COL, '') if target is not None else ''
            })
        
        match_summary_consolidated = pd.DataFrame(summary_data_consolidated)
        
        # Keep separate summaries for backward compatibility (if needed)
        match_summary_pos = pd.DataFrame()
        match_summary_neg = pd.DataFrame()
        
        # 7. Generate Skyline output if MGF files provided
        skyline_pos = pd.DataFrame()
        skyline_neg = pd.DataFrame()
        skyline_unmatched_pos = pd.DataFrame()
        skyline_unmatched_neg = pd.DataFrame()

        if generate_skyline:
            if mgf_pos_files and not final_pos.empty:
                try:
                    with st.spinner("📊 Extracting Skyline transitions from ESI+ MGF..."):
                        skyline_pos, skyline_unmatched_pos = extract_skyline_transitions_from_mgf(mgf_pos_files, final_pos, "Positive", fragment_dedup_ppm, compound_match_ppm_tolerance)
                        if not skyline_pos.empty:
                            st.write(f"✓ Extracted {len(skyline_pos)} transitions from ESI+ MGF")
                        else:
                            st.warning("⚠️ No transitions extracted from ESI+ MGF")
                except Exception as e:
                    st.error(f"❌ Error processing ESI+ MGF: {str(e)}")
            elif not mgf_pos_files and polarity_mode in ["Positive & Negative", "Positive Only"] and not final_pos.empty:
                st.info("ℹ️ Skyline requested but no ESI+ MGF file was uploaded — ESI+ Skyline list will be skipped.")

            if mgf_neg_files and not final_neg.empty:
                try:
                    with st.spinner("📊 Extracting Skyline transitions from ESI− MGF..."):
                        skyline_neg, skyline_unmatched_neg = extract_skyline_transitions_from_mgf(mgf_neg_files, final_neg, "Negative", fragment_dedup_ppm, compound_match_ppm_tolerance)
                        if not skyline_neg.empty:
                            st.write(f"✓ Extracted {len(skyline_neg)} transitions from ESI− MGF")
                        else:
                            st.warning("⚠️ No transitions extracted from ESI− MGF")
                except Exception as e:
                    st.error(f"❌ Error processing ESI− MGF: {str(e)}")
            elif not mgf_neg_files and polarity_mode in ["Positive & Negative", "Negative Only"] and not final_neg.empty:
                st.info("ℹ️ Skyline requested but no ESI− MGF file was uploaded — ESI− Skyline list will be skipped.")

            if skyline_pos.empty and skyline_neg.empty:
                st.warning(f"⚠️ **No Skyline transitions were generated**, so no Skyline_Transition-List file(s) will appear in the ZIP. This happens when: (1) no MGF file was uploaded for a polarity, or (2) none of the MGF spectra's precursor m/z fell within ±{compound_match_ppm_tolerance} ppm of a matched target's m/z (and, if RT was available in the MGF, within that target's RT window). Check the debug log above for exact counts.")

        st.session_state['results'] = {
            'pos': final_pos, 'neg': final_neg, 
            'fc_pos': fig_c_pos, 'fc_neg': fig_c_neg,
            'fp_pos': fig_p_pos, 'fp_neg': fig_p_neg,
            'fig_rt_pos': fig_rt_pos, 'fig_rt_neg': fig_rt_neg,
            'fig_xic_pos': fig_xic_pos, 'fig_xic_neg': fig_xic_neg,
            'fig_mz_rt_2d_pos': fig_mz_rt_2d_pos, 'fig_mz_rt_2d_neg': fig_mz_rt_2d_neg,
            'match_summary': match_summary_consolidated,
            'skyline_pos': skyline_pos, 'skyline_neg': skyline_neg,
            'skyline_unmatched_pos': skyline_unmatched_pos, 'skyline_unmatched_neg': skyline_unmatched_neg,
            'mode': polarity_mode,
            'resolution': orbitrap_resolution,
            'it_mode': it_mode,
            'hcd_energies': hcd_energies,
            'xic_ppm': xic_ppm_tolerance,
            'fragment_dedup_ppm': fragment_dedup_ppm,
            'compound_match_ppm_tolerance': compound_match_ppm_tolerance,
            'rt_window_mode': rt_window_mode,
            'rt_margin_min': rt_margin_min,
            'rt_margin_pct': rt_margin_pct
        }
        status.update(label="Evaluation Complete!", state="complete")

def build_compound_selector(df, key_prefix, title="Select Compounds to Include"):
    """
    Creates an interactive compound selector with checkboxes.
    Returns a filtered dataframe based on user selections.
    
    Args:
        df: DataFrame with compounds
        key_prefix: Unique prefix for session state keys (e.g., "pos_g1", "neg_g2")
        title: Display title
    
    Returns:
        Filtered dataframe with only selected compounds
    """
    if df.empty:
        return df
    
    # Initialize session state for this selector if not exists
    selection_key = f"compounds_{key_prefix}"
    current_compounds = df['Compound'].unique()
    if selection_key not in st.session_state:
        st.session_state[selection_key] = {cmp: True for cmp in current_compounds}
    else:
        # The matched compound list can change between runs (different inputs,
        # settings, etc.) — backfill any compound not seen in a prior run so
        # the checkbox lookup below never KeyErrors on a new/renamed compound.
        for cmp in current_compounds:
            if cmp not in st.session_state[selection_key]:
                st.session_state[selection_key][cmp] = True
    
    with st.expander(f"✏️ {title} ({len(df)} compounds)", expanded=False):
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("✓ Select All", key=f"select_all_{key_prefix}"):
                st.session_state[selection_key] = {cmp: True for cmp in df['Compound'].unique()}
                st.rerun()
        with col2:
            if st.button("✗ Deselect All", key=f"deselect_all_{key_prefix}"):
                st.session_state[selection_key] = {cmp: False for cmp in df['Compound'].unique()}
                st.rerun()
        with col3:
            selected_count = sum(st.session_state[selection_key].values())
            st.write(f"**Selected: {selected_count}/{len(df)}**")
        
        # Create checkboxes in columns for better layout
        cols = st.columns(3)
        for idx, compound in enumerate(df['Compound'].unique()):
            col = cols[idx % 3]
            with col:
                st.session_state[selection_key][compound] = st.checkbox(
                    compound,
                    value=st.session_state[selection_key].get(compound, True),
                    key=f"checkbox_{key_prefix}_{compound}"
                )
    
    # Filter dataframe based on selections
    selected_compounds = [cmp for cmp, selected in st.session_state[selection_key].items() if selected]
    filtered_df = df[df['Compound'].isin(selected_compounds)].copy()
    return filtered_df

if 'results' in st.session_state:
    r = st.session_state['results']
    
    # Check if settings changed since last evaluation
    settings_changed = False
    if r.get('resolution') != orbitrap_resolution: settings_changed = True
    if r.get('it_mode') != it_mode: settings_changed = True
    if r.get('rt_window_mode') != rt_window_mode: settings_changed = True
    
    if settings_changed:
        st.warning("⚠️ **Settings Modified:** You have changed the sidebar settings. Click '**▶ Evaluate & Optimize Method**' again to recalculate the cycle times and update the figures.")
    
    st.header("📊 Results & Optimization")
    
    # Info box about multiple datasets
    st.info("""
        **ℹ️ About your results:**
        
        If you uploaded **multiple datasets**, all results have been aggregated:
        - Each dataset was processed independently
        - For compounds found in multiple datasets, the **highest intensity** peak was selected
        - Results from all datasets are combined in the tables below
        - No data mixing occurs — each dataset's data is processed in isolation first, then merged
        """)
    
    st.divider()
    
    # Option to split into multiplex groups (post-run decision)
    st.subheader("Multiplex Splitting Decision")
    split_method = st.checkbox(
        "Split into 2 Multiplex Groups?", 
        value=st.session_state.get('split_method_selected', False),
        help="Divides targets into two separate inclusion lists to drastically improve Points/Peak. Re-run the analysis above to apply this change.",
        key="split_checkbox"
    )
    
    # Update session state with the current checkbox value
    if split_method != st.session_state.get('split_method_selected', False):
        st.session_state['split_method_selected'] = split_method
    
    if split_method:
        st.info("🔄 Click the **Evaluate & Optimize Method** button above to re-run the analysis with Multiplex Splitting enabled.")
    
    st.divider()
    
    # Match Summary Table (Consolidated)
    st.subheader("📋 Compound Match Summary (All Targets)")
    st.markdown("**Complete list of all target compounds from Step 1 with match status for each ionization mode:**")
    
    if 'match_summary' in r and not r['match_summary'].empty:
        col1, col2 = st.columns([4, 1])
        with col1:
            st.dataframe(r['match_summary'], width='stretch', use_container_width=True)
        with col2:
            total_matched = len(r['match_summary'][r['match_summary']['Match_Status'] == '✓ Matched'])
            total_compounds = len(r['match_summary'])
            st.metric("Total Matched", f"{total_matched}/{total_compounds}", f"{100*total_matched/total_compounds:.0f}%")
    
    st.divider()
    st.subheader("📊 Results Summary")
    
    if r['mode'] in ["Positive & Negative", "Positive Only"] and not r['pos'].empty:
        st.markdown("#### 📍 ESI+ (Positive)")
        st.dataframe(r['pos'], width='stretch')
    
    if r['mode'] in ["Positive & Negative", "Negative Only"] and not r['neg'].empty:
        st.markdown("#### 📍 ESI− (Negative)")
        st.dataframe(r['neg'], width='stretch')
    
    st.subheader("📊 Points Per Peak Figures")
    cols = st.columns(2) if r['mode'] == "Positive & Negative" else [st.container()]
    
    if r['mode'] in ["Positive & Negative", "Positive Only"] and r['fp_pos']:
        with cols[0]:
            st.markdown("**ESI+ Points/Peak**")
            for grp, fig in r['fp_pos']: st.pyplot(fig)
    
    if r['mode'] in ["Positive & Negative", "Negative Only"] and r['fp_neg']:
        with cols[1] if r['mode'] == "Positive & Negative" else st.container():
            st.markdown("**ESI− Points/Peak**")
            for grp, fig in r['fp_neg']: st.pyplot(fig)
        
    st.subheader("📈 Concurrency Density Plots")
    cols = st.columns(2) if r['mode'] == "Positive & Negative" else [st.container()]
    
    if r['mode'] in ["Positive & Negative", "Positive Only"] and r['fc_pos']:
        with cols[0]:
            st.markdown("**ESI+ Concurrency**")
            for grp, fig in r['fc_pos']: st.pyplot(fig)
    
    if r['mode'] in ["Positive & Negative", "Negative Only"] and r['fc_neg']:
        with cols[1] if r['mode'] == "Positive & Negative" else st.container():
            st.markdown("**ESI− Concurrency**")
            for grp, fig in r['fc_neg']: st.pyplot(fig)
    
    st.subheader("⏱️ Retention Time Alignment")
    cols = st.columns(2) if r['mode'] == "Positive & Negative" else [st.container()]
    
    if r['mode'] in ["Positive & Negative", "Positive Only"] and r['fig_rt_pos']:
        with cols[0]:
            st.markdown("**ESI+ RT Windows**")
            # Handle both old format (single Figure) and new format (list of tuples)
            if isinstance(r['fig_rt_pos'], list):
                for grp, fig in r['fig_rt_pos']: st.pyplot(fig)
            else:
                st.pyplot(r['fig_rt_pos'])
    
    if r['mode'] in ["Positive & Negative", "Negative Only"] and r['fig_rt_neg']:
        with cols[1] if r['mode'] == "Positive & Negative" else st.container():
            st.markdown("**ESI− RT Windows**")
            # Handle both old format (single Figure) and new format (list of tuples)
            if isinstance(r['fig_rt_neg'], list):
                for grp, fig in r['fig_rt_neg']: st.pyplot(fig)
            else:
                st.pyplot(r['fig_rt_neg'])
    
    st.divider()
    
    # m/z vs Retention Time Visualizations
    if r.get('fig_mz_rt_2d_pos') or r.get('fig_mz_rt_2d_neg'):
        st.subheader("📈 m/z vs Retention Time")
        cols = st.columns(2) if r['mode'] == "Positive & Negative" and r.get('fig_mz_rt_2d_pos') and r.get('fig_mz_rt_2d_neg') else [st.container()]
        
        if r.get('fig_mz_rt_2d_pos'):
            with cols[0] if r['mode'] == "Positive & Negative" else st.container():
                st.markdown("**ESI+ m/z vs RT**")
                st.plotly_chart(r['fig_mz_rt_2d_pos'], use_container_width=True)
        
        if r.get('fig_mz_rt_2d_neg'):
            with cols[1] if r['mode'] == "Positive & Negative" else st.container():
                st.markdown("**ESI− m/z vs RT**")
                st.plotly_chart(r['fig_mz_rt_2d_neg'], use_container_width=True)
    
    st.divider()
    
    if r.get('fig_xic_pos') or r.get('fig_xic_neg'):
        st.subheader("🗂️ Extracted Ion Chromatograms (XICs)")
        cols = st.columns(2) if r['mode'] == "Positive & Negative" and r.get('fig_xic_pos') and r.get('fig_xic_neg') else [st.container()]

        # Handle both old format (single Figure, from a cached pre-pagination run)
        # and new format (list of Figures, one per page)
        fig_xic_pos_list = r['fig_xic_pos'] if isinstance(r.get('fig_xic_pos'), list) else ([r['fig_xic_pos']] if r.get('fig_xic_pos') else [])
        fig_xic_neg_list = r['fig_xic_neg'] if isinstance(r.get('fig_xic_neg'), list) else ([r['fig_xic_neg']] if r.get('fig_xic_neg') else [])

        if r.get('fig_xic_pos') is not None and not isinstance(r.get('fig_xic_pos'), list):
            st.warning("⚠️ This ESI+ XIC figure is from an older run (capped at ~80 compounds, no pagination). Click **'▶ Evaluate & Optimize Method'** again to regenerate it with full pagination.")
        if r.get('fig_xic_neg') is not None and not isinstance(r.get('fig_xic_neg'), list):
            st.warning("⚠️ This ESI− XIC figure is from an older run (capped at ~80 compounds, no pagination). Click **'▶ Evaluate & Optimize Method'** again to regenerate it with full pagination.")

        if fig_xic_pos_list:
            with cols[0] if r['mode'] == "Positive & Negative" else st.container():
                st.markdown(f"**ESI+ XICs from LC-MS Raw Data** ({len(fig_xic_pos_list)} page(s))")
                for page_idx, fig in enumerate(fig_xic_pos_list, start=1):
                    if len(fig_xic_pos_list) > 1:
                        st.caption(f"Page {page_idx}/{len(fig_xic_pos_list)}")
                    st.pyplot(fig)

        if fig_xic_neg_list:
            with cols[1] if r['mode'] == "Positive & Negative" else st.container():
                st.markdown(f"**ESI− XICs from LC-MS Raw Data** ({len(fig_xic_neg_list)} page(s))")
                for page_idx, fig in enumerate(fig_xic_neg_list, start=1):
                    if len(fig_xic_neg_list) > 1:
                        st.caption(f"Page {page_idx}/{len(fig_xic_neg_list)}")
                    st.pyplot(fig)
    else:
        st.info("ℹ️ No XIC figures available. Upload mzML files for chromatogram visualization.")
    
    st.divider()
    
    # Skyline Output Section
    has_skyline_content = (not r['skyline_pos'].empty or not r['skyline_neg'].empty or
                            not r.get('skyline_unmatched_pos', pd.DataFrame()).empty or
                            not r.get('skyline_unmatched_neg', pd.DataFrame()).empty)
    if has_skyline_content:
        st.subheader("🎯 Skyline Mass List Table")
        st.markdown("**Formatted MS/MS transitions for direct import into Skyline. Includes 2 most abundant fragments per precursor.**")
        
        # Skyline Match Summary Metrics
        col1, col2 = st.columns([1, 1]) if r['mode'] == "Positive & Negative" and not r['skyline_pos'].empty and not r['skyline_neg'].empty else [st.container()]
        
        if not r['skyline_pos'].empty:
            with col1 if r['mode'] == "Positive & Negative" else st.container():
                skyline_precursors_pos = r['skyline_pos']['Precursor Name'].nunique() if 'Precursor Name' in r['skyline_pos'].columns else len(r['skyline_pos'])
                final_pos_count = len(r['pos']) if not r['pos'].empty else 0
                st.metric("ESI+ Matched Precursors", f"{skyline_precursors_pos}/{final_pos_count}", f"{100*skyline_precursors_pos/final_pos_count:.0f}%" if final_pos_count > 0 else "0%")
        
        if not r['skyline_neg'].empty:
            with col2 if r['mode'] == "Positive & Negative" else st.container():
                skyline_precursors_neg = r['skyline_neg']['Precursor Name'].nunique() if 'Precursor Name' in r['skyline_neg'].columns else len(r['skyline_neg'])
                final_neg_count = len(r['neg']) if not r['neg'].empty else 0
                st.metric("ESI− Matched Precursors", f"{skyline_precursors_neg}/{final_neg_count}", f"{100*skyline_precursors_neg/final_neg_count:.0f}%" if final_neg_count > 0 else "0%")
        
        if not r['skyline_pos'].empty:
            st.markdown("#### 📍 ESI+ (Positive)")
            st.dataframe(r['skyline_pos'], width='stretch', use_container_width=True)

            # Download Skyline CSV
            skyline_csv_pos = r['skyline_pos'].to_csv(index=False)
            st.download_button(
                label="⬇️ Download Skyline_Transition-List_ESI-pos",
                data=skyline_csv_pos,
                file_name="Skyline_Transition-List_ESI-pos.csv",
                mime="text/csv",
                key="download_skyline_pos"
            )

        unmatched_pos = r.get('skyline_unmatched_pos', pd.DataFrame())
        if not unmatched_pos.empty:
            st.markdown(f"##### ⚠️ ESI+ Unmatched Compounds ({len(unmatched_pos)})")
            st.dataframe(unmatched_pos, width='stretch', use_container_width=True)
            unmatched_csv_pos = unmatched_pos.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Skyline_Unmatched-Compounds_ESI-pos",
                data=unmatched_csv_pos,
                file_name="Skyline_Unmatched-Compounds_ESI-pos.csv",
                mime="text/csv",
                key="download_skyline_unmatched_pos"
            )

        if not r['skyline_neg'].empty:
            st.markdown("#### 📍 ESI− (Negative)")
            st.dataframe(r['skyline_neg'], width='stretch', use_container_width=True)

            # Download Skyline CSV
            skyline_csv_neg = r['skyline_neg'].to_csv(index=False)
            st.download_button(
                label="⬇️ Download Skyline_Transition-List_ESI-neg",
                data=skyline_csv_neg,
                file_name="Skyline_Transition-List_ESI-neg.csv",
                mime="text/csv",
                key="download_skyline_neg"
            )

        unmatched_neg = r.get('skyline_unmatched_neg', pd.DataFrame())
        if not unmatched_neg.empty:
            st.markdown(f"##### ⚠️ ESI− Unmatched Compounds ({len(unmatched_neg)})")
            st.dataframe(unmatched_neg, width='stretch', use_container_width=True)
            unmatched_csv_neg = unmatched_neg.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Skyline_Unmatched-Compounds_ESI-neg",
                data=unmatched_csv_neg,
                file_name="Skyline_Unmatched-Compounds_ESI-neg.csv",
                mime="text/csv",
                key="download_skyline_unmatched_neg"
            )

        st.divider()
    
    # Inclusion List / MS Export
    st.subheader("📋 Inclusion List for MS Acquisition")
    st.markdown("""
    The tables below show your final PRM targets organized by polarity and multiplex group.  
    These compounds are ready to be imported into your mass spectrometry software (e.g., Thermo Exploris, Skyline).
    """)
    
    # Check if multiplex splitting was used
    has_multiplex = 'Multiplex_Group' in r['pos'].columns if not r['pos'].empty else 'Multiplex_Group' in r['neg'].columns if not r['neg'].empty else False
    
    if has_multiplex and (not r['pos'].empty or not r['neg'].empty) and split_method:
        # MULTIPLEX MODE: Show separate groups
        
        # ESI+ Groups
        if r['mode'] in ["Positive & Negative", "Positive Only"] and not r['pos'].empty:
            st.markdown("### ESI+ (Positive Mode)")
            pos_group1_orig = r['pos'][r['pos']['Multiplex_Group'] == 1].copy()
            pos_group2_orig = r['pos'][r['pos']['Multiplex_Group'] == 2].copy()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔵 Group 1 (Inclusion List 1)")
                # Interactive selector for Group 1
                pos_group1 = build_compound_selector(pos_group1_orig, "pos_g1", "Select ESI+ Group 1 Compounds")
                
                if not pos_group1.empty:
                    cols_to_use = ['Compound', 'm/z', 'z', 't start (min)', 't stop (min)']
                    if 'Formula' in pos_group1.columns:
                        cols_to_use.insert(1, 'Formula')
                    if 'Adduct' in pos_group1.columns:
                        cols_to_use.insert(2 if 'Formula' in cols_to_use else 1, 'Adduct')
                    st.dataframe(pos_group1[cols_to_use], width='stretch')
                    
                    # Download Group 1
                    thermo_csv_pos_g1 = create_thermo_fisher_csv(pos_group1, r['rt_window_mode'])
                    st.download_button(
                        label="⬇️ Download Group 1 (Exploris Format)",
                        data=thermo_csv_pos_g1,
                        file_name="Exploris_Inclusion-List_ESI-pos_Group1.csv",
                        mime="text/csv",
                        key="download_pos_g1_thermo"
                    )
                else:
                    st.info("No targets in Group 1")
            
            with col2:
                st.markdown("#### 🟠 Group 2 (Inclusion List 2)")
                # Interactive selector for Group 2
                pos_group2 = build_compound_selector(pos_group2_orig, "pos_g2", "Select ESI+ Group 2 Compounds")
                
                if not pos_group2.empty:
                    cols_to_use = ['Compound', 'm/z', 'z', 't start (min)', 't stop (min)']
                    if 'Formula' in pos_group2.columns:
                        cols_to_use.insert(1, 'Formula')
                    if 'Adduct' in pos_group2.columns:
                        cols_to_use.insert(2 if 'Formula' in cols_to_use else 1, 'Adduct')
                    st.dataframe(pos_group2[cols_to_use], width='stretch')
                    
                    # Download Group 2
                    thermo_csv_pos_g2 = create_thermo_fisher_csv(pos_group2, r['rt_window_mode'])
                    st.download_button(
                        label="⬇️ Download Group 2 (Exploris Format)",
                        data=thermo_csv_pos_g2,
                        file_name="Exploris_Inclusion-List_ESI-pos_Group2.csv",
                        mime="text/csv",
                        key="download_pos_g2_thermo"
                    )
                else:
                    st.info("No targets in Group 2")
        
        st.divider()
        
        # ESI- Groups
        if r['mode'] in ["Positive & Negative", "Negative Only"] and not r['neg'].empty:
            st.markdown("### ESI− (Negative Mode)")
            neg_group1_orig = r['neg'][r['neg']['Multiplex_Group'] == 1].copy()
            neg_group2_orig = r['neg'][r['neg']['Multiplex_Group'] == 2].copy()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔵 Group 1 (Inclusion List 1)")
                # Interactive selector for Group 1
                neg_group1 = build_compound_selector(neg_group1_orig, "neg_g1", "Select ESI− Group 1 Compounds")
                
                if not neg_group1.empty:
                    cols_to_use = ['Compound', 'm/z', 'z', 't start (min)', 't stop (min)']
                    if 'Formula' in neg_group1.columns:
                        cols_to_use.insert(1, 'Formula')
                    if 'Adduct' in neg_group1.columns:
                        cols_to_use.insert(2 if 'Formula' in cols_to_use else 1, 'Adduct')
                    st.dataframe(neg_group1[cols_to_use], width='stretch')
                    
                    # Download Group 1
                    thermo_csv_neg_g1 = create_thermo_fisher_csv(neg_group1, r['rt_window_mode'])
                    st.download_button(
                        label="⬇️ Download Group 1 (Exploris Format)",
                        data=thermo_csv_neg_g1,
                        file_name="Exploris_Inclusion-List_ESI-neg_Group1.csv",
                        mime="text/csv",
                        key="download_neg_g1_thermo"
                    )
                else:
                    st.info("No targets in Group 1")
            
            with col2:
                st.markdown("#### 🟠 Group 2 (Inclusion List 2)")
                # Interactive selector for Group 2
                neg_group2 = build_compound_selector(neg_group2_orig, "neg_g2", "Select ESI− Group 2 Compounds")
                
                if not neg_group2.empty:
                    cols_to_use = ['Compound', 'm/z', 'z', 't start (min)', 't stop (min)']
                    if 'Formula' in neg_group2.columns:
                        cols_to_use.insert(1, 'Formula')
                    if 'Adduct' in neg_group2.columns:
                        cols_to_use.insert(2 if 'Formula' in cols_to_use else 1, 'Adduct')
                    st.dataframe(neg_group2[cols_to_use], width='stretch')
                    
                    # Download Group 2
                    thermo_csv_neg_g2 = create_thermo_fisher_csv(neg_group2, r['rt_window_mode'])
                    st.download_button(
                        label="⬇️ Download Group 2 (Exploris Format)",
                        data=thermo_csv_neg_g2,
                        file_name="Exploris_Inclusion-List_ESI-neg_Group2.csv",
                        mime="text/csv",
                        key="download_neg_g2_thermo"
                    )
                else:
                    st.info("No targets in Group 2")
    else:
        # NON-MULTIPLEX MODE: Show combined lists
        col1, col2 = st.columns(2)
        
        if r['mode'] in ["Positive & Negative", "Positive Only"] and not r['pos'].empty:
            with col1:
                st.markdown("#### ESI+ Inclusion List")
                # Interactive selector for positive
                pos_filtered = build_compound_selector(r['pos'], "pos_single", "Select ESI+ Compounds")
                
                if not pos_filtered.empty:
                    cols_to_use = ['Compound', 'm/z', 'z', 't start (min)', 't stop (min)']
                    # Add Formula and Adduct if they exist
                    if 'Formula' in pos_filtered.columns:
                        cols_to_use.insert(1, 'Formula')
                    if 'Adduct' in pos_filtered.columns:
                        cols_to_use.insert(2 if 'Formula' in cols_to_use else 1, 'Adduct')
                    st.dataframe(pos_filtered[cols_to_use], width='stretch')
                    
                    # Thermo Fisher format download
                    thermo_csv_pos = create_thermo_fisher_csv(pos_filtered, r['rt_window_mode'])
                    st.download_button(
                        label="⬇️ Download ESI+ (Exploris Format)",
                        data=thermo_csv_pos,
                        file_name="Exploris_Inclusion-List_ESI-pos.csv",
                        mime="text/csv",
                        key="download_pos_thermo"
                    )
                else:
                    st.info("No compounds selected for ESI+")
        
        if r['mode'] in ["Positive & Negative", "Negative Only"] and not r['neg'].empty:
            with col2 if r['mode'] == "Positive & Negative" else st.container():
                st.markdown("#### ESI− Inclusion List")
                # Interactive selector for negative
                neg_filtered = build_compound_selector(r['neg'], "neg_single", "Select ESI− Compounds")
                
                if not neg_filtered.empty:
                    cols_to_use = ['Compound', 'm/z', 'z', 't start (min)', 't stop (min)']
                    # Add Formula and Adduct if they exist
                    if 'Formula' in neg_filtered.columns:
                        cols_to_use.insert(1, 'Formula')
                    if 'Adduct' in neg_filtered.columns:
                        cols_to_use.insert(2 if 'Formula' in cols_to_use else 1, 'Adduct')
                    st.dataframe(neg_filtered[cols_to_use], width='stretch')
                    
                    # Thermo Fisher format download
                    thermo_csv_neg = create_thermo_fisher_csv(neg_filtered, r['rt_window_mode'])
                    st.download_button(
                        label="⬇️ Download ESI− (Exploris Format)",
                        data=thermo_csv_neg,
                        file_name="Exploris_Inclusion-List_ESI-neg.csv",
                        mime="text/csv",
                        key="download_neg_thermo"
                    )
                else:
                    st.info("No compounds selected for ESI−")
    
    st.divider()
    
    # Download button
    st.subheader("💾 Download Results")
    zip_data = create_results_zip(r, r['rt_window_mode'])
    st.download_button(
        label="⬇️ Download All Results (ZIP)",
        data=zip_data,
        file_name="PRM_Method_Results.zip",
        mime="application/zip",
        help="Downloads all results tables (CSV) and figures (PNG) as a zip file"
    )
# 🔬 GNPS2-Quant — PRM Method Generator

A Streamlit web application that converts **GNPS2** spectral library matches and **MZmine 3** feature tables into **Parallel Reaction Monitoring (PRM) inclusion lists** ready for import into Thermo Xcalibur on the **Thermo Exploris 480**.

---

## Features

- **Multi-strategy compound matching** — matches targets by CAS number, compound name, SMILES, or molecular formula against GNPS2 library hits
- **Cross-polarity conflict resolution** — when a compound is detected in both ESI+ and ESI−, the higher-intensity polarity is automatically selected
- **Flexible RT windowing** — choose between a composite range across all GNPS hits ± margin, or a symmetric window around the peak RT
- **Concurrency analysis** — plots the number of simultaneously active PRM targets over the chromatographic run and estimates scan cycle time and data points per peak
- **XIC chromatograms** — optional TIC and extracted ion chromatogram (XIC) plots from raw `.mzML` files
- **RT alignment figures** — horizontal bar chart of each compound's RT window with peak RT markers
- **Interactive output editing** — check/uncheck individual compounds before export
- **One-click ZIP download** — all CSVs and SVG figures bundled into a single archive

---

## Requirements

- Python **3.9 or higher**
- The following Python packages:

| Package | Purpose |
|---|---|
| `streamlit` | Web app framework |
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical computation |
| `matplotlib` | Concurrency, XIC, and RT figures |
| `pyteomics` | Parsing `.mzML` raw data files |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/gnps2-quant.git
cd gnps2-quant
```

> Replace `your-username/gnps2-quant` with your actual GitHub repository URL.

### 2. Create and activate a virtual environment (recommended)

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the App

```bash
streamlit run app.py
```

Your browser should open automatically at `http://localhost:8501`. If it doesn't, navigate there manually.

---

## Input Files

### Required (5 files)

| File | Format | Description |
|---|---|---|
| **Compounds.csv** | CSV | Your target compound list (see format below) |
| **GNPS2 Results — ESI+** | CSV | GNPS2 spectral library search results for positive-mode data |
| **GNPS2 Results — ESI−** | CSV | GNPS2 spectral library search results for negative-mode data |
| **MZmine 3 Features — ESI+** | CSV | MZmine 3 feature table for positive-mode data |
| **MZmine 3 Features — ESI−** | CSV | MZmine 3 feature table for negative-mode data |

### Optional (for chromatogram plots)

| File | Format | Description |
|---|---|---|
| **LC-MS Raw Data — ESI+** | mzML | Raw positive-mode data for TIC/XIC extraction |
| **LC-MS Raw Data — ESI−** | mzML | Raw negative-mode data for TIC/XIC extraction |

---

### Compounds.csv Format

Your target list should contain at least a `Compound` column. CAS, SMILES, and Formula columns improve matching accuracy.

```
Compound,CAS,SMILES,Formula
Caffeine,58-08-2,Cn1cnc2c1c(=O)n(c(=o)n2C)C,C8H10N4O2
Aspirin,50-78-2,CC(=O)Oc1ccccc1C(O)=O,C9H8O4
```

**Default column names** (all configurable in the sidebar):

| Column | Default Header |
|---|---|
| Compound name | `Compound` |
| CAS number | `CAS` |
| SMILES | `SMILES` |
| Molecular formula | `Formula` |

---

### GNPS2 Results Format

Standard GNPS2 `.csv` output. Default expected columns:

| Column | Default Header |
|---|---|
| Compound name | `Compound_Name` |
| Scan number | `#Scan#` |
| Molecular formula | `molecular_formula` |
| Adduct | `Adduct` |
| CAS number | `CAS_Number` |
| SMILES | `Smiles` |

---

### MZmine 3 Feature Table Format

Standard MZmine 3 feature table export. Default expected columns:

| Column | Default Header |
|---|---|
| Feature ID | `id` |
| m/z | `mz` |
| Retention time | `rt` |
| RT range start | `rt_range:min` |
| RT range end | `rt_range:max` |
| Peak height | `height` |
| Charge | `charge` |

> **Note:** All column names are fully customisable via **⚙️ Instrument Settings → 🔧 Advanced: Column Mappings** in the sidebar.

---

## Sidebar Settings

| Setting | Description |
|---|---|
| **HCD Collision Energies (%)** | Comma-separated NCE values applied to all targets (e.g. `25,35,45`) |
| **XIC m/z Tolerance (ppm)** | Mass accuracy window used when extracting XICs from mzML |
| **Expected Peak Width at Base (sec)** | Used to estimate data points per chromatographic peak |
| **Orbitrap Resolution** | Sets estimated scan cycle time (15k / 30k / 60k / 120k) |
| **AGC Target (%)** | Logged in reports; does not affect m/z or RT calculations |
| **RT Window Mode** | `Composite range ± margin` spans the full RT range of all GNPS hits; `Symmetric around peak RT` uses peak RT ± margin |
| **RT Margin (min)** | Buffer added/subtracted at each edge of the RT window |

---

## Output Tabs

After clicking **▶ Run Pipeline**, results appear in the following tabs:

| Tab | Contents |
|---|---|
| **Match Report** | Per-compound match status (✓ / ✗), assigned polarity, GNPS match name, concurrency, and estimated points/peak |
| **PRM List — ESI+** | Editable inclusion list for positive mode; check/uncheck rows before export |
| **PRM List — ESI−** | Editable inclusion list for negative mode |
| **Intermediate List** | Full list before Formula/Adduct columns are blanked, useful for review |
| **Concurrency** | Density plots showing simultaneous target counts over the run |
| **Chromatograms** | TIC + overlaid XIC + individual XIC panels (only if mzML files were uploaded) |
| **RT Alignment** | Horizontal bar chart of RT windows with peak RT markers |
| **⬇ Downloads** | ZIP of all outputs, or individual CSV/SVG downloads |

---

## Output Files

| File | Description |
|---|---|
| `Output_Match_Report.csv` | Compound match summary |
| `Output_Intermediate_Mass-List-Table.csv` | Full merged list with all metadata |
| `Output_Mass-List-Table_pos.csv` | Xcalibur-ready PRM list for ESI+ |
| `Output_Mass-List-Table_neg.csv` | Xcalibur-ready PRM list for ESI− |
| `Output_Concurrency_pos/neg.svg` | Concurrency density figures |
| `Output_Chromatograms_pos/neg.svg` | TIC + XIC figures (if mzML provided) |
| `Output_Retention-Time-Alignment_pos/neg.svg` | RT window alignment figures |

---

## Troubleshooting

**Compounds are not matching**
- Check that your column names match the defaults (or update them in **Advanced: Column Mappings**)
- Matching works on CAS number first, then compound name, then SMILES, then formula — providing more identifiers improves match rates
- CAS numbers are normalised automatically (dashes and spaces are stripped before comparison)

**`ModuleNotFoundError` on startup**
- Make sure your virtual environment is activated and all dependencies are installed (`pip install streamlit pandas numpy matplotlib pyteomics`)

**mzML file not loading**
- Ensure the file uses the standard `.mzML` extension (case-insensitive)
- Very large mzML files may take a minute to parse; the status bar will update as it progresses

---

## License

This project is provided as-is. See `LICENSE` for details (if included in the repository).

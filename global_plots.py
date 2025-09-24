import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.ticker as mticker

# Macros & Config
PORT_COL = "voyage_itinerary__imp_principal_port_slave_dis__name"
SLAVES_COL = "voyage_slaves_numbers__imp_total_num_slaves_disembarked"
TARGET_PORT_RAW = "new orleans"  # comparison done in lowercase
TOP_N_PORTS = 20
INVALID_PORT_TOKENS = {'', 'nan', '0'}

# Load datasets with dtype handling for mixed columns
_intra_header = pd.read_csv('data/intra-american.csv', nrows=0)
mixed_idx = [17, 21, 39]  # indices known to need string coercion
mixed_cols = [_intra_header.columns[i] for i in mixed_idx]
dtype_map = {c: 'string' for c in mixed_cols}

intra_american_df = pd.read_csv('data/intra-american.csv', dtype=dtype_map, low_memory=False)
trans_atlantic_df = pd.read_csv('data/trans-atlantic.csv')

intra_american_df['dataset_source'] = 'intra-american'
trans_atlantic_df['dataset_source'] = 'trans-atlantic'

combined_df = pd.concat([intra_american_df, trans_atlantic_df], ignore_index=True)

# Clean port names on a copy 
filtered_df = combined_df.copy()
filtered_df[PORT_COL] = (
    filtered_df[PORT_COL]
    .astype('string')
    .str.replace(r",\s*place unspecified$", ", Unknown", case=False, regex=True)
    .str.strip()
    .str.replace(r"\s{2,}", " ", regex=True)
)

# Replace invalid with No Data (do not drop)
mask_invalid = filtered_df[PORT_COL].isna() | filtered_df[PORT_COL].str.lower().isin(INVALID_PORT_TOKENS)
filtered_df.loc[mask_invalid, PORT_COL] = "No Data"

# Voyage count aggregation
voyage_counts = (
    filtered_df
    .groupby(['dataset_source', PORT_COL])
    .size()
    .reset_index(name='voyage_count')
)

# Determine top ports by voyage count
top_ports = (
    voyage_counts
    .groupby(PORT_COL)['voyage_count']
    .sum()
    .sort_values(ascending=False)
    .head(TOP_N_PORTS)
    .index
)

# Force inclusion of New Orleans if present but excluded
target_exists_mask = filtered_df[PORT_COL].str.lower() == TARGET_PORT_RAW
if target_exists_mask.any():
    target_port_cased = filtered_df.loc[target_exists_mask, PORT_COL].iloc[0]
    if target_port_cased not in top_ports:
        top_ports = list(top_ports) + [target_port_cased]
top_voyage_counts = voyage_counts[voyage_counts[PORT_COL].isin(top_ports)]

voyage_order = (
    top_voyage_counts
    .groupby(PORT_COL)['voyage_count']
    .sum()
    .sort_values(ascending=False)
    .index
    .tolist()
)

hue_order = ['trans-atlantic', 'intra-american']
palette_first = {ds: col for ds, col in zip(hue_order, sns.color_palette('rocket', n_colors=len(hue_order)))}

# New Orleans voyage counts (cleaned)
no_mask_voy = filtered_df[PORT_COL].str.lower() == TARGET_PORT_RAW
new_orleans_voy_counts = (
    filtered_df[no_mask_voy]
    .groupby('dataset_source')
    .size()
    .reindex(hue_order)
    .fillna(0)
    .astype(int)
)

# ------------------ PLOT 1: Voyage Counts ------------------
plt.figure(figsize=(max(10, len(voyage_order) * 0.6), 8))
ax1 = sns.barplot(
    data=top_voyage_counts,
    x=PORT_COL,
    y='voyage_count',
    hue='dataset_source',
    hue_order=hue_order,
    order=voyage_order,
    edgecolor='black',
    palette=[palette_first[h] for h in hue_order]
)
ax1.set_yscale("log")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{int(y):,}" if y >= 1 else f"{y:g}"))
plt.title(f"Top {TOP_N_PORTS} Disembarkment Ports by Voyage Count (Year 1514-1887)")
plt.xlabel("Disembarkment Port")
plt.ylabel("Voyage Count (log scale)")
plt.xticks(rotation=45, ha='right')

# Horizontal lines for New Orleans voyage counts
line_color_map_voy = {
    'trans-atlantic': 'blue',
    'intra-american': 'red'
}
line_handles_1 = []
line_labels_1 = []
for ds in hue_order:
    val = new_orleans_voy_counts.loc[ds]
    lc = line_color_map_voy[ds]
    if val > 0:
        ax1.axhline(val, color=lc, linestyle='--', linewidth=1.4, alpha=0.9)
    line_handles_1.append(Line2D([0], [0], color=lc, linestyle='--', linewidth=1.4))
    line_labels_1.append(f"{ds.replace('-', ' ').title()} New Orleans: {val:,}")

bar_handles_1 = [Patch(facecolor=palette_first[ds], edgecolor='black') for ds in hue_order]
bar_labels_1 = [f"{ds.replace('-', ' ').title()} (bars)" for ds in hue_order]

ax1.legend(bar_handles_1 + line_handles_1,
           bar_labels_1 + line_labels_1,
           title="Legend",
           frameon=True)
plt.tight_layout()
plt.show()

# ------------------ PLOT 2: Enslaved Disembarked ------------------
filtered_df[SLAVES_COL] = pd.to_numeric(filtered_df[SLAVES_COL], errors='coerce')
enslaved_metrics = (
        filtered_df
        .groupby(['dataset_source', PORT_COL])[SLAVES_COL]
        .sum(min_count=1)
        .reset_index(name='enslaved_total')
    )

top_ports_slaves = (
    enslaved_metrics
    .groupby(PORT_COL)['enslaved_total']
    .sum()
    .sort_values(ascending=False)
    .head(TOP_N_PORTS)
    .index
)

# Force include New Orleans if present
target_exists_mask2 = filtered_df[PORT_COL].str.lower() == TARGET_PORT_RAW
if target_exists_mask2.any():
    target_port_cased2 = filtered_df.loc[target_exists_mask2, PORT_COL].iloc[0]
    if target_port_cased2 not in top_ports_slaves:
        top_ports_slaves = list(top_ports_slaves) + [target_port_cased2]

top_enslaved = enslaved_metrics[enslaved_metrics[PORT_COL].isin(top_ports_slaves)]

enslaved_order = (
    top_enslaved
    .groupby(PORT_COL)['enslaved_total']
    .sum()
    .sort_values(ascending=False)
    .index
    .tolist()
)

new_orleans_slave_totals = (
    filtered_df[target_exists_mask2]
    .groupby('dataset_source')[SLAVES_COL]
    .sum()
    .reindex(hue_order)
    .fillna(0)
    .astype(int)
)

# Different palette for second chart
palette_second = {ds: col for ds, col in zip(hue_order, sns.color_palette('mako', n_colors=len(hue_order)))}

# Plot 2
plt.figure(figsize=(max(10, len(enslaved_order) * 0.6), 8))
ax2 = sns.barplot(
    data=top_enslaved,
    x=PORT_COL,
    y='enslaved_total',
    hue='dataset_source',
    hue_order=hue_order,
    order=enslaved_order,
    edgecolor='black',
    palette=[palette_second[h] for h in hue_order]
)
ax2.set_yscale("log")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{int(y):,}" if y >= 1 else f"{y:g}"))
plt.title(f"Top {TOP_N_PORTS} Disembarkment Ports by Total Enslaved Disembarked Including New Orleans (Year 1514-1887)")
plt.xlabel("Disembarkment Port")
plt.ylabel("Total Enslaved Disembarked (log scale)")
plt.xticks(rotation=45, ha='right')

enslaved_line_color_map = {
    'trans-atlantic': 'blue',
    'intra-american': 'red'
}
line_handles_2 = []
line_labels_2 = []
for ds in hue_order:
    val = new_orleans_slave_totals.loc[ds] if target_exists_mask2.any() else 0
    lc = enslaved_line_color_map[ds]
    if val > 0:
        ax2.axhline(val, color=lc, linestyle='--', linewidth=1.4, alpha=0.9)
    line_handles_2.append(Line2D([0], [0], color=lc, linestyle='--', linewidth=1.4))
    line_labels_2.append(f"{ds.replace('-', ' ').title()} New Orleans: {val:,}")

bar_handles_2 = [Patch(facecolor=palette_second[ds], edgecolor='black') for ds in hue_order]
bar_labels_2 = [f"{ds.replace('-', ' ').title()} (bars)" for ds in hue_order]

ax2.legend(bar_handles_2 + line_handles_2,
           bar_labels_2 + line_labels_2,
           title="Legend",
           frameon=True)
plt.tight_layout()
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.ticker as mticker
import matplotlib.cm as cm
import matplotlib.colors as mcolors


# Load the cleaned combined dataset 
df = pd.read_csv("data/new_orleans_combined.csv")
df.dropna()

# ---------------------- LINE PLOT ---------------------------
# ---- Timeline: Voyages Arriving in New Orleans per year ----
year_col = "voyage_dates__imp_arrival_at_port_of_dis_sparsedate__year"
voyage_per_year = df.groupby(year_col).size().reset_index(name='voyage_count')

# Create color mapping based on voyage counts
voyage_counts = voyage_per_year['voyage_count'].values
norm = mcolors.Normalize(vmin=voyage_counts.min(), vmax=voyage_counts.max())
colors = cm.get_cmap('Reds')(norm(voyage_counts))

plt.figure(figsize=(12, 8))

# Create scatter plot with color-coded points
scatter = plt.scatter(voyage_per_year[year_col], 
                     voyage_per_year['voyage_count'],
                     c=voyage_counts,
                     cmap='Reds',
                     s=100,  # Size of markers
                     edgecolors='darkred',
                     linewidth=1.5,
                     alpha=0.8)

# Add connecting line
plt.plot(voyage_per_year[year_col], 
         voyage_per_year['voyage_count'], 
         color='darkred', 
         linewidth=2,
         alpha=0.6,
         zorder=1)  # Put line behind markers

# Add colorbar
cbar = plt.colorbar(scatter, shrink=0.8)
cbar.set_label('Number of Voyages', rotation=270, labelpad=20, fontsize=12)
cbar.ax.tick_params(labelsize=10)

# Enhance formatting
plt.title("Timeline: Slave Trade Voyages to New Orleans\n(Color intensity reflects voyage frequency)", 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel("Year", fontsize=12, fontweight='bold')
plt.ylabel("Number of Voyages", fontsize=12, fontweight='bold')

# Add grid with better styling
plt.grid(True, alpha=0.3, linestyle='--')

# Add statistical annotations
max_voyages = voyage_counts.max()
max_year = voyage_per_year.loc[voyage_per_year['voyage_count'].idxmax(), year_col]
avg_voyages = voyage_counts.mean()

plt.axhline(y=avg_voyages, color='blue', linestyle='--', alpha=0.7, 
           label=f'Average: {avg_voyages:.1f} voyages/year')
plt.axhline(y=max_voyages, color='red', linestyle=':', alpha=0.7,
           label=f'Peak: {max_voyages} voyages in {int(max_year)}')

plt.legend(loc='upper left', framealpha=0.9)

# Improve layout
plt.tight_layout()
plt.show()

# ---------------------- STACKED BAR CHART ----------------------
# ------- Number of enslaved people disembarked per Year --------
year_col = "voyage_dates__imp_arrival_at_port_of_dis_sparsedate__year"
disembarked_col = "voyage_slaves_numbers__imp_total_num_slaves_disembarked"
source_col = "dataset_source"  

# Retrieve a group by year and sum the disembarked enslaved people
disembarked_per_year = (
    df.groupby([year_col, source_col], as_index=False)[disembarked_col]
        .sum()
        .rename(columns={disembarked_col: 'disembarked_total'})
        .sort_values(by=year_col)
)

# Pivot to get data sources as columns
pivot_data = disembarked_per_year.pivot(
    index=year_col, 
    columns=source_col, 
    values='disembarked_total'
).fillna(0)

# Create the stacked bar plot
fig, ax = plt.subplots(figsize=(14, 8))

# Get unique data sources and assign colors
data_sources = pivot_data.columns.tolist()
colors = cm.get_cmap("berlin")(np.linspace(0, 1, len(data_sources)))

# Create stacked bars 
bottom = np.zeros(len(pivot_data))
bars = []

for i, source in enumerate(data_sources):
    bar = ax.bar(
        pivot_data.index,
        pivot_data[source],
        bottom=bottom,
        label=source,
        color=colors[i],
        edgecolor='black',
        linewidth=0.5
    )
    bars.append(bar)
    bottom += pivot_data[source]

# Plot reference lines 
total_per_year = pivot_data.sum(axis=1)
overall_average = total_per_year.mean()
max_year = total_per_year.idxmax()
max_total = total_per_year.max()
ax.axhline(overall_average, 
           color='crimson', 
           linestyle='--', 
           alpha=0.8,
           label=f'Average: {overall_average:,.0f} enslaved per year')
ax.axhline(max_total, 
           color='turquoise', 
           linestyle='--', 
           alpha=0.8,
           label=f'Maximum: {max_total:,.0f} enslaved in {int(max_year)}')

# Formatting 
ax.set_title("Number of Enslaved People Disembarked in New Orleans per Year by Route", fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel("Year")
ax.set_ylabel("Number of Enslaved People Disembarked")
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max_total * 1.1)  # Set y-max to 110% of the maximum value

# Create custom legend comnbining data sources and reference lines 
source_patches = [Patch(color=colors[i], label=source) 
                  for i, source in enumerate(data_sources)]
reference_lines = [
    Line2D([0], [0], color='red', linestyle='--', alpha=0.8, label=f'Average: {overall_average:,.0f} per year'),
    Line2D([0], [0], color='blue', linestyle='--', alpha=0.8, label=f'Maximum: {max_total:,.0f} in {int(max_year)}')
]
ax.legend(handles=source_patches + reference_lines, 
          loc='upper right',
          bbox_to_anchor=(1.15, 1))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ---------------------- STACKED BAR CHART ----------------------
# ------- Number of enslaved people embarked per Year -----------
year_col = "voyage_dates__imp_arrival_at_port_of_dis_sparsedate__year"
embarked_col = "voyage_slaves_numbers__imp_total_num_slaves_embarked"
source_col = "dataset_source"  

# Retrieve a group by year and sum the embarked enslaved people
embarked_per_year = (
    df.groupby([year_col, source_col], as_index=False)[embarked_col]
        .sum()
        .rename(columns={embarked_col: 'embarked_total'})
        .sort_values(by=year_col)
)

# Pivot to get data sources as columns
pivot_data_embarked = embarked_per_year.pivot(  # Changed variable name to avoid conflicts
    index=year_col, 
    columns=source_col, 
    values='embarked_total'
).fillna(0)

# Create the stacked bar plot
fig, ax = plt.subplots(figsize=(14, 8))

# Get unique data sources and assign colors
data_sources = pivot_data_embarked.columns.tolist()
colors = cm.get_cmap("berlin")(np.linspace(0, 1, len(data_sources)))  # Changed colormap to differentiate from disembarked

# Create stacked bars 
bottom = np.zeros(len(pivot_data_embarked))
bars = []

for i, source in enumerate(data_sources):
    bar = ax.bar(
        pivot_data_embarked.index,
        pivot_data_embarked[source],
        bottom=bottom,
        label=source,
        color=colors[i],
        edgecolor='black',
        linewidth=0.5
    )
    bars.append(bar)
    bottom += pivot_data_embarked[source]

# Plot reference lines 
total_per_year = pivot_data_embarked.sum(axis=1)
overall_average = total_per_year.mean()
max_year = total_per_year.idxmax()
max_total = total_per_year.max()
ax.axhline(overall_average, 
           color='crimson', 
           linestyle='--', 
           alpha=0.8,
           label=f'Average: {overall_average:,.0f} enslaved per year')
ax.axhline(max_total, 
           color='turquoise', 
           linestyle='--', 
           alpha=0.8,
           label=f'Maximum: {max_total:,.0f} enslaved in {int(max_year)}')

# Formatting 
ax.set_title("Number of Enslaved People Embarked for New Orleans per Year by Data Source", fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel("Year")
ax.set_ylabel("Number of Enslaved People Embarked")
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max_total * 1.1)  # Set y-max to 110% of the maximum value

# Create custom legend combining data sources and reference lines
source_patches = [Patch(color=colors[i], label=source)
                  for i, source in enumerate(data_sources)]
reference_lines = [
    Line2D([0], [0], color='crimson', linestyle='--', alpha=0.8, label=f'Average: {overall_average:,.0f} per year'),  # Fixed color
    Line2D([0], [0], color='turquoise', linestyle='--', alpha=0.8, label=f'Maximum: {max_total:,.0f} in {int(max_year)}')  # Fixed color
]
ax.legend(handles=source_patches + reference_lines, 
          loc='upper right',
          bbox_to_anchor=(1.15, 1))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ---------------------- BAR CHART --------------------------
# --- Top 10 Enslavers by Number of Voyages to New Orleans ---
enslaver_col = "enslavers"
main_enslaver_col = "main_enslaver"

# Extract captain's name if present
df[main_enslaver_col] = df[enslaver_col].str.extract(r'[^:]*:\s*([^|]+)', expand=False).str.strip()
df[main_enslaver_col] = df[main_enslaver_col].fillna('Non-Specified')  # Use fillna() instead of replace('')
df[main_enslaver_col] = df[main_enslaver_col].str.strip()

# Count voyages per enslaver
enslaver_counts = df[main_enslaver_col].value_counts()
top10 = enslaver_counts.nlargest(10)

# Create a new column that replaces non-top10 enslavers with "Others"
df["enslaver_grouped"] = df[main_enslaver_col].where(df[main_enslaver_col].isin(top10.index), "Others")

# Count again (top 10 + Others)
enslaver_grouped_counts = df["enslaver_grouped"].value_counts()

# Begin plotting - CORRECTED: use plt.subplots() not plt.plot()
fig, ax = plt.subplots(figsize=(12, 6))

# Get unique names and assign colors
enslaver_names = enslaver_grouped_counts.index.tolist()
colors = cm.get_cmap("rocket")(np.linspace(0, 1, len(enslaver_names)))

# Plot
ax.bar(
    enslaver_grouped_counts.index,
    enslaver_grouped_counts.values,
    color=colors,
    edgecolor='black',
    linewidth=0.5
)
ax.set_title("Top 10 Enslavers by Number of Voyages to New Orleans", fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel("Enslaver")
ax.set_ylabel("Voyage Count")
ax.set_yscale('log')
ax.set_xticklabels(enslaver_grouped_counts.index, rotation=45, ha="right")
ax.grid(axis="y", alpha=0.3)
ax.axhline(enslaver_grouped_counts.mean(), 
                color='crimson',
                linestyle='--',
                alpha=0.8,
                label=f'Average: {enslaver_grouped_counts.mean():.1f} voyages'
    )
ax.legend()
plt.tight_layout()
plt.show()

# --- Top Ships Nation by Number of Voyages to New Orleans ---
ship_col = "voyage_ship__imputed_nationality__name"

# Replace NaN, empty strings, 0 values and Unknown with "Non-Specified"
df[ship_col] = df[ship_col].replace(['', '0', 0, 'Unknown'], 'Non-Specified')
df[ship_col] = df[ship_col].fillna('Non-Specified')
ship_counts = df[ship_col].value_counts()
top10_ships = ship_counts.nlargest(10)
ship_grouped_counts = df[ship_col].where(df[ship_col].isin(top10_ships.index), "Others").value_counts()

# Begin plotting - CORRECTED: use plt.subplots() and create new ax
fig, ax = plt.subplots(figsize=(12, 6))

# Define colors for ships
ship_names = ship_grouped_counts.index.tolist()
colors = cm.get_cmap("viridis")(np.linspace(0, 1, len(ship_names)))

# Plot - CORRECTED: ax.bar() returns bars, not ax
bars = ax.bar(
    ship_grouped_counts.index,
    ship_grouped_counts.values,
    color=colors,
    edgecolor='black',
    linewidth=0.5
)
ax.set_title("Top Ship Nationalities by Number of Voyages to New Orleans", fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel("Ship Nationality")
ax.set_ylabel("Voyage Count")
ax.set_xticklabels(ship_grouped_counts.index, rotation=45, ha="right")
ax.grid(axis="y", alpha=0.3)
ax.axhline(ship_grouped_counts.mean(),
                color='crimson',
                linestyle='--',
                alpha=0.8,
                label=f'Average: {ship_grouped_counts.mean():.1f} voyages'
    )
ax.legend()
plt.tight_layout()
plt.show()

# ---------------------- PIE CHART --------------------------
# --- Demographics: Children, Females, and Males among Enslaved People ---

# Define the relevant columns
male_col = "voyage_slaves_numbers__percentage_male"
female_col = "voyage_slaves_numbers__percentage_female" 
child_col = "voyage_slaves_numbers__percentage_child"

# Calculate weighted averages based on the number of people disembarked
# This gives us a more accurate representation than simple column averages
disembarked_col = "voyage_slaves_numbers__imp_total_num_slaves_disembarked"

# Filter out rows with missing demographic data or zero disembarked
demographic_data = df.dropna(subset=[male_col, female_col, child_col, disembarked_col])
demographic_data = demographic_data[demographic_data[disembarked_col] > 0]

# Calculate weighted averages
total_disembarked = demographic_data[disembarked_col].sum()

weighted_male = (demographic_data[male_col] * demographic_data[disembarked_col]).sum() / total_disembarked
weighted_female = (demographic_data[female_col] * demographic_data[disembarked_col]).sum() / total_disembarked
weighted_child = (demographic_data[child_col] * demographic_data[disembarked_col]).sum() / total_disembarked

# Create the pie chart
fig, ax = plt.subplots(figsize=(10, 8))

# Data for pie chart
labels = ['Males', 'Females', 'Children']
sizes = [weighted_male, weighted_female, weighted_child]
colors = ['steelblue', 'lightcoral', 'gold']
explode = (0.05, 0.05, 0.1)  # Slightly separate the children slice

# Create pie chart
wedges, texts, autotexts = ax.pie(sizes, 
                                  labels=labels, 
                                  colors=colors, 
                                  explode=explode,
                                  autopct='%1.1f%%',
                                  startangle=90,
                                  textprops={'fontsize': 12})

# Enhance the appearance
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

ax.set_title('Demographics of Enslaved People\nDisembarked in New Orleans', 
             fontsize=16, fontweight='bold', pad=20)

# Add a text box with summary statistics
textstr = f'''Total Records: {len(demographic_data):,}
Total Enslaved: {total_disembarked:,.0f}
Average per Voyage: {total_disembarked/len(demographic_data):,.0f}'''

props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()

# --- Voyage Outcomes: Captured, Shipwrecked, Completed, etc. ---
outcome_col = "voyage_outcome__particular_outcome__name"

# Replace NaN, empty strings, and Unknown with "Unknown"
df[outcome_col] = df[outcome_col].replace(['', 'Unknown'], 'Unknown')
df[outcome_col] = df[outcome_col].fillna('Unknown')
outcome_data = df.copy()
outcome_counts = df[outcome_col].value_counts()

# Create the pie chart
fig, ax = plt.subplots(figsize=(16, 8))

# Data for pie chart
labels = outcome_counts.index.tolist()
sizes = outcome_counts.values

# Define colors for different outcomes - match number of categories
num_categories = len(labels)
colors = cm.get_cmap('Set3')(np.linspace(0, 1, num_categories))

# Create explode tuple to match number of categories
explode = tuple(0.1 if i > 0 else 0 for i in range(num_categories))

# Create pie chart without labels on slices
wedges, texts, autotexts = ax.pie(sizes, 
                                  labels=None,  # Remove labels from slices
                                  colors=colors, 
                                  explode=explode,
                                  autopct='%1.1f%%',
                                  startangle=90,
                                  textprops={'fontsize': 12})
                                  # Removed invalid 'location' parameter

# Enhance the appearance of percentage text
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

# Create legend
ax.legend(wedges, labels, 
          title="Voyage Outcomes",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))

ax.set_title('Voyage Outcomes: Slave Trade to New Orleans', 
             fontsize=16, fontweight='bold', pad=20)

# Add a text box with summary statistics
total_voyages = len(outcome_data)
completed_successfully = outcome_counts.iloc[0]  # First item (highest count)
captured_unknown = total_voyages - completed_successfully
success_rate = (completed_successfully / total_voyages) * 100

textstr = f'''Total Voyages: {total_voyages:,}
Completed Successfully: {completed_successfully:,}
Captured/Unknown: {captured_unknown:,}
Success Rate: {success_rate:.1f}%'''

props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()

# ---------------------- BAR CHART --------------------------
# --- Top Points of Embarkation for Slave Voyages to New Orleans ---
embarkation_col = "voyage_itinerary__imp_principal_place_of_slave_purchase__name"

# Replace NaN, empty strings with "Unknown"
df[embarkation_col] = df[embarkation_col].replace(['', 'Unknown'], 'Unknown')
df[embarkation_col] = df[embarkation_col].fillna('Unknown')

# Count voyages per embarkation point
embarkation_counts = df[embarkation_col].value_counts()
top15_embarkation = embarkation_counts.nlargest(15)

# Create a new column that replaces non-top15 with "Others"
df["embarkation_grouped"] = df[embarkation_col].where(df[embarkation_col].isin(top15_embarkation.index), "Others")
embarkation_grouped_counts = df["embarkation_grouped"].value_counts()

# Create the bar plot
fig, ax = plt.subplots(figsize=(14, 8))

# Get unique names and assign colors
embarkation_names = embarkation_grouped_counts.index.tolist()
colors = cm.get_cmap("plasma")(np.linspace(0, 1, len(embarkation_names)))

# Plot
bars = ax.bar(
    embarkation_grouped_counts.index,
    embarkation_grouped_counts.values,
    color=colors,
    edgecolor='black',
    linewidth=0.5
)

ax.set_title("Top 15 Points of Embarkation for Slave Voyages to New Orleans", 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel("Embarkation Point")
ax.set_ylabel("Number of Voyages")
ax.set_xticklabels(embarkation_grouped_counts.index, rotation=45, ha="right")
ax.grid(axis="y", alpha=0.3)

# Add average line
avg_embarkation = embarkation_grouped_counts.mean()
max_embarkation = embarkation_grouped_counts.max()
ax.axhline(avg_embarkation, 
           color='crimson',
           linestyle='--',
           alpha=0.8,
           label=f'Average: {avg_embarkation:.1f} voyages')
ax.axhline(max_embarkation, 
           color='orange',
           linestyle='--',
           alpha=0.8,
           label=f'Max: {max_embarkation:.1f} voyages for {embarkation_grouped_counts.idxmax()}')
ax.legend()
plt.tight_layout()
plt.show()

# ---------------------- MORTALITY ANALYSIS ----------------------
# --- Mortality Rate Over Time: (Embarked - Disembarked) / Embarked * 100 ---

# Define columns
embarked_col = "voyage_slaves_numbers__imp_total_num_slaves_embarked"
disembarked_col = "voyage_slaves_numbers__imp_total_num_slaves_disembarked"
year_col = "voyage_dates__imp_arrival_at_port_of_dis_sparsedate__year"

# Filter data with both embarked and disembarked numbers
mortality_data = df.dropna(subset=[embarked_col, disembarked_col])
mortality_data = mortality_data[(mortality_data[embarked_col] > 0) & (mortality_data[disembarked_col] >= 0)]

# Calculate mortality rate: (embarked - disembarked) / embarked * 100
mortality_data['mortality_count'] = mortality_data[embarked_col] - mortality_data[disembarked_col]
mortality_data['mortality_rate'] = (mortality_data['mortality_count'] / mortality_data[embarked_col]) * 100

# Group by year to see trends
mortality_by_year = mortality_data.groupby(year_col).agg({
    'mortality_rate': 'mean',
    embarked_col: 'sum',
    disembarked_col: 'sum',
    'mortality_count': 'sum'
}).reset_index()

# Recalculate overall mortality rate by year
mortality_by_year['overall_mortality_rate'] = (mortality_by_year['mortality_count'] / mortality_by_year[embarked_col]) * 100

# Create single plot figure
fig, ax = plt.subplots(figsize=(14, 8))

# Mortality rate over time (line plot)
ax.plot(mortality_by_year[year_col], mortality_by_year['overall_mortality_rate'], 
        marker='o', linewidth=2, markersize=6, color='darkred', alpha=0.8)
ax.fill_between(mortality_by_year[year_col], mortality_by_year['overall_mortality_rate'], 
                alpha=0.3, color='red')

ax.set_title('Mortality Rate in Slave Voyages to New Orleans Over Time', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Year')
ax.set_ylabel('Mortality Rate (%)')
ax.grid(True, alpha=0.3)

# Add average mortality line
avg_mortality = mortality_by_year['overall_mortality_rate'].mean()
max_mortality = mortality_by_year['overall_mortality_rate'].max()
ax.axhline(avg_mortality, color='blue', linestyle='--', alpha=0.7,
           label=f'Average Mortality: {avg_mortality:.1f}%')
ax.axhline(max_mortality, color='red', linestyle='--', alpha=0.7,   
           label=f'Max Mortality: {max_mortality:.1f}% in {int(mortality_by_year.loc[mortality_by_year["overall_mortality_rate"].idxmax(), year_col])}')
ax.legend()

plt.tight_layout()
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.ticker as mticker
import matplotlib.cm as cm
import matplotlib.colors as mcolors


# Load the cleaned combined dataset 
df = pd.read_csv("data/new_orleans_combined.csv")
df.dropna()

# ---- Timeline: Voyages per year ----
year_col = "voyage_dates__imp_arrival_at_port_of_dis_sparsedate__year"
voyage_per_year = df.groupby(year_col).size().reset_index(name='voyage_count')

plt.figure(figsize=(10,6))
plt.plot(voyage_per_year[year_col], 
         voyage_per_year['voyage_count'], 
         color='red', 
         markerfacecolor='white', 
         markeredgecolor='red', 
         marker='o', 
         linewidth=2
)
plt.title("Number of Voyages to New Orleans per Year")
plt.xlabel("Year")
plt.ylabel("Voyage Count")
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- Histogram: Number of enslaved people disembarked per Year ----
year_col = "voyage_dates__imp_arrival_at_port_of_dis_sparsedate__year"
disembarked_col = "voyage_slaves_numbers__imp_total_num_slaves_disembarked"

# Retrieve a group by year and sum the disembarked enslaved people
disembarked_per_year = (
    df.groupby(year_col, as_index=False)[disembarked_col]
        .sum()
        .rename(columns={disembarked_col: 'disembarked_total'})
        .sort_values(by=year_col)
)

# Create the bar plot
fig, ax = plt.subplots(figsize=(12,6))

# Color mapping based on the number of disembarked enslaved people
heights = disembarked_per_year["disembarked_total"].to_numpy()
norm = mcolors.Normalize(vmin=heights.min(), vmax=heights.max())
colors = cm.get_cmap("rocket")(norm(heights))

ax.bar(
    disembarked_per_year[year_col],
    disembarked_per_year["disembarked_total"],
    color=colors,
    edgecolor='black'
)
ax.set_title("Total Number of Enslaved People Disembarked in New Orleans per Year")
ax.set_xlabel("Year")
ax.set_ylabel("Number of Enslaved People Disembarked")
ax.grid(axis="y", alpha=0.3)
ax.axhline(disembarked_per_year["disembarked_total"].mean(), 
           color='red', 
           linestyle='--', 
           label=f'Average: {disembarked_per_year["disembarked_total"].mean():,.0f} enslaved per year')
ax.axhline(disembarked_per_year["disembarked_total"].max(), 
           color='green', 
           linestyle=':', 
           label=f'Maximum: {disembarked_per_year["disembarked_total"].max():,.0f} enslaved in {disembarked_per_year[year_col][disembarked_per_year["disembarked_total"].idxmax()]}')
ax.legend(loc="upper right")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot for where they embarked from 

# Plot of embarked, disembakred and number of dead

# Plot of voyages that got captured or shipwrecked 

# Voyage lengths by dataset source

# Plot of where the ships originated from 

# Plot of percentage of children, females, and males 


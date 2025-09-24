import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.ticker as mticker


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
plt.figure(figsize=(10,6))
df["voyage_slaves_numbers__imp_total_num_slaves_disembarked"].dropna().hist(bins=30)
plt.title("Distribution of Enslaved People Disembarked in New Orleans")
plt.xlabel("Number of People Disembarked")
plt.ylabel("Frequency")
plt.show()

# Plot for where they embarked from 

# Plot of embarked, disembakred and number of dead

# Plot of voyages that got captured or shipwrecked 

# Voyage lengths by dataset source

# Plot of where the ships originated from 

# Plot of percentage of children, females, and males 


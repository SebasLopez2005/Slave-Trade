import pandas as pd
import matplotlib.pyplot as plt

# Load the cleaned Excel
df = pd.read_csv("new_orleans_combined.csv")
df.dropna()
# ---- Timeline: Voyages per year ----
plt.figure(figsize=(10,6))
df.groupby("voyage_dates__imp_arrival_at_port_of_dis_sparsedate__year")["id"].count().plot(kind="line", marker="o")
plt.title("Number of Voyages to New Orleans per Year")
plt.xlabel("Year")
plt.ylabel("Voyage Count")
plt.grid(True)
plt.show()

# ---- Histogram: Number of enslaved people disembarked ----
plt.figure(figsize=(10,6))
df["voyage_slaves_numbers__imp_total_num_slaves_disembarked"].dropna().hist(bins=30)
plt.title("Distribution of Enslaved People Disembarked in New Orleans")
plt.xlabel("Number of People Disembarked")
plt.ylabel("Frequency")
plt.show()

# ---- Boxplot: Compare voyage lengths (days) by dataset source ----
plt.figure(figsize=(8,6))
df.boxplot(column="voyage_dates__length_middle_passage_days", by="dataset_source")
plt.title("Middle Passage Length by Dataset Source")
plt.suptitle("")
plt.xlabel("Dataset Source")
plt.ylabel("Days")
plt.show()

# ---- Bar chart: Average number disembarked by dataset source ----
plt.figure(figsize=(8,6))
df.groupby("dataset_source")["voyage_slaves_numbers__imp_total_num_slaves_disembarked"].mean().plot(kind="bar")
plt.title("Average Number of Enslaved People Disembarked (New Orleans)")
plt.xlabel("Dataset Source")
plt.ylabel("Average Disembarked")
plt.show()

# Column with enslaver names
enslaver_col = "enslavers"

# Count voyages per enslaver
enslaver_counts = df[enslaver_col].value_counts()

# Get top 10 enslavers
top10 = enslaver_counts.nlargest(10)

# Create a new column that replaces non-top10 enslavers with "Others"
df["enslaver_grouped"] = df[enslaver_col].where(df[enslaver_col].isin(top10.index), "Others")

# Count again (top 10 + Others)
enslaver_grouped_counts = df["enslaver_grouped"].value_counts()

# Plot
plt.figure(figsize=(12,6))
enslaver_grouped_counts.plot(kind="bar")

plt.title("Top 10 Enslavers by Number of Voyages to New Orleans")
plt.xlabel("Enslaver")
plt.ylabel("Voyage Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Do it by ship and nationality 
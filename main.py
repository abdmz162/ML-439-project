import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv('titanic.csv')
df.columns = df.columns.str.lower()

print("Loaded dataset:", df.shape)


# =========================
# 2. STRATIFIED SPLIT
# =========================
target = 'survived'

df_0 = df[df[target] == 0].sample(frac=1, random_state=42)
df_1 = df[df[target] == 1].sample(frac=1, random_state=42)

split_0 = int(0.8 * len(df_0))
split_1 = int(0.8 * len(df_1))

dev_df = pd.concat([df_0[:split_0], df_1[:split_1]]).sample(frac=1, random_state=30)
test_df = pd.concat([df_0[split_0:], df_1[split_1:]]).sample(frac=1, random_state=30)


# =========================
# 3. CREATE FOLDERS
# =========================
os.makedirs("output/data", exist_ok=True)
os.makedirs("output/plots/illustrative", exist_ok=True)
os.makedirs("output/plots/non-illustrative", exist_ok=True)


# =========================
# 4. SAVE DATASETS
# =========================
dev_df.to_csv("output/data/dev_df.csv", index=False)
test_df.to_csv("output/data/test_df.csv", index=False)

print("Saved dev/test datasets ✔")


# =========================
# 5. SAVE BASIC STATS
# =========================
with open("output/data/basic_stats.txt", "w") as f:
    f.write("=== DATASET SHAPES ===\n")
    f.write(f"Original: {df.shape}\n")
    f.write(f"Dev: {dev_df.shape}\n")
    f.write(f"Test: {test_df.shape}\n\n")

    # =========================
    # BASIC STATISTICS (NUMERIC)
    # =========================

    selected_cols = ['survived', 'age', 'sibsp', 'parch', 'fare']
    numeric_df = dev_df[selected_cols]
    
    f.write("=== SELECTED NUMERIC SUMMARY ===\n\n")
    f.write(numeric_df.describe().to_string())
    f.write("\n\n")


    print("Saved stats ✔")


# =========================
# 6. VISUAL EXPLORATION
# =========================
#MUIZ=>survival_distribution, survival_by_sex, age_distribution
#WALEED=>survival_by_class,survival_vs_age

# 1. Survival distribution
plt.figure()

ax=sns.countplot(data=dev_df, x='survived')

for container in ax.containers:
    ax.bar_label(container)

plt.title("Survival Distribution")
plt.savefig("output/plots/illustrative/survival_distribution.png")
plt.close()

# 2. Sex vs survival
plt.figure()

ax=sns.countplot(data=dev_df, x='sex', hue='survived')

for container in ax.containers:
    ax.bar_label(container)
    
plt.title("Survival by Sex")
plt.savefig("output/plots/illustrative/survival_by_sex.png")
plt.close()

# 3. Embarked vs Survival (weak signal) VERIFIED
plt.figure()
ax=sns.countplot(data=dev_df, x='embarked', hue='survived')

for container in ax.containers:
    ax.bar_label(container)
    
plt.title("Embarked vs Survival")
plt.savefig("output/plots/illustrative/embarked_vs_survival.png")
plt.close()

# 4. Class vs survival
plt.figure()
ax=sns.countplot(data=dev_df, x='pclass', hue='survived')

for container in ax.containers:
    ax.bar_label(container)
    
plt.title("Survival by Class")
plt.savefig("output/plots/illustrative/survival_by_class.png")
plt.close()

# 5. Age vs survival

data = dev_df.dropna(subset=['age'])

# Create age bins (you can adjust bin size)
bins = np.arange(0, 85, 5)  # 0–80 in steps of 5
data['age_bin'] = pd.cut(data['age'], bins)

# Count for each survival class
age_survival = data.groupby(['age_bin', 'survived']).size().unstack(fill_value=0)

# Plot
plt.figure(figsize=(12, 5))

# Bars
plt.bar(age_survival.index.astype(str), age_survival[1], color='blue', label='Survived')
plt.bar(age_survival.index.astype(str), age_survival[0],
        bottom=age_survival[1], color='red', label='Did Not Survive')

plt.xticks(rotation=45)
plt.xlabel("Age (years)")
plt.ylabel("Count")
plt.title("Age vs Survival (Count)")

plt.legend()
plt.tight_layout()

plt.savefig("output/plots/illustrative/age_vs_survival.png", dpi=300)
plt.close()

print("Saved Age vs Survival ✔")

# BOXPLOT
plt.figure(figsize=(6, 5))

sns.boxplot(
    data=data,
    x='survived',
    y='age',
    hue='survived',        
    palette={0: 'red', 1: 'blue'},
    legend=False            
)

plt.xticks([0, 1], ['Did Not Survive', 'Survived'])

plt.xlabel("Survival Status")
plt.ylabel("Age (years)")
plt.title("Age BoxPlot by Survival")

plt.tight_layout()

plt.savefig("output/plots/illustrative/age_boxplot_by_survival.png", dpi=300)
plt.close()


# =========================
# FINAL CHECK
# =========================
print("\nFILES CREATED:")
print("✔ output/data/dev_df.csv")
print("✔ output/data/test_df.csv")
print("✔ output/data/basic_stats.txt")
print("✔ output/plots/*.png")

# =========================
# EXTRA FIGURES (NOT USED IN FINAL ANALYSIS)
# =========================
#Waleed=>parch_distribution, sibsp_distribution, cabin_distribution 
#Muiz=> passengerid_vs_survival, ticket_frequency 
#Both=>fare_vs_count

# 1. Parch vs Count (non-illustrative)
plt.figure()

dev_df['parch'].value_counts().sort_index().plot(kind='bar')

plt.title("Parch vs Count (Non-informative Distribution)")
plt.xlabel("Number of Parents/Children aboard (Parch)")
plt.ylabel("Count of Passengers")

plt.tight_layout()

plt.savefig("output/plots/non-illustrative/parch_distribution.png", dpi=300)
plt.close()

# 2. SibSp vs Count (non-illustrative)
plt.figure()

dev_df['sibsp'].value_counts().sort_index().plot(kind='bar')

plt.title("SibSp vs Count (Non-informative Distribution)")
plt.xlabel("Number of Siblings/Spouses aboard (SibSp)")
plt.ylabel("Count of Passengers")

plt.tight_layout()

plt.savefig("output/plots/non-illustrative/sibsp_distribution.png", dpi=300)
plt.close()

# 3. Cabin feature (too sparse / many missing values) MWA
if 'cabin' in dev_df.columns:
    plt.figure()
    dev_df['cabin'].fillna('Unknown').value_counts().head(15).plot(kind='bar')
    plt.title("Cabin Distribution (Top 15)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("output/plots/non-illustrative/cabin_distribution.png")
    plt.close()

# 4. Passenger ID vs Survival (binary scatter plot)
plt.figure(figsize=(10, 4))

plt.scatter(
    dev_df['passengerid'].values,
    dev_df['survived'].values,
    alpha=0.4,
    s=10
)

plt.title("Passenger ID vs Survival (Non-informative)")
plt.xlabel("Passenger ID")
plt.ylabel("Survived (0/1)")
plt.yticks([0, 1])

plt.tight_layout()

plt.savefig("output/plots/non-illustrative/passengerid_vs_survival.png", dpi=300)
plt.close()

# 5. Ticket frequency (high cardinality, noisy)
plt.figure()
dev_df['ticket'].value_counts().head(20).plot(kind='bar')
plt.title("Ticket Frequency (Top 20)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output/plots/non-illustrative/ticket_frequency.png")
plt.close()

#6. Count vs Fare (dot-based frequency plot)
fare_counts = dev_df['fare'].value_counts().reset_index()
fare_counts.columns = ['fare', 'count']

plt.figure(figsize=(10, 4))

plt.scatter(
    fare_counts['fare'],
    fare_counts['count'],
    alpha=0.6,
    s=20
)

plt.title("Fare vs Count (Dot Plot)")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")

plt.tight_layout()

plt.savefig("output/plots/non-illustrative/fare_vs_count.png", dpi=300)
plt.close()


print("Saved extra (non-used) figures")
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from functools import reduce
from sklearn.metrics import r2_score

def get_antisense(seq):
    antisense_seq = ""
    map_antisense = {"A":"T", "T": "A", "C":"G", "G":"C"}
    for nucleotide in seq:
        antisense_seq = map_antisense[nucleotide] + antisense_seq
    return antisense_seq

df_guide = pd.read_csv("data/variant_seq.csv",dtype={'Nucleotide_edits': object})
df_zscores_primary_screening = pd.read_csv("data/zscores_from_primary_screening.csv")
map_guide = df_guide[["guide_seq","name"]].set_index("name").to_dict()["guide_seq"]
map_guide_fixed = df_guide[["guide_seq_fixed","sgRNA_Strand","Nucleotide_edits","name"]].set_index("name").to_dict()
map_zscores_primary_screening = df_zscores_primary_screening.set_index("variant").to_dict()["zscore_primary_screening"]

#%% fold change aligned_sequence
dict1 = {
    "variant": [],
    "treatment": [],
    "reads_aligned": [],
    "%reads_aligned": [],
    "reads_total": [],
}
for variant, sgRNA in map_guide.items():
    sgRNA = map_guide[variant] 
    dfs = []
    map1 = { 1:"vo1", 2:"vo2", 3:"vo3", 4:"no_vo1", 5:"no_vo2", 6:"no_vo3", }
    map2 = { 1:"vo", 2:"vo", 3:"vo", 4:"no_vo", 5:"no_vo", 6:"no_vo", }
    ref_seq = ""
    for i in [4,5,6,1,2,3]:
        filei = f"data/CRISPRessoBatch_on_batch_file/CRISPResso_on_{variant}_VO_{i}/Alleles_frequency_table_around_sgRNA_{sgRNA}.txt"
        dfi = pd.read_csv(filei, sep="\t")
        ref_seq = dfi["Reference_Sequence"][0]
        filei2 = f"data/CRISPRessoBatch_on_batch_file/CRISPResso_on_{variant}_VO_{i}/CRISPResso_quantification_of_editing_frequency.txt"
        dfi2 = pd.read_csv(filei2, sep="\t")
        reads_aligned = dfi2["Reads_aligned"][0]
        reads_total = dfi2["Reads_in_input"][0]
        percentage_reads_aligned = reads_aligned/reads_total * 100
        dict1["variant"].append(variant)
        dict1["treatment"].append(map2[i])
        dict1["reads_aligned"].append(reads_aligned)
        dict1["reads_total"].append(reads_total)
        dict1["%reads_aligned"].append(percentage_reads_aligned)

df_a1 = pd.DataFrame(dict1)
df_a2 = df_a1.groupby(["variant","treatment"]).mean().reset_index()

df_a3 = df_a2.pivot(index='variant', columns='treatment', values='%reads_aligned').reset_index()
df_a3.to_csv("results/percentage_aligned.csv", index=False)
map_percentage_aligned = df_a3.set_index("variant").to_dict()["no_vo"]


df_a4 = df_a2.pivot(index='variant', columns='treatment', values='reads_aligned').reset_index()
df_a4["%reads_aligned"] = df_a4["variant"].map(map_percentage_aligned)
df_a4["fc_aligned_seq"] = df_a4["vo"] / df_a4["no_vo"]
df_a4["zscores_primary_screening"] = df_a4["variant"].map(map_zscores_primary_screening)


#%%
slope, intercept = np.polyfit(df_a4["fc_aligned_seq"], df_a4["zscores_primary_screening"], 1)
r_squared = r2_score(df_a4["zscores_primary_screening"],df_a4["fc_aligned_seq"] * slope + intercept)
df_a4.plot.bar(x="variant", y="%reads_aligned")
plt.title(f"%reads_aligned")
plt.grid("on")
plt.tight_layout()
plt.savefig("results/%reads_aligned.png")


#%% fold change % modifed
dict1 = {
    "variant": [],
    "treatment": [],
    "%modified": [],
}
for variant, sgRNA in map_guide.items():
    sgRNA = map_guide[variant] 
    dfs = []
    map2 = { 1:"vo", 2:"vo", 3:"vo", 4:"no_vo", 5:"no_vo", 6:"no_vo", }
    ref_seq = ""
    for i in [4,5,6,1,2,3]:
        filei = f"data/CRISPRessoBatch_on_batch_file/CRISPResso_on_{variant}_VO_{i}/Alleles_frequency_table_around_sgRNA_{sgRNA}.txt"
        dfi = pd.read_csv(filei, sep="\t")
        ref_seq = dfi["Reference_Sequence"][0]
        filei2 = f"data/CRISPRessoBatch_on_batch_file/CRISPResso_on_{variant}_VO_{i}/CRISPResso_quantification_of_editing_frequency.txt"
        dfi2 = pd.read_csv(filei2, sep="\t")
        modified = dfi2["Modified%"][0]
        dict1["variant"].append(variant)
        dict1["treatment"].append(map2[i])
        dict1["%modified"].append(modified)

df1 = pd.DataFrame(dict1)
df2 = df1.groupby(["variant","treatment"]).mean().reset_index()
df3 = df2.pivot(index='variant', columns='treatment', values='%modified').reset_index()
df3 = df3.rename(columns={
    "no_vo":"%modified_no_vo",
    "vo":"%modified_vo",
    })
df3["%reads_aligned"] = df3["variant"].map(map_percentage_aligned)
df3["zscores_primary_screening"] = df3["variant"].map(map_zscores_primary_screening)
df3["fc_%modified"] = df3["%modified_vo"] / df3["%modified_no_vo"]
df3["lfc_%modified"] = np.log2(df3["fc_%modified"])
df3["weighted_fc_%modified"] = df3["fc_%modified"] * df3["%modified_vo"] / 100

df3["log_weighted_fc_%modified"] = np.log2(df3["weighted_fc_%modified"])
df3 = df3[ df3["%reads_aligned"] > 5 ]
df3.to_csv("results/weighted_fc_%modified.csv", index=False)


#%%
slope, intercept = np.polyfit(df3["weighted_fc_%modified"], df3["zscores_primary_screening"], 1)
r_squared = r2_score(df3["zscores_primary_screening"],df3["weighted_fc_%modified"] * slope + intercept)
df3.plot.scatter(x="weighted_fc_%modified", y="zscores_primary_screening")
for i,row in df3.iterrows():
    plt.text(row["weighted_fc_%modified"], row["zscores_primary_screening"], f"{row['variant']} ({ round( row['%modified_vo'],2 ) }%)", alpha=0.5, fontsize=8)
plt.plot(df3["weighted_fc_%modified"], df3["weighted_fc_%modified"] * slope + intercept, color="red")
plt.title(f"r_squared: {round(r_squared, 4)}")
plt.grid("on")
plt.tight_layout()
plt.savefig("results/weighted_fc_%modified_vs_zscores_primary_screening.png")


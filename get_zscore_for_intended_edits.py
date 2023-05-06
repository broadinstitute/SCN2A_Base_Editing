import os
import numpy as np
import pandas as pd
from functools import reduce


def get_antisense(seq):
    antisense_seq = ""
    map_antisense = {"A":"T", "T": "A", "C":"G", "G":"C"}
    for nucleotide in seq:
        antisense_seq = map_antisense[nucleotide] + antisense_seq
    return antisense_seq

df_guide = pd.read_csv("data/variant_seq.csv",dtype={'Nucleotide_edits': object})
map_guide = df_guide[["guide_seq","name"]].set_index("name").to_dict()["guide_seq"]
map_guide_fixed = df_guide[["guide_seq_fixed","sgRNA_Strand","Nucleotide_edits","name"]].set_index("name").to_dict()


#%%
zscore_intended_edits = [] 
for variant, sgRNA in map_guide.items():
    sgRNA = map_guide[variant] 
    map1 = {
        1:"vo1", 2:"vo2", 3:"vo3",
        4:"no_vo1", 5:"no_vo2", 6:"no_vo3",
    }
    filei = f"data/CRISPRessoBatch_on_batch_file/CRISPResso_on_{variant}_VO_1/Alleles_frequency_table_around_sgRNA_{sgRNA}.txt"
    dfi = pd.read_csv(filei, sep="\t")
    ref_seq = dfi["Reference_Sequence"][0]
    sgRNA_fixed = map_guide_fixed["guide_seq_fixed"][variant]

    guide_start_index = ref_seq.find(sgRNA_fixed)
    is_antisense = False
    if guide_start_index == -1:
        guide_start_index = ref_seq.find(get_antisense(sgRNA_fixed))
        is_antisense = True

    # edit_positions is relative to the sgRNA, need to consider sense or antisense
    edit_positions = map_guide_fixed["Nucleotide_edits"][variant]
    # edit_positions2 is relative to the reference sequence, it is based on left to right sequence
    edit_positions2 = [] 
    for position in edit_positions:
        if is_antisense:
            position2 = guide_start_index + len(sgRNA_fixed) - int(position)
            edit_positions2.append(position2)
        else:
            position2 = guide_start_index + int(position) - 1
            edit_positions2.append(position2)
    edit_positions2.sort()

    dfs = []
    for i in [4,5,6,1,2,3]:
        filei = f"data/CRISPRessoBatch_on_batch_file/CRISPResso_on_{variant}_VO_{i}/Alleles_frequency_table_around_sgRNA_{sgRNA}.txt"
        dfi = pd.read_csv(filei, sep="\t")
        ref_seq = dfi["Reference_Sequence"][0]
        unedited_index = dfi.index[dfi["Aligned_Sequence"] == dfi["Reference_Sequence"]].tolist()[0]
        dfi["rank"] = range(1, len(dfi)+1)
        dfi.at[unedited_index,"rank"] = 0
        dfi = dfi.sort_values("rank").drop("rank", axis=1)
        dfi = dfi.rename(columns={
            "%Reads":f"%Reads_{map1[i]}",
            "#Reads":f"#Reads_{map1[i]}"
        })
        if len(edit_positions) == 1:
            index_single = edit_positions2[0]
            dfi["edit"] = dfi["Aligned_Sequence"].str.slice(start=index_single, stop=index_single+1)
        if len(edit_positions) == 2:
            index1 = edit_positions2[0]
            index2 = edit_positions2[1]
            dfi["edit"] = dfi["Aligned_Sequence"].str.slice(start=index1, stop=index1+1) + dfi["Aligned_Sequence"].str.slice(start=index2, stop=index2+1)
            dfi = dfi[[ "Aligned_Sequence", "edit", f"%Reads_{map1[i]}" ]]
        dfi = dfi.groupby(["Aligned_Sequence", "edit"]).sum()[[ f"%Reads_{map1[i]}" ]].reset_index()
        dfi = dfi.sort_values(f"%Reads_{map1[i]}", ascending=False)
        dfs.append(dfi)

    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['Aligned_Sequence','edit'], how='inner'), dfs)
    unedited_index = df_merged.index[df_merged["Aligned_Sequence"] == ref_seq].values[0]
    df_merged["rank"] = range(1, len(df_merged)+1)
    df_merged.at[unedited_index,"rank"] = 0
    df_merged = df_merged.sort_values("rank").drop("rank", axis=1)
    df_merged["%Reads_no_vo"] = df_merged[["%Reads_no_vo1","%Reads_no_vo2","%Reads_no_vo3"]].mean(axis=1)
    df_merged["%Reads_vo"] = df_merged[["%Reads_vo1","%Reads_vo2","%Reads_vo3"]].mean(axis=1)
    df_merged["fc_%reads"] = df_merged["%Reads_vo"] / df_merged["%Reads_no_vo"]
    df_merged["weighted_fc_%reads"] = df_merged["fc_%reads"] * df_merged["%Reads_vo"] / 100
    print(df_merged)

    #%%
    if len(edit_positions) == 2:
        df_merged["edit1"] = df_merged["edit"].str.slice(start=0,stop=1)
        df_merged["edit2"] = df_merged["edit"].str.slice(start=1,stop=2)
        if is_antisense:
            features = [ "edit1_A",  "edit2_A" ]
            df_merged = df_merged[
                df_merged["edit1"].isin(["G","A"]) &
                df_merged["edit2"].isin(["G","A"])
            ]
        else:
            features = [ "edit1_T",  "edit2_T" ]
            df_merged = df_merged[
                df_merged["edit1"].isin(["T","C"]) &
                df_merged["edit2"].isin(["T","C"])
            ]
        df_merged = pd.merge(df_merged,pd.get_dummies(df_merged[[ "edit1", "edit2" ]]), left_index=True, right_index=True)
        df_merged["edit_both"] = np.nan
        df_merged.loc[ (df_merged[features]==1).all(axis=1), "edit_both" ] = 1
        df_merged.loc[ (df_merged[features]==0).all(axis=1), "edit_both" ] = 0

        df_edited = df_merged[ df_merged["edit_both"] == 1 ]
        df_control = df_merged[ df_merged["edit_both"] == 0 ]

    if len(edit_positions) == 1:
        print(df_merged)
        if is_antisense:
            feature = "A"
            df_merged = df_merged[ df_merged["edit"].isin(["G","A"]) ]
        else:
            feature = "T"
            df_merged = df_merged[ df_merged["edit"].isin(["T","C"]) ]
        df_merged = pd.merge(df_merged,pd.get_dummies(df_merged["edit"]), left_index=True, right_index=True)

        df_edited = df_merged[ df_merged[feature] == 1 ]
        df_control = df_merged[ df_merged[feature] == 0 ]

    mean_control = df_control["weighted_fc_%reads"].mean()
    mean_edited = df_edited["weighted_fc_%reads"].mean()
    n_control = df_control["weighted_fc_%reads"].count()
    n_edited = df_edited["weighted_fc_%reads"].count()
    variance_control = df_control["weighted_fc_%reads"].var()
    variance_edited = df_edited["weighted_fc_%reads"].var()
    numerator = mean_edited - mean_control
    denominator = (variance_control/n_control + variance_edited/n_edited) ** 0.5
    zscore = numerator / denominator 


    zscore_intended_edits.append({
        "variant": variant,
        "zscore_weighted_fc_%reads": zscore
    })

df = pd.DataFrame(zscore_intended_edits)
df.to_csv("results/zscore_intended_edits.csv", index=False)

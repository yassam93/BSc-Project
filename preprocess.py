import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

plt.rc("font", size=14)
plt.rc("axes", labelsize=14, titlesize=14)
plt.rc("legend", fontsize=14)
plt.rc("xtick", labelsize=10)
plt.rc("ytick", labelsize=10)

IMAGES_PATH = Path() / "images" / "preprocessing"
IMAGES_PATH.mkdir(parents=True, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# ### Load dataset

lifestyle_data = pd.read_csv("datasets/Sleep_health_and_lifestyle_dataset.csv")

lifestyle_data.shape

lifestyle_data.info()

lifestyle_data.head(5)

lifestyle_data.columns


# ### Fill NaN values with None

counts = lifestyle_data.isna().sum()
print(counts.sort_values())

lifestyle_data["Sleep Disorder"] = lifestyle_data["Sleep Disorder"].fillna("None")
lifestyle_data.head(5)


# ### Check unique rows in dataframe

print(
    f"number of non duplicate rows is {len(lifestyle_data[lifestyle_data.columns].drop_duplicates())}"
)
number_entities = lifestyle_data.drop_duplicates().shape[0]


# ### Check duplicate rows in dataframe

print(f"number of duplicate rows is {lifestyle_data.duplicated().sum()}")


# ### Check number of the NaN in the dataframe

counts = lifestyle_data.isna().sum()
print(counts.sort_values())


# ### drop 'Person ID' column, its not a feature

lifestyle_data = lifestyle_data.drop("Person ID", axis=1)
lifestyle_data.shape


# ### Check statistic of the dataset

lifestyle_data.describe()


# ### Count distinct values, use nunique:
# #### for example gender is Male or Female, Sleep Disorder is Sleep Apnea, None or Insomnia

lifestyle_data.nunique()


# ## Check Features
#
# ### Gender is categorical, nominal
# ### Age is int
# ### Occupation is categorical, nominal
# ### Sleep Duration is float
# ### Quality of Sleep is float
# ### Physical Activity Level is float
# ### Stress Level is int
# ### BMI Category is categorical, ordinal
# ### Blood Pressure is categorical, ordinal
# ### Heart Rate is int
# ### Daily Steps is int
# ### Sleep Disorder is categorcal ordinal

# ### Check unique value of the Sleep Disorder

lifestyle_data["Sleep Disorder"].unique()


# ### Check number of repeatness for each state

lifestyle_data["Sleep Disorder"].value_counts()


lifestyle_data["Sleep Disorder"].value_counts() / number_entities


# ## As it seems this database is imbalanced. the percentage of the repeatness for None is much more than others. But it is not extermely imbalanced

# #### check gender


lifestyle_data["Gender"].unique()

lifestyle_data["Gender"].value_counts() / number_entities


# ### distribution of the gender and sleep disorder

x = lifestyle_data.groupby("Sleep Disorder")["Gender"].value_counts()
x


lifestyle_data["Sleep Disorder"].unique(), lifestyle_data["Gender"].unique()


labels = [
    "Insomnia_Male",
    "Insomnia_Female",
    "None_Male",
    "None_Female",
    "Sleep_Apnea_Female",
    "Sleep_Apnea_Male",
]
palette_color = sns.color_palette("bright")

plt.pie(
    lifestyle_data.groupby("Sleep Disorder")["Gender"].value_counts(),
    labels=labels,
    colors=palette_color,
    autopct="%.0f%%",
)

save_fig("sleep_order_gender")
plt.show()


lifestyle_data.groupby("Sleep Disorder")["Gender"].value_counts() / 374


lifestyle_data.groupby("Sleep Disorder")["Gender"].value_counts() / lifestyle_data[
    "Sleep Disorder"
].value_counts()


# ### number of male which have Sleep Apnea is too small: approximately 3 percent of the total data

# #### check age: range of people age


lifestyle_data["Age"].describe()

lifestyle_data["Age"].value_counts().sort_values()


lifestyle_data.groupby("Sleep Disorder")["Age"].value_counts()


sns.jointplot(data=lifestyle_data, x="Sleep Disorder", y="Age")
save_fig("sleep_order_age")


g = sns.JointGrid(data=lifestyle_data, x="Sleep Disorder", y="Age")
g.plot_joint(sns.histplot)
g.plot_marginals(sns.boxplot)
save_fig("sleep_order_age_box_plot")


# ### people age is between 27 and 43

columns = [
    "Age",
    "Sleep Duration",
    "Quality of Sleep",
    "Physical Activity Level",
    "Stress Level",
    "Heart Rate",
    "Daily Steps",
]
for col in lifestyle_data.columns:
    if col in columns:
        # sns.displot(lifestyle_data, x=col, kind="kde")
        sns.displot(lifestyle_data, x=col, kde=True)
        save_fig(f"sleep_order_{col}_kde_hist")

df1 = lifestyle_data[columns]
sns.pairplot(df1)

save_fig(f"sleep_order_pair_plot")


g = sns.PairGrid(df1)
g.map_upper(sns.histplot)
g.map_lower(sns.kdeplot, fill=False)
g.map_diag(sns.histplot, kde=True)

save_fig(f"sleep_order_pair_grid")


g = sns.PairGrid(lifestyle_data)
g.map_upper(sns.histplot, fill=False)
g.map_lower(sns.kdeplot, fill=False)
g.map_diag(sns.histplot, kde=True)
save_fig(f"sleep_order_pair_grid_1")


# ### check job

lifestyle_data["Occupation"].unique()


lifestyle_data["Occupation"].value_counts() / number_entities


lifestyle_data["Occupation"].value_counts()


# ### some jobs have too small candidate in the dataset. So, its hard to reach a proper result for those job, if someone wants to check sleep disorder based on  job, especially those with less than 5 percent(here: Software Engineer, Scientist, Sales and Representative Manager).

lifestyle_data.groupby("Sleep Disorder")["Occupation"].value_counts()


lifestyle_data.groupby("Sleep Disorder")["Occupation"].value_counts() / lifestyle_data[
    "Sleep Disorder"
].value_counts()


# #### when number of sample for specific target be too small, it's hard to disciminate them from each other.

# ### check BMI Category


lifestyle_data["BMI Category"].unique()


lifestyle_data["BMI Category"].value_counts() / number_entities

lifestyle_data.groupby("Sleep Disorder")["BMI Category"].value_counts()


lifestyle_data.groupby("Sleep Disorder")[
    "BMI Category"
].value_counts() / lifestyle_data["Sleep Disorder"].value_counts()


# ### check Sleep Duration


lifestyle_data["Sleep Duration"].unique()


lifestyle_data["Sleep Duration"].value_counts() / number_entities


lifestyle_data.groupby("Sleep Disorder")["Sleep Duration"].value_counts()


lifestyle_data.groupby("Sleep Disorder")[
    "Sleep Duration"
].value_counts() / lifestyle_data["Sleep Disorder"].value_counts()


# ### check blood pressure

lifestyle_data["Blood Pressure"].unique()


lifestyle_data["Blood Pressure"].value_counts() / number_entities

lifestyle_data.groupby("Sleep Disorder")["Blood Pressure"].value_counts()

lifestyle_data.groupby("Sleep Disorder")[
    "Blood Pressure"
].value_counts() / lifestyle_data["Sleep Disorder"].value_counts()


# ### check quality of sleep

lifestyle_data["Quality of Sleep"].value_counts()


lifestyle_data["Quality of Sleep"].value_counts() / number_entities

lifestyle_data.groupby("Sleep Disorder")["Quality of Sleep"].value_counts()

lifestyle_data.groupby("Sleep Disorder")[
    "Quality of Sleep"
].value_counts() / lifestyle_data["Sleep Disorder"].value_counts()


# ### check Physical Activity Level


lifestyle_data["Physical Activity Level"].value_counts()


lifestyle_data["Physical Activity Level"].value_counts() / number_entities


lifestyle_data.groupby("Sleep Disorder")["Physical Activity Level"].value_counts()


lifestyle_data.groupby("Sleep Disorder")[
    "Physical Activity Level"
].value_counts() / lifestyle_data["Sleep Disorder"].value_counts()


# ### check Stress Level


lifestyle_data["Stress Level"].value_counts()


lifestyle_data["Stress Level"].value_counts() / number_entities


lifestyle_data.groupby("Sleep Disorder")["Stress Level"].value_counts()


lifestyle_data.groupby("Sleep Disorder")[
    "Stress Level"
].value_counts() / lifestyle_data["Sleep Disorder"].value_counts()


# ### check Heart Rate


lifestyle_data["Heart Rate"].value_counts()


lifestyle_data["Heart Rate"].value_counts() / number_entities


lifestyle_data.groupby("Sleep Disorder")["Heart Rate"].value_counts()


lifestyle_data.groupby("Sleep Disorder")["Heart Rate"].value_counts() / lifestyle_data[
    "Sleep Disorder"
].value_counts()


# ### check Daily Steps

lifestyle_data["Daily Steps"].value_counts()

lifestyle_data["Daily Steps"].value_counts() / number_entities


lifestyle_data.groupby("Sleep Disorder")["Daily Steps"].value_counts()


lifestyle_data.groupby("Sleep Disorder")["Daily Steps"].value_counts() / lifestyle_data[
    "Sleep Disorder"
].value_counts()

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import dataframe_image as dfi
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer

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


# ### Replace NaN values with 'None'

counts = lifestyle_data.isna().sum()
print(counts.sort_values())

lifestyle_data["Sleep Disorder"] = lifestyle_data["Sleep Disorder"].fillna("None")
lifestyle_data.head(5)


# ### Check unique rows in dataframe

print(
    f"number of non duplicate rows: {len(lifestyle_data[lifestyle_data.columns].drop_duplicates())}"
)
number_entities = lifestyle_data.drop_duplicates().shape[0]


# ### Check duplicate rows in dataframe

print(f"number of duplicate rows: {lifestyle_data.duplicated().sum()}")


# ### Check the number of NaNs in the dataframe

counts = lifestyle_data.isna().sum()
print(counts.sort_values())


# ### Drop 'Person ID' column, its not a feature

lifestyle_data = lifestyle_data.drop("Person ID", axis=1)
lifestyle_data.shape


# ### Check statistics of the dataset

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
# ### Sleep Disorder is categorical ordinal

# ### Check unique value of the Sleep Disorder

lifestyle_data["Sleep Disorder"].unique()


# ### Check frequency of each category in 'Sleep Disorder'

lifestyle_data["Sleep Disorder"].value_counts()


lifestyle_data["Sleep Disorder"].value_counts() / number_entities


# ## It appears this database is imbalanced. The proportion of 'None' for Sleep Disorder is significantly higher in compare to other categories, indicating an imbalance, though it is not extremely severe.

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
lifestyle_data["Blood Pressure"].value_counts()


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


# ### Gender is categorical, nominal
# ### Occupation is categorical, nominal
#
# ### BMI Category is categorical, ordinal
# ### Blood Pressure is categorical, ordinal
# ### Sleep Disorder is categorcal ordinal

# # Convert ordinal features

# ## Blood Pressure


# https://en.wikipedia.org/wiki/Blood_pressure
# https://www.medicinenet.com/blood_pressure_chart_reading_by_age/article.htm

blood_uniq = lifestyle_data["Blood Pressure"].unique()
for pressure in blood_uniq:
    lst_int = list(map(int, pressure.split("/")))
    if lst_int[0] < 100 or lst_int[1] < 70:
        print(f"Hypotension: {pressure} {lst_int[0], lst_int[1]}")  # ?
    elif lst_int[0] < 120 and lst_int[1] < 80:
        print(f"Normal: {pressure} {lst_int[0], lst_int[1]}")  # 0
    elif 120 < lst_int[0] < 129 and lst_int[1] < 80:
        print(f"Elevated: {pressure} {lst_int[0], lst_int[1]}")  # 1
    elif 130 < lst_int[0] < 139 or 80 < lst_int[1] < 89:
        print(f"Hypertension, stage 1: {pressure} {lst_int[0], lst_int[1]}")  # 2
    elif lst_int[0] >= 140 or lst_int[1] >= 90:
        print(f"Hypertension, stage 2: {pressure} {lst_int[0], lst_int[1]}")  # 3


# since Hypotension does not have any records its possible to ignore that.


normal_pressure = ["117/76", "118/76", "115/75", "115/78", "119/77", "118/75"]
elevated_pressure = ["121/79"]
hypertension_stage_1 = [
    "126/83",
    "132/87",
    "130/86",
    "128/85",
    "131/86",
    "128/84",
    "135/88",
    "129/84",
    "130/85",
    "125/82",
    "135/90",
]
hypertension_stage_2 = ["140/90", "142/92", "140/95", "139/91"]

# level 0 : Normal blood pressure
# level 1 : Elevated blood pressure
# level 2 : Hypertension, stage 1 blood pressure
# level 3 : Hypertension, stage 2 blood pressure


def func(x):
    if x in normal_pressure:
        return 0
    elif x in elevated_pressure:
        return 1
    elif x in hypertension_stage_1:
        return 2
    else:
        return 3


lifestyle_data["Blood Pressure"] = lifestyle_data["Blood Pressure"].apply(func)


lifestyle_data["Blood Pressure"].unique()


lifestyle_data["Blood Pressure"].value_counts()


# ## BMI Category

# according to this link:
#
# https://www.health.nsw.gov.au/heal/Pages/bmi.aspx#:~:text=Measuring%20Body%20Mass%20Index&text=It%20measures%20weight%20in%20relation,BMI%20of%2030%20or%20higher


# ['Normal Weight', 'Normal', 'Overweight', 'Obese']
# [0, 1, 2, 3]

weight_mapping = {"Normal Weight": 0, "Normal": 1, "Overweight": 2, "Obese": 3}

lifestyle_data["BMI Category"] = lifestyle_data["BMI Category"].map(weight_mapping)


lifestyle_data["BMI Category"].unique()


lifestyle_data["BMI Category"].value_counts()


# ## Sleep Disorder


# ['None', 'Sleep Apnea', 'Insomnia']
# [0, 1, 2]

sleep_disorder_mapping = {"None": 0, "Sleep Apnea": 1, "Insomnia": 2}

lifestyle_data["Sleep Disorder"] = lifestyle_data["Sleep Disorder"].map(
    sleep_disorder_mapping
)


lifestyle_data["Sleep Disorder"].unique()


lifestyle_data["Sleep Disorder"].value_counts()


lifestyle_data.shape


display(lifestyle_data.head())

# # Convert nominal features

# ## Gender


gender_ohe = OneHotEncoder()

transformed = gender_ohe.fit_transform(lifestyle_data[["Gender"]])
# Getting one hot encoded categories
print(f"Gender categories are: {gender_ohe.categories_}")
# adding the one hot encoded values back to the original df
lifestyle_data[gender_ohe.categories_[0]] = transformed.toarray()
lifestyle_data = lifestyle_data.drop("Gender", axis=1)
display(lifestyle_data.head())


lifestyle_data.shape


# ## Occupation


occupation_ohe = OneHotEncoder()

transformed = occupation_ohe.fit_transform(lifestyle_data[["Occupation"]])
# Getting one hot encoded categories
print(f"Occupation categories are: {occupation_ohe.categories_}")
# adding the one hot encoded values back to the original df
lifestyle_data[occupation_ohe.categories_[0]] = transformed.toarray()
lifestyle_data = lifestyle_data.drop("Occupation", axis=1)
display(lifestyle_data.head())


lifestyle_data.shape


lifestyle_data["Sleep Disorder"] = lifestyle_data.pop("Sleep Disorder")
lifestyle_data


# # Save CSV to a file


file_name = "preprocessed_dataset.csv"
lifestyle_data.to_csv(file_name, sep="\t", encoding="utf-8", index=False, header=True)

# # Check dependencies between features


columns = lifestyle_data.columns
columns


columns_gender = ["Female", "Male", "Sleep Disorder"]
df_gender = lifestyle_data[columns_gender]
g = sns.PairGrid(df_gender)
g.map_upper(sns.histplot, fill=False)
g.map_lower(sns.kdeplot, fill=False)
g.map_diag(sns.histplot, kde=True)
save_fig(f"sleep_order_pair_grid_gender")


columns_bmi = ["BMI Category", "Sleep Disorder"]
df3_bmi = lifestyle_data[columns_bmi]
g = sns.PairGrid(df3_bmi)
g.map_upper(sns.histplot)
g.map_lower(sns.kdeplot, fill=False)
g.map_diag(sns.histplot, kde=True)

save_fig(f"sleep_order_pair_grid_bmi")


columns_bp = ["Blood Pressure", "Sleep Disorder"]
df4_bp = lifestyle_data[columns_bp]
g = sns.PairGrid(df4_bp)
g.map_upper(sns.histplot)
g.map_lower(sns.kdeplot, fill=False)
g.map_diag(sns.histplot, kde=True)

save_fig(f"sleep_order_pair_grid_bp")


columns_occupation = [
    "Accountant",
    "Doctor",
    "Engineer",
    "Lawyer",
    "Manager",
    "Nurse",
    "Sales Representative",
    "Salesperson",
    "Scientist",
    "Software Engineer",
    "Teacher",
    "Sleep Disorder",
]
df5_occupation = lifestyle_data[columns_occupation]
g = sns.PairGrid(df5_occupation)
g.map_upper(sns.histplot)
g.map_lower(sns.kdeplot, fill=False)
g.map_diag(sns.histplot, kde=True)

save_fig(f"sleep_order_pair_grid_occupation")

correlation = lifestyle_data.corr()
sns.heatmap(correlation)


# https://stackoverflow.com/a/42323184/8185618
cmap = sns.diverging_palette(10, 250, as_cmap=True)


def magnify():
    return [
        dict(selector="th", props=[("font-size", "7pt")]),
        dict(selector="td", props=[("padding", "0em 0em")]),
        dict(selector="th:hover", props=[("font-size", "12pt")]),
        dict(
            selector="tr:hover td:hover",
            props=[("max-width", "200px"), ("font-size", "12pt")],
        ),
    ]


correlation.style.background_gradient(cmap, axis=1).format(precision=2).set_properties(
    **{"max-width": "100px", "font-size": "10pt"}
).set_caption("Hover to magify").set_table_styles(magnify())

# https://stackoverflow.com/a/42323184/8185618
cmap = sns.diverging_palette(10, 250, as_cmap=True)


def magnify():
    return [
        dict(selector="th", props=[("font-size", "7pt")]),
        dict(selector="td", props=[("padding", "0em 0em")]),
        dict(selector="th:hover", props=[("font-size", "12pt")]),
        dict(
            selector="tr:hover td:hover",
            props=[("max-width", "200px"), ("font-size", "12pt")],
        ),
    ]


style_correlation = (
    correlation.style.background_gradient(cmap, axis=1)
    .format(precision=2)
    .set_properties(**{"max-width": "100px", "font-size": "10pt"})
    .set_caption("Hover to magify")
    .set_table_styles(magnify())
)
dfi.export(style_correlation, "./images/preprocessing/correlation_1.png")

#https://stackoverflow.com/a/50703596/8185618
correlation.style.background_gradient(cmap="coolwarm").format(precision=2)


style_correlation = correlation.style.background_gradient(cmap="coolwarm").format(
    precision=2
)
dfi.export(style_correlation, "./images/preprocessing/correlation_2.png")


correlation.style.background_gradient(cmap="coolwarm", axis=None).format(precision=2)


style_correlation = correlation.style.background_gradient(
    cmap="coolwarm", axis=None
).format(precision=2)
dfi.export(style_correlation, "./images/preprocessing/correlation_3.png")


correlation = lifestyle_data.corr()

# Fill diagonal and upper half with NaNs
mask = np.zeros_like(correlation, dtype=bool)
mask[np.triu_indices_from(mask)] = True
correlation[mask] = np.nan
(
    correlation.style.background_gradient(cmap="coolwarm", axis=None, vmin=-1, vmax=1)
    .highlight_null(color="#f1f1f1")  # Colour NaNs grey
    .format(precision=2)
)


correlation = lifestyle_data.corr()

# Fill diagonal and upper half with NaNs
mask = np.zeros_like(correlation, dtype=bool)
mask[np.triu_indices_from(mask)] = True
correlation[mask] = np.nan
style_correlation = (
    correlation.style.background_gradient(cmap="coolwarm", axis=None, vmin=-1, vmax=1)
    .highlight_null(color="#f1f1f1")
    .format(precision=2)
)

dfi.export(style_correlation, "./images/preprocessing/correlation_4.png")

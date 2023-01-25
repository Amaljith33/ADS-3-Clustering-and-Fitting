import pandas as pd
import numpy as np
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import matplotlib.pyplot as plt
import scipy.optimize as opt


def read(file_name):
    """
    This function is for reading the csv dataset returning the
    data frame, cleaning the data frame and transposing the dataset

    Returns
    -------
    Dataframes(original), Transposed Dataframes
    """
    data = pd.read_csv(file_name)
    data = data.set_index('Country Name')
    data = data.drop(['Series Name', 'Series Code',
                     'Country Code'], axis=1)
    data = data.dropna()
    data = data.replace('..', 0)
    data_transpose = data.transpose()
    return data, data_transpose


def heat_corr(df, size=10):
    """Function creates heatmap of correlation matrix for each pair of columns
    in the dataframe.
    Input:
        df: pandas DataFrame
        size: vertical and horizontal size of the plot (in inch)
    """
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr, cmap='coolwarm')
    # setting ticks to column names
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.show()

    return


def norm(array):
    """ Returns array normalised to [0,1]. Array can be a numpy array
    or a column of a dataframe"""
    min_val = np.min(array)
    max_val = np.max(array)

    scaled = (array-min_val) / (max_val-min_val)

    return scaled


def norm_df(df, first=0, last=None):
    """
    Returns all columns of the dataframe normalised to [0,1] with the
    exception of the first (containing the names)
    Calls function norm to do the normalisation of one column, but
    doing all in one function is also fine.
    First, last: columns from first to last (including) are normalised.
    Defaulted to all. None is the empty entry. The default corresponds
    """

    # iterate over all numerical columns
    for col in df.columns[first:last]:
        # excluding the first column
        df[col] = norm(df[col])

    return df


def exp_growth(t, scale, growth):
    """ Computes exponential function with scale and growth as free parameters
    """
    f = scale * np.exp(growth * (t-1950))
    return f


# calling the functions
df_CO2, df_CO2_t = read("CO2_123.csv")

heat_corr(df_CO2, 10)

# plotting the scatterplot
pd.plotting.scatter_matrix(df_CO2, figsize=(9.0, 9.0))
plt.tight_layout()
plt.show()

# for normalising dataframe
df_fit = df_CO2[['1996 [YR1996]'	, '2017 [YR2017]']].copy()
df_fit = norm_df(df_fit)
print(df_fit.describe())

# calculating silhouette score
for ic in range(2, 12):
    kmeans = cluster.KMeans(n_clusters=ic)
    kmeans.fit(df_fit)
    labels = kmeans.labels_
    print(ic, skmet.silhouette_score(df_fit, labels))

# plotting scatter plot
kmeans = cluster.KMeans(n_clusters=2)
kmeans.fit(df_fit)
labels = kmeans.labels_
cen = kmeans.cluster_centers_
plt.figure(figsize=(6.0, 6.0))
plt.scatter(df_fit["1996 [YR1996]"],
            df_fit["2017 [YR2017]"], c=labels, cmap="Accent")

# finding the centroid of the cluster
for ic in range(2):
    xc, yc = cen[ic, :]
    plt.plot(xc, yc, "dk", markersize=10)

# labeling the cluster plot
plt.xlabel("1996 [YR1996]")
plt.ylabel("2017 [YR2017]")
plt.title("2 clusters")
plt.show()


def read(file_name):
    """
    This function is for reading the csv dataset returning the
    data frame, cleaning the data frame and transposing the dataset

    Returns
    -------
    Dataframes(original), Transposed Dataframes and Manipulated dataframe
    """
    data = pd.read_csv(file_name)
    data = data.set_index('Country Name')
    data = data.drop(['Indicator Name', 'Indicator Code',
                     'Country Code'], axis=1)
    data = data.fillna(0)
    data_transpose = data.transpose()
    data_final = data_transpose.tail(10)
    data_ult = data_final.head(9)
    return data, data_transpose, data_ult


# calling the function
df_GDP, df_GDP_t, df_GDP_ult = read("GDP_234.csv")

# converting into arrays
years = np.array(df_GDP_ult.index.values)
Portugal = np.array(df_GDP_ult.Portugal.values)

# fit exponential growth
GDP, covar = opt.curve_fit(exp_growth, years, Portugal)
print("Fit parameter", GDP)

# use *GDP to pass on the fit parameters
df_GDP_ult["gdp_exp"] = exp_growth(Portugal, *GDP)
plt.figure()
plt.plot(years, Portugal, label="data")
plt.plot(years, df_GDP_ult["gdp_exp"], label="fit")
plt.legend()
plt.title("First fitting")
plt.xlabel("years")
plt.ylabel("GDP of Portugal")
plt.show()
print()

# find a feasible start value the pedestrian way
# the scale factor is way too small. The exponential factor too large.
# Try scaling with the GDP in 2017 and a smaller exponential factor
# decrease or increase exponential factor until rough agreement is reached
# growth of -1.04193486e-20 gives a reasonable start value
GDP = [3.34582, -1.04193486e-20]
df_GDP_ult["gdp_exp"] = exp_growth(Portugal, *GDP)
plt.figure()
plt.plot(years, Portugal, label="data")
plt.plot(years, df_GDP_ult["gdp_exp"], label="fit")
plt.legend()
plt.xlabel("years")
plt.ylabel("GDP of Portugal")
plt.title("Improved start value")
plt.show()

# fit exponential growth
GDP, covar = opt.curve_fit(exp_growth, years, Portugal)
print("Fit parameter", GDP)
df_GDP_ult["gdp_exp"] = exp_growth(Portugal, *GDP)
plt.figure()
plt.plot(years, Portugal, label="data")
plt.plot(years, df_GDP_ult["gdp_exp"], label="fit")
plt.legend()
plt.xlabel("years")
plt.ylabel("GDP of Portugal")
plt.title("Final fitting")
plt.show()
print()
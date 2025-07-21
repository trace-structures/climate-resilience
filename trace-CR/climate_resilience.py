import json
import os
import earthkit.data
import xarray as xr
import pandas as pd
import cftime
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genextreme as gev
import pandas as pd
import numpy as np
from scipy.stats import genextreme as gev
# import pprint
import cdsapi
from IPython.display import display
import reverse_geocoder as rg
import folium
# from folium import plugins
# import rioxarray as rxr
# import earthpy as et
# import earthpy.spatial as es
import math
import time

base_dir = os.path.dirname(os.path.abspath(__file__))

def download_dfs_new(requests, dictionay_name):
    for key_name, request in requests:
        try:
            # "Download" the dataset and convert it into a xarray then convert the dataset into a dataframe
            ds = earthkit.data.from_source(
                "cds", "projections-cordex-domains-single-levels", request
            )
            # df = (
            #     ds.to_xarray(
            #         xarray_open_mfdataset_kwargs=dict(decode_cf=True, decode_times=True)
            #     )
            #     .to_dataframe()
            #     .reset_index()
            # )

            # Add to dictionary
            dictionay_name[key_name] = ds
            print(f"Dataset '{key_name}' downloaded and added to dictionary")

        except Exception as e:
            print(f"Error with the dataset '{key_name}': {e}")



def download_dfs(requests, dictionay_name):
    """
    Function to "download" as a temporary file with Earthkit and add the orography/variables of several climate models to the dictionary (dfs_orography or dfs_models).
    :param requests: the parameters of the request (those indicated in the Copernicus dataset: names RCM, GCM, resolution, etc.).
    Example of a request
    irnet_orography = {
    "domain": "europe",
    "experiment": "historical",
    "horizontal_resolution": "0_11_degree_x_0_11_degree",
    "temporal_resolution": "fixed",
    "variable": ["orography"],
    "gcm_model": "ichec_ec_earth",
    "rcm_model": "knmi_racmo22e",
    "ensemble_member": "r1i1p1"
    }
    :param dictionary_name: the name of the dictionary to save the dataset in.
    """
    for key_name, request in requests:
        try:
            # "Download" the dataset and convert it into a xarray then convert the dataset into a dataframe
            ds = earthkit.data.from_source(
                "cds", "projections-cordex-domains-single-levels", request
            )
            
            df = (
                ds.to_xarray(
                    xarray_open_mfdataset_kwargs=dict(decode_cf=True, decode_times=True)
                )
                .to_dataframe()
                .reset_index()
            )

            # Add to dictionary
            dictionay_name[key_name] = df
            print(f"Dataset '{key_name}' downloaded and added to dictionary")

        except Exception as e:
            print(f"Error with the dataset '{key_name}': {e}")


def print_dfs(dictionay_name):
    """
    Function to print the contents of the dictionary 'dfs_' legibly.
    It shows the first rows of each dataset.
    """
    if dictionay_name:
        for name, df in dictionay_name.items():
            print(f"Data for '{name}':")
            display(df.head())  # Print the first 5 rows of each DataFrame
            # display(df.head(20)) # Print the first 20 rows of each DataFrame
            print("\n" + "=" * 50 + "\n")
    else:
        print("The dictionary 'dfs_' is empty.")

def ev1_invcdf(y, location, scale):
    """
    Inverse cumulative distribution function (CDF) of the Gumbel distribution (EV1 Type).

    Parameters:
    y : Values to be transformed.
    location: Location parameter of the distribution.
    scale: Scale parameter of the distribution.

    Returns:
    x : The inverse CDF values.
    """
    y = np.array(y)  # Ensure it's a numpy array
    x = -np.log(-np.log(y)) * scale + location
    return x


def ev1_param(dfs_processed, T_r, method="MLM", zeroyear="No", CM="no"):
    R = {}
    parmhat = {}
    parmci = {}
    Q = {}
    std_dev = {}
    COV = {}
    FC = {}  #  the value for first time windows is R[key][0]

    p = 1 / T_r
    yp = -np.log(-np.log(1 - p))

    for dataset_name, windows in dfs_processed.items():
        for window, df in windows.items():
            key = f"{dataset_name}_{window}"
            if "tasmax" in df.columns:
                data = df["tasmax"].values
                value="Maxima"
            elif "tasmin" in df.columns:
                data = df["tasmin"].values
                value="Minima"
            elif "pr" in df.columns:
                data = df["pr"].values
                value="Maxima"
            elif "sfcWind" in df.columns:
                data = df["sfcWind"].values
                value="Maxima"


            # Keep only positive values (if required) and sort in descending order
            if zeroyear == "yes":
                if CM == "yes":
                    data = data[data > 0.05]
                else:
                    data = data[data > 0]
            data = np.sort(data)[::-1]  # Sorting in decrescending order

            n = len(data)
            pp = np.arange(n, 0, -1) / (n + 1)

            if method == "MLM":
                # Maximum likelihood estimation (Gumbel distribution: Type I Extreme Value)
                parmhat_1, parmhat_2 = gev.fit(data, f0=0)[1:]
                parmhat[key] = [parmhat_1, parmhat_2]
                R[key] = parmhat_1 + parmhat_2 * yp
                std_dev[key] = np.std(data)
                Q[key] = np.mean(data)
                COV[key] = np.std(data) / np.mean(data)

            elif method == "LSM":
                # Least squares method (Gumbel distribution: Type I Extreme Value)
                loglog_pp = np.log(-np.log(pp))
                parmhat_2 = (np.sum(data) ** 2 - n * np.sum(data**2)) / (
                    n * np.sum(data * loglog_pp) - np.sum(data) * np.sum(loglog_pp)
                )
                parmhat_1 = np.mean(data) + np.sum(loglog_pp * parmhat_2) / n
                parmhat[key] = [parmhat_1, parmhat_2]
                R[key] = parmhat_1 + parmhat_2 * yp
                std_dev[key] = np.std(data)
                Q[key] = np.mean(data)
                COV[key] = np.std(data) / np.mean(data)

            elif method == "MOM":
                # Method of moments
                mean = np.mean(data)
                std_dev_value = np.sqrt(
                    np.var(data)
                )  # Nota: per la campionaria, usa np.var(data, ddof=1)
                parmhat_2 = np.sqrt(6) / np.pi * std_dev_value
                parmhat_1 = mean - 0.5772157 * parmhat_2
                parmhat[key] = [parmhat_1, parmhat_2]
                R[key] = parmhat_1 + parmhat_2 * yp
                std_dev[key] = std_dev_value
                Q[key] = np.mean(data)
                COV[key] = np.std(data) / np.mean(data)

            if value == "Minima":
                parmhat[key] = [-parmhat[key][0], parmhat[key][1]]
                R[key] = -R[key]
                std_dev[key] = np.std(data)
                Q[key] = np.mean(data)
                COV[key] = np.std(data) / np.mean(data)

    # Evaluation of Factors of change
    # print("R.keys(): ", R.keys())
    reference_key = list(R.keys())[0]
    reference_value = R[reference_key]

    if "tasmax" in df.columns or "tasmin" in df.columns:
        for key, value in R.items():
            FC[key] = value - reference_value

    elif "pr" in df.columns or "sfcWind" in df.columns:
        for key, value in R.items():
            FC[key] = value/reference_value


    # Results
    results = {
        "R": R,
        "FC": FC,
        "std_dev": std_dev,
        "Q": Q,
        "COV": COV,
        "parmhat": parmhat,
        "parmci": parmci
    }

    # Print results
    # print("\n--- Results ---")
    # pprint.pprint(results)

    return results


# Funzione ev1_invcdf per calcolare i valori inversi
# def ev1_invcdf(p, loc, scale):
#    return loc - scale * np.log(-np.log(p))

def get_gumbel_plot(dfs_processed, T_r, window):
        # Dictionary
    parmhat_results = {}

    # Calculation of parameters for each metohd
    # for method in ["MOM", "MLM", "LSM", "ESLRP"]:
    for method in ["MOM", "MLM", "LSM"]:

        R, parmhat, parmci, Q, std_dev, COV, FC = ev1_param(
            dfs_processed,
            T_r=T_r,
            method=method,
            value="Maxima",
            zeroyear="No",
            CM="no",
        )
        key = list(parmhat.keys())[0]
        parmhat_results[method] = parmhat[key]
    # print("parmhat_results: ", parmhat_results)
    #
    dataset_name = list(dfs_processed.keys())[0]
    df = dfs_processed[dataset_name][window]
    # for dataset_name, windows in dfs_processed.items():
    #     for window, df in windows.items():
    #         if window == time_window:
    key = f"{dataset_name}_{window}"

    # Sort data in descending order
    data = df["tasmax"].dropna().values
    data = np.sort(data)[::-1]  # Decreasing
    # data = np.sort(data) # Growing

    # Some parameters useful for graph
    p = 1 / T_r
    yp = -np.log(-np.log(1 - p))
    n = len(data)
    pp = np.arange(n, 0, -1) / (n + 1)
    Tr = np.arange(1.01, 1000 + 10, 10)
    pt = -np.log(-np.log(1 - 1 / Tr))

    # Plot
    fig, ax = plt.subplots()
    ax.plot(
        data, -np.log(-np.log(pp)), "ko", markersize=3, label="Observed Data"
    )

    # Generation of curves for each method
    for method, parmhat in parmhat_results.items():
        zp = ev1_invcdf(1 - 1 / Tr, parmhat[0], parmhat[1])
        ax.plot(zp, pt, linewidth=2, label=f"Method {method}")

        R_method = parmhat[0] + parmhat[1] * yp
        ax.plot([R_method, R_method], [-2, yp], "--")

    # Horizontal line for Return Period (2% of probability of being exceeded in one year)
    ax.axhline(
        y=yp, linestyle="--", color="r", linewidth=2, label=f"T_r = {T_r} years"
    )

    formatted_window = "(" + window[0] + " - " + window[1] + ")"

    # Choose style
    ax.set_xlim([0, max(data) + 1])
    ax.set_ylim([-2, 5])
    ax.set_xlabel("Maximum Temperature [°C]", fontsize=12, fontweight="bold")
    ax.set_ylabel("Y", fontsize=12, fontweight="bold")
    ax.set_title("Gumbel Probability Paper {}".format(formatted_window), fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True)

    # Show plot
    return fig

def get_window_names(windows):
    names = []
    for _, window in enumerate(windows):
        start = window[0]
        end = window[1]
        names.append("{} - {}".format(start, end))
    return names

def get_gumbel_plot_new(dfs_processed, T_r, window):
        # Dictionary
    # parmhat_results = {}

    # Calculation of parameters for each metohd
    # for method in ["MOM", "MLM", "LSM", "ESLRP"]:
    # for method in ["MOM", "MLM", "LSM"]:

    #     R, parmhat, parmci, Q, std_dev, COV, FC = ev1_param(
    #         dfs_processed,
    #         T_r=T_r,
    #         method=method,
    #         value="Maxima",
    #         zeroyear="No",
    #         CM="no",
    #     )
    #     key = list(parmhat.keys())[0]
    #     parmhat_results[method] = parmhat[key]
    # print("parmhat_results: ", parmhat_results)
    #
    method_colors = {"MOM": 'blue', "MLM": 'purple', "LSM": 'green'}

    dataset_name = list(dfs_processed.keys())[0]
    df = dfs_processed[dataset_name][window]
    # for dataset_name, windows in dfs_processed.items():
    #     for window, df in windows.items():
    #         if window == time_window:
    key = f"{dataset_name}_{window}"

    # Sort data in descending order
    if "tasmin" in df.columns:
        data = df['tasmin'].dropna().values
        value = "Minima"
        variable = 'Minimum Temperature [°C]'

    elif "pr" in df.columns:
        data = df['pr'].dropna().values
        value = "Maxima"
        variable = "Precipitation [mm]"

    elif "sfcWind" in df.columns:
        data = df['sfcWind'].dropna().values
        value = "Maxima"
        variable = "Wind Speed [ms-1]"

    elif "tasmax" in df.columns:
        data = df['tasmax'].dropna().values
        value = "Maxima"
        variable = 'Maximum Temperature [°C]'
    data = np.sort(data)[::-1]  # Decreasing
    # data = np.sort(data) # Growing

    # Some parameters useful for graph
    p = 1 / T_r
    yp = -np.log(-np.log(1 - p))
    n = len(data)
    pp = np.arange(n, 0, -1) / (n + 1)
    Tr = np.arange(1.01, 1000 + 10, 10)
    pt = -np.log(-np.log(1 - 1 / Tr))

    # Plot
    fig, ax = plt.subplots()
    ax.plot(
        data, -np.log(-np.log(pp)), "ko", markersize=3, label="Observed Data"
    )

    intersection_x = []

    # Generation of curves for each method
    for method in ["MOM", "MLM", "LSM"]:
        # Calculate parameters for the current window and method
        # R, parmhat, parmci, Q, std_dev, COV, FC = ev1_param({dataset_name: {window: df}}, T_r=T_r, method=method, value="Maxima", zeroyear="No", CM="no")
        results = ev1_param({dataset_name: {window: df}}, T_r=T_r, method=method, zeroyear="No", CM="no")
        parmhat = list(results["parmhat"].values())[0]  # Extract parameter values
        
        # Calculate zp for the current method
        zp = ev1_invcdf(1 - 1 / Tr, parmhat[0], parmhat[1])
        ax.plot(zp, pt, linewidth=2, label=f"Method {method}", color=method_colors[method])
        
        # Calculate R_method and plot the dashed line
        R_method = parmhat[0] + parmhat[1] * yp
        ax.plot([R_method, R_method], [-2, yp], '--', color=method_colors[method])
        intersection_x.append(R_method)

    # Horizontal line for Return Period (2% of probability of being exceeded in one year)
    ax.axhline(
        y=yp, linestyle="--", color="r", linewidth=2, label=f"T_r = {T_r} years"
    )

    min_value = data.min()
    gap = (max(intersection_x) - min_value)*0.1

    # Choose style
    ax.set_xlim([min_value-gap, max(max(intersection_x), data.max())+gap])
    ax.set_ylim([-2, 5])
    ax.set_xlabel(variable, fontsize=10)
    ax.set_ylabel('Y', fontsize=10)
    ax.set_title("Gumbel Probability Paper ({} - {})".format(window[0], window[1]), fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True)

    # Show plot
    return fig



def single_model_outputs(results, value, dfs_processed, T_r=50, window=None):
    R = results.get("R", {})
    FC = results.get("FC", {})
    COV = results.get("COV", {})
    Q = results.get("Q", {})
    std_dev = results.get("std_dev", {})

    # Crea liste per le etichette (da rivedere)
    keys = list(R.keys())
    labels = []
    for i in range(len(keys)):
        start = keys[i][-11:-7]
        end = keys[i][-5:-1]
        labels.append(start + " - " + end)
    R_values = list(R.values())
    FC_values = list(FC.values())
    COV_values = list(COV.values())
    Q_values = list(Q.values())
    std_dev_values = list(std_dev.values())

    match value:
        case 'Characteristic Value':
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(keys, R_values, color="blue")
            ax.set_xlabel("Time Windows")
            ax.set_ylabel("R")
            fig.suptitle("Characteristic Values (R)")
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.grid(axis="y")
            fig.tight_layout()

            dataframe = pd.DataFrame(np.array(R_values).reshape(1, -1), columns=labels)
            return fig, dataframe

        case 'Factors of Change of Characteristic value':
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(keys, FC_values, color="green")
            ax.set_xlabel("Time Windows")
            ax.set_ylabel("FC [°C]")
            fig.suptitle("Factors of Change (FC)")
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.grid(axis="y")
            fig.tight_layout()
            dataframe = pd.DataFrame(np.array(FC_values).reshape(1, -1), columns=labels)
            return fig, dataframe

        case 'Coefficient of Variance':
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(keys, COV_values, color="purple")
            ax.set_xlabel("Time Windows")
            ax.set_ylabel("COV [-]")
            fig.suptitle("Coefficient of Variation (COV)")
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.grid(axis="y")
            fig.tight_layout()
            dataframe = pd.DataFrame(np.array(COV_values).reshape(1, -1), columns=labels)
            return fig, dataframe

        case 'Mean Value':
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(keys, Q_values, color="orange")
            ax.set_xlabel("Time Windows")
            ax.set_ylabel("Q [°C]")
            fig.suptitle("Mean Values (Q)")
            ax.set_xticks(np.arange(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.grid(axis="y")
            fig.tight_layout()
            dataframe = pd.DataFrame(np.array(Q_values).reshape(1, -1), columns=labels)
            return fig, dataframe
        
        case 'Gumbel Probability Paper':
            return get_gumbel_plot_new(dfs_processed, T_r, window)

def ev1_test_new(dfs_processed, T_r):
    """
    Plot and compare all the methods for evaluating characteristic values,
    for each window in the dataset.
    
    Parameters:
    - dfs_processed: Dictionary.
    - T_r: Return period.
    """

    method_colors = {"MOM": 'blue', "MLM": 'purple', "LSM": 'green'}

    for dataset_name, windows in dfs_processed.items():
        for window, df in windows.items():
            key = f"{dataset_name}_{window}"
            
            # Sort data in descending order
            if "tasmin" in df.columns:
                data = df['tasmin'].dropna().values

            elif "pr" in df.columns:
                data = df['pr'].dropna().values

            elif "sfcWind" in df.columns:
                data = df['sfcWind'].dropna().values

            elif "tasmax" in df.columns:
                data = df['tasmax'].dropna().values
                value = "Maxima"

            # data = df['tasmax'].dropna().values
            data = np.sort(data)[::-1]  # Decreasing
            
            # Some parameters useful for graph
            p = 1 / T_r
            yp = -np.log(-np.log(1 - p))
            n = len(data)
            pp = np.arange(n, 0, -1) / (n + 1)
            Tr = np.arange(1.01, 1000 + 10, 10)
            pt = -np.log(-np.log(1 - 1 / Tr))
            
            # Plot
            fig, ax = plt.subplots()
            ax.plot(data, -np.log(-np.log(pp)), 'ko', markersize=3, label='Observed Data')
            
            intersection_x = []
            # Calculate and plot curves for each method
            for method in ["MOM", "MLM", "LSM"]:
                # Calculate parameters for the current window and method
                R, parmhat, parmci, Q, std_dev, COV, FC = ev1_param({dataset_name: {window: df}}, T_r=T_r, method=method, value="Maxima", zeroyear="No", CM="no")
                parmhat = list(parmhat.values())[0]  # Extract parameter values
                
                # Calculate zp for the current method
                zp = ev1_invcdf(1 - 1 / Tr, parmhat[0], parmhat[1])
                ax.plot(zp, pt, linewidth=2, label=f"Method {method}", color=method_colors[method])
                
                # Calculate R_method and plot the dashed line
                R_method = parmhat[0] + parmhat[1] * yp
                ax.plot([R_method, R_method], [-2, yp], '--', color=method_colors[method])
                intersection_x.append(R_method)

            # Horizontal line for Return Period (T_r)
            ax.axhline(y=yp, linestyle='--', color='r', linewidth=2, label=f'T_r = {T_r} years')
            
            min_value = data.min()
            gap = (max(intersection_x) - min_value)*0.1

            # Configure plot style
            ax.set_xlim([min_value-gap, max(intersection_x)+gap])
            ax.set_ylim([-2, 5])
            ax.set_xlabel('Maximum Temperature [°C]', fontsize=10)
            ax.set_ylabel('Y', fontsize=10)
            ax.set_title(f'Gumbel Probability Paper', fontsize=12)
            ax.legend()
            ax.grid(True)
            
            # Show plot
            plt.show()



def ev1_test(dfs_processed, T_r):
    """
    Plot and compare all the methos fot evaluating characteristic values

    Parametrers:
    - dfs_processed: Dictionary.
    - T_r: Return period.
    """
    # Dictionary
    for dataset_name, windows in dfs_processed.items():
        for window, df in windows.items():
            key = f"{dataset_name}_{window}"

            # Sort data in descending order
            data = df["tasmax"].dropna().values
            data = np.sort(data)[::-1]  # Decreasing
            # data = np.sort(data) # Growing

            # Some parameters useful for graph
            p = 1 / T_r
            yp = -np.log(-np.log(1 - p))
            n = len(data)
            pp = np.arange(n, 0, -1) / (n + 1)
            Tr = np.arange(1.01, 1000 + 10, 10)
            pt = -np.log(-np.log(1 - 1 / Tr))

            # Plot
            fig, ax = plt.subplots()
            ax.plot(
                data, -np.log(-np.log(pp)), "ko", markersize=3, label="Observed Data"
            )

            # parmhat_results = {}

            # Calculation of parameters for each metohd
            # for method in ["MOM", "MLM", "LSM", "ESLRP"]:
            for method in ["MOM", "MLM", "LSM"]:

                R, parmhat, parmci, Q, std_dev, COV, FC = ev1_param(
                    dfs_processed,
                    T_r=T_r,
                    method=method,
                    value="Maxima",
                    zeroyear="No",
                    CM="no",
                )
                parmhat = list(parmhat.values())[0]
            #     key = list(parmhat.keys())[0]
            #     parmhat_results[method] = parmhat[key]
            # print("parmhat_results: ", parmhat_results)

            # Generation of curves for each method
            # for method, parmhat in parmhat_results.items():
                zp = ev1_invcdf(1 - 1 / Tr, parmhat[0], parmhat[1])
                ax.plot(zp, pt, linewidth=2, label=f"Method {method}")

                R_method = parmhat[0] + parmhat[1] * yp
                ax.plot([R_method, R_method], [-2, yp], "--")

            # Horizontal line for Return Period (2% of probability of being exceeded in one year)
            ax.axhline(
                y=yp, linestyle="--", color="r", linewidth=2, label=f"T_r = {T_r} years"
            )

            # Choose style
            ax.set_xlim([0, max(data) + 1])
            ax.set_ylim([-2, 5])
            ax.set_xlabel("Maximum Temperature [°C]", fontsize=12, fontweight="bold")
            ax.set_ylabel("Y", fontsize=12, fontweight="bold")
            ax.set_title("Gumbel Probability Paper", fontsize=12, fontweight="bold")
            ax.legend()
            ax.grid(True)

            # Show plot
            plt.show()


def find_extract_coords(
    city_coords, dfs_orography, mode="point", delta_lat_km=50, delta_lon_km=50
):
    """
    Find the closest point (or all points in a specified area) to the reference coordinates of one or more specified cities/buildngs
    and return a structured dictionary with named areas.

    :param city_coords: List of tuple (city_name, lat, lon).
    :param dfs_orography: DataFrame dictionary with columns ['lat', 'lon', 'rlat', 'rlon'].
    :param mode: 'point' for the nearest point, 'area' for a rectangular selection.
    :param delta_lat_km: Rectangular area extent in kilometers (latitude, used in mode='area').
    :param delta_lon_km: Rectangular area extension in kilometers (longitude, used in mode='area').
    :return: Dictionary with named areas for each model and city.
    """
    coordinates = {}

    for key, df in dfs_orography.items():
        for city_name, lat, lon in city_coords:
            print(f"\nProcessing city: {city_name} (Lat: {lat}, Lon: {lon})")

            distances = np.sqrt((df["lat"] - lat) ** 2 + (df["lon"] - lon) ** 2)
            min_idx = distances.idxmin()
            closest_point = {
                "city": city_name,
                "lat": float(df.loc[min_idx]["lat"]),
                "lon": float(df.loc[min_idx]["lon"]),
                "rlat": float(df.loc[min_idx]["rlat"]),
                "rlon": float(df.loc[min_idx]["rlon"]),
            }

            if mode == "point":
                area = [
                    closest_point["rlat"],
                    closest_point["rlon"],
                    closest_point["rlat"],
                    closest_point["rlon"],
                ]

            elif mode == "area":
                delta_lat = (
                    delta_lat_km / 111
                )  # Approximate conversion from km to degrees
                delta_lon = delta_lon_km / (111 * np.cos(np.radians(lat)))

                area_points = df[
                    (df["lat"] >= lat - delta_lat)
                    & (df["lat"] <= lat + delta_lat)
                    & (df["lon"] >= lon - delta_lon)
                    & (df["lon"] <= lon + delta_lon)
                ]

                if not area_points.empty:
                    area = [
                        float(area_points["rlat"].max()),
                        float(area_points["rlon"].min()),
                        float(area_points["rlat"].min()),
                        float(area_points["rlon"].max()),
                    ]
                else:
                    print(f"  No points found in the area for city {city_name}.")
                    continue

            area_name = f"area"
            coordinates[area_name] = area

            print(f"  Closest Point in {key}:")
            print(
                f"  Latitude: {closest_point['lat']:.6f}, Longitude: {closest_point['lon']:.6f}"
            )
            print(
                f"  rlat: {closest_point['rlat']:.6f}, rlon: {closest_point['rlon']:.6f}"
            )
            print(f"  Area {area_name}: {area}")

    return coordinates

def plot_fc_value(results, value):
    R = results.get("R", {})
    FC = results.get("FC", {})
    COV = results.get("COV", {})
    Q = results.get("Q", {})
    std_dev = results.get("std_dev", {})

    # Crea liste per le etichette (da rivedere)
    keys = list(R.keys())
    R_values = list(R.values())
    FC_values = list(FC.values())
    COV_values = list(COV.values())
    Q_values = list(Q.values())
    std_dev_values = list(std_dev.values())

    match value:
        case 'Characteristic Value':
            fig, ax = plt.subplot(figsize=(8, 5))
            ax.plot(keys, R_values, color="blue")
            ax.set_xlabel("Dataset/Window")
            ax.set_ylabel("R")
            fig.title("Characteristic Values (R)")
            fig.set_xticks(rotation=45, ha="right")
            fig.grid(axis="y")
            fig.tight_layout()
            return fig



def plot_fc(results):
    """
    Funzione per plottare i valori di R, FC, COV, Q e std_dev in grafici separati.

    Parametri:
    - results: dizionario contenente i risultati con chiavi "R", "FC", "COV", "Q", "std_dev".
               Es. results = {
                   "R": {...},
                   "FC": {...},
                   "COV": {...},
                   "Q": {...},
                   "std_dev": {...}
               }
    """
    # Estrai i dati
    R = results.get("R", {})
    FC = results.get("FC", {})
    COV = results.get("COV", {})
    Q = results.get("Q", {})
    std_dev = results.get("std_dev", {})

    # Crea liste per le etichette (da rivedere)
    keys = list(R.keys())
    R_values = list(R.values())
    FC_values = list(FC.values())
    COV_values = list(COV.values())
    Q_values = list(Q.values())
    std_dev_values = list(std_dev.values())

    figures = []

    # 1. Grafico di R
    fig1 = plt.figure(figsize=(8, 5))
    plt.plot(keys, R_values, color="blue")
    plt.xlabel("Dataset/Window")
    plt.ylabel("R")
    plt.title("Characteristic Values (R)")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y")
    plt.tight_layout()
    figures.append(fig1)
    plt.show()

    # 2. Grafico di FC
    fig2 = plt.figure(figsize=(8, 5))
    plt.plot(keys, FC_values, color="green")
    plt.xlabel("Dataset/Window")
    plt.ylabel("FC")
    plt.title("Factors of Change (FC)")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y")
    plt.tight_layout()
    figures.append(fig2)
    plt.show()


    # 3. Grafico di COV
    fig3 = plt.figure(figsize=(8, 5))
    plt.plot(keys, COV_values, color="purple")
    plt.xlabel("Dataset/Window")
    plt.ylabel("Coefficient of Variation (COV)")
    plt.title("Coefficient of Variation (COV)")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y")
    plt.tight_layout()
    figures.append(fig3)
    plt.show()


    # 4. Grafico di Q
    fig4 = plt.figure(figsize=(8, 5))
    plt.plot(keys, Q_values, color="orange")
    plt.xlabel("Dataset/Window")
    plt.ylabel("Mean Values (Q)")
    plt.title("Mean Values (Q)")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y")
    plt.tight_layout()
    figures.append(fig4)
    plt.show()


    # 5. Grafico di std_dev
    fig5 = plt.figure(figsize=(8, 5))
    plt.plot(keys, std_dev_values, color="red")
    plt.xlabel("Dataset/Window")
    plt.ylabel("Standard Deviation (std_dev)")
    plt.title("Standard Deviation (std_dev)")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y")
    plt.tight_layout()
    figures.append(fig5)
    plt.show()


    return figures


def processed_dataframe_new(dfs_model, window_length=40, shift=10):
    # Column to remove
    columns_to_drop = [
        "bnds",
        "rlat",
        "rlon",
        "rotated_pole",
        "time_bnds",
        "height",
        "lon",
        "lat",
    ]

    # model_name = list(dfs_model.keys())[0]
    # ds = dfs_model[model_name]
    # ds = ds.to_xarray(xarray_open_mfdataset_kwargs=dict(decode_cf=True, decode_times=True))
    # ds = xr.open_mfdataset(
    #     ds,
    #     concat_dim="valid_time",
    #     combine="nested",
    #     chunks="auto",
    #     parallel=True,
    #     engine="netcdf4")
    # ds = ds.compute()

    # ds = ds.reset_index()

    # dfs_model[model_name] = ds


    # Dictionary for modified dataframe
    dfs_processed = {}

    # Iteration on all dataframe of the dictionary dfs_model
    for name, df in dfs_model.items():
        # Convert column time in a 'datetime' format
        if isinstance(df["time"].iloc[0], cftime.datetime):
            # Se è un oggetto cftime, converti prima in stringa
            df["time"] = df["time"].astype(str)

        # Handle errors
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

        # Remove rows in which 'bnds' is equal to 0
        df = df[df["bnds"] != 0]

        # Remove specified columns
        df = df.drop(columns=columns_to_drop, errors="ignore")

        # Turns values > 10^20 into NaN (for each column eccept 'time'
        for col in df.columns:
            if col != "time":  # Ignora la colonna 'time'
                df[col] = df[col].apply(lambda x: np.nan if x > 1e20 else x)

        # Convert the units of measurement
        if "tasmax" in df.columns:
            df["tasmax"] = df["tasmax"] - 273.15

        if "tasmin" in df.columns:
            df["tasmin"] = df["tasmin"] - 273.15

        if "pr" in df.columns:
            df["pr"] = df["pr"] * 100000

        # Add the other climatic variables

        # Calculation of annual maximum (minimum) values for each variable
        df["year"] = df["time"].dt.year
        # if extreme_type == "Maxima":
        # annual_extreme = df.groupby("year").max()
        annual_extreme = df.groupby("year").max()
        if "tasmin" in df.columns:
            annual_min_tasmin = df.groupby("year")["tasmin"].min()
            annual_extreme["tasmin"] = annual_min_tasmin

        if "pr" in df.columns:
            annual_extreme_pr = df.groupby("year")["pr"].max()
            annual_extreme["pr"] = annual_extreme_pr
        
        if "sfcWind" in df.columns:
            annual_extreme_sfcWind = df.groupby("year")["sfcWind"].max()
            annual_extreme["sfcWind"] = annual_extreme_sfcWind

        if "tasmax" in df.columns:
            annual_extreme_tasmax = df.groupby("year")["tasmax"].max()
            annual_extreme["tasmax"] = annual_extreme_tasmax

        # Time windows
        max_year = df["year"].iloc[-1]
        windows = [
            (start, start + window_length - 1)
            for start in range(1950, round(max_year), shift)
            if start + window_length - 1 <= max_year
        ]

        windowed_data = {}
        for start, end in windows:
            mask = (annual_extreme.index >= start) & (annual_extreme.index <= end)
            df_window = annual_extreme[mask]
            windowed_data[(start, end)] = df_window

            # Display results for each window
            print(f"Dataset: {name}, Window: {start}-{end}")
            # display(df_window.head())

        # Save the modified dataframes in the new dictionary
        dfs_processed[name] = windowed_data

    return dfs_processed, windows



def processed_dataframe(dfs_model, window_length=40, shift=10):
    # Column to remove
    columns_to_drop = [
        "bnds",
        "rlat",
        "rlon",
        "rotated_pole",
        "time_bnds",
        "height",
        "lon",
        "lat",
    ]

    # Dictionary for modified dataframe
    dfs_processed = {}

    # Iteration on all dataframe of the dictionary dfs_model
    for name, df in dfs_model.items():
        # Convert column time in a 'datetime' format
        if isinstance(df["time"].iloc[0], cftime.datetime):
            # Se è un oggetto cftime, converti prima in stringa
            df["time"] = df["time"].astype(str)

        # Handle errors
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

        # Remove rows in which 'bnds' is equal to 0
        df = df[df["bnds"] != 0]

        # Remove specified columns
        df = df.drop(columns=columns_to_drop, errors="ignore")

        # Turns values > 10^20 into NaN (for each column eccept 'time'
        for col in df.columns:
            if col != "time":  # Ignora la colonna 'time'
                df[col] = df[col].apply(lambda x: np.nan if x > 1e20 else x)

        # Convert the units of measurement
        if "tasmax" in df.columns:
            df["tasmax"] = df["tasmax"] - 273.15

        if "tasmin" in df.columns:
            df["tasmin"] = df["tasmin"] - 273.15

        

        # Add the other climatic variables

        # Calculation of annual maximum (minimum) values for each variable
        df["year"] = df["time"].dt.year
        annual_max = df.groupby("year").max()
        if "tasmin" in df.columns:
            annual_min_tasmin = df.groupby("year")["tasmin"].min()
            annual_max["tasmin"] = annual_min_tasmin

        # Time windows
        min_year, max_year = df["year"].min(), df["year"].max()
        windows = [
            (start, start + window_length - 1)
            for start in range(1950, max_year, shift)
            if start + window_length - 1 <= max_year
        ]

        windowed_data = {}
        for start, end in windows:
            mask = (annual_max.index >= start) & (annual_max.index <= end)
            df_window = annual_max[mask]
            windowed_data[(start, end)] = df_window

            # Display results for each window
            print(f"Dataset: {name}, Window: {start}-{end}")
            display(df_window.head())

        # Save the modified dataframes in the new dictionary
        dfs_processed[name] = windowed_data

    return dfs_processed


base_dir = os.path.dirname(os.path.abspath(__file__))


def download_all_models():
    cdsapi_rc_path = os.path.join(base_dir, ".cdsapirc")

    os.environ["CDSAPI_RC"] = cdsapi_rc_path

    # Read models from json file
    with open(os.path.join(base_dir, "climate_models_good.json")) as f:
        models = json.load(f)
    
    downloaded_models = {}
    # Iterate through the models and download the data
    for model in models:
        gcm = models[model]["gcm"]
        rcm = models[model]["rcm"]
        ensemble_orography = models[model]["ensemble_orography"]
        ensemble_model = models[model]["ensemble_model"]
        start_years = models[model]["start_years"]
        end_years = models[model]["end_years"]
        requests_orography = [
            (
                "{}_orography".format(model),
                {
                    "domain": "europe",
                    "experiment": "historical",
                    "horizontal_resolution": "0_11_degree_x_0_11_degree",
                    "temporal_resolution": "fixed",
                    "variable": ["orography"],
                    "gcm_model": gcm,
                    "rcm_model": rcm,
                    "ensemble_member": ensemble_orography,
                },
            ),
        ]
        # Define the name of the dictionary containing orography variables for the whole Europe Domain
        dfs_orography = {}

        # Download of the dictionary
        download_dfs(requests_orography, dfs_orography)

        # Print the dictionary
        print_dfs(dfs_orography)

        city_coords = [
            ("Florence", 43.7697, 11.2558),  # Name of the city, latitude, longitude
            # ("Berlin", 52.5200, 13.4050),
        ]

        coordinates = find_extract_coords(
            city_coords, dfs_orography, mode="point", delta_lat_km=20, delta_lon_km=20
        )
        request_models = [
            (
                model,
                {
                    "domain": "europe",  # fixed
                    "experiment": ["historical", "rcp_4_5"],
                    "horizontal_resolution": "0_11_degree_x_0_11_degree",  # fixed
                    "temporal_resolution": "daily_mean",  # fixed
                    "variable": "maximum_2m_temperature_in_the_last_24_hours",
                    "gcm_model": gcm,
                    "rcm_model": rcm,
                    "ensemble_member": ensemble_model,
                    "start_year": start_years,
                    "end_year": end_years,
                    "area": coordinates.get(
                        "area_irnet_orography_florence"
                    ),  # [latitude_max longitude_min latitude_min longitude_max]
                },
            ),
        ]
        dfs_models = {}

        print(model)
        # Download of the dictionary
        download_dfs(request_models, dfs_models)

        downloaded_models[model] = dfs_models
    
    return downloaded_models


def multi_model_params(models, window_length, shift, T_r):
    keys = list(models.keys())
    result_dict = {}
    for i in range(len(keys)):
        dfs_processed = processed_dataframe(models[keys[i]], window_length, shift)
        R, parmhat, parmci, Q, std_dev, COV, FC = ev1_param(dfs_processed, T_r, method="LSM", value="Maxima", zeroyear="No", CM="no")
        results = {"R": R, "FC": FC, "COV": COV, "Q": Q, "std_dev": std_dev}
        result_dict[keys[i]] = results
    return result_dict

def multi_model_plots(results, value):
    keys = list(results.keys())
    # R = results.get("R", {})
    # FC = results.get("FC", {})
    # COV = results.get("COV", {})
    # Q = results.get("Q", {})
    # std_dev = results.get("std_dev", {})
    label_values = list(results[keys[0]].get("R", {}).keys())
    # # Crea liste per le etichette (da rivedere)
    # keys = list(R.keys())
    labels = []
    for i in range(len(label_values)):
        start = label_values[i][-11:-7]
        end = label_values[i][-5:-1]
        labels.append(start + " - " + end)
    # R_values = list(R.values())
    # FC_values = list(FC.values())
    # COV_values = list(COV.values())
    # Q_values = list(Q.values())
    # std_dev_values = list(std_dev.values())

    match value:
        case 'Characteristic Value':
            fig, ax = plt.subplots(figsize=(8, 5))
            all_data = np.zeros((len(keys), len(label_values)))
            for i in range(len(keys)):
                R = results[keys[i]].get("R", {})
                R_values = list(R.values())
                ax.plot(label_values, R_values, label=keys[i])
                all_data[i, :] = R_values

            mean = np.mean(all_data, axis=0)
            ax.plot(label_values, mean, label='mean R', color='black')

            ax.set_xlabel("Time Windows")
            ax.set_ylabel("R")
            fig.suptitle("Characteristic Values (R)")
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.grid(axis="y")
            fig.tight_layout()
            fig.legend()

            dataframe = pd.DataFrame(all_data, columns=labels)
            return fig, dataframe

        case 'Factors of Change of Characteristic value':
            fig, ax = plt.subplots(figsize=(8, 5))
            all_data = np.zeros((len(keys), len(label_values)))
            for i in range(len(keys)):
                FC = results[keys[i]].get("FC", {})
                FC_values = list(FC.values())
                ax.plot(label_values, FC_values, label=keys[i])
                all_data[i, :] = FC_values

            mean = np.mean(all_data, axis=0)
            ax.plot(label_values, mean, label='mean FC', color='black')

            ax.set_xlabel("Time Windows")
            ax.set_ylabel("FC [°C]")
            fig.suptitle("Factors of Change (FC)")
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.grid(axis="y")
            fig.tight_layout()
            fig.legend()

            dataframe = pd.DataFrame(all_data, columns=labels)
            return fig, dataframe

            # fig, ax = plt.subplots(figsize=(8, 5))
            # ax.plot(keys, FC_values, color="green")
            # ax.set_xlabel("Time Windows")
            # ax.set_ylabel("FC [°C]")
            # fig.suptitle("Factors of Change (FC)")
            # ax.set_xticklabels(labels, rotation=45, ha="right")
            # ax.grid(axis="y")
            # fig.tight_layout()
            # dataframe = pd.DataFrame(np.array(FC_values).reshape(1, -1), columns=labels)
            # return fig, dataframe

        case 'Coefficient of Variance':
            fig, ax = plt.subplots(figsize=(8, 5))
            all_data = np.zeros((len(keys), len(label_values)))
            for i in range(len(keys)):
                COV = results[keys[i]].get("COV", {})
                COV_values = list(COV.values())
                ax.plot(label_values, COV_values, label=keys[i])
                all_data[i, :] = COV_values

            # mean = np.mean(all_data, axis=0)
            # ax.plot(label_values, mean, label='mean COV', color='black')

            ax.set_xlabel("Time Windows")
            ax.set_ylabel("COV [-]")
            fig.suptitle("Coefficient of Variation (COV)")
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.grid(axis="y")
            fig.tight_layout()
            fig.legend()

            dataframe = pd.DataFrame(all_data, columns=labels)
            return fig, dataframe

        #     fig, ax = plt.subplots(figsize=(8, 5))
        #     ax.plot(keys, COV_values, color="purple")
        #     ax.set_xlabel("Time Windows")
        #     ax.set_ylabel("COV [-]")
        #     fig.suptitle("Coefficient of Variation (COV)")
        #     ax.set_xticklabels(labels, rotation=45, ha="right")
        #     ax.grid(axis="y")
        #     fig.tight_layout()
        #     dataframe = pd.DataFrame(np.array(COV_values).reshape(1, -1), columns=labels)
        #     return fig, dataframe
    

        case 'Mean Value':
            fig, ax = plt.subplots(figsize=(8, 5))
            all_data = np.zeros((len(keys), len(label_values)))
            for i in range(len(keys)):
                Q = results[keys[i]].get("Q", {})
                Q_values = list(Q.values())
                ax.plot(label_values, Q_values, label=keys[i])
                all_data[i, :] = Q_values

            mean = np.mean(all_data, axis=0)
            ax.plot(label_values, mean, label='mean Q', color='black')

            ax.set_xlabel("Time Windows")
            ax.set_ylabel("Q [°C]")
            fig.suptitle("Mean Values (Q)")
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.grid(axis="y")
            fig.tight_layout()
            fig.legend()

            dataframe = pd.DataFrame(all_data, columns=labels)
            return fig, dataframe

            # fig, ax = plt.subplots(figsize=(8, 5))
            # ax.plot(keys, Q_values, color="orange")
            # ax.set_xlabel("Time Windows")
            # ax.set_ylabel("Q [°C]")
            # fig.suptitle("Mean Values (Q)")
            # ax.set_xticklabels(labels, rotation=45, ha="right")
            # ax.grid(axis="y")
            # fig.tight_layout()
            # dataframe = pd.DataFrame(np.array(Q_values).reshape(1, -1), columns=labels)
            # return fig, dataframe



def run_all_models():
    cdsapi_rc_path = os.path.join(base_dir, ".cdsapirc")

    os.environ["CDSAPI_RC"] = cdsapi_rc_path

    R_dict = {}
    Q_dict = {}
    std_dev_dict = {}
    COV_dict = {}
    FC_dict = {}
    # Read models from json file
    with open(os.path.join(base_dir, "climate_models.json")) as f:
        models = json.load(f)
    # Iterate through the models and download the data
    for model in models:
        gcm = models[model]["gcm"]
        rcm = models[model]["rcm"]
        ensemble_orography = models[model]["ensemble_orography"]
        ensemble_model = models[model]["ensemble_model"]
        start_years = models[model]["start_years"]
        end_years = models[model]["end_years"]
        requests_orography = [
            (
                "{}_orography".format(model),
                {
                    "domain": "europe",
                    "experiment": "historical",
                    "horizontal_resolution": "0_11_degree_x_0_11_degree",
                    "temporal_resolution": "fixed",
                    "variable": ["orography"],
                    "gcm_model": gcm,
                    "rcm_model": rcm,
                    "ensemble_member": ensemble_orography,
                },
            ),
        ]
        # Define the name of the dictionary containing orography variables for the whole Europe Domain
        dfs_orography = {}

        # Download of the dictionary
        download_dfs(requests_orography, dfs_orography)

        # Print the dictionary
        print_dfs(dfs_orography)

        city_coords = [
            ("Florence", 43.7697, 11.2558),  # Name of the city, latitude, longitude
            # ("Berlin", 52.5200, 13.4050),
        ]

        coordinates = find_extract_coords(
            city_coords, dfs_orography, mode="point", delta_lat_km=20, delta_lon_km=20
        )
        request_models = [
            (
                model,
                {
                    "domain": "europe",  # fixed
                    "experiment": ["historical", "rcp_4_5"],
                    "horizontal_resolution": "0_11_degree_x_0_11_degree",  # fixed
                    "temporal_resolution": "daily_mean",  # fixed
                    "variable": "maximum_2m_temperature_in_the_last_24_hours",
                    "gcm_model": gcm,
                    "rcm_model": rcm,
                    "ensemble_member": ensemble_model,
                    "start_year": start_years,
                    "end_year": end_years,
                    "area": coordinates.get(
                        "area_irnet_orography_florence"
                    ),  # [latitude_max longitude_min latitude_min longitude_max]
                },
            ),
        ]
        dfs_models = {}

        print(model)
        # Download of the dictionary
        download_dfs(request_models, dfs_models)
        # Print the dictionary
        print_dfs(dfs_models)

        # Extract yearly maxima for each variable, divide into time windows and standardizes the units of measurement.
        dfs_processed = processed_dataframe(dfs_models, window_length=40, shift=10)
        R, parmhat, parmci, Q, std_dev, COV, FC = ev1_param(
            dfs_processed,
            50,
            method="LSM",
            value="Maxima",
            zeroyear="No",
            CM="no",
        )
        R_dict[model] = R
        Q_dict[model] = Q
        std_dev_dict[model] = std_dev
        COV_dict[model] = COV
        FC_dict[model] = FC

        print("{} done!".format(model))
        # Plot Gumbel Probability Paper
        # ev1_test(dfs_processed, T_r=50)
        # # Plot q_k, mean values, standard deviations, COV, FC
        # # Dictionary
        # results = {"R": R, "FC": FC, "COV": COV, "Q": Q, "std_dev": std_dev}
        # plot_fc(results)
    return R_dict, Q_dict, std_dev_dict, COV_dict, FC_dict

def get_map(latitude, longitude, city, country):
    f = folium.Figure(width=500, height=500)
    m = folium.Map(location=[latitude, longitude], zoom_start=10).add_to(f)
    folium.Marker(
        location=[latitude, longitude],
        popup=folium.Popup('<b>{}, {}</b><br>({} {})'.format(city, country, format_coordinates(latitude, "latitude"), format_coordinates(longitude, "longitude")), max_width=400, min_width=80), # pop-up label for the marker
        icon=folium.Icon()
    ).add_to(m)
    return f

def request_orography(api_key, model_name, experiment_option, variable_option, latitude, longitude, city):
    filename = ".cdsapirc_{}".format(api_key)
    with open(base_dir + '/' + filename, "w") as file:
        file.write("url: https://cds.climate.copernicus.eu/api\nkey: {}".format(api_key))

    cdsapi_rc_path = os.path.join(base_dir, filename)

    os.environ["CDSAPI_RC"] = cdsapi_rc_path

    with open(base_dir + "/request_data/unified_models_with_request_by_variable_new.json") as f:
        models = json.load(f)

    gcm = models[experiment_option][variable_option][model_name]["gcm"]
    rcm = models[experiment_option][variable_option][model_name]["rcm"]
    ensemble_orography = models[experiment_option][variable_option][model_name]["ensemble_orography"]

    requests_orography = [
        (
            "{}_orography".format(model_name),
            {
                "domain": "europe",
                "experiment": "historical",
                "horizontal_resolution": "0_11_degree_x_0_11_degree",
                "temporal_resolution": "fixed",
                "variable": ["orography"],
                "gcm_model": gcm,
                "rcm_model": rcm,
                "ensemble_member": ensemble_orography,
            },
        ),
    ]

    # Define the name of the dictionary containing orography variables for the whole Europe Domain
    dfs_orography = {}

    # Download of the dictionary
    download_dfs(requests_orography, dfs_orography)

    # Print the dictionary
    print_dfs(dfs_orography)
    city_coords = [
        (city, latitude, longitude)  # Name of the city, latitude, longitude
    ]

    coordinates = find_extract_coords(
        city_coords, dfs_orography, mode="point", delta_lat_km=20, delta_lon_km=20
    )
    print("coordinates: ", coordinates)
    os.remove(cdsapi_rc_path)
    return coordinates
    
def request_model(api_key, model_name, experiment_option, variable_option, coordinates):
    filename = ".cdsapirc_{}".format(api_key)
    with open(base_dir + '/' + filename, "w") as file:
        file.write("url: https://cds.climate.copernicus.eu/api\nkey: {}".format(api_key))

    cdsapi_rc_path = os.path.join(base_dir, filename)

    os.environ["CDSAPI_RC"] = cdsapi_rc_path

    with open(base_dir + "/request_data/unified_models_with_request_by_variable_new.json") as f:
        models = json.load(f)

    experiment = models[experiment_option]["experiment"]
    variable = models[experiment_option][variable_option]["variable"]

    gcm = models[experiment_option][variable_option][model_name]["gcm"]
    rcm = models[experiment_option][variable_option][model_name]["rcm"]
    ensemble_model = models[experiment_option][variable_option][model_name]["ensemble_model"]
    start_years = models[experiment_option][variable_option][model_name]["start_year"]
    end_years = models[experiment_option][variable_option][model_name]["end_year"]

    request_models = [
        (
            model_name,
            {
                "domain": "europe",  # fixed
                "experiment": experiment,
                "horizontal_resolution": "0_11_degree_x_0_11_degree",  # fixed
                "temporal_resolution": "daily_mean",  # fixed
                "variable": variable,
                "gcm_model": gcm,
                "rcm_model": rcm,
                "ensemble_member": ensemble_model,
                "start_year": start_years,
                "end_year": end_years,
                "area": coordinates.get("area"),
            },
        )
    ]
    dfs_models = {}

    print(model_name)
    # Download of the dictionary
    download_dfs(request_models, dfs_models)
    print(dfs_models)
    # print_dfs(dfs_models)
    os.remove(cdsapi_rc_path)

    return dfs_models


def request_orography_multiple(api_key, model_name, experiment_option, variable_option, latitude, longitude, city):
    filename = ".cdsapirc_{}".format(api_key)
    with open(filename, "w") as file:
        file.write("url: https://cds.climate.copernicus.eu/api\nkey: {}".format(api_key))

    cdsapi_rc_path = os.path.join(base_dir, filename)

    os.environ["CDSAPI_RC"] = cdsapi_rc_path

    with open("unified_models_with_request_by_variable_test.json") as f:
        models = json.load(f)        

    gcm = models[variable_option][model_name]["gcm"]
    rcm = models[variable_option][model_name]["rcm"]
    ensemble_orography = models[variable_option][model_name]["ensemble_orography"]

    requests_orography = [
        (
            "{}_orography".format(model_name),
            {
                "domain": "europe",
                "experiment": "historical",
                "horizontal_resolution": "0_11_degree_x_0_11_degree",
                "temporal_resolution": "fixed",
                "variable": ["orography"],
                "gcm_model": gcm,
                "rcm_model": rcm,
                "ensemble_member": ensemble_orography,
            },
        ),
    ]

    # Define the name of the dictionary containing orography variables for the whole Europe Domain
    dfs_orography = {}

    # Download of the dictionary
    download_dfs(requests_orography, dfs_orography)

    # Print the dictionary
    print_dfs(dfs_orography)
    city_coords = [
        (city, latitude, longitude)  # Name of the city, latitude, longitude
    ]

    coordinates = find_extract_coords(
        city_coords, dfs_orography, mode="point", delta_lat_km=20, delta_lon_km=20
    )
    print("coordinates: ", coordinates)
    os.remove(cdsapi_rc_path)
    return coordinates
    
def request_model_multiple(api_key, model_name, experiment_option, variable_option, coordinates):
    filename = ".cdsapirc_{}".format(api_key)
    with open(filename, "w") as file:
        file.write("url: https://cds.climate.copernicus.eu/api\nkey: {}".format(api_key))

    cdsapi_rc_path = os.path.join(base_dir, filename)

    os.environ["CDSAPI_RC"] = cdsapi_rc_path

    with open("unified_models_with_request_by_variable_test.json") as f:
        models = json.load(f)

    experiment = ["historical", "rcp_4_5"]
    variable = models[variable_option]['variable']

    gcm = models[variable_option][model_name]["gcm"]
    rcm = models[variable_option][model_name]["rcm"]
    ensemble_model = models[variable_option][model_name]["ensemble_model"]
    start_years = models[variable_option][model_name]["start_years"]
    end_years = models[variable_option][model_name]["end_years"]

    request_models = [
        (
            model_name,
            {
                "domain": "europe",  # fixed
                "experiment": experiment,
                "horizontal_resolution": "0_11_degree_x_0_11_degree",  # fixed
                "temporal_resolution": "daily_mean",  # fixed
                "variable": variable,
                "gcm_model": gcm,
                "rcm_model": rcm,
                "ensemble_member": ensemble_model,
                "start_year": start_years,
                "end_year": end_years,
                "area": coordinates.get("area"),
            },
        )
    ]
    dfs_models = {}

    print(model_name)
    # Download of the dictionary
    download_dfs(request_models, dfs_models)
    print(dfs_models)
    # print_dfs(dfs_models)
    os.remove(cdsapi_rc_path)

    return dfs_models

def request_multiple_models(api_key, experiment_option, variable_option, latitude, longitude, city):
    with open("unified_models_with_request_by_variable_test.json") as f:
        models = json.load(f)
    model_names = list(models[variable_option].keys())
    all_data = {}
    for i in range(1, len(model_names)):
        tick = time.time()
        coordinates = request_orography_multiple(api_key, model_names[i], experiment_option, variable_option, latitude, longitude, city)
        model_data = request_model_multiple(api_key, model_names[i], experiment_option, variable_option, coordinates)
        elapsed = time.time() - tick
        all_data[model_names[i]] = model_data[model_names[i]]
        print(model_names[i], i, "elapsed time: ", elapsed)
        print("done!")
    return all_data

def request_data(api_key, model_name, experiment_option, variable_option, latitude, longitude, city):
    coordinates = request_orography(api_key, model_name, experiment_option, variable_option, latitude, longitude, city)
    model_data = request_model(api_key, model_name, experiment_option, variable_option, coordinates)
    return model_data

def request_climate_model(coordinates):
    cdsapi_rc_path = os.path.join(base_dir, ".cdsapirc")

    os.environ["CDSAPI_RC"] = cdsapi_rc_path

    request_models = [
        (
            "irnet_model",
            {
                "domain": "europe",  # fixed
                "experiment": ["historical", "rcp_4_5"],
                "horizontal_resolution": "0_11_degree_x_0_11_degree",  # fixed
                "temporal_resolution": "daily_mean",  # fixed
                "variable": "minimum_2m_temperature_in_the_last_24_hours",
                "gcm_model": "ichec_ec_earth",
                "rcm_model": "knmi_racmo22e",
                "ensemble_member": "r1i1p1",
                "start_year": [
                    "1950",
                    "1951",
                    "1956",
                    "1961",
                    "1966",
                    "1971",
                    "1976",
                    "1981",
                    "1986",
                    "1991",
                    "1996",
                    "2001",
                    "2006",
                    "2011",
                    "2016",
                    "2021",
                    "2026",
                    "2031",
                    "2036",
                    "2041",
                    "2046",
                    "2051",
                    "2056",
                    "2061",
                    "2066",
                    "2071",
                    "2076",
                    "2081",
                    "2086",
                    "2091",
                    "2096",
                ],
                "end_year": [
                    "1950",
                    "1955",
                    "1960",
                    "1965",
                    "1970",
                    "1975",
                    "1980",
                    "1985",
                    "1990",
                    "1995",
                    "2000",
                    "2005",
                    "2010",
                    "2015",
                    "2020",
                    "2025",
                    "2030",
                    "2035",
                    "2040",
                    "2045",
                    "2050",
                    "2055",
                    "2060",
                    "2065",
                    "2070",
                    "2075",
                    "2080",
                    "2085",
                    "2090",
                    "2095",
                    "2100",
                ],
                "area": coordinates.get(
                    "area"
                ),  # [latitude_max longitude_min latitude_min longitude_max]
            },
        )
    ]

    dfs_models = {}

    download_dfs(request_models, dfs_models)
    # Print the dictionary
    print_dfs(dfs_models)

    return dfs_models

def format_coordinates(coordinate, type="latitude"):
    abs_degrees = abs(coordinate)
    degrees = math.floor(abs_degrees)
    minutes = math.floor(60*(abs_degrees-degrees))
    seconds = round(3600 * (abs_degrees-degrees) - 60*minutes)
    if type == "latitude":
        if coordinate < 0:
            return """{}° {}' {}" S""".format(degrees, minutes, seconds)
        else:
            return """{}° {}' {}" N""".format(degrees, minutes, seconds)
    else:
        if coordinate < 0:
            return """{}° {}' {}" W""".format(degrees, minutes, seconds)
        else:
            return """{}° {}' {}" E""".format(degrees, minutes, seconds)

def coordinate_check(latitude, longitude):
    if latitude >= 34.80 and latitude <= 81.8067 and longitude >= -28.8333 and longitude <= 69.0334:
        coordinates = (latitude, longitude)
        location = rg.search(coordinates)
        country = location[0]['cc']
        city = location[0]['name']
        return city, country
    else:
        raise ValueError("Climate resilience analysis is not supported in the selected region. Please select European coordinates.")

if __name__ == "__main__":
    cdsapi_rc_path = os.path.join(base_dir, ".cdsapirc")

    os.environ["CDSAPI_RC"] = cdsapi_rc_path

    client = cdsapi.Client()
    requests_orography = [
        (
            "irnet_orography",
            {
                "domain": "europe",
                "experiment": "historical",  # Not necessary to specify other experitments, since the orography is the same for RCP4.5 and RCP8.5
                "horizontal_resolution": "mean_precipitation_flux",
                "temporal_resolution": "fixed",
                "variable": ["orography"],
                "gcm_model": "ichec_ec_earth",
                "rcm_model": "knmi_racmo22e",
                "ensemble_member": "r1i1p1",
            },
        ),
    ]

    # Define the name of the dictionary containing orography variables for the whole Europe Domain
    dfs_orography = {}

    # Download of the dictionary
    download_dfs(requests_orography, dfs_orography)

    # Print the dictionary
    print_dfs(dfs_orography)
    city_coords = [
        ("Florence", 43.7697, 11.2558),  # Name of the city, latitude, longitude
        # ("Berlin", 52.5200, 13.4050),
    ]

    coordinates = find_extract_coords(
        city_coords, dfs_orography, mode="point", delta_lat_km=20, delta_lon_km=20
    )
    print("coordinates: ", coordinates)
    request_models = [
        (
            "irnet_model",
            {
                "domain": "europe",  # fixed
                "experiment": ["historical", "rcp_4_5"],
                "horizontal_resolution": "0_11_degree_x_0_11_degree",  # fixed
                "temporal_resolution": "daily_mean",  # fixed
                "variable": "mean_precipitation_flux",
                "gcm_model": "ichec_ec_earth",
                "rcm_model": "knmi_racmo22e",
                "ensemble_member": "r1i1p1",
                "start_year": [
                    "1950",
                    "1951",
                    "1956",
                    "1961",
                    "1966",
                    "1971",
                    "1976",
                    "1981",
                    "1986",
                    "1991",
                    "1996",
                    "2001",
                    "2006",
                    "2011",
                    "2016",
                    "2021",
                    "2026",
                    "2031",
                    "2036",
                    "2041",
                    "2046",
                    "2051",
                    "2056",
                    "2061",
                    "2066",
                    "2071",
                    "2076",
                    "2081",
                    "2086",
                    "2091",
                    "2096",
                ],
                "end_year": [
                    "1950",
                    "1955",
                    "1960",
                    "1965",
                    "1970",
                    "1975",
                    "1980",
                    "1985",
                    "1990",
                    "1995",
                    "2000",
                    "2005",
                    "2010",
                    "2015",
                    "2020",
                    "2025",
                    "2030",
                    "2035",
                    "2040",
                    "2045",
                    "2050",
                    "2055",
                    "2060",
                    "2065",
                    "2070",
                    "2075",
                    "2080",
                    "2085",
                    "2090",
                    "2095",
                    "2100",
                ],
                "area": coordinates.get(
                    "area_irnet_orography_florence"
                ),  # [latitude_max longitude_min latitude_min longitude_max]
            },
        )
    ]

    dfs_models = {}

    download_dfs(request_models, dfs_models)
    # Print the dictionary
    print_dfs(dfs_models)


    # historical_request_models = [
    #     (
    #         "irnet_model",
    #         {
    #             "domain": "europe",  # fixed
    #             "experiment": "historical",
    #             "horizontal_resolution": "0_11_degree_x_0_11_degree",  # fixed
    #             "temporal_resolution": "daily_mean",  # fixed
    #             "variable": ["maximum_2m_temperature_in_the_last_24_hours"],
    #             "gcm_model": "ichec_ec_earth",
    #             "rcm_model": "knmi_racmo22e",
    #             "ensemble_member": "r1i1p1",
    #             "start_year": [
    #                 "1950",
    #                 "1951",
    #                 "1956",
    #                 "1961",
    #                 "1966",
    #                 "1971",
    #                 "1976",
    #                 "1981",
    #                 "1986",
    #                 "1991",
    #                 "1996",
    #                 "2001"
    #             ],
    #             "end_year": [
    #                 "1950",
    #                 "1955",
    #                 "1960",
    #                 "1965",
    #                 "1970",
    #                 "1975",
    #                 "1980",
    #                 "1985",
    #                 "1990",
    #                 "1995",
    #                 "2000",
    #                 "2005"
    #             ],
    #             "area": coordinates.get(
    #                 "area_irnet_orography_florence"
    #             ),  # [latitude_max longitude_min latitude_min longitude_max]
    #         },
    #     )
    # ]

    # historical_models = {}

    # # Download of the dictionary
    # # download_dfs(request_models, dfs_models)
    # download_dfs(historical_request_models, historical_models)
    # # Print the dictionary
    # print_dfs(historical_models)

    # future_request_models = [
    #     (
    #         "irnet_model",
    #         {
    #             "domain": "europe",  # fixed
    #             "experiment": "rcp_4_5",
    #             "horizontal_resolution": "0_11_degree_x_0_11_degree",  # fixed
    #             "temporal_resolution": "daily_mean",  # fixed
    #             "variable": ["maximum_2m_temperature_in_the_last_24_hours"],
    #             "gcm_model": "ichec_ec_earth",
    #             "rcm_model": "knmi_racmo22e",
    #             "ensemble_member": "r1i1p1",
    #             "start_year": [
    #                 "2006",
    #                 "2011",
    #                 "2016",
    #                 "2021",
    #                 "2026",
    #                 "2031",
    #                 "2036",
    #                 "2041",
    #                 "2046",
    #                 "2051",
    #                 "2056",
    #                 "2061",
    #                 "2066",
    #                 "2071",
    #                 "2076",
    #                 "2081",
    #                 "2086",
    #                 "2091",
    #                 "2096",
    #             ],
    #             "end_year": [
    #                 "2010",
    #                 "2015",
    #                 "2020",
    #                 "2025",
    #                 "2030",
    #                 "2035",
    #                 "2040",
    #                 "2045",
    #                 "2050",
    #                 "2055",
    #                 "2060",
    #                 "2065",
    #                 "2070",
    #                 "2075",
    #                 "2080",
    #                 "2085",
    #                 "2090",
    #                 "2095",
    #                 "2100",
    #             ],
    #             "area": coordinates.get(
    #                 "area_irnet_orography_florence"
    #             ),  # [latitude_max longitude_min latitude_min longitude_max]
    #         },
    #     )
    # ]

    # future_models = {}

    # # Download of the dictionary
    # # download_dfs(request_models, dfs_models)
    # download_dfs(future_request_models, future_models)
    # # Print the dictionary
    # print_dfs(future_models)

    # dfs_models = {}

    # h_values = historical_models['irnet_model'].values
    # f_values = future_models['irnet_model'].values
    # columns = historical_models['irnet_model'].columns

    # combined = np.concatenate((h_values, f_values))
    # dataframe = pd.DataFrame(combined, columns=columns)
    # dfs_models['irnet_model'] = dataframe

    # Extract yearly maxima for each variable, divide into time windows and standardizes the units of measurement.
    dfs_processed = processed_dataframe(dfs_models, window_length=40, shift=10)
    # Evaluate statistical values (q_k, parameters (location, scale), mean values, standaard deviations, COV, FC (only delta changes)
    R, parmhat, parmci, Q, std_dev, COV, FC = ev1_param(
        dfs_processed,
        50,
        method="LSM",
        value="Maxima",
        zeroyear="No",
        CM="no",
    )
    # Plot Gumbel Probability Paper
    ev1_test(dfs_processed, T_r=50)
    # Plot q_k, mean values, standard deviations, COV, FC
    # Dictionary
    results = {"R": R, "FC": FC, "COV": COV, "Q": Q, "std_dev": std_dev}
    print(results)
    plot_fc(results)
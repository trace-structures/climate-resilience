import json
import os
import earthkit.data
# import xarray as xr
import pandas as pd
import cftime
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genextreme as gev
from scipy.stats import genextreme as gev
# import cdsapi
from IPython.display import display
import reverse_geocoder as rg
import folium
import math
# import time

base_dir = os.path.dirname(os.path.abspath(__file__))

def _download_dfs(requests, dictionay_name):
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

def _print_dfs(dictionay_name):
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

def _ev1_invcdf(y, location, scale):
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
    return results

def get_window_names(windows):
    names = []
    for _, window in enumerate(windows):
        start = window[0]
        end = window[1]
        names.append("{} - {}".format(start, end))
    return names

def _get_gumbel_plot_new(dfs_processed, T_r, window):
    method_colors = {"MOM": 'blue', "MLM": 'purple', "LSM": 'green'}

    dataset_name = list(dfs_processed.keys())[0]
    df = dfs_processed[dataset_name][window]
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
        zp = _ev1_invcdf(1 - 1 / Tr, parmhat[0], parmhat[1])
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

def model_output(results, value, dfs_processed=None, T_r=50, window=None):
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
            return _get_gumbel_plot_new(dfs_processed, T_r, window)


def _find_extract_coords(
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

def get_map(latitude, longitude, city, country):
    f = folium.Figure(width=500, height=500)
    m = folium.Map(location=[latitude, longitude], zoom_start=10).add_to(f)
    folium.Marker(
        location=[latitude, longitude],
        popup=folium.Popup('<b>{}, {}</b><br>({} {})'.format(city, country, _format_coordinates(latitude, "latitude"), _format_coordinates(longitude, "longitude")), max_width=400, min_width=80), # pop-up label for the marker
        icon=folium.Icon()
    ).add_to(m)
    return f

def _request_orography(api_key, model_name, experiment_option, variable_option, latitude, longitude, city):
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
    _download_dfs(requests_orography, dfs_orography)

    # Print the dictionary
    _print_dfs(dfs_orography)
    city_coords = [
        (city, latitude, longitude)  # Name of the city, latitude, longitude
    ]

    coordinates = _find_extract_coords(
        city_coords, dfs_orography, mode="point", delta_lat_km=20, delta_lon_km=20
    )
    print("coordinates: ", coordinates)
    os.remove(cdsapi_rc_path)
    return coordinates
    
def _request_model(api_key, model_name, experiment_option, variable_option, coordinates):
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
    _download_dfs(request_models, dfs_models)
    print(dfs_models)
    os.remove(cdsapi_rc_path)

    return dfs_models

def request_data(api_key, model_name, experiment_option, variable_option, latitude, longitude, city):
    coordinates = _request_orography(api_key, model_name, experiment_option, variable_option, latitude, longitude, city)
    model_data = _request_model(api_key, model_name, experiment_option, variable_option, coordinates)
    return model_data

def _format_coordinates(coordinate, type="latitude"):
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
        country_code = location[0]['cc']
        city = location[0]['name']
        return city, country_code
    else:
        raise ValueError("Climate resilience analysis is not supported in the selected region. Please select European coordinates.")
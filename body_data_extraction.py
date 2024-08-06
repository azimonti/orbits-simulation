import json
import re
import requests
from urllib.parse import urlencode


def get_mass(response_text):
    # regular expression to find the mass value and exponent (case-insensitive)
    mass_match = re.search(
        r"Mass\s*x\s*10\^(\d+)\s*\((g|kg)\)\s*=\s*([\d\.]+)",
        response_text, re.IGNORECASE
    )
    if mass_match:
        exponent = int(mass_match.group(1))  # extract the exponent
        unit = mass_match.group(2).lower()  # extract the unit (g or kg)
        coefficient = float(mass_match.group(3))  # extract the coefficient

        # Compute the mass based on the unit
        if unit == 'g':
            # Convert grams to kilograms
            mass_value_kg = coefficient * (10 ** exponent) / 1e3
        elif unit == 'kg':
            mass_value_kg = coefficient * (10 ** exponent)

        return mass_value_kg
    else:
        raise ValueError("Mass not found in the response.")


def get_radius(response_text):
    # regular expression to find the vol. mean radius value in km
    radius_match = re.search(
        r"vol\.\s*mean\s*radius\s*\(km\)\s*=\s*([\d\.]+)",
        response_text, re.IGNORECASE)
    if radius_match:
        mean_radius_km = float(radius_match.group(1))
        # Convert to meters
        mean_radius_m = mean_radius_km * 1000
        return mean_radius_m
    else:
        print(response_text)
        raise ValueError("Mean radius not found in the response.")


def to_julian_date(date_string):
    from datetime import datetime
    # parse date string
    dt = datetime.strptime(date_string, '%Y-%m-%d %H:%M:%S')
    # calculate Julian date
    julian_date = dt.toordinal() + 1721424.5 + (
        dt.hour + dt.minute / 60 + dt.second / 3600) / 24
    return julian_date


def get_position_velocity(response_text, start_date):
    # convert start date to Julian date as float
    start_jd = float(to_julian_date(start_date))

    # Regular expression to find the first set of position and velocity values
    pv_match = re.search(
        r"^\s*(\d+\.\d+)\s*=\s*[A-Za-z0-9\.\-\s:]+TDB\s+"
        r"X\s*=\s*([-\d\.E+]+)\s*Y\s*=\s*([-\d\.E+]+"
        r")\s*Z\s*=\s*([-\d\.E+]+)\s*"
        r"VX\s*=\s*([-\d\.E+]+)\s*VY\s*=\s*([-\d\.E+]+"
        r")\s*VZ\s*=\s*([-\d\.E+]+)",
        response_text, re.IGNORECASE | re.MULTILINE
    )

    if pv_match:
        jd = float(pv_match.group(1))
        # allow a small tolerance for JD matching
        if abs(jd - start_jd) > 1e-5:
            raise ValueError(f"Julian Date {jd} does not match the"
                             f"start date JD {start_jd}")

        position = [
            float(pv_match.group(2)) * 1000,
            float(pv_match.group(3)) * 1000,
            float(pv_match.group(4)) * 1000
        ]
        velocity = [
            float(pv_match.group(5)) * 1000,
            float(pv_match.group(6)) * 1000,
            float(pv_match.group(7)) * 1000
        ]
        return position, velocity
    else:
        raise ValueError("Position and velocity not found in the response.")


# Load the JSON file
with open('data_input/bodies_extraction_config.json', 'r') as file:
    data = json.load(file)

# Extract relevant information
bodies = data["bodies"]
start_date = data["start_date"]
end_date = data["end_date"]
step = data["step"]
center_code = data[data["center"]]["code"]
expected_api_version = data["api_version"]

# Initialize output data
output_data = {
    "bodies": bodies,
}


# Base URL for the JPL Horizons API
base_url = "https://ssd.jpl.nasa.gov/api/horizons.api"

# Generate URLs for each body
for body in bodies:
    print(f'processing {body}')
    body_info = data[body]
    query_params = {
        "format": "json",
        "COMMAND": f"'{body_info['code']}'",
        "CSV_FORMAT": "NO",
        "OBJ_DATA": "'YES'",
        "MAKE_EPHEM": "'YES'",
        "EPHEM_TYPE": "'VECTOR'",
        "CENTER": f"'{center_code}'",
        "OUT_UNITS": "KM-S",
        "START_TIME": f"'{start_date}'",
        "STOP_TIME": f"'{end_date}'",
        "STEP_SIZE": f"'{step}'",
        "VEC_TABLE": "'3'"
    }
    url = f"{base_url}?{urlencode(query_params)}"
    response = requests.get(url)
    response_data = response.json()

    # Extract and check api_version
    api_version = response_data.get("signature", {}).get("version")
    if not api_version:
        raise ValueError("API Version not found.")
    elif api_version != expected_api_version:
        raise ValueError("Wrong API version")
    result = response_data['result']
    position, velocity = get_position_velocity(result, start_date)
    # Accumulate data for the body
    output_data[body] = {
        "label": data[body]["label"],
        "mass": get_mass(result),
        "radius": get_radius(result),
        "position": position,
        "velocity": velocity
    }

formatted_date = start_date.replace(" ", "_").replace(":", "_")
with open(f"data_input/bodies_extraction_{formatted_date}.json", 'w') \
        as outfile:
    json.dump(output_data, outfile, indent=2)

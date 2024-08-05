# Read Parking Signs

This project is a combination of multiple ML models designed to detect, read, and analyze parking signs to determine parking eligibility.

## Installation

To install dependancies for this project we can run the command
`pip install -r requirements.txt`

We also require an api key for accessing the RoboFlow computer vision models. We create a file named `api_key.py` in the root directory and add the single line with the api key obtained from RoboFlow as a string.
`API_KEY = "XXXXXXXXXXXXXXXX"`

## Usage
Currently this tool is accessible via a CLI.
Simply provide the path to the file you would like to analyze and your API key to access the roboflow model and the interface will read out what the sign says. A sample command is as follows:
` python3 -m read_parking_signs.py "./no_parking.jpg' "<your API key here>"`

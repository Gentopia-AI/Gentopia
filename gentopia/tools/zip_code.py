from typing import AnyStr

import requests
from geopy import Nominatim

from .basetool import *


class ZipCodeRetriever(BaseTool):
    name = "zip_code_retriever"
    description = "A zip code retriever. Useful when you need to get users' current zip code. Input can be " \
                  "left blank."
    args_schema = create_model("ZipCodeRetrieverArgs")

    def get_ip_address(self):
        response = requests.get("https://ipinfo.io/json")
        data = response.json()
        return data["ip"]

    def get_location_data(sefl, ip_address):
        url = f"https://ipinfo.io/{ip_address}/json"
        response = requests.get(url)
        data = response.json()
        return data

    def get_zipcode_from_lat_long(self, lat, long):
        geolocator = Nominatim(user_agent="zipcode_locator")
        location = geolocator.reverse((lat, long))
        return location.raw["address"]["postcode"]

    def get_current_zipcode(self):
        ip_address = self.get_ip_address()
        location_data = self.get_location_data(ip_address)
        lat, long = location_data["loc"].split(",")
        zipcode = self.get_zipcode_from_lat_long(float(lat), float(long))
        return zipcode

    def _run(self) -> AnyStr:
        evidence = self.get_current_zipcode()
        return evidence

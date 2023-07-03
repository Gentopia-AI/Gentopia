import requests
import json
import os
from typing import AnyStr, Any
from .basetool import *


class Weather(BaseTool):
    api_key: str = os.getenv("WEATHER_API_KEY")
    URL_CURRENT_WEATHER = "http://api.weatherapi.com/v1/current.json"
    URL_FORECAST_WEATHER = "http://api.weatherapi.com/v1/forecast.json"


class GetTodayWeather(Weather):
    """Tool that looks up today's weather information"""

    name = "GetTodayWeather"
    description = (
        "Look up the current weather information for a given location"
    )
    args_schema: Optional[Type[BaseModel]] = create_model("GetTodayWeatherArgs", location=(str, ...))

    def _run(self, location: AnyStr) -> AnyStr:
        param = {
            "key": self.api_key,
            "q": location
        }
        res_completion = requests.get(self.URL_CURRENT_WEATHER, params=param)
        data = json.loads(res_completion.text.strip())
        output = {}
        output["overall"]= f"{data['current']['condition']['text']},\n"
        output["name"]= f"{data['location']['name']},\n"
        output["region"]= f"{data['location']['region']},\n"
        output["country"]= f"{data['location']['country']},\n"
        output["localtime"]= f"{data['location']['localtime']},\n"
        output["temperature"]= f"{data['current']['temp_c']}(C), {data['current']['temp_f']}(F),\n"
        output["percipitation"]= f"{data['current']['precip_mm']}(mm), {data['current']['precip_in']}(inch),\n"
        output["pressure"]= f"{data['current']['pressure_mb']}(milibar),\n"
        output["humidity"]= f"{data['current']['humidity']},\n"
        output["cloud"]= f"{data['current']['cloud']},\n"
        output["body temperature"]= f"{data['current']['feelslike_c']}(C), {data['current']['feelslike_f']}(F),\n"
        output["wind speed"]= f"{data['current']['gust_kph']}(kph), {data['current']['gust_mph']}(mph),\n"
        output["visibility"]= f"{data['current']['vis_km']}(km), {data['current']['vis_miles']}(miles),\n"
        output["UV index"]= f"{data['current']['uv']},\n"
        
        text_output = f"Today's weather report for {data['location']['name']} is:\n"+"".join([f"{key}: {output[key]}" for key in output.keys()])
        return text_output

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class GetFutureWeather(Weather):
    """Tool that looks up the weather information in the future"""

    name = "GetFutureWeather"
    description = (
        "Look up the weather information in the upcoming days for a given location"
    )
    args_schema: Optional[Type[BaseModel]] = create_model("GetFutureWeatherArgs", location=(str, ...), days=(int, ...))

    def _run(self, location: AnyStr, days: int) -> AnyStr:
        param = {
            "key": self.api_key,
            "q": location,
            "days": int(days),
        }
        res_completion = requests.get(self.URL_FORECAST_WEATHER, params=param)
        res_completion = json.loads(res_completion.text.strip())
        MAX_DAYS = 14
        import pdb
        pdb.set_trace()
        res_completion = res_completion["forecast"]["forecastday"][int(days)-1 if int(days) < MAX_DAYS else MAX_DAYS-1]
        output_dict = {}
        for k, v in res_completion["day"].items():
            output_dict[k] = v
        for k, v in res_completion["astro"].items():
            output_dict[k] = v
        
        output = {}
        output["over all weather"] = f"{output_dict['condition']['text']},\n"
        output["max temperature"] = f"{output_dict['maxtemp_c']}(C), {output_dict['maxtemp_f']}(F),\n"
        output["min temperature"] = f"{output_dict['mintemp_c']}(C), {output_dict['mintemp_f']}(F),\n"
        output["average temperature"] = f"{output_dict['avgtemp_c']}(C), {output_dict['avgtemp_f']}(F),\n"
        output["max wind speed"] = f"{output_dict['maxwind_kph']}(kph), {output_dict['maxwind_mph']}(mph),\n"
        output["total precipitation"] = f"{output_dict['totalprecip_mm']}(mm), {output_dict['totalprecip_in']}(inch),\n"
        output["will rain today"] = f"{output_dict['daily_will_it_rain']},\n"
        output["chance of rain"] = f"{output_dict['daily_chance_of_rain']},\n"
        output["total snow"] = f"{output_dict['totalsnow_cm']}(cm),\n"
        output["will snow today"] = f"{output_dict['daily_will_it_snow']},\n"
        output["chance of snow"] = f"{output_dict['daily_chance_of_snow']},\n"
        output["average visibility"] = f"{output_dict['avgvis_km']}(km), {output_dict['avgvis_miles']}(miles),\n"
        output["average humidity"] = f"{output_dict['avghumidity']},\n"
        output["UV index"] = f"{output_dict['uv']},\n"
        output["sunrise time"] = f"{output_dict['sunrise']},\n"
        output["sunset time"] = f"{output_dict['sunset']},\n"
        output["moonrise time"] = f"{output_dict['moonrise']},\n"
        output["moonset time"] = f"{output_dict['moonset']},\n"
        
        text_output = f"The weather forecast for {param['q']} at {param['days']} days later is: \n"+"".join([f"{key}: {output[key]}" for key in output.keys()])
        return text_output

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

if __name__ == "__main__":
    # ans = GetTodayWeather()._run("San francisco")
    ans = GetFutureWeather()._run("San francisco", 10)
    print(ans)

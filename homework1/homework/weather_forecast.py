from typing import Tuple

import torch


class WeatherForecast:
    def __init__(self, data_raw: list[list[float]]):
        """
        You are given a list of 10 weather measurements per day.
        Save the data as a PyTorch (num_days, 10) tensor,
        where the first dimension represents the day,
        and the second dimension represents the measurements.
        """
        self.data = torch.as_tensor(data_raw).view(-1, 10)

    def find_min_and_max_per_day(self) -> Tuple[torch.Tensor]:
        """
        Find the max and min temperatures per day

        Returns:
            min_per_day: tensor of size (num_days,)
            max_per_day: tensor of size (num_days,)
        """
        min_per_day = torch.min(self.data, dim=1).values
        max_per_day = torch.max(self.data, dim=1).values
        return min_per_day, max_per_day

    def find_the_largest_drop(self) -> torch.Tensor:
        """
        Find the largest change in day over day average temperature.
        This should be a negative number.

        Returns:
            tensor of a single value, the difference in temperature
        """
        daily_avg = torch.mean(self.data, dim=1)
        day_to_day_diff = torch.diff(daily_avg)
        largest_drop = torch.min(day_to_day_diff)
        return largest_drop

    def find_the_most_extreme_day(self) -> torch.Tensor:
        """
        For each day, find the measurement that differs the most from the day's average temperature

        Returns:
            tensor with size (num_days,)
        """
        daily_avg = torch.mean(self.data, dim=1, keepdim=True)
        deviation = torch.abs(self.data - daily_avg)
        most_extreme_day = self.data.gather(1, torch.max(deviation, dim=1).indices.unsqueeze(1)).squeeze(1)
        return most_extreme_day

    def max_last_k_days(self, k: int) -> torch.Tensor:
        """
        Find the maximum temperature over the last k days

        Returns:
            tensor of size (k,)
        """
        return torch.max(self.data[-k:], dim=1).values

    def predict_temperature(self, k: int) -> torch.Tensor:
        """
        From the dataset, predict the temperature of the next day.
        The prediction will be the average of the temperatures over the past k days.

        Args:
            k: int, number of days to consider

        Returns:
            tensor of a single value, the predicted temperature
        """
        avg_last_k_days = torch.mean(self.data[-k:], dim=0)
        return torch.mean(avg_last_k_days)

    def what_day_is_this_from(self, t: torch.FloatTensor) -> torch.LongTensor:
        """
        You go on a stroll next to the weather station, where this data was collected.
        You find a phone with severe water damage.
        The only thing that you can see in the screen are the
        temperature reading of one full day, right before it broke.

        You want to figure out what day it broke.

        The dataset we have starts from Monday.
        Given a list of 10 temperature measurements, find the day in a week
        that the temperature is most likely measured on.

        We measure the difference using 'sum of absolute difference
        per measurement':
            d = |x1-t1| + |x2-t2| + ... + |x10-t10|

        Args:
            t: tensor of size (10,), temperature measurements

        Returns:
            tensor of a single value, the index of the closest data element
        """
        differences = torch.sum(torch.abs(self.data - t), dim=1)
        return torch.argmin(differences)



# Sample data: 10 weather measurements for 5 days (each day has 10 temperature readings)
data_raw = [
    [30.5, 31.0, 32.0, 33.5, 34.0, 35.0, 36.5, 37.0, 38.0, 39.5],  # Day 1
    [29.0, 30.0, 31.5, 32.0, 33.0, 34.5, 35.5, 36.0, 37.5, 38.0],  # Day 2
    [28.5, 29.5, 30.0, 31.0, 32.5, 33.0, 34.0, 35.0, 36.0, 37.0],  # Day 3
    [27.5, 28.0, 29.0, 30.0, 31.0, 32.0, 33.5, 34.0, 35.0, 36.0],  # Day 4
    [26.5, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.5, 35.0],  # Day 5
]

# Create an instance of WeatherForecast
weather_forecast = WeatherForecast(data_raw)

# Find min and max temperatures per day
min_per_day, max_per_day = weather_forecast.find_min_and_max_per_day()
print("Min temperatures per day:", min_per_day)
print("Max temperatures per day:", max_per_day)

# Find the largest drop in average temperature from one day to the next
largest_drop = weather_forecast.find_the_largest_drop()
print("Largest day-to-day temperature drop:", largest_drop)

# Find the most extreme temperature deviation from the daily average for each day
most_extreme_day = weather_forecast.find_the_most_extreme_day()
print("Most extreme temperature deviation per day:", most_extreme_day)

# Find the maximum temperature over the last 3 days
max_last_3_days = weather_forecast.max_last_k_days(3)
print("Max temperatures over the last 3 days:", max_last_3_days)

# Predict the temperature for the next day based on the last 3 days
predicted_temperature = weather_forecast.predict_temperature(3)
print("Predicted temperature for the next day:", predicted_temperature)

# Find the closest matching day in the dataset for a given temperature measurement (for 1 day)
test_measurements = torch.tensor([27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0], dtype=torch.float32)
closest_day = weather_forecast.what_day_is_this_from(test_measurements)
print("Closest day for the given temperature readings:", closest_day)

from bs4 import BeautifulSoup
from requests import Request
import csv
from datetime import datetime
from urllib.request import urlopen
url = "https://www.accuweather.com/en/in/muzaffarnagar/191054/weather-forecast/191054"
request=Request(url)
webpage = urlopen(request)
soup = BeautifulSoup(webpage.read())

temp_block = soup.find('span', attrs={'class': 'large-temp'})
stats_block = soup.find('ul', attrs={'class': 'stats'})
sunrise_block = soup.find('ul', attrs={'class': 'time-period'})

temp = temp_block.text.strip()
stats = stats_block.text.strip()
sunrise= sunrise_block.text.strip()

print ("Temperature:"+temp)
print ("\nStats:\n\n"+stats)
print ("\nSunrise:\n\n"+sunrise)

# # request = Request("https://www.accuweather.com/en/in/muzaffarnagar/191054/weather-forecast/191054")
# webpage = urllib.request.urlopen('https://www.accuweather.com/en/in/muzaffarnagar/191054/weather-forecast/191054')
# soup = BeautifulSoup(webpage)

# temp_block = soup.find('span', attrs={'class': 'large-temp'})
# stats_block = soup.find('ul', attrs={'class': 'stats'})
# sunrise_block = soup.find('ul', attrs={'class': 'time-period'})

# temp = temp_block.text.strip()
# stats = stats_block.text.strip()
# sunrise= sunrise_block.text.strip()

# print ("Temperature:"+temp)
# print ("\nStats:\n\n"+stats)
# print ("\nSunrise:\n\n"+sunrise)
"""
Download and save Data from Entso-e in "day-ahead" format as a csv file. Day-ahead means a data value for each 15 min of
the day (96 values for every 24 hours)
INPUT: URL for API request at Entso-e. All references for the data, such as country, load, generation, etc., has to be
specified in the URL. Please refer to the notebook
OUTPUT: csv files for the specified time horizon
"""

import requests
import pandas as pd
import datetime
import os
from bs4 import BeautifulSoup as bs
from Name_convention_dictionaries import DocumentTypeDict, ProcessTypeDict, AreaDict, PsrTypeDict


def Load_data_entsoe(url):

    ##### Load data and check status
    response = requests.get(url)

    if response.status_code != requests.codes.ok:
        soup = bs(response.text,"lxml-xml")
        print(soup.Reason)
        print(response.raise_for_status())

    #print(response.url)

    data = response.text
    soup = bs(data, "lxml-xml")


    #### Look at the data and print out what was downloaded
    if soup.find("inBiddingZone_Domain.mRID"):
        Zone = soup.find("inBiddingZone_Domain.mRID").text
    elif soup.find("outBiddingZone_Domain.mRID"):
        Zone = soup.find("outBiddingZone_Domain.mRID").text
    processType = soup.find("process.processType").text
    DocType = soup.type.text
    if soup.psrType:
        psrType = soup.psrType.text

    start_datetime_initial = convert_to_datetime(soup.find("time_Period.timeInterval").start.text)
    end_datetime_initial = convert_to_datetime(soup.find("time_Period.timeInterval").end.text)

    if Zone in AreaDict:
        print(f"You downloaded data for Zone \"{Zone}\" which is \"{AreaDict[Zone]}\"")

        #to understand it for now:
        if "BZA" in AreaDict[Zone]:
            print("BZA = Bidding Zone Aggregation")
        elif "BZ" in AreaDict[Zone]:
            print("BZ = Bidding Zone")
        if "CA" in AreaDict[Zone]:
            print("CA = Control Area")
        if "MBA" in AreaDict[Zone]:
            print("MBA = Market Balance Area")

    else:
        print("Warning! Zone/Area not found! The following code might not work.")
    print("--------------------------------------------------------------------------------------------------")
    print(f"The downloaded time period is from {start_datetime_initial} until {end_datetime_initial} "  )
    print("--------------------------------------------------------------------------------------------------")

    if DocType in DocumentTypeDict:
        print(f"The loaded document type is \"{DocType}\" which corresponds to: \"{DocumentTypeDict[DocType]}\"")
    else:
        print("Warning! The Document type is not recognized! The following code might not work.")
    if soup.find("process.processType").text in ProcessTypeDict:
        print(f"The loaded process type is \"{processType}\" which corresponds to: \"{ProcessTypeDict[processType]}\" ,other examples: day Ahead, week ahead")
    else:
        print("Warning! The Process type is not recognized! The following code might not work.")
    if soup.psrType:
        print(f"The loaded prsrtype is \"{psrType}\" which corresponds to: \"{PsrTypeDict[psrType]}\" ")


    ###### Convert the data from the xml time series

    if soup.find_all("TimeSeries")[0].curveType.text != "A01":
        raise ValueError(f"Curve type not implemented yet") #TODO look at the different curve types and change code when necessary

    measure_unit = soup.find("quantity_Measure_Unit.name").text
    TS = soup.find_all("TimeSeries")[0]

    all_time_Series = soup.find_all("TimeSeries")

    if len(all_time_Series) < 1:
        raise ValueError("Unexpected amount of time series")


    date_col = []
    value_col = []

    for TS in all_time_Series:
        time_resolution_str = TS.resolution.text
        if time_resolution_str[-1]!="M":
            raise ValueError("Time Resolution is not in minutes")  #TODO: implement other time intervals
        time_resolution = int(time_resolution_str[2:-1])
        start_datetime =  convert_to_datetime(TS.timeInterval.start.text)
        end_datetime = convert_to_datetime(TS.timeInterval.end.text)

        for i in TS.find_all("Point"):
            position = int(i.position.text)
            qty = int(i.quantity.text)
            current_time = start_datetime + datetime.timedelta(minutes=time_resolution*(position-1))
            date_col.append(current_time)
            value_col.append(qty)

        if end_datetime == end_datetime_initial:
            break

    ##### create a folder called data to store the data locally
    folder_name = 'data_day_ahead'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    ##### Create a pandas dataframe and write a csv
    if soup.psrType:
        d = {'Date': date_col, ProcessTypeDict[processType]+"_"+PsrTypeDict[psrType]+"_in_"+measure_unit : value_col}
    else:
        d = {'Date': date_col, ProcessTypeDict[processType]+"_"+DocumentTypeDict[DocType]+"_in_"+measure_unit : value_col}
    df = pd.DataFrame(d)

    if soup.psrType:
        path = AreaDict[Zone]+"__&docType="+DocType+"__&rpocessType="+processType+"__&psrType="+psrType+"from"+ convert_from_date(start_datetime_initial) +"to"+ convert_from_date(end_datetime) +".csv"
    else:
        path = AreaDict[Zone]+"__&docType="+DocType+"__&rpocessType="+processType+"from"+ convert_from_date(start_datetime_initial) +"to"+ convert_from_date(end_datetime) +".csv"
    df.to_csv(("./"+folder_name+"/"+path),index=False)
    print("######################################################")
    print("Sucessfully saved to:")
    print(path)
    print("######################################################\n\n\n")

    if end_datetime_initial != end_datetime:
        print(f"\n\n\n\n Data was just loaded until {end_datetime} please have a look \n\n\n\n\n")#TODO figure out why sometimes it does not load all


### supporting functions to handle datetimes
def convert_to_datetime(date_str):
    return datetime.datetime.strptime(date_str, '%Y-%m-%dT%H:%MZ')

def convert_from_date(date):
    return datetime.datetime.strftime(date, "%Y-%m-%d_%H:%M")

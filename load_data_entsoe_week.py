import requests
import pandas as pd
import datetime
from bs4 import BeautifulSoup as bs

# Import dictionaries and security_token

from Name_convention_dictionaries import DocumentTypeDict, ProcessTypeDict,\
AreaDict, PsrTypeDict

# Reverse dictionaries for easier understanding

zone_dict = {v:k for k,v in AreaDict.items()}
document_type_dict = {v:k for k,v in DocumentTypeDict.items()}
process_type_dict = {v:k for k,v in ProcessTypeDict.items()}
generation_type_dict = {v:k for k,v in PsrTypeDict.items()}

# handle datetimes 
def convert_to_datetime(date_str):
    return datetime.datetime.strptime(date_str, '%Y-%m-%dT%H:%MZ')

def convert_from_date(date):
    return datetime.datetime.strftime(date, "%Y-%m-%d_%H:%M")


# create load week_ahead data function

def load_data_week_ahead(url):
    
    # load data and check status:
    
    response = requests.get(url)
    
    if response.status_code != requests.codes.ok:
        soup = bs(response.text, 'lxml-xml')
        print(soup.Reason)
        print(response.raise_for_status())
        
    # store xml-data in memory and create a BeautifulSoup object
    
    data = response.text 
    soup = bs(data, 'lxml-xml')
    
    # look at the data and print what was downloaded - quality control!
    
    if soup.find('inBiddingZone_Domain.mRID'):
        Zone = soup.find('inBiddingZone_Domain.mRID').text
    elif soup.find('outBiddingZone_Domain.mRID'):
        Zone = soup.find('outBiddingZone_Domain.mRID').text
    elif soup.find('in_Domain'):
        Zone = soup.find('in_Domain').text
    processType = soup.find('process.processType').text
    docType = soup.type.text
    if soup.psrType:
        psrType = soup.psrType.text
        
    start_datetime_initial = convert_to_datetime(soup.find('time_Period.timeInterval').start.text)
    end_datetime_initial = convert_to_datetime(soup.find('time_Period.timeInterval').end.text)
    
    if Zone in AreaDict:
        print(f"You downloaded data for Zone \"{Zone}\" which is \"{AreaDict[Zone]}\"")
    else:
        print('Warning! Zone/Country not found! The following code might not work')
    
    print("--------------------------------------------------------------------------------------------------")
    print(f"The downloaded time period is from {start_datetime_initial} until {end_datetime_initial} "  )
    print("--------------------------------------------------------------------------------------------------")
            
    if docType in DocumentTypeDict:
        print(f"The loaded document type is \"{docType}\" which corresponds to: \"{DocumentTypeDict[docType]}\"")
    else:
        print("Warning! The Document type is not recognized! The following code might not work.")
    if soup.find("process.processType").text in ProcessTypeDict:
        print(f"The loaded process type is \"{processType}\" which corresponds to: \"{ProcessTypeDict[processType]}\" , other examples: day Ahead, week ahead")
    else:
        print("Warning! The Process type is not recognized! The following code might not work.")
    if soup.psrType:
        print(f"The loaded prsrtype is \"{psrType}\" which corresponds to: \"{PsrTypeDict[psrType]}\" ")
    
    # Convert the xml file into data
    
    if soup.find_all('TimeSeries')[0].curveType.text != 'A01':
        raise ValueError('Curve type not implemented yet')
        
    measure_unit = soup.find('quantity_Measure_Unit.name').text
    TS = soup.find_all('TimeSeries')[0]
    
    all_time_Series = soup.find_all('TimeSeries')
    
    if len(all_time_Series) < 1:
        raise ValueError('Unexpected amount of time series')
    
    # create lists to store the data we will extract from the xml file using a loop    
    min_date_col = []
    max_date_col = []
    min_value_col = []
    max_value_col = []
    
    # create a logic statement to avoid extracting information more than once
    first = True
    second = True
    
    # loop over the xml file to extract data
    for TS in all_time_Series:
        time_resolution_str = TS.resolution.text
        if time_resolution_str[-1] != 'D':
            raise ValueError('Time Resolution is not in days!')
        time_resolution = int(time_resolution_str[1:-1])
        start_datetime = convert_to_datetime(TS.timeInterval.start.text)
        end_datetime = convert_to_datetime(TS.timeInterval.end.text)
        min_or_max_value = TS.businessType.text
    
        if min_or_max_value == 'A60':
            for i in TS.find_all('Point'):
                position = int(i.position.text)
                qty = int(i.quantity.text)
                current_time = start_datetime + datetime.timedelta(days=time_resolution*(position-1))
                min_date_col.append(current_time)
                min_value_col.append(qty)
        elif min_or_max_value == 'A61':
            for i in TS.find_all('Point'):
                position = int(i.position.text)
                qty = int(i.quantity.text)
                current_time = start_datetime + datetime.timedelta(days=time_resolution*(position-1))
                max_date_col.append(current_time)
                max_value_col.append(qty)
    
    # use logical statements to avoid extracting information more than once            
        if first:
            first = False
        elif second:
            second = False
        else:        
            if start_datetime == start_datetime_initial:
                break
    
    
    # Create a pandas dataframe and write a csv
    d = {'min_date': min_date_col,
         'max_date': max_date_col,
         'min_forecast_in_'+measure_unit: min_value_col,
         'max_forecast_in_'+measure_unit: max_value_col}
    df = pd.DataFrame(d)
    
    path = 'Week_ahead_'+AreaDict[Zone]+"__"+convert_from_date(start_datetime_initial)+"to"+convert_from_date(end_datetime)+".csv"
    df.to_csv(("./data/"+path),index=False)
    print("######################################################")
    print("Sucessfully saved to:")
    print(path)
    print("######################################################")
    
    if end_datetime_initial != end_datetime:
        print(f"\n\n\n\n Data was just loaded until {end_datetime} please have a look \n\n\n\n\n")#TODO figure out why sometimes it does not load all
        

















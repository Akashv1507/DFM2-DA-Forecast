import datetime as dt
from typing import List, Tuple, Union
import pandas as pd
from src.dayAheadForecastCreator.blockwiseDemandFetch import DemandFetchForModelRepo
from src.dayAheadForecastCreator.mlrPredictions import MlrPredictions
from src.dayAheadForecastCreator.daForecastInsertion import DayAheadDemandForecastInsertion
from src.dayAheadForecastCreator.weatherDataFetcher import WeatherDataFetcher
from src.dayAheadForecastCreator.holidayDummyGenerator import HolidayDummyGenerator


def createDayAheadForecast(startDate:dt.datetime ,endDate: dt.datetime, configDict:dict)->bool:
    """ create DA forecast using DFM-3
    Args:
        startDate (dt.datetime): start date
        endDate (dt.datetime): end date
        configDict (dict):   apllication configuration dictionary
    Returns:
        bool: return true if insertion is success.
    """    

    
    conString:str = configDict['con_string_mis_warehouse']
    conStringWeather:str = configDict['con_string_mis_weather']
    modelPath:str = configDict['model_path']
    holidayExcelPath:str =configDict['holidayExcelPath']
    # listOfEntity =['WRLDCMP.SCADA1.A0046945','WRLDCMP.SCADA1.A0046948','WRLDCMP.SCADA1.A0046953','WRLDCMP.SCADA1.A0046957','WRLDCMP.SCADA1.A0046962','WRLDCMP.SCADA1.A0046978','WRLDCMP.SCADA1.A0046980','WRLDCMP.SCADA1.A0047000']
    listOfEntity =['WRLDCMP.SCADA1.A0047000', 'WRLDCMP.SCADA1.A0046980', 'WRLDCMP.SCADA1.A0046957', 'WRLDCMP.SCADA1.A0046978']

    #creating instance of class
    obj_demandFetchForModelRepo = DemandFetchForModelRepo(conString)
    obj_mlrPredictions = MlrPredictions(modelPath)
    obj_daDemandForecastInsertion = DayAheadDemandForecastInsertion(conString)
    obj_weatherDataFetcher = WeatherDataFetcher(conStringWeather)
    obj_holidayDummyGenerator = HolidayDummyGenerator(holidayExcelPath)
    
    
    insertSuccessCount=0
    currDate = startDate
    
    # Iterating through each day and each entities , storing DA forecast in storeForecastDf anf psuhing into db 
    while currDate <= endDate:

        #intializing empty dataframe to store forecast of all entities
        storeForecastDf = pd.DataFrame(columns = [ 'timestamp','entityTag','forecastedDemand']) 
        for entity in listOfEntity:
            lagDemandDf = obj_demandFetchForModelRepo.fetchBlockwiseDemandForModel(currDate, entity)
            
            #fetching weather for forecast date
            forecastDate = currDate + dt.timedelta(days=1)
            weatherDf = obj_weatherDataFetcher.fetchWeatherData(forecastDate, "WRLDCMP.SCADA1.A0047000")
        
            #generating holiday, weekend dummy variable
            holidayDummyDf = obj_holidayDummyGenerator.generateHolidayDummies(forecastDate)
            
            #fetching virtual station no. from config for a particular entity, then generating mlr with weather prediction
            virtualWeatherStsnNo = configDict[entity]
            predictedDaDf = obj_mlrPredictions.predictDaMlr(lagDemandDf, weatherDf, holidayDummyDf, entity, virtualWeatherStsnNo)
            storeForecastDf = pd.concat([storeForecastDf, predictedDaDf],ignore_index=True)
            
        isInsertionSuccess =  obj_daDemandForecastInsertion.insertDayAheadDemandForecast(storeForecastDf)
       
        if isInsertionSuccess:
            insertSuccessCount = insertSuccessCount + 1
        currDate += dt.timedelta(days=1)
    
    numOfDays = (endDate-startDate).days

    #checking whether data is inserted for each day or not
    if insertSuccessCount  == numOfDays +1:
        return True
    else:
        return False
    
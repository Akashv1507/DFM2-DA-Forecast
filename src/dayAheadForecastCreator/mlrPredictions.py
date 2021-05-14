import pandas as pd 
import datetime as dt
import joblib
import numpy as np
from src.utils.getSortedStationList import getSortedStationList


class MlrPredictions():
    """MLR prediction class
    """

    def __init__(self, modelPath: str) -> None:
        """load prediction model path
        Args:
            modelPath ([type]): path of model
        """
        self.modelPath = modelPath
        self.modelPathStr =""


    def dummyVariableGenerator(self, modelStartDate= "2018-01-01 00:00:00"):
        """
        Description: This function returns dummy variables for timeblock of the day, day of the week and 
        time of the year.
        
        Args:
        startDate: The start date of training data. (Here: "2018-01-01 00:00:00")
        
        endDate: The ending date of test (or current year) data. (Here: "2021-12-31 23:59:59")
        """
        #Making a time index for accomodating dummy variables
        timeIndex = pd.date_range(start = pd.Timestamp(modelStartDate), 
                                end = pd.Timestamp(dt.datetime.today().strftime("%Y-12-31 23:45:00")),
                                freq ='15min').rename("time").to_frame()
        #Making series to get dummy variables
        month = pd.Series(timeIndex.index.month.astype(str), index=timeIndex.index, name="month").apply(lambda x: "m{}".format(x)) #m1 -> January
        day = pd.Series(timeIndex.index.dayofweek.astype(str), index=timeIndex.index, name="day").apply(lambda x: "d{}".format(x)) #d0 -> Monday
        timeblock = pd.Series(timeIndex.index.strftime("%H:%M"), index=timeIndex.index, name="timeblock")
        daytimeblock = (day + "_" + timeblock).rename("daytimeblock")
        #Making dummy variables
        monthDummies = pd.get_dummies(month.sort_values()).sort_index()
        dayDummies = pd.get_dummies(day.sort_values()).sort_index()
        timeblockDummies = pd.get_dummies(timeblock.sort_values()).sort_index()
        daytimeblockDummies = pd.get_dummies(daytimeblock.sort_values()).sort_index()
        #Making trend variable
        trend = pd.Series(np.arange(1, len(timeIndex) + 1), timeIndex.index, name="trend")
        return trend, timeblockDummies, dayDummies, monthDummies, daytimeblockDummies


    def getInteractions(self,tempSeries , dummies, poly_degree=1):
        if tempSeries.name == None: tempSeries.name = "_"
        s1 = dummies.apply(lambda x: x * tempSeries).rename("{}".format(tempSeries.name + "_{}").format, axis=1)
        s2 = (dummies.apply(lambda x: x * tempSeries) **2).rename("{}".format(tempSeries.name + "2_{}").format, axis=1)
        s3 = (dummies.apply(lambda x: x * tempSeries) **3).rename("{}".format(tempSeries.name + "3_{}").format, axis=1)
        
        if poly_degree == 1: s2, s3 = None, None
        if poly_degree == 2: s3 = None

        df = pd.concat([s1, s2, s3], axis=1)
        return df

    def stationCombinationFinder(self, tempData, entityTag):
           
        # get sortedStation list
        sortedStationList= getSortedStationList(entityTag)
        
        #Reordering temperature dataframe
        reorderedTempData = tempData.reindex(columns = sortedStationList)
        #Creating combination for all the weather station temperature series (Simple Average)
        combinationDf = pd.DataFrame()
        for i in range(len(reorderedTempData.columns)):
            combinationDf["C" + str(i+1)] = reorderedTempData.iloc[:,0:i+1].mean(axis=1)
        return combinationDf
        
    def modelPredictor(self,demandData, tempData, holidayDf, entityTag, vStation, modelStartDate = "2018-01-01 00:00:00", 
                  addTrend= False, addLagDF= True, addHoliday= True):
    
        
        #Bucket to store predictions from different combinations
        predBucket = pd.DataFrame(index= np.arange(1,97))
        predBucket.index.name = "Timeblock"
        #Generating dummy variable using dummy variable generator
        trend, timeblockDummies, dayDummies, monthDummies, daytimeblockDummies = self.dummyVariableGenerator()
        
        testCombinationTemp = self.stationCombinationFinder(tempData, entityTag)
        
        X_input = pd.concat([trend.loc[demandData.index[0]:demandData.index[-1]] if addTrend== True else None,
                                #timeblockDummies.loc[demandData.index[0]:demandData.index[-1]].iloc[:, :-1],
                                #dayDummies.loc[demandData.index[0]:demandData.index[-1]].iloc[:, :-1],
                                monthDummies.loc[demandData.index[0]:demandData.index[-1]].iloc[:, :-1],
                                daytimeblockDummies.loc[demandData.index[0]:demandData.index[-1]].iloc[:, :-1],
                                testCombinationTemp["{}".format(vStation)].loc[demandData.index[0]:demandData.index[-1]], 
                                testCombinationTemp["{}".format(vStation)].loc[demandData.index[0]:demandData.index[-1]] ** 2, 
                                testCombinationTemp["{}".format(vStation)].loc[demandData.index[0]:demandData.index[-1]] ** 3,
                               self.getInteractions(testCombinationTemp["{}".format(vStation)].loc[demandData.index[0]:demandData.index[-1]], 
                                                timeblockDummies.loc[demandData.index[0]:demandData.index[-1]], poly_degree= 3),
                                self.getInteractions(testCombinationTemp["{}".format(vStation)].loc[demandData.index[0]:demandData.index[-1]], 
                                                monthDummies.loc[demandData.index[0]:demandData.index[-1]], poly_degree= 3),
                                demandData if addLagDF== True else None,
                                holidayDf if addHoliday== True else None
                                ], axis=1, join= 'inner')
        
        X_arr_input = X_input.values
        #Loading saved model
        prediction_obj = joblib.load(self.modelPathStr)
        #Predicting DA ahead
        Y_pred = pd.Series(prediction_obj.predict(X_arr_input).flatten(), index= pd.DatetimeIndex(X_input.index) + pd.DateOffset(0), name= "Y_pred")
        #Storing predictions in bucket
        predBucket[str(Y_pred.index[0].strftime("%Y-%m-%d"))] = Y_pred.values
        return Y_pred
    

    def predictDaMlr(self, lagDemandDf:pd.core.frame.DataFrame, weatherDf:pd.core.frame.DataFrame, holidayDf:pd.core.frame.DataFrame, entityTag:str, virtualWeatherStsnNo:str)-> pd.core.frame.DataFrame:
        """predict DA forecast using model based on entityTag

        Args:
            lagDemandDf (pd.core.frame.DataFrame): dataframe containing blockwise D-2, D-7, D-14, D-21 demand with index timestamp of 'D'
            weatherDf(pd.core.frame.DataFrame): Weather data of forecasted date
            holidayDf(pd.core.frame.DataFrame) : holiday dummies in df
            entityTag (str): entity tag like 'WRLDCMP.SCADA1.A0047000'
            virtualWeatherStsnNo(str): virtual weather stsn number of particular entity say 'C6'
        Returns:
            pd.core.frame.DataFrame: DA demand forecast with column(timestamp, entityTag, demandValue)
        """    

        #setting model path string(class variable) based on entityTag(means deciding which model t0 use)
        self.modelPathStr = self.modelPath + '\\' + str(entityTag) +'_with_weather.pkl'
        
        daPredictionSeries= self.modelPredictor(lagDemandDf, weatherDf, holidayDf, entityTag, virtualWeatherStsnNo )
        daPredictionDf = daPredictionSeries.to_frame()

        #adding entityTag column and resetting index
        daPredictionDf.insert(0, "entityTag", entityTag)  
        daPredictionDf.reset_index(inplace=True)

        #renaming columns
        daPredictionDf.rename(columns={'index': 'timestamp', 'Y_pred': 'forecastedDemand'}, inplace=True)
        
        return daPredictionDf


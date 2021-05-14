import cx_Oracle
import datetime as dt
import pandas as pd 
from typing import List, Tuple
import numpy as np


class HolidayDummyGenerator():
    """repository to generate holiday dummies and return datframe .
    """

    def __init__(self, holidayExcelPathStr: str) -> None:
        """initialize connection string
        Args:
           holidayExcelPathStr ([str]): holiday excel path string
        """
        self.holidayExcelPathStr = holidayExcelPathStr
    
    def generateHolidayDummies(self,forecastDate:dt.datetime) -> pd.core.frame.DataFrame:
        """generate holiday dummies for fprecaste date

        Args:
            forecastDate (dt.datetime): day of forecast

        Returns:
            pd.core.frame.DataFrame: dataframe containing holiday dummies
        """        

        holidayExcelPath = self.holidayExcelPathStr + "\\Holidays_List.xlsx"
        holidayListDf = pd.read_excel(holidayExcelPath)
        
        timeIndex = pd.date_range(start = pd.Timestamp(forecastDate), periods=96,freq ='15min').rename("time").to_frame()
        holidayDummyDF = pd.DataFrame(index = timeIndex.index)
        holidayDummyDF["Is_Weekend"] = np.where(timeIndex.index.weekday < 6, 0, 1)
        holidayDummyDF["Holiday"] = np.isin(holidayDummyDF.index.date, holidayListDf.Date.dt.date).astype("int")
        return holidayDummyDF
import cx_Oracle
import datetime as dt
import pandas as pd 
from typing import List, Tuple


class DayAheadDemandForecastInsertion():
    """repository to push day ahead forecasted demand of entities to db.
    """

    def __init__(self, con_string: str) -> None:
        """initialize connection string
        Args:
            con_string ([type]): connection string 
        """
        self.connString = con_string
    
    def toListOfTuple(self,df:pd.core.frame.DataFrame) -> dict:
        """convert forecasted BLOCKWISE demand data to list of tuples[(timestamp,entityTag,forecastedValue),]
        Args:
            df (pd.core.frame.DataFrame): forecasted block wise demand dataframe
        Returns:
            List[Tuple]: list of tuple of forecasted blockwise demand data [(timestamp,entityTag,forecastedValue),]
        """ 
        forecastedData:List[Tuple] = []
        r0aForecastStore:List[Tuple]= []
        data = {'forecastData':forecastedData, 'r0aForecastStore':r0aForecastStore}

        for ind in df.index:
            forecastedTuple = (str(df['timestamp'][ind]), df['entityTag'][ind], float(df['forecastedDemand'][ind]) )
            forecastedData.append(forecastedTuple)
            r0aForecastedStoreTuple = (str(df['timestamp'][ind]), df['entityTag'][ind],'R0A', float(df['forecastedDemand'][ind]) )
            r0aForecastStore.append(r0aForecastedStoreTuple)
        return data

    def insertDayAheadDemandForecast(self, daForecastDf:pd.core.frame.DataFrame) -> bool:
        """Insert blockwise DayAheadDemandForecast of entities to db
        Args:
            self : object of class 
            daForecastDf(pd.core.frame.DataFrame:): dataframe of(timestamp, entityTag, forecastedDemand)
        Returns:
            bool: return true if insertion is successful else false
        """

        #converting dataframe to list of tuples.
        data = self.toListOfTuple(daForecastDf)
        
        
        # making list of tuple of timestamp(unique),entityTag based on which deletion takes place before insertion of duplicate
        existingForecastRows = [(x[0],x[1]) for x in data['forecastData']]
        existingR0aForecastRows = [(x[0],x[1],x[2]) for x in data['r0aForecastStore']]

        try:
            
            connection = cx_Oracle.connect(self.connString)
            isInsertionSuccess = True

        except Exception as err:
            print('error while creating a connection', err)
        else:

            try:
                cur = connection.cursor()
                try:
                    cur.execute("ALTER SESSION SET NLS_DATE_FORMAT = 'YYYY-MM-DD HH24:MI:SS' ")
                    #inserting DA forecast
                    del_sql_forecast = "DELETE FROM dfm3_dayahead_demand_forecast WHERE time_stamp = :1 and entity_tag=:2"
                    cur.executemany(del_sql_forecast,existingForecastRows)
                    insert_sql = "INSERT INTO dfm3_dayahead_demand_forecast(time_stamp,ENTITY_TAG,forecasted_demand_value) VALUES(:1, :2, :3)"
                    cur.executemany(insert_sql, data['forecastData'])
                    #storing DA forecast as r0A
                    del_sql_r0a = "DELETE FROM dfm3_forecast_revision_store WHERE time_stamp = :1 and entity_tag=:2 and revision_no=:3"
                    cur.executemany(del_sql_r0a, existingR0aForecastRows)
                    insert_sql = "INSERT INTO dfm3_forecast_revision_store(time_stamp,ENTITY_TAG,revision_no, forecasted_demand_value) VALUES(:1, :2, :3, :4)"
                    cur.executemany(insert_sql, data['r0aForecastStore'])

                except Exception as e:
                    print("error while insertion/deletion->", e)
                    isInsertionSuccess = False
            except Exception as err:
                print('error while creating a cursor', err)
                isInsertionSuccess = False
            else:
                connection.commit()
        finally:
            cur.close()
            connection.close()
        return isInsertionSuccess
import pandas as pd
import datetime as dt
import os,io,cx_Oracle

class WeatherDataFetcher():
    """this class is to fetch weather data
    """    
    def __init__(self, con_string):
        """initialize connection string
        Args:
            con_string ([type]): connection string 
        """
        self.connString = con_string

    def fetchCityData(self, con:cx_Oracle.Connection, cur:cx_Oracle.Cursor, startTime:dt.datetime, endTime:dt.datetime, tag:str)->pd.core.frame.DataFrame:
        """fethc weather data of all station of a particular entity

        Args:
            con (cx_Oracle.Connection): [connection]
            cur (cx_Oracle.Cursor): [cursor]
            startTime (dt.datetime): [startime]
            endTime (dt.datetime): [endTIme]
            tag (str): [entityTag]

        Returns:
            pd.core.frame.DataFrame: [return weather dataframe of all station of a particular entity]
        """        
        guj,mp,mah,chh,goa,dd,dnh=False,False,False,False,False,False,False
        if tag=="WRLDCMP.SCADA1.A0047000":
            guj,mp,mah,chh,goa,dd,dnh=True,True,True,True,True,True,True
        elif tag=="WRLDCMP.SCADA1.A0046957":
            guj=True
        elif tag=="WRLDCMP.SCADA1.A0046978":
            mp=True
        elif tag=="WRLDCMP.SCADA1.A0046980":
            mah=True
        elif tag=="WRLDCMP.SCADA1.A0046945":
            chh=True
        elif tag=="WRLDCMP.SCADA1.A0046962":
            goa=True
        elif tag=="WRLDCMP.SCADA1.A0046948":
            dd=True
        elif tag=="WRLDCMP.SCADA1.A0046953":
            dnh=True
        try:
            df2=pd.DataFrame()
            if guj:
                cur.execute('''select time,temperature from solar where city='station_1' and (time between :1 and :2 ) order by time''',(startTime, endTime)) 
                df=pd.DataFrame(cur.fetchall(),columns=['time', 'station_1'])
                df.set_index('time',inplace=True)
                df2=pd.concat([df2,df],axis=1)

                cur.execute('''select time,temperature from solar where city='station_2' and (time between :1 and :2 ) order by time''',(startTime, endTime)) 
                df=pd.DataFrame(cur.fetchall(),columns=['time', 'station_2'])
                df.set_index('time',inplace=True)
                df2=pd.concat([df2,df],axis=1)
                
                cur.execute('''select time,temperature from solar where city='station_3' and (time between :1 and :2 ) order by time''',(startTime, endTime)) 
                df=pd.DataFrame(cur.fetchall(),columns=['time', 'station_3'])
                df.set_index('time',inplace=True)
                df2=pd.concat([df2,df],axis=1)
                
                cur.execute('''select time,temperature from solar where city='station_4' and (time between :1 and :2 ) order by time''',(startTime, endTime)) 
                df=pd.DataFrame(cur.fetchall(),columns=['time', 'station_4'])
                df.set_index('time',inplace=True)
                df2=pd.concat([df2,df],axis=1)
                
                cur.execute('''select time,temperature from solar where city='station_5' and (time between :1 and :2 ) order by time''',(startTime, endTime)) 
                df=pd.DataFrame(cur.fetchall(),columns=['time', 'station_5'])
                df.set_index('time',inplace=True)
                df2=pd.concat([df2,df],axis=1)
                
                cur.execute('''select time,temperature from solar where city='station_6' and (time between :1 and :2 ) order by time''',(startTime, endTime)) 
                df=pd.DataFrame(cur.fetchall(),columns=['time', 'station_6'])
                df.set_index('time',inplace=True)
                df2=pd.concat([df2,df],axis=1)
                
                cur.execute('''select time,temperature from solar where city='station_7' and (time between :1 and :2 ) order by time''',(startTime, endTime)) 
                df=pd.DataFrame(cur.fetchall(),columns=['time', 'station_7'])
                df.set_index('time',inplace=True)
                df2=pd.concat([df2,df],axis=1)
                
                cur.execute('''select time,temperature from solar where city='station_8' and (time between :1 and :2 ) order by time''',(startTime, endTime)) 
                df=pd.DataFrame(cur.fetchall(),columns=['time', 'station_8'])
                df.set_index('time',inplace=True)
                df2=pd.concat([df2,df],axis=1)
                
                cur.execute('''select time,temperature from solar where city='station_9' and (time between :1 and :2 ) order by time''',(startTime, endTime)) 
                df=pd.DataFrame(cur.fetchall(),columns=['time', 'station_9'])
                df.set_index('time',inplace=True)
                df2=pd.concat([df2,df],axis=1)
            
            if mp:
                cur.execute('''select time,temperature from solar where city='station_10' and (time between :1 and :2 ) order by time''',(startTime, endTime)) 
                df=pd.DataFrame(cur.fetchall(),columns=['time', 'station_10'])
                df.set_index('time',inplace=True)
                df2=pd.concat([df2,df],axis=1)
                
                cur.execute('''select time,temperature from solar where city='station_11' and (time between :1 and :2 ) order by time''',(startTime, endTime)) 
                df=pd.DataFrame(cur.fetchall(),columns=['time', 'station_11'])
                df.set_index('time',inplace=True)
                df2=pd.concat([df2,df],axis=1)
                
                cur.execute('''select time,temperature from solar  where city='station_12' and (time between :1 and :2 ) order by time''',(startTime, endTime)) 
                df=pd.DataFrame(cur.fetchall(),columns=['time', 'station_12'])
                df.set_index('time',inplace=True)
                df2=pd.concat([df2,df],axis=1)
                
                cur.execute('''select time,temperature from solar  where city='station_13' and (time between :1 and :2 ) order by time''',(startTime, endTime)) 
                df=pd.DataFrame(cur.fetchall(),columns=['time', 'station_13'])
                df.set_index('time',inplace=True)
                df2=pd.concat([df2,df],axis=1)
                
                cur.execute('''select time,temperature from solar  where city='station_14' and (time between :1 and :2 ) order by time''',(startTime, endTime)) 
                df=pd.DataFrame(cur.fetchall(),columns=['time', 'station_14'])
                df.set_index('time',inplace=True)
                df2=pd.concat([df2,df],axis=1)
                
                cur.execute('''select time,temperature from solar  where city='station_15' and (time between :1 and :2 ) order by time''',(startTime, endTime)) 
                df=pd.DataFrame(cur.fetchall(),columns=['time', 'station_15'])
                df.set_index('time',inplace=True)
                df2=pd.concat([df2,df],axis=1)
                
                cur.execute('''select time,temperature from solar  where city='station_16' and (time between :1 and :2 ) order by time''',(startTime, endTime)) 
                df=pd.DataFrame(cur.fetchall(),columns=['time', 'station_16'])
                df.set_index('time',inplace=True)
                df2=pd.concat([df2,df],axis=1)
                
            if mah:
                cur.execute('''select time,temperature from solar where city='station_17' and (time between :1 and :2 ) order by time''',(startTime, endTime)) 
                df=pd.DataFrame(cur.fetchall(),columns=['time', 'station_17'])
                df.set_index('time',inplace=True)
                df2=pd.concat([df2,df],axis=1)
                
                cur.execute('''select time,temperature from solar where city='station_18' and (time between :1 and :2 ) order by time''',(startTime, endTime)) 
                df=pd.DataFrame(cur.fetchall(),columns=['time', 'station_18'])
                df.set_index('time',inplace=True)
                df2=pd.concat([df2,df],axis=1)
                
                cur.execute('''select time,temperature from solar where city='station_19' and (time between :1 and :2 ) order by time''',(startTime, endTime)) 
                df=pd.DataFrame(cur.fetchall(),columns=['time', 'station_19'])
                df.set_index('time',inplace=True)
                df2=pd.concat([df2,df],axis=1)
                
                cur.execute('''select time,temperature from solar where city='station_20' and (time between :1 and :2 ) order by time''',(startTime, endTime)) 
                df=pd.DataFrame(cur.fetchall(),columns=['time', 'station_20'])
                df.set_index('time',inplace=True)
                df2=pd.concat([df2,df],axis=1)
                
                cur.execute('''select time,temperature from solar where city='station_21' and (time between :1 and :2 ) order by time''',(startTime, endTime)) 
                df=pd.DataFrame(cur.fetchall(),columns=['time', 'station_21'])
                df.set_index('time',inplace=True)
                df2=pd.concat([df2,df],axis=1)
                
                cur.execute('''select time,temperature from solar where city='station_22' and (time between :1 and :2 ) order by time''',(startTime, endTime)) 
                df=pd.DataFrame(cur.fetchall(),columns=['time', 'station_22'])
                df.set_index('time',inplace=True)
                df2=pd.concat([df2,df],axis=1)
                
                cur.execute('''select time,temperature from solar where city='station_23' and (time between :1 and :2 ) order by time''',(startTime, endTime)) 
                df=pd.DataFrame(cur.fetchall(),columns=['time', 'station_23'])
                df.set_index('time',inplace=True)
                df2=pd.concat([df2,df],axis=1)
                
                cur.execute('''select time,temperature from solar where city='station_24' and (time between :1 and :2 ) order by time''',(startTime, endTime)) 
                df=pd.DataFrame(cur.fetchall(),columns=['time', 'station_24'])
                df.set_index('time',inplace=True)
                df2=pd.concat([df2,df],axis=1)
                
                cur.execute('''select time,temperature from solar where city='station_25' and (time between :1 and :2 ) order by time''',(startTime, endTime)) 
                df=pd.DataFrame(cur.fetchall(),columns=['time', 'station_25'])
                df.set_index('time',inplace=True)
                df2=pd.concat([df2,df],axis=1)
                
                cur.execute('''select time,temperature from solar where city='station_26' and (time between :1 and :2 ) order by time''',(startTime, endTime)) 
                df=pd.DataFrame(cur.fetchall(),columns=['time', 'station_26'])
                df.set_index('time',inplace=True)
                df2=pd.concat([df2,df],axis=1)
                
                cur.execute('''select time,temperature from solar where city='station_27' and (time between :1 and :2 ) order by time''',(startTime, endTime)) 
                df=pd.DataFrame(cur.fetchall(),columns=['time', 'station_27'])
                df.set_index('time',inplace=True)
                df2=pd.concat([df2,df],axis=1)
                
                cur.execute('''select time,temperature from solar where city='station_28' and (time between :1 and :2 ) order by time''',(startTime, endTime)) 
                df=pd.DataFrame(cur.fetchall(),columns=['time', 'station_28'])
                df.set_index('time',inplace=True)
                df2=pd.concat([df2,df],axis=1)
                
                cur.execute('''select time,temperature from solar where city='station_29' and (time between :1 and :2 ) order by time''',(startTime, endTime)) 
                df=pd.DataFrame(cur.fetchall(),columns=['time', 'station_29'])
                df.set_index('time',inplace=True)
                df2=pd.concat([df2,df],axis=1)
                
                cur.execute('''select time,temperature from solar where city='station_30' and (time between :1 and :2 ) order by time''',(startTime, endTime)) 
                df=pd.DataFrame(cur.fetchall(),columns=['time', 'station_30'])
                df.set_index('time',inplace=True)
                df2=pd.concat([df2,df],axis=1)
                
                cur.execute('''select time,temperature from solar where city='station_31' and (time between :1 and :2 ) order by time''',(startTime, endTime)) 
                df=pd.DataFrame(cur.fetchall(),columns=['time', 'station_31'])
                df.set_index('time',inplace=True)
                df2=pd.concat([df2,df],axis=1)
                
                cur.execute('''select time,temperature from solar where city='station_32' and (time between :1 and :2 ) order by time''',(startTime, endTime)) 
                df=pd.DataFrame(cur.fetchall(),columns=['time', 'station_32'])
                df.set_index('time',inplace=True)
                df2=pd.concat([df2,df],axis=1)
                
                cur.execute('''select time,temperature from solar where city='station_33' and (time between :1 and :2 ) order by time''',(startTime, endTime)) 
                df=pd.DataFrame(cur.fetchall(),columns=['time', 'station_33'])
                df.set_index('time',inplace=True)
                df2=pd.concat([df2,df],axis=1)
                
                cur.execute('''select time,temperature from solar where city='station_34' and (time between :1 and :2 ) order by time''',(startTime, endTime)) 
                df=pd.DataFrame(cur.fetchall(),columns=['time', 'station_34'])
                df.set_index('time',inplace=True)
                df2=pd.concat([df2,df],axis=1)
                
                cur.execute('''select time,temperature from solar where city='station_35' and (time between :1 and :2 ) order by time''',(startTime, endTime)) 
                df=pd.DataFrame(cur.fetchall(),columns=['time', 'station_35'])
                df.set_index('time',inplace=True)
                df2=pd.concat([df2,df],axis=1)
                
                cur.execute('''select time,temperature from solar where city='station_36' and (time between :1 and :2 ) order by time''',(startTime, endTime)) 
                df=pd.DataFrame(cur.fetchall(),columns=['time', 'station_36'])
                df.set_index('time',inplace=True)
                df2=pd.concat([df2,df],axis=1)
                
                cur.execute('''select time,temperature from solar where city='station_37' and (time between :1 and :2 ) order by time''',(startTime, endTime)) 
                df=pd.DataFrame(cur.fetchall(),columns=['time', 'station_37'])
                df.set_index('time',inplace=True)
                df2=pd.concat([df2,df],axis=1)
                
            if chh:
                cur.execute('''select time,temperature from solar where city='station_38' and (time between :1 and :2 ) order by time''',(startTime, endTime)) 
                df=pd.DataFrame(cur.fetchall(),columns=['time', 'station_38'])
                df.set_index('time',inplace=True)
                df2=pd.concat([df2,df],axis=1)
                
                cur.execute('''select time,temperature from solar where city='station_39' and (time between :1 and :2 ) order by time''',(startTime, endTime)) 
                df=pd.DataFrame(cur.fetchall(),columns=['time', 'station_39'])
                df.set_index('time',inplace=True)
                df2=pd.concat([df2,df],axis=1)
                
                cur.execute('''select time,temperature from solar where city='station_40' and (time between :1 and :2 ) order by time''',(startTime, endTime)) 
                df=pd.DataFrame(cur.fetchall(),columns=['time', 'station_40'])
                df.set_index('time',inplace=True)
                df2=pd.concat([df2,df],axis=1)
                
            if goa:
                cur.execute('''select time,temperature from solar where city='station_41' and (time between :1 and :2 ) order by time''',(startTime, endTime)) 
                df=pd.DataFrame(cur.fetchall(),columns=['time', 'station_41'])
                df.set_index('time',inplace=True)
                df2=pd.concat([df2,df],axis=1)
                
            if dd:
                cur.execute('''select time,temperature from solar where city='station_42' and (time between :1 and :2 ) order by time''',(startTime, endTime)) 
                df=pd.DataFrame(cur.fetchall(),columns=['time', 'station_42'])
                df.set_index('time',inplace=True)
                df2=pd.concat([df2,df],axis=1)
                
                cur.execute('''select time,temperature from solar where city='station_43' and (time between :1 and :2 ) order by time''',(startTime, endTime)) 
                df=pd.DataFrame(cur.fetchall(),columns=['time', 'station_43'])
                df.set_index('time',inplace=True)
                df2=pd.concat([df2,df],axis=1)
                
            if dnh:
                cur.execute('''select time,temperature from solar where city='station_44' and (time between :1 and :2 ) order by time''',(startTime, endTime)) 
                df=pd.DataFrame(cur.fetchall(),columns=['time', 'station_44'])
                df.set_index('time',inplace=True)
                df2=pd.concat([df2,df],axis=1)
                # print(df2)
        except Exception as e:
            print(e)
            return pd.DataFrame()
        return df2

    def fetchWeatherData(self, forecastDate:dt.datetime, entityTag:str)->pd.core.frame.DataFrame:
        """method to fetch weather data

        Args:
            forecastDate (dt.datetime): [forecast date]
            entityTag (str): [entity tag of wr entities]

        Returns:
            pd.core.frame.DataFrame: [return weather dataframe of all station of a particular entity]
        """        

        startTime= forecastDate
        endTime= forecastDate + dt.timedelta(hours=23, minutes=59)

        try: 
            connection = cx_Oracle.connect(self.connString)

        except Exception as err:
            print('error while creating a connection', err)
        else:
            try:
                cur = connection.cursor()
                weatherDf=self.fetchCityData(connection, cur, startTime, endTime, entityTag)

            except Exception as err:
                print('error while creating a cursor', err)
            else:
                connection.commit()
        finally:
            cur.close()
            connection.close()
        return weatherDf
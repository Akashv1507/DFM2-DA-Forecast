def getAppConfigDict()->dict:
    """returns dictionary that contains appConfiguration
    Returns:
        dict: dictionary containing app configuration.
    """    
    import pandas as pd 
    # file path of config excel
    configFilePath = "config.xlsx"
    # get the config excel into a df
    configDf = pd.read_excel(configFilePath, header=None)
    # make the first column as index
    configDf.set_index(configDf.columns[0], inplace=True)
    # convert the first column into a series and then to a dictionary
    # now we have a dictionary that has config key as key and config value as value
    configDict = configDf[configDf.columns[0]].to_dict()
    # return the config value based on it key from the config dictionary
    return configDict
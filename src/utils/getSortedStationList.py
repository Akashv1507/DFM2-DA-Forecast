

from typing import List


def getSortedStationList(entityTag: str)-> List:
    """get sorted station list for any particular entity

    Args:
        entityTag (str): entity tag (WRLDCMP.SCADA1.A0046978)

    Returns:
        List: ['station_5', 'station_28', 'station_17', 'station_44' and so on..]
    """    

    # creating variable from function parameter.
    resDict={}
    entitySortedStationList = "sortedStationList_" + entityTag[-8:]
    resDict[entitySortedStationList] =[]
    # WR-sorted station list
    resDict['sortedStationList_A0047000']= ['station_43', 'station_11', 'station_10', 'station_15',
       'station_12', 'station_14', 'station_5', 'station_13',
       'station_31', 'station_26', 'station_16', 'station_9',
       'station_6', 'station_36', 'station_32', 'station_37',
       'station_40', 'station_22', 'station_7', 'station_19',
       'station_30', 'station_4', 'station_1', 'station_8',
       'station_27', 'station_3', 'station_34', 'station_39',
       'station_28', 'station_33', 'station_20', 'station_2',
       'station_38', 'station_24', 'station_35', 'station_23',
       'station_44', 'station_25', 'station_29', 'station_21',
       'station_17', 'station_42', 'station_18', 'station_41']

    # Mah-sorted station list
    resDict['sortedStationList_A0046980']= ['station_43', 'station_8', 'station_10', 'station_14',
       'station_22', 'station_24', 'station_16', 'station_20',
       'station_23', 'station_11', 'station_7', 'station_31',
       'station_5', 'station_40', 'station_33', 'station_15',
       'station_26', 'station_9', 'station_34', 'station_12',
       'station_36', 'station_2', 'station_37', 'station_1',
       'station_13', 'station_28', 'station_35', 'station_30',
       'station_6', 'station_32', 'station_4', 'station_25',
       'station_19', 'station_44', 'station_39', 'station_3',
       'station_27', 'station_38', 'station_21', 'station_29',
       'station_41', 'station_18', 'station_17', 'station_42']

    # Guj-sorted station list
    resDict['sortedStationList_A0046957']= ['station_5', 'station_43', 'station_1', 'station_4',
       'station_9', 'station_11', 'station_10', 'station_3',
       'station_24', 'station_32', 'station_31', 'station_14',
       'station_6', 'station_20', 'station_26', 'station_15',
       'station_16', 'station_37', 'station_27', 'station_22',
       'station_12', 'station_36', 'station_2', 'station_33',
       'station_7', 'station_8', 'station_19', 'station_34',
       'station_23', 'station_35', 'station_28', 'station_17',
       'station_44', 'station_30', 'station_25', 'station_29',
       'station_21', 'station_13', 'station_42', 'station_39',
       'station_38', 'station_18', 'station_40', 'station_41']

    # MP-sorted station list
    resDict['sortedStationList_A0046978']=['station_13', 'station_15', 'station_12', 'station_11',
       'station_40', 'station_6', 'station_14', 'station_10',
       'station_9', 'station_19', 'station_39', 'station_38',
       'station_43', 'station_1', 'station_16', 'station_5',
       'station_4', 'station_7', 'station_30', 'station_36',
       'station_44', 'station_26', 'station_31', 'station_20',
       'station_34', 'station_2', 'station_28', 'station_37',
       'station_8', 'station_3', 'station_22', 'station_24',
       'station_21', 'station_35', 'station_41', 'station_32',
       'station_27', 'station_42', 'station_25', 'station_33',
       'station_18', 'station_17', 'station_29', 'station_23']

    resDict['sortedStationList_A0046945'] =['station_38', 'station_40', 'station_12', 'station_39',
       'station_26', 'station_11', 'station_15', 'station_43',
       'station_19', 'station_22', 'station_30', 'station_10',
       'station_31', 'station_34', 'station_5', 'station_14',
       'station_13', 'station_27', 'station_33', 'station_36',
       'station_28', 'station_35', 'station_32', 'station_2',
       'station_6', 'station_37', 'station_16', 'station_23',
       'station_7', 'station_9', 'station_3', 'station_25',
       'station_8', 'station_44', 'station_20', 'station_21',
       'station_29', 'station_1', 'station_24', 'station_4',
       'station_18', 'station_42', 'station_17', 'station_41']

    return resDict[entitySortedStationList]


import os
from mouse_imaging.session import dates_to_keys, get_metadata, define_path
from mouse_imaging.functions import load_pickle


keys = []

keys.extend(
    dates_to_keys(
    mouse='JG202', 
    dates=['201205', '201206', '201207', '201208', '201209', '201210', '201211', '201212', '201213', '201214', 
         '201215', '201216', '201217', '201218']))

keys.extend(
    dates_to_keys(
    mouse='JG203', 
    dates = ['201205', '201206', '201207', '201208', '201209', '201210', '201211', '201212', '201213', 
         '201214', '201215', '201216', '201217', '201218']))

keys.extend(
    dates_to_keys(
    mouse='JG209', 
    dates = ['210114', '210119', '210120', '210121', '210125', '210126', '210127', '210128', '210129', '210130', 
         '210131',  '210203', '210204', '210205', '210206', '210207',]))

keys.extend(
    dates_to_keys(
    mouse='JG210', 
    dates = ['201227', '201228', '201231', '210101', '210102', '210104', '210105', '210106', '210107', '210108',
         '210109', '210110']))

keys.extend(
    dates_to_keys(
    mouse='JG212', 
    dates = [ dict(date='201222', session='session_2'), '201223', '201224', '201225', '201227', '201228', 
         '201231', '210101', '210102', '210104', '210105', '210107',]))

keys.extend(
    dates_to_keys(
    mouse='JG220', 
    dates = ['210218', '210219', '210220', '210221', '210223', '210224', '210225', '210226', '210227', 
         '210228', '210301', '210302', '210303', '210304', '210305', '210306', '210307', '210308', 
         '210310', '210311', '210314', '210315', '210316', '210317', '210318', '210319']))

keys.extend(
    dates_to_keys(
    mouse='JG221', 
    dates = ['210218', '210219', dict(date='210220', session='session_2'), '210221', '210222', '210224', 
         '210225', '210226', '210227', '210228', '210301',]))

keys.extend(
    dates_to_keys(
    mouse='JG222', 
    dates = [dict(date='210219', session='session_2'), '210221', '210222', '210223', '210224', '210225', 
         '210227', '210228', '210301', '210302', '210303']))

keys.extend(
    dates_to_keys(
    mouse='JG224', 
    dates = ['210219', '210220', '210221', '210222', '210223', '210224', '210225', '210226', '210227', 
         '210228', '210301', '210302']))

keys.extend(
    dates_to_keys(
    mouse='JG230', 
    dates = ['210514', '210515', '210516', '210517', '210518', '210519', '210521', '210522', '210523', 
         '210524', '210525', '210526', '210527', '210528', '210529', '210530']))

keys.extend(
    dates_to_keys(
    mouse='JG231', 
    dates = ['210409', '210410', '210411', '210412', '210414', '210415', '210416', '210417', '210418', 
         '210419', '210420', '210422', '210423', '210424', '210425', '210426', '210427', 
         '210428', '210429']))

keys.extend(
    dates_to_keys(
    mouse='JG233', 
    dates = ['210409', '210410', '210411', '210412', '210413', '210414', '210415', '210416', '210417', 
         '210418', '210419', '210420', '210421', '210422', '210423',]))

keys.extend(
    dates_to_keys(
    mouse='JG234', 
    dates = ['210412', '210413', '210414', '210415', '210416', '210417', '210418', '210419', '210420', 
         '210421', '210422', '210423', '210424', '210425', '210426', '210427', '210428', '210429', 
         '210430',]))

keys.extend(
    dates_to_keys(
    mouse='JG235', 
    dates = ['210410', '210411', '210412', '210413', '210414', '210415', '210416', '210417', '210418', 
         dict(date='210419', session='session_3'), '210420', '210421', '210422', '210423', '210424', '210425', '210426', '210427', 
         '210428', '210429', '210430', '210503', '210504', '210505']))


def filter_keys(keys, filter_key):
    keys2 = []
    for key in keys:
        path = define_path(**key)
        if os.path.isfile(path['metadata_pickle']):
            md = load_pickle(path['metadata_pickle'])
        else:
            md = get_metadata(path)
        try:
            test = [md[k]==v for k, v in filter_key.items()]
        except KeyError:
            print(key)
        if all(test):
            keys2.append(key)
    return keys2

mice = set([key['mouse'] for key in keys])

def load(filter_key=None):
    global keys
    keys2 = keys.copy()
    if filter_key is not None:
        keys2 = filter_keys(keys2, filter_key)
    return keys2
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 10:36:35 2019

@author: yindandan
"""

import json
import pandas as pd
import numpy as np
from sklearn.externals import joblib
# from model.MyEncoder import MyEncoder, MyDecoder


keys = ['sb_yl_pay_cardi_sum',
 'sb_yli_total_pay_std',
 'sb_yli_pson_pay_avg',
 'sb_gs_pay_cardi_max',
 'sb_yli_pay_cardi_max',
 'sb_syu_comp_pay_std',
 'sb_yli_total_pay_avg',
 'sb_yl_pson_pay_avg',
 'sb_yl_pson_pay_sum',
 'sb_gs_comp_pay_min',
 'sb_syu_pay_months',
 'sb_yli_soins_company',
 'sb_yli_total_pay_sum',
 'sb_sy_pson_pay_std',
 'sb_yl_total_pay_avg',
 'sb_sy_comp_pay_max',
 'sb_yl_pson_pay_max',
 'sb_yli_pay_cardi_std',
 'sb_yl_comp_pay_std',
 'sb_gs_comp_pay_sum',
 'sb_sy_comp_pay_std',
 'sb_yl_pay_cardi_max',
 'sb_yl_comp_pay_min',
 'sb_gs_comp_pay_avg',
 'sb_yli_comp_pay_avg',
 'sb_sy_pson_pay_min',
 'sb_sy_pay_cardi_min',
 'sb_sy_pay_cardi_max',
 'sb_yli_pson_pay_max',
 'sb_syu_pay_cardi_std',
 'sb_yl_pay_cardi_std']

#levels = [10.036, 12.186, 13.945, 15.738, 17.543, 19.496, 21.944, 25.268, 30.906, 10000]

def is_not_exits(json_dict, key_list):
    ll = set(key_list) - (set(key_list) & json_dict.keys())
    if len(ll) > 0:
        raise KeyError('Not Found Key :' + str(ll))


#def get_level(score):
#    for x in range(levels.__len__()):
#        if score < levels[x]:
#            return 'L' + str(x + 1)
#    return 'L10'


def get_model_data(str_json_dict):
    json_dict = json.loads(str_json_dict)

    is_not_exits(json_dict, keys)

    mg_credit_data = pd.DataFrame.from_dict(json_dict, orient='index').T

    mg_credit_feature = mg_credit_data.loc[:, keys]
    mg_credit_feature = mg_credit_feature.astype(float)

    # model calc
    mg_credit_clf = joblib.load('fqt_sb_model_v1.pkl')

    mg_credit_predict = mg_credit_clf.predict_proba(mg_credit_feature.values)[:, 1]
    mg_credit_score = mg_credit_predict

    # return result
    import datetime
    nowTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data_return = {
        "score": int(base_score+pdo/np.log(2)*np.log(base_odds/(mg_credit_score/(1-mg_credit_score)))),
        "create_time": nowTime}
        # "score": float('%.6f' % mg_credit_score), "credit_level": get_level(mg_credit_score), "create_time": nowTime}

    return json.dumps(data_return)

base_score = 500; base_odds = 0.1; pdo = 40

if __name__ == '__main__':
    data = {'bound_time': '2018-08-06 11:14:54',
             'customer_id': '00013e00bfb547b092fd5db7773d9377',
             'sb_gs_comp_pay_avg': 8.47338235294118,
             'sb_gs_comp_pay_min': '10.44',
             'sb_gs_comp_pay_sum': 576.1900000000002,
             'sb_gs_pay_cardi_max': '2130.0',
             'sb_score_stand': 516,
             'sb_sy_comp_pay_max': '32.48',
             'sb_sy_comp_pay_std': 5.843863658600997,
             'sb_sy_pay_cardi_max': '2130.0',
             'sb_sy_pay_cardi_min': '1500.0',
             'sb_sy_pson_pay_min': '10.15',
             'sb_sy_pson_pay_std': 4.018782279734087,
             'sb_syu_comp_pay_std': np.nan,
             'sb_syu_pay_cardi_std':np.nan,
             'sb_syu_pay_months':np.nan,
             'sb_yl_comp_pay_min': '132.0',
             'sb_yl_comp_pay_std': 46.1627004114165,
             'sb_yl_pay_cardi_max': '2130.0',
             'sb_yl_pay_cardi_std': 255.82547446862438,
             'sb_yl_pay_cardi_sum': 63270.0,
             'sb_yl_pson_pay_avg': 158.17499999999998,
             'sb_yl_pson_pay_max': '170.4',
             'sb_yl_pson_pay_sum': 5061.599999999999,
             'sb_yl_total_pay_avg': 410.0906249999998,
             'sb_yli_comp_pay_avg': 25.07558823529414,
             'sb_yli_pay_cardi_max': '4205.0',
             'sb_yli_pay_cardi_std': 0.0,
             'sb_yli_pson_pay_avg': 6.063088235294124,
             'sb_yli_pson_pay_max': '8.41',
             'sb_yli_soins_company': 2.0,
             'sb_yli_total_pay_avg': 31.138676470588265,
             'sb_yli_total_pay_std': 9.27232616717524,
             'sb_yli_total_pay_sum': 2117.430000000002}
    print(get_model_data(json.dumps(data))) # score:10.2319


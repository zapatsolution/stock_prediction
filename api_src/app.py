from flask import Flask
from flask_restful import Resource, Api,reqparse
# import modules to load the model
import numpy as np
import os
import sys
import time
from datetime import datetime
import pandas as pd
from tqdm._tqdm_notebook import tqdm_notebook
from keras.models import Sequential, load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import logging
import tensorflow as tf
import pickle
from flask import jsonify
from flask_cors import CORS
from flask import Flask, render_template, make_response
from matplotlib import pyplot as plt
from statsmodels.tsa.seasonal import STL
import statsmodels as sm
from pytz import timezone
app = Flask(__name__)
origin = r'*'
CORS(app)
# CORS(app, origin=origin, allow_headers=[
#         "Content-Type", "Authorization", "Access-Control-Allow-Credentials", "Access-Control-Allow-Origin"],
#          supports_credentials=True, intercept_exceptions=True)
# cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
# load the model only once
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ['TZ'] = 'Asia/Kolkata'  # to set timezone; needed when running on cloud
time.tzset()

params = {
    "batch_size": 20,  # 20<16<10, 25 was a bust
    "epochs": 200,
    "lr": 0.00010000,
    "time_steps": 60
}
COL_INDEX = 1
iter_changes = "dropout_layers_0.4_0.4"
OUTPUT_PATH = 'models/best_model_1.h5'
TIME_STEPS = params["time_steps"]
BATCH_SIZE = params["batch_size"]
stime = time.time()

# check if directory already exists
if not os.path.exists(OUTPUT_PATH):
    raise Exception("Directory already exists. Don't override.")
else:
    pass
def print_time(text, stime):
    seconds = (time.time() - stime)
    print(text, seconds // 60, "minutes : ", np.round(seconds % 60), "seconds")


def trim_dataset(mat, batch_size):
    """
    trims dataset to a size that's divisible by BATCH_SIZE
    """
    no_of_rows_drop = mat.shape[0] % batch_size
    if no_of_rows_drop > 0:
        return mat[:-no_of_rows_drop]
    else:
        return mat


def build_timeseries(mat, y_col_index):
    """
    Converts ndarray into timeseries format and supervised data format. Takes first TIME_STEPS
    number of rows as input and sets the TIME_STEPS+1th data as corresponding output and so on.
    :param mat: ndarray which holds the dataset
    :param y_col_index: index of column which acts as output
    :return: returns two ndarrays-- input and output in format suitable to feed
    to LSTM.
    """
    # total number of time-series samples would be len(mat) - TIME_STEPS
    dim_0 = mat.shape[0] - TIME_STEPS
    dim_1 = mat.shape[1]
    x = np.zeros((dim_0, TIME_STEPS, dim_1))


    for i in tqdm_notebook(range(dim_0)):
        x[i] = mat[i:TIME_STEPS + i]

    #         if i < 10:
    #           print(i,"-->", x[i,-1,:], y[i])
    print("length of time-series i/o", x.shape)
    return x
min_max_scaler = MinMaxScaler()
min_max_scaler_t = MinMaxScaler()
# load model here
# saved_model = load_model(os.path.join(OUTPUT_PATH))  # , "lstm_best_7-3-19_12AM",

# model = pickle.load(open('/home/manoj/Downloads/office/trading_bot/Research/lstm_model_1', 'rb'))


# print(saved_model)
# saved_model._make_predict_function()
# graph = tf.get_default_graph()

app = Flask(__name__)
api = Api(app)

df  = pd.read_csv('Data/bitcoin_seasonal_Data.csv')
df = df[["time","open", "high", "low", "close", "Volume","open_seasonal","close_seasonal","low_seasonal","high_seasonal"]]
sdate = 1587808800
no_days = 7
hours = 24*no_days

def add_seasonal_data(df):
    try:

        resopen = sm.tsa.seasonal.seasonal_decompose(df['open'], model='additive', freq=24)
        df['open_seasonal']  = resopen.seasonal
        resclose = sm.tsa.seasonal.seasonal_decompose(df['close'], model='additive', freq=24)
        df['close_seasonal'] = resclose.seasonal
        reslow = sm.tsa.seasonal.seasonal_decompose(df['low'], model='additive', freq=24)
        df['low_seasonal']= reslow.seasonal
        reshigh = sm.tsa.seasonal.seasonal_decompose(df['high'], model='additive', freq=24)
        df['high_seasonal'] =reshigh.seasonal

        del resopen
        del resclose
        del reslow
        del reshigh
    except Exception as ex:
        pass
    return df

def season_data(df_t,df_pred,test_cols):
    df_t = df_t[test_cols]
    df_pred = pd.DataFrame(df_pred)
    df_pred.columns = test_cols
    df_t =df_t.append(df_pred)
    df_t = add_seasonal_data(df_t)
    return df_t.tail(len(df_pred)).to_numpy()


# saved_model = load_model(os.path.join(OUTPUT_PATH))
def predict_future(sdate,no_days,saved_model):
    dates = range(sdate,(hours*3600)+(sdate),3600)
    dates = list(dates)
    y_pred_main = []
    dft = df[df['time']<sdate]

    # initialize the min_max_scaler to be used in overall data
    train_cols = ["open", "high", "low", "close", "Volume", "open_seasonal", "close_seasonal", "low_seasonal",
                  "high_seasonal"]
    test_cols = ["open", "high", "low", "close", "Volume"]
    test_t = min_max_scaler.fit_transform(dft.loc[:,train_cols])
    min_t = min_max_scaler_t.fit_transform(dft.loc[:, test_cols])


    dfp = pd.DataFrame()
    if len(dfp)<hours:
        if len(dfp)==0:
            while len(y_pred_main)<=len(dates):
                n_Data = 80-len(y_pred_main)
                if n_Data>0:

                    df_test = dft.tail(n_Data)
                    df_test = df_test.append(pd.DataFrame(y_pred_main,columns=df_test.columns[1:]))
                else:
                    df_test = y_pred_main[-80:]
                    df_test = pd.DataFrame(df_test,columns=df.columns[1:])
                    # df_test= df_test.append(y_pred_main[:n])
                if len(df_test.columns)==9:

                    x_test = df_test
                else:
                    x_test = df_test.iloc[:,1:]

                x_test = min_max_scaler.transform(x_test)
                x_temp = build_timeseries(x_test, COL_INDEX)
                x_test_t = trim_dataset(x_temp, BATCH_SIZE)
                # global graph
                # with graph.as_default():
                y_pred = saved_model.predict(trim_dataset(x_test_t, BATCH_SIZE), batch_size=BATCH_SIZE)
                # y_pred_org = (y_pred * min_max_scaler.data_range_[3]) + min_max_scaler.data_min_[3]
                # print(y_pred_org)
                y_pred_org = min_max_scaler_t.inverse_transform(y_pred)
                if len(y_pred_main) == 0:
                    y_pred_main = y_pred_org
                    y_pred_main = season_data(dft, pd.DataFrame(y_pred_main), test_cols)
                else:
                    y_pred_org = season_data(dft, pd.DataFrame(y_pred_org), test_cols)
                    y_pred_main = np.vstack([y_pred_main,y_pred_org])


            y_pred_main = pd.DataFrame(y_pred_main)
            y_pred_main = y_pred_main[:len(dates)]
            y_pred_main.insert(0, 'time', dates)

            # y_pred_main = y_pred_org.iloc([:len(dates)])

            return y_pred_main
# y_pred_m = predict_future(sdate,no_days)


def create_comparision(pred,real):
    predc = pred.copy()

    realc = real.copy()
    predc.columns = ["time","open", "high", "low", "close", "Volume", "open_seasonal", "close_seasonal", "low_seasonal",
                  "high_seasonal"]
    st = predc.head(1)

    end = predc.tail(1)
    realc = realc[realc['time']>=st['time'].iloc[0]]
    realc = realc[realc['time'] <= end['time'].iloc[-1]]
    realc = realc.reset_index()
    realc = realc.drop('index',axis=1)
    colspred = predc.columns
    colsreal = realc.columns
    for i in range(1,5):
        plt.figure()
        plt.plot(predc[colspred[i]])
        plt.plot(realc[colsreal[i]])
        plt.title('Prediction vs Real Stock Price')
        plt.ylabel(colsreal[i])
        plt.xlabel('Days')
        plt.legend(['Prediction', 'Real'], loc='upper left')
    # plt.show()
        plt.savefig(os.path.join('/home/manoj/Downloads/office/trading_bot/comparision', colsreal[i]+"_" + time.ctime() + '.png'))

    pass

class GetRealData(Resource):
    def get(self):
        headers = {'Content-Type': 'text/html'}
        return make_response(render_template('index.html'), 200, headers)
        # return render_template("index.html", message="Hello Flask!");

class GetPredictedData(Resource):
    def get(self):
        saved_model = load_model(OUTPUT_PATH)  # , "lstm_best_7-3-19_12AM",

        parser = reqparse.RequestParser()
        parser.add_argument('starttime', type=str)
        parser.add_argument('endtime', type=str)
        keys = parser.parse_args()
        starttime = keys['starttime']
        starttime = datetime.strptime(starttime, '%Y-%m-%d')
        starttime = starttime.replace(tzinfo=timezone('GMT'))
        starttime = datetime.timestamp(starttime)
        endtime = keys['endtime']
        starttime = int(starttime)
        # 1589369411000
        endtime = int(endtime)
        hours = endtime-starttime

        # x_test = df[df['time']>starttime]
        # x_test = x_test[x_test['time']<endtime]
        real_Data = df.copy()

        # x_test = x_test.iloc[:,1:]
        #
        # x_test = min_max_scaler.fit_transform(x_test)
        # x_temp = build_timeseries(x_test, COL_INDEX)
        # x_test_t = trim_dataset(x_temp, BATCH_SIZE)
        # global graph
        # with graph.as_default():
        # y_pred = saved_model.predict(trim_dataset(x_test_t, BATCH_SIZE), batch_size=BATCH_SIZE)
        # y_pred_org = (y_pred * min_max_scaler.data_range_[3]) + min_max_scaler.data_min_[3]
        # y_pred_org = pd.DataFrame(y_pred_org)

        # y_pred_org.insert(0, 'time', real_Data['time'][:len(y_pred_org)].reset_index()['time'])
        # real_Data = real_Data.reset_index()
        # real_Data = real_Data.drop('index',axis=1)
        # y_pred_org['time'] = y_pred_org['time'].apply(lambda x: '{0:0<12}'.format(x))
        # y_pred_org['time'] = y_pred_org['time']*1000

        # y_pred_org = y_pred_org.to_json()

        y_pred_m = predict_future(starttime, no_days,saved_model)
        # create_comparision(y_pred_m, real_Data)
        real_Data['time'] = real_Data['time'] * 1000
        y_pred_m['time']= y_pred_m['time']*1000


        real_Data = real_Data.to_json()
        y_pred_m = y_pred_m.to_json()
        # y_pred_org =y_pred_org.to_numpy()
        # real_Data = real_Data.to_numpy()
        # return jsonify({'predictions':y_pred_org.tolist(),'realdata':real_Data.tolist()})
        return {"pred":y_pred_m,"real":real_Data}

api.add_resource(GetRealData, '/api/real')

api.add_resource(GetPredictedData, '/api/predicted')

if __name__ == '__main__':
    app.run(host='0.0.0.0',port='8006',debug=True)
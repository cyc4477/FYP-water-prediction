import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.svm import SVR, LinearSVR
from sklearn.neural_network import MLPRegressor
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU
from keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt


def load_data():

    # Set the file path of datasets
    file3 = Path('D:\yun chi\大四\FYP\开农3').glob('*.xlsx')
    file7 = Path('D:\yun chi\大四\FYP\开农7').glob('*.xlsx')
    file8 = Path('D:\yun chi\大四\FYP\开农8').glob('*.xlsx')
    dfs = []
    dfs_Y = []

    # Start reading datasets
    for f in file3:
        X_col = [0]
        df = pd.read_excel(f, usecols=X_col)
        Y_col = [1]
        df_Y = pd.read_excel(f, usecols=Y_col)
        title = f.stem.split("-")
        df['temperature(℃)']=title[0].replace("℃","")
        df['initial water(%)'] = title[1].replace("%","")
        df['velocity(m^3/(kg,h))'] = title[2].replace("立方米每小时每千克", "")
        df['height(mm)'] = title[3].replace("mm","")

        dfs.append(df)
        dfs_Y.append(df_Y)
    for f in file7:
        X_col = [0]
        df = pd.read_excel(f, usecols=X_col)
        Y_col = [1]
        df_Y = pd.read_excel(f, usecols=Y_col)
        title = (f.stem).split("-")
        df['temperature(℃)'] = title[0].replace("℃", "")
        df['initial water(%)'] = title[1].replace("%", "")
        df['velocity(m^3/(kg,h))'] = title[2].replace("立方米每小时每千克", "")
        df['height(mm)'] = title[3].replace("mm", "")

        dfs.append(df)
        dfs_Y.append(df_Y)
    for f in file8:
        X_col = [0]
        df = pd.read_excel(f, usecols=X_col)
        Y_col = [1]
        df_Y = pd.read_excel(f, usecols=Y_col)
        title = (f.stem).split("-")
        df['temperature(℃)'] = title[0].replace("℃", "")
        df['initial water(%)'] = title[1].replace("%", "")
        df['velocity(m^3/(kg,h))'] = title[2].replace("立方米每小时每千克", "")
        df['height(mm)'] = title[3].replace("mm", "")

        dfs.append(df)
        dfs_Y.append(df_Y)
    df = pd.concat(dfs, ignore_index=True)
    df_Y = pd.concat(dfs_Y,ignore_index=True)
    #print(df)
    #print(df_Y)
    return df,df_Y

def preprocess_data(df, df_Y, user_df):

    combine_df = df.append(user_df, ignore_index=True)
    # print(combine_df)
    scaler = MinMaxScaler()
    scaler.fit(combine_df)
    transform_df = pd.DataFrame(data=scaler.transform(combine_df), columns=['t(h)', 'temp(℃)', 'i-water(%)', 'v(m^3/(kg,h))', 'h(mm)'])
    scaler.fit(df_Y)
    transform_df_Y = pd.DataFrame(data=scaler.transform(df_Y), columns=['water_contain(%)'])

    transform_user_data = transform_df.iloc[1620:]
    transform_df = transform_df.drop([1620])
    # print(transform_df.loc[[1001]])
    # print(transform_df)
    return transform_df, transform_df_Y, transform_user_data

def preprocess_data1(df, df_Y):

    scaler = MinMaxScaler()
    scaler.fit(df)
    transform_df = pd.DataFrame(data=scaler.transform(df), columns=['t(h)', 'temp(℃)', 'i-water(%)', 'v(m^3/(kg,h))', 'h(mm)'])
    scaler.fit(df_Y)
    transform_df_Y = pd.DataFrame(data=scaler.transform(df_Y), columns=['water_contain(%)'])

    return transform_df, transform_df_Y

def clean_data(transform_df, transform_df_Y):
    Sum = transform_df.isnull().sum()
    Percentage = (transform_df.isnull().sum() / transform_df.isnull().count())

    pd.concat([Sum, Percentage], axis=1, keys=['Sum', 'Percentage'])

def null_cell(df):
    total_missing_values = df.isnull().sum()
    missing_values_per = df.isnull().sum()/df.isnull().count()
    null_values = pd.concat([total_missing_values, missing_values_per], axis=1, keys=['total_null', 'total_null_perc'])
    null_values = null_values.sort_values('total_null', ascending=False)
    return null_values[null_values['total_null'] > 0]

def undoTransform(prediction):

    scaler = MinMaxScaler()
    scaler.fit(prediction)
    undoPrediction = scaler.inverse_transform(prediction)

    return undoPrediction

def ridge_linear_regression(transform_df, transform_df_y):

    # Set alpha in range and split the training and testing datasets
    alpha = np.arange(0.00001,3).tolist()
    X_train, X_test, y_train, y_test = train_test_split(transform_df, transform_df_y, test_size=0.3, random_state=5)
    print(y_train)
    print(y_test)
    # Set configurations and fit the ridge model --> Output the prediction
    ridge_reg = RidgeCV(alphas=alpha,cv=5)
    ridge_reg.fit(X_train, y_train)
    test_prediction = ridge_reg.predict(X_test)
    print(test_prediction.mean())

    # Calculate the R square score
    score = r2_score(X_train, y_train)
    print("R^2 value Score: ", score)

    # Calculate the train and test RMSE
    train_prediction = ridge_reg.predict(X_train)
    train_rmse = (mean_squared_error(train_prediction, y_train)) ** 0.5
    print("train_rmse = ", train_rmse)
    test_rmse = (mean_squared_error(test_prediction, y_test)) ** 0.5
    print("test_rmse = ", test_rmse)
    print("alpha = ", ridge_reg.alpha_)

    plt.subplot(2,2,1)
    XP1 = transform_df['t(h)'].values.reshape(-1,1)
    YP1 = transform_df_y['water_contain(%)'].values.reshape(-1,1)
    ridge_reg.fit(XP1, YP1)
    plt.title("water content - time")
    plt.xlabel("time")
    plt.ylabel("water content")
    plt.plot(XP1, YP1,'k.')
    plt.plot(XP1, ridge_reg.predict(XP1), 'g')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    XP2 = transform_df['temp(℃)'].values.reshape(-1, 1)
    YP2 = transform_df_y['water_contain(%)'].values.reshape(-1, 1)
    ridge_reg.fit(XP2, YP2)
    plt.title("water content - temp")
    plt.xlabel("temperature")
    plt.ylabel("water content")
    plt.plot(XP2, YP2, 'k.')
    plt.plot(XP2, ridge_reg.predict(XP2), 'g')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    XP3 = transform_df['v(m^3/(kg,h))'].values.reshape(-1, 1)
    YP3 = transform_df_y['water_contain(%)'].values.reshape(-1, 1)
    ridge_reg.fit(XP3, YP3)
    plt.title("water content - vel")
    plt.xlabel("velocity")
    plt.ylabel("water content")
    plt.plot(XP3, YP3, 'k.')
    plt.plot(XP3, ridge_reg.predict(XP3), 'g')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    XP4 = transform_df['h(mm)'].values.reshape(-1, 1)
    YP4 = transform_df_y['water_contain(%)'].values.reshape(-1, 1)
    ridge_reg.fit(XP4, YP4)
    plt.title("water content - height")
    plt.xlabel("height")
    plt.ylabel("water content")
    plt.plot(XP4, YP4, 'k.')
    plt.plot(XP4, ridge_reg.predict(XP4), 'g')
    plt.grid(True)
    plt.show()

def lasso_linear_regression(transform_df, transform_df_y):

    # Set alpha in range and split the training and testing datasets
    alpha = np.arange(0.00001, 3).tolist()
    X_train, X_test, y_train, y_test = train_test_split(transform_df, transform_df_y, test_size=0.3, random_state=5)

    # Set configurations and fit the ridge model --> Output the prediction
    lasso_reg = LassoCV(alphas=alpha, cv=5)
    lasso_reg.fit(X_train, y_train)
    test_prediction = lasso_reg.predict(X_test)
    print(test_prediction.mean())

    # Calculate the R square score
    score = r2_score(X_train, y_train)
    print("R^2 value Score: ", score)

    # Calculate the train and test RMSE
    train_prediction = lasso_reg.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(train_prediction, y_train))
    print("train_rmse = ", train_rmse)
    test_rmse = np.sqrt(mean_squared_error(y_test,test_prediction))
    print("test_rmse = ", test_rmse)
    print("alphas = ", lasso_reg.alpha_.mean())

def SVRegression(transform_df, transform_df_y):

    # Set epsilon and split the training and testing datasets
    eps = 0.1
    X_train, X_test, y_train, y_test = train_test_split(transform_df, transform_df_y, test_size=0.3, random_state=5)

    # Set configurations and fit the ridge model --> Output the prediction
    support_vector_reg = SVR(gamma=2**(-1), C=18, epsilon=eps)
    support_vector_reg.fit(X_train, y_train)
    test_prediction = support_vector_reg.predict(X_test)
    print(test_prediction.mean())

    # Calculate the R square score
    score = r2_score(y_test, test_prediction)
    print("R^2 value Score: ", score)

    # Calculate the train and test RMSE
    train_prediction = support_vector_reg.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(train_prediction, y_train))
    print("train_rmse = ", train_rmse)
    test_rmse = np.sqrt(mean_squared_error(y_test,test_prediction))
    print("test_rmse = ", test_rmse, "C: ", support_vector_reg.C, "gamma: ", support_vector_reg.gamma,"eps: ", support_vector_reg.epsilon)


    df_prediction = X_test

    df_prediction.insert(5, 'prediction', test_prediction)

    test_index = df_prediction.index

    idx = df_prediction.index[df_prediction['h(mm)'] == 1.0].tolist()
    df_prediction1 = df_prediction.loc[idx]
    idx1_q1 = df_prediction1.index[df_prediction1['v(m^3/(kg,h))'] <= 0.25].tolist()

    idx1_q2_t = df_prediction1.index[df_prediction1['v(m^3/(kg,h))'] > 0.25].tolist()
    df_temp = df_prediction1.loc[idx1_q2_t]
    idx1_q2 = df_temp.index[df_temp['v(m^3/(kg,h))'] <= 0.5].tolist()

    idx1_q3_t = df_prediction1.index[df_prediction1['v(m^3/(kg,h))'] > 0.5].tolist()
    df_temp = df_prediction1.loc[idx1_q3_t]
    idx1_q3 = df_temp.index[df_temp['v(m^3/(kg,h))'] <= 0.75].tolist()

    idx1_q4 = df_prediction1.index[df_prediction1['v(m^3/(kg,h))'] > 0.75].tolist()

    idx2 = df_prediction.index[df_prediction['h(mm)'] == 0.0].tolist()
    df_prediction2 = df_prediction.loc[idx2]
    idx2_q1 = df_prediction2.index[df_prediction2['v(m^3/(kg,h))'] <= 0.25].tolist()

    idx2_q2_t = df_prediction2.index[df_prediction2['v(m^3/(kg,h))'] > 0.25].tolist()
    df_temp = df_prediction2.loc[idx2_q2_t]
    idx2_q2 = df_temp.index[df_temp['v(m^3/(kg,h))'] <= 0.5].tolist()

    idx2_q3_t = df_prediction2.index[df_prediction2['v(m^3/(kg,h))'] > 0.5].tolist()
    df_temp = df_prediction2.loc[idx2_q3_t]
    idx2_q3 = df_temp.index[df_temp['v(m^3/(kg,h))'] <= 0.75].tolist()

    idx2_q4 = df_prediction2.index[df_prediction2['v(m^3/(kg,h))'] > 0.75].tolist()

    return test_prediction, test_index,  idx1_q1, idx1_q2, idx1_q3, idx1_q4, idx2_q1, idx2_q2, idx2_q3, idx2_q4

def SVRegressionTEST(transform_df, transform_df_y,xtest):

    X_train, X_test, y_train, y_test = train_test_split(transform_df, transform_df_y, test_size=0.3, random_state=5)

    support_vector_reg = SVR(gamma=2**(-1), C=2**2, epsilon=0.1)
    support_vector_reg.fit(X_train, y_train)
    test_prediction = support_vector_reg.predict(xtest)
    print(test_prediction)

    score = support_vector_reg.score(X_train, y_train)
    print("R^2 value Score: ", score)

    train_prediction = support_vector_reg.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(train_prediction, y_train))
    print("train_rmse = ", train_rmse)

    print()
    return test_prediction

def LinearSVRegression(transform_df, transform_df_y):

    # Set epsilon and split the training and testing datasets
    eps = 0.1
    X_train, X_test, y_train, y_test = train_test_split(transform_df, transform_df_y, test_size=0.3, random_state=5)

    # Set configurations and fit the ridge model --> Output the prediction
    support_vector_reg = LinearSVR(C=6, epsilon=eps)
    support_vector_reg.fit(X_train, y_train)
    test_prediction = support_vector_reg.predict(X_test)
    print(test_prediction.mean())

    # Calculate the R square score
    score = r2_score(X_train, y_train)
    print("R^2 value Score: ", score)

    # Calculate the train and test RMSE
    train_prediction = support_vector_reg.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(train_prediction, y_train))
    print("train_rmse = ", train_rmse)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_prediction))
    print("test_rmse = ", test_rmse)


    df_prediction = X_test
    df_prediction.insert(5, 'prediction',  test_prediction)

    test_index = df_prediction.index

    idx = df_prediction.index[df_prediction['h(mm)'] == 1.0].tolist()
    df_prediction1 = df_prediction.loc[idx]
    idx1_q1 = df_prediction1.index[df_prediction1['v(m^3/(kg,h))'] <= 0.25].tolist()

    idx1_q2_t = df_prediction1.index[df_prediction1['v(m^3/(kg,h))'] > 0.25].tolist()
    df_temp = df_prediction1.loc[idx1_q2_t]
    idx1_q2 = df_temp.index[df_temp['v(m^3/(kg,h))'] <= 0.5].tolist()

    idx1_q3_t = df_prediction1.index[df_prediction1['v(m^3/(kg,h))'] > 0.5].tolist()
    df_temp = df_prediction1.loc[idx1_q3_t]
    idx1_q3 = df_temp.index[df_temp['v(m^3/(kg,h))'] <= 0.75].tolist()

    idx1_q4 = df_prediction1.index[df_prediction1['v(m^3/(kg,h))'] > 0.75].tolist()

    idx2 = df_prediction.index[df_prediction['h(mm)'] == 0.0].tolist()
    df_prediction2 = df_prediction.loc[idx2]
    idx2_q1 = df_prediction2.index[df_prediction2['v(m^3/(kg,h))'] <= 0.25].tolist()

    idx2_q2_t = df_prediction2.index[df_prediction2['v(m^3/(kg,h))'] > 0.25].tolist()
    df_temp = df_prediction2.loc[idx2_q2_t]
    idx2_q2 = df_temp.index[df_temp['v(m^3/(kg,h))'] <= 0.5].tolist()

    idx2_q3_t = df_prediction2.index[df_prediction2['v(m^3/(kg,h))'] > 0.5].tolist()
    df_temp = df_prediction2.loc[idx2_q3_t]
    idx2_q3 = df_temp.index[df_temp['v(m^3/(kg,h))'] <= 0.75].tolist()

    idx2_q4 = df_prediction2.index[df_prediction2['v(m^3/(kg,h))'] > 0.75].tolist()
    return test_prediction, test_index,  idx1_q1, idx1_q2, idx1_q3, idx1_q4, idx2_q1, idx2_q2, idx2_q3, idx2_q4

def MLP(transform_df, transform_df_y):

    # Split the training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(transform_df, transform_df_y, test_size=0.3, random_state=5)

    # Set configurations and fit the ridge model --> Output the prediction
    mlp_reg = MLPRegressor(hidden_layer_sizes=3000, random_state=210)
    mlp_reg.fit(X_train, y_train)
    test_prediction = mlp_reg.predict(X_test)
    print(test_prediction.mean())

    # Calculate the R square score
    score = r2_score(X_train, y_train)
    print("R^2 value Score: ", score)

    # Calculate the train and test RMSE
    train_prediction = mlp_reg.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(train_prediction, y_train))
    print("train_rmse = ", train_rmse)
    test_rmse = np.sqrt(mean_squared_error(y_test,test_prediction))
    print("test_rmse = ", test_rmse)

    df_prediction = X_test
    df_prediction.insert(5, 'prediction', test_prediction)
    test_index = df_prediction.index

    idx = df_prediction.index[df_prediction['h(mm)'] == 1.0].tolist()
    df_prediction1 = df_prediction.loc[idx]
    idx1_q1 = df_prediction1.index[df_prediction1['v(m^3/(kg,h))'] <= 0.25].tolist()

    idx1_q2_t = df_prediction1.index[df_prediction1['v(m^3/(kg,h))'] > 0.25].tolist()
    df_temp = df_prediction1.loc[idx1_q2_t]
    idx1_q2 = df_temp.index[df_temp['v(m^3/(kg,h))'] <= 0.5].tolist()

    idx1_q3_t = df_prediction1.index[df_prediction1['v(m^3/(kg,h))'] > 0.5].tolist()
    df_temp = df_prediction1.loc[idx1_q3_t]
    idx1_q3 = df_temp.index[df_temp['v(m^3/(kg,h))'] <= 0.75].tolist()

    idx1_q4 = df_prediction1.index[df_prediction1['v(m^3/(kg,h))'] > 0.75].tolist()

    idx2 = df_prediction.index[df_prediction['h(mm)'] == 0.0].tolist()
    df_prediction2 = df_prediction.loc[idx2]
    idx2_q1 = df_prediction2.index[df_prediction2['v(m^3/(kg,h))'] <= 0.25].tolist()

    idx2_q2_t = df_prediction2.index[df_prediction2['v(m^3/(kg,h))'] > 0.25].tolist()
    df_temp = df_prediction2.loc[idx2_q2_t]
    idx2_q2 = df_temp.index[df_temp['v(m^3/(kg,h))'] <= 0.5].tolist()

    idx2_q3_t = df_prediction2.index[df_prediction2['v(m^3/(kg,h))'] > 0.5].tolist()
    df_temp = df_prediction2.loc[idx2_q3_t]
    idx2_q3 = df_temp.index[df_temp['v(m^3/(kg,h))'] <= 0.75].tolist()

    idx2_q4 = df_prediction2.index[df_prediction2['v(m^3/(kg,h))'] > 0.75].tolist()

    return test_prediction, test_index,  idx1_q1, idx1_q2, idx1_q3, idx1_q4, idx2_q1, idx2_q2, idx2_q3, idx2_q4

def Lstm(transform_df, transform_df_y):
    X_train, X_test, y_train, y_test = train_test_split(transform_df, transform_df_y, test_size=0.3, random_state=5)
    X_train, y_train = np.array(X_train), np.array(y_train)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # The LSTM architecture
    regressor = Sequential()
    # First LSTM layer with Dropout regularisation
    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))
    # Second LSTM layer
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))
    # Third LSTM layer
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))
    # Fourth LSTM layer
    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.2))
    # The output layer
    regressor.add(Dense(units=1))

    # Compiling the RNN
    regressor.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=100)
    # Fitting to the training set
    history = regressor.fit(X_train, y_train, epochs=200, batch_size=32,validation_split=0.001,
                               callbacks=[early_stop])
    print(regressor.summary())

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    test_prediction = regressor.predict(X_test)

    score = r2_score(y_test, test_prediction)
    print("R^2 value Score: ", score)

    train_prediction = regressor.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(train_prediction, y_train))
    print("train_rmse = ", train_rmse)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_prediction))
    print("test_rmse = ", test_rmse)
    print()

    return history, test_prediction, y_test

def GRU(transform_df, transform_df_y):
    X_train, X_test, y_train, y_test = train_test_split(transform_df, transform_df_y, test_size=0.3, random_state=5)

    X_train, y_train = np.array(X_train), np.array(y_train)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # The GRU architecture
    regressorGRU = Sequential()
    # First GRU layer with Dropout regularisation
    regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1), activation='tanh'))
    regressorGRU.add(Dropout(0.2))
    # Second GRU layer
    regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1), activation='tanh'))
    regressorGRU.add(Dropout(0.2))
    # Third GRU layer
    regressorGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1), activation='tanh'))
    regressorGRU.add(Dropout(0.2))
    # Fourth GRU layer
    regressorGRU.add(GRU(units=50, activation='tanh'))
    regressorGRU.add(Dropout(0.2))
    # The output layer
    regressorGRU.add(Dense(1))

    # Compiling the GRU
    regressorGRU.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=100)
    # Fitting to the training set
    history = regressorGRU.fit(X_train, y_train, epochs=250, batch_size=150, validation_split = 0.001, callbacks=[early_stop])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    test_prediction = regressorGRU.predict(X_test)
    print(regressorGRU.summary())

    score = r2_score(y_test, test_prediction)
    print("R^2 value Score: ", score)

    train_prediction = regressorGRU.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(train_prediction, y_train))
    print("train_rmse = ", train_rmse)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_prediction))
    print("test_rmse = ", test_rmse)
    print()

    return history, test_prediction, y_test

def SVRPLOT(final_df, inverse_pred, idx1_q1, idx1_q2, idx1_q3, idx1_q4, idx2_q1, idx2_q2, idx2_q3, idx2_q4):
    final_df.insert(5, 'prediction', inverse_pred)
    #print(scaler.inverse_transform(np.array(1.0).reshape(-1, 1)))
    plt.subplot(1, 2, 1)
    line_plot = sns.lineplot(data=final_df.loc[idx1_q1], x="时间（h）", y="prediction", hue="temperature(℃)")
    line_plot.set_title("temp prediction when h-400mm, v-<17", fontsize=12)
    line_plot.set_xlabel("time (h)")
    line_plot.set_ylabel("Moisture content (%)")
    # plt.show()

    plt.subplot(1, 2, 2)
    line_plot2 = sns.lineplot(data=final_df.loc[idx1_q2], x="时间（h）", y="prediction", hue="temperature(℃)")
    line_plot2.set_title("temp prediction when h-400mm, v-<28", fontsize=12)
    line_plot2.set_xlabel("time (h)")
    line_plot2.set_ylabel("Moisture content (%)")
    plt.show()

    plt.subplot(1, 2, 1)
    line_plot3 = sns.lineplot(data=final_df.loc[idx1_q3], x="时间（h）", y="prediction", hue="temperature(℃)")
    line_plot3.set_title("temp prediction when h-400mm, v-<38", fontsize=12)
    line_plot3.set_xlabel("time (h)")
    line_plot3.set_ylabel("Moisture content (%)")
    # plt.show()

    plt.subplot(1, 2, 2)
    line_plot4 = sns.lineplot(data=final_df.loc[idx1_q4], x="时间（h）", y="prediction", hue="temperature(℃)")
    line_plot4.set_title("temp prediction when h-400mm, v-<49", fontsize=12)
    line_plot4.set_xlabel("time (h)")
    line_plot4.set_ylabel("Moisture content (%)")
    plt.show()

    plt.subplot(1, 2, 1)
    line_plot2_1 = sns.lineplot(data=final_df.loc[idx2_q1], x="时间（h）", y="prediction", hue="temperature(℃)")
    line_plot2_1.set_title("temp prediction when h-100mm, v-<17", fontsize=12)
    line_plot2_1.set_xlabel("time (h)")
    line_plot2_1.set_ylabel("Moisture content (%)")
    # plt.show()

    plt.subplot(1, 2, 2)
    line_plot2_2 = sns.lineplot(data=final_df.loc[idx2_q2], x="时间（h）", y="prediction", hue="temperature(℃)")
    line_plot2_2.set_title("temp prediction when h-100mm, v-<28", fontsize=12)
    line_plot2_2.set_xlabel("time (h)")
    line_plot2_2.set_ylabel("Moisture content (%)")
    plt.show()

    plt.subplot(1, 2, 1)
    line_plot2_3 = sns.lineplot(data=final_df.loc[idx2_q3], x="时间（h）", y="prediction", hue="temperature(℃)")
    line_plot2_3.set_title("temp prediction when h-100mm, v-<38", fontsize=12)
    line_plot2_3.set_xlabel("time (h)")
    line_plot2_3.set_ylabel("Moisture content (%)")
    # plt.show()

    plt.subplot(1, 2, 2)
    line_plot2_4 = sns.lineplot(data=final_df.loc[idx2_q4], x="时间（h）", y="prediction", hue="temperature(℃)")
    line_plot2_4.set_title("temp prediction when h-100mm, v-<49", fontsize=12)
    line_plot2_4.set_xlabel("time (h)")
    line_plot2_4.set_ylabel("Moisture content (%)")
    plt.show()

def NNPLOT(history, inverse_pred, inverse_test):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['Train loss', 'Validation loss'], loc='upper right')
    plt.show()

    plt.figure(figsize=(10, 6))
    range_future = len(inverse_pred)
    plt.plot(np.arange(range_future), np.array(inverse_test),
             label='True Future')
    plt.plot(np.arange(range_future), np.array(inverse_pred),
             label='Prediction')
    plt.legend(loc='upper left')
    plt.title("True future vs prediction of moisture content prediction")
    plt.xlabel('future')
    plt.ylabel('water content')
    plt.show()

def peanut_moisture_prediction(xtest):
    df, df_Y = load_data()
    #print(df_Y)
    for col in df_Y:
        df_Y.loc[:,col].fillna(0, inplace=True)
    df_Y['water_contain'] = df_Y['100mm水分（%）'] + df_Y['400mm水分（%）']
    df_Y=df_Y.drop(['100mm水分（%）', '400mm水分（%）'], axis=1)
    print(df)

    transform_df, transform_df_Y, transform_user_data = preprocess_data(df, df_Y, xtest)
    return transform_df,transform_df_Y, transform_user_data

def peanut_moisture_prediction1():
    df, df_Y = load_data()
    #print(df_Y)
    for col in df_Y:
        df_Y.loc[:,col].fillna(0, inplace=True)
    df_Y['water_contain'] = df_Y['100mm水分（%）'] + df_Y['400mm水分（%）']
    df_Y=df_Y.drop(['100mm水分（%）', '400mm水分（%）'], axis=1)
    print(df)
    return df,df_Y

def startPrediction(xtest):

    transform_df, transform_df_y, transform_user_data = peanut_moisture_prediction(xtest)
    print(transform_df_y)
    fill_list = (null_cell(transform_df)).index
    cleaned_df = transform_df.copy()
    for col in fill_list:
        cleaned_df.loc[:,col].fillna(cleaned_df.loc[:,col].mean(), inplace=True)
    #print(cleaned_df.isnull().sum())
    #print(transform_df_y.isnull().sum())

    #ridge_linear_regression(transform_df,transform_df_y)
    #lasso_linear_regression(transform_df, transform_df_y)

    #prediction=SVRegression(transform_df, transform_df_y)
    prediction = SVRegressionTEST(transform_df, transform_df_y, transform_user_data)
    #LinearSVRegression(transform_df, transform_df_y)

    #MLP(transform_df, transform_df_y)
    #Lstm(transform_df, transform_df_y)
    #LstmGRU(transform_df, transform_df_y)
    final = undoTransform(prediction.reshape(-1, 1))
    #print(final.item(0))

    return final.item(0)

def main():
    # ======================================Read Data=====================================
    df, df_y = peanut_moisture_prediction1()

    # ======================================Scale Data=====================================
    scaler = MinMaxScaler()
    scaler.fit(df)
    transform_df = pd.DataFrame(data=scaler.transform(df),
                                columns=['t(h)', 'temp(℃)', 'i-water(%)', 'v(m^3/(kg,h))', 'h(mm)'])
    scaler.fit(df_y)
    transform_df_y = pd.DataFrame(data=scaler.transform(df_y), columns=['water_contain(%)'])
    print(transform_df_y)

    # ======================================Data Preprocessing=====================================
    fill_list = (null_cell(transform_df)).index
    cleaned_df = transform_df.copy()
    for col in fill_list:
        cleaned_df.loc[:, col].fillna(cleaned_df.loc[:, col].mean(), inplace=True)
    # print(cleaned_df.isnull().sum())

    # ======================================Linear Regression=====================================
    ridge_linear_regression(transform_df,transform_df_y)
    #lasso_linear_regression(transform_df, transform_df_y)

    # ======================================Supported Vector Regression=====================================
    #prediction, test_index,  idx1_q1, idx1_q2, idx1_q3, idx1_q4, idx2_q1, idx2_q2, idx2_q3, idx2_q4 =SVRegression(transform_df, transform_df_y)
    #prediction = SVRegressionTEST(transform_df, transform_df_y)
    #prediction, test_index,  idx1_q1, idx1_q2, idx1_q3, idx1_q4, idx2_q1, idx2_q2, idx2_q3, idx2_q4 = LinearSVRegression(transform_df, transform_df_y, df)

     # ======================================Neural Network=====================================
    #prediction, test_index,  idx1_q1, idx1_q2, idx1_q3, idx1_q4, idx2_q1, idx2_q2, idx2_q3, idx2_q4 = MLP(transform_df, transform_df_y)
    #history, prediction, y_test = Lstm(transform_df, transform_df_y)
    #history, prediction, y_test = GRU(transform_df, transform_df_y)
    # final = undoTransform(prediction.reshape(-1, 1))
    # print(final.item(0))

    # ======================================Plotting preparation and transformation=====================================
    #inverse_pred = scaler.inverse_transform(prediction.reshape(-1, 1))
    #final_df = df.loc[test_index]
    #SVRPLOT(final_df, inverse_pred, idx1_q1, idx1_q2, idx1_q3, idx1_q4, idx2_q1, idx2_q2, idx2_q3, idx2_q4)

    #inverse_test = scaler.inverse_transform(y_test)
    #NNPLOT(history, inverse_pred, inverse_test)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

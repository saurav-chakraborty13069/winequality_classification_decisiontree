import pickle
import pandas as pd
# from logger import App_Logger
# from patsy import dmatrices

#log_writer = App_Logger()
#file_object = open("logs/ModelPredictLog.txt", 'a+')

def load_models(log_writer,file_object):
    log_writer.log(file_object, 'Starting to load models')
    with open("models/standardScalar.sav", 'rb') as f:
        scalar = pickle.load(f)

    with open("models/modelForPrediction.sav", 'rb') as f:
        model = pickle.load(f)

    with open("models/modelpca.sav", 'rb') as f:
        pca = pickle.load(f)

    return scalar, model


def validate_data(dict_pred, log_writer,file_object):

    log_writer.log(file_object, 'Converting data to dataframe')
    final_df = pd.DataFrame(dict_pred, index = [1,])



    return final_df


def predict_data(dict_pred, log_writer):

    #validate the data entered
    #preprocess to get X in sme format
    #then apply models to predict
    file_object = open("logs/PredictionLogs.txt", 'a+')
    log_writer.log(file_object, 'Starting the predict data')

    scalar, model, pca = load_models(log_writer,file_object)
    log_writer.log(file_object, 'Loading of models completed')
    final_df = validate_data(dict_pred, log_writer, file_object)
    log_writer.log(file_object, 'Prepared the final dataframe')
    log_writer.log(file_object, 'Preprocessing the final dataframe with scalar and pca transform')
    scaled_data = scalar.transform(final_df)
    pca_data = pca.transform(scaled_data)
    principal_data = pd.DataFrame(pca_data,columns=['PC-1','PC-2','PC-3','PC-4','PC-5','PC-6','PC-7','PC-8','PC-9','PC-10'])
    log_writer.log(file_object, 'Predicting the result')
    predict = model.predict(principal_data)

    print('Class is:    ', predict[0])
    log_writer.log(file_object, 'Prediction completed')
    log_writer.log(file_object, '=================================================')
    return predict[0]



# predict_data(mydict)
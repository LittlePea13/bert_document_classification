"""
Unfortunately, one cannot include the exact data utilized to the train both the clinical models due to HIPPA constraints.
The data can be found here if you fill out the appropriate online forms:
https://portal.dbmi.hms.harvard.edu/data-challenges/

For training, simply alter the config.ini present in /examples file for your purposes. Relevant variables are:

model_storage_directory: directory to store logging information, tensorboard checkpoints, model checkpoints

bert_model_path: the file path to a pretrained bert mode. can be the pytorch-transformers alias.

labels: an ordered list of labels you are training against. this should match the order given in a .fit() instance.


"""

from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import resample
from bert_document_classification.document_bert import BertForDocumentClassification
from pprint import pformat
import sqlite3
import pandas as pd
import time, logging, torch, configargparse, os, socket
import numpy as np
log = logging.getLogger()

def _initialize_arguments(p: configargparse.ArgParser):
    p.add('--model_storage_directory', help='The directory caching all model runs')
    p.add('--bert_model_path', help='Model path to BERT')
    p.add('--labels', help='Numbers of labels to predict over', type=str)
    p.add('--architecture', help='Training architecture', type=str)
    p.add('--freeze_bert', help='Whether to freeze bert', type=bool)

    p.add('--batch_size', help='Batch size for training multi-label document classifier', type=int)
    p.add('--bert_batch_size', help='Batch size for feeding 510 token subsets of documents through BERT', type=int)
    p.add('--epochs', help='Epochs to train', type=int)
    #Optimizer arguments
    p.add('--learning_rate', help='Optimizer step size', type=float)
    p.add('--weight_decay', help='Adam regularization', type=float)
    p.add('--loss_bias', help='Loss factor to substract', type=float, default = 1.0)

    p.add('--evaluation_interval', help='Evaluate model on test set every evaluation_interval epochs', type=int)
    p.add('--checkpoint_interval', help='Save a model checkpoint to disk every checkpoint_interval epochs', type=int)

    #Non-config arguments
    p.add('--cuda', action='store_true', help='Utilize GPU for training or prediction')
    p.add('--device')
    p.add('--timestamp', help='Run specific signature')
    p.add('--model_directory', help='The directory storing this model run, a sub-directory of model_storage_directory')
    p.add('--use_tensorboard', help='Use tensorboard logging', type=bool)
    p.add('--downsampling', help='Use tensorboard logging', type=bool)
    args = p.parse_args()

    args.labels = [x for x in args.labels.split(', ')]





    #Set run specific envirorment configurations
    args.timestamp = time.strftime("run_%Y_%m_%d_%H_%M_%S") + "_{machine}".format(machine=socket.gethostname())
    args.model_directory = os.path.join(args.model_storage_directory, args.timestamp) #directory
    os.makedirs(args.model_directory, exist_ok=True)

    #Handle logging configurations
    log.handlers.clear()
    formatter = logging.Formatter('%(message)s')
    fh = logging.FileHandler(os.path.join(args.model_directory, "log.txt"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    log.setLevel(logging.INFO)
    log.addHandler(ch)
    log.info(p.format_values())


    #Set global GPU state
    if torch.cuda.is_available() and args.cuda:
        if torch.cuda.device_count() > 1:
            log.info("Using %i CUDA devices" % torch.cuda.device_count() )
        else:
            log.info("Using CUDA device:{0}".format(torch.cuda.current_device()))
        args.device = 'cuda'
    else:
        log.info("Not using CUDA :(")
        args.dev = 'cpu'

    return args

def load_data(database_path = 'database.db'):
    conn = sqlite3.connect(database_path)
    mapping = {-1:0,0:0,1:1,2:1}
    articles = pd.read_sql_query(
        "Select article_data.*, submited.Relevance, submited.disease, "
        "submited.technique from 'Submited Articles' as submited, "
        "Articles as article_data where submited.PMID = article_data.PMID group by submited.PMID", conn)
    countries_list = pd.read_sql_query("Select * from 'countries'", conn)
    conn.close()

    # full_data_set = fetch_details(list(articles.PMID.astype(str)))
    # full_data_set = pd.DataFrame([full_data_set[x] for x in full_data_set])

    articles["abstract"] = articles["abstract_text"]

    country_train = []
    for indx, pubmed_article in articles.iterrows():
        # affiliation
        try:
            # default country is empty
            country = ""

            for word in pubmed_article['affiliation'].split():
                word = re.sub('[^\w\s]', '', word)
                country_list = countries_list["name"][countries_list["alternate"] == word]
                # take the first country name (all the same)
                if len(country_list) > 0:
                    country = country_list.iloc[0]
                    break
        except:
            country = ""
        country_train.append(country)
    articles['country'] = country_train
    articles["estimation_text"] = (articles["title"] + " ") + \
                            (articles["country"] + " ") + \
                            articles["abstract"]

    articles["Relevance_Raw"] = articles["Relevance"]
    articles.loc[articles["Relevance"].isin([1, 2]), "Relevance"] = "Relevant"
    articles.loc[articles["Relevance"].isin([-1, 0]), "Relevance"] = "Not Relevant"
    return articles

if __name__ == "__main__":
    p = configargparse.ArgParser(default_config_files=["config.ini"])
    args = _initialize_arguments(p)

    torch.cuda.empty_cache()

    articles = load_data()

    kf = StratifiedKFold(n_splits=5, 
                     shuffle=True, 
                     random_state=29)
    i = 0
    for training_index, test_index in kf.split(articles, articles["Relevance"]):
        i += 1
        df_training = articles.iloc[training_index, :]
        df_test = articles.iloc[test_index, :]

        if args.downsampling:
            train_df_relevant = df_training[df_training["Relevance"] == "Relevant"]
            train_df_non_relevant = df_training[df_training["Relevance"] == "Not Relevant"] 
            train_df_down_not_relevant = resample(train_df_non_relevant,
                                                replace=False,
                                                n_samples=train_df_relevant.shape[0],
                                                random_state=32)
            df_training = pd.concat([train_df_relevant, train_df_down_not_relevant])
        #documents and labels for training
        training_labels = np.array(df_training["Relevance_Raw"]).reshape(-1,1)
        dev_labels = np.array(df_test["Relevance_Raw"]).reshape(-1,1)
        #training_labels = df_training["Relevance"].map({"Relevant":[1], "Not Relevant":[0]})
        #dev_labels = df_test["Relevance"].map({"Relevant":[1], "Not Relevant":[0]})
        train = (list(df_training["estimation_text"]), training_labels)
        dev = (list(df_test["estimation_text"]), dev_labels)

    #train_documents, train_labels = list(articles['title']+articles.abstract_text)[1:round(0.8*len(articles))],[[mapping[element]] for element in list(articles.Relevance)][1:round(0.8*len(articles))]


    #documents and labels for development
    #dev_documents, dev_labels = list(articles['title']+articles.abstract_text)[round(0.8*len(articles)):],[[mapping[element]] for element in list(articles.Relevance)][round(0.8*len(articles)):]

        model = BertForDocumentClassification(args=args)
        model.fit(train, dev)
        predictions, predictions_cont, precisions, recalls, fmeasures, aprecisions, precisionk = model.predict(dev)

        with open(os.path.join(args.model_directory "eval_cross_%s.csv" % i), 'w') as eval_results:
            eval_results.write('Precision\t' + '\t'.join([str(precisions[label_idx]) for label_idx in range(predictions.shape[0])]) + '\n' )
            eval_results.write('Recall\t' + '\t'.join([str(recalls[label_idx]) for label_idx in range(predictions.shape[0])]) + '\n' )
            eval_results.write('F1\t' + '\t'.join([ str(fmeasures[label_idx]) for label_idx in range(predictions.shape[0])]) + '\n' )
            eval_results.write('Average-Precision\t' + '\t'.join([ str(aprecisions[label_idx]) for label_idx in range(predictions.shape[0])]) + '\n' )
            eval_results.write('Precision-at-30\t' + '\t'.join([ str(precisionk[label_idx]) for label_idx in range(predictions.shape[0])]) + '\n' )
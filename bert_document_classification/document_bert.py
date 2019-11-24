from pytorch_transformers.modeling_bert import BertPreTrainedModel, BertConfig, BertModel
from pytorch_transformers.modeling_utils import WEIGHTS_NAME, CONFIG_NAME
from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.optimization import AdamW
from torch import nn
import torch,math,logging,os, warnings
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
from tqdm import tqdm
import numpy as np
from itertools import groupby
import importlib
spam_spec = importlib.util.find_spec("apex")
if spam_spec is not None:
    from apex import amp
#from sklearn.metrics.classification import precision_at_k_score
from .document_bert_architectures import DocumentBertLSTM, DocumentBertLinear, DocumentBertTransformer, DocumentBertMaxPool, DocumentBertMean, DocumentBertLSTMAtt, DocumentBertAtt

def move_to_device(model, device, num_gpus=None):
    """Moves a model to the specified device (cpu or gpu/s)
       and implements data parallelism when multiple gpus are specified.
    Args:
        model (Module): A PyTorch model
        device (torch.device): A PyTorch device
        num_gpus (int): The number of GPUs to be used. Defaults to None,
            all gpus are used.
    Returns:
        Module, DataParallel: A PyTorch Module or
            a DataParallel wrapper (when multiple gpus are used).
    """
    if isinstance(model, nn.DataParallel):
        model = model.module

    if not isinstance(device, torch.device):
        raise ValueError("device must be of type torch.device.")

    if device.type == "cuda":
        model.to(device)  # inplace
        if num_gpus == 0:
            raise ValueError("num_gpus must be non-zero when device.type is 'cuda'")
        elif num_gpus == 1:
            return model
        else:
            # parallelize
            num_cuda_devices = torch.cuda.device_count()
            if num_cuda_devices < 1:
                raise Exception("CUDA devices are not available.")
            elif num_cuda_devices < 2:
                print("Warning: Only 1 CUDA device is available. Data parallelism is not possible.")
                return model
            else:
                if num_gpus is None:
                    # use all available devices
                    return nn.DataParallel(model, device_ids=None)
                elif num_gpus > num_cuda_devices:
                    print(
                        "Warning: Only {0} devices are available. "
                        "Setting the number of gpus to {0}".format(num_cuda_devices)
                    )
                    return nn.DataParallel(model, device_ids=None)
                else:
                    return nn.DataParallel(model, device_ids=list(range(num_gpus)))
    elif device.type == "cpu":
        if num_gpus != 0 and num_gpus is not None:
            warnings.warn("Device type is 'cpu'. num_gpus is ignored.")
        return model.to(device)

    else:
        raise Exception(
            "Device type '{}' not supported. Currently, only cpu "
            "and cuda devices are supported.".format(device.type)
        )

def get_device(device="gpu"):
    """Gets a PyTorch device.
    Args:
        device (str, optional): Device string: "cpu" or "gpu". Defaults to "gpu".
    Returns:
        torch.device: A PyTorch device (cpu or gpu).
    """
    if device == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        raise Exception("CUDA device not available")
    elif device == "cpu":
        return torch.device("cpu")
    else:
        raise ValueError("Only 'cpu' and 'cuda' devices are supported.")

def encode_documents(documents: list, tokenizer: BertTokenizer, max_input_length=512):
    """
    Returns a len(documents) * max_sequences_per_document * 3 * 512 tensor where len(documents) is the batch
    dimension and the others encode bert input.

    This is the input to any of the document bert architectures.

    :param documents: a list of text documents
    :param tokenizer: the sentence piece bert tokenizer
    :return:
    """
    tokenized_documents = [tokenizer.tokenize(document) for document in documents]
    max_sequences_per_document = math.ceil(max(len(x)/(max_input_length-2) for x in tokenized_documents))
    assert max_sequences_per_document <= 20, "Your document is to large, arbitrary size when writing"

    output = torch.zeros(size=(len(documents), max_sequences_per_document, 3, max_input_length), dtype=torch.long)
    document_seq_lengths = [] #number of sequence generated per document
    #Need to use 510 to account for 2 padding tokens
    for doc_index, tokenized_document in enumerate(tokenized_documents):
        max_seq_index = 0
        for seq_index, i in enumerate(range(0, len(tokenized_document), (max_input_length-2))):
            raw_tokens = tokenized_document[i:i+(max_input_length-2)]
            tokens = []
            input_type_ids = []

            tokens.append("[CLS]")
            input_type_ids.append(0)
            for token in raw_tokens:
                tokens.append(token)
                input_type_ids.append(0)
            tokens.append("[SEP]")
            input_type_ids.append(0)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            attention_masks = [1] * len(input_ids)

            while len(input_ids) < max_input_length:
                input_ids.append(0)
                input_type_ids.append(0)
                attention_masks.append(0)

            assert len(input_ids) == max_input_length and len(attention_masks) == max_input_length and len(input_type_ids) == max_input_length

            #we are ready to rumble
            output[doc_index][seq_index] = torch.cat((torch.LongTensor(input_ids).unsqueeze(0),
                                                           torch.LongTensor(input_type_ids).unsqueeze(0),
                                                           torch.LongTensor(attention_masks).unsqueeze(0)),
                                                          dim=0)
            max_seq_index = seq_index
        document_seq_lengths.append(max_seq_index+1)
    return output, torch.LongTensor(document_seq_lengths)



def encode_documents_seq(documents: list, tokenizer: BertTokenizer, max_input_length=512, bert_batch_size=5 ):
    """
    Returns a len(documents) * max_sequences_per_document * 3 * 512 tensor where len(documents) is the batch
    dimension and the others encode bert input.

    This is the input to any of the document bert architectures.

    :param documents: a list of text documents
    :param tokenizer: the sentence piece bert tokenizer
    :return:
    """
    tokenized_documents = [tokenizer.tokenize(document) for document in documents]  
    #max_sequences_per_document = math.ceil(max(len(x)/(max_input_length-2) for x in tokenized_documents))
    new_tokenized_documents = []
    number_of_sen = []
    max_sen_len = []
    for document in tokenized_documents:
        i = (list(g) for _, g in groupby(document, key='.'.__ne__))
        sentences = [a + b for a, b in zip(i, i)]
        try:
            max_sen_len.append(len(max(sentences,key=len)))
        except:
            max_sen_len.append(0)
        number_of_sen.append(len(sentences))
        new_tokenized_documents.append(sentences)
    max_sequences_per_document = max(number_of_sen)
    assert max_sequences_per_document <= 110, "Your document is to large, arbitrary size when writing"

    output = torch.zeros(size=(len(documents), bert_batch_size, 3, max_input_length), dtype=torch.long)
    document_seq_lengths = [] #number of sequence generated per document
    #Need to use 510 to account for 2 padding tokens
    for doc_index, tokenized_document in enumerate(tokenized_documents):
        max_seq_index = 0
        for seq_index, i in enumerate(range(0, bert_batch_size)):
            try:
                raw_tokens = tokenized_document[i][:(max_input_length-2)]
            except:
                raw_tokens = []
            tokens = []
            input_type_ids = []

            tokens.append("[CLS]")
            input_type_ids.append(0)
            for token in raw_tokens:
                tokens.append(token)
                input_type_ids.append(0)
            tokens.append("[SEP]")
            input_type_ids.append(0)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            attention_masks = [1] * len(input_ids)

            while len(input_ids) < max_input_length:
                input_ids.append(0)
                input_type_ids.append(0)
                attention_masks.append(0)

            assert len(input_ids) == max_input_length and len(attention_masks) == max_input_length and len(input_type_ids) == max_input_length

            #we are ready to rumble
            output[doc_index][seq_index] = torch.cat((torch.LongTensor(input_ids).unsqueeze(0),
                                                           torch.LongTensor(input_type_ids).unsqueeze(0),
                                                           torch.LongTensor(attention_masks).unsqueeze(0)),
                                                          dim=0)
            max_seq_index = seq_index
        document_seq_lengths.append(max_seq_index+1)
    return output, torch.LongTensor(document_seq_lengths)






document_bert_architectures = {
    'DocumentBertLSTM': DocumentBertLSTM,
    'DocumentBertTransformer': DocumentBertTransformer,
    'DocumentBertLinear': DocumentBertLinear,
    'DocumentBertMaxPool': DocumentBertMaxPool,
    'DocumentBertMean': DocumentBertMean,
    'DocumentBertLSTMAtt': DocumentBertLSTMAtt,
    'DocumentBertAtt': DocumentBertAtt
}

class BertForDocumentClassification():
    def __init__(self,args=None,
                 labels=None,
                 device='cuda',
                 bert_model_path='bert-base-uncased',
                 architecture="DocumentBertLSTM",
                 batch_size=10,
                 bert_batch_size=7,
                 learning_rate = 5e-5,
                 weight_decay=0,
                 use_tensorboard=False):
        if args is not None:
            self.args = vars(args)
        if not args:
            self.args = {}
            self.args['bert_model_path'] = bert_model_path
            self.args['device'] = device
            self.args['learning_rate'] = learning_rate
            self.args['weight_decay'] = weight_decay
            self.args['batch_size'] = batch_size
            self.args['labels'] = labels
            self.args['bert_batch_size'] = bert_batch_size
            self.args['architecture'] = architecture
            self.args['use_tensorboard'] = use_tensorboard
        if 'fold' not in self.args:
            self.args['fold'] = 0

        assert self.args['labels'] is not None, "Must specify all labels in prediction"
        '''
        if args.tpu:
            if args.tpu_ip_address:
                os.environ["TPU_IP_ADDRESS"] = args.tpu_ip_address
            if args.tpu_name:
                os.environ["TPU_NAME"] = args.tpu_name
            if args.xrt_tpu_config:
                os.environ["XRT_TPU_CONFIG"] = args.xrt_tpu_config

            assert "TPU_IP_ADDRESS" in os.environ
            assert "TPU_NAME" in os.environ
            assert "XRT_TPU_CONFIG" in os.environ

            import torch_xla
            import torch_xla.core.xla_model as xm
            args['device'] = xm.xla_device()
            args['xla_model'] = xm
        '''
        self.log = logging.getLogger()
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.args['bert_model_path'])


        #account for some random tensorflow naming scheme
        if os.path.exists(self.args['bert_model_path']):
            if os.path.exists(os.path.join(self.args['bert_model_path'], CONFIG_NAME)):
                config = BertConfig.from_json_file(os.path.join(self.args['bert_model_path'], CONFIG_NAME))
            elif os.path.exists(os.path.join(self.args['bert_model_path'], 'bert_config.json')):
                config = BertConfig.from_json_file(os.path.join(self.args['bert_model_path'], 'bert_config.json'))
            else:
                raise ValueError("Cannot find a configuration for the BERT based model you are attempting to load.")
        else:
            config = BertConfig.from_pretrained(self.args['bert_model_path'])
        config.__setattr__('num_labels',len(self.args['labels']))
        config.__setattr__('bert_batch_size',self.args['bert_batch_size'])

        if 'use_tensorboard' in self.args and self.args['use_tensorboard']:
            assert 'model_directory' in self.args is not None, "Must have a logging and checkpoint directory set."
            from torch.utils.tensorboard import SummaryWriter
            self.tensorboard_writer = SummaryWriter(os.path.join(self.args['model_directory'],
                                                                 "..",
                                                                 "runs",
                                                                 self.args['model_directory'].split(os.path.sep)[-1]+'_'+self.args['architecture']+'_'+str(self.args['fold'])))


        self.bert_doc_classification = document_bert_architectures[self.args['architecture']].from_pretrained(self.args['bert_model_path'], config=config)
        self.optimizer = AdamW(
            self.bert_doc_classification.parameters(),
            weight_decay=self.args['weight_decay'],
            lr=self.args['learning_rate']
        )

    def fit(self, train, dev):
        """
        A list of
        :param documents: a list of documents
        :param labels: a list of label vectors
        :return:
        """

        train_documents, train_labels = train
        dev_documents, dev_labels = dev

        self.bert_doc_classification.train()
        if 'sequence_wise' in self.args and self.args['sequence_wise']:
            document_representations, _  = encode_documents_seq(train_documents, self.bert_tokenizer, self.args['max_len_size'], self.args['bert_batch_size'])
        else:
            document_representations, _  = encode_documents(train_documents, self.bert_tokenizer, self.args['max_len_size'])

        correct_output = torch.FloatTensor(train_labels)
        
        binary_output = torch.where(correct_output > 0, torch.ones(correct_output.shape),torch.zeros(correct_output.shape)) 
        loss_weight = ((binary_output.shape[0] / torch.sum(binary_output, dim=0))-self.args['loss_bias']).to(device=self.args['device'])
        self.loss_function = torch.nn.BCEWithLogitsLoss(reduction='none',pos_weight=loss_weight)
        self.log.info('Loss weight: %f' % (loss_weight))

        assert document_representations.shape[0] == correct_output.shape[0]

        '''if torch.cuda.device_count() > 1:
            self.bert_doc_classification = torch.nn.DataParallel(self.bert_doc_classification)
        self.bert_doc_classification.to(device=self.args['device'])'''

        self.bert_doc_classification = move_to_device(self.bert_doc_classification, get_device(self.args['device']))
        if spam_spec is not None:
            self.bert_doc_classification, self.optimizer = amp.initialize(self.bert_doc_classification, self.optimizer, opt_level="O1")

        self.log.info('Training on %s GPUS' % (torch.cuda.device_count()))
        self.log.info('Training starting')
        for epoch in tqdm(range(1,self.args['epochs']+1)):
            # shuffle
            permutation = torch.randperm(document_representations.shape[0])
            document_representations = document_representations[permutation]
            correct_output = correct_output[permutation]
            binary_output = binary_output[permutation]
            self.epoch = epoch
            epoch_loss = 0.0
            for i in tqdm(range(0, document_representations.shape[0], self.args['batch_size'])):

                batch_document_tensors = document_representations[i:i + self.args['batch_size']].to(device=self.args['device'])
                #self.log.info(batch_document_tensors.shape)
                self.optimizer.zero_grad()
                batch_predictions = self.bert_doc_classification(batch_document_tensors,
                                                                 freeze_bert=self.args['freeze_bert'], device=self.args['device'])

                batch_correct_output = correct_output[i:i + self.args['batch_size']].to(device=self.args['device'])
                batch_binary_output = binary_output[i:i + self.args['batch_size']].to(device=self.args['device']) 
                loss = self.loss_function(batch_predictions, batch_binary_output)
                #loss = loss * torch.where((batch_correct_output == 2) | (batch_correct_output == -1), 2*torch.ones(batch_correct_output.shape, device=self.args['device']), torch.ones(batch_correct_output.shape, device=self.args['device']))
                loss = loss.mean()
                epoch_loss += float(loss.item())
                #self.log.info(batch_predictions)
                if spam_spec is not None:
                    with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                self.optimizer.step()

            epoch_loss /= int(document_representations.shape[0] / self.args['batch_size'])  # divide by number of batches per epoch

            if 'use_tensorboard' in self.args and self.args['use_tensorboard']:
                self.tensorboard_writer.add_scalar('Loss/Train', epoch_loss, self.epoch)

            self.log.info('Epoch %i Completed: %f' % (epoch, epoch_loss))

            if epoch % self.args['checkpoint_interval'] == 0:
                self.save_checkpoint(os.path.join(self.args['model_directory'], "checkpoint_%s" % epoch))

            # evaluate on development data
            if epoch % self.args['evaluation_interval'] == 0:
                self.predict((dev_documents, dev_labels))
                #self.predict((train_documents, train_labels))
        del [document_representations, correct_output, binary_output]
        torch.cuda.empty_cache()
    def predict(self, data, threshold=0.5):
        """
        A tuple containing
        :param data:
        :return:
        """
        document_representations = None
        correct_output = None
        if isinstance(data, list):
            if 'sequence_wise' in self.args and self.args['sequence_wise']:
                document_representations, _  = encode_documents_seq(data, self.bert_tokenizer, self.args['max_len_size'], self.args['bert_batch_size'])
            else:
                document_representations, _  = encode_documents(data, self.bert_tokenizer, self.args['max_len_size'])

        if isinstance(data, tuple) and len(data) == 2:
            self.log.info('Evaluating on Epoch %i' % (self.epoch))
            if 'sequence_wise' in self.args and self.args['sequence_wise']:
                document_representations, _  = encode_documents_seq(data[0], self.bert_tokenizer, self.args['max_len_size'], self.args['bert_batch_size'])
            else:
                document_representations, _  = encode_documents(data[0], self.bert_tokenizer, self.args['max_len_size'])
            correct_output = torch.FloatTensor(data[1]).transpose(0,1)
            correct_output = torch.where(correct_output > 0, torch.ones(correct_output.shape), torch.zeros(correct_output.shape))
            assert self.args['labels'] is not None

        self.bert_doc_classification.to(device=self.args['device'])
        self.bert_doc_classification.eval()
        with torch.no_grad():
            predictions = torch.empty((document_representations.shape[0], len(self.args['labels'])))
            for i in range(0, document_representations.shape[0], self.args['batch_size']):
                batch_document_tensors = document_representations[i:i + self.args['batch_size']].to(device=self.args['device'])

                prediction = self.bert_doc_classification(batch_document_tensors, device=self.args['device'])
                predictions[i:i + self.args['batch_size']] = prediction
        
        sigmoid = nn.Sigmoid()
        predictions = sigmoid(predictions)
        predictions_cont = predictions.transpose(0, 1).clone()
        #logging.info(str(predictions_cont[:50]))
        for r in range(0, predictions.shape[0]):
            for c in range(0, predictions.shape[1]):
                if predictions[r][c] > threshold:
                    predictions[r][c] = 1
                else:
                    predictions[r][c] = 0
        predictions = predictions.transpose(0, 1)

        if correct_output is None:
            return predictions.cpu() ,predictions_cont.cpu()
        else:
            assert correct_output.shape == predictions.shape
            precisions = []
            recalls = []
            fmeasures = []
            aprecisions = []
            precisionk = []
            for label_idx in range(predictions.shape[0]):
                correct = correct_output[label_idx].cpu().view(-1).numpy()
                predicted = predictions[label_idx].cpu().view(-1).numpy()
                predicted_cont = predictions_cont[label_idx].cpu().view(-1).numpy()
                sorted_predict = np.argsort(-predicted_cont)[:30]
                present_f1_score = f1_score(correct, predicted, average='binary', pos_label=1)
                present_precision_score = precision_score(correct, predicted, average='binary', pos_label=1)
                present_recall_score = recall_score(correct, predicted, average='binary', pos_label=1)
                present_average_precision_score = average_precision_score(correct, predicted)
                try:
                    present_precisionk_score = precision_score(correct[sorted_predict], predicted[sorted_predict],average='binary', pos_label=1)
                except:
                    # logging.info('Not enough samples for Precision@30')
                    # logging.info(str(sum(predicted)))
                    # logging.info(str(len(predicted)))
                    # logging.info(str(predicted_cont[:50]))
                    present_precisionk_score = 0.0
                precisions.append(present_precision_score)
                recalls.append(present_recall_score)
                fmeasures.append(present_f1_score)
                aprecisions.append(present_average_precision_score)
                precisionk.append(present_precisionk_score)
                #logging.info('F1\t%s\t%f' % (self.args['labels'][label_idx], present_f1_score))
                logging.info('Precision-at-30\t%s\t%f' % (self.args['labels'][label_idx], present_precisionk_score))

            micro_f1 = f1_score(correct_output.reshape(-1).numpy(), predictions.reshape(-1).numpy(), average='micro')
            macro_f1 = f1_score(correct_output.reshape(-1).numpy(), predictions.reshape(-1).numpy(), average='macro')

            if 'use_tensorboard' in self.args and self.args['use_tensorboard']:
                for label_idx in range(predictions.shape[0]):
                    self.tensorboard_writer.add_scalar('Precision/%s/Test' % self.args['labels'][label_idx].replace(" ", "_"), precisions[label_idx], self.epoch)
                    self.tensorboard_writer.add_scalar('Recall/%s/Test' % self.args['labels'][label_idx].replace(" ", "_"), recalls[label_idx], self.epoch)
                    self.tensorboard_writer.add_scalar('F1/%s/Test' % self.args['labels'][label_idx].replace(" ", "_"), fmeasures[label_idx], self.epoch)
                    self.tensorboard_writer.add_scalar('Average-Precision/%s/Test' % self.args['labels'][label_idx].replace(" ", "_"), aprecisions[label_idx], self.epoch)
                    self.tensorboard_writer.add_scalar('Precision-at-30/%s/Test' % self.args['labels'][label_idx].replace(" ", "_"), precisionk[label_idx], self.epoch)
                self.tensorboard_writer.add_scalar('Micro-F1/Test', micro_f1, self.epoch)
                self.tensorboard_writer.add_scalar('Macro-F1/Test', macro_f1, self.epoch)

            with open(os.path.join(self.args['model_directory'], "eval_%s.csv" % self.epoch), 'w') as eval_results:
                eval_results.write('Metric\t' + '\t'.join([self.args['labels'][label_idx] for label_idx in range(predictions.shape[0])]) +'\n' )
                eval_results.write('Precision\t' + '\t'.join([str(precisions[label_idx]) for label_idx in range(predictions.shape[0])]) + '\n' )
                eval_results.write('Recall\t' + '\t'.join([str(recalls[label_idx]) for label_idx in range(predictions.shape[0])]) + '\n' )
                eval_results.write('F1\t' + '\t'.join([ str(fmeasures[label_idx]) for label_idx in range(predictions.shape[0])]) + '\n' )
                eval_results.write('Average-Precision\t' + '\t'.join([ str(aprecisions[label_idx]) for label_idx in range(predictions.shape[0])]) + '\n' )
                eval_results.write('Precision-at-30\t' + '\t'.join([ str(precisionk[label_idx]) for label_idx in range(predictions.shape[0])]) + '\n' )
                eval_results.write('Micro-F1\t' + str(micro_f1) + '\n' )
                eval_results.write('Macro-F1\t' + str(macro_f1) + '\n' )

        self.bert_doc_classification.train()
        return predictions.cpu() ,predictions_cont.cpu(), precisions, recalls, fmeasures, aprecisions, precisionk

    def save_checkpoint(self, checkpoint_path: str):
        """
        Saves an instance of the current model to the specified path.
        :return:
        """
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
        else:
            raise ValueError("Attempting to save checkpoint to an existing directory")
        self.log.info("Saving checkpoint: %s" % checkpoint_path )

        #save finetune parameters
        net = self.bert_doc_classification
        if isinstance(self.bert_doc_classification, nn.DataParallel):
            net = self.bert_doc_classification.module
        torch.save(net.state_dict(), os.path.join(checkpoint_path, WEIGHTS_NAME))
        #save configurations
        net.config.to_json_file(os.path.join(checkpoint_path, CONFIG_NAME))
        #save exact vocabulary utilized
        self.bert_tokenizer.save_vocabulary(checkpoint_path)


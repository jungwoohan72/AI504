import torch.nn as nn
import torch.optim as optim
from easydict import EasyDict
import os
# import time

import torch
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator

import pprint

from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm
import json

import wandb
# os.environ["CUDA_VISIBLE_DEVICES"]="6"

class Transformer(nn.Module):
    def __init__(self, config, SRC, TRG):
        super(Transformer,self).__init__()
        self.encoder_embedding = nn.Embedding(len(SRC.vocab),config.emb_dim)
        self.decoder_embedding = nn.Embedding(len(TRG.vocab),config.emb_dim)
        self.transformer = nn.Transformer(d_model=config.emb_dim, nhead=config.attention_heads,
                                          num_encoder_layers=config.encoder_layers, num_decoder_layers=config.decoder_layers,
                                          dim_feedforward=config.ffn_dim, dropout=config.dropout, activation='gelu')
        self.prediction_head = nn.Linear(config.emb_dim,len(TRG.vocab))

    def forward(self, src, trg, PAD_IDX, device):
        src_emb = self.encoder_embedding(src)
        trg_emb = self.decoder_embedding(trg)
        output = self.transformer(src_emb, trg_emb,
                                  tgt_mask=self.transformer.generate_square_subsequent_mask(trg.size(0)).to(device),
                                  src_key_padding_mask=src.eq(PAD_IDX).permute(1,0).to(device),
                                  memory_key_padding_mask=src.eq(PAD_IDX).permute(1,0).to(device),
                                  tgt_key_padding_mask=trg.eq(PAD_IDX).permute(1,0).to(device))
        prediction = self.prediction_head(output)
        return prediction

def train(model: nn.Module,
          iterator: BucketIterator,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          clip: float,
          PAD_IDX,
          device):
    model.train()

    epoch_loss = 0

    for idx, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output = model(src, trg, PAD_IDX, device)

        output = output[:-1].reshape(-1, output.shape[-1])
        trg = trg[1:].reshape(-1)

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model: nn.Module,
             iterator: BucketIterator,
             criterion: nn.Module,
             PAD_IDX,
             device):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for _, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg, PAD_IDX, device)

            output = output[:-1].reshape(-1, output.shape[-1])

            trg = trg[1:].reshape(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def measure_BLEU(model: nn.Module,
                 iterator: BucketIterator,
                 TRG,
                 PAD_IDX,
                 device
                 ):
    model.eval()
    iterator.batch_size = 1
    BLEU_scores = list()

    with torch.no_grad():
        for idx, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg, PAD_IDX, device)
            predicted = [TRG.vocab.itos[token] for token in output[:-1].argmax(dim=2).squeeze().tolist() if token!=PAD_IDX]
            GT = [TRG.vocab.itos[token] for token in trg[1:].squeeze().tolist() if token!=PAD_IDX]
            BLEU_scores.append(sentence_bleu([GT], predicted))
    return sum(BLEU_scores)/len(BLEU_scores)

def train_sweep(config = None):

    with wandb.init(config = config):
        config = wandb.config
        ## Data Load
        SRC = Field(tokenize = "spacy",
                    tokenizer_language="de_core_news_sm",
                    eos_token = '<eos>',
                    lower = True)

        TRG = Field(tokenize = "spacy",
                    tokenizer_language="en_core_web_sm",
                    init_token = '<sos>',
                    eos_token = '<eos>',
                    lower = True)

        train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'),
                                                            fields = (SRC, TRG))

        SRC.build_vocab(train_data, min_freq = 3)
        TRG.build_vocab(train_data, min_freq = 3)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
            (train_data, valid_data, test_data),
            batch_size = config.batch_size,
            device = device,
            shuffle=False)

        PAD_IDX = TRG.vocab.stoi['<pad>']

        ## Model
        CLIP = 1 # For gradient clipping

        model = Transformer(config, SRC, TRG)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

        ## Training
        best_valid_loss = float('inf')

        patience=0

        for epoch in tqdm(range(config.nepochs), total=config.nepochs):
            train_loss = train(model, train_iterator, optimizer, criterion, CLIP, PAD_IDX, device)
            valid_loss = evaluate(model, valid_iterator, criterion, PAD_IDX, device)
            test_bleu = measure_BLEU(model, test_iterator, TRG, PAD_IDX, device)
            # print("Test BLEU score : {}".format(test_bleu * 100))
            # print("Epoch : {} / Training loss : {} / Validation loss : {}".format(epoch+1, train_loss, valid_loss))

            wandb.log({"val_loss": valid_loss, "epoch": epoch})
            wandb.log({"BLEU": test_bleu*100, "epoch": epoch})

            # Early stopping
            # You can change early stop criterion
            if best_valid_loss < valid_loss:
                patience += 1
                if patience > config.patience:
                    break
            else:
                best_valid_loss = valid_loss
                patience = 0

# In the project, You need to change the below hyperparameters.
# config = EasyDict({
#     "emb_dim":64,
#     "ffn_dim":128,
#     "attention_heads":8,
#     "dropout":0.2518,
#     "encoder_layers":3,
#     "decoder_layers":3,
#     "lr":0.0007894,
#     "batch_size":461,
#     "nepochs":100,
#     "patience":10,
# })

sweep_config = {
    'method': 'grid'
}

parameters_dict = {
    "emb_dim": {
        'values': [64,128,256,512]
        },
    "ffn_dim": {
        'values': [128,256,512]
        },
    "attention_heads": {
        'values': [8,10,12,14,16]
        },
    "dropout": {
        'values': [i*0.01 for i in range(20,50,10)]
        },
    "encoder_layers": {
        'values': [3,4,5,6,7,8]
        },
    "decoder_layers": {
        'values': [3,4,5,6,7,8]
        },
    "lr": {
        'values': [i*0.0001 for i in range(1,11,1)]
        },
    "batch_size": {
        'values': [i*64 for i in range(6,10,1)]
        },
    "nepochs": {
        'values': [70]
        },
    "patience": {
        'values': [10]
        }
    }

sweep_config['parameters'] = parameters_dict

pprint.pprint(sweep_config)

torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

sweep_id = wandb.sweep(sweep_config, project="NMT")

wandb.agent(sweep_id, train_sweep)

# ## Model Test
# test_bleu = measure_BLEU(model, test_iterator)
# print("Test BLEU score : {}".format(test_bleu * 100))
#
# ## Save Model
# with open('config.json','w') as f:
#     json.dump(vars(config),f)
# torch.save(model.state_dict(),'model.pt')
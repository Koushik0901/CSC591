import os
import glob
import re
import random
import itertools
import math
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer

def seed_everything(seed=42):
    """
    Seeds basic parameters for reproducibility of results.

    :param seed: The seed for random number generators. Default is 42.
    """
    random.seed(seed)        # Python random library seeding
    np.random.seed(seed)     # Numpy library seeding
    torch.manual_seed(seed)  # PyTorch random number generation
    torch.cuda.manual_seed(seed)  # Seed all GPUs with the same seed if available
    torch.cuda.manual_seed_all(seed)  # For multiGPU setups
    torch.backends.cudnn.deterministic = True  # Ensures consistent results on the backend
    torch.backends.cudnn.benchmark = False  # False makes things slower but more reproducible

    print(f"Everything seeded with seed {seed}.")

seed_everything(42)

class Config:
    def __init__(self):
        self.hidden_size = 256
        self.num_attention_heads = 8
        self.num_hidden_layers = 3
        self.intermediate_size = 512
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.max_position_embeddings = 512
        self.vocab_size = 30_000
        self.type_vocab_size = 3
        self.initializer_range = 0.02
        self.max_len = 64
        self.num_epochs = 200
        self.batch_size = 512
config = Config()

### loading all data into memory
movie_convos = '/content/cornell-moviedialog-corpus/movie_conversations.txt'
movie_lines = '/content/cornell-moviedialog-corpus/movie_lines.txt'
with open(movie_convos, 'r', encoding='iso-8859-1') as c:
    convos = c.readlines()
with open(movie_lines, 'r', encoding='iso-8859-1') as l:
    lines = l.readlines()

### splitting text using special lines
lines_dict = {}
for line in lines:
    objects = line.split(" +++$+++ ")
    lines_dict[objects[0]] = objects[-1]

### generate question answer pairs
pairs = []
for con in tqdm(convos):
    ids = eval(con.split(" +++$+++ ")[-1])
    for i in range(len(ids)):
        qa_pairs = []

        if i == len(ids) - 1:
            break

        first = lines_dict[ids[i]].strip()
        second = lines_dict[ids[i+1]].strip()

        qa_pairs.append(' '.join(first.split()[:config.max_len]))
        qa_pairs.append(' '.join(second.split()[:config.max_len]))
        pairs.append(qa_pairs)

pairs[:5]

# WordPiece tokenizer
### save data as txt file
os.makedirs('./data', exist_ok=True)
text_data = []
file_count = 0

for sample in tqdm([x[0] for x in pairs]):
    text_data.append(sample)

    # once we hit the 10K mark, save to file
    if len(text_data) == 10000:
        with open(f'./data/text_{file_count}.txt', 'w', encoding='utf-8') as fp:
            fp.write('\n'.join(text_data))
        text_data = []
        file_count += 1

paths = [str(x) for x in glob.glob('./data/*.txt')]

### training own tokenizer
tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=False,
    strip_accents=True,
    lowercase=True
)

tokenizer.train(
    files=paths,
    vocab_size=30_000,
    min_frequency=5,
    limit_alphabet=1000,
    wordpieces_prefix='##',
    special_tokens=['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']
    )

os.makedirs('./bert-tokenizer', exist_ok=True)
tokenizer.save_model('./bert-tokenizer', 'bert-tokenizer')
tokenizer = BertTokenizer.from_pretrained(
    './bert-tokenizer/bert-tokenizer-vocab.txt',
    local_files_only=True
)

config.vocab_size = tokenizer.vocab_size
config.vocab_size

class BERTDataset(Dataset):
    def __init__(self, data_pair, tokenizer, seq_len=64):

        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.lines = data_pair

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, item):

        # Step 1: get random sentence pair, either negative or positive (saved as is_next_label)
        t1, t2, is_next_label = self.get_sent(item)

        # Step 2: replace random words in sentence with mask / random words
        t1_random, t1_label = self.random_word(t1)
        t2_random, t2_label = self.random_word(t2)

        # Step 3: Adding CLS and SEP tokens to the start and end of sentences
         # Adding PAD token for labels
        t1 = [self.tokenizer.vocab['[CLS]']] + t1_random + [self.tokenizer.vocab['[SEP]']]
        t2 = t2_random + [self.tokenizer.vocab['[SEP]']]
        t1_label = [self.tokenizer.vocab['[PAD]']] + t1_label + [self.tokenizer.vocab['[PAD]']]
        t2_label = t2_label + [self.tokenizer.vocab['[PAD]']]

        # Step 4: combine sentence 1 and 2 as one input
        # adding PAD tokens to make the sentence same length as seq_len
        segment_label = ([1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]
        bert_input = (t1 + t2)[:self.seq_len]
        bert_label = (t1_label + t2_label)[:self.seq_len]
        padding = [self.tokenizer.vocab['[PAD]'] for _ in range(self.seq_len - len(bert_input))]
        bert_input.extend(padding), bert_label.extend(padding), segment_label.extend(padding)

        output = {"bert_input": bert_input,
                  "bert_label": bert_label,
                  "segment_label": segment_label,
                  "is_next": is_next_label}

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, sentence):
        tokens = sentence.split()
        output_label = []
        output = []

        # 15% of the tokens would be replaced
        for i, token in enumerate(tokens):
            prob = random.random()

            # remove cls and sep token
            token_id = self.tokenizer(token)['input_ids'][1:-1]

            if prob < 0.15:
                prob /= 0.15

                # 80% chance change token to mask token
                if prob < 0.8:
                    for i in range(len(token_id)):
                        output.append(self.tokenizer.vocab['[MASK]'])

                # 10% chance change token to random token
                elif prob < 0.9:
                    for i in range(len(token_id)):
                        output.append(random.randrange(len(self.tokenizer.vocab)))

                # 10% chance change token to current token
                else:
                    output.append(token_id)

                output_label.append(token_id)

            else:
                output.append(token_id)
                for i in range(len(token_id)):
                    output_label.append(0)

        # flattening
        output = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in output]))
        output_label = list(itertools.chain(*[[x] if not isinstance(x, list) else x for x in output_label]))
        assert len(output) == len(output_label)
        return output, output_label

    def get_sent(self, index):
        '''return random sentence pair'''
        t1, t2 = self.get_corpus_line(index)

        # negative or positive pair, for next sentence prediction
        if random.random() > 0.5:
            return t1, t2, 1
        else:
            return t1, self.get_random_line(), 0

    def get_corpus_line(self, item):
        '''return sentence pair'''
        return self.lines[item][0], self.lines[item][1]

    def get_random_line(self):
        '''return random single sentence'''
        return self.lines[random.randrange(len(self.lines))][1]

train_len = round(len(pairs) * 0.8)
print(train_len)
random.shuffle(pairs)
train_pairs = pairs[:train_len]
val_pairs = pairs[train_len:]

train_dataset = BERTDataset(train_pairs, tokenizer)
val_dataset = BERTDataset(val_pairs, tokenizer)

train_loader = DataLoader(
    train_dataset,
    batch_size = config.batch_size,
    shuffle=True,
    num_workers = 2,
    pin_memory = True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size = config.batch_size,
    shuffle = False,
    num_workers = 2,
    pin_memory = True,
)

# Embeddings
class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.layerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Attention score calculation
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        attention_output = torch.matmul(attention_probs, value_layer)
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
        new_attention_output_shape = attention_output.size()[:-2] + (self.all_head_size,)
        attention_output = attention_output.view(*new_attention_output_shape)

        return attention_output

# Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.layer_norm1(attention_output + hidden_states)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        layer_output = self.layer_norm2(layer_output + attention_output)
        return layer_output

# BERT Model
class BertModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
#         print("Unique token types:", torch.unique(token_type_ids))

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Apply mask to attention scores
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = (1.0 - attention_mask) * -10000.0

        x = self.embeddings(input_ids, token_type_ids)
        for layer_module in self.encoder:
            x = layer_module(x, attention_mask)

        return x

class NextSentencePrediction(torch.nn.Module):
    """
    2-class classification model : is_next, is_not_next
    """

    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = torch.nn.Linear(hidden, 2)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # use only the first token which is the [CLS]
        return self.softmax(self.linear(x[:, 0]))

class MaskedLanguageModel(torch.nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = torch.nn.Linear(hidden, vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))

class BERTLM(torch.nn.Module):
    """
    BERT Language Model
    Next Sentence Prediction Model + Masked Language Model
    """

    def __init__(self, bert, config):
        """
        :param bert: BERT model which should be trained
        :param vocab_size: total vocab size for masked_lm
        """

        super().__init__()
        self.bert = bert
#         self.next_sentence = NextSentencePrediction(config.hidden_size)
        self.mask_lm = MaskedLanguageModel(config.hidden_size, config.vocab_size)

    def forward(self, x, segment_label):
        x = self.bert(x, segment_label)
        return self.mask_lm(x) #self.next_sentence(x),

class BertScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super(BertScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Calculate learning rate using linear warmup and decay."""
        if self.last_epoch < self.warmup_steps:
            # Warmup phase: scale lr linearly with the number of steps
            lr_scale = self.last_epoch / self.warmup_steps
        else:
            # Decay phase: decrease lr linearly after warmup
            lr_scale = max(0.0, float(self.total_steps - self.last_epoch) / float(max(1.0, self.total_steps - self.warmup_steps)))

        return [base_lr * lr_scale for base_lr in self.base_lrs]

from sklearn.metrics import f1_score, accuracy_score
from torch.utils.tensorboard import SummaryWriter

def calculate_metrics(preds, labels, average_type='macro'):
    preds = preds.argmax(dim=1)

    valid_indices = labels != tokenizer.vocab['[PAD]']
    preds = preds[valid_indices].cpu().numpy()
    labels = labels[valid_indices].cpu().numpy()

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average=average_type, zero_division=0)
    return acc, f1

def train_loop(model, dataloader, criterion, optimizer, scheduler, device, epoch, writer):
    model.train()
    total_loss = 0
    total_mlm_acc, total_nsp_acc = 0, 0
    total_mlm_f1, total_nsp_f1 = 0, 0
    num_batches = len(dataloader)

    for batch_idx, data in enumerate(tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch}")):
        data = {key: value.to(device) for key, value in data.items()}
        mlm_output = model(data["bert_input"].to(device), data["segment_label"].to(device))

#         loss = criterion(nsp_output, data["is_next"].to(device))
        loss = criterion(mlm_output.transpose(1, 2), data["bert_label"].to(device))
#         loss = nsp_loss + mlm_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        # Calculate metrics
        mlm_acc, mlm_f1 = calculate_metrics(mlm_output.transpose(1, 2), data["bert_label"], average_type='macro')
#         nsp_acc, nsp_f1 = calculate_metrics(nsp_output, data["is_next"], average_type='binary')

        total_mlm_acc += mlm_acc
#         total_nsp_acc += nsp_acc
        total_mlm_f1 += mlm_f1
#         total_nsp_f1 += nsp_f1

        # Log metrics and loss for each batch
        writer.add_scalar('Loss/train', loss.item(), epoch * num_batches + batch_idx)
        writer.add_scalar('Accuracy/MLM_train', mlm_acc, epoch * num_batches + batch_idx)
        writer.add_scalar('F1_Score/MLM_train', mlm_f1, epoch * num_batches + batch_idx)
#         writer.add_scalar('Accuracy/NSP_train', nsp_acc, epoch * num_batches + batch_idx)
#         writer.add_scalar('F1_Score/NSP_train', nsp_f1, epoch * num_batches + batch_idx)

    avg_loss = total_loss / num_batches
    avg_mlm_acc = total_mlm_acc / num_batches
#     avg_nsp_acc = total_nsp_acc / num_batches
    avg_mlm_f1 = total_mlm_f1 / num_batches
#     avg_nsp_f1 = total_nsp_f1 / num_batches
    print(f"Training - Loss: {avg_loss:.4f}, MLM Acc: {avg_mlm_acc:.4f}, MLM F1: {avg_mlm_f1:.4f}")

def validation_loop(model, dataloader, criterion, device, epoch, writer):
    model.eval()
    total_loss = 0
    total_mlm_acc, total_nsp_acc = 0, 0
    total_mlm_f1, total_nsp_f1 = 0, 0
    num_batches = len(dataloader)

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(dataloader, total=len(dataloader), desc=f"Validation")):
            data = {key: value.to(device) for key, value in data.items()}
            mlm_output = model(data["bert_input"].to(device), data["segment_label"].to(device))

#             nsp_loss = criterion(nsp_output, data["is_next"].to(device))
            loss = criterion(mlm_output.transpose(1, 2), data["bert_label"].to(device))
#             loss = nsp_loss + mlm_loss

            total_loss += loss.item()

            # Calculate metrics
            mlm_acc, mlm_f1 = calculate_metrics(mlm_output.transpose(1, 2), data["bert_label"], average_type='macro')
#             nsp_acc, nsp_f1 = calculate_metrics(nsp_output, data["is_next"], average_type='binary')

            total_mlm_acc += mlm_acc
#             total_nsp_acc += nsp_acc
            total_mlm_f1 += mlm_f1
#             total_nsp_f1 += nsp_f1

            # Log metrics and loss for each batch
            writer.add_scalar('Loss/validation', loss.item(), epoch * num_batches + batch_idx)
            writer.add_scalar('Accuracy/MLM_validation', mlm_acc, epoch * num_batches + batch_idx)
            writer.add_scalar('F1_Score/MLM_validation', mlm_f1, epoch * num_batches + batch_idx)
#             writer.add_scalar('Accuracy/NSP_validation', nsp_acc, epoch * num_batches + batch_idx)
#             writer.add_scalar('F1_Score/NSP_validation', nsp_f1, epoch * num_batches + batch_idx)

    avg_loss = total_loss / num_batches
    avg_mlm_acc = total_mlm_acc / num_batches
#     avg_nsp_acc = total_nsp_acc / num_batches
    avg_mlm_f1 = total_mlm_f1 / num_batches
#     avg_nsp_f1 = total_nsp_f1 / num_batches
    print(f"Validation - Loss: {avg_loss:.4f}, MLM Acc: {avg_mlm_acc:.4f}, MLM F1: {avg_mlm_f1:.4f}")
    return avg_mlm_f1

def save_checkpoint(model, optimizer, scheduler, epoch, filepath):
    """
    Saves a model checkpoint.

    :param model: The model whose parameters are to be saved.
    :param optimizer: The optimizer state to save.
    :param scheduler: The learning rate scheduler state to save.
    :param epoch: Current epoch number.
    :param filepath: Path to save the checkpoint.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(filepath, model, optimizer, scheduler, device):
    """
    Loads a model checkpoint.

    :param filepath: Path to the checkpoint file.
    :param model: The model to load the parameters into.
    :param optimizer: The optimizer to load the state into.
    :param scheduler: The scheduler to load the state into.
    :param device: The device to load the checkpoint onto.
    :returns: The epoch number of the loaded checkpoint.
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from {filepath}, epoch {epoch}")
    return epoch

os.makedirs("./checkpoints", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
writer = SummaryWriter('runs/bert_pretraining_experiment')

bert_model = BertModel(config).to(device)

bert_lm = BERTLM(bert_model, config).to(device)
optimizer = torch.optim.Adam(bert_lm.parameters(), lr=1e-3)
scheduler = BertScheduler(optimizer, warmup_steps=1000, total_steps=10000)
criterion = torch.nn.NLLLoss(ignore_index=0)


best_f1 = 0
for epoch in range(config.num_epochs):
    train_loop(bert_lm, train_loader, criterion, optimizer, scheduler, device, epoch, writer)
    mean_f1 = validation_loop(bert_lm, val_loader, criterion, device, epoch, writer)
    if mean_f1 > best_f1:
        best_f1 = mean_f1
        save_checkpoint(
            bert_lm,
            optimizer,
            scheduler,
            epoch,
            f"./checkpoints/bert_pretraining_epoch={epoch}_f1={best_f1:.2f}.pt"
        )

writer.close()
















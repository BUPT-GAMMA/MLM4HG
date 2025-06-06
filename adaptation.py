import torch
import pickle
from tqdm import tqdm
from LM import LM_Model
from collections import defaultdict
from sklearn.metrics import f1_score
from torch.nn import CrossEntropyLoss
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb
wandb.init(mode="disabled")


def open_pkl_file(file_path):
    with open(file_path, 'rb') as file:
        file_content = pickle.load(file)
        return file_content


def save_pkl_file(file_path, contents):
    with open(file_path, 'wb') as file:
        pickle.dump(contents, file)
    print("having saved pkl...")


def open_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        contents = [line.rstrip("\n") for line in file.readlines()]
        return contents


def save_txt_file(file_path, contents):
    with open(file_path, 'w') as file:
        for paragraph in contents:
            file.write(paragraph + "\n")
    print("having saved txt...")


def build_LM_model(model_config):

    LM_model_name = model_config['lm_model'].lower()

    # build LM_model
    LM_model = LM_Model(model_config).to(model_config['device'])

    # bulid tokenizer
    LM_tokenizer = AutoTokenizer.from_pretrained(model_config['card'])

    print('Information about LM model:')
    print('total params:', sum(p.numel() for p in LM_model.parameters()))
    return LM_model, LM_tokenizer


def compute_valid_lengths(data, pad_id):
    lengths = []
    for item in data:
        valid_length = len([x for x in item if x != pad_id])
        lengths.append(valid_length)
    return lengths


def batch_to_tensor(tokenizer, batch, max_length, device):
    valid_batch, valid_labels = [], []

    mask_token = tokenizer.mask_token
    batch[0] = [sentence.replace('<mask>', mask_token) for sentence in batch[0]]

    encoding = tokenizer.batch_encode_plus(batch[0], padding='longest', truncation=True, return_tensors='pt')

    valid_lengths = compute_valid_lengths(encoding['input_ids'], tokenizer.pad_token_id)
    
    valid_indices = [i for i, length in enumerate(valid_lengths) if length <= max_length-10]
    valid_batch = [batch[0][i] for i in valid_indices]
    valid_labels = [batch[1][i] for i in valid_indices]
    
    if not valid_batch:
        return None, None
    
    tokenized_tensors = tokenizer(valid_batch, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length', add_special_tokens=False)
    for key in tokenized_tensors.keys():
        tokenized_tensors[key] = tokenized_tensors[key].to(device)
    return tokenized_tensors, valid_labels


def get_optimizer(parameters, optimizer_args):
    optimizer = torch.optim.AdamW(parameters, **optimizer_args)
    return optimizer


def get_scheduler(optimizer, train_steps):
    return CosineAnnealingLR(optimizer, T_max=train_steps, eta_min=0)


class LM_dataset(Dataset):
    def __init__(self, data_token: torch.Tensor, labels: torch.Tensor):
        super().__init__()
        self.data_token = data_token
        self.labels = labels
        
    def __getitem__(self, index):
        token_list = self.data_token[index]
        label = self.labels[index]
        return token_list, label

    def __len__(self):
        return len(self.data_token)
    

def build_LM_dataloader(batch_size, idx, user_seq, labels, mode):
    if mode == 'train':
        user_text = []
        train_labels = []
        for i in idx:
            user_text.append(user_seq[i.item()])
            train_labels.append(labels[i.item()])
        loader = DataLoader(dataset=LM_dataset(user_text, train_labels), batch_size=batch_size, shuffle=True)

    
    elif mode == 'infer':  # no shuffle
        loader = DataLoader(dataset=LM_dataset(user_seq, labels), batch_size=batch_size*5)

    return loader


def evaluation(data, tokenizer, model, batch_size, all_data_labels, all_label2num, device, max_length=512):     
    eval_data, labels = data
    eval_idx_all = torch.tensor([i for i in range(len(eval_data))]).to(device)
    eval_loader = build_LM_dataloader(batch_size, eval_idx_all, eval_data, labels, 'infer')
    model.eval()

    all_preds = []
    all_labels = []

    possible_tokens = {}
    for key, values in all_data_labels.items():
        possible_tokens[key] = []
        for value in values:
            true_label = tokenizer(value, add_special_tokens=False)['input_ids']
            possible_tokens[key].append(true_label)

    with torch.no_grad():
        for batch in tqdm(eval_loader):
            tokenized_tensors, all_eval_labels = batch_to_tensor(tokenizer, batch, max_length, device)

            if tokenized_tensors is None: continue

            logits = model(tokenized_tensors)

            cur_labels = tokenized_tensors.input_ids.clone()
            mask_token_index = (tokenized_tensors.input_ids == tokenizer.mask_token_id)
            cur_labels[~mask_token_index] = -100

            mask_token_index = mask_token_index.to(torch.int64)
            final_mask_idxs = mask_token_index.argmax(dim=1)

            final_labels = []
            eval_labels = [item.split(",", 1)[1] for item in all_eval_labels]
            eval_dataset_name = [item.split(",", 1)[0].lower() for item in all_eval_labels]
            for i in range(len(eval_labels)):
                final_labels.append(all_label2num[eval_dataset_name[i]][eval_labels[i]])
                true_label = tokenizer(eval_labels[i].lower(), add_special_tokens=False)['input_ids']
                for j in range(len(true_label)):
                    cur_labels[i][final_mask_idxs[i]+j] = true_label[j]

            result = {}
            
            category_idxs = defaultdict(list)
            for i in range(len(eval_dataset_name)):
                category_idxs[eval_dataset_name[i]].append(i)
            
            logits = logits[0]

            for key, values in possible_tokens.items():
                cur_result = [] 
                for i, idx in enumerate(values):
                    idx_tensor = torch.tensor(idx).to(logits.device)
                    idx_tensor = idx_tensor.unsqueeze(0).unsqueeze(0)
                    idx_tensor = idx_tensor.expand(logits.size(0), logits.size(1), -1)

                    gathered = torch.gather(logits, dim=2, index=idx_tensor)

                    if len(idx) > 1:
                        gathered = gathered.mean(dim=2, keepdim=True)

                    cur_result.append(gathered)
                result[key] = cur_result

            predicted_token_ids = {}
            for key, values in result.items():
                predicted_token_ids[key] = torch.cat(values, dim=2)

            final_preds = []
            cur_labels = cur_labels.tolist()

            for i in range(len(cur_labels)):
                cur_p = []
                for j in range(final_mask_idxs[i], len(cur_labels[i])):
                    if cur_labels[i][j] != -100:
                        cur_p.append(predicted_token_ids[eval_dataset_name[i]][i][j])
                    else:  break
                if len(cur_p) == 1: final_preds.append(cur_p[0])
                else:
                    cur_p = torch.mean(torch.stack(cur_p), dim=0)
                    final_preds.append(cur_p)

            final_final_preds = {}
            final_final_labels = {}
            for key, values in category_idxs.items():
                final_final_preds[key] = []
                final_final_labels[key] = []
                for idx in values:
                    final_final_preds[key].append(final_preds[idx])
                    final_final_labels[key].append(final_labels[idx])

            for key, values in final_final_preds.items():
                final_final_preds[key] = torch.stack(values)
                final_final_preds[key] = final_final_preds[key].argmax(dim=-1)
                final_final_preds[key] = final_final_preds[key].tolist()
            
                for i in range(len(final_final_preds[key])):
                    if final_final_preds[key][i] == final_final_labels[key][i]: all_preds.append(1)
                    else: all_preds.append(0)
                    all_labels.append(1)
            
            del tokenized_tensors, logits, cur_labels, category_idxs, mask_token_index, result, predicted_token_ids, final_labels, final_preds, final_final_preds, final_final_labels
            torch.cuda.empty_cache()


    correct = sum([1 if pred == true else 0 for pred, true in zip(all_preds, all_labels)])
    accuracy = correct / len(all_preds)

    micro_f1 = f1_score(all_labels, all_preds, average='micro', zero_division=1)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=1)

    return accuracy, micro_f1, macro_f1


if __name__ == "__main__":
    device = 1
    target_dataset = "imdb"
    max_length = 512
    save_data_dir = f"../data/{target_dataset}_output/"

    train_data = open_txt_file(save_data_dir+f"train_{target_dataset}_nc_adaptation.txt")
    test_data = open_txt_file(save_data_dir+f"test_{target_dataset}_nc_adaptation.txt")
    train_labels = open_txt_file(save_data_dir+f"train_labels_{target_dataset}_nc_adaptation.txt")
    test_labels = open_txt_file(save_data_dir+f"test_labels_{target_dataset}_nc_adaptation.txt")


    card = 'distilroberta-base'
    model_config = {'lm_model': 'distilbert', 'hidden_dim': 128, 'device': device, 'card': card}
    model, tokenizer = build_LM_model(model_config)

    # load ck for adaptation
    ckpt_filepath = f"../model_output/{target_dataset}/"
    save_path = ckpt_filepath+'cross_domain_multi_task_fine_tuning_wo_imdb.pkl'
    lm_ckpt = torch.load(save_path, map_location=torch.device(f'cuda:{device}'))
    model.load_state_dict(lm_ckpt['model'])
    print("ck has been loaded.")

    batch_size = 16
    all_data_labels = {
        'imdb': ['action', 'comedy', 'drama'], 
        'dblp': ['0', '1', '2', '3'], 
        'pubmed': ['cardiovascular disease', 'glandular disease', 'nervous disorder', 
                   'communicable disease', 'inflammatory disease', 'pycnosis', 
                   'skin disease', 'cancer'], 
        'yelp': ['shopping', 'event planning & services', 'automotive', 'Italian', 
                 'beauty & spas', 'pizza', 'sandwiches', 'food', 
                 'bars', 'breakfast & brunch', 'restaurants', 'American (traditional)', 
                 'nightlife', 'burgers', 'Mexican', 'American (new)']
    }

    all_label2num = {
        'imdb': {'action': 0, 'comedy': 1, 'drama': 2},
        'dblp': {'0': 0, '1': 1, '2': 2, '3': 3}, 
        'pubmed': {'cardiovascular disease': 0, 'glandular disease': 1, 'nervous disorder': 2, 
                   'communicable disease': 3, 'inflammatory disease': 4, 'pycnosis': 5, 
                   'skin disease': 6, 'cancer': 7}, 
        'yelp': {'shopping': 0, 'event planning & services': 1, 'automotive': 2, 
                 'Italian': 3, 'beauty & spas': 4, 'pizza': 5, 'sandwiches': 6, 
                 'food': 7, 'bars': 8, 'breakfast & brunch': 9, 'restaurants': 10,
                 'American (traditional)': 11, 'nightlife': 12, 'burgers': 13, 
                 'Mexican': 14, 'American (new)': 15}
    }

    eval_data = test_data, test_labels
    valid_acc, valid_mi_f1, valid_ma_f1 = evaluation(eval_data, tokenizer, model, batch_size, all_data_labels, all_label2num, device, max_length)

    print(f'LM Valid Accuracy = {valid_acc}')
    print(f'LM Valid Micro F1 = {valid_mi_f1}')
    print(f'LM Valid Macro F1 = {valid_ma_f1}')
import os
import time
import argparse
import random
import numpy as np
from models.encoder_SkipFlow import build_ProTACT
import tensorflow as tf
from configs.configs import Configs
from utils.read_data_pr import read_pos_vocab, read_word_vocab, read_prompts_we, read_essays_prompts, read_prompts_pos
from utils.general_utils import get_scaled_down_scores, pad_hierarchical_text_sequences, get_attribute_masks, load_word_embedding_dict, build_embedd_table, separate_and_rescale_attributes_for_scoring
from evaluators.multitask_evaluator_all_attributes import Evaluator as AllAttEvaluator
from tensorflow import keras
import matplotlib.pyplot as plt
from metrics.metrics import *



def calc_kappa(pred, original, weight='quadratic'):
        kappa_score = kappa(original, pred, weight)
        return kappa_score



def main():
    parser = argparse.ArgumentParser(description="ProTACT model")
    parser.add_argument('--test_prompt_id', type=int, default=1, help='prompt id of test essay set')
    parser.add_argument('--seed', type=int, default=12, help='set random seed')
    parser.add_argument('--model_name', type=str,
                        choices=['ProTACT'],
                        help='name of model')
    parser.add_argument('--num_heads', type=int, default=2, help='set the number of heads in Multihead Attention')
    parser.add_argument('--features_path', type=str, default='data/hand_crafted_v3.csv')
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    test_prompt_id = args.test_prompt_id
    seed = args.seed
    num_heads = args.num_heads
    epochs = args.epochs
    features_path = args.features_path + str(test_prompt_id) + '.csv'

    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    print("Test prompt id is {} of type {}".format(test_prompt_id, type(test_prompt_id)))
    print("Seed: {}".format(seed))

    configs = Configs()

    checkpoint_path = configs.CHECKPOINT_PATH
    model_name = configs.MODEL_NAME
    data_path = configs.DATA_PATH
    train_path = data_path + str(test_prompt_id) + '/train.pk'
    dev_path = data_path + str(test_prompt_id) + '/dev.pk'
    test_path = data_path + str(test_prompt_id) + '/test.pk'
    pretrained_embedding = configs.PRETRAINED_EMBEDDING
    embedding_path = configs.EMBEDDING_PATH
    readability_path = configs.READABILITY_PATH
    prompt_path = configs.PROMPT_PATH
    vocab_size = configs.VOCAB_SIZE
    # epochs = configs.EPOCHS
    batch_size = configs.BATCH_SIZE
    
    print("Numhead : ", num_heads, " | Features : ", features_path, " | Pos_emb : ", configs.EMBEDDING_DIM)

    read_configs = {
        'train_path': train_path,
        'dev_path': dev_path,
        'test_path': test_path,
        'features_path': features_path,
        'readability_path': readability_path,
        'vocab_size': vocab_size
    }
    # read POS for prompts
    pos_vocab = read_pos_vocab(read_configs)
    prompt_pos_data = read_prompts_pos(prompt_path, pos_vocab) # for prompt POS embedding 

    # read words for prompts 
    word_vocab = read_word_vocab(read_configs)
    prompt_data = read_prompts_we(prompt_path, word_vocab) # for prompt word embedding 
    
    # read essays and prompts
    train_data, dev_data, test_data = read_essays_prompts(read_configs, prompt_data, prompt_pos_data, pos_vocab) 

    if pretrained_embedding:
        embedd_dict, embedd_dim, _ = load_word_embedding_dict(embedding_path)
        embedd_matrix = build_embedd_table(word_vocab, embedd_dict, embedd_dim, caseless=True)
        embed_table = [embedd_matrix]
    else:
        embed_table = None

    max_sentlen = max(train_data['max_sentlen'], dev_data['max_sentlen'], test_data['max_sentlen'])
    max_sentnum = max(train_data['max_sentnum'], dev_data['max_sentnum'], test_data['max_sentnum'])
    prompt_max_sentlen = prompt_data['max_sentlen']
    prompt_max_sentnum = prompt_data['max_sentnum']

    print('max sent length: {}'.format(max_sentlen))
    print('max sent num: {}'.format(max_sentnum))
    print('max prompt sent length: {}'.format(prompt_max_sentlen))
    print('max prompt sent num: {}'.format(prompt_max_sentnum))

    test_data['y_scaled'] = get_scaled_down_scores(test_data['data_y'], test_data['prompt_ids'])
    X_test_pos = pad_hierarchical_text_sequences(test_data['pos_x'], max_sentnum, max_sentlen)
    X_test_pos = X_test_pos.reshape((X_test_pos.shape[0], X_test_pos.shape[1] * X_test_pos.shape[2]))
    X_test_prompt = pad_hierarchical_text_sequences(test_data['prompt_words'], max_sentnum, max_sentlen)
    X_test_prompt = X_test_prompt.reshape((X_test_prompt.shape[0], X_test_prompt.shape[1] * X_test_prompt.shape[2]))
    X_test_prompt_pos = pad_hierarchical_text_sequences(test_data['prompt_pos'], max_sentnum, max_sentlen)
    X_test_prompt_pos = X_test_prompt_pos.reshape((X_test_prompt_pos.shape[0], X_test_prompt_pos.shape[1] * X_test_prompt_pos.shape[2]))
    X_test_linguistic_features = np.array(test_data['features_x'])
    X_test_readability = np.array(test_data['readability_x'])
    Y_test = np.array(test_data['y_scaled'])
    X_test_attribute_rel = get_attribute_masks(Y_test)

    print('================================')
    print('X_test_pos: ', X_test_pos.shape)
    print('X_test_prompt_words: ', X_test_prompt.shape)
    print('X_test_prompt_pos: ', X_test_prompt_pos.shape)
    print('X_test_readability: ', X_test_readability.shape)
    print('X_test_ling: ', X_test_linguistic_features.shape)
    print('X_test_attribute_rel: ', X_test_attribute_rel.shape)
    print('Y_test: ', Y_test.shape)
    print('================================')

    test_features_list = [X_test_pos, X_test_prompt, X_test_prompt_pos, X_test_linguistic_features, X_test_readability]

    model = build_ProTACT(len(pos_vocab), len(word_vocab), max_sentnum, max_sentlen, 
                      X_test_readability.shape[1],
                      X_test_linguistic_features.shape[1],
                      configs, Y_test.shape[1], num_heads, embed_table)
    
    Y_test_org = separate_and_rescale_attributes_for_scoring(Y_test, test_data['prompt_ids'])
    
    test_mean_list = []
        
    for epoch in range(1, epochs+1):
        model.load_weights(f"{checkpoint_path}{model_name}/{model_name}_{epoch}.weights.h5")
        test_pred = model.predict(test_features_list, batch_size=32)
        test_pred_dict = separate_and_rescale_attributes_for_scoring(test_pred, test_data['prompt_ids'])
        kappa_test = {key: calc_kappa(test_pred_dict[key], Y_test_org[key]) for key in
                         test_pred_dict.keys()}
        test_kappa_mean = np.mean(list(kappa_test.values()))
        test_mean_list.append(test_kappa_mean)
        
    if not os.path.exists('images'):
        os.makedirs('images')

    plt.plot(range(len(test_mean_list)), test_mean_list)
    plt.savefig(f"images/{model_name}.png")

if __name__ == '__main__':
    main()

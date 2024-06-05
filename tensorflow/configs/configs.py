class Configs:
    DROPOUT = 0.5
    CNN_FILTERS = 100
    CNN_KERNEL_SIZE = 5
    LSTM_UNITS = 144
    EMBEDDING_DIM = 50
    PRETRAINED_EMBEDDING = True
    EMBEDDING_PATH = './../glove/glove.6B.50d.txt'
    VOCAB_SIZE = 4000
    DATA_PATH = './../data/cross_prompt_attributes/'
    FEATURES_PATH = './../data/hand_crafted_v3.csv'
    READABILITY_PATH = './../data/allreadability.pickle'
    PROMPT_PATH = './../data/prompt_info_pp.csv'
    EPOCHS = 50
    BATCH_SIZE = 32
    OUTPUT_PATH = './../outputs/'
    CHECKPOINT_PATH = "./../checkpoints/"
    MODEL_NAME = "encoder_skipflow"

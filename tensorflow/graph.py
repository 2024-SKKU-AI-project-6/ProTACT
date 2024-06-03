from tqdm import tqdm
import numpy as np
import pickle
import argparse
import matplotlib.pyplot as plt



def get_args():
    parser = argparse.ArgumentParser(description="graph")
    parser.add_argument('--model_name', type=str, default="baseline", help='name of model')
    parser.add_argument("--epochs", type=int, default=50, help="epochs of model")
    parser.add_argument("--output_path", type=str, default="./../outputs")
    return parser.parse_args()



def main():
    args = get_args()
    model_name = "skipflow_1"
    epochs = args.epochs
    output_path = args.output_path
    
    path = f"{output_path}/{model_name}_{epochs}"
    baseline_path = f"{output_path}/baseline_{epochs}"
    keys = ['score', 'content', 'organization', 'word_choice', 'sentence_fluency', 'conventions']
    keys_2 = ["qwk_test", "train_loss", "epoch_times"]
    data = {}
    baseline_data = {}
    baseline_2 = {}
    bigru = {}
    bigru_2 = {}
    gru = {}
    for key in keys:
        data[key] =[]
        baseline_data[key] =[]
        bigru[key] =[]
    
    for key in keys_2:
        bigru_2[key] = []
        gru[key] = []
        baseline_2[key] = []
    for epoch in tqdm(range(1, epochs+1)):
        avg = []
        with open(path+f"/test_{epoch}.pkl", 'rb') as f:
            test = pickle.load(f)
            for key in keys:
                data[key].append(test[key])
                avg.append(test[key])
        baseline_2["qwk_test"].append(np.mean(avg))
        
    for epoch in tqdm(range(1, epochs+1)):
        with open(baseline_path+f"/test_{epoch}.pkl", 'rb') as f:
            test = pickle.load(f)
            for key in keys:
                baseline_data[key].append(test[key])
    
    
    with open("/home/pea/ProTACT/outputs/bigru.pkl", 'rb') as f:
        test = pickle.load(f)
        print(test[0].keys())
        for epoch in tqdm(range(epoch)):
            avg = []
            for key in keys:
                bigru[key].append(test[epoch]["TEST_"+key+"_QWK"])
                avg.append(test[epoch]["TEST_"+key+"_QWK"])
            bigru_2["qwk_test"].append(np.mean(avg))
            bigru_2["train_loss"].append(test[epoch]["loss"])
            bigru_2["epoch_times"].append(test[epoch]["epoch_time"])
                
    with open("/home/pea/ProTACT/outputs/training_data_epoch_50.pkl", 'rb') as f:
        test = pickle.load(f)
        for key in keys_2:
            gru[key] = test[key]
            
    for key in keys:
        fig, graph = plt.subplots()
        graph.plot(baseline_data[key],'r',label='baseline')
        graph.plot(data[key],'y',label=model_name)
        graph.set_xlabel('epoch')
        graph.set_ylabel('QWK')
        plt.savefig(f'images/{model_name}_{key}_result.png')
        
    for key in keys:
        fig, graph = plt.subplots()
        graph.plot(baseline_data[key],'r',label='baseline')
        graph.plot(bigru[key],'y',label="bigru")
        graph.set_xlabel('epoch')
        graph.set_ylabel('QWK')
        plt.savefig(f'images/bigru_{key}_result.png')
    
    for key in keys:
        fig, graph = plt.subplots()
        graph.plot(baseline_data[key],'r',label='baseline')
        graph.plot(data[key],'g',label=model_name)
        graph.plot(bigru[key],'b',label="bigru")
        graph.set_xlabel('epoch')
        graph.set_ylabel('QWK')
        plt.savefig(f'images/all_{key}_result.png')
        
    for key in ["qwk_test"]:
        fig, graph = plt.subplots()
        graph.plot(baseline_2[key],'r',label='baseline')
        graph.plot(gru[key],'g',label="gru")
        graph.plot(bigru_2[key],'b',label="bigru")
        graph.set_xlabel('epoch')
        graph.set_ylabel('QWK')
        plt.savefig(f'images/gru_{key}_result.png')
        
    
    
if __name__=="__main__":
    main()
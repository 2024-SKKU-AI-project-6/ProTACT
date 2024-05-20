from metrics.metrics import *
from utils.general_utils import separate_attributes_for_scoring, separate_and_rescale_attributes_for_scoring
import torch
from tqdm import tqdm
'''import os
cur_dir = os.getcwd()
ckpt_dir = 'checkpoints'
dir = os.path.join(cur_dir, ckpt_dir)
os.makedirs(dir, exist_ok=True)
'''


class Evaluator():

    def __init__(self, X_dev_prompt_ids, X_test_prompt_ids, dev_loader, test_loader,
                 Y_dev, Y_test, seed, device, criterion):
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.X_dev_prompt_ids, self.X_test_prompt_ids = X_dev_prompt_ids, X_test_prompt_ids
        self.Y_dev, self.Y_test = Y_dev, Y_test
        self.Y_dev_upscale = Y_dev * 100
        self.Y_dev_org = separate_attributes_for_scoring(
            self.Y_dev_upscale, self.X_dev_prompt_ids)
        self.Y_test_org = separate_and_rescale_attributes_for_scoring(
            Y_test, self.X_test_prompt_ids)
        self.best_dev_kappa_mean = -1
        self.best_test_kappa_mean = -1
        self.best_dev_kappa_set = {}
        self.best_test_kappa_set = {}
        self.seed = seed
        self.device = device
        self.criterion = criterion

    @staticmethod
    def calc_pearson(pred, original):
        pr = pearson(pred, original)
        return pr

    @staticmethod
    def calc_spearman(pred, original):
        spr = spearman(pred, original)
        return spr

    @staticmethod
    def calc_kappa(pred, original, weight='quadratic'):
        kappa_score = kappa(original, pred, weight)
        return kappa_score

    @staticmethod
    def calc_rmse(pred, original):
        rmse = root_mean_square_error(original, pred)
        return rmse

    def evaluate(self, model, epoch, print_info=True):
        self.current_epoch = epoch
        model.eval()

        with torch.no_grad():
            dev_pred = []
            dev_loss = 0.
            for batch_data in self.dev_loader:
                batch_data = [x.to(self.device) for x in batch_data]
                inputs, targets = batch_data[:-1], batch_data[-1]
                pred = model(*inputs)
                loss = self.criterion(targets, pred)  # 개발 데이터에 대한 loss 계산
                dev_loss += loss.item()  # 개발 데이터의 전체 loss 누적
                dev_pred.append(pred.cpu().numpy())
            # NumPy 배열로 concatenate
            dev_pred = np.concatenate(dev_pred, axis=0)
            dev_loss /= len(self.dev_loader)  # 개발 데이터의 평균 loss 계산

            test_pred = []
            test_loss = 0.
            for batch_data in self.test_loader:
                batch_data = [x.to(self.device) for x in batch_data]
                inputs, targets = batch_data[:-1], batch_data[-1]
                pred = model(*inputs)
                loss = self.criterion(targets, pred)  # 테스트 데이터에 대한 loss 계산
                test_loss += loss.item()  # 테스트 데이터의 전체 loss 누적
                test_pred.append(pred.cpu().numpy())
            test_pred = np.concatenate(
                test_pred, axis=0)  # NumPy 배열로 concatenate
            test_loss /= len(self.test_loader)  # 테스트 데이터의 평균 loss 계산

        print("Epoch: {}, Dev Loss: {:.4f}, Test Loss: {:.4f}".format(
            self.current_epoch, dev_loss, test_loss))

        print("dev_pred_shape: ", dev_pred.shape)
        print("test_pred_shape: ", test_pred.shape)

        dev_pred_int = dev_pred * 100
        dev_pred_dict = separate_attributes_for_scoring(
            dev_pred_int, self.X_dev_prompt_ids)
        test_pred_dict = separate_and_rescale_attributes_for_scoring(
            test_pred, self.X_test_prompt_ids)

        # pearson_dev = {key: self.calc_pearson(
        #     dev_pred_dict[key], self.Y_dev_org[key]) for key in dev_pred_dict.keys()}
        # pearson_test = {key: self.calc_pearson(
        #     test_pred_dict[key], self.Y_test_org[key]) for key in test_pred_dict.keys()}
        # spearman_dev = {key: self.calc_spearman(
        #     dev_pred_dict[key], self.Y_dev_org[key]) for key in dev_pred_dict.keys()}
        # spearman_test = {key: self.calc_spearman(
        #     test_pred_dict[key], self.Y_test_org[key]) for key in test_pred_dict.keys()}

        self.kappa_dev = {key: self.calc_kappa(
            dev_pred_dict[key], self.Y_dev_org[key]) for key in dev_pred_dict.keys()}
        self.kappa_test = {key: self.calc_kappa(
            test_pred_dict[key], self.Y_test_org[key]) for key in test_pred_dict.keys()}

        self.dev_kappa_mean = np.mean(list(self.kappa_dev.values()))
        self.test_kappa_mean = np.mean(list(self.kappa_test.values()))

        if self.dev_kappa_mean > self.best_dev_kappa_mean:
            self.best_dev_kappa_mean = self.dev_kappa_mean
            self.best_test_kappa_mean = self.test_kappa_mean
            self.best_dev_kappa_set = self.kappa_dev
            self.best_test_kappa_set = self.kappa_test
            self.best_dev_epoch = epoch

            '''
            file_path = os.path.join(
                dir, f"checkpoint_best{self.test_prompt_id}_{self.seed}.pt")
            torch.save(model.state_dict(), file_path)
            print("Save best model to", file_path)
            '''

        if print_info:
            self.print_info()

    def print_info(self):
        print('CURRENT EPOCH: {}'.format(self.current_epoch))
        print('[DEV] AVG QWK: {}'.format(round(self.dev_kappa_mean, 3)))
        for att in self.kappa_dev.keys():
            print('[DEV] {} QWK: {}'.format(
                att, round(self.kappa_dev[att], 3)))
        print('------------------------')
        print('[TEST] AVG QWK: {}'.format(round(self.test_kappa_mean, 3)))
        for att in self.kappa_test.keys():
            print('[TEST] {} QWK: {}'.format(
                att, round(self.kappa_test[att], 3)))
        print('------------------------')
        print('[BEST TEST] AVG QWK: {}, {{epoch}}: {}'.format(
            round(self.best_test_kappa_mean, 3), self.best_dev_epoch))
        for att in self.best_test_kappa_set.keys():
            print('[BEST TEST] {} QWK: {}'.format(
                att, round(self.best_test_kappa_set[att], 3)))
        print('--------------------------------------------------------------------------------------------------------------------------')

    def print_final_info(self):
        print('[BEST TEST] AVG QWK: {}, {{epoch}}: {}'.format(
            round(self.best_test_kappa_mean, 3), self.best_dev_epoch))
        for att in self.best_test_kappa_set.keys():
            print('[BEST TEST] {} QWK: {}'.format(
                att, round(self.best_test_kappa_set[att], 3)))
        print('--------------------------------------------------------------------------------------------------------------------------')

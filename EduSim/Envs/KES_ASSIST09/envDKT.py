# -*- coding:utf-8 _*-
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn import metrics
from EduSim.utils import get_proj_path, get_raw_data_path
from EduSim.deep_model import KTnet
from EduSim.utils import mds_concat
cur_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(cur_path[:cur_path.find('IDALPR')] + 'IDALPR')


class ASSIST09DataSet:
    def __init__(self, data_path, num_skills, feature_dim, max_sequence_length):
        self.data_path = data_path
        self.feature_dim = feature_dim
        self.num_skills = num_skills
        self.max_sequence_length = max_sequence_length

    def __getitem__(self, index):
        with open(self.data_path + str(index) + '.csv', 'r') as f:
            data = f.readlines()[1:]  # header exists
            data = data[:self.max_sequence_length]
        data = [[int(line.rstrip().split(',')[0]) - 1, int(line.rstrip().split(',')[1])] for i, line in enumerate(data)]

        session = self.get_feature_matrix(data)
        return session  # [max_sequence_length, feature_dim]


    def __len__(self):
        return len(os.listdir(self.data_path))

    def get_feature_matrix(self, session):
        input_data = np.zeros(shape=(self.max_sequence_length, self.feature_dim), dtype=np.float32)

        j = 0
        while j < self.max_sequence_length and j < len(session):
            problem_id = session[j][0]
            if session[j][1] == 0:
                input_data[j][problem_id] = 1.0
            elif session[j][1] == 1:
                input_data[j][problem_id + self.num_skills] = 1.0
            j += 1
        return torch.Tensor(input_data)


class EnvKTtrainer:
    def __init__(self, num_skills, train_goal='env_KT'):

        self.num_skills = num_skills
        self.feature_dim = 2 * self.num_skills


        self.train_goal = train_goal
        self.test_only = True
        print('Current training goal is:' + self.train_goal)


        self.base_data_path = f'{get_raw_data_path()}/ASSIST19/processed/'
        self.max_sequence_length = 100
        self.epoch_num = 10
        self.test_acc = 0.0
        self.test_auc = 0.0
        self.val_max_auc = 0.0
        self.test_max_auc = 0.0
        if self.train_goal == 'env_KT':
            self.weight_path = f'{get_proj_path()}/EduSim/Envs/KES_ASSIST09/meta_data/env_weights/'
            os.makedirs(self.weight_path, exist_ok=True)
        elif self.train_goal == 'agent_KT':
            self.weight_path = f'{get_proj_path()}/EduSim/Envs/KES_ASSIST09/meta_data/agent_weights/'
            os.makedirs(self.weight_path, exist_ok=True)


        self.learning_rate = 0.001
        self.embed_dim = 64
        self.hidden_size = 128
        self.batch_size = 16


        self.train_dataset = ASSIST09DataSet(f'{self.base_data_path}1/train/',
                                             self.num_skills, self.feature_dim, self.max_sequence_length)
        self.val_dataset = ASSIST09DataSet(f'{self.base_data_path}1/val/',
                                           self.num_skills, self.feature_dim, self.max_sequence_length)
        self.test_dataset = ASSIST09DataSet(f'{self.base_data_path}1/test/',
                                            self.num_skills, self.feature_dim, self.max_sequence_length)

        self.train_dataset = DataLoader(self.train_dataset, column_names=['session'])
        self.train_dataset = self.train_dataset.batch(self.batch_size)

        self.val_dataset = DataLoader(self.val_dataset, column_names=['session'])
        self.val_dataset = self.val_dataset.batch(self.batch_size)

        self.test_dataset = DataLoader(self.test_dataset, column_names=['session'])
        self.test_dataset = self.test_dataset.batch(self.batch_size)

        self.train_size = self.train_dataset.get_dataset_size()
        self.val_size = self.val_dataset.get_dataset_size()
        self.test_size = self.test_dataset.get_dataset_size()


        KT_para_dict = {
            'input_size': self.feature_dim,
            'emb_dim': self.embed_dim,
            'hidden_size': self.hidden_size,
            'num_skills': self.num_skills,
            'nlayers': 2,
            'dropout': 0.0,
        }
        self.KTnet = KTnet(KT_para_dict)

        self.optimizer = nn.Adam(self.KTnet.trainable_params(), learning_rate=self.learning_rate)
        self.loss_f = nn.BCEWithLogitsLoss()


        if self.test_only:
            param_dict = torch.load(f'{self.weight_path}ValBest.ckpt')
            _, _ = torch.load(self.KTnet, param_dict)
            self.test(name='Test', dataloader=self.test_dataset)
            assert 0

    def forward_fn(self, batch_data):
        output = self.KTnet(batch_data)
        loss, batch_pred_probs, batch_true_labels, sequence_lengths = self.compute_loss(output, batch_data)
        return loss, batch_pred_probs, batch_true_labels, sequence_lengths

    def train(self):

        train_loss_list = []
        correct = 0
        true_labels = []
        pred_probs = []
        self.KTnet.set_train()
        for epoch in range(self.epoch_num):
            for i, batch_data in enumerate(self.train_dataset.create_tuple_iterator()):
                (loss, batch_pred_probs, batch_true_labels, sequence_lengths), grads = self.grad_fn(*batch_data)
                self.optimizer(grads)

                train_loss_list.append(loss.asnumpy())
                if i % 20 == 0:
                    print(f"epcoh:{epoch + 1}  iteration:{i}   loss:{np.mean(train_loss_list):.6f}")

                batch_pred_labels = (batch_pred_probs >= 0.5).astype(torch.float32)
                for j in range(batch_pred_labels.shape[0]):
                    if sequence_lengths[j] == 1:
                        continue
                    a = batch_pred_labels[j, 0:sequence_lengths[j] - 1]
                    b = batch_true_labels[j, 1:sequence_lengths[j]]

                    correct = correct + torch.equal(a, b).sum().asnumpy()
                    for p in batch_pred_probs[j, 0:sequence_lengths[j] - 1].view(-1):
                        pred_probs.append(p.asnumpy())
                    for t in batch_true_labels[j, 1:sequence_lengths[j]].view(-1):
                        true_labels.append(t.asnumpy())

            print(f'Cal train acc ..')
            acc = correct / len(true_labels)
            auc = metrics.roc_auc_score(true_labels, pred_probs)
            print(f'train acc:{acc}')
            print(f'train auc:{auc}')
            self.test('Val', self.val_dataset)

        self.test('test', self.test_dataset)

    def test(self, name, dataloader):
        print(f'...testing on {name} ...')
        test_loss = 0
        correct = 0
        true_labels = []
        pred_probs = []
        self.KTnet.set_train(False)
        k = 0
        for i, batch_data in enumerate(dataloader.create_tuple_iterator()):
            y_hat = self.KTnet(*batch_data)
            batch_loss, batch_pred_outs, batch_true_labels, sequence_lengths = self.compute_loss(y_hat, *batch_data)
            test_loss += batch_loss.asnumpy()
            batch_pred_labels = (batch_pred_outs >= 0.5).astype(torch.float32)

            for j in range(batch_pred_labels.shape[0]):
                if sequence_lengths[j] == 1:
                    continue
                a = batch_pred_labels[j, 0:sequence_lengths[j] - 1]
                b = batch_true_labels[j, 1:sequence_lengths[j]]

                correct = correct + torch.equal(a, b).sum().asnumpy()
                for p in batch_pred_outs[j, 0:sequence_lengths[j] - 1].view(-1):
                    pred_probs.append(p.asnumpy())
                for t in batch_true_labels[j, 1:sequence_lengths[j]].view(-1):
                    true_labels.append(t.asnumpy())
            k = i

        test_loss /= (k + 1)

        acc = correct / len(pred_probs)
        auc = metrics.roc_auc_score(true_labels, pred_probs)
        print(f'Test result: Average loss: {float(test_loss):.4f}  '
              f'acc: {correct}/{len(pred_probs)}={acc}  '
              f'auc: {auc}')
        self.KTnet.set_train()

        if name == 'Val':
            if self.val_max_auc < auc:
                self.val_max_auc = auc
                torch.save(self.KTnet, f'{self.weight_path}ValBest.pt')
        else:
            self.test_acc = acc
            self.test_auc = auc

    def compute_loss(self, output, batch_data):
        sequence_lengths = [int(torch.sum(sample)) for sample in batch_data]
        target_corrects = torch.Tensor([], dtype=torch.float32)
        target_ids = torch.Tensor([], dtype=torch.int32)
        output = output.permute(1, 0, 2)
        for episode in range(batch_data.shape[0]):
            tmp_target_id = torch.argmax(batch_data[episode, :, :], dim=-1)
            ones = torch.ones(tmp_target_id.shape, dtype=torch.float32)
            zeros = torch.zeros(tmp_target_id.shape, dtype=torch.float32)
            # [sequence_length, 1]
            target_correct = torch.where(tmp_target_id > self.num_skills - 1, ones, zeros).unsqueeze(0).unsqueeze(0)
            target_id = torch.where(tmp_target_id > self.num_skills - 1, tmp_target_id - self.num_skills, tmp_target_id)  # [sequence_length]

            target_id = torch.roll(target_id, -1, 0).unsqueeze(1).unsqueeze(0)  # [1, sequence_length, 1]

            target_ids = mds_concat((target_ids, target_id), 0)
            target_corrects = mds_concat((target_corrects, target_correct), 0)

        logits = torch.nn.Sigmoid(output)
        preds = logits.gather_elements(dim=2, index=target_ids)
        loss = torch.Tensor([0.0])
        for i, sequence_length in enumerate(sequence_lengths):
            if sequence_length <= 1:
                continue
            a = logits[i, 0:sequence_length - 1]
            b = target_corrects[i, 1:sequence_length]
            loss = loss + self.loss_f(a, b)

        return loss, preds, target_corrects, sequence_lengths


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"

    handler = EnvKTtrainer(num_skills=100, train_goal='env_KT')
    handler.train()
    handler_A = EnvKTtrainer(num_skills=100, train_goal='agent_KT')
    handler_A.train()

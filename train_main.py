import os
import torch
from tqdm import tqdm
import logging
from torchmeta.utils.prototype import  prototypical_loss
from torchvision.transforms import ToTensor, Resize, Compose
from model import PrototypicalNetwork
from model import PrototypicalNetworkJoint
from utils import get_accuracy, get_accuracy2, get_accuracy3, Reservoir_memory,  CE_loss, unlabel_select
from utils import get_prototypes
import numpy as np
from datasets import *
import pickle
import random
import glob
import cv2
import solvers
from PIL import Image
from mi_estimators import *
import warnings
warnings.filterwarnings("ignore")



datanames = ['Plantae', 'CUB', 'MiniImagenet', 'CIFARFS', 'Aircraft', 'Butterfly']


class SequentialMeta(object):
    def __init__(self,model, optimizer, args=None):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.step = 0
        self.memory_rep = []
        str_save = '_'.join(datanames)

        sample_dim = 256
        hidden_size = 10
        estimator_name = "Upper" 
        self.mi_estimatorOOD = eval(estimator_name)(sample_dim, sample_dim, hidden_size).to(device=args.device)

        estimator_name = "Lower" 
        self.mi_estimatorID = eval(estimator_name)(sample_dim, sample_dim, hidden_size).to(device=args.device)

        self.iterator = True
        if self.iterator:
            dir_path = '/media/zheshiyige/Elements/NIPS2021 data/data/fishprocess.pt'
            self.fish = torch.load(dir_path)
            num_train = int(0.7*len(self.fish))
            self.train_fishOOD = self.fish[0:num_train]
            self.test_fishOOD = self.fish[num_train:]


        self.filepath = os.path.join(self.args.output_folder, 'protonet_semi_{}'.format(str_save), 'Block{}'.format(self.args.num_block), 'shot{}'.format(self.args.num_shot), 'way{}'.format(self.args.num_way))
        if not os.path.exists(self.filepath):
            os.makedirs(self.filepath)
        
    def train(self, dataloader_dict, OOD_dict_list, domain_id = None):
        self.model.train()
        refine = True
        select_unlabel = True

        OOD_dict1 = OOD_dict_list[0]
        OOD_dict2 = OOD_dict_list[1]

        for key in OOD_dict1:
            OODloader1 = OOD_dict1[key]
        OOD_iterator1 = iter(OODloader1)

        for key in OOD_dict2:
            OODloader2 = OOD_dict2[key]
        OOD_iterator2 = iter(OODloader2)

        for dataname, dataloader in dataloader_dict.items():
            
            with tqdm(dataloader, total=self.args.num_batches) as pbar:
                for batch_idx, batch in enumerate(pbar):
                    self.model.zero_grad()
                    train_inputs, train_targets = batch['train']
                    train_inputs = train_inputs.to(device=self.args.device)
                    train_targets = train_targets.to(device=self.args.device)
                    if train_inputs.size(2) == 1:
                        train_inputs = train_inputs.repeat(1, 1, 3, 1, 1)
                    train_embeddings = self.model(train_inputs)

                    test_inputs, test_targets = batch['test']
                    test_inputs = test_inputs.to(device=self.args.device)
                    test_targets = test_targets.to(device=self.args.device)
                    if test_inputs.size(2) == 1:
                        test_inputs = test_inputs.repeat(1, 1, 3, 1, 1)
                    test_embeddings = self.model(test_inputs)


                    unlabel_inputs, unlabel_targets = batch['unlabel']
                    unlabel_inputs = unlabel_inputs.to(device=self.args.device)
                    unlabel_targets = unlabel_targets.to(device=self.args.device)
                    if unlabel_inputs.size(2) == 1:
                        unlabel_inputs = unlabel_inputs.repeat(1, 1, 3, 1, 1)

                    
                    metabatch = unlabel_inputs.size(0)

                    if self.iterator:
                        nextbatch  = next(OOD_iterator1)
                        OOD_inputs, OOD_targets = nextbatch['train']
                        OOD_inputs1 = OOD_inputs.to(device=self.args.device)
                        OOD_targets = OOD_targets.to(device=self.args.device)
                        if OOD_inputs.size(2) == 1:
                            OOD_inputs1 = OOD_inputs1.repeat(1, 1, 3, 1, 1)

                        nextbatch  = next(OOD_iterator2)
                        OOD_inputs, OOD_targets = nextbatch['train']
                        OOD_inputs2 = OOD_inputs.to(device=self.args.device)
                        OOD_targets = OOD_targets.to(device=self.args.device)
                        if OOD_inputs.size(2) == 1:
                            OOD_inputs2 = OOD_inputs.repeat(1, 1, 3, 1, 1)


                        batchOOD = []
                        for ind in range (metabatch):

                            samplelist = random.sample(range(len(self.train_fishOOD)), self.args.num_unlabel)
                            selectOOD = []
                            for ind in samplelist:
                                image = self.train_fishOOD[ind]
                                selectOOD.append(image)
                            selectOOD = torch.stack(selectOOD)
                            batchOOD.append(selectOOD)

                        OOD_inputs3 = torch.stack(batchOOD).to(self.args.device)
                        concat_OOD = torch.cat([OOD_inputs1, OOD_inputs2, OOD_inputs3, unlabel_inputs], dim =1)
                        unlabel_embeddings = self.model(concat_OOD)
                    else:
                        unlabel_embeddings = self.model(unlabel_inputs)
                    
                    if select_unlabel:
                        prototypes, in_distribution, OOD_distribution = unlabel_select(train_embeddings, test_embeddings, unlabel_embeddings, self.args, refine)
                    else:
                        prototypes = get_prototypes(train_embeddings, train_targets, self.args, unlabel_embeddings, refine)
                    loss = CE_loss(prototypes, test_embeddings, test_targets, self.args)
                    
               
                    model_loss = self.mi_estimatorID.learning_loss(in_distribution[0], in_distribution[1])
                    OODmodel_loss = self.mi_estimatorOOD.learning_loss(OOD_distribution[0], OOD_distribution[1])
                    lambda1 = 1e-5
                    loss += - lambda1*model_loss
                    loss += lambda1*OODmodel_loss


                    if self.memory_rep:
                        OOD_iterator_list = (OOD_iterator1, OOD_iterator2)
                        memory_loss = self.Reservoir_memory(OOD_iterator_list)
                        loss += memory_loss

                        sampleOT, featureOT = self.sample_memory()
                        sampleOT = sampleOT.unsqueeze(0)
                        sampleOT_embeddings = self.model(sampleOT).squeeze()
                        
                        cost = self.form_cost_matrix(sampleOT_embeddings, featureOT)
                        dist1 = featureOT.cpu().data.numpy()
                        dist2 = sampleOT_embeddings.cpu().data.numpy()
                        if self.args.solver == 'OT':
                            solver = solvers.OTSolver(dist1, dist2, ground_cost=args.ground_cost, logdir=args.logdir)
                            OT_dist, coupling = solver.solve()


                        coupling = torch.tensor(coupling).to(self.args.device)
                        OT_dist = torch.sum(torch.mul(cost, coupling))
                        beta = 0.007
                        loss += beta*OT_dist


                    #Reservoir sampling
                    if self.step < self.args.memory_limit:
                        savedict = {'data': batch, 'feature': train_embeddings.detach()}
                        self.memory_rep.append(savedict)

                    else:
                        randind = random.randint(0, self.step)
                        if randind < self.args.memory_limit:
                            savedict = {'data': batch, 'feature': train_embeddings.detach()}
                            self.memory_rep[randind] = savedict
                

                    self.step = self.step+1
                    
                    loss.backward()
                    self.optimizer.step()
                    if batch_idx >= self.args.num_batches:
                        break

    def form_cost_matrix(self, x, y):
        if self.args.ground_cost == 'l2':
            C = torch.einsum('id,jd->ij', x, y)
            return torch.sum(x ** 2, 1) + torch.sum(y ** 2, 1) - 2 *C

    def sample_memory(self):

        count = self.args.sample_OT
        num_memory = len(self.memory_rep)
        if num_memory<count:
            selectmemory = self.memory_rep
        else:
            samplelist = random.sample(range(num_memory), count)
            selectmemory = []
            for ind in samplelist:
                selectmemory.append(self.memory_rep[ind])

        task_count = 5
        total_list = []
        feature_list = []
        for selectm in selectmemory:
                select = selectm['data']
                feature = selectm['feature']
                train_inputs, train_targets = select['train']
                train_inputs = train_inputs.to(device=self.args.device)
                train_targets = train_targets.to(device=self.args.device)
                selecttask = train_inputs[0]
                selectfeature = feature[0]
                num_data = self.args.num_way * self.args.num_shot
                samplelist = random.sample(range(num_data), task_count)
                total_list.append(selecttask[samplelist])
                feature_list.append(selectfeature[samplelist])
        total_list = torch.cat(total_list)
        feature_list = torch.cat(feature_list)
        return total_list, feature_list



    def Reservoir_memory(self, OOD_iterator_list):

        refine = True
        OOD_iterator1, OOD_iterator2 = OOD_iterator_list
        memory_loss =0
        count = self.args.sample
        num_memory = len(self.memory_rep)
        if num_memory<count:
            selectmemory = self.memory_rep
        else:
            samplelist = random.sample(range(num_memory), count)
            selectmemory = []
            for ind in samplelist:
                selectmemory.append(self.memory_rep[ind])
        for selectm in selectmemory:
            select = selectm['data']
            feature = selectm['feature']
            memory_train_inputs, memory_train_targets = select['train'] 
            memory_train_inputs = memory_train_inputs.to(device=self.args.device)
            memory_train_targets = memory_train_targets.to(device=self.args.device)
            if memory_train_inputs.size(2) == 1:
                memory_train_inputs = memory_train_inputs.repeat(1, 1, 3, 1, 1)
            memory_train_embeddings = self.model(memory_train_inputs)

            memory_test_inputs, memory_test_targets = select['test'] 
            memory_test_inputs = memory_test_inputs.to(device=self.args.device)
            memory_test_targets = memory_test_targets.to(device=self.args.device)
            if memory_test_inputs.size(2) == 1:
                memory_test_inputs = memory_test_inputs.repeat(1, 1, 3, 1, 1)
            memory_test_embeddings = self.model(memory_test_inputs)


            unlabel_inputs, unlabel_targets = select['unlabel']
            unlabel_inputs = unlabel_inputs.to(device=self.args.device)
            unlabel_targets = unlabel_targets.to(device=self.args.device)
            if unlabel_inputs.size(2) == 1:
                unlabel_inputs = unlabel_inputs.repeat(1, 1, 3, 1, 1)
            unlabel_embeddings = self.model(unlabel_inputs)

            metabatch = unlabel_inputs.size(0)
            unlabel_embeddings = self.model(unlabel_inputs)
            memory_prototypes = get_prototypes(memory_train_embeddings, memory_train_targets, self.args, unlabel_embeddings, refine)
            
            memory_prototypes = get_prototypes(memory_train_embeddings, memory_train_targets, self.args, unlabel_embeddings, refine)
            memory_loss += CE_loss(memory_prototypes, memory_test_embeddings, memory_test_targets, self.args)

        return memory_loss

    def save(self, epoch):
        # Save model
        if self.args.output_folder is not None:
            filename = os.path.join(self.filepath, 'epoch{0}.pt'.format(epoch))
            with open(filename, 'wb') as f:
                state_dict = self.model.state_dict()
                torch.save(state_dict, f)

    def load(self, args, epoch, model):
        filename = os.path.join(self.filepath, 'epoch{0}.pt'.format(epoch))
        self.model.load_state_dict(torch.load(filename))


    def valid(self, epoch, dataloader_dict, OOD_dict_list, domain_id):
        self.model.eval()
        refine = True
        select_unlabel = True
        OOD = True
        acc_list = []
        acc_dict = {}

        OOD_dict1 = OOD_dict_list[0]
        OOD_dict2 = OOD_dict_list[1]

        for key in OOD_dict1:
            OODloader1 = OOD_dict1[key]
        OOD_iterator1 = iter(OODloader1)

        for key in OOD_dict2:
            OODloader2 = OOD_dict2[key]
        OOD_iterator2 = iter(OODloader2)


        for dataname, dataloader in dataloader_dict.items():
            with torch.no_grad():
                with tqdm(dataloader, total=self.args.num_valid_batches) as pbar:
                    for batch_idx, batch in enumerate(pbar):
                        self.model.zero_grad()

                        train_inputs, train_targets = batch['train']
                        train_inputs = train_inputs.to(device=self.args.device)
                        train_targets = train_targets.to(device=self.args.device)
                        if train_inputs.size(2) == 1:
                            train_inputs = train_inputs.repeat(1, 1, 3, 1, 1)
                        train_embeddings = self.model(train_inputs)

                        test_inputs, test_targets = batch['test']
                        test_inputs = test_inputs.to(device=self.args.device)
                        test_targets = test_targets.to(device=self.args.device)
                        if test_inputs.size(2) == 1:
                            test_inputs = test_inputs.repeat(1, 1, 3, 1, 1)
                        test_embeddings = self.model(test_inputs)

                        unlabel_inputs, unlabel_targets = batch['unlabel']
                        unlabel_inputs = unlabel_inputs.to(device=self.args.device)
                        unlabel_targets = unlabel_targets.to(device=self.args.device)
                        if unlabel_inputs.size(2) == 1:
                            unlabel_inputs = unlabel_inputs.repeat(1, 1, 3, 1, 1)


                        metabatch = unlabel_inputs.size(0)
                        if OOD:
                            nextbatch  = next(OOD_iterator1)
                            OOD_inputs, OOD_targets = nextbatch['train']
                            OOD_inputs1 = OOD_inputs.to(device=self.args.device)
                            OOD_targets = OOD_targets.to(device=self.args.device)
                            if OOD_inputs.size(2) == 1:
                                OOD_inputs1 = OOD_inputs1.repeat(1, 1, 3, 1, 1)

                            nextbatch  = next(OOD_iterator2)
                            OOD_inputs, OOD_targets = nextbatch['train']
                            OOD_inputs2 = OOD_inputs.to(device=self.args.device)
                            OOD_targets = OOD_targets.to(device=self.args.device)
                            if OOD_inputs.size(2) == 1:
                                OOD_inputs2 = OOD_inputs2.repeat(1, 1, 3, 1, 1)

                            batchOOD = []
                            for ind in range (metabatch):
                                samplelist = random.sample(range(len(self.test_fishOOD)), self.args.num_unlabel)
                                selectOOD = []
                                tosensor = ToTensor()
                                for ind in samplelist:
                                    image = self.test_fishOOD[ind]
                                    selectOOD.append(image)
                                selectOOD = torch.stack(selectOOD)
                                batchOOD.append(selectOOD)

                            OOD_inputs3 = torch.stack(batchOOD).to(self.args.device)
                            concat_OOD = torch.cat([OOD_inputs1, OOD_inputs2, OOD_inputs3, unlabel_inputs], dim =1)
                            unlabel_embeddings = self.model(concat_OOD)
                        else:                    
                            unlabel_embeddings = self.model(unlabel_inputs)


                        if select_unlabel:
                            prototypes, in_distribution, OOD_distribution = unlabel_select(train_embeddings, test_embeddings, unlabel_embeddings, self.args, refine)
                        else:
                            prototypes = get_prototypes(train_embeddings, train_targets, self.args, unlabel_embeddings, refine)
                        accuracy, _, _ = get_accuracy3(prototypes, test_embeddings, test_targets, self.args)
                        

                        acc_list.append(accuracy.cpu().data.numpy())
                        pbar.set_description('dataname {} accuracy ={:.4f}'.format(dataname, np.mean(acc_list)))
                        if batch_idx >= self.args.num_valid_batches:
                            break

            avg_accuracy = np.round(np.mean(acc_list), 4)
            acc_dict = {dataname:avg_accuracy}
            logging.debug('Epoch_{}_{}_accuracy_{}'.format(epoch, dataname, avg_accuracy))
            return acc_dict

def main(args):

        all_accdict = {}
        train_loader_list, valid_loader_list, test_loader_list = dataset(args, datanames)

        OOD_dataset = ['Omniglot', 'Vggflower']
        OODtrain_loader_list, OODvalid_loader_list, OODtest_loader_list = dataset(args, OOD_dataset)
        OOD_trainloader = OODtrain_loader_list

        model = PrototypicalNetworkJoint(3,
                                    args.embedding_size,
                                    hidden_size=args.hidden_size)
        model.to(device=args.device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        each_epoch = args.num_epoch
        seqmeta = SequentialMeta(model, optimizer, args=args)

        dataname = []
        for loaderindex, train_loader in enumerate(train_loader_list):
            for epoch in range(each_epoch*loaderindex, each_epoch*(loaderindex+1)):
                print('Epoch {}'.format(epoch))
                dataname.append(list(train_loader.keys())[0])
                
                total_acc = 0.0
                seqmeta.train(train_loader, OOD_trainloader, domain_id = loaderindex)
                epoch_acc = []
                for index, test_loader in enumerate(test_loader_list[:loaderindex+1]):
                    test_accuracy_dict = seqmeta.valid(epoch, test_loader, OOD_trainloader, domain_id = index)
                    epoch_acc.append(test_accuracy_dict)
                    acc = list(test_accuracy_dict.values())[0]
                    total_acc += acc

                avg_acc = total_acc/(loaderindex+1)
                print('average testing accuracy', avg_acc)

                all_accdict[str(epoch)] = epoch_acc
                with open(seqmeta.filepath + '/stats_acc.pickle', 'wb') as handle:
                    pickle.dump(all_accdict, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
        indexlist = []
        for memory in seqmeta.memory_rep:
            indexlist.append(list(memory))



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Prototypical Networks')
    parser.add_argument('--data_path', type=str, default='/data/',
        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--logdir', type=str, default='results')
    parser.add_argument('--num-OODshot1', type=int, default=8,
        help='Number of examples per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-OODshot2', type=int, default=1,
        help='Number of examples per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-shot', type=int, default=5,
        help='Number of examples per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-way', type=int, default=5,
        help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--num_distractor_classes', type=int, default=0,
        help='Number of distractor classes.')
    parser.add_argument('--OODnum-way', type=int, default=5,
        help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--embedding-size', type=int, default=64,
        help='Dimension of the embedding/latent space (default: 64).')
    parser.add_argument('--hidden-size', type=int, default=64,
        help='Number of channels for each convolutional layer (default: 64).')
    parser.add_argument('--output_folder', type=str, default='output/datasset/',
        help='Path to the output folder for saving the model (optional).')
    parser.add_argument('--batch-size', type=int, default=5,
        help='Number of tasks in a mini-batch of tasks (default: 16).')


    parser.add_argument('--MiniImagenet_batch_size', type=int, default=2,
        help='Number of tasks in a mini-batch of tasks for MiniImagenet (default: 4).')
    parser.add_argument('--CIFARFS_batch_size', type=int, default=2,
        help='Number of tasks in a mini-batch of tasks for CIFARFS (default: 4).')
    parser.add_argument('--CUB_batch_size', type=int, default=2,
        help='Number of tasks in a mini-batch of tasks for CUB (default: 4).')
    parser.add_argument('--Aircraft_batch_size', type=int, default=2,
        help='Number of tasks in a mini-batch of tasks for Aircraft (default: 4).')
    parser.add_argument('--Omniglot_batch_size', type=int, default=2,
        help='Number of tasks in a mini-batch of tasks for Omniglot (default: 4).')
    parser.add_argument('--Plantae_batch_size', type=int, default=2,
        help='Number of tasks in a mini-batch of tasks for Aircraft (default: 4).')
    parser.add_argument('--Quickdraw_batch_size', type=int, default=2,
        help='Number of tasks in a mini-batch of tasks for Quickdraw (default: 4).')
    parser.add_argument('--Butterfly_batch_size', type=int, default=2,
        help='Number of tasks in a mini-batch of tasks for Butterfly (default: 4).')
    parser.add_argument('--VGGflower_batch_size', type=int, default=2,
        help='Number of tasks in a mini-batch of tasks for Butterfly (default: 4).')

    parser.add_argument('--solver', type=str, default='OT', choices=['OT'])
    parser.add_argument('--OT_loss', action='store_true',
        help='Use OT_loss.')
    parser.add_argument('--sample_OT', type=int, default=10,
        help='Number of OT samples tasks.')
    parser.add_argument('--num-batches', type=int, default=200,
        help='Number of batches the prototypical network is trained over (default: 100).')
    parser.add_argument('--num_valid_batches', type=int, default=200,
        help='Number of batches the model is trained over (default: 150).')
    parser.add_argument('--num_memory_batches', type=int, default=1,
        help='Number of batches the model is trained over (default: 150).')
    parser.add_argument('--sample', type=int, default=1,
        help='Number of batches the model is trained over (default: 150).')
    parser.add_argument('--memory_limit', type=int, default=100,
        help='Number of batches the model is trained over (default: 150).')
    parser.add_argument('--num-workers', type=int, default=1,
        help='Number of workers for data loading (default: 1).')
    parser.add_argument('--num_query', type=int, default=10,
        help='Number of query examples per class (k in "k-query", default: 15).')
    parser.add_argument('--num_unlabel', type=int, default=5,
        help='Number of unlabeled examples per class.')

    parser.add_argument('--topK', type=int, default=50,
        help='Number of topK unlabeled examples.')
    parser.add_argument('--num_block', type=int, default=4,
        help='Number of convolution block.')
    parser.add_argument('--download', action='store_true',
        help='Download the Omniglot dataset in the data folder.')
    parser.add_argument('--use-cuda', action='store_true',
        help='Use CUDA if available.')
    parser.add_argument('--ground-cost', type=str, default='l2', choices=['l1', 'l2'])
    parser.add_argument('--num_epoch', type=int, default=30,
        help='Number of epochs for meta train.') 
    parser.add_argument('--valid_batch_size', type=int, default=2,
        help='Number of tasks in a mini-batch of tasks for validation (default: 4).')

    args = parser.parse_args()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print('args.device', args.device)
    main(args)




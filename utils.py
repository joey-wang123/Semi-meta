import torch
import random
from torchmeta.utils.prototype import get_prototypes, prototypical_loss
import torch.nn.functional as F
import torch.nn as nn


def get_accuracy(prototypes, embeddings, targets):
    """Compute the accuracy of the prototypical network on the test/query points.

    Parameters
    ----------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape 
        `(meta_batch_size, num_classes, embedding_size)`.
    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the query points. This tensor has 
        shape `(meta_batch_size, num_examples, embedding_size)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has 
        shape `(meta_batch_size, num_examples)`.

    Returns
    -------
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points.
    """
    sq_distances = torch.sum((prototypes.unsqueeze(1)
        - embeddings.unsqueeze(2)) ** 2, dim=-1)

    prototypes -= prototypes.min(-1, keepdim=True)[0]
    prototypes /= prototypes.max(-1, keepdim=True)[0]

    embeddings -= embeddings.min(-1, keepdim=True)[0]
    embeddings /= embeddings.max(-1, keepdim=True)[0]

    norm_distances = torch.sum((prototypes.unsqueeze(1)
        - embeddings.unsqueeze(2)) ** 2, dim=-1)
   
    tau = 1.0
    norm_distances = norm_distances/tau
    softprob = -1.0*F.softmax(norm_distances, dim=-1) * F.log_softmax(norm_distances, dim=-1)
    min_dist, predictions = torch.min(sq_distances, dim=-1)
    return torch.mean(predictions.eq(targets).float()), softprob, predictions



def get_accuracy2(prototypes, embeddings, targets):
    """Compute the accuracy of the prototypical network on the test/query points.

    Parameters
    ----------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape 
        `(meta_batch_size, num_classes, embedding_size)`.
    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the query points. This tensor has 
        shape `(meta_batch_size, num_examples, embedding_size)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has 
        shape `(meta_batch_size, num_examples)`.

    Returns
    -------
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points.
    """
    sq_distances = torch.sum((prototypes.unsqueeze(1)
        - embeddings.unsqueeze(2)) ** 2, dim=-1)

    _, predictions = torch.min(sq_distances, dim=-1)

    return torch.mean(predictions.eq(targets).float()), _, _



def get_accuracy3(prototypes, embeddings, targets, args):
    """Compute the accuracy of the prototypical network on the test/query points.

    Parameters
    ----------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape 
        `(meta_batch_size, num_classes, embedding_size)`.
    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the query points. This tensor has 
        shape `(meta_batch_size, num_examples, embedding_size)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has 
        shape `(meta_batch_size, num_examples)`.

    Returns
    -------
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points.
    """

    num_classes = args.num_way - args.num_distractor_classes
    total_acc = 0.0
    for prototype, embedding, target in zip(prototypes, embeddings, targets):

        sq_distances = torch.sum((prototype
            - embedding.unsqueeze(1)) ** 2, dim=-1)
        _, predictions = torch.min(sq_distances, dim=-1)

        device = sq_distances.device
        label_list = []
        index = 0
        for index in range(num_classes):
            label_list += [index]*args.num_query
            index += 1
        new_label = torch.tensor(label_list).to(device)
        accuracy = torch.mean(predictions.eq(new_label).float())
        total_acc += accuracy

    avg_acc = total_acc/(len(prototypes))
   
    return avg_acc, _, _



def get_num_samples(targets, num_classes, dtype=None):
    batch_size = targets.size(0)
    with torch.no_grad():
        ones = torch.ones_like(targets, dtype=dtype)
        num_samples = ones.new_zeros((batch_size, num_classes))
        num_samples.scatter_add_(1, targets, ones)
    return num_samples



def get_prototypes(embeddings, targets, args, unlabel_embeddings, refine):
    """Compute the prototypes (the mean vector of the embedded training/support 
    points belonging to its class) for each classes in the task.

    Parameters
    ----------
    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the support points. This tensor 
        has shape `(batch_size, num_examples, embedding_size)`.

    targets : `torch.LongTensor` instance
        A tensor containing the targets of the support points. This tensor has 
        shape `(batch_size, num_examples)`.

    num_classes : int
        Number of classes in the task.

    Returns
    -------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape
        `(batch_size, num_classes, embedding_size)`.
    """
    num_classes = args.num_way - args.num_distractor_classes
    num_unlabel = args.num_unlabel

    prototype_list = []
    for embedding, unlabel_embedding in zip(embeddings, unlabel_embeddings):
        prototypes = []
        for ind in range (num_classes):
            class_embedding = embedding[ind*args.num_shot:(ind+1)*args.num_shot]
            ith_map = class_embedding
            ith_map = torch.sum(ith_map, dim=0, keepdim=True) / (args.num_shot)
            prototypes.append(ith_map)
        prototypes = torch.cat(prototypes)
        prototype_list.append(prototypes)

    prototype_list = torch.stack(prototype_list)
    if not refine:
        return prototype_list

    new_prototype_list = []
    for prototype, unlabel_embedding in zip(prototype_list, unlabel_embeddings):
        sq_distances = torch.sum((prototype
        - unlabel_embedding.unsqueeze(1)) ** 2, dim=-1)
        probs = F.softmax(-1.0*sq_distances, dim=-1).unsqueeze(1)

        new_prototypes = []
        for ind in range (num_classes):
            class_prob = probs[:,:,ind]
            class_embedding = prototype[ind, :]

            unlabel_weight = unlabel_embedding*class_prob
            unlabel_map = torch.sum(unlabel_weight, dim=0, keepdim=True)

            ith_map = unlabel_map + class_embedding*args.num_shot
            ith_map = ith_map / (class_prob.sum() + args.num_shot)
            new_prototypes.append(ith_map)
        new_prototypes = torch.cat(new_prototypes)
        new_prototype_list.append(new_prototypes)
    new_prototype_list = torch.stack(new_prototype_list)

    return new_prototype_list
    #return prototype_list



def unlabel_select(support_embeddings, query_embeddings, unlabel_embeddings, args, refine):
   

    num_classes = args.num_way - args.num_distractor_classes
    num_select = args.topK

    prototype_list = []
    for embedding in support_embeddings:
        prototypes = []
        for ind in range (num_classes):
            class_embedding = embedding[ind*args.num_shot:(ind+1)*args.num_shot]
            ith_map = class_embedding
            ith_map = torch.sum(ith_map, dim=0, keepdim=True) / (args.num_shot)
            prototypes.append(ith_map)

        prototypes = torch.cat(prototypes)
        prototype_list.append(prototypes)

    prototype_list = torch.stack(prototype_list)
    if not refine:
        return prototype_list

    new_prototype_list = refine_prototype(prototype_list, query_embeddings, num_classes, args)

    select_embeddings_list = []
    for prototype, unlabel_embedding, query_embedding in zip(new_prototype_list, unlabel_embeddings, query_embeddings):

        sq_distances = torch.sum((prototype
        - unlabel_embedding.unsqueeze(1)) ** 2, dim=-1)
        min_dist, indices = torch.min(sq_distances, dim = 1)


        query_distances = torch.sum((prototype
        - query_embedding.unsqueeze(1)) ** 2, dim=-1)
        min_querydist, indices = torch.min(query_distances, dim = 1)

        querystd = torch.std(min_querydist)
        topkvalues, topkindices = torch.topk(min_dist, num_select, largest = False)

        select_embeddings = []
        selectprototype_list = []
        selectunlabel_list = []
        for indice in topkindices:
            select_embeddings.append(unlabel_embedding[indice])
            sq_distances = torch.sum((prototype-unlabel_embedding[indice].unsqueeze(0).unsqueeze(0)) ** 2, dim=-1)
            min_dist, indices = torch.min(sq_distances, dim = 1)
            selectprototype = prototype[indices]
            selectprototype_list.append(selectprototype.squeeze())
            selectunlabel_list.append(unlabel_embedding[indice])

        selectprototype_list = torch.stack(selectprototype_list)
        selectunlabel_list = torch.stack(selectunlabel_list)

        in_distribution = (selectprototype_list, selectunlabel_list)
        select_embeddings = torch.stack(select_embeddings)
        select_embeddings_list.append(select_embeddings)

        selectprototype_list = []
        selectunlabel_list = []

        for indice in range(unlabel_embedding.size(0)):
            if indice not in topkindices:
                sq_distances = torch.sum((prototype-unlabel_embedding[indice].unsqueeze(0).unsqueeze(0)) ** 2, dim=-1)
                min_dist, indices = torch.min(sq_distances, dim = 1)
                selectprototype = prototype[indices]
                selectprototype_list.append(selectprototype.squeeze())
                selectunlabel_list.append(unlabel_embedding[indice])

        selectprototype_list = torch.stack(selectprototype_list)
        selectunlabel_list = torch.stack(selectunlabel_list)

        OOD_distribution = (selectprototype_list, selectunlabel_list)
    select_refine = refine_prototype(prototype_list, select_embeddings_list, num_classes, args)

    return select_refine, in_distribution, OOD_distribution


def refine_prototype(prototype_list, unlabel_embeddings, num_classes, args):

    new_prototype_list = []
    for prototype, unlabel_embedding in zip(prototype_list, unlabel_embeddings):
        sq_distances = torch.sum((prototype
        - unlabel_embedding.unsqueeze(1)) ** 2, dim=-1)
        probs = F.softmax(-1.0*sq_distances, dim=-1).unsqueeze(1)

        new_prototypes = []
        for ind in range (num_classes):
            class_prob = probs[:,:,ind]
            class_embedding = prototype[ind, :]

            unlabel_weight = unlabel_embedding*class_prob
            unlabel_map = torch.sum(unlabel_weight, dim=0, keepdim=True)

            ith_map = unlabel_map + class_embedding*args.num_shot
            ith_map = ith_map / (class_prob.sum() + args.num_shot)
            new_prototypes.append(ith_map)
        new_prototypes = torch.cat(new_prototypes)
        new_prototype_list.append(new_prototypes)
    new_prototype_list = torch.stack(new_prototype_list)

    return new_prototype_list



def CE_loss( prototypes, queries, labels_IC, args):


    loss_total = 0.0
    num_classes = args.num_way - args.num_distractor_classes
    for proto, query, label in zip(prototypes, queries, labels_IC):
        CE = nn.CrossEntropyLoss().to(args.device)
        diff = query.unsqueeze(1) - proto
        distance = diff.pow(2).sum(dim=2)

        device = distance.device
        label_list = []
        index = 0
        for index in range(num_classes):
            label_list += [index]*args.num_query
            index += 1
        new_label = torch.tensor(label_list).to(device)

        loss_IC = CE(-distance, new_label)
        loss_total += loss_IC
        
    return loss_total

def rep_memory(args, model, memory_train):
        memory_loss =0
        for dataidx, dataloader_dict in enumerate(memory_train):
                for dataname, memory_list in dataloader_dict.items():
                    select = random.choice(memory_list)
                    memory_train_inputs, memory_train_targets = select['train'] 
                    memory_train_inputs = memory_train_inputs.to(device=args.device)
                    memory_train_targets = memory_train_targets.to(device=args.device)
                    if memory_train_inputs.size(2) == 1:
                        memory_train_inputs = memory_train_inputs.repeat(1, 1, 3, 1, 1)
                    memory_train_embeddings = model(memory_train_inputs, dataidx)

                    memory_test_inputs, memory_test_targets = select['test'] 
                    memory_test_inputs = memory_test_inputs.to(device=args.device)
                    memory_test_targets = memory_test_targets.to(device=args.device)
                    if memory_test_inputs.size(2) == 1:
                        memory_test_inputs = memory_test_inputs.repeat(1, 1, 3, 1, 1)

                    memory_test_embeddings = model(memory_test_inputs, dataidx)
                    memory_prototypes = get_prototypes(memory_train_embeddings, memory_train_targets, args.num_way)
                    memory_loss += prototypical_loss(memory_prototypes, memory_test_embeddings, memory_test_targets)

        return memory_loss


def Reservoir_memory(args, model, memory_train):
        memory_loss =0
        for select in memory_train:
            memory_train_inputs, memory_train_targets = select['train'] 
            memory_train_inputs = memory_train_inputs.to(device=args.device)
            memory_train_targets = memory_train_targets.to(device=args.device)
            if memory_train_inputs.size(2) == 1:
                memory_train_inputs = memory_train_inputs.repeat(1, 1, 3, 1, 1)
            memory_train_embeddings = model(memory_train_inputs, dataidx)

            memory_test_inputs, memory_test_targets = select['test'] 
            memory_test_inputs = memory_test_inputs.to(device=args.device)
            memory_test_targets = memory_test_targets.to(device=args.device)
            if memory_test_inputs.size(2) == 1:
                memory_test_inputs = memory_test_inputs.repeat(1, 1, 3, 1, 1)

            memory_test_embeddings = model(memory_test_inputs, dataidx)
            memory_prototypes = get_prototypes(memory_train_embeddings, memory_train_targets, args.num_way)
            memory_loss += prototypical_loss(memory_prototypes, memory_test_embeddings, memory_test_targets)

        return memory_loss
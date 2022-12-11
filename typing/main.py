from options import args
import random
import numpy as np
import torch
import csv
import sys
from utils import prepare_instance_bert, MyDataset, my_collate_bert, prepare_instance_generate, my_collate_generate, get_all_types
from models import pick_model
from torch.utils.data import DataLoader
import os
import time
from train_test import train, test, generate
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule

if __name__ == "__main__":


    if args.random_seed != 0:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)

    print(args)

    csv.field_size_limit(sys.maxsize)

    model = pick_model(args)

    if args.mode == "train":

        prepare_instance_func = prepare_instance_bert

        train_instances = prepare_instance_func(args.DATA_DIR + "/crowd/train_m.json",
                                                args, args.max_seq_length)
        print("train_instances {}".format(len(train_instances)))
        dev_instances = prepare_instance_func(args.DATA_DIR + "/crowd/dev.json",
                                              args, args.max_seq_length)
        print("dev_instances {}".format(len(dev_instances)))
        test_instances = prepare_instance_func(args.DATA_DIR + "/crowd/test.json",
                                               args, args.max_seq_length)
        print("test_instances {}".format(len(test_instances)))

        collate_func = my_collate_bert

        train_loader = DataLoader(MyDataset(train_instances), args.batch_size, shuffle=True, collate_fn=collate_func)
        dev_loader = DataLoader(MyDataset(dev_instances), args.batch_size, shuffle=False, collate_fn=collate_func)
        test_loader = DataLoader(MyDataset(test_instances), args.batch_size, shuffle=False, collate_fn=collate_func)

        no_decay = ['bias', 'gamma', 'beta']
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() if n not in no_decay], 'weight_decay_rate': 0.01},
            {'params': [p for n, p in model.named_parameters() if n in no_decay], 'weight_decay_rate': 0.0}
        ]

        optimizer = AdamW(
            # model.parameters(),
            optimizer_parameters,
            lr=args.lr,
            weight_decay=args.weight_decay,
            eps=1e-8
        )

        total_steps = len(train_loader) * args.n_epochs

        # Create the learning rate scheduler.
        scheduler = WarmupLinearSchedule(
            optimizer,
            warmup_steps=0.1,
            t_total=total_steps
        )

        test_only = args.test_model is not None

        if not test_only:
            for epoch in range(args.n_epochs):
                if epoch == 0 and not args.test_model:
                    dir_name = args.model + args.dir_name
                    model_dir = os.path.join(args.MODEL_DIR, dir_name)
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)
                epoch_start = time.time()
                losses = train(args, model, optimizer, epoch, args.gpu, train_loader, scheduler)
                loss = np.mean(losses)
                epoch_finish = time.time()
                print("epoch finish in %.2fs, loss: %.4f" % (epoch_finish - epoch_start, loss))
                fold = 'dev'
                # test on dev
                evaluation_start = time.time()
                test(args, model, fold, args.gpu, dev_loader)
                evaluation_finish = time.time()
                print("evaluation finish in %.2fs" % (evaluation_finish - evaluation_start))
                if epoch == args.n_epochs - 1:
                    print("last epoch: testing on dev and test sets")
                    test(args, model, "test", args.gpu, test_loader)
                    torch.save(model.state_dict(), model_dir + '/' + model.name)

    if args.mode == "generate":
        model_dir = args.load_model
        pretrained_model_path = args.load_model + '/' + model.name
        state_dict = torch.load(pretrained_model_path, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict)
        prepare_instance_func = prepare_instance_generate
        collate_func = my_collate_generate
        WORLDS = [
            'american_football',
            'doctor_who',
            'fallout',
            'final_fantasy',
            'military',
            'pro_wrestling',
            'starwars',
            'world_of_warcraft',
            'coronation_street',
            'muppets',
            'ice_hockey',
            'elder_scrolls',
            'forgotten_realms',
            'lego',
            'star_trek',
            'yugioh'
        ]
        all_types = get_all_types(args.DATA_DIR + "/crowd/types.txt")
        id_to_types = {k: v for k, v in enumerate(all_types)}
        types_to_id = {v: k for k, v in enumerate(all_types)}
        for world in WORLDS:
            instances = prepare_instance_func(args.DATA_DIR + "/zeshel/documents/" + world + ".json",
                                                    args, args.max_seq_length)
            print("instances {}".format(len(instances)))

            loader = DataLoader(MyDataset(instances), 1, shuffle=False, collate_fn=collate_func)

            generate(args, model, "test", args.gpu, loader, world)

    sys.stdout.flush()

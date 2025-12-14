import argparse


def args_parse():
    parser = argparse.ArgumentParser(description='Data-Model params')

    parser.add_argument("--data_name", help="CIFAR10/CIFAR100", type=str, default="CIFAR10",
                        choices=["CIFAR10", "CIFAR100"])
    parser.add_argument("--seed", help="Seed", type=int, default=0)

    # for model training
    parser.add_argument("--batch_size", help="batch size of training", type=int, default=64)
    parser.add_argument("--train_epochs", help="model training epoch", type=int, default=20)
    parser.add_argument("--lr_max", help="Max LR", type=float, default=0.1)
    parser.add_argument("--lr_min", help="Min LR", type=float, default=0.)

    parser.add_argument("--exact_mode", help="model exaction attack type", type=str, default="teacher",
                        choices=['zero-shot', 'prune', 'fine-tune', 'extract-label', 'extract-logit', 'distillation', 'teacher'])
    parser.add_argument("--pseudo_labels", help="alternative dataset", type=int, default=0, choices=[0, 1])

    # for embedding generation
    parser.add_argument("--feature_mode", help="embedding compute type", type=str, default="MinAD", choices=['MinAD', 'MinAD_KRM'])
    parser.add_argument("--num_samples", help="number of sample for embedding", type=int, default=200)
    parser.add_argument("--num_attack_iter", help="number of iterations of MinAD", type=int, default=20)
    parser.add_argument("--num_evals_boundary", help="number of queries per iteration of MinAD", type=int, default=1000)
    parser.add_argument("--attack_lr_begin", help="initial LR of MinAD", type=float, default=16)
    parser.add_argument("--attack_lr_end", help="LR lower threshold of MinAD", type=float, default=0.2)

    # for KR
    parser.add_argument("--num_center", help="number of KRM", type=int, default=1)

    # for dataset inference
    parser.add_argument("--a_num", help="number of sample for dataset inference", type=int, default=10)


    return parser


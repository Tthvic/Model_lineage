import argparse


def args_parse():
    parser = argparse.ArgumentParser(description='Data-Model params')

    parser.add_argument("--data_name", help="Dataset", type=str, default="CIFAR10", choices=["CIFAR10", "CIFAR100"])
    parser.add_argument("--seed", help="Seed", type=int, default=0)

    parser.add_argument("--batch_size", help="batch size on model training and testing", type=int, default=64)
    parser.add_argument("--train_epochs", help="model training epoch", type=int, default=25)

    parser.add_argument("--exact_mode", help="model exaction attack type", type=str, default="teacher",
                        choices=['teacher', 'fine-tune', 'retrain', 'prune', 'SA', 'DA-LENET', 'DA-VGG'])

    parser.add_argument("--feature_mode", help="embedding compute type", type=str, default= 'MinAD_KRM', choices=['MinAD', 'MinAD_KRM'])

    # parser.add_argument("--feature_mode", help="embedding compute type", type=str, default="MinAD", choices=['MinAD', 'MinAD_KRM'])
    parser.add_argument("--num_samples", help="number of samples for embedding", type=int, default=100)
    parser.add_argument("--num_attack_iter", help="number of iterations of MinAD", type=int, default=20)

    parser.add_argument("--num_evals_boundary", help="number of queries per iteration of MinAD", type=int, default=200)
    parser.add_argument("--attack_lr_begin", help="initial LR of MinAD", type=float, default=16)
    parser.add_argument("--attack_lr_end", help="LR lower threshold of MinAD", type=float, default=0.2)

    parser.add_argument("--i_num", help="number of sample for ipguard", type=int, default=100)


    return parser


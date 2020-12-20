# Standard library
import sys

# Gym
import gym

# SophieDL
import sophiedl as S

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog = "python test.py",
        description = "Test script for mah RL babies."
    )

    parser.add_argument(
        "--factory",
        type = str,
        action = "append",
        help = "Which runner factory to use, --list-factories shows available factories."
    )

    parser.add_argument(
        "--list-factories",
        action = "store_true",
        help = "Lists available runner factories."
    )

    parser.add_argument(
        "--episode-count",
        type = int,
        action = "append",
        help = "How many episodes to run."
    )

    parser.add_argument(
        "--epoch-count",
        type = int,
        action = "append",
        help = "How many epochs to run."
    )

    parser.add_argument(
        "--tensorboard-output-dir",
        type = str,
        action = "append",
        help = "Output directory for TensorBoard data."
    )

    args = parser.parse_args()

    if args.list_factories:
        print("Available factories (use with --factory FACTORY):")

        for i in S.list_runner_factories():
            print("  {0}".format(i.__name__))
        
        sys.exit(1)
    
    if args.factory and len(args.factory) > 0:
        if len(args.factory) > 1:
            sys.stderr.write("error: please specify only one factory\n")
            sys.exit(1)
        elif len(args.factory) == 1:
            if not args.factory[0] in S.__dict__:
                sys.stderr.write("error: no such factory {0}\n".format(repr(args.factory[0])))
                sys.exit(1)
            else:
                factory_type = S.__dict__[args.factory[0]]

                if not isinstance(factory_type, type) or not issubclass(factory_type, S.RunnerFactoryBase):
                    sys.stderr.write("error: no such factory {0}".format(repr(args.factory[0])))
                    sys.exit(1)
                
                factory = factory_type()

                hyperparameter_set = factory.create_default_hyperparameter_set()

                if args.episode_count and len(args.episode_count) > 0:
                    hyperparameter_set["episode_count"] = args.episode_count[-1]

                if args.epoch_count and len(args.epoch_count) > 0:
                    hyperparameter_set["epoch_count"] = args.epoch_count[-1]

                factory.create_runner(
                    hyperparameter_set = hyperparameter_set,
                    tensorboard_output_dir = args.tensorboard_output_dir[-1] if args.tensorboard_output_dir and len(args.tensorboard_output_dir) > 0 else None
                ).run()

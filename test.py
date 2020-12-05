import sys

import gym

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
        help = "Which agent/environment factory to use, --list-factories shows available factories."
    )

    parser.add_argument(
        "--list-factories",
        action = "store_true",
        help = "Lists available agent/environment factories."
    )

    parser.add_argument(
        "--episode-count",
        type = int,
        action = "append",
        help = "How many episodes to run."
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

        for i in S.list_agent_factories():
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

                if not isinstance(factory_type, type) or not issubclass(factory_type, S.AgentEnvironmentFactoryBase):
                    sys.stderr.write("error: no such factory {0}".format(repr(args.factory[0])))
                    sys.exit(1)
                
                if not args.episode_count or len(args.episode_count) == 0:
                    sys.stderr.write("error: missing required argument --episode-count EPISODE_COUNT\n")
                    sys.exit(1)
                
                factory_type().create_runner(
                    episode_count = args.episode_count[-1],
                    tensorboard_output_dir = args.tensorboard_output_dir[-1] if args.tensorboard_output_dir and len(args.tensorboard_output_dir) > 0 else None
                ).run()

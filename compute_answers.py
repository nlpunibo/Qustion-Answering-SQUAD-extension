import argparse
from core.ner_extension import ner
from core.classifier_extension import classifier
from core.convolutional_classifier_extension import conv_classifier
from core.multiplechoice_extension import multiple_choice


def get_args_from_command_line():
    parser = argparse.ArgumentParser(description='Argument parser')
    parser.add_argument('--extension',  help='Chose the extension you want to use',
                        choices=['ner', 'multiple_choice', 'classifier', 'convolutional_classifier'],  type=str)
    parser.add_argument('--test', help="Valid path to a json file, squad_v1 or squad_v2", type=str)
    args = parser.parse_args()
    return args


def main():
    # Get args from command line
    args = get_args_from_command_line()

    if args.extension == "ner":
        ner(args.test,"./datasets/ner_weights.json")
    elif args.extension == "multiple_choice":
        multiple_choice(args.test)
    elif args.extension == "classifier":
        classifier(args.test)
    else:
        conv_classifier(args.test)

if __name__ == '__main__':
    main()

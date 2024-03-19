import argparse 
from vlmeval.config import supported_VLM

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, nargs='+', required=True)
    parser.add_argument('--text', type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--nproc", type=int, default=4, required=False)
    parser.add_argument("--verbose", action='store_true')
    args = parser.parse_args()
    return args

def main():
	args = parse_args()
	model_name = args.model
	model = supported_VLM[model_name]() 
	response = model.generate(prompt=args.text, image_path=args.image_path, dataset='demo')

	print(response)


if __name__ == '__main__':
    main()

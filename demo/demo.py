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
	template = 'Answer with the detailed reasoning process for obtaining the answer, incorporating professional terms when necessary, to address inquiries regarding scene description, key elements identification, and potential activities in the provided image.'
	text = args.text + template
	image_path = args.image_path
	response = model.generate(prompt=text, image_path=image_path, dataset='demo')

	print(response)


if __name__ == '__main__':
    main()

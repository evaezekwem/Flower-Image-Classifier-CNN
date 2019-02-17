import argparse
import helper_predict as hp
import json


def main(image_path, checkpoint, top_k, cat_to_name_json, on_gpu):
	
	if (not hp.file_exist(image_path)) or (not hp.file_exist(checkpoint)):
		print("Either the image or model check point does not exist")
		
		
	with open(cat_to_name_json, 'r') as f:
		cat_to_name = json.load(f)
	
	
	result = hp.load_checkpoint(checkpoint)
	model = result[1]
	class_to_idx = result[3]
	
	prediction = hp.predict(image_path, model, top_k, on_gpu)
	names_probs = hp.get_names_probs(prediction, cat_to_name, class_to_idx)
	
	
	print("Names: {}".format(names_probs[0]))
	print("Probabilities: {}".format(names_probs[1]))
	
		



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='A simple app predict an image using a trained neural net')
	
	parser.add_argument('image_path', action='store',
	                    help="Specifies the path to the image file to predict it's class")
	parser.add_argument('checkpoint', action='store',
	                    help="Specifies the path to the trained model's checkpoint")
	parser.add_argument('--top_k', action='store', type=int, default=5, dest='top_k',
	                    help="Specifies the number of classes to return")
	parser.add_argument('--category_names', action='store', default='cat_to_name.json', dest='cat_to_name_json',
	                    help="Specifies the json file containing the category name and mapping for all categories")
	parser.add_argument('--gpu', action="store_false", default=True, dest='on_gpu')
	
	args = parser.parse_args()
	
	print("#################....Parameters....##################")
	print("save_dir.........................{}".format(args.image_path))
	print("arch.............................{}".format(args.checkpoint))
	print("learning_rate....................{}".format(args.top_k))
	print("hidden_units.....................{}".format(args.cat_to_name_json))
	print("gpu..............................{}".format(args.on_gpu))
	print("#################....Parameters....##################")
	
	
	main(args.image_path, args.checkpoint, args.top_k, args.cat_to_name_json, args.on_gpu)
	
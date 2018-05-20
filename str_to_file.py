import argparse
import os
import obj_tools

def get_arguments():
    parser = argparse.ArgumentParser(description="SMILES string to Wavefront .obj conversion by file search.")
    parser.add_argument("in_folder", type=str, help="The folder containing the input .obj files.")
    parser.add_argument("in_string", type=str, help="The smiles string encoding the graph to search for.")
    return parser.parse_args()

def process_folder(folder_name, query_word):
    for item_name in os.listdir(folder_name):
        subfolfer_name = os.path.join(folder_name, item_name)
        if os.path.isdir(subfolfer_name):
            process_folder(subfolfer_name, query_word)
        if not item_name.endswith("_coll_graph.obj") and item_name.endswith(".obj"): 
            current_str = obj_tools.obj2strings(folder_name + "/" + item_name)
            current_words = current_str.split("\n")            
            for w in current_words:
                if w == query_word:
                    print(query_word + " found in " + item_name)
                    return True
    return False

def main():
    args = get_arguments()

    in_smiles_string = args.in_string
    success = process_folder(args.in_folder, in_smiles_string)
    if not success:
        print("Did not find " + in_smiles_string)

if __name__ == "__main__":

    main()
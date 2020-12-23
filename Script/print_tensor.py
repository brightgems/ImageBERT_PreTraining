import sys
import torch

if __name__=="__main__":
    argv=sys.argv
    filepath=argv[1]

    tensor=torch.load(filepath,map_location=torch.device("cpu"))

    print("Filepath: {}".format(filepath))    
    print("Size: {}".format(tensor.size()))
    print(tensor)

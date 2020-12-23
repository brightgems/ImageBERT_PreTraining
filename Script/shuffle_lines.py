import argparse
import random

def main(input_filepath:str,save_filepath:str):
    with open(input_filepath,"r",encoding="utf_8") as r:
        lines=r.read().splitlines()

    shuffled_lines=random.sample(lines,len(lines))

    with open(save_filepath,"w",encoding="utf_8") as w:
        for line in shuffled_lines:
            w.write(line+"\n")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--input_filepath",type=str)
    parser.add_argument("--save_filepath",type=str)
    args=parser.parse_args()

    main(args.input_filepath,args.save_filepath)

import argparse
import random

def main(input_filepath:str,save_filepath:str,create_negative_prob:float):
    with open(input_filepath,"r",encoding="utf_8") as r:
        filenames=r.read().splitlines()

    data_list=[]
    for filename in filenames:
        #正例は必ず含む。
        data={}
        data["input_ids_filename"]=filename
        data["roi_filename"]=filename
        data_list.append(data)

        #負例の作成
        if random.random()<create_negative_prob:
            data={}
            data["input_ids_filename"]=filename
            data["roi_filename"]=random.choice(filenames)
            data_list.append(data)

    with open(save_filepath,"w",encoding="utf_8") as w:
        for data in data_list:
            w.write("{}\t{}\n".format(data["input_ids_filename"],data["roi_filename"]))
    
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--input_filepath",type=str)
    parser.add_argument("--save_filepath",type=str)
    parser.add_argument("--create_negative_prob",type=float,default=0.2)
    args=parser.parse_args()

    main(args.input_filepath,args.save_filepath,args.create_negative_prob)

import os, json
import pandas as pd

def json2csv(in_json, in_folder = None, out_folder = None , name_prefix = None):
    if(name_prefix is None):
        name_prefix = os.path.basename(in_json)



    if (in_folder is None):
        in_folder = os.path.dirname(in_json)

    if (out_folder is None):
        out_folder = in_folder

        # load json
    with open(in_json) as f:
        dat = json.load(f)

    keys = list(dat.keys())
    out_dict = {}
    out_dict['folder'] = in_folder
    out_dict['json'] = os.path.basename(in_json)
    for k in keys:
        if(isinstance(dat[k],list)):
            # save as csv
            output_csv_name= name_prefix + "_" + k + ".csv"
            datafarme = pd.DataFrame(dat[k])
            datafarme.to_csv(os.path.join(out_folder, output_csv_name), index = False, index_label = "index_"+k)
            out_dict[k] = os.path.join(out_folder, output_csv_name)

    return out_dict


def many_jsons2csv(in_json_list, out_folder, out_prefix = "comb", in_folder_list = None, out_folder_list = None , name_prefix_list = None):

    extra_columns = ['folder', 'json'] # columns to add to each table to know the folder
    skip_append = ['categories'] # no need to concat this tables
    skip_keys = ['licenses'] # no need

    N = len(in_json_list)
    if(in_folder_list is None):
        in_folder_list = [None]*N
    if (out_folder_list is None):
        out_folder_list = [None] * N
    if (name_prefix_list is None):
        name_prefix_list = [None] * N

    if(not os.path.exists(out_folder)):
        os.mkdir(out_folder)

    out_dicts_list = []
    for i,js in enumerate(in_json_list):
        out_d = json2csv(in_json=js, in_folder=in_folder_list[i], out_folder=out_folder_list[i] , name_prefix=name_prefix_list[i])
        out_dicts_list.append(out_d)

    #my_keys = ['categories', 'images','annotations']
    my_keys = []

    for kk in out_dicts_list[0].keys():
        s = out_dicts_list[0][kk]
        if(".csv" in s) or (".txt" in s):
            if(str(kk) not in skip_keys):
                my_keys.append(str(kk))



    all_df = {}
    for k in my_keys:
        for i in range(N):
            temp_df = pd.read_csv(out_dicts_list[i][k])
            for ec in extra_columns:
                print(out_dicts_list[i][ec])
                temp_df[ec] = out_dicts_list[i][ec]
            if( k not in all_df):
                all_df[k] = temp_df.copy()
            else:
                if(k not in skip_append):
                    all_df[k] = pd.concat([all_df[k],temp_df.copy()],ignore_index=True)

    for k in my_keys:
        df = all_df[k]
        output_csv_name = out_prefix + '_' + k + '.csv'
        df.to_csv(os.path.join(out_folder, output_csv_name), index = False, index_label = "index_"+k)






if __name__=="__main__":
    print("test")
    in_folders = ["/home/borisef/datasets/racoon/test", "/home/borisef/datasets/racoon/train"]
    in_jsons = ["data.json"]*len(in_folders)

    for i,js in enumerate(in_jsons):
        in_jsons[i] =  os.path.join(in_folders[i], in_jsons[i])

    if(0):
        for jso, fo in zip(in_jsons,in_folders):
            json2csv(in_json=jso, in_folder = fo, out_folder = fo, name_prefix = jso)

    many_jsons2csv(in_json_list = in_jsons, out_folder = "/home/borisef/datasets/try1", out_prefix="try1")
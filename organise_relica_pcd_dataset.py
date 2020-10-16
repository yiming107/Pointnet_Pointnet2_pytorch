"""
Organise the replica dataset for classification
Author: Yiming
"""
import os
import lib.filetool as ft
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR

data_path = os.path.join(ROOT_DIR, "data")
dataset_name = "replica"
dataset_path = os.path.join(data_path, dataset_name)


reject_class = ["anonymize_text", "floor", "wall", "non-plane", "anonymize_picture", "undefined", "ceiling", "panel", "anony"]


def generate_lists_obj():
    id_list = []
    id_list_selected = []
    filelist_each_class_all = {}
    file_num_each_class_all = {}

    list_rooms = ft.grab_directory(dataset_path, fullpath=True)

    for room in list_rooms:
        list_pcds = ft.grab_files(room, fullpath=True, extra = "*.pcd")
        for pcd_path in list_pcds:
            pcd_name = os.path.split(pcd_path)[1]
            class_name = pcd_name.split("_")[1][:-4]
            if class_name not in reject_class:
                print(class_name)
                if class_name not in id_list:
                    id_list.append(class_name)
                    # create also a key entry for that class
                    filelist_each_class_all[class_name] = []

                filelist_each_class_all[class_name].append(pcd_path)
    print("There are in total {:d} classes in".format(len(id_list)))

    shape_name_path = os.path.join(dataset_path, "{}_shape_names.txt".format(dataset_name))
    shape_stat_path = os.path.join(dataset_path, "{}_shape_stats.txt".format(dataset_name))
    shape_name_all_path = os.path.join(dataset_path, "{}_shape_names_all.txt".format(dataset_name))

    print("1. save files corresponding to a single class into their corresponding file")
    for class_name in filelist_each_class_all:
        print("\tsave for class: {} ... ".format(class_name))
        filelist_path_each_class = os.path.join(dataset_path, "filelist_{}.txt".format(class_name))
        filelist_temp = filelist_each_class_all[class_name]
        file_num_each_class_all[class_name] = len(filelist_temp)
        if file_num_each_class_all[class_name] > 20:
            id_list_selected.append(class_name)
        if os.path.exists(filelist_path_each_class):
            print("\tfile exists already, skip ...")
        else:
            with open(filelist_path_each_class, 'w') as f:
                f.writelines("%s\n" % each_path for each_path in filelist_temp)

    print("2. save the id name list of all and its statistics")
    if os.path.exists(shape_name_all_path):
        print("\tfile exists already, skip ...")
    else:
        with open(shape_name_all_path, 'w') as f:
            f.writelines("%s\n" % id_name for id_name in id_list)

    if os.path.exists(shape_stat_path):
        print("\tfile exists already, skip ...")
    else:
        with open(shape_stat_path, 'w') as f:
            f.writelines("{}, {:d}\n".format(id_name,file_num_each_class_all[id_name]) for id_name in file_num_each_class_all)

    print("3. save the id name list of classes of more than 20 instances")
    if os.path.exists(shape_name_path):
        print("\tfile exists already, skip ...")
    else:
        with open(shape_name_path, 'w') as f:
            f.writelines("%s\n" % id_name for id_name in id_list_selected)

    return id_list_selected


def organise_train_test_split(id_list, ratio=0.8):
    filelist_train_all = []
    filelist_test_all = []
    print("1. merge all the file lists with train/test split")
    for class_name in id_list:
        filelist_path_each_class = os.path.join(dataset_path, "filelist_{}.txt".format(class_name))
        with open(filelist_path_each_class, "r") as f:
            file_list = f.read().split("\n")[:-1] # remove the last one of "" string
            num_items = len(file_list)
            train_num = int(float(num_items)*ratio)
            train_list_once = random.sample(file_list, k=train_num)
            test_list_once = list(set(train_list_once) ^ set(file_list))
            filelist_train_all = filelist_train_all + train_list_once
            filelist_test_all = filelist_test_all + test_list_once

    print("2. shuffle all the file lists in train and test")
    random.shuffle(filelist_train_all)
    random.shuffle(filelist_test_all)

    print("3. shuffle the file lists in train and test, again")
    random.shuffle(filelist_train_all)
    random.shuffle(filelist_test_all)

    print("4. save the file lists to the train and test")
    train_path = os.path.join(dataset_path, "{}_train.txt".format(dataset_name))
    if os.path.exists(train_path):
        print("\tfile exists already, skip ...")
    else:
        with open(train_path, 'w') as f:
            f.writelines("%s\n" % file_name for file_name in filelist_train_all)

    test_path = os.path.join(dataset_path, "{}_test.txt".format(dataset_name))
    if os.path.exists(test_path):
        print("\tfile exists already, skip ...")
    else:
        with open(test_path, 'w') as f:
            f.writelines("%s\n" % file_name for file_name in filelist_test_all)


if __name__ == '__main__':

    id_list_selected = generate_lists_obj()

    organise_train_test_split(id_list_selected)


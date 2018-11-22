import csv
import pathlib
import numpy as np

ROOT = pathlib.Path("../celeba_data")
DATA_ROOT = str(ROOT / "img_align_celeba")
ANNO_ROOT = str(ROOT / "Anno" / "new_attr_celeba.txt")
EVAL_ROOT = str(ROOT / 'Anno' / "list_eval_partition.txt")
LANDMARK_ROOT = str(ROOT/'all_landmark.json')
CelebA_HQ_ROOT = str(ROOT/'CelebA-HQ')
CelebA_HQ_MAPPING_ROOT = str(ROOT/'image_list.txt')


def extract_attr_dict(annoroot, sel_attrs=None):
    with open(annoroot) as infile:
        lines = infile.readlines()
        attr_names = lines[1].split()
        attr_dict = {}
        if sel_attrs is None:
            sel_attr_idx = np.arange(0, len(attr_names))
        else:
            sel_attr_idx = [attr_names.index(sel_attr)
                            for sel_attr in sel_attrs
                            if sel_attr in attr_names]
            sel_attr_idx = np.array(sel_attr_idx)
            assert len(sel_attr_idx) == len(sel_attrs)
        for line in lines[2:]:
            splits = line.split()
            attr_val = [int(x) for x in splits[1:]]
            attr_val = np.array(attr_val)
            attr_val[attr_val == -1] = 0
            attr_dict[splits[0]] = attr_val[sel_attr_idx]
        return attr_dict


def extract_train_val_fnames(partitionroot):
    with open(partitionroot) as infile:
        lines = infile.readlines()
        lines = [line.strip() for line in lines]
        train_fnames = [line.split()[0] for
                        line in lines if line.split()[1] == '0']
        val_fnames = [line.split()[0] for line in lines if line.split()[1] == '1']
        return train_fnames, val_fnames


def parse_landmarks(annoroot):
    with open(annoroot) as infile:
        lines = infile.readlines()
    lines = lines[2:]
    landmark_dict = {}
    for line in lines:
        splits = line.split()
        name = splits[0]
        landmarks = np.array([float(i) for i in splits[1:]])
        landmark_dict[name] = landmarks
    return landmark_dict


def find_nearest_in_dict(name, landmark_dict):
    dist, nn = 1000, None
    candidates = np.random.choice(list(landmark_dict.keys()), 500, replace=False)
    for another_file in candidates:
        if another_file == name:
            continue
        cur_dist = np.mean((landmark_dict[another_file] - landmark_dict[name]) ** 2)
        if cur_dist < dist:
            dist, nn = cur_dist, another_file
    return nn


def build_nn_file(names1, names2, landmark_dict):
    nn_pairs = []
    new_landmark_dict = {name: landmark_dict[name] for name in names2}
    for idx, name1 in enumerate(names1):
        print(idx)
        new_landmark_dict[name1] = landmark_dict[name1]
        nn = find_nearest_in_dict(name1, new_landmark_dict)
        del new_landmark_dict[name1]
        nn_pairs.append((name1, nn))
    with open("nn.txt", "w") as outfile:
        for pair in nn_pairs:
            outfile.write("{} {}\n".format(pair[0], pair[1]))


def parse_group(infilename):
    with open(infilename) as infile:
        lines = infile.readlines()
    group1 = []
    group2 = []
    for line in lines:
        name1, name2 = line.strip().split()
        group1.append(name1)
        group2.append(name2)
    return dict(zip(group1, group2))


def parse_opt(fname):
    with open(fname) as infile:
        lines = infile.readlines()
    kv_pair = {}
    for line in lines[1:-1]:
        key, val = line.strip().split(":")
        try:
            kv_pair[key] = eval(val)
        except Exception:
            kv_pair[key] = val.strip()

    class Option:
        pass

    option = Option()
    for k in kv_pair:
        setattr(option, k, kv_pair[k])
    return option


def extract_celeba_hq_mapping():
    mapping = {}
    with open(CelebA_HQ_MAPPING_ROOT) as infile:
        reader = csv.DictReader(infile, delimiter=' ')
        for row in reader:
            mapping[row['orig_file']] = int(row['idx'])
    return mapping

def visualize_opt_flow(flow_numpy,size):
    #hsv = np.zeros((size,size,3),dtype=np.uint8)
    #hsv[...,1] = 255
    #mag,ang = cv2.cartToPolar(flow_numpy[...,0],flow_numpy[...,1])
    #hsv[...,0] = ang * 180 / np.pi /2
    #hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    #rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    #return rgb
    u = flow_numpy[...,0]
    v = flow_numpy[...,1]
    from util.flow_util import viz_flow
    return viz_flow(u,v)


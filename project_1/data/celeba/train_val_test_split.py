import config

train_list = []
val_list = []
test_list = []
with open(config.PARTITION_FILE) as f:
    for line in f.readlines():
        row = line.strip().split(" ")
        img_name = row[0]
        label = row[-1]
        if label == "0":
            train_list.append(img_name)
        if label == "1":
            val_list.append(img_name)
        if label == "2":
            test_list.append(img_name)

with open(config.ATTRIBUTE_FILE, encoding="utf-8") as f:
    for line in f.readlines():
        row = line.strip().split(" ")
        row = [i.replace("-1", "0") for i in row if i != ""]
        if len(row) == 41:
            img_name = row[0]
            if img_name in train_list:
                with open(config.TRAIN_ATTRIBUTE_LIST, "a", encoding="utf-8") as ff:
                    ff.writelines(" ".join(row) + "\n")
            if img_name in val_list:
                with open(config.VAL_ATTRIBUTE_LIST, "a", encoding="utf-8") as ff:
                    ff.writelines(" ".join(row) + "\n")
            if img_name in test_list:
                with open(config.TEST_ATTRIBUTE_LIST, "a", encoding="utf-8") as ff:
                    ff.writelines(" ".join(row) + "\n")
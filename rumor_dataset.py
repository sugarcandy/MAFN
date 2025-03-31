from torch.utils.data import Dataset


class Rumor_data(Dataset):
    def __init__(self, X_train_tid, X_train, train_content, y_train):
        self.id = X_train_tid
        self.train = X_train
        self.y = y_train
        self.content = train_content

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return self.id[item], self.train[item], self.content[item], self.y[item]


class Rumor_dataWithEntity(Dataset):
    def __init__(self, X_train_tid, X_train, train_content, X_train_entity, y_train):
        self.id = X_train_tid
        self.train = X_train
        self.y = y_train
        self.content = train_content
        self.entity = X_train_entity

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return self.id[item], self.train[item], self.content[item], self.entity[item], self.y[item]


class Rumor_data_test(Dataset):
    def __init__(self, X_train_tid, X_train, train_content):
        self.id = X_train_tid
        self.train = X_train
        # self.y = y_train
        self.content = train_content

    def __len__(self):
        return len(self.id)

    def __getitem__(self, item):
        return self.id[item], self.train[item], self.content[item]


class Rumor_data_test_entity(Dataset):
    def __init__(self, X_train_tid, X_train, X_train_entity, train_content):
        self.id = X_train_tid
        self.train = X_train
        # self.y = y_train
        self.content = train_content
        self.entity = X_train_entity

    def __len__(self):
        return len(self.id)

    def __getitem__(self, item):
        return self.id[item], self.train[item], self.content[item], self.entity[item]


class MR_Dataset(Dataset):
    def __init__(self, X_id, X_train_content, X_train_img, y_train):
        self.id = X_id
        self.content = X_train_content
        self.img = X_train_img
        self.label = y_train

    def __len__(self):
        return len(self.id)

    def __getitem__(self, item):
        return self.id[item], self.content[item], self.img[item], self.label[item]



class MR_Dataset_Test(Dataset):
    def __init__(self, X_id, X_content, X_img):
        self.id = X_id
        self.content = X_content
        self.img = X_img
        # self.label = y_train

    def __len__(self):
        return len(self.id)

    def __getitem__(self, item):
        return self.id[item], self.content[item], self.img[item]

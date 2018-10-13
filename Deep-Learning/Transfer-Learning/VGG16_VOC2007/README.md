# Introduce
In train.py, there are two method for train.

## 1. train()
If you want use this method, you need to classify data firstly(In data_pretrain is two scrip for this).And you can run train()

## 2. train\_vgg11\_with\_pretrained()
train\_vgg11\_with\_pretrained(dataloaders\_dict, num\_classes, batch\_size, num\_epochs, feature\_extract)

**Parameters:**

dataloaders_dict: type=dictionary key="train, val" value=torch.utils.data.DataLoader()

num_classes: type=int

batch_size: type=int

num_epochs: type=int

feature_extract: type=bool
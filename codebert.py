import pandas as pd
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, AdamW

# Load your data into a pandas DataFrame
df = pd.read_csv("https://raw.githubusercontent.com/jvirma/code-smell-prototype/master/src/long_method.csv", sep=",")

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

# Define a custom dataset class
class CodeDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        code = self.data.loc[idx, 'codesample']
        label = self.data.loc[idx, 'labels']

        # Encode the code sample with the tokenizer
        inputs = self.tokenizer(code, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        inputs = {key: tensor.squeeze(0) for key, tensor in inputs.items()}
        return inputs, label

    def __len__(self):
        return len(self.data)

# Create an instance of your custom dataset
dataset = CodeDataset(df, tokenizer)

# Split your dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])



# Initialize dataloaders for your training and validation datasets
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=8)

# Load the model
model = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base")

# Set up the optimizer and learning rate scheduler
optimizer = AdamW(model.parameters(), lr=0.0001)
num_training_steps = 1000
num_warmup_steps = 100
lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-5, total_steps=num_training_steps, anneal_strategy='linear', cycle_momentum=False, base_momentum=0.85, max_momentum=0.95, div_factor=25.0, pct_start=float(num_warmup_steps)/float(num_training_steps))


# Fine-tune the model
model.train()
for epoch in range(3): # Replace 3 with the number of epochs you want to run
    for batch in train_dataloader:
        inputs = batch[0]
        labels = batch[1]
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

    # Evaluate the model on the validation dataset after each epoch
    model.eval()
    num_correct = 0
    num_total = 0
    for batch in val_dataloader:
        inputs = batch[0]
        labels = batch[1]
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            num_correct += torch.sum(predictions == labels)
            num_total += len(labels)

    accuracy = float(num_correct) / num_total
    print("Epoch {}: Accuracy: {:.2f}%".format(epoch+1, accuracy*100))
    print("Loss: {:.2f}".format(loss))

    # Put the model back in training mode
    model.train()        
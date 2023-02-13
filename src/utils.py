import string, random, sys, unidecode
import torch
from moore import MooreMachineLSTM

# get all ascii characters
all_characters = string.printable
n_characters = len(all_characters)


class Helper:
    def __init__(self):
        self.chunk_len = 200
        self.all_characters = string.printable
        self.n_characters = len(self.all_characters)
        self.file = unidecode.unidecode(open('data/names.txt').read())
        self.num_epochs = 2000
        self.batch_size = 1
        self.lr = 0.001
        self.hidden_size = 100
        self.num_layers = 2
        self.print_every = 100
        self.plot_every = 10
        self.all_losses = []

    # convert a string into it tensor representation based on the all_characters index
    def char_to_tensor(self, characters):
        # create a tensor of zeros with the length of the characters
        tensor = torch.zeros(len(characters)).long()
        # for each character in the string,
        # get the index of the character in the all_characters string
        # and set the char index in the tensor to the index of the character
        for idx in range(len(characters)):
            tensor[idx] = self.all_characters.index(characters[idx])
        return tensor

    # get a random batch of data from the training da
    def get_random_batch(self):
        # get a random starting index for the chunk of text
        start_idx = random.randint(0, len(self.file) - self.chunk_len)
        # get the chunk end index based on the start index and the chunk length
        end_idx = start_idx + self.chunk_len + 1
        # get the chunk of text
        text_str = self.file[start_idx:end_idx]
        text_input = torch.zeros(self.batch_size, self.chunk_len)
        text_target = torch.zeros(self.batch_size, self.chunk_len)
        for i in range(self.batch_size):
            text_input = self.char_to_tensor(text_str[:-1])
            text_target = self.char_to_tensor(text_str[1:])
        return text_input.long(), text_target.long()

    def moore_generator(self, prime_str='A', predict_len=100, temperature=0.8):

        pass

    def train(self):
        # create a moore machine
        moore = MooreMachineLSTM(self.n_characters, self.hidden_size, self.n_characters, self.num_layers)
        # create the loss function
        loss_fn = torch.nn.CrossEntropyLoss()
        # create the optimizer
        loss_fn, optimizer = moore.loss_optimizer()
        optimizer = torch.optim.Adam(moore.parameters(), lr=self.lr)
        print(f'Training starting...')
        for epoch in range(1, self.num_epochs + 1):
            # get the input and target tensors
            inputs, targets = self.get_random_batch()
            hidden, cell = moore.init_hidden(self.batch_size)
            # zero the gradients
            optimizer.zero_grad()
            loss = 0
            for c in range(self.chunk_len):
                output, (hidden, cell) = moore(inputs[:, c], (hidden, cell))
                loss += loss_fn(output, targets[:, c])
            # perform backpropagation
            loss.backward()
            # update the weights, biases, and loss
            optimizer.step()
            loss = loss.item() / self.chunk_len
            # if the epoch is a multiple of the print every
            if epoch % self.print_every == 0:
                # print the epoch and the loss
                print(f'epoch: {epoch}, loss: {loss.item()}')
            # if the epoch is a multiple of the plot every
            if epoch % self.plot_every == 0:
                # add the loss to the all losses list
                self.all_losses.append(loss.item())

            writer.add_scalar('Loss/train', loss, epoch)
        # save the model
        torch.save(moore.state_dict(), 'moore.pt')


# print(f'Helper', torch.zeros(len('adfd')).long().shape)
# tensor = torch.zeros(len('adfd')).long()
# char = 'adfd'
# for idx in range(len(char)):
#     print(f'idx', idx)
#     tensor[idx] = all_characters.index(char[idx])
#
# print(f'tensor', tensor)
file = unidecode.unidecode(open('../data.txt').read())
# # print(f'file', file)
# print(f'len(file)', len(file))
# for i in file:
#     print(f'i', i)
#     break
#
# print(f'len(file)', len(file) - 200)
# v = random.randint(0, len(file) - 200)
# b = v + 200 + 1
# print(f'v', v)
# print(f'b', b)
# text_str = file[v:b]
# text_input = text_str[:-1]
# text_target = text_str[1:]
# print(f'text_str', text_str)
# print(f'text_input', text_input)
# print(f'text_target', text_target)


d = torch.zeros(1, 5)

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TokenLabelConverter(object):
    """ Convert between text`-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        # [GO] for the start token of the attention decoder. [s] for end-of-sentence token.
        self.SPACE = '[s]'
        self.GO = '[GO]'
        #self.MASK = '[MASK]'

        #self.list_token = [self.GO, self.SPACE, self.MASK]
        self.list_token = [self.GO, self.SPACE]
        self.character = self.list_token + list(character)
        

        self.dict = {word: i for i, word in enumerate(self.character)}
        self.batch_max_length = 25 + len(self.list_token) # 이거 설명좀 해주세여!! 

    def encode(self, text):
        """ convert text-label into text-index.
        """
        length = [len(s) + len(self.list_token) for s in text]  # +2 for [GO] and [s] at end of sentence.
        batch_text = torch.LongTensor(len(text), self.batch_max_length).fill_(self.dict[self.GO])
        
        for i, t in enumerate(text):
            # 수정 이전
            # c = [[char] for char in t[0]]
            # txt = [self.GO] + list(c) + [self.SPACE]
            # txt = [self.dict[char] if j == 0 or j == int(len(txt)-1) else self.dict[char[0]] for j,char in enumerate(txt)]
            
            # 수정 후 - 수정 전과 동작은 동일
            txt = [self.GO] + list(t[0]) + [self.SPACE]
            
            txt = [self.dict[char] for char in txt]
            
            batch_text[i][:len(txt)] = torch.LongTensor(txt)  # batch_text[:, 0] = [GO] token

        return batch_text

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            text = ''.join([self.character[i] for i in text_index[index, :]])
            texts.append(text)
        return texts
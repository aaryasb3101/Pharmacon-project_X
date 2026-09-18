"""
tokenizer.py
============

SMILES Tokenizer for Molecular Transformer

Pipeline:
    SMILES
        ↓
    Regex Tokenization
        ↓
    Vocabulary Mapping
        ↓
    Token IDs
"""

import re #regular expression library, used to identify patterns inside smiles string
import torch #using pytorch to create tensors

# Special Tokens (user defined, for sequence processing)
PAD_TOKEN = "[PAD]" #padding token
CLS_TOKEN = "[CLS]" #classification/summary token (later the transformer will take the
                    #final representation of this token and use it as a molecular embedding)
SEP_TOKEN = "[SEP]" #marks the end of SMILES sequence
UNK_TOKEN = "[UNK]" #unknown, if a token is not in the vocabulary we map it to unk to prevent the model from crashing

# SMILES Regular Expression (takes a regex pattern and recompiles it into a reusable regex object)
SMILES_REGEX = re.compile(
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])"
) #? says that the letter is optional so c and cl are mentioned in cl?


class SmilesTokenizer:
    """
    Regex-based tokenizer for SMILES strings.
    """

    def __init__(self):

        self.token2idx = {} #create an empty dictionary (map token to integer id)
        self.idx2token = {} #creates the dictionary for reverse mapping

    # Tokenize
    @staticmethod
    def tokenize(smiles: str):
        #makes a python list called tokens
        tokens = SMILES_REGEX.findall(smiles) #find all matches of regex inside teh smiles string

        if "".join(tokens) != smiles: #joins the tokes back together and compare it to original smiles
                                      #if it doesn't match, the regex tokenizer did not find some matches
            raise ValueError(
                f"Tokenizer failed to fully parse SMILES:\n{smiles}"
            )

        return tokens

    # Build Vocabulary
    def build_vocab(self, smiles_list): 

        specials = [ #put the special tokens into the vocabulary first so they have deterministic ids
            PAD_TOKEN,
            CLS_TOKEN,
            SEP_TOKEN,
            UNK_TOKEN,
        ]

        vocab = set() #creating a set because we dont want the tokens to repeat

        for smiles in smiles_list:
            vocab.update(self.tokenize(smiles))

        all_tokens = specials + sorted(vocab) #we sort it to get deterministic ids, so if reproduced it gives the same ids to tokens

        #creating the dictionary for token to id, so basically what token has what id
        self.token2idx = {
            token: idx
            for idx, token in enumerate(all_tokens)
        }

        #creating reverse mapping
        self.idx2token = {
            idx: token
            for token, idx in self.token2idx.items()
        }

    # Properties
    @property
    def vocab_size(self):
        return len(self.token2idx)

    @property
    def pad_id(self):
        return self.token2idx[PAD_TOKEN]

    @property
    def cls_id(self):
        return self.token2idx[CLS_TOKEN]

    @property
    def sep_id(self):
        return self.token2idx[SEP_TOKEN]

    @property
    def unk_id(self):
        return self.token2idx[UNK_TOKEN]

    # Encode (we transform our smiles string toids)
    def encode(self, smiles: str):

        tokens = self.tokenize(smiles)

        ids = [self.cls_id] #starts with CLS id

        ids.extend(
            self.token2idx.get(token, self.unk_id) #gives integer token ids and UNK id if not available
            for token in tokens
        )

        ids.append(self.sep_id) #adds sep id

        return ids #gives a list of integer ids 

   
    # Decode
    def decode(self, ids):

        tokens = []

        for idx in ids:

            if idx == self.pad_id:
                continue

            tokens.append(
                self.idx2token.get(idx, UNK_TOKEN)
            )

        return tokens

    
    # Encode Batch
    def encode_batch(
        self,
        smiles_list,
        max_len=None,
    ):

        encoded = [
            self.encode(smiles)
            for smiles in smiles_list
        ]

        if max_len is None:
            max_len = max(len(seq) for seq in encoded)

        batch_size = len(encoded)

        #create pytorch tensors for the transformer
        input_ids = torch.full(
            (batch_size, max_len),
            self.pad_id, #create a pytorch tensor filled entirely with pad ids
            dtype=torch.long,
        )

        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=torch.bool, #creates another tensor (8, 256) initially all false that tells
                              #the transformer which  position contains real tokens
        )

        for i, ids in enumerate(encoded):

            length = min(len(ids), max_len)

            #takes the encoded ids and inserts them into the correct row of input_ids
            input_ids[i, :length] = torch.tensor(
                ids[:length],
                dtype=torch.long,
            )
            #this tells the transformer that the first "length" are actually tokens
            attention_mask[i, :length] = True

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
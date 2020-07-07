#!/usr/bin/env python
# coding: utf-8

# In[10]:


from gensim import utils
import gensim.models

class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""

    def __iter__(self):
        for line in open('myCorpus.txt'):
            # assume there's one document per line, tokens separated by whitespace
            yield utils.simple_preprocess(line)



sentences = MyCorpus()
model = gensim.models.Word2Vec(sentences=sentences, size=256, min_count=2)
model.save('modelPath/gensim-model')
#     new_model = gensim.models.Word2Vec.load(temporary_filepath)


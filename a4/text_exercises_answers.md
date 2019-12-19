Below are answers to the question, which are part of the assignment described [here.](http://web.stanford.edu/class/cs224n/assignments/a4.pdf)
* 1(g):

    The masks set the input to the softmax for 'pad' tokens to -inf. Then the output of softmax for these tokens equals 0. 
    This cause the tokens to be ignored when calculating attention output. 
    It is necessary because for the attention we care only about actual context of a sentence (represented by values of a hidden state in the Encoder),
    because it brings some information to the model. 'pad' tokens do not brings much relevant information.
    Including 'pad' tokens for short sentences would cause weight assigned to hidden states corresponding to normal words to be smaller. 
* 1(j):
    * Dot product attention: doesn't introduce additional parameters to the model, requires the 2 vector to be of the same shape 
    * Multiplicative attention: introduce additional parameters to the model, enables to calculate attention between vectors of different shape
    * Additive attention: slower to calculate, more operations, more learnable parameters
* 2(a):
    * Source Sentence: Aqu´ı otro de mis favoritos, “La noche estrellada”.
    Reference Translation: So another one of my favorites, “The Starry Night”.
    NMT Translation: Here’s another favorite of my favorites, “The Starry Night”.
        1. Double favourite.
        2. Two parts of the English sentence with favourite make sense separately so they are highly probable to be returned by a model.
        3. Make a model to notice that a similar word is already in the text and make probability of using it again lower.
        Increase beam size in beam search.

    * Source Sentence: Ustedes saben que lo que yo hago es escribir para los ni˜nos, y,
    de hecho, probablemente soy el autor para ni˜nos, ms ledo en los EEUU.
    Reference Translation: You know, what I do is write for children, and I’m probably America’s
    most widely read children’s author, in fact.
    NMT Translation: You know what I do is write for children, and in fact, I’m probably the
    author for children, more reading in the U.S.
        1. Too direct translation, incorrect gramatically in English.
        2. Model might have put too much weight on specific words.
        3. Increase number of parameters in the model, increase beam size.
    
    * Source Sentence: Un amigo me hizo eso – Richard Bolingbroke.
    Reference Translation: A friend of mine did that – Richard Bolingbroke.
    NMT Translation: A friend of mine did that – Richard <unk>
        1. Unknown token.
        2. Not all words can in included in the vocabulary.
        3. Detect named entitites and copy them directly in the model.
    * Source Sentence: Solo tienes que dar vuelta a la manzana para verlo como una
    epifan´ıa.
    Reference Translation: You’ve just got to go around the block to see it as an epiphany.
    NMT Translation: You just have to go back to the apple to see it as a epiphany.
        1. Sentence does not make sense.
        2. Manzana has 2 meanings: block and apple. The model picked the wrong one.
        3. Use context dependent embeddings of source language. 
    * Source Sentence: Ella salv´o mi vida al permitirme entrar al ba˜no de la sala de
    profesores.
    Reference Translation: She saved my life by letting me go to the bathroom in the teachers’
    lounge.
    NMT Translation: She saved my life by letting me go to the bathroom in the women’s room.
        1. The output sentence does not contain information about teachers.
        2. Model picked the popular term that matched the context. 
    * Source Sentence: Eso es m´as de 100,000 hect´areas.
    Reference Translation: That’s more than 250 thousand acres.
    NMT Translation: That’s over 100,000 acres.
        1. Both translations seems to be incorrect.
        2. Model maybe picked the more common word - acres? 
        

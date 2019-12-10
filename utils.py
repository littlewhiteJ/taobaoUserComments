import os

def cluster(candidateWords, nClusters, vecModelPath, vectorSize, outputsPath):
    from sklearn.cluster import KMeans
    from gensim.models import Word2Vec
    import numpy as np

    model = Word2Vec.load(vecModelPath)

    vectors = []
    # print(candidateWords)
    for words in candidateWords:
        v = np.zeros(vectorSize)
        for word in words:
            try:
                v += np.array(model.wv[word])
            except:
                pass
        vectors.append(v)



    # print(vectors)
    kmeans = KMeans(n_clusters=nClusters, random_state=0).fit(vectors)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    min_dis = np.zeros(nClusters)
    real_centers = np.zeros(nClusters)
    dis_list = []

    for i in range(nClusters):
        min_dis[i] = np.linalg.norm(vectors[0] - centers[i])


    for i in range(0, len(candidateWords)):
        dis = np.linalg.norm(vectors[i] - centers[labels[i]])
        dis_list.append(dis)
        if dis < min_dis[labels[i]]:
            min_dis[labels[i]] = dis
            real_centers[labels[i]] = i
    
    with open(outputsPath, 'w') as f:
        for i in range(nClusters):
            f.write('center ' + str(i) + ':\n')
            write_words(candidateWords[int(real_centers[i])], f)
            f.write('neighbor words:\n')
            cnt = 0
            for j in range(len(labels)):
                if labels[j] == i:
                    cnt += 1
                    write_words(candidateWords[j], f)
                    f.write('dis: ' + str(dis_list[j]) + '\n')
            f.write('center ' + str(i) + ' has ' + str(cnt) + ' neighbors!\n')
            f.write('center ' + str(i) + ' end\n')
            f.write('\n')

    center_words = []
    for i in range(nClusters):
        center_words.append(candidateWords[int(real_centers[i])])

    return center_words

def candidateSelectPyltp(testWordList, ltpPath, sentimentWordsPath, selectedPath):
    with open(sentimentWordsPath, 'r') as f:
        sentimentWords = f.readlines()

    sentimentSet = set()
    for i in range(len(sentimentWords)):
        sentimentSet.add(sentimentWords[i].strip())

    from pyltp import Parser
    from pyltp import Postagger

    par_model_path = os.path.join(ltpPath, 'parser.model') 
    pos_model_path = os.path.join(ltpPath, 'pos.model')

    parser = Parser() 
    parser.load(par_model_path)

    postagger = Postagger() 
    postagger.load(pos_model_path)

    all_words_coll = []
    for words in testWordList:
        postags = postagger.postag(words)
        arcs = parser.parse(words, postags)

        # relation = " ".join("%d:%s" % (arc.head, arc.relation) for arc in arcs)

        hed_num = 0
        for i in range(len(arcs)):
            if arcs[i].relation == 'HED':
                hed_num = i + 1
        
        if hed_num == 0:
            continue
        words_coll = []
        
        if postags[hed_num - 1] != 'a':
            continue
        if words[hed_num - 1] not in sentimentSet:
            continue

        if hed_num == 2:
            if arcs[0].relation == 'ADV':
                words_coll.append(words[0])
                words_coll.append(words[1])
        elif hed_num > 2:
            if arcs[hed_num - 2].relation == 'ADV' and arcs[hed_num - 3].relation == 'SBV':
                words_coll.append(words[hed_num - 3])
                words_coll.append(words[hed_num - 2])
                words_coll.append(words[hed_num - 1])
        else:
            continue


        if len(words_coll) != 0:
            all_words_coll.append(words_coll)

    parser.release() 
    postagger.release() 

    if selectedPath != '':
        with open(selectedPath, 'w') as f:
            for words in all_words_coll:
                write_words(words, f)
    
    return all_words_coll

def candidateSelectJieba(testSentences, stopWordsPath, sentimentWordsPath, userDict, selectedPath):
    with open(sentimentWordsPath, 'r') as f:
        sentimentWords = f.readlines()
    sentimentSet = set()
    for i in range(len(sentimentWords)):
        sentimentSet.add(sentimentWords[i].strip())

    with open(stopWordsPath, 'r') as f:
        stopWords = f.readlines()
    stopSet = set()
    for i in range(len(stopWords)):
        stopSet.add(stopWords[i].strip())

    import jieba
    import jieba.posseg as pseg
    if userDict != '':
        jieba.load_userdict(userDict)

    all_words_coll = []
    for sentence in testSentences:
        words = pseg.cut(sentence.strip())

        ws = []
        ts = []

        cflag = 0
        for word, tag in words:
            if word.isdigit():
                cflag = 1
                break
            if word in stopSet:
                continue
            ws.append(word)
            ts.append(tag)
        if cflag:
            continue

        words_coll = []
        
        if 'a' not in ts:
            continue

        idx = ts.index('a')
        # idx = idx[0]
        if ws[idx] not in sentimentSet:
            continue

        if idx == 1 and ts[0] == 'n':
            words_coll.append(ws[0])
            words_coll.append(ws[1])
        if idx > 1:
            if ts[idx - 1] == 'd':
                if ts[idx - 2] == 'n' or ts[idx - 2] == 'v':
                    words_coll.append(ws[idx - 2])
                    words_coll.append(ws[idx - 1])
                    words_coll.append(ws[idx])

        if len(words_coll) != 0:
            all_words_coll.append(words_coll)

    # sentiment
    if selectedPath != '':
        with open(selectedPath, 'w') as f:
            for words in all_words_coll:
                write_words(words, f)

    return all_words_coll

def word2Vec(trainWordList, vectorSize, vecModelPath):
    from gensim.models import Word2Vec

    model = Word2Vec(trainWordList, 
                 size=50, 
                 window=5,
                 min_count=1, 
                 workers=2)

    if vecModelPath != '':
        model.save(vecModelPath)
    
    # return model



def segment(sentences, ltpPath, stopWordsPath, userDict, method, segPath):
    cws_model_path = os.path.join(ltpPath, 'cws.model')
    if method == 'pyltp':
        from pyltp import Segmentor
        segmentor = Segmentor()
        segmentor.load(cws_model_path)
    else:
        import jieba
        if userDict != '':
            jieba.load_userdict(userDict)

    with open(stopWordsPath, 'r') as f:
        stopWords = f.readlines()
    stopSet = set()
    for i in range(len(stopWords)):
        stopSet.add(stopWords[i].strip())

    word_lib = []

    for sentence in sentences:
        wl = []
        if method == 'pyltp':
            word_list = list(segmentor.segment(sentence))
        else: # jieba
            word_list = jieba.lcut(sentence.strip())

        cflag = 0
        for word in word_list:
            if word.isdigit():
                cflag = 1
                break
            if word in stopSet:
                continue
            wl.append(word)
        
        if cflag:
            continue

        if len(wl) == 0:
            continue
        word_lib.append(wl)
        
    if segPath != '':
        with open(segPath, 'w') as f:
            for words in word_lib:
                write_words(words, f)

    if method == 'pyltp':
        segmentor.release()
    return word_lib

def preSegment(sentences, preSegPath):
    import re
    sentence_list = []
    for sentence in sentences:
        #print(sentence)
        sen_list = re.split(r'，|。| |、', sentence.strip())
        #print(sen_list)
        #break
        
        for sen in sen_list:
            if sen.strip() != '':
                sentence_list.append(sen)
    if preSegPath != '':
        with open(preSegPath, 'w') as f:
            for sentence in sentence_list:
                f.write(sentence)
                f.write('\n')

    return sentence_list

def write_words(wl, f):
    for w in wl:
        f.write(w)
        f.write(' ')
    f.write('\n')           

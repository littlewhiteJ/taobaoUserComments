import json
from utils import *


def word2vec_g(config):
    ltpPath = config['ltpPath']
    stopWordsPath = config['stopWordsPath']
    segmentMethod = config['segmentMethod']

    vectorSize = config['vectorSize']
    vecModelPath = config['vecModelPath']

    trainPath = config['trainPath']
    preSegTrainPath = config['preSegTrainPath']
    segTrainPath = config['segTrainPath']


    with open(trainPath, 'r') as trainf:
        trainSentences = trainf.readlines()

    trainSentences = preSegment(trainSentences, preSegTrainPath)
    trainWordList = segment(trainSentences, ltpPath, stopWordsPath, segmentMethod, segTrainPath)
    word2Vec(trainWordList, vectorSize, vecModelPath)

    # print(trainWordList)

def cluster_g(config):
    ltpPath = config['ltpPath']
    stopWordsPath = config['stopWordsPath']
    segmentMethod = config['segmentMethod']

    selectedPath = config['selectedPath']
    selectMethod = config['selectedMethod']
    sentimentWordsPath = config['sentimentWordsPath']
    vecModelPath = config['vecModelPath']
    vectorSize = config['vectorSize']
    nClusters = config['nClusters']

    testPath = config['testPath']
    preSegTestPath = config['preSegTestPath']
    segTestPath = config['segTestPath']
    outputsPath = config['outputsPath']
    


    with open(testPath, 'r') as testf:
        testSentences = testf.readlines()
    
    testSentences = preSegment(testSentences, preSegTestPath)

    if selectMethod == 'pyltp':
        testWordList = segment(testSentences, ltpPath, stopWordsPath, segmentMethod, segTestPath)
        candidateWords = candidateSelectPyltp(testWordList, ltpPath, sentimentWordsPath, selectedPath)
    else:
        candidateWords = candidateSelectJieba(testSentences, stopWordsPath, sentimentWordsPath, selectedPath)

    tags = cluster(candidateWords, nClusters, vecModelPath, vectorSize, outputsPath)

    for tag in tags:
        print(''.join(tag))

if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)

    word2vec_g(config)
    cluster_g(config)



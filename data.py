import os
from collections import Counter
import zipfile
import numpy as np
from nltk.tokenize.treebank import TreebankWordTokenizer
import re

import xml.etree.ElementTree as ET

def get_lines(fname, xml_bool=False):
    if not xml_bool:
        if os.path.isfile(fname):
            with open(fname) as f:
                lines = f.readlines()
        else:
            raise("[!] Data %s not found" % fname)

        line_counter = 0
        dict_lines = {}
        for line in lines:
            dict_lines[line_counter] = line
            line_counter += 1

        return dict_lines, None
    else:
        dict_lines = {}
        target_attribs = {}
        counter = 0
        for sent in ET.parse(fname).getroot().findall('sentence'):
            sent_text = sent.find('text').text
            # convert all the words to lower case
            sent_text = sent_text.lower()
            dict_lines[counter] = sent_text
            target_attribs[counter] = []
            for aterm in sent.find('aspectTerms') or []:
                target_attribs[counter].append(aterm)
            counter += 1
        return dict_lines, target_attribs


def populate_wrd2idx(words, dict_lines, count, word2idx):
    if len(count) == 0:
        count.append(['<eos>', 0])

    #count[0][1] += len(lines)
    count[0][1] += len(dict_lines.keys())
    count.extend(Counter(words).most_common())

    if len(word2idx) == 0:
        word2idx['<eos>'] = 0

    for word, _ in count:
        if word not in word2idx:
            word2idx[word] = len(word2idx)



def read_data(fname, count, word2idx):
    dict_lines, target_attribs = get_lines(fname)


    words = []
    for counter, line in dict_lines.items():
        words.extend(line.split())


    populate_wrd2idx(words, dict_lines, count, word2idx)

    data = list()
    for counter, line in dict_lines.items():
        for word in line.split():
            index = word2idx[word]
            data.append(index)
        data.append(word2idx['<eos>'])

    print("Read %s words from %s" % (len(data), fname))
    return data


def get_embedding_dict(embd_pth):
    # Read the zip file in py for glove
    embed_dict = {}
    zf_glove = zipfile.ZipFile(embd_pth)
    ifile = zf_glove.open(zf_glove.infolist()[0])
    for line in zf_glove.readlines():
        word = line.strip().split[0]
        embed_dict[word] = np.array([float(val) for val in line.strip().split()[1:]])
    ifile.close(); zf_glove.close()
    return embed_dict



def split_data(data, trgt_aspect, trgt_Y, trgt_pos, split_frac=0.2):
    train_data=[]; train_trgt_aspect=[]; train_trgt_Y=[]; train_trgt_pos=[];
    test_data=[]; test_trgt_aspect=[]; test_trgt_Y=[]; test_trgt_pos=[];
    for i in xrange(len(data)):
        if np.random.uniform(0, 1) > split_frac:
            train_data.append(data[i])
            train_trgt_aspect.append(trgt_aspect[i])
            train_trgt_Y.append(trgt_Y[i])
            train_trgt_pos.append(trgt_pos[i])
        else:
            test_data.append(data[i])
            test_trgt_aspect.append(trgt_aspect[i])
            test_trgt_Y.append(trgt_Y[i])
            test_trgt_pos.append(trgt_pos[i])
    return train_data, train_trgt_aspect, train_trgt_Y, train_trgt_pos, \
        test_data, test_trgt_aspect, test_trgt_Y, test_trgt_pos



def get_context_maskNpad(data, mem_size):
    mask = []
    for line in data:
        orig_len = len(line)
        mask_line = [1]*orig_len
        mask_line.extend([0]*(mem_size-orig_len))
        mask.append(mask_line)
        line.extend([0]*(mem_size-orig_len))
        #NOTE:TODO: should we not use o indx as eos?
    return mask


def get_dataNembedding(fname, embed_dict, indxd_embedn, embedn_count,
        word2idx, multiwrd_indx_cmpnnt):
    # Only get those words that are in the train/test/valid set

    # read the xml doc and get the sentence to convert into wrdindx
    # then read each of the aspectTerms to get the target aspects
    # Each aspect will be one datapoint?  -- yes,

    # get embed for train, tets and val
    # simultanouesly get the wrd_indx, and indexed_emebedn,
    # and finally at the end, convert indexed_embedn to numpy array embedding
    dict_lines, target_attribs = get_lines(fname, True)

    words = []
    _treebank_word_tokenizer = TreebankWordTokenizer()
    wrdpos_dict = {}
    max_sent_len = 0
    for counter, line in dict_lines.items():
        #words.extend([ wrd.lower().strip() for wrd in line.split()])
        tokens_pos_tupl = [(token, pos+1) for pos, token in \
                enumerate(_treebank_word_tokenizer.tokenize(line.strip()))]
        wrdpos_dict[counter] = dict(tokens_pos_tupl)
        words.extend([token for token in
            _treebank_word_tokenizer.tokenize(line.strip())])

    populate_wrd2idx(words, dict_lines, embedn_count, word2idx)

    #for line in lines:
    data = []
    trgt_aspect=[]
    trgt_Y = []
    trgt_pos = []
    dup_counter = 0
    for counter, line in dict_lines.items():
        indxd_line = []
	print line
	# get the apect and related attributes such as positions etc.
	list_line = list(line)
    	trgtNsenti = []
        dup_flag = False
    	for aterm in target_attribs[counter]:
    	    print aterm.attrib

            ## For the code below using split doesnt work due to
            ## indetation symbols -- hence use tekenizer. And the
            ## only way tokenzier will work is by assuming one space between
            ## words
    	    #charPrev = ' '; cntr = 0; wrd_pos = 0
    	    #while cntr < len(line):
    	    #    if charPrev==' ' and line[cntr]!=' ':
    	    #        wrd_pos += 1
    	    #    if cntr == int(aterm.attrib["from"]):
    	    #        break
    	    #    charPrev = line[cntr]
    	    #    cntr += 1
    	    ## the above code is for calulcating word position
    	    ## This wrd_pos doesnt take care of multi word targets

            # NOTE: we assume that the aspect is present only once
            # our wrd_pos will break if this assumption is not true

            # Check for wheter the aspect is present multiple times:
            aspect = aterm.attrib["term"].lower()
            dup_count = sum(1 for _ in re.finditer(r'\b%s\b' % re.escape(aspect), line))
            if dup_count > 1:
                print "=========================ASPECT DUPLICATED========================"
                print aspect, "\t", line
                print "=========================ASPECT DUPLICATED========================"
                dup_flag = True
            aspect_tkns = _treebank_word_tokenizer.tokenize(aspect)
            if len(aspect_tkns)>1:
                aspect_wrd = [wrd for wrd in aspect_tkns if len(wrd)>1][0]
                wrd_pos = wrdpos_dict[counter][aspect_wrd]
            else:
                if aspect in wrdpos_dict[counter]:
                    wrd_pos = wrdpos_dict[counter][aspect]
                else:
                    rel_tkn = [tkn for tkn in _treebank_word_tokenizer.tokenize(line) \
                                    if len(re.split('\W+', tkn))>1]
                    wrd = [tkn for tkn in rel_tkn if aspect in re.split('\W+', tkn)][0]
                    wrd_pos = wrdpos_dict[counter][wrd]
            trgtNsenti.append((aspect, aterm.attrib["polarity"], wrd_pos,
                len(wrdpos_dict[counter])))
    	    for indx in range(int(aterm.attrib["from"]), int(aterm.attrib["to"])):
    	        list_line[indx] = ''
                # this needs that when we read thsi sentence back we do a
                # strip() on each word -- meh
    	print trgtNsenti

        #for word in line.split():
        for word in _treebank_word_tokenizer.tokenize(line.strip()):
            index = word2idx[word]
            indxd_line.append(index)

        # TODO: remove target wrd from context

	# The code below also takes care of each target
	for attrb_tpl in trgtNsenti:
            target, polarity, position, sent_len = attrb_tpl
	    trgt_indcs = []
            trgt_indx = None
            trgt_wrd_list = []
            for trgt_wrd in _treebank_word_tokenizer.tokenize(target): #target.split():
                # the tokenizer above takes care of multi-word targets
                if trgt_wrd in word2idx:
                    trgt_indcs.append(word2idx[trgt_wrd])
                    trgt_wrd_list.append(trgt_wrd)
                else:
                    rel_tkn = [tkn for tkn in _treebank_word_tokenizer.tokenize(line) \
                                    if len(re.split('\W+', tkn))>1]
                    wrd = [tkn for tkn in rel_tkn if trgt_wrd in re.split('\W+', tkn)][0]
                    print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~", trgt_wrd, wrd
                    # This is to take care of weird chars in multi-word target
                    trgt_indcs.append(word2idx[wrd])
                    trgt_wrd_list.append(wrd)

            if len(trgt_indcs)==1:
                trgt_indx = trgt_indcs[0]
            else:
                #NOTE: should we join using space
                multiwrd_aspect = " ".join(trgt_wrd_list)
                multiwrd_indx = None
                if multiwrd_aspect in word2idx:
                    multiwrd_indx = word2idx[multiwrd_aspect]
                else:
                    multiwrd_indx = len(word2idx)
                    word2idx[multiwrd_aspect] = multiwrd_indx
                multiwrd_indx_cmpnnt[multiwrd_indx] = trgt_indcs
                trgt_indx = multiwrd_indx


            #NOTE: due to multiple use of _treebank_word_tokenizer the indexes
            # above might be different from context indexes
	    #NOTE: the above is to take care of multi word targets
	    data.append(indxd_line)
	    #trgt_aspect.append(trgt_indcs)
	    trgt_aspect.append(trgt_indx)
	    trgt_Y.append(polarity)
	    trgt_pos.append((position, sent_len))
            max_sent_len = max(max_sent_len, len(indxd_line))

        dup_counter += (1 if dup_flag else 0)

    print("Read %s setences, %s words from %s" % (len(data), \
            sum([len(line) for line in data]), fname))
    print "dup_counter %f" % dup_counter
    return max_sent_len, data, trgt_aspect, trgt_Y, trgt_pos


def get_embedding_dict(embd_pth):
    # Read the zip file in py for glove
    embed_dict = {}
    zf_glove = zipfile.ZipFile(embd_pth)
    ifile = zf_glove.open(zf_glove.infolist()[0])
    for line in ifile.readlines():
        values = line.strip().split()
        word = values[0]
        embed_dict[word] = np.asarray(values[1:], dtype=np.float32)
    return embed_dict


def get_nparray_embedding(embed_dict, nrm_std, word2idx,  multiwrd_indx_cmpnnt):
    #(train_trgt_aspect, valid_trgt_aspect, test_trgt_aspect) = cnsldtd_aspect
    nparr_embed_list = []
    embed_dim = len(embed_dict.values()[0])
    sorted_wrdindx = sorted(word2idx.items(), key=lambda x: x[1], reverse=False)
    for wrd, indx in sorted_wrdindx:
        if indx not in multiwrd_indx_cmpnnt:
            weights = embed_dict.get(wrd, np.random.normal(scale=nrm_std, size=embed_dim))
            nparr_embed_list.append(weights)
        else:
            new_vec = np.zeros(embed_dim)
            for trgt_indx in multiwrd_indx_cmpnnt[indx]:
                new_vec += embed_dict.get(wrd, np.random.normal(scale=nrm_std, \
                                    size=embed_dim))
            weights = new_vec/(len(multiwrd_indx_cmpnnt[indx])*1.)
            nparr_embed_list.append(weights)
    #multiwrd_indx_dict = {}
    #multiwrd_indx_cntr = len(nparr_embed_list)-1
    #for data_aspect in cnsldtd_aspect:
    #    for aspect_multiwrd_list in data_aspect if len(aspect_multiwrd_list)>1:
    #        new_wrd = " ".join(aspect_multiwrd_list)
    #        if new_wrd in multi_wrd_indx_dict:
    #            continue
    #        new_vec = np.zero(embed_dim)
    #        for indx in aspect_multiwrd_list:
    #            new_vec += nparr_embed_list[indx]
    #        new_vec = new_vec/(len(aspect_multiwrd_list)*1.)
    #        multiwrd_indx_dict[new_wrd] = multiwrd_indx_cntr
    #        multiwrd_indx_cntr += 1
    #        nparr_embed_list.append(new_vec)
    #return (np.asarray(nparr_embed_list, dtype=np.float32), multiwrd_indx_dict)
    return np.asarray(nparr_embed_list, dtype=np.float32)


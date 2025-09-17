import json
import re
import sys
import gzip
import codecs
import string
from math import log2
from collections import Counter
from spacy.lang.en import English

# The following module is optional but useful for debugging
from traceback_with_variables import activate_by_import

# The tqdm package is handy for progress bars. You can use tqdm around any list or iterator, e.g.:
#   import tqdm
#   for line in tqdm(lines):
#      do stuff
#
from tqdm import tqdm



# Congressional Speech Analyzer - Data Processing Module
#
# Read in congressional speeches jsonlines, i.e. a file with one well formed json element per line.
# Limiting to just speeches where the chamber was the Senate, return a list of strings
# in the following format:
#   '<party>TAB<text>'
# where <party> and <text> refer to the elements of those names in the json.
# Make sure to replace line-internal whitespace (one or more newlines, tabs, spaces, etc.) in text with a single space.
#
# For information on how to read from a gzipped file, rather than uncompressing and reading, see
# https://stackoverflow.com/questions/10566558/python-read-lines-from-compressed-text-files#30868178
#
# For info on parsing jsonlines, see https://www.geeksforgeeks.org/json-loads-in-python/.
# (There are other ways of doing it, of course.)
#
def read_and_clean_lines(infile):
    print("\nReading and cleaning text from {}".format(infile))
    lines = []
    with gzip.open(infile,'rt') as f:
        for line in tqdm(f):
            try:
                data = json.loads(line.strip())
                if data.get('chamber') == 'Senate':
                    party = data.get('party', '')
                    text = data.get('text', '')
                    cleaned_text = re.sub(r'\s+', ' ', text.strip())
                    lines.append(f"{party}\t{cleaned_text}")
            except json.JSONDecodeError:
                continue
    return(lines)

# Input: lines containing <party> TAB <text>
# Writes just the text to outfile 
def write_party_speeches(lines, outfile, party_to_write):
    print("{} speeches being written to {}".format(party_to_write, outfile))
    with open(outfile, "w") as f:
        for line in tqdm(lines):
            party, text = line.split('\t')
            if party == party_to_write:
                f.write(text + '\n')

# Read a set of stoplist words from filename, assuming it contains one word per line
# Return a python Set data structure (https://www.w3schools.com/python/python_sets.asp)
def load_stopwords(filename):
    stopwords = []
    with open(filename, 'r') as f:
        for line in f:
            word = line.strip()
            if word:
                stopwords.append(word)
    return set(stopwords)

# Take a list of string tokens and return all ngrams of length n,
# representing each ngram as a list of  tokens.
# E.g. ngrams(['the','quick','brown','fox'], 2)
# returns [['the','quick'], ['quick','brown'], ['brown','fox']]
# Note that this should work for any n, not just unigrams and bigrams
def ngrams(tokens, n):
    ngrams_list = []
    for i in range(len(tokens) - n + 1):
        ngram = tokens[i:i+n]
        ngrams_list.append(ngram)
    return ngrams_list

def filter_punctuation_bigrams(ngrams):
    # Input: assume ngrams is a list of ['token1','token2'] bigrams
    # Removes ngrams like ['today','.'] where either token is a single punctuation character
    # Note that this does not mean tokens that merely *contain* punctuation, e.g. "'s"
    # Returns list with the items that were not removed
    punct = string.punctuation
    return [ngram   for ngram in ngrams   if ngram[0] not in punct and ngram[1] not in punct]

def filter_stopword_bigrams(ngrams, stopwords):
    # Input: assume ngrams is a list of ['token1','token2'] bigrams, stopwords is a set of words like 'the'
    # Removes ngrams like ['in','the'] and ['senator','from'] where either word is a stopword
    # Returns list with the items that were not removed
    return [ngram for ngram in ngrams if ngram[0] not in stopwords and ngram[1] not in stopwords]


def normalize_tokens(tokenlist):
    # Input: list of tokens as strings,  e.g. ['I', ' ', 'saw', ' ', '@psresnik', ' ', 'on', ' ','Twitter']
    # Output: list of tokens where
    #   - All tokens are lowercased
    #   - All tokens starting with a whitespace character have been filtered out
    #   - All handles (tokens starting with @) have been filtered out
    #   - Any underscores have been replaced with + (since we use _ as a special character in bigrams)

    normalized_tokens = []
    for token in tokenlist:
        if token.startswith(' ') or token.startswith('\t') or token.startswith('\n'):
            continue
        if token.startswith('@'):
            continue
        normalized_token = token.lower().replace('_', '+')
        normalized_tokens.append(normalized_token)

    return normalized_tokens

def collect_bigram_counts(lines, stopwords, remove_stopword_bigrams = False):
    # Input lines is a list of raw text strings, stopwords is a set of stopwords
    #
    # Create a bigram counter
    # For each line:
    #   Extract all the bigrams from the line 
    #   If remove_stopword_bigrams is True:
    #     Filter out any bigram where either word is a stopword
    #   Increment the count for each bigram
    # Return the counter
    #
    # In the returned counter, the bigrams should be represented as string tokens containing underscores.
    # 
    if (remove_stopword_bigrams):
        print("Collecting bigram counts with stopword-filtered bigrams")
    else:
        print("Collecting bigram counts with all bigrams")
    
    print("Initializing spacy")
    nlp       = English(parser=False)
    counter   = Counter()

    for line in tqdm(lines):

        spacy_analysis = nlp(line.strip())
        spacy_tokens = [token.orth_ for token in spacy_analysis]

        normalized_tokens = normalize_tokens(spacy_tokens)

        bigrams = ngrams(normalized_tokens, 2)

        filtered_bigrams = filter_punctuation_bigrams(bigrams)

        if remove_stopword_bigrams:
            filtered_bigrams = filter_stopword_bigrams(filtered_bigrams, stopwords)

        for bigram in filtered_bigrams:
            bigram_string = '_'.join(bigram)
            counter[bigram_string] += 1

    return counter

def get_unigram_counts(bigram_counts, position):
    unigram_counts = Counter()
    for bigram_string, count in bigram_counts.items():
        words = bigram_string.split('_')
        if len(words) == 2:
            word = words[position]
            unigram_counts[word] += count
    return unigram_counts

def print_sorted_items(dict, n=10, order='ascending'):
    if order == 'descending':
        multiplier = -1
    else:
        multiplier = 1
    ranked = sorted(dict.items(), key=lambda x: x[1] * multiplier)
    for key, value in ranked[:n] :
        print(key, value)



################################################################
# Congressional Speech Analyzer - Main Execution
################################################################

#input_speechfile   = "./speeches2020.jsonl.gz"
input_speechfile   = "./speeches2020_jan_to_jun.jsonl.gz"
text_dems          = "./speeches_dem.txt"
text_reps          = "./speeches_rep.txt"
stopwords_file     = "./mallet_en_stoplist.txt"
min_freq_for_pmi   =  5
topN_to_show       = 50

def main():
    
    stopwords = load_stopwords(stopwords_file)

    print("\nProcessing text from input file {}".format(input_speechfile))
    lines = read_and_clean_lines(input_speechfile)

    print("\nWriting Democrats' speeches to {}".format(text_dems))
    write_party_speeches(lines, text_dems, "Democrat")

    print("\nWriting Republicans' speeches to {}".format(text_reps))
    write_party_speeches(lines, text_reps, "Republican")

    print("\nGetting Dem unigram and bigram counts")
    with open(text_dems) as f:
        dem_speeches = f.readlines()
    dem_bigram_counts     = collect_bigram_counts(dem_speeches, stopwords, True)
    dem_unigram_w1_counts = get_unigram_counts(dem_bigram_counts,0)
    dem_unigram_w2_counts = get_unigram_counts(dem_bigram_counts,1)
    print("\nTop Dem bigrams by frequency")
    print_sorted_items(dem_bigram_counts, topN_to_show, 'descending')

    print("\nGetting Rep unigram and bigram counts")
    with open(text_reps) as f:
        rep_speeches = f.readlines()
    rep_bigram_counts     = collect_bigram_counts(rep_speeches, stopwords, True)
    rep_unigram_w1_counts = get_unigram_counts(rep_bigram_counts,0)
    rep_unigram_w2_counts = get_unigram_counts(rep_bigram_counts,1)
    print("\nTop Rep bigrams by frequency")
    print_sorted_items(rep_bigram_counts, topN_to_show, 'descending')

    
if __name__ == "__main__":
    main()


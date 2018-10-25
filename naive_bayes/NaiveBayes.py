import numpy
import nltk
from nltk.corpus import sentiwordnet

# load data
posproc = numpy.load('posproc.npy')
negproc = numpy.load('negproc.npy')

sarcastic_tweets = []
non_sarcastic_tweets = []

for tweet in posproc:
    sarcastic_tweets.append(tweet.decode('utf-8'))

for tweet in negproc:
    non_sarcastic_tweets.append(tweet.decode('utf-8'))

print('num sarcastic tweets: ' + str(len(sarcastic_tweets)))
print('num non sarcastic tweets: ' + str(len(non_sarcastic_tweets)))

# Separate training and testing data
training_sarcastic_tweets = sarcastic_tweets[0:20000]
testing_sarcastic_tweets = sarcastic_tweets[20000:]

training_non_sarcastic_tweets = non_sarcastic_tweets[0:100000]
testing_non_sarcastic_tweets = non_sarcastic_tweets[100000:]

# --- Training ---

# - Unigrams -
sarcastic_unigram_counts = {}
non_sarcastic_unigram_counts = {}

total_sarcastic_unigram_count = 0
total_non_sarcastic_unigram_count = 0
unique_unigram_count = 0

# get unigram counts for sarcastic tweets
for tweet in training_sarcastic_tweets:
    tokenized_tweet = nltk.word_tokenize(tweet)
    for token in tokenized_tweet:
        count = sarcastic_unigram_counts.get(token) or 0
        if count == 0 and non_sarcastic_unigram_counts.get(token) is None:
            unique_unigram_count += 1
        sarcastic_unigram_counts[token] = count + 1
        total_sarcastic_unigram_count += 1

# get unigram counts for non sarcastic tweets
for tweet in training_non_sarcastic_tweets:
    tokenized_tweet = nltk.word_tokenize(tweet)
    for token in tokenized_tweet:
        count = non_sarcastic_unigram_counts.get(token) or 0
        if count == 0 and sarcastic_unigram_counts.get(token) is None:
            unique_unigram_count += 1
        non_sarcastic_unigram_counts[token] = count + 1
        total_non_sarcastic_unigram_count += 1

print('')
print('unique unigram count: ' + str(unique_unigram_count))
print('total sarcastic unigram count: ' + str(total_sarcastic_unigram_count))
print('total non sarcastic unigram count: ' + str(total_non_sarcastic_unigram_count))


# - Bigrams -
sarcastic_bigram_counts = {}
non_sarcastic_bigram_counts = {}

total_sarcastic_bigram_count = 0
total_non_sarcastic_bigram_count = 0
unique_bigram_count = 0

# get bigram counts for sarcastic tweets
for tweet in training_sarcastic_tweets:
    tokenized_tweet = nltk.word_tokenize(tweet)
    tokenized_tweet.append('<end>')
    bigram_array = ['', '<start>']

    for word in tokenized_tweet:
        bigram_array.pop(0)
        bigram_array.append(word)
        bigram = ' '.join(bigram_array)
        count = sarcastic_bigram_counts.get(bigram) or 0
        if count == 0 and non_sarcastic_bigram_counts.get(bigram) is None:
            unique_bigram_count += 1
        sarcastic_bigram_counts[bigram] = count + 1
        total_sarcastic_bigram_count += 1

# get bigram counts for non sarcastic tweets
for tweet in training_non_sarcastic_tweets:
    tokenized_tweet = nltk.word_tokenize(tweet)
    tokenized_tweet.append('<end>')
    bigram_array = ['', '<start>']

    for word in tokenized_tweet:
        bigram_array.pop(0)
        bigram_array.append(word)
        bigram = ' '.join(bigram_array)
        count = non_sarcastic_bigram_counts.get(bigram) or 0
        if count == 0 and sarcastic_bigram_counts.get(bigram) is None:
            unique_bigram_count += 1
        non_sarcastic_bigram_counts[bigram] = count + 1
        total_non_sarcastic_bigram_count += 1

print('')
print('unique bigram count: ' + str(unique_bigram_count))
print('total sarcastic bigram count: ' + str(total_sarcastic_bigram_count))
print('total non sarcastic bigram count: ' + str(total_non_sarcastic_bigram_count))


# - Repeated Characters -
sarcastic_repeated_character_count = 0
non_sarcastic_repeated_character_count = 0

# get repeated character count for sarcastic tweets
for tweet in training_sarcastic_tweets:
    characters = ['null_1', 'null_2', 'null_3']
    repeated_characters = False
    for character in tweet:
        characters.pop(0)
        characters.append(character)
        if characters[0] == characters[1] and characters[1] == characters[2]:
            repeated_characters = True
            break
    if repeated_characters:
        sarcastic_repeated_character_count += 1

# get repeated character count for non sarcastic tweets
for tweet in training_non_sarcastic_tweets:
    characters = ['null_1', 'null_2', 'null_3']
    repeated_characters = False
    for character in tweet:
        characters.pop(0)
        characters.append(character)
        if characters[0] == characters[1] and characters[1] == characters[2]:
            repeated_characters = True
            break
    if repeated_characters:
        non_sarcastic_repeated_character_count += 1


print('')
print('sarcastic repeated character count: ' + str(sarcastic_repeated_character_count))
print('non sarcastic repeated character count: ' + str(non_sarcastic_repeated_character_count))


# --- Testing ---

# combine sarcastic and non-sarcastic tweets into one testing set
testing_tweets = testing_sarcastic_tweets
testing_tweets.extend(testing_non_sarcastic_tweets)

# results matrix (true positive, false positive, false negative, true negative)
results = {
    'tp': 0, 'fp': 0,
    'fn': 0, 'tn': 0
}

# determine result of each tweet in testing set
for tweet in testing_tweets:
    tokenized_tweet = nltk.word_tokenize(tweet)

    sarcastic_prob = 1
    non_sarcastic_prob = 1

    # unigram testing
    for word in tokenized_tweet:
        # probability this unigram is sarcastic
        s_count = (sarcastic_unigram_counts.get(word) or 0) + 1
        s_prob = s_count / (total_sarcastic_unigram_count + unique_unigram_count)
        sarcastic_prob *= s_prob

        # probability this unigram is non sarcatic
        ns_count = (non_sarcastic_unigram_counts.get(word) or 0) + 1
        ns_prob = ns_count / (total_non_sarcastic_unigram_count + unique_unigram_count)
        non_sarcastic_prob *= ns_prob

    # bigram testing
    tokenized_tweet.append('<end>')
    bigram_array = ['', '<start>']
    for word in tokenized_tweet:
        bigram_array.pop(0)
        bigram_array.append(word)
        bigram = ' '.join(bigram_array)

        # probability this bigram is sarcastic
        s_count = (sarcastic_bigram_counts.get(bigram) or 0) + 1
        s_prob = s_count / (total_sarcastic_bigram_count + unique_bigram_count)
        sarcastic_prob *= s_prob

        # probability this bigram is non sarcastic
        ns_count = (non_sarcastic_bigram_counts.get(bigram) or 0) + 1
        ns_prob = ns_count / (total_non_sarcastic_bigram_count + unique_bigram_count)
        non_sarcastic_prob *= ns_prob

    repeated_characters = False
    for character in tweet:
        characters.pop(0)
        characters.append(character)
        if characters[0] == characters[1] and characters[1] == characters[2]:
            repeated_characters = True
            break
    if repeated_characters:
        sarcastic_prob *= (sarcastic_repeated_character_count / 20000)
        non_sarcastic_prob *= (non_sarcastic_repeated_character_count / 100000)
    else:
        sarcastic_prob *= ((20000 - sarcastic_repeated_character_count) / 20000)
        non_sarcastic_prob *= ((100000 - non_sarcastic_repeated_character_count) / 100000)

    result = 's'
    if non_sarcastic_prob > sarcastic_prob:
        result = 'ns'

    if result == 's':
        if tweet in sarcastic_tweets:
            results['tp'] = results.get('tp') + 1
        else:
            results['fp'] = results.get('fp') + 1
    else:
        if tweet in sarcastic_tweets:
            results['fn'] = results.get('fn') + 1
        else:
            results['tn'] = results.get('tn') + 1

precision = results.get('tp') / (results.get('tp') + results.get('fp'))
recall = results.get('tp') / (results.get('tp') + results.get('fn'))
f_score = (2 * precision * recall) / (precision + recall)


print('')
print('precision: ' + str(precision))
print('recall: ' + str(recall))
print('f-score: ' + str(f_score))


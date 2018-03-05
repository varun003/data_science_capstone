setwd("F:/ANALYTICS DATA/R/DATA MANIPULATION/Data Science Capstone/final/en_US")


library(RWekajars)
library(qdapDictionaries)
library(qdapRegex)
library(qdapTools)
library(RColorBrewer)
library(qdap)
library(NLP)
library(tm)
library(SnowballC)
library(slam)
library(RWeka)
library(rJava)
library(stringr)
library(DT)
library(stringi)
library(tidyverse)

# Reading the dataset

twitter <- readLines("en_US.twitter.txt",skipNul = T,encoding = "UTF-8")
news <- readLines("en_US.news.txt",skipNul = T,encoding = "UTF-8")
blogs <- readLines("en_US.blogs.txt",skipNul = T,encoding = "UTF-8")


# Making a sample data

sampleTwitter <- twitter[sample(1:length(twitter),5000)]
sampleNews <- news[sample(1:length(news),5000)]
sampleBlogs <- blogs[sample(1:length(blogs),5000)]
textSample <- c(sampleTwitter,sampleNews,sampleBlogs)


## Save sample
writeLines(textSample, "./new1/textSample.txt")

theSampleCon <- file("./textSample.txt")
theSample <- readLines(theSampleCon)
#close(theSampleCon)


## Build the corpus, and specify the source to be character vectors 
cleanSample <- Corpus(VectorSource(theSample))

## Make it work with the new tm package
cleanSample <- tm_map(cleanSample,
                      content_transformer(function(x) 
                        iconv(x, to="UTF-8", sub="byte")))

## Convert to lower case
cleanSample <- tm_map(cleanSample, content_transformer(tolower))

## remove punction, numbers, URLs, stop, profanity and stem wordson
cleanSample <- tm_map(cleanSample, content_transformer(removePunctuation))
cleanSample <- tm_map(cleanSample, content_transformer(removeNumbers))
removeURL <- function(x) gsub("http[[:alnum:]]*", "", x) 
cleanSample <- tm_map(cleanSample, content_transformer(removeURL))
cleanSample <- tm_map(cleanSample, stripWhitespace)
cleanSample <- tm_map(cleanSample, removeWords, stopwords("english"))
#cleanSample <- tm_map(cleanSample, removeWords, profanityWords)
cleanSample <- tm_map(cleanSample, stemDocument)
cleanSample <- tm_map(cleanSample, stripWhitespace)

## Saving the final corpus
saveRDS(cleanSample, file = "./new1/finalCorpus.RData")



## Budilding the n-grams


finalCorpus <- readRDS("./new1/finalCorpus.RData")
finalCorpusDF <-data.frame(text=unlist(sapply(finalCorpus,`[`, "content")), 
                           stringsAsFactors = FALSE)

## Building the tokenization function for the n-grams
ngramTokenizer <- function(theCorpus, ngramCount) {
  ngramFunction <- NGramTokenizer(theCorpus, 
                                  Weka_control(min = ngramCount, max = ngramCount, 
                                               delimiters = " \\r\\n\\t.,;:\"()?!"))
  ngramFunction <- data.frame(table(ngramFunction))
  ngramFunction <- ngramFunction[order(ngramFunction$Freq, 
                                       decreasing = TRUE),][1:10,]
  colnames(ngramFunction) <- c("String","Count")
  ngramFunction
}

unigram <- ngramTokenizer(finalCorpusDF, 1)
saveRDS(unigram, file = "./new1/unigram.RData")
bigram <- ngramTokenizer(finalCorpusDF, 2)
saveRDS(bigram, file = "./new1/bigram.RData")
trigram <- ngramTokenizer(finalCorpusDF, 3)
saveRDS(trigram, file = "./new1/trigram.RData")
quadgram <- ngramTokenizer(finalCorpusDF, 4)
saveRDS(quadgram, file = "./new1/quadgram.RData")

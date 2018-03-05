
setwd("F:/ANALYTICS DATA/R/DATA MANIPULATION/Data Science Capstone/final/en_US")

library(tm)
library(dplyr)
library(parallel)
library(RWeka)
library(rJava)
library(RWekajars)
library(R.utils)

twitter <- readLines("en_US.twitter.txt",skipNul = T,encoding = "UTF-8")
news <- readLines("en_US.news.txt",skipNul = T,encoding = "UTF-8")
blogs <- readLines("en_US.blogs.txt",skipNul = T,encoding = "UTF-8")




sampletext <- function(textbody, portion) {
  taking <- sample(1:length(textbody), length(textbody)*portion)
  sampletext <- textbody[taking]
  sampletext
}

# sampling text files 
set.seed(65364)
portion <- 0.01
SampleTwitter <- sampletext(twitter, portion)
SampleBlog <- sampletext(blogs, portion)
SampleNews <- sampletext(news, portion)

# Combining sample texts into one variable

SampleAll <- c(SampleBlog,SampleTwitter,SampleNews)

# Write sample lines into a new file

writeLines(SampleAll,"./SampleAll.txt")


# Data Cleaning

cleansing <- function (text) {
  text <- tm_map(text, content_transformer(tolower))
  text <- tm_map(text, stripWhitespace)
  text <- tm_map(text, removePunctuation)
  text <- tm_map(text, removeNumbers)
  text
}

SampleAll <- VCorpus(DirSource("./SampleAll.txt",encoding = "UTF-8"))


# tokenizing sampled text 
SampleAll <- cleansing(SampleAll)



# Define function to make N grams
tdm_Ngram <- function (text, n) {
  NgramTokenizer <- function(x) {RWeka::NGramTokenizer(x, RWeka::Weka_control(min = n, max = n))}
  tdm_ngram <- TermDocumentMatrix(text, control = list(tokenizer = NgramTokenizer))
  tdm_ngram
}

# Define function to extract the N grams and sort
ngram_sorted_df <- function (tdm_ngram) {
  tdm_ngram_m <- as.matrix(tdm_ngram)
  tdm_ngram_df <- as.data.frame(tdm_ngram_m)
  colnames(tdm_ngram_df) <- "Count"
  tdm_ngram_df <- tdm_ngram_df[order(-tdm_ngram_df$Count), , drop = FALSE]
  tdm_ngram_df
}

# Calculate N-Grams
tdm_1gram <- tdm_Ngram(SampleAll, 1)
tdm_2gram <- tdm_Ngram(SampleAll, 2)
tdm_3gram <- tdm_Ngram(SampleAll, 3)
tdm_4gram <- tdm_Ngram(SampleAll, 4)


# Extract term-count tables from N-Grams and sort 
tdm_1gram_df <- ngram_sorted_df(tdm_1gram)
tdm_2gram_df <- ngram_sorted_df(tdm_2gram)
tdm_3gram_df <- ngram_sorted_df(tdm_3gram)
tdm_4gram_df <- ngram_sorted_df(tdm_4gram)



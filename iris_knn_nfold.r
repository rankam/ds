library(class)
library(ggplot2)

###########################
# PREPROCESSING
###########################
data <- iris               # create copy of iris dataframe
set.seed(1)                # keeps data consistent

N <- nrow(data)            # 150 rows of data
index <- sample(1:N, N)     
data <- data[index, ]
folds <- 10                 # number of folds
fold.rows <- N/folds        # number of rows per fold

labels <- data$Species      # store labels
data$Species <- NULL 

###########################
# APPLY MODEL
###########################
err.rates <- data.frame()
for (a in 1:folds)
{
  start.index <- (((a - 1) * fold.rows) + 1) # starting point for each fold
  end.index <- (a*fold.rows)
  
  test.index <- start.index:end.index 
  
  train.data <- data[-test.index, ]  # perform train/test split
  test.data <- data[test.index, ]    # extract test set labels  
  
  train.labels <- as.factor(as.matrix(labels)[-test.index, ])      # extract training set labels
  test.labels <- as.factor(as.matrix(labels)[test.index, ]) 
  
  max.k <- 10
  for (k in 1:max.k)                          # perform fit for various values of k
  {
    knn.fit <- knn(train = train.data,        # training set
                  test = test.data,           # test set
                  cl = train.labels,          # true labels
                  k = k                       # number of NN to poll
    )
  
    print(table(test.labels, knn.fit))        # print confusion matrix
    
    cat('\n', 'k = ', k, '\n', sep='')                               # print params
    this.err <- sum(test.labels != knn.fit) / length(test.labels)    # store gzn err
    err.rates <- rbind(err.rates, this.err)     # append err to total results
    
  }
}

###########################
# OUTPUT RESULTS
###########################
avg.err <- colMeans(err.rates) # estimates OOS accuracy

results <- data.frame(1:max.k, err.rates)   # create results summary data frame
names(results) <- c('k', 'err.rate')        # label columns of results df                             # create title for results plot

title <- paste('n fold knn results', sep='')

# create results plot
results.plot <- ggplot(results, aes(x=k, y=err.rate)) + geom_point() + geom_line()
results.plot <- results.plot + ggtitle(title)

# draw results plot (note need for print stmt inside script to draw ggplot)
print(results.plot)
cat('Average Generalization Error = ',avg.err)

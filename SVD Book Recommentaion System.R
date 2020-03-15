library(dplyr)
library(reshape2)
library(data.table)

# read dataset
rating <- fread("BX-CSV-Dump/BX-Book-Ratings.csv",sep=";")
colnames(rating) <- c("User.ID", "ISBN", "Book.Rating")
rating$User.ID <- as.character(rating$User.ID)

# create frequency table based on User.ID
user.table <- table(rating$User.ID)
user.order <- as.data.frame(user.table[order(user.table, decreasing = TRUE)])
user.order[1:10,]
user.order[99:105,]

# Take the top 100 frequent users and subset
user.select <- as.character(user.order[1:100,1])
rating.select <- rating[which(rating$User.ID %in% user.select),1:3]
rating.select$User.ID.num <- as.factor(rating.select$User.ID)
rating.select$User.ID.num <- as.numeric(rating.select$User.ID.num)
rating.select$ISBN.num <- as.factor(rating.select$ISBN)
rating.select$ISBN.num <- as.numeric(rating.select$ISBN.num)

# count number of unique users and unique books
nuser <- length(unique(rating.select$User.ID))
nbook <- length(unique(rating.select$ISBN))
nrating <- length(unique(rating.select$Book.Rating))
cat("Number of unique users:\t", nuser, "\n")
cat("Number of unique books:\t", nbook, "\n")
cat("Level of ratings:\t", nrating, "\n")
summary(rating.select)

set.seed(1)
# split train/test set
train <- sample(1: nrow(rating.select), 1e5)
test <- setdiff(1: nrow(rating.select), train)

train.set <- rating.select[,c(4,5,3)]
train.set$Book.Rating[test] <- NA

test.set <- rating.select[,c(4,5,3)]
test.set$Book.Rating[train] <- NA

# convert training set into a sparse matrix
train.matrix <- dcast(train.set, User.ID.num ~ ISBN.num, value.var="Book.Rating")
train.matrix <- as.matrix(train.matrix[,-1])

# convert test set into a sparse matrix
test.matrix <- dcast(test.set, User.ID.num ~ ISBN.num, value.var="Book.Rating")
test.matrix <- as.matrix(test.matrix[,-1])

row_means <- rowMeans(train.matrix, na.rm = TRUE)

# create fully specified matrix R1 by replacing NA with row means
R1 <- train.matrix
for (i in 1:nrow(R1)){
  R1[i,is.na(R1[i,])] <- row_means[i]
}

# apply SVD on R1
SVD <- svd(R1)
plot(1:length(SVD$d),SVD$d)
plot(2:length(SVD$d),SVD$d[-1])

# take the first 15 eigenvalues
R2 <- SVD$u[,1:15] %*% diag(SVD$d)[1:15,1:15] %*% t(SVD$v[,1:15])

# calculate prediction error on test set
mse <- sum((test.matrix-R2)**2, na.rm=TRUE) / length(test)
cat("Test Error (MSE):\t", mse, "\n")
cat("Test Error (RMSE):\t", mse**0.5, "\n")
benchmark <- sum((test.matrix-R1)**2, na.rm=TRUE) / length(test)
cat("Benchmark (MSE):\t", benchmark, "\n")
cat("Benchmark (MSE):\t", benchmark**0.5, "\n")

# initialize rating matrix as original train set
R3 <- train.matrix

# replace the missing values with results in R2
for (j in 1:nrow(R3)){
  index <- is.na(R3[j,])
  R3[j,index] <- R2[j,index]
}

SVD2 <- svd(R3)
R4 <- SVD2$u[,1:15] %*% diag(SVD2$d)[1:15,1:15] %*% t(SVD2$v[,1:15])
mse2 <- sum((test.matrix-R4)**2, na.rm=TRUE) / length(test)

cat("Test Error (MSE):\t", mse2, "\n")
cat("Test Error (RMSE):\t", mse2**0.5, "\n")
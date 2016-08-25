f.MergeExperian <- function(){
  
  require(data.table)
  home <- read.csv(file = "homeProfileExpIN.csv", sep = ",", header = T, colClasses = "character")
  person <- read.csv(file = "personProfileExpIN.csv", sep = ",", header = T, colClasses = "character")
  dem <- read.csv(file = "demProfileExpIN.csv", sep = ",", header = T, colClasses = "character")
  
  home <- data.table(home, key = "record_id")
  person <- data.table(person, key = "record_id")
  dem <- data.table(dem, key = "record_id")
  
  final <- Reduce(merge, list(home, person, dem))
  colnames(final)[1] <- "account.id"
  return(final)
}
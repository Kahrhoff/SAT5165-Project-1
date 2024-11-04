This project was created to prepare a spreadsheet derived from the data of CTU-IoT-Malware-Capture-9-1 from the Aposemat IoT-23 dataset
for the use in a future machine learning model. 
This program is designed to be run between two machines running Hadoop and Spark split the workload amongst each node. 

Please note, file paths were hardcoded in this first attempt, so please be aware it will require editing for other uses. 

project1.py will perform some data processing on the log.xlsx file and return final-log.xlsx which should be ready for machine learning. 

big_data_project3 (1).py is an updated version. The file path can be found on line 19 of the file and will need to be updated for each user. This program does the same processing as project1.py with a few exceptions: 
 - the output of a feature 'Others' has been removed and instead the 'history' feature now appropriately only changes instances with less than 10 appearances to the word 'Other' before classification
 - the output of the classification to the classificaiton categories (history, proto, service) now uses label encoding to turn each classification into a numerical value

The new version additionally also creates a random forest classifier as well as training and testing it. Current testing has the model performing without error. As the dataset is expanded to mimic more realistic network traffic, this will likely change. 

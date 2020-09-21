# -- GEO1001.2020--hw01
# -- [Simon Pena Pereira] 
# -- [5391210]

# Assignment 01
# Statistical analysis of a heat stress measurement dataset

# Predefinitions 
- manage the excel dataset by using pandas to read them and defining the rows you want to read
- then I defined values for each column of Temperatures/Wind Speed/Direction , True / Crosswind Speed/ WBGT which are implemented in A1.3/A1.4/A3.1/4.2/4.3
## to access the data sets, all excel files need to be in the same directory as the code. Furthermore, they need to be named the same

- Predefine tuples for A1.1/A1.2/A2.1/A2.2 access the dataset 

# A1: Compute mean statistics (mean, variance and standard deviation for each of the sensors variables)
# A1.2: Then I used matplotlib.pyplot to create one plot for two histograms
# A1.3: Here I used the predefined variables and implemented them in the code to create frequency  poligons
# A1.4: The function to create boxplots is called by the previous mentioned variables 

# A2.1: I used a for loop to iterate through the predefined tuples in order to create PMFs, PDFs and CDFs for the 5 sensors Temperature values in independent plots   
# A2.2: Here, I did the same as described in A2.1 just for PDFs and KDEs for the 5 sensors Wind Speed values

# A3: In the function for correlation, the correlations of all sensors are calculated by interpolating and normalizing them. Then the statistics for the Pearson's and Spearmann's rank coeffients are computed, followed by the computation of scatter plots

# A4.1: Similar to A2.1: CDFs for all the sensors and for variables Temperature and Wind Speed are computed
# 4.2: A for loop iterates through two tuples in order to obtain the start and end values of a 95% confidence intervals for the variables Temperature and Wind Speed for all the sensors. Afterwards they are saved in a .txt file called confidence_int.txt
# 4.3: The function for the ttst is called 8 times to compute the t-values and p-values for each given sensor pair



import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.stats import norm


#set font size for our plots

font = {'weight' : 'bold',
        'size'   : 12}

plt.rc('font', **font)


#prepare our data array
data = np.zeros((50,25))

#data is from https://www.kaggle.com/datasets/asaniczka/median-and-avg-hourly-wages-in-the-usa-1973-2022

#read in the data from the csv file
with open('Assignment2/median_average_wages.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    #record line count
    line_count = 0
    for row in csv_reader:
        #if line count is 0 then these are just the column names
        if line_count == 0:
            column_names = row
            #increment line count
            line_count += 1
        else:
            #not looking at the final columns since they have much less data points
            for i in range(0,25):
                #enter data into our array
                data[line_count - 1, i] = float(row[i])
            #increment line count
            line_count +=1

#store years as well as the median wages for
#total population, white population, black population and hispanic population
years = data[:,0]
med_wages = data[:,1]
#here we consider the difference between each racial group and the total population
med_white_wages = data[:,7] / med_wages * 100
med_black_wages = data[:,9] / med_wages * 100
med_hispanic_wages = data[:,11] / med_wages * 100



#calculating k0 and k1 for linear regression for each racial group
denom = np.mean(years ** 2)  - np.mean(years) ** 2

k0_white = np.mean(med_white_wages) * np.mean(years ** 2) - np.mean(years)  * np.mean(years * med_white_wages)
k0_white = k0_white / denom
k1_white = np.mean(med_white_wages * years) - np.mean(med_white_wages) * np.mean(years)
k1_white = k1_white / denom
print('w_white = ', k1_white, 't + ', k0_white)


k0_black = np.mean(med_black_wages) * np.mean(years ** 2) - np.mean(years)  * np.mean(years * med_black_wages)
k0_black = k0_black / denom
k1_black = np.mean(med_black_wages * years) - np.mean(med_black_wages) * np.mean(years)
k1_black = k1_black / denom
print('w_black = ', k1_black, 't + ', k0_black)


k0_hispanic = np.mean(med_hispanic_wages) * np.mean(years ** 2) - np.mean(years)  * np.mean(years * med_hispanic_wages)
k0_hispanic = k0_hispanic / denom
k1_hispanic = np.mean(med_hispanic_wages * years) - np.mean(med_hispanic_wages) * np.mean(years)
k1_hispanic = k1_hispanic/ denom
print('w_hispanic = ', k1_hispanic, 't + ', k0_hispanic)


#plot our data for each race as well as a comparison line
plt.plot(years, med_white_wages, 'ro', label='data: white percentage difference')
plt.plot(years, med_black_wages, 'bo', label='data: black percentage difference')
plt.plot(years, med_hispanic_wages, 'go', label='data: hispanic percentage difference')

#now look an extra 25 years ahead as well
years = np.linspace(1973,1973+75,75)
#plot a comparison line
plt.plot(years, 100 * np.ones(75), 'k', label = 'comparison line (national average)')

#and plot our linear regresion for each race
plt.plot(years, k1_white * years + k0_white, 'r', label='linear regression: white percentage difference')
plt.plot(years, k1_black * years + k0_black, 'b', label='linear regression: black percentage difference')
plt.plot(years, k1_hispanic * years + k0_hispanic, 'g', label='linear regression: hispanic percentage difference')

#add labels and a legend, then plot
plt.xlabel('Year')
plt.ylabel('Difference from median hourly wage')
plt.legend()
plt.show()


#We are now going to compare yearly percentage increase in median hourly wages
#have years include all but the first year
years = data[range(1,50),0]
#calculate percentage increase each year for total population
total_inc = (med_wages[range(1,50)] - med_wages[range(0,49)]) / med_wages[range(0,49)] * 100
#calculate percentage increase each year for each group
hisp_inc = (data[range(1,50),11] - data[range(0,49),11]) / data[range(0,49),11] * 100
black_inc = (data[range(1,50),9] - data[range(0,49),9]) / data[range(0,49),9] * 100
white_inc = (data[range(1,50),7] - data[range(0,49),7]) / data[range(0,49),7] * 100

#calculate means for each population
TotalMean = np.mean(total_inc)
HispanicMean = np.mean(hisp_inc)
WhiteMean = np.mean(white_inc)
BlackMean = np.mean(black_inc)

#calculate unbiased standard deviations for each population
Totalstd = np.std(total_inc, ddof=1)
Hispanicstd = np.std(hisp_inc, ddof=1)
Whitestd = np.std(white_inc, ddof=1)
Blackstd = np.std(black_inc, ddof=1)

#calculate our z statistic for comparing yearly increase
#for the black and hispanic groups to the white population
z_stat_hisp = (WhiteMean - HispanicMean) / np.sqrt(Whitestd**2 / len(white_inc) + Hispanicstd **2 / len(hisp_inc))
z_stat_black = (WhiteMean - BlackMean) / np.sqrt(Whitestd**2 / len(white_inc) + Blackstd **2 / len(black_inc))

#output he z-statistics
print('z-stat (hispanic compared to white)', z_stat_hisp)
print('z-stat (black compared to white)', z_stat_black)


#calculate p-value for each comparison
p_value_hisp = 2 * norm.cdf(np.abs(z_stat_hisp))
p_value_hisp = min(p_value_hisp, 2 - p_value_hisp)

p_value_black = 2 * norm.cdf(np.abs(z_stat_black))
p_value_black = min(p_value_black, 2 - p_value_black)

#output the p-values
print('p-value (hispanic compared to white)', p_value_hisp)
print('p-value (black compared to white)', p_value_black)



#by taking the mean from our data, we find our MLE for our variance
#in order to get a normal distribution
mu = TotalMean

#output our mean
print('Population mean = ', mu)

N = 250001
#prepare arrays for sigma and likelihood
sigma_values = np.linspace(0.5,3,N)
likelihood = np.zeros(N)

#calculate likelihood at every sigma value
for i in range(0,N):
    eachlikelihood = norm.logpdf(total_inc ,loc = mu, scale = sigma_values[i])
    likelihood[i] = np.sum(eachlikelihood)

#plot our likelihood against sigma
plt.plot(sigma_values, likelihood)
plt.xlabel('standard deviation')
plt.ylabel('log likelihood')
plt.show()

#find the sigma that maximises the likelihood
max_location = np.argmax(likelihood)
sigma = sigma_values[max_location]

#output our sigma
print('MLE for sigma = ', sigma)

#obtain our normal pdf for this sigma and mu
values = np.linspace(-7, 5, 100)
norm_dist = norm.pdf(values, loc = mu, scale = sigma)

#plot histogram of total population's increases
plt.hist(total_inc, label='total population', alpha = 0.7, density=1)
#add our fitted normal distribution
plt.plot(values,norm_dist, label='fitted normal distribution for total population')
#add labels and a legend, then plot
plt.xlabel('Yearly percentage increase in average hourly wage')
plt.ylabel('frequency density')
plt.legend()
plt.show()


#plot histogram of hispanic population's increases
plt.hist(hisp_inc, label='hispanic population', alpha = 0.7, density=1)
#add our fitted normal distribution
plt.plot(values,norm_dist, label='fitted normal distribution for total population')
#add labels and a legend, then plot
plt.xlabel('Yearly percentage increase in average hourly wage')
plt.ylabel('frequency density')
plt.legend()
plt.show()

#plot histogram of black population's increases
plt.hist(black_inc, label='black population', alpha = 0.7, density=1)
#add our fitted normal distribution
plt.plot(values,norm_dist, label='fitted normal distribution for total population')
#add labels and a legend, then plot
plt.xlabel('Yearly percentage increase in average hourly wage')
plt.ylabel('frequency density')
plt.legend()
plt.show()

#plot histogram of white population's increases
plt.hist(white_inc, label='white population', alpha = 0.7, density=1)
#add our fitted normal distribution
plt.plot(values,norm_dist, label='fitted normal distribution for total population')
#add labels and a legend, then plot
plt.xlabel('Yearly percentage increase in average hourly wage')
plt.ylabel('frequency density')
plt.legend()
plt.show()

#outlier of -6 for many populations is likely a result of the 1973-1975 recession


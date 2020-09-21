#-- GEO1001.2020--hw01
#-- [Simon Pena Pereira] 
#-- [5391210]

#########################
import scipy as stats
from scipy import stats
from scipy.stats import norm
from scipy.stats import pearsonr
from scipy.stats import sem, t
import numpy as np	
import matplotlib.pyplot as plt	
import matplotlib.patches as mpatches
import pandas as pd
import csv as csv
import seaborn as sns
import statistics as stat


################################################################################################################################

# manage excel dataset

df_A = pd.read_excel("HEAT - A_final.xls", header = 3, skiprows = range(4,5))
df_B = pd.read_excel("HEAT - B_final.xls", header = 3, skiprows = range(4,5))
df_C = pd.read_excel("HEAT - C_final.xls", header = 3, skiprows = range(4,5))
df_D = pd.read_excel("HEAT - D_final.xls", header = 3, skiprows = range(4,5))
df_E = pd.read_excel("HEAT - E_final.xls", header = 3, skiprows = range(4,5))

# the following assigns the values of each column of Temperatures/Wind Speed/Direction , True/Crosswind Speed/ 
# WBGT to variables which are implemented in A1.3/A1.4/A3.1/4.2/4.3

temp_a = df_A['Temperature']
temp_b = df_B['Temperature']
temp_c = df_C['Temperature']
temp_d = df_D['Temperature']
temp_e = df_E['Temperature']

ws_a = df_A['Wind Speed']
ws_b = df_B['Wind Speed']
ws_c = df_C['Wind Speed']
ws_d = df_D['Wind Speed']
ws_e = df_E['Wind Speed']

wd_a = df_A['Direction ‚ True']
wd_b = df_B['Direction ‚ True']
wd_c = df_C['Direction ‚ True']
wd_d = df_D['Direction ‚ True']
wd_e = df_E['Direction ‚ True']

cws_a=df_A["Crosswind Speed"]
cws_b=df_B["Crosswind Speed"]
cws_c=df_C["Crosswind Speed"]
cws_d=df_D["Crosswind Speed"]
cws_e=df_E["Crosswind Speed"]

wbg_a=df_A["WBGT"]
wbg_b=df_B["WBGT"]
wbg_c=df_C["WBGT"]
wbg_d=df_D["WBGT"]
wbg_e=df_E["WBGT"]

# Predefine tuples for A1.1/A1.2/A2.1/A2.2 access the dataset 
Temperatures = (df_A['Temperature'], df_B['Temperature'], df_C['Temperature'], df_D['Temperature'], df_E['Temperature'])
WindSpeed = (df_A['Wind Speed'], df_B['Wind Speed'], df_C['Wind Speed'], df_D['Wind Speed'], df_E['Wind Speed'])

################################################################################################################################
# A1.1 # Compute mean statistics (mean, variance and standard deviation for each of the sensors variables)                     #
################################################################################################################################

mean = ("Mean: ", df_A.mean(), df_B.mean(), df_C.mean(), df_D.mean(), df_E.mean())
var = ("Variance: ", df_A.var(), df_B.var(), df_C.var(), df_D.var(), df_E.var())
stdv = ("Standard Deviation:", df_A.std(), df_B.std(), df_C.std(), df_D.std(), df_E.std())
print(mean, var, stdv)

################################################################################################################################
# A1.2 # Create 1 plot that contains histograms for the 5 sensors Temperature values. Compare histograms with 5 and 50 bins    #
################################################################################################################################

fig, ax = plt.subplots(1,2)
plt.xlabel("Temperature [°C]")
plt.ylabel("Frequency")
plt.suptitle("Sensors A/B/C/D/E:Temperature")
ax[0].hist(Temperatures, bins=50, density=False)
ax[1].hist(Temperatures, bins=5, density=False)
plt.show()


################################################################################################################################
# A1.3 # Create 1 plot where frequency poligons for the 5 sensors Temperature values overlap in different colors with a legend #
################################################################################################################################

fig = plt.figure(figsize=(21,6))
ax1 = fig.add_subplot(111)
ax2 = fig.add_subplot(111)
ax3 = fig.add_subplot(111)
ax4 = fig.add_subplot(111)
ax5 = fig.add_subplot(111)

[freq_a,bins]=np.histogram(temp_a,bins=27)
[freq_b,bins]=np.histogram(temp_b,bins=27)
[freq_c,bins]=np.histogram(temp_c,bins=27)
[freq_d,bins]=np.histogram(temp_d,bins=27)
[freq_e,bins]=np.histogram(temp_e,bins=27)

ax1.plot(bins[:-1],freq_a,label='Sensor A')
ax2.plot(bins[:-1],freq_b,label='Sensor B')
ax3.plot(bins[:-1],freq_c,label='Sensor C')
ax4.plot(bins[:-1],freq_d,label='Sensor D')
ax5.plot(bins[:-1],freq_e,label='Sensor E')
ax1.set_ylabel('Frequency')
ax1.set_xlabel('Temperature [°C]')
plt.title("Sensors A/B/C/D/E: Overlap of Temperature values")
plt.legend(prop={"size": 10}, title="Legend")
plt.grid()
plt.show()

################################################################################################################################
# A1.4 # Generate 3 plots that include the 5 sensors boxplot for: Wind Speed, Wind Direction and Temperature                   #
################################################################################################################################

def boxplt(a,b,c,d,e,t):
    data = [a,b,c,d,e]
    fig,ax = plt.subplots()
    ax.boxplot(data,showmeans = True)
    ax.set_ylabel(t)
    ax.set_xlabel("Sensors")
    plt.title(f"Sensors 1/2/3/4/5: Boxplots of {t}", loc="center")
    plt.show() 

boxplt(temp_a.astype(float), temp_b.astype(float), temp_c.astype(float), temp_d.astype(float), temp_e.astype(float), "Temperature [°C]")
boxplt(ws_a.astype(float), ws_b.astype(float), ws_c.astype(float), ws_d.astype(float), ws_e.astype(float), "Wind Speed [m/s]")
boxplt(wd_a.astype(float), wd_b.astype(float), wd_c.astype(float), wd_d.astype(float), wd_e.astype(float), "Direction ‚ True")


################################################################################################################################
# A2.1 # Plot PMF, PDF and CDF for the 5 sensors Temperature values in independent plots                                       #
################################################################################################################################

x = 1 

def pmf(sample):
 	c = sample.value_counts()
 	p = c/len(sample)
 	return p
for i in Temperatures:
    #plot Probability Mass Function (PMF)
    df = pmf(i)
    c = df.sort_index()
    fig = plt.figure(figsize=(17,6))
    ax1 = fig.add_subplot(111)
    ax1.bar(c.index,c, width=0.1,edgecolor='k')
    ax1.set_ylabel('Probability')
    ax1.set_xlabel('Temperatures [°C]')
    plt.title(f'PMF, Sensor {x}')
    plt.show()

    #plot Probability Density Function (PDF)
    fig = plt.figure(figsize=(17,6))  
    ax1 = fig.add_subplot(111)
    a1=ax1.hist(x=i.astype(float),bins=27, density=True, color='b',alpha=0.7, rwidth=0.85)
    sns.distplot(i.astype(float), color='k',ax=ax1, hist=False) #KDE of pdf
    ax1.set_ylabel('Probability Density')
    ax1.set_xlabel('Temperatures [°C]')
    plt.title(f'PDF, Sensor {x}')
    plt.show()

    #plot Cumulative Density Functions (CDF)
    fig = plt.figure(figsize=(17,6))
    ax1 = fig.add_subplot(111)
    a1=ax1.hist(x=i.astype(float),bins=27, cumulative=True, color='b',alpha=0.7, rwidth=0.85)
    ax1.plot(a1[1][1:]-(a1[1][1:]-a1[1][:-1])/2,a1[0], color='k')
    ax1.set_ylabel('CDF')
    ax1.set_xlabel('Temperatures [°C]')
    plt.title(f'CDF, Sensor {x}')
    plt.show()
    x += 1


################################################################################################################################
# A2.2 # For the Wind Speed values, plot the pdf and the kernel density estimation                                             # 
################################################################################################################################

x=1
for i in WindSpeed:
   #plot pdf
    fig = plt.figure(figsize=(17,6))  
    ax1 = fig.add_subplot(111)
    a1=ax1.hist(x=i.astype(float),bins=27, density=True, color='b',alpha=0.7, rwidth=0.85)
    sns.distplot(i.astype(float), color='k',ax=ax1, hist=False,
            hist_kws={'edgecolor':'black'},
            kde_kws={'linewidth': 2.5}) #KDE of pdf
    ax1.set_ylabel('Frequency')
    ax1.set_xlabel('Wind Speed [m/s]')
    plt.title(f'PDF, Sensor {x}')
    plt.show()
    #plot KDE
    sns.distplot(i, hist=True, kde=True, bins=27, color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 2.5})#ax=axes[0,1])
    ax1.set_ylabel('Frequency')
    ax1.set_xlabel('Wind Speed [m/s]')
    plt.title(f'KDE, Sensor {x}')
    plt.show()
    x += 1

################################################################################################################################
# A3.1 # Compute the correlations between all the sensors for the variables: Temperature, Wet Bulb Globe Temperature (WBGT),   #
# #### # Crosswind Speed                                                                                                       # 
################################################################################################################################

lista=["Direction ‚ True", "Wind Speed", "Crosswind Speed", "Headwind Speed",
 "Temperature", "Globe Temperature","Wind Chill","Relative Humidity","Heat Stress Index",
 "Dew Point","Psychro Wet Bulb Temperature","Station Pressure","Barometric Pressure",
 "Altitude", "Density Altitude", "NA Wet Bulb Temperature", "WBGT", "TWL", "Direction ‚ Mag"]

lista1=['A-B','A-C','A-D','A-E','B-C','B-D','B-E','C-D','C-E','D-E']

def corr(a,b,c,d,e,t):
    
    ab = np.interp(np.linspace(0,len(b),len(b)),np.linspace(0,len(a),len(a)),a)
    ac = np.interp(np.linspace(0,len(c),len(c)),np.linspace(0,len(a),len(a)),a)
    ad = np.interp(np.linspace(0,len(d),len(d)),np.linspace(0,len(a),len(a)),a)
    ae = np.interp(np.linspace(0,len(e),len(e)),np.linspace(0,len(a),len(a)),a)

    bc = np.interp(np.linspace(0,len(c),len(c)),np.linspace(0,len(b),len(b)),b)
    bd = np.interp(np.linspace(0,len(d),len(d)),np.linspace(0,len(b),len(b)),b)
    be = np.interp(np.linspace(0,len(e),len(e)),np.linspace(0,len(b),len(b)),b)

    cd = np.interp(np.linspace(0,len(d),len(d)),np.linspace(0,len(c),len(c)),c)
    ce = np.interp(np.linspace(0,len(e),len(e)),np.linspace(0,len(c),len(c)),c)

    de = np.interp(np.linspace(0,len(e),len(e)),np.linspace(0,len(d),len(d)),d)

    # interpolate to equal size samples

    norm_ab = (ab - ab.mean())/ab.std()
    norm_ac = (ac - ac.mean())/ac.std()
    norm_ad = (ad - ad.mean())/ad.std()
    norm_ae = (ae - ae.mean())/ae.std()

    norm_bc = (bc - bc.mean())/bc.std()
    norm_bd = (bd - bd.mean())/bd.std()
    norm_be = (be - be.mean())/be.std()

    norm_cd = (cd - cd.mean())/cd.std()
    norm_ce = (ce - ce.mean())/ce.std()

    norm_de = (de - de.mean())/de.std()

    # normalize because variables have different units

    a_norm= (a-a.mean())/a.std()
    b_norm= (b-b.mean())/b.std()
    c_norm= (c-b.mean())/c.std()
    d_norm= (d-b.mean())/d.std()
    e_norm= (e-b.mean())/e.std()

    # compute statistics

    p=[]
    s=[]

    pcoef_ab = stats.pearsonr(norm_ab,b_norm)[0]
    prcoef_ab = stats.spearmanr(norm_ab,b_norm)[0]
    p.append(pcoef_ab)
    s.append(prcoef_ab)

    pcoef_ac = stats.pearsonr(norm_ac,c_norm)[0]
    prcoef_ac = stats.spearmanr(norm_ac,c_norm)[0]
    p.append(pcoef_ac)
    s.append(prcoef_ac)
  
    pcoef_ad = stats.pearsonr(norm_ad,d_norm)[0]
    prcoef_ad = stats.spearmanr(norm_ad,d_norm)[0]
    p.append(pcoef_ad)
    s.append(prcoef_ad)

    pcoef_ae = stats.pearsonr(norm_ae,e_norm)[0]
    prcoef_ae = stats.spearmanr(norm_ae,e_norm)[0]
    p.append(pcoef_ae)
    s.append(prcoef_ae)

    pcoef_bc = stats.pearsonr(norm_bc,c_norm)[0]
    prcoef_bc = stats.spearmanr(norm_bc,c_norm)[0]
    p.append(pcoef_bc)
    s.append(prcoef_bc)

    pcoef_bd = stats.pearsonr(norm_bd,d_norm)[0]
    prcoef_bd = stats.spearmanr(norm_bd,d_norm)[0]
    p.append(pcoef_bd)
    s.append(prcoef_bd)

    pcoef_be = stats.pearsonr(norm_be,e_norm)[0]
    prcoef_be = stats.spearmanr(norm_be,e_norm)[0]
    p.append(pcoef_be)
    s.append(prcoef_be)

    pcoef_cd = stats.pearsonr(norm_cd,d_norm)[0]
    prcoef_cd = stats.spearmanr(norm_cd,d_norm)[0]
    p.append(pcoef_cd)
    s.append(prcoef_cd)

    pcoef_ce = stats.pearsonr(norm_ce,e_norm)[0]
    prcoef_ce = stats.spearmanr(norm_ce,e_norm)[0]
    p.append(pcoef_ce)
    s.append(prcoef_ce)

    pcoef_de = stats.pearsonr(norm_de,e_norm)[0]
    prcoef_de = stats.spearmanr(norm_de,e_norm)[0]
    p.append(pcoef_de)
    s.append(prcoef_de)
   
    # scatter plot dimensional variables
    
    #sns.set_theme(style="whitegrid")
    dict1 = {'Coeff':p,'Sensor Pair':lista1}
    df_p=pd.DataFrame(dict1)
    
    dict2 = {'Coeff':s,'Sensor Pair':lista1}
    df_s=pd.DataFrame(dict2)
    
    print(f"Pearson's coefficients \n {df_p}")
    print(f"Spearmann's coefficients \n {df_s}")
    
    fig, axes = plt.subplots(figsize=(17,7),sharey=True)
    ax1 = fig.add_subplot(121)
    ax1 = sns.stripplot(x="Sensor Pair", y="Coeff", data=df_s)
    ax2 = fig.add_subplot(122)
    ax2 = sns.stripplot(x="Sensor Pair", y="Coeff", data=df_p)
    
    ax1.set_xlabel('Sensor - Pairs')
    ax1.set_ylabel("Spearman correlation")

    ax2.set_xlabel('Sensor - Pairs')
    ax2.set_ylabel("Pearson correlation")
    
    fig.suptitle(t)
    fig.tight_layout()
    
    plt.show()
corr(temp_a.astype(float),temp_b.astype(float),temp_c.astype(float),temp_d.astype(float),temp_e.astype(float),"Temperature - sensor correlation")
corr(wbg_a.astype(float),wbg_b.astype(float),wbg_c.astype(float),wbg_d.astype(float),wbg_e.astype(float),"WBGT - sensor correlation")
corr(cws_a.astype(float),cws_b.astype(float),cws_c.astype(float),cws_d.astype(float),cws_e.astype(float),"Cross Wind Speed - sensor correlation")

################################################################################################################################
# A4.1 # Plot the CDF for all the sensors and for variables Temperature and Wind Speed                                         # 
################################################################################################################################

#  CDF for Temperature 
def cdf_temp(a, b, c, d, e):
    fig = plt.figure(figsize=(20,8))
    plt.subplots_adjust(wspace = 0.3, hspace = 0.5)
    fig.suptitle("CDF Temperature [°C]")
    
    ax1 = fig.add_subplot(321)
    a1=ax1.hist(x=a.astype(float),bins=27, cumulative=True, color='b',alpha=0.7, rwidth=0.85)
    ax1.plot(a1[1][1:]-(a1[1][1:]-a1[1][:-1])/2,a1[0], color='k')
    ax1.set_ylabel("CDF")
    ax1.set_xlabel("Temperature [°C]")
    plt.title('CDF, Sensor A')

    ax2 = fig.add_subplot(322)
    a2=ax2.hist(x=b.astype(float),bins=27, cumulative=True, color='b',alpha=0.7, rwidth=0.85)
    ax2.plot(a2[1][1:]-(a2[1][1:]-a2[1][:-1])/2,a2[0], color='k')
    ax2.set_ylabel("CDF")
    ax2.set_xlabel("Temperature [°C]")
    plt.title('CDF, Sensor B')

    ax3 = fig.add_subplot(323)
    a3=ax3.hist(x=c.astype(float),bins=27, cumulative=True, color='b',alpha=0.7, rwidth=0.85)
    ax3.plot(a3[1][1:]-(a3[1][1:]-a3[1][:-1])/2,a3[0], color='k')
    ax3.set_ylabel("CDF")
    ax3.set_xlabel("Temperature [°C]")
    plt.title('CDF, Sensor C')
    
    ax4 = fig.add_subplot(324)
    a4=ax4.hist(x=d.astype(float),bins=27, cumulative=True, color='b',alpha=0.7, rwidth=0.85)
    ax4.plot(a4[1][1:]-(a4[1][1:]-a4[1][:-1])/2,a4[0], color='k')
    ax4.set_ylabel("CDF")
    ax4.set_xlabel("Temperature [°C]")
    plt.title('CDF, Sensor D')

    ax5 = fig.add_subplot(325)
    a5=ax5.hist(x=e.astype(float),bins=27, cumulative=True, color='b',alpha=0.7, rwidth=0.85)
    ax5.plot(a5[1][1:]-(a5[1][1:]-a5[1][:-1])/2,a5[0], color='k')
    ax5.set_ylabel("CDF")
    ax5.set_xlabel("Temperature [°C]") 
    plt.title('CDF, Sensor E')
    plt.show()

#  CDF for Wind Speed
def cdf_ws(a, b, c, d, e):
    fig = plt.figure(figsize=(20,8))
    plt.subplots_adjust(wspace = 0.3, hspace = 0.5)
    fig.suptitle("CDF Wind Speed [m/s]")
    
    ax1 = fig.add_subplot(321)
    a1=ax1.hist(x=a.astype(float),bins=27, cumulative=True, color='b',alpha=0.7, rwidth=0.85)
    ax1.plot(a1[1][1:]-(a1[1][1:]-a1[1][:-1])/2,a1[0], color='k')
    ax1.set_ylabel("CDF")
    ax1.set_xlabel("Wind Speed [m/s]")
    plt.title('CDF, Sensor A')

    ax2 = fig.add_subplot(322)
    a2=ax2.hist(x=b.astype(float),bins=27, cumulative=True, color='b',alpha=0.7, rwidth=0.85)
    ax2.plot(a2[1][1:]-(a2[1][1:]-a2[1][:-1])/2,a2[0], color='k')
    ax2.set_ylabel("CDF")
    ax2.set_xlabel("Wind Speed [m/s]")
    plt.title('CDF, Sensor B')

    ax3 = fig.add_subplot(323)
    a3=ax3.hist(x=c.astype(float),bins=27, cumulative=True, color='b',alpha=0.7, rwidth=0.85)
    ax3.plot(a3[1][1:]-(a3[1][1:]-a3[1][:-1])/2,a3[0], color='k')
    ax3.set_ylabel("CDF")
    ax3.set_xlabel("Wind Speed [m/s]")
    plt.title('CDF, Sensor C')
    
    ax4 = fig.add_subplot(324)
    a4=ax4.hist(x=d.astype(float),bins=27, cumulative=True, color='b',alpha=0.7, rwidth=0.85)
    ax4.plot(a4[1][1:]-(a4[1][1:]-a4[1][:-1])/2,a4[0], color='k')
    ax4.set_ylabel("CDF")
    ax4.set_xlabel("Wind Speed [m/s]")
    plt.title('CDF, Sensor D')

    ax5 = fig.add_subplot(325)
    a5=ax5.hist(x=e.astype(float),bins=27, cumulative=True, color='b',alpha=0.7, rwidth=0.85)
    ax5.plot(a5[1][1:]-(a5[1][1:]-a5[1][:-1])/2,a5[0], color='k')
    ax5.set_ylabel("CDF")
    ax5.set_xlabel("Wind Speed [m/s]")
    plt.title('CDF, Sensor E')
    plt.show()

cdf_temp(temp_a.astype(float),temp_b.astype(float),temp_c.astype(float),temp_d.astype(float),temp_e.astype(float))
cdf_ws(temp_a.astype(float),temp_b.astype(float),temp_c.astype(float),temp_d.astype(float),temp_e.astype(float))

################################################################################################################################
# A4.2 # Compute the 95% confidence intervals for variables Temperature and Wind Speed for all the sensors                     #
# #### # and save them in a .txt-file                                                                                          # 
################################################################################################################################

conf_temp = (temp_a.astype(float).values, temp_b.astype(float).values, temp_c.astype(float).values, temp_d.astype(float).values, temp_e.astype(float).values)
conf_ws = (ws_a.astype(float).values, ws_b.astype(float).values, ws_c.astype(float).values, ws_d.astype(float).values, ws_e.astype(float).values)

f = open("confidence_int.txt", "a")
#def conf(a,b,c,d,e):
for i in conf_temp:
    confidence = 0.95
    data = i        

    n = len(data)
    m = np.mean(data)
    std_err = sem(data)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)

    start1 = m - h
    end1 = m + h
    print(f'Wind Speed, start: {start1}, end: {end1}')
    f.write("Confidence Intervals, Temp:" + str(start1)+"," + str(end1) + "\n")

for i in conf_ws:
    confidence = 0.95
    data = i        

    n = len(data)
    m = np.mean(data)
    std_err = sem(data)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)

    start2 = m - h
    end2 = m + h
    print(f'Wind Speed, start: {start2}, end: {end2}')
    f.write("Confidence Intervals, WS:" + str(start2)+"," + str(end2) + "\n")
f.close()     

################################################################################################################################
# A4.3# Test the hypothesis for the sensors: E,D; D,C; C,B; B,A / Compute the p-values                                         #
################################################################################################################################

def student_t(arr1,arr2,pair):
    
    data = arr1, arr2, pair
    t, p=stats.ttest_ind(data[0],data[1])
    print(f'{data[2]} t-value: {t}, p-value: {p}')
    #return t, p

student_t(temp_e.astype(float).values, temp_d.astype(float).values, "Temp.: E,D")
student_t(temp_d.astype(float).values, temp_c.astype(float).values, "Temp.: D,C")
student_t(temp_c.astype(float).values, temp_b.astype(float).values, "Temp.: C,B")
student_t(temp_b.astype(float).values, temp_a.astype(float).values, "Temp.: B,A")

student_t(ws_e.astype(float).values, ws_d.astype(float).values, "WS: E,D")
student_t(ws_d.astype(float).values, ws_c.astype(float).values, "WS: D,C")
student_t(ws_c.astype(float).values, ws_b.astype(float).values, "WS: C,B")
student_t(ws_b.astype(float).values, ws_a.astype(float).values, "WS: B,A")


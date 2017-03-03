'''
Test to evaluate wether if the evaluation suite is working correctly
'''
import ModelEvaluationSuite as mes
reload(mes)
import Jesuslib as jl
import numpy as np
import matplotlib.pyplot as plt


header=1
data_terrestrial=np.genfromtxt('/Users/jesusvergaratemprado/work/INP_DATA/TERRESTIAL_INFLUENCED.dat',delimiter="\t",skip_header=header)
data_marine=np.genfromtxt('/Users/jesusvergaratemprado/work/INP_DATA/MARINE_INFLUENCED.dat',delimiter="\t",skip_header=header)

'''
Dimensions
Temperature,pressure,latitude,longitude
'''
dimension_names=['Temperature','pressure','latitude','longitude']


terrestrial_data=np.zeros((len(data_terrestrial[:,0]),5))
terrestrial_data[:,0]=data_terrestrial[:,2]#first column correspond to the values
terrestrial_data[:,1]=data_terrestrial[:,1]#Temperature
terrestrial_data[:,2]=data_terrestrial[:,5]#pressure
terrestrial_data[:,3]=data_terrestrial[:,3]#Latitude
terrestrial_data[:,4]=data_terrestrial[:,4]#longitude
terrestrial_array=mes.Data_Array(terrestrial_data,dimensions_name=dimension_names)


marine_data=np.zeros((len(data_marine[:,0]),5))
marine_data[:,0]=data_marine[:,2]#first column correspond to the values
marine_data[:,1]=data_marine[:,1]#Temperature
marine_data[:,2]=data_marine[:,5]#pressure
marine_data[:,3]=data_marine[:,3]#Latitude
marine_data[:,4]=data_marine[:,4]#longitude
marine_array=mes.Data_Array(marine_data,dimensions_name=dimension_names)


# INP_marine=np.load('/Users/jesusvergaratemprado/work/INP/INP_marine_alltemps.npy')
# INP_marine_year_mean=INP_marine.mean(axis=-1)
# np.save('/Users/jesusvergaratemprado/work/INP/INP_marine_alltemps_year_mean_cm.npy',INP_marine_year_mean*1e-6)
INP_marine=np.load('/Users/jesusvergaratemprado/work/INP/INP_marine_alltemps_year_mean_cm.npy')
# INP_feld_ext=np.load('/Users/jesusvergaratemprado/work/INP/INP_feld_ext_alltemps.npy')
# INP_feld_ext_year_mean=INP_feld_ext.mean(axis=-1)
# np.save('/Users/jesusvergaratemprado/work/INP/INP_feld_ext_alltemps_year_mean_cm.npy',INP_feld_ext_year_mean)
INP_feld_ext=np.load('/Users/jesusvergaratemprado/work/INP/INP_feld_ext_alltemps_year_mean_cm.npy')

INP_total=INP_marine+INP_feld_ext
print INP_total.shape

INP_glomap_dimensions=[]
temperatures=np.arange(0,-38,-1)
INP_glomap_dimensions.append(temperatures)
pressures=jl.pressure
pressures[-1]=1001
INP_glomap_dimensions.append(pressures)
INP_glomap_dimensions.append(jl.lat)

longitudes=jl.lon
longitudes[longitudes>180]=longitudes[longitudes>180]-360
INP_glomap_dimensions.append(longitudes)

glomap_array=mes.Model_Array(data=INP_total,dimensions=INP_glomap_dimensions,dimensions_name=dimension_names)


modelled_values,observed_values,dimensional_values=mes.Evaluate_data(M_array=glomap_array,D_array=terrestrial_array)
modelled_values_marine,observed_values_marine,dimensional_values_marine=mes.Evaluate_data(M_array=glomap_array,D_array=marine_array)



plt.figure()

# plt.plot(modelled_values,observed_values,'bo')
plot=plt.scatter(observed_values,modelled_values,c=dimensional_values[:,0],cmap=plt.cm.RdBu_r,marker='o',s=10,vmin=-38, vmax=0)
plot=plt.scatter(observed_values_marine,modelled_values_marine,c=dimensional_values_marine[:,0],cmap=plt.cm.RdBu_r,marker='^',s=10,vmin=-38, vmax=0)
plt.colorbar()
min_val=1e-9
max_val=1e1
minx=np.min(min_val)
maxx=np.max(max_val)
miny=np.min(min_val)
maxy=np.max(max_val)
min_plot=min_val
max_plot=max_val

x=np.linspace(0.1*min_plot,10*max_plot,100)
plt.plot(x,x,'k-')
plt.plot(x,10*x,'k--')
plt.plot(x,10**1.5*x,'k-.')
plt.plot(x,0.1*x,'k--')
plt.plot(x,10**(-1.5)*x,'k-.')
plt.ylim(miny*0.1,maxy*10)
plt.xlim(minx*0.1,maxx*10)

plt.ylabel('Simulated [INP] ($cm^{-3}$)')
plt.xlabel('Observed [INP] ($cm^{-3}$)')
plt.xscale('log')
plt.yscale('log')

plt.savefig('GLOMAP_with_feldspar_marine.png')
plt.show()









#

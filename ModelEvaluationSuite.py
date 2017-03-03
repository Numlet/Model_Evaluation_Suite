'''
Model Evaluation Suite (MES)

Suite for evaluating modeling results with atmospheric datasets.

Jesus Vergara-Temprado

Institute for climate and atmospheric science (ICAS)

University of Leeds
2017
'''
import numpy as np
import scipy

def isascending(array):
    return sorted(array)==array.tolist()

class Model_Array():
    '''
    Class for structuring model array data
    -data: refers to the modelled values of the variable to study (mass concentration, temperature, water content ....)
    -dimensions:  a list of arrays with every array being the coordinate points of every dimension,
     the same order as in the data dimension is required
     Ex:
        If data has a shape (12,30,64,128) refering to time, level, latitude and longitude,
        then, dimensions have to be composed of a list with and array corresponding to the time values in the fist place,
        an array with the level values in the second and so on...


    '''
    def __init__(self,data,dimensions,model_name="Unknown",dimensions_name=0):
        self.ndim=len(dimensions)
        self.data=data
        self.dimensions=dimensions
        self.model_name=model_name
        self.dimensions_name=dimensions_name
        self.ascending_dim=[]
        if dimensions_name:
            if len(dimensions_name)!=len(dimensions):
                raise StandardError('The number of dimensions is different from the number of dimension names given')
        if self.data.ndim!=self.ndim:
            raise ValueError('Number of dimensions of data %s different from number of dimensions given: %i'%(self.data.shape,self.ndim))
        for i in range(self.ndim):
            if self.data.shape[i]!=len(self.dimensions[i]):
                raise ValueError('Data size in dimension %i not the same as the size of dimension %i'%(i,i))
        for dim in self.dimensions:
            if isascending(dim):
                self.ascending_dim.append(1)
            else:
                self.ascending_dim.append(0)


class Data_Array():
    '''
    Data array corresponding to atmospheric measurements

    The data given has to have the following shape: (N,D+1)
    where N is the number of observations and D is the number of dimensions on to which interpolate/compare to model data
    data[:,0] has to correspond to the values of the variable to compare
    '''
    def __init__(self,data,dimensions_name=0,info='None'):
        self.data=data
        if dimensions_name:
            self.dimensions_name=dimensions_name
            if len(dimensions_name)!=len(data[0,:])-1:
                raise StandardError('The number of dimensions is different from the number of dimension names given')
        self.info=info
        self.values=data[:,0]
        self.dimensions=data[:,1:]
        self.ndim=len(self.dimensions[0,:])


def find_nearest_vector_index(array, value):
    n = np.array([abs(i-value) for i in array])
    nindex=np.apply_along_axis(np.argmin,0,n)
    return nindex



from itertools import product

def interpolator(coords, data, point) :
    dims = len(point)
    indices = []
    sub_coords = []
    for j in xrange(dims) :
        idx = np.digitize([point[j]], coords[j])[0]
        indices += [[idx - 1, idx]]
        sub_coords += [coords[j][indices[-1]]]
    indices = np.array([j for j in product(*indices)])
    sub_coords = np.array([j for j in product(*sub_coords)])
    sub_data = data[list(np.swapaxes(indices, 0, 1))]
    li = LinearNDInterpolator(sub_coords, sub_data)
    return li([point])[0]

from scipy.interpolate import RegularGridInterpolator


def Evaluate_data(M_array,D_array,method='Nearest'):
    if M_array.ndim!=D_array.ndim:
        raise StandardError('Number of dimensions of Model data (%i) and observations (%i) not the same'%(M_array.ndim,D_array.ndim))
    for i in range(M_array.ndim):
        max_val=M_array.dimensions[i].max()
        min_val=M_array.dimensions[i].min()
        if any(D_array.dimensions[:,i]>max_val) or any (D_array.dimensions[:,i]<min_val):
            print ('WARNING: Observational data has some values with dimension %i not within \
             the range of values of modelling data for that dimension \
             \n max: %f  \n min: %f  '%(i,max_val,min_val))
            if M_array.dimensions_name:
                print M_array.dimensions_name[i]
    modelled_values=[]
    if method=='Nearest':
        for i in range(len(D_array.values)):
            index_list=[]
            for idim in range (M_array.ndim):
                indxDim=find_nearest_vector_index(M_array.dimensions[idim],D_array.dimensions[i,idim])
                index_list.append(indxDim)
            modelled_values.append(M_array.data[tuple(index_list)])
    elif method=='Linear':
        model_dimensions=[]
        data_dimensions=D_array.dimensions
        for i in range(len(M_array.dimensions)):
            if M_array.ascending_dim[i]:
                model_dimensions.append(M_array.dimensions[i])
            else:
                model_dimensions.append(M_array.dimensions[i]*-1)
                data_dimensions[i]=data_dimensions[i]*-1
                print 'Corrected dimension %i to make is ascendant'%i
                print isascending(model_dimensions[i])
        interpolating_function = RegularGridInterpolator(model_dimensions, M_array.data)
        modelled_values=interpolating_function(data_dimensions).tolist()
        # for i in range(len(D_array.values)):
            # point=D_array.dimensions[i]
            # modelled_values.append(interpolator(M_array.dimensions,M_array.data,point))
    else: raise StandardError('Method non existing or not yet implemented')
    return modelled_values,D_array.values,D_array.dimensions




# >>> point = np.array([12.3,-4.2, 500.5, 2.5])
# >>> interpolator((lats, lons, alts, time), data, point)


#

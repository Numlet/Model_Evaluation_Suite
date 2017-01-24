'''
Model Evaluation Suite (MES)

Suite for evaluating modeling results with atmospheric datasets.

Jesus Vergara-Temprado

Institute for climate and atmospheric science (ICAS)

University of Leeds
2017
'''


class Model_Array():
    def __init__(self,data,dimensions,model_name="Unknown",dimensions_name=None):
        self.ndim=len(dimensions)
        self.data=data
        self.dimensions=dimensions
        self.model_name=model_name
        if dimensions_name:
            self.dimensions_name=dimensions_name
            if len(dimensions_name)!=len(dimensions):
                raise StandardError('The number of dimensions is different from the number of dimension names given')
        if self.data.ndim!=self.ndim:
            raise ValueError('Number of dimensions of data %s different from number of dimensions given: %i'%(self.data.shape,self.ndim))
        for i in range(self.ndim):
            if self.data.shape[i]!=len(self.dimensions[i]):
                raise ValueError('Data size in dimension %i not the same as the size of dimension %i'%(i,i))

class Data_Array():
    '''
    Data array corresponding to atmospheric measurements

    The data given has to be consistent of an array (N,D+1)
    where N is the number of observations and D is the number of dimensions on to which interpolate/compare to model data
    data[:,0] has to correspond to the values of the variable to compare
    '''
    def __init__(self,data,dimensions_name=None,info='None'):
        self.data=data
        if dimensions_name:
            self.dimensions_name=dimensions_name
            if len(dimensions_name)!=len(data[0,:])-1:
                raise StandardError('The number of dimensions is different from the number of dimension names given')
        self.info=info
        self.values=data[:,0]
        self.dimensions=data[:,1:]
        self.ndim=self.dimensions.ndim


def find_nearest_vector_index(array, value):
    n = np.array([abs(i-value) for i in array])
    nindex=np.apply_along_axis(np.argmin,0,n)
    return nindex


def Evaluate_data(M_array,D_array,method='Nearest'):
    if M_array.ndim!=D_array.ndim:
        raise StandardError('Number of dimensions of Model data and observations not the same')
    for i in range(M_array.dimensions):
        max_val=M_array.dimensions[i].max()
        min_val=M_array.dimensions[i].min()
        if any(D_array.dimensions[:,i]>max_val) or any (D_array.dimensions[:,i]<min_val):
            raise StandardError('Observational data has some values with dimension %i not within\
             the range of values of modelling data for that dimension')
    if method=='Nearest':
        modelled_values=[]
        for i in range(D_array.values):
            index_list=[]
            for idim in range (M_array.ndim):
                indxDim=find_nearest_vector_index(M_array.dimensions[idim],D_array.dimensions[i,idim])
                index_list.append(indxDim)
            modelled_values.append(M_array.data[tuple(index_list)])
        return modelled_values,D_array.values,D_array.dimensions
    else: raise StandardError('Method non existing or not yet implemented')




#

#Collaboration:Deepthi Devaraj,Sushanti Prabhu
import numpy as np


def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''

    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data-  numpy array of points
    :param generator: random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.


    :return: the center points array of length n_clusters with each entry being index to a sample
             which is chosen as centroid.
    '''
    # TODO:
    # implement the Kmeans++ algorithm of how to choose the centers according to the lecture and notebook
    # Choose 1st center randomly and use Euclidean distance to calculate other centers.
    centers=[]
    dist=[]
    prob=[]
    first_center=generator.randint(n-1)
    centers.append(first_center)
    for i in range(0,n_cluster):
        dist=compute_distance(x[centers[len(centers)-1]],x,centers)
        prob=compute_prob(dist,sum(dist))
        next_center=np.argmax(prob)
        centers.append(next_center)
    #raise Exception(
             #'Implement get_k_means_plus_plus_center_indices function in Kmeans.py')

    

    # DO NOT CHANGE CODE BELOW THIS LINE

    print("[+] returning center for [{}, {}] points: {}".format(n, len(x), centers))
    return centers

def compute_distance(center,points,centers):
    distances=[]
    dis=0
    curr_point=[]
    for i in range(0,len(points)):
        curr_point=points[i]
        if i not in centers:
            '''for dim1 in range(0,len(curr_point)):
                for dim2 in range(0,len(center)):
                              dis+=(curr_point[dim1]-center[dim2])**2
            distances.append(dis**1/2)'''
            distances.append(np.linalg.norm(curr_point-center))
        else:
            distances.append(0)
            
    return distances

def compute_prob(distances,sum_dist):
    prob=[]
    for i in range(0,len(distances)):
        prob.append(np.divide(distances[i],sum_dist))
    return prob

def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)




class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''
    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)
            returns:
                A tuple
                (centroids a n_cluster X D numpy array, y a length (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates a Int)
            Note: Number of iterations is the number of time you update the assignment
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
        euc_dis = np.zeros((N, self.n_cluster))
        y_pos = np.arange(x.shape[0])
        new_means =  np.random.choice(y_pos,self.n_cluster,True)
        # means = np.take(x, means_indices)
        centroids= x[new_means,:]

        y = np.zeros(N)
        j= np.power(10,10)
        # np.sum([np.sum((x[membership == k]-means[k]) **2)  for k in range(self.n_cluster)])/ x.shape[0]
        euc_dis = np.sum ((x - centroids[:,np.newaxis]) ** 2,axis=2)
        updates=0
        i=0
        while (i<self.max_iter):
            euc_dis= np.sum((x - centroids[:, np.newaxis]) ** 2, axis=2)
            y = np.argmin(euc_dis, axis=0)
            distortion  =  calc_distortion(x,y,centroids,self.n_cluster)
            
            if np.absolute(j-distortion) <= self.e:
                break
            j = distortion
            centroids =new_centers(x,y,self.n_cluster)
            
            updates+=1
            i+=1
        return (centroids,y,updates)
        #raise Exception(
             #'Implement fit function in KMeans class')
        
        # DO NOT CHANGE CODE BELOW THIS LINE
        return centroids, y, self.max_iter

        
def calc_distortion(x,y,centroids,total_clusters):
    return np.sum([np.sum((x[y == clus]-centroids[clus]) **2)  for clus in range(total_clusters)])/ x.shape[0]


def new_centers(x,y,total_clusters):
    return np.array([np.mean(x[y == clus],axis=0 ) for clus in range(total_clusters)])


def compute_assignment(point, centers):
    min_d=10000
    clus=0
    for cen in range(0,len(centers)):
        if min_d>np.linalg.norm(point-centers[cen]):
            min_d=np.linalg.norm(point-centers[cen])
            clus=cen
    return clus
        
def new_mean(cluster,assign,points):
    total=0
    count=0
    for x in range(0,len(assign)):
        if assign[x]==cluster:
            total+=points[x]
            count+=1
    return total/count


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results,
            We will be using our own random number generator.
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator


    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)

            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (N,) numpy array)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        

        kmeans = KMeans(self.n_cluster,self.max_iter,self.e)
        (centroids, membership, num_iter) = kmeans.fit(x)

        votes =[{} for i in range(0, len(centroids))]
        for idx in range(len(membership)):
            if y[idx] not in  votes[membership[idx]]:
                votes[membership[idx]][y[idx]] =1
            else:
                votes[membership[idx]][y[idx]] +=1
        centroid_labels =[]

        for centr in votes:
            if centr:
                predicted_label = max(centr, key=centr.get)
                centroid_labels.append(predicted_label)
            else:
                centroid_labels.append(0)
        centroid_labels =np.array(centroid_labels)
        #raise Exception(
             #'Implement fit function in KMeansClassifier class')

        

        
        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels
        euc_dis = np.sum((x - self.centroids[:, np.newaxis]) ** 2, axis=2)
        y = np.argmin(euc_dis, axis=0)
        labels = self.centroid_labels[y]

        # DONOT CHANGE CODE ABOVE THIS LINE
        #raise Exception(
             #'Implement predict function in KMeansClassifier class')
        
        
        # DO NOT CHANGE CODE BELOW THIS LINE
        return labels
        

def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors

        Return new image from the image by replacing each RGB value in image with nearest code vectors (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'

    # TODO
    # - comment/remove the exception
    # - implement the function

    # DONOT CHANGE CODE ABOVE THIS LINE
    (a,b,c) = image.shape
    image = image_reshape(image)
    #
    euc_dis = calc_dis(image,code_vectors)
    np.sum((image - code_vectors[:, np.newaxis]) ** 2, axis=2)
    membership = np.argmin(euc_dis, axis=0)
    new_im = code_vectors[membership].reshape(a ,b ,c )
    #raise Exception(
             #'Implement transform_image function')
    

    # DONOT CHANGE CODE BELOW THIS LINE
    return new_im
def image_reshape(image):
    return image.reshape(image.shape[0] * image.shape[1] ,image.shape[2])

def calc_dis(image,code_vectors):
    return np.sum((image - code_vectors[:, np.newaxis]) ** 2, axis=2)
    


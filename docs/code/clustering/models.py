
"""MLEngine Classes for Clustering

barn-owl provides algorithms for the purpose of clustering with endpoints `/v1/clustering/<algorithm_name>`.

"""
__author__ = "G.Ito <ito@supreme-system.com>"
__version__ = "1.00"
__date__ = "4 Feb. 2016"
__status__ = "production"

from django.db import models
from jsonfield import JSONField
from django.core.exceptions import ValidationError

'''model's status'''
STATUS = ['', 'Created', 'Submitted Learning Request', 'Learning-in', 'Learned', 'Error']
STATUS_CHOICES=[(item,item) for item in STATUS]


class MLEngine(models.Model):
    '''
    MLEngine class for Virtual Basic
    '''
    owner = models.ForeignKey('auth.User', related_name='clustering_basic_models')
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now = True)
    title = models.CharField(max_length=100, blank=True, default='')
    code = models.TextField(blank=True)
    status = models.CharField(choices=STATUS_CHOICES, default=STATUS_CHOICES[1][1], max_length=100)
    train_data = models.FileField(upload_to='train_data', blank=True)
    trained_data=JSONField()
    
    def __unicode__(self):
        return self.title
    class Meta:
        ordering = ('created',)

        
class MLEngineKMeans(models.Model):
    """**KMeans**
    
    KMeans clustering. The number of clustering should be given by `n_clusters`.
    
    endpoint
        `/v1/clustering/kmeans/`
            
    Parameters:
        id (int, read only) : Unique and sequential number which is given automatically.
        title (char[100], optional, default='') : A title of the model.
        code (text, optional, default='') : An explanation of the model.
        owner (ForeignKey, read only) : Owner's username of the model.
        created (DateTime, read only) : Universal datetime when the model was created.
        updated (DateTime, read only) : Universal datetime when the model was updated.
        status (ChoiceField, read only) : Status represented from among the state list:[``Created``, ``Submitted Learning Request``, ``Learning-in``, ``Learned``, ``Error``].
            Each status restricts some of request as follows,
            
            * ``Created`` means a model is created. `GET`, `PUT` and `DELETE` method are accepted. `GET` method with query parameter mode=evaluate , however, is NOT accepted.
            * ``Submitted Learning Request`` means that `PUT` method with query mode=learn has been accepted. Only `GET` method with no query parameter is acceptable in this status.
            * ``Learning-in`` means fitting process for the model has been called. Only `GET` method with no query parameter is acceptable in this status.
            * ``Learned`` means fitting for the model has been done.
            * ``Error`` means fitting process has stopped by internal error.
            
        train_data (FileField) : Dataset used for fitting or evaluation.
        trained_data (list of FileField) : List of dataset already used for fitting process.
        n_clusters (int,default=3) : The number of clusters.
        features (list) : A features list for fitting.
        
    Returns:
        Json : list of tupples as follows,
        
        data (list of tuples) : 
            Predicted list of tuples consisting of `id` and `prediction`.
            
            id :
                user id
            prediction :
                predicted cluster's label of the learned model
        score (int) :
            This score represents coefficient :math:`R^2= 1-\\frac{\\sum_{i}(y_i - f_i)^2}{\\sum_{i}(y_i-\\overline{y})^2}` ,
            where :math:`y_i, f_i, \\overline{y}` is training data, predicted data and mean of training data respectively.
        cluster_centers(n_clusters,n_features) :
            The list of cluster's center.
        
    """
    owner = models.ForeignKey('auth.User', related_name='clustering_kmeans_models')
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now = True)
    title = models.CharField(max_length=100, blank=True, default='')
    code = models.TextField(blank=True)
    status = models.CharField(choices=STATUS_CHOICES, default=STATUS_CHOICES[1][1], max_length=100)
    features =  models.TextField(max_length=100, blank=True)
    train_data = models.FileField(upload_to='train_data', blank=True)
    trained_data = JSONField()
    n_clusters = models.IntegerField(default=3, blank=True)
    features =  models.TextField(max_length=100, blank=True)


class MLEngineMiniBatchKMeans(models.Model):
    """**BatchKMeans**
    
    BatchKMeans clustering. The number of clustering should be given by `n_clusters`.
    
    endpoint
        `/v1/clustering/batchkmeans/`
            
    Parameters:
        id (int, read only) : Unique and sequential number which is given automatically.
        title (char[100], optional, default='') : A title of the model.
        code (text, optional, default='') : An explanation of the model.
        owner (ForeignKey, read only) : Owner's username of the model.
        created (DateTime, read only) : Universal datetime when the model was created.
        updated (DateTime, read only) : Universal datetime when the model was updated.
        status (ChoiceField, read only) : Status represented from among the state list:[``Created``, ``Submitted Learning Request``, ``Learning-in``, ``Learned``, ``Error``].
            Each status restricts some of request as follows,
            
            * ``Created`` means a model is created. `GET`, `PUT` and `DELETE` method are accepted. `GET` method with query parameter mode=evaluate , however, is NOT accepted.
            * ``Submitted Learning Request`` means that `PUT` method with query mode=learn has been accepted. Only `GET` method with no query parameter is acceptable in this status.
            * ``Learning-in`` means fitting process for the model has been called. Only `GET` method with no query parameter is acceptable in this status.
            * ``Learned`` means fitting for the model has been done.
            * ``Error`` means fitting process has stopped by internal error.
            
        train_data (FileField) : Dataset used for fitting or evaluation.
        trained_data (list of FileField) : List of dataset already used for fitting process.
        n_clusters (int, default=3) : The number of clusters.
        features (list) : A features list for fitting.
        
    Returns:
        Json : list of tupples as follows,
        
        data (list of tuples) : 
            Predicted list of tuples consisting of `id` and `prediction`.
            
            id :
                user id
            prediction :
                predicted cluster's label of the learned model
        score (int) :
            This score represents coefficient :math:`R^2= 1-\\frac{\\sum_{i}(y_i - f_i)^2}{\\sum_{i}(y_i-\\overline{y})^2}` ,
            where :math:`y_i, f_i, \\overline{y}` is training data, predicted data and mean of training data respectively.
        cluster_centers(n_clusters,n_features) :
            The list of cluster's center.
        
    """
    owner = models.ForeignKey('auth.User', related_name='clustering_batchkmeans_models')
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now = True)
    title = models.CharField(max_length=100, blank=True, default='')
    code = models.TextField(blank=True)
    status = models.CharField(choices=STATUS_CHOICES, default=STATUS_CHOICES[1][1], max_length=100)
    features =  models.TextField(max_length=100, blank=True)
    train_data = models.FileField(upload_to='train_data', blank=True)
    trained_data = JSONField()
    n_clusters = models.IntegerField(default=3, blank=True)
    n_batches = models.IntegerField(default=100, blank=True)
    features =  models.TextField(max_length=100, blank=True)
    
    
class MLEngineMeanShift(models.Model):
    """**MeanShift**
    
    MeanShift clustering. The bandwidth of clustering can be given with `bandwidth`.
    
    endpoint
        `/v1/clustering/meanshift/`
            
    Parameters:
        id (int, read only) : Unique and sequential number which is given automatically.
        title (char[100], optional, default='') : A title of the model.
        code (text, optional, default='') : An explanation of the model.
        owner (ForeignKey, read only) : Owner's username of the model.
        created (DateTime, read only) : Universal datetime when the model was created.
        updated (DateTime, read only) : Universal datetime when the model was updated.
        status (ChoiceField, read only) : Status represented from among the state list:[``Created``, ``Submitted Learning Request``, ``Learning-in``, ``Learned``, ``Error``].
            Each status restricts some of request as follows,
            
            * ``Created`` means a model is created. `GET`, `PUT` and `DELETE` method are accepted. `GET` method with query parameter mode=evaluate , however, is NOT accepted.
            * ``Submitted Learning Request`` means that `PUT` method with query mode=learn has been accepted. Only `GET` method with no query parameter is acceptable in this status.
            * ``Learning-in`` means fitting process for the model has been called. Only `GET` method with no query parameter is acceptable in this status.
            * ``Learned`` means fitting for the model has been done.
            * ``Error`` means fitting process has stopped by internal error.
            
        train_data (FileField) : Dataset used for fitting or evaluation.
        trained_data (list of FileField) : List of dataset already used for fitting process.
        bandwidth (float, optional, default=0.0) : Bandwidth of RBF kernel. If zero or negative value is given, estimated value will be used in library.
        features (list) : A features list for fitting.
        
    Returns:
        Json : list of tupples as follows,
        
        data (list of tuples) : 
            Predicted list of tuples consisting of `id` and `prediction`.
            
            id :
                user id
            prediction :
                predicted cluster's label of the learned model
        cluster_centers(n_clusters,n_features) :
            The list of cluster's center.
        
    """
    owner = models.ForeignKey('auth.User', related_name='clustering_meanshift_models')
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now = True)
    title = models.CharField(max_length=100, blank=True, default='')
    code = models.TextField(blank=True)
    status = models.CharField(choices=STATUS_CHOICES, default=STATUS_CHOICES[1][1], max_length=100)
    features =  models.TextField(max_length=100, blank=True)
    train_data = models.FileField(upload_to='train_data', blank=True)
    trained_data = JSONField()
    features =  models.TextField(max_length=100, blank=True)
    bandwidth = models.FloatField(default=0.0, blank=True)


"""MLEngine Classes for Regression

barn-owl provides algorithms for the purpose of regression with endpoints `/v1/regression/<algorithm_name>`.

"""
__author__ = "G.Ito <ito@supreme-system.com>"
__version__ = "1.00"
__date__ = "17 Nov. 2015"
__status__ = "production"

from django.db import models
from jsonfield import JSONField
from django.core.exceptions import ValidationError

'''model's status'''
STATUS = ['', 'Created', 'Submitted Learning Request', 'Learning-in', 'Learned', 'Error']
STATUS_CHOICES=[(item,item) for item in STATUS]
ESTIMATOR = ['', 'Linear', 'DecisionTree', 'Lasso', 'Ridge', 'Bayesian']
ESTIMATOR_CHOICES = [(item,item) for item in ESTIMATOR]
DECOMPOSITION = ['','PCA']
DECOMPOSITION_CHOICES = [(item,item) for item in DECOMPOSITION]

class MLEngine(models.Model):
    """**Regression with DNN**
    
    Regression model based on deep neural network.
    
    endpoint
        `/v1/regression/dnn/`
            
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
        target (string) : Target column's name. Regression uses continuous variable as the target.
        
    Returns:
        Json : list of tupples as follows,
        
        data (list of tuples) : 
            Predicted list of tuples consisting of `id`, `prediction` and `target`.
            
            id :
                user id
            prediction :
                predicted value of the learned model
            target :
                target value of training dataset
        loss (float) :
            Sum of loss function.
        target (string) :
            Target column's name.
        
    """
    owner = models.ForeignKey('auth.User', related_name='regression_dnn_models')
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now = True)
    title = models.CharField(max_length=100, blank=True, default='')
    code = models.TextField(blank=True)
    status = models.CharField(choices=STATUS_CHOICES, default=STATUS_CHOICES[1][1], max_length=100)
    target = models.TextField(max_length=100, blank=True)
    train_data = models.FileField(upload_to='train_data', blank=True)
    trained_data=JSONField()
    
    def __unicode__(self):
        return self.title
    class Meta:
        ordering = ('created',)

class MLEngineLinear(models.Model):
    
    """**Linear Regression**

    Linear Regression fits a linear model with a dataset by least square method.
    If you set decomp_flg to `True`, decomposed features are used for fitting.

    endpoint
        `/v1/regression/linear/`
    
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
        target (string) : Target column's name. Regression uses continuous variable as the target.
        features (list) : A features list for fitting.
        decomp_flg (bool,default=False) : Decomposition flag. Default is false.
        decomp_components (int, optional, default=2) : The number of decomposition components.
        decomp_algorithm (ChoiceField, default='PCA') : Decomposition Algorithm. Only Principal Component Analysis(PCA) can be selected, for now.
        
    Returns:
        Json : list of tupples as follows,
        
        data (list of tuples) : 
            Predicted list of tuples consisting of `id`, `prediction` and `target`.
            
            id :
                user id
            prediction :
                predicted value of the learned model
            target :
                target value of training dataset
        features (list) :
            A features list used for training the model.
        target (string) :
            Target column's name.
        coefficient (list) :
            A coefficients list associated with each features element.
        intercept (int) :
            The value of intercept term.
        score (int) :
            This score represents coefficient :math:`R^2= 1-\\frac{\\sum_{i}(y_i - f_i)^2}{\\sum_{i}(y_i-\\overline{y})^2}` ,
            where :math:`y_i, f_i, \\overline{y}` is training data, predicted data and mean of training data respectively.
        
    """
    owner = models.ForeignKey('auth.User', related_name='regression_linear_models')
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now = True)
    title = models.CharField(max_length=100, blank=True, default='')
    code = models.TextField(blank=True)
    status = models.CharField(choices=STATUS_CHOICES, default=STATUS_CHOICES[1][1], max_length=100)
    target = models.TextField(max_length=100, blank=True)
    features =  models.TextField(max_length=100, blank=True)
    train_data = models.FileField(upload_to='train_data', blank=True)
    trained_data = JSONField()
    decomp_flg = models.NullBooleanField(default=False)
    decomp_components = models.IntegerField(default=2,blank=True)
    decomp_algorithm = models.CharField(choices=DECOMPOSITION_CHOICES,default=DECOMPOSITION_CHOICES[1][1], max_length=100)


class MLEngineLasso(models.Model):
    """**Lasso Regression**

    Lasso regression fits a linear sparse model with a dataset.
    If you set decomp_flg to `True`, decomposed features are used for fitting.
    
    endpoint
        `/v1/regression/lasso/`
    
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
        target (string) : Target column's name. Regression uses continuous variable as the target.
        features (list) : A features list for fitting.
        regularization (float, optional, default=1.0) : Regularization parameter of Lasso term.
        decomp_flg (bool, default=False) : Decomposition flag. Default is false.
        decomp_components (int, optional, default=2) : The number of decomposition components.
        decomp_algorithm (ChoiceField, default='PCA') : Decomposition Algorithm. Only Principal Component Analysis(PCA) can be selected, for now.
        
    Returns:
        Json : list of tupples as follows,
        
        data (list of tuples) : 
            Predicted list of tuples consisting of `id`, `prediction` and `target`.
            
            id :
                user id
            prediction :
                predicted value of the learned model
            target :
                target value of training dataset
        features (list) :
            A features list used for training the model.
        target (string) :
            Target column's name.
        regularization (float) : 
            Used regularization parameter of lasso term.
        coefficient (list) :
            A coefficients list associated with each features element.
        intercept (int) :
            The value of intercept term.
        score (int) :
            This score represents coefficient :math:`R^2= 1-\\frac{\\sum_{i}(y_i - f_i)^2}{\\sum_{i}(y_i-\\overline{y})^2}` ,
            where :math:`y_i, f_i, \\overline{y}` is training data, predicted data and mean of training data respectively.
        
    """
    owner = models.ForeignKey('auth.User', related_name='regression_lasso_models')
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now = True)
    title = models.CharField(max_length=100, blank=True, default='')
    code = models.TextField(blank=True)
    status = models.CharField(choices=STATUS_CHOICES, default=STATUS_CHOICES[1][1], max_length=100)
    target = models.TextField(max_length=100, blank=True)
    features =  models.TextField(max_length=100, blank=True)
    train_data = models.FileField(upload_to='train_data', blank=True)
    trained_data = JSONField()
    regularization = models.FloatField(blank=True , default=1.0)
    decomp_flg = models.NullBooleanField(default=False)
    decomp_components = models.IntegerField(default=2,blank=True)
    decomp_algorithm = models.CharField(choices=DECOMPOSITION_CHOICES,default=DECOMPOSITION_CHOICES[1][1], max_length=100)
        
class MLEngineRidge(models.Model):
    """**Ridge Regression**

    Ridge regression fits a linear dense model with a dataset.
    If you set decomp_flg to `True`, decomposed features are used for fitting.
     
    endpoint
        `/v1/regression/ridge/`
    
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
        target (string) : Target column's name. Regression uses continuous variable as the target.
        features (list) : A features list for fitting.
        regularization (float, optional, default=1.0) : Regularization parameter of Ridge term.
        decomp_flg (bool, default=False) : Decomposition flag. Default is false.
        decomp_components (int, optional, default=2) : The number of decomposition components.
        decomp_algorithm (ChoiceField, default='PCA') : Decomposition Algorithm. Only Principal Component Analysis(PCA) can be selected, for now.
        
    Returns:
        Json : list of tupples as follows,
        
        data (list of tuples) : 
            Predicted list of tuples consisting of `id`, `prediction` and `target`.
            
            id :
                user id
            prediction :
                predicted value of the learned model
            target :
                target value of training dataset
        features (list) :
            A features list used for training the model.
        target (string) :
            Target column's name.
        regularization (float) : 
            Used regularization parameter of ridge term.
        coefficient (list) :
            A coefficients list associated with each features element.
        intercept (int) :
            The value of intercept term.
        score (int) :
            This score represents coefficient :math:`R^2= 1-\\frac{\\sum_{i}(y_i - f_i)^2}{\\sum_{i}(y_i-\\overline{y})^2}` ,
            where :math:`y_i, f_i, \\overline{y}` is training data, predicted data and mean of training data respectively.
        
    """
    owner = models.ForeignKey('auth.User', related_name='regression_ridge_models')
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now = True)
    title = models.CharField(max_length=100, blank=True, default='')
    code = models.TextField(blank=True)
    status = models.CharField(choices=STATUS_CHOICES, default=STATUS_CHOICES[1][1], max_length=100)
    target = models.TextField(max_length=100, blank=True)
    features =  models.TextField(max_length=100, blank=True)
    train_data = models.FileField(upload_to='train_data', blank=True)
    trained_data = JSONField()
    regularization = models.FloatField(blank=True , default=1.0)
    decomp_flg = models.NullBooleanField(default=False)
    decomp_components = models.IntegerField(default=2,blank=True)
    decomp_algorithm = models.CharField(choices=DECOMPOSITION_CHOICES,default=DECOMPOSITION_CHOICES[1][1], max_length=100)

def validate_ratio(self):
        if self<0 or self>1.0:
            raise ValidationError(u'regularization parameter is between 0 and 1:%s' % self)
    
class MLEngineElasticNet(models.Model):
    """**Elastic Net**

    Elastic-Net regression is a combination model of Lasso and Ridge regression.
    If you set decomp_flg to `True`, decomposed features are used for fitting.

    endpoint
        `/v1/regression/elasticnet/`
    
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
        target (string) : Target column's name. Regression uses continuous variable as the target.
        features (list) : A features list for fitting.
        regularization (float, optional, default=1.0) : Regularization parameter of Lasso and Ridge term.
        l1_ratio (float, optional, default=0.5) : ElasticNet mixing parameter(:math:`0.0 \leq l1\_ratio \leq 1.0`) of lasso penarlty against ridge penalty.
        decomp_flg (bool, default=False) : Decomposition flag. Default is false.
        decomp_components (int, optional, default=2) : The number of decomposition components.
        decomp_algorithm (ChoiceField, default='PCA') : Decomposition Algorithm. Only Principal Component Analysis(PCA) can be selected, for now.
        
    Returns:
        Json : list of tupples as follows,
        
        data (list of tuples) : 
            Predicted list of tuples consisting of `id`, `prediction` and `target`.
            
            id :
                user id
            prediction :
                predicted value of the learned model
            target :
                target value of training dataset
        features (list) :
            A features list used for training the model.
        target (string) :
            Target column's name.
        regularization (float) : 
            Used regularization parameter of Lasso and Ridge term.
        coefficient (list) :
            A coefficients list associated with each features element.
        intercept (int) :
            The value of intercept term.
        score (int) :
            This score represents coefficient :math:`R^2= 1-\\frac{\\sum_{i}(y_i - f_i)^2}{\\sum_{i}(y_i-\\overline{y})^2}` ,
            where :math:`y_i, f_i, \\overline{y}` is training data, predicted data and mean of training data respectively.
        
    """
    owner = models.ForeignKey('auth.User', related_name='regression_elasticnet_models')
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now = True)
    title = models.CharField(max_length=100, blank=True, default='')
    code = models.TextField(blank=True)
    status = models.CharField(choices=STATUS_CHOICES, default=STATUS_CHOICES[1][1], max_length=100)
    target = models.TextField(max_length=100, blank=True)
    features =  models.TextField(max_length=100, blank=True)
    train_data = models.FileField(upload_to='train_data', blank=True)
    trained_data = JSONField()
    regularization = models.FloatField(blank=True , default=1.0)
    l1_ratio =  models.FloatField(blank=True , default=0.5, validators=[validate_ratio])
    decomp_flg = models.NullBooleanField(default=False)
    decomp_components = models.IntegerField(default=2,blank=True)
    decomp_algorithm = models.CharField(choices=DECOMPOSITION_CHOICES,default=DECOMPOSITION_CHOICES[1][1], max_length=100)

class MLEngineBayesianRidge(models.Model):
    """**Bayesian Regression**

    Bayesian Regression based on prior probability with gamma function.
    If you set decomp_flg to `True`, decomposed features are used for fitting.

    endpoint
        `/v1/regression/bayesian/`
    
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
        target (string) : Target column's name. Regression uses continuous variable as the target.
        features (list) : A features list for fitting.
        regularization_rate (float, optional, default=1e-06) : Gamma rate parameter (inverse of scale parameter) of regularization term.
        regularization_shape (float, optional, default=1e-06) : Gamma shape factor of regularization term.
        noise_rate (float, optional, default=1e-06) : Gamma rate parameter (inverse of scale parameter) of noise term.
        noise_shape (float, optional, default=1e-06) : Gamma shape factor of noise term.
        decomp_flg (bool, default=False) : Decomposition flag. Default is false.
        decomp_components (int, optional, default=2) : The number of decomposition components.
        decomp_algorithm (ChoiceField, default='PCA') : Decomposition Algorithm. Only Principal Component Analysis(PCA) can be selected, for now.
        
    Returns:
        Json : list of tupples as follows,
        
        data (list of tuples) : 
            Predicted list of tuples consisting of `id`, `prediction` and `target`.
            
            id :
                user id
            prediction :
                predicted value of the learned model
            target :
                target value of training dataset
        features (list) :
            A features list used for training the model.
        target (string) :
            Target column's name.
        precision_regularization (float) : 
            Inverse of variance for regularization term.
        precision_noise (float) :
            Inverse of variance for noise term.
        coefficient (list) :
            A coefficients list associated with each features element.
        intercept (int) :
            The value of intercept term.
        score (int) :
            This score represents coefficient :math:`R^2= 1-\\frac{\\sum_{i}(y_i - f_i)^2}{\\sum_{i}(y_i-\\overline{y})^2}` ,
            where :math:`y_i, f_i, \\overline{y}` is training data, predicted data and mean of training data respectively.
        
    """
    owner = models.ForeignKey('auth.User', related_name='regression_bayesian_models')
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now = True)
    title = models.CharField(max_length=100, blank=True, default='')
    code = models.TextField(blank=True)
    status = models.CharField(choices=STATUS_CHOICES, default=STATUS_CHOICES[1][1], max_length=100)
    target = models.TextField(max_length=100, blank=True)
    features =  models.TextField(max_length=100, blank=True)
    train_data = models.FileField(upload_to='train_data', blank=True)
    trained_data = JSONField()
    regularization_shape = models.FloatField(blank=True , default=1.0e-6)
    regularization_rate = models.FloatField(blank=True , default=1.0e-6)
    noise_shape = models.FloatField(blank=True , default=1.0e-6)
    noise_rate = models.FloatField(blank=True , default=1.0e-6)
    decomp_flg = models.NullBooleanField(default=False)
    decomp_components = models.IntegerField(default=2,blank=True)
    decomp_algorithm = models.CharField(choices=DECOMPOSITION_CHOICES,default=DECOMPOSITION_CHOICES[1][1], max_length=100)

class MLEngineAdaBoostRegressor(models.Model):
    """**AdaBoost Regression**

    AdaBoost is one of the unsemble regressor.

    endpoint
        `/v1/regression/adaboost/`
    
    Parameters:
        id (int, read only) : Unique and sequential number which is given automatically.
        title (char[100], optional, default='') : A title of the model.
        code (text, optional, default='') : An explanation of the model.
        owner (ForeignKey, read only) : Owner's username of the model.
        created (DateTime, read only) : Universal datetime when the model was created.
        updated (DateTime, read only) : Universal datetime when the model was updated.
        status (char, read only) : Status represented from among the state list:[``Created``, ``Submitted Learning Request``, ``Learning-in``, ``Learned``, ``Error``].
            Each status restricts some of request as follows,
            
            * ``Created`` means a model is created. `GET`, `PUT` and `DELETE` method are accepted. `GET` method with query parameter mode=evaluate , however, is NOT accepted.
            * ``Submitted Learning Request`` means that `PUT` method with query mode=learn has been accepted. Only `GET` method with no query parameter is acceptable in this status.
            * ``Learning-in`` means fitting process for the model has been called. Only `GET` method with no query parameter is acceptable in this status.
            * ``Learned`` means fitting for the model has been done.
            * ``Error`` means fitting process has stopped by internal error.
            
        train_data (FileField) : Dataset used for fitting or evaluation.
        trained_data (list of FileField) : List of dataset already used for fitting process.
        target (string) : Target column's name. Regression uses continuous variable as the target.
        features (list) : A features list for fitting.
        regularization (float, optional, default=1.0) : Regularization parameter.
        estimator (ChoiceField, optional, default=`DecisionTree` ) : Choose at least one algorithm from ``DecisionTree``, ``Linear``, ``Lasso``, ``Ridge`` and ``Bayesian``.
       
    Returns:
        Json : list of tupples as follows,
        
        data (list of tuples) : 
            Predicted list of tuples consisting of `id`, `prediction` and `target`.
            
            id :
                user id
            prediction :
                predicted value of the learned model
            target :
                target value of training dataset
        features (list) :
            A features list used for training the model.
        target (string) :
            Target column's name.
        regularization (float) : 
            Used regularization parameter of Lasso and Ridge term.
        estimator (string) :
            Choosed estimator algorithm.
        score (int) :
            This score represents coefficient :math:`R^2= 1-\\frac{\\sum_{i}(y_i - f_i)^2}{\\sum_{i}(y_i-\\overline{y})^2}` ,
            where :math:`y_i, f_i, \\overline{y}` is training data, predicted data and mean of training data respectively.
        
    """
    owner = models.ForeignKey('auth.User', related_name='regression_adaboost_models')
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now = True)
    title = models.CharField(max_length=100, blank=True, default='')
    code = models.TextField(blank=True)
    status = models.CharField(choices=STATUS_CHOICES, default=STATUS_CHOICES[1][1], max_length=100)
    target = models.TextField(max_length=100, blank=True)
    features =  models.TextField(max_length=100, blank=True)
    train_data = models.FileField(upload_to='train_data', blank=True)
    trained_data = JSONField()
    regularization = models.FloatField(blank=True , default=1.0)
    estimator = models.CharField(choices=ESTIMATOR_CHOICES, default=ESTIMATOR_CHOICES[2][1], max_length=100)
    
class MLEngineBaggingRegressor(models.Model):
    """**Bagging Regression**

    Bagging is one of the unsemble regressor.

    endpoint
        `/v1/regression/bagging/`
    
    Parameters:
        id (int, read only) : Unique and sequential number which is given automatically.
        title (char[100], optional, default='') : A title of the model.
        code (text, optional, default='') : An explanation of the model.
        owner (ForeignKey, read only) : Owner's username of the model.
        created (DateTime, read only) : Universal datetime when the model was created.
        updated (DateTime, read only) : Universal datetime when the model was updated.
        status (char, read only) : Status represented from among the state list:[``Created``, ``Submitted Learning Request``, ``Learning-in``, ``Learned``, ``Error``].
            Each status restricts some of request as follows,
            
            * ``Created`` means a model is created. `GET`, `PUT` and `DELETE` method are accepted. `GET` method with query parameter mode=evaluate , however, is NOT accepted.
            * ``Submitted Learning Request`` means that `PUT` method with query mode=learn has been accepted. Only `GET` method with no query parameter is acceptable in this status.
            * ``Learning-in`` means fitting process for the model has been called. Only `GET` method with no query parameter is acceptable in this status.
            * ``Learned`` means fitting for the model has been done.
            * ``Error`` means fitting process has stopped by internal error.
            
        train_data (FileField) : Dataset used for fitting or evaluation.
        trained_data (list of FileField) : List of dataset already used for fitting process.
        target (string) : Target column's name. Regression uses continuous variable as the target.
        features (list) : A features list for fitting.
        regularization (float, optional, default=1.0) : Regularization parameter.
        estimator (ChoiceField, optional, default=`DecisionTree` ) : Choose at least one algorithm from ``DecisionTree``, ``Linear``, ``Lasso``, ``Ridge`` and ``Bayesian``.
       
    Returns:
        Json : list of tupples as follows,
        
        data (list of tuples) : 
            Predicted list of tuples consisting of `id`, `prediction` and `target`.
            
            id :
                user id
            prediction :
                predicted value of the learned model
            target :
                target value of training dataset
        features (list) :
            A features list used for training the model.
        target (string) :
            Target column's name.
        regularization (float) : 
            Used regularization parameter of Lasso and Ridge term.
        estimator (string) :
            Choosed estimator algorithm.
        score (int) :
            This score represents coefficient :math:`R^2= 1-\\frac{\\sum_{i}(y_i - f_i)^2}{\\sum_{i}(y_i-\\overline{y})^2}` ,
            where :math:`y_i, f_i, \\overline{y}` is training data, predicted data and mean of training data respectively.
        
    """
    owner = models.ForeignKey('auth.User', related_name='regression_bagging_models')
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now = True)
    title = models.CharField(max_length=100, blank=True, default='')
    code = models.TextField(blank=True)
    status = models.CharField(choices=STATUS_CHOICES, default=STATUS_CHOICES[1][1], max_length=100)
    target = models.TextField(max_length=100, blank=True)
    features =  models.TextField(max_length=100, blank=True)
    train_data = models.FileField(upload_to='train_data', blank=True)
    trained_data = JSONField()
    regularization = models.FloatField(blank=True , default=1.0)
    estimator = models.CharField(choices=ESTIMATOR_CHOICES, default=ESTIMATOR_CHOICES[2][1], max_length=100)

class MLEngineSVR(models.Model):
    """**SVR**

    Support Vector Regression.
    If you set decomp_flg to `True`, decomposed features are used for fitting.

    endpoint
        `/v1/regression/svr/`
    
    Parameters:
        id (int, read only) : Unique and sequential number which is given automatically.
        title (char[100], optional, default='') : A title of the model.
        code (text, optional, default='') : An explanation of the model.
        owner (ForeignKey, read only) : Owner's username of the model.
        created (DateTime, read only) : Universal datetime when the model was created.
        updated (DateTime, read only) : Universal datetime when the model was updated.
        status (char, read only) : Status represented from among the state list:[``Created``, ``Submitted Learning Request``, ``Learning-in``, ``Learned``, ``Error``].
            Each status restricts some of request as follows,
            
            * ``Created`` means a model is created. `GET`, `PUT` and `DELETE` method are accepted. `GET` method with query parameter mode=evaluate , however, is NOT accepted.
            * ``Submitted Learning Request`` means that `PUT` method with query mode=learn has been accepted. Only `GET` method with no query parameter is acceptable in this status.
            * ``Learning-in`` means fitting process for the model has been called. Only `GET` method with no query parameter is acceptable in this status.
            * ``Learned`` means fitting for the model has been done.
            * ``Error`` means fitting process has stopped by internal error.
            
        train_data (FileField) : Dataset used for fitting or evaluation.
        trained_data (list of FileField) : List of dataset already used for fitting process.
        target (string) : Target column's name. Regression uses continuous variable as the target.
        features (list) : A features list for fitting.
        regularization (float, optional, default=1.0) : Regularization parameter.
        decomp_flg (bool, default=False) : Decomposition flag. Default is false.
        decomp_components (int, optional, default=2) : The number of decomposition components.
        decomp_algorithm (ChoiceField, default='PCA') : Decomposition Algorithm. Only Principal Component Analysis(PCA) can be selected, for now.
       
    Returns:
        Json : list of tupples as follows,
        
        data (list of tuples) : 
            Predicted list of tuples consisting of `id`, `prediction` and `target`.
            
            id :
                user id
            prediction :
                predicted value of the learned model
            target :
                target value of training dataset
        features (list) :
            A features list used for training the model.
        target (string) :
            Target column's name.
        regularization (float) : 
            Used regularization parameter of Lasso and Ridge term.
        coefficient (list) :
            A coefficients list associated with each features element.
        intercept (int) :
            The value of intercept term.
        score (int) :
            This score represents coefficient :math:`R^2= 1-\\frac{\\sum_{i}(y_i - f_i)^2}{\\sum_{i}(y_i-\\overline{y})^2}` ,
            where :math:`y_i, f_i, \\overline{y}` is training data, predicted data and mean of training data respectively.
        
    """
    owner = models.ForeignKey('auth.User', related_name='regression_svr_models')
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now = True)
    title = models.CharField(max_length=100, blank=True, default='')
    code = models.TextField(blank=True)
    status = models.CharField(choices=STATUS_CHOICES, default=STATUS_CHOICES[1][1], max_length=100)
    target = models.TextField(max_length=100, blank=True)
    features =  models.TextField(max_length=100, blank=True)
    train_data = models.FileField(upload_to='train_data', blank=True)
    trained_data = JSONField()
    regularization = models.FloatField(blank=True , default=1.0)
    decomp_flg = models.NullBooleanField(default=False)
    decomp_components = models.IntegerField(default=2,blank=True)
    decomp_algorithm = models.CharField(choices=DECOMPOSITION_CHOICES,default=DECOMPOSITION_CHOICES[1][1], max_length=100)
    
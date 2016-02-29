
"""MLEngine Classes for Classification

barn-owl provides algorithms for the purpose of classification with endpoints `/v1/classification/<algorithm_name>`.

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
ESTIMATOR = ['', 'SGD', 'DecisionTree', 'Ridge', 'SVC']
ESTIMATOR_CHOICES = [(item,item) for item in ESTIMATOR]
DECOMPOSITION = ['','PCA']
DECOMPOSITION_CHOICES = [(item,item) for item in DECOMPOSITION]

class MLEngineC(models.Model):
    """**Classification with DNN**
    
    Classification model based on deep neural network.
    
    endpoint
        `/v1/classification/dnn/`
            
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
        target (string) : Target column's name. Classification uses categorical variable as the target.
        
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
    owner=models.ForeignKey('auth.User',related_name='classification_dnn_models')
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

def validate_positive(self):
        if self<0:
            raise ValidationError(u'regularization parameter should be positive number:%s' % self)

class MLEngineLogistic(models.Model):
    """**Classification with Logistic Regression**

    Logistic regression is a probability regression model with categorical target variable.
    If you set decomp_flg to `True`, decomposed features are used for fitting.
     
    endpoint
        `/v1/classification/logistic/`
    
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
        target (string) : Target column's name. Classification uses categorical variable as the target.
        features (list) : A features list for fitting.
        regularization_inverse (float, optional, default=1.0) : Inverse of regularization parameter.
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
    owner = models.ForeignKey('auth.User', related_name='classification_logistic_models')
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now = True)
    title = models.CharField(max_length=100, blank=True, default='')
    code = models.TextField(blank=True)
    status = models.CharField(choices=STATUS_CHOICES, default=STATUS_CHOICES[1][1], max_length=100)
    target = models.TextField(max_length=100, blank=True)
    features =  models.TextField(max_length=100, blank=True)
    train_data = models.FileField(upload_to='train_data', blank=True)
    trained_data = JSONField()
    regularization_inverse = models.FloatField(blank=True ,validators=[validate_positive], default=1.0)
    decomp_flg = models.NullBooleanField(default=False)
    decomp_components = models.IntegerField(default=2,blank=True)
    decomp_algorithm = models.CharField(choices=DECOMPOSITION_CHOICES,default=DECOMPOSITION_CHOICES[1][1], max_length=100)

class MLEngineRidgeC(models.Model):
    """**Ridge Classification**

    Ridge classification is a linear sparse model for classification analysis.
    If you set decomp_flg to `True`, decomposed features are used for fitting.
     
    endpoint
        `/v1/classification/ridge/`
    
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
        target (string) : Target column's name. Classification uses categorical variable as the target.
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
    owner = models.ForeignKey('auth.User', related_name='classification_ridge_models')
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now = True)
    title = models.CharField(max_length=100, blank=True, default='')
    code = models.TextField(blank=True)
    status = models.CharField(choices=STATUS_CHOICES, default=STATUS_CHOICES[1][1], max_length=100)
    target = models.TextField(max_length=100, blank=True)
    features =  models.TextField(max_length=100, blank=True)
    train_data = models.FileField(upload_to='train_data', blank=True)
    trained_data = JSONField()
    regularization = models.FloatField(blank=True ,validators=[validate_positive], default=1.0)
    decomp_flg = models.NullBooleanField(default=False)
    decomp_components = models.IntegerField(default=2,blank=True)
    decomp_algorithm = models.CharField(choices=DECOMPOSITION_CHOICES,default=DECOMPOSITION_CHOICES[1][1], max_length=100)

class MLEngineAdaBoostClassifier(models.Model):
    """**AdaBoost Classification**

    AdaBoost is one of the unsemble classifier.

    endpoint
        `/v1/classification/adaboost/`
    
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
        target (string) : Target column's name. Classification uses categorical variable as the target.
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
    owner = models.ForeignKey('auth.User', related_name='classification_adaboost_models')
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
    
class MLEngineBaggingClassifier(models.Model):
    """**Bagging Classification**

    Bagging is one of the unsemble classifier.

    endpoint
        `/v1/classification/bagging/`
    
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
        target (string) : Target column's name. Classification uses categorical variable as the target.
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
    owner = models.ForeignKey('auth.User', related_name='classification_bagging_models')
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

class MLEngineSVC(models.Model):
    """**SVC**

    Support Vector Classification.

    endpoint
        `/v1/classification/svc/`
    
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
        target (string) : Target column's name. Classification uses categorical variable as the target.
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
    owner = models.ForeignKey('auth.User', related_name='classification_svc_models')
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now = True)
    title = models.CharField(max_length=100, blank=True, default='')
    code = models.TextField(blank=True)
    status = models.CharField(choices=STATUS_CHOICES, default=STATUS_CHOICES[1][1], max_length=100)
    target = models.TextField(max_length=100, blank=True)
    features =  models.TextField(max_length=100, blank=True)
    train_data = models.FileField(upload_to='train_data', blank=True)
    trained_data = JSONField()
    regularization = models.FloatField(blank=True ,validators=[validate_positive], default=1.0)
    decomp_flg = models.NullBooleanField(default=False)
    decomp_components = models.IntegerField(default=2,blank=True)
    decomp_algorithm = models.CharField(choices=DECOMPOSITION_CHOICES,default=DECOMPOSITION_CHOICES[1][1], max_length=100)

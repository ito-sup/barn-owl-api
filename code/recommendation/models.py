
"""MLEngine Classes for Recommendation

barn-owl provides algorithms for the purpose of action recommendation with endpoints `/v1/recommendation/<algorithm_name>`.

"""
__author__ = "G.Ito <ito@supreme-system.com>"
__version__ = "1.00"
__date__ = "17 Nov. 2015"
__status__ = "production"

from django.db import models
from jsonfield import JSONField

'''model's status'''
STATUS = ['', 'Created', 'Submitted Learning Request', 'Learning-in', 'Learned', 'Error']
STATUS_CHOICES=[(item,item) for item in STATUS]

class MLEngineR(models.Model):
    """**Recommendation with DQN**
    
    Recommendation model based on deep Q-network.
    
    endpoint
        `/v1/recommendation/dqn/`
            
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
            
            id (int) :
                user id
            best_action (int) :
                predicted best action id 
            Qvalue (float) :
                Qvalue for each action
        action_list (list) :
            List of enable action.
        target (string) :
            Target column's name.
        
    """
    owner=models.ForeignKey('auth.User',related_name='recommendation_dqn_models')
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
        
    
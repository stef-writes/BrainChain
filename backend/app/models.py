from django.db import models

class Node(models.Model):
    node_id = models.CharField(max_length=100)
    node_type = models.CharField(max_length=100)
    input_keys = models.JSONField(default=list)
    output_keys = models.JSONField(default=list)
    model_config = models.JSONField(default=dict)
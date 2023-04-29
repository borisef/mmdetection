## Heads Custom Weights

### Motivation
In mmdetection  "_Head_" is a network component which is connected to loss. Some Object Detection models have two or more _heads_. For example FasterRCNN (two-stage detector) 
has _RPN-Head_ and _ROI-Head_. 

If your training set contains different datasets, you may want to weight 
the loss of each dataset for each head differently. For example, you may want the impact of synthetic dataset on classifier loss 
to be larger than on RPN loss.

To address such request we implemented several custom heads: 
* RPNHeadWithWeightPerImage -- which extends RPNHead
* ConvFCBBoxHeadWithWeightPerImage  -- which extends ConvFCBBoxHead (type of ROI/RCNN head)
* Shared2FCBBoxHeadWithWeightPerImage --  which extends Shared2FCBBoxHead (type of ROI/RCNN head)
* BBoxHeadWithWeightPerImage -- which extends BBoxHead (type of ROI/RCNN head)

### Usage 

Need to make two changes in config:

1) Replace appropriate head (i.e. RPNHead) with weight per image in model. Example:
````
 rpn_head=dict(
        #type='RPNHead',
        type='RPNHeadWithWeightPerImage', #LOSS WEIGHT PER IMAGE
````
2) Add to data pipeline additional step of type 'AddFieldToImgMetas'.\
This is just to add additional data field to metadata of each image. Example:
````
train_pipeline1 = ...
train_pipeline2 = ....

train_pipeline1 = train_pipeline1 + [
    dict(type='AddFieldToImgMetas', fields = ['rpn_head.loss_weight'], values = [10.0], replace = True),#<RFL> LOSS WEIGHT PER IMAGE
]
...
#you suppose to heave something like that later in config:
train_data1 = dict(
        ...
        pipeline=train_pipeline1,
        ...
        )

train_data2 = dict(
        ...
        pipeline=train_pipeline2,
        ...
        )
        

train_data = dict(
    type = 'ConcatDataset',
    datasets = [train_data1, train_data2],
)

````
Same idea if you want to use `'roi_head.loss_weight'` instead of `'rpn_head.loss_weight'`. 
You can use one or both of them. 

### Remarks

* You can also define weight per image rather than per dataset. 
But this is slightly more complicated and less practical. Talk to @borisef, if you interested
* Be careful with custom weights. For example, zero weights may be risky. 
* Take into account the relative size of the dataset to the whole data. 

### Relevant code changes

* custom_rpn_head.py -- implementation
* custom_bbox_head.py -- implementation
* test_custom_rpn_head.py -- unittest 
* test_custom_bbox_head.py -- unittest
* custom_formatting.py -- implementation of `AddFieldToImgMetas` 
* standard_roi_head_with_extra.py, bbox_head.py, standard_roi_head.py, anchor_head.py -- small changes of legacy code,\
  (see comments in code)

## Samplers with Ignore Negatives

### Motivation 

Unfortunately, sometimes we have non-perfectly labeled data. 
Especially if we want to re-use data from different project.
One very common issue: **not all targets are labeled.**

Such dataset is problematic for us. 
Why? Because training procedure will see unlabeled targets as "negative" examples. 

The component for sampling "negative" and "positive" samples in `mmdetection` is 'Sampler'. 
We propose modified version of it. 

## Usage 
Need to modify config file as following: 

1) Replace Sampler(s): 
```
 rpn=dict(
            ...
            sampler=dict(
                #type='RandomSampler'
                type='RandomSamplerWithIgnore', #<RFL> WITH IGNORE
                ...
                )
 rcnn=dict(
           ...
            sampler=dict(
                #type='RandomSamplerWithIgnore',
                type='OHEMSamplerWithIgnore', #<RFL> WITH IGNORE
                ...
                )
```
2) Add to data pipeline additional step of type 'AddFieldToImgMetas'.\
This is just to add additional data field to metadata of each image. Example:
````
train_pipeline1 = ...
train_pipeline2 = ....

train_pipeline1 = train_pipeline1 + [
    dict(type='AddFieldToImgMetas', fields = ['ignore_negatives'], values = [1], replace = False), #IGNORE NEGATIVES <#RFL> 
]
#you suppose to heave something like that later in config:
train_data1 = dict(
        ...
        pipeline=train_pipeline1,
        ...
        )

train_data2 = dict(
        ...
        pipeline=train_pipeline2,
        ...
        )
        

train_data = dict(
    type = 'ConcatDataset',
    datasets = [train_data1, train_data2],
)

````
## Remarks 

* So far we implemented only few samplers: OHEMSampler, RandomSampler
* Be careful with using this feature, 
it will work only if the dataset is small relative to the full data

### Relevant code changes

* custom_sampler.py -- implementation
* test_sampler_boris.py -- unittest
* custom_formatting.py -- implementation of `AddFieldToImgMetas`
* standard_roi_head.py, anchor_head.py -- small changes

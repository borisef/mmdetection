## Custom Head Weights

### Motivation
In mmdetection  "Head" is a network component which outputs loss. Some Object Detection models have two or more heads. For example FasterRCNN (two-stage detector) 
has RPN-head and ROI-head. 

If your training set contains different datasets, you may want to weight 
the loss of each dataset for each head differently. For example, you may want the impact of synthetic dataset on classifier loss 
to be larger than on RPN loss.

To address such request we implemented several custom heads: 
* RPNHeadWithWeightPerImage -- which extends RPNHead
* ConvFCBBoxHeadWithWeightPerImage  -- which extends ConvFCBBoxHead (type of ROI/RCNN head)
* Shared2FCBBoxHeadWithWeightPerImage --  which extends Shared2FCBBoxHead (type of ROI/RCNN head)
* BBoxHeadWithWeightPerImage -- which extends BBoxHead (type of ROI/RCNN head)

### Usage 

Need to make several changes in config:

1) Replace appropriate head (i.e. RPNHead) with weight per image in model. Example:
````
 rpn_head=dict(
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
Same idea if you want to use 'roi_head.loss_weight' instead of 'rpn_head.loss_weight'. 
You can use one or both of them. 

### Remarks

* You can also define weight per image rather than per dataset. 
But this is slightly more complicated and less useful 
* If you want to implement any other head 
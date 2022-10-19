## Intro 

Congratulations! You are about to develop _mmdetection_ code. We want this code to be useful and maintainable. 
We also want to be able to keep this code up-to-date with the web version. To achieve this, we all will need to stick to some basic guidelines. 


## Git Guidelines

#### Remote repositories
Our repository for _mmdetection_ code is `XXX`. First of all you suppose to `fork` it. 
Let's suppose your fork is `XXX`

Let's list all the remote repositories your _git_ "knows":

``````
git remote -v
``````

You suppose to see something like: 

``````
origin XXX
39484 XXX

``````
If you don't see, use `git remote add`  and `git remote remove` commands to map nicknames of remote repositories
So, from now on you can do 
````
pull/fetch/push origin branch_name ` 
````
and sometimes 
````
pull/fetch 39484 branch_name
````

#### Branches and Tags

The most updated code will usually be under `master` branch in `39484`. 
We use tags to keep track of versions.Official releases will always be tagged. 
We recommend to follow XXX in order to be up-to-date with the latest release. 
Obviously, each developer is supposed to develop in his/her own branch or project branch (not `master`).
Each developer may create his own tags, but not in the format of official releases to avoid confusion.

#### Merge request

In order to merge your code into 39484, you will have to create `merge request`. 
Never do `push` into `39484`, unless you are administrator.    

## Code Guidelines

#### Types of code
We have three types of code in this project: 
1) _Legacy code_: original `mmdetection` code from the web 
2) _Group code_: code developed by us and used by 2 or more projects 
3) _Private/experimental code_: code developed by us and used by 1 project or under experiment

Similarly, we have 3 types of files. Sometimes _legacy files_ may contain _group code_ . 
Here are guidelines for code developers: 

* Avoid changing _legacy code_, unless you have to !!!
* If you change _legacy code_ always add `#<RFL>` comment to each block of changes
* If you add _private/experimental code_ into _legacy code_ let it stay in your branch forever
* If you add _group code_ into _legacy code_ do it carefully, think about workarounds to avoid it
* All your _private files_ have to be under `yourname` folder or `your_project` folder, for example `myname\smart_loss.py`
* All our _group files_ have to be under `xxx` folder for example: XXX

#### Unittests and documentation

It is recommended to have unittest, comments and documentation for each piece of your code. But our guidelines are simple: 

* You have to deliver unittest with your __group code__
* If your __group code__ is a standalone module or capability 
(for example: adversarial training, domain adaptation, hierchical loss) you have to deliver documentation with tutorial and example. 
BTW, sometimes unittest can serve as example. 
